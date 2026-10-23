"""The mean field's own energy as a potential-energy surface.

`MeanFieldSurface` is the ground-state surface every walk and Hessian of the
property layer can take: no excited-state machinery, no frozen conventions.
`mean_field_force` is its force, the one definition the property layer's
convergence checks read too.
"""
import numpy as np

from src.Base.declaration import SurfacePhysics
from src.Base.factorization.distributed_isdf_jk import scf_pair_layout
from src.environments.environment import environment_label
from src.Base.environment_seam import resolve_environment
from src.Base.factorization.isdf_jk import frozen_pair_layout
from src.gradients.isdf_mean_field import mean_field_skeleton_force
from src.Base.utils.parallel.mpi_grid import lockstep, lockstep_mean_field
from src.SingleReference.LinearResponse.rpa_energy import declared_ground_state
from src.gradients.evaluation import GeometryCache
from src.gradients.reference_mean_field import converged_factory
from src.gradients.surface import PotentialEnergySurface


def mean_field_force(mf):
    """dE/dR of the energy this mean field reported, whatever built its exchange.

    The definition is `isdf_jk.mean_field_skeleton_force`, which keeps
    `grid_response`: pyscf defaults it to False, dropping d(becke weights)/dR,
    and that force breaks translational invariance by 8.1e-06 Ha/Bohr on
    water/cc-pVDZ/B3LYP against 4.2e-15 with it.

    Rank 0's force on every rank: pyscf's threaded GEMM adds its partial sums
    in thread-arrival order and its density-fitted gradient blocks by each
    process's free memory, so every rank's own call differs in its last bits.
    """
    return lockstep(mean_field_skeleton_force(mf))


class MeanFieldSurface(PotentialEnergySurface):
    """The mean field's own energy as a `PotentialEnergySurface`.

    The ground state needs no excited-state machinery and no frozen
    conventions, so `refreeze` is the identity. It lets `optimize` relax a
    ground state with neither geomeTRIC nor pyberny installed.

    The SCF is the only stage, and it is what the ranks divide: a factory that
    hands back a mean field built but not run leaves the convergence to
    `converged_factory`, where every rank runs pyscf's driver against the
    reduced J/K and quadrature, contributing its block of the auxiliary index
    and of the grid. A factory that converges its own is used as it is, its
    orbitals locked to rank 0's. Each rank forms the force from rank 0's
    orbitals and `mean_field_force` hands every rank rank 0's; the energy is
    the locked mean field's own.

    In a continuum (`environment`, a `SolventScreening`) every geometry's SCF
    is the environment's ground state, PCM at eps_static on a cavity rebuilt
    around that geometry (`for_geometry`), and the force is pyscf's PCM
    gradient of it, or the ISDF one completed with the reaction field's term.
    This is the surface for e.g. a UKS geometry of an ion in solvent, where no
    correlated surface reaches.
    """

    #: The ingredients this surface provides (`surface.INGREDIENTS`).
    INGREDIENTS = frozenset({'mean_field', 'physics'})

    def __init__(self, mol, scf_factory, mf=None, environment=None,
                 own=False):
        """`mf` is the reference mean field where the caller already has one.

        The declaration (which functional E_0 is, a property of the reference
        and not of a geometry) reads it, so a surface built without one
        converges its own the first time it is asked. With `own` it is also
        the mean field this surface evaluates at `mol` (the gas phase, where
        the factory's mean field is the reference), handed back by
        `mean_field(mol)` rather than converged again.
        """
        self.mol0 = mol
        self._scf = scf_factory
        self._mf0 = mf
        self.environment = resolve_environment(environment, None)
        self._own = bool(own) and mf is not None
        # The ISDF-K pair layout of the factory's mean field at mol0, which
        # the mean field at every geometry of this surface fits on: [] until
        # read, then [layout], None for a mean field without an ISDF-K fit.
        self._layout = []
        # one SCF per geometry: the mean fields of the last ones evaluated
        self._mean_fields = GeometryCache()

    def _mean_field(self, mol):
        """The factory's mean field at `mol` in this surface's environment,
        converged over the ranks, its ISDF-K fit on the pair layout of mol0's
        (`pair_layout`); a geometry asked again gets the one it got."""
        held = self._mean_fields.get(mol)
        if held is not None:
            return held
        at_reference = np.array_equal(mol.atom_coords(),
                                      self.mol0.atom_coords())
        layout = (None if at_reference and not self._layout
                  else self.pair_layout())
        with frozen_pair_layout(layout):
            mf = self._converged(mol)
        if not self._layout:
            self._layout.append(scf_pair_layout(mf))
        return self._mean_fields.put(mol, mf)

    def _converged(self, mol):
        """The factory's mean field at `mol` in this environment, converged
        over the ranks, rank 0's on every rank."""
        return lockstep_mean_field(
            self.environment.for_geometry(mol).mean_field(
                mol, converged_factory(self._scf)))

    def pair_layout(self):
        """The AO-pair columns the ISDF-K fit of the mean field at mol0
        kept, which every geometry of this surface fits on, so that its
        energy is one function of the nuclei and the force its derivative;
        a surface asked elsewhere first converges mol0's for it. None for a
        mean field without an ISDF-K fit. `refreeze` takes it again at the
        new reference."""
        if not self._layout:
            mf = (self._mf0 if self._own else
                  self._converged(self.mol0))
            self._layout.append(scf_pair_layout(mf))
        return self._layout[0]

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the mean field this surface evaluates on: the factory's
        in this surface's environment, `mf` as given, or the reference
        geometry's own reference where the surface owns one."""
        mol = self.mol0 if mol is None else mol
        if mf is None and self._own and mol is self.mol0:
            return mol, self._mf0
        return mol, (self._mean_field(mol) if mf is None else mf)

    def scf_factory(self, mol):
        return self._scf(mol)

    @property
    def physics_ground_state(self):
        """E_0 = E_KS[xc], the mean field's own energy; 'hf' for Hartree-Fock."""
        if self._mf0 is None:
            self._mf0 = self._mean_field(self.mol0)
        return declared_ground_state(self._mf0, 'dft')

    @property
    def physics(self):
        """What this surface computes: E_0 alone, with no state on it, in
        the environment its SCF is converged in."""
        return SurfacePhysics(self.physics_ground_state, None,
                              environment_label(self.environment))

    def total_energy(self, mol=None, mf=None):
        mol = mol if mol is not None else self.mol0
        return float((mf or self._mean_field(mol)).e_tot)

    def total_gradient(self, mol=None, mf=None):
        mol = mol if mol is not None else self.mol0
        mf = mf or self._mean_field(mol)
        return np.asarray(mean_field_force(mf)), float(mf.e_tot), {}

    def refreeze(self, mol):
        return MeanFieldSurface(mol, self._scf, environment=self.environment)

    def label(self):
        return 'mean-field ground state'
