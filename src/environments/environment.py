"""What the surroundings do to a molecule, in one contract every route consumes.

The two-electron interaction is dressed on
the auxiliary metric, v -> v + vtilde with vtilde = v chi v (Duchemin,
Jacquemin and Blase, J. Chem. Phys. 144, 164106 (2016), Eq. 16), which
`separable_ri.aux_metric_sqrt` applies to every separable factorization; and
the static one-body term of the quasiparticle equation gains the reaction
field: on a GW route the Eq. (18) self-element of Delta W the dressing
implies, on a route that forms no W the static COHSEX Sigma^solv of their
Eqs. (20)-(22), which `qp_solve.static_exchange_diagonal` adds.

`Environment` is structural: `SolventScreening` satisfies it without inheriting
from anything. An environment that does not screen returns None from
`aux_kernel`; `static_self_energy` is None wherever no COHSEX operator is
offered, and the two are independent of each other -- `dresses_interaction`
below is the single place the first question is asked. Routes find the
environment of a calculation on the mean field (`attach_environment`,
`environment_of`); that seam, and `Environment` itself, are
`src.Base.environment_seam`.
"""
import numpy as np
from pyscf import qmmm
from pyscf.qmmm import itrf as qmmm_itrf

from src.Base.environment_seam import NoEnvironment
from src.Base.numerics.constants import BOHR_TO_ANGSTROM
from src.environments.split_equilibrium import (SOLVENT_SCHEMES,
                                         ClassicalStateSpecific,
                                         SplitEquilibrium)
from src.Base.utils.parallel.threads import blas_single_threaded


class PointCharges:
    """Fixed classical charges: an external potential in the mean field.

    A fixed charge does not respond, so it screens nothing -- the auxiliary
    metric stays bare and the static one-body term is empty. Everything the
    charges do enters through the mean field: pyscf's `qmmm.mm_charge` puts
    sum_k q_k / |r - R_k| into h_core and the charge-nucleus repulsion into
    `energy_nuc`. The charges do not move with the QM atoms, so `for_geometry`
    is the identity.
    """

    screens = False
    differentiable = True

    def __init__(self, coords, charges, unit='Angstrom'):
        self.coords = np.asarray(coords, float).reshape(-1, 3)
        self.charges = np.asarray(charges, float).reshape(-1)
        if len(self.coords) != len(self.charges):
            raise ValueError(f'{len(self.coords)} charge positions for '
                             f'{len(self.charges)} charges')
        if unit.lower() not in ('angstrom', 'ang', 'a', 'bohr', 'b', 'au'):
            raise ValueError(f"unit {unit!r}: 'Angstrom' or 'Bohr'")
        self.unit = unit

    @property
    def coords_bohr(self):
        if self.unit.lower() in ('bohr', 'b', 'au'):
            return self.coords
        return self.coords / BOHR_TO_ANGSTROM

    def state_specific_term(self):
        """None: this medium carries no classical state-specific term; what a
        state's density does to it is inside its excitation energy."""
        return None

    def for_geometry(self, mol):
        return self

    def mean_field(self, mol, scf_factory):
        """
        The factory's mean field with the charges in its Hamiltonian.

        A factory that already applied `qmmm.mm_charge` with THESE charges is
        returned as it is. Otherwise the converged gas-phase mean field is
        wrapped and re-converged from its own density: one extra SCF from a
        good guess, avoided by putting the charges into the factory.
        """
        mf = scf_factory(mol)
        if isinstance(mf, qmmm_itrf.QMMM):
            self._check(mf)
            return mf
        wrapped = qmmm.mm_charge(mf, self.coords, self.charges, unit=self.unit)
        with blas_single_threaded():
            wrapped.kernel(dm0=mf.make_rdm1())
        if not wrapped.converged:
            raise RuntimeError('the SCF did not re-converge with the point '
                               'charges attached')
        return wrapped

    def _check(self, mf):
        mm = mf.mm_mol
        if (mm.atom_coords().shape != self.coords.shape
                or np.abs(mm.atom_coords() - self.coords_bohr).max() > 1e-8
                or np.abs(mm.atom_charges() - self.charges).max() > 1e-12):
            raise ValueError('the mean field carries a different set of point '
                             'charges than this environment')

    def aux_kernel(self, auxmol):
        return None

    def whitened_transform(self, mol, mf):
        return None

    def kernel_mo(self, mol, mo_bra, mo_ket=None):
        return None

    def kernel_ao(self, mol):
        return None

    def static_self_energy(self, mf, mol=None):
        return None

    def dynamic_factor(self, omega):
        # a fixed charge does not respond at any frequency
        return np.ones_like(np.asarray(omega, float))

    def aux_kernel_adjoint(self, auxmol, v_bar):
        return np.zeros((auxmol.natm, 3))

    def static_self_energy_adjoint(self, mf, weights):
        return 0.0, np.zeros((mf.mol.natm, 3))

    def __repr__(self):
        return (f'PointCharges({len(self.charges)} charges, total '
                f'{self.charges.sum():+.3f})')


def dresses_interaction(environment, auxmol) -> bool:
    """
    Does `environment` change the interaction the post-SCF methods see?
    """
    return (environment is not None and getattr(environment, 'screens', True)
            and environment.aux_kernel(auxmol) is not None)


def environment_label(environment):
    """
    The environment's printable name for a `SurfacePhysics`; 'gas' is none.

    The declaration carries a STRING, not the object: two surfaces standing in
    the same continuum are comparable whether or not they were handed the same
    instance of it, and a declaration must be readable off a record that no
    longer holds the cavity.
    """
    if environment is None or isinstance(environment, NoEnvironment):
        return 'gas'
    return repr(environment)


def equilibrium_environment(environment, mode='split'):
    """`environment` as a relaxed excited state stands in it.

    mode 'split' (default): the state-specific response -- Eq. (18) and the
    direct W -- at eps_static, the transition-density exchange at eps_inf
    (`split_equilibrium`, which argues the partition). 'full': the whole
    response at eps_static, the equilibrium LR-PCM convention
    (`SolventScreening(equilibrium=True)`). The gas phase and fixed charges
    have no slow polarization to relax and come back as they are. A composite
    holding a continuum is refused: which of its members relaxes is a physics
    choice it does not declare.
    """
    if mode not in ('split', 'full'):
        raise ValueError(f"mode={mode!r}: 'split' or 'full'")
    if environment is None:
        return None
    if hasattr(environment, 'equilibrium_partner'):
        kind = getattr(environment, 'equilibrium', False)
        if isinstance(environment, ClassicalStateSpecific):
            raise ValueError('scheme B already fixes its partition; build it '
                             'from the non-equilibrium continuum instead')
        if kind == 'split':
            return (environment if mode == 'split'
                    else environment.static_response())
        if kind:
            if mode == 'full':
                return environment
            raise ValueError('a split equilibrium wraps the NON-equilibrium '
                             'continuum; pass that one')
        return (SplitEquilibrium(environment) if mode == 'split'
                else environment.equilibrium_partner())
    if any(hasattr(m, 'equilibrium_partner')
           for m in getattr(environment, 'members', ())):
        raise ValueError(
            f'{type(environment).__name__} holds a continuum; equilibrium '
            f'solvation is defined for a continuum on its own')
    return environment


def solvent_scheme(environment, scheme='duchemin', ss_density='relaxed',
                   ss_equilibrium=False):
    """`environment` as a solvated excitation reads it, by scheme.

    'duchemin': the solvated BSE of Duchemin et al. as it is -- the split
    equilibrium (`equilibrium_environment`). 'classical-ss': scheme B
    (`split_equilibrium.ClassicalStateSpecific`), the state-specific part on
    the bare interaction and the classical E_SS of the `ss_density`
    ('relaxed' or 'unrelaxed') density difference added, vertical or at
    `ss_equilibrium`. The gas phase comes back as it is; scheme B needs a
    continuum with eps_static.
    """
    if scheme not in SOLVENT_SCHEMES:
        raise ValueError(f'scheme={scheme!r}: one of {SOLVENT_SCHEMES}')
    if scheme == 'duchemin':
        return equilibrium_environment(environment, 'split')
    if environment is None or isinstance(environment, NoEnvironment):
        return environment
    if not hasattr(environment, 'equilibrium_partner'):
        raise ValueError(f'scheme B needs a polarizable continuum; '
                         f'{type(environment).__name__} has no surface')
    if getattr(environment, 'equilibrium', False):
        raise ValueError('scheme B wraps the NON-equilibrium continuum; pass '
                         'that one')
    return ClassicalStateSpecific(environment, ss_density=ss_density,
                                  ss_equilibrium=ss_equilibrium)


