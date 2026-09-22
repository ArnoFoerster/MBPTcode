"""What a property routine is allowed to know about the electronic structure
underneath it.

A geometry optimization, a Huang-Rhys spectrum, an adiabatic singlet-triplet
gap or a Marcus rate needs the same four things from a method: the TOTAL energy
of one state as a function of the nuclei, its derivative, a way to rebuild
whatever the method froze at a reference geometry, and a name to print. Nothing
in that list mentions BSE@GW, a factorization, or even the existence of an
analytic gradient, which is why the protocol lives here rather than next to any
one chain.

TOTAL energy, never an excitation energy. Two states relax to different minima,
so E_0 does not cancel in an adiabatic gap and cannot be left out of the
surface; a method that reports Omega alone is not a surface.

REFREEZE BELONGS TO THE METHOD. Every cubic chain fixes discrete choices at a
reference geometry -- the quasiparticle set, the frame orientation, the
interpolation pair layout, the Newton branch, the residue backend -- because
each is a discontinuity in the surface otherwise. An optimizer that walks far
enough leaves the geometry those choices were made for, and the only honest
answer is to rebuild them at the new geometry and measure how far the minimum
moves. Only the method knows what it froze, so `refreeze` is its method and the
optimizer merely calls it.
"""
from typing import Protocol, runtime_checkable

import numpy as np

from src.Base.constants import NUCLEAR_FD_STEP

#: The attribute names a surface may use for its selected root, most specific
#: first. Only consulted when the class declares no `ROOT_ATTR`.
ROOT_ATTR_CANDIDATES = ('state', 'root')


@runtime_checkable
class PotentialEnergySurface(Protocol):
    """One state's total energy over the nuclei, and the conventions behind it.

    `mol0` is the reference geometry the frozen conventions were chosen at.

    `mean_field(mol) -> (mol, mf)` IS THE ACCESSOR for the mean field a surface
    evaluates its energy on, and the only one a property routine may use --
    through `surface_mean_field` below, which is what `vibronic` and
    `FiniteDifferenceGradient` call. When the surface stands in an environment
    that mean field is the environment's ground state: PCM at the static
    dielectric constant for a continuum, the classical charges in h_core for a
    point-charge field. Every surface owes this accessor, a gas-phase one
    returning its factory's mean field. It is named here rather than declared as
    a member because `isinstance` on a structural protocol is a presence check:
    declaring it would stop the gas-phase surfaces that still lean on
    `surface_mean_field`'s fallback from being surfaces at all.

    `scf_factory(mol) -> mf` IS NOT PART OF THIS PROTOCOL. It is the raw user
    function each surface keeps so that `refreeze` and a displaced geometry can
    rebuild themselves, and a solvated surface's own mean field is NOT that one
    -- the dRPA ground state of water/cc-pVDZ B3LYP in PCM(1.78/78.39) sits
    0.2385 eV away from it. For a gradient the factory must converge the ORBITAL
    gradient to ~1e-11, since the Lagrangian assumes the occupied-virtual Fock
    block vanishes.
    """

    mol0: object

    def total_energy(self, mol=None, mf=None) -> float:
        """Total energy of THIS state in Hartree, at `mol` or at the reference."""

    def total_gradient(self, mol=None, mf=None):
        """(dE/dR of shape (natm, 3), E, diagnostics), everything in Hartree.

        The diagnostics dict is method-specific; `omega` is there when the state
        is an excitation and absent on a ground-state surface, so a consumer
        reads it with `.get`.
        """

    def refreeze(self, mol):
        """The same surface with every frozen convention rebuilt at `mol`."""

    def label(self) -> str:
        """A short name for the state and the method, for logs and records."""


class FiniteDifferenceGradient:
    """A gradient for an ENERGY-ONLY surface, by central differences.

    The solvated chain is the case this exists for: its cavity moves with the
    atoms, its reverse pass does not carry that motion, and returning the
    gas-phase force would be wrong by the entire reaction-field response while
    looking perfectly reasonable. Differencing the energy is exact instead --
    every displaced geometry rebuilds its own cavity -- and it is the reference
    an analytic term has to reproduce.

    IT COSTS 6 natm ENERGIES PER GRADIENT (one displaced pair per Cartesian
    component) plus one at the reference point, each with its own SCF. That is
    a few hundred BSE@GW evaluations for a geometry optimization of anything
    past a handful of atoms, so it is a validation route and a last resort, not
    a way to run production.

    `h` is a Bohr displacement. The two-point difference has an O(h^2)
    truncation error and an O(eps/h) noise floor from the energy's own
    convergence, so a mean field converged to 1e-14 supports 1e-3 comfortably
    and a loose one does not.

    `map_fn` takes `map`'s signature and runs the 6 natm displaced energies:
    the default evaluates them in order in this process; a process pool
    spreads them over workers, since no displaced energy depends on another.
    """

    def __init__(self, surface, h=NUCLEAR_FD_STEP, map_fn=None):
        self.surface, self.h, self.map_fn = surface, h, map_fn

    @property
    def mol0(self):
        return self.surface.mol0

    def scf_factory(self, mol):
        return self.surface.scf_factory(mol)

    def mean_field(self, mol=None, mf=None):
        """(mol, mf) from the WRAPPED surface, never from this wrapper's factory.

        The solvated chain is the case: its mean field is the PCM one, and a
        record built from `scf_factory` would carry the bare one under a
        solvated energy.
        """
        mol = self.surface.mol0 if mol is None else mol
        return mol, (surface_mean_field(self.surface, mol) if mf is None else mf)

    def total_energy(self, mol=None, mf=None):
        """The wrapped surface's energy, unchanged."""
        return self.surface.total_energy(mol, mf)

    def total_gradient(self, mol=None, mf=None):
        """(dE/dR, E, diagnostics) with dE/dR differenced, not differentiated.

        EVERY DISPLACED ENERGY IS LEFT TO BUILD ITS OWN MEAN FIELD. Handing it
        `scf_factory`'s would difference a gas-phase surface under a solvated
        reference energy -- 0.24 eV apart on water in PCM -- which is not a
        derivative of anything.
        """
        mol = self.surface.mol0 if mol is None else mol
        e0 = self.surface.total_energy(mol, mf)
        crd = np.asarray(mol.atom_coords())

        def displaced_energy(job):
            ia, x, sign = job
            d = np.zeros((mol.natm, 3))
            d[ia, x] = sign * self.h
            m = mol.copy()
            m.set_geom_(crd + d, unit='Bohr')
            m.build(False, False)
            return self.surface.total_energy(m)

        # The 6 natm displaced energies are independent; the pair of one
        # component is differenced afterwards, in the same arithmetic as before.
        jobs = [(ia, x, sign) for ia in range(mol.natm) for x in range(3)
                for sign in (-1.0, 1.0)]
        runner = map if self.map_fn is None else self.map_fn
        energy = dict(zip(jobs, runner(displaced_energy, jobs)))
        grad = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            for x in range(3):
                grad[ia, x] = ((energy[(ia, x, 1.0)] - energy[(ia, x, -1.0)])
                               / (2.0 * self.h))
        return grad, e0, {'fd_step': self.h}

    def refreeze(self, mol):
        """Refreeze the wrapped surface and keep differencing it."""
        return FiniteDifferenceGradient(self.surface.refreeze(mol), self.h,
                                        map_fn=self.map_fn)

    def label(self):
        return f'{self.surface.label()} [finite difference, h = {self.h} Bohr]'


def root_attribute(surface):
    """The name of the attribute that selects `surface`'s root.

    Declared as `ROOT_ATTR` on the class, or inferred when exactly one of
    `ROOT_ATTR_CANDIDATES` is present. The chains spell it differently --
    `state` on the BSE chains, `root` on the downfolded ones. Ambiguity is
    refused rather than resolved by precedence: a surface carrying both would
    silently have one of its two meanings driven.
    """
    declared = getattr(type(surface), 'ROOT_ATTR', None)
    if declared is not None:
        if not hasattr(surface, declared):
            raise AttributeError(
                f'{type(surface).__name__}.ROOT_ATTR names {declared!r}, which '
                f'the instance does not have')
        return declared
    present = [a for a in ROOT_ATTR_CANDIDATES if hasattr(surface, a)]
    if len(present) == 1:
        return present[0]
    raise AttributeError(
        f'{type(surface).__name__} carries {present or "no"} root attribute, so '
        f'the manifold cannot tell which selects the state; declare '
        f'ROOT_ATTR = \'<name>\' on the class')


def own_root_attribute(surface):
    """`root_attribute` where it resolves, None where it does not.

    A composed surface may carry a root index of its own or none at all, and a
    caller has to set the one that exists without refusing the one that does not.
    """
    try:
        return root_attribute(surface)
    except AttributeError:
        return None


def driven_chain(surface):
    """The half of `surface` whose root attribute selects the state.

    A COMPOSED surface holds its ground state in `ground` and its state in
    `excited`. Driving the outer index alone moves nothing on the BSE surface,
    whose excitation is read from `excited.state`: every root then comes back
    as root 0, which on water is 2.08 eV low and raises nothing anywhere. The
    half that owns the spectrum is the one to drive.

    Protocol vocabulary, and here rather than beside `StateManifold` because
    both a manifold in `src.gradients` and the derivative couplings in
    `src.properties.nonadiabatic` drive a surface through it; a copy on either
    side would be two answers to which half owns the spectrum.
    """
    half = getattr(surface, 'excited', None)
    if half is None or not (hasattr(half, '_forward')
                            or own_root_attribute(half) is not None):
        return surface
    return half


def surface_mean_field(surface, mol):
    """The mean field `surface` itself evaluates at `mol`, through `mean_field`.

    THE PROTOCOL'S ACCESSOR IS `mean_field(mol) -> (mol, mf)`; `scf_factory` is
    the raw user factory and is not part of the protocol. A surface standing in
    an environment builds its mean field THROUGH it -- PCM at the static
    dielectric for a continuum, the classical charges in h_core for a point-
    charge field -- so the factory's is a different surface. The fallback is for
    a gas-phase surface whose factory IS its mean field. Measured on
    water/cc-pVDZ B3LYP in PCM(1.78/78.39), the dRPA ground state is
    -76.31502422 Ha on its own mean field and -76.30626038 Ha on the factory's,
    0.2385 eV apart; a record, an off-diagonal energy or a reorganization
    energy built from the second is not on the surface the optimizer relaxed.
    """
    if hasattr(surface, 'mean_field'):
        return surface.mean_field(mol)[1]
    return surface.scf_factory(mol)
