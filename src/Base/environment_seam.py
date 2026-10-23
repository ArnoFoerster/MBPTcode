"""The seam between a calculation and its surroundings: what an environment
is (`Environment`, structural), the gas phase (`NoEnvironment`), and where a
route finds the environment of a calculation -- on its mean field
(`attach_environment`, `environment_of`, `attached_environment`,
`resolve_environment`).

It sits in Base because the integral getters of the pySCF interface read the
environment a mean field carries; the media themselves (continuum, point
charges, polarizable sites) are the environments package
(`src.environments`), and nothing here knows any of them.
"""
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Environment(Protocol):
    """The contract: two entries into the energy.

    TWO CHANNELS, AND ONLY ONE OF THEM IS A CHOICE.

    `aux_kernel` dresses the two-electron interaction, v -> v + vtilde. Every
    quantity built from the interaction then carries the environment: the
    screening in W, the BSE kernel, and -- because the dressing IS what moves a
    quasiparticle -- the self-polarization shift

        Delta eps_p = -/+ (1/2) <pp|Delta W|pp>,   Delta W = W[v + vtilde] - W[v]

    of Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18).
    That shift is the static approximation to Sigma[W_dressed] - Sigma[W_bare],

    `static_self_energy` is the other channel the
    static COHSEX Sigma^solv of Duchemin, Jacquemin and Blase, J. Chem. Phys.
    144, 164106 (2016) Eqs. (20)-(22), summed over every orbital rather than
    taken on the self-element, for the routes that never form Delta W -- the ADC
    self-energies and an unrestricted reference. It
    may legitimately be None while the interaction is dressed, and an
    environment that offers it does not thereby gain or lose the shift.

    What the three concrete environments do:

    - `NoEnvironment`: neither channel. The gas phase.
    - `PointCharges`: neither. A permanent charge does not respond, so it
      changes the mean field and nothing after it.
    - `SolventScreening`: dresses (the continuum's reaction field), and also
      offers the COHSEX operator.
    """

    #: whether this environment dresses the interaction at all
    screens: bool

    #: whether both adjoints below are built, i.e. whether a chain in this
    #: environment can return a force at all. A chain asks BEFORE the reverse
    #: pass, so a missing derivative costs a message rather than the run.
    differentiable: bool

    def for_geometry(self, mol):
        """
        This environment around the atoms of `mol` (a cavity moves with them).
        """

    def mean_field(self, mol, scf_factory):
        """
        The converged mean field of `mol` in this environment, built with `scf_factory`.
        """

    def aux_kernel(self, auxmol):
        """
        vtilde_PQ between auxiliary functions, (naux, naux), or None when nothing screens.
        """

    def whitened_transform(self, mol, mf):
        """
        T with B -> T B turning a Coulomb-fitted RI factor into a
        (v + vtilde)-fitted one, or None when nothing screens.
        """

    def kernel_mo(self, mol, mo_bra, mo_ket=None):
        """
        vtilde as a chemist-order (pq|rs) correction in an MO basis, or None.
        """

    def kernel_ao(self, mol):
        """
        vtilde as a chemist-order (mu nu|lam sig) correction in the AO basis, or None.
        """

    def static_self_energy(self, mf, mol=None):
        """
        The COHSEX reaction-field operator in the MO basis of `mf`, or None
        when this environment offers none -- the routes that form Delta W do
        not read it.
        """

    def dynamic_factor(self, omega):
        """
        g(iw) in [0, 1]: the fraction of `aux_kernel` surviving at imaginary
        frequency w, elementwise over an array. An environment with no
        dynamics of its own returns 1 (the adiabatic limit).
        """

    def aux_kernel_adjoint(self, auxmol, v_bar):
        """
        (natm, 3): d/dR sum_PQ v_bar[P,Q] vtilde_PQ, for a symmetric adjoint v_bar.
        """

    def static_self_energy_adjoint(self, mf, weights):
        """
        (Y_extra, skeleton) of sum_p weights[p] <p|Sigma^env|p>: the orbital-rotation
        gradient that joins the Lagrangian before the multiplier solve, and the
        (natm, 3) skeleton at fixed coefficients.
        """


class NoEnvironment:
    """
    The gas phase: nothing is dressed, nothing is added.
    """

    screens = False
    differentiable = True

    def for_geometry(self, mol):
        return self

    def mean_field(self, mol, scf_factory):
        return scf_factory(mol)

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
        return np.ones_like(np.asarray(omega, float))

    def aux_kernel_adjoint(self, auxmol, v_bar):
        return np.zeros((auxmol.natm, 3))

    def static_self_energy_adjoint(self, mf, weights):
        return 0.0, np.zeros((mf.mol.natm, 3))

    def __repr__(self):
        return 'NoEnvironment()'


def environment_of(mf):
    """
    The environment attached to `mf`, or the gas phase.
    """
    environment = getattr(mf, 'with_screening', None)
    return NoEnvironment() if environment is None else environment


def attach_environment(mf, environment):
    """
    Make every post-SCF route on `mf` see `environment`; returns `mf`.
    """
    mf.with_screening = environment
    keys = getattr(mf, '_keys', None)
    if keys is not None:
        keys.add('with_screening')          # keep pyscf's check_sanity quiet
    return mf


@contextmanager
def attached_environment(mf, environment):
    """
    `environment` on `mf` for the length of the block, the previous one put back.

    A route evaluates part of its chain on a mean field the caller may use
    elsewhere; leaving a cavity attached would silently move that caller's
    numbers.
    """
    previous = getattr(mf, 'with_screening', None)
    attach_environment(mf, environment)
    try:
        yield mf
    finally:
        mf.with_screening = previous


def resolve_environment(environment=None, mf=None):
    """
    The environment a gradient chain carries: the one given, else the one
    attached to its reference mean field, else the gas phase.

    Adopting the attached one is what makes half an answer impossible: the
    static term the mean field would already have shown the chain and the
    substitution in the metric it would not come from one object.
    """
    if environment is not None:
        return environment
    if mf is not None:
        return environment_of(mf)
    return NoEnvironment()
