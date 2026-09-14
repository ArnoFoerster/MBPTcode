"""What the surroundings do to a molecule, in one contract every route consumes.

The two-electron interaction is dressed on
the auxiliary metric, v -> v + vtilde with vtilde = v chi v (Duchemin,
Jacquemin and Blase, J. Chem. Phys. 144, 164106 (2016), Eq. 16), which
`separable_ri.aux_metric_sqrt` applies to every separable factorization; and
the static one-body term of the quasiparticle equation gains the first-order
reaction-field operator, the static COHSEX Sigma^solv of their Eqs. (20)-(22),
which `qp_solve.static_exchange_diagonal` adds.

`Environment` is structural: `SolventScreening` satisfies it without inheriting
from anything. An environment that does not screen returns None from
`aux_kernel`; `static_self_energy` is None wherever no COHSEX operator is
offered, and the two are independent of each other -- `dresses_interaction`
below is the single place the first question is asked. Routes find the
environment of a calculation on the mean field (`attach_environment`,
`environment_of`).
"""
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

import numpy as np
from pyscf import qmmm
from pyscf.qmmm import itrf as qmmm_itrf

from src.Base.constants import BOHR_TO_ANGSTROM


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
    144, 164106 (2016) Eqs. (20)-(22), summed over every orbital
    What the concrete environments do:

    - `NoEnvironment`: neither channel. The gas phase.
    - `PointCharges`: neither. A permanent charge does not respond, so it
      changes the mean field and nothing after it.
    - `SolventScreening`: dresses (the continuum's reaction field), and also
      offers the COHSEX operator.
    """

    #: whether this environment dresses the interaction at all
    screens: bool

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


class NoEnvironment:
    """
    The gas phase: nothing is dressed, nothing is added.
    """

    screens = False

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

    def __repr__(self):
        return 'NoEnvironment()'


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

    def __repr__(self):
        return (f'PointCharges({len(self.charges)} charges, total '
                f'{self.charges.sum():+.3f})')


def dresses_interaction(environment, auxmol) -> bool:
    """
    Does `environment` change the interaction the post-SCF methods see?
    """
    return (environment is not None and getattr(environment, 'screens', True)
            and environment.aux_kernel(auxmol) is not None)


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
