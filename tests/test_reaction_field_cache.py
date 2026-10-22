"""The Eq. (18) reaction field is cached under every input of the shift.

Beside the spectrum and the environment (tests/test_reaction_field.py), the
cache key carries the orbitals, the occupation, the geometry and both basis
sets: a second auxiliary basis, another occupation or a copy at a displaced
geometry whose eigenvalues were left as they were must be rebuilt, not served
the resident array (water/cc-pVDZ in eps 1.78: cc-pvdz-jkfit against -ri
differs by 1.2e-3 Ha).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.environment import attach_environment
from src.Base.solvent_screening import (SolventScreening,
                                        detach_solvent_screening)
from src.SingleReference.GW.reaction_field import environment_quasiparticle_shift
from tests.pins import assert_same


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                 basis='cc-pvdz', verbose=0)


@pytest.fixture
def mf(mol):
    mean_field = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


def gap_shift_ev(shift, nocc):
    """What Eq. (18) does to the quasiparticle gap, in eV."""
    return (shift[nocc] - shift[nocc - 1]) * HARTREE_TO_EV


def _uncached(mf, **kwargs):
    """Eq. (18) built afresh, the resident entry set aside and put back."""
    resident = mf.__dict__.pop('_reaction_field_shift', None)
    try:
        return environment_quasiparticle_shift(mf, **kwargs)
    finally:
        if resident is not None:
            mf._reaction_field_shift = resident


def test_another_auxiliary_basis_is_not_served_the_cached_shift(mf):
    """The fit basis is an input of Delta W: a second one rebuilds."""
    attach_environment(mf, SolventScreening(mf.mol, eps=1.78))
    ri = environment_quasiparticle_shift(mf, auxbasis='cc-pvdz-ri')
    jk = environment_quasiparticle_shift(mf, auxbasis='cc-pvdz-jkfit')
    expected = _uncached(mf, auxbasis='cc-pvdz-jkfit')
    print(f'max |jkfit - ri| = {np.abs(expected - ri).max():.3e} Ha')
    assert np.abs(expected - ri).max() > 1e-6
    # the rebuild and a second build of it: two computations, so to a few
    # ulp, not bitwise
    assert_same(jk, expected, quantity='energy', label='jkfit')
    detach_solvent_screening(mf)


def test_another_occupation_is_not_served_the_cached_shift(mf):
    """nocc splits chi0 into its pairs: a second nocc rebuilds."""
    nocc = mf.mol.nelectron // 2
    attach_environment(mf, SolventScreening(mf.mol, eps=1.78))
    environment_quasiparticle_shift(mf, nocc=nocc)
    other = environment_quasiparticle_shift(mf, nocc=nocc - 1)
    assert_same(other, _uncached(mf, nocc=nocc - 1), quantity='energy',
                label='nocc - 1')
    detach_solvent_screening(mf)


def test_another_geometry_is_not_served_the_cached_shift(mf):
    """A copy that carries the cache to new orbitals at a displaced geometry,
    with the eigenvalues left as they were, rebuilds."""
    attach_environment(mf, SolventScreening(mf.mol, eps=1.78))
    resident = environment_quasiparticle_shift(mf)
    moved = mf.mol.set_geom_('O 0 0 0.127; H 0 0.767 -0.478; H 0 -0.767 -0.478',
                             inplace=False)
    displaced = scf.RHF(moved).density_fit(auxbasis='cc-pvdz-ri')
    displaced.kernel()
    copy = mf.copy()
    copy.mol, copy.mo_coeff = moved, displaced.mo_coeff
    assert copy._reaction_field_shift is mf._reaction_field_shift
    shift = environment_quasiparticle_shift(copy)
    expected = _uncached(copy)
    assert np.abs(expected - resident).max() > 1e-6
    assert_same(shift, expected, quantity='energy', label='displaced')
    detach_solvent_screening(mf)
