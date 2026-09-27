"""Refusals that keep a solvated or open-shell run from returning a number for a
different system.

  1. An ROHF/ROKS reference is refused by the restricted routes: it is not an
     unrestricted object, so a UHF test lets it through, and the restricted
     split at nelectron // 2 would count the singly occupied orbital as virtual.
     A UHF reference passes.
  2. Coupled cluster refuses a mean field that carries a continuum -- an
     attached screening or a pyscf PCM ground state -- since no CC-in-continuum
     model exists here and the integral paths would treat it inconsistently;
     GW@CC in a continuum is refused too, since its Eq. (18) shift would be
     RPA-screened beside a CC-screened Sigma.
  3. A pyscf PCM ground state and the optical screening attached to it must
     share one cavity; a different Lebedev order is refused.
  4. `SolventScreening.mean_field` without eps_static warns that the ground
     state stays in the gas phase (the frozen-polarization limit).
  5. The Eq. (18) cache distinguishes calls that differ in anything but the
     environment -- here, the occupation the orbitals are split by.

Run as a script (`python tests/test_environment_guards.py`) or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.solvent_screening import (SolventScreening, attach_solvent_screening,
                                        detach_solvent_screening)
from src.SingleReference.CC.integrals import (build_restricted_integrals_from_mf,
                                              build_spinorbital_integrals_from_mf)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.reaction_field import environment_quasiparticle_shift

WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
OH = 'O 0 0 0; H 0 0 0.97'


def _water():
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    return mol, scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()


def test_rohf_is_refused_uhf_passes():
    mol = gto.M(atom=OH, basis='6-31g', spin=1, verbose=0)
    rohf = scf.ROHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()
    with pytest.raises(NotImplementedError, match='closed-shell aufbau'):
        calc_qp_energy(rohf, selfenergy='GW', df=True, state='homo')
    attach_solvent_screening(rohf, eps=1.78)
    try:
        with pytest.raises(NotImplementedError, match='closed-shell aufbau'):
            environment_quasiparticle_shift(rohf)
    finally:
        detach_solvent_screening(rohf)
    uhf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()
    assert np.isfinite(calc_qp_energy(uhf, selfenergy='GW', df=True,
                                      state='homo'))


def test_cc_refuses_a_continuum():
    mol, mf = _water()
    attach_solvent_screening(mf, eps=1.78)
    try:
        with pytest.raises(NotImplementedError, match='no continuum model'):
            build_restricted_integrals_from_mf(mf)
        with pytest.raises(NotImplementedError, match='no continuum model'):
            build_spinorbital_integrals_from_mf(mf)
    finally:
        detach_solvent_screening(mf)
    pcm = scf.RHF(mol).PCM()
    pcm.with_solvent.eps = 78.355
    pcm.run()
    with pytest.raises(NotImplementedError, match='pyscf PCM'):
        build_restricted_integrals_from_mf(pcm)
    build_restricted_integrals_from_mf(mf)          # the gas phase still runs
    attach_solvent_screening(mf, eps=1.78)
    try:
        with pytest.raises(NotImplementedError, match='two screenings'):
            calc_qp_energy(mf, selfenergy='GW', polarizability='CCSD',
                           state='homo')
    finally:
        detach_solvent_screening(mf)


def test_ground_state_and_screening_share_one_cavity():
    mol, _ = _water()
    pcm = scf.RHF(mol).PCM()
    pcm.with_solvent.eps = 78.355
    pcm.with_solvent.method = 'IEF-PCM'
    pcm.with_solvent.lebedev_order = 17                 # not the screening's
    pcm.run()
    with pytest.raises(ValueError, match='lebedev_order'):
        attach_solvent_screening(pcm, eps=1.78)
    screening = SolventScreening(mol, eps=1.78, eps_static=78.355)
    matched = screening.mean_field(mol, lambda m: scf.RHF(m).run())
    attach_solvent_screening(matched, eps=1.78)       # one cavity: accepted
    detach_solvent_screening(matched)


def test_missing_eps_static_warns():
    mol, _ = _water()
    screening = SolventScreening(mol, eps=1.78)
    with pytest.warns(RuntimeWarning, match='frozen-polarization'):
        screening.mean_field(mol, lambda m: scf.RHF(m).run())


def test_shift_cache_keys_on_the_occupation():
    mol, mf = _water()
    nocc = mol.nelectron // 2
    attach_solvent_screening(mf, eps=1.78)
    try:
        s_full = environment_quasiparticle_shift(mf, nocc=nocc)
        s_less = environment_quasiparticle_shift(mf, nocc=nocc - 1)
    finally:
        detach_solvent_screening(mf)
    assert not np.array_equal(s_full, s_less), \
        'a different occupation must not be served the cached array'


TESTS = [test_rohf_is_refused_uhf_passes, test_cc_refuses_a_continuum,
         test_ground_state_and_screening_share_one_cavity,
         test_missing_eps_static_warns, test_shift_cache_keys_on_the_occupation]


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f'[OK  ] {test.__name__}')
        except (Exception, pytest.fail.Exception) as exc:   # noqa: BLE001
            failed += 1
            print(f'[FAIL] {test.__name__} -- {type(exc).__name__}: {exc}')
    print('\nALL PASSED' if not failed else '\nFAILURES DETECTED')
    sys.exit(0 if not failed else 1)
