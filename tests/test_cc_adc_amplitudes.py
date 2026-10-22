"""CC-ADC(3) amplitude substitution: pyscf CC -> ADC amplitude-pool conventions
(src/SingleReference/ADC/cc_adc_amplitudes.py).

The conventions are the whole risk surface -- a transposed or sign-flipped
amplitude produces a plausible IP -- so each is pinned against an independent
pyscf construction rather than against itself:

  * t2's layout/sign against pyscf MP2 (the CC t2 shares the layout of the MP2
    t2 both codes agree on);
  * t1's layout/sign against one pyscf CCSD singles update from
    (t1 = 0, t2 = t2_MP1), which is the second-order singles T1^(2).

The supermatrix-level checks (Hermiticity under the substitution, restricted
against spin-orbital IP, the null test, the g-free DF route) need the
generated ADC(3) runtime and arrive with it.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import cc, gto, mp, scf
from pyscf.cc import dfccsd

from src.Base.pyscf_interface import (DFIntegrals,
                                      get_antisymmetrized_spin_block_eri,
                                      get_orbital_energies)
from src.SingleReference.ADC.cc_adc_amplitudes import (
    InjectedAmplitudePool, cc_amplitudes_restricted,
    cc_amplitudes_spinorbital, mbpt_amplitudes_restricted,
    mbpt_amplitudes_restricted_df)
from src.SingleReference.ADC.cc_amplitudes import _run_cc
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import (
    MPnDensityDriverRestricted)

EXACT_DF_TOL = 1e-10     # exact DF factor: a pure plumbing check
RI_MIN = 1e-8            # a real auxiliary basis must differ by more than round-off


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12          # f_ov noise leaks straight into T1^(2)
    mf.run()
    return mf


def test_mbpt_pool_matches_pyscf_mp2_and_a_first_ccsd_singles_update(water):
    """The pool the substitution replaces, pinned to pyscf: t2 is the MP2
    doubles, t1 is what one CCSD singles update from (0, t2_MP1) produces."""
    mf = water
    nocc = mf.mol.nelectron // 2
    pool = mbpt_amplitudes_restricted(mf)
    pt = mp.MP2(mf).run()
    t2_abab = pt.t2.transpose(2, 3, 0, 1)
    assert np.abs(pool['t2_abab'] - t2_abab).max() < 1e-13
    assert np.abs(pool['t2_aaaa']
                  - (t2_abab - t2_abab.transpose(1, 0, 2, 3))).max() < 1e-13

    mycc = cc.CCSD(mf).run()
    t1_2, _ = mycc.update_amps(np.zeros_like(mycc.t1), pt.t2,
                               mycc.ao2mo(mf.mo_coeff))
    assert np.abs(pool['t1_aa'] - t1_2.T).max() < 1e-8

    # the restricted driver is the pool's own source: same numbers
    g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mf.mol, mf)
    mpn = MPnDensityDriverRestricted(
        np.diag(get_orbital_energies(mf, representation='spatial')),
        g_aaaa, g_abab, g_bbbb, nocc)
    t2_aaaa, t2_abab_drv = mpn.compute_t2_1()
    assert np.abs(pool['t2_aaaa'] - t2_aaaa).max() < 1e-14
    assert np.abs(pool['t2_abab'] - t2_abab_drv).max() < 1e-14


def test_cc_amplitudes_restricted_and_spinorbital_are_the_same_solution(water):
    """The spin-block and spin-orbital converters must expand one CCSD
    solution: t2_aaaa/t2_abab are the alpha-alpha/alpha-beta slices of the
    interleaved spin-orbital t2."""
    r = cc_amplitudes_restricted(water)
    s = cc_amplitudes_spinorbital(water)
    a_v, a_o = slice(0, None, 2), slice(0, None, 2)
    b_v, b_o = slice(1, None, 2), slice(1, None, 2)
    assert np.abs(r['t1_aa'] - s['t1'][a_v, a_o]).max() < 1e-13
    assert np.abs(r['t2_aaaa'] - s['t2'][a_v, a_v, a_o, a_o]).max() < 1e-13
    assert np.abs(r['t2_abab'] - s['t2'][a_v, b_v, a_o, b_o]).max() < 1e-13


def test_ccd_has_no_singles(water):
    """CCD is the T1-free CC doubles model: substituting it deletes the
    second-order singles channel from U^(2) rather than replacing it."""
    amps = cc_amplitudes_restricted(water, method='ccd')
    assert np.abs(amps['t1_aa']).max() == 0.0
    assert np.abs(amps['l1_aa']).max() == 0.0
    assert np.abs(amps['t2_abab']).max() > 1e-3


def test_missing_amplitude_tag_raises_with_a_diagnosis(water):
    """An ADC(4)+ consumer asking for T3^(2) must hit a named error, not a
    KeyError from inside a generated split."""
    pool = InjectedAmplitudePool(cc_amplitudes_spinorbital(water), 'spin-orbital CC')
    assert pool['t2'] is not None
    with pytest.raises(NotImplementedError, match='t3'):
        pool['t3']


@pytest.fixture(scope='module')
def water_df():
    """DF mean field: the production route's starting point."""
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = 1e-12
    mf.run()
    return mf


def test_ccd_is_pyscfs_ccd_and_follows_the_mean_field_into_df(water, water_df):
    """CCD must reproduce pyscf's own on a conventional mean field, and must
    not drop density fitting on a DF one (pyscf.cc.ccd.CCD does: it hard-codes
    the conventional update_amps)."""
    from pyscf.cc import ccd as pyscf_ccd
    ref = pyscf_ccd.CCD(water).run()
    mine = _run_cc(water, 'ccd', None)
    assert abs(ref.e_corr - mine.e_corr) < 1e-10
    assert np.abs(np.asarray(ref.t2) - np.asarray(mine.t2)).max() < 1e-10

    df_cc = _run_cc(water_df, 'ccd', None)
    assert isinstance(df_cc, dfccsd.RCCSD)
    assert np.abs(np.asarray(df_cc.t1)).max() == 0.0


@pytest.mark.parametrize('atom', ['H 0 0 0; F 0 0 0.917',
                                  'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'])
def test_df_pool_equals_dense_pool(atom):
    """mbpt_amplitudes_restricted_df against its dense twin. With the exact
    DF factor (naux = norb^2) every key, doubles and singles alike, agrees to
    solver precision; with a real auxiliary basis the two differ by the RI
    error (identical would mean the DF path fell back to the dense one)."""
    mol = gto.M(atom=atom, basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.run()
    dense = mbpt_amplitudes_restricted(mf)
    exact = mbpt_amplitudes_restricted_df(
        mf, DFIntegrals.from_scf(mol, mf, exact=True).B_aa)
    assert set(exact) == set(dense)
    worst = max(float(np.max(np.abs(dense[k] - exact[k]))) for k in dense)
    print(f"{atom.split()[0]}../6-31g exact DF pool vs dense: {worst:.2e}")
    assert worst < EXACT_DF_TOL, worst

    mf_df = scf.RHF(mol).density_fit()
    mf_df.conv_tol = 1e-12
    mf_df.run()
    ri = mbpt_amplitudes_restricted_df(mf_df, DFIntegrals.from_scf(mol, mf_df).B_aa)
    dense_df = mbpt_amplitudes_restricted(mf_df)
    diff = max(float(np.max(np.abs(dense_df[k] - ri[k]))) for k in ri)
    print(f"{atom.split()[0]}../6-31g real RI pool vs dense: {diff:.2e}")
    assert diff > RI_MIN, diff
