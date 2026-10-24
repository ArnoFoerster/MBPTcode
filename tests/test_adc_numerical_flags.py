"""Iterations inside the ADC routes that stop before their tolerance warn
(solve.ADCNotConverged), keep their last iterate and hand a converged=False
flag to the caller, instead of raising:

- the CCD/CCSD amplitudes of Faddeev-ADC(3) and CC-ADC(3) (cc_amplitudes.run_cc):
  `out['converged']` and `mycc.converged`;
- the Brueckner reference: `info['converged']`;
- the ladder CCD of the Faddeev-ADC(3) Riccati route: `info['converged']`;
- a FLEX Lanczos resolvent: `settled`, and `solve_qp(...)['converged']`;
- the Lanczos inverse square root of the Faddeev metric.

H2O/STO-3G and stand-ins; each check runs in about a second.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.SingleReference.ADC import adc_r_faddeev, adc_r_flex, cc_amplitudes
from src.SingleReference.ADC.solve import ADCNotConverged, apply_inverse_sqrt


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
                basis='sto-3g', verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-12)


def test_cc_amplitudes_out_of_cycles_warn_and_flag(water):
    out = {}
    with pytest.warns(ADCNotConverged, match='CCD did not converge'):
        mycc = cc_amplitudes.run_cc(water, 'ccd', None, out=out, max_cycle=1)
    assert out == {'converged': False} and not mycc.converged
    assert np.all(np.isfinite(mycc.t2))
    out = {}
    cc_amplitudes.run_cc(water, 'ccsd', None, out=out)
    assert out == {'converged': True}


def test_a_short_brueckner_reference_keeps_its_orbitals_and_is_flagged(water):
    with pytest.warns(ADCNotConverged, match='BCCD did not converge'):
        mfb, info = cc_amplitudes.brueckner_reference(water, conv_tol_normu=1e-14,
                                                      max_cycle=1)
    assert info['converged'] is False
    assert np.all(np.isfinite(mfb.mo_energy))


def test_a_short_ladder_ccd_keeps_its_iterate_and_is_flagged(water):
    eps, nocc = water.mo_energy, water.mol.nelectron // 2
    C = water.mo_coeff
    eri = water.mol.ao2mo(C, compact=False).reshape((len(eps),) * 4)
    with pytest.warns(ADCNotConverged, match='ladder CCD not converged'):
        lad = adc_r_faddeev.solve_ladder_amplitudes_restricted(
            eps, nocc, eri_chemist=eri, max_iter=1)
    assert lad.info['converged'] is False and lad.info['niter'] == 1
    full = adc_r_faddeev.solve_ladder_amplitudes_restricted(eps, nocc, eri_chemist=eri)
    assert full.info['converged'] is True


def test_an_unsettled_flex_resolvent_keeps_its_value_and_is_flagged(monkeypatch):
    rng = np.random.default_rng(3)
    A = rng.standard_normal((300, 300))
    H = A + A.T
    monkeypatch.setattr(adc_r_flex, 'LANCZOS_MAX_STEPS', 2 * adc_r_flex.LANCZOS_BLOCK)
    monkeypatch.setattr(adc_r_flex, 'SIGMA_CF_TOL', 0.0)
    r = adc_r_flex._Resolvent(lambda x: H @ x, rng.standard_normal(300))
    with pytest.warns(ADCNotConverged, match='not settled'):
        s, d = r(0.3)
    assert np.isfinite(s) and np.isfinite(d) and r.settled is False


def test_a_short_inverse_square_root_keeps_its_approximation():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((40, 40))
    N = np.eye(40) + 0.3 * (A @ A.T) / 40
    with pytest.warns(ADCNotConverged, match='not settled'):
        y = apply_inverse_sqrt(lambda x: N @ x, rng.standard_normal(40), tol=0.0,
                               max_steps=2)
    assert np.all(np.isfinite(y))
