"""Warn-and-continue sites name their event where the result is kept
(refuse inputs, warn numerics).

Each site keeps warning and carries on; the event is also handed to the
object the caller records:

- the BSE Davidson's unit-vector guess cut inside a degenerate cluster:
  `counters['guess_truncated']` -> timings 'davidson_guess_truncated';
- `sigma_sop` outside the compression condition: `out['compression_excluded']`;
- complex EOM-CC eigenvalues: a `ComplexEOMEigenvalues` warning (no longer a
  print) and `EOMResult.complex_eigenvalues` / `max_imag`;
- an ambiguous orbital track: `track_orbital(out=...)['ambiguous']`;
- a spin-contaminated unrestricted reference: 'spin_contaminated' in the
  diagnostics, kept on the mean field as `reference_diagnostics`;
- an ill-conditioned Matsubara IR fit: `IRBasis.last_fit`;
- the fragment pole guard deciding on an unconverged Davidson:
  `guard_flags['q_lowest_unconverged']`;
- a followed quasiparticle root's branch: `bracketed_root(out=...)`.

Stand-ins and tiny molecules (H2, H atom); each runs in well under a second.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base import pyscf_interface
from src.Base.utils import matsubara
from src.Base.utils.matsubara import IRBasis
from src.properties import characters, fragment_bse
from src.SingleReference.CC import eom
from src.SingleReference.GW.sum_over_poles import sigma_sop
from src.SingleReference.LinearResponse.davidson import _davidson_counters
from src.SingleReference.LinearResponse.davidson_shared import _guess_indices
from src.Solvers.qp_equation import bracketed_root


def h2(**kw):
    return gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0, **kw)


def test_a_truncated_davidson_guess_is_counted():
    assert _davidson_counters()['guess_truncated'] is False
    counters = _davidson_counters()
    with pytest.warns(RuntimeWarning, match='guess truncated'):
        order = _guess_indices(np.zeros((10, 10)), 10, counters=counters)
    assert len(order) == 40
    assert counters['guess_truncated'] is True
    # a gapped spectrum is not cut and not flagged
    clean = _davidson_counters()
    _guess_indices(np.arange(100.0).reshape(10, 10), 10, counters=clean)
    assert clean['guess_truncated'] is False


def test_sigma_sop_names_an_excluded_state():
    eps = np.array([-10.0, -0.5, 0.3, 1.0])
    poles = np.array([0.8, 1.5])
    amplitudes = np.full((2, 4), 0.01)
    out = {}
    with pytest.warns(RuntimeWarning, match='excludes this state'):
        value = sigma_sop(-9.9, amplitudes, poles, eps, 2, out=out)
    assert out['compression_excluded'] > 1.0
    assert value == sigma_sop(-9.9, amplitudes, poles, eps, 2, check=False)


def test_complex_eom_eigenvalues_warn_and_flag(monkeypatch):
    mf = scf.RHF(h2()).run()
    solver = eom.EOMCC(mf, level='ccsd')

    def complex_roots(aop, guess, precond, nroots=1, **kw):
        es = np.array([0.9 + 2e-3j, 1.1 - 2e-3j, 1.3, 1.5])[:nroots]
        return [True] * nroots, es, [guess[0]] * nroots
    monkeypatch.setattr(eom.lib, 'davidson_nosym1', complex_roots)
    with pytest.warns(eom.ComplexEOMEigenvalues, match=r'\|Im\| up to 2'):
        result = solver.kernel(sector='ee', nroots=2)
    assert result.complex_eigenvalues and result.max_imag == pytest.approx(2e-3)


def test_an_ambiguous_orbital_track_is_named():
    mf = scf.RHF(h2()).run()
    mol = mf.mol
    fingerprint = {'mol': mol, 'localized': np.eye(2),
                   'weights': np.array([0.5, 0.5]), 'window': (0, 2),
                   'occupied': True}
    out = {}
    with pytest.warns(RuntimeWarning, match='ambiguous'):
        characters.track_orbital(fingerprint, mf, mo_coeff=np.eye(2),
                                 candidates=[0, 1], out=out)
    assert sorted(out['ambiguous']['candidates']) == [0, 1]
    assert out['ambiguous']['margin'] < characters.TRACKING_MARGIN


def test_a_spin_contaminated_reference_is_flagged(monkeypatch):
    mf = scf.UHF(gto.M(atom='H 0 0 0', spin=1, basis='sto-3g',
                       verbose=0)).run()
    mf.spin_square = lambda: (1.5, 3.0)
    # pyscf's stability analysis divides by zero on a one-orbital system;
    # the test is about the contamination flag, so the analysis is stood in
    monkeypatch.setattr(pyscf_interface.scf_stability, 'uhf_internal',
                        lambda mf, return_status, verbose: (None, True))
    with pytest.warns(RuntimeWarning, match='spin-contaminated'):
        out = pyscf_interface.unrestricted_reference_diagnostics(mf, 'test')
    assert out['spin_contaminated'] is True
    assert mf.reference_diagnostics is out


def test_an_ill_conditioned_ir_fit_is_kept_on_the_basis(monkeypatch):
    b = IRBasis(100.0, eps=1e-8)
    assert b.last_fit is None
    n = b.default_matsubara_sampling()
    values = b.uhat(n).T @ np.linspace(1.0, 0.1, b.size)
    b.fit_matsubara(n, values)
    assert b.last_fit['ill_conditioned'] is False
    monkeypatch.setattr(matsubara, '_COND_WARN', 0.0)
    with pytest.warns(RuntimeWarning, match='conditioned at'):
        b.fit_matsubara(n, values)
    assert b.last_fit['ill_conditioned'] is True
    assert b.last_fit['condition'] > 1.0


def test_the_pole_guard_flags_an_unconverged_decision(monkeypatch):
    monkeypatch.setattr(fragment_bse, '_guard_lowest',
                        lambda *a, **k: (0.5, None, 1e-4, False))
    flags = {}
    with pytest.warns(RuntimeWarning, match='pole guard'):
        q = fragment_bse.FragmentPartition._q_lowest(
            None, lambda x, om: x, None, np.ones(3), np.zeros((3, 1)),
            np.ones(3), 3, True, 0.1, 0.01, 1e-8, dense_max=0, flags=flags)
    assert q == 0.5
    assert flags['q_lowest_unconverged']['residual'] == 1e-4


def test_a_followed_root_hands_its_branch_to_the_caller():
    out = {}
    w, z = bracketed_root(np.array([0.1, 0.1]), np.array([-1.0, 1.0]), 0.0,
                          0.0, out=out)
    assert set(out) == {'labels', 'largest', 'followed', 'flipped'}
    assert out['flipped'] is False and 0.0 < z <= 1.0
