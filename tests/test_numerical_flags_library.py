"""Warn-and-continue sites name their event where the result is kept
(refuse inputs, warn numerics).

Each site keeps warning and carries on; the event is also handed to the
object the caller records:

- the BSE Davidson's unit-vector guess cut inside a degenerate cluster:
  `counters['guess_truncated']` -> timings 'davidson_guess_truncated';
- `sigma_sop` outside the compression condition: `out['compression_excluded']`;
- complex EOM-CC eigenvalues: a `ComplexEOMEigenvalues` warning (no longer a
  print) and `EOMResult.complex_eigenvalues` / `max_imag`;
- unconverged EOM-CC Davidson roots: `UnconvergedEOMRoots` and
  `EOMResult.conv` / `converged`;
- a CC amplitude or Lambda solve out of iterations: `CCNotConverged` and
  `out['converged']` (EOMCC: `t_converged`, `lambda_converged`);
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

from src.Base.pyscf import pyscf_interface
from src.Base.utils.grids import matsubara
from src.Base.utils.grids.matsubara import IRBasis
from src.Base.numerics.constants import ROOT_FOLLOW_MARGIN_MIN
from src.gradients import state_identity
from src.gradients.gwbse import fragment_bse
from src.SingleReference.CC import eom
from src.SingleReference.GW.sum_over_poles import sigma_sop
from src.SingleReference.LinearResponse.davidson import _davidson_counters
from src.SingleReference.LinearResponse.davidson_shared import guess_indices
from src.Solvers.qp_equation import bracketed_root


def h2(**kw):
    return gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0, **kw)


def test_a_truncated_davidson_guess_is_counted():
    assert _davidson_counters()['guess_truncated'] is False
    counters = _davidson_counters()
    with pytest.warns(RuntimeWarning, match='guess truncated'):
        order = guess_indices(np.zeros((10, 10)), 10, counters=counters)
    assert len(order) == 40
    assert counters['guess_truncated'] is True
    # a gapped spectrum is not cut and not flagged
    clean = _davidson_counters()
    guess_indices(np.arange(100.0).reshape(10, 10), 10, counters=clean)
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
    solver = eom.EOMCC(mf, level='ccsdt')

    def complex_roots(aop, guess, precond, nroots=1, **kw):
        es = np.array([0.9 + 2e-3j, 1.1 - 2e-3j, 1.3, 1.5])[:nroots]
        return [True] * nroots, es, [guess[0]] * nroots
    monkeypatch.setattr(eom.lib, 'davidson_nosym1', complex_roots)
    with pytest.warns(eom.ComplexEOMEigenvalues, match=r'\|Im\| up to 2'):
        result = solver.kernel(sector='ee', nroots=2)
    assert result.complex_eigenvalues and result.max_imag == pytest.approx(2e-3)


def test_unconverged_eom_roots_warn_and_flag(monkeypatch):
    mf = scf.RHF(h2()).run()
    solver = eom.EOMCC(mf, level='ccsdt')

    def half_converged(aop, guess, precond, nroots=1, **kw):
        return [True, False][:nroots], np.array([0.9, 1.1])[:nroots], guess[:nroots]
    monkeypatch.setattr(eom.lib, 'davidson_nosym1', half_converged)
    with pytest.warns(eom.UnconvergedEOMRoots, match='1 of 2 roots unconverged'):
        result = solver.kernel(sector='ee', nroots=2)
    assert result.converged is False and list(result.conv) == [True, False]


def test_cc_iterations_warn_and_flag_when_out_of_cycles():
    from src.SingleReference.CC import amplitudes, solver as lambda_solver
    from src.SingleReference.CC.fixed_point import CCNotConverged
    from src.SingleReference.CC.integrals import (build_spinorbital_integrals_from_mf,
                                                  energy_denominators)
    ints = build_spinorbital_integrals_from_mf(scf.RHF(h2()).run())
    no, nv = ints['nocc'], ints['nvir']
    o, v = slice(None, no), slice(no, None)
    dens = energy_denominators(ints['fock'], no, nv)
    t = (np.zeros((nv, no)), np.zeros((nv, nv, no, no)), np.zeros((nv,) * 3 + (no,) * 3))
    out = {}
    with pytest.warns(CCNotConverged, match='CCSDT iterations did not converge'):
        t = amplitudes.kernel(*t, ints['fock'], ints['g'], o, v, *dens, ints['hf_energy'],
                              max_iter=1, out=out)
    assert out == {'converged': False, 'n_iter': 1}
    out = {}
    with pytest.warns(CCNotConverged, match='Lambda-CCSDT'):
        lambda_solver.solve_lambda_ccsdt(*t, ints['fock'], ints['g'], o, v, *dens,
                                         max_iter=1, verbose=False, out=out)
    assert out['converged'] is False


def test_ccsdt_density_refuses_uhf_and_eom_refuses_ccsd():
    from src.SingleReference.CC.pipeline import compute_ccsdt_density_matrix
    with pytest.raises(ValueError, match='no UHF path'):
        compute_ccsdt_density_matrix(scf.UHF(h2()).run())
    with pytest.raises(ValueError, match='pyscf'):
        eom.EOMCC(scf.RHF(h2()).run(), level='ccsd')


def test_an_ambiguous_orbital_track_is_named():
    mf = scf.RHF(h2()).run()
    mol = mf.mol
    fingerprint = {'mol': mol, 'localized': np.eye(2),
                   'weights': np.array([0.5, 0.5]), 'window': (0, 2),
                   'occupied': True}
    out = {}
    with pytest.warns(RuntimeWarning, match='ambiguous'):
        state_identity.track_orbital(fingerprint, mf, mo_coeff=np.eye(2),
                                     candidates=[0, 1], out=out)
    assert sorted(out['ambiguous']['candidates']) == [0, 1]
    assert out["ambiguous"]["margin"] < ROOT_FOLLOW_MARGIN_MIN


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


def test_a_short_resolvent_solve_keeps_its_iterate_and_is_flagged():
    """A batched conjugate-gradient solve stopped by its iteration cap
    returns its last iterate, warns and flags; it raised."""
    a = np.diag([1.0, 2.0, 3.0]) + 0.1
    flags = {}
    with pytest.warns(RuntimeWarning, match='resolvent solve did not'):
        x = fragment_bse._batched_cg(lambda v: a @ v, np.ones((3, 1)),
                                     lambda v: v, 1e-14, maxiter=1,
                                     flags=flags)
    assert x.shape == (3, 1) and np.abs(x).max() > 0.0
    assert flags['resolvent_not_converged']['columns'] == 1


def test_a_short_feshbach_root_is_flagged():
    """The dense oracle's Newton stopped by its cap keeps its last iterate,
    warns and flags; it returned silently."""
    rng = np.random.default_rng(3)
    k = rng.normal(size=(6, 6))
    k = k @ k.T + 6.0 * np.eye(6)
    oracle = fragment_bse.FeshbachOracle(k, np.eye(6)[:, :2])
    flags = {}
    with pytest.warns(RuntimeWarning, match='Feshbach root'):
        om, c, z = oracle.root(float(np.diag(k)[0]), maxiter=1, flags=flags)
    assert flags['feshbach_root_not_converged']['omega'] == om


def test_a_followed_root_hands_its_branch_to_the_caller():
    out = {}
    w, z = bracketed_root(np.array([0.1, 0.1]), np.array([-1.0, 1.0]), 0.0,
                          0.0, out=out)
    assert set(out) == {'labels', 'largest', 'followed', 'flipped', 'no_root'}
    assert out['no_root'] is False
    assert out['flipped'] is False and 0.0 < z <= 1.0
