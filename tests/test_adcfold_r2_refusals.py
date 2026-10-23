"""Refusals and numerical flags of the folded ADC(2) line (refuse inputs, warn
numerics).
Each refusal test raises where the code used to hang,
returned a number or went on; each flag test shows the warning and the named
flag (ee_fold.FoldResult.flags) where the code refused before: water 6-31G (bent,
non-symmetric) unless named.

  - a raise inside on_root on rank 0 is raised on every rank (2 and 3
        simulated ranks), not left in the other ranks' broadcast
  - an inner eigensolve that does not converge leaves the root unconverged,
        flagged 'roots_not_converged'
  - nroots above the channel is refused (an input); the count's missing
        root is returned short, flagged 'roots_missing'; a root index the
        solve did not return is NaN, flagged 'adc_root_missing', in the call,
        the force and the loop (which stops, 'solvation_loop_stopped'), and a
        walk rejects that step; the upfolded BSE
        flags an unconverged Davidson and returns a complex dense root as NaN,
        flagged; an invalid MBPT_STREAM_GB is refused
  - a model branch that does not reach tol ends the model's roots there,
        flagged 'model_branch_unsolved', not skipped
  - the photophysics driver keys a record by the xyz file's content
  - a force whose root is not the call's is flagged (follow='index')
  - the walker retries a StepFailed and lets any other exception out; a
        solvation loop that did not converge is no exception any more (C1)
  - state_specific='equilibrium' without eps_static is refused before the
        solve
  - the solvation loop refuses max_cycle < 1
  - extend_orthonormal refuses missing thresholds
"""
import importlib
import json
import os
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.environments import solvation_loop as sl
from src.Base.numerics.numerics import Numerics
from src.environments.solvent_screening import SolventScreening
from src.Base.step_failed import StepFailed
from src.Base.utils.linear_algebra.orthogonalization import extend_orthonormal
from src.Base.utils.parallel.mpi_grid import distributed
from src.Base.utils.parallel.simulated_comm import simulated_world
from src.gradients import df_correlation_force_tiled as tcf
from src.SingleReference.ADC.eeADC import ee_driver, ee_fold, ee_spin_blocks
from src.gradients.adc import ee_gradient
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
from src.gradients.adc import ee_state_model
from src.SingleReference.ADC.eeADC.ee_follow import FollowingWarning, root_missing
from src.SingleReference.BSE import bse_upfolded as B
from tests.adc_common import ATOMS, df_rhf_converged, molecule
from tests.test_ee_adc_fold import _synthetic_pieces
from tests.test_solvation_refusals import _args, adc, driver
from tests.test_solvation_refusals import toluene as sl_toluene

# the module: src.properties exports the function `optimize` under its name
optimize = importlib.import_module('src.properties.optimize')


# ------------------------------------------------------------------------------

def _on_ranks(size, fn):
    """on_root(fn, [(2,)], comm) on `size` simulated ranks in threads: each
    rank's (result, exception). A rank left waiting after 60 s is freed by
    aborting the barrier (it then holds a BrokenBarrierError)."""
    comms = simulated_world(size)
    out = [(None, None)] * size

    def worker(r):
        try:
            with distributed(comms[r]):
                out[r] = (tcf.on_root(fn, [(2,)], comms[r]), None)
        except BaseException as exc:            # noqa: BLE001 -- the verdict
            out[r] = (None, exc)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(60)
    if any(th.is_alive() for th in threads):
        comms[0]._world.barrier.abort()
        for th in threads:
            th.join(10)
    return out


@pytest.mark.parametrize('size', (2, 3))
def test_on_root_raises_on_every_rank(size):
    def fails():
        raise RuntimeError('CPHF not converged (planted)')
    out = _on_ranks(size, fails)
    assert all(isinstance(exc, RuntimeError) and 'CPHF not converged' in str(exc)
               for _, exc in out), out


@pytest.mark.parametrize('size', (2, 3))
def test_on_root_returns_rank_0s_arrays(size):
    out = _on_ranks(size, lambda: (np.array([1.0, 2.0]),))
    assert all(exc is None and np.array_equal(res[0], [1.0, 2.0]) for res, exc in out)


# ------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def water_pieces():
    mf = df_rhf_converged(molecule(ATOMS))
    from src.SingleReference.ADC.eeADC import ee_ri_fold
    from src.Base.pyscf.pyscf_interface import DFIntegrals, get_orbital_energies
    no = mf.mol.nelectron // 2
    F = ee_ri_fold.RIFactors.from_full(np.asarray(get_orbital_energies(mf), float),
                                       DFIntegrals.from_scf(mf.mol, mf).B_aa, no)
    return ee_ri_fold.RIFoldADC2(F, 'singlet').pieces()


def _inner_fails(monkeypatch):
    real = ee_fold._solve

    def stub(*a, **k):
        e, X, conv = real(*a, **k)
        return e, X, np.zeros(np.size(conv), bool)
    monkeypatch.setattr(ee_fold, '_solve', stub)


def test_an_unconverged_inner_eigensolve_leaves_the_root_unconverged(water_pieces,
                                                                     monkeypatch):
    _inner_fails(monkeypatch)
    with pytest.warns(RuntimeWarning, match='not converged'):
        res = ee_fold.solve_folded(water_pieces, 2, spin='singlet', solver='newton',
                                   dense_limit=0)
    assert not res.converged.any(), res.converged
    assert 'roots_not_converged' in [f['flag'] for f in res.flags]


# ----------------------------------------------------------------------------

def test_nroots_above_the_channel_is_refused():
    mf = scf.RHF(gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
                 ).density_fit().run()
    with pytest.raises(ValueError, match='fewer than nroots'):
        solve_ee_adc(mf, level='adc2', nroots=3, spin='singlet', fold=True)


def _close_pair():
    """test_ee_adc_fold's pair 1e-8 Ha apart at the default tol_omega: the
    count's index solve lands on the found root."""
    D = np.array([1.0, 1.4])
    C = np.array([[0.3, 0.3, 0.0], [0.3, -0.54, 0.0]])
    s = np.sum(C ** 2 / (D[:, None] - 0.5), axis=0)
    return _synthetic_pieces(np.array([0.5 + s[0], 0.5 + s[1] + 1e-8, 1.5]), C, D)


def test_a_missing_root_is_flagged():
    with pytest.warns(RuntimeWarning, match='missing from the result'):
        res = ee_fold.solve_folded(_close_pair(), 2, spin='singlet', solver='newton')
    assert res.omega.size == 1
    assert 'roots_missing' in [f['flag'] for f in res.flags]
    # a caller that takes root 1 has no number for it: flagged
    flags = []
    with pytest.warns(FollowingWarning, match='root 1 asked for'):
        assert root_missing(1, res.omega.size, 'test', flags)
    assert [f['flag'] for f in flags] == ['adc_root_missing']


def _one_root(res):
    """The fold's result cut to its lowest root (a solve short of root 1)."""
    res.omega, res.t1 = res.omega[:1], res.t1[:1]
    res.y = np.asarray(res.y)[..., :1]
    res.converged = np.atleast_1d(res.converged)[:1]
    res.delta_density_relaxed = list(res.delta_density_relaxed)[:1]
    return res


def _short_solve(monkeypatch):
    real = ee_state_model.solve_ee_adc

    def solve(*a, **k):
        out = real(*a, **k)
        return out[:-1] + (_one_root(out[-1]),)
    monkeypatch.setattr(ee_state_model, 'solve_ee_adc', solve)


def test_root_index_beyond_nroots_is_refused():
    with pytest.raises(ValueError, match='root 3 of nroots=3'):
        adc(molecule(ATOMS), root=3, nroots=3)


def test_a_missing_root_is_nan_and_flagged_in_the_call_and_the_loop(monkeypatch):
    mol = molecule(ATOMS)
    _short_solve(monkeypatch)
    state = adc(mol, root=1, nroots=2, follow='index')
    with pytest.warns(FollowingWarning, match='root 1 asked for'):
        r = state(np.zeros((mol.nao, mol.nao)), 0.0)
    assert np.isnan(r.energy) and np.isnan(r.dm_eff).all()
    assert 'adc_root_missing' in [f['flag'] for f in r.info['flags']]
    assert r.info['converged'] is False
    # the loop stops at the gas-phase call: NaN, flagged, and its force NaN
    field = sl_toluene(mol)
    with pytest.warns(sl.SolvationNotConverged, match='stopped'):
        loop = sl.equilibrate(adc(mol, root=1, nroots=2, follow='index'), field)
    assert np.isnan(loop.energy) and not loop.converged and loop.history == []
    assert {'adc_root_missing', 'solvation_loop_stopped'} <= {
        f['flag'] for f in loop.flags}
    F = sl.force(state, field, loop)
    assert F.shape == (mol.natm, 3) and np.isnan(F).all()


def test_a_missing_root_is_a_nan_force_flagged(monkeypatch):
    mol = molecule(ATOMS)
    _patched_root(monkeypatch, lambda out: dict(
        out, omega=out['omega'][:1], y=out['y'][:, :1], t1=out['t1'][:1],
        converged=np.atleast_1d(out['converged'])[:1]))
    with pytest.warns(FollowingWarning, match='root 1 asked for'):
        F, info = adc(mol, root=1, nroots=2, follow='index').force(
            np.zeros((mol.nao, mol.nao)), 0.0)
    assert F.shape == (mol.natm, 3) and np.isnan(F).all()
    assert np.isnan(info['energy']) and np.isnan(info['dm_eff']).all()
    assert 'adc_root_missing' in [f['flag'] for f in info['flags']]


def test_a_clean_fold_carries_no_flags(water_pieces):
    res = ee_fold.solve_folded(water_pieces, 2, spin='singlet', solver='newton',
                               dense_limit=0)
    assert res.converged.all() and res.flags == []


def test_upfolded_bse_flags_an_unconverged_davidson(monkeypatch):
    rng = np.random.default_rng(0)
    A = np.diag(np.linspace(0.2, 2.0, 200)) + 1e-3 * rng.standard_normal((200, 200))
    flags = []
    with pytest.warns(RuntimeWarning, match='not converged'):
        B.davidson_nonsymmetric(lambda v: A @ v, np.diag(A).copy(), nroots=3,
                                max_cycle=1, flags=flags)
    assert [f['flag'] for f in flags] == ['upfolded_bse_not_converged']
    H = np.diag([1.0, 1.0, 3.0, 4.0])
    H[0, 1], H[1, 0] = 0.5, -0.5                 # the lowest pair: 1 +- 0.5i
    monkeypatch.setattr(B, 'build_hamiltonian', lambda *a: H)
    flags = []
    with pytest.warns(RuntimeWarning, match='complex'):
        w, v = B.solve_dense(None, None, None, 1, 1, 'singlet', nroots=3,
                             flags=flags)
    # the pair has no real root: NaN, flagged; the real root 3 is kept
    assert np.isnan(w[:2]).all() and np.isnan(v[:, :2]).all() and w[2] == 3.0
    assert [f['flag'] for f in flags] == ['upfolded_bse_complex_roots']


def test_an_invalid_stream_budget_is_refused():
    with pytest.raises(ValueError, match="'abc'"):
        ee_spin_blocks._stream_gb('abc')


def test_a_model_branch_short_of_tol_is_flagged():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((6, 6))
    H0 = 0.5 * (A + A.T)
    om0 = float(np.linalg.eigvalsh(H0)[0]) - 0.3
    flags = []
    token = ee_fold._FLAGS.set(flags)
    try:
        with pytest.warns(RuntimeWarning, match='not solved'):
            oms, C = ee_fold._model_roots(H0, -np.eye(6), om0, 6, 1e9, 1e-17)
    finally:
        ee_fold._FLAGS.reset(token)
    assert oms.size < 6 and C.shape[1] == oms.size
    assert [f['flag'] for f in flags] == ['model_branch_unsolved']


# ------------------------------------------------------------------------------

def test_driver_keys_the_geometry_not_the_path(tmp_path):
    mod = driver()
    first = _args(tmp_path)
    mod._save(mod._start_record(first), first.out)
    with open(first.xyz, 'w') as fh:
        fh.write('3\n\nO 0 0 0\nH 0 0.80 0.59\nH 0 -0.76 0.59\n')
    with pytest.raises(SystemExit, match='xyz_sha256'):
        mod._start_record(first)


# ------------------------------------------------------------------------------

def _patched_root(monkeypatch, change):
    real = ee_gradient.solve_root

    def solve_root(*a, **k):
        op, rf, out = real(*a, **k)
        return op, rf, change(dict(out))
    monkeypatch.setattr(ee_gradient, 'solve_root', solve_root)


def test_a_force_on_reordered_roots_is_flagged(monkeypatch):
    mol = molecule(ATOMS)
    h = np.zeros((mol.nao, mol.nao))
    state = adc(mol, follow='index')
    state(h, 0.0)

    def swap(out):
        order = [1, 0] + list(range(2, out['omega'].size))
        out.update(omega=out['omega'][order], y=out['y'][:, order],
                   t1=out['t1'][order], converged=out['converged'][order])
        return out
    _patched_root(monkeypatch, swap)
    with pytest.warns(FollowingWarning, match='order the roots differently'):
        F, info = state.force(h, 0.0)
    assert np.isfinite(F).all()
    assert 'force_root_order_differs' in [f['flag'] for f in info['flags']]


def test_a_force_of_another_omega_is_flagged(monkeypatch):
    mol = molecule(ATOMS)
    h = np.zeros((mol.nao, mol.nao))
    state = adc(mol, follow='index')
    state(h, 0.0)
    _patched_root(monkeypatch, lambda out: dict(out, omega=out['omega'] + 1e-6))
    with pytest.warns(FollowingWarning, match='same field'):
        F, info = state.force(h, 0.0)
    assert np.isfinite(F).all()
    assert 'force_omega_mismatch' in [f['flag'] for f in info['flags']]


# ------------------------------------------------------------------------------

def _walk(monkeypatch, exc):
    """_trust_region_walk on a stub surface whose every trial step raises exc."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.80', basis='sto-3g', verbose=0)
    calls = []

    def evaluate(surface, m):
        calls.append(1)
        if len(calls) > 1:
            raise exc
        return np.array([[0.0, 0.0, -0.05], [0.0, 0.0, 0.05]]), -1.0, {}
    monkeypatch.setattr(optimize, 'evaluate', evaluate)
    return optimize._trust_region_walk(None, mol, 5, 0.3, 0.5, 1e-3, None, None, 0,
                                       False)


def test_the_walker_retries_a_step_failed(monkeypatch):
    _, info = _walk(monkeypatch, StepFailed('planted'))
    assert not info['converged'] and 'trust radius collapsed' in info['status']


def test_the_walker_rejects_a_nan_evaluation(monkeypatch):
    """A surface that returns NaN (a missing root, flagged there) is a rejected
    step, retried shorter, not a walk on NaN."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.80', basis='sto-3g', verbose=0)
    calls = []

    def evaluate(surface, m):
        calls.append(1)
        if len(calls) > 1:
            return np.full((2, 3), np.nan), np.nan, {}
        return np.array([[0.0, 0.0, -0.05], [0.0, 0.0, 0.05]]), -1.0, {}
    monkeypatch.setattr(optimize, 'evaluate', evaluate)
    _, info = optimize._trust_region_walk(None, mol, 5, 0.3, 0.5, 1e-3, None, None,
                                          0, False)
    assert not info['converged'] and 'trust radius collapsed' in info['status']
    assert len(calls) > 2


def test_the_walker_lets_other_exceptions_out(monkeypatch):
    with pytest.raises(ValueError, match='planted'):
        _walk(monkeypatch, ValueError('planted'))


def test_driver_records_the_flags_at_a_walks_last_geometry(tmp_path, monkeypatch):
    """An unconverged loop on the way is no exception: the surface logs its
    flag and the minimum's record names it with 'walk_not_converged'."""
    mod = driver()
    mol = gto.M(atom='H 0 0 0; H 0 0 0.80', basis='sto-3g', verbose=0)

    class Surface:
        flags = [dict(coords=mol.atom_coords().tolist(),
                      flags=['solvation_loop_not_converged'], messages=['planted'])]

        def flags_at(self, m):
            return ['solvation_loop_not_converged']

        def total_energy(self, m):
            return -1.0
    monkeypatch.setattr(mod, 'relax', lambda *a, **k: (
        mol, dict(converged=False, cycles=3, opt_grad_max=1e-2, status='planted')))
    args = SimpleNamespace(maxiter=3, saddle_check=True, out=str(tmp_path / 'r.json'),
                           method='adc2', refreeze=False)
    _, entry = mod._minimum(args, Surface(), mol, 0, 'S0')
    assert entry['flags'] == ['solvation_loop_not_converged', 'walk_not_converged']
    assert 'saddle_check' not in entry


# ------------------------------------------------------------------------------

def test_equilibrium_without_eps_static_is_refused_before_the_solve(monkeypatch):
    mol = molecule(ATOMS)
    mf = df_rhf_converged(mol)

    def ran(*a, **k):
        raise AssertionError('the solve ran')
    monkeypatch.setattr(ee_driver, 'solve_folded_ri', ran)
    with pytest.raises(ValueError, match='eps_static'):
        solve_ee_adc(mf, level='adc2', nroots=2, spin='singlet', fold=True,
                     environment=SolventScreening(mol, eps=2.4),
                     state_specific='equilibrium')


def test_loop_refuses_max_cycle_below_one():
    with pytest.raises(ValueError, match='max_cycle'):
        sl.equilibrate(None, None, numerics=Numerics(solvent={'loop_max_cycle': 0}))


# ------------------------------------------------------------------------------

def test_extend_orthonormal_refuses_missing_thresholds():
    with pytest.raises(ValueError, match='rtol or atol'):
        extend_orthonormal(np.empty((3, 0)), np.eye(3))
