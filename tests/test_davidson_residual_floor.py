"""The Casida Davidson reaches the residual it is asked for, or says why not.

pyscf's `real_eig` drops a correction whose squared norm is under its `lindep`
before normalizing it. At its fixed 1e-12 the preconditioned residual
r / (d - omega) of a large, widely spread pair space falls under that test
while |r| is still ~1e-7, so the subspace stops growing and the solve ends
unconverged at any tighter tolerance: 2.4e-7 at naphthalene/cc-pVDZ against
the gradient chain's 1e-8, the loop leaving at cycle 81 of 100.

The model is a Casida problem of the RPA shape -- A = D + 2K, B = 2K with
K = V^T V, D orbital-energy differences spread like an all-electron
triple-zeta set -- so A - B = D is positive definite and every root is known
densely. At 500 pairs it is the smallest one found that stalls: four of five
roots stopped at 2.65e-8 against 1e-8.

Run as a script, this file hands itself to pytest and exits with its verdict.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

import src.SingleReference.LinearResponse.davidson as davidson
from src.Base.numerics import Numerics
from src.Base.constants import BSE_DAVIDSON_CONV_TOL, BSE_DAVIDSON_NROOTS
from src.gradients import excited_state, state_following
from src.SingleReference.LinearResponse.davidson import (DavidsonNotConverged,
                                                         solve_casida_davidson)
from src.SingleReference.LinearResponse.block_actions import bse_pair_diagonal
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)
from tests.isdf_test_grid import g1_numerics

NOCC, NVIR, NAUX = 10, 50, 20


@pytest.fixture()
def model(monkeypatch):
    """(lr, dense roots, calls, apply_AB): the model's solver input with its
    block action patched in, every root of it densely, a list each build of
    the action appends to, and the action itself."""
    rng = np.random.default_rng(7)
    eps = np.concatenate([-np.sort(rng.uniform(0.3, 12.0, NOCC))[::-1],
                          np.sort(rng.uniform(0.05, 3.0, NVIR))])
    diag = bse_pair_diagonal(eps, NOCC)
    n_ov = diag.size
    V = rng.standard_normal((NAUX, n_ov)) / np.sqrt(n_ov) * 0.5
    calls = []

    def apply_AB(z):
        f = z.reshape(len(z), -1)
        v = (2 * (f @ V.T) @ V).reshape(z.shape)
        return diag[None] * z + v, v

    # RPA's screened diagonal is d itself, as the production actions return
    # it; the default preconditioner is now the screened diagonal, which a
    # block action without one refuses.
    apply_AB.screened_diagonal = lambda: diag

    def block_action(lr_solver, nocc, polarizability, W_aux, isdf_factors,
                     spin='singlet', comm=None, tda=False):
        calls.append(nocc)
        return apply_AB, diag

    monkeypatch.setattr(davidson, '_block_action', block_action)
    # (A - B)^1/2 (A + B) (A - B)^1/2 has the eigenvalues omega^2.
    sqrt_d = np.sqrt(diag.ravel())
    m = sqrt_d[:, None] * (np.diag(diag.ravel()) + 4 * V.T @ V) * sqrt_d[None, :]
    dense = np.sqrt(np.linalg.eigvalsh(m))
    return (LinearResponseSolver(eps, spin_mode='restricted'), dense, calls,
            apply_AB)


def casida_residual(apply_AB, omega, X, Y):
    """|[A X + B Y - omega X; B X + A Y + omega Y]| per root."""
    shape = (-1, NOCC, NVIR)
    Ax, Bx = apply_AB(X.T.reshape(shape))
    Ay, By = apply_AB(Y.T.reshape(shape))
    n = len(omega)
    top = (Ax + By).reshape(n, -1) - omega[:, None] * X.T
    bot = (Bx + Ay).reshape(n, -1) + omega[:, None] * Y.T
    return np.sqrt(np.sum(top**2, axis=1) + np.sum(bot**2, axis=1))


def test_the_chain_tolerance_is_reached_below_the_lindep_plateau(model):
    """At the chain's own tolerance every root converges -- no warning -- to the
    dense roots, and the vectors it hands back carry the residual it claims."""
    lr, dense, _, apply_AB = model
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        om, X, Y = solve_casida_davidson(lr, NOCC, numerics=Numerics(bse={'nroots': BSE_DAVIDSON_NROOTS, 'conv_tol': BSE_DAVIDSON_CONV_TOL}))
    # The Casida problem is not Hermitian, so a root is good to its residual,
    # not to its square.
    order = np.argsort(om)
    assert np.abs(om[order] - dense[:BSE_DAVIDSON_NROOTS]).max() < BSE_DAVIDSON_CONV_TOL
    r = casida_residual(apply_AB, om, X, Y)
    assert r.max() <= BSE_DAVIDSON_CONV_TOL * (1 + 1e-6), r


def test_a_tolerance_below_the_round_off_floor_is_refused_before_any_action(model):
    """A residual of one eps times the largest pair energy is the round-off of
    a single block action; asking for it is refused before the action is even
    built, rather than after the cycles have been spent on noise."""
    lr, _, calls, _ = model
    diag = bse_pair_diagonal(lr.eps, NOCC)
    with pytest.raises(ValueError, match='below the residual'):
        solve_casida_davidson(lr, NOCC, numerics=Numerics(bse={'nroots': BSE_DAVIDSON_NROOTS, 'conv_tol': np.finfo(float).eps * diag.max()}))
    assert calls == []
    # At the floor itself the solve goes ahead.
    floor = davidson.check_residual_floor(BSE_DAVIDSON_CONV_TOL, diag)
    solve_casida_davidson(lr, NOCC, numerics=Numerics(bse={'nroots': BSE_DAVIDSON_NROOTS, 'conv_tol': floor}))
    assert calls == [NOCC]


def test_an_unconverged_root_is_kept_and_flagged_by_default(model):
    """Two cycles cannot converge anything: by default the solve returns the
    partial roots with a warning that says the cap stopped them, and the
    record names them (`davidson_flags`: the roots and their residuals
    rebuilt from the returned vectors);
    refuse_unconverged=True refuses (`DavidsonNotConverged`) naming the
    roots, their residuals and conv_tol."""
    lr, _, _, apply_AB = model
    kw = dict(numerics=Numerics(bse={'nroots': BSE_DAVIDSON_NROOTS, 'conv_tol': BSE_DAVIDSON_CONV_TOL, 'max_cycle': 2}))
    t = {}
    with pytest.warns(RuntimeWarning, match='at the cycle cap, max_cycle=2'):
        om, X, Y = solve_casida_davidson(lr, NOCC, timings=t, **kw)
    assert t['davidson_verdict'] == 'unconverged'
    r = casida_residual(apply_AB, om, X, Y)
    assert np.allclose(t['davidson_residuals'], r, rtol=1e-10, atol=0)
    flags = davidson.davidson_flags(t)
    stuck = flags['davidson_not_converged']['roots']
    assert stuck == [k for k, c in enumerate(t['davidson_converged'])
                     if not c] and len(stuck) > 0
    assert flags['davidson_not_converged']['residuals'] == [
        t['davidson_residuals'][k] for k in stuck]
    assert min(flags['davidson_not_converged']['residuals']) > \
        BSE_DAVIDSON_CONV_TOL
    t = {}
    with pytest.raises(DavidsonNotConverged, match='Refused') as refusal:
        solve_casida_davidson(lr, NOCC, timings=t, refuse_unconverged=True,
                              **kw)
    exc = refusal.value
    assert exc.roots == stuck
    assert exc.tol == BSE_DAVIDSON_CONV_TOL
    assert exc.residuals == [t['davidson_residuals'][k] for k in stuck]
    assert f'root {stuck[0]} |r| = {exc.residuals[0]:.2e}' in str(exc)
    assert f'conv_tol = {BSE_DAVIDSON_CONV_TOL:g}' in str(exc)
    assert t['davidson_verdict'] == 'refused'


def test_a_converged_solve_carries_no_flag(model):
    lr = model[0]
    t = {}
    solve_casida_davidson(lr, NOCC, timings=t, numerics=Numerics(
        bse={'nroots': BSE_DAVIDSON_NROOTS,
             'conv_tol': BSE_DAVIDSON_CONV_TOL}))
    assert t['davidson_verdict'] == 'converged'
    assert davidson.davidson_flags(t) == {}


def test_the_gradient_chain_keeps_and_logs_rather_than_refuses(monkeypatch):
    """The chain reads forces off the eigenvectors; a read root left
    unconverged is a numerical event (refuse inputs,
    warn numerics), so its Davidson returns the roots and the chain judges
    them itself, logging any in casida_log: empty on a converged solve."""
    seen = []

    def recording(*args, **kwargs):
        seen.append(kwargs.get('refuse_unconverged'))
        return solve_casida_davidson(*args, **kwargs)

    monkeypatch.setattr(state_following, 'solve_casida_davidson', recording)
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis='cc-pvdz',
                verbose=0)

    def factory(m):
        mf = scf.RHF(m)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-11
        mf.kernel()
        return mf

    chain = excited_state.ExcitedStateChain(
        mol, factory, numerics=g1_numerics(mol, Numerics(bse={'solver': 'davidson', 'nroots': BSE_DAVIDSON_NROOTS})))
    chain.excitation(mol)
    assert seen == [False]
    assert chain.casida_log == []


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
