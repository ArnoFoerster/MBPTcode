"""The folded ADC(2) block solve's corrections and searches, on the density-fitted
operator (ee_ri_fold.RIFoldADC2) of water aug-cc-pVDZ (bent, non-symmetric),
frozen 1, aug-cc-pvdz-ri, 3 singlet roots at tol_omega 1e-8.

  1. the block's correction of an unconverged root is the Jacobi-Davidson one on
     M's diagonal plus the root's fold plus the spread of the operator's model of
     the fold's diagonal (fold_diagonal) about the root's average; an operator
     without the model gets diag(M) + y F y alone.
  2. the block roots are converged: exact products at their own frequencies give
     |lambda - omega| < tol_omega and residuals below the asked one.
  3. the completeness check off the found roots costs at most 50 products (74 when
     each of its two searches also ran the eigensolver's own search off its
     converged vector, with the shifted diagonal correction in both); the states
     it must find are the synthetic hidden ones of test_ee_adc_fold_block (7) and
     test_ee_adc_fold (check_hidden_branch).
  4. with a start (a neighbouring problem's roots, here the cold roots at a moved
     geometry carried through the AO basis) the check runs its seeded search
     alone, the cold solve both searches; the warm roots equal a cold solve's at
     that geometry to 1e-9 Ha.
  5. a cold block first solves A_eff at one frequency from the seeds, loosely (one
     eigensolve labelled 'seeds of A_eff'); a start skips it.
  6. a state that drops below root 1 between two calls and is absent from the
     start (synthetic pieces of test_ee_adc_fold_block, 800 singles, one double,
     nroots 1 and 3) is found by the warm solve, every root equal to the
     supermatrix's to 1e-8 Ha: a configuration last on M's diagonal, with the
     model of the fold's diagonal (the seeded search starts on the model's lowest
     entries), and a block of 50 configurations, without the model (the check runs
     the random-vector walk as cold).

Run: python -m pytest -q tests/test_ee_adc_fold_solve_speed.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.gradients.state_models import HFState
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
from src.Solvers.davidson import jacobi_davidson_precond
from tests.adc_common import ATOMS, NF, make_scf_loop, molecule

MOL = molecule(ATOMS, basis='aug-cc-pvdz')
TOL = 1e-8


def operator():
    hf = HFState(MOL, make_scf_loop)
    hf(np.zeros((MOL.nao, MOL.nao)), 0.0)
    operator.mf = hf.mf
    f = ee_ri_fold.bare_factors(MOL, hf.mf, frozen=NF, auxbasis='aug-cc-pvdz-ri')
    return ee_ri_fold.RIFoldADC2(f, 'singlet')


OP = operator()


def solve(op, **kw):
    """(roots, M and A_eff product columns, image columns) of one block solve."""
    m0, i0 = op.matvecs, op.image_columns
    r = ee_fold.solve_folded(op.pieces(), 3, spin='singlet', tol_omega=TOL,
                             tol_residual=np.sqrt(TOL * 0.1), solver='block', **kw)
    return r, op.matvecs - m0, op.image_columns - i0


COLD, _, _ = solve(OP)


def model_inputs(seed=3):
    """A random 4-vector subspace with its images at 3 nodes, the model's pieces."""
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.standard_normal((OP.n, 4)))[0]
    nodes = np.array([0.30, 0.33, 0.36])
    MV, FV = OP.fold_images(V, nodes)
    Mk, Fk = ee_fold._project(V, MV, FV)
    om0 = np.linalg.eigvalsh(Mk + Fk[1])
    return V, MV, FV, Mk, Fk, nodes, om0


def test_block_correction_on_the_fold_model():
    V, MV, FV, Mk, Fk, nodes, om0 = model_inputs()
    args = (V, MV, FV, Mk, Fk, nodes, om0, 4, 2, 0.1, 0.6, OP.diag, 1e-12, 1e-12)
    om, C, res, folds, conv, new = ee_fold._block_model_roots(*args, OP.fold_diagonal)
    om_b, _, _, folds_b, _, new_b = ee_fold._block_model_roots(*args)
    assert np.array_equal(om, om_b) and np.array_equal(folds, folds_b)
    for r in range(2):
        y = V @ C[r]
        rv = MV @ C[r] + np.tensordot(ee_fold._lagrange(nodes, om[r])[0], FV, 1) @ C[r] \
            - om[r] * y
        f = OP.fold_diagonal(float(om[r]))
        d = OP.diag + folds[r] + f - float(y @ (f * y)) / float(y @ y)
        assert np.allclose(new[r], jacobi_davidson_precond(d)(rv, om[r], y),
                           rtol=1e-12, atol=1e-14)
        assert np.allclose(new_b[r], jacobi_davidson_precond(OP.diag + folds[r])(
            rv, om[r], y), rtol=1e-12, atol=1e-14)
        # the model spreads the fold over the configurations: another correction
        assert np.linalg.norm(new[r] - new_b[r]) > 1e-3 * np.linalg.norm(new_b[r])


def test_block_roots_are_converged():
    assert COLD.converged.all() and set(COLD.loop) == {'block'}
    lam, res, _, _ = ee_fold._exact_roots(OP.pieces(), 'singlet', COLD.omega, COLD.y)
    assert np.max(np.abs(lam - COLD.omega)) < TOL
    assert np.max(res) < np.sqrt(TOL * 0.1)


def test_check_cost():
    seen, beside = [], ee_fold._lowest_beside

    def counted(*a, **k):
        m0 = OP.matvecs
        out = beside(*a, **k)
        seen.append(OP.matvecs - m0)
        return out
    ee_fold._lowest_beside = counted
    try:
        r, _, _ = solve(OP)
    finally:
        ee_fold._lowest_beside = beside
    assert np.max(np.abs(r.omega - COLD.omega)) < 1e-12
    assert len(seen) == 1 and seen[0] <= 50, seen


def searches(op, tag=' check', **kw):
    """(roots, eigensolves whose label ends in tag) of one block solve."""
    labels, solve_ = [], ee_fold._solve

    def counted(mv, diag, report=True, **k):
        labels.append(k.get('label', ''))
        return solve_(mv, diag, report, **k)
    ee_fold._solve = counted
    try:
        r, _, _ = solve(op, **kw)
    finally:
        ee_fold._solve = solve_
    return r, sum(lab.endswith(tag) for lab in labels)


def test_start_runs_the_seeded_search_alone():
    shift = np.zeros((len(ATOMS), 3))
    shift[0, 2] = 0.02
    mol = molecule(ATOMS, shift=shift, basis='aug-cc-pvdz')
    hf = HFState(mol, make_scf_loop)
    hf(np.zeros((mol.nao, mol.nao)), 0.0)
    f = ee_ri_fold.bare_factors(mol, hf.mf, frozen=NF, auxbasis='aug-cc-pvdz-ri')
    op = ee_ri_fold.RIFoldADC2(f, 'singlet')
    # the cold roots carried through the AO basis onto the moved orbitals
    nocc = MOL.nelectron // 2
    C0 = np.asarray(operator.mf.mo_coeff)
    C1 = np.asarray(hf.mf.mo_coeff)
    S = mol.intor('int1e_ovlp')
    Po = C1[:, NF:nocc].T @ S @ C0[:, NF:nocc]
    Pv = C0[:, nocc:nocc + OP.nv].T @ S @ C1[:, nocc:nocc + op.nv]
    start = np.column_stack([(Po @ COLD.y[:, r].reshape(OP.no, OP.nv) @ Pv).ravel()
                             for r in range(COLD.y.shape[1])])
    cold, n_cold = searches(op)
    warm, n_warm = searches(op, start=start)
    assert (n_cold, n_warm) == (2, 1), (n_cold, n_warm)
    assert np.max(np.abs(warm.omega - cold.omega)) < 1e-9


def test_cold_block_seeds_on_aeff():
    r, n_cold = searches(OP, tag=' seeds of A_eff')
    _, n_warm = searches(OP, tag=' seeds of A_eff', start=np.asarray(COLD.y))
    assert (n_cold, n_warm) == (1, 0), (n_cold, n_warm)
    assert np.max(np.abs(r.omega - COLD.omega)) < 1e-12


def crossing(kind, c, model):
    """Synthetic fold pieces (test_ee_adc_fold_block) of 800 singles and one
    double, coupling c, and their exact roots: 'alone', a configuration last on
    M's diagonal (3.0 Ha) coupled to a double at 3.1 Ha; 'coupled', a block of 50
    configurations (1.5-2.0 Ha) coupled to a double at 2.2 Ha by a random unit
    row. model: pieces['fold_diagonal'] is the exact fold diagonal."""
    from tests.test_ee_adc_fold_block import supermatrix_roots, synthetic_pieces
    n, rng = 800, np.random.default_rng(7)
    m = np.diag(0.5 + 0.5 * np.arange(n) / n)
    C = np.zeros((1, n))
    if kind == 'alone':
        m[-1, -1], C[0, -1], D = 3.0, 1.0, 3.1
    else:
        B = rng.standard_normal((50, 50)) * 2e-3
        m[-50:, -50:] = np.diag(1.5 + 0.5 * np.arange(50) / 50) + 0.5 * (B + B.T)
        C[0, -50:] = rng.standard_normal(50)
        C, D = C / np.linalg.norm(C), 2.2
    p = synthetic_pieces(m, c * C, [D])
    if model:
        p['fold_diagonal'] = lambda w: (c * C[0]) ** 2 / (w - D)
    return p, supermatrix_roots(m, c * C, np.array([D]))


def test_warm_check_finds_a_crossing_state():
    for kind, c0, c1, model in (('alone', 0.5, 2.75, True), ('coupled', 0.6, 2.2, False)):
        for nroots in (1, 3):
            p0, w0 = crossing(kind, c0, model)
            r0 = ee_fold.solve_folded(p0, nroots, spin='singlet', dense_limit=0,
                                      solver='block', tol_omega=1e-10)
            assert np.max(np.abs(r0.omega[:nroots] - w0[:nroots])) < 1e-8
            # the planted state drops below root 1 and the cold roots start
            p1, w1 = crossing(kind, c1, model)
            assert w1[0] < w0[0] - 0.1
            r1 = ee_fold.solve_folded(p1, nroots, spin='singlet', dense_limit=0,
                                      solver='block', tol_omega=1e-10,
                                      start=np.asarray(r0.y))
            d = np.max(np.abs(r1.omega[:nroots] - w1[:nroots]))
            assert d < 1e-8 and r1.converged.all(), (kind, nroots, d)
