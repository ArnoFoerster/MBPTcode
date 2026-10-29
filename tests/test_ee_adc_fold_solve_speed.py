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
