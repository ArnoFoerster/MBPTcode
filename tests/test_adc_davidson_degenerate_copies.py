"""The symmetric Davidson (`solve_symmetric`) on a degenerate ADC spectrum:
the lowest k roots, every copy, or a False convergence flag.

H2O/sto-3g spin-orbital ADC(3) IP block (the 1h rows and the 2h1p rows of the
dense supermatrix, dim 194): the lowest six eigenvalues are three Kramers
pairs, -44.317007 (x2), -44.263821 (x2) and -22.009281 (x2) Ha; the next level
is -21.929149 Ha. The case comes from a degenerate IP spectrum on which a block
Davidson returned the next level as the sixth root, the second copy of the
highest pair never reached.

Checks:
  1. default start: the dense lowest six, converged;
  2. a start with no weight on one copy of the -22.009 pair: the dense lowest
     six;
  3. two uncoupled copies of the block, started inside one: each level is
     four-fold, the dense lowest six are four copies of -44.317 and two of
     -44.264. Floating point never leaves the started copy, so the Davidson
     iteration alone returns that copy's own lowest six, every root
     converged; solve_symmetric's lowest-k completion (the solve on the
     complement of the converged roots) finds the missing copies (before the
     completion this returned 22.25 Ha too high a sixth root, flagged
     converged);
  4. a solve stopped short of tol reports conv False and warns;
  5. the doubled block from the default start, seven roots at 1e-10 (a
     degenerate level split by the k-th root): the completion certifies the
     dense lowest seven instead of refusing a correct solve;
  6. a completion that cannot finish (two cycles for its complement solve)
     refuses: conv all False and a warning naming the certificate.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Solvers.davidson import solve_symmetric
from src.SingleReference.ADC.base import ADCSolverUnrestricted

K = 6
TOL = 1e-10


@pytest.fixture(scope='module')
def water_ip():
    """(diagonal, dense matrix) of the H2O/sto-3g ADC(3) IP block."""
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.96 0 0', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    s = ADCSolverUnrestricted(mf, level='adc3', matrix_free=False)
    n = s.norb + s.dimensions(s.nocc)['n2h1p']
    H = s.build_supermatrix(s.nocc)[:n, :n]
    H = 0.5 * (H + H.T)
    return np.diag(H).copy(), H


def unit_start(dg):
    """The solver's own default start: unit vectors on the 2k+4 lowest diagonal entries."""
    cols = np.argsort(dg)[:2 * K + 4]
    E = np.zeros((dg.size, cols.size))
    E[cols, np.arange(cols.size)] = 1.0
    return E


def test_the_dense_spectrum_has_three_pairs_in_the_lowest_six(water_ip):
    w = np.linalg.eigvalsh(water_ip[1])
    assert np.abs(w[0:6:2] - w[1:6:2]).max() < 1e-10
    assert np.allclose(w[4:6], -22.009281, atol=1e-6)
    assert w[6] > w[5] + 0.07


def test_default_start_returns_the_lowest_six(water_ip):
    dg, H = water_ip
    w, X, conv = solve_symmetric(lambda v: H @ v, dg, nroots=K, tol_residual=TOL)
    assert conv.all()
    assert np.abs(w - np.linalg.eigvalsh(H)[:K]).max() < 1e-8
    assert np.linalg.norm(H @ X - X * w, axis=0).max() < TOL


def test_a_start_blind_to_one_copy_returns_the_lowest_six(water_ip):
    dg, H = water_ip
    wd, Z = np.linalg.eigh(H)
    E = unit_start(dg)
    pair = Z[:, 4:6]
    u = np.linalg.svd(pair.T @ E)[0]
    blind = pair @ u[:, -1]                          # the copy the start sees least
    v0 = E - np.outer(blind, blind @ E)
    w, X, conv = solve_symmetric(lambda v: H @ v, dg, nroots=K, x0=v0, tol_residual=TOL)
    assert conv.all()
    assert np.abs(w - wd[:K]).max() < 1e-8, w
    assert np.linalg.norm(H @ X - X * w, axis=0).max() < TOL


def test_a_start_inside_one_of_two_uncoupled_copies_returns_both(water_ip):
    dg, H = water_ip
    n = H.shape[0]
    H2 = np.block([[H, np.zeros_like(H)], [np.zeros_like(H), H]])
    dg2 = np.concatenate([dg, dg])
    v0 = np.vstack([unit_start(dg), np.zeros((n, 2 * K + 4))])
    w, X, conv = solve_symmetric(lambda v: H2 @ v, dg2, nroots=K, x0=v0, tol_residual=TOL)
    assert conv.all()
    assert np.abs(w - np.linalg.eigvalsh(H2)[:K]).max() < 1e-8, w
    assert np.linalg.norm(H2 @ X - X * w, axis=0).max() < 10 * TOL
    assert np.abs(X.T @ X - np.eye(K)).max() < 1e-10


def test_a_solve_left_above_tol_says_so(water_ip):
    dg, H = water_ip
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, _, conv = solve_symmetric(lambda v: H @ v, dg, nroots=K,
                                     tol_residual=TOL, max_cycle=2)
    assert not conv.any()
    assert any('unconverged' in str(r.message) for r in rec)



def test_a_split_degenerate_level_is_certified_at_tight_tol(water_ip):
    dg, H = water_ip
    H2 = np.block([[H, np.zeros_like(H)], [np.zeros_like(H), H]])
    dg2 = np.concatenate([dg, dg])
    w, X, conv = solve_symmetric(lambda v: H2 @ v, dg2, nroots=7, tol_residual=TOL)
    assert conv.all()
    assert np.abs(w - np.linalg.eigvalsh(H2)[:7]).max() < 1e-8, w


def test_a_completion_that_cannot_finish_refuses(water_ip):
    dg, H = water_ip
    x0 = np.linalg.eigh(H)[1][:, :3]           # converged at once; the complement gets 2 cycles
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, _, conv = solve_symmetric(lambda v: H @ v, dg, nroots=3, x0=x0,
                                     tol_residual=TOL, max_cycle=2)
    assert not conv.any()
    assert any('not certified' in str(r.message) for r in rec)

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
