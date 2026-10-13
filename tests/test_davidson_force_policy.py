"""The Davidson refuses an unconverged root only where a force reads it.

`solve_casida_davidson(refuse_unconverged=True, read_roots=k, read_tol=t)`:
a caller that differentiates the lowest k roots reads only their
eigenvectors, so a root above them left unconverged is warned about and
recorded, and a read root left above conv_tol refuses only if its residual,
recomputed from the returned vectors, exceeds t. Without read_roots and
read_tol every root is refused at conv_tol.

Gated on water/cc-pVDZ BSE (ISDF action), 3 roots at conv_tol 1e-8, stopped
early by max_cycle so the roots are unconverged:
  * the refusal without read_roots raises, and so does a read root above
    read_tol;
  * a read root within read_tol returns, with `davidson_verdict`
    'read_within_tol', every root's flag and the recomputed residuals in
    the timings;
  * the recomputed residual of a converged root is within conv_tol, and a
    converged solve records 'converged' with no residual recomputed;
  * over 2 and 3 simulated ranks the verdict and residuals are rank 0's.
"""
import os
import sys
import copy
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from tests.test_davidson_preconditioner import WATER, _system, lr_of

NROOTS = 3
CONV_TOL = 1e-8
EARLY = 3
SIZES = (2, 3)


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    return _system(WATER)


def solve(s, timings, **kw):
    """One BSE Davidson on water's ISDF action."""
    return solve_casida_davidson(lr_of(s), s['nocc'], nroots=NROOTS,
                                 polarizability='BSE', W_aux=s['W_aux'],
                                 isdf_factors=s['factors'], conv_tol=CONV_TOL,
                                 timings=timings, **kw)


def test_the_old_refusal_is_unchanged(water):
    with pytest.raises(RuntimeError, match='Refused'):
        solve(water, {}, max_cycle=EARLY, refuse_unconverged=True)


def test_a_read_root_above_its_tolerance_is_refused(water):
    t = {}
    with pytest.raises(RuntimeError, match='read_tol'):
        solve(water, t, max_cycle=EARLY, refuse_unconverged=True,
              read_roots=1, read_tol=1e-14)
    assert t['davidson_verdict'] == 'refused'
    assert t['davidson_read_residuals'][0] > 1e-14


def test_a_read_root_within_its_tolerance_is_accepted_and_recorded(water):
    t = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        om, X, Y = solve(water, t, max_cycle=EARLY, refuse_unconverged=True,
                         read_roots=1, read_tol=1.0)
    assert any('unconverged' in str(w.message) for w in caught)
    assert t['davidson_verdict'] == 'read_within_tol'
    assert len(t['davidson_converged']) == NROOTS
    assert not all(t['davidson_converged'])
    (r,) = t['davidson_read_residuals']
    assert CONV_TOL < r < 1.0
    assert om.shape == (NROOTS,)


def test_the_recomputed_residual_is_the_solvers(water):
    """On a converged solve the residual rebuilt from the action lies within
    conv_tol, and the verdict needs none."""
    t = {}
    om, X, Y = solve(water, t, refuse_unconverged=True, read_roots=1,
                     read_tol=CONV_TOL)
    assert t['davidson_verdict'] == 'converged'
    assert t['davidson_read_residuals'] is None
    act, diag = davidson._casida_action(lr_of(water), water['nocc'], 'BSE',
                                        water['W_aux'], water['factors'],
                                        CONV_TOL)
    r = davidson._root_residuals(act, om, X, Y, diag.shape)
    assert np.all(r <= CONV_TOL), r


@pytest.mark.parametrize('size', SIZES)
def test_every_rank_holds_rank_0s_verdict(water, size):
    def one_rank(comm):
        s = dict(water, eps=water['eps'].copy(),
                 factors=tuple(np.array(a, copy=True)
                               for a in water['factors']),
                 W_aux=copy.deepcopy(water['W_aux']))
        t = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solve(s, t, max_cycle=EARLY, refuse_unconverged=True,
                  read_roots=1, read_tol=1.0, comm=comm)
        return t['davidson_verdict'], t['davidson_read_residuals']

    out = run_simulated(one_rank, size)
    assert all(v == out[0] for v in out[1:])
    assert out[0][0] == 'read_within_tol'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
