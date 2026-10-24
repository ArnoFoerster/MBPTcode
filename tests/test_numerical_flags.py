"""A numerical fallback warns with its own class and leaves a named flag where the caller
collects (`Base.numerical_flags`): the screening factorization that falls back from
Cholesky to LU, and the vector-fitting least squares that falls back from gelsd. Outside
a `collecting` block only the warning is left.
"""
import numpy as np
import pytest

from src.Base.numerical_flags import collecting, flag
from src.SingleReference.GW import contour_deformation as cd
from src.SingleReference.GW import sum_over_poles as sop


def test_a_flag_reaches_the_innermost_collecting_block_only():
    outer, inner = [], []
    with collecting(outer):
        with collecting(inner), pytest.warns(RuntimeWarning, match='planted'):
            flag('planted', 'a planted shortfall', RuntimeWarning)
    assert inner == [{'flag': 'planted', 'message': 'a planted shortfall'}]
    assert outer == []
    with pytest.warns(RuntimeWarning):
        flag('planted', 'nobody collects this', RuntimeWarning)


def test_a_screening_not_positive_definite_is_solved_by_lu_and_flagged():
    eps_mat = np.diag([2.0, -1.0, 3.0])
    flags = []
    with collecting(flags), pytest.warns(cd.ScreeningNotPositive):
        solve = cd.screening_solver(eps_mat)
    b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(eps_mat @ solve(b), b)
    assert [f['flag'] for f in flags] == ['screening_not_positive_definite']


def test_a_failed_least_squares_falls_back_and_is_flagged(monkeypatch):
    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError('SVD did not converge')
    monkeypatch.setattr(sop.np.linalg, 'lstsq', broken)
    block = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    rhs = np.array([1.0, 2.0, 2.0])
    flags = []
    with collecting(flags), pytest.warns(sop.VectorFitFallback):
        x = sop._vector_fit_lstsq(block, rhs)
    assert np.allclose(x, [1.0, 1.0])
    assert [f['flag'] for f in flags] == ['vector_fit_lstsq_fallback']
