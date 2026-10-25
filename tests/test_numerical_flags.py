"""A numerical fallback warns with its own class and leaves a named flag where the caller
collects (`Base.numerical_flags`): the screening factorization that falls back from
Cholesky to LU, and the vector-fitting least squares that falls back from gelsd. Outside
a `collecting` block only the warning is left. A flag reaches every enclosing block, the
block of the thread that started a `row_map`, and, gathered, every rank's record.
"""
import numpy as np
import pytest

from src.Base.numerical_flags import collecting, flag, gathered
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.Base.utils.parallel.threads import row_map
from src.SingleReference.GW import contour_deformation as cd
from src.SingleReference.GW import sum_over_poles as sop

PLANTED = {'flag': 'planted', 'message': 'a planted shortfall'}


def test_a_flag_reaches_every_enclosing_collecting_block():
    outer, inner = [], []
    with collecting(outer):
        with collecting(inner), pytest.warns(RuntimeWarning, match='planted'):
            flag('planted', 'a planted shortfall', RuntimeWarning)
    assert inner == [PLANTED]
    assert outer == [PLANTED]
    with pytest.warns(RuntimeWarning):
        flag('planted', 'nobody collects this', RuntimeWarning)


def test_a_flag_raised_in_a_row_map_worker_reaches_the_callers_block():
    def rows(r0, r1):
        for r in range(r0, r1):
            flag('planted', f'row {r}', RuntimeWarning)
        return r1 - r0
    flags = []
    with collecting(flags), pytest.warns(RuntimeWarning):
        assert sum(row_map(rows, 4, 2)) == 4
    assert gathered(flags) == [{'flag': 'planted', 'message': f'row {r}'}
                               for r in range(4)]


def test_gathered_flags_are_every_ranks_on_every_rank():
    def rank(comm):
        flags = []
        with collecting(flags), pytest.warns(RuntimeWarning):
            flag('planted', f'rank {comm.Get_rank()}', RuntimeWarning)
            flag('planted', 'every rank', RuntimeWarning)
        return gathered(flags)
    expected = sorted([{'flag': 'planted', 'message': m}
                       for m in ('every rank', 'rank 0', 'rank 1', 'rank 2')],
                      key=lambda e: e['message'])
    assert run_simulated(rank, 3) == [expected] * 3


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
