"""Krylov solves whose operator is a collective: rank 0 iterates, the other
ranks serve its operator applications until a sentinel.

An operator built on distributed Fock pieces reduces over the ranks inside
every application. If each rank iterated on its own arithmetic, a rank that
converged one iteration earlier would leave the others waiting inside a
reduction. Here rank 0 alone decides the next vector and when to stop: it
broadcasts each vector before applying it, the others apply the same vector
with it, and rank 0's solution is replicated to every rank.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres

from src.Base.utils.mpi_grid import broadcast, current_comm, replicate


def lgmres_root_driven(matvec, rhs, precond, atol, maxiter, comm=None):
    """(x, info) of scipy's lgmres on matvec x = rhs, preconditioned by
    `precond`, with `rtol=0` and `atol`; rank 0's on every rank.

    Serially, and on a world of one, this is scipy's call. What rank 0
    raises is raised on every rank, with rank 0's reason attached.
    """
    comm = current_comm() if comm is None else comm
    n = int(np.size(rhs))
    precond_op = LinearOperator((n, n), matvec=precond)
    if comm is None or comm.Get_size() == 1:
        return lgmres(LinearOperator((n, n), matvec=matvec), rhs,
                      M=precond_op, rtol=0.0, atol=atol, maxiter=maxiter)
    if comm.Get_rank() != 0:
        return _follow_root(matvec, n, comm)

    def announced(v):
        v = np.array(v, dtype=float, order='C')
        broadcast(True, comm)
        replicate(v, comm=comm)
        return matvec(v)

    try:
        x, info = lgmres(LinearOperator((n, n), matvec=announced), rhs,
                         M=precond_op, rtol=0.0, atol=atol, maxiter=maxiter)
    except Exception as exc:
        broadcast(None, comm)
        broadcast(('error', f'{type(exc).__name__}: {exc}'), comm)
        raise
    broadcast(None, comm)
    broadcast(('ok', int(info)), comm)
    x = np.array(x, dtype=float, order='C')
    replicate(x, comm=comm)
    return x, int(info)


def _follow_root(matvec, n, comm):
    """A worker's side: apply rank 0's vectors until the sentinel, then take
    rank 0's (x, info), or raise what rank 0 raised."""
    while broadcast(None, comm) is not None:
        v = np.empty(n)
        replicate(v, comm=comm)
        matvec(v)
    status, payload = broadcast(None, comm)
    if status != 'ok':
        raise RuntimeError(f'rank 0 raised inside the root-driven lgmres, so '
                           f'this rank has no solution: {payload}')
    x = np.empty(n)
    replicate(x, comm=comm)
    return x, payload
