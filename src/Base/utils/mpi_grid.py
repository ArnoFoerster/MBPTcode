"""Distribution of imaginary-time / imaginary-frequency grid points over ranks.

The space-time loops are sums over grid points, so a rank computes the terms it
owns and the results are added: no halo, no ordering constraint, and the answer
is bit-comparable to the serial one up to summation order.

mpi4py is imported ON DEMAND, never at module import time -- the same rule
`linearAlgebra/diagonalization.py` documents at length: `from mpi4py import MPI`
runs MPI_Init at import, and on a node whose interconnect is unusable MPI aborts
the process from C rather than raising, so an import at module scope kills jobs
that never wanted MPI at all.

BEFORE USING THIS, weigh the all-reduce against the compute. On a 1 GbE
interconnect the chi0 all-reduce (nfreq x naux^2) costs more than the compute it saves at
every acene size measured -- 2.3 s against 0.0 s at one ring, 121.7 s against
18.3 s at twelve. Distributing tau across nodes there makes the calculation
SLOWER. It pays on a fast fabric, or when naux is small relative to the grid
work. Job-level parallelism (one molecule or one state per node) needs none of
this and always scales.

Set MBPT_USE_MPI=0 to force the serial path.
"""
import os

import numpy as np

MPI = None
_HAS_MPI = False
_TRIED = False


def _try_init_mpi():
    """Import mpi4py on first real use. Returns True if usable."""
    global MPI, _HAS_MPI, _TRIED
    if _TRIED:
        return _HAS_MPI
    _TRIED = True
    if os.environ.get('MBPT_USE_MPI', '').lower() in ('0', 'false', 'no'):
        return False
    try:
        from mpi4py import MPI as _MPI               # may MPI_Init here
        MPI, _HAS_MPI = _MPI, True
    except (ImportError, RuntimeError):
        _HAS_MPI = False
    return _HAS_MPI


def grid_comm(comm=None):
    """(comm, rank, size), or (None, 0, 1) when MPI is unavailable or disabled.

    Passing a comm explicitly skips the probe, so a caller that already has one
    never risks a second MPI_Init.
    """
    if comm is not None:
        return comm, comm.Get_rank(), comm.Get_size()
    if not _try_init_mpi():
        return None, 0, 1
    c = MPI.COMM_WORLD
    return c, c.Get_rank(), c.Get_size()


def partition(n, rank, size):
    """Grid indices owned by `rank`, round-robin.

    Round-robin rather than contiguous blocks: tau points cost the same here,
    but the frequency grids span decades and any future per-point cost variation
    is spread evenly instead of landing on one rank. With size > n the surplus
    ranks get nothing and contribute zero to the reduction, which is correct
    rather than an error -- 18 tau points on 32 ranks is a legitimate, if
    wasteful, configuration.
    """
    return np.arange(rank, n, size)


def reduce_sum(a, comm):
    """In-place all-reduce of a numpy array. No-op without a comm."""
    if comm is None or comm.Get_size() == 1:
        return a
    buf = np.ascontiguousarray(a)
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    if buf is not a:
        a[...] = buf
    return a
