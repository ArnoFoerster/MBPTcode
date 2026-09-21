import atexit
import os
import sys
import traceback
import warnings

import numpy as np
import scipy.linalg as la

# mpi4py/ELPA are imported ON DEMAND, never at module import time.
#
# WHY (this is not a style preference): `from mpi4py import MPI` runs MPI_Init at
# import. If the interconnect is unavailable -- e.g. a compute node whose IB
# device returns I/O errors -- MPI ABORTS THE PROCESS from C. It does not raise,
# so the try/except that used to wrap this import could never catch it, and
# merely importing this module killed the job with a bare
#     Abort(...) Fatal error in internal_Init_thread ... ucx function returned
#     with failed status
# and no Python traceback. It reached callers that never use ELPA at all, via
# casida.py -> diagonalization.py.
#
# Deferring the import is the whole fix: code that never asks for a large
# diagonalization now never initializes MPI, so it cannot abort. Code that
# DOES cross the ELPA threshold still gets ELPA automatically,
# exactly as before -- no behaviour change where ELPA was actually wanted.
# Set MBPT_USE_ELPA=0 to force the scipy path even above the threshold (useful
# on a node with a broken interconnect, where MPI_Init would abort).
MPI = None
ElpaEigensolver = None
HAS_MPI = False
_MPI_TRIED = False


def _try_init_mpi():
    """Import mpi4py + ELPA on first real use. Returns True if usable."""
    global MPI, ElpaEigensolver, HAS_MPI, _MPI_TRIED
    if _MPI_TRIED:
        return HAS_MPI
    _MPI_TRIED = True
    if os.environ.get('MBPT_USE_ELPA', '').lower() in ('0', 'false', 'no'):
        return False                                  # explicit opt-OUT only
    try:
        # elpa.py imports the binding before mpi4py, the order it insists on;
        # MPI_Init happens inside that import.
        from src.Base.utils.linearAlgebra.elpa import ElpaEigensolver as _Elpa
        from mpi4py import MPI as _MPI
        MPI, ElpaEigensolver, HAS_MPI = _MPI, _Elpa, True
    except (ImportError, RuntimeError):
        HAS_MPI = False
    return HAS_MPI

def get_global_indices_1d(local_size, block_size, grid_dim, process_coord):
    """Generates global coordinates mapping for 2D block-cyclic layout."""
    local_indices = np.arange(local_size)
    block_num_local = local_indices // block_size
    offset_in_block = local_indices % block_size
    global_block_num = block_num_local * grid_dim + process_coord
    global_indices = global_block_num * block_size + offset_in_block
    return global_indices

def _chunk_indices(solver, rank):
    """Global (rows, cols) index arrays of the block-cyclic chunk `rank` owns."""
    prow, pcol = rank % solver.Pr, rank // solver.Pr
    nrows = solver._numroc(solver.global_N, solver.Nb, prow, 0, solver.Pr)
    ncols = solver._numroc(solver.global_N, solver.Nb, pcol, 0, solver.Pc)
    return (get_global_indices_1d(nrows, solver.Nb, solver.Pr, prow),
            get_global_indices_1d(ncols, solver.Nb, solver.Pc, pcol))

# Both transfers are buffered point-to-point rather than comm.gather/scatter:
# the pickled collectives cap one message at 2 GB and hold every chunk a second
# time on rank 0, so neither reaches a Casida matrix of 10^5 pair states.

def gather_block_cyclic(Z_local, global_N, solver, comm):
    """Gathers distributed block-cyclic matrix Z_local to Rank 0; None elsewhere."""
    rank = comm.Get_rank()
    if rank != 0:
        comm.Send(np.ascontiguousarray(Z_local), dest=0, tag=1)
        return None
    Z_full = np.empty((global_N, global_N), dtype=Z_local.dtype)
    for src in range(comm.Get_size()):
        idx_i, idx_j = _chunk_indices(solver, src)
        if src == 0:
            chunk = Z_local
        else:
            chunk = np.empty((len(idx_i), len(idx_j)), dtype=Z_local.dtype)
            comm.Recv(chunk, source=src, tag=1)
        Z_full[np.ix_(idx_i, idx_j)] = chunk
    return Z_full

def scatter_block_cyclic(matrix_full, solver, comm):
    """Scatters global matrix on Rank 0 to all ranks as block-cyclic chunks."""
    rank = comm.Get_rank()
    dtype = comm.bcast(matrix_full.dtype if rank == 0 else None, root=0)
    if rank != 0:
        idx_i, idx_j = _chunk_indices(solver, rank)
        chunk = np.empty((len(idx_i), len(idx_j)), dtype=dtype)
        comm.Recv(chunk, source=0, tag=2)
        return chunk
    own = None
    for dest in range(comm.Get_size()):
        idx_i, idx_j = _chunk_indices(solver, dest)
        chunk = matrix_full[np.ix_(idx_i, idx_j)]
        if dest == 0:
            own = chunk
        else:
            comm.Send(chunk, dest=dest, tag=2)
    return own

def diagonalize_matrix(M, threshold=5000):
    """Diagonalize symmetric M: distributed ELPA if dim >= threshold and MPI available, else local scipy.linalg.eigh.

    Returns (eigenvalues, Z, is_distributed, solver, comm). Distributed, M is
    read on rank 0 only, scattered in block-cyclic chunks, and Z comes back whole
    on rank 0 and None on every other rank; the eigenvalues are on all ranks.
    """
    global_N = M.shape[0] if M is not None else 0

    # size check FIRST, so MPI is never initialized for small matrices; elpa.py
    # solves real matrices only, so a complex Hermitian M stays local
    if global_N >= threshold and not np.iscomplexobj(M) and _try_init_mpi():
        try:
            comm = MPI.COMM_WORLD
            comm.bcast(global_N, root=0)          # served workers read it here
            solver = ElpaEigensolver(global_N=global_N, block_size=64, comm=comm)
            M_local = scatter_block_cyclic(M, solver, comm)
            eigenvalues, Z_local = solver.solve(M_local)
            del M_local
            Z = gather_block_cyclic(Z_local, global_N, solver, comm)
            del Z_local
            return eigenvalues, Z, True, solver, comm
        except Exception as exc:
            if _SERVING:
                # the workers are inside this solve's transfers; a local
                # fallback would leave them waiting until the walltime
                _abort(exc)
            # Do not fail silently: without this the caller cannot tell a
            # distributed solve from a local one, and the local fallback has a
            # hard size ceiling (below) that the distributed path does not.
            warnings.warn(f'distributed ELPA diagonalization unavailable at '
                          f'N={global_N} ({type(exc).__name__}: {exc}); falling '
                          f'back to a local solve', RuntimeWarning, stacklevel=2)

    # Driver choice is a correctness matter, not a tuning knob. 'evd'
    # (divide-and-conquer) needs 1 + 6N + 2N^2 workspace, which crosses
    # INT32_MAX at N ~ 32700 and then fails outright against a 32-bit LAPACK:
    #   "Too large work array required -- computation cannot be performed with
    #    standard 32-bit LAPACK."
    # For a Casida problem that is N_ov, so it bites between anthracene
    # (N_ov = 24111, 0.54 x INT32_MAX) and tetracene (38880, 1.41 x).
    # 'evr' (RRR) needs O(N) workspace instead and has no such ceiling.
    driver = 'evd' if 1 + 6*global_N + 2*global_N**2 <= 2**31 - 1 else 'evr'
    eigenvalues, Z = la.eigh(M, driver=driver)
    return eigenvalues, Z, False, None, None


_SERVING = False


def serve_distributed_solves():
    """Park every rank but 0 to serve the distributed solves rank 0 requests.

    Called once at the top of a driver started under mpirun. Rank 0 returns
    False at once and runs the driver alone, so the matrices are built once;
    the other ranks return True only after rank 0 calls release_workers(), and
    the driver exits on True. In between, each diagonalize_matrix on rank 0
    above its threshold is one served solve: the size by broadcast, the chunk
    of M, the eigenvectors back. Without MPI, or with MBPT_USE_ELPA=0, every
    rank returns False and the driver runs as before.

    A failure inside a served solve, on any rank, aborts every rank: the others
    are mid-transfer and would otherwise wait until the walltime. A driver that
    ends without release_workers(), by an exception or sys.exit, releases the
    workers at interpreter exit.
    """
    global _SERVING
    if not _try_init_mpi():
        return False
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        _SERVING = comm.Get_size() > 1
        if _SERVING:
            atexit.register(release_workers)
        return False
    while True:
        try:
            global_N = comm.bcast(None, root=0)
            if global_N == 0:
                return True
            solver = ElpaEigensolver(global_N=global_N, block_size=64, comm=comm)
            M_local = scatter_block_cyclic(None, solver, comm)
            _, Z_local = solver.solve(M_local)
            del M_local
            gather_block_cyclic(Z_local, global_N, solver, comm)
            del Z_local
            solver.destroy()
        except Exception as exc:
            _abort(exc)


def _abort(exc):
    """Print exc and abort every rank; a served solve cannot recover."""
    traceback.print_exception(exc)
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(1)


def release_workers():
    """End the service serve_distributed_solves started; rank 0, driver end."""
    global _SERVING
    if _SERVING:
        MPI.COMM_WORLD.bcast(0, root=0)
        _SERVING = False
