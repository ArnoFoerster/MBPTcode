import atexit
import os
import sys
import traceback
import warnings

import numpy as np
import scipy.linalg as la

from src.Base.constants import MPI_COUNT_MAX

# mpi4py/ELPA are imported ON DEMAND, never at module import time.
#
# WHY (this is not a style preference): `from mpi4py import MPI` runs MPI_Init at
# import. If the interconnect is unavailable -- e.g. a machine whose IB
# device returns I/O errors -- MPI ABORTS THE PROCESS from C. It does not raise,
# so a try/except around this import cannot catch it, and merely importing this
# module would kill the job with a bare
#     Abort(...) Fatal error in internal_Init_thread ... ucx function returned
#     with failed status
# and no Python traceback, in callers that never use ELPA at all, via
# casida.py -> diagonalization.py.
#
# Deferring the import is the whole fix: code that never asks for a large
# diagonalization never initializes MPI, so it cannot abort. Code that DOES
# cross the ELPA threshold still gets ELPA automatically.
# Set MBPT_USE_ELPA=0 to force the scipy path even above the threshold (useful
# on a machine with a broken interconnect, where MPI_Init would abort).
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
# An MPI count is a C int, so a chunk past 2**31 - 1 elements makes Recv raise
# MPI_ERR_ARG (8 ranks, N > 131071); every chunk goes in pieces of at most
# MPI_COUNT_MAX elements, the bound mpi_grid's collectives keep.

def _send_pieces(comm, chunk, dest, tag):
    """Send a chunk, flattened in C order, in pieces of at most MPI_COUNT_MAX."""
    flat = chunk.reshape(-1)
    for k in range(0, flat.size, MPI_COUNT_MAX):
        comm.Send(flat[k:k + MPI_COUNT_MAX], dest=dest, tag=tag)

def _recv_pieces(comm, chunk, source, tag):
    """Receive _send_pieces' pieces, in send order, into a C-contiguous chunk."""
    flat = chunk.reshape(-1)          # a view, so the pieces land in chunk
    for k in range(0, flat.size, MPI_COUNT_MAX):
        comm.Recv(flat[k:k + MPI_COUNT_MAX], source=source, tag=tag)

def gather_block_cyclic(Z_local, global_N, solver, comm):
    """Gathers distributed block-cyclic matrix Z_local to Rank 0; None elsewhere."""
    rank = comm.Get_rank()
    if rank != 0:
        _send_pieces(comm, np.ascontiguousarray(Z_local), 0, 1)
        return None
    Z_full = np.empty((global_N, global_N), dtype=Z_local.dtype)
    for src in range(comm.Get_size()):
        idx_i, idx_j = _chunk_indices(solver, src)
        if src == 0:
            chunk = Z_local
        else:
            chunk = np.empty((len(idx_i), len(idx_j)), dtype=Z_local.dtype)
            _recv_pieces(comm, chunk, src, 1)
        Z_full[np.ix_(idx_i, idx_j)] = chunk
    return Z_full

def scatter_block_cyclic(matrix_full, solver, comm):
    """Scatters global matrix on Rank 0 to all ranks as block-cyclic chunks."""
    rank = comm.Get_rank()
    dtype = comm.bcast(matrix_full.dtype if rank == 0 else None, root=0)
    if rank != 0:
        idx_i, idx_j = _chunk_indices(solver, rank)
        chunk = np.empty((len(idx_i), len(idx_j)), dtype=dtype)
        _recv_pieces(comm, chunk, 0, 2)
        return chunk
    own = None
    for dest in range(comm.Get_size()):
        idx_i, idx_j = _chunk_indices(solver, dest)
        chunk = matrix_full[np.ix_(idx_i, idx_j)]
        if dest == 0:
            own = chunk
        else:
            _send_pieces(comm, chunk, dest, 2)
    return own

_FALLBACK_WARNED = False

# Every common launcher advertises its size in the environment. Read it from
# there rather than from MPI: this is the path where mpi4py is absent or
# unusable, so there is nothing to ask.
_LAUNCHER_RANK_VARS = ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE',
                       'MPI_LOCALNRANKS', 'SLURM_NTASKS')


def _launcher_ranks():
    """How many ranks the launcher started, or 1 if it did not say."""
    for var in _LAUNCHER_RANK_VARS:
        value = os.environ.get(var, '')
        if value.isdigit():
            return int(value)
    return 1


def _warn_elpa_unavailable(global_N):
    """Say so, ONCE, when a matrix cleared the threshold and ELPA was missing.

    Without this the fallback is silent, and silence here is expensive: under
    `mpirun -n 4` every rank runs the whole calculation serially and gets the
    same answer, so nothing looks wrong -- the numbers agree to the last bit
    and a four-way redundant job burns its allocation to no effect. The only
    visible symptom was a test asserting `distributed=True`, which a
    production run does not do.

    Silent for an explicit MBPT_USE_ELPA=0: that is the caller saying they
    know. Silent below the threshold too -- a small matrix was never going to
    distribute.
    """
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED or os.environ.get(
            'MBPT_USE_ELPA', '').lower() in ('0', 'false', 'no'):
        return
    _FALLBACK_WARNED = True
    n_ranks = _launcher_ranks()
    where = (f'and this job has {n_ranks} ranks, so every one of them is now '
             f'solving it redundantly on its own'
             if n_ranks > 1 else 'so it is being solved locally')
    warnings.warn(
        f'ELPA is unavailable (mpi4py and/or pyelpa did not import), {where}. '
        f'The N={global_N} eigensolve cleared the distribution threshold and '
        f'would otherwise have been spread over the ranks. Install pyelpa (see '
        f'"Distributed eigensolve" in docs/parallel.md), or set '
        f'MBPT_USE_ELPA=0 to silence this.', RuntimeWarning, stacklevel=3)


def diagonalize_matrix(M, threshold=5000):
    """Diagonalize symmetric M: distributed ELPA if dim >= threshold and MPI available, else local scipy.linalg.eigh.

    Returns (eigenvalues, Z, is_distributed, solver, comm). Distributed, M is
    read on rank 0 only, scattered in block-cyclic chunks, and Z comes back whole
    on rank 0 and None on every other rank; the eigenvalues are on all ranks.
    """
    global_N = M.shape[0] if M is not None else 0

    # size check FIRST, so MPI is never initialized for small matrices; elpa.py
    # solves real matrices only, so a complex Hermitian M stays local
    wants_elpa = global_N >= threshold and not np.iscomplexobj(M)
    if wants_elpa and _try_init_mpi():
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
    elif wants_elpa:
        # above the threshold, real, and ELPA never came up
        _warn_elpa_unavailable(global_N)

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


def eigh_symmetric(M, threshold=5000):
    """THE dense symmetric eigensolver for a method's final matrix. Every
    route that ends in a dense diagonalization -- ADC (charged and ee),
    Casida -- goes through here, so they share one backend, one threshold and
    one distributed convention instead of a mix of this and bare
    `np.linalg.eigh`.

    Wraps diagonalize_matrix and absorbs the bookkeeping every caller was
    repeating: destroying the ELPA solver, and returning early on the ranks
    that do not own the eigenvectors.

    Returns (w, Z, is_distributed). `w` is on EVERY rank; `Z` is the full
    eigenvector matrix on rank 0 and **None on every other rank** when the
    solve was distributed.

    That None is the whole reason this is not a drop-in replacement for every
    eigh in the tree. It is correct for a SOLVER ENDPOINT, where rank 0 owns
    the answer and the other ranks are done. It is wrong for an intermediate
    FACTORIZATION whose result every rank needs to continue computing with --
    an ISDF/RI fitting matrix, a cavity operator, a W_aux eigendecomposition.
    Those must stay local, or gain an explicit broadcast first; handing them a
    None on rank 1 is how a replicated calculation silently diverges across
    ranks. Small-by-construction blocks (a few-configuration S^2 block, a
    downfolded seed window) stay local too -- there is nothing to distribute,
    and diagonalize_matrix would short-circuit anyway.
    """
    w, Z, is_distributed, solver, comm = diagonalize_matrix(M, threshold=threshold)
    if is_distributed:
        # Z came back whole on rank 0; the other ranks are done.
        solver.destroy()
        if comm.Get_rank() != 0:
            return w, None, True
    return w, Z, is_distributed


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
