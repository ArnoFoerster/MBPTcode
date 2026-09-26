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
interconnect the chi0 all-reduce (nfreq x naux^2) costs more than the compute
it saves at every acene size measured -- 2.3 s against 0.0 s at one ring,
121.7 s against 18.3 s at twelve. Distributing tau across nodes there makes
the calculation SLOWER. That is a property of the FABRIC, not of the split:
the volume is set by the problem, and on InfiniBand the same bytes move two
orders of magnitude faster. Job-level parallelism (one molecule or one state
per node) needs none of this and always scales; `mpi_map` below is that axis
for the loops a driver used to run in one process.

WHICH AXIS TO SPLIT. Tau is where the M^2 sweep is, and every tau point feeds
every frequency, so a tau split always ends in one reduction of (nfreq or
ntau) x naux^2 -- chi0 and proj(tau) -- or npade x nao^2 for the AO
self-energy. Frequency is where a per-point LU is, and a frequency's result
is its own row, so a frequency split needs no reduction; but splitting chi0
over frequency would REPLICATE the tau sweep and save nothing. So: tau for
the sweeps (`polarizability_projected_sweep`, the self-energy, the static W
of the BSE kernel), frequency for the contour-deformation contraction and the
dRPA frequency loop, rows of Zt for the BSE block action. All of them take a
`comm`; every rank runs the whole driver in lockstep and the reductions make
the ranks agree bitwise.

WHAT THE RANKS MUST AGREE ON. A reduction adds the ranks' partials, so the
partials have to be partials OF ONE CALCULATION: every rank's copy of the
inputs it contracts must hold the same bits. That does not happen by itself.
Each rank runs its own SCF and its own ISDF fit, on its own node, with its own
threads, and two runs of the same threaded dense arithmetic on different nodes
are not bit-identical -- measured across two nodes, where
the ranks' interpolation points and eigenvalues drift apart and the reduced
sum is then of terms belonging to different calculations. Worse, a DISCRETE
decision taken from that arithmetic (an auto-sized grid, a pivot count, how
many trial vectors a Davidson keeps) can differ outright, and then the ranks
call the same collective with buffers of different sizes: MPI_ERR_TRUNCATE, or
a deadlock. `replicate` and `lockstep` close the first half -- rank 0's inputs
overwrite everyone's. The second is closed either by a ROOT-DRIVEN iterative
solver, which leaves every decision to rank 0 and lets the other ranks serve
only the partials it asks for, or by `lockstep` at the entry of every kernel,
which makes every kernel's output, and so every decision taken from it, the
same on every rank.

THE DISTRIBUTION CONTEXT. `with distributed(comm):` makes `comm` the current
communicator of everything that runs inside the block on this thread, and
`current_comm()` reads it. Only realization kernels communicate: a kernel takes
`comm=None` and falls back to `current_comm()`, locksteps its inputs, computes
its serial-shaped partial and gathers or all-reduces it. An all-reduced buffer
is identical on every rank by the MPI standard and a gathered one travels
verbatim, so the kernel's output holds the same bits everywhere, and a driver
replicated on every rank (a Davidson loop, an SCF loop, a geometry walk) takes
the same decisions everywhere without carrying a communicator or speaking a
protocol. The context is per THREAD because `run_simulated` runs its ranks as
threads of one process; on a real communicator the block also aborts the job
when an exception leaves it, since the rank that raised will never enter the
collective its peers are waiting in.

Set MBPT_USE_MPI=0 to force the serial path.

`simulated_world(size)` gives `size` communicators that reduce through shared
memory across THREADS of one process. It exists so every distributed path can
be exercised where MPI cannot start -- the sandboxed test runner, a laptop
without mpirun -- with the real reduction, not a no-op: each rank runs in its
own thread and `reduce_sum` blocks until every rank has contributed, then
hands all of them the rank-ordered sum. tests/test_mpi_routes.py and
tests/test_mpi_grid_distribution.py under mpirun remain the check that the
wire protocol matches.
"""
import contextlib
import copy
import hashlib
import os
import pickle
import sys
import threading
import traceback

import numpy as np

from src.Base.constants import (AGREEMENT_DIGEST_BLOCK, AGREEMENT_DIGEST_SEED,
                                MPI_COUNT_MAX)

MPI = None
_HAS_MPI = False
_TRIED = False


class _Frame:
    """One `distributed` block: its communicator and whether it audits."""

    __slots__ = ('comm', 'audit')

    def __init__(self, comm, audit):
        self.comm, self.audit = comm, audit


class _ContextState(threading.local):
    """Per-thread distribution state: each simulated rank is a thread."""

    def __init__(self):
        self.stack = []                        # _Frame, innermost last
        self.guarded = False                   # an abort guard is active here
        self.reset_stats()

    def reset_stats(self):
        """Zero this thread's lockstep counters."""
        self.stats = {'calls': 0, 'bytes': 0, 'checked_calls': 0,
                      'skipped_bytes': 0, 'audited_calls': 0,
                      'mismatched_calls': 0, 'max_abs_diff': 0.0}


_context = _ContextState()


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


def contiguous_block(n, rank, size):
    """[start, stop) of the rows `rank` owns when n rows are cut into `size`
    contiguous blocks, the first n % size of them one row longer.

    Contiguous rather than round-robin: this partitions the ROW index of a
    grid-space matrix (the Zt of the BSE block action), where a GEMM on a
    contiguous slab is the operation and a strided gather would be paid on
    every application. Surplus ranks own an empty block.
    """
    q, r = divmod(int(n), int(size))
    start = rank * q + min(rank, r)
    return start, start + q + (1 if rank < r else 0)


def reduce_sum(a, comm):
    """In-place all-reduce (sum) of a numpy array. No-op without a comm.

    A `SimulatedComm` reduces through shared memory; anything else is an
    mpi4py communicator. The sum is elementwise, so it is taken in windows of
    at most `MPI_COUNT_MAX` elements, one call each; an array below that is
    one window and one call, as it always was.
    """
    if comm is None or comm.Get_size() == 1:
        return a
    buf = np.ascontiguousarray(a)
    for part in _element_windows(buf):
        if isinstance(comm, SimulatedComm):
            comm.allreduce_sum(part)
        else:
            comm.Allreduce(MPI.IN_PLACE, part, op=MPI.SUM)
    if buf is not a:
        a[...] = buf
    return a


def reduce_scatter_rows(a, comm, out=None):
    """This rank's `contiguous_block` of rows of the sum over ranks of `a`.

    The part of `reduce_sum` a rank that reads only its own rows of the sum
    needs: every rank hands in its whole partial `a` (left unchanged) and
    receives the summed rows [start, stop) it owns, into `out` (allocated when
    None) and nothing else, so each rank sends and receives one partial's worth
    of data where a bandwidth-optimal all-reduce moves two. Serially the one
    block is the whole array: `a` itself, or copied into `out`.

    A SUM, RE-ASSOCIATED like any: MPI's Reduce_scatter may add the partials
    in another order than its Allreduce, so the rows agree with `reduce_sum`'s
    to the summation order; the simulated communicator adds in rank order for
    both, bitwise. The sum is elementwise, so a caller may lay its own data out
    in any order inside each rank's rows. float64; past `MPI_COUNT_MAX`
    doubles it runs in rounds, each rank's next window of rows per call, the
    counts in doubles.
    """
    if comm is None or comm.Get_size() == 1:
        if out is None:
            return a
        out[...] = a
        return out
    size, rank = comm.Get_size(), comm.Get_rank()
    buf = np.ascontiguousarray(a)
    if buf.dtype != np.float64:
        raise TypeError(f'reduce_scatter_rows sums float64, not {buf.dtype}')
    row = _row_length(buf)
    _check_row(row)
    blocks = [contiguous_block(buf.shape[0], r, size) for r in range(size)]
    r0, r1 = blocks[rank]
    shape = (r1 - r0,) + buf.shape[1:]
    if out is None:
        out = np.empty(shape)
    elif (out.shape != shape or out.dtype != np.float64
          or not out.flags.c_contiguous):
        raise ValueError(f'reduce_scatter_rows writes a C-contiguous float64 '
                         f'{shape} block, not a {out.dtype} {out.shape} one')
    if buf.size <= MPI_COUNT_MAX:
        _reduce_scatter_once(buf.reshape(-1), out.reshape(-1),
                             [(s1 - s0) * row for s0, s1 in blocks], comm)
        return out
    per = max(1, MPI_COUNT_MAX // max(size * row, 1))
    longest = max(s1 - s0 for s0, s1 in blocks)
    for t0 in range(0, longest, per):
        windows = [(s0 + min(t0, s1 - s0), s0 + min(t0 + per, s1 - s0))
                   for s0, s1 in blocks]
        send = np.concatenate([buf[w0:w1].reshape(-1) for w0, w1 in windows])
        w0, w1 = windows[rank]
        _reduce_scatter_once(send, out[w0 - r0:w1 - r0].reshape(-1),
                             [(v1 - v0) * row for v0, v1 in windows], comm)
    return out


def _reduce_scatter_once(send, recv, counts, comm):
    """recv <- the sum over ranks of their segment `rank` of the flat send,
    cut into consecutive segments of counts[r] doubles: one Reduce_scatter."""
    if isinstance(comm, SimulatedComm):
        comm.reduce_scatter_sum(send, recv, counts)
    else:
        comm.Reduce_scatter([send, MPI.DOUBLE], [recv, MPI.DOUBLE],
                            recvcounts=counts, op=MPI.SUM)


def allgather_blocks(a, comm):
    """In-place all-gather of the `contiguous_block` row blocks of an array.

    Rank r arrives having filled a[start_r:stop_r] and leaves holding the whole
    array. This is an OUTPUT PARTITION, not a reduction: every row is computed
    once, by the rank that owns it, and travels verbatim, so the gathered array
    carries no summation order and holds the same bits on every rank. Use it
    where an intermediate that a rank needs WHOLE is cheaper to split and
    exchange than to form whole on each rank -- z X_v^T in the BSE block
    action, whose exchange reads the entire grid index of it.

    float64, which is every array split this way here; Allgatherv needs the
    datatype named and a silent reinterpretation of another one is not worth
    the generality. An array past `MPI_COUNT_MAX` doubles is gathered in
    rounds of row windows, as `allgather_rows` does.
    """
    if comm is None or comm.Get_size() == 1:
        return a
    size = comm.Get_size()
    blocks = [contiguous_block(a.shape[0], r, size) for r in range(size)]
    buf = np.ascontiguousarray(a)
    if buf.dtype != np.float64:
        raise TypeError(f'allgather_blocks moves float64, not {buf.dtype}')
    if buf.size > MPI_COUNT_MAX:
        _gather_in_rounds(buf, [[b] for b in blocks], comm)
    elif isinstance(comm, SimulatedComm):
        comm.allgather_blocks(buf, blocks)
    else:
        row = int(np.prod(buf.shape[1:], dtype=int))
        counts = [(stop - start) * row for start, stop in blocks]
        displs = [start * row for start, _ in blocks]
        comm.Allgatherv(MPI.IN_PLACE, [buf, counts, displs, MPI.DOUBLE])
    if buf is not a:
        a[...] = buf
    return a


def allgather_rows(a, comm):
    """`allgather_blocks` counted in ROWS, for an array of any size.

    Allgatherv's counts and displacements are C ints. Counted in doubles, the
    displacement of the last block of D at the chlorophyllide hexamer
    (117762 x 28236 = 3.3e9 doubles) is past 2^31, so `allgather_blocks`
    cannot move it at all; counted in rows of one contiguous row datatype
    they stay below the grid size. The same output partition, verbatim rows,
    the same bits on every rank. float64, like `allgather_blocks`.

    What one call gathers is still bounded: an array past `MPI_COUNT_MAX`
    doubles (D above is 3.3e9) is gathered in rounds, each rank sending the
    next window of its rows, so no call moves more than that; the rows travel
    verbatim either way and the result is the one call's bit for bit.
    """
    if comm is None or comm.Get_size() == 1:
        return a
    size = comm.Get_size()
    blocks = [contiguous_block(a.shape[0], r, size) for r in range(size)]
    buf = np.ascontiguousarray(a)
    if buf.dtype != np.float64:
        raise TypeError(f'allgather_rows moves float64, not {buf.dtype}')
    if buf.size > MPI_COUNT_MAX:
        _gather_in_rounds(buf, [[b] for b in blocks], comm)
    elif isinstance(comm, SimulatedComm):
        comm.allgather_blocks(buf, blocks)
    else:
        row = int(np.prod(buf.shape[1:], dtype=int))
        if row:
            rowtype = MPI.DOUBLE.Create_contiguous(row).Commit()
            try:
                comm.Allgatherv(MPI.IN_PLACE,
                                [buf, [stop - start for start, stop in blocks],
                                 [start for start, _ in blocks], rowtype])
            finally:
                rowtype.Free()
    if buf is not a:
        a[...] = buf
    return a


def reduce_max(a, comm):
    """In-place all-reduce (max) of a float64 array. No-op without a comm.

    A maximum is exact and commutative, so every rank ends on the same bits
    whatever order the contributions meet in and whatever the rank count --
    where `reduce_sum`'s additions re-associate with the partition. A maximum
    over each rank's own grid points is therefore the maximum over the whole
    grid, bit for bit, at any number of ranks.
    """
    if comm is None or comm.Get_size() == 1:
        return a
    buf = np.ascontiguousarray(a)
    if buf.dtype != np.float64:
        raise TypeError(f'reduce_max reduces float64, not {buf.dtype}')
    for part in _element_windows(buf):
        if isinstance(comm, SimulatedComm):
            comm.allreduce_max(part)
        else:
            comm.Allreduce(MPI.IN_PLACE, part, op=MPI.MAX)
    if buf is not a:
        a[...] = buf
    return a


def broadcast_rows(a, root, comm):
    """Rank `root`'s C-contiguous float64 array written into every rank's `a`.

    The in-place broadcast of one block from the rank that owns it -- a
    diagonal factor, a solved substitution block, a block of fitting
    coefficients -- counted in ROWS of one contiguous row datatype, like
    `allgather_rows`, so a block past 2^31 doubles still moves. The shape is
    the caller's to know on every rank, as for any Bcast. A block past
    `MPI_COUNT_MAX` doubles goes in row windows, at most that many a call.
    """
    if comm is None or comm.Get_size() == 1:
        return a
    if a.dtype != np.float64 or not a.flags.c_contiguous:
        raise TypeError('broadcast_rows moves a C-contiguous float64 array, '
                        f'not a {a.dtype} one with flags {a.flags}')
    row = _row_length(a)
    windows = _windows(a.shape[0], MPI_COUNT_MAX // max(row, 1))
    for w0, w1 in windows:
        part = a if len(windows) == 1 else a[w0:w1]
        if isinstance(comm, SimulatedComm):
            comm.bcast_into(part, root)
        elif part.size:
            rowtype = MPI.DOUBLE.Create_contiguous(row).Commit()
            try:
                comm.Bcast([part, part.shape[0], rowtype], root=root)
            finally:
                rowtype.Free()
    return a


def allgather_ranges(a, ranges, comm):
    """In place: rank r arrives having filled the rows `ranges[r]` of `a`, a
    list of [start, stop) pairs, and leaves holding every rank's.

    The gather of a BLOCK-CYCLIC distribution, where a rank owns many
    disjoint row ranges rather than one contiguous block: an output
    partition, each row computed by its owner and moved verbatim, so every
    rank ends on the same bits and nothing is added. Counted in rows of one
    contiguous row datatype, as `allgather_rows` is. The ranges must be
    disjoint and the same list on every rank; float64. Past `MPI_COUNT_MAX`
    doubles in all the ranges together it gathers in rounds, each rank
    sending the next window of its rows in their order.
    """
    if comm is None or comm.Get_size() == 1:
        return a
    size = comm.Get_size()
    if len(ranges) != size:
        raise ValueError(f'allgather_ranges needs one range list per rank: '
                         f'{len(ranges)} on a comm of {size}')
    buf = np.ascontiguousarray(a)
    if buf.dtype != np.float64:
        raise TypeError(f'allgather_ranges moves float64, not {buf.dtype}')
    total = sum(s1 - s0 for own in ranges for s0, s1 in own)
    if total * _row_length(buf) > MPI_COUNT_MAX:
        _gather_in_rounds(buf, ranges, comm)
    else:
        _gather_ranges_once(buf, ranges, comm)
    if buf is not a:
        a[...] = buf
    return a


def _gather_ranges_once(buf, ranges, comm):
    """`allgather_ranges` in one collective, in place on the contiguous buf."""
    rank = comm.Get_rank()
    if isinstance(comm, SimulatedComm):
        comm.allgather_ranges(buf, ranges)
    else:
        counts = [sum(stop - start for start, stop in own) for own in ranges]
        row = int(np.prod(buf.shape[1:], dtype=int))
        if row and sum(counts):
            tail = buf.shape[1:]
            mine = (np.concatenate([buf[s0:s1] for s0, s1 in ranges[rank]])
                    if counts[rank] else np.empty((0,) + tail))
            recv = np.empty((sum(counts),) + tail)
            rowtype = MPI.DOUBLE.Create_contiguous(row).Commit()
            try:
                comm.Allgatherv([mine, counts[rank], rowtype],
                                [recv, counts, _displacements(counts),
                                 rowtype])
            finally:
                rowtype.Free()
            at = 0
            for own in ranges:
                for s0, s1 in own:
                    buf[s0:s1] = recv[at:at + s1 - s0]
                    at += s1 - s0


def _gather_in_rounds(buf, ranges, comm):
    """`allgather_ranges` of the C-contiguous buf in rounds of at most
    `MPI_COUNT_MAX` doubles: in round t each rank sends the t-th window of its
    rows, in the order its ranges list them. Every rank derives the rounds
    from the same ranges, so the collectives pair."""
    size, row = comm.Get_size(), _row_length(buf)
    _check_row(row)
    per = max(1, MPI_COUNT_MAX // max(size * row, 1))
    owned = [np.concatenate([np.arange(s0, s1) for s0, s1 in own] or
                            [np.empty(0, dtype=int)]) for own in ranges]
    nrounds = max(1, max(-(-len(rows) // per) for rows in owned))
    for t in range(nrounds):
        _gather_ranges_once(buf, [_runs(rows[t * per:(t + 1) * per])
                                  for rows in owned], comm)


def _runs(rows):
    """[start, stop) of each run of consecutive integers in `rows`."""
    if len(rows) == 0:
        return []
    cut = np.flatnonzero(np.diff(rows) != 1) + 1
    return [(int(r[0]), int(r[-1]) + 1) for r in np.split(rows, cut)]


def _windows(n, per):
    """[start, stop) windows of at most `per` over range(n); a single window,
    empty for n = 0, when one call carries it all."""
    per = max(int(per), 1)
    if n <= per:
        return [(0, int(n))]
    return [(w0, min(w0 + per, n)) for w0 in range(0, n, per)]


def _element_windows(buf):
    """The C-contiguous buf itself, or flat views of it of at most
    `MPI_COUNT_MAX` elements each: what an elementwise collective sends."""
    windows = _windows(buf.size, MPI_COUNT_MAX)
    if len(windows) == 1:
        return [buf]
    flat = buf.reshape(-1)
    return [flat[w0:w1] for w0, w1 in windows]


def _row_length(a):
    """Elements in one row (axis 0) of `a`: the length of its row datatype."""
    return int(np.prod(a.shape[1:], dtype=int))


def _check_row(row):
    """A row datatype is built from a C-int count as well."""
    if row > MPI_COUNT_MAX:
        raise ValueError(f'one row of {row} doubles is past MPI_COUNT_MAX = '
                         f'{MPI_COUNT_MAX}; split the trailing axes')


def _bcast_array(buf, root, comm):
    """Rank `root`'s C-contiguous buf into every rank's, in windows of at most
    `MPI_COUNT_MAX` elements: the broadcast `lockstep` and `replicate` send."""
    for part in _element_windows(buf):
        if isinstance(comm, SimulatedComm):
            comm.bcast_into(part, root)
        else:
            comm.Bcast(part, root=root)


def exchange_rows(send, send_ranges, recv, recv_ranges, comm):
    """In place, for every ordered pair of ranks at once: the rows
    send[a:b] go to rank s for (a, b) = send_ranges[s], and rank r's rows land
    in recv[c:d] for (c, d) = recv_ranges[r].

    `exchange_blocks` for pieces that are row ranges of two arrays with rows
    of one length -- a (ntau, naux, naux) sweep's tau slices turned into
    auxiliary rows and back, a frequency's rows gathered to the rank that
    factorizes it -- so nothing is packed or unpacked: the ranges ARE the
    Alltoallv's displacements, counted in rows of one contiguous row datatype
    (`allgather_rows`). An output partition, every row moved verbatim, no
    summation order.

    Counts and displacements stay C ints at any size, and no call moves more
    than `MPI_COUNT_MAX` doubles into or out of a rank: each pair's range goes
    in windows of that budget over the ranks, in as many rounds as the longest
    range anywhere needs, which one small allgather settles so that every
    rank makes the same calls. The pair's two ranges must have the same
    length -- what s receives from r is what r sends to s -- and the shapes
    are the caller's to know, as for `exchange_blocks`. float64, C-contiguous.
    """
    size = 1 if comm is None else comm.Get_size()
    for a in (send, recv):
        if a.dtype != np.float64 or not a.flags.c_contiguous:
            raise TypeError('exchange_rows moves C-contiguous float64 arrays, '
                            f'not a {a.dtype} one with flags {a.flags}')
    if len(send_ranges) != size or len(recv_ranges) != size:
        raise ValueError(
            f'exchange_rows needs one range per rank: {len(send_ranges)} to '
            f'send and {len(recv_ranges)} to receive on a comm of {size}')
    if size == 1:
        (a0, a1), (c0, c1) = send_ranges[0], recv_ranges[0]
        recv[c0:c1] = send[a0:a1]
        return recv
    row = max(_row_length(send), _row_length(recv))
    _check_row(row)
    per = max(1, MPI_COUNT_MAX // max(size * row, 1))
    longest = max([b - a for a, b in send_ranges]
                  + [d - c for c, d in recv_ranges])
    nrounds = max(comm.allgather(-(-longest // per)) + [1])
    for t in range(nrounds):
        sends = [(a + min(t * per, b - a), a + min((t + 1) * per, b - a))
                 for a, b in send_ranges]
        recvs = [(c + min(t * per, d - c), c + min((t + 1) * per, d - c))
                 for c, d in recv_ranges]
        if isinstance(comm, SimulatedComm):
            got = comm.alltoall_blocks([send[a:b] for a, b in sends])
            for (c, d), piece in zip(recvs, got):
                recv[c:d] = piece
        elif row:
            rowtype = MPI.DOUBLE.Create_contiguous(row).Commit()
            try:
                comm.Alltoallv(
                    [send, [b - a for a, b in sends], [a for a, _ in sends],
                     rowtype],
                    [recv, [d - c for c, d in recvs], [c for c, _ in recvs],
                     rowtype])
            finally:
                rowtype.Free()
    return recv


def cyclic_tiles_to_blocks(tiles, npts, block, ncol, comm):
    """This rank's `contiguous_block` rows of an (npts, ncol) array that the
    ranks hold as BLOCK-CYCLIC tiles: tile t is rows [t*block, (t+1)*block),
    owned by rank t % size, and `tiles` maps each tile this rank owns to its
    rows.

    The change of layout between a distribution that balances a right-looking
    factorization (block-cyclic: the early tiles fall silent first) and the
    one `SlicedFactors` consumers read (contiguous). An output partition, one
    `exchange_blocks` per round of `size` consecutive tiles -- every rank owns
    at most one tile of a round -- so each round moves at most one tile per
    rank, and every row arrives verbatim. Serially the tiles are simply
    stacked.
    """
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    ntiles = -(-int(npts) // int(block))
    r0, r1 = contiguous_block(npts, rank, size)
    out = np.empty((r1 - r0, ncol))
    for c0 in range(0, ntiles, size):
        send, shapes, lands = [], [], []
        for peer in range(size):
            s0, s1 = contiguous_block(npts, peer, size)
            t = c0 + rank
            if t < ntiles:
                a0, a1 = t * block, min((t + 1) * block, npts)
                lo, hi = max(a0, s0), min(a1, s1)
                send.append(np.ascontiguousarray(
                    tiles[t][max(lo - a0, 0):max(hi - a0, 0)]))
            else:
                send.append(np.empty((0, ncol)))
            t = c0 + peer
            lo = hi = r0
            if t < ntiles:
                lo = max(t * block, r0)
                hi = max(min((t + 1) * block, npts, r1), lo)
            shapes.append((hi - lo, ncol))
            lands.append(lo)
        for lo, piece in zip(lands, exchange_blocks(send, shapes, comm)):
            out[lo - r0:lo - r0 + piece.shape[0]] = piece
    return out


def exchange_blocks(send_blocks, recv_shapes, comm):
    """Rank r's `send_blocks[s]` to rank s, for every ordered pair, at once.

    The collective for TRANSPOSING a distribution: an array cut one way across
    the ranks arriving cut the other way. `send_blocks[s]` is this rank's
    piece for rank s and `recv_shapes[s]` the shape of the piece rank s is
    sending here; what comes back is that list of pieces, rank-ordered. Like
    `allgather_blocks` this is an OUTPUT PARTITION and not a reduction --
    every element is computed once, by one rank, and travels verbatim -- so
    the exchanged array holds the bits its owner computed and no summation
    order at all.

    The shapes are the caller's to know: `Alltoallv` posts the receive before
    anything arrives, and a rank that guessed wrong gets MPI_ERR_TRUNCATE
    rather than a resize. Both sides of a distribution split by the same rule
    can compute them, which is the case this exists for; anything else should
    exchange them through `broadcast` first.

    float64, as `allgather_blocks` is, and for the same reason.
    """
    if comm is None or comm.Get_size() == 1:
        return [np.ascontiguousarray(send_blocks[0])]
    size = comm.Get_size()
    if len(send_blocks) != size or len(recv_shapes) != size:
        raise ValueError(
            f'exchange_blocks needs one block per rank: {len(send_blocks)} to '
            f'send and {len(recv_shapes)} to receive on a comm of {size}')
    if any(np.asarray(b).dtype != np.float64 for b in send_blocks):
        raise TypeError('exchange_blocks moves float64')
    counts = [int(np.prod(shape, dtype=int)) for shape in recv_shapes]
    moved = max(sum(counts), sum(int(np.size(b)) for b in send_blocks))
    if moved > MPI_COUNT_MAX:
        raise ValueError(
            'exchange_blocks counts doubles in one Alltoallv and this rank '
            f'would move more than MPI_COUNT_MAX = {MPI_COUNT_MAX}; '
            '`exchange_rows` moves row ranges in rounds')
    if isinstance(comm, SimulatedComm):
        got = comm.alltoall_blocks(send_blocks)
    else:
        sendbuf = np.concatenate([np.ravel(b) for b in send_blocks])
        sendcounts = [int(np.size(b)) for b in send_blocks]
        recvbuf = np.empty(sum(counts))
        comm.Alltoallv(
            [sendbuf, sendcounts, _displacements(sendcounts), MPI.DOUBLE],
            [recvbuf, counts, _displacements(counts), MPI.DOUBLE])
        got = [recvbuf[d:d + n] for d, n
               in zip(_displacements(counts), counts)]
    return [np.reshape(piece, shape) for piece, shape in zip(got, recv_shapes)]


def _displacements(counts):
    """Where each rank's piece starts in a concatenated buffer."""
    out, off = [], 0
    for n in counts:
        out.append(off)
        off += n
    return out


def broadcast(obj, comm, root=0):
    """Rank `root`'s object on every rank; the object itself, not a partial.

    For the SMALL things a root-driven algorithm has to agree on -- a shape, a
    sentinel, a converged flag, a result tuple at the end. mpi4py's lowercase
    `bcast` pickles, so keep it small; arrays belong in `lockstep` or
    `replicate`.
    """
    if comm is None or comm.Get_size() == 1:
        return obj
    if isinstance(comm, SimulatedComm):
        return comm.bcast(obj, root)
    return comm.bcast(obj, root=root)


def replicate(*objects, comm=None, root=0):
    """Overwrite every rank's copy of each object with rank `root`'s.

    THE RESULT OF A DISTRIBUTED CALL IS THE RESULT OF RANK `root`'s INPUTS.
    The other ranks' copies are discarded -- in place for arrays, so the
    caller's own buffers change under it -- and what comes back is one
    calculation's, not an average of several. Call it on everything a
    distributed entry point contracts into a reduced sum, and call it BEFORE
    anything is decided from those numbers: an auto-sized grid or an
    interpolation-point count read off diverged inputs changes the shape of a
    later collective, which no amount of agreeing afterwards can repair.

    Arrays travel through `Bcast` and keep their identity; anything else is
    pickled through `broadcast` and must be taken from the return value.
    Replication is idempotent, so nested entry points re-send data that is
    already equal rather than trusting the call path.
    """
    if comm is None or comm.Get_size() == 1:
        return objects[0] if len(objects) == 1 else objects
    out = []
    for obj in objects:
        if not isinstance(obj, np.ndarray):
            out.append(broadcast(obj, comm, root=root))
            continue
        # Ranks whose arrays have different SHAPES have diverged before this
        # call -- an ISDF fit that took a different number of interpolation
        # points, say -- and a Bcast would only report MPI_ERR_TRUNCATE.
        want = broadcast((obj.shape, obj.dtype.str), comm, root=root)
        if (obj.shape, obj.dtype.str) != want:
            raise ValueError(
                f'rank {comm.Get_rank()} holds a {obj.shape} {obj.dtype} array '
                f'where rank {root} holds {want[0]} {want[1]}: the ranks built '
                'different calculations, and replicating one into the other '
                'would only hide it. Build this object once and pass it in.')
        buf = np.ascontiguousarray(obj)
        _bcast_array(buf, root, comm)
        if buf is not obj:
            obj[...] = buf
        out.append(obj)
    return out[0] if len(out) == 1 else tuple(out)


@contextlib.contextmanager
def distributed(comm, audit=None):
    """Make `comm` the current communicator of this thread inside the block.

    THE CONTRACT. Kernels take `comm=None` and fall back to `current_comm()`;
    drivers, chains, surfaces and optimizers never see a communicator. Only an
    entry point enters this block (a job script around its whole run,
    `run_simulated` around each rank); a kernel that holds a comm passes it to
    the kernels it calls rather than entering the block again.

    The innermost block wins and leaving it -- normally or by an exception --
    restores the enclosing one. `distributed(None)` is a serial region inside a
    distributed one: `mpi_map` runs its items there, because the ranks call
    them on DIFFERENT items and a collective inside would pair mismatched
    calls. The state is thread-local, so each rank-thread of `run_simulated`
    sees its own `SimulatedComm` and the calling thread sees none of them.

    On a real mpi4py communicator with more than one rank the block is also an
    `abort_on_exception` guard; on a `SimulatedComm` or None nothing is
    installed and exceptions propagate as usual.

    audit: True makes every `lockstep` inside compare each rank's copy with
           the one it received (`lockstep_stats`); None inherits the enclosing
           block's setting, False outside every block.
    """
    stack = _context.stack
    depth = len(stack)
    if audit is None:
        audit = stack[-1].audit if stack else False
    stack.append(_Frame(comm, bool(audit)))
    try:
        with abort_on_exception(comm):
            yield comm
    finally:
        del stack[depth:]


def current_comm():
    """This thread's innermost `distributed` communicator, or None."""
    stack = _context.stack
    return stack[-1].comm if stack else None


@contextlib.contextmanager
def abort_on_exception(comm):
    """MPI_Abort the job when an exception escapes the block on a real comm.

    A rank that raises leaves the lockstep: its peers wait in a collective it
    will never enter, and at interpreter exit mpi4py's MPI_Finalize waits for
    them in turn, so the job hangs until the wall clock kills it. Here the
    traceback is printed with the rank and `comm.Abort(1)` ends every rank.

    An exception raised inside the block reaches this guard's exit before it
    could reach any excepthook, so the exit is where it aborts; the
    `sys.excepthook` and `threading.excepthook` installed for the block's
    duration cover a thread started inside it that dies. Only the outermost
    guard of a thread acts, so an exception handled inside the outermost
    region -- on every rank alike -- aborts nothing. SystemExit with a success
    code is a normal exit and propagates. A `SimulatedComm`, None and a
    one-rank communicator install nothing: no rank can be left waiting, and
    `run_simulated` already turns one rank's exception into the caller's.
    """
    if (comm is None or isinstance(comm, SimulatedComm)
            or comm.Get_size() == 1 or _context.guarded):
        yield
        return

    def abort(exc_type, exc, tb):
        try:
            sys.stderr.write(
                f'rank {comm.Get_rank()} of {comm.Get_size()}: an exception '
                'left a distributed region; aborting every rank\n')
            traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
            sys.stderr.flush()
        finally:
            comm.Abort(1)

    previous_sys, previous_thread = sys.excepthook, threading.excepthook

    def abort_thread(args):
        if issubclass(args.exc_type, SystemExit):
            previous_thread(args)              # an exit is not a failure
        else:
            abort(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook, threading.excepthook = abort, abort_thread
    _context.guarded = True
    try:
        yield
    except SystemExit as exc:
        if exc.code not in (None, 0):
            abort(type(exc), exc, exc.__traceback__)
        raise
    except BaseException as exc:                # noqa: BLE001 -- re-raised
        abort(type(exc), exc, exc.__traceback__)
        raise
    finally:
        _context.guarded = False
        sys.excepthook, threading.excepthook = previous_sys, previous_thread


def lockstep(x, comm=None, check=False):
    """Rank 0's copy of `x` on every rank, written into every rank's `x`.

    WHY. The ranks' copies of a replicated input are not the same bits: two
    runs of one threaded SCF on two nodes gave orbital
    energies 1.3e-10 Ha apart and orbitals in different gauges. Every
    realization kernel locksteps its inputs at entry, so its partials are
    partials of ONE calculation and its output is identical on every rank;
    every decision a driver then takes from that output is the same on every
    rank, and the replicated drivers stay in lockstep with no protocol.

    comm: defaults to `current_comm()`; None or one rank returns `x` itself,
          untouched and uncopied.
    check: prove agreement before moving anything. Each rank reduces each
           array to `agreement`'s 64-bit digest, the digests ride in the
           allgather that compares the structure anyway, and only an array
           whose digests differ between ranks is broadcast. Pickled leaves
           (scalars, small objects) are broadcast either way.

    WHY CHECK. Where the ranks agree by construction -- trial vectors of a
    Davidson run on an all-reduced action, a density from an all-reduced Fock
    matrix, factors fitted on lockstepped orbitals and points -- the broadcast
    is insurance and a digest is the cheaper proof. At pentacene cc-pVTZ on
    eight nodes the BSE stage's 332 locksteps (664 MB) found
    no rank apart, while broadcasting them raised `davidson_comm` from 0.23 s
    to 3.0 s; the digest runs at 17-18 GB/s (173 MB in 9.8 ms) and sends 8
    bytes an array. A drifted array still shows in its digest and is
    broadcast as unchecked (one changed word always does, several cancel with
    probability ~2^-64), and every rank decides from the same gathered
    digests, so the collectives pair.

    A C-contiguous array is broadcast IN PLACE and returned (the same object);
    any other array through a C-ordered temporary copied back into it. A
    tuple, list, namedtuple or dict is handled element by element, recursively,
    and comes back as the same container type holding the same array objects.
    None stays None; anything else (a scalar, a small object, an object-dtype
    array) is pickled across in one `broadcast` and must be taken from the
    return value. Shapes, dtypes and the container structure must already
    agree on every rank -- these are replicated inputs -- and are checked in
    one allgather first: a disagreement, or a read-only array on a rank that
    would be written, raises ValueError on EVERY rank before any buffer moves.
    A checked call refuses a read-only receiver as well, whether or not the
    digests agree: otherwise the refusal would surface only on the run whose
    ranks drifted.

    This is a collective: every rank calls it the same number of times in the
    same order. Each call adds the array bytes it broadcast to `bytes` in
    `lockstep_stats`; a checked call also counts in `checked_calls` and adds
    the bytes it did not send to `skipped_bytes`. Inside `distributed(comm,
    audit=True)` it compares each rank's copy of every array it broadcast with
    the one it received and counts the calls where any element differed.
    """
    if comm is None:
        comm = current_comm()
    if comm is None or comm.Get_size() == 1:
        return x
    rank = comm.Get_rank()
    signature, arrays, objects = [], [], []
    _flatten(x, '', signature, arrays, objects)
    read_only = tuple(p for p, a in arrays if not a.flags.writeable)
    digests = tuple(_digest(a) for _, a in arrays) if check else None
    seen = comm.allgather((tuple(signature), read_only, digests))
    _refuse_divergence(seen, rank)
    agreed = ([len({entry[2][i] for entry in seen}) == 1
               for i in range(len(arrays))] if check
              else [False] * len(arrays))
    audit = bool(_context.stack) and _context.stack[-1].audit
    mismatched, largest = False, 0.0
    if any(kind == 'object' for _, kind, _ in seen[0][0]):
        before = pickle.dumps(objects) if audit and rank != 0 else None
        objects = broadcast(objects, comm, root=0)
        if before is not None and pickle.dumps(objects) != before:
            mismatched = True
    moved = skipped = 0
    for (_, a), same in zip(arrays, agreed):
        if same:                               # rank 0's bytes everywhere
            skipped += int(a.nbytes)
            continue
        moved += int(a.nbytes)
        if a.size == 0:
            continue
        if a.flags.c_contiguous:
            buf = a
        elif rank == 0:
            buf = np.ascontiguousarray(a)
        else:
            buf = np.empty(a.shape, dtype=a.dtype)
        before = _audit_snapshot(a) if audit and rank != 0 else None
        _bcast_array(buf, 0, comm)
        if buf is not a and rank != 0:
            a[...] = buf
        if before is not None and not _same_bits(before, buf):
            mismatched = True
            largest = max(largest, _largest_difference(before, buf))
    stats = _context.stats
    stats['calls'] += 1
    stats['bytes'] += moved
    if check:
        stats['checked_calls'] += 1
        stats['skipped_bytes'] += skipped
    if audit:
        stats['audited_calls'] += 1
        stats['mismatched_calls'] += int(mismatched)
        stats['max_abs_diff'] = max(stats['max_abs_diff'], largest)
    return _rebuild(x, iter(a for _, a in arrays), iter(objects))


def lockstep_stats(reset=False):
    """This thread's lockstep counters, a copy; `reset` zeroes them after.

    calls, bytes: every `lockstep` that communicated and the array bytes it
        broadcast
    checked_calls, skipped_bytes: of those, the calls made with `check=True`
        and the array bytes they did not broadcast because every rank's
        digest agreed; `bytes + skipped_bytes` is the volume the calls covered
    audited_calls, mismatched_calls: of those inside an audit block, how many
        found this rank's copy different from rank 0's (always 0 on rank 0)
    max_abs_diff: the largest finite |difference| such a call repaired
    agreement_calls, disagreements: every `agreement` that communicated, and
        how many found the ranks' bytes different (the same on every rank)
    first_disagreement: the label of the first such call, None if none
    The last three appear with this thread's first `agreement` across ranks
    and are absent before it, so a run that audits nothing reports exactly
    the lockstep counters; read them with `.get(key, 0)`.
    """
    out = dict(_context.stats)
    if reset:
        _context.reset_stats()
    return out


def agreement(x, comm=None, label=None, audit_only=False):
    """Whether every rank holds the same bytes in `x`, without moving them.

    WHY. A kernel's large inputs -- the factors, W, proj(tau), 10 to 27 GB at
    the chlorophyllide hexamer -- come out of upstream kernels that lockstep
    or all-reduce them, so they are identical by construction and a kernel
    does not broadcast them again. This checks the construction instead: each
    rank reduces its copy to a 64-bit digest and one allgather compares them,
    so a run can show that every kernel's inputs and outputs held one set of
    bits on every rank while moving 8 bytes an array.

    THE DIGEST. The C-ordered bytes are read as 64-bit words u_i, the tail
    zero-padded, and summed mod 2^64 against W[i mod L] c[i div L], W and c
    odd words of one PCG64 stream (`AGREEMENT_DIGEST_SEED`, L =
    `AGREEMENT_DIGEST_BLOCK`). A change confined to one word always changes
    the sum, because an odd weight is invertible mod 2^64; changes in several
    words cancel only where their differences stand in the ratio of two
    pseudo-random weights, ~2^-64, which also catches swapped elements. It
    measured 10 to 18 GB/s on the laptop, 26 GB in 1.4 to 2.6 s, where
    blake2b runs at 0.29 GB/s (90 s), sha1 at 0.49 and a 32-bit crc32 at 3.5;
    a sum over a strided sample would miss the one-ulp drift in an unsampled
    element, and that drift is what this looks for. The shapes, dtypes and
    container structure are compared with the digests; any other leaf (a
    scalar, a mapping value) by a blake2b digest of its pickle.

    comm: defaults to `current_comm()`; None or one rank is True at once,
          uncounted.
    label: recorded as `first_disagreement` in `lockstep_stats` when this is
           the first call on this thread to find the ranks apart.
    audit_only: communicate only inside `distributed(comm, audit=True)` and
                return True elsewhere -- how a kernel checks its inputs and
                outputs in an audited run at no cost in a production one. The
                audit setting must then be the same on every rank, as it is
                when every rank enters the same block.

    A collective, like `lockstep`: every rank calls it the same number of
    times in the same order, and every rank gets the same verdict.
    """
    if comm is None:
        comm = current_comm()
    if comm is None or comm.Get_size() == 1:
        return True
    if audit_only and not (_context.stack and _context.stack[-1].audit):
        return True
    signature, arrays, objects = [], [], []
    _flatten(x, '', signature, arrays, objects)
    mine = (tuple(signature), tuple(_digest(a) for _, a in arrays),
            tuple(_object_digest(o) for o in objects))
    seen = comm.allgather(mine)
    same = all(other == seen[0] for other in seen[1:])
    stats = _context.stats
    stats['agreement_calls'] = stats.get('agreement_calls', 0) + 1
    stats['disagreements'] = stats.get('disagreements', 0) + int(not same)
    if stats.get('first_disagreement') is None:
        stats['first_disagreement'] = None if same else label
    return same


def lockstep_mean_field(mf, comm=None):
    """Rank 0's orbital energies, orbitals, occupations, e_tot and converged
    flag on every rank, written into `mf`; `mf` is returned.

    The spectrum is where a run over ranks diverges first: two runs of one
    threaded SCF on two nodes gave orbital energies 1.3e-10
    Ha apart and orbitals in different gauges, and an auto-sized minimax grid
    then reads a different point count off them. A kernel reading a mean field
    locksteps these arrays at its entry; this is the same call on the object,
    one packed `lockstep` of whichever of the five attributes `mf` carries,
    never of the object itself (which would be pickled). It lives here, below
    every kernel, so replicating a mean field imports no GW code.

    comm: defaults to `current_comm()`; None or one rank returns `mf`
          untouched. Collective, as `lockstep` is.
    """
    if comm is None:
        comm = current_comm()
    if comm is None or comm.Get_size() == 1:
        return mf
    fields = [name for name in ('mo_energy', 'mo_coeff', 'mo_occ', 'e_tot',
                                'converged') if hasattr(mf, name)]
    values = lockstep(tuple(getattr(mf, name) for name in fields), comm)
    for name, value in zip(fields, values):
        setattr(mf, name, value)
    return mf


def _flatten(x, path, signature, arrays, objects):
    """Walk `x` in a fixed order: one (path, kind, meta) signature entry per
    node, the (path, array) leaves and the other leaves in that order."""
    if x is None:
        signature.append((path, 'none', ()))
    elif _is_array(x):
        signature.append((path, 'array', (x.shape, x.dtype.str)))
        arrays.append((path, x))
    elif _is_container(x):
        signature.append((path, 'container', (type(x).__name__, len(x))))
        items = x.items() if type(x) is dict else enumerate(x)
        for key, value in items:
            _flatten(value, f'{path}[{key!r}]', signature, arrays, objects)
    else:
        signature.append((path, 'object', ()))
        objects.append(x)


def _rebuild(x, arrays, objects):
    """`x` with its arrays and other leaves taken, in `_flatten` order, from
    the two iterators."""
    if x is None:
        return None
    if _is_array(x):
        return next(arrays)
    if type(x) is dict:
        return {key: _rebuild(value, arrays, objects)
                for key, value in x.items()}
    if _is_container(x):
        items = [_rebuild(value, arrays, objects) for value in x]
        if type(x) in (tuple, list):
            return type(x)(items)
        return type(x)(*items)                 # a namedtuple takes fields
    return next(objects)


def _is_array(x):
    """An array a buffer broadcast can move; object arrays are pickled."""
    return isinstance(x, np.ndarray) and not x.dtype.hasobject


def _is_container(x):
    """tuple, list, dict or namedtuple: walked element by element."""
    return (type(x) in (tuple, list, dict)
            or (isinstance(x, tuple) and hasattr(type(x), '_fields')))


def _describe(entry):
    """'a float64 array of shape (3, 4) at [1]' for one signature entry."""
    path, kind, meta = entry
    where = f' at {path}' if path else ''
    if kind == 'array':
        return f'a {np.dtype(meta[1]).name} array of shape {meta[0]}{where}'
    if kind == 'container':
        return f'a {meta[0]} of length {meta[1]}{where}'
    return f'{"None" if kind == "none" else "an object"}{where}'


def _refuse_divergence(seen, rank):
    """Raise on every rank if any rank's structure differs from rank 0's or a
    rank that will be written holds a read-only array."""
    reference = seen[0][0]
    for other, entry in enumerate(seen):
        signature = entry[0]
        if signature == reference:
            continue
        i = next((k for k, (a, b) in enumerate(zip(signature, reference))
                  if a != b), min(len(signature), len(reference)))
        theirs = (_describe(signature[i]) if i < len(signature)
                  else 'nothing more')
        ours = (_describe(reference[i]) if i < len(reference)
                else 'nothing more')
        raise ValueError(
            f'lockstep (raised on rank {rank}): rank {other} holds {theirs} '
            f'where rank 0 holds {ours}. The ranks built different '
            'calculations before this call; broadcasting one into the other '
            'would corrupt a buffer or hide the divergence. Build this input '
            'identically on every rank before it reaches a kernel.')
    for other, entry in enumerate(seen):
        read_only = entry[1]
        if other and read_only:
            where = ', '.join(p or 'the top level' for p in read_only)
            raise ValueError(
                f'lockstep (raised on rank {rank}): rank {other} holds a '
                f"read-only array at {where}; lockstep writes rank 0's values "
                'into it in place.')
    checked = [other for other, entry in enumerate(seen)
               if entry[2] is not None]
    if checked and len(checked) != len(seen):
        raise ValueError(
            f'lockstep (raised on rank {rank}): ranks {checked} asked for a '
            'checked call and the others did not; one lockstep is one '
            'collective, called the same way on every rank.')


def _audit_snapshot(a):
    """A C-ordered copy of this rank's array before the broadcast."""
    return np.array(a, order='C')


def _same_bits(before, after):
    """Whether two C-contiguous arrays of one dtype hold identical bytes."""
    return np.array_equal(before.reshape(-1).view(np.uint8),
                          after.reshape(-1).view(np.uint8))


def _largest_difference(before, after):
    """Largest finite |before - after|; 0 for a dtype without a difference."""
    if before.dtype.kind not in 'iufc':
        return 0.0
    kind = complex if before.dtype.kind == 'c' else float
    diff = np.abs(np.asarray(before, dtype=kind)
                  - np.asarray(after, dtype=kind))
    return float(np.max(diff, initial=0.0, where=np.isfinite(diff)))


def _digest(a):
    """`agreement`'s 64-bit digest of the C-ordered bytes of one array."""
    raw = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
    nwords, tail = divmod(raw.size, 8)
    words = raw[:8 * nwords].view(np.uint64)
    if tail:
        last = np.frombuffer(raw[8 * nwords:].tobytes() + bytes(8 - tail),
                             dtype=np.uint64)
        words = np.concatenate([words, last])
    size = AGREEMENT_DIGEST_BLOCK
    nblocks = -(-words.size // size)
    odd = (np.random.default_rng(AGREEMENT_DIGEST_SEED).bit_generator
           .random_raw(size + nblocks) | np.uint64(1))
    weights, scales = odd[:size], odd[size:]
    full = words.size // size
    sums = np.empty(nblocks, dtype=np.uint64)
    if full:
        sums[:full] = words[:full * size].reshape(full, size) @ weights
    if full < nblocks:
        rest = words[full * size:]
        sums[full] = rest @ weights[:rest.size]
    return int(sums @ scales)              # uint64 arithmetic wraps mod 2^64


def _object_digest(obj):
    """`agreement`'s digest of a leaf that is not an array: its pickle's."""
    return hashlib.blake2b(pickle.dumps(obj), digest_size=8).hexdigest()


class _SimulatedWorld:
    """Shared state of one simulated communicator group: a barrier and the
    per-rank contributions of the reduction or gather in flight."""

    def __init__(self, size):
        self.size = int(size)
        self.barrier = threading.Barrier(self.size)
        self.parts = [None] * self.size
        self.total = None
        self.shared = None                     # the broadcast in flight


class SimulatedComm:
    """One rank of `simulated_world`: the communicator surface the routes use
    (Get_rank, Get_size, the reductions through `reduce_sum`/`reduce_max`/
    `reduce_scatter_rows`,
    the broadcasts through `broadcast`/`replicate`/`broadcast_rows`, the
    gathers and the transposing exchange through `allgather_blocks`/
    `allgather_ranges`/`exchange_blocks`), backed by threads instead
    of MPI, so a root-driven solver's whole message sequence runs here. Every
    rank must call each collective the same number of times in the same order,
    exactly as under MPI; a rank that does not deadlocks the group, which is
    the same failure MPI would show.
    """

    def __init__(self, world, rank):
        self._world, self._rank = world, int(rank)

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._world.size

    def allgather(self, obj):
        """[obj of rank 0, obj of rank 1, ...]: the objects themselves, shared
        through memory, where mpi4py's lowercase allgather would pickle."""
        w = self._world
        w.parts[self._rank] = obj
        w.barrier.wait()                       # every object deposited
        out = list(w.parts)
        w.barrier.wait()                       # everyone has copied the list
        if self._rank == 0:
            w.parts = [None] * w.size
        w.barrier.wait()                       # ...before it is cleared
        return out

    def bcast(self, obj, root=0):
        """Rank `root`'s object, DEEP COPIED onto the others.

        Threads share memory, so handing the object itself over would leave
        every rank holding rank `root`'s arrays -- aliasing that MPI's pickled
        bcast cannot produce, and that would make a test comparing the ranks'
        results compare one object with itself."""
        w = self._world
        if self._rank == root:
            w.shared = obj
        w.barrier.wait()                       # the object is there
        out = obj if self._rank == root else copy.deepcopy(w.shared)
        w.barrier.wait()                       # everyone has copied it
        if self._rank == root:
            w.shared = None
        w.barrier.wait()                       # ...before it is cleared
        return out

    def bcast_into(self, buf, root=0):
        """buf (contiguous ndarray) <- rank `root`'s buf, in place."""
        w = self._world
        if self._rank == root:
            w.shared = buf
        w.barrier.wait()                       # the source is there
        if self._rank != root:
            src = w.shared
            if src.shape != buf.shape or src.dtype != buf.dtype:
                raise ValueError(
                    f'rank {self._rank} offered a {buf.shape} {buf.dtype} '
                    f'buffer to a broadcast of a {src.shape} {src.dtype} one; '
                    'under MPI this is MPI_ERR_TRUNCATE')
            buf[...] = src
        w.barrier.wait()                       # everyone has read it
        if self._rank == root:
            w.shared = None
        w.barrier.wait()                       # ...before it is cleared

    def allgather_blocks(self, buf, blocks):
        """buf[start_r:stop_r] <- rank r's rows, for every rank r.

        The blocks are disjoint, so nothing is added and every rank ends on
        the same bits. A rank deposits a COPY of its own rows: threads share
        memory, and handing the slice itself over would leave the others
        aliasing the depositing rank's buffer.
        """
        w = self._world
        start, stop = blocks[self._rank]
        w.parts[self._rank] = buf[start:stop].copy()
        w.barrier.wait()                       # every block deposited
        for r, (s0, s1) in enumerate(blocks):
            if r != self._rank:
                buf[s0:s1] = w.parts[r]
        w.barrier.wait()                       # everyone has copied them
        if self._rank == 0:
            w.parts = [None] * w.size
        w.barrier.wait()                       # ...before it is cleared

    def alltoall_blocks(self, send_blocks):
        """[rank 0's piece for me, rank 1's piece for me, ...].

        A rank deposits COPIES of its pieces: threads share memory, and
        handing the slice itself over would leave the receiver aliasing the
        sender's buffer, which the next round overwrites.
        """
        w = self._world
        w.parts[self._rank] = [np.ascontiguousarray(b).copy()
                               for b in send_blocks]
        w.barrier.wait()                       # every piece deposited
        out = [w.parts[s][self._rank] for s in range(w.size)]
        w.barrier.wait()                       # everyone has taken theirs
        if self._rank == 0:
            w.parts = [None] * w.size
        w.barrier.wait()                       # ...before it is cleared
        return out

    def allreduce_sum(self, buf):
        """buf (contiguous ndarray) <- sum over ranks of buf, in RANK ORDER, so
        the result is deterministic and the same bits on every rank."""
        w = self._world
        w.parts[self._rank] = buf.copy()
        w.barrier.wait()                       # every contribution deposited
        if self._rank == 0:
            total = w.parts[0]
            for part in w.parts[1:]:
                total = total + part
            w.total = total
        w.barrier.wait()                       # the total exists
        buf[...] = w.total
        w.barrier.wait()                       # everyone has read it
        if self._rank == 0:
            w.parts = [None] * w.size
            w.total = None
        w.barrier.wait()                       # ...before it is cleared

    def reduce_scatter_sum(self, send, recv, counts):
        """recv <- the sum over ranks, in RANK ORDER, of segment `rank` of
        their flat send buffers, cut into consecutive segments of counts[r]
        elements: `allreduce_sum`'s additions on those elements, so its bits.

        The send buffers are read in place, never written: each rank's stays
        untouched until the barrier after every rank has summed its segment.
        """
        w = self._world
        if send.size != sum(counts) or recv.size != counts[self._rank]:
            raise ValueError(
                f'rank {self._rank} offered {send.size} doubles and room for '
                f'{recv.size} to a reduce-scatter of counts {list(counts)}; '
                'under MPI this is MPI_ERR_TRUNCATE')
        w.parts[self._rank] = send
        w.barrier.wait()                       # every partial deposited
        s0 = int(sum(counts[:self._rank]))
        s1 = s0 + int(counts[self._rank])
        total = w.parts[0][s0:s1].copy()
        for part in w.parts[1:]:
            total = total + part[s0:s1]
        recv[...] = total
        w.barrier.wait()                       # every segment summed
        if self._rank == 0:
            w.parts = [None] * w.size
        w.barrier.wait()                       # ...before it is cleared

    def allreduce_max(self, buf):
        """buf (contiguous ndarray) <- elementwise max over ranks of buf."""
        w = self._world
        w.parts[self._rank] = buf.copy()
        w.barrier.wait()                       # every contribution deposited
        if self._rank == 0:
            total = w.parts[0]
            for part in w.parts[1:]:
                total = np.maximum(total, part)
            w.total = total
        w.barrier.wait()                       # the maximum exists
        buf[...] = w.total
        w.barrier.wait()                       # everyone has read it
        if self._rank == 0:
            w.parts = [None] * w.size
            w.total = None
        w.barrier.wait()                       # ...before it is cleared

    def allgather_ranges(self, buf, ranges):
        """buf[start:stop] <- rank r's rows, for every range of every rank r.

        A rank deposits COPIES of its rows, for the aliasing reason
        `allgather_blocks` gives."""
        w = self._world
        w.parts[self._rank] = [buf[s0:s1].copy()
                               for s0, s1 in ranges[self._rank]]
        w.barrier.wait()                       # every range deposited
        for r, own in enumerate(ranges):
            if r != self._rank:
                for (s0, s1), rows in zip(own, w.parts[r]):
                    buf[s0:s1] = rows
        w.barrier.wait()                       # everyone has copied them
        if self._rank == 0:
            w.parts = [None] * w.size
        w.barrier.wait()                       # ...before it is cleared


def mpi_map(fn, items, comm=None):
    """map(fn, items) with the items striped over the ranks and the results
    all-gathered, in order, onto every rank.

    This is job-level parallelism for the loops whose iterations are whole
    calculations -- the 6 natm displaced solves of a numerical Hessian or a
    finite-difference gradient, the starts of a conformer search -- which the
    task pool used to be the only way to spread and which the drivers ran in
    one process. Rank r evaluates items r, r + size, ..., so a ragged cost is
    spread rather than piled on one rank; the results travel as Python
    objects (mpi4py's lowercase allgather pickles them), so keep them small
    and picklable -- coordinates and numbers, not a Mole. Without a comm it
    is the builtin map.

    The comm here and any comm the mapped function uses inside must not be the
    same one: the ranks call fn on DIFFERENT items, so a collective inside fn
    would pair up mismatched calls. Give the inner chain None, or a
    sub-communicator; fn runs inside `distributed(None)`, so a kernel in it
    that falls back to `current_comm()` runs serially.

    Use it as a `map_fn`: `functools.partial(mpi_map, comm=comm)`.
    """
    items = list(items)
    if comm is None or comm.Get_size() == 1:
        return list(map(fn, items))
    rank, size = comm.Get_rank(), comm.Get_size()
    with distributed(None):
        mine = [(i, fn(items[i])) for i in range(rank, len(items), size)]
    out = [None] * len(items)
    for chunk in comm.allgather(mine):
        for i, result in chunk:
            out[i] = result
    return out


def simulated_world(size):
    """`size` SimulatedComm objects sharing one reduction group, one per rank.
    Run the ranks in `size` threads and hand rank r the r-th communicator;
    `run_simulated` does both and enters `distributed` on each rank-thread,
    and a caller running its own threads wraps each in `distributed(comm)`."""
    world = _SimulatedWorld(size)
    return [SimulatedComm(world, r) for r in range(world.size)]


def run_simulated(fn, size, *args, **kwargs):
    """fn(comm, *args, **kwargs) on every rank of a `simulated_world(size)`,
    each in its own thread; returns the list of per-rank results. Each
    rank-thread runs inside `distributed(its comm)`, so a kernel called there
    without `comm=` finds it through `current_comm()`. A rank that raises
    brings the exception to the caller instead of hanging the rest: the
    barrier is aborted so the other threads exit, and the first exception is
    re-raised."""
    comms = simulated_world(size)
    results, errors = [None] * size, [None] * size

    def worker(r):
        try:
            with distributed(comms[r]):
                results[r] = fn(comms[r], *args, **kwargs)
        except BaseException as exc:            # noqa: BLE001 -- re-raised below
            errors[r] = exc
            comms[r]._world.barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for exc in errors:
        if exc is not None and not isinstance(exc, threading.BrokenBarrierError):
            raise exc
    for exc in errors:
        if exc is not None:
            raise exc
    return results
