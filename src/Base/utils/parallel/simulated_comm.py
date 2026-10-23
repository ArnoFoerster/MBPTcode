"""A thread-simulated communicator group: `size` ranks run as threads of
one process and reduce, gather and broadcast through shared memory, so the
distributed kernels of `mpi_grid` run and are tested without MPI."""
import copy
import threading

import numpy as np


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
    # the rank context lives in mpi_grid, which imports this module
    from src.Base.utils.parallel.mpi_grid import distributed

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
