"""Block-cyclic scatter and gather: exact round trip, no message over the cap.

`scatter_block_cyclic` and `gather_block_cyclic` move each rank's chunk with
point-to-point Send/Recv. An MPI count is a C int, so one message holds at most
2**31 - 1 elements; above that Recv raises MPI_ERR_ARG. At 8 ranks that is any
matrix past N = 131071, e.g. a Casida matrix of 134640 pair states. The transfers
therefore go in row slabs of at most `_MAX_MESSAGE` elements.

No MPI here: eight threads stand in for eight ranks on a 4 x 2 grid, joined by a
comm that refuses any message over a limit. With the cap lowered to that limit,
a chunk that went whole would be refused, so the round trip passing is the proof
that every message was sliced.

Run: python tests/test_block_cyclic_transfer.py
"""
import os
import queue
import sys
import threading
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# elpa.py imports pyelpa and mpi4py at module scope; neither is needed for the
# transfers, and pyelpa is not installable from PyPI, so stub them.
for name, attrs in (('pyelpa', {'Elpa': object}),
                    ('mpi4py', {'MPI': types.SimpleNamespace(COMM_WORLD=None)})):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

from src.Base.utils.linearAlgebra import diagonalization as dg
from src.Base.utils.linearAlgebra.elpa import ElpaEigensolver

INT_MAX = 2**31 - 1


class ThreadComm:
    """Blocking point-to-point between threads; refuses a message over `limit`."""

    def __init__(self, rank, size, boxes, limit, sizes):
        self.rank, self.size, self.boxes = rank, size, boxes
        self.limit, self.sizes = limit, sizes

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def bcast(self, obj, root=0):
        if self.rank == root:
            for dest in range(self.size):
                if dest != root:
                    self.boxes[root, dest].put(('bcast', obj))
            return obj
        return self.boxes[root, self.rank].get(timeout=5)[1]

    def _refuse(self, buf):
        if buf.size > self.limit:
            raise RuntimeError(f'{buf.size} elements in one message, '
                               f'limit {self.limit}')

    def Send(self, buf, dest, tag):
        buf = np.asarray(buf)
        self._refuse(buf)
        self.sizes.append(buf.size)
        self.boxes[self.rank, dest].put((tag, buf.copy()))

    def Recv(self, buf, source, tag):
        self._refuse(buf)
        got_tag, data = self.boxes[source, self.rank].get(timeout=5)
        if got_tag != tag or data.shape != buf.shape:
            raise RuntimeError(f'expected tag {tag} shape {buf.shape}, '
                               f'got tag {got_tag} shape {data.shape}')
        buf[...] = data


def solver_stub(n, nb, pr, pc):
    """What `_chunk_indices` reads off an ElpaEigensolver, without ELPA."""
    return types.SimpleNamespace(
        global_N=n, Nb=nb, Pr=pr, Pc=pc,
        _numroc=lambda *a: ElpaEigensolver._numroc(None, *a))


def round_trip(n, nb, pr, pc, limit):
    """Scatter an n x n matrix over pr*pc threads and gather it back."""
    size = pr * pc
    M = np.arange(n * n, dtype=np.float64).reshape(n, n)
    solver = solver_stub(n, nb, pr, pc)
    boxes = {(a, b): queue.Queue() for a in range(size) for b in range(size) if a != b}
    sizes, out = [], {}

    def rank_main(rank):
        comm = ThreadComm(rank, size, boxes, limit, sizes)
        try:
            local = dg.scatter_block_cyclic(M if rank == 0 else None, solver, comm)
            out[rank] = (local, dg.gather_block_cyclic(local, n, solver, comm))
        except Exception as exc:
            out[rank] = exc

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    errors = [f'rank {r}: {v}' for r, v in sorted(out.items())
              if isinstance(v, Exception)]
    if errors:
        return False, errors[0], sizes
    chunks_ok = all(
        np.array_equal(out[r][0], M[np.ix_(*dg._chunk_indices(solver, r))])
        for r in range(size))
    return chunks_ok and np.array_equal(out[0][1], M), '', sizes


def check(ok, label, detail=''):
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def main():
    ok = True

    print('\n=== 1. the default cap fits an MPI count ===')
    cap = getattr(dg, '_MAX_MESSAGE', None)
    ok &= check(cap is not None and 0 < cap <= INT_MAX,
                '_MAX_MESSAGE <= 2**31 - 1', f'_MAX_MESSAGE = {cap}')

    print('\n=== 2. round trip under a message limit, 8 ranks on 4 x 2 ===')
    # 203 = 25 blocks of 8 and a remainder of 3, so every chunk shape differs
    limit = 97
    saved = cap
    dg._MAX_MESSAGE = limit
    try:
        passed, err, sizes = round_trip(203, 8, 4, 2, limit)
    finally:
        dg._MAX_MESSAGE = saved
    ok &= check(passed, 'every chunk and the gathered matrix exact',
                err or f'{len(sizes)} messages')
    ok &= check(passed and max(sizes) <= limit and len(sizes) > 2 * 7,
                'no message over the cap, and chunks were sliced',
                f'largest {max(sizes) if sizes else 0} of {limit} elements')

    print('\n=== 3. the default cap leaves a small matrix in one message per chunk ===')
    passed, err, sizes = round_trip(203, 8, 4, 2, INT_MAX)
    ok &= check(passed and len(sizes) == 2 * 7,
                'one Send per worker chunk each way',
                err or f'{len(sizes)} messages for 7 workers')

    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
