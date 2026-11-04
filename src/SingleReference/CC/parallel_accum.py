"""Threaded, fused accumulation of permutation sums -- `pacc`.

Why this exists
---------------
The spin-orbital CCSDT residuals (amplitudes.py) spend most of their time NOT
in tensor contractions but in the antisymmetrizer lines that follow them, e.g.

    contracted_intermediate = -1.0 * einsum('ld,dajk,bcil->abcijk', ...)
    triples_res += (1.0 * contracted_intermediate
                    + -1.0 * einsum('abcijk->abcjik', contracted_intermediate)
                    + -1.0 * einsum('abcijk->bacijk', contracted_intermediate)
                    + 1.0 * einsum('abcijk->bacjik', contracted_intermediate))

There are 133 such lines in triples_residual alone. Written that way they are
(a) single-threaded -- numpy elementwise ufuncs never use BLAS, so MKL_NUM_THREADS
has no effect on them whatsoever -- and (b) needlessly memory-heavy: each
`coef * array` materializes a full rank-6 temporary, and each `+` materializes
another, so a 4-permutation term makes ~7 passes over a (nv^3 no^3) array where
2 would do.

Profiled on nv=24/no=8 (57 MB per rank-6 array), once the contractions were
given a real `optimize=` path this pattern was 84% of triples_residual wall
time and scaled at 1.0x with thread count.

What this does
--------------
`pacc(res, terms)` computes `res += sum(coef * arr)` by splitting the leading
axis into one chunk per worker thread and, within each chunk, summing all
permutation views into a single chunk-sized temporary before one read-modify-
write of `res`. numpy releases the GIL for elementwise ufuncs on large arrays,
so plain Python threads give real parallelism here; the operation is
memory-bandwidth bound, not flop bound, and a single core cannot saturate the
memory system.

Measured (nv=24, no=8, 4-permutation term, one machine):
    current `coef * a + coef * b + ...`   35.1 ms   (25.7 GB/s, 1 core)
    serial in-place +=/-=                 23.1 ms
    pacc, 8 threads                        8.1 ms   (4.3x, ~92 GB/s)
Results are bit-identical to the expression form in every case checked
(max|diff| = 0.0), since the arithmetic performed per element is the same
sum in the same order.

Thread count is the process's thread budget (`threads.budget`: OMP_NUM_THREADS,
else the CPUs this thread may run on); OMP_NUM_THREADS=1 gives the plain serial
path back. Read at call time, not import time, so a caller that pins threads
after import (as the test sweeps do) still wins.
"""
import numpy as np

from src.Base.utils.parallel.threads import budget, pool

# Below this many elements the thread hand-off costs more than the work saved
# (measured crossover is around 1e5 on this hardware; 1<<19 is comfortably
# past it and keeps small singles/doubles blocks on the serial path).
_MIN_PARALLEL_SIZE = 1 << 19

# One chunk per thread is only balanced when nthreads divides shape[0]. It
# often doesn't: the rank-6 triples block has shape[0] = nv = 24, so on 10
# threads linspace hands out chunks of [2,2,3,2,3,2,2,3,2,3] and wall time is
# set by the threads that drew 3 -- a measured 1.25x penalty, exactly the 3/2.4
# ratio. Splitting into one chunk per leading-axis row instead lets the pool
# schedule dynamically and recovers all of it (8.38 -> 6.72 ms).
#
# Only worth it for big blocks: on the rank-5/rank-4 blocks (~1 MB) the extra
# dispatches cost more than the imbalance, measured 0.43-0.73x -- a real
# slowdown. So do it only above _FINE_CHUNK_SIZE and only when the split is
# actually uneven (ee_sigma3's shape[0] = nv = 20 on 10 threads already
# divides evenly, and forcing finer chunks there measured 0.93x).
_FINE_CHUNK_SIZE = 1 << 20
_MAX_CHUNKS_PER_THREAD = 4

def _serial(res, terms):
    for coef, arr in terms:
        if coef == 1:
            res += arr
        elif coef == -1:
            res -= arr
        else:
            res += coef * arr
    return res


def pacc(res, terms):
    """In-place `res += sum(coef * arr for coef, arr in terms)`.

    `terms` is a sequence of (scalar, ndarray) pairs whose arrays broadcast to
    `res.shape` (in practice: permutation views of one contraction result).
    Returns `res`, which is modified in place.
    """
    terms = [(c, a) for c, a in terms]
    if not terms:
        return res

    n = budget()
    if (n == 1 or res.size < _MIN_PARALLEL_SIZE or res.ndim == 0
            or res.shape[0] < 2 or any(a.shape != res.shape for _, a in terms)):
        return _serial(res, terms)

    # Aliasing (an accumulator appearing on its own right-hand side) would make
    # chunked in-place updates read already-updated data, unlike the expression
    # form which evaluates the whole RHS first. Cheap bounds-based check; on a
    # maybe-hit, materialize the sum first and fall back.
    if any(np.may_share_memory(res, a) for _, a in terms):
        return _serial(res, [(1.0, sum(c * a for c, a in terms))])

    nrow = res.shape[0]
    nchunk = min(n, nrow)
    if res.size >= _FINE_CHUNK_SIZE and nrow % nchunk:
        nchunk = min(nrow, _MAX_CHUNKS_PER_THREAD * n)
    bounds = np.linspace(0, nrow, nchunk + 1).astype(int)

    def work(k):
        lo, hi = bounds[k], bounds[k + 1]
        if lo == hi:
            return
        acc = None
        for coef, arr in terms:
            chunk = arr[lo:hi]
            if acc is None:
                acc = chunk.copy() if coef == 1 else (-chunk if coef == -1 else coef * chunk)
            elif coef == 1:
                acc += chunk
            elif coef == -1:
                acc -= chunk
            else:
                acc += coef * chunk
        res[lo:hi] += acc

    list(pool(nchunk, 'accumulate').map(work, range(nchunk)))
    return res
