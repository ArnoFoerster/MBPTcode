"""Which thread pool owns a stage: pyscf's OpenMP or the BLAS library's.

The two pools spin against each other, and on sixteen cores the split is an
order of magnitude either way: the SCF costs 117.5 s with both pools at sixteen
threads and 9.4 s with BLAS held at one (`get_veff` 10.38 -> 0.63 s, the DF K
1.90 -> 0.04 s), while a 4000^2 GEMM goes the other direction, 0.24 s at sixteen
BLAS threads against 3.36 s at one. So neither one environment setting nor one
thread count serves a whole run, and the choice is per stage: wrap the kernels
whose work is pyscf's own OpenMP -- K, v_xc, the numint grid pass, the libcint
integral passes -- in `blas_single_threaded`, and never the GEMM-dominated ones
(chi0, Sigma, the BSE block action, the Gram and Cholesky of the ISDF fit),
which need every core inside BLAS.

A wrapped region may still hold GEMM-bound stages: the distributed ISDF-K SCF
runs inside the SCF's wrap, and its row fit, its interaction's rows
Z = (M^T V) M and its K builds are numpy GEMMs -- at one thread the fit alone
costs a sixteen-core node three times its wall (98 s against 31.5 s at
pentacene/cc-pVTZ). `blas_full_pool` gives such a stage back the count the
innermost enclosing `blas_single_threaded` found on entry, and leaves the pool
alone where no wrap took one, so the stage runs on one count inside the SCF
and outside it, the count a one-rank reference ran on. A libcint or pyscf pass
nested inside it takes `blas_single_threaded` again and drops to one.

The gain is a contention effect, so it needs cores to appear, and below
`BLAS_WRAP_MIN_THREADS` there are too few to contend over: the wrap buys
nothing and costs its entry, 1-2 ms of rescanning the loaded pools. So the
threshold is IN the context manager rather than at each call site, and every
caller is free to wrap unconditionally.

What that gate protects is bitwise reproducibility on a laptop. Two threads
dropped to one re-associates the sums BLAS itself makes, which is invisible in
an energy and not invisible to a record compared with `==`: wrapping the SCF of
a quasiparticle route audit at two threads moved its `xc_correction_eV` by
1e-16 and failed six of the thirteen checks of the frozen record it is compared
against. Below the threshold nothing is limited and those gates hold.
`min_threads=1` exercises the mechanics where the ambient count is smaller.

The count is PROCESS-global, not thread-local, so only the main thread may move
it: a worker thread entering the wrap would re-associate a GEMM running on a
neighbouring thread mid-flight. That is not hypothetical -- a simulated-rank
test runs every rank in its own thread of one process, and wrapping the static
exchange build unconditionally scattered the excited-state chain's force by
1.3e-08 Ha/Bohr, past the 1e-08 ISDF reproducibility floor it is gated at. Off
the main thread the wrap is therefore a no-op, which costs a real MPI rank
nothing: ranks are processes and each calls this from its own main thread.

A THIRD POOL: numpy's element-wise work. A ufunc runs on the thread that
calls it, so an element-wise stage between the libcint and the BLAS calls --
the ISDF fit's test co-densities, a shell block of them 3 GB at the
chlorophyllide dimer -- runs on one core whatever OMP_NUM_THREADS says, and
more ranks are the only thing that divides it. `row_map` cuts such a stage's
rows over Python threads (numpy releases the GIL inside its loops), as many
as the process's OpenMP pool (`openmp_threads`). Every element is still made
by the one sequence of operations the unsplit call applies to it, and a
maximum is exact in any order, so the result is the same bits at every count.
It moves no library's thread count, so any thread may call it.

threadpoolctl is a SOFT dependency. Absent, `blas_single_threaded` and
`blas_full_pool` are no-ops, `blas_threads` returns None and `openmp_threads`
1: the process keeps whatever its environment set and every result is
unchanged, since thread counts move only the summation order inside BLAS,
never the arithmetic pyscf does.
"""
import contextlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from src.Base.constants import BLAS_WRAP_MIN_THREADS

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:                      # soft dependency: every routine no-ops
    threadpool_info = None
    threadpool_limits = None


#: Per thread, the BLAS count each enclosing `blas_single_threaded` found on
#: entry, innermost last: what `blas_full_pool` gives back. Only a thread
#: that owns the pool (`_pool_is_ours`) ever fills it.
_TAKEN = threading.local()

#: The executors `row_map` runs on, one per (process, thread count), made on
#: first use and kept: a pass calls it several times per shell block.
_ROW_POOLS = {}
_ROW_POOLS_LOCK = threading.Lock()


class _BlasLimit:
    """Every BLAS pool at `limit` threads from entry to exit, the counts found
    put back on exit, raise or no raise; `taken`, the count a one-thread wrap
    took away, is on `_TAKEN` while it holds."""

    def __init__(self, limit, taken=None):
        self.limit, self.taken = limit, taken
        self._limits = None

    def __enter__(self):
        self._limits = threadpool_limits(limits=self.limit, user_api='blas')
        if self.taken is not None:
            _taken().append(self.taken)
        return self

    def __exit__(self, *exc):
        if self.taken is not None:
            _taken().pop()
        self._limits.__exit__(*exc)
        return False


def blas_single_threaded(min_threads=BLAS_WRAP_MIN_THREADS):
    """Context manager holding every BLAS pool at one thread, OpenMP untouched.

    The limit is applied on entry and the counts it found are given back on
    exit, including when the body raises; while it holds, the count it found
    is what `blas_full_pool` restores. `contextlib.nullcontext()` in the three
    cases where the limit is not this caller's to set or not worth setting:
    threadpoolctl absent, a thread other than the main one -- the count is
    process-global -- and a BLAS pool already below `min_threads`, where there
    is nothing to win and a re-associated sum to lose.
    """
    if not _pool_is_ours():
        return contextlib.nullcontext()
    found = blas_threads() or 1
    if found < min_threads:
        return contextlib.nullcontext()
    return _BlasLimit(1, taken=found)


@contextlib.contextmanager
def blas_full_pool():
    """Context manager giving a GEMM-bound stage the BLAS pool the innermost
    enclosing `blas_single_threaded` took away, until exit.

    Yields the count the stage's GEMMs run on (None where it cannot be read).
    Where no wrap holds the pool -- none entered, one that stayed a no-op, a
    thread other than the main one, threadpoolctl absent -- the pool is the
    caller's and is left as it is, so a stage takes one count inside a wrapped
    region and outside it.
    """
    taken = _taken() if _pool_is_ours() else None
    if not taken:
        yield blas_threads()
        return
    with _BlasLimit(taken[-1]):
        yield blas_threads()


def _pool_is_ours():
    """Whether this thread may move the BLAS pool: threadpoolctl loaded and
    the main thread, since the count is process-global."""
    return (threadpool_limits is not None
            and threading.current_thread() is threading.main_thread())


def _taken():
    """This thread's stack of the counts its wraps took away."""
    if not hasattr(_TAKEN, 'counts'):
        _TAKEN.counts = []
    return _TAKEN.counts


def blas_threads():
    """Threads the linked BLAS is set to now, or None if it cannot be read.

    The largest count over the loaded BLAS libraries -- there is more than one
    when MKL and OpenBLAS are both mapped in -- since that is the pool a GEMM
    would actually use.
    """
    if threadpool_info is None:
        return None
    counts = [lib['num_threads'] for lib in threadpool_info()
              if lib.get('user_api') == 'blas' and lib.get('num_threads')]
    return max(counts) if counts else None


def openmp_threads():
    """Threads the process's OpenMP pools are set to, the largest of them --
    the count OMP_NUM_THREADS gave the run, which no wrap here moves -- or 1
    where none can be read."""
    if threadpool_info is None:
        return 1
    counts = [lib['num_threads'] for lib in threadpool_info()
              if lib.get('user_api') == 'openmp' and lib.get('num_threads')]
    return max(counts) if counts else 1


def row_map(fn, n, threads):
    """[fn(r0, r1)] over contiguous ranges tiling [0, n), one per thread of a
    pool of `threads`, in order; inline where one range is all there is."""
    parts = min(int(threads), n)
    if parts < 1:
        return []
    ranges = [(p * n // parts, (p + 1) * n // parts) for p in range(parts)]
    if parts == 1:
        return [fn(*ranges[0])]
    return list(_row_pool(int(threads)).map(lambda r: fn(*r), ranges))


def _row_pool(threads):
    """This process's executor of `threads` workers, made once."""
    key = (os.getpid(), threads)
    with _ROW_POOLS_LOCK:
        if key not in _ROW_POOLS:
            _ROW_POOLS[key] = ThreadPoolExecutor(threads)
        return _ROW_POOLS[key]
