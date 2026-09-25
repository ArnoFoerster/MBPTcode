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

threadpoolctl is a SOFT dependency. Absent, `blas_single_threaded` is a no-op
and `blas_threads` returns None: the process keeps whatever its environment set
and every result is unchanged, since thread counts move only the summation
order inside BLAS, never the arithmetic pyscf does.
"""
import contextlib
import threading

from src.Base.constants import BLAS_WRAP_MIN_THREADS

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:                      # soft dependency: every routine no-ops
    threadpool_info = None
    threadpool_limits = None


def blas_single_threaded(min_threads=BLAS_WRAP_MIN_THREADS):
    """Context manager holding every BLAS pool at one thread, OpenMP untouched.

    Use it as a context manager and nothing else: threadpoolctl applies the
    limit when the object is MADE and gives it back on exit, so one that is
    built and never entered leaves the whole process at a single thread.
    Restores the counts it found on exit, including when the body raises.
    `contextlib.nullcontext()` in the three cases where the limit is not this
    caller's to set or not worth setting: threadpoolctl absent, a thread other
    than the main one -- the count is process-global -- and a BLAS pool already
    below `min_threads`, where there is nothing to win and a re-associated sum
    to lose.
    """
    off_main = threading.current_thread() is not threading.main_thread()
    if threadpool_limits is None or off_main:
        return contextlib.nullcontext()
    if (blas_threads() or 1) < min_threads:
        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api='blas')


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
