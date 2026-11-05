"""Threaded operand copies for contractions through BLAS.

A contraction handed to BLAS first copies each operand into the (free, contracted)
matrix layout; numpy does that copy on one thread, which on a many-core node
outlasts the multithreaded GEMM it feeds. use_threaded_transposes() splits the
copies over a thread pool and routes the cached einsum through opt_einsum with a
tensordot that uses them: this module is that opt_einsum backend (`BACKEND`), whose
`tensordot`, `transpose` and `einsum` opt_einsum imports by name. The matrices
handed to BLAS are the same ones numpy builds, so results do not change. Off for
every caller until switched on; the DF ADC operators switch it on
(threaded_transposes_default).
"""
import math

import numpy as np

from src.Base.numerics.constants import THREADED_COPY_MAX_THREADS
from src.Base.utils.parallel.threads import budget, pool

_COPY_POOL = None
_COPY_THREADS = 1
_COPY_MIN_ELEMENTS = 1 << 18
_COPY_CHOSEN = False
# the module path opt_einsum imports the backend functions below from
BACKEND = __name__


def _copy_axis(shape):
    """The axis a copy is split along: the first one long enough to give every pool
    thread four slices, else the longest (a short leading axis, e.g. one chunk of
    occupied indices, would leave most threads idle)."""
    for ax, n in enumerate(shape):
        if n >= 4 * _COPY_THREADS:
            return ax
    return max(range(len(shape)), key=lambda ax: shape[ax])


def ascontiguous_threaded(x):
    """numpy.ascontiguousarray(x) with the copy split over the pool when x is large
    and not already contiguous."""
    if x.flags.c_contiguous or x.size < _COPY_MIN_ELEMENTS or _COPY_POOL is None or x.ndim == 0:
        return np.ascontiguousarray(x)
    out = np.empty(x.shape, dtype=x.dtype)
    ax = _copy_axis(x.shape)
    n0 = x.shape[ax]
    step = max(1, -(-n0 // (4 * _COPY_THREADS)))
    lead = (slice(None),) * ax

    def copy(lo):
        sl = lead + (slice(lo, lo + step),)
        out[sl] = x[sl]

    list(_COPY_POOL.map(copy, range(0, n0, step)))
    return out


def _matrix_view(x, shape2d):
    """x (a transposed view) as a C-contiguous matrix, the copy split over threads."""
    return ascontiguous_threaded(x).reshape(shape2d)


def threaded_tensordot(a, b, axes):
    """numpy.tensordot for axes given as (axes_a, axes_b), with threaded copies."""
    a, b = np.asarray(a), np.asarray(b)
    axes_a = [ax % a.ndim for ax in axes[0]]
    axes_b = [ax % b.ndim for ax in axes[1]]
    free_a = [i for i in range(a.ndim) if i not in axes_a]
    free_b = [i for i in range(b.ndim) if i not in axes_b]
    k = math.prod(a.shape[i] for i in axes_a)
    shape_a = [a.shape[i] for i in free_a]
    shape_b = [b.shape[i] for i in free_b]
    am = _matrix_view(a.transpose(free_a + axes_a), (math.prod(shape_a), k))
    bm = _matrix_view(b.transpose(axes_b + free_b), (k, math.prod(shape_b)))
    return np.dot(am, bm).reshape(shape_a + shape_b)


def use_threaded_transposes(n_threads=None, min_elements=None):
    """Switch the einsum below to threaded operand copies (n_threads: default the
    process's thread budget (`threads.budget`), at most THREADED_COPY_MAX_THREADS;
    min_elements: the smallest operand copied on the pool, default 2^18);
    n_threads=1 switches it off. Copies below about 2^18 elements gain nothing
    from the pool."""
    global _COPY_POOL, _COPY_THREADS, _COPY_MIN_ELEMENTS, _COPY_CHOSEN
    _COPY_CHOSEN = True
    if min_elements is not None:
        _COPY_MIN_ELEMENTS = int(min_elements)
    if n_threads is None:
        n_threads = min(THREADED_COPY_MAX_THREADS, budget())
    if n_threads <= 1:
        _COPY_POOL, _COPY_THREADS = None, 1
        return
    _COPY_POOL, _COPY_THREADS = pool(n_threads, 'copies'), n_threads


def threaded_transposes_default():
    """Switch threaded copies on with the defaults of use_threaded_transposes, unless
    a caller already chose (either way)."""
    if not _COPY_CHOSEN:
        use_threaded_transposes()


def threaded_copies_active():
    """True while the copy pool is switched on."""
    return _COPY_POOL is not None


# The opt_einsum backend interface (`BACKEND`).
tensordot = threaded_tensordot
transpose = np.transpose


def einsum(subscripts, *operands):
    """numpy's einsum with its own path search, for the steps opt_einsum does not
    hand to tensordot."""
    return np.einsum(subscripts, *operands, optimize=True)
