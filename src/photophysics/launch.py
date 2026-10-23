"""The thread environment of a launch, set before numpy is imported: the BLAS
and OpenMP pools are sized at import, so `__main__` and the launcher call
`set_threads(threads_in(argv))` before anything numerical loads. This module
imports nothing numerical."""
import argparse
import os

from src.Base.numerics.constants import OPENBLAS_MAX_THREADS, TADF_THREADS

#: The thread variables every launch sets to its count.
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def threads_in(argv):
    """The --threads count of the command line `argv` (`cli.parser`'s
    option), TADF_THREADS where it is not given."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--threads', type=int, default=TADF_THREADS)
    return ap.parse_known_args(argv)[0].threads


def set_threads(threads):
    """The thread environment of `threads` per rank; OpenBLAS clamped to
    OPENBLAS_MAX_THREADS."""
    for name in THREAD_VARIABLES:
        os.environ[name] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(min(threads,
                                                 OPENBLAS_MAX_THREADS))
