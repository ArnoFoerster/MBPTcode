"""`threads.spread_over` moves threads, never the thread count, and gives
every mask back.

The X2C operator runs its dense eigensolve on one rank with that rank's
threads spread over its node (`mpi_grid.on_root_with_node_cpus`); the bits stay
those of the same thread count bound to one domain because only the masks move.
Pinned here: on exit every thread that existed has its own mask back and a
thread started inside has the main thread's; the BLAS thread count is the same
inside as outside; with no CPUs, or where the OS has no per-thread masks
(macOS), it is a no-op.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest                                                          # noqa: E402

from src.Base.utils.parallel.threads import blas_threads, spread_over  # noqa: E402

MASKS = hasattr(os, 'sched_setaffinity') and os.path.isdir('/proc/self/task')


def test_no_cpus_is_a_no_op():
    before = os.sched_getaffinity(0) if MASKS else None
    with spread_over(set()):
        if MASKS:
            assert os.sched_getaffinity(0) == before
    if MASKS:
        assert os.sched_getaffinity(0) == before


@pytest.mark.skipif(not MASKS, reason='no per-thread CPU masks on this OS')
def test_masks_move_and_come_back():
    own = os.sched_getaffinity(0)
    one = {min(own)}
    count = blas_threads()
    ready, done = threading.Event(), threading.Event()
    seen = {}

    def worker():
        ready.set()
        done.wait()
        seen['after'] = os.sched_getaffinity(0)

    with spread_over(one):
        assert os.sched_getaffinity(0) == one
        assert blas_threads() == count
        t = threading.Thread(target=worker)
        t.start()
        ready.wait()
    done.set()
    t.join()
    assert os.sched_getaffinity(0) == own
    assert seen['after'] == own
