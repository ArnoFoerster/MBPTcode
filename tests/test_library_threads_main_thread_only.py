"""Only the main thread reads the libraries' thread counts.

threadpoolctl finds the loaded libraries with dl_iterate_phdr and a Python
callback, so it holds the dynamic loader's lock while it waits for the GIL. A
second thread that holds the GIL and needs that lock waits for the first, and
neither moves: numpy's temporary elision walks the C stack through
dl_iterate_phdr on every large `scalar * array`. On a cluster (py311, numpy 2.4)
two simulated ranks of `test_block_action_split` deadlocked so in every run,
one in `openmp_threads`, the other in the perturbed rank's array product (gdb
showed the two locks). The workstation's numpy does not
take that stack walk, so the deadlock itself cannot be reproduced here; what
is gated is that no thread but the main one calls threadpoolctl at all:

  openmp_threads  read once per process by the main thread and kept; a worker
                  gets that count, or 1 before the main thread has read one.
  blas_threads    None in a worker (the count moves under the wraps, so it is
                  not kept).
"""
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.Base.utils import threads


@pytest.fixture
def recorded(monkeypatch):
    """threadpoolctl's `threadpool_info` replaced by one that records the
    thread of every call and reports one OpenMP pool of 3, one BLAS of 5;
    no count kept from earlier calls."""
    calls = []

    def info():
        calls.append(threading.current_thread() is threading.main_thread())
        return [{'user_api': 'openmp', 'num_threads': 3},
                {'user_api': 'blas', 'num_threads': 5}]

    monkeypatch.setattr(threads, 'threadpool_info', info)
    monkeypatch.setattr(threads, '_OPENMP_THREADS', {}, raising=False)
    return calls


def in_a_worker(fn):
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault('v', fn()))
    worker.start()
    worker.join()
    return out['v']


def test_a_worker_never_calls_threadpoolctl(recorded):
    assert in_a_worker(threads.openmp_threads) == 1
    assert in_a_worker(threads.blas_threads) is None
    assert recorded == [], 'a worker thread called threadpool_info'


def test_the_main_threads_openmp_count_is_read_once_and_kept(recorded):
    assert threads.openmp_threads() == 3
    assert threads.openmp_threads() == 3
    assert in_a_worker(threads.openmp_threads) == 3
    assert recorded == [True], 'the OpenMP count was read more than once'


def test_the_main_thread_reads_the_blas_count_every_time(recorded):
    assert threads.blas_threads() == 5
    assert threads.blas_threads() == 5
    assert recorded == [True, True]
