"""Wall-clock seconds of a named stage, accumulated into a dict."""
import time
from contextlib import contextmanager


@contextmanager
def stage(timer, key):
    """Add the block's seconds to `timer[key]` (accumulating) by one
    assignment at its end; nothing when `timer` is None."""
    if timer is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timer[key] = timer.get(key, 0.0) + time.perf_counter() - t0
