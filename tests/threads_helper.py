"""One pyscf OpenMP thread for a test, so its K partials add in one order."""
import warnings

import pyscf.lib
import pytest


@pytest.fixture
def pyscf_one_thread():
    """pyscf's OpenMP GEMM on one thread, where it adds its K partials in one
    order, so repeated runs of one mean field repeat their bits."""
    threads = pyscf.lib.num_threads()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pyscf.lib.num_threads(1)
    yield
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pyscf.lib.num_threads(threads)
