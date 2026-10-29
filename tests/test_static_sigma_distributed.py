"""The streamed MP2 density correction (Sigma(infinity)) over ranks: every rank streams its own occupied
chunks and the accumulators are summed, so each rank's (oo, ov, vv) equals the serial result. Simulated
ranks (threads, real reductions)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import compute_delta_gamma2_df_streamed
from tests.test_nondyson_df_handwritten import _setup

NH3 = 'N 0 0 0.1; H 0.94 0 -0.25; H -0.47 0.81 -0.25; H -0.47 -0.81 -0.25'


@pytest.fixture(scope='module')
def system():
    return _setup(NH3, 'cc-pvdz')


def _gamma(system):
    _, eps, nocc, B, _, _ = system
    return compute_delta_gamma2_df_streamed(B, eps, nocc, chunk_size=2)


@pytest.mark.parametrize('size', [2, 3, 5])
def test_streamed_density_over_ranks_equals_serial(system, size):
    ref = _gamma(system)
    out = run_simulated(lambda comm: _gamma(system), size)
    for r, res in enumerate(out):
        for a, b in zip(res, ref):
            assert np.max(np.abs(a - b)) < 1e-12 * max(1.0, np.max(np.abs(b))), f'rank {r}'
        for a, b in zip(res, out[0]):
            assert np.array_equal(a, b), f'rank {r} differs from rank 0'
