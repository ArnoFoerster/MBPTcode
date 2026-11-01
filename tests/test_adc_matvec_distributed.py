"""Dyson and non-Dyson ADC(3) operators over ranks: the chunk loops of the streamed couplings and
couplings-blocks are split by occupied index (and the pp-ladder by auxiliary slab), every rank sums
the partials, so each rank's sigma vector equals the serial one. Simulated ranks (threads, real
reductions); the operator is built on every rank because the setup is distributed too."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.ADC import adc_r_nondyson_df as nd
from src.SingleReference.ADC import adc_r_sigma_df as dyson
from tests.test_nondyson_df_handwritten import _setup

NH3 = 'N 0 0 0.1; H 0.94 0 -0.25; H -0.47 0.81 -0.25; H -0.47 -0.81 -0.25'


@pytest.fixture(scope='module')
def system():
    return _setup(NH3, 'cc-pvdz')


def _dyson(system):
    _, _, nocc, _, s, sc = system
    aop, diag, _ = dyson.build_operator(s, nocc, sc)
    x = np.random.default_rng(3).standard_normal(diag.size)
    return aop(x), x


def _nondyson(system, sector):
    _, _, nocc, _, s, sc = system
    aop, diag, _ = nd.build_operator(s, nocc, sector, sc)
    x = np.random.default_rng(4).standard_normal(diag.size)
    return aop(x), x


@pytest.mark.parametrize('size', [2, 3, 7])
def test_dyson_operator_over_ranks_equals_serial(system, size):
    y0, x = _dyson(system)
    out = run_simulated(lambda comm: _dyson(system)[0], size)
    for r, y in enumerate(out):
        assert np.max(np.abs(y - y0)) < 1e-11 * np.max(np.abs(y0)), f'rank {r}'
        assert np.array_equal(y, out[0]), f'rank {r} differs from rank 0'


@pytest.mark.parametrize('sector', ['IP', 'EA'])
@pytest.mark.parametrize('size', [2, 3])
def test_nondyson_operator_over_ranks_equals_serial(system, sector, size):
    y0, x = _nondyson(system, sector)
    out = run_simulated(lambda comm: _nondyson(system, sector)[0], size)
    for r, y in enumerate(out):
        assert np.max(np.abs(y - y0)) < 1e-10 * np.max(np.abs(y0)), f'rank {r}'
        assert np.array_equal(y, out[0]), f'rank {r} differs from rank 0'
