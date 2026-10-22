"""Inputs the generated charged-ADC runtime cannot honour are refused before
anything is built, so these checks need no generated module and run in CI:
ADC(6) (never in MBPTcode), the non-Dyson n = 5 (MP5 numerators the generators
do not emit), U^(5), more than one root on the one-root downfold routes, a
dense-diagonalization threshold on a matrix-free route, and a missing
generated module (which names the call that writes it).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.CodeGeneration.generate_adc import GenerateRuntime
from src.SingleReference.ADC.base import ADCSolverRestricted, ADCSolverUnrestricted
from src.SingleReference.ADC.generated_modules import generated
from src.SingleReference.ADC.nondyson_isr import NonDysonISR
from src.SingleReference.ADC.sigma_driver import u_raw_series


def _toy_restricted(**opts):
    """Four spatial orbitals, two occupied, random symmetric integrals: enough
    for the refusals, which come before any block is built."""
    rng = np.random.default_rng(0)
    eri = rng.standard_normal((4, 4, 4, 4)) * 0.01
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    eps = np.array([-1.0, -0.5, 0.3, 0.8])
    return ADCSolverRestricted.from_arrays(eps, eri, nocc=2, **opts)


@pytest.mark.parametrize('cls', [ADCSolverRestricted, ADCSolverUnrestricted])
def test_adc6_level_is_refused(cls):
    with pytest.raises(ValueError, match="level='adc6'"):
        cls(None, level='adc6', pdaggerq=True)


def test_generate_runtime_refuses_adc6():
    with pytest.raises(ValueError, match='ADC\\(6\\)'):
        GenerateRuntime(orders=(2, 6), verbose=False)


def test_nondyson_order5_is_refused():
    eps = np.array([-1.0, -0.9, 0.5, 0.6])
    g = np.zeros((4, 4, 4, 4))
    with pytest.raises(ValueError, match='MP5 amplitude numerators'):
        NonDysonISR(eps, g, 2, 'IP').eigenpairs(5)


def test_u5_raw_series_is_refused():
    with pytest.raises(ValueError, match='ADC\\(6\\)'):
        u_raw_series(None, 'IP', 5)


@pytest.mark.parametrize('opts', [dict(level='adc4'), dict(level='adc4', diagonal=True),
                                  dict(level='adc4t2'), dict(level='adc5')])
def test_one_root_routes_refuse_nroots(opts):
    with pytest.raises(ValueError, match='one root per solve'):
        _toy_restricted(pdaggerq=True, **opts).solve(nroots=2)


def test_matrix_free_routes_refuse_a_dense_threshold():
    with pytest.raises(ValueError, match='matrix-free'):
        _toy_restricted(pdaggerq=True, level='adc3').solve(threshold=100)


def test_a_missing_generated_module_names_its_producer():
    with pytest.raises(ImportError, match='GenerateRuntime'):
        generated('adc9_no_such_blocks')
