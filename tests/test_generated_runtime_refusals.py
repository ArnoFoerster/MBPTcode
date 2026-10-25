"""Inputs the generated charged-ADC runtime cannot honour are refused before
anything is built, so these checks need no generated module and run in CI:
ADC(6) (never in MBPTcode), the spin-orbital non-Dyson n = 5 (MP5 numerators the
generators do not emit) and the restricted one above n = 5, DF at the ADC(5)
cross blocks and on the non-Dyson route (no DF twins), U^(5), more than one root on the one-root downfold routes, a
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
from src.SingleReference.ADC.nondyson_restricted import NonDysonRestricted
from src.SingleReference.ADC.sigma_driver import restricted_dyson_operator, u_raw_series


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


@pytest.mark.parametrize('opts', [dict(level='adc4', construction='twosector'),
                                  dict(level='adc4', diagonal=True, construction='twosector'),
                                  dict(level='adc4t2', construction='twosector'),
                                  dict(level='adc5', construction='twosector')])
def test_one_root_routes_refuse_nroots(opts):
    with pytest.raises(ValueError, match='one root per solve'):
        _toy_restricted(pdaggerq=True, **opts).solve(nroots=2)


@pytest.mark.parametrize('level', ['adc4', 'adc4t2', 'adc5'])
def test_adc4_and_higher_name_their_construction(level):
    # from ADC(4) on the constructions differ and none is the default: a run
    # names its construction
    with pytest.raises(ValueError, match='needs construction='):
        _toy_restricted(pdaggerq=True, level=level)
    with pytest.raises(ValueError, match='needs construction='):
        ADCSolverUnrestricted.from_arrays(np.array([-1.0, -0.9, 0.3, 0.8]),
                                          np.zeros((4, 4, 4, 4)), nocc=2,
                                          pdaggerq=True, level='adc4')
    assert _toy_restricted(pdaggerq=True, level='adc3').construction == 'twosector'


@pytest.mark.parametrize('construction', ['union', 'diagrammatic'])
def test_complete_adc5_refuses_df(construction):
    # the ADC(5) cross blocks have no DF twins
    leaves = {'B_aa': np.zeros((1, 4, 4)), 'B_bb': np.zeros((1, 4, 4))}
    with pytest.raises(ValueError, match='no DF twins'):
        restricted_dyson_operator(None, 'adc5', extra_leaves=leaves, construction=construction)


@pytest.mark.parametrize('level,construction', [('adc5', 'union'), ('adc5', 'twosector'),
                                                ('adc4', 'diagrammatic')])
def test_decoupling_grid_is_diagrammatic_adc5_only(level, construction):
    with pytest.raises(ValueError, match='decoupling_ntau'):
        restricted_dyson_operator(None, level, construction=construction, decoupling_ntau=8)


def test_nondyson_restricted_refuses_df_and_order6():
    class _DF:
        df = True
    with pytest.raises(ValueError, match='no DF twins'):
        NonDysonRestricted(_DF(), 'IP')

    class _Dense:
        df = False
    with pytest.raises(ValueError, match='ADC\\(5\\)'):
        NonDysonRestricted(_Dense(), 'IP').raw_series(6)


def test_matrix_free_routes_refuse_a_dense_threshold():
    with pytest.raises(ValueError, match='matrix-free'):
        _toy_restricted(pdaggerq=True, level='adc3').solve(threshold=100)


def test_a_missing_generated_module_names_its_producer():
    with pytest.raises(ImportError, match='GenerateRuntime'):
        generated('adc9_no_such_blocks')
