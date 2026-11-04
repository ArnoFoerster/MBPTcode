"""The mean field of the solvated driver's ADC(2) route
(photophysics.solvated.adc2_numerics, _factories): converged as the one
Numerics object says. Water 6-31G, frozen 1, cc-pvdz-ri / -jkfit.

  1. adc2_numerics carries the walk route's convergence: energy test
     SCF_OPTIMIZATION_CONV_TOL, orbital gradient STATE_MODEL_SCF_GRAD_TOL,
     SCF_MAX_CYCLE cycles.
  2. the MP2 and ADC(2) state models _factories builds converge their SCF to
     the Numerics' scf section, whatever it holds (a derived one reaches the
     mean field too).
  3. a mean field that reaches its cycle cap is kept, warned about and flagged
     'scf_not_converged' by the state model.

Run: python -m pytest -q tests/test_solvated_adc2_scf.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import (SCF_MAX_CYCLE, SCF_OPTIMIZATION_CONV_TOL,
                                         STATE_MODEL_SCF_GRAD_TOL)
from src.environments.reaction_field import NoMedium
from src.photophysics import solvated
from tests.adc_common import ATOMS

ARGS = argparse.Namespace(method='adc2', jk_auxbasis='cc-pvdz-jkfit', reference=None,
                          frozen=1, auxbasis='cc-pvdz-ri', nroots=3, track='index',
                          loop_tol_q=1e-9, loop_tol_energy=1e-11)
MOL = solvated._mol_of([[a, *xyz] for a, xyz in ATOMS], '6-31g', 4000)


def mean_field(numerics, kind='S0'):
    _, make_state = solvated._factories(ARGS, 'gas', {'numerics': numerics})
    state = make_state(kind)(MOL, NoMedium(MOL))
    hf = state.hf if kind == 'S0' else state.mp2.hf
    return state, hf


def test_walk_convergence():
    s = solvated.adc2_numerics(ARGS).scf
    assert (s.conv_tol, s.conv_tol_grad, s.max_cycle) == (
        SCF_OPTIMIZATION_CONV_TOL, STATE_MODEL_SCF_GRAD_TOL, SCF_MAX_CYCLE)


@pytest.mark.parametrize('kind', ['S0', 'singlet'])
def test_numerics_reach_the_mean_field(kind):
    num = solvated.adc2_numerics(ARGS)
    _, hf = mean_field(num, kind)
    assert (hf.mf.conv_tol, hf.mf.conv_tol_grad, hf.mf.max_cycle) == (
        num.scf.conv_tol, num.scf.conv_tol_grad, num.scf.max_cycle)
    other = num.derive('a tighter mean field for this check',
                       scf={'conv_tol': 1e-12, 'conv_tol_grad': 1e-10, 'max_cycle': 77})
    _, hf = mean_field(other, kind)
    assert (hf.mf.conv_tol, hf.mf.conv_tol_grad, hf.mf.max_cycle) == (1e-12, 1e-10, 77)


def test_cap_is_flagged():
    num = solvated.adc2_numerics(ARGS).derive('a cap the SCF cannot meet',
                                              scf={'max_cycle': 2})
    state, _ = mean_field(num)
    with pytest.warns(RuntimeWarning, match='did not converge'):
        r = state(np.zeros((MOL.nao, MOL.nao)), 0.0)
    assert 'scf_not_converged' in [f['flag'] for f in r.info['flags']]
