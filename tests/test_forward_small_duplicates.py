"""Small quantities of the forward GW layer that have ONE definition each.

  * the evGW mode list: `evGW` accepts every mode of the dispatcher's
    `qp_energy.EVGW_MODES`, 'imag-frequency' included;
  * the vertex modes: `self_energy.KNOWN_VERTEX_MODES` is METHOD_REGISTRY's;
  * the quasiparticle-weight test (0, 1]: `qp_selection.ac_unjudgeable` and
    `qp_states.is_quasiparticle_root` use the solver's;
  * the Eq. (27) reach: `sum_over_poles.compressible` and
    `qp_states.admitted_by_pole_condition` share one computation;
  * w0 = 0.5 e_min: `imaginary_axis` uses `grids.gap_scaled_w0` at finite
    temperature.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import METHOD_REGISTRY
from src.Base.utils.grids.grids import gap_scaled_w0
import src.SingleReference.GW.evGW as evgw
import src.SingleReference.GW.imaginary_axis as ia
import src.SingleReference.GW.qp_selection as qsel
import src.SingleReference.GW.qp_states as qst
import src.SingleReference.GW.self_energy as se
import src.SingleReference.GW.sum_over_poles as sop
from src.SingleReference.GW.qp_energy import EVGW_MODES
from src.Solvers.qp_equation import quasiparticle_weight

EPS = np.array([-1.3, -0.62, -0.48, 0.12, 0.31, 0.77])
NOCC = 3


def test_evgw_accepts_every_dispatcher_mode():
    src = inspect.getsource(evgw.evgw_eigenvalues)
    assert 'EVGW_MODES' in src and not hasattr(evgw, 'MODES')
    assert 'imag-frequency' in EVGW_MODES


def test_vertex_modes_are_the_registry_s():
    assert set(se.KNOWN_VERTEX_MODES) == {i['vertex_mode']
                                          for i in METHOD_REGISTRY.values()}
    assert 'PSD3' not in se.KNOWN_VERTEX_MODES


@pytest.mark.parametrize('z', [0.0, 1e-9, 0.3, 1.0, 1.0 + 1e-12, -0.2,
                               np.nan, np.inf])
def test_one_weight_test(z):
    assert qst.is_quasiparticle_root(z) == quasiparticle_weight(z, 0.0)
    judged = qsel.ac_unjudgeable({2: 0.0}, {2: z}, 10.0)
    assert (2 in judged) == (not quasiparticle_weight(z, 0.0))


@pytest.mark.parametrize('p', range(len(EPS)))
@pytest.mark.parametrize('limit', [0.2, 0.6, 1.5])
def test_one_eq27_reach(p, limit):
    assert (qst.admitted_by_pole_condition(EPS, NOCC, p, limit)
            == sop.pole_condition(float(EPS[p]), EPS, NOCC, limit))
    gap = float(EPS[NOCC] - EPS[NOCC - 1])
    assert sop.compressible(EPS[p], EPS, NOCC) == sop.pole_condition(
        EPS[p], EPS, NOCC, gap)


def test_finite_temperature_w0_is_gap_scaled():
    assert gap_scaled_w0(EPS, NOCC, gap=0.04) == 0.5 * 0.04
    assert gap_scaled_w0(EPS, NOCC) == 0.5 * (EPS[NOCC] - EPS[NOCC - 1])
    assert '0.5 * e_min' not in inspect.getsource(
        ia.solve_qp_energy_imaginary_axis)
