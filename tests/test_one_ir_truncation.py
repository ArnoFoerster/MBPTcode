"""The molecular IR grids read one truncation and one basis construction.

`TimeFrequencyGrid.ir`, `IRBasis` and the finite-temperature Pade cap
`ir_continuation_order` all default to IR_EPS, so the transforms and the cap
sized against them describe one basis, and the transforms are built from
`matsubara.ir_sampling`, the construction `scGW.IRGrid` can adopt.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.constants import IR_EPS
import src.Base.utils.grids.matsubara as matsubara
import src.Base.utils.grids.time_frequency as time_frequency
from tests.pins import assert_same


def _default(fn, name='eps'):
    return inspect.signature(fn).parameters[name].default


def test_one_truncation_default():
    assert _default(time_frequency.TimeFrequencyGrid.ir) == IR_EPS
    assert _default(matsubara.IRBasis.__init__) == IR_EPS
    assert _default(matsubara.ir_continuation_order) == IR_EPS
    assert _default(matsubara.ir_sampling) == IR_EPS


def test_the_pade_cap_counts_the_basis_the_transforms_use():
    """ir_continuation_order is the size of the basis ir_sampling builds."""
    for beta, wmax in ((50.0, 3.0), (200.0, 2.4)):
        basis = matsubara.ir_sampling(beta, wmax, 'fermion')[0]
        assert matsubara.ir_continuation_order(beta, wmax) == basis.size


def test_the_grid_is_built_from_ir_sampling(monkeypatch):
    seen = []
    real = time_frequency.ir_sampling

    def spy(*a, **k):
        seen.append(a)
        return real(*a, **k)

    monkeypatch.setattr(time_frequency, 'ir_sampling', spy)
    g = time_frequency.TimeFrequencyGrid.ir(50.0, 5.0, statistics='boson')
    assert seen == [(50.0, 5.0, 'boson', IR_EPS)]
    assert g.meta['eps'] == IR_EPS
    x = matsubara.ir_sampling(50.0, 5.0, 'boson')[1]
    assert_same(g.tau_points, 0.5 * 50.0 * (x + 1.0), unit='1',
                quantity='root', label='tau points')
