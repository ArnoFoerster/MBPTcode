"""The quasiparticle root finders warn and flag a numerical event instead of
refusing it (a bad input is refused; a numerical shortfall warns):

- a pole sum with no real root returns a NaN flagged 'no_root' with a
  QPNoRoot warning (`solve_qp_equation_poles`);
- the root of largest weight with Z outside (0, 1] is kept with a
  QPWeightOutside warning and 'z_outside' (`largest_weight_root`), and `QPqb` logs it in `z_outside`.
"""
import numpy as np
import pytest

from src.Solvers.qp_equation import (QPNoRoot, QPWeightOutside,
                                     largest_weight_root,
                                     solve_qp_equation_poles)


def test_a_pole_sum_without_a_root_is_a_flagged_nan():
    # f(w) = w + 0.5 / w: w^2 = -0.5 has no real root
    out = {}
    with pytest.warns(QPNoRoot, match='no root'):
        w, z = solve_qp_equation_poles([-0.5], [0.0], 0.0, out=out)
    assert np.isnan(w) and np.isnan(z) and out['no_root']


def test_a_weight_outside_is_kept_warned_and_flagged():
    # every root of a positive-weight pole sum has 0 < Z <= 1; a negative
    # weight gives Z = 1.1455, to see what the caller does with it
    out = {}
    with pytest.warns(QPWeightOutside, match='outside'):
        w, z = largest_weight_root([-0.1], [1.0], 0.0, out=out)
    assert abs(w - 0.1127016654) < 1e-6 and z > 1.0
    assert out == {'z_outside': True, 'no_root': False}


def test_a_weight_inside_is_quiet():
    out = {}
    w, z = largest_weight_root([0.1, 0.2], [-1.0, 1.0], 0.0, out=out)
    assert 0.0 < z <= 1.0 and out == {'z_outside': False, 'no_root': False}
