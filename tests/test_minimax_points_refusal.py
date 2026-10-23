"""A minimax time grid that cannot be sized is refused, never handed on as ntau=None.

A zero gap, an inverted range and an infinite one each raise with the range
named; the BSE and static-screening resolvers that read the count raise too.
"""
import numpy as np
import pytest

from src.Base.utils.grids.time_frequency import (minimax_points_for_accuracy,
                                                 minimax_points_for_ranges)
from src.SingleReference.LinearResponse.space_time import (
    minimax_points_for_bse, static_screening_grid)


@pytest.mark.parametrize('e_min, e_max, message', [
    (0.0, 10.0, 'not positive'),
    (2.0, 1.0, 'must exceed'),
    (1.0, np.inf, 'no tabulated minimax grid'),
])
def test_an_unrepresentable_range_is_refused(e_min, e_max, message):
    with pytest.raises(ValueError, match=message):
        minimax_points_for_accuracy(e_min, e_max, target=1e-6)


def test_the_ratio_resolver_refuses_instead_of_returning_infinity():
    with pytest.raises(ValueError, match='no tabulated minimax grid'):
        minimax_points_for_ranges([100.0, np.inf])


def test_a_gapless_spectrum_gives_no_bse_grid():
    eps = np.array([-1.0, -0.2, -0.2, 0.5])     # HOMO = LUMO
    with pytest.raises(ValueError, match='not positive'):
        minimax_points_for_bse(eps, 2)
    with pytest.raises(ValueError, match='not positive'):
        static_screening_grid(eps, 2)


def test_a_representable_range_is_sized_as_before():
    n, err = minimax_points_for_accuracy(1.0, 100.0, target=1e-6)
    assert n in range(6, 35, 2) and err <= 1e-6
