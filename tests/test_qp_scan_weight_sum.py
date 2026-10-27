"""A positive-weight self-energy known pointwise is searched outward from the
bare level and stops once the weights found leave no more for the rest.

The pole strengths of the roots of f(w) = w - a - Sigma(w) sum to one when
Sigma is a sum of poles of positive weight (the contour deformation's), so a
root found with Z above what the others can still hold is the root of largest
weight without the rest of the window refined, or even evaluated
(`qp_equation._weighed_outward`, `SampledEquation.outward`). The cells and
brackets are those of the full scan, so the root kept is the full scan's to
the bit; only the work goes. Checked on pole sums, against the same source
with the sum rule switched off (`weight_sum = None`: every bracket refined).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import QP_WINDOW_Z_MIN
from src.Solvers.qp_equation import SampledEquation, find_qp_root

#: (a, pole positions, positive weights): a quasiparticle (Z 0.55) beside
#: the bare level with three satellites in its window, and a level split
#: between two poles that leaves no root above Z 1/2 (the largest 0.49,
#: searched through every window).
QUASIPARTICLE = (0.3, [-0.6, 0.05, 0.55, 0.7, 1.5], [0.1, 0.01, 0.02, 0.01, 0.2])
NO_QUASIPARTICLE = (0.0, [-0.08, 0.07, 0.9], [0.2, 0.15, 0.05])


def equation(a, pol, num):
    """f, f' and an evaluation counter for f = w - a - sum num/(w - pol)."""
    pol, num = np.asarray(pol, float), np.asarray(num, float)
    calls = [0]

    def f(w):
        calls[0] += 1
        return float(w - a - (num / (w - pol)).sum())

    def fp(w):
        return float(1.0 + (num / (w - pol) ** 2).sum())
    return f, fp, calls


def source(case, weight_sum=True):
    f, fp, calls = equation(*case)
    src = SampledEquation(f, fprime=fp, tol=1e-11, positive_weight=True)
    if not weight_sum:
        src.weight_sum = None          # every bracket of a window refined
    return src, calls


@pytest.mark.parametrize('case', [QUASIPARTICLE, NO_QUASIPARTICLE],
                         ids=['quasiparticle', 'none'])
def test_the_sum_rule_keeps_the_full_scans_root_to_the_bit(case):
    full, n_full = source(case, weight_sum=False)
    fast, n_fast = source(case)
    want, got = find_qp_root(full, case[0]), find_qp_root(fast, case[0])
    assert (got.w, got.z, got.quasiparticle) == (want.w, want.z,
                                                 want.quasiparticle)
    assert set(got.found) <= set(want.found)
    assert n_fast[0] < n_full[0]


def test_the_premise_the_weights_sum_to_one():
    """Every root the full scan finds in the widest window of the split
    level, weighed: they hold the whole weight, none of it above 1/2."""
    full, _ = source(NO_QUASIPARTICLE, weight_sum=False)
    got = find_qp_root(full, NO_QUASIPARTICLE[0])
    assert not got.quasiparticle
    zs = [z for _, z in got.found]
    assert max(zs) <= QP_WINDOW_Z_MIN
    assert abs(sum(zs) - 1.0) < 1e-6


def test_a_quasiparticle_stops_the_walk_within_a_few_cells():
    fast, calls = source(QUASIPARTICLE)
    got = find_qp_root(fast, QUASIPARTICLE[0])
    assert got.quasiparticle
    assert calls[0] < 40               # the window's grid alone is 150 points


def test_the_largest_weight_beside_a_followed_satellite_is_the_full_scans():
    """A followed root below Z 1/2: the window rescanned for the largest
    weight (`SampledEquation.follow_root`), outward on the points already
    evaluated, finds the full rescan's largest root."""
    a = QUASIPARTICLE[0]
    full, _ = source(QUASIPARTICLE, weight_sum=False)
    sat = min((w for w, z in find_qp_root(full, a).found
               if z < QP_WINDOW_Z_MIN), key=lambda w: abs(w - 0.59))
    full, n_full = source(QUASIPARTICLE, weight_sum=False)
    fast, n_fast = source(QUASIPARTICLE)
    want = find_qp_root(full, a, follow=(sat, None))
    got = find_qp_root(fast, a, follow=(sat, None))
    assert (got.w, got.z, got.largest) == (want.w, want.z, want.largest)
    assert got.largest[1] > QP_WINDOW_Z_MIN
    assert n_fast[0] < n_full[0]
