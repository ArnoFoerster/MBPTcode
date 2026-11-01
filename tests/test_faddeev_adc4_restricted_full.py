"""Restricted matrix-free Faddeev-ADC(4): the complete setting of
test_faddeev_adc4_restricted.py (which documents the checks and runs the cheap
one): H2O and LiH in STO-3G and LiH in 6-31+G at three lambdas, the roots of
all three routes from the two highest occupied orbitals, DF on H2O/6-31G.

Run: python tests/test_faddeev_adc4_restricted_full.py, or under pytest.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from tests.generated_runtime import requires_runtime
from tests.test_faddeev_adc4_restricted import run

pytestmark = requires_runtime(4)


@pytest.mark.parametrize('which', ['h2o/sto-3g', 'lih/sto-3g', 'lih/6-31+g', 'df'])
def test_faddeev_adc4_restricted_full(which):
    assert run((which,), full=True)


if __name__ == '__main__':
    sys.exit(0 if run(tuple(sys.argv[1:]) or None, full=True) else 1)
