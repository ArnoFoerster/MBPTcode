"""The two memory walls of restricted diagrammatic ADC(4): the complete setting
of test_adc4_scale_walls.py (which documents the checks and runs the cheap
one): the dense eigh of the H2O/STO-3G supermatrices and the g-free pool on
H2O/6-31G with the roots of the two highest occupied orbitals.

Run: python tests/test_adc4_scale_walls_full.py, or under pytest.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.generated_runtime import requires_runtime
from tests.test_adc4_scale_walls import run

pytestmark = requires_runtime(4)


def test_adc4_scale_walls_full():
    assert run(full=True)


if __name__ == '__main__':
    sys.exit(0 if run(full=True) else 1)
