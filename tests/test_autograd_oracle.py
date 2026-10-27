"""The torch.autograd oracle against the production adjoints, slow tier.

Water/cc-pVDZ in PCM(water) on the G1 grid: the production reverse pass from
the factors to Omega must be the derivative torch.autograd computes of the
same function, to round-off. The torch half runs in its own process (see
src/autograd_oracle/__init__.py), so this test never imports torch itself.
"""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.autograd_oracle import gate_factors

pytestmark = pytest.mark.skipif(importlib.util.find_spec('torch') is None,
                                reason='torch is not installed')


CASES = {
    'bse_pcm': [],
    'bse_gas': ['--gas'],
    'qp_pcm': ['--target', 'qp'],
    'tda_triplet_ift': ['--tda', '--spin', 'triplet', '--root-mode', 'ift'],
}


@pytest.mark.parametrize('extra', CASES.values(), ids=CASES.keys())
def test_factor_level_adjoints_are_the_autograd_derivative(extra, tmp_path):
    out = gate_factors.main(['--basis', 'cc-pvdz', '--grid-accuracy', 'G1',
                             '--workdir', str(tmp_path)] + extra)
    assert abs(out['value_torch'] - out['value_prod']) < 1e-12
    assert out['worst_rel'] < 1e-10
