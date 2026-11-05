"""The particle-particle ladder through a DF factor (common.df_ladder).

Checks:

  1. both evaluations ('aux', 'slab') of several Y at once equal the explicit
     sum_cd (ac|bd) Y_ijcd to 2e-15 relative, with equal and unequal left and
     right virtual counts (the UHF spin cases) and slabs narrower than the
     virtual space;
  2. 'auto' takes 'slab' for many rows and 'aux' for few rows and many virtuals;
  3. an unknown kernel name is refused.

Run: python tests/test_df_ladder.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.SingleReference.common import df_ladder
from tests.adc_common import check


def check_values():
    """Check 1."""
    ok = True
    rng = np.random.default_rng(3)
    for o, va, vb, naux in ((3, 7, 7, 20), (4, 9, 6, 25), (5, 12, 12, 30)):
        Bl = rng.standard_normal((naux, va, va))
        Br = rng.standard_normal((naux, vb, vb))
        Ys = [rng.standard_normal((o, o, va, vb)) for _ in range(3)]
        ref = [np.einsum('Qac,Qbd,ijcd->ijab', Bl, Br, Y, optimize=True) for Y in Ys]
        for kernel in ('aux', 'slab'):
            out = df_ladder.ladders(Bl, Br, Ys, mem_gb=1e-5, kernel=kernel)
            err = max(float(np.abs(a - b).max() / np.abs(a).max()) for a, b in zip(ref, out))
            ok &= check(err < 2e-15 and all(b.shape == a.shape for a, b in zip(ref, out)),
                        f"{kernel}: o {o}, v {va}/{vb}: the explicit ladder", f'{err:.1e}')
    return ok


def check_choice():
    """Checks 2 and 3."""
    ok = check(df_ladder.choose_kernel(672, 3 * 24 ** 2, 146, 146) == 'slab',
               'many rows take the slab evaluation')
    ok &= check(df_ladder.choose_kernel(300, 4 ** 2, 200, 200) == 'aux',
                'few rows and many virtuals take the auxiliary chunks')
    try:
        df_ladder.choose_kernel(10, 10, 10, 10, 'vvvv')
        ok &= check(False, 'an unknown kernel is refused')
    except ValueError as exc:
        ok &= check(True, 'an unknown kernel is refused', str(exc)[:50])
    return ok


def run():
    """Every check; True when all passed."""
    ok = check_values()
    ok &= check_choice()
    return ok


def test_df_ladder():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
