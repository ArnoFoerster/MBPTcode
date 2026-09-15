import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Solvers.qp_equation import solve_qp_equation


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" +
          (f'   ({detail})' if detail else ''))
    return bool(ok)


class Counted:
    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, w):
        self.calls += 1
        return self.f(w)


if __name__ == '__main__':
    all_ok = True
    sigma = lambda w: 0.05 / (w + 0.8) + 0.01 / (w - 0.9)      # array-safe
    f = lambda w: w + 0.2 - sigma(w)
    for method in ('pole_strength', 'graphical'):
        s = Counted(f)
        v = Counted(f)
        r_s = solve_qp_equation(s, -0.2, method=method)
        r_v = solve_qp_equation(v, -0.2, method=method, vectorized=True)
        all_ok &= check(r_s == r_v, f'{method}: same root, scalar vs vectorized grid',
                         f'{r_v:.12f}')
        all_ok &= check(v.calls < s.calls - 100, f'{method}: grid took one call',
                         f'{v.calls} vs {s.calls}')
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
