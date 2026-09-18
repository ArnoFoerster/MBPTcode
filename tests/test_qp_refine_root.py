import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Solvers.qp_equation import _refine_root, solve_qp_equation


def check(ok, label, detail=''):
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


class Counted:
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, w):
        self.n += 1
        return self.f(w)


def reference_refine(func, a, b, tol, max_bisection):
    """Reference bisection loop, evaluating f(a) every step (no caching)."""
    for _ in range(max_bisection):
        c = 0.5 * (a + b)
        if func(a) * func(c) <= 0.0:
            b = c
        else:
            a = c
        if abs(b - a) <= tol:
            break
    return 0.5 * (a + b)


if __name__ == '__main__':
    all_ok = True
    # f(w) = w - eps - Sigma(w) with one pole below the bracket: a plain
    # crossing near 0.3
    f = lambda w: w - 0.3 - 0.02 / (w + 0.5)
    for a, b in ((0.25, 0.35), (0.35, 0.25), (0.0, 1.0)):
        ref = Counted(f)
        new = Counted(f)
        r_ref = reference_refine(ref, a, b, 1e-8, 100)
        r_new = _refine_root(new, a, b, 1e-8, 100)
        all_ok &= check(
            r_new == r_ref, f'same root from [{a}, {b}]', f'{r_new:.12f}')
        all_ok &= check(
            new.n <= ref.n // 2 + 1, 'about half the evaluations',
            f'{new.n} vs {ref.n}')
    # end to end through the pole-strength solver, on a synthetic Sigma
    # with two poles
    sigma = lambda w: 0.05 / (w + 0.8) + 0.01 / (w - 0.9)
    g = Counted(lambda w: w + 0.2 - sigma(w))
    root = solve_qp_equation(g, -0.2, method='pole_strength')
    all_ok &= check(
        abs(g(root)) < 1e-7, 'pole_strength root satisfies f(root) ~ 0',
        f'{g(root):.1e}')
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
