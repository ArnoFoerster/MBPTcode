import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.utils.grids.analytic_continuation import (
    PadeFit, thiele_coefficients, pade_eval, greedy_pade_order)

if __name__ == '__main__':
    all_ok = True

    # A Pade approximant built from exact samples of a rational function must
    # reproduce that function exactly (up to floating-point roundoff) at any
    # other point -- the strongest correctness check available for this code.
    def f(z):
        return 1.0 / (z - 2.0) + 1.0 / (z + 3.0) - 0.5

    z_fit = np.array([0.1j, 0.5j, 1.0j, 2.0j, 3.0j, 5.0j, 8.0j, 12.0j], dtype=complex)
    f_fit = f(z_fit)

    a = thiele_coefficients(z_fit, f_fit)
    z_query = np.array([0.3j, 4.0j, 10.0j, 1.0 + 0.5j])
    got = pade_eval(z_query, z_fit, a)
    expected = f(z_query)
    ok = np.allclose(got, expected, atol=1e-6)
    all_ok &= ok
    print(f"thiele/pade_eval reproduces exact rational function: {'OK' if ok else 'FAIL'} (max err={np.max(np.abs(got-expected)):.2e})")

    got_greedy = PadeFit(z_fit, f_fit, greedy=True)(z_query)
    ok = np.allclose(got_greedy, expected, atol=1e-6)
    all_ok &= ok
    print(f"PadeFit(greedy=True) reproduces exact rational function: {'OK' if ok else 'FAIL'} (max err={np.max(np.abs(got_greedy-expected)):.2e})")

    got_naive = PadeFit(z_fit, f_fit, greedy=False)(z_query)
    ok = np.allclose(got_naive, expected, atol=1e-6)
    all_ok &= ok
    print(f"PadeFit(greedy=False) reproduces exact rational function: {'OK' if ok else 'FAIL'} (max err={np.max(np.abs(got_naive-expected)):.2e})")

    order = greedy_pade_order(z_fit, f_fit)
    ok = sorted(order.tolist()) == list(range(len(z_fit)))
    all_ok &= ok
    print(f"greedy_pade_order returns a valid permutation: {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
