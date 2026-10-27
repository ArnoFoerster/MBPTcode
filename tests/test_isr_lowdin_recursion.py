"""isr_engine's order-by-order Loewdin S^{-1/2} recursion and the (K+C) / U
convolutions built on it: pure linear algebra, no generator or ADC physics.

Primary check: the closed-form Taylor coefficients of (1+X)^{-1/2} through
order 4 from the binomial series c_n = (-1)^n (2n-1)!!/(2^n n!) (1, -1/2, 3/8,
-5/16, 35/128), collecting at each order m every ordered composition of m into
n parts -- a textbook formula, not isr_engine's own recursion.

Second, differently implemented check: the exact S(lambda)^{-1/2} (scipy's
principal fractional power, analytic near S = 1) and the exactly
orthogonalized Y^T (K+C) Y and U Y, sampled on a complex-lambda circle; the
FFT gives their Taylor coefficients through order 4 to rounding.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.linalg import fractional_matrix_power

from src.SingleReference.ADC.isr_engine import (
    assemble_kplusc, assemble_u, lowdin_sqrt_inv_series)

DIM = 4
M = 4
TOL = 1e-12
RADIUS = 0.3
N_POINTS = 32


def _random_symmetric(rng, dim):
    a = rng.standard_normal((dim, dim)) * 0.05
    return a + a.T


def _exact_y_through_4(ds):
    """Closed-form Y^(0..4) from (1+X)^{-1/2} = sum_n c_n X^n with
    X = sum_m lambda^m ds[m], order-m terms summed over every ordered
    composition of m into n parts >= 1 (matrix products, order kept)."""
    c = {0: 1.0, 1: -0.5, 2: 0.375, 3: -0.3125, 4: 0.2734375}

    def compositions(total, parts):
        if parts == 1:
            yield (total,)
            return
        for first in range(1, total - parts + 2):
            for rest in compositions(total - first, parts - 1):
                yield (first,) + rest

    dim = ds[1].shape[0]
    y = {0: np.eye(dim)}
    for m in range(1, 5):
        acc = np.zeros((dim, dim))
        for n in range(1, m + 1):
            for comp in compositions(m, n):
                prod = np.eye(dim)
                for part in comp:
                    prod = prod @ ds[part]
                acc = acc + c[n] * prod
        y[m] = acc
    return y


def _taylor(fn, orders):
    """Taylor coefficients of an analytic matrix function of lambda."""
    lams = RADIUS * np.exp(2j * np.pi * np.arange(N_POINTS) / N_POINTS)
    coefs = np.fft.fft(np.array([fn(lam) for lam in lams]), axis=0) / N_POINTS
    return {n: coefs[n] / RADIUS ** n for n in orders}


def _series(coeffs, lam):
    return sum(lam ** m * a for m, a in coeffs.items())


def test_lowdin_recursion_equals_the_binomial_series():
    rng = np.random.default_rng(0)
    delta_s = {m: _random_symmetric(rng, DIM) for m in range(1, M + 1)}
    y = lowdin_sqrt_inv_series(delta_s, M)
    ref = _exact_y_through_4(delta_s)
    for m in range(M + 1):
        d = float(np.max(np.abs(y[m] - ref[m])))
        print(f"Y^({m}) recursion vs closed form: {d:.1e}")
        assert d < TOL, (m, d)


def test_lowdin_and_convolutions_equal_the_exact_orthogonalization():
    rng = np.random.default_rng(1)
    delta_s = {m: _random_symmetric(rng, DIM) for m in range(1, M + 1)}
    kplusc_raw = {m: _random_symmetric(rng, DIM) for m in range(M + 1)}
    u_raw = {m: rng.standard_normal((3, DIM)) * 0.05 for m in range(M + 1)}
    y = lowdin_sqrt_inv_series(delta_s, M)

    def y_exact(lam):
        return fractional_matrix_power(np.eye(DIM) + _series(delta_s, lam), -0.5)

    orders = range(M + 1)
    y_ref = _taylor(y_exact, orders)
    kc_ref = _taylor(lambda lam: y_exact(lam).T @ _series(kplusc_raw, lam) @ y_exact(lam),
                     orders)
    u_ref = _taylor(lambda lam: _series(u_raw, lam) @ y_exact(lam), orders)
    for n in orders:
        dy = float(np.max(np.abs(y[n] - y_ref[n])))
        dk = float(np.max(np.abs(assemble_kplusc(kplusc_raw, y, n) - kc_ref[n])))
        du = float(np.max(np.abs(assemble_u(u_raw, y, n) - u_ref[n])))
        print(f"order {n}: Y {dy:.1e}, (K+C) {dk:.1e}, U {du:.1e} against the exact series")
        assert max(dy, dk, du) < TOL, (n, dy, dk, du)
