"""pade_fit_elementwise / pade_eval_elementwise against the per-element
continuation they replace, and the streamed Tr[G(w)] of scGW against the one
built from the full (nfreq, nmo, nmo) Sigma(w)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.utils.grids.analytic_continuation import (
    _greedy_thiele, pade_eval, pade_continue_elementwise, pade_fit_elementwise,
    pade_eval_elementwise)
from src.SingleReference.GW.scgw import build_Greal_from_F, trace_Greal_from_pade

D = 6
MAX_ORDER = 10


def sigma_model(z):
    """A symmetric (len(z), D, D) sum of poles; the last orbital decouples
    from the rest, so its off-diagonal elements are exactly zero."""
    rng = np.random.default_rng(0)
    poles = np.array([-1.3, -0.7, 0.4, 1.1, 2.0])
    v = rng.normal(size=(len(poles), D))
    v[:, -1] = 0.0
    v[2, -1] = 1.0
    v[2, :-1] = 0.0
    return np.einsum('kp,kq,zk->zpq', v, v, 0.3 / (z[:, None] - poles))


def fit_points():
    z = 0.05 + 1j * np.linspace(0.05, 6.0, 20)
    return z, sigma_model(z)


def reference_elementwise(z, F, z_query, symmetric):
    """The element-by-element continuation pade_continue_elementwise used to do."""
    floor = 1e-10 * np.abs(F).max()
    out = np.zeros((len(z_query), D, D), dtype=complex)
    for p in range(D):
        for q in range(p if symmetric else 0, D):
            f = F[:, p, q]
            if np.abs(f).max() <= floor:
                continue
            order, a = _greedy_thiele(z, f)
            out[:, p, q] = pade_eval(z_query, z[order[:MAX_ORDER]], a[:MAX_ORDER])
            if symmetric:
                out[:, q, p] = out[:, p, q]
    return out


def test_wrapper_matches_per_element_continuation():
    z, F = fit_points()
    z_query = np.linspace(-2.0, 3.0, 37) + 1e-2j
    for symmetric in (True, False):
        new = pade_continue_elementwise(z, F, z_query, symmetric=symmetric,
                                        max_order=MAX_ORDER)
        ref = reference_elementwise(z, F, z_query, symmetric)
        np.testing.assert_allclose(new, ref, rtol=1e-12, atol=1e-14)


def test_scalar_eval_is_one_row_and_floored_elements_are_zero():
    z, F = fit_points()
    Z, A = pade_fit_elementwise(z, F, symmetric=True, max_order=MAX_ORDER)
    z_query = np.linspace(-2.0, 3.0, 9) + 1e-2j
    full = pade_eval_elementwise(z_query, Z, A)
    for k, zk in enumerate(z_query):
        one = pade_eval_elementwise(zk, Z, A)
        assert one.shape == (D, D)
        np.testing.assert_allclose(one, full[k], rtol=1e-14, atol=0)
    assert np.all(full[:, -1, :-1] == 0) and np.all(full[:, :-1, -1] == 0)


def test_streamed_trace_matches_full_sigma():
    z, F_samples = fit_points()
    Z, A = pade_fit_elementwise(z, F_samples, symmetric=True, max_order=MAX_ORDER)
    rng = np.random.default_rng(1)
    F = rng.normal(size=(D, D))
    F = 0.5 * (F + F.T)
    eta = 1e-2
    w = np.linspace(-3.0, 3.0, 101)

    sigma_real = pade_eval_elementwise(w + 1j*eta, Z, A)
    ref = build_Greal_from_F(w, sigma_real, F, eta, trace_only=True)
    # Chunk sizes that do and do not divide len(w), and one larger than it.
    for chunk in (1, 7, 16, 500):
        np.testing.assert_allclose(trace_Greal_from_pade(w, Z, A, F, eta, chunk=chunk),
                                   ref, rtol=1e-11, atol=1e-13)
