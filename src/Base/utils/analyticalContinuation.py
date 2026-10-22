"""Thiele-Pade continuation of imaginary-axis data to the real axis: one node
rule (`pade_nodes`), greedy ordering and the continued fraction."""
import numpy as np

from src.Base.constants import PADE_NPADE, PADE_W0
from src.Base.utils.grids import gauss_legendre_grid


def pade_nodes(npade=PADE_NPADE, w0=PADE_W0):
    """The imaginary frequencies Sigma_c(i omega) is sampled at for its Pade
    continuation: npade Gauss-Legendre nodes of the half axis scaled by w0."""
    return gauss_legendre_grid(npade, w0=w0)[0]


def thiele_coefficients(z, f):
    """Thiele's reciprocal-difference recursion for the continued-fraction Pade coefficients a_i from (z_i, f_i)."""
    n = len(z)
    g = np.zeros((n, n), dtype=complex)
    g[:, 0] = f
    for i in range(1, n):
        g[i:, i] = (g[i - 1, i - 1] - g[i:, i - 1]) / ((z[i:] - z[i - 1]) * g[i:, i - 1])
    return g.diagonal().copy()


def pade_eval(z_query, z, a):
    """Evaluates the Thiele continued fraction built by thiele_coefficients at z_query."""
    n = len(a)
    result = a[-1] * np.ones_like(np.asarray(z_query, dtype=complex))
    for i in range(n - 2, -1, -1):
        result = a[i] / (1.0 + (z_query - z[i]) * result)
    return result


def greedy_pade_order(z, f):
    """Greedily order (z_i, f_i) for Thiele construction: naive fixed ordering can give huge, ill-conditioned a_i.

    Starting from the point closest to the real axis, repeatedly add whichever
    remaining point keeps the next Thiele coefficient smallest in magnitude.
    """
    n = len(z)
    remaining = list(range(n))
    start = int(np.argmin(np.abs(z)))
    order = [start]
    remaining.remove(start)

    while remaining:
        best_j, best_score = None, np.inf
        for j in remaining:
            trial_order = order + [j]
            zt, ft = z[trial_order], f[trial_order]
            g = np.zeros((len(trial_order), len(trial_order)), dtype=complex)
            g[:, 0] = ft
            for i in range(1, len(trial_order)):
                g[i:, i] = (g[i - 1, i - 1] - g[i:, i - 1]) / ((zt[i:] - zt[i - 1]) * g[i:, i - 1])
            score = abs(g[-1, -1])
            if np.isfinite(score) and score < best_score:
                best_score, best_j = score, j
        if best_j is None:
            raise FloatingPointError(
                f'greedy Pade ordering: every next Thiele coefficient of the '
                f'{len(remaining)} points left is non-finite after '
                f'{len(order)} nodes')
        order.append(best_j)
        remaining.remove(best_j)

    return np.array(order)


def pade_continue(z, f, z_query, greedy=True):
    """Fits a Thiele-Pade approximant to (z,f) and evaluates it at z_query."""
    z = np.asarray(z, dtype=complex)
    f = np.asarray(f, dtype=complex)
    if greedy:
        order = greedy_pade_order(z, f)
        z, f = z[order], f[order]
    a = thiele_coefficients(z, f)
    return pade_eval(z_query, z, a)


def thiele_coefficients_matrix(z, F):
    """Matrix generalization of thiele_coefficients: F is (n,d,d) samples; scalar division becomes a matrix inverse.

    NOT equivalent to fitting each matrix element independently -- the
    inversion couples all elements at every recursion step. Reduces to
    thiele_coefficients when d=1.
    """
    n, d, _ = F.shape
    G = np.zeros((n, n, d, d), dtype=complex)
    G[:, 0] = F
    I = np.eye(d, dtype=complex)
    for i in range(1, n):
        for k in range(i, n):
            diff = G[i - 1, i - 1] - G[k, i - 1]
            denom = (z[k] - z[i - 1]) * G[k, i - 1]
            G[k, i] = diff @ np.linalg.inv(denom)
    return np.array([G[i, i] for i in range(n)])


def pade_eval_matrix(z_query, z, a):
    """Evaluates the matrix Thiele continued fraction built by thiele_coefficients_matrix."""
    n, d, _ = a.shape
    I = np.eye(d, dtype=complex)
    z_query = np.atleast_1d(np.asarray(z_query, dtype=complex))
    result = np.tile(a[-1], (len(z_query), 1, 1))
    for i in range(n - 2, -1, -1):
        for iq, zq in enumerate(z_query):
            result[iq] = a[i] @ np.linalg.inv(I + (zq - z[i]) * result[iq])
    return result


def greedy_pade_order_matrix(z, F):
    """Matrix analog of greedy_pade_order: scores candidate orderings by the Frobenius
    norm of the next Thiele coefficient matrix instead of a scalar magnitude."""
    n = len(z)
    remaining = list(range(n))
    start = int(np.argmin(np.abs(z)))
    order = [start]
    remaining.remove(start)

    while remaining:
        best_j, best_score = None, np.inf
        for j in remaining:
            trial_order = order + [j]
            zt, Ft = z[trial_order], F[trial_order]
            try:
                g_last = thiele_coefficients_matrix(zt, Ft)[-1]
                score = np.linalg.norm(g_last)
            except np.linalg.LinAlgError:
                score = np.inf
            if np.isfinite(score) and score < best_score:
                best_score, best_j = score, j
        order.append(best_j)
        remaining.remove(best_j)

    return np.array(order)


def pade_continue_matrix(z, F, z_query, greedy=True):
    """Fits a matrix Thiele-Pade approximant to (z, F) and evaluates it at z_query."""
    z = np.asarray(z, dtype=complex)
    F = np.asarray(F, dtype=complex)
    if greedy:
        order = greedy_pade_order_matrix(z, F)
        z, F = z[order], F[order]
    a = thiele_coefficients_matrix(z, F)
    return pade_eval_matrix(z_query, z, a)
