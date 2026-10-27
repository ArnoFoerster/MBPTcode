"""Thiele-Pade continuation of imaginary-axis data to the real axis: one node
rule (`pade_nodes`), greedy ordering and the continued fraction."""
import numpy as np

from src.Base.utils.grids.grids import gauss_legendre_grid


def pade_nodes(quadrature):
    """The imaginary frequencies Sigma_c(i omega) is sampled at for its Pade
    continuation: `quadrature.npade` Gauss-Legendre nodes of the half axis
    scaled by `quadrature.w0_gl` (a run's `QuadratureNumerics`)."""
    return gauss_legendre_grid(quadrature.npade, w0=quadrature.w0_gl)[0]


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
            score = abs(thiele_coefficients(z[trial_order], f[trial_order])[-1])
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


class PadeFit:
    """The Thiele-Pade continued fraction through the points (z_i, f_i):
    ordered greedily (`greedy_pade_order`) unless greedy is False, cut to the
    first `max_order` points after that ordering (None keeps all), its
    coefficients formed once (`thiele_coefficients`). Called at z_query, it
    evaluates the fraction there (`pade_eval`); `z` and `f` are the points it
    was built on, in their order."""

    def __init__(self, z, f, greedy=True, max_order=None):
        if greedy:
            order = greedy_pade_order(z, f)
            z, f = z[order], f[order]
        if max_order is not None and max_order < len(z):
            # Beyond what the data resolve (a finite temperature's few
            # independent structures) more nodes fit noise; cutting AFTER the
            # greedy ordering keeps the most informative points.
            if max_order < 2:
                raise ValueError(
                    f"max_order = {max_order} leaves too few points for a Pade "
                    f"fit; at least two are needed.")
            z, f = z[:max_order], f[:max_order]
        self.z, self.f = z, f
        self.coefficients = thiele_coefficients(z, f)

    def __call__(self, z_query):
        return pade_eval(z_query, self.z, self.coefficients)


def _greedy_thiele(z, f):
    """greedy_pade_order and thiele_coefficients in one O(n^2) pass. Returns (order, a).

    Same ordering and coefficients as the two separate calls: a candidate's next
    Thiele coefficient only needs its own row of the reciprocal-difference table,
    so every remaining row is advanced one column per added node instead of
    rebuilding the whole table for each candidate.
    """
    n = len(z)
    row = np.array(f, dtype=complex)
    remaining = np.ones(n, dtype=bool)
    order = np.empty(n, dtype=int)
    a = np.empty(n, dtype=complex)

    last = int(np.argmin(np.abs(z)))
    for i in range(n):
        if i > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                row[remaining] = (a[i - 1] - row[remaining]) / ((z[remaining] - z[last]) * row[remaining])
            score = np.where(remaining & np.isfinite(row), np.abs(row), np.inf)
            last = int(np.argmin(score))
            if not np.isfinite(score[last]):
                raise FloatingPointError(
                    f'greedy Pade ordering: every next Thiele coefficient of the '
                    f'{n - i} points left is non-finite after {i} nodes')
        order[i], a[i] = last, row[last]
        remaining[last] = False

    return order, a


def pade_continue_elementwise(z, F, z_query, greedy=True, symmetric=False, rel_tol=1e-10,
                              max_order=None):
    """Thiele-Pade continuation of every element of a matrix-valued function independently.

    z         : (n,) sample points
    F         : (n,d,d) samples
    z_query   : (nq,) points to evaluate at
    symmetric : F[k] = F[k].T for every k, so only the upper triangle is fitted.
    rel_tol   : elements whose samples all stay below rel_tol*max|F| are returned as
                exactly zero. Thiele divides by the samples, so a symmetry-forbidden
                element would otherwise be a fit to round-off.
    max_order : cap on the number of Pade nodes, applied after the ordering (None
                keeps all of them). Same meaning as in PadeFit.

    Returns (nq,d,d). Each element gets its own greedy ordering, as a PadeFit per
    element would, but through the single-pass _greedy_thiele.
    """
    z = np.asarray(z, dtype=complex)
    F = np.asarray(F, dtype=complex)
    z_query = np.atleast_1d(np.asarray(z_query, dtype=complex))
    d = F.shape[-1]
    out = np.zeros((len(z_query), d, d), dtype=complex)
    floor = rel_tol * np.abs(F).max()
    n_nodes = len(z) if max_order is None else min(max_order, len(z))

    for p in range(d):
        for q in range(p if symmetric else 0, d):
            f = F[:, p, q]
            if np.abs(f).max() <= floor:
                continue
            if greedy:
                order, a = _greedy_thiele(z, f)
                # Thiele coefficients are nested, so truncating them is the lower-order fit.
                out[:, p, q] = pade_eval(z_query, z[order[:n_nodes]], a[:n_nodes])
            else:
                out[:, p, q] = pade_eval(z_query, z[:n_nodes], thiele_coefficients(z[:n_nodes], f[:n_nodes]))
            if symmetric:
                out[:, q, p] = out[:, p, q]

    return out
