import numpy as np


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
                keeps all of them). Same meaning as in qp_solve.solve_qp_from_imaginary_axis.

    Returns (nq,d,d). Unlike pade_continue_matrix the elements are not coupled, and
    each one gets its own greedy ordering.
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
