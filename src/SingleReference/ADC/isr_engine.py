"""Order-by-order ISR assembly engine of the generated ADC(n) runtime.

Convention: a "block series" is a dict {order: numpy array}, order=0,1,2,...,
representing a perturbative expansion Q(lambda) = sum_m lambda^m Q^(m). All
functions here operate on such dicts. S^(0) is required to be the identity
(precursor classes are orthonormal at zeroth order -- distinct Slater
determinants are exactly orthogonal, same-class self-overlap is 1), so
Delta S = S - 1 has no order-0 piece (Delta S^(0) does not appear as a key).
"""
import numpy as np


def s_inverse_series(delta_s, n):
    """Order-by-order coefficients of S(lambda)^{-1} = (1+Delta S(lambda))^{-1},
    via the standard Neumann-series recursion Z^(0)=I,
    Z^(m) = -sum_{p=1}^{m} Delta S^(p) @ Z^(m-p) for m=1..n.
    """
    dim = next(iter(delta_s.values())).shape[0] if delta_s else None
    z = {}
    for m in range(0, n + 1):
        if m == 0:
            z[0] = np.eye(dim) if dim is not None else None
            continue
        acc = np.zeros_like(z[0])
        for p in range(1, m + 1):
            ds_p = delta_s.get(p)
            if ds_p is None:
                continue
            acc = acc + ds_p @ z[m - p]
        z[m] = -acc
    return z


def lowdin_sqrt_inv_series(delta_s, n):
    """Order-by-order coefficients of S(lambda)^{-1/2}, via
    Y^(0)=I, Y^(m) = (1/2)[Z^(m) - sum_{p=1}^{m-1} Y^(p) @ Y^(m-p)], m=1..n,
    where Z=s_inverse_series(delta_s, n). This is the generic Loewdin
    binomial-series recursion (1+X)^{-1/2} = 1 - X/2 + 3/8 X^2 - ...,
    applied order-by-order in the perturbation parameter rather than
    treated as a single small-X expansion.
    """
    z = s_inverse_series(delta_s, n)
    dim = z[0].shape[0]
    y = {0: np.eye(dim)}
    for m in range(1, n + 1):
        acc = np.zeros((dim, dim))
        for p in range(1, m):
            acc = acc + y[p] @ y[m - p]
        y[m] = 0.5 * (z[m] - acc)
    return y


def assemble_kplusc(kplusc_raw, y_series, n):
    """(K+C)^(n)_orthonormal = sum_{p+q+r=n} Y^(p).T @ (K+C)^(q) @ Y^(r),
    the ISR-orthogonalized secular-matrix block at order n, built purely
    from already-computed raw (precursor-basis) (K+C)^(m) blocks and the
    Loewdin Y^(m)=S^{-1/2 (m)} coefficients from lowdin_sqrt_inv_series.
    """
    out = None
    for p in range(0, n + 1):
        for q in range(0, n - p + 1):
            r = n - p - q
            y_p, kc_q, y_r = y_series.get(p), kplusc_raw.get(q), y_series.get(r)
            if y_p is None or kc_q is None or y_r is None:
                continue
            term = y_p.T @ kc_q @ y_r
            out = term if out is None else out + term
    return out


def assemble_u(u_raw, y_series, n):
    """U^(n)_orthonormal = sum_{p+q=n} U^(q) @ Y^(p), the ISR-orthogonalized
    coupling-vector block at order n."""
    out = None
    for p in range(0, n + 1):
        q = n - p
        y_p, u_q = y_series.get(p), u_raw.get(q)
        if y_p is None or u_q is None:
            continue
        term = u_q @ y_p
        out = term if out is None else out + term
    return out
