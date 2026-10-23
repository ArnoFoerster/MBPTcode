"""Extending an orthonormal basis by candidate vectors (Gram-Schmidt, twice)."""
import numpy as np


def extend_orthonormal(B, cand, rtol=None, atol=None, limit=None):
    """Q (n, m), orthonormal and orthogonal to B (n, b, orthonormal): each
    candidate in turn (the columns of an (n, k) array, or a sequence of (n,)
    vectors) with its components along B and the columns kept before it
    removed twice, normalised, and kept while its norm stays above rtol times
    its own or above atol (one of the two must be given); at most limit
    columns."""
    if rtol is None and atol is None:
        raise ValueError('extend_orthonormal needs rtol or atol')
    if isinstance(cand, np.ndarray) and cand.ndim == 2:
        cand = cand.T
    Q = np.empty((B.shape[0], 0))
    for g in cand:
        r = g
        for _ in range(2):
            # r_p - sum_j B_pj (B r)_j - sum_j Q_pj (Q r)_j
            r = r - B @ (B.T @ r) - Q @ (Q.T @ r)
        nr = np.linalg.norm(r)
        if nr > (atol if atol is not None else rtol * np.linalg.norm(g)):
            Q = np.column_stack([Q, r / nr])
        if limit is not None and Q.shape[1] == limit:
            break
    return Q
