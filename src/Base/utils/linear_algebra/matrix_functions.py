"""Functions of a real symmetric matrix through its eigendecomposition."""
import numpy as np


def eigh_sym(M):
    return np.linalg.eigh(0.5 * (M + M.T))


def funm_sym(M, f, eig=None):
    """f(M) for symmetric M via eigh. `eig` reuses a precomputed (w, V)."""
    w, V = eigh_sym(M) if eig is None else eig
    return (V * f(w)[None, :]) @ V.T


def sqrtm_sym(M, eig=None):
    return funm_sym(M, np.sqrt, eig)


def invsqrtm_sym(M, eig=None):
    return funm_sym(M, lambda w: 1.0 / np.sqrt(w), eig)
