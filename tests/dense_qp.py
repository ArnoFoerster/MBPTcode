"""The dense check of a quasi-boson quasiparticle root, for tests only."""
import numpy as np


def dense_max_weight_root(qp, p):
    """(w, Z): the eigenvalue of `qp.supermatrix_diag(p)` of largest weight on
    the bare orbital, by a dense eigensolve: what `QPqb.quasiparticle_root`
    finds from the poles, checked the expensive way."""
    w, v = np.linalg.eigh(qp.supermatrix_diag(p))
    k = np.argmax(v[0, :] ** 2)
    return w[k], v[0, k] ** 2
