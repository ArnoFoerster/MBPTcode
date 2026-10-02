"""Hand-written restricted orchestration: pick the route module from the
solver's flags, run dense eigh (benchmarking) or root-following Davidson
(production), return (e, Z) with details on s.last_result."""
import numpy as np

from src.SingleReference.ADC.solve import davidson_follow, diag_dense


def solve(s, static_correction=None, nroots=1, homo_index=None, ref_vec=None,
          conv_tol=1e-6, threshold=5000, verbose=0):
    nocc = s.nocc
    if not s.matrix_free:
        H = s.build_supermatrix(nocc, static_correction)
        e, Z, vec = diag_dense(H, s.norb, threshold=threshold)
        # dense: eigh either returns every root or raises, so this one
        # really is all-True rather than a stand-in for an unread flag
        s.last_result = {'vec': vec, 'converged': np.ones_like(e, dtype=bool)}
        return e, Z

    aop, diag, dims = s.build_matrix_free_operator(nocc, static_correction)
    homo = homo_index if homo_index is not None else nocc - 1
    e, Z, vec, conv = davidson_follow(aop, diag, dims['nH'], s.norb, homo,
                                      ref_vec, nroots, conv_tol=conv_tol,
                                      verbose=verbose)
    # the solver's own flag, not a placeholder: a Davidson that ran out of
    # cycles still returns an energy for every root it was asked for
    s.last_result = {'vec': vec, 'converged': conv}
    return e, Z
