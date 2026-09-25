"""Hand-written spin-orbital orchestration: dense eigh or root-following
Davidson on the adc_u_* route modules; the EN-dressed t2 hook is threaded
here. Returns (e, Z) with details on s.last_result."""
import numpy as np

from src.SingleReference.ADC import adc_u_dense_full, adc_u_dense_df
from src.SingleReference.ADC import adc_u_sigma_full, adc_u_sigma_df
from src.SingleReference.ADC.adc_u_utils import dressed_t2_amplitudes
from src.SingleReference.ADC.solve import davidson_follow, diag_dense


def solve(s, static_correction=None, nroots=1, homo_index=None, ref_vec=None,
          conv_tol=1e-6, threshold=5000, verbose=0):
    nocc = s.nocc
    if not s.matrix_free:
        mod = adc_u_dense_df if s.g is None else adc_u_dense_full
        H = mod.build_supermatrix(s, nocc, static_correction)
        e, Z, vec = diag_dense(H, s.norb, threshold=threshold)
        # dense: eigh either returns every root or raises, so this one
        # really is all-True rather than a stand-in for an unread flag
        s.last_result = {'vec': vec, 'converged': np.ones_like(e, dtype=bool)}
        return e, Z

    t2 = None
    if s.en_dress is not None:
        t2 = dressed_t2_amplitudes(s, nocc, s.en_dress)
    mod = adc_u_sigma_df if s.B_spin is not None else adc_u_sigma_full
    aop, diag, dims = mod.build_operator(s, nocc, static_correction, t2_ijcd=t2)
    homo = homo_index if homo_index is not None else nocc - 1
    e, Z, vec, conv = davidson_follow(aop, diag, dims['nH'], s.norb, homo,
                                      ref_vec, nroots, conv_tol=conv_tol,
                                      verbose=verbose)
    # the solver's own flag, not a placeholder: a Davidson that ran out of
    # cycles still returns an energy for every root it was asked for
    s.last_result = {'vec': vec, 'converged': conv}
    return e, Z
