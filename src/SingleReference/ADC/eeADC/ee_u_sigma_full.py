"""Matrix-free spin-orbital EE-ADC sigma operator (dense-g route), the
Davidson counterpart of ee_u_dense_full -- mirroring adc_u_sigma_full's role
on the IP/EA side.

Implements the paper's matrix-vector product equations DIRECTLY, i.e. the
ph-row equations A41/A42 rather than the transpose of the 2p2h-row block the
dense module builds. The two are therefore independent transcriptions of the
same Hermitian operator, and tests/test_ee_adc_spinorbital_sigma.py checks
sigma against the dense supermatrix column by column -- which is also what
verifies the doubles metric convention end to end.

Vector layout (external): singles (i, a) then doubles on the UNIQUE
configurations i<j, a<b, exactly as ee_u_dense_full. Internally the doubles
segment is expanded to the paper's full antisymmetric tensor with

    Y = unfold_doubles(y_D) / 2 ,   w_D = 2 * fold_doubles(W_D)

(ee_utils.PAPER_DOUBLES_SCALE). Both factors are applied in _apply, so every
equation below can be transcribed verbatim from the paper.

Every intermediate is at most four-index, as required for the production
path: the amplitude-dressed couplings factor through the two-index u_ac /
w_ki carriers, and the ph/ph block is precomputed once (A29/A30 -- N^6 at
ADC(3), the same cost as evaluating it on the fly, per the paper's own
scaling discussion).
"""
import numpy as np

from src.SingleReference.CC.cached_einsum import einsum as _einsum
from src.SingleReference.ADC.eeADC import ee_utils, ee_u_dense_full as _dense
from src.SingleReference.ADC.eeADC import ee_equations as _eq

LEVELS = _dense.LEVELS


def build_operator(eps, g, nocc, level='adc3', en_dress=None):
    """(aop, diag, dims) for a Davidson solve. aop takes and returns a flat
    (nH,) vector in the layout described in the module docstring."""
    if level not in LEVELS:
        raise ValueError(f"level={level!r}; expected one of {LEVELS}")
    norb = len(eps)
    o_ss, o_sd, o_dd = _dense._BLOCK_ORDERS[level]
    dims = ee_utils.dimensions(nocc, norb)
    n_s, n_d = dims['n_s'], dims['n_d']

    amps, zint, rho = _dense._ingredients(eps, g, nocc, norb, level,
                                          None, None, None, en_dress)
    gb = ee_utils.g_blocks(g, nocc, norb)
    m_ss = _dense.m_ss(eps, g, nocc, norb, o_ss, amps, zint, rho).reshape(n_s, n_s)

    if o_sd is None:                                    # ADC(1): singles only
        return (lambda v: m_ss @ v), np.diag(m_ss), dims

    d_ijab, _, _ = ee_utils._denominators(eps, nocc)
    I, J, A, B = ee_utils.configs_doubles(nocc, norb)
    d_diag = eps[A] + eps[B] - eps[I] - eps[J]
    if o_dd >= 1:
        # A63 diagonal: <ab||ab> + <ij||ij> - the four surviving ring terms
        d_diag = d_diag + (g[A, B, A, B] + g[I, J, I, J]
                           - g[I, A, I, A] - g[J, B, J, B]
                           - g[J, A, J, A] - g[I, B, I, B])
    diag = np.concatenate([np.diag(m_ss), d_diag])

    def aop(v):
        v = np.asarray(v).ravel()
        y1 = v[:n_s].reshape(nocc, norb - nocc)
        Y = ee_utils.unfold_doubles(v[n_s:], nocc, norb) / ee_utils.PAPER_DOUBLES_SCALE
        w1 = m_ss @ v[:n_s] + _eq.sigma_s_from_d(
            _eq.NUMPY, gb, amps, zint, Y, o_sd).ravel()
        W2 = (_eq.sigma_d_from_s(_eq.NUMPY, gb, amps, zint, y1, o_sd)
              + _eq.sigma_d_from_d(_eq.NUMPY, gb, Y, d_ijab, o_dd))
        w2 = ee_utils.PAPER_DOUBLES_SCALE * ee_utils.fold_doubles(W2, nocc, norb)
        return np.concatenate([w1, w2])

    return aop, diag, dims
