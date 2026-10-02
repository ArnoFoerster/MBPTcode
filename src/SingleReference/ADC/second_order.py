"""The second-order self-energy Sigma(2) (ADC(2), GF2) of a closed-shell
reference.

    Sigma(2)_pp(w) = sum_iab (pa|ib) [2 (pa|ib) - (pb|ia)] / (w + eps_i - eps_a - eps_b)
                   + sum_ija (pi|ja) [2 (pi|ja) - (pj|ia)] / (w + eps_a - eps_i - eps_j),

split into its direct part D (the 2|.|^2 terms) and its second-order exchange
X (the SOX terms), Sigma(2) = D + X. Methods that hold Sigma(2) once too often
remove it with these (FLEX: -2 Sigma(2) against the two pair terms; PSD2 -
Sigma(2), PSD1 - D/4 - X), so the pole lists come out separately and on the
same poles. Restricted, DF factor or dense chemist tensor; o v^2 + o^2 v per
orbital.
"""
import numpy as np

from src.Base.eri_blocks import g_slice


def sigma2_poles(eps, nocc, p, B_aa=None, eri_chemist=None):
    """(residues_D, residues_X, poles) of Sigma(2)_pp as flat pole lists:
    Sigma(2)_pp(w) = sum (residues_D + residues_X) / (w - poles).

    Parameters
    ----------
    eps : ndarray, shape (norb,)
        Orbital energies, the nocc doubly occupied first.
    nocc : int
    p : int
    B_aa : ndarray, shape (naux, norb, norb), optional
        DF factor, (pq|rs) = sum_Q B_aa[Q,p,q] B_aa[Q,r,s].
    eri_chemist : ndarray, shape (norb,)*4, optional
        Dense (pq|rs), used when B_aa is None.
    """
    eps = np.asarray(eps)
    o, v = slice(0, nocc), slice(nocc, len(eps))
    eo, ev = eps[o], eps[v]
    Vp = g_slice(B_aa, eri_chemist, [p], o, v, v)[0]              # (pa|ib) [i,a,b]
    Vh = g_slice(B_aa, eri_chemist, [p], o, o, v)[0].transpose(1, 0, 2)  # (pi|ja) [i,j,a]
    poles = np.concatenate([
        (ev[None, :, None] + ev[None, None, :] - eo[:, None, None]).ravel(),
        (eo[:, None, None] + eo[None, :, None] - ev[None, None, :]).ravel()])
    res_D = np.concatenate([(2 * Vp * Vp).ravel(), (2 * Vh * Vh).ravel()])
    res_X = np.concatenate([(-Vp * Vp.transpose(0, 2, 1)).ravel(),
                            (-Vh * Vh.transpose(1, 0, 2)).ravel()])
    return res_D, res_X, poles


def sigma2(eps, nocc, p, w, B_aa=None, eri_chemist=None):
    """(Sigma(2)_pp(w), dSigma(2)_pp/dw) at a real w."""
    res_D, res_X, poles = sigma2_poles(eps, nocc, p, B_aa, eri_chemist)
    d = 1.0 / (w - poles)
    r = res_D + res_X
    return float(r @ d), -float(r @ (d * d))
