"""Eq 71 of Monino and Loos 2023 as the signed `pieces` of the EE fold (level 'dbse2').

The dynamical second-order kernel Θ(ω) of the BSE@GW lives on the ordered 2h2p doubles
K = (k, l, c, d) with D_K = ε^GW_c + ε^GW_d - ε^GW_k - ε^GW_l and the W lines of eq 73,
the static screened interaction in DF form, w(pq|rs) = sum_P B~_P,pq B~_P,rs
(dressed_factors). Four couplings take the pair (i, a) to the doubles,

    I_ia,K = δ_ac w(ik|ld)    J_ia,K = δ_ad w(il|kc)
    K_ia,K = δ_ik w(ac|ld)    L_ia,K = δ_il w(ad|kc)

and with X.Y := sum_K X_ia,K Y_jb,K / (ω - D_K) the two kernels are (coupling_pairs)

    'eq71'     Θ = I.J + K.L - I.L - L.I     X = I - K,  Y = J - L   (Yamada's kernel)
    'printed'  Θ = I.J + K.L - I.K - K.I     X = I - L,  Y = J - K   (eq 74 as printed)

each equal to (X.Y + Y.X)/2 since I.L + L.I = J.K + K.J and I.K = J.L, K.I = L.J. So Θ
folds exactly as two copies on one diagonal with the metric η = diag(+1, -1),
V_± = (X ± Y)/2:

    A_eff(ω) = M - sum_K V+_K,ia V+_K,jb / (D_K - ω) + sum_K V-_K,ia V-_K,jb / (D_K - ω)
    dλ/dω    = -|Z+|² + |Z-|²,    Z±_K = sum_jb V±_K,jb y_jb / (D_K - ω) ,

pieces['Vt'] is the η-transpose w_ia = sum_K V+_K,ia Z+_K - sum_K V-_K,ia Z-_K and
pieces['dnorm2'] is |Z+|² - |Z-|²; T1 = 1/(1 - dλ/dω) may leave (0, 1] (the driver
guards it). M is eq 62 at the GW energies, the singlet
A_ia,jb = (ε^GW_a - ε^GW_i) δ_ij δ_ab + 2 (ia|jb) - (ij|ab) and the triplet -(ij|ab);
singles='eq62+eq59' puts eq 59's static w(ij|ab) in place of (ij|ab), the full static
BSE@GW singles block (the milestone 1 check against BSE2@GW, not a level).

Spin. Every W line conserves spin along itself, so for the singles block of spin s1 the
vertices I and K put their image in the doubles blocks whose pair (k, c) carries s1, J
and L in those whose pair (l, d) carries s1, the other pair running over both spins. The
doubles are the four SB blocks 'aaaa', 'bbbb', 'abab', 'baba' of shape (no, no, nv, nv),
index order (k, l, c, d), the spin string in that order; each copy is one set of them,
keyed '+aaaa', '-aaaa', ..., and all eight share D. The singlet/triplet projection acts
on the singles, as at every level. Eq 71's Θ has no triplet part, every product of it
pairing an (I, K) vertex with a (J, L) one (coupling_pairs('eq71', 'triplet') is
empty); the printed Θ's triplet carries its loop terms I.K + K.I. A later class of the
second-order block enters as a pair of its own, its vertex type added to VERTICES.

References
----------
E. Monino and P.-F. Loos, J. Chem. Phys. 159, 034105 (2023), eqs 59, 62, 71 to 74.
A. Yamada et al., Phys. Rev. B 106, 045113 (2022), eqs 12 and 13.
"""
import numpy as np

from src.SingleReference.ADC.eeADC import ee_equations as _eq
from src.SingleReference.ADC.eeADC import ee_r_sigma_df
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB
from src.SingleReference.LinearResponse.linear_response import (
    static_screened_coulomb_aux)

VERTICES = ('I', 'J', 'K', 'L')
KERNELS = ('eq71', 'printed')
SINGLES = ('eq62', 'eq62+eq59')
SETS = ('eq71',)
_CHANNELS = ('singlet', 'triplet')
_SQRT2 = np.sqrt(2.0)
# the pair (x, y) of each kernel over VERTICES: with X = sum_A x_A A the copies
# V± = (X ± Y)/√2 fold to X.Y + Y.X, so the 1/√2 inside x and y gives the (X.Y + Y.X)/2
# of the module docstring
_PAIRS = {'eq71': (np.array([1.0, 0.0, -1.0, 0.0]) / _SQRT2,
                   np.array([0.0, 1.0, 0.0, -1.0]) / _SQRT2),
          'printed': (np.array([1.0, 0.0, 0.0, -1.0]) / _SQRT2,
                      np.array([0.0, 1.0, -1.0, 0.0]) / _SQRT2)}
# kernels without a triplet term: every product mixes an (I, K) and a (J, L) vertex,
# whose aa,aa and aa,bb blocks are equal
_TRIPLET_FREE = ('eq71',)
# a channel of up to 40 singles builds A_eff densely, as the gw builder
_DENSE_LIMIT = 40


def dressed_factors(eps_hf, B, nocc):
    """L and B~ of the static screened interaction, w(pq|rs) = sum_P B~_P,pq B~_P,rs.

    Parameters
    ----------
    eps_hf : ndarray, shape (norb,), HF orbital energies ε_p, Hartree, the reference
        the W lines are screened at
    B : ndarray, shape (naux, norb, norb), index order (Q, p, q), DF factors with
        (pq|rs) = sum_Q B_Q,pq B_Q,rs
    nocc : int, doubly occupied orbitals

    Returns
    -------
    L : ndarray, shape (naux, naux), index order (Q, P): the Cholesky factor of the
        static RPA metric W_QQ' of static_screened_coulomb_aux, W_QQ' = sum_P L_QP L_Q'P
    Bt : ndarray, shape (naux, norb, norb), index order (P, p, q),
        B~_P,pq = sum_Q L_QP B_Q,pq, so that sum_P B~_P,pq B~_P,rs =
        sum_QQ' B_Q,pq W_QQ' B_Q',rs is eq 73's W in chemist order, bare term included

    Notes
    -----
    One Cholesky, as static_second_order_kernel_df; a W_QQ' that is not positive
    definite (an unstable reference) raises ValueError as there.
    """
    eps_hf = np.asarray(eps_hf, float)
    B = np.asarray(B)
    n = eps_hf.size
    if B.ndim != 3 or B.shape[1:] != (n, n):
        raise ValueError(f'B of shape {B.shape}; expected (naux, {n}, {n}), index '
                         'order (Q, p, q)')
    W_aux = static_screened_coulomb_aux(eps_hf, B, nocc)
    try:
        L = np.linalg.cholesky(0.5 * (W_aux + W_aux.T))
    except np.linalg.LinAlgError:
        raise ValueError('W_aux is not positive definite: the static RPA screening of '
                         'an unstable reference.')
    naux = B.shape[0]
    # B~_P,pq = sum_Q L_QP B_Q,pq
    Bt = (L.T @ B.reshape(naux, n * n)).reshape(naux, n, n)
    return L, Bt


def coupling_pairs(kernel, channel):
    """The coupling pairs of a kernel in one spin channel.

    Parameters
    ----------
    kernel : {'eq71', 'printed'}
    channel : {'singlet', 'triplet'}

    Returns
    -------
    list of (x, y)
        x, y : ndarray, shape (4,), coefficients over VERTICES = (I, J, K, L):
        X = sum_A x_A A, Y = sum_B y_B B, the copies V± = (X ± Y)/√2, the kernel
        X.Y + Y.X. Empty when the kernel has no term in the channel ('eq71' in the
        triplet).
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel={kernel!r}; expected one of {KERNELS}")
    if channel not in _CHANNELS:
        raise ValueError(f"channel={channel!r}; expected one of {_CHANNELS}")
    if channel == 'triplet' and kernel in _TRIPLET_FREE:
        return []
    x, y = _PAIRS[kernel]
    return [(x.copy(), y.copy())]


def _check_names(kernel, sets, singles, screening):
    if kernel not in KERNELS:
        raise ValueError(f"kernel={kernel!r}; expected one of {KERNELS}")
    if tuple(sets) != SETS:
        raise ValueError(f"sets={tuple(sets)!r}; expected {SETS} (eq 61's set joins at "
                         'milestone 3)')
    if singles not in SINGLES:
        raise ValueError(f"singles={singles!r}; expected one of {SINGLES}")
    if screening not in ('tda', 'rpa'):
        raise ValueError(f"screening={screening!r}; expected 'tda' or 'rpa'")
