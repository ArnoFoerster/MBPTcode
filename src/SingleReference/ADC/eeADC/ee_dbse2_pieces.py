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


def _vertex_images(Bt, no, y):
    """The images t = I y and u = K y of a spatial singles block on the ordered doubles.

    Parameters
    ----------
    Bt : ndarray, shape (naux, norb, norb), index order (P, p, q), the dressed factors
    no : int
    y : ndarray, shape (no, nv), index order (i, a)

    Returns
    -------
    t : ndarray, shape (no, no, nv, nv), index order (k, l, c, d),
        t_klcd = sum_i w(ik|ld) y_ic, vertex I; vertex J's image is t with the swap
        (k, c) <-> (l, d)
    u : ndarray, shape (no, no, nv, nv), index order (k, l, c, d),
        u_klcd = sum_a w(ac|ld) y_ka, vertex K; vertex L's image is u with the swap
        (k, c) <-> (l, d)
    """
    naux = Bt.shape[0]
    nv = y.shape[1]
    Boo, Bov, Bvv = Bt[:, :no, :no], Bt[:, :no, no:], Bt[:, no:, no:]
    # ZI_P,kc = sum_i B~_P,ik y_ic ;  ZK_P,kc = sum_a y_ka B~_P,ac
    ZI = np.matmul(Boo.transpose(0, 2, 1), y)
    ZK = np.matmul(y, Bvv)
    Bld = Bov.reshape(naux, no * nv)
    out = []
    for Z in (ZI, ZK):
        # x_kc,ld = sum_P Z_P,kc B~_P,ld, then (k, c, l, d) -> (k, l, c, d)
        T = (Z.reshape(naux, no * nv).T @ Bld).reshape(no, nv, no, nv)
        out.append(T.transpose(0, 2, 1, 3))
    return out[0], out[1]


def _vertex_images_t(Bt, no, Y):
    """The transposes I^T Y and K^T Y of one doubles block onto a singles block.

    Parameters
    ----------
    Bt : ndarray, shape (naux, norb, norb), index order (P, p, q), the dressed factors
    no : int
    Y : ndarray, shape (no, no, nv, nv), index order (k, l, c, d); for J^T and L^T pass
        Y with (k, c) <-> (l, d), Y.transpose(1, 0, 3, 2)

    Returns
    -------
    wI : ndarray, shape (no, nv), index order (i, a),
        (I^T Y)_ia = sum_kld w(ik|ld) Y_klad = sum_Pk B~_P,ik G_P,ka
    wK : ndarray, shape (no, nv), index order (i, a),
        (K^T Y)_ia = sum_lcd w(ac|ld) Y_ilcd = sum_Pc B~_P,ac G_P,ic
    with G_P,kc = sum_ld B~_P,ld Y_klcd.
    """
    naux = Bt.shape[0]
    nv = Y.shape[2]
    Boo, Bov, Bvv = Bt[:, :no, :no], Bt[:, :no, no:], Bt[:, no:, no:]
    # G_P,kc = sum_ld B~_P,ld Y_klcd : (P, ld) @ (ld, kc)
    Yx = Y.transpose(1, 3, 0, 2).reshape(no * nv, no * nv)
    G = (Bov.reshape(naux, no * nv) @ Yx).reshape(naux, no, nv)
    # wI_ia = sum_Pk B~_P,ik G_P,ka ;  wK_ia = sum_Pc B~_P,ac G_P,ic
    wI = np.einsum('Pik,Pka->ia', Boo, G, optimize=True)
    wK = np.einsum('Pac,Pic->ia', Bvv, G, optimize=True)
    return wI, wK


def _singles_block(eps_gw, B, Bt, nocc, singles):
    """M of eq 62 at the GW energies, with eq 59's static W for singles='eq62+eq59'.

    Returns SB, blocks (no, nv, no, nv), index order (i, a, j, b): the CIS block of the
    gf2 builder with the GW energies on its diagonal, A_ia,jb = (ε_a - ε_i) δ_ij δ_ab +
    (ia|jb) - (ij|ab) on 'aaaa' and 'bbbb', (ia|jb) on 'aabb' and 'bbaa'; for
    'eq62+eq59' minus Wc_ia,jb = sum_P B~_P,ij B~_P,ab - sum_P B_P,ij B_P,ab on the
    same-spin blocks, the screened exchange of eq 59 in place of the bare one.
    """
    no = nocc
    nv = eps_gw.size - no
    # A^GW of eq 62: the gf2 builder's first-order singles block at the GW energies
    M = ee_r_sigma_df.build_operator(eps_gw, B, no, level='gf2', pieces=True)[3]['M']
    if singles == 'eq62+eq59':
        naux = B.shape[0]
        # Wc_ij,ab = sum_P B~_P,ij B~_P,ab - sum_P B_P,ij B_P,ab
        Wc = (Bt[:, :no, :no].reshape(naux, -1).T @ Bt[:, no:, no:].reshape(naux, -1)
              - B[:, :no, :no].reshape(naux, -1).T @ B[:, no:, no:].reshape(naux, -1))
        # (i, j, a, b) -> (i, a, j, b)
        Wc = Wc.reshape(no, no, nv, nv).transpose(0, 2, 1, 3)
        M = M - SB({'aaaa': Wc, 'bbbb': Wc})
    return M


def _combine(c, t1, u1, t2, u2):
    """sum over the vertices of c_A times A's image on one doubles block: c_I t(s1) +
    c_K u(s1) + c_J t(s2)^T + c_L u(s2)^T, ^T the swap (k, c) <-> (l, d)."""
    out = None
    for coef, arr in ((c[0], t1), (c[2], u1), (c[1], t2.transpose(1, 0, 3, 2)),
                      (c[3], u2.transpose(1, 0, 3, 2))):
        if coef == 0.0:
            continue
        out = coef * arr if out is None else out + coef * arr
    return out


def build_pieces_dbse2(eps_gw, eps_hf, B, nocc, kernel='eq71', sets=('eq71',),
                       screening='rpa', singles='eq62'):
    """The pieces of ee_fold.folded_operator for eq 71's doubles on the signed fold.

    Parameters
    ----------
    eps_gw : ndarray, shape (norb,), the GW quasiparticle energies ε^GW_p, Hartree,
        occupied first: the singles diagonal of eq 62 and the doubles diagonal D
    eps_hf : ndarray, shape (norb,), HF orbital energies, Hartree, the reference the W
        lines are screened at
    B : ndarray, shape (naux, norb, norb), index order (Q, p, q), DF factors with
        (pq|rs) = sum_Q B_Q,pq B_Q,rs
    nocc : int, doubly occupied orbitals
    kernel : {'eq71', 'printed'}, the module docstring's table
    sets : tuple of str, the doubles sets; ('eq71',) alone until eq 61's set joins
    screening : {'tda', 'rpa'}, the screening of eq 61's modes, read when that set joins
    singles : {'eq62', 'eq62+eq59'}, the singles block M (_singles_block)

    Returns
    -------
    dict
        'M' SB, blocks (no, nv, no, nv), index order (i, a, j, b); 'V' SB singles
        {'aa', 'bb'} of (no, nv), index order (i, a) -> SB doubles on the eight keys
        '+aaaa', '-aaaa', '+bbbb', '-bbbb', '+abab', '-abab', '+baba', '-baba', each
        (no, no, nv, nv), index order (k, l, c, d), the copy's sign then the spin string
        of (k, l, c, d); 'Vt' the η-transpose, w_ia = sum_K V+_K,ia Z+_K - sum_K
        V-_K,ia Z-_K; 'D' ndarray, shape (no, no, nv, nv), index order (k, l, c, d),
        D_klcd = ε_c + ε_d - ε_k - ε_l at the GW energies, shared by every block;
        'dnorm2' SB doubles -> |Z+|² - |Z-|²; 'level' 'dbse2'; 'no', 'nv', 'be';
        'dense_limit' 40; 'Bt' the dressed factors and 'pairs' the singlet list of
        coupling_pairs, for inspection.

    Notes
    -----
    With the pair (x, y) over (I, J, K, L), X = sum_A x_A A and Y = sum_B y_B B, the
    copies are V± = (X ± Y)/√2; on the block whose (k, c) pair carries spin s1 and
    whose (l, d) pair carries s2, X y1 = x_I t(y^s1) + x_K u(y^s1) + x_J t(y^s2)^T +
    x_L u(y^s2)^T with (t, u) of _vertex_images and ^T the swap (k, c) <-> (l, d).
    Vt(Z) = V+^T Z+ - V-^T Z- = X^T Zx + Y^T Zy with Zx = (Z+ - Z-)/√2 and
    Zy = (Z+ + Z-)/√2: I^T and K^T read the blocks whose (k, c) carries the output
    spin, J^T and L^T those whose (l, d) does, each through _vertex_images_t. The
    doubles exist inside one call only.
    """
    _check_names(kernel, sets, singles, screening)
    eps_gw = np.asarray(eps_gw, float)
    eps_hf = np.asarray(eps_hf, float)
    no = nocc
    nv = eps_gw.size - no
    if eps_gw[:no].max() >= eps_gw[no:].min():
        raise ValueError('eps_gw: the GW gap is closed (occupied max '
                         f'{eps_gw[:no].max():.4f} Ha, virtual min '
                         f'{eps_gw[no:].min():.4f} Ha); energies in Hartree, occupied '
                         'first')
    pairs = coupling_pairs(kernel, 'singlet')
    if len(pairs) != 1:
        raise NotImplementedError('one coupling pair per set at milestone 1')
    x, y = pairs[0]
    _, Bt = dressed_factors(eps_hf, B, no)
    M = _singles_block(eps_gw, B, Bt, no, singles)
    eo, ev = eps_gw[:no], eps_gw[no:]
    # D_klcd = ε_c + ε_d - ε_k - ε_l
    D = (ev[None, None, :, None] + ev[None, None, None, :]
         - eo[:, None, None, None] - eo[None, :, None, None])
    blocks = (('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b'))      # (s1, s2)

    def V(y1):
        if y1.parity is not None:
            raise NotImplementedError('V takes y1 without a flip parity')
        img = {s: _vertex_images(Bt, no, y1.get(s + s)) for s in 'ab'}
        out = {}
        for s1, s2 in blocks:
            t1, u1 = img[s1]
            t2, u2 = img[s2]
            Xb = _combine(x, t1, u1, t2, u2)
            Yb = _combine(y, t1, u1, t2, u2)
            key = s1 + s2 + s1 + s2
            # V± = (X ± Y)/√2 on this block
            out['+' + key] = (Xb + Yb) / _SQRT2
            out['-' + key] = (Xb - Yb) / _SQRT2
        return SB(out)

    def Vt(Z):
        def combined(keys, sign):
            # (Z+ + sign Z-)/√2 summed over the given spin blocks
            return sum(Z.get('+' + k) + sign * Z.get('-' + k) for k in keys) / _SQRT2

        w = {}
        for a in 'ab':
            # the blocks whose (k, c) pair, and whose (l, d) pair, carries spin a
            kc = [s1 + s2 + s1 + s2 for s1, s2 in blocks if s1 == a]
            ld = [s1 + s2 + s1 + s2 for s1, s2 in blocks if s2 == a]
            wa = np.zeros((no, nv))
            # Zx = (Z+ - Z-)/√2 meets X^T, Zy = (Z+ + Z-)/√2 meets Y^T
            for c, sign in ((x, -1.0), (y, 1.0)):
                if c[0] != 0.0 or c[2] != 0.0:
                    # I^T, K^T read the blocks whose (k, c) pair carries spin a
                    wI, wK = _vertex_images_t(Bt, no, combined(kc, sign))
                    wa = wa + c[0] * wI + c[2] * wK
                if c[1] != 0.0 or c[3] != 0.0:
                    # J^T, L^T read those whose (l, d) pair does, with (k, c) <-> (l, d)
                    Zs = combined(ld, sign)
                    wI, wK = _vertex_images_t(Bt, no, Zs.transpose(1, 0, 3, 2))
                    wa = wa + c[1] * wI + c[3] * wK
            w[a + a] = wa
        return SB(w)

    def dnorm2(Z):
        # |Z+|² - |Z-|²
        return float(sum((1.0 if k[0] == '+' else -1.0) * np.vdot(v, v)
                         for k, v in Z.items()))

    return {'M': M, 'V': V, 'Vt': Vt, 'D': D, 'dnorm2': dnorm2, 'level': 'dbse2',
            'no': no, 'nv': nv, 'be': _eq.SPIN_BLOCKED, 'dense_limit': _DENSE_LIMIT,
            'Bt': Bt, 'pairs': pairs}
