"""The dBSE2 doubles as a tiled RI operator (level 'dbse2'): A_eff(ω) of one spin
channel on flat singles vectors, in RIFoldADC2's interface, from the DF factors at the
GW energies, on the host or a GPU; the doubles sets selected by `sets` (SETS, and
('eq66',) for level gw), each in two forms (FORMS).

Eq 71's set. The kernel is ee_dbse2_pieces' (its module docstring: the vertices I, J,
K, L on the ordered doubles K = (k, l, c, d), D_K = ε_c + ε_d - ε_k - ε_l, the pair
(x, y) of coupling_pairs with X = Σ_A x_A A, Y = Σ_B y_B B, Θ = (X.Y + Y.X)/2, the
copies V± = (X ± Y)/2 on the metric η = diag(+1, -1)). Here the four spin blocks of the
pieces
collapse onto the spin-adapted channel vector u, y^aa = u/√2, y^bb = ±u/√2 (singlet +,
triplet -): with X̂, Ŷ the vertex sums of the images of u and X̃, Ỹ the same with the J
and L coefficients negated,

    Θ(ω) u = c_hat sym[X̂ᵀ w Ŷ] u + c_tilde sym[X̃ᵀ w Ỹ] u,   w_K = 1 / (ω - D_K),
    sym[Aᵀ w B] u = (Aᵀ (w B u) + Bᵀ (w A u)) / 2,

singlet c_hat = 2, c_tilde = 0; triplet c_hat = c_tilde = 1; in one line, Θ = Σ_AB m_AB
A.B with m = c_hat (x yᵀ + y xᵀ)/2 + c_tilde (x̃ ỹᵀ + ỹ x̃ᵀ)/2 and A.B the product
Σ_K A_ia,K B_jb,K w_K. The signed norm |Z+|² - |Z-|² is Σ_K Zx_K Zy_K with Zx = w X̂ u,
Zy = w Ŷ u, the same factors; the ω-derivative replaces w by -w². Eq 71's triplet gives
X̂.Ŷ + X̃.Ỹ = 0; the printed kernel's triplet keeps its loop terms.

Two factor sets: the bare factors B at the GW energies build the singles block M, eq 62
(the inner RIFoldADC2 at level 'adc1', CIS on those factors), with eq 59's w(ij|ab) in
place of (ij|ab) for singles='eq62+eq59'; the dressed factors B~ (the W lines of eq 73,
ee_dbse2_pieces.dressed_factors) feed the doubles through the factors of one channel
vector, ZI_P,kc = Σ_i B~_P,ik u_ic and ZK_P,kc = Σ_a u_ka B~_P,ac.

form='exact': the operator's own tiles over the occupied index (sized by _tile_bytes),
RIFoldADC2's auxiliary chunks and back-transformation halves on the dressed factors.
Per tile the images I = H, J = Hᵀ of ZI and K = H, L = Hᵀ of ZK are _vertex_images'
four GEMMs (RIFoldADC2._pair's shape), the swap (k, c) <-> (l, d) being the pair's
transpose. Every
back-transformation is row-local to the tile: with v^sw the swap of a doubles vector v
and D swap-symmetric,
Aᵀ v = (A^sw)ᵀ v^sw, so Jᵀ R_J = Iᵀ R_J^sw and Lᵀ R_L = Kᵀ R_L^sw, where the swapped
R is the same combination of the four images with the coefficients of I and J (K and L)
exchanged. Θ u = Iᵀ R_I + Kᵀ R_K then closes through B~_oo (_back_oo) and B~_vv
(_back_vv) on the tile's rows, as RIFoldADC2's doubles do; the divide is exact.

form='laplace', the default where it runs (spec §2.6; form=None takes it on one rank
with every B_vv whole and 'exact' with a warning otherwise): 1/(D_K - ω) =
Σ_t w_t exp(-τ_t (D_K - ω)) on
[min D - ω, max D - ω], the minimax grid of grids.minimax_time_grid sized by
minimax_points_for_accuracy at kernel_laplace's relative target, 1/(D_K - ω)² with
the weights w_t τ_t. The exponential splits into one-index weights, and every product
A.B closes through an n_aux² intermediate at each τ, Π_QP = Σ_ld B~_Q,ld b_l a_d F_P,ld
with F one of B~, ZI, ZK (_laplace_sums' Notes), so no doubles-sized array exists.
A node at or above min D, or one whose range no tabulated grid reaches at the target,
goes through the exact tile (laplace_fallbacks counts them), kernel_laplace's rule. The
form runs on one rank with the B_vv block whole, the dressed one for eq 71's set and
the bare one for a mode set.

Eq 61's set (k, c, m), the mode set (ee_dbse2_pieces' module docstring): the modes Ω_m,
ρ_P,m of gw_modes_df at the HF energies, W^m_pq = √2 Σ_P B_P,pq ρ_P,m through the bare
factors, D_kcm = Ω_m + ε_c - ε_k at the bare factors' energies, and eq 64's couplings
J_ia,kcm = -δ_ac W^m_ik, K_ia,kcm = +δ_ik W^m_ac. Neither carries exchange, so the
channel vector u sees Θ^61 u = Θ^61_aa,aa u in both channels. On one occupied tile the
images are two GEMMs of the bare vertex factors against the modes, Y^J = -√2 ZIᵀ ρ and
Y^K = +√2 ZKᵀ ρ, each (kt nv, nm), and the signed cross Θ^61 = J.K + K.J (the copies
V± = (J ± K)/√2 on η) closes as

    Θ^61 u = Jᵀ R_J + Kᵀ R_K,   R_J = w Y^K,   R_K = w Y^J,
    |Z+|² - |Z-|² = 2 Σ w² Y^J Y^K,

through the bare B_oo and B_vv: with Γ = R ρᵀ, Jᵀ R_J = -√2 _back_oo(Γ_J) and Kᵀ R_K =
-√2 _back_vv(Γ_K) (the +√2 of Kᵀ against _back_vv's minus), the inner operator's
back-transformation on this operator's tiles (_mode_tile, _mode_overlap). With
form='laplace' the mode set runs _mode_laplace instead: per τ one Π^τ_PQ = Σ_m exp(-τ
(Ω_m - Ω_min)) ρ_P,m ρ_Q,m (n_aux² n_m), the screened interaction at imaginary time,
then Jᵀ and Kᵀ close H = Π^τ Z through the bare B_oo and B_vv with the one-index
weights of eq 71's form (its Notes), on a grid over the mode set's own [min D - ω,
max D - ω]; a node at or above that minimum, or without a grid, goes through the mode
tile (laplace_fallbacks counts it). The level's d_min is
the lower of the sets' minima, 2 (min ε_v - max ε_o) for eq 71's and Ω_min + min ε_v -
max ε_o for the mode set; the tile budget sums _tile_bytes and _mode_bytes.

sets=('eq66',) is level 'gw' (eq 66, ee_gw_pieces' module docstring) through the same
mode tile with the square pairing: at the HF energies throughout (from_mo(eps_hf,
eps_hf, ...)), the singles block A^HF + Ā^GW with eq 70's Ā applied as u (P + Pᵀ) -
(Q + Qᵀ) u from static_term_pq's two blocks, and the doubles Θ^66 = (J + K).(J + K),
R_J = R_K = w (Y^J + Y^K), |Z|² = Σ w² (Y^J + Y^K)², the one copy on a positive metric;
no dressed factors, kernel unread, singles 'eq62' only; its Laplace form takes H =
Π^τ (ZK - ZI) on both closures (_mode_laplace's Notes).

The same cores run on numpy and on cupy (_tile_core, _mode_tile, _laplace_sums and
_mode_laplace take the array module).

References
----------
E. Monino and P.-F. Loos, J. Chem. Phys. 159, 034105 (2023), eqs 59, 61, 62, 64, 71 to
74.
"""
import warnings

import numpy as np

from src.Base.constants import EE_RI_TILE_GB
from src.Base.utils.grids import minimax_time_grid
from src.Base.utils.mpi_grid import allgather_ranges, ordered_chain_sum, ordered_sum
from src.Base.utils.time_frequency import minimax_points_for_accuracy
from src.SingleReference.ADC.eeADC import ee_dbse2_pieces as _pieces
from src.SingleReference.ADC.eeADC.ee_gw_pieces import _LAPLACE_TARGET as LAPLACE_TARGET
from src.SingleReference.ADC.eeADC.ee_gw_pieces import gw_modes_df, static_term_pq
from src.SingleReference.ADC.eeADC.ee_ri_fold import DEVICES, RIFactors, RIFoldADC2
from src.SingleReference.LinearResponse.linear_response import device_memory_check

try:
    import cupy as cp
except ImportError:
    cp = None

FORMS = ('exact', 'laplace')
# the doubles sets, each tuple sorted: eq 71's, eq 61's, the level dbse2, level gw
SETS = (('eq71',), ('eq61',), ('eq61', 'eq71'), ('eq66',))
# the J and L coefficients change sign in the triplet's second pair (X̃, Ỹ)
_FLIP = np.array([1.0, -1.0, 1.0, -1.0])
# the swap (k, c) <-> (l, d) exchanges I with J and K with L
_SWAP = (1, 0, 3, 2)


def _channel_factors(spin):
    """(c_hat, c_tilde) of the module docstring."""
    return (2.0, 0.0) if spin == 'singlet' else (1.0, 1.0)


def _kernel_matrix(pair, spin):
    """m (4, 4) over (I, J, K, L) with Θ = Σ_AB m_AB A.B: c_hat (x yᵀ + y xᵀ)/2 plus
    c_tilde (x̃ ỹᵀ + ỹ x̃ᵀ)/2."""
    x, y = pair
    ch, ct = _channel_factors(spin)
    m = ch * 0.5 * (np.outer(x, y) + np.outer(y, x))
    if ct:
        xt, yt = x * _FLIP, y * _FLIP
        m = m + ct * 0.5 * (np.outer(xt, yt) + np.outer(yt, xt))
    return m


def _denominator(xp, eo, ev, k0, k1):
    """D_kcld = ε_c + ε_d - ε_k - ε_l for the tile's rows k0..k1-1, (kt, nv, no, nv)."""
    return (ev[None, :, None, None] + ev[None, None, None, :]
            - eo[k0:k1, None, None, None] - eo[None, None, :, None])


def _tile_bytes(kt, no, nv, naux, p=1):
    """Bytes one occupied tile of kt rows holds at its peak: 18 tile-sized arrays
    (kt, nv, no, nv) (the two-vector overlap of _tile_overlap in the triplet: eight
    images, four combinations, the denominator, its reciprocal and the products) and
    the Gam pair (p, kt nv, naux) of _tile_core at p nodes."""
    return 8 * kt * nv * no * nv * 18 + 2 * 8 * p * kt * nv * naux


def _laplace_bytes(no, nv, naux):
    """Bytes one Laplace node of _laplace_sums holds at its peak beside the residents:
    six (naux, naux) arrays (Π^ZI, Π^ZK, Π^BB and the three temporaries of their
    weighted sum) and seven factor-sized (naux, no nv) arrays (Bw, H, part and the
    temporaries of the closures); traced on water at 7.1 naux² + 0 to 3 L2 for the
    printed singlet, the largest case."""
    return 6 * 8 * naux ** 2 + 7 * 8 * naux * no * nv


def _mode_laplace_bytes(no, nv, naux, nm):
    """Bytes one node of _mode_laplace holds at its peak beside the residents: Π^τ
    (naux, naux), the weighted modes (naux, nm) while Π^τ forms, and six
    factor-sized (naux, no nv) arrays (H, Ha, the closures' temporaries and the
    products' buffers; the previous node's arrays are freed before the next forms).
    Check 9 of tests/test_ee_dbse2_ri_fold.py traces the cross and the square."""
    return 8 * (naux ** 2 + naux * nm + 6 * naux * no * nv)


# ---- the exact tile -----------------------------------------------------------


def _vertex_images(xp, L2, no, nv, k0, k1, ZI, ZK):
    """The images I, J, K, L of one channel vector on one tile.

    Parameters
    ----------
    xp : the array module, numpy or cupy
    L2 : ndarray, shape (naux, no nv), index order (P, (l, d)), the dressed B~_P,ld
    no, nv : int
    k0, k1 : int, the tile's occupied rows
    ZI, ZK : ndarray, shape (naux, no nv), index order (P, (k, c)), the vertex factors

    Returns
    -------
    tuple of four ndarray, shape (kt, nv, no, nv), index order (k, c, l, d):
        I_kc,ld = Σ_P ZI_P,kc B~_P,ld, J_kc,ld = Σ_P B~_P,kc ZI_P,ld (the swap), and
        K, L the same of ZK
    """
    kt, rows = k1 - k0, slice(k0 * nv, k1 * nv)
    shape = (kt, nv, no, nv)
    return ((ZI[:, rows].T @ L2).reshape(shape), (L2[:, rows].T @ ZI).reshape(shape),
            (ZK[:, rows].T @ L2).reshape(shape), (L2[:, rows].T @ ZK).reshape(shape))


def _combine(verts, c):
    """Σ_B c_B B over the four images, None when every c_B is 0."""
    out = None
    for coef, arr in zip(c, verts):
        if coef == 0.0:
            continue
        out = coef * arr if out is None else out + coef * arr
    return out


def _row_coefficients(pair, spin):
    """(b_I, b_K), each shape (4,): R_I = w Σ_B b_I,B B and R_K = w Σ_B b_K,B B over the
    images (I, J, K, L), so that Θ u = Iᵀ R_I + Kᵀ R_K on every tile.

    Notes
    -----
    From Θ u = Σ_A Aᵀ R_A with R_A = w Σ_B m_AB B u and Jᵀ R_J = Iᵀ R_J^sw,
    Lᵀ R_L = Kᵀ R_L^sw: the swap of a combination of the images is the combination
    with I <-> J and K <-> L exchanged (_SWAP).
    """
    m = _kernel_matrix(pair, spin)
    b = np.zeros((2, 4))
    for row, a in enumerate((0, 2)):
        asw = _SWAP[a]
        for v in range(4):
            # Aᵀ Σ_B m_AB B: image v with m_Av
            b[row, v] += m[a, v]
            # (A^sw)ᵀ Σ_B m_A^sw,B B^sw: image v lands on image _SWAP[v]
            b[row, _SWAP[v]] += m[asw, v]
    return b[0], b[1]


def _tile_core(xp, L2, eo, ev, k0, k1, ZI, ZK, pair, spin, nodes, derivative):
    """One occupied tile of the folded doubles of one channel vector at p frequencies.

    Parameters
    ----------
    xp, L2, k0, k1, ZI, ZK : as _vertex_images
    eo, ev : ndarray, shape (no,), (nv,), the GW energies on xp
    pair : (x, y), each shape (4,), coefficients over (I, J, K, L)
    spin : {'singlet', 'triplet'}
    nodes : ndarray, shape (p,), the frequencies ω_q, Hartree
    derivative : ndarray of bool, shape (p,), the ω-derivative at node q

    Returns
    -------
    gam_I, gam_K : ndarray, shape (p, kt nv, naux), index order (q, (k, c), P),
        Gam_kc,P = Σ_ld R_kc,ld B~_P,ld for R = R_I and R = R_K: the Iᵀ half closes
        through B~_oo on the tile's rows (_back_oo), the Kᵀ half through B~_vv
        (_back_vv, with its minus sign)
    nrm : ndarray, shape (p,), host, the signed norm Σ_K Zx_K Zy_K at each node

    Notes
    -----
    w = 1/(ω_q - D), or -1/(ω_q - D)² where derivative[q]; R_I and R_K are
    _row_coefficients' combinations of the images times w. The norm uses the plain w:
    Σ w² X̂ Ŷ c_hat + Σ w² X̃ Ỹ c_tilde.
    """
    no, nv = int(eo.size), int(ev.size)
    kt = k1 - k0
    naux = L2.shape[0]
    verts = _vertex_images(xp, L2, no, nv, k0, k1, ZI, ZK)
    x, y = pair
    ch, ct = _channel_factors(spin)
    Xh, Yh = _combine(verts, x), _combine(verts, y)
    Xt, Yt = (None, None) if ct == 0.0 else (_combine(verts, x * _FLIP),
                                             _combine(verts, y * _FLIP))
    bI, bK = _row_coefficients(pair, spin)
    RI, RK = _combine(verts, bI), _combine(verts, bK)
    del verts
    D = _denominator(xp, eo, ev, k0, k1)
    p = len(nodes)
    gam_I, gam_K = xp.empty((p, kt * nv, naux)), xp.empty((p, kt * nv, naux))
    nrm = np.zeros(p)
    for q, omega in enumerate(nodes):
        w = 1.0 / (float(omega) - D)
        # Σ_K Zx_K Zy_K = Σ w² X̂ Ŷ c_hat + Σ w² X̃ Ỹ c_tilde
        s = ch * float(xp.sum(w * w * Xh * Yh))
        if ct:
            s += ct * float(xp.sum(w * w * Xt * Yt))
        nrm[q] = s
        if derivative[q]:
            # d/dω 1/(ω - D) = -1/(ω - D)²
            w = -w * w
        # Gam_kc,P = Σ_ld (w R)_kc,ld B~_P,ld; R None where the kernel has no term
        # (eq 71's triplet)
        for gam, R in ((gam_I, RI), (gam_K, RK)):
            gam[q] = 0.0 if R is None else (w * R).reshape(kt * nv, no * nv) @ L2.T
    return gam_I, gam_K, nrm


def _tile_overlap(xp, L2, eo, ev, k0, k1, Zr, wr, Zs, ws, pair, spin):
    """Σ_K Y_K(u_r, ω_r)ᵀ η Y_K(u_s, ω_s) on one tile, the signed overlap of two doubles
    images: with Zx = w X̂ u, Zy = w Ŷ u, (Zx_r . Zy_s + Zy_r . Zx_s) / 2 per pair with
    the channel factors. Zr, Zs are the (ZI, ZK) factor pairs of the two vectors."""
    no, nv = int(eo.size), int(ev.size)
    x, y = pair
    ch, ct = _channel_factors(spin)
    vr = _vertex_images(xp, L2, no, nv, k0, k1, Zr[0], Zr[1])
    vs = vr if Zs is Zr else _vertex_images(xp, L2, no, nv, k0, k1, Zs[0], Zs[1])
    D = _denominator(xp, eo, ev, k0, k1)
    w2 = 1.0 / ((wr - D) * (ws - D))
    v = 0.0
    for c, cx, cy in ((ch, x, y), (ct, x * _FLIP, y * _FLIP)):
        if c == 0.0:
            continue
        Xr, Yr = _combine(vr, cx), _combine(vr, cy)
        Xs, Ys = (Xr, Yr) if vs is vr else (_combine(vs, cx), _combine(vs, cy))
        v += 0.5 * c * float(xp.sum(w2 * (Xr * Ys + Yr * Xs)))
    return v


# ---- the mode tile ------------------------------------------------------------


def _mode_bytes(kt, nv, nm, naux, p=1):
    """Bytes one occupied tile of kt rows of the mode set holds at its peak: nine
    arrays (kt nv, nm), the peak of _mode_overlap on two vectors (their four images,
    the denominator, the weight product and three products; one _mode_tile at a node
    holds six), one row's (1, nv, nm) broadcast temporary of the denominator (the
    traced peaks of check 9 of tests/test_ee_dbse2_ri_fold.py) and the Gam pair
    (p, kt nv, naux) of _mode_tile at p nodes."""
    return 8 * nv * nm * (9 * kt + 1) + 2 * 8 * p * kt * nv * naux


def _mode_denominator(xp, om, eo, ev, k0, k1):
    """D_kcm = Ω_m + ε_c - ε_k for the tile's rows k0..k1-1, (kt nv, nm)."""
    D = om[None, None, :] + ev[None, :, None] - eo[k0:k1, None, None]
    return D.reshape((k1 - k0) * int(ev.size), int(om.size))


def _mode_images(xp, rho, k0, k1, ZI, ZK, nv):
    """The images Y^J = J u and Y^K = K u of one channel vector on one tile of the mode
    set.

    Parameters
    ----------
    xp : the array module, numpy or cupy
    rho : ndarray, shape (naux, nm), index order (P, m), the modes ρ_P,m
    k0, k1 : int, the tile's occupied rows
    ZI, ZK : ndarray, shape (naux, no nv), index order (P, (k, c)), the bare vertex
        factors ZI_P,kc = Σ_i B_P,ik u_ic and ZK_P,kc = Σ_a u_ka B_P,ac
    nv : int

    Returns
    -------
    YJ, YK : ndarray, shape (kt nv, nm), index order ((k, c), m),
        Y^J_kcm = -√2 Σ_P ZI_P,kc ρ_P,m = -Σ_i W^m_ik u_ic and
        Y^K_kcm = +√2 Σ_P ZK_P,kc ρ_P,m = +Σ_a W^m_ac u_ka
    """
    rows = slice(k0 * nv, k1 * nv)
    return -np.sqrt(2.0) * (ZI[:, rows].T @ rho), np.sqrt(2.0) * (ZK[:, rows].T @ rho)


def _mode_tile(xp, rho, om, eo, ev, k0, k1, ZI, ZK, nodes, derivative, square):
    """One occupied tile of the folded mode doubles of one channel vector at p
    frequencies.

    Parameters
    ----------
    xp, rho, k0, k1, ZI, ZK : as _mode_images
    om : ndarray, shape (nm,), Ω_m on xp
    eo, ev : ndarray, shape (no,), (nv,), the bare factors' energies on xp
    nodes : ndarray, shape (p,), the frequencies ω_q, Hartree
    derivative : ndarray of bool, shape (p,), the ω-derivative at node q
    square : bool, False the signed cross J.K + K.J, True the square (J + K).(J + K)

    Returns
    -------
    gam_J, gam_K : ndarray, shape (p, kt nv, naux), index order (q, (k, c), P),
        Γ_kc,P = Σ_m R_kcm ρ_P,m for R = R_J and R = R_K: the Jᵀ half closes through
        B_oo on the tile's rows (-√2 _back_oo), the Kᵀ half through B_vv (-√2 _back_vv)
    nrm : ndarray, shape (p,), host, the signed norm at each node

    Notes
    -----
    w = 1/(ω_q - D), or -1/(ω_q - D)² where derivative[q]. The cross: R_J = w Y^K,
    R_K = w Y^J, nrm = 2 Σ w² Y^J Y^K; the square: R_J = R_K = w (Y^J + Y^K),
    nrm = Σ w² (Y^J + Y^K)². The norm uses the plain w.
    """
    nv = int(ev.size)
    kt = k1 - k0
    naux = rho.shape[0]
    YJ, YK = _mode_images(xp, rho, k0, k1, ZI, ZK, nv)
    if square:
        # one copy Y^J + Y^K on both halves
        YJ = YJ + YK
        YK = YJ
    D = _mode_denominator(xp, om, eo, ev, k0, k1)
    p = len(nodes)
    gam_J, gam_K = xp.empty((p, kt * nv, naux)), xp.empty((p, kt * nv, naux))
    nrm = np.zeros(p)
    for q, omega in enumerate(nodes):
        w = 1.0 / (float(omega) - D)
        # 2 Σ w² Y^J Y^K, or Σ w² (Y^J + Y^K)² on the square
        nrm[q] = (1.0 if square else 2.0) * float(xp.sum(w * w * YJ * YK))
        if derivative[q]:
            # d/dω 1/(ω - D) = -1/(ω - D)²
            w = -w * w
        # Γ_kc,P = Σ_m (w R)_kcm ρ_P,m with R_J = Y^K and R_K = Y^J, into the slab
        xp.matmul(w * YK, rho.T, out=gam_J[q])
        xp.matmul(w * YJ, rho.T, out=gam_K[q])
    return gam_J, gam_K, nrm


def _mode_overlap(xp, rho, om, eo, ev, k0, k1, Zr, wr, Zs, ws, square):
    """Σ_K Y_K(u_r, ω_r)ᵀ η Y_K(u_s, ω_s) on one tile of the mode set, the signed
    overlap of two doubles images: Σ w_r w_s (Y^J_r Y^K_s + Y^K_r Y^J_s) for the cross,
    Σ w_r w_s (Y^J_r + Y^K_r)(Y^J_s + Y^K_s) for the square, w = 1/(ω - D). Zr, Zs are
    the bare (ZI, ZK) factor pairs of the two vectors."""
    nv = int(ev.size)
    YJr, YKr = _mode_images(xp, rho, k0, k1, Zr[0], Zr[1], nv)
    YJs, YKs = ((YJr, YKr) if Zs is Zr
                else _mode_images(xp, rho, k0, k1, Zs[0], Zs[1], nv))
    D = _mode_denominator(xp, om, eo, ev, k0, k1)
    w2 = 1.0 / ((wr - D) * (ws - D))
    if square:
        return float(xp.sum(w2 * (YJr + YKr) * (YJs + YKs)))
    return float(xp.sum(w2 * (YJr * YKs + YKr * YJs)))


# ---- the Laplace form ---------------------------------------------------------


def _closure_coefficients(pair, spin):
    """g (8,): the weights of _laplace_sums' eight closures, from m = _kernel_matrix.

    Notes
    -----
    A vertex closes its product through B~_oo (A = I, J) or B~_vv (A = K, L); the
    other vertex sets the intermediate Π and the factor F: a vertex on the other pair
    (I with J or L, K with J or L, and the swaps) gives Π^Z of its own factor Z with
    F = B~, a vertex on the same pair gives Π^BB with F = Z. So, over (I, J, K, L):
    oo through Π^ZI: m_IJ + m_JI; oo through Π^ZK: m_IL + m_JK; oo through Π^BB with
    ZI: m_II + m_JJ; with ZK: m_IK + m_JL; vv through Π^ZI: m_KJ + m_LI; through Π^ZK:
    m_KL + m_LK; through Π^BB with ZI: m_KI + m_LJ; with ZK: m_KK + m_LL.
    """
    m = _kernel_matrix(pair, spin)
    i, j, k, l = range(4)
    return np.array([m[i, j] + m[j, i], m[i, l] + m[j, k], m[i, i] + m[j, j],
                     m[i, k] + m[j, l], m[k, j] + m[l, i], m[k, l] + m[l, k],
                     m[k, i] + m[l, j], m[k, k] + m[l, l]])


def _laplace_sums(xp, L2, Boo_i, Bvv_a, eo, ev, ZI, ZK, g, tau, wt, lo, coefs):
    """Weighted Laplace sums of the kernel's closure vector of one channel vector.

    Parameters
    ----------
    xp : the array module, numpy or cupy
    L2 : ndarray, shape (naux, no nv), index order (P, (l, d)), B~_P,ld
    Boo_i : ndarray, shape (no, naux no), index order (i, (Q, k)), B~_Q,ik
    Bvv_a : ndarray, shape (naux nv, nv), index order ((Q, c), a), B~_Q,ac
    eo, ev : ndarray, shape (no,), (nv,), the GW energies on xp
    ZI, ZK : ndarray, shape (naux, no nv), index order (P, (k, c)), the vertex factors
    g : ndarray, shape (8,), _closure_coefficients
    tau, wt : ndarray, shape (ntau,), the minimax grid, 1/x = Σ_t wt_t exp(-τ_t x)
    lo : float, min D - ω, the shift that keeps every exponent at or below zero
    coefs : list of callables c(τ, w) -> float, the weight of the node τ in each sum
        (w for 1/x, w τ for 1/x², with the sum's sign)

    Returns
    -------
    list of ndarray, shape (no nv,), Σ_t c(τ_t, w_t) v_t for each c in coefs, v_t the
        closure vector at τ_t

    Notes
    -----
    exp(-τ (D_K - ω)) = s b_k b_l a_c a_d with s = exp(-τ lo), a_c = exp(-τ (ε_c -
    min ε_v)), b_k = exp(-τ (max ε_o - ε_k)). Per τ, Bw_Q,ld = B~_Q,ld b_l a_d,
    Π^BB_QP = Σ_ld Bw_Q,ld B~_P,ld, Π^ZI and Π^ZK the same with ZI, ZK as the second
    factor (n_aux² each), then
        H_Q,ka  = Σ_P [(g0 Π^ZI + g1 Π^ZK)_QP B~_P,ka + Π^BB_QP (g2 ZI + g3 ZK)_P,ka],
        oo_ia   = Σ_Qk B~_Q,ik b_k H_Q,ka,
        H'_Q,ic = the same with g4 to g7,
        vv_ia   = Σ_Qc B~_Q,ac a_c H'_Q,ic,
        v_t     = s (a_a oo_ia + b_i vv_ia).
    """
    no, nv, naux = int(eo.size), int(ev.size), L2.shape[0]
    ev_min, eo_max = float(ev.min()), float(eo.max())
    sums = [xp.zeros((no, nv)) for _ in coefs]
    for t, w in zip(tau, wt):
        a = xp.exp(-t * (ev - ev_min))
        b = xp.exp(-t * (eo_max - eo))
        s = float(np.exp(-t * lo))
        # Bw_Q,ld = B~_Q,ld b_l a_d ; Π_QP = Σ_ld Bw_Q,ld F_P,ld
        Bw = L2 * (b[:, None] * a[None, :]).reshape(1, -1)
        Pi_Z = {}
        if g[0] or g[1] or g[4] or g[5]:
            Pi_Z['I'], Pi_Z['K'] = Bw @ ZI.T, Bw @ ZK.T
        Pi_BB = Bw @ L2.T if g[2] or g[3] or g[6] or g[7] else None
        v = None
        for gz_i, gz_k, gb_i, gb_k, close in ((g[0], g[1], g[2], g[3], 'oo'),
                                               (g[4], g[5], g[6], g[7], 'vv')):
            # H_Q,(k a) = Σ_P Π_QP F_P,(k a)
            H = None
            if gz_i or gz_k:
                H = (gz_i * Pi_Z['I'] + gz_k * Pi_Z['K']) @ L2
            if gb_i or gb_k:
                part = Pi_BB @ (gb_i * ZI + gb_k * ZK)
                H = part if H is None else H + part
            if H is None:
                continue
            H = H.reshape(naux, no, nv)
            if close == 'oo':
                # oo_ia = Σ_Qk B~_Q,ik b_k H_Q,ka, weighted a_a
                term = a[None, :] * (Boo_i @ (H * b[None, :, None]).reshape(-1, nv))
            else:
                # vv_ia = Σ_Qc B~_Q,ac a_c H_Q,ic, weighted b_i
                Ha = (H * a[None, None, :]).transpose(1, 0, 2).reshape(no, -1)
                term = b[:, None] * (Ha @ Bvv_a)
            v = term if v is None else v + term
        if v is None:
            continue
        for k, c in enumerate(coefs):
            sums[k] += (s * c(t, w)) * v
    return [x.ravel() for x in sums]


def _mode_laplace(xp, rho, om, Boo_i, Bvv_a, eo, ev, ZI, ZK, tau, wt, lo, coefs,
                  square):
    """Weighted Laplace sums of the mode set's closure vector of one channel vector.

    Parameters
    ----------
    xp : the array module, numpy or cupy
    rho : ndarray, shape (naux, nm), index order (P, m), the modes ρ_P,m
    om : ndarray, shape (nm,), Ω_m on xp
    Boo_i : ndarray, shape (no, naux no), index order (i, (Q, k)), the bare B_Q,ik
    Bvv_a : ndarray, shape (naux nv, nv), index order ((Q, c), a), the bare B_Q,ac
    eo, ev : ndarray, shape (no,), (nv,), the bare factors' energies on xp
    ZI, ZK : ndarray, shape (naux, no nv), index order (P, (k, c)), the bare vertex
        factors of _mode_images
    tau, wt : ndarray, shape (ntau,), the minimax grid, 1/x = Σ_t wt_t exp(-τ_t x)
    lo : float, min D_kcm - ω, the shift that keeps every exponent at or below zero
    coefs : list of callables c(τ, w) -> float, the weight of the node τ in each sum
    square : bool, False eq 61's cross J.K + K.J, True eq 66's square (J + K).(J + K)

    Returns
    -------
    list of ndarray, shape (no nv,), Σ_t c(τ_t, w_t) v_t for each c in coefs, v_t the
        closure vector of exp(-τ_t (D - ω)) in place of 1/(D - ω)

    Notes
    -----
    exp(-τ (D_kcm - ω)) = s e_m b_k a_c with s = exp(-τ lo), e_m = exp(-τ (Ω_m -
    Ω_min)), a_c = exp(-τ (ε_c - min ε_v)) and b_k = exp(-τ (max ε_o - ε_k)). The mode
    sum is the screened interaction at imaginary time τ, Π^τ_PQ = Σ_m e_m ρ_P,m ρ_Q,m
    (naux² nm), and with H = Π^τ Z (naux² no nv) the two closures of the cross,
    Jᵀ(e Y^K) + Kᵀ(e Y^J) with J = -δ_ac W^m_ik and K = +δ_ik W^m_ac, read

        oo_ia = a_a Σ_Qk B_Q,ik b_k (Π^τ ZK)_Q,ka,
        vv_ia = b_i Σ_Qc B_Q,ac a_c (Π^τ ZI)_Q,ic,
        v_t   = -2 s (oo + vv),

    the minus J's; the square takes H = Π^τ (ZK - ZI) on both closures and v_t = -2 s
    (oo - vv). The image at ω is Σ_t (-w_t) v_t, since 1/(ω - D) = -1/(D - ω), its
    ω-derivative Σ_t (-w_t τ_t) v_t, and u . Σ_t (w_t τ_t) v_t is the signed norm,
    2 Σ Y^J Y^K / (D - ω)² on the cross and Σ (Y^J + Y^K)² / (D - ω)² on the square.
    """
    no, nv = int(eo.size), int(ev.size)
    ev_min, eo_max, om_min = float(ev.min()), float(eo.max()), float(om.min())
    sums = [xp.zeros((no, nv)) for _ in coefs]
    if square:
        # one factor on both closures
        ZK = ZK - ZI
    for t, w in zip(tau, wt):
        a = xp.exp(-t * (ev - ev_min))
        b = xp.exp(-t * (eo_max - eo))
        e = xp.exp(-t * (om - om_min))
        s = float(np.exp(-t * lo))
        # Π^τ_PQ = Σ_m e_m ρ_P,m ρ_Q,m
        Pi = (rho * e[None, :]) @ rho.T
        # oo_ia = a_a Σ_Qk B_Q,ik b_k H_Q,ka with H = Π^τ ZK
        H = (Pi @ ZK).reshape(-1, no, nv)
        oo = a[None, :] * (Boo_i @ (H * b[None, :, None]).reshape(-1, nv))
        if not square:
            # the first H dies before the second forms
            H = None
            H = (Pi @ ZI).reshape(-1, no, nv)
        # vv_ia = b_i Σ_Qc B_Q,ac a_c H_Q,ic with H = Π^τ ZI, or Π^τ (ZK - ZI) again
        Ha = (H * a[None, None, :]).transpose(1, 0, 2).reshape(no, -1)
        vv = b[:, None] * (Ha @ Bvv_a)
        v = (-2.0 * s) * ((oo - vv) if square else (oo + vv))
        for k, c in enumerate(coefs):
            sums[k] += c(t, w) * v
        Pi = H = Ha = None      # freed before the next node's arrays form
    return [x.ravel() for x in sums]


def _dot_weight(wr, ws):
    """doubles_dot's Laplace weight over one grid for both frequencies: with x = D -
    ω_a, ω_a = max(ω_r, ω_s), and Δ = |ω_r - ω_s|, 1/(x (x + Δ)) = Σ_t w_t e^{-τ_t x}
    (1 - e^{-τ_t Δ})/Δ, the weights w_t τ_t at Δ = 0 (well conditioned at every Δ,
    where the difference of two sums over Δ is not)."""
    delta = abs(wr - ws)
    if delta == 0.0:
        return lambda t, w: w * t
    return lambda t, w: -w * np.expm1(-t * delta) / delta


class _DoublesDBSE2GPU:
    """The doubles of an RIFoldDBSE2 on this rank's device through the array module xp
    (CuPy, FP64; numpy stands in for the host test of the same lines), the residents
    by op.sets: for eq 71's set the dressed factor L2, for form 'laplace' the dressed oo
    and vv blocks too; for a mode set the modes rho and om, and for form 'laplace' the
    bare oo and vv blocks too; the energies always. A
    column's vertex factors go up once per factor set (upload); the exact form runs
    _tile_core and _tile_overlap on xp and hands the Gam halves and the norm back to
    the host, the Laplace form runs _laplace_sums whole on xp, the mode set runs
    _mode_tile and _mode_overlap, or _mode_laplace, on xp. The arithmetic is the host
    path's; the device
    sums in another order, so the two agree to rounding."""

    def __init__(self, op, xp=None):
        xp = cp if xp is None else xp
        if xp is None:
            raise ImportError("device='gpu' needs CuPy, which is not importable")
        self.xp = xp
        self.pair, self.spin, self.square = op.pair, op.spin, op.square
        self.to_host = getattr(xp, 'asnumpy', np.asarray)
        has71, mode_set = 'eq71' in op.sets, op.rho is not None
        laplace = op.form == 'laplace'
        # per set: its residents, the uploaded (ZI, ZK) of two vectors (doubles_dot's
        # four images) and one tile at its peak; the Laplace form adds its residents
        # and one node, and keeps the tile for the nodes that fall back to it
        need, parts = 0, []
        if has71:
            need += 5 * op.L2.nbytes + _tile_bytes(op.kt, op.no, op.nv, op.f.naux)
            parts.append('the dressed factor, the vertex factors of two vectors and '
                         'one tile')
            if laplace:
                need += (op.Boo_i.nbytes + op.Bvv_a.nbytes
                         + _laplace_bytes(op.no, op.nv, op.f.naux))
                parts[-1] = ('the dressed factors, their vertex factors, one Laplace '
                             'node and the fallback tile')
        if mode_set:
            need += (op.rho.nbytes + op.om.nbytes + 4 * op.L2b.nbytes
                     + _mode_bytes(op.kt, op.nv, op.nm, op.m_op.f.naux))
            parts.append('the modes, the bare vertex factors of two vectors and one '
                         'mode tile')
            if laplace:
                need += (op.Boo_ib.nbytes + op.Bvv_ab.nbytes
                         + _mode_laplace_bytes(op.no, op.nv, op.m_op.f.naux, op.nm))
                parts[-1] = ('the modes, the bare factors, their vertex factors, one '
                             'mode node and the fallback mode tile')
        self.need = need
        if hasattr(xp, 'cuda'):
            free = (xp.cuda.Device().mem_info[0]
                    + xp.get_default_memory_pool().free_bytes())
            device_memory_check(need, free, ' and '.join(parts))
        self.eo, self.ev = xp.asarray(op.f.eo), xp.asarray(op.f.ev)
        if has71:
            self.L2 = xp.asarray(op.L2)
            if laplace:
                self.Boo_i, self.Bvv_a = xp.asarray(op.Boo_i), xp.asarray(op.Bvv_a)
        if mode_set:
            self.rho, self.om = xp.asarray(op.rho), xp.asarray(op.om)
            if laplace:
                self.Boo_ib, self.Bvv_ab = xp.asarray(op.Boo_ib), xp.asarray(op.Bvv_ab)

    def upload(self, Z):
        """The vertex factors (ZI, ZK) of one column on the device."""
        return tuple(self.xp.asarray(z) for z in Z)

    def mode_tile(self, k0, k1, Zb, nodes, derivative):
        """_mode_tile on the device: (gam_J, gam_K, nrm), the halves as host arrays."""
        gJ, gK, nrm = _mode_tile(self.xp, self.rho, self.om, self.eo, self.ev, k0, k1,
                                 Zb[0], Zb[1], nodes, derivative, self.square)
        return self.to_host(gJ), self.to_host(gK), nrm

    def mode_overlap(self, k0, k1, Zr, wr, Zs, ws):
        """_mode_overlap on the device, a float."""
        return _mode_overlap(self.xp, self.rho, self.om, self.eo, self.ev, k0, k1, Zr,
                             wr, Zs, ws, self.square)

    def tile(self, k0, k1, Z, nodes, derivative):
        """_tile_core on the device: (gam_I, gam_K, nrm), the halves as host arrays."""
        gI, gK, nrm = _tile_core(self.xp, self.L2, self.eo, self.ev, k0, k1, Z[0],
                                 Z[1], self.pair, self.spin, nodes, derivative)
        return self.to_host(gI), self.to_host(gK), nrm

    def overlap(self, k0, k1, Zr, wr, Zs, ws):
        """_tile_overlap on the device, a float."""
        return _tile_overlap(self.xp, self.L2, self.eo, self.ev, k0, k1, Zr, wr, Zs,
                             ws, self.pair, self.spin)

    def laplace(self, Z, g, tau, wt, lo, coefs):
        """_laplace_sums on the device, the sums as host arrays."""
        return [self.to_host(v) for v in _laplace_sums(
            self.xp, self.L2, self.Boo_i, self.Bvv_a, self.eo, self.ev, Z[0], Z[1], g,
            tau, wt, lo, coefs)]

    def mode_laplace(self, Zb, tau, wt, lo, coefs):
        """_mode_laplace on the device, the sums as host arrays."""
        return [self.to_host(v) for v in _mode_laplace(
            self.xp, self.rho, self.om, self.Boo_ib, self.Bvv_ab, self.eo, self.ev,
            Zb[0], Zb[1], tau, wt, lo, coefs, self.square)]


def _resolve_form(form, comm, has71, mode_set, bare, dressed):
    """The form the operator runs: `form` as given (an unknown one raises
    ValueError), or for None 'laplace' where it runs and 'exact' with a UserWarning
    naming the reason otherwise. The Laplace closures run on one rank with the B_vv
    of each set in use whole (bare_factors keeps it in auxiliary chunks, on one rank
    too); an explicit 'laplace' that cannot run raises here, before any factor, mode
    or inner operator is built."""
    if form is not None and form not in FORMS:
        raise ValueError(f"form={form!r}; expected one of {FORMS} or None")
    if form == 'exact':
        return form
    reason = None
    if (1 if comm is None else comm.Get_size()) > 1:
        reason = 'several ranks'
    elif has71 and dressed is not None and dressed.Bvv is None:
        reason = "eq 71's dressed B_vv in auxiliary chunks"
    elif mode_set and bare.Bvv is None:
        reason = "the mode set's bare B_vv in auxiliary chunks"
    if reason is None:
        return 'laplace'
    if form == 'laplace':
        raise NotImplementedError('the Laplace form runs on one rank with every B_vv '
                                  f"whole, not with {reason}; pass form='exact'")
    warnings.warn(f'form=None with {reason}: the Laplace form runs on one rank with '
                  "every B_vv whole, so this operator takes form='exact'",
                  UserWarning, stacklevel=3)
    return 'exact'


class RIFoldDBSE2(RIFoldADC2):
    """A_eff(ω) of one spin channel at level 'dbse2' on flat singles vectors u (no nv,),
    (i, a) row-major: the singles block M of eq 62 at the GW energies plus the doubles
    of `sets` folded in, eq 71's through the dressed factors and eq 61's through the
    bare factors and the screening modes.

    bare : RIFactors of the DF factors B at the GW energies, for M and the mode set;
    dressed : RIFactors of the dressed factors B~ of ee_dbse2_pieces.dressed_factors at
    the same energies, for eq 71's doubles and eq 59, None when neither is asked (else
    ValueError naming dressed). kernel : 'eq71' or 'printed' (ee_dbse2_pieces.KERNELS),
    eq 71's set's; singles : 'eq62' or 'eq62+eq59' (its SINGLES; beside eq 61's set
    the latter counts the static W twice; dbse2 takes 'eq62'); form : None (the
    default: 'laplace' on one rank with every B_vv whole, else 'exact' with a
    UserWarning naming the reason), 'laplace' (NotImplementedError where it cannot
    run) or 'exact' (FORMS, the module docstring), for every set in `sets`; sets :
    one of SETS in any order, ('eq66',) level gw at the HF energies (no
    dressed factors, singles 'eq62', kernel unread; another set beside it names sets);
    modes : (omega (nm,), rho (naux, nm)) of gw_modes_df at the HF energies, required
    by a mode set, eq 61's or eq 66's (ValueError naming modes; a rho of another row
    count than bare.naux names naux). tile_gb, threads, comm,
    aux_chunk as RIFoldADC2 (threads reaches the inner operator and the
    back-transformation; the tile's element-wise passes run on one core; rho is held
    whole on every rank, never chunked); device 'cpu',
    or 'gpu' for the doubles of every set and both forms on this rank's GPU
    (_DoublesDBSE2GPU, CuPy). Attributes: sets, screening (from_mo's, else None), nm,
    om, rho, L2b (the
    bare factor as (naux, no nv)), d_min the lower of the sets' minima, kt from
    _tile_bytes plus _mode_bytes, dm_min and dm_max the mode set's own extremes of D
    (None without one). The inherited fold_diagonal returns zeros (no second-order
    model at this level); pieces() hands the operator to ee_fold.solve_folded;
    laplace_fallbacks counts, per set, the (column, node) pairs the Laplace form handed
    to that set's exact tile, plus one per set per doubles_dot call that did.
    """

    def __init__(self, bare, dressed, spin, kernel='eq71', singles='eq62',
                 form=None, sets=('eq71',), modes=None, tile_gb=EE_RI_TILE_GB,
                 threads=None, comm=None, aux_chunk=None, device='cpu'):
        sets = tuple(sorted(sets))
        if sets not in SETS:
            raise ValueError(f"sets={sets!r}; expected one of {SETS}")
        has71, square = 'eq71' in sets, sets == ('eq66',)
        mode_set = 'eq61' in sets or square
        if square and singles != 'eq62':
            raise ValueError(f"singles={singles!r} with sets=('eq66',): level gw's "
                             "singles block is eq 62 plus eq 70; pass singles='eq62'")
        if square and dressed is not None:
            raise ValueError("dressed factors with sets=('eq66',): level gw has no W "
                             'lines; pass dressed=None')
        if kernel not in _pieces.KERNELS:
            raise ValueError(f"kernel={kernel!r}; expected one of {_pieces.KERNELS}")
        if singles not in _pieces.SINGLES:
            raise ValueError(f"singles={singles!r}; expected one of {_pieces.SINGLES}")
        if device not in DEVICES:
            raise ValueError(f'device={device!r}; expected one of {DEVICES}')
        if device == 'gpu' and cp is None:
            raise ImportError("device='gpu' needs CuPy, which is not importable")
        if dressed is None and (has71 or singles == 'eq62+eq59'):
            raise ValueError(f"dressed=None with sets={sets!r}, singles={singles!r}: "
                             "eq 71's set and eq 59's exchange take dressed factors")
        if dressed is not None:
            if not (np.array_equal(bare.eo, dressed.eo)
                    and np.array_equal(bare.ev, dressed.ev)):
                raise ValueError('bare and dressed factors carry different energies; '
                                 'both take the GW energies')
            if bare.naux != dressed.naux:
                raise ValueError('bare and dressed factors carry different auxiliary '
                                 f'dimensions, {bare.naux} and {dressed.naux}')
        om = rho = None
        if mode_set:
            if modes is None:
                raise ValueError(f"modes=None with sets={sets!r}: a mode set takes "
                                 '(omega, rho) of gw_modes_df at the HF energies')
            om, rho = np.asarray(modes[0], float), np.asarray(modes[1], float)
            if rho.ndim != 2 or rho.shape[0] != bare.naux:
                raise ValueError(f'modes: rho of shape {rho.shape}; expected (naux='
                                 f'{bare.naux}, nm), index order (P, m)')
            if om.shape != (rho.shape[1],):
                raise ValueError(f'modes: omega of shape {om.shape} against rho '
                                 f'{rho.shape}; expected (nm,) with nm = rho.shape[1]')
        form = _resolve_form(form, comm, has71, mode_set, bare, dressed)
        self.sets, self.kernel, self.singles, self.form = sets, kernel, singles, form
        self.screening = None
        self.pair = _pieces.coupling_pairs(kernel, 'singlet')[0]
        self.laplace_fallbacks = 0
        self.square = square
        # eq 70's Ā as its two blocks, read by _setup and _singles
        self.P, self.Q = ((None, None) if not square else
                          static_term_pq(np.concatenate([bare.eo, bare.ev]), bare.LT,
                                         rho, om, bare.no))
        # M = eq 62 at the GW energies: CIS on the bare factors
        self.m_op = RIFoldADC2(bare, spin, level='adc1', tile_gb=tile_gb,
                               threads=threads, comm=comm, aux_chunk=aux_chunk)
        super().__init__(bare if dressed is None else dressed, spin, level='adc1',
                         tile_gb=tile_gb, threads=threads, comm=comm,
                         aux_chunk=aux_chunk)
        self.level = 'gw' if square else 'dbse2'
        # L2[P, (l, d)] = B~_P,ld, the dressed factor eq 71's images close through;
        # L2b the bare one of the mode set
        self.L2 = self.LT.reshape(self.f.naux, -1)
        self.L2b = self.m_op.LT.reshape(bare.naux, -1)
        self.om, self.rho, self.nm = om, rho, 0 if om is None else int(om.size)
        per_k = float(_tile_bytes(1, self.no, self.nv, self.f.naux)) if has71 else 0.0
        # eq 71's own minimum, the Laplace grid's lower end and exponent shift (_grid);
        # the level's d_min may fall below it through the mode set
        self.d71_min = self.d_min if has71 else None
        self.dm_min = self.dm_max = None
        if mode_set:
            fb = self.m_op.f
            # Ω_min + min ε_v - max ε_o, against eq 71's 2 (min ε_v - max ε_o); the
            # mode set's extremes span its own Laplace grid
            self.dm_min = float(om.min() + fb.ev.min() - fb.eo.max())
            self.dm_max = float(om.max() + fb.ev.max() - fb.eo.min())
            self.d_min = min(self.d_min, self.dm_min) if has71 else self.dm_min
            per_k += float(_mode_bytes(1, self.nv, self.nm, bare.naux))
        # the tile budget holds one tile of every set at its peak, per occupied row
        self.kt = int(max(1, min(-(-self.no // self.size),
                                 tile_gb * 1024 ** 3 // per_k)))
        self.tiles = [(k0, min(k0 + self.kt, self.no))
                      for k0 in range(0, self.no, self.kt)]
        self.my_tiles = [(t, self.tiles[t])
                         for t in range(self.rank, len(self.tiles), self.size)]
        # the inner operator's tiles follow this operator's: its back-transformation
        # closes the mode halves on these tiles (its own level has no doubles)
        self.m_op.kt, self.m_op.tiles, self.m_op.my_tiles = (self.kt, self.tiles,
                                                             self.my_tiles)
        if form == 'laplace':
            if has71:
                self.d_max = 2.0 * (self.f.ev.max() - self.f.eo.min())
                self.g = _closure_coefficients(self.pair, spin)
                # the closures' layouts: B~_Q,ik as [i, (Q, k)] (one copy) and B~_Q,ac
                # as [(Q, c), a], a view of Bvv's [a, Q, c]
                self.Boo_i = np.ascontiguousarray(
                    self.f.BooP.transpose(1, 0, 2)).reshape(self.no, -1)
                self.Bvv_a = self.f.Bvv.reshape(self.nv, -1).T
            if mode_set:
                # the same layouts of the bare factors for the mode set's closures
                fb = self.m_op.f
                self.Boo_ib = np.ascontiguousarray(
                    fb.BooP.transpose(1, 0, 2)).reshape(self.no, -1)
                self.Bvv_ab = fb.Bvv.reshape(self.nv, -1).T
        self.gpu = _DoublesDBSE2GPU(self) if device == 'gpu' else None

    @classmethod
    def from_mo(cls, eps_gw, eps_hf, B, nocc, spin, screening='rpa', **kw):
        """From the MO quantities: check_energies, the bare RIFactors at the GW
        energies, dressed_factors at the HF energies when eq 71's set or eq 59 is asked,
        gw_modes_df at the HF energies when a mode set is, then cls(bare, dressed, spin,
        **kw).

        Parameters
        ----------
        eps_gw, eps_hf : ndarray, shape (norb,), Hartree, occupied first
        B : ndarray, shape (naux, norb, norb), index order (Q, p, q)
        nocc : int
        spin : {'singlet', 'triplet'}
        screening : {'tda', 'rpa'}, the screening of the modes (read for a mode set)
        **kw : the constructor's keywords (kernel, singles, form, sets, modes, tile_gb,
            threads, comm, aux_chunk, device); modes given here are taken as they are

        Returns
        -------
        RIFoldDBSE2
        """
        eps_gw, eps_hf, no, _ = _pieces.check_energies(eps_gw, eps_hf, B, nocc)
        sets = tuple(sorted(kw.get('sets', ('eq71',))))
        # level gw sits at the reference energies throughout: GW energies on D and
        # A-bar beside HF modes would be a level that exists nowhere
        if 'eq66' in sets and not np.array_equal(eps_gw, eps_hf):
            raise ValueError("sets=('eq66',) with eps_gw differing from eps_hf: level "
                             'gw runs at the reference energies; pass the same '
                             'energies twice')
        bare = RIFactors.from_full(eps_gw, B, no)
        dressed = None
        if 'eq71' in sets or kw.get('singles', 'eq62') == 'eq62+eq59':
            _, Bt = _pieces.dressed_factors(eps_hf, B, no)
            dressed = RIFactors.from_full(eps_gw, Bt, no)
        mode_set = 'eq61' in sets or 'eq66' in sets
        kw['form'] = _resolve_form(kw.get('form'), kw.get('comm'), 'eq71' in sets,
                                   mode_set, bare, dressed)
        if mode_set and kw.get('modes') is None:
            kw['modes'] = gw_modes_df(eps_hf, B, no, screening)
        op = cls(bare, dressed, spin, **kw)
        op.screening = screening if mode_set else None
        return op

    # ---- one-time pieces --------------------------------------------------

    def _exch_diag(self, f):
        """(ii|aa) of one factor set, (no nv,), the auxiliary chunks in order."""
        Bii = np.ascontiguousarray(np.einsum('iiP->iP', f.Boo))
        terms = {}
        for c in self.my_chunks:
            p0, p1 = self.chunks[c]
            terms[c] = (Bii[:, p0:p1] @ np.einsum('aPa->Pa', f.bvv(c, p0, p1))).ravel()
        return ordered_chain_sum(terms, self.owners, self.comm,
                                 np.zeros(self.no * self.nv))

    def _setup(self):
        """The singles diagonal: M's from the inner operator; for singles 'eq62+eq59'
        eq 59's swap on it, (ii|aa) - w(ii|aa); for level gw eq 70's diagonal,
        (P + Pᵀ)_aa - (Q + Qᵀ)_ii. No second-order pieces at this level."""
        diag = self.m_op.diag
        if self.singles == 'eq62+eq59':
            diag = diag + self._exch_diag(self.m_op.f) - self._exch_diag(self.f)
        if self.square:
            # Ā_ia,ia = (P + Pᵀ)_aa - (Q + Qᵀ)_ii
            diag = diag + (2.0 * np.diag(self.P)[None, :]
                           - 2.0 * np.diag(self.Q)[:, None]).ravel()
        self.diag = diag

    def _grid(self, omega, omega2=None, span=None):
        """(tau, wt, lo) of the minimax grid for 1/(D - ω) over one set's doubles on
        [min D - max ω, max D - min ω] over the one or two frequencies at
        LAPLACE_TARGET, lo its lower end and the exponent shift of the set's sums; span
        is the set's own (min D, max D), eq 71's (d71_min, d_max) by default and the
        mode set's (dm_min, dm_max) for _mode_laplace, since each set's split is
        against its own extremes and not the level's; None where a frequency is at or
        above the set's min D (no Laplace transform) or no tabulated grid reaches the
        target: kernel_laplace's rule."""
        d_lo, d_hi = (self.d71_min, self.d_max) if span is None else span
        omegas = (omega,) if omega2 is None else (omega, omega2)
        if max(omegas) >= d_lo:
            return None
        lo, hi = d_lo - max(omegas), d_hi - min(omegas)
        ntau, err = minimax_points_for_accuracy(lo, hi, LAPLACE_TARGET, relative=True)
        if ntau is None or err > LAPLACE_TARGET:
            return None
        tau, wt = minimax_time_grid(ntau, lo, hi)
        return np.asarray(tau, float), np.asarray(wt, float), lo

    # ---- the singles block and the vertex factors -------------------------

    def _singles(self, U):
        """M U (n, m): eq 62 at the GW energies from the inner operator; for singles
        'eq62+eq59' the exchange swap -(ij|ab) -> -w(ij|ab) on top; for level gw eq 70's
        Ā u = u (P + Pᵀ) - (Q + Qᵀ) u per column."""
        out = self.m_op.apply(U, None)[0]
        if self.singles == 'eq62+eq59':
            out = (out + self.m_op._exchange(U, ('oovv',))[0]
                   - self._exchange(U, ('oovv',))[0])
        if self.square:
            Pt, Qt = self.P + self.P.T, self.Q + self.Q.T
            for j in range(U.shape[1]):
                u = U[:, j].reshape(self.no, self.nv)
                # (Ā u)_ia = Σ_b (P + Pᵀ)_ab u_ib - Σ_j (Q + Qᵀ)_ij u_ja
                out[:, j] += (u @ Pt - Qt @ u).ravel()
        return out

    def _vertex_factors(self, u, f=None):
        """(ZI, ZK), each (naux, no nv) auxiliary-major, index order (P, (k, c)):
        ZI_P,kc = Σ_i B~_P,ik u_ic and ZK_P,kc = Σ_a u_ka B~_P,ac on the factors f
        (self.f, the dressed ones, by default; self.m_op.f for the bare ones of the
        mode set). Each rank forms the rows of its auxiliary chunks; they travel whole
        to every rank (allgather_ranges), as RIFoldADC2._dressed does."""
        f = self.f if f is None else f
        no, nv = self.no, self.nv
        u = np.ascontiguousarray(u, float).reshape(no, nv)
        ZI = np.zeros((f.naux, no * nv))
        ZK = np.zeros((f.naux, no * nv))
        for c in self.my_chunks:
            p0, p1 = self.chunks[c]
            pc = p1 - p0
            # ZI_P,kc = Σ_i B~_P,ki u_ic (B~_oo symmetric)
            ZI[p0:p1] = (f.BooP[p0:p1] @ u).reshape(pc, -1)
            # ZK_P,kc = Σ_a u_ka B~_P,ac, bvv as [a, P, c]
            ZK[p0:p1] = (u @ f.bvv(c, p0, p1).reshape(nv, -1)).reshape(
                no, pc, nv).transpose(1, 0, 2).reshape(pc, -1)
        return (allgather_ranges(ZI, self.chunk_ranges, self.comm),
                allgather_ranges(ZK, self.chunk_ranges, self.comm))

    # ---- the doubles ------------------------------------------------------

    def _doubles_exact(self, Z, nodes, derivative):
        """(F (p, n), nrm (p,)) of the exact tile: Iᵀ R_I + Kᵀ R_K of the channel vector
        with vertex factors Z = (ZI, ZK), on the device where self.gpu is set, at the
        nodes, the folded doubles (or their
        ω-derivative), and the signed norm at each node. The tiles' Iᵀ halves and
        norms are summed in tile order, the Kᵀ halves by the auxiliary chunks' owners
        in chunk order; no collective inside the tile loop."""
        n, p = self.n, nodes.size
        ZI, ZK = Z
        parts, gams_K = [], {}
        for t, (k0, k1) in self.my_tiles:
            if self.gpu is None:
                gI, gK, nrm = _tile_core(np, self.L2, self.f.eo, self.f.ev, k0, k1, ZI,
                                         ZK, self.pair, self.spin, nodes, derivative)
            else:
                gI, gK, nrm = self.gpu.tile(k0, k1, Z, nodes, derivative)
            parts.append((t, np.concatenate([self._back_oo(gI, k0, k1).ravel(), nrm])))
            gams_K[t] = gK
        tot = ordered_sum(parts, self.comm, shape=(p * n + p,),
                          onto=np.zeros(p * n + p))
        # _back_vv carries ADC(2)'s minus sign; Kᵀ R enters with plus
        F = tot[:p * n].reshape(p, n) - self._back_vv(gams_K, p)
        return F, tot[p * n:]

    def _laplace(self, Z, grid, coefs):
        """_laplace_sums at one grid, on the host or the device."""
        tau, wt, lo = grid
        if self.gpu is not None:
            return self.gpu.laplace(Z, self.g, tau, wt, lo, coefs)
        return _laplace_sums(np, self.L2, self.Boo_i, self.Bvv_a, self.f.eo, self.f.ev,
                             Z[0], Z[1], self.g, tau, wt, lo, coefs)

    def _modes_exact(self, Zb, nodes, derivative):
        """(F (p, n), nrm (p,)) of the mode set's exact tile: Jᵀ R_J + Kᵀ R_K of the
        channel vector with bare vertex factors Zb = (ZI, ZK) at the nodes (or the
        ω-derivative) and the signed norm, the tiles' Jᵀ halves and norms summed in
        tile order, the Kᵀ halves by the auxiliary chunks' owners in chunk order
        through the inner operator's _back_vv on this operator's tiles; no collective
        inside the tile loop."""
        n, p = self.n, nodes.size
        fb = self.m_op.f
        parts, gams_K = [], {}
        for t, (k0, k1) in self.my_tiles:
            if self.gpu is None:
                gJ, gK, nrm = _mode_tile(np, self.rho, self.om, fb.eo, fb.ev, k0, k1,
                                         Zb[0], Zb[1], nodes, derivative, self.square)
            else:
                gJ, gK, nrm = self.gpu.mode_tile(k0, k1, Zb, nodes, derivative)
            parts.append((t, np.concatenate([self.m_op._back_oo(gJ, k0, k1).ravel(),
                                             nrm])))
            gams_K[t] = gK
        tot = ordered_sum(parts, self.comm, shape=(p * n + p,),
                          onto=np.zeros(p * n + p))
        # Jᵀ R_J = -√2 _back_oo(Γ_J); Kᵀ R_K = +√2 Σ_Pc B_P,ac Γ_ic,P, which is
        # -√2 _back_vv(Γ_K) against its minus
        F = -np.sqrt(2.0) * (tot[:p * n].reshape(p, n) + self.m_op._back_vv(gams_K, p))
        return F, tot[p * n:]

    def _mode_laplace(self, Zb, grid, coefs):
        """_mode_laplace at one grid, on the host or the device."""
        tau, wt, lo = grid
        if self.gpu is not None:
            return self.gpu.mode_laplace(Zb, tau, wt, lo, coefs)
        fb = self.m_op.f
        return _mode_laplace(np, self.rho, self.om, self.Boo_ib, self.Bvv_ab, fb.eo,
                             fb.ev, Zb[0], Zb[1], tau, wt, lo, coefs, self.square)

    def _doubles_modes(self, u, nodes, derivative):
        """(F (p, n), nrm (p,)) of the mode set by the operator's form; a Laplace node
        without a grid goes through the mode tile."""
        Zb = self._vertex_factors(u, self.m_op.f)
        if self.gpu is not None:
            Zb = self.gpu.upload(Zb)
        if self.form == 'exact':
            return self._modes_exact(Zb, nodes, derivative)
        return self._laplace_nodes(u, Zb, nodes, derivative, (self.dm_min, self.dm_max),
                                   self._mode_laplace, self._modes_exact)

    def _doubles(self, u, nodes, derivative):
        """(F (p, n), nrm (p,)): the folded doubles of one channel vector at the nodes
        (or their ω-derivative) and the signed norm at each node, summed over the
        sets, each by the operator's form."""
        p = nodes.size
        F, nrm = np.zeros((p, self.n)), np.zeros(p)
        if 'eq71' in self.sets:
            F71, n71 = self._doubles_eq71(u, nodes, derivative)
            F += F71
            nrm += n71
        if self.rho is not None:
            Fm, nm = self._doubles_modes(u, nodes, derivative)
            F += Fm
            nrm += nm
        return F, nrm

    def _doubles_eq71(self, u, nodes, derivative):
        """(F (p, n), nrm (p,)) of eq 71's set by the operator's form; a Laplace node
        without a grid goes through the exact tile."""
        Z = self._vertex_factors(u)
        if self.gpu is not None:
            Z = self.gpu.upload(Z)
        if self.form == 'exact':
            return self._doubles_exact(Z, nodes, derivative)
        return self._laplace_nodes(u, Z, nodes, derivative, None, self._laplace,
                                   self._doubles_exact)

    def _laplace_nodes(self, u, Z, nodes, derivative, span, sums, exact):
        """(F (p, n), nrm (p,)) of one set's Laplace form: the nodes at one frequency
        (an image and its derivative) share one grid over the set's span (_grid) and
        one pass of `sums` (Z, grid, coefs), a coefficient each; the nodes without a
        grid go through `exact` (Z, nodes, derivative), counted in
        laplace_fallbacks."""
        u = np.ravel(np.asarray(u, float))
        p = nodes.size
        F, nrm = np.zeros((p, self.n)), np.zeros(p)
        fall, done = [], np.zeros(p, bool)
        for q in range(p):
            if done[q]:
                continue
            group = [r for r in range(q, p) if nodes[r] == nodes[q]]
            done[group] = True
            grid = self._grid(float(nodes[q]), None, span)
            if grid is None:
                fall.extend(group)
                continue
            # the image with -w_t (1/(ω - D) = -1/(D - ω)), its derivative with
            # -w_t τ_t; the norm Σ_K Zx_K Zy_K = u . Θ^(2) u with +w_t τ_t
            coefs = [(lambda t, w: -w * t) if derivative[r] else (lambda t, w: -w)
                     for r in group] + [lambda t, w: w * t]
            parts = sums(Z, grid, coefs)
            for r, v in zip(group, parts[:-1]):
                F[r] = v
            nrm[group] = float(np.dot(u, parts[-1]))
        if fall:
            self.laplace_fallbacks += len(fall)
            Fe, ne = exact(Z, nodes[fall], derivative[fall])
            F[fall], nrm[fall] = Fe, ne
        return F, nrm

    # ---- the RIFoldADC2 interface -----------------------------------------

    def apply(self, U, omegas=None):
        """(A_eff U, |Z+|² - |Z-|²): U (n, m); omegas None (M alone) or (m,) Hartree,
        one frequency per column; the signed norm (m,) of each column's doubles
        image."""
        U = np.ascontiguousarray(U, float)
        if U.ndim == 1:
            U = U[:, None]
        m = U.shape[1]
        self.matvecs += m
        out = self._singles(U)
        norms = np.zeros(m)
        if omegas is None:
            return out, norms
        omegas = np.broadcast_to(np.asarray(omegas, float), (m,))
        for j in range(m):
            F, nrm = self._doubles(U[:, j], np.array([omegas[j]]), np.array([False]))
            out[:, j] += F[0]
            norms[j] = nrm[0]
        return out, norms

    def fold_images(self, U, nodes, derivative=None):
        """(M U, F): F (p, n, m), F[q] = A_eff(ω_q) U - M U, or dA_eff/dω U where
        derivative[q]; a column's vertex factors are formed once for every node."""
        U = np.ascontiguousarray(U, float)
        if U.ndim == 1:
            U = U[:, None]
        nodes = np.atleast_1d(np.asarray(nodes, float))
        n, m, p = self.n, U.shape[1], nodes.size
        derivative = (np.zeros(p, bool) if derivative is None
                      else np.broadcast_to(np.asarray(derivative, bool), (p,)))
        MU = self._singles(U)
        self.matvecs += m
        self.image_columns += m
        self.node_images += m * p
        F = np.empty((p, n, m))
        for j in range(m):
            F[:, :, j] = self._doubles(U[:, j], nodes, derivative)[0]
        return MU, F

    def doubles_dot(self, ur, wr, us, ws):
        """Σ_K Y_K(ur, wr)ᵀ η Y_K(us, ws), the signed overlap of two doubles images,
        summed over the sets: eq 71's Σ_AB m_AB Σ_K (A ur)_K (B us)_K / ((D_K - wr)(D_K
        - ws)) and the mode set's, each by the operator's form.
        The Laplace form takes one grid over both frequencies and one pass over the
        factors of us with the weights of 1/((D - ω_a)(D - ω_b)), ω_a ≥ ω_b: with x = D
        - ω_a and Δ = ω_a - ω_b, 1/(x (x + Δ)) = Σ_t w_t e^{-τ_t x} (1 - e^{-τ_t Δ})/Δ,
        the weights w_t τ_t at Δ = 0 (well conditioned at every Δ, where the difference
        of two sums over Δ is not); the exact tile where a frequency has no grid."""
        ur, us = np.ravel(np.asarray(ur, float)), np.ravel(np.asarray(us, float))
        same = ur is us or np.array_equal(ur, us)
        wr, ws = float(wr), float(ws)
        v = 0.0
        if 'eq71' in self.sets:
            v += self._dot_eq71(ur, wr, us, ws, same)
        if self.rho is not None:
            v += self._dot_modes(ur, wr, us, ws, same)
        return v

    def _dot_modes(self, ur, wr, us, ws, same):
        """The mode set's share of doubles_dot by the operator's form: _mode_laplace
        on one grid over both frequencies (_dot_weight), else _mode_overlap over this
        rank's tiles, summed in tile order."""
        fb = self.m_op.f
        if self.form == 'laplace':
            grid = self._grid(wr, ws, (self.dm_min, self.dm_max))
            if grid is not None:
                Zs = self._vertex_factors(us, fb)
                if self.gpu is not None:
                    Zs = self.gpu.upload(Zs)
                v, = self._mode_laplace(Zs, grid, [_dot_weight(wr, ws)])
                return float(np.dot(ur, v))
            self.laplace_fallbacks += 1
        Zr = self._vertex_factors(ur, fb)
        Zs = Zr if same else self._vertex_factors(us, fb)
        if self.gpu is not None:
            Zr = self.gpu.upload(Zr)
            Zs = Zr if same else self.gpu.upload(Zs)
        parts = []
        for t, (k0, k1) in self.my_tiles:
            v = (_mode_overlap(np, self.rho, self.om, fb.eo, fb.ev, k0, k1, Zr, wr, Zs,
                               ws, self.square) if self.gpu is None
                 else self.gpu.mode_overlap(k0, k1, Zr, wr, Zs, ws))
            parts.append((t, np.array([v])))
        return float(ordered_sum(parts, self.comm, shape=(1,), onto=np.zeros(1))[0])

    def _dot_eq71(self, ur, wr, us, ws, same):
        """Eq 71's share of doubles_dot, by the operator's form."""
        if self.form == 'laplace':
            grid = self._grid(wr, ws)
            if grid is not None:
                Zs = self._vertex_factors(us)
                if self.gpu is not None:
                    Zs = self.gpu.upload(Zs)
                v, = self._laplace(Zs, grid, [_dot_weight(wr, ws)])
                return float(np.dot(ur, v))
            self.laplace_fallbacks += 1
        Zr = self._vertex_factors(ur)
        Zs = Zr if same else self._vertex_factors(us)
        if self.gpu is not None:
            Zr = self.gpu.upload(Zr)
            Zs = Zr if same else self.gpu.upload(Zs)
        parts = []
        for t, (k0, k1) in self.my_tiles:
            v = (_tile_overlap(np, self.L2, self.f.eo, self.f.ev, k0, k1, Zr, wr, Zs,
                               ws, self.pair, self.spin) if self.gpu is None
                 else self.gpu.overlap(k0, k1, Zr, wr, Zs, ws))
            parts.append((t, np.array([v])))
        return float(ordered_sum(parts, self.comm, shape=(1,), onto=np.zeros(1))[0])

    def pieces(self):
        """The dict ee_fold.solve_folded takes: the operator stands in for M, V, Vt, D
        and dnorm2 (pieces['operator']); d_min = min D at the GW energies."""
        return {'operator': self, 'level': self.level, 'no': self.no, 'nv': self.nv,
                'd_min': self.d_min, 'spin': self.spin, 'dense_limit': 0}
