"""Eq 71's doubles as a tiled RI operator (level 'dbse2'): A_eff(ω) of one spin channel
on flat singles vectors, in RIFoldADC2's interface, from two sets of DF factors at the
GW energies, on the host or a GPU, in two forms.

The kernel is ee_dbse2_pieces' (its module docstring: the vertices I, J, K, L on the
ordered doubles K = (k, l, c, d), D_K = ε_c + ε_d - ε_k - ε_l, the pair (x, y) of
coupling_pairs with X = Σ_A x_A A, Y = Σ_B y_B B, Θ = (X.Y + Y.X)/2, the copies
V± = (X ± Y)/2 on the metric η = diag(+1, -1)). Here the four spin blocks of the pieces
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

form='laplace' (spec §2.6): 1/(D_K - ω) = Σ_t w_t exp(-τ_t (D_K - ω)) on
[min D - ω, max D - ω], the minimax grid of grids.minimax_time_grid sized by
minimax_points_for_accuracy at kernel_laplace's relative target, 1/(D_K - ω)² with
the weights w_t τ_t. The exponential splits into one-index weights, and every product
A.B closes through an n_aux² intermediate at each τ, Π_QP = Σ_ld B~_Q,ld b_l a_d F_P,ld
with F one of B~, ZI, ZK (_laplace_sums' Notes), so no doubles-sized array exists.
A node at or above min D, or one whose range no tabulated grid reaches at the target,
goes through the exact tile (laplace_fallbacks counts them), kernel_laplace's rule. The
form runs on one rank with the dressed B~_vv whole.

The same cores run on numpy and on cupy (_tile_core and _laplace_sums take the array
module).

References
----------
E. Monino and P.-F. Loos, J. Chem. Phys. 159, 034105 (2023), eqs 59, 62, 71 to 74.
"""
import numpy as np

from src.Base.constants import EE_RI_TILE_GB
from src.Base.utils.grids import minimax_time_grid
from src.Base.utils.mpi_grid import allgather_ranges, ordered_chain_sum, ordered_sum
from src.Base.utils.time_frequency import minimax_points_for_accuracy
from src.SingleReference.ADC.eeADC import ee_dbse2_pieces as _pieces
from src.SingleReference.ADC.eeADC.ee_gw_pieces import _LAPLACE_TARGET as LAPLACE_TARGET
from src.SingleReference.ADC.eeADC.ee_ri_fold import DEVICES, RIFactors, RIFoldADC2
from src.SingleReference.LinearResponse.linear_response import device_memory_check

try:
    import cupy as cp
except ImportError:
    cp = None

FORMS = ('exact', 'laplace')
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


class _DoublesDBSE2GPU:
    """The doubles of an RIFoldDBSE2 on this rank's device through the array module xp
    (CuPy, FP64; numpy stands in for the host test of the same lines): the dressed
    factor L2 and the GW energies resident, for form 'laplace' the dressed oo and vv
    blocks too. A column's vertex factors go up once (upload); the exact form runs
    _tile_core and _tile_overlap on xp and hands the Gam halves and the norm back to
    the host, the Laplace form runs _laplace_sums whole on xp. The arithmetic is the
    host path's; the device sums in another order, so the two agree to rounding."""

    def __init__(self, op, xp=None):
        xp = cp if xp is None else xp
        if xp is None:
            raise ImportError("device='gpu' needs CuPy, which is not importable")
        self.xp = xp
        self.pair, self.spin = op.pair, op.spin
        self.to_host = getattr(xp, 'asnumpy', np.asarray)
        laplace = op.form == 'laplace'
        # the factor, one column's uploaded (ZI, ZK) and one tile at its peak; the
        # Laplace form adds its residents and one node, and keeps the tile for the
        # nodes that fall back to it
        need = 3 * op.L2.nbytes + _tile_bytes(op.kt, op.no, op.nv, op.f.naux)
        label = 'the dressed factor, the vertex factors and one tile'
        if laplace:
            need += (op.Boo_i.nbytes + op.Bvv_a.nbytes
                     + _laplace_bytes(op.no, op.nv, op.f.naux))
            label = ('the dressed factors, the vertex factors, one Laplace node and '
                     'the fallback tile')
        self.need = need
        if hasattr(xp, 'cuda'):
            free = (xp.cuda.Device().mem_info[0]
                    + xp.get_default_memory_pool().free_bytes())
            device_memory_check(need, free, label)
        self.L2 = xp.asarray(op.L2)
        self.eo, self.ev = xp.asarray(op.f.eo), xp.asarray(op.f.ev)
        if laplace:
            self.Boo_i, self.Bvv_a = xp.asarray(op.Boo_i), xp.asarray(op.Bvv_a)

    def upload(self, Z):
        """The vertex factors (ZI, ZK) of one column on the device."""
        return tuple(self.xp.asarray(z) for z in Z)

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


class RIFoldDBSE2(RIFoldADC2):
    """A_eff(ω) of one spin channel at level 'dbse2' on flat singles vectors u (no nv,),
    (i, a) row-major: the singles block M of eq 62 at the GW energies plus eq 71's
    doubles folded in, from two RIFactors at the GW energies.

    bare : RIFactors of the DF factors B, for M; dressed : RIFactors of the dressed
    factors B~ of ee_dbse2_pieces.dressed_factors, for the doubles (and eq 59); both
    carry the GW energies (eo, ev), else ValueError. kernel : 'eq71' or 'printed'
    (ee_dbse2_pieces.KERNELS); singles : 'eq62' or 'eq62+eq59' (its SINGLES); form :
    'exact' or 'laplace' (FORMS, the module docstring). tile_gb, threads, comm,
    aux_chunk as RIFoldADC2 (threads reaches the inner operator and the
    back-transformation; the tile's element-wise passes run on one core); device 'cpu',
    or 'gpu' for the doubles of both forms on this rank's GPU (_DoublesDBSE2GPU, CuPy).
    The inherited fold_diagonal returns zeros (no second-order model at this level);
    pieces() hands the operator to ee_fold.solve_folded; laplace_fallbacks counts the
    (column, node) pairs the Laplace form handed to the exact tile, plus one per
    doubles_dot call that did.
    """

    def __init__(self, bare, dressed, spin, kernel='eq71', singles='eq62', form='exact',
                 tile_gb=EE_RI_TILE_GB, threads=None, comm=None, aux_chunk=None,
                 device='cpu'):
        if not (np.array_equal(bare.eo, dressed.eo)
                and np.array_equal(bare.ev, dressed.ev)):
            raise ValueError('bare and dressed factors carry different energies; both '
                             'take the GW energies')
        if kernel not in _pieces.KERNELS:
            raise ValueError(f"kernel={kernel!r}; expected one of {_pieces.KERNELS}")
        if singles not in _pieces.SINGLES:
            raise ValueError(f"singles={singles!r}; expected one of {_pieces.SINGLES}")
        if form not in FORMS:
            raise ValueError(f"form={form!r}; expected one of {FORMS}")
        if device not in DEVICES:
            raise ValueError(f'device={device!r}; expected one of {DEVICES}')
        if device == 'gpu' and cp is None:
            raise ImportError("device='gpu' needs CuPy, which is not importable")
        if bare.naux != dressed.naux:
            raise ValueError('bare and dressed factors carry different auxiliary '
                             f'dimensions, {bare.naux} and {dressed.naux}')
        self.kernel, self.singles, self.form = kernel, singles, form
        self.pair = _pieces.coupling_pairs(kernel, 'singlet')[0]
        self.laplace_fallbacks = 0
        # M = eq 62 at the GW energies: CIS on the bare factors
        self.m_op = RIFoldADC2(bare, spin, level='adc1', tile_gb=tile_gb,
                               threads=threads, comm=comm, aux_chunk=aux_chunk)
        super().__init__(dressed, spin, level='adc1', tile_gb=tile_gb, threads=threads,
                         comm=comm, aux_chunk=aux_chunk)
        self.level = 'dbse2'
        # L2[P, (l, d)] = B~_P,ld, the dressed factor the images close through
        self.L2 = self.LT.reshape(self.f.naux, -1)
        # the tile budget holds one tile at its peak (_tile_bytes, per occupied row)
        per_k = float(_tile_bytes(1, self.no, self.nv, self.f.naux))
        self.kt = int(max(1, min(-(-self.no // self.size),
                                 tile_gb * 1024 ** 3 // per_k)))
        self.tiles = [(k0, min(k0 + self.kt, self.no))
                      for k0 in range(0, self.no, self.kt)]
        self.my_tiles = [(t, self.tiles[t])
                         for t in range(self.rank, len(self.tiles), self.size)]
        if form == 'laplace':
            if self.size > 1 or self.f.Bvv is None:
                raise NotImplementedError('the Laplace form runs on one rank with the '
                                          'dressed B_vv whole')
            self.d_max = 2.0 * (self.f.ev.max() - self.f.eo.min())
            self.g = _closure_coefficients(self.pair, spin)
            # the closures' layouts: B~_Q,ik as [i, (Q, k)] (one copy) and B~_Q,ac as
            # [(Q, c), a], a view of Bvv's [a, Q, c]
            self.Boo_i = np.ascontiguousarray(self.f.BooP.transpose(1, 0, 2)).reshape(
                self.no, -1)
            self.Bvv_a = self.f.Bvv.reshape(self.nv, -1).T
        self.gpu = _DoublesDBSE2GPU(self) if device == 'gpu' else None

    @classmethod
    def from_mo(cls, eps_gw, eps_hf, B, nocc, spin, **kw):
        """From the MO quantities: check_energies, dressed_factors at the HF energies,
        the two RIFactors at the GW energies, then cls(bare, dressed, spin, **kw).

        Parameters
        ----------
        eps_gw, eps_hf : ndarray, shape (norb,), Hartree, occupied first
        B : ndarray, shape (naux, norb, norb), index order (Q, p, q)
        nocc : int
        spin : {'singlet', 'triplet'}
        **kw : the constructor's keywords (kernel, singles, form, tile_gb, threads,
            comm, aux_chunk, device)

        Returns
        -------
        RIFoldDBSE2
        """
        eps_gw, eps_hf, no, _ = _pieces.check_energies(eps_gw, eps_hf, B, nocc)
        _, Bt = _pieces.dressed_factors(eps_hf, B, no)
        return cls(RIFactors.from_full(eps_gw, B, no),
                   RIFactors.from_full(eps_gw, Bt, no), spin, **kw)

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
        eq 59's swap on it, (ii|aa) - w(ii|aa). No second-order pieces at this level."""
        diag = self.m_op.diag
        if self.singles == 'eq62+eq59':
            diag = diag + self._exch_diag(self.m_op.f) - self._exch_diag(self.f)
        self.diag = diag

    def _grid(self, omega, omega2=None):
        """(tau, wt, lo) of the minimax grid for 1/(D - ω) on [min D - max ω, max D -
        min ω] over the one or two frequencies at LAPLACE_TARGET, lo its lower end, or
        None where a frequency is at or above min D (no Laplace transform) or no
        tabulated grid reaches the target: kernel_laplace's rule."""
        omegas = (omega,) if omega2 is None else (omega, omega2)
        if max(omegas) >= self.d_min:
            return None
        lo, hi = self.d_min - max(omegas), self.d_max - min(omegas)
        ntau, err = minimax_points_for_accuracy(lo, hi, LAPLACE_TARGET, relative=True)
        if ntau is None or err > LAPLACE_TARGET:
            return None
        tau, wt = minimax_time_grid(ntau, lo, hi)
        return np.asarray(tau, float), np.asarray(wt, float), lo

    # ---- the singles block and the vertex factors -------------------------

    def _singles(self, U):
        """M U (n, m): eq 62 at the GW energies from the inner operator; for singles
        'eq62+eq59' the exchange swap -(ij|ab) -> -w(ij|ab) on top."""
        out = self.m_op.apply(U, None)[0]
        if self.singles == 'eq62+eq59':
            out = (out + self.m_op._exchange(U, ('oovv',))[0]
                   - self._exchange(U, ('oovv',))[0])
        return out

    def _vertex_factors(self, u):
        """(ZI, ZK), each (naux, no nv) auxiliary-major, index order (P, (k, c)):
        ZI_P,kc = Σ_i B~_P,ik u_ic and ZK_P,kc = Σ_a u_ka B~_P,ac. Each rank forms the
        rows of its auxiliary chunks; they travel whole to every rank
        (allgather_ranges), as RIFoldADC2._dressed does."""
        f, no, nv = self.f, self.no, self.nv
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

    def _doubles(self, u, nodes, derivative):
        """(F (p, n), nrm (p,)): the folded doubles of one channel vector at the nodes
        (or their ω-derivative) and the signed norm at each node, by the operator's
        form; a Laplace node without a grid goes through the exact tile."""
        Z = self._vertex_factors(u)
        if self.gpu is not None:
            Z = self.gpu.upload(Z)
        if self.form == 'exact':
            return self._doubles_exact(Z, nodes, derivative)
        u = np.ravel(np.asarray(u, float))
        p = nodes.size
        F, nrm = np.zeros((p, self.n)), np.zeros(p)
        exact, done = [], np.zeros(p, bool)
        for q in range(p):
            if done[q]:
                continue
            # the nodes at this frequency (an image and its derivative) share one grid
            # and one pass, a coefficient each
            group = [r for r in range(q, p) if nodes[r] == nodes[q]]
            done[group] = True
            grid = self._grid(float(nodes[q]))
            if grid is None:
                exact.extend(group)
                continue
            # the image with -w_t (1/(ω - D) = -1/(D - ω)), its derivative with
            # -w_t τ_t; the norm Σ_K Zx_K Zy_K = u . Θ^(2) u with +w_t τ_t
            coefs = [(lambda t, w: -w * t) if derivative[r] else (lambda t, w: -w)
                     for r in group] + [lambda t, w: w * t]
            sums = self._laplace(Z, grid, coefs)
            for r, v in zip(group, sums[:-1]):
                F[r] = v
            nrm[group] = float(np.dot(u, sums[-1]))
        if exact:
            self.laplace_fallbacks += len(exact)
            Fe, ne = self._doubles_exact(Z, nodes[exact], derivative[exact])
            F[exact], nrm[exact] = Fe, ne
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
        """Σ_K Y_K(ur, wr)ᵀ η Y_K(us, ws), the signed overlap of two doubles images:
        Σ_AB m_AB Σ_K (A ur)_K (B us)_K / ((D_K - wr)(D_K - ws)). The Laplace form
        takes one grid over both frequencies and one pass over the factors of us with
        the weights of 1/((D - ω_a)(D - ω_b)), ω_a ≥ ω_b: with x = D - ω_a and
        Δ = ω_a - ω_b, 1/(x (x + Δ)) = Σ_t w_t e^{-τ_t x} (1 - e^{-τ_t Δ})/Δ, the
        weights w_t τ_t at Δ = 0 (well conditioned at every Δ, where the difference of
        two sums over Δ is not); the exact tile where a frequency has no grid."""
        ur, us = np.ravel(np.asarray(ur, float)), np.ravel(np.asarray(us, float))
        same = ur is us or np.array_equal(ur, us)
        wr, ws = float(wr), float(ws)
        if self.form == 'laplace':
            grid = self._grid(wr, ws)
            if grid is not None:
                Zs = self._vertex_factors(us)
                if self.gpu is not None:
                    Zs = self.gpu.upload(Zs)
                delta = abs(wr - ws)
                # c_t = w_t (1 - e^{-τ_t Δ})/Δ -> w_t τ_t as Δ -> 0
                weight = ((lambda t, w: w * t) if delta == 0.0
                          else (lambda t, w: -w * np.expm1(-t * delta) / delta))
                v, = self._laplace(Zs, grid, [weight])
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
