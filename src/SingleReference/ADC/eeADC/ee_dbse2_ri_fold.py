"""Eq 71's doubles as a tiled RI operator (level 'dbse2'): A_eff(ω) of one spin channel
on flat singles vectors, in RIFoldADC2's interface, from two sets of DF factors at the
GW energies, on the host or a GPU.

The kernel is ee_dbse2_pieces' (its module docstring: the vertices I, J, K, L on the
ordered doubles K = (k, l, c, d), D_K = ε_c + ε_d - ε_k - ε_l, the pair (x, y) of
coupling_pairs with X = Σ_A x_A A, Y = Σ_B y_B B, Θ = (X.Y + Y.X)/2, the copies
V± = (X ± Y)/2 on the metric η = diag(+1, -1)). Here the four spin blocks of the pieces
collapse onto the spin-adapted channel vector u, y^aa = u/√2, y^bb = ±u/√2 (singlet +,
triplet -): with X̂, Ŷ the vertex sums of the images of u and X̃, Ỹ the same with the J
and L coefficients negated,

    Θ(ω) u = c_hat sym[X̂ᵀ w Ŷ] u + c_tilde sym[X̃ᵀ w Ỹ] u,   w_K = 1 / (ω - D_K),
    sym[Aᵀ w B] u = (Aᵀ (w B u) + Bᵀ (w A u)) / 2,

singlet c_hat = 2, c_tilde = 0; triplet c_hat = c_tilde = 1. The signed norm
|Z+|² - |Z-|² is Σ_K Zx_K Zy_K with Zx = w X̂ u, Zy = w Ŷ u, the same factors; the
ω-derivative replaces w by -w². Eq 71's triplet gives X̂.Ŷ + X̃.Ỹ = 0; the printed
kernel's triplet keeps its loop terms.

Two factor sets: the bare factors B at the GW energies build the singles block M, eq 62
(the inner RIFoldADC2 at level 'adc1', CIS on those factors), with eq 59's w(ij|ab) in
place of (ij|ab) for singles='eq62+eq59'; the dressed factors B~ (the W lines of eq 73,
ee_dbse2_pieces.dressed_factors) feed the doubles. The inherited tiles over the occupied
index, auxiliary chunks and back-transformation halves of RIFoldADC2 run on the dressed
factors. Per tile the images I = H, J = Hᵀ of the factor ZI_P,kc = Σ_i B~_P,ik u_ic and
K = H, L = Hᵀ of ZK_P,kc = Σ_a u_ka B~_P,ac come from RIFoldADC2._pair's two GEMMs each,
the swap (k, c) <-> (l, d) being the pair's transpose. Every back-transformation is
row-local to the tile: with v^sw the swap of a doubles vector v and D swap-symmetric,
Aᵀ v = (A^sw)ᵀ v^sw, so Jᵀ R_J = Iᵀ R_J^sw and Lᵀ R_L = Kᵀ R_L^sw, where the swapped
R is the same combination of the four images with the coefficients of I and J (K and L)
exchanged. Θ u = Iᵀ R_I + Kᵀ R_K then closes through B~_oo (_back_oo) and B~_vv
(_back_vv) on the tile's rows, as RIFoldADC2's doubles do; the divide is exact. The
same tile core runs on numpy and on cupy (_tile_core takes the array module).

References
----------
E. Monino and P.-F. Loos, J. Chem. Phys. 159, 034105 (2023), eqs 59, 62, 71 to 74.
"""
import numpy as np

from src.Base.constants import EE_RI_TILE_GB
from src.Base.utils.mpi_grid import allgather_ranges, ordered_chain_sum, ordered_sum
from src.SingleReference.ADC.eeADC import ee_dbse2_pieces as _pieces
from src.SingleReference.ADC.eeADC.ee_ri_fold import DEVICES, RIFactors, RIFoldADC2

try:
    import cupy as cp
except ImportError:
    cp = None

# the J and L coefficients change sign in the triplet's second pair (X̃, Ỹ)
_FLIP = np.array([1.0, -1.0, 1.0, -1.0])
# the swap (k, c) <-> (l, d) exchanges I with J and K with L
_SWAP = (1, 0, 3, 2)


def _channel_factors(spin):
    """(c_hat, c_tilde) of the module docstring."""
    return (2.0, 0.0) if spin == 'singlet' else (1.0, 1.0)


def _denominator(xp, eo, ev, k0, k1):
    """D_kcld = ε_c + ε_d - ε_k - ε_l for the tile's rows k0..k1-1, (kt, nv, no, nv)."""
    return (ev[None, :, None, None] + ev[None, None, None, :]
            - eo[k0:k1, None, None, None] - eo[None, None, :, None])


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
    From Θ u = Σ_A Aᵀ R_A with R_A = w [c_hat (x_A Ŷ + y_A X̂) + c_tilde (x̃_A Ỹ +
    ỹ_A X̃)] / 2 and Jᵀ R_J = Iᵀ R_J^sw, Lᵀ R_L = Kᵀ R_L^sw: the swap of a combination
    of the images is the combination with I <-> J and K <-> L exchanged (_SWAP).
    """
    x, y = pair
    ch, ct = _channel_factors(spin)
    b = np.zeros((2, 4))
    for xx, yy, c in ((x, y, ch), (x * _FLIP, y * _FLIP, ct)):
        if c == 0.0:
            continue
        for row, a in enumerate((0, 2)):
            asw = _SWAP[a]
            for v in range(4):
                # Aᵀ [x_A Ŷ + y_A X̂]: image v with x_A y_v + y_A x_v
                b[row, v] += c * (xx[a] * yy[v] + yy[a] * xx[v])
                # (A^sw)ᵀ [...]^sw: image v of Ŷ, X̂ lands on image _SWAP[v]
                b[row, _SWAP[v]] += c * (xx[asw] * yy[v] + yy[asw] * xx[v])
    return 0.5 * b[0], 0.5 * b[1]


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


class RIFoldDBSE2(RIFoldADC2):
    """A_eff(ω) of one spin channel at level 'dbse2' on flat singles vectors u (no nv,),
    (i, a) row-major: the singles block M of eq 62 at the GW energies plus eq 71's
    doubles folded in, from two RIFactors at the GW energies.

    bare : RIFactors of the DF factors B, for M; dressed : RIFactors of the dressed
    factors B~ of ee_dbse2_pieces.dressed_factors, for the doubles (and eq 59); both
    carry the GW energies (eo, ev), else ValueError. kernel : 'eq71' or 'printed'
    (ee_dbse2_pieces.KERNELS); singles : 'eq62' or 'eq62+eq59' (its SINGLES). tile_gb,
    threads, comm, aux_chunk as RIFoldADC2; device 'cpu' or 'gpu'. The inherited
    fold_diagonal returns zeros (no second-order model at this level); pieces()
    hands the operator to ee_fold.solve_folded.
    """

    def __init__(self, bare, dressed, spin, kernel='eq71', singles='eq62',
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
        if device not in DEVICES:
            raise ValueError(f'device={device!r}; expected one of {DEVICES}')
        self.kernel, self.singles = kernel, singles
        self.pair = _pieces.coupling_pairs(kernel, 'singlet')[0]
        # M = eq 62 at the GW energies: CIS on the bare factors
        self.m_op = RIFoldADC2(bare, spin, level='adc1', tile_gb=tile_gb,
                               threads=threads, comm=comm, aux_chunk=aux_chunk)
        super().__init__(dressed, spin, level='adc1', tile_gb=tile_gb, threads=threads,
                         comm=comm, aux_chunk=aux_chunk)
        self.level = 'dbse2'
        # L2[P, (l, d)] = B~_P,ld, the dressed factor the images close through
        self.L2 = self.LT.reshape(self.f.naux, -1)
        # ten tile-sized arrays live in one tile: the four images, their four
        # combinations, the denominator and a divided image
        per_k = 8.0 * self.nv * self.no * self.nv * 10
        self.kt = int(max(1, min(-(-self.no // self.size),
                                 tile_gb * 1024 ** 3 // per_k)))
        self.tiles = [(k0, min(k0 + self.kt, self.no))
                      for k0 in range(0, self.no, self.kt)]
        self.my_tiles = [(t, self.tiles[t])
                         for t in range(self.rank, len(self.tiles), self.size)]
        if device == 'gpu':
            raise NotImplementedError('the device path of RIFoldDBSE2 is not written')
        self.gpu = None

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

    def _doubles(self, u, nodes, derivative):
        """(F (p, n), nrm (p,)): Iᵀ R_I + Kᵀ R_K of one channel vector at the nodes, the
        folded doubles (or their ω-derivative), and the signed norm at each node. The
        tiles' Iᵀ halves and norms are summed in tile order, the Kᵀ halves by the
        auxiliary chunks' owners in chunk order; no collective inside the tile loop."""
        n, p = self.n, nodes.size
        ZI, ZK = self._vertex_factors(u)
        parts, gams_K = [], {}
        for t, (k0, k1) in self.my_tiles:
            gI, gK, nrm = _tile_core(np, self.L2, self.f.eo, self.f.ev, k0, k1, ZI, ZK,
                                     self.pair, self.spin, nodes, derivative)
            parts.append((t, np.concatenate([self._back_oo(gI, k0, k1).ravel(), nrm])))
            gams_K[t] = gK
        tot = ordered_sum(parts, self.comm, shape=(p * n + p,),
                          onto=np.zeros(p * n + p))
        # _back_vv carries ADC(2)'s minus sign; Kᵀ R enters with plus
        F = tot[:p * n].reshape(p, n) - self._back_vv(gams_K, p)
        return F, tot[p * n:]

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
        self.image_columns += m
        self.node_images += m * p
        F = np.empty((p, n, m))
        for j in range(m):
            F[:, :, j] = self._doubles(U[:, j], nodes, derivative)[0]
        return MU, F

    def doubles_dot(self, ur, wr, us, ws):
        """Σ_K Y_K(ur, wr)ᵀ η Y_K(us, ws), the signed overlap of two doubles images."""
        ur, us = np.ravel(ur), np.ravel(us)
        same = ur is us or np.array_equal(ur, us)
        Zr = self._vertex_factors(ur)
        Zs = Zr if same else self._vertex_factors(us)
        parts = []
        for t, (k0, k1) in self.my_tiles:
            v = _tile_overlap(np, self.L2, self.f.eo, self.f.ev, k0, k1, Zr, float(wr),
                              Zs, float(ws), self.pair, self.spin)
            parts.append((t, np.array([v])))
        return float(ordered_sum(parts, self.comm, shape=(1,), onto=np.zeros(1))[0])

    def pieces(self):
        """The dict ee_fold.solve_folded takes: the operator stands in for M, V, Vt, D
        and dnorm2 (pieces['operator']); d_min = min D at the GW energies."""
        return {'operator': self, 'level': self.level, 'no': self.no, 'nv': self.nv,
                'd_min': self.d_min, 'spin': self.spin, 'dense_limit': 0}
