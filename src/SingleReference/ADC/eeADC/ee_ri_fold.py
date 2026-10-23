"""Density-fitted ADC(2) singles operator with the doubles folded in, for large
molecules: the doubles exist one occupied tile at a time; only the MP1
amplitudes of M's second-order term are kept, per rank and up to
EE_RI_AMPLITUDE_CACHE_GB, the tiles past it rebuilt in every product.

The folded problem of ee_fold, A_eff(w) y = w y with

    A_eff(w) = M - C (D - w)^-1 C^T ,

needs three things per singles vector: the singles block M (zeroth, first and
second order), the doubles image of the vector and its weight in the full
eigenvector. ee_r_sigma_df builds M as an (nocc nvir)^2 array and the
coupling from <ov||vv> blocks; at nocc 120, nvir 900 either is about 100 GB.
Here every doubles-shaped quantity exists only one occupied tile at a time,
made from the DF factor B^P_pq in the RI-CC2 way (Haettig and Weigend, J.
Chem. Phys. 113, 5154 (2000); Haettig, Adv. Quantum Chem. 50, 37 (2005)):

    doubles of a singles vector u:
        Bt^P_kc = sum_i B^P_ki u_ic - sum_a u_ka B^P_ac
        H_kc,ld = sum_P Bt^P_kc B^P_ld
    singlet:  G = H + H^T,  R = w (2 G - G~),  |Y|^2 = sum w^2 G (G - G~/2)
    triplet:  S = H + H^T,  R = w (2 H - S~),
              |Y|^2 = sum w^2 [ (H - H^T)^2 / 2 + (S - S~)^2 / 4 ]
    with w_kcld = 1 / (omega - D_kcld), X~_kcld = X_kdlc, and the folded image
        (A2 u)_jb = sum_P [ sum_k B^P_kj G^P_kb - sum_c G^P_jc B^P_cb ],
        G^P_kc = sum_ld R_kcld B^P_ld .

These are the spin-orbital ADC(2) working equations of ee_equations (first
order coupling, A54 / A41) projected on one spin channel of a closed shell:
the alpha-beta doubles block is H + H^T (singlet) or H - H^T (triplet) over
sqrt 2, the same-spin blocks its antisymmetrization, and the doubles norm the
one of the flat adc2 layout. A2 u is half the gradient of the quadratic form
sum_K w_K Y_K^2, which is how R follows from the norm.

The second-order part of M (A33) needs the MP1 amplitudes, made the same way
per tile from I_kc,ld = (kc|ld):

    M2 = (x_oo + x_oo^T) (x) 1 + 1 (x) (x_vv + x_vv^T) - (K T + T K) / 2
    singlet  K = 2 (ia|kc) - (ic|ka),   T = [2 (ia|kc) - (ic|ka)] / D
    triplet  K =           - (ic|ka),   T =           - (ic|ka)  / D

with x_oo, x_vv the occupied/virtual MP2 density-like contractions.

Spin-component scaling (SCS/SOS-ADC(2), Hellweg, Gruen and Haettig, Phys.
Chem. Chem. Phys. 10, 4119 (2008)) weights every doubles configuration by its
spin class, c_os for an opposite-spin pair and c_ss for a same-spin pair, in
both places doubles enter: the MP1 amplitudes of M2 and the folded coupling.
With tau = (kc|ld) / D the opposite-spin and t_ss = tau - tau~ the same-spin
amplitude of a closed shell,

    singlet  T = c_os tau + c_ss (tau - tau~)      (x_oo, x_vv, K T + T K)
    triplet  T = c_ss (tau - tau~) - c_os tau      (K T + T K; x_oo, x_vv as
                                                   the singlet's)
    singlet  R = w [c_os G + c_ss (G - G~)],
             |Y|^2 = sum w^2 [c_os G^2 / 2 + c_ss (G - G~)^2 / 4]
    triplet  R = w [c_os (H - H^T) + c_ss (S - S~)],
             |Y|^2 = sum w^2 [c_os (H - H^T)^2 / 2 + c_ss (S - S~)^2 / 4]

c_os = c_ss = 1 is ADC(2). Scaling the coupling on one side, as SCS-CC2's
Jacobian does, has the eigenvalues of the symmetric problem with the couplings
scaled by sqrt(c), whose doubles weight is the scaled |Y|^2 above.

Per
matrix-vector product the cost is four GEMMs of shape
(n_tile nvir) x naux x (nocc nvir) per tile, O(nocc^2 nvir^2 naux) in all, and
the memory a few (n_tile nvir, nocc nvir) blocks plus the factors.

tests/test_ee_adc_ri_fold.py pins this operator against ee_fold.folded_operator
on ee_r_sigma_df's pieces (same factors) to rounding: matvec, doubles norm and
diagonal, singlets and triplets, at several frequencies.

The environment
---------------
A polarizable continuum enters the way linear-response PCM enters CIS/TDHF:
the ground state carries the slow field (the SCF relaxed at eps_static, so the
orbitals and orbital energies do), and the fast field of the TRANSITION
density, eps_optical, adds 2 (ia|vtilde|jb) to the singlet singles block (a
triplet's transition density has no charge, so nothing). Every
second-order term -- M2, the coupling, the doubles -- is built with the bare
interaction on the solvated orbitals: correlation inside a frozen reaction
field, the PTE convention of PCM-ADC (Lunkenheimer and Koehn, J. Chem. Theory
Comput. 9, 977 (2013); Mewes et al., J. Phys. Chem. A 119, 5446 (2015)).
At ADC(1) this is pyscf's TDA in PCM with a non-equilibrium (optical)
response. The other half of the response
to the excitation, the reaction field of the state's difference density, is
added after the solve when asked for (ee_ri_density, ee_state_specific).
"""
import numpy as np
from pyscf import df as pyscf_df
from pyscf import lib

from src.Base.numerics.constants import (BYTES_PER_GB, EE_RI_AMPLITUDE_CACHE_GB, EE_RI_AUX_CHUNK,
                                         EE_RI_EXCHANGE_ARRAYS, EE_RI_EXCHANGE_CHUNK,
                                         EE_RI_FOLD_DIAG_EXPONENT, EE_RI_FOLD_DIAG_GAP,
                                         EE_RI_FOLD_DIAG_HOLE, EE_RI_FOLD_DIAG_PARTICLE,
                                         EE_RI_FOLD_DIAG_STEP, EE_RI_FOLD_DIAG_TAIL,
                                         EE_RI_FOLD_TILE_ARRAYS, EE_RI_TILE_GB)
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.memory import tile_width
from src.Base.utils.parallel.mpi_grid import (NodeShared, allgather_ranges, current_comm,
                                              exchange_blocks, ordered_chain_sum, ordered_sum,
                                              reduce_sum)
from src.Base.utils.parallel.threads import openmp_threads, row_map

try:
    import cupy as cp
except ImportError:              # optional: only device='gpu' needs it
    cp = None

SPINS = ('singlet', 'triplet')
DEVICES = ('cpu', 'gpu')
LEVELS = ('adc1', 'adc2')


# ----------------------------------------------------------------------
# factors
# ----------------------------------------------------------------------

class RIFactors:
    """The DF factor B^P_pq = sum_mu nu C_mu p C_nu q (P|mu nu)~ of one closed-
    shell reference, in the layouts the operator contracts:

    eo, ev : (no,), (nv,) active orbital energies, Hartree
    LT     : (naux, no, nv), B^P_ia, auxiliary-major (the stored form)
    L      : (no nv, naux), rows (i, a): B^P_ia, a view of LT
    BooP   : (naux, no, no), B^P_ij as [P, i, j] (the stored form)
    Boo    : (no, no, naux), B^P_ij, a view of BooP
    Bvv    : (nv, naux, nv), B^P_ab as [a, P, b], whole; or None, and then
    bvv_chunks : {c: (nv, p1 - p0, nv)}, only the auxiliary chunks
             [p0, p1) = aux_chunks(...)[0][c] this rank owns: v^2 naux is the
             largest factor, and over MPI ranks it is held once, split
    LT and BooP, read whole by every rank, may live in node-shared windows
    (`shared`, the NodeShared objects that hold them), one copy per node.
    """

    def __init__(self, eo, ev, L, Boo, Bvv=None, bvv_chunks=None, LT=None,
                 BooP=None, shared=()):
        self.eo, self.ev = np.asarray(eo, float), np.asarray(ev, float)
        self.no, self.nv = self.eo.size, self.ev.size
        self.LT = (LT if LT is not None else np.ascontiguousarray(
            np.asarray(L).reshape(self.no, self.nv, -1).transpose(2, 0, 1)))
        self.naux = self.LT.shape[0]
        self.BooP = (BooP if BooP is not None else
                     np.ascontiguousarray(np.asarray(Boo).transpose(2, 0, 1)))
        self.Bvv = None if Bvv is None else np.ascontiguousarray(Bvv)
        self.bvv_chunks = bvv_chunks
        self.shared = tuple(shared)

    @property
    def L(self):
        return self.LT.reshape(self.naux, -1).T

    @property
    def Boo(self):
        return self.BooP.transpose(1, 2, 0)

    def bvv(self, c, p0, p1):
        """B^P_ab of the auxiliary chunk c = [p0, p1) as [a, P, b], contiguous."""
        if self.Bvv is not None:
            return np.ascontiguousarray(self.Bvv[:, p0:p1])
        return self.bvv_chunks[c]

    @classmethod
    def from_full(cls, eps, B, no):
        """From an MO factor B (naux, norb, norb) and the active orbital
        energies; for the tests and small molecules."""
        o, v = slice(0, no), slice(no, len(eps))
        nv = len(eps) - no
        L = B[:, o, v].reshape(-1, no * nv).T
        return cls(eps[:no], eps[no:], L, B[:, o, o].transpose(1, 2, 0),
                   B[:, v, v].transpose(1, 0, 2))

    def nbytes(self):
        """Bytes of this rank's factors, a node-shared array counted whole."""
        vv = (self.Bvv.nbytes if self.Bvv is not None else
              sum(a.nbytes for a in self.bvv_chunks.values()))
        return self.LT.nbytes + self.BooP.nbytes + vv


def aux_chunks(naux, nv, size, width=None):
    """(p0, p1) chunks of the auxiliary index, fixed by naux and nv alone (or
    an explicit width), and the rank owning each: consecutive runs, so that a
    sum over chunks in chunk order (ordered_chain_sum) has the one-rank bits
    at every rank count. That covers the B_vv sums; the occupied tiles keep
    their bits across rank counts only at one tile size (RIFoldADC2's kt)."""
    step = width or tile_width(8.0 * EE_RI_EXCHANGE_ARRAYS * nv * nv,
                               EE_RI_EXCHANGE_CHUNK,
                               Numerics(layout={'tile_gb': EE_RI_TILE_GB}))
    chunks = [(p0, min(p0 + step, naux)) for p0 in range(0, naux, step)]
    return chunks, [c * size // len(chunks) for c in range(len(chunks))]


def bare_factors(mol, mf, frozen=0, auxbasis=None, chunk=EE_RI_AUX_CHUNK, comm=None,
                 aux_chunk=None):
    """RIFactors of the BARE Coulomb interaction on the orbitals of `mf`, the
    `frozen` lowest occupied dropped.

    auxbasis None takes the mean field's own DF object (the JK fit of a DF
    SCF); a name builds a separate one (the correlation fit, e.g.
    'aug-cc-pvdz-ri'). An environment attached to `mf` is deliberately not
    applied: the continuum enters this route through the singles block alone
    (module docstring), never through the factor. Over the ranks of comm
    B_vv is kept in this rank's auxiliary chunks (aux_chunks, width aux_chunk;
    the operator must be given the same), all of them on a single rank.
    """
    if auxbasis is None:
        with_df = getattr(mf, 'with_df', None)
        if with_df is None:
            raise ValueError('the folded route is density-fitted: give an '
                             'auxbasis, or a DF mean field')
    else:
        with_df = pyscf_df.DF(mol, auxbasis=auxbasis)
        with_df.max_memory = mf.max_memory
        with_df.build()
    nocc = mol.nelectron // 2
    C = np.asarray(mf.mo_coeff)
    eps = np.asarray(mf.mo_energy, float)
    Co = np.ascontiguousarray(C[:, frozen:nocc])
    Cv = np.ascontiguousarray(C[:, nocc:])
    no, nv = Co.shape[1], Cv.shape[1]
    naux = with_df.get_naoaux()
    # L and B_oo auxiliary-major, once per node, written by its first rank
    sh_l, sh_o = NodeShared((naux, no, nv), comm), NodeShared((naux, no, no), comm)
    LT, BooP = sh_l.array, sh_o.array
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    # B_vv in its auxiliary chunks on one rank too: a chunk sliced out of a
    # whole [a, P, b] array is a copy of v^2 per auxiliary row at every use
    chunks, owners = aux_chunks(naux, nv, size, aux_chunk)
    mine = {c: (chunks[c], np.empty((nv, chunks[c][1] - chunks[c][0], nv)))
            for c in range(len(chunks)) if owners[c] == rank}
    p0 = 0
    for blk in with_df.loop(blksize=chunk):
        p1 = p0 + blk.shape[0]
        ao = lib.unpack_tril(blk)                       # (p, nao, nao)
        nao = ao.shape[1]
        if sh_l.fills:
            half_o = (ao.reshape(-1, nao) @ Co).reshape(-1, nao, no)  # (p, mu, i)
            half_o = half_o.transpose(0, 2, 1).reshape(-1, nao)       # (p i, mu)
            LT[p0:p1] = (half_o @ Cv).reshape(-1, no, nv)
            BooP[p0:p1] = (half_o @ Co).reshape(-1, no, no)
            del half_o
        # the rows of this block that B_vv keeps here
        keep = [(c, max(p0, q0), min(p1, q1)) for c, ((q0, q1), _) in mine.items()
                if max(p0, q0) < min(p1, q1)]
        if keep:
            half_v = (ao.reshape(-1, nao) @ Cv).reshape(-1, nao, nv)
            half_v = half_v.transpose(0, 2, 1).reshape(-1, nao)    # (p a, mu)
            vv = (half_v @ Cv).reshape(-1, nv, nv).transpose(1, 0, 2)
            for c, lo, hi in keep:
                q0 = mine[c][0][0]
                mine[c][1][:, lo - q0:hi - q0, :] = vv[:, lo - p0:hi - p0, :]
            del half_v, vv
        del ao
        p0 = p1
    if p0 != naux:
        raise RuntimeError(f'the DF loop returned {p0} of {naux} auxiliary rows')
    sh_l.ready()
    sh_o.ready()
    return RIFactors(eps[frozen:nocc], eps[nocc:], None, None, None,
                     {c: a for c, (_, a) in mine.items()},
                     LT=LT, BooP=BooP, shared=(sh_l, sh_o))


# ----------------------------------------------------------------------
# the continuum's singles term
# ----------------------------------------------------------------------

class TransitionReactionField:
    """(ia|vtilde|jb) of a surface-charge continuum as V Q V^T: V (no nv, ng)
    the surface potential of the orbital pairs, Q (ng, ng) the symmetrized
    surface response at the optical dielectric constant (negative
    semi-definite). `apply(X)` maps (no nv, m) to (no nv, m)."""

    def __init__(self, V, Q):
        self.V = np.ascontiguousarray(V)
        self.Q = np.ascontiguousarray(Q)

    @classmethod
    def from_environment(cls, env, mol, mf, frozen=0):
        nocc = mol.nelectron // 2
        C = np.asarray(mf.mo_coeff)
        V = env.ov_grid_potential(mol, C[:, frozen:nocc], C[:, nocc:])
        return cls(V.reshape(-1, V.shape[-1]), env.response_matrix())

    def apply(self, X):
        return self.V @ (self.Q @ (self.V.T @ X))

    def diagonal(self):
        return np.einsum('pk,pk->p', self.V @ self.Q, self.V, optimize=True)


# ----------------------------------------------------------------------
# the operator
# ----------------------------------------------------------------------

def _segments(r0, r1, nv):
    """(k, c0, c1) runs of the flattened (k, c) rows r0..r1-1 of a tile."""
    out, r = [], r0
    while r < r1:
        k, c0 = divmod(r, nv)
        c1 = min(nv, c0 + (r1 - r))
        out.append((k, c0, c1))
        r += c1 - c0
    return out


class _DoublesGPU:
    """The doubles kernel of an RIFoldADC2 on this rank's GPU (CuPy, FP64): the
    factor L auxiliary-major and the orbital energies resident; per tile the
    pair GEMMs H = Bt L^T and H^T = L Bt^T, one fused pass forming
    Gt = 2 S - S~ (singlet, S = H + H^T) or 2 H - S~ (triplet), spin-component
    scaled where the operator is (module docstring), one fused pass
    scaling it by (w - D)^-1 with D_kcld = e_c + e_d - e_k - e_l formed on
    the fly, the Gam GEMM and the B_oo half of the back-transformation. Gam
    goes back to the host, where the B_vv half is formed by the chunks' owners.
    For doubles_dot, one fused pass forms the overlap summand of two columns'
    images and only its sum goes back.
    The arithmetic is the CPU path's; the sums run in another order, so the two
    agree to rounding, not to the bit."""

    _GT = r"""
        long long q = i; const long long d = q % nv; q /= nv;
        const long long l = q % no; q /= no; const long long c = q % nv;
        const long long k = q / nv;
        const long long x = ((k * nv + d) * no + l) * nv + c;   // (k, d, l, c)
        const double sx = H[x] + HT[x];
        if (!scaled) { G = singlet ? 2.0 * (H[i] + HT[i]) - sx : 2.0 * H[i] - sx; }
        else if (singlet) { G = (c_os + c_ss) * (H[i] + HT[i]) - c_ss * sx; }
        else { G = c_os * (H[i] - HT[i]) + c_ss * (H[i] + HT[i] - sx); }
    """
    _R = r"""
        long long q = i; const long long d = q % nv; q /= nv;
        const long long l = q % no; q /= no; const long long c = q % nv;
        const long long k = q / nv;
        const double x = w - (ev[c] + ev[d] - eo[k0 + k] - eo[l]);
        R = G[i] / x;
    """
    # the ω-derivative of R: d/dω G / (ω - D) = -G / (ω - D)^2
    _R2 = r"""
        long long q = i; const long long d = q % nv; q /= nv;
        const long long l = q % no; q /= no; const long long c = q % nv;
        const long long k = q / nv;
        const double x = w - (ev[c] + ev[d] - eo[k0 + k] - eo[l]);
        R = -G[i] / (x * x);
    """
    # |Y|^2 summand of the image at w: singlet w^2 S (S - S~/2), triplet
    # w^2 [(H - H^T)^2 / 2 + (S - S~)^2 / 4], S = H + H^T
    _N = r"""
        long long q = i; const long long d = q % nv; q /= nv;
        const long long l = q % no; q /= no; const long long c = q % nv;
        const long long k = q / nv;
        const long long x = ((k * nv + d) * no + l) * nv + c;
        const double y = 1.0 / (w - (ev[c] + ev[d] - eo[k0 + k] - eo[l]));
        const double s = H[i] + HT[i], sx = H[x] + HT[x];
        if (scaled && singlet) {
            N = y * y * (0.5 * c_os * s * s + 0.25 * c_ss * (s - sx) * (s - sx)); }
        else if (scaled) { const double a = H[i] - HT[i], b = s - sx;
            N = y * y * (0.5 * c_os * a * a + 0.25 * c_ss * b * b); }
        else if (singlet) { N = y * y * s * (s - 0.5 * sx); }
        else { const double a = H[i] - HT[i], b = s - sx;
               N = y * y * (0.5 * a * a + 0.25 * b * b); }
    """
    # overlap summand of the images of columns r at wr and s at ws,
    # y = 1 / ((wr - D)(ws - D)): singlet y S_r (S_s - S~_s/2), triplet
    # y [(H_r - H^T_r)(H_s - H^T_s) / 2 + (S_r - S~_r)(S_s - S~_s) / 4];
    # scaled, _N's weights: singlet y [c_os S_r S_s / 2
    # + c_ss (S_r - S~_r)(S_s - S~_s) / 4], triplet c_os and c_ss on the two terms
    _O = r"""
        long long q = i; const long long d = q % nv; q /= nv;
        const long long l = q % no; q /= no; const long long c = q % nv;
        const long long k = q / nv;
        const long long x = ((k * nv + d) * no + l) * nv + c;
        const double D = ev[c] + ev[d] - eo[k0 + k] - eo[l];
        const double y = 1.0 / ((wr - D) * (ws - D));
        const double sr = Hr[i] + HTr[i], ss = Hs[i] + HTs[i];
        const double sxs = Hs[x] + HTs[x];
        if (scaled && singlet) { const double sxr = Hr[x] + HTr[x];
            O = y * (0.5 * c_os * sr * ss + 0.25 * c_ss * (sr - sxr) * (ss - sxs)); }
        else if (scaled) { const double sxr = Hr[x] + HTr[x];
            O = y * (0.5 * c_os * (Hr[i] - HTr[i]) * (Hs[i] - HTs[i])
                     + 0.25 * c_ss * (sr - sxr) * (ss - sxs)); }
        else if (singlet) { O = y * sr * (ss - 0.5 * sxs); }
        else { const double sxr = Hr[x] + HTr[x];
               O = y * (0.5 * (Hr[i] - HTr[i]) * (Hs[i] - HTs[i])
                        + 0.25 * (sr - sxr) * (ss - sxs)); }
    """

    def __init__(self, op):
        if cp is None:
            raise ImportError("device='gpu' needs CuPy, which is not importable")
        self.op = op
        f = op.f
        self.no, self.nv, self.naux = op.no, op.nv, f.naux
        self.L2 = cp.asarray(f.LT.reshape(f.naux, -1))              # (naux, no nv)
        self.Boo = cp.asarray(np.ascontiguousarray(f.Boo))          # (no, no, naux)
        self.eo, self.ev = cp.asarray(f.eo), cp.asarray(f.ev)
        self.singlet = op.spin == 'singlet'
        # spin-component scaling: (scaled, c_os, c_ss); unscaled keeps ADC(2)'s
        # own arithmetic
        self.scs = ((False, 1.0, 1.0) if op.scs is None else (True,) + op.scs)
        self.k_gt = cp.ElementwiseKernel(
            'raw float64 H, raw float64 HT, int64 nv, int64 no, bool singlet, '
            'bool scaled, float64 c_os, float64 c_ss',
            'float64 G', self._GT, 'ee_fold_gt')
        self.k_r = cp.ElementwiseKernel(
            'raw float64 G, raw float64 eo, raw float64 ev, int64 k0, int64 nv, '
            'int64 no, float64 w', 'float64 R', self._R, 'ee_fold_r')
        self.k_r2 = cp.ElementwiseKernel(
            'raw float64 G, raw float64 eo, raw float64 ev, int64 k0, int64 nv, '
            'int64 no, float64 w', 'float64 R', self._R2, 'ee_fold_r2')
        self.k_n = cp.ElementwiseKernel(
            'raw float64 H, raw float64 HT, raw float64 eo, raw float64 ev, '
            'int64 k0, int64 nv, int64 no, float64 w, bool singlet, '
            'bool scaled, float64 c_os, float64 c_ss', 'float64 N',
            self._N, 'ee_fold_n')
        self.k_o = cp.ElementwiseKernel(
            'raw float64 Hr, raw float64 HTr, raw float64 Hs, raw float64 HTs, '
            'raw float64 eo, raw float64 ev, int64 k0, int64 nv, int64 no, '
            'float64 wr, float64 ws, bool singlet, bool scaled, float64 c_os, '
            'float64 c_ss', 'float64 O', self._O, 'ee_fold_o')

    def upload(self, BtT):
        """The dressed factor of one column, (naux, no nv), on the device."""
        return cp.asarray(BtT)

    def _pair(self, Btg, k0, k1):
        nv = self.nv
        rows = slice(k0 * nv, k1 * nv)
        H = Btg[:, rows].T @ self.L2
        HT = self.L2[:, rows].T @ Btg
        return H, HT

    def _back_oo(self, Gam, k0, k1):
        """sum_kP B_kjP Gam_kbP for Gam (kt nv, naux) on the device, (no nv,)."""
        kt = k1 - k0
        GkPb = Gam.reshape(kt, self.nv, -1).transpose(0, 2, 1).reshape(-1, self.nv)
        Bjk = self.Boo[k0:k1].transpose(1, 0, 2).reshape(self.no, -1)
        return (Bjk @ GkPb).ravel()

    def images_tile(self, k0, k1, Btg, nodes, derivative):
        """The CPU path's _images_tile: (Gam (p, kt nv, naux), B_oo half (p, no nv))
        as host arrays."""
        nv, no = self.nv, self.no
        H, HT = self._pair(Btg, k0, k1)
        G = cp.empty_like(H)
        self.k_gt(H, HT, nv, no, self.singlet, *self.scs, G)
        del H, HT
        R = cp.empty_like(G)
        gams = np.empty((len(nodes), (k1 - k0) * nv, self.naux))
        img = np.empty((len(nodes), no * nv))
        for q, w in enumerate(nodes):
            kern = self.k_r2 if derivative[q] else self.k_r
            kern(G, self.eo, self.ev, k0, nv, no, float(w), R)
            Gam = R @ self.L2.T
            img[q] = cp.asnumpy(self._back_oo(Gam, k0, k1))
            gams[q] = cp.asnumpy(Gam)
        return gams, img

    def doubles_tile(self, k0, k1, Btg, omega):
        """The CPU path's _doubles_tile: (Gam, B_oo half, |Y|^2) on the host."""
        nv, no = self.nv, self.no
        H, HT = self._pair(Btg, k0, k1)
        N = cp.empty_like(H)
        self.k_n(H, HT, self.eo, self.ev, k0, nv, no, float(omega), self.singlet,
                 *self.scs, N)
        nrm = float(N.sum())
        G = N                                                       # reused
        self.k_gt(H, HT, nv, no, self.singlet, *self.scs, G)
        del H, HT
        R = cp.empty_like(G)
        self.k_r(G, self.eo, self.ev, k0, nv, no, float(omega), R)
        Gam = R @ self.L2.T
        return cp.asnumpy(Gam), cp.asnumpy(self._back_oo(Gam, k0, k1)), nrm

    def overlap(self, k0, k1, Bgr, wr, Bgs, ws):
        """One tile's share of doubles_dot for the dressed factors Bgr at wr and
        Bgs at ws on the device, a float."""
        nv, no = self.nv, self.no
        Hr, HTr = self._pair(Bgr, k0, k1)
        Hs, HTs = (Hr, HTr) if Bgs is Bgr else self._pair(Bgs, k0, k1)
        O = cp.empty_like(Hr)
        self.k_o(Hr, HTr, Hs, HTs, self.eo, self.ev, k0, nv, no, float(wr),
                 float(ws), self.singlet, *self.scs, O)
        return float(O.sum())


class SpinScaling:
    """The (c_os, c_ss) weights of the doubles of spin-component scaled ADC(2),
    (1, 1) for ADC(2) itself. G is a pair tensor (k, c, l, d) and Gx its
    exchange partner G_kdlc; on the triplet A = H - H^T and S = H + H^T - S~
    of the doubles' pair tensor H."""

    def __init__(self, scaling):
        self.cos, self.css = (1.0, 1.0) if scaling is None else scaling

    def singlet(self, G, Gx):
        """(c_os + c_ss) G - c_ss Gx: the singlet amplitude's numerator."""
        return (self.cos + self.css) * G - self.css * Gx

    def triplet(self, G, Gx):
        """(c_ss - c_os) G - c_ss Gx: the triplet amplitude's numerator."""
        return (self.css - self.cos) * G - self.css * Gx

    def triplet_image(self, A, S):
        """c_os A + c_ss S: the triplet folded image's numerator."""
        return self.cos * A + self.css * S

    def singlet_overlap(self, gr, xr, gs, xs):
        """The singlet summand of sum_K Y_K Y'_K over w w'."""
        return 0.5 * self.cos * gr * gs + 0.25 * self.css * (gr - xr) * (gs - xs)

    def triplet_overlap(self, ar, sr, as_, ss):
        """The triplet summand of sum_K Y_K Y'_K over w w'."""
        return 0.5 * self.cos * ar * as_ + 0.25 * self.css * sr * ss


class RIFoldADC2:
    """A_eff(w) of one spin channel of closed-shell ADC(1)/ADC(2) on flat
    singles vectors u (no nv,), (i, a) row-major, from RIFactors.

    level 'adc1' is CIS (no second order, no doubles); 'adc2' ADC(2).
    reaction_field: a TransitionReactionField or None. tile_gb bounds the
    tile working set; threads the element-wise passes (default: the
    process's OpenMP count); aux_chunk the width of the auxiliary chunks
    (default: aux_chunks' rule), which split factors were built with; device
    'gpu' runs the doubles kernel on this rank's GPU (_DoublesGPU), 'cpu' on
    the host. scaling: None (ADC(2)) or the spin-component factors
    (c_os, c_ss) of SCS/SOS-ADC(2) (module docstring).

    Over MPI ranks (comm, default the current one) the occupied tiles are
    dealt round-robin and every tile's contribution is summed in tile order
    (ordered_sum), so at one tile size a product has the same bits at every
    rank count. Each rank holds L and B_oo whole and B_vv in its auxiliary
    chunks (aux_chunks); the dressed factor is gathered whole and the B_vv
    term of the back-transformation is formed by the chunks' owners and summed
    in chunk order. Construction, `apply`, `fold_images` and `doubles_dot` are
    then collectives: every rank calls them with the same arguments
    (ee_driver drives them from rank 0 with root_driven_solve)."""

    def __init__(self, factors, spin, level='adc2', reaction_field=None,
                 tile_gb=EE_RI_TILE_GB, threads=None, comm=None, aux_chunk=None,
                 device='cpu', scaling=None):
        if spin not in SPINS:
            raise ValueError(f'spin={spin!r}; the folded route solves one '
                             f'channel, one of {SPINS}')
        if level not in LEVELS:
            raise ValueError(f'level={level!r}; expected one of {LEVELS}')
        if device not in DEVICES:
            raise ValueError(f'device={device!r}; expected one of {DEVICES}')
        if scaling is not None and (len(scaling) != 2 or min(scaling) < 0):
            raise ValueError(f'scaling={scaling!r}; expected (c_os, c_ss) >= 0')
        self.f, self.spin, self.level = factors, spin, level
        # (c_os, c_ss), or None for ADC(2); the CPU arithmetic is weights'
        self.scs = None if scaling is None else tuple(float(c) for c in scaling)
        self.weights = SpinScaling(self.scs)
        self.rf = reaction_field
        self.threads = int(threads or openmp_threads())
        self.tile_gb = tile_gb
        no, nv = factors.no, factors.nv
        self.no, self.nv, self.n = no, nv, no * nv
        # LT[P, i, a] = B^P_ia, the factor's own auxiliary-major storage
        self.LT = factors.LT
        self.delta = (factors.ev[None, :] - factors.eo[:, None]).ravel()
        self.d_min = 2.0 * (factors.ev.min() - factors.eo.max())
        self.e_mp2 = None
        # columns of M (every product), trial vectors given fold images, and
        # their images summed over frequencies
        self.matvecs = 0
        self.image_columns = 0
        self.node_images = 0
        self.comm = current_comm() if comm is None else comm
        rank, size = ((0, 1) if self.comm is None else
                      (self.comm.Get_rank(), self.comm.Get_size()))
        # occupied orbitals per tile: what tile_gb allows, and no more than
        # gives every rank a tile
        per_k = 8.0 * nv * no * nv * EE_RI_FOLD_TILE_ARRAYS
        self.kt = tile_width(per_k, -(-no // size), Numerics(layout={'tile_gb': tile_gb}))
        self.tiles = [(k0, min(k0 + self.kt, no)) for k0 in range(0, no, self.kt)]
        # (tile index, (k0, k1)) of this rank's tiles
        self.my_tiles = [(t, self.tiles[t])
                         for t in range(rank, len(self.tiles), size)]
        self.size, self.rank = size, rank
        # the auxiliary chunks, their owners, this rank's, and every rank's
        # row ranges of an auxiliary-major array
        self.chunks, self.owners = aux_chunks(factors.naux, nv, size, aux_chunk)
        self.my_chunks = [c for c, o in enumerate(self.owners) if o == rank]
        if factors.bvv_chunks is not None and (
                sorted(factors.bvv_chunks) != self.my_chunks or any(
                    factors.bvv_chunks[c].shape[1] != self.chunks[c][1] - self.chunks[c][0]
                    for c in self.my_chunks)):
            raise ValueError('the factors hold other B_vv chunks than this operator '
                             'reads: build both with the same comm and aux_chunk')
        self.chunk_ranges = [[self.chunks[c] for c, o in enumerate(self.owners)
                              if o == r] for r in range(size)]
        self._setup()
        self.gpu = _DoublesGPU(self) if device == 'gpu' and level == 'adc2' else None

    # ---- element-wise passes over one tile --------------------------------

    def _rows(self, kt, fn):
        """fn(k, c0, c1) over every row segment of a tile, threaded; the
        partial results in row order."""
        nv = self.nv

        def run(r0, r1):
            return [fn(k, c0, c1) for k, c0, c1 in _segments(r0, r1, nv)]

        parts = row_map(run, kt * nv, self.threads)
        return [x for p in parts for x in p]

    def _denom(self, kg, c0, c1):
        """D[c, l, d] = e_c + e_d - e_k - e_l for the rows (kg, c0..c1-1)."""
        eo, ev = self.f.eo, self.f.ev
        return (ev[c0:c1, None, None] + ev[None, None, :]
                - eo[kg] - eo[None, :, None])

    def _ints(self, k0, k1):
        """I[k, c, l, d] = (kc|ld) for k in the tile, shape (kt, nv, no, nv)."""
        L, nv = self.f.L, self.nv
        I = L[k0 * nv:k1 * nv] @ L.T
        return I.reshape(k1 - k0, nv, self.no, nv)

    # ---- one-time pieces --------------------------------------------------

    def _setup(self):
        """The diagonal of the singles block, and at adc2 the MP1 amplitude pass:
        e_mp2, the x_oo and x_vv intermediates and the second-order diagonal."""
        no, nv = self.no, self.nv
        diag = self._first_order_diagonal()
        self.x_oo = np.zeros((no, no))
        self.x_vv = np.zeros((nv, nv))
        if self.level == 'adc1':
            self.diag = diag
            return
        tot = self._amplitude_pass()
        self.e_mp2 = float(tot[0])
        xdiag = tot[1:1 + no * nv].reshape(no, nv)
        self.x_oo = tot[1 + no * nv:1 + no * nv + no * no].reshape(no, no)
        self.x_vv = tot[1 + no * nv + no * no:].reshape(nv, nv)
        self.xo = self.x_oo + self.x_oo.T
        self.xv = self.x_vv + self.x_vv.T
        diag = (diag + (np.diag(self.xo)[:, None] + np.diag(self.xv)[None, :]).ravel()
                - xdiag.ravel())
        self.diag = diag
        self._fold_diag_setup()

    def _first_order_diagonal(self):
        """The orbital-energy differences plus the first-order (and reaction
        field) diagonal of the singles block, (no nv,)."""
        f, no, nv = self.f, self.no, self.nv
        coul = np.einsum('pP,pP->p', f.L, f.L, optimize=True)        # (ia|ia)
        # (ii|aa) = sum_P B_iiP B_aaP, summed over the auxiliary chunks in order
        Bii = np.ascontiguousarray(np.einsum('iiP->iP', f.Boo))
        terms = {}
        for c in self.my_chunks:
            p0, p1 = self.chunks[c]
            terms[c] = (Bii[:, p0:p1] @ np.einsum('aPa->Pa', f.bvv(c, p0, p1))).ravel()
        exch = ordered_chain_sum(terms, self.owners, self.comm, np.zeros(no * nv))
        first = (2.0 * coul - exch) if self.spin == 'singlet' else -exch
        diag = self.delta + first
        if self.rf is not None and self.spin == 'singlet':
            diag = diag + 2.0 * self.rf.diagonal()
        return diag

    def _amplitude_pass(self):
        """One pass over this rank's occupied tiles, summed in tile order:
        [e_mp2, the x diagonal, x_oo, x_vv] packed, the spin channel's MP1
        amplitudes of each tile kept for M's products while they fit
        EE_RI_AMPLITUDE_CACHE_GB on this rank."""
        no, nv = self.no, self.nv
        singlet = self.spin == 'singlet'
        parts = []
        # the spin channel's MP1 amplitudes of each tile, kept for M's
        # products while they fit EE_RI_AMPLITUDE_CACHE_GB on this rank
        self._T, kept = {}, 0.0
        for t, (k0, k1) in self.my_tiles:
            I = self._ints(k0, k1)
            tau = np.empty_like(I)

            def seg(k, c0, c1, I=I, tau=tau, k0=k0):
                A = I[k, c0:c1]
                Ax = I[k, :, :, c0:c1].transpose(2, 1, 0)
                D = self._denom(k0 + k, c0, c1)
                # scaled amplitudes; K (2 A - Ax, -Ax) stays bare
                t = self.weights.singlet(A, Ax) / D
                tau[k, c0:c1] = t
                e = -float(np.sum(A * t))
                xd = ((2.0 * A - Ax) * t).sum(0) if singlet else (
                    -Ax * self.weights.triplet(A, Ax) / D).sum(0)
                return e, xd

            # one addend per tile, packed [e_mp2, x diag, x_oo, x_vv]
            e_t, xd_t = 0.0, np.zeros((no, nv))
            for e, xd in self._rows(k1 - k0, seg):
                e_t += e
                xd_t += xd
            kt = k1 - k0
            # x_oo[i, j] = 1/2 sum_kdc I[k,d,i,c] tau[k,d,j,c]
            Io = I.transpose(2, 0, 1, 3).reshape(no, -1)
            To = tau.transpose(2, 0, 1, 3).reshape(no, -1)
            xo_t = 0.5 * (Io @ To.T)
            del Io, To
            # x_vv[a, b] = 1/2 sum_klc I[k,a,l,c] tau[k,b,l,c]
            xv_t = np.zeros((nv, nv))
            for k in range(kt):
                xv_t += 0.5 * (I[k].reshape(nv, -1) @ tau[k].reshape(nv, -1).T)
            if kept + tau.nbytes / BYTES_PER_GB <= EE_RI_AMPLITUDE_CACHE_GB:
                # singlet: tau is T itself; triplet: T = -(ic|ka) / D, from I
                self._T[t] = tau if singlet else self._triplet_amplitudes(I, k0)
                kept += tau.nbytes / BYTES_PER_GB
            del I, tau
            parts.append((t, np.concatenate([[e_t], xd_t.ravel(), xo_t.ravel(),
                                             xv_t.ravel()])))
        return ordered_sum(parts, self.comm, shape=(1 + no * nv + no * no + nv * nv,),
                           onto=np.zeros(1 + no * nv + no * no + nv * nv))

    def _fold_diag_setup(self):
        """The Laplace sums of fold_diagonal, once: with 1/x = sum_t W_t e^(-t x)
        and every exponent split into non-negative parts (so nothing overflows),

            P[t, a] = sum_c e^(-t (e_c - e_v)) sum_PQ B_acP X_t,PQ B_acQ
            H[t, i] = sum_k e^(-t (e_o - e_k)) sum_PQ B_kiP X_t,PQ B_kiQ
            X_t,PQ  = sum_ld B_ldP B_ldQ e^(-t (e_d - e_l - e_v + e_o))

        e_v, e_o the lowest virtual and highest occupied energy. The virtuals
        are dealt round-robin and taken in batches of tile_gb, each
        virtual's B_vv rows whole on its rank and its sum one fixed-shape
        product, so the bits do not depend on the rank count."""
        f, no, nv, naux = self.f, self.no, self.nv, self.f.naux
        eo, ev = f.eo, f.ev
        x_hi = 2.0 * (ev.max() - eo.min())
        s = np.arange(np.log(EE_RI_FOLD_DIAG_TAIL / x_hi),
                      np.log(EE_RI_FOLD_DIAG_EXPONENT / EE_RI_FOLD_DIAG_GAP)
                      + EE_RI_FOLD_DIAG_STEP, EE_RI_FOLD_DIAG_STEP)
        self.lap_t = np.exp(s)
        self.lap_w = EE_RI_FOLD_DIAG_STEP * self.lap_t
        gap = ev.min() - eo.max()
        L2 = self.LT.reshape(naux, -1)
        eld = (ev[None, :] - eo[:, None]).ravel() - gap
        Boo2 = f.BooP.reshape(naux, no * no)
        nt = self.lap_t.size
        P, H = np.zeros((nt, nv)), np.empty((nt, no))
        deal = [list(range(r, nv, self.size)) for r in range(self.size)]
        rank = self.rank
        nb = max(1, int(self.tile_gb * BYTES_PER_GB // (8.0 * naux * nv)))
        for b in range(max(1, -(-len(deal[0]) // nb))):
            batch = [d[b * nb:(b + 1) * nb] for d in deal]
            rows = self._bvv_rows(batch)
            for j, t in enumerate(self.lap_t):
                Xt = (L2 * np.exp(-t * eld)) @ L2.T
                ec = np.exp(-t * (ev - ev.min()))
                for a, Ba in zip(batch[rank], rows):
                    P[j, a] = np.sum(Ba * (Xt @ Ba), axis=0) @ ec
                if b == 0:
                    # [k, i]: sum_PQ B_kiP X_PQ B_kiQ
                    q = np.sum(Boo2 * (Xt @ Boo2), axis=0).reshape(no, no)
                    H[j] = np.exp(-t * (eo.max() - eo)) @ q
            del rows
        # each virtual's sum on one rank, zeros elsewhere: exact
        self.lap_p = reduce_sum(P, self.comm)
        self.lap_h = H

    def _bvv_rows(self, batch):
        """B^P_ac (naux, nv) for each a in batch[rank], batch[r] the virtuals
        rank r wants: views of the whole B_vv, or gathered from the auxiliary
        chunks' owners in one exchange_blocks that every rank enters."""
        f, nv, naux = self.f, self.nv, self.f.naux
        rank = self.rank
        if f.Bvv is not None:
            return [f.Bvv[a] for a in batch[rank]]
        mine_of = [[c for c, o in enumerate(self.owners) if o == r]
                   for r in range(self.size)]
        send = [np.concatenate([np.ravel(f.bvv(c, *self.chunks[c])[a])
                                for a in batch[r] for c in self.my_chunks] or
                               [np.empty(0)]) for r in range(self.size)]
        shapes = [(len(batch[rank]) * nv * sum(self.chunks[c][1] - self.chunks[c][0]
                                                for c in mine_of[r]),)
                  for r in range(self.size)]
        got = exchange_blocks(send, shapes, self.comm)
        out = [np.empty((naux, nv)) for _ in batch[rank]]
        for r in range(self.size):
            off = 0
            for k in range(len(out)):
                for c in mine_of[r]:
                    p0, p1 = self.chunks[c]
                    out[k][p0:p1] = got[r][off:off + (p1 - p0) * nv].reshape(p1 - p0, nv)
                    off += (p1 - p0) * nv
        return out

    def fold_diagonal(self, omega):
        """The configuration-dependent model of the folded doubles' diagonal at
        omega, (no nv,), Hartree: -EE_RI_FOLD_DIAG_PARTICLE times the particle
        relaxation sum_cld (ac|ld)^2 / (D - w) - EE_RI_FOLD_DIAG_HOLE times the
        hole one sum_kld (ki|ld)^2 / (D - w); zeros without doubles. For the
        block solver's seeds: the spread over the configurations, not the level."""
        if self.level != 'adc2':
            return np.zeros(self.n)
        eo, ev, t = self.f.eo, self.f.ev, self.lap_t
        gap = ev.min() - eo.max()
        # the rest of each exponent, >= d_min - w for every configuration
        rp = np.maximum(2.0 * ev.min() - eo.max() - eo - omega, 0.0)       # (no,)
        rh = np.maximum(ev + gap - eo.max() - omega, 0.0)                  # (nv,)
        part = (self.lap_w[None, :] * np.exp(-np.outer(rp, t))) @ self.lap_p
        hole = (self.lap_w[None, :] * np.exp(-np.outer(rh, t))) @ self.lap_h
        return -(EE_RI_FOLD_DIAG_PARTICLE * part
                 + EE_RI_FOLD_DIAG_HOLE * hole.T).ravel()

    def _triplet_amplitudes(self, I, k0):
        """T[k, c, l, d] = -(kd|lc) / D_kcld of a tile, from I = (kc|ld);
        spin-component scaled, [(c_ss - c_os) (kc|ld) - c_ss (kd|lc)] / D."""
        T = np.empty_like(I)

        def seg(k, c0, c1):
            Ax = I[k, :, :, c0:c1].transpose(2, 1, 0)
            T[k, c0:c1] = (self.weights.triplet(I[k, c0:c1], Ax)
                           / self._denom(k0 + k, c0, c1))
        self._rows(I.shape[0], seg)
        return T

    def _amplitudes(self, t, k0, k1):
        """The spin channel's MP1 amplitudes of tile t: cached, or rebuilt from
        (kc|ld) by the same arithmetic, so the bits do not depend on the cache."""
        if t in self._T:
            return self._T[t]
        I = self._ints(k0, k1)
        if self.spin != 'singlet':
            return self._triplet_amplitudes(I, k0)
        T = np.empty_like(I)

        def seg(k, c0, c1):
            T[k, c0:c1] = (self.weights.singlet(I[k, c0:c1],
                                                I[k, :, :, c0:c1].transpose(2, 1, 0))
                           / self._denom(k0 + k, c0, c1))
        self._rows(k1 - k0, seg)
        return T

    # ---- global (cheap) pieces of a product ------------------------------

    def _coulomb(self, X):
        """sum_jb (ia|jb) X_jb, X (no nv, m)."""
        return self.f.L @ (self.f.L.T @ X)

    def _exchange(self, X, kinds):
        """The exchange terms of X (no nv, m) named in kinds, each (no nv, m):
        'ovov' sum_kc (ic|ka) X_kc, 'oovv' sum_jb (ij|ab) X_jb; summed over
        auxiliary chunks dealt over the ranks, every column of X at once."""
        f, no, nv, m = self.f, self.no, self.nv, X.shape[1]
        Xc = X.reshape(no, nv, m)
        if 'ovov' in kinds:
            Xcat = np.ascontiguousarray(Xc.transpose(1, 0, 2)).reshape(nv, no * m)
        if 'oovv' in kinds:
            Xs = np.ascontiguousarray(Xc.transpose(2, 0, 1)).reshape(m * no, nv)
        terms = {}
        for c in self.my_chunks:
            p0, p1 = self.chunks[c]
            pc = p1 - p0
            t = np.empty((len(kinds), no * nv, m))
            for q, kind in enumerate(kinds):
                if kind == 'ovov':
                    # LTc[(P, i), c] = L_icP, read again as [(P, k), a] = L_kaP
                    LTc = self.LT[p0:p1].reshape(pc * no, nv)
                    # Z[P, i, k, j] = sum_c L_icP x^j_kc: O(no^2 nv naux)
                    Z = (LTc @ Xcat).reshape(pc, no, no, m)
                    # t^j_ia = sum_Pk Z[P, i, k, j] L_kaP
                    Zt = np.ascontiguousarray(Z.transpose(3, 1, 0, 2)
                                              ).reshape(m * no, pc * no)
                    t[q] = (Zt @ LTc).reshape(m, no * nv).T
                else:
                    # Y[(j, k), (P, a)] = sum_b x^j_kb B_baP, B_vv symmetric
                    Bc = f.bvv(c, p0, p1).reshape(nv, pc * nv)
                    Y = (Xs @ Bc).reshape(m, no * pc, nv)
                    # t^j_ia = sum_kP B_ikP Y[(j, k), (P, a)]
                    Boo2 = np.ascontiguousarray(f.Boo[:, :, p0:p1]).reshape(no, -1)
                    for j in range(m):
                        t[q, :, j] = (Boo2 @ Y[j]).ravel()
            terms[c] = t
        return ordered_chain_sum(terms, self.owners, self.comm,
                                 np.zeros((len(kinds), no * nv, m)))

    def _k_apply(self, X, ex=None):
        """K X: singlet 2 (ia|kc) - (ic|ka), triplet -(ic|ka); ex the term
        sum_kc (ic|ka) X_kc where already made."""
        if ex is None:
            ex = self._exchange(X, ('ovov',))[0]
        if self.spin == 'singlet':
            return 2.0 * self._coulomb(X) - ex
        return -ex

    def _dressed(self, x):
        """Bt^P_kc = sum_i B_kiP x_ic - sum_a x_ka B_aPc, auxiliary-major as
        (naux, no nv): each rank forms the rows of its auxiliary chunks, which
        travel verbatim to every rank (allgather_ranges)."""
        f, no, nv = self.f, self.no, self.nv
        x = np.ascontiguousarray(x, float).reshape(no, nv)
        BtT = np.zeros((f.naux, no * nv))
        for c in self.my_chunks:
            p0, p1 = self.chunks[c]
            pc = p1 - p0
            t = f.BooP[p0:p1] @ x                                       # [P, k, c]
            t -= (x @ f.bvv(c, p0, p1).reshape(nv, -1)).reshape(no, pc, nv
                                                                ).transpose(1, 0, 2)
            BtT[p0:p1] = t.reshape(pc, -1)
        return allgather_ranges(BtT, self.chunk_ranges, self.comm)

    def _pair(self, BtT, k0, k1):
        """(H, H^T) of a tile, each (kt, nv, no, nv): H_kc,ld = sum_P Bt_kcP B_ldP
        and H^T_kc,ld = sum_P B_kcP Bt_ldP, from the auxiliary-major factors."""
        nv, no = self.nv, self.no
        kt, rows = k1 - k0, slice(k0 * nv, k1 * nv)
        LT2 = self.LT.reshape(self.f.naux, -1)
        H = (BtT[:, rows].T @ LT2).reshape(kt, nv, no, nv)
        HT = (LT2[:, rows].T @ BtT).reshape(kt, nv, no, nv)
        return H, HT

    def _back_oo(self, Gam, k0, k1):
        """sum_kP B_kjP Gam_kbP of the rows k0..k1-1 of G, Gam (..., kt nv, naux),
        (..., no nv): the B_oo half of the back-transformation, local to a tile."""
        f, no, nv = self.f, self.no, self.nv
        kt = k1 - k0
        lead = Gam.shape[:-2]
        GkPb = np.ascontiguousarray(Gam.reshape(lead + (kt, nv, -1)).swapaxes(-1, -2)
                                    ).reshape(lead + (-1, nv))       # [(k, P), b]
        Bjk = np.ascontiguousarray(f.Boo[k0:k1].transpose(1, 0, 2)).reshape(no, -1)
        return (Bjk @ GkPb).reshape(lead + (-1,))

    def _back_vv(self, gams, p):
        """-sum_cP Gam_kcP B_cPb for every tile's rows, gams {t: (p, kt nv, naux)}
        of this rank's tiles: each auxiliary slice goes to its chunk's owner
        (exchange_blocks), which contracts it with its B_vv chunk; the chunks are
        summed in chunk order (ordered_chain_sum). Returns (p, no nv), every rank."""
        f, nv, no = self.f, self.nv, self.no
        rank = self.rank
        mine_of = [[c for c, o in enumerate(self.owners) if o == r]
                   for r in range(self.size)]
        tiles_of = [list(range(r, len(self.tiles), self.size))
                    for r in range(self.size)]

        def width(t, c):
            return p * (self.tiles[t][1] - self.tiles[t][0]) * nv * (
                self.chunks[c][1] - self.chunks[c][0])

        send = [np.concatenate([np.ravel(gams[t][:, :, slice(*self.chunks[c])])
                                for t in tiles_of[rank] for c in mine_of[s]] or
                               [np.empty(0)]) for s in range(self.size)]
        shapes = [(sum(width(t, c) for t in tiles_of[r] for c in mine_of[rank]),)
                  for r in range(self.size)]
        got = exchange_blocks(send, shapes, self.comm)
        # where each (tile, chunk) block lies in what rank r sent
        at = {}
        for r in range(self.size):
            off = 0
            for t in tiles_of[r]:
                for c in self.my_chunks:
                    at[t, c] = (r, off)
                    off += width(t, c)
        terms = {}
        for c in self.my_chunks:
            p0, p1 = self.chunks[c]
            # every tile's rows of the chunk in one (p no, nv pc) block, so the
            # chunk is read once and the product's shape is the rank count's none
            Gc = np.empty((p, no, nv * (p1 - p0)))
            for t, (k0, k1) in enumerate(self.tiles):
                r, off = at[t, c]
                Gc[:, k0:k1] = got[r][off:off + width(t, c)].reshape(
                    p, k1 - k0, nv * (p1 - p0))
            terms[c] = -(Gc.reshape(p * no, -1) @ f.bvv(c, p0, p1).reshape(-1, nv)
                         ).reshape(p, no * nv)
        return ordered_chain_sum(terms, self.owners, self.comm, np.zeros((p, no * nv)))

    # ---- the product -------------------------------------------------------

    def apply(self, U, omegas=None):
        """(A_eff U, |Y|^2): U (n, m); omegas None (M alone) or (m,) Hartree,
        one frequency per column. |Y|^2 (m,) is sum_K Y_K^2 of the doubles
        image Y = (D - w)^-1 C^T u (zeros without omegas)."""
        # contiguous, as the MPI stand-in hands it: a Davidson's strided or
        # column-major block would take other BLAS paths and change the bits
        U = np.ascontiguousarray(U, float)
        if U.ndim == 1:
            U = U[:, None]
        m = U.shape[1]
        self.matvecs += m
        no, nv = self.no, self.nv
        singlet = self.spin == 'singlet'
        out = self.delta[:, None] * U
        # one pass over the auxiliary chunks for both exchange terms of U
        ex = self._exchange(U, ('oovv',) if self.level == 'adc1'
                            else ('oovv', 'ovov'))
        out += (2.0 * self._coulomb(U) - ex[0]) if singlet else -ex[0]
        if self.rf is not None and singlet:
            out += 2.0 * self.rf.apply(U)
        norms = np.zeros(m)
        if self.level == 'adc1':
            return out, norms
        for j in range(m):
            u = U[:, j].reshape(no, nv)
            out[:, j] += (self.xo @ u + u @ self.xv).ravel()
        KU = self._k_apply(U, ex[1])
        W = np.concatenate([U, KU], axis=1)            # (n, 2m)
        # T W by tiles of rows; rows of other ranks' tiles stay zero, so the
        # sum over ranks is exact in any order
        TW = np.zeros_like(W)
        for t, (k0, k1) in self.my_tiles:
            kt = k1 - k0
            T = self._amplitudes(t, k0, k1)
            TW[k0 * nv:k1 * nv] = T.reshape(kt * nv, -1) @ W
            del T
        reduce_sum(TW, self.comm)
        out -= 0.5 * (self._k_apply(TW[:, :m]) + TW[:, m:])
        if omegas is None:
            return out, norms
        # the doubles, one vector at a time: one dressed factor is held
        omegas = np.broadcast_to(np.asarray(omegas, float), (m,))
        for j in range(m):
            BtT = self._dressed(U[:, j])
            Btg = None if self.gpu is None else self.gpu.upload(BtT)
            parts, gams = [], {}
            for t, (k0, k1) in self.my_tiles:
                Gam, img, nrm = (self._doubles_tile(k0, k1, BtT, omegas[j])
                                 if Btg is None else
                                 self.gpu.doubles_tile(k0, k1, Btg, omegas[j]))
                parts.append((t, np.concatenate([img, [nrm]])))
                gams[t] = Gam[None]
            del BtT
            # the tiles in tile order, then the B_vv half in chunk order
            tot = ordered_sum(parts, self.comm, shape=(self.n + 1,),
                              onto=np.zeros(self.n + 1))
            out[:, j] += tot[:-1]
            out[:, j] += self._back_vv(gams, 1)[0]
            norms[j] = tot[-1]
        return out, norms

    def _doubles_tile(self, k0, k1, BtT, omega):
        """One tile of the doubles: (Gam (kt nv, naux), the B_oo half of its
        folded image (no nv,), its |Y|^2)."""
        nv, L = self.nv, self.f.L
        kt = k1 - k0
        H, HT = self._pair(BtT, k0, k1)
        wt = self.weights
        if self.spin == 'singlet':
            def seg_g(k, c0, c1):
                H[k, c0:c1] += HT[k, c0:c1]
            self._rows(kt, seg_g)
            R = HT                                     # HT is consumed

            def seg_r(k, c0, c1):
                G = H[k, c0:c1]
                Gx = H[k, :, :, c0:c1].transpose(2, 1, 0)
                w = 1.0 / (omega - self._denom(k0 + k, c0, c1))
                R[k, c0:c1] = w * wt.singlet(G, Gx)
                return float(np.sum(w * w * wt.singlet_overlap(G, Gx, G, Gx)))
        else:
            R = np.empty_like(H)

            def seg_r(k, c0, c1):
                Hb, Tb = H[k, c0:c1], HT[k, c0:c1]
                Sx = (H[k, :, :, c0:c1] + HT[k, :, :, c0:c1]).transpose(2, 1, 0)
                w = 1.0 / (omega - self._denom(k0 + k, c0, c1))
                A = Hb - Tb
                S = Hb + Tb - Sx
                R[k, c0:c1] = w * wt.triplet_image(A, S)
                return float(np.sum(w * w * wt.triplet_overlap(A, S, A, S)))
        nrm = sum(self._rows(kt, seg_r))
        Gam = R.reshape(kt * nv, -1) @ L
        del H, HT, R
        return Gam, self._back_oo(Gam, k0, k1), nrm

    def fold_images(self, U, nodes, derivative=None):
        """(M U, F): F (p, n, m), F[q] = sum_K C_K (C^T U)_K / (w_q - D_K) =
        A_eff(w_q) U - M U at each of the p frequencies w_q, or, where
        derivative[q] (bool, (p,); default none), its ω-derivative dA_eff/dω U =
        -sum_K C_K (C^T U)_K / (w_q - D_K)^2; a column's doubles are built once and
        only the denominator changes between frequencies."""
        # contiguous: a strided column would leave BLAS and change the bits
        U = np.ascontiguousarray(U, float)
        if U.ndim == 1:
            U = U[:, None]
        nodes = np.atleast_1d(np.asarray(nodes, float))
        n, m, p = self.n, U.shape[1], nodes.size
        derivative = (np.zeros(p, bool) if derivative is None
                      else np.broadcast_to(np.asarray(derivative, bool), (p,)))
        MU = self.apply(U, None)[0]
        self.image_columns += m
        self.node_images += m * p
        if self.level == 'adc1':
            return MU, np.zeros((p, n, m))
        F = np.empty((p, n, m))
        for j in range(m):
            BtT = self._dressed(U[:, j])
            Btg = None if self.gpu is None else self.gpu.upload(BtT)
            parts, gams = [], {}
            for t, (k0, k1) in self.my_tiles:
                gams[t], img = (self._images_tile(k0, k1, BtT, nodes, derivative)
                                if Btg is None else
                                self.gpu.images_tile(k0, k1, Btg, nodes, derivative))
                parts.append((t, img.ravel()))
            del BtT, Btg
            # the tiles in tile order, then the B_vv half in chunk order
            F[:, :, j] = ordered_sum(parts, self.comm, shape=(p * n,),
                                     onto=np.zeros(p * n)).reshape(p, n)
            F[:, :, j] += self._back_vv(gams, p)
        return MU, F

    def _images_tile(self, k0, k1, BtT, nodes, derivative):
        """One tile at the frequencies nodes: (Gam (p, kt nv, naux), the B_oo
        half of its images (p, no nv)), derivative[q] as fold_images'."""
        nv, L = self.nv, self.f.L
        kt = k1 - k0
        H, HT = self._pair(BtT, k0, k1)
        # S = H + H^T, held in HT
        def seg_s(k, c0, c1):
            HT[k, c0:c1] += H[k, c0:c1]
        self._rows(kt, seg_s)
        wt = self.weights
        if self.spin == 'singlet':
            # Gt = (c_os + c_ss) S - c_ss S~, held in H: R = w Gt at every frequency
            def seg_g(k, c0, c1):
                H[k, c0:c1] = wt.singlet(HT[k, c0:c1],
                                         HT[k, :, :, c0:c1].transpose(2, 1, 0))
        else:
            # Gt = c_os (2 H - S) + c_ss (S - S~), held in H
            def seg_g(k, c0, c1):
                S = HT[k, c0:c1]
                H[k, c0:c1] = wt.triplet_image(2.0 * H[k, c0:c1] - S,
                                               S - HT[k, :, :, c0:c1].transpose(2, 1, 0))
        self._rows(kt, seg_g)
        Gt, R = H, HT
        gams = np.empty((nodes.size, kt * nv, self.f.naux))
        for q, w in enumerate(nodes):
            def seg_r(k, c0, c1, w=w, two=bool(derivative[q])):
                x = w - self._denom(k0 + k, c0, c1)
                # Gt / (ω - D), or its ω-derivative -Gt / (ω - D)^2
                R[k, c0:c1] = -Gt[k, c0:c1] / (x * x) if two else Gt[k, c0:c1] / x
            self._rows(kt, seg_r)
            gams[q] = R.reshape(kt * nv, -1) @ L
        return gams, self._back_oo(gams, k0, k1)

    def doubles_dot(self, ur, wr, us, ws):
        """sum_K Y_K(ur, wr) Y_K(us, ws), the doubles images' overlap."""
        if self.level == 'adc1':
            return 0.0
        same = ur is us or np.array_equal(ur, us)
        Br = self._dressed(ur)
        Bs = Br if same else self._dressed(us)
        if self.gpu is not None:
            Bgr = self.gpu.upload(Br)
            Bgs = Bgr if same else self.gpu.upload(Bs)
            parts = [(t, np.array([self.gpu.overlap(k0, k1, Bgr, wr, Bgs, ws)]))
                     for t, (k0, k1) in self.my_tiles]
            return float(ordered_sum(parts, self.comm, shape=(1,),
                                     onto=np.zeros(1))[0])
        singlet, wt = self.spin == 'singlet', self.weights
        parts = []
        for t, (k0, k1) in self.my_tiles:
            kt = k1 - k0
            Hr, Tr = self._pair(Br, k0, k1)
            Hs, Ts = (Hr, Tr) if same else self._pair(Bs, k0, k1)

            def seg(k, c0, c1):
                D = self._denom(k0 + k, c0, c1)
                w2 = 1.0 / ((wr - D) * (ws - D))
                hr, tr, hs, ts = (X[k, c0:c1] for X in (Hr, Tr, Hs, Ts))
                xr = (Hr[k, :, :, c0:c1] + Tr[k, :, :, c0:c1]).transpose(2, 1, 0)
                xs = (Hs[k, :, :, c0:c1] + Ts[k, :, :, c0:c1]).transpose(2, 1, 0)
                if singlet:
                    return float(np.sum(w2 * wt.singlet_overlap(hr + tr, xr, hs + ts, xs)))
                return float(np.sum(w2 * wt.triplet_overlap(hr - tr, hr + tr - xr,
                                                            hs - ts, hs + ts - xs)))

            parts.append((t, np.array([sum(self._rows(kt, seg))])))
        return float(ordered_sum(parts, self.comm, shape=(1,),
                                 onto=np.zeros(1))[0])

    # ---- the ee_fold interface --------------------------------------------

    def pieces(self):
        """The dict ee_fold.solve_folded takes: the operator and its spin
        channel stand in for M, V, Vt, D (pieces['operator'])."""
        return {'operator': self, 'level': self.level, 'no': self.no,
                'nv': self.nv, 'd_min': self.d_min if self.level == 'adc2'
                else np.inf, 'spin': self.spin, 'dense_limit': 0}


class FoldedMatvec:
    """u -> A_eff(w) u with a block form `matmat` the Davidson batches."""

    def __init__(self, op, omega):
        self.op, self.omega = op, omega

    def _omegas(self, m):
        return None if self.omega is None else np.full(m, float(self.omega))

    def __call__(self, u):
        return self.op.apply(np.asarray(u, float).ravel()[:, None],
                             self._omegas(1))[0][:, 0]

    def matmat(self, U):
        return self.op.apply(U, self._omegas(U.shape[1]))[0]

    def both(self, U):
        """(A_eff(w) U, dA_eff/dw U), U (n, m): one pass of the doubles for both."""
        w = float(self.omega)
        MU, F = self.op.fold_images(U, [w, w], derivative=[False, True])
        return MU + F[0], F[1]

    def norm2(self, u):
        if self.omega is None:
            return 0.0
        return float(self.op.apply(np.asarray(u, float).ravel()[:, None],
                                   self._omegas(1))[1][0])


class _RootDriven:
    """Rank 0's stand-in for a distributed RIFoldADC2: every product and
    doubles overlap is announced to the other ranks (root_driven_solve's
    `act`) and computed by all of them together."""

    def __init__(self, op, act):
        self.op, self.act = op, act
        for name in ('spin', 'level', 'n', 'no', 'nv', 'diag', 'd_min',
                     'e_mp2', 'kt'):
            setattr(self, name, getattr(op, name))

    @property
    def matvecs(self):
        return self.op.matvecs

    def apply(self, U, omegas=None):
        U = np.asarray(U, float)
        if U.ndim == 1:
            U = U[:, None]
        m = U.shape[1]
        om = (np.full(m, np.nan) if omegas is None else
              np.broadcast_to(np.asarray(omegas, float), (m,)))
        Y = self.act(np.vstack([U, om[None, :], np.zeros((1, m))]))
        return Y[:self.n], Y[self.n]

    def fold_images(self, U, nodes, derivative=None):
        U = np.asarray(U, float)
        if U.ndim == 1:
            U = U[:, None]
        nodes = np.atleast_1d(np.asarray(nodes, float))
        n, m, p = self.n, U.shape[1], nodes.size
        # columns: U, padded to hold the nodes in row n; row n+1 the tag 2,
        # then m, p and the p derivative flags
        X = np.zeros((n + 2, max(m, 3 + p)))
        X[:n, :m], X[n, :p] = U, nodes
        X[n + 1, :3] = 2.0, m, p
        if derivative is not None:
            X[n + 1, 3:3 + p] = np.broadcast_to(np.asarray(derivative, bool), (p,))
        Y = self.act(X)
        return Y[:n], Y[n:].reshape(p, n, m)

    def doubles_dot(self, ur, wr, us, ws):
        X = np.column_stack([np.concatenate([np.ravel(ur), [wr, 1.0]]),
                             np.concatenate([np.ravel(us), [ws, 1.0]])])
        return float(self.act(X)[0, 0])

    def fold_diagonal(self, omega):
        # made at construction on every rank: local
        return self.op.fold_diagonal(omega)

    def pieces(self):
        p = self.op.pieces()
        p['operator'] = self
        return p


def drive(op, solve):
    """solve(operator) with `op` distributed over its communicator: rank 0
    runs `solve` on a stand-in whose products every rank computes together,
    the other ranks serve them; rank 0's return value (keep it to arrays and
    lists: it is pickled) comes back on every rank. Serially, solve(op)."""
    from src.Base.utils.linear_algebra.krylov import root_driven_solve
    comm = op.comm
    if comm is None or comm.Get_size() == 1:
        return solve(op)
    n = op.n

    def collective(X):
        if X[n + 1, 0] == 0.0:
            om = X[n]
            AU, nrm = op.apply(X[:n], None if np.isnan(om[0]) else om)
            return np.vstack([AU, nrm[None, :]])
        if X[n + 1, 0] == 2.0:
            m, p = int(X[n + 1, 1]), int(X[n + 1, 2])
            MU, F = op.fold_images(X[:n, :m], X[n, :p], X[n + 1, 3:3 + p] > 0.5)
            return np.vstack([MU, F.reshape(p * n, m)])
        return np.array([[op.doubles_dot(X[:n, 0], X[n, 0], X[:n, 1], X[n, 1])]])

    return root_driven_solve(lambda act: solve(_RootDriven(op, act)),
                             collective, rows=n + 2, comm=comm)
