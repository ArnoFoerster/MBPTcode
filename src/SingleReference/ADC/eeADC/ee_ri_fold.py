"""Density-fitted ADC(2) singles operator with the doubles folded in, for large
molecules: no array of doubles size is ever held whole.

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

with x_oo, x_vv the occupied/virtual MP2 density-like contractions. Per
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
Comput. 9, 977 (2013); Mewes et al., J. Chem. Theory Comput. 11, 4643 (2015)).
At ADC(1) this is pyscf's TDA in PCM with a non-equilibrium (optical)
response, which the tests check to rounding.
"""
import numpy as np
from pyscf import df as pyscf_df
from pyscf import lib

from src.Base.constants import EE_RI_AUX_CHUNK, EE_RI_TILE_GB
from src.Base.utils.mpi_grid import current_comm, ordered_sum, reduce_sum
from src.Base.utils.threads import openmp_threads, row_map

SPINS = ('singlet', 'triplet')
LEVELS = ('adc1', 'adc2')


# ----------------------------------------------------------------------
# factors
# ----------------------------------------------------------------------

class RIFactors:
    """The DF factor B^P_pq = sum_mu nu C_mu p C_nu q (P|mu nu)~ of one closed-
    shell reference, in the layouts the operator contracts:

    eo, ev : (no,), (nv,) active orbital energies, Hartree
    L      : (no nv, naux), rows (i, a): B^P_ia
    Boo    : (no, no, naux), B^P_ij
    BooT   : (no, naux, no), B^P_ij as [i, P, j]
    Bvv    : (nv, naux, nv), B^P_ab as [a, P, b]
    """

    def __init__(self, eo, ev, L, Boo, Bvv):
        self.eo, self.ev = np.asarray(eo, float), np.asarray(ev, float)
        self.no, self.nv = self.eo.size, self.ev.size
        self.L = np.ascontiguousarray(L)
        self.naux = self.L.shape[1]
        self.Boo = np.ascontiguousarray(Boo)
        self.BooT = np.ascontiguousarray(Boo.transpose(0, 2, 1))
        self.Bvv = np.ascontiguousarray(Bvv)

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
        return self.L.nbytes + self.Boo.nbytes + self.BooT.nbytes + self.Bvv.nbytes


def bare_factors(mol, mf, frozen=0, auxbasis=None, chunk=EE_RI_AUX_CHUNK):
    """RIFactors of the BARE Coulomb interaction on the orbitals of `mf`, the
    `frozen` lowest occupied dropped.

    auxbasis None takes the mean field's own DF object (the JK fit of a DF
    SCF); a name builds a separate one (the correlation fit, e.g.
    'aug-cc-pvdz-ri'). An environment attached to `mf` is deliberately not
    applied: the continuum enters this route through the singles block alone
    (module docstring), never through the factor.
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
    L = np.empty((no, nv, naux))
    Boo = np.empty((no, no, naux))
    Bvv = np.empty((nv, naux, nv))
    p0 = 0
    for blk in with_df.loop(blksize=chunk):
        p1 = p0 + blk.shape[0]
        ao = lib.unpack_tril(blk)                       # (p, nao, nao)
        nao = ao.shape[1]
        half_o = (ao.reshape(-1, nao) @ Co).reshape(-1, nao, no)   # (p, mu, i)
        half_o = half_o.transpose(0, 2, 1).reshape(-1, nao)        # (p i, mu)
        L[:, :, p0:p1] = (half_o @ Cv).reshape(-1, no, nv).transpose(1, 2, 0)
        Boo[:, :, p0:p1] = (half_o @ Co).reshape(-1, no, no).transpose(1, 2, 0)
        del half_o
        half_v = (ao.reshape(-1, nao) @ Cv).reshape(-1, nao, nv)
        half_v = half_v.transpose(0, 2, 1).reshape(-1, nao)        # (p a, mu)
        Bvv[:, p0:p1, :] = (half_v @ Cv).reshape(-1, nv, nv).transpose(1, 0, 2)
        del half_v, ao
        p0 = p1
    if p0 != naux:
        raise RuntimeError(f'the DF loop returned {p0} of {naux} auxiliary rows')
    return RIFactors(eps[frozen:nocc], eps[nocc:], L.reshape(no * nv, naux),
                     Boo, Bvv)


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


class RIFoldADC2:
    """A_eff(w) of one spin channel of closed-shell ADC(1)/ADC(2) on flat
    singles vectors u (no nv,), (i, a) row-major, from RIFactors.

    level 'adc1' is CIS (no second order, no doubles); 'adc2' the strict
    ADC(2). reaction_field: a TransitionReactionField or None. tile_gb bounds
    the tile working set; threads the element-wise passes (default: the
    process's OpenMP count).

    Over MPI ranks (comm, default the current one) the occupied tiles are
    dealt round-robin and every tile's contribution is summed in tile order
    (ordered_sum), so at one tile size a product has the same bits at every
    rank count; each rank holds the factors whole. Construction, `apply` and `doubles_dot` are
    then collectives: every rank calls them with the same arguments
    (ee_driver drives them from rank 0 with root_driven_solve)."""

    def __init__(self, factors, spin, level='adc2', reaction_field=None,
                 tile_gb=EE_RI_TILE_GB, threads=None, comm=None):
        if spin not in SPINS:
            raise ValueError(f'spin={spin!r}; the folded route solves one '
                             f'channel, one of {SPINS}')
        if level not in LEVELS:
            raise ValueError(f'level={level!r}; expected one of {LEVELS}')
        self.f, self.spin, self.level = factors, spin, level
        self.rf = reaction_field
        self.threads = int(threads or openmp_threads())
        no, nv = factors.no, factors.nv
        self.no, self.nv, self.n = no, nv, no * nv
        self.delta = (factors.ev[None, :] - factors.eo[:, None]).ravel()
        self.d_min = 2.0 * (factors.ev.min() - factors.eo.max())
        self.e_mp2 = None
        self.matvecs = 0
        self.comm = current_comm() if comm is None else comm
        rank, size = ((0, 1) if self.comm is None else
                      (self.comm.Get_rank(), self.comm.Get_size()))
        # occupied orbitals per tile: what tile_gb allows, and no more than
        # gives every rank a tile
        per_k = 8.0 * nv * no * nv * 5
        self.kt = int(max(1, min(-(-no // size), tile_gb * 1024 ** 3 // per_k)))
        tiles = [(k0, min(k0 + self.kt, no)) for k0 in range(0, no, self.kt)]
        # (tile index, (k0, k1)) of this rank's tiles
        self.my_tiles = [(t, tiles[t]) for t in range(rank, len(tiles), size)]
        self._setup()

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
        f, no, nv = self.f, self.no, self.nv
        coul = np.einsum('pP,pP->p', f.L, f.L, optimize=True)        # (ia|ia)
        exch = np.einsum('iiP,aPa->ia', f.Boo, f.Bvv, optimize=True).ravel()
        first = (2.0 * coul - exch) if self.spin == 'singlet' else -exch
        diag = self.delta + first
        if self.rf is not None and self.spin == 'singlet':
            diag = diag + 2.0 * self.rf.diagonal()
        self.x_oo = np.zeros((no, no))
        self.x_vv = np.zeros((nv, nv))
        if self.level == 'adc1':
            self.diag = diag
            return
        singlet = self.spin == 'singlet'
        parts = []
        for t, (k0, k1) in self.my_tiles:
            I = self._ints(k0, k1)
            tau = np.empty_like(I)

            def seg(k, c0, c1, I=I, tau=tau, k0=k0):
                A = I[k, c0:c1]
                Ax = I[k, :, :, c0:c1].transpose(2, 1, 0)
                D = self._denom(k0 + k, c0, c1)
                N = 2.0 * A - Ax
                t = N / D
                tau[k, c0:c1] = t
                e = -float(np.sum(A * t))
                xd = (N * t).sum(0) if singlet else (Ax * Ax / D).sum(0)
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
            del I, tau
            parts.append((t, np.concatenate([[e_t], xd_t.ravel(), xo_t.ravel(),
                                             xv_t.ravel()])))
        tot = ordered_sum(parts, self.comm, shape=(1 + no * nv + no * no + nv * nv,),
                          onto=np.zeros(1 + no * nv + no * no + nv * nv))
        self.e_mp2 = float(tot[0])
        xdiag = tot[1:1 + no * nv].reshape(no, nv)
        self.x_oo = tot[1 + no * nv:1 + no * nv + no * no].reshape(no, no)
        self.x_vv = tot[1 + no * nv + no * no:].reshape(nv, nv)
        self.xo = self.x_oo + self.x_oo.T
        self.xv = self.x_vv + self.x_vv.T
        diag = (diag + (np.diag(self.xo)[:, None] + np.diag(self.xv)[None, :]).ravel()
                - xdiag.ravel())
        self.diag = diag

    # ---- global (cheap) pieces of a product ------------------------------

    def _coulomb(self, X):
        """sum_jb (ia|jb) X_jb, X (no nv, m)."""
        return self.f.L @ (self.f.L.T @ X)

    def _exchange_oovv(self, X):
        """sum_jb (ij|ab) X_jb, X (no nv, m)."""
        f, no, nv = self.f, self.no, self.nv
        out = np.empty_like(X)
        Bvv2 = f.Bvv.reshape(nv, -1)                    # [b, (P, a)]
        Boo2 = f.Boo.reshape(no, -1)                    # [i, (j, P)]
        for j in range(X.shape[1]):
            Y = X[:, j].reshape(no, nv) @ Bvv2          # [j, (P, a)]
            out[:, j] = (Boo2 @ Y.reshape(-1, nv)).ravel()
        return out

    def _exchange_ovov(self, X):
        """sum_kc (ic|ka) X_kc, X (no nv, m), in auxiliary chunks."""
        f, no, nv = self.f, self.no, self.nv
        L3 = f.L.reshape(no, nv, -1)
        out = np.zeros_like(X)
        step = max(1, int(EE_RI_TILE_GB * 1024 ** 3 / (8.0 * 3 * nv * nv)))
        for p0 in range(0, f.naux, step):
            p1 = min(p0 + step, f.naux)
            Lp = np.ascontiguousarray(L3[:, :, p0:p1])            # (k, a, P)
            Lp2 = Lp.reshape(no, -1)
            for j in range(X.shape[1]):
                x = X[:, j].reshape(no, nv)
                # W[c, a, P] = sum_k x_kc L_kaP; out_ia += sum_cP L_icP W_caP
                W = (x.T @ Lp2).reshape(nv, nv, p1 - p0)
                W = np.ascontiguousarray(W.transpose(0, 2, 1)).reshape(-1, nv)
                out[:, j] += (Lp2 @ W).ravel()
        return out

    def _k_apply(self, X):
        """K X: singlet 2 (ia|kc) - (ic|ka), triplet -(ic|ka)."""
        ex = self._exchange_ovov(X)
        if self.spin == 'singlet':
            return 2.0 * self._coulomb(X) - ex
        return -ex

    def _dressed(self, x):
        """Bt[k, c, P] = sum_i B_kiP x_ic - sum_a x_ka B_aPc, as (no nv, naux)."""
        f, no, nv = self.f, self.no, self.nv
        x = x.reshape(no, nv)
        t = (f.BooT.reshape(-1, no) @ x).reshape(no, -1, nv)       # [k, P, c]
        t -= (x @ f.Bvv.reshape(nv, -1)).reshape(no, -1, nv)
        return np.ascontiguousarray(t.transpose(0, 2, 1)).reshape(no * nv, -1)

    def _back(self, Gam, k0, k1):
        """The rows k0..k1-1 of G, Gam (kt nv, naux), carried back to the
        singles: sum_P [sum_k B_kjP Gam_kbP - sum_c Gam_jcP B_cPb], (no nv,)."""
        f, no, nv = self.f, self.no, self.nv
        kt = k1 - k0
        GkPb = np.ascontiguousarray(Gam.reshape(kt, nv, -1).transpose(0, 2, 1)
                                    ).reshape(-1, nv)                 # [(k, P), b]
        Bjk = np.ascontiguousarray(f.Boo[k0:k1].transpose(1, 0, 2)).reshape(no, -1)
        out = Bjk @ GkPb
        out[k0:k1] -= Gam.reshape(kt, -1) @ f.Bvv.reshape(-1, nv)
        return out.ravel()

    # ---- the product -------------------------------------------------------

    def apply(self, U, omegas=None):
        """(A_eff U, |Y|^2): U (n, m); omegas None (M alone) or (m,) Hartree,
        one frequency per column. |Y|^2 (m,) is sum_K Y_K^2 of the doubles
        image Y = (D - w)^-1 C^T u (zeros without omegas)."""
        U = np.asarray(U, float)
        if U.ndim == 1:
            U = U[:, None]
        m = U.shape[1]
        self.matvecs += m
        no, nv = self.no, self.nv
        singlet = self.spin == 'singlet'
        out = self.delta[:, None] * U
        ex = self._exchange_oovv(U)
        out += (2.0 * self._coulomb(U) - ex) if singlet else -ex
        if self.rf is not None and singlet:
            out += 2.0 * self.rf.apply(U)
        norms = np.zeros(m)
        if self.level == 'adc1':
            return out, norms
        for j in range(m):
            u = U[:, j].reshape(no, nv)
            out[:, j] += (self.xo @ u + u @ self.xv).ravel()
        KU = self._k_apply(U)
        W = np.concatenate([U, KU], axis=1)            # (n, 2m)
        # T W by tiles of rows; rows of other ranks' tiles stay zero, so the
        # sum over ranks is exact in any order
        TW = np.zeros_like(W)
        for t, (k0, k1) in self.my_tiles:
            kt = k1 - k0
            rows = slice(k0 * nv, k1 * nv)
            I = self._ints(k0, k1)
            T = np.empty_like(I)

            def seg_t(k, c0, c1, I=I, T=T, k0=k0):
                Ax = I[k, :, :, c0:c1].transpose(2, 1, 0)
                D = self._denom(k0 + k, c0, c1)
                if singlet:
                    T[k, c0:c1] = (2.0 * I[k, c0:c1] - Ax) / D
                else:
                    T[k, c0:c1] = -Ax / D

            self._rows(kt, seg_t)
            TW[rows] = T.reshape(kt * nv, -1) @ W
            del T, I
        reduce_sum(TW, self.comm)
        out -= 0.5 * (self._k_apply(TW[:, :m]) + TW[:, m:])
        if omegas is None:
            return out, norms
        # the doubles, one vector at a time: one dressed factor is held
        omegas = np.broadcast_to(np.asarray(omegas, float), (m,))
        parts = []
        for j in range(m):
            Bt = self._dressed(U[:, j])
            for t, (k0, k1) in self.my_tiles:
                img, nrm = self._doubles_tile(k0, k1, Bt, omegas[j])
                parts.append(((t, j), np.concatenate([img, [nrm]])))
            del Bt
        # per column, the tiles in tile order
        for j in range(m):
            mine = [(t, a) for (t, jj), a in parts if jj == j]
            tot = ordered_sum(mine, self.comm, shape=(self.n + 1,),
                              onto=np.zeros(self.n + 1))
            out[:, j] += tot[:-1]
            norms[j] = tot[-1]
        return out, norms

    def _doubles_tile(self, k0, k1, Bt, omega):
        """One tile of the doubles: (its folded image (no nv,), its |Y|^2)."""
        nv, no, L = self.nv, self.no, self.f.L
        kt = k1 - k0
        rows = slice(k0 * nv, k1 * nv)
        H = (Bt[rows] @ L.T).reshape(kt, nv, no, nv)
        HT = (L[rows] @ Bt.T).reshape(kt, nv, no, nv)
        singlet = self.spin == 'singlet'
        if singlet:
            def seg_g(k, c0, c1):
                H[k, c0:c1] += HT[k, c0:c1]
            self._rows(kt, seg_g)
            R = HT                                     # HT is consumed

            def seg_r(k, c0, c1):
                G = H[k, c0:c1]
                Gx = H[k, :, :, c0:c1].transpose(2, 1, 0)
                w = 1.0 / (omega - self._denom(k0 + k, c0, c1))
                R[k, c0:c1] = w * (2.0 * G - Gx)
                return float(np.sum(w * w * G * (G - 0.5 * Gx)))
        else:
            R = np.empty_like(H)

            def seg_r(k, c0, c1):
                Hb, Tb = H[k, c0:c1], HT[k, c0:c1]
                Sx = (H[k, :, :, c0:c1] + HT[k, :, :, c0:c1]).transpose(2, 1, 0)
                w = 1.0 / (omega - self._denom(k0 + k, c0, c1))
                R[k, c0:c1] = w * (2.0 * Hb - Sx)
                A = Hb - Tb
                S = Hb + Tb - Sx
                return float(np.sum(w * w * (0.5 * A * A + 0.25 * S * S)))
        nrm = sum(self._rows(kt, seg_r))
        Gam = R.reshape(kt * nv, -1) @ L
        del H, HT, R
        return self._back(Gam, k0, k1), nrm

    def doubles_dot(self, ur, wr, us, ws):
        """sum_K Y_K(ur, wr) Y_K(us, ws), the doubles images' overlap."""
        if self.level == 'adc1':
            return 0.0
        nv, no, L = self.nv, self.no, self.f.L
        same = ur is us or np.array_equal(ur, us)
        Br = self._dressed(ur)
        Bs = Br if same else self._dressed(us)
        singlet = self.spin == 'singlet'
        parts = []
        for t, (k0, k1) in self.my_tiles:
            kt = k1 - k0
            rows = slice(k0 * nv, k1 * nv)
            Hr = (Br[rows] @ L.T).reshape(kt, nv, no, nv)
            Tr = (L[rows] @ Br.T).reshape(kt, nv, no, nv)
            if same:
                Hs, Ts = Hr, Tr
            else:
                Hs = (Bs[rows] @ L.T).reshape(kt, nv, no, nv)
                Ts = (L[rows] @ Bs.T).reshape(kt, nv, no, nv)

            def seg(k, c0, c1):
                D = self._denom(k0 + k, c0, c1)
                w2 = 1.0 / ((wr - D) * (ws - D))
                hr, tr, hs, ts = (X[k, c0:c1] for X in (Hr, Tr, Hs, Ts))
                xr = (Hr[k, :, :, c0:c1] + Tr[k, :, :, c0:c1]).transpose(2, 1, 0)
                xs = (Hs[k, :, :, c0:c1] + Ts[k, :, :, c0:c1]).transpose(2, 1, 0)
                if singlet:
                    gr, gs = hr + tr, hs + ts
                    return float(np.sum(w2 * gr * (gs - 0.5 * xs)))
                return float(np.sum(w2 * (0.5 * (hr - tr) * (hs - ts)
                                           + 0.25 * (hr + tr - xr) * (hs + ts - xs))))

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

    def doubles_dot(self, ur, wr, us, ws):
        X = np.column_stack([np.concatenate([np.ravel(ur), [wr, 1.0]]),
                             np.concatenate([np.ravel(us), [ws, 1.0]])])
        return float(self.act(X)[0, 0])

    def pieces(self):
        p = self.op.pieces()
        p['operator'] = self
        return p


def drive(op, solve):
    """solve(operator) with `op` distributed over its communicator: rank 0
    runs `solve` on a stand-in whose products every rank computes together,
    the other ranks serve them; rank 0's return value (keep it to arrays and
    lists: it is pickled) comes back on every rank. Serially, solve(op)."""
    from src.Base.utils.krylov import root_driven_solve
    comm = op.comm
    if comm is None or comm.Get_size() == 1:
        return solve(op)
    n = op.n

    def collective(X):
        if X[n + 1, 0] == 0.0:
            om = X[n]
            AU, nrm = op.apply(X[:n], None if np.isnan(om[0]) else om)
            return np.vstack([AU, nrm[None, :]])
        return np.array([[op.doubles_dot(X[:n, 0], X[n, 0], X[:n, 1], X[n, 1])]])

    return root_driven_solve(lambda act: solve(_RootDriven(op, act)),
                             collective, rows=n + 2, comm=comm)
