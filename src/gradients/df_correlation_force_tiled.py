"""Nuclear force of a density-fitted correlated energy on an RHF reference,
tiled and distributed: the production route of src.SingleReference
.df_correlation_force (the dense reference, whose formulas are used here
unchanged), for molecules where no MO factor, two-particle density or
derivative-integral tensor fits whole.

The factored densities. The correlation fit enters through the derivatives
Gamma^P_pq = dE_c/dB^P_pq of the folded ADC(2) operator's factors
(src.SingleReference.ADC.eeADC.ee_ri_fold, B = X J, X = L^-1, V = L L^T),
nonzero in three active blocks, each held as an (n_occ n, naux) array:

    Gamma[o, v] = GL,   Gamma[o, o] = Goo,   Gamma^P_ab = sum_k y_ka X^P_kb

(the virtual block is the singles vector y times X; MP2 has GL alone). Gamma
itself, nv^2 naux for the vv block, is never formed.

The orbital gradient. YG_rp = sum_qP Gs2^P_pq B^P_rq (Gs2 = Gamma + Gamma^T)
needs B^P_rq of every row r against the active columns: L, B_oo and the core
factors are held whole, B_vv in the fold's auxiliary chunks; each chunk's
owner adds its chunk's share, summed in chunk order. The Z-vector and W are
df_correlation_force.relax_gradient's.

The correlation-fit skeleton, 2 sum Gamma3 dJ - sum Gamma2 dV with Gamma3 =
(1/4) (Gamma + Gamma^T) X in the raw auxiliary basis and Gamma2 = K^T Gamma3:

    Gamma2 = X^T N X,   N_SP = (1/2) sum_pq B^S_pq Gamma^P_pq

(N from the chunks, each owner its rows), and Gamma3 of one AO shell block mu
against one auxiliary chunk from the X-transformed factored arrays,

    Gamma3^Q[mu, :] = (1/4) [Lo Co^T + Lv Cv^T + Ly Cy^T],   Cy = Cv y^T
    Lo = Co[mu] (Goo + Goo^T)^Q + Cv[mu] GL^Q^T,  Lv = Co[mu] GL^Q + Cy[mu] X^Q,
    Ly = Cv[mu] X^Q^T,

contracted with the derivative integrals of that block alone.

The JK-fit skeleton, sum Dc G'[D_HF], D_HF = 2 C_o C_o^T: Gamma3 = (1/2)(Dc
c_P + D_HF c_D) - (1/4)(ex + ex^T), c_D = V^-1 (J . Dc), ex^Q = 2 (Dc W^Q)
C_o^T with W^Q = sum_R J^R C_o V^-1_RQ, and Gamma2 = (1/2)(c_D c_P^T + c_P
c_D^T) - W . (Dc W)^T. W and Dc W (nao nocc naux each) are the only
intermediates held whole, as in any density-fitted exchange gradient.

Distribution. Every collective runs on the fold's communicator. A sum over
occupied tiles, auxiliary chunks or AO shell blocks is taken in tile, chunk or
block order (ordered_sum, ordered_chain_sum); arrays whose rows one owner
computes travel verbatim (allgather_ranges); the replicated small pieces (the
Z-vector, the metric factors, the SCF force) are rank 0's (lockstep). At one
tile size and one chunk width the force has the same bits on every rank and at
every rank count.
"""
import numpy as np
import scipy.linalg
from pyscf import df as pyscf_df
from pyscf.gto.moleintor import getints, make_cintopt

from src.Base.numerics.constants import (BYTES_PER_GB, DF_FORCE_AUX_CHUNK, DF_FORCE_BLOCK_ARRAYS,
                                         DF_FORCE_BLOCK_GB, DF_FORCE_MP2_TILE_ARRAYS,
                                         DF_FORCE_PIECES_TILE_ARRAYS, EE_RI_AMPLITUDE_CACHE_GB,
                                         EE_RI_TILE_GB, MB_PER_GB)
from src.Base.utils.parallel.mpi_grid import (allgather_ranges, broadcast, current_comm,
                                              lockstep, lockstep_mean_field, ordered_chain_sum,
                                              ordered_sum, reduce_sum)
from src.gradients import df_correlation_force as dcf
from src.SingleReference.common import df_amplitudes as dfa
from src.SingleReference.ADC.eeADC import ee_relaxed_density as erd
from src.SingleReference.ADC.eeADC import ee_ri_fold
from src.SingleReference.ADC.eeADC.ee_ri_density import Pass, tile_ranges
from src.gradients.grad_engine import one_electron_skeleton



class FactoredDerivatives:
    """dE_c/df and dE_c/dB of a correlated energy on the fold's factors,
    active orbitals: g_oo (no, no), g_vv (nv, nv); GL (no nv, naux) =
    dE/dB^P_ia; Goo (no, no, naux) = dE/dB^P_ij or None; dE/dB^P_ab = sum_k
    y_ka X[(k, b), P] with X (no nv, naux) and y (no, nv), or None."""

    def __init__(self, g_oo, g_vv, GL, Goo=None, X=None, y=None):
        self.g_oo, self.g_vv, self.GL = g_oo, g_vv, GL
        self.Goo, self.X, self.y = Goo, X, y

    def plus(self, t1, pc):
        """self + t1 x the ADC(2) Pieces pc (ee_relaxed_density); self is the
        MP2 part, which has no oo or vv derivative blocks of its own."""
        assert self.Goo is None and self.X is None
        return FactoredDerivatives(self.g_oo + t1 * pc.g_oo, self.g_vv + t1 * pc.g_vv,
                                   self.GL + t1 * pc.GL, t1 * pc.Goo, t1 * pc.Xvv,
                                   pc.y)


# ----------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------

def estimate(mol, nfrozen, auxbasis, jk_auxbasis, ranks=1, tile_gb=EE_RI_TILE_GB,
             excited=True, block_gb=DF_FORCE_BLOCK_GB):
    """GB one process of `ranks` holds at the peak of the tiled force, with
    the parts, and the floating-point work of the force beyond the root
    (flop). Held throughout: the fold's factors and amplitude cache, the core
    factors and the SCF's JK tensor (pyscf keeps it in memory when it fits);
    on top, the largest of the ADC(2) pieces, the force's own arrays (the
    factored densities, the JK exchange intermediates, one derivative unit)
    and the correlation fit's AO tensor while a factor is built."""
    nocc = mol.nelectron // 2
    no, nv, nao = nocc - nfrozen, mol.nao - nocc, mol.nao
    nmo = mol.nao
    naux = pyscf_df.addons.make_auxmol(mol, auxbasis).nao
    naux_jk = pyscf_df.addons.make_auxmol(mol, jk_auxbasis).nao
    gb = 8.0 / BYTES_PER_GB
    npair = nao * (nao + 1) / 2
    A = no * nv * naux * gb
    factors = (no * nv * naux + no * no * naux + nv * nv * naux / ranks) * gb
    amplitudes = min(EE_RI_AMPLITUDE_CACHE_GB, (no * nv) ** 2 * gb / ranks) if excited else 0.0
    core = naux * nfrozen * (nmo - nfrozen) * gb
    cderi_jk, cderi_ri = npair * naux_jk * gb, npair * naux * gb
    per_k = (DF_FORCE_PIECES_TILE_ARRAYS if excited
              else DF_FORCE_MP2_TILE_ARRAYS) * nv * no * nv * gb
    tile = per_k * max(1, min(-(-no // ranks), int(tile_gb // per_k)))
    # GL, Xvv, the running GL sum, one tile's term, the dressed factor twice,
    # rho_o L, B_oo-shaped Goo; MP2 alone: GL
    pieces = (7.0 * A + no * no * naux * gb if excited else A) + tile
    # GL, X and their raw-basis copies; the JK fit's W and Dc W
    exchange = 2.0 * nao * nocc * naux_jk * gb
    engine = (4.0 if excited else 2.0) * A + exchange + block_gb
    held = factors + amplitudes + core + cderi_jk
    peak = held + max(pieces, engine, cderi_ri)
    flop = (2.0 * nao * nao * (nmo + no) * naux                 # Gamma3 blocks
            + 6.0 * no * nv * naux * naux                       # X transforms, N
            + 8.0 * nao * nao * nocc * naux_jk                  # J C_o, Dc W, ex
            + 4.0 * nao * nocc * naux_jk * naux_jk              # W, Gamma2_jk
            + (16.0 if excited else 4.0) * (no * nv) ** 2 * naux)   # the pieces
    return dict(nocc_active=no, nvir=nv, nao=nao, naux=naux, naux_jk=naux_jk,
                ranks=ranks, factors_gb=factors, amplitudes_gb=amplitudes,
                core_gb=core, cderi_jk_gb=cderi_jk, cderi_ri_gb=cderi_ri,
                pieces_gb=pieces, tile_gb=tile, engine_gb=engine,
                exchange_gb=exchange, peak_gb=peak, flop=flop)


def refuse_over(est, max_memory_mb):
    """Raise MemoryError when the estimate's peak exceeds max_memory (MB)."""
    if est['peak_gb'] * MB_PER_GB > max_memory_mb:
        raise MemoryError(
            f"the tiled force needs about {est['peak_gb'] * MB_PER_GB:.1f} MB per rank "
            f"({est['ranks']} ranks), max_memory {max_memory_mb:.1f} MB: "
            f"factors {est['factors_gb']:.1f}, amplitudes "
            f"{est['amplitudes_gb']:.1f}, SCF JK tensor {est['cderi_jk_gb']:.1f}, "
            f"pieces {est['pieces_gb']:.1f}, engine {est['engine_gb']:.1f} GB")


# ----------------------------------------------------------------------
# MP2
# ----------------------------------------------------------------------

def mp2_pieces(op, tile_gb=EE_RI_TILE_GB):
    """(e_mp2, FactoredDerivatives) of closed-shell DF-MP2 on the fold's
    factors (df_correlation_force.mp2_pieces one occupied tile at a time);
    a collective over op's ranks."""
    f, no, nv, naux = op.f, op.no, op.nv, op.f.naux
    comm = op.comm
    ranks = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    L = f.L
    tiles = tile_ranges(no, 8.0 * nv * no * nv * DF_FORCE_MP2_TILE_ARRAYS, tile_gb, ranks)
    GL = np.zeros((no * nv, naux))
    acc = Pass(op, {'e': (1,), 'g_oo': (no, no), 'g_vv': (nv, nv)})
    for t, (k0, k1) in enumerate(tiles):
        if t % ranks != rank:
            continue
        I = op._ints(k0, k1)
        D = dfa.doubles_denominator(f.eo, f.ev, f.eo[k0:k1])
        tau = (2.0 * I - I.transpose(dfa.PAIR)) / D
        oo, vv = dfa.tile_amplitude_fock(-I / D, tau)
        e = -float(np.sum(I * tau))
        del I, D
        GL[k0 * nv:k1 * nv] = -4.0 * (tau.reshape((k1 - k0) * nv, -1) @ L)
        acc.add(t, {'e': [e], 'g_oo': oo, 'g_vv': vv})
    # every row of GL from one tile, the rest zeros: exact in any order
    reduce_sum(GL, comm)
    tot = acc.total()
    return float(tot['e'][0]), FactoredDerivatives(erd.symmetrize(tot['g_oo']),
                                                   erd.symmetrize(tot['g_vv']), GL)


# ----------------------------------------------------------------------
# the orbital gradient and the correlation fit's metric density
# ----------------------------------------------------------------------

def orbital_gradient(op, d, nf, Bc=None):
    """YG (nmo, nmo), YG_rp = sum_qP Gs2^P_pq B^P_rq over every row r
    (frozen core, active occupied, virtual), from the fold's factors chunk by
    chunk; Bc the core factors (ee_relaxed_density.core_factors) when nf."""
    f, no, nv = op.f, op.no, op.nv
    nmo = nf + no + nv
    c, o, v = slice(0, nf), slice(nf, nf + no), slice(nf + no, nmo)
    GL3 = d.GL.reshape(no, nv, -1)
    X3 = None if d.X is None else d.X.reshape(no, nv, -1)
    y = d.y
    terms = {}
    for ch in op.my_chunks:
        p0, p1 = op.chunks[ch]
        Lc = f.LT[p0:p1]                                      # B^P_kb [P, k, b]
        Bo = f.BooP[p0:p1]                                    # B^P_kj
        Bv = f.bvv(ch, p0, p1).transpose(1, 0, 2)             # B^P_db [P, d, b]
        G = np.ascontiguousarray(GL3[:, :, p0:p1].transpose(2, 0, 1))
        rows = [(o, Lc, Bo), (v, Bv, Lc.transpose(0, 2, 1))]
        if nf:
            # B^P_c(virtual), B^P_c(occupied)
            rows.append((c, Bc[p0:p1, :, no:], Bc[p0:p1, :, :no]))
        Y = np.zeros((nmo, nmo))
        S = None
        if d.Goo is not None:
            Gc = d.Goo[:, :, p0:p1]
            S = np.ascontiguousarray((Gc + Gc.transpose(1, 0, 2)).transpose(2, 0, 1))
        for r, Brv, Bro in rows:
            # p = i: sum_b GL_ib B_rb + sum_j Gs_ij B_rj
            Y[r, o] += np.einsum('Prb,Pib->ri', Brv, G, optimize=True)
            # p = a: sum_j GL_ja B_rj
            Y[r, v] += np.einsum('Prj,Pja->ra', Bro, G, optimize=True)
            if S is not None:
                Y[r, o] += np.einsum('Prj,Pij->ri', Bro, S, optimize=True)
            if X3 is not None:
                # p = a: sum_b Gs_ab B_rb, Gs_ab = sum_m (y_ma X_mb + X_ma y_mb)
                Xc = np.ascontiguousarray(X3[:, :, p0:p1].transpose(2, 0, 1))
                XB = np.einsum('Pmb,Prb->mr', Xc, Brv, optimize=True)
                yB = np.einsum('mb,Prb->Pmr', y, Brv, optimize=True)
                Y[r, v] += XB.T @ y + np.einsum('Pmr,Pma->ra', yB, Xc, optimize=True)
        terms[ch] = Y
    return ordered_chain_sum(terms, op.owners, op.comm, np.zeros((nmo, nmo)))


def metric_density(op, d, Xw):
    """Gamma2 (naux, naux) of the correlation fit, raw basis, symmetrized:
    X^T N X, N_SP = (1/2) sum_pq B^S_pq Gamma^P_pq, each chunk's rows from its
    owner."""
    f, no, nv, naux = op.f, op.no, op.nv, op.f.naux
    N = np.zeros((naux, naux))
    for ch in op.my_chunks:
        p0, p1 = op.chunks[ch]
        pc = p1 - p0
        n = f.LT[p0:p1].reshape(pc, -1) @ d.GL
        if d.Goo is not None:
            n += f.BooP[p0:p1].reshape(pc, -1) @ d.Goo.reshape(no * no, naux)
        if d.X is not None:
            # sum_a y_ka B^S_ab as [S, (k, b)]
            yB = (d.y @ f.bvv(ch, p0, p1).reshape(nv, -1)).reshape(no, pc, nv)
            n += yB.transpose(1, 0, 2).reshape(pc, -1) @ d.X
        N[p0:p1] = 0.5 * n
    allgather_ranges(N, op.chunk_ranges, op.comm)
    G2 = Xw.T @ N @ Xw
    return 0.5 * (G2 + G2.T)


def raw_basis(op, A, Xw):
    """(A Xw)^T (naux, rows), auxiliary-major, each chunk of columns from its
    owner: a factored Gamma (rows, naux) taken to the raw auxiliary basis."""
    naux = Xw.shape[0]
    out = np.zeros((naux, A.shape[0]))
    for ch in op.my_chunks:
        p0, p1 = op.chunks[ch]
        # X = L^-1 is lower triangular
        out[p0:p1] = (A[:, p0:] @ Xw[p0:, p0:p1]).T
    return allgather_ranges(out, op.chunk_ranges, op.comm)


# ----------------------------------------------------------------------
# integrals in blocks
# ----------------------------------------------------------------------

def _int3c(mol, auxmol, intor):
    """get(shls_slice) of (mu nu|P)-type integrals over shell ranges of mol,
    mol, auxmol, as [comp,] [P, nu, mu] (C-contiguous)."""
    pmol = mol + auxmol
    name = mol._add_suffix(intor)
    opt = make_cintopt(mol._atm, mol._bas, mol._env, name)
    nbas = mol.nbas

    def get(s0, s1, a0, a1):
        out = getints(name, pmol._atm, pmol._bas, pmol._env,
                      (s0, s1, 0, nbas, nbas + a0, nbas + a1), aosym='s1',
                      cintopt=opt, comp=3 if intor.endswith(('_ip1', '_ip2')) else 1)
        return out.transpose(0, 3, 2, 1) if out.ndim == 4 else out.transpose(2, 1, 0)
    return get


def shell_blocks(ao_loc, width):
    """(s0, s1, p0, p1) runs of consecutive shells of at most `width`
    functions each (at least one shell)."""
    out, s0 = [], 0
    nsh = len(ao_loc) - 1
    while s0 < nsh:
        s1 = s0 + 1
        while s1 < nsh and ao_loc[s1 + 1] - ao_loc[s0] <= width:
            s1 += 1
        out.append((s0, s1, int(ao_loc[s0]), int(ao_loc[s1])))
        s0 = s1
    return out


def _owners(n, size):
    """Consecutive runs of n units over size ranks."""
    return [k * size // max(n, 1) for k in range(n)]


def _ranges(blocks, owners, size):
    return [[(b[2], b[3]) for b, o in zip(blocks, owners) if o == r]
            for r in range(size)]


def on_root(fn, shapes, comm):
    """The arrays fn() returns, computed on rank 0 alone and broadcast
    (lockstep); shapes theirs. For the replicated pyscf pieces: once instead
    of on every rank, and never concurrently on one molecule (pyscf's
    one-electron derivatives move its rinv origin). A raise in fn() on rank 0
    (an unconverged CPHF) is broadcast and raised on every rank, so the other
    ranks do not wait in the lockstep for ever."""
    rank = 0 if comm is None else comm.Get_rank()
    if rank != 0:
        failed = broadcast(None, comm)
        if failed is not None:
            raise failed
        return lockstep(tuple(np.zeros(s) for s in shapes), comm)
    try:
        out = tuple(np.ascontiguousarray(a, dtype=float) for a in fn())
    except Exception as exc:
        broadcast(exc, comm)
        raise
    broadcast(None, comm)
    return lockstep(out, comm)


# ----------------------------------------------------------------------
# the JK fit's exchange intermediates
# ----------------------------------------------------------------------

def jk_intermediates(mol, auxmol, Dc, Co, comm, width=DF_FORCE_AUX_CHUNK):
    """(c_D, c_P, Dc W (naux, nao, nocc), Gamma2) of sum Dc G'[D_HF] in the
    JK fit (module docstring), Co every occupied orbital; a collective."""
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    nao, nocc, naux = mol.nao, Co.shape[1], auxmol.nao
    Dhf = 2.0 * Co @ Co.T
    chunks = shell_blocks(auxmol.ao_loc, width)
    owners = _owners(len(chunks), size)
    ranges = _ranges(chunks, owners, size)
    mine = [b for b, o in zip(chunks, owners) if o == rank]
    get = _int3c(mol, auxmol, 'int3c2e')
    JCo = np.zeros((naux, nao, nocc))
    jd = np.zeros((naux, 2))
    for a0, a1, q0, q1 in mine:
        J = get(0, mol.nbas, a0, a1)                          # J^Q [Q, nu, mu]
        JCo[q0:q1] = J @ Co
        J2 = J.reshape(q1 - q0, -1)
        jd[q0:q1, 0] = J2 @ Dc.ravel()
        jd[q0:q1, 1] = J2 @ Dhf.ravel()
        del J, J2
    allgather_ranges(JCo, ranges, comm)
    allgather_ranges(jd, ranges, comm)
    V = auxmol.intor('int2c2e')
    Vinv = scipy.linalg.cho_solve(scipy.linalg.cho_factor(V), np.eye(naux))
    lockstep(Vinv, comm)
    cD, cP = Vinv @ jd[:, 0], Vinv @ jd[:, 1]
    W = np.zeros_like(JCo)
    for a0, a1, q0, q1 in mine:
        W[q0:q1] = (Vinv[q0:q1] @ JCo.reshape(naux, -1)).reshape(q1 - q0, nao, nocc)
    del JCo
    allgather_ranges(W, ranges, comm)
    DcW = np.zeros_like(W)
    for a0, a1, q0, q1 in mine:
        DcW[q0:q1] = np.matmul(Dc, W[q0:q1])
    allgather_ranges(DcW, ranges, comm)
    G2 = np.zeros((naux, naux))
    for a0, a1, q0, q1 in mine:
        G2[q0:q1] = (0.5 * (np.outer(cD[q0:q1], cP) + np.outer(cP[q0:q1], cD))
                     - W[q0:q1].reshape(q1 - q0, -1) @ DcW.reshape(naux, -1).T)
    allgather_ranges(G2, ranges, comm)
    return cD, cP, DcW, 0.5 * (G2 + G2.T)


# ----------------------------------------------------------------------
# the derivative-integral pass
# ----------------------------------------------------------------------

class _RIFit:
    """Gamma3 blocks of the correlation fit from the raw factored arrays."""

    def __init__(self, auxmol, C, nf, no, d, Xw, op):
        nocc = nf + no
        self.auxmol = auxmol
        naux = Xw.shape[0]
        nv = C.shape[1] - nocc
        self.Co, self.Cv = C[:, nf:nocc], C[:, nocc:]
        self.GL = raw_basis(op, d.GL, Xw).reshape(naux, no, nv)
        self.Goo = None
        if d.Goo is not None:
            G = raw_basis(op, d.Goo.reshape(no * no, -1), Xw).reshape(naux, no, no)
            self.Goo = G + G.transpose(0, 2, 1)
        self.X = None
        parts = [self.Co, self.Cv]
        if d.X is not None:
            self.X = raw_basis(op, d.X, Xw).reshape(naux, no, nv)
            self.Cy = self.Cv @ d.y.T
            parts.append(self.Cy)
        self.Cbig = np.ascontiguousarray(np.hstack(parts))

    def block(self, m0, m1, q0, q1):
        """Gamma3 [Q, nu, mu] of the AO rows m0..m1-1 and chunk q0..q1-1."""
        Com, Cvm = self.Co[m0:m1], self.Cv[m0:m1]
        GL = self.GL[q0:q1]
        Lo = np.matmul(Cvm, GL.transpose(0, 2, 1))
        if self.Goo is not None:
            Lo += np.matmul(Com, self.Goo[q0:q1])
        Lv = np.matmul(Com, GL)
        parts = [Lo, Lv]
        if self.X is not None:
            Xq = self.X[q0:q1]
            Lv += np.matmul(self.Cy[m0:m1], Xq)
            parts.append(np.matmul(Cvm, Xq.transpose(0, 2, 1)))
        Lcat = np.concatenate(parts, axis=2)
        return 0.25 * np.matmul(self.Cbig, Lcat.transpose(0, 2, 1))


class _JKFit:
    """Gamma3 blocks of sum Dc G'[D_HF] in the JK fit (module docstring)."""

    def __init__(self, auxmol, Dc, Co, cD, cP, DcW):
        self.auxmol = auxmol
        self.Dc, self.Dhf = Dc, 2.0 * Co @ Co.T
        self.Co, self.cD, self.cP, self.DcW = Co, cD, cP, DcW

    def block(self, m0, m1, q0, q1):
        DcW = self.DcW[q0:q1]
        G = (0.5 * self.cP[q0:q1, None, None] * self.Dc[None, :, m0:m1]
             + 0.5 * self.cD[q0:q1, None, None] * self.Dhf[None, :, m0:m1])
        # (ex + ex^T)[nu, mu] / 4, ex = 2 Dc W C_o^T
        G -= 0.5 * np.matmul(self.Co, DcW[:, m0:m1].transpose(0, 2, 1))
        G -= 0.5 * np.matmul(DcW, self.Co[m0:m1].T)
        return G


def skeleton_pass(mol, fits, comm, block_gb=DF_FORCE_BLOCK_GB,
                  width=DF_FORCE_AUX_CHUNK):
    """sum over fits of 2 sum Gamma3 dJ (AO and auxiliary centres) in AO
    shell blocks against auxiliary chunks, the blocks dealt over the ranks and
    summed in block order; (natm, 3). The metric term is the caller's."""
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    nao, natm = mol.nao, mol.natm
    ao_atom = np.zeros(nao, dtype=int)
    for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        ao_atom[p0:p1] = a
    nmu = max(1, int(block_gb * BYTES_PER_GB
                     / (8.0 * DF_FORCE_BLOCK_ARRAYS * nao * width)))
    blocks = shell_blocks(mol.ao_loc, nmu)
    owners = _owners(len(blocks), size)
    prepared = []
    for fit in fits:
        aux = fit.auxmol
        aux_atom = np.zeros(aux.nao, dtype=int)
        for a, (_, _, p0, p1) in enumerate(aux.aoslice_by_atom()):
            aux_atom[p0:p1] = a
        prepared.append((fit, shell_blocks(aux.ao_loc, width), aux_atom,
                         _int3c(mol, aux, 'int3c2e_ip1'),
                         _int3c(mol, aux, 'int3c2e_ip2')))
    addends = []
    for b, (s0, s1, m0, m1) in enumerate(blocks):
        if owners[b] != rank:
            continue
        F = np.zeros((natm, 3))
        for fit, chunks, aux_atom, ip1, ip2 in prepared:
            t_ao = np.zeros((3, m1 - m0))
            t_aux = np.zeros((3, fit.auxmol.nao))
            for a0, a1, q0, q1 in chunks:
                G = fit.block(m0, m1, q0, q1)                 # [Q, nu, mu]
                G2d = G.reshape(-1, m1 - m0)
                d1 = ip1(s0, s1, a0, a1)
                for x in range(3):
                    t_ao[x] -= 2.0 * np.einsum('km,km->m', d1[x].reshape(-1, m1 - m0), G2d)
                del d1
                d2 = ip2(s0, s1, a0, a1)
                Gq = G.reshape(q1 - q0, -1)
                for x in range(3):
                    t_aux[x, q0:q1] -= np.einsum('qk,qk->q', d2[x].reshape(q1 - q0, -1), Gq)
                del d2, G, G2d, Gq
            for x in range(3):
                np.add.at(F[:, x], ao_atom[m0:m1], 2.0 * t_ao[x])
                np.add.at(F[:, x], aux_atom, 2.0 * t_aux[x])
        addends.append((b, F))
    return ordered_sum(addends, comm, shape=(natm, 3), onto=np.zeros((natm, 3)))


def metric_force(auxmol, G2):
    """-sum Gamma2 dV on the auxiliary centres, (natm, 3)."""
    v1 = auxmol.intor('int2c2e_ip1', comp=3)
    t_v = 2.0 * np.einsum('xPQ,PQ->xP', v1, G2, optimize=True)
    out = np.zeros((auxmol.natm, 3))
    for a, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        out[a] = t_v[:, q0:q1].sum(axis=1)
    return out


# ----------------------------------------------------------------------
# the force
# ----------------------------------------------------------------------

def correlation_force(mol, mf, op, auxmol, d, nf, Bc=None, Y_extra=None,
                      block_gb=DF_FORCE_BLOCK_GB, width=DF_FORCE_AUX_CHUNK):
    """(force (natm, 3) of E_c, gamma (MO, relaxed), max |Y - Y^T|): the
    tiled df_correlation_force.correlation_force, a collective over op's
    ranks; auxmol the correlation fit of op's factors."""
    comm = op.comm
    C = np.asarray(mf.mo_coeff)
    nocc = mol.nelectron // 2
    nmo, no = C.shape[1], op.no
    if auxmol.nao != op.f.naux:
        raise ValueError(f'the factors hold {op.f.naux} auxiliary functions, the fit '
                         f'{auxmol.nao}: a linearly dependent metric was pruned')
    if nmo != nf + op.no + op.nv:
        raise ValueError('the operator does not span the mean field\'s orbitals')
    YG = orbital_gradient(op, d, nf, Bc)
    if Y_extra is not None:
        YG = YG + Y_extra
    gamma0 = np.zeros((nmo, nmo))
    o, v = slice(nf, nocc), slice(nocc, nmo)
    gamma0[o, o], gamma0[v, v] = d.g_oo, d.g_vv
    gamma, Y = on_root(lambda: dcf.relax_gradient(mf, mol, YG, gamma0, nf),
                       [(nmo, nmo)] * 2, comm)
    asym = float(np.max(np.abs(Y - Y.T)))
    W = 0.25 * (Y + Y.T)
    gamma_ao, W_ao = C @ gamma @ C.T, C @ W @ C.T
    f1, = on_root(lambda: one_electron_skeleton(mol, mf, [gamma_ao], [-W_ao]),
                  [(mol.natm, 3)], comm)
    # the correlation fit
    _, Xw = dcf.whitening(auxmol)
    lockstep(Xw, comm)
    G2ri = metric_density(op, d, Xw)
    ri = _RIFit(auxmol, C, nf, no, d, Xw, op)
    # the SCF's JK fit
    auxjk = pyscf_df.addons.make_auxmol(mol, mf.with_df.auxbasis)
    Co = np.ascontiguousarray(C[:, :nocc])
    cD, cP, DcW, G2jk = jk_intermediates(mol, auxjk, gamma_ao, Co, comm, width)
    jk = _JKFit(auxjk, gamma_ao, Co, cD, cP, DcW)
    f2 = skeleton_pass(mol, [ri, jk], comm, block_gb, width)
    f2 += metric_force(auxmol, G2ri) + metric_force(auxjk, G2jk)
    return lockstep(f1 + f2, comm), gamma, asym


def ground_operator(mol, mf, frozen, auxbasis, tile_gb=EE_RI_TILE_GB, comm=None,
                    aux_chunk=None):
    """The fold's factors, tiles and chunks without a root (ADC(1) setup):
    what the tiled MP2 force runs on."""
    f = ee_ri_fold.bare_factors(mol, mf, frozen=frozen, auxbasis=auxbasis, comm=comm,
                                aux_chunk=aux_chunk)
    return ee_ri_fold.RIFoldADC2(f, 'singlet', level='adc1', tile_gb=tile_gb,
                                 comm=comm, aux_chunk=aux_chunk)


def prepare(mf, mol, frozen, auxbasis, excited, tile_gb=EE_RI_TILE_GB,
            block_gb=DF_FORCE_BLOCK_GB, max_memory=None, comm=None):
    """(comm, estimate) of a tiled force over comm's ranks (default: the current
    communicator), refused up front (MemoryError) when the per-rank estimate
    exceeds max_memory (MB; default mol.max_memory); mf is then rank 0's on
    every rank."""
    if auxbasis is None:
        raise ValueError('the tiled force needs the correlation fit by name (auxbasis)')
    comm = current_comm() if comm is None else comm
    ranks = 1 if comm is None else comm.Get_size()
    est = estimate(mol, frozen, auxbasis, mf.with_df.auxbasis, ranks, tile_gb,
                   excited=excited, block_gb=block_gb)
    refuse_over(est, mol.max_memory if max_memory is None else max_memory)
    # every rank's partials are of rank 0's orbitals
    lockstep_mean_field(mf, comm)
    return comm, est


def force(mf, mol, op, frozen, auxbasis, excited=None, tile_gb=EE_RI_TILE_GB,
          block_gb=DF_FORCE_BLOCK_GB, width=DF_FORCE_AUX_CHUNK):
    """df_correlation_force.mp2_force on the fold's tiles and ranks, a
    collective over op's communicator with the same bits on every rank;
    excited(d, C, nocc, nmo) -> (d plus the excitation's factored derivatives,
    Y_extra, extra)."""
    C = np.asarray(mf.mo_coeff)
    nmo, nocc = C.shape[1], mol.nelectron // 2
    e2, d = mp2_pieces(op, tile_gb)
    Y_extra = extra = None
    if excited is not None:
        d, Y_extra, extra = excited(d, C, nocc, nmo)
    Bc = erd.core_factors(mol, mf, frozen, auxbasis) if frozen else None
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis)
    fc, gam, asym = correlation_force(mol, mf, op, auxmol, d, frozen, Bc,
                                      Y_extra, block_gb, width)
    F = on_root(lambda: (dcf.hf_force(mf),), [(mol.natm, 3)], op.comm)[0] + fc
    return F, (e2, gam, asym, extra)


def mp2_force_tiled(mf, mol, frozen=0, auxbasis=None, tile_gb=EE_RI_TILE_GB,
                    max_memory=None, block_gb=DF_FORCE_BLOCK_GB,
                    width=DF_FORCE_AUX_CHUNK, comm=None, aux_chunk=None):
    """(dE/dR, (e_mp2, gamma, max |Y - Y^T|, estimate)) of E_HF + E_MP2 on the
    tiled route over comm's ranks (prepare)."""
    comm, est = prepare(mf, mol, frozen, auxbasis, False, tile_gb, block_gb,
                        max_memory, comm)
    op = ground_operator(mol, mf, frozen, auxbasis, tile_gb, comm, aux_chunk)
    F, (e2, gam, asym, _) = force(mf, mol, op, frozen, auxbasis, None, tile_gb,
                                  block_gb, width)
    return F, (e2, gam, asym, est)
