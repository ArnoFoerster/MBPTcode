"""Nuclear force of a density-fitted correlated energy on an RHF reference,
tiled and distributed: the production route of src.gradients
.df_correlation_force (the dense reference, whose formulas are used here
unchanged), for molecules where no MO factor, two-particle density or
derivative-integral tensor fits whole.

The factor provider. The force reads the RI factors of the correlation fit
(B = X J, X = L^-1, V = L L^T) through one object, which the method hands
over (`gradients.adc.fold_factors` for MP2 and ADC(2)): `f` the factors
(`L`, `LT`, `BooP`, `bvv(chunk, p0, p1)`, `eo`, `ev`, `naux`), `no` and `nv`,
`comm`, the auxiliary `chunks` with this rank's `my_chunks`, their `owners`
and `chunk_ranges`, `ints(k0, k1)` the (ia|jb) of an occupied tile, and
`core_factors(mol, mf, frozen, auxbasis)` the frozen core's.

The factored densities. The correlation fit enters through the derivatives
Gamma^P_pq = dE_c/dB^P_pq of the provider's factors, nonzero in three active
blocks, each held as an (n_occ n, naux) array:

    Gamma[o, v] = GL,   Gamma[o, o] = Goo,   Gamma^P_ab = sum_k y_ka X^P_kb

(the virtual block is the singles vector y times X; MP2 has GL alone). Gamma
itself, nv^2 naux for the vv block, is never formed.

The orbital gradient. YG_rp = sum_qP Gs2^P_pq B^P_rq (Gs2 = Gamma + Gamma^T)
needs B^P_rq of every row r against the active columns: L, B_oo and the core
factors are held whole, B_vv in the fold's auxiliary chunks; each chunk's
owner adds its chunk's share, summed in chunk order. The Z-vector and W are
df_correlation_force.relax_gradient's, solved on rank 0.

The correlation-fit skeleton, 2 sum Gamma3 dJ - sum Gamma2 dV with Gamma3 =
(1/4) (Gamma + Gamma^T) X in the raw auxiliary basis and Gamma2 = K^T Gamma3:

    Gamma2 = X^T N X,   N_SP = (1/2) sum_pq B^S_pq Gamma^P_pq

(N from the chunks, each owner its rows), and Gamma3 of one AO shell block mu
against one auxiliary chunk from the X-transformed factored arrays,

    Gamma3^Q[mu, :] = (1/4) [Lo Co^T + Lv Cv^T + Ly Cy^T],   Cy = Cv y^T
    Lo = Co[mu] (Goo + Goo^T)^Q + Cv[mu] GL^Q^T,  Lv = Co[mu] GL^Q + Cy[mu] X^Q,
    Ly = Cv[mu] X^Q^T,

contracted with the derivative integrals of that block alone.

The JK-fit skeleton, sum Dc G'[D_HF] with D_HF = 2 C_o C_o^T, is the fitted
Fock skeleton every gradient shares (`skeleton_tiles.fitted_fock_skeleton`,
the SCF's auxiliary basis cut into fixed tiles over the ranks).

Distribution. Every collective runs on the fold's communicator. A sum over
occupied tiles, auxiliary chunks or AO shell blocks is taken in tile, chunk or
block order (ordered_sum, ordered_chain_sum); arrays whose rows one owner
computes travel verbatim (allgather_ranges); the replicated small pieces (the
Z-vector, the metric factors, the SCF force) are rank 0's (lockstep). At one
tile size and one chunk width the force has the same bits on every rank and at
every rank count.
"""
import numpy as np
from pyscf import df as pyscf_df
from pyscf.gto.moleintor import getints, make_cintopt

from src.Base.numerics.constants import (BYTES_PER_GB, DF_FORCE_AUX_CHUNK, DF_FORCE_BLOCK_ARRAYS,
                                         DF_FORCE_BLOCK_GB, DF_FORCE_MP2_TILE_ARRAYS,
                                         DF_FORCE_PIECES_TILE_ARRAYS, EE_RI_AMPLITUDE_CACHE_GB,
                                         EE_RI_TILE_GB, MB_PER_GB)
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.memory import tile_width
from src.Base.utils.parallel.mpi_grid import (allgather_ranges, broadcast, current_comm,
                                              distributed, lockstep, lockstep_mean_field,
                                              ordered_chain_sum, ordered_sum, reduce_sum)
from src.gradients import df_correlation_force as dcf
from src.SingleReference.common import df_amplitudes as dfa
from src.gradients.grad_engine import one_electron_skeleton
from src.gradients.skeleton_tiles import fitted_fock_skeleton, shell_runs



class FactoredDerivatives:
    """dE_c/df and dE_c/dB of a correlated energy on the fold's factors,
    active orbitals: g_oo (no, no), g_vv (nv, nv); GL (no nv, naux) =
    dE/dB^P_ia; Goo (no, no, naux) = dE/dB^P_ij or None; dE/dB^P_ab = sum_k
    y_ka X[(k, b), P] with X (no nv, naux) and y (no, nv), or None."""

    def __init__(self, g_oo, g_vv, GL, Goo=None, X=None, y=None):
        self.g_oo, self.g_vv, self.GL = g_oo, g_vv, GL
        self.Goo, self.X, self.y = Goo, X, y

    def plus(self, t1, pc):
        """self + t1 x an excitation's factored pieces pc (its g_oo, g_vv, GL,
        Goo, Xvv and y; ADC(2)'s are ee_relaxed_density's); self is the MP2
        part, which has no oo or vv derivative blocks of its own."""
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
    factored densities, the JK fit's exchange tiles, one derivative unit)
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
    # GL, X and their raw-basis copies; the fitted Fock skeleton's exchange
    # tiles (L, g L, Kt, then g Kt over this rank's auxiliary tiles) and its
    # metric inverse, T and one tile's V^-1 T V^-1 rows
    exchange = 3.0 * nao * nocc * naux_jk * gb / ranks + 3.0 * naux_jk * naux_jk * gb
    engine = (4.0 if excited else 2.0) * A + exchange + block_gb
    held = factors + amplitudes + core + cderi_jk
    peak = held + max(pieces, engine, cderi_ri)
    flop = (2.0 * nao * nao * (nmo + no) * naux                 # Gamma3 blocks
            + 6.0 * no * nv * naux * naux                       # X transforms, N
            + 8.0 * nao * nao * nocc * naux_jk                  # J C_o, g Kt, S_x
            + 4.0 * nao * nocc * naux_jk * naux_jk              # Kt, T
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

def mp2_pieces(factors, tile_gb=EE_RI_TILE_GB):
    """(e_mp2, FactoredDerivatives) of closed-shell DF-MP2 on the provider's
    factors (df_correlation_force.mp2_pieces one occupied tile at a time);
    a collective over its ranks."""
    f, no, nv, naux = factors.f, factors.no, factors.nv, factors.f.naux
    comm = factors.comm
    ranks = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    L = f.L
    # occupied tiles of at most tile_gb, and no more than gives every rank one
    k = tile_width(8.0 * nv * no * nv * DF_FORCE_MP2_TILE_ARRAYS, -(-no // ranks),
                   Numerics(layout={'tile_gb': tile_gb}))
    tiles = [(k0, min(k0 + k, no)) for k0 in range(0, no, k)]
    GL = np.zeros((no * nv, naux))
    # e, g_oo and g_vv of each tile, flat, summed in tile order
    size = 1 + no * no + nv * nv
    parts = []
    for t, (k0, k1) in enumerate(tiles):
        if t % ranks != rank:
            continue
        I = factors.ints(k0, k1)
        D = dfa.doubles_denominator(f.eo, f.ev, f.eo[k0:k1])
        tau = (2.0 * I - I.transpose(dfa.PAIR)) / D
        oo, vv = dfa.tile_amplitude_fock(-I / D, tau)
        e = -float(np.sum(I * tau))
        del I, D
        GL[k0 * nv:k1 * nv] = -4.0 * (tau.reshape((k1 - k0) * nv, -1) @ L)
        flat = np.zeros(size)
        flat[0] = e
        flat[1:1 + no * no] = np.asarray(oo).ravel()
        flat[1 + no * no:] = np.asarray(vv).ravel()
        parts.append((t, flat))
    # every row of GL from one tile, the rest zeros: exact in any order
    reduce_sum(GL, comm)
    tot = ordered_sum(parts, comm, shape=(size,), onto=np.zeros(size))
    g_oo = tot[1:1 + no * no].reshape(no, no)
    g_vv = tot[1 + no * no:].reshape(nv, nv)
    return float(tot[0]), FactoredDerivatives(0.5 * (g_oo + g_oo.T),
                                              0.5 * (g_vv + g_vv.T), GL)


# ----------------------------------------------------------------------
# the orbital gradient and the correlation fit's metric density
# ----------------------------------------------------------------------

def orbital_gradient(factors, d, nf, Bc=None):
    """YG (nmo, nmo), YG_rp = sum_qP Gs2^P_pq B^P_rq over every row r
    (frozen core, active occupied, virtual), from the fold's factors chunk by
    chunk; Bc the core factors (ee_relaxed_density.core_factors) when nf."""
    f, no, nv = factors.f, factors.no, factors.nv
    nmo = nf + no + nv
    c, o, v = slice(0, nf), slice(nf, nf + no), slice(nf + no, nmo)
    GL3 = d.GL.reshape(no, nv, -1)
    X3 = None if d.X is None else d.X.reshape(no, nv, -1)
    y = d.y
    terms = {}
    for ch in factors.my_chunks:
        p0, p1 = factors.chunks[ch]
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
    return ordered_chain_sum(terms, factors.owners, factors.comm, np.zeros((nmo, nmo)))


def metric_density(factors, d, Xw):
    """Gamma2 (naux, naux) of the correlation fit, raw basis, symmetrized:
    X^T N X, N_SP = (1/2) sum_pq B^S_pq Gamma^P_pq, each chunk's rows from its
    owner."""
    f, no, nv, naux = factors.f, factors.no, factors.nv, factors.f.naux
    N = np.zeros((naux, naux))
    for ch in factors.my_chunks:
        p0, p1 = factors.chunks[ch]
        pc = p1 - p0
        n = f.LT[p0:p1].reshape(pc, -1) @ d.GL
        if d.Goo is not None:
            n += f.BooP[p0:p1].reshape(pc, -1) @ d.Goo.reshape(no * no, naux)
        if d.X is not None:
            # sum_a y_ka B^S_ab as [S, (k, b)]
            yB = (d.y @ f.bvv(ch, p0, p1).reshape(nv, -1)).reshape(no, pc, nv)
            n += yB.transpose(1, 0, 2).reshape(pc, -1) @ d.X
        N[p0:p1] = 0.5 * n
    allgather_ranges(N, factors.chunk_ranges, factors.comm)
    G2 = Xw.T @ N @ Xw
    return 0.5 * (G2 + G2.T)


def raw_basis(factors, A, Xw):
    """(A Xw)^T (naux, rows), auxiliary-major, each chunk of columns from its
    owner: a factored Gamma (rows, naux) taken to the raw auxiliary basis."""
    naux = Xw.shape[0]
    out = np.zeros((naux, A.shape[0]))
    for ch in factors.my_chunks:
        p0, p1 = factors.chunks[ch]
        # X = L^-1 is lower triangular
        out[p0:p1] = (A[:, p0:] @ Xw[p0:, p0:p1]).T
    return allgather_ranges(out, factors.chunk_ranges, factors.comm)


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
    functions each (`skeleton_tiles.shell_runs`) with their function ranges."""
    return [(s0, s1, int(ao_loc[s0]), int(ao_loc[s1]))
            for s0, s1 in shell_runs(ao_loc, width)]


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
    (a refused input) is broadcast and raised on every rank, so the other
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
# the derivative-integral pass
# ----------------------------------------------------------------------

class _RIFit:
    """Gamma3 blocks of the correlation fit from the raw factored arrays."""

    def __init__(self, auxmol, C, nf, no, d, Xw, factors):
        nocc = nf + no
        self.auxmol = auxmol
        naux = Xw.shape[0]
        nv = C.shape[1] - nocc
        self.Co, self.Cv = C[:, nf:nocc], C[:, nocc:]
        self.GL = raw_basis(factors, d.GL, Xw).reshape(naux, no, nv)
        self.Goo = None
        if d.Goo is not None:
            G = raw_basis(factors, d.Goo.reshape(no * no, -1), Xw).reshape(naux, no, no)
            self.Goo = G + G.transpose(0, 2, 1)
        self.X = None
        parts = [self.Co, self.Cv]
        if d.X is not None:
            self.X = raw_basis(factors, d.X, Xw).reshape(naux, no, nv)
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

def correlation_force(mol, mf, factors, auxmol, d, nf, Bc=None, Y_extra=None,
                      block_gb=DF_FORCE_BLOCK_GB, width=DF_FORCE_AUX_CHUNK,
                      numerics=None, flags=None):
    """(force (natm, 3) of E_c, gamma (MO, relaxed), max |Y - Y^T|): the
    tiled df_correlation_force.correlation_force, a collective over the
    provider's ranks; auxmol the correlation fit of its factors."""
    comm = factors.comm
    C = np.asarray(mf.mo_coeff)
    nocc = mol.nelectron // 2
    nmo, no = C.shape[1], factors.no
    if auxmol.nao != factors.f.naux:
        raise ValueError(f'the factors hold {factors.f.naux} auxiliary functions, the fit '
                         f'{auxmol.nao}: a linearly dependent metric was pruned')
    if nmo != nf + factors.no + factors.nv:
        raise ValueError('the operator does not span the mean field\'s orbitals')
    YG = orbital_gradient(factors, d, nf, Bc)
    if Y_extra is not None:
        YG = YG + Y_extra
    gamma0 = np.zeros((nmo, nmo))
    o, v = slice(nf, nocc), slice(nocc, nmo)
    gamma0[o, o], gamma0[v, v] = d.g_oo, d.g_vv

    def relaxed():
        # rank 0 alone: its Z-vector solve is serial
        with distributed(None):
            return dcf.relax_gradient(mf, mol, YG, gamma0, nf, numerics, flags)
    gamma, Y = on_root(relaxed, [(nmo, nmo)] * 2, comm)
    asym = float(np.max(np.abs(Y - Y.T)))
    W = 0.25 * (Y + Y.T)
    gamma_ao, W_ao = C @ gamma @ C.T, C @ W @ C.T
    f1, = on_root(lambda: one_electron_skeleton(mol, mf, [gamma_ao], [-W_ao]),
                  [(mol.natm, 3)], comm)
    # the correlation fit
    _, Xw = dcf.whitening(auxmol)
    lockstep(Xw, comm)
    G2ri = metric_density(factors, d, Xw)
    ri = _RIFit(auxmol, C, nf, no, d, Xw, factors)
    f2 = skeleton_pass(mol, [ri], comm, block_gb, width) + metric_force(auxmol, G2ri)
    # the SCF's JK fit
    auxjk = pyscf_df.addons.make_auxmol(mol, dcf.jk_fit(mf))
    Co = np.ascontiguousarray(C[:, :nocc])
    f2 += fitted_fock_skeleton(mol, auxjk, gamma_ao, 2.0 * Co @ Co.T,
                               occ=np.sqrt(2.0) * Co, exchange=1.0, comm=comm)
    return lockstep(f1 + f2, comm), gamma, asym


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
    est = estimate(mol, frozen, auxbasis, dcf.jk_fit(mf), ranks, tile_gb,
                   excited=excited, block_gb=block_gb)
    refuse_over(est, mol.max_memory if max_memory is None else max_memory)
    # every rank's partials are of rank 0's orbitals
    lockstep_mean_field(mf, comm)
    return comm, est


def force(mf, mol, factors, frozen, auxbasis, excited=None, tile_gb=EE_RI_TILE_GB,
          block_gb=DF_FORCE_BLOCK_GB, width=DF_FORCE_AUX_CHUNK, numerics=None,
          flags=None):
    """df_correlation_force.mp2_force on the provider's tiles and ranks, a
    collective over its communicator with the same bits on every rank;
    excited(d, C, nocc, nmo) -> (d plus the excitation's factored derivatives,
    Y_extra, extra)."""
    C = np.asarray(mf.mo_coeff)
    nmo, nocc = C.shape[1], mol.nelectron // 2
    e2, d = mp2_pieces(factors, tile_gb)
    Y_extra = extra = None
    if excited is not None:
        d, Y_extra, extra = excited(d, C, nocc, nmo)
    Bc = factors.core_factors(mol, mf, frozen, auxbasis) if frozen else None
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis)
    fc, gam, asym = correlation_force(mol, mf, factors, auxmol, d, frozen, Bc,
                                      Y_extra, block_gb, width, numerics, flags)
    F = on_root(lambda: (dcf.hf_force(mf),), [(mol.natm, 3)], factors.comm)[0] + fc
    return F, (e2, gam, asym, extra)


def mp2_force_tiled(mf, mol, factors, frozen=0, auxbasis=None, tile_gb=EE_RI_TILE_GB,
                    max_memory=None, block_gb=DF_FORCE_BLOCK_GB,
                    width=DF_FORCE_AUX_CHUNK, comm=None, aux_chunk=None,
                    numerics=None, flags=None):
    """(dE/dR, (e_mp2, gamma, max |Y - Y^T|, estimate)) of E_HF + E_MP2 on the
    tiled route over comm's ranks (prepare); factors(mol, mf, frozen,
    auxbasis, tile_gb, comm, aux_chunk) -> the factor provider (module
    docstring), built once the memory estimate has passed
    (`gradients.adc.fold_factors.ground_factors`)."""
    comm, est = prepare(mf, mol, frozen, auxbasis, False, tile_gb, block_gb,
                        max_memory, comm)
    provider = factors(mol, mf, frozen, auxbasis, tile_gb, comm, aux_chunk)
    F, (e2, gam, asym, _) = force(mf, mol, provider, frozen, auxbasis, None, tile_gb,
                                  block_gb, width, numerics, flags)
    return F, (e2, gam, asym, est)
