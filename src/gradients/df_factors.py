"""The RI factors of a correlation fit on a closed-shell mean field, as the
factor provider the tiled correlation force reads
(`df_correlation_force_tiled`, module docstring): the default of the tiled MP2
force, built from pySCF's own density fitting with no method of its own.

B^P_pq = sum_mu nu C_mu p C_nu q (P|mu nu)~, the whitened three-index factor
pySCF's `df.DF` holds, in the layouts the force contracts: B_ia and B_ij whole
and auxiliary-major (one copy per node, `NodeShared`), B_ab only in the
auxiliary chunks this rank owns (`aux_chunks`). The chunks, the occupied
tiles' integrals and the frozen core's factors are made exactly as the folded
ADC(2) operator makes its own (`gradients.adc.fold_factors`), so either
provider gives the force the same bits.
"""
import numpy as np
from pyscf import df as pyscf_df
from pyscf import lib

from src.Base.numerics.constants import (EE_RI_AUX_CHUNK, EE_RI_EXCHANGE_ARRAYS,
                                         EE_RI_EXCHANGE_CHUNK, EE_RI_TILE_GB)
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.memory import tile_width
from src.Base.utils.parallel.mpi_grid import NodeShared, current_comm


def aux_chunks(naux, nv, size, width=None):
    """(p0, p1) chunks of the auxiliary index, fixed by naux and nv alone (or
    an explicit width), and the rank owning each: consecutive runs, so that a
    sum over chunks in chunk order has the one-rank bits at every rank
    count."""
    step = width or tile_width(8.0 * EE_RI_EXCHANGE_ARRAYS * nv * nv,
                               EE_RI_EXCHANGE_CHUNK,
                               Numerics(layout={'tile_gb': EE_RI_TILE_GB}))
    chunks = [(p0, min(p0 + step, naux)) for p0 in range(0, naux, step)]
    return chunks, [c * size // len(chunks) for c in range(len(chunks))]


class Factors:
    """B_ia (LT, (naux, no, nv), and L, its (no nv, naux) view), B_ij (BooP,
    (naux, no, no)) and this rank's B_ab chunks as [a, P, b], with the active
    orbital energies eo, ev."""

    def __init__(self, eo, ev, LT, BooP, bvv_chunks, shared=()):
        self.eo, self.ev = np.asarray(eo, float), np.asarray(ev, float)
        self.LT, self.BooP, self.bvv_chunks = LT, BooP, bvv_chunks
        self.naux = LT.shape[0]
        self.shared = tuple(shared)

    @property
    def L(self):
        return self.LT.reshape(self.naux, -1).T

    def bvv(self, c, p0, p1):
        """B^P_ab of this rank's auxiliary chunk c = [p0, p1) as [a, P, b]."""
        return self.bvv_chunks[c]


class DFFactors:
    """The factor provider of one correlation fit on `mf` (module docstring);
    `build` makes it."""

    def __init__(self, with_df, f, comm, chunks, owners):
        size = 1 if comm is None else comm.Get_size()
        rank = 0 if comm is None else comm.Get_rank()
        self.with_df, self.f, self.comm = with_df, f, comm
        self.no, self.nv = f.eo.size, f.ev.size
        self.chunks, self.owners = chunks, owners
        self.my_chunks = [c for c, o in enumerate(owners) if o == rank]
        self.chunk_ranges = [[chunks[c] for c, o in enumerate(owners) if o == r]
                             for r in range(size)]

    @classmethod
    def build(cls, mol, mf, frozen, auxbasis, _tile_gb=None, comm=None,
              aux_chunk=None):
        """The provider of the fit `auxbasis` on mf's orbitals, the `frozen`
        lowest occupied dropped, over the ranks of comm (default: the
        current communicator). `_tile_gb` is not read (the call shape of
        `df_correlation_force_tiled.mp2_force_tiled`'s factors)."""
        comm = current_comm() if comm is None else comm
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
        sh_l, sh_o = NodeShared((naux, no, nv), comm), NodeShared((naux, no, no), comm)
        LT, BooP = sh_l.array, sh_o.array
        size = 1 if comm is None else comm.Get_size()
        rank = 0 if comm is None else comm.Get_rank()
        chunks, owners = aux_chunks(naux, nv, size, aux_chunk)
        mine = {c: (chunks[c], np.empty((nv, chunks[c][1] - chunks[c][0], nv)))
                for c in range(len(chunks)) if owners[c] == rank}
        p0 = 0
        for blk in with_df.loop(blksize=EE_RI_AUX_CHUNK):
            p1 = p0 + blk.shape[0]
            ao = lib.unpack_tril(blk)                       # (p, nao, nao)
            nao = ao.shape[1]
            if sh_l.fills:
                half_o = (ao.reshape(-1, nao) @ Co).reshape(-1, nao, no)
                half_o = half_o.transpose(0, 2, 1).reshape(-1, nao)  # (p i, mu)
                LT[p0:p1] = (half_o @ Cv).reshape(-1, no, nv)
                BooP[p0:p1] = (half_o @ Co).reshape(-1, no, no)
                del half_o
            keep = [(c, max(p0, q0), min(p1, q1)) for c, ((q0, q1), _) in mine.items()
                    if max(p0, q0) < min(p1, q1)]
            if keep:
                half_v = (ao.reshape(-1, nao) @ Cv).reshape(-1, nao, nv)
                half_v = half_v.transpose(0, 2, 1).reshape(-1, nao)  # (p a, mu)
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
        f = Factors(eps[frozen:nocc], eps[nocc:], LT, BooP,
                    {c: a for c, (_, a) in mine.items()}, shared=(sh_l, sh_o))
        return cls(with_df, f, comm, chunks, owners)

    def ints(self, k0, k1):
        """(kc|ld) for k in the occupied rows k0..k1-1, (k1 - k0, nv, no, nv)."""
        L, nv = self.f.L, self.nv
        I = L[k0 * nv:k1 * nv] @ L.T
        return I.reshape(k1 - k0, nv, self.no, nv)

    def core_factors(self, mol, mf, frozen, auxbasis):
        """B^P_cq (naux, frozen, nmo - frozen): the frozen core against every
        active orbital, in this fit."""
        with_df = pyscf_df.DF(mol, auxbasis=auxbasis)
        with_df.max_memory = mf.max_memory
        with_df.build()
        C = np.asarray(mf.mo_coeff)
        Cc, Ca = C[:, :frozen], C[:, frozen:]
        out = []
        for blk in with_df.loop():
            ao = lib.unpack_tril(blk)
            out.append(np.einsum('Pmn,mc,nq->Pcq', ao, Cc, Ca, optimize=True))
        return np.concatenate(out, axis=0)


#: The tiled MP2 force's default factors: factors(mol, mf, frozen, auxbasis,
#: tile_gb, comm, aux_chunk) -> provider.
mp2_factors = DFFactors.build
