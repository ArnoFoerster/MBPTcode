"""The RI factors of a correlation fit on a closed-shell mean field, as the
one factor provider the tiled correlation force reads
(`df_correlation_force_tiled`, module docstring): `DFFactors.build` for the
tiled MP2 force (its default, `mp2_factors`), `DFFactors.of_fold` for an
ADC(2) state's force on the fold that solved it.

The factors are the folded ADC(2) operator's own, made by one function
(`ee_ri_fold.bare_factors`): B^P_ia and B^P_ij whole and auxiliary-major (one
copy per node, `NodeShared`), B^P_ab only in the auxiliary chunks this rank
owns (`ee_ri_fold.aux_chunks`). The occupied tiles' integrals and the frozen
core's factors are made here for both, so either gives the force the same
bits.

The correlation fit is named: `build` refuses auxbasis None, which in
`bare_factors` means the mean field's own JK fit. The tiled force refuses it
as well (`df_correlation_force_tiled.prepare`).
"""
from src.Base.utils.parallel.mpi_grid import current_comm
from src.SingleReference.ADC.eeADC import ee_ri_fold
from src.SingleReference.ADC.eeADC.ee_relaxed_density import core_factors


class DFFactors:
    """The factor provider of one correlation fit on `mf` (module docstring);
    `build` makes it."""

    def __init__(self, f, comm, chunks, owners):
        size = 1 if comm is None else comm.Get_size()
        rank = 0 if comm is None else comm.Get_rank()
        self.f, self.comm = f, comm
        self.no, self.nv = f.eo.size, f.ev.size
        self.chunks, self.owners = chunks, owners
        self.my_chunks = [c for c, o in enumerate(owners) if o == rank]
        self.chunk_ranges = [[chunks[c] for c, o in enumerate(owners) if o == r]
                             for r in range(size)]

    @classmethod
    def build(cls, mol, mf, frozen, auxbasis, _tile_gb=None, comm=None,
              aux_chunk=None):
        """The provider of the fit `auxbasis` (a name) on mf's orbitals, the
        `frozen` lowest occupied dropped, over the ranks of comm (default: the
        current communicator). `_tile_gb` is not read (the call shape of
        `df_correlation_force_tiled.mp2_force_tiled`'s factors)."""
        if auxbasis is None:
            raise ValueError('the correlation fit is given by name (auxbasis)')
        comm = current_comm() if comm is None else comm
        f = ee_ri_fold.bare_factors(mol, mf, frozen=frozen, auxbasis=auxbasis,
                                    comm=comm, aux_chunk=aux_chunk)
        size = 1 if comm is None else comm.Get_size()
        chunks, owners = ee_ri_fold.aux_chunks(f.naux, f.nv, size, aux_chunk)
        return cls(f, comm, chunks, owners)

    @classmethod
    def of_fold(cls, op):
        """The provider of the folded operator `op`'s own factors, chunks and
        communicator (`ee_ri_fold.RIFoldADC2`)."""
        return cls(op.f, op.comm, op.chunks, op.owners)

    def ints(self, k0, k1):
        """(kc|ld) for k in the occupied rows k0..k1-1, (k1 - k0, nv, no, nv)."""
        L, nv = self.f.L, self.nv
        I = L[k0 * nv:k1 * nv] @ L.T
        return I.reshape(k1 - k0, nv, self.no, nv)

    def core_factors(self, mol, mf, frozen, auxbasis):
        """B^P_cq (naux, frozen, nmo - frozen): the frozen core against every
        active orbital, in this fit (`ee_relaxed_density.core_factors`)."""
        return core_factors(mol, mf, frozen, auxbasis)


#: The tiled MP2 force's default factors: factors(mol, mf, frozen, auxbasis,
#: tile_gb, comm, aux_chunk) -> provider.
mp2_factors = DFFactors.build
