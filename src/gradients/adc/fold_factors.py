"""The folded ADC(2) operator's RI factors as the factor provider the tiled
correlation force reads (`df_correlation_force_tiled`, module docstring).

An ADC(2) state's force runs on the fold that solved it (`FoldFactors(op)`);
the fold set up without a root (`ground_factors`, the ADC(1) level) gives
the MP2 force the same bits as its default provider
(`gradients.df_factors.mp2_factors`). Either way the provider hands over the fold's factors,
its occupied-tile integrals, its auxiliary chunks and their owners, its
communicator, and the frozen core's factors in the same fit.

The occupied-tile integrals are the fold's own (`RIFoldADC2.ints`), read here
and nowhere else in the gradient layer.
"""
from src.Base.numerics.constants import EE_RI_TILE_GB
from src.SingleReference.ADC.eeADC import ee_ri_fold
from src.SingleReference.ADC.eeADC.ee_relaxed_density import core_factors


class FoldFactors:
    """The factor provider of one folded operator `op` (module docstring)."""

    def __init__(self, op):
        self.op = op
        self.f, self.no, self.nv, self.comm = op.f, op.no, op.nv, op.comm
        self.chunks, self.my_chunks = op.chunks, op.my_chunks
        self.owners, self.chunk_ranges = op.owners, op.chunk_ranges

    def ints(self, k0, k1):
        """(ia|jb) of the occupied rows k0..k1-1 from the fold's factors."""
        return self.op.ints(k0, k1)

    def core_factors(self, mol, mf, frozen, auxbasis):
        """B^P_cq of the frozen core against every active orbital, in the
        fold's fit."""
        return core_factors(mol, mf, frozen, auxbasis)


def ground_factors(mol, mf, frozen, auxbasis, tile_gb=EE_RI_TILE_GB, comm=None,
                   aux_chunk=None):
    """The fold's factors, tiles and chunks without a root (ADC(1) setup), as
    a provider, bitwise equal to `gradients.df_factors.mp2_factors` for the
    tiled MP2 force."""
    f = ee_ri_fold.bare_factors(mol, mf, frozen=frozen, auxbasis=auxbasis, comm=comm,
                                aux_chunk=aux_chunk)
    return FoldFactors(ee_ri_fold.RIFoldADC2(f, 'singlet', level='adc1',
                                             tile_gb=tile_gb, comm=comm,
                                             aux_chunk=aux_chunk))
