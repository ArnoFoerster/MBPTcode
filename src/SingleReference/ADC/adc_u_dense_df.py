"""Spin-orbital dense supermatrix from the DF factor B_spin: the
antisymmetrized g is reconstructed once (nso^4 -- dense benchmarking
route, the matrix-free adc_u_sigma_df is the production DF path)."""
from src.SingleReference.ADC.adc_utils import g_block_df
from src.SingleReference.ADC import adc_u_dense_full


def build_supermatrix(s, nocc, static_correction=None):
    full = slice(0, s.norb)
    g = g_block_df(s.B_spin, full, full, full, full)
    return adc_u_dense_full.build_supermatrix(s, nocc, static_correction, g=g)
