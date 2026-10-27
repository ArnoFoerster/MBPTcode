"""Amplitude pieces of a density-fitted doubles energy that both the relaxed
densities and the nuclear forces read: the pair swap of a (kc|ld) block, the
orbital-energy denominator, and the Fock derivative of an amplitude A = N / D.

They sit below the gradient layer because the ADC(2) relaxed densities
(electronic structure) use them as well as the correlation force
(`src.gradients.df_correlation_force`).
"""
import numpy as np

PAIR = (0, 3, 2, 1)


def doubles_denominator(eo, ev, rows=None):
    """D_kcld = e_c + e_d - e_k - e_l, k over `rows` (default eo) and l over eo."""
    rows = eo if rows is None else rows
    return (ev[None, :, None, None] + ev[None, None, None, :]
            - rows[:, None, None, None] - eo[None, None, :, None])


def amplitude_fock(lam, A):
    """dE/df of an amplitude A = N / D (D = e_a + e_b - e_i - e_j) whose
    adjoint over D is lam = (dE/dA) / D, symmetric under the pair swap:
    (oo, vv)."""
    oo = (np.einsum('kcld,jcld->kj', lam, A)
          + np.einsum('lckd,lcjd->kj', lam, A))
    vv = -(np.einsum('kcld,kbld->cb', lam, A)
           + np.einsum('kdlc,kdlb->cb', lam, A))
    return oo, vv


def tile_amplitude_fock(lam, A):
    """amplitude_fock on one tile of rows, the swap symmetry of lam and A
    putting both terms of each block on the tile's first index."""
    oo = 2.0 * np.einsum('lckd,lcjd->kj', lam, A, optimize=True)
    vv = -2.0 * np.einsum('kcld,kbld->cb', lam, A, optimize=True)
    return oo, vv
