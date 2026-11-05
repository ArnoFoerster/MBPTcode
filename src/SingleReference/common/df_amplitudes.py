"""Amplitude pieces of a density-fitted doubles energy that both the relaxed
densities and the nuclear forces read: the pair swap of a (kc|ld) block, the
orbital-energy denominator, the spin-component weights of a pair tensor, and the
Fock derivative of an amplitude A = N / D.

Spin-component scaling (SCS-MP2, Grimme, J. Chem. Phys. 118, 9095 (2003); SCS-ADC(2),
Hellweg, Gruen and Haettig, Phys. Chem. Chem. Phys. 10, 4119 (2008)) weights an
opposite-spin pair by c_os and a same-spin pair by c_ss. For a closed-shell pair
tensor X_kcld and its exchange partner X~_kcld = X_kdlc the opposite-spin part is X
and the same-spin part X - X~, so the singlet-coupled combination 2 X - X~ becomes
(c_os + c_ss) X - c_ss X~ and the triplet one -X~ becomes (c_ss - c_os) X - c_ss X~.
Written this way, c_os = c_ss = 1 gives the unscaled arithmetic bit for bit.

They sit below the gradient layer because the ADC(2) relaxed densities
(electronic structure) use them as well as the correlation force
(`src.gradients.df_correlation_force`).
"""
import numpy as np

from src.Base.numerics.constants import ADC2_SPIN_SCALING

PAIR = (0, 3, 2, 1)


def spin_scaling_factors(spin_scaling):
    """(c_os, c_ss) of a spin_scaling argument -- a name of ADC2_SPIN_SCALING
    ('scs', 'sos') or the factors -- or None for the unscaled method."""
    if spin_scaling is None:
        return None
    if isinstance(spin_scaling, str):
        if spin_scaling not in ADC2_SPIN_SCALING:
            raise ValueError(f'spin_scaling={spin_scaling!r}; expected one of '
                             f'{sorted(ADC2_SPIN_SCALING)} or (c_os, c_ss)')
        return ADC2_SPIN_SCALING[spin_scaling]
    c_os, c_ss = (float(c) for c in spin_scaling)
    return c_os, c_ss


def spin_weights(scaling):
    """(c_os, c_ss) of a spin scaling (`spin_scaling_factors`); (1, 1)
    unscaled."""
    factors = spin_scaling_factors(scaling)
    return (1.0, 1.0) if factors is None else factors


def singlet_pairs(X, Xx, scaling=None):
    """(c_os + c_ss) X - c_ss Xx: 2 X - Xx unscaled (module docstring)."""
    c_os, c_ss = spin_weights(scaling)
    return (c_os + c_ss) * X - c_ss * Xx


def triplet_pairs(X, Xx, scaling=None):
    """(c_ss - c_os) X - c_ss Xx: -Xx unscaled (module docstring)."""
    c_os, c_ss = spin_weights(scaling)
    return (c_ss - c_os) * X - c_ss * Xx


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
