"""Spin-orbital dense EE-ADC (electronic-excitation) supermatrix, levels
adc1/adc2/adc2x/adc3 -- the explicit-diagonalization route, mirroring
adc_u_dense_full.py's role on the IP/EA side.

Block content by level (Schirmer's order rule n - mu_I - mu_J, mu = 0 for
ph and 1 for 2p2h; identical to adc_order_rule.blocks_at, which is
particle-number agnostic):

    level | ph/ph      | ph/2p2h      | 2p2h/2p2h
    adc1  | A31 A32    | -            | -
    adc2  | + A33      | A54          | A61
    adc2x | + A33      | A54          | A61 + A63
    adc3  | + A34      | A54 + A55    | A61 + A63

Equation labels are Leitner/Dempwolff/Dreuw, JCP 157, 184101 (2022); see
ee_utils for the conventions and the truncation table.

The ph/2p2h block is built ONLY from the doubles-row equations (A54/A55),
where the singles ket carries no configuration redundancy, and the ph-row
block is its transpose. The independent ph-row equations (A41/A42) are
implemented in ee_u_sigma_full and cross-checked against this transpose --
that check is what validates the unique-configuration (i<j, a<b) mapping of
the paper's full-range sums.

Validation scale: the ph/2p2h builder materializes a six-index temporary
(nocc^2 nvirt^2 nocc nvirt) for the two amplitude-dressed A55 terms. That is
smaller than the supermatrix it feeds, so it is never the binding
constraint, but it is a dense-route object -- the <=4-index production rule
applies to the matrix-free modules.
"""
import numpy as np

from src.SingleReference.CC.cached_einsum import einsum as _einsum
from src.SingleReference.ADC.eeADC import ee_utils
from src.SingleReference.ADC.eeADC import ee_equations as _eq
from src.SingleReference.ADC.eeADC import ee_en as _en

LEVELS = ('adc1', 'adc2', 'adc2x', 'adc3')

# (ph/ph order, ph/2p2h order, 2p2h/2p2h order); None = block absent
_BLOCK_ORDERS = {
    'adc1':  (1, None, None),
    'adc2':  (2, 1, 0),
    'adc2x': (2, 1, 1),
    'adc3':  (3, 2, 1),
}


def build_supermatrix(eps, g, nocc, level='adc3', amps=None, zint=None,
                      rho=None, en_dress=None):
    """(nH, nH) spin-orbital EE-ADC supermatrix at `level`.

    eps: (norb,) spin-orbital energies, occupied first.
    g:   (norb,)*4 antisymmetrized <pq||rs>.
    Returns a symmetric array; singles segment first, then doubles."""
    if level not in LEVELS:
        raise ValueError(f"level={level!r}; expected one of {LEVELS}")
    norb = len(eps)
    o_ss, o_sd, o_dd = _BLOCK_ORDERS[level]
    d = ee_utils.dimensions(nocc, norb)
    n_s, n_d = d['n_s'], d['n_d']

    amps, zint, rho = _ingredients(eps, g, nocc, norb, level, amps, zint,
                                   rho, en_dress)

    if o_sd is None:                                   # ADC(1): singles only
        return m_ss(eps, g, nocc, norb, o_ss, amps, zint, rho).reshape(n_s, n_s)

    H = np.zeros((n_s + n_d, n_s + n_d))
    H[:n_s, :n_s] = m_ss(eps, g, nocc, norb, o_ss, amps, zint, rho).reshape(n_s, n_s)
    block_ds = m_ds(g, nocc, norb, o_sd, amps, zint)
    H[n_s:, :n_s] = block_ds
    H[:n_s, n_s:] = block_ds.T
    H[n_s:, n_s:] = m_dd(eps, g, nocc, norb, o_dd)
    return H


def _ingredients(eps, g, nocc, norb, level, amps, zint, rho, en_dress=None):
    """MP amplitudes / Z intermediates / density, computed on demand."""
    order = {'adc1': 1, 'adc2': 2, 'adc2x': 2, 'adc3': 3}[level]
    if order == 1:
        return None, None, None
    if amps is None:
        den = (None if en_dress is None else
               _en.en_denominators_spin_orbital(eps, g, nocc, en_dress))
        amps = ee_utils.mp_amplitudes(eps, g, nocc, norb, order=order,
                                      denominators=den)
    if zint is None:
        zint = ee_utils.z_intermediates(g, amps, nocc, norb, order=order)
    if rho is None and order >= 3:
        rho = ee_utils.mp_density2(amps)
    return amps, zint, rho


# ======================================================================
# ph/ph block  M_ia,jb   (A31-A34)
# ======================================================================

def m_ss(eps, g, nocc, norb, order, amps=None, zint=None, rho=None):
    """(nocc, nvirt, nocc, nvirt) tensor M_ia,jb through `order` (A31-A34).

    The equations live in ee_equations, evaluated here on full spin-orbital
    arrays; the spin-free route evaluates the same strings on spatial spin
    blocks."""
    gb = ee_utils.g_blocks(g, nocc, norb)
    _, _, d_ph = ee_utils._denominators(eps, nocc)
    return _eq.m_ss(_eq.NUMPY, gb, order, amps, zint, rho, d_ph,
                    nocc, norb - nocc)


# ======================================================================
# 2p2h/ph block  (A54, A55) -- rows = doubles configs, cols = singles
# ======================================================================

def m_ds(g, nocc, norb, order, amps=None, zint=None):
    """(n_d, n_s) coupling block through `order` (1 or 2)."""
    gb = ee_utils.g_blocks(g, nocc, norb)
    nvirt = norb - nocc
    eye_o, eye_v = np.eye(nocc), np.eye(nvirt)

    # B[i,j,a,b,k,c]: coefficient of the singles amplitude Y^c_k in W^ab_ij.
    # A54, first order.
    B = -0.5 * (_einsum('ijka,bc->ijabkc', gb['ooov'], eye_v, optimize=True)
                - _einsum('ijkb,ac->ijabkc', gb['ooov'], eye_v, optimize=True)
                + _einsum('icab,jk->ijabkc', gb['ovvv'], eye_o, optimize=True)
                - _einsum('jcab,ik->ijabkc', gb['ovvv'], eye_o, optimize=True))

    if order >= 2:                                      # A55, second order
        ZA, ZB, t1 = zint['ZA'], zint['ZB'], amps['t2_1']
        B += 0.5 * (
            _einsum('ijka,bc->ijabkc', ZA, eye_v, optimize=True)
            - _einsum('ijkb,ac->ijabkc', ZA, eye_v, optimize=True)
            + _einsum('ijbd,kadc->ijabkc', t1, gb['ovvv'], optimize=True)
            - _einsum('ijad,kbdc->ijabkc', t1, gb['ovvv'], optimize=True)
            + _einsum('jcab,ik->ijabkc', ZB, eye_o, optimize=True)
            - _einsum('icab,jk->ijabkc', ZB, eye_o, optimize=True)
            + _einsum('jmab,mkic->ijabkc', t1, gb['ooov'], optimize=True)
            - _einsum('imab,mkjc->ijabkc', t1, gb['ooov'], optimize=True))

    iu, ju, au, bu = ee_utils.pair_indices(nocc, norb)
    blk = B[iu[:, None], ju[:, None], au[None, :], bu[None, :]]      # (npo,npv,nocc,nvirt)
    n_d = len(iu) * len(au)
    # METRIC FACTOR 2 -- see ee_utils.PAPER_DOUBLES_SCALE. The paper normalizes
    # its amplitude vector with UNRESTRICTED sums (Y'Y = 1 over all ijab), so
    # its doubles components are half the orthonormal unique-configuration
    # coefficients; converting the doubles ROW of a block to that basis is
    # U^dagger = 2 x restrict-to-unique. Verified elementwise against the exact
    # Slater-Condon <Phi_ij^ab|H|Phi_k^c> at 1e-15 on HF/H2O/LiH.
    return 2.0 * blk.reshape(n_d, nocc * nvirt)


# ======================================================================
# 2p2h/2p2h block  (A61 diagonal, A63 first order)
# ======================================================================

def m_dd(eps, g, nocc, norb, order):
    """(n_d, n_d) doubles block: A61 (Fock diagonal) plus, at order>=1, the
    A63 ladder/ring terms. Built by configuration-index broadcasting with
    delta masks (the adc_u_dense_full.C_2h1p_block idiom)."""
    I, J, A, B = ee_utils.configs_doubles(nocc, norb)
    n_d = len(I)
    Ic, Jc, Ac, Bc = I[:, None], J[:, None], A[:, None], B[:, None]
    K, L, C, D = I[None, :], J[None, :], A[None, :], B[None, :]

    H = np.zeros((n_d, n_d))
    # A61 in a canonical HF basis: (eps_a + eps_b - eps_i - eps_j) on the diagonal
    np.fill_diagonal(H, eps[A] + eps[B] - eps[I] - eps[J])
    if order < 1:
        return H

    d_ik, d_jl = (Ic == K), (Jc == L)
    d_ac, d_bd = (Ac == C), (Bc == D)
    d_jk, d_il = (Jc == K), (Ic == L)
    d_bc, d_ad = (Bc == C), (Ac == D)

    # 1/2 sum_cd <ab||cd> Y^cd_ij  ->  delta_ik delta_jl <ab||cd>
    H += (d_ik & d_jl) * g[Ac, Bc, C, D]
    # 1/2 sum_kl <ij||kl> Y^ab_kl  ->  delta_ac delta_bd <ij||kl>
    H += (d_ac & d_bd) * g[Ic, Jc, K, L]

    # - P^-_ij P^-_ab sum_kc <ka||ic> Y^bc_jk
    def _ex(i, j, a, b):
        """-[ d_jk d_bc <la||id> - d_jk d_bd <la||ic>
             - d_jl d_bc <ka||id> + d_jl d_bd <ka||ic> ]"""
        djk, djl = (j == K), (j == L)
        dbc, dbd = (b == C), (b == D)
        return -((djk & dbc) * g[L, a, i, D] - (djk & dbd) * g[L, a, i, C]
                 - (djl & dbc) * g[K, a, i, D] + (djl & dbd) * g[K, a, i, C])

    H += (_ex(Ic, Jc, Ac, Bc) - _ex(Jc, Ic, Ac, Bc)
          - _ex(Ic, Jc, Bc, Ac) + _ex(Jc, Ic, Bc, Ac))
    return H
