"""Spin-orbital matrix-free ADC(3) operator, dense-g route: the C blocks
are applied via sigma-vector contractions, U blocks via the bilinear
t2 closed forms (dressing hook: t2_ijcd + u1_shift)."""
import numpy as np

from src.SingleReference.CC.cached_einsum import einsum as _cached_einsum
from src.SingleReference.ADC.adc_u_utils import (
    u2_denominators, u1_dressing_shift,
    u1_shift_terms_2h1p, u1_shift_terms_2p1h)


def apply_U_2h1p(s, nocc, z_p, Vfull, t2_ijcd=None, u1_shift=None,
                 t2_ring=None):
    """(dy_2h1p_full, dy_p): both matvec directions of the U_2h1p block.

    t2_ring: separate amplitudes for the ring (KC) terms, t2_ijcd then only
    serving the ladder (CD) term -- Faddeev-ADC(3)'s particle-hole and
    particle-particle channels; default t2_ijcd for both."""
    norb = s.norb
    occ, virt = slice(0, nocc), slice(nocc, norb)
    dy_shift, dy_p_shift = u1_shift_terms_2h1p(norb, nocc, z_p, Vfull, u1_shift)

    g = s.g
    if t2_ijcd is None:
        denom_ij_cd, _ = u2_denominators(s.eps, nocc)
        t2_ijcd = g[occ, occ, virt, virt] / denom_ij_cd

    t2r = t2_ijcd if t2_ring is None else t2_ring

    g_ij_p_a = g[occ, occ, :, virt]        # (O,O,norb,V): g[i,j,p,a]
    g_v_p_v = g[virt, virt, :, virt]        # (V,V,norb,V): g[c,d,p,a]
    g_v_o_o = g[virt, occ, :, occ]          # (V,O,norb,O): g[c,m,p,k]

    # ---- forward: z_p -> dy_2h1p ----
    T0 = _cached_einsum('ijpa,p->ija', g_ij_p_a, z_p, optimize=True)
    W1 = _cached_einsum('cdpa,p->cda', g_v_p_v, z_p, optimize=True)
    CD = 0.5 * _cached_einsum('ijcd,cda->ija', t2_ijcd, W1, optimize=True)
    W2 = _cached_einsum('cmpk,p->cmk', g_v_o_o, z_p, optimize=True)   # (V,O,O)
    KC1 = -_cached_einsum('ikca,cjk->ija', t2r, W2, optimize=True)
    KC2 = _cached_einsum('jkca,cik->ija', t2r, W2, optimize=True)
    dy_2h1p_full = T0 + CD + KC1 + KC2

    # ---- adjoint: Vfull -> dy_p ----
    T0_adj = 0.5 * _cached_einsum('ijpa,ija->p', g_ij_p_a, Vfull, optimize=True)
    X1 = _cached_einsum('ijcd,ija->cda', t2_ijcd, Vfull, optimize=True)
    CD_adj = 0.25 * _cached_einsum('cda,cdpa->p', X1, g_v_p_v, optimize=True)
    Y2 = _cached_einsum('ikca,ija->kcj', t2r, Vfull, optimize=True)
    KC1_adj = -0.5 * _cached_einsum('kcj,cjpk->p', Y2, g_v_o_o, optimize=True)
    Y3 = _cached_einsum('jkca,ija->kci', t2r, Vfull, optimize=True)
    KC2_adj = 0.5 * _cached_einsum('kci,cipk->p', Y3, g_v_o_o, optimize=True)
    dy_p = T0_adj + CD_adj + KC1_adj + KC2_adj

    return dy_2h1p_full + dy_shift, dy_p + dy_p_shift


def apply_U_2p1h(s, nocc, z_p, Vfull, t2_ijcd=None, u1_shift=None,
                 t2_ring=None):
    """2p1h mirror of apply_U_2h1p; one shared t2 array serves both
    sectors (transposed view, signs absorbed). t2_ring as in apply_U_2h1p."""
    norb = s.norb
    occ, virt = slice(0, nocc), slice(nocc, norb)
    dy_shift, dy_p_shift = u1_shift_terms_2p1h(norb, nocc, z_p, Vfull, u1_shift)

    g = s.g
    if t2_ijcd is None:
        denom_ij_cd, _ = u2_denominators(s.eps, nocc)
        t2_ijcd = g[occ, occ, virt, virt] / denom_ij_cd
    t2m = t2_ijcd.transpose(2, 3, 0, 1)      # == -t2_abkl, VIEW (see docstring)
    t2mr = t2m if t2_ring is None else t2_ring.transpose(2, 3, 0, 1)

    g_ab_p_i = g[virt, virt, :, occ]        # (V,V,norb,O): g[a,b,p,i]
    g_o_o_p = g[occ, occ, :, occ]           # (O,O,norb,O): g[k,l,p,i]
    g_o_v_v = g[occ, virt, :, virt]         # (O,V,norb,V): g[k,m,p,c]

    # ---- forward: z_p -> dy_2p1h ----
    T0 = _cached_einsum('abpi,p->iab', g_ab_p_i, z_p, optimize=True)
    W1 = _cached_einsum('klpi,p->kli', g_o_o_p, z_p, optimize=True)
    CD = 0.5 * _cached_einsum('abkl,kli->iab', t2m, W1, optimize=True)
    W2 = _cached_einsum('kmpc,p->kmc', g_o_v_v, z_p, optimize=True)   # (O,V,V)
    KC1 = -_cached_einsum('acki,kbc->iab', t2mr, W2, optimize=True)
    KC2 = _cached_einsum('bcki,kac->iab', t2mr, W2, optimize=True)
    dy_2p1h_full = T0 + CD + KC1 + KC2

    # ---- adjoint: Vfull -> dy_p ----
    T0_adj = 0.5 * _cached_einsum('abpi,iab->p', g_ab_p_i, Vfull, optimize=True)
    X1 = -_cached_einsum('abkl,iab->kli', t2m, Vfull, optimize=True)
    CD_adj = -0.25 * _cached_einsum('kli,klpi->p', X1, g_o_o_p, optimize=True)
    Y2 = -_cached_einsum('acki,iab->kcb', t2mr, Vfull, optimize=True)
    KC1_adj = 0.5 * _cached_einsum('kcb,kbpc->p', Y2, g_o_v_v, optimize=True)
    Y3 = -_cached_einsum('bcki,iab->kca', t2mr, Vfull, optimize=True)
    KC2_adj = -0.5 * _cached_einsum('kca,kapc->p', Y3, g_o_v_v, optimize=True)
    dy_p = T0_adj + CD_adj + KC1_adj + KC2_adj

    return dy_2p1h_full + dy_shift, dy_p + dy_p_shift


def apply_C_2h1p(s, nocc, V):
    """(Vfull, c_2h1p): C_2h1p sigma-vector contribution (K added by the
    caller); Vfull is the antisymmetric extension of the packed V."""
    ing = s._build_matrix_free_ingredients(nocc)
    iu, ju, nvirt = ing['iu'], ing['ju'], ing['nvirt']
    Vfull = np.zeros((nocc, nocc, nvirt))
    Vfull[iu, ju, :] = V
    Vfull[ju, iu, :] = -V

    g_oooo, g_cvov = ing['g_oooo'], ing['g_cvov']
    Y1 = -0.5 * _cached_einsum('ijkl,kla->ija', g_oooo, Vfull, optimize=True)
    Y2n4 = _cached_einsum('cjam,imc->ija', g_cvov, Vfull, optimize=True)
    Y3n5 = _cached_einsum('ciam,jmc->ija', g_cvov, Vfull, optimize=True)
    return Vfull, (Y1 + Y2n4 - Y3n5)[iu, ju, :]


def apply_C_2p1h(s, nocc, V):
    """Mirror of apply_C_2h1p (occ<->virt swapped)."""
    ing = s._build_matrix_free_ingredients(nocc)
    au, bu, nvirt = ing['au'], ing['bu'], ing['nvirt']
    Vfull = np.zeros((nocc, nvirt, nvirt))
    Vfull[:, au, bu] = V
    Vfull[:, bu, au] = -V

    g_vvvv, g_ovov = ing['g_vvvv'], ing['g_ovov']
    Y1 = 0.5 * _cached_einsum('abcd,icd->iab', g_vvvv, Vfull, optimize=True)
    Y2n4 = -_cached_einsum('kbim,kam->iab', g_ovov, Vfull, optimize=True)
    Y3n5 = _cached_einsum('kaim,kbm->iab', g_ovov, Vfull, optimize=True)
    return Vfull, (Y1 + Y2n4 + Y3n5)[:, au, bu]


def build_operator(s, nocc, static_correction=None, t2_ijcd=None):
    """(aop, diag, dims): matrix-free spin-orbital ADC(3) operator.
    t2_ijcd: optional EN-dressed T2^(1) (u1 shift threaded automatically)."""
    ing = s._build_matrix_free_ingredients(nocc)
    norb = s.norb
    eps = s.eps
    d = ing['d']
    nH = d['nH']
    npair_o, npair_v = ing['npair_o'], ing['npair_v']
    off_2h1p, off_2p1h = ing['off_2h1p'], ing['off_2p1h']
    K_2h1p, K_2p1h = ing['K_2h1p'], ing['K_2p1h']
    iu, ju, au, bu = ing['iu'], ing['ju'], ing['au'], ing['bu']

    if static_correction is None:
        F = np.diag(eps)
    else:
        F = np.diag(eps) + static_correction

    u1_shift = None
    if t2_ijcd is not None:
        u1_shift = u1_dressing_shift(s, nocc, t2_ijcd)

    def aop(z):
        z_p = z[:norb]
        z_2h1p = z[off_2h1p:off_2p1h].reshape(npair_o, -1)
        z_2p1h = z[off_2p1h:nH].reshape(nocc, npair_v)

        Vfull_2h1p, c_2h1p = apply_C_2h1p(s, nocc, z_2h1p)
        Vfull_2p1h, c_2p1h = apply_C_2p1h(s, nocc, z_2p1h)
        u2h1p_full, u_from_2h1p = apply_U_2h1p(s, nocc, z_p, Vfull_2h1p, t2_ijcd,
                                               u1_shift=u1_shift)
        u2p1h_full, u_from_2p1h = apply_U_2p1h(s, nocc, z_p, Vfull_2p1h, t2_ijcd,
                                               u1_shift=u1_shift)

        y_p = F @ z_p + u_from_2h1p + u_from_2p1h
        y_2h1p = u2h1p_full[iu, ju, :].reshape(-1) + K_2h1p * z_2h1p.reshape(-1) + c_2h1p.reshape(-1)
        y_2p1h = u2p1h_full[:, au, bu].reshape(-1) + K_2p1h * z_2p1h.reshape(-1) + c_2p1h.reshape(-1)
        return np.concatenate([y_p, y_2h1p, y_2p1h])

    diag = np.concatenate([np.diag(F), K_2h1p, K_2p1h])
    return aop, diag, d
