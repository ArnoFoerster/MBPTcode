"""Spin-orbital dense ADC(3) supermatrix (dense g route) and the
first-order C-block batch builders (g-free capable via GProxy)."""
import numpy as np

from src.SingleReference.CC.cached_einsum import einsum as _cached_einsum


def build_supermatrix(s, nocc, static_correction=None, g=None):
    """(nH, nH) spin-orbital ADC(3) supermatrix. g overrides s.g (the
    DF route passes a B-reconstructed tensor)."""
    if g is None:
        s._require_dense_g('build_supermatrix (dense ADC(3) supermatrix)')
        g = s.g
    eps = s.eps
    norb = s.norb
    occ = np.arange(nocc)
    virt = np.arange(nocc, norb)
    nvirt = len(virt)
    p_ = np.arange(norb)

    d = s.dimensions(nocc)
    nH = d['nH']

    # 2h1p configurations: (i<j, a), pair outer / a inner (matches vec_2h1p[iu,ju,:,:].reshape below)
    iu, ju = np.triu_indices(nocc, k=1)
    npair_o = len(iu)
    i_2h = np.repeat(occ[iu], nvirt)
    j_2h = np.repeat(occ[ju], nvirt)
    a_2h = np.tile(virt, npair_o)

    # 2p1h configurations: (i, a<b), i outer / pair inner
    au, bu = np.triu_indices(nvirt, k=1)
    npair_v = len(au)
    i_2p = np.repeat(occ, npair_v)
    a_2p = np.tile(virt[au], nocc)
    b_2p = np.tile(virt[bu], nocc)

    off_2h1p = norb
    off_2p1h = off_2h1p + d['n2h1p']

    H = np.zeros((nH, nH))

    # ============================= Block F =============================
    # (exact static 3h2p/3p2h correction NOT implemented; static_correction is
    # an optional approximate substitute, see class docstring)
    if static_correction is None:
        H[p_, p_] = eps
    else:
        H[:norb, :norb] = np.diag(eps) + static_correction

    # ============================ Block U_2h1p ============================
    g_ij_full = g[np.ix_(occ, occ, p_, p_)]              # (nocc,nocc,norb,norb): g[i,j,p,q]
    term0 = np.transpose(g_ij_full[:, :, :, virt], (0, 1, 3, 2))   # (i,j,a,p): g[i,j,p,a]

    g_ijcd = g[np.ix_(occ, occ, virt, virt)]             # (nocc,nocc,nvirt,nvirt): g[i,j,c,d]
    g_cd_pq = g[np.ix_(virt, virt, p_, p_)]              # (nvirt,nvirt,norb,norb): g[c,d,p,q]
    g_cdpa = g_cd_pq[:, :, :, virt]                       # (nvirt,nvirt,norb,nvirt): g[c,d,p,a]
    # EN-dressed when u2_denom_dress is set; identical to the bare arrays
    # otherwise. Built once and shared with the g_Xoca/g_Xvki terms below,
    # which use the SAME denominators (see _u2_denominators).
    denom_ij_cd, denom_ab_kl = s._u2_denominators(nocc)
    corr_cd = 0.5 * _cached_einsum('ijcd,cdpa->ijap', g_ijcd / denom_ij_cd, g_cdpa, optimize=True)

    g_Xoca = g[np.ix_(occ, occ, virt, virt)]             # generic occ,occ,virt,virt: g[X,Y,c,a]
    g_cXpk = g[np.ix_(virt, occ, p_, occ)]               # (nvirt,nocc,norb,nocc): g[c,X,p,k]
    denom_Xkca = denom_ij_cd            # same shape, same eps combination
    corr_kc1 = -_cached_einsum('ikca,cjpk->ijap', g_Xoca / denom_Xkca, g_cXpk, optimize=True)
    corr_kc2 = _cached_einsum('jkca,cipk->ijap', g_Xoca / denom_Xkca, g_cXpk, optimize=True)

    vec_2h1p = term0 + corr_cd + corr_kc1 + corr_kc2     # (nocc,nocc,nvirt,norb)
    block_U_2h1p = vec_2h1p[iu, ju, :, :].reshape(d['n2h1p'], norb)

    H[:norb, off_2h1p:off_2p1h] = block_U_2h1p.T
    H[off_2h1p:off_2p1h, :norb] = block_U_2h1p

    # ============================ Block U_2p1h ============================
    g_ab_full = g[np.ix_(virt, virt, p_, p_)]            # (nvirt,nvirt,norb,norb): g[a,b,p,q]
    term0_abip = np.transpose(g_ab_full[:, :, :, occ], (0, 1, 3, 2))   # (a,b,i,p): g[a,b,p,i]

    g_abkl = g[np.ix_(virt, virt, occ, occ)]             # (nvirt,nvirt,nocc,nocc): g[a,b,k,l]
    g_kl_pq = g[np.ix_(occ, occ, p_, p_)]                # (nocc,nocc,norb,norb): g[k,l,p,q]
    g_klpi = g_kl_pq[:, :, :, occ]                        # (nocc,nocc,norb,nocc): g[k,l,p,i]
    corr_kl_abip = -0.5 * _cached_einsum('abkl,klip->abip',
                                     g_abkl / denom_ab_kl,
                                     np.transpose(g_klpi, (0, 1, 3, 2)), optimize=True)

    term0 = np.transpose(term0_abip, (2, 0, 1, 3))       # (i,a,b,p)
    corr_kl = np.transpose(corr_kl_abip, (2, 0, 1, 3))

    g_Xvki = g[np.ix_(virt, virt, occ, occ)]             # generic virt,virt,occ,occ: g[X,Y,k,i]
    g_kXpc = g[np.ix_(occ, virt, p_, virt)]              # (nocc,nvirt,norb,nvirt): g[k,X,p,c]
    denom_XYki = denom_ab_kl            # same shape, same eps combination
    corr_kc1_iabp = _cached_einsum('acki,kbpc->iabp', g_Xvki / denom_XYki, g_kXpc, optimize=True)
    corr_kc2_iabp = -_cached_einsum('bcki,kapc->iabp', g_Xvki / denom_XYki, g_kXpc, optimize=True)

    vec_2p1h = term0 + corr_kl + corr_kc1_iabp + corr_kc2_iabp   # (nocc,nvirt,nvirt,norb)
    block_U_2p1h = vec_2p1h[:, au, bu, :].reshape(d['n2p1h'], norb)

    H[:norb, off_2p1h:nH] = block_U_2p1h.T
    H[off_2p1h:nH, :norb] = block_U_2p1h

    # ===================== Blocks K_2h1p / K_2p1h (diagonal) =====================
    rows_2h1p = np.arange(off_2h1p, off_2p1h)
    rows_2p1h = np.arange(off_2p1h, nH)
    H[rows_2h1p, rows_2h1p] = eps[i_2h] + eps[j_2h] - eps[a_2h]
    H[rows_2p1h, rows_2p1h] = eps[a_2p] + eps[b_2p] - eps[i_2p]

    # ============================ Blocks C_2h1p / C_2p1h ============================
    H[off_2h1p:off_2p1h, off_2h1p:off_2p1h] += C_2h1p_block(s, nocc, np.arange(d['n2h1p']))
    H[off_2p1h:nH, off_2p1h:nH] += C_2p1h_block(s, nocc, np.arange(d['n2p1h']))

    return H


def C_2h1p_block(s, nocc, ket_idx, bra_idx=None):
    """First-order C_2h1p block for the ket configs (optionally bra-
    restricted) -- delta-times-bare-integral; GProxy when s.g is None."""
    g = s.g if s.g is not None else s._g_proxy()
    i_2h, j_2h, a_2h = s._configs_2h1p(nocc)
    bi = np.arange(len(i_2h)) if bra_idx is None else bra_idx
    I, J, A = i_2h[bi][:, None], j_2h[bi][:, None], a_2h[bi][:, None]
    K, L, C = i_2h[ket_idx][None, :], j_2h[ket_idx][None, :], a_2h[ket_idx][None, :]
    delta_ac = (A == C).astype(float)
    delta_ik, delta_jl = (I == K).astype(float), (J == L).astype(float)
    delta_il, delta_jk = (I == L).astype(float), (J == K).astype(float)
    return (
        -delta_ac * g[I, J, K, L]
        + delta_ik * g[C, J, A, L] + delta_jl * g[C, I, A, K]
        - delta_il * g[C, J, A, K] - delta_jk * g[C, I, A, L]
    )


def C_2p1h_block(s, nocc, ket_idx, bra_idx=None):
    """Mirror of C_2h1p_block (occ<->virt roles swapped)."""
    g = s.g if s.g is not None else s._g_proxy()
    i_2p, a_2p, b_2p = s._configs_2p1h(nocc)
    bi = np.arange(len(i_2p)) if bra_idx is None else bra_idx
    I, A, B = i_2p[bi][:, None], a_2p[bi][:, None], b_2p[bi][:, None]
    K, C, D = i_2p[ket_idx][None, :], a_2p[ket_idx][None, :], b_2p[ket_idx][None, :]
    delta_ik = (I == K).astype(float)
    delta_ac, delta_bd = (A == C).astype(float), (B == D).astype(float)
    delta_ad, delta_bc = (A == D).astype(float), (B == C).astype(float)
    return (
        delta_ik * g[A, B, C, D]
        - delta_ac * g[K, B, I, D] - delta_bd * g[K, A, I, C]
        + delta_ad * g[K, B, I, C] + delta_bc * g[K, A, I, D]
    )
