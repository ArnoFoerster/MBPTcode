"""Restricted dense supermatrix from the dense eri_chemist tensor.

The CSF-basis assembly (assemble) is written once here; adc_r_dense_df
feeds it the same named integral slices gathered from B_aa instead."""
import numpy as np

from src.SingleReference.EpsteinNesbet import (EpsteinNesbetDenominators,
                                               restricted_channel_shifts)
from src.SingleReference.ADC.adc_r_utils import build_g_blocks, csf_layout, u2_spin_amplitudes


def build_supermatrix(s, nocc, static_correction=None):
    """(nH, nH) restricted supermatrix at s.level, dense-integral route."""
    return assemble(s, nocc, static_correction, build_g_blocks(s.eri, nocc))


def assemble(s, nocc, static_correction, blk):
    """Shared CSF assembly consuming the 20 named integral slices."""
    eps = s.eps
    norb = s.norb
    (O, V, eps_o, eps_v, s32, s12, s2, s3, nI, nII, nIII, nIp, nIIp, nIIIp, nH, nP_o,
     nP_v, oI, oII, oIII, oIp, oIIp, oIIIp, iu_o, ju_o, iu_v, ju_v, oidx, vidx_local,
     vidx_abs) = csf_layout(s, nocc).take(
        'O V eps_o eps_v s32 s12 s2 s3 nI nII nIII nIp nIIp nIIIp nH nP_o nP_v '
        'oI oII oIII oIp oIIp oIIIp iu_o ju_o iu_v ju_v oidx vidx_local '
        'vidx_abs')

    H = np.zeros((nH, nH))

    # F block
    if static_correction is None:
        np.fill_diagonal(H[:norb, :norb], eps)
    else:
        H[:norb, :norb] = np.diag(eps) + static_correction

    # Common integral slices
    g_oovv = blk['g_oovv']
    g_vvoo = blk['g_vvoo']
    g_vv_pv = blk['g_vv_pv']
    g_vv_vp = blk['g_vv_vp']
    g_oo_po = blk['g_oo_po']
    g_oo_op = blk['g_oo_op']
    g_vopo = blk['g_vopo']
    g_voho = blk['g_voho']
    g_ovpv = blk['g_ovpv']
    g_ovvp = blk['g_ovvp']
    g_oo_pp = blk['g_oo_pp']
    g_vv_pp = blk['g_vv_pp']
    g_ii = blk['g_ii']
    g_aa = blk['g_aa']
    g_voov = blk['g_voov']
    g_vovo = blk['g_vovo']
    g_oooo = blk['g_oooo']
    g_vvvv = blk['g_vvvv']
    g_ovov = blk['g_ovov']
    g_ovvo = blk['g_ovvo']
    g_oovv_T = g_oovv.transpose(0, 1, 3, 2)    # g[i,k,a,c] at pos [i,k,c,a]
    g_vvoo_T = g_vvoo.transpose(0, 1, 3, 2)    # g[a,b,l,k] at pos [a,b,k,l]

    # BARE copies for u2_denom_dress (Epstein-Nesbet denominator dressing,
    # an independent mechanism from v->W vertex screening below -- must
    # always use the bare integral, captured here before any potential
    # W_chemist substitution overwrites g_oooo/g_vvvv/g_vovo/g_voov).
    g_oooo_bare, g_vvvv_bare = g_oooo, g_vvvv
    g_vovo_bare, g_voov_bare = g_vovo, g_voov

    # screened MCDE (arXiv:2603.27329): v -> static RPA W
    # The U^(2) coupling vertex is always BARE.
    g_ii_C = g_ii
    g_aa_C = g_aa

    # U^(1) coupling tensors: bare unless screen_coupling re-points them below
    g_ii_U, g_aa_U, g_oo_pp_U, g_vv_pp_U = g_ii, g_aa, g_oo_pp, g_vv_pp
    if s.W_chemist is not None:

        # physicist <pq|rs>_W
        Wp = s.W_chemist.transpose(0, 2, 1, 3)

        # hh ladder (C)
        g_oooo = Wp[:O, :O, :O, :O]

        # pp ladder (C)
        g_vvvv = Wp[O:, O:, O:, O:]

        # 2h1p ring exchange <ci|ak>
        g_vovo = Wp[O:, :O, O:, :O]

        # 2p1h ring exchange <ka|ic>
        g_ovov = Wp[:O, O:, :O, O:]

        # (O,norb,norb) g[i,i,:,:]_W
        g_ii_C = Wp[oidx, oidx]

        # (V,norb,norb) g[a,a,:,:]_W
        g_aa_C = Wp[vidx_abs, vidx_abs]
        if s.screen_coupling:
            # U^(1) COUPLING screening: v -> v - W/2, i.e. the 1/2(2V-W)
            # of upfolded PSD1. Separate names so the AMPLITUDE slices
            # (g_iicd/g_aakl) and the C^(1) blocks keep their own tensors.
            g_ii_U = g_ii - 0.5 * g_ii_C
            g_aa_U = g_aa - 0.5 * g_aa_C
            g_oo_pp_U = g_oo_pp - 0.5 * Wp[:O, :O, :, :]
            g_vv_pp_U = g_vv_pp - 0.5 * Wp[O:, O:, :, :]

    # ===================== Energy denominators =====================
    # EpsteinNesbetDenominators builds each layout on demand and, when
    # u2_denom_dress is on, gives the same-spin and opposite-spin cases
    # their OWN Epstein-Nesbet shift (they are diagonal elements of
    # different determinants and do not spin-adapt into one number).
    #   'ikca' [i,k,c,a] = eps_i+eps_k-eps_c-eps_a
    #   'ijcd' [i,j,c,d] = eps_c+eps_d-eps_i-eps_j
    #   'ackl' [a,c,k,l] = eps_a+eps_c-eps_k-eps_l
    #   'abkl' [a,b,k,l] = eps_k+eps_l-eps_a-eps_b
    _deltas = (None, None, None)
    if s.u2_denom_dress and not s.is_adc2x:
        _deltas = restricted_channel_shifts(
            s.u2_denom_dress, s.B_aa,
            g_oooo_bare, g_vvvv_bare, O, vidx_abs,
            g_vovo=g_vovo_bare, g_voov=g_voov_bare)
    dens = EpsteinNesbetDenominators(eps_o, eps_v, *_deltas)

    # ============================================================
    #                    U BLOCKS -- 2h1p
    # ============================================================

    # --- U_I: Type I (i,a), dim nI = O*V ---
    u_I = g_ii_U[:, :, O:].copy()  # (O,norb,V) = g[i,i,p,a] -- 1st-order piece

    if not s.is_adc2x:
        # i=j slice: the same-spin amplitude is antisymmetric in its
        # occupied pair and vanishes identically here, so this is PURELY
        # opposite-spin -- and its hh shift is the <ii|ii> diagonal.
        g_iicd = g_ii[:, O:, O:]     # (O,V,V) = g[i,i,c,d]
        X_icd = g_iicd / dens.build_icd()
        u_I -= 0.5 * (np.einsum('icd,cdpa->ipa', X_icd, g_vv_pv, optimize=True)
                      + np.einsum('icd,cdap->ipa', X_icd, g_vv_vp, optimize=True))

        # t_same/t_opp = the aaaa/abab T2^(1) amplitudes on 'ikca'; every
        # W/A object below is their unique linear combination.
        t_same, t_opp = u2_spin_amplitudes(g_oovv, g_oovv_T, dens, 'ikca')
        W1 = t_same - 2 * t_opp                     # -(g + g^T)/D
        u_I += np.einsum('ikca,cipk->ipa', W1, g_vopo, optimize=True)
        W2 = t_opp - 2 * t_same                     # (-g + 2g^T)/D
        u_I += np.einsum('ikca,cikp->ipa', W2, g_voho, optimize=True)

    U_I = u_I.transpose(1, 0, 2).reshape(norb, nI)
    H[:norb, oI:oI + nI] = U_I
    H[oI:oI + nI, :norb] = U_I.T

    # --- U_II: Type II (i<j,a), dim nII = nP_o*V, factor sqrt(3/2) ---
    g_ijpa = g_oo_pp_U[:, :, :, O:]                            # (O,O,norb,V) = g[i,j,p,a]
    g_ijap_T = g_oo_pp_U[:, :, O:, :].transpose(0, 1, 3, 2)    # (O,O,norb,V) = g[i,j,a,p]

    u_II = s32 * (g_ijpa - g_ijap_T)  # bare 1st-order piece

    if not s.is_adc2x:
        # pure opposite-spin (v/D), so only that denominator is built
        X_ijcd = g_oovv / dens.denom('ijcd', 'opp')  # (O,O,V,V)
        u_II -= s32 * (np.einsum('ijcd,cdpa->ijpa', X_ijcd, g_vv_pv, optimize=True)
                       - np.einsum('ijcd,cdap->ijpa', X_ijcd, g_vv_vp, optimize=True))

        A_anti = t_same                             # (g - g^T)/D
        A_mix = 2 * t_same - t_opp                  # (g - 2g^T)/D

        u_II -= s32 * np.einsum('ikca,cjpk->ijpa', A_anti, g_vopo, optimize=True)
        u_II += s32 * np.einsum('ikca,cjkp->ijpa', A_mix, g_voho, optimize=True)

        u_II += s32 * np.einsum('jkca,cipk->jipa', A_anti, g_vopo, optimize=True).transpose(1, 0, 2, 3)
        u_II -= s32 * np.einsum('jkca,cikp->jipa', A_mix, g_voho, optimize=True).transpose(1, 0, 2, 3)

    u_II_sel = u_II[iu_o, ju_o]  # (nP_o, norb, V)
    U_II = u_II_sel.transpose(1, 0, 2).reshape(norb, nII)
    H[:norb, oII:oII + nII] = U_II
    H[oII:oII + nII, :norb] = U_II.T

    # --- U_III: Type III (i<j,a), dim nIII = nP_o*V, factor sqrt(1/2) ---
    u_III = s12 * (g_ijpa + g_ijap_T)  # bare 1st-order piece

    if not s.is_adc2x:
        u_III -= s12 * (np.einsum('ijcd,cdpa->ijpa', X_ijcd, g_vv_pv, optimize=True)
                        + np.einsum('ijcd,cdap->ijpa', X_ijcd, g_vv_vp, optimize=True))

        W1_coul, W2_coul = W1, W2      # identical to U_I's own W1/W2

        u_III += s12 * np.einsum('jkca,cipk->jipa', W1_coul, g_vopo, optimize=True).transpose(1, 0, 2, 3)
        u_III += s12 * np.einsum('jkca,cikp->jipa', W2_coul, g_voho, optimize=True).transpose(1, 0, 2, 3)
        u_III += s12 * np.einsum('ikca,cjpk->ijpa', W1_coul, g_vopo, optimize=True)
        u_III += s12 * np.einsum('ikca,cjkp->ijpa', W2_coul, g_voho, optimize=True)

    u_III_sel = u_III[iu_o, ju_o]  # (nP_o, norb, V)
    U_III = u_III_sel.transpose(1, 0, 2).reshape(norb, nIII)
    H[:norb, oIII:oIII + nIII] = U_III
    H[oIII:oIII + nIII, :norb] = U_III.T

    # ============================================================
    #                    U BLOCKS -- 2p1h
    # ============================================================

    # --- U_Ip: Type I' (i,a), dim nIp = O*V ---
    u_Ip = g_aa_U[:, :, :O].transpose(2, 1, 0).copy()  # (O,norb,V) = g[a,a,p,i] at [i,p,a] -- 1st-order piece

    if not s.is_adc2x:
        # a=b slice: mirror of X_icd -- purely opposite-spin, pp shift is
        # the <aa|aa> diagonal.
        g_aakl = g_aa[:, :O, :O]     # (V,O,O) = g[a,a,k,l]
        X_akl = g_aakl / dens.build_akl()  # (V,O,O)

        g_klpi_sym = g_oo_po + g_oo_op.transpose(0, 1, 3, 2)  # g[k,l,p,i]+g[k,l,i,p]
        corr_kl_Ip = 0.5 * np.einsum('akl,klpi->api', X_akl, g_klpi_sym, optimize=True)
        u_Ip += corr_kl_Ip.transpose(2, 1, 0)

        # 'ackl' relabeled [a,c,k,i] = eps_a+eps_c-eps_i-eps_k
        tp_same, tp_opp = u2_spin_amplitudes(g_vvoo, g_vvoo_T, dens, 'ackl')
        W1p = 2 * tp_opp - tp_same                   # (g + g^T)/D
        W2p = 2 * tp_same - tp_opp                   # (g - 2g^T)/D

        u_Ip += np.einsum('acki,kapc->api', W1p, g_ovpv, optimize=True).transpose(2, 1, 0)
        u_Ip += np.einsum('acki,kacp->api', W2p, g_ovvp, optimize=True).transpose(2, 1, 0)

    U_Ip = u_Ip.transpose(1, 0, 2).reshape(norb, nIp)
    H[:norb, oIp:oIp + nIp] = U_Ip
    H[oIp:oIp + nIp, :norb] = U_Ip.T

    # --- U_IIp: Type II' (i,a<b), dim nIIp = O*nP_v, factor sqrt(3/2) ---
    g_abpi = g_vv_pp_U[:, :, :, :O]                            # (V,V,norb,O) = g[a,b,p,i]
    g_abip_T = g_vv_pp_U[:, :, :O, :].transpose(0, 1, 3, 2)    # (V,V,norb,O) = g[a,b,i,p]

    u_IIp_full = s32 * (g_abpi - g_abip_T)  # bare 1st-order piece

    if not s.is_adc2x:
        X_abkl = g_vvoo / dens.denom('abkl', 'opp')   # pure opposite-spin
        g_klpi_anti = g_oo_po - g_oo_op.transpose(0, 1, 3, 2)
        u_IIp_full += s32 * np.einsum('abkl,klpi->abpi', X_abkl, g_klpi_anti, optimize=True)

        A_anti_p = tp_same                            # (g - g^T)/D
        A_mix_p = W2p                                 # (g - 2g^T)/D

        u_IIp_full += s32 * np.einsum('acki,kbpc->abpi', A_anti_p, g_ovpv, optimize=True)
        u_IIp_full -= s32 * np.einsum('acki,kbcp->abpi', A_mix_p, g_ovvp, optimize=True)
        u_IIp_full -= s32 * np.einsum('bcki,kapc->bapi', A_anti_p, g_ovpv, optimize=True).transpose(1, 0, 2, 3)
        u_IIp_full += s32 * np.einsum('bcki,kacp->bapi', A_mix_p, g_ovvp, optimize=True).transpose(1, 0, 2, 3)

    # Config order: outer i, inner (a<b) -> (O, nP_v, norb) -> (norb, O*nP_v)
    u_IIp_sel = u_IIp_full[iu_v, ju_v].transpose(2, 0, 1)  # (O, nP_v, norb)
    U_IIp = u_IIp_sel.transpose(2, 0, 1).reshape(norb, nIIp)
    H[:norb, oIIp:oIIp + nIIp] = U_IIp
    H[oIIp:oIIp + nIIp, :norb] = U_IIp.T

    # --- U_IIIp: Type III' (i,a<b), dim nIIIp = O*nP_v, factor sqrt(1/2) ---
    u_IIIp_full = s12 * (g_abpi + g_abip_T)  # bare 1st-order piece

    if not s.is_adc2x:
        g_klpi_sym = g_oo_po + g_oo_op.transpose(0, 1, 3, 2)  # g[k,l,p,i]+g[k,l,i,p]
        u_IIIp_full += s12 * np.einsum('abkl,klpi->abpi', X_abkl, g_klpi_sym, optimize=True)
        u_IIIp_full += s12 * np.einsum('bcki,kapc->bapi', W1p, g_ovpv, optimize=True).transpose(1, 0, 2, 3)
        u_IIIp_full += s12 * np.einsum('bcki,kacp->bapi', W2p, g_ovvp, optimize=True).transpose(1, 0, 2, 3)
        u_IIIp_full += s12 * np.einsum('acki,kbpc->abpi', W1p, g_ovpv, optimize=True)
        u_IIIp_full += s12 * np.einsum('acki,kbcp->abpi', W2p, g_ovvp, optimize=True)

    u_IIIp_sel = u_IIIp_full[iu_v, ju_v].transpose(2, 0, 1)  # (O, nP_v, norb)
    U_IIIp = u_IIIp_sel.transpose(2, 0, 1).reshape(norb, nIIIp)
    H[:norb, oIIIp:oIIIp + nIIIp] = U_IIIp
    H[oIIIp:oIIIp + nIIIp, :norb] = U_IIIp.T

    # ============================================================
    #                    K DIAGONAL
    # ============================================================
    e_I = (2 * eps_o[:, None] - eps_v[None, :]).ravel()
    H[oI + np.arange(nI), oI + np.arange(nI)] = e_I

    e_II = (eps_o[iu_o, None] + eps_o[ju_o, None] - eps_v[None, :]).ravel()
    H[oII + np.arange(nII), oII + np.arange(nII)] = e_II
    H[oIII + np.arange(nIII), oIII + np.arange(nIII)] = e_II

    e_Ip = (2 * eps_v[None, :] - eps_o[:, None]).ravel()
    H[oIp + np.arange(nIp), oIp + np.arange(nIp)] = e_Ip

    e_IIp_pair = eps_v[iu_v] + eps_v[ju_v]  # (nP_v,)
    e_IIp = (e_IIp_pair[None, :] - eps_o[:, None]).ravel()
    H[oIIp + np.arange(nIIp), oIIp + np.arange(nIIp)] = e_IIp
    H[oIIIp + np.arange(nIIIp), oIIIp + np.arange(nIIIp)] = e_IIp

    # ============================================================
    #                    C BLOCKS -- 2h1p
    # ============================================================

    # --- C_I_I: Type I x Type I, shape (nI, nI) ---
    g_ii_kk = g_ii_C[:, oidx, oidx]  # (O,O) = g[i,i,k,k]

    T1 = -np.eye(V)[None, :, None, :] * g_ii_kk[:, None, :, None]

    g_ciia = g_voov[:, oidx, oidx, :]                          # (V,O,V) = g[c,i,i,a]
    g_ciai = g_vovo[:, oidx, :, oidx].transpose(1, 0, 2)       # (V,O,V) = g[c,i,a,i]
    inner_I = -(g_ciia - 2 * g_ciai).transpose(1, 2, 0)  # (O,V,V) = (i,a,c)
    T2 = np.eye(O)[:, None, :, None] * inner_I[:, :, None, :]

    C_I_I = (T1 + T2).reshape(nI, nI)
    H[oI:oI + nI, oI:oI + nI] += C_I_I

    # --- C_II_II: Type II x Type II, shape (nII, nII) ---
    delta_jl = (ju_o[:, None] == ju_o[None, :]).astype(float)
    delta_jk = (ju_o[:, None] == iu_o[None, :]).astype(float)
    delta_ik = (iu_o[:, None] == iu_o[None, :]).astype(float)
    delta_il = (iu_o[:, None] == ju_o[None, :]).astype(float)

    g_oooo_anti = g_oooo - g_oooo.transpose(0, 1, 3, 2)
    g_pp_anti = g_oooo_anti[iu_o[:, None], ju_o[:, None], iu_o[None, :], ju_o[None, :]]  # (nP_o,nP_o)
    C_II_II_4d = np.zeros((nP_o, V, nP_o, V))
    C_II_II_4d -= np.eye(V)[None, :, None, :] * g_pp_anti[:, None, :, None]

    def _sel_voov(occ1_arr, occ2_arr):
        """g[c, occ1[r], occ2[s], a] -> (r, a, s, c)"""
        sel = g_voov[np.ix_(np.arange(V), occ1_arr, occ2_arr, np.arange(V))]
        return sel.transpose(1, 3, 2, 0)

    def _sel_vovo(occ1_arr, occ2_arr):
        """g[c, occ1[r], a, occ2[s]] -> (r, a, s, c)"""
        sel = g_vovo[np.ix_(np.arange(V), occ1_arr, np.arange(V), occ2_arr)]
        return sel.transpose(1, 2, 3, 0)

    C_II_II_4d += delta_jl[:, None, :, None] * _sel_vovo(iu_o, iu_o)
    C_II_II_4d -= delta_jk[:, None, :, None] * _sel_vovo(iu_o, ju_o)
    C_II_II_4d -= 1.5 * delta_ik[:, None, :, None] * _sel_voov(ju_o, ju_o)
    C_II_II_4d += 1.5 * delta_il[:, None, :, None] * _sel_voov(ju_o, iu_o)
    C_II_II_4d -= delta_il[:, None, :, None] * _sel_vovo(ju_o, iu_o)
    C_II_II_4d += delta_ik[:, None, :, None] * _sel_vovo(ju_o, ju_o)
    C_II_II_4d -= 1.5 * delta_jl[:, None, :, None] * _sel_voov(iu_o, iu_o)
    C_II_II_4d += 1.5 * delta_jk[:, None, :, None] * _sel_voov(iu_o, ju_o)

    C_II_II = C_II_II_4d.reshape(nII, nII)
    H[oII:oII + nII, oII:oII + nII] += C_II_II

    # --- C_III_III: Type III x Type III, shape (nIII, nIII) ---
    C_III_III_4d = np.zeros((nP_o, V, nP_o, V))

    g_pp_sym1 = g_oooo[iu_o[:, None], ju_o[:, None], iu_o[None, :], ju_o[None, :]]
    C_III_III_4d -= np.eye(V)[None, :, None, :] * g_pp_sym1[:, None, :, None]
    C_III_III_4d += delta_ik[:, None, :, None] * _sel_vovo(ju_o, ju_o)
    C_III_III_4d += delta_jl[:, None, :, None] * _sel_vovo(iu_o, iu_o)
    C_III_III_4d -= 0.5 * delta_ik[:, None, :, None] * _sel_voov(ju_o, ju_o)
    C_III_III_4d -= 0.5 * delta_jl[:, None, :, None] * _sel_voov(iu_o, iu_o)

    g_pp_sym2 = g_oooo[iu_o[:, None], ju_o[:, None], ju_o[None, :], iu_o[None, :]]
    C_III_III_4d -= np.eye(V)[None, :, None, :] * g_pp_sym2[:, None, :, None]
    C_III_III_4d += delta_il[:, None, :, None] * _sel_vovo(ju_o, iu_o)
    C_III_III_4d += delta_jk[:, None, :, None] * _sel_vovo(iu_o, ju_o)
    C_III_III_4d -= 0.5 * delta_il[:, None, :, None] * _sel_voov(ju_o, iu_o)
    C_III_III_4d -= 0.5 * delta_jk[:, None, :, None] * _sel_voov(iu_o, ju_o)

    C_III_III = C_III_III_4d.reshape(nIII, nIII)
    H[oIII:oIII + nIII, oIII:oIII + nIII] += C_III_III

    # --- C_I_II: Type I x Type II, shape (nI, nII) + transpose ---
    delta_i_ir = (oidx[:, None] == iu_o[None, :]).astype(float)  # (O, nP_o) = delta(i, k_r)
    delta_i_jr = (oidx[:, None] == ju_o[None, :]).astype(float)  # (O, nP_o) = delta(i, l_r)

    g_ci_jr_a_T = g_voov[:, :, ju_o, :].transpose(1, 3, 2, 0)  # (O,V,nP_o,V) = (i,a,r,c)
    g_ci_ir_a_T = g_voov[:, :, iu_o, :].transpose(1, 3, 2, 0)  # (O,V,nP_o,V)

    C_I_II_4d = -s32 * (delta_i_ir[:, None, :, None] * g_ci_jr_a_T
                        - delta_i_jr[:, None, :, None] * g_ci_ir_a_T)

    C_I_II = C_I_II_4d.reshape(nI, nII)
    H[oI:oI + nI, oII:oII + nII] = C_I_II
    H[oII:oII + nII, oI:oI + nI] = C_I_II.T

    # --- C_I_III: Type I x Type III, shape (nI, nIII) + transpose ---
    g_ii_pair = g_ii_C[:, iu_o, ju_o]  # (O, nP_o) = g[i,i,k_r,l_r]

    C_I_III_4d = np.zeros((O, V, nP_o, V))
    C_I_III_4d -= s2 * np.eye(V)[None, :, None, :] * g_ii_pair[:, None, :, None]

    g_cia_lr = g_vovo[:, :, :, ju_o].transpose(1, 2, 3, 0)     # (O,V,nP_o,V) = (i,a,r,c)
    g_ci_lr_a2 = g_voov[:, :, ju_o, :].transpose(1, 3, 2, 0)   # (O,V,nP_o,V) = (i,a,r,c)
    C_I_III_4d += s2 * delta_i_ir[:, None, :, None] * (g_cia_lr - 0.5 * g_ci_lr_a2)

    g_cia_kr = g_vovo[:, :, :, iu_o].transpose(1, 2, 3, 0)
    g_ci_kr_a2 = g_voov[:, :, iu_o, :].transpose(1, 3, 2, 0)
    C_I_III_4d += s2 * delta_i_jr[:, None, :, None] * (g_cia_kr - 0.5 * g_ci_kr_a2)

    C_I_III = C_I_III_4d.reshape(nI, nIII)
    H[oI:oI + nI, oIII:oIII + nIII] = C_I_III
    H[oIII:oIII + nIII, oI:oI + nI] = C_I_III.T

    # --- C_II_III / C_III_II: shape (nII, nIII) and its transpose block ---
    C_II_III_4d = np.zeros((nP_o, V, nP_o, V))
    C_II_III_4d -= 0.5 * s3 * delta_ik[:, None, :, None] * _sel_voov(ju_o, ju_o)
    C_II_III_4d += 0.5 * s3 * delta_jl[:, None, :, None] * _sel_voov(iu_o, iu_o)
    C_II_III_4d -= 0.5 * s3 * delta_il[:, None, :, None] * _sel_voov(ju_o, iu_o)
    C_II_III_4d += 0.5 * s3 * delta_jk[:, None, :, None] * _sel_voov(iu_o, ju_o)

    C_II_III = C_II_III_4d.reshape(nII, nIII)
    H[oII:oII + nII, oIII:oIII + nIII] = C_II_III

    C_III_II_4d = np.zeros((nP_o, V, nP_o, V))
    C_III_II_4d -= 0.5 * s3 * delta_ik[:, None, :, None] * _sel_voov(ju_o, ju_o)
    C_III_II_4d += 0.5 * s3 * delta_jl[:, None, :, None] * _sel_voov(iu_o, iu_o)
    C_III_II_4d += 0.5 * s3 * delta_il[:, None, :, None] * _sel_voov(ju_o, iu_o)
    C_III_II_4d -= 0.5 * s3 * delta_jk[:, None, :, None] * _sel_voov(iu_o, ju_o)

    C_III_II = C_III_II_4d.reshape(nIII, nII)
    H[oIII:oIII + nIII, oII:oII + nII] = C_III_II

    # ============================================================
    #                    C BLOCKS -- 2p1h (occ<->virt mirror of above)
    # ============================================================

    def _sel_ovov_v(virt1_arr, virt2_arr):
        """g[k, virt1[r], i, virt2[s]] -> (i, r, k, s)"""
        sel = g_ovov[np.ix_(np.arange(O), virt1_arr, np.arange(O), virt2_arr)]
        return sel.transpose(2, 1, 0, 3)

    def _sel_ovvo_v(virt1_arr, virt2_arr):
        """g[k, virt1[r], virt2[s], i] -> (i, r, k, s)"""
        sel = g_ovvo[np.ix_(np.arange(O), virt1_arr, virt2_arr, np.arange(O))]
        return sel.transpose(3, 1, 0, 2)

    # --- C_Ip_Ip: Type I' x Type I', shape (nIp, nIp) ---
    g_aabb = g_aa_C[:, vidx_abs, vidx_abs]  # (V,V) = g[a,a,c,c]

    T1p = np.eye(O)[:, None, :, None] * g_aabb[None, :, None, :]

    g_kaai = g_ovvo[:, vidx_local, vidx_local, :]                          # (O,V,O) = g[k,a,a,i]
    g_kaia = g_ovov[:, vidx_local, :, vidx_local].transpose(1, 0, 2)       # (O,V,O) = g[k,a,i,a]
    inner_Ip_iak = (g_kaai - 2 * g_kaia).transpose(2, 1, 0)  # (O,V,O) = (i,a,k)
    T2p = np.eye(V)[None, :, None, :] * inner_Ip_iak[:, :, :, None]

    C_Ip_Ip = (T1p + T2p).reshape(nIp, nIp)
    H[oIp:oIp + nIp, oIp:oIp + nIp] += C_Ip_Ip

    # --- C_IIp_IIp: Type II' x Type II', shape (nIIp, nIIp) ---
    delta_bd = (ju_v[:, None] == ju_v[None, :]).astype(float)
    delta_bc = (ju_v[:, None] == iu_v[None, :]).astype(float)
    delta_ac = (iu_v[:, None] == iu_v[None, :]).astype(float)
    delta_ad = (iu_v[:, None] == ju_v[None, :]).astype(float)

    g_vvvv_anti = g_vvvv - g_vvvv.transpose(0, 1, 3, 2)
    g_pp_v_anti = g_vvvv_anti[iu_v[:, None], ju_v[:, None], iu_v[None, :], ju_v[None, :]]
    C_IIp_IIp = np.zeros((O, nP_v, O, nP_v))
    C_IIp_IIp += np.eye(O)[:, None, :, None] * g_pp_v_anti[None, :, None, :]

    C_IIp_IIp -= delta_bd[None, :, None, :] * _sel_ovov_v(iu_v, iu_v)
    C_IIp_IIp += delta_bc[None, :, None, :] * _sel_ovov_v(iu_v, ju_v)
    C_IIp_IIp += 1.5 * delta_ac[None, :, None, :] * _sel_ovvo_v(ju_v, ju_v)
    C_IIp_IIp -= 1.5 * delta_ad[None, :, None, :] * _sel_ovvo_v(ju_v, iu_v)
    C_IIp_IIp += delta_ad[None, :, None, :] * _sel_ovov_v(ju_v, iu_v)
    C_IIp_IIp -= delta_ac[None, :, None, :] * _sel_ovov_v(ju_v, ju_v)
    C_IIp_IIp -= 1.5 * delta_bc[None, :, None, :] * _sel_ovvo_v(iu_v, ju_v)
    C_IIp_IIp += 1.5 * delta_bd[None, :, None, :] * _sel_ovvo_v(iu_v, iu_v)

    H[oIIp:oIIp + nIIp, oIIp:oIIp + nIIp] += C_IIp_IIp.reshape(nIIp, nIIp)

    # --- C_IIIp_IIIp: Type III' x Type III', shape (nIIIp, nIIIp) ---
    C_IIIp_IIIp = np.zeros((O, nP_v, O, nP_v))

    g_pp_v_sym1 = g_vvvv[iu_v[:, None], ju_v[:, None], iu_v[None, :], ju_v[None, :]]
    g_pp_v_sym2 = g_vvvv[iu_v[:, None], ju_v[:, None], ju_v[None, :], iu_v[None, :]]

    C_IIIp_IIIp += np.eye(O)[:, None, :, None] * g_pp_v_sym1[None, :, None, :]
    C_IIIp_IIIp -= delta_ac[None, :, None, :] * _sel_ovov_v(ju_v, ju_v)
    C_IIIp_IIIp -= delta_bd[None, :, None, :] * _sel_ovov_v(iu_v, iu_v)
    C_IIIp_IIIp += 0.5 * delta_ac[None, :, None, :] * _sel_ovvo_v(ju_v, ju_v)
    C_IIIp_IIIp += 0.5 * delta_bd[None, :, None, :] * _sel_ovvo_v(iu_v, iu_v)
    C_IIIp_IIIp += np.eye(O)[:, None, :, None] * g_pp_v_sym2[None, :, None, :]
    C_IIIp_IIIp -= delta_ad[None, :, None, :] * _sel_ovov_v(ju_v, iu_v)
    C_IIIp_IIIp -= delta_bc[None, :, None, :] * _sel_ovov_v(iu_v, ju_v)
    C_IIIp_IIIp += 0.5 * delta_ad[None, :, None, :] * _sel_ovvo_v(ju_v, iu_v)
    C_IIIp_IIIp += 0.5 * delta_bc[None, :, None, :] * _sel_ovvo_v(iu_v, ju_v)

    H[oIIIp:oIIIp + nIIIp, oIIIp:oIIIp + nIIIp] += C_IIIp_IIIp.reshape(nIIIp, nIIIp)

    # --- C_Ip_IIp: Type I' x Type II', shape (nIp, nIIp) + transpose ---
    delta_a_ar = (vidx_local[:, None] == iu_v[None, :]).astype(float)  # (V, nP_v) = delta(a, c_r)
    delta_a_br = (vidx_local[:, None] == ju_v[None, :]).astype(float)  # (V, nP_v) = delta(a, d_r)

    g_ka_dr_i = g_ovvo[:, :, ju_v, :]  # (O, V, nP_v, O)
    g_ka_cr_i = g_ovvo[:, :, iu_v, :]  # (O, V, nP_v, O)

    C_Ip_IIp_4d = (s32 * delta_a_ar[None, :, None, :] * g_ka_dr_i.transpose(3, 1, 0, 2)
                   - s32 * delta_a_br[None, :, None, :] * g_ka_cr_i.transpose(3, 1, 0, 2))

    C_Ip_IIp = C_Ip_IIp_4d.reshape(nIp, nIIp)
    H[oIp:oIp + nIp, oIIp:oIIp + nIIp] = C_Ip_IIp
    H[oIIp:oIIp + nIIp, oIp:oIp + nIp] = C_Ip_IIp.T

    # --- C_Ip_IIIp: Type I' x Type III', shape (nIp, nIIIp) + transpose ---
    g_aa_cd = g_aa_C[:, vidx_abs[iu_v], vidx_abs[ju_v]]  # (V, nP_v) = g[a,a,c_r,d_r]

    C_Ip_IIIp_4d = np.zeros((O, V, O, nP_v))
    C_Ip_IIIp_4d += s2 * np.eye(O)[:, None, :, None] * g_aa_cd[None, :, None, :]

    g_kai_dr_T = g_ovov[:, :, :, ju_v].transpose(2, 1, 0, 3)
    g_ka_dr_i_T = g_ka_dr_i.transpose(3, 1, 0, 2)
    C_Ip_IIIp_4d += s2 * delta_a_ar[None, :, None, :] * (-g_kai_dr_T + 0.5 * g_ka_dr_i_T)

    g_kai_cr_T = g_ovov[:, :, :, iu_v].transpose(2, 1, 0, 3)
    g_ka_cr_i_T = g_ka_cr_i.transpose(3, 1, 0, 2)
    C_Ip_IIIp_4d += s2 * delta_a_br[None, :, None, :] * (-g_kai_cr_T + 0.5 * g_ka_cr_i_T)

    C_Ip_IIIp = C_Ip_IIIp_4d.reshape(nIp, nIIIp)
    H[oIp:oIp + nIp, oIIIp:oIIIp + nIIIp] = C_Ip_IIIp
    H[oIIIp:oIIIp + nIIIp, oIp:oIp + nIp] = C_Ip_IIIp.T

    # --- C_IIp_IIIp / C_IIIp_IIp: shape (nIIp, nIIIp) and its transpose block ---
    C_IIp_IIIp = np.zeros((O, nP_v, O, nP_v))
    C_IIp_IIIp += 0.5 * s3 * delta_ac[None, :, None, :] * _sel_ovvo_v(ju_v, ju_v)
    C_IIp_IIIp -= 0.5 * s3 * delta_bd[None, :, None, :] * _sel_ovvo_v(iu_v, iu_v)
    C_IIp_IIIp += 0.5 * s3 * delta_ad[None, :, None, :] * _sel_ovvo_v(ju_v, iu_v)
    C_IIp_IIIp -= 0.5 * s3 * delta_bc[None, :, None, :] * _sel_ovvo_v(iu_v, ju_v)

    H[oIIp:oIIp + nIIp, oIIIp:oIIIp + nIIIp] = C_IIp_IIIp.reshape(nIIp, nIIIp)

    C_IIIp_IIp = np.zeros((O, nP_v, O, nP_v))
    C_IIIp_IIp += 0.5 * s3 * delta_ac[None, :, None, :] * _sel_ovvo_v(ju_v, ju_v)
    C_IIIp_IIp -= 0.5 * s3 * delta_bd[None, :, None, :] * _sel_ovvo_v(iu_v, iu_v)
    C_IIIp_IIp -= 0.5 * s3 * delta_ad[None, :, None, :] * _sel_ovvo_v(ju_v, iu_v)
    C_IIIp_IIp += 0.5 * s3 * delta_bc[None, :, None, :] * _sel_ovvo_v(iu_v, ju_v)

    H[oIIIp:oIIIp + nIIIp, oIIp:oIIp + nIIp] = C_IIIp_IIp.reshape(nIIIp, nIIp)

    return H
