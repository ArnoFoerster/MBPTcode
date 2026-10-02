"""Restricted matrix-free (sigma-vector) operator from dense integrals.

U blocks are materialized once (dense, cheap at O(norb*nH)); the fourteen
C sub-blocks are applied per matvec. W_chemist screening (incl.
screen_coupling) lives here -- the dense screening route."""
import math
import numpy as np

from src.SingleReference.EpsteinNesbet import (EpsteinNesbetDenominators,
                                               restricted_channel_shifts)
from src.SingleReference.ADC.adc_r_utils import (_build_u_blocks_unstreamed,
                                                 _no_u_blocks)


def build_operator(s, nocc, static_correction=None, couplings=True):
    """(aop, diag, dims) for the dense-integral route. couplings=False: U = 0,
    i.e. F and the satellite block K + C only."""
    eps, eri_chemist = s.eps, s.eri
    norb = s.norb
    O, V = nocc, norb - nocc
    eps_o, eps_v = eps[:O], eps[O:]

    s32 = math.sqrt(1.5)
    s12 = math.sqrt(0.5)
    s2 = math.sqrt(2.0)
    s3 = math.sqrt(3.0)

    d = s.dimensions(nocc)
    nI, nII, nIII, n2h1p = d['nI'], d['nII'], d['nIII'], d['n2h1p']
    nIp, nIIp, nIIIp = d['nIp'], d['nIIp'], d['nIIIp']
    nH = d['nH']
    nP_o = O * (O - 1) // 2
    nP_v = V * (V - 1) // 2

    oI = norb
    oII = oI + nI
    oIII = oII + nII
    oIp = norb + n2h1p
    oIIp = oIp + nIp
    oIIIp = oIIp + nIIp

    iu_o, ju_o = np.triu_indices(O, k=1)
    iu_v, ju_v = np.triu_indices(V, k=1)
    oidx = np.arange(O)
    vidx_local = np.arange(V)
    vidx_abs = np.arange(O, norb)

    # ===================== F block =====================
    if static_correction is None:
        F = np.diag(eps)
    else:
        F = np.diag(eps) + static_correction

    # Common integral slices (identical to the dense build)
    g = s.eri.transpose(0, 2, 1, 3)
    g_oovv = g[:O, :O, O:, O:]
    g_vvoo = g[O:, O:, :O, :O]
    g_vv_pv = g[O:, O:, :, O:]
    g_vv_vp = g[O:, O:, O:, :]
    g_oo_po = g[:O, :O, :, :O]
    g_oo_op = g[:O, :O, :O, :]
    g_vopo = g[O:, :O, :, :O]
    g_voho = g[O:, :O, :O, :]
    g_ovpv = g[:O, O:, :, O:]
    g_ovvp = g[:O, O:, O:, :]
    g_oo_pp = g[:O, :O, :, :]
    g_vv_pp = g[O:, O:, :, :]
    g_ii = g[oidx, oidx]
    g_aa = g[vidx_abs, vidx_abs]
    g_voov = g[O:, :O, :O, O:]
    g_vovo = g[O:, :O, O:, :O]
    g_oooo = g[:O, :O, :O, :O]
    g_vvvv = g[O:, O:, O:, O:]
    g_ovov = g[:O, O:, :O, O:]
    g_ovvo = g[:O, O:, O:, :O]

    g_oovv_T = g_oovv.transpose(0, 1, 3, 2)
    g_vvoo_T = g_vvoo.transpose(0, 1, 3, 2)

    # BARE copies for u2_denom_dress (captured before any v->W substitution)
    g_oooo_bare = g_oooo
    g_vvvv_bare = g_vvvv
    g_vovo_bare, g_voov_bare = g_vovo, g_voov

    # screened MCDE C^(1) substitution (dense W) + optional U^(1) coupling
    # screening -- see the dense build's identical section.
    g_ii_C = g_ii
    g_aa_C = g_aa
    g_ii_U = g_ii
    g_aa_U = g_aa
    g_oo_pp_U = g_oo_pp
    g_vv_pp_U = g_vv_pp
    if s.W_chemist is not None:
        Wp = s.W_chemist.transpose(0, 2, 1, 3)
        g_oooo = Wp[:O, :O, :O, :O]
        g_vvvv = Wp[O:, O:, O:, O:]

        # 2h1p ring exchange <ci|ak>
        g_vovo = Wp[O:, :O, O:, :O]

        # 2p1h ring exchange <ka|ic>
        g_ovov = Wp[:O, O:, :O, O:]
        g_ii_C = Wp[oidx, oidx]
        g_aa_C = Wp[vidx_abs, vidx_abs]

        # U^(1): v -> v - W/2, see base.py
        if s.screen_coupling:
            g_ii_U = g_ii - 0.5 * g_ii_C
            g_aa_U = g_aa - 0.5 * g_aa_C
            g_oo_pp_U = g_oo_pp - 0.5 * Wp[:O, :O, :, :]
            if g_vv_pp_U is not None:
                g_vv_pp_U = g_vv_pp_U - 0.5 * Wp[O:, O:, :, :]

    g_ii_kk = g_ii_C[:, oidx, oidx]
    g_ii_pair = g_ii_C[:, iu_o, ju_o]
    g_aabb = g_aa_C[:, vidx_abs, vidx_abs]
    g_aa_cd = g_aa_C[:, vidx_abs[iu_v], vidx_abs[ju_v]]
    # See build_supermatrix's identical block for the layout key and the
    # spin-resolved EN rationale (kept in sync deliberately).
    _deltas = (None, None, None)
    if s.u2_denom_dress and not s._is_adc2x:
        _deltas = restricted_channel_shifts(
            s.u2_denom_dress, s.B_aa,
            g_oooo_bare, g_vvvv_bare, O, vidx_abs,
            g_vovo=g_vovo_bare, g_voov=g_voov_bare)
    dens = EpsteinNesbetDenominators(eps_o, eps_v, *_deltas)

    # ============ U blocks: dense materialized builder ============
    _u_ints = dict(
        g_ii_U=g_ii_U, g_ii=g_ii, g_aa_U=g_aa_U, g_aa=g_aa,
        g_oo_po=g_oo_po, g_oo_op=g_oo_op,
        g_oovv=g_oovv, g_oovv_T=g_oovv_T, g_vvoo=g_vvoo, g_vvoo_T=g_vvoo_T,
        g_vopo=g_vopo, g_voho=g_voho, g_ovpv=g_ovpv, g_ovvp=g_ovvp,
        g_vv_pv=g_vv_pv, g_vv_vp=g_vv_vp,
        g_oo_pp_U=g_oo_pp_U, g_vv_pp_U=g_vv_pp_U)
    if couplings:
        (U_I, U_Ip, apply_U_2h1p_fwd, apply_U_2h1p_adj,
         apply_U_2p1h_fwd, apply_U_2p1h_adj) = _build_u_blocks_unstreamed(
            None, O, V, norb, eps_o, eps_v, dens,
            None, None, None, None, None, None,
            iu_o, ju_o, iu_v, ju_v, _u_ints,
            is_adc2x=s._is_adc2x, use_materialized=True, w_chemist=s.W_chemist)
    else:
        (U_I, U_Ip, apply_U_2h1p_fwd, apply_U_2h1p_adj,
         apply_U_2p1h_fwd, apply_U_2p1h_adj) = _no_u_blocks(norb, O, V, nP_o, nP_v)
    # ===================== K diagonals =====================
    e_I = (2 * eps_o[:, None] - eps_v[None, :]).ravel()
    e_II = (eps_o[iu_o, None] + eps_o[ju_o, None] - eps_v[None, :]).ravel()
    e_III = e_II.copy()
    e_Ip = (2 * eps_v[None, :] - eps_o[:, None]).ravel()
    e_IIp_pair = eps_v[iu_v] + eps_v[ju_v]
    e_IIp = (e_IIp_pair[None, :] - eps_o[:, None]).ravel()
    e_IIIp = e_IIp.copy()

    # ==================== C-block sigma-vector pieces ====================
    g_ciia = g_voov[:, oidx, oidx, :]
    g_ciai = g_vovo[:, oidx, :, oidx].transpose(1, 0, 2)
    inner_I = -(g_ciia - 2 * g_ciai).transpose(1, 2, 0)

    def apply_C_I_I(Vmat):
        term1 = -g_ii_kk @ Vmat
        term2 = np.einsum('iac,ic->ia', inner_I, Vmat, optimize=True)
        return term1 + term2

    g_oooo_anti = g_oooo - g_oooo.transpose(0, 1, 3, 2)
    def unfold_pair(Vmat, iu, ju, O_, Vv):
        Vfull = np.zeros((O_, O_, Vv))
        Vfull[iu, ju, :] = Vmat
        Vfull[ju, iu, :] = -Vmat
        return Vfull

    def upper_pair(Vmat, iu, ju, O_, Vv):
        Vup = np.zeros((O_, O_, Vv))
        Vup[iu, ju, :] = Vmat
        return Vup
    M1a = g_vovo.transpose(1, 2, 3, 0)
    M1b = g_voov.transpose(1, 3, 2, 0)
    M1 = -M1a + 1.5 * M1b
    M2 = -1.5 * M1b + M1a
    bracket3 = M1a - 0.5 * M1b

    # (O,V,O,V) = g[c,i,x,a] -> [i,a,x,c]
    G_I_II = g_voov.transpose(1, 3, 2, 0)
    bracket2 = M1a - 0.5 * M1b  # (i,a,x,c) -- same functional form as bracket3

    def apply_C_II_II(Vmat):
        Vfull2 = unfold_pair(Vmat, iu_o, ju_o, O, V)
        T0 = np.einsum('ijkl,kla->ija', g_oooo_anti, Vfull2, optimize=True)
        T1 = np.einsum('iamb,kmb->iak', M1, Vfull2, optimize=True)
        T2 = np.einsum('jamb,kmb->jak', M2, Vfull2, optimize=True)
        return -0.5 * T0[iu_o, ju_o, :] + T1[iu_o, :, ju_o] + T2[ju_o, :, iu_o]

    def apply_C_III_III(Vmat):
        Vupper = upper_pair(Vmat, iu_o, ju_o, O, V)
        term_sym1 = -np.einsum('ijkl,kla->ija', g_oooo, Vupper, optimize=True)
        term_sym2 = -np.einsum('ijkl,kla->ija', g_oooo.transpose(0, 1, 3, 2), Vupper, optimize=True)
        BV1 = np.einsum('xamb,Kmb->Kxa', bracket3, Vupper, optimize=True)
        BV2 = np.einsum('xamb,mKb->Kxa', bracket3, Vupper, optimize=True)
        groupA = BV1[iu_o, ju_o, :] + BV2[iu_o, ju_o, :]
        groupB = BV1[ju_o, iu_o, :] + BV2[ju_o, iu_o, :]
        return term_sym1[iu_o, ju_o, :] + term_sym2[iu_o, ju_o, :] + groupA + groupB

    def apply_C_I_II(V_II):
        Vupper = upper_pair(V_II, iu_o, ju_o, O, V)
        term1 = np.einsum('ialc,ilc->ia', G_I_II, Vupper, optimize=True)
        term2 = np.einsum('iaxc,xic->ia', G_I_II, Vupper, optimize=True)
        return -s32 * (term1 - term2)

    def apply_C_II_I(V_I):
        W = np.einsum('iaxc,ia->ixc', G_I_II, V_I, optimize=True)
        return -s32 * (W[iu_o, ju_o, :] - W[ju_o, iu_o, :])

    def apply_C_I_III(V_III):
        Vupper = upper_pair(V_III, iu_o, ju_o, O, V)
        term0 = -s2 * (g_ii_pair @ V_III)
        term1 = s2 * np.einsum('ialc,ilc->ia', bracket2, Vupper, optimize=True)
        term2 = s2 * np.einsum('iaxc,xic->ia', bracket2, Vupper, optimize=True)
        return term0 + term1 + term2

    def apply_C_III_I(V_I):
        Wb = np.einsum('iaxc,ia->ixc', bracket2, V_I, optimize=True)
        term0 = -s2 * (g_ii_pair.T @ V_I)
        return term0 + s2 * Wb[iu_o, ju_o, :] + s2 * Wb[ju_o, iu_o, :]

    def Tupper_Tlower_o(Vupper):
        Tu = np.einsum('xanc,Knc->Kxa', G_I_II, Vupper, optimize=True)
        Tl = np.einsum('xamc,mKc->Kxa', G_I_II, Vupper, optimize=True)
        return Tu, Tl

    def apply_C_II_III(V_III):
        Vupper = upper_pair(V_III, iu_o, ju_o, O, V)
        Tu, Tl = Tupper_Tlower_o(Vupper)
        return 0.5 * s3 * (Tu[ju_o, iu_o, :] - Tu[iu_o, ju_o, :] + Tl[ju_o, iu_o, :] - Tl[iu_o, ju_o, :])

    def apply_C_III_II(V_II):
        Vupper = upper_pair(V_II, iu_o, ju_o, O, V)
        Tu, Tl = Tupper_Tlower_o(Vupper)
        return (-0.5 * s3 * (Tu[iu_o, ju_o, :] + Tu[ju_o, iu_o, :])
                + 0.5 * s3 * (Tl[ju_o, iu_o, :] + Tl[iu_o, ju_o, :]))

    g_kaai = g_ovvo[:, vidx_local, vidx_local, :]
    g_kaia = g_ovov[:, vidx_local, :, vidx_local].transpose(1, 0, 2)
    inner_Ip = (g_kaai - 2 * g_kaia).transpose(2, 1, 0)

    def apply_C_Ip_Ip(Vmat):
        term1 = np.einsum('ic,ac->ia', Vmat, g_aabb, optimize=True)
        term2 = np.einsum('iak,ka->ia', inner_Ip, Vmat, optimize=True)
        return term1 + term2
    # --- App1..4 building blocks for the virt-pair (2p1h) C-blocks ---
    def make_upper_k(Vmat, iu, ju, O_, Vv):
        Vup = np.zeros((O_, Vv, Vv))
        Vup[:, iu, ju] = Vmat
        return Vup
    def compute_Apps(Vupper_k):
        App1 = np.einsum('kXin,kYn->iXY', g_ovov, Vupper_k, optimize=True)
        App2 = np.einsum('kXni,kYn->iXY', g_ovvo, Vupper_k, optimize=True)
        App3 = np.einsum('kXim,kmY->iXY', g_ovov, Vupper_k, optimize=True)
        App4 = np.einsum('kXmi,kmY->iXY', g_ovvo, Vupper_k, optimize=True)
        return App1, App2, App3, App4

    def apply_C_Ip_IIp(V_IIp):
        Vupper_kp = make_upper_k(V_IIp, iu_v, ju_v, O, V)
        term1 = s32 * np.einsum('kaDi,kaD->ia', g_ovvo, Vupper_kp, optimize=True)
        term2 = s32 * np.einsum('kaCi,kCa->ia', g_ovvo, Vupper_kp, optimize=True)
        return term1 - term2

    def apply_C_IIp_Ip(V_Ip):
        Wc = np.einsum('kXYi,iX->kXY', g_ovvo, V_Ip, optimize=True)
        C_r, D_r = iu_v, ju_v
        return s32 * Wc[:, C_r, D_r] - s32 * Wc[:, D_r, C_r]

    def apply_C_Ip_IIIp(V_IIIp):
        Vupper_kp3 = make_upper_k(V_IIIp, iu_v, ju_v, O, V)
        term0 = s2 * (V_IIIp @ g_aa_cd.T)
        term1 = s2 * (-np.einsum('kaiD,kaD->ia', g_ovov, Vupper_kp3, optimize=True)
                      + 0.5 * np.einsum('kaDi,kaD->ia', g_ovvo, Vupper_kp3, optimize=True))
        term2 = s2 * (-np.einsum('kaiC,kCa->ia', g_ovov, Vupper_kp3, optimize=True)
                      + 0.5 * np.einsum('kaCi,kCa->ia', g_ovvo, Vupper_kp3, optimize=True))
        return term0 + term1 + term2

    def apply_C_IIIp_Ip(V_Ip):
        Wd1 = (-np.einsum('kXiY,iX->kXY', g_ovov, V_Ip, optimize=True)
               + 0.5 * np.einsum('kXYi,iX->kXY', g_ovvo, V_Ip, optimize=True))
        term0 = s2 * (V_Ip @ g_aa_cd)
        C_r, D_r = iu_v, ju_v
        return term0 + s2 * Wd1[:, C_r, D_r] + s2 * Wd1[:, D_r, C_r]

    # pp-ladder Term0, dense
    g_vvvv_anti = g_vvvv - g_vvvv.transpose(0, 1, 3, 2)
    g_pp_v_anti = g_vvvv_anti[iu_v[:, None], ju_v[:, None], iu_v[None, :], ju_v[None, :]]
    g_pp_v_sym1 = g_vvvv[iu_v[:, None], ju_v[:, None], iu_v[None, :], ju_v[None, :]]
    g_pp_v_sym2 = g_vvvv[iu_v[:, None], ju_v[:, None], ju_v[None, :], iu_v[None, :]]

    def _term0_IIp(Vmat):
        return Vmat @ g_pp_v_anti.T

    def _term0_IIIp(Vmat):
        return Vmat @ g_pp_v_sym1.T + Vmat @ g_pp_v_sym2.T
    def apply_C_IIp_IIp(Vmat, Apps):
        App1, App2, App3, App4 = Apps
        Term0 = _term0_IIp(Vmat)
        A, B = iu_v, ju_v
        return (Term0 - App3[:, A, B] + App1[:, A, B] + 1.5 * App2[:, B, A]
                - App1[:, B, A] - 1.5 * App4[:, B, A] + App3[:, B, A]
                - 1.5 * App2[:, A, B] + 1.5 * App4[:, A, B])

    def apply_C_IIIp_IIIp(Vmat, Apps):
        App1, App2, App3, App4 = Apps
        Term0 = _term0_IIIp(Vmat)
        A, B = iu_v, ju_v
        return (Term0 - (App1[:, A, B] + App1[:, B, A]) - (App3[:, A, B] + App3[:, B, A])
                + 0.5 * (App2[:, A, B] + App2[:, B, A]) + 0.5 * (App4[:, A, B] + App4[:, B, A]))

    def apply_C_IIp_IIIp(Apps_IIIp):
        App1, App2, App3, App4 = Apps_IIIp
        A, B = iu_v, ju_v
        return 0.5 * s3 * (App2[:, B, A] - App2[:, A, B]) + 0.5 * s3 * (App4[:, B, A] - App4[:, A, B])

    def apply_C_IIIp_IIp(Apps_IIp):
        App1, App2, App3, App4 = Apps_IIp
        A, B = iu_v, ju_v
        return 0.5 * s3 * (App2[:, B, A] + App2[:, A, B]) - 0.5 * s3 * (App4[:, A, B] + App4[:, B, A])

    def aop(z):
        z_p = z[:norb]
        z_I = z[oI:oII].reshape(O, V)
        z_II = z[oII:oIII].reshape(nP_o, V)
        z_III = z[oIII:oIp].reshape(nP_o, V)
        z_Ip = z[oIp:oIIp].reshape(O, V)
        z_IIp = z[oIIp:oIIIp].reshape(O, nP_v)
        z_IIIp = z[oIIIp:nH].reshape(O, nP_v)

        y_p = (F @ z_p + U_I @ z_I.reshape(-1) + apply_U_2h1p_fwd(z_II, z_III)
               + U_Ip @ z_Ip.reshape(-1) + apply_U_2p1h_fwd(z_IIp, z_IIIp))

        c2h1p_I = apply_C_I_II(z_II) + apply_C_I_III(z_III)
        c2h1p_II = apply_C_II_I(z_I) + apply_C_II_II(z_II) + apply_C_II_III(z_III)
        c2h1p_III = apply_C_III_I(z_I) + apply_C_III_II(z_II) + apply_C_III_III(z_III)

        y_I = (U_I.T @ z_p) + e_I * z_I.reshape(-1) + (apply_C_I_I(z_I) + c2h1p_I).reshape(-1)
        uT_II, uT_III = apply_U_2h1p_adj(z_p)
        y_II = uT_II.reshape(-1) + e_II * z_II.reshape(-1) + c2h1p_II.reshape(-1)
        y_III = uT_III.reshape(-1) + e_III * z_III.reshape(-1) + c2h1p_III.reshape(-1)

        Apps_IIp = compute_Apps(make_upper_k(z_IIp, iu_v, ju_v, O, V))
        Apps_IIIp = compute_Apps(make_upper_k(z_IIIp, iu_v, ju_v, O, V))
        c2p1h_Ip = apply_C_Ip_IIp(z_IIp) + apply_C_Ip_IIIp(z_IIIp)
        c2p1h_IIp = apply_C_IIp_Ip(z_Ip)
        c2p1h_IIIp = apply_C_IIIp_Ip(z_Ip)

        y_Ip = (U_Ip.T @ z_p) + e_Ip * z_Ip.reshape(-1) + (apply_C_Ip_Ip(z_Ip) + c2p1h_Ip).reshape(-1)
        uT_IIp, uT_IIIp = apply_U_2p1h_adj(z_p)
        y_IIp = uT_IIp.reshape(-1) + e_IIp * z_IIp.reshape(-1) + (
            c2p1h_IIp + apply_C_IIp_IIp(z_IIp, Apps_IIp) + apply_C_IIp_IIIp(Apps_IIIp)).reshape(-1)
        y_IIIp = uT_IIIp.reshape(-1) + e_IIIp * z_IIIp.reshape(-1) + (
            c2p1h_IIIp + apply_C_IIIp_IIp(Apps_IIp) + apply_C_IIIp_IIIp(z_IIIp, Apps_IIIp)).reshape(-1)

        return np.concatenate([y_p, y_I, y_II, y_III, y_Ip, y_IIp, y_IIIp])

    diag = np.concatenate([np.diag(F), e_I, e_II, e_III, e_Ip, e_IIp, e_IIIp])
    return aop, diag, d
