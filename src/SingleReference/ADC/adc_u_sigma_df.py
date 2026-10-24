"""Spin-orbital matrix-free ADC(3) operator, DF/g-free route: every
matvec term is a rank<=3 B-factor chain (no 4-index intermediate
anywhere); the production open-shell path."""
import numpy as np

from src.SingleReference.CC.cached_einsum import einsum as _cached_einsum
from src.SingleReference.ADC.adc_utils import g_block_df
from src.SingleReference.ADC.adc_u_utils import (
    assemble_operator, u2_denominators, u1_dressing_shift,
    u1_shift_terms_2h1p, u1_shift_terms_2p1h)


def apply_U_2h1p(s, nocc, z_p, Vfull, t2_ijcd=None, u1_shift=None):
    """(dy_2h1p_full, dy_p), rank<=3 B-factor chains throughout."""
    norb = s.norb
    occ, virt = slice(0, nocc), slice(nocc, norb)
    B = s.B_spin
    dy_shift, dy_p_shift = u1_shift_terms_2h1p(norb, nocc, z_p, Vfull, u1_shift)

    ing = s.matrix_free_ingredients(nocc)
    B_v, B_v_full, B_o_full = ing['B_v'], ing['B_v_full'], ing['B_o_full']
    B_ov, B_oo, B_vo = ing['B_ov'], ing['B_oo'], ing['B_vo']
    if t2_ijcd is None:
        denom_ij_cd, _ = u2_denominators(s.eps, nocc)
        t2_ijcd = g_block_df(B, occ, occ, virt, virt) / denom_ij_cd

    Zo = _cached_einsum('Qip,p->Qi', B_o_full, z_p, optimize=True)   # (naux,O)
    Zv = _cached_einsum('Qcp,p->Qc', B_v_full, z_p, optimize=True)   # (naux,V)

    # ---- forward ----
    # T0 = einsum('ijpa,p->ija', g_ij_p_a, z_p) without forming g_ij_p_a
    T0 = (_cached_einsum('Qja,Qi->ija', B_ov, Zo, optimize=True)
          - _cached_einsum('Qia,Qj->ija', B_ov, Zo, optimize=True))
    W1 = (_cached_einsum('Qda,Qc->cda', B_v, Zv, optimize=True)
          - _cached_einsum('Qca,Qd->cda', B_v, Zv, optimize=True))
    CD = 0.5 * _cached_einsum('ijcd,cda->ija', t2_ijcd, W1, optimize=True)
    # W2 = einsum('cmpk,p->cmk', g_v_o_o, z_p) without forming g_v_o_o
    W2 = (_cached_einsum('Qmk,Qc->cmk', B_oo, Zv, optimize=True)
          - _cached_einsum('Qck,Qm->cmk', B_vo, Zo, optimize=True))
    KC1 = -_cached_einsum('ikca,cjk->ija', t2_ijcd, W2, optimize=True)
    KC2 = _cached_einsum('jkca,cik->ija', t2_ijcd, W2, optimize=True)
    dy_2h1p_full = T0 + CD + KC1 + KC2

    # ---- adjoint ----
    R1 = _cached_einsum('Qja,ija->Qi', B_ov, Vfull, optimize=True)
    R2 = _cached_einsum('Qia,ija->Qj', B_ov, Vfull, optimize=True)
    T0_adj = 0.5 * (_cached_einsum('Qip,Qi->p', B_o_full, R1, optimize=True)
                    - _cached_einsum('Qjp,Qj->p', B_o_full, R2, optimize=True))
    X1 = _cached_einsum('ijcd,ija->cda', t2_ijcd, Vfull, optimize=True)   # (V,V,V)
    S1 = _cached_einsum('cda,Qda->Qc', X1, B_v, optimize=True)
    S2 = _cached_einsum('cda,Qca->Qd', X1, B_v, optimize=True)
    CD_adj = 0.25 * (_cached_einsum('Qc,Qcp->p', S1, B_v_full, optimize=True)
                     - _cached_einsum('Qd,Qdp->p', S2, B_v_full, optimize=True))
    Y2 = _cached_einsum('ikca,ija->kcj', t2_ijcd, Vfull, optimize=True)
    Ta = _cached_einsum('kcj,Qjk->Qc', Y2, B_oo, optimize=True)
    Tb = _cached_einsum('kcj,Qck->Qj', Y2, B_vo, optimize=True)
    KC1_adj = -0.5 * (_cached_einsum('Qcp,Qc->p', B_v_full, Ta, optimize=True)
                      - _cached_einsum('Qjp,Qj->p', B_o_full, Tb, optimize=True))
    Y3 = _cached_einsum('jkca,ija->kci', t2_ijcd, Vfull, optimize=True)
    Tc = _cached_einsum('kci,Qik->Qc', Y3, B_oo, optimize=True)
    Td = _cached_einsum('kci,Qck->Qi', Y3, B_vo, optimize=True)
    KC2_adj = 0.5 * (_cached_einsum('Qcp,Qc->p', B_v_full, Tc, optimize=True)
                     - _cached_einsum('Qip,Qi->p', B_o_full, Td, optimize=True))
    dy_p = T0_adj + CD_adj + KC1_adj + KC2_adj
    return dy_2h1p_full + dy_shift, dy_p + dy_p_shift


def apply_U_2p1h(s, nocc, z_p, Vfull, t2_ijcd=None, u1_shift=None):
    """2p1h mirror of apply_U_2h1p (shared t2 as transposed view)."""
    norb = s.norb
    occ, virt = slice(0, nocc), slice(nocc, norb)
    B = s.B_spin
    dy_shift, dy_p_shift = u1_shift_terms_2p1h(norb, nocc, z_p, Vfull, u1_shift)

    ing = s.matrix_free_ingredients(nocc)
    B_v, B_v_full, B_o_full = ing['B_v'], ing['B_v_full'], ing['B_o_full']
    B_ov, B_oo, B_vo = ing['B_ov'], ing['B_oo'], ing['B_vo']
    if t2_ijcd is None:
        denom_ij_cd, _ = u2_denominators(s.eps, nocc)
        t2_ijcd = g_block_df(B, occ, occ, virt, virt) / denom_ij_cd
    # t2m == -t2_abkl, a transposed VIEW of the shared t2_ijcd array
    # (no copy); each t2-linear term below flips sign accordingly --
    # on the term prefactor for forward terms, on the small rank-3
    # X1/Y2/Y3 intermediates for the adjoint chain.
    t2m = t2_ijcd.transpose(2, 3, 0, 1)

    Zo = _cached_einsum('Qip,p->Qi', B_o_full, z_p, optimize=True)   # (naux,O)
    Zv = _cached_einsum('Qcp,p->Qc', B_v_full, z_p, optimize=True)   # (naux,V)

    # ---- forward ----
    # T0 = einsum('abpi,p->iab', g_ab_p_i, z_p) without forming g_ab_p_i
    T0 = (_cached_einsum('Qbi,Qa->iab', B_vo, Zv, optimize=True)
          - _cached_einsum('Qai,Qb->iab', B_vo, Zv, optimize=True))
    # W1 = einsum('klpi,p->kli', g_o_o_p, z_p) without forming g_o_o_p
    W1 = (_cached_einsum('Qli,Qk->kli', B_oo, Zo, optimize=True)
          - _cached_einsum('Qki,Ql->kli', B_oo, Zo, optimize=True))
    CD = 0.5 * _cached_einsum('abkl,kli->iab', t2m, W1, optimize=True)
    # W2 = einsum('kmpc,p->kmc', g_o_v_v, z_p) without forming g_o_v_v
    W2 = (_cached_einsum('Qmc,Qk->kmc', B_v, Zo, optimize=True)
          - _cached_einsum('Qkc,Qm->kmc', B_ov, Zv, optimize=True))
    KC1 = -_cached_einsum('acki,kbc->iab', t2m, W2, optimize=True)
    KC2 = _cached_einsum('bcki,kac->iab', t2m, W2, optimize=True)
    dy_2p1h_full = T0 + CD + KC1 + KC2

    # ---- adjoint ----
    Ra = _cached_einsum('Qbi,iab->Qa', B_vo, Vfull, optimize=True)
    Rb = _cached_einsum('Qai,iab->Qb', B_vo, Vfull, optimize=True)
    T0_adj = 0.5 * (_cached_einsum('Qap,Qa->p', B_v_full, Ra, optimize=True)
                    - _cached_einsum('Qbp,Qb->p', B_v_full, Rb, optimize=True))
    X1 = -_cached_einsum('abkl,iab->kli', t2m, Vfull, optimize=True)
    Sa = _cached_einsum('kli,Qli->Qk', X1, B_oo, optimize=True)
    Sb = _cached_einsum('kli,Qki->Ql', X1, B_oo, optimize=True)
    CD_adj = -0.25 * (_cached_einsum('Qkp,Qk->p', B_o_full, Sa, optimize=True)
                      - _cached_einsum('Qlp,Ql->p', B_o_full, Sb, optimize=True))
    Y2 = -_cached_einsum('acki,iab->kcb', t2m, Vfull, optimize=True)
    Ua = _cached_einsum('kcb,Qbc->Qk', Y2, B_v, optimize=True)
    Ub = _cached_einsum('kcb,Qkc->Qb', Y2, B_ov, optimize=True)
    KC1_adj = 0.5 * (_cached_einsum('Qkp,Qk->p', B_o_full, Ua, optimize=True)
                     - _cached_einsum('Qbp,Qb->p', B_v_full, Ub, optimize=True))
    Y3 = -_cached_einsum('bcki,iab->kca', t2m, Vfull, optimize=True)
    Va = _cached_einsum('kca,Qac->Qk', Y3, B_v, optimize=True)
    Vb = _cached_einsum('kca,Qkc->Qa', Y3, B_ov, optimize=True)
    KC2_adj = -0.5 * (_cached_einsum('Qkp,Qk->p', B_o_full, Va, optimize=True)
                      - _cached_einsum('Qap,Qa->p', B_v_full, Vb, optimize=True))
    dy_p = T0_adj + CD_adj + KC1_adj + KC2_adj
    return dy_2p1h_full + dy_shift, dy_p + dy_p_shift


def apply_C_2h1p(s, nocc, V):
    """(Vfull, c_2h1p) from rank<=3 B chains -- per-occupied-index loop
    (triangle contraction graph, no rank-4-free batched order exists)."""
    ing = s.matrix_free_ingredients(nocc)
    iu, ju, nvirt = ing['iu'], ing['ju'], ing['nvirt']
    Vfull = np.zeros((nocc, nocc, nvirt))
    Vfull[iu, ju, :] = V
    Vfull[ju, iu, :] = -V

    B_oo, B_ov, B_vo, B_v_full = ing['B_oo'], ing['B_ov'], ing['B_vo'], ing['B_v_full']
    Y1 = np.zeros((nocc, nocc, nvirt))
    F = np.zeros((nocc, nocc, nvirt))
    for i in range(nocc):
        # ---- Y1 (from g_oooo) ----
        Boo_i = B_oo[:, i, :]                                            # (naux,O)
        M = _cached_einsum('Qk,klA->QlA', Boo_i, Vfull, optimize=True)        # (naux,O,V)
        term = _cached_einsum('Qjl,QlA->jA', B_oo, M, optimize=True)          # (O,V)
        P = _cached_einsum('Ql,klA->QkA', Boo_i, Vfull, optimize=True)        # (naux,O,V)
        exch = _cached_einsum('Qjk,QkA->jA', B_oo, P, optimize=True)          # (O,V)
        Y1[i] = -0.5 * (term - exch)

        # ---- F[i,j,a] (from g_cvov), Y2n4=F, Y3n5=F.transpose(1,0,2) ----
        Vi = Vfull[i]  # (O,V) indexed [m,c]
        Mc = _cached_einsum('Qca,mc->Qam', B_v_full[:, :, nocc:], Vi, optimize=True)   # (naux,V,O)
        term_c = _cached_einsum('Qjm,Qam->ja', B_oo, Mc, optimize=True)                # (O,V)
        K = _cached_einsum('Qcm,mc->Q', B_vo, Vi, optimize=True)                       # (naux,)
        exch_c = _cached_einsum('Qja,Q->ja', B_ov, K, optimize=True)                   # (O,V)
        F[i] = term_c - exch_c

    Y2n4 = F
    Y3n5 = F.transpose(1, 0, 2)
    return Vfull, (Y1 + Y2n4 - Y3n5)[iu, ju, :]


def apply_C_2p1h(s, nocc, V):
    """Mirror of apply_C_2h1p."""
    ing = s.matrix_free_ingredients(nocc)
    au, bu, nvirt = ing['au'], ing['bu'], ing['nvirt']
    Vfull = np.zeros((nocc, nvirt, nvirt))
    Vfull[:, au, bu] = V
    Vfull[:, bu, au] = -V

    B_v, B_ov, B_vo, B_oo = ing['B_v'], ing['B_ov'], ing['B_vo'], ing['B_oo']
    Y1 = np.zeros((nocc, nvirt, nvirt))
    G = np.zeros((nocc, nvirt, nvirt))
    # i-independent: one (naux,V) reduction shared by every iteration's
    # exchange term (was recomputed nocc times inside the loop).
    R = _cached_einsum('Qkm,kam->Qa', B_ov, Vfull, optimize=True)     # (naux,V)
    for i in range(nocc):
        # ---- Y1 (from g_vvvv) ----
        Vi = Vfull[i]  # (V,V) indexed [c,d]
        N = _cached_einsum('Qbd,cd->Qbc', B_v, Vi, optimize=True)     # (naux,V,V)
        direct = _cached_einsum('Qac,Qbc->ab', B_v, N, optimize=True)
        P = _cached_einsum('Qbc,cd->Qbd', B_v, Vi, optimize=True)     # (naux,V,V)
        exch = _cached_einsum('Qad,Qbd->ab', B_v, P, optimize=True)
        Y1[i] = 0.5 * (direct - exch)

        # ---- G[i,a,b] (from g_ovov), Y2n4=-G, Y3n5=G.transpose(0,2,1) ----
        Bvo_i = B_vo[:, :, i]  # (naux,V)
        Boo_i = B_oo[:, :, i]  # (naux,O)
        Pg = _cached_einsum('Qk,kam->Qam', Boo_i, Vfull, optimize=True)   # (naux,V,V) -- Vfull[k,a,m]
        direct_ba = _cached_einsum('Qbm,Qam->ba', B_v, Pg, optimize=True)
        exch_ab = _cached_einsum('Qa,Qb->ab', R, Bvo_i, optimize=True)
        G[i] = direct_ba.T - exch_ab

    Y2n4 = -G
    Y3n5 = G.transpose(0, 2, 1)
    return Vfull, (Y1 + Y2n4 + Y3n5)[:, au, bu]


def build_operator(s, nocc, static_correction=None, t2_ijcd=None):
    """(aop, diag, dims): matrix-free spin-orbital ADC(3) operator.
    t2_ijcd: optional EN-dressed T2^(1) (u1 shift threaded automatically)."""
    u1_shift = None
    if t2_ijcd is not None:
        u1_shift = u1_dressing_shift(s, nocc, t2_ijcd)

    # Build the bare-t2 default ONCE (apply_U_* would otherwise rebuild the
    # rank-4 DF gather + division on EVERY matvec when handed t2=None).
    if t2_ijcd is None:
        occ_, virt_ = slice(0, nocc), slice(nocc, s.norb)
        denom_ij_cd, _ = u2_denominators(s.eps, nocc)
        t2_ijcd = g_block_df(s.B_spin, occ_, occ_, virt_, virt_) / denom_ij_cd
    return assemble_operator(s, nocc, static_correction, t2_ijcd, u1_shift, apply_C_2h1p,
                             apply_C_2p1h, apply_U_2h1p, apply_U_2p1h)
