# GENERATED CODE -- Laplace-fused T3^(2) (all four spin blocks)
# contribution to t1_3_{aa,bb}_numerator/m3_ov_{a,b}_12_unrestricted
# (the only live consumers; see
# generate_mp3_t3_laplace_unrestricted.py's module docstring for scope).
# Do not edit by hand.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum
from src.Base.utils.grids.grids import minimax_time_grid


def t1_3_aa_t3_laplace(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, eps_a, eps_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    ei_a, ea_a = eps_a[o_a], eps_a[v_a]
    ei_b, ea_b = eps_b[o_b], eps_b[v_b]
    gap_min = max(min(ea_a.min(), ea_b.min()) - max(ei_a.max(), ei_b.max()), 1e-3)
    gap_max = max(ea_a.max(), ea_b.max()) - min(ei_a.min(), ei_b.min())
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    t1_3_aa_t3 = np.zeros((nv_a, no_a))
    for _tk in range(ntau):
        _t = tau[_tk]
        _w = -sigma[_tk]
        Oe_a = np.exp(ei_a * _t)
        Oe_b = np.exp(ei_b * _t)
        Ve_a = np.exp(-ea_a * _t)
        Ve_b = np.exp(-ea_b * _t)
        _iter = np.zeros((nv_a, no_a))
        _cse0 = ((((g_aaaa[o_a, o_a, v_a, v_a] * Oe_a[:, None, None, None]) * Oe_a[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_a[None, None, None, :])
        _cse1 = g_aaaa[o_a, v_a, o_a, o_a]
        _cse2 = ((t2_aaaa * Ve_a[None, :, None, None]) * Oe_a[None, None, :, None])
        _cse3 = (g_aaaa[o_a, v_a, o_a, o_a] * Oe_a[None, None, :, None])
        _cse4 = (t2_aaaa * Ve_a[None, :, None, None])
        _cse5 = (t2_aaaa * Oe_a[None, None, :, None])
        _cse6 = ((g_aaaa[o_a, v_a, o_a, o_a] * Ve_a[None, :, None, None]) * Oe_a[None, None, :, None])
        _cse7 = t2_aaaa
        _cse8 = g_aaaa[v_a, v_a, v_a, o_a]
        _cse9 = (g_aaaa[v_a, v_a, v_a, o_a] * Ve_a[None, :, None, None])
        _cse10 = ((g_aaaa[v_a, v_a, v_a, o_a] * Ve_a[None, :, None, None]) * Oe_a[None, None, None, :])
        _cse11 = ((((g_abab[o_a, o_b, v_a, v_b] * Oe_a[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse12 = g_abab[v_a, o_b, o_a, o_b]
        _cse13 = ((t2_abab * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse14 = (g_abab[v_a, o_b, o_a, o_b] * Ve_a[:, None, None, None])
        _cse15 = (t2_abab * Oe_a[None, None, :, None])
        _cse16 = (g_abab[v_a, o_b, o_a, o_b] * Oe_a[None, None, :, None])
        _cse17 = (t2_abab * Ve_a[:, None, None, None])
        _cse18 = ((g_abab[v_a, o_b, o_a, o_b] * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse19 = t2_abab
        _cse20 = g_abab[o_a, v_b, o_a, o_b]
        _cse21 = (g_abab[o_a, v_b, o_a, o_b] * Oe_a[None, None, :, None])
        _cse22 = g_abab[v_a, v_b, v_a, o_b]
        _cse23 = g_abab[v_a, v_b, o_a, v_b]
        _cse24 = (g_abab[v_a, v_b, o_a, v_b] * Oe_a[None, None, :, None])
        _cse25 = (g_abab[v_a, v_b, v_a, o_b] * Ve_a[:, None, None, None])
        _cse26 = (g_abab[v_a, v_b, o_a, v_b] * Ve_a[:, None, None, None])
        _cse27 = ((g_abab[v_a, v_b, o_a, v_b] * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse28 = ((t2_aaaa * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse29 = (t2_aaaa * Ve_a[:, None, None, None])
        _cse30 = (g_aaaa[v_a, v_a, v_a, o_a] * Ve_a[:, None, None, None])
        _cse31 = ((g_aaaa[v_a, v_a, v_a, o_a] * Ve_a[:, None, None, None]) * Oe_a[None, None, None, :])
        _cse32 = ((((g_bbbb[o_b, o_b, v_b, v_b] * Oe_b[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_b[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse33 = g_bbbb[o_b, v_b, o_b, o_b]
        _cse34 = t2_bbbb
        _cse35 = g_bbbb[v_b, v_b, v_b, o_b]
        _tmp = einsum('kjbc,obkj,caio->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,baio->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obij,cako->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,bako->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,obik,cajo->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ocik,bajo->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oakj,bcio->ai', _cse0, (g_aaaa[o_a, v_a, o_a, o_a] * Ve_a[None, :, None, None]), _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,oaij,bcko->ai', _cse0, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oaik,bcjo->ai', _cse0, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgj,gaik->ai', _cse0, _cse8, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagj,gcik->ai', _cse0, _cse9, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bcgk,gaij->ai', _cse0, _cse8, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bagk,gcij->ai', _cse0, _cse9, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgi,gakj->ai', _cse0, (g_aaaa[v_a, v_a, v_a, o_a] * Oe_a[None, None, None, :]), _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagi,gckj->ai', _cse0, _cse10, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cagj,gbik->ai', _cse0, _cse9, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cagk,gbij->ai', _cse0, _cse9, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cagi,gbkj->ai', _cse0, _cse10, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bokj,acio->ai', _cse11, _cse12, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,aokj,bcio->ai', _cse11, _cse14, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,boij,acko->ai', _cse11, _cse16, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,aoij,bcko->ai', _cse11, _cse18, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,obik,acoj->ai', _cse11, _cse3, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,oaik,bcoj->ai', _cse11, _cse6, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,baio->ai', _cse11, _cse20, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,bako->ai', _cse11, _cse21, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgj,gaik->ai', _cse11, _cse22, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagk,gcij->ai', _cse11, _cse9, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bckg,agij->ai', _cse11, _cse23, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bagi,gckj->ai', _cse11, _cse10, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bcig,agkj->ai', _cse11, _cse24, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acgj,gbik->ai', _cse11, _cse25, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ackg,bgij->ai', _cse11, _cse26, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acig,bgkj->ai', _cse11, _cse27, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bojk,acio->ai', _cse11, _cse12, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,aojk,bcio->ai', _cse11, _cse14, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,boik,acjo->ai', _cse11, _cse16, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,aoik,bcjo->ai', _cse11, _cse18, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,obij,acok->ai', _cse11, _cse3, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,oaij,bcok->ai', _cse11, _cse6, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,ocjk,baio->ai', _cse11, _cse20, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,ocik,bajo->ai', _cse11, _cse21, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bcgk,gaij->ai', _cse11, _cse22, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bagj,gcik->ai', _cse11, _cse9, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bcjg,agik->ai', _cse11, _cse23, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bagi,gcjk->ai', _cse11, _cse10, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bcig,agjk->ai', _cse11, _cse24, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,acgk,gbij->ai', _cse11, _cse25, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,acjg,bgik->ai', _cse11, _cse26, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,acig,bgjk->ai', _cse11, _cse27, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,aokj,cbio->ai', _cse11, _cse14, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cokj,abio->ai', _cse11, _cse12, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,aoij,cbko->ai', _cse11, _cse18, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,coij,abko->ai', _cse11, _cse16, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,oaik,cboj->ai', _cse11, _cse6, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,ocik,aboj->ai', _cse11, _cse3, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,obkj,acio->ai', _cse11, _cse20, _cse28, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,obij,acko->ai', _cse11, _cse21, _cse29, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,abgj,gcik->ai', _cse11, _cse25, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,acgk,gbij->ai', _cse11, _cse30, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,abkg,cgij->ai', _cse11, _cse26, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,acgi,gbkj->ai', _cse11, _cse31, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,abig,cgkj->ai', _cse11, _cse27, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cbgj,gaik->ai', _cse11, _cse22, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,cbkg,agij->ai', _cse11, _cse23, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cbig,agkj->ai', _cse11, _cse24, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,aojk,cbio->ai', _cse11, _cse14, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cojk,abio->ai', _cse11, _cse12, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,aoik,cbjo->ai', _cse11, _cse18, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,coik,abjo->ai', _cse11, _cse16, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,oaij,cbok->ai', _cse11, _cse6, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,ocij,abok->ai', _cse11, _cse3, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,objk,acio->ai', _cse11, _cse20, _cse28, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,obik,acjo->ai', _cse11, _cse21, _cse29, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,abgk,gcij->ai', _cse11, _cse25, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,acgj,gbik->ai', _cse11, _cse30, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,abjg,cgik->ai', _cse11, _cse26, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,acgi,gbjk->ai', _cse11, _cse31, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,abig,cgjk->ai', _cse11, _cse27, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbgk,gaij->ai', _cse11, _cse22, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,cbjg,agik->ai', _cse11, _cse23, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbig,agjk->ai', _cse11, _cse24, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,abio->ai', _cse32, _cse33, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,aoij,cbko->ai', _cse32, _cse18, _cse34, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,abok->ai', _cse32, _cse21, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,aoik,cbjo->ai', _cse32, _cse18, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ocik,aboj->ai', _cse32, _cse21, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obkj,acio->ai', _cse32, _cse33, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obij,acok->ai', _cse32, _cse21, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obik,acoj->ai', _cse32, _cse21, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acgj,gbik->ai', _cse32, _cse25, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,abgj,gcik->ai', _cse32, _cse25, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acgk,gbij->ai', _cse32, _cse25, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,abgk,gcij->ai', _cse32, _cse25, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,acig,gbkj->ai', _cse32, _cse27, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,abig,gckj->ai', _cse32, _cse27, _cse34, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cbgj,agik->ai', _cse32, _cse35, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cbgk,agij->ai', _cse32, _cse35, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        t1_3_aa_t3 += _w * _iter
    return t1_3_aa_t3


def t1_3_bb_t3_laplace(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, eps_a, eps_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    ei_a, ea_a = eps_a[o_a], eps_a[v_a]
    ei_b, ea_b = eps_b[o_b], eps_b[v_b]
    gap_min = max(min(ea_a.min(), ea_b.min()) - max(ei_a.max(), ei_b.max()), 1e-3)
    gap_max = max(ea_a.max(), ea_b.max()) - min(ei_a.min(), ei_b.min())
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    t1_3_bb_t3 = np.zeros((nv_b, no_b))
    for _tk in range(ntau):
        _t = tau[_tk]
        _w = -sigma[_tk]
        Oe_a = np.exp(ei_a * _t)
        Oe_b = np.exp(ei_b * _t)
        Ve_a = np.exp(-ea_a * _t)
        Ve_b = np.exp(-ea_b * _t)
        _iter = np.zeros((nv_b, no_b))
        _cse0 = ((((g_aaaa[o_a, o_a, v_a, v_a] * Oe_a[:, None, None, None]) * Oe_a[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_a[None, None, None, :])
        _cse1 = (g_abab[v_a, o_b, o_a, o_b] * Oe_b[None, None, None, :])
        _cse2 = (t2_abab * Ve_b[None, :, None, None])
        _cse3 = g_aaaa[o_a, v_a, o_a, o_a]
        _cse4 = ((t2_abab * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse5 = ((g_abab[o_a, v_b, o_a, o_b] * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse6 = t2_aaaa
        _cse7 = ((g_abab[v_a, v_b, v_a, o_b] * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse8 = g_aaaa[v_a, v_a, v_a, o_a]
        _cse9 = (g_abab[v_a, v_b, o_a, v_b] * Ve_b[None, :, None, None])
        _cse10 = (t2_abab * Oe_b[None, None, None, :])
        _cse11 = ((((g_abab[o_a, o_b, v_a, v_b] * Oe_a[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse12 = (g_bbbb[o_b, v_b, o_b, o_b] * Oe_b[None, None, :, None])
        _cse13 = g_abab[v_a, o_b, o_a, o_b]
        _cse14 = ((t2_bbbb * Ve_b[None, :, None, None]) * Oe_b[None, None, :, None])
        _cse15 = g_abab[o_a, v_b, o_a, o_b]
        _cse16 = (t2_bbbb * Ve_b[None, :, None, None])
        _cse17 = (g_abab[o_a, v_b, o_a, o_b] * Oe_b[None, None, None, :])
        _cse18 = ((g_bbbb[o_b, v_b, o_b, o_b] * Ve_b[None, :, None, None]) * Oe_b[None, None, :, None])
        _cse19 = t2_abab
        _cse20 = (g_abab[o_a, v_b, o_a, o_b] * Ve_b[None, :, None, None])
        _cse21 = g_abab[v_a, v_b, v_a, o_b]
        _cse22 = (g_abab[v_a, v_b, v_a, o_b] * Ve_b[None, :, None, None])
        _cse23 = (g_abab[v_a, v_b, v_a, o_b] * Oe_b[None, None, None, :])
        _cse24 = g_abab[v_a, v_b, o_a, v_b]
        _cse25 = (t2_bbbb * Oe_b[None, None, :, None])
        _cse26 = (g_bbbb[v_b, v_b, v_b, o_b] * Ve_b[None, :, None, None])
        _cse27 = ((g_bbbb[v_b, v_b, v_b, o_b] * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse28 = (g_bbbb[o_b, v_b, o_b, o_b] * Oe_b[None, None, None, :])
        _cse29 = ((g_bbbb[o_b, v_b, o_b, o_b] * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse30 = ((t2_bbbb * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse31 = (t2_bbbb * Oe_b[None, None, None, :])
        _cse32 = ((((g_bbbb[o_b, o_b, v_b, v_b] * Oe_b[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_b[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse33 = g_bbbb[o_b, v_b, o_b, o_b]
        _cse34 = t2_bbbb
        _cse35 = g_bbbb[v_b, v_b, v_b, o_b]
        _tmp = einsum('kjbc,boki,cajo->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,coki,bajo->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,boji,cako->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,coji,bako->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,objk,caoi->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ocjk,baoi->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oaki,bcjo->ai', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oaji,bcko->ai', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagi,gcjk->ai', _cse0, _cse7, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgk,gaji->ai', _cse0, _cse8, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bakg,cgji->ai', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bcgj,gaki->ai', _cse0, _cse8, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bajg,cgki->ai', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cagi,gbjk->ai', _cse0, _cse7, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cakg,bgji->ai', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cajg,bgki->ai', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,bako->ai', _cse11, _cse12, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bokj,caio->ai', _cse11, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,baoi->ai', _cse11, _cse15, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,boki,cajo->ai', _cse11, _cse1, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ocki,baoj->ai', _cse11, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oaij,bcko->ai', _cse11, _cse18, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oakj,bcoi->ai', _cse11, _cse20, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oaki,bcoj->ai', _cse11, _cse5, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgj,gaki->ai', _cse11, _cse21, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bagj,gcki->ai', _cse11, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgi,gakj->ai', _cse11, _cse23, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagi,gckj->ai', _cse11, _cse7, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bckg,gaij->ai', _cse11, _cse24, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bakg,gcij->ai', _cse11, _cse9, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cagj,bgki->ai', _cse11, _cse26, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cagi,bgkj->ai', _cse11, _cse27, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,ocki,bajo->ai', _cse11, _cse28, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,boji,cako->ai', _cse11, _cse1, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,ocji,baok->ai', _cse11, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bojk,caio->ai', _cse11, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,ocjk,baoi->ai', _cse11, _cse15, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,oaki,bcjo->ai', _cse11, _cse29, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,oaji,bcok->ai', _cse11, _cse5, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,oajk,bcoi->ai', _cse11, _cse20, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bcgi,gajk->ai', _cse11, _cse23, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bagi,gcjk->ai', _cse11, _cse7, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bcgk,gaji->ai', _cse11, _cse21, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bagk,gcji->ai', _cse11, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bcjg,gaki->ai', _cse11, _cse24, _cse30, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bajg,gcki->ai', _cse11, _cse9, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,cagi,bgjk->ai', _cse11, _cse27, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,cagk,bgji->ai', _cse11, _cse26, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,obij,cako->ai', _cse11, _cse12, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,cokj,baio->ai', _cse11, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,obkj,caoi->ai', _cse11, _cse15, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,coki,bajo->ai', _cse11, _cse1, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,obki,caoj->ai', _cse11, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,oaij,cbko->ai', _cse11, _cse18, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,oakj,cboi->ai', _cse11, _cse20, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,oaki,cboj->ai', _cse11, _cse5, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,cbgj,gaki->ai', _cse11, _cse21, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cagj,gbki->ai', _cse11, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,cbgi,gakj->ai', _cse11, _cse23, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,cagi,gbkj->ai', _cse11, _cse7, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cbkg,gaij->ai', _cse11, _cse24, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,cakg,gbij->ai', _cse11, _cse9, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,bagj,cgki->ai', _cse11, _cse26, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,bagi,cgkj->ai', _cse11, _cse27, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,obki,cajo->ai', _cse11, _cse28, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,coji,bako->ai', _cse11, _cse1, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,obji,caok->ai', _cse11, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cojk,baio->ai', _cse11, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,objk,caoi->ai', _cse11, _cse15, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,oaki,cbjo->ai', _cse11, _cse29, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,oaji,cbok->ai', _cse11, _cse5, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,oajk,cboi->ai', _cse11, _cse20, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbgi,gajk->ai', _cse11, _cse23, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,cagi,gbjk->ai', _cse11, _cse7, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbgk,gaji->ai', _cse11, _cse21, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cagk,gbji->ai', _cse11, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,cbjg,gaki->ai', _cse11, _cse24, _cse30, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cajg,gbki->ai', _cse11, _cse9, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,bagi,cgjk->ai', _cse11, _cse27, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,bagk,cgji->ai', _cse11, _cse26, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,obkj,caio->ai', _cse32, _cse33, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,baio->ai', _cse32, _cse33, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obij,cako->ai', _cse32, _cse12, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,bako->ai', _cse32, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,obik,cajo->ai', _cse32, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ocik,bajo->ai', _cse32, _cse12, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oakj,bcio->ai', _cse32, (g_bbbb[o_b, v_b, o_b, o_b] * Ve_b[None, :, None, None]), _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,oaij,bcko->ai', _cse32, _cse18, _cse34, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oaik,bcjo->ai', _cse32, _cse18, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgj,gaik->ai', _cse32, _cse35, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagj,gcik->ai', _cse32, _cse26, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bcgk,gaij->ai', _cse32, _cse35, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bagk,gcij->ai', _cse32, _cse26, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgi,gakj->ai', _cse32, (g_bbbb[v_b, v_b, v_b, o_b] * Oe_b[None, None, None, :]), _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagi,gckj->ai', _cse32, _cse27, _cse34, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cagj,gbik->ai', _cse32, _cse26, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cagk,gbij->ai', _cse32, _cse26, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cagi,gbkj->ai', _cse32, _cse27, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        t1_3_bb_t3 += _w * _iter
    return t1_3_bb_t3


def m3_ov_a_t3_laplace(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, eps_a, eps_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    ei_a, ea_a = eps_a[o_a], eps_a[v_a]
    ei_b, ea_b = eps_b[o_b], eps_b[v_b]
    gap_min = max(min(ea_a.min(), ea_b.min()) - max(ei_a.max(), ei_b.max()), 1e-3)
    gap_max = max(ea_a.max(), ea_b.max()) - min(ei_a.min(), ei_b.min())
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    m3_ov_a_t3 = np.zeros((no_a, nv_a))
    for _tk in range(ntau):
        _t = tau[_tk]
        _w = -sigma[_tk]
        Oe_a = np.exp(ei_a * _t)
        Oe_b = np.exp(ei_b * _t)
        Ve_a = np.exp(-ea_a * _t)
        Ve_b = np.exp(-ea_b * _t)
        _iter = np.zeros((no_a, nv_a))
        _cse0 = ((((l2_aaaa * Oe_a[:, None, None, None]) * Oe_a[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_a[None, None, None, :])
        _cse1 = ((g_aaaa[o_a, v_a, o_a, o_a] * Ve_a[None, :, None, None]) * Oe_a[None, None, None, :])
        _cse2 = t2_aaaa
        _cse3 = (g_aaaa[o_a, v_a, o_a, o_a] * Oe_a[None, None, None, :])
        _cse4 = (t2_aaaa * Ve_a[:, None, None, None])
        _cse5 = (t2_aaaa * Oe_a[None, None, :, None])
        _cse6 = g_aaaa[o_a, v_a, o_a, o_a]
        _cse7 = ((t2_aaaa * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse8 = ((g_aaaa[v_a, v_a, v_a, o_a] * Ve_a[:, None, None, None]) * Oe_a[None, None, None, :])
        _cse9 = (g_aaaa[v_a, v_a, v_a, o_a] * Ve_a[:, None, None, None])
        _cse10 = (t2_aaaa * Oe_a[None, None, None, :])
        _cse11 = g_aaaa[v_a, v_a, v_a, o_a]
        _cse12 = ((t2_aaaa * Ve_a[None, :, None, None]) * Oe_a[None, None, None, :])
        _cse13 = ((((l2_abab * Oe_a[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse14 = ((g_abab[v_a, o_b, o_a, o_b] * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse15 = t2_abab
        _cse16 = (g_abab[v_a, o_b, o_a, o_b] * Oe_a[None, None, :, None])
        _cse17 = (t2_abab * Ve_a[:, None, None, None])
        _cse18 = (g_abab[v_a, o_b, o_a, o_b] * Ve_a[:, None, None, None])
        _cse19 = (t2_abab * Oe_a[None, None, :, None])
        _cse20 = g_abab[v_a, o_b, o_a, o_b]
        _cse21 = ((t2_abab * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse22 = (g_abab[o_a, v_b, o_a, o_b] * Oe_a[None, None, :, None])
        _cse23 = g_abab[o_a, v_b, o_a, o_b]
        _cse24 = (g_abab[v_a, v_b, v_a, o_b] * Ve_a[:, None, None, None])
        _cse25 = ((g_abab[v_a, v_b, o_a, v_b] * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse26 = (g_abab[v_a, v_b, o_a, v_b] * Ve_a[:, None, None, None])
        _cse27 = g_abab[v_a, v_b, v_a, o_b]
        _cse28 = (g_abab[v_a, v_b, o_a, v_b] * Oe_a[None, None, :, None])
        _cse29 = g_abab[v_a, v_b, o_a, v_b]
        _cse30 = ((g_aaaa[o_a, v_a, o_a, o_a] * Ve_a[None, :, None, None]) * Oe_a[None, None, :, None])
        _cse31 = (g_aaaa[o_a, v_a, o_a, o_a] * Oe_a[None, None, :, None])
        _cse32 = ((t2_aaaa * Ve_a[None, :, None, None]) * Oe_a[None, None, :, None])
        _cse33 = ((((l2_bbbb * Oe_b[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_b[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse34 = g_bbbb[o_b, v_b, o_b, o_b]
        _cse35 = t2_bbbb
        _cse36 = g_bbbb[v_b, v_b, v_b, o_b]
        _tmp = einsum('ijba,oejm,baio->me', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,objm,eaio->me', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oeim,bajo->me', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,obim,eajo->me', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeij,bamo->me', _cse0, (g_aaaa[o_a, v_a, o_a, o_a] * Ve_a[None, :, None, None]), _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obij,eamo->me', _cse0, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oajm,ebio->me', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oaim,ebjo->me', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oaij,ebmo->me', _cse0, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgm,gaij->me', _cse0, _cse8, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eagm,gbij->me', _cse0, _cse8, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,ebgj,gaim->me', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,eagj,gbim->me', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgi,gajm->me', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eagi,gbjm->me', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bagm,geij->me', _cse0, (g_aaaa[v_a, v_a, v_a, o_a] * Oe_a[None, None, None, :]), (t2_aaaa * Ve_a[None, :, None, None]), optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bagj,geim->me', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bagi,gejm->me', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eomj,baio->me', _cse13, _cse14, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bomj,eaio->me', _cse13, _cse16, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,eoij,bamo->me', _cse13, _cse18, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,boij,eamo->me', _cse13, _cse20, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeim,baoj->me', _cse13, _cse1, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obim,eaoj->me', _cse13, _cse3, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oamj,ebio->me', _cse13, _cse22, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oaij,ebmo->me', _cse13, _cse23, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eagj,gbim->me', _cse13, _cse24, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgm,gaij->me', _cse13, _cse8, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eamg,bgij->me', _cse13, _cse25, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,ebgi,gamj->me', _cse13, _cse9, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,eaig,bgmj->me', _cse13, _cse26, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bagj,geim->me', _cse13, _cse27, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bamg,egij->me', _cse13, _cse28, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,baig,egmj->me', _cse13, _cse29, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,eomj,abio->me', _cse13, _cse14, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,aomj,ebio->me', _cse13, _cse16, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,eoij,abmo->me', _cse13, _cse18, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,aoij,ebmo->me', _cse13, _cse20, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oeim,aboj->me', _cse13, _cse1, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oaim,eboj->me', _cse13, _cse3, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,obmj,eaio->me', _cse13, _cse22, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,obij,eamo->me', _cse13, _cse23, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,ebgj,gaim->me', _cse13, _cse24, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,eagm,gbij->me', _cse13, _cse8, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,ebmg,agij->me', _cse13, _cse25, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,eagi,gbmj->me', _cse13, _cse9, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,ebig,agmj->me', _cse13, _cse26, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,abgj,geim->me', _cse13, _cse27, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,abmg,egij->me', _cse13, _cse28, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,abig,egmj->me', _cse13, _cse29, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,eoji,bamo->me', _cse13, _cse18, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,boji,eamo->me', _cse13, _cse20, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,eomi,bajo->me', _cse13, _cse14, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,bomi,eajo->me', _cse13, _cse16, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oemj,baoi->me', _cse13, _cse30, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,obmj,eaoi->me', _cse13, _cse31, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oaji,ebmo->me', _cse13, _cse23, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oami,ebjo->me', _cse13, _cse22, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,eagi,gbmj->me', _cse13, _cse24, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,ebgj,gami->me', _cse13, _cse9, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,eajg,bgmi->me', _cse13, _cse26, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,ebgm,gaji->me', _cse13, _cse8, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,eamg,bgji->me', _cse13, _cse25, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,bagi,gemj->me', _cse13, _cse27, _cse32, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,bajg,egmi->me', _cse13, _cse29, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,bamg,egji->me', _cse13, _cse28, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,eoji,abmo->me', _cse13, _cse18, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,aoji,ebmo->me', _cse13, _cse20, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,eomi,abjo->me', _cse13, _cse14, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,aomi,ebjo->me', _cse13, _cse16, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oemj,aboi->me', _cse13, _cse30, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oamj,eboi->me', _cse13, _cse31, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,obji,eamo->me', _cse13, _cse23, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,obmi,eajo->me', _cse13, _cse22, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,ebgi,gamj->me', _cse13, _cse24, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,eagj,gbmi->me', _cse13, _cse9, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,ebjg,agmi->me', _cse13, _cse26, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,eagm,gbji->me', _cse13, _cse8, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,ebmg,agji->me', _cse13, _cse25, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,abgi,gemj->me', _cse13, _cse27, _cse32, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,abjg,egmi->me', _cse13, _cse29, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,abmg,egji->me', _cse13, _cse28, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obji,eamo->me', _cse33, _cse34, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eomi,bajo->me', _cse33, _cse14, _cse35, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,obmi,eaoj->me', _cse33, _cse22, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eomj,baio->me', _cse33, _cse14, _cse35, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obmj,eaoi->me', _cse33, _cse22, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oaji,ebmo->me', _cse33, _cse34, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oami,eboj->me', _cse33, _cse22, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oamj,eboi->me', _cse33, _cse22, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgi,gamj->me', _cse33, _cse24, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,eagi,gbmj->me', _cse33, _cse24, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgj,gami->me', _cse33, _cse24, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eagj,gbmi->me', _cse33, _cse24, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,ebmg,gaji->me', _cse33, _cse25, _cse35, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eamg,gbji->me', _cse33, _cse25, _cse35, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bagi,egmj->me', _cse33, _cse36, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bagj,egmi->me', _cse33, _cse36, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        m3_ov_a_t3 += _w * _iter
    return m3_ov_a_t3


def m3_ov_b_t3_laplace(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, eps_a, eps_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    ei_a, ea_a = eps_a[o_a], eps_a[v_a]
    ei_b, ea_b = eps_b[o_b], eps_b[v_b]
    gap_min = max(min(ea_a.min(), ea_b.min()) - max(ei_a.max(), ei_b.max()), 1e-3)
    gap_max = max(ea_a.max(), ea_b.max()) - min(ei_a.min(), ei_b.min())
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    m3_ov_b_t3 = np.zeros((no_b, nv_b))
    for _tk in range(ntau):
        _t = tau[_tk]
        _w = -sigma[_tk]
        Oe_a = np.exp(ei_a * _t)
        Oe_b = np.exp(ei_b * _t)
        Ve_a = np.exp(-ea_a * _t)
        Ve_b = np.exp(-ea_b * _t)
        _iter = np.zeros((no_b, nv_b))
        _cse0 = ((((l2_aaaa * Oe_a[:, None, None, None]) * Oe_a[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_a[None, None, None, :])
        _cse1 = (g_abab[v_a, o_b, o_a, o_b] * Oe_b[None, None, None, :])
        _cse2 = (t2_abab * Ve_b[None, :, None, None])
        _cse3 = g_aaaa[o_a, v_a, o_a, o_a]
        _cse4 = ((t2_abab * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse5 = ((g_abab[o_a, v_b, o_a, o_b] * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse6 = t2_aaaa
        _cse7 = ((g_abab[v_a, v_b, v_a, o_b] * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse8 = g_aaaa[v_a, v_a, v_a, o_a]
        _cse9 = (g_abab[v_a, v_b, o_a, v_b] * Ve_b[None, :, None, None])
        _cse10 = (t2_abab * Oe_b[None, None, None, :])
        _cse11 = ((((l2_abab * Oe_a[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse12 = ((g_bbbb[o_b, v_b, o_b, o_b] * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse13 = t2_abab
        _cse14 = (t2_bbbb * Ve_b[:, None, None, None])
        _cse15 = g_abab[v_a, o_b, o_a, o_b]
        _cse16 = ((t2_bbbb * Ve_b[:, None, None, None]) * Oe_b[None, None, :, None])
        _cse17 = (g_abab[o_a, v_b, o_a, o_b] * Ve_b[None, :, None, None])
        _cse18 = (g_bbbb[o_b, v_b, o_b, o_b] * Oe_b[None, None, None, :])
        _cse19 = (g_abab[o_a, v_b, o_a, o_b] * Oe_b[None, None, None, :])
        _cse20 = g_abab[o_a, v_b, o_a, o_b]
        _cse21 = (g_abab[v_a, v_b, v_a, o_b] * Oe_b[None, None, None, :])
        _cse22 = (g_abab[v_a, v_b, v_a, o_b] * Ve_b[None, :, None, None])
        _cse23 = g_abab[v_a, v_b, v_a, o_b]
        _cse24 = (t2_bbbb * Oe_b[None, None, None, :])
        _cse25 = g_abab[v_a, v_b, o_a, v_b]
        _cse26 = ((t2_bbbb * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse27 = ((g_bbbb[v_b, v_b, v_b, o_b] * Ve_b[:, None, None, None]) * Oe_b[None, None, None, :])
        _cse28 = (g_bbbb[v_b, v_b, v_b, o_b] * Ve_b[:, None, None, None])
        _cse29 = (t2_bbbb * Ve_b[None, :, None, None])
        _cse30 = ((t2_bbbb * Ve_b[None, :, None, None]) * Oe_b[None, None, :, None])
        _cse31 = ((g_bbbb[v_b, v_b, v_b, o_b] * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse32 = (g_bbbb[v_b, v_b, v_b, o_b] * Ve_b[None, :, None, None])
        _cse33 = ((((l2_bbbb * Oe_b[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_b[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse34 = t2_bbbb
        _cse35 = g_bbbb[o_b, v_b, o_b, o_b]
        _cse36 = g_bbbb[v_b, v_b, v_b, o_b]
        _tmp = einsum('ijba,aojm,beio->me', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bojm,aeio->me', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,aoim,bejo->me', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,boim,aejo->me', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oaij,beom->me', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obij,aeom->me', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oejm,abio->me', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oeim,abjo->me', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,aegm,gbij->me', _cse0, _cse7, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,abgj,geim->me', _cse0, _cse8, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,aejg,bgim->me', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,abgi,gejm->me', _cse0, _cse8, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,aeig,bgjm->me', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,begm,gaij->me', _cse0, _cse7, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bejg,agim->me', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,beig,agjm->me', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oejm,baio->me', _cse11, _cse12, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,boim,eajo->me', _cse11, _cse1, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oeim,baoj->me', _cse11, _cse5, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,boij,eamo->me', _cse11, _cse15, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeij,baom->me', _cse11, _cse17, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oajm,beio->me', _cse11, _cse18, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oaim,beoj->me', _cse11, _cse19, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oaij,beom->me', _cse11, _cse20, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,begm,gaij->me', _cse11, _cse7, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bagm,geij->me', _cse11, _cse21, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,begj,gaim->me', _cse11, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bagj,geim->me', _cse11, _cse23, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,beig,gajm->me', _cse11, _cse9, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,baig,gejm->me', _cse11, _cse25, _cse26, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,eagm,bgij->me', _cse11, _cse27, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eagj,bgim->me', _cse11, _cse28, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,objm,aeio->me', _cse11, _cse18, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,aoim,bejo->me', _cse11, _cse1, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,obim,aeoj->me', _cse11, _cse19, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,aoij,bemo->me', _cse11, _cse15, _cse30, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,obij,aeom->me', _cse11, _cse20, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oejm,abio->me', _cse11, _cse12, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oeim,aboj->me', _cse11, _cse5, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oeij,abom->me', _cse11, _cse17, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,abgm,geij->me', _cse11, _cse21, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,aegm,gbij->me', _cse11, _cse7, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,abgj,geim->me', _cse11, _cse23, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,aegj,gbim->me', _cse11, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,abig,gejm->me', _cse11, _cse25, _cse26, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,aeig,gbjm->me', _cse11, _cse9, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,begm,agij->me', _cse11, _cse31, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,begj,agim->me', _cse11, _cse32, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oeim,bajo->me', _cse11, _cse12, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,bojm,eaio->me', _cse11, _cse1, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oejm,baoi->me', _cse11, _cse5, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,boji,eamo->me', _cse11, _cse15, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oeji,baom->me', _cse11, _cse17, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oaim,bejo->me', _cse11, _cse18, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oajm,beoi->me', _cse11, _cse19, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oaji,beom->me', _cse11, _cse20, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,begm,gaji->me', _cse11, _cse7, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,bagm,geji->me', _cse11, _cse21, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,begi,gajm->me', _cse11, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,bagi,gejm->me', _cse11, _cse23, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,bejg,gaim->me', _cse11, _cse9, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,bajg,geim->me', _cse11, _cse25, _cse26, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,eagm,bgji->me', _cse11, _cse27, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,eagi,bgjm->me', _cse11, _cse28, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,obim,aejo->me', _cse11, _cse18, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,aojm,beio->me', _cse11, _cse1, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,objm,aeoi->me', _cse11, _cse19, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,aoji,bemo->me', _cse11, _cse15, _cse30, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,obji,aeom->me', _cse11, _cse20, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oeim,abjo->me', _cse11, _cse12, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oejm,aboi->me', _cse11, _cse5, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oeji,abom->me', _cse11, _cse17, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,abgm,geji->me', _cse11, _cse21, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,aegm,gbji->me', _cse11, _cse7, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,abgi,gejm->me', _cse11, _cse23, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,aegi,gbjm->me', _cse11, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,abjg,geim->me', _cse11, _cse25, _cse26, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,aejg,gbim->me', _cse11, _cse9, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,begm,agji->me', _cse11, _cse31, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,begi,agjm->me', _cse11, _cse32, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oejm,baio->me', _cse33, _cse12, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,objm,eaio->me', _cse33, _cse18, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oeim,bajo->me', _cse33, _cse12, _cse34, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,obim,eajo->me', _cse33, _cse18, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeij,bamo->me', _cse33, (g_bbbb[o_b, v_b, o_b, o_b] * Ve_b[None, :, None, None]), (t2_bbbb * Oe_b[None, None, :, None]), optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obij,eamo->me', _cse33, _cse35, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oajm,ebio->me', _cse33, _cse18, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oaim,ebjo->me', _cse33, _cse18, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oaij,ebmo->me', _cse33, _cse35, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgm,gaij->me', _cse33, _cse27, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eagm,gbij->me', _cse33, _cse27, _cse34, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,ebgj,gaim->me', _cse33, _cse28, _cse24, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,eagj,gbim->me', _cse33, _cse28, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgi,gajm->me', _cse33, _cse28, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eagi,gbjm->me', _cse33, _cse28, _cse24, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bagm,geij->me', _cse33, (g_bbbb[v_b, v_b, v_b, o_b] * Oe_b[None, None, None, :]), _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bagj,geim->me', _cse33, _cse36, _cse26, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bagi,gejm->me', _cse33, _cse36, _cse26, optimize=True)
        _iter -= 0.25 * _tmp
        m3_ov_b_t3 += _w * _iter
    return m3_ov_b_t3

