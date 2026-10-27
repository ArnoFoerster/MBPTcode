# GENERATED CODE -- do not edit by hand.
# Fully Laplace-fused t1_4 (order-4 singles) numerator, nested
# t1_4 -> t3_3 -> {t3_2,t4_2} (depth-qualified taus). No rank>=3
# tensor materialized. Produced by generate_mp4_t1_4_laplace_
# restricted.py; consumed by compute_t1_4_laplace.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum
from src.Base.utils.grids.grids import minimax_time_grid


def t1_4_aa_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t1_3_aa, t2_3_aaaa, t2_3_abab, t2_2_aaaa, t2_2_abab, t1_2_aa, t2_1_aaaa, t2_1_abab, ntau=6):
    t1_3_bb = t1_3_aa
    t2_3_bbbb = t2_3_aaaa
    t2_2_bbbb = t2_2_aaaa
    t1_2_bb = t1_2_aa
    t2_1_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, no))
    _iter = np.zeros((nv, no))
    _cse0 = t2_3_aaaa
    _cse1 = g_abab[o, o, o, v]
    _cse2 = t2_3_abab
    _cse3 = g_abab[v, o, v, v]
    _tmp = einsum('jabi,bj->ai', g_aaaa[o, v, v, o], t1_3_aa, optimize=True)
    _iter += 1 * _tmp
    _tmp = einsum('ajib,bj->ai', g_abab[v, o, o, v], t1_3_bb, optimize=True)
    _iter += 1 * _tmp
    _tmp = einsum('kjbi,bakj->ai', g_aaaa[o, o, v, o], _cse0, optimize=True)
    _iter -= 0.5 * _tmp
    _tmp = einsum('kjib,abkj->ai', _cse1, _cse2, optimize=True)
    _iter -= 0.5 * _tmp
    _tmp = einsum('jkib,abjk->ai', _cse1, _cse2, optimize=True)
    _iter -= 0.5 * _tmp
    _tmp = einsum('jabc,bcij->ai', g_aaaa[o, v, v, v], _cse0, optimize=True)
    _iter -= 0.5 * _tmp
    _tmp = einsum('ajbc,bcij->ai', _cse3, _cse2, optimize=True)
    _iter += 0.5 * _tmp
    _tmp = einsum('ajcb,cbij->ai', _cse3, _cse2, optimize=True)
    _iter += 0.5 * _tmp
    out += 1.0 * _iter
    for _tk_t3 in range(ntau):
        Oe_t3 = np.exp(ei * tau[_tk_t3])
        Ve_t3 = np.exp(-ea * tau[_tk_t3])
        _iter = np.zeros((nv, no))
        _cse0 = ((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :])
        _cse1 = ((t1_2_aa * Ve_t3[:, None]) * Oe_t3[None, :])
        _cse2 = (g_aaaa[v, v, o, o] * Ve_t3[None, :, None, None])
        _cse3 = (t1_2_aa * Oe_t3[None, :])
        _cse4 = (g_aaaa[v, v, o, o] * Oe_t3[None, None, :, None])
        _cse5 = (t1_2_aa * Ve_t3[:, None])
        _cse6 = ((g_aaaa[v, v, o, o] * Ve_t3[None, :, None, None]) * Oe_t3[None, None, :, None])
        _cse7 = t1_2_aa
        _cse8 = g_aaaa[o, v, o, o]
        _cse9 = ((t2_2_aaaa * Ve_t3[None, :, None, None]) * Oe_t3[None, None, :, None])
        _cse10 = (g_aaaa[o, v, o, o] * Oe_t3[None, None, :, None])
        _cse11 = (t2_2_aaaa * Ve_t3[None, :, None, None])
        _cse12 = (t2_2_aaaa * Oe_t3[None, None, :, None])
        _cse13 = ((g_aaaa[o, v, o, o] * Ve_t3[None, :, None, None]) * Oe_t3[None, None, :, None])
        _cse14 = t2_2_aaaa
        _cse15 = g_aaaa[v, v, v, o]
        _cse16 = (g_aaaa[v, v, v, o] * Ve_t3[None, :, None, None])
        _cse17 = ((g_aaaa[v, v, v, o] * Ve_t3[None, :, None, None]) * Oe_t3[None, None, None, :])
        _cse18 = ((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :])
        _cse19 = g_abab[v, v, o, o]
        _cse20 = (g_abab[v, v, o, o] * Oe_t3[None, None, :, None])
        _cse21 = t1_2_bb
        _cse22 = (g_abab[v, v, o, o] * Ve_t3[:, None, None, None])
        _cse23 = ((g_abab[v, v, o, o] * Ve_t3[:, None, None, None]) * Oe_t3[None, None, :, None])
        _cse24 = g_abab[v, o, o, o]
        _cse25 = ((t2_2_abab * Ve_t3[:, None, None, None]) * Oe_t3[None, None, :, None])
        _cse26 = (g_abab[v, o, o, o] * Ve_t3[:, None, None, None])
        _cse27 = (t2_2_abab * Oe_t3[None, None, :, None])
        _cse28 = (g_abab[v, o, o, o] * Oe_t3[None, None, :, None])
        _cse29 = (t2_2_abab * Ve_t3[:, None, None, None])
        _cse30 = ((g_abab[v, o, o, o] * Ve_t3[:, None, None, None]) * Oe_t3[None, None, :, None])
        _cse31 = t2_2_abab
        _cse32 = g_abab[o, v, o, o]
        _cse33 = (g_abab[o, v, o, o] * Oe_t3[None, None, :, None])
        _cse34 = g_abab[v, v, v, o]
        _cse35 = g_abab[v, v, o, v]
        _cse36 = (g_abab[v, v, o, v] * Oe_t3[None, None, :, None])
        _cse37 = (g_abab[v, v, v, o] * Ve_t3[:, None, None, None])
        _cse38 = (g_abab[v, v, o, v] * Ve_t3[:, None, None, None])
        _cse39 = ((g_abab[v, v, o, v] * Ve_t3[:, None, None, None]) * Oe_t3[None, None, :, None])
        _cse40 = ((g_aaaa[v, v, o, o] * Ve_t3[:, None, None, None]) * Oe_t3[None, None, :, None])
        _cse41 = ((t2_2_aaaa * Ve_t3[:, None, None, None]) * Oe_t3[None, None, :, None])
        _cse42 = (t2_2_aaaa * Ve_t3[:, None, None, None])
        _cse43 = (g_aaaa[v, v, v, o] * Ve_t3[:, None, None, None])
        _cse44 = ((g_aaaa[v, v, v, o] * Ve_t3[:, None, None, None]) * Oe_t3[None, None, None, :])
        _cse45 = ((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :])
        _cse46 = g_bbbb[o, v, o, o]
        _cse47 = t2_2_bbbb
        _cse48 = g_bbbb[v, v, v, o]
        _tmp = einsum('kjbc,bckj,ai->ai', _cse0, g_aaaa[v, v, o, o], _cse1, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bakj,ci->ai', _cse0, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcij,ak->ai', _cse0, _cse4, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,baij,ck->ai', _cse0, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bcik,aj->ai', _cse0, _cse4, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,baik,cj->ai', _cse0, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cakj,bi->ai', _cse0, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,caij,bk->ai', _cse0, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,caik,bj->ai', _cse0, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obkj,caio->ai', _cse0, _cse8, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,baio->ai', _cse0, _cse8, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obij,cako->ai', _cse0, _cse10, _cse11, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,bako->ai', _cse0, _cse10, _cse11, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,obik,cajo->ai', _cse0, _cse10, _cse11, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ocik,bajo->ai', _cse0, _cse10, _cse11, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oakj,bcio->ai', _cse0, (g_aaaa[o, v, o, o] * Ve_t3[None, :, None, None]), _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,oaij,bcko->ai', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,oaik,bcjo->ai', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgj,gaik->ai', _cse0, _cse15, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagj,gcik->ai', _cse0, _cse16, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bcgk,gaij->ai', _cse0, _cse15, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bagk,gcij->ai', _cse0, _cse16, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgi,gakj->ai', _cse0, (g_aaaa[v, v, v, o] * Oe_t3[None, None, None, :]), _cse11, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagi,gckj->ai', _cse0, _cse17, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cagj,gbik->ai', _cse0, _cse16, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cagk,gbij->ai', _cse0, _cse16, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cagi,gbkj->ai', _cse0, _cse17, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bckj,ai->ai', _cse18, _cse19, _cse1, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bcij,ak->ai', _cse18, _cse20, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,baik,cj->ai', _cse18, _cse6, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ackj,bi->ai', _cse18, _cse22, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acij,bk->ai', _cse18, _cse23, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bokj,acio->ai', _cse18, _cse24, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,aokj,bcio->ai', _cse18, _cse26, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,boij,acko->ai', _cse18, _cse28, _cse29, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,aoij,bcko->ai', _cse18, _cse30, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,obik,acoj->ai', _cse18, _cse10, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,oaik,bcoj->ai', _cse18, _cse13, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,baio->ai', _cse18, _cse32, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,bako->ai', _cse18, _cse33, _cse11, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgj,gaik->ai', _cse18, _cse34, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagk,gcij->ai', _cse18, _cse16, _cse27, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bckg,agij->ai', _cse18, _cse35, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bagi,gckj->ai', _cse18, _cse17, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bcig,agkj->ai', _cse18, _cse36, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acgj,gbik->ai', _cse18, _cse37, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ackg,bgij->ai', _cse18, _cse38, _cse27, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acig,bgkj->ai', _cse18, _cse39, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bcjk,ai->ai', _cse18, _cse19, _cse1, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bcik,aj->ai', _cse18, _cse20, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,baij,ck->ai', _cse18, _cse6, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,acjk,bi->ai', _cse18, _cse22, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,acik,bj->ai', _cse18, _cse23, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bojk,acio->ai', _cse18, _cse24, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,aojk,bcio->ai', _cse18, _cse26, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,boik,acjo->ai', _cse18, _cse28, _cse29, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,aoik,bcjo->ai', _cse18, _cse30, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,obij,acok->ai', _cse18, _cse10, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,oaij,bcok->ai', _cse18, _cse13, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,ocjk,baio->ai', _cse18, _cse32, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,ocik,bajo->ai', _cse18, _cse33, _cse11, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bcgk,gaij->ai', _cse18, _cse34, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bagj,gcik->ai', _cse18, _cse16, _cse27, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bcjg,agik->ai', _cse18, _cse35, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bagi,gcjk->ai', _cse18, _cse17, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bcig,agjk->ai', _cse18, _cse36, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,acgk,gbij->ai', _cse18, _cse37, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,acjg,bgik->ai', _cse18, _cse38, _cse27, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,acig,bgjk->ai', _cse18, _cse39, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,abkj,ci->ai', _cse18, _cse22, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,abij,ck->ai', _cse18, _cse23, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,acik,bj->ai', _cse18, _cse40, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cbkj,ai->ai', _cse18, _cse19, _cse1, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cbij,ak->ai', _cse18, _cse20, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,aokj,cbio->ai', _cse18, _cse26, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cokj,abio->ai', _cse18, _cse24, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,aoij,cbko->ai', _cse18, _cse30, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,coij,abko->ai', _cse18, _cse28, _cse29, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,oaik,cboj->ai', _cse18, _cse13, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,ocik,aboj->ai', _cse18, _cse10, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,obkj,acio->ai', _cse18, _cse32, _cse41, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,obij,acko->ai', _cse18, _cse33, _cse42, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,abgj,gcik->ai', _cse18, _cse37, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,acgk,gbij->ai', _cse18, _cse43, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,abkg,cgij->ai', _cse18, _cse38, _cse27, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,acgi,gbkj->ai', _cse18, _cse44, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,abig,cgkj->ai', _cse18, _cse39, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cbgj,gaik->ai', _cse18, _cse34, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,cbkg,agij->ai', _cse18, _cse35, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cbig,agkj->ai', _cse18, _cse36, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,abjk,ci->ai', _cse18, _cse22, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,abik,cj->ai', _cse18, _cse23, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,acij,bk->ai', _cse18, _cse40, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbjk,ai->ai', _cse18, _cse19, _cse1, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbik,aj->ai', _cse18, _cse20, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,aojk,cbio->ai', _cse18, _cse26, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cojk,abio->ai', _cse18, _cse24, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,aoik,cbjo->ai', _cse18, _cse30, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,coik,abjo->ai', _cse18, _cse28, _cse29, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,oaij,cbok->ai', _cse18, _cse13, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,ocij,abok->ai', _cse18, _cse10, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,objk,acio->ai', _cse18, _cse32, _cse41, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,obik,acjo->ai', _cse18, _cse33, _cse42, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,abgk,gcij->ai', _cse18, _cse37, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,acgj,gbik->ai', _cse18, _cse43, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,abjg,cgik->ai', _cse18, _cse38, _cse27, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,acgi,gbjk->ai', _cse18, _cse44, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,abig,cgjk->ai', _cse18, _cse39, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbgk,gaij->ai', _cse18, _cse34, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,cbjg,agik->ai', _cse18, _cse35, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbig,agjk->ai', _cse18, _cse36, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acij,bk->ai', _cse45, _cse23, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,abij,ck->ai', _cse45, _cse23, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acik,bj->ai', _cse45, _cse23, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,abik,cj->ai', _cse45, _cse23, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cbkj,ai->ai', _cse45, g_bbbb[v, v, o, o], _cse1, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,abio->ai', _cse45, _cse46, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,aoij,cbko->ai', _cse45, _cse30, _cse47, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,abok->ai', _cse45, _cse33, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,aoik,cbjo->ai', _cse45, _cse30, _cse47, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ocik,aboj->ai', _cse45, _cse33, _cse29, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obkj,acio->ai', _cse45, _cse46, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obij,acok->ai', _cse45, _cse33, _cse29, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obik,acoj->ai', _cse45, _cse33, _cse29, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acgj,gbik->ai', _cse45, _cse37, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,abgj,gcik->ai', _cse45, _cse37, _cse27, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acgk,gbij->ai', _cse45, _cse37, _cse27, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,abgk,gcij->ai', _cse45, _cse37, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,acig,gbkj->ai', _cse45, _cse39, _cse47, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,abig,gckj->ai', _cse45, _cse39, _cse47, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cbgj,agik->ai', _cse45, _cse48, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cbgk,agij->ai', _cse45, _cse48, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        out += (-sigma[_tk_t3]) * _iter
    for _tk_t3 in range(ntau):
        Oe_t3 = np.exp(ei * tau[_tk_t3])
        Ve_t3 = np.exp(-ea * tau[_tk_t3])
        for _tk_t3_d1 in range(ntau):
            Oe_t3_d1 = np.exp(ei * tau[_tk_t3_d1])
            Ve_t3_d1 = np.exp(-ea * tau[_tk_t3_d1])
            _iter = np.zeros((nv, no))
            _cse0 = ((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse1 = ((g_aaaa[o, o, o, o] * Oe_t3_d1[:, None, None, None]) * Oe_t3_d1[None, :, None, None])
            _cse2 = g_aaaa[o, v, o, o]
            _cse3 = ((((t2_1_aaaa * Ve_t3[None, :, None, None]) * Ve_t3_d1[None, :, None, None]) * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse4 = ((g_aaaa[o, v, o, o] * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse5 = ((t2_1_aaaa * Ve_t3[None, :, None, None]) * Ve_t3_d1[None, :, None, None])
            _cse6 = ((g_aaaa[o, v, o, o] * Ve_t3[None, :, None, None]) * Ve_t3_d1[None, :, None, None])
            _cse7 = ((t2_1_aaaa * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse8 = ((((g_aaaa[o, v, o, o] * Ve_t3[None, :, None, None]) * Ve_t3_d1[None, :, None, None]) * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse9 = t2_1_aaaa
            _cse10 = g_aaaa[v, v, v, o]
            _cse11 = ((g_aaaa[v, v, v, o] * Ve_t3[None, :, None, None]) * Ve_t3_d1[None, :, None, None])
            _cse12 = ((g_aaaa[v, v, v, o] * Oe_t3[None, None, None, :]) * Oe_t3_d1[None, None, None, :])
            _cse13 = ((((g_aaaa[v, v, v, o] * Ve_t3[None, :, None, None]) * Ve_t3_d1[None, :, None, None]) * Oe_t3[None, None, None, :]) * Oe_t3_d1[None, None, None, :])
            _cse14 = (((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse15 = (((g_aaaa[o, o, o, o] * Oe_t3_d1[:, None, None, None]) * Oe_t3_d1[None, :, None, None]) * Oe_t3[None, None, :, None])
            _cse16 = (((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse17 = ((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse18 = ((g_aaaa[o, v, v, o] * Oe_t3_d1[:, None, None, None]) * Ve_t3_d1[None, None, :, None])
            _cse19 = ((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse20 = ((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse21 = ((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse22 = (((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse23 = (((g_aaaa[o, v, v, o] * Oe_t3_d1[:, None, None, None]) * Ve_t3_d1[None, None, :, None]) * Oe_t3[None, None, None, :])
            _cse24 = (((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse25 = (((g_aaaa[o, v, v, o] * Oe_t3_d1[:, None, None, None]) * Ve_t3[None, :, None, None]) * Ve_t3_d1[None, None, :, None])
            _cse26 = ((((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse27 = ((((g_aaaa[o, v, v, o] * Oe_t3_d1[:, None, None, None]) * Ve_t3[None, :, None, None]) * Ve_t3_d1[None, None, :, None]) * Oe_t3[None, None, None, :])
            _cse28 = ((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse29 = ((g_aaaa[v, v, v, v] * Ve_t3_d1[None, None, :, None]) * Ve_t3_d1[None, None, None, :])
            _cse30 = (((g_aaaa[v, v, v, v] * Ve_t3[None, :, None, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3_d1[None, None, None, :])
            _cse31 = ((g_abab[v, o, o, v] * Oe_t3_d1[None, :, None, None]) * Ve_t3_d1[None, None, None, :])
            _cse32 = ((g_abab[v, o, o, o] * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None])
            _cse33 = ((t2_1_abab * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse34 = g_abab[v, o, o, o]
            _cse35 = ((((t2_1_abab * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None]) * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse36 = ((((g_abab[v, o, o, o] * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None]) * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse37 = t2_1_abab
            _cse38 = ((g_abab[v, o, o, o] * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse39 = ((t2_1_abab * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None])
            _cse40 = g_abab[o, v, o, o]
            _cse41 = ((((t2_1_aaaa * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None]) * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse42 = ((g_abab[o, v, o, o] * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse43 = ((t2_1_aaaa * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None])
            _cse44 = ((g_abab[v, v, v, o] * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None])
            _cse45 = ((g_aaaa[v, v, v, o] * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None])
            _cse46 = ((g_abab[v, v, o, v] * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None])
            _cse47 = ((((g_aaaa[v, v, v, o] * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None]) * Oe_t3[None, None, None, :]) * Oe_t3_d1[None, None, None, :])
            _cse48 = ((((g_abab[v, v, o, v] * Ve_t3[:, None, None, None]) * Ve_t3_d1[:, None, None, None]) * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse49 = g_abab[v, v, v, o]
            _cse50 = g_abab[v, v, o, v]
            _cse51 = ((g_abab[v, v, o, v] * Oe_t3[None, None, :, None]) * Oe_t3_d1[None, None, :, None])
            _cse52 = (((g_abab[v, o, o, v] * Oe_t3_d1[None, :, None, None]) * Oe_t3[None, None, :, None]) * Ve_t3_d1[None, None, None, :])
            _cse53 = (((g_abab[v, o, o, v] * Ve_t3[:, None, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3_d1[None, None, None, :])
            _cse54 = ((((g_abab[v, o, o, v] * Ve_t3[:, None, None, None]) * Oe_t3_d1[None, :, None, None]) * Oe_t3[None, None, :, None]) * Ve_t3_d1[None, None, None, :])
            _cse55 = ((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse56 = ((g_abab[o, v, v, o] * Oe_t3_d1[:, None, None, None]) * Ve_t3_d1[None, None, :, None])
            _cse57 = ((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse58 = ((g_abab[o, o, o, o] * Oe_t3_d1[:, None, None, None]) * Oe_t3_d1[None, :, None, None])
            _cse59 = (((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse60 = (((g_abab[o, o, o, o] * Oe_t3_d1[:, None, None, None]) * Oe_t3_d1[None, :, None, None]) * Oe_t3[None, None, :, None])
            _cse61 = (((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse62 = ((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse63 = ((g_abab[v, o, v, o] * Oe_t3_d1[None, :, None, None]) * Ve_t3_d1[None, None, :, None])
            _cse64 = (((g_abab[v, o, v, o] * Ve_t3[:, None, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3_d1[None, None, :, None])
            _cse65 = ((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse66 = (((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse67 = ((((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse68 = ((g_bbbb[o, v, v, o] * Oe_t3_d1[:, None, None, None]) * Ve_t3_d1[None, None, :, None])
            _cse69 = ((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse70 = ((g_abab[o, v, o, v] * Oe_t3_d1[:, None, None, None]) * Ve_t3_d1[None, None, None, :])
            _cse71 = (((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse72 = (((g_abab[o, v, o, v] * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, None, :, None]) * Ve_t3_d1[None, None, None, :])
            _cse73 = ((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse74 = ((g_abab[v, v, v, v] * Ve_t3_d1[None, None, :, None]) * Ve_t3_d1[None, None, None, :])
            _cse75 = (((g_abab[v, v, v, v] * Ve_t3[:, None, None, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3_d1[None, None, None, :])
            _cse76 = g_bbbb[o, v, o, o]
            _cse77 = t2_1_bbbb
            _cse78 = g_bbbb[v, v, v, o]
            _cse79 = (((g_aaaa[v, v, v, v] * Ve_t3[:, None, None, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3_d1[None, None, None, :])
            _cse80 = ((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse81 = ((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse82 = ((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse83 = ((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse84 = ((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse85 = ((g_bbbb[o, o, o, o] * Oe_t3_d1[:, None, None, None]) * Oe_t3_d1[None, :, None, None])
            _cse86 = (((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse87 = (((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse88 = ((((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse89 = (((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse90 = (((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t3_d1[None, None, None, :])
            _cse91 = ((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t3_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse92 = ((g_bbbb[v, v, v, v] * Ve_t3_d1[None, None, :, None]) * Ve_t3_d1[None, None, None, :])
            _tmp = einsum('kjbc,pokj,qbpo,caiq->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qcpo,baiq->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qbio,capq->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qcio,bapq->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qbip,caoq->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qcip,baoq->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qapo,bciq->ai', _cse0, _cse1, _cse6, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qaio,bcpq->ai', _cse0, _cse1, _cse8, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qaip,bcoq->ai', _cse0, _cse1, _cse8, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bcgo,gaip->ai', _cse0, _cse1, _cse10, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bago,gcip->ai', _cse0, _cse1, _cse11, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bcgp,gaio->ai', _cse0, _cse1, _cse10, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bagp,gcio->ai', _cse0, _cse1, _cse11, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bcgi,gapo->ai', _cse0, _cse1, _cse12, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bagi,gcpo->ai', _cse0, _cse1, _cse13, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,cago,gbip->ai', _cse0, _cse1, _cse11, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,cagp,gbio->ai', _cse0, _cse1, _cse11, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,cagi,gbpo->ai', _cse0, _cse1, _cse13, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qbpo,cakq->ai', _cse14, _cse15, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qcpo,bakq->ai', _cse14, _cse15, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qbko,capq->ai', _cse14, _cse15, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qcko,bapq->ai', _cse14, _cse15, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qbkp,caoq->ai', _cse14, _cse15, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qckp,baoq->ai', _cse14, _cse15, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qapo,bckq->ai', _cse14, _cse15, _cse6, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qako,bcpq->ai', _cse14, _cse15, _cse6, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qakp,bcoq->ai', _cse14, _cse15, _cse6, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bcgo,gakp->ai', _cse14, _cse15, _cse10, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bago,gckp->ai', _cse14, _cse15, _cse11, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bcgp,gako->ai', _cse14, _cse15, _cse10, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bagp,gcko->ai', _cse14, _cse15, _cse11, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bcgk,gapo->ai', _cse14, _cse15, _cse10, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bagk,gcpo->ai', _cse14, _cse15, _cse11, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,cago,gbkp->ai', _cse14, _cse15, _cse11, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,cagp,gbko->ai', _cse14, _cse15, _cse11, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,cagk,gbpo->ai', _cse14, _cse15, _cse11, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qbpo,cajq->ai', _cse16, _cse15, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qcpo,bajq->ai', _cse16, _cse15, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qbjo,capq->ai', _cse16, _cse15, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qcjo,bapq->ai', _cse16, _cse15, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qbjp,caoq->ai', _cse16, _cse15, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qcjp,baoq->ai', _cse16, _cse15, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qapo,bcjq->ai', _cse16, _cse15, _cse6, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qajo,bcpq->ai', _cse16, _cse15, _cse6, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qajp,bcoq->ai', _cse16, _cse15, _cse6, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bcgo,gajp->ai', _cse16, _cse15, _cse10, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bago,gcjp->ai', _cse16, _cse15, _cse11, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bcgp,gajo->ai', _cse16, _cse15, _cse10, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bagp,gcjo->ai', _cse16, _cse15, _cse11, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bcgj,gapo->ai', _cse16, _cse15, _cse10, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bagj,gcpo->ai', _cse16, _cse15, _cse11, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,cago,gbjp->ai', _cse16, _cse15, _cse11, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,cagp,gbjo->ai', _cse16, _cse15, _cse11, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,cagj,gbpo->ai', _cse16, _cse15, _cse11, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgj,pgko,caip->ai', _cse17, _cse18, _cse2, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pcko,gaip->ai', _cse17, _cse18, _cse2, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pgio,cakp->ai', _cse17, _cse18, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pcio,gakp->ai', _cse17, _cse18, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pgik,caop->ai', _cse17, _cse18, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pcik,gaop->ai', _cse17, _cse18, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pako,gcip->ai', _cse17, _cse18, _cse6, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,paio,gckp->ai', _cse17, _cse18, _cse8, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,paik,gcop->ai', _cse17, _cse18, _cse8, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gcho,haik->ai', _cse17, _cse18, _cse10, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gaho,hcik->ai', _cse17, _cse18, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gchk,haio->ai', _cse17, _cse18, _cse10, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gahk,hcio->ai', _cse17, _cse18, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gchi,hako->ai', _cse17, _cse18, _cse12, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gahi,hcko->ai', _cse17, _cse18, _cse13, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,caho,hgik->ai', _cse17, _cse18, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,cahk,hgio->ai', _cse17, _cse18, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,cahi,hgko->ai', _cse17, _cse18, _cse13, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgko,baip->ai', _cse19, _cse18, _cse2, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbko,gaip->ai', _cse19, _cse18, _cse2, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgio,bakp->ai', _cse19, _cse18, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbio,gakp->ai', _cse19, _cse18, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgik,baop->ai', _cse19, _cse18, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbik,gaop->ai', _cse19, _cse18, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pako,gbip->ai', _cse19, _cse18, _cse6, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,paio,gbkp->ai', _cse19, _cse18, _cse8, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,paik,gbop->ai', _cse19, _cse18, _cse8, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbho,haik->ai', _cse19, _cse18, _cse10, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gaho,hbik->ai', _cse19, _cse18, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbhk,haio->ai', _cse19, _cse18, _cse10, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gahk,hbio->ai', _cse19, _cse18, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbhi,hako->ai', _cse19, _cse18, _cse12, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gahi,hbko->ai', _cse19, _cse18, _cse13, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,baho,hgik->ai', _cse19, _cse18, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,bahk,hgio->ai', _cse19, _cse18, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,bahi,hgko->ai', _cse19, _cse18, _cse13, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pgjo,caip->ai', _cse20, _cse18, _cse2, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcjo,gaip->ai', _cse20, _cse18, _cse2, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pgio,cajp->ai', _cse20, _cse18, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcio,gajp->ai', _cse20, _cse18, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pgij,caop->ai', _cse20, _cse18, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcij,gaop->ai', _cse20, _cse18, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pajo,gcip->ai', _cse20, _cse18, _cse6, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,paio,gcjp->ai', _cse20, _cse18, _cse8, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,paij,gcop->ai', _cse20, _cse18, _cse8, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gcho,haij->ai', _cse20, _cse18, _cse10, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gaho,hcij->ai', _cse20, _cse18, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gchj,haio->ai', _cse20, _cse18, _cse10, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gahj,hcio->ai', _cse20, _cse18, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gchi,hajo->ai', _cse20, _cse18, _cse12, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gahi,hcjo->ai', _cse20, _cse18, _cse13, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,caho,hgij->ai', _cse20, _cse18, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,cahj,hgio->ai', _cse20, _cse18, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,cahi,hgjo->ai', _cse20, _cse18, _cse13, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pgjo,baip->ai', _cse21, _cse18, _cse2, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pbjo,gaip->ai', _cse21, _cse18, _cse2, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pgio,bajp->ai', _cse21, _cse18, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pbio,gajp->ai', _cse21, _cse18, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pgij,baop->ai', _cse21, _cse18, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pbij,gaop->ai', _cse21, _cse18, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pajo,gbip->ai', _cse21, _cse18, _cse6, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,paio,gbjp->ai', _cse21, _cse18, _cse8, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,paij,gbop->ai', _cse21, _cse18, _cse8, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gbho,haij->ai', _cse21, _cse18, _cse10, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gaho,hbij->ai', _cse21, _cse18, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gbhj,haio->ai', _cse21, _cse18, _cse10, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gahj,hbio->ai', _cse21, _cse18, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gbhi,hajo->ai', _cse21, _cse18, _cse12, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gahi,hbjo->ai', _cse21, _cse18, _cse13, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,baho,hgij->ai', _cse21, _cse18, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,bahj,hgio->ai', _cse21, _cse18, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,bahi,hgjo->ai', _cse21, _cse18, _cse13, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pgjo,cakp->ai', _cse22, _cse23, _cse2, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pcjo,gakp->ai', _cse22, _cse23, _cse2, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pgko,cajp->ai', _cse22, _cse23, _cse2, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pcko,gajp->ai', _cse22, _cse23, _cse2, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pgkj,caop->ai', _cse22, _cse23, _cse2, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pckj,gaop->ai', _cse22, _cse23, _cse2, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pajo,gckp->ai', _cse22, _cse23, _cse6, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pako,gcjp->ai', _cse22, _cse23, _cse6, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pakj,gcop->ai', _cse22, _cse23, _cse6, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gcho,hakj->ai', _cse22, _cse23, _cse10, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gaho,hckj->ai', _cse22, _cse23, _cse11, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gchj,hako->ai', _cse22, _cse23, _cse10, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gahj,hcko->ai', _cse22, _cse23, _cse11, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gchk,hajo->ai', _cse22, _cse23, _cse10, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gahk,hcjo->ai', _cse22, _cse23, _cse11, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,caho,hgkj->ai', _cse22, _cse23, _cse11, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,cahj,hgko->ai', _cse22, _cse23, _cse11, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,cahk,hgjo->ai', _cse22, _cse23, _cse11, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,pgjo,bakp->ai', _cse24, _cse23, _cse2, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,pbjo,gakp->ai', _cse24, _cse23, _cse2, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,pgko,bajp->ai', _cse24, _cse23, _cse2, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,pbko,gajp->ai', _cse24, _cse23, _cse2, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,pgkj,baop->ai', _cse24, _cse23, _cse2, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,pbkj,gaop->ai', _cse24, _cse23, _cse2, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,pajo,gbkp->ai', _cse24, _cse23, _cse6, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,pako,gbjp->ai', _cse24, _cse23, _cse6, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,pakj,gbop->ai', _cse24, _cse23, _cse6, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,gbho,hakj->ai', _cse24, _cse23, _cse10, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,gaho,hbkj->ai', _cse24, _cse23, _cse11, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,gbhj,hako->ai', _cse24, _cse23, _cse10, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,gahj,hbko->ai', _cse24, _cse23, _cse11, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,gbhk,hajo->ai', _cse24, _cse23, _cse10, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,gahk,hbjo->ai', _cse24, _cse23, _cse11, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,baho,hgkj->ai', _cse24, _cse23, _cse11, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,bahj,hgko->ai', _cse24, _cse23, _cse11, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgi,bahk,hgjo->ai', _cse24, _cse23, _cse11, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,pgko,bcip->ai', _cse14, _cse25, _cse2, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,pbko,gcip->ai', _cse14, _cse25, _cse2, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,pgio,bckp->ai', _cse14, _cse25, _cse4, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,pbio,gckp->ai', _cse14, _cse25, _cse4, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,pgik,bcop->ai', _cse14, _cse25, _cse4, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,pbik,gcop->ai', _cse14, _cse25, _cse4, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,pcko,gbip->ai', _cse14, _cse25, _cse2, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,pcio,gbkp->ai', _cse14, _cse25, _cse4, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,pcik,gbop->ai', _cse14, _cse25, _cse4, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,gbho,hcik->ai', _cse14, _cse25, _cse10, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,gcho,hbik->ai', _cse14, _cse25, _cse10, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,gbhk,hcio->ai', _cse14, _cse25, _cse10, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,gchk,hbio->ai', _cse14, _cse25, _cse10, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,gbhi,hcko->ai', _cse14, _cse25, _cse12, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,gchi,hbko->ai', _cse14, _cse25, _cse12, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,bcho,hgik->ai', _cse14, _cse25, _cse10, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,bchk,hgio->ai', _cse14, _cse25, _cse10, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagj,bchi,hgko->ai', _cse14, _cse25, _cse12, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pgjo,bcip->ai', _cse16, _cse25, _cse2, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pbjo,gcip->ai', _cse16, _cse25, _cse2, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pgio,bcjp->ai', _cse16, _cse25, _cse4, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pbio,gcjp->ai', _cse16, _cse25, _cse4, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pgij,bcop->ai', _cse16, _cse25, _cse4, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pbij,gcop->ai', _cse16, _cse25, _cse4, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pcjo,gbip->ai', _cse16, _cse25, _cse2, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pcio,gbjp->ai', _cse16, _cse25, _cse4, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pcij,gbop->ai', _cse16, _cse25, _cse4, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gbho,hcij->ai', _cse16, _cse25, _cse10, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gcho,hbij->ai', _cse16, _cse25, _cse10, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gbhj,hcio->ai', _cse16, _cse25, _cse10, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gchj,hbio->ai', _cse16, _cse25, _cse10, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gbhi,hcjo->ai', _cse16, _cse25, _cse12, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gchi,hbjo->ai', _cse16, _cse25, _cse12, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,bcho,hgij->ai', _cse16, _cse25, _cse10, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,bchj,hgio->ai', _cse16, _cse25, _cse10, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,bchi,hgjo->ai', _cse16, _cse25, _cse12, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pgjo,bckp->ai', _cse26, _cse27, _cse2, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pbjo,gckp->ai', _cse26, _cse27, _cse2, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pgko,bcjp->ai', _cse26, _cse27, _cse2, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pbko,gcjp->ai', _cse26, _cse27, _cse2, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pgkj,bcop->ai', _cse26, _cse27, _cse2, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pbkj,gcop->ai', _cse26, _cse27, _cse2, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pcjo,gbkp->ai', _cse26, _cse27, _cse2, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pcko,gbjp->ai', _cse26, _cse27, _cse2, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pckj,gbop->ai', _cse26, _cse27, _cse2, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gbho,hckj->ai', _cse26, _cse27, _cse10, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gcho,hbkj->ai', _cse26, _cse27, _cse10, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gbhj,hcko->ai', _cse26, _cse27, _cse10, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gchj,hbko->ai', _cse26, _cse27, _cse10, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gbhk,hcjo->ai', _cse26, _cse27, _cse10, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gchk,hbjo->ai', _cse26, _cse27, _cse10, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,bcho,hgkj->ai', _cse26, _cse27, _cse10, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,bchj,hgko->ai', _cse26, _cse27, _cse10, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,bchk,hgjo->ai', _cse26, _cse27, _cse10, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bcgh,ogkj,haio->ai', _cse28, _cse29, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ohkj,gaio->ai', _cse28, _cse29, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ogij,hako->ai', _cse28, _cse29, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ohij,gako->ai', _cse28, _cse29, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ogik,hajo->ai', _cse28, _cse29, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ohik,gajo->ai', _cse28, _cse29, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,oakj,ghio->ai', _cse28, _cse29, _cse6, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,oaij,ghko->ai', _cse28, _cse29, _cse8, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,oaik,ghjo->ai', _cse28, _cse29, _cse8, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ghuj,uaik->ai', _cse28, _cse29, _cse10, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,gauj,uhik->ai', _cse28, _cse29, _cse11, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ghuk,uaij->ai', _cse28, _cse29, _cse10, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,gauk,uhij->ai', _cse28, _cse29, _cse11, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ghui,uakj->ai', _cse28, _cse29, _cse12, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,gaui,uhkj->ai', _cse28, _cse29, _cse13, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,hauj,ugik->ai', _cse28, _cse29, _cse11, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,hauk,ugij->ai', _cse28, _cse29, _cse11, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,haui,ugkj->ai', _cse28, _cse29, _cse13, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ogkj,hcio->ai', _cse22, _cse30, _cse2, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ohkj,gcio->ai', _cse22, _cse30, _cse2, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ogij,hcko->ai', _cse22, _cse30, _cse4, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ohij,gcko->ai', _cse22, _cse30, _cse4, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ogik,hcjo->ai', _cse22, _cse30, _cse4, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ohik,gcjo->ai', _cse22, _cse30, _cse4, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ockj,ghio->ai', _cse22, _cse30, _cse2, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ocij,ghko->ai', _cse22, _cse30, _cse4, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ocik,ghjo->ai', _cse22, _cse30, _cse4, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ghuj,ucik->ai', _cse22, _cse30, _cse10, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,gcuj,uhik->ai', _cse22, _cse30, _cse10, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ghuk,ucij->ai', _cse22, _cse30, _cse10, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,gcuk,uhij->ai', _cse22, _cse30, _cse10, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ghui,uckj->ai', _cse22, _cse30, _cse12, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,gcui,uhkj->ai', _cse22, _cse30, _cse12, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,hcuj,ugik->ai', _cse22, _cse30, _cse10, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,hcuk,ugij->ai', _cse22, _cse30, _cse10, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,hcui,ugkj->ai', _cse22, _cse30, _cse12, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,ogkj,hbio->ai', _cse24, _cse30, _cse2, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,ohkj,gbio->ai', _cse24, _cse30, _cse2, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,ogij,hbko->ai', _cse24, _cse30, _cse4, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,ohij,gbko->ai', _cse24, _cse30, _cse4, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,ogik,hbjo->ai', _cse24, _cse30, _cse4, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,ohik,gbjo->ai', _cse24, _cse30, _cse4, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,obkj,ghio->ai', _cse24, _cse30, _cse2, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,obij,ghko->ai', _cse24, _cse30, _cse4, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,obik,ghjo->ai', _cse24, _cse30, _cse4, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,ghuj,ubik->ai', _cse24, _cse30, _cse10, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,gbuj,uhik->ai', _cse24, _cse30, _cse10, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,ghuk,ubij->ai', _cse24, _cse30, _cse10, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,gbuk,uhij->ai', _cse24, _cse30, _cse10, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,ghui,ubkj->ai', _cse24, _cse30, _cse12, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,gbui,uhkj->ai', _cse24, _cse30, _cse12, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,hbuj,ugik->ai', _cse24, _cse30, _cse10, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,hbuk,ugij->ai', _cse24, _cse30, _cse10, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cagh,hbui,ugkj->ai', _cse24, _cse30, _cse12, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bojg,apko,cgip->ai', _cse17, _cse31, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,cpko,agip->ai', _cse17, _cse31, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,apio,cgkp->ai', _cse17, _cse31, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,cpio,agkp->ai', _cse17, _cse31, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,paik,cgpo->ai', _cse17, _cse31, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,pcik,agpo->ai', _cse17, _cse31, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,pgko,acip->ai', _cse17, _cse31, _cse40, _cse41, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,pgio,ackp->ai', _cse17, _cse31, _cse42, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,agho,hcik->ai', _cse17, _cse31, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,achk,hgio->ai', _cse17, _cse31, _cse45, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,agkh,chio->ai', _cse17, _cse31, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,achi,hgko->ai', _cse17, _cse31, _cse47, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,agih,chko->ai', _cse17, _cse31, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,cgho,haik->ai', _cse17, _cse31, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,cgkh,ahio->ai', _cse17, _cse31, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bojg,cgih,ahko->ai', _cse17, _cse31, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,apko,bgip->ai', _cse19, _cse31, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,bpko,agip->ai', _cse19, _cse31, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,apio,bgkp->ai', _cse19, _cse31, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,bpio,agkp->ai', _cse19, _cse31, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,paik,bgpo->ai', _cse19, _cse31, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,pbik,agpo->ai', _cse19, _cse31, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,pgko,abip->ai', _cse19, _cse31, _cse40, _cse41, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,pgio,abkp->ai', _cse19, _cse31, _cse42, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,agho,hbik->ai', _cse19, _cse31, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,abhk,hgio->ai', _cse19, _cse31, _cse45, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,agkh,bhio->ai', _cse19, _cse31, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,abhi,hgko->ai', _cse19, _cse31, _cse47, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,agih,bhko->ai', _cse19, _cse31, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,bgho,haik->ai', _cse19, _cse31, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,bgkh,ahio->ai', _cse19, _cse31, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cojg,bgih,ahko->ai', _cse19, _cse31, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,apjo,cgip->ai', _cse20, _cse31, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,cpjo,agip->ai', _cse20, _cse31, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,apio,cgjp->ai', _cse20, _cse31, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,cpio,agjp->ai', _cse20, _cse31, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,paij,cgpo->ai', _cse20, _cse31, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,pcij,agpo->ai', _cse20, _cse31, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,pgjo,acip->ai', _cse20, _cse31, _cse40, _cse41, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,pgio,acjp->ai', _cse20, _cse31, _cse42, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,agho,hcij->ai', _cse20, _cse31, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,achj,hgio->ai', _cse20, _cse31, _cse45, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,agjh,chio->ai', _cse20, _cse31, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,achi,hgjo->ai', _cse20, _cse31, _cse47, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,agih,chjo->ai', _cse20, _cse31, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,cgho,haij->ai', _cse20, _cse31, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,cgjh,ahio->ai', _cse20, _cse31, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,cgih,ahjo->ai', _cse20, _cse31, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,apjo,bgip->ai', _cse21, _cse31, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,bpjo,agip->ai', _cse21, _cse31, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,apio,bgjp->ai', _cse21, _cse31, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,bpio,agjp->ai', _cse21, _cse31, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,paij,bgpo->ai', _cse21, _cse31, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,pbij,agpo->ai', _cse21, _cse31, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,pgjo,abip->ai', _cse21, _cse31, _cse40, _cse41, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,pgio,abjp->ai', _cse21, _cse31, _cse42, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,agho,hbij->ai', _cse21, _cse31, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,abhj,hgio->ai', _cse21, _cse31, _cse45, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,agjh,bhio->ai', _cse21, _cse31, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,abhi,hgjo->ai', _cse21, _cse31, _cse47, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,agih,bhjo->ai', _cse21, _cse31, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,bgho,haij->ai', _cse21, _cse31, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,bgjh,ahio->ai', _cse21, _cse31, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,cokg,bgih,ahjo->ai', _cse21, _cse31, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,apjo,cgkp->ai', _cse22, _cse52, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,cpjo,agkp->ai', _cse22, _cse52, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,apko,cgjp->ai', _cse22, _cse52, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,cpko,agjp->ai', _cse22, _cse52, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pakj,cgpo->ai', _cse22, _cse52, _cse6, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pckj,agpo->ai', _cse22, _cse52, _cse2, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pgjo,ackp->ai', _cse22, _cse52, _cse40, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pgko,acjp->ai', _cse22, _cse52, _cse40, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,agho,hckj->ai', _cse22, _cse52, _cse44, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,achj,hgko->ai', _cse22, _cse52, _cse45, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,agjh,chko->ai', _cse22, _cse52, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,achk,hgjo->ai', _cse22, _cse52, _cse45, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,agkh,chjo->ai', _cse22, _cse52, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,cgho,hakj->ai', _cse22, _cse52, _cse49, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,cgjh,ahko->ai', _cse22, _cse52, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,cgkh,ahjo->ai', _cse22, _cse52, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,coig,apjo,bgkp->ai', _cse24, _cse52, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,coig,bpjo,agkp->ai', _cse24, _cse52, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,coig,apko,bgjp->ai', _cse24, _cse52, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,coig,bpko,agjp->ai', _cse24, _cse52, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,coig,pakj,bgpo->ai', _cse24, _cse52, _cse6, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,coig,pbkj,agpo->ai', _cse24, _cse52, _cse2, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,coig,pgjo,abkp->ai', _cse24, _cse52, _cse40, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,coig,pgko,abjp->ai', _cse24, _cse52, _cse40, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,coig,agho,hbkj->ai', _cse24, _cse52, _cse44, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,coig,abhj,hgko->ai', _cse24, _cse52, _cse45, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,coig,agjh,bhko->ai', _cse24, _cse52, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,coig,abhk,hgjo->ai', _cse24, _cse52, _cse45, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,coig,agkh,bhjo->ai', _cse24, _cse52, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,coig,bgho,hakj->ai', _cse24, _cse52, _cse49, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,coig,bgjh,ahko->ai', _cse24, _cse52, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,coig,bgkh,ahjo->ai', _cse24, _cse52, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,cpko,bgip->ai', _cse14, _cse53, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,bpko,cgip->ai', _cse14, _cse53, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,cpio,bgkp->ai', _cse14, _cse53, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,bpio,cgkp->ai', _cse14, _cse53, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,pcik,bgpo->ai', _cse14, _cse53, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,pbik,cgpo->ai', _cse14, _cse53, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,pgko,cbip->ai', _cse14, _cse53, _cse40, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,pgio,cbkp->ai', _cse14, _cse53, _cse42, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,cgho,hbik->ai', _cse14, _cse53, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,cbhk,hgio->ai', _cse14, _cse53, _cse10, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,cgkh,bhio->ai', _cse14, _cse53, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,cbhi,hgko->ai', _cse14, _cse53, _cse12, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,cgih,bhko->ai', _cse14, _cse53, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,bgho,hcik->ai', _cse14, _cse53, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,bgkh,chio->ai', _cse14, _cse53, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aojg,bgih,chko->ai', _cse14, _cse53, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,cpjo,bgip->ai', _cse16, _cse53, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bpjo,cgip->ai', _cse16, _cse53, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,cpio,bgjp->ai', _cse16, _cse53, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bpio,cgjp->ai', _cse16, _cse53, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pcij,bgpo->ai', _cse16, _cse53, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pbij,cgpo->ai', _cse16, _cse53, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pgjo,cbip->ai', _cse16, _cse53, _cse40, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pgio,cbjp->ai', _cse16, _cse53, _cse42, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,cgho,hbij->ai', _cse16, _cse53, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,cbhj,hgio->ai', _cse16, _cse53, _cse10, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,cgjh,bhio->ai', _cse16, _cse53, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,cbhi,hgjo->ai', _cse16, _cse53, _cse12, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,cgih,bhjo->ai', _cse16, _cse53, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bgho,hcij->ai', _cse16, _cse53, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bgjh,chio->ai', _cse16, _cse53, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bgih,chjo->ai', _cse16, _cse53, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cpjo,bgkp->ai', _cse26, _cse54, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bpjo,cgkp->ai', _cse26, _cse54, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cpko,bgjp->ai', _cse26, _cse54, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bpko,cgjp->ai', _cse26, _cse54, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pckj,bgpo->ai', _cse26, _cse54, _cse2, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pbkj,cgpo->ai', _cse26, _cse54, _cse2, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pgjo,cbkp->ai', _cse26, _cse54, _cse40, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pgko,cbjp->ai', _cse26, _cse54, _cse40, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cgho,hbkj->ai', _cse26, _cse54, _cse49, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cbhj,hgko->ai', _cse26, _cse54, _cse10, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cgjh,bhko->ai', _cse26, _cse54, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cbhk,hgjo->ai', _cse26, _cse54, _cse10, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cgkh,bhjo->ai', _cse26, _cse54, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bgho,hckj->ai', _cse26, _cse54, _cse49, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bgjh,chko->ai', _cse26, _cse54, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bgkh,chjo->ai', _cse26, _cse54, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgko,baip->ai', _cse55, _cse56, _cse2, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbko,gaip->ai', _cse55, _cse56, _cse2, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgio,bakp->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbio,gakp->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgik,baop->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbik,gaop->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pako,gbip->ai', _cse55, _cse56, _cse6, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,paio,gbkp->ai', _cse55, _cse56, _cse8, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,paik,gbop->ai', _cse55, _cse56, _cse8, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbho,haik->ai', _cse55, _cse56, _cse10, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gaho,hbik->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbhk,haio->ai', _cse55, _cse56, _cse10, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gahk,hbio->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbhi,hako->ai', _cse55, _cse56, _cse12, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gahi,hbko->ai', _cse55, _cse56, _cse13, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,baho,hgik->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,bahk,hgio->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,bahi,hgko->ai', _cse55, _cse56, _cse13, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,pokj,bqpo,aciq->ai', _cse57, _cse58, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,aqpo,bciq->ai', _cse57, _cse58, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bqio,acpq->ai', _cse57, _cse58, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,aqio,bcpq->ai', _cse57, _cse58, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qbip,acqo->ai', _cse57, _cse58, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qaip,bcqo->ai', _cse57, _cse58, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qcpo,baiq->ai', _cse57, _cse58, _cse40, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qcio,bapq->ai', _cse57, _cse58, _cse42, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bcgo,gaip->ai', _cse57, _cse58, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bagp,gcio->ai', _cse57, _cse58, _cse11, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bcpg,agio->ai', _cse57, _cse58, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bagi,gcpo->ai', _cse57, _cse58, _cse13, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,bcig,agpo->ai', _cse57, _cse58, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,acgo,gbip->ai', _cse57, _cse58, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,acpg,bgio->ai', _cse57, _cse58, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,acig,bgpo->ai', _cse57, _cse58, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bqpo,ackq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,aqpo,bckq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bqko,acpq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,aqko,bcpq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qbkp,acqo->ai', _cse59, _cse60, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qakp,bcqo->ai', _cse59, _cse60, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qcpo,bakq->ai', _cse59, _cse60, _cse40, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qcko,bapq->ai', _cse59, _cse60, _cse40, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bcgo,gakp->ai', _cse59, _cse60, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bagp,gcko->ai', _cse59, _cse60, _cse11, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bcpg,agko->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bagk,gcpo->ai', _cse59, _cse60, _cse11, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,bckg,agpo->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,acgo,gbkp->ai', _cse59, _cse60, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,acpg,bgko->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,ackg,bgpo->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,bqop,aciq->ai', _cse57, _cse58, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,aqop,bciq->ai', _cse57, _cse58, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,bqip,acoq->ai', _cse57, _cse58, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,aqip,bcoq->ai', _cse57, _cse58, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,qbio,acqp->ai', _cse57, _cse58, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,qaio,bcqp->ai', _cse57, _cse58, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,qcop,baiq->ai', _cse57, _cse58, _cse40, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,qcip,baoq->ai', _cse57, _cse58, _cse42, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,bcgp,gaio->ai', _cse57, _cse58, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,bago,gcip->ai', _cse57, _cse58, _cse11, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,bcog,agip->ai', _cse57, _cse58, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,bagi,gcop->ai', _cse57, _cse58, _cse13, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,bcig,agop->ai', _cse57, _cse58, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,acgp,gbio->ai', _cse57, _cse58, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,acog,bgip->ai', _cse57, _cse58, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkj,acig,bgop->ai', _cse57, _cse58, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,bqop,ackq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,aqop,bckq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,bqkp,acoq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,aqkp,bcoq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qbko,acqp->ai', _cse59, _cse60, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qako,bcqp->ai', _cse59, _cse60, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qcop,bakq->ai', _cse59, _cse60, _cse40, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qckp,baoq->ai', _cse59, _cse60, _cse40, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,bcgp,gako->ai', _cse59, _cse60, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,bago,gckp->ai', _cse59, _cse60, _cse11, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,bcog,agkp->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,bagk,gcop->ai', _cse59, _cse60, _cse11, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,bckg,agop->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,acgp,gbko->ai', _cse59, _cse60, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,acog,bgkp->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,ackg,bgop->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bqpj,acoq->ai', _cse61, _cse15, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,aqpj,bcoq->ai', _cse61, _cse15, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bqoj,acpq->ai', _cse61, _cse15, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,aqoj,bcpq->ai', _cse61, _cse15, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qbop,acqj->ai', _cse61, _cse15, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qaop,bcqj->ai', _cse61, _cse15, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qcpj,baoq->ai', _cse61, _cse15, _cse40, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qcoj,bapq->ai', _cse61, _cse15, _cse40, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bcgj,gaop->ai', _cse61, _cse15, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bagp,gcoj->ai', _cse61, _cse15, _cse11, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bcpg,agoj->ai', _cse61, _cse15, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bago,gcpj->ai', _cse61, _cse15, _cse11, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,bcog,agpj->ai', _cse61, _cse15, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,acgj,gbop->ai', _cse61, _cse15, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,acpg,bgoj->ai', _cse61, _cse15, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,acog,bgpj->ai', _cse61, _cse15, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogj,gpko,acip->ai', _cse62, _cse63, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,apko,gcip->ai', _cse62, _cse63, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,gpio,ackp->ai', _cse62, _cse63, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,apio,gckp->ai', _cse62, _cse63, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,pgik,acpo->ai', _cse62, _cse63, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,paik,gcpo->ai', _cse62, _cse63, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,pcko,gaip->ai', _cse62, _cse63, _cse40, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,pcio,gakp->ai', _cse62, _cse63, _cse42, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,gcho,haik->ai', _cse62, _cse63, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,gahk,hcio->ai', _cse62, _cse63, _cse11, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,gckh,ahio->ai', _cse62, _cse63, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,gahi,hcko->ai', _cse62, _cse63, _cse13, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,gcih,ahko->ai', _cse62, _cse63, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,acho,hgik->ai', _cse62, _cse63, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,ackh,ghio->ai', _cse62, _cse63, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bogj,acih,ghko->ai', _cse62, _cse63, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gpko,bcip->ai', _cse59, _cse64, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,bpko,gcip->ai', _cse59, _cse64, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gpio,bckp->ai', _cse59, _cse64, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,bpio,gckp->ai', _cse59, _cse64, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,pgik,bcpo->ai', _cse59, _cse64, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,pbik,gcpo->ai', _cse59, _cse64, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,pcko,gbip->ai', _cse59, _cse64, _cse40, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,pcio,gbkp->ai', _cse59, _cse64, _cse42, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gcho,hbik->ai', _cse59, _cse64, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gbhk,hcio->ai', _cse59, _cse64, _cse10, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gckh,bhio->ai', _cse59, _cse64, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gbhi,hcko->ai', _cse59, _cse64, _cse12, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gcih,bhko->ai', _cse59, _cse64, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,bcho,hgik->ai', _cse59, _cse64, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,bckh,ghio->ai', _cse59, _cse64, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,bcih,ghko->ai', _cse59, _cse64, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gpoj,acip->ai', _cse65, _cse18, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,apoj,gcip->ai', _cse65, _cse18, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gpij,acop->ai', _cse65, _cse18, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,apij,gcop->ai', _cse65, _cse18, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pgio,acpj->ai', _cse65, _cse18, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,paio,gcpj->ai', _cse65, _cse18, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcoj,gaip->ai', _cse65, _cse18, _cse40, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcij,gaop->ai', _cse65, _cse18, _cse42, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gchj,haio->ai', _cse65, _cse18, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gaho,hcij->ai', _cse65, _cse18, _cse11, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gcoh,ahij->ai', _cse65, _cse18, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gahi,hcoj->ai', _cse65, _cse18, _cse13, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gcih,ahoj->ai', _cse65, _cse18, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,achj,hgio->ai', _cse65, _cse18, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,acoh,ghij->ai', _cse65, _cse18, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,acih,ghoj->ai', _cse65, _cse18, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gpoj,bcip->ai', _cse61, _cse25, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,bpoj,gcip->ai', _cse61, _cse25, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gpij,bcop->ai', _cse61, _cse25, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,bpij,gcop->ai', _cse61, _cse25, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pgio,bcpj->ai', _cse61, _cse25, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pbio,gcpj->ai', _cse61, _cse25, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pcoj,gbip->ai', _cse61, _cse25, _cse40, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,pcij,gbop->ai', _cse61, _cse25, _cse42, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gchj,hbio->ai', _cse61, _cse25, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gbho,hcij->ai', _cse61, _cse25, _cse10, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gcoh,bhij->ai', _cse61, _cse25, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gbhi,hcoj->ai', _cse61, _cse25, _cse12, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,gcih,bhoj->ai', _cse61, _cse25, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,bchj,hgio->ai', _cse61, _cse25, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,bcoh,ghij->ai', _cse61, _cse25, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagk,bcih,ghoj->ai', _cse61, _cse25, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gpoj,ackp->ai', _cse66, _cse23, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,apoj,gckp->ai', _cse66, _cse23, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gpkj,acop->ai', _cse66, _cse23, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,apkj,gcop->ai', _cse66, _cse23, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pgko,acpj->ai', _cse66, _cse23, _cse2, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pako,gcpj->ai', _cse66, _cse23, _cse6, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pcoj,gakp->ai', _cse66, _cse23, _cse40, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,pckj,gaop->ai', _cse66, _cse23, _cse40, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gchj,hako->ai', _cse66, _cse23, _cse49, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gaho,hckj->ai', _cse66, _cse23, _cse11, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gcoh,ahkj->ai', _cse66, _cse23, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gahk,hcoj->ai', _cse66, _cse23, _cse11, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,gckh,ahoj->ai', _cse66, _cse23, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,achj,hgko->ai', _cse66, _cse23, _cse44, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,acoh,ghkj->ai', _cse66, _cse23, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgi,ackh,ghoj->ai', _cse66, _cse23, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gpoj,bckp->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,bpoj,gckp->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gpkj,bcop->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,bpkj,gcop->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pgko,bcpj->ai', _cse67, _cse27, _cse2, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pbko,gcpj->ai', _cse67, _cse27, _cse2, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pcoj,gbkp->ai', _cse67, _cse27, _cse40, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pckj,gbop->ai', _cse67, _cse27, _cse40, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gchj,hbko->ai', _cse67, _cse27, _cse49, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gbho,hckj->ai', _cse67, _cse27, _cse10, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gcoh,bhkj->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gbhk,hcoj->ai', _cse67, _cse27, _cse10, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gckh,bhoj->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,bchj,hgko->ai', _cse67, _cse27, _cse49, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,bcoh,ghkj->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,bckh,ghoj->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,apko,bgip->ai', _cse55, _cse68, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,bpko,agip->ai', _cse55, _cse68, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,apio,bgkp->ai', _cse55, _cse68, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,bpio,agkp->ai', _cse55, _cse68, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,paik,bgpo->ai', _cse55, _cse68, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbik,agpo->ai', _cse55, _cse68, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgko,abip->ai', _cse55, _cse68, _cse40, _cse41, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgio,abkp->ai', _cse55, _cse68, _cse42, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,agho,hbik->ai', _cse55, _cse68, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,abhk,hgio->ai', _cse55, _cse68, _cse45, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,agkh,bhio->ai', _cse55, _cse68, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,abhi,hgko->ai', _cse55, _cse68, _cse47, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,agih,bhko->ai', _cse55, _cse68, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,bgho,haik->ai', _cse55, _cse68, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,bgkh,ahio->ai', _cse55, _cse68, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,bgih,ahko->ai', _cse55, _cse68, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,apoj,bgip->ai', _cse69, _cse70, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,bpoj,agip->ai', _cse69, _cse70, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,apij,bgop->ai', _cse69, _cse70, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,bpij,agop->ai', _cse69, _cse70, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,paio,bgpj->ai', _cse69, _cse70, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,pbio,agpj->ai', _cse69, _cse70, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,pgoj,abip->ai', _cse69, _cse70, _cse40, _cse41, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,pgij,abop->ai', _cse69, _cse70, _cse42, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,aghj,hbio->ai', _cse69, _cse70, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,abho,hgij->ai', _cse69, _cse70, _cse45, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,agoh,bhij->ai', _cse69, _cse70, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,abhi,hgoj->ai', _cse69, _cse70, _cse47, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,agih,bhoj->ai', _cse69, _cse70, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,bghj,haio->ai', _cse69, _cse70, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,bgoh,ahij->ai', _cse69, _cse70, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ockg,bgih,ahoj->ai', _cse69, _cse70, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,apoj,bgkp->ai', _cse71, _cse72, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,bpoj,agkp->ai', _cse71, _cse72, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,apkj,bgop->ai', _cse71, _cse72, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,bpkj,agop->ai', _cse71, _cse72, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pako,bgpj->ai', _cse71, _cse72, _cse6, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pbko,agpj->ai', _cse71, _cse72, _cse2, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pgoj,abkp->ai', _cse71, _cse72, _cse40, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pgkj,abop->ai', _cse71, _cse72, _cse40, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,aghj,hbko->ai', _cse71, _cse72, _cse44, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,abho,hgkj->ai', _cse71, _cse72, _cse45, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,agoh,bhkj->ai', _cse71, _cse72, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,abhk,hgoj->ai', _cse71, _cse72, _cse45, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,agkh,bhoj->ai', _cse71, _cse72, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,bghj,hako->ai', _cse71, _cse72, _cse49, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,bgoh,ahkj->ai', _cse71, _cse72, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,bgkh,ahoj->ai', _cse71, _cse72, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bagh,gokj,hcio->ai', _cse66, _cse30, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,hokj,gcio->ai', _cse66, _cse30, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,goij,hcko->ai', _cse66, _cse30, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,hoij,gcko->ai', _cse66, _cse30, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ogik,hcoj->ai', _cse66, _cse30, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ohik,gcoj->ai', _cse66, _cse30, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ockj,ghio->ai', _cse66, _cse30, _cse40, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ocij,ghko->ai', _cse66, _cse30, _cse42, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,gcuj,uhik->ai', _cse66, _cse30, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ghuk,ucij->ai', _cse66, _cse30, _cse10, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,gcku,huij->ai', _cse66, _cse30, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,ghui,uckj->ai', _cse66, _cse30, _cse12, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,gciu,hukj->ai', _cse66, _cse30, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,hcuj,ugik->ai', _cse66, _cse30, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,hcku,guij->ai', _cse66, _cse30, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bagh,hciu,gukj->ai', _cse66, _cse30, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,gokj,ahio->ai', _cse73, _cse74, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,aokj,ghio->ai', _cse73, _cse74, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,goij,ahko->ai', _cse73, _cse74, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,aoij,ghko->ai', _cse73, _cse74, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ogik,ahoj->ai', _cse73, _cse74, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,oaik,ghoj->ai', _cse73, _cse74, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ohkj,gaio->ai', _cse73, _cse74, _cse40, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ohij,gako->ai', _cse73, _cse74, _cse42, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ghuj,uaik->ai', _cse73, _cse74, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,gauk,uhij->ai', _cse73, _cse74, _cse11, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ghku,auij->ai', _cse73, _cse74, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,gaui,uhkj->ai', _cse73, _cse74, _cse13, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ghiu,aukj->ai', _cse73, _cse74, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ahuj,ugik->ai', _cse73, _cse74, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ahku,guij->ai', _cse73, _cse74, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bcgh,ahiu,gukj->ai', _cse73, _cse74, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,aokj,hgio->ai', _cse73, _cse74, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,hokj,agio->ai', _cse73, _cse74, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,aoij,hgko->ai', _cse73, _cse74, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,hoij,agko->ai', _cse73, _cse74, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,oaik,hgoj->ai', _cse73, _cse74, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,ohik,agoj->ai', _cse73, _cse74, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,ogkj,ahio->ai', _cse73, _cse74, _cse40, _cse41, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,ogij,ahko->ai', _cse73, _cse74, _cse42, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,aguj,uhik->ai', _cse73, _cse74, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,ahuk,ugij->ai', _cse73, _cse74, _cse45, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,agku,huij->ai', _cse73, _cse74, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,ahui,ugkj->ai', _cse73, _cse74, _cse47, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,agiu,hukj->ai', _cse73, _cse74, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,hguj,uaik->ai', _cse73, _cse74, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,hgku,auij->ai', _cse73, _cse74, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bchg,hgiu,aukj->ai', _cse73, _cse74, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,gokj,bhio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,bokj,ghio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,goij,bhko->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,boij,ghko->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ogik,bhoj->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,obik,ghoj->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ohkj,gbio->ai', _cse71, _cse75, _cse40, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ohij,gbko->ai', _cse71, _cse75, _cse42, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ghuj,ubik->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,gbuk,uhij->ai', _cse71, _cse75, _cse10, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ghku,buij->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,gbui,uhkj->ai', _cse71, _cse75, _cse12, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ghiu,bukj->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,bhuj,ugik->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,bhku,guij->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,bhiu,gukj->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,bokj,hgio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hokj,bgio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,boij,hgko->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hoij,bgko->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,obik,hgoj->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,ohik,bgoj->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,ogkj,bhio->ai', _cse71, _cse75, _cse40, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,ogij,bhko->ai', _cse71, _cse75, _cse42, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,bguj,uhik->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,bhuk,ugij->ai', _cse71, _cse75, _cse10, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,bgku,huij->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,bhui,ugkj->ai', _cse71, _cse75, _cse12, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,bgiu,hukj->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hguj,ubik->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hgku,buij->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hgiu,bukj->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bokg,pgjo,acip->ai', _cse65, _cse31, _cse76, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,apio,gcjp->ai', _cse65, _cse31, _cse36, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,pgio,acpj->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,apij,gcop->ai', _cse65, _cse31, _cse36, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,pgij,acpo->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,pcjo,agip->ai', _cse65, _cse31, _cse76, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,pcio,agpj->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,pcij,agpo->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,agho,hcij->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,acho,hgij->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,aghj,hcio->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,achj,hgio->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,agih,hcjo->ai', _cse65, _cse31, _cse48, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,acih,hgjo->ai', _cse65, _cse31, _cse48, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,gcho,ahij->ai', _cse65, _cse31, _cse78, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,bokg,gchj,ahio->ai', _cse65, _cse31, _cse78, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pgjo,bcip->ai', _cse61, _cse53, _cse76, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bpio,gcjp->ai', _cse61, _cse53, _cse38, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pgio,bcpj->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bpij,gcop->ai', _cse61, _cse53, _cse38, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pgij,bcpo->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pcjo,bgip->ai', _cse61, _cse53, _cse76, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pcio,bgpj->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,pcij,bgpo->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bgho,hcij->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bcho,hgij->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bghj,hcio->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bchj,hgio->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bgih,hcjo->ai', _cse61, _cse53, _cse51, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,bcih,hgjo->ai', _cse61, _cse53, _cse51, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,gcho,bhij->ai', _cse61, _cse53, _cse78, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aokg,gchj,bhio->ai', _cse61, _cse53, _cse78, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pgjo,ackp->ai', _cse66, _cse52, _cse76, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,apko,gcjp->ai', _cse66, _cse52, _cse32, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pgko,acpj->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,apkj,gcop->ai', _cse66, _cse52, _cse32, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pgkj,acpo->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pcjo,agkp->ai', _cse66, _cse52, _cse76, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pcko,agpj->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,pckj,agpo->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,agho,hckj->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,acho,hgkj->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,aghj,hcko->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,achj,hgko->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,agkh,hcjo->ai', _cse66, _cse52, _cse46, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,ackh,hgjo->ai', _cse66, _cse52, _cse46, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,boig,gcho,ahkj->ai', _cse66, _cse52, _cse78, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,boig,gchj,ahko->ai', _cse66, _cse52, _cse78, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pgjo,bckp->ai', _cse67, _cse54, _cse76, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bpko,gcjp->ai', _cse67, _cse54, _cse34, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pgko,bcpj->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bpkj,gcop->ai', _cse67, _cse54, _cse34, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pgkj,bcpo->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pcjo,bgkp->ai', _cse67, _cse54, _cse76, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pcko,bgpj->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pckj,bgpo->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bgho,hckj->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bcho,hgkj->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bghj,hcko->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bchj,hgko->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bgkh,hcjo->ai', _cse67, _cse54, _cse50, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,bckh,hgjo->ai', _cse67, _cse54, _cse50, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,gcho,bhkj->ai', _cse67, _cse54, _cse78, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,gchj,bhko->ai', _cse67, _cse54, _cse78, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pgjo,baip->ai', _cse55, _cse56, _cse2, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pbjo,gaip->ai', _cse55, _cse56, _cse2, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pgio,bajp->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pbio,gajp->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pgij,baop->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pbij,gaop->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pajo,gbip->ai', _cse55, _cse56, _cse6, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,paio,gbjp->ai', _cse55, _cse56, _cse8, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,paij,gbop->ai', _cse55, _cse56, _cse8, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,gbho,haij->ai', _cse55, _cse56, _cse10, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,gaho,hbij->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,gbhj,haio->ai', _cse55, _cse56, _cse10, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,gahj,hbio->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,gbhi,hajo->ai', _cse55, _cse56, _cse12, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,gahi,hbjo->ai', _cse55, _cse56, _cse13, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,baho,hgij->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,bahj,hgio->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,bahi,hgjo->ai', _cse55, _cse56, _cse13, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,pojk,bqpo,aciq->ai', _cse57, _cse58, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,aqpo,bciq->ai', _cse57, _cse58, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,bqio,acpq->ai', _cse57, _cse58, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,aqio,bcpq->ai', _cse57, _cse58, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,qbip,acqo->ai', _cse57, _cse58, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,qaip,bcqo->ai', _cse57, _cse58, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,qcpo,baiq->ai', _cse57, _cse58, _cse40, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,qcio,bapq->ai', _cse57, _cse58, _cse42, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,bcgo,gaip->ai', _cse57, _cse58, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,bagp,gcio->ai', _cse57, _cse58, _cse11, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,bcpg,agio->ai', _cse57, _cse58, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,bagi,gcpo->ai', _cse57, _cse58, _cse13, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,bcig,agpo->ai', _cse57, _cse58, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,acgo,gbip->ai', _cse57, _cse58, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,acpg,bgio->ai', _cse57, _cse58, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojk,acig,bgpo->ai', _cse57, _cse58, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poik,bqpo,acjq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poik,aqpo,bcjq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poik,bqjo,acpq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poik,aqjo,bcpq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poik,qbjp,acqo->ai', _cse59, _cse60, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poik,qajp,bcqo->ai', _cse59, _cse60, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poik,qcpo,bajq->ai', _cse59, _cse60, _cse40, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poik,qcjo,bapq->ai', _cse59, _cse60, _cse40, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poik,bcgo,gajp->ai', _cse59, _cse60, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poik,bagp,gcjo->ai', _cse59, _cse60, _cse11, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poik,bcpg,agjo->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poik,bagj,gcpo->ai', _cse59, _cse60, _cse11, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poik,bcjg,agpo->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poik,acgo,gbjp->ai', _cse59, _cse60, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poik,acpg,bgjo->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poik,acjg,bgpo->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,bqop,aciq->ai', _cse57, _cse58, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,aqop,bciq->ai', _cse57, _cse58, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,bqip,acoq->ai', _cse57, _cse58, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,aqip,bcoq->ai', _cse57, _cse58, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,qbio,acqp->ai', _cse57, _cse58, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,qaio,bcqp->ai', _cse57, _cse58, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,qcop,baiq->ai', _cse57, _cse58, _cse40, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,qcip,baoq->ai', _cse57, _cse58, _cse42, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,bcgp,gaio->ai', _cse57, _cse58, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,bago,gcip->ai', _cse57, _cse58, _cse11, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,bcog,agip->ai', _cse57, _cse58, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,bagi,gcop->ai', _cse57, _cse58, _cse13, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,bcig,agop->ai', _cse57, _cse58, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,acgp,gbio->ai', _cse57, _cse58, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,acog,bgip->ai', _cse57, _cse58, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjk,acig,bgop->ai', _cse57, _cse58, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opik,bqop,acjq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opik,aqop,bcjq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opik,bqjp,acoq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opik,aqjp,bcoq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opik,qbjo,acqp->ai', _cse59, _cse60, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opik,qajo,bcqp->ai', _cse59, _cse60, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opik,qcop,bajq->ai', _cse59, _cse60, _cse40, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opik,qcjp,baoq->ai', _cse59, _cse60, _cse40, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opik,bcgp,gajo->ai', _cse59, _cse60, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opik,bago,gcjp->ai', _cse59, _cse60, _cse11, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opik,bcog,agjp->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opik,bagj,gcop->ai', _cse59, _cse60, _cse11, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opik,bcjg,agop->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opik,acgp,gbjo->ai', _cse59, _cse60, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opik,acog,bgjp->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opik,acjg,bgop->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poij,bqpk,acoq->ai', _cse61, _cse15, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poij,aqpk,bcoq->ai', _cse61, _cse15, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poij,bqok,acpq->ai', _cse61, _cse15, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poij,aqok,bcpq->ai', _cse61, _cse15, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poij,qbop,acqk->ai', _cse61, _cse15, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poij,qaop,bcqk->ai', _cse61, _cse15, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poij,qcpk,baoq->ai', _cse61, _cse15, _cse40, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poij,qcok,bapq->ai', _cse61, _cse15, _cse40, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poij,bcgk,gaop->ai', _cse61, _cse15, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poij,bagp,gcok->ai', _cse61, _cse15, _cse11, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poij,bcpg,agok->ai', _cse61, _cse15, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poij,bago,gcpk->ai', _cse61, _cse15, _cse11, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poij,bcog,agpk->ai', _cse61, _cse15, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poij,acgk,gbop->ai', _cse61, _cse15, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poij,acpg,bgok->ai', _cse61, _cse15, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poij,acog,bgpk->ai', _cse61, _cse15, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogk,gpjo,acip->ai', _cse62, _cse63, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,apjo,gcip->ai', _cse62, _cse63, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,gpio,acjp->ai', _cse62, _cse63, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,apio,gcjp->ai', _cse62, _cse63, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,pgij,acpo->ai', _cse62, _cse63, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,paij,gcpo->ai', _cse62, _cse63, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,pcjo,gaip->ai', _cse62, _cse63, _cse40, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,pcio,gajp->ai', _cse62, _cse63, _cse42, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,gcho,haij->ai', _cse62, _cse63, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,gahj,hcio->ai', _cse62, _cse63, _cse11, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,gcjh,ahio->ai', _cse62, _cse63, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,gahi,hcjo->ai', _cse62, _cse63, _cse13, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,gcih,ahjo->ai', _cse62, _cse63, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,acho,hgij->ai', _cse62, _cse63, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,acjh,ghio->ai', _cse62, _cse63, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bogk,acih,ghjo->ai', _cse62, _cse63, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,gpjo,bcip->ai', _cse59, _cse64, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,bpjo,gcip->ai', _cse59, _cse64, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,gpio,bcjp->ai', _cse59, _cse64, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,bpio,gcjp->ai', _cse59, _cse64, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,pgij,bcpo->ai', _cse59, _cse64, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,pbij,gcpo->ai', _cse59, _cse64, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,pcjo,gbip->ai', _cse59, _cse64, _cse40, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,pcio,gbjp->ai', _cse59, _cse64, _cse42, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,gcho,hbij->ai', _cse59, _cse64, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,gbhj,hcio->ai', _cse59, _cse64, _cse10, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,gcjh,bhio->ai', _cse59, _cse64, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,gbhi,hcjo->ai', _cse59, _cse64, _cse12, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,gcih,bhjo->ai', _cse59, _cse64, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,bcho,hgij->ai', _cse59, _cse64, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,bcjh,ghio->ai', _cse59, _cse64, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aogk,bcih,ghjo->ai', _cse59, _cse64, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,gpok,acip->ai', _cse65, _cse18, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,apok,gcip->ai', _cse65, _cse18, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,gpik,acop->ai', _cse65, _cse18, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,apik,gcop->ai', _cse65, _cse18, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,pgio,acpk->ai', _cse65, _cse18, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,paio,gcpk->ai', _cse65, _cse18, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,pcok,gaip->ai', _cse65, _cse18, _cse40, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,pcik,gaop->ai', _cse65, _cse18, _cse42, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,gchk,haio->ai', _cse65, _cse18, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,gaho,hcik->ai', _cse65, _cse18, _cse11, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,gcoh,ahik->ai', _cse65, _cse18, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,gahi,hcok->ai', _cse65, _cse18, _cse13, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,gcih,ahok->ai', _cse65, _cse18, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,achk,hgio->ai', _cse65, _cse18, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,acoh,ghik->ai', _cse65, _cse18, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgj,acih,ghok->ai', _cse65, _cse18, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,gpok,bcip->ai', _cse61, _cse25, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,bpok,gcip->ai', _cse61, _cse25, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,gpik,bcop->ai', _cse61, _cse25, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,bpik,gcop->ai', _cse61, _cse25, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,pgio,bcpk->ai', _cse61, _cse25, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,pbio,gcpk->ai', _cse61, _cse25, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,pcok,gbip->ai', _cse61, _cse25, _cse40, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,pcik,gbop->ai', _cse61, _cse25, _cse42, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,gchk,hbio->ai', _cse61, _cse25, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,gbho,hcik->ai', _cse61, _cse25, _cse10, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,gcoh,bhik->ai', _cse61, _cse25, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,gbhi,hcok->ai', _cse61, _cse25, _cse12, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,gcih,bhok->ai', _cse61, _cse25, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,bchk,hgio->ai', _cse61, _cse25, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,bcoh,ghik->ai', _cse61, _cse25, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagj,bcih,ghok->ai', _cse61, _cse25, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,gpok,acjp->ai', _cse66, _cse23, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,apok,gcjp->ai', _cse66, _cse23, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,gpjk,acop->ai', _cse66, _cse23, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,apjk,gcop->ai', _cse66, _cse23, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,pgjo,acpk->ai', _cse66, _cse23, _cse2, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,pajo,gcpk->ai', _cse66, _cse23, _cse6, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,pcok,gajp->ai', _cse66, _cse23, _cse40, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,pcjk,gaop->ai', _cse66, _cse23, _cse40, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,gchk,hajo->ai', _cse66, _cse23, _cse49, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,gaho,hcjk->ai', _cse66, _cse23, _cse11, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,gcoh,ahjk->ai', _cse66, _cse23, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,gahj,hcok->ai', _cse66, _cse23, _cse11, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,gcjh,ahok->ai', _cse66, _cse23, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,achk,hgjo->ai', _cse66, _cse23, _cse44, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,acoh,ghjk->ai', _cse66, _cse23, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,obgi,acjh,ghok->ai', _cse66, _cse23, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,gpok,bcjp->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,bpok,gcjp->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,gpjk,bcop->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,bpjk,gcop->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,pgjo,bcpk->ai', _cse67, _cse27, _cse2, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,pbjo,gcpk->ai', _cse67, _cse27, _cse2, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,pcok,gbjp->ai', _cse67, _cse27, _cse40, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,pcjk,gbop->ai', _cse67, _cse27, _cse40, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,gchk,hbjo->ai', _cse67, _cse27, _cse49, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,gbho,hcjk->ai', _cse67, _cse27, _cse10, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,gcoh,bhjk->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,gbhj,hcok->ai', _cse67, _cse27, _cse10, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,gcjh,bhok->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,bchk,hgjo->ai', _cse67, _cse27, _cse49, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,bcoh,ghjk->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,oagi,bcjh,ghok->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,apjo,bgip->ai', _cse55, _cse68, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,bpjo,agip->ai', _cse55, _cse68, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,apio,bgjp->ai', _cse55, _cse68, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,bpio,agjp->ai', _cse55, _cse68, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,paij,bgpo->ai', _cse55, _cse68, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pbij,agpo->ai', _cse55, _cse68, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pgjo,abip->ai', _cse55, _cse68, _cse40, _cse41, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,pgio,abjp->ai', _cse55, _cse68, _cse42, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,agho,hbij->ai', _cse55, _cse68, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,abhj,hgio->ai', _cse55, _cse68, _cse45, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,agjh,bhio->ai', _cse55, _cse68, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,abhi,hgjo->ai', _cse55, _cse68, _cse47, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,agih,bhjo->ai', _cse55, _cse68, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,bgho,haij->ai', _cse55, _cse68, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,bgjh,ahio->ai', _cse55, _cse68, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocgk,bgih,ahjo->ai', _cse55, _cse68, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,apok,bgip->ai', _cse69, _cse70, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,bpok,agip->ai', _cse69, _cse70, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,apik,bgop->ai', _cse69, _cse70, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,bpik,agop->ai', _cse69, _cse70, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,paio,bgpk->ai', _cse69, _cse70, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,pbio,agpk->ai', _cse69, _cse70, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,pgok,abip->ai', _cse69, _cse70, _cse40, _cse41, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,pgik,abop->ai', _cse69, _cse70, _cse42, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,aghk,hbio->ai', _cse69, _cse70, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,abho,hgik->ai', _cse69, _cse70, _cse45, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,agoh,bhik->ai', _cse69, _cse70, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,abhi,hgok->ai', _cse69, _cse70, _cse47, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,agih,bhok->ai', _cse69, _cse70, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,bghk,haio->ai', _cse69, _cse70, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,bgoh,ahik->ai', _cse69, _cse70, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocjg,bgih,ahok->ai', _cse69, _cse70, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,apok,bgjp->ai', _cse71, _cse72, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,bpok,agjp->ai', _cse71, _cse72, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,apjk,bgop->ai', _cse71, _cse72, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,bpjk,agop->ai', _cse71, _cse72, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,pajo,bgpk->ai', _cse71, _cse72, _cse6, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,pbjo,agpk->ai', _cse71, _cse72, _cse2, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,pgok,abjp->ai', _cse71, _cse72, _cse40, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,pgjk,abop->ai', _cse71, _cse72, _cse40, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,aghk,hbjo->ai', _cse71, _cse72, _cse44, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,abho,hgjk->ai', _cse71, _cse72, _cse45, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,agoh,bhjk->ai', _cse71, _cse72, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,abhj,hgok->ai', _cse71, _cse72, _cse45, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,agjh,bhok->ai', _cse71, _cse72, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,bghk,hajo->ai', _cse71, _cse72, _cse49, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,bgoh,ahjk->ai', _cse71, _cse72, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,ocig,bgjh,ahok->ai', _cse71, _cse72, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bagh,gojk,hcio->ai', _cse66, _cse30, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,hojk,gcio->ai', _cse66, _cse30, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,goik,hcjo->ai', _cse66, _cse30, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,hoik,gcjo->ai', _cse66, _cse30, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,ogij,hcok->ai', _cse66, _cse30, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,ohij,gcok->ai', _cse66, _cse30, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,ocjk,ghio->ai', _cse66, _cse30, _cse40, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,ocik,ghjo->ai', _cse66, _cse30, _cse42, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,gcuk,uhij->ai', _cse66, _cse30, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,ghuj,ucik->ai', _cse66, _cse30, _cse10, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,gcju,huik->ai', _cse66, _cse30, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,ghui,ucjk->ai', _cse66, _cse30, _cse12, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,gciu,hujk->ai', _cse66, _cse30, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,hcuk,ugij->ai', _cse66, _cse30, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,hcju,guik->ai', _cse66, _cse30, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bagh,hciu,gujk->ai', _cse66, _cse30, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,gojk,ahio->ai', _cse73, _cse74, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,aojk,ghio->ai', _cse73, _cse74, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,goik,ahjo->ai', _cse73, _cse74, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,aoik,ghjo->ai', _cse73, _cse74, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,ogij,ahok->ai', _cse73, _cse74, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,oaij,ghok->ai', _cse73, _cse74, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,ohjk,gaio->ai', _cse73, _cse74, _cse40, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,ohik,gajo->ai', _cse73, _cse74, _cse42, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,ghuk,uaij->ai', _cse73, _cse74, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,gauj,uhik->ai', _cse73, _cse74, _cse11, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,ghju,auik->ai', _cse73, _cse74, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,gaui,uhjk->ai', _cse73, _cse74, _cse13, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,ghiu,aujk->ai', _cse73, _cse74, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,ahuk,ugij->ai', _cse73, _cse74, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,ahju,guik->ai', _cse73, _cse74, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bcgh,ahiu,gujk->ai', _cse73, _cse74, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,aojk,hgio->ai', _cse73, _cse74, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,hojk,agio->ai', _cse73, _cse74, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,aoik,hgjo->ai', _cse73, _cse74, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,hoik,agjo->ai', _cse73, _cse74, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,oaij,hgok->ai', _cse73, _cse74, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,ohij,agok->ai', _cse73, _cse74, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,ogjk,ahio->ai', _cse73, _cse74, _cse40, _cse41, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,ogik,ahjo->ai', _cse73, _cse74, _cse42, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,aguk,uhij->ai', _cse73, _cse74, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,ahuj,ugik->ai', _cse73, _cse74, _cse45, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,agju,huik->ai', _cse73, _cse74, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,ahui,ugjk->ai', _cse73, _cse74, _cse47, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,agiu,hujk->ai', _cse73, _cse74, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,hguk,uaij->ai', _cse73, _cse74, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,hgju,auik->ai', _cse73, _cse74, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bchg,hgiu,aujk->ai', _cse73, _cse74, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,gojk,bhio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,bojk,ghio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,goik,bhjo->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,boik,ghjo->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,ogij,bhok->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,obij,ghok->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,ohjk,gbio->ai', _cse71, _cse75, _cse40, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,ohik,gbjo->ai', _cse71, _cse75, _cse42, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,ghuk,ubij->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,gbuj,uhik->ai', _cse71, _cse75, _cse10, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,ghju,buik->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,gbui,uhjk->ai', _cse71, _cse75, _cse12, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,ghiu,bujk->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,bhuk,ugij->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,bhju,guik->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,acgh,bhiu,gujk->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,achg,bojk,hgio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,achg,hojk,bgio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,achg,boik,hgjo->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,achg,hoik,bgjo->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,achg,obij,hgok->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,achg,ohij,bgok->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,achg,ogjk,bhio->ai', _cse71, _cse75, _cse40, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,achg,ogik,bhjo->ai', _cse71, _cse75, _cse42, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,achg,bguk,uhij->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,achg,bhuj,ugik->ai', _cse71, _cse75, _cse10, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,achg,bgju,huik->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,achg,bhui,ugjk->ai', _cse71, _cse75, _cse12, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,achg,bgiu,hujk->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,achg,hguk,ubij->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,achg,hgju,buik->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,achg,hgiu,bujk->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bojg,pgko,acip->ai', _cse65, _cse31, _cse76, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,apio,gckp->ai', _cse65, _cse31, _cse36, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,pgio,acpk->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,apik,gcop->ai', _cse65, _cse31, _cse36, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,pgik,acpo->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,pcko,agip->ai', _cse65, _cse31, _cse76, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,pcio,agpk->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,pcik,agpo->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,agho,hcik->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,acho,hgik->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,aghk,hcio->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,achk,hgio->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,agih,hcko->ai', _cse65, _cse31, _cse48, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,acih,hgko->ai', _cse65, _cse31, _cse48, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,gcho,ahik->ai', _cse65, _cse31, _cse78, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,bojg,gchk,ahio->ai', _cse65, _cse31, _cse78, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,pgko,bcip->ai', _cse61, _cse53, _cse76, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,bpio,gckp->ai', _cse61, _cse53, _cse38, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,pgio,bcpk->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,bpik,gcop->ai', _cse61, _cse53, _cse38, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,pgik,bcpo->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,pcko,bgip->ai', _cse61, _cse53, _cse76, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,pcio,bgpk->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,pcik,bgpo->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,bgho,hcik->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,bcho,hgik->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,bghk,hcio->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,bchk,hgio->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,bgih,hcko->ai', _cse61, _cse53, _cse51, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,bcih,hgko->ai', _cse61, _cse53, _cse51, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,gcho,bhik->ai', _cse61, _cse53, _cse78, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aojg,gchk,bhio->ai', _cse61, _cse53, _cse78, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,boig,pgko,acjp->ai', _cse66, _cse52, _cse76, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,boig,apjo,gckp->ai', _cse66, _cse52, _cse32, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,boig,pgjo,acpk->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,boig,apjk,gcop->ai', _cse66, _cse52, _cse32, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,boig,pgjk,acpo->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,boig,pcko,agjp->ai', _cse66, _cse52, _cse76, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,boig,pcjo,agpk->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,boig,pcjk,agpo->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,boig,agho,hcjk->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,boig,acho,hgjk->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,boig,aghk,hcjo->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,boig,achk,hgjo->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,boig,agjh,hcko->ai', _cse66, _cse52, _cse46, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,boig,acjh,hgko->ai', _cse66, _cse52, _cse46, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,boig,gcho,ahjk->ai', _cse66, _cse52, _cse78, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,boig,gchk,ahjo->ai', _cse66, _cse52, _cse78, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,pgko,bcjp->ai', _cse67, _cse54, _cse76, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,bpjo,gckp->ai', _cse67, _cse54, _cse34, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,pgjo,bcpk->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,bpjk,gcop->ai', _cse67, _cse54, _cse34, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,pgjk,bcpo->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,pcko,bgjp->ai', _cse67, _cse54, _cse76, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,pcjo,bgpk->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,pcjk,bgpo->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,bgho,hcjk->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,bcho,hgjk->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,bghk,hcjo->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,bchk,hgjo->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,bgjh,hcko->ai', _cse67, _cse54, _cse50, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,bcjh,hgko->ai', _cse67, _cse54, _cse50, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,gcho,bhjk->ai', _cse67, _cse54, _cse78, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkbc,aoig,gchk,bhjo->ai', _cse67, _cse54, _cse78, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pgko,acip->ai', _cse55, _cse56, _cse2, _cse41, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pako,gcip->ai', _cse55, _cse56, _cse6, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pgio,ackp->ai', _cse55, _cse56, _cse4, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,paio,gckp->ai', _cse55, _cse56, _cse8, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pgik,acop->ai', _cse55, _cse56, _cse4, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,paik,gcop->ai', _cse55, _cse56, _cse8, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pcko,gaip->ai', _cse55, _cse56, _cse2, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pcio,gakp->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pcik,gaop->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,gaho,hcik->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,gcho,haik->ai', _cse55, _cse56, _cse10, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,gahk,hcio->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,gchk,haio->ai', _cse55, _cse56, _cse10, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,gahi,hcko->ai', _cse55, _cse56, _cse13, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,gchi,hako->ai', _cse55, _cse56, _cse12, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,acho,hgik->ai', _cse55, _cse56, _cse45, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,achk,hgio->ai', _cse55, _cse56, _cse45, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,achi,hgko->ai', _cse55, _cse56, _cse47, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,pokj,aqpo,cbiq->ai', _cse57, _cse58, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,cqpo,abiq->ai', _cse57, _cse58, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,aqio,cbpq->ai', _cse57, _cse58, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,cqio,abpq->ai', _cse57, _cse58, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,qaip,cbqo->ai', _cse57, _cse58, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,qcip,abqo->ai', _cse57, _cse58, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,qbpo,aciq->ai', _cse57, _cse58, _cse40, _cse41, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,qbio,acpq->ai', _cse57, _cse58, _cse42, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,abgo,gcip->ai', _cse57, _cse58, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,acgp,gbio->ai', _cse57, _cse58, _cse45, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,abpg,cgio->ai', _cse57, _cse58, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,acgi,gbpo->ai', _cse57, _cse58, _cse47, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,abig,cgpo->ai', _cse57, _cse58, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,cbgo,gaip->ai', _cse57, _cse58, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,cbpg,agio->ai', _cse57, _cse58, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokj,cbig,agpo->ai', _cse57, _cse58, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poij,aqpo,cbkq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poij,cqpo,abkq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poij,aqko,cbpq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poij,cqko,abpq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poij,qakp,cbqo->ai', _cse59, _cse60, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poij,qckp,abqo->ai', _cse59, _cse60, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poij,qbpo,ackq->ai', _cse59, _cse60, _cse40, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poij,qbko,acpq->ai', _cse59, _cse60, _cse40, _cse43, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poij,abgo,gckp->ai', _cse59, _cse60, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poij,acgp,gbko->ai', _cse59, _cse60, _cse45, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poij,abpg,cgko->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poij,acgk,gbpo->ai', _cse59, _cse60, _cse45, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poij,abkg,cgpo->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poij,cbgo,gakp->ai', _cse59, _cse60, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poij,cbpg,agko->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poij,cbkg,agpo->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,aqop,cbiq->ai', _cse57, _cse58, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,cqop,abiq->ai', _cse57, _cse58, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,aqip,cboq->ai', _cse57, _cse58, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,cqip,aboq->ai', _cse57, _cse58, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,qaio,cbqp->ai', _cse57, _cse58, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,qcio,abqp->ai', _cse57, _cse58, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,qbop,aciq->ai', _cse57, _cse58, _cse40, _cse41, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,qbip,acoq->ai', _cse57, _cse58, _cse42, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,abgp,gcio->ai', _cse57, _cse58, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,acgo,gbip->ai', _cse57, _cse58, _cse45, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,abog,cgip->ai', _cse57, _cse58, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,acgi,gbop->ai', _cse57, _cse58, _cse47, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,abig,cgop->ai', _cse57, _cse58, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,cbgp,gaio->ai', _cse57, _cse58, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,cbog,agip->ai', _cse57, _cse58, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkj,cbig,agop->ai', _cse57, _cse58, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opij,aqop,cbkq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opij,cqop,abkq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opij,aqkp,cboq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opij,cqkp,aboq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opij,qako,cbqp->ai', _cse59, _cse60, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opij,qcko,abqp->ai', _cse59, _cse60, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opij,qbop,ackq->ai', _cse59, _cse60, _cse40, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opij,qbkp,acoq->ai', _cse59, _cse60, _cse40, _cse43, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opij,abgp,gcko->ai', _cse59, _cse60, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opij,acgo,gbkp->ai', _cse59, _cse60, _cse45, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opij,abog,cgkp->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opij,acgk,gbop->ai', _cse59, _cse60, _cse45, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opij,abkg,cgop->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opij,cbgp,gako->ai', _cse59, _cse60, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opij,cbog,agkp->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opij,cbkg,agop->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poik,aqpj,cboq->ai', _cse61, _cse15, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poik,cqpj,aboq->ai', _cse61, _cse15, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poik,aqoj,cbpq->ai', _cse61, _cse15, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poik,cqoj,abpq->ai', _cse61, _cse15, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poik,qaop,cbqj->ai', _cse61, _cse15, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poik,qcop,abqj->ai', _cse61, _cse15, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poik,qbpj,acoq->ai', _cse61, _cse15, _cse40, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poik,qboj,acpq->ai', _cse61, _cse15, _cse40, _cse43, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poik,abgj,gcop->ai', _cse61, _cse15, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poik,acgp,gboj->ai', _cse61, _cse15, _cse45, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poik,abpg,cgoj->ai', _cse61, _cse15, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poik,acgo,gbpj->ai', _cse61, _cse15, _cse45, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poik,abog,cgpj->ai', _cse61, _cse15, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poik,cbgj,gaop->ai', _cse61, _cse15, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poik,cbpg,agoj->ai', _cse61, _cse15, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poik,cbog,agpj->ai', _cse61, _cse15, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogj,gpko,cbip->ai', _cse59, _cse64, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,cpko,gbip->ai', _cse59, _cse64, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,gpio,cbkp->ai', _cse59, _cse64, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,cpio,gbkp->ai', _cse59, _cse64, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,pgik,cbpo->ai', _cse59, _cse64, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,pcik,gbpo->ai', _cse59, _cse64, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,pbko,gcip->ai', _cse59, _cse64, _cse40, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,pbio,gckp->ai', _cse59, _cse64, _cse42, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,gbho,hcik->ai', _cse59, _cse64, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,gchk,hbio->ai', _cse59, _cse64, _cse10, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,gbkh,chio->ai', _cse59, _cse64, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,gchi,hbko->ai', _cse59, _cse64, _cse12, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,gbih,chko->ai', _cse59, _cse64, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,cbho,hgik->ai', _cse59, _cse64, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,cbkh,ghio->ai', _cse59, _cse64, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aogj,cbih,ghko->ai', _cse59, _cse64, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,gpko,abip->ai', _cse62, _cse63, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,apko,gbip->ai', _cse62, _cse63, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,gpio,abkp->ai', _cse62, _cse63, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,apio,gbkp->ai', _cse62, _cse63, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,pgik,abpo->ai', _cse62, _cse63, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,paik,gbpo->ai', _cse62, _cse63, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,pbko,gaip->ai', _cse62, _cse63, _cse40, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,pbio,gakp->ai', _cse62, _cse63, _cse42, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,gbho,haik->ai', _cse62, _cse63, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,gahk,hbio->ai', _cse62, _cse63, _cse11, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,gbkh,ahio->ai', _cse62, _cse63, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,gahi,hbko->ai', _cse62, _cse63, _cse13, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,gbih,ahko->ai', _cse62, _cse63, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,abho,hgik->ai', _cse62, _cse63, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,abkh,ghio->ai', _cse62, _cse63, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cogj,abih,ghko->ai', _cse62, _cse63, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,gpoj,cbip->ai', _cse61, _cse25, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,cpoj,gbip->ai', _cse61, _cse25, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,gpij,cbop->ai', _cse61, _cse25, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,cpij,gbop->ai', _cse61, _cse25, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,pgio,cbpj->ai', _cse61, _cse25, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,pcio,gbpj->ai', _cse61, _cse25, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,pboj,gcip->ai', _cse61, _cse25, _cse40, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,pbij,gcop->ai', _cse61, _cse25, _cse42, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,gbhj,hcio->ai', _cse61, _cse25, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,gcho,hbij->ai', _cse61, _cse25, _cse10, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,gboh,chij->ai', _cse61, _cse25, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,gchi,hboj->ai', _cse61, _cse25, _cse12, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,gbih,choj->ai', _cse61, _cse25, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,cbhj,hgio->ai', _cse61, _cse25, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,cboh,ghij->ai', _cse61, _cse25, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagk,cbih,ghoj->ai', _cse61, _cse25, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,gpoj,abip->ai', _cse65, _cse18, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,apoj,gbip->ai', _cse65, _cse18, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,gpij,abop->ai', _cse65, _cse18, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,apij,gbop->ai', _cse65, _cse18, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,pgio,abpj->ai', _cse65, _cse18, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,paio,gbpj->ai', _cse65, _cse18, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,pboj,gaip->ai', _cse65, _cse18, _cse40, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,pbij,gaop->ai', _cse65, _cse18, _cse42, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,gbhj,haio->ai', _cse65, _cse18, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,gaho,hbij->ai', _cse65, _cse18, _cse11, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,gboh,ahij->ai', _cse65, _cse18, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,gahi,hboj->ai', _cse65, _cse18, _cse13, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,gbih,ahoj->ai', _cse65, _cse18, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,abhj,hgio->ai', _cse65, _cse18, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,aboh,ghij->ai', _cse65, _cse18, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgk,abih,ghoj->ai', _cse65, _cse18, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,gpoj,cbkp->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,cpoj,gbkp->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,gpkj,cbop->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,cpkj,gbop->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,pgko,cbpj->ai', _cse67, _cse27, _cse2, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,pcko,gbpj->ai', _cse67, _cse27, _cse2, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,pboj,gckp->ai', _cse67, _cse27, _cse40, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,pbkj,gcop->ai', _cse67, _cse27, _cse40, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,gbhj,hcko->ai', _cse67, _cse27, _cse49, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,gcho,hbkj->ai', _cse67, _cse27, _cse10, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,gboh,chkj->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,gchk,hboj->ai', _cse67, _cse27, _cse10, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,gbkh,choj->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,cbhj,hgko->ai', _cse67, _cse27, _cse49, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,cboh,ghkj->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,oagi,cbkh,ghoj->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,gpoj,abkp->ai', _cse66, _cse23, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,apoj,gbkp->ai', _cse66, _cse23, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,gpkj,abop->ai', _cse66, _cse23, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,apkj,gbop->ai', _cse66, _cse23, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,pgko,abpj->ai', _cse66, _cse23, _cse2, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,pako,gbpj->ai', _cse66, _cse23, _cse6, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,pboj,gakp->ai', _cse66, _cse23, _cse40, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,pbkj,gaop->ai', _cse66, _cse23, _cse40, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,gbhj,hako->ai', _cse66, _cse23, _cse49, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,gaho,hbkj->ai', _cse66, _cse23, _cse11, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,gboh,ahkj->ai', _cse66, _cse23, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,gahk,hboj->ai', _cse66, _cse23, _cse11, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,gbkh,ahoj->ai', _cse66, _cse23, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,abhj,hgko->ai', _cse66, _cse23, _cse44, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,aboh,ghkj->ai', _cse66, _cse23, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,ocgi,abkh,ghoj->ai', _cse66, _cse23, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,cpko,agip->ai', _cse55, _cse68, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,apko,cgip->ai', _cse55, _cse68, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,cpio,agkp->ai', _cse55, _cse68, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,apio,cgkp->ai', _cse55, _cse68, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pcik,agpo->ai', _cse55, _cse68, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,paik,cgpo->ai', _cse55, _cse68, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pgko,caip->ai', _cse55, _cse68, _cse40, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,pgio,cakp->ai', _cse55, _cse68, _cse42, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,cgho,haik->ai', _cse55, _cse68, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,cahk,hgio->ai', _cse55, _cse68, _cse11, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,cgkh,ahio->ai', _cse55, _cse68, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,cahi,hgko->ai', _cse55, _cse68, _cse13, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,cgih,ahko->ai', _cse55, _cse68, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,agho,hcik->ai', _cse55, _cse68, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,agkh,chio->ai', _cse55, _cse68, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obgj,agih,chko->ai', _cse55, _cse68, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,cpoj,agip->ai', _cse69, _cse70, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,apoj,cgip->ai', _cse69, _cse70, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,cpij,agop->ai', _cse69, _cse70, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,apij,cgop->ai', _cse69, _cse70, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,pcio,agpj->ai', _cse69, _cse70, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,paio,cgpj->ai', _cse69, _cse70, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,pgoj,caip->ai', _cse69, _cse70, _cse40, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,pgij,caop->ai', _cse69, _cse70, _cse42, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,cghj,haio->ai', _cse69, _cse70, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,caho,hgij->ai', _cse69, _cse70, _cse11, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,cgoh,ahij->ai', _cse69, _cse70, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,cahi,hgoj->ai', _cse69, _cse70, _cse13, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,cgih,ahoj->ai', _cse69, _cse70, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,aghj,hcio->ai', _cse69, _cse70, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,agoh,chij->ai', _cse69, _cse70, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obkg,agih,choj->ai', _cse69, _cse70, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obig,cpoj,agkp->ai', _cse71, _cse72, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obig,apoj,cgkp->ai', _cse71, _cse72, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obig,cpkj,agop->ai', _cse71, _cse72, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obig,apkj,cgop->ai', _cse71, _cse72, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obig,pcko,agpj->ai', _cse71, _cse72, _cse2, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obig,pako,cgpj->ai', _cse71, _cse72, _cse6, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obig,pgoj,cakp->ai', _cse71, _cse72, _cse40, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obig,pgkj,caop->ai', _cse71, _cse72, _cse40, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obig,cghj,hako->ai', _cse71, _cse72, _cse49, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obig,caho,hgkj->ai', _cse71, _cse72, _cse11, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obig,cgoh,ahkj->ai', _cse71, _cse72, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obig,cahk,hgoj->ai', _cse71, _cse72, _cse11, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obig,cgkh,ahoj->ai', _cse71, _cse72, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obig,aghj,hcko->ai', _cse71, _cse72, _cse44, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,obig,agoh,chkj->ai', _cse71, _cse72, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,obig,agkh,choj->ai', _cse71, _cse72, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,acgh,gokj,hbio->ai', _cse66, _cse79, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,hokj,gbio->ai', _cse66, _cse79, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,goij,hbko->ai', _cse66, _cse79, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,hoij,gbko->ai', _cse66, _cse79, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,ogik,hboj->ai', _cse66, _cse79, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,ohik,gboj->ai', _cse66, _cse79, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,obkj,ghio->ai', _cse66, _cse79, _cse40, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,obij,ghko->ai', _cse66, _cse79, _cse42, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,gbuj,uhik->ai', _cse66, _cse79, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,ghuk,ubij->ai', _cse66, _cse79, _cse10, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,gbku,huij->ai', _cse66, _cse79, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,ghui,ubkj->ai', _cse66, _cse79, _cse12, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,gbiu,hukj->ai', _cse66, _cse79, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,hbuj,ugik->ai', _cse66, _cse79, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,hbku,guij->ai', _cse66, _cse79, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,acgh,hbiu,gukj->ai', _cse66, _cse79, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,gokj,chio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,cokj,ghio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,goij,chko->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,coij,ghko->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,ogik,choj->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,ocik,ghoj->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,ohkj,gcio->ai', _cse71, _cse75, _cse40, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,ohij,gcko->ai', _cse71, _cse75, _cse42, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,ghuj,ucik->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,gcuk,uhij->ai', _cse71, _cse75, _cse10, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,ghku,cuij->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,gcui,uhkj->ai', _cse71, _cse75, _cse12, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,ghiu,cukj->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,chuj,ugik->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,chku,guij->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abgh,chiu,gukj->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,cokj,hgio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,hokj,cgio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,coij,hgko->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,hoij,cgko->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,ocik,hgoj->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,ohik,cgoj->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,ogkj,chio->ai', _cse71, _cse75, _cse40, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,ogij,chko->ai', _cse71, _cse75, _cse42, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,cguj,uhik->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,chuk,ugij->ai', _cse71, _cse75, _cse10, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,cgku,huij->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,chui,ugkj->ai', _cse71, _cse75, _cse12, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,cgiu,hukj->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,hguj,ucik->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,hgku,cuij->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,abhg,hgiu,cukj->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,gokj,ahio->ai', _cse73, _cse74, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,aokj,ghio->ai', _cse73, _cse74, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,goij,ahko->ai', _cse73, _cse74, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,aoij,ghko->ai', _cse73, _cse74, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,ogik,ahoj->ai', _cse73, _cse74, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,oaik,ghoj->ai', _cse73, _cse74, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,ohkj,gaio->ai', _cse73, _cse74, _cse40, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,ohij,gako->ai', _cse73, _cse74, _cse42, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,ghuj,uaik->ai', _cse73, _cse74, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,gauk,uhij->ai', _cse73, _cse74, _cse11, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,ghku,auij->ai', _cse73, _cse74, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,gaui,uhkj->ai', _cse73, _cse74, _cse13, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,ghiu,aukj->ai', _cse73, _cse74, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,ahuj,ugik->ai', _cse73, _cse74, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,ahku,guij->ai', _cse73, _cse74, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbgh,ahiu,gukj->ai', _cse73, _cse74, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,aokj,hgio->ai', _cse73, _cse74, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,hokj,agio->ai', _cse73, _cse74, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,aoij,hgko->ai', _cse73, _cse74, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,hoij,agko->ai', _cse73, _cse74, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,oaik,hgoj->ai', _cse73, _cse74, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,ohik,agoj->ai', _cse73, _cse74, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,ogkj,ahio->ai', _cse73, _cse74, _cse40, _cse41, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,ogij,ahko->ai', _cse73, _cse74, _cse42, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,aguj,uhik->ai', _cse73, _cse74, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,ahuk,ugij->ai', _cse73, _cse74, _cse45, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,agku,huij->ai', _cse73, _cse74, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,ahui,ugkj->ai', _cse73, _cse74, _cse47, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,agiu,hukj->ai', _cse73, _cse74, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,hguj,uaik->ai', _cse73, _cse74, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,hgku,auij->ai', _cse73, _cse74, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cbhg,hgiu,aukj->ai', _cse73, _cse74, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aokg,pgjo,cbip->ai', _cse61, _cse53, _cse76, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,cpio,gbjp->ai', _cse61, _cse53, _cse38, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,pgio,cbpj->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,cpij,gbop->ai', _cse61, _cse53, _cse38, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,pgij,cbpo->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,pbjo,cgip->ai', _cse61, _cse53, _cse76, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,pbio,cgpj->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,pbij,cgpo->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,cgho,hbij->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,cbho,hgij->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,cghj,hbio->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,cbhj,hgio->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,cgih,hbjo->ai', _cse61, _cse53, _cse51, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,cbih,hgjo->ai', _cse61, _cse53, _cse51, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,gbho,chij->ai', _cse61, _cse53, _cse78, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aokg,gbhj,chio->ai', _cse61, _cse53, _cse78, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,pgjo,abip->ai', _cse65, _cse31, _cse76, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,apio,gbjp->ai', _cse65, _cse31, _cse36, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,pgio,abpj->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,apij,gbop->ai', _cse65, _cse31, _cse36, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,pgij,abpo->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,pbjo,agip->ai', _cse65, _cse31, _cse76, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,pbio,agpj->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,pbij,agpo->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,agho,hbij->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,abho,hgij->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,aghj,hbio->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,abhj,hgio->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,agih,hbjo->ai', _cse65, _cse31, _cse48, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,abih,hgjo->ai', _cse65, _cse31, _cse48, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,gbho,ahij->ai', _cse65, _cse31, _cse78, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,cokg,gbhj,ahio->ai', _cse65, _cse31, _cse78, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,pgjo,cbkp->ai', _cse67, _cse54, _cse76, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,cpko,gbjp->ai', _cse67, _cse54, _cse34, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,pgko,cbpj->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,cpkj,gbop->ai', _cse67, _cse54, _cse34, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,pgkj,cbpo->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,pbjo,cgkp->ai', _cse67, _cse54, _cse76, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,pbko,cgpj->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,pbkj,cgpo->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,cgho,hbkj->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,cbho,hgkj->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,cghj,hbko->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,cbhj,hgko->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,cgkh,hbjo->ai', _cse67, _cse54, _cse50, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,cbkh,hgjo->ai', _cse67, _cse54, _cse50, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,gbho,chkj->ai', _cse67, _cse54, _cse78, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,aoig,gbhj,chko->ai', _cse67, _cse54, _cse78, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,coig,pgjo,abkp->ai', _cse66, _cse52, _cse76, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,coig,apko,gbjp->ai', _cse66, _cse52, _cse32, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,coig,pgko,abpj->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,coig,apkj,gbop->ai', _cse66, _cse52, _cse32, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,coig,pgkj,abpo->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,coig,pbjo,agkp->ai', _cse66, _cse52, _cse76, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,coig,pbko,agpj->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,coig,pbkj,agpo->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,coig,agho,hbkj->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,coig,abho,hgkj->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,coig,aghj,hbko->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,coig,abhj,hgko->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,coig,agkh,hbjo->ai', _cse66, _cse52, _cse46, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,coig,abkh,hgjo->ai', _cse66, _cse52, _cse46, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjcb,coig,gbho,ahkj->ai', _cse66, _cse52, _cse78, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjcb,coig,gbhj,ahko->ai', _cse66, _cse52, _cse78, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pgjo,acip->ai', _cse55, _cse56, _cse2, _cse41, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pajo,gcip->ai', _cse55, _cse56, _cse6, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pgio,acjp->ai', _cse55, _cse56, _cse4, _cse43, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,paio,gcjp->ai', _cse55, _cse56, _cse8, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pgij,acop->ai', _cse55, _cse56, _cse4, _cse43, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,paij,gcop->ai', _cse55, _cse56, _cse8, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pcjo,gaip->ai', _cse55, _cse56, _cse2, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pcio,gajp->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pcij,gaop->ai', _cse55, _cse56, _cse4, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,gaho,hcij->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,gcho,haij->ai', _cse55, _cse56, _cse10, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,gahj,hcio->ai', _cse55, _cse56, _cse11, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,gchj,haio->ai', _cse55, _cse56, _cse10, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,gahi,hcjo->ai', _cse55, _cse56, _cse13, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,gchi,hajo->ai', _cse55, _cse56, _cse12, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,acho,hgij->ai', _cse55, _cse56, _cse45, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,achj,hgio->ai', _cse55, _cse56, _cse45, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,achi,hgjo->ai', _cse55, _cse56, _cse47, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,pojk,aqpo,cbiq->ai', _cse57, _cse58, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,cqpo,abiq->ai', _cse57, _cse58, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,aqio,cbpq->ai', _cse57, _cse58, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,cqio,abpq->ai', _cse57, _cse58, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,qaip,cbqo->ai', _cse57, _cse58, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,qcip,abqo->ai', _cse57, _cse58, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,qbpo,aciq->ai', _cse57, _cse58, _cse40, _cse41, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,qbio,acpq->ai', _cse57, _cse58, _cse42, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,abgo,gcip->ai', _cse57, _cse58, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,acgp,gbio->ai', _cse57, _cse58, _cse45, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,abpg,cgio->ai', _cse57, _cse58, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,acgi,gbpo->ai', _cse57, _cse58, _cse47, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,abig,cgpo->ai', _cse57, _cse58, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,cbgo,gaip->ai', _cse57, _cse58, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,cbpg,agio->ai', _cse57, _cse58, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojk,cbig,agpo->ai', _cse57, _cse58, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poik,aqpo,cbjq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poik,cqpo,abjq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poik,aqjo,cbpq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poik,cqjo,abpq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poik,qajp,cbqo->ai', _cse59, _cse60, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poik,qcjp,abqo->ai', _cse59, _cse60, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poik,qbpo,acjq->ai', _cse59, _cse60, _cse40, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poik,qbjo,acpq->ai', _cse59, _cse60, _cse40, _cse43, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poik,abgo,gcjp->ai', _cse59, _cse60, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poik,acgp,gbjo->ai', _cse59, _cse60, _cse45, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poik,abpg,cgjo->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poik,acgj,gbpo->ai', _cse59, _cse60, _cse45, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poik,abjg,cgpo->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poik,cbgo,gajp->ai', _cse59, _cse60, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poik,cbpg,agjo->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poik,cbjg,agpo->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,aqop,cbiq->ai', _cse57, _cse58, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,cqop,abiq->ai', _cse57, _cse58, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,aqip,cboq->ai', _cse57, _cse58, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,cqip,aboq->ai', _cse57, _cse58, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,qaio,cbqp->ai', _cse57, _cse58, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,qcio,abqp->ai', _cse57, _cse58, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,qbop,aciq->ai', _cse57, _cse58, _cse40, _cse41, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,qbip,acoq->ai', _cse57, _cse58, _cse42, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,abgp,gcio->ai', _cse57, _cse58, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,acgo,gbip->ai', _cse57, _cse58, _cse45, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,abog,cgip->ai', _cse57, _cse58, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,acgi,gbop->ai', _cse57, _cse58, _cse47, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,abig,cgop->ai', _cse57, _cse58, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,cbgp,gaio->ai', _cse57, _cse58, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,cbog,agip->ai', _cse57, _cse58, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjk,cbig,agop->ai', _cse57, _cse58, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opik,aqop,cbjq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opik,cqop,abjq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opik,aqjp,cboq->ai', _cse59, _cse60, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opik,cqjp,aboq->ai', _cse59, _cse60, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opik,qajo,cbqp->ai', _cse59, _cse60, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opik,qcjo,abqp->ai', _cse59, _cse60, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opik,qbop,acjq->ai', _cse59, _cse60, _cse40, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opik,qbjp,acoq->ai', _cse59, _cse60, _cse40, _cse43, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opik,abgp,gcjo->ai', _cse59, _cse60, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opik,acgo,gbjp->ai', _cse59, _cse60, _cse45, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opik,abog,cgjp->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opik,acgj,gbop->ai', _cse59, _cse60, _cse45, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opik,abjg,cgop->ai', _cse59, _cse60, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opik,cbgp,gajo->ai', _cse59, _cse60, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opik,cbog,agjp->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opik,cbjg,agop->ai', _cse59, _cse60, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poij,aqpk,cboq->ai', _cse61, _cse15, _cse32, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poij,cqpk,aboq->ai', _cse61, _cse15, _cse34, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poij,aqok,cbpq->ai', _cse61, _cse15, _cse32, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poij,cqok,abpq->ai', _cse61, _cse15, _cse34, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poij,qaop,cbqk->ai', _cse61, _cse15, _cse6, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poij,qcop,abqk->ai', _cse61, _cse15, _cse2, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poij,qbpk,acoq->ai', _cse61, _cse15, _cse40, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poij,qbok,acpq->ai', _cse61, _cse15, _cse40, _cse43, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poij,abgk,gcop->ai', _cse61, _cse15, _cse44, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poij,acgp,gbok->ai', _cse61, _cse15, _cse45, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poij,abpg,cgok->ai', _cse61, _cse15, _cse46, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poij,acgo,gbpk->ai', _cse61, _cse15, _cse45, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poij,abog,cgpk->ai', _cse61, _cse15, _cse46, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poij,cbgk,gaop->ai', _cse61, _cse15, _cse49, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poij,cbpg,agok->ai', _cse61, _cse15, _cse50, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poij,cbog,agpk->ai', _cse61, _cse15, _cse50, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogk,gpjo,cbip->ai', _cse59, _cse64, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,cpjo,gbip->ai', _cse59, _cse64, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,gpio,cbjp->ai', _cse59, _cse64, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,cpio,gbjp->ai', _cse59, _cse64, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,pgij,cbpo->ai', _cse59, _cse64, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,pcij,gbpo->ai', _cse59, _cse64, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,pbjo,gcip->ai', _cse59, _cse64, _cse40, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,pbio,gcjp->ai', _cse59, _cse64, _cse42, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,gbho,hcij->ai', _cse59, _cse64, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,gchj,hbio->ai', _cse59, _cse64, _cse10, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,gbjh,chio->ai', _cse59, _cse64, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,gchi,hbjo->ai', _cse59, _cse64, _cse12, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,gbih,chjo->ai', _cse59, _cse64, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,cbho,hgij->ai', _cse59, _cse64, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,cbjh,ghio->ai', _cse59, _cse64, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aogk,cbih,ghjo->ai', _cse59, _cse64, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,gpjo,abip->ai', _cse62, _cse63, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,apjo,gbip->ai', _cse62, _cse63, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,gpio,abjp->ai', _cse62, _cse63, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,apio,gbjp->ai', _cse62, _cse63, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,pgij,abpo->ai', _cse62, _cse63, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,paij,gbpo->ai', _cse62, _cse63, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,pbjo,gaip->ai', _cse62, _cse63, _cse40, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,pbio,gajp->ai', _cse62, _cse63, _cse42, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,gbho,haij->ai', _cse62, _cse63, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,gahj,hbio->ai', _cse62, _cse63, _cse11, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,gbjh,ahio->ai', _cse62, _cse63, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,gahi,hbjo->ai', _cse62, _cse63, _cse13, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,gbih,ahjo->ai', _cse62, _cse63, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,abho,hgij->ai', _cse62, _cse63, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,abjh,ghio->ai', _cse62, _cse63, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cogk,abih,ghjo->ai', _cse62, _cse63, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,gpok,cbip->ai', _cse61, _cse25, _cse34, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,cpok,gbip->ai', _cse61, _cse25, _cse34, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,gpik,cbop->ai', _cse61, _cse25, _cse38, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,cpik,gbop->ai', _cse61, _cse25, _cse38, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,pgio,cbpk->ai', _cse61, _cse25, _cse4, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,pcio,gbpk->ai', _cse61, _cse25, _cse4, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,pbok,gcip->ai', _cse61, _cse25, _cse40, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,pbik,gcop->ai', _cse61, _cse25, _cse42, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,gbhk,hcio->ai', _cse61, _cse25, _cse49, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,gcho,hbik->ai', _cse61, _cse25, _cse10, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,gboh,chik->ai', _cse61, _cse25, _cse50, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,gchi,hbok->ai', _cse61, _cse25, _cse12, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,gbih,chok->ai', _cse61, _cse25, _cse51, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,cbhk,hgio->ai', _cse61, _cse25, _cse49, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,cboh,ghik->ai', _cse61, _cse25, _cse50, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagj,cbih,ghok->ai', _cse61, _cse25, _cse51, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,gpok,abip->ai', _cse65, _cse18, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,apok,gbip->ai', _cse65, _cse18, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,gpik,abop->ai', _cse65, _cse18, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,apik,gbop->ai', _cse65, _cse18, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,pgio,abpk->ai', _cse65, _cse18, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,paio,gbpk->ai', _cse65, _cse18, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,pbok,gaip->ai', _cse65, _cse18, _cse40, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,pbik,gaop->ai', _cse65, _cse18, _cse42, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,gbhk,haio->ai', _cse65, _cse18, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,gaho,hbik->ai', _cse65, _cse18, _cse11, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,gboh,ahik->ai', _cse65, _cse18, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,gahi,hbok->ai', _cse65, _cse18, _cse13, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,gbih,ahok->ai', _cse65, _cse18, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,abhk,hgio->ai', _cse65, _cse18, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,aboh,ghik->ai', _cse65, _cse18, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgj,abih,ghok->ai', _cse65, _cse18, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,gpok,cbjp->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,cpok,gbjp->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,gpjk,cbop->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,cpjk,gbop->ai', _cse67, _cse27, _cse34, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,pgjo,cbpk->ai', _cse67, _cse27, _cse2, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,pcjo,gbpk->ai', _cse67, _cse27, _cse2, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,pbok,gcjp->ai', _cse67, _cse27, _cse40, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,pbjk,gcop->ai', _cse67, _cse27, _cse40, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,gbhk,hcjo->ai', _cse67, _cse27, _cse49, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,gcho,hbjk->ai', _cse67, _cse27, _cse10, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,gboh,chjk->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,gchj,hbok->ai', _cse67, _cse27, _cse10, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,gbjh,chok->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,cbhk,hgjo->ai', _cse67, _cse27, _cse49, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,cboh,ghjk->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,oagi,cbjh,ghok->ai', _cse67, _cse27, _cse50, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,gpok,abjp->ai', _cse66, _cse23, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,apok,gbjp->ai', _cse66, _cse23, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,gpjk,abop->ai', _cse66, _cse23, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,apjk,gbop->ai', _cse66, _cse23, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,pgjo,abpk->ai', _cse66, _cse23, _cse2, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,pajo,gbpk->ai', _cse66, _cse23, _cse6, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,pbok,gajp->ai', _cse66, _cse23, _cse40, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,pbjk,gaop->ai', _cse66, _cse23, _cse40, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,gbhk,hajo->ai', _cse66, _cse23, _cse49, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,gaho,hbjk->ai', _cse66, _cse23, _cse11, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,gboh,ahjk->ai', _cse66, _cse23, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,gahj,hbok->ai', _cse66, _cse23, _cse11, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,gbjh,ahok->ai', _cse66, _cse23, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,abhk,hgjo->ai', _cse66, _cse23, _cse44, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,aboh,ghjk->ai', _cse66, _cse23, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,ocgi,abjh,ghok->ai', _cse66, _cse23, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,cpjo,agip->ai', _cse55, _cse68, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,apjo,cgip->ai', _cse55, _cse68, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,cpio,agjp->ai', _cse55, _cse68, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,apio,cgjp->ai', _cse55, _cse68, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pcij,agpo->ai', _cse55, _cse68, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,paij,cgpo->ai', _cse55, _cse68, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pgjo,caip->ai', _cse55, _cse68, _cse40, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,pgio,cajp->ai', _cse55, _cse68, _cse42, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,cgho,haij->ai', _cse55, _cse68, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,cahj,hgio->ai', _cse55, _cse68, _cse11, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,cgjh,ahio->ai', _cse55, _cse68, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,cahi,hgjo->ai', _cse55, _cse68, _cse13, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,cgih,ahjo->ai', _cse55, _cse68, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,agho,hcij->ai', _cse55, _cse68, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,agjh,chio->ai', _cse55, _cse68, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obgk,agih,chjo->ai', _cse55, _cse68, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,objg,cpok,agip->ai', _cse69, _cse70, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,objg,apok,cgip->ai', _cse69, _cse70, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,objg,cpik,agop->ai', _cse69, _cse70, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,objg,apik,cgop->ai', _cse69, _cse70, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,objg,pcio,agpk->ai', _cse69, _cse70, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,objg,paio,cgpk->ai', _cse69, _cse70, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,objg,pgok,caip->ai', _cse69, _cse70, _cse40, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,objg,pgik,caop->ai', _cse69, _cse70, _cse42, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,objg,cghk,haio->ai', _cse69, _cse70, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,objg,caho,hgik->ai', _cse69, _cse70, _cse11, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,objg,cgoh,ahik->ai', _cse69, _cse70, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,objg,cahi,hgok->ai', _cse69, _cse70, _cse13, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,objg,cgih,ahok->ai', _cse69, _cse70, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,objg,aghk,hcio->ai', _cse69, _cse70, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,objg,agoh,chik->ai', _cse69, _cse70, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,objg,agih,chok->ai', _cse69, _cse70, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obig,cpok,agjp->ai', _cse71, _cse72, _cse34, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obig,apok,cgjp->ai', _cse71, _cse72, _cse32, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obig,cpjk,agop->ai', _cse71, _cse72, _cse34, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obig,apjk,cgop->ai', _cse71, _cse72, _cse32, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obig,pcjo,agpk->ai', _cse71, _cse72, _cse2, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obig,pajo,cgpk->ai', _cse71, _cse72, _cse6, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obig,pgok,cajp->ai', _cse71, _cse72, _cse40, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obig,pgjk,caop->ai', _cse71, _cse72, _cse40, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obig,cghk,hajo->ai', _cse71, _cse72, _cse49, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obig,caho,hgjk->ai', _cse71, _cse72, _cse11, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obig,cgoh,ahjk->ai', _cse71, _cse72, _cse50, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obig,cahj,hgok->ai', _cse71, _cse72, _cse11, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obig,cgjh,ahok->ai', _cse71, _cse72, _cse50, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obig,aghk,hcjo->ai', _cse71, _cse72, _cse44, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,obig,agoh,chjk->ai', _cse71, _cse72, _cse46, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,obig,agjh,chok->ai', _cse71, _cse72, _cse46, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,acgh,gojk,hbio->ai', _cse66, _cse79, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,hojk,gbio->ai', _cse66, _cse79, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,goik,hbjo->ai', _cse66, _cse79, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,hoik,gbjo->ai', _cse66, _cse79, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,ogij,hbok->ai', _cse66, _cse79, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,ohij,gbok->ai', _cse66, _cse79, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,objk,ghio->ai', _cse66, _cse79, _cse40, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,obik,ghjo->ai', _cse66, _cse79, _cse42, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,gbuk,uhij->ai', _cse66, _cse79, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,ghuj,ubik->ai', _cse66, _cse79, _cse10, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,gbju,huik->ai', _cse66, _cse79, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,ghui,ubjk->ai', _cse66, _cse79, _cse12, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,gbiu,hujk->ai', _cse66, _cse79, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,hbuk,ugij->ai', _cse66, _cse79, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,hbju,guik->ai', _cse66, _cse79, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,acgh,hbiu,gujk->ai', _cse66, _cse79, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,gojk,chio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,cojk,ghio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,goik,chjo->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,coik,ghjo->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,ogij,chok->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,ocij,ghok->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,ohjk,gcio->ai', _cse71, _cse75, _cse40, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,ohik,gcjo->ai', _cse71, _cse75, _cse42, _cse9, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,ghuk,ucij->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,gcuj,uhik->ai', _cse71, _cse75, _cse10, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,ghju,cuik->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,gcui,uhjk->ai', _cse71, _cse75, _cse12, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,ghiu,cujk->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,chuk,ugij->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,chju,guik->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abgh,chiu,gujk->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,cojk,hgio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,hojk,cgio->ai', _cse71, _cse75, _cse34, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,coik,hgjo->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,hoik,cgjo->ai', _cse71, _cse75, _cse38, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,ocij,hgok->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,ohij,cgok->ai', _cse71, _cse75, _cse4, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,ogjk,chio->ai', _cse71, _cse75, _cse40, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,ogik,chjo->ai', _cse71, _cse75, _cse42, _cse9, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,cguk,uhij->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,chuj,ugik->ai', _cse71, _cse75, _cse10, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,cgju,huik->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,chui,ugjk->ai', _cse71, _cse75, _cse12, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,cgiu,hujk->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,hguk,ucij->ai', _cse71, _cse75, _cse49, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,hgju,cuik->ai', _cse71, _cse75, _cse50, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,abhg,hgiu,cujk->ai', _cse71, _cse75, _cse51, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,gojk,ahio->ai', _cse73, _cse74, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,aojk,ghio->ai', _cse73, _cse74, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,goik,ahjo->ai', _cse73, _cse74, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,aoik,ghjo->ai', _cse73, _cse74, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,ogij,ahok->ai', _cse73, _cse74, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,oaij,ghok->ai', _cse73, _cse74, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,ohjk,gaio->ai', _cse73, _cse74, _cse40, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,ohik,gajo->ai', _cse73, _cse74, _cse42, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,ghuk,uaij->ai', _cse73, _cse74, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,gauj,uhik->ai', _cse73, _cse74, _cse11, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,ghju,auik->ai', _cse73, _cse74, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,gaui,uhjk->ai', _cse73, _cse74, _cse13, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,ghiu,aujk->ai', _cse73, _cse74, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,ahuk,ugij->ai', _cse73, _cse74, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,ahju,guik->ai', _cse73, _cse74, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbgh,ahiu,gujk->ai', _cse73, _cse74, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,aojk,hgio->ai', _cse73, _cse74, _cse32, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,hojk,agio->ai', _cse73, _cse74, _cse34, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,aoik,hgjo->ai', _cse73, _cse74, _cse36, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,hoik,agjo->ai', _cse73, _cse74, _cse38, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,oaij,hgok->ai', _cse73, _cse74, _cse8, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,ohij,agok->ai', _cse73, _cse74, _cse4, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,ogjk,ahio->ai', _cse73, _cse74, _cse40, _cse41, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,ogik,ahjo->ai', _cse73, _cse74, _cse42, _cse43, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,aguk,uhij->ai', _cse73, _cse74, _cse44, _cse7, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,ahuj,ugik->ai', _cse73, _cse74, _cse45, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,agju,huik->ai', _cse73, _cse74, _cse46, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,ahui,ugjk->ai', _cse73, _cse74, _cse47, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,agiu,hujk->ai', _cse73, _cse74, _cse48, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,hguk,uaij->ai', _cse73, _cse74, _cse49, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,hgju,auik->ai', _cse73, _cse74, _cse50, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cbhg,hgiu,aujk->ai', _cse73, _cse74, _cse51, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aojg,pgko,cbip->ai', _cse61, _cse53, _cse76, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,cpio,gbkp->ai', _cse61, _cse53, _cse38, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,pgio,cbpk->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,cpik,gbop->ai', _cse61, _cse53, _cse38, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,pgik,cbpo->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,pbko,cgip->ai', _cse61, _cse53, _cse76, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,pbio,cgpk->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,pbik,cgpo->ai', _cse61, _cse53, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,cgho,hbik->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,cbho,hgik->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,cghk,hbio->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,cbhk,hgio->ai', _cse61, _cse53, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,cgih,hbko->ai', _cse61, _cse53, _cse51, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,cbih,hgko->ai', _cse61, _cse53, _cse51, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,gbho,chik->ai', _cse61, _cse53, _cse78, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aojg,gbhk,chio->ai', _cse61, _cse53, _cse78, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,pgko,abip->ai', _cse65, _cse31, _cse76, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,apio,gbkp->ai', _cse65, _cse31, _cse36, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,pgio,abpk->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,apik,gbop->ai', _cse65, _cse31, _cse36, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,pgik,abpo->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,pbko,agip->ai', _cse65, _cse31, _cse76, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,pbio,agpk->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,pbik,agpo->ai', _cse65, _cse31, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,agho,hbik->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,abho,hgik->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,aghk,hbio->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,abhk,hgio->ai', _cse65, _cse31, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,agih,hbko->ai', _cse65, _cse31, _cse48, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,abih,hgko->ai', _cse65, _cse31, _cse48, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,gbho,ahik->ai', _cse65, _cse31, _cse78, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,cojg,gbhk,ahio->ai', _cse65, _cse31, _cse78, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,pgko,cbjp->ai', _cse67, _cse54, _cse76, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,cpjo,gbkp->ai', _cse67, _cse54, _cse34, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,pgjo,cbpk->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,cpjk,gbop->ai', _cse67, _cse54, _cse34, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,pgjk,cbpo->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,pbko,cgjp->ai', _cse67, _cse54, _cse76, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,pbjo,cgpk->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,pbjk,cgpo->ai', _cse67, _cse54, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,cgho,hbjk->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,cbho,hgjk->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,cghk,hbjo->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,cbhk,hgjo->ai', _cse67, _cse54, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,cgjh,hbko->ai', _cse67, _cse54, _cse50, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,cbjh,hgko->ai', _cse67, _cse54, _cse50, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,gbho,chjk->ai', _cse67, _cse54, _cse78, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,aoig,gbhk,chjo->ai', _cse67, _cse54, _cse78, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,coig,pgko,abjp->ai', _cse66, _cse52, _cse76, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,coig,apjo,gbkp->ai', _cse66, _cse52, _cse32, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,coig,pgjo,abpk->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,coig,apjk,gbop->ai', _cse66, _cse52, _cse32, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,coig,pgjk,abpo->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,coig,pbko,agjp->ai', _cse66, _cse52, _cse76, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,coig,pbjo,agpk->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,coig,pbjk,agpo->ai', _cse66, _cse52, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,coig,agho,hbjk->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,coig,abho,hgjk->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,coig,aghk,hbjo->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,coig,abhk,hgjo->ai', _cse66, _cse52, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,coig,agjh,hbko->ai', _cse66, _cse52, _cse46, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,coig,abjh,hgko->ai', _cse66, _cse52, _cse46, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('jkcb,coig,gbho,ahjk->ai', _cse66, _cse52, _cse78, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('jkcb,coig,gbhk,ahjo->ai', _cse66, _cse52, _cse78, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gpok,abip->ai', _cse80, _cse56, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,apok,gbip->ai', _cse80, _cse56, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gpik,abop->ai', _cse80, _cse56, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,apik,gbop->ai', _cse80, _cse56, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgio,abpk->ai', _cse80, _cse56, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,paio,gbpk->ai', _cse80, _cse56, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbok,gaip->ai', _cse80, _cse56, _cse40, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbik,gaop->ai', _cse80, _cse56, _cse42, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbhk,haio->ai', _cse80, _cse56, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gaho,hbik->ai', _cse80, _cse56, _cse11, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gboh,ahik->ai', _cse80, _cse56, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gahi,hbok->ai', _cse80, _cse56, _cse13, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbih,ahok->ai', _cse80, _cse56, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,abhk,hgio->ai', _cse80, _cse56, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,aboh,ghik->ai', _cse80, _cse56, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,abih,ghok->ai', _cse80, _cse56, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gpoj,abip->ai', _cse81, _cse56, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,apoj,gbip->ai', _cse81, _cse56, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gpij,abop->ai', _cse81, _cse56, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,apij,gbop->ai', _cse81, _cse56, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pgio,abpj->ai', _cse81, _cse56, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,paio,gbpj->ai', _cse81, _cse56, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pboj,gaip->ai', _cse81, _cse56, _cse40, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pbij,gaop->ai', _cse81, _cse56, _cse42, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gbhj,haio->ai', _cse81, _cse56, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gaho,hbij->ai', _cse81, _cse56, _cse11, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gboh,ahij->ai', _cse81, _cse56, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gahi,hboj->ai', _cse81, _cse56, _cse13, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gbih,ahoj->ai', _cse81, _cse56, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,abhj,hgio->ai', _cse81, _cse56, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,aboh,ghij->ai', _cse81, _cse56, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,abih,ghoj->ai', _cse81, _cse56, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gpok,acip->ai', _cse82, _cse56, _cse34, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,apok,gcip->ai', _cse82, _cse56, _cse32, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gpik,acop->ai', _cse82, _cse56, _cse38, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,apik,gcop->ai', _cse82, _cse56, _cse36, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pgio,acpk->ai', _cse82, _cse56, _cse4, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,paio,gcpk->ai', _cse82, _cse56, _cse8, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pcok,gaip->ai', _cse82, _cse56, _cse40, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pcik,gaop->ai', _cse82, _cse56, _cse42, _cse5, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gchk,haio->ai', _cse82, _cse56, _cse49, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gaho,hcik->ai', _cse82, _cse56, _cse11, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gcoh,ahik->ai', _cse82, _cse56, _cse50, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gahi,hcok->ai', _cse82, _cse56, _cse13, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gcih,ahok->ai', _cse82, _cse56, _cse51, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,achk,hgio->ai', _cse82, _cse56, _cse44, _cse7, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,acoh,ghik->ai', _cse82, _cse56, _cse46, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,acih,ghok->ai', _cse82, _cse56, _cse48, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gpoj,acip->ai', _cse83, _cse56, _cse34, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,apoj,gcip->ai', _cse83, _cse56, _cse32, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gpij,acop->ai', _cse83, _cse56, _cse38, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,apij,gcop->ai', _cse83, _cse56, _cse36, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pgio,acpj->ai', _cse83, _cse56, _cse4, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,paio,gcpj->ai', _cse83, _cse56, _cse8, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcoj,gaip->ai', _cse83, _cse56, _cse40, _cse3, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcij,gaop->ai', _cse83, _cse56, _cse42, _cse5, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gchj,haio->ai', _cse83, _cse56, _cse49, _cse3, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gaho,hcij->ai', _cse83, _cse56, _cse11, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gcoh,ahij->ai', _cse83, _cse56, _cse50, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gahi,hcoj->ai', _cse83, _cse56, _cse13, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gcih,ahoj->ai', _cse83, _cse56, _cse51, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,achj,hgio->ai', _cse83, _cse56, _cse44, _cse7, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,acoh,ghij->ai', _cse83, _cse56, _cse46, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,acih,ghoj->ai', _cse83, _cse56, _cse48, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,pokj,qcpo,abiq->ai', _cse84, _cse85, _cse76, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,aqio,cbpq->ai', _cse84, _cse85, _cse36, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qcio,abqp->ai', _cse84, _cse85, _cse42, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,aqip,cboq->ai', _cse84, _cse85, _cse36, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qcip,abqo->ai', _cse84, _cse85, _cse42, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qbpo,aciq->ai', _cse84, _cse85, _cse76, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qbio,acqp->ai', _cse84, _cse85, _cse42, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,qbip,acqo->ai', _cse84, _cse85, _cse42, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,acgo,gbip->ai', _cse84, _cse85, _cse44, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,abgo,gcip->ai', _cse84, _cse85, _cse44, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,acgp,gbio->ai', _cse84, _cse85, _cse44, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,abgp,gcio->ai', _cse84, _cse85, _cse44, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,acig,gbpo->ai', _cse84, _cse85, _cse48, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,abig,gcpo->ai', _cse84, _cse85, _cse48, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,cbgo,agip->ai', _cse84, _cse85, _cse78, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokj,cbgp,agio->ai', _cse84, _cse85, _cse78, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qcko,abpq->ai', _cse86, _cse60, _cse76, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,aqpo,cbkq->ai', _cse86, _cse60, _cse32, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qcpo,abqk->ai', _cse86, _cse60, _cse40, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,aqpk,cboq->ai', _cse86, _cse60, _cse32, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qcpk,abqo->ai', _cse86, _cse60, _cse40, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qbko,acpq->ai', _cse86, _cse60, _cse76, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qbpo,acqk->ai', _cse86, _cse60, _cse40, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,qbpk,acqo->ai', _cse86, _cse60, _cse40, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,acgo,gbpk->ai', _cse86, _cse60, _cse44, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,abgo,gcpk->ai', _cse86, _cse60, _cse44, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,acgk,gbpo->ai', _cse86, _cse60, _cse44, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,abgk,gcpo->ai', _cse86, _cse60, _cse44, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,acpg,gbko->ai', _cse86, _cse60, _cse46, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,abpg,gcko->ai', _cse86, _cse60, _cse46, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poij,cbgo,agpk->ai', _cse86, _cse60, _cse78, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poij,cbgk,agpo->ai', _cse86, _cse60, _cse78, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qcpk,aboq->ai', _cse86, _cse60, _cse76, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,aqok,cbpq->ai', _cse86, _cse60, _cse32, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qcok,abqp->ai', _cse86, _cse60, _cse40, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,aqop,cbkq->ai', _cse86, _cse60, _cse32, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qcop,abqk->ai', _cse86, _cse60, _cse40, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qbpk,acoq->ai', _cse86, _cse60, _cse76, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qbok,acqp->ai', _cse86, _cse60, _cse40, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,qbop,acqk->ai', _cse86, _cse60, _cse40, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,acgk,gbop->ai', _cse86, _cse60, _cse44, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,abgk,gcop->ai', _cse86, _cse60, _cse44, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,acgp,gbok->ai', _cse86, _cse60, _cse44, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,abgp,gcok->ai', _cse86, _cse60, _cse44, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,acog,gbpk->ai', _cse86, _cse60, _cse46, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,abog,gcpk->ai', _cse86, _cse60, _cse46, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opij,cbgk,agop->ai', _cse86, _cse60, _cse78, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opij,cbgp,agok->ai', _cse86, _cse60, _cse78, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qcjo,abpq->ai', _cse87, _cse60, _cse76, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,aqpo,cbjq->ai', _cse87, _cse60, _cse32, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qcpo,abqj->ai', _cse87, _cse60, _cse40, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,aqpj,cboq->ai', _cse87, _cse60, _cse32, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qcpj,abqo->ai', _cse87, _cse60, _cse40, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qbjo,acpq->ai', _cse87, _cse60, _cse76, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qbpo,acqj->ai', _cse87, _cse60, _cse40, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,qbpj,acqo->ai', _cse87, _cse60, _cse40, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,acgo,gbpj->ai', _cse87, _cse60, _cse44, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,abgo,gcpj->ai', _cse87, _cse60, _cse44, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,acgj,gbpo->ai', _cse87, _cse60, _cse44, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,abgj,gcpo->ai', _cse87, _cse60, _cse44, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,acpg,gbjo->ai', _cse87, _cse60, _cse46, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,abpg,gcjo->ai', _cse87, _cse60, _cse46, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poik,cbgo,agpj->ai', _cse87, _cse60, _cse78, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poik,cbgj,agpo->ai', _cse87, _cse60, _cse78, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opik,qcpj,aboq->ai', _cse87, _cse60, _cse76, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opik,aqoj,cbpq->ai', _cse87, _cse60, _cse32, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opik,qcoj,abqp->ai', _cse87, _cse60, _cse40, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opik,aqop,cbjq->ai', _cse87, _cse60, _cse32, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opik,qcop,abqj->ai', _cse87, _cse60, _cse40, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opik,qbpj,acoq->ai', _cse87, _cse60, _cse76, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opik,qboj,acqp->ai', _cse87, _cse60, _cse40, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opik,qbop,acqj->ai', _cse87, _cse60, _cse40, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opik,acgj,gbop->ai', _cse87, _cse60, _cse44, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opik,abgj,gcop->ai', _cse87, _cse60, _cse44, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opik,acgp,gboj->ai', _cse87, _cse60, _cse44, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opik,abgp,gcoj->ai', _cse87, _cse60, _cse44, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opik,acog,gbpj->ai', _cse87, _cse60, _cse46, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opik,abog,gcpj->ai', _cse87, _cse60, _cse46, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opik,cbgj,agop->ai', _cse87, _cse60, _cse78, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opik,cbgp,agoj->ai', _cse87, _cse60, _cse78, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogj,pcko,gbip->ai', _cse86, _cse64, _cse76, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gpio,cbkp->ai', _cse86, _cse64, _cse38, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,pcio,gbpk->ai', _cse86, _cse64, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gpik,cbop->ai', _cse86, _cse64, _cse38, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,pcik,gbpo->ai', _cse86, _cse64, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,pbko,gcip->ai', _cse86, _cse64, _cse76, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,pbio,gcpk->ai', _cse86, _cse64, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,pbik,gcpo->ai', _cse86, _cse64, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gcho,hbik->ai', _cse86, _cse64, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gbho,hcik->ai', _cse86, _cse64, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gchk,hbio->ai', _cse86, _cse64, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gbhk,hcio->ai', _cse86, _cse64, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gcih,hbko->ai', _cse86, _cse64, _cse51, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,gbih,hcko->ai', _cse86, _cse64, _cse51, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,cbho,ghik->ai', _cse86, _cse64, _cse78, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogj,cbhk,ghio->ai', _cse86, _cse64, _cse78, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,pcjo,gbip->ai', _cse87, _cse64, _cse76, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,gpio,cbjp->ai', _cse87, _cse64, _cse38, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,pcio,gbpj->ai', _cse87, _cse64, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,gpij,cbop->ai', _cse87, _cse64, _cse38, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,pcij,gbpo->ai', _cse87, _cse64, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,pbjo,gcip->ai', _cse87, _cse64, _cse76, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,pbio,gcpj->ai', _cse87, _cse64, _cse42, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,pbij,gcpo->ai', _cse87, _cse64, _cse42, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,gcho,hbij->ai', _cse87, _cse64, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,gbho,hcij->ai', _cse87, _cse64, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,gchj,hbio->ai', _cse87, _cse64, _cse49, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,gbhj,hcio->ai', _cse87, _cse64, _cse49, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,gcih,hbjo->ai', _cse87, _cse64, _cse51, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,gbih,hcjo->ai', _cse87, _cse64, _cse51, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,cbho,ghij->ai', _cse87, _cse64, _cse78, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aogk,cbhj,ghio->ai', _cse87, _cse64, _cse78, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgko,abip->ai', _cse80, _cse68, _cse76, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,apio,gbkp->ai', _cse80, _cse68, _cse36, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgio,abpk->ai', _cse80, _cse68, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,apik,gbop->ai', _cse80, _cse68, _cse36, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pgik,abpo->ai', _cse80, _cse68, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbko,agip->ai', _cse80, _cse68, _cse76, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbio,agpk->ai', _cse80, _cse68, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,pbik,agpo->ai', _cse80, _cse68, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,agho,hbik->ai', _cse80, _cse68, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,abho,hgik->ai', _cse80, _cse68, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,aghk,hbio->ai', _cse80, _cse68, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,abhk,hgio->ai', _cse80, _cse68, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,agih,hbko->ai', _cse80, _cse68, _cse48, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,abih,hgko->ai', _cse80, _cse68, _cse48, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbho,ahik->ai', _cse80, _cse68, _cse78, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgj,gbhk,ahio->ai', _cse80, _cse68, _cse78, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pgjo,abip->ai', _cse81, _cse68, _cse76, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,apio,gbjp->ai', _cse81, _cse68, _cse36, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pgio,abpj->ai', _cse81, _cse68, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,apij,gbop->ai', _cse81, _cse68, _cse36, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pgij,abpo->ai', _cse81, _cse68, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pbjo,agip->ai', _cse81, _cse68, _cse76, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pbio,agpj->ai', _cse81, _cse68, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,pbij,agpo->ai', _cse81, _cse68, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,agho,hbij->ai', _cse81, _cse68, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,abho,hgij->ai', _cse81, _cse68, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,aghj,hbio->ai', _cse81, _cse68, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,abhj,hgio->ai', _cse81, _cse68, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,agih,hbjo->ai', _cse81, _cse68, _cse48, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,abih,hgjo->ai', _cse81, _cse68, _cse48, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gbho,ahij->ai', _cse81, _cse68, _cse78, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocgk,gbhj,ahio->ai', _cse81, _cse68, _cse78, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pcjk,gbop->ai', _cse88, _cse27, _cse76, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gpok,cbjp->ai', _cse88, _cse27, _cse34, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pcok,gbpj->ai', _cse88, _cse27, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gpoj,cbkp->ai', _cse88, _cse27, _cse34, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pcoj,gbpk->ai', _cse88, _cse27, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pbjk,gcop->ai', _cse88, _cse27, _cse76, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pbok,gcpj->ai', _cse88, _cse27, _cse40, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,pboj,gcpk->ai', _cse88, _cse27, _cse40, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gchk,hboj->ai', _cse88, _cse27, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gbhk,hcoj->ai', _cse88, _cse27, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gchj,hbok->ai', _cse88, _cse27, _cse49, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gbhj,hcok->ai', _cse88, _cse27, _cse49, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gcoh,hbjk->ai', _cse88, _cse27, _cse50, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,gboh,hcjk->ai', _cse88, _cse27, _cse50, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,cbhk,ghoj->ai', _cse88, _cse27, _cse78, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,oagi,cbhj,ghok->ai', _cse88, _cse27, _cse78, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pgjk,abop->ai', _cse89, _cse72, _cse76, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,apok,gbjp->ai', _cse89, _cse72, _cse32, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pgok,abpj->ai', _cse89, _cse72, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,apoj,gbkp->ai', _cse89, _cse72, _cse32, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pgoj,abpk->ai', _cse89, _cse72, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pbjk,agop->ai', _cse89, _cse72, _cse76, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pbok,agpj->ai', _cse89, _cse72, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,pboj,agpk->ai', _cse89, _cse72, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,aghk,hboj->ai', _cse89, _cse72, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,abhk,hgoj->ai', _cse89, _cse72, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,aghj,hbok->ai', _cse89, _cse72, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,abhj,hgok->ai', _cse89, _cse72, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,agoh,hbjk->ai', _cse89, _cse72, _cse46, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,aboh,hgjk->ai', _cse89, _cse72, _cse46, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,gbhk,ahoj->ai', _cse89, _cse72, _cse78, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,ocig,gbhj,ahok->ai', _cse89, _cse72, _cse78, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pgko,acip->ai', _cse82, _cse68, _cse76, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,apio,gckp->ai', _cse82, _cse68, _cse36, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pgio,acpk->ai', _cse82, _cse68, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,apik,gcop->ai', _cse82, _cse68, _cse36, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pgik,acpo->ai', _cse82, _cse68, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pcko,agip->ai', _cse82, _cse68, _cse76, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pcio,agpk->ai', _cse82, _cse68, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,pcik,agpo->ai', _cse82, _cse68, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,agho,hcik->ai', _cse82, _cse68, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,acho,hgik->ai', _cse82, _cse68, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,aghk,hcio->ai', _cse82, _cse68, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,achk,hgio->ai', _cse82, _cse68, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,agih,hcko->ai', _cse82, _cse68, _cse48, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,acih,hgko->ai', _cse82, _cse68, _cse48, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gcho,ahik->ai', _cse82, _cse68, _cse78, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgj,gchk,ahio->ai', _cse82, _cse68, _cse78, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pgjo,acip->ai', _cse83, _cse68, _cse76, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,apio,gcjp->ai', _cse83, _cse68, _cse36, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pgio,acpj->ai', _cse83, _cse68, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,apij,gcop->ai', _cse83, _cse68, _cse36, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pgij,acpo->ai', _cse83, _cse68, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcjo,agip->ai', _cse83, _cse68, _cse76, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcio,agpj->ai', _cse83, _cse68, _cse42, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,pcij,agpo->ai', _cse83, _cse68, _cse42, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,agho,hcij->ai', _cse83, _cse68, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,acho,hgij->ai', _cse83, _cse68, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,aghj,hcio->ai', _cse83, _cse68, _cse44, _cse33, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,achj,hgio->ai', _cse83, _cse68, _cse44, _cse33, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,agih,hcjo->ai', _cse83, _cse68, _cse48, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,acih,hgjo->ai', _cse83, _cse68, _cse48, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gcho,ahij->ai', _cse83, _cse68, _cse78, _cse35, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obgk,gchj,ahio->ai', _cse83, _cse68, _cse78, _cse35, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obig,pgjk,acop->ai', _cse90, _cse72, _cse76, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obig,apok,gcjp->ai', _cse90, _cse72, _cse32, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obig,pgok,acpj->ai', _cse90, _cse72, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obig,apoj,gckp->ai', _cse90, _cse72, _cse32, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obig,pgoj,acpk->ai', _cse90, _cse72, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obig,pcjk,agop->ai', _cse90, _cse72, _cse76, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obig,pcok,agpj->ai', _cse90, _cse72, _cse40, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obig,pcoj,agpk->ai', _cse90, _cse72, _cse40, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obig,aghk,hcoj->ai', _cse90, _cse72, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obig,achk,hgoj->ai', _cse90, _cse72, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obig,aghj,hcok->ai', _cse90, _cse72, _cse44, _cse37, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obig,achj,hgok->ai', _cse90, _cse72, _cse44, _cse37, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obig,agoh,hcjk->ai', _cse90, _cse72, _cse46, _cse77, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obig,acoh,hgjk->ai', _cse90, _cse72, _cse46, _cse77, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,obig,gchk,ahoj->ai', _cse90, _cse72, _cse78, _cse39, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,obig,gchj,ahok->ai', _cse90, _cse72, _cse78, _cse39, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,acgh,ohkj,gbio->ai', _cse89, _cse75, _cse76, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,goij,hbko->ai', _cse89, _cse75, _cse38, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ohij,gbok->ai', _cse89, _cse75, _cse42, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,goik,hbjo->ai', _cse89, _cse75, _cse38, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ohik,gboj->ai', _cse89, _cse75, _cse42, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,obkj,ghio->ai', _cse89, _cse75, _cse76, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,obij,ghok->ai', _cse89, _cse75, _cse42, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,obik,ghoj->ai', _cse89, _cse75, _cse42, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ghuj,ubik->ai', _cse89, _cse75, _cse49, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,gbuj,uhik->ai', _cse89, _cse75, _cse49, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ghuk,ubij->ai', _cse89, _cse75, _cse49, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,gbuk,uhij->ai', _cse89, _cse75, _cse49, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,ghiu,ubkj->ai', _cse89, _cse75, _cse51, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,gbiu,uhkj->ai', _cse89, _cse75, _cse51, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,hbuj,guik->ai', _cse89, _cse75, _cse78, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,acgh,hbuk,guij->ai', _cse89, _cse75, _cse78, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,ohkj,gcio->ai', _cse90, _cse75, _cse76, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,goij,hcko->ai', _cse90, _cse75, _cse38, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,ohij,gcok->ai', _cse90, _cse75, _cse42, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,goik,hcjo->ai', _cse90, _cse75, _cse38, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,ohik,gcoj->ai', _cse90, _cse75, _cse42, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,ockj,ghio->ai', _cse90, _cse75, _cse76, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,ocij,ghok->ai', _cse90, _cse75, _cse42, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,ocik,ghoj->ai', _cse90, _cse75, _cse42, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,ghuj,ucik->ai', _cse90, _cse75, _cse49, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,gcuj,uhik->ai', _cse90, _cse75, _cse49, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,ghuk,ucij->ai', _cse90, _cse75, _cse49, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,gcuk,uhij->ai', _cse90, _cse75, _cse49, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,ghiu,uckj->ai', _cse90, _cse75, _cse51, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,gciu,uhkj->ai', _cse90, _cse75, _cse51, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,hcuj,guik->ai', _cse90, _cse75, _cse78, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abgh,hcuk,guij->ai', _cse90, _cse75, _cse78, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,ogkj,hbio->ai', _cse89, _cse75, _cse76, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hoij,gbko->ai', _cse89, _cse75, _cse38, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,ogij,hbok->ai', _cse89, _cse75, _cse42, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hoik,gbjo->ai', _cse89, _cse75, _cse38, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,ogik,hboj->ai', _cse89, _cse75, _cse42, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,obkj,hgio->ai', _cse89, _cse75, _cse76, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,obij,hgok->ai', _cse89, _cse75, _cse42, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,obik,hgoj->ai', _cse89, _cse75, _cse42, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hguj,ubik->ai', _cse89, _cse75, _cse49, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hbuj,ugik->ai', _cse89, _cse75, _cse49, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hguk,ubij->ai', _cse89, _cse75, _cse49, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hbuk,ugij->ai', _cse89, _cse75, _cse49, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hgiu,ubkj->ai', _cse89, _cse75, _cse51, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,hbiu,ugkj->ai', _cse89, _cse75, _cse51, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,achg,gbuj,huik->ai', _cse89, _cse75, _cse78, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,achg,gbuk,huij->ai', _cse89, _cse75, _cse78, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,ogkj,hcio->ai', _cse90, _cse75, _cse76, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,hoij,gcko->ai', _cse90, _cse75, _cse38, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,ogij,hcok->ai', _cse90, _cse75, _cse42, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,hoik,gcjo->ai', _cse90, _cse75, _cse38, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,ogik,hcoj->ai', _cse90, _cse75, _cse42, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,ockj,hgio->ai', _cse90, _cse75, _cse76, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,ocij,hgok->ai', _cse90, _cse75, _cse42, _cse37, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,ocik,hgoj->ai', _cse90, _cse75, _cse42, _cse37, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,hguj,ucik->ai', _cse90, _cse75, _cse49, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,hcuj,ugik->ai', _cse90, _cse75, _cse49, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,hguk,ucij->ai', _cse90, _cse75, _cse49, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,hcuk,ugij->ai', _cse90, _cse75, _cse49, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,hgiu,uckj->ai', _cse90, _cse75, _cse51, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,hciu,ugkj->ai', _cse90, _cse75, _cse51, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,gcuj,huik->ai', _cse90, _cse75, _cse78, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,abhg,gcuk,huij->ai', _cse90, _cse75, _cse78, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,ohkj,agio->ai', _cse91, _cse92, _cse76, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,aoij,hgko->ai', _cse91, _cse92, _cse36, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,ohij,agok->ai', _cse91, _cse92, _cse42, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,aoik,hgjo->ai', _cse91, _cse92, _cse36, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,ohik,agoj->ai', _cse91, _cse92, _cse42, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,ogkj,ahio->ai', _cse91, _cse92, _cse76, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,ogij,ahok->ai', _cse91, _cse92, _cse42, _cse39, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,ogik,ahoj->ai', _cse91, _cse92, _cse42, _cse39, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,ahuj,ugik->ai', _cse91, _cse92, _cse44, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,aguj,uhik->ai', _cse91, _cse92, _cse44, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,ahuk,ugij->ai', _cse91, _cse92, _cse44, _cse33, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,aguk,uhij->ai', _cse91, _cse92, _cse44, _cse33, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,ahiu,ugkj->ai', _cse91, _cse92, _cse48, _cse77, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,agiu,uhkj->ai', _cse91, _cse92, _cse48, _cse77, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,hguj,auik->ai', _cse91, _cse92, _cse78, _cse35, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cbgh,hguk,auij->ai', _cse91, _cse92, _cse78, _cse35, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aoig,pgjo,cbkp->ai', _cse88, _cse54, _cse2, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pcjo,gbkp->ai', _cse88, _cse54, _cse2, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pgko,cbjp->ai', _cse88, _cse54, _cse2, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pcko,gbjp->ai', _cse88, _cse54, _cse2, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pgkj,cbop->ai', _cse88, _cse54, _cse2, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pckj,gbop->ai', _cse88, _cse54, _cse2, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pbjo,gckp->ai', _cse88, _cse54, _cse2, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pbko,gcjp->ai', _cse88, _cse54, _cse2, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,pbkj,gcop->ai', _cse88, _cse54, _cse2, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,gcho,hbkj->ai', _cse88, _cse54, _cse10, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,gbho,hckj->ai', _cse88, _cse54, _cse10, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,gchj,hbko->ai', _cse88, _cse54, _cse10, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,gbhj,hcko->ai', _cse88, _cse54, _cse10, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,gchk,hbjo->ai', _cse88, _cse54, _cse10, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,gbhk,hcjo->ai', _cse88, _cse54, _cse10, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cbho,hgkj->ai', _cse88, _cse54, _cse10, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cbhj,hgko->ai', _cse88, _cse54, _cse10, _cse9, optimize=True)
            _iter -= 0.25 * _tmp
            _tmp = einsum('kjbc,aoig,cbhk,hgjo->ai', _cse88, _cse54, _cse10, _cse9, optimize=True)
            _iter += 0.25 * _tmp
            out += (-sigma[_tk_t3]) * (-sigma[_tk_t3_d1]) * _iter
    for _tk_t3 in range(ntau):
        Oe_t3 = np.exp(ei * tau[_tk_t3])
        Ve_t3 = np.exp(-ea * tau[_tk_t3])
        for _tk_t4_d1 in range(ntau):
            Oe_t4_d1 = np.exp(ei * tau[_tk_t4_d1])
            Ve_t4_d1 = np.exp(-ea * tau[_tk_t4_d1])
            _iter = np.zeros((nv, no))
            _cse0 = (((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse1 = (((g_aaaa[o, o, v, o] * Oe_t4_d1[:, None, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t4_d1[None, None, :, None])
            _cse2 = g_aaaa[v, v, o, o]
            _cse3 = ((((t2_1_aaaa * Ve_t3[None, :, None, None]) * Ve_t4_d1[None, :, None, None]) * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse4 = ((g_aaaa[v, v, o, o] * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse5 = ((t2_1_aaaa * Ve_t3[None, :, None, None]) * Ve_t4_d1[None, :, None, None])
            _cse6 = ((g_aaaa[v, v, o, o] * Oe_t3[None, None, None, :]) * Oe_t4_d1[None, None, None, :])
            _cse7 = ((g_aaaa[v, v, o, o] * Ve_t3[None, :, None, None]) * Ve_t4_d1[None, :, None, None])
            _cse8 = ((t2_1_aaaa * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse9 = ((((g_aaaa[v, v, o, o] * Ve_t3[None, :, None, None]) * Ve_t4_d1[None, :, None, None]) * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse10 = t2_1_aaaa
            _cse11 = ((((g_aaaa[v, v, o, o] * Ve_t3[None, :, None, None]) * Ve_t4_d1[None, :, None, None]) * Oe_t3[None, None, None, :]) * Oe_t4_d1[None, None, None, :])
            _cse12 = ((g_aaaa[v, v, o, o] * Ve_t3[:, None, None, None]) * Ve_t4_d1[:, None, None, None])
            _cse13 = ((((g_aaaa[v, v, o, o] * Ve_t3[:, None, None, None]) * Ve_t4_d1[:, None, None, None]) * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse14 = ((((g_aaaa[v, v, o, o] * Ve_t3[:, None, None, None]) * Ve_t4_d1[:, None, None, None]) * Oe_t3[None, None, None, :]) * Oe_t4_d1[None, None, None, :])
            _cse15 = (((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse16 = ((((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse17 = ((((g_aaaa[o, o, v, o] * Oe_t4_d1[:, None, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t4_d1[None, None, :, None]) * Oe_t3[None, None, None, :])
            _cse18 = (((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse19 = (((g_aaaa[o, v, v, v] * Oe_t4_d1[:, None, None, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t4_d1[None, None, None, :])
            _cse20 = (((((((g_aaaa[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse21 = ((((g_aaaa[o, v, v, v] * Oe_t4_d1[:, None, None, None]) * Ve_t3[None, :, None, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t4_d1[None, None, None, :])
            _cse22 = (((g_abab[o, o, o, v] * Oe_t4_d1[:, None, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t4_d1[None, None, None, :])
            _cse23 = t2_1_abab
            _cse24 = ((t2_1_abab * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse25 = ((g_abab[v, v, o, o] * Ve_t3[:, None, None, None]) * Ve_t4_d1[:, None, None, None])
            _cse26 = g_abab[v, v, o, o]
            _cse27 = ((((t2_1_aaaa * Ve_t3[:, None, None, None]) * Ve_t4_d1[:, None, None, None]) * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse28 = ((((g_abab[v, v, o, o] * Ve_t3[:, None, None, None]) * Ve_t4_d1[:, None, None, None]) * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse29 = ((g_abab[v, v, o, o] * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse30 = ((t2_1_aaaa * Ve_t3[:, None, None, None]) * Ve_t4_d1[:, None, None, None])
            _cse31 = ((t2_1_abab * Ve_t3[:, None, None, None]) * Ve_t4_d1[:, None, None, None])
            _cse32 = ((((t2_1_abab * Ve_t3[:, None, None, None]) * Ve_t4_d1[:, None, None, None]) * Oe_t3[None, None, :, None]) * Oe_t4_d1[None, None, :, None])
            _cse33 = ((((g_abab[o, o, o, v] * Oe_t4_d1[:, None, None, None]) * Oe_t4_d1[None, :, None, None]) * Oe_t3[None, None, :, None]) * Ve_t4_d1[None, None, None, :])
            _cse34 = (((g_abab[v, o, v, v] * Oe_t4_d1[None, :, None, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t4_d1[None, None, None, :])
            _cse35 = ((((g_abab[v, o, v, v] * Ve_t3[:, None, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t4_d1[None, None, None, :])
            _cse36 = (((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse37 = (((g_abab[o, o, v, o] * Oe_t4_d1[:, None, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t4_d1[None, None, :, None])
            _cse38 = (((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse39 = ((((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse40 = (((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse41 = (((((((g_abab[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse42 = (((g_abab[o, v, v, v] * Oe_t4_d1[:, None, None, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t4_d1[None, None, None, :])
            _cse43 = (((g_bbbb[o, o, v, o] * Oe_t4_d1[:, None, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t4_d1[None, None, :, None])
            _cse44 = t2_1_bbbb
            _cse45 = g_bbbb[v, v, o, o]
            _cse46 = (((g_bbbb[o, v, v, v] * Oe_t4_d1[:, None, None, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t4_d1[None, None, None, :])
            _cse47 = (((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse48 = (((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse49 = ((((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _cse50 = (((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t4_d1[None, None, :, None]) * Ve_t3[None, None, None, :])
            _cse51 = (((((((g_bbbb[o, o, v, v] * Oe_t3[:, None, None, None]) * Oe_t4_d1[:, None, None, None]) * Oe_t3[None, :, None, None]) * Oe_t4_d1[None, :, None, None]) * Ve_t3[None, None, :, None]) * Ve_t3[None, None, None, :]) * Ve_t4_d1[None, None, None, :])
            _tmp = einsum('kjbc,pogj,gbpo,caik->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcpo,baik->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbko,caip->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcko,baip->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbio,cakp->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcio,bakp->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbip,cako->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcip,bako->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbkp,caio->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gckp,baio->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbki,capo->ai', _cse0, _cse1, _cse6, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcki,bapo->ai', _cse0, _cse1, _cse6, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gapo,bcik->ai', _cse0, _cse1, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bapo,gcik->ai', _cse0, _cse1, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gako,bcip->ai', _cse0, _cse1, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bako,gcip->ai', _cse0, _cse1, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gaio,bckp->ai', _cse0, _cse1, _cse9, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,baio,gckp->ai', _cse0, _cse1, _cse9, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gaip,bcko->ai', _cse0, _cse1, _cse9, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,baip,gcko->ai', _cse0, _cse1, _cse9, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gakp,bcio->ai', _cse0, _cse1, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bakp,gcio->ai', _cse0, _cse1, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gaki,bcpo->ai', _cse0, _cse1, _cse11, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,baki,gcpo->ai', _cse0, _cse1, _cse11, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcpo,gaik->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acpo,gbik->ai', _cse0, _cse1, _cse12, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcko,gaip->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acko,gbip->ai', _cse0, _cse1, _cse12, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcio,gakp->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acio,gbkp->ai', _cse0, _cse1, _cse13, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcip,gako->ai', _cse0, _cse1, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acip,gbko->ai', _cse0, _cse1, _cse13, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bckp,gaio->ai', _cse0, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,ackp,gbio->ai', _cse0, _cse1, _cse12, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcki,gapo->ai', _cse0, _cse1, _cse6, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acki,gbpo->ai', _cse0, _cse1, _cse14, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbpo,caij->ai', _cse15, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcpo,baij->ai', _cse15, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbjo,caip->ai', _cse15, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcjo,baip->ai', _cse15, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbio,cajp->ai', _cse15, _cse1, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcio,bajp->ai', _cse15, _cse1, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbip,cajo->ai', _cse15, _cse1, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcip,bajo->ai', _cse15, _cse1, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbjp,caio->ai', _cse15, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcjp,baio->ai', _cse15, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbji,capo->ai', _cse15, _cse1, _cse6, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcji,bapo->ai', _cse15, _cse1, _cse6, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gapo,bcij->ai', _cse15, _cse1, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bapo,gcij->ai', _cse15, _cse1, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gajo,bcip->ai', _cse15, _cse1, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bajo,gcip->ai', _cse15, _cse1, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gaio,bcjp->ai', _cse15, _cse1, _cse9, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,baio,gcjp->ai', _cse15, _cse1, _cse9, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gaip,bcjo->ai', _cse15, _cse1, _cse9, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,baip,gcjo->ai', _cse15, _cse1, _cse9, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gajp,bcio->ai', _cse15, _cse1, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bajp,gcio->ai', _cse15, _cse1, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gaji,bcpo->ai', _cse15, _cse1, _cse11, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,baji,gcpo->ai', _cse15, _cse1, _cse11, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcpo,gaij->ai', _cse15, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acpo,gbij->ai', _cse15, _cse1, _cse12, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcjo,gaip->ai', _cse15, _cse1, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acjo,gbip->ai', _cse15, _cse1, _cse12, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcio,gajp->ai', _cse15, _cse1, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acio,gbjp->ai', _cse15, _cse1, _cse13, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcip,gajo->ai', _cse15, _cse1, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acip,gbjo->ai', _cse15, _cse1, _cse13, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcjp,gaio->ai', _cse15, _cse1, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acjp,gbio->ai', _cse15, _cse1, _cse12, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcji,gapo->ai', _cse15, _cse1, _cse6, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acji,gbpo->ai', _cse15, _cse1, _cse14, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbpo,cakj->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcpo,bakj->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbjo,cakp->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcjo,bakp->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbko,cajp->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcko,bajp->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbkp,cajo->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gckp,bajo->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbjp,cako->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcjp,bako->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbjk,capo->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcjk,bapo->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gapo,bckj->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bapo,gckj->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gajo,bckp->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bajo,gckp->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gako,bcjp->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bako,gcjp->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gakp,bcjo->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bakp,gcjo->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gajp,bcko->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bajp,gcko->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gajk,bcpo->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bajk,gcpo->ai', _cse16, _cse17, _cse7, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bcpo,gakj->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acpo,gbkj->ai', _cse16, _cse17, _cse12, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bcjo,gakp->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acjo,gbkp->ai', _cse16, _cse17, _cse12, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bcko,gajp->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acko,gbjp->ai', _cse16, _cse17, _cse12, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bckp,gajo->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,ackp,gbjo->ai', _cse16, _cse17, _cse12, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bcjp,gako->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acjp,gbko->ai', _cse16, _cse17, _cse12, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bcjk,gapo->ai', _cse16, _cse17, _cse2, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acjk,gbpo->ai', _cse16, _cse17, _cse12, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghjo,caik->ai', _cse18, _cse19, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcjo,haik->ai', _cse18, _cse19, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghko,caij->ai', _cse18, _cse19, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcko,haij->ai', _cse18, _cse19, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghio,cakj->ai', _cse18, _cse19, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcio,hakj->ai', _cse18, _cse19, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghij,cako->ai', _cse18, _cse19, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcij,hako->ai', _cse18, _cse19, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghkj,caio->ai', _cse18, _cse19, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gckj,haio->ai', _cse18, _cse19, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghki,cajo->ai', _cse18, _cse19, _cse6, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcki,hajo->ai', _cse18, _cse19, _cse6, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gajo,hcik->ai', _cse18, _cse19, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hajo,gcik->ai', _cse18, _cse19, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gako,hcij->ai', _cse18, _cse19, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hako,gcij->ai', _cse18, _cse19, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gaio,hckj->ai', _cse18, _cse19, _cse9, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,haio,gckj->ai', _cse18, _cse19, _cse9, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gaij,hcko->ai', _cse18, _cse19, _cse9, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,haij,gcko->ai', _cse18, _cse19, _cse9, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gakj,hcio->ai', _cse18, _cse19, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hakj,gcio->ai', _cse18, _cse19, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gaki,hcjo->ai', _cse18, _cse19, _cse11, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,haki,gcjo->ai', _cse18, _cse19, _cse11, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hcjo,gaik->ai', _cse18, _cse19, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acjo,ghik->ai', _cse18, _cse19, _cse12, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hcko,gaij->ai', _cse18, _cse19, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acko,ghij->ai', _cse18, _cse19, _cse12, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hcio,gakj->ai', _cse18, _cse19, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acio,ghkj->ai', _cse18, _cse19, _cse13, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hcij,gako->ai', _cse18, _cse19, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acij,ghko->ai', _cse18, _cse19, _cse13, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hckj,gaio->ai', _cse18, _cse19, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ackj,ghio->ai', _cse18, _cse19, _cse12, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hcki,gajo->ai', _cse18, _cse19, _cse6, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acki,ghjo->ai', _cse18, _cse19, _cse14, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghjo,baik->ai', _cse20, _cse19, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbjo,haik->ai', _cse20, _cse19, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghko,baij->ai', _cse20, _cse19, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbko,haij->ai', _cse20, _cse19, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghio,bakj->ai', _cse20, _cse19, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbio,hakj->ai', _cse20, _cse19, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghij,bako->ai', _cse20, _cse19, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbij,hako->ai', _cse20, _cse19, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghkj,baio->ai', _cse20, _cse19, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbkj,haio->ai', _cse20, _cse19, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghki,bajo->ai', _cse20, _cse19, _cse6, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbki,hajo->ai', _cse20, _cse19, _cse6, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gajo,hbik->ai', _cse20, _cse19, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hajo,gbik->ai', _cse20, _cse19, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gako,hbij->ai', _cse20, _cse19, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hako,gbij->ai', _cse20, _cse19, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gaio,hbkj->ai', _cse20, _cse19, _cse9, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,haio,gbkj->ai', _cse20, _cse19, _cse9, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gaij,hbko->ai', _cse20, _cse19, _cse9, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,haij,gbko->ai', _cse20, _cse19, _cse9, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gakj,hbio->ai', _cse20, _cse19, _cse7, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hakj,gbio->ai', _cse20, _cse19, _cse7, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gaki,hbjo->ai', _cse20, _cse19, _cse11, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,haki,gbjo->ai', _cse20, _cse19, _cse11, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hbjo,gaik->ai', _cse20, _cse19, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abjo,ghik->ai', _cse20, _cse19, _cse12, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hbko,gaij->ai', _cse20, _cse19, _cse2, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abko,ghij->ai', _cse20, _cse19, _cse12, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hbio,gakj->ai', _cse20, _cse19, _cse4, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abio,ghkj->ai', _cse20, _cse19, _cse13, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hbij,gako->ai', _cse20, _cse19, _cse4, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abij,ghko->ai', _cse20, _cse19, _cse13, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hbkj,gaio->ai', _cse20, _cse19, _cse2, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abkj,ghio->ai', _cse20, _cse19, _cse12, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hbki,gajo->ai', _cse20, _cse19, _cse6, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abki,ghjo->ai', _cse20, _cse19, _cse14, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghjo,bcik->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbjo,hcik->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghko,bcij->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbko,hcij->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghio,bckj->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbio,hckj->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghij,bcko->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbij,hcko->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghkj,bcio->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbkj,hcio->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghki,bcjo->ai', _cse16, _cse21, _cse6, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbki,hcjo->ai', _cse16, _cse21, _cse6, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcjo,hbik->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcjo,gbik->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcko,hbij->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcko,gbij->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcio,hbkj->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcio,gbkj->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcij,hbko->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcij,gbko->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gckj,hbio->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hckj,gbio->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcki,hbjo->ai', _cse16, _cse21, _cse6, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcki,gbjo->ai', _cse16, _cse21, _cse6, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbjo,gcik->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,cbjo,ghik->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbko,gcij->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,cbko,ghij->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbio,gckj->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,cbio,ghkj->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbij,gcko->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,cbij,ghko->ai', _cse16, _cse21, _cse4, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbkj,gcio->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,cbkj,ghio->ai', _cse16, _cse21, _cse2, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbki,gcjo->ai', _cse16, _cse21, _cse6, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,cbki,ghjo->ai', _cse16, _cse21, _cse6, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,abip,cgko->ai', _cse0, _cse22, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,acip,bgko->ai', _cse0, _cse22, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,abkp,cgio->ai', _cse0, _cse22, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,ackp,bgio->ai', _cse0, _cse22, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,abki,cgpo->ai', _cse0, _cse22, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,acki,bgpo->ai', _cse0, _cse22, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,agpo,bcik->ai', _cse0, _cse22, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,bgpo,acik->ai', _cse0, _cse22, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,agko,bcip->ai', _cse0, _cse22, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,bgko,acip->ai', _cse0, _cse22, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,agio,bckp->ai', _cse0, _cse22, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,bgio,ackp->ai', _cse0, _cse22, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,cgpo,abik->ai', _cse0, _cse22, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,cgko,abip->ai', _cse0, _cse22, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,cgio,abkp->ai', _cse0, _cse22, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,bcip,agko->ai', _cse0, _cse22, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,bckp,agio->ai', _cse0, _cse22, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pojg,bcki,agpo->ai', _cse0, _cse22, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,abip,cgjo->ai', _cse15, _cse22, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,acip,bgjo->ai', _cse15, _cse22, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,abjp,cgio->ai', _cse15, _cse22, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,acjp,bgio->ai', _cse15, _cse22, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,abji,cgpo->ai', _cse15, _cse22, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,acji,bgpo->ai', _cse15, _cse22, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,agpo,bcij->ai', _cse15, _cse22, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bgpo,acij->ai', _cse15, _cse22, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,agjo,bcip->ai', _cse15, _cse22, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bgjo,acip->ai', _cse15, _cse22, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,agio,bcjp->ai', _cse15, _cse22, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bgio,acjp->ai', _cse15, _cse22, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,cgpo,abij->ai', _cse15, _cse22, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,cgjo,abip->ai', _cse15, _cse22, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,cgio,abjp->ai', _cse15, _cse22, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bcip,agjo->ai', _cse15, _cse22, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bcjp,agio->ai', _cse15, _cse22, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bcji,agpo->ai', _cse15, _cse22, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,abio,cgkp->ai', _cse0, _cse22, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,acio,bgkp->ai', _cse0, _cse22, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,abko,cgip->ai', _cse0, _cse22, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,acko,bgip->ai', _cse0, _cse22, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,abki,cgop->ai', _cse0, _cse22, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,acki,bgop->ai', _cse0, _cse22, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,agop,bcik->ai', _cse0, _cse22, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,bgop,acik->ai', _cse0, _cse22, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,agkp,bcio->ai', _cse0, _cse22, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,bgkp,acio->ai', _cse0, _cse22, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,agip,bcko->ai', _cse0, _cse22, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,bgip,acko->ai', _cse0, _cse22, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,cgop,abik->ai', _cse0, _cse22, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,cgkp,abio->ai', _cse0, _cse22, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,cgip,abko->ai', _cse0, _cse22, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,bcio,agkp->ai', _cse0, _cse22, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,bcko,agip->ai', _cse0, _cse22, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opjg,bcki,agop->ai', _cse0, _cse22, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,abio,cgjp->ai', _cse15, _cse22, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,acio,bgjp->ai', _cse15, _cse22, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,abjo,cgip->ai', _cse15, _cse22, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,acjo,bgip->ai', _cse15, _cse22, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,abji,cgop->ai', _cse15, _cse22, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,acji,bgop->ai', _cse15, _cse22, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,agop,bcij->ai', _cse15, _cse22, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bgop,acij->ai', _cse15, _cse22, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,agjp,bcio->ai', _cse15, _cse22, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bgjp,acio->ai', _cse15, _cse22, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,agip,bcjo->ai', _cse15, _cse22, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bgip,acjo->ai', _cse15, _cse22, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,cgop,abij->ai', _cse15, _cse22, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,cgjp,abio->ai', _cse15, _cse22, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,cgip,abjo->ai', _cse15, _cse22, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bcio,agjp->ai', _cse15, _cse22, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bcjo,agip->ai', _cse15, _cse22, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bcji,agop->ai', _cse15, _cse22, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,abkp,cgjo->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,ackp,bgjo->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,abjp,cgko->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,acjp,bgko->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,abjk,cgpo->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,acjk,bgpo->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agpo,bckj->ai', _cse16, _cse33, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bgpo,ackj->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agjo,bckp->ai', _cse16, _cse33, _cse25, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bgjo,ackp->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agko,bcjp->ai', _cse16, _cse33, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bgko,acjp->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,cgpo,abkj->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,cgjo,abkp->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,cgko,abjp->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bckp,agjo->ai', _cse16, _cse33, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bcjp,agko->ai', _cse16, _cse33, _cse2, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bcjk,agpo->ai', _cse16, _cse33, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,abko,cgjp->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,acko,bgjp->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,abjo,cgkp->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,acjo,bgkp->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,abjk,cgop->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,acjk,bgop->ai', _cse16, _cse33, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agop,bckj->ai', _cse16, _cse33, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bgop,ackj->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agjp,bcko->ai', _cse16, _cse33, _cse25, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bgjp,acko->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agkp,bcjo->ai', _cse16, _cse33, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bgkp,acjo->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,cgop,abkj->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,cgjp,abko->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,cgkp,abjo->ai', _cse16, _cse33, _cse26, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bcko,agjp->ai', _cse16, _cse33, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bcjo,agkp->ai', _cse16, _cse33, _cse2, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bcjk,agop->ai', _cse16, _cse33, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gaij,chko->ai', _cse18, _cse34, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gcij,ahko->ai', _cse18, _cse34, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gakj,chio->ai', _cse18, _cse34, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gckj,ahio->ai', _cse18, _cse34, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gaki,chjo->ai', _cse18, _cse34, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gcki,ahjo->ai', _cse18, _cse34, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ghjo,acik->ai', _cse18, _cse34, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ahjo,gcik->ai', _cse18, _cse34, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ghko,acij->ai', _cse18, _cse34, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ahko,gcij->ai', _cse18, _cse34, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ghio,ackj->ai', _cse18, _cse34, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ahio,gckj->ai', _cse18, _cse34, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,chjo,gaik->ai', _cse18, _cse34, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,chko,gaij->ai', _cse18, _cse34, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,chio,gakj->ai', _cse18, _cse34, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,acij,ghko->ai', _cse18, _cse34, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ackj,ghio->ai', _cse18, _cse34, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,acki,ghjo->ai', _cse18, _cse34, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,gaij,bhko->ai', _cse20, _cse34, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,gbij,ahko->ai', _cse20, _cse34, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,gakj,bhio->ai', _cse20, _cse34, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,gbkj,ahio->ai', _cse20, _cse34, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,gaki,bhjo->ai', _cse20, _cse34, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,gbki,ahjo->ai', _cse20, _cse34, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,ghjo,abik->ai', _cse20, _cse34, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,ahjo,gbik->ai', _cse20, _cse34, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,ghko,abij->ai', _cse20, _cse34, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,ahko,gbij->ai', _cse20, _cse34, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,ghio,abkj->ai', _cse20, _cse34, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,ahio,gbkj->ai', _cse20, _cse34, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,bhjo,gaik->ai', _cse20, _cse34, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,bhko,gaij->ai', _cse20, _cse34, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,bhio,gakj->ai', _cse20, _cse34, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,abij,ghko->ai', _cse20, _cse34, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,abkj,ghio->ai', _cse20, _cse34, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cogh,abki,ghjo->ai', _cse20, _cse34, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,ahij,cgko->ai', _cse18, _cse34, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,acij,hgko->ai', _cse18, _cse34, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,ahkj,cgio->ai', _cse18, _cse34, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,ackj,hgio->ai', _cse18, _cse34, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,ahki,cgjo->ai', _cse18, _cse34, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,acki,hgjo->ai', _cse18, _cse34, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,agjo,hcik->ai', _cse18, _cse34, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hgjo,acik->ai', _cse18, _cse34, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,agko,hcij->ai', _cse18, _cse34, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hgko,acij->ai', _cse18, _cse34, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,agio,hckj->ai', _cse18, _cse34, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hgio,ackj->ai', _cse18, _cse34, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,cgjo,ahik->ai', _cse18, _cse34, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,cgko,ahij->ai', _cse18, _cse34, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,cgio,ahkj->ai', _cse18, _cse34, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hcij,agko->ai', _cse18, _cse34, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hckj,agio->ai', _cse18, _cse34, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hcki,agjo->ai', _cse18, _cse34, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,ahij,bgko->ai', _cse20, _cse34, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,abij,hgko->ai', _cse20, _cse34, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,ahkj,bgio->ai', _cse20, _cse34, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,abkj,hgio->ai', _cse20, _cse34, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,ahki,bgjo->ai', _cse20, _cse34, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,abki,hgjo->ai', _cse20, _cse34, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,agjo,hbik->ai', _cse20, _cse34, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,hgjo,abik->ai', _cse20, _cse34, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,agko,hbij->ai', _cse20, _cse34, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,hgko,abij->ai', _cse20, _cse34, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,agio,hbkj->ai', _cse20, _cse34, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,hgio,abkj->ai', _cse20, _cse34, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,bgjo,ahik->ai', _cse20, _cse34, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,bgko,ahij->ai', _cse20, _cse34, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,bgio,ahkj->ai', _cse20, _cse34, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,hbij,agko->ai', _cse20, _cse34, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,hbkj,agio->ai', _cse20, _cse34, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,cohg,hbki,agjo->ai', _cse20, _cse34, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gcij,bhko->ai', _cse16, _cse35, _cse4, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gbij,chko->ai', _cse16, _cse35, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gckj,bhio->ai', _cse16, _cse35, _cse2, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gbkj,chio->ai', _cse16, _cse35, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gcki,bhjo->ai', _cse16, _cse35, _cse6, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gbki,chjo->ai', _cse16, _cse35, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghjo,cbik->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,chjo,gbik->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghko,cbij->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,chko,gbij->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghio,cbkj->ai', _cse16, _cse35, _cse29, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,chio,gbkj->ai', _cse16, _cse35, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bhjo,gcik->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bhko,gcij->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bhio,gckj->ai', _cse16, _cse35, _cse29, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,cbij,ghko->ai', _cse16, _cse35, _cse4, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,cbkj,ghio->ai', _cse16, _cse35, _cse2, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,cbki,ghjo->ai', _cse16, _cse35, _cse6, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,chij,bgko->ai', _cse16, _cse35, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,cbij,hgko->ai', _cse16, _cse35, _cse4, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,chkj,bgio->ai', _cse16, _cse35, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,cbkj,hgio->ai', _cse16, _cse35, _cse2, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,chki,bgjo->ai', _cse16, _cse35, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,cbki,hgjo->ai', _cse16, _cse35, _cse6, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,cgjo,hbik->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgjo,cbik->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,cgko,hbij->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgko,cbij->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,cgio,hbkj->ai', _cse16, _cse35, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgio,cbkj->ai', _cse16, _cse35, _cse29, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bgjo,chik->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bgko,chij->ai', _cse16, _cse35, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bgio,chkj->ai', _cse16, _cse35, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hbij,cgko->ai', _cse16, _cse35, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hbkj,cgio->ai', _cse16, _cse35, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hbki,cgjo->ai', _cse16, _cse35, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbip,acko->ai', _cse36, _cse37, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gaip,bcko->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbkp,acio->ai', _cse36, _cse37, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gakp,bcio->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbki,acpo->ai', _cse36, _cse37, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gaki,bcpo->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcpo,baik->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcpo,gaik->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcko,baip->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcko,gaip->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcio,bakp->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcio,gakp->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acpo,gbik->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acko,gbip->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acio,gbkp->ai', _cse36, _cse37, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,baip,gcko->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bakp,gcio->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,baki,gcpo->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gbio,ackp->ai', _cse36, _cse37, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gaio,bckp->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gbko,acip->ai', _cse36, _cse37, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gako,bcip->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gbki,acop->ai', _cse36, _cse37, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gaki,bcop->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gcop,baik->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,bcop,gaik->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gckp,baio->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,bckp,gaio->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gcip,bako->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,bcip,gako->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,acop,gbik->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,ackp,gbio->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,acip,gbko->ai', _cse36, _cse37, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,baio,gckp->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,bako,gcip->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,baki,gcop->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbip,acoj->ai', _cse38, _cse1, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gaip,bcoj->ai', _cse38, _cse1, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbop,acij->ai', _cse38, _cse1, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gaop,bcij->ai', _cse38, _cse1, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gboi,acpj->ai', _cse38, _cse1, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gaoi,bcpj->ai', _cse38, _cse1, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcpj,baio->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcpj,gaio->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcoj,baip->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcoj,gaip->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcij,baop->ai', _cse38, _cse1, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcij,gaop->ai', _cse38, _cse1, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acpj,gbio->ai', _cse38, _cse1, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acoj,gbip->ai', _cse38, _cse1, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acij,gbop->ai', _cse38, _cse1, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,baip,gcoj->ai', _cse38, _cse1, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,baop,gcij->ai', _cse38, _cse1, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,baoi,gcpj->ai', _cse38, _cse1, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbkp,acoj->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gakp,bcoj->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbop,ackj->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gaop,bckj->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbok,acpj->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gaok,bcpj->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcpj,bako->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bcpj,gako->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcoj,bakp->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bcoj,gakp->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gckj,baop->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bckj,gaop->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acpj,gbko->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acoj,gbkp->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,ackj,gbop->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bakp,gcoj->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,baop,gckj->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,baok,gcpj->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghio,ackj->ai', _cse40, _cse19, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gaio,hckj->ai', _cse40, _cse19, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghko,acij->ai', _cse40, _cse19, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gako,hcij->ai', _cse40, _cse19, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghki,acoj->ai', _cse40, _cse19, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gaki,hcoj->ai', _cse40, _cse19, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcoj,haik->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hcoj,gaik->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gckj,haio->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hckj,gaio->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcij,hako->ai', _cse40, _cse19, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hcij,gako->ai', _cse40, _cse19, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acoj,ghik->ai', _cse40, _cse19, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ackj,ghio->ai', _cse40, _cse19, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acij,ghko->ai', _cse40, _cse19, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,haio,gckj->ai', _cse40, _cse19, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hako,gcij->ai', _cse40, _cse19, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,haki,gcoj->ai', _cse40, _cse19, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghio,bckj->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbio,hckj->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghko,bcij->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbko,hcij->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghki,bcoj->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbki,hcoj->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcoj,hbik->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcoj,gbik->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gckj,hbio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hckj,gbio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcij,hbko->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcij,gbko->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,bcoj,ghik->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,bckj,ghio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,bcij,ghko->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbio,gckj->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbko,gcij->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbki,gcoj->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gaio,bhkj->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbio,ahkj->ai', _cse41, _cse42, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gako,bhij->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbko,ahij->ai', _cse41, _cse42, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gaki,bhoj->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbki,ahoj->ai', _cse41, _cse42, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghoj,abik->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahoj,gbik->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghkj,abio->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahkj,gbio->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghij,abko->ai', _cse41, _cse42, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahij,gbko->ai', _cse41, _cse42, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bhoj,gaik->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bhkj,gaio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bhij,gako->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abio,ghkj->ai', _cse41, _cse42, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abko,ghij->ai', _cse41, _cse42, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abki,ghoj->ai', _cse41, _cse42, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,ahio,bgkj->ai', _cse41, _cse42, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,abio,hgkj->ai', _cse41, _cse42, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,ahko,bgij->ai', _cse41, _cse42, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,abko,hgij->ai', _cse41, _cse42, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,ahki,bgoj->ai', _cse41, _cse42, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,abki,hgoj->ai', _cse41, _cse42, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,agoj,hbik->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hgoj,abik->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,agkj,hbio->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hgkj,abio->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,agij,hbko->ai', _cse41, _cse42, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hgij,abko->ai', _cse41, _cse42, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,bgoj,ahik->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,bgkj,ahio->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,bgij,ahko->ai', _cse41, _cse42, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hbio,agkj->ai', _cse41, _cse42, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hbko,agij->ai', _cse41, _cse42, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hbki,agoj->ai', _cse41, _cse42, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,agko,bcip->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,agio,bckp->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,agip,bcko->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,agkp,bcio->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,abki,gcpo->ai', _cse36, _cse43, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acko,bgip->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcko,agip->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acio,bgkp->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcio,agkp->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acip,bgko->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcip,agko->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,ackp,bgio->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bckp,agio->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,cgpo,abik->ai', _cse36, _cse43, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bgko,acip->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bgio,ackp->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bgip,acko->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bgkp,acio->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,agpo,bcij->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,agio,bcpj->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,agij,bcpo->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,agpj,bcio->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,abpi,gcjo->ai', _cse38, _cse22, _cse14, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,acpo,bgij->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bcpo,agij->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,acio,bgpj->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bcio,agpj->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,acij,bgpo->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bcij,agpo->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,acpj,bgio->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bcpj,agio->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,cgjo,abip->ai', _cse38, _cse22, _cse45, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bgpo,acij->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bgio,acpj->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bgij,acpo->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pokg,bgpj,acio->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,agoj,bcip->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,agij,bcop->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,agip,bcoj->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,agop,bcij->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,aboi,gcpj->ai', _cse38, _cse22, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,acoj,bgip->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bcoj,agip->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,acij,bgop->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bcij,agop->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,acip,bgoj->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bcip,agoj->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,acop,bgij->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bcop,agij->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,cgpj,abio->ai', _cse38, _cse22, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bgoj,acip->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bgij,acop->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bgip,acoj->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opkg,bgop,acij->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agpo,bckj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agko,bcpj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agkj,bcpo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agpj,bcko->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,abpk,gcjo->ai', _cse39, _cse33, _cse12, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,acpo,bgkj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bcpo,agkj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,acko,bgpj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bcko,agpj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,ackj,bgpo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bckj,agpo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,acpj,bgko->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bcpj,agko->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,cgjo,abkp->ai', _cse39, _cse33, _cse45, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bgpo,ackj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bgko,acpj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bgkj,acpo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bgpj,acko->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agoj,bckp->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agkj,bcop->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agkp,bcoj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agop,bckj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,abok,gcpj->ai', _cse39, _cse33, _cse12, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,acoj,bgkp->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bcoj,agkp->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,ackj,bgop->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bckj,agop->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,ackp,bgoj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bckp,agoj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,acop,bgkj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bcop,agkj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,cgpj,abko->ai', _cse39, _cse33, _cse45, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bgoj,ackp->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bgkj,acop->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bgkp,acoj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bgop,ackj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ghko,acij->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ghio,ackj->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ghij,acko->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ghkj,acio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gaki,hcjo->ai', _cse40, _cse34, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gcko,ahij->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,acko,ghij->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gcio,ahkj->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,acio,ghkj->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gcij,ahko->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,acij,ghko->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,gckj,ahio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ackj,ghio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,chjo,gaik->ai', _cse40, _cse34, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ahko,gcij->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ahio,gckj->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ahij,gcko->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bogh,ahkj,gcio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghko,bcij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghio,bckj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghij,bcko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghkj,bcio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gbki,hcjo->ai', _cse39, _cse35, _cse6, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gcko,bhij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bcko,ghij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gcio,bhkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bcio,ghkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gcij,bhko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bcij,ghko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gckj,bhio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bckj,ghio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,chjo,gbik->ai', _cse39, _cse35, _cse45, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bhko,gcij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bhio,gckj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bhij,gcko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bhkj,gcio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,agko,hcij->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,agio,hckj->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,agij,hcko->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,agkj,hcio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,ahki,gcjo->ai', _cse40, _cse34, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,acko,hgij->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hcko,agij->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,acio,hgkj->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hcio,agkj->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,acij,hgko->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hcij,agko->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,ackj,hgio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hckj,agio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,cgjo,ahik->ai', _cse40, _cse34, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hgko,acij->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hgio,ackj->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hgij,acko->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,bohg,hgkj,acio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bgko,hcij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bgio,hckj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bgij,hcko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bgkj,hcio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bhki,gcjo->ai', _cse39, _cse35, _cse6, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bcko,hgij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hcko,bgij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bcio,hgkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hcio,bgkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bcij,hgko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hcij,bgko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bckj,hgio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hckj,bgio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,cgjo,bhik->ai', _cse39, _cse35, _cse45, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgko,bcij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgio,bckj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgij,bcko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgkj,bcio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bgko,ahij->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bgio,ahkj->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bgij,ahko->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bgkj,ahio->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,baki,ghjo->ai', _cse41, _cse46, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bhko,agij->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahko,bgij->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bhio,agkj->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahio,bgkj->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bhij,agko->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahij,bgko->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bhkj,agio->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahkj,bgio->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hgjo,baik->ai', _cse41, _cse46, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,agko,bhij->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,agio,bhkj->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,agij,bhko->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,agkj,bhio->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,gbip,acjo->ai', _cse36, _cse37, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,gaip,bcjo->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,gbjp,acio->ai', _cse36, _cse37, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,gajp,bcio->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,gbji,acpo->ai', _cse36, _cse37, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,gaji,bcpo->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,gcpo,baij->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bcpo,gaij->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,gcjo,baip->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bcjo,gaip->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,gcio,bajp->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bcio,gajp->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,acpo,gbij->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,acjo,gbip->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,acio,gbjp->ai', _cse36, _cse37, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,baip,gcjo->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bajp,gcio->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,baji,gcpo->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,gbio,acjp->ai', _cse36, _cse37, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,gaio,bcjp->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,gbjo,acip->ai', _cse36, _cse37, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,gajo,bcip->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,gbji,acop->ai', _cse36, _cse37, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,gaji,bcop->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,gcop,baij->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,bcop,gaij->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,gcjp,baio->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,bcjp,gaio->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,gcip,bajo->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,bcip,gajo->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,acop,gbij->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,acjp,gbio->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,acip,gbjo->ai', _cse36, _cse37, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,baio,gcjp->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,bajo,gcip->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opgk,baji,gcop->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,gbip,acok->ai', _cse38, _cse1, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,gaip,bcok->ai', _cse38, _cse1, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,gbop,acik->ai', _cse38, _cse1, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,gaop,bcik->ai', _cse38, _cse1, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,gboi,acpk->ai', _cse38, _cse1, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,gaoi,bcpk->ai', _cse38, _cse1, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,gcpk,baio->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,bcpk,gaio->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,gcok,baip->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,bcok,gaip->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,gcik,baop->ai', _cse38, _cse1, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,bcik,gaop->ai', _cse38, _cse1, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,acpk,gbio->ai', _cse38, _cse1, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,acok,gbip->ai', _cse38, _cse1, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,acik,gbop->ai', _cse38, _cse1, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,baip,gcok->ai', _cse38, _cse1, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,baop,gcik->ai', _cse38, _cse1, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogj,baoi,gcpk->ai', _cse38, _cse1, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,gbjp,acok->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,gajp,bcok->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,gbop,acjk->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,gaop,bcjk->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,gboj,acpk->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,gaoj,bcpk->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,gcpk,bajo->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,bcpk,gajo->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,gcok,bajp->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,bcok,gajp->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,gcjk,baop->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,bcjk,gaop->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,acpk,gbjo->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,acok,gbjp->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,acjk,gbop->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,bajp,gcok->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,baop,gcjk->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogi,baoj,gcpk->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,ghio,acjk->ai', _cse40, _cse19, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,gaio,hcjk->ai', _cse40, _cse19, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,ghjo,acik->ai', _cse40, _cse19, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,gajo,hcik->ai', _cse40, _cse19, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,ghji,acok->ai', _cse40, _cse19, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,gaji,hcok->ai', _cse40, _cse19, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,gcok,haij->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,hcok,gaij->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,gcjk,haio->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,hcjk,gaio->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,gcik,hajo->ai', _cse40, _cse19, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,hcik,gajo->ai', _cse40, _cse19, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,acok,ghij->ai', _cse40, _cse19, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,acjk,ghio->ai', _cse40, _cse19, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,acik,ghjo->ai', _cse40, _cse19, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,haio,gcjk->ai', _cse40, _cse19, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,hajo,gcik->ai', _cse40, _cse19, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,obgh,haji,gcok->ai', _cse40, _cse19, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,ghio,bcjk->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,gbio,hcjk->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,ghjo,bcik->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,gbjo,hcik->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,ghji,bcok->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,gbji,hcok->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,gcok,hbij->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,hcok,gbij->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,gcjk,hbio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,hcjk,gbio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,gcik,hbjo->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,hcik,gbjo->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,bcok,ghij->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,bcjk,ghio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,bcik,ghjo->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,hbio,gcjk->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,hbjo,gcik->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,oagh,hbji,gcok->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,gaio,bhjk->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,gbio,ahjk->ai', _cse41, _cse42, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,gajo,bhik->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,gbjo,ahik->ai', _cse41, _cse42, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,gaji,bhok->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,gbji,ahok->ai', _cse41, _cse42, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ghok,abij->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ahok,gbij->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ghjk,abio->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ahjk,gbio->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ghik,abjo->ai', _cse41, _cse42, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ahik,gbjo->ai', _cse41, _cse42, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bhok,gaij->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bhjk,gaio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bhik,gajo->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,abio,ghjk->ai', _cse41, _cse42, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,abjo,ghik->ai', _cse41, _cse42, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,abji,ghok->ai', _cse41, _cse42, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,ahio,bgjk->ai', _cse41, _cse42, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,abio,hgjk->ai', _cse41, _cse42, _cse13, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,ahjo,bgik->ai', _cse41, _cse42, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,abjo,hgik->ai', _cse41, _cse42, _cse12, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,ahji,bgok->ai', _cse41, _cse42, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,abji,hgok->ai', _cse41, _cse42, _cse14, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,agok,hbij->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,hgok,abij->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,agjk,hbio->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,hgjk,abio->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,agik,hbjo->ai', _cse41, _cse42, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,hgik,abjo->ai', _cse41, _cse42, _cse29, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,bgok,ahij->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,bgjk,ahio->ai', _cse41, _cse42, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,bgik,ahjo->ai', _cse41, _cse42, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,hbio,agjk->ai', _cse41, _cse42, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,hbjo,agik->ai', _cse41, _cse42, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ochg,hbji,agok->ai', _cse41, _cse42, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,agjo,bcip->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,agio,bcjp->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,agip,bcjo->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,agjp,bcio->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,abji,gcpo->ai', _cse36, _cse43, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,acjo,bgip->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bcjo,agip->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,acio,bgjp->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bcio,agjp->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,acip,bgjo->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bcip,agjo->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,acjp,bgio->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bcjp,agio->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,cgpo,abij->ai', _cse36, _cse43, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bgjo,acip->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bgio,acjp->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bgip,acjo->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pogk,bgjp,acio->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,agpo,bcik->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,agio,bcpk->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,agik,bcpo->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,agpk,bcio->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,abpi,gcko->ai', _cse38, _cse22, _cse14, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,acpo,bgik->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,bcpo,agik->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,acio,bgpk->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,bcio,agpk->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,acik,bgpo->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,bcik,agpo->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,acpk,bgio->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,bcpk,agio->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,cgko,abip->ai', _cse38, _cse22, _cse45, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,bgpo,acik->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,bgio,acpk->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,bgik,acpo->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,pojg,bgpk,acio->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,agok,bcip->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,agik,bcop->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,agip,bcok->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,agop,bcik->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,aboi,gcpk->ai', _cse38, _cse22, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,acok,bgip->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,bcok,agip->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,acik,bgop->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,bcik,agop->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,acip,bgok->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,bcip,agok->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,acop,bgik->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,bcop,agik->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,cgpk,abio->ai', _cse38, _cse22, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,bgok,acip->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,bgik,acop->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,bgip,acok->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opjg,bgop,acik->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poig,agpo,bcjk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poig,agjo,bcpk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,agjk,bcpo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poig,agpk,bcjo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,abpj,gcko->ai', _cse39, _cse33, _cse12, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,acpo,bgjk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,bcpo,agjk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poig,acjo,bgpk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poig,bcjo,agpk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,acjk,bgpo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,bcjk,agpo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poig,acpk,bgjo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poig,bcpk,agjo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,cgko,abjp->ai', _cse39, _cse33, _cse45, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,bgpo,acjk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,bgjo,acpk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,poig,bgjk,acpo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,poig,bgpk,acjo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,agok,bcjp->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opig,agjk,bcop->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,agjp,bcok->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opig,agop,bcjk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,aboj,gcpk->ai', _cse39, _cse33, _cse12, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,acok,bgjp->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,bcok,agjp->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opig,acjk,bgop->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opig,bcjk,agop->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,acjp,bgok->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,bcjp,agok->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opig,acop,bgjk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opig,bcop,agjk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,cgpk,abjo->ai', _cse39, _cse33, _cse45, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,bgok,acjp->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,bgjk,acop->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,opig,bgjp,acok->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,opig,bgop,acjk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,ghjo,acik->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,ghio,acjk->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,ghik,acjo->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,ghjk,acio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,gaji,hcko->ai', _cse40, _cse34, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,gcjo,ahik->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,acjo,ghik->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,gcio,ahjk->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,acio,ghjk->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,gcik,ahjo->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,acik,ghjo->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,gcjk,ahio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,acjk,ghio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,chko,gaij->ai', _cse40, _cse34, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,ahjo,gcik->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,ahio,gcjk->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,ahik,gcjo->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bogh,ahjk,gcio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,ghjo,bcik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,ghio,bcjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,ghik,bcjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,ghjk,bcio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,gbji,hcko->ai', _cse39, _cse35, _cse6, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,gcjo,bhik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,bcjo,ghik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,gcio,bhjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,bcio,ghjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,gcik,bhjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,bcik,ghjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,gcjk,bhio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,bcjk,ghio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,chko,gbij->ai', _cse39, _cse35, _cse45, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,bhjo,gcik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,bhio,gcjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,bhik,gcjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aogh,bhjk,gcio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,agjo,hcik->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,agio,hcjk->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,agik,hcjo->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,agjk,hcio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,ahji,gcko->ai', _cse40, _cse34, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,acjo,hgik->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,hcjo,agik->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,acio,hgjk->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,hcio,agjk->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,acik,hgjo->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,hcik,agjo->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,acjk,hgio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,hcjk,agio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,cgko,ahij->ai', _cse40, _cse34, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,hgjo,acik->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,hgio,acjk->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,hgik,acjo->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,bohg,hgjk,acio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,bgjo,hcik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,bgio,hcjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,bgik,hcjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,bgjk,hcio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,bhji,gcko->ai', _cse39, _cse35, _cse6, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,bcjo,hgik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,hcjo,bgik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,bcio,hgjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,hcio,bgjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,bcik,hgjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,hcik,bgjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,bcjk,hgio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,hcjk,bgio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,cgko,bhij->ai', _cse39, _cse35, _cse45, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,hgjo,bcik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,hgio,bcjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,hgik,bcjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,aohg,hgjk,bcio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bgjo,ahik->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bgio,ahjk->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bgik,ahjo->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bgjk,ahio->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,baji,ghko->ai', _cse41, _cse46, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bhjo,agik->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ahjo,bgik->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bhio,agjk->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ahio,bgjk->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bhik,agjo->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ahik,bgjo->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,bhjk,agio->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,ahjk,bgio->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,hgko,baij->ai', _cse41, _cse46, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,agjo,bhik->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,agio,bhjk->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,agik,bhjo->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkbc,ocgh,agjk,bhio->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,gaip,cbko->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,gcip,abko->ai', _cse36, _cse37, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,gakp,cbio->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,gckp,abio->ai', _cse36, _cse37, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,gaki,cbpo->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,gcki,abpo->ai', _cse36, _cse37, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,gbpo,acik->ai', _cse36, _cse37, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,abpo,gcik->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,gbko,acip->ai', _cse36, _cse37, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,abko,gcip->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,gbio,ackp->ai', _cse36, _cse37, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,abio,gckp->ai', _cse36, _cse37, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cbpo,gaik->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cbko,gaip->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cbio,gakp->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,acip,gbko->ai', _cse36, _cse37, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,ackp,gbio->ai', _cse36, _cse37, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,acki,gbpo->ai', _cse36, _cse37, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,gaio,cbkp->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,gcio,abkp->ai', _cse36, _cse37, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,gako,cbip->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,gcko,abip->ai', _cse36, _cse37, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,gaki,cbop->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,gcki,abop->ai', _cse36, _cse37, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,gbop,acik->ai', _cse36, _cse37, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,abop,gcik->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,gbkp,acio->ai', _cse36, _cse37, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,abkp,gcio->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,gbip,acko->ai', _cse36, _cse37, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,abip,gcko->ai', _cse36, _cse37, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,cbop,gaik->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,cbkp,gaio->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,cbip,gako->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,acio,gbkp->ai', _cse36, _cse37, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,acko,gbip->ai', _cse36, _cse37, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opgj,acki,gbop->ai', _cse36, _cse37, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,gaip,cboj->ai', _cse38, _cse1, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,gcip,aboj->ai', _cse38, _cse1, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,gaop,cbij->ai', _cse38, _cse1, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,gcop,abij->ai', _cse38, _cse1, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,gaoi,cbpj->ai', _cse38, _cse1, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,gcoi,abpj->ai', _cse38, _cse1, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,gbpj,acio->ai', _cse38, _cse1, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,abpj,gcio->ai', _cse38, _cse1, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,gboj,acip->ai', _cse38, _cse1, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,aboj,gcip->ai', _cse38, _cse1, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,gbij,acop->ai', _cse38, _cse1, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,abij,gcop->ai', _cse38, _cse1, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,cbpj,gaio->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,cboj,gaip->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,cbij,gaop->ai', _cse38, _cse1, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,acip,gboj->ai', _cse38, _cse1, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,acop,gbij->ai', _cse38, _cse1, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogk,acoi,gbpj->ai', _cse38, _cse1, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,gakp,cboj->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,gckp,aboj->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,gaop,cbkj->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,gcop,abkj->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,gaok,cbpj->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,gcok,abpj->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,gbpj,acko->ai', _cse39, _cse17, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,abpj,gcko->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,gboj,ackp->ai', _cse39, _cse17, _cse26, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,aboj,gckp->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,gbkj,acop->ai', _cse39, _cse17, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,abkj,gcop->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,cbpj,gako->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,cboj,gakp->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,cbkj,gaop->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,ackp,gboj->ai', _cse39, _cse17, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,acop,gbkj->ai', _cse39, _cse17, _cse12, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogi,acok,gbpj->ai', _cse39, _cse17, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,ghio,cbkj->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,gcio,hbkj->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,ghko,cbij->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,gcko,hbij->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,ghki,cboj->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,gcki,hboj->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,gboj,hcik->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,hboj,gcik->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,gbkj,hcio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,hbkj,gcio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,gbij,hcko->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,hbij,gcko->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,cboj,ghik->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,cbkj,ghio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,cbij,ghko->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,hcio,gbkj->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,hcko,gbij->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,oagh,hcki,gboj->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,ghio,abkj->ai', _cse40, _cse19, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,gaio,hbkj->ai', _cse40, _cse19, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,ghko,abij->ai', _cse40, _cse19, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,gako,hbij->ai', _cse40, _cse19, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,ghki,aboj->ai', _cse40, _cse19, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,gaki,hboj->ai', _cse40, _cse19, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,gboj,haik->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,hboj,gaik->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,gbkj,haio->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,hbkj,gaio->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,gbij,hako->ai', _cse40, _cse19, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,hbij,gako->ai', _cse40, _cse19, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,aboj,ghik->ai', _cse40, _cse19, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,abkj,ghio->ai', _cse40, _cse19, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,abij,ghko->ai', _cse40, _cse19, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,haio,gbkj->ai', _cse40, _cse19, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,hako,gbij->ai', _cse40, _cse19, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,ocgh,haki,gboj->ai', _cse40, _cse19, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,gcio,ahkj->ai', _cse41, _cse42, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,gaio,chkj->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,gcko,ahij->ai', _cse41, _cse42, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,gako,chij->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,gcki,ahoj->ai', _cse41, _cse42, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,gaki,choj->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ghoj,caik->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,choj,gaik->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ghkj,caio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,chkj,gaio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ghij,cako->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,chij,gako->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ahoj,gcik->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ahkj,gcio->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ahij,gcko->ai', _cse41, _cse42, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,caio,ghkj->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,cako,ghij->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,caki,ghoj->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,chio,agkj->ai', _cse41, _cse42, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,caio,hgkj->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,chko,agij->ai', _cse41, _cse42, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,cako,hgij->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,chki,agoj->ai', _cse41, _cse42, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,caki,hgoj->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,cgoj,haik->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,hgoj,caik->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,cgkj,haio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,hgkj,caio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,cgij,hako->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,hgij,cako->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,agoj,chik->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,agkj,chio->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,agij,chko->ai', _cse41, _cse42, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,haio,cgkj->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,hako,cgij->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obhg,haki,cgoj->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cgko,abip->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cgio,abkp->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cgip,abko->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cgkp,abio->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,caki,gbpo->ai', _cse36, _cse43, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cbko,agip->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,abko,cgip->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cbio,agkp->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,abio,cgkp->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cbip,agko->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,abip,cgko->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,cbkp,agio->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,abkp,cgio->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,bgpo,caik->ai', _cse36, _cse43, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,agko,cbip->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,agio,cbkp->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,agip,cbko->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pogj,agkp,cbio->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,cgpo,abij->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,cgio,abpj->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,cgij,abpo->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,cgpj,abio->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,capi,gbjo->ai', _cse38, _cse22, _cse11, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,cbpo,agij->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,abpo,cgij->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,cbio,agpj->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,abio,cgpj->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,cbij,agpo->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,abij,cgpo->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,cbpj,agio->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,abpj,cgio->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,bgjo,caip->ai', _cse38, _cse22, _cse45, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,agpo,cbij->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,agio,cbpj->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,agij,cbpo->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,pokg,agpj,cbio->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,cgoj,abip->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,cgij,abop->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,cgip,aboj->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,cgop,abij->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,caoi,gbpj->ai', _cse38, _cse22, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,cboj,agip->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,aboj,cgip->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,cbij,agop->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,abij,cgop->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,cbip,agoj->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,abip,cgoj->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,cbop,agij->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,abop,cgij->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,bgpj,caio->ai', _cse38, _cse22, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,agoj,cbip->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,agij,cbop->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,agip,cboj->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opkg,agop,cbij->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poig,cgpo,abkj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poig,cgko,abpj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,cgkj,abpo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poig,cgpj,abko->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,capk,gbjo->ai', _cse39, _cse33, _cse7, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,cbpo,agkj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,abpo,cgkj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poig,cbko,agpj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poig,abko,cgpj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,cbkj,agpo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,abkj,cgpo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poig,cbpj,agko->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poig,abpj,cgko->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,bgjo,cakp->ai', _cse39, _cse33, _cse45, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,agpo,cbkj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,agko,cbpj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,poig,agkj,cbpo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,poig,agpj,cbko->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,cgoj,abkp->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opig,cgkj,abop->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,cgkp,aboj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opig,cgop,abkj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,caok,gbpj->ai', _cse39, _cse33, _cse7, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,cboj,agkp->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,aboj,cgkp->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opig,cbkj,agop->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opig,abkj,cgop->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,cbkp,agoj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,abkp,cgoj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opig,cbop,agkj->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opig,abop,cgkj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,bgpj,cako->ai', _cse39, _cse33, _cse45, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,agoj,cbkp->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,agkj,cbop->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,opig,agkp,cboj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,opig,agop,cbkj->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,ghko,cbij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,ghio,cbkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,ghij,cbko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,ghkj,cbio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,gcki,hbjo->ai', _cse39, _cse35, _cse6, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,gbko,chij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,cbko,ghij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,gbio,chkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,cbio,ghkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,gbij,chko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,cbij,ghko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,gbkj,chio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,cbkj,ghio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,bhjo,gcik->ai', _cse39, _cse35, _cse45, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,chko,gbij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,chio,gbkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,chij,gbko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aogh,chkj,gbio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,ghko,abij->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,ghio,abkj->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,ghij,abko->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,ghkj,abio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,gaki,hbjo->ai', _cse40, _cse34, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,gbko,ahij->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,abko,ghij->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,gbio,ahkj->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,abio,ghkj->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,gbij,ahko->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,abij,ghko->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,gbkj,ahio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,abkj,ghio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,bhjo,gaik->ai', _cse40, _cse34, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,ahko,gbij->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,ahio,gbkj->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,ahij,gbko->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cogh,ahkj,gbio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,cgko,hbij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,cgio,hbkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,cgij,hbko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,cgkj,hbio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,chki,gbjo->ai', _cse39, _cse35, _cse6, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,cbko,hgij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,hbko,cgij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,cbio,hgkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,hbio,cgkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,cbij,hgko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,hbij,cgko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,cbkj,hgio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,hbkj,cgio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,bgjo,chik->ai', _cse39, _cse35, _cse45, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,hgko,cbij->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,hgio,cbkj->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,hgij,cbko->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,aohg,hgkj,cbio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,agko,hbij->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,agio,hbkj->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,agij,hbko->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,agkj,hbio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,ahki,gbjo->ai', _cse40, _cse34, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,abko,hgij->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,hbko,agij->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,abio,hgkj->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,hbio,agkj->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,abij,hgko->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,hbij,agko->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,abkj,hgio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,hbkj,agio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,bgjo,ahik->ai', _cse40, _cse34, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,hgko,abij->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,hgio,abkj->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,hgij,abko->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,cohg,hgkj,abio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,agko,chij->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,agio,chkj->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,agij,chko->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,agkj,chio->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,acki,ghjo->ai', _cse41, _cse46, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ahko,cgij->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,chko,agij->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ahio,cgkj->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,chio,agkj->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ahij,cgko->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,chij,agko->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,ahkj,cgio->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,chkj,agio->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,hgjo,acik->ai', _cse41, _cse46, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,cgko,ahij->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,cgio,ahkj->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,cgij,ahko->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjcb,obgh,cgkj,ahio->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,gaip,cbjo->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,gcip,abjo->ai', _cse36, _cse37, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,gajp,cbio->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,gcjp,abio->ai', _cse36, _cse37, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,gaji,cbpo->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,gcji,abpo->ai', _cse36, _cse37, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,gbpo,acij->ai', _cse36, _cse37, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,abpo,gcij->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,gbjo,acip->ai', _cse36, _cse37, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,abjo,gcip->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,gbio,acjp->ai', _cse36, _cse37, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,abio,gcjp->ai', _cse36, _cse37, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cbpo,gaij->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cbjo,gaip->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cbio,gajp->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,acip,gbjo->ai', _cse36, _cse37, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,acjp,gbio->ai', _cse36, _cse37, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,acji,gbpo->ai', _cse36, _cse37, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,gaio,cbjp->ai', _cse36, _cse37, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,gcio,abjp->ai', _cse36, _cse37, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,gajo,cbip->ai', _cse36, _cse37, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,gcjo,abip->ai', _cse36, _cse37, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,gaji,cbop->ai', _cse36, _cse37, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,gcji,abop->ai', _cse36, _cse37, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,gbop,acij->ai', _cse36, _cse37, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,abop,gcij->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,gbjp,acio->ai', _cse36, _cse37, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,abjp,gcio->ai', _cse36, _cse37, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,gbip,acjo->ai', _cse36, _cse37, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,abip,gcjo->ai', _cse36, _cse37, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,cbop,gaij->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,cbjp,gaio->ai', _cse36, _cse37, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,cbip,gajo->ai', _cse36, _cse37, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,acio,gbjp->ai', _cse36, _cse37, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,acjo,gbip->ai', _cse36, _cse37, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opgk,acji,gbop->ai', _cse36, _cse37, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,gaip,cbok->ai', _cse38, _cse1, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,gcip,abok->ai', _cse38, _cse1, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,gaop,cbik->ai', _cse38, _cse1, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,gcop,abik->ai', _cse38, _cse1, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,gaoi,cbpk->ai', _cse38, _cse1, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,gcoi,abpk->ai', _cse38, _cse1, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,gbpk,acio->ai', _cse38, _cse1, _cse26, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,abpk,gcio->ai', _cse38, _cse1, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,gbok,acip->ai', _cse38, _cse1, _cse26, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,abok,gcip->ai', _cse38, _cse1, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,gbik,acop->ai', _cse38, _cse1, _cse29, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,abik,gcop->ai', _cse38, _cse1, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,cbpk,gaio->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,cbok,gaip->ai', _cse38, _cse1, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,cbik,gaop->ai', _cse38, _cse1, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,acip,gbok->ai', _cse38, _cse1, _cse13, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,acop,gbik->ai', _cse38, _cse1, _cse12, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogj,acoi,gbpk->ai', _cse38, _cse1, _cse14, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,gajp,cbok->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,gcjp,abok->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,gaop,cbjk->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,gcop,abjk->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,gaoj,cbpk->ai', _cse39, _cse17, _cse7, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,gcoj,abpk->ai', _cse39, _cse17, _cse2, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,gbpk,acjo->ai', _cse39, _cse17, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,abpk,gcjo->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,gbok,acjp->ai', _cse39, _cse17, _cse26, _cse30, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,abok,gcjp->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,gbjk,acop->ai', _cse39, _cse17, _cse26, _cse30, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,abjk,gcop->ai', _cse39, _cse17, _cse25, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,cbpk,gajo->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,cbok,gajp->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,cbjk,gaop->ai', _cse39, _cse17, _cse26, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,acjp,gbok->ai', _cse39, _cse17, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,acop,gbjk->ai', _cse39, _cse17, _cse12, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogi,acoj,gbpk->ai', _cse39, _cse17, _cse12, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,ghio,cbjk->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,gcio,hbjk->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,ghjo,cbik->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,gcjo,hbik->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,ghji,cbok->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,gcji,hbok->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,gbok,hcij->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,hbok,gcij->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,gbjk,hcio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,hbjk,gcio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,gbik,hcjo->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,hbik,gcjo->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,cbok,ghij->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,cbjk,ghio->ai', _cse39, _cse21, _cse26, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,cbik,ghjo->ai', _cse39, _cse21, _cse29, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,hcio,gbjk->ai', _cse39, _cse21, _cse4, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,hcjo,gbik->ai', _cse39, _cse21, _cse2, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,oagh,hcji,gbok->ai', _cse39, _cse21, _cse6, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,ghio,abjk->ai', _cse40, _cse19, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,gaio,hbjk->ai', _cse40, _cse19, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,ghjo,abik->ai', _cse40, _cse19, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,gajo,hbik->ai', _cse40, _cse19, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,ghji,abok->ai', _cse40, _cse19, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,gaji,hbok->ai', _cse40, _cse19, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,gbok,haij->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,hbok,gaij->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,gbjk,haio->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,hbjk,gaio->ai', _cse40, _cse19, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,gbik,hajo->ai', _cse40, _cse19, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,hbik,gajo->ai', _cse40, _cse19, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,abok,ghij->ai', _cse40, _cse19, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,abjk,ghio->ai', _cse40, _cse19, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,abik,ghjo->ai', _cse40, _cse19, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,haio,gbjk->ai', _cse40, _cse19, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,hajo,gbik->ai', _cse40, _cse19, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,ocgh,haji,gbok->ai', _cse40, _cse19, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,gcio,ahjk->ai', _cse41, _cse42, _cse4, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,gaio,chjk->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,gcjo,ahik->ai', _cse41, _cse42, _cse2, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,gajo,chik->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,gcji,ahok->ai', _cse41, _cse42, _cse6, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,gaji,chok->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ghok,caij->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,chok,gaij->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ghjk,caio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,chjk,gaio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ghik,cajo->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,chik,gajo->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ahok,gcij->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ahjk,gcio->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ahik,gcjo->ai', _cse41, _cse42, _cse28, _cse10, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,caio,ghjk->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,cajo,ghik->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,caji,ghok->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,chio,agjk->ai', _cse41, _cse42, _cse4, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,caio,hgjk->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,chjo,agik->ai', _cse41, _cse42, _cse2, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,cajo,hgik->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,chji,agok->ai', _cse41, _cse42, _cse6, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,caji,hgok->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,cgok,haij->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,hgok,caij->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,cgjk,haio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,hgjk,caio->ai', _cse41, _cse42, _cse26, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,cgik,hajo->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,hgik,cajo->ai', _cse41, _cse42, _cse29, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,agok,chij->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,agjk,chio->ai', _cse41, _cse42, _cse25, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,agik,chjo->ai', _cse41, _cse42, _cse28, _cse10, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,haio,cgjk->ai', _cse41, _cse42, _cse9, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,hajo,cgik->ai', _cse41, _cse42, _cse7, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obhg,haji,cgok->ai', _cse41, _cse42, _cse11, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cgjo,abip->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cgio,abjp->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cgip,abjo->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cgjp,abio->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,caji,gbpo->ai', _cse36, _cse43, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cbjo,agip->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,abjo,cgip->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cbio,agjp->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,abio,cgjp->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cbip,agjo->ai', _cse36, _cse43, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,abip,cgjo->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,cbjp,agio->ai', _cse36, _cse43, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,abjp,cgio->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,bgpo,caij->ai', _cse36, _cse43, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,agjo,cbip->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,agio,cbjp->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,agip,cbjo->ai', _cse36, _cse43, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pogk,agjp,cbio->ai', _cse36, _cse43, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,cgpo,abik->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,cgio,abpk->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,cgik,abpo->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,cgpk,abio->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,capi,gbko->ai', _cse38, _cse22, _cse11, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,cbpo,agik->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,abpo,cgik->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,cbio,agpk->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,abio,cgpk->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,cbik,agpo->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,abik,cgpo->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,cbpk,agio->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,abpk,cgio->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,bgko,caip->ai', _cse38, _cse22, _cse45, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,agpo,cbik->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,agio,cbpk->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,agik,cbpo->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,pojg,agpk,cbio->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,cgok,abip->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,cgik,abop->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,cgip,abok->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,cgop,abik->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,caoi,gbpk->ai', _cse38, _cse22, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,cbok,agip->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,abok,cgip->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,cbik,agop->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,abik,cgop->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,cbip,agok->ai', _cse38, _cse22, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,abip,cgok->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,cbop,agik->ai', _cse38, _cse22, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,abop,cgik->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,bgpk,caio->ai', _cse38, _cse22, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,agok,cbip->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,agik,cbop->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,agip,cbok->ai', _cse38, _cse22, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opjg,agop,cbik->ai', _cse38, _cse22, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poig,cgpo,abjk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poig,cgjo,abpk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,cgjk,abpo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poig,cgpk,abjo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,capj,gbko->ai', _cse39, _cse33, _cse7, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,cbpo,agjk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,abpo,cgjk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poig,cbjo,agpk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poig,abjo,cgpk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,cbjk,agpo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,abjk,cgpo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poig,cbpk,agjo->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poig,abpk,cgjo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,bgko,cajp->ai', _cse39, _cse33, _cse45, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,agpo,cbjk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,agjo,cbpk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,poig,agjk,cbpo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,poig,agpk,cbjo->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,cgok,abjp->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opig,cgjk,abop->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,cgjp,abok->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opig,cgop,abjk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,caoj,gbpk->ai', _cse39, _cse33, _cse7, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,cbok,agjp->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,abok,cgjp->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opig,cbjk,agop->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opig,abjk,cgop->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,cbjp,agok->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,abjp,cgok->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opig,cbop,agjk->ai', _cse39, _cse33, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opig,abop,cgjk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,bgpk,cajo->ai', _cse39, _cse33, _cse45, _cse5, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,agok,cbjp->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,agjk,cbop->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,opig,agjp,cbok->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,opig,agop,cbjk->ai', _cse39, _cse33, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,ghjo,cbik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,ghio,cbjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,ghik,cbjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,ghjk,cbio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,gcji,hbko->ai', _cse39, _cse35, _cse6, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,gbjo,chik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,cbjo,ghik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,gbio,chjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,cbio,ghjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,gbik,chjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,cbik,ghjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,gbjk,chio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,cbjk,ghio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,bhko,gcij->ai', _cse39, _cse35, _cse45, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,chjo,gbik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,chio,gbjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,chik,gbjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aogh,chjk,gbio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,ghjo,abik->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,ghio,abjk->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,ghik,abjo->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,ghjk,abio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,gaji,hbko->ai', _cse40, _cse34, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,gbjo,ahik->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,abjo,ghik->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,gbio,ahjk->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,abio,ghjk->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,gbik,ahjo->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,abik,ghjo->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,gbjk,ahio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,abjk,ghio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,bhko,gaij->ai', _cse40, _cse34, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,ahjo,gbik->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,ahio,gbjk->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,ahik,gbjo->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cogh,ahjk,gbio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,cgjo,hbik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,cgio,hbjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,cgik,hbjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,cgjk,hbio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,chji,gbko->ai', _cse39, _cse35, _cse6, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,cbjo,hgik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,hbjo,cgik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,cbio,hgjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,hbio,cgjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,cbik,hgjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,hbik,cgjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,cbjk,hgio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,hbjk,cgio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,bgko,chij->ai', _cse39, _cse35, _cse45, _cse8, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,hgjo,cbik->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,hgio,cbjk->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,hgik,cbjo->ai', _cse39, _cse35, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,aohg,hgjk,cbio->ai', _cse39, _cse35, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,agjo,hbik->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,agio,hbjk->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,agik,hbjo->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,agjk,hbio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,ahji,gbko->ai', _cse40, _cse34, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,abjo,hgik->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,hbjo,agik->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,abio,hgjk->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,hbio,agjk->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,abik,hgjo->ai', _cse40, _cse34, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,hbik,agjo->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,abjk,hgio->ai', _cse40, _cse34, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,hbjk,agio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,bgko,ahij->ai', _cse40, _cse34, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,hgjo,abik->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,hgio,abjk->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,hgik,abjo->ai', _cse40, _cse34, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,cohg,hgjk,abio->ai', _cse40, _cse34, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,agjo,chik->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,agio,chjk->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,agik,chjo->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,agjk,chio->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,acji,ghko->ai', _cse41, _cse46, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ahjo,cgik->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,chjo,agik->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ahio,cgjk->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,chio,agjk->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ahik,cgjo->ai', _cse41, _cse46, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,chik,agjo->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,ahjk,cgio->ai', _cse41, _cse46, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,chjk,agio->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,hgko,acij->ai', _cse41, _cse46, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,cgjo,ahik->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,cgio,ahjk->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,cgik,ahjo->ai', _cse41, _cse46, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('jkcb,obgh,cgjk,ahio->ai', _cse41, _cse46, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcpo,abik->ai', _cse47, _cse37, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcio,abpk->ai', _cse47, _cse37, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcik,abpo->ai', _cse47, _cse37, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcpk,abio->ai', _cse47, _cse37, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gapi,cbko->ai', _cse47, _cse37, _cse11, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbpo,acik->ai', _cse47, _cse37, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,abpo,gcik->ai', _cse47, _cse37, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbio,acpk->ai', _cse47, _cse37, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,abio,gcpk->ai', _cse47, _cse37, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbik,acpo->ai', _cse47, _cse37, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,abik,gcpo->ai', _cse47, _cse37, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbpk,acio->ai', _cse47, _cse37, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,abpk,gcio->ai', _cse47, _cse37, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcko,gaip->ai', _cse47, _cse37, _cse45, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acpo,gbik->ai', _cse47, _cse37, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acio,gbpk->ai', _cse47, _cse37, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acik,gbpo->ai', _cse47, _cse37, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acpk,gbio->ai', _cse47, _cse37, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcpo,abij->ai', _cse48, _cse37, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcio,abpj->ai', _cse48, _cse37, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcij,abpo->ai', _cse48, _cse37, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcpj,abio->ai', _cse48, _cse37, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gapi,cbjo->ai', _cse48, _cse37, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbpo,acij->ai', _cse48, _cse37, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,abpo,gcij->ai', _cse48, _cse37, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbio,acpj->ai', _cse48, _cse37, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,abio,gcpj->ai', _cse48, _cse37, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbij,acpo->ai', _cse48, _cse37, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,abij,gcpo->ai', _cse48, _cse37, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbpj,acio->ai', _cse48, _cse37, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,abpj,gcio->ai', _cse48, _cse37, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcjo,gaip->ai', _cse48, _cse37, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acpo,gbij->ai', _cse48, _cse37, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acio,gbpj->ai', _cse48, _cse37, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acij,gbpo->ai', _cse48, _cse37, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acpj,gbio->ai', _cse48, _cse37, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gcok,abip->ai', _cse47, _cse37, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gcik,abop->ai', _cse47, _cse37, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gcip,abok->ai', _cse47, _cse37, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gcop,abik->ai', _cse47, _cse37, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gaoi,cbpk->ai', _cse47, _cse37, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gbok,acip->ai', _cse47, _cse37, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,abok,gcip->ai', _cse47, _cse37, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gbik,acop->ai', _cse47, _cse37, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,abik,gcop->ai', _cse47, _cse37, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gbip,acok->ai', _cse47, _cse37, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,abip,gcok->ai', _cse47, _cse37, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,gbop,acik->ai', _cse47, _cse37, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,abop,gcik->ai', _cse47, _cse37, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,bcpk,gaio->ai', _cse47, _cse37, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,acok,gbip->ai', _cse47, _cse37, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,acik,gbop->ai', _cse47, _cse37, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,acip,gbok->ai', _cse47, _cse37, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgj,acop,gbik->ai', _cse47, _cse37, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,gcoj,abip->ai', _cse48, _cse37, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,gcij,abop->ai', _cse48, _cse37, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,gcip,aboj->ai', _cse48, _cse37, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,gcop,abij->ai', _cse48, _cse37, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,gaoi,cbpj->ai', _cse48, _cse37, _cse11, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,gboj,acip->ai', _cse48, _cse37, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,aboj,gcip->ai', _cse48, _cse37, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,gbij,acop->ai', _cse48, _cse37, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,abij,gcop->ai', _cse48, _cse37, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,gbip,acoj->ai', _cse48, _cse37, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,abip,gcoj->ai', _cse48, _cse37, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,gbop,acij->ai', _cse48, _cse37, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,abop,gcij->ai', _cse48, _cse37, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,bcpj,gaio->ai', _cse48, _cse37, _cse45, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,acoj,gbip->ai', _cse48, _cse37, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,acij,gbop->ai', _cse48, _cse37, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,acip,gboj->ai', _cse48, _cse37, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opgk,acop,gbij->ai', _cse48, _cse37, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcoj,abpk->ai', _cse49, _cse17, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcpj,abok->ai', _cse49, _cse17, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcpk,aboj->ai', _cse49, _cse17, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gcok,abpj->ai', _cse49, _cse17, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gaop,cbkj->ai', _cse49, _cse17, _cse7, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gboj,acpk->ai', _cse49, _cse17, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,aboj,gcpk->ai', _cse49, _cse17, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbpj,acok->ai', _cse49, _cse17, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,abpj,gcok->ai', _cse49, _cse17, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbpk,acoj->ai', _cse49, _cse17, _cse26, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,abpk,gcoj->ai', _cse49, _cse17, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,gbok,acpj->ai', _cse49, _cse17, _cse26, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,abok,gcpj->ai', _cse49, _cse17, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,bckj,gapo->ai', _cse49, _cse17, _cse45, _cse5, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acoj,gbpk->ai', _cse49, _cse17, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acpj,gbok->ai', _cse49, _cse17, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acpk,gboj->ai', _cse49, _cse17, _cse25, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogi,acok,gbpj->ai', _cse49, _cse17, _cse25, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcok,hbij->ai', _cse49, _cse21, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcik,hboj->ai', _cse49, _cse21, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcij,hbok->ai', _cse49, _cse21, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gcoj,hbik->ai', _cse49, _cse21, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,ghoi,cbjk->ai', _cse49, _cse21, _cse6, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbok,hcij->ai', _cse49, _cse21, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbok,gcij->ai', _cse49, _cse21, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbik,hcoj->ai', _cse49, _cse21, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbik,gcoj->ai', _cse49, _cse21, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gbij,hcok->ai', _cse49, _cse21, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hbij,gcok->ai', _cse49, _cse21, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,gboj,hcik->ai', _cse49, _cse21, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hboj,gcik->ai', _cse49, _cse21, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,bcjk,ghio->ai', _cse49, _cse21, _cse45, _cse8, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcok,gbij->ai', _cse49, _cse21, _cse26, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcik,gboj->ai', _cse49, _cse21, _cse29, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcij,gbok->ai', _cse49, _cse21, _cse29, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,oagh,hcoj,gbik->ai', _cse49, _cse21, _cse26, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghok,abij->ai', _cse50, _cse42, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghik,aboj->ai', _cse50, _cse42, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghij,abok->ai', _cse50, _cse42, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ghoj,abik->ai', _cse50, _cse42, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gaoi,hbjk->ai', _cse50, _cse42, _cse11, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbok,ahij->ai', _cse50, _cse42, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abok,ghij->ai', _cse50, _cse42, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbik,ahoj->ai', _cse50, _cse42, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abik,ghoj->ai', _cse50, _cse42, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gbij,ahok->ai', _cse50, _cse42, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abij,ghok->ai', _cse50, _cse42, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,gboj,ahik->ai', _cse50, _cse42, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,aboj,ghik->ai', _cse50, _cse42, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bhjk,gaio->ai', _cse50, _cse42, _cse45, _cse3, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahok,gbij->ai', _cse50, _cse42, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahik,gboj->ai', _cse50, _cse42, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahij,gbok->ai', _cse50, _cse42, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahoj,gbik->ai', _cse50, _cse42, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,agok,hbij->ai', _cse50, _cse42, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,agik,hboj->ai', _cse50, _cse42, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,agij,hbok->ai', _cse50, _cse42, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,agoj,hbik->ai', _cse50, _cse42, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,ahoi,gbjk->ai', _cse50, _cse42, _cse14, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,abok,hgij->ai', _cse50, _cse42, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hbok,agij->ai', _cse50, _cse42, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,abik,hgoj->ai', _cse50, _cse42, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hbik,agoj->ai', _cse50, _cse42, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,abij,hgok->ai', _cse50, _cse42, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hbij,agok->ai', _cse50, _cse42, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,aboj,hgik->ai', _cse50, _cse42, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hboj,agik->ai', _cse50, _cse42, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,bgjk,ahio->ai', _cse50, _cse42, _cse45, _cse27, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hgok,abij->ai', _cse50, _cse42, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hgik,aboj->ai', _cse50, _cse42, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hgij,abok->ai', _cse50, _cse42, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ochg,hgoj,abik->ai', _cse50, _cse42, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghok,acij->ai', _cse51, _cse42, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghik,acoj->ai', _cse51, _cse42, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghij,acok->ai', _cse51, _cse42, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ghoj,acik->ai', _cse51, _cse42, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gaoi,hcjk->ai', _cse51, _cse42, _cse11, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcok,ahij->ai', _cse51, _cse42, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acok,ghij->ai', _cse51, _cse42, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcik,ahoj->ai', _cse51, _cse42, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acik,ghoj->ai', _cse51, _cse42, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcij,ahok->ai', _cse51, _cse42, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acij,ghok->ai', _cse51, _cse42, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,gcoj,ahik->ai', _cse51, _cse42, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acoj,ghik->ai', _cse51, _cse42, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,chjk,gaio->ai', _cse51, _cse42, _cse45, _cse3, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ahok,gcij->ai', _cse51, _cse42, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ahik,gcoj->ai', _cse51, _cse42, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ahij,gcok->ai', _cse51, _cse42, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ahoj,gcik->ai', _cse51, _cse42, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,agok,hcij->ai', _cse51, _cse42, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,agik,hcoj->ai', _cse51, _cse42, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,agij,hcok->ai', _cse51, _cse42, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,agoj,hcik->ai', _cse51, _cse42, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,ahoi,gcjk->ai', _cse51, _cse42, _cse14, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,acok,hgij->ai', _cse51, _cse42, _cse25, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,hcok,agij->ai', _cse51, _cse42, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,acik,hgoj->ai', _cse51, _cse42, _cse28, _cse23, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,hcik,agoj->ai', _cse51, _cse42, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,acij,hgok->ai', _cse51, _cse42, _cse28, _cse23, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,hcij,agok->ai', _cse51, _cse42, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,acoj,hgik->ai', _cse51, _cse42, _cse25, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,hcoj,agik->ai', _cse51, _cse42, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,cgjk,ahio->ai', _cse51, _cse42, _cse45, _cse27, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,hgok,acij->ai', _cse51, _cse42, _cse26, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,hgik,acoj->ai', _cse51, _cse42, _cse29, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,hgij,acok->ai', _cse51, _cse42, _cse29, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obhg,hgoj,acik->ai', _cse51, _cse42, _cse26, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,agio,cbkp->ai', _cse47, _cse43, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acio,gbkp->ai', _cse47, _cse43, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,agip,cbko->ai', _cse47, _cse43, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acip,gbko->ai', _cse47, _cse43, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,agik,cbpo->ai', _cse47, _cse43, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,acik,gbpo->ai', _cse47, _cse43, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbpo,acik->ai', _cse47, _cse43, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbko,acip->ai', _cse47, _cse43, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,abio,gckp->ai', _cse47, _cse43, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,abip,gcko->ai', _cse47, _cse43, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gbkp,acio->ai', _cse47, _cse43, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,abik,gcpo->ai', _cse47, _cse43, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcpo,abik->ai', _cse47, _cse43, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcpo,agik->ai', _cse47, _cse43, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gcko,abip->ai', _cse47, _cse43, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bcko,agip->ai', _cse47, _cse43, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,gckp,abio->ai', _cse47, _cse43, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogj,bckp,agio->ai', _cse47, _cse43, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,agio,cbjp->ai', _cse48, _cse43, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acio,gbjp->ai', _cse48, _cse43, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,agip,cbjo->ai', _cse48, _cse43, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acip,gbjo->ai', _cse48, _cse43, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,agij,cbpo->ai', _cse48, _cse43, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,acij,gbpo->ai', _cse48, _cse43, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbpo,acij->ai', _cse48, _cse43, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbjo,acip->ai', _cse48, _cse43, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,abio,gcjp->ai', _cse48, _cse43, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,abip,gcjo->ai', _cse48, _cse43, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gbjp,acio->ai', _cse48, _cse43, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,abij,gcpo->ai', _cse48, _cse43, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcpo,abij->ai', _cse48, _cse43, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcpo,agij->ai', _cse48, _cse43, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcjo,abip->ai', _cse48, _cse43, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcjo,agip->ai', _cse48, _cse43, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,gcjp,abio->ai', _cse48, _cse43, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,pogk,bcjp,agio->ai', _cse48, _cse43, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agpo,cbjk->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,acpo,gbjk->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agpk,cbjo->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,acpk,gbjo->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,agpj,cbko->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,acpj,gbko->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,gbko,acpj->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,gbjo,acpk->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,abpo,gcjk->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,abpk,gcjo->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,gbjk,acpo->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,abpj,gcko->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,gcko,abpj->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bcko,agpj->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,gcjo,abpk->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bcjo,agpk->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,gcjk,abpo->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,poig,bcjk,agpo->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agok,cbjp->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,acok,gbjp->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agop,cbjk->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,acop,gbjk->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,agoj,cbpk->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,acoj,gbpk->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,gbpk,acoj->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,gbjk,acop->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,abok,gcjp->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,abop,gcjk->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,gbjp,acok->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,aboj,gcpk->ai', _cse49, _cse33, _cse25, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,gcpk,aboj->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bcpk,agoj->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,gcjk,abop->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bcjk,agop->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,gcjp,abok->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,opig,bcjp,agok->ai', _cse49, _cse33, _cse45, _cse31, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghio,cbkj->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gcio,hbkj->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghij,cbko->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gcij,hbko->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,ghik,cbjo->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gcik,hbjo->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,hbjo,gcik->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,hbko,gcij->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gbio,hckj->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gbij,hcko->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,hbkj,gcio->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,gbik,hcjo->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,hcjo,gbik->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bcjo,ghik->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,hcko,gbij->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bcko,ghij->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,hckj,gbio->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aogh,bckj,ghio->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgio,cbkj->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hcio,gbkj->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgij,cbko->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hcij,gbko->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hgik,cbjo->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hcik,gbjo->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,gbjo,hcik->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,gbko,hcij->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hbio,gckj->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hbij,gcko->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,gbkj,hcio->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,hbik,gcjo->ai', _cse49, _cse35, _cse29, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,gcjo,hbik->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bcjo,hgik->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,gcko,hbij->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bcko,hgij->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,gckj,hbio->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,aohg,bckj,hgio->ai', _cse49, _cse35, _cse45, _cse24, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahio,gbkj->ai', _cse50, _cse46, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,agio,hbkj->ai', _cse50, _cse46, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahij,gbko->ai', _cse50, _cse46, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,agij,hbko->ai', _cse50, _cse46, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,ahik,gbjo->ai', _cse50, _cse46, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,agik,hbjo->ai', _cse50, _cse46, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hbjo,agik->ai', _cse50, _cse46, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hbko,agij->ai', _cse50, _cse46, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abio,hgkj->ai', _cse50, _cse46, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abij,hgko->ai', _cse50, _cse46, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hbkj,agio->ai', _cse50, _cse46, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,abik,hgjo->ai', _cse50, _cse46, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hgjo,abik->ai', _cse50, _cse46, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bgjo,ahik->ai', _cse50, _cse46, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hgko,abij->ai', _cse50, _cse46, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bgko,ahij->ai', _cse50, _cse46, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,hgkj,abio->ai', _cse50, _cse46, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,ocgh,bgkj,ahio->ai', _cse50, _cse46, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ahio,gckj->ai', _cse51, _cse46, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,agio,hckj->ai', _cse51, _cse46, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ahij,gcko->ai', _cse51, _cse46, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,agij,hcko->ai', _cse51, _cse46, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,ahik,gcjo->ai', _cse51, _cse46, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,agik,hcjo->ai', _cse51, _cse46, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hcjo,agik->ai', _cse51, _cse46, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hcko,agij->ai', _cse51, _cse46, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acio,hgkj->ai', _cse51, _cse46, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acij,hgko->ai', _cse51, _cse46, _cse28, _cse44, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hckj,agio->ai', _cse51, _cse46, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,acik,hgjo->ai', _cse51, _cse46, _cse28, _cse44, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hgjo,acik->ai', _cse51, _cse46, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,cgjo,ahik->ai', _cse51, _cse46, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hgko,acij->ai', _cse51, _cse46, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,cgko,ahij->ai', _cse51, _cse46, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,hgkj,acio->ai', _cse51, _cse46, _cse45, _cse32, optimize=True)
            _iter -= 0.125 * _tmp
            _tmp = einsum('kjbc,obgh,cgkj,ahio->ai', _cse51, _cse46, _cse45, _cse32, optimize=True)
            _iter += 0.125 * _tmp
            out += (-sigma[_tk_t3]) * (-sigma[_tk_t4_d1]) * _iter
    return out

