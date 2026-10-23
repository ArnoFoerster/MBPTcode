# GENERATED CODE -- Laplace-fused T3^(2)_aabaab/T3^(2)_abbabb
# cross-spin contribution to t1_3_aa_numerator/m3_ov_12_restricted
# (the only two live consumers; see
# generate_mp3_t3_laplace_restricted.py's module docstring for scope).
# Do not edit by hand.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum
from src.Base.utils.grids.grids import minimax_time_grid


def t1_3_aa_t3crossspin_laplace(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    t1_3_aa_t3cs = np.zeros((nv, no))
    for _tk in range(ntau):
        _t = tau[_tk]
        _w = -sigma[_tk]
        Oe = np.exp(ei * _t)
        Ve = np.exp(-ea * _t)
        _iter = np.zeros((nv, no))
        _cse0 = ((((g_abab[o, o, v, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse1 = g_abab[v, o, o, o]
        _cse2 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse3 = (g_abab[v, o, o, o] * Ve[:, None, None, None])
        _cse4 = (t2_abab * Oe[None, None, :, None])
        _cse5 = (g_abab[v, o, o, o] * Oe[None, None, :, None])
        _cse6 = (t2_abab * Ve[:, None, None, None])
        _cse7 = ((g_abab[v, o, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse8 = t2_abab
        _cse9 = (g_aaaa[o, v, o, o] * Oe[None, None, :, None])
        _cse10 = ((g_aaaa[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse11 = g_abab[o, v, o, o]
        _cse12 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse13 = (g_abab[o, v, o, o] * Oe[None, None, :, None])
        _cse14 = (t2_aaaa * Ve[None, :, None, None])
        _cse15 = g_abab[v, v, v, o]
        _cse16 = (g_aaaa[v, v, v, o] * Ve[None, :, None, None])
        _cse17 = g_abab[v, v, o, v]
        _cse18 = ((g_aaaa[v, v, v, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse19 = (g_abab[v, v, o, v] * Oe[None, None, :, None])
        _cse20 = (g_abab[v, v, v, o] * Ve[:, None, None, None])
        _cse21 = (t2_aaaa * Oe[None, None, :, None])
        _cse22 = (g_abab[v, v, o, v] * Ve[:, None, None, None])
        _cse23 = ((g_abab[v, v, o, v] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse24 = ((t2_aaaa * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse25 = (t2_aaaa * Ve[:, None, None, None])
        _cse26 = (g_aaaa[v, v, v, o] * Ve[:, None, None, None])
        _cse27 = ((g_aaaa[v, v, v, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse28 = ((((g_bbbb[o, o, v, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse29 = g_bbbb[o, v, o, o]
        _cse30 = t2_bbbb
        _cse31 = g_bbbb[v, v, v, o]
        _tmp = einsum('kjbc,bokj,acio->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,aokj,bcio->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,boij,acko->ai', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,aoij,bcko->ai', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,obik,acoj->ai', _cse0, _cse9, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,oaik,bcoj->ai', _cse0, _cse10, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,baio->ai', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,bako->ai', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bcgj,gaik->ai', _cse0, _cse15, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bagk,gcij->ai', _cse0, _cse16, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,bckg,agij->ai', _cse0, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bagi,gckj->ai', _cse0, _cse18, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,bcig,agkj->ai', _cse0, _cse19, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acgj,gbik->ai', _cse0, _cse20, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ackg,bgij->ai', _cse0, _cse22, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acig,bgkj->ai', _cse0, _cse23, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bojk,acio->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,aojk,bcio->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,boik,acjo->ai', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,aoik,bcjo->ai', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,obij,acok->ai', _cse0, _cse9, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,oaij,bcok->ai', _cse0, _cse10, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,ocjk,baio->ai', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,ocik,bajo->ai', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bcgk,gaij->ai', _cse0, _cse15, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bagj,gcik->ai', _cse0, _cse16, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,bcjg,agik->ai', _cse0, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bagi,gcjk->ai', _cse0, _cse18, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,bcig,agjk->ai', _cse0, _cse19, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,acgk,gbij->ai', _cse0, _cse20, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkbc,acjg,bgik->ai', _cse0, _cse22, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkbc,acig,bgjk->ai', _cse0, _cse23, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,aokj,cbio->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cokj,abio->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,aoij,cbko->ai', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,coij,abko->ai', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,oaik,cboj->ai', _cse0, _cse10, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,ocik,aboj->ai', _cse0, _cse9, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,obkj,acio->ai', _cse0, _cse11, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,obij,acko->ai', _cse0, _cse13, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,abgj,gcik->ai', _cse0, _cse20, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,acgk,gbij->ai', _cse0, _cse26, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,abkg,cgij->ai', _cse0, _cse22, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,acgi,gbkj->ai', _cse0, _cse27, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,abig,cgkj->ai', _cse0, _cse23, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cbgj,gaik->ai', _cse0, _cse15, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjcb,cbkg,agij->ai', _cse0, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjcb,cbig,agkj->ai', _cse0, _cse19, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,aojk,cbio->ai', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cojk,abio->ai', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,aoik,cbjo->ai', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,coik,abjo->ai', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,oaij,cbok->ai', _cse0, _cse10, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,ocij,abok->ai', _cse0, _cse9, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,objk,acio->ai', _cse0, _cse11, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,obik,acjo->ai', _cse0, _cse13, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,abgk,gcij->ai', _cse0, _cse20, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,acgj,gbik->ai', _cse0, _cse26, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,abjg,cgik->ai', _cse0, _cse22, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,acgi,gbjk->ai', _cse0, _cse27, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,abig,cgjk->ai', _cse0, _cse23, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbgk,gaij->ai', _cse0, _cse15, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jkcb,cbjg,agik->ai', _cse0, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jkcb,cbig,agjk->ai', _cse0, _cse19, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ockj,abio->ai', _cse28, _cse29, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,aoij,cbko->ai', _cse28, _cse7, _cse30, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,ocij,abok->ai', _cse28, _cse13, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,aoik,cbjo->ai', _cse28, _cse7, _cse30, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,ocik,aboj->ai', _cse28, _cse13, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obkj,acio->ai', _cse28, _cse29, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obij,acok->ai', _cse28, _cse13, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,obik,acoj->ai', _cse28, _cse13, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acgj,gbik->ai', _cse28, _cse20, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,abgj,gcik->ai', _cse28, _cse20, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,acgk,gbij->ai', _cse28, _cse20, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,abgk,gcij->ai', _cse28, _cse20, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,acig,gbkj->ai', _cse28, _cse23, _cse30, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,abig,gckj->ai', _cse28, _cse23, _cse30, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kjbc,cbgj,agik->ai', _cse28, _cse31, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kjbc,cbgk,agij->ai', _cse28, _cse31, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        t1_3_aa_t3cs += _w * _iter
    return t1_3_aa_t3cs


def m3_ov_a_t3crossspin_laplace(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, l_2_1_aaaa, l_2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    m3_ov_t3cs = np.zeros((no, nv))
    for _tk in range(ntau):
        _t = tau[_tk]
        _w = -sigma[_tk]
        Oe = np.exp(ei * _t)
        Ve = np.exp(-ea * _t)
        _iter = np.zeros((no, nv))
        _cse0 = ((((l2_abab * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse1 = ((g_abab[v, o, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse2 = t2_abab
        _cse3 = (g_abab[v, o, o, o] * Oe[None, None, :, None])
        _cse4 = (t2_abab * Ve[:, None, None, None])
        _cse5 = (g_abab[v, o, o, o] * Ve[:, None, None, None])
        _cse6 = (t2_abab * Oe[None, None, :, None])
        _cse7 = g_abab[v, o, o, o]
        _cse8 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse9 = ((g_aaaa[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse10 = (g_aaaa[o, v, o, o] * Oe[None, None, None, :])
        _cse11 = (g_abab[o, v, o, o] * Oe[None, None, :, None])
        _cse12 = (t2_aaaa * Ve[:, None, None, None])
        _cse13 = g_abab[o, v, o, o]
        _cse14 = ((t2_aaaa * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse15 = (g_abab[v, v, v, o] * Ve[:, None, None, None])
        _cse16 = (t2_aaaa * Oe[None, None, None, :])
        _cse17 = ((g_aaaa[v, v, v, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse18 = ((g_abab[v, v, o, v] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse19 = (g_aaaa[v, v, v, o] * Ve[:, None, None, None])
        _cse20 = (g_abab[v, v, o, v] * Ve[:, None, None, None])
        _cse21 = g_abab[v, v, v, o]
        _cse22 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse23 = (g_abab[v, v, o, v] * Oe[None, None, :, None])
        _cse24 = g_abab[v, v, o, v]
        _cse25 = ((g_aaaa[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse26 = (g_aaaa[o, v, o, o] * Oe[None, None, :, None])
        _cse27 = (t2_aaaa * Oe[None, None, :, None])
        _cse28 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse29 = ((((l2_bbbb * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse30 = g_bbbb[o, v, o, o]
        _cse31 = t2_bbbb
        _cse32 = g_bbbb[v, v, v, o]
        _tmp = einsum('ijba,eomj,baio->me', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bomj,eaio->me', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,eoij,bamo->me', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,boij,eamo->me', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeim,baoj->me', _cse0, _cse9, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obim,eaoj->me', _cse0, _cse10, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oamj,ebio->me', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oaij,ebmo->me', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eagj,gbim->me', _cse0, _cse15, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgm,gaij->me', _cse0, _cse17, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eamg,bgij->me', _cse0, _cse18, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,ebgi,gamj->me', _cse0, _cse19, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,eaig,bgmj->me', _cse0, _cse20, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bagj,geim->me', _cse0, _cse21, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bamg,egij->me', _cse0, _cse23, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,baig,egmj->me', _cse0, _cse24, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,eomj,abio->me', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,aomj,ebio->me', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,eoij,abmo->me', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,aoij,ebmo->me', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oeim,aboj->me', _cse0, _cse9, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oaim,eboj->me', _cse0, _cse10, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,obmj,eaio->me', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,obij,eamo->me', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,ebgj,gaim->me', _cse0, _cse15, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,eagm,gbij->me', _cse0, _cse17, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,ebmg,agij->me', _cse0, _cse18, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,eagi,gbmj->me', _cse0, _cse19, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,ebig,agmj->me', _cse0, _cse20, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,abgj,geim->me', _cse0, _cse21, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,abmg,egij->me', _cse0, _cse23, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,abig,egmj->me', _cse0, _cse24, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,eoji,bamo->me', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,boji,eamo->me', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,eomi,bajo->me', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,bomi,eajo->me', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oemj,baoi->me', _cse0, _cse25, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,obmj,eaoi->me', _cse0, _cse26, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oaji,ebmo->me', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oami,ebjo->me', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,eagi,gbmj->me', _cse0, _cse15, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,ebgj,gami->me', _cse0, _cse19, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,eajg,bgmi->me', _cse0, _cse20, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,ebgm,gaji->me', _cse0, _cse17, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,eamg,bgji->me', _cse0, _cse18, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,bagi,gemj->me', _cse0, _cse21, _cse28, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,bajg,egmi->me', _cse0, _cse24, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,bamg,egji->me', _cse0, _cse23, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,eoji,abmo->me', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,aoji,ebmo->me', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,eomi,abjo->me', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,aomi,ebjo->me', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oemj,aboi->me', _cse0, _cse25, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oamj,eboi->me', _cse0, _cse26, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,obji,eamo->me', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,obmi,eajo->me', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,ebgi,gamj->me', _cse0, _cse15, _cse27, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,eagj,gbmi->me', _cse0, _cse19, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,ebjg,agmi->me', _cse0, _cse20, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,eagm,gbji->me', _cse0, _cse17, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,ebmg,agji->me', _cse0, _cse18, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,abgi,gemj->me', _cse0, _cse21, _cse28, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,abjg,egmi->me', _cse0, _cse24, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,abmg,egji->me', _cse0, _cse23, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obji,eamo->me', _cse29, _cse30, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eomi,bajo->me', _cse29, _cse1, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,obmi,eaoj->me', _cse29, _cse11, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eomj,baio->me', _cse29, _cse1, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obmj,eaoi->me', _cse29, _cse11, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oaji,ebmo->me', _cse29, _cse30, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oami,eboj->me', _cse29, _cse11, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oamj,eboi->me', _cse29, _cse11, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgi,gamj->me', _cse29, _cse15, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,eagi,gbmj->me', _cse29, _cse15, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,ebgj,gami->me', _cse29, _cse15, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eagj,gbmi->me', _cse29, _cse15, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,ebmg,gaji->me', _cse29, _cse18, _cse31, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,eamg,gbji->me', _cse29, _cse18, _cse31, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,bagi,egmj->me', _cse29, _cse32, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,bagj,egmi->me', _cse29, _cse32, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        m3_ov_t3cs += _w * _iter
    return m3_ov_t3cs

