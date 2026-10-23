# GENERATED CODE -- do not edit by hand.
# Fully Laplace-fused t2_3 (order-3 doubles) numerator: one
# function per rank>=3 leaf tag (t3_2/t4_2, all spins) so t2_3
# is built with no rank>=3 tensor materialized.
# generate_mp4_t2_3_laplace_restricted.py (see its docstring);
# consumed by MPnDensityDriverRestricted.compute_t2_3_laplace.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum
from src.Base.utils.grids.grids import minimax_time_grid


def t2_3_aaaa_t3_aaaaaa_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = (((g_aaaa[o, o, v, o] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None])
        _cse1 = g_aaaa[o, v, o, o]
        _cse2 = (((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse3 = (g_aaaa[o, v, o, o] * Ve[None, :, None, None])
        _cse4 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse5 = (g_aaaa[o, v, o, o] * Oe[None, None, :, None])
        _cse6 = ((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse7 = ((g_aaaa[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse8 = (t2_aaaa * Ve[None, :, None, None])
        _cse9 = (g_aaaa[v, v, v, o] * Ve[None, :, None, None])
        _cse10 = ((g_aaaa[v, v, v, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse11 = ((g_aaaa[v, v, v, o] * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse12 = (t2_aaaa * Oe[None, None, :, None])
        _cse13 = (((g_aaaa[v, v, v, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse14 = t2_aaaa
        _cse15 = (((g_aaaa[o, v, v, v] * Oe[:, None, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse16 = ((g_aaaa[o, v, o, o] * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse17 = (((g_aaaa[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse18 = g_aaaa[v, v, v, o]
        _cse19 = (((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse20 = ((t2_aaaa * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse21 = (g_aaaa[v, v, v, o] * Oe[None, None, None, :])
        _tmp = einsum('lkcj,oclk,abio->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,oalk,cbio->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,ocik,ablo->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,oaik,cblo->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,ocil,abko->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,oail,cbko->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,oblk,caio->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,obik,calo->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,obil,cako->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,cagk,gbil->abij', _cse0, _cse9, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,cbgk,gail->abij', _cse0, _cse9, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,cagl,gbik->abij', _cse0, _cse9, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,cbgl,gaik->abij', _cse0, _cse9, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,cagi,gblk->abij', _cse0, _cse10, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,cbgi,galk->abij', _cse0, _cse10, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,abgk,gcil->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,abgl,gcik->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,abgi,gclk->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,oclk,abjo->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,oalk,cbjo->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,ocjk,ablo->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,oajk,cblo->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,ocjl,abko->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,oajl,cbko->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,oblk,cajo->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,objk,calo->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,objl,cako->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,cagk,gbjl->abij', _cse0, _cse9, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,cbgk,gajl->abij', _cse0, _cse9, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,cagl,gbjk->abij', _cse0, _cse9, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,cbgl,gajk->abij', _cse0, _cse9, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,cagj,gblk->abij', _cse0, _cse10, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,cbgj,galk->abij', _cse0, _cse10, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,abgk,gcjl->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,abgl,gcjk->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,abgj,gclk->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,ocjk,dbio->abij', _cse15, _cse5, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,odjk,cbio->abij', _cse15, _cse5, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,ocik,dbjo->abij', _cse15, _cse5, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,odik,cbjo->abij', _cse15, _cse5, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,ocij,dbko->abij', _cse15, _cse16, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,odij,cbko->abij', _cse15, _cse16, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,objk,cdio->abij', _cse15, _cse7, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,obik,cdjo->abij', _cse15, _cse7, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,obij,cdko->abij', _cse15, _cse17, _cse14, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,cdgk,gbij->abij', _cse15, _cse18, _cse19, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,cbgk,gdij->abij', _cse15, _cse9, _cse20, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,cdgj,gbik->abij', _cse15, _cse21, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,cbgj,gdik->abij', _cse15, _cse10, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,cdgi,gbjk->abij', _cse15, _cse21, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,cbgi,gdjk->abij', _cse15, _cse10, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,dbgk,gcij->abij', _cse15, _cse9, _cse20, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,dbgj,gcik->abij', _cse15, _cse10, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,dbgi,gcjk->abij', _cse15, _cse10, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,ocjk,daio->abij', _cse15, _cse5, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,odjk,caio->abij', _cse15, _cse5, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,ocik,dajo->abij', _cse15, _cse5, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,odik,cajo->abij', _cse15, _cse5, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,ocij,dako->abij', _cse15, _cse16, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,odij,cako->abij', _cse15, _cse16, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,oajk,cdio->abij', _cse15, _cse7, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,oaik,cdjo->abij', _cse15, _cse7, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,oaij,cdko->abij', _cse15, _cse17, _cse14, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,cdgk,gaij->abij', _cse15, _cse18, _cse19, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,cagk,gdij->abij', _cse15, _cse9, _cse20, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,cdgj,gaik->abij', _cse15, _cse21, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,cagj,gdik->abij', _cse15, _cse10, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,cdgi,gajk->abij', _cse15, _cse21, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,cagi,gdjk->abij', _cse15, _cse10, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,dagk,gcij->abij', _cse15, _cse9, _cse20, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,dagj,gcik->abij', _cse15, _cse10, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,dagi,gcjk->abij', _cse15, _cse10, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out


def t2_3_aaaa_t3_aabaab_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = (((g_abab[o, o, o, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, None, :])
        _cse1 = (g_abab[v, o, o, o] * Ve[:, None, None, None])
        _cse2 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse3 = ((g_abab[v, o, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse4 = (t2_abab * Ve[:, None, None, None])
        _cse5 = ((g_aaaa[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse6 = g_abab[o, v, o, o]
        _cse7 = (((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse8 = (g_abab[o, v, o, o] * Oe[None, None, :, None])
        _cse9 = ((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse10 = (g_abab[v, v, v, o] * Ve[:, None, None, None])
        _cse11 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse12 = ((g_aaaa[v, v, v, o] * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse13 = (t2_abab * Oe[None, None, :, None])
        _cse14 = (g_abab[v, v, o, v] * Ve[:, None, None, None])
        _cse15 = (((g_aaaa[v, v, v, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse16 = t2_abab
        _cse17 = ((g_abab[v, v, o, v] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse18 = (((g_abab[v, o, v, v] * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse19 = (g_abab[v, o, o, o] * Oe[None, None, :, None])
        _cse20 = ((g_aaaa[o, v, o, o] * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse21 = (((g_aaaa[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse22 = g_abab[v, v, v, o]
        _cse23 = (((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse24 = ((g_aaaa[v, v, v, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse25 = (g_abab[v, v, o, v] * Oe[None, None, :, None])
        _cse26 = ((t2_aaaa * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse27 = ((t2_aaaa * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse28 = ((g_aaaa[v, v, v, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _tmp = einsum('lkjc,bolk,acio->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkjc,aolk,bcio->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkjc,boik,aclo->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkjc,aoik,bclo->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkjc,obil,acok->abij', _cse0, _cse5, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkjc,oail,bcok->abij', _cse0, _cse5, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkjc,oclk,baio->abij', _cse0, _cse6, _cse7, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkjc,ocik,balo->abij', _cse0, _cse8, _cse9, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkjc,bcgk,gail->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkjc,bagl,gcik->abij', _cse0, _cse12, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkjc,bclg,agik->abij', _cse0, _cse14, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkjc,bagi,gclk->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkjc,bcig,aglk->abij', _cse0, _cse17, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkjc,acgk,gbil->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkjc,aclg,bgik->abij', _cse0, _cse14, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkjc,acig,bglk->abij', _cse0, _cse17, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,bolk,acjo->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,aolk,bcjo->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,bojk,aclo->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,aojk,bclo->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,objl,acok->abij', _cse0, _cse5, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,oajl,bcok->abij', _cse0, _cse5, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,oclk,bajo->abij', _cse0, _cse6, _cse7, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,ocjk,balo->abij', _cse0, _cse8, _cse9, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,bcgk,gajl->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,bagl,gcjk->abij', _cse0, _cse12, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,bclg,agjk->abij', _cse0, _cse14, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,bagj,gclk->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,bcjg,aglk->abij', _cse0, _cse17, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,acgk,gbjl->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,aclg,bgjk->abij', _cse0, _cse14, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,acjg,bglk->abij', _cse0, _cse17, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kljc,bokl,acio->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kljc,aokl,bcio->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kljc,boil,acko->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kljc,aoil,bcko->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kljc,obik,acol->abij', _cse0, _cse5, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kljc,oaik,bcol->abij', _cse0, _cse5, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kljc,ockl,baio->abij', _cse0, _cse6, _cse7, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kljc,ocil,bako->abij', _cse0, _cse8, _cse9, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kljc,bcgl,gaik->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kljc,bagk,gcil->abij', _cse0, _cse12, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kljc,bckg,agil->abij', _cse0, _cse14, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kljc,bagi,gckl->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kljc,bcig,agkl->abij', _cse0, _cse17, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kljc,acgl,gbik->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kljc,ackg,bgil->abij', _cse0, _cse14, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kljc,acig,bgkl->abij', _cse0, _cse17, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,bokl,acjo->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,aokl,bcjo->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,bojl,acko->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,aojl,bcko->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,objk,acol->abij', _cse0, _cse5, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,oajk,bcol->abij', _cse0, _cse5, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,ockl,bajo->abij', _cse0, _cse6, _cse7, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,ocjl,bako->abij', _cse0, _cse8, _cse9, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,bcgl,gajk->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,bagk,gcjl->abij', _cse0, _cse12, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,bckg,agjl->abij', _cse0, _cse14, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,bagj,gckl->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,bcjg,agkl->abij', _cse0, _cse17, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,acgl,gbjk->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,ackg,bgjl->abij', _cse0, _cse14, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,acjg,bgkl->abij', _cse0, _cse17, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,cojk,bdio->abij', _cse18, _cse19, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,bojk,cdio->abij', _cse18, _cse3, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,coik,bdjo->abij', _cse18, _cse19, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,boik,cdjo->abij', _cse18, _cse3, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,ocij,bdok->abij', _cse18, _cse20, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,obij,cdok->abij', _cse18, _cse21, _cse16, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,odjk,cbio->abij', _cse18, _cse8, _cse11, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,odik,cbjo->abij', _cse18, _cse8, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,cdgk,gbij->abij', _cse18, _cse22, _cse23, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,cbgj,gdik->abij', _cse18, _cse24, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,cdjg,bgik->abij', _cse18, _cse25, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,cbgi,gdjk->abij', _cse18, _cse24, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,cdig,bgjk->abij', _cse18, _cse25, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,bdgk,gcij->abij', _cse18, _cse10, _cse26, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,bdjg,cgik->abij', _cse18, _cse17, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,bdig,cgjk->abij', _cse18, _cse17, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkcd,cojk,adio->abij', _cse18, _cse19, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkcd,aojk,cdio->abij', _cse18, _cse3, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkcd,coik,adjo->abij', _cse18, _cse19, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkcd,aoik,cdjo->abij', _cse18, _cse3, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkcd,ocij,adok->abij', _cse18, _cse20, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkcd,oaij,cdok->abij', _cse18, _cse21, _cse16, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkcd,odjk,caio->abij', _cse18, _cse8, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkcd,odik,cajo->abij', _cse18, _cse8, _cse11, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkcd,cdgk,gaij->abij', _cse18, _cse22, _cse23, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkcd,cagj,gdik->abij', _cse18, _cse24, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkcd,cdjg,agik->abij', _cse18, _cse25, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkcd,cagi,gdjk->abij', _cse18, _cse24, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkcd,cdig,agjk->abij', _cse18, _cse25, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkcd,adgk,gcij->abij', _cse18, _cse10, _cse26, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkcd,adjg,cgik->abij', _cse18, _cse17, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkcd,adig,cgjk->abij', _cse18, _cse17, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,bojk,dcio->abij', _cse18, _cse3, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,dojk,bcio->abij', _cse18, _cse19, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,boik,dcjo->abij', _cse18, _cse3, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,doik,bcjo->abij', _cse18, _cse19, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,obij,dcok->abij', _cse18, _cse21, _cse16, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,odij,bcok->abij', _cse18, _cse20, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,ocjk,bdio->abij', _cse18, _cse8, _cse27, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,ocik,bdjo->abij', _cse18, _cse8, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,bcgk,gdij->abij', _cse18, _cse10, _cse26, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,bdgj,gcik->abij', _cse18, _cse28, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,bcjg,dgik->abij', _cse18, _cse17, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,bdgi,gcjk->abij', _cse18, _cse28, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,bcig,dgjk->abij', _cse18, _cse17, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,dcgk,gbij->abij', _cse18, _cse22, _cse23, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,dcjg,bgik->abij', _cse18, _cse25, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,dcig,bgjk->abij', _cse18, _cse25, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkdc,aojk,dcio->abij', _cse18, _cse3, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkdc,dojk,acio->abij', _cse18, _cse19, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkdc,aoik,dcjo->abij', _cse18, _cse3, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkdc,doik,acjo->abij', _cse18, _cse19, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkdc,oaij,dcok->abij', _cse18, _cse21, _cse16, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkdc,odij,acok->abij', _cse18, _cse20, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkdc,ocjk,adio->abij', _cse18, _cse8, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkdc,ocik,adjo->abij', _cse18, _cse8, _cse27, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkdc,acgk,gdij->abij', _cse18, _cse10, _cse26, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkdc,adgj,gcik->abij', _cse18, _cse28, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkdc,acjg,dgik->abij', _cse18, _cse17, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkdc,adgi,gcjk->abij', _cse18, _cse28, _cse13, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkdc,acig,dgjk->abij', _cse18, _cse17, _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkdc,dcgk,gaij->abij', _cse18, _cse22, _cse23, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('bkdc,dcjg,agik->abij', _cse18, _cse25, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('bkdc,dcig,agjk->abij', _cse18, _cse25, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out


def t2_3_aaaa_t4_aaaaaaaa_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = ((((g_aaaa[o, o, v, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse1 = (g_aaaa[v, v, o, o] * Ve[None, :, None, None])
        _cse2 = (((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse3 = (g_aaaa[v, v, o, o] * Oe[None, None, :, None])
        _cse4 = (((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse5 = ((g_aaaa[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse6 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse7 = (((g_aaaa[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse8 = (t2_aaaa * Ve[None, :, None, None])
        _cse9 = (((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse10 = (t2_aaaa * Oe[None, None, :, None])
        _tmp = einsum('lkcd,cdlk,abij->abij', _cse0, g_aaaa[v, v, o, o], ((((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,calk,dbij->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cdjk,abil->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cajk,dbil->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cdik,abjl->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,caik,dbjl->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cdil,abjk->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cail,dbjk->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cdjl,abik->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cajl,dbik->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cdji,ablk->abij', _cse0, ((g_aaaa[v, v, o, o] * Oe[None, None, :, None]) * Oe[None, None, None, :]), ((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]), optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,caji,dblk->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cblk,daij->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dblk,caij->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cbjk,dail->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dbjk,cail->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cbik,dajl->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dbik,cajl->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cbil,dajk->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dbil,cajk->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cbjl,daik->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dbjl,caik->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cbji,dalk->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dbji,calk->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dalk,cbij->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,balk,cdij->abij', _cse0, ((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]), ((t2_aaaa * Oe[None, None, :, None]) * Oe[None, None, None, :]), optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dajk,cbil->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bajk,cdil->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,daik,cbjl->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,baik,cdjl->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dail,cbjk->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bail,cdjk->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dajl,cbik->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,bajl,cdik->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,daji,cblk->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,baji,cdlk->abij', _cse0, ((((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), t2_aaaa, optimize=True)
        _iter += 0.25 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out


def t2_3_aaaa_t4_aaabaaab_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = ((((g_abab[o, o, v, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse1 = ((g_aaaa[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse2 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse3 = (((g_aaaa[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse4 = (t2_abab * Ve[:, None, None, None])
        _cse5 = g_abab[v, v, o, o]
        _cse6 = ((((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse7 = (g_abab[v, v, o, o] * Ve[:, None, None, None])
        _cse8 = (((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse9 = (g_abab[v, v, o, o] * Oe[None, None, :, None])
        _cse10 = (((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse11 = ((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse12 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse13 = (((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse14 = (t2_abab * Oe[None, None, :, None])
        _cse15 = ((((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse16 = t2_abab
        _cse17 = ((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse18 = (((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse19 = (((t2_aaaa * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse20 = ((t2_aaaa * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _tmp = einsum('lkcd,cbil,adjk->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cail,bdjk->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cbjl,adik->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cajl,bdik->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cbji,adlk->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,caji,bdlk->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cdlk,baij->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bdlk,caij->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cdjk,bail->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,bdjk,cail->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cdik,bajl->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bdik,cajl->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,adlk,cbij->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,adjk,cbil->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,adik,cbjl->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bail,cdjk->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,bajl,cdik->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,baji,cdlk->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,cbik,adjl->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,caik,bdjl->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,cbjk,adil->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,cajk,bdil->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,cbji,adkl->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,caji,bdkl->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,cdkl,baij->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,bdkl,caij->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,cdjl,baik->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,bdjl,caik->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,cdil,bajk->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,bdil,cajk->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,adkl,cbij->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,adjl,cbik->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,adil,cbjk->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,baik,cdjl->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,bajk,cdil->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,baji,cdkl->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,bdil,acjk->abij', _cse0, _cse17, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,bail,dcjk->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,bdjl,acik->abij', _cse0, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,bajl,dcik->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,bdji,aclk->abij', _cse0, _cse18, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,baji,dclk->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,bclk,daij->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,dclk,baij->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,bcjk,dail->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,dcjk,bail->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,bcik,dajl->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,dcik,bajl->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,aclk,bdij->abij', _cse0, _cse7, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,acjk,bdil->abij', _cse0, _cse11, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,acik,bdjl->abij', _cse0, _cse11, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,dail,bcjk->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,dajl,bcik->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,daji,bclk->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,bdik,acjl->abij', _cse0, _cse17, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,baik,dcjl->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,bdjk,acil->abij', _cse0, _cse17, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,bajk,dcil->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,bdji,ackl->abij', _cse0, _cse18, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,baji,dckl->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,bckl,daij->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,dckl,baij->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,bcjl,daik->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,dcjl,baik->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,bcil,dajk->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,dcil,bajk->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,ackl,bdij->abij', _cse0, _cse7, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,acjl,bdik->abij', _cse0, _cse11, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,acil,bdjk->abij', _cse0, _cse11, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,daik,bcjl->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,dajk,bcil->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,daji,bckl->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out


def t2_3_aaaa_t4_aabbaabb_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = ((((g_bbbb[o, o, v, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse1 = ((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse2 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _tmp = einsum('lkcd,acjk,bdil->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,acik,bdjl->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,acil,bdjk->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,acjl,bdik->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,abji,cdlk->abij', _cse0, ((((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), t2_bbbb, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,adjk,bcil->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bdjk,acil->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,adik,bcjl->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,bdik,acjl->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,adil,bcjk->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bdil,acjk->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,adjl,bcik->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,bdjl,acik->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dclk,abij->abij', _cse0, g_bbbb[v, v, o, o], ((((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bcjk,adil->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bcik,adjl->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,bcil,adjk->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bcjl,adik->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out


def t2_3_abab_t3_aabaab_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = (((g_abab[o, o, v, o] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None])
        _cse1 = g_abab[v, o, o, o]
        _cse2 = (((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse3 = (g_abab[v, o, o, o] * Ve[:, None, None, None])
        _cse4 = ((t2_abab * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse5 = (g_abab[v, o, o, o] * Oe[None, None, :, None])
        _cse6 = ((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse7 = ((g_abab[v, o, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse8 = (t2_abab * Ve[None, :, None, None])
        _cse9 = (g_aaaa[o, v, o, o] * Oe[None, None, :, None])
        _cse10 = ((g_aaaa[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse11 = (g_abab[o, v, o, o] * Ve[None, :, None, None])
        _cse12 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse13 = ((g_abab[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse14 = (t2_aaaa * Ve[None, :, None, None])
        _cse15 = (g_abab[v, v, v, o] * Ve[None, :, None, None])
        _cse16 = (g_aaaa[v, v, v, o] * Ve[None, :, None, None])
        _cse17 = (g_abab[v, v, o, v] * Ve[None, :, None, None])
        _cse18 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse19 = ((g_aaaa[v, v, v, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse20 = ((g_abab[v, v, o, v] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse21 = (t2_abab * Ve[:, None, None, None])
        _cse22 = ((g_abab[v, v, v, o] * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse23 = (t2_aaaa * Oe[None, None, :, None])
        _cse24 = ((g_abab[v, v, o, v] * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse25 = (t2_abab * Oe[None, None, :, None])
        _cse26 = (((g_abab[v, v, o, v] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse27 = t2_abab
        _cse28 = (((g_aaaa[o, o, v, o] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None])
        _cse29 = (g_abab[v, o, o, o] * Oe[None, None, None, :])
        _cse30 = ((g_abab[v, o, o, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse31 = ((t2_abab * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse32 = ((g_abab[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse33 = ((g_abab[v, v, v, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse34 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse35 = t2_aaaa
        _cse36 = (t2_abab * Oe[None, None, None, :])
        _cse37 = (((g_aaaa[o, v, v, v] * Oe[:, None, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse38 = ((g_abab[v, o, o, o] * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse39 = ((t2_abab * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse40 = (((g_abab[o, v, v, v] * Oe[:, None, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse41 = (((g_abab[v, o, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse42 = (g_abab[o, v, o, o] * Oe[None, None, None, :])
        _cse43 = ((g_abab[o, v, o, o] * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse44 = (g_abab[v, v, v, o] * Oe[None, None, None, :])
        _cse45 = g_abab[v, v, o, v]
        _cse46 = (((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse47 = (g_abab[v, v, o, v] * Oe[None, None, :, None])
        _cse48 = ((g_abab[v, v, v, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse49 = (g_abab[v, v, o, v] * Ve[:, None, None, None])
        _cse50 = ((g_abab[v, v, o, v] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _tmp = einsum('lkcj,colk,abio->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,aolk,cbio->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,coik,ablo->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,aoik,cblo->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,ocil,abok->abij', _cse0, _cse9, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,oail,cbok->abij', _cse0, _cse10, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,oblk,caio->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,obik,calo->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,cbgk,gail->abij', _cse0, _cse15, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,cagl,gbik->abij', _cse0, _cse16, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,cblg,agik->abij', _cse0, _cse17, _cse18, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,cagi,gblk->abij', _cse0, _cse19, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,cbig,aglk->abij', _cse0, _cse20, _cse21, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,abgk,gcil->abij', _cse0, _cse22, _cse23, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,ablg,cgik->abij', _cse0, _cse24, _cse25, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,abig,cglk->abij', _cse0, _cse26, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klcj,cokl,abio->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klcj,aokl,cbio->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klcj,coil,abko->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klcj,aoil,cbko->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klcj,ocik,abol->abij', _cse0, _cse9, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klcj,oaik,cbol->abij', _cse0, _cse10, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klcj,obkl,caio->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klcj,obil,cako->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klcj,cbgl,gaik->abij', _cse0, _cse15, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klcj,cagk,gbil->abij', _cse0, _cse16, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klcj,cbkg,agil->abij', _cse0, _cse17, _cse18, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klcj,cagi,gbkl->abij', _cse0, _cse19, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klcj,cbig,agkl->abij', _cse0, _cse20, _cse21, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klcj,abgl,gcik->abij', _cse0, _cse22, _cse23, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klcj,abkg,cgil->abij', _cse0, _cse24, _cse25, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klcj,abig,cgkl->abij', _cse0, _cse26, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,colj,abko->abij', _cse28, _cse29, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,aolj,cbko->abij', _cse28, _cse30, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,cokj,ablo->abij', _cse28, _cse29, _cse6, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,aokj,cblo->abij', _cse28, _cse30, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,ockl,aboj->abij', _cse28, g_aaaa[o, v, o, o], (((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :]), optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,oakl,cboj->abij', _cse28, (g_aaaa[o, v, o, o] * Ve[None, :, None, None]), _cse31, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,oblj,cako->abij', _cse28, _cse32, _cse14, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,obkj,calo->abij', _cse28, _cse32, _cse14, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,cbgj,gakl->abij', _cse28, _cse33, _cse14, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,cagl,gbkj->abij', _cse28, _cse16, _cse31, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,cblg,agkj->abij', _cse28, _cse17, _cse34, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,cagk,gblj->abij', _cse28, _cse16, _cse31, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,cbkg,aglj->abij', _cse28, _cse17, _cse34, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,abgj,gckl->abij', _cse28, (((g_abab[v, v, v, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :]), _cse35, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkci,ablg,cgkj->abij', _cse28, _cse24, _cse36, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkci,abkg,cglj->abij', _cse28, _cse24, _cse36, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,cokj,dbio->abij', _cse37, _cse29, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,dokj,cbio->abij', _cse37, _cse29, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,coij,dbko->abij', _cse37, _cse38, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,doij,cbko->abij', _cse37, _cse38, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,ocik,dboj->abij', _cse37, _cse9, _cse31, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,odik,cboj->abij', _cse37, _cse9, _cse31, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,obkj,cdio->abij', _cse37, _cse32, _cse23, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,obij,cdko->abij', _cse37, (((g_abab[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), _cse35, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,cbgj,gdik->abij', _cse37, _cse33, _cse23, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,cdgk,gbij->abij', _cse37, g_aaaa[v, v, v, o], (((t2_abab * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,cbkg,dgij->abij', _cse37, _cse17, _cse39, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,cdgi,gbkj->abij', _cse37, (g_aaaa[v, v, v, o] * Oe[None, None, None, :]), _cse31, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,cbig,dgkj->abij', _cse37, _cse20, _cse36, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,dbgj,gcik->abij', _cse37, _cse33, _cse23, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kacd,dbkg,cgij->abij', _cse37, _cse17, _cse39, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kacd,dbig,cgkj->abij', _cse37, _cse20, _cse36, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,cokj,adio->abij', _cse40, _cse29, _cse18, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,aokj,cdio->abij', _cse40, _cse30, _cse25, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,coij,adko->abij', _cse40, _cse38, _cse21, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,aoij,cdko->abij', _cse40, _cse41, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,ocik,adoj->abij', _cse40, _cse9, _cse34, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,oaik,cdoj->abij', _cse40, _cse10, _cse36, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,odkj,caio->abij', _cse40, _cse42, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,odij,cako->abij', _cse40, _cse43, _cse14, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,cdgj,gaik->abij', _cse40, _cse44, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,cagk,gdij->abij', _cse40, _cse16, _cse39, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,cdkg,agij->abij', _cse40, _cse45, _cse46, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,cagi,gdkj->abij', _cse40, _cse19, _cse36, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,cdig,agkj->abij', _cse40, _cse47, _cse34, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,adgj,gcik->abij', _cse40, _cse48, _cse23, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,adkg,cgij->abij', _cse40, _cse49, _cse39, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,adig,cgkj->abij', _cse40, _cse50, _cse36, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbdc,aokj,dcio->abij', _cse40, _cse30, _cse25, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbdc,dokj,acio->abij', _cse40, _cse29, _cse18, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbdc,aoij,dcko->abij', _cse40, _cse41, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbdc,doij,acko->abij', _cse40, _cse38, _cse21, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbdc,oaik,dcoj->abij', _cse40, _cse10, _cse36, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbdc,odik,acoj->abij', _cse40, _cse9, _cse34, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbdc,ockj,adio->abij', _cse40, _cse42, ((t2_aaaa * Ve[:, None, None, None]) * Oe[None, None, :, None]), optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbdc,ocij,adko->abij', _cse40, _cse43, (t2_aaaa * Ve[:, None, None, None]), optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbdc,acgj,gdik->abij', _cse40, _cse48, _cse23, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbdc,adgk,gcij->abij', _cse40, (g_aaaa[v, v, v, o] * Ve[:, None, None, None]), _cse39, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbdc,ackg,dgij->abij', _cse40, _cse49, _cse39, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbdc,adgi,gckj->abij', _cse40, ((g_aaaa[v, v, v, o] * Ve[:, None, None, None]) * Oe[None, None, None, :]), _cse36, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbdc,acig,dgkj->abij', _cse40, _cse50, _cse36, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbdc,dcgj,gaik->abij', _cse40, _cse44, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbdc,dckg,agij->abij', _cse40, _cse45, _cse46, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbdc,dcig,agkj->abij', _cse40, _cse47, _cse34, optimize=True)
        _iter -= 0.5 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out


def t2_3_abab_t3_abbabb_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = (((g_bbbb[o, o, v, o] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None])
        _cse1 = ((g_abab[v, o, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse2 = (t2_bbbb * Ve[None, :, None, None])
        _cse3 = (g_abab[o, v, o, o] * Oe[None, None, :, None])
        _cse4 = ((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse5 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse6 = ((g_abab[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse7 = (t2_abab * Ve[:, None, None, None])
        _cse8 = (g_abab[v, v, v, o] * Ve[:, None, None, None])
        _cse9 = ((t2_abab * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse10 = ((g_abab[v, v, v, o] * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse11 = (t2_abab * Oe[None, None, :, None])
        _cse12 = ((g_abab[v, v, o, v] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse13 = t2_bbbb
        _cse14 = (g_bbbb[v, v, v, o] * Ve[None, :, None, None])
        _cse15 = (((g_abab[o, o, o, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, None, :])
        _cse16 = (g_bbbb[o, v, o, o] * Oe[None, None, :, None])
        _cse17 = (g_abab[v, o, o, o] * Ve[:, None, None, None])
        _cse18 = ((t2_bbbb * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse19 = g_abab[o, v, o, o]
        _cse20 = (((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse21 = ((g_abab[v, o, o, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse22 = (g_abab[o, v, o, o] * Oe[None, None, None, :])
        _cse23 = ((g_bbbb[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse24 = (g_abab[o, v, o, o] * Ve[None, :, None, None])
        _cse25 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse26 = ((g_abab[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse27 = ((t2_abab * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse28 = (t2_abab * Oe[None, None, None, :])
        _cse29 = ((g_abab[v, v, v, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse30 = (t2_abab * Ve[None, :, None, None])
        _cse31 = (((g_abab[v, v, v, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse32 = t2_abab
        _cse33 = (g_abab[v, v, o, v] * Ve[:, None, None, None])
        _cse34 = ((g_abab[v, v, o, v] * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse35 = (t2_bbbb * Oe[None, None, :, None])
        _cse36 = ((g_bbbb[v, v, v, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse37 = (((g_abab[v, o, v, v] * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse38 = (g_abab[v, o, o, o] * Oe[None, None, :, None])
        _cse39 = ((g_abab[v, o, o, o] * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse40 = ((g_abab[o, v, o, o] * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse41 = (((g_abab[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse42 = g_abab[v, v, v, o]
        _cse43 = (((t2_abab * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse44 = (g_abab[v, v, v, o] * Ve[None, :, None, None])
        _cse45 = ((t2_abab * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse46 = (g_abab[v, v, v, o] * Oe[None, None, None, :])
        _cse47 = ((g_abab[v, v, v, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse48 = (g_abab[v, v, o, v] * Oe[None, None, :, None])
        _cse49 = ((g_abab[v, v, o, v] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse50 = (((g_bbbb[o, v, v, v] * Oe[:, None, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _tmp = einsum('lkcj,oclk,abio->abij', _cse0, g_bbbb[o, v, o, o], (((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]), optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,aoik,cblo->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,ocik,abol->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,aoil,cbko->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,ocil,abok->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,oblk,acio->abij', _cse0, (g_bbbb[o, v, o, o] * Ve[None, :, None, None]), _cse5, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,obik,acol->abij', _cse0, _cse6, _cse7, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,obil,acok->abij', _cse0, _cse6, _cse7, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,acgk,gbil->abij', _cse0, _cse8, _cse9, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,abgk,gcil->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,acgl,gbik->abij', _cse0, _cse8, _cse9, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,abgl,gcik->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,acig,gblk->abij', _cse0, _cse12, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,abig,gclk->abij', _cse0, (((g_abab[v, v, o, v] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]), _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkcj,cbgk,agil->abij', _cse0, _cse14, _cse5, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkcj,cbgl,agik->abij', _cse0, _cse14, _cse5, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,ocjk,ablo->abij', _cse15, _cse16, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,aolk,cbjo->abij', _cse15, _cse17, _cse18, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,oclk,aboj->abij', _cse15, _cse19, _cse20, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,aolj,cbko->abij', _cse15, _cse21, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,oclj,abok->abij', _cse15, _cse22, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,objk,aclo->abij', _cse15, _cse23, _cse7, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,oblk,acoj->abij', _cse15, _cse24, _cse25, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,oblj,acok->abij', _cse15, _cse26, _cse7, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,acgk,gblj->abij', _cse15, _cse8, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,abgk,gclj->abij', _cse15, _cse10, _cse28, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,acgj,gblk->abij', _cse15, _cse29, _cse30, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,abgj,gclk->abij', _cse15, _cse31, _cse32, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,aclg,gbjk->abij', _cse15, _cse33, _cse18, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,ablg,gcjk->abij', _cse15, _cse34, _cse35, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lkic,cbgk,aglj->abij', _cse15, _cse14, _cse25, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lkic,cbgj,aglk->abij', _cse15, _cse36, _cse7, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,oclj,abko->abij', _cse15, (g_bbbb[o, v, o, o] * Oe[None, None, None, :]), _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,aokj,cblo->abij', _cse15, _cse21, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,ockj,abol->abij', _cse15, _cse22, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,aokl,cbjo->abij', _cse15, _cse17, _cse18, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,ockl,aboj->abij', _cse15, _cse19, _cse20, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,oblj,acko->abij', _cse15, ((g_bbbb[o, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, None, :]), _cse7, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,obkj,acol->abij', _cse15, _cse26, _cse7, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,obkl,acoj->abij', _cse15, _cse24, _cse25, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,acgj,gbkl->abij', _cse15, _cse29, _cse30, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,abgj,gckl->abij', _cse15, _cse31, _cse32, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,acgl,gbkj->abij', _cse15, _cse8, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,abgl,gckj->abij', _cse15, _cse10, _cse28, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,ackg,gblj->abij', _cse15, _cse33, ((t2_bbbb * Ve[None, :, None, None]) * Oe[None, None, None, :]), optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,abkg,gclj->abij', _cse15, _cse34, (t2_bbbb * Oe[None, None, None, :]), optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('klic,cbgj,agkl->abij', _cse15, _cse36, _cse7, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('klic,cbgl,agkj->abij', _cse15, _cse14, _cse25, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,odjk,cbio->abij', _cse37, _cse16, _cse9, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,coik,dbjo->abij', _cse37, _cse38, _cse18, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,odik,cboj->abij', _cse37, _cse3, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,coij,dbko->abij', _cse37, _cse39, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,odij,cbok->abij', _cse37, _cse40, _cse30, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,objk,cdio->abij', _cse37, _cse23, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,obik,cdoj->abij', _cse37, _cse6, _cse28, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,obij,cdok->abij', _cse37, _cse41, _cse32, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,cdgk,gbij->abij', _cse37, _cse42, _cse43, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,cbgk,gdij->abij', _cse37, _cse44, _cse45, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,cdgj,gbik->abij', _cse37, _cse46, _cse9, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,cbgj,gdik->abij', _cse37, _cse47, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,cdig,gbjk->abij', _cse37, _cse48, _cse18, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,cbig,gdjk->abij', _cse37, _cse49, _cse35, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akcd,dbgk,cgij->abij', _cse37, _cse14, _cse45, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akcd,dbgj,cgik->abij', _cse37, _cse36, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,ocjk,dbio->abij', _cse37, _cse16, _cse9, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,doik,cbjo->abij', _cse37, _cse38, _cse18, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,ocik,dboj->abij', _cse37, _cse3, _cse27, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,doij,cbko->abij', _cse37, _cse39, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,ocij,dbok->abij', _cse37, _cse40, _cse30, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,objk,dcio->abij', _cse37, _cse23, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,obik,dcoj->abij', _cse37, _cse6, _cse28, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,obij,dcok->abij', _cse37, _cse41, _cse32, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,dcgk,gbij->abij', _cse37, _cse42, _cse43, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,dbgk,gcij->abij', _cse37, _cse44, _cse45, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,dcgj,gbik->abij', _cse37, _cse46, _cse9, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,dbgj,gcik->abij', _cse37, _cse47, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,dcig,gbjk->abij', _cse37, _cse48, _cse18, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,dbig,gcjk->abij', _cse37, _cse49, _cse35, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('akdc,cbgk,dgij->abij', _cse37, _cse14, _cse45, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('akdc,cbgj,dgik->abij', _cse37, _cse36, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,odjk,acio->abij', _cse50, _cse16, _cse5, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,aoik,dcjo->abij', _cse50, _cse1, _cse35, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,odik,acoj->abij', _cse50, _cse3, _cse25, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,aoij,dcko->abij', _cse50, (((g_abab[v, o, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), _cse13, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,odij,acok->abij', _cse50, _cse40, _cse7, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,ocjk,adio->abij', _cse50, _cse16, _cse5, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,ocik,adoj->abij', _cse50, _cse3, _cse25, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,ocij,adok->abij', _cse50, _cse40, _cse7, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,adgk,gcij->abij', _cse50, _cse8, _cse45, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,acgk,gdij->abij', _cse50, _cse8, _cse45, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,adgj,gcik->abij', _cse50, _cse29, _cse11, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,acgj,gdik->abij', _cse50, _cse29, _cse11, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,adig,gcjk->abij', _cse50, _cse12, _cse35, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,acig,gdjk->abij', _cse50, _cse12, _cse35, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('kbcd,dcgk,agij->abij', _cse50, g_bbbb[v, v, v, o], (((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('kbcd,dcgj,agik->abij', _cse50, (g_bbbb[v, v, v, o] * Oe[None, None, None, :]), _cse5, optimize=True)
        _iter -= 0.5 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out


def t2_3_abab_t4_aaabaaab_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = ((((g_aaaa[o, o, v, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse1 = (((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse2 = ((g_aaaa[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse3 = ((t2_abab * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse4 = (g_aaaa[v, v, o, o] * Ve[None, :, None, None])
        _cse5 = (((t2_abab * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse6 = ((g_aaaa[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse7 = ((g_abab[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse8 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse9 = (((g_abab[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse10 = (t2_aaaa * Ve[None, :, None, None])
        _cse11 = (((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse12 = (t2_aaaa * Oe[None, None, :, None])
        _tmp = einsum('lkcd,cdil,abkj->abij', _cse0, (g_aaaa[v, v, o, o] * Oe[None, None, :, None]), _cse1, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cail,dbkj->abij', _cse0, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cdkl,abij->abij', _cse0, g_aaaa[v, v, o, o], ((((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cakl,dbij->abij', _cse0, _cse4, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cdki,ablj->abij', _cse0, (g_aaaa[v, v, o, o] * Oe[None, None, None, :]), _cse1, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,caki,dblj->abij', _cse0, _cse6, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cblj,daik->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dblj,caik->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cbkj,dail->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dbkj,cail->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cbij,dakl->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dbij,cakl->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,ablj,cdik->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,abkj,cdil->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,abij,cdkl->abij', _cse0, ((((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), t2_aaaa, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dail,cbkj->abij', _cse0, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dakl,cbij->abij', _cse0, _cse4, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,daki,cblj->abij', _cse0, _cse6, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out


def t2_3_abab_t4_aabbaabb_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = ((((g_abab[o, o, v, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse1 = g_abab[v, v, o, o]
        _cse2 = ((((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse3 = (g_abab[v, v, o, o] * Oe[None, None, :, None])
        _cse4 = (((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse5 = ((g_abab[v, v, o, o] * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse6 = ((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse7 = (g_abab[v, v, o, o] * Oe[None, None, None, :])
        _cse8 = (((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse9 = ((g_aaaa[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse10 = ((t2_bbbb * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse11 = (g_abab[v, v, o, o] * Ve[None, :, None, None])
        _cse12 = (((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse13 = ((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse14 = ((t2_abab * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse15 = ((g_abab[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse16 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse17 = (((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse18 = (t2_abab * Oe[None, None, None, :])
        _cse19 = (((g_abab[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse20 = (t2_abab * Ve[:, None, None, None])
        _cse21 = ((((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse22 = t2_abab
        _cse23 = ((g_abab[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse24 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse25 = (((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse26 = (t2_abab * Oe[None, None, :, None])
        _cse27 = ((g_bbbb[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse28 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse29 = (g_abab[v, v, o, o] * Ve[:, None, None, None])
        _cse30 = (((t2_abab * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse31 = ((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse32 = ((t2_abab * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse33 = (((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse34 = (t2_abab * Ve[None, :, None, None])
        _cse35 = ((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse36 = ((t2_abab * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse37 = ((t2_bbbb * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse38 = ((g_bbbb[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse39 = ((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, None, :])
        _cse40 = ((t2_aaaa * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _tmp = einsum('lkcd,cdlk,abij->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cdik,ablj->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cdij,ablk->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cdlj,abik->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cali,dbjk->abij', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cblk,adij->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,ablk,cdij->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cbik,adlj->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,abik,cdlj->abij', _cse0, _cse17, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,cbij,adlk->abij', _cse0, _cse19, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,abij,cdlk->abij', _cse0, _cse21, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,cblj,adik->abij', _cse0, _cse23, _cse24, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,ablj,cdik->abij', _cse0, _cse25, _cse26, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bdjk,cail->abij', _cse0, _cse27, _cse28, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,adlk,cbij->abij', _cse0, _cse29, _cse30, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,adik,cblj->abij', _cse0, _cse31, _cse32, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,adij,cblk->abij', _cse0, _cse33, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,adlj,cbik->abij', _cse0, _cse35, _cse36, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,cdkj,abil->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,cdij,abkl->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,cdil,abkj->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,cdkl,abij->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,caki,dblj->abij', _cse0, _cse9, _cse37, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,cbkj,adil->abij', _cse0, _cse23, _cse24, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,abkj,cdil->abij', _cse0, _cse25, _cse26, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,cbij,adkl->abij', _cse0, _cse19, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,abij,cdkl->abij', _cse0, _cse21, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,cbil,adkj->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,abil,cdkj->abij', _cse0, _cse17, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,cbkl,adij->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,abkl,cdij->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,bdlj,caik->abij', _cse0, _cse38, _cse28, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,adkj,cbil->abij', _cse0, _cse35, _cse36, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,adij,cbkl->abij', _cse0, _cse33, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('klcd,adil,cbkj->abij', _cse0, _cse31, _cse32, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('klcd,adkl,cbij->abij', _cse0, _cse29, _cse30, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,aclk,dbij->abij', _cse0, _cse29, _cse30, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,acik,dblj->abij', _cse0, _cse31, _cse32, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,acij,dblk->abij', _cse0, _cse33, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,aclj,dbik->abij', _cse0, _cse35, _cse36, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,adli,cbjk->abij', _cse0, _cse39, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,ablk,dcij->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,dblk,acij->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,abik,dclj->abij', _cse0, _cse17, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,dbik,aclj->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,abij,dclk->abij', _cse0, _cse21, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,dbij,aclk->abij', _cse0, _cse19, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,ablj,dcik->abij', _cse0, _cse25, _cse26, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,dblj,acik->abij', _cse0, _cse23, _cse24, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,bcjk,adil->abij', _cse0, _cse27, _cse40, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,dclk,abij->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,dcik,ablj->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkdc,dcij,ablk->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkdc,dclj,abik->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,ackj,dbil->abij', _cse0, _cse35, _cse36, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,acij,dbkl->abij', _cse0, _cse33, _cse34, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,acil,dbkj->abij', _cse0, _cse31, _cse32, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,ackl,dbij->abij', _cse0, _cse29, _cse30, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,adki,cblj->abij', _cse0, _cse39, _cse37, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,abkj,dcil->abij', _cse0, _cse25, _cse26, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,dbkj,acil->abij', _cse0, _cse23, _cse24, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,abij,dckl->abij', _cse0, _cse21, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,dbij,ackl->abij', _cse0, _cse19, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,abil,dckj->abij', _cse0, _cse17, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,dbil,ackj->abij', _cse0, _cse15, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,abkl,dcij->abij', _cse0, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,dbkl,acij->abij', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,bclj,adik->abij', _cse0, _cse38, _cse40, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,dckj,abil->abij', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,dcij,abkl->abij', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('kldc,dcil,abkj->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('kldc,dckl,abij->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out


def t2_3_abab_t4_abbbabbb_laplace(g_aaaa, g_abab, g_bbbb, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    out = np.zeros((nv, nv, no, no))
    for _tk0 in range(ntau):
        Oe = np.exp(ei * tau[_tk0])
        Ve = np.exp(-ea * tau[_tk0])
        _iter = np.zeros((nv, nv, no, no))
        _cse0 = ((((g_bbbb[o, o, v, v] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse1 = ((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse2 = ((t2_bbbb * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse3 = (((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse4 = (t2_bbbb * Ve[None, :, None, None])
        _cse5 = (((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse6 = ((g_bbbb[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse7 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse8 = (((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse9 = (t2_bbbb * Oe[None, None, :, None])
        _cse10 = (g_bbbb[v, v, o, o] * Oe[None, None, :, None])
        _cse11 = (((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse12 = ((g_bbbb[v, v, o, o] * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _tmp = einsum('lkcd,adik,cbjl->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,acik,dbjl->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,adil,cbjk->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,acil,dbjk->abij', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,adij,cblk->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,acij,dblk->abij', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dblk,acij->abij', _cse0, (g_bbbb[v, v, o, o] * Ve[None, :, None, None]), _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dbjk,acil->abij', _cse0, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,abik,dcjl->abij', _cse0, _cse8, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,abil,dcjk->abij', _cse0, _cse8, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dbjl,acik->abij', _cse0, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,abij,dclk->abij', _cse0, ((((g_abab[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), t2_bbbb, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dclk,abij->abij', _cse0, g_bbbb[v, v, o, o], ((((t2_abab * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :]), optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bclk,adij->abij', _cse0, (g_bbbb[v, v, o, o] * Ve[:, None, None, None]), _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,dcjk,abil->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('lkcd,bcjk,adil->abij', _cse0, _cse12, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,dcjl,abik->abij', _cse0, _cse10, _cse11, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('lkcd,bcjl,adik->abij', _cse0, _cse12, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        out += (-sigma[_tk0]) * _iter
    return out

