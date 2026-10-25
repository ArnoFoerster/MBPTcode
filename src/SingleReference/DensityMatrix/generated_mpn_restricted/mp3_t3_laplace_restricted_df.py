# GENERATED CODE -- DF/RI variant of mp3_t3_laplace_restricted.py:
# takes B_aa/B_bb instead of
# g_aaaa/g_abab/g_bbbb, never forming a norb^4-scale integral block.
# Do not edit by hand.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum
from src.Base.utils.grids.grids import minimax_time_grid


def t1_3_aa_t3crossspin_laplace_df(B_aa, B_bb, d_aa, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, ntau=6):
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
        _cse0 = ((B_aa[:, o, v] * Oe[None, :, None]) * Ve[None, None, :])
        _cse1 = ((B_bb[:, o, v] * Oe[None, :, None]) * Ve[None, None, :])
        _cse2 = B_aa[:, v, o]
        _cse3 = B_bb[:, o, o]
        _cse4 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse5 = (B_aa[:, v, o] * Ve[None, :, None])
        _cse6 = (t2_abab * Oe[None, None, :, None])
        _cse7 = (B_aa[:, v, o] * Oe[None, None, :])
        _cse8 = (t2_abab * Ve[:, None, None, None])
        _cse9 = ((B_aa[:, v, o] * Ve[None, :, None]) * Oe[None, None, :])
        _cse10 = t2_abab
        _cse11 = (B_aa[:, o, o] * Oe[None, None, :])
        _cse12 = B_aa[:, o, o]
        _cse13 = B_bb[:, v, o]
        _cse14 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse15 = (t2_aaaa * Ve[None, :, None, None])
        _cse16 = B_aa[:, v, v]
        _cse17 = (B_aa[:, v, v] * Ve[None, :, None])
        _cse18 = B_bb[:, v, v]
        _cse19 = (t2_aaaa * Oe[None, None, :, None])
        _cse20 = ((t2_aaaa * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse21 = (t2_aaaa * Ve[:, None, None, None])
        _cse22 = t2_bbbb
        _tmp = einsum('Qkb,Qjc,obk,opj,acip->ai', _cse0, _cse1, _cse2, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oak,opj,bcip->ai', _cse0, _cse1, _cse5, _cse3, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,obi,opj,ackp->ai', _cse0, _cse1, _cse7, _cse3, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oai,opj,bckp->ai', _cse0, _cse1, _cse9, _cse3, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opi,obk,acpj->ai', _cse0, _cse1, _cse11, _cse2, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opi,oak,bcpj->ai', _cse0, _cse1, _cse11, _cse5, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opk,obi,acpj->ai', _cse0, _cse1, _cse12, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opk,oai,bcpj->ai', _cse0, _cse1, _cse12, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opk,ocj,baip->ai', _cse0, _cse1, _cse12, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opi,ocj,bakp->ai', _cse0, _cse1, _cse11, _cse13, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,obg,ocj,gaik->ai', _cse0, _cse1, _cse16, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,obg,oak,gcij->ai', _cse0, _cse1, _cse16, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,obk,oag,gcij->ai', _cse0, _cse1, _cse2, _cse17, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,obk,ocg,agij->ai', _cse0, _cse1, _cse2, _cse18, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,obg,oai,gckj->ai', _cse0, _cse1, _cse16, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,obi,oag,gckj->ai', _cse0, _cse1, _cse7, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,obi,ocg,agkj->ai', _cse0, _cse1, _cse7, _cse18, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oag,ocj,gbik->ai', _cse0, _cse1, _cse17, _cse13, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oak,ocg,bgij->ai', _cse0, _cse1, _cse5, _cse18, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oai,ocg,bgkj->ai', _cse0, _cse1, _cse9, _cse18, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,obj,opk,acip->ai', _cse0, _cse1, _cse2, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,oaj,opk,bcip->ai', _cse0, _cse1, _cse5, _cse3, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,obi,opk,acjp->ai', _cse0, _cse1, _cse7, _cse3, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,oai,opk,bcjp->ai', _cse0, _cse1, _cse9, _cse3, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,opi,obj,acpk->ai', _cse0, _cse1, _cse11, _cse2, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,opi,oaj,bcpk->ai', _cse0, _cse1, _cse11, _cse5, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,opj,obi,acpk->ai', _cse0, _cse1, _cse12, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,opj,oai,bcpk->ai', _cse0, _cse1, _cse12, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,opj,ock,baip->ai', _cse0, _cse1, _cse12, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,opi,ock,bajp->ai', _cse0, _cse1, _cse11, _cse13, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,obg,ock,gaij->ai', _cse0, _cse1, _cse16, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,obg,oaj,gcik->ai', _cse0, _cse1, _cse16, _cse5, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,obj,oag,gcik->ai', _cse0, _cse1, _cse2, _cse17, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,obj,ocg,agik->ai', _cse0, _cse1, _cse2, _cse18, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,obg,oai,gcjk->ai', _cse0, _cse1, _cse16, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,obi,oag,gcjk->ai', _cse0, _cse1, _cse7, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,obi,ocg,agjk->ai', _cse0, _cse1, _cse7, _cse18, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,oag,ock,gbij->ai', _cse0, _cse1, _cse17, _cse13, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,oaj,ocg,bgik->ai', _cse0, _cse1, _cse5, _cse18, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,oai,ocg,bgjk->ai', _cse0, _cse1, _cse9, _cse18, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oak,opj,cbip->ai', _cse0, _cse1, _cse5, _cse3, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,ock,opj,abip->ai', _cse0, _cse1, _cse2, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oai,opj,cbkp->ai', _cse0, _cse1, _cse9, _cse3, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oci,opj,abkp->ai', _cse0, _cse1, _cse7, _cse3, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opi,oak,cbpj->ai', _cse0, _cse1, _cse11, _cse5, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opi,ock,abpj->ai', _cse0, _cse1, _cse11, _cse2, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opk,oai,cbpj->ai', _cse0, _cse1, _cse12, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opk,oci,abpj->ai', _cse0, _cse1, _cse12, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opk,obj,acip->ai', _cse0, _cse1, _cse12, _cse13, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opi,obj,ackp->ai', _cse0, _cse1, _cse11, _cse13, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oag,obj,gcik->ai', _cse0, _cse1, _cse17, _cse13, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oag,ock,gbij->ai', _cse0, _cse1, _cse17, _cse2, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oak,ocg,gbij->ai', _cse0, _cse1, _cse5, _cse16, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oak,obg,cgij->ai', _cse0, _cse1, _cse5, _cse18, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oag,oci,gbkj->ai', _cse0, _cse1, _cse17, _cse7, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oai,ocg,gbkj->ai', _cse0, _cse1, _cse9, _cse16, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oai,obg,cgkj->ai', _cse0, _cse1, _cse9, _cse18, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,ocg,obj,gaik->ai', _cse0, _cse1, _cse16, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,ock,obg,agij->ai', _cse0, _cse1, _cse2, _cse18, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oci,obg,agkj->ai', _cse0, _cse1, _cse7, _cse18, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oaj,opk,cbip->ai', _cse0, _cse1, _cse5, _cse3, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,ocj,opk,abip->ai', _cse0, _cse1, _cse2, _cse3, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oai,opk,cbjp->ai', _cse0, _cse1, _cse9, _cse3, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oci,opk,abjp->ai', _cse0, _cse1, _cse7, _cse3, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,opi,oaj,cbpk->ai', _cse0, _cse1, _cse11, _cse5, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,opi,ocj,abpk->ai', _cse0, _cse1, _cse11, _cse2, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,opj,oai,cbpk->ai', _cse0, _cse1, _cse12, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,opj,oci,abpk->ai', _cse0, _cse1, _cse12, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,opj,obk,acip->ai', _cse0, _cse1, _cse12, _cse13, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,opi,obk,acjp->ai', _cse0, _cse1, _cse11, _cse13, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oag,obk,gcij->ai', _cse0, _cse1, _cse17, _cse13, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oag,ocj,gbik->ai', _cse0, _cse1, _cse17, _cse2, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oaj,ocg,gbik->ai', _cse0, _cse1, _cse5, _cse16, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oaj,obg,cgik->ai', _cse0, _cse1, _cse5, _cse18, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oag,oci,gbjk->ai', _cse0, _cse1, _cse17, _cse7, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oai,ocg,gbjk->ai', _cse0, _cse1, _cse9, _cse16, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oai,obg,cgjk->ai', _cse0, _cse1, _cse9, _cse18, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,ocg,obk,gaij->ai', _cse0, _cse1, _cse16, _cse13, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,ocj,obg,agik->ai', _cse0, _cse1, _cse2, _cse18, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,oci,obg,agjk->ai', _cse0, _cse1, _cse7, _cse18, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opk,ocj,abip->ai', _cse1, _cse1, _cse3, _cse13, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opj,ock,abip->ai', _cse1, _cse1, _cse3, _cse13, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oai,opj,cbkp->ai', _cse1, _cse1, _cse9, _cse3, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opi,ocj,abpk->ai', _cse1, _cse1, _cse11, _cse13, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oai,opk,cbjp->ai', _cse1, _cse1, _cse9, _cse3, _cse22, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opi,ock,abpj->ai', _cse1, _cse1, _cse11, _cse13, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opk,obj,acip->ai', _cse1, _cse1, _cse3, _cse13, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opj,obk,acip->ai', _cse1, _cse1, _cse3, _cse13, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opi,obj,acpk->ai', _cse1, _cse1, _cse11, _cse13, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,opi,obk,acpj->ai', _cse1, _cse1, _cse11, _cse13, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oag,ocj,gbik->ai', _cse1, _cse1, _cse17, _cse13, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oag,obj,gcik->ai', _cse1, _cse1, _cse17, _cse13, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oag,ock,gbij->ai', _cse1, _cse1, _cse17, _cse13, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oag,obk,gcij->ai', _cse1, _cse1, _cse17, _cse13, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oai,ocg,gbkj->ai', _cse1, _cse1, _cse9, _cse18, _cse22, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,oai,obg,gckj->ai', _cse1, _cse1, _cse9, _cse18, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,ocg,obj,agik->ai', _cse1, _cse1, _cse18, _cse13, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,ocg,obk,agij->ai', _cse1, _cse1, _cse18, _cse13, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,ocj,obg,agik->ai', _cse1, _cse1, _cse13, _cse18, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,ock,obg,agij->ai', _cse1, _cse1, _cse13, _cse18, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opk,ocj,abip->ai', _cse1, _cse1, _cse3, _cse13, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opj,ock,abip->ai', _cse1, _cse1, _cse3, _cse13, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oai,opj,cbkp->ai', _cse1, _cse1, _cse9, _cse3, _cse22, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opi,ocj,abpk->ai', _cse1, _cse1, _cse11, _cse13, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oai,opk,cbjp->ai', _cse1, _cse1, _cse9, _cse3, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opi,ock,abpj->ai', _cse1, _cse1, _cse11, _cse13, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opk,obj,acip->ai', _cse1, _cse1, _cse3, _cse13, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opj,obk,acip->ai', _cse1, _cse1, _cse3, _cse13, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opi,obj,acpk->ai', _cse1, _cse1, _cse11, _cse13, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,opi,obk,acpj->ai', _cse1, _cse1, _cse11, _cse13, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oag,ocj,gbik->ai', _cse1, _cse1, _cse17, _cse13, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oag,obj,gcik->ai', _cse1, _cse1, _cse17, _cse13, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oag,ock,gbij->ai', _cse1, _cse1, _cse17, _cse13, _cse6, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oag,obk,gcij->ai', _cse1, _cse1, _cse17, _cse13, _cse6, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oai,ocg,gbkj->ai', _cse1, _cse1, _cse9, _cse18, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,oai,obg,gckj->ai', _cse1, _cse1, _cse9, _cse18, _cse22, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,ocg,obj,agik->ai', _cse1, _cse1, _cse18, _cse13, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,ocg,obk,agij->ai', _cse1, _cse1, _cse18, _cse13, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,ocj,obg,agik->ai', _cse1, _cse1, _cse13, _cse18, _cse4, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,ock,obg,agij->ai', _cse1, _cse1, _cse13, _cse18, _cse4, optimize=True)
        _iter += 0.25 * _tmp
        t1_3_aa_t3cs += _w * _iter
    return t1_3_aa_t3cs


def m3_ov_a_t3crossspin_laplace_df(B_aa, B_bb, d_aa, o, v, nv, no, eps_a, t2_1_aaaa, t2_1_abab, l_2_1_aaaa, l_2_1_abab, ntau=6):
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
        _cse1 = ((B_aa[:, v, o] * Ve[None, :, None]) * Oe[None, None, :])
        _cse2 = B_bb[:, o, o]
        _cse3 = t2_abab
        _cse4 = (B_aa[:, v, o] * Oe[None, None, :])
        _cse5 = (t2_abab * Ve[:, None, None, None])
        _cse6 = (B_aa[:, v, o] * Ve[None, :, None])
        _cse7 = (t2_abab * Oe[None, None, :, None])
        _cse8 = B_aa[:, v, o]
        _cse9 = ((t2_abab * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse10 = B_aa[:, o, o]
        _cse11 = (B_aa[:, o, o] * Oe[None, None, :])
        _cse12 = B_bb[:, v, o]
        _cse13 = (t2_aaaa * Ve[:, None, None, None])
        _cse14 = ((t2_aaaa * Ve[:, None, None, None]) * Oe[None, None, :, None])
        _cse15 = (B_aa[:, v, v] * Ve[None, :, None])
        _cse16 = (t2_aaaa * Oe[None, None, None, :])
        _cse17 = B_aa[:, v, v]
        _cse18 = B_bb[:, v, v]
        _cse19 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, None, :])
        _cse20 = (t2_aaaa * Oe[None, None, :, None])
        _cse21 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse22 = ((((l2_bbbb * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse23 = t2_bbbb
        _tmp = einsum('ijba,oem,opj,baip->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obm,opj,eaip->me', _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oei,opj,bamp->me', _cse0, _cse6, _cse2, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,obi,opj,eamp->me', _cse0, _cse8, _cse2, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,opi,oem,bapj->me', _cse0, _cse10, _cse1, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,opi,obm,eapj->me', _cse0, _cse10, _cse4, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,opm,oei,bapj->me', _cse0, _cse11, _cse6, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,opm,obi,eapj->me', _cse0, _cse11, _cse8, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,opm,oaj,ebip->me', _cse0, _cse11, _cse12, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,opi,oaj,ebmp->me', _cse0, _cse10, _cse12, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeg,oaj,gbim->me', _cse0, _cse15, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeg,obm,gaij->me', _cse0, _cse15, _cse4, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oem,obg,gaij->me', _cse0, _cse1, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oem,oag,bgij->me', _cse0, _cse1, _cse18, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oeg,obi,gamj->me', _cse0, _cse15, _cse8, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oei,obg,gamj->me', _cse0, _cse6, _cse17, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oei,oag,bgmj->me', _cse0, _cse6, _cse18, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obg,oaj,geim->me', _cse0, _cse17, _cse12, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,obm,oag,egij->me', _cse0, _cse4, _cse18, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obi,oag,egmj->me', _cse0, _cse8, _cse18, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,oem,opj,abip->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oam,opj,ebip->me', _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,oei,opj,abmp->me', _cse0, _cse6, _cse2, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,oai,opj,ebmp->me', _cse0, _cse8, _cse2, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,opi,oem,abpj->me', _cse0, _cse10, _cse1, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,opi,oam,ebpj->me', _cse0, _cse10, _cse4, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,opm,oei,abpj->me', _cse0, _cse11, _cse6, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,opm,oai,ebpj->me', _cse0, _cse11, _cse8, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,opm,obj,eaip->me', _cse0, _cse11, _cse12, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,opi,obj,eamp->me', _cse0, _cse10, _cse12, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oeg,obj,gaim->me', _cse0, _cse15, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oeg,oam,gbij->me', _cse0, _cse15, _cse4, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oem,oag,gbij->me', _cse0, _cse1, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,oem,obg,agij->me', _cse0, _cse1, _cse18, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,oeg,oai,gbmj->me', _cse0, _cse15, _cse8, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,oei,oag,gbmj->me', _cse0, _cse6, _cse17, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oei,obg,agmj->me', _cse0, _cse6, _cse18, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oag,obj,geim->me', _cse0, _cse17, _cse12, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,oam,obg,egij->me', _cse0, _cse4, _cse18, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,oai,obg,egmj->me', _cse0, _cse8, _cse18, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oej,opi,bamp->me', _cse0, _cse6, _cse2, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,obj,opi,eamp->me', _cse0, _cse8, _cse2, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oem,opi,bajp->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,obm,opi,eajp->me', _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,opm,oej,bapi->me', _cse0, _cse11, _cse6, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,opm,obj,eapi->me', _cse0, _cse11, _cse8, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,opj,oem,bapi->me', _cse0, _cse10, _cse1, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,opj,obm,eapi->me', _cse0, _cse10, _cse4, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,opj,oai,ebmp->me', _cse0, _cse10, _cse12, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,opm,oai,ebjp->me', _cse0, _cse11, _cse12, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oeg,oai,gbmj->me', _cse0, _cse15, _cse12, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oeg,obj,gami->me', _cse0, _cse15, _cse8, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oej,obg,gami->me', _cse0, _cse6, _cse17, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oej,oag,bgmi->me', _cse0, _cse6, _cse18, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oeg,obm,gaji->me', _cse0, _cse15, _cse4, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,oem,obg,gaji->me', _cse0, _cse1, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,oem,oag,bgji->me', _cse0, _cse1, _cse18, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,obg,oai,gemj->me', _cse0, _cse17, _cse12, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,obj,oag,egmi->me', _cse0, _cse8, _cse18, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,obm,oag,egji->me', _cse0, _cse4, _cse18, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oej,opi,abmp->me', _cse0, _cse6, _cse2, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oaj,opi,ebmp->me', _cse0, _cse8, _cse2, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oem,opi,abjp->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oam,opi,ebjp->me', _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,opm,oej,abpi->me', _cse0, _cse11, _cse6, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,opm,oaj,ebpi->me', _cse0, _cse11, _cse8, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,opj,oem,abpi->me', _cse0, _cse10, _cse1, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,opj,oam,ebpi->me', _cse0, _cse10, _cse4, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,opj,obi,eamp->me', _cse0, _cse10, _cse12, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,opm,obi,eajp->me', _cse0, _cse11, _cse12, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oeg,obi,gamj->me', _cse0, _cse15, _cse12, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oeg,oaj,gbmi->me', _cse0, _cse15, _cse8, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oej,oag,gbmi->me', _cse0, _cse6, _cse17, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oej,obg,agmi->me', _cse0, _cse6, _cse18, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oeg,oam,gbji->me', _cse0, _cse15, _cse4, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oem,oag,gbji->me', _cse0, _cse1, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oem,obg,agji->me', _cse0, _cse1, _cse18, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oag,obi,gemj->me', _cse0, _cse17, _cse12, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,oaj,obg,egmi->me', _cse0, _cse8, _cse18, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,oam,obg,egji->me', _cse0, _cse4, _cse18, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,opj,obi,eamp->me', _cse22, _cse2, _cse12, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,opi,obj,eamp->me', _cse22, _cse2, _cse12, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oem,opi,bajp->me', _cse22, _cse1, _cse2, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,opm,obi,eapj->me', _cse22, _cse11, _cse12, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oem,opj,baip->me', _cse22, _cse1, _cse2, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,opm,obj,eapi->me', _cse22, _cse11, _cse12, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,opj,oai,ebmp->me', _cse22, _cse2, _cse12, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,opi,oaj,ebmp->me', _cse22, _cse2, _cse12, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,opm,oai,ebpj->me', _cse22, _cse11, _cse12, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,opm,oaj,ebpi->me', _cse22, _cse11, _cse12, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeg,obi,gamj->me', _cse22, _cse15, _cse12, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oeg,oai,gbmj->me', _cse22, _cse15, _cse12, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeg,obj,gami->me', _cse22, _cse15, _cse12, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oeg,oaj,gbmi->me', _cse22, _cse15, _cse12, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,oem,obg,gaji->me', _cse22, _cse1, _cse18, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,oem,oag,gbji->me', _cse22, _cse1, _cse18, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,obg,oai,egmj->me', _cse22, _cse18, _cse12, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,obg,oaj,egmi->me', _cse22, _cse18, _cse12, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,obi,oag,egmj->me', _cse22, _cse12, _cse18, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,obj,oag,egmi->me', _cse22, _cse12, _cse18, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        m3_ov_t3cs += _w * _iter
    return m3_ov_t3cs

