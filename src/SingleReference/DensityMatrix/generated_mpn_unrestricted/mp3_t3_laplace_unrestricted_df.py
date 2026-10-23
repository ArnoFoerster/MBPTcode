# GENERATED CODE -- DF/RI variant of mp3_t3_laplace_unrestricted.py: takes B_aa/B_bb instead of
# g_aaaa/g_abab/g_bbbb, never forming a norb^4-scale integral block.
# Do not edit by hand.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum
from src.Base.utils.grids.grids import minimax_time_grid


def t1_3_aa_t3_laplace_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, eps_a, eps_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb, ntau=6):
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
        _cse0 = ((B_aa[:, o_a, v_a] * Oe_a[None, :, None]) * Ve_a[None, None, :])
        _cse1 = B_aa[:, o_a, o_a]
        _cse2 = B_aa[:, v_a, o_a]
        _cse3 = ((t2_aaaa * Ve_a[None, :, None, None]) * Oe_a[None, None, :, None])
        _cse4 = (B_aa[:, o_a, o_a] * Oe_a[None, None, :])
        _cse5 = (t2_aaaa * Ve_a[None, :, None, None])
        _cse6 = (B_aa[:, v_a, o_a] * Oe_a[None, None, :])
        _cse7 = (B_aa[:, v_a, o_a] * Ve_a[None, :, None])
        _cse8 = (t2_aaaa * Oe_a[None, None, :, None])
        _cse9 = t2_aaaa
        _cse10 = ((B_aa[:, v_a, o_a] * Ve_a[None, :, None]) * Oe_a[None, None, :])
        _cse11 = B_aa[:, v_a, v_a]
        _cse12 = (B_aa[:, v_a, v_a] * Ve_a[None, :, None])
        _cse13 = ((B_bb[:, o_b, v_b] * Oe_b[None, :, None]) * Ve_b[None, None, :])
        _cse14 = B_bb[:, o_b, o_b]
        _cse15 = ((t2_abab * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse16 = (t2_abab * Oe_a[None, None, :, None])
        _cse17 = (t2_abab * Ve_a[:, None, None, None])
        _cse18 = t2_abab
        _cse19 = B_bb[:, v_b, o_b]
        _cse20 = B_bb[:, v_b, v_b]
        _cse21 = ((t2_aaaa * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse22 = (t2_aaaa * Ve_a[:, None, None, None])
        _cse23 = t2_bbbb
        _tmp = einsum('Qkb,Qjc,gok,gbj,caio->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gcj,baio->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gbj,cako->ai', _cse0, _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gcj,bako->ai', _cse0, _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gbk,caio->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gck,baio->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gbi,cako->ai', _cse0, _cse0, _cse1, _cse6, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gci,bako->ai', _cse0, _cse0, _cse1, _cse6, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gbk,cajo->ai', _cse0, _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gck,bajo->ai', _cse0, _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gbi,cajo->ai', _cse0, _cse0, _cse1, _cse6, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gci,bajo->ai', _cse0, _cse0, _cse1, _cse6, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gaj,bcio->ai', _cse0, _cse0, _cse1, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gaj,bcko->ai', _cse0, _cse0, _cse4, _cse7, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gak,bcio->ai', _cse0, _cse0, _cse1, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gai,bcko->ai', _cse0, _cse0, _cse1, _cse10, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gak,bcjo->ai', _cse0, _cse0, _cse4, _cse7, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gai,bcjo->ai', _cse0, _cse0, _cse1, _cse10, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gcj,haik->ai', _cse0, _cse0, _cse11, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gaj,hcik->ai', _cse0, _cse0, _cse11, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gck,haij->ai', _cse0, _cse0, _cse11, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gak,hcij->ai', _cse0, _cse0, _cse11, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbj,gch,haik->ai', _cse0, _cse0, _cse2, _cse11, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbj,gah,hcik->ai', _cse0, _cse0, _cse2, _cse12, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gch,haij->ai', _cse0, _cse0, _cse2, _cse11, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gah,hcij->ai', _cse0, _cse0, _cse2, _cse12, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gci,hakj->ai', _cse0, _cse0, _cse11, _cse6, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gai,hckj->ai', _cse0, _cse0, _cse11, _cse10, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbi,gch,hakj->ai', _cse0, _cse0, _cse6, _cse11, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbi,gah,hckj->ai', _cse0, _cse0, _cse6, _cse12, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gaj,hbik->ai', _cse0, _cse0, _cse11, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gak,hbij->ai', _cse0, _cse0, _cse11, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gcj,gah,hbik->ai', _cse0, _cse0, _cse2, _cse12, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gck,gah,hbij->ai', _cse0, _cse0, _cse2, _cse12, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gai,hbkj->ai', _cse0, _cse0, _cse11, _cse10, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gci,gah,hbkj->ai', _cse0, _cse0, _cse6, _cse12, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gbj,caio->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gcj,baio->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gbj,cako->ai', _cse0, _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gcj,bako->ai', _cse0, _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gbk,caio->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gck,baio->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gbi,cako->ai', _cse0, _cse0, _cse1, _cse6, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gci,bako->ai', _cse0, _cse0, _cse1, _cse6, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gbk,cajo->ai', _cse0, _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gck,bajo->ai', _cse0, _cse0, _cse4, _cse2, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gbi,cajo->ai', _cse0, _cse0, _cse1, _cse6, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gci,bajo->ai', _cse0, _cse0, _cse1, _cse6, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gaj,bcio->ai', _cse0, _cse0, _cse1, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gaj,bcko->ai', _cse0, _cse0, _cse4, _cse7, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gak,bcio->ai', _cse0, _cse0, _cse1, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gai,bcko->ai', _cse0, _cse0, _cse1, _cse10, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gak,bcjo->ai', _cse0, _cse0, _cse4, _cse7, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gai,bcjo->ai', _cse0, _cse0, _cse1, _cse10, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gcj,haik->ai', _cse0, _cse0, _cse11, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gaj,hcik->ai', _cse0, _cse0, _cse11, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gck,haij->ai', _cse0, _cse0, _cse11, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gak,hcij->ai', _cse0, _cse0, _cse11, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbj,gch,haik->ai', _cse0, _cse0, _cse2, _cse11, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbj,gah,hcik->ai', _cse0, _cse0, _cse2, _cse12, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbk,gch,haij->ai', _cse0, _cse0, _cse2, _cse11, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbk,gah,hcij->ai', _cse0, _cse0, _cse2, _cse12, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gci,hakj->ai', _cse0, _cse0, _cse11, _cse6, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gai,hckj->ai', _cse0, _cse0, _cse11, _cse10, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbi,gch,hakj->ai', _cse0, _cse0, _cse6, _cse11, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbi,gah,hckj->ai', _cse0, _cse0, _cse6, _cse12, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gaj,hbik->ai', _cse0, _cse0, _cse11, _cse7, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gak,hbij->ai', _cse0, _cse0, _cse11, _cse7, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gcj,gah,hbik->ai', _cse0, _cse0, _cse2, _cse12, _cse8, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,gah,hbij->ai', _cse0, _cse0, _cse2, _cse12, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gai,hbkj->ai', _cse0, _cse0, _cse11, _cse10, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gci,gah,hbkj->ai', _cse0, _cse0, _cse6, _cse12, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,goj,acio->ai', _cse0, _cse13, _cse2, _cse14, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gak,goj,bcio->ai', _cse0, _cse13, _cse7, _cse14, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbi,goj,acko->ai', _cse0, _cse13, _cse6, _cse14, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gai,goj,bcko->ai', _cse0, _cse13, _cse10, _cse14, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gbk,acoj->ai', _cse0, _cse13, _cse4, _cse2, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gak,bcoj->ai', _cse0, _cse13, _cse4, _cse7, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gbi,acoj->ai', _cse0, _cse13, _cse1, _cse6, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gai,bcoj->ai', _cse0, _cse13, _cse1, _cse10, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gcj,baio->ai', _cse0, _cse13, _cse1, _cse19, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gcj,bako->ai', _cse0, _cse13, _cse4, _cse19, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gcj,haik->ai', _cse0, _cse13, _cse11, _cse19, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gak,hcij->ai', _cse0, _cse13, _cse11, _cse7, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gah,hcij->ai', _cse0, _cse13, _cse2, _cse12, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gch,ahij->ai', _cse0, _cse13, _cse2, _cse20, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gai,hckj->ai', _cse0, _cse13, _cse11, _cse10, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbi,gah,hckj->ai', _cse0, _cse13, _cse6, _cse12, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbi,gch,ahkj->ai', _cse0, _cse13, _cse6, _cse20, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gah,gcj,hbik->ai', _cse0, _cse13, _cse12, _cse19, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gak,gch,bhij->ai', _cse0, _cse13, _cse7, _cse20, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gai,gch,bhkj->ai', _cse0, _cse13, _cse10, _cse20, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbj,gok,acio->ai', _cse0, _cse13, _cse2, _cse14, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gaj,gok,bcio->ai', _cse0, _cse13, _cse7, _cse14, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbi,gok,acjo->ai', _cse0, _cse13, _cse6, _cse14, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gai,gok,bcjo->ai', _cse0, _cse13, _cse10, _cse14, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goi,gbj,acok->ai', _cse0, _cse13, _cse4, _cse2, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goi,gaj,bcok->ai', _cse0, _cse13, _cse4, _cse7, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goj,gbi,acok->ai', _cse0, _cse13, _cse1, _cse6, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goj,gai,bcok->ai', _cse0, _cse13, _cse1, _cse10, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goj,gck,baio->ai', _cse0, _cse13, _cse1, _cse19, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goi,gck,bajo->ai', _cse0, _cse13, _cse4, _cse19, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbh,gck,haij->ai', _cse0, _cse13, _cse11, _cse19, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbh,gaj,hcik->ai', _cse0, _cse13, _cse11, _cse7, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbj,gah,hcik->ai', _cse0, _cse13, _cse2, _cse12, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbj,gch,ahik->ai', _cse0, _cse13, _cse2, _cse20, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbh,gai,hcjk->ai', _cse0, _cse13, _cse11, _cse10, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbi,gah,hcjk->ai', _cse0, _cse13, _cse6, _cse12, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbi,gch,ahjk->ai', _cse0, _cse13, _cse6, _cse20, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gah,gck,hbij->ai', _cse0, _cse13, _cse12, _cse19, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gaj,gch,bhik->ai', _cse0, _cse13, _cse7, _cse20, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gai,gch,bhjk->ai', _cse0, _cse13, _cse10, _cse20, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gak,goj,cbio->ai', _cse0, _cse13, _cse7, _cse14, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,goj,abio->ai', _cse0, _cse13, _cse2, _cse14, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gai,goj,cbko->ai', _cse0, _cse13, _cse10, _cse14, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gci,goj,abko->ai', _cse0, _cse13, _cse6, _cse14, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gak,cboj->ai', _cse0, _cse13, _cse4, _cse7, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gck,aboj->ai', _cse0, _cse13, _cse4, _cse2, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gai,cboj->ai', _cse0, _cse13, _cse1, _cse10, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gci,aboj->ai', _cse0, _cse13, _cse1, _cse6, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gbj,acio->ai', _cse0, _cse13, _cse1, _cse19, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gbj,acko->ai', _cse0, _cse13, _cse4, _cse19, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gah,gbj,hcik->ai', _cse0, _cse13, _cse12, _cse19, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gah,gck,hbij->ai', _cse0, _cse13, _cse12, _cse2, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gak,gch,hbij->ai', _cse0, _cse13, _cse7, _cse11, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gak,gbh,chij->ai', _cse0, _cse13, _cse7, _cse20, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gah,gci,hbkj->ai', _cse0, _cse13, _cse12, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gai,gch,hbkj->ai', _cse0, _cse13, _cse10, _cse11, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gai,gbh,chkj->ai', _cse0, _cse13, _cse10, _cse20, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gbj,haik->ai', _cse0, _cse13, _cse11, _cse19, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,gbh,ahij->ai', _cse0, _cse13, _cse2, _cse20, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gci,gbh,ahkj->ai', _cse0, _cse13, _cse6, _cse20, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gaj,gok,cbio->ai', _cse0, _cse13, _cse7, _cse14, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gcj,gok,abio->ai', _cse0, _cse13, _cse2, _cse14, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gai,gok,cbjo->ai', _cse0, _cse13, _cse10, _cse14, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gci,gok,abjo->ai', _cse0, _cse13, _cse6, _cse14, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goi,gaj,cbok->ai', _cse0, _cse13, _cse4, _cse7, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goi,gcj,abok->ai', _cse0, _cse13, _cse4, _cse2, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goj,gai,cbok->ai', _cse0, _cse13, _cse1, _cse10, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goj,gci,abok->ai', _cse0, _cse13, _cse1, _cse6, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goj,gbk,acio->ai', _cse0, _cse13, _cse1, _cse19, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goi,gbk,acjo->ai', _cse0, _cse13, _cse4, _cse19, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gah,gbk,hcij->ai', _cse0, _cse13, _cse12, _cse19, _cse8, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gah,gcj,hbik->ai', _cse0, _cse13, _cse12, _cse2, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gaj,gch,hbik->ai', _cse0, _cse13, _cse7, _cse11, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gaj,gbh,chik->ai', _cse0, _cse13, _cse7, _cse20, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gah,gci,hbjk->ai', _cse0, _cse13, _cse12, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gai,gch,hbjk->ai', _cse0, _cse13, _cse10, _cse11, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gai,gbh,chjk->ai', _cse0, _cse13, _cse10, _cse20, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gch,gbk,haij->ai', _cse0, _cse13, _cse11, _cse19, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gcj,gbh,ahik->ai', _cse0, _cse13, _cse2, _cse20, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gci,gbh,ahjk->ai', _cse0, _cse13, _cse6, _cse20, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gcj,abio->ai', _cse13, _cse13, _cse14, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gck,abio->ai', _cse13, _cse13, _cse14, _cse19, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gai,goj,cbko->ai', _cse13, _cse13, _cse10, _cse14, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gcj,abok->ai', _cse13, _cse13, _cse4, _cse19, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gai,gok,cbjo->ai', _cse13, _cse13, _cse10, _cse14, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gck,aboj->ai', _cse13, _cse13, _cse4, _cse19, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gbj,acio->ai', _cse13, _cse13, _cse14, _cse19, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gbk,acio->ai', _cse13, _cse13, _cse14, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gbj,acok->ai', _cse13, _cse13, _cse4, _cse19, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gbk,acoj->ai', _cse13, _cse13, _cse4, _cse19, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gah,gcj,hbik->ai', _cse13, _cse13, _cse12, _cse19, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gah,gbj,hcik->ai', _cse13, _cse13, _cse12, _cse19, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gah,gck,hbij->ai', _cse13, _cse13, _cse12, _cse19, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gah,gbk,hcij->ai', _cse13, _cse13, _cse12, _cse19, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gai,gch,hbkj->ai', _cse13, _cse13, _cse10, _cse20, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gai,gbh,hckj->ai', _cse13, _cse13, _cse10, _cse20, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gbj,ahik->ai', _cse13, _cse13, _cse20, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gbk,ahij->ai', _cse13, _cse13, _cse20, _cse19, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gcj,gbh,ahik->ai', _cse13, _cse13, _cse19, _cse20, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gck,gbh,ahij->ai', _cse13, _cse13, _cse19, _cse20, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gcj,abio->ai', _cse13, _cse13, _cse14, _cse19, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gck,abio->ai', _cse13, _cse13, _cse14, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gai,goj,cbko->ai', _cse13, _cse13, _cse10, _cse14, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gcj,abok->ai', _cse13, _cse13, _cse4, _cse19, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gai,gok,cbjo->ai', _cse13, _cse13, _cse10, _cse14, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gck,aboj->ai', _cse13, _cse13, _cse4, _cse19, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gbj,acio->ai', _cse13, _cse13, _cse14, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gbk,acio->ai', _cse13, _cse13, _cse14, _cse19, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gbj,acok->ai', _cse13, _cse13, _cse4, _cse19, _cse17, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gbk,acoj->ai', _cse13, _cse13, _cse4, _cse19, _cse17, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gah,gcj,hbik->ai', _cse13, _cse13, _cse12, _cse19, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gah,gbj,hcik->ai', _cse13, _cse13, _cse12, _cse19, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gah,gck,hbij->ai', _cse13, _cse13, _cse12, _cse19, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gah,gbk,hcij->ai', _cse13, _cse13, _cse12, _cse19, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gai,gch,hbkj->ai', _cse13, _cse13, _cse10, _cse20, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gai,gbh,hckj->ai', _cse13, _cse13, _cse10, _cse20, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gbj,ahik->ai', _cse13, _cse13, _cse20, _cse19, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gbk,ahij->ai', _cse13, _cse13, _cse20, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gcj,gbh,ahik->ai', _cse13, _cse13, _cse19, _cse20, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,gbh,ahij->ai', _cse13, _cse13, _cse19, _cse20, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        t1_3_aa_t3 += _w * _iter
    return t1_3_aa_t3


def t1_3_bb_t3_laplace_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, eps_a, eps_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb, ntau=6):
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
        _cse0 = ((B_aa[:, o_a, v_a] * Oe_a[None, :, None]) * Ve_a[None, None, :])
        _cse1 = B_aa[:, v_a, o_a]
        _cse2 = (B_bb[:, o_b, o_b] * Oe_b[None, None, :])
        _cse3 = (t2_abab * Ve_b[None, :, None, None])
        _cse4 = B_aa[:, o_a, o_a]
        _cse5 = ((t2_abab * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse6 = ((B_bb[:, v_b, o_b] * Ve_b[None, :, None]) * Oe_b[None, None, :])
        _cse7 = t2_aaaa
        _cse8 = B_aa[:, v_a, v_a]
        _cse9 = (B_bb[:, v_b, v_b] * Ve_b[None, :, None])
        _cse10 = (t2_abab * Oe_b[None, None, None, :])
        _cse11 = ((B_bb[:, o_b, v_b] * Oe_b[None, :, None]) * Ve_b[None, None, :])
        _cse12 = B_bb[:, v_b, o_b]
        _cse13 = B_bb[:, o_b, o_b]
        _cse14 = (B_bb[:, v_b, o_b] * Oe_b[None, None, :])
        _cse15 = ((t2_bbbb * Ve_b[None, :, None, None]) * Oe_b[None, None, :, None])
        _cse16 = (t2_bbbb * Ve_b[None, :, None, None])
        _cse17 = (B_bb[:, v_b, o_b] * Ve_b[None, :, None])
        _cse18 = t2_abab
        _cse19 = B_bb[:, v_b, v_b]
        _cse20 = (t2_bbbb * Oe_b[None, None, :, None])
        _cse21 = ((t2_bbbb * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse22 = (t2_bbbb * Oe_b[None, None, None, :])
        _cse23 = t2_bbbb
        _tmp = einsum('Qkb,Qjc,gbk,goi,cajo->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gck,goi,bajo->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbj,goi,cako->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gcj,goi,bako->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gbk,caoi->ai', _cse0, _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gck,baoi->ai', _cse0, _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gbj,caoi->ai', _cse0, _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gcj,baoi->ai', _cse0, _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gai,bcjo->ai', _cse0, _cse0, _cse4, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gai,bcko->ai', _cse0, _cse0, _cse4, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gai,hcjk->ai', _cse0, _cse0, _cse8, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gck,haji->ai', _cse0, _cse0, _cse8, _cse1, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gch,haji->ai', _cse0, _cse0, _cse1, _cse8, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gah,chji->ai', _cse0, _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gcj,haki->ai', _cse0, _cse0, _cse8, _cse1, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbj,gch,haki->ai', _cse0, _cse0, _cse1, _cse8, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbj,gah,chki->ai', _cse0, _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gai,hbjk->ai', _cse0, _cse0, _cse8, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gck,gah,bhji->ai', _cse0, _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gcj,gah,bhki->ai', _cse0, _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbk,goi,cajo->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,goi,bajo->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbj,goi,cako->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gcj,goi,bako->ai', _cse0, _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gbk,caoi->ai', _cse0, _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gck,baoi->ai', _cse0, _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gbj,caoi->ai', _cse0, _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gcj,baoi->ai', _cse0, _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gai,bcjo->ai', _cse0, _cse0, _cse4, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gai,bcko->ai', _cse0, _cse0, _cse4, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gai,hcjk->ai', _cse0, _cse0, _cse8, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gck,haji->ai', _cse0, _cse0, _cse8, _cse1, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbk,gch,haji->ai', _cse0, _cse0, _cse1, _cse8, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbk,gah,chji->ai', _cse0, _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gcj,haki->ai', _cse0, _cse0, _cse8, _cse1, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbj,gch,haki->ai', _cse0, _cse0, _cse1, _cse8, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbj,gah,chki->ai', _cse0, _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gai,hbjk->ai', _cse0, _cse0, _cse8, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,gah,bhji->ai', _cse0, _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gcj,gah,bhki->ai', _cse0, _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gcj,bako->ai', _cse0, _cse11, _cse2, _cse12, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gci,bako->ai', _cse0, _cse11, _cse13, _cse14, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,goj,caio->ai', _cse0, _cse11, _cse1, _cse13, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gcj,baoi->ai', _cse0, _cse11, _cse4, _cse12, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,goi,cajo->ai', _cse0, _cse11, _cse1, _cse2, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gci,baoj->ai', _cse0, _cse11, _cse4, _cse14, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gaj,bcko->ai', _cse0, _cse11, _cse2, _cse17, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gai,bcko->ai', _cse0, _cse11, _cse13, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gaj,bcoi->ai', _cse0, _cse11, _cse4, _cse17, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gai,bcoj->ai', _cse0, _cse11, _cse4, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gcj,haki->ai', _cse0, _cse11, _cse8, _cse12, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gaj,hcki->ai', _cse0, _cse11, _cse8, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gci,hakj->ai', _cse0, _cse11, _cse8, _cse14, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gai,hckj->ai', _cse0, _cse11, _cse8, _cse6, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gch,haij->ai', _cse0, _cse11, _cse1, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gah,hcij->ai', _cse0, _cse11, _cse1, _cse9, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gaj,bhki->ai', _cse0, _cse11, _cse19, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gai,bhkj->ai', _cse0, _cse11, _cse19, _cse6, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gcj,gah,bhki->ai', _cse0, _cse11, _cse12, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gci,gah,bhkj->ai', _cse0, _cse11, _cse14, _cse9, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gok,gci,bajo->ai', _cse0, _cse11, _cse13, _cse14, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goi,gck,bajo->ai', _cse0, _cse11, _cse2, _cse12, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbj,goi,cako->ai', _cse0, _cse11, _cse1, _cse2, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goj,gci,baok->ai', _cse0, _cse11, _cse4, _cse14, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbj,gok,caio->ai', _cse0, _cse11, _cse1, _cse13, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goj,gck,baoi->ai', _cse0, _cse11, _cse4, _cse12, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gok,gai,bcjo->ai', _cse0, _cse11, _cse13, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goi,gak,bcjo->ai', _cse0, _cse11, _cse2, _cse17, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goj,gai,bcok->ai', _cse0, _cse11, _cse4, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,goj,gak,bcoi->ai', _cse0, _cse11, _cse4, _cse17, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbh,gci,hajk->ai', _cse0, _cse11, _cse8, _cse14, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbh,gai,hcjk->ai', _cse0, _cse11, _cse8, _cse6, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbh,gck,haji->ai', _cse0, _cse11, _cse8, _cse12, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbh,gak,hcji->ai', _cse0, _cse11, _cse8, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbj,gch,haki->ai', _cse0, _cse11, _cse1, _cse19, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gbj,gah,hcki->ai', _cse0, _cse11, _cse1, _cse9, _cse22, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gch,gai,bhjk->ai', _cse0, _cse11, _cse19, _cse6, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gch,gak,bhji->ai', _cse0, _cse11, _cse19, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gci,gah,bhjk->ai', _cse0, _cse11, _cse14, _cse9, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjb,Qkc,gck,gah,bhji->ai', _cse0, _cse11, _cse12, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gbj,cako->ai', _cse0, _cse11, _cse2, _cse12, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gbi,cako->ai', _cse0, _cse11, _cse13, _cse14, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,goj,baio->ai', _cse0, _cse11, _cse1, _cse13, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gbj,caoi->ai', _cse0, _cse11, _cse4, _cse12, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,goi,bajo->ai', _cse0, _cse11, _cse1, _cse2, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gbi,caoj->ai', _cse0, _cse11, _cse4, _cse14, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gaj,cbko->ai', _cse0, _cse11, _cse2, _cse17, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gai,cbko->ai', _cse0, _cse11, _cse13, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gaj,cboi->ai', _cse0, _cse11, _cse4, _cse17, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gai,cboj->ai', _cse0, _cse11, _cse4, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gbj,haki->ai', _cse0, _cse11, _cse8, _cse12, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gaj,hbki->ai', _cse0, _cse11, _cse8, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gbi,hakj->ai', _cse0, _cse11, _cse8, _cse14, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gai,hbkj->ai', _cse0, _cse11, _cse8, _cse6, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,gbh,haij->ai', _cse0, _cse11, _cse1, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,gah,hbij->ai', _cse0, _cse11, _cse1, _cse9, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gaj,chki->ai', _cse0, _cse11, _cse19, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gai,chkj->ai', _cse0, _cse11, _cse19, _cse6, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbj,gah,chki->ai', _cse0, _cse11, _cse12, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbi,gah,chkj->ai', _cse0, _cse11, _cse14, _cse9, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gok,gbi,cajo->ai', _cse0, _cse11, _cse13, _cse14, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goi,gbk,cajo->ai', _cse0, _cse11, _cse2, _cse12, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gcj,goi,bako->ai', _cse0, _cse11, _cse1, _cse2, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goj,gbi,caok->ai', _cse0, _cse11, _cse4, _cse14, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gcj,gok,baio->ai', _cse0, _cse11, _cse1, _cse13, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goj,gbk,caoi->ai', _cse0, _cse11, _cse4, _cse12, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gok,gai,cbjo->ai', _cse0, _cse11, _cse13, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goi,gak,cbjo->ai', _cse0, _cse11, _cse2, _cse17, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goj,gai,cbok->ai', _cse0, _cse11, _cse4, _cse6, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,goj,gak,cboi->ai', _cse0, _cse11, _cse4, _cse17, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gch,gbi,hajk->ai', _cse0, _cse11, _cse8, _cse14, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gch,gai,hbjk->ai', _cse0, _cse11, _cse8, _cse6, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gch,gbk,haji->ai', _cse0, _cse11, _cse8, _cse12, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gch,gak,hbji->ai', _cse0, _cse11, _cse8, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gcj,gbh,haki->ai', _cse0, _cse11, _cse1, _cse19, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gcj,gah,hbki->ai', _cse0, _cse11, _cse1, _cse9, _cse22, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gbh,gai,chjk->ai', _cse0, _cse11, _cse19, _cse6, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gbh,gak,chji->ai', _cse0, _cse11, _cse19, _cse17, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gbi,gah,chjk->ai', _cse0, _cse11, _cse14, _cse9, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qjc,Qkb,gbk,gah,chji->ai', _cse0, _cse11, _cse12, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gbj,caio->ai', _cse11, _cse11, _cse13, _cse12, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gcj,baio->ai', _cse11, _cse11, _cse13, _cse12, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gbj,cako->ai', _cse11, _cse11, _cse2, _cse12, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gcj,bako->ai', _cse11, _cse11, _cse2, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gbk,caio->ai', _cse11, _cse11, _cse13, _cse12, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gck,baio->ai', _cse11, _cse11, _cse13, _cse12, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gbi,cako->ai', _cse11, _cse11, _cse13, _cse14, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gci,bako->ai', _cse11, _cse11, _cse13, _cse14, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gbk,cajo->ai', _cse11, _cse11, _cse2, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gck,bajo->ai', _cse11, _cse11, _cse2, _cse12, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gbi,cajo->ai', _cse11, _cse11, _cse13, _cse14, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gci,bajo->ai', _cse11, _cse11, _cse13, _cse14, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gaj,bcio->ai', _cse11, _cse11, _cse13, _cse17, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gaj,bcko->ai', _cse11, _cse11, _cse2, _cse17, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gak,bcio->ai', _cse11, _cse11, _cse13, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goj,gai,bcko->ai', _cse11, _cse11, _cse13, _cse6, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,goi,gak,bcjo->ai', _cse11, _cse11, _cse2, _cse17, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gok,gai,bcjo->ai', _cse11, _cse11, _cse13, _cse6, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gcj,haik->ai', _cse11, _cse11, _cse19, _cse12, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gaj,hcik->ai', _cse11, _cse11, _cse19, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gck,haij->ai', _cse11, _cse11, _cse19, _cse12, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gak,hcij->ai', _cse11, _cse11, _cse19, _cse17, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbj,gch,haik->ai', _cse11, _cse11, _cse12, _cse19, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbj,gah,hcik->ai', _cse11, _cse11, _cse12, _cse9, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gch,haij->ai', _cse11, _cse11, _cse12, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbk,gah,hcij->ai', _cse11, _cse11, _cse12, _cse9, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gci,hakj->ai', _cse11, _cse11, _cse19, _cse14, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbh,gai,hckj->ai', _cse11, _cse11, _cse19, _cse6, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbi,gch,hakj->ai', _cse11, _cse11, _cse14, _cse19, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gbi,gah,hckj->ai', _cse11, _cse11, _cse14, _cse9, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gaj,hbik->ai', _cse11, _cse11, _cse19, _cse17, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gak,hbij->ai', _cse11, _cse11, _cse19, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gcj,gah,hbik->ai', _cse11, _cse11, _cse12, _cse9, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gck,gah,hbij->ai', _cse11, _cse11, _cse12, _cse9, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gch,gai,hbkj->ai', _cse11, _cse11, _cse19, _cse6, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkb,Qjc,gci,gah,hbkj->ai', _cse11, _cse11, _cse14, _cse9, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gbj,caio->ai', _cse11, _cse11, _cse13, _cse12, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gcj,baio->ai', _cse11, _cse11, _cse13, _cse12, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gbj,cako->ai', _cse11, _cse11, _cse2, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gcj,bako->ai', _cse11, _cse11, _cse2, _cse12, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gbk,caio->ai', _cse11, _cse11, _cse13, _cse12, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gck,baio->ai', _cse11, _cse11, _cse13, _cse12, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gbi,cako->ai', _cse11, _cse11, _cse13, _cse14, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gci,bako->ai', _cse11, _cse11, _cse13, _cse14, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gbk,cajo->ai', _cse11, _cse11, _cse2, _cse12, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gck,bajo->ai', _cse11, _cse11, _cse2, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gbi,cajo->ai', _cse11, _cse11, _cse13, _cse14, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gci,bajo->ai', _cse11, _cse11, _cse13, _cse14, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gaj,bcio->ai', _cse11, _cse11, _cse13, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gaj,bcko->ai', _cse11, _cse11, _cse2, _cse17, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gak,bcio->ai', _cse11, _cse11, _cse13, _cse17, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goj,gai,bcko->ai', _cse11, _cse11, _cse13, _cse6, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,goi,gak,bcjo->ai', _cse11, _cse11, _cse2, _cse17, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gok,gai,bcjo->ai', _cse11, _cse11, _cse13, _cse6, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gcj,haik->ai', _cse11, _cse11, _cse19, _cse12, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gaj,hcik->ai', _cse11, _cse11, _cse19, _cse17, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gck,haij->ai', _cse11, _cse11, _cse19, _cse12, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gak,hcij->ai', _cse11, _cse11, _cse19, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbj,gch,haik->ai', _cse11, _cse11, _cse12, _cse19, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbj,gah,hcik->ai', _cse11, _cse11, _cse12, _cse9, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbk,gch,haij->ai', _cse11, _cse11, _cse12, _cse19, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbk,gah,hcij->ai', _cse11, _cse11, _cse12, _cse9, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gci,hakj->ai', _cse11, _cse11, _cse19, _cse14, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbh,gai,hckj->ai', _cse11, _cse11, _cse19, _cse6, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbi,gch,hakj->ai', _cse11, _cse11, _cse14, _cse19, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gbi,gah,hckj->ai', _cse11, _cse11, _cse14, _cse9, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gaj,hbik->ai', _cse11, _cse11, _cse19, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gak,hbij->ai', _cse11, _cse11, _cse19, _cse17, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gcj,gah,hbik->ai', _cse11, _cse11, _cse12, _cse9, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gck,gah,hbij->ai', _cse11, _cse11, _cse12, _cse9, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gch,gai,hbkj->ai', _cse11, _cse11, _cse19, _cse6, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('Qkc,Qjb,gci,gah,hbkj->ai', _cse11, _cse11, _cse14, _cse9, _cse23, optimize=True)
        _iter -= 0.25 * _tmp
        t1_3_bb_t3 += _w * _iter
    return t1_3_bb_t3


def m3_ov_a_t3_laplace_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, eps_a, eps_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, ntau=6):
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
        _cse1 = B_aa[:, o_a, o_a]
        _cse2 = ((B_aa[:, v_a, o_a] * Ve_a[None, :, None]) * Oe_a[None, None, :])
        _cse3 = t2_aaaa
        _cse4 = (B_aa[:, v_a, o_a] * Oe_a[None, None, :])
        _cse5 = (t2_aaaa * Ve_a[:, None, None, None])
        _cse6 = (B_aa[:, o_a, o_a] * Oe_a[None, None, :])
        _cse7 = (B_aa[:, v_a, o_a] * Ve_a[None, :, None])
        _cse8 = B_aa[:, v_a, o_a]
        _cse9 = (t2_aaaa * Oe_a[None, None, :, None])
        _cse10 = ((t2_aaaa * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse11 = (B_aa[:, v_a, v_a] * Ve_a[None, :, None])
        _cse12 = (t2_aaaa * Oe_a[None, None, None, :])
        _cse13 = B_aa[:, v_a, v_a]
        _cse14 = (t2_aaaa * Ve_a[None, :, None, None])
        _cse15 = ((t2_aaaa * Ve_a[None, :, None, None]) * Oe_a[None, None, None, :])
        _cse16 = ((((l2_abab * Oe_a[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse17 = B_bb[:, o_b, o_b]
        _cse18 = t2_abab
        _cse19 = (t2_abab * Ve_a[:, None, None, None])
        _cse20 = (t2_abab * Oe_a[None, None, :, None])
        _cse21 = ((t2_abab * Ve_a[:, None, None, None]) * Oe_a[None, None, :, None])
        _cse22 = B_bb[:, v_b, o_b]
        _cse23 = B_bb[:, v_b, v_b]
        _cse24 = ((t2_aaaa * Ve_a[None, :, None, None]) * Oe_a[None, None, :, None])
        _cse25 = ((((l2_bbbb * Oe_b[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_b[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse26 = t2_bbbb
        _tmp = einsum('ijba,goj,gem,baio->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goj,gbm,eaio->me', _cse0, _cse1, _cse4, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gem,bajo->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gbm,eajo->me', _cse0, _cse1, _cse4, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gej,baio->me', _cse0, _cse6, _cse7, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gbj,eaio->me', _cse0, _cse6, _cse8, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gei,bajo->me', _cse0, _cse6, _cse7, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gbi,eajo->me', _cse0, _cse6, _cse8, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gej,bamo->me', _cse0, _cse1, _cse7, _cse9, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gbj,eamo->me', _cse0, _cse1, _cse8, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gei,bamo->me', _cse0, _cse1, _cse7, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gbi,eamo->me', _cse0, _cse1, _cse8, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goj,gam,ebio->me', _cse0, _cse1, _cse4, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gam,ebjo->me', _cse0, _cse1, _cse4, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gaj,ebio->me', _cse0, _cse6, _cse8, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gai,ebjo->me', _cse0, _cse6, _cse8, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gaj,ebmo->me', _cse0, _cse1, _cse8, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goj,gai,ebmo->me', _cse0, _cse1, _cse8, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbm,haij->me', _cse0, _cse11, _cse4, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gam,hbij->me', _cse0, _cse11, _cse4, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbj,haim->me', _cse0, _cse11, _cse8, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gaj,hbim->me', _cse0, _cse11, _cse8, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gem,gbh,haij->me', _cse0, _cse2, _cse13, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gem,gah,hbij->me', _cse0, _cse2, _cse13, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gej,gbh,haim->me', _cse0, _cse7, _cse13, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gej,gah,hbim->me', _cse0, _cse7, _cse13, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbi,hajm->me', _cse0, _cse11, _cse8, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gai,hbjm->me', _cse0, _cse11, _cse8, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gei,gbh,hajm->me', _cse0, _cse7, _cse13, _cse12, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gei,gah,hbjm->me', _cse0, _cse7, _cse13, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gam,heij->me', _cse0, _cse13, _cse4, _cse14, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gaj,heim->me', _cse0, _cse13, _cse8, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbm,gah,heij->me', _cse0, _cse4, _cse13, _cse14, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbj,gah,heim->me', _cse0, _cse8, _cse13, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gai,hejm->me', _cse0, _cse13, _cse8, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbi,gah,hejm->me', _cse0, _cse8, _cse13, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gem,goj,baio->me', _cse16, _cse2, _cse17, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbm,goj,eaio->me', _cse16, _cse4, _cse17, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gei,goj,bamo->me', _cse16, _cse7, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbi,goj,eamo->me', _cse16, _cse8, _cse17, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gem,baoj->me', _cse16, _cse1, _cse2, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gbm,eaoj->me', _cse16, _cse1, _cse4, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gei,baoj->me', _cse16, _cse6, _cse7, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gbi,eaoj->me', _cse16, _cse6, _cse8, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gaj,ebio->me', _cse16, _cse6, _cse22, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gaj,ebmo->me', _cse16, _cse1, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gaj,hbim->me', _cse16, _cse11, _cse22, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbm,haij->me', _cse16, _cse11, _cse4, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gem,gbh,haij->me', _cse16, _cse2, _cse13, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gem,gah,bhij->me', _cse16, _cse2, _cse23, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbi,hamj->me', _cse16, _cse11, _cse8, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gei,gbh,hamj->me', _cse16, _cse7, _cse13, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gei,gah,bhmj->me', _cse16, _cse7, _cse23, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gaj,heim->me', _cse16, _cse13, _cse22, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbm,gah,ehij->me', _cse16, _cse4, _cse23, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbi,gah,ehmj->me', _cse16, _cse8, _cse23, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gem,goj,abio->me', _cse16, _cse2, _cse17, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gam,goj,ebio->me', _cse16, _cse4, _cse17, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gei,goj,abmo->me', _cse16, _cse7, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gai,goj,ebmo->me', _cse16, _cse8, _cse17, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,goi,gem,aboj->me', _cse16, _cse1, _cse2, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,goi,gam,eboj->me', _cse16, _cse1, _cse4, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gom,gei,aboj->me', _cse16, _cse6, _cse7, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gom,gai,eboj->me', _cse16, _cse6, _cse8, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gom,gbj,eaio->me', _cse16, _cse6, _cse22, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,goi,gbj,eamo->me', _cse16, _cse1, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,geh,gbj,haim->me', _cse16, _cse11, _cse22, _cse12, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,geh,gam,hbij->me', _cse16, _cse11, _cse4, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gem,gah,hbij->me', _cse16, _cse2, _cse13, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gem,gbh,ahij->me', _cse16, _cse2, _cse23, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,geh,gai,hbmj->me', _cse16, _cse11, _cse8, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gei,gah,hbmj->me', _cse16, _cse7, _cse13, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gei,gbh,ahmj->me', _cse16, _cse7, _cse23, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gah,gbj,heim->me', _cse16, _cse13, _cse22, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gam,gbh,ehij->me', _cse16, _cse4, _cse23, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gai,gbh,ehmj->me', _cse16, _cse8, _cse23, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gej,goi,bamo->me', _cse16, _cse7, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gbj,goi,eamo->me', _cse16, _cse8, _cse17, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gem,goi,bajo->me', _cse16, _cse2, _cse17, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gbm,goi,eajo->me', _cse16, _cse4, _cse17, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gom,gej,baoi->me', _cse16, _cse6, _cse7, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gom,gbj,eaoi->me', _cse16, _cse6, _cse8, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,goj,gem,baoi->me', _cse16, _cse1, _cse2, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,goj,gbm,eaoi->me', _cse16, _cse1, _cse4, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,goj,gai,ebmo->me', _cse16, _cse1, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gom,gai,ebjo->me', _cse16, _cse6, _cse22, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,geh,gai,hbmj->me', _cse16, _cse11, _cse22, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,geh,gbj,hami->me', _cse16, _cse11, _cse8, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gej,gbh,hami->me', _cse16, _cse7, _cse13, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gej,gah,bhmi->me', _cse16, _cse7, _cse23, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,geh,gbm,haji->me', _cse16, _cse11, _cse4, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gem,gbh,haji->me', _cse16, _cse2, _cse13, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gem,gah,bhji->me', _cse16, _cse2, _cse23, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gbh,gai,hemj->me', _cse16, _cse13, _cse22, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gbj,gah,ehmi->me', _cse16, _cse8, _cse23, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gbm,gah,ehji->me', _cse16, _cse4, _cse23, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gej,goi,abmo->me', _cse16, _cse7, _cse17, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gaj,goi,ebmo->me', _cse16, _cse8, _cse17, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gem,goi,abjo->me', _cse16, _cse2, _cse17, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gam,goi,ebjo->me', _cse16, _cse4, _cse17, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gom,gej,aboi->me', _cse16, _cse6, _cse7, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gom,gaj,eboi->me', _cse16, _cse6, _cse8, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,goj,gem,aboi->me', _cse16, _cse1, _cse2, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,goj,gam,eboi->me', _cse16, _cse1, _cse4, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,goj,gbi,eamo->me', _cse16, _cse1, _cse22, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gom,gbi,eajo->me', _cse16, _cse6, _cse22, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,geh,gbi,hamj->me', _cse16, _cse11, _cse22, _cse9, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,geh,gaj,hbmi->me', _cse16, _cse11, _cse8, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gej,gah,hbmi->me', _cse16, _cse7, _cse13, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gej,gbh,ahmi->me', _cse16, _cse7, _cse23, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,geh,gam,hbji->me', _cse16, _cse11, _cse4, _cse18, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gem,gah,hbji->me', _cse16, _cse2, _cse13, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gem,gbh,ahji->me', _cse16, _cse2, _cse23, _cse18, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gah,gbi,hemj->me', _cse16, _cse13, _cse22, _cse24, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gaj,gbh,ehmi->me', _cse16, _cse8, _cse23, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gam,gbh,ehji->me', _cse16, _cse4, _cse23, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goj,gbi,eamo->me', _cse25, _cse17, _cse22, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gbj,eamo->me', _cse25, _cse17, _cse22, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gem,goi,bajo->me', _cse25, _cse2, _cse17, _cse26, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gbi,eaoj->me', _cse25, _cse6, _cse22, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gem,goj,baio->me', _cse25, _cse2, _cse17, _cse26, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gbj,eaoi->me', _cse25, _cse6, _cse22, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gai,ebmo->me', _cse25, _cse17, _cse22, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gaj,ebmo->me', _cse25, _cse17, _cse22, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gai,eboj->me', _cse25, _cse6, _cse22, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gaj,eboi->me', _cse25, _cse6, _cse22, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbi,hamj->me', _cse25, _cse11, _cse22, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gai,hbmj->me', _cse25, _cse11, _cse22, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbj,hami->me', _cse25, _cse11, _cse22, _cse20, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gaj,hbmi->me', _cse25, _cse11, _cse22, _cse20, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gem,gbh,haji->me', _cse25, _cse2, _cse23, _cse26, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gem,gah,hbji->me', _cse25, _cse2, _cse23, _cse26, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gai,ehmj->me', _cse25, _cse23, _cse22, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gaj,ehmi->me', _cse25, _cse23, _cse22, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbi,gah,ehmj->me', _cse25, _cse22, _cse23, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbj,gah,ehmi->me', _cse25, _cse22, _cse23, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        m3_ov_a_t3 += _w * _iter
    return m3_ov_a_t3


def m3_ov_b_t3_laplace_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, eps_a, eps_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, ntau=6):
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
        _cse1 = B_aa[:, v_a, o_a]
        _cse2 = (B_bb[:, o_b, o_b] * Oe_b[None, None, :])
        _cse3 = (t2_abab * Ve_b[None, :, None, None])
        _cse4 = B_aa[:, o_a, o_a]
        _cse5 = ((t2_abab * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse6 = ((B_bb[:, v_b, o_b] * Ve_b[None, :, None]) * Oe_b[None, None, :])
        _cse7 = t2_aaaa
        _cse8 = B_aa[:, v_a, v_a]
        _cse9 = (B_bb[:, v_b, v_b] * Ve_b[None, :, None])
        _cse10 = (t2_abab * Oe_b[None, None, None, :])
        _cse11 = ((((l2_abab * Oe_a[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_a[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse12 = B_bb[:, o_b, o_b]
        _cse13 = t2_abab
        _cse14 = (B_bb[:, v_b, o_b] * Ve_b[None, :, None])
        _cse15 = (t2_bbbb * Ve_b[:, None, None, None])
        _cse16 = ((t2_bbbb * Ve_b[:, None, None, None]) * Oe_b[None, None, :, None])
        _cse17 = (B_bb[:, v_b, o_b] * Oe_b[None, None, :])
        _cse18 = B_bb[:, v_b, o_b]
        _cse19 = (t2_bbbb * Oe_b[None, None, None, :])
        _cse20 = B_bb[:, v_b, v_b]
        _cse21 = ((t2_bbbb * Ve_b[None, :, None, None]) * Oe_b[None, None, None, :])
        _cse22 = (t2_bbbb * Ve_b[None, :, None, None])
        _cse23 = ((t2_bbbb * Ve_b[None, :, None, None]) * Oe_b[None, None, :, None])
        _cse24 = ((((l2_bbbb * Oe_b[:, None, None, None]) * Oe_b[None, :, None, None]) * Ve_b[None, None, :, None]) * Ve_b[None, None, None, :])
        _cse25 = t2_bbbb
        _cse26 = (t2_bbbb * Oe_b[None, None, :, None])
        _tmp = einsum('ijba,gaj,gom,beio->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbj,gom,aeio->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gai,gom,bejo->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbi,gom,aejo->me', _cse0, _cse1, _cse2, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gaj,beom->me', _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gbj,aeom->me', _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gai,beom->me', _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gbi,aeom->me', _cse0, _cse4, _cse1, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goj,gem,abio->me', _cse0, _cse4, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gem,abjo->me', _cse0, _cse4, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gah,gem,hbij->me', _cse0, _cse8, _cse6, _cse7, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gah,gbj,heim->me', _cse0, _cse8, _cse1, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gaj,gbh,heim->me', _cse0, _cse1, _cse8, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gaj,geh,bhim->me', _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gah,gbi,hejm->me', _cse0, _cse8, _cse1, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gai,gbh,hejm->me', _cse0, _cse1, _cse8, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gai,geh,bhjm->me', _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gem,haij->me', _cse0, _cse8, _cse6, _cse7, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbj,geh,ahim->me', _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbi,geh,ahjm->me', _cse0, _cse1, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gem,baio->me', _cse11, _cse12, _cse6, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gej,baio->me', _cse11, _cse2, _cse14, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbi,gom,eajo->me', _cse11, _cse1, _cse2, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gem,baoj->me', _cse11, _cse4, _cse6, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbi,goj,eamo->me', _cse11, _cse1, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gej,baom->me', _cse11, _cse4, _cse14, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gam,beio->me', _cse11, _cse12, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gaj,beio->me', _cse11, _cse2, _cse18, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gam,beoj->me', _cse11, _cse4, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gaj,beom->me', _cse11, _cse4, _cse18, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gem,haij->me', _cse11, _cse8, _cse6, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gam,heij->me', _cse11, _cse8, _cse17, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gej,haim->me', _cse11, _cse8, _cse14, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gaj,heim->me', _cse11, _cse8, _cse18, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbi,geh,hajm->me', _cse11, _cse1, _cse9, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbi,gah,hejm->me', _cse11, _cse1, _cse20, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gam,bhij->me', _cse11, _cse9, _cse17, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gaj,bhim->me', _cse11, _cse9, _cse18, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gem,gah,bhij->me', _cse11, _cse6, _cse20, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gej,gah,bhim->me', _cse11, _cse14, _cse20, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,goj,gbm,aeio->me', _cse11, _cse12, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gom,gbj,aeio->me', _cse11, _cse2, _cse18, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gai,gom,bejo->me', _cse11, _cse1, _cse2, _cse22, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,goi,gbm,aeoj->me', _cse11, _cse4, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gai,goj,bemo->me', _cse11, _cse1, _cse12, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,goi,gbj,aeom->me', _cse11, _cse4, _cse18, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,goj,gem,abio->me', _cse11, _cse12, _cse6, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gom,gej,abio->me', _cse11, _cse2, _cse14, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,goi,gem,aboj->me', _cse11, _cse4, _cse6, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,goi,gej,abom->me', _cse11, _cse4, _cse14, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gah,gbm,heij->me', _cse11, _cse8, _cse17, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gah,gem,hbij->me', _cse11, _cse8, _cse6, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gah,gbj,heim->me', _cse11, _cse8, _cse18, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gah,gej,hbim->me', _cse11, _cse8, _cse14, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gai,gbh,hejm->me', _cse11, _cse1, _cse20, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gai,geh,hbjm->me', _cse11, _cse1, _cse9, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gbh,gem,ahij->me', _cse11, _cse20, _cse6, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijab,gbh,gej,ahim->me', _cse11, _cse20, _cse14, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gbm,geh,ahij->me', _cse11, _cse17, _cse9, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijab,gbj,geh,ahim->me', _cse11, _cse18, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,goi,gem,bajo->me', _cse11, _cse12, _cse6, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gom,gei,bajo->me', _cse11, _cse2, _cse14, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gbj,gom,eaio->me', _cse11, _cse1, _cse2, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,goj,gem,baoi->me', _cse11, _cse4, _cse6, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gbj,goi,eamo->me', _cse11, _cse1, _cse12, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,goj,gei,baom->me', _cse11, _cse4, _cse14, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,goi,gam,bejo->me', _cse11, _cse12, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gom,gai,bejo->me', _cse11, _cse2, _cse18, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,goj,gam,beoi->me', _cse11, _cse4, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,goj,gai,beom->me', _cse11, _cse4, _cse18, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gbh,gem,haji->me', _cse11, _cse8, _cse6, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gbh,gam,heji->me', _cse11, _cse8, _cse17, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gbh,gei,hajm->me', _cse11, _cse8, _cse14, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gbh,gai,hejm->me', _cse11, _cse8, _cse18, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gbj,geh,haim->me', _cse11, _cse1, _cse9, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,gbj,gah,heim->me', _cse11, _cse1, _cse20, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,geh,gam,bhji->me', _cse11, _cse9, _cse17, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiba,geh,gai,bhjm->me', _cse11, _cse9, _cse18, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gem,gah,bhji->me', _cse11, _cse6, _cse20, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiba,gei,gah,bhjm->me', _cse11, _cse14, _cse20, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,goi,gbm,aejo->me', _cse11, _cse12, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gom,gbi,aejo->me', _cse11, _cse2, _cse18, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gaj,gom,beio->me', _cse11, _cse1, _cse2, _cse22, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,goj,gbm,aeoi->me', _cse11, _cse4, _cse17, _cse3, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gaj,goi,bemo->me', _cse11, _cse1, _cse12, _cse23, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,goj,gbi,aeom->me', _cse11, _cse4, _cse18, _cse5, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,goi,gem,abjo->me', _cse11, _cse12, _cse6, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gom,gei,abjo->me', _cse11, _cse2, _cse14, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,goj,gem,aboi->me', _cse11, _cse4, _cse6, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,goj,gei,abom->me', _cse11, _cse4, _cse14, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gah,gbm,heji->me', _cse11, _cse8, _cse17, _cse3, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gah,gem,hbji->me', _cse11, _cse8, _cse6, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gah,gbi,hejm->me', _cse11, _cse8, _cse18, _cse5, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gah,gei,hbjm->me', _cse11, _cse8, _cse14, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gaj,gbh,heim->me', _cse11, _cse1, _cse20, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gaj,geh,hbim->me', _cse11, _cse1, _cse9, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gbh,gem,ahji->me', _cse11, _cse20, _cse6, _cse13, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('jiab,gbh,gei,ahjm->me', _cse11, _cse20, _cse14, _cse10, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gbm,geh,ahji->me', _cse11, _cse17, _cse9, _cse13, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('jiab,gbi,geh,ahjm->me', _cse11, _cse18, _cse9, _cse10, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gem,baio->me', _cse24, _cse12, _cse6, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goj,gbm,eaio->me', _cse24, _cse12, _cse17, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gem,bajo->me', _cse24, _cse12, _cse6, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gbm,eajo->me', _cse24, _cse12, _cse17, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gej,baio->me', _cse24, _cse2, _cse14, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gbj,eaio->me', _cse24, _cse2, _cse18, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gei,bajo->me', _cse24, _cse2, _cse14, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gom,gbi,eajo->me', _cse24, _cse2, _cse18, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goi,gej,bamo->me', _cse24, _cse12, _cse14, _cse26, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gbj,eamo->me', _cse24, _cse12, _cse18, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gei,bamo->me', _cse24, _cse12, _cse14, _cse26, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,goj,gbi,eamo->me', _cse24, _cse12, _cse18, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goj,gam,ebio->me', _cse24, _cse12, _cse17, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gam,ebjo->me', _cse24, _cse12, _cse17, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gaj,ebio->me', _cse24, _cse2, _cse18, _cse15, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gom,gai,ebjo->me', _cse24, _cse2, _cse18, _cse15, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goi,gaj,ebmo->me', _cse24, _cse12, _cse18, _cse16, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,goj,gai,ebmo->me', _cse24, _cse12, _cse18, _cse16, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbm,haij->me', _cse24, _cse9, _cse17, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gam,hbij->me', _cse24, _cse9, _cse17, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbj,haim->me', _cse24, _cse9, _cse18, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gaj,hbim->me', _cse24, _cse9, _cse18, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gem,gbh,haij->me', _cse24, _cse6, _cse20, _cse25, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gem,gah,hbij->me', _cse24, _cse6, _cse20, _cse25, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gej,gbh,haim->me', _cse24, _cse14, _cse20, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gej,gah,hbim->me', _cse24, _cse14, _cse20, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,geh,gbi,hajm->me', _cse24, _cse9, _cse18, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,geh,gai,hbjm->me', _cse24, _cse9, _cse18, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gei,gbh,hajm->me', _cse24, _cse14, _cse20, _cse19, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gei,gah,hbjm->me', _cse24, _cse14, _cse20, _cse19, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gam,heij->me', _cse24, _cse20, _cse17, _cse22, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gaj,heim->me', _cse24, _cse20, _cse18, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbm,gah,heij->me', _cse24, _cse17, _cse20, _cse22, optimize=True)
        _iter += 0.25 * _tmp
        _tmp = einsum('ijba,gbj,gah,heim->me', _cse24, _cse18, _cse20, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbh,gai,hejm->me', _cse24, _cse20, _cse18, _cse21, optimize=True)
        _iter -= 0.25 * _tmp
        _tmp = einsum('ijba,gbi,gah,hejm->me', _cse24, _cse18, _cse20, _cse21, optimize=True)
        _iter += 0.25 * _tmp
        m3_ov_b_t3 += _w * _iter
    return m3_ov_b_t3

