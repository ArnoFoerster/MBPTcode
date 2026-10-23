# GENERATED CODE -- Laplace-fused T4^(2)_aaaaaaaa contribution to
# t2_3_aaaa_numerator/t3_3_aaaaaa_numerator (the only two of five
# T2^(3)/T3^(3) equations that reference t4_aaaaaaaa; see
# generate_mp4_laplace_restricted.py's module docstring for scope).
# Do not edit by hand.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum
from src.Base.utils.grids.grids import minimax_time_grid


def t2_3_aaaa_t4_laplace(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, eps_a, t2_1_aaaa, ntau=6):
    t2_aaaa = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    t2_3_aaaa_t4 = np.zeros((nv, nv, no, no))
    for _tk in range(ntau):
        _t = tau[_tk]
        _w = -sigma[_tk]
        Oe = np.exp(ei * _t)
        Ve = np.exp(-ea * _t)
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
        t2_3_aaaa_t4 += _w * _iter
    return t2_3_aaaa_t4


def t3_3_aaaaaa_t4_laplace(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, eps_a, t2_1_aaaa, ntau=6):
    t2_aaaa = t2_1_aaaa
    ei = eps_a[o]
    ea = eps_a[v]
    gap_min = max(ea.min() - ei.max(), 1e-3)
    gap_max = ea.max() - ei.min()
    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)
    t3_3_aaaaaa_t4 = np.zeros((nv, nv, nv, no, no, no))
    for _tk in range(ntau):
        _t = tau[_tk]
        _w = -sigma[_tk]
        Oe = np.exp(ei * _t)
        Ve = np.exp(-ea * _t)
        _iter = np.zeros((nv, nv, nv, no, no, no))
        _cse0 = (((g_aaaa[o, o, v, o] * Oe[:, None, None, None]) * Oe[None, :, None, None]) * Ve[None, None, :, None])
        _cse1 = (g_aaaa[v, v, o, o] * Ve[None, :, None, None])
        _cse2 = ((((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse3 = ((g_aaaa[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse4 = (((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse5 = (((g_aaaa[v, v, o, o] * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse6 = ((t2_aaaa * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse7 = ((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None])
        _cse8 = (((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse9 = (((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse10 = ((t2_aaaa * Ve[None, :, None, None]) * Oe[None, None, :, None])
        _cse11 = ((((g_aaaa[v, v, o, o] * Ve[:, None, None, None]) * Ve[None, :, None, None]) * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse12 = (t2_aaaa * Ve[None, :, None, None])
        _cse13 = (((g_aaaa[o, v, v, v] * Oe[:, None, None, None]) * Ve[None, None, :, None]) * Ve[None, None, None, :])
        _cse14 = (g_aaaa[v, v, o, o] * Oe[None, None, :, None])
        _cse15 = ((g_aaaa[v, v, o, o] * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse16 = ((t2_aaaa * Oe[None, None, :, None]) * Oe[None, None, None, :])
        _cse17 = (t2_aaaa * Oe[None, None, :, None])
        _tmp = einsum('mldk,daml,bcij->abcijk', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,dbml,acij->abcijk', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,dajl,bcim->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,dbjl,acim->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,dail,bcjm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,dbil,acjm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,daim,bcjl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,dbim,acjl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,dajm,bcil->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,dbjm,acil->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,daji,bcml->abcijk', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,dbji,acml->abcijk', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,dcml,abij->abcijk', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,acml,dbij->abcijk', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,dcjl,abim->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,acjl,dbim->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,dcil,abjm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,acil,dbjm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,dcim,abjl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,acim,dbjl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,dcjm,abil->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,acjm,dbil->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,dcji,abml->abcijk', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,acji,dbml->abcijk', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,abml,dcij->abcijk', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,cbml,daij->abcijk', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,abjl,dcim->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,cbjl,daim->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,abil,dcjm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,cbil,dajm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,abim,dcjl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,cbim,dajl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,abjm,dcil->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldk,cbjm,dail->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,abji,dcml->abcijk', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldk,cbji,daml->abcijk', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,daml,bcik->abcijk', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,dbml,acik->abcijk', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,dakl,bcim->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,dbkl,acim->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,dail,bckm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,dbil,ackm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,daim,bckl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,dbim,ackl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,dakm,bcil->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,dbkm,acil->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,daki,bcml->abcijk', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,dbki,acml->abcijk', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,dcml,abik->abcijk', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,acml,dbik->abcijk', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,dckl,abim->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,ackl,dbim->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,dcil,abkm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,acil,dbkm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,dcim,abkl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,acim,dbkl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,dckm,abil->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,ackm,dbil->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,dcki,abml->abcijk', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,acki,dbml->abcijk', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,abml,dcik->abcijk', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,cbml,daik->abcijk', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,abkl,dcim->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,cbkl,daim->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,abil,dckm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,cbil,dakm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,abim,dckl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,cbim,dakl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,abkm,dcil->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldj,cbkm,dail->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,abki,dcml->abcijk', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldj,cbki,daml->abcijk', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,daml,bcjk->abcijk', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,dbml,acjk->abcijk', _cse0, _cse1, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,dakl,bcjm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,dbkl,acjm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,dajl,bckm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,dbjl,ackm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,dajm,bckl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,dbjm,ackl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,dakm,bcjl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,dbkm,acjl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,dakj,bcml->abcijk', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,dbkj,acml->abcijk', _cse0, _cse5, _cse6, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,dcml,abjk->abcijk', _cse0, _cse1, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,acml,dbjk->abcijk', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,dckl,abjm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,ackl,dbjm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,dcjl,abkm->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,acjl,dbkm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,dcjm,abkl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,acjm,dbkl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,dckm,abjl->abcijk', _cse0, _cse3, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,ackm,dbjl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,dckj,abml->abcijk', _cse0, _cse5, _cse6, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,ackj,dbml->abcijk', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,abml,dcjk->abcijk', _cse0, _cse7, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,cbml,dajk->abcijk', _cse0, _cse7, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,abkl,dcjm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,cbkl,dajm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,abjl,dckm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,cbjl,dakm->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,abjm,dckl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,cbjm,dakl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,abkm,dcjl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('mldi,cbkm,dajl->abcijk', _cse0, _cse9, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,abkj,dcml->abcijk', _cse0, _cse11, _cse12, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('mldi,cbkj,daml->abcijk', _cse0, _cse11, _cse12, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,dekl,bcij->abcijk', _cse13, _cse14, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,dbkl,ecij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,dejl,bcik->abcijk', _cse13, _cse14, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,dbjl,ecik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,deil,bcjk->abcijk', _cse13, _cse14, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,dbil,ecjk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,deik,bcjl->abcijk', _cse13, _cse15, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,dbik,ecjl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,dejk,bcil->abcijk', _cse13, _cse15, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,dbjk,ecil->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,deji,bckl->abcijk', _cse13, _cse15, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,dbji,eckl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,dckl,ebij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,eckl,dbij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,dcjl,ebik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,ecjl,dbik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,dcil,ebjk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,ecil,dbjk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,dcik,ebjl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,ecik,dbjl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,dcjk,ebil->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,ecjk,dbil->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,dcji,ebkl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,ecji,dbkl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,ebkl,dcij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,cbkl,deij->abcijk', _cse13, _cse9, _cse16, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,ebjl,dcik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,cbjl,deik->abcijk', _cse13, _cse9, _cse16, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,ebil,dcjk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,cbil,dejk->abcijk', _cse13, _cse9, _cse16, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,ebik,dcjl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,cbik,dejl->abcijk', _cse13, _cse11, _cse17, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,ebjk,dcil->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lade,cbjk,deil->abcijk', _cse13, _cse11, _cse17, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,ebji,dckl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lade,cbji,dekl->abcijk', _cse13, _cse11, _cse17, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,dekl,acij->abcijk', _cse13, _cse14, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,dakl,ecij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,dejl,acik->abcijk', _cse13, _cse14, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,dajl,ecik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,deil,acjk->abcijk', _cse13, _cse14, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,dail,ecjk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,deik,acjl->abcijk', _cse13, _cse15, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,daik,ecjl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,dejk,acil->abcijk', _cse13, _cse15, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,dajk,ecil->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,deji,ackl->abcijk', _cse13, _cse15, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,daji,eckl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,dckl,eaij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,eckl,daij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,dcjl,eaik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,ecjl,daik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,dcil,eajk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,ecil,dajk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,dcik,eajl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,ecik,dajl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,dcjk,eail->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,ecjk,dail->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,dcji,eakl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,ecji,dakl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,eakl,dcij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,cakl,deij->abcijk', _cse13, _cse9, _cse16, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,eajl,dcik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,cajl,deik->abcijk', _cse13, _cse9, _cse16, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,eail,dcjk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,cail,dejk->abcijk', _cse13, _cse9, _cse16, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,eaik,dcjl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,caik,dejl->abcijk', _cse13, _cse11, _cse17, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,eajk,dcil->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lbde,cajk,deil->abcijk', _cse13, _cse11, _cse17, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,eaji,dckl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lbde,caji,dekl->abcijk', _cse13, _cse11, _cse17, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,dekl,abij->abcijk', _cse13, _cse14, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,dakl,ebij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,dejl,abik->abcijk', _cse13, _cse14, _cse2, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,dajl,ebik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,deil,abjk->abcijk', _cse13, _cse14, _cse2, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,dail,ebjk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,deik,abjl->abcijk', _cse13, _cse15, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,daik,ebjl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,dejk,abil->abcijk', _cse13, _cse15, _cse4, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,dajk,ebil->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,deji,abkl->abcijk', _cse13, _cse15, _cse4, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,daji,ebkl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,dbkl,eaij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,ebkl,daij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,dbjl,eaik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,ebjl,daik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,dbil,eajk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,ebil,dajk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,dbik,eajl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,ebik,dajl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,dbjk,eail->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,ebjk,dail->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,dbji,eakl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,ebji,dakl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,eakl,dbij->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,bakl,deij->abcijk', _cse13, _cse9, _cse16, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,eajl,dbik->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,bajl,deik->abcijk', _cse13, _cse9, _cse16, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,eail,dbjk->abcijk', _cse13, _cse3, _cse8, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,bail,dejk->abcijk', _cse13, _cse9, _cse16, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,eaik,dbjl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,baik,dejl->abcijk', _cse13, _cse11, _cse17, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,eajk,dbil->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter -= 0.5 * _tmp
        _tmp = einsum('lcde,bajk,deil->abcijk', _cse13, _cse11, _cse17, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,eaji,dbkl->abcijk', _cse13, _cse5, _cse10, optimize=True)
        _iter += 0.5 * _tmp
        _tmp = einsum('lcde,baji,dekl->abcijk', _cse13, _cse11, _cse17, optimize=True)
        _iter -= 0.5 * _tmp
        t3_3_aaaaaa_t4 += _w * _iter
    return t3_3_aaaaaa_t4

