# GENERATED CODE -- DF/RI variant of t2_1/t1_2/t2_2/t3_2/t1_3 numerators
# (Phases 1+3 of the DF rewrite): takes B_aa/B_bb (3-index RI
# factors) instead of g_aaaa/g_abab/g_bbbb, never forming a norb^4-scale
# integral block. Every OTHER density piece (overlap2/3, m2_*/m3_* cross-
# terms) has no bracket integral factor at all and is reused UNCHANGED from
# mpn_density_pieces_restricted.py.
# Produced by src/CodeGeneration/generate_mpn_density_restricted.py.
# Do not edit by hand.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum


def t2_1_aaaa_numerator_df(B_aa, B_bb, d_aa, o, v, nv, no):
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('Qai,Qbj->abij', B_aa[:, v, o], B_aa[:, v, o], optimize=True)
    result += 1 * _tmp0
    del _tmp0
    _tmp1 = einsum('Qaj,Qbi->abij', B_aa[:, v, o], B_aa[:, v, o], optimize=True)
    result -= 1 * _tmp1
    del _tmp1
    return result


def t2_1_abab_numerator_df(B_aa, B_bb, d_aa, o, v, nv, no):
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('Qai,Qbj->abij', B_aa[:, v, o], B_bb[:, v, o], optimize=True)
    result += 1 * _tmp0
    del _tmp0
    return result


def t1_2_aa_numerator_df(B_aa, B_bb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, no))
    _tmp0 = einsum('Qkb,Qji->kbji', B_aa[:, o, v], B_aa[:, o, o], optimize=True)
    result -= 0.5 * einsum('bakj,kbji->ai', t2_aaaa, _tmp0, optimize=True)
    del _tmp0
    _tmp1 = einsum('Qki,Qjb->kijb', B_aa[:, o, o], B_aa[:, o, v], optimize=True)
    result += 0.5 * einsum('bakj,kijb->ai', t2_aaaa, _tmp1, optimize=True)
    del _tmp1
    _tmp2 = einsum('Qki,Qjb->kijb', B_aa[:, o, o], B_bb[:, o, v], optimize=True)
    result -= 1 * einsum('abkj,kijb->ai', t2_abab, _tmp2, optimize=True)
    del _tmp2
    _tmp3 = einsum('Qjb,bcij->Qci', B_aa[:, o, v], t2_aaaa, optimize=True)
    result -= 0.5 * einsum('Qac,Qci->ai', B_aa[:, v, v], _tmp3, optimize=True)
    del _tmp3
    _tmp4 = einsum('Qjc,bcij->Qbi', B_aa[:, o, v], t2_aaaa, optimize=True)
    result += 0.5 * einsum('Qab,Qbi->ai', B_aa[:, v, v], _tmp4, optimize=True)
    del _tmp4
    _tmp5 = einsum('Qjc,bcij->Qbi', B_bb[:, o, v], t2_abab, optimize=True)
    result += 1 * einsum('Qab,Qbi->ai', B_aa[:, v, v], _tmp5, optimize=True)
    del _tmp5
    return result


def t2_2_aaaa_numerator_df(B_aa, B_bb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('Qli,Qkj->likj', B_aa[:, o, o], B_aa[:, o, o], optimize=True)
    result += 0.5 * einsum('ablk,likj->abij', t2_aaaa, _tmp0, optimize=True)
    del _tmp0
    _tmp1 = einsum('Qlj,Qki->ljki', B_aa[:, o, o], B_aa[:, o, o], optimize=True)
    result -= 0.5 * einsum('ablk,ljki->abij', t2_aaaa, _tmp1, optimize=True)
    del _tmp1
    _tmp2 = einsum('Qkc,cbik->Qbi', B_aa[:, o, v], t2_aaaa, optimize=True)
    _tmp2 = einsum('Qaj,Qbi->abij', B_aa[:, v, o], _tmp2, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(1, 0, 2, 3)
    result -= 1 * _tmp2.transpose(0, 1, 3, 2)
    result += 1 * _tmp2.transpose(1, 0, 3, 2)
    del _tmp2
    _tmp3 = einsum('Qkj,Qac->kjac', B_aa[:, o, o], B_aa[:, v, v], optimize=True)
    _tmp3 = einsum('cbik,kjac->abij', t2_aaaa, _tmp3, optimize=True)
    result -= 1 * _tmp3
    result += 1 * _tmp3.transpose(1, 0, 2, 3)
    result += 1 * _tmp3.transpose(0, 1, 3, 2)
    result -= 1 * _tmp3.transpose(1, 0, 3, 2)
    del _tmp3
    _tmp4 = einsum('Qkc,bcik->Qbi', B_bb[:, o, v], t2_abab, optimize=True)
    _tmp4 = einsum('Qaj,Qbi->abij', B_aa[:, v, o], _tmp4, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(1, 0, 2, 3)
    result += 1 * _tmp4.transpose(0, 1, 3, 2)
    result -= 1 * _tmp4.transpose(1, 0, 3, 2)
    del _tmp4
    _tmp5 = einsum('Qac,Qbd,cdij->abij', B_aa[:, v, v], B_aa[:, v, v], t2_aaaa, optimize=True)
    result += 0.5 * _tmp5
    del _tmp5
    _tmp6 = einsum('Qad,Qbc,cdij->abij', B_aa[:, v, v], B_aa[:, v, v], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp6
    del _tmp6
    return result


def t2_2_abab_numerator_df(B_aa, B_bb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('Qli,Qkj->likj', B_aa[:, o, o], B_bb[:, o, o], optimize=True)
    result += 1 * einsum('ablk,likj->abij', t2_abab, _tmp0, optimize=True)
    del _tmp0
    _tmp1 = einsum('Qac,Qkj->ackj', B_aa[:, v, v], B_bb[:, o, o], optimize=True)
    result -= 1 * einsum('cbik,ackj->abij', t2_abab, _tmp1, optimize=True)
    del _tmp1
    _tmp2 = einsum('Qkc,caik->Qai', B_aa[:, o, v], t2_aaaa, optimize=True)
    result -= 1 * einsum('Qbj,Qai->abij', B_bb[:, v, o], _tmp2, optimize=True)
    del _tmp2
    _tmp3 = einsum('Qkc,acik->Qai', B_bb[:, o, v], t2_abab, optimize=True)
    result += 1 * einsum('Qbj,Qai->abij', B_bb[:, v, o], _tmp3, optimize=True)
    del _tmp3
    _tmp4 = einsum('Qkj,Qbc->kjbc', B_bb[:, o, o], B_bb[:, v, v], optimize=True)
    result -= 1 * einsum('acik,kjbc->abij', t2_abab, _tmp4, optimize=True)
    del _tmp4
    _tmp5 = einsum('Qkc,cbkj->Qbj', B_aa[:, o, v], t2_abab, optimize=True)
    result += 1 * einsum('Qai,Qbj->abij', B_aa[:, v, o], _tmp5, optimize=True)
    del _tmp5
    _tmp6 = einsum('Qki,Qac->kiac', B_aa[:, o, o], B_aa[:, v, v], optimize=True)
    result -= 1 * einsum('cbkj,kiac->abij', t2_abab, _tmp6, optimize=True)
    del _tmp6
    _tmp7 = einsum('Qkc,cbjk->Qbj', B_bb[:, o, v], t2_bbbb, optimize=True)
    result -= 1 * einsum('Qai,Qbj->abij', B_aa[:, v, o], _tmp7, optimize=True)
    del _tmp7
    _tmp8 = einsum('Qki,Qbc->kibc', B_aa[:, o, o], B_bb[:, v, v], optimize=True)
    result -= 1 * einsum('ackj,kibc->abij', t2_abab, _tmp8, optimize=True)
    del _tmp8
    _tmp9 = einsum('Qac,Qbd,cdij->abij', B_aa[:, v, v], B_bb[:, v, v], t2_abab, optimize=True)
    result += 0.5 * _tmp9
    result += 0.5 * _tmp9
    del _tmp9
    return result


def t3_2_aaaaaa_numerator_df(B_aa, B_bb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('Qlj,Qak->ljak', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    _tmp0 = einsum('bcil,ljak->abcijk', t2_aaaa, _tmp0, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result -= 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    del _tmp0
    _tmp1 = einsum('Qlk,Qaj->lkaj', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    _tmp1 = einsum('bcil,lkaj->abcijk', t2_aaaa, _tmp1, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp1.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp1.transpose(1, 0, 2, 4, 3, 5)
    del _tmp1
    _tmp2 = einsum('Qli,Qaj->liaj', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    _tmp2 = einsum('bckl,liaj->abcijk', t2_aaaa, _tmp2, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(1, 0, 2, 3, 4, 5)
    del _tmp2
    _tmp3 = einsum('Qlj,Qai->ljai', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    _tmp3 = einsum('bckl,ljai->abcijk', t2_aaaa, _tmp3, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5)
    del _tmp3
    _tmp4 = einsum('Qlj,Qck->ljck', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    _tmp4 = einsum('abil,ljck->abcijk', t2_aaaa, _tmp4, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(0, 1, 2, 4, 3, 5)
    del _tmp4
    _tmp5 = einsum('Qlk,Qcj->lkcj', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    _tmp5 = einsum('abil,lkcj->abcijk', t2_aaaa, _tmp5, optimize=True)
    result += 1 * _tmp5
    result -= 1 * _tmp5.transpose(0, 1, 2, 4, 3, 5)
    del _tmp5
    _tmp6 = einsum('Qli,Qcj->licj', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    result -= 1 * einsum('abkl,licj->abcijk', t2_aaaa, _tmp6, optimize=True)
    del _tmp6
    _tmp7 = einsum('Qlj,Qci->ljci', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    result += 1 * einsum('abkl,ljci->abcijk', t2_aaaa, _tmp7, optimize=True)
    del _tmp7
    _tmp8 = einsum('Qad,Qbk->adbk', B_aa[:, v, v], B_aa[:, v, o], optimize=True)
    _tmp8 = einsum('dcij,adbk->abcijk', t2_aaaa, _tmp8, optimize=True)
    result -= 1 * _tmp8
    result += 1 * _tmp8.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp8.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp8.transpose(0, 2, 1, 3, 5, 4)
    del _tmp8
    _tmp9 = einsum('Qak,Qbd->akbd', B_aa[:, v, o], B_aa[:, v, v], optimize=True)
    _tmp9 = einsum('dcij,akbd->abcijk', t2_aaaa, _tmp9, optimize=True)
    result += 1 * _tmp9
    result -= 1 * _tmp9.transpose(0, 2, 1, 3, 4, 5)
    result -= 1 * _tmp9.transpose(0, 1, 2, 3, 5, 4)
    result += 1 * _tmp9.transpose(0, 2, 1, 3, 5, 4)
    del _tmp9
    _tmp10 = einsum('Qad,Qbi->adbi', B_aa[:, v, v], B_aa[:, v, o], optimize=True)
    _tmp10 = einsum('dcjk,adbi->abcijk', t2_aaaa, _tmp10, optimize=True)
    result -= 1 * _tmp10
    result += 1 * _tmp10.transpose(0, 2, 1, 3, 4, 5)
    del _tmp10
    _tmp11 = einsum('Qai,Qbd->aibd', B_aa[:, v, o], B_aa[:, v, v], optimize=True)
    _tmp11 = einsum('dcjk,aibd->abcijk', t2_aaaa, _tmp11, optimize=True)
    result += 1 * _tmp11
    result -= 1 * _tmp11.transpose(0, 2, 1, 3, 4, 5)
    del _tmp11
    _tmp12 = einsum('Qbd,Qck->bdck', B_aa[:, v, v], B_aa[:, v, o], optimize=True)
    _tmp12 = einsum('daij,bdck->abcijk', t2_aaaa, _tmp12, optimize=True)
    result -= 1 * _tmp12
    result += 1 * _tmp12.transpose(0, 1, 2, 3, 5, 4)
    del _tmp12
    _tmp13 = einsum('Qbk,Qcd->bkcd', B_aa[:, v, o], B_aa[:, v, v], optimize=True)
    _tmp13 = einsum('daij,bkcd->abcijk', t2_aaaa, _tmp13, optimize=True)
    result += 1 * _tmp13
    result -= 1 * _tmp13.transpose(0, 1, 2, 3, 5, 4)
    del _tmp13
    _tmp14 = einsum('Qbd,Qci->bdci', B_aa[:, v, v], B_aa[:, v, o], optimize=True)
    result -= 1 * einsum('dajk,bdci->abcijk', t2_aaaa, _tmp14, optimize=True)
    del _tmp14
    _tmp15 = einsum('Qbi,Qcd->bicd', B_aa[:, v, o], B_aa[:, v, v], optimize=True)
    result += 1 * einsum('dajk,bicd->abcijk', t2_aaaa, _tmp15, optimize=True)
    del _tmp15
    return result


def t3_2_aabaab_numerator_df(B_aa, B_bb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('Qaj,Qlk->ajlk', B_aa[:, v, o], B_bb[:, o, o], optimize=True)
    _tmp0 = einsum('bcil,ajlk->abcijk', t2_abab, _tmp0, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    del _tmp0
    _tmp1 = einsum('Qli,Qaj->liaj', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    _tmp1 = einsum('bclk,liaj->abcijk', t2_abab, _tmp1, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    del _tmp1
    _tmp2 = einsum('Qlj,Qai->ljai', B_aa[:, o, o], B_aa[:, v, o], optimize=True)
    _tmp2 = einsum('bclk,ljai->abcijk', t2_abab, _tmp2, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(1, 0, 2, 3, 4, 5)
    del _tmp2
    _tmp3 = einsum('Qlj,Qck->ljck', B_aa[:, o, o], B_bb[:, v, o], optimize=True)
    _tmp3 = einsum('abil,ljck->abcijk', t2_aaaa, _tmp3, optimize=True)
    result -= 1 * _tmp3
    result += 1 * _tmp3.transpose(0, 1, 2, 4, 3, 5)
    del _tmp3
    _tmp4 = einsum('Qad,Qck->adck', B_aa[:, v, v], B_bb[:, v, o], optimize=True)
    result += 1 * einsum('dbij,adck->abcijk', t2_aaaa, _tmp4, optimize=True)
    del _tmp4
    _tmp5 = einsum('Qad,Qbj->adbj', B_aa[:, v, v], B_aa[:, v, o], optimize=True)
    result += 1 * einsum('dcik,adbj->abcijk', t2_abab, _tmp5, optimize=True)
    del _tmp5
    _tmp6 = einsum('Qaj,Qbd->ajbd', B_aa[:, v, o], B_aa[:, v, v], optimize=True)
    result -= 1 * einsum('dcik,ajbd->abcijk', t2_abab, _tmp6, optimize=True)
    del _tmp6
    _tmp7 = einsum('Qaj,Qcd->ajcd', B_aa[:, v, o], B_bb[:, v, v], optimize=True)
    result -= 1 * einsum('bdik,ajcd->abcijk', t2_abab, _tmp7, optimize=True)
    del _tmp7
    _tmp8 = einsum('Qad,Qbi->adbi', B_aa[:, v, v], B_aa[:, v, o], optimize=True)
    result -= 1 * einsum('dcjk,adbi->abcijk', t2_abab, _tmp8, optimize=True)
    del _tmp8
    _tmp9 = einsum('Qai,Qbd->aibd', B_aa[:, v, o], B_aa[:, v, v], optimize=True)
    result += 1 * einsum('dcjk,aibd->abcijk', t2_abab, _tmp9, optimize=True)
    del _tmp9
    _tmp10 = einsum('Qai,Qcd->aicd', B_aa[:, v, o], B_bb[:, v, v], optimize=True)
    result += 1 * einsum('bdjk,aicd->abcijk', t2_abab, _tmp10, optimize=True)
    del _tmp10
    _tmp11 = einsum('Qbd,Qck->bdck', B_aa[:, v, v], B_bb[:, v, o], optimize=True)
    result -= 1 * einsum('daij,bdck->abcijk', t2_aaaa, _tmp11, optimize=True)
    del _tmp11
    _tmp12 = einsum('Qbj,Qcd->bjcd', B_aa[:, v, o], B_bb[:, v, v], optimize=True)
    result += 1 * einsum('adik,bjcd->abcijk', t2_abab, _tmp12, optimize=True)
    del _tmp12
    _tmp13 = einsum('Qbi,Qcd->bicd', B_aa[:, v, o], B_bb[:, v, v], optimize=True)
    result -= 1 * einsum('adjk,bicd->abcijk', t2_abab, _tmp13, optimize=True)
    del _tmp13
    return result


def t3_2_abbabb_numerator_df(B_aa, B_bb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('Qlj,Qbk->ljbk', B_bb[:, o, o], B_bb[:, v, o], optimize=True)
    result += 1 * einsum('acil,ljbk->abcijk', t2_abab, _tmp0, optimize=True)
    del _tmp0
    _tmp1 = einsum('Qlk,Qbj->lkbj', B_bb[:, o, o], B_bb[:, v, o], optimize=True)
    result -= 1 * einsum('acil,lkbj->abcijk', t2_abab, _tmp1, optimize=True)
    del _tmp1
    _tmp2 = einsum('Qai,Qlk->ailk', B_aa[:, v, o], B_bb[:, o, o], optimize=True)
    result -= 1 * einsum('bcjl,ailk->abcijk', t2_bbbb, _tmp2, optimize=True)
    del _tmp2
    _tmp3 = einsum('Qli,Qbk->libk', B_aa[:, o, o], B_bb[:, v, o], optimize=True)
    result += 1 * einsum('aclj,libk->abcijk', t2_abab, _tmp3, optimize=True)
    del _tmp3
    _tmp4 = einsum('Qai,Qlj->ailj', B_aa[:, v, o], B_bb[:, o, o], optimize=True)
    result += 1 * einsum('bckl,ailj->abcijk', t2_bbbb, _tmp4, optimize=True)
    del _tmp4
    _tmp5 = einsum('Qli,Qbj->libj', B_aa[:, o, o], B_bb[:, v, o], optimize=True)
    result -= 1 * einsum('aclk,libj->abcijk', t2_abab, _tmp5, optimize=True)
    del _tmp5
    _tmp6 = einsum('Qlj,Qck->ljck', B_bb[:, o, o], B_bb[:, v, o], optimize=True)
    result -= 1 * einsum('abil,ljck->abcijk', t2_abab, _tmp6, optimize=True)
    del _tmp6
    _tmp7 = einsum('Qlk,Qcj->lkcj', B_bb[:, o, o], B_bb[:, v, o], optimize=True)
    result += 1 * einsum('abil,lkcj->abcijk', t2_abab, _tmp7, optimize=True)
    del _tmp7
    _tmp8 = einsum('Qli,Qck->lick', B_aa[:, o, o], B_bb[:, v, o], optimize=True)
    result -= 1 * einsum('ablj,lick->abcijk', t2_abab, _tmp8, optimize=True)
    del _tmp8
    _tmp9 = einsum('Qli,Qcj->licj', B_aa[:, o, o], B_bb[:, v, o], optimize=True)
    result += 1 * einsum('ablk,licj->abcijk', t2_abab, _tmp9, optimize=True)
    del _tmp9
    _tmp10 = einsum('Qad,Qbk->adbk', B_aa[:, v, v], B_bb[:, v, o], optimize=True)
    _tmp10 = einsum('dcij,adbk->abcijk', t2_abab, _tmp10, optimize=True)
    result -= 1 * _tmp10
    result += 1 * _tmp10.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp10.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp10.transpose(0, 2, 1, 3, 5, 4)
    del _tmp10
    _tmp11 = einsum('Qai,Qbd->aibd', B_aa[:, v, o], B_bb[:, v, v], optimize=True)
    _tmp11 = einsum('dcjk,aibd->abcijk', t2_bbbb, _tmp11, optimize=True)
    result += 1 * _tmp11
    result -= 1 * _tmp11.transpose(0, 2, 1, 3, 4, 5)
    del _tmp11
    _tmp12 = einsum('Qbd,Qck->bdck', B_bb[:, v, v], B_bb[:, v, o], optimize=True)
    _tmp12 = einsum('adij,bdck->abcijk', t2_abab, _tmp12, optimize=True)
    result += 1 * _tmp12
    result -= 1 * _tmp12.transpose(0, 1, 2, 3, 5, 4)
    del _tmp12
    _tmp13 = einsum('Qbk,Qcd->bkcd', B_bb[:, v, o], B_bb[:, v, v], optimize=True)
    _tmp13 = einsum('adij,bkcd->abcijk', t2_abab, _tmp13, optimize=True)
    result -= 1 * _tmp13
    result += 1 * _tmp13.transpose(0, 1, 2, 3, 5, 4)
    del _tmp13
    return result


def t1_3_aa_numerator_df(B_aa, B_bb, d_aa, o, v, nv, no, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_aa
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    result = np.zeros((nv, no))
    _tmp0 = einsum('Qjb,bj->Q', B_aa[:, o, v], t1_aa, optimize=True)
    result += 1 * einsum('Qai,Q->ai', B_aa[:, v, o], _tmp0, optimize=True)
    del _tmp0
    _tmp1 = einsum('Qji,bj->Qib', B_aa[:, o, o], t1_aa, optimize=True)
    result -= 1 * einsum('Qab,Qib->ai', B_aa[:, v, v], _tmp1, optimize=True)
    del _tmp1
    _tmp2 = einsum('Qjb,bj->Q', B_bb[:, o, v], t1_bb, optimize=True)
    result += 1 * einsum('Qai,Q->ai', B_aa[:, v, o], _tmp2, optimize=True)
    del _tmp2
    _tmp3 = einsum('Qkb,Qji->kbji', B_aa[:, o, v], B_aa[:, o, o], optimize=True)
    result -= 0.5 * einsum('bakj,kbji->ai', t2_aaaa, _tmp3, optimize=True)
    del _tmp3
    _tmp4 = einsum('Qki,Qjb->kijb', B_aa[:, o, o], B_aa[:, o, v], optimize=True)
    result += 0.5 * einsum('bakj,kijb->ai', t2_aaaa, _tmp4, optimize=True)
    del _tmp4
    _tmp5 = einsum('Qki,Qjb->kijb', B_aa[:, o, o], B_bb[:, o, v], optimize=True)
    result -= 1 * einsum('abkj,kijb->ai', t2_abab, _tmp5, optimize=True)
    del _tmp5
    _tmp6 = einsum('Qjb,bcij->Qci', B_aa[:, o, v], t2_aaaa, optimize=True)
    result -= 0.5 * einsum('Qac,Qci->ai', B_aa[:, v, v], _tmp6, optimize=True)
    del _tmp6
    _tmp7 = einsum('Qjc,bcij->Qbi', B_aa[:, o, v], t2_aaaa, optimize=True)
    result += 0.5 * einsum('Qab,Qbi->ai', B_aa[:, v, v], _tmp7, optimize=True)
    del _tmp7
    _tmp8 = einsum('Qjc,bcij->Qbi', B_bb[:, o, v], t2_abab, optimize=True)
    result += 1 * einsum('Qab,Qbi->ai', B_aa[:, v, v], _tmp8, optimize=True)
    del _tmp8
    _tmp9 = einsum('Qkb,Qjc->kbjc', B_aa[:, o, v], B_aa[:, o, v], optimize=True)
    result += 0.25 * einsum('bcaikj,kbjc->ai', t3_aaaaaa, _tmp9, optimize=True)
    del _tmp9
    _tmp10 = einsum('Qkc,Qjb->kcjb', B_aa[:, o, v], B_aa[:, o, v], optimize=True)
    result -= 0.25 * einsum('bcaikj,kcjb->ai', t3_aaaaaa, _tmp10, optimize=True)
    del _tmp10
    _tmp11 = einsum('Qkb,Qjc->kbjc', B_aa[:, o, v], B_bb[:, o, v], optimize=True)
    result -= 0.5 * einsum('bacikj,kbjc->ai', t3_aabaab, _tmp11, optimize=True)
    del _tmp11
    _tmp12 = einsum('Qkc,Qjb->kcjb', B_aa[:, o, v], B_bb[:, o, v], optimize=True)
    result += 0.5 * einsum('acbikj,kcjb->ai', t3_aabaab, _tmp12, optimize=True)
    del _tmp12
    _tmp13 = einsum('Qkb,Qjc->kbjc', B_bb[:, o, v], B_bb[:, o, v], optimize=True)
    result -= 0.25 * einsum('acbikj,kbjc->ai', t3_abbabb, _tmp13, optimize=True)
    del _tmp13
    _tmp14 = einsum('Qkc,Qjb->kcjb', B_bb[:, o, v], B_bb[:, o, v], optimize=True)
    result += 0.25 * einsum('acbikj,kcjb->ai', t3_abbabb, _tmp14, optimize=True)
    del _tmp14
    return result


def t1_3_aa_numerator_no_t3_df(B_aa, B_bb, d_aa, o, v, nv, no, t1_2_aa, t2_2_aaaa, t2_2_abab):
    t1_aa = t1_2_aa
    t1_bb = t1_2_aa
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    result = np.zeros((nv, no))
    _tmp0 = einsum('Qjb,bj->Q', B_aa[:, o, v], t1_aa, optimize=True)
    result += 1 * einsum('Qai,Q->ai', B_aa[:, v, o], _tmp0, optimize=True)
    del _tmp0
    _tmp1 = einsum('Qji,bj->Qib', B_aa[:, o, o], t1_aa, optimize=True)
    result -= 1 * einsum('Qab,Qib->ai', B_aa[:, v, v], _tmp1, optimize=True)
    del _tmp1
    _tmp2 = einsum('Qjb,bj->Q', B_bb[:, o, v], t1_bb, optimize=True)
    result += 1 * einsum('Qai,Q->ai', B_aa[:, v, o], _tmp2, optimize=True)
    del _tmp2
    _tmp3 = einsum('Qkb,Qji->kbji', B_aa[:, o, v], B_aa[:, o, o], optimize=True)
    result -= 0.5 * einsum('bakj,kbji->ai', t2_aaaa, _tmp3, optimize=True)
    del _tmp3
    _tmp4 = einsum('Qki,Qjb->kijb', B_aa[:, o, o], B_aa[:, o, v], optimize=True)
    result += 0.5 * einsum('bakj,kijb->ai', t2_aaaa, _tmp4, optimize=True)
    del _tmp4
    _tmp5 = einsum('Qki,Qjb->kijb', B_aa[:, o, o], B_bb[:, o, v], optimize=True)
    result -= 1 * einsum('abkj,kijb->ai', t2_abab, _tmp5, optimize=True)
    del _tmp5
    _tmp6 = einsum('Qjb,bcij->Qci', B_aa[:, o, v], t2_aaaa, optimize=True)
    result -= 0.5 * einsum('Qac,Qci->ai', B_aa[:, v, v], _tmp6, optimize=True)
    del _tmp6
    _tmp7 = einsum('Qjc,bcij->Qbi', B_aa[:, o, v], t2_aaaa, optimize=True)
    result += 0.5 * einsum('Qab,Qbi->ai', B_aa[:, v, v], _tmp7, optimize=True)
    del _tmp7
    _tmp8 = einsum('Qjc,bcij->Qbi', B_bb[:, o, v], t2_abab, optimize=True)
    result += 1 * einsum('Qab,Qbi->ai', B_aa[:, v, v], _tmp8, optimize=True)
    del _tmp8
    return result

