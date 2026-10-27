# GENERATED CODE -- do not edit by hand.
# Produced by src/CodeGeneration/generate_mpn_density_spin.py (generate_unrestricted).
# DF/RI twins of the t2_1/t1_2/t2_2/t3_2/t1_3 numerators (UHF).
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum


def t2_1_aaaa_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b):
    result = np.zeros((nv_a, nv_a, no_a, no_a))
    _tmp0 = einsum('Qai,Qbj->abij', B_aa[:, v_a, o_a], B_aa[:, v_a, o_a], optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('Qaj,Qbi->abij', B_aa[:, v_a, o_a], B_aa[:, v_a, o_a], optimize=True)
    result -= 1 * _tmp1
    return result


def t2_1_abab_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b):
    result = np.zeros((nv_a, nv_b, no_a, no_b))
    _tmp0 = einsum('Qai,Qbj->abij', B_aa[:, v_a, o_a], B_bb[:, v_b, o_b], optimize=True)
    result += 1 * _tmp0
    return result


def t2_1_bbbb_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b):
    result = np.zeros((nv_b, nv_b, no_b, no_b))
    _tmp0 = einsum('Qai,Qbj->abij', B_bb[:, v_b, o_b], B_bb[:, v_b, o_b], optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('Qaj,Qbi->abij', B_bb[:, v_b, o_b], B_bb[:, v_b, o_b], optimize=True)
    result -= 1 * _tmp1
    return result


def t1_2_aa_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, no_a))
    _tmp0 = einsum('Qkb,Qji,bakj->ai', B_aa[:, o_a, v_a], B_aa[:, o_a, o_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp0
    _tmp1 = einsum('Qki,Qjb,bakj->ai', B_aa[:, o_a, o_a], B_aa[:, o_a, v_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp1
    _tmp2 = einsum('Qki,Qjb,abkj->ai', B_aa[:, o_a, o_a], B_bb[:, o_b, v_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp2
    result -= 0.5 * _tmp2
    _tmp3 = einsum('Qjb,Qac,bcij->ai', B_aa[:, o_a, v_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('Qjc,Qab,bcij->ai', B_aa[:, o_a, v_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp4
    _tmp5 = einsum('Qab,Qjc,bcij->ai', B_aa[:, v_a, v_a], B_bb[:, o_b, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp5
    result += 0.5 * _tmp5
    return result


def t1_2_bb_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_b, no_b))
    _tmp0 = einsum('Qkb,Qji,bakj->ai', B_aa[:, o_a, v_a], B_bb[:, o_b, o_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp0
    result -= 0.5 * _tmp0
    _tmp1 = einsum('Qkb,Qji,bakj->ai', B_bb[:, o_b, v_b], B_bb[:, o_b, o_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp1
    _tmp2 = einsum('Qki,Qjb,bakj->ai', B_bb[:, o_b, o_b], B_bb[:, o_b, v_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp2
    _tmp3 = einsum('Qjb,Qac,bcji->ai', B_aa[:, o_a, v_a], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp3
    result += 0.5 * _tmp3
    _tmp4 = einsum('Qjb,Qac,bcij->ai', B_bb[:, o_b, v_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp4
    _tmp5 = einsum('Qjc,Qab,bcij->ai', B_bb[:, o_b, v_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp5
    return result


def t2_2_aaaa_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_a, no_a, no_a))
    _tmp0 = einsum('Qli,Qkj,ablk->abij', B_aa[:, o_a, o_a], B_aa[:, o_a, o_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('Qlj,Qki,ablk->abij', B_aa[:, o_a, o_a], B_aa[:, o_a, o_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp1
    _tmp2 = einsum('Qkc,Qaj,cbik->abij', B_aa[:, o_a, v_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(1, 0, 2, 3)
    result -= 1 * _tmp2.transpose(0, 1, 3, 2)
    result += 1 * _tmp2.transpose(1, 0, 3, 2)
    _tmp3 = einsum('Qkj,Qac,cbik->abij', B_aa[:, o_a, o_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp3
    result += 1 * _tmp3.transpose(1, 0, 2, 3)
    result += 1 * _tmp3.transpose(0, 1, 3, 2)
    result -= 1 * _tmp3.transpose(1, 0, 3, 2)
    _tmp4 = einsum('Qaj,Qkc,bcik->abij', B_aa[:, v_a, o_a], B_bb[:, o_b, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(1, 0, 2, 3)
    result += 1 * _tmp4.transpose(0, 1, 3, 2)
    result -= 1 * _tmp4.transpose(1, 0, 3, 2)
    _tmp5 = einsum('Qac,Qbd,cdij->abij', B_aa[:, v_a, v_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp5
    _tmp6 = einsum('Qad,Qbc,cdij->abij', B_aa[:, v_a, v_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp6
    return result


def t2_2_abab_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_b, no_a, no_b))
    _tmp0 = einsum('Qli,Qkj,ablk->abij', B_aa[:, o_a, o_a], B_bb[:, o_b, o_b], t2_abab, optimize=True)
    result += 0.5 * _tmp0
    result += 0.5 * _tmp0
    _tmp1 = einsum('Qac,Qkj,cbik->abij', B_aa[:, v_a, v_a], B_bb[:, o_b, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp1
    _tmp2 = einsum('Qkc,Qbj,caik->abij', B_aa[:, o_a, v_a], B_bb[:, v_b, o_b], t2_aaaa, optimize=True)
    result -= 1 * _tmp2
    _tmp3 = einsum('Qkc,Qbj,acik->abij', B_bb[:, o_b, v_b], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp3
    _tmp4 = einsum('Qkj,Qbc,acik->abij', B_bb[:, o_b, o_b], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp4
    _tmp5 = einsum('Qkc,Qai,cbkj->abij', B_aa[:, o_a, v_a], B_aa[:, v_a, o_a], t2_abab, optimize=True)
    result += 1 * _tmp5
    _tmp6 = einsum('Qki,Qac,cbkj->abij', B_aa[:, o_a, o_a], B_aa[:, v_a, v_a], t2_abab, optimize=True)
    result -= 1 * _tmp6
    _tmp7 = einsum('Qai,Qkc,cbjk->abij', B_aa[:, v_a, o_a], B_bb[:, o_b, v_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp7
    _tmp8 = einsum('Qki,Qbc,ackj->abij', B_aa[:, o_a, o_a], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp8
    _tmp9 = einsum('Qac,Qbd,cdij->abij', B_aa[:, v_a, v_a], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp9
    result += 0.5 * _tmp9
    return result


def t2_2_bbbb_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_b, nv_b, no_b, no_b))
    _tmp0 = einsum('Qli,Qkj,ablk->abij', B_bb[:, o_b, o_b], B_bb[:, o_b, o_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('Qlj,Qki,ablk->abij', B_bb[:, o_b, o_b], B_bb[:, o_b, o_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp1
    _tmp2 = einsum('Qkc,Qaj,cbki->abij', B_aa[:, o_a, v_a], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(1, 0, 2, 3)
    result += 1 * _tmp2.transpose(0, 1, 3, 2)
    result -= 1 * _tmp2.transpose(1, 0, 3, 2)
    _tmp3 = einsum('Qkc,Qaj,cbik->abij', B_bb[:, o_b, v_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3)
    result -= 1 * _tmp3.transpose(0, 1, 3, 2)
    result += 1 * _tmp3.transpose(1, 0, 3, 2)
    _tmp4 = einsum('Qkj,Qac,cbik->abij', B_bb[:, o_b, o_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(1, 0, 2, 3)
    result += 1 * _tmp4.transpose(0, 1, 3, 2)
    result -= 1 * _tmp4.transpose(1, 0, 3, 2)
    _tmp5 = einsum('Qac,Qbd,cdij->abij', B_bb[:, v_b, v_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp5
    _tmp6 = einsum('Qad,Qbc,cdij->abij', B_bb[:, v_b, v_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp6
    return result


def t3_2_aaaaaa_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_a, nv_a, no_a, no_a, no_a))
    _tmp0 = einsum('Qlj,Qak,bcil->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result -= 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    _tmp1 = einsum('Qlk,Qaj,bcil->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp1.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp1.transpose(1, 0, 2, 4, 3, 5)
    _tmp2 = einsum('Qli,Qaj,bckl->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(1, 0, 2, 3, 4, 5)
    _tmp3 = einsum('Qlj,Qai,bckl->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5)
    _tmp4 = einsum('Qlj,Qck,abil->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(0, 1, 2, 4, 3, 5)
    _tmp5 = einsum('Qlk,Qcj,abil->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result += 1 * _tmp5
    result -= 1 * _tmp5.transpose(0, 1, 2, 4, 3, 5)
    _tmp6 = einsum('Qli,Qcj,abkl->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp6
    _tmp7 = einsum('Qlj,Qci,abkl->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result += 1 * _tmp7
    _tmp8 = einsum('Qad,Qbk,dcij->abcijk', B_aa[:, v_a, v_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp8
    result += 1 * _tmp8.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp8.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp8.transpose(0, 2, 1, 3, 5, 4)
    _tmp9 = einsum('Qak,Qbd,dcij->abcijk', B_aa[:, v_a, o_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result += 1 * _tmp9
    result -= 1 * _tmp9.transpose(0, 2, 1, 3, 4, 5)
    result -= 1 * _tmp9.transpose(0, 1, 2, 3, 5, 4)
    result += 1 * _tmp9.transpose(0, 2, 1, 3, 5, 4)
    _tmp10 = einsum('Qad,Qbi,dcjk->abcijk', B_aa[:, v_a, v_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp10
    result += 1 * _tmp10.transpose(0, 2, 1, 3, 4, 5)
    _tmp11 = einsum('Qai,Qbd,dcjk->abcijk', B_aa[:, v_a, o_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result += 1 * _tmp11
    result -= 1 * _tmp11.transpose(0, 2, 1, 3, 4, 5)
    _tmp12 = einsum('Qbd,Qck,daij->abcijk', B_aa[:, v_a, v_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp12
    result += 1 * _tmp12.transpose(0, 1, 2, 3, 5, 4)
    _tmp13 = einsum('Qbk,Qcd,daij->abcijk', B_aa[:, v_a, o_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result += 1 * _tmp13
    result -= 1 * _tmp13.transpose(0, 1, 2, 3, 5, 4)
    _tmp14 = einsum('Qbd,Qci,dajk->abcijk', B_aa[:, v_a, v_a], B_aa[:, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp14
    _tmp15 = einsum('Qbi,Qcd,dajk->abcijk', B_aa[:, v_a, o_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result += 1 * _tmp15
    return result


def t3_2_aabaab_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_a, nv_b, no_a, no_a, no_b))
    _tmp0 = einsum('Qaj,Qlk,bcil->abcijk', B_aa[:, v_a, o_a], B_bb[:, o_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    _tmp1 = einsum('Qli,Qaj,bclk->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_abab, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    _tmp2 = einsum('Qlj,Qai,bclk->abcijk', B_aa[:, o_a, o_a], B_aa[:, v_a, o_a], t2_abab, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(1, 0, 2, 3, 4, 5)
    _tmp3 = einsum('Qlj,Qck,abil->abcijk', B_aa[:, o_a, o_a], B_bb[:, v_b, o_b], t2_aaaa, optimize=True)
    result -= 1 * _tmp3
    result += 1 * _tmp3.transpose(0, 1, 2, 4, 3, 5)
    _tmp4 = einsum('Qad,Qck,dbij->abcijk', B_aa[:, v_a, v_a], B_bb[:, v_b, o_b], t2_aaaa, optimize=True)
    result += 1 * _tmp4
    _tmp5 = einsum('Qad,Qbj,dcik->abcijk', B_aa[:, v_a, v_a], B_aa[:, v_a, o_a], t2_abab, optimize=True)
    result += 1 * _tmp5
    _tmp6 = einsum('Qaj,Qbd,dcik->abcijk', B_aa[:, v_a, o_a], B_aa[:, v_a, v_a], t2_abab, optimize=True)
    result -= 1 * _tmp6
    _tmp7 = einsum('Qaj,Qcd,bdik->abcijk', B_aa[:, v_a, o_a], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp7
    _tmp8 = einsum('Qad,Qbi,dcjk->abcijk', B_aa[:, v_a, v_a], B_aa[:, v_a, o_a], t2_abab, optimize=True)
    result -= 1 * _tmp8
    _tmp9 = einsum('Qai,Qbd,dcjk->abcijk', B_aa[:, v_a, o_a], B_aa[:, v_a, v_a], t2_abab, optimize=True)
    result += 1 * _tmp9
    _tmp10 = einsum('Qai,Qcd,bdjk->abcijk', B_aa[:, v_a, o_a], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result += 1 * _tmp10
    _tmp11 = einsum('Qbd,Qck,daij->abcijk', B_aa[:, v_a, v_a], B_bb[:, v_b, o_b], t2_aaaa, optimize=True)
    result -= 1 * _tmp11
    _tmp12 = einsum('Qbj,Qcd,adik->abcijk', B_aa[:, v_a, o_a], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result += 1 * _tmp12
    _tmp13 = einsum('Qbi,Qcd,adjk->abcijk', B_aa[:, v_a, o_a], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp13
    return result


def t3_2_abbabb_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_b, nv_b, no_a, no_b, no_b))
    _tmp0 = einsum('Qlj,Qbk,acil->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('Qlk,Qbj,acil->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp1
    _tmp2 = einsum('Qai,Qlk,bcjl->abcijk', B_aa[:, v_a, o_a], B_bb[:, o_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp2
    _tmp3 = einsum('Qli,Qbk,aclj->abcijk', B_aa[:, o_a, o_a], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp3
    _tmp4 = einsum('Qai,Qlj,bckl->abcijk', B_aa[:, v_a, o_a], B_bb[:, o_b, o_b], t2_bbbb, optimize=True)
    result += 1 * _tmp4
    _tmp5 = einsum('Qli,Qbj,aclk->abcijk', B_aa[:, o_a, o_a], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp5
    _tmp6 = einsum('Qlj,Qck,abil->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp6
    _tmp7 = einsum('Qlk,Qcj,abil->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp7
    _tmp8 = einsum('Qli,Qck,ablj->abcijk', B_aa[:, o_a, o_a], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp8
    _tmp9 = einsum('Qli,Qcj,ablk->abcijk', B_aa[:, o_a, o_a], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp9
    _tmp10 = einsum('Qad,Qbk,dcij->abcijk', B_aa[:, v_a, v_a], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp10
    result += 1 * _tmp10.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp10.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp10.transpose(0, 2, 1, 3, 5, 4)
    _tmp11 = einsum('Qai,Qbd,dcjk->abcijk', B_aa[:, v_a, o_a], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result += 1 * _tmp11
    result -= 1 * _tmp11.transpose(0, 2, 1, 3, 4, 5)
    _tmp12 = einsum('Qbd,Qck,adij->abcijk', B_bb[:, v_b, v_b], B_bb[:, v_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp12
    result -= 1 * _tmp12.transpose(0, 1, 2, 3, 5, 4)
    _tmp13 = einsum('Qbk,Qcd,adij->abcijk', B_bb[:, v_b, o_b], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp13
    result += 1 * _tmp13.transpose(0, 1, 2, 3, 5, 4)
    return result


def t3_2_bbbbbb_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_b, nv_b, nv_b, no_b, no_b, no_b))
    _tmp0 = einsum('Qlj,Qak,bcil->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result -= 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    _tmp1 = einsum('Qlk,Qaj,bcil->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp1.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp1.transpose(1, 0, 2, 4, 3, 5)
    _tmp2 = einsum('Qli,Qaj,bckl->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(1, 0, 2, 3, 4, 5)
    _tmp3 = einsum('Qlj,Qai,bckl->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5)
    _tmp4 = einsum('Qlj,Qck,abil->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(0, 1, 2, 4, 3, 5)
    _tmp5 = einsum('Qlk,Qcj,abil->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result += 1 * _tmp5
    result -= 1 * _tmp5.transpose(0, 1, 2, 4, 3, 5)
    _tmp6 = einsum('Qli,Qcj,abkl->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp6
    _tmp7 = einsum('Qlj,Qci,abkl->abcijk', B_bb[:, o_b, o_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result += 1 * _tmp7
    _tmp8 = einsum('Qad,Qbk,dcij->abcijk', B_bb[:, v_b, v_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp8
    result += 1 * _tmp8.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp8.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp8.transpose(0, 2, 1, 3, 5, 4)
    _tmp9 = einsum('Qak,Qbd,dcij->abcijk', B_bb[:, v_b, o_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result += 1 * _tmp9
    result -= 1 * _tmp9.transpose(0, 2, 1, 3, 4, 5)
    result -= 1 * _tmp9.transpose(0, 1, 2, 3, 5, 4)
    result += 1 * _tmp9.transpose(0, 2, 1, 3, 5, 4)
    _tmp10 = einsum('Qad,Qbi,dcjk->abcijk', B_bb[:, v_b, v_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp10
    result += 1 * _tmp10.transpose(0, 2, 1, 3, 4, 5)
    _tmp11 = einsum('Qai,Qbd,dcjk->abcijk', B_bb[:, v_b, o_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result += 1 * _tmp11
    result -= 1 * _tmp11.transpose(0, 2, 1, 3, 4, 5)
    _tmp12 = einsum('Qbd,Qck,daij->abcijk', B_bb[:, v_b, v_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp12
    result += 1 * _tmp12.transpose(0, 1, 2, 3, 5, 4)
    _tmp13 = einsum('Qbk,Qcd,daij->abcijk', B_bb[:, v_b, o_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result += 1 * _tmp13
    result -= 1 * _tmp13.transpose(0, 1, 2, 3, 5, 4)
    _tmp14 = einsum('Qbd,Qci,dajk->abcijk', B_bb[:, v_b, v_b], B_bb[:, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp14
    _tmp15 = einsum('Qbi,Qcd,dajk->abcijk', B_bb[:, v_b, o_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result += 1 * _tmp15
    return result


def t1_3_aa_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_abab, t2_2_bbbb, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t3_2_bbbbbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_bb
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_bbbb
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_bbbbbb
    result = np.zeros((nv_a, no_a))
    _tmp0 = einsum('Qjb,Qai,bj->ai', B_aa[:, o_a, v_a], B_aa[:, v_a, o_a], t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('Qji,Qab,bj->ai', B_aa[:, o_a, o_a], B_aa[:, v_a, v_a], t1_aa, optimize=True)
    result -= 1 * _tmp1
    _tmp2 = einsum('Qai,Qjb,bj->ai', B_aa[:, v_a, o_a], B_bb[:, o_b, v_b], t1_bb, optimize=True)
    result += 1 * _tmp2
    _tmp3 = einsum('Qkb,Qji,bakj->ai', B_aa[:, o_a, v_a], B_aa[:, o_a, o_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('Qki,Qjb,bakj->ai', B_aa[:, o_a, o_a], B_aa[:, o_a, v_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp4
    _tmp5 = einsum('Qki,Qjb,abkj->ai', B_aa[:, o_a, o_a], B_bb[:, o_b, v_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp5
    result -= 0.5 * _tmp5
    _tmp6 = einsum('Qjb,Qac,bcij->ai', B_aa[:, o_a, v_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp6
    _tmp7 = einsum('Qjc,Qab,bcij->ai', B_aa[:, o_a, v_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp7
    _tmp8 = einsum('Qab,Qjc,bcij->ai', B_aa[:, v_a, v_a], B_bb[:, o_b, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp8
    result += 0.5 * _tmp8
    _tmp9 = einsum('Qkb,Qjc,bcaikj->ai', B_aa[:, o_a, v_a], B_aa[:, o_a, v_a], t3_aaaaaa, optimize=True)
    result += 0.25 * _tmp9
    _tmp10 = einsum('Qkc,Qjb,bcaikj->ai', B_aa[:, o_a, v_a], B_aa[:, o_a, v_a], t3_aaaaaa, optimize=True)
    result -= 0.25 * _tmp10
    _tmp11 = einsum('Qkb,Qjc,bacikj->ai', B_aa[:, o_a, v_a], B_bb[:, o_b, v_b], t3_aabaab, optimize=True)
    result -= 0.25 * _tmp11
    result -= 0.25 * _tmp11
    _tmp12 = einsum('Qkc,Qjb,acbikj->ai', B_aa[:, o_a, v_a], B_bb[:, o_b, v_b], t3_aabaab, optimize=True)
    result += 0.25 * _tmp12
    result += 0.25 * _tmp12
    _tmp13 = einsum('Qkb,Qjc,acbikj->ai', B_bb[:, o_b, v_b], B_bb[:, o_b, v_b], t3_abbabb, optimize=True)
    result -= 0.25 * _tmp13
    _tmp14 = einsum('Qkc,Qjb,acbikj->ai', B_bb[:, o_b, v_b], B_bb[:, o_b, v_b], t3_abbabb, optimize=True)
    result += 0.25 * _tmp14
    return result


def t1_3_bb_numerator_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_abab, t2_2_bbbb, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t3_2_bbbbbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_bb
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_bbbb
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_bbbbbb
    result = np.zeros((nv_b, no_b))
    _tmp0 = einsum('Qjb,Qai,bj->ai', B_aa[:, o_a, v_a], B_bb[:, v_b, o_b], t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('Qjb,Qai,bj->ai', B_bb[:, o_b, v_b], B_bb[:, v_b, o_b], t1_bb, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('Qji,Qab,bj->ai', B_bb[:, o_b, o_b], B_bb[:, v_b, v_b], t1_bb, optimize=True)
    result -= 1 * _tmp2
    _tmp3 = einsum('Qkb,Qji,bakj->ai', B_aa[:, o_a, v_a], B_bb[:, o_b, o_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    _tmp4 = einsum('Qkb,Qji,bakj->ai', B_bb[:, o_b, v_b], B_bb[:, o_b, o_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp4
    _tmp5 = einsum('Qki,Qjb,bakj->ai', B_bb[:, o_b, o_b], B_bb[:, o_b, v_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp5
    _tmp6 = einsum('Qjb,Qac,bcji->ai', B_aa[:, o_a, v_a], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp6
    result += 0.5 * _tmp6
    _tmp7 = einsum('Qjb,Qac,bcij->ai', B_bb[:, o_b, v_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp7
    _tmp8 = einsum('Qjc,Qab,bcij->ai', B_bb[:, o_b, v_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp8
    _tmp9 = einsum('Qkb,Qjc,bcajki->ai', B_aa[:, o_a, v_a], B_aa[:, o_a, v_a], t3_aabaab, optimize=True)
    result -= 0.25 * _tmp9
    _tmp10 = einsum('Qkc,Qjb,bcajki->ai', B_aa[:, o_a, v_a], B_aa[:, o_a, v_a], t3_aabaab, optimize=True)
    result += 0.25 * _tmp10
    _tmp11 = einsum('Qkb,Qjc,bcakij->ai', B_aa[:, o_a, v_a], B_bb[:, o_b, v_b], t3_abbabb, optimize=True)
    result -= 0.25 * _tmp11
    _tmp12 = einsum('Qjb,Qkc,bcajki->ai', B_aa[:, o_a, v_a], B_bb[:, o_b, v_b], t3_abbabb, optimize=True)
    result += 0.25 * _tmp12
    result -= 0.25 * _tmp11
    result += 0.25 * _tmp12
    _tmp13 = einsum('Qkb,Qjc,bcaikj->ai', B_bb[:, o_b, v_b], B_bb[:, o_b, v_b], t3_bbbbbb, optimize=True)
    result += 0.25 * _tmp13
    _tmp14 = einsum('Qkc,Qjb,bcaikj->ai', B_bb[:, o_b, v_b], B_bb[:, o_b, v_b], t3_bbbbbb, optimize=True)
    result -= 0.25 * _tmp14
    return result


def t1_3_aa_numerator_no_t3_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_abab, t2_2_bbbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_bb
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_bbbb
    result = np.zeros((nv_a, no_a))
    _tmp0 = einsum('Qjb,Qai,bj->ai', B_aa[:, o_a, v_a], B_aa[:, v_a, o_a], t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('Qji,Qab,bj->ai', B_aa[:, o_a, o_a], B_aa[:, v_a, v_a], t1_aa, optimize=True)
    result -= 1 * _tmp1
    _tmp2 = einsum('Qai,Qjb,bj->ai', B_aa[:, v_a, o_a], B_bb[:, o_b, v_b], t1_bb, optimize=True)
    result += 1 * _tmp2
    _tmp3 = einsum('Qkb,Qji,bakj->ai', B_aa[:, o_a, v_a], B_aa[:, o_a, o_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('Qki,Qjb,bakj->ai', B_aa[:, o_a, o_a], B_aa[:, o_a, v_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp4
    _tmp5 = einsum('Qki,Qjb,abkj->ai', B_aa[:, o_a, o_a], B_bb[:, o_b, v_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp5
    result -= 0.5 * _tmp5
    _tmp6 = einsum('Qjb,Qac,bcij->ai', B_aa[:, o_a, v_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp6
    _tmp7 = einsum('Qjc,Qab,bcij->ai', B_aa[:, o_a, v_a], B_aa[:, v_a, v_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp7
    _tmp8 = einsum('Qab,Qjc,bcij->ai', B_aa[:, v_a, v_a], B_bb[:, o_b, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp8
    result += 0.5 * _tmp8
    return result


def t1_3_bb_numerator_no_t3_df(B_aa, B_bb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_abab, t2_2_bbbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_bb
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_bbbb
    result = np.zeros((nv_b, no_b))
    _tmp0 = einsum('Qjb,Qai,bj->ai', B_aa[:, o_a, v_a], B_bb[:, v_b, o_b], t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('Qjb,Qai,bj->ai', B_bb[:, o_b, v_b], B_bb[:, v_b, o_b], t1_bb, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('Qji,Qab,bj->ai', B_bb[:, o_b, o_b], B_bb[:, v_b, v_b], t1_bb, optimize=True)
    result -= 1 * _tmp2
    _tmp3 = einsum('Qkb,Qji,bakj->ai', B_aa[:, o_a, v_a], B_bb[:, o_b, o_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    _tmp4 = einsum('Qkb,Qji,bakj->ai', B_bb[:, o_b, v_b], B_bb[:, o_b, o_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp4
    _tmp5 = einsum('Qki,Qjb,bakj->ai', B_bb[:, o_b, o_b], B_bb[:, o_b, v_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp5
    _tmp6 = einsum('Qjb,Qac,bcji->ai', B_aa[:, o_a, v_a], B_bb[:, v_b, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp6
    result += 0.5 * _tmp6
    _tmp7 = einsum('Qjb,Qac,bcij->ai', B_bb[:, o_b, v_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp7
    _tmp8 = einsum('Qjc,Qab,bcij->ai', B_bb[:, o_b, v_b], B_bb[:, v_b, v_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp8
    return result

