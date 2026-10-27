# GENERATED CODE -- do not edit by hand.
# Produced by src/CodeGeneration/generate_mpn_density_spin.py (generate_unrestricted).
# Unrestricted (separate alpha/beta) MPn amplitude-numerator, overlap and density
# cross-term pieces (MP2+MP3) of the order-generic pdaggerq recursion.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum


def t2_1_aaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b):
    result = np.zeros((nv_a, nv_a, no_a, no_a))
    _tmp0 = einsum('abij->abij', g_aaaa[v_a, v_a, o_a, o_a])
    result += 1 * _tmp0
    return result


def t2_1_abab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b):
    result = np.zeros((nv_a, nv_b, no_a, no_b))
    _tmp0 = einsum('abij->abij', g_abab[v_a, v_b, o_a, o_b])
    result += 1 * _tmp0
    return result


def t2_1_bbbb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b):
    result = np.zeros((nv_b, nv_b, no_b, no_b))
    _tmp0 = einsum('abij->abij', g_bbbb[v_b, v_b, o_b, o_b])
    result += 1 * _tmp0
    return result


def t1_2_aa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, no_a))
    _tmp0 = einsum('kjbi,bakj->ai', g_aaaa[o_a, o_a, v_a, o_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp0
    _tmp1 = einsum('kjib,abkj->ai', g_abab[o_a, o_b, o_a, v_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp1
    result -= 0.5 * _tmp1
    _tmp2 = einsum('jabc,bcij->ai', g_aaaa[o_a, v_a, v_a, v_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp2
    _tmp3 = einsum('ajbc,bcij->ai', g_abab[v_a, o_b, v_a, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp3
    result += 0.5 * _tmp3
    return result


def t1_2_bb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_b, no_b))
    _tmp0 = einsum('kjbi,bakj->ai', g_abab[o_a, o_b, v_a, o_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp0
    result -= 0.5 * _tmp0
    _tmp1 = einsum('kjbi,bakj->ai', g_bbbb[o_b, o_b, v_b, o_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp1
    _tmp2 = einsum('jabc,bcji->ai', g_abab[o_a, v_b, v_a, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp2
    result += 0.5 * _tmp2
    _tmp3 = einsum('jabc,bcij->ai', g_bbbb[o_b, v_b, v_b, v_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp3
    return result


def t2_2_aaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_a, no_a, no_a))
    _tmp0 = einsum('lkij,ablk->abij', g_aaaa[o_a, o_a, o_a, o_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('kacj,cbik->abij', g_aaaa[o_a, v_a, v_a, o_a], t2_aaaa, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3)
    result -= 1 * _tmp1.transpose(0, 1, 3, 2)
    result += 1 * _tmp1.transpose(1, 0, 3, 2)
    _tmp2 = einsum('akjc,bcik->abij', g_abab[v_a, o_b, o_a, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(1, 0, 2, 3)
    result += 1 * _tmp2.transpose(0, 1, 3, 2)
    result -= 1 * _tmp2.transpose(1, 0, 3, 2)
    _tmp3 = einsum('abcd,cdij->abij', g_aaaa[v_a, v_a, v_a, v_a], t2_aaaa, optimize=True)
    result += 0.5 * _tmp3
    return result


def t2_2_abab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_b, no_a, no_b))
    _tmp0 = einsum('lkij,ablk->abij', g_abab[o_a, o_b, o_a, o_b], t2_abab, optimize=True)
    result += 0.5 * _tmp0
    result += 0.5 * _tmp0
    _tmp1 = einsum('akcj,cbik->abij', g_abab[v_a, o_b, v_a, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp1
    _tmp2 = einsum('kbcj,caik->abij', g_abab[o_a, v_b, v_a, o_b], t2_aaaa, optimize=True)
    result -= 1 * _tmp2
    _tmp3 = einsum('kbcj,acik->abij', g_bbbb[o_b, v_b, v_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp3
    _tmp4 = einsum('kaci,cbkj->abij', g_aaaa[o_a, v_a, v_a, o_a], t2_abab, optimize=True)
    result += 1 * _tmp4
    _tmp5 = einsum('akic,cbjk->abij', g_abab[v_a, o_b, o_a, v_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp5
    _tmp6 = einsum('kbic,ackj->abij', g_abab[o_a, v_b, o_a, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp6
    _tmp7 = einsum('abcd,cdij->abij', g_abab[v_a, v_b, v_a, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp7
    result += 0.5 * _tmp7
    return result


def t2_2_bbbb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_b, nv_b, no_b, no_b))
    _tmp0 = einsum('lkij,ablk->abij', g_bbbb[o_b, o_b, o_b, o_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('kacj,cbki->abij', g_abab[o_a, v_b, v_a, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp1
    result += 1 * _tmp1.transpose(1, 0, 2, 3)
    result += 1 * _tmp1.transpose(0, 1, 3, 2)
    result -= 1 * _tmp1.transpose(1, 0, 3, 2)
    _tmp2 = einsum('kacj,cbik->abij', g_bbbb[o_b, v_b, v_b, o_b], t2_bbbb, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(1, 0, 2, 3)
    result -= 1 * _tmp2.transpose(0, 1, 3, 2)
    result += 1 * _tmp2.transpose(1, 0, 3, 2)
    _tmp3 = einsum('abcd,cdij->abij', g_bbbb[v_b, v_b, v_b, v_b], t2_bbbb, optimize=True)
    result += 0.5 * _tmp3
    return result


def t3_2_aaaaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_a, nv_a, no_a, no_a, no_a))
    _tmp0 = einsum('lajk,bcil->abcijk', g_aaaa[o_a, v_a, o_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result -= 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    _tmp1 = einsum('laij,bckl->abcijk', g_aaaa[o_a, v_a, o_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp1
    result += 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    _tmp2 = einsum('lcjk,abil->abcijk', g_aaaa[o_a, v_a, o_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(0, 1, 2, 4, 3, 5)
    _tmp3 = einsum('lcij,abkl->abcijk', g_aaaa[o_a, v_a, o_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp3
    _tmp4 = einsum('abdk,dcij->abcijk', g_aaaa[v_a, v_a, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp4.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp4.transpose(0, 2, 1, 3, 5, 4)
    _tmp5 = einsum('abdi,dcjk->abcijk', g_aaaa[v_a, v_a, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp5
    result += 1 * _tmp5.transpose(0, 2, 1, 3, 4, 5)
    _tmp6 = einsum('bcdk,daij->abcijk', g_aaaa[v_a, v_a, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp6
    result += 1 * _tmp6.transpose(0, 1, 2, 3, 5, 4)
    _tmp7 = einsum('bcdi,dajk->abcijk', g_aaaa[v_a, v_a, v_a, o_a], t2_aaaa, optimize=True)
    result -= 1 * _tmp7
    return result


def t3_2_aabaab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_a, nv_b, no_a, no_a, no_b))
    _tmp0 = einsum('aljk,bcil->abcijk', g_abab[v_a, o_b, o_a, o_b], t2_abab, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    _tmp1 = einsum('laij,bclk->abcijk', g_aaaa[o_a, v_a, o_a, o_a], t2_abab, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    _tmp2 = einsum('lcjk,abil->abcijk', g_abab[o_a, v_b, o_a, o_b], t2_aaaa, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(0, 1, 2, 4, 3, 5)
    _tmp3 = einsum('acdk,dbij->abcijk', g_abab[v_a, v_b, v_a, o_b], t2_aaaa, optimize=True)
    result += 1 * _tmp3
    _tmp4 = einsum('abdj,dcik->abcijk', g_aaaa[v_a, v_a, v_a, o_a], t2_abab, optimize=True)
    result += 1 * _tmp4
    _tmp5 = einsum('acjd,bdik->abcijk', g_abab[v_a, v_b, o_a, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp5
    _tmp6 = einsum('abdi,dcjk->abcijk', g_aaaa[v_a, v_a, v_a, o_a], t2_abab, optimize=True)
    result -= 1 * _tmp6
    _tmp7 = einsum('acid,bdjk->abcijk', g_abab[v_a, v_b, o_a, v_b], t2_abab, optimize=True)
    result += 1 * _tmp7
    _tmp8 = einsum('bcdk,daij->abcijk', g_abab[v_a, v_b, v_a, o_b], t2_aaaa, optimize=True)
    result -= 1 * _tmp8
    _tmp9 = einsum('bcjd,adik->abcijk', g_abab[v_a, v_b, o_a, v_b], t2_abab, optimize=True)
    result += 1 * _tmp9
    _tmp10 = einsum('bcid,adjk->abcijk', g_abab[v_a, v_b, o_a, v_b], t2_abab, optimize=True)
    result -= 1 * _tmp10
    return result


def t3_2_abbabb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_a, nv_b, nv_b, no_a, no_b, no_b))
    _tmp0 = einsum('lbjk,acil->abcijk', g_bbbb[o_b, v_b, o_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('alik,bcjl->abcijk', g_abab[v_a, o_b, o_a, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp1
    _tmp2 = einsum('lbik,aclj->abcijk', g_abab[o_a, v_b, o_a, o_b], t2_abab, optimize=True)
    result += 1 * _tmp2
    _tmp3 = einsum('alij,bckl->abcijk', g_abab[v_a, o_b, o_a, o_b], t2_bbbb, optimize=True)
    result += 1 * _tmp3
    _tmp4 = einsum('lbij,aclk->abcijk', g_abab[o_a, v_b, o_a, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp4
    _tmp5 = einsum('lcjk,abil->abcijk', g_bbbb[o_b, v_b, o_b, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp5
    _tmp6 = einsum('lcik,ablj->abcijk', g_abab[o_a, v_b, o_a, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp6
    _tmp7 = einsum('lcij,ablk->abcijk', g_abab[o_a, v_b, o_a, o_b], t2_abab, optimize=True)
    result += 1 * _tmp7
    _tmp8 = einsum('abdk,dcij->abcijk', g_abab[v_a, v_b, v_a, o_b], t2_abab, optimize=True)
    result -= 1 * _tmp8
    result += 1 * _tmp8.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp8.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp8.transpose(0, 2, 1, 3, 5, 4)
    _tmp9 = einsum('abid,dcjk->abcijk', g_abab[v_a, v_b, o_a, v_b], t2_bbbb, optimize=True)
    result += 1 * _tmp9
    result -= 1 * _tmp9.transpose(0, 2, 1, 3, 4, 5)
    _tmp10 = einsum('bcdk,adij->abcijk', g_bbbb[v_b, v_b, v_b, o_b], t2_abab, optimize=True)
    result += 1 * _tmp10
    result -= 1 * _tmp10.transpose(0, 1, 2, 3, 5, 4)
    return result


def t3_2_bbbbbb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t2_1_aaaa, t2_1_abab, t2_1_bbbb):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_bbbb
    result = np.zeros((nv_b, nv_b, nv_b, no_b, no_b, no_b))
    _tmp0 = einsum('lajk,bcil->abcijk', g_bbbb[o_b, v_b, o_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result -= 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    _tmp1 = einsum('laij,bckl->abcijk', g_bbbb[o_b, v_b, o_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp1
    result += 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    _tmp2 = einsum('lcjk,abil->abcijk', g_bbbb[o_b, v_b, o_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(0, 1, 2, 4, 3, 5)
    _tmp3 = einsum('lcij,abkl->abcijk', g_bbbb[o_b, v_b, o_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp3
    _tmp4 = einsum('abdk,dcij->abcijk', g_bbbb[v_b, v_b, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp4.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp4.transpose(0, 2, 1, 3, 5, 4)
    _tmp5 = einsum('abdi,dcjk->abcijk', g_bbbb[v_b, v_b, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp5
    result += 1 * _tmp5.transpose(0, 2, 1, 3, 4, 5)
    _tmp6 = einsum('bcdk,daij->abcijk', g_bbbb[v_b, v_b, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp6
    result += 1 * _tmp6.transpose(0, 1, 2, 3, 5, 4)
    _tmp7 = einsum('bcdi,dajk->abcijk', g_bbbb[v_b, v_b, v_b, o_b], t2_bbbb, optimize=True)
    result -= 1 * _tmp7
    return result


def t1_3_aa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_abab, t2_2_bbbb, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t3_2_bbbbbb):
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
    _tmp0 = einsum('jabi,bj->ai', g_aaaa[o_a, v_a, v_a, o_a], t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('ajib,bj->ai', g_abab[v_a, o_b, o_a, v_b], t1_bb, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('kjbi,bakj->ai', g_aaaa[o_a, o_a, v_a, o_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp2
    _tmp3 = einsum('kjib,abkj->ai', g_abab[o_a, o_b, o_a, v_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    _tmp4 = einsum('jabc,bcij->ai', g_aaaa[o_a, v_a, v_a, v_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp4
    _tmp5 = einsum('ajbc,bcij->ai', g_abab[v_a, o_b, v_a, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp5
    result += 0.5 * _tmp5
    _tmp6 = einsum('kjbc,bcaikj->ai', g_aaaa[o_a, o_a, v_a, v_a], t3_aaaaaa, optimize=True)
    result += 0.25 * _tmp6
    _tmp7 = einsum('kjbc,bacikj->ai', g_abab[o_a, o_b, v_a, v_b], t3_aabaab, optimize=True)
    result -= 0.25 * _tmp7
    result -= 0.25 * _tmp7
    _tmp8 = einsum('kjcb,acbikj->ai', g_abab[o_a, o_b, v_a, v_b], t3_aabaab, optimize=True)
    result += 0.25 * _tmp8
    result += 0.25 * _tmp8
    _tmp9 = einsum('kjbc,acbikj->ai', g_bbbb[o_b, o_b, v_b, v_b], t3_abbabb, optimize=True)
    result -= 0.25 * _tmp9
    return result


def t1_3_bb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_abab, t2_2_bbbb, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t3_2_bbbbbb):
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
    _tmp0 = einsum('jabi,bj->ai', g_abab[o_a, v_b, v_a, o_b], t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('jabi,bj->ai', g_bbbb[o_b, v_b, v_b, o_b], t1_bb, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('kjbi,bakj->ai', g_abab[o_a, o_b, v_a, o_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp2
    result -= 0.5 * _tmp2
    _tmp3 = einsum('kjbi,bakj->ai', g_bbbb[o_b, o_b, v_b, o_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('jabc,bcji->ai', g_abab[o_a, v_b, v_a, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp4
    result += 0.5 * _tmp4
    _tmp5 = einsum('jabc,bcij->ai', g_bbbb[o_b, v_b, v_b, v_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp5
    _tmp6 = einsum('kjbc,bcajki->ai', g_aaaa[o_a, o_a, v_a, v_a], t3_aabaab, optimize=True)
    result -= 0.25 * _tmp6
    _tmp7 = einsum('kjbc,bcakij->ai', g_abab[o_a, o_b, v_a, v_b], t3_abbabb, optimize=True)
    result -= 0.25 * _tmp7
    _tmp8 = einsum('jkbc,bcajki->ai', g_abab[o_a, o_b, v_a, v_b], t3_abbabb, optimize=True)
    result += 0.25 * _tmp8
    result -= 0.25 * _tmp7
    result += 0.25 * _tmp8
    _tmp9 = einsum('kjbc,bcaikj->ai', g_bbbb[o_b, o_b, v_b, v_b], t3_bbbbbb, optimize=True)
    result += 0.25 * _tmp9
    return result


def t1_3_aa_numerator_no_t3(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_abab, t2_2_bbbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_bb
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_bbbb
    result = np.zeros((nv_a, no_a))
    _tmp0 = einsum('jabi,bj->ai', g_aaaa[o_a, v_a, v_a, o_a], t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('ajib,bj->ai', g_abab[v_a, o_b, o_a, v_b], t1_bb, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('kjbi,bakj->ai', g_aaaa[o_a, o_a, v_a, o_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp2
    _tmp3 = einsum('kjib,abkj->ai', g_abab[o_a, o_b, o_a, v_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    _tmp4 = einsum('jabc,bcij->ai', g_aaaa[o_a, v_a, v_a, v_a], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp4
    _tmp5 = einsum('ajbc,bcij->ai', g_abab[v_a, o_b, v_a, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp5
    result += 0.5 * _tmp5
    return result


def t1_3_bb_numerator_no_t3(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_abab, t2_2_bbbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_bb
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_bbbb
    result = np.zeros((nv_b, no_b))
    _tmp0 = einsum('jabi,bj->ai', g_abab[o_a, v_b, v_a, o_b], t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('jabi,bj->ai', g_bbbb[o_b, v_b, v_b, o_b], t1_bb, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('kjbi,bakj->ai', g_abab[o_a, o_b, v_a, o_b], t2_abab, optimize=True)
    result -= 0.5 * _tmp2
    result -= 0.5 * _tmp2
    _tmp3 = einsum('kjbi,bakj->ai', g_bbbb[o_b, o_b, v_b, o_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('jabc,bcji->ai', g_abab[o_a, v_b, v_a, v_b], t2_abab, optimize=True)
    result += 0.5 * _tmp4
    result += 0.5 * _tmp4
    _tmp5 = einsum('jabc,bcij->ai', g_bbbb[o_b, v_b, v_b, v_b], t2_bbbb, optimize=True)
    result -= 0.5 * _tmp5
    return result


def overlap2_unrestricted(l_aaaa, l_abab, l_bbbb, t_aaaa, t_abab, t_bbbb):
    l2_aaaa = l_aaaa
    t2_aaaa = t_aaaa
    l2_abab = l_abab
    t2_abab = t_abab
    l2_bbbb = l_bbbb
    t2_bbbb = t_bbbb
    result = np.zeros(())
    _tmp0 = einsum('ijba,baij->', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('ijba,baij->', l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('ijba,baij->', l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    return result


def m2_oo_a_11_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((no_a, no_a))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o_a, o_a], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o_a, o_a], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o_a, o_a], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m2_oo_a_20_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    result = np.zeros((no_a, no_a))
    return result


def m2_oo_a_02_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_2_aa, t_1_2_bb):
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    result = np.zeros((no_a, no_a))
    return result


def m3_oo_a_12_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_1_2_aa, t_1_2_bb, t_2_2_aaaa, t_2_2_abab, t_2_2_bbbb, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_3_2_bbbbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_bbbb
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_bbbbbb
    result = np.zeros((no_a, no_a))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o_a, o_a], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o_a, o_a], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o_a, o_a], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m3_oo_a_21_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb, l_2_2_aaaa, l_2_2_abab, l_2_2_bbbb, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_3_2_bbbbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_bbbb
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_bbbbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((no_a, no_a))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o_a, o_a], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o_a, o_a], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o_a, o_a], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m3_oo_a_30_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_3_aa, l_1_3_bb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_bb
    result = np.zeros((no_a, no_a))
    return result


def m3_oo_a_03_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_3_aa, t_1_3_bb):
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_bb
    result = np.zeros((no_a, no_a))
    return result


def m2_oo_b_11_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((no_b, no_b))
    _tmp0 = einsum('mn,ijba,baij->mn', d_bb[o_b, o_b], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_bb[o_b, o_b], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_bb[o_b, o_b], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    _tmp4 = einsum('inba,baim->mn', l2_bbbb, t2_bbbb, optimize=True)
    result -= 0.5 * _tmp4
    return result


def m2_oo_b_20_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    result = np.zeros((no_b, no_b))
    return result


def m2_oo_b_02_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_2_aa, t_1_2_bb):
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    result = np.zeros((no_b, no_b))
    return result


def m3_oo_b_12_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_1_2_aa, t_1_2_bb, t_2_2_aaaa, t_2_2_abab, t_2_2_bbbb, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_3_2_bbbbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_bbbb
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_bbbbbb
    result = np.zeros((no_b, no_b))
    _tmp0 = einsum('mn,ijba,baij->mn', d_bb[o_b, o_b], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_bb[o_b, o_b], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_bb[o_b, o_b], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    _tmp4 = einsum('inba,baim->mn', l2_bbbb, t2_bbbb, optimize=True)
    result -= 0.5 * _tmp4
    return result


def m3_oo_b_21_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb, l_2_2_aaaa, l_2_2_abab, l_2_2_bbbb, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_3_2_bbbbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_bbbb
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_bbbbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((no_b, no_b))
    _tmp0 = einsum('mn,ijba,baij->mn', d_bb[o_b, o_b], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_bb[o_b, o_b], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_bb[o_b, o_b], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    _tmp4 = einsum('inba,baim->mn', l2_bbbb, t2_bbbb, optimize=True)
    result -= 0.5 * _tmp4
    return result


def m3_oo_b_30_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_3_aa, l_1_3_bb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_bb
    result = np.zeros((no_b, no_b))
    return result


def m3_oo_b_03_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_3_aa, t_1_3_bb):
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_bb
    result = np.zeros((no_b, no_b))
    return result


def m2_vv_a_11_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((nv_a, nv_a))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m2_vv_a_20_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    result = np.zeros((nv_a, nv_a))
    return result


def m2_vv_a_02_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_2_aa, t_1_2_bb):
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    result = np.zeros((nv_a, nv_a))
    return result


def m3_vv_a_12_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_1_2_aa, t_1_2_bb, t_2_2_aaaa, t_2_2_abab, t_2_2_bbbb, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_3_2_bbbbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_bbbb
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_bbbbbb
    result = np.zeros((nv_a, nv_a))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m3_vv_a_21_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb, l_2_2_aaaa, l_2_2_abab, l_2_2_bbbb, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_3_2_bbbbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_bbbb
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_bbbbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((nv_a, nv_a))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m3_vv_a_30_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_3_aa, l_1_3_bb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_bb
    result = np.zeros((nv_a, nv_a))
    return result


def m3_vv_a_03_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_3_aa, t_1_3_bb):
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_bb
    result = np.zeros((nv_a, nv_a))
    return result


def m2_vv_b_11_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((nv_b, nv_b))
    _tmp0 = einsum('ijae,afij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp0
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_bbbb, t2_bbbb, optimize=True)
    result += 0.5 * _tmp1
    return result


def m2_vv_b_20_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    result = np.zeros((nv_b, nv_b))
    return result


def m2_vv_b_02_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_2_aa, t_1_2_bb):
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    result = np.zeros((nv_b, nv_b))
    return result


def m3_vv_b_12_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_1_2_aa, t_1_2_bb, t_2_2_aaaa, t_2_2_abab, t_2_2_bbbb, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_3_2_bbbbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_bbbb
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_bbbbbb
    result = np.zeros((nv_b, nv_b))
    _tmp0 = einsum('ijae,afij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp0
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_bbbb, t2_bbbb, optimize=True)
    result += 0.5 * _tmp1
    return result


def m3_vv_b_21_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb, l_2_2_aaaa, l_2_2_abab, l_2_2_bbbb, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_3_2_bbbbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_bbbb
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_bbbbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((nv_b, nv_b))
    _tmp0 = einsum('ijae,afij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp0
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_bbbb, t2_bbbb, optimize=True)
    result += 0.5 * _tmp1
    return result


def m3_vv_b_30_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_3_aa, l_1_3_bb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_bb
    result = np.zeros((nv_b, nv_b))
    return result


def m3_vv_b_03_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_3_aa, t_1_3_bb):
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_bb
    result = np.zeros((nv_b, nv_b))
    return result


def m2_ov_a_11_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((no_a, nv_a))
    return result


def m2_ov_a_20_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    result = np.zeros((no_a, nv_a))
    return result


def m2_ov_a_02_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_2_aa, t_1_2_bb):
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    result = np.zeros((no_a, nv_a))
    _tmp0 = einsum('em->me', t1_aa)
    result += 1 * _tmp0
    return result


def m3_ov_a_12_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_1_2_aa, t_1_2_bb, t_2_2_aaaa, t_2_2_abab, t_2_2_bbbb, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_3_2_bbbbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_bbbb
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_bbbbbb
    result = np.zeros((no_a, nv_a))
    _tmp0 = einsum('ijba,ebaijm->me', l2_aaaa, t3_aaaaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('ijba,ebaimj->me', l2_abab, t3_aabaab, optimize=True)
    result -= 0.25 * _tmp1
    result -= 0.25 * _tmp1
    _tmp2 = einsum('jiba,ebamji->me', l2_abab, t3_aabaab, optimize=True)
    result += 0.25 * _tmp2
    result += 0.25 * _tmp2
    _tmp3 = einsum('ijba,ebamji->me', l2_bbbb, t3_abbabb, optimize=True)
    result -= 0.25 * _tmp3
    return result


def m3_ov_a_21_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb, l_2_2_aaaa, l_2_2_abab, l_2_2_bbbb, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_3_2_bbbbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_bbbb
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_bbbbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((no_a, nv_a))
    _tmp0 = einsum('ia,eaim->me', l1_aa, t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    _tmp1 = einsum('ia,eami->me', l1_bb, t2_abab, optimize=True)
    result += 1 * _tmp1
    return result


def m3_ov_a_30_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_3_aa, l_1_3_bb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_bb
    result = np.zeros((no_a, nv_a))
    return result


def m3_ov_a_03_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_3_aa, t_1_3_bb):
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_bb
    result = np.zeros((no_a, nv_a))
    _tmp0 = einsum('em->me', t1_aa)
    result += 1 * _tmp0
    return result


def m3_ov_a_12_unrestricted_no_t3(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_1_2_aa, t_1_2_bb, t_2_2_aaaa, t_2_2_abab, t_2_2_bbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_bbbb
    result = np.zeros((no_a, nv_a))
    return result


def m2_ov_b_11_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((no_b, nv_b))
    return result


def m2_ov_b_20_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    result = np.zeros((no_b, nv_b))
    return result


def m2_ov_b_02_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_2_aa, t_1_2_bb):
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    result = np.zeros((no_b, nv_b))
    _tmp0 = einsum('em->me', t1_bb)
    result += 1 * _tmp0
    return result


def m3_ov_b_12_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_1_2_aa, t_1_2_bb, t_2_2_aaaa, t_2_2_abab, t_2_2_bbbb, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_3_2_bbbbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_bbbb
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_bbbbbb
    result = np.zeros((no_b, nv_b))
    _tmp0 = einsum('ijba,abeijm->me', l2_aaaa, t3_aabaab, optimize=True)
    result -= 0.25 * _tmp0
    _tmp1 = einsum('ijba,beaijm->me', l2_abab, t3_abbabb, optimize=True)
    result -= 0.25 * _tmp1
    _tmp2 = einsum('ijab,abeijm->me', l2_abab, t3_abbabb, optimize=True)
    result += 0.25 * _tmp2
    result -= 0.25 * _tmp1
    result += 0.25 * _tmp2
    _tmp3 = einsum('ijba,ebaijm->me', l2_bbbb, t3_bbbbbb, optimize=True)
    result += 0.25 * _tmp3
    return result


def m3_ov_b_21_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_2_aa, l_1_2_bb, l_2_2_aaaa, l_2_2_abab, l_2_2_bbbb, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_3_2_bbbbbb, t_2_1_aaaa, t_2_1_abab, t_2_1_bbbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_bb
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_bbbb
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_bbbbbb
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_bbbb
    result = np.zeros((no_b, nv_b))
    _tmp0 = einsum('ia,aeim->me', l1_aa, t2_abab, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('ia,eaim->me', l1_bb, t2_bbbb, optimize=True)
    result -= 1 * _tmp1
    return result


def m3_ov_b_30_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_1_3_aa, l_1_3_bb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_bb
    result = np.zeros((no_b, nv_b))
    return result


def m3_ov_b_03_unrestricted(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, t_1_3_aa, t_1_3_bb):
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_bb
    result = np.zeros((no_b, nv_b))
    _tmp0 = einsum('em->me', t1_bb)
    result += 1 * _tmp0
    return result


def m3_ov_b_12_unrestricted_no_t3(g_aaaa, g_abab, g_bbbb, d_aa, d_bb, o_a, v_a, o_b, v_b, nv_a, no_a, nv_b, no_b, l_2_1_aaaa, l_2_1_abab, l_2_1_bbbb, t_1_2_aa, t_1_2_bb, t_2_2_aaaa, t_2_2_abab, t_2_2_bbbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_bbbb
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_bb
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_bbbb
    result = np.zeros((no_b, nv_b))
    return result

