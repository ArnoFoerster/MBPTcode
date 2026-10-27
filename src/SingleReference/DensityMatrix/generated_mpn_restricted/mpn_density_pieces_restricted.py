# GENERATED CODE -- do not edit by hand.
# Produced by src/CodeGeneration/generate_mpn_density_spin.py (generate_restricted).
# Restricted (spin-blocked) MPn amplitude-numerator, overlap and density cross-term
# pieces of the order-generic pdaggerq recursion (generate_mpn_density.py).
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum


def t2_1_aaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no):
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('abij->abij', g_aaaa[v, v, o, o])
    result += 1 * _tmp0
    del _tmp0
    return result


def t2_1_abab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no):
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('abij->abij', g_abab[v, v, o, o])
    result += 1 * _tmp0
    del _tmp0
    return result


def t1_2_aa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, no))
    _tmp0 = einsum('kjbi,bakj->ai', g_aaaa[o, o, v, o], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp0
    del _tmp0
    _tmp1 = einsum('kjib,abkj->ai', g_abab[o, o, o, v], t2_abab, optimize=True)
    result -= 0.5 * _tmp1
    result -= 0.5 * _tmp1
    del _tmp1
    _tmp2 = einsum('jabc,bcij->ai', g_aaaa[o, v, v, v], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp2
    del _tmp2
    _tmp3 = einsum('ajbc,bcij->ai', g_abab[v, o, v, v], t2_abab, optimize=True)
    result += 0.5 * _tmp3
    result += 0.5 * _tmp3
    del _tmp3
    return result


def t2_2_aaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('lkij,ablk->abij', g_aaaa[o, o, o, o], t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    del _tmp0
    _tmp1 = einsum('kacj,cbik->abij', g_aaaa[o, v, v, o], t2_aaaa, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3)
    result -= 1 * _tmp1.transpose(0, 1, 3, 2)
    result += 1 * _tmp1.transpose(1, 0, 3, 2)
    del _tmp1
    _tmp2 = einsum('akjc,bcik->abij', g_abab[v, o, o, v], t2_abab, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(1, 0, 2, 3)
    result += 1 * _tmp2.transpose(0, 1, 3, 2)
    result -= 1 * _tmp2.transpose(1, 0, 3, 2)
    del _tmp2
    _tmp3 = einsum('abcd,cdij->abij', g_aaaa[v, v, v, v], t2_aaaa, optimize=True)
    result += 0.5 * _tmp3
    del _tmp3
    return result


def t2_2_abab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('lkij,ablk->abij', g_abab[o, o, o, o], t2_abab, optimize=True)
    result += 0.5 * _tmp0
    result += 0.5 * _tmp0
    del _tmp0
    _tmp1 = einsum('akcj,cbik->abij', g_abab[v, o, v, o], t2_abab, optimize=True)
    result -= 1 * _tmp1
    del _tmp1
    _tmp2 = einsum('kbcj,caik->abij', g_abab[o, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp2
    del _tmp2
    _tmp3 = einsum('kbcj,acik->abij', g_bbbb[o, v, v, o], t2_abab, optimize=True)
    result += 1 * _tmp3
    del _tmp3
    _tmp4 = einsum('kaci,cbkj->abij', g_aaaa[o, v, v, o], t2_abab, optimize=True)
    result += 1 * _tmp4
    del _tmp4
    _tmp5 = einsum('akic,cbjk->abij', g_abab[v, o, o, v], t2_bbbb, optimize=True)
    result -= 1 * _tmp5
    del _tmp5
    _tmp6 = einsum('kbic,ackj->abij', g_abab[o, v, o, v], t2_abab, optimize=True)
    result -= 1 * _tmp6
    del _tmp6
    _tmp7 = einsum('abcd,cdij->abij', g_abab[v, v, v, v], t2_abab, optimize=True)
    result += 0.5 * _tmp7
    result += 0.5 * _tmp7
    del _tmp7
    return result


def t3_2_aaaaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('lajk,bcil->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result -= 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    del _tmp0
    _tmp1 = einsum('laij,bckl->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp1
    result += 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    del _tmp1
    _tmp2 = einsum('lcjk,abil->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(0, 1, 2, 4, 3, 5)
    del _tmp2
    _tmp3 = einsum('lcij,abkl->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp3
    del _tmp3
    _tmp4 = einsum('abdk,dcij->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp4.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp4.transpose(0, 2, 1, 3, 5, 4)
    del _tmp4
    _tmp5 = einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp5
    result += 1 * _tmp5.transpose(0, 2, 1, 3, 4, 5)
    del _tmp5
    _tmp6 = einsum('bcdk,daij->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp6
    result += 1 * _tmp6.transpose(0, 1, 2, 3, 5, 4)
    del _tmp6
    _tmp7 = einsum('bcdi,dajk->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp7
    del _tmp7
    return result


def t3_2_aabaab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('aljk,bcil->abcijk', g_abab[v, o, o, o], t2_abab, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    del _tmp0
    _tmp1 = einsum('laij,bclk->abcijk', g_aaaa[o, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    del _tmp1
    _tmp2 = einsum('lcjk,abil->abcijk', g_abab[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(0, 1, 2, 4, 3, 5)
    del _tmp2
    _tmp3 = einsum('acdk,dbij->abcijk', g_abab[v, v, v, o], t2_aaaa, optimize=True)
    result += 1 * _tmp3
    del _tmp3
    _tmp4 = einsum('abdj,dcik->abcijk', g_aaaa[v, v, v, o], t2_abab, optimize=True)
    result += 1 * _tmp4
    del _tmp4
    _tmp5 = einsum('acjd,bdik->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    result -= 1 * _tmp5
    del _tmp5
    _tmp6 = einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_abab, optimize=True)
    result -= 1 * _tmp6
    del _tmp6
    _tmp7 = einsum('acid,bdjk->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    result += 1 * _tmp7
    del _tmp7
    _tmp8 = einsum('bcdk,daij->abcijk', g_abab[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp8
    del _tmp8
    _tmp9 = einsum('bcjd,adik->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    result += 1 * _tmp9
    del _tmp9
    _tmp10 = einsum('bcid,adjk->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    result -= 1 * _tmp10
    del _tmp10
    return result


def t3_2_abbabb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('lbjk,acil->abcijk', g_bbbb[o, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp0
    del _tmp0
    _tmp1 = einsum('alik,bcjl->abcijk', g_abab[v, o, o, o], t2_bbbb, optimize=True)
    result -= 1 * _tmp1
    del _tmp1
    _tmp2 = einsum('lbik,aclj->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp2
    del _tmp2
    _tmp3 = einsum('alij,bckl->abcijk', g_abab[v, o, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp3
    del _tmp3
    _tmp4 = einsum('lbij,aclk->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp4
    del _tmp4
    _tmp5 = einsum('lcjk,abil->abcijk', g_bbbb[o, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp5
    del _tmp5
    _tmp6 = einsum('lcik,ablj->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp6
    del _tmp6
    _tmp7 = einsum('lcij,ablk->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp7
    del _tmp7
    _tmp8 = einsum('abdk,dcij->abcijk', g_abab[v, v, v, o], t2_abab, optimize=True)
    result -= 1 * _tmp8
    result += 1 * _tmp8.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp8.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp8.transpose(0, 2, 1, 3, 5, 4)
    del _tmp8
    _tmp9 = einsum('abid,dcjk->abcijk', g_abab[v, v, o, v], t2_bbbb, optimize=True)
    result += 1 * _tmp9
    result -= 1 * _tmp9.transpose(0, 2, 1, 3, 4, 5)
    del _tmp9
    _tmp10 = einsum('bcdk,adij->abcijk', g_bbbb[v, v, v, o], t2_abab, optimize=True)
    result += 1 * _tmp10
    result -= 1 * _tmp10.transpose(0, 1, 2, 3, 5, 4)
    del _tmp10
    return result


def t4_2_aaaaaaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('abkl,cdij->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp0.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp0.transpose(0, 2, 1, 3, 4, 6, 5, 7)
    del _tmp0
    _tmp1 = einsum('abil,cdjk->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp1.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 7, 6)
    del _tmp1
    _tmp2 = einsum('abjk,cdil->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp2.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp2.transpose(0, 2, 1, 3, 6, 5, 4, 7)
    del _tmp2
    _tmp3 = einsum('adkl,bcij->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp3.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp3.transpose(1, 0, 2, 3, 4, 6, 5, 7)
    del _tmp3
    _tmp4 = einsum('adil,bcjk->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp4
    result -= 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp4.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp4
    _tmp5 = einsum('adjk,bcil->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp5
    result -= 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp5.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp5.transpose(1, 0, 2, 3, 6, 5, 4, 7)
    del _tmp5
    _tmp6 = einsum('bckl,adij->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp6
    result -= 1 * _tmp6.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp6.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp6.transpose(0, 3, 2, 1, 4, 6, 5, 7)
    del _tmp6
    _tmp7 = einsum('bcil,adjk->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp7
    result -= 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp7.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 7, 6)
    del _tmp7
    _tmp8 = einsum('bcjk,adil->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp8
    result -= 1 * _tmp8.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp8.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp8.transpose(0, 3, 2, 1, 6, 5, 4, 7)
    del _tmp8
    return result


def t4_2_aaabaaab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('abik,cdjl->abcdijkl', g_aaaa[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp0
    _tmp1 = einsum('abjk,cdil->abcdijkl', g_aaaa[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp1.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp1.transpose(0, 2, 1, 3, 6, 5, 4, 7)
    del _tmp1
    _tmp2 = einsum('adkl,bcij->abcdijkl', g_abab[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp2.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp2.transpose(1, 0, 2, 3, 4, 6, 5, 7)
    del _tmp2
    _tmp3 = einsum('adil,bcjk->abcdijkl', g_abab[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp3
    _tmp4 = einsum('cdkl,abij->abcdijkl', g_abab[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp4
    result -= 1 * _tmp4.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp4
    _tmp5 = einsum('cdil,abjk->abcdijkl', g_abab[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp5
    del _tmp5
    _tmp6 = einsum('bcik,adjl->abcdijkl', g_aaaa[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp6
    del _tmp6
    _tmp7 = einsum('bcjk,adil->abcdijkl', g_aaaa[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp7
    result -= 1 * _tmp7.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    del _tmp7
    return result


def t4_2_aabbaabb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('acjl,bdik->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp0
    del _tmp0
    _tmp1 = einsum('acil,bdjk->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp1
    result += 1 * _tmp1.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp1
    _tmp2 = einsum('acjk,bdil->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp2
    del _tmp2
    _tmp3 = einsum('abji,cdkl->abcdijkl', g_aaaa[v, v, o, o], t2_bbbb, optimize=True)
    result -= 1 * _tmp3
    del _tmp3
    _tmp4 = einsum('adjl,bcik->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp4
    _tmp5 = einsum('adil,bcjk->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp5
    result -= 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp5.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp5
    _tmp6 = einsum('adjk,bcil->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp6
    result -= 1 * _tmp6.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp6
    _tmp7 = einsum('dckl,abij->abcdijkl', g_bbbb[v, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp7
    del _tmp7
    _tmp8 = einsum('bcjl,adik->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp8
    del _tmp8
    _tmp9 = einsum('bcil,adjk->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp9
    result -= 1 * _tmp9.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp9
    _tmp10 = einsum('bcjk,adil->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp10
    del _tmp10
    return result


def t4_2_abbbabbb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_1_aaaa, t2_1_abab):
    t2_aaaa = t2_1_aaaa
    t2_abab = t2_1_abab
    t2_bbbb = t2_1_aaaa
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('abil,cdjk->abcdijkl', g_abab[v, v, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp0.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 7, 6)
    del _tmp0
    _tmp1 = einsum('abij,cdkl->abcdijkl', g_abab[v, v, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp1
    _tmp2 = einsum('bdkl,acij->abcdijkl', g_bbbb[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp2
    _tmp3 = einsum('adil,bcjk->abcdijkl', g_abab[v, v, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp3
    _tmp4 = einsum('bdjk,acil->abcdijkl', g_bbbb[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp4
    del _tmp4
    _tmp5 = einsum('adij,bckl->abcdijkl', g_abab[v, v, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp5
    del _tmp5
    _tmp6 = einsum('bckl,adij->abcdijkl', g_bbbb[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp6
    result -= 1 * _tmp6.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp6.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp6.transpose(0, 3, 2, 1, 4, 6, 5, 7)
    del _tmp6
    _tmp7 = einsum('bcjk,adil->abcdijkl', g_bbbb[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp7
    result -= 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    del _tmp7
    return result


def t1_3_aa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb):
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
    _tmp0 = einsum('jabi,bj->ai', g_aaaa[o, v, v, o], t1_aa, optimize=True)
    result += 1 * _tmp0
    del _tmp0
    _tmp1 = einsum('ajib,bj->ai', g_abab[v, o, o, v], t1_bb, optimize=True)
    result += 1 * _tmp1
    del _tmp1
    _tmp2 = einsum('kjbi,bakj->ai', g_aaaa[o, o, v, o], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp2
    del _tmp2
    _tmp3 = einsum('kjib,abkj->ai', g_abab[o, o, o, v], t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    del _tmp3
    _tmp4 = einsum('jabc,bcij->ai', g_aaaa[o, v, v, v], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp4
    del _tmp4
    _tmp5 = einsum('ajbc,bcij->ai', g_abab[v, o, v, v], t2_abab, optimize=True)
    result += 0.5 * _tmp5
    result += 0.5 * _tmp5
    del _tmp5
    _tmp6 = einsum('kjbc,bcaikj->ai', g_aaaa[o, o, v, v], t3_aaaaaa, optimize=True)
    result += 0.25 * _tmp6
    del _tmp6
    _tmp7 = einsum('kjbc,bacikj->ai', g_abab[o, o, v, v], t3_aabaab, optimize=True)
    result -= 0.25 * _tmp7
    result -= 0.25 * _tmp7
    del _tmp7
    _tmp8 = einsum('kjcb,acbikj->ai', g_abab[o, o, v, v], t3_aabaab, optimize=True)
    result += 0.25 * _tmp8
    result += 0.25 * _tmp8
    del _tmp8
    _tmp9 = einsum('kjbc,acbikj->ai', g_bbbb[o, o, v, v], t3_abbabb, optimize=True)
    result -= 0.25 * _tmp9
    del _tmp9
    return result


def t2_3_aaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_aa
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    t4_aaaaaaaa = t4_2_aaaaaaaa
    t4_aaabaaab = t4_2_aaabaaab
    t4_aabbaabb = t4_2_aabbaabb
    t4_abbbabbb = t4_2_abbbabbb
    t4_bbbbbbbb = t4_2_aaaaaaaa
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('kaij,bk->abij', g_aaaa[o, v, o, o], t1_aa, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(1, 0, 2, 3)
    del _tmp0
    _tmp1 = einsum('abcj,ci->abij', g_aaaa[v, v, v, o], t1_aa, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 1, 3, 2)
    del _tmp1
    _tmp2 = einsum('lkij,ablk->abij', g_aaaa[o, o, o, o], t2_aaaa, optimize=True)
    result += 0.5 * _tmp2
    del _tmp2
    _tmp3 = einsum('kacj,cbik->abij', g_aaaa[o, v, v, o], t2_aaaa, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3)
    result -= 1 * _tmp3.transpose(0, 1, 3, 2)
    result += 1 * _tmp3.transpose(1, 0, 3, 2)
    del _tmp3
    _tmp4 = einsum('akjc,bcik->abij', g_abab[v, o, o, v], t2_abab, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(1, 0, 2, 3)
    result += 1 * _tmp4.transpose(0, 1, 3, 2)
    result -= 1 * _tmp4.transpose(1, 0, 3, 2)
    del _tmp4
    _tmp5 = einsum('abcd,cdij->abij', g_aaaa[v, v, v, v], t2_aaaa, optimize=True)
    result += 0.5 * _tmp5
    del _tmp5
    _tmp6 = einsum('lkcj,cabilk->abij', g_aaaa[o, o, v, o], t3_aaaaaa, optimize=True)
    result += 0.5 * _tmp6
    result -= 0.5 * _tmp6.transpose(0, 1, 3, 2)
    del _tmp6
    _tmp7 = einsum('lkjc,bacilk->abij', g_abab[o, o, o, v], t3_aabaab, optimize=True)
    result += 0.5 * _tmp7
    result -= 0.5 * _tmp7.transpose(0, 1, 3, 2)
    result += 0.5 * _tmp7
    result -= 0.5 * _tmp7.transpose(0, 1, 3, 2)
    del _tmp7
    _tmp8 = einsum('kacd,cdbijk->abij', g_aaaa[o, v, v, v], t3_aaaaaa, optimize=True)
    result += 0.5 * _tmp8
    result -= 0.5 * _tmp8.transpose(1, 0, 2, 3)
    del _tmp8
    _tmp9 = einsum('akcd,cbdijk->abij', g_abab[v, o, v, v], t3_aabaab, optimize=True)
    result += 0.5 * _tmp9
    result -= 0.5 * _tmp9.transpose(1, 0, 2, 3)
    del _tmp9
    _tmp10 = einsum('akdc,bdcijk->abij', g_abab[v, o, v, v], t3_aabaab, optimize=True)
    result -= 0.5 * _tmp10
    result += 0.5 * _tmp10.transpose(1, 0, 2, 3)
    del _tmp10
    _tmp11 = einsum('lkcd,cdabijlk->abij', g_aaaa[o, o, v, v], t4_aaaaaaaa, optimize=True)
    result += 0.25 * _tmp11
    del _tmp11
    _tmp12 = einsum('lkcd,cbadijlk->abij', g_abab[o, o, v, v], t4_aaabaaab, optimize=True)
    result -= 0.25 * _tmp12
    result -= 0.25 * _tmp12
    del _tmp12
    _tmp13 = einsum('lkdc,bdacijlk->abij', g_abab[o, o, v, v], t4_aaabaaab, optimize=True)
    result += 0.25 * _tmp13
    result += 0.25 * _tmp13
    del _tmp13
    _tmp14 = einsum('lkcd,abcdijlk->abij', g_bbbb[o, o, v, v], t4_aabbaabb, optimize=True)
    result += 0.25 * _tmp14
    del _tmp14
    return result


def t2_3_abab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_aa
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    t4_aaaaaaaa = t4_2_aaaaaaaa
    t4_aaabaaab = t4_2_aaabaaab
    t4_aabbaabb = t4_2_aabbaabb
    t4_abbbabbb = t4_2_abbbabbb
    t4_bbbbbbbb = t4_2_aaaaaaaa
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('akij,bk->abij', g_abab[v, o, o, o], t1_bb, optimize=True)
    result -= 1 * _tmp0
    del _tmp0
    _tmp1 = einsum('kbij,ak->abij', g_abab[o, v, o, o], t1_aa, optimize=True)
    result -= 1 * _tmp1
    del _tmp1
    _tmp2 = einsum('abcj,ci->abij', g_abab[v, v, v, o], t1_aa, optimize=True)
    result += 1 * _tmp2
    del _tmp2
    _tmp3 = einsum('abic,cj->abij', g_abab[v, v, o, v], t1_bb, optimize=True)
    result += 1 * _tmp3
    del _tmp3
    _tmp4 = einsum('lkij,ablk->abij', g_abab[o, o, o, o], t2_abab, optimize=True)
    result += 0.5 * _tmp4
    result += 0.5 * _tmp4
    del _tmp4
    _tmp5 = einsum('akcj,cbik->abij', g_abab[v, o, v, o], t2_abab, optimize=True)
    result -= 1 * _tmp5
    del _tmp5
    _tmp6 = einsum('kbcj,caik->abij', g_abab[o, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp6
    del _tmp6
    _tmp7 = einsum('kbcj,acik->abij', g_bbbb[o, v, v, o], t2_abab, optimize=True)
    result += 1 * _tmp7
    del _tmp7
    _tmp8 = einsum('kaci,cbkj->abij', g_aaaa[o, v, v, o], t2_abab, optimize=True)
    result += 1 * _tmp8
    del _tmp8
    _tmp9 = einsum('akic,cbjk->abij', g_abab[v, o, o, v], t2_bbbb, optimize=True)
    result -= 1 * _tmp9
    del _tmp9
    _tmp10 = einsum('kbic,ackj->abij', g_abab[o, v, o, v], t2_abab, optimize=True)
    result -= 1 * _tmp10
    del _tmp10
    _tmp11 = einsum('abcd,cdij->abij', g_abab[v, v, v, v], t2_abab, optimize=True)
    result += 0.5 * _tmp11
    result += 0.5 * _tmp11
    del _tmp11
    _tmp12 = einsum('lkcj,cabilk->abij', g_abab[o, o, v, o], t3_aabaab, optimize=True)
    result += 0.5 * _tmp12
    result += 0.5 * _tmp12
    del _tmp12
    _tmp13 = einsum('lkcj,acbilk->abij', g_bbbb[o, o, v, o], t3_abbabb, optimize=True)
    result -= 0.5 * _tmp13
    del _tmp13
    _tmp14 = einsum('lkci,cabklj->abij', g_aaaa[o, o, v, o], t3_aabaab, optimize=True)
    result += 0.5 * _tmp14
    del _tmp14
    _tmp15 = einsum('lkic,acbljk->abij', g_abab[o, o, o, v], t3_abbabb, optimize=True)
    result += 0.5 * _tmp15
    del _tmp15
    _tmp16 = einsum('klic,acbklj->abij', g_abab[o, o, o, v], t3_abbabb, optimize=True)
    result -= 0.5 * _tmp16
    del _tmp16
    _tmp17 = einsum('kacd,cdbikj->abij', g_aaaa[o, v, v, v], t3_aabaab, optimize=True)
    result -= 0.5 * _tmp17
    del _tmp17
    _tmp18 = einsum('akcd,cdbijk->abij', g_abab[v, o, v, v], t3_abbabb, optimize=True)
    result -= 0.5 * _tmp18
    result -= 0.5 * _tmp18
    del _tmp18
    _tmp19 = einsum('kbcd,cadikj->abij', g_abab[o, v, v, v], t3_aabaab, optimize=True)
    result -= 0.5 * _tmp19
    del _tmp19
    _tmp20 = einsum('kbdc,adcikj->abij', g_abab[o, v, v, v], t3_aabaab, optimize=True)
    result += 0.5 * _tmp20
    del _tmp20
    _tmp21 = einsum('kbcd,adcijk->abij', g_bbbb[o, v, v, v], t3_abbabb, optimize=True)
    result += 0.5 * _tmp21
    del _tmp21
    _tmp22 = einsum('lkcd,cdabiklj->abij', g_aaaa[o, o, v, v], t4_aaabaaab, optimize=True)
    result -= 0.25 * _tmp22
    del _tmp22
    _tmp23 = einsum('lkcd,cadbiljk->abij', g_abab[o, o, v, v], t4_aabbaabb, optimize=True)
    result += 0.25 * _tmp23
    del _tmp23
    _tmp24 = einsum('klcd,cadbiklj->abij', g_abab[o, o, v, v], t4_aabbaabb, optimize=True)
    result -= 0.25 * _tmp24
    del _tmp24
    _tmp25 = einsum('lkdc,adcbiljk->abij', g_abab[o, o, v, v], t4_aabbaabb, optimize=True)
    result -= 0.25 * _tmp25
    del _tmp25
    _tmp26 = einsum('kldc,adcbiklj->abij', g_abab[o, o, v, v], t4_aabbaabb, optimize=True)
    result += 0.25 * _tmp26
    del _tmp26
    _tmp27 = einsum('lkcd,adcbijlk->abij', g_bbbb[o, o, v, v], t4_abbbabbb, optimize=True)
    result -= 0.25 * _tmp27
    del _tmp27
    return result


def t3_3_aaaaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_aa
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    t4_aaaaaaaa = t4_2_aaaaaaaa
    t4_aaabaaab = t4_2_aaabaaab
    t4_aabbaabb = t4_2_aabbaabb
    t4_abbbabbb = t4_2_abbbabbb
    t4_bbbbbbbb = t4_2_aaaaaaaa
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('abjk,ci->abcijk', g_aaaa[v, v, o, o], t1_aa, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5)
    result -= 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp0.transpose(0, 2, 1, 4, 3, 5)
    del _tmp0
    _tmp1 = einsum('abij,ck->abcijk', g_aaaa[v, v, o, o], t1_aa, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5)
    del _tmp1
    _tmp2 = einsum('bcjk,ai->abcijk', g_aaaa[v, v, o, o], t1_aa, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(0, 1, 2, 4, 3, 5)
    del _tmp2
    _tmp3 = einsum('bcij,ak->abcijk', g_aaaa[v, v, o, o], t1_aa, optimize=True)
    result += 1 * _tmp3
    del _tmp3
    _tmp4 = einsum('lajk,bcil->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp4.transpose(0, 1, 2, 4, 3, 5)
    result -= 1 * _tmp4.transpose(1, 0, 2, 4, 3, 5)
    del _tmp4
    _tmp5 = einsum('laij,bckl->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp5
    result += 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5)
    del _tmp5
    _tmp6 = einsum('lcjk,abil->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp6
    result += 1 * _tmp6.transpose(0, 1, 2, 4, 3, 5)
    del _tmp6
    _tmp7 = einsum('lcij,abkl->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp7
    del _tmp7
    _tmp8 = einsum('abdk,dcij->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp8
    result += 1 * _tmp8.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp8.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp8.transpose(0, 2, 1, 3, 5, 4)
    del _tmp8
    _tmp9 = einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp9
    result += 1 * _tmp9.transpose(0, 2, 1, 3, 4, 5)
    del _tmp9
    _tmp10 = einsum('bcdk,daij->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp10
    result += 1 * _tmp10.transpose(0, 1, 2, 3, 5, 4)
    del _tmp10
    _tmp11 = einsum('bcdi,dajk->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp11
    del _tmp11
    _tmp12 = einsum('mljk,abciml->abcijk', g_aaaa[o, o, o, o], t3_aaaaaa, optimize=True)
    result += 0.5 * _tmp12
    result -= 0.5 * _tmp12.transpose(0, 1, 2, 4, 3, 5)
    del _tmp12
    _tmp13 = einsum('mlij,abckml->abcijk', g_aaaa[o, o, o, o], t3_aaaaaa, optimize=True)
    result += 0.5 * _tmp13
    del _tmp13
    _tmp14 = einsum('ladk,dbcijl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp14
    result -= 1 * _tmp14.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp14.transpose(0, 1, 2, 3, 5, 4)
    result += 1 * _tmp14.transpose(1, 0, 2, 3, 5, 4)
    del _tmp14
    _tmp15 = einsum('alkd,cbdijl->abcijk', g_abab[v, o, o, v], t3_aabaab, optimize=True)
    result -= 1 * _tmp15
    result += 1 * _tmp15.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp15.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp15.transpose(1, 0, 2, 3, 5, 4)
    del _tmp15
    _tmp16 = einsum('ladi,dbcjkl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp16
    result -= 1 * _tmp16.transpose(1, 0, 2, 3, 4, 5)
    del _tmp16
    _tmp17 = einsum('alid,cbdjkl->abcijk', g_abab[v, o, o, v], t3_aabaab, optimize=True)
    result -= 1 * _tmp17
    result += 1 * _tmp17.transpose(1, 0, 2, 3, 4, 5)
    del _tmp17
    _tmp18 = einsum('lcdk,dabijl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp18
    result -= 1 * _tmp18.transpose(0, 1, 2, 3, 5, 4)
    del _tmp18
    _tmp19 = einsum('clkd,badijl->abcijk', g_abab[v, o, o, v], t3_aabaab, optimize=True)
    result -= 1 * _tmp19
    result += 1 * _tmp19.transpose(0, 1, 2, 3, 5, 4)
    del _tmp19
    _tmp20 = einsum('lcdi,dabjkl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp20
    del _tmp20
    _tmp21 = einsum('clid,badjkl->abcijk', g_abab[v, o, o, v], t3_aabaab, optimize=True)
    result -= 1 * _tmp21
    del _tmp21
    _tmp22 = einsum('abde,decijk->abcijk', g_aaaa[v, v, v, v], t3_aaaaaa, optimize=True)
    result += 0.5 * _tmp22
    result -= 0.5 * _tmp22.transpose(0, 2, 1, 3, 4, 5)
    del _tmp22
    _tmp23 = einsum('bcde,deaijk->abcijk', g_aaaa[v, v, v, v], t3_aaaaaa, optimize=True)
    result += 0.5 * _tmp23
    del _tmp23
    _tmp24 = einsum('mldk,dabcijml->abcijk', g_aaaa[o, o, v, o], t4_aaaaaaaa, optimize=True)
    result -= 0.5 * _tmp24
    result += 0.5 * _tmp24.transpose(0, 1, 2, 3, 5, 4)
    del _tmp24
    _tmp25 = einsum('mlkd,cabdijml->abcijk', g_abab[o, o, o, v], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp25
    result += 0.5 * _tmp25.transpose(0, 1, 2, 3, 5, 4)
    result -= 0.5 * _tmp25
    result += 0.5 * _tmp25.transpose(0, 1, 2, 3, 5, 4)
    del _tmp25
    _tmp26 = einsum('mldi,dabcjkml->abcijk', g_aaaa[o, o, v, o], t4_aaaaaaaa, optimize=True)
    result -= 0.5 * _tmp26
    del _tmp26
    _tmp27 = einsum('mlid,cabdjkml->abcijk', g_abab[o, o, o, v], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp27
    result -= 0.5 * _tmp27
    del _tmp27
    _tmp28 = einsum('lade,debcijkl->abcijk', g_aaaa[o, v, v, v], t4_aaaaaaaa, optimize=True)
    result -= 0.5 * _tmp28
    result += 0.5 * _tmp28.transpose(1, 0, 2, 3, 4, 5)
    del _tmp28
    _tmp29 = einsum('alde,dcbeijkl->abcijk', g_abab[v, o, v, v], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp29
    result += 0.5 * _tmp29.transpose(1, 0, 2, 3, 4, 5)
    del _tmp29
    _tmp30 = einsum('aled,cebdijkl->abcijk', g_abab[v, o, v, v], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp30
    result -= 0.5 * _tmp30.transpose(1, 0, 2, 3, 4, 5)
    del _tmp30
    _tmp31 = einsum('lcde,deabijkl->abcijk', g_aaaa[o, v, v, v], t4_aaaaaaaa, optimize=True)
    result -= 0.5 * _tmp31
    del _tmp31
    _tmp32 = einsum('clde,dbaeijkl->abcijk', g_abab[v, o, v, v], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp32
    del _tmp32
    _tmp33 = einsum('cled,beadijkl->abcijk', g_abab[v, o, v, v], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp33
    del _tmp33
    return result


def t3_3_aabaab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_aa
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    t4_aaaaaaaa = t4_2_aaaaaaaa
    t4_aaabaaab = t4_2_aaabaaab
    t4_aabbaabb = t4_2_aabbaabb
    t4_abbbabbb = t4_2_abbbabbb
    t4_bbbbbbbb = t4_2_aaaaaaaa
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('acjk,bi->abcijk', g_abab[v, v, o, o], t1_aa, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    del _tmp0
    _tmp1 = einsum('abij,ck->abcijk', g_aaaa[v, v, o, o], t1_bb, optimize=True)
    result += 1 * _tmp1
    del _tmp1
    _tmp2 = einsum('bcjk,ai->abcijk', g_abab[v, v, o, o], t1_aa, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(0, 1, 2, 4, 3, 5)
    del _tmp2
    _tmp3 = einsum('aljk,bcil->abcijk', g_abab[v, o, o, o], t2_abab, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp3.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp3.transpose(1, 0, 2, 4, 3, 5)
    del _tmp3
    _tmp4 = einsum('laij,bclk->abcijk', g_aaaa[o, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp4
    result -= 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5)
    del _tmp4
    _tmp5 = einsum('lcjk,abil->abcijk', g_abab[o, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp5
    result += 1 * _tmp5.transpose(0, 1, 2, 4, 3, 5)
    del _tmp5
    _tmp6 = einsum('acdk,dbij->abcijk', g_abab[v, v, v, o], t2_aaaa, optimize=True)
    result += 1 * _tmp6
    del _tmp6
    _tmp7 = einsum('abdj,dcik->abcijk', g_aaaa[v, v, v, o], t2_abab, optimize=True)
    result += 1 * _tmp7
    del _tmp7
    _tmp8 = einsum('acjd,bdik->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    result -= 1 * _tmp8
    del _tmp8
    _tmp9 = einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_abab, optimize=True)
    result -= 1 * _tmp9
    del _tmp9
    _tmp10 = einsum('acid,bdjk->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    result += 1 * _tmp10
    del _tmp10
    _tmp11 = einsum('bcdk,daij->abcijk', g_abab[v, v, v, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp11
    del _tmp11
    _tmp12 = einsum('bcjd,adik->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    result += 1 * _tmp12
    del _tmp12
    _tmp13 = einsum('bcid,adjk->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    result -= 1 * _tmp13
    del _tmp13
    _tmp14 = einsum('mljk,abciml->abcijk', g_abab[o, o, o, o], t3_aabaab, optimize=True)
    result += 0.5 * _tmp14
    result -= 0.5 * _tmp14.transpose(0, 1, 2, 4, 3, 5)
    result += 0.5 * _tmp14
    result -= 0.5 * _tmp14.transpose(0, 1, 2, 4, 3, 5)
    del _tmp14
    _tmp15 = einsum('mlij,abclmk->abcijk', g_aaaa[o, o, o, o], t3_aabaab, optimize=True)
    result -= 0.5 * _tmp15
    del _tmp15
    _tmp16 = einsum('aldk,dbcijl->abcijk', g_abab[v, o, v, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp16
    result += 1 * _tmp16.transpose(1, 0, 2, 3, 4, 5)
    del _tmp16
    _tmp17 = einsum('ladj,dbcilk->abcijk', g_aaaa[o, v, v, o], t3_aabaab, optimize=True)
    result += 1 * _tmp17
    result -= 1 * _tmp17.transpose(1, 0, 2, 3, 4, 5)
    del _tmp17
    _tmp18 = einsum('aljd,bdcikl->abcijk', g_abab[v, o, o, v], t3_abbabb, optimize=True)
    result += 1 * _tmp18
    result -= 1 * _tmp18.transpose(1, 0, 2, 3, 4, 5)
    del _tmp18
    _tmp19 = einsum('ladi,dbcjlk->abcijk', g_aaaa[o, v, v, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp19
    result += 1 * _tmp19.transpose(1, 0, 2, 3, 4, 5)
    del _tmp19
    _tmp20 = einsum('alid,bdcjkl->abcijk', g_abab[v, o, o, v], t3_abbabb, optimize=True)
    result -= 1 * _tmp20
    result += 1 * _tmp20.transpose(1, 0, 2, 3, 4, 5)
    del _tmp20
    _tmp21 = einsum('lcdk,dabijl->abcijk', g_abab[o, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp21
    del _tmp21
    _tmp22 = einsum('lcdk,badijl->abcijk', g_bbbb[o, v, v, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp22
    del _tmp22
    _tmp23 = einsum('lcjd,badilk->abcijk', g_abab[o, v, o, v], t3_aabaab, optimize=True)
    result += 1 * _tmp23
    del _tmp23
    _tmp24 = einsum('lcid,badjlk->abcijk', g_abab[o, v, o, v], t3_aabaab, optimize=True)
    result -= 1 * _tmp24
    del _tmp24
    _tmp25 = einsum('abde,decijk->abcijk', g_aaaa[v, v, v, v], t3_aabaab, optimize=True)
    result += 0.5 * _tmp25
    del _tmp25
    _tmp26 = einsum('acde,dbeijk->abcijk', g_abab[v, v, v, v], t3_aabaab, optimize=True)
    result += 0.5 * _tmp26
    del _tmp26
    _tmp27 = einsum('aced,bedijk->abcijk', g_abab[v, v, v, v], t3_aabaab, optimize=True)
    result -= 0.5 * _tmp27
    del _tmp27
    _tmp28 = einsum('bcde,daeijk->abcijk', g_abab[v, v, v, v], t3_aabaab, optimize=True)
    result -= 0.5 * _tmp28
    del _tmp28
    _tmp29 = einsum('bced,aedijk->abcijk', g_abab[v, v, v, v], t3_aabaab, optimize=True)
    result += 0.5 * _tmp29
    del _tmp29
    _tmp30 = einsum('mldk,dabcijml->abcijk', g_abab[o, o, v, o], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp30
    result -= 0.5 * _tmp30
    del _tmp30
    _tmp31 = einsum('mldk,badcijml->abcijk', g_bbbb[o, o, v, o], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp31
    del _tmp31
    _tmp32 = einsum('mldj,dabcilmk->abcijk', g_aaaa[o, o, v, o], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp32
    del _tmp32
    _tmp33 = einsum('mljd,badcimkl->abcijk', g_abab[o, o, o, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp33
    del _tmp33
    _tmp34 = einsum('lmjd,badcilmk->abcijk', g_abab[o, o, o, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp34
    del _tmp34
    _tmp35 = einsum('mldi,dabcjlmk->abcijk', g_aaaa[o, o, v, o], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp35
    del _tmp35
    _tmp36 = einsum('mlid,badcjmkl->abcijk', g_abab[o, o, o, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp36
    del _tmp36
    _tmp37 = einsum('lmid,badcjlmk->abcijk', g_abab[o, o, o, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp37
    del _tmp37
    _tmp38 = einsum('lade,debcijlk->abcijk', g_aaaa[o, v, v, v], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp38
    result -= 0.5 * _tmp38.transpose(1, 0, 2, 3, 4, 5)
    del _tmp38
    _tmp39 = einsum('alde,dbecijkl->abcijk', g_abab[v, o, v, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp39
    result += 0.5 * _tmp39.transpose(1, 0, 2, 3, 4, 5)
    del _tmp39
    _tmp40 = einsum('aled,bedcijkl->abcijk', g_abab[v, o, v, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp40
    result -= 0.5 * _tmp40.transpose(1, 0, 2, 3, 4, 5)
    del _tmp40
    _tmp41 = einsum('lcde,dbaeijlk->abcijk', g_abab[o, v, v, v], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp41
    del _tmp41
    _tmp42 = einsum('lced,beadijlk->abcijk', g_abab[o, v, v, v], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp42
    del _tmp42
    _tmp43 = einsum('lcde,abdeijkl->abcijk', g_bbbb[o, v, v, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp43
    del _tmp43
    return result


def t3_3_abbabb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb):
    t1_aa = t1_2_aa
    t1_bb = t1_2_aa
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    t4_aaaaaaaa = t4_2_aaaaaaaa
    t4_aaabaaab = t4_2_aaabaaab
    t4_aabbaabb = t4_2_aabbaabb
    t4_abbbabbb = t4_2_abbbabbb
    t4_bbbbbbbb = t4_2_aaaaaaaa
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('abik,cj->abcijk', g_abab[v, v, o, o], t1_bb, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5)
    del _tmp0
    _tmp1 = einsum('abij,ck->abcijk', g_abab[v, v, o, o], t1_bb, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5)
    del _tmp1
    _tmp2 = einsum('bcjk,ai->abcijk', g_bbbb[v, v, o, o], t1_aa, optimize=True)
    result += 1 * _tmp2
    del _tmp2
    _tmp3 = einsum('lbjk,acil->abcijk', g_bbbb[o, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp3
    del _tmp3
    _tmp4 = einsum('alik,bcjl->abcijk', g_abab[v, o, o, o], t2_bbbb, optimize=True)
    result -= 1 * _tmp4
    del _tmp4
    _tmp5 = einsum('lbik,aclj->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp5
    del _tmp5
    _tmp6 = einsum('alij,bckl->abcijk', g_abab[v, o, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp6
    del _tmp6
    _tmp7 = einsum('lbij,aclk->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp7
    del _tmp7
    _tmp8 = einsum('lcjk,abil->abcijk', g_bbbb[o, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp8
    del _tmp8
    _tmp9 = einsum('lcik,ablj->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp9
    del _tmp9
    _tmp10 = einsum('lcij,ablk->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp10
    del _tmp10
    _tmp11 = einsum('abdk,dcij->abcijk', g_abab[v, v, v, o], t2_abab, optimize=True)
    result -= 1 * _tmp11
    result += 1 * _tmp11.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp11.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp11.transpose(0, 2, 1, 3, 5, 4)
    del _tmp11
    _tmp12 = einsum('abid,dcjk->abcijk', g_abab[v, v, o, v], t2_bbbb, optimize=True)
    result += 1 * _tmp12
    result -= 1 * _tmp12.transpose(0, 2, 1, 3, 4, 5)
    del _tmp12
    _tmp13 = einsum('bcdk,adij->abcijk', g_bbbb[v, v, v, o], t2_abab, optimize=True)
    result += 1 * _tmp13
    result -= 1 * _tmp13.transpose(0, 1, 2, 3, 5, 4)
    del _tmp13
    _tmp14 = einsum('mljk,abciml->abcijk', g_bbbb[o, o, o, o], t3_abbabb, optimize=True)
    result += 0.5 * _tmp14
    del _tmp14
    _tmp15 = einsum('mlik,abcmjl->abcijk', g_abab[o, o, o, o], t3_abbabb, optimize=True)
    result += 0.5 * _tmp15
    del _tmp15
    _tmp16 = einsum('lmik,abclmj->abcijk', g_abab[o, o, o, o], t3_abbabb, optimize=True)
    result -= 0.5 * _tmp16
    del _tmp16
    _tmp17 = einsum('mlij,abcmkl->abcijk', g_abab[o, o, o, o], t3_abbabb, optimize=True)
    result -= 0.5 * _tmp17
    del _tmp17
    _tmp18 = einsum('lmij,abclmk->abcijk', g_abab[o, o, o, o], t3_abbabb, optimize=True)
    result += 0.5 * _tmp18
    del _tmp18
    _tmp19 = einsum('aldk,dbcijl->abcijk', g_abab[v, o, v, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp19
    result += 1 * _tmp19.transpose(0, 1, 2, 3, 5, 4)
    del _tmp19
    _tmp20 = einsum('lbdk,dacilj->abcijk', g_abab[o, v, v, o], t3_aabaab, optimize=True)
    result += 1 * _tmp20
    result -= 1 * _tmp20.transpose(0, 1, 2, 3, 5, 4)
    del _tmp20
    _tmp21 = einsum('lbdk,adcijl->abcijk', g_bbbb[o, v, v, o], t3_abbabb, optimize=True)
    result += 1 * _tmp21
    result -= 1 * _tmp21.transpose(0, 1, 2, 3, 5, 4)
    del _tmp21
    _tmp22 = einsum('ladi,dbclkj->abcijk', g_aaaa[o, v, v, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp22
    del _tmp22
    _tmp23 = einsum('alid,dbcjkl->abcijk', g_abab[v, o, o, v], t3_bbbbbb, optimize=True)
    result += 1 * _tmp23
    del _tmp23
    _tmp24 = einsum('lbid,adclkj->abcijk', g_abab[o, v, o, v], t3_abbabb, optimize=True)
    result += 1 * _tmp24
    del _tmp24
    _tmp25 = einsum('lcdk,dabilj->abcijk', g_abab[o, v, v, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp25
    result += 1 * _tmp25.transpose(0, 1, 2, 3, 5, 4)
    del _tmp25
    _tmp26 = einsum('lcdk,adbijl->abcijk', g_bbbb[o, v, v, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp26
    result += 1 * _tmp26.transpose(0, 1, 2, 3, 5, 4)
    del _tmp26
    _tmp27 = einsum('lcid,adblkj->abcijk', g_abab[o, v, o, v], t3_abbabb, optimize=True)
    result -= 1 * _tmp27
    del _tmp27
    _tmp28 = einsum('abde,decijk->abcijk', g_abab[v, v, v, v], t3_abbabb, optimize=True)
    result += 0.5 * _tmp28
    result -= 0.5 * _tmp28.transpose(0, 2, 1, 3, 4, 5)
    result += 0.5 * _tmp28
    result -= 0.5 * _tmp28.transpose(0, 2, 1, 3, 4, 5)
    del _tmp28
    _tmp29 = einsum('bcde,aedijk->abcijk', g_bbbb[v, v, v, v], t3_abbabb, optimize=True)
    result -= 0.5 * _tmp29
    del _tmp29
    _tmp30 = einsum('mldk,dabcimjl->abcijk', g_abab[o, o, v, o], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp30
    result -= 0.5 * _tmp30.transpose(0, 1, 2, 3, 5, 4)
    del _tmp30
    _tmp31 = einsum('lmdk,dabcilmj->abcijk', g_abab[o, o, v, o], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp31
    result += 0.5 * _tmp31.transpose(0, 1, 2, 3, 5, 4)
    del _tmp31
    _tmp32 = einsum('mldk,adbcijml->abcijk', g_bbbb[o, o, v, o], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp32
    result -= 0.5 * _tmp32.transpose(0, 1, 2, 3, 5, 4)
    del _tmp32
    _tmp33 = einsum('mldi,dabcmljk->abcijk', g_aaaa[o, o, v, o], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp33
    del _tmp33
    _tmp34 = einsum('mlid,adbcmkjl->abcijk', g_abab[o, o, o, v], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp34
    del _tmp34
    _tmp35 = einsum('lmid,adbclkmj->abcijk', g_abab[o, o, o, v], t4_abbbabbb, optimize=True)
    result -= 0.5 * _tmp35
    del _tmp35
    _tmp36 = einsum('lade,debcilkj->abcijk', g_aaaa[o, v, v, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp36
    del _tmp36
    _tmp37 = einsum('alde,debcijkl->abcijk', g_abab[v, o, v, v], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp37
    result += 0.5 * _tmp37
    del _tmp37
    _tmp38 = einsum('lbde,daecilkj->abcijk', g_abab[o, v, v, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp38
    del _tmp38
    _tmp39 = einsum('lbed,aedcilkj->abcijk', g_abab[o, v, v, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp39
    del _tmp39
    _tmp40 = einsum('lbde,aedcijkl->abcijk', g_bbbb[o, v, v, v], t4_abbbabbb, optimize=True)
    result -= 0.5 * _tmp40
    del _tmp40
    _tmp41 = einsum('lcde,daebilkj->abcijk', g_abab[o, v, v, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp41
    del _tmp41
    _tmp42 = einsum('lced,aedbilkj->abcijk', g_abab[o, v, v, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp42
    del _tmp42
    _tmp43 = einsum('lcde,aedbijkl->abcijk', g_bbbb[o, v, v, v], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp43
    del _tmp43
    return result


def t4_3_aaaaaaaa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb):
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    t4_aaaaaaaa = t4_2_aaaaaaaa
    t4_aaabaaab = t4_2_aaabaaab
    t4_aabbaabb = t4_2_aabbaabb
    t4_abbbabbb = t4_2_abbbabbb
    t4_bbbbbbbb = t4_2_aaaaaaaa
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('abkl,cdij->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp0.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp0.transpose(0, 2, 1, 3, 4, 6, 5, 7)
    del _tmp0
    _tmp1 = einsum('abil,cdjk->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp1.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 7, 6)
    del _tmp1
    _tmp2 = einsum('abjk,cdil->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp2.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp2.transpose(0, 2, 1, 3, 6, 5, 4, 7)
    del _tmp2
    _tmp3 = einsum('adkl,bcij->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp3.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp3.transpose(1, 0, 2, 3, 4, 6, 5, 7)
    del _tmp3
    _tmp4 = einsum('adil,bcjk->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp4
    result -= 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp4.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp4
    _tmp5 = einsum('adjk,bcil->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp5
    result -= 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp5.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp5.transpose(1, 0, 2, 3, 6, 5, 4, 7)
    del _tmp5
    _tmp6 = einsum('bckl,adij->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp6
    result -= 1 * _tmp6.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp6.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp6.transpose(0, 3, 2, 1, 4, 6, 5, 7)
    del _tmp6
    _tmp7 = einsum('bcil,adjk->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp7
    result -= 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp7.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 7, 6)
    del _tmp7
    _tmp8 = einsum('bcjk,adil->abcdijkl', g_aaaa[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp8
    result -= 1 * _tmp8.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp8.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp8.transpose(0, 3, 2, 1, 6, 5, 4, 7)
    del _tmp8
    _tmp9 = einsum('makl,bcdijm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp9
    result -= 1 * _tmp9.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp9.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp9.transpose(1, 0, 2, 3, 4, 6, 5, 7)
    del _tmp9
    _tmp10 = einsum('mail,bcdjkm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp10
    result -= 1 * _tmp10.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp10.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp10.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp10
    _tmp11 = einsum('majk,bcdilm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp11
    result -= 1 * _tmp11.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp11.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp11.transpose(1, 0, 2, 3, 6, 5, 4, 7)
    del _tmp11
    _tmp12 = einsum('mckl,abdijm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp12
    result -= 1 * _tmp12.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp12.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp12.transpose(0, 1, 3, 2, 4, 6, 5, 7)
    del _tmp12
    _tmp13 = einsum('mcil,abdjkm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp13
    result -= 1 * _tmp13.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp13.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp13.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    del _tmp13
    _tmp14 = einsum('mcjk,abdilm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp14
    result -= 1 * _tmp14.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp14.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp14.transpose(0, 1, 3, 2, 6, 5, 4, 7)
    del _tmp14
    _tmp15 = einsum('abel,ecdijk->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp15
    result -= 1 * _tmp15.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp15.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp15.transpose(0, 2, 1, 3, 4, 5, 7, 6)
    del _tmp15
    _tmp16 = einsum('abej,ecdikl->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp16
    result -= 1 * _tmp16.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp16.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp16.transpose(0, 2, 1, 3, 5, 4, 6, 7)
    del _tmp16
    _tmp17 = einsum('adel,ebcijk->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp17
    result -= 1 * _tmp17.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp17.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp17.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp17
    _tmp18 = einsum('adej,ebcikl->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp18
    result -= 1 * _tmp18.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp18.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp18.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    del _tmp18
    _tmp19 = einsum('bcel,eadijk->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp19
    result -= 1 * _tmp19.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp19.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp19.transpose(0, 3, 2, 1, 4, 5, 7, 6)
    del _tmp19
    _tmp20 = einsum('bcej,eadikl->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp20
    result -= 1 * _tmp20.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp20.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp20.transpose(0, 3, 2, 1, 5, 4, 6, 7)
    del _tmp20
    _tmp21 = einsum('nmkl,abcdijnm->abcdijkl', g_aaaa[o, o, o, o], t4_aaaaaaaa, optimize=True)
    result += 0.5 * _tmp21
    result -= 0.5 * _tmp21.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp21
    _tmp22 = einsum('nmil,abcdjknm->abcdijkl', g_aaaa[o, o, o, o], t4_aaaaaaaa, optimize=True)
    result += 0.5 * _tmp22
    result -= 0.5 * _tmp22.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp22
    _tmp23 = einsum('nmjk,abcdilnm->abcdijkl', g_aaaa[o, o, o, o], t4_aaaaaaaa, optimize=True)
    result += 0.5 * _tmp23
    result -= 0.5 * _tmp23.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    del _tmp23
    _tmp24 = einsum('mael,ebcdijkm->abcdijkl', g_aaaa[o, v, v, o], t4_aaaaaaaa, optimize=True)
    result += 1 * _tmp24
    result -= 1 * _tmp24.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp24.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp24.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp24
    _tmp25 = einsum('amle,dbceijkm->abcdijkl', g_abab[v, o, o, v], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp25
    result += 1 * _tmp25.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp25.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp25.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp25
    _tmp26 = einsum('maej,ebcdiklm->abcdijkl', g_aaaa[o, v, v, o], t4_aaaaaaaa, optimize=True)
    result += 1 * _tmp26
    result -= 1 * _tmp26.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp26.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp26.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    del _tmp26
    _tmp27 = einsum('amje,dbceiklm->abcdijkl', g_abab[v, o, o, v], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp27
    result += 1 * _tmp27.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp27.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result -= 1 * _tmp27.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    del _tmp27
    _tmp28 = einsum('mcel,eabdijkm->abcdijkl', g_aaaa[o, v, v, o], t4_aaaaaaaa, optimize=True)
    result += 1 * _tmp28
    result -= 1 * _tmp28.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp28.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp28.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    del _tmp28
    _tmp29 = einsum('cmle,dabeijkm->abcdijkl', g_abab[v, o, o, v], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp29
    result += 1 * _tmp29.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result += 1 * _tmp29.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp29.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    del _tmp29
    _tmp30 = einsum('mcej,eabdiklm->abcdijkl', g_aaaa[o, v, v, o], t4_aaaaaaaa, optimize=True)
    result += 1 * _tmp30
    result -= 1 * _tmp30.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp30.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp30.transpose(0, 1, 3, 2, 5, 4, 6, 7)
    del _tmp30
    _tmp31 = einsum('cmje,dabeiklm->abcdijkl', g_abab[v, o, o, v], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp31
    result += 1 * _tmp31.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result += 1 * _tmp31.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result -= 1 * _tmp31.transpose(0, 1, 3, 2, 5, 4, 6, 7)
    del _tmp31
    _tmp32 = einsum('abef,efcdijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaaaaaaa, optimize=True)
    result += 0.5 * _tmp32
    result -= 0.5 * _tmp32.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp32
    _tmp33 = einsum('adef,efbcijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaaaaaaa, optimize=True)
    result += 0.5 * _tmp33
    result -= 0.5 * _tmp33.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp33
    _tmp34 = einsum('bcef,efadijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaaaaaaa, optimize=True)
    result += 0.5 * _tmp34
    result -= 0.5 * _tmp34.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    del _tmp34
    return result


def t4_3_aaabaaab_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb):
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    t4_aaaaaaaa = t4_2_aaaaaaaa
    t4_aaabaaab = t4_2_aaabaaab
    t4_aabbaabb = t4_2_aabbaabb
    t4_abbbabbb = t4_2_abbbabbb
    t4_bbbbbbbb = t4_2_aaaaaaaa
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('abik,cdjl->abcdijkl', g_aaaa[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp0
    _tmp1 = einsum('abjk,cdil->abcdijkl', g_aaaa[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp1.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp1.transpose(0, 2, 1, 3, 6, 5, 4, 7)
    del _tmp1
    _tmp2 = einsum('adkl,bcij->abcdijkl', g_abab[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp2.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp2.transpose(1, 0, 2, 3, 4, 6, 5, 7)
    del _tmp2
    _tmp3 = einsum('adil,bcjk->abcdijkl', g_abab[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp3
    _tmp4 = einsum('cdkl,abij->abcdijkl', g_abab[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp4
    result -= 1 * _tmp4.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp4
    _tmp5 = einsum('cdil,abjk->abcdijkl', g_abab[v, v, o, o], t2_aaaa, optimize=True)
    result += 1 * _tmp5
    del _tmp5
    _tmp6 = einsum('bcik,adjl->abcdijkl', g_aaaa[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp6
    del _tmp6
    _tmp7 = einsum('bcjk,adil->abcdijkl', g_aaaa[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp7
    result -= 1 * _tmp7.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    del _tmp7
    _tmp8 = einsum('amkl,bcdijm->abcdijkl', g_abab[v, o, o, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp8
    result += 1 * _tmp8.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp8.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result -= 1 * _tmp8.transpose(1, 0, 2, 3, 4, 6, 5, 7)
    del _tmp8
    _tmp9 = einsum('amil,bcdjkm->abcdijkl', g_abab[v, o, o, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp9
    result += 1 * _tmp9.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp9
    _tmp10 = einsum('maik,bcdjml->abcdijkl', g_aaaa[o, v, o, o], t3_aabaab, optimize=True)
    result += 1 * _tmp10
    result -= 1 * _tmp10.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp10
    _tmp11 = einsum('majk,bcdiml->abcdijkl', g_aaaa[o, v, o, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp11
    result += 1 * _tmp11.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp11.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result -= 1 * _tmp11.transpose(1, 0, 2, 3, 6, 5, 4, 7)
    del _tmp11
    _tmp12 = einsum('cmkl,abdijm->abcdijkl', g_abab[v, o, o, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp12
    result += 1 * _tmp12.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp12
    _tmp13 = einsum('mdkl,abcijm->abcdijkl', g_abab[o, v, o, o], t3_aaaaaa, optimize=True)
    result -= 1 * _tmp13
    result += 1 * _tmp13.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp13
    _tmp14 = einsum('cmil,abdjkm->abcdijkl', g_abab[v, o, o, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp14
    del _tmp14
    _tmp15 = einsum('mdil,abcjkm->abcdijkl', g_abab[o, v, o, o], t3_aaaaaa, optimize=True)
    result -= 1 * _tmp15
    del _tmp15
    _tmp16 = einsum('mcik,abdjml->abcdijkl', g_aaaa[o, v, o, o], t3_aabaab, optimize=True)
    result += 1 * _tmp16
    del _tmp16
    _tmp17 = einsum('mcjk,abdiml->abcdijkl', g_aaaa[o, v, o, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp17
    result += 1 * _tmp17.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    del _tmp17
    _tmp18 = einsum('abek,ecdijl->abcdijkl', g_aaaa[v, v, v, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp18
    result += 1 * _tmp18.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp18
    _tmp19 = einsum('abej,ecdikl->abcdijkl', g_aaaa[v, v, v, o], t3_aabaab, optimize=True)
    result += 1 * _tmp19
    result -= 1 * _tmp19.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp19.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp19.transpose(0, 2, 1, 3, 5, 4, 6, 7)
    del _tmp19
    _tmp20 = einsum('adel,ebcijk->abcdijkl', g_abab[v, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp20
    result -= 1 * _tmp20.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp20
    _tmp21 = einsum('adke,cbeijl->abcdijkl', g_abab[v, v, o, v], t3_aabaab, optimize=True)
    result -= 1 * _tmp21
    result += 1 * _tmp21.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp21
    _tmp22 = einsum('adje,cbeikl->abcdijkl', g_abab[v, v, o, v], t3_aabaab, optimize=True)
    result += 1 * _tmp22
    result -= 1 * _tmp22.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp22.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp22.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    del _tmp22
    _tmp23 = einsum('cdel,eabijk->abcdijkl', g_abab[v, v, v, o], t3_aaaaaa, optimize=True)
    result += 1 * _tmp23
    del _tmp23
    _tmp24 = einsum('bcek,eadijl->abcdijkl', g_aaaa[v, v, v, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp24
    del _tmp24
    _tmp25 = einsum('cdke,baeijl->abcdijkl', g_abab[v, v, o, v], t3_aabaab, optimize=True)
    result -= 1 * _tmp25
    del _tmp25
    _tmp26 = einsum('bcej,eadikl->abcdijkl', g_aaaa[v, v, v, o], t3_aabaab, optimize=True)
    result += 1 * _tmp26
    result -= 1 * _tmp26.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    del _tmp26
    _tmp27 = einsum('cdje,baeikl->abcdijkl', g_abab[v, v, o, v], t3_aabaab, optimize=True)
    result += 1 * _tmp27
    result -= 1 * _tmp27.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    del _tmp27
    _tmp28 = einsum('nmkl,abcdijnm->abcdijkl', g_abab[o, o, o, o], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp28
    result -= 0.5 * _tmp28.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 0.5 * _tmp28
    result -= 0.5 * _tmp28.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp28
    _tmp29 = einsum('nmil,abcdjknm->abcdijkl', g_abab[o, o, o, o], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp29
    result += 0.5 * _tmp29
    del _tmp29
    _tmp30 = einsum('nmik,abcdjmnl->abcdijkl', g_aaaa[o, o, o, o], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp30
    del _tmp30
    _tmp31 = einsum('nmjk,abcdimnl->abcdijkl', g_aaaa[o, o, o, o], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp31
    result += 0.5 * _tmp31.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    del _tmp31
    _tmp32 = einsum('amel,ebcdijkm->abcdijkl', g_abab[v, o, v, o], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp32
    result += 1 * _tmp32.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp32
    _tmp33 = einsum('maek,ebcdijml->abcdijkl', g_aaaa[o, v, v, o], t4_aaabaaab, optimize=True)
    result += 1 * _tmp33
    result -= 1 * _tmp33.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp33
    _tmp34 = einsum('amke,cbedijlm->abcdijkl', g_abab[v, o, o, v], t4_aabbaabb, optimize=True)
    result += 1 * _tmp34
    result -= 1 * _tmp34.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp34
    _tmp35 = einsum('maej,ebcdikml->abcdijkl', g_aaaa[o, v, v, o], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp35
    result += 1 * _tmp35.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp35.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result -= 1 * _tmp35.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    del _tmp35
    _tmp36 = einsum('amje,cbediklm->abcdijkl', g_abab[v, o, o, v], t4_aabbaabb, optimize=True)
    result -= 1 * _tmp36
    result += 1 * _tmp36.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp36.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result -= 1 * _tmp36.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    del _tmp36
    _tmp37 = einsum('cmel,eabdijkm->abcdijkl', g_abab[v, o, v, o], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp37
    del _tmp37
    _tmp38 = einsum('mdel,eabcijkm->abcdijkl', g_abab[o, v, v, o], t4_aaaaaaaa, optimize=True)
    result -= 1 * _tmp38
    del _tmp38
    _tmp39 = einsum('mdel,cabeijkm->abcdijkl', g_bbbb[o, v, v, o], t4_aaabaaab, optimize=True)
    result += 1 * _tmp39
    del _tmp39
    _tmp40 = einsum('mcek,eabdijml->abcdijkl', g_aaaa[o, v, v, o], t4_aaabaaab, optimize=True)
    result += 1 * _tmp40
    del _tmp40
    _tmp41 = einsum('cmke,baedijlm->abcdijkl', g_abab[v, o, o, v], t4_aabbaabb, optimize=True)
    result += 1 * _tmp41
    del _tmp41
    _tmp42 = einsum('mdke,cabeijml->abcdijkl', g_abab[o, v, o, v], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp42
    del _tmp42
    _tmp43 = einsum('mcej,eabdikml->abcdijkl', g_aaaa[o, v, v, o], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp43
    result += 1 * _tmp43.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    del _tmp43
    _tmp44 = einsum('cmje,baediklm->abcdijkl', g_abab[v, o, o, v], t4_aabbaabb, optimize=True)
    result -= 1 * _tmp44
    result += 1 * _tmp44.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    del _tmp44
    _tmp45 = einsum('mdje,cabeikml->abcdijkl', g_abab[o, v, o, v], t4_aaabaaab, optimize=True)
    result += 1 * _tmp45
    result -= 1 * _tmp45.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    del _tmp45
    _tmp46 = einsum('abef,efcdijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp46
    result -= 0.5 * _tmp46.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp46
    _tmp47 = einsum('adef,ecbfijkl->abcdijkl', g_abab[v, v, v, v], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp47
    result += 0.5 * _tmp47.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp47
    _tmp48 = einsum('adfe,cfbeijkl->abcdijkl', g_abab[v, v, v, v], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp48
    result -= 0.5 * _tmp48.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp48
    _tmp49 = einsum('bcef,efadijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp49
    del _tmp49
    _tmp50 = einsum('cdef,ebafijkl->abcdijkl', g_abab[v, v, v, v], t4_aaabaaab, optimize=True)
    result -= 0.5 * _tmp50
    del _tmp50
    _tmp51 = einsum('cdfe,bfaeijkl->abcdijkl', g_abab[v, v, v, v], t4_aaabaaab, optimize=True)
    result += 0.5 * _tmp51
    del _tmp51
    return result


def t4_3_aabbaabb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb):
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    t4_aaaaaaaa = t4_2_aaaaaaaa
    t4_aaabaaab = t4_2_aaabaaab
    t4_aabbaabb = t4_2_aabbaabb
    t4_abbbabbb = t4_2_abbbabbb
    t4_bbbbbbbb = t4_2_aaaaaaaa
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('acjl,bdik->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp0
    del _tmp0
    _tmp1 = einsum('acil,bdjk->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp1
    result += 1 * _tmp1.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp1
    _tmp2 = einsum('acjk,bdil->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp2
    del _tmp2
    _tmp3 = einsum('abji,cdkl->abcdijkl', g_aaaa[v, v, o, o], t2_bbbb, optimize=True)
    result -= 1 * _tmp3
    del _tmp3
    _tmp4 = einsum('adjl,bcik->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp4
    _tmp5 = einsum('adil,bcjk->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp5
    result -= 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp5.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp5
    _tmp6 = einsum('adjk,bcil->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp6
    result -= 1 * _tmp6.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp6
    _tmp7 = einsum('dckl,abij->abcdijkl', g_bbbb[v, v, o, o], t2_aaaa, optimize=True)
    result -= 1 * _tmp7
    del _tmp7
    _tmp8 = einsum('bcjl,adik->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp8
    del _tmp8
    _tmp9 = einsum('bcil,adjk->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp9
    result -= 1 * _tmp9.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp9
    _tmp10 = einsum('bcjk,adil->abcdijkl', g_abab[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp10
    del _tmp10
    _tmp11 = einsum('amjl,bcdikm->abcdijkl', g_abab[v, o, o, o], t3_abbabb, optimize=True)
    result += 1 * _tmp11
    result -= 1 * _tmp11.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp11
    _tmp12 = einsum('amil,bcdjkm->abcdijkl', g_abab[v, o, o, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp12
    result += 1 * _tmp12.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp12.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp12.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp12
    _tmp13 = einsum('amjk,bcdilm->abcdijkl', g_abab[v, o, o, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp13
    result += 1 * _tmp13.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp13
    _tmp14 = einsum('maji,bcdmlk->abcdijkl', g_aaaa[o, v, o, o], t3_abbabb, optimize=True)
    result += 1 * _tmp14
    result -= 1 * _tmp14.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp14
    _tmp15 = einsum('mckl,abdijm->abcdijkl', g_bbbb[o, v, o, o], t3_aabaab, optimize=True)
    result += 1 * _tmp15
    result -= 1 * _tmp15.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    del _tmp15
    _tmp16 = einsum('mcjl,abdimk->abcdijkl', g_abab[o, v, o, o], t3_aabaab, optimize=True)
    result += 1 * _tmp16
    result -= 1 * _tmp16.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    del _tmp16
    _tmp17 = einsum('mcil,abdjmk->abcdijkl', g_abab[o, v, o, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp17
    result += 1 * _tmp17.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result += 1 * _tmp17.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp17.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    del _tmp17
    _tmp18 = einsum('mcjk,abdiml->abcdijkl', g_abab[o, v, o, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp18
    result += 1 * _tmp18.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    del _tmp18
    _tmp19 = einsum('acel,ebdijk->abcdijkl', g_abab[v, v, v, o], t3_aabaab, optimize=True)
    result -= 1 * _tmp19
    result += 1 * _tmp19.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp19
    _tmp20 = einsum('abej,ecdikl->abcdijkl', g_aaaa[v, v, v, o], t3_abbabb, optimize=True)
    result += 1 * _tmp20
    result -= 1 * _tmp20.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    del _tmp20
    _tmp21 = einsum('acje,bedikl->abcdijkl', g_abab[v, v, o, v], t3_abbabb, optimize=True)
    result -= 1 * _tmp21
    result += 1 * _tmp21.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    del _tmp21
    _tmp22 = einsum('adel,ebcijk->abcdijkl', g_abab[v, v, v, o], t3_aabaab, optimize=True)
    result += 1 * _tmp22
    result -= 1 * _tmp22.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp22.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp22.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp22
    _tmp23 = einsum('adje,becikl->abcdijkl', g_abab[v, v, o, v], t3_abbabb, optimize=True)
    result += 1 * _tmp23
    result -= 1 * _tmp23.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp23.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp23.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    del _tmp23
    _tmp24 = einsum('bcel,eadijk->abcdijkl', g_abab[v, v, v, o], t3_aabaab, optimize=True)
    result += 1 * _tmp24
    result -= 1 * _tmp24.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp24
    _tmp25 = einsum('dcel,baeijk->abcdijkl', g_bbbb[v, v, v, o], t3_aabaab, optimize=True)
    result += 1 * _tmp25
    result -= 1 * _tmp25.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp25
    _tmp26 = einsum('bcje,aedikl->abcdijkl', g_abab[v, v, o, v], t3_abbabb, optimize=True)
    result += 1 * _tmp26
    result -= 1 * _tmp26.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    del _tmp26
    _tmp27 = einsum('nmkl,abcdijnm->abcdijkl', g_bbbb[o, o, o, o], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp27
    del _tmp27
    _tmp28 = einsum('nmjl,abcdinkm->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp28
    del _tmp28
    _tmp29 = einsum('mnjl,abcdimnk->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp29
    del _tmp29
    _tmp30 = einsum('nmil,abcdjnkm->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp30
    result += 0.5 * _tmp30.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp30
    _tmp31 = einsum('mnil,abcdjmnk->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp31
    result -= 0.5 * _tmp31.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp31
    _tmp32 = einsum('nmjk,abcdinlm->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp32
    del _tmp32
    _tmp33 = einsum('mnjk,abcdimnl->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp33
    del _tmp33
    _tmp34 = einsum('nmji,abcdnmkl->abcdijkl', g_aaaa[o, o, o, o], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp34
    del _tmp34
    _tmp35 = einsum('amel,ebcdijkm->abcdijkl', g_abab[v, o, v, o], t4_aabbaabb, optimize=True)
    result -= 1 * _tmp35
    result += 1 * _tmp35.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp35.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp35.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    del _tmp35
    _tmp36 = einsum('maej,ebcdimlk->abcdijkl', g_aaaa[o, v, v, o], t4_aabbaabb, optimize=True)
    result -= 1 * _tmp36
    result += 1 * _tmp36.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp36.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result -= 1 * _tmp36.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    del _tmp36
    _tmp37 = einsum('amje,becdiklm->abcdijkl', g_abab[v, o, o, v], t4_abbbabbb, optimize=True)
    result -= 1 * _tmp37
    result += 1 * _tmp37.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result += 1 * _tmp37.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result -= 1 * _tmp37.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    del _tmp37
    _tmp38 = einsum('mcel,eabdijmk->abcdijkl', g_abab[o, v, v, o], t4_aaabaaab, optimize=True)
    result -= 1 * _tmp38
    result += 1 * _tmp38.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result += 1 * _tmp38.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp38.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    del _tmp38
    _tmp39 = einsum('mcel,baedijkm->abcdijkl', g_bbbb[o, v, v, o], t4_aabbaabb, optimize=True)
    result -= 1 * _tmp39
    result += 1 * _tmp39.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result += 1 * _tmp39.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp39.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    del _tmp39
    _tmp40 = einsum('mcje,baedimlk->abcdijkl', g_abab[o, v, o, v], t4_aabbaabb, optimize=True)
    result -= 1 * _tmp40
    result += 1 * _tmp40.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result += 1 * _tmp40.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result -= 1 * _tmp40.transpose(0, 1, 3, 2, 5, 4, 6, 7)
    del _tmp40
    _tmp41 = einsum('abef,efcdijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp41
    del _tmp41
    _tmp42 = einsum('acef,ebfdijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp42
    del _tmp42
    _tmp43 = einsum('acfe,bfedijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp43
    del _tmp43
    _tmp44 = einsum('adef,ebfcijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp44
    result += 0.5 * _tmp44.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp44
    _tmp45 = einsum('adfe,bfecijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp45
    result -= 0.5 * _tmp45.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    del _tmp45
    _tmp46 = einsum('bcef,eafdijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp46
    del _tmp46
    _tmp47 = einsum('bcfe,afedijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb, optimize=True)
    result += 0.5 * _tmp47
    del _tmp47
    _tmp48 = einsum('dcef,abefijkl->abcdijkl', g_bbbb[v, v, v, v], t4_aabbaabb, optimize=True)
    result -= 0.5 * _tmp48
    del _tmp48
    return result


def t4_3_abbbabbb_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb, t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb):
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    t3_aaaaaa = t3_2_aaaaaa
    t3_aabaab = t3_2_aabaab
    t3_abbabb = t3_2_abbabb
    t3_bbbbbb = t3_2_aaaaaa
    t4_aaaaaaaa = t4_2_aaaaaaaa
    t4_aaabaaab = t4_2_aaabaaab
    t4_aabbaabb = t4_2_aabbaabb
    t4_abbbabbb = t4_2_abbbabbb
    t4_bbbbbbbb = t4_2_aaaaaaaa
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('abil,cdjk->abcdijkl', g_abab[v, v, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp0.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 7, 6)
    del _tmp0
    _tmp1 = einsum('abij,cdkl->abcdijkl', g_abab[v, v, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp1
    _tmp2 = einsum('bdkl,acij->abcdijkl', g_bbbb[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp2
    _tmp3 = einsum('adil,bcjk->abcdijkl', g_abab[v, v, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp3
    _tmp4 = einsum('bdjk,acil->abcdijkl', g_bbbb[v, v, o, o], t2_abab, optimize=True)
    result -= 1 * _tmp4
    del _tmp4
    _tmp5 = einsum('adij,bckl->abcdijkl', g_abab[v, v, o, o], t2_bbbb, optimize=True)
    result += 1 * _tmp5
    del _tmp5
    _tmp6 = einsum('bckl,adij->abcdijkl', g_bbbb[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp6
    result -= 1 * _tmp6.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp6.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp6.transpose(0, 3, 2, 1, 4, 6, 5, 7)
    del _tmp6
    _tmp7 = einsum('bcjk,adil->abcdijkl', g_bbbb[v, v, o, o], t2_abab, optimize=True)
    result += 1 * _tmp7
    result -= 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    del _tmp7
    _tmp8 = einsum('mbkl,acdijm->abcdijkl', g_bbbb[o, v, o, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp8
    result += 1 * _tmp8.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp8
    _tmp9 = einsum('amil,bcdjkm->abcdijkl', g_abab[v, o, o, o], t3_bbbbbb, optimize=True)
    result -= 1 * _tmp9
    result += 1 * _tmp9.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp9
    _tmp10 = einsum('mbil,acdmkj->abcdijkl', g_abab[o, v, o, o], t3_abbabb, optimize=True)
    result += 1 * _tmp10
    result -= 1 * _tmp10.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp10
    _tmp11 = einsum('mbjk,acdilm->abcdijkl', g_bbbb[o, v, o, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp11
    del _tmp11
    _tmp12 = einsum('amij,bcdklm->abcdijkl', g_abab[v, o, o, o], t3_bbbbbb, optimize=True)
    result -= 1 * _tmp12
    del _tmp12
    _tmp13 = einsum('mbij,acdmlk->abcdijkl', g_abab[o, v, o, o], t3_abbabb, optimize=True)
    result += 1 * _tmp13
    del _tmp13
    _tmp14 = einsum('mckl,abdijm->abcdijkl', g_bbbb[o, v, o, o], t3_abbabb, optimize=True)
    result += 1 * _tmp14
    result -= 1 * _tmp14.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp14.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp14.transpose(0, 1, 3, 2, 4, 6, 5, 7)
    del _tmp14
    _tmp15 = einsum('mcil,abdmkj->abcdijkl', g_abab[o, v, o, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp15
    result += 1 * _tmp15.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result += 1 * _tmp15.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp15.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    del _tmp15
    _tmp16 = einsum('mcjk,abdilm->abcdijkl', g_bbbb[o, v, o, o], t3_abbabb, optimize=True)
    result += 1 * _tmp16
    result -= 1 * _tmp16.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    del _tmp16
    _tmp17 = einsum('mcij,abdmlk->abcdijkl', g_abab[o, v, o, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp17
    result += 1 * _tmp17.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    del _tmp17
    _tmp18 = einsum('abel,ecdijk->abcdijkl', g_abab[v, v, v, o], t3_abbabb, optimize=True)
    result += 1 * _tmp18
    result -= 1 * _tmp18.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp18.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp18.transpose(0, 2, 1, 3, 4, 5, 7, 6)
    del _tmp18
    _tmp19 = einsum('abej,ecdikl->abcdijkl', g_abab[v, v, v, o], t3_abbabb, optimize=True)
    result += 1 * _tmp19
    result -= 1 * _tmp19.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp19
    _tmp20 = einsum('abie,ecdjkl->abcdijkl', g_abab[v, v, o, v], t3_bbbbbb, optimize=True)
    result += 1 * _tmp20
    result -= 1 * _tmp20.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp20
    _tmp21 = einsum('adel,ebcijk->abcdijkl', g_abab[v, v, v, o], t3_abbabb, optimize=True)
    result += 1 * _tmp21
    result -= 1 * _tmp21.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp21
    _tmp22 = einsum('bdel,aecijk->abcdijkl', g_bbbb[v, v, v, o], t3_abbabb, optimize=True)
    result += 1 * _tmp22
    result -= 1 * _tmp22.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp22
    _tmp23 = einsum('adej,ebcikl->abcdijkl', g_abab[v, v, v, o], t3_abbabb, optimize=True)
    result += 1 * _tmp23
    del _tmp23
    _tmp24 = einsum('bdej,aecikl->abcdijkl', g_bbbb[v, v, v, o], t3_abbabb, optimize=True)
    result += 1 * _tmp24
    del _tmp24
    _tmp25 = einsum('adie,ebcjkl->abcdijkl', g_abab[v, v, o, v], t3_bbbbbb, optimize=True)
    result += 1 * _tmp25
    del _tmp25
    _tmp26 = einsum('bcel,aedijk->abcdijkl', g_bbbb[v, v, v, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp26
    result += 1 * _tmp26.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result += 1 * _tmp26.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp26.transpose(0, 3, 2, 1, 4, 5, 7, 6)
    del _tmp26
    _tmp27 = einsum('bcej,aedikl->abcdijkl', g_bbbb[v, v, v, o], t3_abbabb, optimize=True)
    result -= 1 * _tmp27
    result += 1 * _tmp27.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    del _tmp27
    _tmp28 = einsum('nmkl,abcdijnm->abcdijkl', g_bbbb[o, o, o, o], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp28
    result -= 0.5 * _tmp28.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    del _tmp28
    _tmp29 = einsum('nmil,abcdnkjm->abcdijkl', g_abab[o, o, o, o], t4_abbbabbb, optimize=True)
    result -= 0.5 * _tmp29
    result += 0.5 * _tmp29.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp29
    _tmp30 = einsum('mnil,abcdmknj->abcdijkl', g_abab[o, o, o, o], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp30
    result -= 0.5 * _tmp30.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp30
    _tmp31 = einsum('nmjk,abcdilnm->abcdijkl', g_bbbb[o, o, o, o], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp31
    del _tmp31
    _tmp32 = einsum('nmij,abcdnlkm->abcdijkl', g_abab[o, o, o, o], t4_abbbabbb, optimize=True)
    result -= 0.5 * _tmp32
    del _tmp32
    _tmp33 = einsum('mnij,abcdmlnk->abcdijkl', g_abab[o, o, o, o], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp33
    del _tmp33
    _tmp34 = einsum('amel,ebcdijkm->abcdijkl', g_abab[v, o, v, o], t4_abbbabbb, optimize=True)
    result -= 1 * _tmp34
    result += 1 * _tmp34.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp34
    _tmp35 = einsum('mbel,eacdimkj->abcdijkl', g_abab[o, v, v, o], t4_aabbaabb, optimize=True)
    result += 1 * _tmp35
    result -= 1 * _tmp35.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp35
    _tmp36 = einsum('mbel,aecdijkm->abcdijkl', g_bbbb[o, v, v, o], t4_abbbabbb, optimize=True)
    result += 1 * _tmp36
    result -= 1 * _tmp36.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    del _tmp36
    _tmp37 = einsum('amej,ebcdiklm->abcdijkl', g_abab[v, o, v, o], t4_abbbabbb, optimize=True)
    result -= 1 * _tmp37
    del _tmp37
    _tmp38 = einsum('mbej,eacdimlk->abcdijkl', g_abab[o, v, v, o], t4_aabbaabb, optimize=True)
    result += 1 * _tmp38
    del _tmp38
    _tmp39 = einsum('mbej,aecdiklm->abcdijkl', g_bbbb[o, v, v, o], t4_abbbabbb, optimize=True)
    result += 1 * _tmp39
    del _tmp39
    _tmp40 = einsum('maei,ebcdmklj->abcdijkl', g_aaaa[o, v, v, o], t4_abbbabbb, optimize=True)
    result += 1 * _tmp40
    del _tmp40
    _tmp41 = einsum('amie,ebcdjklm->abcdijkl', g_abab[v, o, o, v], t4_bbbbbbbb, optimize=True)
    result -= 1 * _tmp41
    del _tmp41
    _tmp42 = einsum('mbie,aecdmklj->abcdijkl', g_abab[o, v, o, v], t4_abbbabbb, optimize=True)
    result -= 1 * _tmp42
    del _tmp42
    _tmp43 = einsum('mcel,eabdimkj->abcdijkl', g_abab[o, v, v, o], t4_aabbaabb, optimize=True)
    result -= 1 * _tmp43
    result += 1 * _tmp43.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result += 1 * _tmp43.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp43.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    del _tmp43
    _tmp44 = einsum('mcel,aebdijkm->abcdijkl', g_bbbb[o, v, v, o], t4_abbbabbb, optimize=True)
    result -= 1 * _tmp44
    result += 1 * _tmp44.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result += 1 * _tmp44.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result -= 1 * _tmp44.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    del _tmp44
    _tmp45 = einsum('mcej,eabdimlk->abcdijkl', g_abab[o, v, v, o], t4_aabbaabb, optimize=True)
    result -= 1 * _tmp45
    result += 1 * _tmp45.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    del _tmp45
    _tmp46 = einsum('mcej,aebdiklm->abcdijkl', g_bbbb[o, v, v, o], t4_abbbabbb, optimize=True)
    result -= 1 * _tmp46
    result += 1 * _tmp46.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    del _tmp46
    _tmp47 = einsum('mcie,aebdmklj->abcdijkl', g_abab[o, v, o, v], t4_abbbabbb, optimize=True)
    result += 1 * _tmp47
    result -= 1 * _tmp47.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    del _tmp47
    _tmp48 = einsum('abef,efcdijkl->abcdijkl', g_abab[v, v, v, v], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp48
    result -= 0.5 * _tmp48.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result += 0.5 * _tmp48
    result -= 0.5 * _tmp48.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    del _tmp48
    _tmp49 = einsum('adef,efbcijkl->abcdijkl', g_abab[v, v, v, v], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp49
    result += 0.5 * _tmp49
    del _tmp49
    _tmp50 = einsum('bdef,afecijkl->abcdijkl', g_bbbb[v, v, v, v], t4_abbbabbb, optimize=True)
    result += 0.5 * _tmp50
    del _tmp50
    _tmp51 = einsum('bcef,afedijkl->abcdijkl', g_bbbb[v, v, v, v], t4_abbbabbb, optimize=True)
    result -= 0.5 * _tmp51
    result += 0.5 * _tmp51.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    del _tmp51
    return result


def t1_4_aa_numerator(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t1_3_aa, t2_3_aaaa, t2_3_abab, t3_3_aaaaaa, t3_3_aabaab, t3_3_abbabb):
    t1_aa = t1_3_aa
    t1_bb = t1_3_aa
    t2_aaaa = t2_3_aaaa
    t2_abab = t2_3_abab
    t2_bbbb = t2_3_aaaa
    t3_aaaaaa = t3_3_aaaaaa
    t3_aabaab = t3_3_aabaab
    t3_abbabb = t3_3_abbabb
    t3_bbbbbb = t3_3_aaaaaa
    result = np.zeros((nv, no))
    _tmp0 = einsum('jabi,bj->ai', g_aaaa[o, v, v, o], t1_aa, optimize=True)
    result += 1 * _tmp0
    del _tmp0
    _tmp1 = einsum('ajib,bj->ai', g_abab[v, o, o, v], t1_bb, optimize=True)
    result += 1 * _tmp1
    del _tmp1
    _tmp2 = einsum('kjbi,bakj->ai', g_aaaa[o, o, v, o], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp2
    del _tmp2
    _tmp3 = einsum('kjib,abkj->ai', g_abab[o, o, o, v], t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    del _tmp3
    _tmp4 = einsum('jabc,bcij->ai', g_aaaa[o, v, v, v], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp4
    del _tmp4
    _tmp5 = einsum('ajbc,bcij->ai', g_abab[v, o, v, v], t2_abab, optimize=True)
    result += 0.5 * _tmp5
    result += 0.5 * _tmp5
    del _tmp5
    _tmp6 = einsum('kjbc,bcaikj->ai', g_aaaa[o, o, v, v], t3_aaaaaa, optimize=True)
    result += 0.25 * _tmp6
    del _tmp6
    _tmp7 = einsum('kjbc,bacikj->ai', g_abab[o, o, v, v], t3_aabaab, optimize=True)
    result -= 0.25 * _tmp7
    result -= 0.25 * _tmp7
    del _tmp7
    _tmp8 = einsum('kjcb,acbikj->ai', g_abab[o, o, v, v], t3_aabaab, optimize=True)
    result += 0.25 * _tmp8
    result += 0.25 * _tmp8
    del _tmp8
    _tmp9 = einsum('kjbc,acbikj->ai', g_bbbb[o, o, v, v], t3_abbabb, optimize=True)
    result -= 0.25 * _tmp9
    del _tmp9
    return result


def t1_3_aa_numerator_no_t3(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t1_2_aa, t2_2_aaaa, t2_2_abab):
    t1_aa = t1_2_aa
    t1_bb = t1_2_aa
    t2_aaaa = t2_2_aaaa
    t2_abab = t2_2_abab
    t2_bbbb = t2_2_aaaa
    result = np.zeros((nv, no))
    _tmp0 = einsum('jabi,bj->ai', g_aaaa[o, v, v, o], t1_aa, optimize=True)
    result += 1 * _tmp0
    del _tmp0
    _tmp1 = einsum('ajib,bj->ai', g_abab[v, o, o, v], t1_bb, optimize=True)
    result += 1 * _tmp1
    del _tmp1
    _tmp2 = einsum('kjbi,bakj->ai', g_aaaa[o, o, v, o], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp2
    del _tmp2
    _tmp3 = einsum('kjib,abkj->ai', g_abab[o, o, o, v], t2_abab, optimize=True)
    result -= 0.5 * _tmp3
    result -= 0.5 * _tmp3
    del _tmp3
    _tmp4 = einsum('jabc,bcij->ai', g_aaaa[o, v, v, v], t2_aaaa, optimize=True)
    result -= 0.5 * _tmp4
    del _tmp4
    _tmp5 = einsum('ajbc,bcij->ai', g_abab[v, o, v, v], t2_abab, optimize=True)
    result += 0.5 * _tmp5
    result += 0.5 * _tmp5
    del _tmp5
    return result


def overlap1_restricted(l_aa, t_aa):
    l1_aa = l_aa
    t1_aa = t_aa
    l1_bb = l_aa
    t1_bb = t_aa
    result = np.zeros(())
    _tmp0 = einsum('ia,ai->', l1_aa, t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('ia,ai->', l1_bb, t1_bb, optimize=True)
    result += 1 * _tmp1
    return result


def overlap2_restricted(l_aaaa, l_abab, t_aaaa, t_abab):
    l2_aaaa = l_aaaa
    t2_aaaa = t_aaaa
    l2_abab = l_abab
    t2_abab = t_abab
    l2_bbbb = l_aaaa
    t2_bbbb = t_aaaa
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


def overlap3_restricted(l_aaaaaa, l_aabaab, l_abbabb, t_aaaaaa, t_aabaab, t_abbabb):
    l3_aaaaaa = l_aaaaaa
    t3_aaaaaa = t_aaaaaa
    l3_aabaab = l_aabaab
    t3_aabaab = t_aabaab
    l3_abbabb = l_abbabb
    t3_abbabb = t_abbabb
    l3_bbbbbb = l_aaaaaa
    t3_bbbbbb = t_aaaaaa
    result = np.zeros(())
    _tmp0 = einsum('ijkcba,cbaijk->', l3_aaaaaa, t3_aaaaaa, optimize=True)
    result += 0.0277777777777778 * _tmp0
    _tmp1 = einsum('ijkcba,cbaijk->', l3_aabaab, t3_aabaab, optimize=True)
    result += 0.0277777777777778 * _tmp1
    result += 0.0277777777777778 * _tmp1
    result += 0.0277777777777778 * _tmp1
    result += 0.0277777777777778 * _tmp1
    result += 0.0277777777777778 * _tmp1
    result += 0.0277777777777778 * _tmp1
    _tmp2 = einsum('ijkcba,cbaijk->', l3_abbabb, t3_abbabb, optimize=True)
    result += 0.0277777777777778 * _tmp2
    result += 0.0277777777777778 * _tmp2
    result += 0.0277777777777778 * _tmp2
    result += 0.0277777777777778 * _tmp1
    result += 0.0277777777777778 * _tmp1
    result += 0.0277777777777778 * _tmp1
    result += 0.0277777777777778 * _tmp2
    result += 0.0277777777777778 * _tmp2
    result += 0.0277777777777778 * _tmp2
    result += 0.0277777777777778 * _tmp2
    result += 0.0277777777777778 * _tmp2
    result += 0.0277777777777778 * _tmp2
    _tmp3 = einsum('ijkcba,cbaijk->', l3_bbbbbb, t3_bbbbbb, optimize=True)
    result += 0.0277777777777778 * _tmp3
    return result


def overlap4_restricted(l_aaaaaaaa, l_aaabaaab, l_aabbaabb, l_abbbabbb, t_aaaaaaaa, t_aaabaaab, t_aabbaabb, t_abbbabbb):
    l4_aaaaaaaa = l_aaaaaaaa
    t4_aaaaaaaa = t_aaaaaaaa
    l4_aaabaaab = l_aaabaaab
    t4_aaabaaab = t_aaabaaab
    l4_aabbaabb = l_aabbaabb
    t4_aabbaabb = t_aabbaabb
    l4_abbbabbb = l_abbbabbb
    t4_abbbabbb = t_abbbabbb
    l4_bbbbbbbb = l_aaaaaaaa
    t4_bbbbbbbb = t_aaaaaaaa
    result = np.zeros(())
    _tmp0 = einsum('ijkldcba,dcbaijkl->', l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result += 0.00173611111111111 * _tmp0
    _tmp1 = einsum('ijkldcba,dcbaijkl->', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    _tmp2 = einsum('ijkldcba,dcbaijkl->', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    _tmp3 = einsum('ijkldcba,dcbaijkl->', l4_abbbabbb, t4_abbbabbb, optimize=True)
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp1
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp2
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    result += 0.00173611111111111 * _tmp3
    _tmp4 = einsum('ijkldcba,dcbaijkl->', l4_bbbbbbbb, t4_bbbbbbbb, optimize=True)
    result += 0.00173611111111111 * _tmp4
    return result


def m2_oo_11_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_2_1_aaaa, t_2_1_abab):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m2_oo_20_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    result = np.zeros((no, no))
    return result


def m2_oo_02_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_2_aa):
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    result = np.zeros((no, no))
    return result


def m3_oo_12_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_2_aa, t_2_2_aaaa, t_2_2_abab, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_aaaaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m3_oo_21_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa, l_2_2_aaaa, l_2_2_abab, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, t_2_1_aaaa, t_2_1_abab):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_aaaa
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_aaaaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m3_oo_30_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_3_aa, l_2_3_aaaa, l_2_3_abab, l_3_3_aaaaaa, l_3_3_aabaab, l_3_3_abbabb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_aa
    l2_aaaa = l_2_3_aaaa
    l2_abab = l_2_3_abab
    l2_bbbb = l_2_3_aaaa
    l3_aaaaaa = l_3_3_aaaaaa
    l3_aabaab = l_3_3_aabaab
    l3_abbabb = l_3_3_abbabb
    l3_bbbbbb = l_3_3_aaaaaa
    result = np.zeros((no, no))
    return result


def m3_oo_03_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_3_aa, t_2_3_aaaa, t_2_3_abab, t_3_3_aaaaaa, t_3_3_aabaab, t_3_3_abbabb):
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_aa
    t2_aaaa = t_2_3_aaaa
    t2_abab = t_2_3_abab
    t2_bbbb = t_2_3_aaaa
    t3_aaaaaa = t_3_3_aaaaaa
    t3_aabaab = t_3_3_aabaab
    t3_abbabb = t_3_3_abbabb
    t3_bbbbbb = t_3_3_aaaaaa
    result = np.zeros((no, no))
    return result


def m2_vv_11_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_2_1_aaaa, t_2_1_abab):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m2_vv_20_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    result = np.zeros((nv, nv))
    return result


def m2_vv_02_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_2_aa):
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    result = np.zeros((nv, nv))
    return result


def m3_vv_12_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_2_aa, t_2_2_aaaa, t_2_2_abab, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_aaaaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m3_vv_21_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa, l_2_2_aaaa, l_2_2_abab, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, t_2_1_aaaa, t_2_1_abab):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_aaaa
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_aaaaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m3_vv_30_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_3_aa, l_2_3_aaaa, l_2_3_abab, l_3_3_aaaaaa, l_3_3_aabaab, l_3_3_abbabb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_aa
    l2_aaaa = l_2_3_aaaa
    l2_abab = l_2_3_abab
    l2_bbbb = l_2_3_aaaa
    l3_aaaaaa = l_3_3_aaaaaa
    l3_aabaab = l_3_3_aabaab
    l3_abbabb = l_3_3_abbabb
    l3_bbbbbb = l_3_3_aaaaaa
    result = np.zeros((nv, nv))
    return result


def m3_vv_03_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_3_aa, t_2_3_aaaa, t_2_3_abab, t_3_3_aaaaaa, t_3_3_aabaab, t_3_3_abbabb):
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_aa
    t2_aaaa = t_2_3_aaaa
    t2_abab = t_2_3_abab
    t2_bbbb = t_2_3_aaaa
    t3_aaaaaa = t_3_3_aaaaaa
    t3_aabaab = t_3_3_aabaab
    t3_abbabb = t_3_3_abbabb
    t3_bbbbbb = t_3_3_aaaaaa
    result = np.zeros((nv, nv))
    return result


def m2_ov_11_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_2_1_aaaa, t_2_1_abab):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((no, nv))
    return result


def m2_ov_20_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    result = np.zeros((no, nv))
    return result


def m2_ov_02_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_2_aa):
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    result = np.zeros((no, nv))
    _tmp0 = einsum('em->me', t1_aa)
    result += 1 * _tmp0
    return result


def m3_ov_12_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_2_aa, t_2_2_aaaa, t_2_2_abab, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_aaaaaa
    result = np.zeros((no, nv))
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


def m3_ov_21_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa, l_2_2_aaaa, l_2_2_abab, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, t_2_1_aaaa, t_2_1_abab):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_aaaa
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_aaaaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((no, nv))
    _tmp0 = einsum('ia,eaim->me', l1_aa, t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    _tmp1 = einsum('ia,eami->me', l1_bb, t2_abab, optimize=True)
    result += 1 * _tmp1
    return result


def m3_ov_30_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_3_aa, l_2_3_aaaa, l_2_3_abab, l_3_3_aaaaaa, l_3_3_aabaab, l_3_3_abbabb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_aa
    l2_aaaa = l_2_3_aaaa
    l2_abab = l_2_3_abab
    l2_bbbb = l_2_3_aaaa
    l3_aaaaaa = l_3_3_aaaaaa
    l3_aabaab = l_3_3_aabaab
    l3_abbabb = l_3_3_abbabb
    l3_bbbbbb = l_3_3_aaaaaa
    result = np.zeros((no, nv))
    return result


def m3_ov_03_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_3_aa, t_2_3_aaaa, t_2_3_abab, t_3_3_aaaaaa, t_3_3_aabaab, t_3_3_abbabb):
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_aa
    t2_aaaa = t_2_3_aaaa
    t2_abab = t_2_3_abab
    t2_bbbb = t_2_3_aaaa
    t3_aaaaaa = t_3_3_aaaaaa
    t3_aabaab = t_3_3_aabaab
    t3_abbabb = t_3_3_abbabb
    t3_bbbbbb = t_3_3_aaaaaa
    result = np.zeros((no, nv))
    _tmp0 = einsum('em->me', t1_aa)
    result += 1 * _tmp0
    return result


def m3_ov_12_restricted_no_t3(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_2_aa, t_2_2_aaaa, t_2_2_abab):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    result = np.zeros((no, nv))
    return result


def m4_oo_13_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_3_aa, t_2_3_aaaa, t_2_3_abab, t_3_3_aaaaaa, t_3_3_aabaab, t_3_3_abbabb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_aa
    t2_aaaa = t_2_3_aaaa
    t2_abab = t_2_3_abab
    t2_bbbb = t_2_3_aaaa
    t3_aaaaaa = t_3_3_aaaaaa
    t3_aabaab = t_3_3_aabaab
    t3_abbabb = t_3_3_abbabb
    t3_bbbbbb = t_3_3_aaaaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m4_oo_31_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_3_aa, l_2_3_aaaa, l_2_3_abab, l_3_3_aaaaaa, l_3_3_aabaab, l_3_3_abbabb, t_2_1_aaaa, t_2_1_abab):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_aa
    l2_aaaa = l_2_3_aaaa
    l2_abab = l_2_3_abab
    l2_bbbb = l_2_3_aaaa
    l3_aaaaaa = l_3_3_aaaaaa
    l3_aabaab = l_3_3_aabaab
    l3_abbabb = l_3_3_abbabb
    l3_bbbbbb = l_3_3_aaaaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m4_oo_22_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa, l_2_2_aaaa, l_2_2_abab, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_4_2_aaaaaaaa, l_4_2_aaabaaab, l_4_2_aabbaabb, l_4_2_abbbabbb, t_1_2_aa, t_2_2_aaaa, t_2_2_abab, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_4_2_aaaaaaaa, t_4_2_aaabaaab, t_4_2_aabbaabb, t_4_2_abbbabbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_aaaa
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_aaaaaa
    l4_aaaaaaaa = l_4_2_aaaaaaaa
    l4_aaabaaab = l_4_2_aaabaaab
    l4_aabbaabb = l_4_2_aabbaabb
    l4_abbbabbb = l_4_2_abbbabbb
    l4_bbbbbbbb = l_4_2_aaaaaaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_aaaaaa
    t4_aaaaaaaa = t_4_2_aaaaaaaa
    t4_aaabaaab = t_4_2_aaabaaab
    t4_aabbaabb = t_4_2_aabbaabb
    t4_abbbabbb = t_4_2_abbbabbb
    t4_bbbbbbbb = t_4_2_aaaaaaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ia,ai->mn', d_aa[o, o], l1_aa, t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('mn,ia,ai->mn', d_aa[o, o], l1_bb, t1_bb, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('na,am->mn', l1_aa, t1_aa, optimize=True)
    result -= 1 * _tmp2
    _tmp3 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp3
    _tmp4 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    _tmp5 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp5
    _tmp6 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp6
    _tmp7 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp7
    result -= 0.5 * _tmp7
    _tmp8 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_aaaaaa, t3_aaaaaa, optimize=True)
    result += 0.0277777777777778 * _tmp8
    _tmp9 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_aabaab, t3_aabaab, optimize=True)
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    _tmp10 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_abbabb, t3_abbabb, optimize=True)
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    _tmp11 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_bbbbbb, t3_bbbbbb, optimize=True)
    result += 0.0277777777777778 * _tmp11
    _tmp12 = einsum('ijncba,cbaijm->mn', l3_aaaaaa, t3_aaaaaa, optimize=True)
    result -= 0.0833333333333333 * _tmp12
    _tmp13 = einsum('injcba,cbaimj->mn', l3_aabaab, t3_aabaab, optimize=True)
    result -= 0.0833333333333333 * _tmp13
    result -= 0.0833333333333333 * _tmp13
    result -= 0.0833333333333333 * _tmp13
    _tmp14 = einsum('njicba,cbamji->mn', l3_aabaab, t3_aabaab, optimize=True)
    result -= 0.0833333333333333 * _tmp14
    result -= 0.0833333333333333 * _tmp14
    result -= 0.0833333333333333 * _tmp14
    _tmp15 = einsum('njicba,cbamji->mn', l3_abbabb, t3_abbabb, optimize=True)
    result -= 0.0833333333333333 * _tmp15
    result -= 0.0833333333333333 * _tmp15
    result -= 0.0833333333333333 * _tmp15
    _tmp16 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result += 0.00173611111111111 * _tmp16
    _tmp17 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_aaabaaab, t4_aaabaaab, optimize=True)
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    _tmp18 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_aabbaabb, t4_aabbaabb, optimize=True)
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    _tmp19 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_abbbabbb, t4_abbbabbb, optimize=True)
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    _tmp20 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_bbbbbbbb, t4_bbbbbbbb, optimize=True)
    result += 0.00173611111111111 * _tmp20
    _tmp21 = einsum('ijkndcba,dcbaijkm->mn', l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result -= 0.00694444444444444 * _tmp21
    _tmp22 = einsum('ijnkdcba,dcbaijmk->mn', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result -= 0.00694444444444444 * _tmp22
    result -= 0.00694444444444444 * _tmp22
    result -= 0.00694444444444444 * _tmp22
    result -= 0.00694444444444444 * _tmp22
    _tmp23 = einsum('inkjdcba,dcbaimkj->mn', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result -= 0.00694444444444444 * _tmp23
    result -= 0.00694444444444444 * _tmp23
    result -= 0.00694444444444444 * _tmp23
    result -= 0.00694444444444444 * _tmp23
    _tmp24 = einsum('inkjdcba,dcbaimkj->mn', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    _tmp25 = einsum('njkidcba,dcbamjki->mn', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result -= 0.00694444444444444 * _tmp25
    result -= 0.00694444444444444 * _tmp25
    result -= 0.00694444444444444 * _tmp25
    result -= 0.00694444444444444 * _tmp25
    _tmp26 = einsum('njkidcba,dcbamjki->mn', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    _tmp27 = einsum('njkidcba,dcbamjki->mn', l4_abbbabbb, t4_abbbabbb, optimize=True)
    result -= 0.00694444444444444 * _tmp27
    result -= 0.00694444444444444 * _tmp27
    result -= 0.00694444444444444 * _tmp27
    result -= 0.00694444444444444 * _tmp27
    return result


def m4_oo_40_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_4_aa):
    l1_aa = l_1_4_aa
    l1_bb = l_1_4_aa
    result = np.zeros((no, no))
    return result


def m4_oo_04_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_4_aa):
    t1_aa = t_1_4_aa
    t1_bb = t_1_4_aa
    result = np.zeros((no, no))
    return result


def m4_vv_13_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_3_aa, t_2_3_aaaa, t_2_3_abab, t_3_3_aaaaaa, t_3_3_aabaab, t_3_3_abbabb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_aa
    t2_aaaa = t_2_3_aaaa
    t2_abab = t_2_3_abab
    t2_bbbb = t_2_3_aaaa
    t3_aaaaaa = t_3_3_aaaaaa
    t3_aabaab = t_3_3_aabaab
    t3_abbabb = t_3_3_abbabb
    t3_bbbbbb = t_3_3_aaaaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m4_vv_31_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_3_aa, l_2_3_aaaa, l_2_3_abab, l_3_3_aaaaaa, l_3_3_aabaab, l_3_3_abbabb, t_2_1_aaaa, t_2_1_abab):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_aa
    l2_aaaa = l_2_3_aaaa
    l2_abab = l_2_3_abab
    l2_bbbb = l_2_3_aaaa
    l3_aaaaaa = l_3_3_aaaaaa
    l3_aabaab = l_3_3_aabaab
    l3_abbabb = l_3_3_abbabb
    l3_bbbbbb = l_3_3_aaaaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m4_vv_22_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa, l_2_2_aaaa, l_2_2_abab, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_4_2_aaaaaaaa, l_4_2_aaabaaab, l_4_2_aabbaabb, l_4_2_abbbabbb, t_1_2_aa, t_2_2_aaaa, t_2_2_abab, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_4_2_aaaaaaaa, t_4_2_aaabaaab, t_4_2_aabbaabb, t_4_2_abbbabbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_aaaa
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_aaaaaa
    l4_aaaaaaaa = l_4_2_aaaaaaaa
    l4_aaabaaab = l_4_2_aaabaaab
    l4_aabbaabb = l_4_2_aabbaabb
    l4_abbbabbb = l_4_2_abbbabbb
    l4_bbbbbbbb = l_4_2_aaaaaaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_aaaaaa
    t4_aaaaaaaa = t_4_2_aaaaaaaa
    t4_aaabaaab = t_4_2_aaabaaab
    t4_aabbaabb = t_4_2_aabbaabb
    t4_abbbabbb = t_4_2_abbbabbb
    t4_bbbbbbbb = t_4_2_aaaaaaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ie,fi->ef', l1_aa, t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp1
    _tmp2 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp2
    result += 0.5 * _tmp2
    _tmp3 = einsum('ijkeba,fbaijk->ef', l3_aaaaaa, t3_aaaaaa, optimize=True)
    result += 0.0833333333333333 * _tmp3
    _tmp4 = einsum('ijkeba,fbaijk->ef', l3_aabaab, t3_aabaab, optimize=True)
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    _tmp5 = einsum('ijkeba,fbaijk->ef', l3_abbabb, t3_abbabb, optimize=True)
    result += 0.0833333333333333 * _tmp5
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp5
    result += 0.0833333333333333 * _tmp5
    _tmp6 = einsum('ijklecba,fcbaijkl->ef', l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result += 0.00694444444444444 * _tmp6
    _tmp7 = einsum('ijklecba,fcbaijkl->ef', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    _tmp8 = einsum('ijklecba,fcbaijkl->ef', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    _tmp9 = einsum('ijklecba,fcbaijkl->ef', l4_abbbabbb, t4_abbbabbb, optimize=True)
    result += 0.00694444444444444 * _tmp9
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp9
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp9
    result += 0.00694444444444444 * _tmp9
    return result


def m4_vv_40_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_4_aa):
    l1_aa = l_1_4_aa
    l1_bb = l_1_4_aa
    result = np.zeros((nv, nv))
    return result


def m4_vv_04_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_4_aa):
    t1_aa = t_1_4_aa
    t1_bb = t_1_4_aa
    result = np.zeros((nv, nv))
    return result


def m4_ov_13_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_3_aa, t_2_3_aaaa, t_2_3_abab, t_3_3_aaaaaa, t_3_3_aabaab, t_3_3_abbabb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_aa
    t2_aaaa = t_2_3_aaaa
    t2_abab = t_2_3_abab
    t2_bbbb = t_2_3_aaaa
    t3_aaaaaa = t_3_3_aaaaaa
    t3_aabaab = t_3_3_aabaab
    t3_abbabb = t_3_3_abbabb
    t3_bbbbbb = t_3_3_aaaaaa
    result = np.zeros((no, nv))
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


def m4_ov_31_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_3_aa, l_2_3_aaaa, l_2_3_abab, l_3_3_aaaaaa, l_3_3_aabaab, l_3_3_abbabb, t_2_1_aaaa, t_2_1_abab):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_aa
    l2_aaaa = l_2_3_aaaa
    l2_abab = l_2_3_abab
    l2_bbbb = l_2_3_aaaa
    l3_aaaaaa = l_3_3_aaaaaa
    l3_aabaab = l_3_3_aabaab
    l3_abbabb = l_3_3_abbabb
    l3_bbbbbb = l_3_3_aaaaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((no, nv))
    _tmp0 = einsum('ia,eaim->me', l1_aa, t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    _tmp1 = einsum('ia,eami->me', l1_bb, t2_abab, optimize=True)
    result += 1 * _tmp1
    return result


def m4_ov_22_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa, l_2_2_aaaa, l_2_2_abab, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_4_2_aaaaaaaa, l_4_2_aaabaaab, l_4_2_aabbaabb, l_4_2_abbbabbb, t_1_2_aa, t_2_2_aaaa, t_2_2_abab, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_4_2_aaaaaaaa, t_4_2_aaabaaab, t_4_2_aabbaabb, t_4_2_abbbabbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_aaaa
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_aaaaaa
    l4_aaaaaaaa = l_4_2_aaaaaaaa
    l4_aaabaaab = l_4_2_aaabaaab
    l4_aabbaabb = l_4_2_aabbaabb
    l4_abbbabbb = l_4_2_abbbabbb
    l4_bbbbbbbb = l_4_2_aaaaaaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_aaaaaa
    t4_aaaaaaaa = t_4_2_aaaaaaaa
    t4_aaabaaab = t_4_2_aaabaaab
    t4_aabbaabb = t_4_2_aabbaabb
    t4_abbbabbb = t_4_2_abbbabbb
    t4_bbbbbbbb = t_4_2_aaaaaaaa
    result = np.zeros((no, nv))
    _tmp0 = einsum('ia,eaim->me', l1_aa, t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    _tmp1 = einsum('ia,eami->me', l1_bb, t2_abab, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('ijba,ebaijm->me', l2_aaaa, t3_aaaaaa, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('ijba,ebaimj->me', l2_abab, t3_aabaab, optimize=True)
    result -= 0.25 * _tmp3
    result -= 0.25 * _tmp3
    _tmp4 = einsum('jiba,ebamji->me', l2_abab, t3_aabaab, optimize=True)
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    _tmp5 = einsum('ijba,ebamji->me', l2_bbbb, t3_abbabb, optimize=True)
    result -= 0.25 * _tmp5
    _tmp6 = einsum('ijkcba,ecbaijkm->me', l3_aaaaaa, t4_aaaaaaaa, optimize=True)
    result -= 0.0277777777777778 * _tmp6
    _tmp7 = einsum('ijkcba,ecbaijmk->me', l3_aabaab, t4_aaabaaab, optimize=True)
    result += 0.0277777777777778 * _tmp7
    result += 0.0277777777777778 * _tmp7
    result += 0.0277777777777778 * _tmp7
    _tmp8 = einsum('ikjcba,ecbaimkj->me', l3_aabaab, t4_aaabaaab, optimize=True)
    result -= 0.0277777777777778 * _tmp8
    result -= 0.0277777777777778 * _tmp8
    result -= 0.0277777777777778 * _tmp8
    _tmp9 = einsum('ijkcba,ecbaimkj->me', l3_abbabb, t4_aabbaabb, optimize=True)
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    _tmp10 = einsum('kjicba,ecbamjki->me', l3_aabaab, t4_aaabaaab, optimize=True)
    result -= 0.0277777777777778 * _tmp10
    result -= 0.0277777777777778 * _tmp10
    result -= 0.0277777777777778 * _tmp10
    _tmp11 = einsum('jikcba,ecbamjki->me', l3_abbabb, t4_aabbaabb, optimize=True)
    result -= 0.0277777777777778 * _tmp11
    result -= 0.0277777777777778 * _tmp11
    result -= 0.0277777777777778 * _tmp11
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    _tmp12 = einsum('ijkcba,ecbamjki->me', l3_bbbbbb, t4_abbbabbb, optimize=True)
    result += 0.0277777777777778 * _tmp12
    return result


def m4_ov_40_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_4_aa):
    l1_aa = l_1_4_aa
    l1_bb = l_1_4_aa
    result = np.zeros((no, nv))
    return result


def m4_ov_04_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_4_aa):
    t1_aa = t_1_4_aa
    t1_bb = t_1_4_aa
    result = np.zeros((no, nv))
    _tmp0 = einsum('em->me', t1_aa)
    result += 1 * _tmp0
    return result


def m5_oo_14_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_4_aa, t_2_4_aaaa, t_2_4_abab, t_3_4_aaaaaa, t_3_4_aabaab, t_3_4_abbabb, t_4_4_aaaaaaaa, t_4_4_aaabaaab, t_4_4_aabbaabb, t_4_4_abbbabbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_4_aa
    t1_bb = t_1_4_aa
    t2_aaaa = t_2_4_aaaa
    t2_abab = t_2_4_abab
    t2_bbbb = t_2_4_aaaa
    t3_aaaaaa = t_3_4_aaaaaa
    t3_aabaab = t_3_4_aabaab
    t3_abbabb = t_3_4_abbabb
    t3_bbbbbb = t_3_4_aaaaaa
    t4_aaaaaaaa = t_4_4_aaaaaaaa
    t4_aaabaaab = t_4_4_aaabaaab
    t4_aabbaabb = t_4_4_aabbaabb
    t4_abbbabbb = t_4_4_abbbabbb
    t4_bbbbbbbb = t_4_4_aaaaaaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m5_oo_41_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_4_aa, l_2_4_aaaa, l_2_4_abab, l_3_4_aaaaaa, l_3_4_aabaab, l_3_4_abbabb, l_4_4_aaaaaaaa, l_4_4_aaabaaab, l_4_4_aabbaabb, l_4_4_abbbabbb, t_2_1_aaaa, t_2_1_abab):
    l1_aa = l_1_4_aa
    l1_bb = l_1_4_aa
    l2_aaaa = l_2_4_aaaa
    l2_abab = l_2_4_abab
    l2_bbbb = l_2_4_aaaa
    l3_aaaaaa = l_3_4_aaaaaa
    l3_aabaab = l_3_4_aabaab
    l3_abbabb = l_3_4_abbabb
    l3_bbbbbb = l_3_4_aaaaaa
    l4_aaaaaaaa = l_4_4_aaaaaaaa
    l4_aaabaaab = l_4_4_aaabaaab
    l4_aabbaabb = l_4_4_aabbaabb
    l4_abbbabbb = l_4_4_abbbabbb
    l4_bbbbbbbb = l_4_4_aaaaaaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    result += 0.25 * _tmp1
    _tmp2 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp4
    result -= 0.5 * _tmp4
    return result


def m5_oo_23_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa, l_2_2_aaaa, l_2_2_abab, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_4_2_aaaaaaaa, l_4_2_aaabaaab, l_4_2_aabbaabb, l_4_2_abbbabbb, t_1_3_aa, t_2_3_aaaa, t_2_3_abab, t_3_3_aaaaaa, t_3_3_aabaab, t_3_3_abbabb, t_4_3_aaaaaaaa, t_4_3_aaabaaab, t_4_3_aabbaabb, t_4_3_abbbabbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_aaaa
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_aaaaaa
    l4_aaaaaaaa = l_4_2_aaaaaaaa
    l4_aaabaaab = l_4_2_aaabaaab
    l4_aabbaabb = l_4_2_aabbaabb
    l4_abbbabbb = l_4_2_abbbabbb
    l4_bbbbbbbb = l_4_2_aaaaaaaa
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_aa
    t2_aaaa = t_2_3_aaaa
    t2_abab = t_2_3_abab
    t2_bbbb = t_2_3_aaaa
    t3_aaaaaa = t_3_3_aaaaaa
    t3_aabaab = t_3_3_aabaab
    t3_abbabb = t_3_3_abbabb
    t3_bbbbbb = t_3_3_aaaaaa
    t4_aaaaaaaa = t_4_3_aaaaaaaa
    t4_aaabaaab = t_4_3_aaabaaab
    t4_aabbaabb = t_4_3_aabbaabb
    t4_abbbabbb = t_4_3_abbbabbb
    t4_bbbbbbbb = t_4_3_aaaaaaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ia,ai->mn', d_aa[o, o], l1_aa, t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('mn,ia,ai->mn', d_aa[o, o], l1_bb, t1_bb, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('na,am->mn', l1_aa, t1_aa, optimize=True)
    result -= 1 * _tmp2
    _tmp3 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp3
    _tmp4 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    _tmp5 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp5
    _tmp6 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp6
    _tmp7 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp7
    result -= 0.5 * _tmp7
    _tmp8 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_aaaaaa, t3_aaaaaa, optimize=True)
    result += 0.0277777777777778 * _tmp8
    _tmp9 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_aabaab, t3_aabaab, optimize=True)
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    _tmp10 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_abbabb, t3_abbabb, optimize=True)
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    _tmp11 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_bbbbbb, t3_bbbbbb, optimize=True)
    result += 0.0277777777777778 * _tmp11
    _tmp12 = einsum('ijncba,cbaijm->mn', l3_aaaaaa, t3_aaaaaa, optimize=True)
    result -= 0.0833333333333333 * _tmp12
    _tmp13 = einsum('injcba,cbaimj->mn', l3_aabaab, t3_aabaab, optimize=True)
    result -= 0.0833333333333333 * _tmp13
    result -= 0.0833333333333333 * _tmp13
    result -= 0.0833333333333333 * _tmp13
    _tmp14 = einsum('njicba,cbamji->mn', l3_aabaab, t3_aabaab, optimize=True)
    result -= 0.0833333333333333 * _tmp14
    result -= 0.0833333333333333 * _tmp14
    result -= 0.0833333333333333 * _tmp14
    _tmp15 = einsum('njicba,cbamji->mn', l3_abbabb, t3_abbabb, optimize=True)
    result -= 0.0833333333333333 * _tmp15
    result -= 0.0833333333333333 * _tmp15
    result -= 0.0833333333333333 * _tmp15
    _tmp16 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result += 0.00173611111111111 * _tmp16
    _tmp17 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_aaabaaab, t4_aaabaaab, optimize=True)
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    _tmp18 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_aabbaabb, t4_aabbaabb, optimize=True)
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    _tmp19 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_abbbabbb, t4_abbbabbb, optimize=True)
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    _tmp20 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_bbbbbbbb, t4_bbbbbbbb, optimize=True)
    result += 0.00173611111111111 * _tmp20
    _tmp21 = einsum('ijkndcba,dcbaijkm->mn', l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result -= 0.00694444444444444 * _tmp21
    _tmp22 = einsum('ijnkdcba,dcbaijmk->mn', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result -= 0.00694444444444444 * _tmp22
    result -= 0.00694444444444444 * _tmp22
    result -= 0.00694444444444444 * _tmp22
    result -= 0.00694444444444444 * _tmp22
    _tmp23 = einsum('inkjdcba,dcbaimkj->mn', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result -= 0.00694444444444444 * _tmp23
    result -= 0.00694444444444444 * _tmp23
    result -= 0.00694444444444444 * _tmp23
    result -= 0.00694444444444444 * _tmp23
    _tmp24 = einsum('inkjdcba,dcbaimkj->mn', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    _tmp25 = einsum('njkidcba,dcbamjki->mn', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result -= 0.00694444444444444 * _tmp25
    result -= 0.00694444444444444 * _tmp25
    result -= 0.00694444444444444 * _tmp25
    result -= 0.00694444444444444 * _tmp25
    _tmp26 = einsum('njkidcba,dcbamjki->mn', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    _tmp27 = einsum('njkidcba,dcbamjki->mn', l4_abbbabbb, t4_abbbabbb, optimize=True)
    result -= 0.00694444444444444 * _tmp27
    result -= 0.00694444444444444 * _tmp27
    result -= 0.00694444444444444 * _tmp27
    result -= 0.00694444444444444 * _tmp27
    return result


def m5_oo_32_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_3_aa, l_2_3_aaaa, l_2_3_abab, l_3_3_aaaaaa, l_3_3_aabaab, l_3_3_abbabb, l_4_3_aaaaaaaa, l_4_3_aaabaaab, l_4_3_aabbaabb, l_4_3_abbbabbb, t_1_2_aa, t_2_2_aaaa, t_2_2_abab, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_4_2_aaaaaaaa, t_4_2_aaabaaab, t_4_2_aabbaabb, t_4_2_abbbabbb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_aa
    l2_aaaa = l_2_3_aaaa
    l2_abab = l_2_3_abab
    l2_bbbb = l_2_3_aaaa
    l3_aaaaaa = l_3_3_aaaaaa
    l3_aabaab = l_3_3_aabaab
    l3_abbabb = l_3_3_abbabb
    l3_bbbbbb = l_3_3_aaaaaa
    l4_aaaaaaaa = l_4_3_aaaaaaaa
    l4_aaabaaab = l_4_3_aaabaaab
    l4_aabbaabb = l_4_3_aabbaabb
    l4_abbbabbb = l_4_3_abbbabbb
    l4_bbbbbbbb = l_4_3_aaaaaaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_aaaaaa
    t4_aaaaaaaa = t_4_2_aaaaaaaa
    t4_aaabaaab = t_4_2_aaabaaab
    t4_aabbaabb = t_4_2_aabbaabb
    t4_abbbabbb = t_4_2_abbbabbb
    t4_bbbbbbbb = t_4_2_aaaaaaaa
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ia,ai->mn', d_aa[o, o], l1_aa, t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('mn,ia,ai->mn', d_aa[o, o], l1_bb, t1_bb, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('na,am->mn', l1_aa, t1_aa, optimize=True)
    result -= 1 * _tmp2
    _tmp3 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_aaaa, t2_aaaa, optimize=True)
    result += 0.25 * _tmp3
    _tmp4 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_abab, t2_abab, optimize=True)
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    _tmp5 = einsum('mn,ijba,baij->mn', d_aa[o, o], l2_bbbb, t2_bbbb, optimize=True)
    result += 0.25 * _tmp5
    _tmp6 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    result -= 0.5 * _tmp6
    _tmp7 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    result -= 0.5 * _tmp7
    result -= 0.5 * _tmp7
    _tmp8 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_aaaaaa, t3_aaaaaa, optimize=True)
    result += 0.0277777777777778 * _tmp8
    _tmp9 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_aabaab, t3_aabaab, optimize=True)
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    _tmp10 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_abbabb, t3_abbabb, optimize=True)
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    result += 0.0277777777777778 * _tmp10
    _tmp11 = einsum('mn,ijkcba,cbaijk->mn', d_aa[o, o], l3_bbbbbb, t3_bbbbbb, optimize=True)
    result += 0.0277777777777778 * _tmp11
    _tmp12 = einsum('ijncba,cbaijm->mn', l3_aaaaaa, t3_aaaaaa, optimize=True)
    result -= 0.0833333333333333 * _tmp12
    _tmp13 = einsum('injcba,cbaimj->mn', l3_aabaab, t3_aabaab, optimize=True)
    result -= 0.0833333333333333 * _tmp13
    result -= 0.0833333333333333 * _tmp13
    result -= 0.0833333333333333 * _tmp13
    _tmp14 = einsum('njicba,cbamji->mn', l3_aabaab, t3_aabaab, optimize=True)
    result -= 0.0833333333333333 * _tmp14
    result -= 0.0833333333333333 * _tmp14
    result -= 0.0833333333333333 * _tmp14
    _tmp15 = einsum('njicba,cbamji->mn', l3_abbabb, t3_abbabb, optimize=True)
    result -= 0.0833333333333333 * _tmp15
    result -= 0.0833333333333333 * _tmp15
    result -= 0.0833333333333333 * _tmp15
    _tmp16 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result += 0.00173611111111111 * _tmp16
    _tmp17 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_aaabaaab, t4_aaabaaab, optimize=True)
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    _tmp18 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_aabbaabb, t4_aabbaabb, optimize=True)
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    _tmp19 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_abbbabbb, t4_abbbabbb, optimize=True)
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp17
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp18
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    result += 0.00173611111111111 * _tmp19
    _tmp20 = einsum('mn,ijkldcba,dcbaijkl->mn', d_aa[o, o], l4_bbbbbbbb, t4_bbbbbbbb, optimize=True)
    result += 0.00173611111111111 * _tmp20
    _tmp21 = einsum('ijkndcba,dcbaijkm->mn', l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result -= 0.00694444444444444 * _tmp21
    _tmp22 = einsum('ijnkdcba,dcbaijmk->mn', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result -= 0.00694444444444444 * _tmp22
    result -= 0.00694444444444444 * _tmp22
    result -= 0.00694444444444444 * _tmp22
    result -= 0.00694444444444444 * _tmp22
    _tmp23 = einsum('inkjdcba,dcbaimkj->mn', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result -= 0.00694444444444444 * _tmp23
    result -= 0.00694444444444444 * _tmp23
    result -= 0.00694444444444444 * _tmp23
    result -= 0.00694444444444444 * _tmp23
    _tmp24 = einsum('inkjdcba,dcbaimkj->mn', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    _tmp25 = einsum('njkidcba,dcbamjki->mn', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result -= 0.00694444444444444 * _tmp25
    result -= 0.00694444444444444 * _tmp25
    result -= 0.00694444444444444 * _tmp25
    result -= 0.00694444444444444 * _tmp25
    _tmp26 = einsum('njkidcba,dcbamjki->mn', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp26
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    result -= 0.00694444444444444 * _tmp24
    _tmp27 = einsum('njkidcba,dcbamjki->mn', l4_abbbabbb, t4_abbbabbb, optimize=True)
    result -= 0.00694444444444444 * _tmp27
    result -= 0.00694444444444444 * _tmp27
    result -= 0.00694444444444444 * _tmp27
    result -= 0.00694444444444444 * _tmp27
    return result


def m5_oo_50_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_5_aa):
    l1_aa = l_1_5_aa
    l1_bb = l_1_5_aa
    result = np.zeros((no, no))
    return result


def m5_oo_05_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_5_aa):
    t1_aa = t_1_5_aa
    t1_bb = t_1_5_aa
    result = np.zeros((no, no))
    return result


def m5_vv_14_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_4_aa, t_2_4_aaaa, t_2_4_abab, t_3_4_aaaaaa, t_3_4_aabaab, t_3_4_abbabb, t_4_4_aaaaaaaa, t_4_4_aaabaaab, t_4_4_aabbaabb, t_4_4_abbbabbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_4_aa
    t1_bb = t_1_4_aa
    t2_aaaa = t_2_4_aaaa
    t2_abab = t_2_4_abab
    t2_bbbb = t_2_4_aaaa
    t3_aaaaaa = t_3_4_aaaaaa
    t3_aabaab = t_3_4_aabaab
    t3_abbabb = t_3_4_abbabb
    t3_bbbbbb = t_3_4_aaaaaa
    t4_aaaaaaaa = t_4_4_aaaaaaaa
    t4_aaabaaab = t_4_4_aaabaaab
    t4_aabbaabb = t_4_4_aabbaabb
    t4_abbbabbb = t_4_4_abbbabbb
    t4_bbbbbbbb = t_4_4_aaaaaaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m5_vv_41_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_4_aa, l_2_4_aaaa, l_2_4_abab, l_3_4_aaaaaa, l_3_4_aabaab, l_3_4_abbabb, l_4_4_aaaaaaaa, l_4_4_aaabaaab, l_4_4_aabbaabb, l_4_4_abbbabbb, t_2_1_aaaa, t_2_1_abab):
    l1_aa = l_1_4_aa
    l1_bb = l_1_4_aa
    l2_aaaa = l_2_4_aaaa
    l2_abab = l_2_4_abab
    l2_bbbb = l_2_4_aaaa
    l3_aaaaaa = l_3_4_aaaaaa
    l3_aabaab = l_3_4_aabaab
    l3_abbabb = l_3_4_abbabb
    l3_bbbbbb = l_3_4_aaaaaa
    l4_aaaaaaaa = l_4_4_aaaaaaaa
    l4_aaabaaab = l_4_4_aaabaaab
    l4_aabbaabb = l_4_4_aabbaabb
    l4_abbbabbb = l_4_4_abbbabbb
    l4_bbbbbbbb = l_4_4_aaaaaaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp1
    result += 0.5 * _tmp1
    return result


def m5_vv_23_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa, l_2_2_aaaa, l_2_2_abab, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_4_2_aaaaaaaa, l_4_2_aaabaaab, l_4_2_aabbaabb, l_4_2_abbbabbb, t_1_3_aa, t_2_3_aaaa, t_2_3_abab, t_3_3_aaaaaa, t_3_3_aabaab, t_3_3_abbabb, t_4_3_aaaaaaaa, t_4_3_aaabaaab, t_4_3_aabbaabb, t_4_3_abbbabbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_aaaa
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_aaaaaa
    l4_aaaaaaaa = l_4_2_aaaaaaaa
    l4_aaabaaab = l_4_2_aaabaaab
    l4_aabbaabb = l_4_2_aabbaabb
    l4_abbbabbb = l_4_2_abbbabbb
    l4_bbbbbbbb = l_4_2_aaaaaaaa
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_aa
    t2_aaaa = t_2_3_aaaa
    t2_abab = t_2_3_abab
    t2_bbbb = t_2_3_aaaa
    t3_aaaaaa = t_3_3_aaaaaa
    t3_aabaab = t_3_3_aabaab
    t3_abbabb = t_3_3_abbabb
    t3_bbbbbb = t_3_3_aaaaaa
    t4_aaaaaaaa = t_4_3_aaaaaaaa
    t4_aaabaaab = t_4_3_aaabaaab
    t4_aabbaabb = t_4_3_aabbaabb
    t4_abbbabbb = t_4_3_abbbabbb
    t4_bbbbbbbb = t_4_3_aaaaaaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ie,fi->ef', l1_aa, t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp1
    _tmp2 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp2
    result += 0.5 * _tmp2
    _tmp3 = einsum('ijkeba,fbaijk->ef', l3_aaaaaa, t3_aaaaaa, optimize=True)
    result += 0.0833333333333333 * _tmp3
    _tmp4 = einsum('ijkeba,fbaijk->ef', l3_aabaab, t3_aabaab, optimize=True)
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    _tmp5 = einsum('ijkeba,fbaijk->ef', l3_abbabb, t3_abbabb, optimize=True)
    result += 0.0833333333333333 * _tmp5
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp5
    result += 0.0833333333333333 * _tmp5
    _tmp6 = einsum('ijklecba,fcbaijkl->ef', l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result += 0.00694444444444444 * _tmp6
    _tmp7 = einsum('ijklecba,fcbaijkl->ef', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    _tmp8 = einsum('ijklecba,fcbaijkl->ef', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    _tmp9 = einsum('ijklecba,fcbaijkl->ef', l4_abbbabbb, t4_abbbabbb, optimize=True)
    result += 0.00694444444444444 * _tmp9
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp9
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp9
    result += 0.00694444444444444 * _tmp9
    return result


def m5_vv_32_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_3_aa, l_2_3_aaaa, l_2_3_abab, l_3_3_aaaaaa, l_3_3_aabaab, l_3_3_abbabb, l_4_3_aaaaaaaa, l_4_3_aaabaaab, l_4_3_aabbaabb, l_4_3_abbbabbb, t_1_2_aa, t_2_2_aaaa, t_2_2_abab, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_4_2_aaaaaaaa, t_4_2_aaabaaab, t_4_2_aabbaabb, t_4_2_abbbabbb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_aa
    l2_aaaa = l_2_3_aaaa
    l2_abab = l_2_3_abab
    l2_bbbb = l_2_3_aaaa
    l3_aaaaaa = l_3_3_aaaaaa
    l3_aabaab = l_3_3_aabaab
    l3_abbabb = l_3_3_abbabb
    l3_bbbbbb = l_3_3_aaaaaa
    l4_aaaaaaaa = l_4_3_aaaaaaaa
    l4_aaabaaab = l_4_3_aaabaaab
    l4_aabbaabb = l_4_3_aabbaabb
    l4_abbbabbb = l_4_3_abbbabbb
    l4_bbbbbbbb = l_4_3_aaaaaaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_aaaaaa
    t4_aaaaaaaa = t_4_2_aaaaaaaa
    t4_aaabaaab = t_4_2_aaabaaab
    t4_aabbaabb = t_4_2_aabbaabb
    t4_abbbabbb = t_4_2_abbbabbb
    t4_bbbbbbbb = t_4_2_aaaaaaaa
    result = np.zeros((nv, nv))
    _tmp0 = einsum('ie,fi->ef', l1_aa, t1_aa, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    result += 0.5 * _tmp1
    _tmp2 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    result += 0.5 * _tmp2
    result += 0.5 * _tmp2
    _tmp3 = einsum('ijkeba,fbaijk->ef', l3_aaaaaa, t3_aaaaaa, optimize=True)
    result += 0.0833333333333333 * _tmp3
    _tmp4 = einsum('ijkeba,fbaijk->ef', l3_aabaab, t3_aabaab, optimize=True)
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    _tmp5 = einsum('ijkeba,fbaijk->ef', l3_abbabb, t3_abbabb, optimize=True)
    result += 0.0833333333333333 * _tmp5
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp4
    result += 0.0833333333333333 * _tmp5
    result += 0.0833333333333333 * _tmp5
    _tmp6 = einsum('ijklecba,fcbaijkl->ef', l4_aaaaaaaa, t4_aaaaaaaa, optimize=True)
    result += 0.00694444444444444 * _tmp6
    _tmp7 = einsum('ijklecba,fcbaijkl->ef', l4_aaabaaab, t4_aaabaaab, optimize=True)
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    _tmp8 = einsum('ijklecba,fcbaijkl->ef', l4_aabbaabb, t4_aabbaabb, optimize=True)
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    _tmp9 = einsum('ijklecba,fcbaijkl->ef', l4_abbbabbb, t4_abbbabbb, optimize=True)
    result += 0.00694444444444444 * _tmp9
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp7
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp9
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp8
    result += 0.00694444444444444 * _tmp9
    result += 0.00694444444444444 * _tmp9
    return result


def m5_vv_50_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_5_aa):
    l1_aa = l_1_5_aa
    l1_bb = l_1_5_aa
    result = np.zeros((nv, nv))
    return result


def m5_vv_05_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_5_aa):
    t1_aa = t_1_5_aa
    t1_bb = t_1_5_aa
    result = np.zeros((nv, nv))
    return result


def m5_ov_14_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_2_1_aaaa, l_2_1_abab, t_1_4_aa, t_2_4_aaaa, t_2_4_abab, t_3_4_aaaaaa, t_3_4_aabaab, t_3_4_abbabb, t_4_4_aaaaaaaa, t_4_4_aaabaaab, t_4_4_aabbaabb, t_4_4_abbbabbb):
    l2_aaaa = l_2_1_aaaa
    l2_abab = l_2_1_abab
    l2_bbbb = l_2_1_aaaa
    t1_aa = t_1_4_aa
    t1_bb = t_1_4_aa
    t2_aaaa = t_2_4_aaaa
    t2_abab = t_2_4_abab
    t2_bbbb = t_2_4_aaaa
    t3_aaaaaa = t_3_4_aaaaaa
    t3_aabaab = t_3_4_aabaab
    t3_abbabb = t_3_4_abbabb
    t3_bbbbbb = t_3_4_aaaaaa
    t4_aaaaaaaa = t_4_4_aaaaaaaa
    t4_aaabaaab = t_4_4_aaabaaab
    t4_aabbaabb = t_4_4_aabbaabb
    t4_abbbabbb = t_4_4_abbbabbb
    t4_bbbbbbbb = t_4_4_aaaaaaaa
    result = np.zeros((no, nv))
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


def m5_ov_41_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_4_aa, l_2_4_aaaa, l_2_4_abab, l_3_4_aaaaaa, l_3_4_aabaab, l_3_4_abbabb, l_4_4_aaaaaaaa, l_4_4_aaabaaab, l_4_4_aabbaabb, l_4_4_abbbabbb, t_2_1_aaaa, t_2_1_abab):
    l1_aa = l_1_4_aa
    l1_bb = l_1_4_aa
    l2_aaaa = l_2_4_aaaa
    l2_abab = l_2_4_abab
    l2_bbbb = l_2_4_aaaa
    l3_aaaaaa = l_3_4_aaaaaa
    l3_aabaab = l_3_4_aabaab
    l3_abbabb = l_3_4_abbabb
    l3_bbbbbb = l_3_4_aaaaaa
    l4_aaaaaaaa = l_4_4_aaaaaaaa
    l4_aaabaaab = l_4_4_aaabaaab
    l4_aabbaabb = l_4_4_aabbaabb
    l4_abbbabbb = l_4_4_abbbabbb
    l4_bbbbbbbb = l_4_4_aaaaaaaa
    t2_aaaa = t_2_1_aaaa
    t2_abab = t_2_1_abab
    t2_bbbb = t_2_1_aaaa
    result = np.zeros((no, nv))
    _tmp0 = einsum('ia,eaim->me', l1_aa, t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    _tmp1 = einsum('ia,eami->me', l1_bb, t2_abab, optimize=True)
    result += 1 * _tmp1
    return result


def m5_ov_23_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_2_aa, l_2_2_aaaa, l_2_2_abab, l_3_2_aaaaaa, l_3_2_aabaab, l_3_2_abbabb, l_4_2_aaaaaaaa, l_4_2_aaabaaab, l_4_2_aabbaabb, l_4_2_abbbabbb, t_1_3_aa, t_2_3_aaaa, t_2_3_abab, t_3_3_aaaaaa, t_3_3_aabaab, t_3_3_abbabb, t_4_3_aaaaaaaa, t_4_3_aaabaaab, t_4_3_aabbaabb, t_4_3_abbbabbb):
    l1_aa = l_1_2_aa
    l1_bb = l_1_2_aa
    l2_aaaa = l_2_2_aaaa
    l2_abab = l_2_2_abab
    l2_bbbb = l_2_2_aaaa
    l3_aaaaaa = l_3_2_aaaaaa
    l3_aabaab = l_3_2_aabaab
    l3_abbabb = l_3_2_abbabb
    l3_bbbbbb = l_3_2_aaaaaa
    l4_aaaaaaaa = l_4_2_aaaaaaaa
    l4_aaabaaab = l_4_2_aaabaaab
    l4_aabbaabb = l_4_2_aabbaabb
    l4_abbbabbb = l_4_2_abbbabbb
    l4_bbbbbbbb = l_4_2_aaaaaaaa
    t1_aa = t_1_3_aa
    t1_bb = t_1_3_aa
    t2_aaaa = t_2_3_aaaa
    t2_abab = t_2_3_abab
    t2_bbbb = t_2_3_aaaa
    t3_aaaaaa = t_3_3_aaaaaa
    t3_aabaab = t_3_3_aabaab
    t3_abbabb = t_3_3_abbabb
    t3_bbbbbb = t_3_3_aaaaaa
    t4_aaaaaaaa = t_4_3_aaaaaaaa
    t4_aaabaaab = t_4_3_aaabaaab
    t4_aabbaabb = t_4_3_aabbaabb
    t4_abbbabbb = t_4_3_abbbabbb
    t4_bbbbbbbb = t_4_3_aaaaaaaa
    result = np.zeros((no, nv))
    _tmp0 = einsum('ia,eaim->me', l1_aa, t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    _tmp1 = einsum('ia,eami->me', l1_bb, t2_abab, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('ijba,ebaijm->me', l2_aaaa, t3_aaaaaa, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('ijba,ebaimj->me', l2_abab, t3_aabaab, optimize=True)
    result -= 0.25 * _tmp3
    result -= 0.25 * _tmp3
    _tmp4 = einsum('jiba,ebamji->me', l2_abab, t3_aabaab, optimize=True)
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    _tmp5 = einsum('ijba,ebamji->me', l2_bbbb, t3_abbabb, optimize=True)
    result -= 0.25 * _tmp5
    _tmp6 = einsum('ijkcba,ecbaijkm->me', l3_aaaaaa, t4_aaaaaaaa, optimize=True)
    result -= 0.0277777777777778 * _tmp6
    _tmp7 = einsum('ijkcba,ecbaijmk->me', l3_aabaab, t4_aaabaaab, optimize=True)
    result += 0.0277777777777778 * _tmp7
    result += 0.0277777777777778 * _tmp7
    result += 0.0277777777777778 * _tmp7
    _tmp8 = einsum('ikjcba,ecbaimkj->me', l3_aabaab, t4_aaabaaab, optimize=True)
    result -= 0.0277777777777778 * _tmp8
    result -= 0.0277777777777778 * _tmp8
    result -= 0.0277777777777778 * _tmp8
    _tmp9 = einsum('ijkcba,ecbaimkj->me', l3_abbabb, t4_aabbaabb, optimize=True)
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    _tmp10 = einsum('kjicba,ecbamjki->me', l3_aabaab, t4_aaabaaab, optimize=True)
    result -= 0.0277777777777778 * _tmp10
    result -= 0.0277777777777778 * _tmp10
    result -= 0.0277777777777778 * _tmp10
    _tmp11 = einsum('jikcba,ecbamjki->me', l3_abbabb, t4_aabbaabb, optimize=True)
    result -= 0.0277777777777778 * _tmp11
    result -= 0.0277777777777778 * _tmp11
    result -= 0.0277777777777778 * _tmp11
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    _tmp12 = einsum('ijkcba,ecbamjki->me', l3_bbbbbb, t4_abbbabbb, optimize=True)
    result += 0.0277777777777778 * _tmp12
    return result


def m5_ov_32_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_3_aa, l_2_3_aaaa, l_2_3_abab, l_3_3_aaaaaa, l_3_3_aabaab, l_3_3_abbabb, l_4_3_aaaaaaaa, l_4_3_aaabaaab, l_4_3_aabbaabb, l_4_3_abbbabbb, t_1_2_aa, t_2_2_aaaa, t_2_2_abab, t_3_2_aaaaaa, t_3_2_aabaab, t_3_2_abbabb, t_4_2_aaaaaaaa, t_4_2_aaabaaab, t_4_2_aabbaabb, t_4_2_abbbabbb):
    l1_aa = l_1_3_aa
    l1_bb = l_1_3_aa
    l2_aaaa = l_2_3_aaaa
    l2_abab = l_2_3_abab
    l2_bbbb = l_2_3_aaaa
    l3_aaaaaa = l_3_3_aaaaaa
    l3_aabaab = l_3_3_aabaab
    l3_abbabb = l_3_3_abbabb
    l3_bbbbbb = l_3_3_aaaaaa
    l4_aaaaaaaa = l_4_3_aaaaaaaa
    l4_aaabaaab = l_4_3_aaabaaab
    l4_aabbaabb = l_4_3_aabbaabb
    l4_abbbabbb = l_4_3_abbbabbb
    l4_bbbbbbbb = l_4_3_aaaaaaaa
    t1_aa = t_1_2_aa
    t1_bb = t_1_2_aa
    t2_aaaa = t_2_2_aaaa
    t2_abab = t_2_2_abab
    t2_bbbb = t_2_2_aaaa
    t3_aaaaaa = t_3_2_aaaaaa
    t3_aabaab = t_3_2_aabaab
    t3_abbabb = t_3_2_abbabb
    t3_bbbbbb = t_3_2_aaaaaa
    t4_aaaaaaaa = t_4_2_aaaaaaaa
    t4_aaabaaab = t_4_2_aaabaaab
    t4_aabbaabb = t_4_2_aabbaabb
    t4_abbbabbb = t_4_2_abbbabbb
    t4_bbbbbbbb = t_4_2_aaaaaaaa
    result = np.zeros((no, nv))
    _tmp0 = einsum('ia,eaim->me', l1_aa, t2_aaaa, optimize=True)
    result -= 1 * _tmp0
    _tmp1 = einsum('ia,eami->me', l1_bb, t2_abab, optimize=True)
    result += 1 * _tmp1
    _tmp2 = einsum('ijba,ebaijm->me', l2_aaaa, t3_aaaaaa, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('ijba,ebaimj->me', l2_abab, t3_aabaab, optimize=True)
    result -= 0.25 * _tmp3
    result -= 0.25 * _tmp3
    _tmp4 = einsum('jiba,ebamji->me', l2_abab, t3_aabaab, optimize=True)
    result += 0.25 * _tmp4
    result += 0.25 * _tmp4
    _tmp5 = einsum('ijba,ebamji->me', l2_bbbb, t3_abbabb, optimize=True)
    result -= 0.25 * _tmp5
    _tmp6 = einsum('ijkcba,ecbaijkm->me', l3_aaaaaa, t4_aaaaaaaa, optimize=True)
    result -= 0.0277777777777778 * _tmp6
    _tmp7 = einsum('ijkcba,ecbaijmk->me', l3_aabaab, t4_aaabaaab, optimize=True)
    result += 0.0277777777777778 * _tmp7
    result += 0.0277777777777778 * _tmp7
    result += 0.0277777777777778 * _tmp7
    _tmp8 = einsum('ikjcba,ecbaimkj->me', l3_aabaab, t4_aaabaaab, optimize=True)
    result -= 0.0277777777777778 * _tmp8
    result -= 0.0277777777777778 * _tmp8
    result -= 0.0277777777777778 * _tmp8
    _tmp9 = einsum('ijkcba,ecbaimkj->me', l3_abbabb, t4_aabbaabb, optimize=True)
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    _tmp10 = einsum('kjicba,ecbamjki->me', l3_aabaab, t4_aaabaaab, optimize=True)
    result -= 0.0277777777777778 * _tmp10
    result -= 0.0277777777777778 * _tmp10
    result -= 0.0277777777777778 * _tmp10
    _tmp11 = einsum('jikcba,ecbamjki->me', l3_abbabb, t4_aabbaabb, optimize=True)
    result -= 0.0277777777777778 * _tmp11
    result -= 0.0277777777777778 * _tmp11
    result -= 0.0277777777777778 * _tmp11
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    result += 0.0277777777777778 * _tmp9
    _tmp12 = einsum('ijkcba,ecbamjki->me', l3_bbbbbb, t4_abbbabbb, optimize=True)
    result += 0.0277777777777778 * _tmp12
    return result


def m5_ov_50_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, l_1_5_aa):
    l1_aa = l_1_5_aa
    l1_bb = l_1_5_aa
    result = np.zeros((no, nv))
    return result


def m5_ov_05_restricted(g_aaaa, g_abab, g_bbbb, d_aa, o, v, nv, no, t_1_5_aa):
    t1_aa = t_1_5_aa
    t1_bb = t_1_5_aa
    result = np.zeros((no, nv))
    _tmp0 = einsum('em->me', t1_aa)
    result += 1 * _tmp0
    return result

