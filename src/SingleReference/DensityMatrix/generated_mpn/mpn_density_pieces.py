# GENERATED CODE -- MPn amplitude-numerator, overlap, and density cross-term
# pieces (spin-orbital), order-generic recursion. Do not edit by hand.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum


def t2_1_numerator(g, kd, o, v, nv, no):
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('abij->abij', g[v, v, o, o])
    result += 1 * _tmp0
    return result


def t1_2_numerator(g, kd, o, v, nv, no, t2_1):
    t2 = t2_1
    result = np.zeros((nv, no))
    _tmp0 = einsum('kjbi,bakj->ai', g[o, o, v, o], t2, optimize=True)
    result -= 0.5 * _tmp0
    _tmp1 = einsum('jabc,bcij->ai', g[o, v, v, v], t2, optimize=True)
    result -= 0.5 * _tmp1
    return result


def t2_2_numerator(g, kd, o, v, nv, no, t2_1):
    t2 = t2_1
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('lkij,ablk->abij', g[o, o, o, o], t2, optimize=True)
    result += 0.5 * _tmp0
    _tmp1 = einsum('kacj,cbik->abij', g[o, v, v, o], t2, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(1, 0, 2, 3)
    result -= 1 * _tmp1.transpose(0, 1, 3, 2)
    result += 1 * _tmp1.transpose(1, 0, 3, 2)
    _tmp2 = einsum('abcd,cdij->abij', g[v, v, v, v], t2, optimize=True)
    result += 0.5 * _tmp2
    return result


def t3_2_numerator(g, kd, o, v, nv, no, t2_1):
    t2 = t2_1
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('lajk,bcil->abcijk', g[o, v, o, o], t2, optimize=True)
    result -= 1 * _tmp0
    result += 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result -= 1 * _tmp0.transpose(1, 0, 2, 4, 3, 5)
    _tmp1 = einsum('laij,bckl->abcijk', g[o, v, o, o], t2, optimize=True)
    result -= 1 * _tmp1
    result += 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    _tmp2 = einsum('lcjk,abil->abcijk', g[o, v, o, o], t2, optimize=True)
    result -= 1 * _tmp2
    result += 1 * _tmp2.transpose(0, 1, 2, 4, 3, 5)
    _tmp3 = einsum('lcij,abkl->abcijk', g[o, v, o, o], t2, optimize=True)
    result -= 1 * _tmp3
    _tmp4 = einsum('abdk,dcij->abcijk', g[v, v, v, o], t2, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp4.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp4.transpose(0, 2, 1, 3, 5, 4)
    _tmp5 = einsum('abdi,dcjk->abcijk', g[v, v, v, o], t2, optimize=True)
    result -= 1 * _tmp5
    result += 1 * _tmp5.transpose(0, 2, 1, 3, 4, 5)
    _tmp6 = einsum('bcdk,daij->abcijk', g[v, v, v, o], t2, optimize=True)
    result -= 1 * _tmp6
    result += 1 * _tmp6.transpose(0, 1, 2, 3, 5, 4)
    _tmp7 = einsum('bcdi,dajk->abcijk', g[v, v, v, o], t2, optimize=True)
    result -= 1 * _tmp7
    return result


def t4_2_numerator(g, kd, o, v, nv, no, t2_1):
    t2 = t2_1
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('abkl,cdij->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp0.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp0.transpose(0, 2, 1, 3, 4, 6, 5, 7)
    _tmp1 = einsum('abil,cdjk->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp1.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 7, 6)
    _tmp2 = einsum('abjk,cdil->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp2.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp2.transpose(0, 2, 1, 3, 6, 5, 4, 7)
    _tmp3 = einsum('adkl,bcij->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp3.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp3.transpose(1, 0, 2, 3, 4, 6, 5, 7)
    _tmp4 = einsum('adil,bcjk->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp4
    result -= 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp4.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    _tmp5 = einsum('adjk,bcil->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp5
    result -= 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp5.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp5.transpose(1, 0, 2, 3, 6, 5, 4, 7)
    _tmp6 = einsum('bckl,adij->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp6
    result -= 1 * _tmp6.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp6.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp6.transpose(0, 3, 2, 1, 4, 6, 5, 7)
    _tmp7 = einsum('bcil,adjk->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp7
    result -= 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp7.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 7, 6)
    _tmp8 = einsum('bcjk,adil->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp8
    result -= 1 * _tmp8.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp8.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp8.transpose(0, 3, 2, 1, 6, 5, 4, 7)
    return result


def t1_3_numerator(g, kd, o, v, nv, no, t1_2, t2_2, t3_2):
    t1 = t1_2
    t2 = t2_2
    t3 = t3_2
    result = np.zeros((nv, no))
    _tmp0 = einsum('jabi,bj->ai', g[o, v, v, o], t1, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('kjbi,bakj->ai', g[o, o, v, o], t2, optimize=True)
    result -= 0.5 * _tmp1
    _tmp2 = einsum('jabc,bcij->ai', g[o, v, v, v], t2, optimize=True)
    result -= 0.5 * _tmp2
    _tmp3 = einsum('kjbc,bcaikj->ai', g[o, o, v, v], t3, optimize=True)
    result += 0.25 * _tmp3
    return result


def t2_3_numerator(g, kd, o, v, nv, no, t1_2, t2_2, t3_2, t4_2):
    t1 = t1_2
    t2 = t2_2
    t3 = t3_2
    t4 = t4_2
    result = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('kaij,bk->abij', g[o, v, o, o], t1, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(1, 0, 2, 3)
    _tmp1 = einsum('abcj,ci->abij', g[v, v, v, o], t1, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 1, 3, 2)
    _tmp2 = einsum('lkij,ablk->abij', g[o, o, o, o], t2, optimize=True)
    result += 0.5 * _tmp2
    _tmp3 = einsum('kacj,cbik->abij', g[o, v, v, o], t2, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3)
    result -= 1 * _tmp3.transpose(0, 1, 3, 2)
    result += 1 * _tmp3.transpose(1, 0, 3, 2)
    _tmp4 = einsum('abcd,cdij->abij', g[v, v, v, v], t2, optimize=True)
    result += 0.5 * _tmp4
    _tmp5 = einsum('lkcj,cabilk->abij', g[o, o, v, o], t3, optimize=True)
    result += 0.5 * _tmp5
    result -= 0.5 * _tmp5.transpose(0, 1, 3, 2)
    _tmp6 = einsum('kacd,cdbijk->abij', g[o, v, v, v], t3, optimize=True)
    result += 0.5 * _tmp6
    result -= 0.5 * _tmp6.transpose(1, 0, 2, 3)
    _tmp7 = einsum('lkcd,cdabijlk->abij', g[o, o, v, v], t4, optimize=True)
    result += 0.25 * _tmp7
    return result


def t3_3_numerator(g, kd, o, v, nv, no, t1_2, t2_2, t3_2, t4_2):
    t1 = t1_2
    t2 = t2_2
    t3 = t3_2
    t4 = t4_2
    result = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('abjk,ci->abcijk', g[v, v, o, o], t1, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5)
    result -= 1 * _tmp0.transpose(0, 1, 2, 4, 3, 5)
    result += 1 * _tmp0.transpose(0, 2, 1, 4, 3, 5)
    _tmp1 = einsum('abij,ck->abcijk', g[v, v, o, o], t1, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5)
    _tmp2 = einsum('bcjk,ai->abcijk', g[v, v, o, o], t1, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(0, 1, 2, 4, 3, 5)
    _tmp3 = einsum('bcij,ak->abcijk', g[v, v, o, o], t1, optimize=True)
    result += 1 * _tmp3
    _tmp4 = einsum('lajk,bcil->abcijk', g[o, v, o, o], t2, optimize=True)
    result -= 1 * _tmp4
    result += 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5)
    result += 1 * _tmp4.transpose(0, 1, 2, 4, 3, 5)
    result -= 1 * _tmp4.transpose(1, 0, 2, 4, 3, 5)
    _tmp5 = einsum('laij,bckl->abcijk', g[o, v, o, o], t2, optimize=True)
    result -= 1 * _tmp5
    result += 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5)
    _tmp6 = einsum('lcjk,abil->abcijk', g[o, v, o, o], t2, optimize=True)
    result -= 1 * _tmp6
    result += 1 * _tmp6.transpose(0, 1, 2, 4, 3, 5)
    _tmp7 = einsum('lcij,abkl->abcijk', g[o, v, o, o], t2, optimize=True)
    result -= 1 * _tmp7
    _tmp8 = einsum('abdk,dcij->abcijk', g[v, v, v, o], t2, optimize=True)
    result -= 1 * _tmp8
    result += 1 * _tmp8.transpose(0, 2, 1, 3, 4, 5)
    result += 1 * _tmp8.transpose(0, 1, 2, 3, 5, 4)
    result -= 1 * _tmp8.transpose(0, 2, 1, 3, 5, 4)
    _tmp9 = einsum('abdi,dcjk->abcijk', g[v, v, v, o], t2, optimize=True)
    result -= 1 * _tmp9
    result += 1 * _tmp9.transpose(0, 2, 1, 3, 4, 5)
    _tmp10 = einsum('bcdk,daij->abcijk', g[v, v, v, o], t2, optimize=True)
    result -= 1 * _tmp10
    result += 1 * _tmp10.transpose(0, 1, 2, 3, 5, 4)
    _tmp11 = einsum('bcdi,dajk->abcijk', g[v, v, v, o], t2, optimize=True)
    result -= 1 * _tmp11
    _tmp12 = einsum('mljk,abciml->abcijk', g[o, o, o, o], t3, optimize=True)
    result += 0.5 * _tmp12
    result -= 0.5 * _tmp12.transpose(0, 1, 2, 4, 3, 5)
    _tmp13 = einsum('mlij,abckml->abcijk', g[o, o, o, o], t3, optimize=True)
    result += 0.5 * _tmp13
    _tmp14 = einsum('ladk,dbcijl->abcijk', g[o, v, v, o], t3, optimize=True)
    result += 1 * _tmp14
    result -= 1 * _tmp14.transpose(1, 0, 2, 3, 4, 5)
    result -= 1 * _tmp14.transpose(0, 1, 2, 3, 5, 4)
    result += 1 * _tmp14.transpose(1, 0, 2, 3, 5, 4)
    _tmp15 = einsum('ladi,dbcjkl->abcijk', g[o, v, v, o], t3, optimize=True)
    result += 1 * _tmp15
    result -= 1 * _tmp15.transpose(1, 0, 2, 3, 4, 5)
    _tmp16 = einsum('lcdk,dabijl->abcijk', g[o, v, v, o], t3, optimize=True)
    result += 1 * _tmp16
    result -= 1 * _tmp16.transpose(0, 1, 2, 3, 5, 4)
    _tmp17 = einsum('lcdi,dabjkl->abcijk', g[o, v, v, o], t3, optimize=True)
    result += 1 * _tmp17
    _tmp18 = einsum('abde,decijk->abcijk', g[v, v, v, v], t3, optimize=True)
    result += 0.5 * _tmp18
    result -= 0.5 * _tmp18.transpose(0, 2, 1, 3, 4, 5)
    _tmp19 = einsum('bcde,deaijk->abcijk', g[v, v, v, v], t3, optimize=True)
    result += 0.5 * _tmp19
    _tmp20 = einsum('mldk,dabcijml->abcijk', g[o, o, v, o], t4, optimize=True)
    result -= 0.5 * _tmp20
    result += 0.5 * _tmp20.transpose(0, 1, 2, 3, 5, 4)
    _tmp21 = einsum('mldi,dabcjkml->abcijk', g[o, o, v, o], t4, optimize=True)
    result -= 0.5 * _tmp21
    _tmp22 = einsum('lade,debcijkl->abcijk', g[o, v, v, v], t4, optimize=True)
    result -= 0.5 * _tmp22
    result += 0.5 * _tmp22.transpose(1, 0, 2, 3, 4, 5)
    _tmp23 = einsum('lcde,deabijkl->abcijk', g[o, v, v, v], t4, optimize=True)
    result -= 0.5 * _tmp23
    return result


def t4_3_numerator(g, kd, o, v, nv, no, t2_2, t3_2, t4_2):
    t2 = t2_2
    t3 = t3_2
    t4 = t4_2
    result = np.zeros((nv, nv, nv, nv, no, no, no, no))
    _tmp0 = einsum('abkl,cdij->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp0
    result -= 1 * _tmp0.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp0.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp0.transpose(0, 2, 1, 3, 4, 6, 5, 7)
    _tmp1 = einsum('abil,cdjk->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp1
    result -= 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp1.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp1.transpose(0, 2, 1, 3, 4, 5, 7, 6)
    _tmp2 = einsum('abjk,cdil->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp2
    result -= 1 * _tmp2.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp2.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp2.transpose(0, 2, 1, 3, 6, 5, 4, 7)
    _tmp3 = einsum('adkl,bcij->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp3
    result -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp3.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp3.transpose(1, 0, 2, 3, 4, 6, 5, 7)
    _tmp4 = einsum('adil,bcjk->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp4
    result -= 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp4.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp4.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    _tmp5 = einsum('adjk,bcil->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp5
    result -= 1 * _tmp5.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp5.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp5.transpose(1, 0, 2, 3, 6, 5, 4, 7)
    _tmp6 = einsum('bckl,adij->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp6
    result -= 1 * _tmp6.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp6.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp6.transpose(0, 3, 2, 1, 4, 6, 5, 7)
    _tmp7 = einsum('bcil,adjk->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp7
    result -= 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp7.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp7.transpose(0, 3, 2, 1, 4, 5, 7, 6)
    _tmp8 = einsum('bcjk,adil->abcdijkl', g[v, v, o, o], t2, optimize=True)
    result += 1 * _tmp8
    result -= 1 * _tmp8.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp8.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp8.transpose(0, 3, 2, 1, 6, 5, 4, 7)
    _tmp9 = einsum('makl,bcdijm->abcdijkl', g[o, v, o, o], t3, optimize=True)
    result += 1 * _tmp9
    result -= 1 * _tmp9.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp9.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp9.transpose(1, 0, 2, 3, 4, 6, 5, 7)
    _tmp10 = einsum('mail,bcdjkm->abcdijkl', g[o, v, o, o], t3, optimize=True)
    result += 1 * _tmp10
    result -= 1 * _tmp10.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp10.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp10.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    _tmp11 = einsum('majk,bcdilm->abcdijkl', g[o, v, o, o], t3, optimize=True)
    result += 1 * _tmp11
    result -= 1 * _tmp11.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp11.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp11.transpose(1, 0, 2, 3, 6, 5, 4, 7)
    _tmp12 = einsum('mckl,abdijm->abcdijkl', g[o, v, o, o], t3, optimize=True)
    result += 1 * _tmp12
    result -= 1 * _tmp12.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp12.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    result += 1 * _tmp12.transpose(0, 1, 3, 2, 4, 6, 5, 7)
    _tmp13 = einsum('mcil,abdjkm->abcdijkl', g[o, v, o, o], t3, optimize=True)
    result += 1 * _tmp13
    result -= 1 * _tmp13.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp13.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp13.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    _tmp14 = einsum('mcjk,abdilm->abcdijkl', g[o, v, o, o], t3, optimize=True)
    result += 1 * _tmp14
    result -= 1 * _tmp14.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp14.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    result += 1 * _tmp14.transpose(0, 1, 3, 2, 6, 5, 4, 7)
    _tmp15 = einsum('abel,ecdijk->abcdijkl', g[v, v, v, o], t3, optimize=True)
    result += 1 * _tmp15
    result -= 1 * _tmp15.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp15.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp15.transpose(0, 2, 1, 3, 4, 5, 7, 6)
    _tmp16 = einsum('abej,ecdikl->abcdijkl', g[v, v, v, o], t3, optimize=True)
    result += 1 * _tmp16
    result -= 1 * _tmp16.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    result -= 1 * _tmp16.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp16.transpose(0, 2, 1, 3, 5, 4, 6, 7)
    _tmp17 = einsum('adel,ebcijk->abcdijkl', g[v, v, v, o], t3, optimize=True)
    result += 1 * _tmp17
    result -= 1 * _tmp17.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp17.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp17.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    _tmp18 = einsum('adej,ebcikl->abcdijkl', g[v, v, v, o], t3, optimize=True)
    result += 1 * _tmp18
    result -= 1 * _tmp18.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp18.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp18.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    _tmp19 = einsum('bcel,eadijk->abcdijkl', g[v, v, v, o], t3, optimize=True)
    result += 1 * _tmp19
    result -= 1 * _tmp19.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp19.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp19.transpose(0, 3, 2, 1, 4, 5, 7, 6)
    _tmp20 = einsum('bcej,eadikl->abcdijkl', g[v, v, v, o], t3, optimize=True)
    result += 1 * _tmp20
    result -= 1 * _tmp20.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    result -= 1 * _tmp20.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp20.transpose(0, 3, 2, 1, 5, 4, 6, 7)
    _tmp21 = einsum('nmkl,abcdijnm->abcdijkl', g[o, o, o, o], t4, optimize=True)
    result += 0.5 * _tmp21
    result -= 0.5 * _tmp21.transpose(0, 1, 2, 3, 4, 6, 5, 7)
    _tmp22 = einsum('nmil,abcdjknm->abcdijkl', g[o, o, o, o], t4, optimize=True)
    result += 0.5 * _tmp22
    result -= 0.5 * _tmp22.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    _tmp23 = einsum('nmjk,abcdilnm->abcdijkl', g[o, o, o, o], t4, optimize=True)
    result += 0.5 * _tmp23
    result -= 0.5 * _tmp23.transpose(0, 1, 2, 3, 6, 5, 4, 7)
    _tmp24 = einsum('mael,ebcdijkm->abcdijkl', g[o, v, v, o], t4, optimize=True)
    result += 1 * _tmp24
    result -= 1 * _tmp24.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp24.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp24.transpose(1, 0, 2, 3, 4, 5, 7, 6)
    _tmp25 = einsum('maej,ebcdiklm->abcdijkl', g[o, v, v, o], t4, optimize=True)
    result += 1 * _tmp25
    result -= 1 * _tmp25.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    result -= 1 * _tmp25.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp25.transpose(1, 0, 2, 3, 5, 4, 6, 7)
    _tmp26 = einsum('mcel,eabdijkm->abcdijkl', g[o, v, v, o], t4, optimize=True)
    result += 1 * _tmp26
    result -= 1 * _tmp26.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp26.transpose(0, 1, 2, 3, 4, 5, 7, 6)
    result += 1 * _tmp26.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    _tmp27 = einsum('mcej,eabdiklm->abcdijkl', g[o, v, v, o], t4, optimize=True)
    result += 1 * _tmp27
    result -= 1 * _tmp27.transpose(0, 1, 3, 2, 4, 5, 6, 7)
    result -= 1 * _tmp27.transpose(0, 1, 2, 3, 5, 4, 6, 7)
    result += 1 * _tmp27.transpose(0, 1, 3, 2, 5, 4, 6, 7)
    _tmp28 = einsum('abef,efcdijkl->abcdijkl', g[v, v, v, v], t4, optimize=True)
    result += 0.5 * _tmp28
    result -= 0.5 * _tmp28.transpose(0, 2, 1, 3, 4, 5, 6, 7)
    _tmp29 = einsum('adef,efbcijkl->abcdijkl', g[v, v, v, v], t4, optimize=True)
    result += 0.5 * _tmp29
    result -= 0.5 * _tmp29.transpose(1, 0, 2, 3, 4, 5, 6, 7)
    _tmp30 = einsum('bcef,efadijkl->abcdijkl', g[v, v, v, v], t4, optimize=True)
    result += 0.5 * _tmp30
    result -= 0.5 * _tmp30.transpose(0, 3, 2, 1, 4, 5, 6, 7)
    return result


def t1_4_numerator(g, kd, o, v, nv, no, t1_3, t2_3, t3_3):
    t1 = t1_3
    t2 = t2_3
    t3 = t3_3
    result = np.zeros((nv, no))
    _tmp0 = einsum('jabi,bj->ai', g[o, v, v, o], t1, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('kjbi,bakj->ai', g[o, o, v, o], t2, optimize=True)
    result -= 0.5 * _tmp1
    _tmp2 = einsum('jabc,bcij->ai', g[o, v, v, v], t2, optimize=True)
    result -= 0.5 * _tmp2
    _tmp3 = einsum('kjbc,bcaikj->ai', g[o, o, v, v], t3, optimize=True)
    result += 0.25 * _tmp3
    return result


def overlap1(l_amp, t_amp):
    l1 = l_amp
    t1 = t_amp
    result = np.zeros(())
    _tmp0 = einsum('ai,ia->', t1, l1, optimize=True)
    result += 1 * _tmp0
    return result


def overlap2(l_amp, t_amp):
    l2 = l_amp
    t2 = t_amp
    result = np.zeros(())
    _tmp0 = einsum('baij,ijba->', t2, l2, optimize=True)
    result += 0.25 * _tmp0
    return result


def overlap3(l_amp, t_amp):
    l3 = l_amp
    t3 = t_amp
    result = np.zeros(())
    _tmp0 = einsum('cbaijk,ijkcba->', t3, l3, optimize=True)
    result += 0.0277777777777778 * _tmp0
    return result


def overlap4(l_amp, t_amp):
    l4 = l_amp
    t4 = t_amp
    result = np.zeros(())
    _tmp0 = einsum('dcbaijkl,ijkldcba->', t4, l4, optimize=True)
    result += 0.00173611111111111 * _tmp0
    return result


def m2_oo_02(g, kd, o, v, nv, no, t1):
    result = np.zeros((no, no))
    return result


def m2_oo_11(g, kd, o, v, nv, no, l2, t2):
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,baij,ijba->mn', kd[o, o], t2, l2, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('baim,inba->mn', t2, l2, optimize=True)
    result -= 0.5 * _tmp1
    return result


def m2_oo_20(g, kd, o, v, nv, no, l1):
    result = np.zeros((no, no))
    return result


def m2_vv_02(g, kd, o, v, nv, no, t1):
    result = np.zeros((nv, nv))
    return result


def m2_vv_11(g, kd, o, v, nv, no, l2, t2):
    result = np.zeros((nv, nv))
    _tmp0 = einsum('faij,ijea->ef', t2, l2, optimize=True)
    result += 0.5 * _tmp0
    return result


def m2_vv_20(g, kd, o, v, nv, no, l1):
    result = np.zeros((nv, nv))
    return result


def m2_ov_02(g, kd, o, v, nv, no, t1):
    result = np.zeros((no, nv))
    _tmp0 = einsum('em->me', t1)
    result += 1 * _tmp0
    return result


def m2_ov_11(g, kd, o, v, nv, no, l2, t2):
    result = np.zeros((no, nv))
    return result


def m2_ov_20(g, kd, o, v, nv, no, l1):
    result = np.zeros((no, nv))
    return result


def m3_oo_03(g, kd, o, v, nv, no, t1):
    result = np.zeros((no, no))
    return result


def m3_oo_12(g, kd, o, v, nv, no, l2, t1, t2, t3):
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,baij,ijba->mn', kd[o, o], t2, l2, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('baim,inba->mn', t2, l2, optimize=True)
    result -= 0.5 * _tmp1
    return result


def m3_oo_21(g, kd, o, v, nv, no, l1, l2, l3, t2):
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,baij,ijba->mn', kd[o, o], t2, l2, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('baim,inba->mn', t2, l2, optimize=True)
    result -= 0.5 * _tmp1
    return result


def m3_oo_30(g, kd, o, v, nv, no, l1):
    result = np.zeros((no, no))
    return result


def m3_vv_03(g, kd, o, v, nv, no, t1):
    result = np.zeros((nv, nv))
    return result


def m3_vv_12(g, kd, o, v, nv, no, l2, t1, t2, t3):
    result = np.zeros((nv, nv))
    _tmp0 = einsum('faij,ijea->ef', t2, l2, optimize=True)
    result += 0.5 * _tmp0
    return result


def m3_vv_21(g, kd, o, v, nv, no, l1, l2, l3, t2):
    result = np.zeros((nv, nv))
    _tmp0 = einsum('faij,ijea->ef', t2, l2, optimize=True)
    result += 0.5 * _tmp0
    return result


def m3_vv_30(g, kd, o, v, nv, no, l1):
    result = np.zeros((nv, nv))
    return result


def m3_ov_03(g, kd, o, v, nv, no, t1):
    result = np.zeros((no, nv))
    _tmp0 = einsum('em->me', t1)
    result += 1 * _tmp0
    return result


def m3_ov_12(g, kd, o, v, nv, no, l2, t1, t2, t3):
    result = np.zeros((no, nv))
    _tmp0 = einsum('ebaijm,ijba->me', t3, l2, optimize=True)
    result += 0.25 * _tmp0
    return result


def m3_ov_21(g, kd, o, v, nv, no, l1, l2, l3, t2):
    result = np.zeros((no, nv))
    _tmp0 = einsum('eaim,ia->me', t2, l1, optimize=True)
    result -= 1 * _tmp0
    return result


def m3_ov_30(g, kd, o, v, nv, no, l1):
    result = np.zeros((no, nv))
    return result


def m4_oo_04(g, kd, o, v, nv, no, t1):
    result = np.zeros((no, no))
    return result


def m4_oo_13(g, kd, o, v, nv, no, l2, t1, t2, t3):
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,baij,ijba->mn', kd[o, o], t2, l2, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('baim,inba->mn', t2, l2, optimize=True)
    result -= 0.5 * _tmp1
    return result


def m4_oo_22(g, kd, o, v, nv, no, l1, l2, l3, l4, t1, t2, t3, t4):
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,ai,ia->mn', kd[o, o], t1, l1, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('am,na->mn', t1, l1, optimize=True)
    result -= 1 * _tmp1
    _tmp2 = einsum('mn,baij,ijba->mn', kd[o, o], t2, l2, optimize=True)
    result += 0.25 * _tmp2
    _tmp3 = einsum('baim,inba->mn', t2, l2, optimize=True)
    result -= 0.5 * _tmp3
    _tmp4 = einsum('mn,cbaijk,ijkcba->mn', kd[o, o], t3, l3, optimize=True)
    result += 0.0277777777777778 * _tmp4
    _tmp5 = einsum('cbaijm,ijncba->mn', t3, l3, optimize=True)
    result -= 0.0833333333333333 * _tmp5
    _tmp6 = einsum('mn,dcbaijkl,ijkldcba->mn', kd[o, o], t4, l4, optimize=True)
    result += 0.00173611111111111 * _tmp6
    _tmp7 = einsum('dcbaijkm,ijkndcba->mn', t4, l4, optimize=True)
    result -= 0.00694444444444444 * _tmp7
    return result


def m4_oo_31(g, kd, o, v, nv, no, l1, l2, l3, t2):
    result = np.zeros((no, no))
    _tmp0 = einsum('mn,baij,ijba->mn', kd[o, o], t2, l2, optimize=True)
    result += 0.25 * _tmp0
    _tmp1 = einsum('baim,inba->mn', t2, l2, optimize=True)
    result -= 0.5 * _tmp1
    return result


def m4_oo_40(g, kd, o, v, nv, no, l1):
    result = np.zeros((no, no))
    return result


def m4_vv_04(g, kd, o, v, nv, no, t1):
    result = np.zeros((nv, nv))
    return result


def m4_vv_13(g, kd, o, v, nv, no, l2, t1, t2, t3):
    result = np.zeros((nv, nv))
    _tmp0 = einsum('faij,ijea->ef', t2, l2, optimize=True)
    result += 0.5 * _tmp0
    return result


def m4_vv_22(g, kd, o, v, nv, no, l1, l2, l3, l4, t1, t2, t3, t4):
    result = np.zeros((nv, nv))
    _tmp0 = einsum('fi,ie->ef', t1, l1, optimize=True)
    result += 1 * _tmp0
    _tmp1 = einsum('faij,ijea->ef', t2, l2, optimize=True)
    result += 0.5 * _tmp1
    _tmp2 = einsum('fbaijk,ijkeba->ef', t3, l3, optimize=True)
    result += 0.0833333333333333 * _tmp2
    _tmp3 = einsum('fcbaijkl,ijklecba->ef', t4, l4, optimize=True)
    result += 0.00694444444444444 * _tmp3
    return result


def m4_vv_31(g, kd, o, v, nv, no, l1, l2, l3, t2):
    result = np.zeros((nv, nv))
    _tmp0 = einsum('faij,ijea->ef', t2, l2, optimize=True)
    result += 0.5 * _tmp0
    return result


def m4_vv_40(g, kd, o, v, nv, no, l1):
    result = np.zeros((nv, nv))
    return result


def m4_ov_04(g, kd, o, v, nv, no, t1):
    result = np.zeros((no, nv))
    _tmp0 = einsum('em->me', t1)
    result += 1 * _tmp0
    return result


def m4_ov_13(g, kd, o, v, nv, no, l2, t1, t2, t3):
    result = np.zeros((no, nv))
    _tmp0 = einsum('ebaijm,ijba->me', t3, l2, optimize=True)
    result += 0.25 * _tmp0
    return result


def m4_ov_22(g, kd, o, v, nv, no, l1, l2, l3, l4, t1, t2, t3, t4):
    result = np.zeros((no, nv))
    _tmp0 = einsum('eaim,ia->me', t2, l1, optimize=True)
    result -= 1 * _tmp0
    _tmp1 = einsum('ebaijm,ijba->me', t3, l2, optimize=True)
    result += 0.25 * _tmp1
    _tmp2 = einsum('ecbaijkm,ijkcba->me', t4, l3, optimize=True)
    result -= 0.0277777777777778 * _tmp2
    return result


def m4_ov_31(g, kd, o, v, nv, no, l1, l2, l3, t2):
    result = np.zeros((no, nv))
    _tmp0 = einsum('eaim,ia->me', t2, l1, optimize=True)
    result -= 1 * _tmp0
    return result


def m4_ov_40(g, kd, o, v, nv, no, l1):
    result = np.zeros((no, nv))
    return result

