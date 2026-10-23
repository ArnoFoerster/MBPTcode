# GENERATED CODE -- do not edit by hand.
# Produced by src/CodeGeneration/generate_restricted_ccsdt.py.
# Restricted (spin-blocked) CCSDT energy and T-residuals.
# See that module's docstring for the equations/conventions.
# allow numpy built with MKL to consume more threads for tensordot -- but only
# if the caller hasn't already pinned the thread count (same convention as
# amplitudes.py's generalized-pipeline counterpart)
import os
os.environ.setdefault("MKL_NUM_THREADS", "{}".format(os.cpu_count() - 1))

import numpy as np
from src.SingleReference.CC.cached_einsum import einsum


def cc_energy_restricted(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v):
    nv, no = t1_aa.shape
    energy = np.zeros(())
    _tmp0 = einsum('ii->', f_aa[o, o])
    energy += 1 * _tmp0
    _tmp1 = einsum('ii->', f_bb[o, o])
    energy += 1 * _tmp1
    _tmp2 = einsum('ia,ai->', f_aa[o, v], t1_aa, optimize=True)
    energy += 1 * _tmp2
    _tmp3 = einsum('ia,ai->', f_bb[o, v], t1_bb, optimize=True)
    energy += 1 * _tmp3
    _tmp4 = einsum('jiji->', g_aaaa[o, o, o, o])
    energy -= 0.5 * _tmp4
    _tmp5 = einsum('jiji->', g_abab[o, o, o, o])
    energy -= 0.5 * _tmp5
    energy -= 0.5 * _tmp5
    _tmp6 = einsum('jiji->', g_bbbb[o, o, o, o])
    energy -= 0.5 * _tmp6
    _tmp7 = einsum('jiab,abji->', g_aaaa[o, o, v, v], t2_aaaa, optimize=True)
    energy += 0.25 * _tmp7
    _tmp8 = einsum('jiab,abji->', g_abab[o, o, v, v], t2_abab, optimize=True)
    energy += 0.25 * _tmp8
    energy += 0.25 * _tmp8
    energy += 0.25 * _tmp8
    energy += 0.25 * _tmp8
    _tmp9 = einsum('jiab,abji->', g_bbbb[o, o, v, v], t2_bbbb, optimize=True)
    energy += 0.25 * _tmp9
    _tmp10 = einsum('jiab,ai,bj->', g_aaaa[o, o, v, v], t1_aa, t1_aa, optimize=True)
    energy -= 0.5 * _tmp10
    _tmp11 = einsum('ijab,ai,bj->', g_abab[o, o, v, v], t1_aa, t1_bb, optimize=True)
    energy += 0.5 * _tmp11
    energy += 0.5 * _tmp11
    _tmp12 = einsum('jiab,ai,bj->', g_bbbb[o, o, v, v], t1_bb, t1_bb, optimize=True)
    energy -= 0.5 * _tmp12
    return energy


def t1_aa_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v):
    nv, no = t1_aa.shape
    t1_res = np.zeros((nv, no))
    _tmp0 = einsum('ai->ai', f_aa[v, o])
    t1_res += 1 * _tmp0
    _tmp1 = einsum('ji,aj->ai', f_aa[o, o], t1_aa, optimize=True)
    t1_res -= 1 * _tmp1
    _tmp2 = einsum('ab,bi->ai', f_aa[v, v], t1_aa, optimize=True)
    t1_res += 1 * _tmp2
    _tmp3 = einsum('jb,baij->ai', f_aa[o, v], t2_aaaa, optimize=True)
    t1_res -= 1 * _tmp3
    _tmp4 = einsum('jb,abij->ai', f_bb[o, v], t2_abab, optimize=True)
    t1_res += 1 * _tmp4
    _tmp5 = einsum('jb,aj,bi->ai', f_aa[o, v], t1_aa, t1_aa, optimize=True)
    t1_res -= 1 * _tmp5
    _tmp6 = einsum('jabi,bj->ai', g_aaaa[o, v, v, o], t1_aa, optimize=True)
    t1_res += 1 * _tmp6
    _tmp7 = einsum('ajib,bj->ai', g_abab[v, o, o, v], t1_bb, optimize=True)
    t1_res += 1 * _tmp7
    _tmp8 = einsum('kjbi,bakj->ai', g_aaaa[o, o, v, o], t2_aaaa, optimize=True)
    t1_res -= 0.5 * _tmp8
    _tmp9 = einsum('kjib,abkj->ai', g_abab[o, o, o, v], t2_abab, optimize=True)
    t1_res -= 0.5 * _tmp9
    t1_res -= 0.5 * _tmp9
    _tmp10 = einsum('jabc,bcij->ai', g_aaaa[o, v, v, v], t2_aaaa, optimize=True)
    t1_res -= 0.5 * _tmp10
    _tmp11 = einsum('ajbc,bcij->ai', g_abab[v, o, v, v], t2_abab, optimize=True)
    t1_res += 0.5 * _tmp11
    t1_res += 0.5 * _tmp11
    _tmp12 = einsum('kjbc,bcaikj->ai', g_aaaa[o, o, v, v], t3_aaaaaa, optimize=True)
    t1_res += 0.25 * _tmp12
    _tmp13 = einsum('kjbc,bacikj->ai', g_abab[o, o, v, v], t3_aabaab, optimize=True)
    t1_res -= 0.25 * _tmp13
    t1_res -= 0.25 * _tmp13
    _tmp14 = einsum('kjcb,acbikj->ai', g_abab[o, o, v, v], t3_aabaab, optimize=True)
    t1_res += 0.25 * _tmp14
    t1_res += 0.25 * _tmp14
    _tmp15 = einsum('kjbc,acbikj->ai', g_bbbb[o, o, v, v], t3_abbabb, optimize=True)
    t1_res -= 0.25 * _tmp15
    _tmp16 = einsum('kjbc,caik,bj->ai', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, optimize=True)
    t1_res += 1 * _tmp16
    _tmp17 = einsum('kjcb,caik,bj->ai', g_abab[o, o, v, v], t2_aaaa, t1_bb, optimize=True)
    t1_res -= 1 * _tmp17
    _tmp18 = einsum('jkbc,acik,bj->ai', g_abab[o, o, v, v], t2_abab, t1_aa, optimize=True)
    t1_res += 1 * _tmp18
    _tmp19 = einsum('kjbc,acik,bj->ai', g_bbbb[o, o, v, v], t2_abab, t1_bb, optimize=True)
    t1_res -= 1 * _tmp19
    _tmp20 = einsum('kjbc,cakj,bi->ai', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, optimize=True)
    t1_res += 0.5 * _tmp20
    _tmp21 = einsum('kjbc,ackj,bi->ai', g_abab[o, o, v, v], t2_abab, t1_aa, optimize=True)
    t1_res -= 0.5 * _tmp21
    t1_res -= 0.5 * _tmp21
    _tmp22 = einsum('kjbc,aj,bcik->ai', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, optimize=True)
    t1_res += 0.5 * _tmp22
    _tmp23 = einsum('jkbc,aj,bcik->ai', g_abab[o, o, v, v], t1_aa, t2_abab, optimize=True)
    t1_res -= 0.5 * _tmp23
    t1_res -= 0.5 * _tmp23
    _tmp24 = einsum('kjbi,ak,bj->ai', g_aaaa[o, o, v, o], t1_aa, t1_aa, optimize=True)
    t1_res += 1 * _tmp24
    _tmp25 = einsum('kjib,ak,bj->ai', g_abab[o, o, o, v], t1_aa, t1_bb, optimize=True)
    t1_res -= 1 * _tmp25
    _tmp26 = einsum('jabc,bj,ci->ai', g_aaaa[o, v, v, v], t1_aa, t1_aa, optimize=True)
    t1_res += 1 * _tmp26
    _tmp27 = einsum('ajcb,bj,ci->ai', g_abab[v, o, v, v], t1_bb, t1_aa, optimize=True)
    t1_res += 1 * _tmp27
    _tmp28 = einsum('kjbc,ak,bj,ci->ai', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, optimize=True)
    t1_res += 1 * _tmp28
    _tmp29 = einsum('kjcb,ak,bj,ci->ai', g_abab[o, o, v, v], t1_aa, t1_bb, t1_aa, optimize=True)
    t1_res -= 1 * _tmp29
    return t1_res


def t2_aaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v):
    nv, no = t1_aa.shape
    t2_res = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('kj,abik->abij', f_aa[o, o], t2_aaaa, optimize=True)
    t2_res -= 1 * _tmp0
    t2_res += 1 * _tmp0.transpose(0, 1, 3, 2)
    _tmp1 = einsum('ac,cbij->abij', f_aa[v, v], t2_aaaa, optimize=True)
    t2_res += 1 * _tmp1
    t2_res -= 1 * _tmp1.transpose(1, 0, 2, 3)
    _tmp2 = einsum('kc,cabijk->abij', f_aa[o, v], t3_aaaaaa, optimize=True)
    t2_res += 1 * _tmp2
    _tmp3 = einsum('kc,bacijk->abij', f_bb[o, v], t3_aabaab, optimize=True)
    t2_res -= 1 * _tmp3
    _tmp4 = einsum('kc,abik,cj->abij', f_aa[o, v], t2_aaaa, t1_aa, optimize=True)
    t2_res -= 1 * _tmp4
    t2_res += 1 * _tmp4.transpose(0, 1, 3, 2)
    _tmp5 = einsum('kc,ak,cbij->abij', f_aa[o, v], t1_aa, t2_aaaa, optimize=True)
    t2_res -= 1 * _tmp5
    t2_res += 1 * _tmp5.transpose(1, 0, 2, 3)
    _tmp6 = einsum('abij->abij', g_aaaa[v, v, o, o])
    t2_res += 1 * _tmp6
    _tmp7 = einsum('kaij,bk->abij', g_aaaa[o, v, o, o], t1_aa, optimize=True)
    t2_res += 1 * _tmp7
    t2_res -= 1 * _tmp7.transpose(1, 0, 2, 3)
    _tmp8 = einsum('abcj,ci->abij', g_aaaa[v, v, v, o], t1_aa, optimize=True)
    t2_res += 1 * _tmp8
    t2_res -= 1 * _tmp8.transpose(0, 1, 3, 2)
    _tmp9 = einsum('lkij,ablk->abij', g_aaaa[o, o, o, o], t2_aaaa, optimize=True)
    t2_res += 0.5 * _tmp9
    _tmp10 = einsum('kacj,cbik->abij', g_aaaa[o, v, v, o], t2_aaaa, optimize=True)
    t2_res += 1 * _tmp10
    t2_res -= 1 * _tmp10.transpose(1, 0, 2, 3)
    t2_res -= 1 * _tmp10.transpose(0, 1, 3, 2)
    t2_res += 1 * _tmp10.transpose(1, 0, 3, 2)
    _tmp11 = einsum('akjc,bcik->abij', g_abab[v, o, o, v], t2_abab, optimize=True)
    t2_res -= 1 * _tmp11
    t2_res += 1 * _tmp11.transpose(1, 0, 2, 3)
    t2_res += 1 * _tmp11.transpose(0, 1, 3, 2)
    t2_res -= 1 * _tmp11.transpose(1, 0, 3, 2)
    _tmp12 = einsum('abcd,cdij->abij', g_aaaa[v, v, v, v], t2_aaaa, optimize=True)
    t2_res += 0.5 * _tmp12
    _tmp13 = einsum('lkcj,cabilk->abij', g_aaaa[o, o, v, o], t3_aaaaaa, optimize=True)
    t2_res += 0.5 * _tmp13
    t2_res -= 0.5 * _tmp13.transpose(0, 1, 3, 2)
    _tmp14 = einsum('lkjc,bacilk->abij', g_abab[o, o, o, v], t3_aabaab, optimize=True)
    t2_res += 0.5 * _tmp14
    t2_res -= 0.5 * _tmp14.transpose(0, 1, 3, 2)
    t2_res += 0.5 * _tmp14
    t2_res -= 0.5 * _tmp14.transpose(0, 1, 3, 2)
    _tmp15 = einsum('kacd,cdbijk->abij', g_aaaa[o, v, v, v], t3_aaaaaa, optimize=True)
    t2_res += 0.5 * _tmp15
    t2_res -= 0.5 * _tmp15.transpose(1, 0, 2, 3)
    _tmp16 = einsum('akcd,cbdijk->abij', g_abab[v, o, v, v], t3_aabaab, optimize=True)
    t2_res += 0.5 * _tmp16
    t2_res -= 0.5 * _tmp16.transpose(1, 0, 2, 3)
    _tmp17 = einsum('akdc,bdcijk->abij', g_abab[v, o, v, v], t3_aabaab, optimize=True)
    t2_res -= 0.5 * _tmp17
    t2_res += 0.5 * _tmp17.transpose(1, 0, 2, 3)
    _tmp18 = einsum('lkcj,abil,ck->abij', g_aaaa[o, o, v, o], t2_aaaa, t1_aa, optimize=True)
    t2_res += 1 * _tmp18
    t2_res -= 1 * _tmp18.transpose(0, 1, 3, 2)
    _tmp19 = einsum('lkjc,abil,ck->abij', g_abab[o, o, o, v], t2_aaaa, t1_bb, optimize=True)
    t2_res -= 1 * _tmp19
    t2_res += 1 * _tmp19.transpose(0, 1, 3, 2)
    _tmp20 = einsum('lkcj,ablk,ci->abij', g_aaaa[o, o, v, o], t2_aaaa, t1_aa, optimize=True)
    t2_res += 0.5 * _tmp20
    t2_res -= 0.5 * _tmp20.transpose(0, 1, 3, 2)
    _tmp21 = einsum('lkcj,ak,cbil->abij', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, optimize=True)
    t2_res -= 1 * _tmp21
    t2_res += 1 * _tmp21.transpose(1, 0, 2, 3)
    t2_res += 1 * _tmp21.transpose(0, 1, 3, 2)
    t2_res -= 1 * _tmp21.transpose(1, 0, 3, 2)
    _tmp22 = einsum('kljc,ak,bcil->abij', g_abab[o, o, o, v], t1_aa, t2_abab, optimize=True)
    t2_res += 1 * _tmp22
    t2_res -= 1 * _tmp22.transpose(1, 0, 2, 3)
    t2_res -= 1 * _tmp22.transpose(0, 1, 3, 2)
    t2_res += 1 * _tmp22.transpose(1, 0, 3, 2)
    _tmp23 = einsum('kacd,dbij,ck->abij', g_aaaa[o, v, v, v], t2_aaaa, t1_aa, optimize=True)
    t2_res += 1 * _tmp23
    t2_res -= 1 * _tmp23.transpose(1, 0, 2, 3)
    _tmp24 = einsum('akdc,dbij,ck->abij', g_abab[v, o, v, v], t2_aaaa, t1_bb, optimize=True)
    t2_res += 1 * _tmp24
    t2_res -= 1 * _tmp24.transpose(1, 0, 2, 3)
    _tmp25 = einsum('kacd,dbik,cj->abij', g_aaaa[o, v, v, v], t2_aaaa, t1_aa, optimize=True)
    t2_res -= 1 * _tmp25
    t2_res += 1 * _tmp25.transpose(1, 0, 2, 3)
    t2_res += 1 * _tmp25.transpose(0, 1, 3, 2)
    t2_res -= 1 * _tmp25.transpose(1, 0, 3, 2)
    _tmp26 = einsum('akcd,bdik,cj->abij', g_abab[v, o, v, v], t2_abab, t1_aa, optimize=True)
    t2_res -= 1 * _tmp26
    t2_res += 1 * _tmp26.transpose(1, 0, 2, 3)
    t2_res += 1 * _tmp26.transpose(0, 1, 3, 2)
    t2_res -= 1 * _tmp26.transpose(1, 0, 3, 2)
    _tmp27 = einsum('kacd,bk,cdij->abij', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, optimize=True)
    t2_res += 0.5 * _tmp27
    t2_res -= 0.5 * _tmp27.transpose(1, 0, 2, 3)
    _tmp28 = einsum('lkcd,dabijl,ck->abij', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, optimize=True)
    t2_res -= 1 * _tmp28
    _tmp29 = einsum('lkdc,dabijl,ck->abij', g_abab[o, o, v, v], t3_aaaaaa, t1_bb, optimize=True)
    t2_res += 1 * _tmp29
    _tmp30 = einsum('klcd,badijl,ck->abij', g_abab[o, o, v, v], t3_aabaab, t1_aa, optimize=True)
    t2_res -= 1 * _tmp30
    _tmp31 = einsum('lkcd,badijl,ck->abij', g_bbbb[o, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t2_res += 1 * _tmp31
    _tmp32 = einsum('lkcd,dabilk,cj->abij', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, optimize=True)
    t2_res -= 0.5 * _tmp32
    t2_res += 0.5 * _tmp32.transpose(0, 1, 3, 2)
    _tmp33 = einsum('lkcd,badilk,cj->abij', g_abab[o, o, v, v], t3_aabaab, t1_aa, optimize=True)
    t2_res += 0.5 * _tmp33
    t2_res -= 0.5 * _tmp33.transpose(0, 1, 3, 2)
    t2_res += 0.5 * _tmp33
    t2_res -= 0.5 * _tmp33.transpose(0, 1, 3, 2)
    _tmp34 = einsum('lkcd,ak,cdbijl->abij', g_aaaa[o, o, v, v], t1_aa, t3_aaaaaa, optimize=True)
    t2_res -= 0.5 * _tmp34
    t2_res += 0.5 * _tmp34.transpose(1, 0, 2, 3)
    _tmp35 = einsum('klcd,ak,cbdijl->abij', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=True)
    t2_res -= 0.5 * _tmp35
    t2_res += 0.5 * _tmp35.transpose(1, 0, 2, 3)
    _tmp36 = einsum('kldc,ak,bdcijl->abij', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=True)
    t2_res += 0.5 * _tmp36
    t2_res -= 0.5 * _tmp36.transpose(1, 0, 2, 3)
    _tmp37 = einsum('lkij,ak,bl->abij', g_aaaa[o, o, o, o], t1_aa, t1_aa, optimize=True)
    t2_res -= 1 * _tmp37
    _tmp38 = einsum('kacj,bk,ci->abij', g_aaaa[o, v, v, o], t1_aa, t1_aa, optimize=True)
    t2_res += 1 * _tmp38
    t2_res -= 1 * _tmp38.transpose(1, 0, 2, 3)
    t2_res -= 1 * _tmp38.transpose(0, 1, 3, 2)
    t2_res += 1 * _tmp38.transpose(1, 0, 3, 2)
    _tmp39 = einsum('abcd,cj,di->abij', g_aaaa[v, v, v, v], t1_aa, t1_aa, optimize=True)
    t2_res -= 1 * _tmp39
    _tmp40 = einsum('lkcd,abil,cdjk->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t2_res -= 0.5 * _tmp40
    t2_res += 0.5 * _tmp40.transpose(0, 1, 3, 2)
    _tmp41 = einsum('lkcd,abil,cdjk->abij', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=True)
    t2_res -= 0.5 * _tmp41
    t2_res += 0.5 * _tmp41.transpose(0, 1, 3, 2)
    t2_res -= 0.5 * _tmp41
    t2_res += 0.5 * _tmp41.transpose(0, 1, 3, 2)
    _tmp42 = einsum('lkcd,ablk,cdij->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t2_res += 0.25 * _tmp42
    _tmp43 = einsum('lkcd,calk,dbij->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t2_res -= 0.5 * _tmp43
    _tmp44 = einsum('lkdc,aclk,dbij->abij', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=True)
    t2_res -= 0.5 * _tmp44
    t2_res -= 0.5 * _tmp44
    _tmp45 = einsum('lkcd,cajk,dbil->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t2_res += 1 * _tmp45
    t2_res -= 1 * _tmp45.transpose(0, 1, 3, 2)
    _tmp46 = einsum('klcd,cajk,bdil->abij', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=True)
    t2_res += 1 * _tmp46
    t2_res -= 1 * _tmp46.transpose(0, 1, 3, 2)
    _tmp47 = einsum('lkdc,acjk,dbil->abij', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=True)
    t2_res += 1 * _tmp47
    t2_res -= 1 * _tmp47.transpose(0, 1, 3, 2)
    _tmp48 = einsum('lkcd,acjk,bdil->abij', g_bbbb[o, o, v, v], t2_abab, t2_abab, optimize=True)
    t2_res += 1 * _tmp48
    t2_res -= 1 * _tmp48.transpose(0, 1, 3, 2)
    _tmp49 = einsum('lkcd,caij,dblk->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t2_res -= 0.5 * _tmp49
    _tmp50 = einsum('lkcd,caij,bdlk->abij', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=True)
    t2_res += 0.5 * _tmp50
    t2_res += 0.5 * _tmp50
    _tmp51 = einsum('lkcd,abil,ck,dj->abij', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t2_res += 1 * _tmp51
    t2_res -= 1 * _tmp51.transpose(0, 1, 3, 2)
    _tmp52 = einsum('lkdc,abil,ck,dj->abij', g_abab[o, o, v, v], t2_aaaa, t1_bb, t1_aa, optimize=True)
    t2_res -= 1 * _tmp52
    t2_res += 1 * _tmp52.transpose(0, 1, 3, 2)
    _tmp53 = einsum('lkcd,al,dbij,ck->abij', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t2_res += 1 * _tmp53
    t2_res -= 1 * _tmp53.transpose(1, 0, 2, 3)
    _tmp54 = einsum('lkdc,al,dbij,ck->abij', g_abab[o, o, v, v], t1_aa, t2_aaaa, t1_bb, optimize=True)
    t2_res -= 1 * _tmp54
    t2_res += 1 * _tmp54.transpose(1, 0, 2, 3)
    _tmp55 = einsum('lkcd,ablk,cj,di->abij', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t2_res -= 0.5 * _tmp55
    _tmp56 = einsum('lkcd,ak,dbil,cj->abij', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t2_res += 1 * _tmp56
    t2_res -= 1 * _tmp56.transpose(1, 0, 2, 3)
    t2_res -= 1 * _tmp56.transpose(0, 1, 3, 2)
    t2_res += 1 * _tmp56.transpose(1, 0, 3, 2)
    _tmp57 = einsum('klcd,ak,bdil,cj->abij', g_abab[o, o, v, v], t1_aa, t2_abab, t1_aa, optimize=True)
    t2_res += 1 * _tmp57
    t2_res -= 1 * _tmp57.transpose(1, 0, 2, 3)
    t2_res -= 1 * _tmp57.transpose(0, 1, 3, 2)
    t2_res += 1 * _tmp57.transpose(1, 0, 3, 2)
    _tmp58 = einsum('lkcd,ak,bl,cdij->abij', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=True)
    t2_res -= 0.5 * _tmp58
    _tmp59 = einsum('lkcj,ak,bl,ci->abij', g_aaaa[o, o, v, o], t1_aa, t1_aa, t1_aa, optimize=True)
    t2_res -= 1 * _tmp59
    t2_res += 1 * _tmp59.transpose(0, 1, 3, 2)
    _tmp60 = einsum('kacd,bk,cj,di->abij', g_aaaa[o, v, v, v], t1_aa, t1_aa, t1_aa, optimize=True)
    t2_res -= 1 * _tmp60
    t2_res += 1 * _tmp60.transpose(1, 0, 2, 3)
    _tmp61 = einsum('lkcd,ak,bl,cj,di->abij', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t1_aa, optimize=True)
    t2_res += 1 * _tmp61
    return t2_res


def t2_abab_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v):
    nv, no = t1_aa.shape
    t2_res = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('kj,abik->abij', f_bb[o, o], t2_abab, optimize=True)
    t2_res -= 1 * _tmp0
    _tmp1 = einsum('ki,abkj->abij', f_aa[o, o], t2_abab, optimize=True)
    t2_res -= 1 * _tmp1
    _tmp2 = einsum('ac,cbij->abij', f_aa[v, v], t2_abab, optimize=True)
    t2_res += 1 * _tmp2
    _tmp3 = einsum('bc,acij->abij', f_bb[v, v], t2_abab, optimize=True)
    t2_res += 1 * _tmp3
    _tmp4 = einsum('kc,cabikj->abij', f_aa[o, v], t3_aabaab, optimize=True)
    t2_res -= 1 * _tmp4
    _tmp5 = einsum('kc,acbijk->abij', f_bb[o, v], t3_abbabb, optimize=True)
    t2_res -= 1 * _tmp5
    _tmp6 = einsum('kc,abik,cj->abij', f_bb[o, v], t2_abab, t1_bb, optimize=True)
    t2_res -= 1 * _tmp6
    _tmp7 = einsum('kc,abkj,ci->abij', f_aa[o, v], t2_abab, t1_aa, optimize=True)
    t2_res -= 1 * _tmp7
    _tmp8 = einsum('kc,ak,cbij->abij', f_aa[o, v], t1_aa, t2_abab, optimize=True)
    t2_res -= 1 * _tmp8
    _tmp9 = einsum('kc,acij,bk->abij', f_bb[o, v], t2_abab, t1_bb, optimize=True)
    t2_res -= 1 * _tmp9
    _tmp10 = einsum('abij->abij', g_abab[v, v, o, o])
    t2_res += 1 * _tmp10
    _tmp11 = einsum('akij,bk->abij', g_abab[v, o, o, o], t1_bb, optimize=True)
    t2_res -= 1 * _tmp11
    _tmp12 = einsum('kbij,ak->abij', g_abab[o, v, o, o], t1_aa, optimize=True)
    t2_res -= 1 * _tmp12
    _tmp13 = einsum('abcj,ci->abij', g_abab[v, v, v, o], t1_aa, optimize=True)
    t2_res += 1 * _tmp13
    _tmp14 = einsum('abic,cj->abij', g_abab[v, v, o, v], t1_bb, optimize=True)
    t2_res += 1 * _tmp14
    _tmp15 = einsum('lkij,ablk->abij', g_abab[o, o, o, o], t2_abab, optimize=True)
    t2_res += 0.5 * _tmp15
    t2_res += 0.5 * _tmp15
    _tmp16 = einsum('akcj,cbik->abij', g_abab[v, o, v, o], t2_abab, optimize=True)
    t2_res -= 1 * _tmp16
    _tmp17 = einsum('kbcj,caik->abij', g_abab[o, v, v, o], t2_aaaa, optimize=True)
    t2_res -= 1 * _tmp17
    _tmp18 = einsum('kbcj,acik->abij', g_bbbb[o, v, v, o], t2_abab, optimize=True)
    t2_res += 1 * _tmp18
    _tmp19 = einsum('kaci,cbkj->abij', g_aaaa[o, v, v, o], t2_abab, optimize=True)
    t2_res += 1 * _tmp19
    _tmp20 = einsum('akic,cbjk->abij', g_abab[v, o, o, v], t2_bbbb, optimize=True)
    t2_res -= 1 * _tmp20
    _tmp21 = einsum('kbic,ackj->abij', g_abab[o, v, o, v], t2_abab, optimize=True)
    t2_res -= 1 * _tmp21
    _tmp22 = einsum('abcd,cdij->abij', g_abab[v, v, v, v], t2_abab, optimize=True)
    t2_res += 0.5 * _tmp22
    t2_res += 0.5 * _tmp22
    _tmp23 = einsum('lkcj,cabilk->abij', g_abab[o, o, v, o], t3_aabaab, optimize=True)
    t2_res += 0.5 * _tmp23
    t2_res += 0.5 * _tmp23
    _tmp24 = einsum('lkcj,acbilk->abij', g_bbbb[o, o, v, o], t3_abbabb, optimize=True)
    t2_res -= 0.5 * _tmp24
    _tmp25 = einsum('lkci,cabklj->abij', g_aaaa[o, o, v, o], t3_aabaab, optimize=True)
    t2_res += 0.5 * _tmp25
    _tmp26 = einsum('lkic,acbljk->abij', g_abab[o, o, o, v], t3_abbabb, optimize=True)
    t2_res += 0.5 * _tmp26
    _tmp27 = einsum('klic,acbklj->abij', g_abab[o, o, o, v], t3_abbabb, optimize=True)
    t2_res -= 0.5 * _tmp27
    _tmp28 = einsum('kacd,cdbikj->abij', g_aaaa[o, v, v, v], t3_aabaab, optimize=True)
    t2_res -= 0.5 * _tmp28
    _tmp29 = einsum('akcd,cdbijk->abij', g_abab[v, o, v, v], t3_abbabb, optimize=True)
    t2_res -= 0.5 * _tmp29
    t2_res -= 0.5 * _tmp29
    _tmp30 = einsum('kbcd,cadikj->abij', g_abab[o, v, v, v], t3_aabaab, optimize=True)
    t2_res -= 0.5 * _tmp30
    _tmp31 = einsum('kbdc,adcikj->abij', g_abab[o, v, v, v], t3_aabaab, optimize=True)
    t2_res += 0.5 * _tmp31
    _tmp32 = einsum('kbcd,adcijk->abij', g_bbbb[o, v, v, v], t3_abbabb, optimize=True)
    t2_res += 0.5 * _tmp32
    _tmp33 = einsum('klcj,abil,ck->abij', g_abab[o, o, v, o], t2_abab, t1_aa, optimize=True)
    t2_res -= 1 * _tmp33
    _tmp34 = einsum('lkcj,abil,ck->abij', g_bbbb[o, o, v, o], t2_abab, t1_bb, optimize=True)
    t2_res += 1 * _tmp34
    _tmp35 = einsum('lkci,ablj,ck->abij', g_aaaa[o, o, v, o], t2_abab, t1_aa, optimize=True)
    t2_res += 1 * _tmp35
    _tmp36 = einsum('lkic,ablj,ck->abij', g_abab[o, o, o, v], t2_abab, t1_bb, optimize=True)
    t2_res -= 1 * _tmp36
    _tmp37 = einsum('lkcj,ablk,ci->abij', g_abab[o, o, v, o], t2_abab, t1_aa, optimize=True)
    t2_res += 0.5 * _tmp37
    t2_res += 0.5 * _tmp37
    _tmp38 = einsum('lkic,ablk,cj->abij', g_abab[o, o, o, v], t2_abab, t1_bb, optimize=True)
    t2_res += 0.5 * _tmp38
    t2_res += 0.5 * _tmp38
    _tmp39 = einsum('klcj,ak,cbil->abij', g_abab[o, o, v, o], t1_aa, t2_abab, optimize=True)
    t2_res += 1 * _tmp39
    _tmp40 = einsum('lkcj,cail,bk->abij', g_abab[o, o, v, o], t2_aaaa, t1_bb, optimize=True)
    t2_res += 1 * _tmp40
    _tmp41 = einsum('lkcj,acil,bk->abij', g_bbbb[o, o, v, o], t2_abab, t1_bb, optimize=True)
    t2_res -= 1 * _tmp41
    _tmp42 = einsum('lkci,ak,cblj->abij', g_aaaa[o, o, v, o], t1_aa, t2_abab, optimize=True)
    t2_res -= 1 * _tmp42
    _tmp43 = einsum('klic,ak,cbjl->abij', g_abab[o, o, o, v], t1_aa, t2_bbbb, optimize=True)
    t2_res += 1 * _tmp43
    _tmp44 = einsum('lkic,aclj,bk->abij', g_abab[o, o, o, v], t2_abab, t1_bb, optimize=True)
    t2_res += 1 * _tmp44
    _tmp45 = einsum('kacd,dbij,ck->abij', g_aaaa[o, v, v, v], t2_abab, t1_aa, optimize=True)
    t2_res += 1 * _tmp45
    _tmp46 = einsum('akdc,dbij,ck->abij', g_abab[v, o, v, v], t2_abab, t1_bb, optimize=True)
    t2_res += 1 * _tmp46
    _tmp47 = einsum('kbcd,adij,ck->abij', g_abab[o, v, v, v], t2_abab, t1_aa, optimize=True)
    t2_res += 1 * _tmp47
    _tmp48 = einsum('kbcd,adij,ck->abij', g_bbbb[o, v, v, v], t2_abab, t1_bb, optimize=True)
    t2_res += 1 * _tmp48
    _tmp49 = einsum('akdc,dbik,cj->abij', g_abab[v, o, v, v], t2_abab, t1_bb, optimize=True)
    t2_res -= 1 * _tmp49
    _tmp50 = einsum('kbdc,daik,cj->abij', g_abab[o, v, v, v], t2_aaaa, t1_bb, optimize=True)
    t2_res -= 1 * _tmp50
    _tmp51 = einsum('kbcd,adik,cj->abij', g_bbbb[o, v, v, v], t2_abab, t1_bb, optimize=True)
    t2_res -= 1 * _tmp51
    _tmp52 = einsum('kacd,dbkj,ci->abij', g_aaaa[o, v, v, v], t2_abab, t1_aa, optimize=True)
    t2_res -= 1 * _tmp52
    _tmp53 = einsum('akcd,dbjk,ci->abij', g_abab[v, o, v, v], t2_bbbb, t1_aa, optimize=True)
    t2_res -= 1 * _tmp53
    _tmp54 = einsum('kbcd,adkj,ci->abij', g_abab[o, v, v, v], t2_abab, t1_aa, optimize=True)
    t2_res -= 1 * _tmp54
    _tmp55 = einsum('akcd,bk,cdij->abij', g_abab[v, o, v, v], t1_bb, t2_abab, optimize=True)
    t2_res -= 0.5 * _tmp55
    t2_res -= 0.5 * _tmp55
    _tmp56 = einsum('kbcd,ak,cdij->abij', g_abab[o, v, v, v], t1_aa, t2_abab, optimize=True)
    t2_res -= 0.5 * _tmp56
    t2_res -= 0.5 * _tmp56
    _tmp57 = einsum('lkcd,dabilj,ck->abij', g_aaaa[o, o, v, v], t3_aabaab, t1_aa, optimize=True)
    t2_res += 1 * _tmp57
    _tmp58 = einsum('lkdc,dabilj,ck->abij', g_abab[o, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t2_res -= 1 * _tmp58
    _tmp59 = einsum('klcd,adbijl,ck->abij', g_abab[o, o, v, v], t3_abbabb, t1_aa, optimize=True)
    t2_res -= 1 * _tmp59
    _tmp60 = einsum('lkcd,adbijl,ck->abij', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, optimize=True)
    t2_res += 1 * _tmp60
    _tmp61 = einsum('lkdc,dabilk,cj->abij', g_abab[o, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t2_res += 0.5 * _tmp61
    t2_res += 0.5 * _tmp61
    _tmp62 = einsum('lkcd,adbilk,cj->abij', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, optimize=True)
    t2_res += 0.5 * _tmp62
    _tmp63 = einsum('lkcd,dabklj,ci->abij', g_aaaa[o, o, v, v], t3_aabaab, t1_aa, optimize=True)
    t2_res -= 0.5 * _tmp63
    _tmp64 = einsum('lkcd,adbljk,ci->abij', g_abab[o, o, v, v], t3_abbabb, t1_aa, optimize=True)
    t2_res += 0.5 * _tmp64
    _tmp65 = einsum('klcd,adbklj,ci->abij', g_abab[o, o, v, v], t3_abbabb, t1_aa, optimize=True)
    t2_res -= 0.5 * _tmp65
    _tmp66 = einsum('lkcd,ak,cdbilj->abij', g_aaaa[o, o, v, v], t1_aa, t3_aabaab, optimize=True)
    t2_res += 0.5 * _tmp66
    _tmp67 = einsum('klcd,ak,cdbijl->abij', g_abab[o, o, v, v], t1_aa, t3_abbabb, optimize=True)
    t2_res += 0.5 * _tmp67
    t2_res += 0.5 * _tmp67
    _tmp68 = einsum('lkcd,cadilj,bk->abij', g_abab[o, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t2_res += 0.5 * _tmp68
    _tmp69 = einsum('lkdc,adcilj,bk->abij', g_abab[o, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t2_res -= 0.5 * _tmp69
    _tmp70 = einsum('lkcd,adcijl,bk->abij', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, optimize=True)
    t2_res -= 0.5 * _tmp70
    _tmp71 = einsum('klij,ak,bl->abij', g_abab[o, o, o, o], t1_aa, t1_bb, optimize=True)
    t2_res += 1 * _tmp71
    _tmp72 = einsum('akcj,bk,ci->abij', g_abab[v, o, v, o], t1_bb, t1_aa, optimize=True)
    t2_res -= 1 * _tmp72
    _tmp73 = einsum('kbcj,ak,ci->abij', g_abab[o, v, v, o], t1_aa, t1_aa, optimize=True)
    t2_res -= 1 * _tmp73
    _tmp74 = einsum('akic,bk,cj->abij', g_abab[v, o, o, v], t1_bb, t1_bb, optimize=True)
    t2_res -= 1 * _tmp74
    _tmp75 = einsum('kbic,ak,cj->abij', g_abab[o, v, o, v], t1_aa, t1_bb, optimize=True)
    t2_res -= 1 * _tmp75
    _tmp76 = einsum('abdc,cj,di->abij', g_abab[v, v, v, v], t1_bb, t1_aa, optimize=True)
    t2_res += 1 * _tmp76
    _tmp77 = einsum('klcd,abil,cdkj->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=True)
    t2_res -= 0.5 * _tmp77
    t2_res -= 0.5 * _tmp77
    _tmp78 = einsum('lkcd,abil,cdjk->abij', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, optimize=True)
    t2_res -= 0.5 * _tmp78
    _tmp79 = einsum('lkcd,ablj,cdik->abij', g_aaaa[o, o, v, v], t2_abab, t2_aaaa, optimize=True)
    t2_res -= 0.5 * _tmp79
    _tmp80 = einsum('lkcd,ablj,cdik->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=True)
    t2_res -= 0.5 * _tmp80
    t2_res -= 0.5 * _tmp80
    _tmp81 = einsum('lkcd,ablk,cdij->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=True)
    t2_res += 0.25 * _tmp81
    t2_res += 0.25 * _tmp81
    t2_res += 0.25 * _tmp81
    t2_res += 0.25 * _tmp81
    _tmp82 = einsum('lkcd,calk,dbij->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=True)
    t2_res -= 0.5 * _tmp82
    _tmp83 = einsum('lkdc,aclk,dbij->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=True)
    t2_res -= 0.5 * _tmp83
    t2_res -= 0.5 * _tmp83
    _tmp84 = einsum('kldc,ackj,dbil->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=True)
    t2_res += 1 * _tmp84
    _tmp85 = einsum('lkcd,caik,dblj->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=True)
    t2_res += 1 * _tmp85
    _tmp86 = einsum('klcd,caik,dbjl->abij', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, optimize=True)
    t2_res += 1 * _tmp86
    _tmp87 = einsum('lkdc,acik,dblj->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=True)
    t2_res += 1 * _tmp87
    _tmp88 = einsum('lkcd,acik,dbjl->abij', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, optimize=True)
    t2_res += 1 * _tmp88
    _tmp89 = einsum('lkdc,acij,dblk->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=True)
    t2_res -= 0.5 * _tmp89
    t2_res -= 0.5 * _tmp89
    _tmp90 = einsum('lkcd,acij,dblk->abij', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, optimize=True)
    t2_res += 0.5 * _tmp90
    _tmp91 = einsum('klcd,abil,ck,dj->abij', g_abab[o, o, v, v], t2_abab, t1_aa, t1_bb, optimize=True)
    t2_res -= 1 * _tmp91
    _tmp92 = einsum('lkcd,abil,ck,dj->abij', g_bbbb[o, o, v, v], t2_abab, t1_bb, t1_bb, optimize=True)
    t2_res += 1 * _tmp92
    _tmp93 = einsum('lkcd,ablj,ck,di->abij', g_aaaa[o, o, v, v], t2_abab, t1_aa, t1_aa, optimize=True)
    t2_res += 1 * _tmp93
    _tmp94 = einsum('lkdc,ablj,ck,di->abij', g_abab[o, o, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t2_res -= 1 * _tmp94
    _tmp95 = einsum('lkcd,al,dbij,ck->abij', g_aaaa[o, o, v, v], t1_aa, t2_abab, t1_aa, optimize=True)
    t2_res += 1 * _tmp95
    _tmp96 = einsum('lkdc,al,dbij,ck->abij', g_abab[o, o, v, v], t1_aa, t2_abab, t1_bb, optimize=True)
    t2_res -= 1 * _tmp96
    _tmp97 = einsum('klcd,adij,bl,ck->abij', g_abab[o, o, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t2_res -= 1 * _tmp97
    _tmp98 = einsum('lkcd,adij,bl,ck->abij', g_bbbb[o, o, v, v], t2_abab, t1_bb, t1_bb, optimize=True)
    t2_res += 1 * _tmp98
    _tmp99 = einsum('lkdc,ablk,cj,di->abij', g_abab[o, o, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t2_res += 0.5 * _tmp99
    t2_res += 0.5 * _tmp99
    _tmp100 = einsum('kldc,ak,dbil,cj->abij', g_abab[o, o, v, v], t1_aa, t2_abab, t1_bb, optimize=True)
    t2_res += 1 * _tmp100
    _tmp101 = einsum('lkdc,dail,bk,cj->abij', g_abab[o, o, v, v], t2_aaaa, t1_bb, t1_bb, optimize=True)
    t2_res += 1 * _tmp101
    _tmp102 = einsum('lkcd,adil,bk,cj->abij', g_bbbb[o, o, v, v], t2_abab, t1_bb, t1_bb, optimize=True)
    t2_res += 1 * _tmp102
    _tmp103 = einsum('lkcd,ak,dblj,ci->abij', g_aaaa[o, o, v, v], t1_aa, t2_abab, t1_aa, optimize=True)
    t2_res += 1 * _tmp103
    _tmp104 = einsum('klcd,ak,dbjl,ci->abij', g_abab[o, o, v, v], t1_aa, t2_bbbb, t1_aa, optimize=True)
    t2_res += 1 * _tmp104
    _tmp105 = einsum('lkcd,adlj,bk,ci->abij', g_abab[o, o, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t2_res += 1 * _tmp105
    _tmp106 = einsum('klcd,ak,bl,cdij->abij', g_abab[o, o, v, v], t1_aa, t1_bb, t2_abab, optimize=True)
    t2_res += 0.5 * _tmp106
    t2_res += 0.5 * _tmp106
    _tmp107 = einsum('klcj,ak,bl,ci->abij', g_abab[o, o, v, o], t1_aa, t1_bb, t1_aa, optimize=True)
    t2_res += 1 * _tmp107
    _tmp108 = einsum('klic,ak,bl,cj->abij', g_abab[o, o, o, v], t1_aa, t1_bb, t1_bb, optimize=True)
    t2_res += 1 * _tmp108
    _tmp109 = einsum('akdc,bk,cj,di->abij', g_abab[v, o, v, v], t1_bb, t1_bb, t1_aa, optimize=True)
    t2_res -= 1 * _tmp109
    _tmp110 = einsum('kbdc,ak,cj,di->abij', g_abab[o, v, v, v], t1_aa, t1_bb, t1_aa, optimize=True)
    t2_res -= 1 * _tmp110
    _tmp111 = einsum('kldc,ak,bl,cj,di->abij', g_abab[o, o, v, v], t1_aa, t1_bb, t1_bb, t1_aa, optimize=True)
    t2_res += 1 * _tmp111
    return t2_res


def t3_aaaaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v):
    nv, no = t1_aa.shape
    t3_res = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('lk,abcijl->abcijk', f_aa[o, o], t3_aaaaaa, optimize=True)
    t3_res -= 1 * _tmp0
    t3_res += 1 * _tmp0.transpose(0, 1, 2, 3, 5, 4)
    _tmp1 = einsum('li,abcjkl->abcijk', f_aa[o, o], t3_aaaaaa, optimize=True)
    t3_res -= 1 * _tmp1
    _tmp2 = einsum('ad,dbcijk->abcijk', f_aa[v, v], t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp2
    t3_res -= 1 * _tmp2.transpose(1, 0, 2, 3, 4, 5)
    _tmp3 = einsum('cd,dabijk->abcijk', f_aa[v, v], t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp3
    _tmp4 = einsum('ld,abcijl,dk->abcijk', f_aa[o, v], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp4
    t3_res += 1 * _tmp4.transpose(0, 1, 2, 3, 5, 4)
    _tmp5 = einsum('ld,abcjkl,di->abcijk', f_aa[o, v], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp5
    _tmp6 = einsum('ld,al,dbcijk->abcijk', f_aa[o, v], t1_aa, t3_aaaaaa, optimize=True)
    t3_res -= 1 * _tmp6
    t3_res += 1 * _tmp6.transpose(1, 0, 2, 3, 4, 5)
    _tmp7 = einsum('ld,dabijk,cl->abcijk', f_aa[o, v], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp7
    _tmp8 = einsum('ld,dajk,bcil->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp8
    t3_res += 1 * _tmp8.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp8.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp8.transpose(1, 0, 2, 4, 3, 5)
    _tmp9 = einsum('ld,daij,bckl->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp9
    t3_res += 1 * _tmp9.transpose(1, 0, 2, 3, 4, 5)
    _tmp10 = einsum('ld,abil,dcjk->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp10
    t3_res += 1 * _tmp10.transpose(0, 1, 2, 4, 3, 5)
    _tmp11 = einsum('ld,abkl,dcij->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp11
    _tmp12 = einsum('lajk,bcil->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp12
    t3_res += 1 * _tmp12.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp12.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp12.transpose(1, 0, 2, 4, 3, 5)
    _tmp13 = einsum('laij,bckl->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp13
    t3_res += 1 * _tmp13.transpose(1, 0, 2, 3, 4, 5)
    _tmp14 = einsum('lcjk,abil->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp14
    t3_res += 1 * _tmp14.transpose(0, 1, 2, 4, 3, 5)
    _tmp15 = einsum('lcij,abkl->abcijk', g_aaaa[o, v, o, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp15
    _tmp16 = einsum('abdk,dcij->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp16
    t3_res += 1 * _tmp16.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 1 * _tmp16.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp16.transpose(0, 2, 1, 3, 5, 4)
    _tmp17 = einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp17
    t3_res += 1 * _tmp17.transpose(0, 2, 1, 3, 4, 5)
    _tmp18 = einsum('bcdk,daij->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp18
    t3_res += 1 * _tmp18.transpose(0, 1, 2, 3, 5, 4)
    _tmp19 = einsum('bcdi,dajk->abcijk', g_aaaa[v, v, v, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp19
    _tmp20 = einsum('mljk,abciml->abcijk', g_aaaa[o, o, o, o], t3_aaaaaa, optimize=True)
    t3_res += 0.5 * _tmp20
    t3_res -= 0.5 * _tmp20.transpose(0, 1, 2, 4, 3, 5)
    _tmp21 = einsum('mlij,abckml->abcijk', g_aaaa[o, o, o, o], t3_aaaaaa, optimize=True)
    t3_res += 0.5 * _tmp21
    _tmp22 = einsum('ladk,dbcijl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp22
    t3_res -= 1 * _tmp22.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp22.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp22.transpose(1, 0, 2, 3, 5, 4)
    _tmp23 = einsum('alkd,cbdijl->abcijk', g_abab[v, o, o, v], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp23
    t3_res += 1 * _tmp23.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp23.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp23.transpose(1, 0, 2, 3, 5, 4)
    _tmp24 = einsum('ladi,dbcjkl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp24
    t3_res -= 1 * _tmp24.transpose(1, 0, 2, 3, 4, 5)
    _tmp25 = einsum('alid,cbdjkl->abcijk', g_abab[v, o, o, v], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp25
    t3_res += 1 * _tmp25.transpose(1, 0, 2, 3, 4, 5)
    _tmp26 = einsum('lcdk,dabijl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp26
    t3_res -= 1 * _tmp26.transpose(0, 1, 2, 3, 5, 4)
    _tmp27 = einsum('clkd,badijl->abcijk', g_abab[v, o, o, v], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp27
    t3_res += 1 * _tmp27.transpose(0, 1, 2, 3, 5, 4)
    _tmp28 = einsum('lcdi,dabjkl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp28
    _tmp29 = einsum('clid,badjkl->abcijk', g_abab[v, o, o, v], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp29
    _tmp30 = einsum('abde,decijk->abcijk', g_aaaa[v, v, v, v], t3_aaaaaa, optimize=True)
    t3_res += 0.5 * _tmp30
    t3_res -= 0.5 * _tmp30.transpose(0, 2, 1, 3, 4, 5)
    _tmp31 = einsum('bcde,deaijk->abcijk', g_aaaa[v, v, v, v], t3_aaaaaa, optimize=True)
    t3_res += 0.5 * _tmp31
    _tmp32 = einsum('mljk,al,bcim->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp32
    t3_res -= 1 * _tmp32.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp32.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp32.transpose(1, 0, 2, 4, 3, 5)
    _tmp33 = einsum('mljk,abim,cl->abcijk', g_aaaa[o, o, o, o], t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp33
    t3_res -= 1 * _tmp33.transpose(0, 1, 2, 4, 3, 5)
    _tmp34 = einsum('mlij,al,bckm->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp34
    t3_res -= 1 * _tmp34.transpose(1, 0, 2, 3, 4, 5)
    _tmp35 = einsum('mlij,abkm,cl->abcijk', g_aaaa[o, o, o, o], t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp35
    _tmp36 = einsum('ladk,bcil,dj->abcijk', g_aaaa[o, v, v, o], t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp36
    t3_res += 1 * _tmp36.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp36.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp36.transpose(1, 0, 2, 4, 3, 5)
    _tmp37 = einsum('ladk,bl,dcij->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp37
    t3_res += 1 * _tmp37.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 1 * _tmp37.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp37.transpose(0, 2, 1, 3, 5, 4)
    _tmp38 = einsum('ladj,bcil,dk->abcijk', g_aaaa[o, v, v, o], t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp38
    t3_res -= 1 * _tmp38.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp38.transpose(0, 1, 2, 5, 4, 3)
    t3_res += 1 * _tmp38.transpose(1, 0, 2, 5, 4, 3)
    _tmp39 = einsum('ladi,bcjl,dk->abcijk', g_aaaa[o, v, v, o], t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp39
    t3_res += 1 * _tmp39.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp39.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp39.transpose(1, 0, 2, 3, 5, 4)
    _tmp40 = einsum('ladi,bl,dcjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp40
    t3_res += 1 * _tmp40.transpose(0, 2, 1, 3, 4, 5)
    _tmp41 = einsum('lbdk,al,dcij->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp41
    t3_res -= 1 * _tmp41.transpose(2, 1, 0, 3, 4, 5)
    t3_res -= 1 * _tmp41.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp41.transpose(2, 1, 0, 3, 5, 4)
    _tmp42 = einsum('lbdi,al,dcjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp42
    t3_res -= 1 * _tmp42.transpose(2, 1, 0, 3, 4, 5)
    _tmp43 = einsum('lcdk,abil,dj->abcijk', g_aaaa[o, v, v, o], t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp43
    t3_res += 1 * _tmp43.transpose(0, 1, 2, 4, 3, 5)
    _tmp44 = einsum('lcdk,al,dbij->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp44
    t3_res += 1 * _tmp44.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp44.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp44.transpose(1, 0, 2, 3, 5, 4)
    _tmp45 = einsum('lcdj,abil,dk->abcijk', g_aaaa[o, v, v, o], t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp45
    t3_res -= 1 * _tmp45.transpose(0, 1, 2, 5, 4, 3)
    _tmp46 = einsum('lcdi,abjl,dk->abcijk', g_aaaa[o, v, v, o], t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp46
    t3_res += 1 * _tmp46.transpose(0, 1, 2, 3, 5, 4)
    _tmp47 = einsum('lcdi,al,dbjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp47
    t3_res += 1 * _tmp47.transpose(1, 0, 2, 3, 4, 5)
    _tmp48 = einsum('abde,ecij,dk->abcijk', g_aaaa[v, v, v, v], t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp48
    t3_res -= 1 * _tmp48.transpose(0, 2, 1, 3, 4, 5)
    t3_res -= 1 * _tmp48.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp48.transpose(0, 2, 1, 3, 5, 4)
    _tmp49 = einsum('abde,ecjk,di->abcijk', g_aaaa[v, v, v, v], t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp49
    t3_res -= 1 * _tmp49.transpose(0, 2, 1, 3, 4, 5)
    _tmp50 = einsum('bcde,eaij,dk->abcijk', g_aaaa[v, v, v, v], t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp50
    t3_res -= 1 * _tmp50.transpose(0, 1, 2, 3, 5, 4)
    _tmp51 = einsum('bcde,eajk,di->abcijk', g_aaaa[v, v, v, v], t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp51
    _tmp52 = einsum('mldk,abcijm,dl->abcijk', g_aaaa[o, o, v, o], t3_aaaaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp52
    t3_res -= 1 * _tmp52.transpose(0, 1, 2, 3, 5, 4)
    _tmp53 = einsum('mlkd,abcijm,dl->abcijk', g_abab[o, o, o, v], t3_aaaaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp53
    t3_res += 1 * _tmp53.transpose(0, 1, 2, 3, 5, 4)
    _tmp54 = einsum('mldk,abciml,dj->abcijk', g_aaaa[o, o, v, o], t3_aaaaaa, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp54
    t3_res -= 0.5 * _tmp54.transpose(0, 1, 2, 4, 3, 5)
    _tmp55 = einsum('mldk,al,dbcijm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=True)
    t3_res -= 1 * _tmp55
    t3_res += 1 * _tmp55.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp55.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp55.transpose(1, 0, 2, 3, 5, 4)
    _tmp56 = einsum('lmkd,al,cbdijm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp56
    t3_res -= 1 * _tmp56.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp56.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp56.transpose(1, 0, 2, 3, 5, 4)
    _tmp57 = einsum('mldk,dabijm,cl->abcijk', g_aaaa[o, o, v, o], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp57
    t3_res += 1 * _tmp57.transpose(0, 1, 2, 3, 5, 4)
    _tmp58 = einsum('lmkd,badijm,cl->abcijk', g_abab[o, o, o, v], t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp58
    t3_res -= 1 * _tmp58.transpose(0, 1, 2, 3, 5, 4)
    _tmp59 = einsum('mldj,abciml,dk->abcijk', g_aaaa[o, o, v, o], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp59
    t3_res += 0.5 * _tmp59.transpose(0, 1, 2, 5, 4, 3)
    _tmp60 = einsum('mldi,abcjkm,dl->abcijk', g_aaaa[o, o, v, o], t3_aaaaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp60
    _tmp61 = einsum('mlid,abcjkm,dl->abcijk', g_abab[o, o, o, v], t3_aaaaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp61
    _tmp62 = einsum('mldi,abcjml,dk->abcijk', g_aaaa[o, o, v, o], t3_aaaaaa, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp62
    t3_res -= 0.5 * _tmp62.transpose(0, 1, 2, 3, 5, 4)
    _tmp63 = einsum('mldi,al,dbcjkm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=True)
    t3_res -= 1 * _tmp63
    t3_res += 1 * _tmp63.transpose(1, 0, 2, 3, 4, 5)
    _tmp64 = einsum('lmid,al,cbdjkm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp64
    t3_res -= 1 * _tmp64.transpose(1, 0, 2, 3, 4, 5)
    _tmp65 = einsum('mldi,dabjkm,cl->abcijk', g_aaaa[o, o, v, o], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp65
    _tmp66 = einsum('lmid,badjkm,cl->abcijk', g_abab[o, o, o, v], t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp66
    _tmp67 = einsum('lade,ebcijk,dl->abcijk', g_aaaa[o, v, v, v], t3_aaaaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp67
    t3_res -= 1 * _tmp67.transpose(1, 0, 2, 3, 4, 5)
    _tmp68 = einsum('aled,ebcijk,dl->abcijk', g_abab[v, o, v, v], t3_aaaaaa, t1_bb, optimize=True)
    t3_res += 1 * _tmp68
    t3_res -= 1 * _tmp68.transpose(1, 0, 2, 3, 4, 5)
    _tmp69 = einsum('lade,ebcijl,dk->abcijk', g_aaaa[o, v, v, v], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp69
    t3_res += 1 * _tmp69.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp69.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp69.transpose(1, 0, 2, 3, 5, 4)
    _tmp70 = einsum('alde,cbeijl,dk->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp70
    t3_res += 1 * _tmp70.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp70.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp70.transpose(1, 0, 2, 3, 5, 4)
    _tmp71 = einsum('lade,ebcjkl,di->abcijk', g_aaaa[o, v, v, v], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp71
    t3_res += 1 * _tmp71.transpose(1, 0, 2, 3, 4, 5)
    _tmp72 = einsum('alde,cbejkl,di->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp72
    t3_res += 1 * _tmp72.transpose(1, 0, 2, 3, 4, 5)
    _tmp73 = einsum('lade,bl,decijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=True)
    t3_res += 0.5 * _tmp73
    t3_res -= 0.5 * _tmp73.transpose(0, 2, 1, 3, 4, 5)
    _tmp74 = einsum('lbde,al,decijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=True)
    t3_res -= 0.5 * _tmp74
    t3_res += 0.5 * _tmp74.transpose(2, 1, 0, 3, 4, 5)
    _tmp75 = einsum('lcde,eabijk,dl->abcijk', g_aaaa[o, v, v, v], t3_aaaaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp75
    _tmp76 = einsum('cled,eabijk,dl->abcijk', g_abab[v, o, v, v], t3_aaaaaa, t1_bb, optimize=True)
    t3_res += 1 * _tmp76
    _tmp77 = einsum('lcde,eabijl,dk->abcijk', g_aaaa[o, v, v, v], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp77
    t3_res += 1 * _tmp77.transpose(0, 1, 2, 3, 5, 4)
    _tmp78 = einsum('clde,baeijl,dk->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp78
    t3_res += 1 * _tmp78.transpose(0, 1, 2, 3, 5, 4)
    _tmp79 = einsum('lcde,eabjkl,di->abcijk', g_aaaa[o, v, v, v], t3_aaaaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp79
    _tmp80 = einsum('clde,baejkl,di->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp80
    _tmp81 = einsum('lcde,al,debijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=True)
    t3_res += 0.5 * _tmp81
    t3_res -= 0.5 * _tmp81.transpose(1, 0, 2, 3, 4, 5)
    _tmp82 = einsum('mlde,abcijm,dekl->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp82
    t3_res += 0.5 * _tmp82.transpose(0, 1, 2, 3, 5, 4)
    _tmp83 = einsum('mlde,abcijm,dekl->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp83
    t3_res += 0.5 * _tmp83.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp83
    t3_res += 0.5 * _tmp83.transpose(0, 1, 2, 3, 5, 4)
    _tmp84 = einsum('mlde,abcjkm,deil->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp84
    _tmp85 = einsum('mlde,abcjkm,deil->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp85
    t3_res -= 0.5 * _tmp85
    _tmp86 = einsum('mlde,abciml,dejk->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res += 0.25 * _tmp86
    t3_res -= 0.25 * _tmp86.transpose(0, 1, 2, 4, 3, 5)
    _tmp87 = einsum('mlde,abckml,deij->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res += 0.25 * _tmp87
    _tmp88 = einsum('mlde,daml,ebcijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=True)
    t3_res -= 0.5 * _tmp88
    t3_res += 0.5 * _tmp88.transpose(1, 0, 2, 3, 4, 5)
    _tmp89 = einsum('mled,adml,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=True)
    t3_res -= 0.5 * _tmp89
    t3_res += 0.5 * _tmp89.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp89
    t3_res += 0.5 * _tmp89.transpose(1, 0, 2, 3, 4, 5)
    _tmp90 = einsum('mlde,dakl,ebcijm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp90
    t3_res -= 1 * _tmp90.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp90.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp90.transpose(1, 0, 2, 3, 5, 4)
    _tmp91 = einsum('lmde,dakl,cbeijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp91
    t3_res -= 1 * _tmp91.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp91.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp91.transpose(1, 0, 2, 3, 5, 4)
    _tmp92 = einsum('mled,adkl,ebcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp92
    t3_res -= 1 * _tmp92.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp92.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp92.transpose(1, 0, 2, 3, 5, 4)
    _tmp93 = einsum('mlde,adkl,cbeijm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp93
    t3_res -= 1 * _tmp93.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp93.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp93.transpose(1, 0, 2, 3, 5, 4)
    _tmp94 = einsum('mlde,dail,ebcjkm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp94
    t3_res -= 1 * _tmp94.transpose(1, 0, 2, 3, 4, 5)
    _tmp95 = einsum('lmde,dail,cbejkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp95
    t3_res -= 1 * _tmp95.transpose(1, 0, 2, 3, 4, 5)
    _tmp96 = einsum('mled,adil,ebcjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp96
    t3_res -= 1 * _tmp96.transpose(1, 0, 2, 3, 4, 5)
    _tmp97 = einsum('mlde,adil,cbejkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp97
    t3_res -= 1 * _tmp97.transpose(1, 0, 2, 3, 4, 5)
    _tmp98 = einsum('mlde,dajk,ebciml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=True)
    t3_res -= 0.5 * _tmp98
    t3_res += 0.5 * _tmp98.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp98.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp98.transpose(1, 0, 2, 4, 3, 5)
    _tmp99 = einsum('mlde,dajk,cbeiml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp99
    t3_res -= 0.5 * _tmp99.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp99.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp99.transpose(1, 0, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp99
    t3_res -= 0.5 * _tmp99.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp99.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp99.transpose(1, 0, 2, 4, 3, 5)
    _tmp100 = einsum('mlde,daij,ebckml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=True)
    t3_res -= 0.5 * _tmp100
    t3_res += 0.5 * _tmp100.transpose(1, 0, 2, 3, 4, 5)
    _tmp101 = einsum('mlde,daij,cbekml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp101
    t3_res -= 0.5 * _tmp101.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp101
    t3_res -= 0.5 * _tmp101.transpose(1, 0, 2, 3, 4, 5)
    _tmp102 = einsum('mlde,eabijk,dcml->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp102
    _tmp103 = einsum('mled,eabijk,cdml->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp103
    t3_res -= 0.5 * _tmp103
    _tmp104 = einsum('mlde,eabijm,dckl->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp104
    t3_res -= 1 * _tmp104.transpose(0, 1, 2, 3, 5, 4)
    _tmp105 = einsum('mled,eabijm,cdkl->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp105
    t3_res -= 1 * _tmp105.transpose(0, 1, 2, 3, 5, 4)
    _tmp106 = einsum('lmde,baeijm,dckl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp106
    t3_res -= 1 * _tmp106.transpose(0, 1, 2, 3, 5, 4)
    _tmp107 = einsum('mlde,baeijm,cdkl->abcijk', g_bbbb[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 1 * _tmp107
    t3_res -= 1 * _tmp107.transpose(0, 1, 2, 3, 5, 4)
    _tmp108 = einsum('mlde,eabjkm,dcil->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp108
    _tmp109 = einsum('mled,eabjkm,cdil->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp109
    _tmp110 = einsum('lmde,baejkm,dcil->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp110
    _tmp111 = einsum('mlde,baejkm,cdil->abcijk', g_bbbb[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 1 * _tmp111
    _tmp112 = einsum('mlde,eabiml,dcjk->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp112
    t3_res += 0.5 * _tmp112.transpose(0, 1, 2, 4, 3, 5)
    _tmp113 = einsum('mlde,baeiml,dcjk->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp113
    t3_res -= 0.5 * _tmp113.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp113
    t3_res -= 0.5 * _tmp113.transpose(0, 1, 2, 4, 3, 5)
    _tmp114 = einsum('mlde,eabkml,dcij->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp114
    _tmp115 = einsum('mlde,baekml,dcij->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp115
    t3_res += 0.5 * _tmp115
    _tmp116 = einsum('mlde,abml,decijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=True)
    t3_res += 0.25 * _tmp116
    t3_res -= 0.25 * _tmp116.transpose(0, 2, 1, 3, 4, 5)
    _tmp117 = einsum('mlde,abkl,decijm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=True)
    t3_res -= 0.5 * _tmp117
    t3_res += 0.5 * _tmp117.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.5 * _tmp117.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp117.transpose(0, 2, 1, 3, 5, 4)
    _tmp118 = einsum('lmde,abkl,dceijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp118
    t3_res += 0.5 * _tmp118.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.5 * _tmp118.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp118.transpose(0, 2, 1, 3, 5, 4)
    _tmp119 = einsum('lmed,abkl,cedijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp119
    t3_res -= 0.5 * _tmp119.transpose(0, 2, 1, 3, 4, 5)
    t3_res -= 0.5 * _tmp119.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 0.5 * _tmp119.transpose(0, 2, 1, 3, 5, 4)
    _tmp120 = einsum('mlde,abil,decjkm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=True)
    t3_res -= 0.5 * _tmp120
    t3_res += 0.5 * _tmp120.transpose(0, 2, 1, 3, 4, 5)
    _tmp121 = einsum('lmde,abil,dcejkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp121
    t3_res += 0.5 * _tmp121.transpose(0, 2, 1, 3, 4, 5)
    _tmp122 = einsum('lmed,abil,cedjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp122
    t3_res -= 0.5 * _tmp122.transpose(0, 2, 1, 3, 4, 5)
    _tmp123 = einsum('mlde,deaijk,bcml->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res += 0.25 * _tmp123
    _tmp124 = einsum('mlde,deaijm,bckl->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp124
    t3_res += 0.5 * _tmp124.transpose(0, 1, 2, 3, 5, 4)
    _tmp125 = einsum('lmde,daeijm,bckl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp125
    t3_res += 0.5 * _tmp125.transpose(0, 1, 2, 3, 5, 4)
    _tmp126 = einsum('lmed,aedijm,bckl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp126
    t3_res -= 0.5 * _tmp126.transpose(0, 1, 2, 3, 5, 4)
    _tmp127 = einsum('mlde,deajkm,bcil->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp127
    _tmp128 = einsum('lmde,daejkm,bcil->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp128
    _tmp129 = einsum('lmed,aedjkm,bcil->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp129
    _tmp130 = einsum('mldk,dajl,bcim->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp130
    t3_res -= 1 * _tmp130.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp130.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp130.transpose(1, 0, 2, 4, 3, 5)
    _tmp131 = einsum('mlkd,adjl,bcim->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp131
    t3_res -= 1 * _tmp131.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp131.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp131.transpose(1, 0, 2, 4, 3, 5)
    _tmp132 = einsum('mldk,daij,bcml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp132
    t3_res += 0.5 * _tmp132.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp132.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp132.transpose(1, 0, 2, 3, 5, 4)
    _tmp133 = einsum('mldk,abim,dcjl->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp133
    t3_res -= 1 * _tmp133.transpose(0, 1, 2, 4, 3, 5)
    _tmp134 = einsum('mlkd,abim,cdjl->abcijk', g_abab[o, o, o, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp134
    t3_res -= 1 * _tmp134.transpose(0, 1, 2, 4, 3, 5)
    _tmp135 = einsum('mldk,abml,dcij->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp135
    t3_res += 0.5 * _tmp135.transpose(0, 1, 2, 3, 5, 4)
    _tmp136 = einsum('mldj,dakl,bcim->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp136
    t3_res += 1 * _tmp136.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp136.transpose(0, 1, 2, 5, 4, 3)
    t3_res -= 1 * _tmp136.transpose(1, 0, 2, 5, 4, 3)
    _tmp137 = einsum('mljd,adkl,bcim->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp137
    t3_res += 1 * _tmp137.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp137.transpose(0, 1, 2, 5, 4, 3)
    t3_res -= 1 * _tmp137.transpose(1, 0, 2, 5, 4, 3)
    _tmp138 = einsum('mldj,abim,dckl->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp138
    t3_res += 1 * _tmp138.transpose(0, 1, 2, 5, 4, 3)
    _tmp139 = einsum('mljd,abim,cdkl->abcijk', g_abab[o, o, o, v], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp139
    t3_res += 1 * _tmp139.transpose(0, 1, 2, 5, 4, 3)
    _tmp140 = einsum('mldi,dakl,bcjm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp140
    t3_res -= 1 * _tmp140.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp140.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp140.transpose(1, 0, 2, 3, 5, 4)
    _tmp141 = einsum('mlid,adkl,bcjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp141
    t3_res -= 1 * _tmp141.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp141.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp141.transpose(1, 0, 2, 3, 5, 4)
    _tmp142 = einsum('mldi,dajk,bcml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp142
    t3_res += 0.5 * _tmp142.transpose(1, 0, 2, 3, 4, 5)
    _tmp143 = einsum('mldi,abjm,dckl->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp143
    t3_res -= 1 * _tmp143.transpose(0, 1, 2, 3, 5, 4)
    _tmp144 = einsum('mlid,abjm,cdkl->abcijk', g_abab[o, o, o, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp144
    t3_res -= 1 * _tmp144.transpose(0, 1, 2, 3, 5, 4)
    _tmp145 = einsum('mldi,abml,dcjk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp145
    _tmp146 = einsum('lade,bcil,dejk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp146
    t3_res += 0.5 * _tmp146.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp146.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp146.transpose(1, 0, 2, 4, 3, 5)
    _tmp147 = einsum('lade,bckl,deij->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp147
    t3_res += 0.5 * _tmp147.transpose(1, 0, 2, 3, 4, 5)
    _tmp148 = einsum('lade,dbkl,ecij->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp148
    t3_res -= 1 * _tmp148.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp148.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp148.transpose(1, 0, 2, 3, 5, 4)
    _tmp149 = einsum('aled,bdkl,ecij->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp149
    t3_res += 1 * _tmp149.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp149.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp149.transpose(1, 0, 2, 3, 5, 4)
    _tmp150 = einsum('lade,dbil,ecjk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp150
    t3_res -= 1 * _tmp150.transpose(1, 0, 2, 3, 4, 5)
    _tmp151 = einsum('aled,bdil,ecjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp151
    t3_res += 1 * _tmp151.transpose(1, 0, 2, 3, 4, 5)
    _tmp152 = einsum('lade,dbjk,ecil->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp152
    t3_res -= 1 * _tmp152.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp152.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp152.transpose(1, 0, 2, 4, 3, 5)
    _tmp153 = einsum('alde,dbjk,ceil->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp153
    t3_res -= 1 * _tmp153.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp153.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp153.transpose(1, 0, 2, 4, 3, 5)
    _tmp154 = einsum('lade,dbij,eckl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp154
    t3_res -= 1 * _tmp154.transpose(1, 0, 2, 3, 4, 5)
    _tmp155 = einsum('alde,dbij,cekl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp155
    t3_res -= 1 * _tmp155.transpose(1, 0, 2, 3, 4, 5)
    _tmp156 = einsum('lcde,abil,dejk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp156
    t3_res += 0.5 * _tmp156.transpose(0, 1, 2, 4, 3, 5)
    _tmp157 = einsum('lcde,abkl,deij->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp157
    _tmp158 = einsum('lcde,dakl,ebij->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp158
    t3_res -= 1 * _tmp158.transpose(0, 1, 2, 3, 5, 4)
    _tmp159 = einsum('cled,adkl,ebij->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp159
    t3_res += 1 * _tmp159.transpose(0, 1, 2, 3, 5, 4)
    _tmp160 = einsum('lcde,dail,ebjk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp160
    _tmp161 = einsum('cled,adil,ebjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp161
    _tmp162 = einsum('lcde,dajk,ebil->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp162
    t3_res -= 1 * _tmp162.transpose(0, 1, 2, 4, 3, 5)
    _tmp163 = einsum('clde,dajk,beil->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp163
    t3_res -= 1 * _tmp163.transpose(0, 1, 2, 4, 3, 5)
    _tmp164 = einsum('lcde,daij,ebkl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp164
    _tmp165 = einsum('clde,daij,bekl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp165
    _tmp166 = einsum('mlde,eajk,bcim,dl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp166
    t3_res -= 1 * _tmp166.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp166.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp166.transpose(1, 0, 2, 4, 3, 5)
    _tmp167 = einsum('mled,eajk,bcim,dl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_aaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp167
    t3_res += 1 * _tmp167.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp167.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp167.transpose(1, 0, 2, 4, 3, 5)
    _tmp168 = einsum('mlde,eaij,bckm,dl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp168
    t3_res -= 1 * _tmp168.transpose(1, 0, 2, 3, 4, 5)
    _tmp169 = einsum('mled,eaij,bckm,dl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_aaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp169
    t3_res += 1 * _tmp169.transpose(1, 0, 2, 3, 4, 5)
    _tmp170 = einsum('mlde,abim,ecjk,dl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp170
    t3_res -= 1 * _tmp170.transpose(0, 1, 2, 4, 3, 5)
    _tmp171 = einsum('mled,abim,ecjk,dl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_aaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp171
    t3_res += 1 * _tmp171.transpose(0, 1, 2, 4, 3, 5)
    _tmp172 = einsum('mlde,abkm,ecij,dl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp172
    _tmp173 = einsum('mled,abkm,ecij,dl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_aaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp173
    _tmp174 = einsum('mlde,eajl,bcim,dk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp174
    t3_res += 1 * _tmp174.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp174.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp174.transpose(1, 0, 2, 4, 3, 5)
    _tmp175 = einsum('mlde,aejl,bcim,dk->abcijk', g_abab[o, o, v, v], t2_abab, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp175
    t3_res -= 1 * _tmp175.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp175.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp175.transpose(1, 0, 2, 4, 3, 5)
    _tmp176 = einsum('mlde,eaij,bcml,dk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp176
    t3_res -= 0.5 * _tmp176.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp176.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 0.5 * _tmp176.transpose(1, 0, 2, 3, 5, 4)
    _tmp177 = einsum('mlde,abim,ecjl,dk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp177
    t3_res += 1 * _tmp177.transpose(0, 1, 2, 4, 3, 5)
    _tmp178 = einsum('mlde,abim,cejl,dk->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp178
    t3_res -= 1 * _tmp178.transpose(0, 1, 2, 4, 3, 5)
    _tmp179 = einsum('mlde,abml,ecij,dk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp179
    t3_res -= 0.5 * _tmp179.transpose(0, 1, 2, 3, 5, 4)
    _tmp180 = einsum('mlde,eakl,bcim,dj->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp180
    t3_res -= 1 * _tmp180.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp180.transpose(0, 1, 2, 5, 4, 3)
    t3_res += 1 * _tmp180.transpose(1, 0, 2, 5, 4, 3)
    _tmp181 = einsum('mlde,aekl,bcim,dj->abcijk', g_abab[o, o, v, v], t2_abab, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp181
    t3_res += 1 * _tmp181.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp181.transpose(0, 1, 2, 5, 4, 3)
    t3_res -= 1 * _tmp181.transpose(1, 0, 2, 5, 4, 3)
    _tmp182 = einsum('mlde,abim,eckl,dj->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp182
    t3_res -= 1 * _tmp182.transpose(0, 1, 2, 5, 4, 3)
    _tmp183 = einsum('mlde,abim,cekl,dj->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp183
    t3_res += 1 * _tmp183.transpose(0, 1, 2, 5, 4, 3)
    _tmp184 = einsum('mlde,eakl,bcjm,di->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp184
    t3_res += 1 * _tmp184.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp184.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp184.transpose(1, 0, 2, 3, 5, 4)
    _tmp185 = einsum('mlde,aekl,bcjm,di->abcijk', g_abab[o, o, v, v], t2_abab, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp185
    t3_res -= 1 * _tmp185.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp185.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp185.transpose(1, 0, 2, 3, 5, 4)
    _tmp186 = einsum('mlde,eajk,bcml,di->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp186
    t3_res -= 0.5 * _tmp186.transpose(1, 0, 2, 3, 4, 5)
    _tmp187 = einsum('mlde,abjm,eckl,di->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp187
    t3_res += 1 * _tmp187.transpose(0, 1, 2, 3, 5, 4)
    _tmp188 = einsum('mlde,abjm,cekl,di->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp188
    t3_res -= 1 * _tmp188.transpose(0, 1, 2, 3, 5, 4)
    _tmp189 = einsum('mlde,abml,ecjk,di->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp189
    _tmp190 = einsum('mlde,al,bcim,dejk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp190
    t3_res -= 0.5 * _tmp190.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp190.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp190.transpose(1, 0, 2, 4, 3, 5)
    _tmp191 = einsum('mlde,al,bckm,deij->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp191
    t3_res -= 0.5 * _tmp191.transpose(1, 0, 2, 3, 4, 5)
    _tmp192 = einsum('mlde,al,dbkm,ecij->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp192
    t3_res += 1 * _tmp192.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp192.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp192.transpose(1, 0, 2, 3, 5, 4)
    _tmp193 = einsum('lmed,al,bdkm,ecij->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp193
    t3_res -= 1 * _tmp193.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp193.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp193.transpose(1, 0, 2, 3, 5, 4)
    _tmp194 = einsum('mlde,al,dbim,ecjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp194
    t3_res += 1 * _tmp194.transpose(1, 0, 2, 3, 4, 5)
    _tmp195 = einsum('lmed,al,bdim,ecjk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp195
    t3_res -= 1 * _tmp195.transpose(1, 0, 2, 3, 4, 5)
    _tmp196 = einsum('mlde,al,dbjk,ecim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp196
    t3_res += 1 * _tmp196.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp196.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp196.transpose(1, 0, 2, 4, 3, 5)
    _tmp197 = einsum('lmde,al,dbjk,ceim->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp197
    t3_res += 1 * _tmp197.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp197.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp197.transpose(1, 0, 2, 4, 3, 5)
    _tmp198 = einsum('mlde,al,dbij,eckm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp198
    t3_res += 1 * _tmp198.transpose(1, 0, 2, 3, 4, 5)
    _tmp199 = einsum('lmde,al,dbij,cekm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp199
    t3_res += 1 * _tmp199.transpose(1, 0, 2, 3, 4, 5)
    _tmp200 = einsum('mlde,abim,cl,dejk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp200
    t3_res -= 0.5 * _tmp200.transpose(0, 1, 2, 4, 3, 5)
    _tmp201 = einsum('mlde,abkm,cl,deij->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp201
    _tmp202 = einsum('mlde,dakm,ebij,cl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp202
    t3_res += 1 * _tmp202.transpose(0, 1, 2, 3, 5, 4)
    _tmp203 = einsum('lmed,adkm,ebij,cl->abcijk', g_abab[o, o, v, v], t2_abab, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp203
    t3_res -= 1 * _tmp203.transpose(0, 1, 2, 3, 5, 4)
    _tmp204 = einsum('mlde,daim,ebjk,cl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp204
    _tmp205 = einsum('lmed,adim,ebjk,cl->abcijk', g_abab[o, o, v, v], t2_abab, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp205
    _tmp206 = einsum('mlde,dajk,ebim,cl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp206
    t3_res += 1 * _tmp206.transpose(0, 1, 2, 4, 3, 5)
    _tmp207 = einsum('lmde,dajk,beim,cl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp207
    t3_res += 1 * _tmp207.transpose(0, 1, 2, 4, 3, 5)
    _tmp208 = einsum('mlde,daij,ebkm,cl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp208
    _tmp209 = einsum('lmde,daij,bekm,cl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp209
    _tmp210 = einsum('mldk,al,bcim,dj->abcijk', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp210
    t3_res -= 1 * _tmp210.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp210.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp210.transpose(1, 0, 2, 4, 3, 5)
    _tmp211 = einsum('mldk,abim,cl,dj->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp211
    t3_res -= 1 * _tmp211.transpose(0, 1, 2, 4, 3, 5)
    _tmp212 = einsum('mldk,al,bm,dcij->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp212
    t3_res -= 1 * _tmp212.transpose(0, 2, 1, 3, 4, 5)
    t3_res -= 1 * _tmp212.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp212.transpose(0, 2, 1, 3, 5, 4)
    _tmp213 = einsum('mldk,daij,bl,cm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp213
    t3_res -= 1 * _tmp213.transpose(0, 1, 2, 3, 5, 4)
    _tmp214 = einsum('mldj,al,bcim,dk->abcijk', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp214
    t3_res += 1 * _tmp214.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp214.transpose(0, 1, 2, 5, 4, 3)
    t3_res -= 1 * _tmp214.transpose(1, 0, 2, 5, 4, 3)
    _tmp215 = einsum('mldj,abim,cl,dk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp215
    t3_res += 1 * _tmp215.transpose(0, 1, 2, 5, 4, 3)
    _tmp216 = einsum('mldi,al,bcjm,dk->abcijk', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp216
    t3_res -= 1 * _tmp216.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp216.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp216.transpose(1, 0, 2, 3, 5, 4)
    _tmp217 = einsum('mldi,abjm,cl,dk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp217
    t3_res -= 1 * _tmp217.transpose(0, 1, 2, 3, 5, 4)
    _tmp218 = einsum('mldi,al,bm,dcjk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp218
    t3_res -= 1 * _tmp218.transpose(0, 2, 1, 3, 4, 5)
    _tmp219 = einsum('mldi,dajk,bl,cm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp219
    _tmp220 = einsum('lade,bcil,dk,ej->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp220
    t3_res -= 1 * _tmp220.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp220.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp220.transpose(1, 0, 2, 4, 3, 5)
    _tmp221 = einsum('lade,bl,ecij,dk->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp221
    t3_res -= 1 * _tmp221.transpose(0, 2, 1, 3, 4, 5)
    t3_res -= 1 * _tmp221.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp221.transpose(0, 2, 1, 3, 5, 4)
    _tmp222 = einsum('lade,bckl,dj,ei->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp222
    t3_res -= 1 * _tmp222.transpose(1, 0, 2, 3, 4, 5)
    _tmp223 = einsum('lade,bl,ecjk,di->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp223
    t3_res -= 1 * _tmp223.transpose(0, 2, 1, 3, 4, 5)
    _tmp224 = einsum('lbde,al,ecij,dk->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp224
    t3_res += 1 * _tmp224.transpose(2, 1, 0, 3, 4, 5)
    t3_res += 1 * _tmp224.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp224.transpose(2, 1, 0, 3, 5, 4)
    _tmp225 = einsum('lbde,al,ecjk,di->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp225
    t3_res += 1 * _tmp225.transpose(2, 1, 0, 3, 4, 5)
    _tmp226 = einsum('lcde,abil,dk,ej->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp226
    t3_res -= 1 * _tmp226.transpose(0, 1, 2, 4, 3, 5)
    _tmp227 = einsum('lcde,al,ebij,dk->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp227
    t3_res -= 1 * _tmp227.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp227.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp227.transpose(1, 0, 2, 3, 5, 4)
    _tmp228 = einsum('lcde,abkl,dj,ei->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp228
    _tmp229 = einsum('lcde,al,ebjk,di->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp229
    t3_res -= 1 * _tmp229.transpose(1, 0, 2, 3, 4, 5)
    _tmp230 = einsum('mlde,abcijm,dl,ek->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp230
    t3_res -= 1 * _tmp230.transpose(0, 1, 2, 3, 5, 4)
    _tmp231 = einsum('mled,abcijm,dl,ek->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp231
    t3_res += 1 * _tmp231.transpose(0, 1, 2, 3, 5, 4)
    _tmp232 = einsum('mlde,abcjkm,dl,ei->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp232
    _tmp233 = einsum('mled,abcjkm,dl,ei->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp233
    _tmp234 = einsum('mlde,am,ebcijk,dl->abcijk', g_aaaa[o, o, v, v], t1_aa, t3_aaaaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp234
    t3_res -= 1 * _tmp234.transpose(1, 0, 2, 3, 4, 5)
    _tmp235 = einsum('mled,am,ebcijk,dl->abcijk', g_abab[o, o, v, v], t1_aa, t3_aaaaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp235
    t3_res += 1 * _tmp235.transpose(1, 0, 2, 3, 4, 5)
    _tmp236 = einsum('mlde,eabijk,cm,dl->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp236
    _tmp237 = einsum('mled,eabijk,cm,dl->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t1_aa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp237
    _tmp238 = einsum('mlde,abciml,dk,ej->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp238
    t3_res += 0.5 * _tmp238.transpose(0, 1, 2, 4, 3, 5)
    _tmp239 = einsum('mlde,al,ebcijm,dk->abcijk', g_aaaa[o, o, v, v], t1_aa, t3_aaaaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp239
    t3_res -= 1 * _tmp239.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp239.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp239.transpose(1, 0, 2, 3, 5, 4)
    _tmp240 = einsum('lmde,al,cbeijm,dk->abcijk', g_abab[o, o, v, v], t1_aa, t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp240
    t3_res -= 1 * _tmp240.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp240.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp240.transpose(1, 0, 2, 3, 5, 4)
    _tmp241 = einsum('mlde,eabijm,cl,dk->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp241
    t3_res -= 1 * _tmp241.transpose(0, 1, 2, 3, 5, 4)
    _tmp242 = einsum('lmde,baeijm,cl,dk->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp242
    t3_res -= 1 * _tmp242.transpose(0, 1, 2, 3, 5, 4)
    _tmp243 = einsum('mlde,abckml,dj,ei->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp243
    _tmp244 = einsum('mlde,al,ebcjkm,di->abcijk', g_aaaa[o, o, v, v], t1_aa, t3_aaaaaa, t1_aa, optimize=True)
    t3_res += 1 * _tmp244
    t3_res -= 1 * _tmp244.transpose(1, 0, 2, 3, 4, 5)
    _tmp245 = einsum('lmde,al,cbejkm,di->abcijk', g_abab[o, o, v, v], t1_aa, t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp245
    t3_res -= 1 * _tmp245.transpose(1, 0, 2, 3, 4, 5)
    _tmp246 = einsum('mlde,eabjkm,cl,di->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp246
    _tmp247 = einsum('lmde,baejkm,cl,di->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp247
    _tmp248 = einsum('mlde,al,bm,decijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=True)
    t3_res -= 0.5 * _tmp248
    t3_res += 0.5 * _tmp248.transpose(0, 2, 1, 3, 4, 5)
    _tmp249 = einsum('mlde,deaijk,bl,cm->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t1_aa, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp249
    _tmp250 = einsum('mlde,al,bcim,dk,ej->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp250
    t3_res += 1 * _tmp250.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp250.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp250.transpose(1, 0, 2, 4, 3, 5)
    _tmp251 = einsum('mlde,abim,cl,dk,ej->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp251
    t3_res += 1 * _tmp251.transpose(0, 1, 2, 4, 3, 5)
    _tmp252 = einsum('mlde,al,bm,ecij,dk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp252
    t3_res += 1 * _tmp252.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 1 * _tmp252.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp252.transpose(0, 2, 1, 3, 5, 4)
    _tmp253 = einsum('mlde,eaij,bl,cm,dk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp253
    t3_res += 1 * _tmp253.transpose(0, 1, 2, 3, 5, 4)
    _tmp254 = einsum('mlde,al,bckm,dj,ei->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp254
    t3_res += 1 * _tmp254.transpose(1, 0, 2, 3, 4, 5)
    _tmp255 = einsum('mlde,abkm,cl,dj,ei->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp255
    _tmp256 = einsum('mlde,al,bm,ecjk,di->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp256
    t3_res += 1 * _tmp256.transpose(0, 2, 1, 3, 4, 5)
    _tmp257 = einsum('mlde,eajk,bl,cm,di->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp257
    return t3_res


def t3_aabaab_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v):
    nv, no = t1_aa.shape
    t3_res = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('lk,abcijl->abcijk', f_bb[o, o], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp0
    _tmp1 = einsum('lj,abcilk->abcijk', f_aa[o, o], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp1
    _tmp2 = einsum('li,abcjlk->abcijk', f_aa[o, o], t3_aabaab, optimize=True)
    t3_res += 1 * _tmp2
    _tmp3 = einsum('ad,dbcijk->abcijk', f_aa[v, v], t3_aabaab, optimize=True)
    t3_res += 1 * _tmp3
    t3_res -= 1 * _tmp3.transpose(1, 0, 2, 3, 4, 5)
    _tmp4 = einsum('cd,badijk->abcijk', f_bb[v, v], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp4
    _tmp5 = einsum('ld,abcijl,dk->abcijk', f_bb[o, v], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp5
    _tmp6 = einsum('ld,abcilk,dj->abcijk', f_aa[o, v], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp6
    _tmp7 = einsum('ld,abcjlk,di->abcijk', f_aa[o, v], t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp7
    _tmp8 = einsum('ld,al,dbcijk->abcijk', f_aa[o, v], t1_aa, t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp8
    t3_res += 1 * _tmp8.transpose(1, 0, 2, 3, 4, 5)
    _tmp9 = einsum('ld,badijk,cl->abcijk', f_bb[o, v], t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp9
    _tmp10 = einsum('ld,adjk,bcil->abcijk', f_bb[o, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp10
    t3_res -= 1 * _tmp10.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp10.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp10.transpose(1, 0, 2, 4, 3, 5)
    _tmp11 = einsum('ld,daij,bclk->abcijk', f_aa[o, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp11
    t3_res -= 1 * _tmp11.transpose(1, 0, 2, 3, 4, 5)
    _tmp12 = einsum('ld,abil,dcjk->abcijk', f_aa[o, v], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp12
    t3_res += 1 * _tmp12.transpose(0, 1, 2, 4, 3, 5)
    _tmp13 = einsum('aljk,bcil->abcijk', g_abab[v, o, o, o], t2_abab, optimize=True)
    t3_res += 1 * _tmp13
    t3_res -= 1 * _tmp13.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp13.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp13.transpose(1, 0, 2, 4, 3, 5)
    _tmp14 = einsum('laij,bclk->abcijk', g_aaaa[o, v, o, o], t2_abab, optimize=True)
    t3_res += 1 * _tmp14
    t3_res -= 1 * _tmp14.transpose(1, 0, 2, 3, 4, 5)
    _tmp15 = einsum('lcjk,abil->abcijk', g_abab[o, v, o, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp15
    t3_res += 1 * _tmp15.transpose(0, 1, 2, 4, 3, 5)
    _tmp16 = einsum('acdk,dbij->abcijk', g_abab[v, v, v, o], t2_aaaa, optimize=True)
    t3_res += 1 * _tmp16
    _tmp17 = einsum('abdj,dcik->abcijk', g_aaaa[v, v, v, o], t2_abab, optimize=True)
    t3_res += 1 * _tmp17
    _tmp18 = einsum('acjd,bdik->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    t3_res -= 1 * _tmp18
    _tmp19 = einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_abab, optimize=True)
    t3_res -= 1 * _tmp19
    _tmp20 = einsum('acid,bdjk->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    t3_res += 1 * _tmp20
    _tmp21 = einsum('bcdk,daij->abcijk', g_abab[v, v, v, o], t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp21
    _tmp22 = einsum('bcjd,adik->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    t3_res += 1 * _tmp22
    _tmp23 = einsum('bcid,adjk->abcijk', g_abab[v, v, o, v], t2_abab, optimize=True)
    t3_res -= 1 * _tmp23
    _tmp24 = einsum('mljk,abciml->abcijk', g_abab[o, o, o, o], t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp24
    t3_res -= 0.5 * _tmp24.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp24
    t3_res -= 0.5 * _tmp24.transpose(0, 1, 2, 4, 3, 5)
    _tmp25 = einsum('mlij,abclmk->abcijk', g_aaaa[o, o, o, o], t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp25
    _tmp26 = einsum('aldk,dbcijl->abcijk', g_abab[v, o, v, o], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp26
    t3_res += 1 * _tmp26.transpose(1, 0, 2, 3, 4, 5)
    _tmp27 = einsum('ladj,dbcilk->abcijk', g_aaaa[o, v, v, o], t3_aabaab, optimize=True)
    t3_res += 1 * _tmp27
    t3_res -= 1 * _tmp27.transpose(1, 0, 2, 3, 4, 5)
    _tmp28 = einsum('aljd,bdcikl->abcijk', g_abab[v, o, o, v], t3_abbabb, optimize=True)
    t3_res += 1 * _tmp28
    t3_res -= 1 * _tmp28.transpose(1, 0, 2, 3, 4, 5)
    _tmp29 = einsum('ladi,dbcjlk->abcijk', g_aaaa[o, v, v, o], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp29
    t3_res += 1 * _tmp29.transpose(1, 0, 2, 3, 4, 5)
    _tmp30 = einsum('alid,bdcjkl->abcijk', g_abab[v, o, o, v], t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp30
    t3_res += 1 * _tmp30.transpose(1, 0, 2, 3, 4, 5)
    _tmp31 = einsum('lcdk,dabijl->abcijk', g_abab[o, v, v, o], t3_aaaaaa, optimize=True)
    t3_res += 1 * _tmp31
    _tmp32 = einsum('lcdk,badijl->abcijk', g_bbbb[o, v, v, o], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp32
    _tmp33 = einsum('lcjd,badilk->abcijk', g_abab[o, v, o, v], t3_aabaab, optimize=True)
    t3_res += 1 * _tmp33
    _tmp34 = einsum('lcid,badjlk->abcijk', g_abab[o, v, o, v], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp34
    _tmp35 = einsum('abde,decijk->abcijk', g_aaaa[v, v, v, v], t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp35
    _tmp36 = einsum('acde,dbeijk->abcijk', g_abab[v, v, v, v], t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp36
    _tmp37 = einsum('aced,bedijk->abcijk', g_abab[v, v, v, v], t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp37
    _tmp38 = einsum('bcde,daeijk->abcijk', g_abab[v, v, v, v], t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp38
    _tmp39 = einsum('bced,aedijk->abcijk', g_abab[v, v, v, v], t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp39
    _tmp40 = einsum('lmjk,al,bcim->abcijk', g_abab[o, o, o, o], t1_aa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp40
    t3_res += 1 * _tmp40.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp40.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp40.transpose(1, 0, 2, 4, 3, 5)
    _tmp41 = einsum('mljk,abim,cl->abcijk', g_abab[o, o, o, o], t2_aaaa, t1_bb, optimize=True)
    t3_res += 1 * _tmp41
    t3_res -= 1 * _tmp41.transpose(0, 1, 2, 4, 3, 5)
    _tmp42 = einsum('mlij,al,bcmk->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp42
    t3_res += 1 * _tmp42.transpose(1, 0, 2, 3, 4, 5)
    _tmp43 = einsum('aldk,bcil,dj->abcijk', g_abab[v, o, v, o], t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp43
    t3_res -= 1 * _tmp43.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp43.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp43.transpose(1, 0, 2, 4, 3, 5)
    _tmp44 = einsum('aldk,dbij,cl->abcijk', g_abab[v, o, v, o], t2_aaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp44
    _tmp45 = einsum('ladj,bl,dcik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=True)
    t3_res += 1 * _tmp45
    _tmp46 = einsum('aljd,bdik,cl->abcijk', g_abab[v, o, o, v], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp46
    _tmp47 = einsum('aljd,bcil,dk->abcijk', g_abab[v, o, o, v], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp47
    t3_res -= 1 * _tmp47.transpose(1, 0, 2, 3, 4, 5)
    _tmp48 = einsum('ladj,bclk,di->abcijk', g_aaaa[o, v, v, o], t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp48
    t3_res -= 1 * _tmp48.transpose(1, 0, 2, 3, 4, 5)
    _tmp49 = einsum('alid,bcjl,dk->abcijk', g_abab[v, o, o, v], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp49
    t3_res += 1 * _tmp49.transpose(1, 0, 2, 3, 4, 5)
    _tmp50 = einsum('ladi,bclk,dj->abcijk', g_aaaa[o, v, v, o], t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp50
    t3_res += 1 * _tmp50.transpose(1, 0, 2, 3, 4, 5)
    _tmp51 = einsum('ladi,bl,dcjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp51
    _tmp52 = einsum('alid,bdjk,cl->abcijk', g_abab[v, o, o, v], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp52
    _tmp53 = einsum('bldk,daij,cl->abcijk', g_abab[v, o, v, o], t2_aaaa, t1_bb, optimize=True)
    t3_res += 1 * _tmp53
    _tmp54 = einsum('lbdj,al,dcik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp54
    _tmp55 = einsum('bljd,adik,cl->abcijk', g_abab[v, o, o, v], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp55
    _tmp56 = einsum('lbdi,al,dcjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=True)
    t3_res += 1 * _tmp56
    _tmp57 = einsum('blid,adjk,cl->abcijk', g_abab[v, o, o, v], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp57
    _tmp58 = einsum('lcdk,abil,dj->abcijk', g_abab[o, v, v, o], t2_aaaa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp58
    t3_res += 1 * _tmp58.transpose(0, 1, 2, 4, 3, 5)
    _tmp59 = einsum('lcdk,al,dbij->abcijk', g_abab[o, v, v, o], t1_aa, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp59
    t3_res += 1 * _tmp59.transpose(1, 0, 2, 3, 4, 5)
    _tmp60 = einsum('lcjd,al,bdik->abcijk', g_abab[o, v, o, v], t1_aa, t2_abab, optimize=True)
    t3_res += 1 * _tmp60
    t3_res -= 1 * _tmp60.transpose(1, 0, 2, 3, 4, 5)
    _tmp61 = einsum('lcjd,abil,dk->abcijk', g_abab[o, v, o, v], t2_aaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp61
    _tmp62 = einsum('lcid,abjl,dk->abcijk', g_abab[o, v, o, v], t2_aaaa, t1_bb, optimize=True)
    t3_res += 1 * _tmp62
    _tmp63 = einsum('lcid,al,bdjk->abcijk', g_abab[o, v, o, v], t1_aa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp63
    t3_res += 1 * _tmp63.transpose(1, 0, 2, 3, 4, 5)
    _tmp64 = einsum('aced,ebij,dk->abcijk', g_abab[v, v, v, v], t2_aaaa, t1_bb, optimize=True)
    t3_res += 1 * _tmp64
    _tmp65 = einsum('abde,ecik,dj->abcijk', g_aaaa[v, v, v, v], t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp65
    _tmp66 = einsum('acde,beik,dj->abcijk', g_abab[v, v, v, v], t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp66
    _tmp67 = einsum('abde,ecjk,di->abcijk', g_aaaa[v, v, v, v], t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp67
    _tmp68 = einsum('acde,bejk,di->abcijk', g_abab[v, v, v, v], t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp68
    _tmp69 = einsum('bced,eaij,dk->abcijk', g_abab[v, v, v, v], t2_aaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp69
    _tmp70 = einsum('bcde,aeik,dj->abcijk', g_abab[v, v, v, v], t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp70
    _tmp71 = einsum('bcde,aejk,di->abcijk', g_abab[v, v, v, v], t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp71
    _tmp72 = einsum('lmdk,abcijm,dl->abcijk', g_abab[o, o, v, o], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp72
    _tmp73 = einsum('mldk,abcijm,dl->abcijk', g_bbbb[o, o, v, o], t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp73
    _tmp74 = einsum('mldj,abcimk,dl->abcijk', g_aaaa[o, o, v, o], t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp74
    _tmp75 = einsum('mljd,abcimk,dl->abcijk', g_abab[o, o, o, v], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp75
    _tmp76 = einsum('mldk,abciml,dj->abcijk', g_abab[o, o, v, o], t3_aabaab, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp76
    t3_res -= 0.5 * _tmp76.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp76
    t3_res -= 0.5 * _tmp76.transpose(0, 1, 2, 4, 3, 5)
    _tmp77 = einsum('lmdk,al,dbcijm->abcijk', g_abab[o, o, v, o], t1_aa, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp77
    t3_res -= 1 * _tmp77.transpose(1, 0, 2, 3, 4, 5)
    _tmp78 = einsum('mldj,al,dbcimk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp78
    t3_res += 1 * _tmp78.transpose(1, 0, 2, 3, 4, 5)
    _tmp79 = einsum('lmjd,al,bdcikm->abcijk', g_abab[o, o, o, v], t1_aa, t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp79
    t3_res += 1 * _tmp79.transpose(1, 0, 2, 3, 4, 5)
    _tmp80 = einsum('mldk,dabijm,cl->abcijk', g_abab[o, o, v, o], t3_aaaaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp80
    _tmp81 = einsum('mldk,badijm,cl->abcijk', g_bbbb[o, o, v, o], t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp81
    _tmp82 = einsum('mljd,badimk,cl->abcijk', g_abab[o, o, o, v], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp82
    _tmp83 = einsum('mljd,abciml,dk->abcijk', g_abab[o, o, o, v], t3_aabaab, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp83
    t3_res += 0.5 * _tmp83
    _tmp84 = einsum('mldj,abclmk,di->abcijk', g_aaaa[o, o, v, o], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp84
    _tmp85 = einsum('mldi,abcjmk,dl->abcijk', g_aaaa[o, o, v, o], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp85
    _tmp86 = einsum('mlid,abcjmk,dl->abcijk', g_abab[o, o, o, v], t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp86
    _tmp87 = einsum('mlid,abcjml,dk->abcijk', g_abab[o, o, o, v], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp87
    t3_res -= 0.5 * _tmp87
    _tmp88 = einsum('mldi,abclmk,dj->abcijk', g_aaaa[o, o, v, o], t3_aabaab, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp88
    _tmp89 = einsum('mldi,al,dbcjmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp89
    t3_res -= 1 * _tmp89.transpose(1, 0, 2, 3, 4, 5)
    _tmp90 = einsum('lmid,al,bdcjkm->abcijk', g_abab[o, o, o, v], t1_aa, t3_abbabb, optimize=True)
    t3_res += 1 * _tmp90
    t3_res -= 1 * _tmp90.transpose(1, 0, 2, 3, 4, 5)
    _tmp91 = einsum('mlid,badjmk,cl->abcijk', g_abab[o, o, o, v], t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp91
    _tmp92 = einsum('lade,ebcijk,dl->abcijk', g_aaaa[o, v, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp92
    t3_res -= 1 * _tmp92.transpose(1, 0, 2, 3, 4, 5)
    _tmp93 = einsum('aled,ebcijk,dl->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp93
    t3_res -= 1 * _tmp93.transpose(1, 0, 2, 3, 4, 5)
    _tmp94 = einsum('aled,ebcijl,dk->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp94
    t3_res += 1 * _tmp94.transpose(1, 0, 2, 3, 4, 5)
    _tmp95 = einsum('lade,ebcilk,dj->abcijk', g_aaaa[o, v, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp95
    t3_res += 1 * _tmp95.transpose(1, 0, 2, 3, 4, 5)
    _tmp96 = einsum('alde,becikl,dj->abcijk', g_abab[v, o, v, v], t3_abbabb, t1_aa, optimize=True)
    t3_res += 1 * _tmp96
    t3_res -= 1 * _tmp96.transpose(1, 0, 2, 3, 4, 5)
    _tmp97 = einsum('lade,ebcjlk,di->abcijk', g_aaaa[o, v, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp97
    t3_res -= 1 * _tmp97.transpose(1, 0, 2, 3, 4, 5)
    _tmp98 = einsum('alde,becjkl,di->abcijk', g_abab[v, o, v, v], t3_abbabb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp98
    t3_res += 1 * _tmp98.transpose(1, 0, 2, 3, 4, 5)
    _tmp99 = einsum('lade,bl,decijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp99
    _tmp100 = einsum('alde,dbeijk,cl->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp100
    _tmp101 = einsum('aled,bedijk,cl->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp101
    _tmp102 = einsum('lbde,al,decijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp102
    _tmp103 = einsum('blde,daeijk,cl->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp103
    _tmp104 = einsum('bled,aedijk,cl->abcijk', g_abab[v, o, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp104
    _tmp105 = einsum('lcde,baeijk,dl->abcijk', g_abab[o, v, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp105
    _tmp106 = einsum('lcde,baeijk,dl->abcijk', g_bbbb[o, v, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp106
    _tmp107 = einsum('lced,eabijl,dk->abcijk', g_abab[o, v, v, v], t3_aaaaaa, t1_bb, optimize=True)
    t3_res += 1 * _tmp107
    _tmp108 = einsum('lcde,baeijl,dk->abcijk', g_bbbb[o, v, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp108
    _tmp109 = einsum('lcde,baeilk,dj->abcijk', g_abab[o, v, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp109
    _tmp110 = einsum('lcde,baejlk,di->abcijk', g_abab[o, v, v, v], t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp110
    _tmp111 = einsum('lcde,al,dbeijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp111
    t3_res += 0.5 * _tmp111.transpose(1, 0, 2, 3, 4, 5)
    _tmp112 = einsum('lced,al,bedijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp112
    t3_res -= 0.5 * _tmp112.transpose(1, 0, 2, 3, 4, 5)
    _tmp113 = einsum('lmde,abcijm,delk->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp113
    t3_res -= 0.5 * _tmp113
    _tmp114 = einsum('mlde,abcijm,dekl->abcijk', g_bbbb[o, o, v, v], t3_aabaab, t2_bbbb, optimize=True)
    t3_res -= 0.5 * _tmp114
    _tmp115 = einsum('mlde,abcimk,dejl->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp115
    _tmp116 = einsum('mlde,abcimk,dejl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp116
    t3_res -= 0.5 * _tmp116
    _tmp117 = einsum('mlde,abcjmk,deil->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp117
    _tmp118 = einsum('mlde,abcjmk,deil->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp118
    t3_res += 0.5 * _tmp118
    _tmp119 = einsum('mlde,abciml,dejk->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.25 * _tmp119
    t3_res -= 0.25 * _tmp119.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.25 * _tmp119
    t3_res -= 0.25 * _tmp119.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.25 * _tmp119
    t3_res -= 0.25 * _tmp119.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.25 * _tmp119
    t3_res -= 0.25 * _tmp119.transpose(0, 1, 2, 4, 3, 5)
    _tmp120 = einsum('mlde,abclmk,deij->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t2_aaaa, optimize=True)
    t3_res -= 0.25 * _tmp120
    _tmp121 = einsum('mlde,daml,ebcijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp121
    t3_res += 0.5 * _tmp121.transpose(1, 0, 2, 3, 4, 5)
    _tmp122 = einsum('mled,adml,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp122
    t3_res += 0.5 * _tmp122.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp122
    t3_res += 0.5 * _tmp122.transpose(1, 0, 2, 3, 4, 5)
    _tmp123 = einsum('lmed,adlk,ebcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp123
    t3_res -= 1 * _tmp123.transpose(1, 0, 2, 3, 4, 5)
    _tmp124 = einsum('mlde,dajl,ebcimk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp124
    t3_res -= 1 * _tmp124.transpose(1, 0, 2, 3, 4, 5)
    _tmp125 = einsum('lmde,dajl,becikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp125
    t3_res += 1 * _tmp125.transpose(1, 0, 2, 3, 4, 5)
    _tmp126 = einsum('mled,adjl,ebcimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res += 1 * _tmp126
    t3_res -= 1 * _tmp126.transpose(1, 0, 2, 3, 4, 5)
    _tmp127 = einsum('mlde,adjl,becikm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp127
    t3_res += 1 * _tmp127.transpose(1, 0, 2, 3, 4, 5)
    _tmp128 = einsum('mlde,dail,ebcjmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp128
    t3_res += 1 * _tmp128.transpose(1, 0, 2, 3, 4, 5)
    _tmp129 = einsum('lmde,dail,becjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=True)
    t3_res += 1 * _tmp129
    t3_res -= 1 * _tmp129.transpose(1, 0, 2, 3, 4, 5)
    _tmp130 = einsum('mled,adil,ebcjmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp130
    t3_res += 1 * _tmp130.transpose(1, 0, 2, 3, 4, 5)
    _tmp131 = einsum('mlde,adil,becjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res += 1 * _tmp131
    t3_res -= 1 * _tmp131.transpose(1, 0, 2, 3, 4, 5)
    _tmp132 = einsum('mled,adjk,ebciml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp132
    t3_res += 0.5 * _tmp132.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp132.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp132.transpose(1, 0, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp132
    t3_res += 0.5 * _tmp132.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp132.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp132.transpose(1, 0, 2, 4, 3, 5)
    _tmp133 = einsum('mlde,adjk,beciml->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp133
    t3_res += 0.5 * _tmp133.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp133.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp133.transpose(1, 0, 2, 4, 3, 5)
    _tmp134 = einsum('mlde,daij,ebclmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp134
    t3_res -= 0.5 * _tmp134.transpose(1, 0, 2, 3, 4, 5)
    _tmp135 = einsum('mlde,daij,becmkl->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp135
    t3_res += 0.5 * _tmp135.transpose(1, 0, 2, 3, 4, 5)
    _tmp136 = einsum('lmde,daij,beclmk->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp136
    t3_res -= 0.5 * _tmp136.transpose(1, 0, 2, 3, 4, 5)
    _tmp137 = einsum('mlde,baeijk,dcml->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp137
    t3_res += 0.5 * _tmp137
    _tmp138 = einsum('mlde,baeijk,dcml->abcijk', g_bbbb[o, o, v, v], t3_aabaab, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp138
    _tmp139 = einsum('mlde,eabijm,dclk->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp139
    _tmp140 = einsum('mled,eabijm,dckl->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp140
    _tmp141 = einsum('lmde,baeijm,dclk->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp141
    _tmp142 = einsum('mlde,baeijm,dckl->abcijk', g_bbbb[o, o, v, v], t3_aabaab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp142
    _tmp143 = einsum('mlde,baeimk,dcjl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp143
    _tmp144 = einsum('mlde,baejmk,dcil->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 1 * _tmp144
    _tmp145 = einsum('mlde,eabiml,dcjk->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp145
    t3_res += 0.5 * _tmp145.transpose(0, 1, 2, 4, 3, 5)
    _tmp146 = einsum('mlde,baeiml,dcjk->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp146
    t3_res -= 0.5 * _tmp146.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp146
    t3_res -= 0.5 * _tmp146.transpose(0, 1, 2, 4, 3, 5)
    _tmp147 = einsum('mlde,abml,decijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 0.25 * _tmp147
    _tmp148 = einsum('mlde,acml,dbeijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res += 0.25 * _tmp148
    _tmp149 = einsum('mled,acml,bedijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res -= 0.25 * _tmp149
    t3_res += 0.25 * _tmp148
    t3_res -= 0.25 * _tmp149
    _tmp150 = einsum('mlde,aclk,debijm->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aaaaaa, optimize=True)
    t3_res -= 0.5 * _tmp150
    _tmp151 = einsum('lmde,aclk,dbeijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp151
    _tmp152 = einsum('lmed,aclk,bedijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp152
    _tmp153 = einsum('mlde,abjl,decimk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp153
    _tmp154 = einsum('lmde,abjl,decikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp154
    t3_res -= 0.5 * _tmp154
    _tmp155 = einsum('mlde,acjl,dbeimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp155
    _tmp156 = einsum('mled,acjl,bedimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp156
    _tmp157 = einsum('mlde,acjl,bedikm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp157
    _tmp158 = einsum('mlde,abil,decjmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp158
    _tmp159 = einsum('lmde,abil,decjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp159
    t3_res += 0.5 * _tmp159
    _tmp160 = einsum('mlde,acil,dbejmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res += 0.5 * _tmp160
    _tmp161 = einsum('mled,acil,bedjmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp161
    _tmp162 = einsum('mlde,acil,bedjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp162
    _tmp163 = einsum('mlde,daeijk,bcml->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 0.25 * _tmp163
    t3_res -= 0.25 * _tmp163
    _tmp164 = einsum('mled,aedijk,bcml->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.25 * _tmp164
    t3_res += 0.25 * _tmp164
    _tmp165 = einsum('mlde,deaijm,bclk->abcijk', g_aaaa[o, o, v, v], t3_aaaaaa, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp165
    _tmp166 = einsum('lmde,daeijm,bclk->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp166
    _tmp167 = einsum('lmed,aedijm,bclk->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp167
    _tmp168 = einsum('mlde,daeimk,bcjl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp168
    _tmp169 = einsum('mled,aedimk,bcjl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp169
    _tmp170 = einsum('mlde,aedikm,bcjl->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp170
    _tmp171 = einsum('mlde,daejmk,bcil->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp171
    _tmp172 = einsum('mled,aedjmk,bcil->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp172
    _tmp173 = einsum('mlde,aedjkm,bcil->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp173
    _tmp174 = einsum('lmdk,dajl,bcim->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp174
    t3_res += 1 * _tmp174.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp174.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp174.transpose(1, 0, 2, 4, 3, 5)
    _tmp175 = einsum('mldk,adjl,bcim->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp175
    t3_res += 1 * _tmp175.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp175.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp175.transpose(1, 0, 2, 4, 3, 5)
    _tmp176 = einsum('mldk,daij,bcml->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp176
    t3_res += 0.5 * _tmp176.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp176
    t3_res += 0.5 * _tmp176.transpose(1, 0, 2, 3, 4, 5)
    _tmp177 = einsum('mljd,adik,bcml->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp177
    t3_res -= 0.5 * _tmp177.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp177
    t3_res -= 0.5 * _tmp177.transpose(1, 0, 2, 3, 4, 5)
    _tmp178 = einsum('mldk,abim,dcjl->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp178
    t3_res -= 1 * _tmp178.transpose(0, 1, 2, 4, 3, 5)
    _tmp179 = einsum('mldj,abml,dcik->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp179
    _tmp180 = einsum('lmjd,adlk,bcim->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp180
    t3_res += 1 * _tmp180.transpose(1, 0, 2, 3, 4, 5)
    _tmp181 = einsum('mldj,dail,bcmk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp181
    t3_res += 1 * _tmp181.transpose(1, 0, 2, 3, 4, 5)
    _tmp182 = einsum('mljd,adil,bcmk->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp182
    t3_res += 1 * _tmp182.transpose(1, 0, 2, 3, 4, 5)
    _tmp183 = einsum('mldj,abim,dclk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp183
    _tmp184 = einsum('mljd,abim,dckl->abcijk', g_abab[o, o, o, v], t2_aaaa, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp184
    _tmp185 = einsum('lmid,adlk,bcjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp185
    t3_res -= 1 * _tmp185.transpose(1, 0, 2, 3, 4, 5)
    _tmp186 = einsum('mldi,dajl,bcmk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp186
    t3_res -= 1 * _tmp186.transpose(1, 0, 2, 3, 4, 5)
    _tmp187 = einsum('mlid,adjl,bcmk->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp187
    t3_res -= 1 * _tmp187.transpose(1, 0, 2, 3, 4, 5)
    _tmp188 = einsum('mlid,adjk,bcml->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp188
    t3_res += 0.5 * _tmp188.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp188
    t3_res += 0.5 * _tmp188.transpose(1, 0, 2, 3, 4, 5)
    _tmp189 = einsum('mldi,abjm,dclk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp189
    _tmp190 = einsum('mlid,abjm,dckl->abcijk', g_abab[o, o, o, v], t2_aaaa, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp190
    _tmp191 = einsum('mldi,abml,dcjk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp191
    _tmp192 = einsum('alde,bcil,dejk->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp192
    t3_res -= 0.5 * _tmp192.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp192.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp192.transpose(1, 0, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp192
    t3_res -= 0.5 * _tmp192.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp192.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp192.transpose(1, 0, 2, 4, 3, 5)
    _tmp193 = einsum('lade,bclk,deij->abcijk', g_aaaa[o, v, v, v], t2_abab, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp193
    t3_res -= 0.5 * _tmp193.transpose(1, 0, 2, 3, 4, 5)
    _tmp194 = einsum('lade,dbjl,ecik->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp194
    t3_res += 1 * _tmp194.transpose(1, 0, 2, 3, 4, 5)
    _tmp195 = einsum('aled,bdjl,ecik->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp195
    t3_res -= 1 * _tmp195.transpose(1, 0, 2, 3, 4, 5)
    _tmp196 = einsum('lade,dbil,ecjk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp196
    t3_res -= 1 * _tmp196.transpose(1, 0, 2, 3, 4, 5)
    _tmp197 = einsum('aled,bdil,ecjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp197
    t3_res += 1 * _tmp197.transpose(1, 0, 2, 3, 4, 5)
    _tmp198 = einsum('aled,bdjk,ecil->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp198
    t3_res += 1 * _tmp198.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp198.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp198.transpose(1, 0, 2, 4, 3, 5)
    _tmp199 = einsum('lade,dbij,eclk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp199
    t3_res += 1 * _tmp199.transpose(1, 0, 2, 3, 4, 5)
    _tmp200 = einsum('alde,dbij,eckl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp200
    t3_res += 1 * _tmp200.transpose(1, 0, 2, 3, 4, 5)
    _tmp201 = einsum('lcde,abil,dejk->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp201
    t3_res += 0.5 * _tmp201.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp201
    t3_res += 0.5 * _tmp201.transpose(0, 1, 2, 4, 3, 5)
    _tmp202 = einsum('lced,adlk,ebij->abcijk', g_abab[o, v, v, v], t2_abab, t2_aaaa, optimize=True)
    t3_res -= 1 * _tmp202
    _tmp203 = einsum('lcde,dajl,beik->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp203
    _tmp204 = einsum('lcde,adjl,beik->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp204
    _tmp205 = einsum('lcde,dail,bejk->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp205
    _tmp206 = einsum('lcde,adil,bejk->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp206
    _tmp207 = einsum('lced,adjk,ebil->abcijk', g_abab[o, v, v, v], t2_abab, t2_aaaa, optimize=True)
    t3_res += 1 * _tmp207
    t3_res -= 1 * _tmp207.transpose(0, 1, 2, 4, 3, 5)
    _tmp208 = einsum('lcde,adjk,beil->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp208
    t3_res -= 1 * _tmp208.transpose(0, 1, 2, 4, 3, 5)
    _tmp209 = einsum('lcde,daij,belk->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp209
    _tmp210 = einsum('lmde,aejk,bcim,dl->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp210
    t3_res -= 1 * _tmp210.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp210.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp210.transpose(1, 0, 2, 4, 3, 5)
    _tmp211 = einsum('mlde,aejk,bcim,dl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp211
    t3_res += 1 * _tmp211.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp211.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp211.transpose(1, 0, 2, 4, 3, 5)
    _tmp212 = einsum('mlde,eaij,bcmk,dl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp212
    t3_res += 1 * _tmp212.transpose(1, 0, 2, 3, 4, 5)
    _tmp213 = einsum('mled,eaij,bcmk,dl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp213
    t3_res -= 1 * _tmp213.transpose(1, 0, 2, 3, 4, 5)
    _tmp214 = einsum('mlde,abim,ecjk,dl->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp214
    t3_res -= 1 * _tmp214.transpose(0, 1, 2, 4, 3, 5)
    _tmp215 = einsum('mled,abim,ecjk,dl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp215
    t3_res += 1 * _tmp215.transpose(0, 1, 2, 4, 3, 5)
    _tmp216 = einsum('lmed,eajl,bcim,dk->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp216
    t3_res += 1 * _tmp216.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp216.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp216.transpose(1, 0, 2, 4, 3, 5)
    _tmp217 = einsum('mlde,aejl,bcim,dk->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp217
    t3_res -= 1 * _tmp217.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp217.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp217.transpose(1, 0, 2, 4, 3, 5)
    _tmp218 = einsum('mled,eaij,bcml,dk->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp218
    t3_res += 0.5 * _tmp218.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp218
    t3_res += 0.5 * _tmp218.transpose(1, 0, 2, 3, 4, 5)
    _tmp219 = einsum('mlde,aeik,bcml,dj->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp219
    t3_res -= 0.5 * _tmp219.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp219
    t3_res -= 0.5 * _tmp219.transpose(1, 0, 2, 3, 4, 5)
    _tmp220 = einsum('mled,abim,ecjl,dk->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp220
    t3_res -= 1 * _tmp220.transpose(0, 1, 2, 4, 3, 5)
    _tmp221 = einsum('mlde,abml,ecik,dj->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp221
    _tmp222 = einsum('lmde,aelk,bcim,dj->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp222
    t3_res += 1 * _tmp222.transpose(1, 0, 2, 3, 4, 5)
    _tmp223 = einsum('mlde,eail,bcmk,dj->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp223
    t3_res -= 1 * _tmp223.transpose(1, 0, 2, 3, 4, 5)
    _tmp224 = einsum('mlde,aeil,bcmk,dj->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp224
    t3_res += 1 * _tmp224.transpose(1, 0, 2, 3, 4, 5)
    _tmp225 = einsum('mlde,abim,eclk,dj->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp225
    _tmp226 = einsum('mlde,abim,eckl,dj->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp226
    _tmp227 = einsum('lmde,aelk,bcjm,di->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp227
    t3_res -= 1 * _tmp227.transpose(1, 0, 2, 3, 4, 5)
    _tmp228 = einsum('mlde,eajl,bcmk,di->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp228
    t3_res += 1 * _tmp228.transpose(1, 0, 2, 3, 4, 5)
    _tmp229 = einsum('mlde,aejl,bcmk,di->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp229
    t3_res -= 1 * _tmp229.transpose(1, 0, 2, 3, 4, 5)
    _tmp230 = einsum('mlde,aejk,bcml,di->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp230
    t3_res += 0.5 * _tmp230.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 0.5 * _tmp230
    t3_res += 0.5 * _tmp230.transpose(1, 0, 2, 3, 4, 5)
    _tmp231 = einsum('mlde,abjm,eclk,di->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp231
    _tmp232 = einsum('mlde,abjm,eckl,di->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp232
    _tmp233 = einsum('mlde,abml,ecjk,di->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp233
    _tmp234 = einsum('lmde,al,bcim,dejk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp234
    t3_res += 0.5 * _tmp234.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp234.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp234.transpose(1, 0, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp234
    t3_res += 0.5 * _tmp234.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 0.5 * _tmp234.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 0.5 * _tmp234.transpose(1, 0, 2, 4, 3, 5)
    _tmp235 = einsum('mlde,al,bcmk,deij->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=True)
    t3_res -= 0.5 * _tmp235
    t3_res += 0.5 * _tmp235.transpose(1, 0, 2, 3, 4, 5)
    _tmp236 = einsum('mlde,al,dbjm,ecik->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp236
    t3_res -= 1 * _tmp236.transpose(1, 0, 2, 3, 4, 5)
    _tmp237 = einsum('lmed,al,bdjm,ecik->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp237
    t3_res += 1 * _tmp237.transpose(1, 0, 2, 3, 4, 5)
    _tmp238 = einsum('mlde,al,dbim,ecjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp238
    t3_res += 1 * _tmp238.transpose(1, 0, 2, 3, 4, 5)
    _tmp239 = einsum('lmed,al,bdim,ecjk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp239
    t3_res -= 1 * _tmp239.transpose(1, 0, 2, 3, 4, 5)
    _tmp240 = einsum('lmed,al,bdjk,ecim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp240
    t3_res -= 1 * _tmp240.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp240.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp240.transpose(1, 0, 2, 4, 3, 5)
    _tmp241 = einsum('mlde,al,dbij,ecmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=True)
    t3_res += 1 * _tmp241
    t3_res -= 1 * _tmp241.transpose(1, 0, 2, 3, 4, 5)
    _tmp242 = einsum('lmde,al,dbij,eckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp242
    t3_res -= 1 * _tmp242.transpose(1, 0, 2, 3, 4, 5)
    _tmp243 = einsum('mlde,abim,cl,dejk->abcijk', g_abab[o, o, v, v], t2_aaaa, t1_bb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp243
    t3_res -= 0.5 * _tmp243.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp243
    t3_res -= 0.5 * _tmp243.transpose(0, 1, 2, 4, 3, 5)
    _tmp244 = einsum('mled,admk,ebij,cl->abcijk', g_abab[o, o, v, v], t2_abab, t2_aaaa, t1_bb, optimize=True)
    t3_res += 1 * _tmp244
    _tmp245 = einsum('mlde,dajm,beik,cl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp245
    _tmp246 = einsum('mlde,adjm,beik,cl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp246
    _tmp247 = einsum('mlde,daim,bejk,cl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp247
    _tmp248 = einsum('mlde,adim,bejk,cl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp248
    _tmp249 = einsum('mled,adjk,ebim,cl->abcijk', g_abab[o, o, v, v], t2_abab, t2_aaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp249
    t3_res += 1 * _tmp249.transpose(0, 1, 2, 4, 3, 5)
    _tmp250 = einsum('mlde,adjk,beim,cl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp250
    t3_res += 1 * _tmp250.transpose(0, 1, 2, 4, 3, 5)
    _tmp251 = einsum('mlde,daij,bemk,cl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp251
    _tmp252 = einsum('lmdk,al,bcim,dj->abcijk', g_abab[o, o, v, o], t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp252
    t3_res += 1 * _tmp252.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp252.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp252.transpose(1, 0, 2, 4, 3, 5)
    _tmp253 = einsum('mldk,abim,cl,dj->abcijk', g_abab[o, o, v, o], t2_aaaa, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp253
    t3_res -= 1 * _tmp253.transpose(0, 1, 2, 4, 3, 5)
    _tmp254 = einsum('lmdk,al,dbij,cm->abcijk', g_abab[o, o, v, o], t1_aa, t2_aaaa, t1_bb, optimize=True)
    t3_res += 1 * _tmp254
    _tmp255 = einsum('mldj,al,bm,dcik->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp255
    _tmp256 = einsum('lmjd,al,bdik,cm->abcijk', g_abab[o, o, o, v], t1_aa, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp256
    _tmp257 = einsum('lmdk,daij,bl,cm->abcijk', g_abab[o, o, v, o], t2_aaaa, t1_aa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp257
    _tmp258 = einsum('lmjd,adik,bl,cm->abcijk', g_abab[o, o, o, v], t2_abab, t1_aa, t1_bb, optimize=True)
    t3_res += 1 * _tmp258
    _tmp259 = einsum('lmjd,al,bcim,dk->abcijk', g_abab[o, o, o, v], t1_aa, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp259
    t3_res += 1 * _tmp259.transpose(1, 0, 2, 3, 4, 5)
    _tmp260 = einsum('mldj,al,bcmk,di->abcijk', g_aaaa[o, o, v, o], t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp260
    t3_res += 1 * _tmp260.transpose(1, 0, 2, 3, 4, 5)
    _tmp261 = einsum('mljd,abim,cl,dk->abcijk', g_abab[o, o, o, v], t2_aaaa, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp261
    _tmp262 = einsum('lmid,al,bcjm,dk->abcijk', g_abab[o, o, o, v], t1_aa, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp262
    t3_res -= 1 * _tmp262.transpose(1, 0, 2, 3, 4, 5)
    _tmp263 = einsum('mldi,al,bcmk,dj->abcijk', g_aaaa[o, o, v, o], t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp263
    t3_res -= 1 * _tmp263.transpose(1, 0, 2, 3, 4, 5)
    _tmp264 = einsum('mlid,abjm,cl,dk->abcijk', g_abab[o, o, o, v], t2_aaaa, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp264
    _tmp265 = einsum('mldi,al,bm,dcjk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=True)
    t3_res += 1 * _tmp265
    _tmp266 = einsum('lmid,al,bdjk,cm->abcijk', g_abab[o, o, o, v], t1_aa, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp266
    _tmp267 = einsum('lmid,adjk,bl,cm->abcijk', g_abab[o, o, o, v], t2_abab, t1_aa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp267
    _tmp268 = einsum('aled,bcil,dk,ej->abcijk', g_abab[v, o, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp268
    t3_res -= 1 * _tmp268.transpose(1, 0, 2, 3, 4, 5)
    t3_res -= 1 * _tmp268.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 1 * _tmp268.transpose(1, 0, 2, 4, 3, 5)
    _tmp269 = einsum('aled,ebij,cl,dk->abcijk', g_abab[v, o, v, v], t2_aaaa, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp269
    _tmp270 = einsum('lade,bl,ecik,dj->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp270
    _tmp271 = einsum('alde,beik,cl,dj->abcijk', g_abab[v, o, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp271
    _tmp272 = einsum('lade,bclk,dj,ei->abcijk', g_aaaa[o, v, v, v], t2_abab, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp272
    t3_res += 1 * _tmp272.transpose(1, 0, 2, 3, 4, 5)
    _tmp273 = einsum('lade,bl,ecjk,di->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp273
    _tmp274 = einsum('alde,bejk,cl,di->abcijk', g_abab[v, o, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp274
    _tmp275 = einsum('bled,eaij,cl,dk->abcijk', g_abab[v, o, v, v], t2_aaaa, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp275
    _tmp276 = einsum('lbde,al,ecik,dj->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp276
    _tmp277 = einsum('blde,aeik,cl,dj->abcijk', g_abab[v, o, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp277
    _tmp278 = einsum('lbde,al,ecjk,di->abcijk', g_aaaa[o, v, v, v], t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp278
    _tmp279 = einsum('blde,aejk,cl,di->abcijk', g_abab[v, o, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp279
    _tmp280 = einsum('lced,abil,dk,ej->abcijk', g_abab[o, v, v, v], t2_aaaa, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp280
    t3_res += 1 * _tmp280.transpose(0, 1, 2, 4, 3, 5)
    _tmp281 = einsum('lced,al,ebij,dk->abcijk', g_abab[o, v, v, v], t1_aa, t2_aaaa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp281
    t3_res += 1 * _tmp281.transpose(1, 0, 2, 3, 4, 5)
    _tmp282 = einsum('lcde,al,beik,dj->abcijk', g_abab[o, v, v, v], t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp282
    t3_res -= 1 * _tmp282.transpose(1, 0, 2, 3, 4, 5)
    _tmp283 = einsum('lcde,al,bejk,di->abcijk', g_abab[o, v, v, v], t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp283
    t3_res += 1 * _tmp283.transpose(1, 0, 2, 3, 4, 5)
    _tmp284 = einsum('lmde,abcijm,dl,ek->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_aa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp284
    _tmp285 = einsum('mlde,abcijm,dl,ek->abcijk', g_bbbb[o, o, v, v], t3_aabaab, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp285
    _tmp286 = einsum('mlde,abcimk,dl,ej->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp286
    _tmp287 = einsum('mled,abcimk,dl,ej->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp287
    _tmp288 = einsum('mlde,abcjmk,dl,ei->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp288
    _tmp289 = einsum('mled,abcjmk,dl,ei->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp289
    _tmp290 = einsum('mlde,am,ebcijk,dl->abcijk', g_aaaa[o, o, v, v], t1_aa, t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp290
    t3_res -= 1 * _tmp290.transpose(1, 0, 2, 3, 4, 5)
    _tmp291 = einsum('mled,am,ebcijk,dl->abcijk', g_abab[o, o, v, v], t1_aa, t3_aabaab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp291
    t3_res += 1 * _tmp291.transpose(1, 0, 2, 3, 4, 5)
    _tmp292 = einsum('lmde,baeijk,cm,dl->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp292
    _tmp293 = einsum('mlde,baeijk,cm,dl->abcijk', g_bbbb[o, o, v, v], t3_aabaab, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp293
    _tmp294 = einsum('mled,abciml,dk,ej->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_bb, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp294
    t3_res -= 0.5 * _tmp294.transpose(0, 1, 2, 4, 3, 5)
    t3_res += 0.5 * _tmp294
    t3_res -= 0.5 * _tmp294.transpose(0, 1, 2, 4, 3, 5)
    _tmp295 = einsum('lmed,al,ebcijm,dk->abcijk', g_abab[o, o, v, v], t1_aa, t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp295
    t3_res -= 1 * _tmp295.transpose(1, 0, 2, 3, 4, 5)
    _tmp296 = einsum('mlde,al,ebcimk,dj->abcijk', g_aaaa[o, o, v, v], t1_aa, t3_aabaab, t1_aa, optimize=True)
    t3_res += 1 * _tmp296
    t3_res -= 1 * _tmp296.transpose(1, 0, 2, 3, 4, 5)
    _tmp297 = einsum('lmde,al,becikm,dj->abcijk', g_abab[o, o, v, v], t1_aa, t3_abbabb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp297
    t3_res += 1 * _tmp297.transpose(1, 0, 2, 3, 4, 5)
    _tmp298 = einsum('mled,eabijm,cl,dk->abcijk', g_abab[o, o, v, v], t3_aaaaaa, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp298
    _tmp299 = einsum('mlde,baeijm,cl,dk->abcijk', g_bbbb[o, o, v, v], t3_aabaab, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp299
    _tmp300 = einsum('mlde,baeimk,cl,dj->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp300
    _tmp301 = einsum('mlde,abclmk,dj,ei->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t1_aa, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp301
    _tmp302 = einsum('mlde,al,ebcjmk,di->abcijk', g_aaaa[o, o, v, v], t1_aa, t3_aabaab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp302
    t3_res += 1 * _tmp302.transpose(1, 0, 2, 3, 4, 5)
    _tmp303 = einsum('lmde,al,becjkm,di->abcijk', g_abab[o, o, v, v], t1_aa, t3_abbabb, t1_aa, optimize=True)
    t3_res += 1 * _tmp303
    t3_res -= 1 * _tmp303.transpose(1, 0, 2, 3, 4, 5)
    _tmp304 = einsum('mlde,baejmk,cl,di->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp304
    _tmp305 = einsum('mlde,al,bm,decijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp305
    _tmp306 = einsum('lmde,al,dbeijk,cm->abcijk', g_abab[o, o, v, v], t1_aa, t3_aabaab, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp306
    _tmp307 = einsum('lmed,al,bedijk,cm->abcijk', g_abab[o, o, v, v], t1_aa, t3_aabaab, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp307
    _tmp308 = einsum('lmde,daeijk,bl,cm->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_aa, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp308
    _tmp309 = einsum('lmed,aedijk,bl,cm->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_aa, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp309
    _tmp310 = einsum('lmed,al,bcim,dk,ej->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp310
    t3_res += 1 * _tmp310.transpose(1, 0, 2, 3, 4, 5)
    t3_res += 1 * _tmp310.transpose(0, 1, 2, 4, 3, 5)
    t3_res -= 1 * _tmp310.transpose(1, 0, 2, 4, 3, 5)
    _tmp311 = einsum('mled,abim,cl,dk,ej->abcijk', g_abab[o, o, v, v], t2_aaaa, t1_bb, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp311
    t3_res -= 1 * _tmp311.transpose(0, 1, 2, 4, 3, 5)
    _tmp312 = einsum('lmed,al,ebij,cm,dk->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp312
    _tmp313 = einsum('mlde,al,bm,ecik,dj->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp313
    _tmp314 = einsum('lmde,al,beik,cm,dj->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp314
    _tmp315 = einsum('lmed,eaij,bl,cm,dk->abcijk', g_abab[o, o, v, v], t2_aaaa, t1_aa, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp315
    _tmp316 = einsum('lmde,aeik,bl,cm,dj->abcijk', g_abab[o, o, v, v], t2_abab, t1_aa, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp316
    _tmp317 = einsum('mlde,al,bcmk,dj,ei->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t1_aa, t1_aa, optimize=True)
    t3_res += 1 * _tmp317
    t3_res -= 1 * _tmp317.transpose(1, 0, 2, 3, 4, 5)
    _tmp318 = einsum('mlde,al,bm,ecjk,di->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp318
    _tmp319 = einsum('lmde,al,bejk,cm,di->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp319
    _tmp320 = einsum('lmde,aejk,bl,cm,di->abcijk', g_abab[o, o, v, v], t2_abab, t1_aa, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp320
    return t3_res


def t3_abbabb_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v):
    nv, no = t1_aa.shape
    t3_res = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('lk,abcijl->abcijk', f_bb[o, o], t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp0
    t3_res += 1 * _tmp0.transpose(0, 1, 2, 3, 5, 4)
    _tmp1 = einsum('li,abclkj->abcijk', f_aa[o, o], t3_abbabb, optimize=True)
    t3_res += 1 * _tmp1
    _tmp2 = einsum('ad,dbcijk->abcijk', f_aa[v, v], t3_abbabb, optimize=True)
    t3_res += 1 * _tmp2
    _tmp3 = einsum('bd,adcijk->abcijk', f_bb[v, v], t3_abbabb, optimize=True)
    t3_res += 1 * _tmp3
    _tmp4 = einsum('cd,adbijk->abcijk', f_bb[v, v], t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp4
    _tmp5 = einsum('ld,abcijl,dk->abcijk', f_bb[o, v], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp5
    t3_res += 1 * _tmp5.transpose(0, 1, 2, 3, 5, 4)
    _tmp6 = einsum('ld,abclkj,di->abcijk', f_aa[o, v], t3_abbabb, t1_aa, optimize=True)
    t3_res += 1 * _tmp6
    _tmp7 = einsum('ld,al,dbcijk->abcijk', f_aa[o, v], t1_aa, t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp7
    _tmp8 = einsum('ld,adcijk,bl->abcijk', f_bb[o, v], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp8
    _tmp9 = einsum('ld,adbijk,cl->abcijk', f_bb[o, v], t3_abbabb, t1_bb, optimize=True)
    t3_res += 1 * _tmp9
    _tmp10 = einsum('ld,acil,dbjk->abcijk', f_bb[o, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp10
    _tmp11 = einsum('ld,adik,bcjl->abcijk', f_bb[o, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp11
    _tmp12 = einsum('ld,aclj,dbik->abcijk', f_aa[o, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp12
    _tmp13 = einsum('ld,adij,bckl->abcijk', f_bb[o, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp13
    _tmp14 = einsum('ld,aclk,dbij->abcijk', f_aa[o, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp14
    _tmp15 = einsum('ld,abil,dcjk->abcijk', f_bb[o, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp15
    _tmp16 = einsum('ld,ablj,dcik->abcijk', f_aa[o, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp16
    _tmp17 = einsum('ld,ablk,dcij->abcijk', f_aa[o, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp17
    _tmp18 = einsum('lbjk,acil->abcijk', g_bbbb[o, v, o, o], t2_abab, optimize=True)
    t3_res += 1 * _tmp18
    _tmp19 = einsum('alik,bcjl->abcijk', g_abab[v, o, o, o], t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp19
    _tmp20 = einsum('lbik,aclj->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    t3_res += 1 * _tmp20
    _tmp21 = einsum('alij,bckl->abcijk', g_abab[v, o, o, o], t2_bbbb, optimize=True)
    t3_res += 1 * _tmp21
    _tmp22 = einsum('lbij,aclk->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    t3_res -= 1 * _tmp22
    _tmp23 = einsum('lcjk,abil->abcijk', g_bbbb[o, v, o, o], t2_abab, optimize=True)
    t3_res -= 1 * _tmp23
    _tmp24 = einsum('lcik,ablj->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    t3_res -= 1 * _tmp24
    _tmp25 = einsum('lcij,ablk->abcijk', g_abab[o, v, o, o], t2_abab, optimize=True)
    t3_res += 1 * _tmp25
    _tmp26 = einsum('abdk,dcij->abcijk', g_abab[v, v, v, o], t2_abab, optimize=True)
    t3_res -= 1 * _tmp26
    t3_res += 1 * _tmp26.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 1 * _tmp26.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp26.transpose(0, 2, 1, 3, 5, 4)
    _tmp27 = einsum('abid,dcjk->abcijk', g_abab[v, v, o, v], t2_bbbb, optimize=True)
    t3_res += 1 * _tmp27
    t3_res -= 1 * _tmp27.transpose(0, 2, 1, 3, 4, 5)
    _tmp28 = einsum('bcdk,adij->abcijk', g_bbbb[v, v, v, o], t2_abab, optimize=True)
    t3_res += 1 * _tmp28
    t3_res -= 1 * _tmp28.transpose(0, 1, 2, 3, 5, 4)
    _tmp29 = einsum('mljk,abciml->abcijk', g_bbbb[o, o, o, o], t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp29
    _tmp30 = einsum('mlik,abcmjl->abcijk', g_abab[o, o, o, o], t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp30
    _tmp31 = einsum('lmik,abclmj->abcijk', g_abab[o, o, o, o], t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp31
    _tmp32 = einsum('mlij,abcmkl->abcijk', g_abab[o, o, o, o], t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp32
    _tmp33 = einsum('lmij,abclmk->abcijk', g_abab[o, o, o, o], t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp33
    _tmp34 = einsum('aldk,dbcijl->abcijk', g_abab[v, o, v, o], t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp34
    t3_res += 1 * _tmp34.transpose(0, 1, 2, 3, 5, 4)
    _tmp35 = einsum('lbdk,dacilj->abcijk', g_abab[o, v, v, o], t3_aabaab, optimize=True)
    t3_res += 1 * _tmp35
    t3_res -= 1 * _tmp35.transpose(0, 1, 2, 3, 5, 4)
    _tmp36 = einsum('lbdk,adcijl->abcijk', g_bbbb[o, v, v, o], t3_abbabb, optimize=True)
    t3_res += 1 * _tmp36
    t3_res -= 1 * _tmp36.transpose(0, 1, 2, 3, 5, 4)
    _tmp37 = einsum('ladi,dbclkj->abcijk', g_aaaa[o, v, v, o], t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp37
    _tmp38 = einsum('alid,dbcjkl->abcijk', g_abab[v, o, o, v], t3_bbbbbb, optimize=True)
    t3_res += 1 * _tmp38
    _tmp39 = einsum('lbid,adclkj->abcijk', g_abab[o, v, o, v], t3_abbabb, optimize=True)
    t3_res += 1 * _tmp39
    _tmp40 = einsum('lcdk,dabilj->abcijk', g_abab[o, v, v, o], t3_aabaab, optimize=True)
    t3_res -= 1 * _tmp40
    t3_res += 1 * _tmp40.transpose(0, 1, 2, 3, 5, 4)
    _tmp41 = einsum('lcdk,adbijl->abcijk', g_bbbb[o, v, v, o], t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp41
    t3_res += 1 * _tmp41.transpose(0, 1, 2, 3, 5, 4)
    _tmp42 = einsum('lcid,adblkj->abcijk', g_abab[o, v, o, v], t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp42
    _tmp43 = einsum('abde,decijk->abcijk', g_abab[v, v, v, v], t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp43
    t3_res -= 0.5 * _tmp43.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.5 * _tmp43
    t3_res -= 0.5 * _tmp43.transpose(0, 2, 1, 3, 4, 5)
    _tmp44 = einsum('bcde,aedijk->abcijk', g_bbbb[v, v, v, v], t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp44
    _tmp45 = einsum('mljk,acim,bl->abcijk', g_bbbb[o, o, o, o], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp45
    _tmp46 = einsum('lmik,al,bcjm->abcijk', g_abab[o, o, o, o], t1_aa, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp46
    _tmp47 = einsum('mlik,acmj,bl->abcijk', g_abab[o, o, o, o], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp47
    _tmp48 = einsum('mljk,abim,cl->abcijk', g_bbbb[o, o, o, o], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp48
    _tmp49 = einsum('mlik,abmj,cl->abcijk', g_abab[o, o, o, o], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp49
    _tmp50 = einsum('lmij,al,bckm->abcijk', g_abab[o, o, o, o], t1_aa, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp50
    _tmp51 = einsum('mlij,acmk,bl->abcijk', g_abab[o, o, o, o], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp51
    _tmp52 = einsum('mlij,abmk,cl->abcijk', g_abab[o, o, o, o], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp52
    _tmp53 = einsum('lbdk,acil,dj->abcijk', g_bbbb[o, v, v, o], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp53
    _tmp54 = einsum('aldk,bcjl,di->abcijk', g_abab[v, o, v, o], t2_bbbb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp54
    _tmp55 = einsum('lbdk,aclj,di->abcijk', g_abab[o, v, v, o], t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp55
    _tmp56 = einsum('aldk,bl,dcij->abcijk', g_abab[v, o, v, o], t1_bb, t2_abab, optimize=True)
    t3_res += 1 * _tmp56
    t3_res -= 1 * _tmp56.transpose(0, 2, 1, 3, 4, 5)
    t3_res -= 1 * _tmp56.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp56.transpose(0, 2, 1, 3, 5, 4)
    _tmp57 = einsum('lbdj,acil,dk->abcijk', g_bbbb[o, v, v, o], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp57
    _tmp58 = einsum('aldj,bckl,di->abcijk', g_abab[v, o, v, o], t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp58
    _tmp59 = einsum('lbdj,aclk,di->abcijk', g_abab[o, v, v, o], t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp59
    _tmp60 = einsum('alid,bcjl,dk->abcijk', g_abab[v, o, o, v], t2_bbbb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp60
    t3_res += 1 * _tmp60.transpose(0, 1, 2, 3, 5, 4)
    _tmp61 = einsum('lbid,aclj,dk->abcijk', g_abab[o, v, o, v], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp61
    t3_res -= 1 * _tmp61.transpose(0, 1, 2, 3, 5, 4)
    _tmp62 = einsum('alid,bl,dcjk->abcijk', g_abab[v, o, o, v], t1_bb, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp62
    t3_res += 1 * _tmp62.transpose(0, 2, 1, 3, 4, 5)
    _tmp63 = einsum('lbdk,al,dcij->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=True)
    t3_res += 1 * _tmp63
    t3_res -= 1 * _tmp63.transpose(0, 1, 2, 3, 5, 4)
    _tmp64 = einsum('lbdk,adij,cl->abcijk', g_bbbb[o, v, v, o], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp64
    t3_res -= 1 * _tmp64.transpose(0, 1, 2, 3, 5, 4)
    _tmp65 = einsum('lbid,al,dcjk->abcijk', g_abab[o, v, o, v], t1_aa, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp65
    _tmp66 = einsum('lcdk,abil,dj->abcijk', g_bbbb[o, v, v, o], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp66
    _tmp67 = einsum('lcdk,ablj,di->abcijk', g_abab[o, v, v, o], t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp67
    _tmp68 = einsum('lcdk,al,dbij->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=True)
    t3_res -= 1 * _tmp68
    t3_res += 1 * _tmp68.transpose(0, 1, 2, 3, 5, 4)
    _tmp69 = einsum('lcdk,adij,bl->abcijk', g_bbbb[o, v, v, o], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp69
    t3_res += 1 * _tmp69.transpose(0, 1, 2, 3, 5, 4)
    _tmp70 = einsum('lcdj,abil,dk->abcijk', g_bbbb[o, v, v, o], t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp70
    _tmp71 = einsum('lcdj,ablk,di->abcijk', g_abab[o, v, v, o], t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp71
    _tmp72 = einsum('lcid,ablj,dk->abcijk', g_abab[o, v, o, v], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp72
    t3_res += 1 * _tmp72.transpose(0, 1, 2, 3, 5, 4)
    _tmp73 = einsum('lcid,al,dbjk->abcijk', g_abab[o, v, o, v], t1_aa, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp73
    _tmp74 = einsum('abed,ecij,dk->abcijk', g_abab[v, v, v, v], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp74
    t3_res += 1 * _tmp74.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 1 * _tmp74.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp74.transpose(0, 2, 1, 3, 5, 4)
    _tmp75 = einsum('abde,ecjk,di->abcijk', g_abab[v, v, v, v], t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp75
    t3_res -= 1 * _tmp75.transpose(0, 2, 1, 3, 4, 5)
    _tmp76 = einsum('bcde,aeij,dk->abcijk', g_bbbb[v, v, v, v], t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp76
    t3_res += 1 * _tmp76.transpose(0, 1, 2, 3, 5, 4)
    _tmp77 = einsum('lmdk,abcijm,dl->abcijk', g_abab[o, o, v, o], t3_abbabb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp77
    t3_res += 1 * _tmp77.transpose(0, 1, 2, 3, 5, 4)
    _tmp78 = einsum('mldk,abcijm,dl->abcijk', g_bbbb[o, o, v, o], t3_abbabb, t1_bb, optimize=True)
    t3_res += 1 * _tmp78
    t3_res -= 1 * _tmp78.transpose(0, 1, 2, 3, 5, 4)
    _tmp79 = einsum('mldk,abciml,dj->abcijk', g_bbbb[o, o, v, o], t3_abbabb, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp79
    _tmp80 = einsum('mldk,abcmjl,di->abcijk', g_abab[o, o, v, o], t3_abbabb, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp80
    _tmp81 = einsum('lmdk,abclmj,di->abcijk', g_abab[o, o, v, o], t3_abbabb, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp81
    _tmp82 = einsum('lmdk,al,dbcijm->abcijk', g_abab[o, o, v, o], t1_aa, t3_abbabb, optimize=True)
    t3_res += 1 * _tmp82
    t3_res -= 1 * _tmp82.transpose(0, 1, 2, 3, 5, 4)
    _tmp83 = einsum('mldk,dacimj,bl->abcijk', g_abab[o, o, v, o], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp83
    t3_res += 1 * _tmp83.transpose(0, 1, 2, 3, 5, 4)
    _tmp84 = einsum('mldk,adcijm,bl->abcijk', g_bbbb[o, o, v, o], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp84
    t3_res += 1 * _tmp84.transpose(0, 1, 2, 3, 5, 4)
    _tmp85 = einsum('mldk,dabimj,cl->abcijk', g_abab[o, o, v, o], t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp85
    t3_res -= 1 * _tmp85.transpose(0, 1, 2, 3, 5, 4)
    _tmp86 = einsum('mldk,adbijm,cl->abcijk', g_bbbb[o, o, v, o], t3_abbabb, t1_bb, optimize=True)
    t3_res += 1 * _tmp86
    t3_res -= 1 * _tmp86.transpose(0, 1, 2, 3, 5, 4)
    _tmp87 = einsum('mldj,abciml,dk->abcijk', g_bbbb[o, o, v, o], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp87
    _tmp88 = einsum('mldj,abcmkl,di->abcijk', g_abab[o, o, v, o], t3_abbabb, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp88
    _tmp89 = einsum('lmdj,abclmk,di->abcijk', g_abab[o, o, v, o], t3_abbabb, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp89
    _tmp90 = einsum('mldi,abcmkj,dl->abcijk', g_aaaa[o, o, v, o], t3_abbabb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp90
    _tmp91 = einsum('mlid,abcmkj,dl->abcijk', g_abab[o, o, o, v], t3_abbabb, t1_bb, optimize=True)
    t3_res += 1 * _tmp91
    _tmp92 = einsum('mlid,abcmjl,dk->abcijk', g_abab[o, o, o, v], t3_abbabb, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp92
    t3_res -= 0.5 * _tmp92.transpose(0, 1, 2, 3, 5, 4)
    _tmp93 = einsum('lmid,abclmj,dk->abcijk', g_abab[o, o, o, v], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp93
    t3_res += 0.5 * _tmp93.transpose(0, 1, 2, 3, 5, 4)
    _tmp94 = einsum('mldi,al,dbcmkj->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_abbabb, optimize=True)
    t3_res += 1 * _tmp94
    _tmp95 = einsum('lmid,al,dbcjkm->abcijk', g_abab[o, o, o, v], t1_aa, t3_bbbbbb, optimize=True)
    t3_res -= 1 * _tmp95
    _tmp96 = einsum('mlid,adcmkj,bl->abcijk', g_abab[o, o, o, v], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp96
    _tmp97 = einsum('mlid,adbmkj,cl->abcijk', g_abab[o, o, o, v], t3_abbabb, t1_bb, optimize=True)
    t3_res += 1 * _tmp97
    _tmp98 = einsum('lade,ebcijk,dl->abcijk', g_aaaa[o, v, v, v], t3_abbabb, t1_aa, optimize=True)
    t3_res += 1 * _tmp98
    _tmp99 = einsum('aled,ebcijk,dl->abcijk', g_abab[v, o, v, v], t3_abbabb, t1_bb, optimize=True)
    t3_res += 1 * _tmp99
    _tmp100 = einsum('lbde,aecijk,dl->abcijk', g_abab[o, v, v, v], t3_abbabb, t1_aa, optimize=True)
    t3_res += 1 * _tmp100
    _tmp101 = einsum('lbde,aecijk,dl->abcijk', g_bbbb[o, v, v, v], t3_abbabb, t1_bb, optimize=True)
    t3_res += 1 * _tmp101
    _tmp102 = einsum('aled,ebcijl,dk->abcijk', g_abab[v, o, v, v], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp102
    t3_res += 1 * _tmp102.transpose(0, 1, 2, 3, 5, 4)
    _tmp103 = einsum('lbed,eacilj,dk->abcijk', g_abab[o, v, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res += 1 * _tmp103
    t3_res -= 1 * _tmp103.transpose(0, 1, 2, 3, 5, 4)
    _tmp104 = einsum('lbde,aecijl,dk->abcijk', g_bbbb[o, v, v, v], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp104
    t3_res += 1 * _tmp104.transpose(0, 1, 2, 3, 5, 4)
    _tmp105 = einsum('lade,ebclkj,di->abcijk', g_aaaa[o, v, v, v], t3_abbabb, t1_aa, optimize=True)
    t3_res += 1 * _tmp105
    _tmp106 = einsum('alde,ebcjkl,di->abcijk', g_abab[v, o, v, v], t3_bbbbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp106
    _tmp107 = einsum('lbde,aeclkj,di->abcijk', g_abab[o, v, v, v], t3_abbabb, t1_aa, optimize=True)
    t3_res += 1 * _tmp107
    _tmp108 = einsum('alde,bl,decijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp108
    t3_res += 0.5 * _tmp108.transpose(0, 2, 1, 3, 4, 5)
    t3_res -= 0.5 * _tmp108
    t3_res += 0.5 * _tmp108.transpose(0, 2, 1, 3, 4, 5)
    _tmp109 = einsum('lbde,al,decijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp109
    t3_res -= 0.5 * _tmp109
    _tmp110 = einsum('lbde,aedijk,cl->abcijk', g_bbbb[o, v, v, v], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp110
    _tmp111 = einsum('lcde,aebijk,dl->abcijk', g_abab[o, v, v, v], t3_abbabb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp111
    _tmp112 = einsum('lcde,aebijk,dl->abcijk', g_bbbb[o, v, v, v], t3_abbabb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp112
    _tmp113 = einsum('lced,eabilj,dk->abcijk', g_abab[o, v, v, v], t3_aabaab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp113
    t3_res += 1 * _tmp113.transpose(0, 1, 2, 3, 5, 4)
    _tmp114 = einsum('lcde,aebijl,dk->abcijk', g_bbbb[o, v, v, v], t3_abbabb, t1_bb, optimize=True)
    t3_res += 1 * _tmp114
    t3_res -= 1 * _tmp114.transpose(0, 1, 2, 3, 5, 4)
    _tmp115 = einsum('lcde,aeblkj,di->abcijk', g_abab[o, v, v, v], t3_abbabb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp115
    _tmp116 = einsum('lcde,al,debijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp116
    t3_res += 0.5 * _tmp116
    _tmp117 = einsum('lcde,aedijk,bl->abcijk', g_bbbb[o, v, v, v], t3_abbabb, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp117
    _tmp118 = einsum('lmde,abcijm,delk->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp118
    t3_res += 0.5 * _tmp118.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp118
    t3_res += 0.5 * _tmp118.transpose(0, 1, 2, 3, 5, 4)
    _tmp119 = einsum('mlde,abcijm,dekl->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res -= 0.5 * _tmp119
    t3_res += 0.5 * _tmp119.transpose(0, 1, 2, 3, 5, 4)
    _tmp120 = einsum('mlde,abcmkj,deil->abcijk', g_aaaa[o, o, v, v], t3_abbabb, t2_aaaa, optimize=True)
    t3_res += 0.5 * _tmp120
    _tmp121 = einsum('mlde,abcmkj,deil->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp121
    t3_res += 0.5 * _tmp121
    _tmp122 = einsum('mlde,abciml,dejk->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res += 0.25 * _tmp122
    _tmp123 = einsum('mlde,abcmjl,deik->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 0.25 * _tmp123
    t3_res += 0.25 * _tmp123
    _tmp124 = einsum('lmde,abclmj,deik->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 0.25 * _tmp124
    t3_res -= 0.25 * _tmp124
    _tmp125 = einsum('mlde,abcmkl,deij->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 0.25 * _tmp125
    t3_res -= 0.25 * _tmp125
    _tmp126 = einsum('lmde,abclmk,deij->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 0.25 * _tmp126
    t3_res += 0.25 * _tmp126
    _tmp127 = einsum('mlde,daml,ebcijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp127
    _tmp128 = einsum('mled,adml,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp128
    t3_res -= 0.5 * _tmp128
    _tmp129 = einsum('mlde,aecijk,dbml->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp129
    t3_res -= 0.5 * _tmp129
    _tmp130 = einsum('mlde,aecijk,dbml->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res -= 0.5 * _tmp130
    _tmp131 = einsum('lmed,adlk,ebcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res += 1 * _tmp131
    t3_res -= 1 * _tmp131.transpose(0, 1, 2, 3, 5, 4)
    _tmp132 = einsum('mlde,eacimj,dblk->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp132
    t3_res += 1 * _tmp132.transpose(0, 1, 2, 3, 5, 4)
    _tmp133 = einsum('mled,eacimj,dbkl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp133
    t3_res += 1 * _tmp133.transpose(0, 1, 2, 3, 5, 4)
    _tmp134 = einsum('lmde,aecijm,dblk->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 1 * _tmp134
    t3_res -= 1 * _tmp134.transpose(0, 1, 2, 3, 5, 4)
    _tmp135 = einsum('mlde,aecijm,dbkl->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp135
    t3_res -= 1 * _tmp135.transpose(0, 1, 2, 3, 5, 4)
    _tmp136 = einsum('mlde,dail,ebcmkj->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp136
    _tmp137 = einsum('lmde,dail,ebcjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_bbbbbb, optimize=True)
    t3_res -= 1 * _tmp137
    _tmp138 = einsum('mled,adil,ebcmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res -= 1 * _tmp138
    _tmp139 = einsum('mlde,adil,ebcjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_bbbbbb, optimize=True)
    t3_res -= 1 * _tmp139
    _tmp140 = einsum('mlde,aecmkj,dbil->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 1 * _tmp140
    _tmp141 = einsum('mled,eaciml,dbjk->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_bbbb, optimize=True)
    t3_res -= 0.5 * _tmp141
    t3_res -= 0.5 * _tmp141
    _tmp142 = einsum('mlde,aeciml,dbjk->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res -= 0.5 * _tmp142
    _tmp143 = einsum('mled,adik,ebcmjl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp143
    _tmp144 = einsum('lmed,adik,ebclmj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp144
    _tmp145 = einsum('mlde,adik,ebcjml->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_bbbbbb, optimize=True)
    t3_res -= 0.5 * _tmp145
    _tmp146 = einsum('mlde,eaclmj,dbik->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp146
    _tmp147 = einsum('mlde,aecmjl,dbik->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp147
    _tmp148 = einsum('lmde,aeclmj,dbik->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp148
    _tmp149 = einsum('mled,adij,ebcmkl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp149
    _tmp150 = einsum('lmed,adij,ebclmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp150
    _tmp151 = einsum('mlde,adij,ebckml->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_bbbbbb, optimize=True)
    t3_res += 0.5 * _tmp151
    _tmp152 = einsum('mlde,eaclmk,dbij->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp152
    _tmp153 = einsum('mlde,aecmkl,dbij->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp153
    _tmp154 = einsum('lmde,aeclmk,dbij->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp154
    _tmp155 = einsum('mlde,aebijk,dcml->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp155
    t3_res += 0.5 * _tmp155
    _tmp156 = einsum('mlde,aebijk,dcml->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp156
    _tmp157 = einsum('mlde,eabimj,dclk->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 1 * _tmp157
    t3_res -= 1 * _tmp157.transpose(0, 1, 2, 3, 5, 4)
    _tmp158 = einsum('mled,eabimj,dckl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp158
    t3_res -= 1 * _tmp158.transpose(0, 1, 2, 3, 5, 4)
    _tmp159 = einsum('lmde,aebijm,dclk->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 1 * _tmp159
    t3_res += 1 * _tmp159.transpose(0, 1, 2, 3, 5, 4)
    _tmp160 = einsum('mlde,aebijm,dckl->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp160
    t3_res += 1 * _tmp160.transpose(0, 1, 2, 3, 5, 4)
    _tmp161 = einsum('mlde,aebmkj,dcil->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 1 * _tmp161
    _tmp162 = einsum('mled,eabiml,dcjk->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp162
    t3_res += 0.5 * _tmp162
    _tmp163 = einsum('mlde,aebiml,dcjk->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp163
    _tmp164 = einsum('mlde,eablmj,dcik->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp164
    _tmp165 = einsum('mlde,aebmjl,dcik->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp165
    _tmp166 = einsum('lmde,aeblmj,dcik->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp166
    _tmp167 = einsum('mlde,eablmk,dcij->abcijk', g_aaaa[o, o, v, v], t3_aabaab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp167
    _tmp168 = einsum('mlde,aebmkl,dcij->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp168
    _tmp169 = einsum('lmde,aeblmk,dcij->abcijk', g_abab[o, o, v, v], t3_abbabb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp169
    _tmp170 = einsum('mlde,abml,decijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res += 0.25 * _tmp170
    t3_res -= 0.25 * _tmp170.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.25 * _tmp170
    t3_res -= 0.25 * _tmp170.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.25 * _tmp170
    t3_res -= 0.25 * _tmp170.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.25 * _tmp170
    t3_res -= 0.25 * _tmp170.transpose(0, 2, 1, 3, 4, 5)
    _tmp171 = einsum('mlde,ablk,decimj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=True)
    t3_res -= 0.5 * _tmp171
    t3_res += 0.5 * _tmp171.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.5 * _tmp171.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp171.transpose(0, 2, 1, 3, 5, 4)
    _tmp172 = einsum('lmde,ablk,decijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res -= 0.5 * _tmp172
    t3_res += 0.5 * _tmp172.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.5 * _tmp172.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp172.transpose(0, 2, 1, 3, 5, 4)
    t3_res -= 0.5 * _tmp172
    t3_res += 0.5 * _tmp172.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.5 * _tmp172.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp172.transpose(0, 2, 1, 3, 5, 4)
    _tmp173 = einsum('mlde,abil,decmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp173
    t3_res -= 0.5 * _tmp173.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.5 * _tmp173
    t3_res -= 0.5 * _tmp173.transpose(0, 2, 1, 3, 4, 5)
    _tmp174 = einsum('mlde,abil,decjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_bbbbbb, optimize=True)
    t3_res -= 0.5 * _tmp174
    t3_res += 0.5 * _tmp174.transpose(0, 2, 1, 3, 4, 5)
    _tmp175 = einsum('mlde,aedijk,bcml->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res -= 0.25 * _tmp175
    _tmp176 = einsum('mlde,daeimj,bckl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_bbbb, optimize=True)
    t3_res -= 0.5 * _tmp176
    t3_res += 0.5 * _tmp176.transpose(0, 1, 2, 3, 5, 4)
    _tmp177 = einsum('mled,aedimj,bckl->abcijk', g_abab[o, o, v, v], t3_aabaab, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp177
    t3_res -= 0.5 * _tmp177.transpose(0, 1, 2, 3, 5, 4)
    _tmp178 = einsum('mlde,aedijm,bckl->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp178
    t3_res -= 0.5 * _tmp178.transpose(0, 1, 2, 3, 5, 4)
    _tmp179 = einsum('lmdk,acim,dblj->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp179
    _tmp180 = einsum('mldk,acim,dbjl->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp180
    _tmp181 = einsum('lmdk,dail,bcjm->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp181
    _tmp182 = einsum('mldk,adil,bcjm->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp182
    _tmp183 = einsum('mldk,acmj,dbil->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp183
    _tmp184 = einsum('mldk,adij,bcml->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp184
    t3_res -= 0.5 * _tmp184.transpose(0, 1, 2, 3, 5, 4)
    _tmp185 = einsum('mldk,acml,dbij->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp185
    t3_res -= 0.5 * _tmp185.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 0.5 * _tmp185
    t3_res -= 0.5 * _tmp185.transpose(0, 1, 2, 3, 5, 4)
    _tmp186 = einsum('lmdk,abim,dclj->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp186
    _tmp187 = einsum('mldk,abim,dcjl->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp187
    _tmp188 = einsum('mldk,abmj,dcil->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp188
    _tmp189 = einsum('mldk,abml,dcij->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp189
    t3_res += 0.5 * _tmp189.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp189
    t3_res += 0.5 * _tmp189.transpose(0, 1, 2, 3, 5, 4)
    _tmp190 = einsum('lmdj,acim,dblk->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp190
    _tmp191 = einsum('mldj,acim,dbkl->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp191
    _tmp192 = einsum('lmdj,dail,bckm->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp192
    _tmp193 = einsum('mldj,adil,bckm->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp193
    _tmp194 = einsum('mldj,acmk,dbil->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp194
    _tmp195 = einsum('lmdj,abim,dclk->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp195
    _tmp196 = einsum('mldj,abim,dckl->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp196
    _tmp197 = einsum('mldj,abmk,dcil->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp197
    _tmp198 = einsum('lmid,adlk,bcjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp198
    t3_res -= 1 * _tmp198.transpose(0, 1, 2, 3, 5, 4)
    _tmp199 = einsum('mldi,acmj,dblk->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp199
    t3_res += 1 * _tmp199.transpose(0, 1, 2, 3, 5, 4)
    _tmp200 = einsum('mlid,acmj,dbkl->abcijk', g_abab[o, o, o, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp200
    t3_res += 1 * _tmp200.transpose(0, 1, 2, 3, 5, 4)
    _tmp201 = einsum('mlid,acml,dbjk->abcijk', g_abab[o, o, o, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 0.5 * _tmp201
    t3_res -= 0.5 * _tmp201
    _tmp202 = einsum('mldi,abmj,dclk->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp202
    t3_res -= 1 * _tmp202.transpose(0, 1, 2, 3, 5, 4)
    _tmp203 = einsum('mlid,abmj,dckl->abcijk', g_abab[o, o, o, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp203
    t3_res -= 1 * _tmp203.transpose(0, 1, 2, 3, 5, 4)
    _tmp204 = einsum('mlid,abml,dcjk->abcijk', g_abab[o, o, o, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp204
    t3_res += 0.5 * _tmp204
    _tmp205 = einsum('lbde,acil,dejk->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp205
    _tmp206 = einsum('alde,bcjl,deik->abcijk', g_abab[v, o, v, v], t2_bbbb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp206
    t3_res -= 0.5 * _tmp206
    _tmp207 = einsum('lbde,aclj,deik->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp207
    t3_res += 0.5 * _tmp207
    _tmp208 = einsum('alde,bckl,deij->abcijk', g_abab[v, o, v, v], t2_bbbb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp208
    t3_res += 0.5 * _tmp208
    _tmp209 = einsum('lbde,aclk,deij->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp209
    t3_res -= 0.5 * _tmp209
    _tmp210 = einsum('lade,dblk,ecij->abcijk', g_aaaa[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp210
    t3_res += 1 * _tmp210.transpose(0, 1, 2, 3, 5, 4)
    _tmp211 = einsum('aled,dbkl,ecij->abcijk', g_abab[v, o, v, v], t2_bbbb, t2_abab, optimize=True)
    t3_res += 1 * _tmp211
    t3_res -= 1 * _tmp211.transpose(0, 1, 2, 3, 5, 4)
    _tmp212 = einsum('lbed,adlk,ecij->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp212
    t3_res -= 1 * _tmp212.transpose(0, 1, 2, 3, 5, 4)
    _tmp213 = einsum('alde,dbil,ecjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp213
    _tmp214 = einsum('lbde,dail,ecjk->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp214
    _tmp215 = einsum('lbde,adil,ecjk->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp215
    _tmp216 = einsum('aled,dbjk,ecil->abcijk', g_abab[v, o, v, v], t2_bbbb, t2_abab, optimize=True)
    t3_res += 1 * _tmp216
    _tmp217 = einsum('lade,dbik,eclj->abcijk', g_aaaa[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp217
    _tmp218 = einsum('alde,dbik,ecjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp218
    _tmp219 = einsum('lbed,adik,eclj->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp219
    _tmp220 = einsum('lbde,adik,ecjl->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp220
    _tmp221 = einsum('lade,dbij,eclk->abcijk', g_aaaa[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp221
    _tmp222 = einsum('alde,dbij,eckl->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp222
    _tmp223 = einsum('lbed,adij,eclk->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp223
    _tmp224 = einsum('lbde,adij,eckl->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp224
    _tmp225 = einsum('lcde,abil,dejk->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 0.5 * _tmp225
    _tmp226 = einsum('lcde,ablj,deik->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp226
    t3_res -= 0.5 * _tmp226
    _tmp227 = einsum('lcde,ablk,deij->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp227
    t3_res += 0.5 * _tmp227
    _tmp228 = einsum('lced,adlk,ebij->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp228
    t3_res += 1 * _tmp228.transpose(0, 1, 2, 3, 5, 4)
    _tmp229 = einsum('lcde,dail,ebjk->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp229
    _tmp230 = einsum('lcde,adil,ebjk->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp230
    _tmp231 = einsum('lced,adik,eblj->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp231
    _tmp232 = einsum('lcde,adik,ebjl->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp232
    _tmp233 = einsum('lced,adij,eblk->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp233
    _tmp234 = einsum('lcde,adij,ebkl->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp234
    _tmp235 = einsum('lmde,acim,ebjk,dl->abcijk', g_abab[o, o, v, v], t2_abab, t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp235
    _tmp236 = einsum('mlde,acim,ebjk,dl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp236
    _tmp237 = einsum('lmde,aeik,bcjm,dl->abcijk', g_abab[o, o, v, v], t2_abab, t2_bbbb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp237
    _tmp238 = einsum('mlde,aeik,bcjm,dl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res += 1 * _tmp238
    _tmp239 = einsum('mlde,acmj,ebik,dl->abcijk', g_aaaa[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp239
    _tmp240 = einsum('mled,acmj,ebik,dl->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp240
    _tmp241 = einsum('lmde,aeij,bckm,dl->abcijk', g_abab[o, o, v, v], t2_abab, t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp241
    _tmp242 = einsum('mlde,aeij,bckm,dl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp242
    _tmp243 = einsum('mlde,acmk,ebij,dl->abcijk', g_aaaa[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp243
    _tmp244 = einsum('mled,acmk,ebij,dl->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp244
    _tmp245 = einsum('lmde,abim,ecjk,dl->abcijk', g_abab[o, o, v, v], t2_abab, t2_bbbb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp245
    _tmp246 = einsum('mlde,abim,ecjk,dl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res += 1 * _tmp246
    _tmp247 = einsum('mlde,abmj,ecik,dl->abcijk', g_aaaa[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp247
    _tmp248 = einsum('mled,abmj,ecik,dl->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp248
    _tmp249 = einsum('mlde,abmk,ecij,dl->abcijk', g_aaaa[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp249
    _tmp250 = einsum('mled,abmk,ecij,dl->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp250
    _tmp251 = einsum('lmed,acim,eblj,dk->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp251
    _tmp252 = einsum('mlde,acim,ebjl,dk->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res += 1 * _tmp252
    _tmp253 = einsum('lmed,eail,bcjm,dk->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t1_bb, optimize=True)
    t3_res += 1 * _tmp253
    _tmp254 = einsum('mlde,aeil,bcjm,dk->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp254
    _tmp255 = einsum('mled,acmj,ebil,dk->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp255
    _tmp256 = einsum('mlde,aeij,bcml,dk->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp256
    t3_res += 0.5 * _tmp256.transpose(0, 1, 2, 3, 5, 4)
    _tmp257 = einsum('mled,acml,ebij,dk->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp257
    t3_res -= 0.5 * _tmp257.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 0.5 * _tmp257
    t3_res -= 0.5 * _tmp257.transpose(0, 1, 2, 3, 5, 4)
    _tmp258 = einsum('lmed,abim,eclj,dk->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp258
    _tmp259 = einsum('mlde,abim,ecjl,dk->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp259
    _tmp260 = einsum('mled,abmj,ecil,dk->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp260
    _tmp261 = einsum('mled,abml,ecij,dk->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp261
    t3_res += 0.5 * _tmp261.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 0.5 * _tmp261
    t3_res += 0.5 * _tmp261.transpose(0, 1, 2, 3, 5, 4)
    _tmp262 = einsum('lmed,acim,eblk,dj->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp262
    _tmp263 = einsum('mlde,acim,ebkl,dj->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp263
    _tmp264 = einsum('lmed,eail,bckm,dj->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp264
    _tmp265 = einsum('mlde,aeil,bckm,dj->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res += 1 * _tmp265
    _tmp266 = einsum('mled,acmk,ebil,dj->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp266
    _tmp267 = einsum('lmed,abim,eclk,dj->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp267
    _tmp268 = einsum('mlde,abim,eckl,dj->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res += 1 * _tmp268
    _tmp269 = einsum('mled,abmk,ecil,dj->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp269
    _tmp270 = einsum('lmde,aelk,bcjm,di->abcijk', g_abab[o, o, v, v], t2_abab, t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp270
    t3_res -= 1 * _tmp270.transpose(0, 1, 2, 3, 5, 4)
    _tmp271 = einsum('mlde,acmj,eblk,di->abcijk', g_aaaa[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res += 1 * _tmp271
    t3_res -= 1 * _tmp271.transpose(0, 1, 2, 3, 5, 4)
    _tmp272 = einsum('mlde,acmj,ebkl,di->abcijk', g_abab[o, o, v, v], t2_abab, t2_bbbb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp272
    t3_res += 1 * _tmp272.transpose(0, 1, 2, 3, 5, 4)
    _tmp273 = einsum('mlde,acml,ebjk,di->abcijk', g_abab[o, o, v, v], t2_abab, t2_bbbb, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp273
    t3_res -= 0.5 * _tmp273
    _tmp274 = einsum('mlde,abmj,eclk,di->abcijk', g_aaaa[o, o, v, v], t2_abab, t2_abab, t1_aa, optimize=True)
    t3_res -= 1 * _tmp274
    t3_res += 1 * _tmp274.transpose(0, 1, 2, 3, 5, 4)
    _tmp275 = einsum('mlde,abmj,eckl,di->abcijk', g_abab[o, o, v, v], t2_abab, t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp275
    t3_res -= 1 * _tmp275.transpose(0, 1, 2, 3, 5, 4)
    _tmp276 = einsum('mlde,abml,ecjk,di->abcijk', g_abab[o, o, v, v], t2_abab, t2_bbbb, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp276
    t3_res += 0.5 * _tmp276
    _tmp277 = einsum('mlde,acim,bl,dejk->abcijk', g_bbbb[o, o, v, v], t2_abab, t1_bb, t2_bbbb, optimize=True)
    t3_res -= 0.5 * _tmp277
    _tmp278 = einsum('lmde,al,bcjm,deik->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp278
    t3_res += 0.5 * _tmp278
    _tmp279 = einsum('mlde,acmj,bl,deik->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp279
    t3_res -= 0.5 * _tmp279
    _tmp280 = einsum('lmde,al,bckm,deij->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp280
    t3_res -= 0.5 * _tmp280
    _tmp281 = einsum('mlde,acmk,bl,deij->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp281
    t3_res += 0.5 * _tmp281
    _tmp282 = einsum('mlde,al,dbmk,ecij->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp282
    t3_res -= 1 * _tmp282.transpose(0, 1, 2, 3, 5, 4)
    _tmp283 = einsum('lmed,al,dbkm,ecij->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=True)
    t3_res -= 1 * _tmp283
    t3_res += 1 * _tmp283.transpose(0, 1, 2, 3, 5, 4)
    _tmp284 = einsum('mled,admk,bl,ecij->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t2_abab, optimize=True)
    t3_res -= 1 * _tmp284
    t3_res += 1 * _tmp284.transpose(0, 1, 2, 3, 5, 4)
    _tmp285 = einsum('lmde,al,dbim,ecjk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp285
    _tmp286 = einsum('mlde,daim,bl,ecjk->abcijk', g_abab[o, o, v, v], t2_aaaa, t1_bb, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp286
    _tmp287 = einsum('mlde,adim,bl,ecjk->abcijk', g_bbbb[o, o, v, v], t2_abab, t1_bb, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp287
    _tmp288 = einsum('lmed,al,dbjk,ecim->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=True)
    t3_res -= 1 * _tmp288
    _tmp289 = einsum('mlde,al,dbik,ecmj->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=True)
    t3_res -= 1 * _tmp289
    _tmp290 = einsum('lmde,al,dbik,ecjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp290
    _tmp291 = einsum('mled,adik,bl,ecmj->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t2_abab, optimize=True)
    t3_res += 1 * _tmp291
    _tmp292 = einsum('mlde,adik,bl,ecjm->abcijk', g_bbbb[o, o, v, v], t2_abab, t1_bb, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp292
    _tmp293 = einsum('mlde,al,dbij,ecmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=True)
    t3_res += 1 * _tmp293
    _tmp294 = einsum('lmde,al,dbij,eckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp294
    _tmp295 = einsum('mled,adij,bl,ecmk->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t2_abab, optimize=True)
    t3_res -= 1 * _tmp295
    _tmp296 = einsum('mlde,adij,bl,eckm->abcijk', g_bbbb[o, o, v, v], t2_abab, t1_bb, t2_bbbb, optimize=True)
    t3_res -= 1 * _tmp296
    _tmp297 = einsum('mlde,abim,cl,dejk->abcijk', g_bbbb[o, o, v, v], t2_abab, t1_bb, t2_bbbb, optimize=True)
    t3_res += 0.5 * _tmp297
    _tmp298 = einsum('mlde,abmj,cl,deik->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t2_abab, optimize=True)
    t3_res += 0.5 * _tmp298
    t3_res += 0.5 * _tmp298
    _tmp299 = einsum('mlde,abmk,cl,deij->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t2_abab, optimize=True)
    t3_res -= 0.5 * _tmp299
    t3_res -= 0.5 * _tmp299
    _tmp300 = einsum('mled,admk,ebij,cl->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp300
    t3_res -= 1 * _tmp300.transpose(0, 1, 2, 3, 5, 4)
    _tmp301 = einsum('mlde,daim,ebjk,cl->abcijk', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp301
    _tmp302 = einsum('mlde,adim,ebjk,cl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res += 1 * _tmp302
    _tmp303 = einsum('mled,adik,ebmj,cl->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp303
    _tmp304 = einsum('mlde,adik,ebjm,cl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp304
    _tmp305 = einsum('mled,adij,ebmk,cl->abcijk', g_abab[o, o, v, v], t2_abab, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp305
    _tmp306 = einsum('mlde,adij,ebkm,cl->abcijk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t1_bb, optimize=True)
    t3_res += 1 * _tmp306
    _tmp307 = einsum('mldk,acim,bl,dj->abcijk', g_bbbb[o, o, v, o], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp307
    _tmp308 = einsum('lmdk,al,bcjm,di->abcijk', g_abab[o, o, v, o], t1_aa, t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp308
    _tmp309 = einsum('mldk,acmj,bl,di->abcijk', g_abab[o, o, v, o], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp309
    _tmp310 = einsum('mldk,abim,cl,dj->abcijk', g_bbbb[o, o, v, o], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp310
    _tmp311 = einsum('mldk,abmj,cl,di->abcijk', g_abab[o, o, v, o], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp311
    _tmp312 = einsum('lmdk,al,bm,dcij->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_abab, optimize=True)
    t3_res -= 1 * _tmp312
    t3_res += 1 * _tmp312.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 1 * _tmp312.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp312.transpose(0, 2, 1, 3, 5, 4)
    _tmp313 = einsum('mldk,adij,bl,cm->abcijk', g_bbbb[o, o, v, o], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp313
    t3_res += 1 * _tmp313.transpose(0, 1, 2, 3, 5, 4)
    _tmp314 = einsum('mldj,acim,bl,dk->abcijk', g_bbbb[o, o, v, o], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp314
    _tmp315 = einsum('lmdj,al,bckm,di->abcijk', g_abab[o, o, v, o], t1_aa, t2_bbbb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp315
    _tmp316 = einsum('mldj,acmk,bl,di->abcijk', g_abab[o, o, v, o], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp316
    _tmp317 = einsum('mldj,abim,cl,dk->abcijk', g_bbbb[o, o, v, o], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp317
    _tmp318 = einsum('mldj,abmk,cl,di->abcijk', g_abab[o, o, v, o], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp318
    _tmp319 = einsum('lmid,al,bcjm,dk->abcijk', g_abab[o, o, o, v], t1_aa, t2_bbbb, t1_bb, optimize=True)
    t3_res += 1 * _tmp319
    t3_res -= 1 * _tmp319.transpose(0, 1, 2, 3, 5, 4)
    _tmp320 = einsum('mlid,acmj,bl,dk->abcijk', g_abab[o, o, o, v], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp320
    t3_res += 1 * _tmp320.transpose(0, 1, 2, 3, 5, 4)
    _tmp321 = einsum('mlid,abmj,cl,dk->abcijk', g_abab[o, o, o, v], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp321
    t3_res -= 1 * _tmp321.transpose(0, 1, 2, 3, 5, 4)
    _tmp322 = einsum('lmid,al,bm,dcjk->abcijk', g_abab[o, o, o, v], t1_aa, t1_bb, t2_bbbb, optimize=True)
    t3_res += 1 * _tmp322
    t3_res -= 1 * _tmp322.transpose(0, 2, 1, 3, 4, 5)
    _tmp323 = einsum('lbde,acil,dk,ej->abcijk', g_bbbb[o, v, v, v], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp323
    _tmp324 = einsum('aled,bcjl,dk,ei->abcijk', g_abab[v, o, v, v], t2_bbbb, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp324
    _tmp325 = einsum('lbed,aclj,dk,ei->abcijk', g_abab[o, v, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp325
    _tmp326 = einsum('aled,bl,ecij,dk->abcijk', g_abab[v, o, v, v], t1_bb, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp326
    t3_res -= 1 * _tmp326.transpose(0, 2, 1, 3, 4, 5)
    t3_res -= 1 * _tmp326.transpose(0, 1, 2, 3, 5, 4)
    t3_res += 1 * _tmp326.transpose(0, 2, 1, 3, 5, 4)
    _tmp327 = einsum('aled,bckl,dj,ei->abcijk', g_abab[v, o, v, v], t2_bbbb, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp327
    _tmp328 = einsum('lbed,aclk,dj,ei->abcijk', g_abab[o, v, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp328
    _tmp329 = einsum('alde,bl,ecjk,di->abcijk', g_abab[v, o, v, v], t1_bb, t2_bbbb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp329
    t3_res += 1 * _tmp329.transpose(0, 2, 1, 3, 4, 5)
    _tmp330 = einsum('lbed,al,ecij,dk->abcijk', g_abab[o, v, v, v], t1_aa, t2_abab, t1_bb, optimize=True)
    t3_res += 1 * _tmp330
    t3_res -= 1 * _tmp330.transpose(0, 1, 2, 3, 5, 4)
    _tmp331 = einsum('lbde,aeij,cl,dk->abcijk', g_bbbb[o, v, v, v], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp331
    t3_res += 1 * _tmp331.transpose(0, 1, 2, 3, 5, 4)
    _tmp332 = einsum('lbde,al,ecjk,di->abcijk', g_abab[o, v, v, v], t1_aa, t2_bbbb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp332
    _tmp333 = einsum('lcde,abil,dk,ej->abcijk', g_bbbb[o, v, v, v], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp333
    _tmp334 = einsum('lced,ablj,dk,ei->abcijk', g_abab[o, v, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp334
    _tmp335 = einsum('lced,al,ebij,dk->abcijk', g_abab[o, v, v, v], t1_aa, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp335
    t3_res += 1 * _tmp335.transpose(0, 1, 2, 3, 5, 4)
    _tmp336 = einsum('lcde,aeij,bl,dk->abcijk', g_bbbb[o, v, v, v], t2_abab, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp336
    t3_res -= 1 * _tmp336.transpose(0, 1, 2, 3, 5, 4)
    _tmp337 = einsum('lced,ablk,dj,ei->abcijk', g_abab[o, v, v, v], t2_abab, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp337
    _tmp338 = einsum('lcde,al,ebjk,di->abcijk', g_abab[o, v, v, v], t1_aa, t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp338
    _tmp339 = einsum('lmde,abcijm,dl,ek->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_aa, t1_bb, optimize=True)
    t3_res -= 1 * _tmp339
    t3_res += 1 * _tmp339.transpose(0, 1, 2, 3, 5, 4)
    _tmp340 = einsum('mlde,abcijm,dl,ek->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp340
    t3_res -= 1 * _tmp340.transpose(0, 1, 2, 3, 5, 4)
    _tmp341 = einsum('mlde,abcmkj,dl,ei->abcijk', g_aaaa[o, o, v, v], t3_abbabb, t1_aa, t1_aa, optimize=True)
    t3_res -= 1 * _tmp341
    _tmp342 = einsum('mled,abcmkj,dl,ei->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp342
    _tmp343 = einsum('mlde,am,ebcijk,dl->abcijk', g_aaaa[o, o, v, v], t1_aa, t3_abbabb, t1_aa, optimize=True)
    t3_res += 1 * _tmp343
    _tmp344 = einsum('mled,am,ebcijk,dl->abcijk', g_abab[o, o, v, v], t1_aa, t3_abbabb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp344
    _tmp345 = einsum('lmde,aecijk,bm,dl->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp345
    _tmp346 = einsum('mlde,aecijk,bm,dl->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp346
    _tmp347 = einsum('lmde,aebijk,cm,dl->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp347
    _tmp348 = einsum('mlde,aebijk,cm,dl->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp348
    _tmp349 = einsum('mlde,abciml,dk,ej->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, t1_bb, optimize=True)
    t3_res -= 0.5 * _tmp349
    _tmp350 = einsum('mled,abcmjl,dk,ei->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_bb, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp350
    _tmp351 = einsum('lmed,abclmj,dk,ei->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_bb, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp351
    _tmp352 = einsum('lmed,al,ebcijm,dk->abcijk', g_abab[o, o, v, v], t1_aa, t3_abbabb, t1_bb, optimize=True)
    t3_res += 1 * _tmp352
    t3_res -= 1 * _tmp352.transpose(0, 1, 2, 3, 5, 4)
    _tmp353 = einsum('mled,eacimj,bl,dk->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp353
    t3_res += 1 * _tmp353.transpose(0, 1, 2, 3, 5, 4)
    _tmp354 = einsum('mlde,aecijm,bl,dk->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp354
    t3_res -= 1 * _tmp354.transpose(0, 1, 2, 3, 5, 4)
    _tmp355 = einsum('mled,eabimj,cl,dk->abcijk', g_abab[o, o, v, v], t3_aabaab, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp355
    t3_res -= 1 * _tmp355.transpose(0, 1, 2, 3, 5, 4)
    _tmp356 = einsum('mlde,aebijm,cl,dk->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp356
    t3_res += 1 * _tmp356.transpose(0, 1, 2, 3, 5, 4)
    _tmp357 = einsum('mled,abcmkl,dj,ei->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_bb, t1_aa, optimize=True)
    t3_res -= 0.5 * _tmp357
    _tmp358 = einsum('lmed,abclmk,dj,ei->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_bb, t1_aa, optimize=True)
    t3_res += 0.5 * _tmp358
    _tmp359 = einsum('mlde,al,ebcmkj,di->abcijk', g_aaaa[o, o, v, v], t1_aa, t3_abbabb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp359
    _tmp360 = einsum('lmde,al,ebcjkm,di->abcijk', g_abab[o, o, v, v], t1_aa, t3_bbbbbb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp360
    _tmp361 = einsum('mlde,aecmkj,bl,di->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp361
    _tmp362 = einsum('mlde,aebmkj,cl,di->abcijk', g_abab[o, o, v, v], t3_abbabb, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp362
    _tmp363 = einsum('lmde,al,bm,decijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=True)
    t3_res += 0.5 * _tmp363
    t3_res -= 0.5 * _tmp363.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 0.5 * _tmp363
    t3_res -= 0.5 * _tmp363.transpose(0, 2, 1, 3, 4, 5)
    _tmp364 = einsum('mlde,aedijk,bl,cm->abcijk', g_bbbb[o, o, v, v], t3_abbabb, t1_bb, t1_bb, optimize=True)
    t3_res += 0.5 * _tmp364
    _tmp365 = einsum('mlde,acim,bl,dk,ej->abcijk', g_bbbb[o, o, v, v], t2_abab, t1_bb, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp365
    _tmp366 = einsum('lmed,al,bcjm,dk,ei->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp366
    _tmp367 = einsum('mled,acmj,bl,dk,ei->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp367
    _tmp368 = einsum('mlde,abim,cl,dk,ej->abcijk', g_bbbb[o, o, v, v], t2_abab, t1_bb, t1_bb, t1_bb, optimize=True)
    t3_res -= 1 * _tmp368
    _tmp369 = einsum('mled,abmj,cl,dk,ei->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp369
    _tmp370 = einsum('lmed,al,bm,ecij,dk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t2_abab, t1_bb, optimize=True)
    t3_res -= 1 * _tmp370
    t3_res += 1 * _tmp370.transpose(0, 2, 1, 3, 4, 5)
    t3_res += 1 * _tmp370.transpose(0, 1, 2, 3, 5, 4)
    t3_res -= 1 * _tmp370.transpose(0, 2, 1, 3, 5, 4)
    _tmp371 = einsum('mlde,aeij,bl,cm,dk->abcijk', g_bbbb[o, o, v, v], t2_abab, t1_bb, t1_bb, t1_bb, optimize=True)
    t3_res += 1 * _tmp371
    t3_res -= 1 * _tmp371.transpose(0, 1, 2, 3, 5, 4)
    _tmp372 = einsum('lmed,al,bckm,dj,ei->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp372
    _tmp373 = einsum('mled,acmk,bl,dj,ei->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t1_bb, t1_aa, optimize=True)
    t3_res += 1 * _tmp373
    _tmp374 = einsum('mled,abmk,cl,dj,ei->abcijk', g_abab[o, o, v, v], t2_abab, t1_bb, t1_bb, t1_aa, optimize=True)
    t3_res -= 1 * _tmp374
    _tmp375 = einsum('lmde,al,bm,ecjk,di->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t2_bbbb, t1_aa, optimize=True)
    t3_res += 1 * _tmp375
    t3_res -= 1 * _tmp375.transpose(0, 2, 1, 3, 4, 5)
    return t3_res

