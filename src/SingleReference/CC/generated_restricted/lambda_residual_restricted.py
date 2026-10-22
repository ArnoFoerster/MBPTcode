# GENERATED CODE -- restricted (spin-blocked) Lambda-CCSDT residuals
# (Lambda1/Lambda2/Lambda3, full Lambda-CCSDT) -- see this module's
# generator script's docstring for equations/conventions.
# Do not edit by hand.
# allow numpy built with MKL to consume more threads for tensordot -- but only
# if the caller hasn't already pinned the thread count (same convention as
# amplitudes.py's generalized-pipeline counterpart)
import os
os.environ.setdefault("MKL_NUM_THREADS", "{}".format(os.cpu_count() - 1))

import numpy as np
from src.SingleReference.CC.cached_einsum import einsum


def l1_aa_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb):
    nv, no = t1_aa.shape
    l1_res = np.zeros((nv, no))
    _tmp0 = einsum('me->em', f_aa[o, v])
    l1_res += 1 * _tmp0
    _tmp1 = einsum('imea,ai->em', g_aaaa[o, o, v, v], t1_aa, optimize=True)
    l1_res -= 1 * _tmp1
    _tmp2 = einsum('miea,ai->em', g_abab[o, o, v, v], t1_bb, optimize=True)
    l1_res += 1 * _tmp2
    _tmp3 = einsum('mi,ie->em', f_aa[o, o], l1_aa, optimize=True)
    l1_res -= 1 * _tmp3
    _tmp4 = einsum('ae,ma->em', f_aa[v, v], l1_aa, optimize=True)
    l1_res += 1 * _tmp4
    _tmp5 = einsum('ie,ma,ai->em', f_aa[o, v], l1_aa, t1_aa, optimize=True)
    l1_res -= 1 * _tmp5
    _tmp6 = einsum('ma,ie,ai->em', f_aa[o, v], l1_aa, t1_aa, optimize=True)
    l1_res -= 1 * _tmp6
    _tmp7 = einsum('je,imba,baij->em', f_aa[o, v], l2_aaaa, t2_aaaa, optimize=True)
    l1_res -= 0.5 * _tmp7
    _tmp8 = einsum('je,miba,baji->em', f_aa[o, v], l2_abab, t2_abab, optimize=True)
    l1_res -= 0.5 * _tmp8
    l1_res -= 0.5 * _tmp8
    _tmp9 = einsum('mb,ijea,baij->em', f_aa[o, v], l2_aaaa, t2_aaaa, optimize=True)
    l1_res -= 0.5 * _tmp9
    _tmp10 = einsum('mb,ijea,baij->em', f_aa[o, v], l2_abab, t2_abab, optimize=True)
    l1_res -= 0.5 * _tmp10
    l1_res -= 0.5 * _tmp10
    _tmp11 = einsum('ke,ijmcba,cbaijk->em', f_aa[o, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp11
    _tmp12 = einsum('ke,imjcba,cbaikj->em', f_aa[o, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp12
    l1_res -= 0.0833333333333333 * _tmp12
    l1_res -= 0.0833333333333333 * _tmp12
    _tmp13 = einsum('ke,mjicba,cbakji->em', f_aa[o, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp13
    l1_res -= 0.0833333333333333 * _tmp13
    l1_res -= 0.0833333333333333 * _tmp13
    _tmp14 = einsum('ke,mjicba,cbakji->em', f_aa[o, v], l3_abbabb, t3_abbabb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp14
    l1_res -= 0.0833333333333333 * _tmp14
    l1_res -= 0.0833333333333333 * _tmp14
    _tmp15 = einsum('mc,ijkeba,cbaijk->em', f_aa[o, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp15
    _tmp16 = einsum('mc,ijkeba,cbaijk->em', f_aa[o, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp16
    l1_res -= 0.0833333333333333 * _tmp16
    l1_res -= 0.0833333333333333 * _tmp16
    l1_res -= 0.0833333333333333 * _tmp16
    _tmp17 = einsum('mc,ijkeba,cbaijk->em', f_aa[o, v], l3_abbabb, t3_abbabb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp17
    l1_res -= 0.0833333333333333 * _tmp16
    l1_res -= 0.0833333333333333 * _tmp16
    l1_res -= 0.0833333333333333 * _tmp17
    l1_res -= 0.0833333333333333 * _tmp17
    _tmp18 = einsum('maei,ia->em', g_aaaa[o, v, v, o], l1_aa, optimize=True)
    l1_res += 1 * _tmp18
    _tmp19 = einsum('maei,ia->em', g_abab[o, v, v, o], l1_bb, optimize=True)
    l1_res += 1 * _tmp19
    _tmp20 = einsum('maij,ijae->em', g_aaaa[o, v, o, o], l2_aaaa, optimize=True)
    l1_res += 0.5 * _tmp20
    _tmp21 = einsum('maij,ijea->em', g_abab[o, v, o, o], l2_abab, optimize=True)
    l1_res -= 0.5 * _tmp21
    l1_res -= 0.5 * _tmp21
    _tmp22 = einsum('baei,miba->em', g_aaaa[v, v, v, o], l2_aaaa, optimize=True)
    l1_res += 0.5 * _tmp22
    _tmp23 = einsum('baei,miba->em', g_abab[v, v, v, o], l2_abab, optimize=True)
    l1_res += 0.5 * _tmp23
    l1_res += 0.5 * _tmp23
    _tmp24 = einsum('jmei,ia,aj->em', g_aaaa[o, o, v, o], l1_aa, t1_aa, optimize=True)
    l1_res += 1 * _tmp24
    _tmp25 = einsum('mjei,ia,aj->em', g_abab[o, o, v, o], l1_bb, t1_bb, optimize=True)
    l1_res -= 1 * _tmp25
    _tmp26 = einsum('jmai,ie,aj->em', g_aaaa[o, o, v, o], l1_aa, t1_aa, optimize=True)
    l1_res -= 1 * _tmp26
    _tmp27 = einsum('mjia,ie,aj->em', g_abab[o, o, o, v], l1_aa, t1_bb, optimize=True)
    l1_res -= 1 * _tmp27
    _tmp28 = einsum('maeb,ia,bi->em', g_aaaa[o, v, v, v], l1_aa, t1_aa, optimize=True)
    l1_res += 1 * _tmp28
    _tmp29 = einsum('maeb,ia,bi->em', g_abab[o, v, v, v], l1_bb, t1_bb, optimize=True)
    l1_res += 1 * _tmp29
    _tmp30 = einsum('iaeb,ma,bi->em', g_aaaa[o, v, v, v], l1_aa, t1_aa, optimize=True)
    l1_res -= 1 * _tmp30
    _tmp31 = einsum('aieb,ma,bi->em', g_abab[v, o, v, v], l1_aa, t1_bb, optimize=True)
    l1_res += 1 * _tmp31
    _tmp32 = einsum('kmij,ijea,ak->em', g_aaaa[o, o, o, o], l2_aaaa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp32
    _tmp33 = einsum('mkij,ijea,ak->em', g_abab[o, o, o, o], l2_abab, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp33
    l1_res += 0.5 * _tmp33
    _tmp34 = einsum('jbei,miba,aj->em', g_aaaa[o, v, v, o], l2_aaaa, t1_aa, optimize=True)
    l1_res += 1 * _tmp34
    _tmp35 = einsum('bjei,miba,aj->em', g_abab[v, o, v, o], l2_abab, t1_bb, optimize=True)
    l1_res -= 1 * _tmp35
    _tmp36 = einsum('jbei,miab,aj->em', g_abab[o, v, v, o], l2_abab, t1_aa, optimize=True)
    l1_res -= 1 * _tmp36
    _tmp37 = einsum('mabj,ijae,bi->em', g_aaaa[o, v, v, o], l2_aaaa, t1_aa, optimize=True)
    l1_res += 1 * _tmp37
    _tmp38 = einsum('mabj,ijea,bi->em', g_abab[o, v, v, o], l2_abab, t1_aa, optimize=True)
    l1_res -= 1 * _tmp38
    _tmp39 = einsum('majb,jiea,bi->em', g_abab[o, v, o, v], l2_abab, t1_bb, optimize=True)
    l1_res -= 1 * _tmp39
    _tmp40 = einsum('baec,imba,ci->em', g_aaaa[v, v, v, v], l2_aaaa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp40
    _tmp41 = einsum('baec,miba,ci->em', g_abab[v, v, v, v], l2_abab, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp41
    l1_res += 0.5 * _tmp41
    _tmp42 = einsum('jmeb,ia,baij->em', g_aaaa[o, o, v, v], l1_aa, t2_aaaa, optimize=True)
    l1_res += 1 * _tmp42
    _tmp43 = einsum('mjeb,ia,abij->em', g_abab[o, o, v, v], l1_aa, t2_abab, optimize=True)
    l1_res += 1 * _tmp43
    _tmp44 = einsum('jmeb,ia,baji->em', g_aaaa[o, o, v, v], l1_bb, t2_abab, optimize=True)
    l1_res -= 1 * _tmp44
    _tmp45 = einsum('mjeb,ia,baij->em', g_abab[o, o, v, v], l1_bb, t2_bbbb, optimize=True)
    l1_res -= 1 * _tmp45
    _tmp46 = einsum('jieb,ma,baji->em', g_aaaa[o, o, v, v], l1_aa, t2_aaaa, optimize=True)
    l1_res += 0.5 * _tmp46
    _tmp47 = einsum('jieb,ma,abji->em', g_abab[o, o, v, v], l1_aa, t2_abab, optimize=True)
    l1_res -= 0.5 * _tmp47
    l1_res -= 0.5 * _tmp47
    _tmp48 = einsum('jmab,ie,abij->em', g_aaaa[o, o, v, v], l1_aa, t2_aaaa, optimize=True)
    l1_res += 0.5 * _tmp48
    _tmp49 = einsum('mjab,ie,abij->em', g_abab[o, o, v, v], l1_aa, t2_abab, optimize=True)
    l1_res -= 0.5 * _tmp49
    l1_res -= 0.5 * _tmp49
    _tmp50 = einsum('kmej,ijba,baik->em', g_aaaa[o, o, v, o], l2_aaaa, t2_aaaa, optimize=True)
    l1_res += 0.5 * _tmp50
    _tmp51 = einsum('mkej,ijba,baik->em', g_abab[o, o, v, o], l2_abab, t2_abab, optimize=True)
    l1_res -= 0.5 * _tmp51
    l1_res -= 0.5 * _tmp51
    _tmp52 = einsum('kmej,jiba,baki->em', g_aaaa[o, o, v, o], l2_abab, t2_abab, optimize=True)
    l1_res += 0.5 * _tmp52
    l1_res += 0.5 * _tmp52
    _tmp53 = einsum('mkej,ijba,baik->em', g_abab[o, o, v, o], l2_bbbb, t2_bbbb, optimize=True)
    l1_res -= 0.5 * _tmp53
    _tmp54 = einsum('kjei,miba,bakj->em', g_aaaa[o, o, v, o], l2_aaaa, t2_aaaa, optimize=True)
    l1_res += 0.25 * _tmp54
    _tmp55 = einsum('kjei,miba,bakj->em', g_abab[o, o, v, o], l2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp55
    l1_res += 0.25 * _tmp55
    l1_res += 0.25 * _tmp55
    l1_res += 0.25 * _tmp55
    _tmp56 = einsum('kmbj,ijea,baik->em', g_aaaa[o, o, v, o], l2_aaaa, t2_aaaa, optimize=True)
    l1_res -= 1 * _tmp56
    _tmp57 = einsum('mkjb,ijea,abik->em', g_abab[o, o, o, v], l2_aaaa, t2_abab, optimize=True)
    l1_res += 1 * _tmp57
    _tmp58 = einsum('mkbj,ijea,baik->em', g_abab[o, o, v, o], l2_abab, t2_abab, optimize=True)
    l1_res += 1 * _tmp58
    _tmp59 = einsum('kmbj,jiea,baki->em', g_aaaa[o, o, v, o], l2_abab, t2_abab, optimize=True)
    l1_res -= 1 * _tmp59
    _tmp60 = einsum('mkjb,jiea,baik->em', g_abab[o, o, o, v], l2_abab, t2_bbbb, optimize=True)
    l1_res += 1 * _tmp60
    _tmp61 = einsum('mbec,ijba,caij->em', g_aaaa[o, v, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l1_res += 0.5 * _tmp61
    _tmp62 = einsum('mbec,ijba,caij->em', g_aaaa[o, v, v, v], l2_abab, t2_abab, optimize=True)
    l1_res += 0.5 * _tmp62
    _tmp63 = einsum('mbec,ijab,acij->em', g_abab[o, v, v, v], l2_abab, t2_abab, optimize=True)
    l1_res += 0.5 * _tmp63
    l1_res += 0.5 * _tmp62
    l1_res += 0.5 * _tmp63
    _tmp64 = einsum('mbec,ijba,caij->em', g_abab[o, v, v, v], l2_bbbb, t2_bbbb, optimize=True)
    l1_res += 0.5 * _tmp64
    _tmp65 = einsum('jbec,imba,caij->em', g_aaaa[o, v, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l1_res -= 1 * _tmp65
    _tmp66 = einsum('bjec,imba,acij->em', g_abab[v, o, v, v], l2_aaaa, t2_abab, optimize=True)
    l1_res -= 1 * _tmp66
    _tmp67 = einsum('jbec,miba,caji->em', g_aaaa[o, v, v, v], l2_abab, t2_abab, optimize=True)
    l1_res -= 1 * _tmp67
    _tmp68 = einsum('bjec,miba,caij->em', g_abab[v, o, v, v], l2_abab, t2_bbbb, optimize=True)
    l1_res -= 1 * _tmp68
    _tmp69 = einsum('jbec,miab,acji->em', g_abab[o, v, v, v], l2_abab, t2_abab, optimize=True)
    l1_res -= 1 * _tmp69
    _tmp70 = einsum('mabc,ijae,bcij->em', g_aaaa[o, v, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l1_res += 0.25 * _tmp70
    _tmp71 = einsum('mabc,ijea,bcij->em', g_abab[o, v, v, v], l2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp71
    l1_res -= 0.25 * _tmp71
    l1_res -= 0.25 * _tmp71
    l1_res -= 0.25 * _tmp71
    _tmp72 = einsum('lmjk,ijkeba,bail->em', g_aaaa[o, o, o, o], l3_aaaaaa, t2_aaaa, optimize=True)
    l1_res += 0.25 * _tmp72
    _tmp73 = einsum('mljk,ijkeba,bail->em', g_abab[o, o, o, o], l3_aabaab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp73
    l1_res -= 0.25 * _tmp73
    l1_res -= 0.25 * _tmp73
    l1_res -= 0.25 * _tmp73
    _tmp74 = einsum('lmjk,kjieba,bali->em', g_aaaa[o, o, o, o], l3_aabaab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp74
    l1_res += 0.25 * _tmp74
    _tmp75 = einsum('mljk,jikeba,bail->em', g_abab[o, o, o, o], l3_abbabb, t2_bbbb, optimize=True)
    l1_res += 0.25 * _tmp75
    _tmp76 = einsum('mlkj,kjieba,bail->em', g_abab[o, o, o, o], l3_abbabb, t2_bbbb, optimize=True)
    l1_res -= 0.25 * _tmp76
    _tmp77 = einsum('kcej,imjcba,baik->em', g_aaaa[o, v, v, o], l3_aaaaaa, t2_aaaa, optimize=True)
    l1_res -= 0.5 * _tmp77
    _tmp78 = einsum('ckej,imjcba,baik->em', g_abab[v, o, v, o], l3_aabaab, t2_abab, optimize=True)
    l1_res += 0.5 * _tmp78
    l1_res += 0.5 * _tmp78
    _tmp79 = einsum('kcej,imjabc,baik->em', g_abab[o, v, v, o], l3_aabaab, t2_aaaa, optimize=True)
    l1_res += 0.5 * _tmp79
    _tmp80 = einsum('kcej,jmicba,baki->em', g_aaaa[o, v, v, o], l3_aabaab, t2_abab, optimize=True)
    l1_res -= 0.5 * _tmp80
    l1_res -= 0.5 * _tmp80
    _tmp81 = einsum('ckej,mijcba,baik->em', g_abab[v, o, v, o], l3_abbabb, t2_bbbb, optimize=True)
    l1_res -= 0.5 * _tmp81
    _tmp82 = einsum('kcej,mijbca,baki->em', g_abab[o, v, v, o], l3_abbabb, t2_abab, optimize=True)
    l1_res += 0.5 * _tmp82
    _tmp83 = einsum('kcej,mijabc,abki->em', g_abab[o, v, v, o], l3_abbabb, t2_abab, optimize=True)
    l1_res -= 0.5 * _tmp83
    _tmp84 = einsum('mbck,ijkbea,caij->em', g_aaaa[o, v, v, o], l3_aaaaaa, t2_aaaa, optimize=True)
    l1_res -= 0.5 * _tmp84
    _tmp85 = einsum('mbck,ijkaeb,caij->em', g_abab[o, v, v, o], l3_aabaab, t2_aaaa, optimize=True)
    l1_res += 0.5 * _tmp85
    _tmp86 = einsum('mbck,ikjbea,caij->em', g_aaaa[o, v, v, o], l3_aabaab, t2_abab, optimize=True)
    l1_res += 0.5 * _tmp86
    _tmp87 = einsum('mbkc,ikjaeb,acij->em', g_abab[o, v, o, v], l3_aabaab, t2_abab, optimize=True)
    l1_res -= 0.5 * _tmp87
    _tmp88 = einsum('mbck,ijkeba,caij->em', g_abab[o, v, v, o], l3_abbabb, t2_abab, optimize=True)
    l1_res += 0.5 * _tmp88
    _tmp89 = einsum('mbck,kjibea,caji->em', g_aaaa[o, v, v, o], l3_aabaab, t2_abab, optimize=True)
    l1_res -= 0.5 * _tmp89
    _tmp90 = einsum('mbkc,kjiaeb,acji->em', g_abab[o, v, o, v], l3_aabaab, t2_abab, optimize=True)
    l1_res += 0.5 * _tmp90
    l1_res += 0.5 * _tmp88
    _tmp91 = einsum('mbkc,kjieba,caij->em', g_abab[o, v, o, v], l3_abbabb, t2_bbbb, optimize=True)
    l1_res += 0.5 * _tmp91
    _tmp92 = einsum('cbed,ijmcba,daij->em', g_aaaa[v, v, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l1_res += 0.25 * _tmp92
    _tmp93 = einsum('cbed,imjcba,daij->em', g_aaaa[v, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp93
    _tmp94 = einsum('cbed,imjcab,adij->em', g_abab[v, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp94
    _tmp95 = einsum('bced,imjabc,adij->em', g_abab[v, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp95
    _tmp96 = einsum('cbed,mjicba,daji->em', g_aaaa[v, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp96
    _tmp97 = einsum('cbed,mjicab,adji->em', g_abab[v, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp97
    _tmp98 = einsum('bced,mjiabc,adji->em', g_abab[v, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp98
    _tmp99 = einsum('cbed,mjicba,daij->em', g_abab[v, v, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l1_res -= 0.25 * _tmp99
    l1_res -= 0.25 * _tmp99
    _tmp100 = einsum('kmec,ijba,cbaijk->em', g_aaaa[o, o, v, v], l2_aaaa, t3_aaaaaa, optimize=True)
    l1_res -= 0.25 * _tmp100
    _tmp101 = einsum('mkec,ijba,abcijk->em', g_abab[o, o, v, v], l2_aaaa, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp101
    _tmp102 = einsum('kmec,ijba,cbaikj->em', g_aaaa[o, o, v, v], l2_abab, t3_aabaab, optimize=True)
    l1_res += 0.25 * _tmp102
    _tmp103 = einsum('mkec,ijba,bcaijk->em', g_abab[o, o, v, v], l2_abab, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp103
    l1_res += 0.25 * _tmp102
    _tmp104 = einsum('mkec,ijab,abcijk->em', g_abab[o, o, v, v], l2_abab, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp104
    _tmp105 = einsum('kmec,jiba,cbakji->em', g_aaaa[o, o, v, v], l2_abab, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp105
    l1_res -= 0.25 * _tmp103
    l1_res -= 0.25 * _tmp105
    l1_res += 0.25 * _tmp104
    _tmp106 = einsum('kmec,ijba,cbakji->em', g_aaaa[o, o, v, v], l2_bbbb, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp106
    _tmp107 = einsum('mkec,ijba,cbaijk->em', g_abab[o, o, v, v], l2_bbbb, t3_bbbbbb, optimize=True)
    l1_res += 0.25 * _tmp107
    _tmp108 = einsum('kjec,imba,cbaikj->em', g_aaaa[o, o, v, v], l2_aaaa, t3_aaaaaa, optimize=True)
    l1_res -= 0.25 * _tmp108
    _tmp109 = einsum('kjec,imba,abcikj->em', g_abab[o, o, v, v], l2_aaaa, t3_aabaab, optimize=True)
    l1_res += 0.25 * _tmp109
    l1_res += 0.25 * _tmp109
    _tmp110 = einsum('kjec,miba,cbajki->em', g_aaaa[o, o, v, v], l2_abab, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp110
    _tmp111 = einsum('kjec,miba,bcakij->em', g_abab[o, o, v, v], l2_abab, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp111
    _tmp112 = einsum('jkec,miba,bcajki->em', g_abab[o, o, v, v], l2_abab, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp112
    l1_res -= 0.25 * _tmp110
    _tmp113 = einsum('kjec,miab,abckij->em', g_abab[o, o, v, v], l2_abab, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp113
    _tmp114 = einsum('jkec,miab,abcjki->em', g_abab[o, o, v, v], l2_abab, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp114
    _tmp115 = einsum('kmbc,ijea,bcaijk->em', g_aaaa[o, o, v, v], l2_aaaa, t3_aaaaaa, optimize=True)
    l1_res -= 0.25 * _tmp115
    _tmp116 = einsum('mkbc,ijea,bacijk->em', g_abab[o, o, v, v], l2_aaaa, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp116
    _tmp117 = einsum('mkcb,ijea,acbijk->em', g_abab[o, o, v, v], l2_aaaa, t3_aabaab, optimize=True)
    l1_res += 0.25 * _tmp117
    _tmp118 = einsum('kmbc,ijea,bcaikj->em', g_aaaa[o, o, v, v], l2_abab, t3_aabaab, optimize=True)
    l1_res += 0.25 * _tmp118
    _tmp119 = einsum('mkbc,ijea,bcaijk->em', g_abab[o, o, v, v], l2_abab, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp119
    l1_res += 0.25 * _tmp119
    _tmp120 = einsum('kmbc,jiea,bcakji->em', g_aaaa[o, o, v, v], l2_abab, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp120
    l1_res += 0.25 * _tmp119
    l1_res += 0.25 * _tmp119
    _tmp121 = einsum('lmek,ijkcba,cbaijl->em', g_aaaa[o, o, v, o], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp121
    _tmp122 = einsum('mlek,ijkcba,cbaijl->em', g_abab[o, o, v, o], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp122
    l1_res -= 0.0833333333333333 * _tmp122
    l1_res -= 0.0833333333333333 * _tmp122
    _tmp123 = einsum('lmek,ikjcba,cbailj->em', g_aaaa[o, o, v, o], l3_aabaab, t3_aabaab, optimize=True)
    l1_res += 0.0833333333333333 * _tmp123
    l1_res += 0.0833333333333333 * _tmp123
    l1_res += 0.0833333333333333 * _tmp123
    _tmp124 = einsum('mlek,ijkcba,cbaijl->em', g_abab[o, o, v, o], l3_abbabb, t3_abbabb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp124
    l1_res -= 0.0833333333333333 * _tmp124
    l1_res -= 0.0833333333333333 * _tmp124
    _tmp125 = einsum('lmek,kjicba,cbalji->em', g_aaaa[o, o, v, o], l3_aabaab, t3_aabaab, optimize=True)
    l1_res += 0.0833333333333333 * _tmp125
    l1_res += 0.0833333333333333 * _tmp125
    l1_res += 0.0833333333333333 * _tmp125
    l1_res -= 0.0833333333333333 * _tmp124
    l1_res -= 0.0833333333333333 * _tmp124
    l1_res -= 0.0833333333333333 * _tmp124
    _tmp126 = einsum('lmek,kjicba,cbalji->em', g_aaaa[o, o, v, o], l3_abbabb, t3_abbabb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp126
    l1_res += 0.0833333333333333 * _tmp126
    l1_res += 0.0833333333333333 * _tmp126
    _tmp127 = einsum('mlek,ijkcba,cbaijl->em', g_abab[o, o, v, o], l3_bbbbbb, t3_bbbbbb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp127
    _tmp128 = einsum('lkej,imjcba,cbailk->em', g_aaaa[o, o, v, o], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp128
    _tmp129 = einsum('lkej,imjcba,cbailk->em', g_abab[o, o, v, o], l3_aabaab, t3_aabaab, optimize=True)
    l1_res += 0.0833333333333333 * _tmp129
    l1_res += 0.0833333333333333 * _tmp129
    l1_res += 0.0833333333333333 * _tmp129
    l1_res += 0.0833333333333333 * _tmp129
    l1_res += 0.0833333333333333 * _tmp129
    l1_res += 0.0833333333333333 * _tmp129
    _tmp130 = einsum('lkej,jmicba,cbakli->em', g_aaaa[o, o, v, o], l3_aabaab, t3_aabaab, optimize=True)
    l1_res += 0.0833333333333333 * _tmp130
    l1_res += 0.0833333333333333 * _tmp130
    l1_res += 0.0833333333333333 * _tmp130
    _tmp131 = einsum('lkej,mijcba,cbalik->em', g_abab[o, o, v, o], l3_abbabb, t3_abbabb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp131
    _tmp132 = einsum('klej,mijcba,cbakli->em', g_abab[o, o, v, o], l3_abbabb, t3_abbabb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp132
    l1_res += 0.0833333333333333 * _tmp131
    l1_res -= 0.0833333333333333 * _tmp132
    l1_res += 0.0833333333333333 * _tmp131
    l1_res -= 0.0833333333333333 * _tmp132
    _tmp133 = einsum('lmck,ijkeba,cbaijl->em', g_aaaa[o, o, v, o], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l1_res -= 0.25 * _tmp133
    _tmp134 = einsum('mlkc,ijkeba,abcijl->em', g_abab[o, o, o, v], l3_aaaaaa, t3_aabaab, optimize=True)
    l1_res += 0.25 * _tmp134
    _tmp135 = einsum('mlck,ijkeba,cbaijl->em', g_abab[o, o, v, o], l3_aabaab, t3_aabaab, optimize=True)
    l1_res += 0.25 * _tmp135
    l1_res += 0.25 * _tmp135
    _tmp136 = einsum('lmck,ikjeba,cbailj->em', g_aaaa[o, o, v, o], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp136
    _tmp137 = einsum('mlkc,ikjeba,bcaijl->em', g_abab[o, o, o, v], l3_aabaab, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp137
    l1_res -= 0.25 * _tmp136
    _tmp138 = einsum('mlkc,ikjeab,abcijl->em', g_abab[o, o, o, v], l3_aabaab, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp138
    _tmp139 = einsum('mlck,ijkeba,cbaijl->em', g_abab[o, o, v, o], l3_abbabb, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp139
    _tmp140 = einsum('lmck,kjieba,cbalji->em', g_aaaa[o, o, v, o], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp140
    _tmp141 = einsum('mlkc,kjieba,bcajil->em', g_abab[o, o, o, v], l3_aabaab, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp141
    l1_res -= 0.25 * _tmp140
    _tmp142 = einsum('mlkc,kjieab,abcjil->em', g_abab[o, o, o, v], l3_aabaab, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp142
    l1_res += 0.25 * _tmp139
    _tmp143 = einsum('lmck,kjieba,cbalji->em', g_aaaa[o, o, v, o], l3_abbabb, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp143
    _tmp144 = einsum('mlkc,kjieba,cbaijl->em', g_abab[o, o, o, v], l3_abbabb, t3_bbbbbb, optimize=True)
    l1_res += 0.25 * _tmp144
    _tmp145 = einsum('mced,ijkcba,dbaijk->em', g_aaaa[o, v, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp145
    _tmp146 = einsum('mced,ijkcba,dbaijk->em', g_aaaa[o, v, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res += 0.0833333333333333 * _tmp146
    l1_res += 0.0833333333333333 * _tmp146
    _tmp147 = einsum('mced,ijkabc,abdijk->em', g_abab[o, v, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res += 0.0833333333333333 * _tmp147
    l1_res += 0.0833333333333333 * _tmp146
    l1_res += 0.0833333333333333 * _tmp146
    l1_res += 0.0833333333333333 * _tmp147
    _tmp148 = einsum('mced,ijkcba,dbaijk->em', g_aaaa[o, v, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp148
    _tmp149 = einsum('mced,ijkbca,bdaijk->em', g_abab[o, v, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp149
    _tmp150 = einsum('mced,ijkabc,abdijk->em', g_abab[o, v, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp150
    l1_res += 0.0833333333333333 * _tmp146
    l1_res += 0.0833333333333333 * _tmp146
    l1_res += 0.0833333333333333 * _tmp147
    l1_res += 0.0833333333333333 * _tmp148
    l1_res += 0.0833333333333333 * _tmp149
    l1_res += 0.0833333333333333 * _tmp150
    l1_res += 0.0833333333333333 * _tmp148
    l1_res += 0.0833333333333333 * _tmp149
    l1_res += 0.0833333333333333 * _tmp150
    _tmp151 = einsum('mced,ijkcba,dbaijk->em', g_abab[o, v, v, v], l3_bbbbbb, t3_bbbbbb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp151
    _tmp152 = einsum('kced,ijmcba,dbaijk->em', g_aaaa[o, v, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l1_res -= 0.25 * _tmp152
    _tmp153 = einsum('cked,ijmcba,abdijk->em', g_abab[v, o, v, v], l3_aaaaaa, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp153
    _tmp154 = einsum('kced,imjcba,dbaikj->em', g_aaaa[o, v, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp154
    _tmp155 = einsum('cked,imjcba,bdaijk->em', g_abab[v, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp155
    l1_res -= 0.25 * _tmp154
    _tmp156 = einsum('cked,imjcab,abdijk->em', g_abab[v, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp156
    _tmp157 = einsum('kced,imjabc,abdikj->em', g_abab[o, v, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp157
    _tmp158 = einsum('kced,mjicba,dbakji->em', g_aaaa[o, v, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp158
    _tmp159 = einsum('cked,mjicba,bdajik->em', g_abab[v, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp159
    l1_res -= 0.25 * _tmp158
    _tmp160 = einsum('cked,mjicab,abdjik->em', g_abab[v, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l1_res += 0.25 * _tmp160
    _tmp161 = einsum('kced,mjiabc,abdkji->em', g_abab[o, v, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.25 * _tmp161
    _tmp162 = einsum('kced,mjicba,dbakji->em', g_aaaa[o, v, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp162
    _tmp163 = einsum('cked,mjicba,dbaijk->em', g_abab[v, o, v, v], l3_abbabb, t3_bbbbbb, optimize=True)
    l1_res -= 0.25 * _tmp163
    _tmp164 = einsum('kced,mjibca,bdakji->em', g_abab[o, v, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp164
    _tmp165 = einsum('kced,mjiabc,abdkji->em', g_abab[o, v, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l1_res -= 0.25 * _tmp165
    _tmp166 = einsum('mbcd,ijkbea,cdaijk->em', g_aaaa[o, v, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp166
    _tmp167 = einsum('mbcd,ijkbea,cdaijk->em', g_aaaa[o, v, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res += 0.0833333333333333 * _tmp167
    _tmp168 = einsum('mbcd,ijkaeb,cadijk->em', g_abab[o, v, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res += 0.0833333333333333 * _tmp168
    _tmp169 = einsum('mbdc,ijkaeb,adcijk->em', g_abab[o, v, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp169
    l1_res += 0.0833333333333333 * _tmp167
    l1_res += 0.0833333333333333 * _tmp168
    l1_res -= 0.0833333333333333 * _tmp169
    _tmp170 = einsum('mbcd,ijkeba,cdaijk->em', g_abab[o, v, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp170
    l1_res -= 0.0833333333333333 * _tmp170
    l1_res += 0.0833333333333333 * _tmp167
    l1_res += 0.0833333333333333 * _tmp168
    l1_res -= 0.0833333333333333 * _tmp169
    l1_res -= 0.0833333333333333 * _tmp170
    l1_res -= 0.0833333333333333 * _tmp170
    l1_res -= 0.0833333333333333 * _tmp170
    l1_res -= 0.0833333333333333 * _tmp170
    _tmp171 = einsum('kmec,ijba,baik,cj->em', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp171
    _tmp172 = einsum('mkec,ijba,baik,cj->em', g_abab[o, o, v, v], l2_abab, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp172
    l1_res -= 0.5 * _tmp172
    _tmp173 = einsum('kmec,jiba,baki,cj->em', g_aaaa[o, o, v, v], l2_abab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp173
    l1_res += 0.5 * _tmp173
    _tmp174 = einsum('mkec,ijba,baik,cj->em', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp174
    _tmp175 = einsum('kmec,ijba,caij,bk->em', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp175
    _tmp176 = einsum('kmec,ijba,caij,bk->em', g_aaaa[o, o, v, v], l2_abab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp176
    _tmp177 = einsum('mkec,ijab,acij,bk->em', g_abab[o, o, v, v], l2_abab, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp177
    l1_res += 0.5 * _tmp176
    l1_res -= 0.5 * _tmp177
    _tmp178 = einsum('mkec,ijba,caij,bk->em', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp178
    _tmp179 = einsum('kjec,imba,baik,cj->em', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp179
    _tmp180 = einsum('kjec,imba,baik,cj->em', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp180
    _tmp181 = einsum('kjec,miba,baki,cj->em', g_aaaa[o, o, v, v], l2_abab, t2_abab, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp181
    _tmp182 = einsum('kjec,miba,baki,cj->em', g_abab[o, o, v, v], l2_abab, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp182
    l1_res -= 0.5 * _tmp181
    l1_res -= 0.5 * _tmp182
    _tmp183 = einsum('kjec,imba,bakj,ci->em', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp183
    _tmp184 = einsum('kjec,miba,bakj,ci->em', g_abab[o, o, v, v], l2_abab, t2_abab, t1_bb, optimize=True)
    l1_res += 0.25 * _tmp184
    l1_res += 0.25 * _tmp184
    l1_res += 0.25 * _tmp184
    l1_res += 0.25 * _tmp184
    _tmp185 = einsum('kjec,imba,caik,bj->em', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 1 * _tmp185
    _tmp186 = einsum('jkec,imba,acik,bj->em', g_abab[o, o, v, v], l2_aaaa, t2_abab, t1_aa, optimize=True)
    l1_res += 1 * _tmp186
    _tmp187 = einsum('kjec,miba,caki,bj->em', g_aaaa[o, o, v, v], l2_abab, t2_abab, t1_aa, optimize=True)
    l1_res += 1 * _tmp187
    _tmp188 = einsum('jkec,miba,caik,bj->em', g_abab[o, o, v, v], l2_abab, t2_bbbb, t1_aa, optimize=True)
    l1_res += 1 * _tmp188
    _tmp189 = einsum('kjec,miab,acki,bj->em', g_abab[o, o, v, v], l2_abab, t2_abab, t1_bb, optimize=True)
    l1_res += 1 * _tmp189
    _tmp190 = einsum('kmbc,ijea,caij,bk->em', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp190
    _tmp191 = einsum('mkcb,ijea,caij,bk->em', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp191
    _tmp192 = einsum('kmbc,ijea,caij,bk->em', g_aaaa[o, o, v, v], l2_abab, t2_abab, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp192
    _tmp193 = einsum('mkcb,ijea,caij,bk->em', g_abab[o, o, v, v], l2_abab, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp193
    l1_res -= 0.5 * _tmp192
    l1_res -= 0.5 * _tmp193
    _tmp194 = einsum('kmbc,ijea,caik,bj->em', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 1 * _tmp194
    _tmp195 = einsum('mkbc,ijea,acik,bj->em', g_abab[o, o, v, v], l2_aaaa, t2_abab, t1_aa, optimize=True)
    l1_res += 1 * _tmp195
    _tmp196 = einsum('mkcb,ijea,caik,bj->em', g_abab[o, o, v, v], l2_abab, t2_abab, t1_bb, optimize=True)
    l1_res += 1 * _tmp196
    _tmp197 = einsum('kmbc,jiea,caki,bj->em', g_aaaa[o, o, v, v], l2_abab, t2_abab, t1_aa, optimize=True)
    l1_res += 1 * _tmp197
    _tmp198 = einsum('mkbc,jiea,caik,bj->em', g_abab[o, o, v, v], l2_abab, t2_bbbb, t1_aa, optimize=True)
    l1_res += 1 * _tmp198
    _tmp199 = einsum('kmbc,ijea,ak,bcij->em', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp199
    _tmp200 = einsum('mkbc,ijea,ak,bcij->em', g_abab[o, o, v, v], l2_abab, t1_bb, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp200
    l1_res += 0.25 * _tmp200
    l1_res += 0.25 * _tmp200
    l1_res += 0.25 * _tmp200
    _tmp201 = einsum('lkej,imjcba,bail,ck->em', g_aaaa[o, o, v, o], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp201
    _tmp202 = einsum('klej,imjcba,bail,ck->em', g_abab[o, o, v, o], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp202
    l1_res -= 0.5 * _tmp202
    _tmp203 = einsum('lkej,imjabc,bail,ck->em', g_abab[o, o, v, o], l3_aabaab, t2_aaaa, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp203
    _tmp204 = einsum('lkej,jmicba,bali,ck->em', g_aaaa[o, o, v, o], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp204
    l1_res += 0.5 * _tmp204
    _tmp205 = einsum('klej,mijcba,bail,ck->em', g_abab[o, o, v, o], l3_abbabb, t2_bbbb, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp205
    _tmp206 = einsum('lkej,mijbca,bali,ck->em', g_abab[o, o, v, o], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp206
    _tmp207 = einsum('lkej,mijabc,abli,ck->em', g_abab[o, o, v, o], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp207
    _tmp208 = einsum('lmck,ijkeba,bail,cj->em', g_aaaa[o, o, v, o], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp208
    _tmp209 = einsum('mlck,ijkeba,bail,cj->em', g_abab[o, o, v, o], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp209
    l1_res -= 0.5 * _tmp209
    _tmp210 = einsum('mlkc,ikjeba,bail,cj->em', g_abab[o, o, o, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp210
    l1_res -= 0.5 * _tmp210
    _tmp211 = einsum('lmck,kjieba,bali,cj->em', g_aaaa[o, o, v, o], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp211
    l1_res += 0.5 * _tmp211
    _tmp212 = einsum('mlck,jikeba,bail,cj->em', g_abab[o, o, v, o], l3_abbabb, t2_bbbb, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp212
    _tmp213 = einsum('mlkc,kjieba,bail,cj->em', g_abab[o, o, o, v], l3_abbabb, t2_bbbb, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp213
    _tmp214 = einsum('lmck,ijkeba,caij,bl->em', g_aaaa[o, o, v, o], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp214
    _tmp215 = einsum('mlck,ijkeab,caij,bl->em', g_abab[o, o, v, o], l3_aabaab, t2_aaaa, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp215
    _tmp216 = einsum('lmck,ikjeba,caij,bl->em', g_aaaa[o, o, v, o], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp216
    _tmp217 = einsum('mlkc,ikjeab,acij,bl->em', g_abab[o, o, o, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp217
    _tmp218 = einsum('mlck,ijkeba,caij,bl->em', g_abab[o, o, v, o], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp218
    _tmp219 = einsum('lmck,kjieba,caji,bl->em', g_aaaa[o, o, v, o], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp219
    _tmp220 = einsum('mlkc,kjieab,acji,bl->em', g_abab[o, o, o, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp220
    l1_res -= 0.5 * _tmp218
    _tmp221 = einsum('mlkc,kjieba,caij,bl->em', g_abab[o, o, o, v], l3_abbabb, t2_bbbb, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp221
    _tmp222 = einsum('kced,ijmcba,baik,dj->em', g_aaaa[o, v, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp222
    _tmp223 = einsum('cked,imjcba,baik,dj->em', g_abab[v, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp223
    l1_res += 0.5 * _tmp223
    _tmp224 = einsum('kced,imjabc,baik,dj->em', g_abab[o, v, v, v], l3_aabaab, t2_aaaa, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp224
    _tmp225 = einsum('kced,mjicba,baki,dj->em', g_aaaa[o, v, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp225
    l1_res += 0.5 * _tmp225
    _tmp226 = einsum('cked,mjicba,baik,dj->em', g_abab[v, o, v, v], l3_abbabb, t2_bbbb, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp226
    _tmp227 = einsum('kced,mjibca,baki,dj->em', g_abab[o, v, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp227
    _tmp228 = einsum('kced,mjiabc,abki,dj->em', g_abab[o, v, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp228
    _tmp229 = einsum('kced,ijmcba,daij,bk->em', g_aaaa[o, v, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp229
    _tmp230 = einsum('kced,imjcba,daij,bk->em', g_aaaa[o, v, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp230
    _tmp231 = einsum('cked,imjcab,adij,bk->em', g_abab[v, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp231
    _tmp232 = einsum('kced,imjabc,adij,bk->em', g_abab[o, v, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp232
    _tmp233 = einsum('kced,mjicba,daji,bk->em', g_aaaa[o, v, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp233
    _tmp234 = einsum('cked,mjicab,adji,bk->em', g_abab[v, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp234
    _tmp235 = einsum('kced,mjiabc,adji,bk->em', g_abab[o, v, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp235
    _tmp236 = einsum('cked,mjicba,daij,bk->em', g_abab[v, o, v, v], l3_abbabb, t2_bbbb, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp236
    _tmp237 = einsum('kced,mjibca,daij,bk->em', g_abab[o, v, v, v], l3_abbabb, t2_bbbb, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp237
    _tmp238 = einsum('mbcd,ijkbea,daij,ck->em', g_aaaa[o, v, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp238
    _tmp239 = einsum('mbdc,ijkaeb,daij,ck->em', g_abab[o, v, v, v], l3_aabaab, t2_aaaa, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp239
    _tmp240 = einsum('mbcd,ikjbea,daij,ck->em', g_aaaa[o, v, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp240
    _tmp241 = einsum('mbcd,ikjaeb,adij,ck->em', g_abab[o, v, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp241
    _tmp242 = einsum('mbdc,ijkeba,daij,ck->em', g_abab[o, v, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp242
    _tmp243 = einsum('mbcd,kjibea,daji,ck->em', g_aaaa[o, v, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp243
    _tmp244 = einsum('mbcd,kjiaeb,adji,ck->em', g_abab[o, v, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp244
    l1_res += 0.5 * _tmp242
    _tmp245 = einsum('mbcd,kjieba,daij,ck->em', g_abab[o, v, v, v], l3_abbabb, t2_bbbb, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp245
    _tmp246 = einsum('lmed,ijkcba,cbaijl,dk->em', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp246
    _tmp247 = einsum('mled,ijkcba,cbaijl,dk->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp247
    l1_res -= 0.0833333333333333 * _tmp247
    l1_res -= 0.0833333333333333 * _tmp247
    _tmp248 = einsum('lmed,ikjcba,cbailj,dk->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp248
    l1_res += 0.0833333333333333 * _tmp248
    l1_res += 0.0833333333333333 * _tmp248
    _tmp249 = einsum('mled,ijkcba,cbaijl,dk->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp249
    l1_res -= 0.0833333333333333 * _tmp249
    l1_res -= 0.0833333333333333 * _tmp249
    _tmp250 = einsum('lmed,kjicba,cbalji,dk->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp250
    l1_res += 0.0833333333333333 * _tmp250
    l1_res += 0.0833333333333333 * _tmp250
    l1_res -= 0.0833333333333333 * _tmp249
    l1_res -= 0.0833333333333333 * _tmp249
    l1_res -= 0.0833333333333333 * _tmp249
    _tmp251 = einsum('lmed,kjicba,cbalji,dk->em', g_aaaa[o, o, v, v], l3_abbabb, t3_abbabb, t1_aa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp251
    l1_res += 0.0833333333333333 * _tmp251
    l1_res += 0.0833333333333333 * _tmp251
    _tmp252 = einsum('mled,ijkcba,cbaijl,dk->em', g_abab[o, o, v, v], l3_bbbbbb, t3_bbbbbb, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp252
    _tmp253 = einsum('lmed,ijkcba,dbaijk,cl->em', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp253
    _tmp254 = einsum('lmed,ijkcba,dbaijk,cl->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp254
    l1_res += 0.0833333333333333 * _tmp254
    _tmp255 = einsum('mled,ijkabc,abdijk,cl->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp255
    l1_res += 0.0833333333333333 * _tmp254
    l1_res += 0.0833333333333333 * _tmp254
    l1_res -= 0.0833333333333333 * _tmp255
    _tmp256 = einsum('lmed,ijkcba,dbaijk,cl->em', g_aaaa[o, o, v, v], l3_abbabb, t3_abbabb, t1_aa, optimize=True)
    l1_res += 0.0833333333333333 * _tmp256
    _tmp257 = einsum('mled,ijkbca,bdaijk,cl->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp257
    _tmp258 = einsum('mled,ijkabc,abdijk,cl->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp258
    l1_res += 0.0833333333333333 * _tmp254
    l1_res += 0.0833333333333333 * _tmp254
    l1_res -= 0.0833333333333333 * _tmp255
    l1_res += 0.0833333333333333 * _tmp256
    l1_res -= 0.0833333333333333 * _tmp257
    l1_res -= 0.0833333333333333 * _tmp258
    l1_res += 0.0833333333333333 * _tmp256
    l1_res -= 0.0833333333333333 * _tmp257
    l1_res -= 0.0833333333333333 * _tmp258
    _tmp259 = einsum('mled,ijkcba,dbaijk,cl->em', g_abab[o, o, v, v], l3_bbbbbb, t3_bbbbbb, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp259
    _tmp260 = einsum('lked,ijmcba,cbaijl,dk->em', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp260
    _tmp261 = einsum('lked,ijmcba,cbaijl,dk->em', g_abab[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp261
    _tmp262 = einsum('lked,imjcba,cbailj,dk->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp262
    _tmp263 = einsum('lked,imjcba,cbailj,dk->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp263
    l1_res -= 0.0833333333333333 * _tmp262
    l1_res -= 0.0833333333333333 * _tmp263
    l1_res -= 0.0833333333333333 * _tmp262
    l1_res -= 0.0833333333333333 * _tmp263
    _tmp264 = einsum('lked,mjicba,cbalji,dk->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp264
    _tmp265 = einsum('lked,mjicba,cbalji,dk->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp265
    l1_res -= 0.0833333333333333 * _tmp264
    l1_res -= 0.0833333333333333 * _tmp265
    l1_res -= 0.0833333333333333 * _tmp264
    l1_res -= 0.0833333333333333 * _tmp265
    _tmp266 = einsum('lked,mjicba,cbalji,dk->em', g_aaaa[o, o, v, v], l3_abbabb, t3_abbabb, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp266
    _tmp267 = einsum('lked,mjicba,cbalji,dk->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp267
    l1_res -= 0.0833333333333333 * _tmp266
    l1_res -= 0.0833333333333333 * _tmp267
    l1_res -= 0.0833333333333333 * _tmp266
    l1_res -= 0.0833333333333333 * _tmp267
    _tmp268 = einsum('lked,ijmcba,cbailk,dj->em', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp268
    _tmp269 = einsum('lked,imjcba,cbailk,dj->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp269
    l1_res += 0.0833333333333333 * _tmp269
    l1_res += 0.0833333333333333 * _tmp269
    l1_res += 0.0833333333333333 * _tmp269
    l1_res += 0.0833333333333333 * _tmp269
    l1_res += 0.0833333333333333 * _tmp269
    _tmp270 = einsum('lked,mjicba,cbakli,dj->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp270
    l1_res -= 0.0833333333333333 * _tmp270
    l1_res -= 0.0833333333333333 * _tmp270
    _tmp271 = einsum('lked,mjicba,cbalik,dj->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp271
    _tmp272 = einsum('kled,mjicba,cbakli,dj->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp272
    l1_res -= 0.0833333333333333 * _tmp271
    l1_res += 0.0833333333333333 * _tmp272
    l1_res -= 0.0833333333333333 * _tmp271
    l1_res += 0.0833333333333333 * _tmp272
    _tmp273 = einsum('lked,ijmcba,dbaijl,ck->em', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp273
    _tmp274 = einsum('kled,ijmcba,abdijl,ck->em', g_abab[o, o, v, v], l3_aaaaaa, t3_aabaab, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp274
    _tmp275 = einsum('lked,imjcba,dbailj,ck->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp275
    _tmp276 = einsum('kled,imjcba,bdaijl,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp276
    l1_res += 0.25 * _tmp275
    _tmp277 = einsum('kled,imjcab,abdijl,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp277
    _tmp278 = einsum('lked,imjabc,abdilj,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res += 0.25 * _tmp278
    _tmp279 = einsum('lked,mjicba,dbalji,ck->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp279
    _tmp280 = einsum('kled,mjicba,bdajil,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp280
    l1_res += 0.25 * _tmp279
    _tmp281 = einsum('kled,mjicab,abdjil,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp281
    _tmp282 = einsum('lked,mjiabc,abdlji,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res += 0.25 * _tmp282
    _tmp283 = einsum('lked,mjicba,dbalji,ck->em', g_aaaa[o, o, v, v], l3_abbabb, t3_abbabb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp283
    _tmp284 = einsum('kled,mjicba,dbaijl,ck->em', g_abab[o, o, v, v], l3_abbabb, t3_bbbbbb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp284
    _tmp285 = einsum('lked,mjibca,bdalji,ck->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res += 0.25 * _tmp285
    _tmp286 = einsum('lked,mjiabc,abdlji,ck->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res += 0.25 * _tmp286
    _tmp287 = einsum('lmcd,ijkeba,dbaijk,cl->em', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp287
    _tmp288 = einsum('mldc,ijkeba,dbaijk,cl->em', g_abab[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp288
    _tmp289 = einsum('lmcd,ijkeba,dbaijk,cl->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp289
    _tmp290 = einsum('mldc,ijkeba,dbaijk,cl->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp290
    l1_res -= 0.0833333333333333 * _tmp289
    l1_res -= 0.0833333333333333 * _tmp290
    l1_res -= 0.0833333333333333 * _tmp289
    l1_res -= 0.0833333333333333 * _tmp290
    l1_res -= 0.0833333333333333 * _tmp289
    l1_res -= 0.0833333333333333 * _tmp290
    _tmp291 = einsum('lmcd,ijkeba,dbaijk,cl->em', g_aaaa[o, o, v, v], l3_abbabb, t3_abbabb, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp291
    _tmp292 = einsum('mldc,ijkeba,dbaijk,cl->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp292
    l1_res -= 0.0833333333333333 * _tmp289
    l1_res -= 0.0833333333333333 * _tmp290
    l1_res -= 0.0833333333333333 * _tmp289
    l1_res -= 0.0833333333333333 * _tmp290
    l1_res -= 0.0833333333333333 * _tmp291
    l1_res -= 0.0833333333333333 * _tmp292
    l1_res -= 0.0833333333333333 * _tmp291
    l1_res -= 0.0833333333333333 * _tmp292
    _tmp293 = einsum('lmcd,ijkeba,dbaijl,ck->em', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp293
    _tmp294 = einsum('mlcd,ijkeba,abdijl,ck->em', g_abab[o, o, v, v], l3_aaaaaa, t3_aabaab, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp294
    _tmp295 = einsum('mldc,ijkeba,dbaijl,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res += 0.25 * _tmp295
    l1_res += 0.25 * _tmp295
    _tmp296 = einsum('lmcd,ikjeba,dbailj,ck->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp296
    _tmp297 = einsum('mlcd,ikjeba,bdaijl,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp297
    l1_res += 0.25 * _tmp296
    _tmp298 = einsum('mlcd,ikjeab,abdijl,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp298
    _tmp299 = einsum('mldc,ijkeba,dbaijl,ck->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res += 0.25 * _tmp299
    _tmp300 = einsum('lmcd,kjieba,dbalji,ck->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp300
    _tmp301 = einsum('mlcd,kjieba,bdajil,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp301
    l1_res += 0.25 * _tmp300
    _tmp302 = einsum('mlcd,kjieab,abdjil,ck->em', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp302
    l1_res += 0.25 * _tmp299
    _tmp303 = einsum('lmcd,kjieba,dbalji,ck->em', g_aaaa[o, o, v, v], l3_abbabb, t3_abbabb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp303
    _tmp304 = einsum('mlcd,kjieba,dbaijl,ck->em', g_abab[o, o, v, v], l3_abbabb, t3_bbbbbb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp304
    _tmp305 = einsum('lmcd,ijkeba,cdaijk,bl->em', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp305
    _tmp306 = einsum('lmcd,ijkeba,cdaijk,bl->em', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp306
    _tmp307 = einsum('mlcd,ijkeab,cadijk,bl->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp307
    _tmp308 = einsum('mldc,ijkeab,adcijk,bl->em', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, t1_bb, optimize=True)
    l1_res -= 0.0833333333333333 * _tmp308
    l1_res -= 0.0833333333333333 * _tmp306
    l1_res += 0.0833333333333333 * _tmp307
    l1_res -= 0.0833333333333333 * _tmp308
    _tmp309 = einsum('mlcd,ijkeba,cdaijk,bl->em', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, t1_bb, optimize=True)
    l1_res += 0.0833333333333333 * _tmp309
    l1_res += 0.0833333333333333 * _tmp309
    l1_res -= 0.0833333333333333 * _tmp306
    l1_res += 0.0833333333333333 * _tmp307
    l1_res -= 0.0833333333333333 * _tmp308
    l1_res += 0.0833333333333333 * _tmp309
    l1_res += 0.0833333333333333 * _tmp309
    l1_res += 0.0833333333333333 * _tmp309
    l1_res += 0.0833333333333333 * _tmp309
    _tmp310 = einsum('jmeb,ia,aj,bi->em', g_aaaa[o, o, v, v], l1_aa, t1_aa, t1_aa, optimize=True)
    l1_res += 1 * _tmp310
    _tmp311 = einsum('mjeb,ia,aj,bi->em', g_abab[o, o, v, v], l1_bb, t1_bb, t1_bb, optimize=True)
    l1_res -= 1 * _tmp311
    _tmp312 = einsum('jieb,ma,aj,bi->em', g_aaaa[o, o, v, v], l1_aa, t1_aa, t1_aa, optimize=True)
    l1_res -= 1 * _tmp312
    _tmp313 = einsum('jieb,ma,aj,bi->em', g_abab[o, o, v, v], l1_aa, t1_aa, t1_bb, optimize=True)
    l1_res -= 1 * _tmp313
    _tmp314 = einsum('jmab,ie,aj,bi->em', g_aaaa[o, o, v, v], l1_aa, t1_aa, t1_aa, optimize=True)
    l1_res -= 1 * _tmp314
    _tmp315 = einsum('mjba,ie,aj,bi->em', g_abab[o, o, v, v], l1_aa, t1_bb, t1_aa, optimize=True)
    l1_res -= 1 * _tmp315
    _tmp316 = einsum('kjei,miba,ak,bj->em', g_aaaa[o, o, v, o], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp316
    _tmp317 = einsum('jkei,miba,ak,bj->em', g_abab[o, o, v, o], l2_abab, t1_bb, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp317
    l1_res += 0.5 * _tmp317
    _tmp318 = einsum('kmbj,ijea,ak,bi->em', g_aaaa[o, o, v, o], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l1_res -= 1 * _tmp318
    _tmp319 = einsum('mkbj,ijea,ak,bi->em', g_abab[o, o, v, o], l2_abab, t1_bb, t1_aa, optimize=True)
    l1_res += 1 * _tmp319
    _tmp320 = einsum('mkjb,jiea,ak,bi->em', g_abab[o, o, o, v], l2_abab, t1_bb, t1_bb, optimize=True)
    l1_res += 1 * _tmp320
    _tmp321 = einsum('jbec,imba,aj,ci->em', g_aaaa[o, v, v, v], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l1_res -= 1 * _tmp321
    _tmp322 = einsum('bjec,miba,aj,ci->em', g_abab[v, o, v, v], l2_abab, t1_bb, t1_bb, optimize=True)
    l1_res -= 1 * _tmp322
    _tmp323 = einsum('jbec,miab,aj,ci->em', g_abab[o, v, v, v], l2_abab, t1_aa, t1_bb, optimize=True)
    l1_res -= 1 * _tmp323
    _tmp324 = einsum('mabc,ijae,bj,ci->em', g_aaaa[o, v, v, v], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp324
    _tmp325 = einsum('macb,ijea,bj,ci->em', g_abab[o, v, v, v], l2_abab, t1_bb, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp325
    l1_res -= 0.5 * _tmp325
    _tmp326 = einsum('lmed,ijkcba,bail,dcjk->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t2_aaaa, optimize=True)
    l1_res += 0.25 * _tmp326
    _tmp327 = einsum('mled,ijkcba,bail,cdjk->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp327
    l1_res += 0.25 * _tmp327
    _tmp328 = einsum('lmed,ijkabc,bail,dcjk->em', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp328
    l1_res += 0.25 * _tmp327
    l1_res += 0.25 * _tmp327
    l1_res -= 0.25 * _tmp328
    _tmp329 = einsum('mled,ijkbca,bail,dcjk->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t2_bbbb, optimize=True)
    l1_res += 0.25 * _tmp329
    _tmp330 = einsum('mled,ijkabc,abil,dcjk->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t2_bbbb, optimize=True)
    l1_res -= 0.25 * _tmp330
    _tmp331 = einsum('lmed,kjicba,bali,dcjk->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t2_aaaa, optimize=True)
    l1_res += 0.25 * _tmp331
    l1_res += 0.25 * _tmp331
    _tmp332 = einsum('mled,jikcba,bail,cdjk->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp332
    _tmp333 = einsum('lmed,jikbca,bali,dcjk->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp333
    _tmp334 = einsum('lmed,jikabc,abli,dcjk->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp334
    _tmp335 = einsum('mled,kjicba,bail,cdkj->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp335
    _tmp336 = einsum('lmed,kjibca,bali,dckj->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp336
    _tmp337 = einsum('lmed,kjiabc,abli,dckj->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp337
    _tmp338 = einsum('mled,ijkcba,bail,dcjk->em', g_abab[o, o, v, v], l3_bbbbbb, t2_bbbb, t2_bbbb, optimize=True)
    l1_res -= 0.25 * _tmp338
    _tmp339 = einsum('lked,ijmcba,bail,dcjk->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp339
    _tmp340 = einsum('lked,ijmcba,bail,cdjk->em', g_abab[o, o, v, v], l3_aaaaaa, t2_aaaa, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp340
    _tmp341 = einsum('kled,imjcba,bail,cdkj->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp341
    l1_res -= 0.25 * _tmp341
    _tmp342 = einsum('lked,imjabc,bail,dckj->em', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp342
    _tmp343 = einsum('lked,imjabc,bail,dcjk->em', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, t2_bbbb, optimize=True)
    l1_res -= 0.25 * _tmp343
    _tmp344 = einsum('lked,mjicba,bali,dcjk->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp344
    _tmp345 = einsum('lked,mjicba,bali,cdjk->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp345
    l1_res -= 0.25 * _tmp344
    l1_res += 0.25 * _tmp345
    _tmp346 = einsum('kled,mjicba,bail,cdkj->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp346
    _tmp347 = einsum('lked,mjibca,bali,dckj->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp347
    _tmp348 = einsum('lked,mjibca,bali,dcjk->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t2_bbbb, optimize=True)
    l1_res += 0.25 * _tmp348
    _tmp349 = einsum('lked,mjiabc,abli,dckj->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp349
    _tmp350 = einsum('lked,mjiabc,abli,dcjk->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t2_bbbb, optimize=True)
    l1_res -= 0.25 * _tmp350
    _tmp351 = einsum('lked,ijmcba,balk,dcij->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t2_aaaa, optimize=True)
    l1_res += 0.125 * _tmp351
    _tmp352 = einsum('lked,imjcba,balk,cdij->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res += 0.125 * _tmp352
    l1_res += 0.125 * _tmp352
    l1_res += 0.125 * _tmp352
    l1_res += 0.125 * _tmp352
    _tmp353 = einsum('lked,imjabc,balk,dcij->em', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t2_abab, optimize=True)
    l1_res += 0.125 * _tmp353
    _tmp354 = einsum('lked,mjicba,balk,cdji->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.125 * _tmp354
    l1_res -= 0.125 * _tmp354
    l1_res -= 0.125 * _tmp354
    l1_res -= 0.125 * _tmp354
    _tmp355 = einsum('lked,mjiabc,balk,dcji->em', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t2_abab, optimize=True)
    l1_res -= 0.125 * _tmp355
    _tmp356 = einsum('lked,mjibca,balk,dcij->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t2_bbbb, optimize=True)
    l1_res += 0.125 * _tmp356
    l1_res += 0.125 * _tmp356
    _tmp357 = einsum('lked,mjiabc,ablk,dcij->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t2_bbbb, optimize=True)
    l1_res -= 0.125 * _tmp357
    l1_res -= 0.125 * _tmp357
    _tmp358 = einsum('lked,ijmcba,dail,cbjk->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp358
    _tmp359 = einsum('kled,ijmcba,adil,cbjk->em', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp359
    _tmp360 = einsum('lked,imjcab,dail,cbkj->em', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp360
    _tmp361 = einsum('kled,imjcab,adil,cbkj->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp361
    _tmp362 = einsum('lked,imjabc,dail,bckj->em', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp362
    _tmp363 = einsum('kled,imjabc,adil,bckj->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp363
    _tmp364 = einsum('lked,mjicba,dali,cbjk->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp364
    _tmp365 = einsum('kled,mjicba,dail,cbjk->em', g_abab[o, o, v, v], l3_aabaab, t2_bbbb, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp365
    _tmp366 = einsum('lked,mjicab,adli,cbjk->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp366
    _tmp367 = einsum('lked,mjiabc,adli,bcjk->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp367
    _tmp368 = einsum('lked,mjicba,dali,cbkj->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp368
    _tmp369 = einsum('kled,mjicba,dail,cbkj->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp369
    l1_res += 0.25 * _tmp368
    l1_res += 0.25 * _tmp369
    _tmp370 = einsum('lked,mjiabc,adli,cbjk->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t2_bbbb, optimize=True)
    l1_res -= 0.25 * _tmp370
    _tmp371 = einsum('lmcd,ijkeba,bail,cdjk->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t2_aaaa, optimize=True)
    l1_res += 0.125 * _tmp371
    _tmp372 = einsum('mlcd,ijkeba,bail,cdjk->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.125 * _tmp372
    l1_res -= 0.125 * _tmp372
    l1_res -= 0.125 * _tmp372
    l1_res -= 0.125 * _tmp372
    l1_res -= 0.125 * _tmp372
    l1_res -= 0.125 * _tmp372
    l1_res -= 0.125 * _tmp372
    l1_res -= 0.125 * _tmp372
    _tmp373 = einsum('lmcd,kjieba,bali,cdjk->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t2_aaaa, optimize=True)
    l1_res += 0.125 * _tmp373
    l1_res += 0.125 * _tmp373
    _tmp374 = einsum('mlcd,jikeba,bail,cdjk->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t2_abab, optimize=True)
    l1_res += 0.125 * _tmp374
    l1_res += 0.125 * _tmp374
    _tmp375 = einsum('mlcd,kjieba,bail,cdkj->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t2_abab, optimize=True)
    l1_res -= 0.125 * _tmp375
    l1_res -= 0.125 * _tmp375
    _tmp376 = einsum('lmcd,ijkeba,daij,cbkl->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp376
    _tmp377 = einsum('mldc,ijkeba,daij,bckl->em', g_abab[o, o, v, v], l3_aaaaaa, t2_aaaa, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp377
    _tmp378 = einsum('lmcd,ijkeab,daij,cblk->em', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp378
    _tmp379 = einsum('mldc,ijkeab,daij,cbkl->em', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, t2_bbbb, optimize=True)
    l1_res += 0.25 * _tmp379
    _tmp380 = einsum('lmcd,ikjeba,daij,cbkl->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t2_aaaa, optimize=True)
    l1_res += 0.25 * _tmp380
    _tmp381 = einsum('mldc,ikjeba,daij,bckl->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp381
    _tmp382 = einsum('mlcd,ikjeab,adij,cbkl->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp382
    _tmp383 = einsum('lmcd,ijkeba,daij,cblk->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp383
    _tmp384 = einsum('mldc,ijkeba,daij,cbkl->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t2_bbbb, optimize=True)
    l1_res -= 0.25 * _tmp384
    _tmp385 = einsum('lmcd,kjieba,daji,cbkl->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp385
    _tmp386 = einsum('mldc,kjieba,daji,bckl->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp386
    _tmp387 = einsum('mlcd,kjieab,adji,cbkl->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp387
    l1_res += 0.25 * _tmp383
    l1_res -= 0.25 * _tmp384
    _tmp388 = einsum('mlcd,kjieba,daij,cbkl->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp388
    _tmp389 = einsum('lmcd,ijkeba,dail,cbjk->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp389
    _tmp390 = einsum('mlcd,ijkeba,adil,cbjk->em', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp390
    l1_res += 0.25 * _tmp387
    _tmp391 = einsum('lmcd,ijkeab,dail,cbjk->em', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp391
    l1_res += 0.25 * _tmp386
    l1_res += 0.25 * _tmp387
    l1_res += 0.25 * _tmp391
    l1_res += 0.25 * _tmp386
    _tmp392 = einsum('mldc,ijkeba,dail,cbjk->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t2_bbbb, optimize=True)
    l1_res -= 0.25 * _tmp392
    _tmp393 = einsum('lmcd,kjieba,dali,cbjk->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp393
    _tmp394 = einsum('mlcd,kjieba,dail,cbjk->em', g_abab[o, o, v, v], l3_aabaab, t2_bbbb, t2_aaaa, optimize=True)
    l1_res -= 0.25 * _tmp394
    _tmp395 = einsum('lmcd,jikeba,dali,cbjk->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp395
    _tmp396 = einsum('mlcd,jikeba,dail,cbjk->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t2_abab, optimize=True)
    l1_res -= 0.25 * _tmp396
    _tmp397 = einsum('lmcd,kjieba,dali,cbkj->em', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp397
    _tmp398 = einsum('mlcd,kjieba,dail,cbkj->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t2_abab, optimize=True)
    l1_res += 0.25 * _tmp398
    _tmp399 = einsum('lked,ijmcba,bail,ck,dj->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp399
    _tmp400 = einsum('kled,imjcba,bail,ck,dj->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp400
    l1_res -= 0.5 * _tmp400
    _tmp401 = einsum('lked,imjabc,bail,ck,dj->em', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, t1_bb, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp401
    _tmp402 = einsum('lked,mjicba,bali,ck,dj->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp402
    l1_res -= 0.5 * _tmp402
    _tmp403 = einsum('kled,mjicba,bail,ck,dj->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_aa, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp403
    _tmp404 = einsum('lked,mjibca,bali,ck,dj->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp404
    _tmp405 = einsum('lked,mjiabc,abli,ck,dj->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp405
    _tmp406 = einsum('lked,ijmcba,daij,bl,ck->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp406
    _tmp407 = einsum('lked,imjcba,daij,bl,ck->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp407
    _tmp408 = einsum('kled,imjcab,adij,bl,ck->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp408
    _tmp409 = einsum('lked,imjabc,adij,bl,ck->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, t1_bb, optimize=True)
    l1_res += 0.25 * _tmp409
    _tmp410 = einsum('lked,mjicba,daji,bl,ck->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp410
    _tmp411 = einsum('kled,mjicab,adji,bl,ck->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp411
    _tmp412 = einsum('lked,mjiabc,adji,bl,ck->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, t1_bb, optimize=True)
    l1_res -= 0.25 * _tmp412
    _tmp413 = einsum('kled,mjicba,daij,bl,ck->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_bb, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp413
    l1_res -= 0.25 * _tmp413
    _tmp414 = einsum('lmcd,ijkeba,bail,ck,dj->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp414
    _tmp415 = einsum('mldc,ijkeba,bail,ck,dj->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp415
    l1_res -= 0.25 * _tmp415
    l1_res -= 0.25 * _tmp415
    l1_res -= 0.25 * _tmp415
    _tmp416 = einsum('lmcd,kjieba,bali,ck,dj->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.25 * _tmp416
    l1_res -= 0.25 * _tmp416
    _tmp417 = einsum('mldc,jikeba,bail,ck,dj->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_bb, t1_aa, optimize=True)
    l1_res += 0.25 * _tmp417
    _tmp418 = einsum('mlcd,kjieba,bail,ck,dj->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_aa, t1_bb, optimize=True)
    l1_res -= 0.25 * _tmp418
    _tmp419 = einsum('lmcd,ijkeba,daij,bl,ck->em', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp419
    _tmp420 = einsum('mldc,ijkeab,daij,bl,ck->em', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, t1_bb, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp420
    _tmp421 = einsum('lmcd,ikjeba,daij,bl,ck->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp421
    _tmp422 = einsum('mlcd,ikjeab,adij,bl,ck->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp422
    _tmp423 = einsum('mldc,ijkeba,daij,bl,ck->em', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, t1_bb, optimize=True)
    l1_res -= 0.5 * _tmp423
    _tmp424 = einsum('lmcd,kjieba,daji,bl,ck->em', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp424
    _tmp425 = einsum('mlcd,kjieab,adji,bl,ck->em', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp425
    l1_res -= 0.5 * _tmp423
    _tmp426 = einsum('mlcd,kjieba,daij,bl,ck->em', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_bb, t1_aa, optimize=True)
    l1_res -= 0.5 * _tmp426
    _tmp427 = einsum('kjec,imba,ak,bj,ci->em', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, t1_aa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp427
    _tmp428 = einsum('jkec,miba,ak,bj,ci->em', g_abab[o, o, v, v], l2_abab, t1_bb, t1_aa, t1_bb, optimize=True)
    l1_res += 0.5 * _tmp428
    l1_res += 0.5 * _tmp428
    _tmp429 = einsum('kmbc,ijea,ak,bj,ci->em', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, t1_aa, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp429
    _tmp430 = einsum('mkcb,ijea,ak,bj,ci->em', g_abab[o, o, v, v], l2_abab, t1_bb, t1_bb, t1_aa, optimize=True)
    l1_res += 0.5 * _tmp430
    l1_res += 0.5 * _tmp430
    return l1_res


def l2_aaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb):
    nv, no = t1_aa.shape
    l2_res = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('mnef->efnm', g_aaaa[o, o, v, v])
    l2_res += 1 * _tmp0
    _tmp1 = einsum('ne,mf->efnm', f_aa[o, v], l1_aa, optimize=True)
    l2_res -= 1 * _tmp1
    l2_res += 1 * _tmp1.transpose(1, 0, 2, 3)
    l2_res += 1 * _tmp1.transpose(0, 1, 3, 2)
    l2_res -= 1 * _tmp1.transpose(1, 0, 3, 2)
    _tmp2 = einsum('ni,mief->efnm', f_aa[o, o], l2_aaaa, optimize=True)
    l2_res -= 1 * _tmp2
    l2_res += 1 * _tmp2.transpose(0, 1, 3, 2)
    _tmp3 = einsum('ae,mnaf->efnm', f_aa[v, v], l2_aaaa, optimize=True)
    l2_res += 1 * _tmp3
    l2_res -= 1 * _tmp3.transpose(1, 0, 2, 3)
    _tmp4 = einsum('ie,mnfa,ai->efnm', f_aa[o, v], l2_aaaa, t1_aa, optimize=True)
    l2_res += 1 * _tmp4
    l2_res -= 1 * _tmp4.transpose(1, 0, 2, 3)
    _tmp5 = einsum('na,imef,ai->efnm', f_aa[o, v], l2_aaaa, t1_aa, optimize=True)
    l2_res += 1 * _tmp5
    l2_res -= 1 * _tmp5.transpose(0, 1, 3, 2)
    _tmp6 = einsum('je,imnfba,baij->efnm', f_aa[o, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp6
    l2_res += 0.5 * _tmp6.transpose(1, 0, 2, 3)
    _tmp7 = einsum('je,nmifba,baji->efnm', f_aa[o, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp7
    l2_res += 0.5 * _tmp7.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp7
    l2_res += 0.5 * _tmp7.transpose(1, 0, 2, 3)
    _tmp8 = einsum('nb,ijmefa,baij->efnm', f_aa[o, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp8
    l2_res += 0.5 * _tmp8.transpose(0, 1, 3, 2)
    _tmp9 = einsum('nb,imjefa,baij->efnm', f_aa[o, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp9
    l2_res -= 0.5 * _tmp9.transpose(0, 1, 3, 2)
    _tmp10 = einsum('nb,mjiefa,baji->efnm', f_aa[o, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp10
    l2_res += 0.5 * _tmp10.transpose(0, 1, 3, 2)
    _tmp11 = einsum('mnei,if->efnm', g_aaaa[o, o, v, o], l1_aa, optimize=True)
    l2_res -= 1 * _tmp11
    l2_res += 1 * _tmp11.transpose(1, 0, 2, 3)
    _tmp12 = einsum('naef,ma->efnm', g_aaaa[o, v, v, v], l1_aa, optimize=True)
    l2_res -= 1 * _tmp12
    l2_res += 1 * _tmp12.transpose(0, 1, 3, 2)
    _tmp13 = einsum('mnij,ijef->efnm', g_aaaa[o, o, o, o], l2_aaaa, optimize=True)
    l2_res += 0.5 * _tmp13
    _tmp14 = einsum('naei,miaf->efnm', g_aaaa[o, v, v, o], l2_aaaa, optimize=True)
    l2_res += 1 * _tmp14
    l2_res -= 1 * _tmp14.transpose(1, 0, 2, 3)
    l2_res -= 1 * _tmp14.transpose(0, 1, 3, 2)
    l2_res += 1 * _tmp14.transpose(1, 0, 3, 2)
    _tmp15 = einsum('naei,mifa->efnm', g_abab[o, v, v, o], l2_abab, optimize=True)
    l2_res -= 1 * _tmp15
    l2_res += 1 * _tmp15.transpose(1, 0, 2, 3)
    l2_res += 1 * _tmp15.transpose(0, 1, 3, 2)
    l2_res -= 1 * _tmp15.transpose(1, 0, 3, 2)
    _tmp16 = einsum('baef,mnba->efnm', g_aaaa[v, v, v, v], l2_aaaa, optimize=True)
    l2_res += 0.5 * _tmp16
    _tmp17 = einsum('naij,mijaef->efnm', g_aaaa[o, v, o, o], l3_aaaaaa, optimize=True)
    l2_res -= 0.5 * _tmp17
    l2_res += 0.5 * _tmp17.transpose(0, 1, 3, 2)
    _tmp18 = einsum('naij,mijfea->efnm', g_abab[o, v, o, o], l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp18
    l2_res -= 0.5 * _tmp18.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp18
    l2_res -= 0.5 * _tmp18.transpose(0, 1, 3, 2)
    _tmp19 = einsum('baei,mnibaf->efnm', g_aaaa[v, v, v, o], l3_aaaaaa, optimize=True)
    l2_res -= 0.5 * _tmp19
    l2_res += 0.5 * _tmp19.transpose(1, 0, 2, 3)
    _tmp20 = einsum('baei,mnibfa->efnm', g_abab[v, v, v, o], l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp20
    l2_res -= 0.5 * _tmp20.transpose(1, 0, 2, 3)
    _tmp21 = einsum('abei,mnifab->efnm', g_abab[v, v, v, o], l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp21
    l2_res += 0.5 * _tmp21.transpose(1, 0, 2, 3)
    _tmp22 = einsum('inef,ma,ai->efnm', g_aaaa[o, o, v, v], l1_aa, t1_aa, optimize=True)
    l2_res -= 1 * _tmp22
    l2_res += 1 * _tmp22.transpose(0, 1, 3, 2)
    _tmp23 = einsum('mnea,if,ai->efnm', g_aaaa[o, o, v, v], l1_aa, t1_aa, optimize=True)
    l2_res -= 1 * _tmp23
    l2_res += 1 * _tmp23.transpose(1, 0, 2, 3)
    _tmp24 = einsum('inea,mf,ai->efnm', g_aaaa[o, o, v, v], l1_aa, t1_aa, optimize=True)
    l2_res += 1 * _tmp24
    l2_res -= 1 * _tmp24.transpose(1, 0, 2, 3)
    l2_res -= 1 * _tmp24.transpose(0, 1, 3, 2)
    l2_res += 1 * _tmp24.transpose(1, 0, 3, 2)
    _tmp25 = einsum('niea,mf,ai->efnm', g_abab[o, o, v, v], l1_aa, t1_bb, optimize=True)
    l2_res -= 1 * _tmp25
    l2_res += 1 * _tmp25.transpose(1, 0, 2, 3)
    l2_res += 1 * _tmp25.transpose(0, 1, 3, 2)
    l2_res -= 1 * _tmp25.transpose(1, 0, 3, 2)
    _tmp26 = einsum('jnei,mifa,aj->efnm', g_aaaa[o, o, v, o], l2_aaaa, t1_aa, optimize=True)
    l2_res -= 1 * _tmp26
    l2_res += 1 * _tmp26.transpose(1, 0, 2, 3)
    l2_res += 1 * _tmp26.transpose(0, 1, 3, 2)
    l2_res -= 1 * _tmp26.transpose(1, 0, 3, 2)
    _tmp27 = einsum('njei,mifa,aj->efnm', g_abab[o, o, v, o], l2_abab, t1_bb, optimize=True)
    l2_res += 1 * _tmp27
    l2_res -= 1 * _tmp27.transpose(1, 0, 2, 3)
    l2_res -= 1 * _tmp27.transpose(0, 1, 3, 2)
    l2_res += 1 * _tmp27.transpose(1, 0, 3, 2)
    _tmp28 = einsum('mnaj,ijef,ai->efnm', g_aaaa[o, o, v, o], l2_aaaa, t1_aa, optimize=True)
    l2_res += 1 * _tmp28
    _tmp29 = einsum('jnai,mief,aj->efnm', g_aaaa[o, o, v, o], l2_aaaa, t1_aa, optimize=True)
    l2_res -= 1 * _tmp29
    l2_res += 1 * _tmp29.transpose(0, 1, 3, 2)
    _tmp30 = einsum('njia,mief,aj->efnm', g_abab[o, o, o, v], l2_aaaa, t1_bb, optimize=True)
    l2_res -= 1 * _tmp30
    l2_res += 1 * _tmp30.transpose(0, 1, 3, 2)
    _tmp31 = einsum('ibef,mnba,ai->efnm', g_aaaa[o, v, v, v], l2_aaaa, t1_aa, optimize=True)
    l2_res += 1 * _tmp31
    _tmp32 = einsum('naeb,imaf,bi->efnm', g_aaaa[o, v, v, v], l2_aaaa, t1_aa, optimize=True)
    l2_res -= 1 * _tmp32
    l2_res += 1 * _tmp32.transpose(1, 0, 2, 3)
    l2_res += 1 * _tmp32.transpose(0, 1, 3, 2)
    l2_res -= 1 * _tmp32.transpose(1, 0, 3, 2)
    _tmp33 = einsum('naeb,mifa,bi->efnm', g_abab[o, v, v, v], l2_abab, t1_bb, optimize=True)
    l2_res -= 1 * _tmp33
    l2_res += 1 * _tmp33.transpose(1, 0, 2, 3)
    l2_res += 1 * _tmp33.transpose(0, 1, 3, 2)
    l2_res -= 1 * _tmp33.transpose(1, 0, 3, 2)
    _tmp34 = einsum('iaeb,mnaf,bi->efnm', g_aaaa[o, v, v, v], l2_aaaa, t1_aa, optimize=True)
    l2_res -= 1 * _tmp34
    l2_res += 1 * _tmp34.transpose(1, 0, 2, 3)
    _tmp35 = einsum('aieb,mnaf,bi->efnm', g_abab[v, o, v, v], l2_aaaa, t1_bb, optimize=True)
    l2_res += 1 * _tmp35
    l2_res -= 1 * _tmp35.transpose(1, 0, 2, 3)
    _tmp36 = einsum('knij,mijefa,ak->efnm', g_aaaa[o, o, o, o], l3_aaaaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp36
    l2_res += 0.5 * _tmp36.transpose(0, 1, 3, 2)
    _tmp37 = einsum('nkij,mijefa,ak->efnm', g_abab[o, o, o, o], l3_aabaab, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp37
    l2_res -= 0.5 * _tmp37.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp37
    l2_res -= 0.5 * _tmp37.transpose(0, 1, 3, 2)
    _tmp38 = einsum('jbei,mnibfa,aj->efnm', g_aaaa[o, v, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l2_res += 1 * _tmp38
    l2_res -= 1 * _tmp38.transpose(1, 0, 2, 3)
    _tmp39 = einsum('bjei,mnibfa,aj->efnm', g_abab[v, o, v, o], l3_aabaab, t1_bb, optimize=True)
    l2_res -= 1 * _tmp39
    l2_res += 1 * _tmp39.transpose(1, 0, 2, 3)
    _tmp40 = einsum('jbei,mniafb,aj->efnm', g_abab[o, v, v, o], l3_aabaab, t1_aa, optimize=True)
    l2_res -= 1 * _tmp40
    l2_res += 1 * _tmp40.transpose(1, 0, 2, 3)
    _tmp41 = einsum('nabj,imjaef,bi->efnm', g_aaaa[o, v, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l2_res += 1 * _tmp41
    l2_res -= 1 * _tmp41.transpose(0, 1, 3, 2)
    _tmp42 = einsum('nabj,imjfea,bi->efnm', g_abab[o, v, v, o], l3_aabaab, t1_aa, optimize=True)
    l2_res -= 1 * _tmp42
    l2_res += 1 * _tmp42.transpose(0, 1, 3, 2)
    _tmp43 = einsum('najb,jmifea,bi->efnm', g_abab[o, v, o, v], l3_aabaab, t1_bb, optimize=True)
    l2_res -= 1 * _tmp43
    l2_res += 1 * _tmp43.transpose(0, 1, 3, 2)
    _tmp44 = einsum('baec,imnbaf,ci->efnm', g_aaaa[v, v, v, v], l3_aaaaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp44
    l2_res += 0.5 * _tmp44.transpose(1, 0, 2, 3)
    _tmp45 = einsum('baec,nmibfa,ci->efnm', g_abab[v, v, v, v], l3_aabaab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp45
    l2_res += 0.5 * _tmp45.transpose(1, 0, 2, 3)
    _tmp46 = einsum('abec,nmifab,ci->efnm', g_abab[v, v, v, v], l3_aabaab, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp46
    l2_res -= 0.5 * _tmp46.transpose(1, 0, 2, 3)
    _tmp47 = einsum('jnef,imba,baij->efnm', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp47
    l2_res += 0.5 * _tmp47.transpose(0, 1, 3, 2)
    _tmp48 = einsum('jnef,miba,baji->efnm', g_aaaa[o, o, v, v], l2_abab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp48
    l2_res += 0.5 * _tmp48.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp48
    l2_res += 0.5 * _tmp48.transpose(0, 1, 3, 2)
    _tmp49 = einsum('jief,mnba,baji->efnm', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l2_res += 0.25 * _tmp49
    _tmp50 = einsum('mneb,ijfa,baij->efnm', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp50
    l2_res += 0.5 * _tmp50.transpose(1, 0, 2, 3)
    _tmp51 = einsum('mneb,ijfa,baij->efnm', g_aaaa[o, o, v, v], l2_abab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp51
    l2_res += 0.5 * _tmp51.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp51
    l2_res += 0.5 * _tmp51.transpose(1, 0, 2, 3)
    _tmp52 = einsum('jneb,imfa,baij->efnm', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l2_res += 1 * _tmp52
    l2_res -= 1 * _tmp52.transpose(1, 0, 2, 3)
    l2_res -= 1 * _tmp52.transpose(0, 1, 3, 2)
    l2_res += 1 * _tmp52.transpose(1, 0, 3, 2)
    _tmp53 = einsum('njeb,imfa,abij->efnm', g_abab[o, o, v, v], l2_aaaa, t2_abab, optimize=True)
    l2_res += 1 * _tmp53
    l2_res -= 1 * _tmp53.transpose(1, 0, 2, 3)
    l2_res -= 1 * _tmp53.transpose(0, 1, 3, 2)
    l2_res += 1 * _tmp53.transpose(1, 0, 3, 2)
    _tmp54 = einsum('jneb,mifa,baji->efnm', g_aaaa[o, o, v, v], l2_abab, t2_abab, optimize=True)
    l2_res += 1 * _tmp54
    l2_res -= 1 * _tmp54.transpose(1, 0, 2, 3)
    l2_res -= 1 * _tmp54.transpose(0, 1, 3, 2)
    l2_res += 1 * _tmp54.transpose(1, 0, 3, 2)
    _tmp55 = einsum('njeb,mifa,baij->efnm', g_abab[o, o, v, v], l2_abab, t2_bbbb, optimize=True)
    l2_res += 1 * _tmp55
    l2_res -= 1 * _tmp55.transpose(1, 0, 2, 3)
    l2_res -= 1 * _tmp55.transpose(0, 1, 3, 2)
    l2_res += 1 * _tmp55.transpose(1, 0, 3, 2)
    _tmp56 = einsum('jieb,mnfa,baji->efnm', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp56
    l2_res += 0.5 * _tmp56.transpose(1, 0, 2, 3)
    _tmp57 = einsum('jieb,mnfa,abji->efnm', g_abab[o, o, v, v], l2_aaaa, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp57
    l2_res -= 0.5 * _tmp57.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp57
    l2_res -= 0.5 * _tmp57.transpose(1, 0, 2, 3)
    _tmp58 = einsum('mnab,ijef,abij->efnm', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l2_res += 0.25 * _tmp58
    _tmp59 = einsum('jnab,imef,abij->efnm', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp59
    l2_res += 0.5 * _tmp59.transpose(0, 1, 3, 2)
    _tmp60 = einsum('njab,imef,abij->efnm', g_abab[o, o, v, v], l2_aaaa, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp60
    l2_res -= 0.5 * _tmp60.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp60
    l2_res -= 0.5 * _tmp60.transpose(0, 1, 3, 2)
    _tmp61 = einsum('knej,imjfba,baik->efnm', g_aaaa[o, o, v, o], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res += 0.5 * _tmp61
    l2_res -= 0.5 * _tmp61.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp61.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp61.transpose(1, 0, 3, 2)
    _tmp62 = einsum('nkej,imjfba,baik->efnm', g_abab[o, o, v, o], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp62
    l2_res += 0.5 * _tmp62.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp62.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp62.transpose(1, 0, 3, 2)
    l2_res -= 0.5 * _tmp62
    l2_res += 0.5 * _tmp62.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp62.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp62.transpose(1, 0, 3, 2)
    _tmp63 = einsum('knej,jmifba,baki->efnm', g_aaaa[o, o, v, o], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp63
    l2_res -= 0.5 * _tmp63.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp63.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp63.transpose(1, 0, 3, 2)
    l2_res += 0.5 * _tmp63
    l2_res -= 0.5 * _tmp63.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp63.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp63.transpose(1, 0, 3, 2)
    _tmp64 = einsum('nkej,mijfba,baik->efnm', g_abab[o, o, v, o], l3_abbabb, t2_bbbb, optimize=True)
    l2_res += 0.5 * _tmp64
    l2_res -= 0.5 * _tmp64.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp64.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp64.transpose(1, 0, 3, 2)
    _tmp65 = einsum('kjei,mnifba,bakj->efnm', g_aaaa[o, o, v, o], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res -= 0.25 * _tmp65
    l2_res += 0.25 * _tmp65.transpose(1, 0, 2, 3)
    _tmp66 = einsum('kjei,mnifba,bakj->efnm', g_abab[o, o, v, o], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.25 * _tmp66
    l2_res += 0.25 * _tmp66.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp66
    l2_res += 0.25 * _tmp66.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp66
    l2_res += 0.25 * _tmp66.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp66
    l2_res += 0.25 * _tmp66.transpose(1, 0, 2, 3)
    _tmp67 = einsum('mnbk,ijkefa,baij->efnm', g_aaaa[o, o, v, o], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp67
    _tmp68 = einsum('mnbk,ikjefa,baij->efnm', g_aaaa[o, o, v, o], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp68
    _tmp69 = einsum('mnbk,kjiefa,baji->efnm', g_aaaa[o, o, v, o], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp69
    _tmp70 = einsum('knbj,imjefa,baik->efnm', g_aaaa[o, o, v, o], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res += 1 * _tmp70
    l2_res -= 1 * _tmp70.transpose(0, 1, 3, 2)
    _tmp71 = einsum('nkjb,imjefa,abik->efnm', g_abab[o, o, o, v], l3_aaaaaa, t2_abab, optimize=True)
    l2_res -= 1 * _tmp71
    l2_res += 1 * _tmp71.transpose(0, 1, 3, 2)
    _tmp72 = einsum('nkbj,imjefa,baik->efnm', g_abab[o, o, v, o], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 1 * _tmp72
    l2_res += 1 * _tmp72.transpose(0, 1, 3, 2)
    _tmp73 = einsum('knbj,jmiefa,baki->efnm', g_aaaa[o, o, v, o], l3_aabaab, t2_abab, optimize=True)
    l2_res += 1 * _tmp73
    l2_res -= 1 * _tmp73.transpose(0, 1, 3, 2)
    _tmp74 = einsum('nkjb,jmiefa,baik->efnm', g_abab[o, o, o, v], l3_aabaab, t2_bbbb, optimize=True)
    l2_res -= 1 * _tmp74
    l2_res += 1 * _tmp74.transpose(0, 1, 3, 2)
    _tmp75 = einsum('jcef,imncba,baij->efnm', g_aaaa[o, v, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp75
    _tmp76 = einsum('jcef,nmicba,baji->efnm', g_aaaa[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp76
    l2_res -= 0.5 * _tmp76
    _tmp77 = einsum('nbec,ijmbfa,caij->efnm', g_aaaa[o, v, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res += 0.5 * _tmp77
    l2_res -= 0.5 * _tmp77.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp77.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp77.transpose(1, 0, 3, 2)
    _tmp78 = einsum('nbec,imjbfa,caij->efnm', g_aaaa[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp78
    l2_res += 0.5 * _tmp78.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp78.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp78.transpose(1, 0, 3, 2)
    _tmp79 = einsum('nbec,imjafb,acij->efnm', g_abab[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp79
    l2_res += 0.5 * _tmp79.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp79.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp79.transpose(1, 0, 3, 2)
    _tmp80 = einsum('nbec,mjibfa,caji->efnm', g_aaaa[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp80
    l2_res -= 0.5 * _tmp80.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp80.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp80.transpose(1, 0, 3, 2)
    _tmp81 = einsum('nbec,mjiafb,acji->efnm', g_abab[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp81
    l2_res -= 0.5 * _tmp81.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp81.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp81.transpose(1, 0, 3, 2)
    _tmp82 = einsum('nbec,mjifba,caij->efnm', g_abab[o, v, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l2_res += 0.5 * _tmp82
    l2_res -= 0.5 * _tmp82.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp82.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp82.transpose(1, 0, 3, 2)
    _tmp83 = einsum('jbec,imnbfa,caij->efnm', g_aaaa[o, v, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res += 1 * _tmp83
    l2_res -= 1 * _tmp83.transpose(1, 0, 2, 3)
    _tmp84 = einsum('bjec,imnbfa,acij->efnm', g_abab[v, o, v, v], l3_aaaaaa, t2_abab, optimize=True)
    l2_res += 1 * _tmp84
    l2_res -= 1 * _tmp84.transpose(1, 0, 2, 3)
    _tmp85 = einsum('jbec,nmibfa,caji->efnm', g_aaaa[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 1 * _tmp85
    l2_res -= 1 * _tmp85.transpose(1, 0, 2, 3)
    _tmp86 = einsum('bjec,nmibfa,caij->efnm', g_abab[v, o, v, v], l3_aabaab, t2_bbbb, optimize=True)
    l2_res += 1 * _tmp86
    l2_res -= 1 * _tmp86.transpose(1, 0, 2, 3)
    _tmp87 = einsum('jbec,nmiafb,acji->efnm', g_abab[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 1 * _tmp87
    l2_res -= 1 * _tmp87.transpose(1, 0, 2, 3)
    _tmp88 = einsum('nabc,ijmaef,bcij->efnm', g_aaaa[o, v, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res -= 0.25 * _tmp88
    l2_res += 0.25 * _tmp88.transpose(0, 1, 3, 2)
    _tmp89 = einsum('nabc,imjfea,bcij->efnm', g_abab[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.25 * _tmp89
    l2_res += 0.25 * _tmp89.transpose(0, 1, 3, 2)
    l2_res -= 0.25 * _tmp89
    l2_res += 0.25 * _tmp89.transpose(0, 1, 3, 2)
    _tmp90 = einsum('nabc,mjifea,bcji->efnm', g_abab[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.25 * _tmp90
    l2_res -= 0.25 * _tmp90.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp90
    l2_res -= 0.25 * _tmp90.transpose(0, 1, 3, 2)
    _tmp91 = einsum('knef,ijmcba,cbaijk->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp91
    l2_res += 0.0833333333333333 * _tmp91.transpose(0, 1, 3, 2)
    _tmp92 = einsum('knef,imjcba,cbaikj->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp92
    l2_res += 0.0833333333333333 * _tmp92.transpose(0, 1, 3, 2)
    l2_res -= 0.0833333333333333 * _tmp92
    l2_res += 0.0833333333333333 * _tmp92.transpose(0, 1, 3, 2)
    l2_res -= 0.0833333333333333 * _tmp92
    l2_res += 0.0833333333333333 * _tmp92.transpose(0, 1, 3, 2)
    _tmp93 = einsum('knef,mjicba,cbakji->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp93
    l2_res += 0.0833333333333333 * _tmp93.transpose(0, 1, 3, 2)
    l2_res -= 0.0833333333333333 * _tmp93
    l2_res += 0.0833333333333333 * _tmp93.transpose(0, 1, 3, 2)
    l2_res -= 0.0833333333333333 * _tmp93
    l2_res += 0.0833333333333333 * _tmp93.transpose(0, 1, 3, 2)
    _tmp94 = einsum('knef,mjicba,cbakji->efnm', g_aaaa[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp94
    l2_res += 0.0833333333333333 * _tmp94.transpose(0, 1, 3, 2)
    l2_res -= 0.0833333333333333 * _tmp94
    l2_res += 0.0833333333333333 * _tmp94.transpose(0, 1, 3, 2)
    l2_res -= 0.0833333333333333 * _tmp94
    l2_res += 0.0833333333333333 * _tmp94.transpose(0, 1, 3, 2)
    _tmp95 = einsum('kjef,imncba,cbaikj->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l2_res += 0.0833333333333333 * _tmp95
    _tmp96 = einsum('kjef,nmicba,cbajki->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.0833333333333333 * _tmp96
    l2_res += 0.0833333333333333 * _tmp96
    l2_res += 0.0833333333333333 * _tmp96
    _tmp97 = einsum('mnec,ijkfba,cbaijk->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp97
    l2_res += 0.0833333333333333 * _tmp97.transpose(1, 0, 2, 3)
    _tmp98 = einsum('mnec,ijkfba,cbaijk->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp98
    l2_res += 0.0833333333333333 * _tmp98.transpose(1, 0, 2, 3)
    l2_res -= 0.0833333333333333 * _tmp98
    l2_res += 0.0833333333333333 * _tmp98.transpose(1, 0, 2, 3)
    l2_res -= 0.0833333333333333 * _tmp98
    l2_res += 0.0833333333333333 * _tmp98.transpose(1, 0, 2, 3)
    l2_res -= 0.0833333333333333 * _tmp98
    l2_res += 0.0833333333333333 * _tmp98.transpose(1, 0, 2, 3)
    _tmp99 = einsum('mnec,ijkfba,cbaijk->efnm', g_aaaa[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp99
    l2_res += 0.0833333333333333 * _tmp99.transpose(1, 0, 2, 3)
    l2_res -= 0.0833333333333333 * _tmp98
    l2_res += 0.0833333333333333 * _tmp98.transpose(1, 0, 2, 3)
    l2_res -= 0.0833333333333333 * _tmp98
    l2_res += 0.0833333333333333 * _tmp98.transpose(1, 0, 2, 3)
    l2_res -= 0.0833333333333333 * _tmp99
    l2_res += 0.0833333333333333 * _tmp99.transpose(1, 0, 2, 3)
    l2_res -= 0.0833333333333333 * _tmp99
    l2_res += 0.0833333333333333 * _tmp99.transpose(1, 0, 2, 3)
    _tmp100 = einsum('knec,ijmfba,cbaijk->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l2_res += 0.25 * _tmp100
    l2_res -= 0.25 * _tmp100.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp100.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp100.transpose(1, 0, 3, 2)
    _tmp101 = einsum('nkec,ijmfba,abcijk->efnm', g_abab[o, o, v, v], l3_aaaaaa, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp101
    l2_res -= 0.25 * _tmp101.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp101.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp101.transpose(1, 0, 3, 2)
    _tmp102 = einsum('knec,imjfba,cbaikj->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp102
    l2_res -= 0.25 * _tmp102.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp102.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp102.transpose(1, 0, 3, 2)
    _tmp103 = einsum('nkec,imjfba,bcaijk->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp103
    l2_res += 0.25 * _tmp103.transpose(1, 0, 2, 3)
    l2_res += 0.25 * _tmp103.transpose(0, 1, 3, 2)
    l2_res -= 0.25 * _tmp103.transpose(1, 0, 3, 2)
    l2_res += 0.25 * _tmp102
    l2_res -= 0.25 * _tmp102.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp102.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp102.transpose(1, 0, 3, 2)
    _tmp104 = einsum('nkec,imjfab,abcijk->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp104
    l2_res -= 0.25 * _tmp104.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp104.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp104.transpose(1, 0, 3, 2)
    _tmp105 = einsum('knec,mjifba,cbakji->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp105
    l2_res -= 0.25 * _tmp105.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp105.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp105.transpose(1, 0, 3, 2)
    _tmp106 = einsum('nkec,mjifba,bcajik->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp106
    l2_res -= 0.25 * _tmp106.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp106.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp106.transpose(1, 0, 3, 2)
    l2_res += 0.25 * _tmp105
    l2_res -= 0.25 * _tmp105.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp105.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp105.transpose(1, 0, 3, 2)
    _tmp107 = einsum('nkec,mjifab,abcjik->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp107
    l2_res += 0.25 * _tmp107.transpose(1, 0, 2, 3)
    l2_res += 0.25 * _tmp107.transpose(0, 1, 3, 2)
    l2_res -= 0.25 * _tmp107.transpose(1, 0, 3, 2)
    _tmp108 = einsum('knec,mjifba,cbakji->efnm', g_aaaa[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp108
    l2_res -= 0.25 * _tmp108.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp108.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp108.transpose(1, 0, 3, 2)
    _tmp109 = einsum('nkec,mjifba,cbaijk->efnm', g_abab[o, o, v, v], l3_abbabb, t3_bbbbbb, optimize=True)
    l2_res += 0.25 * _tmp109
    l2_res -= 0.25 * _tmp109.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp109.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp109.transpose(1, 0, 3, 2)
    _tmp110 = einsum('kjec,imnfba,cbaikj->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l2_res -= 0.25 * _tmp110
    l2_res += 0.25 * _tmp110.transpose(1, 0, 2, 3)
    _tmp111 = einsum('kjec,imnfba,abcikj->efnm', g_abab[o, o, v, v], l3_aaaaaa, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp111
    l2_res -= 0.25 * _tmp111.transpose(1, 0, 2, 3)
    l2_res += 0.25 * _tmp111
    l2_res -= 0.25 * _tmp111.transpose(1, 0, 2, 3)
    _tmp112 = einsum('kjec,nmifba,cbajki->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp112
    l2_res += 0.25 * _tmp112.transpose(1, 0, 2, 3)
    _tmp113 = einsum('kjec,nmifba,bcakij->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp113
    l2_res -= 0.25 * _tmp113.transpose(1, 0, 2, 3)
    _tmp114 = einsum('jkec,nmifba,bcajki->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp114
    l2_res += 0.25 * _tmp114.transpose(1, 0, 2, 3)
    l2_res -= 0.25 * _tmp112
    l2_res += 0.25 * _tmp112.transpose(1, 0, 2, 3)
    _tmp115 = einsum('kjec,nmifab,abckij->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp115
    l2_res += 0.25 * _tmp115.transpose(1, 0, 2, 3)
    _tmp116 = einsum('jkec,nmifab,abcjki->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp116
    l2_res -= 0.25 * _tmp116.transpose(1, 0, 2, 3)
    _tmp117 = einsum('mnbc,ijkefa,bcaijk->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l2_res += 0.0833333333333333 * _tmp117
    _tmp118 = einsum('mnbc,ijkefa,bcaijk->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.0833333333333333 * _tmp118
    l2_res += 0.0833333333333333 * _tmp118
    l2_res += 0.0833333333333333 * _tmp118
    _tmp119 = einsum('knbc,ijmefa,bcaijk->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l2_res -= 0.25 * _tmp119
    l2_res += 0.25 * _tmp119.transpose(0, 1, 3, 2)
    _tmp120 = einsum('nkbc,ijmefa,bacijk->efnm', g_abab[o, o, v, v], l3_aaaaaa, t3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp120
    l2_res += 0.25 * _tmp120.transpose(0, 1, 3, 2)
    _tmp121 = einsum('nkcb,ijmefa,acbijk->efnm', g_abab[o, o, v, v], l3_aaaaaa, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp121
    l2_res -= 0.25 * _tmp121.transpose(0, 1, 3, 2)
    _tmp122 = einsum('knbc,imjefa,bcaikj->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp122
    l2_res += 0.25 * _tmp122.transpose(0, 1, 3, 2)
    _tmp123 = einsum('nkbc,imjefa,bcaijk->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp123
    l2_res += 0.25 * _tmp123.transpose(0, 1, 3, 2)
    l2_res -= 0.25 * _tmp123
    l2_res += 0.25 * _tmp123.transpose(0, 1, 3, 2)
    _tmp124 = einsum('knbc,mjiefa,bcakji->efnm', g_aaaa[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp124
    l2_res += 0.25 * _tmp124.transpose(0, 1, 3, 2)
    _tmp125 = einsum('nkbc,mjiefa,bcajik->efnm', g_abab[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp125
    l2_res -= 0.25 * _tmp125.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp125
    l2_res -= 0.25 * _tmp125.transpose(0, 1, 3, 2)
    _tmp126 = einsum('kjef,imncba,baik,cj->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp126
    _tmp127 = einsum('kjef,nmicba,baki,cj->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp127
    l2_res += 0.5 * _tmp127
    _tmp128 = einsum('knec,ijmfba,baik,cj->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp128
    l2_res += 0.5 * _tmp128.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp128.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp128.transpose(1, 0, 3, 2)
    _tmp129 = einsum('nkec,imjfba,baik,cj->efnm', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp129
    l2_res += 0.5 * _tmp129.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp129.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp129.transpose(1, 0, 3, 2)
    l2_res -= 0.5 * _tmp129
    l2_res += 0.5 * _tmp129.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp129.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp129.transpose(1, 0, 3, 2)
    _tmp130 = einsum('knec,mjifba,baki,cj->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp130
    l2_res += 0.5 * _tmp130.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp130.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp130.transpose(1, 0, 3, 2)
    l2_res -= 0.5 * _tmp130
    l2_res += 0.5 * _tmp130.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp130.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp130.transpose(1, 0, 3, 2)
    _tmp131 = einsum('nkec,mjifba,baik,cj->efnm', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp131
    l2_res += 0.5 * _tmp131.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp131.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp131.transpose(1, 0, 3, 2)
    _tmp132 = einsum('knec,ijmfba,caij,bk->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp132
    l2_res += 0.5 * _tmp132.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp132.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp132.transpose(1, 0, 3, 2)
    _tmp133 = einsum('knec,imjfba,caij,bk->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp133
    l2_res -= 0.5 * _tmp133.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp133.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp133.transpose(1, 0, 3, 2)
    _tmp134 = einsum('nkec,imjfab,acij,bk->efnm', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp134
    l2_res += 0.5 * _tmp134.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp134.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp134.transpose(1, 0, 3, 2)
    _tmp135 = einsum('knec,mjifba,caji,bk->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp135
    l2_res += 0.5 * _tmp135.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp135.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp135.transpose(1, 0, 3, 2)
    _tmp136 = einsum('nkec,mjifab,acji,bk->efnm', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp136
    l2_res -= 0.5 * _tmp136.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp136.transpose(0, 1, 3, 2)
    l2_res += 0.5 * _tmp136.transpose(1, 0, 3, 2)
    _tmp137 = einsum('nkec,mjifba,caij,bk->efnm', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp137
    l2_res += 0.5 * _tmp137.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp137.transpose(0, 1, 3, 2)
    l2_res -= 0.5 * _tmp137.transpose(1, 0, 3, 2)
    _tmp138 = einsum('kjec,imnfba,baik,cj->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp138
    l2_res += 0.5 * _tmp138.transpose(1, 0, 2, 3)
    _tmp139 = einsum('kjec,imnfba,baik,cj->efnm', g_abab[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp139
    l2_res += 0.5 * _tmp139.transpose(1, 0, 2, 3)
    _tmp140 = einsum('kjec,nmifba,baki,cj->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp140
    l2_res += 0.5 * _tmp140.transpose(1, 0, 2, 3)
    _tmp141 = einsum('kjec,nmifba,baki,cj->efnm', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp141
    l2_res += 0.5 * _tmp141.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp140
    l2_res += 0.5 * _tmp140.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp141
    l2_res += 0.5 * _tmp141.transpose(1, 0, 2, 3)
    _tmp142 = einsum('kjec,imnfba,bakj,ci->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res -= 0.25 * _tmp142
    l2_res += 0.25 * _tmp142.transpose(1, 0, 2, 3)
    _tmp143 = einsum('kjec,nmifba,bakj,ci->efnm', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res += 0.25 * _tmp143
    l2_res -= 0.25 * _tmp143.transpose(1, 0, 2, 3)
    l2_res += 0.25 * _tmp143
    l2_res -= 0.25 * _tmp143.transpose(1, 0, 2, 3)
    l2_res += 0.25 * _tmp143
    l2_res -= 0.25 * _tmp143.transpose(1, 0, 2, 3)
    l2_res += 0.25 * _tmp143
    l2_res -= 0.25 * _tmp143.transpose(1, 0, 2, 3)
    _tmp144 = einsum('kjec,imnfba,caik,bj->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res += 1 * _tmp144
    l2_res -= 1 * _tmp144.transpose(1, 0, 2, 3)
    _tmp145 = einsum('jkec,imnfba,acik,bj->efnm', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, t1_aa, optimize=True)
    l2_res += 1 * _tmp145
    l2_res -= 1 * _tmp145.transpose(1, 0, 2, 3)
    _tmp146 = einsum('kjec,nmifba,caki,bj->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res += 1 * _tmp146
    l2_res -= 1 * _tmp146.transpose(1, 0, 2, 3)
    _tmp147 = einsum('jkec,nmifba,caik,bj->efnm', g_abab[o, o, v, v], l3_aabaab, t2_bbbb, t1_aa, optimize=True)
    l2_res += 1 * _tmp147
    l2_res -= 1 * _tmp147.transpose(1, 0, 2, 3)
    _tmp148 = einsum('kjec,nmifab,acki,bj->efnm', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res += 1 * _tmp148
    l2_res -= 1 * _tmp148.transpose(1, 0, 2, 3)
    _tmp149 = einsum('mnbc,ijkefa,caij,bk->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp149
    _tmp150 = einsum('mnbc,ikjefa,caij,bk->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp150
    _tmp151 = einsum('mnbc,kjiefa,caji,bk->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp151
    _tmp152 = einsum('knbc,ijmefa,caij,bk->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp152
    l2_res += 0.5 * _tmp152.transpose(0, 1, 3, 2)
    _tmp153 = einsum('nkcb,ijmefa,caij,bk->efnm', g_abab[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp153
    l2_res += 0.5 * _tmp153.transpose(0, 1, 3, 2)
    _tmp154 = einsum('knbc,imjefa,caij,bk->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp154
    l2_res -= 0.5 * _tmp154.transpose(0, 1, 3, 2)
    _tmp155 = einsum('nkcb,imjefa,caij,bk->efnm', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp155
    l2_res -= 0.5 * _tmp155.transpose(0, 1, 3, 2)
    _tmp156 = einsum('knbc,mjiefa,caji,bk->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp156
    l2_res += 0.5 * _tmp156.transpose(0, 1, 3, 2)
    _tmp157 = einsum('nkcb,mjiefa,caji,bk->efnm', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp157
    l2_res += 0.5 * _tmp157.transpose(0, 1, 3, 2)
    _tmp158 = einsum('knbc,ijmefa,caik,bj->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res += 1 * _tmp158
    l2_res -= 1 * _tmp158.transpose(0, 1, 3, 2)
    _tmp159 = einsum('nkbc,ijmefa,acik,bj->efnm', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, t1_aa, optimize=True)
    l2_res += 1 * _tmp159
    l2_res -= 1 * _tmp159.transpose(0, 1, 3, 2)
    _tmp160 = einsum('nkcb,imjefa,caik,bj->efnm', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res -= 1 * _tmp160
    l2_res += 1 * _tmp160.transpose(0, 1, 3, 2)
    _tmp161 = einsum('knbc,mjiefa,caki,bj->efnm', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res += 1 * _tmp161
    l2_res -= 1 * _tmp161.transpose(0, 1, 3, 2)
    _tmp162 = einsum('nkbc,mjiefa,caik,bj->efnm', g_abab[o, o, v, v], l3_aabaab, t2_bbbb, t1_aa, optimize=True)
    l2_res += 1 * _tmp162
    l2_res -= 1 * _tmp162.transpose(0, 1, 3, 2)
    _tmp163 = einsum('knbc,ijmefa,ak,bcij->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t2_aaaa, optimize=True)
    l2_res -= 0.25 * _tmp163
    l2_res += 0.25 * _tmp163.transpose(0, 1, 3, 2)
    _tmp164 = einsum('nkbc,imjefa,ak,bcij->efnm', g_abab[o, o, v, v], l3_aabaab, t1_bb, t2_abab, optimize=True)
    l2_res -= 0.25 * _tmp164
    l2_res += 0.25 * _tmp164.transpose(0, 1, 3, 2)
    l2_res -= 0.25 * _tmp164
    l2_res += 0.25 * _tmp164.transpose(0, 1, 3, 2)
    _tmp165 = einsum('nkbc,mjiefa,ak,bcji->efnm', g_abab[o, o, v, v], l3_aabaab, t1_bb, t2_abab, optimize=True)
    l2_res += 0.25 * _tmp165
    l2_res -= 0.25 * _tmp165.transpose(0, 1, 3, 2)
    l2_res += 0.25 * _tmp165
    l2_res -= 0.25 * _tmp165.transpose(0, 1, 3, 2)
    _tmp166 = einsum('jief,mnba,aj,bi->efnm', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp166
    _tmp167 = einsum('jneb,imfa,aj,bi->efnm', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l2_res += 1 * _tmp167
    l2_res -= 1 * _tmp167.transpose(1, 0, 2, 3)
    l2_res -= 1 * _tmp167.transpose(0, 1, 3, 2)
    l2_res += 1 * _tmp167.transpose(1, 0, 3, 2)
    _tmp168 = einsum('njeb,mifa,aj,bi->efnm', g_abab[o, o, v, v], l2_abab, t1_bb, t1_bb, optimize=True)
    l2_res += 1 * _tmp168
    l2_res -= 1 * _tmp168.transpose(1, 0, 2, 3)
    l2_res -= 1 * _tmp168.transpose(0, 1, 3, 2)
    l2_res += 1 * _tmp168.transpose(1, 0, 3, 2)
    _tmp169 = einsum('jieb,mnfa,aj,bi->efnm', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l2_res += 1 * _tmp169
    l2_res -= 1 * _tmp169.transpose(1, 0, 2, 3)
    _tmp170 = einsum('jieb,mnfa,aj,bi->efnm', g_abab[o, o, v, v], l2_aaaa, t1_aa, t1_bb, optimize=True)
    l2_res += 1 * _tmp170
    l2_res -= 1 * _tmp170.transpose(1, 0, 2, 3)
    _tmp171 = einsum('mnab,ijef,aj,bi->efnm', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp171
    _tmp172 = einsum('jnab,imef,aj,bi->efnm', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l2_res += 1 * _tmp172
    l2_res -= 1 * _tmp172.transpose(0, 1, 3, 2)
    _tmp173 = einsum('njba,imef,aj,bi->efnm', g_abab[o, o, v, v], l2_aaaa, t1_bb, t1_aa, optimize=True)
    l2_res += 1 * _tmp173
    l2_res -= 1 * _tmp173.transpose(0, 1, 3, 2)
    _tmp174 = einsum('kjei,mnifba,ak,bj->efnm', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp174
    l2_res -= 0.5 * _tmp174.transpose(1, 0, 2, 3)
    _tmp175 = einsum('jkei,mnifba,ak,bj->efnm', g_abab[o, o, v, o], l3_aabaab, t1_bb, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp175
    l2_res += 0.5 * _tmp175.transpose(1, 0, 2, 3)
    l2_res -= 0.5 * _tmp175
    l2_res += 0.5 * _tmp175.transpose(1, 0, 2, 3)
    _tmp176 = einsum('knbj,imjefa,ak,bi->efnm', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l2_res += 1 * _tmp176
    l2_res -= 1 * _tmp176.transpose(0, 1, 3, 2)
    _tmp177 = einsum('nkbj,imjefa,ak,bi->efnm', g_abab[o, o, v, o], l3_aabaab, t1_bb, t1_aa, optimize=True)
    l2_res -= 1 * _tmp177
    l2_res += 1 * _tmp177.transpose(0, 1, 3, 2)
    _tmp178 = einsum('nkjb,jmiefa,ak,bi->efnm', g_abab[o, o, o, v], l3_aabaab, t1_bb, t1_bb, optimize=True)
    l2_res -= 1 * _tmp178
    l2_res += 1 * _tmp178.transpose(0, 1, 3, 2)
    _tmp179 = einsum('jbec,imnbfa,aj,ci->efnm', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l2_res += 1 * _tmp179
    l2_res -= 1 * _tmp179.transpose(1, 0, 2, 3)
    _tmp180 = einsum('bjec,nmibfa,aj,ci->efnm', g_abab[v, o, v, v], l3_aabaab, t1_bb, t1_bb, optimize=True)
    l2_res += 1 * _tmp180
    l2_res -= 1 * _tmp180.transpose(1, 0, 2, 3)
    _tmp181 = einsum('jbec,nmiafb,aj,ci->efnm', g_abab[o, v, v, v], l3_aabaab, t1_aa, t1_bb, optimize=True)
    l2_res += 1 * _tmp181
    l2_res -= 1 * _tmp181.transpose(1, 0, 2, 3)
    _tmp182 = einsum('nabc,ijmaef,bj,ci->efnm', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp182
    l2_res -= 0.5 * _tmp182.transpose(0, 1, 3, 2)
    _tmp183 = einsum('nacb,imjfea,bj,ci->efnm', g_abab[o, v, v, v], l3_aabaab, t1_bb, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp183
    l2_res += 0.5 * _tmp183.transpose(0, 1, 3, 2)
    _tmp184 = einsum('nabc,mjifea,bj,ci->efnm', g_abab[o, v, v, v], l3_aabaab, t1_aa, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp184
    l2_res -= 0.5 * _tmp184.transpose(0, 1, 3, 2)
    _tmp185 = einsum('kjec,imnfba,ak,bj,ci->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp185
    l2_res -= 0.5 * _tmp185.transpose(1, 0, 2, 3)
    _tmp186 = einsum('jkec,nmifba,ak,bj,ci->efnm', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_aa, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp186
    l2_res -= 0.5 * _tmp186.transpose(1, 0, 2, 3)
    l2_res += 0.5 * _tmp186
    l2_res -= 0.5 * _tmp186.transpose(1, 0, 2, 3)
    _tmp187 = einsum('knbc,ijmefa,ak,bj,ci->efnm', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp187
    l2_res -= 0.5 * _tmp187.transpose(0, 1, 3, 2)
    _tmp188 = einsum('nkcb,imjefa,ak,bj,ci->efnm', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_bb, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp188
    l2_res += 0.5 * _tmp188.transpose(0, 1, 3, 2)
    _tmp189 = einsum('nkbc,mjiefa,ak,bj,ci->efnm', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_aa, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp189
    l2_res -= 0.5 * _tmp189.transpose(0, 1, 3, 2)
    return l2_res


def l2_abab_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb):
    nv, no = t1_aa.shape
    l2_res = np.zeros((nv, nv, no, no))
    _tmp0 = einsum('mnef->efmn', g_abab[o, o, v, v])
    l2_res += 1 * _tmp0
    _tmp1 = einsum('nf,me->efmn', f_bb[o, v], l1_aa, optimize=True)
    l2_res += 1 * _tmp1
    _tmp2 = einsum('me,nf->efmn', f_aa[o, v], l1_bb, optimize=True)
    l2_res += 1 * _tmp2
    _tmp3 = einsum('ni,mief->efmn', f_bb[o, o], l2_abab, optimize=True)
    l2_res -= 1 * _tmp3
    _tmp4 = einsum('mi,inef->efmn', f_aa[o, o], l2_abab, optimize=True)
    l2_res -= 1 * _tmp4
    _tmp5 = einsum('ae,mnaf->efmn', f_aa[v, v], l2_abab, optimize=True)
    l2_res += 1 * _tmp5
    _tmp6 = einsum('af,mnea->efmn', f_bb[v, v], l2_abab, optimize=True)
    l2_res += 1 * _tmp6
    _tmp7 = einsum('ie,ai,mnaf->efmn', f_aa[o, v], t1_aa, l2_abab, optimize=True)
    l2_res -= 1 * _tmp7
    _tmp8 = einsum('if,ai,mnea->efmn', f_bb[o, v], t1_bb, l2_abab, optimize=True)
    l2_res -= 1 * _tmp8
    _tmp9 = einsum('na,ai,mief->efmn', f_bb[o, v], t1_bb, l2_abab, optimize=True)
    l2_res -= 1 * _tmp9
    _tmp10 = einsum('ma,ai,inef->efmn', f_aa[o, v], t1_aa, l2_abab, optimize=True)
    l2_res -= 1 * _tmp10
    _tmp11 = einsum('je,baij,imnabf->efmn', f_aa[o, v], t2_aaaa, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp11
    _tmp12 = einsum('je,baji,minbfa->efmn', f_aa[o, v], t2_abab, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp12
    _tmp13 = einsum('je,abji,minabf->efmn', f_aa[o, v], t2_abab, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp13
    _tmp14 = einsum('jf,baij,imneba->efmn', f_bb[o, v], t2_abab, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp14
    l2_res += 0.5 * _tmp14
    _tmp15 = einsum('jf,baij,mineba->efmn', f_bb[o, v], t2_bbbb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp15
    _tmp16 = einsum('nb,abij,imjeaf->efmn', f_bb[o, v], t2_abab, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp16
    _tmp17 = einsum('nb,abji,mjieaf->efmn', f_bb[o, v], t2_abab, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp17
    _tmp18 = einsum('nb,baij,mjiefa->efmn', f_bb[o, v], t2_bbbb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp18
    _tmp19 = einsum('mb,baij,ijneaf->efmn', f_aa[o, v], t2_aaaa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp19
    _tmp20 = einsum('mb,baij,ijnefa->efmn', f_aa[o, v], t2_abab, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp20
    l2_res += 0.5 * _tmp20
    _tmp21 = einsum('mnei,if->efmn', g_abab[o, o, v, o], l1_bb, optimize=True)
    l2_res -= 1 * _tmp21
    _tmp22 = einsum('mnif,ie->efmn', g_abab[o, o, o, v], l1_aa, optimize=True)
    l2_res -= 1 * _tmp22
    _tmp23 = einsum('anef,ma->efmn', g_abab[v, o, v, v], l1_aa, optimize=True)
    l2_res += 1 * _tmp23
    _tmp24 = einsum('maef,na->efmn', g_abab[o, v, v, v], l1_bb, optimize=True)
    l2_res += 1 * _tmp24
    _tmp25 = einsum('mnij,ijef->efmn', g_abab[o, o, o, o], l2_abab, optimize=True)
    l2_res += 0.5 * _tmp25
    l2_res += 0.5 * _tmp25
    _tmp26 = einsum('anei,miaf->efmn', g_abab[v, o, v, o], l2_abab, optimize=True)
    l2_res -= 1 * _tmp26
    _tmp27 = einsum('anif,miae->efmn', g_abab[v, o, o, v], l2_aaaa, optimize=True)
    l2_res -= 1 * _tmp27
    _tmp28 = einsum('nafi,miea->efmn', g_bbbb[o, v, v, o], l2_abab, optimize=True)
    l2_res += 1 * _tmp28
    _tmp29 = einsum('maei,inaf->efmn', g_aaaa[o, v, v, o], l2_abab, optimize=True)
    l2_res += 1 * _tmp29
    _tmp30 = einsum('maei,niaf->efmn', g_abab[o, v, v, o], l2_bbbb, optimize=True)
    l2_res -= 1 * _tmp30
    _tmp31 = einsum('maif,inea->efmn', g_abab[o, v, o, v], l2_abab, optimize=True)
    l2_res -= 1 * _tmp31
    _tmp32 = einsum('baef,mnba->efmn', g_abab[v, v, v, v], l2_abab, optimize=True)
    l2_res += 0.5 * _tmp32
    l2_res += 0.5 * _tmp32
    _tmp33 = einsum('anij,mijaef->efmn', g_abab[v, o, o, o], l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp33
    l2_res += 0.5 * _tmp33
    _tmp34 = einsum('naij,mijeaf->efmn', g_bbbb[o, v, o, o], l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp34
    _tmp35 = einsum('maij,jinaef->efmn', g_aaaa[o, v, o, o], l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp35
    _tmp36 = einsum('maij,injeaf->efmn', g_abab[o, v, o, o], l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp36
    _tmp37 = einsum('maji,jineaf->efmn', g_abab[o, v, o, o], l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp37
    _tmp38 = einsum('baei,minbaf->efmn', g_aaaa[v, v, v, o], l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp38
    _tmp39 = einsum('baei,mnibaf->efmn', g_abab[v, v, v, o], l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp39
    l2_res -= 0.5 * _tmp39
    _tmp40 = einsum('baif,minbea->efmn', g_abab[v, v, o, v], l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp40
    _tmp41 = einsum('abif,mineab->efmn', g_abab[v, v, o, v], l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp41
    _tmp42 = einsum('bafi,mnieab->efmn', g_bbbb[v, v, v, o], l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp42
    _tmp43 = einsum('inef,ai,ma->efmn', g_abab[o, o, v, v], t1_aa, l1_aa, optimize=True)
    l2_res -= 1 * _tmp43
    _tmp44 = einsum('mief,ai,na->efmn', g_abab[o, o, v, v], t1_bb, l1_bb, optimize=True)
    l2_res -= 1 * _tmp44
    _tmp45 = einsum('mnea,ai,if->efmn', g_abab[o, o, v, v], t1_bb, l1_bb, optimize=True)
    l2_res -= 1 * _tmp45
    _tmp46 = einsum('mnaf,ai,ie->efmn', g_abab[o, o, v, v], t1_aa, l1_aa, optimize=True)
    l2_res -= 1 * _tmp46
    _tmp47 = einsum('inaf,me,ai->efmn', g_abab[o, o, v, v], l1_aa, t1_aa, optimize=True)
    l2_res += 1 * _tmp47
    _tmp48 = einsum('infa,me,ai->efmn', g_bbbb[o, o, v, v], l1_aa, t1_bb, optimize=True)
    l2_res -= 1 * _tmp48
    _tmp49 = einsum('imea,nf,ai->efmn', g_aaaa[o, o, v, v], l1_bb, t1_aa, optimize=True)
    l2_res -= 1 * _tmp49
    _tmp50 = einsum('miea,nf,ai->efmn', g_abab[o, o, v, v], l1_bb, t1_bb, optimize=True)
    l2_res += 1 * _tmp50
    _tmp51 = einsum('jnei,miaf,aj->efmn', g_abab[o, o, v, o], l2_abab, t1_aa, optimize=True)
    l2_res += 1 * _tmp51
    _tmp52 = einsum('jnif,miea,aj->efmn', g_abab[o, o, o, v], l2_aaaa, t1_aa, optimize=True)
    l2_res -= 1 * _tmp52
    _tmp53 = einsum('jnfi,miea,aj->efmn', g_bbbb[o, o, v, o], l2_abab, t1_bb, optimize=True)
    l2_res += 1 * _tmp53
    _tmp54 = einsum('jmei,inaf,aj->efmn', g_aaaa[o, o, v, o], l2_abab, t1_aa, optimize=True)
    l2_res += 1 * _tmp54
    _tmp55 = einsum('mjei,nifa,aj->efmn', g_abab[o, o, v, o], l2_bbbb, t1_bb, optimize=True)
    l2_res -= 1 * _tmp55
    _tmp56 = einsum('mjif,inea,aj->efmn', g_abab[o, o, o, v], l2_abab, t1_bb, optimize=True)
    l2_res += 1 * _tmp56
    _tmp57 = einsum('mnaj,ijef,ai->efmn', g_abab[o, o, v, o], l2_abab, t1_aa, optimize=True)
    l2_res += 1 * _tmp57
    _tmp58 = einsum('mnja,jief,ai->efmn', g_abab[o, o, o, v], l2_abab, t1_bb, optimize=True)
    l2_res += 1 * _tmp58
    _tmp59 = einsum('jnai,aj,mief->efmn', g_abab[o, o, v, o], t1_aa, l2_abab, optimize=True)
    l2_res -= 1 * _tmp59
    _tmp60 = einsum('jnai,aj,mief->efmn', g_bbbb[o, o, v, o], t1_bb, l2_abab, optimize=True)
    l2_res -= 1 * _tmp60
    _tmp61 = einsum('jmai,aj,inef->efmn', g_aaaa[o, o, v, o], t1_aa, l2_abab, optimize=True)
    l2_res -= 1 * _tmp61
    _tmp62 = einsum('mjia,aj,inef->efmn', g_abab[o, o, o, v], t1_bb, l2_abab, optimize=True)
    l2_res -= 1 * _tmp62
    _tmp63 = einsum('bief,mnba,ai->efmn', g_abab[v, o, v, v], l2_abab, t1_bb, optimize=True)
    l2_res -= 1 * _tmp63
    _tmp64 = einsum('ibef,mnab,ai->efmn', g_abab[o, v, v, v], l2_abab, t1_aa, optimize=True)
    l2_res -= 1 * _tmp64
    _tmp65 = einsum('aneb,miaf,bi->efmn', g_abab[v, o, v, v], l2_abab, t1_bb, optimize=True)
    l2_res -= 1 * _tmp65
    _tmp66 = einsum('anbf,imae,bi->efmn', g_abab[v, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l2_res += 1 * _tmp66
    _tmp67 = einsum('nafb,miea,bi->efmn', g_bbbb[o, v, v, v], l2_abab, t1_bb, optimize=True)
    l2_res += 1 * _tmp67
    _tmp68 = einsum('maeb,inaf,bi->efmn', g_aaaa[o, v, v, v], l2_abab, t1_aa, optimize=True)
    l2_res += 1 * _tmp68
    _tmp69 = einsum('maeb,inaf,bi->efmn', g_abab[o, v, v, v], l2_bbbb, t1_bb, optimize=True)
    l2_res += 1 * _tmp69
    _tmp70 = einsum('mabf,inea,bi->efmn', g_abab[o, v, v, v], l2_abab, t1_aa, optimize=True)
    l2_res -= 1 * _tmp70
    _tmp71 = einsum('iaeb,bi,mnaf->efmn', g_aaaa[o, v, v, v], t1_aa, l2_abab, optimize=True)
    l2_res -= 1 * _tmp71
    _tmp72 = einsum('aieb,bi,mnaf->efmn', g_abab[v, o, v, v], t1_bb, l2_abab, optimize=True)
    l2_res += 1 * _tmp72
    _tmp73 = einsum('iabf,bi,mnea->efmn', g_abab[o, v, v, v], t1_aa, l2_abab, optimize=True)
    l2_res += 1 * _tmp73
    _tmp74 = einsum('iafb,bi,mnea->efmn', g_bbbb[o, v, v, v], t1_bb, l2_abab, optimize=True)
    l2_res -= 1 * _tmp74
    _tmp75 = einsum('knij,ak,mijeaf->efmn', g_abab[o, o, o, o], t1_aa, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp75
    l2_res += 0.5 * _tmp75
    _tmp76 = einsum('knij,ak,mijefa->efmn', g_bbbb[o, o, o, o], t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp76
    _tmp77 = einsum('kmij,ak,jineaf->efmn', g_aaaa[o, o, o, o], t1_aa, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp77
    _tmp78 = einsum('mkij,ak,injefa->efmn', g_abab[o, o, o, o], t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp78
    _tmp79 = einsum('mkji,ak,jinefa->efmn', g_abab[o, o, o, o], t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp79
    _tmp80 = einsum('jbei,aj,minbaf->efmn', g_aaaa[o, v, v, o], t1_aa, l3_aabaab, optimize=True)
    l2_res += 1 * _tmp80
    _tmp81 = einsum('jbei,aj,mniafb->efmn', g_abab[o, v, v, o], t1_aa, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp81
    _tmp82 = einsum('bjei,aj,mnibfa->efmn', g_abab[v, o, v, o], t1_bb, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp82
    _tmp83 = einsum('jbif,aj,minaeb->efmn', g_abab[o, v, o, v], t1_aa, l3_aabaab, optimize=True)
    l2_res += 1 * _tmp83
    _tmp84 = einsum('bjif,aj,minbea->efmn', g_abab[v, o, o, v], t1_bb, l3_aabaab, optimize=True)
    l2_res += 1 * _tmp84
    _tmp85 = einsum('jbfi,aj,mnieba->efmn', g_bbbb[o, v, v, o], t1_bb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp85
    _tmp86 = einsum('anbj,bi,imjaef->efmn', g_abab[v, o, v, o], t1_aa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp86
    _tmp87 = einsum('anjb,bi,jmiaef->efmn', g_abab[v, o, o, v], t1_bb, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp87
    _tmp88 = einsum('nabj,bi,mijeaf->efmn', g_bbbb[o, v, v, o], t1_bb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp88
    _tmp89 = einsum('mabj,bi,ijnaef->efmn', g_aaaa[o, v, v, o], t1_aa, l3_aabaab, optimize=True)
    l2_res += 1 * _tmp89
    _tmp90 = einsum('mabj,bi,injeaf->efmn', g_abab[o, v, v, o], t1_aa, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp90
    _tmp91 = einsum('majb,bi,jnieaf->efmn', g_abab[o, v, o, v], t1_bb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp91
    _tmp92 = einsum('baec,ci,imnbaf->efmn', g_aaaa[v, v, v, v], t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp92
    _tmp93 = einsum('baec,ci,minbaf->efmn', g_abab[v, v, v, v], t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp93
    l2_res += 0.5 * _tmp93
    _tmp94 = einsum('bacf,ci,imnbea->efmn', g_abab[v, v, v, v], t1_aa, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp94
    _tmp95 = einsum('abcf,ci,imneab->efmn', g_abab[v, v, v, v], t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp95
    _tmp96 = einsum('bafc,ci,mineab->efmn', g_bbbb[v, v, v, v], t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp96
    _tmp97 = einsum('jnef,baij,imba->efmn', g_abab[o, o, v, v], t2_aaaa, l2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp97
    _tmp98 = einsum('jnef,baji,miba->efmn', g_abab[o, o, v, v], t2_abab, l2_abab, optimize=True)
    l2_res -= 0.5 * _tmp98
    l2_res -= 0.5 * _tmp98
    _tmp99 = einsum('mjef,baij,inba->efmn', g_abab[o, o, v, v], t2_abab, l2_abab, optimize=True)
    l2_res -= 0.5 * _tmp99
    l2_res -= 0.5 * _tmp99
    _tmp100 = einsum('mjef,baij,inba->efmn', g_abab[o, o, v, v], t2_bbbb, l2_bbbb, optimize=True)
    l2_res -= 0.5 * _tmp100
    _tmp101 = einsum('jief,mnba,baji->efmn', g_abab[o, o, v, v], l2_abab, t2_abab, optimize=True)
    l2_res += 0.25 * _tmp101
    l2_res += 0.25 * _tmp101
    l2_res += 0.25 * _tmp101
    l2_res += 0.25 * _tmp101
    _tmp102 = einsum('mneb,abij,ijaf->efmn', g_abab[o, o, v, v], t2_abab, l2_abab, optimize=True)
    l2_res -= 0.5 * _tmp102
    l2_res -= 0.5 * _tmp102
    _tmp103 = einsum('mneb,baij,ijfa->efmn', g_abab[o, o, v, v], t2_bbbb, l2_bbbb, optimize=True)
    l2_res -= 0.5 * _tmp103
    _tmp104 = einsum('mnbf,baij,ijea->efmn', g_abab[o, o, v, v], t2_aaaa, l2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp104
    _tmp105 = einsum('mnbf,baij,ijea->efmn', g_abab[o, o, v, v], t2_abab, l2_abab, optimize=True)
    l2_res -= 0.5 * _tmp105
    l2_res -= 0.5 * _tmp105
    _tmp106 = einsum('jneb,miaf,abji->efmn', g_abab[o, o, v, v], l2_abab, t2_abab, optimize=True)
    l2_res += 1 * _tmp106
    _tmp107 = einsum('jnbf,imea,baij->efmn', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, optimize=True)
    l2_res += 1 * _tmp107
    _tmp108 = einsum('jnfb,imea,abij->efmn', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, optimize=True)
    l2_res += 1 * _tmp108
    _tmp109 = einsum('jnbf,miea,baji->efmn', g_abab[o, o, v, v], l2_abab, t2_abab, optimize=True)
    l2_res += 1 * _tmp109
    _tmp110 = einsum('jnfb,miea,baij->efmn', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, optimize=True)
    l2_res += 1 * _tmp110
    _tmp111 = einsum('jmeb,inaf,baij->efmn', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, optimize=True)
    l2_res += 1 * _tmp111
    _tmp112 = einsum('mjeb,inaf,abij->efmn', g_abab[o, o, v, v], l2_abab, t2_abab, optimize=True)
    l2_res += 1 * _tmp112
    _tmp113 = einsum('jmeb,infa,baji->efmn', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, optimize=True)
    l2_res += 1 * _tmp113
    _tmp114 = einsum('mjeb,infa,baij->efmn', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, optimize=True)
    l2_res += 1 * _tmp114
    _tmp115 = einsum('mjbf,inea,baij->efmn', g_abab[o, o, v, v], l2_abab, t2_abab, optimize=True)
    l2_res += 1 * _tmp115
    _tmp116 = einsum('jieb,baji,mnaf->efmn', g_aaaa[o, o, v, v], t2_aaaa, l2_abab, optimize=True)
    l2_res += 0.5 * _tmp116
    _tmp117 = einsum('jieb,abji,mnaf->efmn', g_abab[o, o, v, v], t2_abab, l2_abab, optimize=True)
    l2_res -= 0.5 * _tmp117
    l2_res -= 0.5 * _tmp117
    _tmp118 = einsum('jibf,baji,mnea->efmn', g_abab[o, o, v, v], t2_abab, l2_abab, optimize=True)
    l2_res -= 0.5 * _tmp118
    l2_res -= 0.5 * _tmp118
    _tmp119 = einsum('jifb,baji,mnea->efmn', g_bbbb[o, o, v, v], t2_bbbb, l2_abab, optimize=True)
    l2_res += 0.5 * _tmp119
    _tmp120 = einsum('mnab,ijef,abij->efmn', g_abab[o, o, v, v], l2_abab, t2_abab, optimize=True)
    l2_res += 0.25 * _tmp120
    l2_res += 0.25 * _tmp120
    l2_res += 0.25 * _tmp120
    l2_res += 0.25 * _tmp120
    _tmp121 = einsum('jnab,abji,mief->efmn', g_abab[o, o, v, v], t2_abab, l2_abab, optimize=True)
    l2_res -= 0.5 * _tmp121
    l2_res -= 0.5 * _tmp121
    _tmp122 = einsum('jnab,abij,mief->efmn', g_bbbb[o, o, v, v], t2_bbbb, l2_abab, optimize=True)
    l2_res += 0.5 * _tmp122
    _tmp123 = einsum('jmab,abij,inef->efmn', g_aaaa[o, o, v, v], t2_aaaa, l2_abab, optimize=True)
    l2_res += 0.5 * _tmp123
    _tmp124 = einsum('mjab,abij,inef->efmn', g_abab[o, o, v, v], t2_abab, l2_abab, optimize=True)
    l2_res -= 0.5 * _tmp124
    l2_res -= 0.5 * _tmp124
    _tmp125 = einsum('knej,imjabf,baik->efmn', g_abab[o, o, v, o], l3_aabaab, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp125
    _tmp126 = einsum('knej,mijbfa,baki->efmn', g_abab[o, o, v, o], l3_abbabb, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp126
    _tmp127 = einsum('knej,mijabf,abki->efmn', g_abab[o, o, v, o], l3_abbabb, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp127
    _tmp128 = einsum('knjf,imjeba,baik->efmn', g_abab[o, o, o, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res += 0.5 * _tmp128
    _tmp129 = einsum('knfj,imjeba,baik->efmn', g_bbbb[o, o, v, o], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp129
    l2_res -= 0.5 * _tmp129
    _tmp130 = einsum('knjf,jmieba,baki->efmn', g_abab[o, o, o, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp130
    l2_res += 0.5 * _tmp130
    _tmp131 = einsum('knfj,mijeba,baik->efmn', g_bbbb[o, o, v, o], l3_abbabb, t2_bbbb, optimize=True)
    l2_res += 0.5 * _tmp131
    _tmp132 = einsum('kmej,ijnabf,baik->efmn', g_aaaa[o, o, v, o], l3_aabaab, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp132
    _tmp133 = einsum('mkej,injbfa,baik->efmn', g_abab[o, o, v, o], l3_abbabb, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp133
    _tmp134 = einsum('mkej,injabf,abik->efmn', g_abab[o, o, v, o], l3_abbabb, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp134
    _tmp135 = einsum('kmej,jnibfa,baki->efmn', g_aaaa[o, o, v, o], l3_abbabb, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp135
    _tmp136 = einsum('kmej,jniabf,abki->efmn', g_aaaa[o, o, v, o], l3_abbabb, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp136
    _tmp137 = einsum('mkej,injfba,baik->efmn', g_abab[o, o, v, o], l3_bbbbbb, t2_bbbb, optimize=True)
    l2_res += 0.5 * _tmp137
    _tmp138 = einsum('mkjf,ijneba,baik->efmn', g_abab[o, o, o, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp138
    l2_res -= 0.5 * _tmp138
    _tmp139 = einsum('mkjf,jnieba,baik->efmn', g_abab[o, o, o, v], l3_abbabb, t2_bbbb, optimize=True)
    l2_res -= 0.5 * _tmp139
    _tmp140 = einsum('kjei,bakj,minabf->efmn', g_aaaa[o, o, v, o], t2_aaaa, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp140
    _tmp141 = einsum('kjei,bakj,mnibfa->efmn', g_abab[o, o, v, o], t2_abab, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp141
    l2_res += 0.25 * _tmp141
    _tmp142 = einsum('kjei,abkj,mniabf->efmn', g_abab[o, o, v, o], t2_abab, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp142
    l2_res -= 0.25 * _tmp142
    _tmp143 = einsum('kjif,bakj,mineba->efmn', g_abab[o, o, o, v], t2_abab, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp143
    l2_res += 0.25 * _tmp143
    l2_res += 0.25 * _tmp143
    l2_res += 0.25 * _tmp143
    _tmp144 = einsum('kjfi,bakj,mnieba->efmn', g_bbbb[o, o, v, o], t2_bbbb, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp144
    _tmp145 = einsum('mnbk,ijkeaf,baij->efmn', g_abab[o, o, v, o], l3_aabaab, t2_aaaa, optimize=True)
    l2_res += 0.5 * _tmp145
    _tmp146 = einsum('mnkb,ikjeaf,abij->efmn', g_abab[o, o, o, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp146
    _tmp147 = einsum('mnbk,ijkefa,baij->efmn', g_abab[o, o, v, o], l3_abbabb, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp147
    _tmp148 = einsum('mnkb,kjieaf,abji->efmn', g_abab[o, o, o, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp148
    l2_res -= 0.5 * _tmp147
    _tmp149 = einsum('mnkb,kjiefa,baij->efmn', g_abab[o, o, o, v], l3_abbabb, t2_bbbb, optimize=True)
    l2_res -= 0.5 * _tmp149
    _tmp150 = einsum('knbj,baik,imjeaf->efmn', g_abab[o, o, v, o], t2_aaaa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp150
    _tmp151 = einsum('knbj,baki,mijefa->efmn', g_abab[o, o, v, o], t2_abab, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp151
    _tmp152 = einsum('knbj,abik,imjeaf->efmn', g_bbbb[o, o, v, o], t2_abab, l3_aabaab, optimize=True)
    l2_res += 1 * _tmp152
    _tmp153 = einsum('knjb,abki,jmieaf->efmn', g_abab[o, o, o, v], t2_abab, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp153
    _tmp154 = einsum('knbj,baik,mijefa->efmn', g_bbbb[o, o, v, o], t2_bbbb, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp154
    _tmp155 = einsum('kmbj,baik,ijneaf->efmn', g_aaaa[o, o, v, o], t2_aaaa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp155
    _tmp156 = einsum('mkbj,baik,injefa->efmn', g_abab[o, o, v, o], t2_abab, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp156
    _tmp157 = einsum('kmbj,baki,jniefa->efmn', g_aaaa[o, o, v, o], t2_abab, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp157
    _tmp158 = einsum('mkjb,abik,ijneaf->efmn', g_abab[o, o, o, v], t2_abab, l3_aabaab, optimize=True)
    l2_res += 1 * _tmp158
    _tmp159 = einsum('mkjb,baik,jniefa->efmn', g_abab[o, o, o, v], t2_bbbb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp159
    _tmp160 = einsum('cjef,imncba,baij->efmn', g_abab[v, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp160
    l2_res += 0.5 * _tmp160
    _tmp161 = einsum('jcef,imnabc,baij->efmn', g_abab[o, v, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l2_res += 0.5 * _tmp161
    _tmp162 = einsum('cjef,mincba,baij->efmn', g_abab[v, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l2_res -= 0.5 * _tmp162
    _tmp163 = einsum('jcef,minbca,baji->efmn', g_abab[o, v, v, v], l3_abbabb, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp163
    _tmp164 = einsum('jcef,minabc,abji->efmn', g_abab[o, v, v, v], l3_abbabb, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp164
    _tmp165 = einsum('bnec,imjbaf,acij->efmn', g_abab[v, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp165
    _tmp166 = einsum('bnec,mjibaf,acji->efmn', g_abab[v, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp166
    _tmp167 = einsum('bnec,mjibfa,caij->efmn', g_abab[v, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l2_res += 0.5 * _tmp167
    _tmp168 = einsum('bncf,ijmbea,caij->efmn', g_abab[v, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l2_res -= 0.5 * _tmp168
    _tmp169 = einsum('bncf,imjbea,caij->efmn', g_abab[v, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp169
    _tmp170 = einsum('nbfc,imjaeb,acij->efmn', g_bbbb[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp170
    _tmp171 = einsum('bncf,mjibea,caji->efmn', g_abab[v, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp171
    _tmp172 = einsum('nbfc,mjiaeb,acji->efmn', g_bbbb[o, v, v, v], l3_aabaab, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp172
    _tmp173 = einsum('nbfc,mjieba,caij->efmn', g_bbbb[o, v, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l2_res -= 0.5 * _tmp173
    _tmp174 = einsum('mbec,ijnbaf,caij->efmn', g_aaaa[o, v, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l2_res += 0.5 * _tmp174
    _tmp175 = einsum('mbec,ijnbfa,caij->efmn', g_aaaa[o, v, v, v], l3_abbabb, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp175
    _tmp176 = einsum('mbec,ijnafb,acij->efmn', g_abab[o, v, v, v], l3_abbabb, t2_abab, optimize=True)
    l2_res -= 0.5 * _tmp176
    l2_res -= 0.5 * _tmp175
    l2_res -= 0.5 * _tmp176
    _tmp177 = einsum('mbec,ijnbfa,caij->efmn', g_abab[o, v, v, v], l3_bbbbbb, t2_bbbb, optimize=True)
    l2_res -= 0.5 * _tmp177
    _tmp178 = einsum('mbcf,ijnaeb,caij->efmn', g_abab[o, v, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l2_res += 0.5 * _tmp178
    _tmp179 = einsum('mbcf,ijneba,caij->efmn', g_abab[o, v, v, v], l3_abbabb, t2_abab, optimize=True)
    l2_res += 0.5 * _tmp179
    l2_res += 0.5 * _tmp179
    _tmp180 = einsum('jbec,caij,imnbaf->efmn', g_aaaa[o, v, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp180
    _tmp181 = einsum('jbec,caji,minbfa->efmn', g_aaaa[o, v, v, v], t2_abab, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp181
    _tmp182 = einsum('bjec,acij,imnbaf->efmn', g_abab[v, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp182
    _tmp183 = einsum('jbec,acji,minafb->efmn', g_abab[o, v, v, v], t2_abab, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp183
    _tmp184 = einsum('bjec,caij,minbfa->efmn', g_abab[v, o, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp184
    _tmp185 = einsum('jbcf,caij,imnaeb->efmn', g_abab[o, v, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp185
    _tmp186 = einsum('bjcf,caij,imnbea->efmn', g_abab[v, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp186
    _tmp187 = einsum('jbcf,caji,mineba->efmn', g_abab[o, v, v, v], t2_abab, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp187
    _tmp188 = einsum('jbfc,acij,imnaeb->efmn', g_bbbb[o, v, v, v], t2_abab, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp188
    _tmp189 = einsum('jbfc,caij,mineba->efmn', g_bbbb[o, v, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp189
    _tmp190 = einsum('anbc,bcij,imjaef->efmn', g_abab[v, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp190
    _tmp191 = einsum('anbc,bcji,mjiaef->efmn', g_abab[v, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp191
    l2_res -= 0.25 * _tmp190
    l2_res += 0.25 * _tmp191
    _tmp192 = einsum('nabc,bcij,mjieaf->efmn', g_bbbb[o, v, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp192
    _tmp193 = einsum('mabc,bcij,ijnaef->efmn', g_aaaa[o, v, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp193
    _tmp194 = einsum('mabc,bcij,ijneaf->efmn', g_abab[o, v, v, v], t2_abab, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp194
    l2_res -= 0.25 * _tmp194
    l2_res -= 0.25 * _tmp194
    l2_res -= 0.25 * _tmp194
    _tmp195 = einsum('knef,cbaijk,ijmcba->efmn', g_abab[o, o, v, v], t3_aaaaaa, l3_aaaaaa, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp195
    _tmp196 = einsum('knef,cbaikj,imjcba->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp196
    _tmp197 = einsum('knef,cbakji,mjicba->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp197
    l2_res -= 0.0833333333333333 * _tmp196
    l2_res -= 0.0833333333333333 * _tmp197
    _tmp198 = einsum('knef,cbakji,mjicba->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp198
    l2_res -= 0.0833333333333333 * _tmp196
    l2_res -= 0.0833333333333333 * _tmp197
    l2_res -= 0.0833333333333333 * _tmp198
    l2_res -= 0.0833333333333333 * _tmp198
    _tmp199 = einsum('mkef,cbaijk,ijncba->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp199
    l2_res -= 0.0833333333333333 * _tmp199
    _tmp200 = einsum('mkef,cbaijk,ijncba->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp200
    l2_res -= 0.0833333333333333 * _tmp200
    l2_res -= 0.0833333333333333 * _tmp199
    l2_res -= 0.0833333333333333 * _tmp200
    l2_res -= 0.0833333333333333 * _tmp200
    l2_res -= 0.0833333333333333 * _tmp200
    l2_res -= 0.0833333333333333 * _tmp200
    _tmp201 = einsum('mkef,cbaijk,ijncba->efmn', g_abab[o, o, v, v], t3_bbbbbb, l3_bbbbbb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp201
    _tmp202 = einsum('kjef,imncba,cbaikj->efmn', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.0833333333333333 * _tmp202
    l2_res += 0.0833333333333333 * _tmp202
    l2_res += 0.0833333333333333 * _tmp202
    l2_res += 0.0833333333333333 * _tmp202
    l2_res += 0.0833333333333333 * _tmp202
    l2_res += 0.0833333333333333 * _tmp202
    _tmp203 = einsum('kjef,mincba,cbakij->efmn', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res += 0.0833333333333333 * _tmp203
    _tmp204 = einsum('jkef,mincba,cbajki->efmn', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp204
    l2_res += 0.0833333333333333 * _tmp203
    l2_res -= 0.0833333333333333 * _tmp204
    l2_res += 0.0833333333333333 * _tmp203
    l2_res -= 0.0833333333333333 * _tmp204
    _tmp205 = einsum('mnec,abcijk,ijkabf->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp205
    l2_res -= 0.0833333333333333 * _tmp205
    l2_res -= 0.0833333333333333 * _tmp205
    _tmp206 = einsum('mnec,bcaijk,ijkbfa->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp206
    l2_res -= 0.0833333333333333 * _tmp206
    l2_res -= 0.0833333333333333 * _tmp206
    _tmp207 = einsum('mnec,abcijk,ijkabf->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp207
    l2_res -= 0.0833333333333333 * _tmp207
    l2_res -= 0.0833333333333333 * _tmp207
    _tmp208 = einsum('mnec,cbaijk,ijkfba->efmn', g_abab[o, o, v, v], t3_bbbbbb, l3_bbbbbb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp208
    _tmp209 = einsum('mncf,cbaijk,ijkeba->efmn', g_abab[o, o, v, v], t3_aaaaaa, l3_aaaaaa, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp209
    _tmp210 = einsum('mncf,cbaijk,ijkeba->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp210
    l2_res -= 0.0833333333333333 * _tmp210
    l2_res -= 0.0833333333333333 * _tmp210
    l2_res -= 0.0833333333333333 * _tmp210
    l2_res -= 0.0833333333333333 * _tmp210
    l2_res -= 0.0833333333333333 * _tmp210
    _tmp211 = einsum('mncf,cbaijk,ijkeba->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp211
    l2_res -= 0.0833333333333333 * _tmp211
    l2_res -= 0.0833333333333333 * _tmp211
    _tmp212 = einsum('knec,imjabf,abcikj->efmn', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp212
    _tmp213 = einsum('knec,mjiabf,abckji->efmn', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp213
    _tmp214 = einsum('knec,mjibfa,bcakji->efmn', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp214
    _tmp215 = einsum('knec,mjiabf,abckji->efmn', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp215
    _tmp216 = einsum('kncf,ijmeba,cbaijk->efmn', g_abab[o, o, v, v], l3_aaaaaa, t3_aaaaaa, optimize=True)
    l2_res += 0.25 * _tmp216
    _tmp217 = einsum('knfc,ijmeba,abcijk->efmn', g_bbbb[o, o, v, v], l3_aaaaaa, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp217
    _tmp218 = einsum('kncf,imjeba,cbaikj->efmn', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp218
    _tmp219 = einsum('knfc,imjeba,bcaijk->efmn', g_bbbb[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp219
    l2_res += 0.25 * _tmp218
    _tmp220 = einsum('knfc,imjeab,abcijk->efmn', g_bbbb[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp220
    _tmp221 = einsum('kncf,mjieba,cbakji->efmn', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp221
    _tmp222 = einsum('knfc,mjieba,bcajik->efmn', g_bbbb[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp222
    l2_res += 0.25 * _tmp221
    _tmp223 = einsum('knfc,mjieab,abcjik->efmn', g_bbbb[o, o, v, v], l3_aabaab, t3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp223
    _tmp224 = einsum('kncf,mjieba,cbakji->efmn', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp224
    _tmp225 = einsum('knfc,mjieba,cbaijk->efmn', g_bbbb[o, o, v, v], l3_abbabb, t3_bbbbbb, optimize=True)
    l2_res += 0.25 * _tmp225
    _tmp226 = einsum('kmec,ijnabf,cbaijk->efmn', g_aaaa[o, o, v, v], l3_aabaab, t3_aaaaaa, optimize=True)
    l2_res += 0.25 * _tmp226
    _tmp227 = einsum('mkec,ijnabf,abcijk->efmn', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp227
    _tmp228 = einsum('kmec,ijnbfa,cbaikj->efmn', g_aaaa[o, o, v, v], l3_abbabb, t3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp228
    _tmp229 = einsum('mkec,ijnbfa,bcaijk->efmn', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp229
    _tmp230 = einsum('kmec,ijnabf,cabikj->efmn', g_aaaa[o, o, v, v], l3_abbabb, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp230
    _tmp231 = einsum('mkec,ijnabf,abcijk->efmn', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp231
    _tmp232 = einsum('kmec,jinbfa,cbakji->efmn', g_aaaa[o, o, v, v], l3_abbabb, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp232
    l2_res += 0.25 * _tmp229
    _tmp233 = einsum('kmec,jinabf,cabkji->efmn', g_aaaa[o, o, v, v], l3_abbabb, t3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp233
    l2_res += 0.25 * _tmp231
    _tmp234 = einsum('kmec,ijnfba,cbakji->efmn', g_aaaa[o, o, v, v], l3_bbbbbb, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp234
    _tmp235 = einsum('mkec,ijnfba,cbaijk->efmn', g_abab[o, o, v, v], l3_bbbbbb, t3_bbbbbb, optimize=True)
    l2_res += 0.25 * _tmp235
    _tmp236 = einsum('mkcf,ijneba,cbaijk->efmn', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp236
    l2_res += 0.25 * _tmp236
    _tmp237 = einsum('mkcf,ijneba,cbaijk->efmn', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp237
    l2_res += 0.25 * _tmp237
    _tmp238 = einsum('kjec,cbaikj,imnabf->efmn', g_aaaa[o, o, v, v], t3_aaaaaa, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp238
    _tmp239 = einsum('kjec,cbajki,minbfa->efmn', g_aaaa[o, o, v, v], t3_aabaab, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp239
    _tmp240 = einsum('kjec,cabjki,minabf->efmn', g_aaaa[o, o, v, v], t3_aabaab, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp240
    _tmp241 = einsum('kjec,abcikj,imnabf->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp241
    l2_res -= 0.25 * _tmp241
    _tmp242 = einsum('kjec,bcakij,minbfa->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp242
    _tmp243 = einsum('jkec,bcajki,minbfa->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp243
    _tmp244 = einsum('kjec,abckij,minabf->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp244
    _tmp245 = einsum('jkec,abcjki,minabf->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp245
    _tmp246 = einsum('kjcf,cbaikj,imneba->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp246
    l2_res -= 0.25 * _tmp246
    l2_res -= 0.25 * _tmp246
    l2_res -= 0.25 * _tmp246
    _tmp247 = einsum('kjcf,cbakij,mineba->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp247
    _tmp248 = einsum('jkcf,cbajki,mineba->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp248
    _tmp249 = einsum('kjfc,bcaikj,imneba->efmn', g_bbbb[o, o, v, v], t3_abbabb, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp249
    _tmp250 = einsum('kjfc,abcikj,imneab->efmn', g_bbbb[o, o, v, v], t3_abbabb, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp250
    _tmp251 = einsum('kjfc,cbaikj,mineba->efmn', g_bbbb[o, o, v, v], t3_bbbbbb, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp251
    _tmp252 = einsum('mnbc,ijkeaf,bacijk->efmn', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res += 0.0833333333333333 * _tmp252
    _tmp253 = einsum('mncb,ijkeaf,acbijk->efmn', g_abab[o, o, v, v], l3_aabaab, t3_aabaab, optimize=True)
    l2_res -= 0.0833333333333333 * _tmp253
    l2_res += 0.0833333333333333 * _tmp252
    l2_res -= 0.0833333333333333 * _tmp253
    _tmp254 = einsum('mnbc,ijkefa,bcaijk->efmn', g_abab[o, o, v, v], l3_abbabb, t3_abbabb, optimize=True)
    l2_res += 0.0833333333333333 * _tmp254
    l2_res += 0.0833333333333333 * _tmp254
    l2_res += 0.0833333333333333 * _tmp252
    l2_res -= 0.0833333333333333 * _tmp253
    l2_res += 0.0833333333333333 * _tmp254
    l2_res += 0.0833333333333333 * _tmp254
    l2_res += 0.0833333333333333 * _tmp254
    l2_res += 0.0833333333333333 * _tmp254
    _tmp255 = einsum('knbc,bacikj,imjeaf->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp255
    _tmp256 = einsum('knbc,backji,mjieaf->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp256
    _tmp257 = einsum('knbc,bcakji,mjiefa->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp257
    _tmp258 = einsum('kncb,acbikj,imjeaf->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp258
    _tmp259 = einsum('kncb,acbkji,mjieaf->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp259
    l2_res -= 0.25 * _tmp257
    _tmp260 = einsum('knbc,acbijk,imjeaf->efmn', g_bbbb[o, o, v, v], t3_abbabb, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp260
    _tmp261 = einsum('knbc,acbjik,mjieaf->efmn', g_bbbb[o, o, v, v], t3_abbabb, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp261
    _tmp262 = einsum('knbc,bcaijk,mjiefa->efmn', g_bbbb[o, o, v, v], t3_bbbbbb, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp262
    _tmp263 = einsum('kmbc,bcaijk,ijneaf->efmn', g_aaaa[o, o, v, v], t3_aaaaaa, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp263
    _tmp264 = einsum('kmbc,bcaikj,ijnefa->efmn', g_aaaa[o, o, v, v], t3_aabaab, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp264
    _tmp265 = einsum('kmbc,bcakji,jinefa->efmn', g_aaaa[o, o, v, v], t3_aabaab, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp265
    _tmp266 = einsum('mkbc,bacijk,ijneaf->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp266
    _tmp267 = einsum('mkbc,bcaijk,ijnefa->efmn', g_abab[o, o, v, v], t3_abbabb, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp267
    l2_res -= 0.25 * _tmp267
    _tmp268 = einsum('mkcb,acbijk,ijneaf->efmn', g_abab[o, o, v, v], t3_aabaab, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp268
    l2_res -= 0.25 * _tmp267
    l2_res -= 0.25 * _tmp267
    _tmp269 = einsum('jkef,imncba,baik,cj->efmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp269
    l2_res -= 0.5 * _tmp269
    _tmp270 = einsum('kjef,imnabc,baik,cj->efmn', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp270
    _tmp271 = einsum('jkef,mincba,baik,cj->efmn', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp271
    _tmp272 = einsum('kjef,minbca,baki,cj->efmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp272
    _tmp273 = einsum('kjef,minabc,abki,cj->efmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp273
    _tmp274 = einsum('knec,imjabf,baik,cj->efmn', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp274
    _tmp275 = einsum('knec,mjibfa,baki,cj->efmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp275
    _tmp276 = einsum('knec,mjiabf,abki,cj->efmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp276
    _tmp277 = einsum('kncf,ijmeba,baik,cj->efmn', g_abab[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp277
    _tmp278 = einsum('knfc,imjeba,baik,cj->efmn', g_bbbb[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp278
    l2_res -= 0.5 * _tmp278
    _tmp279 = einsum('kncf,mjieba,baki,cj->efmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp279
    l2_res -= 0.5 * _tmp279
    _tmp280 = einsum('knfc,mjieba,baik,cj->efmn', g_bbbb[o, o, v, v], l3_abbabb, t2_bbbb, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp280
    _tmp281 = einsum('kmec,ijnabf,baik,cj->efmn', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp281
    _tmp282 = einsum('mkec,ijnbfa,baik,cj->efmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp282
    _tmp283 = einsum('mkec,ijnabf,abik,cj->efmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp283
    _tmp284 = einsum('kmec,jinbfa,baki,cj->efmn', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp284
    _tmp285 = einsum('kmec,jinabf,abki,cj->efmn', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp285
    _tmp286 = einsum('mkec,ijnfba,baik,cj->efmn', g_abab[o, o, v, v], l3_bbbbbb, t2_bbbb, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp286
    _tmp287 = einsum('mkcf,ijneba,baik,cj->efmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp287
    l2_res -= 0.5 * _tmp287
    _tmp288 = einsum('mkcf,jineba,baik,cj->efmn', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp288
    _tmp289 = einsum('knec,imjabf,acij,bk->efmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp289
    _tmp290 = einsum('knec,mjiabf,acji,bk->efmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp290
    _tmp291 = einsum('knec,mjibfa,caij,bk->efmn', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp291
    _tmp292 = einsum('kncf,ijmeba,caij,bk->efmn', g_abab[o, o, v, v], l3_aaaaaa, t2_aaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp292
    _tmp293 = einsum('kncf,imjeba,caij,bk->efmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp293
    _tmp294 = einsum('knfc,imjeab,acij,bk->efmn', g_bbbb[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp294
    _tmp295 = einsum('kncf,mjieba,caji,bk->efmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp295
    _tmp296 = einsum('knfc,mjieab,acji,bk->efmn', g_bbbb[o, o, v, v], l3_aabaab, t2_abab, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp296
    _tmp297 = einsum('knfc,mjieba,caij,bk->efmn', g_bbbb[o, o, v, v], l3_abbabb, t2_bbbb, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp297
    _tmp298 = einsum('kmec,ijnabf,caij,bk->efmn', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp298
    _tmp299 = einsum('kmec,ijnbfa,caij,bk->efmn', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp299
    _tmp300 = einsum('mkec,ijnabf,acij,bk->efmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp300
    l2_res -= 0.5 * _tmp299
    l2_res -= 0.5 * _tmp300
    _tmp301 = einsum('mkec,ijnfba,caij,bk->efmn', g_abab[o, o, v, v], l3_bbbbbb, t2_bbbb, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp301
    _tmp302 = einsum('mkcf,ijneab,caij,bk->efmn', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp302
    _tmp303 = einsum('mkcf,ijneba,caij,bk->efmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp303
    l2_res -= 0.5 * _tmp303
    _tmp304 = einsum('kjec,baik,cj,imnabf->efmn', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp304
    _tmp305 = einsum('kjec,baik,cj,imnabf->efmn', g_abab[o, o, v, v], t2_aaaa, t1_bb, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp305
    _tmp306 = einsum('kjec,baki,cj,minbfa->efmn', g_aaaa[o, o, v, v], t2_abab, t1_aa, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp306
    _tmp307 = einsum('kjec,baki,cj,minbfa->efmn', g_abab[o, o, v, v], t2_abab, t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp307
    _tmp308 = einsum('kjec,abki,cj,minabf->efmn', g_aaaa[o, o, v, v], t2_abab, t1_aa, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp308
    _tmp309 = einsum('kjec,abki,cj,minabf->efmn', g_abab[o, o, v, v], t2_abab, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp309
    _tmp310 = einsum('jkcf,baik,cj,imneba->efmn', g_abab[o, o, v, v], t2_abab, t1_aa, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp310
    _tmp311 = einsum('kjfc,baik,cj,imneba->efmn', g_bbbb[o, o, v, v], t2_abab, t1_bb, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp311
    l2_res += 0.5 * _tmp310
    l2_res += 0.5 * _tmp311
    _tmp312 = einsum('jkcf,baik,cj,mineba->efmn', g_abab[o, o, v, v], t2_bbbb, t1_aa, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp312
    _tmp313 = einsum('kjfc,baik,cj,mineba->efmn', g_bbbb[o, o, v, v], t2_bbbb, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp313
    _tmp314 = einsum('kjec,bakj,ci,imnabf->efmn', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp314
    _tmp315 = einsum('kjec,bakj,ci,minbfa->efmn', g_abab[o, o, v, v], t2_abab, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp315
    l2_res -= 0.25 * _tmp315
    _tmp316 = einsum('kjec,abkj,ci,minabf->efmn', g_abab[o, o, v, v], t2_abab, t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp316
    l2_res += 0.25 * _tmp316
    _tmp317 = einsum('kjcf,bakj,ci,imneba->efmn', g_abab[o, o, v, v], t2_abab, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp317
    l2_res -= 0.25 * _tmp317
    l2_res -= 0.25 * _tmp317
    l2_res -= 0.25 * _tmp317
    _tmp318 = einsum('kjfc,bakj,ci,mineba->efmn', g_bbbb[o, o, v, v], t2_bbbb, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp318
    _tmp319 = einsum('kjec,caik,bj,imnabf->efmn', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp319
    _tmp320 = einsum('kjec,caki,bj,minbfa->efmn', g_aaaa[o, o, v, v], t2_abab, t1_aa, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp320
    _tmp321 = einsum('jkec,acik,bj,imnabf->efmn', g_abab[o, o, v, v], t2_abab, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp321
    _tmp322 = einsum('kjec,acki,bj,minabf->efmn', g_abab[o, o, v, v], t2_abab, t1_bb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp322
    _tmp323 = einsum('jkec,caik,bj,minbfa->efmn', g_abab[o, o, v, v], t2_bbbb, t1_aa, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp323
    _tmp324 = einsum('kjcf,caik,bj,imneab->efmn', g_abab[o, o, v, v], t2_aaaa, t1_bb, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp324
    _tmp325 = einsum('jkcf,caik,bj,imneba->efmn', g_abab[o, o, v, v], t2_abab, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp325
    _tmp326 = einsum('kjcf,caki,bj,mineba->efmn', g_abab[o, o, v, v], t2_abab, t1_bb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp326
    _tmp327 = einsum('kjfc,acik,bj,imneab->efmn', g_bbbb[o, o, v, v], t2_abab, t1_bb, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp327
    _tmp328 = einsum('kjfc,caik,bj,mineba->efmn', g_bbbb[o, o, v, v], t2_bbbb, t1_bb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp328
    _tmp329 = einsum('mncb,ijkeaf,caij,bk->efmn', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, t1_bb, optimize=True)
    l2_res += 0.5 * _tmp329
    _tmp330 = einsum('mnbc,ikjeaf,acij,bk->efmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp330
    _tmp331 = einsum('mncb,ijkefa,caij,bk->efmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, t1_bb, optimize=True)
    l2_res -= 0.5 * _tmp331
    _tmp332 = einsum('mnbc,kjieaf,acji,bk->efmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp332
    l2_res -= 0.5 * _tmp331
    _tmp333 = einsum('mnbc,kjiefa,caij,bk->efmn', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, t1_aa, optimize=True)
    l2_res -= 0.5 * _tmp333
    _tmp334 = einsum('knbc,acij,bk,imjeaf->efmn', g_abab[o, o, v, v], t2_abab, t1_aa, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp334
    _tmp335 = einsum('knbc,acij,bk,imjeaf->efmn', g_bbbb[o, o, v, v], t2_abab, t1_bb, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp335
    _tmp336 = einsum('knbc,acji,bk,mjieaf->efmn', g_abab[o, o, v, v], t2_abab, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp336
    _tmp337 = einsum('knbc,acji,bk,mjieaf->efmn', g_bbbb[o, o, v, v], t2_abab, t1_bb, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp337
    _tmp338 = einsum('knbc,caij,bk,mjiefa->efmn', g_abab[o, o, v, v], t2_bbbb, t1_aa, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp338
    _tmp339 = einsum('knbc,caij,bk,mjiefa->efmn', g_bbbb[o, o, v, v], t2_bbbb, t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp339
    _tmp340 = einsum('kmbc,caij,bk,ijneaf->efmn', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp340
    _tmp341 = einsum('mkcb,caij,bk,ijneaf->efmn', g_abab[o, o, v, v], t2_aaaa, t1_bb, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp341
    _tmp342 = einsum('kmbc,caij,bk,ijnefa->efmn', g_aaaa[o, o, v, v], t2_abab, t1_aa, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp342
    _tmp343 = einsum('mkcb,caij,bk,ijnefa->efmn', g_abab[o, o, v, v], t2_abab, t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp343
    l2_res += 0.5 * _tmp342
    l2_res += 0.5 * _tmp343
    _tmp344 = einsum('kncb,caik,bj,imjeaf->efmn', g_abab[o, o, v, v], t2_aaaa, t1_bb, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp344
    _tmp345 = einsum('kncb,caki,bj,mjiefa->efmn', g_abab[o, o, v, v], t2_abab, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp345
    _tmp346 = einsum('knbc,acik,bj,imjeaf->efmn', g_bbbb[o, o, v, v], t2_abab, t1_bb, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp346
    _tmp347 = einsum('knbc,acki,bj,mjieaf->efmn', g_abab[o, o, v, v], t2_abab, t1_aa, l3_aabaab, optimize=True)
    l2_res += 1 * _tmp347
    _tmp348 = einsum('knbc,caik,bj,mjiefa->efmn', g_bbbb[o, o, v, v], t2_bbbb, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp348
    _tmp349 = einsum('kmbc,caik,bj,ijneaf->efmn', g_aaaa[o, o, v, v], t2_aaaa, t1_aa, l3_aabaab, optimize=True)
    l2_res += 1 * _tmp349
    _tmp350 = einsum('mkcb,caik,bj,ijnefa->efmn', g_abab[o, o, v, v], t2_abab, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp350
    _tmp351 = einsum('kmbc,caki,bj,jinefa->efmn', g_aaaa[o, o, v, v], t2_abab, t1_aa, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp351
    _tmp352 = einsum('mkbc,acik,bj,ijneaf->efmn', g_abab[o, o, v, v], t2_abab, t1_aa, l3_aabaab, optimize=True)
    l2_res += 1 * _tmp352
    _tmp353 = einsum('mkbc,caik,bj,jinefa->efmn', g_abab[o, o, v, v], t2_bbbb, t1_aa, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp353
    _tmp354 = einsum('knbc,ak,bcij,imjeaf->efmn', g_abab[o, o, v, v], t1_aa, t2_abab, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp354
    _tmp355 = einsum('knbc,ak,bcji,mjieaf->efmn', g_abab[o, o, v, v], t1_aa, t2_abab, l3_aabaab, optimize=True)
    l2_res += 0.25 * _tmp355
    l2_res -= 0.25 * _tmp354
    l2_res += 0.25 * _tmp355
    _tmp356 = einsum('knbc,ak,bcij,mjiefa->efmn', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, l3_abbabb, optimize=True)
    l2_res += 0.25 * _tmp356
    _tmp357 = einsum('kmbc,ak,bcij,ijneaf->efmn', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, l3_aabaab, optimize=True)
    l2_res -= 0.25 * _tmp357
    _tmp358 = einsum('mkbc,ak,bcij,ijnefa->efmn', g_abab[o, o, v, v], t1_bb, t2_abab, l3_abbabb, optimize=True)
    l2_res -= 0.25 * _tmp358
    l2_res -= 0.25 * _tmp358
    l2_res -= 0.25 * _tmp358
    l2_res -= 0.25 * _tmp358
    _tmp359 = einsum('ijef,mnba,aj,bi->efmn', g_abab[o, o, v, v], l2_abab, t1_bb, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp359
    l2_res += 0.5 * _tmp359
    _tmp360 = einsum('jneb,miaf,aj,bi->efmn', g_abab[o, o, v, v], l2_abab, t1_aa, t1_bb, optimize=True)
    l2_res += 1 * _tmp360
    _tmp361 = einsum('jnbf,imea,aj,bi->efmn', g_abab[o, o, v, v], l2_aaaa, t1_aa, t1_aa, optimize=True)
    l2_res += 1 * _tmp361
    _tmp362 = einsum('jnfb,miea,aj,bi->efmn', g_bbbb[o, o, v, v], l2_abab, t1_bb, t1_bb, optimize=True)
    l2_res += 1 * _tmp362
    _tmp363 = einsum('jmeb,inaf,aj,bi->efmn', g_aaaa[o, o, v, v], l2_abab, t1_aa, t1_aa, optimize=True)
    l2_res += 1 * _tmp363
    _tmp364 = einsum('mjeb,infa,aj,bi->efmn', g_abab[o, o, v, v], l2_bbbb, t1_bb, t1_bb, optimize=True)
    l2_res += 1 * _tmp364
    _tmp365 = einsum('mjbf,inea,aj,bi->efmn', g_abab[o, o, v, v], l2_abab, t1_bb, t1_aa, optimize=True)
    l2_res += 1 * _tmp365
    _tmp366 = einsum('jieb,aj,bi,mnaf->efmn', g_aaaa[o, o, v, v], t1_aa, t1_aa, l2_abab, optimize=True)
    l2_res -= 1 * _tmp366
    _tmp367 = einsum('jieb,aj,bi,mnaf->efmn', g_abab[o, o, v, v], t1_aa, t1_bb, l2_abab, optimize=True)
    l2_res -= 1 * _tmp367
    _tmp368 = einsum('ijbf,aj,bi,mnea->efmn', g_abab[o, o, v, v], t1_bb, t1_aa, l2_abab, optimize=True)
    l2_res -= 1 * _tmp368
    _tmp369 = einsum('jifb,aj,bi,mnea->efmn', g_bbbb[o, o, v, v], t1_bb, t1_bb, l2_abab, optimize=True)
    l2_res -= 1 * _tmp369
    _tmp370 = einsum('mnba,ijef,aj,bi->efmn', g_abab[o, o, v, v], l2_abab, t1_bb, t1_aa, optimize=True)
    l2_res += 0.5 * _tmp370
    l2_res += 0.5 * _tmp370
    _tmp371 = einsum('jnab,aj,bi,mief->efmn', g_abab[o, o, v, v], t1_aa, t1_bb, l2_abab, optimize=True)
    l2_res -= 1 * _tmp371
    _tmp372 = einsum('jnab,aj,bi,mief->efmn', g_bbbb[o, o, v, v], t1_bb, t1_bb, l2_abab, optimize=True)
    l2_res -= 1 * _tmp372
    _tmp373 = einsum('jmab,aj,bi,inef->efmn', g_aaaa[o, o, v, v], t1_aa, t1_aa, l2_abab, optimize=True)
    l2_res -= 1 * _tmp373
    _tmp374 = einsum('mjba,aj,bi,inef->efmn', g_abab[o, o, v, v], t1_bb, t1_aa, l2_abab, optimize=True)
    l2_res -= 1 * _tmp374
    _tmp375 = einsum('kjei,ak,bj,minabf->efmn', g_aaaa[o, o, v, o], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp375
    _tmp376 = einsum('kjei,ak,bj,mniabf->efmn', g_abab[o, o, v, o], t1_aa, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp376
    _tmp377 = einsum('jkei,ak,bj,mnibfa->efmn', g_abab[o, o, v, o], t1_bb, t1_aa, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp377
    _tmp378 = einsum('kjif,ak,bj,mineab->efmn', g_abab[o, o, o, v], t1_aa, t1_bb, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp378
    l2_res += 0.5 * _tmp378
    _tmp379 = einsum('kjfi,ak,bj,mnieba->efmn', g_bbbb[o, o, v, o], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp379
    _tmp380 = einsum('knbj,ak,bi,imjeaf->efmn', g_abab[o, o, v, o], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp380
    _tmp381 = einsum('knjb,ak,bi,jmieaf->efmn', g_abab[o, o, o, v], t1_aa, t1_bb, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp381
    _tmp382 = einsum('knbj,ak,bi,mijefa->efmn', g_bbbb[o, o, v, o], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp382
    _tmp383 = einsum('kmbj,ak,bi,ijneaf->efmn', g_aaaa[o, o, v, o], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp383
    _tmp384 = einsum('mkbj,ak,bi,injefa->efmn', g_abab[o, o, v, o], t1_bb, t1_aa, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp384
    _tmp385 = einsum('mkjb,ak,bi,jniefa->efmn', g_abab[o, o, o, v], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp385
    _tmp386 = einsum('jbec,aj,ci,imnbaf->efmn', g_aaaa[o, v, v, v], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp386
    _tmp387 = einsum('jbec,aj,ci,minafb->efmn', g_abab[o, v, v, v], t1_aa, t1_bb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp387
    _tmp388 = einsum('bjec,aj,ci,minbfa->efmn', g_abab[v, o, v, v], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l2_res += 1 * _tmp388
    _tmp389 = einsum('jbcf,aj,ci,imnaeb->efmn', g_abab[o, v, v, v], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp389
    _tmp390 = einsum('bjcf,aj,ci,imnbea->efmn', g_abab[v, o, v, v], t1_bb, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 1 * _tmp390
    _tmp391 = einsum('jbfc,aj,ci,mineba->efmn', g_bbbb[o, v, v, v], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 1 * _tmp391
    _tmp392 = einsum('anbc,bj,ci,mjiaef->efmn', g_abab[v, o, v, v], t1_aa, t1_bb, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp392
    _tmp393 = einsum('ancb,bj,ci,imjaef->efmn', g_abab[v, o, v, v], t1_bb, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp393
    _tmp394 = einsum('nabc,bj,ci,mjieaf->efmn', g_bbbb[o, v, v, v], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp394
    _tmp395 = einsum('mabc,bj,ci,ijnaef->efmn', g_aaaa[o, v, v, v], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp395
    _tmp396 = einsum('mabc,bj,ci,jineaf->efmn', g_abab[o, v, v, v], t1_aa, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp396
    l2_res -= 0.5 * _tmp396
    _tmp397 = einsum('kjec,ak,bj,ci,imnabf->efmn', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp397
    _tmp398 = einsum('kjec,ak,bj,ci,minabf->efmn', g_abab[o, o, v, v], t1_aa, t1_bb, t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp398
    _tmp399 = einsum('jkec,ak,bj,ci,minbfa->efmn', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp399
    _tmp400 = einsum('kjcf,ak,bj,ci,imneab->efmn', g_abab[o, o, v, v], t1_aa, t1_bb, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp400
    l2_res -= 0.5 * _tmp400
    _tmp401 = einsum('kjfc,ak,bj,ci,mineba->efmn', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, l3_abbabb, optimize=True)
    l2_res += 0.5 * _tmp401
    _tmp402 = einsum('knbc,ak,bj,ci,mjieaf->efmn', g_abab[o, o, v, v], t1_aa, t1_aa, t1_bb, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp402
    _tmp403 = einsum('kncb,ak,bj,ci,imjeaf->efmn', g_abab[o, o, v, v], t1_aa, t1_bb, t1_aa, l3_aabaab, optimize=True)
    l2_res -= 0.5 * _tmp403
    _tmp404 = einsum('knbc,ak,bj,ci,mjiefa->efmn', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp404
    _tmp405 = einsum('kmbc,ak,bj,ci,ijneaf->efmn', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, l3_aabaab, optimize=True)
    l2_res += 0.5 * _tmp405
    _tmp406 = einsum('mkbc,ak,bj,ci,jinefa->efmn', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, l3_abbabb, optimize=True)
    l2_res -= 0.5 * _tmp406
    l2_res -= 0.5 * _tmp406
    return l2_res


def l3_aaaaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb):
    nv, no = t1_aa.shape
    l3_res = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('nd,lmef->deflmn', f_aa[o, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp0
    l3_res += 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp0.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp0.transpose(1, 0, 2, 3, 5, 4)
    _tmp1 = einsum('ld,mnef->deflmn', f_aa[o, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp1
    l3_res += 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    _tmp2 = einsum('nf,lmde->deflmn', f_aa[o, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp2
    l3_res += 1 * _tmp2.transpose(0, 1, 2, 3, 5, 4)
    _tmp3 = einsum('lf,mnde->deflmn', f_aa[o, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp3
    _tmp4 = einsum('ni,lmidef->deflmn', f_aa[o, o], l3_aaaaaa, optimize=True)
    l3_res += 1 * _tmp4
    l3_res -= 1 * _tmp4.transpose(0, 1, 2, 3, 5, 4)
    _tmp5 = einsum('li,mnidef->deflmn', f_aa[o, o], l3_aaaaaa, optimize=True)
    l3_res += 1 * _tmp5
    _tmp6 = einsum('ad,lmnaef->deflmn', f_aa[v, v], l3_aaaaaa, optimize=True)
    l3_res -= 1 * _tmp6
    l3_res += 1 * _tmp6.transpose(1, 0, 2, 3, 4, 5)
    _tmp7 = einsum('af,lmnade->deflmn', f_aa[v, v], l3_aaaaaa, optimize=True)
    l3_res -= 1 * _tmp7
    _tmp8 = einsum('id,lmnefa,ai->deflmn', f_aa[o, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp8
    l3_res -= 1 * _tmp8.transpose(1, 0, 2, 3, 4, 5)
    _tmp9 = einsum('if,lmndea,ai->deflmn', f_aa[o, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp9
    _tmp10 = einsum('na,ilmdef,ai->deflmn', f_aa[o, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp10
    l3_res -= 1 * _tmp10.transpose(0, 1, 2, 3, 5, 4)
    _tmp11 = einsum('la,imndef,ai->deflmn', f_aa[o, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp11
    _tmp12 = einsum('mnde,lf->deflmn', g_aaaa[o, o, v, v], l1_aa, optimize=True)
    l3_res -= 1 * _tmp12
    l3_res += 1 * _tmp12.transpose(0, 2, 1, 3, 4, 5)
    l3_res += 1 * _tmp12.transpose(0, 1, 2, 4, 3, 5)
    l3_res -= 1 * _tmp12.transpose(0, 2, 1, 4, 3, 5)
    _tmp13 = einsum('lmde,nf->deflmn', g_aaaa[o, o, v, v], l1_aa, optimize=True)
    l3_res -= 1 * _tmp13
    l3_res += 1 * _tmp13.transpose(0, 2, 1, 3, 4, 5)
    _tmp14 = einsum('mnef,ld->deflmn', g_aaaa[o, o, v, v], l1_aa, optimize=True)
    l3_res -= 1 * _tmp14
    l3_res += 1 * _tmp14.transpose(0, 1, 2, 4, 3, 5)
    _tmp15 = einsum('lmef,nd->deflmn', g_aaaa[o, o, v, v], l1_aa, optimize=True)
    l3_res -= 1 * _tmp15
    _tmp16 = einsum('mndi,lief->deflmn', g_aaaa[o, o, v, o], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp16
    l3_res += 1 * _tmp16.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp16.transpose(0, 1, 2, 4, 3, 5)
    l3_res -= 1 * _tmp16.transpose(1, 0, 2, 4, 3, 5)
    _tmp17 = einsum('lmdi,nief->deflmn', g_aaaa[o, o, v, o], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp17
    l3_res += 1 * _tmp17.transpose(1, 0, 2, 3, 4, 5)
    _tmp18 = einsum('mnfi,lide->deflmn', g_aaaa[o, o, v, o], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp18
    l3_res += 1 * _tmp18.transpose(0, 1, 2, 4, 3, 5)
    _tmp19 = einsum('lmfi,nide->deflmn', g_aaaa[o, o, v, o], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp19
    _tmp20 = einsum('nade,lmaf->deflmn', g_aaaa[o, v, v, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp20
    l3_res += 1 * _tmp20.transpose(0, 2, 1, 3, 4, 5)
    l3_res += 1 * _tmp20.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp20.transpose(0, 2, 1, 3, 5, 4)
    _tmp21 = einsum('lade,mnaf->deflmn', g_aaaa[o, v, v, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp21
    l3_res += 1 * _tmp21.transpose(0, 2, 1, 3, 4, 5)
    _tmp22 = einsum('naef,lmad->deflmn', g_aaaa[o, v, v, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp22
    l3_res += 1 * _tmp22.transpose(0, 1, 2, 3, 5, 4)
    _tmp23 = einsum('laef,mnad->deflmn', g_aaaa[o, v, v, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp23
    _tmp24 = einsum('mnij,lijdef->deflmn', g_aaaa[o, o, o, o], l3_aaaaaa, optimize=True)
    l3_res -= 0.5 * _tmp24
    l3_res += 0.5 * _tmp24.transpose(0, 1, 2, 4, 3, 5)
    _tmp25 = einsum('lmij,nijdef->deflmn', g_aaaa[o, o, o, o], l3_aaaaaa, optimize=True)
    l3_res -= 0.5 * _tmp25
    _tmp26 = einsum('nadi,lmiaef->deflmn', g_aaaa[o, v, v, o], l3_aaaaaa, optimize=True)
    l3_res -= 1 * _tmp26
    l3_res += 1 * _tmp26.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp26.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp26.transpose(1, 0, 2, 3, 5, 4)
    _tmp27 = einsum('nadi,lmifea->deflmn', g_abab[o, v, v, o], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp27
    l3_res -= 1 * _tmp27.transpose(1, 0, 2, 3, 4, 5)
    l3_res -= 1 * _tmp27.transpose(0, 1, 2, 3, 5, 4)
    l3_res += 1 * _tmp27.transpose(1, 0, 2, 3, 5, 4)
    _tmp28 = einsum('ladi,mniaef->deflmn', g_aaaa[o, v, v, o], l3_aaaaaa, optimize=True)
    l3_res -= 1 * _tmp28
    l3_res += 1 * _tmp28.transpose(1, 0, 2, 3, 4, 5)
    _tmp29 = einsum('ladi,mnifea->deflmn', g_abab[o, v, v, o], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp29
    l3_res -= 1 * _tmp29.transpose(1, 0, 2, 3, 4, 5)
    _tmp30 = einsum('nafi,lmiade->deflmn', g_aaaa[o, v, v, o], l3_aaaaaa, optimize=True)
    l3_res -= 1 * _tmp30
    l3_res += 1 * _tmp30.transpose(0, 1, 2, 3, 5, 4)
    _tmp31 = einsum('nafi,lmieda->deflmn', g_abab[o, v, v, o], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp31
    l3_res -= 1 * _tmp31.transpose(0, 1, 2, 3, 5, 4)
    _tmp32 = einsum('lafi,mniade->deflmn', g_aaaa[o, v, v, o], l3_aaaaaa, optimize=True)
    l3_res -= 1 * _tmp32
    _tmp33 = einsum('lafi,mnieda->deflmn', g_abab[o, v, v, o], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp33
    _tmp34 = einsum('bade,lmnbaf->deflmn', g_aaaa[v, v, v, v], l3_aaaaaa, optimize=True)
    l3_res -= 0.5 * _tmp34
    l3_res += 0.5 * _tmp34.transpose(0, 2, 1, 3, 4, 5)
    _tmp35 = einsum('baef,lmnbad->deflmn', g_aaaa[v, v, v, v], l3_aaaaaa, optimize=True)
    l3_res -= 0.5 * _tmp35
    _tmp36 = einsum('inde,lmfa,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp36
    l3_res -= 1 * _tmp36.transpose(0, 2, 1, 3, 4, 5)
    l3_res -= 1 * _tmp36.transpose(0, 1, 2, 3, 5, 4)
    l3_res += 1 * _tmp36.transpose(0, 2, 1, 3, 5, 4)
    _tmp37 = einsum('ilde,mnfa,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp37
    l3_res -= 1 * _tmp37.transpose(0, 2, 1, 3, 4, 5)
    _tmp38 = einsum('mnda,ilef,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp38
    l3_res -= 1 * _tmp38.transpose(1, 0, 2, 3, 4, 5)
    l3_res -= 1 * _tmp38.transpose(0, 1, 2, 4, 3, 5)
    l3_res += 1 * _tmp38.transpose(1, 0, 2, 4, 3, 5)
    _tmp39 = einsum('inda,lmef,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp39
    l3_res -= 1 * _tmp39.transpose(1, 0, 2, 3, 4, 5)
    l3_res -= 1 * _tmp39.transpose(0, 1, 2, 3, 5, 4)
    l3_res += 1 * _tmp39.transpose(1, 0, 2, 3, 5, 4)
    _tmp40 = einsum('nida,lmef,ai->deflmn', g_abab[o, o, v, v], l2_aaaa, t1_bb, optimize=True)
    l3_res -= 1 * _tmp40
    l3_res += 1 * _tmp40.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp40.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp40.transpose(1, 0, 2, 3, 5, 4)
    _tmp41 = einsum('lmda,inef,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp41
    l3_res -= 1 * _tmp41.transpose(1, 0, 2, 3, 4, 5)
    _tmp42 = einsum('ilda,mnef,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp42
    l3_res -= 1 * _tmp42.transpose(1, 0, 2, 3, 4, 5)
    _tmp43 = einsum('lida,mnef,ai->deflmn', g_abab[o, o, v, v], l2_aaaa, t1_bb, optimize=True)
    l3_res -= 1 * _tmp43
    l3_res += 1 * _tmp43.transpose(1, 0, 2, 3, 4, 5)
    _tmp44 = einsum('inef,lmda,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp44
    l3_res -= 1 * _tmp44.transpose(0, 1, 2, 3, 5, 4)
    _tmp45 = einsum('ilef,mnda,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp45
    _tmp46 = einsum('mnfa,ilde,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp46
    l3_res -= 1 * _tmp46.transpose(0, 1, 2, 4, 3, 5)
    _tmp47 = einsum('infa,lmde,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp47
    l3_res -= 1 * _tmp47.transpose(0, 1, 2, 3, 5, 4)
    _tmp48 = einsum('nifa,lmde,ai->deflmn', g_abab[o, o, v, v], l2_aaaa, t1_bb, optimize=True)
    l3_res -= 1 * _tmp48
    l3_res += 1 * _tmp48.transpose(0, 1, 2, 3, 5, 4)
    _tmp49 = einsum('lmfa,inde,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp49
    _tmp50 = einsum('ilfa,mnde,ai->deflmn', g_aaaa[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp50
    _tmp51 = einsum('lifa,mnde,ai->deflmn', g_abab[o, o, v, v], l2_aaaa, t1_bb, optimize=True)
    l3_res -= 1 * _tmp51
    _tmp52 = einsum('jndi,lmiefa,aj->deflmn', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp52
    l3_res += 1 * _tmp52.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp52.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp52.transpose(1, 0, 2, 3, 5, 4)
    _tmp53 = einsum('njdi,lmiefa,aj->deflmn', g_abab[o, o, v, o], l3_aabaab, t1_bb, optimize=True)
    l3_res += 1 * _tmp53
    l3_res -= 1 * _tmp53.transpose(1, 0, 2, 3, 4, 5)
    l3_res -= 1 * _tmp53.transpose(0, 1, 2, 3, 5, 4)
    l3_res += 1 * _tmp53.transpose(1, 0, 2, 3, 5, 4)
    _tmp54 = einsum('jldi,mniefa,aj->deflmn', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp54
    l3_res += 1 * _tmp54.transpose(1, 0, 2, 3, 4, 5)
    _tmp55 = einsum('ljdi,mniefa,aj->deflmn', g_abab[o, o, v, o], l3_aabaab, t1_bb, optimize=True)
    l3_res += 1 * _tmp55
    l3_res -= 1 * _tmp55.transpose(1, 0, 2, 3, 4, 5)
    _tmp56 = einsum('jnfi,lmidea,aj->deflmn', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp56
    l3_res += 1 * _tmp56.transpose(0, 1, 2, 3, 5, 4)
    _tmp57 = einsum('njfi,lmidea,aj->deflmn', g_abab[o, o, v, o], l3_aabaab, t1_bb, optimize=True)
    l3_res += 1 * _tmp57
    l3_res -= 1 * _tmp57.transpose(0, 1, 2, 3, 5, 4)
    _tmp58 = einsum('jlfi,mnidea,aj->deflmn', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp58
    _tmp59 = einsum('ljfi,mnidea,aj->deflmn', g_abab[o, o, v, o], l3_aabaab, t1_bb, optimize=True)
    l3_res += 1 * _tmp59
    _tmp60 = einsum('mnaj,iljdef,ai->deflmn', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp60
    l3_res -= 1 * _tmp60.transpose(0, 1, 2, 4, 3, 5)
    _tmp61 = einsum('jnai,lmidef,aj->deflmn', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp61
    l3_res -= 1 * _tmp61.transpose(0, 1, 2, 3, 5, 4)
    _tmp62 = einsum('njia,lmidef,aj->deflmn', g_abab[o, o, o, v], l3_aaaaaa, t1_bb, optimize=True)
    l3_res += 1 * _tmp62
    l3_res -= 1 * _tmp62.transpose(0, 1, 2, 3, 5, 4)
    _tmp63 = einsum('lmaj,injdef,ai->deflmn', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp63
    _tmp64 = einsum('jlai,mnidef,aj->deflmn', g_aaaa[o, o, v, o], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp64
    _tmp65 = einsum('ljia,mnidef,aj->deflmn', g_abab[o, o, o, v], l3_aaaaaa, t1_bb, optimize=True)
    l3_res += 1 * _tmp65
    _tmp66 = einsum('ibde,lmnbfa,ai->deflmn', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp66
    l3_res -= 1 * _tmp66.transpose(0, 2, 1, 3, 4, 5)
    _tmp67 = einsum('nadb,ilmaef,bi->deflmn', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp67
    l3_res += 1 * _tmp67.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp67.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp67.transpose(1, 0, 2, 3, 5, 4)
    _tmp68 = einsum('nadb,mlifea,bi->deflmn', g_abab[o, v, v, v], l3_aabaab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp68
    l3_res += 1 * _tmp68.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp68.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp68.transpose(1, 0, 2, 3, 5, 4)
    _tmp69 = einsum('ladb,imnaef,bi->deflmn', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp69
    l3_res += 1 * _tmp69.transpose(1, 0, 2, 3, 4, 5)
    _tmp70 = einsum('ladb,nmifea,bi->deflmn', g_abab[o, v, v, v], l3_aabaab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp70
    l3_res += 1 * _tmp70.transpose(1, 0, 2, 3, 4, 5)
    _tmp71 = einsum('iadb,lmnaef,bi->deflmn', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp71
    l3_res -= 1 * _tmp71.transpose(1, 0, 2, 3, 4, 5)
    _tmp72 = einsum('aidb,lmnaef,bi->deflmn', g_abab[v, o, v, v], l3_aaaaaa, t1_bb, optimize=True)
    l3_res -= 1 * _tmp72
    l3_res += 1 * _tmp72.transpose(1, 0, 2, 3, 4, 5)
    _tmp73 = einsum('ibef,lmnbda,ai->deflmn', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp73
    _tmp74 = einsum('nafb,ilmade,bi->deflmn', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp74
    l3_res += 1 * _tmp74.transpose(0, 1, 2, 3, 5, 4)
    _tmp75 = einsum('nafb,mlieda,bi->deflmn', g_abab[o, v, v, v], l3_aabaab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp75
    l3_res += 1 * _tmp75.transpose(0, 1, 2, 3, 5, 4)
    _tmp76 = einsum('lafb,imnade,bi->deflmn', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp76
    _tmp77 = einsum('lafb,nmieda,bi->deflmn', g_abab[o, v, v, v], l3_aabaab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp77
    _tmp78 = einsum('iafb,lmnade,bi->deflmn', g_aaaa[o, v, v, v], l3_aaaaaa, t1_aa, optimize=True)
    l3_res += 1 * _tmp78
    _tmp79 = einsum('aifb,lmnade,bi->deflmn', g_abab[v, o, v, v], l3_aaaaaa, t1_bb, optimize=True)
    l3_res -= 1 * _tmp79
    _tmp80 = einsum('jnde,ilmfba,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp80
    l3_res += 0.5 * _tmp80.transpose(0, 2, 1, 3, 4, 5)
    l3_res += 0.5 * _tmp80.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 0.5 * _tmp80.transpose(0, 2, 1, 3, 5, 4)
    _tmp81 = einsum('jnde,mlifba,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp81
    l3_res += 0.5 * _tmp81.transpose(0, 2, 1, 3, 4, 5)
    l3_res += 0.5 * _tmp81.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 0.5 * _tmp81.transpose(0, 2, 1, 3, 5, 4)
    l3_res -= 0.5 * _tmp81
    l3_res += 0.5 * _tmp81.transpose(0, 2, 1, 3, 4, 5)
    l3_res += 0.5 * _tmp81.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 0.5 * _tmp81.transpose(0, 2, 1, 3, 5, 4)
    _tmp82 = einsum('jlde,imnfba,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp82
    l3_res += 0.5 * _tmp82.transpose(0, 2, 1, 3, 4, 5)
    _tmp83 = einsum('jlde,nmifba,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp83
    l3_res += 0.5 * _tmp83.transpose(0, 2, 1, 3, 4, 5)
    l3_res -= 0.5 * _tmp83
    l3_res += 0.5 * _tmp83.transpose(0, 2, 1, 3, 4, 5)
    _tmp84 = einsum('jide,lmnfba,baji->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.25 * _tmp84
    l3_res += 0.25 * _tmp84.transpose(0, 2, 1, 3, 4, 5)
    _tmp85 = einsum('mndb,ijlefa,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp85
    l3_res += 0.5 * _tmp85.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 0.5 * _tmp85.transpose(0, 1, 2, 4, 3, 5)
    l3_res -= 0.5 * _tmp85.transpose(1, 0, 2, 4, 3, 5)
    _tmp86 = einsum('mndb,iljefa,baij->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp86
    l3_res -= 0.5 * _tmp86.transpose(1, 0, 2, 3, 4, 5)
    l3_res -= 0.5 * _tmp86.transpose(0, 1, 2, 4, 3, 5)
    l3_res += 0.5 * _tmp86.transpose(1, 0, 2, 4, 3, 5)
    _tmp87 = einsum('mndb,ljiefa,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp87
    l3_res += 0.5 * _tmp87.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 0.5 * _tmp87.transpose(0, 1, 2, 4, 3, 5)
    l3_res -= 0.5 * _tmp87.transpose(1, 0, 2, 4, 3, 5)
    _tmp88 = einsum('jndb,ilmefa,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 1 * _tmp88
    l3_res += 1 * _tmp88.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp88.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp88.transpose(1, 0, 2, 3, 5, 4)
    _tmp89 = einsum('njdb,ilmefa,abij->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, optimize=True)
    l3_res -= 1 * _tmp89
    l3_res += 1 * _tmp89.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp89.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp89.transpose(1, 0, 2, 3, 5, 4)
    _tmp90 = einsum('jndb,mliefa,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 1 * _tmp90
    l3_res += 1 * _tmp90.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp90.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp90.transpose(1, 0, 2, 3, 5, 4)
    _tmp91 = einsum('njdb,mliefa,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_bbbb, optimize=True)
    l3_res -= 1 * _tmp91
    l3_res += 1 * _tmp91.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp91.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp91.transpose(1, 0, 2, 3, 5, 4)
    _tmp92 = einsum('lmdb,ijnefa,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp92
    l3_res += 0.5 * _tmp92.transpose(1, 0, 2, 3, 4, 5)
    _tmp93 = einsum('lmdb,injefa,baij->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp93
    l3_res -= 0.5 * _tmp93.transpose(1, 0, 2, 3, 4, 5)
    _tmp94 = einsum('lmdb,njiefa,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp94
    l3_res += 0.5 * _tmp94.transpose(1, 0, 2, 3, 4, 5)
    _tmp95 = einsum('jldb,imnefa,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 1 * _tmp95
    l3_res += 1 * _tmp95.transpose(1, 0, 2, 3, 4, 5)
    _tmp96 = einsum('ljdb,imnefa,abij->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, optimize=True)
    l3_res -= 1 * _tmp96
    l3_res += 1 * _tmp96.transpose(1, 0, 2, 3, 4, 5)
    _tmp97 = einsum('jldb,nmiefa,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 1 * _tmp97
    l3_res += 1 * _tmp97.transpose(1, 0, 2, 3, 4, 5)
    _tmp98 = einsum('ljdb,nmiefa,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_bbbb, optimize=True)
    l3_res -= 1 * _tmp98
    l3_res += 1 * _tmp98.transpose(1, 0, 2, 3, 4, 5)
    _tmp99 = einsum('jidb,lmnefa,baji->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp99
    l3_res += 0.5 * _tmp99.transpose(1, 0, 2, 3, 4, 5)
    _tmp100 = einsum('jidb,lmnefa,abji->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp100
    l3_res -= 0.5 * _tmp100.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 0.5 * _tmp100
    l3_res -= 0.5 * _tmp100.transpose(1, 0, 2, 3, 4, 5)
    _tmp101 = einsum('jnef,ilmdba,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp101
    l3_res += 0.5 * _tmp101.transpose(0, 1, 2, 3, 5, 4)
    _tmp102 = einsum('jnef,mlidba,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp102
    l3_res += 0.5 * _tmp102.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 0.5 * _tmp102
    l3_res += 0.5 * _tmp102.transpose(0, 1, 2, 3, 5, 4)
    _tmp103 = einsum('jlef,imndba,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp103
    _tmp104 = einsum('jlef,nmidba,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp104
    l3_res -= 0.5 * _tmp104
    _tmp105 = einsum('jief,lmndba,baji->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.25 * _tmp105
    _tmp106 = einsum('mnfb,ijldea,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp106
    l3_res += 0.5 * _tmp106.transpose(0, 1, 2, 4, 3, 5)
    _tmp107 = einsum('mnfb,iljdea,baij->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp107
    l3_res -= 0.5 * _tmp107.transpose(0, 1, 2, 4, 3, 5)
    _tmp108 = einsum('mnfb,ljidea,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp108
    l3_res += 0.5 * _tmp108.transpose(0, 1, 2, 4, 3, 5)
    _tmp109 = einsum('jnfb,ilmdea,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 1 * _tmp109
    l3_res += 1 * _tmp109.transpose(0, 1, 2, 3, 5, 4)
    _tmp110 = einsum('njfb,ilmdea,abij->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, optimize=True)
    l3_res -= 1 * _tmp110
    l3_res += 1 * _tmp110.transpose(0, 1, 2, 3, 5, 4)
    _tmp111 = einsum('jnfb,mlidea,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 1 * _tmp111
    l3_res += 1 * _tmp111.transpose(0, 1, 2, 3, 5, 4)
    _tmp112 = einsum('njfb,mlidea,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_bbbb, optimize=True)
    l3_res -= 1 * _tmp112
    l3_res += 1 * _tmp112.transpose(0, 1, 2, 3, 5, 4)
    _tmp113 = einsum('lmfb,ijndea,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp113
    _tmp114 = einsum('lmfb,injdea,baij->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp114
    _tmp115 = einsum('lmfb,njidea,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp115
    _tmp116 = einsum('jlfb,imndea,baij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 1 * _tmp116
    _tmp117 = einsum('ljfb,imndea,abij->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, optimize=True)
    l3_res -= 1 * _tmp117
    _tmp118 = einsum('jlfb,nmidea,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 1 * _tmp118
    _tmp119 = einsum('ljfb,nmidea,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_bbbb, optimize=True)
    l3_res -= 1 * _tmp119
    _tmp120 = einsum('jifb,lmndea,baji->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp120
    _tmp121 = einsum('jifb,lmndea,abji->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp121
    l3_res += 0.5 * _tmp121
    _tmp122 = einsum('mnab,ijldef,abij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.25 * _tmp122
    l3_res += 0.25 * _tmp122.transpose(0, 1, 2, 4, 3, 5)
    _tmp123 = einsum('jnab,ilmdef,abij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp123
    l3_res += 0.5 * _tmp123.transpose(0, 1, 2, 3, 5, 4)
    _tmp124 = einsum('njab,ilmdef,abij->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp124
    l3_res -= 0.5 * _tmp124.transpose(0, 1, 2, 3, 5, 4)
    l3_res += 0.5 * _tmp124
    l3_res -= 0.5 * _tmp124.transpose(0, 1, 2, 3, 5, 4)
    _tmp125 = einsum('lmab,ijndef,abij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.25 * _tmp125
    _tmp126 = einsum('jlab,imndef,abij->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp126
    _tmp127 = einsum('ljab,imndef,abij->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp127
    l3_res += 0.5 * _tmp127
    _tmp128 = einsum('jide,lmnfba,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res += 0.5 * _tmp128
    l3_res -= 0.5 * _tmp128.transpose(0, 2, 1, 3, 4, 5)
    _tmp129 = einsum('jndb,ilmefa,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp129
    l3_res += 1 * _tmp129.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp129.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp129.transpose(1, 0, 2, 3, 5, 4)
    _tmp130 = einsum('njdb,mliefa,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp130
    l3_res += 1 * _tmp130.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp130.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp130.transpose(1, 0, 2, 3, 5, 4)
    _tmp131 = einsum('jldb,imnefa,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp131
    l3_res += 1 * _tmp131.transpose(1, 0, 2, 3, 4, 5)
    _tmp132 = einsum('ljdb,nmiefa,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp132
    l3_res += 1 * _tmp132.transpose(1, 0, 2, 3, 4, 5)
    _tmp133 = einsum('jidb,lmnefa,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res += 1 * _tmp133
    l3_res -= 1 * _tmp133.transpose(1, 0, 2, 3, 4, 5)
    _tmp134 = einsum('jidb,lmnefa,aj,bi->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t1_aa, t1_bb, optimize=True)
    l3_res += 1 * _tmp134
    l3_res -= 1 * _tmp134.transpose(1, 0, 2, 3, 4, 5)
    _tmp135 = einsum('jief,lmndba,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res += 0.5 * _tmp135
    _tmp136 = einsum('jnfb,ilmdea,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp136
    l3_res += 1 * _tmp136.transpose(0, 1, 2, 3, 5, 4)
    _tmp137 = einsum('njfb,mlidea,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp137
    l3_res += 1 * _tmp137.transpose(0, 1, 2, 3, 5, 4)
    _tmp138 = einsum('jlfb,imndea,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp138
    _tmp139 = einsum('ljfb,nmidea,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp139
    _tmp140 = einsum('jifb,lmndea,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res += 1 * _tmp140
    _tmp141 = einsum('jifb,lmndea,aj,bi->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t1_aa, t1_bb, optimize=True)
    l3_res += 1 * _tmp141
    _tmp142 = einsum('mnab,ijldef,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res += 0.5 * _tmp142
    l3_res -= 0.5 * _tmp142.transpose(0, 1, 2, 4, 3, 5)
    _tmp143 = einsum('jnab,ilmdef,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res += 1 * _tmp143
    l3_res -= 1 * _tmp143.transpose(0, 1, 2, 3, 5, 4)
    _tmp144 = einsum('njba,ilmdef,aj,bi->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t1_bb, t1_aa, optimize=True)
    l3_res += 1 * _tmp144
    l3_res -= 1 * _tmp144.transpose(0, 1, 2, 3, 5, 4)
    _tmp145 = einsum('lmab,ijndef,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res += 0.5 * _tmp145
    _tmp146 = einsum('jlab,imndef,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aaaaaa, t1_aa, t1_aa, optimize=True)
    l3_res += 1 * _tmp146
    _tmp147 = einsum('ljba,imndef,aj,bi->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t1_bb, t1_aa, optimize=True)
    l3_res += 1 * _tmp147
    return l3_res


def l3_aabaab_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb):
    nv, no = t1_aa.shape
    l3_res = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('md,lnef->deflmn', f_aa[o, v], l2_abab, optimize=True)
    l3_res += 1 * _tmp0
    l3_res -= 1 * _tmp0.transpose(1, 0, 2, 3, 4, 5)
    _tmp1 = einsum('ld,mnef->deflmn', f_aa[o, v], l2_abab, optimize=True)
    l3_res -= 1 * _tmp1
    l3_res += 1 * _tmp1.transpose(1, 0, 2, 3, 4, 5)
    _tmp2 = einsum('nf,lmde->deflmn', f_bb[o, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp2
    _tmp3 = einsum('ni,lmidef->deflmn', f_bb[o, o], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp3
    _tmp4 = einsum('mi,lindef->deflmn', f_aa[o, o], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp4
    _tmp5 = einsum('li,mindef->deflmn', f_aa[o, o], l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp5
    _tmp6 = einsum('ad,lmnaef->deflmn', f_aa[v, v], l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp6
    l3_res += 1 * _tmp6.transpose(1, 0, 2, 3, 4, 5)
    _tmp7 = einsum('af,lmneda->deflmn', f_bb[v, v], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp7
    _tmp8 = einsum('id,lmneaf,ai->deflmn', f_aa[o, v], l3_aabaab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp8
    l3_res += 1 * _tmp8.transpose(1, 0, 2, 3, 4, 5)
    _tmp9 = einsum('if,lmndea,ai->deflmn', f_bb[o, v], l3_aabaab, t1_bb, optimize=True)
    l3_res += 1 * _tmp9
    _tmp10 = einsum('na,ai,mlidef->deflmn', f_bb[o, v], t1_bb, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp10
    _tmp11 = einsum('ma,ai,ilndef->deflmn', f_aa[o, v], t1_aa, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp11
    _tmp12 = einsum('la,imndef,ai->deflmn', f_aa[o, v], l3_aabaab, t1_aa, optimize=True)
    l3_res += 1 * _tmp12
    _tmp13 = einsum('mndf,le->deflmn', g_abab[o, o, v, v], l1_aa, optimize=True)
    l3_res += 1 * _tmp13
    l3_res -= 1 * _tmp13.transpose(0, 1, 2, 4, 3, 5)
    _tmp14 = einsum('lmde,nf->deflmn', g_aaaa[o, o, v, v], l1_bb, optimize=True)
    l3_res -= 1 * _tmp14
    _tmp15 = einsum('mnef,ld->deflmn', g_abab[o, o, v, v], l1_aa, optimize=True)
    l3_res -= 1 * _tmp15
    l3_res += 1 * _tmp15.transpose(0, 1, 2, 4, 3, 5)
    _tmp16 = einsum('mndi,lief->deflmn', g_abab[o, o, v, o], l2_abab, optimize=True)
    l3_res -= 1 * _tmp16
    l3_res += 1 * _tmp16.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp16.transpose(0, 1, 2, 4, 3, 5)
    l3_res -= 1 * _tmp16.transpose(1, 0, 2, 4, 3, 5)
    _tmp17 = einsum('lmdi,inef->deflmn', g_aaaa[o, o, v, o], l2_abab, optimize=True)
    l3_res += 1 * _tmp17
    l3_res -= 1 * _tmp17.transpose(1, 0, 2, 3, 4, 5)
    _tmp18 = einsum('mnif,lide->deflmn', g_abab[o, o, o, v], l2_aaaa, optimize=True)
    l3_res += 1 * _tmp18
    l3_res -= 1 * _tmp18.transpose(0, 1, 2, 4, 3, 5)
    _tmp19 = einsum('andf,lmae->deflmn', g_abab[v, o, v, v], l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp19
    _tmp20 = einsum('made,lnaf->deflmn', g_aaaa[o, v, v, v], l2_abab, optimize=True)
    l3_res += 1 * _tmp20
    _tmp21 = einsum('madf,lnea->deflmn', g_abab[o, v, v, v], l2_abab, optimize=True)
    l3_res += 1 * _tmp21
    _tmp22 = einsum('lade,mnaf->deflmn', g_aaaa[o, v, v, v], l2_abab, optimize=True)
    l3_res -= 1 * _tmp22
    _tmp23 = einsum('ladf,mnea->deflmn', g_abab[o, v, v, v], l2_abab, optimize=True)
    l3_res -= 1 * _tmp23
    _tmp24 = einsum('anef,lmad->deflmn', g_abab[v, o, v, v], l2_aaaa, optimize=True)
    l3_res += 1 * _tmp24
    _tmp25 = einsum('maef,lnda->deflmn', g_abab[o, v, v, v], l2_abab, optimize=True)
    l3_res -= 1 * _tmp25
    _tmp26 = einsum('laef,mnda->deflmn', g_abab[o, v, v, v], l2_abab, optimize=True)
    l3_res += 1 * _tmp26
    _tmp27 = einsum('mnij,lijdef->deflmn', g_abab[o, o, o, o], l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp27
    l3_res += 0.5 * _tmp27.transpose(0, 1, 2, 4, 3, 5)
    l3_res -= 0.5 * _tmp27
    l3_res += 0.5 * _tmp27.transpose(0, 1, 2, 4, 3, 5)
    _tmp28 = einsum('lmij,jindef->deflmn', g_aaaa[o, o, o, o], l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp28
    _tmp29 = einsum('andi,lmiaef->deflmn', g_abab[v, o, v, o], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp29
    l3_res -= 1 * _tmp29.transpose(1, 0, 2, 3, 4, 5)
    _tmp30 = einsum('madi,linaef->deflmn', g_aaaa[o, v, v, o], l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp30
    l3_res += 1 * _tmp30.transpose(1, 0, 2, 3, 4, 5)
    _tmp31 = einsum('madi,lnieaf->deflmn', g_abab[o, v, v, o], l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp31
    l3_res += 1 * _tmp31.transpose(1, 0, 2, 3, 4, 5)
    _tmp32 = einsum('ladi,minaef->deflmn', g_aaaa[o, v, v, o], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp32
    l3_res -= 1 * _tmp32.transpose(1, 0, 2, 3, 4, 5)
    _tmp33 = einsum('ladi,mnieaf->deflmn', g_abab[o, v, v, o], l3_abbabb, optimize=True)
    l3_res += 1 * _tmp33
    l3_res -= 1 * _tmp33.transpose(1, 0, 2, 3, 4, 5)
    _tmp34 = einsum('anif,lmiade->deflmn', g_abab[v, o, o, v], l3_aaaaaa, optimize=True)
    l3_res -= 1 * _tmp34
    _tmp35 = einsum('nafi,lmieda->deflmn', g_bbbb[o, v, v, o], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp35
    _tmp36 = einsum('maif,lineda->deflmn', g_abab[o, v, o, v], l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp36
    _tmp37 = einsum('laif,mineda->deflmn', g_abab[o, v, o, v], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp37
    _tmp38 = einsum('bade,lmnbaf->deflmn', g_aaaa[v, v, v, v], l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp38
    _tmp39 = einsum('badf,lmnbea->deflmn', g_abab[v, v, v, v], l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp39
    _tmp40 = einsum('abdf,lmneab->deflmn', g_abab[v, v, v, v], l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp40
    _tmp41 = einsum('baef,lmnbda->deflmn', g_abab[v, v, v, v], l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp41
    _tmp42 = einsum('abef,lmndab->deflmn', g_abab[v, v, v, v], l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp42
    _tmp43 = einsum('indf,lmea,ai->deflmn', g_abab[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp43
    _tmp44 = einsum('imde,lnaf,ai->deflmn', g_aaaa[o, o, v, v], l2_abab, t1_aa, optimize=True)
    l3_res += 1 * _tmp44
    _tmp45 = einsum('midf,lnea,ai->deflmn', g_abab[o, o, v, v], l2_abab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp45
    _tmp46 = einsum('ilde,ai,mnaf->deflmn', g_aaaa[o, o, v, v], t1_aa, l2_abab, optimize=True)
    l3_res -= 1 * _tmp46
    _tmp47 = einsum('lidf,ai,mnea->deflmn', g_abab[o, o, v, v], t1_bb, l2_abab, optimize=True)
    l3_res += 1 * _tmp47
    _tmp48 = einsum('mnda,lief,ai->deflmn', g_abab[o, o, v, v], l2_abab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp48
    l3_res += 1 * _tmp48.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 1 * _tmp48.transpose(0, 1, 2, 4, 3, 5)
    l3_res -= 1 * _tmp48.transpose(1, 0, 2, 4, 3, 5)
    _tmp49 = einsum('imda,ai,lnef->deflmn', g_aaaa[o, o, v, v], t1_aa, l2_abab, optimize=True)
    l3_res -= 1 * _tmp49
    l3_res += 1 * _tmp49.transpose(1, 0, 2, 3, 4, 5)
    _tmp50 = einsum('mida,ai,lnef->deflmn', g_abab[o, o, v, v], t1_bb, l2_abab, optimize=True)
    l3_res += 1 * _tmp50
    l3_res -= 1 * _tmp50.transpose(1, 0, 2, 3, 4, 5)
    _tmp51 = einsum('lmda,inef,ai->deflmn', g_aaaa[o, o, v, v], l2_abab, t1_aa, optimize=True)
    l3_res += 1 * _tmp51
    l3_res -= 1 * _tmp51.transpose(1, 0, 2, 3, 4, 5)
    _tmp52 = einsum('ilda,mnef,ai->deflmn', g_aaaa[o, o, v, v], l2_abab, t1_aa, optimize=True)
    l3_res += 1 * _tmp52
    l3_res -= 1 * _tmp52.transpose(1, 0, 2, 3, 4, 5)
    _tmp53 = einsum('lida,mnef,ai->deflmn', g_abab[o, o, v, v], l2_abab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp53
    l3_res += 1 * _tmp53.transpose(1, 0, 2, 3, 4, 5)
    _tmp54 = einsum('inef,ai,lmda->deflmn', g_abab[o, o, v, v], t1_aa, l2_aaaa, optimize=True)
    l3_res += 1 * _tmp54
    _tmp55 = einsum('mief,ai,lnda->deflmn', g_abab[o, o, v, v], t1_bb, l2_abab, optimize=True)
    l3_res += 1 * _tmp55
    _tmp56 = einsum('lief,mnda,ai->deflmn', g_abab[o, o, v, v], l2_abab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp56
    _tmp57 = einsum('mnaf,ilde,ai->deflmn', g_abab[o, o, v, v], l2_aaaa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp57
    l3_res += 1 * _tmp57.transpose(0, 1, 2, 4, 3, 5)
    _tmp58 = einsum('inaf,ai,lmde->deflmn', g_abab[o, o, v, v], t1_aa, l2_aaaa, optimize=True)
    l3_res -= 1 * _tmp58
    _tmp59 = einsum('infa,ai,lmde->deflmn', g_bbbb[o, o, v, v], t1_bb, l2_aaaa, optimize=True)
    l3_res += 1 * _tmp59
    _tmp60 = einsum('jndi,aj,lmieaf->deflmn', g_abab[o, o, v, o], t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp60
    l3_res -= 1 * _tmp60.transpose(1, 0, 2, 3, 4, 5)
    _tmp61 = einsum('jmdi,aj,lineaf->deflmn', g_aaaa[o, o, v, o], t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp61
    l3_res -= 1 * _tmp61.transpose(1, 0, 2, 3, 4, 5)
    _tmp62 = einsum('mjdi,aj,lniefa->deflmn', g_abab[o, o, v, o], t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp62
    l3_res += 1 * _tmp62.transpose(1, 0, 2, 3, 4, 5)
    _tmp63 = einsum('jldi,mineaf,aj->deflmn', g_aaaa[o, o, v, o], l3_aabaab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp63
    l3_res += 1 * _tmp63.transpose(1, 0, 2, 3, 4, 5)
    _tmp64 = einsum('ljdi,mniefa,aj->deflmn', g_abab[o, o, v, o], l3_abbabb, t1_bb, optimize=True)
    l3_res += 1 * _tmp64
    l3_res -= 1 * _tmp64.transpose(1, 0, 2, 3, 4, 5)
    _tmp65 = einsum('jnif,aj,lmidea->deflmn', g_abab[o, o, o, v], t1_aa, l3_aaaaaa, optimize=True)
    l3_res += 1 * _tmp65
    _tmp66 = einsum('jnfi,aj,lmidea->deflmn', g_bbbb[o, o, v, o], t1_bb, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp66
    _tmp67 = einsum('mjif,aj,lindea->deflmn', g_abab[o, o, o, v], t1_bb, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp67
    _tmp68 = einsum('ljif,mindea,aj->deflmn', g_abab[o, o, o, v], l3_aabaab, t1_bb, optimize=True)
    l3_res += 1 * _tmp68
    _tmp69 = einsum('mnaj,iljdef,ai->deflmn', g_abab[o, o, v, o], l3_aabaab, t1_aa, optimize=True)
    l3_res += 1 * _tmp69
    l3_res -= 1 * _tmp69.transpose(0, 1, 2, 4, 3, 5)
    _tmp70 = einsum('mnja,jlidef,ai->deflmn', g_abab[o, o, o, v], l3_aabaab, t1_bb, optimize=True)
    l3_res += 1 * _tmp70
    l3_res -= 1 * _tmp70.transpose(0, 1, 2, 4, 3, 5)
    _tmp71 = einsum('jnai,aj,lmidef->deflmn', g_abab[o, o, v, o], t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp71
    _tmp72 = einsum('jnai,aj,lmidef->deflmn', g_bbbb[o, o, v, o], t1_bb, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp72
    _tmp73 = einsum('jmai,aj,lindef->deflmn', g_aaaa[o, o, v, o], t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp73
    _tmp74 = einsum('mjia,aj,lindef->deflmn', g_abab[o, o, o, v], t1_bb, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp74
    _tmp75 = einsum('lmaj,ijndef,ai->deflmn', g_aaaa[o, o, v, o], l3_aabaab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp75
    _tmp76 = einsum('jlai,mindef,aj->deflmn', g_aaaa[o, o, v, o], l3_aabaab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp76
    _tmp77 = einsum('ljia,mindef,aj->deflmn', g_abab[o, o, o, v], l3_aabaab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp77
    _tmp78 = einsum('ibde,ai,lmnbaf->deflmn', g_aaaa[o, v, v, v], t1_aa, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp78
    _tmp79 = einsum('ibdf,ai,lmnaeb->deflmn', g_abab[o, v, v, v], t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp79
    _tmp80 = einsum('bidf,ai,lmnbea->deflmn', g_abab[v, o, v, v], t1_bb, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp80
    _tmp81 = einsum('andb,bi,mliaef->deflmn', g_abab[v, o, v, v], t1_bb, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp81
    l3_res += 1 * _tmp81.transpose(1, 0, 2, 3, 4, 5)
    _tmp82 = einsum('madb,bi,ilnaef->deflmn', g_aaaa[o, v, v, v], t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp82
    l3_res -= 1 * _tmp82.transpose(1, 0, 2, 3, 4, 5)
    _tmp83 = einsum('madb,bi,lineaf->deflmn', g_abab[o, v, v, v], t1_bb, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp83
    l3_res -= 1 * _tmp83.transpose(1, 0, 2, 3, 4, 5)
    _tmp84 = einsum('ladb,imnaef,bi->deflmn', g_aaaa[o, v, v, v], l3_aabaab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp84
    l3_res += 1 * _tmp84.transpose(1, 0, 2, 3, 4, 5)
    _tmp85 = einsum('ladb,mineaf,bi->deflmn', g_abab[o, v, v, v], l3_abbabb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp85
    l3_res += 1 * _tmp85.transpose(1, 0, 2, 3, 4, 5)
    _tmp86 = einsum('iadb,lmnaef,bi->deflmn', g_aaaa[o, v, v, v], l3_aabaab, t1_aa, optimize=True)
    l3_res += 1 * _tmp86
    l3_res -= 1 * _tmp86.transpose(1, 0, 2, 3, 4, 5)
    _tmp87 = einsum('aidb,lmnaef,bi->deflmn', g_abab[v, o, v, v], l3_aabaab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp87
    l3_res += 1 * _tmp87.transpose(1, 0, 2, 3, 4, 5)
    _tmp88 = einsum('bief,lmnbda,ai->deflmn', g_abab[v, o, v, v], l3_aabaab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp88
    _tmp89 = einsum('ibef,lmnadb,ai->deflmn', g_abab[o, v, v, v], l3_aabaab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp89
    _tmp90 = einsum('anbf,bi,ilmade->deflmn', g_abab[v, o, v, v], t1_aa, l3_aaaaaa, optimize=True)
    l3_res -= 1 * _tmp90
    _tmp91 = einsum('nafb,bi,mlieda->deflmn', g_bbbb[o, v, v, v], t1_bb, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp91
    _tmp92 = einsum('mabf,bi,ilneda->deflmn', g_abab[o, v, v, v], t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp92
    _tmp93 = einsum('labf,imneda,bi->deflmn', g_abab[o, v, v, v], l3_aabaab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp93
    _tmp94 = einsum('iabf,lmneda,bi->deflmn', g_abab[o, v, v, v], l3_aabaab, t1_aa, optimize=True)
    l3_res += 1 * _tmp94
    _tmp95 = einsum('iafb,lmneda,bi->deflmn', g_bbbb[o, v, v, v], l3_aabaab, t1_bb, optimize=True)
    l3_res -= 1 * _tmp95
    _tmp96 = einsum('jndf,ilmeba,baij->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res += 0.5 * _tmp96
    _tmp97 = einsum('jndf,mlieba,baji->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp97
    l3_res += 0.5 * _tmp97
    _tmp98 = einsum('jmde,ilnabf,baij->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp98
    _tmp99 = einsum('jmde,linbfa,baji->deflmn', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp99
    _tmp100 = einsum('jmde,linabf,abji->deflmn', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp100
    _tmp101 = einsum('mjdf,ilneba,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp101
    l3_res += 0.5 * _tmp101
    _tmp102 = einsum('mjdf,lineba,baij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res -= 0.5 * _tmp102
    _tmp103 = einsum('jlde,baij,imnabf->deflmn', g_aaaa[o, o, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp103
    _tmp104 = einsum('jlde,baji,minbfa->deflmn', g_aaaa[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp104
    _tmp105 = einsum('jlde,abji,minabf->deflmn', g_aaaa[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp105
    _tmp106 = einsum('ljdf,baij,imneba->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp106
    l3_res -= 0.5 * _tmp106
    _tmp107 = einsum('ljdf,baij,mineba->deflmn', g_abab[o, o, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp107
    _tmp108 = einsum('jide,baji,lmnabf->deflmn', g_aaaa[o, o, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l3_res += 0.25 * _tmp108
    _tmp109 = einsum('jidf,baji,lmneba->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res += 0.25 * _tmp109
    l3_res += 0.25 * _tmp109
    l3_res += 0.25 * _tmp109
    l3_res += 0.25 * _tmp109
    _tmp110 = einsum('mndb,iljeaf,abij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp110
    l3_res -= 0.5 * _tmp110.transpose(1, 0, 2, 3, 4, 5)
    l3_res -= 0.5 * _tmp110.transpose(0, 1, 2, 4, 3, 5)
    l3_res += 0.5 * _tmp110.transpose(1, 0, 2, 4, 3, 5)
    _tmp111 = einsum('mndb,ljieaf,abji->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp111
    l3_res += 0.5 * _tmp111.transpose(1, 0, 2, 3, 4, 5)
    l3_res += 0.5 * _tmp111.transpose(0, 1, 2, 4, 3, 5)
    l3_res -= 0.5 * _tmp111.transpose(1, 0, 2, 4, 3, 5)
    _tmp112 = einsum('mndb,ljiefa,baij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res += 0.5 * _tmp112
    l3_res -= 0.5 * _tmp112.transpose(1, 0, 2, 3, 4, 5)
    l3_res -= 0.5 * _tmp112.transpose(0, 1, 2, 4, 3, 5)
    l3_res += 0.5 * _tmp112.transpose(1, 0, 2, 4, 3, 5)
    _tmp113 = einsum('jndb,abji,mlieaf->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp113
    l3_res += 1 * _tmp113.transpose(1, 0, 2, 3, 4, 5)
    _tmp114 = einsum('jmdb,baij,ilneaf->deflmn', g_aaaa[o, o, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp114
    l3_res += 1 * _tmp114.transpose(1, 0, 2, 3, 4, 5)
    _tmp115 = einsum('jmdb,baji,linefa->deflmn', g_aaaa[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp115
    l3_res -= 1 * _tmp115.transpose(1, 0, 2, 3, 4, 5)
    _tmp116 = einsum('mjdb,abij,ilneaf->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp116
    l3_res += 1 * _tmp116.transpose(1, 0, 2, 3, 4, 5)
    _tmp117 = einsum('mjdb,baij,linefa->deflmn', g_abab[o, o, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp117
    l3_res -= 1 * _tmp117.transpose(1, 0, 2, 3, 4, 5)
    _tmp118 = einsum('lmdb,ijneaf,baij->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res += 0.5 * _tmp118
    l3_res -= 0.5 * _tmp118.transpose(1, 0, 2, 3, 4, 5)
    _tmp119 = einsum('lmdb,ijnefa,baij->deflmn', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp119
    l3_res += 0.5 * _tmp119.transpose(1, 0, 2, 3, 4, 5)
    l3_res -= 0.5 * _tmp119
    l3_res += 0.5 * _tmp119.transpose(1, 0, 2, 3, 4, 5)
    _tmp120 = einsum('jldb,imneaf,baij->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res += 1 * _tmp120
    l3_res -= 1 * _tmp120.transpose(1, 0, 2, 3, 4, 5)
    _tmp121 = einsum('ljdb,imneaf,abij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 1 * _tmp121
    l3_res -= 1 * _tmp121.transpose(1, 0, 2, 3, 4, 5)
    _tmp122 = einsum('jldb,minefa,baji->deflmn', g_aaaa[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res -= 1 * _tmp122
    l3_res += 1 * _tmp122.transpose(1, 0, 2, 3, 4, 5)
    _tmp123 = einsum('ljdb,minefa,baij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res -= 1 * _tmp123
    l3_res += 1 * _tmp123.transpose(1, 0, 2, 3, 4, 5)
    _tmp124 = einsum('jidb,lmneaf,baji->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res += 0.5 * _tmp124
    l3_res -= 0.5 * _tmp124.transpose(1, 0, 2, 3, 4, 5)
    _tmp125 = einsum('jidb,lmneaf,abji->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp125
    l3_res += 0.5 * _tmp125.transpose(1, 0, 2, 3, 4, 5)
    l3_res -= 0.5 * _tmp125
    l3_res += 0.5 * _tmp125.transpose(1, 0, 2, 3, 4, 5)
    _tmp126 = einsum('jnef,baij,ilmdba->deflmn', g_abab[o, o, v, v], t2_aaaa, l3_aaaaaa, optimize=True)
    l3_res -= 0.5 * _tmp126
    _tmp127 = einsum('jnef,baji,mlidba->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp127
    l3_res -= 0.5 * _tmp127
    _tmp128 = einsum('mjef,baij,ilndba->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp128
    l3_res -= 0.5 * _tmp128
    _tmp129 = einsum('mjef,baij,lindba->deflmn', g_abab[o, o, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp129
    _tmp130 = einsum('ljef,imndba,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp130
    l3_res += 0.5 * _tmp130
    _tmp131 = einsum('ljef,mindba,baij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res -= 0.5 * _tmp131
    _tmp132 = einsum('jief,lmndba,baji->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.25 * _tmp132
    l3_res -= 0.25 * _tmp132
    l3_res -= 0.25 * _tmp132
    l3_res -= 0.25 * _tmp132
    _tmp133 = einsum('mnbf,ijldea,baij->deflmn', g_abab[o, o, v, v], l3_aaaaaa, t2_aaaa, optimize=True)
    l3_res += 0.5 * _tmp133
    l3_res -= 0.5 * _tmp133.transpose(0, 1, 2, 4, 3, 5)
    _tmp134 = einsum('mnbf,iljdea,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp134
    l3_res += 0.5 * _tmp134.transpose(0, 1, 2, 4, 3, 5)
    _tmp135 = einsum('mnbf,ljidea,baji->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp135
    l3_res -= 0.5 * _tmp135.transpose(0, 1, 2, 4, 3, 5)
    _tmp136 = einsum('jnbf,baij,ilmdea->deflmn', g_abab[o, o, v, v], t2_aaaa, l3_aaaaaa, optimize=True)
    l3_res += 1 * _tmp136
    _tmp137 = einsum('jnbf,baji,mlidea->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp137
    _tmp138 = einsum('jnfb,abij,ilmdea->deflmn', g_bbbb[o, o, v, v], t2_abab, l3_aaaaaa, optimize=True)
    l3_res += 1 * _tmp138
    _tmp139 = einsum('jnfb,baij,mlidea->deflmn', g_bbbb[o, o, v, v], t2_bbbb, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp139
    _tmp140 = einsum('mjbf,baij,ilndea->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp140
    _tmp141 = einsum('ljbf,imndea,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 1 * _tmp141
    _tmp142 = einsum('jibf,lmndea,baji->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp142
    l3_res += 0.5 * _tmp142
    _tmp143 = einsum('jifb,lmndea,baji->deflmn', g_bbbb[o, o, v, v], l3_aabaab, t2_bbbb, optimize=True)
    l3_res -= 0.5 * _tmp143
    _tmp144 = einsum('mnab,iljdef,abij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.25 * _tmp144
    l3_res -= 0.25 * _tmp144.transpose(0, 1, 2, 4, 3, 5)
    l3_res += 0.25 * _tmp144
    l3_res -= 0.25 * _tmp144.transpose(0, 1, 2, 4, 3, 5)
    _tmp145 = einsum('mnab,ljidef,abji->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.25 * _tmp145
    l3_res += 0.25 * _tmp145.transpose(0, 1, 2, 4, 3, 5)
    l3_res -= 0.25 * _tmp145
    l3_res += 0.25 * _tmp145.transpose(0, 1, 2, 4, 3, 5)
    _tmp146 = einsum('jnab,abji,mlidef->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp146
    l3_res -= 0.5 * _tmp146
    _tmp147 = einsum('jnab,abij,mlidef->deflmn', g_bbbb[o, o, v, v], t2_bbbb, l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp147
    _tmp148 = einsum('jmab,abij,ilndef->deflmn', g_aaaa[o, o, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp148
    _tmp149 = einsum('mjab,abij,ilndef->deflmn', g_abab[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp149
    l3_res -= 0.5 * _tmp149
    _tmp150 = einsum('lmab,ijndef,abij->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res -= 0.25 * _tmp150
    _tmp151 = einsum('jlab,imndef,abij->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp151
    _tmp152 = einsum('ljab,imndef,abij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp152
    l3_res += 0.5 * _tmp152
    _tmp153 = einsum('jide,aj,bi,lmnabf->deflmn', g_aaaa[o, o, v, v], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp153
    _tmp154 = einsum('jidf,aj,bi,lmneab->deflmn', g_abab[o, o, v, v], t1_aa, t1_bb, l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp154
    l3_res += 0.5 * _tmp154
    _tmp155 = einsum('jndb,aj,bi,mlieaf->deflmn', g_abab[o, o, v, v], t1_aa, t1_bb, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp155
    l3_res += 1 * _tmp155.transpose(1, 0, 2, 3, 4, 5)
    _tmp156 = einsum('jmdb,aj,bi,ilneaf->deflmn', g_aaaa[o, o, v, v], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp156
    l3_res += 1 * _tmp156.transpose(1, 0, 2, 3, 4, 5)
    _tmp157 = einsum('mjdb,aj,bi,linefa->deflmn', g_abab[o, o, v, v], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp157
    l3_res -= 1 * _tmp157.transpose(1, 0, 2, 3, 4, 5)
    _tmp158 = einsum('jldb,imneaf,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t1_aa, t1_aa, optimize=True)
    l3_res += 1 * _tmp158
    l3_res -= 1 * _tmp158.transpose(1, 0, 2, 3, 4, 5)
    _tmp159 = einsum('ljdb,minefa,aj,bi->deflmn', g_abab[o, o, v, v], l3_abbabb, t1_bb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp159
    l3_res += 1 * _tmp159.transpose(1, 0, 2, 3, 4, 5)
    _tmp160 = einsum('jidb,lmneaf,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t1_aa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp160
    l3_res += 1 * _tmp160.transpose(1, 0, 2, 3, 4, 5)
    _tmp161 = einsum('jidb,lmneaf,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_aa, t1_bb, optimize=True)
    l3_res -= 1 * _tmp161
    l3_res += 1 * _tmp161.transpose(1, 0, 2, 3, 4, 5)
    _tmp162 = einsum('ijef,lmndba,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_aa, optimize=True)
    l3_res -= 0.5 * _tmp162
    l3_res -= 0.5 * _tmp162
    _tmp163 = einsum('jnbf,aj,bi,ilmdea->deflmn', g_abab[o, o, v, v], t1_aa, t1_aa, l3_aaaaaa, optimize=True)
    l3_res += 1 * _tmp163
    _tmp164 = einsum('jnfb,aj,bi,mlidea->deflmn', g_bbbb[o, o, v, v], t1_bb, t1_bb, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp164
    _tmp165 = einsum('mjbf,aj,bi,ilndea->deflmn', g_abab[o, o, v, v], t1_bb, t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp165
    _tmp166 = einsum('ljbf,imndea,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_aa, optimize=True)
    l3_res -= 1 * _tmp166
    _tmp167 = einsum('ijbf,lmndea,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_aa, optimize=True)
    l3_res += 1 * _tmp167
    _tmp168 = einsum('jifb,lmndea,aj,bi->deflmn', g_bbbb[o, o, v, v], l3_aabaab, t1_bb, t1_bb, optimize=True)
    l3_res += 1 * _tmp168
    _tmp169 = einsum('mnba,iljdef,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_aa, optimize=True)
    l3_res += 0.5 * _tmp169
    l3_res -= 0.5 * _tmp169.transpose(0, 1, 2, 4, 3, 5)
    _tmp170 = einsum('mnab,ljidef,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_aa, t1_bb, optimize=True)
    l3_res -= 0.5 * _tmp170
    l3_res += 0.5 * _tmp170.transpose(0, 1, 2, 4, 3, 5)
    _tmp171 = einsum('jnab,aj,bi,mlidef->deflmn', g_abab[o, o, v, v], t1_aa, t1_bb, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp171
    _tmp172 = einsum('jnab,aj,bi,mlidef->deflmn', g_bbbb[o, o, v, v], t1_bb, t1_bb, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp172
    _tmp173 = einsum('jmab,aj,bi,ilndef->deflmn', g_aaaa[o, o, v, v], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp173
    _tmp174 = einsum('mjba,aj,bi,ilndef->deflmn', g_abab[o, o, v, v], t1_bb, t1_aa, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp174
    _tmp175 = einsum('lmab,ijndef,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t1_aa, t1_aa, optimize=True)
    l3_res += 0.5 * _tmp175
    _tmp176 = einsum('jlab,imndef,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_aabaab, t1_aa, t1_aa, optimize=True)
    l3_res += 1 * _tmp176
    _tmp177 = einsum('ljba,imndef,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_bb, t1_aa, optimize=True)
    l3_res += 1 * _tmp177
    return l3_res


def l3_abbabb_residual(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb):
    nv, no = t1_aa.shape
    l3_res = np.zeros((nv, nv, nv, no, no, no))
    _tmp0 = einsum('ne,lmdf->deflmn', f_bb[o, v], l2_abab, optimize=True)
    l3_res += 1 * _tmp0
    l3_res -= 1 * _tmp0.transpose(0, 1, 2, 3, 5, 4)
    _tmp1 = einsum('ld,mnef->deflmn', f_aa[o, v], l2_bbbb, optimize=True)
    l3_res -= 1 * _tmp1
    _tmp2 = einsum('nf,lmde->deflmn', f_bb[o, v], l2_abab, optimize=True)
    l3_res -= 1 * _tmp2
    l3_res += 1 * _tmp2.transpose(0, 1, 2, 3, 5, 4)
    _tmp3 = einsum('ni,lmidef->deflmn', f_bb[o, o], l3_abbabb, optimize=True)
    l3_res += 1 * _tmp3
    l3_res -= 1 * _tmp3.transpose(0, 1, 2, 3, 5, 4)
    _tmp4 = einsum('li,inmdef->deflmn', f_aa[o, o], l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp4
    _tmp5 = einsum('ad,lmnaef->deflmn', f_aa[v, v], l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp5
    _tmp6 = einsum('ae,lmndaf->deflmn', f_bb[v, v], l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp6
    _tmp7 = einsum('af,lmndae->deflmn', f_bb[v, v], l3_abbabb, optimize=True)
    l3_res += 1 * _tmp7
    _tmp8 = einsum('id,ai,lmnafe->deflmn', f_aa[o, v], t1_aa, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp8
    _tmp9 = einsum('ie,ai,lmndfa->deflmn', f_bb[o, v], t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp9
    _tmp10 = einsum('if,lmndea,ai->deflmn', f_bb[o, v], l3_abbabb, t1_bb, optimize=True)
    l3_res += 1 * _tmp10
    _tmp11 = einsum('na,limdef,ai->deflmn', f_bb[o, v], l3_abbabb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp11
    l3_res += 1 * _tmp11.transpose(0, 1, 2, 3, 5, 4)
    _tmp12 = einsum('la,imndef,ai->deflmn', f_aa[o, v], l3_abbabb, t1_aa, optimize=True)
    l3_res += 1 * _tmp12
    _tmp13 = einsum('lnde,mf->deflmn', g_abab[o, o, v, v], l1_bb, optimize=True)
    l3_res += 1 * _tmp13
    l3_res -= 1 * _tmp13.transpose(0, 2, 1, 3, 4, 5)
    _tmp14 = einsum('lmde,nf->deflmn', g_abab[o, o, v, v], l1_bb, optimize=True)
    l3_res -= 1 * _tmp14
    l3_res += 1 * _tmp14.transpose(0, 2, 1, 3, 4, 5)
    _tmp15 = einsum('mnef,ld->deflmn', g_bbbb[o, o, v, v], l1_aa, optimize=True)
    l3_res -= 1 * _tmp15
    _tmp16 = einsum('mnei,lidf->deflmn', g_bbbb[o, o, v, o], l2_abab, optimize=True)
    l3_res += 1 * _tmp16
    _tmp17 = einsum('lndi,mief->deflmn', g_abab[o, o, v, o], l2_bbbb, optimize=True)
    l3_res += 1 * _tmp17
    _tmp18 = einsum('lnie,imdf->deflmn', g_abab[o, o, o, v], l2_abab, optimize=True)
    l3_res -= 1 * _tmp18
    _tmp19 = einsum('lmdi,nief->deflmn', g_abab[o, o, v, o], l2_bbbb, optimize=True)
    l3_res -= 1 * _tmp19
    _tmp20 = einsum('lmie,indf->deflmn', g_abab[o, o, o, v], l2_abab, optimize=True)
    l3_res += 1 * _tmp20
    _tmp21 = einsum('mnfi,lide->deflmn', g_bbbb[o, o, v, o], l2_abab, optimize=True)
    l3_res -= 1 * _tmp21
    _tmp22 = einsum('lnif,imde->deflmn', g_abab[o, o, o, v], l2_abab, optimize=True)
    l3_res += 1 * _tmp22
    _tmp23 = einsum('lmif,inde->deflmn', g_abab[o, o, o, v], l2_abab, optimize=True)
    l3_res -= 1 * _tmp23
    _tmp24 = einsum('ande,lmaf->deflmn', g_abab[v, o, v, v], l2_abab, optimize=True)
    l3_res += 1 * _tmp24
    l3_res -= 1 * _tmp24.transpose(0, 2, 1, 3, 4, 5)
    l3_res -= 1 * _tmp24.transpose(0, 1, 2, 3, 5, 4)
    l3_res += 1 * _tmp24.transpose(0, 2, 1, 3, 5, 4)
    _tmp25 = einsum('lade,mnaf->deflmn', g_abab[o, v, v, v], l2_bbbb, optimize=True)
    l3_res -= 1 * _tmp25
    l3_res += 1 * _tmp25.transpose(0, 2, 1, 3, 4, 5)
    _tmp26 = einsum('naef,lmda->deflmn', g_bbbb[o, v, v, v], l2_abab, optimize=True)
    l3_res += 1 * _tmp26
    l3_res -= 1 * _tmp26.transpose(0, 1, 2, 3, 5, 4)
    _tmp27 = einsum('mnij,lijdef->deflmn', g_bbbb[o, o, o, o], l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp27
    _tmp28 = einsum('lnij,imjdef->deflmn', g_abab[o, o, o, o], l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp28
    _tmp29 = einsum('lnji,jimdef->deflmn', g_abab[o, o, o, o], l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp29
    _tmp30 = einsum('lmij,injdef->deflmn', g_abab[o, o, o, o], l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp30
    _tmp31 = einsum('lmji,jindef->deflmn', g_abab[o, o, o, o], l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp31
    _tmp32 = einsum('andi,lmiaef->deflmn', g_abab[v, o, v, o], l3_abbabb, optimize=True)
    l3_res += 1 * _tmp32
    l3_res -= 1 * _tmp32.transpose(0, 1, 2, 3, 5, 4)
    _tmp33 = einsum('anie,limadf->deflmn', g_abab[v, o, o, v], l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp33
    l3_res += 1 * _tmp33.transpose(0, 1, 2, 3, 5, 4)
    _tmp34 = einsum('naei,lmidaf->deflmn', g_bbbb[o, v, v, o], l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp34
    l3_res += 1 * _tmp34.transpose(0, 1, 2, 3, 5, 4)
    _tmp35 = einsum('ladi,inmaef->deflmn', g_aaaa[o, v, v, o], l3_abbabb, optimize=True)
    l3_res += 1 * _tmp35
    _tmp36 = einsum('ladi,mniaef->deflmn', g_abab[o, v, v, o], l3_bbbbbb, optimize=True)
    l3_res -= 1 * _tmp36
    _tmp37 = einsum('laie,inmdaf->deflmn', g_abab[o, v, o, v], l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp37
    _tmp38 = einsum('anif,limade->deflmn', g_abab[v, o, o, v], l3_aabaab, optimize=True)
    l3_res += 1 * _tmp38
    l3_res -= 1 * _tmp38.transpose(0, 1, 2, 3, 5, 4)
    _tmp39 = einsum('nafi,lmidae->deflmn', g_bbbb[o, v, v, o], l3_abbabb, optimize=True)
    l3_res += 1 * _tmp39
    l3_res -= 1 * _tmp39.transpose(0, 1, 2, 3, 5, 4)
    _tmp40 = einsum('laif,inmdae->deflmn', g_abab[o, v, o, v], l3_abbabb, optimize=True)
    l3_res += 1 * _tmp40
    _tmp41 = einsum('bade,lmnbaf->deflmn', g_abab[v, v, v, v], l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp41
    l3_res += 0.5 * _tmp41.transpose(0, 2, 1, 3, 4, 5)
    l3_res -= 0.5 * _tmp41
    l3_res += 0.5 * _tmp41.transpose(0, 2, 1, 3, 4, 5)
    _tmp42 = einsum('baef,lmndab->deflmn', g_bbbb[v, v, v, v], l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp42
    _tmp43 = einsum('inde,lmaf,ai->deflmn', g_abab[o, o, v, v], l2_abab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp43
    l3_res += 1 * _tmp43.transpose(0, 2, 1, 3, 4, 5)
    l3_res += 1 * _tmp43.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 1 * _tmp43.transpose(0, 2, 1, 3, 5, 4)
    _tmp44 = einsum('lide,mnfa,ai->deflmn', g_abab[o, o, v, v], l2_bbbb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp44
    l3_res += 1 * _tmp44.transpose(0, 2, 1, 3, 4, 5)
    _tmp45 = einsum('mnea,lidf,ai->deflmn', g_bbbb[o, o, v, v], l2_abab, t1_bb, optimize=True)
    l3_res += 1 * _tmp45
    _tmp46 = einsum('lnda,imef,ai->deflmn', g_abab[o, o, v, v], l2_bbbb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp46
    _tmp47 = einsum('lnae,imdf,ai->deflmn', g_abab[o, o, v, v], l2_abab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp47
    _tmp48 = einsum('inae,ai,lmdf->deflmn', g_abab[o, o, v, v], t1_aa, l2_abab, optimize=True)
    l3_res += 1 * _tmp48
    l3_res -= 1 * _tmp48.transpose(0, 1, 2, 3, 5, 4)
    _tmp49 = einsum('inea,ai,lmdf->deflmn', g_bbbb[o, o, v, v], t1_bb, l2_abab, optimize=True)
    l3_res -= 1 * _tmp49
    l3_res += 1 * _tmp49.transpose(0, 1, 2, 3, 5, 4)
    _tmp50 = einsum('lmda,ai,inef->deflmn', g_abab[o, o, v, v], t1_bb, l2_bbbb, optimize=True)
    l3_res += 1 * _tmp50
    _tmp51 = einsum('lmae,ai,indf->deflmn', g_abab[o, o, v, v], t1_aa, l2_abab, optimize=True)
    l3_res += 1 * _tmp51
    _tmp52 = einsum('ilda,ai,mnef->deflmn', g_aaaa[o, o, v, v], t1_aa, l2_bbbb, optimize=True)
    l3_res += 1 * _tmp52
    _tmp53 = einsum('lida,ai,mnef->deflmn', g_abab[o, o, v, v], t1_bb, l2_bbbb, optimize=True)
    l3_res -= 1 * _tmp53
    _tmp54 = einsum('inef,lmda,ai->deflmn', g_bbbb[o, o, v, v], l2_abab, t1_bb, optimize=True)
    l3_res += 1 * _tmp54
    l3_res -= 1 * _tmp54.transpose(0, 1, 2, 3, 5, 4)
    _tmp55 = einsum('mnfa,ai,lide->deflmn', g_bbbb[o, o, v, v], t1_bb, l2_abab, optimize=True)
    l3_res -= 1 * _tmp55
    _tmp56 = einsum('lnaf,ai,imde->deflmn', g_abab[o, o, v, v], t1_aa, l2_abab, optimize=True)
    l3_res += 1 * _tmp56
    _tmp57 = einsum('inaf,lmde,ai->deflmn', g_abab[o, o, v, v], l2_abab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp57
    l3_res += 1 * _tmp57.transpose(0, 1, 2, 3, 5, 4)
    _tmp58 = einsum('infa,lmde,ai->deflmn', g_bbbb[o, o, v, v], l2_abab, t1_bb, optimize=True)
    l3_res += 1 * _tmp58
    l3_res -= 1 * _tmp58.transpose(0, 1, 2, 3, 5, 4)
    _tmp59 = einsum('lmaf,inde,ai->deflmn', g_abab[o, o, v, v], l2_abab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp59
    _tmp60 = einsum('jndi,aj,lmiafe->deflmn', g_abab[o, o, v, o], t1_aa, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp60
    l3_res -= 1 * _tmp60.transpose(0, 1, 2, 3, 5, 4)
    _tmp61 = einsum('jnie,aj,limdaf->deflmn', g_abab[o, o, o, v], t1_aa, l3_aabaab, optimize=True)
    l3_res -= 1 * _tmp61
    l3_res += 1 * _tmp61.transpose(0, 1, 2, 3, 5, 4)
    _tmp62 = einsum('jnei,aj,lmidfa->deflmn', g_bbbb[o, o, v, o], t1_bb, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp62
    l3_res -= 1 * _tmp62.transpose(0, 1, 2, 3, 5, 4)
    _tmp63 = einsum('jldi,aj,inmafe->deflmn', g_aaaa[o, o, v, o], t1_aa, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp63
    _tmp64 = einsum('ljdi,aj,mniefa->deflmn', g_abab[o, o, v, o], t1_bb, l3_bbbbbb, optimize=True)
    l3_res += 1 * _tmp64
    _tmp65 = einsum('ljie,aj,inmdfa->deflmn', g_abab[o, o, o, v], t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp65
    _tmp66 = einsum('jnif,limdae,aj->deflmn', g_abab[o, o, o, v], l3_aabaab, t1_aa, optimize=True)
    l3_res += 1 * _tmp66
    l3_res -= 1 * _tmp66.transpose(0, 1, 2, 3, 5, 4)
    _tmp67 = einsum('jnfi,lmidea,aj->deflmn', g_bbbb[o, o, v, o], l3_abbabb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp67
    l3_res += 1 * _tmp67.transpose(0, 1, 2, 3, 5, 4)
    _tmp68 = einsum('ljif,inmdea,aj->deflmn', g_abab[o, o, o, v], l3_abbabb, t1_bb, optimize=True)
    l3_res += 1 * _tmp68
    _tmp69 = einsum('mnaj,ai,lijdef->deflmn', g_bbbb[o, o, v, o], t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp69
    _tmp70 = einsum('lnaj,ai,imjdef->deflmn', g_abab[o, o, v, o], t1_aa, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp70
    _tmp71 = einsum('lnja,ai,jmidef->deflmn', g_abab[o, o, o, v], t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp71
    _tmp72 = einsum('jnai,lmidef,aj->deflmn', g_abab[o, o, v, o], l3_abbabb, t1_aa, optimize=True)
    l3_res += 1 * _tmp72
    l3_res -= 1 * _tmp72.transpose(0, 1, 2, 3, 5, 4)
    _tmp73 = einsum('jnai,lmidef,aj->deflmn', g_bbbb[o, o, v, o], l3_abbabb, t1_bb, optimize=True)
    l3_res += 1 * _tmp73
    l3_res -= 1 * _tmp73.transpose(0, 1, 2, 3, 5, 4)
    _tmp74 = einsum('lmaj,injdef,ai->deflmn', g_abab[o, o, v, o], l3_abbabb, t1_aa, optimize=True)
    l3_res += 1 * _tmp74
    _tmp75 = einsum('lmja,jnidef,ai->deflmn', g_abab[o, o, o, v], l3_abbabb, t1_bb, optimize=True)
    l3_res += 1 * _tmp75
    _tmp76 = einsum('jlai,inmdef,aj->deflmn', g_aaaa[o, o, v, o], l3_abbabb, t1_aa, optimize=True)
    l3_res -= 1 * _tmp76
    _tmp77 = einsum('ljia,inmdef,aj->deflmn', g_abab[o, o, o, v], l3_abbabb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp77
    _tmp78 = einsum('bide,lmnbfa,ai->deflmn', g_abab[v, o, v, v], l3_abbabb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp78
    l3_res += 1 * _tmp78.transpose(0, 2, 1, 3, 4, 5)
    _tmp79 = einsum('ibde,lmnafb,ai->deflmn', g_abab[o, v, v, v], l3_abbabb, t1_aa, optimize=True)
    l3_res -= 1 * _tmp79
    l3_res += 1 * _tmp79.transpose(0, 2, 1, 3, 4, 5)
    _tmp80 = einsum('andb,bi,limaef->deflmn', g_abab[v, o, v, v], t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp80
    l3_res += 1 * _tmp80.transpose(0, 1, 2, 3, 5, 4)
    _tmp81 = einsum('anbe,bi,ilmadf->deflmn', g_abab[v, o, v, v], t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp81
    l3_res -= 1 * _tmp81.transpose(0, 1, 2, 3, 5, 4)
    _tmp82 = einsum('naeb,bi,limdaf->deflmn', g_bbbb[o, v, v, v], t1_bb, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp82
    l3_res -= 1 * _tmp82.transpose(0, 1, 2, 3, 5, 4)
    _tmp83 = einsum('ladb,bi,imnaef->deflmn', g_aaaa[o, v, v, v], t1_aa, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp83
    _tmp84 = einsum('ladb,bi,imnaef->deflmn', g_abab[o, v, v, v], t1_bb, l3_bbbbbb, optimize=True)
    l3_res -= 1 * _tmp84
    _tmp85 = einsum('labe,bi,imndaf->deflmn', g_abab[o, v, v, v], t1_aa, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp85
    _tmp86 = einsum('iadb,bi,lmnaef->deflmn', g_aaaa[o, v, v, v], t1_aa, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp86
    _tmp87 = einsum('aidb,bi,lmnaef->deflmn', g_abab[v, o, v, v], t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp87
    _tmp88 = einsum('iabe,bi,lmndaf->deflmn', g_abab[o, v, v, v], t1_aa, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp88
    _tmp89 = einsum('iaeb,bi,lmndaf->deflmn', g_bbbb[o, v, v, v], t1_bb, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp89
    _tmp90 = einsum('ibef,lmndba,ai->deflmn', g_bbbb[o, v, v, v], l3_abbabb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp90
    _tmp91 = einsum('anbf,ilmade,bi->deflmn', g_abab[v, o, v, v], l3_aabaab, t1_aa, optimize=True)
    l3_res -= 1 * _tmp91
    l3_res += 1 * _tmp91.transpose(0, 1, 2, 3, 5, 4)
    _tmp92 = einsum('nafb,limdae,bi->deflmn', g_bbbb[o, v, v, v], l3_abbabb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp92
    l3_res += 1 * _tmp92.transpose(0, 1, 2, 3, 5, 4)
    _tmp93 = einsum('labf,imndae,bi->deflmn', g_abab[o, v, v, v], l3_abbabb, t1_aa, optimize=True)
    l3_res -= 1 * _tmp93
    _tmp94 = einsum('iabf,lmndae,bi->deflmn', g_abab[o, v, v, v], l3_abbabb, t1_aa, optimize=True)
    l3_res += 1 * _tmp94
    _tmp95 = einsum('iafb,lmndae,bi->deflmn', g_bbbb[o, v, v, v], l3_abbabb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp95
    _tmp96 = einsum('jnde,ilmabf,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res += 0.5 * _tmp96
    l3_res -= 0.5 * _tmp96.transpose(0, 2, 1, 3, 4, 5)
    l3_res -= 0.5 * _tmp96.transpose(0, 1, 2, 3, 5, 4)
    l3_res += 0.5 * _tmp96.transpose(0, 2, 1, 3, 5, 4)
    _tmp97 = einsum('jnde,limbfa,baji->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp97
    l3_res -= 0.5 * _tmp97.transpose(0, 2, 1, 3, 4, 5)
    l3_res -= 0.5 * _tmp97.transpose(0, 1, 2, 3, 5, 4)
    l3_res += 0.5 * _tmp97.transpose(0, 2, 1, 3, 5, 4)
    _tmp98 = einsum('jnde,limabf,abji->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp98
    l3_res += 0.5 * _tmp98.transpose(0, 2, 1, 3, 4, 5)
    l3_res += 0.5 * _tmp98.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 0.5 * _tmp98.transpose(0, 2, 1, 3, 5, 4)
    _tmp99 = einsum('ljde,imnbfa,baij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp99
    l3_res += 0.5 * _tmp99.transpose(0, 2, 1, 3, 4, 5)
    _tmp100 = einsum('ljde,imnabf,abij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp100
    l3_res -= 0.5 * _tmp100.transpose(0, 2, 1, 3, 4, 5)
    _tmp101 = einsum('ljde,imnfba,baij->deflmn', g_abab[o, o, v, v], l3_bbbbbb, t2_bbbb, optimize=True)
    l3_res += 0.5 * _tmp101
    l3_res -= 0.5 * _tmp101.transpose(0, 2, 1, 3, 4, 5)
    _tmp102 = einsum('jide,lmnbfa,baji->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 0.25 * _tmp102
    l3_res -= 0.25 * _tmp102.transpose(0, 2, 1, 3, 4, 5)
    l3_res += 0.25 * _tmp102
    l3_res -= 0.25 * _tmp102.transpose(0, 2, 1, 3, 4, 5)
    _tmp103 = einsum('jide,lmnabf,abji->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res -= 0.25 * _tmp103
    l3_res += 0.25 * _tmp103.transpose(0, 2, 1, 3, 4, 5)
    l3_res -= 0.25 * _tmp103
    l3_res += 0.25 * _tmp103.transpose(0, 2, 1, 3, 4, 5)
    _tmp104 = einsum('mneb,iljdaf,abij->deflmn', g_bbbb[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp104
    _tmp105 = einsum('mneb,ljidaf,abji->deflmn', g_bbbb[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp105
    _tmp106 = einsum('mneb,ljidfa,baij->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res -= 0.5 * _tmp106
    _tmp107 = einsum('lndb,ijmafe,abij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp107
    l3_res += 0.5 * _tmp107
    _tmp108 = einsum('lndb,ijmefa,baij->deflmn', g_abab[o, o, v, v], l3_bbbbbb, t2_bbbb, optimize=True)
    l3_res += 0.5 * _tmp108
    _tmp109 = einsum('lnbe,ijmdaf,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp109
    _tmp110 = einsum('lnbe,ijmdfa,baij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp110
    l3_res += 0.5 * _tmp110
    _tmp111 = einsum('jndb,abji,limafe->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp111
    l3_res += 1 * _tmp111.transpose(0, 1, 2, 3, 5, 4)
    _tmp112 = einsum('jnbe,baij,ilmdaf->deflmn', g_abab[o, o, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp112
    l3_res -= 1 * _tmp112.transpose(0, 1, 2, 3, 5, 4)
    _tmp113 = einsum('jnbe,baji,limdfa->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp113
    l3_res += 1 * _tmp113.transpose(0, 1, 2, 3, 5, 4)
    _tmp114 = einsum('jneb,abij,ilmdaf->deflmn', g_bbbb[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp114
    l3_res -= 1 * _tmp114.transpose(0, 1, 2, 3, 5, 4)
    _tmp115 = einsum('jneb,baij,limdfa->deflmn', g_bbbb[o, o, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp115
    l3_res += 1 * _tmp115.transpose(0, 1, 2, 3, 5, 4)
    _tmp116 = einsum('lmdb,abij,ijnafe->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp116
    l3_res -= 0.5 * _tmp116
    _tmp117 = einsum('lmdb,baij,ijnefa->deflmn', g_abab[o, o, v, v], t2_bbbb, l3_bbbbbb, optimize=True)
    l3_res -= 0.5 * _tmp117
    _tmp118 = einsum('lmbe,baij,ijndaf->deflmn', g_abab[o, o, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp118
    _tmp119 = einsum('lmbe,baij,ijndfa->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp119
    l3_res -= 0.5 * _tmp119
    _tmp120 = einsum('jldb,baij,imnafe->deflmn', g_aaaa[o, o, v, v], t2_aaaa, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp120
    _tmp121 = einsum('jldb,baji,imnefa->deflmn', g_aaaa[o, o, v, v], t2_abab, l3_bbbbbb, optimize=True)
    l3_res += 1 * _tmp121
    _tmp122 = einsum('ljdb,abij,imnafe->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp122
    _tmp123 = einsum('ljdb,baij,imnefa->deflmn', g_abab[o, o, v, v], t2_bbbb, l3_bbbbbb, optimize=True)
    l3_res += 1 * _tmp123
    _tmp124 = einsum('ljbe,baij,imndfa->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp124
    _tmp125 = einsum('jidb,baji,lmnafe->deflmn', g_aaaa[o, o, v, v], t2_aaaa, l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp125
    _tmp126 = einsum('jidb,abji,lmnafe->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp126
    l3_res -= 0.5 * _tmp126
    _tmp127 = einsum('jibe,baji,lmndfa->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp127
    l3_res -= 0.5 * _tmp127
    _tmp128 = einsum('jieb,baji,lmndfa->deflmn', g_bbbb[o, o, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp128
    _tmp129 = einsum('jnef,ilmdba,baij->deflmn', g_bbbb[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp129
    l3_res += 0.5 * _tmp129.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 0.5 * _tmp129
    l3_res += 0.5 * _tmp129.transpose(0, 1, 2, 3, 5, 4)
    _tmp130 = einsum('jnef,limdba,baij->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res += 0.5 * _tmp130
    l3_res -= 0.5 * _tmp130.transpose(0, 1, 2, 3, 5, 4)
    _tmp131 = einsum('jief,lmndba,baji->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res -= 0.25 * _tmp131
    _tmp132 = einsum('mnfb,abij,iljdae->deflmn', g_bbbb[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp132
    _tmp133 = einsum('mnfb,abji,ljidae->deflmn', g_bbbb[o, o, v, v], t2_abab, l3_aabaab, optimize=True)
    l3_res -= 0.5 * _tmp133
    _tmp134 = einsum('mnfb,baij,ljidea->deflmn', g_bbbb[o, o, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp134
    _tmp135 = einsum('lnbf,baij,ijmdae->deflmn', g_abab[o, o, v, v], t2_aaaa, l3_aabaab, optimize=True)
    l3_res += 0.5 * _tmp135
    _tmp136 = einsum('lnbf,baij,ijmdea->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp136
    l3_res -= 0.5 * _tmp136
    _tmp137 = einsum('jnbf,ilmdae,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res -= 1 * _tmp137
    l3_res += 1 * _tmp137.transpose(0, 1, 2, 3, 5, 4)
    _tmp138 = einsum('jnfb,ilmdae,abij->deflmn', g_bbbb[o, o, v, v], l3_aabaab, t2_abab, optimize=True)
    l3_res -= 1 * _tmp138
    l3_res += 1 * _tmp138.transpose(0, 1, 2, 3, 5, 4)
    _tmp139 = einsum('jnbf,limdea,baji->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 1 * _tmp139
    l3_res -= 1 * _tmp139.transpose(0, 1, 2, 3, 5, 4)
    _tmp140 = einsum('jnfb,limdea,baij->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res += 1 * _tmp140
    l3_res -= 1 * _tmp140.transpose(0, 1, 2, 3, 5, 4)
    _tmp141 = einsum('lmbf,ijndae,baij->deflmn', g_abab[o, o, v, v], l3_aabaab, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp141
    _tmp142 = einsum('lmbf,ijndea,baij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp142
    l3_res += 0.5 * _tmp142
    _tmp143 = einsum('ljbf,imndea,baij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res -= 1 * _tmp143
    _tmp144 = einsum('jibf,lmndea,baji->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp144
    l3_res += 0.5 * _tmp144
    _tmp145 = einsum('jifb,lmndea,baji->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res -= 0.5 * _tmp145
    _tmp146 = einsum('mnab,abij,ljidef->deflmn', g_bbbb[o, o, v, v], t2_bbbb, l3_abbabb, optimize=True)
    l3_res += 0.25 * _tmp146
    _tmp147 = einsum('lnab,abij,ijmdef->deflmn', g_abab[o, o, v, v], t2_abab, l3_abbabb, optimize=True)
    l3_res += 0.25 * _tmp147
    l3_res += 0.25 * _tmp147
    l3_res += 0.25 * _tmp147
    l3_res += 0.25 * _tmp147
    _tmp148 = einsum('jnab,limdef,abji->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res -= 0.5 * _tmp148
    l3_res += 0.5 * _tmp148.transpose(0, 1, 2, 3, 5, 4)
    l3_res -= 0.5 * _tmp148
    l3_res += 0.5 * _tmp148.transpose(0, 1, 2, 3, 5, 4)
    _tmp149 = einsum('jnab,limdef,abij->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t2_bbbb, optimize=True)
    l3_res += 0.5 * _tmp149
    l3_res -= 0.5 * _tmp149.transpose(0, 1, 2, 3, 5, 4)
    _tmp150 = einsum('lmab,ijndef,abij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res -= 0.25 * _tmp150
    l3_res -= 0.25 * _tmp150
    l3_res -= 0.25 * _tmp150
    l3_res -= 0.25 * _tmp150
    _tmp151 = einsum('jlab,imndef,abij->deflmn', g_aaaa[o, o, v, v], l3_abbabb, t2_aaaa, optimize=True)
    l3_res -= 0.5 * _tmp151
    _tmp152 = einsum('ljab,imndef,abij->deflmn', g_abab[o, o, v, v], l3_abbabb, t2_abab, optimize=True)
    l3_res += 0.5 * _tmp152
    l3_res += 0.5 * _tmp152
    _tmp153 = einsum('ijde,lmnbfa,aj,bi->deflmn', g_abab[o, o, v, v], l3_abbabb, t1_bb, t1_aa, optimize=True)
    l3_res += 0.5 * _tmp153
    l3_res -= 0.5 * _tmp153.transpose(0, 2, 1, 3, 4, 5)
    _tmp154 = einsum('jide,lmnabf,aj,bi->deflmn', g_abab[o, o, v, v], l3_abbabb, t1_aa, t1_bb, optimize=True)
    l3_res -= 0.5 * _tmp154
    l3_res += 0.5 * _tmp154.transpose(0, 2, 1, 3, 4, 5)
    _tmp155 = einsum('jndb,aj,bi,limafe->deflmn', g_abab[o, o, v, v], t1_aa, t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp155
    l3_res += 1 * _tmp155.transpose(0, 1, 2, 3, 5, 4)
    _tmp156 = einsum('jnbe,aj,bi,ilmdaf->deflmn', g_abab[o, o, v, v], t1_aa, t1_aa, l3_aabaab, optimize=True)
    l3_res += 1 * _tmp156
    l3_res -= 1 * _tmp156.transpose(0, 1, 2, 3, 5, 4)
    _tmp157 = einsum('jneb,aj,bi,limdfa->deflmn', g_bbbb[o, o, v, v], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp157
    l3_res += 1 * _tmp157.transpose(0, 1, 2, 3, 5, 4)
    _tmp158 = einsum('jldb,aj,bi,imnafe->deflmn', g_aaaa[o, o, v, v], t1_aa, t1_aa, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp158
    _tmp159 = einsum('ljdb,aj,bi,imnefa->deflmn', g_abab[o, o, v, v], t1_bb, t1_bb, l3_bbbbbb, optimize=True)
    l3_res += 1 * _tmp159
    _tmp160 = einsum('ljbe,aj,bi,imndfa->deflmn', g_abab[o, o, v, v], t1_bb, t1_aa, l3_abbabb, optimize=True)
    l3_res += 1 * _tmp160
    _tmp161 = einsum('jidb,aj,bi,lmnafe->deflmn', g_aaaa[o, o, v, v], t1_aa, t1_aa, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp161
    _tmp162 = einsum('jidb,aj,bi,lmnafe->deflmn', g_abab[o, o, v, v], t1_aa, t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp162
    _tmp163 = einsum('ijbe,aj,bi,lmndfa->deflmn', g_abab[o, o, v, v], t1_bb, t1_aa, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp163
    _tmp164 = einsum('jieb,aj,bi,lmndfa->deflmn', g_bbbb[o, o, v, v], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l3_res -= 1 * _tmp164
    _tmp165 = einsum('jief,lmndba,aj,bi->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t1_bb, t1_bb, optimize=True)
    l3_res += 0.5 * _tmp165
    _tmp166 = einsum('jnbf,ilmdae,aj,bi->deflmn', g_abab[o, o, v, v], l3_aabaab, t1_aa, t1_aa, optimize=True)
    l3_res -= 1 * _tmp166
    l3_res += 1 * _tmp166.transpose(0, 1, 2, 3, 5, 4)
    _tmp167 = einsum('jnfb,limdea,aj,bi->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t1_bb, t1_bb, optimize=True)
    l3_res += 1 * _tmp167
    l3_res -= 1 * _tmp167.transpose(0, 1, 2, 3, 5, 4)
    _tmp168 = einsum('ljbf,imndea,aj,bi->deflmn', g_abab[o, o, v, v], l3_abbabb, t1_bb, t1_aa, optimize=True)
    l3_res -= 1 * _tmp168
    _tmp169 = einsum('ijbf,lmndea,aj,bi->deflmn', g_abab[o, o, v, v], l3_abbabb, t1_bb, t1_aa, optimize=True)
    l3_res += 1 * _tmp169
    _tmp170 = einsum('jifb,lmndea,aj,bi->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t1_bb, t1_bb, optimize=True)
    l3_res += 1 * _tmp170
    _tmp171 = einsum('mnab,aj,bi,ljidef->deflmn', g_bbbb[o, o, v, v], t1_bb, t1_bb, l3_abbabb, optimize=True)
    l3_res -= 0.5 * _tmp171
    _tmp172 = einsum('lnab,aj,bi,jimdef->deflmn', g_abab[o, o, v, v], t1_aa, t1_bb, l3_abbabb, optimize=True)
    l3_res += 0.5 * _tmp172
    l3_res += 0.5 * _tmp172
    _tmp173 = einsum('jnab,limdef,aj,bi->deflmn', g_abab[o, o, v, v], l3_abbabb, t1_aa, t1_bb, optimize=True)
    l3_res -= 1 * _tmp173
    l3_res += 1 * _tmp173.transpose(0, 1, 2, 3, 5, 4)
    _tmp174 = einsum('jnab,limdef,aj,bi->deflmn', g_bbbb[o, o, v, v], l3_abbabb, t1_bb, t1_bb, optimize=True)
    l3_res -= 1 * _tmp174
    l3_res += 1 * _tmp174.transpose(0, 1, 2, 3, 5, 4)
    _tmp175 = einsum('lmba,ijndef,aj,bi->deflmn', g_abab[o, o, v, v], l3_abbabb, t1_bb, t1_aa, optimize=True)
    l3_res -= 0.5 * _tmp175
    l3_res -= 0.5 * _tmp175
    _tmp176 = einsum('jlab,imndef,aj,bi->deflmn', g_aaaa[o, o, v, v], l3_abbabb, t1_aa, t1_aa, optimize=True)
    l3_res += 1 * _tmp176
    _tmp177 = einsum('ljba,imndef,aj,bi->deflmn', g_abab[o, o, v, v], l3_abbabb, t1_bb, t1_aa, optimize=True)
    l3_res += 1 * _tmp177
    return l3_res

