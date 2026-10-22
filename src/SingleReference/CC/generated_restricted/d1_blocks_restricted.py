# GENERATED CODE -- restricted (spin-blocked) 1-RDM oo/vv/ov/vo blocks,
# _aa only (off-diagonal spin blocks vanish; final spatial density is
# 2x this block). Includes full Lambda1/Lambda2/Lambda3 feedback.
# Do not edit by hand.
import numpy as np
from src.SingleReference.CC.cached_einsum import einsum


def d1_oo_aa(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb, d_aa):
    nv, no = t1_aa.shape
    d1_oo = np.zeros((no, no))
    _tmp0 = einsum('mn->mn', d_aa[o, o])
    d1_oo += 1 * _tmp0
    _tmp1 = einsum('na,am->mn', l1_aa, t1_aa, optimize=True)
    d1_oo -= 1 * _tmp1
    _tmp2 = einsum('inba,baim->mn', l2_aaaa, t2_aaaa, optimize=True)
    d1_oo -= 0.5 * _tmp2
    _tmp3 = einsum('niba,bami->mn', l2_abab, t2_abab, optimize=True)
    d1_oo -= 0.5 * _tmp3
    d1_oo -= 0.5 * _tmp3
    _tmp4 = einsum('ijncba,cbaijm->mn', l3_aaaaaa, t3_aaaaaa, optimize=True)
    d1_oo -= 0.0833333333333333 * _tmp4
    _tmp5 = einsum('injcba,cbaimj->mn', l3_aabaab, t3_aabaab, optimize=True)
    d1_oo -= 0.0833333333333333 * _tmp5
    d1_oo -= 0.0833333333333333 * _tmp5
    d1_oo -= 0.0833333333333333 * _tmp5
    _tmp6 = einsum('njicba,cbamji->mn', l3_aabaab, t3_aabaab, optimize=True)
    d1_oo -= 0.0833333333333333 * _tmp6
    d1_oo -= 0.0833333333333333 * _tmp6
    d1_oo -= 0.0833333333333333 * _tmp6
    _tmp7 = einsum('njicba,cbamji->mn', l3_abbabb, t3_abbabb, optimize=True)
    d1_oo -= 0.0833333333333333 * _tmp7
    d1_oo -= 0.0833333333333333 * _tmp7
    d1_oo -= 0.0833333333333333 * _tmp7
    return d1_oo


def d1_vv_aa(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb, d_aa):
    nv, no = t1_aa.shape
    d1_vv = np.zeros((nv, nv))
    _tmp0 = einsum('ie,fi->ef', l1_aa, t1_aa, optimize=True)
    d1_vv += 1 * _tmp0
    _tmp1 = einsum('ijea,faij->ef', l2_aaaa, t2_aaaa, optimize=True)
    d1_vv += 0.5 * _tmp1
    _tmp2 = einsum('ijea,faij->ef', l2_abab, t2_abab, optimize=True)
    d1_vv += 0.5 * _tmp2
    d1_vv += 0.5 * _tmp2
    _tmp3 = einsum('ijkeba,fbaijk->ef', l3_aaaaaa, t3_aaaaaa, optimize=True)
    d1_vv += 0.0833333333333333 * _tmp3
    _tmp4 = einsum('ijkeba,fbaijk->ef', l3_aabaab, t3_aabaab, optimize=True)
    d1_vv += 0.0833333333333333 * _tmp4
    d1_vv += 0.0833333333333333 * _tmp4
    d1_vv += 0.0833333333333333 * _tmp4
    d1_vv += 0.0833333333333333 * _tmp4
    _tmp5 = einsum('ijkeba,fbaijk->ef', l3_abbabb, t3_abbabb, optimize=True)
    d1_vv += 0.0833333333333333 * _tmp5
    d1_vv += 0.0833333333333333 * _tmp4
    d1_vv += 0.0833333333333333 * _tmp4
    d1_vv += 0.0833333333333333 * _tmp5
    d1_vv += 0.0833333333333333 * _tmp5
    return d1_vv


def d1_ov_aa(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb, d_aa):
    nv, no = t1_aa.shape
    d1_ov = np.zeros((no, nv))
    _tmp0 = einsum('em->me', t1_aa)
    d1_ov += 1 * _tmp0
    _tmp1 = einsum('ia,eaim->me', l1_aa, t2_aaaa, optimize=True)
    d1_ov -= 1 * _tmp1
    _tmp2 = einsum('ia,eami->me', l1_bb, t2_abab, optimize=True)
    d1_ov += 1 * _tmp2
    _tmp3 = einsum('ijba,ebaijm->me', l2_aaaa, t3_aaaaaa, optimize=True)
    d1_ov += 0.25 * _tmp3
    _tmp4 = einsum('ijba,ebaimj->me', l2_abab, t3_aabaab, optimize=True)
    d1_ov -= 0.25 * _tmp4
    d1_ov -= 0.25 * _tmp4
    _tmp5 = einsum('jiba,ebamji->me', l2_abab, t3_aabaab, optimize=True)
    d1_ov += 0.25 * _tmp5
    d1_ov += 0.25 * _tmp5
    _tmp6 = einsum('ijba,ebamji->me', l2_bbbb, t3_abbabb, optimize=True)
    d1_ov -= 0.25 * _tmp6
    _tmp7 = einsum('ijba,baim,ej->me', l2_aaaa, t2_aaaa, t1_aa, optimize=True)
    d1_ov -= 0.5 * _tmp7
    _tmp8 = einsum('jiba,bami,ej->me', l2_abab, t2_abab, t1_aa, optimize=True)
    d1_ov -= 0.5 * _tmp8
    d1_ov -= 0.5 * _tmp8
    _tmp9 = einsum('ijba,eaij,bm->me', l2_aaaa, t2_aaaa, t1_aa, optimize=True)
    d1_ov -= 0.5 * _tmp9
    _tmp10 = einsum('ijba,eaij,bm->me', l2_abab, t2_abab, t1_aa, optimize=True)
    d1_ov -= 0.5 * _tmp10
    d1_ov -= 0.5 * _tmp10
    _tmp11 = einsum('ijkcba,cbaijm,ek->me', l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    d1_ov -= 0.0833333333333333 * _tmp11
    _tmp12 = einsum('ikjcba,cbaimj,ek->me', l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    d1_ov -= 0.0833333333333333 * _tmp12
    d1_ov -= 0.0833333333333333 * _tmp12
    d1_ov -= 0.0833333333333333 * _tmp12
    _tmp13 = einsum('kjicba,cbamji,ek->me', l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    d1_ov -= 0.0833333333333333 * _tmp13
    d1_ov -= 0.0833333333333333 * _tmp13
    d1_ov -= 0.0833333333333333 * _tmp13
    _tmp14 = einsum('kjicba,cbamji,ek->me', l3_abbabb, t3_abbabb, t1_aa, optimize=True)
    d1_ov -= 0.0833333333333333 * _tmp14
    d1_ov -= 0.0833333333333333 * _tmp14
    d1_ov -= 0.0833333333333333 * _tmp14
    _tmp15 = einsum('ijkcba,ebaijk,cm->me', l3_aaaaaa, t3_aaaaaa, t1_aa, optimize=True)
    d1_ov -= 0.0833333333333333 * _tmp15
    _tmp16 = einsum('ijkcba,ebaijk,cm->me', l3_aabaab, t3_aabaab, t1_aa, optimize=True)
    d1_ov -= 0.0833333333333333 * _tmp16
    d1_ov -= 0.0833333333333333 * _tmp16
    d1_ov -= 0.0833333333333333 * _tmp16
    d1_ov -= 0.0833333333333333 * _tmp16
    _tmp17 = einsum('ijkcba,ebaijk,cm->me', l3_abbabb, t3_abbabb, t1_aa, optimize=True)
    d1_ov -= 0.0833333333333333 * _tmp17
    d1_ov -= 0.0833333333333333 * _tmp16
    d1_ov -= 0.0833333333333333 * _tmp16
    d1_ov -= 0.0833333333333333 * _tmp17
    d1_ov -= 0.0833333333333333 * _tmp17
    _tmp18 = einsum('ia,am,ei->me', l1_aa, t1_aa, t1_aa, optimize=True)
    d1_ov -= 1 * _tmp18
    _tmp19 = einsum('ijkcba,baim,ecjk->me', l3_aaaaaa, t2_aaaa, t2_aaaa, optimize=True)
    d1_ov -= 0.25 * _tmp19
    _tmp20 = einsum('ijkabc,baim,ecjk->me', l3_aabaab, t2_aaaa, t2_abab, optimize=True)
    d1_ov += 0.25 * _tmp20
    d1_ov += 0.25 * _tmp20
    _tmp21 = einsum('kjicba,bami,ecjk->me', l3_aabaab, t2_abab, t2_aaaa, optimize=True)
    d1_ov -= 0.25 * _tmp21
    d1_ov -= 0.25 * _tmp21
    _tmp22 = einsum('jikbca,bami,ecjk->me', l3_abbabb, t2_abab, t2_abab, optimize=True)
    d1_ov += 0.25 * _tmp22
    _tmp23 = einsum('jikabc,abmi,ecjk->me', l3_abbabb, t2_abab, t2_abab, optimize=True)
    d1_ov -= 0.25 * _tmp23
    _tmp24 = einsum('kjibca,bami,eckj->me', l3_abbabb, t2_abab, t2_abab, optimize=True)
    d1_ov -= 0.25 * _tmp24
    _tmp25 = einsum('kjiabc,abmi,eckj->me', l3_abbabb, t2_abab, t2_abab, optimize=True)
    d1_ov += 0.25 * _tmp25
    return d1_ov


def d1_vo_aa(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, o, v, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb, l3_aaaaaa, l3_aabaab, l3_abbabb, l3_bbbbbb, d_aa):
    nv, no = t1_aa.shape
    d1_vo = np.zeros((nv, no))
    _tmp0 = einsum('me->em', l1_aa)
    d1_vo += 1 * _tmp0
    return d1_vo

