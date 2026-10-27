# GENERATED CODE -- do not edit by hand.
# Produced by src/CodeGeneration/generate_eom_ccsdt.py.
# See that module's docstring for the equations/conventions.
import numpy as np

from ..cached_einsum import einsum
from ..parallel_accum import pacc


def rho_oo(lam1, lam2, lam3, r1, r2, r3, t1, t2, t3, kd, o, v):
    nv, no = t1.shape
    rho = np.zeros((no, no))
    _tmp0 = einsum('mn,ai,ia->mn', kd[o, o], r1, lam1, optimize=True)
    pacc(rho, ((1, _tmp0),))
    _tmp1 = einsum('am,na->mn', r1, lam1, optimize=True)
    pacc(rho, ((-1, _tmp1),))
    _tmp2 = einsum('mn,abji,jiab->mn', kd[o, o], r2, lam2, optimize=True)
    pacc(rho, ((0.25, _tmp2),))
    _tmp3 = einsum('abim,inab->mn', r2, lam2, optimize=True)
    pacc(rho, ((-0.5, _tmp3),))
    _tmp4 = einsum('mn,abckji,kjiabc->mn', kd[o, o], r3, lam3, optimize=True)
    pacc(rho, ((0.0277777777777778, _tmp4),))
    _tmp5 = einsum('abcjim,jinabc->mn', r3, lam3, optimize=True)
    pacc(rho, ((-0.0833333333333333, _tmp5),))
    _tmp6 = einsum('bi,am,inab->mn', r1, t1, lam2, optimize=True)
    pacc(rho, ((1, _tmp6),))
    _tmp7 = einsum('bcji,am,jinabc->mn', r2, t1, lam3, optimize=True)
    pacc(rho, ((-0.25, _tmp7),))
    _tmp8 = einsum('baim,jinbac,cj->mn', t2, lam3, r1, optimize=True)
    pacc(rho, ((-0.5, _tmp8),))
    return rho


def rho_vv(lam1, lam2, lam3, r1, r2, r3, t1, t2, t3, kd, o, v):
    nv, no = t1.shape
    rho = np.zeros((nv, nv))
    _tmp0 = einsum('fi,ie->ef', r1, lam1, optimize=True)
    pacc(rho, ((1, _tmp0),))
    _tmp1 = einsum('faji,jiea->ef', r2, lam2, optimize=True)
    pacc(rho, ((0.5, _tmp1),))
    _tmp2 = einsum('fabkji,kjieab->ef', r3, lam3, optimize=True)
    pacc(rho, ((0.0833333333333333, _tmp2),))
    _tmp3 = einsum('aj,fi,jiea->ef', r1, t1, lam2, optimize=True)
    pacc(rho, ((-1, _tmp3),))
    _tmp4 = einsum('abkj,fi,kjieab->ef', r2, t1, lam3, optimize=True)
    pacc(rho, ((0.25, _tmp4),))
    _tmp5 = einsum('bk,faij,kijeab->ef', r1, t2, lam3, optimize=True)
    pacc(rho, ((0.5, _tmp5),))
    return rho


def rho_ov(lam1, lam2, lam3, r1, r2, r3, t1, t2, t3, kd, o, v):
    nv, no = t1.shape
    rho = np.zeros((no, nv))
    _tmp0 = einsum('em->me', r1)
    pacc(rho, ((1, _tmp0),))
    _tmp1 = einsum('eaim,ia->me', r2, lam1, optimize=True)
    pacc(rho, ((-1, _tmp1),))
    _tmp2 = einsum('eabjim,jiab->me', r3, lam2, optimize=True)
    pacc(rho, ((0.25, _tmp2),))
    _tmp3 = einsum('ia,ai,em->me', lam1, r1, t1, optimize=True)
    pacc(rho, ((1, _tmp3),))
    _tmp4 = einsum('ia,am,ei->me', lam1, r1, t1, optimize=True)
    pacc(rho, ((-1, _tmp4),))
    _tmp5 = einsum('ia,ei,am->me', lam1, r1, t1, optimize=True)
    pacc(rho, ((-1, _tmp5),))
    _tmp6 = einsum('abji,em,jiab->me', r2, t1, lam2, optimize=True)
    pacc(rho, ((0.25, _tmp6),))
    _tmp7 = einsum('ei,jiab,abjm->me', t1, lam2, r2, optimize=True)
    pacc(rho, ((-0.5, _tmp7),))
    _tmp8 = einsum('ebji,am,jiab->me', r2, t1, lam2, optimize=True)
    pacc(rho, ((-0.5, _tmp8),))
    _tmp9 = einsum('em,kjiabc,abckji->me', t1, lam3, r3, optimize=True)
    pacc(rho, ((0.0277777777777778, _tmp9),))
    _tmp10 = einsum('abckjm,ei,kjiabc->me', r3, t1, lam3, optimize=True)
    pacc(rho, ((-0.0833333333333333, _tmp10),))
    _tmp11 = einsum('am,kjiabc,ebckji->me', t1, lam3, r3, optimize=True)
    pacc(rho, ((-0.0833333333333333, _tmp11),))
    _tmp12 = einsum('eaim,jiab,bj->me', t2, lam2, r1, optimize=True)
    pacc(rho, ((1, _tmp12),))
    _tmp13 = einsum('bm,eaij,ijab->me', r1, t2, lam2, optimize=True)
    pacc(rho, ((0.5, _tmp13),))
    _tmp14 = einsum('ej,baim,jiba->me', r1, t2, lam2, optimize=True)
    pacc(rho, ((0.5, _tmp14),))
    _tmp15 = einsum('eaim,kjiabc,bckj->me', t2, lam3, r2, optimize=True)
    pacc(rho, ((-0.25, _tmp15),))
    _tmp16 = einsum('eaij,kijabc,bckm->me', t2, lam3, r2, optimize=True)
    pacc(rho, ((-0.25, _tmp16),))
    _tmp17 = einsum('eckj,baim,kjibac->me', r2, t2, lam3, optimize=True)
    pacc(rho, ((-0.25, _tmp17),))
    _tmp18 = einsum('ck,ebaijm,kijbac->me', r1, t3, lam3, optimize=True)
    pacc(rho, ((0.25, _tmp18),))
    _tmp19 = einsum('ebaijk,ijkbac,cm->me', t3, lam3, r1, optimize=True)
    pacc(rho, ((-0.0833333333333333, _tmp19),))
    _tmp20 = einsum('cbaijm,kijcba,ek->me', t3, lam3, r1, optimize=True)
    pacc(rho, ((-0.0833333333333333, _tmp20),))
    _tmp21 = einsum('ck,baim,ej,kijbac->me', r1, t2, t1, lam3, optimize=True)
    pacc(rho, ((-0.5, _tmp21),))
    _tmp22 = einsum('kijbac,ck,eaij,bm->me', lam3, r1, t2, t1, optimize=True)
    pacc(rho, ((-0.5, _tmp22),))
    _tmp23 = einsum('jiab,bj,am,ei->me', lam2, r1, t1, t1, optimize=True)
    pacc(rho, ((1, _tmp23),))
    _tmp24 = einsum('kjiabc,bckj,am,ei->me', lam3, r2, t1, t1, optimize=True)
    pacc(rho, ((-0.25, _tmp24),))
    return rho


def rho_vo(lam1, lam2, lam3, r1, r2, r3, t1, t2, t3, kd, o, v):
    nv, no = t1.shape
    rho = np.zeros((nv, no))
    _tmp0 = einsum('ai,imea->em', r1, lam2, optimize=True)
    pacc(rho, ((-1, _tmp0),))
    _tmp1 = einsum('abji,jimeab->em', r2, lam3, optimize=True)
    pacc(rho, ((0.25, _tmp1),))
    return rho


def rho_star_oo(l1, l2, l3, t1, t2, t3, kd, o, v):
    nv, no = t1.shape
    rho = np.zeros((no, no))
    _tmp0 = einsum('am,na->mn', t1, l1, optimize=True)
    pacc(rho, ((-1, _tmp0),))
    _tmp1 = einsum('baim,inba->mn', t2, l2, optimize=True)
    pacc(rho, ((-0.5, _tmp1),))
    _tmp2 = einsum('cbaijm,ijncba->mn', t3, l3, optimize=True)
    pacc(rho, ((-0.0833333333333333, _tmp2),))
    return rho


def rho_star_vv(l1, l2, l3, t1, t2, t3, kd, o, v):
    nv, no = t1.shape
    rho = np.zeros((nv, nv))
    _tmp0 = einsum('fi,ie->ef', t1, l1, optimize=True)
    pacc(rho, ((1, _tmp0),))
    _tmp1 = einsum('faij,ijea->ef', t2, l2, optimize=True)
    pacc(rho, ((0.5, _tmp1),))
    _tmp2 = einsum('fbaijk,ijkeba->ef', t3, l3, optimize=True)
    pacc(rho, ((0.0833333333333333, _tmp2),))
    return rho


def rho_star_ov(l1, l2, l3, t1, t2, t3, kd, o, v):
    nv, no = t1.shape
    rho = np.zeros((no, nv))
    _tmp0 = einsum('eaim,ia->me', t2, l1, optimize=True)
    pacc(rho, ((-1, _tmp0),))
    _tmp1 = einsum('ebaijm,ijba->me', t3, l2, optimize=True)
    pacc(rho, ((0.25, _tmp1),))
    _tmp2 = einsum('baim,ej,ijba->me', t2, t1, l2, optimize=True)
    pacc(rho, ((-0.5, _tmp2),))
    _tmp3 = einsum('eaij,bm,ijba->me', t2, t1, l2, optimize=True)
    pacc(rho, ((-0.5, _tmp3),))
    _tmp4 = einsum('cbaijm,ek,ijkcba->me', t3, t1, l3, optimize=True)
    pacc(rho, ((-0.0833333333333333, _tmp4),))
    _tmp5 = einsum('ebaijk,cm,ijkcba->me', t3, t1, l3, optimize=True)
    pacc(rho, ((-0.0833333333333333, _tmp5),))
    _tmp6 = einsum('am,ei,ia->me', t1, t1, l1, optimize=True)
    pacc(rho, ((-1, _tmp6),))
    _tmp7 = einsum('baim,ecjk,ijkcba->me', t2, t2, l3, optimize=True)
    pacc(rho, ((-0.25, _tmp7),))
    return rho


def rho_star_vo(l1, l2, l3, t1, t2, t3, kd, o, v):
    nv, no = t1.shape
    rho = np.zeros((nv, no))
    _tmp0 = einsum('me->em', l1)
    pacc(rho, ((1, _tmp0),))
    return rho

