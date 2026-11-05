# GENERATED CODE -- Lambda1 residual of Lambda-CCSDT, machine-generated
# using the P(x,y) antisymmetrizer / index-letter conventions described
# below. Do not edit by hand. Output shape (nv, no) -- see solver.py for why it
# is transposed before use.
import numpy as np

from src.Base.utils.cached_einsum import einsum
from .parallel_accum import pacc


def lambda1_residual(t1, t2, t3, l1, l2, l3, f, g, o, v):
    nv, no = t1.shape
    lambda1_res = np.zeros((nv, no))
    pacc(lambda1_res, ((1.00, einsum('me->em', f[o, v])), (-1.00, einsum('imea,ai->em', g[o, o, v, v], t1)),))
    pacc(lambda1_res, ((-1.00, einsum('mi,ie->em', f[o, o], l1)), (1.00, einsum('ae,ma->em', f[v, v], l1)),))
    pacc(lambda1_res, ((-1.00, einsum('ie,ai,ma->em', f[o, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])), (-1.00, einsum('ma,ai,ie->em', f[o, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.50, einsum('je,baij,imba->em', f[o, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])), (-0.50, einsum('mb,baij,ijea->em', f[o, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((-0.083333333333333, einsum('ke,cbaijk,ijmcba->em', f[o, v], t3, l3, optimize=['einsum_path', (1, 2), (0, 1)])), (-0.083333333333333, einsum('mc,cbaijk,ijkeba->em', f[o, v], t3, l3, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((1.00, einsum('maei,ia->em', g[o, v, v, o], l1)), (0.50, einsum('maij,ijae->em', g[o, v, o, o], l2)),))
    pacc(lambda1_res, ((0.50, einsum('baei,miba->em', g[v, v, v, o], l2)), (1.00, einsum('jmei,aj,ia->em', g[o, o, v, o], t1, l1, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((-1.00, einsum('jmai,aj,ie->em', g[o, o, v, o], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])), (1.00, einsum('maeb,bi,ia->em', g[o, v, v, v], t1, l1, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((-1.00, einsum('iaeb,bi,ma->em', g[o, v, v, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])), (-0.50, einsum('kmij,ak,ijea->em', g[o, o, o, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((1.00, einsum('jbei,aj,miba->em', g[o, v, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])), (1.00, einsum('mabj,bi,ijae->em', g[o, v, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.50, einsum('baec,ci,imba->em', g[v, v, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])), (1.00, einsum('jmeb,baij,ia->em', g[o, o, v, v], t2, l1, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((0.50, einsum('jieb,baji,ma->em', g[o, o, v, v], t2, l1, optimize=['einsum_path', (0, 1), (0, 1)])), (0.50, einsum('jmab,abij,ie->em', g[o, o, v, v], t2, l1, optimize=['einsum_path', (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((0.50, einsum('kmej,baik,ijba->em', g[o, o, v, o], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])), (0.250, einsum('kjei,bakj,miba->em', g[o, o, v, o], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-1.00, einsum('kmbj,baik,ijea->em', g[o, o, v, o], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])), (0.50, einsum('mbec,caij,ijba->em', g[o, v, v, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((-1.00, einsum('jbec,caij,imba->em', g[o, v, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])), (0.250, einsum('mabc,bcij,ijae->em', g[o, v, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((0.250, einsum('lmjk,bail,ijkeba->em', g[o, o, o, o], t2, l3, optimize=['einsum_path', (1, 2), (0, 1)])), (-0.50, einsum('kcej,baik,imjcba->em', g[o, v, v, o], t2, l3, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((-0.50, einsum('mbck,caij,ijkbea->em', g[o, v, v, o], t2, l3, optimize=['einsum_path', (1, 2), (0, 1)])), (0.250, einsum('cbed,daij,ijmcba->em', g[v, v, v, v], t2, l3, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((-0.250, einsum('kmec,cbaijk,ijba->em', g[o, o, v, v], t3, l2, optimize=['einsum_path', (1, 2), (0, 1)])), (-0.250, einsum('kjec,cbaikj,imba->em', g[o, o, v, v], t3, l2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.250, einsum('kmbc,bcaijk,ijea->em', g[o, o, v, v], t3, l2, optimize=['einsum_path', (0, 1), (0, 1)])), (0.083333333333333, einsum('lmek,cbaijl,ijkcba->em', g[o, o, v, o], t3, l3, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((0.083333333333333, einsum('lkej,cbailk,imjcba->em', g[o, o, v, o], t3, l3, optimize=['einsum_path', (1, 2), (0, 1)])), (-0.250, einsum('lmck,cbaijl,ijkeba->em', g[o, o, v, o], t3, l3, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((0.083333333333333, einsum('mced,dbaijk,ijkcba->em', g[o, v, v, v], t3, l3, optimize=['einsum_path', (1, 2), (0, 1)])), (-0.250, einsum('kced,dbaijk,ijmcba->em', g[o, v, v, v], t3, l3, optimize=['einsum_path', (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((0.083333333333333, einsum('mbcd,cdaijk,ijkbea->em', g[o, v, v, v], t3, l3, optimize=['einsum_path', (1, 2), (0, 1)])), (0.50, einsum('kmec,baik,cj,ijba->em', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((0.50, einsum('kmec,caij,bk,ijba->em', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])), (-0.50, einsum('kjec,baik,cj,imba->em', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.250, einsum('kjec,bakj,ci,imba->em', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])), (1.00, einsum('kjec,caik,bj,imba->em', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.50, einsum('kmbc,caij,bk,ijea->em', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])), (1.00, einsum('kmbc,caik,bj,ijea->em', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.250, einsum('kmbc,ak,bcij,ijea->em', g[o, o, v, v], t1, t2, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])), (0.50, einsum('lkej,bail,ck,imjcba->em', g[o, o, v, o], t2, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((0.50, einsum('lmck,bail,cj,ijkeba->em', g[o, o, v, o], t2, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])), (0.50, einsum('lmck,caij,bl,ijkeba->em', g[o, o, v, o], t2, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((0.50, einsum('kced,baik,dj,ijmcba->em', g[o, v, v, v], t2, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])), (0.50, einsum('kced,daij,bk,ijmcba->em', g[o, v, v, v], t2, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((0.50, einsum('mbcd,daij,ck,ijkbea->em', g[o, v, v, v], t2, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])), (0.083333333333333, einsum('lmed,cbaijl,dk,ijkcba->em', g[o, o, v, v], t3, t1, l3, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((0.083333333333333, einsum('lmed,dbaijk,cl,ijkcba->em', g[o, o, v, v], t3, t1, l3, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])), (-0.083333333333333, einsum('lked,cbaijl,dk,ijmcba->em', g[o, o, v, v], t3, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.083333333333333, einsum('lked,cbailk,dj,ijmcba->em', g[o, o, v, v], t3, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])), (0.250, einsum('lked,dbaijl,ck,ijmcba->em', g[o, o, v, v], t3, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.083333333333333, einsum('lmcd,dbaijk,cl,ijkeba->em', g[o, o, v, v], t3, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])), (0.250, einsum('lmcd,dbaijl,ck,ijkeba->em', g[o, o, v, v], t3, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.083333333333333, einsum('lmcd,cdaijk,bl,ijkeba->em', g[o, o, v, v], t3, t1, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])), (1.00, einsum('jmeb,aj,bi,ia->em', g[o, o, v, v], t1, t1, l1, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])),))
    pacc(lambda1_res, ((-1.00, einsum('jieb,aj,bi,ma->em', g[o, o, v, v], t1, t1, l1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])), (-1.00, einsum('jmab,aj,bi,ie->em', g[o, o, v, v], t1, t1, l1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.50, einsum('kjei,ak,bj,miba->em', g[o, o, v, o], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])), (-1.00, einsum('kmbj,ak,bi,ijea->em', g[o, o, v, o], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-1.00, einsum('jbec,aj,ci,imba->em', g[o, v, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])), (-0.50, einsum('mabc,bj,ci,ijae->em', g[o, v, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((0.250, einsum('lmed,bail,dcjk,ijkcba->em', g[o, o, v, v], t2, t2, l3, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])), (-0.250, einsum('lked,bail,dcjk,ijmcba->em', g[o, o, v, v], t2, t2, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((0.1250, einsum('lked,balk,dcij,ijmcba->em', g[o, o, v, v], t2, t2, l3, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])), (-0.250, einsum('lked,dail,cbjk,ijmcba->em', g[o, o, v, v], t2, t2, l3, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((0.1250, einsum('lmcd,bail,cdjk,ijkeba->em', g[o, o, v, v], t2, t2, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])), (-0.250, einsum('lmcd,daij,cbkl,ijkeba->em', g[o, o, v, v], t2, t2, l3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.250, einsum('lmcd,dail,cbjk,ijkeba->em', g[o, o, v, v], t2, t2, l3, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])), (-0.50, einsum('lked,bail,ck,dj,ijmcba->em', g[o, o, v, v], t2, t1, t1, l3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.250, einsum('lked,daij,bl,ck,ijmcba->em', g[o, o, v, v], t2, t1, t1, l3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])), (-0.250, einsum('lmcd,bail,ck,dj,ijkeba->em', g[o, o, v, v], t2, t1, t1, l3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((-0.50, einsum('lmcd,daij,bl,ck,ijkeba->em', g[o, o, v, v], t2, t1, t1, l3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])), (0.50, einsum('kjec,ak,bj,ci,imba->em', g[o, o, v, v], t1, t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])),))
    pacc(lambda1_res, ((0.50, einsum('kmbc,ak,bj,ci,ijea->em', g[o, o, v, v], t1, t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])),))
    return lambda1_res
