"""
Spin-orbital CCSDT amplitude equations (energy, T1/T2/T3 residuals, kernel),
generated with pdaggerq. Adapted from pdaggerq's own full_cc_codes/ccsdt.py
example (Apache-2.0); validated there against NWChem's CCSDT to 9 decimals
for FH/6-31g. See integrals.py for building the fock/g inputs this expects.

Two deviations from pdaggerq's emitted source, both purely about wall time --
the arithmetic is unchanged (verified: max|diff| ~1e-16 on every residual):

1. `einsum` here is cached_einsum, not numpy's. As pdaggerq emits them, the
   two-operand contractions carry NO `optimize=` argument at all, and numpy's
   default is `optimize=False` -- meaning they run in numpy's own C einsum
   kernel and never reach BLAS. That was 60% of triples_residual wall time in
   20 calls, and it is why MKL_NUM_THREADS appeared to do nothing: there was
   almost no BLAS work to thread. Individually those terms are 3-90x faster
   with a real contraction path (e.g. 'adbk,dcij->abcijk': 0.58s -> 0.006s),
   bit-identical.

2. The P(x,y) antisymmetrizer lines accumulate via `pacc` (parallel_accum.py)
   rather than `res += c0 * tmp + c1 * tmp.transpose(...) + ...`. Once (1)
   landed, those lines were 84% of triples_residual -- single-threaded, since
   numpy elementwise ufuncs never use BLAS, and allocating a full rank-6
   temporary per permutation. See parallel_accum.py for the measurements.

Measured together on nv=24/no=8, 10 threads: triples_residual 36.6s -> 5.1s,
and thread scaling (1 -> 10 threads) went from 1.08x to 3.13x.
"""

# allow numpy built with MKL to consume more threads for tensordot -- but only
# if the caller hasn't already pinned the thread count (the test_all_21_* sweep
# workers deliberately run single-threaded). NB this only bites if it runs
# before MKL initializes its own thread pool; when a caller has already
# imported and used numpy, set MKL_NUM_THREADS in the environment (or the
# submit script) instead of relying on this line.
import os
os.environ.setdefault("MKL_NUM_THREADS", "{}".format(max(1, (os.cpu_count() or 2) - 1)))

import numpy as np

from .cached_einsum import einsum
from .fixed_point import iterate, jacobi_step
from .parallel_accum import pacc


def cc_energy(t1, t2, f, g, o, v):
    """<0|e^-T H e^T|0>. t1/t2: spin-orbital amplitudes; f/g: fock/antisymmetrized ERIs; o/v: occ/virt slices."""

    #	  1.0000 f(i,i)
    energy = 1.0 * einsum('ii', f[o, o])

    #	  1.0000 f(i,a)*t1(a,i)
    energy += 1.0 * einsum('ia,ai', f[o, v], t1)

    #	 -0.5000 <j,i||j,i>
    energy += -0.5 * einsum('jiji', g[o, o, o, o])

    #	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy += 0.25 * einsum('jiab,abji', g[o, o, v, v], t2)

    #	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += -0.5 * einsum('jiab,ai,bj', g[o, o, v, v], t1, t1,
                            optimize=['einsum_path', (0, 1), (0, 1)])

    return energy


def singles_residual(t1, t2, t3, f, g, o, v):
    """<0| m* e e^-T H e^T |0>. Same t1/t2/f/g/o/v convention as cc_energy."""
    #	  1.0000 f(a,i)
    singles_res = 1.0 * einsum('ai->ai', f[v, o])
    
    #	 -1.0000 f(j,i)*t1(a,j)
    pacc(singles_res, ((-1.0, einsum('ji,aj->ai', f[o, o], t1)),))
    
    #	  1.0000 f(a,b)*t1(b,i)
    pacc(singles_res, ((1.0, einsum('ab,bi->ai', f[v, v], t1)),))
    
    #	 -1.0000 f(j,b)*t2(b,a,i,j)
    pacc(singles_res, ((-1.0, einsum('jb,baij->ai', f[o, v], t2)),))
    
    #	 -1.0000 f(j,b)*t1(b,i)*t1(a,j)
    pacc(singles_res, ((-1.0, einsum('jb,bi,aj->ai', f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 <j,a||b,i>*t1(b,j)
    pacc(singles_res, ((1.0, einsum('jabi,bj->ai', g[o, v, v, o], t1)),))
    
    #	 -0.5000 <k,j||b,i>*t2(b,a,k,j)
    pacc(singles_res, ((-0.5, einsum('kjbi,bakj->ai', g[o, o, v, o], t2)),))
    
    #	 -0.5000 <j,a||b,c>*t2(b,c,i,j)
    pacc(singles_res, ((-0.5, einsum('jabc,bcij->ai', g[o, v, v, v], t2)),))
    
    #	  0.2500 <k,j||b,c>*t3(b,c,a,i,k,j)
    pacc(singles_res, ((0.25, einsum('kjbc,bcaikj->ai', g[o, o, v, v], t3)),))
    
    #	  1.0000 <k,j||b,i>*t1(b,j)*t1(a,k)
    pacc(singles_res, ((1.0, einsum('kjbi,bj,ak->ai', g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 <j,a||b,c>*t1(b,j)*t1(c,i)
    pacc(singles_res, ((1.0, einsum('jabc,bj,ci->ai', g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 <k,j||b,c>*t1(b,j)*t2(c,a,i,k)
    pacc(singles_res, ((1.0, einsum('kjbc,bj,caik->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  0.5000 <k,j||b,c>*t1(b,i)*t2(c,a,k,j)
    pacc(singles_res, ((0.5, einsum('kjbc,bi,cakj->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])),))
    
    #	  0.5000 <k,j||b,c>*t1(a,j)*t2(b,c,i,k)
    pacc(singles_res, ((0.5, einsum('kjbc,aj,bcik->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])),))
    
    #	  1.0000 <k,j||b,c>*t1(b,j)*t1(c,i)*t1(a,k)
    pacc(singles_res, ((1.0, einsum('kjbc,bj,ci,ak->ai', g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    return singles_res


def doubles_residual(t1, t2, t3, f, g, o, v):
    """<0| m* n* f e e^-T H e^T |0>. Same t1/t2/f/g/o/v convention as cc_energy."""
    #	 -1.0000 P(i,j)f(k,j)*t2(a,b,i,k)
    contracted_intermediate = -1.0 * einsum('kj,abik->abij', f[o, o], t2)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f(a,c)*t2(c,b,i,j)
    contracted_intermediate = 1.0 * einsum('ac,cbij->abij', f[v, v], t2)
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->baij', contracted_intermediate)),))
    
    #	  1.0000 f(k,c)*t3(c,a,b,i,j,k)
    pacc(doubles_res, ((1.0, einsum('kc,cabijk->abij', f[o, v], t3)),))
    
    #	 -1.0000 P(i,j)f(k,c)*t1(c,j)*t2(a,b,i,k)
    contracted_intermediate = -1.0 * einsum('kc,cj,abik->abij', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)f(k,c)*t1(a,k)*t2(c,b,i,j)
    contracted_intermediate = -1.0 * einsum('kc,ak,cbij->abij', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->baij', contracted_intermediate)),))
    
    #	  1.0000 <a,b||i,j>
    pacc(doubles_res, ((1.0, einsum('abij->abij', g[v, v, o, o])),))
    
    #	  1.0000 P(a,b)<k,a||i,j>*t1(b,k)
    contracted_intermediate = 1.0 * einsum('kaij,bk->abij', g[o, v, o, o], t1)
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->baij', contracted_intermediate)),))
    
    #	  1.0000 P(i,j)<a,b||c,j>*t1(c,i)
    contracted_intermediate = 1.0 * einsum('abcj,ci->abij', g[v, v, v, o], t1)
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	  0.5000 <l,k||i,j>*t2(a,b,l,k)
    pacc(doubles_res, ((0.5, einsum('lkij,ablk->abij', g[o, o, o, o], t2)),))
    
    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>*t2(c,b,i,k)
    contracted_intermediate = 1.0 * einsum('kacj,cbik->abij', g[o, v, v, o], t2)
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)), (-1.00000, einsum('abij->baij', contracted_intermediate)), (1.00000, einsum('abij->baji', contracted_intermediate)),))
    
    #	  0.5000 <a,b||c,d>*t2(c,d,i,j)
    pacc(doubles_res, ((0.5, einsum('abcd,cdij->abij', g[v, v, v, v], t2)),))
    
    #	  0.5000 P(i,j)<l,k||c,j>*t3(c,a,b,i,l,k)
    contracted_intermediate = 0.5 * einsum('lkcj,cabilk->abij', g[o, o, v, o], t3)
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	  0.5000 P(a,b)<k,a||c,d>*t3(c,d,b,i,j,k)
    contracted_intermediate = 0.5 * einsum('kacd,cdbijk->abij', g[o, v, v, v], t3)
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->baij', contracted_intermediate)),))
    
    #	 -1.0000 <l,k||i,j>*t1(a,k)*t1(b,l)
    pacc(doubles_res, ((-1.0, einsum('lkij,ak,bl->abij', g[o, o, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>*t1(c,i)*t1(b,k)
    contracted_intermediate = 1.0 * einsum('kacj,ci,bk->abij', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)), (-1.00000, einsum('abij->baij', contracted_intermediate)), (1.00000, einsum('abij->baji', contracted_intermediate)),))
    
    #	 -1.0000 <a,b||c,d>*t1(c,j)*t1(d,i)
    pacc(doubles_res, ((-1.0, einsum('abcd,cj,di->abij', g[v, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 P(i,j)<l,k||c,j>*t1(c,k)*t2(a,b,i,l)
    contracted_intermediate = 1.0 * einsum('lkcj,ck,abil->abij', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	  0.5000 P(i,j)<l,k||c,j>*t1(c,i)*t2(a,b,l,k)
    contracted_intermediate = 0.5 * einsum('lkcj,ci,ablk->abij', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	 -1.0000 P(i,j)*P(a,b)<l,k||c,j>*t1(a,k)*t2(c,b,i,l)
    contracted_intermediate = -1.0 * einsum('lkcj,ak,cbil->abij', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)), (-1.00000, einsum('abij->baij', contracted_intermediate)), (1.00000, einsum('abij->baji', contracted_intermediate)),))
    
    #	  1.0000 P(a,b)<k,a||c,d>*t1(c,k)*t2(d,b,i,j)
    contracted_intermediate = 1.0 * einsum('kacd,ck,dbij->abij', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->baij', contracted_intermediate)),))
    
    #	 -1.0000 P(i,j)*P(a,b)<k,a||c,d>*t1(c,j)*t2(d,b,i,k)
    contracted_intermediate = -1.0 * einsum('kacd,cj,dbik->abij', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)), (-1.00000, einsum('abij->baij', contracted_intermediate)), (1.00000, einsum('abij->baji', contracted_intermediate)),))
    
    #	  0.5000 P(a,b)<k,a||c,d>*t1(b,k)*t2(c,d,i,j)
    contracted_intermediate = 0.5 * einsum('kacd,bk,cdij->abij', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->baij', contracted_intermediate)),))
    
    #	 -1.0000 <l,k||c,d>*t1(c,k)*t3(d,a,b,i,j,l)
    pacc(doubles_res, ((-1.0, einsum('lkcd,ck,dabijl->abij', g[o, o, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -0.5000 P(i,j)<l,k||c,d>*t1(c,j)*t3(d,a,b,i,l,k)
    contracted_intermediate = -0.5 * einsum('lkcd,cj,dabilk->abij', g[o, o, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	 -0.5000 P(a,b)<l,k||c,d>*t1(a,k)*t3(c,d,b,i,j,l)
    contracted_intermediate = -0.5 * einsum('lkcd,ak,cdbijl->abij', g[o, o, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->baij', contracted_intermediate)),))
    
    #	 -0.5000 P(i,j)<l,k||c,d>*t2(c,d,j,k)*t2(a,b,i,l)
    contracted_intermediate = -0.5 * einsum('lkcd,cdjk,abil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	  0.2500 <l,k||c,d>*t2(c,d,i,j)*t2(a,b,l,k)
    pacc(doubles_res, ((0.25, einsum('lkcd,cdij,ablk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -0.5000 <l,k||c,d>*t2(c,a,l,k)*t2(d,b,i,j)
    pacc(doubles_res, ((-0.5, einsum('lkcd,calk,dbij->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 P(i,j)<l,k||c,d>*t2(c,a,j,k)*t2(d,b,i,l)
    contracted_intermediate = 1.0 * einsum('lkcd,cajk,dbil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	 -0.5000 <l,k||c,d>*t2(c,a,i,j)*t2(d,b,l,k)
    pacc(doubles_res, ((-0.5, einsum('lkcd,caij,dblk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])),))
    
    #	 -1.0000 P(i,j)<l,k||c,j>*t1(c,i)*t1(a,k)*t1(b,l)
    contracted_intermediate = -1.0 * einsum('lkcj,ci,ak,bl->abij', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)<k,a||c,d>*t1(c,j)*t1(d,i)*t1(b,k)
    contracted_intermediate = -1.0 * einsum('kacd,cj,di,bk->abij', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->baij', contracted_intermediate)),))
    
    #	  1.0000 P(i,j)<l,k||c,d>*t1(c,k)*t1(d,j)*t2(a,b,i,l)
    contracted_intermediate = 1.0 * einsum('lkcd,ck,dj,abil->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)),))
    
    #	  1.0000 P(a,b)<l,k||c,d>*t1(c,k)*t1(a,l)*t2(d,b,i,j)
    contracted_intermediate = 1.0 * einsum('lkcd,ck,al,dbij->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->baij', contracted_intermediate)),))
    
    #	 -0.5000 <l,k||c,d>*t1(c,j)*t1(d,i)*t2(a,b,l,k)
    pacc(doubles_res, ((-0.5, einsum('lkcd,cj,di,ablk->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	  1.0000 P(i,j)*P(a,b)<l,k||c,d>*t1(c,j)*t1(a,k)*t2(d,b,i,l)
    contracted_intermediate = 1.0 * einsum('lkcd,cj,ak,dbil->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(doubles_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abij->abji', contracted_intermediate)), (-1.00000, einsum('abij->baij', contracted_intermediate)), (1.00000, einsum('abij->baji', contracted_intermediate)),))
    
    #	 -0.5000 <l,k||c,d>*t1(a,k)*t1(b,l)*t2(c,d,i,j)
    pacc(doubles_res, ((-0.5, einsum('lkcd,ak,bl,cdij->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	  1.0000 <l,k||c,d>*t1(c,j)*t1(d,i)*t1(a,k)*t1(b,l)
    pacc(doubles_res, ((1.0, einsum('lkcd,cj,di,ak,bl->abij', g[o, o, v, v], t1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])),))
    return doubles_res

def triples_residual(t1, t2, t3, f, g, o, v):
    """<0| i* j* k* c b a e^-T H e^T |0>. Same t1/t2/f/g/o/v convention as cc_energy."""
    #	 -1.0000 P(j,k)f(l,k)*t3(a,b,c,i,j,l)
    contracted_intermediate = -1.0 * einsum('lk,abcijl->abcijk', f[o, o], t3)
    triples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 f(l,i)*t3(a,b,c,j,k,l)
    pacc(triples_res, ((-1.0, einsum('li,abcjkl->abcijk', f[o, o], t3)),))
    
    #	  1.0000 P(a,b)f(a,d)*t3(d,b,c,i,j,k)
    contracted_intermediate = 1.0 * einsum('ad,dbcijk->abcijk', f[v, v], t3)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 f(c,d)*t3(d,a,b,i,j,k)
    pacc(triples_res, ((1.0, einsum('cd,dabijk->abcijk', f[v, v], t3)),))
    
    #	 -1.0000 P(j,k)f(l,d)*t1(d,k)*t3(a,b,c,i,j,l)
    contracted_intermediate = -1.0 * einsum('ld,dk,abcijl->abcijk', f[o, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -1.0000 f(l,d)*t1(d,i)*t3(a,b,c,j,k,l)
    pacc(triples_res, ((-1.0, einsum('ld,di,abcjkl->abcijk', f[o, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -1.0000 P(a,b)f(l,d)*t1(a,l)*t3(d,b,c,i,j,k)
    contracted_intermediate = -1.0 * einsum('ld,al,dbcijk->abcijk', f[o, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -1.0000 f(l,d)*t1(c,l)*t3(d,a,b,i,j,k)
    pacc(triples_res, ((-1.0, einsum('ld,cl,dabijk->abcijk', f[o, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -1.0000 P(i,j)*P(a,b)f(l,d)*t2(d,a,j,k)*t2(b,c,i,l)
    contracted_intermediate = -1.0 * einsum('ld,dajk,bcil->abcijk', f[o, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)f(l,d)*t2(d,a,i,j)*t2(b,c,k,l)
    contracted_intermediate = -1.0 * einsum('ld,daij,bckl->abcijk', f[o, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -1.0000 P(i,j)f(l,d)*t2(d,c,j,k)*t2(a,b,i,l)
    contracted_intermediate = -1.0 * einsum('ld,dcjk,abil->abcijk', f[o, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	 -1.0000 f(l,d)*t2(d,c,i,j)*t2(a,b,k,l)
    pacc(triples_res, ((-1.0, einsum('ld,dcij,abkl->abcijk', f[o, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -1.0000 P(i,j)*P(a,b)<l,a||j,k>*t2(b,c,i,l)
    contracted_intermediate = -1.0 * einsum('lajk,bcil->abcijk', g[o, v, o, o], t2)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)<l,a||i,j>*t2(b,c,k,l)
    contracted_intermediate = -1.0 * einsum('laij,bckl->abcijk', g[o, v, o, o], t2)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -1.0000 P(i,j)<l,c||j,k>*t2(a,b,i,l)
    contracted_intermediate = -1.0 * einsum('lcjk,abil->abcijk', g[o, v, o, o], t2)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	 -1.0000 <l,c||i,j>*t2(a,b,k,l)
    pacc(triples_res, ((-1.0, einsum('lcij,abkl->abcijk', g[o, v, o, o], t2)),))
    
    #	 -1.0000 P(j,k)*P(b,c)<a,b||d,k>*t2(d,c,i,j)
    contracted_intermediate = -1.0 * einsum('abdk,dcij->abcijk', g[v, v, v, o], t2)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)), (1.00000, einsum('abcijk->acbikj', contracted_intermediate)),))
    
    #	 -1.0000 P(b,c)<a,b||d,i>*t2(d,c,j,k)
    contracted_intermediate = -1.0 * einsum('abdi,dcjk->abcijk', g[v, v, v, o], t2)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)<b,c||d,k>*t2(d,a,i,j)
    contracted_intermediate = -1.0 * einsum('bcdk,daij->abcijk', g[v, v, v, o], t2)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -1.0000 <b,c||d,i>*t2(d,a,j,k)
    pacc(triples_res, ((-1.0, einsum('bcdi,dajk->abcijk', g[v, v, v, o], t2)),))
    
    #	  0.5000 P(i,j)<m,l||j,k>*t3(a,b,c,i,m,l)
    contracted_intermediate = 0.5 * einsum('mljk,abciml->abcijk', g[o, o, o, o], t3)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  0.5000 <m,l||i,j>*t3(a,b,c,k,m,l)
    pacc(triples_res, ((0.5, einsum('mlij,abckml->abcijk', g[o, o, o, o], t3)),))
    
    #	  1.0000 P(j,k)*P(a,b)<l,a||d,k>*t3(d,b,c,i,j,l)
    contracted_intermediate = 1.0 * einsum('ladk,dbcijl->abcijk', g[o, v, v, o], t3)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	  1.0000 P(a,b)<l,a||d,i>*t3(d,b,c,j,k,l)
    contracted_intermediate = 1.0 * einsum('ladi,dbcjkl->abcijk', g[o, v, v, o], t3)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)<l,c||d,k>*t3(d,a,b,i,j,l)
    contracted_intermediate = 1.0 * einsum('lcdk,dabijl->abcijk', g[o, v, v, o], t3)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	  1.0000 <l,c||d,i>*t3(d,a,b,j,k,l)
    pacc(triples_res, ((1.0, einsum('lcdi,dabjkl->abcijk', g[o, v, v, o], t3)),))
    
    #	  0.5000 P(b,c)<a,b||d,e>*t3(d,e,c,i,j,k)
    contracted_intermediate = 0.5 * einsum('abde,decijk->abcijk', g[v, v, v, v], t3)
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	  0.5000 <b,c||d,e>*t3(d,e,a,i,j,k)
    pacc(triples_res, ((0.5, einsum('bcde,deaijk->abcijk', g[v, v, v, v], t3)),))
    
    #	  1.0000 P(i,j)*P(a,b)<m,l||j,k>*t1(a,l)*t2(b,c,i,m)
    contracted_intermediate = 1.0 * einsum('mljk,al,bcim->abcijk', g[o, o, o, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	  1.0000 P(i,j)<m,l||j,k>*t1(c,l)*t2(a,b,i,m)
    contracted_intermediate = 1.0 * einsum('mljk,cl,abim->abcijk', g[o, o, o, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  1.0000 P(a,b)<m,l||i,j>*t1(a,l)*t2(b,c,k,m)
    contracted_intermediate = 1.0 * einsum('mlij,al,bckm->abcijk', g[o, o, o, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 <m,l||i,j>*t1(c,l)*t2(a,b,k,m)
    pacc(triples_res, ((1.0, einsum('mlij,cl,abkm->abcijk', g[o, o, o, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -1.0000 P(i,j)*P(a,b)<l,a||d,k>*t1(d,j)*t2(b,c,i,l)
    contracted_intermediate = -1.0 * einsum('ladk,dj,bcil->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)*P(b,c)<l,a||d,k>*t1(b,l)*t2(d,c,i,j)
    contracted_intermediate = -1.0 * einsum('ladk,bl,dcij->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)), (1.00000, einsum('abcijk->acbikj', contracted_intermediate)),))
    
    #	  1.0000 P(i,k)*P(a,b)<l,a||d,j>*t1(d,k)*t2(b,c,i,l)
    contracted_intermediate = 1.0 * einsum('ladj,dk,bcil->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abckji', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->backji', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,i>*t1(d,k)*t2(b,c,j,l)
    contracted_intermediate = -1.0 * einsum('ladi,dk,bcjl->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	 -1.0000 P(b,c)<l,a||d,i>*t1(b,l)*t2(d,c,j,k)
    contracted_intermediate = -1.0 * einsum('ladi,bl,dcjk->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(a,c)<l,b||d,k>*t1(a,l)*t2(d,c,i,j)
    contracted_intermediate = 1.0 * einsum('lbdk,al,dcij->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->cbaijk', contracted_intermediate)), (1.00000, einsum('abcijk->cbaikj', contracted_intermediate)),))
    
    #	  1.0000 P(a,c)<l,b||d,i>*t1(a,l)*t2(d,c,j,k)
    contracted_intermediate = 1.0 * einsum('lbdi,al,dcjk->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->cbaijk', contracted_intermediate)),))
    
    #	 -1.0000 P(i,j)<l,c||d,k>*t1(d,j)*t2(a,b,i,l)
    contracted_intermediate = -1.0 * einsum('lcdk,dj,abil->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)*P(a,b)<l,c||d,k>*t1(a,l)*t2(d,b,i,j)
    contracted_intermediate = -1.0 * einsum('lcdk,al,dbij->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	  1.0000 P(i,k)<l,c||d,j>*t1(d,k)*t2(a,b,i,l)
    contracted_intermediate = 1.0 * einsum('lcdj,dk,abil->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abckji', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)<l,c||d,i>*t1(d,k)*t2(a,b,j,l)
    contracted_intermediate = -1.0 * einsum('lcdi,dk,abjl->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)<l,c||d,i>*t1(a,l)*t2(d,b,j,k)
    contracted_intermediate = -1.0 * einsum('lcdi,al,dbjk->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(b,c)<a,b||d,e>*t1(d,k)*t2(e,c,i,j)
    contracted_intermediate = 1.0 * einsum('abde,dk,ecij->abcijk', g[v, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)), (1.00000, einsum('abcijk->acbikj', contracted_intermediate)),))
    
    #	  1.0000 P(b,c)<a,b||d,e>*t1(d,i)*t2(e,c,j,k)
    contracted_intermediate = 1.0 * einsum('abde,di,ecjk->abcijk', g[v, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)<b,c||d,e>*t1(d,k)*t2(e,a,i,j)
    contracted_intermediate = 1.0 * einsum('bcde,dk,eaij->abcijk', g[v, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	  1.0000 <b,c||d,e>*t1(d,i)*t2(e,a,j,k)
    pacc(triples_res, ((1.0, einsum('bcde,di,eajk->abcijk', g[v, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 P(j,k)<m,l||d,k>*t1(d,l)*t3(a,b,c,i,j,m)
    contracted_intermediate = 1.0 * einsum('mldk,dl,abcijm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	  0.5000 P(i,j)<m,l||d,k>*t1(d,j)*t3(a,b,c,i,m,l)
    contracted_intermediate = 0.5 * einsum('mldk,dj,abciml->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,k>*t1(a,l)*t3(d,b,c,i,j,m)
    contracted_intermediate = -1.0 * einsum('mldk,al,dbcijm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)<m,l||d,k>*t1(c,l)*t3(d,a,b,i,j,m)
    contracted_intermediate = -1.0 * einsum('mldk,cl,dabijm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -0.5000 P(i,k)<m,l||d,j>*t1(d,k)*t3(a,b,c,i,m,l)
    contracted_intermediate = -0.5 * einsum('mldj,dk,abciml->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abckji', contracted_intermediate)),))
    
    #	  1.0000 <m,l||d,i>*t1(d,l)*t3(a,b,c,j,k,m)
    pacc(triples_res, ((1.0, einsum('mldi,dl,abcjkm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  0.5000 P(j,k)<m,l||d,i>*t1(d,k)*t3(a,b,c,j,m,l)
    contracted_intermediate = 0.5 * einsum('mldi,dk,abcjml->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)<m,l||d,i>*t1(a,l)*t3(d,b,c,j,k,m)
    contracted_intermediate = -1.0 * einsum('mldi,al,dbcjkm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -1.0000 <m,l||d,i>*t1(c,l)*t3(d,a,b,j,k,m)
    pacc(triples_res, ((-1.0, einsum('mldi,cl,dabjkm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 P(a,b)<l,a||d,e>*t1(d,l)*t3(e,b,c,i,j,k)
    contracted_intermediate = 1.0 * einsum('lade,dl,ebcijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,e>*t1(d,k)*t3(e,b,c,i,j,l)
    contracted_intermediate = -1.0 * einsum('lade,dk,ebcijl->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)<l,a||d,e>*t1(d,i)*t3(e,b,c,j,k,l)
    contracted_intermediate = -1.0 * einsum('lade,di,ebcjkl->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  0.5000 P(b,c)<l,a||d,e>*t1(b,l)*t3(d,e,c,i,j,k)
    contracted_intermediate = 0.5 * einsum('lade,bl,decijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	 -0.5000 P(a,c)<l,b||d,e>*t1(a,l)*t3(d,e,c,i,j,k)
    contracted_intermediate = -0.5 * einsum('lbde,al,decijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->cbaijk', contracted_intermediate)),))
    
    #	  1.0000 <l,c||d,e>*t1(d,l)*t3(e,a,b,i,j,k)
    pacc(triples_res, ((1.0, einsum('lcde,dl,eabijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -1.0000 P(j,k)<l,c||d,e>*t1(d,k)*t3(e,a,b,i,j,l)
    contracted_intermediate = -1.0 * einsum('lcde,dk,eabijl->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -1.0000 <l,c||d,e>*t1(d,i)*t3(e,a,b,j,k,l)
    pacc(triples_res, ((-1.0, einsum('lcde,di,eabjkl->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  0.5000 P(a,b)<l,c||d,e>*t1(a,l)*t3(d,e,b,i,j,k)
    contracted_intermediate = 0.5 * einsum('lcde,al,debijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(i,j)*P(a,b)<m,l||d,k>*t2(d,a,j,l)*t2(b,c,i,m)
    contracted_intermediate = 1.0 * einsum('mldk,dajl,bcim->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	 -0.5000 P(j,k)*P(a,b)<m,l||d,k>*t2(d,a,i,j)*t2(b,c,m,l)
    contracted_intermediate = -0.5 * einsum('mldk,daij,bcml->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	  1.0000 P(i,j)<m,l||d,k>*t2(d,c,j,l)*t2(a,b,i,m)
    contracted_intermediate = 1.0 * einsum('mldk,dcjl,abim->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	 -0.5000 P(j,k)<m,l||d,k>*t2(d,c,i,j)*t2(a,b,m,l)
    contracted_intermediate = -0.5 * einsum('mldk,dcij,abml->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -1.0000 P(i,k)*P(a,b)<m,l||d,j>*t2(d,a,k,l)*t2(b,c,i,m)
    contracted_intermediate = -1.0 * einsum('mldj,dakl,bcim->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abckji', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->backji', contracted_intermediate)),))
    
    #	 -1.0000 P(i,k)<m,l||d,j>*t2(d,c,k,l)*t2(a,b,i,m)
    contracted_intermediate = -1.0 * einsum('mldj,dckl,abim->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abckji', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(a,b)<m,l||d,i>*t2(d,a,k,l)*t2(b,c,j,m)
    contracted_intermediate = 1.0 * einsum('mldi,dakl,bcjm->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	 -0.5000 P(a,b)<m,l||d,i>*t2(d,a,j,k)*t2(b,c,m,l)
    contracted_intermediate = -0.5 * einsum('mldi,dajk,bcml->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)<m,l||d,i>*t2(d,c,k,l)*t2(a,b,j,m)
    contracted_intermediate = 1.0 * einsum('mldi,dckl,abjm->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -0.5000 <m,l||d,i>*t2(d,c,j,k)*t2(a,b,m,l)
    pacc(triples_res, ((-0.5, einsum('mldi,dcjk,abml->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])),))
    
    #	 -0.5000 P(i,j)*P(a,b)<l,a||d,e>*t2(d,e,j,k)*t2(b,c,i,l)
    contracted_intermediate = -0.5 * einsum('lade,dejk,bcil->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	 -0.5000 P(a,b)<l,a||d,e>*t2(d,e,i,j)*t2(b,c,k,l)
    contracted_intermediate = -0.5 * einsum('lade,deij,bckl->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(a,b)<l,a||d,e>*t2(d,b,k,l)*t2(e,c,i,j)
    contracted_intermediate = 1.0 * einsum('lade,dbkl,ecij->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	  1.0000 P(a,b)<l,a||d,e>*t2(d,b,i,l)*t2(e,c,j,k)
    contracted_intermediate = 1.0 * einsum('lade,dbil,ecjk->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(i,j)*P(a,b)<l,a||d,e>*t2(d,b,j,k)*t2(e,c,i,l)
    contracted_intermediate = 1.0 * einsum('lade,dbjk,ecil->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	  1.0000 P(a,b)<l,a||d,e>*t2(d,b,i,j)*t2(e,c,k,l)
    contracted_intermediate = 1.0 * einsum('lade,dbij,eckl->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -0.5000 P(i,j)<l,c||d,e>*t2(d,e,j,k)*t2(a,b,i,l)
    contracted_intermediate = -0.5 * einsum('lcde,dejk,abil->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	 -0.5000 <l,c||d,e>*t2(d,e,i,j)*t2(a,b,k,l)
    pacc(triples_res, ((-0.5, einsum('lcde,deij,abkl->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 P(j,k)<l,c||d,e>*t2(d,a,k,l)*t2(e,b,i,j)
    contracted_intermediate = 1.0 * einsum('lcde,dakl,ebij->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	  1.0000 <l,c||d,e>*t2(d,a,i,l)*t2(e,b,j,k)
    pacc(triples_res, ((1.0, einsum('lcde,dail,ebjk->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 P(i,j)<l,c||d,e>*t2(d,a,j,k)*t2(e,b,i,l)
    contracted_intermediate = 1.0 * einsum('lcde,dajk,ebil->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  1.0000 <l,c||d,e>*t2(d,a,i,j)*t2(e,b,k,l)
    pacc(triples_res, ((1.0, einsum('lcde,daij,ebkl->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])),))
    
    #	 -0.5000 P(j,k)<m,l||d,e>*t2(d,e,k,l)*t3(a,b,c,i,j,m)
    contracted_intermediate = -0.5 * einsum('mlde,dekl,abcijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -0.5000 <m,l||d,e>*t2(d,e,i,l)*t3(a,b,c,j,k,m)
    pacc(triples_res, ((-0.5, einsum('mlde,deil,abcjkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  0.2500 P(i,j)<m,l||d,e>*t2(d,e,j,k)*t3(a,b,c,i,m,l)
    contracted_intermediate = 0.25 * einsum('mlde,dejk,abciml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  0.2500 <m,l||d,e>*t2(d,e,i,j)*t3(a,b,c,k,m,l)
    pacc(triples_res, ((0.25, einsum('mlde,deij,abckml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -0.5000 P(a,b)<m,l||d,e>*t2(d,a,m,l)*t3(e,b,c,i,j,k)
    contracted_intermediate = -0.5 * einsum('mlde,daml,ebcijk->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>*t2(d,a,k,l)*t3(e,b,c,i,j,m)
    contracted_intermediate = 1.0 * einsum('mlde,dakl,ebcijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	  1.0000 P(a,b)<m,l||d,e>*t2(d,a,i,l)*t3(e,b,c,j,k,m)
    contracted_intermediate = 1.0 * einsum('mlde,dail,ebcjkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -0.5000 P(i,j)*P(a,b)<m,l||d,e>*t2(d,a,j,k)*t3(e,b,c,i,m,l)
    contracted_intermediate = -0.5 * einsum('mlde,dajk,ebciml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	 -0.5000 P(a,b)<m,l||d,e>*t2(d,a,i,j)*t3(e,b,c,k,m,l)
    contracted_intermediate = -0.5 * einsum('mlde,daij,ebckml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -0.5000 <m,l||d,e>*t2(d,c,m,l)*t3(e,a,b,i,j,k)
    pacc(triples_res, ((-0.5, einsum('mlde,dcml,eabijk->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	  1.0000 P(j,k)<m,l||d,e>*t2(d,c,k,l)*t3(e,a,b,i,j,m)
    contracted_intermediate = 1.0 * einsum('mlde,dckl,eabijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	  1.0000 <m,l||d,e>*t2(d,c,i,l)*t3(e,a,b,j,k,m)
    pacc(triples_res, ((1.0, einsum('mlde,dcil,eabjkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -0.5000 P(i,j)<m,l||d,e>*t2(d,c,j,k)*t3(e,a,b,i,m,l)
    contracted_intermediate = -0.5 * einsum('mlde,dcjk,eabiml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	 -0.5000 <m,l||d,e>*t2(d,c,i,j)*t3(e,a,b,k,m,l)
    pacc(triples_res, ((-0.5, einsum('mlde,dcij,eabkml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])),))
    
    #	  0.2500 P(b,c)<m,l||d,e>*t2(a,b,m,l)*t3(d,e,c,i,j,k)
    contracted_intermediate = 0.25 * einsum('mlde,abml,decijk->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	 -0.5000 P(j,k)*P(b,c)<m,l||d,e>*t2(a,b,k,l)*t3(d,e,c,i,j,m)
    contracted_intermediate = -0.5 * einsum('mlde,abkl,decijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)), (1.00000, einsum('abcijk->acbikj', contracted_intermediate)),))
    
    #	 -0.5000 P(b,c)<m,l||d,e>*t2(a,b,i,l)*t3(d,e,c,j,k,m)
    contracted_intermediate = -0.5 * einsum('mlde,abil,decjkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	  0.2500 <m,l||d,e>*t2(b,c,m,l)*t3(d,e,a,i,j,k)
    pacc(triples_res, ((0.25, einsum('mlde,bcml,deaijk->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])),))
    
    #	 -0.5000 P(j,k)<m,l||d,e>*t2(b,c,k,l)*t3(d,e,a,i,j,m)
    contracted_intermediate = -0.5 * einsum('mlde,bckl,deaijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -0.5000 <m,l||d,e>*t2(b,c,i,l)*t3(d,e,a,j,k,m)
    pacc(triples_res, ((-0.5, einsum('mlde,bcil,deajkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])),))
    
    #	  1.0000 P(i,j)*P(a,b)<m,l||d,k>*t1(d,j)*t1(a,l)*t2(b,c,i,m)
    contracted_intermediate = 1.0 * einsum('mldk,dj,al,bcim->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	  1.0000 P(i,j)<m,l||d,k>*t1(d,j)*t1(c,l)*t2(a,b,i,m)
    contracted_intermediate = 1.0 * einsum('mldk,dj,cl,abim->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(b,c)<m,l||d,k>*t1(a,l)*t1(b,m)*t2(d,c,i,j)
    contracted_intermediate = 1.0 * einsum('mldk,al,bm,dcij->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)), (1.00000, einsum('abcijk->acbikj', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)<m,l||d,k>*t1(b,l)*t1(c,m)*t2(d,a,i,j)
    contracted_intermediate = 1.0 * einsum('mldk,bl,cm,daij->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -1.0000 P(i,k)*P(a,b)<m,l||d,j>*t1(d,k)*t1(a,l)*t2(b,c,i,m)
    contracted_intermediate = -1.0 * einsum('mldj,dk,al,bcim->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abckji', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->backji', contracted_intermediate)),))
    
    #	 -1.0000 P(i,k)<m,l||d,j>*t1(d,k)*t1(c,l)*t2(a,b,i,m)
    contracted_intermediate = -1.0 * einsum('mldj,dk,cl,abim->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abckji', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(a,b)<m,l||d,i>*t1(d,k)*t1(a,l)*t2(b,c,j,m)
    contracted_intermediate = 1.0 * einsum('mldi,dk,al,bcjm->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)<m,l||d,i>*t1(d,k)*t1(c,l)*t2(a,b,j,m)
    contracted_intermediate = 1.0 * einsum('mldi,dk,cl,abjm->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	  1.0000 P(b,c)<m,l||d,i>*t1(a,l)*t1(b,m)*t2(d,c,j,k)
    contracted_intermediate = 1.0 * einsum('mldi,al,bm,dcjk->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	  1.0000 <m,l||d,i>*t1(b,l)*t1(c,m)*t2(d,a,j,k)
    pacc(triples_res, ((1.0, einsum('mldi,bl,cm,dajk->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	  1.0000 P(i,j)*P(a,b)<l,a||d,e>*t1(d,k)*t1(e,j)*t2(b,c,i,l)
    contracted_intermediate = 1.0 * einsum('lade,dk,ej,bcil->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(b,c)<l,a||d,e>*t1(d,k)*t1(b,l)*t2(e,c,i,j)
    contracted_intermediate = 1.0 * einsum('lade,dk,bl,ecij->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)), (1.00000, einsum('abcijk->acbikj', contracted_intermediate)),))
    
    #	  1.0000 P(a,b)<l,a||d,e>*t1(d,j)*t1(e,i)*t2(b,c,k,l)
    contracted_intermediate = 1.0 * einsum('lade,dj,ei,bckl->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(b,c)<l,a||d,e>*t1(d,i)*t1(b,l)*t2(e,c,j,k)
    contracted_intermediate = 1.0 * einsum('lade,di,bl,ecjk->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)*P(a,c)<l,b||d,e>*t1(d,k)*t1(a,l)*t2(e,c,i,j)
    contracted_intermediate = -1.0 * einsum('lbde,dk,al,ecij->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->cbaijk', contracted_intermediate)), (1.00000, einsum('abcijk->cbaikj', contracted_intermediate)),))
    
    #	 -1.0000 P(a,c)<l,b||d,e>*t1(d,i)*t1(a,l)*t2(e,c,j,k)
    contracted_intermediate = -1.0 * einsum('lbde,di,al,ecjk->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->cbaijk', contracted_intermediate)),))
    
    #	  1.0000 P(i,j)<l,c||d,e>*t1(d,k)*t1(e,j)*t2(a,b,i,l)
    contracted_intermediate = 1.0 * einsum('lcde,dk,ej,abil->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(a,b)<l,c||d,e>*t1(d,k)*t1(a,l)*t2(e,b,i,j)
    contracted_intermediate = 1.0 * einsum('lcde,dk,al,ebij->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	  1.0000 <l,c||d,e>*t1(d,j)*t1(e,i)*t2(a,b,k,l)
    pacc(triples_res, ((1.0, einsum('lcde,dj,ei,abkl->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	  1.0000 P(a,b)<l,c||d,e>*t1(d,i)*t1(a,l)*t2(e,b,j,k)
    contracted_intermediate = 1.0 * einsum('lcde,di,al,ebjk->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)<m,l||d,e>*t1(d,l)*t1(e,k)*t3(a,b,c,i,j,m)
    contracted_intermediate = 1.0 * einsum('mlde,dl,ek,abcijm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	  1.0000 <m,l||d,e>*t1(d,l)*t1(e,i)*t3(a,b,c,j,k,m)
    pacc(triples_res, ((1.0, einsum('mlde,dl,ei,abcjkm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	  1.0000 P(a,b)<m,l||d,e>*t1(d,l)*t1(a,m)*t3(e,b,c,i,j,k)
    contracted_intermediate = 1.0 * einsum('mlde,dl,am,ebcijk->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 <m,l||d,e>*t1(d,l)*t1(c,m)*t3(e,a,b,i,j,k)
    pacc(triples_res, ((1.0, einsum('mlde,dl,cm,eabijk->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	 -0.5000 P(i,j)<m,l||d,e>*t1(d,k)*t1(e,j)*t3(a,b,c,i,m,l)
    contracted_intermediate = -0.5 * einsum('mlde,dk,ej,abciml->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>*t1(d,k)*t1(a,l)*t3(e,b,c,i,j,m)
    contracted_intermediate = 1.0 * einsum('mlde,dk,al,ebcijm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	  1.0000 P(j,k)<m,l||d,e>*t1(d,k)*t1(c,l)*t3(e,a,b,i,j,m)
    contracted_intermediate = 1.0 * einsum('mlde,dk,cl,eabijm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -0.5000 <m,l||d,e>*t1(d,j)*t1(e,i)*t3(a,b,c,k,m,l)
    pacc(triples_res, ((-0.5, einsum('mlde,dj,ei,abckml->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	  1.0000 P(a,b)<m,l||d,e>*t1(d,i)*t1(a,l)*t3(e,b,c,j,k,m)
    contracted_intermediate = 1.0 * einsum('mlde,di,al,ebcjkm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 <m,l||d,e>*t1(d,i)*t1(c,l)*t3(e,a,b,j,k,m)
    pacc(triples_res, ((1.0, einsum('mlde,di,cl,eabjkm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	 -0.5000 P(b,c)<m,l||d,e>*t1(a,l)*t1(b,m)*t3(d,e,c,i,j,k)
    contracted_intermediate = -0.5 * einsum('mlde,al,bm,decijk->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	 -0.5000 <m,l||d,e>*t1(b,l)*t1(c,m)*t3(d,e,a,i,j,k)
    pacc(triples_res, ((-0.5, einsum('mlde,bl,cm,deaijk->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	  1.0000 P(i,j)*P(a,b)<m,l||d,e>*t1(d,l)*t2(e,a,j,k)*t2(b,c,i,m)
    contracted_intermediate = 1.0 * einsum('mlde,dl,eajk,bcim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	  1.0000 P(a,b)<m,l||d,e>*t1(d,l)*t2(e,a,i,j)*t2(b,c,k,m)
    contracted_intermediate = 1.0 * einsum('mlde,dl,eaij,bckm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  1.0000 P(i,j)<m,l||d,e>*t1(d,l)*t2(e,c,j,k)*t2(a,b,i,m)
    contracted_intermediate = 1.0 * einsum('mlde,dl,ecjk,abim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  1.0000 <m,l||d,e>*t1(d,l)*t2(e,c,i,j)*t2(a,b,k,m)
    pacc(triples_res, ((1.0, einsum('mlde,dl,ecij,abkm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>*t1(d,k)*t2(e,a,j,l)*t2(b,c,i,m)
    contracted_intermediate = -1.0 * einsum('mlde,dk,eajl,bcim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	  0.5000 P(j,k)*P(a,b)<m,l||d,e>*t1(d,k)*t2(e,a,i,j)*t2(b,c,m,l)
    contracted_intermediate = 0.5 * einsum('mlde,dk,eaij,bcml->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	 -1.0000 P(i,j)<m,l||d,e>*t1(d,k)*t2(e,c,j,l)*t2(a,b,i,m)
    contracted_intermediate = -1.0 * einsum('mlde,dk,ecjl,abim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  0.5000 P(j,k)<m,l||d,e>*t1(d,k)*t2(e,c,i,j)*t2(a,b,m,l)
    contracted_intermediate = 0.5 * einsum('mlde,dk,ecij,abml->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	  1.0000 P(i,k)*P(a,b)<m,l||d,e>*t1(d,j)*t2(e,a,k,l)*t2(b,c,i,m)
    contracted_intermediate = 1.0 * einsum('mlde,dj,eakl,bcim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abckji', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->backji', contracted_intermediate)),))
    
    #	  1.0000 P(i,k)<m,l||d,e>*t1(d,j)*t2(e,c,k,l)*t2(a,b,i,m)
    contracted_intermediate = 1.0 * einsum('mlde,dj,eckl,abim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abckji', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,e>*t1(d,i)*t2(e,a,k,l)*t2(b,c,j,m)
    contracted_intermediate = -1.0 * einsum('mlde,di,eakl,bcjm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	  0.5000 P(a,b)<m,l||d,e>*t1(d,i)*t2(e,a,j,k)*t2(b,c,m,l)
    contracted_intermediate = 0.5 * einsum('mlde,di,eajk,bcml->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)<m,l||d,e>*t1(d,i)*t2(e,c,k,l)*t2(a,b,j,m)
    contracted_intermediate = -1.0 * einsum('mlde,di,eckl,abjm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	  0.5000 <m,l||d,e>*t1(d,i)*t2(e,c,j,k)*t2(a,b,m,l)
    pacc(triples_res, ((0.5, einsum('mlde,di,ecjk,abml->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])),))
    
    #	  0.5000 P(i,j)*P(a,b)<m,l||d,e>*t1(a,l)*t2(d,e,j,k)*t2(b,c,i,m)
    contracted_intermediate = 0.5 * einsum('mlde,al,dejk,bcim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	  0.5000 P(a,b)<m,l||d,e>*t1(a,l)*t2(d,e,i,j)*t2(b,c,k,m)
    contracted_intermediate = 0.5 * einsum('mlde,al,deij,bckm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,e>*t1(a,l)*t2(d,b,k,m)*t2(e,c,i,j)
    contracted_intermediate = -1.0 * einsum('mlde,al,dbkm,ecij->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacikj', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)<m,l||d,e>*t1(a,l)*t2(d,b,i,m)*t2(e,c,j,k)
    contracted_intermediate = -1.0 * einsum('mlde,al,dbim,ecjk->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>*t1(a,l)*t2(d,b,j,k)*t2(e,c,i,m)
    contracted_intermediate = -1.0 * einsum('mlde,al,dbjk,ecim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)<m,l||d,e>*t1(a,l)*t2(d,b,i,j)*t2(e,c,k,m)
    contracted_intermediate = -1.0 * einsum('mlde,al,dbij,eckm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	  0.5000 P(i,j)<m,l||d,e>*t1(c,l)*t2(d,e,j,k)*t2(a,b,i,m)
    contracted_intermediate = 0.5 * einsum('mlde,cl,dejk,abim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	  0.5000 <m,l||d,e>*t1(c,l)*t2(d,e,i,j)*t2(a,b,k,m)
    pacc(triples_res, ((0.5, einsum('mlde,cl,deij,abkm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	 -1.0000 P(j,k)<m,l||d,e>*t1(c,l)*t2(d,a,k,m)*t2(e,b,i,j)
    contracted_intermediate = -1.0 * einsum('mlde,cl,dakm,ebij->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -1.0000 <m,l||d,e>*t1(c,l)*t2(d,a,i,m)*t2(e,b,j,k)
    pacc(triples_res, ((-1.0, einsum('mlde,cl,daim,ebjk->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])),))
    
    #	 -1.0000 P(i,j)<m,l||d,e>*t1(c,l)*t2(d,a,j,k)*t2(e,b,i,m)
    contracted_intermediate = -1.0 * einsum('mlde,cl,dajk,ebim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	 -1.0000 <m,l||d,e>*t1(c,l)*t2(d,a,i,j)*t2(e,b,k,m)
    pacc(triples_res, ((-1.0, einsum('mlde,cl,daij,ebkm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])),))
    
    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>*t1(d,k)*t1(e,j)*t1(a,l)*t2(b,c,i,m)
    contracted_intermediate = -1.0 * einsum('mlde,dk,ej,al,bcim->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)), (1.00000, einsum('abcijk->bacjik', contracted_intermediate)),))
    
    #	 -1.0000 P(i,j)<m,l||d,e>*t1(d,k)*t1(e,j)*t1(c,l)*t2(a,b,i,m)
    contracted_intermediate = -1.0 * einsum('mlde,dk,ej,cl,abim->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcjik', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)*P(b,c)<m,l||d,e>*t1(d,k)*t1(a,l)*t1(b,m)*t2(e,c,i,j)
    contracted_intermediate = -1.0 * einsum('mlde,dk,al,bm,ecij->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)), (1.00000, einsum('abcijk->acbikj', contracted_intermediate)),))
    
    #	 -1.0000 P(j,k)<m,l||d,e>*t1(d,k)*t1(b,l)*t1(c,m)*t2(e,a,i,j)
    contracted_intermediate = -1.0 * einsum('mlde,dk,bl,cm,eaij->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->abcikj', contracted_intermediate)),))
    
    #	 -1.0000 P(a,b)<m,l||d,e>*t1(d,j)*t1(e,i)*t1(a,l)*t2(b,c,k,m)
    contracted_intermediate = -1.0 * einsum('mlde,dj,ei,al,bckm->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->bacijk', contracted_intermediate)),))
    
    #	 -1.0000 <m,l||d,e>*t1(d,j)*t1(e,i)*t1(c,l)*t2(a,b,k,m)
    pacc(triples_res, ((-1.0, einsum('mlde,dj,ei,cl,abkm->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])),))
    
    #	 -1.0000 P(b,c)<m,l||d,e>*t1(d,i)*t1(a,l)*t1(b,m)*t2(e,c,j,k)
    contracted_intermediate = -1.0 * einsum('mlde,di,al,bm,ecjk->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    pacc(triples_res, ((1.00000, contracted_intermediate), (-1.00000, einsum('abcijk->acbijk', contracted_intermediate)),))
    
    #	 -1.0000 <m,l||d,e>*t1(d,i)*t1(b,l)*t1(c,m)*t2(e,a,j,k)
    pacc(triples_res, ((-1.0, einsum('mlde,di,bl,cm,eajk->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])),))
    return triples_res

def kernel(t1, t2, t3, fock, g, o, v, e_ai, e_abij, e_abcijk, hf_energy, max_iter=100,
           stopping_eps=1.0E-8, diis_size=6, diis_start_cycle=2, out=None):
    """Iterate CCSDT amplitude equations to convergence. e_ai/e_abij/e_abcijk are 1/orbital-energy-denominators.

    diis_size: DIIS subspace dimension; 0 (or None) gives the plain
        fixed-point iteration.
    out: optional dict; receives 'converged' and 'n_iter'. Running out of
        iterations warns (fixed_point.CCNotConverged) and returns the last
        iterate; the loop itself is fixed_point.iterate.

    Undamped Jacobi iteration converges acceptably for well-behaved closed-shell
    cases but crawls or stalls on the open-shell radicals this code is aimed at
    (CN/NO/O2/BN/SiC), where it can burn the whole max_iter budget and stop
    unconverged -- the same reason solver.py says "DIIS is not optional here" for
    Lambda. Same scheme as solver.py: extrapolate the concatenated
    (t1, t2, t3) vector, error vector = change in that vector between cycles.

    MEMORY: DIIS keeps 2 * diis_size flat copies of the whole amplitude set,
    and for CCSDT that set is dominated by t3 (nv^3 no^3). At diis_size=6 that
    is ~12 extra copies of t3 on top of the ~6 the residual step already holds
    -- i.e. it TRIPLES the rank-6 footprint. The startup banner prints the
    actual figure so it is not a surprise mid-run.

    Measured trade curve (BeH/cc-pVDZ, converged to 1e-9; every subspace gives
    the same E_cc to ~1e-11, so this is purely cost, not accuracy):

        diis_size   rank-6 arrays   iterations
            0             6             61
            2            10             33
            3            12             26
            4            14             23
            6            18             20
            8            22             20     <- saturated

    So 6 is the right default when memory allows (it is where the benefit
    stops), but 2-3 is the better pick when the solve is memory-bound:
    diis_size=3 buys 2.3x fewer iterations for 2x the rank-6 memory, whereas
    going 3 -> 6 buys only a further 1.3x for another 1.5x memory.
    """
    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    fock_e_abcijk = np.reciprocal(e_abcijk)
    old_energy = cc_energy(t1, t2, fock, g, o, v)

    if diis_size:
        vec_gb = (t1.nbytes + t2.nbytes + t3.nbytes) / 1e9
        print(f"    ==> CCSDT amplitude equations (DIIS, subspace {diis_size}) <==", flush=True)
        print("", flush=True)
        print(f"    DIIS will hold {2 * diis_size} x {vec_gb:.2f} GB = "
              f"{2 * diis_size * vec_gb:.1f} GB of amplitude history", flush=True)
    else:
        print("    ==> CCSDT amplitude equations (no DIIS) <==", flush=True)
        print("", flush=True)
    # flush=True on every line: a CCSDT iteration can take minutes to hours, and
    # under a batch scheduler stdout is a redirected file, i.e. block-buffered at
    # ~8 KB. At ~85 chars per iteration line that is ~96 iterations of silence --
    # a whole run looks hung until it exits. (pyscf's own logger flushes, which
    # is why SCF output appears and then nothing does.)
    print("     Iter               Energy                 |dE|                 |dT|", flush=True)

    def step(ts):
        t1, t2, t3 = ts
        residual_singles = singles_residual(t1, t2, t3, fock, g, o, v)
        residual_doubles = doubles_residual(t1, t2, t3, fock, g, o, v)
        residual_triples = triples_residual(t1, t2, t3, fock, g, o, v)
        res_norm = (np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles)
                    + np.linalg.norm(residual_triples))
        return [jacobi_step(residual_singles, t1, e_ai, fock_e_ai),
                jacobi_step(residual_doubles, t2, e_abij, fock_e_abij),
                jacobi_step(residual_triples, t3, e_abcijk, fock_e_abcijk)], res_norm

    def is_converged(idx, ts, res_norm):
        nonlocal old_energy
        current_energy = cc_energy(ts[0], ts[1], fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(
            idx, current_energy - hf_energy, delta_e, res_norm), flush=True)
        if delta_e < stopping_eps and res_norm < stopping_eps:
            return True
        old_energy = current_energy
        return False

    (t1, t2, t3), converged, n_iter = iterate(
        step, [t1, t2, t3], is_converged, max_iter, diis_size=diis_size,
        diis_start_cycle=diis_start_cycle, what='CCSDT iterations')
    if out is not None:
        out['converged'], out['n_iter'] = converged, n_iter
    return t1, t2, t3
