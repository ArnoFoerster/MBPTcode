"""Separable RI (RI-RS) of Duchemin and Blase -- a THC factorization whose
CONSTRUCTION is O(N^3).

    (mu nu | lambda sigma) ~= sum_{kk'} X_{mu k} X_{nu k} Z_{kk'} X_{lambda k'} X_{sigma k'}
    X_{mu k} = chi_mu(r_k)          plain collocation, no weight factor
    Z        = M^T V M              V = (beta|gamma), the aux Coulomb metric

Same factorization form as any other separable/ISDF scheme. What differs, and
the entire point, is how M is obtained.

HOW FINE A GRID YOU NEED DEPENDS ON THE OBSERVABLE, NOT JUST THE SYSTEM
----------------------------------------------------------------------
The fit residual is not uniform across orbital blocks -- the virtual-virtual
block is about 100x worse than occupied-occupied -- and whether that matters
depends entirely on how the observable contracts it. Measured on benzene/cc-pVDZ
with IDENTICAL factors and the same vv residual (1.9e-1):

    RPA   kernel never contracts the vv block              1.9 meV
    TDHF  adds only the bare-exchange vv contraction      65.0 meV

Same object, same residual, 34x apart. The rule that follows:

  * TRACE / INTEGRAL observables are safe at a looser grid. A GW self-energy
    takes G's virtual branch as a thermally weighted SUM over all virtuals,
    traced against W and integrated over tau, so vv fit errors enter
    sign-averaged and largely cancel. GW quasiparticle energies come out within
    0.6 meV of Casida through five acenes at the published cc-pVTZ grids.
  * POINTWISE KERNEL observables are not. A BSE exchange term takes specific
    (ij|W|ab) elements, coherently weighted by the exciton vector, with no
    cancellation. Same grid, ~5 meV at cc-pVTZ and 74.6 meV at the cc-pVDZ
    fallback grid.

So "is the grid converged" is the wrong question; ask which contraction the
quantity performs. Anything with a pointwise kernel -- BSE, and by the same
argument a dynamical kernel or an exciton analysis -- needs the tighter grid.

WHY THIS IS CUBIC AND LS-THC IS NOT
-----------------------------------
The least-squares THC route fits Z by contracting the co-density against
three-centre integrals, `einsum('ijq,gi,gj->qg')`, which is O(N^4) and was the
dominant cost once the polarizability went cubic. Duchemin and Blase instead
ask M to reproduce the RI-V FITTING
COEFFICIENTS of a set of test co-densities [JCP 150, 174120 (2019), eq 6]:

    argmin_M  sum_{rho, beta} ( F^RS_beta(rho) - F^V_beta(rho) )^2
    [D]_{k rho} = rho(r_k)                 test co-density sampled on the grid
    [F]_{beta rho} = F^V_beta(rho)         its RI-V fitting coefficients

which in Frobenius norm is `argmin_M ||M D - F||` with the closed-form estimator
(their eqs 8-9, with row balancing and Tikhonov regularization for stability)

    M = F Dt^T (Dt Dt^T + eps I)^-1 d ,    Dt = d D,  d = diag(1/sqrt(diag(D D^T)))

Only matrix products and one inversion: with the number of test co-densities
linear in system size, every step is O(N^3). eps = 4e-7 is their value for
double precision.

The other half of their scheme is that {r_k} is optimized ONCE PER ELEMENT
offline (eq 10, over Lebedev sub-shells replicated at optimized radii) and a
molecular grid is just the superposition of atomic ones -- so no per-molecule
grid search happens at all. `optimize_atomic_radii` here does the offline part.

Reported behaviour to check against: ~3x the auxiliary basis size (320 points
per C/N/O, 180 per H at cc-pVTZ/cc-pVTZ-RI), meV agreement with RI-V, empirical
exponent 3.07, crossover with quartic RI-V at ~350 electrons.
"""
import hashlib
import json
import os
import time

import numpy as np
import scipy.linalg
from scipy.optimize import minimize, basinhopping
from pyscf import df, gto
from pyscf.dft import gen_grid

#: Pair-screening threshold: a test co-density is dropped when its peak
#: amplitude anywhere on the interpolation grid falls below this times the
#: global maximum. It is what makes the pair count linear in system size.
#:
#: NOT a Schwarz bound, and it does not read as an accuracy target. A screened
#: pair is not set to zero in the result -- the factorization is dense in the
#: pair index, so it still returns a value there, just one no equation
#: constrained. So the threshold is a statement about which CONSTRAINTS are
#: redundant, and the failure is a cliff rather than a slope. Measured on
#: benzene and naphthalene at cc-pVTZ, fit error relative to 1e-10:
#:
#:     1e-06   1.00x        76% / 58% of columns kept
#:     1e-05   1.37x/1.97x  65% / 47%
#:     1e-04    102x/136x   45% / 29%
#:
#: 1e-06 is free -- identical to four figures, and the BSE roots move under
#: 0.03 meV at cc-pVDZ against the ~5 meV the grid itself contributes. There is
#: nothing to harvest past it, so do not tune this looking for more.
DEFAULT_PAIR_TOL = 1e-6

#: Their eq 9 regularization, "a reasonable parameter for double precision".
#: Delesma, Golze and Rinke (separable-RI accuracy in the numeric-atomic-orbital
#: framework, preprint, 2023), reimplementing this in FHI-aims, report that "the
#: inversion is numerically stable without L2 regularization" -- so pass 0.0 if
#: the balanced Gram matrix is well conditioned for your grid, and compare.
DEFAULT_REGULARIZATION = 4e-7

#: Emphasis on the low multipoles of the test co-densities (their Sec. II E).
ANGULAR_WEIGHTS = {0: 4.0, 1: 2.0}


def auxmol_key(auxmol):
    """A hashable identity for an auxiliary basis, for caching on its CONTENT.

    Never `id(auxmol)`: a fresh object's id is one python is free to reuse once
    the previous one is collected, and a cache keyed that way hands back
    another basis's result -- intermittently, and looking perfectly reasonable.
    """
    return (auxmol.nbas, auxmol.nao,
            auxmol._bas.tobytes(), auxmol._env.tobytes())


def lebedev_subshells():
    """A1, A2, A3, B1 as unit vectors, from the nesting L3 = A1,
    L5 = A1+A2, L7 = A1+A2+A3, L11 = A1+A2+A3+B1 (their Sec. II D).

    Verified against pyscf's tables: 6, 14, 26, 50 points, so the shells hold
    6, 8, 12 and 24 directions.
    """
    grids = {}
    for order in (3, 5, 7, 11):
        g = gen_grid.MakeAngularGrid(gen_grid.LEBEDEV_ORDER[order])
        grids[order] = g[:, :3]

    def _new(bigger, smaller):
        keep = [i for i, p in enumerate(bigger)
                if not np.any(np.all(np.abs(smaller - p) < 1e-10, axis=1))]
        return bigger[keep]

    a1 = grids[3]
    a2 = _new(grids[5], a1)
    a3 = _new(grids[7], grids[5])
    b1 = _new(grids[11], grids[7])
    return {'A1': a1, 'A2': a2, 'A3': a3, 'B1': b1}


_SHELLS = None


def atomic_points(radii, centre=(0.0, 0.0, 0.0), origin=False):
    """Grid for one atom: each Lebedev sub-shell replicated at its own radii.

    radii : {'A1': [r, ...], 'A2': [...], 'A3': [...], 'B1': [...]}
    origin : include the nucleus itself as a point. Their published tables
        (SI Tables S8-S11) all start with a bare `0.0 0.0 0.0` entry, so the
        nuclear cusp gets its own sample; leaving it out costs accuracy where
        the co-densities are largest.

    The shell sizes and the radii are the only optimization variables, exactly
    as in their Sec. II D.
    """
    global _SHELLS
    if _SHELLS is None:
        _SHELLS = lebedev_subshells()
    out = [np.zeros((1, 3))] if origin else []
    for name, rs in radii.items():
        for r in np.atleast_1d(rs):
            out.append(_SHELLS[name] * float(r))
    return np.vstack(out) + np.asarray(centre) if out else np.zeros((0, 3))


def published_grids():
    """The optimized atomic grids of Duchemin & Blase, JCP 150, 174120 (2019),
    supporting information Tables S8-S11.

    Optimized for **cc-pVTZ / cc-pVTZ-RI**; H, C, N, O only. Sizes 167 (H), 307 (C), 311 (N),
    307 (O), matching the paper's quoted "320 points for each C, N and O atom
    and 180 points for the H atom".

    Using them with a smaller primary basis is over-converged rather than wrong;
    using them for an element not listed is not possible, so
    `optimize_atomic_radii` remains the fallback there.

    Returns {element: (radii dict, include_origin)}.
    """
    path = os.path.join(os.path.dirname(__file__), 'data',
                        'duchemin_blase_ccpvtz_grids.json')
    with open(path) as fh:
        raw = json.load(fh)
    return {el: ({k: np.array(v) for k, v in d['radii'].items()}, d['origin'])
            for el, d in raw.items()}


def _ao_l_labels(mol):
    """Angular momentum of every AO, for the s/p emphasis weights."""
    l = []
    for ib in range(mol.nbas):
        li = mol.bas_angular(ib)
        ndeg = 2 * li + 1
        l += [li] * (ndeg * mol.bas_nctr(ib))
    return np.array(l)


def _screening_reference(ao, second, w):
    """Global scale for the pair-screening threshold, computed once.

    `col_max.max()` inside the block loop is the largest pair density IN THAT
    BLOCK, so the same pair is kept or dropped depending on what it was batched
    with: a block of diffuse AOs sets a low bar, one holding a core function
    sets a high one. Invisible at pair_tol=1e-10, where nothing is screened
    either way (M agrees to 3e-9 across block sizes), and 1e-4 relative at
    1e-6 -- which would make the fit depend on `block_memory_gb`, a MEMORY knob,
    and quietly break every comparison that assumes blocking is neutral.

    max_k |chi_mu chi_j| w_j <= (max_k|chi_mu|)(w_j max_k|chi_j|), and the bound
    is tight for the pair that sets the scale, whose two maxima sit at the same
    point. One O(n_k nao) reduction over an array already in hand.
    """
    s_ao = np.abs(ao).max(axis=0)
    return float(s_ao.max() * (w * s_ao[second]).max())


def build_D_F(mol, auxmol, coords, l_max_second=2, pair_tol=DEFAULT_PAIR_TOL,
              block_memory_gb=4.0):
    """The two matrices of their eq 7.

    D[k, rho] : test co-density rho evaluated at r_k
    F[beta, rho] : its RI-V fitting coefficients, sum_gamma [V^-1]_{beta gamma} (gamma|rho)

    Test set {rho} = ({alpha} x {alpha'}_{l<=2}) U {beta} (their eq 11), with the
    s/p emphasis applied to the second AO index.

    PAIR SCREENING is what makes the whole scheme cubic. Duchemin & Blase: "due
    to the localization properties of the atomic orbitals, the number of atomic
    orbital products scales linearly with system size". Delesma et al. implement
    it as "only include pairs ij where the atomic orbitals have a significant
    overlap". Screening on the pair density the grid actually samples,
    max_k |chi_mu(r_k) chi_nu(r_k)|, is exact to the tolerance.

    BLOCKED over the first AO index, because the unscreened intermediates do not
    fit. At dodecacene/cc-pVTZ the full D_ao is n_k x n_ao x n_second = 425 GB
    and the three-centre array is 146 GB, against a 252 GB node; even hexacene
    needs 63 + 22 GB, which is most of the 142 GB peak that run showed. Blocking
    means only one block of each exists at a time, and screening is applied
    per block so the surviving columns are all that accumulate. Column ORDER is
    preserved -- blocks are processed in ascending mu -- so D and F stay aligned.

    block_memory_gb caps the per-block working set, and it is BOTH a memory and
    a speed knob -- the second half of that was got wrong once and is worth
    stating precisely. The total INTEGRAL work does not depend on how the index
    is cut up, but the number of `aux_e2` CALLS does, and each call rebuilds a
    shell-pair list over mol.nbas x auxmol.nbas. That setup is invisible on a
    small molecule and dominant on a large one:

        naphthalene/cc-pVTZ   nbas  138, aux  310   one call vs 138: 3.26 vs 2.99 s
        chl dimer/cc-pVTZ     nbas 1362, aux 3072   4 GB -> 2 AOs/block, 2034
                                                    calls, 8.5e9 pair-setups

    The measurement above was taken at nbas=138 and generalized to "block size
    does not affect speed", which is false by an order of magnitude at nbas=1362
    -- a production run sat in this loop for twenty minutes because of it. Size
    the budget from the node, not from the default.
    Every term is a sum over the first AO index -- the three-centre integrals,
    the LU solve (2 naux^2 per kept column), the F D^T product -- so splitting
    the index moves that work between calls without creating any. What it DOES
    create is one shell-pair setup per call, which is what the naphthalene
    measurement was too small to see.

    It does not change the answer. The screening keeps ~98% of columns at
    pair_tol=1e-10 at every block size -- the tolerance is far too tight to
    care that `col_max.max()` is a per-block reference -- and M agrees to
    3e-9 relative between a single block and 2-AO blocks.

    So: lower it when peak memory is the constraint, and do not raise it
    expecting speed.
    """
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    nk = len(coords)
    ao = mol.eval_gto('GTOval_sph', coords)              # (nk, nao)
    aux_on_grid = auxmol.eval_gto('GTOval_sph', coords)  # (nk, naux)

    l_ao = _ao_l_labels(mol)
    second = np.where(l_ao <= l_max_second)[0]
    w = np.array([ANGULAR_WEIGHTS.get(l_ao[j], 1.0) for j in second])
    n2 = len(second)

    V = auxmol.intor('int2c2e', aosym='s1')
    lu = scipy.linalg.lu_factor(V)
    screen_ref = _screening_reference(ao, second, w)

    # One block of mu costs nk*|mu|*n2 (D) + |mu|*nao*naux (three-centre).
    per_mu = (nk * n2 + nao * naux) * 8
    ao_loc = mol.ao_loc_nr()
    max_mu = max(1, int(block_memory_gb * 1e9 / max(per_mu, 1)))

    blocks, sh0 = [], 0
    for sh in range(1, mol.nbas + 1):
        if ao_loc[sh] - ao_loc[sh0] >= max_mu or sh == mol.nbas:
            blocks.append((sh0, sh))
            sh0 = sh
        if sh0 >= mol.nbas:
            break

    D_parts, F_parts = [], []
    for sh0, sh1 in blocks:
        a0, a1 = ao_loc[sh0], ao_loc[sh1]
        D_blk = (ao[:, a0:a1, None] * ao[:, None, second]).reshape(nk, -1)
        D_blk *= np.tile(w, a1 - a0)[None, :]
        # Column maxima without materializing |D_blk|. The obvious form calls
        # np.abs(D_blk) TWICE, each a full n_k x n_rho temporary, and that
        # screening line measured 19.6% of the whole factorization. Two
        # reductions over the existing array allocate nothing, and the global
        # max is just the max of the column maxima.
        col_max = np.maximum(D_blk.max(axis=0), -D_blk.min(axis=0))
        keep = col_max > pair_tol * screen_ref
        if not keep.any():
            continue
        e3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1',
                               shls_slice=(sh0, sh1, 0, mol.nbas, 0, auxmol.nbas))
        e3c = e3c.reshape(a1 - a0, nao, naux)[:, second, :].reshape(-1, naux)
        F_blk = scipy.linalg.lu_solve(lu, e3c[keep].T)
        F_blk *= np.tile(w, a1 - a0)[keep][None, :]
        D_parts.append(D_blk[:, keep])
        F_parts.append(F_blk)
        del D_blk, e3c, F_blk

    # auxiliary-function block: F^V_beta(gamma) = delta, since (V^-1 V) = I
    D = np.hstack(D_parts + [aux_on_grid])
    F = np.hstack(F_parts + [np.eye(naux)])
    return D, F


def fit_M_streaming(mol, auxmol, coords, l_max_second=2,
                    pair_tol=DEFAULT_PAIR_TOL,
                    regularization=DEFAULT_REGULARIZATION, block_memory_gb=4.0,
                    progress=False):
    """M without ever holding D or F.

    `build_D_F` + `fit_M` is the readable form and stays the reference, but it
    materializes D as n_k x n_rho. Even screened and blocked that is the largest
    array in the whole method -- roughly 29 GB at hexacene and over 100 GB at
    dodecacene, on a 252 GB node.

    It is also unnecessary. Everything fit_M does contracts over rho:

        S  = D D^T        (n_k x n_k)
        FD = F D^T        (n_aux x n_k)
        row norms of D are diag(S), so the balancing comes free

    and both are sums over blocks of rho. So the blocks can be accumulated and
    discarded, leaving a peak of one block plus S and FD -- at dodecacene about
    3.2 + 0.8 GB instead of 100+.

    Returns M identical to fit_M(*build_D_F(...)) up to floating-point summation
    order.
    """
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    nk = len(coords)
    ao = mol.eval_gto('GTOval_sph', coords)
    aux_on_grid = auxmol.eval_gto('GTOval_sph', coords)

    l_ao = _ao_l_labels(mol)
    second = np.where(l_ao <= l_max_second)[0]
    w = np.array([ANGULAR_WEIGHTS.get(l_ao[j], 1.0) for j in second])
    n2 = len(second)

    def _say(msg):
        # This factorization is minutes to hours at production sizes and had NO
        # output at all: a run sitting in it looked identical to a hung one, and
        # that is how a whole afternoon gets spent on `py-spy`.
        if progress:
            print(f'[isdf {time.strftime("%H:%M:%S")}] {msg}', flush=True)

    _say(f'fit start: nk={nk} nao={nao} naux={naux} n2={n2} '
         f'block_memory_gb={block_memory_gb} pair_tol={pair_tol:.0e} '
         f'l_max_second={l_max_second} regularization={regularization:.0e}')

    V = auxmol.intor('int2c2e', aosym='s1')
    lu = scipy.linalg.lu_factor(V)
    screen_ref = _screening_reference(ao, second, w)

    # S = D D^T WITHOUT EVER FORMING D. The test-pair index is a product basis,
    # rho = (mu, j), and the angular weight depends only on the second index, so
    #
    #   S[k,l] = sum_mu sum_j w_j^2 chi_mu(r_k) chi_j(r_k) chi_mu(r_l) chi_j(r_l)
    #          = [sum_mu chi_mu(r_k) chi_mu(r_l)] * [sum_j w_j^2 chi_j(r_k) chi_j(r_l)]
    #          = (A A^T) .* (B B^T)                            elementwise
    #
    # Two GEMMs costing n_k^2 (n_ao + n_2) in place of one costing n_k^2 n_rho
    # with n_rho = n_ao n_2 -- a factor n_ao n_2 / (n_ao + n_2), about n_ao/2.
    # Screening does not break the identity: it drops pairs contributing below
    # pair_tol^2 ~ 1e-20 relative, orders below the Tikhonov shift, so the
    # unscreened S built here and the screened one it replaces agree to far
    # better than the regularization. FD keeps the screened columns.
    #
    # Only the LOWER triangle is built -- S is a Gram matrix -- and mirrored
    # afterwards, halving what is left. Row-blocked, so neither GEMM
    # intermediate ever reaches n_k x n_k.
    A = np.ascontiguousarray(ao)
    B = np.ascontiguousarray(ao[:, second] * w[None, :])
    S = np.zeros((nk, nk))
    FD = np.zeros((naux, nk))
    rows = max(1, min(nk, int(block_memory_gb * 1e9 / max(2 * nk * 8, 1))))
    for i0b in range(0, nk, rows):
        i1b = min(i0b + rows, nk)
        P = A[i0b:i1b] @ A[:i1b].T
        P *= B[i0b:i1b] @ B[:i1b].T
        P += aux_on_grid[i0b:i1b] @ aux_on_grid[:i1b].T   # auxiliary block
        S[i0b:i1b, :i1b] = P
        del P
    for i0b in range(0, nk, rows):                        # mirror lower -> upper
        i1b = min(i0b + rows, nk)
        S[i0b:i1b, i1b:] = S[i1b:, i0b:i1b].T
    _say(f'Gram matrix built ({nk}x{nk}, {S.nbytes / 1e9:.1f} GB)')

    # FD = F D^T still goes block by block with the pair screening: F is a
    # fitting coefficient, not a product, so it has none of S's structure.
    per_mu = (nk * n2 + nao * naux) * 8
    ao_loc = mol.ao_loc_nr()
    max_mu = max(1, int(block_memory_gb * 1e9 / max(per_mu, 1)))
    blocks, sh0 = [], 0
    for sh in range(1, mol.nbas + 1):
        if ao_loc[sh] - ao_loc[sh0] >= max_mu or sh == mol.nbas:
            blocks.append((sh0, sh))
            sh0 = sh
        if sh0 >= mol.nbas:
            break

    _say(f'three-centre pass: {len(blocks)} blocks of <={max_mu} AOs '
         f'(mol.nbas={mol.nbas}, auxmol.nbas={auxmol.nbas}); each block is one '
         f'aux_e2 call and its shell-pair setup scales with nbas*auxnbas, so '
         f'the COUNT matters at large nbas even though the total integral work '
         f'does not')
    _t_blocks = time.time()
    _n_kept = [0]
    for _ib, (sh0, sh1) in enumerate(blocks):
        if progress and len(blocks) > 20 and _ib and _ib % max(1, len(blocks) // 10) == 0:
            _el = time.time() - _t_blocks
            _say(f'  block {_ib}/{len(blocks)}  {_el:.0f} s elapsed, '
                 f'~{_el * (len(blocks) - _ib) / _ib:.0f} s left')
        a0, a1 = ao_loc[sh0], ao_loc[sh1]
        D_blk = (ao[:, a0:a1, None] * ao[:, None, second]).reshape(nk, -1)
        D_blk *= np.tile(w, a1 - a0)[None, :]
        # Column maxima without materializing |D_blk|. The obvious form calls
        # np.abs(D_blk) TWICE, each a full n_k x n_rho temporary, and that
        # screening line measured 19.6% of the whole factorization. Two
        # reductions over the existing array allocate nothing, and the global
        # max is just the max of the column maxima.
        col_max = np.maximum(D_blk.max(axis=0), -D_blk.min(axis=0))
        keep = col_max > pair_tol * screen_ref
        if not keep.any():
            continue
        _n_kept[0] += int(keep.sum())
        D_blk = np.ascontiguousarray(D_blk[:, keep])
        e3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1',
                               shls_slice=(sh0, sh1, 0, mol.nbas, 0, auxmol.nbas))
        e3c = e3c.reshape(a1 - a0, nao, naux)[:, second, :].reshape(-1, naux)
        F_blk = scipy.linalg.lu_solve(lu, e3c[keep].T)
        F_blk *= np.tile(w, a1 - a0)[keep][None, :]
        FD += F_blk @ D_blk.T
        del D_blk, e3c, F_blk
    FD += aux_on_grid.T
    _kept = _n_kept[0]
    _say(f'three-centre pass done in {time.time() - _t_blocks:.0f} s; '
         f'{_kept:,} of {nao * n2:,} columns survived screening '
         f'({100 * _kept / max(nao * n2, 1):.1f}%); Cholesky solve next '
         f'({nk}x{nk})')

    scale = np.sqrt(np.clip(np.diag(S), 0.0, None))
    scale[scale == 0] = 1.0
    d = 1.0 / scale
    # Balance IN PLACE. `G = (S * d[:, None]) * d[None, :]` allocates two more
    # n_k x n_k arrays and S is dead afterwards; at 10k basis functions n_k is
    # ~106k, so each of those is 90 GB.
    S *= d[:, None]
    S *= d[None, :]
    G = S
    G[np.diag_indices_from(G)] += regularization
    # STRAIGHT TO LAPACK, not scipy.linalg.solve. G is symmetric positive
    # definite (a Gram matrix plus a Tikhonov shift), so one Cholesky solves it
    # -- but `solve(..., overwrite_a=True)` DOES NOT OVERWRITE. Verified on
    # scipy 1.15.3: G comes back unmodified, so scipy copied it, and newer
    # scipy routes through `_batched_linalg` which is no better. At the
    # chlorophyllide dimer/cc-pVTZ that copy is a second 14.9 GB of Gram matrix
    # and it is what a production run died on -- MemoryError inside
    # scipy.linalg.solve, having asked for exactly the array this code was
    # written to avoid allocating.
    #
    # `posv` factors AND solves in place. Both arrays are passed as .T, which
    # for a C-contiguous array is an F-contiguous VIEW and therefore free: G is
    # symmetric so G.T is G, and FD is dead after this.
    FD *= d[None, :]
    posv = scipy.linalg.lapack.get_lapack_funcs('posv', (G, FD))
    _, Y, info = posv(G.T, FD.T, lower=1, overwrite_a=1, overwrite_b=1)
    if info != 0:
        raise np.linalg.LinAlgError(
            f'Cholesky of the balanced Gram matrix failed at leading minor '
            f'{info}: it is not positive definite, which for a Gram matrix plus '
            f'a {regularization:g} shift means the grid is degenerate -- points '
            'coincide, or a whole sub-shell collapsed onto one radius.')
    _say('fit done')
    return Y.T * d[None, :]


def fit_M(D, F, regularization=DEFAULT_REGULARIZATION):
    """Their eqs 8-9: balanced, Tikhonov-regularized least-squares estimator.

    Cost is one (nk x nk) Gram matrix, one inversion and two products -- O(N^3)
    with the test set linear in system size.
    """
    scale = np.sqrt(np.einsum('kr,kr->k', D, D))
    scale[scale == 0] = 1.0
    d = 1.0 / scale
    Dt = D * d[:, None]
    G = Dt @ Dt.T
    G[np.diag_indices_from(G)] += regularization
    return ((F @ Dt.T) @ np.linalg.inv(G)) * d[None, :]


def fit_M_stable(D, F, regularization=DEFAULT_REGULARIZATION):
    """`fit_M` with the explicit inverse replaced by a Cholesky solve.

    The same estimator, a better numerical realization of it. `fit_M` forms
    np.linalg.inv(G) at cond(G) ~ 2e8, which costs digits in the forward pass;
    production's default `fit_M_streaming` already solves rather than inverts,
    and `build_separable_ri` calls it "the better conditioned of the two".
    """
    s = np.sqrt(np.einsum('kr,kr->k', D, D))
    s = np.where(s == 0.0, 1.0, s)
    d = 1.0 / s
    Dt = D * d[:, None]
    G = Dt @ Dt.T
    G[np.diag_indices_from(G)] += regularization
    cho = scipy.linalg.cho_factor(G, lower=True)
    return scipy.linalg.cho_solve(cho, (F @ Dt.T).T).T * d[None, :]


def build_separable_ri(mol, coords, auxbasis=None, auxmol=None,
                       regularization=DEFAULT_REGULARIZATION,
                       l_max_second=2, streaming=True, block_memory_gb=4.0,
                       pair_tol=DEFAULT_PAIR_TOL):
    """Returns (X, Z, M) with X[k, mu] = chi_mu(r_k) and Z = M^T V M.

    X is returned grid-major, matching `space_time.py`.

    block_memory_gb is forwarded to whichever of the two paths runs; it caps
    their per-block working set and is the only handle on the peak, so it has
    to be reachable from here -- see `fit_M_streaming` for what it does and
    does not buy.

    streaming=True accumulates D D^T and F D^T blockwise instead of holding D
    and F, which is what makes the large end of a size series run at all: D is
    n_k x n_rho and reaches 341 GB at undecacene/cc-pVTZ, against a 252 GB node.
    The two agree to the accuracy of the linear solve (8e-9 relative on water,
    where the difference is `fit_M`'s explicit inverse against a Cholesky solve
    on the same regularized Gram matrix -- the streaming path is the better
    conditioned of the two). streaming=False keeps the reference path, which is
    still what `fit_error_coulomb` and the radius optimizer use.
    """
    if auxmol is None:
        auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis)
    if streaming:
        M = fit_M_streaming(mol, auxmol, coords, l_max_second=l_max_second,
                            regularization=regularization,
                            block_memory_gb=block_memory_gb, pair_tol=pair_tol)
    else:
        D, F = build_D_F(mol, auxmol, coords, l_max_second=l_max_second,
                         block_memory_gb=block_memory_gb, pair_tol=pair_tol)
        M = fit_M(D, F, regularization)                  # (naux, nk)
    V = auxmol.intor('int2c2e', aosym='s1')
    Z = M.T @ V @ M
    X = mol.eval_gto('GTOval_sph', coords)               # (nk, nao)
    return X, Z, M


def aux_metric_sqrt(auxmol, environment=None, V=None):
    """V^(1/2) of the auxiliary Coulomb metric: the gauge D = M^T V^(1/2) is built in.

    An environment substitutes v -> v + vtilde, and the interaction enters the
    factorization only through this metric (Z = D D^T fits pair densities
    against V), so dressing it is the whole substitution (Duchemin, Jacquemin
    and Blase, J. Chem. Phys. 144, 164106 (2016), Eq. 16; the DF analogue is
    SolventScreening.whitened_transform). The least-squares fit itself keeps
    the bare metric; only the gauge is dressed. v + vtilde is still a positive
    kernel, so a negative eigenvalue means the discretized reaction field
    over-screens the bare interaction and is an error, not a truncation.

    environment: anything with `aux_kernel(auxmol)`, returning vtilde_PQ or
    None when nothing screens (src.Base.environment). V: the bare (P|Q), if
    already formed.
    """
    V = auxmol.intor('int2c2e', aosym='s1') if V is None else V
    kernel = None if environment is None else environment.aux_kernel(auxmol)
    if kernel is not None:
        V = V + kernel
    w, v = np.linalg.eigh(V)
    if kernel is not None and w.min() < -1e-10 * w.max():
        raise RuntimeError(
            f"the screened auxiliary metric v + vtilde is indefinite "
            f"(smallest eigenvalue {w.min():.3e}): the reaction field "
            f"over-screens the bare interaction. Check eps and the cavity "
            f"(lebedev_order / vdw_scale).")
    keep = w > 1e-12 * w.max()
    return (v[:, keep] * np.sqrt(w[keep])) @ v[:, keep].T


def fit_error_coulomb(mol, auxmol, coords, M=None, l_max_second=2,
                      regularization=DEFAULT_REGULARIZATION):
    """Their eq 10 objective: ||F^RS(rho) - F^V(rho)|| in the COULOMB metric,
    which is what the grid radii are optimized against.
    """
    D, F = build_D_F(mol, auxmol, coords, l_max_second=l_max_second)
    if M is None:
        M = fit_M(D, F, regularization)
    V = auxmol.intor('int2c2e', aosym='s1')
    R = M @ D - F                                        # (naux, nrho)
    return float(np.sqrt(np.einsum('br,bc,cr->', R, V, R)))


# ---------------------------------------------------------------------------
# Offline, per-element grid optimization (their eq 10)
# ---------------------------------------------------------------------------
#
# "The optimized {rk} sets are generated for isolated atoms, once for every
# chemical species and their associated atomic basis sets. These atomic grids
# are then duplicated according to the molecule geometry."
#
# Structure: each Lebedev sub-shell is replicated at its OWN set of radii, so
# the cheap 6-point A1 shell can afford many radial samples while the 24-point
# B1 shell gets few. Giving every shell the same radii -- the obvious first
# guess -- wastes most of the budget on B1 and needs 5-20x the auxiliary basis
# size; with per-shell counts the target is ~3x.
#
# The only variables are the number of radii per shell (fixed by the caller,
# since it sets the grid size) and their lengths (optimized here).

_DEFAULT_COUNTS = {'A1': 8, 'A2': 6, 'A3': 4, 'B1': 2}


def _radii_from_flat(x, counts):
    out, i = {}, 0
    for name in ('A1', 'A2', 'A3', 'B1'):
        n = counts.get(name, 0)
        if n:
            out[name] = np.exp(x[i:i + n])
            i += n
    return out


def _flat_from_radii(radii):
    return np.concatenate([np.log(radii[n]) for n in ('A1', 'A2', 'A3', 'B1')
                           if n in radii and len(np.atleast_1d(radii[n]))])


#: Radial shapes for the multi-start search, as (lo, hi, curvature) triples.
#: The objective is multi-modal and start diversity is the only lever measured
#: to help. One descent per shape, carbon/cc-pVDZ at 148 counts, everything
#: else at production settings:
#:
#:     7.59e-02   4.50e-04   3.03e-03   1.03e-01   1.47e-02   2.61e-02
#:
#: Shape 0 is the plain geometric ladder, so n_start=1 is the old single
#: descent -- and it is the worst of the six. Two variations that look like
#: fixes are not: a monotone reparametrization making coincident radii
#: unreachable was no better, and widening r_max acts only by moving where the
#: starts land, since nothing sits on the bound and the result is not monotone
#: in it.
_START_SHAPES = ((2.0, 0.80, 1.00), (1.0, 0.80, 1.00), (4.0, 0.80, 1.00),
                 (0.5, 0.80, 1.00), (2.0, 0.50, 1.00), (2.0, 1.00, 1.00),
                 (2.0, 0.80, 1.60), (2.0, 0.80, 0.62), (1.0, 0.50, 1.60),
                 (4.0, 1.00, 0.62), (0.5, 0.50, 1.30), (8.0, 0.80, 0.80),
                 (1.5, 0.65, 1.90), (3.0, 0.90, 0.45), (0.8, 0.75, 1.15),
                 (6.0, 0.60, 1.45))


def _start_radii(counts, r_min, r_max, k):
    """Starting radii for descent `k`: a geometric ladder, warped.

    r_i = lo (hi/lo)^((i/(n-1))^p), so p > 1 crowds the samples inwards where
    the co-density is largest and p < 1 pushes them out. Cycles the table if
    more starts are asked for than it holds.
    """
    lo_f, hi_f, p = _START_SHAPES[k % len(_START_SHAPES)]
    lo, hi = r_min * lo_f, r_max * hi_f
    out = {}
    for name, n in counts.items():
        if not n:
            continue
        t = np.zeros(1) if n == 1 else np.linspace(0.0, 1.0, n)
        out[name] = lo * (hi / lo) ** (t ** p)
    return out


#: Radial search box per element, Bohr, for a MULTI-START search. The single
#: geometric descent keeps `LEGACY_R_MAX` so every cached and shipped grid keeps
#: its key. Second-row values are the measured ones: benzene's 23x exchange
#: gain was found at 5.0, and widening to 16 made it worse. Magnesium's density
#: reaches ~14 Bohr -- its fit error is flat in point count and 8x better at 16
#: (radii out to 13.7; Mg(OH)2 exchange error 0.72 -> 0.19 mHa/atom) -- and the
#: other diffuse-valence elements (groups 1-2, period 3) are given the same
#: prior, to be checked against exchange errors on probes containing them.
#: Published carbon's largest A1 radius, 5.295, already lies outside the legacy
#: box.
LEGACY_R_MAX = 5.0
ELEMENT_R_MAX = {'H': 5.0, 'He': 5.0,
                 'B': 5.0, 'C': 5.0, 'N': 5.0, 'O': 5.0, 'F': 5.0, 'Ne': 5.0,
                 'Li': 16.0, 'Be': 16.0, 'Na': 16.0, 'Mg': 16.0,
                 'K': 20.0, 'Ca': 20.0,
                 'Al': 12.0, 'Si': 12.0, 'P': 12.0, 'S': 12.0, 'Cl': 12.0,
                 'Ar': 12.0}


def search_r_max(element, n_start=1, r_max=None):
    """The box a radii search runs in: explicit, legacy for one descent, else per element."""
    if r_max is not None:
        return float(r_max)
    return LEGACY_R_MAX if n_start == 1 else ELEMENT_R_MAX.get(element, 12.0)


def shipped_radii():
    """Optimized atomic radii that travel WITH the source, not in a scratch cache.

    `optimize_atomic_radii`'s on-disk cache is gitignored, so a clean checkout
    re-optimizes from scratch -- and that optimizer is a local descent whose
    result is not reproducible (see the note on the cache below). Measured
    consequence: the pinned BSE roots in tests/test_bse_isdf_driver.py move by
    6.5-8.7 meV between a populated and an empty cache, i.e. a fresh clone fails
    its own regression tests. Shipping the tables fixes the reproducibility;
    it does NOT make a bad grid good, which is why each entry carries the
    `fit_error` it was accepted at. Read those before trusting a row: anything
    approaching 1 is a fit that has failed, not a grid that is merely coarse.

    Keyed on the same tuple as `_radii_cache_path`, so a lookup here and a cache
    hit cannot disagree about what they are answering.

    Returns {key: entry}; use `shipped_radii_lookup` rather than indexing.
    """
    path = os.path.join(os.path.dirname(__file__), 'data', 'optimized_radii.json')
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _shipped_key(element, basis, auxbasis, counts, n_start=1,
                 r_max=LEGACY_R_MAX):
    """Table key. The recipe (n_start, r_max) is appended only when it is not
    the legacy single descent in the legacy box, so every existing key stands
    and rows for different recipes coexist instead of overwriting each other."""
    recipe = ([] if n_start == 1 else [['n_start', int(n_start)]]) + \
             ([] if float(r_max) == LEGACY_R_MAX else [['r_max', float(r_max)]])
    return json.dumps([element, str(basis), str(auxbasis),
                       sorted((counts or {}).items())] + recipe)


def shipped_radii_lookup(element, basis, auxbasis, settings):
    """(radii, fit_error) from the shipped table, or None.

    The FULL settings dict has to match, not just the element and basis. A
    table entry optimized under different settings is a different grid, and
    handing it back is the silent substitution `_radii_settings` exists to
    prevent -- there is no point guarding the scratch cache and leaving the
    shipped table open.
    """
    entry = shipped_radii().get(
        _shipped_key(element, basis, auxbasis, dict(settings['counts']),
                     settings.get('n_start', 1), settings['r_max']))
    if entry is None:
        return None
    # Compare CANONICAL JSON, not the objects: a round trip through the file
    # turns every tuple into a list, so `entry['settings'] != settings` is true
    # for two identical settings and the table silently misses on all of them.
    if json.dumps(entry.get('settings'), sort_keys=True) != json.dumps(settings, sort_keys=True):
        return None
    return ({k: np.array(v) for k, v in entry['radii'].items()},
            entry['fit_error'])


def _radii_settings(counts, r_min, r_max, l_max_second, regularization,
                    maxiter, seed, basin_hopping, temperature, step,
                    n_start=1, origin=False):
    """Every argument that changes the radii this optimizer returns.

    ALL of them belong in the cache key. Leaving one out does not cause a miss,
    it causes a silent HIT on a grid optimized under different settings -- the
    caller asks for one thing, gets another, and the two are indistinguishable
    because the answer looks perfectly reasonable. `l_max_second` was outside
    the key and an experiment that varied it returned four identical numbers in
    zero seconds, which is the only reason it was noticed.
    """
    out = ({'n_start': int(n_start), 'origin': bool(origin)}
           if (n_start != 1 or origin) else {})
    return {**out,
            'counts': sorted((counts or {}).items()), 'r_min': r_min,
            'r_max': r_max, 'l_max_second': l_max_second,
            'regularization': regularization, 'maxiter': maxiter, 'seed': seed,
            'basin_hopping': basin_hopping, 'temperature': temperature,
            'step': step}


def _radii_cache_path(element, basis, auxbasis, settings):
    key = json.dumps([element, str(basis), str(auxbasis), settings], sort_keys=True)
    tag = hashlib.sha1(key.encode()).hexdigest()[:12]
    d = os.path.join(os.path.dirname(__file__), 'data', 'radii_cache')
    return os.path.join(d, f'{element}_{tag}.json')


def optimize_atomic_radii(element, basis, auxbasis, counts=None,
                          r_min=0.05, r_max=None, l_max_second=2,
                          regularization=DEFAULT_REGULARIZATION,
                          maxiter=200, seed=0, verbose=False,
                          basin_hopping=0, temperature=0.5, step=0.35,
                          n_start=1, origin=False, return_candidates=False):
    """Minimize their eq 10 over the radii of one isolated atom.

    Returns (radii dict, final Coulomb-metric fit error). Run once per
    (element, basis, auxbasis) and cache -- this is the whole reason the
    per-molecule step stays O(N^3).

    Optimizes log-radii so positivity is automatic and the search is
    scale-free. basin_hopping=0 does a single L-BFGS-B descent from a geometric
    start; a positive value runs that many basin-hopping restarts on top, which
    is what the paper does ("a basin-hopping mechanism coupled to a limited
    memory Broyden-Fletcher-Goldfarb-Shanno algorithm"). The objective is
    multi-modal in the radii, so the plain local descent lands well above the
    published grids' accuracy.

    n_start:  local descents from different radial SHAPES, best kept. This is
        the knob that matters: the single geometric descent returns carbon's A3
        as 0.199, 0.202 -- duplicates wasting 12 of its 36 points, at fit error
        7.59e-02 -- where four starts reach 4.66e-04, 163x better, for four
        times an offline cost paid once per (element, basis, counts) and
        cached. See `_START_SHAPES`.
    origin:   optimize with the nuclear cusp sampled, as every published table
        does. Worth 2.2-5.5x on those tables, and ~1.00x if bolted onto radii
        optimized without it -- a grid with no cusp point already spends a
        radius near zero doing that job.
    return_candidates: also return every start's (radii, fit_error). The fit
        error does not predict the exchange error -- two carbon grids at
        4.50e-04 differ 2.4x in |dE_x| on benzene -- so a caller with a better
        criterion (`ISDFJK.check_k` on a probe) can choose among the local
        minima itself. The cache still holds the best BY FIT ERROR.

    r_max:    None selects the box by `search_r_max`: the legacy 5.0 for a
        single descent, `ELEMENT_R_MAX[element]` for a multi-start search. The
        legacy box excludes published carbon's largest A1 radius (5.295) and
        magnesium's density altogether; it is kept for n_start=1 only so that
        every existing cached and shipped grid keeps its key.
    """

    counts = counts or _DEFAULT_COUNTS

    # Cache on disk. Two reasons, and the second is the important one:
    #
    #  * it is recomputed on every call otherwise, which is pure waste;
    #  * the result is NOT reproducible across thread counts. The objective is
    #    evaluated with threaded BLAS and differentiated numerically, so a
    #    different reduction order moves L-BFGS-B onto a different local
    #    minimum. Measured on water/cc-pVDZ: grid checksum 1401.302 at one
    #    thread against 1403.094 at eight, and a quasiparticle energy differing
    #    by 1.1 meV -- larger than the accuracy being claimed for the method.
    #    Caching pins whichever grid was found first, so a study is at least
    #    self-consistent; runs that must agree across machines should ship the
    #    cache with them, or use the published tables.
    # The shipped table first, so a clean checkout reproduces a populated one.
    # It is consulted BEFORE the cache: the cache is per-machine scratch and the
    # table is the version-controlled answer, so where they differ the tracked
    # one has to win or the repository does not describe its own results.
    r_max = search_r_max(element, n_start, r_max)
    settings = _radii_settings(counts, r_min, r_max, l_max_second, regularization,
                               maxiter, seed, basin_hopping, temperature, step,
                               n_start, origin)
    # Neither the table nor the cache holds the runners-up, so a caller that
    # wants every candidate has to run the search; the table and cache still
    # receive its best-by-fit result on the way out.
    shipped = shipped_radii_lookup(element, basis, auxbasis, settings)
    if shipped is not None and not return_candidates:
        return shipped

    cache = _radii_cache_path(element, basis, auxbasis, settings)
    if os.path.exists(cache) and not return_candidates:
        # Tolerate a damaged cache rather than trusting it. A truncated or
        # half-written file is not hypothetical: a SLURM array starts every task
        # at once and they all optimize the same H and C radii into the same
        # path. Anything unreadable is treated as a miss and recomputed.
        try:
            with open(cache) as fh:
                d = json.load(fh)
            return {k: np.array(v) for k, v in d['radii'].items()}, d['fit_error']
        except (ValueError, KeyError, OSError):
            pass
    # The fit involves only basis functions, never the electron count, but
    # gto.M still insists on a consistent spin for an odd-Z atom.
    atom = gto.M(atom=f'{element} 0 0 0', basis=basis, verbose=0,
                 spin=gto.charge(element) % 2)
    auxatom = df.addons.make_auxmol(atom, auxbasis=auxbasis)

    decode = lambda x: _radii_from_flat(
        np.clip(x, np.log(r_min), np.log(r_max)), counts)
    bounds = [(np.log(r_min), np.log(r_max))] * sum(counts.values())
    starts = [_flat_from_radii(_start_radii(counts, r_min, r_max, k))
              for k in range(max(1, n_start))]

    def objective(x):
        try:
            return fit_error_coulomb(atom, auxatom,
                                     atomic_points(decode(x), origin=origin),
                                     l_max_second=l_max_second,
                                     regularization=regularization)
        except np.linalg.LinAlgError:
            return 1e6

    kw = dict(method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter})
    res, candidates = None, []
    for x0 in starts:
        if basin_hopping:
            r = basinhopping(objective, x0, niter=basin_hopping,
                             T=temperature, stepsize=step,
                             minimizer_kwargs=kw, seed=seed)
        else:
            r = minimize(objective, x0, **kw)
        candidates.append((decode(r.x), float(r.fun)))
        if res is None or r.fun < res.fun:
            res = r
    radii = decode(res.x)
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        # ATOMIC. `open(cache, 'w')` truncates before it writes, so a concurrent
        # reader sees an empty file; write to a private temporary in the same
        # directory and rename, which is atomic on POSIX. The pid keeps two
        # tasks from colliding on the temporary itself.
        tmp = f'{cache}.{os.getpid()}.tmp'
        with open(tmp, 'w') as fh:
            json.dump({'element': element, 'basis': str(basis),
                       'auxbasis': str(auxbasis), 'counts': counts,
                       'settings': settings, 'fit_error': float(res.fun),
                       'radii': {k: list(map(float, v)) for k, v in radii.items()}},
                      fh, indent=1)
        os.replace(tmp, cache)
    except OSError:
        pass                    # a read-only checkout must not break the run
    if verbose:
        npts = sum(len(_SHELLS[n]) * len(r) for n, r in radii.items())
        print(f'  {element}: {npts} points, {len(starts)} start(s), '
              f'fit err {res.fun:.3e}')
    if return_candidates:
        return radii, float(res.fun), candidates
    return radii, float(res.fun)


# ---------------------------------------------------------------------------
# Covariant atomic frames
# ---------------------------------------------------------------------------
#
# The Lebedev sub-shells are fixed LAB-FRAME direction sets, so placing them at
# rotated atomic positions gives grid(R.M) != R.grid(M): the centres rotate, the
# directions do not. Measured consequence on the GW HOMO over 24 orientations:
# 0.36 meV std for H2O and 2.87 meV for N2 -- the dominant uncertainty at the
# accuracy being targeted.
# Duchemin & Blase absorbed this by averaging over 40 random orientations.
#
# Orienting each atom's shells in a frame built FROM ITS NEIGHBOURS removes it
# instead of averaging it: if the frame is covariant, so is the whole grid.
#
# The frame comes from the weighted second moment of the neighbour directions,
#     T_i = sum_j w(r_ij) d_ij d_ij^T,
# which satisfies T(R.M) = R T(M) R^T, so its eigenvectors rotate correctly.
# Eigenvector SIGNS are fixed by a covariant odd moment, sum_j w (d.e)^3, since
# eigh's sign convention is arbitrary and would otherwise reintroduce the
# problem. Degenerate eigenvalues leave the frame undetermined within a
# subspace; that is reported rather than silently resolved.

_FRAME_DECAY = 3.0          # bohr; smooth neighbour weighting
_FRAME_DEGEN = 1e-6         # relative eigenvalue gap below which a frame is flagged

#: Reference directions that fix the SIGN of each frame axis. Three linearly
#: independent generic directions (cyclic shifts of 1, 1/phi, 1/phi^2; circulant
#: determinant 0.58), so no axis can be perpendicular to all three and the sign
#: rule is total. Deliberately not axis-aligned: a Cartesian reference is
#: perpendicular to the symmetry axes of exactly the molecules that need this.
_FRAME_SIGN_REFS = np.array([[1.0, 0.6180339887498949, 0.38196601125010515],
                             [0.38196601125010515, 1.0, 0.6180339887498949],
                             [0.6180339887498949, 0.38196601125010515, 1.0]])
_FRAME_SIGN_REFS /= np.linalg.norm(_FRAME_SIGN_REFS, axis=1)[:, None]


def atomic_frames(mol, decay=_FRAME_DECAY, degeneracy_tol=_FRAME_DEGEN):
    """Per-atom orthonormal frames, covariant under a global rotation.

    Returns (frames, degenerate) with frames of shape (natm, 3, 3) whose ROWS
    are the frame axes, and `degenerate` a boolean array flagging atoms whose
    neighbour environment does not determine a frame (isolated atoms, and the
    axial degeneracy of a diatomic). For those the lab frame is used, which is
    harmless exactly when the environment is symmetric enough to cause the
    degeneracy in the first place.

    The axes as LINES are covariant under a global rotation, and that is what
    places the grid: signs only permute rows (see the sign block below). Axis
    signs come from fixed generic references, so the frame is a deterministic
    function of the geometry -- reproducible across LAPACK builds -- but it is
    NOT a continuous one, and no convention could be.
    """
    coords = np.asarray(mol.atom_coords())
    natm = len(coords)
    frames = np.zeros((natm, 3, 3))
    degenerate = np.zeros(natm, dtype=bool)

    for i in range(natm):
        d = coords - coords[i]
        r = np.linalg.norm(d, axis=1)
        keep = r > 1e-8
        if not keep.any():
            frames[i] = np.eye(3); degenerate[i] = True
            continue
        dj = d[keep] / r[keep, None]
        w = np.exp(-r[keep] / decay)

        T = np.einsum('j,ja,jb->ab', w, dj, dj)
        evals, evecs = np.linalg.eigh(T)
        order = np.argsort(-evals)
        evals, evecs = evals[order], evecs[:, order]

        # A degenerate pair leaves the frame undetermined WITHIN that subspace,
        # but the axes outside it are still determined and must be kept: falling
        # back to the lab frame wholesale throws away the molecular axis of a
        # diatomic, which is exactly the direction that matters. Complete the
        # degenerate subspace from a fixed reference projected into it -- a
        # deterministic function of the determined axes, so the result is
        # covariant up to a rotation WITHIN the degenerate subspace, and such a
        # rotation is a symmetry of the environment that created the degeneracy.
        scale = max(evals[0], 1e-30)
        axes = evecs.T.copy()                       # rows are axes
        gaps = np.diff(evals) / scale
        degen_pair = np.abs(gaps) < degeneracy_tol
        if degen_pair.any():
            degenerate[i] = True
            k = int(np.argmax(degen_pair))          # axes k, k+1 are mixed
            fixed = axes[k - 1] if k > 0 else None
            if fixed is None:                       # top pair degenerate: anchor
                fixed = axes[2]                     # on the determined third axis
            ref = np.array([1.0, 0.0, 0.0])
            if abs(ref @ fixed) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            e_a = ref - (ref @ fixed) * fixed
            n_a = np.linalg.norm(e_a)
            if n_a < 1e-10:
                frames[i] = np.eye(3)
                continue
            e_a /= n_a
            e_b = np.cross(fixed, e_a)
            if k > 0:
                axes = np.vstack([fixed, e_a, e_b])
            else:
                axes = np.vstack([e_a, e_b, fixed])

        # An axis SIGN is pure gauge: each Lebedev sub-shell is an orbit of the
        # octahedral group, so negating an axis maps the shell onto itself and
        # only permutes grid rows. It must therefore be DETERMINISTIC, not
        # physical -- and an environment moment sum_j w_j (dhat_j . e_k)^3, with
        # the first moment as fallback, is neither: both vanish identically for
        # an axis with no neighbour projection (the out-of-plane axis of any
        # planar environment), leaving eigh's sign, which no LAPACK build
        # promises to reproduce.
        # No convention is continuous everywhere -- equivariance at a symmetric
        # geometry would force an axis to equal its own negative -- so generic
        # references put the unavoidable jump on a generic set rather than on
        # the symmetric configurations molecules actually sit at.
        for k in range(3):
            overlap = _FRAME_SIGN_REFS @ axes[k]
            if overlap[int(np.argmax(np.abs(overlap)))] < 0:
                axes[k] = -axes[k]
        if np.linalg.det(axes) < 0:                 # keep it a proper rotation
            axes[2] = -axes[2]
        frames[i] = axes
    return frames, degenerate


def molecular_points_covariant(mol, radii_by_element, origin_by_element=None,
                               decay=_FRAME_DECAY, return_info=False):
    """The superposition of atomic grids, each atom's shells rotated into its
    local frame.

    Same point count and same radii; only the shell ORIENTATIONS change, so the
    cost and the accuracy at a given grid size are unaffected -- what changes is
    that `grid(R.M) = R.grid(M)` now holds.
    """
    global _SHELLS
    if _SHELLS is None:
        _SHELLS = lebedev_subshells()
    frames, degenerate = atomic_frames(mol, decay=decay)
    coords = []
    for ia in range(mol.natm):
        sym = mol.atom_pure_symbol(ia)
        radii = radii_by_element[sym]
        use_origin = (origin_by_element or {}).get(sym, False)
        pts = atomic_points(radii, centre=(0.0, 0.0, 0.0), origin=use_origin)
        coords.append(pts @ frames[ia] + mol.atom_coord(ia))
    out = np.vstack(coords)
    return (out, frames, degenerate) if return_info else out
