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

THE ROW-DISTRIBUTED FIT
-----------------------
`fit_M_streaming(fit='rows')` (`fit_rows`) solves the same estimator with the
grid index cut into fixed tiles of `FIT_CHOLESKY_BLOCK` points, owned
block-cyclically, so that no rank holds an nk-indexed array whole: the Gram
tiles, the right-hand side, the solve and the collocation (streamed one tile
at a time, never whole) are the rank's own tiles, and D moves to the rank's
contiguous rows at the end. Per rank at the chlorophyllide hexamer/cc-pVTZ
(nk 117762, nao 10980, 9090 AOs with l <= 2, naux 28236), GB:

  array                          replicated     rows, 8 ranks   16 ranks
  Gram S, lower tiles            110.9          7.2             3.7
  (F D^T)^T, then M^T            26.6           3.4             1.7
  X and X[:, l<=2] w             10.3+10.3+8.6  1.3 + 1.1       0.7 + 0.6
  X on the output rows           -              1.4             0.7
  auxiliary collocation          26.6           3.4             1.7
  gathered panel, update block   -              0.5 + 0.5       0.5 + 0.5
  D, X_mo, X_ao                  47.2           5.9             3.0
  metric LU, through the pass    6.4            6.4             6.4
  a shell block's (mu nu|P)      17.4 + 14.4    1.7 + 1.7       1.7 + 1.7
  its co-densities rho(r_k)      59.9           0.26 a tile     0.26 a tile
  V^1/2 for D, rank 0 / others   44.7 / 44.7    12.8 / 0.12     12.8 / 0.12
  peak: Gram/pass/Cholesky/D     ~300 (pass)    21/23/13/19     14/17/7/16

A shell block's three-centre integrals are those of the pairs it keeps
(`KeptIntegrals`). Replicated, one f shell over every nu and its l <= 2
copy; by rows, the largest kept block -- a diffuse Mg p shell, 7.6e3 pairs
-- and one integral call no larger. The screening keeps a pair by distance:
1.0, 0.66, 0.12 and 0 of the pairs at 0-2, 2-4, 4-6 and beyond 6 A in
naphthalene, pentacene and a Mg-chlorin alike, so the chlorophyllide
dimer's per-shell counts with their cross-monomer part doubled bound the
hexamer's: 8% of all pairs kept in the dimer, 28% of that Mg block's.

The metric root: `aux_metric_sqrt` holds up to seven metric-sized arrays at
once on every rank (numpy's eigh copies the metric and asks 2 naux^2 of
workspace; the root is formed from two more copies), where
`RowFit.metric_root_rows` forms it on rank 0 alone in two (`metric_root`)
and streams it in slabs of `block` rows, so the D step's peak is rank 0's
12.8 GB beside its own tiles. The root is `aux_metric_sqrt`'s to 1e-13.

What stays replicated is the metric's LU through the pass. It cannot be
cut: each block's coefficients are solved before the sum over blocks on the
rank that owns the block, and solving once after the sum -- which would drop
the factor from the pass and cut its flops from 2 naux^2 per kept pair to
2 naux^2 per point -- moves D by 24 to 55 of the replicated fit's own
reassociation responses, the metric's conditioning (cond 2e5-5e5 at
cc-pVDZ-RI) multiplying the accumulated sum's rounding; any factorization
solved per block stays within 2.

THE ROW FIT'S ADJOINT
---------------------
`fit_rows_adjoint` carries an adjoint on D back to the nuclei in the same
tiles and under the same rules: every call's shape fixed by tile indices, and
no per-rank partial summed across ranks. A sum over the grid is accumulated
by one rank in tile order from tiles that travel to it: the Gram terms by
the row tile's owner over the streamed column tiles, a shell block's
(mu nu|P) adjoint by the block's owner over the streamed U, the metric's two
(naux, naux) sums on rank 0; every (natm, 3) partial is gathered and added
in its fixed order. The force is therefore the same bits at every rank
count, and the old whole adjoint's (`isdf_derivatives.dfactor_adjoint_gauges`
on every product pair) to the fit's conditioning. Per rank at the hexamer,
GB, 8 / 16 ranks:

  array                               8 ranks     16 ranks
  Gram tiles, then the kept factor    7.2         3.7
  metric LU, through the pass         6.4         6.4
  MT_bar, Q_bar, Q, U, M^T, P_bar     3.4 each    1.7 each
  X, B tiles; X_bar, B_bar            1.3, 1.1    0.7, 0.6
  F_b broadcast / a block's (mu nu|P) 3.4         3.4
  (mu nu|P)' adjoints of one batch    <= 4        <= 4
  derivative integrals of a chunk     2 x 2.1     2 x 2.1
  peak: pass / Gram / (mu nu|P)'      30/18/21    20/10/15
  rank 0, the metric adjoint          25.5        25.5

The pass holds three row arrays beside the kept factor and the LU, which is
its peak; the metric adjoint holds four metric-sized arrays on rank 0 alone
(the eigenvectors, one accumulated sum and two products), whatever the rank
count. D_bar and X_bar arrive whole from the kernels (all-reduced over the
tau partition) and a rank reads its tiles of them; rank 0 reads D_bar whole
for M D_bar.
"""
import hashlib
import json
import os
import tempfile
import time

import numpy as np
import scipy.linalg
from scipy.optimize import minimize, basinhopping
from pyscf import df, gto
from pyscf.dft import gen_grid

from src.Base.constants import (AUX_METRIC_INDEFINITE_TOL,
                                AUX_METRIC_ROOT_FLOOR, FIT_CHOLESKY_BLOCK,
                                FIT_ROW_CHUNK_BYTES, FIT_TRANSPOSE_TILE,
                                ISDF_GRID_ACCURACY, ISDF_GRID_N_START,
                                THREE_CENTER_BLOCK_BYTES)
from src.Base.utils.mpi_grid import (agreement, allgather_ranges, broadcast,
                                     broadcast_rows, contiguous_block,
                                     current_comm, cyclic_tiles_to_blocks,
                                     exchange_blocks, lockstep, partition,
                                     reduce_max, reduce_sum)
from src.Base.utils.threads import (blas_single_threaded, openmp_threads,
                                    row_map)

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


class RowFit:
    """This rank's share of the row-distributed fit (`fit_rows`).

    mt: {tile: rows of M^T}, every tile this rank owns (tile t is grid rows
        [t*block, (t+1)*block), owned by rank t % size), the balanced solve
        back-scaled, (rows, naux).
    X:  the AO collocation on the tiles that overlap this rank's
        `contiguous_block` of the grid, rows [ext[0], ext[1]).
    held: {array: most bytes of it this rank held at once during the fit},
        read off the arrays.

    Nothing in it is whole. The factor rows are read from it in the same
    fixed tiles the fit ran in, so they are bitwise the one-rank rows at
    every rank count.
    """

    def __init__(self, npts, block, comm, mt, X, ext, held):
        self.npts, self.block, self.comm = int(npts), int(block), comm
        size = 1 if comm is None else comm.Get_size()
        rank = 0 if comm is None else comm.Get_rank()
        self.rows = contiguous_block(self.npts, rank, size)
        self.mt, self.X, self.ext, self.held = mt, X, tuple(ext), held

    def ao_rows(self):
        """X_ao[k, mu] = chi_mu(r_k) on this rank's rows, verbatim."""
        (r0, r1), e0 = self.rows, self.ext[0]
        return np.ascontiguousarray(self.X[r0 - e0:r1 - e0])

    def mo_rows(self, mo_coeff):
        """X_mo = X_ao C on this rank's rows: one GEMM per whole tile, so a
        row's bits do not depend on where the rank's block starts."""
        (r0, r1), (e0, e1) = self.rows, self.ext
        out = np.empty((r1 - r0, mo_coeff.shape[1]))
        for t0 in range(e0, e1, self.block):
            t1 = min(t0 + self.block, self.npts)
            lo, hi = max(t0, r0), min(t1, r1)
            out[lo - r0:hi - r0] = (self.X[t0 - e0:t1 - e0]
                                    @ mo_coeff)[lo - t0:hi - t0]
        return out

    def metric_rows(self, metric):
        """D = M^T metric on this rank's rows, `metric` whole on every rank:
        one GEMM per owned tile, the tiles then moved to the contiguous
        layout verbatim."""
        return self._contiguous({t: mt @ metric for t, mt in self.mt.items()},
                                metric.shape[1])

    def metric_root_rows(self, auxmol, environment=None):
        """D = M^T (V + vtilde)^1/2 on this rank's rows, the root on rank 0
        alone: formed there by `metric_root` and streamed to every rank one
        slab of `block` auxiliary rows at a time, one GEMM per owned tile and
        slab, so no other rank holds more of it than a slab and a tile's bits
        do not depend on who owns it."""
        comm = self.comm
        naux = auxmol.nao_nr()
        root, failure = None, None
        self.held['metric_root'] = 0
        if comm is None or comm.Get_rank() == 0:
            try:
                root, self.held['metric_root'] = metric_root(auxmol,
                                                             environment)
            except (RuntimeError, np.linalg.LinAlgError) as err:
                failure = str(err)
        failure = broadcast(failure, comm)   # every rank raises, none waits
        if failure is not None:
            raise RuntimeError(failure)
        tiles = {t: np.empty((len(mt), naux)) for t, mt in self.mt.items()}
        for c0 in range(0, naux, self.block):
            c1 = min(c0 + self.block, naux)
            slab = _root_slab(root, slice(c0, c1), slice(0, naux), comm)
            self.held['metric_slab'] = max(self.held.get('metric_slab', 0),
                                           int(slab.nbytes))
            for t, mt in self.mt.items():
                tiles[t][:, c0:c1] = mt @ slab.T     # rows of B B^T: columns
        del root
        return self._contiguous(tiles, naux)

    def _contiguous(self, tiles, ncol):
        """D's owned tiles moved to this rank's contiguous rows verbatim,
        both sizes in the ledger."""
        self.held['D_tiles'] = sum(int(a.nbytes) for a in tiles.values())
        out = cyclic_tiles_to_blocks(tiles, self.npts, self.block, ncol,
                                     self.comm)
        self.held['D_rows'] = int(out.nbytes)
        return out


class RowFitAdjoint:
    """The row fit's adjoint on this rank (`fit_rows_adjoint`), contracted to
    the few arrays a nuclear gradient reads, the same bits on every rank.

    fit_centre: (natm, 3), what the fit's integrals contribute through their
        centres: the AO and auxiliary collocations', the three-centre
        (mu nu|P) and the two-centre (P|Q) integrals'.
    fit_points: (nk, 3), dE/dr_g through the fit's collocations at every
        point, for the chain through the points and their frames.
    coll_centre, coll_points: the same two of the X_mo collocation
        adjoint X_bar C^T, None without an X_bar.
    held: {array: most bytes of it this rank held at once}.
    """

    def __init__(self, fit_centre, fit_points, coll_centre, coll_points,
                 held):
        self.fit_centre, self.fit_points = fit_centre, fit_points
        self.coll_centre, self.coll_points = coll_centre, coll_points
        self.held = held


class KeptIntegrals:
    """(mu nu|P) of a shell block's kept pairs alone, for the row fit's pass,
    or with `intor` and `comp` their derivative integrals, for its adjoint;
    `block` is a whole block, for the replicated fit's pass.

    Only the nu shells holding a kept pair are evaluated, in runs of
    consecutive shells cut so that no call holds more pairs than the block
    keeps: the block's integrals never exceed twice its kept pairs, where the
    whole block over every nu shell and its l <= 2 copy held (nao + n2) naux
    per mu. The concatenated basis, its AO offsets and libcint's optimizer
    are built once for the pass, since `aux_e2` rebuilds all three per call
    and a block now makes one call per run. The optimizer is the one a
    whole-block call builds (`getints3c` over bas[:max(i1, j1)], every
    orbital shell): its per-pair data exist only below a primitive count and
    libcint evaluates a triplet another way without them, so a call over
    fewer nu shells building its own could change what a triplet computes.
    With it every kept value is the bits the whole-block call gives it.
    """

    def __init__(self, mol, auxmol, intor='int3c2e', comp=1):
        self.intor = mol._add_suffix(intor)
        self.comp = int(comp)
        self.atm, self.bas, self.env = gto.mole.conc_env(
            mol._atm, mol._bas, mol._env,
            auxmol._atm, auxmol._bas, auxmol._env)
        self.ao_loc = gto.moleintor.make_loc(self.bas, self.intor)
        self.cintopt = gto.moleintor.make_cintopt(
            self.atm, self.bas[:mol.nbas], self.env, self.intor)
        self.aux = (mol.nbas, mol.nbas + auxmol.nbas)
        self.naux = auxmol.nao_nr()
        self.shell_of = np.repeat(np.arange(mol.nbas),
                                  np.diff(self.ao_loc[:mol.nbas + 1]))

    def __call__(self, shells, mu, nu):
        """((len(mu), naux) C order, or (comp, len(mu), naux) for a
        derivative integral of `comp` components; bytes of the largest call).

        shells: the block's (sh0, sh1); mu, nu: its kept pairs' AO indices
        in the block's column order, mu in the block, nu in the l <= 2 set.
        """
        sh0, sh1 = shells
        ao_loc = self.ao_loc
        nu_shell = self.shell_of[nu]
        # nu AOs one call may carry: its pairs never outnumber the kept ones
        cap = max(len(mu) // int(ao_loc[sh1] - ao_loc[sh0]), 1)
        runs, j0 = [], None
        for j in np.unique(nu_shell):
            if j0 is None:
                j0 = j1 = j
            elif j != j1 + 1 or ao_loc[j + 1] - ao_loc[j0] > cap:
                runs.append((j0, j1 + 1))
                j0 = j1 = j
            else:
                j1 = j
        runs.append((j0, j1 + 1))
        comp = self.comp
        out = np.empty(((len(mu), self.naux) if comp == 1
                        else (comp, len(mu), self.naux)))
        mu_local = mu - ao_loc[sh0]
        call = 0
        for j0, j1 in runs:
            with blas_single_threaded():             # libcint's OpenMP only
                e3c = gto.moleintor.getints3c(
                    self.intor, self.atm, self.bas, self.env,
                    (sh0, sh1, j0, j1) + self.aux, comp=comp, aosym='s1',
                    ao_loc=ao_loc, cintopt=self.cintopt)  # ([x,] mu, nu, P)
            call = max(call, int(e3c.nbytes))
            rows = np.flatnonzero((nu_shell >= j0) & (nu_shell < j1))
            for i in np.unique(mu_local[rows]):      # one mu's run at a time
                r = rows[mu_local[rows] == i]
                if comp == 1:
                    out[r] = e3c[i, nu[r] - ao_loc[j0]]
                else:
                    out[:, r] = e3c[:, i, nu[r] - ao_loc[j0]]
            del e3c
        return out, call

    def block(self, shells):
        """(mu nu|P) of a shell block over every nu shell, (a, nao, naux) F
        order: the bits `aux_e2` gives, from the environment built once
        rather than once per call."""
        with blas_single_threaded():                 # libcint's OpenMP only
            return gto.moleintor.getints3c(
                self.intor, self.atm, self.bas, self.env,
                tuple(shells) + (0, self.aux[0]) + self.aux, comp=self.comp,
                aosym='s1', ao_loc=self.ao_loc, cintopt=self.cintopt)


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


def subshells():
    """`lebedev_subshells`, built once: fixed direction tables, no reason to rebuild."""
    global _SHELLS
    if _SHELLS is None:
        _SHELLS = lebedev_subshells()
    return _SHELLS


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


PUBLISHED_COUNTS = {'H': (5, 5, 4, 2), 'C': (9, 9, 7, 4),
                    'N': (9, 8, 8, 4), 'O': (9, 9, 7, 4)}


def published_grids():
    """The optimized atomic grids of Duchemin & Blase, JCP 150, 174120 (2019),
    supporting information Tables S8-S11.

    Optimized for **cc-pVTZ / cc-pVTZ-RI**; H, C, N, O only. Sizes 167 (H),
    307 (C), 311 (N), 307 (O), matching the paper's quoted "320 points for each
    C, N and O atom and 180 points for the H atom".

    These are LITERATURE values: the authors optimized them, this project never
    did, so they carry no exchange-probe score and nothing here ranks them.
    They live in the one radii table as ordinary rows, at the counts in
    `PUBLISHED_COUNTS` and with `origin` set, which is why they read back
    through `atomic_grid` like any other row. Their counts appear nowhere else
    in the table, so they compete with nothing and are reached only by asking
    for them -- they are no basis's silent default.

    Returns {element: (radii dict, include_origin)}.
    """
    out = {}
    for el, counts in PUBLISHED_COUNTS.items():
        out[el] = atomic_grid(el, 'cc-pvtz', 'cc-pvtz-ri',
                              dict(zip(_SHELL_ORDER, counts)))
    return out


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


def ao_blocks(mol, nk, n2, naux, block_memory_gb):
    """Shell ranges (sh0, sh1) the first AO index is cut into, ascending in mu.

    One block of mu costs nk*|mu|*n2 for the co-density plus |mu|*nao*naux for
    its three-centre integrals, and `block_memory_gb` caps that working set.
    Ascending mu is what keeps the columns of D and F aligned with the test-set
    order, so the blocks are a partition of the pair index and nothing else.

    Cost per block grows with the block's AO count, and a shell's AO count is
    2l+1 times its contraction depth, so the blocks of one molecule are ragged
    by up to the s-to-f ratio.
    """
    per_mu = (nk * n2 + mol.nao_nr() * naux) * 8
    max_mu = max(1, int(block_memory_gb * 1e9 / max(per_mu, 1)))
    ao_loc = mol.ao_loc_nr()
    blocks, sh0 = [], 0
    for sh in range(1, mol.nbas + 1):
        if ao_loc[sh] - ao_loc[sh0] >= max_mu or sh == mol.nbas:
            blocks.append((sh0, sh))
            sh0 = sh
        if sh0 >= mol.nbas:
            break
    return blocks


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

    ao_loc = mol.ao_loc_nr()
    blocks = ao_blocks(mol, nk, n2, naux, block_memory_gb)

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
        with blas_single_threaded():                     # libcint's OpenMP only
            e3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1',
                                   shls_slice=(sh0, sh1, 0, mol.nbas,
                                               0, auxmol.nbas))
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


def test_set_layout(mol, coords, l_max_second=2, pair_tol=DEFAULT_PAIR_TOL):
    """(mu, nu, weight) of the AO-pair columns `build_D_F` keeps, in its order.

    The test set is (all AOs) x (AOs with l <= l_max_second), angular-weighted
    on the second index and screened on the pair density the grid samples,
    max_k |chi_mu(r_k) chi_nu(r_k)| > pair_tol x a global reference. The
    reference is global and the screen is per column, so the column set does
    NOT depend on how `build_D_F` blocks the first index, and the order is
    simply ascending mu then ascending position in the second set.

    The auxiliary block that follows carries F = identity exactly, so it
    contributes nothing through F and only a collocation through D.

    SCREENING IS A DISCRETE CHOICE and is frozen with everything else: a
    differentiated column set must be the reference geometry's, or the surface
    steps where a pair crosses the tolerance.
    """
    ao = mol.eval_gto('GTOval_sph', coords)
    l_ao = _ao_l_labels(mol)
    second = np.where(l_ao <= l_max_second)[0]
    w = np.array([ANGULAR_WEIGHTS.get(l_ao[j], 1.0) for j in second])
    ref = _screening_reference(ao, second, w)
    mu_keep, nu_keep, w_keep = [], [], []
    for mu in range(mol.nao_nr()):
        blk = ao[:, mu][:, None] * ao[:, second] * w[None, :]
        col_max = np.maximum(blk.max(axis=0), -blk.min(axis=0))
        keep = np.flatnonzero(col_max > pair_tol * ref)
        mu_keep.append(np.full(len(keep), mu, dtype=int))
        nu_keep.append(second[keep])
        w_keep.append(w[keep])
    return (np.concatenate(mu_keep), np.concatenate(nu_keep),
            np.concatenate(w_keep))


def test_set_D(mol, auxmol, coords, layout):
    """The D of `build_D_F` rebuilt from a frozen layout: pairs then auxiliaries.

    Filled in blocks of `FIT_CHOLESKY_BLOCK` pair columns, so no transient
    of the whole pair width is formed beside D; each element is the same
    product. D keeps the collocations' memory order (pyscf's is grid-fastest),
    which is the order every product of the fit reads it in.
    """
    mu, nu, w = layout
    ao = mol.eval_gto('GTOval_sph', coords)
    aux = auxmol.eval_gto('GTOval_sph', coords)
    npair = len(mu)
    D = np.empty((ao.shape[0], npair + aux.shape[1]),
                 order='F' if ao.flags.f_contiguous else 'C')
    for c0 in range(0, npair, FIT_CHOLESKY_BLOCK):
        c1 = min(c0 + FIT_CHOLESKY_BLOCK, npair)
        D[:, c0:c1] = ao[:, mu[c0:c1]] * ao[:, nu[c0:c1]] * w[None, c0:c1]
    D[:, npair:] = aux
    return D


def fit_M_streaming(mol, auxmol, coords, l_max_second=2,
                    pair_tol=DEFAULT_PAIR_TOL,
                    regularization=DEFAULT_REGULARIZATION, block_memory_gb=4.0,
                    progress=False, comm=None, timings=None,
                    fit='replicated', block=None, layout=None):
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

    comm distributes the THREE-CENTRE PASS over ranks: the blocks of the mu
    index are independent -- one `aux_e2` call, one LU solve, one contraction
    into F D^T -- so a rank accumulates its own blocks and the pass ends in ONE
    all-reduce of F D^T, (naux, nk). Everything else (the Gram matrix, the
    Cholesky solve) is replicated, so every rank returns the same M and no
    caller has to know whether a comm was passed. None is `current_comm()`.
    The points are a `lockstep` of rank 0's at entry, IN PLACE: a caller that
    placed them itself (an `eigh` for the atomic frames, a run-time radius
    search) may hold other last bits on another node, and the stripes of two
    grids do not add. An audited run compares the digest of M on the way out
    (`mpi_grid.agreement`), which says whether the replicated Cholesky tail
    repeated bitwise across the ranks.

    The accumulator is REPLICATED, not distributed: every rank holds the full
    (naux, nk) array, 0.78 GB at the chlorophyllide dimer/cc-pVTZ. That is the
    price of one reduction instead of a redistribution, and it is what makes
    the pass -- hours at that size, ~2000 blocks -- the only part worth
    splitting.

    The distributed answer is not bitwise the serial one: rank-ordered partial
    sums reorder the additions of a sum whose terms are the same. That is
    3e-16 relative on the accumulator, pure rounding; the Cholesky solve then
    amplifies it by the conditioning of the balanced Gram matrix, which
    tests/test_isdf_fit_ranks.py measures and bounds.

    fit='rows' returns a `RowFit` instead of M: the same estimator with no
    nk-indexed array whole on any rank and every step in fixed tiles of
    `block` points (`FIT_CHOLESKY_BLOCK` when None), so that its rows are
    bitwise identical at every rank count -- see `fit_rows`. The default,
    'replicated', is the path described here; `block` is read only by 'rows'.

    timings: dict, filled at phase boundaries, never read -- a caller reuses
    one across repeated calls exactly as `solve_qp_energy_space_time` does, and
    the clock reads touch no bit of M. `fit_collocation` covers every one-time,
    REPLICATED setup step before the Gram matrix and the three-centre pass
    each get their own array: the AO and auxiliary collocation, the test-set
    index/weight selection, and the auxiliary Coulomb metric's LU
    factorization plus the screening threshold (both consumed only inside the
    three-centre pass, but built once, not per block). `fit_integrals` is the
    three-centre pass THIS RANK walked -- `ao_blocks` and `partition` are
    cheap enough to fold into its start -- and `fit_blocks`/`fit_blocks_total`
    are the blocks this rank walked and the blocks there are, equal on a
    serial fit. `fit_integrals_reduce` is the one collective that closes the
    pass: the (naux, nk) reduction, the replicated auxiliary-block addition
    that follows it, and the scalar reduction of the screening tally the
    progress line reports. `fit_gram` is the Gram
    matrix itself, built before the three-centre pass runs, PLUS its balancing
    (row-normalize, regularize), read only after the pass -- both are O(n_k^2)
    passes over the same array. `fit_cholesky` is `FD *= d` and the one LAPACK
    `posv` call: scipy's symmetric-positive-definite driver factors AND solves
    in a single call, so `fit_solve` is always 0.0 here -- the code has no
    separate solve to time, and the key is kept so the set of keys does not
    depend on how the solve is realized.
    """
    if fit not in ('replicated', 'rows'):
        raise ValueError(f"fit={fit!r}: 'replicated' or 'rows'")
    if fit == 'rows':
        return fit_rows(mol, auxmol, coords, l_max_second=l_max_second,
                        pair_tol=pair_tol, regularization=regularization,
                        block_memory_gb=block_memory_gb, progress=progress,
                        comm=comm, timings=timings, block=block,
                        layout=layout)
    if layout is not None:
        raise ValueError("a frozen pair layout is read by fit='rows' alone; "
                         'the replicated fit screens at its own points')
    comm = current_comm() if comm is None else comm
    rank, nranks = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    if nranks > 1:
        coords = lockstep(coords, comm)
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    nk = len(coords)
    _t = time.time()
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
        if progress and rank == 0:
            print(f'[isdf {time.strftime("%H:%M:%S")}] {msg}', flush=True)

    _say(f'fit start: nk={nk} nao={nao} naux={naux} n2={n2} '
         f'block_memory_gb={block_memory_gb} pair_tol={pair_tol:.0e} '
         f'l_max_second={l_max_second} regularization={regularization:.0e}')

    V = auxmol.intor('int2c2e', aosym='s1')
    lu = scipy.linalg.lu_factor(V)
    screen_ref = _screening_reference(ao, second, w)
    if timings is not None:
        timings['fit_collocation'] = time.time() - _t

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
    # Only the LOWER block triangle is built -- S is a Gram matrix -- and its
    # transpose balanced into the upper one, halving what is left.
    # Row-blocked, so neither GEMM buffer ever reaches n_k x n_k.
    _t = time.time()
    A = np.ascontiguousarray(ao)
    B = np.ascontiguousarray(ao[:, second] * w[None, :])
    S = np.zeros((nk, nk))
    FD = np.zeros((naux, nk))
    rows = max(1, min(nk, int(block_memory_gb * 1e9 / max(2 * nk * 8, 1))))
    # The GEMMs are BLAS's; the Hadamard product, the auxiliary sum, the
    # transpose and the balancing are O(n_k^2) element-wise passes numpy runs
    # on one core, so they run in `row_map`'s threads, each element the same
    # operations in the same order.
    threads = openmp_threads()
    _gram_lower(S, A, B, aux_on_grid, rows, threads)
    if timings is not None:
        timings['fit_gram'] = time.time() - _t
    _say(f'Gram matrix built ({nk}x{nk}, {S.nbytes / 1e9:.1f} GB)')

    # FD = F D^T still goes block by block with the pair screening: F is a
    # fitting coefficient, not a product, so it has none of S's structure.
    _t = time.time()
    ao_loc = mol.ao_loc_nr()
    blocks = ao_blocks(mol, nk, n2, naux, block_memory_gb)
    # ROUND-ROBIN, not contiguous. A block's cost grows with the AOs it holds
    # and a shell holds 2l+1 per contraction, so the blocks of one molecule are
    # ragged -- and they are ordered by atom, so contiguous slices would hand
    # one rank the heavy atoms and another the hydrogens. Striping spreads that.
    # Nothing here wants contiguity: each block is an independent int3c2e call
    # contracted into the same accumulator, with no slab GEMM to keep whole.
    mine = partition(len(blocks), rank, nranks)
    # The integrals are libcint's OpenMP and the solve and product BLAS's;
    # numpy's element-wise work -- the co-densities, their screen, the kept
    # (mu nu|P) rows, the accumulation -- runs on one core unless cut, so it
    # runs in `row_map`'s threads a slab of rows at a time, each element the
    # same product, on A: a slab of pyscf's F-order `ao` is strided.

    _say(f'three-centre pass: {len(blocks)} blocks of '
         f'<={max(ao_loc[s1] - ao_loc[s0] for s0, s1 in blocks)} AOs over '
         f'{nranks} rank(s) (mol.nbas={mol.nbas}, auxmol.nbas={auxmol.nbas}); '
         f'each block is one int3c2e call on an optimizer built once, its '
         f'element-wise work on {threads} thread(s)')
    _t_blocks = time.time()
    _n_kept = [0]
    integrals = KeptIntegrals(mol, auxmol)
    product = np.empty_like(FD)
    for _done, _ib in enumerate(mine):
        sh0, sh1 = blocks[_ib]
        if progress and len(mine) > 20 and _done and _done % max(1, len(mine) // 10) == 0:
            _el = time.time() - _t_blocks
            _say(f'  block {_done}/{len(mine)}  {_el:.0f} s elapsed, '
                 f'~{_el * (len(mine) - _done) / _done:.0f} s left')
        a0, a1 = ao_loc[sh0], ao_loc[sh1]
        keep = (_pair_peaks(A, a0, a1, second, w, threads)
                > pair_tol * screen_ref)
        if not keep.any():
            continue
        kept = np.flatnonzero(keep)
        _n_kept[0] += len(kept)
        D_blk = _pair_columns(A, a0 + kept // n2, second[kept % n2],
                              w[kept % n2], threads)
        rhs = _kept_integrals(integrals.block((sh0, sh1)), kept // n2,
                              second[kept % n2], threads)
        F_blk = scipy.linalg.lu_solve(lu, rhs.T, overwrite_b=True)
        F_blk *= w[kept % n2][None, :]
        np.matmul(F_blk, D_blk.T, out=product)
        _accumulate(FD, product, threads)
        del D_blk, rhs, F_blk
    del product, integrals
    if timings is not None:
        timings['fit_integrals'] = time.time() - _t
        timings['fit_blocks'] = len(mine)
        timings['fit_blocks_total'] = len(blocks)

    # The one collective of the fit. Every rank started from zeros and touched
    # only its own blocks, so the sum over ranks is the serial sum with its
    # additions reordered; the auxiliary block is added after, replicated, so
    # it enters once.
    _t = time.time()
    reduce_sum(FD, comm)
    _accumulate(FD, aux_on_grid.T, threads)
    # Screening is reported over the whole test set, not over this rank's share.
    _kept = int(reduce_sum(np.array([_n_kept[0]]), comm)[0])
    if timings is not None:
        timings['fit_integrals_reduce'] = time.time() - _t
    _say(f'three-centre pass done in {time.time() - _t_blocks:.0f} s; '
         f'{_kept:,} of {nao * n2:,} columns survived screening '
         f'({100 * _kept / max(nao * n2, 1):.1f}%); Cholesky solve next '
         f'({nk}x{nk})')

    # Balancing reads S, built in the Gram phase above, and is the same O(n_k^2)
    # order as building it -- so its cost is ADDED to `fit_gram` rather than
    # given a phase of its own that this function's structure does not have.
    _t = time.time()
    scale = np.sqrt(np.clip(np.diag(S), 0.0, None))
    scale[scale == 0] = 1.0
    d = 1.0 / scale
    # Balance IN PLACE, S[r, c] d_r d_c. `G = (S * d[:, None]) * d[None, :]`
    # allocates two more n_k x n_k arrays and S is dead afterwards; at 10k
    # basis functions n_k is ~106k, so each of those is 90 GB. Only the
    # diagonal blocks and what lies right of them are balanced, the lower
    # block triangle's transpose written there first: that is the lower
    # triangle of the F-order view `posv` factors, and LAPACK never
    # references the rest.
    _balanced_upper(S, d, rows, threads)
    G = S
    G[np.diag_indices_from(G)] += regularization
    if timings is not None:
        timings['fit_gram'] += time.time() - _t

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
    # symmetric so G.T is G, and FD is dead after this. It is ALSO one LAPACK
    # call that factors and solves together, so there is no separate solve left
    # to time: `fit_cholesky` carries the whole call and `fit_solve` is always
    # 0.0.
    _t = time.time()
    _scale_columns(FD, d, threads)
    posv = scipy.linalg.lapack.get_lapack_funcs('posv', (G, FD))
    _, Y, info = posv(G.T, FD.T, lower=1, overwrite_a=1, overwrite_b=1)
    if info != 0:
        raise np.linalg.LinAlgError(
            f'Cholesky of the balanced Gram matrix failed at leading minor '
            f'{info}: it is not positive definite, which for a Gram matrix plus '
            f'a {regularization:g} shift means the grid is degenerate -- points '
            'coincide, or a whole sub-shell collapsed onto one radius.')
    if timings is not None:
        timings['fit_cholesky'] = time.time() - _t
        timings['fit_solve'] = 0.0
    _say('fit done')
    M = Y.T                          # (naux, nk) C order, scaled in place
    _scale_columns(M, d, threads)
    if nranks > 1:
        agreement(M, comm, audit_only=True, label='fit_M_streaming output')
    return M


def fit_rows(mol, auxmol, coords, l_max_second=2, pair_tol=DEFAULT_PAIR_TOL,
             regularization=DEFAULT_REGULARIZATION, block_memory_gb=4.0,
             progress=False, comm=None, timings=None, block=None,
             layout=None):
    """`fit_M_streaming`'s estimator with the grid index distributed: M^T as
    this rank's block-cyclic tiles of `block` points (a `RowFit`).

    Every nk-indexed array exists only as the tiles a rank owns (tile t, rank
    t % size): the Gram matrix's lower-triangle tiles, the right-hand side
    F D^T, the solve, the collocation. Every arithmetic step is one call of a
    shape fixed by the tile indices alone, so a rank count changes who makes a
    call and never its shape, and the result is bitwise the one-rank result:

      Gram     S_ik = (X_i X_k^T) o (B_i B_k^T) + P_i P_k^T for i >= k, i
               owned, with X, B = X[:, l <= 2] w and P the auxiliary
               collocation evaluated one tile at a time, the column tiles
               streamed (every rank evaluates each once); balanced by
               diag(S), gathered.
      pass     every rank walks every shell block of `ao_blocks`, largest
               first: its screening maxima over its own points, max-reduced
               (exact at any rank count); the block's fitting coefficients
               F_b = V^-1 (Q|rho) on one rank, in rounds of one block per
               rank, with the replicated fit's own arithmetic on the (Q|rho)
               of the pairs the block keeps, evaluated alone
               (`KeptIntegrals`, the whole-block call's bits); F_b broadcast;
               each rank adds D_b[tile] F_b^T into its tiles of (F D^T)^T in
               the fixed block order. F moves and the accumulator does not:
               a reduction of per-rank partials is neither rows-only nor
               bitwise across rank counts, and the contraction does
               2 nk/size flops per double of F received.
      Cholesky right-looking over the tiles: the owner of tile j factors it
               (dpotrf), broadcasts it and its solved right-hand side block;
               every rank solves its rows of the panel (trsm), updates its
               right-hand sides, gathers the panel and updates its trailing
               tiles one row tile per GEMM. The forward substitution rides
               in the same sweep; the backward one runs from the panels the
               owners keep, each solved block broadcast by its owner.

    The factor is not the replicated fit's bits -- another blocking of the
    same Cholesky and other GEMM shapes -- but a deterministic realization of
    the same estimator, within a few of its own reassociation responses
    (`FIT_REASSOCIATION_K`). On one rank the same tiles run with no
    collective.

    block: tile edge in grid points, `FIT_CHOLESKY_BLOCK` when None.
    layout: a frozen `test_set_layout` (mu, nu, weight), the pair columns
    F D^T keeps, in place of the screen at these points: a walk keeps the
    reference geometry's pairs, where screening again at every geometry
    steps the surface when a pair crosses `pair_tol`. The geometry's own
    layout gives the screened fit's bits; None screens here.
    timings: the keys of `fit_M_streaming`. `fit_collocation` is the metric's
    LU and the test-set labels; the collocation streams through `fit_gram`,
    which includes the balancing; `fit_integrals` is the whole pass and
    `fit_integrals_reduce` its collectives (the screening maxima and the
    coefficient broadcasts); `fit_cholesky` the factorization; `fit_solve`
    both substitutions; `fit_blocks` the blocks whose coefficients this rank
    computed.
    """
    comm = current_comm() if comm is None else comm
    rank, nranks = ((0, 1) if comm is None
                    else (comm.Get_rank(), comm.Get_size()))
    if nranks > 1:
        coords = lockstep(coords, comm)
    block = FIT_CHOLESKY_BLOCK if block is None else int(block)
    if block < 1:
        raise ValueError(f'block={block}: a tile holds at least one point')
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    nk = len(coords)
    ntiles = -(-nk // block)
    tiles = [(t * block, min((t + 1) * block, nk)) for t in range(ntiles)]
    owners = [[int(t) for t in partition(ntiles, r, nranks)]
              for r in range(nranks)]
    mine = owners[rank]
    r0, r1 = contiguous_block(nk, rank, nranks)
    e0 = e1 = r0             # the whole tiles over this rank's contiguous rows
    if r1 > r0:
        e0, e1 = (r0 // block) * block, min(-(-r1 // block) * block, nk)
    # every array the fit may hold, zero until it does
    held = dict.fromkeys(('metric_lu', 'X_rows', 'B_rows', 'aux_rows', 'X_ext',
                          'stream_tile', 'S_rows', 'FD_rows', 'shell_block',
                          'F_block', 'panel'), 0)

    def hold(name, nbytes):
        held[name] = max(held[name], int(nbytes))

    def total(arrays):
        return sum(int(a.nbytes) for a in arrays)

    def _say(msg):
        # Same reason as `fit_M_streaming`: hours at production size.
        if progress and rank == 0:
            print(f'[isdf {time.strftime("%H:%M:%S")}] {msg}', flush=True)

    _say(f'row-distributed fit: nk={nk} in {ntiles} tiles of {block} over '
         f'{nranks} rank(s), nao={nao} naux={naux}')
    _t = time.time()
    l_ao = _ao_l_labels(mol)
    second = np.where(l_ao <= l_max_second)[0]
    w = np.array([ANGULAR_WEIGHTS.get(l_ao[j], 1.0) for j in second])
    n2 = len(second)
    lu = scipy.linalg.lu_factor(auxmol.intor('int2c2e', aosym='s1'))
    if nranks > 1:
        # F_b is computed on one rank and read by all: one metric, rank 0's.
        lu = lockstep(lu, comm, check=True)
    hold('metric_lu', lu[0].nbytes)
    if timings is not None:
        timings['fit_collocation'] = time.time() - _t

    # Gram tiles, lower triangle, the collocation streamed by column tile.
    _t = time.time()
    S, d, X_own, B_own, P_own, X_ext, screen_ref = _gram_tiles(
        mol, auxmol, coords, tiles, owners, rank, second, w,
        regularization, (e0, e1), comm, hold)
    del B_own, P_own
    if timings is not None:
        timings['fit_gram'] = time.time() - _t
    _say(f'Gram tiles built: {total(S.values()) / 1e9:.2f} GB on rank 0')

    # The three-centre pass: F moves, each rank accumulates its own points.
    _t = time.time()
    t_comm = 0.0
    ao_loc = mol.ao_loc_nr()
    integrals = KeptIntegrals(mol, auxmol)
    frozen = None if layout is None else _frozen_columns(layout, second, w,
                                                          nao)
    blocks = ao_blocks(mol, nk, n2, naux, block_memory_gb)
    width = [ao_loc[s1] - ao_loc[s0] for s0, s1 in blocks]
    # Largest first, so a round's blocks cost alike; the order is fixed by the
    # molecule alone and is the order every point accumulates in.
    order = sorted(range(len(blocks)), key=lambda ib: (-width[ib], ib))
    R = {i: np.zeros((tiles[i][1] - tiles[i][0], naux)) for i in mine}
    hold('FD_rows', total(R.values()))
    n_mine = n_kept = 0
    # the co-densities in `row_map`'s threads, as in the replicated pass
    threads = openmp_threads()
    for c0 in range(0, len(order), nranks):
        batch = order[c0:c0 + nranks]
        spans = [(ao_loc[blocks[ib][0]], ao_loc[blocks[ib][1]])
                 for ib in batch]
        if frozen is not None:
            keeps = [frozen[a0 * n2:a1 * n2] for a0, a1 in spans]
        else:
            keeps = []
            for a0, a1 in spans:
                peak = np.zeros((a1 - a0) * n2)
                for i in mine:
                    np.maximum(peak, _pair_peaks(X_own[i], a0, a1, second, w,
                                                 threads), out=peak)
                keeps.append(peak)
            col_max = np.concatenate(keeps)
            _tc = time.time()
            reduce_max(col_max, comm)
            t_comm += time.time() - _tc
            offsets = np.cumsum([0] + [len(k) for k in keeps])
            keeps = [col_max[offsets[p]:offsets[p + 1]] > pair_tol * screen_ref
                     for p in range(len(batch))]
        F_own = None
        if rank < len(batch):
            n_mine += 1
            keep = keeps[rank]
            if keep.any():
                F_own = _block_coefficients(integrals, lu, blocks[batch[rank]],
                                            spans[rank][0], keep, second, w,
                                            hold)
        for pos, keep in enumerate(keeps):
            kept = np.flatnonzero(keep)
            if not len(kept):
                continue
            n_kept += len(kept)
            Ft = F_own if pos == rank else np.empty((len(kept), naux))
            hold('F_block', Ft.nbytes + (0 if F_own is None or pos == rank
                                         else F_own.nbytes))
            _tc = time.time()
            broadcast_rows(Ft, pos, comm)
            t_comm += time.time() - _tc
            mu = spans[pos][0] + kept // n2
            nu = second[kept % n2]
            wk = w[kept % n2]
            for i in mine:
                D_t = _pair_columns(X_own[i], mu, nu, wk, threads)
                R[i] += D_t @ Ft
                del D_t
            del Ft
        F_own = None
    del X_own, lu, integrals
    for i in mine:
        # auxiliary-function block, F = identity, added last as in the
        # replicated pass
        R[i] += _tile_collocation(auxmol, coords[slice(*tiles[i])])
        R[i] *= d[slice(*tiles[i]), None]
    if timings is not None:
        timings['fit_integrals'] = time.time() - _t
        timings['fit_integrals_reduce'] = t_comm
        timings['fit_blocks'] = n_mine
        timings['fit_blocks_total'] = len(blocks)
    _say(f'three-centre pass done: {len(blocks)} blocks, {n_kept:,} of '
         f'{nao * n2:,} columns survived screening')

    # Right-looking Cholesky, the forward substitution in the same sweep. Its
    # GEMMs are BLAS's; each update's subtraction -- one flop an element
    # against the GEMM's 2 `block`, which on a pool is as slow as the GEMM --
    # is numpy's and runs in `row_map`'s threads, the same difference an
    # element.
    _t = time.time()
    t_solve = 0.0
    diagonal, panels = {}, {}
    for j, (j0, j1) in enumerate(tiles):
        owner = j % nranks
        LT = np.empty((j1 - j0, j1 - j0))
        info = 0
        if rank == owner:
            # row tile j is down to its diagonal block
            L, info = scipy.linalg.lapack.dpotrf(S.pop(j), lower=1, clean=1)
            LT = L.T                                 # C order, L in F order
        info = broadcast(info, comm, root=owner)
        if info != 0:
            raise np.linalg.LinAlgError(
                f'Cholesky of the balanced Gram matrix failed at leading '
                f'minor {j0 + info}: it is not positive definite, which for a '
                f'Gram matrix plus a {regularization:g} shift means the grid '
                'is degenerate -- points coincide, or a whole sub-shell '
                'collapsed onto one radius.')
        broadcast_rows(LT, owner, comm)
        L = LT.T
        _ts = time.time()
        y = R[j] if rank == owner else np.empty((j1 - j0, naux))
        if rank == owner:
            _trsm(L, y.T, side=1, trans_a=1)             # y^T L^T = R_j^T
            diagonal[j] = L
        broadcast_rows(y, owner, comm)
        t_solve += time.time() - _ts
        below = [i for i in mine if i > j]
        Lcol = {}
        for i in below:
            Lcol[i] = np.ascontiguousarray(S[i][:, :j1 - j0])
            _trsm(L, Lcol[i].T, side=0, trans_a=0)       # L L_ij^T = S_ij^T
        _ts = time.time()
        for i in below:
            _subtract(R[i], Lcol[i] @ y, threads)
        t_solve += time.time() - _ts
        del y
        panel = np.empty((nk - j1, j1 - j0))
        for i in below:
            panel[tiles[i][0] - j1:tiles[i][1] - j1] = Lcol[i]
        allgather_ranges(panel, [[(tiles[i][0] - j1, tiles[i][1] - j1)
                                  for i in own if i > j] for own in owners],
                         comm)
        for i in below:
            # one GEMM per row tile; the consumed column block is dropped
            U = Lcol.pop(i) @ panel[:tiles[i][1] - j1].T
            S[i] = _difference(S[i][:, j1 - j0:], U, threads)
            del U
        if rank == owner:
            panels[j] = panel
        hold('panel', panel.nbytes)
        hold('S_rows', total(S.values()) + total(panels.values())
             + total(diagonal.values()))
        del panel
    # Backward substitution: z_j = L_jj^-T (y_j - sum_{k>j} L_kj^T z_k), the
    # sum pushed by each solved block into the rows that own a panel.
    _ts = time.time()
    for j in reversed(range(ntiles)):
        j0, j1 = tiles[j]
        owner = j % nranks
        z = R[j] if rank == owner else np.empty((j1 - j0, naux))
        if rank == owner:
            _trsm(diagonal.pop(j), z.T, side=1, trans_a=0)   # z^T L = y^T
            panels.pop(j, None)
        broadcast_rows(z, owner, comm)
        for i in mine:
            if i < j:
                i1 = tiles[i][1]
                _subtract(R[i], panels[i][j0 - i1:j1 - i1].T @ z, threads)
        del z
    for i in mine:
        R[i] *= d[slice(*tiles[i]), None]
    t_solve += time.time() - _ts
    if timings is not None:
        timings['fit_cholesky'] = time.time() - _t - t_solve
        timings['fit_solve'] = t_solve
    _say('row-distributed fit done')
    return RowFit(nk, block, comm, R, X_ext, (e0, e1), held)


def fit_rows_adjoint(mol, auxmol, coords, d_bar, layout, x_bar=None,
                     mo_coeff=None, l_max_second=2,
                     regularization=DEFAULT_REGULARIZATION,
                     block_memory_gb=4.0, comm=None, block=None):
    """The nuclear adjoint of `fit_rows`' factor D = M^T V^1/2 on the frozen
    `layout`, in the forward fit's own tiles: a `RowFitAdjoint`.

    d_bar: (nk, naux) adjoint on D, the same bits on every rank (a kernel's
    all-reduced output); x_bar, mo_coeff: the adjoint on X_mo = X_ao C and
    C, whose collocation adjoint rides on the same derivative collocation.

    The estimator is `fit_rows`' product form, and its reverse pass never
    forms an nk x nk or a test-set-wide array (Q = (F D^T)^T, R = d Q,
    Z = G^-1 R, M^T = d Z, G = d S d + reg, S = (X X^T) o (B B^T) + P P^T):

        MT_bar = D_bar V^1/2,  W = G^-1 d MT_bar,  Q_bar = d W,
        U = Q_bar V^-1,  H = Q_bar M + M^T Q_bar^T  (tile by tile)
        d_bar_k = MT_bar_k . Z_k + W_k . Q_k - (1/d_k) sum_l H_kl S_kl
        X_bar = -(H o B B^T) X,  B_bar = -(H o X X^T) B,
        P_bar = Q_bar - H P,  and the balancing's diagonal term
        (mu nu|P)_bar = w U^T D_pair,  V_bar = (V^1/2)'^T(M D_bar)
                        - (Q - P)^T U

    since G^-1 applied to the solve's adjoint is low rank, G_bar = -W Z^T,
    and the V^-1 inside F reaches the metric as a sum over the GRID. Every
    step is a call of a shape fixed by the tile indices, and nothing is
    summed across ranks: a sum over the grid is accumulated on one rank in
    tile order from tiles that travel (the metric's two, on rank 0), the
    Gram terms by the row tile's owner over the column tiles streamed in
    order, each block's (mu nu|P) adjoint by the block's owner over U
    streamed in order, and every (natm, 3) partial -- a tile's, a block's,
    a metric slab's -- is gathered and added in its fixed order. So the
    result is the same bits at every rank count, one rank included.

    Its arrays (tile t of `block` points owned by rank t % size):
      Gram S, then its factor L   this rank's lower row tiles, then its
                                  diagonal blocks and gathered panels
      X, B, P collocation         this rank's tiles (B, P dropped through
                                  the pass); column tiles streamed
      MT_bar, Q_bar, Q, U, M^T,   this rank's tiles, (rows, naux) each
      P_bar
      X_bar, B_bar                this rank's tiles
      M D_bar, (Q - P)^T U        rank 0, (naux, naux), from tiles sent to
                                  it one at a time
      V^1/2 and its adjoint       rank 0 (`metric_root`, an in-place
                                  eigendecomposition), slabs elsewhere
      F_b, (mu nu|P) of a block   its owner, the kept pairs alone, F_b
                                  broadcast as in the forward pass
      the (mu nu|P) adjoints      the owner's blocks of one batch of at
                                  most `block_memory_gb` per rank, U
                                  streamed once per batch
      (P|Q)' of a slab            the slab's owner, `block` rows
    D_bar and X_bar arrive whole; a rank reads its tiles of them, and rank 0
    D_bar whole for M D_bar.
    """
    comm = current_comm() if comm is None else comm
    rank, nranks = ((0, 1) if comm is None
                    else (comm.Get_rank(), comm.Get_size()))
    if nranks > 1:
        coords = lockstep(coords, comm)
    block = FIT_CHOLESKY_BLOCK if block is None else int(block)
    nao, naux, natm = mol.nao_nr(), auxmol.nao_nr(), mol.natm
    nk = len(coords)
    ntiles = -(-nk // block)
    tiles = [(t * block, min((t + 1) * block, nk)) for t in range(ntiles)]
    owners = [[int(t) for t in partition(ntiles, r, nranks)]
              for r in range(nranks)]
    mine = owners[rank]
    held = dict.fromkeys((
        'metric_lu', 'X_rows', 'B_rows', 'aux_rows', 'X_ext', 'stream_tile',
        'S_rows', 'panel', 'metric_root', 'metric_slab', 'MT_bar_rows',
        'Q_bar_rows', 'Q_rows', 'U_rows', 'MT_rows', 'X_bar_rows',
        'B_bar_rows', 'P_bar_rows', 'shell_block', 'F_block', 'root_gather',
        'column_tile', 'g_bar_batch', 'derivative_block', 'root_adjoint',
        'two_centre_slab'), 0)

    def hold(name, nbytes):
        held[name] = max(held[name], int(nbytes))

    l_ao = _ao_l_labels(mol)
    second = np.where(l_ao <= l_max_second)[0]
    w = np.array([ANGULAR_WEIGHTS.get(l_ao[j], 1.0) for j in second])
    n2 = len(second)
    lu = scipy.linalg.lu_factor(auxmol.intor('int2c2e', aosym='s1'))
    if nranks > 1:
        lu = lockstep(lu, comm, check=True)
    hold('metric_lu', lu[0].nbytes)
    frozen = _frozen_columns(layout, second, w, nao)
    ao_loc = mol.ao_loc_nr()
    blocks = ao_blocks(mol, nk, n2, naux, block_memory_gb)
    width = [ao_loc[s1] - ao_loc[s0] for s0, s1 in blocks]
    order = sorted(range(len(blocks)), key=lambda ib: (-width[ib], ib))
    size_of = {t: tiles[t][1] - tiles[t][0] for t in range(ntiles)}

    # the forward's Gram tiles; B and P re-evaluated where next read, the
    # same calls, so they are not held through the pass
    S, d, X_own, B_own, P_own, _, _ = _gram_tiles(
        mol, auxmol, coords, tiles, owners, rank, second, w, regularization,
        (0, 0), comm, hold)
    del B_own, P_own

    # MT_bar = D_bar V^1/2, the root on rank 0 and streamed in slabs
    MTb = {t: np.empty((size_of[t], naux)) for t in mine}
    root, failure = None, None
    if rank == 0:
        try:
            root, nbytes = metric_root(auxmol)
            hold('metric_root', nbytes)
        except (RuntimeError, np.linalg.LinAlgError) as err:
            failure = str(err)
    failure = broadcast(failure, comm)
    if failure is not None:
        raise RuntimeError(failure)
    for c0 in range(0, naux, block):
        c1 = min(c0 + block, naux)
        slab = _root_slab(root, slice(c0, c1), slice(0, naux), comm)
        hold('metric_slab', slab.nbytes)
        for t in mine:
            MTb[t][:, c0:c1] = d_bar[slice(*tiles[t])] @ slab.T
    del root
    hold('MT_bar_rows', _total_bytes(MTb.values()))

    # W = G^-1 d MT_bar, the forward substitution in the factorization sweep
    Qb = {t: MTb[t] * d[slice(*tiles[t]), None] for t in mine}
    hold('Q_bar_rows', _total_bytes(Qb.values()))
    diagonal, panels = _cholesky_tiles(S, tiles, owners, rank, comm, Qb,
                                       regularization, hold)
    del S
    _backward_tiles(diagonal, panels, Qb, tiles, owners, rank, comm)
    for t in mine:
        Qb[t] *= d[slice(*tiles[t]), None]
    # the three-centre pass: Q as the forward accumulates it, and the
    # adjoint of its pair columns, Q_bar F_b, onto the AO collocation
    Q = {t: np.zeros((size_of[t], naux)) for t in mine}
    hold('Q_rows', _total_bytes(Q.values()))
    aob = {t: np.zeros((size_of[t], nao)) for t in mine}
    hold('X_bar_rows', _total_bytes(aob.values()))
    integrals = KeptIntegrals(mol, auxmol)
    for c0 in range(0, len(order), nranks):
        batch = order[c0:c0 + nranks]
        spans = [(ao_loc[blocks[ib][0]], ao_loc[blocks[ib][1]])
                 for ib in batch]
        keeps = [frozen[a0 * n2:a1 * n2] for a0, a1 in spans]
        F_own = None
        if rank < len(batch) and keeps[rank].any():
            F_own = _block_coefficients(integrals, lu, blocks[batch[rank]],
                                        spans[rank][0], keeps[rank], second,
                                        w, hold)
        for pos, keep in enumerate(keeps):
            kept = np.flatnonzero(keep)
            if not len(kept):
                continue
            Ft = F_own if pos == rank else np.empty((len(kept), naux))
            hold('F_block', Ft.nbytes + (0 if F_own is None or pos == rank
                                         else F_own.nbytes))
            broadcast_rows(Ft, pos, comm)
            mu = spans[pos][0] + kept // n2
            nu = second[kept % n2]
            wk = w[kept % n2]
            for i in mine:
                D_t = X_own[i][:, mu] * X_own[i][:, nu]
                D_t *= wk[None, :]
                Q[i] += D_t @ Ft
                del D_t
                E = Qb[i] @ Ft.T
                E *= wk[None, :]
                _scatter_pairs(aob[i], E, X_own[i], mu, nu)
                del E
            del Ft
        F_own = None
    del integrals

    # U = Q_bar V^-1, the grid sum (Q - P)^T U on rank 0, then R = d Q
    U = {t: np.ascontiguousarray(scipy.linalg.lu_solve(lu, Qb[t].T).T)
         for t in mine}
    hold('U_rows', _total_bytes(U.values()))
    del lu
    VbarF = np.zeros((naux, naux)) if rank == 0 else None
    for t, got in _tiles_at_root([Q, U], tiles, rank, nranks, comm, hold):
        if got is not None:
            VbarF -= got[0].T @ got[1]
    P_own = {t: _tile_collocation(auxmol, coords[slice(*tiles[t])])
             for t in mine}
    d6 = {}
    for t in mine:
        Q[t] += P_own[t]                 # the auxiliary block, F = identity
        # W . Q with W = Q_bar / d
        d6[t] = np.einsum('kb,kb->k', Qb[t], Q[t]) / d[slice(*tiles[t])]
        Q[t] *= d[slice(*tiles[t]), None]

    # Z = G^-1 R on the kept factor, M^T = d Z: the forward's M^T
    _forward_tiles(diagonal, panels, Q, tiles, owners, rank, comm, hold)
    _backward_tiles(diagonal, panels, Q, tiles, owners, rank, comm,
                    keep=False)
    del diagonal, panels
    d8 = {}
    for t in mine:
        d8[t] = np.einsum('kb,kb->k', MTb[t], Q[t])
        Q[t] *= d[slice(*tiles[t]), None]
    MT = Q
    del MTb, Q
    hold('MT_rows', _total_bytes(MT.values()))
    Vhbar = np.zeros((naux, naux)) if rank == 0 else None
    for t, got in _tiles_at_root([MT], tiles, rank, nranks, comm, hold):
        if got is not None:
            Vhbar += got[0].T @ d_bar[slice(*tiles[t])]

    # the (mu nu|P) adjoints, w U^T D_pair, by the block's owner
    three = _three_centre_rows_adjoint(
        mol, auxmol, coords, U, X_own, tiles, owners, rank, comm, blocks,
        order, frozen, second, w, block_memory_gb * 1e9, hold)
    del U

    # the Gram matrix's adjoint, H = Q_bar M + M^T Q_bar^T, by row tile
    B_own = {t: X_own[t][:, second] * w[None, :] for t in mine}
    Xb = {t: np.zeros((size_of[t], nao)) for t in mine}
    Bb = {t: np.zeros((size_of[t], n2)) for t in mine}
    Pb = {t: np.zeros((size_of[t], naux)) for t in mine}
    dg = {t: np.zeros(size_of[t]) for t in mine}
    hold('B_bar_rows', _total_bytes(Bb.values()))
    hold('P_bar_rows', _total_bytes(Pb.values()))
    for k, (k0, k1) in enumerate(tiles):
        owner = k % nranks
        both = (np.hstack([MT[k], Qb[k]]) if rank == owner
                else np.empty((k1 - k0, 2 * naux)))
        broadcast_rows(both, owner, comm)
        hold('column_tile', both.nbytes)
        MT_k, Qb_k = both[:, :naux], both[:, naux:]
        if k in X_own:
            X_k, B_k, P_k = X_own[k], B_own[k], P_own[k]
        else:
            X_k = _tile_collocation(mol, coords[k0:k1])
            B_k = X_k[:, second] * w[None, :]
            P_k = _tile_collocation(auxmol, coords[k0:k1])
            hold('stream_tile', _total_bytes((X_k, B_k, P_k)))
        for i in mine:
            XX = X_own[i] @ X_k.T
            BB = B_own[i] @ B_k.T
            H = Qb[i] @ MT_k.T
            H += MT[i] @ Qb_k.T
            S_ik = XX * BB
            S_ik += P_own[i] @ P_k.T
            dg[i] -= np.einsum('kl,kl->k', H, S_ik)
            del S_ik
            BB *= H
            Xb[i] -= BB @ X_k
            XX *= H
            Bb[i] -= XX @ B_k
            Pb[i] -= H @ P_k
            del XX, BB, H
        del both, X_k, B_k, P_k
    # the balancing s = diag(S)^1/2, d = 1/s: its adjoint on diag(S)
    for t in mine:
        rows = slice(*tiles[t])
        dbar = d8[t] + d6[t] + dg[t] / d[rows]
        c = -dbar * d[rows] ** 3         # s_bar / s, s_bar = -d_bar d^2
        Xb[t] += (c * np.einsum('kj,kj->k', B_own[t], B_own[t]))[:, None] \
            * X_own[t]
        Bb[t] += (c * np.einsum('km,km->k', X_own[t], X_own[t]))[:, None] \
            * B_own[t]
        Pb[t] += c[:, None] * P_own[t]
        Pb[t] += Qb[t]                   # Q's auxiliary block
        aob[t] += Xb[t]
        aob[t][:, second] += Bb[t] * w[None, :]
    del Xb, Bb, d6, d8, dg, MT, Qb, B_own

    # the collocations' centres and points, one tile at a time
    ao_slices = [(p0, p1) for _, _, p0, p1 in mol.aoslice_by_atom()]
    aux_slices = [(q0, q1) for _, _, q0, q1 in auxmol.aoslice_by_atom()]
    fit_tile = np.zeros((ntiles, natm * 3))
    fit_points = np.zeros((nk, 3))
    with_coll = x_bar is not None
    coll_tile = np.zeros((ntiles, natm * 3)) if with_coll else None
    coll_points = np.zeros((nk, 3)) if with_coll else None
    for t in mine:
        rows = slice(*tiles[t])
        g = mol.eval_gto('GTOval_ip_sph', coords[rows])
        f = _centre_forces(g, aob[t], ao_slices)
        fit_points[rows] = np.einsum('xgm,gm->gx', g, aob[t])
        if with_coll:
            xa = x_bar[rows] @ mo_coeff.T
            coll_tile[t] = _centre_forces(g, xa, ao_slices).ravel()
            coll_points[rows] = np.einsum('xgm,gm->gx', g, xa)
            del xa
        del g
        g = auxmol.eval_gto('GTOval_ip_sph', coords[rows])
        f += _centre_forces(g, Pb[t], aux_slices)
        fit_points[rows] += np.einsum('xgm,gm->gx', g, Pb[t])
        fit_tile[t] = f.ravel()
        del g
    del aob, Pb, X_own, P_own
    tile_ranges = [[(t, t + 1) for t in own] for own in owners]
    point_ranges = [[tiles[t] for t in own] for own in owners]
    allgather_ranges(fit_tile, tile_ranges, comm)
    allgather_ranges(fit_points, point_ranges, comm)
    if with_coll:
        allgather_ranges(coll_tile, tile_ranges, comm)
        allgather_ranges(coll_points, point_ranges, comm)

    # the metric: (V^1/2)'s adjoint on rank 0, (P|Q)' by slab
    V_bar = None
    if rank == 0:
        sums = {'root': Vhbar, 'fit': VbarF}
        del Vhbar, VbarF
        V_bar = _metric_root_adjoint(auxmol, sums, hold)
    else:
        del Vhbar, VbarF
    two = _two_centre_rows_adjoint(auxmol, V_bar, block, rank, nranks, comm,
                                   hold)
    del V_bar

    fit_centre = np.zeros(natm * 3)
    for t in range(ntiles):
        fit_centre += fit_tile[t]
    fit_centre += three
    fit_centre += two
    coll_centre = None
    if with_coll:
        coll_centre = np.zeros(natm * 3)
        for t in range(ntiles):
            coll_centre += coll_tile[t]
        coll_centre = coll_centre.reshape(natm, 3)
    return RowFitAdjoint(fit_centre.reshape(natm, 3), fit_points,
                         coll_centre, coll_points, held)


def rows_transpose_product(rows, r0, x_bar, block=None, comm=None):
    """Y = X^T X_bar, (ncol, ncol), from this rank's `contiguous_block` rows
    of X, starting at grid row r0, with X_bar whole on every rank.

    The grid index runs in fixed tiles of `block` points; each tile of X is
    broadcast from the ranks whose rows hold it, and each rank adds
    X_bar[tile, slab]^T X[tile] into its own slabs of Y^T's rows (slab s of
    `block` columns to rank s % size) in tile order; the slabs are then
    gathered verbatim. So no rank holds X whole, nothing is summed across
    ranks, and Y is the same bits at every rank count -- not the bits of the
    whole product X^T X_bar, another blocking of the same sum.
    """
    block = FIT_CHOLESKY_BLOCK if block is None else int(block)
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    nk, ncol = x_bar.shape
    bounds = [contiguous_block(nk, r, size) for r in range(size)]
    slabs = [(c0, min(c0 + block, ncol)) for c0 in range(0, ncol, block)]
    mine = [slab for s, slab in enumerate(slabs) if s % size == rank]
    yt = np.zeros((ncol, ncol))
    for t0 in range(0, nk, block):
        t1 = min(t0 + block, nk)
        tile = np.empty((t1 - t0, ncol))
        for r, (s0, s1) in enumerate(bounds):
            lo, hi = max(t0, s0), min(t1, s1)
            if hi > lo:
                piece = (np.ascontiguousarray(rows[lo - r0:hi - r0])
                         if r == rank else np.empty((hi - lo, ncol)))
                broadcast_rows(piece, r, comm)
                tile[lo - t0:hi - t0] = piece
        for c0, c1 in mine:
            yt[c0:c1] += x_bar[t0:t1, c0:c1].T @ tile
        del tile
    allgather_ranges(yt, [[slab for s, slab in enumerate(slabs)
                           if s % size == r] for r in range(size)], comm)
    return yt.T


def _gram_tiles(mol, auxmol, coords, tiles, owners, rank, second, w,
                regularization, ext, comm, hold):
    """(S, d, X, B, P, X_ext, screen_ref) of `fit_rows`: the balanced,
    regularized Gram tiles S_ik = (X_i X_k^T) o (B_i B_k^T) + P_i P_k^T
    (i >= k) of the row tiles rank `rank` owns, with the collocation of
    its tiles, the rows [ext) of X and the screening reference; the
    column tiles are streamed, each evaluated once by every rank."""
    nao, nk = mol.nao_nr(), len(coords)
    mine = owners[rank]
    e0, e1 = ext
    X_own, B_own, P_own = {}, {}, {}
    for t in mine:
        X_own[t] = _tile_collocation(mol, coords[slice(*tiles[t])])
        B_own[t] = X_own[t][:, second] * w[None, :]
        P_own[t] = _tile_collocation(auxmol, coords[slice(*tiles[t])])
    hold('X_rows', _total_bytes(X_own.values()))
    hold('B_rows', _total_bytes(B_own.values()))
    hold('aux_rows', _total_bytes(P_own.values()))
    X_ext = np.empty((e1 - e0, nao))
    hold('X_ext', X_ext.nbytes)
    s_ao = np.zeros(nao)                 # max_k |chi_mu(r_k)|, for the screen
    # S[i]: row tile i's live columns, [0, end of tile i) until the sweep
    S = {i: np.empty((tiles[i][1] - tiles[i][0], tiles[i][1])) for i in mine}
    for k, (k0, k1) in enumerate(tiles):
        if k in X_own:
            X_k, B_k, P_k = X_own[k], B_own[k], P_own[k]
        else:
            X_k = _tile_collocation(mol, coords[k0:k1])
            B_k = X_k[:, second] * w[None, :]
            P_k = _tile_collocation(auxmol, coords[k0:k1])
            hold('stream_tile', _total_bytes((X_k, B_k, P_k)))
        np.maximum(s_ao, np.abs(X_k).max(axis=0), out=s_ao)
        if k0 < e1 and k1 > e0:
            X_ext[k0 - e0:k1 - e0] = X_k
        for i in mine:
            if i >= k:
                G = X_own[i] @ X_k.T
                G *= B_own[i] @ B_k.T
                G += P_own[i] @ P_k.T
                S[i][:, k0:k1] = G
        del X_k, B_k, P_k
    hold('S_rows', _total_bytes(S.values()))
    # `_screening_reference` over the whole grid: a maximum, exact by tile.
    screen_ref = float(s_ao.max() * (w * s_ao[second]).max())
    diag = np.zeros(nk)
    for i in mine:
        diag[slice(*tiles[i])] = np.diagonal(S[i][:, slice(*tiles[i])])
    allgather_ranges(diag, [[tiles[t] for t in own] for own in owners], comm)
    scale = np.sqrt(np.clip(diag, 0.0, None))
    scale[scale == 0] = 1.0
    d = 1.0 / scale
    for i in mine:
        i0, i1 = tiles[i]
        S[i] *= d[i0:i1, None]
        S[i] *= d[None, :i1]
        S[i][:, i0:i1][np.diag_indices(i1 - i0)] += regularization
    return S, d, X_own, B_own, P_own, X_ext, screen_ref


def _block_coefficients(integrals, lu, shells, a0, keep, second, w, hold):
    """F_b^T = (V^-1 (Q|rho) w)^T, (kept, naux) C order, of one shell block's
    kept pairs rho = (mu, nu): the arithmetic of the replicated pass on the
    integrals `KeptIntegrals` evaluates for them alone."""
    n2 = len(second)
    kept = np.flatnonzero(keep)
    E, call = integrals(shells, a0 + kept // n2, second[kept % n2])
    hold('shell_block', E.nbytes + call)
    # solved in place: E^T is the F-order right-hand side
    F_b = scipy.linalg.lu_solve(lu, E.T, overwrite_b=True)
    F_b *= np.tile(w, len(keep) // n2)[keep][None, :]
    return np.ascontiguousarray(F_b.T)


def _chunk_rows(ncol):
    """Rows of `ncol` doubles in one `FIT_ROW_CHUNK_BYTES` slab, at least 1."""
    return max(FIT_ROW_CHUNK_BYTES // (8 * max(int(ncol), 1)), 1)


def _pair_peaks(X, a0, a1, second, w, threads):
    """max_k |D[k, (mu, j)]| of a shell block's test co-densities
    D = X[:, mu] X[:, second[j]] w_j, mu in [a0, a1), one slab of rows of X
    at a time on `threads` threads, so the block is never whole.

    The bits np.maximum(D.max(0), -D.min(0)) gives the whole block, from one
    product and one maximum an element instead of two products and two
    extrema: |fl(fl(a b) w)| = fl(fl(|a| |b|) |w|), rounding being symmetric,
    and x -> fl(x |w|) is monotone, so the maximum over k is taken before the
    weight and the weight applied to the maxima alone; a maximum is exact in
    any order.
    """
    na, n2 = a1 - a0, len(second)
    whole = n2 == X.shape[1] and np.array_equal(second, np.arange(n2))
    step = _chunk_rows(na * n2)

    def peaks(r0, r1):
        buf = np.empty((min(step, r1 - r0), na, n2))
        hi = None
        for c0 in range(r0, r1, step):
            c1 = min(c0 + step, r1)
            rows = np.abs(X[c0:c1])
            pair = rows if whole else rows[:, second]
            prod = np.multiply(rows[:, a0:a1, None], pair[:, None, :],
                               out=buf[:c1 - c0]).reshape(c1 - c0, -1)
            if hi is None:
                hi = prod.max(axis=0)
            else:
                np.maximum(hi, prod.max(axis=0), out=hi)
        return hi

    parts = row_map(peaks, len(X), threads)
    hi = parts[0]
    for h in parts[1:]:
        np.maximum(hi, h, out=hi)
    return hi * np.tile(np.abs(w), na)


def _pair_columns(X, mu, nu, wk, threads):
    """D[:, kept] = X[:, mu] X[:, nu] wk, a block's kept test co-densities,
    (rows, kept) C order, one slab of rows at a time on `threads` threads:
    the elements the whole block holds for those columns. The kept pairs come
    in runs of one mu, so a run is one column times its nu columns."""
    out = np.empty((len(X), len(mu)))
    starts = np.flatnonzero(np.r_[True, mu[1:] != mu[:-1]])
    runs = list(zip(starts, np.r_[starts[1:], len(mu)]))
    step = _chunk_rows(len(mu))

    def fill(r0, r1):
        for c0 in range(r0, r1, step):
            c1 = min(c0 + step, r1)
            rows = np.ascontiguousarray(X[c0:c1])
            for s, e in runs:
                np.multiply(rows[:, mu[s], None], rows[:, nu[s:e]],
                            out=out[c0:c1, s:e])
            out[c0:c1] *= wk

    row_map(fill, len(X), threads)
    return out


def _kept_integrals(e3c, i, nu, threads):
    """(kept, naux) C order, row c the (i_c nu_c|P) of a block's (a, nao,
    naux) F-order integrals: one gather in slabs of P, each reading the
    contiguous (nu, i) planes of its P, on `threads` threads."""
    a, nao, naux = e3c.shape
    planes = e3c.T.reshape(naux, nao * a)      # a view: libcint writes P last
    flat = nu * a + i
    out = np.empty((len(flat), naux))
    step = _chunk_rows(len(flat))

    def fill(p0, p1):
        for q0 in range(p0, p1, step):
            q1 = min(q0 + step, p1)
            out[:, q0:q1] = np.take(planes[q0:q1], flat, axis=1).T

    row_map(fill, naux, threads)
    return out


def _accumulate(acc, part, threads):
    """acc += part, element by element, rows on `threads` threads."""
    def add(r0, r1):
        np.add(acc[r0:r1], part[r0:r1], out=acc[r0:r1])

    row_map(add, len(acc), threads)


def _subtract(acc, part, threads):
    """acc -= part, element by element, rows on `threads` threads."""
    def sub(r0, r1):
        np.subtract(acc[r0:r1], part[r0:r1], out=acc[r0:r1])

    row_map(sub, len(acc), threads)


def _difference(a, b, threads):
    """a - b, a new C-order array, rows on `threads` threads."""
    out = np.empty(b.shape)

    def sub(r0, r1):
        np.subtract(a[r0:r1], b[r0:r1], out=out[r0:r1])

    row_map(sub, len(out), threads)
    return out


def _scale_columns(X, d, threads):
    """X *= d[None, :], element by element, rows on `threads` threads."""
    def scale(r0, r1):
        np.multiply(X[r0:r1], d[None, :], out=X[r0:r1])

    row_map(scale, len(X), threads)


def _gram_lower(S, A, B, P, rows, threads):
    """S's lower block triangle over row blocks of `rows`,
    (A A^T) o (B B^T) + P P^T: the first product written by BLAS into S
    itself, the other two into two buffers made once, then combined a slab
    of rows at a time on `threads` threads -- each element the three whole
    products' bits, multiplied and then added."""
    nk = len(S)
    buffers = np.empty((2, min(rows, nk), nk))
    for i0 in range(0, nk, rows):
        i1 = min(i0 + rows, nk)
        out = S[i0:i1, :i1]
        q, x = buffers[0, :i1 - i0, :i1], buffers[1, :i1 - i0, :i1]
        np.matmul(A[i0:i1], A[:i1].T, out=out)
        np.matmul(B[i0:i1], B[:i1].T, out=q)
        np.matmul(P[i0:i1], P[:i1].T, out=x)       # auxiliary block
        _hadamard_sum(out, q, x, threads)


def _hadamard_sum(out, q, x, threads):
    """out = out o q + x, element by element, a slab of rows at a time on
    `threads` threads: the product rounded, then the sum."""
    step = _chunk_rows(out.shape[1])

    def combine(r0, r1):
        for c0 in range(r0, r1, step):
            c1 = min(c0 + step, r1)
            np.multiply(out[c0:c1], q[c0:c1], out=out[c0:c1])
            np.add(out[c0:c1], x[c0:c1], out=out[c0:c1])

    row_map(combine, len(out), threads)


def _balanced_upper(S, d, rows, threads):
    """S[r, c] d_r d_c over the diagonal blocks of `_gram_lower`'s `rows` and
    everything right of them, which is first the lower block triangle's
    transpose: the bits the whole mirror and then the row and the column
    scaling give there. Rows on `threads` threads, each writing its own and
    reading only columns left of the blocks still to come."""
    nk = len(S)
    for i0 in range(0, nk, rows):
        i1 = min(i0 + rows, nk)

        def balance(r0, r1):
            _balance_rows(S, d, i0 + r0, i0 + r1, i0, i1)

        row_map(balance, i1 - i0, threads)


def _balance_rows(S, d, r0, r1, i0, i1):
    """Rows [r0, r1) of `_balanced_upper` in the diagonal block [i0, i1), in
    square tiles of `FIT_TRANSPOSE_TILE`, so a transposed read stays in
    cache."""
    tile = FIT_TRANSPOSE_TILE
    for t0 in range(r0, r1, tile):
        t1 = min(t0 + tile, r1)
        rows = d[t0:t1, None]
        own = S[t0:t1, i0:i1]
        np.multiply(own, rows, out=own)
        np.multiply(own, d[None, i0:i1], out=own)
        for c0 in range(i1, len(S), tile):
            c1 = min(c0 + tile, len(S))
            out = S[t0:t1, c0:c1]
            np.multiply(S[c0:c1, t0:t1].T, rows, out=out)
            np.multiply(out, d[None, c0:c1], out=out)


def _cholesky_tiles(S, tiles, owners, rank, comm, rhs, regularization, hold):
    """(diagonal, panels): `fit_rows`' right-looking Cholesky of the Gram
    tiles S (consumed), the forward substitution of `rhs` riding in the
    sweep as it does there, and the factor KEPT for later solves: the
    diagonal block and the gathered panel below it of every tile this rank
    owns, the same arrays and the same calls as that sweep's."""
    nk, nranks = tiles[-1][1], len(owners)
    mine = owners[rank]
    ncol = next(iter(rhs.values())).shape[1] if rhs else 0
    ncol = broadcast(ncol, comm)
    diagonal, panels = {}, {}
    for j, (j0, j1) in enumerate(tiles):
        owner = j % nranks
        LT = np.empty((j1 - j0, j1 - j0))
        info = 0
        if rank == owner:
            L, info = scipy.linalg.lapack.dpotrf(S.pop(j), lower=1, clean=1)
            LT = L.T
        info = broadcast(info, comm, root=owner)
        if info != 0:
            raise np.linalg.LinAlgError(
                f'Cholesky of the balanced Gram matrix failed at leading '
                f'minor {j0 + info}: it is not positive definite, which for a '
                f'Gram matrix plus a {regularization:g} shift means the grid '
                'is degenerate.')
        broadcast_rows(LT, owner, comm)
        L = LT.T
        y = rhs[j] if rank == owner else np.empty((j1 - j0, ncol))
        if rank == owner:
            _trsm(L, y.T, side=1, trans_a=1)
            diagonal[j] = L
        broadcast_rows(y, owner, comm)
        below = [i for i in mine if i > j]
        Lcol = {}
        for i in below:
            Lcol[i] = np.ascontiguousarray(S[i][:, :j1 - j0])
            _trsm(L, Lcol[i].T, side=0, trans_a=0)
        for i in below:
            rhs[i] -= Lcol[i] @ y
        del y
        panel = np.empty((nk - j1, j1 - j0))
        for i in below:
            panel[tiles[i][0] - j1:tiles[i][1] - j1] = Lcol[i]
        allgather_ranges(panel, [[(tiles[i][0] - j1, tiles[i][1] - j1)
                                  for i in own if i > j] for own in owners],
                         comm)
        for i in below:
            U = Lcol.pop(i) @ panel[:tiles[i][1] - j1].T
            S[i] = S[i][:, j1 - j0:] - U
            del U
        if rank == owner:
            panels[j] = panel
        hold('panel', panel.nbytes)
        hold('S_rows', _total_bytes(S.values()) + _total_bytes(panels.values())
             + _total_bytes(diagonal.values()))
        del panel
    return diagonal, panels


def _forward_tiles(diagonal, panels, rhs, tiles, owners, rank, comm, hold):
    """rhs <- L^-1 rhs on the kept factor: the owner of tile j solves its
    block and broadcasts it with its panel, and every rank updates its tiles
    below with the call `_cholesky_tiles` makes there."""
    nk, nranks = tiles[-1][1], len(owners)
    mine = owners[rank]
    ncol = next(iter(rhs.values())).shape[1] if rhs else 0
    ncol = broadcast(ncol, comm)
    for j, (j0, j1) in enumerate(tiles):
        owner = j % nranks
        y = rhs[j] if rank == owner else np.empty((j1 - j0, ncol))
        if rank == owner:
            _trsm(diagonal[j], y.T, side=1, trans_a=1)
        broadcast_rows(y, owner, comm)
        panel = panels[j] if rank == owner else np.empty((nk - j1, j1 - j0))
        broadcast_rows(panel, owner, comm)
        hold('panel', panel.nbytes)
        for i in mine:
            if i > j:
                rhs[i] -= panel[tiles[i][0] - j1:tiles[i][1] - j1] @ y
        del y, panel


def _backward_tiles(diagonal, panels, rhs, tiles, owners, rank, comm,
                    keep=True):
    """rhs <- L^-T rhs on the kept factor, `fit_rows`' backward sweep: each
    solved block broadcast by its owner and pushed into the rows that own a
    panel. keep=False drops the factor as it goes, as that sweep does."""
    nranks = len(owners)
    mine = owners[rank]
    ncol = next(iter(rhs.values())).shape[1] if rhs else 0
    ncol = broadcast(ncol, comm)
    for j in reversed(range(len(tiles))):
        j0, j1 = tiles[j]
        owner = j % nranks
        z = rhs[j] if rank == owner else np.empty((j1 - j0, ncol))
        if rank == owner:
            _trsm(diagonal[j] if keep else diagonal.pop(j), z.T, side=1,
                  trans_a=0)
            if not keep:
                panels.pop(j, None)
        broadcast_rows(z, owner, comm)
        for i in mine:
            if i < j:
                i1 = tiles[i][1]
                rhs[i] -= panels[i][j0 - i1:j1 - i1].T @ z
        del z


def _tiles_at_root(arrays, tiles, rank, nranks, comm, hold):
    """Yield (t, [a[t] for a in arrays]) for every tile in order, on rank 0
    (None elsewhere): the owner sends its tile's rows to rank 0 alone, one
    `exchange_blocks` per tile, verbatim, so rank 0 holds one tile at a
    time."""
    ncol = [a[next(iter(a))].shape[1] if a else 0 for a in arrays]
    ncol = broadcast(ncol, comm)
    width = sum(ncol)
    for t, (t0, t1) in enumerate(tiles):
        owner = t % nranks
        send = [np.empty((0, width)) for _ in range(nranks)]
        if rank == owner:
            send[0] = np.hstack([a[t] for a in arrays])
        shapes = [(0, width)] * nranks
        if rank == 0:
            shapes[owner] = (t1 - t0, width)
        got = exchange_blocks(send, shapes, comm)[owner if rank == 0 else 0]
        hold('root_gather', got.nbytes if rank == 0 else 0)
        del send
        if rank != 0:
            yield t, None
            continue
        pieces, at = [], 0
        for n in ncol:
            pieces.append(got[:, at:at + n])
            at += n
        yield t, pieces

def _scatter_pairs(out, E, X, mu, nu):
    """out[:, m] += sum over the pairs (m, n) of E X[:, n], and
    out[:, n] += E X[:, m]: an adjoint E on the pair columns
    X[:, mu] X[:, nu] onto the AO collocation, one mu's run at a time."""
    starts = np.flatnonzero(np.r_[True, mu[1:] != mu[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(mu)]):
        m, nus = mu[a], nu[a:b]
        out[:, m] += np.einsum('kp,kp->k', E[:, a:b], X[:, nus])
        out[:, nus] += E[:, a:b] * X[:, m][:, None]


def _centre_forces(g, bar, slices):
    """(natm, 3) of sum_{k mu} bar[k, mu] chi_mu(r_k) through the centres:
    g = GTOval_ip = +d chi/dr, whose nuclear derivative is its negative."""
    f = np.zeros((len(slices), 3))
    for ia, (p0, p1) in enumerate(slices):
        f[ia] = -np.einsum('xgm,gm->x', g[:, :, p0:p1], bar[:, p0:p1])
    return f


def _three_centre_rows_adjoint(mol, auxmol, coords, U, X_own, tiles, owners,
                               rank, comm, blocks, order, frozen, second, w,
                               budget, hold):
    """(natm * 3,) of sum_p sum_P g[p, P] d(mu_p nu_p|P)/dR, with
    g_p = w_p U^T D[:, p] the adjoint on the kept pairs' integrals.

    Block order[c] belongs to rank c % size, as in the forward pass. Its
    owner accumulates g over U streamed tile by tile in grid order, in
    batches of consecutive rounds whose accumulators stay within `budget`
    bytes on every rank, then evaluates the kept pairs' derivative integrals
    (`KeptIntegrals` of int3c2e_ip1 and _ip2) and contracts them. The nu
    centre's derivative is -(the mu and P centres'), by translational
    invariance of (mu nu|P), and the integrals are evaluated in chunks of
    pairs holding `THREE_CENTER_BLOCK_BYTES` of each. Each block's (natm, 3)
    is gathered and the blocks are added in `order`."""
    nranks, n2 = len(owners), len(second)
    nao, naux, natm = mol.nao_nr(), auxmol.nao_nr(), mol.natm
    ao_loc = mol.ao_loc_nr()
    ao_atom = np.empty(nao, dtype=int)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        ao_atom[p0:p1] = ia
    aux_atom = np.empty(naux, dtype=int)
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        aux_atom[q0:q1] = ia
    pairs = {}
    for ib, (s0, s1) in enumerate(blocks):
        a0, a1 = ao_loc[s0], ao_loc[s1]
        kept = np.flatnonzero(frozen[a0 * n2:a1 * n2])
        pairs[ib] = (a0 + kept // n2, second[kept % n2], w[kept % n2])
    batches, current, load = [], [], np.zeros(nranks)
    for c0 in range(0, len(order), nranks):
        add = np.zeros(nranks)
        for pos, ib in enumerate(order[c0:c0 + nranks]):
            add[pos] = len(pairs[ib][0]) * naux * 8
        if current and np.any(load + add > budget):
            batches.append(current)
            current, load = [], np.zeros(nranks)
        current.extend(order[c0:c0 + nranks])
        load += add
    if current:
        batches.append(current)
    owner_of = {ib: c % nranks for c, ib in enumerate(order)}
    # pairs per derivative evaluation: three components of (kept, naux)
    chunk = max(1, THREE_CENTER_BLOCK_BYTES // (3 * naux * 8))
    first = KeptIntegrals(mol, auxmol, 'int3c2e_ip1', comp=3)
    other = KeptIntegrals(mol, auxmol, 'int3c2e_ip2', comp=3)
    forces = np.zeros((len(blocks), natm * 3))
    for batch in batches:
        own = [ib for ib in batch
               if owner_of[ib] == rank and len(pairs[ib][0])]
        acc = {ib: np.zeros((len(pairs[ib][0]), naux)) for ib in own}
        hold('g_bar_batch', _total_bytes(acc.values()))
        for t, (t0, t1) in enumerate(tiles):
            owner = t % nranks
            U_t = U[t] if rank == owner else np.empty((t1 - t0, naux))
            broadcast_rows(U_t, owner, comm)
            if not acc:
                continue
            if t in X_own:
                X_t = X_own[t]
            else:
                X_t = _tile_collocation(mol, coords[t0:t1])
                hold('stream_tile', X_t.nbytes)
            for ib in own:
                mu, nu, wk = pairs[ib]
                D_t = X_t[:, mu] * X_t[:, nu]
                D_t *= wk[None, :]
                acc[ib] += D_t.T @ U_t
                del D_t
            del X_t
        for ib in own:
            mu, nu, wk = pairs[ib]
            g = acc.pop(ib)
            g *= wk[:, None]
            f = np.zeros((natm, 3))
            for p0 in range(0, len(mu), chunk):
                c = slice(p0, min(p0 + chunk, len(mu)))
                I, call = first(blocks[ib], mu[c], nu[c])
                hold('derivative_block', I.nbytes + call)
                t_mu = np.einsum('xpP,pP->xp', I, g[c])    # -d/dR_mu
                del I
                I, call = other(blocks[ib], mu[c], nu[c])
                hold('derivative_block', I.nbytes + call)
                t_aux = np.einsum('xpP,pP->xP', I, g[c])   # -d/dR_P
                t_nu = t_mu + np.einsum('xpP,pP->xp', I, g[c])
                del I
                np.add.at(f, ao_atom[mu[c]], -t_mu.T)
                np.add.at(f, ao_atom[nu[c]], t_nu.T)
                np.add.at(f, aux_atom, -t_aux.T)
            del g
            forces[ib] = f.ravel()
    allgather_ranges(forces, [[(ib, ib + 1) for ib in order
                               if owner_of[ib] == r] for r in range(nranks)],
                     comm)
    total = np.zeros(natm * 3)
    for ib in order:
        total += forces[ib]
    return total


def _two_centre_rows_adjoint(auxmol, V_bar, block, rank, nranks, comm,
                             hold):
    """(natm * 3,) of sum_PQ V_bar[P, Q] d(P|Q)/dR, V_bar held on rank 0
    alone (None elsewhere). Its symmetric part travels in slabs of
    aux shells of at most `block` functions, slab s to rank s % size, which
    evaluates that slab's (grad P|Q) alone; the slabs' (natm, 3) are
    gathered and added in slab order."""
    naux, natm = auxmol.nao_nr(), auxmol.natm
    aux_loc = auxmol.ao_loc_nr()
    aux_atom = np.empty(naux, dtype=int)
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        aux_atom[q0:q1] = ia
    slabs, sh0 = [], 0
    while sh0 < auxmol.nbas:
        sh1 = sh0 + 1
        while (sh1 < auxmol.nbas
               and int(aux_loc[sh1 + 1] - aux_loc[sh0]) <= block):
            sh1 += 1
        slabs.append((sh0, sh1))
        sh0 = sh1
    forces = np.zeros((len(slabs), natm * 3))
    for s, (sh0, sh1) in enumerate(slabs):
        p0, p1 = aux_loc[sh0], aux_loc[sh1]
        slab = (np.ascontiguousarray(0.5 * (V_bar[p0:p1] + V_bar[:, p0:p1].T))
                if rank == 0 else np.empty((p1 - p0, naux)))
        broadcast_rows(slab, 0, comm)
        if rank == s % nranks:
            v1 = auxmol.intor('int2c2e_ip1', comp=3,
                              shls_slice=(sh0, sh1, 0, auxmol.nbas))
            hold('two_centre_slab', slab.nbytes + v1.nbytes)
            # +(grad P|Q), so the nuclear derivative carries the minus
            t_P = -2.0 * np.einsum('xPQ,PQ->xP', v1, slab)
            f = np.zeros((natm, 3))
            np.add.at(f, aux_atom[p0:p1], t_P.T)
            forces[s] = f.ravel()
            del v1
        del slab
    allgather_ranges(forces, [[(s, s + 1) for s in range(len(slabs))
                               if s % nranks == r] for r in range(nranks)],
                     comm)
    total = np.zeros(natm * 3)
    for s in range(len(slabs)):
        total += forces[s]
    return total


def _metric_root_adjoint(auxmol, sums, hold):
    """V_bar = (V^1/2)'s adjoint of sums['root'] plus sums['fit'], both
    (naux, naux) and taken out of `sums` so that each is dropped as soon as
    it is read: at most four metric-sized arrays at once.

    The Frechet derivative of the root, `metric_root`'s, is the Hadamard
    multiplier 1 / (w_i^1/2 + w_j^1/2) in V's eigenbasis, self-adjoint,
    over the spectrum `_metric_spectrum` keeps.
    """
    V = np.asfortranarray(auxmol.intor('int2c2e', aosym='s1'))
    w, Z = scipy.linalg.eigh(V, lower=True, overwrite_a=True,
                             check_finite=False, driver='evr')
    hold('root_adjoint', 4 * Z.nbytes)
    del V
    keep = _metric_spectrum(w, False)
    root = np.where(keep, np.sqrt(np.where(keep, w, 0.0)), np.inf)
    E = Z.T @ sums.pop('root')
    E = E @ Z
    E += E.T
    E *= 0.5
    E /= root[:, None] + root[None, :]    # a dropped direction carries none
    Y = Z @ E
    del E
    V_bar = Y @ Z.T
    del Y, Z
    V_bar += sums.pop('fit')
    return V_bar

def _total_bytes(arrays):
    """Bytes of the arrays held together."""
    return sum(int(a.nbytes) for a in arrays)


def _frozen_columns(layout, second, w, nao):
    """The kept mask over the pair index mu * n2 + (position of nu in
    `second`) of a frozen `test_set_layout`, checked against this test set:
    its nu in the l <= l_max_second set, its weights this set's."""
    mu, nu, weight = (np.asarray(a) for a in layout)
    at = np.full(nao, -1)
    at[second] = np.arange(len(second))
    if len(nu) and (at[nu].min() < 0 or not np.array_equal(weight,
                                                           w[at[nu]])):
        raise ValueError('the frozen layout is not a subsequence of this '
                         'test set: a second index outside l <= l_max_second '
                         'or another angular weighting')
    out = np.zeros(nao * len(second), dtype=bool)
    out[mu * len(second) + at[nu]] = True
    return out

def _tile_collocation(mol, coords):
    """chi_mu(r_k) on one tile of points, C order: the one call every rank
    makes for that tile, so its bits do not depend on who makes it."""
    return np.ascontiguousarray(mol.eval_gto('GTOval_sph', coords))


def _trsm(L, bt, side, trans_a):
    """bt <- op(L)^-1 bt (side 0) or bt op(L)^-1 (side 1), L lower
    triangular in F order and bt an F-order view of a tile's rows: BLAS
    solves into the caller's own memory."""
    x = scipy.linalg.blas.dtrsm(1.0, L, bt, side=side, lower=1,
                                trans_a=trans_a, overwrite_b=1)
    if not np.shares_memory(x, bt):
        bt[...] = x


def _root_slab(matrix, rows, cols, comm):
    """Rank 0's `matrix[rows, cols]` on every rank, C order: one broadcast,
    `matrix` read on rank 0 alone."""
    shape = (rows.stop - rows.start, cols.stop - cols.start)
    slab = (np.ascontiguousarray(matrix[rows, cols])
            if comm is None or comm.Get_rank() == 0 else np.empty(shape))
    return broadcast_rows(slab, 0, comm)

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
                       pair_tol=DEFAULT_PAIR_TOL, comm=None, timings=None,
                       with_Z=True):
    """Returns (X, Z, M) with X[k, mu] = chi_mu(r_k) and Z = M^T V M.

    with_Z=False returns None for Z: an (nk, nk) array, 111 GB and
    2 nk^2 naux + 2 nk naux^2 = 9.7e14 flops at the chlorophyllide hexamer
    /cc-pVTZ (nk 117762, naux 28236) on every rank, for a caller that forms
    D = M^T V^1/2 instead and never reads it.

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

    comm spreads the streaming path's three-centre pass -- the hours-long part
    at production size -- over ranks, striping the shell-pair blocks and
    reducing one (naux, nk) accumulator, which every rank holds in full. Each
    rank comes back with the same X, Z and M, so a caller downstream of this
    never branches on it. None is `current_comm()`; without either nothing is
    probed and no collective is called: the serial path is bit for bit what it
    was. The reference path (streaming=False) ignores it and every rank builds
    the whole fit, since its output is a concatenation over blocks rather than
    a sum.

    timings: dict, forwarded into `fit_M_streaming` for its own phases; this
    function adds `fit_assembly`, the V-metric rebuild and Z = M^T V M where
    `with_Z` asks for them, and the returned X's collocation.
    `separable_factors` adds its own tail (the
    dressed metric, the MO/AO projections, `replicate_factors`) onto the same
    key, so `fit_assembly` is the two functions' combined tail, not this
    one's alone. The reference
    path (streaming=False) never fills `timings` beyond that: it has no phases
    of its own here to report.
    """
    comm = current_comm() if comm is None else comm
    if auxmol is None:
        auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis)
    if streaming:
        M = fit_M_streaming(mol, auxmol, coords, l_max_second=l_max_second,
                            regularization=regularization,
                            block_memory_gb=block_memory_gb, pair_tol=pair_tol,
                            comm=comm, timings=timings)
    else:
        D, F = build_D_F(mol, auxmol, coords, l_max_second=l_max_second,
                         block_memory_gb=block_memory_gb, pair_tol=pair_tol)
        M = fit_M(D, F, regularization)                  # (naux, nk)
    _t = time.time()
    Z = None
    if with_Z:
        V = auxmol.intor('int2c2e', aosym='s1')
        Z = M.T @ V @ M
    X = mol.eval_gto('GTOval_sph', coords)               # (nk, nao)
    if timings is not None:
        timings['fit_assembly'] = time.time() - _t
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
    keep = _metric_spectrum(w, kernel is not None)
    return (v[:, keep] * np.sqrt(w[keep])) @ v[:, keep].T


def metric_root(auxmol, environment=None):
    """(V + vtilde)^1/2 in `aux_metric_sqrt`'s gauge, and the most bytes of
    (naux, naux) arrays it held at once: two, where `numpy.linalg.eigh` and
    that function's tail hold up to seven.

    LAPACK's dsyevr overwrites the metric and returns the eigenvectors Z
    beside it with workspace linear in naux; Z is scaled in place to
    B = Z w^1/4 over the kept spectrum and the root is B B^T. Another
    eigensolver than `aux_metric_sqrt`'s, so the same root to rounding, not
    its bits.
    """
    V = np.asfortranarray(auxmol.intor('int2c2e', aosym='s1'))
    kernel = None if environment is None else environment.aux_kernel(auxmol)
    dressed = kernel is not None
    if dressed:
        V += kernel
        del kernel
    w, Z = scipy.linalg.eigh(V, lower=True, overwrite_a=True,
                             check_finite=False, driver='evr')
    held = V.nbytes + Z.nbytes
    del V
    keep = _metric_spectrum(w, dressed)
    Z *= np.sqrt(np.sqrt(np.where(keep, w, 0.0)))[None, :]
    root = Z @ Z.T
    return root, max(held, Z.nbytes + root.nbytes)


def _metric_spectrum(w, dressed):
    """The eigenvalues w of an auxiliary metric that its square root keeps.

    Below `AUX_METRIC_ROOT_FLOOR` of the largest a direction is the numerical
    null space and is dropped. A dressed metric v + vtilde is still a
    positive kernel, so a negative eigenvalue past
    `AUX_METRIC_INDEFINITE_TOL` is refused.
    """
    if dressed and w.min() < -AUX_METRIC_INDEFINITE_TOL * w.max():
        raise RuntimeError(
            f"the screened auxiliary metric v + vtilde is indefinite "
            f"(smallest eigenvalue {w.min():.3e}): the reaction field "
            f"over-screens the bare interaction. Check eps and the cavity "
            f"(lebedev_order / vdw_scale).")
    return w > AUX_METRIC_ROOT_FLOOR * w.max()


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


#: Order of the sub-shells in the flat optimization vector. One spelling: a
#: gradient concatenated in a different order is silently a different variable.
_SHELL_ORDER = ('A1', 'A2', 'A3', 'B1')


def _radii_from_flat(x, counts):
    out, i = {}, 0
    for name in _SHELL_ORDER:
        n = counts.get(name, 0)
        if n:
            out[name] = np.exp(x[i:i + n])
            i += n
    return out


def _flat_from_radii(radii):
    return np.concatenate([np.log(radii[n]) for n in _SHELL_ORDER
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


RADII_TABLE_SCHEMA = 2


def shipped_radii():
    """THE radii table: one row per grid a caller can ask for.

    Optimized atomic radii travel WITH the source, not in a scratch cache.
    `optimize_atomic_radii`'s on-disk cache is gitignored, so a clean checkout
    re-optimizes from scratch -- and that optimizer is a local descent whose
    result is not reproducible (see the note on the cache below). Measured
    consequence: the pinned BSE roots in tests/test_bse_isdf_driver.py move by
    6.5-8.7 meV between a populated and an empty cache, i.e. a fresh clone fails
    its own regression tests. Shipping the table fixes the reproducibility;
    it does NOT make a bad grid good, which is why each row carries both the
    `fit_error` and the `score_mHa_per_atom` it was accepted at. Read those
    before trusting a row: a fit error approaching 1 is a fit that has failed,
    not a grid that is merely coarse.

    Rows are keyed on the PHYSICS -- `element|basis|auxbasis|A1,A2,A3,B1` --
    and on nothing else. The optimizer recipe that found a grid (`n_start`,
    `r_max`) is recorded per row under `optimizer` but is not part of the key:
    it is how a grid was found, not what was asked for. Keying on it left the
    table with up to nine rows for one physical request, to be chosen between
    at run time by `search_r_max`, a per-element heuristic that knows nothing
    about grid quality and picked the worse box in 272 of 612 cases. Those axes
    were collapsed by the exchange-probe score.

    Returns {key: row}; use `shipped_radii_lookup` rather than indexing.
    """
    path = os.path.join(os.path.dirname(__file__), 'data', 'optimized_radii.json')
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    schema = raw.get('schema')
    if schema != RADII_TABLE_SCHEMA:
        raise ValueError(
            f'{path} declares schema {schema!r}, and this code reads '
            f'{RADII_TABLE_SCHEMA}. A pre-consolidation table is keyed on the '
            f'optimizer recipe as well as the physics, so reading it here '
            f'would miss on every lookup and silently re-optimize instead.')
    return raw['rows']


def _shipped_key(element, basis, auxbasis, counts):
    """Table key: the physics, and nothing about how the grid was found."""
    return f'{element}|{basis}|{auxbasis}|' + \
           ','.join(str(int((counts or {}).get(name, 0))) for name in _SHELL_ORDER)


def _tabulated_counts(element, basis, auxbasis):
    """The count tuples the table holds for this atom, with their point counts.

    What a caller needs when its own tuple missed: the miss is almost always a
    tuple nobody ever optimized rather than a corrupt table.
    """
    out = []
    for key, row in shipped_radii().items():
        el, bas, aux, counts = key.split('|')
        if (el, bas, aux) != (str(element), str(basis), str(auxbasis)):
            continue
        out.append((row['points'], f'({counts}) {row["points"]} pts'
                                   + ('' if row.get('gated', True)
                                      else ' [ungated]')))
    return [text for _, text in sorted(out)] or 'none for this element and basis'


def shipped_radii_lookup(element, basis, auxbasis, counts):
    """(radii, fit_error, origin) from the table, or None.

    `origin` comes back because it is part of the GRID, not of the recipe: a
    row carrying it places one extra point at the nucleus, so honouring the
    radii while dropping the flag builds a 306-point grid where the row
    describes a 307-point one. Only the transcribed Duchemin & Blase rows set
    it, and they are the rows most likely to be asked for by someone
    reproducing a published number.
    """
    row = shipped_radii().get(_shipped_key(element, basis, auxbasis, counts))
    if row is None:
        return None
    return ({k: np.array(v) for k, v in row['radii'].items()},
            row['fit_error'], bool(row.get('origin', False)))


def atomic_grid(element, basis, auxbasis=None, counts=None):
    """The ONE lookup for a tabulated atomic grid. Returns (radii, origin).

    What a grid BUILDER needs, as opposed to `optimize_atomic_radii`, which is
    the optimizer and takes recipe arguments this does not: the radii and
    whether the nuclear cusp is sampled, which together are the grid.
    """
    auxbasis = auxbasis or (str(basis) + '-ri')
    hit = shipped_radii_lookup(element, str(basis), str(auxbasis), counts)
    if hit is None:
        raise KeyError(
            f'no tabulated grid for {element}/{basis}/{auxbasis} at counts '
            f'{sorted((counts or {}).items())}. Held for this element and '
            f'basis: {_tabulated_counts(element, basis, auxbasis)}.')
    radii, _, origin = hit
    return radii, origin


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


def write_json_atomic(path, payload):
    """Write `payload` to `path` so that no reader can ever see a partial file.

    `open(path, 'w')` truncates before it writes, so a concurrent reader sees an
    empty or half-written file; writing to a temporary in the SAME directory and
    renaming is atomic on POSIX.

    THE TEMPORARY'S NAME MUST BE UNIQUE ACROSS NODES, not merely across
    processes. A pid identifies a process on ONE machine and repeats on the next
    node of a multi-node job, so two ranks that share a filesystem open the same
    pid-named temporary, each truncating the other, and the rename then publishes
    a torn file -- a collision that cannot happen on one machine, where pids are
    unique, and so cannot be reproduced there. `mkstemp` creates the file with
    O_EXCL under a name no other writer can hold.
    """
    directory = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + '.',
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as fh:
            json.dump(payload, fh, indent=1)
        os.replace(tmp, path)
    except BaseException:
        # a private name is only unique while it exists: a failed write leaves
        # the temporary behind where the pid-named one was simply overwritten
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    hit = shipped_radii_lookup(element, basis, auxbasis, counts)
    if hit is not None and not return_candidates:
        radii, fit_error, row_origin = hit
        # The row's cusp flag is part of the grid, so it cannot be quietly
        # overridden by the argument: handing these radii back under
        # origin=False builds a grid one point smaller than the row describes.
        if row_origin != bool(origin):
            raise ValueError(
                f'the tabulated grid for {element}/{basis}/{auxbasis} at counts '
                f'{sorted((counts or {}).items())} has origin={row_origin} and '
                f'origin={bool(origin)} was asked for. The cusp point is part '
                f'of the grid, not of the recipe, so the two are different '
                f'grids at different point counts. Use `atomic_grid(...)`, '
                f'which returns the row\'s flag alongside its radii.')
        return radii, fit_error

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
        # Every rank of a distributed run that misses the table optimizes the
        # same radii into the same path, from as many nodes as the job holds.
        write_json_atomic(cache, {
            'element': element, 'basis': str(basis),
            'auxbasis': str(auxbasis), 'counts': counts,
            'settings': settings, 'fit_error': float(res.fun),
            'radii': {k: list(map(float, v)) for k, v in radii.items()}})
    except OSError:
        pass                    # a read-only checkout must not break the run
    if verbose:
        npts = sum(len(_SHELLS[n]) * len(r) for n, r in radii.items())
        print(f'  {element}: {npts} points, {len(starts)} start(s), '
              f'fit err {res.fun:.3e}')
    if return_candidates:
        return radii, float(res.fun), candidates
    return radii, float(res.fun)


def grid_points_per_atom(counts):
    """Interpolation points one atom contributes: |A1| A1 + |A2| A2 + ... .

    The sub-shell sizes come from `subshells()` rather than the literals
    6/8/12/24, so the count and the directions cannot drift apart.
    """
    shells = subshells()
    return sum(len(shells[name]) * int(counts.get(name, 0))
               for name in _SHELL_ORDER)


def _explicit_counts(spec):
    """Four shell counts from 'a,b,c,d', a sequence of four, or a dict.

    The dict form is keyed either on the shell names the fit uses or on the
    positions 0..3 of `_SHELL_ORDER`; both spell the same grid.
    """
    if isinstance(spec, dict):
        if set(spec) == set(_SHELL_ORDER):
            return {name: int(spec[name]) for name in _SHELL_ORDER}
        if set(spec) == set(range(len(_SHELL_ORDER))):
            return {name: int(spec[i]) for i, name in enumerate(_SHELL_ORDER)}
        raise ValueError(
            f'grid counts {spec!r}: a dict must be keyed on '
            f'{list(_SHELL_ORDER)} or on 0..{len(_SHELL_ORDER) - 1}.')
    if isinstance(spec, str):
        values = [v for v in spec.replace(' ', '').split(',') if v]
    else:
        try:
            values = list(spec)
        except TypeError:
            raise ValueError(
                f'grid {spec!r}: give an accuracy level, or shell counts as '
                f'"A1,A2,A3,B1", a sequence of {len(_SHELL_ORDER)} or a dict.')
    if len(values) != len(_SHELL_ORDER):
        raise ValueError(
            f'grid counts {spec!r}: {len(_SHELL_ORDER)} numbers are needed, '
            f'one per Lebedev sub-shell {list(_SHELL_ORDER)}, and '
            f'{len(values)} were given.')
    try:
        return {name: int(v) for name, v in zip(_SHELL_ORDER, values)}
    except (TypeError, ValueError):
        raise ValueError(f'grid counts {spec!r}: every entry must be an integer.')


def resolve_isdf_grid(grid, basis, elements=(), auxbasis=None,
                      n_start=ISDF_GRID_N_START):
    """The ONE named way to ask for an interpolation grid. Returns (counts, n_start).

    grid: an accuracy level of `ISDF_GRID_ACCURACY` -- 'G1' < 8 meV, 'G2'
        < 4 meV, 'G3' < 2 meV on the three lowest BSE roots against
        `solve_bse_df` -- or four explicit shell counts as 'A1,A2,A3,B1', a
        sequence, or a {shell: count} dict.
    elements: the atoms the grid will be placed on. Each is required to have a
        row in the shipped radii table, because a missing row does not make the
        run slower, it makes it a different grid: the radii are then
        re-optimized at run time onto another local minimum of a multi-modal
        surface, which no scored campaign describes.

    `n_start` is returned alongside the counts as the recipe to re-optimize
    with should a caller go on to build a row the table does not hold. It is
    not needed to READ one: the table is keyed on the physics alone, so a
    lookup asks for (element, basis, auxbasis, counts) and nothing else.

    COVERAGE IS STILL PER ELEMENT, so the check below is per element rather
    than per basis, and a molecule can be refused at a level its basis
    validates.

    There is no fallback anywhere in here. An accuracy level absent at a basis
    was never measured there, and the nearest level, a larger count or another
    basis are guesses dressed as answers -- the measured ladder is not even
    monotone, so a larger grid is not a safer one.
    """
    if grid is None:
        raise ValueError(
            'no grid asked for: pass an accuracy level of ISDF_GRID_ACCURACY '
            'or four shell counts.')
    if isinstance(grid, str) and ',' not in grid:
        level = grid.strip().upper()
        validated = ISDF_GRID_ACCURACY.get(str(basis).lower(), {})
        if level not in validated:
            raise ValueError(
                f'grid does not exist: no interpolation grid is validated at '
                f'accuracy {grid!r} for {basis}. '
                + (f'Validated at {basis}: {", ".join(sorted(validated))}.'
                   if validated else
                   f'Nothing is validated at {basis} at any accuracy; bases '
                   f'that have a level: {", ".join(sorted(ISDF_GRID_ACCURACY))}.')
                + f' A missing level was never measured, so there is nothing '
                  f'to fall back to -- score and optimize a grid table entry '
                  f'first, or pass four explicit shell counts.')
        counts = dict(zip(_SHELL_ORDER, validated[level]))
        asked = f'which is what accuracy {level} asks for at {basis}'
    else:
        counts = _explicit_counts(grid)
        asked = 'which is what was asked for explicitly'
    auxbasis = auxbasis or (str(basis) + '-ri')
    # The counts naming a grid and the table HOLDING one are two questions: a
    # level validated at a basis can still have no row for one of these
    # elements. The lookup is the physics key, so this asks exactly what a
    # fitting run will ask.
    shape = ','.join(str(counts[name]) for name in _SHELL_ORDER)
    missing = []
    for element in sorted(set(elements)):
        if shipped_radii_lookup(element, str(basis), str(auxbasis), counts) is None:
            held = _tabulated_counts(element, basis, auxbasis)
            missing.append(f'{element}: '
                           + (held if isinstance(held, str) else ', '.join(held)))
    if missing:
        raise KeyError(
            f'grid does not exist: the shipped radii table has no row at '
            f'({shape}) for '
            f'{", ".join(m.split(":")[0] for m in missing)} at '
            f'{basis}/{auxbasis}, {asked}'
            + '. Re-optimizing at run time is hours per element and lands on a '
              'different local minimum from every tabulated row, so the result '
              'would be neither cheap nor the grid that was scored. What the '
              'table does hold, with the recipe each row was found under -- '
            + '; '.join(missing))
    return counts, n_start


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
