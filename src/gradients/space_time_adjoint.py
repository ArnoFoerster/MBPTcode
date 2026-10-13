"""Reverse mode of the space-time/ISDF correlation chain.

The forward chain that `LinearResponse/space_time.py` and `GW/imaginary_time.py`
evaluate is, for every imaginary-time point,

    G^o_PQ(tau) = sum_i X_o[P,i] X_o[Q,i] e^{ e_i tau}
    G^v_PQ(tau) = sum_a X_v[P,a] X_v[Q,a] e^{-e_a tau}
    Pi_PQ(tau)  = G^o_PQ G^v_PQ                      (elementwise)
    proj(tau)   = -2 D^T Pi(tau) D
    chi0(i.w)   = sum_tau cosft_wt[w,tau] proj(tau)

and then a matrix function of chi0. Every step is a GEMM or a Hadamard product,
so every adjoint is a GEMM or a Hadamard product of the same shape: the reverse
pass costs what the forward pass costs, and the gradient is O(N^3) with the
forward's prefactor. Per tau point and row block,

    Pibar = -2 D projbar D^T                         (Hadamard operand)
    Gobar = Pibar * G^v,   Gvbar = Pibar * G^o       (elementwise)
    Xbar_o[R,i] = 2 e^{e_i tau} (Gobar X_o)[R,i]
    ebar_o[i]   = tau e^{e_i tau} sum_P X_o[P,i] (Gobar X_o)[P,i]

X is (M, norb) and D is (M, naux), and nothing may grow faster than they do:

  * The polarizability Pi = G^o * G^v is the one (M x M) object, and it is
    never held: `polarizability_tau` runs the energy routes' kernel,
    `LinearResponse.space_time.polarizability_projected_tau`, over every tau
    point; `polarizability_backward` tiles over grid rows as that kernel does;
    and the Green's functions are rebuilt in the reverse pass rather than
    taped. One forward kernel serves the energy and the gradient.
  * Everything downstream lives in the auxiliary basis. proj(tau), (ntau, naux,
    naux), is the one object of that size (every frequency is a fixed linear
    combination of it), and adjoints come back in the same space as
    projbar(tau). The dRPA and self-energy routes keep it whole; the
    quasiparticle solves hold it and projbar by auxiliary rows over the ranks
    (`ProjRows`), which `polarizability_backward` reads one gathered tau slice
    at a time. Frequencies are visited in blocks sized by `tile_gb`
    (`frequency_blocks`), so nothing scales as (nfreq, naux, naux); at
    naux ~ 5000 each frequency is 200 MB.
  * The self-energy is not built on the grid. Sigma_pp(tau) = sum_PQ X[P,p]
    (Zt * G)_PQ X[Q,p] with Zt = D Wt D^T is three (M x M) arrays per tau and
    O(M^2 naux) to form. The same number is

        Sigma^<_pp(tau) = sum_i e^{e_i tau} B_p[:,i]^T Wt(tau) B_p[:,i],
        B_p[P,q] = sum_k D[k,P] X[k,p] X[k,q],

    the pair density of one bra state in the auxiliary basis, (naux x norb),
    and one GEMM per tau. `three_index_slice` builds it in O(M naux norb) and
    `three_index_slice_backward` chains its adjoint back to X and D at the
    same cost.

Adjoints are returned with respect to (eps, X, D); dX/dR and dD/dR are a
separate layer.

X and D may be `SlicedFactors` (each rank's grid rows of X_mo and D); every
entry point that reads them gathers what it reads whole once (`whole_factor`)
and drops it on return, since the M^2 sweeps contract the grid index on both
sides of every GEMM. `polarizability_backward` and `chi0_frequency` take the
occupied and virtual branches of X_mo rather than X_mo and two copies. The
gathered arrays are the whole ones verbatim, so every output is the whole
factors' bit for bit. The adjoints come back whole (all-reduced sums over the
tau partition).

The chain is differentiable up to Sigma_c(i.omega) and no further: the Pade
continuation to the real axis has a derivative of order 1e11
(`pade_continuation_jvp`), so a quasiparticle gradient needs a
continuation-free equation; `contour_deformation` supplies it and
`qp_space_time` wires the two.

Re-exported here: the self-energy's transform pair (`sigma_fit_ranges`,
`sigma_transforms`, `sigma_transform_error`, from
`Base/utils/time_frequency.py`); the self-energy's forward halves
(`SingleReference.GW.imaginary_time`), so the reverse passes differentiate the
routine the energy routes evaluate; and the real-frequency screening below the
particle-hole gap (`real_frequency_weights` from
`SingleReference.GW.real_screening`, and `LaplaceRealScreening`, its adjoint
subclass).
"""
from dataclasses import dataclass

import numpy as np

from src.Base.constants import FIT_CHOLESKY_BLOCK, ISDF_TILE_GB
from src.Base.sliced_factors import (GridTileRows, SlicedFactors,
                                     whole_factor)
from src.Base.utils.analyticalContinuation import (greedy_pade_order, pade_eval,
                                                   thiele_coefficients)
from src.Base.utils.time_frequency import (sigma_fit_ranges,  # noqa: F401
                                           sigma_transform_error,
                                           sigma_transforms)
from src.Base.utils.mpi_grid import (agreement, allgather_ranges,
                                     current_comm, partition, reduce_sum)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.real_screening import \
    real_frequency_weights  # noqa: F401
from src.SingleReference.LinearResponse.space_time import (  # noqa: F401
    ProjRows, chi0_imaginary_frequency, frequency_blocks,
    laplace_representation_error, owned_frequency_blocks,
    polarizability_projected_sweep, rpa_correlation_energy_space_time,
    split_branches, three_index_ov, three_index_slice, tile_rows)
from src.SingleReference.GW.imaginary_time import (  # noqa: F401
    screened_interaction_tau, selfenergy_block, selfenergy_diag)
from src.SingleReference.GW.qp_solve import imaginary_axis_sample_points
from src.Solvers.qp_equation import solve_qp_equation
from src.gradients.contour_deformation_adjoint import \
    LaplaceRealScreeningAdjoint as LaplaceRealScreening  # noqa: F401


# ---------------------------------------------------------------------------
# the polarizability: the (M x M) sweep, forward and reverse
# ---------------------------------------------------------------------------

def polarizability_tau(X, D, eps, nocc, grid, mu=None, tile_gb=ISDF_TILE_GB,
                       tau_indices=None):
    """proj(tau) = -2 D^T Pi(tau) D on every tau point: (ntau, naux, naux).

    The energy routes' kernel, `polarizability_projected_tau`, run over
    `grid.tau_points` and kept whole rather than streamed into chi0. This is
    the one whole object of the N^3 route, and `polarizability_backward` below
    is its reverse, block for block; tests/test_space_time_shared_kernel.py
    pins the two callers together.

    tau_indices: compute these points only and leave the rest zero, for a
    sweep split over ranks; the sum over disjoint subsets is the whole.

    `SlicedFactors` are gathered whole, X_mo and D once each: the energy
    routes' sweep reads the collocation as an array.
    """
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    return polarizability_projected_sweep(X, D, eps, nocc, grid.tau_points,
                                          mu=mu, tau_indices=tau_indices,
                                          tile_memory_gb=tile_gb)


def chi0_frequency(X, D, eps, nocc, grid, mu=None, tile_gb=ISDF_TILE_GB,
                   tau_indices=None):
    """chi0(i.omega) in the auxiliary basis, (nfreq, naux, naux) -- whole.

    `LinearResponse.space_time.chi0_imaginary_frequency` under this package's
    argument names. It holds the entire frequency axis, so it is a reference
    and a small-grid convenience; the routes below never call it.

    tau_indices: sweep only these tau points, for a split over ranks. chi0 is
    accumulated as the points arrive rather than kept per point, so the sum
    over disjoint subsets is the whole up to summation order, not bitwise.

    `SlicedFactors` as X reach the sweep as they are, which gathers its two
    branches; as D they are gathered whole once.
    """
    return chi0_imaginary_frequency(X, whole_factor(D, 'D'), eps, nocc, grid,
                                    mu=mu, tau_indices=tau_indices,
                                    tile_memory_gb=tile_gb)


def polarizability_backward(proj_bar, X, D, eps, nocc, grid, mu=None,
                            tile_gb=ISDF_TILE_GB, tau_indices=None):
    """(eps_bar, X_bar, D_bar) for an adjoint proj_bar, (ntau, naux, naux), on proj(tau).

    One reverse sweep of the same loop, at the same cost: per tau and row block
    two Green's-function GEMMs, two Hadamard products and three more GEMMs.
    proj_bar is symmetrized per tau here, so a caller may hand over the raw
    adjoint of whatever it built from proj.

    tau_indices: only these tau points contribute. All three outputs are sums
    over tau, so the sum over disjoint subsets is the whole -- the tau
    partition of the forward sweep, reduced once per gradient, M (norb + naux)
    doubles.

    proj_bar may be `ProjRows`: each of this rank's tau points is then
    gathered whole from the ranks' rows as the sweep reaches it
    (`ProjRows.tau_slices`), the same bits as the whole array's slice, and no
    rank holds more than its rows and one slice.

    `SlicedFactors` as X are read as their two branches, gathered once each by
    `split_branches`, and as D gathered whole once; the adjoints are whole.
    """
    if isinstance(proj_bar, ProjRows):
        slabs = proj_bar.tau_slices(tau_indices)
    else:
        which = (range(grid.ntau) if tau_indices is None
                 else np.atleast_1d(tau_indices))
        slabs = ((k, proj_bar[k]) for k in which)
    return _sweep_backward(slabs, X, D, eps, nocc, grid, mu, tile_gb)


def _sweep_backward(slabs, X, D, eps, nocc, grid, mu, tile_gb):
    """`polarizability_backward` over the (k, projbar slice k) pairs `slabs`
    yields, one tau slice at a time and in its order, so a caller whose
    projbar has a closed form or is held by rows never hands over the
    (ntau, naux, naux) array."""
    X_o, X_v, e_o, e_v, _, occ, virt = split_branches(X, eps, nocc, mu)
    D = whole_factor(D, 'D')
    M, naux = X_o.shape[0], D.shape[1]
    rows = tile_rows(M, tile_gb, 4 * M * 8)

    eps_bar = np.zeros_like(np.asarray(eps, float))
    X_bar = (np.zeros((M, X.nmo)) if isinstance(X, SlicedFactors)
             else np.zeros_like(X))
    D_bar = np.zeros_like(D)

    # Four (rows, M) buffers reused across every tau point and block: the
    # backward pass touches six such arrays per block, and allocating them is
    # comparable to the products themselves at these shapes.
    Go = np.empty((rows, M))
    Gv = np.empty((rows, M))
    scratch = np.empty((rows, M))
    work = np.empty((rows, M))
    # E = -2 D pb, one (M, naux) buffer for every tau point: the product
    # written into it and scaled in place, the same bits as -2.0 * (D @ pb)
    # (the same GEMM, then an exact scaling by a power of two) without its
    # two whole temporaries
    E = np.empty((M, naux))
    for k, slab in slabs:
        tau = grid.tau_points[k]
        # proj = -2 D^T Pi D: D appears on both sides, Pi and proj_bar are
        # symmetric, and E carries the -2 for everything downstream
        pb = 0.5 * (slab + slab.T)
        np.matmul(D, pb, out=E)
        E *= -2.0
        w, u = np.exp(e_o * tau), np.exp(-e_v * tau)
        for p0 in range(0, M, rows):
            p1 = min(p0 + rows, M)
            b = p1 - p0
            np.matmul(X_o[p0:p1] * w, X_o.T, out=Go[:b])
            np.matmul(X_v[p0:p1] * u, X_v.T, out=Gv[:b])
            np.multiply(Go[:b], Gv[:b], out=scratch[:b])          # Pi
            D_bar[p0:p1] += 2.0 * (scratch[:b] @ E)
            np.matmul(E[p0:p1], D.T, out=scratch[:b])             # Pi_bar
            np.multiply(scratch[:b], Gv[:b], out=work[:b])        # Go_bar
            GoX = work[:b] @ X_o
            np.multiply(scratch[:b], Go[:b], out=work[:b])        # Gv_bar
            GvX = work[:b] @ X_v
            X_bar[p0:p1, occ] += 2.0 * GoX * w
            X_bar[p0:p1, virt] += 2.0 * GvX * u
            eps_bar[occ] += tau * w * np.einsum('bi,bi->i', X_o[p0:p1], GoX)
            eps_bar[virt] -= tau * u * np.einsum('ba,ba->a', X_v[p0:p1], GvX)
    return eps_bar, X_bar, D_bar


def chi0_backward(chi0_bar, X, D, eps, nocc, grid, mu=None,
                  tile_gb=ISDF_TILE_GB):
    """(eps_bar, X_bar, D_bar) for an adjoint chi0_bar on chi0(i.omega).

    proj(tau) enters chi0 through the fixed transform weights only, so the
    adjoint transposes them, projbar = cosft_wt^T chi0bar. Each tau slice is
    formed when the sweep reads it and dropped after, so neither the
    (ntau, naux, naux) projbar nor a (nfreq, naux, naux) copy is held.
    For a single frequency's chi0_bar, (1, naux, naux) -- what the static
    screening hands over -- a slice is one product per element and the same
    bits as the whole transform's; for several it is a sum over frequencies
    in another association than the whole transform's GEMM.
    """
    return _sweep_backward(
        ((k, np.tensordot(grid.cosft_wt[:, k], chi0_bar, axes=(0, 0)))
         for k in range(grid.ntau)), X, D, eps, nocc, grid, mu, tile_gb)


def polarizability_backward_rows(proj_bar, X, D, eps, nocc, grid, mu=None,
                                 block=None, comm=None):
    """(eps_bar, X_bar, D_bar) of `polarizability_backward` with X_bar and
    D_bar held in the grid's fixed tiles over the ranks (`GridTileRows`).

    Tile-major, the mirror of the forward's (tau, grid-row tile) split: every
    rank owns the tiles t % size of `block` points and sweeps every tau point
    for them (`_sweep_backward_rows`), so the rows are an output partition
    and no rank holds an adjoint pair whole. proj_bar is `ProjRows` (each
    slice then gathered to every rank as the sweep reaches it,
    `ProjRows.tau_slices(None)`) or a whole (ntau, naux, naux) array.

    The same bits at every rank count, ranks owning no tile included; another
    association than `polarizability_backward`'s (E and D_bar per tile).
    X: the MO collocation or `SlicedFactors` (its two branches gathered);
    D whole or `SlicedFactors` (gathered whole). block: None is
    `FIT_CHOLESKY_BLOCK`. comm: None is `current_comm()`.
    """
    comm = current_comm() if comm is None else comm
    block = FIT_CHOLESKY_BLOCK if block is None else int(block)
    if isinstance(proj_bar, ProjRows):
        slabs = proj_bar.tau_slices(None)
    else:
        slabs = ((k, proj_bar[k]) for k in range(grid.ntau))
    return _sweep_backward_rows(slabs, X, D, eps, nocc, grid, mu, block, comm)


def chi0_backward_rows(chi0_bar, X, D, eps, nocc, grid, mu=None, block=None,
                       comm=None):
    """(eps_bar, X_bar, D_bar) of `chi0_backward` with X_bar and D_bar in the
    grid's fixed tiles over the ranks, tile-major (`polarizability_backward_
    rows`): chi0_bar is whole and the same on every rank, so each rank forms
    projbar_k = cosft_wt[:, k]^T chi0_bar itself and no slice travels."""
    comm = current_comm() if comm is None else comm
    block = FIT_CHOLESKY_BLOCK if block is None else int(block)
    return _sweep_backward_rows(
        ((k, np.tensordot(grid.cosft_wt[:, k], chi0_bar, axes=(0, 0)))
         for k in range(grid.ntau)), X, D, eps, nocc, grid, mu, block, comm)


def _sweep_backward_rows(slabs, X, D, eps, nocc, grid, mu, block, comm=None):
    """`_sweep_backward` over this rank's fixed grid tiles, for every
    (k, projbar slice k) pair `slabs` yields.

    Per tau point, for each own tile t and each column tile q in order,

        Go = (X_o[t] w) X_o[q]^T,  Gv = (X_v[t] u) X_v[q]^T,  Pi = Go o Gv
        PiD_t += Pi D[q],          Pibar = E_t D[q]^T,  E_t = -2 D[t] pb
        GoX_t += (Pibar o Gv) X_o[q],  GvX_t += (Pibar o Go) X_v[q]

    and then D_bar[t] += -4 PiD_t pb, X_bar[t] += 2 (GoX_t w | GvX_t u):
    Pi and pb are symmetric, so every write lands on the tile's own rows, and
    E and Pi exist for one (tile, tile) block at a time. D_bar reads Pi
    through PiD_t rather than through a whole E, which is the one extra
    product, 2 M naux^2 per tau. eps_bar gathers the tiles' partials and
    adds them in tile order. Every call's shape is fixed by the tiles.

    A whole X is read through column views of its two branches where they
    are contiguous ranges, so no copy of either stands beside it; slices are
    gathered as the two branches (`split_branches`)."""
    viewed = isinstance(X, np.ndarray)
    # on a whole X the split reads one row, for the energies and index sets
    X_o, X_v, e_o, e_v, _, occ, virt = split_branches(X[:1] if viewed else X,
                                                      eps, nocc, mu)
    if viewed:
        X_o, X_v = _branch_views(X, occ, virt)
    D = whole_factor(D, 'D')
    M, naux = D.shape
    nmo = len(np.asarray(eps))
    X_bar = GridTileRows.zeros(M, nmo, block, comm)
    D_bar = GridTileRows.zeros(M, naux, block, comm)
    tiles = X_bar.bounds
    eps_tile = np.zeros((len(tiles), nmo))
    for k, slab in slabs:
        tau = grid.tau_points[k]
        pb = 0.5 * (slab + slab.T)
        w, u = np.exp(e_o * tau), np.exp(-e_v * tau)
        for t in X_bar.mine:
            t0, t1 = tiles[t]
            Xo_t, Xv_t = X_o[t0:t1] * w, X_v[t0:t1] * u
            E = D[t0:t1] @ pb
            E *= -2.0
            PiD = np.zeros((t1 - t0, naux))
            GoX = np.zeros((t1 - t0, len(occ)))
            GvX = np.zeros((t1 - t0, len(virt)))
            for q0, q1 in tiles:
                Go = Xo_t @ X_o[q0:q1].T
                Gv = Xv_t @ X_v[q0:q1].T
                PiD += (Go * Gv) @ D[q0:q1]
                Pb = E @ D[q0:q1].T
                GoX += (Pb * Gv) @ X_o[q0:q1]
                GvX += (Pb * Go) @ X_v[q0:q1]
            db = PiD @ pb
            db *= -4.0
            D_bar.tile(t)[...] += db
            xt = X_bar.tile(t)
            xt[:, occ] += 2.0 * GoX * w
            xt[:, virt] += 2.0 * GvX * u
            eps_tile[t, occ] += tau * w * np.einsum('bi,bi->i', X_o[t0:t1], GoX)
            eps_tile[t, virt] -= tau * u * np.einsum('ba,ba->a', X_v[t0:t1],
                                                     GvX)
    if X_bar.comm is not None:
        allgather_ranges(eps_tile, [[(t, t + 1) for t in X_bar.owned(r)]
                                    for r in range(X_bar.size)], X_bar.comm)
    eps_bar = np.zeros(nmo)
    for t in range(len(tiles)):
        eps_bar += eps_tile[t]
    return eps_bar, X_bar, D_bar


def _branch_views(X, occ, virt):
    """(X_o, X_v) as column views of X where each index set is a contiguous
    range, copies where it is not."""
    def view(idx):
        idx = np.asarray(idx)
        if len(idx) and np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
            return X[:, idx[0]:idx[0] + len(idx)]
        return np.ascontiguousarray(X[:, idx])
    return view(occ), view(virt)


def three_index_slice_backward_rows(X, D, p, Bp_bar, out):
    """`three_index_slice_backward` into `GridTileRows` (X_acc, D_acc), on
    this rank's tiles: every term of a row reads that row and the replicated
    Bp_bar alone, so a tile is the same bits whoever owns it. X and D whole
    (or `SlicedFactors`, gathered whole once each)."""
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    X_bar, D_bar = out
    for t in X_bar.mine:
        t0, t1 = X_bar.bounds[t]
        xp = X[t0:t1, p, None]
        G = D[t0:t1] @ Bp_bar
        term = xp * G
        term[:, p] += np.einsum('kq,kq->k', G, X[t0:t1])
        X_bar.tile(t)[...] += term
        D_bar.tile(t)[...] += xp * (X[t0:t1] @ Bp_bar.T)
    return X_bar, D_bar


def fold_owned_frequencies(proj_bar, c, blk):
    """proj_bar += c^T blk for a block of this rank's frequencies, (ntau, naux,
    naux) from c (nb, ntau) and blk (nb, naux, naux), one auxiliary row at a
    time, so the temporary is (ntau, naux) per row rather than
    (ntau, naux, naux).

    A partial over the frequency partition, reduced after, so its call shape
    follows the partition and its rows are a summation order, not an output
    partition's (`ProjRows.fold` is one). A block of one frequency is one
    product per element either way; for several, the rows are the whole
    product's on a BLAS row-stable in its shape (MKL at even naux) and a
    re-association of the block's sum elsewhere."""
    for i in range(blk.shape[1]):
        proj_bar[:, i, :] += np.tensordot(c, blk[:, i, :], axes=(0, 0))


@dataclass(frozen=True)
class FoldAdjoint:
    """dE/dN and dE/dC of one folded correlation energy, (naux, naux) each.

    N is the reaction field in the dressed auxiliary gauge and C the
    counter-term matrix, both inputs to `rpa_energy_and_adjoint`. They are held
    separate here and combined by whoever chose them: the fold sets
    S_w = I - (1 - g_w) N and C = I - N out of one N, so its own N adjoint is
    `n_bar - counter_bar`, while another caller pairing a different C with the
    same N would combine them differently. `counter_bar` is None when no
    counter-term was given.
    """

    n_bar: np.ndarray
    counter_bar: np.ndarray


def rpa_energy_and_adjoint(X, D, eps, nocc, grid, mu=None, want_grad=True,
                           tile_gb=ISDF_TILE_GB, screening=None,
                           counter_term=None, comm=None):
    """E_c^dRPA and (eps_bar, X_bar, D_bar, fold_bar), one forward and one reverse sweep.

    The forward is `LinearResponse.space_time.rpa_correlation_energy_space_
    time`, which returns E_c and its tape (proj(tau) and the tau and frequency
    partitions it was built on), so the gradient differentiates the functional
    the energy routes evaluate; `screening`, `counter_term` and `comm` are
    that routine's arguments. want_grad=False is the forward alone.

    E_c = (1/2pi) sum_w W_w [ log det(1 - chi0_w) + tr chi0_w ], so
    chi0_bar_w = (W_w/2pi) [ I - (1 - chi0_w)^-1 ], naux^3 per frequency. The
    frequencies come out of proj(tau) one block at a time (a second
    ntau x naux^2 transform per block, against its naux^3 factorization);
    each block is overwritten by its adjoint and folded into projbar one
    auxiliary row at a time (`fold_owned_frequencies`), so neither an
    (ntau, naux, naux) temporary nor the whole frequency axis is held. Over
    ranks each frequency's chi0 is the serial one bitwise
    (`owned_frequency_blocks`) and only the sum into projbar re-associates.

    With a fold the frequency step is
    blk[m] = (W_w/2pi) [ C^T - S_w^T (I - S_w c_w)^-T ], the gas-phase
    expression at S = C = I. N and C are inputs, not functions of the
    factors, so their adjoints leave as `fold_bar` for the caller to route
    through vtilde and the dressed metric root (`fold_bar_to_gauge` in
    `src/gradients/solvated_rpa_energy.py`); None when neither term was given.

    comm: None is `current_comm()`, handed to the forward explicitly. An
    audited run compares the digests of the four outputs (the forward
    compares the inputs'). `SlicedFactors` are gathered whole, X_mo and D once
    each, for both sweeps.
    """
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    n_mat, g = (None, None) if screening is None else screening
    folding = n_mat is not None or counter_term is not None
    comm = current_comm() if comm is None else comm
    if not want_grad:
        return rpa_correlation_energy_space_time(
            X, D, eps, nocc, grid, mu=mu, tile_gb=tile_gb, screening=screening,
            counter_term=counter_term, comm=comm)
    e_c, tape = rpa_correlation_energy_space_time(
        X, D, eps, nocc, grid, mu=mu, tile_gb=tile_gb, screening=screening,
        counter_term=counter_term, comm=comm, want_tape=True)
    proj_tau = tape.proj_tau
    naux = proj_tau.shape[-1]
    eye = np.eye(naux)
    proj_bar = np.zeros_like(proj_tau)
    n_bar = np.zeros((naux, naux)) if folding else None
    counter_bar = (np.zeros((naux, naux)) if counter_term is not None else None)
    for ks, blk in owned_frequency_blocks(proj_tau, grid.cosft_wt, tile_gb,
                                          tape.freq_indices):
        for m, k in enumerate(ks):
            c0 = blk[m]
            # S_w c_w, the argument of the logarithm the forward took, rebuilt
            # here rather than taped: it is one GEMM against the factorization
            # below it.
            one_minus_g = 0.0 if n_mat is None else 1.0 - g[k]
            c = c0 if n_mat is None else c0 - one_minus_g * (n_mat @ c0)
            scale = grid.omega_weights[k] / (2.0 * np.pi)
            # A = (I - S c)^-1. The default S = C = I collapses the two
            # expressions below to (I - A^T), which is the gas-phase branch.
            a_t = np.linalg.inv(eye - c).T
            if not folding:
                blk[m] = scale * (eye - a_t)
                continue
            s_t = (eye if n_mat is None else eye - one_minus_g * n_mat.T)
            c_t = eye if counter_term is None else counter_term.T
            # c0 is blk[m], the block the adjoint overwrites, so the two terms
            # that read chi0 are taken first
            n_bar += (scale * one_minus_g) * (a_t @ c0.T)
            if counter_bar is not None:
                counter_bar += scale * c0.T
            blk[m] = scale * (c_t - s_t @ a_t)
        fold_owned_frequencies(proj_bar, grid.cosft_wt[ks], blk)
    if comm is not None and comm.Get_size() > 1:
        reduce_sum(proj_bar, comm)
        if n_bar is not None:
            reduce_sum(n_bar, comm)
        if counter_bar is not None:
            reduce_sum(counter_bar, comm)
    fold_bar = None if not folding else FoldAdjoint(n_bar, counter_bar)
    eps_bar, X_bar, D_bar = polarizability_backward(
        proj_bar, X, D, eps, nocc, grid, mu=mu, tile_gb=tile_gb,
        tau_indices=tape.tau_indices)
    if comm is not None and comm.Get_size() > 1:
        for arr in (eps_bar, X_bar, D_bar):
            reduce_sum(arr, comm)
        agreement((e_c, eps_bar, X_bar, D_bar, n_bar, counter_bar), comm,
                  audit_only=True, label='rpa_energy_and_adjoint outputs')
    return e_c, eps_bar, X_bar, D_bar, fold_bar


def rpa_frequency_traces(X, D, eps, nocc, grid, mat, mu=None,
                         tile_gb=ISDF_TILE_GB):
    """Tr[mat c_w] at every imaginary frequency, (nfreq,), from the same sweep.

    With mat = N = V_d^(-1/2) vtilde V_d^(-1/2) and c_w the dressed-gauge
    projected chi0 this is Tr[vtilde P_1(iw)]: the integrand of the
    first-order-in-vtilde term of the correlation energy, i.e. the
    solute-solvent dispersion at the RPA level before any frequency damping.
    It carries no dependence on how the interaction is scaled, so one sweep
    serves a whole scan over the solvent's single-pole energy. `SlicedFactors`
    are gathered by that sweep.
    """
    proj_tau = polarizability_tau(X, D, eps, nocc, grid, mu=mu, tile_gb=tile_gb)
    naux = proj_tau.shape[-1]
    out = np.empty(grid.nfreq)
    for k0, k1 in frequency_blocks(grid.nfreq, naux, tile_gb, live=3):
        blk = np.tensordot(grid.cosft_wt[k0:k1], proj_tau, axes=(1, 0))
        out[k0:k1] = np.einsum('pq,kqp->k', mat, blk, optimize=True)
    return out


# ---------------------------------------------------------------------------
# the three-index tensor: never whole, one slice or one block at a time.
# The forward halves are `LinearResponse.space_time.three_index_slice` and
# `three_index_ov`, imported above and re-exported under the same names.
# ---------------------------------------------------------------------------

def three_index_slice_backward(X, D, p, Bp_bar, tile_gb=ISDF_TILE_GB,
                               out=None):
    """(X_bar, D_bar) of sum_Pq Bp_bar[P,q] B_p[P,q]; same tiling, same cost.

    X[k,p] appears in every entry of the slice and X[k,q] in one column, so the
    bra state collects the row sum and every state its own column.
    `SlicedFactors` are gathered whole, once each.

    out: (X_acc, D_acc) to add the adjoints into and return, instead of a new
         pair. Each tile's X term, the bra's column sum included, is formed
         before it is added, so every element of the accumulator takes the
         one addition `X_acc += X_bar` gave it: the same bits (up to the sign
         of an exact zero), one whole pair fewer.
    """
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    M, norb = X.shape
    if out is None:
        X_bar, D_bar = np.zeros_like(X), np.zeros_like(D)
    else:
        X_bar, D_bar = out
    rows = tile_rows(M, tile_gb, 2 * max(norb, D.shape[1]) * 8)
    for p0 in range(0, M, rows):
        p1 = min(p0 + rows, M)
        xp = X[p0:p1, p, None]
        G = D[p0:p1] @ Bp_bar                              # (rows, norb)
        term = xp * G
        term[:, p] += np.einsum('kq,kq->k', G, X[p0:p1])
        X_bar[p0:p1] += term
        D_bar[p0:p1] += xp * (X[p0:p1] @ Bp_bar.T)         # (rows, naux)
    return X_bar, D_bar


def three_index_ov_backward(X, D, eps, nocc, Cov_bar, tile_gb=ISDF_TILE_GB):
    """(X_bar, D_bar) of sum Cov_bar[P,(i,a)] C_ov[P,(i,a)]; same tiling.
    `SlicedFactors` are gathered whole, once each."""
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    occ, virt = get_occ_virt_indices(eps, nocc)
    M = X.shape[0]
    n_o, n_v = len(occ), len(virt)
    X_bar = np.zeros_like(X)
    D_bar = np.zeros_like(D)
    rows = tile_rows(M, tile_gb, 2 * n_o * n_v * 8)
    for p0 in range(0, M, rows):
        p1 = min(p0 + rows, M)
        Xo, Xv = X[p0:p1][:, occ], X[p0:p1][:, virt]
        G = (D[p0:p1] @ Cov_bar).reshape(p1 - p0, n_o, n_v)
        X_bar[p0:p1, occ] += np.einsum('kia,ka->ki', G, Xv)
        X_bar[p0:p1, virt] += np.einsum('kia,ki->ka', G, Xo)
        pair = (Xo[:, :, None] * Xv[:, None, :]).reshape(p1 - p0, n_o * n_v)
        D_bar[p0:p1] += pair @ Cov_bar.T
    return X_bar, D_bar


# ---------------------------------------------------------------------------
# the self-energy on the imaginary axis, in the auxiliary basis. Its forward
# halves (`screened_interaction_tau`, `selfenergy_block`, `selfenergy_diag`)
# are `GW.imaginary_time`'s, imported above and re-exported; only the reverse
# passes live here.
# ---------------------------------------------------------------------------

def screened_interaction_tau_backward(Wt_bar, proj_tau, grid, Ctw,
                                      tile_gb=ISDF_TILE_GB, comm=None):
    """projbar for an adjoint Wt_bar on Wt(tau): the reverse of `screened_interaction_tau`.

    Wt(tau) = sum_w Ctw[t,w] (W_w - I) with W = [1 - chi0]^-1, so the adjoint
    of the Dyson step is chi0_bar = W Wt_bar W at each frequency and the two
    transforms turn round. W is rebuilt from proj(tau) one frequency block at
    a time rather than taped, so nothing here is (nfreq, naux, naux) either.

    comm: the frequency split of the forward, run backwards -- the naux^3
    factorization is again the per-point work and projbar is again a sum over
    frequencies, reduced once, ntau x naux^2. Each W is the serial one
    bitwise (`owned_frequency_blocks`); only that sum re-associates. None is
    `current_comm()`; an audited run compares the digests of the inputs and
    of projbar. Each block is folded into projbar one auxiliary row at a time
    (`fold_owned_frequencies`), with no (ntau, naux, naux) temporary.
    """
    naux = proj_tau.shape[-1]
    eye = np.eye(naux)
    comm = current_comm() if comm is None else comm
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if nranks > 1:
        agreement((Wt_bar, proj_tau, Ctw), comm, audit_only=True,
                  label='screened_interaction_tau_backward inputs')
    nu_mine = partition(grid.nfreq, rank, nranks) if nranks > 1 else None
    proj_bar = np.zeros_like(proj_tau)
    for ks, chi0_blk in owned_frequency_blocks(proj_tau, grid.cosft_wt, tile_gb,
                                               nu_mine, live=4):
        Wbar_blk = np.tensordot(Ctw[:, ks], Wt_bar, axes=(0, 0))
        for m in range(len(ks)):
            Wk = np.linalg.inv(eye - chi0_blk[m])
            chi0_blk[m] = Wk @ Wbar_blk[m] @ Wk
        fold_owned_frequencies(proj_bar, grid.cosft_wt[ks], chi0_blk)
    if nranks > 1:
        reduce_sum(proj_bar, comm)
        agreement(proj_bar, comm, audit_only=True,
                  label='screened_interaction_tau_backward outputs')
    return proj_bar


def selfenergy_block_backward(a_re, a_im, X, D, eps, nocc, grid, states,
                              transforms, mu, cache, intermediate=None,
                              tile_gb=ISDF_TILE_GB, comm=None):
    """(eps_bar, X_bar, D_bar) for a real functional of Sigma^c_pq(i.omega_out).

    a_re, a_im : (nfreq_out, nstates, nstates) adjoints on the real and
        imaginary parts, in `selfenergy_block`'s own index order.

    `selfenergy_diag_backward` with the bra free as well. Sigma_pq is symmetric
    in (p, q) because Wt(tau) is, so only the symmetric part of the adjoint can
    act and it is symmetrized on the way in; the bra and ket slices then take
    separate contributions where the diagonal case took one of twice the size.
    `intermediate` masks the summed index as in the forward, which also
    zeroes eps_bar on the orbitals it excludes -- they carry no dependence.

    comm: the forward's splits run backwards, axis for axis -- the self-energy
    sweep over tau (reducing eps_bar, the slice adjoints and Wt_bar, the last
    ntau x naux^2 over disjoint slots), the Dyson step over frequency inside
    `screened_interaction_tau_backward`, and the M^2 reverse sweep over tau
    again. The per-state slice adjoints at the end are replicated, so they are
    added after the reduction and not through it. None is `current_comm()`;
    an audited run compares the digests of the inputs, the forward's tape
    included, and of the three adjoints. `SlicedFactors` are gathered whole,
    X_mo and D once each, for the whole reverse pass.
    """
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    Ctw, C, S = transforms
    _, _, e_o, e_v, _, occ, virt = split_branches(X, eps, nocc, mu)
    comm = current_comm() if comm is None else comm
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if nranks > 1:
        agreement((a_re, a_im, X, D, eps, transforms, cache, states,
                   intermediate), comm, audit_only=True,
                  label='selfenergy_block_backward inputs')
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    proj_tau, Wt_tau, Bs = cache
    states = np.atleast_1d(states)
    n_s, _, norb = Bs.shape

    keep = np.ones(len(eps), bool)
    if intermediate is not None:
        keep[:] = False
        keep[np.atleast_1d(intermediate)] = True
    w_occ = np.where(keep[occ], 1.0, 0.0)
    w_virt = np.where(keep[virt], 1.0, 0.0)

    a_re = 0.5 * (np.asarray(a_re) + np.swapaxes(np.asarray(a_re), 1, 2))
    a_im = 0.5 * (np.asarray(a_im) + np.swapaxes(np.asarray(a_im), 1, 2))
    ac = np.tensordot(C, a_re, axes=(0, 0))           # (ntau, nstates, nstates)
    bs = np.tensordot(S, a_im, axes=(0, 0))
    sg_bar = -0.5 * (ac + bs)
    sl_bar = -0.5 * (ac - bs)

    eps_bar = np.zeros_like(np.asarray(eps, float))
    Bs_bar = np.zeros_like(Bs)
    Wt_bar = np.zeros_like(Wt_tau)
    c = np.empty(norb)
    for k in (range(grid.ntau) if tau_mine is None else tau_mine):
        tau = grid.tau_points[k]
        w, u = np.exp(e_o * tau) * w_occ, np.exp(-e_v * tau) * w_virt
        for s in range(n_s):
            Y = Wt_tau[k] @ Bs[s]                              # (naux, norb)
            # Both the ket adjoint and Wt's are linear in B_t c_ts, so the sum
            # over the bra runs first and each takes one GEMM per ket instead
            # of one per (bra, ket) -- the same count the forward pays.
            acc = np.zeros_like(Y)
            for t in range(n_s):
                b = Bs[t]
                c[occ] = sl_bar[k, t, s] * w
                c[virt] = -sg_bar[k, t, s] * u
                acc += b * c
                Bs_bar[t] += Y * c                             # the bra slice
                bYb = np.einsum('Pq,Pq->q', b, Y)
                # w_i = e^{e_i tau} and u_a = e^{-e_a tau}: both slopes carry a
                # tau, and the sign of u's cancels the sign in front of sig_g
                eps_bar[occ] += sl_bar[k, t, s] * tau * w * bYb[occ]
                eps_bar[virt] += sg_bar[k, t, s] * tau * u * bYb[virt]
            Bs_bar[s] += Wt_tau[k].T @ acc                     # the ket slice
            Wt_bar[k] += acc @ Bs[s].T
    if nranks > 1:
        for arr in (eps_bar, Bs_bar, Wt_bar):
            reduce_sum(arr, comm)

    proj_bar = screened_interaction_tau_backward(Wt_bar, proj_tau, grid, Ctw,
                                                 tile_gb=tile_gb, comm=comm)
    eps2, X_bar, D_bar = polarizability_backward(proj_bar, X, D, eps, nocc, grid,
                                                 mu=mu, tile_gb=tile_gb,
                                                 tau_indices=tau_mine)
    if nranks > 1:
        for arr in (eps2, X_bar, D_bar):
            reduce_sum(arr, comm)
    for s, p in enumerate(states):
        three_index_slice_backward(X, D, int(p), Bs_bar[s], tile_gb=tile_gb,
                                   out=(X_bar, D_bar))
    eps_bar = eps_bar + eps2
    if nranks > 1:
        agreement((eps_bar, X_bar, D_bar), comm, audit_only=True,
                  label='selfenergy_block_backward outputs')
    return eps_bar, X_bar, D_bar


def selfenergy_diag_backward(a_re, a_im, X, D, eps, nocc, grid, states,
                             transforms, mu, cache, tile_gb=ISDF_TILE_GB,
                             comm=None):
    """(eps_bar, X_bar, D_bar) for a real functional of Sigma^c_pp(i.omega_out).

    a_re, a_im : (nstates, nfreq_out) adjoints on the real and imaginary parts.

    The reverse of every step is its own shape: the tau->omega weights
    transpose; sum_q c_q b_q^T Wt b_q has adjoints 2 c_q Wt b_q on b_q and
    sum_q c_q b_q b_q^T on Wt, one GEMM each; W = [1 - chi0]^-1 contributes
    chi0_bar = W W_bar W, rebuilt per frequency block from proj(tau) rather than
    kept; and the sweep at the end is `polarizability_backward`. The whole
    self-energy gradient costs one more sweep of what the self-energy cost.

    comm: `selfenergy_diag`'s splits run backwards -- tau for the self-energy
    sweep (reducing eps_bar, the slice adjoints and Wt_bar) and for the M^2
    reverse sweep, frequency for the Dyson step. None is `current_comm()`; an
    audited run compares digests as `selfenergy_block_backward` does.
    `SlicedFactors` are gathered whole, X_mo and D once each, for the whole
    reverse pass.
    """
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    Ctw, C, S = transforms
    _, _, e_o, e_v, _, occ, virt = split_branches(X, eps, nocc, mu)
    comm = current_comm() if comm is None else comm
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if nranks > 1:
        agreement((a_re, a_im, X, D, eps, transforms, cache, states), comm,
                  audit_only=True, label='selfenergy_diag_backward inputs')
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    proj_tau, Wt_tau, Bs = cache
    states = np.atleast_1d(states)
    norb = Bs.shape[2]

    # Sigma = -0.5 [ (sig_g + sig_l)^T C^T + i (sig_g - sig_l)^T S^T ]
    ac, bs = a_re @ C, a_im @ S                       # (nstates, ntau)
    sg_bar = -0.5 * (ac + bs).T                       # (ntau, nstates)
    sl_bar = -0.5 * (ac - bs).T

    eps_bar = np.zeros_like(np.asarray(eps, float))
    Bs_bar = np.zeros_like(Bs)
    Wt_bar = np.zeros_like(Wt_tau)
    c = np.empty(norb)
    for k in (range(grid.ntau) if tau_mine is None else tau_mine):
        tau = grid.tau_points[k]
        w, u = np.exp(e_o * tau), np.exp(-e_v * tau)
        for s in range(len(states)):
            b = Bs[s]
            Y = Wt_tau[k] @ b                                   # (naux, norb)
            # sig_l = sum_i w_i b_i^T Y_i,  sig_g = -sum_a u_a b_a^T Y_a
            c[occ] = sl_bar[k, s] * w
            c[virt] = -sg_bar[k, s] * u
            Bs_bar[s] += 2.0 * Y * c
            Wt_bar[k] += (b * c) @ b.T
            bYb = np.einsum('Pq,Pq->q', b, Y)
            # w_i = e^{e_i tau} and u_a = e^{-e_a tau}: both slopes carry a tau,
            # and the sign of u's cancels the sign in front of sig_g
            eps_bar[occ] += sl_bar[k, s] * tau * w * bYb[occ]
            eps_bar[virt] += sg_bar[k, s] * tau * u * bYb[virt]
    if nranks > 1:
        for arr in (eps_bar, Bs_bar, Wt_bar):
            reduce_sum(arr, comm)

    proj_bar = screened_interaction_tau_backward(Wt_bar, proj_tau, grid, Ctw,
                                                 tile_gb=tile_gb, comm=comm)
    eps2, X_bar, D_bar = polarizability_backward(proj_bar, X, D, eps, nocc, grid,
                                                 mu=mu, tile_gb=tile_gb,
                                                 tau_indices=tau_mine)
    if nranks > 1:
        for arr in (eps2, X_bar, D_bar):
            reduce_sum(arr, comm)
    for s, p in enumerate(states):
        three_index_slice_backward(X, D, int(p), Bs_bar[s], tile_gb=tile_gb,
                                   out=(X_bar, D_bar))
    eps_bar = eps_bar + eps2
    if nranks > 1:
        agreement((eps_bar, X_bar, D_bar), comm, audit_only=True,
                  label='selfenergy_diag_backward outputs')
    return eps_bar, X_bar, D_bar


# ---------------------------------------------------------------------------
# the continuation: forward only, and why
# ---------------------------------------------------------------------------

def thiele_jvp(z, f, df):
    """Thiele coefficients and their exact directional derivative along df.

    The recursion is complex add/divide, so forward mode carries it twice. No
    finite difference can measure this map's derivative (see
    `pade_continuation_jvp`).
    """
    n = len(z)
    g = np.zeros((n, n), dtype=complex)
    dg = np.zeros((n, n), dtype=complex)
    g[:, 0], dg[:, 0] = f, df
    for i in range(1, n):
        num = g[i - 1, i - 1] - g[i:, i - 1]
        den = (z[i:] - z[i - 1]) * g[i:, i - 1]
        dnum = dg[i - 1, i - 1] - dg[i:, i - 1]
        dden = (z[i:] - z[i - 1]) * dg[i:, i - 1]
        g[i:, i] = num / den
        dg[i:, i] = dnum / den - num * dden / den ** 2
    return g.diagonal().copy(), dg.diagonal().copy()


def pade_continuation_jvp(z_ord, data, direction, w):
    """d[Pade(data)(w)]/d[data] along `direction`, exactly, at fixed nodes.

    A diagnostic, not a gradient link. On water/cc-pVDZ the result is of order
    1e11 for a unit-norm perturbation of Sigma(i.omega) at 8 to 24 Thiele
    nodes: the inverse-difference recursion divides by differences that come
    arbitrarily close to zero, so the quasiparticle energy is not a usefully
    differentiable function of the self-energy, and finite differences of it
    are noise. A space-time GW nuclear gradient therefore needs a
    continuation-free quasiparticle equation (contour deformation, or the
    algebraic eta = 0 root of the quasi-boson route).
    """
    a, da = thiele_jvp(z_ord, data, direction)
    r, dr = a[-1] + 0j, da[-1] + 0j
    for i in range(len(a) - 2, -1, -1):
        den = 1.0 + (w - z_ord[i]) * r
        dr = da[i] / den - a[i] * ((w - z_ord[i]) * dr) / den ** 2
        r = a[i] / den
    return r, dr


def qp_energy_space_time(X, D, eps, nocc, grid, p, mu, pade_freq,
                         xc_correction=0.0, solver_mode='pole_strength',
                         tile_gb=ISDF_TILE_GB, fd_rel=1e-6):
    """(eps^QP_p, Z_p, z_ord, data) by the space-time route, forward only.

    No adjoint is returned. The chain up to Sigma_c(i.omega) is differentiable
    and `selfenergy_diag_backward` handles it; the Pade continuation that turns
    Sigma_c(i.omega) into a real-axis root is not (see `pade_continuation_jvp`),
    so a QP gradient assembled through it would be a number with no meaning.

    xc_correction is <Sigma_x - v_xc>_pp, zero on a Hartree-Fock reference.
    `SlicedFactors` are gathered whole, X_mo and D once each.
    """
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    p = int(p)
    transforms = sigma_transforms(eps, nocc, grid.tau_points, grid.omega_points,
                                  pade_freq, mu=mu)
    sigma = selfenergy_diag(X, D, eps, nocc, grid, [p], transforms, mu,
                            tile_gb=tile_gb)[0]
    # occupied states sample the negative branch, where Sigma(-i w) = conj
    s_p = np.conj(sigma[0]) if p < nocc else sigma[0]
    z_fit = imaginary_axis_sample_points(pade_freq, nocc, p, mu)[0]
    order = greedy_pade_order(z_fit, s_p)
    z_ord, data = z_fit[order], s_p[order]

    def sigma_c(w):
        coeffs = thiele_coefficients(z_ord, data)
        return pade_eval(np.array([w], dtype=complex), z_ord, coeffs)[0].real

    w_star = solve_qp_equation(lambda w: w - eps[p] - xc_correction - sigma_c(w),
                               eps[p], method=solver_mode)
    dw = fd_rel * max(abs(w_star), 1.0)
    z_factor = 1.0 / (1.0 - (sigma_c(w_star + dw) - sigma_c(w_star - dw))
                      / (2.0 * dw))
    return w_star, z_factor, z_ord, data
