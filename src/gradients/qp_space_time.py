"""The cubic quasiparticle gradient: contour deformation on space-time screening.

`space_time_adjoint` differentiates the ISDF/imaginary-time chain up to
chi0(i.omega); `contour_deformation` turns W(i.nu) into a quasiparticle energy
without a continuation. This wires the two, streaming, so that nothing in the
route is larger than proj(tau):

    proj(tau) = -2 D^T Pi(tau) D                (ntau, naux, naux)  the N^3 sweep
    Bp        = D^T (X_p * X)                   (naux, norb)
    per block of CD frequencies nu:
        chi0(i.nu) = sum_tau cosft[nu,tau] proj(tau)
        W Bp       = [I - chi0]^-1 Bp    ->  wc[nu,q] = Bp[:,q]^T (W - I) Bp[:,q]
    residues:   chi0(w') = sum_tau 2 w_tau cosh(w' tau) proj(tau)   (w' < gap)
    Newton on w = eps_p + Sigma^CD(w) from wc, Z from the closed-form slope
    reverse: the same blocks again, W Bp rebuilt per frequency, each block's
             chi0_bar folded straight into projbar(tau); the residues' rank-one
             adjoints folded into the SAME projbar; then one
             `polarizability_backward` and the Bp chain back to (X, D).

Peak memory is this rank's rows of proj(tau) and projbar(tau) (`ProjRows`),
one frequency's chi0 on the rank that factorizes it, one tau slice on the
rank that sweeps it, and the M-row tiles, all governed by one `tile_gb`: at
naux ~ 5000 a few GB, against the full three-index tensor's much larger
footprint, and independent of how many frequencies the CD quadrature has.
Rebuilding W Bp in the reverse pass costs one more LU per frequency than
holding it would -- naux^3 each, against the sweep's M^2 (norb + naux) per
tau -- and removes the (nfreq, naux, norb) array.

At the chlorophyllide hexamer (M 117762, nmo 10980, nocc 972, nvir 10008,
naux 28236, ntau 24, a four-state set, one frequency per block) proj(tau) is
153.1 GB and a rank's auxiliary rows of it 19.1 at 8 ranks, 9.6 at 16. Whole
on every rank stay X_mo + D gathered 36.9, the solve's X_bar + D_bar 36.9, the
slices and their adjoints 19.8 and the sweep's branches, adjoints, E and tiles
77.9 -- sums over the grid or over the tau partition -- and a frequency's or a
tau point's (naux, naux) matrices, 6.4 each, on the rank working on it. The
reverse call's peak per rank in GB: proj(tau), projbar and each block's
(ntau, naux, naux) fold temporary whole on every rank; without the temporary
(`fold_owned_frequencies`); and by rows at 8 / 16 ranks:

    route      frequency pass                 reverse sweep
               whole  no temporary  rows      whole  rows
    sop         601       448    181 / 162     344   216 / 207
    laplace     601       448    181 / 162     344   216 / 207
    explicit   refused: C_ov alone is 2197 GB

The frequency pass holds the two row blocks and, at the owner of a frequency,
its chi0, LU and adjoint and the identity, 48.2 with the slices' solves; the
sweep holds projbar's rows and one gathered slice. The forward pass, every
weight zero, holds 262 whole and 129 / 119 by rows.

Residues -- states whose root has crossed a neighbouring orbital energy, which
is every state but the frontier pair -- need W at a REAL frequency. Below the
particle-hole gap it has the same imaginary-time form as the imaginary axis
(cos -> cosh) and comes out of proj(tau): `LaplaceRealScreening`, cubic, with
no O(N^4) object; its adjoint is a rank-one update of projbar per residue. The
tau grid has to carry the frequency -- its e_min at most gap - w' -- and that
is checked on the pair energies themselves, not assumed. Above the gap chi0
has real poles and no imaginary-time form; only the explicit backend serves it,
through C_ov = B[:, occ, virt], (naux, nocc*nvirt), built from (X, D) and
chained back to them -- the O(N^4) end of the route, and now confined to
states whose residues lie above the gap (inner valence and core).

X and D may be `SlicedFactors`, each rank's grid rows of X_mo and D. Every
step above reads them whole -- the tau sweeps on both sides of each M^2 GEMM,
the slices Bp and C_ov as sums over the grid -- so each solve gathers X_mo and
D once at entry, holds them for the solve and its reverse pass and drops them
on return; the gathered arrays are the whole ones verbatim and every output is
the whole factors' bit for bit.
"""
from collections.abc import Mapping

import numpy as np
import scipy.linalg

from src.Base.constants import (EXPLICIT_RESIDUE_MAX_GB, ISDF_TILE_GB,
                                LAPLACE_SCREENING_TOL, QP_CD_NEWTON_TOL,
                                QP_POLE_OFFSET, SOP_N_POLES)
from src.Base.utils.mpi_grid import (agreement, current_comm, lockstep,
                                     partition, reduce_sum)
# cd_screening_contraction is re-exported, not used here: the routes below
# call the multi-state form, which is what it is a one-state wrapper for.
from src.SingleReference.GW.contour_deformation import (  # noqa: F401
    cd_screening_contraction, cd_screening_contraction_multi,
    residue_route_auto)
# calibrate_scissor and frozen_scissor are re-exported, not used here: a
# surface reaches the frozen shift through this module.
from src.SingleReference.GW.qp_states import (calibrate_scissor,
                                              frozen_scissor, scissor_route)
from src.SingleReference.GW.contour_deformation import qp_energy_cd
from src.SingleReference.GW.sum_over_poles import (compressible, qp_energy_sop,
                                                   sop_from_wc)
from src.gradients.contour_deformation_adjoint import (
    ExplicitRealScreeningAdjoint as ExplicitRealScreening,
    integral_term_backward, residue_terms_backward)
from src.gradients.sum_over_poles_adjoint import sigma_sop_backward
from src.Base.sliced_factors import whole_factor
from src.SingleReference.LinearResponse.space_time import (
    ProjRows, polarizability_projected_rows)
from src.gradients.space_time_adjoint import (LaplaceRealScreening,
                                              polarizability_backward,
                                              three_index_ov,
                                              three_index_ov_backward,
                                              three_index_slice,
                                              three_index_slice_backward)


def frozen_newton_seed(w0, p, eps, nocc, offset):
    """Where the Newton for orbital p starts.

    A frozen seed -- a scalar, or a mapping orbital -> energy holding the
    REFERENCE geometry's converged roots -- starts a displaced solve on the
    branch the reference found. The default eps_p pushed `offset` toward the
    chemical potential starts it on whichever branch is nearest the mean-field
    eigenvalue, and where a root has walked up to a neighbouring orbital energy
    those are different branches: f(w) = w - eps_p - Sigma(w) has a zero of
    vanishing pole strength on either side of every eps_q, and the iteration
    started at the guard reaches the satellite first.
    """
    if isinstance(w0, Mapping):
        got = w0.get(int(p))
        w0 = None if got is None else float(got)
    if w0 is None:
        return float(eps[p] + (offset if p < nocc else -offset))
    return float(w0)


def _stride_kw(sop_stride):
    """`stride=` for `sop_from_wc`, or nothing when the default is wanted.

    The poles are COMMON to all orbitals and placed from every stride-th
    column. Placement is a stronger lever on dSigma/domega than on Sigma, so
    this is a knob a gradient study needs and an energy one does not.
    """
    return {} if sop_stride is None else {'stride': int(sop_stride)}


def frozen_pole_offset(pole_offset, p):
    """(the guard offset orbital p uses, whether it may still be relaxed).

    A frozen offset -- a scalar, or a mapping orbital -> offset -- pins the
    Newton's guard band so that every geometry takes the same path to the root.
    None leaves the default, which each geometry relaxes on its own whenever a
    root sits inside the band, and an orbital a mapping does not name falls
    back to that.
    """
    if pole_offset is None:
        return QP_POLE_OFFSET, True
    if isinstance(pole_offset, Mapping):
        got = pole_offset.get(int(p))
        return (QP_POLE_OFFSET, True) if got is None else (float(got), False)
    return float(pole_offset), False


def integral_term_reverse(state, WtB, k, nu, wt):
    """(Bp_bar, chi0_bar, eps_bar) of ONE contour-deformation frequency for one state.

    Sigma^int_p(w) = -(1/pi) sum_k W_k Re[ sum_q wc[k,q] (w - eps_q) /
    ((w - eps_q)^2 + nu_k^2) ], so at a fixed frequency the adjoint on the
    screening is the weight vector `weights_k` on wc[k, :] -- which
    `integral_term_backward` pushes back onto the state's slice and onto
    chi0(i.nu_k) -- and the explicit eps dependence of the Lorentzian gives
    the third term. On the pole-model route the omega dependence is analytic
    and `wc_bar` already holds the whole weight, so that term is zero.

    state: one entry of the reverse pass's active set -- the state's slice
    `Bp`, its Z-weighted adjoint `zw`, `de` = w* - eps and `wc_bar` or None.
    """
    Bp = state['Bp']
    if state['wc_bar'] is not None:
        bb, _, cb = integral_term_backward(Bp, WtB, state['wc_bar'][k])
        return bb, cb, 0.0
    de = state['de']
    g = de / (de ** 2 + nu ** 2)
    coeff = -state['zw'] * wt / np.pi
    bb, wck, cb = integral_term_backward(Bp, WtB, coeff * g)
    return bb, cb, -coeff * wck * (nu ** 2 - de ** 2) / (de ** 2 + nu ** 2) ** 2


def require_explicit_fits(naux, nocc, nvir, p):
    """Refuse the explicit residue backend where its blocks cannot fit.

    C_ov is (naux, nocc*nvir) and the backend holds up to three such blocks at
    once on every rank (`EXPLICIT_RESIDUE_MAX_GB`); the Laplace backend and
    the pole model serve the same residues from proj(tau) without one.
    """
    gb = naux * nocc * nvir * 8 / 1e9
    if gb > EXPLICIT_RESIDUE_MAX_GB:
        raise MemoryError(
            f"residue_route 'explicit' (state {int(p)}) builds C_ov = "
            f"B[:, occ, virt], ({naux}, {nocc}*{nvir}), {gb:.3g} GB, and "
            f"holds up to three such blocks at once on every rank, above "
            f"EXPLICIT_RESIDUE_MAX_GB = {EXPLICIT_RESIDUE_MAX_GB:g} GB. Use "
            f"residue_route='laplace' (residues below the particle-hole gap, "
            f"from proj(tau)) or 'sop' (the pole model, no residues).")


def fold_residue_adjoints(proj_bar, backends, tau_indices=None):
    """Add the Laplace residues' adjoints on proj(tau) into proj_bar, in place.

    One (naux, naux) slice at a time: the backends' slices summed in state
    order from zero, then added to proj_bar -- the association of a zeroed
    projbar the whole per-state arrays were added into before the integral
    term's, so the result is that sum bit for bit and no per-state array
    exists. With no backend every slice gains an exact zero, as the zeroed
    projbar gave it. tau_indices: the slices the reverse sweep reads (a tau
    partition), the rest being unread.

    proj_bar may be `ProjRows`: every slice then gains its rows alone, the
    same elements with the same operations, for whichever rank sweeps it.
    """
    if isinstance(proj_bar, ProjRows):
        for k in range(proj_bar.shape[0]):
            s = np.zeros(proj_bar.rows.shape[1:])
            for rs in backends:
                s += rs.slice_adjoint(k, (proj_bar.r0, proj_bar.r1))
            proj_bar.rows[k] += s
        return
    which = (range(len(proj_bar)) if tau_indices is None
             else np.atleast_1d(tau_indices))
    for k in which:
        s = np.zeros(proj_bar.shape[1:])
        for rs in backends:
            s += rs.slice_adjoint(k)
        proj_bar[k] += s


def qp_gradient_space_time(X, D, eps, nocc, grid, nu_points, nu_weights, p,
                           mu=None, xc_correction=0.0, w0=None,
                           tol=QP_CD_NEWTON_TOL, residue_route='auto',
                           laplace_tol=LAPLACE_SCREENING_TOL,
                           want_grad=True, tile_gb=ISDF_TILE_GB,
                           pole_offset=None, route_out=None,
                           n_poles=SOP_N_POLES, sop_stride=None,
                           scissor=None, comm=None):
    """(eps^QP_p, Z, eps_bar, X_bar, D_bar), or (eps^QP_p, Z) with want_grad=False.

    grid:          TimeFrequencyGrid whose frequency axis IS the CD quadrature
                   (its cosft_wt maps proj(tau) onto nu_points) and whose tau
                   axis, for a state with residues, reaches down to
                   gap - w'_max -- `residue_route_auto` decides from that.
    residue_route: 'auto' (module docstring), 'laplace', 'explicit', 'none'
                   (raise if the iteration meets a residue), or 'sop'.
    'sop':         no residues at all -- W is modelled by `n_poles` auxiliary
                   poles and Sigma_c(omega) is closed form
                   (`sum_over_poles`), so the screening is never evaluated
                   off the imaginary axis and no backend is built. A state that
                   Eq. (27) does not admit CANNOT be served this way and falls
                   back to 'auto'; `route_out['residue_route']` says which ran,
                   and the decision is taken at the Newton START, frozen like
                   every other branch here.
    pole_offset:   the Newton's guard band, frozen by the caller
                   (`frozen_pole_offset`); None relaxes it per geometry.
    w0:            the Newton's start, a scalar or a mapping orbital -> energy
                   frozen by the caller (`frozen_newton_seed`).
    route_out:     a dict, if given, receives 'residue_route', 'residues',
                   'pole_offsets' and 'roots'.
    comm:          an MPI communicator (or `simulated_world` rank) whose ranks
                   all call this in lockstep; None is `current_comm()`. eps is
                   a `lockstep` of RANK 0's at entry, because the route and
                   the Newton start are read off it; X and D are identical by
                   construction (`separable_factors` leaves them so) and are
                   not broadcast again -- an audited run compares their
                   digests on entry and the outputs' on exit.
                   `qp_set_gradient`'s partition on
                   one state: proj(tau) and projbar held by auxiliary rows
                   (`ProjRows`), the tau sweep split over tau and its slices
                   exchanged into rows, its reverse split over tau and reduced
                   once (the three adjoints); the forward contraction and the
                   integral term's reverse pass split over the
                   contour-deformation frequencies, each frequency's chi0
                   gathered to its owner and its adjoint handed back as rows,
                   reduced once (wc; then Bp_bar and the frequency part of
                   eps_bar). The Newton, the route decision and the residues
                   are replicated -- taken from eps and from the ALREADY
                   REDUCED wc, never from a rank's own partial -- so they are
                   identical on every rank and the collectives below them keep
                   their shapes. Every transform over tau is one auxiliary row
                   per call, so wc -- and with it the root and Z on every
                   route -- and projbar are the serial ones bitwise at every
                   rank count; the adjoints are partial sums over the
                   partitions, exact up to summation order.
    """
    eps = np.asarray(eps, float)
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    p = int(p)
    nu_points = np.asarray(nu_points)
    if len(nu_points) != grid.cosft_wt.shape[0]:
        raise ValueError("the grid's frequency axis must be the CD quadrature")
    comm = current_comm() if comm is None else comm
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if nranks > 1:
        eps = lockstep(eps, comm)
        agreement((X, D, xc_correction, w0, pole_offset, scissor), comm,
                  audit_only=True, label='qp_gradient_space_time inputs')
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    nu_mine = partition(len(nu_points), rank, nranks) if nranks > 1 else None

    proj_tau = polarizability_projected_rows(X, D, eps, nocc, grid.tau_points,
                                             mu=mu, tile_memory_gb=tile_gb,
                                             comm=comm)
    Bp = three_index_slice(X, D, p, tile_gb=tile_gb)
    wc = cd_screening_contraction_multi(proj_tau, grid.cosft_wt, [Bp],
                                        tile_gb=tile_gb,
                                        freq_indices=nu_mine)[0]
    if nranks > 1:
        reduce_sum(wc, comm)                  # the others' rows are zero

    offset, relax = frozen_pole_offset(pole_offset, p)
    start = frozen_newton_seed(w0, p, eps, nocc, offset)
    route = residue_route
    shift = None
    if route == 'sop':
        route, shift = scissor_route(scissor, p, eps, nocc, start)
        if route == 'sop' and not compressible(start, eps, nocc)[0]:
            route = 'auto'
    if route == 'auto':
        route = residue_route_auto(grid, eps, nocc, start, laplace_tol)
    poles = amp = None
    if route in ('sop', 'scissor'):
        rs = None
    elif route == 'laplace':
        rs = LaplaceRealScreening(proj_tau, grid, eps, nocc, tol=laplace_tol)
    elif route == 'explicit':
        require_explicit_fits(D.shape[1], nocc, len(eps) - nocc, p)
        rs = ExplicitRealScreening(three_index_ov(X, D, eps, nocc, tile_gb=tile_gb),
                                   eps, nocc)
    elif route == 'none':
        rs = None
    else:
        raise ValueError(f"residue_route {route!r}")
    guard = {'pole_offset': offset}
    if route == 'scissor':
        # The shift is a constant, so the root moves with the orbital energy
        # alone and the screening never enters this state's gradient.
        w_star, z, res = float(eps[p] + xc_correction + shift), 1.0, []
    elif route == 'sop':
        poles, amp = sop_from_wc(wc, nu_points, eps, nocc, n_poles=n_poles,
                                 **_stride_kw(sop_stride))
        w_star, z = qp_energy_sop(p, amp, poles, eps, nocc,
                                  xc_correction=xc_correction, tol=tol,
                                  w0=start)
        res = []
    else:
        w_star, z, res = qp_energy_cd(p, Bp, eps, nocc, nu_points, nu_weights,
                                      xc_correction=xc_correction, tol=tol,
                                      w0=start, wc=wc, real_screening=rs,
                                      pole_offset=offset, relax_offset=relax,
                                      guard_out=guard)
    if route_out is not None:
        route_out['residue_route'] = route
        route_out['residues'] = res
        route_out['pole_offsets'] = {p: guard['pole_offset']}
        route_out['roots'] = {p: float(w_star)}
    if not want_grad:
        if nranks > 1:
            agreement((w_star, z), comm, audit_only=True,
                      label='qp_gradient_space_time outputs')
        return w_star, z

    # Sigma^int's reverse pass, streamed over the same frequency blocks
    naux = proj_tau.shape[-1]
    eye = np.eye(naux)
    de = w_star - eps
    eps_bar = np.zeros_like(eps)
    Bp_bar = np.zeros_like(Bp)
    if route == 'scissor':
        eps_bar[p] += z
        return w_star, z, eps_bar, np.zeros_like(X), np.zeros_like(D)
    wc_bar = None
    if route == 'sop':
        # The pole model's whole omega dependence is analytic, so its adjoint
        # on the imaginary-axis data is one transpose and enters the screening
        # exactly where the integral term's own weights do.
        eps_sop, wc_bar, _ = sigma_sop_backward(w_star, amp, poles, eps, nocc,
                                                nu_points, sigma_bar=z)
        eps_bar += eps_sop
    state = dict(Bp=Bp, zw=z, de=de, wc_bar=wc_bar)
    proj_bar = proj_tau.zeros_like()
    eps_bar_part = np.zeros_like(eps)
    for fb in proj_tau.blocks(grid.cosft_wt, tile_gb, nu_mine):
        for m, k in enumerate(fb.ks):
            # ONE state, so the factorization is used once and the plain solve
            # is it; `qp_set_gradient` takes the LU instead and shares it.
            WtB = np.linalg.solve(eye - fb.chi0[m], Bp)
            bb, fb.chi0[m], e_k = integral_term_reverse(
                state, WtB, k, nu_points[k], nu_weights[k])
            Bp_bar += bb
            eps_bar_part += e_k
        proj_bar.fold(grid.cosft_wt, fb)
    fb = None
    if nranks > 1:
        reduce_sum(Bp_bar, comm)
        reduce_sum(eps_bar_part, comm)
    eps_bar += eps_bar_part
    eps_bar[p] += z

    # Sigma^res: the shared part, then each backend's own chain
    X3 = D3 = None
    pushed = []
    if res:
        e_r, b_r, _ = residue_terms_backward(p, w_star, Bp, eps, nocc, res, z, rs)
        eps_bar += e_r
        Bp_bar += b_r
        if route == 'laplace':
            pushed.append(rs)
        else:
            e_c, Cov_bar = rs.adjoints()
            eps_bar += e_c
            X3, D3 = three_index_ov_backward(X, D, eps, nocc, Cov_bar,
                                             tile_gb=tile_gb)
            Cov_bar = None
    fold_residue_adjoints(proj_bar, pushed)
    del proj_tau, rs, pushed

    e2, X_bar, D_bar = polarizability_backward(proj_bar, X, D, eps, nocc,
                                               grid, mu=mu, tile_gb=tile_gb,
                                               tau_indices=tau_mine)
    if nranks > 1:
        for arr in (e2, X_bar, D_bar):
            reduce_sum(arr, comm)
    del proj_bar
    if X3 is not None:
        X_bar += X3
        D_bar += D3
    X2, D2 = three_index_slice_backward(X, D, p, Bp_bar, tile_gb=tile_gb)
    out = w_star, z, eps_bar + e2, X_bar + X2, D_bar + D2
    if nranks > 1:
        agreement(out, comm, audit_only=True,
                  label='qp_gradient_space_time outputs')
    return out


def qp_set_gradient(X, D, eps, nocc, grid, nu_points, nu_weights, states,
                    weights, mu=None, xc_correction=0.0, w0=None,
                    tol=QP_CD_NEWTON_TOL, residue_route='auto',
                    laplace_tol=LAPLACE_SCREENING_TOL,
                    tile_gb=ISDF_TILE_GB, pole_offset=None, route_out=None,
                    n_poles=SOP_N_POLES, sop_stride=None,
                    scissor=None, comm=None):
    """(eps^QP values, eps_bar, X_bar, D_bar) for sum_p weights[p] eps^QP_p.

    A BSE@GW excitation energy carries a weight on EVERY quasiparticle on the
    BSE diagonal, so the naive chain runs the whole reverse pass once per
    orbital. Almost all of it is shared: proj(tau) is built once, and every
    state's adjoint on chi0 folds into ONE projbar, so the expensive sweep --
    `polarizability_backward`, the only (M x M) work in the route -- runs once
    for the entire set instead of once per state. What remains per state is a
    three-index slice, a screening contraction and a Newton solve, all
    (naux, norb) at worst.

    Only states with a non-zero weight are solved at all, so a quasiparticle
    set smaller than the orbital basis costs proportionally less.

    THE REVERSE PASS RUNS FREQUENCY-OUTSIDE, STATES INSIDE, exactly as the
    forward contraction does: [1 - chi0(i.nu)] is factorized ONCE per
    frequency and every weighted state takes a triangular solve from it, and
    the states' adjoints on chi0(i.nu) are summed before the block is folded
    into projbar. The state loop used to sit outside and refactorize the same
    matrix once per state -- nstates naux^3 for nothing, the largest cost of
    the gradient at large naux. So the work is now three passes: every
    state's root and route (replicated, cheap), the integral term's reverse
    over frequencies, then each state's direct term, residues and slice
    adjoint.

    comm: an MPI communicator (or `simulated_world` rank) whose ranks all call
    this in lockstep; None is `current_comm()`. eps and the weights are a
    `lockstep` of RANK 0's at entry -- the routes are read off eps and the
    active set off the weights; X and D are identical by construction and are
    not broadcast again, an audited run comparing their digests on entry and
    the outputs' on exit. proj(tau) and projbar are held by auxiliary rows
    (`ProjRows`, 19.1 GB a rank at the chlorophyllide hexamer over 8 ranks
    where the whole array is 153): the tau sweep is split over tau and its
    slices exchanged into rows, its reverse split over tau and reduced once
    (the three adjoints); the forward contraction and the reverse pass are
    split over the contour-deformation frequencies, each frequency's chi0
    gathered whole to its owner and its adjoint handed back as rows, and
    reduced once (wc per state; then Bp_bar per state and the frequency part
    of eps_bar). Everything decided per state --
    the route, the guard band, the Newton itself -- is decided from eps and
    from the ALREADY REDUCED wc, never from a rank's own partial, so the
    decisions cannot diverge once the inputs agree. They are what fixes the
    ACTIVE SET, and the active set fixes how many buffers pass 2 reduces:
    ranks reading different eps would call that collective with different
    shapes. Every transform over tau is one auxiliary row per call, a shape
    no rank count changes, so every wc, every root and projbar itself are the
    serial ones bitwise at every rank count; the adjoints are partial sums
    over the partitions, exact up to summation order.

    pole_offset: the Newton's guard band, one value or a mapping orbital ->
                 offset frozen by the caller (`frozen_pole_offset`).
    w0:          the Newton's start, one value or a mapping orbital -> energy
                 frozen by the caller (`frozen_newton_seed`).
    route_out:   a dict, if given, receives 'routes', 'z', 'pole_offsets' and
                 'roots'.
    """
    eps = np.asarray(eps, float)
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    states = np.atleast_1d(states)
    weights = np.atleast_1d(weights)
    nu_points = np.asarray(nu_points)
    if len(nu_points) != grid.cosft_wt.shape[0]:
        raise ValueError("the grid's frequency axis must be the CD quadrature")
    comm = current_comm() if comm is None else comm
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if nranks > 1:
        # Before the routes are chosen: they are read off eps, and with the
        # weights' zeros they set the active set whose adjoints pass 2 reduces
        # as one buffer.
        eps, weights = lockstep((eps, weights), comm)
        agreement((X, D, xc_correction, w0, pole_offset, scissor), comm,
                  audit_only=True, label='qp_set_gradient inputs')
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    nu_mine = partition(len(nu_points), rank, nranks) if nranks > 1 else None

    proj_tau = polarizability_projected_rows(X, D, eps, nocc, grid.tau_points,
                                             mu=mu, tile_memory_gb=tile_gb,
                                             comm=comm)
    naux = proj_tau.shape[-1]
    eye = np.eye(naux)
    eps_bar = np.zeros_like(eps)
    X_bar = np.zeros_like(X)
    D_bar = np.zeros_like(D)
    w_stars = np.zeros(len(states))
    routes, offsets, roots, C_ov = {}, {}, {}, None
    z_of = np.zeros(len(states))

    Bps = [three_index_slice(X, D, int(p), tile_gb=tile_gb) for p in states]
    wcs = cd_screening_contraction_multi(proj_tau, grid.cosft_wt, Bps,
                                         tile_gb=tile_gb, freq_indices=nu_mine)
    if nranks > 1:
        wcs = list(reduce_sum(np.stack(wcs), comm))

    # PASS 1 -- every state's route, root and residue set. Replicated: a
    # Newton on wc and, for a residue, a rank-one push into its backend.
    active = []
    for si, (p, wgt) in enumerate(zip(states, weights)):
        p = int(p)
        Bp, wc = Bps[si], wcs[si]
        # PER STATE. <p|Sigma_x - v_xc|p> is a different number for every
        # orbital, so a scalar shared across the set would put the HOMO's
        # correction on the LUMO. A scalar is still accepted, and is the right
        # thing on Hartree-Fock, where every entry is zero.
        xc_p = (float(np.asarray(xc_correction).ravel()[si])
                if np.ndim(xc_correction) else float(xc_correction))
        offset, relax = frozen_pole_offset(pole_offset, p)
        start = frozen_newton_seed(w0, p, eps, nocc, offset)
        route = residue_route
        # A state Eq. (27) does not admit cannot be carried by the pole model
        # at any M, so it takes the residue route instead and the record says
        # so. On a valence quasiparticle set some states always will.
        shift = None
        if route == 'sop':
            route, shift = scissor_route(scissor, p, eps, nocc, start)
            if route == 'sop' and not compressible(start, eps, nocc)[0]:
                route = 'auto'
        if route == 'auto':
            route = residue_route_auto(grid, eps, nocc, start, laplace_tol)
        if route == 'laplace':
            rs = LaplaceRealScreening(proj_tau, grid, eps, nocc, tol=laplace_tol)
        elif route == 'explicit':
            if C_ov is None:
                require_explicit_fits(D.shape[1], nocc, len(eps) - nocc, p)
                C_ov = three_index_ov(X, D, eps, nocc, tile_gb=tile_gb)
            rs = ExplicitRealScreening(C_ov, eps, nocc)
        else:
            rs = None
        guard = {'pole_offset': offset}
        poles = amp = None
        if route == 'scissor':
            w_star, z, res = float(eps[p] + xc_p + shift), 1.0, []
        elif route == 'sop':
            poles, amp = sop_from_wc(wc, nu_points, eps, nocc, n_poles=n_poles,
                                     **_stride_kw(sop_stride))
            w_star, z = qp_energy_sop(p, amp, poles, eps, nocc,
                                      xc_correction=xc_p, tol=tol, w0=start)
            res = []
        else:
            w_star, z, res = qp_energy_cd(p, Bp, eps, nocc, nu_points, nu_weights,
                                          xc_correction=xc_p, tol=tol,
                                          w0=start, wc=wc, real_screening=rs,
                                          pole_offset=offset, relax_offset=relax,
                                          guard_out=guard)
        w_stars[si] = w_star
        z_of[si] = z
        routes[p] = route
        offsets[p] = guard['pole_offset']
        roots[p] = float(w_star)
        if wgt == 0.0:
            continue

        zw = z * wgt
        if route == 'scissor':
            eps_bar[p] += zw
            continue
        wc_bar = None
        if route == 'sop':
            # The pole model's whole omega dependence is analytic, so its
            # adjoint on the imaginary-axis data is one transpose and enters
            # the screening exactly where the integral term's own weights do.
            eps_sop, wc_bar, _ = sigma_sop_backward(w_star, amp, poles, eps,
                                                    nocc, nu_points,
                                                    sigma_bar=zw)
            eps_bar += eps_sop
        active.append(dict(si=si, p=p, zw=zw, de=w_star - eps, Bp=Bp,
                           wc_bar=wc_bar, res=res, rs=rs, route=route))

    # PASS 2 -- the integral term's reverse pass, frequency outside, over the
    # frequencies this rank owns. Partial over frequencies: reduced once.
    Bp_bars = {a['si']: np.zeros_like(a['Bp']) for a in active}
    eps_bar_part = np.zeros_like(eps)
    if active:
        proj_bar = proj_tau.zeros_like()
        for fb in proj_tau.blocks(grid.cosft_wt, tile_gb, nu_mine):
            for m, k in enumerate(fb.ks):
                nu, wt = nu_points[k], nu_weights[k]
                lu = scipy.linalg.lu_factor(eye - fb.chi0[m])
                chi0_bar = np.zeros((naux, naux))
                for a in active:
                    WtB = scipy.linalg.lu_solve(lu, a['Bp'])
                    bb, cb, e_k = integral_term_reverse(a, WtB, k, nu, wt)
                    Bp_bars[a['si']] += bb
                    chi0_bar += cb
                    eps_bar_part += e_k
                fb.chi0[m] = chi0_bar
            proj_bar.fold(grid.cosft_wt, fb)
        fb = lu = chi0_bar = None
        if nranks > 1:
            reduce_sum(eps_bar_part, comm)
            stacked = reduce_sum(np.stack([Bp_bars[a['si']] for a in active]),
                                 comm)
            for a, bar in zip(active, stacked):
                Bp_bars[a['si']] = bar
    eps_bar += eps_bar_part

    # PASS 3 -- per state: the direct term, the residues, the slice adjoint.
    # Replicated, like pass 1. Each backend leaves with its state, so the
    # explicit route holds one Cov_bar at a time; the Laplace backends keep
    # only their recorded pushes, folded into projbar after the pass.
    rs = C_ov = None
    pushed = []
    for a in active:
        p, zw, Bp, res, route = a['p'], a['zw'], a['Bp'], a['res'], a['route']
        rs = a.pop('rs')
        Bp_bar = Bp_bars[a['si']]
        eps_bar[p] += zw
        if res:
            e_r, b_r, _ = residue_terms_backward(p, roots[p], Bp, eps, nocc,
                                                 res, zw, rs)
            eps_bar += e_r
            Bp_bar += b_r
            if route == 'laplace':
                pushed.append(rs)
            else:
                e_c, Cov_bar = rs.adjoints()
                eps_bar += e_c
                X3, D3 = three_index_ov_backward(X, D, eps, nocc, Cov_bar,
                                                 tile_gb=tile_gb)
                X_bar += X3
                D_bar += D3
                Cov_bar = None
        rs = None
        X2, D2 = three_index_slice_backward(X, D, p, Bp_bar, tile_gb=tile_gb)
        X_bar += X2
        D_bar += D2
    if active:
        fold_residue_adjoints(proj_bar, pushed)
    pushed = None
    del proj_tau
    if route_out is not None:
        route_out['routes'] = routes
        # The guard band each state's Newton actually used. A caller freezing
        # the branch records these and hands them back at the next geometry;
        # re-deciding them there is a discontinuity in the same family as
        # re-deciding the residue route.
        route_out['pole_offsets'] = offsets
        # The converged roots, which the same caller hands back as `w0` so that
        # the displaced Newton starts on the branch this one found.
        route_out['roots'] = roots
    if active:
        e2, Xs, Ds = polarizability_backward(proj_bar, X, D, eps, nocc,
                                             grid, mu=mu, tile_gb=tile_gb,
                                             tau_indices=tau_mine)
        del proj_bar
        if nranks > 1:
            for arr in (e2, Xs, Ds):
                reduce_sum(arr, comm)
    else:
        # Nothing carries a weight: the sweep would run on a zero projbar and
        # return exact zeros, so it is not run. The active set is identical on
        # every rank, so every rank skips the same collectives.
        e2, Xs, Ds = np.zeros_like(eps), np.zeros_like(X), np.zeros_like(D)
    if route_out is not None:
        # The pole strengths, for callers that must weight a term the Newton
        # condition renormalizes. Anything added to the RIGHT of
        # w = eps_p + Delta_p + Sigma_c(w) reaches dw multiplied by Z, so a
        # caller carrying its own Delta (the Sigma_x - v_xc correction on a
        # Kohn-Sham reference) needs these and cannot reconstruct them.
        route_out['z'] = z_of
        route_out['routes'] = routes
    out = w_stars, eps_bar + e2, X_bar + Xs, D_bar + Ds
    if nranks > 1:
        agreement(out, comm, audit_only=True, label='qp_set_gradient outputs')
    return out
