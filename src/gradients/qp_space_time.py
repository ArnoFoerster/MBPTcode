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
             adjoints folded into the same projbar; then one
             `polarizability_backward` and the Bp chain back to (X, D).

Peak memory is this rank's rows of proj(tau) and projbar(tau) (`ProjRows`),
one frequency's chi0 on the rank that factorizes it, one tau slice on the
rank that sweeps it, and the M-row tiles, all governed by one `tile_gb` and
independent of the number of CD frequencies. Rebuilding W Bp in the reverse
pass costs one more LU per frequency (naux^3, against the sweep's
M^2 (norb + naux) per tau) and removes the (nfreq, naux, norb) array.

Residues (states whose root has crossed a neighbouring orbital energy, which
is every state but the frontier pair) need W at a real frequency. Below the
particle-hole gap it has the same imaginary-time form as on the imaginary axis
(cos -> cosh) and comes out of proj(tau): `LaplaceRealScreening`, cubic, with
no O(N^4) object; its adjoint is a rank-one update of projbar per residue. The
tau grid has to carry the frequency (its e_min at most gap - w'), which is
checked on the pair energies. Above the gap chi0 has real poles and no
imaginary-time form; only the explicit backend serves it, through
C_ov = B[:, occ, virt], (naux, nocc*nvirt), built from (X, D) and chained back
to them: the O(N^4) end of the route, confined to states whose residues lie
above the gap (inner valence and core).

X and D may be `SlicedFactors`, each rank's grid rows of X_mo and D. Every
step above reads them whole, so each solve gathers X_mo and D once at entry
and drops them on return; the gathered arrays are the whole ones verbatim and
every output is the whole factors' bit for bit.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from src.Base.constants import (EXPLICIT_RESIDUE_MAX_GB, ISDF_TILE_GB,
                                LAPLACE_SCREENING_TOL, QP_CD_NEWTON_TOL,
                                QP_POLE_OFFSET, SOP_N_POLES)
from src.Base.utils.mpi_grid import (agreement, allgather_rows,
                                     contiguous_block, current_comm, lockstep,
                                     mpi_map, partition, reduce_sum)
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
from src.SingleReference.GW.sum_over_poles import (compressible,
                                                   pole_amplitudes,
                                                   qp_energy_sop, sop_from_wc)
from src.gradients.contour_deformation_adjoint import (
    ExplicitRealScreeningAdjoint as ExplicitRealScreening,
    integral_term_backward, residue_terms_backward)
from src.gradients.sum_over_poles_adjoint import sigma_sop_backward
from src.Base.sliced_factors import GridTileRows, SlicedFactors, whole_factor
from src.SingleReference.LinearResponse.space_time import (
    ProjRows, polarizability_projected_rows)
from src.gradients.space_time_adjoint import (
    LaplaceRealScreening, polarizability_backward,
    polarizability_backward_rows, three_index_ov, three_index_ov_backward,
    three_index_slice, three_index_slice_backward,
    three_index_slice_backward_rows)


def frozen_newton_seed(w0, p, eps, nocc, offset):
    """Where the Newton for orbital p starts.

    A frozen seed (a scalar, or a mapping orbital -> the reference geometry's
    converged root) starts a displaced solve on the branch the reference
    found. The default, eps_p pushed `offset` toward the chemical potential,
    starts on the branch nearest the mean-field eigenvalue. Where a root has
    walked up to a neighbouring orbital energy these differ:
    f(w) = w - eps_p - Sigma(w) has a zero of vanishing pole strength on
    either side of every eps_q, and the iteration started at the guard
    reaches the satellite first.
    """
    if isinstance(w0, Mapping):
        got = w0.get(int(p))
        w0 = None if got is None else float(got)
    if w0 is None:
        return float(eps[p] + (offset if p < nocc else -offset))
    return float(w0)


def static_term(xc_correction, si):
    """The static term <p|Sigma_x - v_xc|p> + Sigma^env_pp the si-th state of
    a set is solved with: one entry per state, or one scalar for all."""
    return (float(np.asarray(xc_correction).ravel()[si])
            if np.ndim(xc_correction) else float(xc_correction))


def frozen_on_pole_model(p, w0, sop_poles):
    """Whether the reference solve put orbital p on the pole model: True with
    its poles frozen, False with its root frozen and no poles, None before.

    The Newton start is eps_p at the reference solve and the frozen root
    after it, and the quasiparticle correction can carry a state across the
    Eq. (27) limit (formaldehyde PBE0 orbitals 4, 5 and 10), so the verdict is
    frozen with the root and the poles, as the scissor tiers are; only a state
    with nothing frozen reads `compressible` at its Newton start.
    """
    if isinstance(sop_poles, Mapping) and sop_poles.get(int(p)) is not None:
        return True
    if isinstance(w0, Mapping) and w0.get(int(p)) is not None:
        return False
    return None


def pole_model_admitted(p, start, eps, nocc, w0, sop_poles):
    """Eq. (27)'s verdict on orbital p: the frozen one once the reference
    solve has decided it (`frozen_on_pole_model`), else `compressible` at the
    Newton start."""
    frozen = frozen_on_pole_model(p, w0, sop_poles)
    return compressible(start, eps, nocc)[0] if frozen is None else frozen


def _stride_kw(sop_stride):
    """`stride=` for `sop_from_wc`, or nothing when the default is wanted.

    The poles are common to all orbitals and placed from every stride-th
    column. Placement moves dSigma/domega more than Sigma, so a gradient study
    needs this knob and an energy one does not.
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


def sop_model(wc, nu_points, eps, nocc, p, sop_poles, n_poles, sop_stride):
    """(poles, amplitudes) of orbital p's pole model, on the frozen poles if any.

    The pole positions are a realization of Sigma(omega), fitted by vector
    fitting on the reference geometry's wc and then held: the adjoint
    (`sigma_sop_backward`) differentiates A = F^+ wc at fixed poles, so a set
    re-fitted per geometry makes the force the derivative of a different
    surface from the one the energy walks. With the poles frozen only the
    amplitudes move, and they are linear in wc.

    sop_poles: a mapping orbital -> frozen pole positions, or None; an orbital
               it does not name is fitted here (`sop_from_wc`).
    """
    frozen = None if sop_poles is None else sop_poles.get(int(p))
    if frozen is None:
        return sop_from_wc(wc, nu_points, eps, nocc, n_poles=n_poles,
                           **_stride_kw(sop_stride))
    poles = np.asarray(frozen, float)
    return poles, pole_amplitudes(wc, poles, nu_points)


def slices_over_ranks(X, D, states, tile_gb, comm=None):
    """[B_p for p in states], each slice built by one rank and gathered onto
    every one; serially, a plain list.

    A slice is O(M naux norb) per state, so building a window of hundreds of
    states on every rank would not divide with the rank count. Each slice is
    the same `three_index_slice` call on the same whole factors wherever it
    runs and travels verbatim (`allgather_rows`): every rank holds the serial
    slices bitwise, as views into one (nstates, naux, norb) array.
    """
    nranks = 1 if comm is None else comm.Get_size()
    if nranks == 1:
        return [three_index_slice(X, D, int(p), tile_gb=tile_gb)
                for p in states]
    out = np.zeros((len(states), D.shape[1], X.shape[1]))
    start, stop = contiguous_block(len(states), comm.Get_rank(), nranks)
    for i in range(start, stop):
        out[i] = three_index_slice(X, D, int(states[i]), tile_gb=tile_gb)
    allgather_rows(out, comm)
    return list(out)


def fit_poles_over_ranks(states, wcs, nu_points, eps, nocc, residue_route,
                         scissor, w0, pole_offset, sop_poles, n_poles,
                         sop_stride, comm=None):
    """`sop_poles` with the fit of every pole-model state it lacks added, the
    fits striped over the ranks (`mpi_map`) and gathered onto every one.

    The fit is a dense least squares of about norb^3 a state, the one cost of
    pass 1 that grows with the molecule. It reads only the state's wc, which
    is reduced and identical on every rank, so the owner's poles are the
    serial poles bitwise; `sop_model` then reads them as frozen poles, and
    the amplitudes are the call `sop_from_wc` would have made. The states
    fitted are those pass 1 sends to the pole model (route, scissor tier,
    Eq. (27)'s verdict, `pole_model_admitted`) and has no frozen poles for.
    """
    if residue_route != 'sop':
        return sop_poles
    need = []
    for si, p in enumerate(states):
        p = int(p)
        if sop_poles is not None and sop_poles.get(p) is not None:
            continue
        offset, _ = frozen_pole_offset(pole_offset, p)
        start = frozen_newton_seed(w0, p, eps, nocc, offset)
        route, _ = scissor_route(scissor, p, eps, nocc, start)
        if route == 'sop' and pole_model_admitted(p, start, eps, nocc, w0,
                                                  sop_poles):
            need.append(si)
    if not need:
        return sop_poles

    def fit(si):
        return sop_from_wc(wcs[si], nu_points, eps, nocc, n_poles=n_poles,
                           **_stride_kw(sop_stride))[0]

    out = dict(sop_poles or {})
    for si, poles in zip(need, mpi_map(fit, need, comm)):
        out[int(states[si])] = poles
    return out


def integral_term_reverse(state, WtB, k, nu, wt):
    """(Bp_bar, chi0_bar, eps_bar) of one contour-deformation frequency for one state.

    Sigma^int_p(w) = -(1/pi) sum_k W_k Re[ sum_q wc[k,q] (w - eps_q) /
    ((w - eps_q)^2 + nu_k^2) ], so at a fixed frequency the adjoint on the
    screening is the weight vector `weights_k` on wc[k, :], which
    `integral_term_backward` pushes back onto the state's slice and onto
    chi0(i.nu_k); the explicit eps dependence of the Lorentzian gives the
    third term. On the pole-model route the omega dependence is analytic
    and `wc_bar` already holds the whole weight, so that term is zero.

    state: one entry of the reverse pass's active set: the state's slice
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
    order from zero, then added to proj_bar (this association is part of the
    bitwise result), so no per-state array exists. With no backend every slice
    gains an exact zero. tau_indices: the slices the reverse sweep reads (a
    tau partition); the rest are unread.

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
                           scissor=None, sop_poles=None, comm=None):
    """(eps^QP_p, Z, eps_bar, X_bar, D_bar), or (eps^QP_p, Z) with want_grad=False.

    grid:          TimeFrequencyGrid whose frequency axis is the CD quadrature
                   (its cosft_wt maps proj(tau) onto nu_points) and whose tau
                   axis, for a state with residues, reaches down to
                   gap - w'_max; `residue_route_auto` decides from that.
    residue_route: 'auto' (module docstring), 'laplace', 'explicit', 'none'
                   (raise if the iteration meets a residue), or 'sop'.
    'sop':         no residues: W is modelled by `n_poles` auxiliary poles and
                   Sigma_c(omega) is closed form (`sum_over_poles`), so the
                   screening is never evaluated off the imaginary axis. A
                   state that Eq. (27) does not admit falls back to 'auto';
                   `route_out['residue_route']` says which ran. The decision
                   is taken at the reference solve's Newton start and then
                   frozen with the root and the poles
                   (`frozen_on_pole_model`), like every other branch here.
    pole_offset:   the Newton's guard band, frozen by the caller
                   (`frozen_pole_offset`); None relaxes it per geometry.
    w0:            the Newton's start, a scalar or a mapping orbital -> energy
                   frozen by the caller (`frozen_newton_seed`).
    sop_poles:     the pole model's positions, a mapping orbital -> poles
                   frozen by the caller (`sop_model`); None fits them here.
    route_out:     a dict, if given, receives 'residue_route', 'residues',
                   'pole_offsets', 'roots' and 'sop_poles'.
    comm:          an MPI communicator (or `simulated_world` rank) whose ranks
                   all call this in lockstep; None is `current_comm()`. eps is
                   a `lockstep` of rank 0's at entry (the route and the Newton
                   start are read off it); X and D are identical by
                   construction and only audited. The partition is
                   `qp_set_gradient`'s on one state. The Newton, the route
                   decision and the residues are replicated, taken from eps
                   and the reduced wc, so the collectives below them keep
                   their shapes; wc, the root, Z and projbar are the serial
                   ones bitwise at every rank count, and the adjoints are
                   partial sums exact up to summation order.
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
        agreement((X, D, xc_correction, w0, pole_offset, scissor, sop_poles),
                  comm, audit_only=True, label='qp_gradient_space_time inputs')
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
        if route == 'sop' and not pole_model_admitted(p, start, eps, nocc, w0,
                                                      sop_poles):
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
        poles, amp = sop_model(wc, nu_points, eps, nocc, p, sop_poles, n_poles,
                               sop_stride)
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
        route_out['sop_poles'] = {} if poles is None else {p: poles}
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
        # where the integral term's own weights do.
        eps_sop, wc_bar, _ = sigma_sop_backward(w_star, amp, poles, eps, nocc,
                                                nu_points, sigma_bar=z)
        eps_bar += eps_sop
    state = dict(Bp=Bp, zw=z, de=de, wc_bar=wc_bar)
    proj_bar = proj_tau.zeros_like()
    eps_bar_part = np.zeros_like(eps)
    for fb in proj_tau.blocks(grid.cosft_wt, tile_gb, nu_mine):
        for m, k in enumerate(fb.ks):
            # one state, so a plain solve; `qp_set_gradient` takes the LU and
            # shares it over the states
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
        X3 = D3 = None
    three_index_slice_backward(X, D, p, Bp_bar, tile_gb=tile_gb,
                               out=(X_bar, D_bar))
    out = w_star, z, eps_bar + e2, X_bar, D_bar
    if nranks > 1:
        agreement(out, comm, audit_only=True,
                  label='qp_gradient_space_time outputs')
    return out


@dataclass
class QPSetTape:
    """What `qp_set_gradient`'s forward built, for a later call to read back.

    proj(tau) held by rows, the states' slices Bps and their contractions
    wcs, with the arguments they were built from: the factor objects
    themselves, eps, mu, the tile budget, the states, the tau points and the
    transform onto the CD frequencies. A later call on the same objects and
    bits reads proj and Bps from here, and wcs as well when the frequency
    axis is also the same -- the same arrays, so the same bits. `release`
    drops the arrays once no later call is to read them, and a released tape
    applies to nothing.
    """

    X: object
    D: object
    eps: np.ndarray
    mu: object
    tile_gb: float
    states: np.ndarray
    tau_points: np.ndarray
    cosft_wt: np.ndarray
    proj_tau: ProjRows
    Bps: list
    wcs: list

    def release(self):
        """Drop what the tape holds."""
        self.X = self.D = self.proj_tau = self.Bps = self.wcs = None

    def reads(self, X, D, eps, mu, tile_gb, states, grid):
        """(proj and Bps apply, wcs apply as well) for a call on these."""
        same = (self.proj_tau is not None and self.X is X and self.D is D
                and np.array_equal(self.eps, eps) and self.mu == mu
                and self.tile_gb == tile_gb
                and np.array_equal(self.states, states)
                and np.array_equal(self.tau_points, grid.tau_points))
        return same, same and np.array_equal(self.cosft_wt, grid.cosft_wt)


def _audited(factor):
    """What an audited run digests of a factor: the array, or the grid
    points of `SlicedFactors` (their rows are this rank's alone)."""
    return factor.coords if isinstance(factor, SlicedFactors) else factor


def _grid_shape(factor, name):
    """The whole shape of a factor, (M, ncol), without gathering it."""
    if isinstance(factor, SlicedFactors):
        return (factor.npts, getattr(factor, name).shape[1])
    return np.shape(factor)


def qp_set_gradient(X, D, eps, nocc, grid, nu_points, nu_weights, states,
                    weights, mu=None, xc_correction=0.0, w0=None,
                    tol=QP_CD_NEWTON_TOL, residue_route='auto',
                    laplace_tol=LAPLACE_SCREENING_TOL,
                    tile_gb=ISDF_TILE_GB, pole_offset=None, route_out=None,
                    n_poles=SOP_N_POLES, sop_stride=None,
                    scissor=None, sop_poles=None, comm=None,
                    rows_block=None, tape=None):
    """(eps^QP values, eps_bar, X_bar, D_bar) for sum_p weights[p] eps^QP_p.

    A BSE@GW excitation energy carries a weight on every quasiparticle on the
    BSE diagonal. proj(tau) is built once and every state's adjoint on chi0
    folds into one projbar, so the M^2 sweep (`polarizability_backward`) runs
    once for the set; per state remain a three-index slice, a screening
    contraction and a Newton solve, (naux, norb) at worst. Only states with a
    non-zero weight are differentiated; with every weight zero (a forward
    solve) X_bar and D_bar are read-only broadcasts of one zero.

    The work is three passes: every state's root and route (replicated), the
    integral term's reverse over frequencies, then each state's direct term,
    residues and slice adjoint. The reverse pass runs frequency outside,
    states inside, as the forward contraction does: [1 - chi0(i.nu)] is
    factorized once per frequency, every weighted state takes a triangular
    solve from it, and the states' adjoints on chi0(i.nu) are summed before
    the block is folded into projbar.

    comm: an MPI communicator (or `simulated_world` rank) whose ranks all call
    this in lockstep; None is `current_comm()`. eps and the weights are a
    `lockstep` of rank 0's at entry (the routes are read off eps, the active
    set off the weights); X and D are identical by construction and only
    audited. proj(tau) and projbar are held by auxiliary rows (`ProjRows`):
    the tau sweep is split over tau and its slices exchanged into rows, its
    reverse split over tau and reduced once; the forward contraction and the
    reverse pass are split over the CD frequencies, each frequency's chi0
    gathered whole to its owner and its adjoint handed back as rows, reduced
    once. Everything decided per state (route, guard band, Newton) is decided
    from eps and the reduced wc, never from a rank's own partial: it fixes
    the active set, and with it the shapes pass 2 reduces. Every transform
    over tau is one auxiliary row per call, so every wc, root and projbar are
    the serial ones bitwise at every rank count; the adjoints are partial
    sums over the partitions, exact up to summation order.

    pole_offset: the Newton's guard band, one value or a mapping orbital ->
                 offset frozen by the caller (`frozen_pole_offset`).
    w0:          the Newton's start, one value or a mapping orbital -> energy
                 frozen by the caller (`frozen_newton_seed`).
    sop_poles:   the pole model's positions, a mapping orbital -> poles frozen
                 by the caller (`sop_model`); None fits every state's here.
    route_out:   a dict, if given, receives 'routes', 'z', 'pole_offsets',
                 'roots', 'sop_poles' and 'tape' (`QPSetTape`).
    tape:        a `QPSetTape` of an earlier call; proj(tau), the slices and
                 their contractions are read from it where it applies
                 (`QPSetTape.reads`), bitwise, and a forward (every weight
                 zero) on a tape that applies gathers no factor at all.
    rows_block:  None returns X_bar and D_bar whole, reduced over the tau
                 partition. A tile edge returns them as `GridTileRows` in
                 tiles of that many points owned t % size: the slice
                 adjoints per tile and the M^2 reverse sweep tile-major over
                 every tau (`polarizability_backward_rows`), so no rank
                 holds a whole pair and the rows are the same bits at every
                 rank count for the same projbar and Bp_bar.
    """
    eps = np.asarray(eps, float)
    X_in, D_in = X, D
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
        agreement((_audited(X_in), _audited(D_in), xc_correction, w0,
                   pole_offset, scissor, sop_poles),
                  comm, audit_only=True, label='qp_set_gradient inputs')
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    nu_mine = partition(len(nu_points), rank, nranks) if nranks > 1 else None
    whole = {}

    def factors():
        """X_mo and D whole, gathered at the first call that reads them."""
        if not whole:
            whole['X'] = whole_factor(X_in, 'X_mo')
            whole['D'] = whole_factor(D_in, 'D')
        return whole['X'], whole['D']

    same, same_wc = ((False, False) if tape is None
                     else tape.reads(X_in, D_in, eps, mu, tile_gb, states,
                                     grid))
    if same:
        proj_tau, Bps = tape.proj_tau, tape.Bps
    else:
        X, D = factors()
        proj_tau = polarizability_projected_rows(
            X, D, eps, nocc, grid.tau_points, mu=mu, tile_memory_gb=tile_gb,
            comm=comm)
        Bps = slices_over_ranks(X, D, states, tile_gb, comm)
    naux = proj_tau.shape[-1]
    eye = np.eye(naux)
    eps_bar = np.zeros_like(eps)
    w_stars = np.zeros(len(states))
    routes, offsets, roots, fitted, C_ov = {}, {}, {}, {}, None
    z_of = np.zeros(len(states))

    if same_wc:
        wcs = tape.wcs
    else:
        wcs = cd_screening_contraction_multi(proj_tau, grid.cosft_wt, Bps,
                                             tile_gb=tile_gb,
                                             freq_indices=nu_mine)
        if nranks > 1:
            wcs = list(reduce_sum(np.stack(wcs), comm))
    if route_out is not None:
        route_out['tape'] = QPSetTape(X_in, D_in, eps.copy(), mu, tile_gb,
                                      states.copy(),
                                      np.array(grid.tau_points),
                                      np.array(grid.cosft_wt), proj_tau, Bps,
                                      wcs)

    # Pass 1: every state's route, root and residue set. Replicated: a
    # Newton on wc and, for a residue, a rank-one push into its backend. The
    # pole fits are the exception, striped over the ranks first.
    sop_poles = fit_poles_over_ranks(states, wcs, nu_points, eps, nocc,
                                     residue_route, scissor, w0, pole_offset,
                                     sop_poles, n_poles, sop_stride, comm)
    active = []
    for si, (p, wgt) in enumerate(zip(states, weights)):
        p = int(p)
        Bp, wc = Bps[si], wcs[si]
        # Per state: <p|Sigma_x - v_xc|p> differs from orbital to orbital. A
        # scalar is accepted, and is right on Hartree-Fock, where every entry
        # is zero.
        xc_p = static_term(xc_correction, si)
        offset, relax = frozen_pole_offset(pole_offset, p)
        start = frozen_newton_seed(w0, p, eps, nocc, offset)
        route = residue_route
        # A state Eq. (27) does not admit cannot be carried by the pole model
        # at any M, so it takes the residue route and the record says so. The
        # verdict is the reference solve's once it has frozen the root and the
        # poles (`frozen_on_pole_model`).
        shift = None
        if route == 'sop':
            route, shift = scissor_route(scissor, p, eps, nocc, start)
            if route == 'sop' and not pole_model_admitted(
                    p, start, eps, nocc, w0, sop_poles):
                route = 'auto'
        if route == 'auto':
            route = residue_route_auto(grid, eps, nocc, start, laplace_tol)
        if route == 'laplace':
            rs = LaplaceRealScreening(proj_tau, grid, eps, nocc, tol=laplace_tol)
        elif route == 'explicit':
            if C_ov is None:
                require_explicit_fits(naux, nocc, len(eps) - nocc, p)
                C_ov = three_index_ov(*factors(), eps, nocc, tile_gb=tile_gb)
            rs = ExplicitRealScreening(C_ov, eps, nocc)
        else:
            rs = None
        guard = {'pole_offset': offset}
        poles = amp = None
        if route == 'scissor':
            w_star, z, res = float(eps[p] + xc_p + shift), 1.0, []
        elif route == 'sop':
            poles, amp = sop_model(wc, nu_points, eps, nocc, p, sop_poles,
                                   n_poles, sop_stride)
            fitted[p] = poles
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
            # the screening where the integral term's own weights do.
            eps_sop, wc_bar, _ = sigma_sop_backward(w_star, amp, poles, eps,
                                                    nocc, nu_points,
                                                    sigma_bar=zw)
            eps_bar += eps_sop
        active.append(dict(si=si, p=p, zw=zw, de=w_star - eps, Bp=Bp,
                           wc_bar=wc_bar, res=res, rs=rs, route=route))

    # Pass 2: the integral term's reverse pass, frequency outside, over the
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

    # Pass 3: per state, the direct term, the residues, the slice adjoint.
    # Replicated, like pass 1. Each backend leaves with its state, so the
    # explicit route holds one Cov_bar at a time; the Laplace backends keep
    # only their recorded pushes, folded into projbar after the pass.
    rs = C_ov = None
    pushed = []
    # The factors' adjoints exist only where a state carries one: one pair,
    # every state's slice and residue adjoints added into it in turn.
    X_bar = D_bar = None
    if active:
        X, D = factors()
        if rows_block is None:
            X_bar, D_bar = np.zeros_like(X), np.zeros_like(D)
        else:
            X_bar = GridTileRows.zeros(X.shape[0], X.shape[1], rows_block,
                                       comm)
            D_bar = GridTileRows.zeros(D.shape[0], naux, rows_block, comm)
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
                X3 = D3 = Cov_bar = None
        rs = None
        if rows_block is None:
            three_index_slice_backward(X, D, p, Bp_bar, tile_gb=tile_gb,
                                       out=(X_bar, D_bar))
        else:
            three_index_slice_backward_rows(X, D, p, Bp_bar, (X_bar, D_bar))
    if active:
        fold_residue_adjoints(proj_bar, pushed)
    pushed = None
    del proj_tau
    if route_out is not None:
        route_out['routes'] = routes
        # The guard band each state's Newton used. A caller freezing
        # the branch records these and hands them back at the next geometry;
        # re-deciding them there is a discontinuity in the same family as
        # re-deciding the residue route.
        route_out['pole_offsets'] = offsets
        # The converged roots, which the same caller hands back as `w0` so that
        # the displaced Newton starts on the branch this one found.
        route_out['roots'] = roots
        # The pole positions each pole-model state used, handed back as
        # `sop_poles` so that the displaced geometries share the reference fit.
        route_out['sop_poles'] = fitted
    if active and rows_block is not None:
        # the sweep reads the gathered X_mo through column views of its
        # branches, so no copy of either stands beside it
        e2, Xs, Ds = polarizability_backward_rows(proj_bar, X, D, eps, nocc,
                                                  grid, mu=mu,
                                                  block=rows_block, comm=comm)
        del proj_bar
        X_bar += Xs
        D_bar += Ds
        del Xs, Ds
    elif active:
        e2, Xs, Ds = polarizability_backward(proj_bar, X, D, eps, nocc,
                                             grid, mu=mu, tile_gb=tile_gb,
                                             tau_indices=tau_mine)
        del proj_bar
        if nranks > 1:
            for arr in (e2, Xs, Ds):
                reduce_sum(arr, comm)
        X_bar += Xs
        D_bar += Ds
        del Xs, Ds
    else:
        # Nothing carries a weight: the sweep would run on a zero projbar and
        # return exact zeros, so it is not run. The active set is identical on
        # every rank, so every rank skips the same collectives. The factors'
        # adjoints are exact zeros held as read-only broadcasts of one zero:
        # a forward solve returns no adjoint pair.
        e2 = np.zeros_like(eps)
        X_bar = np.broadcast_to(np.zeros(()), _grid_shape(X_in, 'X_mo'))
        D_bar = np.broadcast_to(np.zeros(()), _grid_shape(D_in, 'D'))
    if route_out is not None:
        # The pole strengths, for callers that must weight a term the Newton
        # condition renormalizes. Anything added to the right of
        # w = eps_p + Delta_p + Sigma_c(w) reaches dw multiplied by Z, so a
        # caller carrying its own Delta (the Sigma_x - v_xc correction on a
        # Kohn-Sham reference) needs these and cannot reconstruct them.
        route_out['z'] = z_of
        route_out['routes'] = routes
    out = w_stars, eps_bar + e2, X_bar, D_bar
    if nranks > 1:
        # grid tiles are each rank's own rows: what ranks share is the rest
        agreement(out if rows_block is None else out[:2], comm,
                  audit_only=True, label='qp_set_gradient outputs')
    return out
