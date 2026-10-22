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
    root of w = eps_p + Sigma^CD(w) from wc (the one bracket-and-refine
    finder), Z from the closed-form slope
    reverse: the same blocks again, W Bp rebuilt per frequency, each block's
             chi0_bar folded straight into projbar(tau); the residues' rank-one
             adjoints folded into the SAME projbar; then one
             `polarizability_backward` and the Bp chain back to (X, D).

Peak memory is this rank's rows of proj(tau) and projbar(tau) (`ProjRows`),
one frequency's chi0 on the rank that factorizes it, one tau slice on the
rank that sweeps it, and the M-row tiles, all governed by one `tile_gb`: at
naux ~ 5000 a few GB, against the 90 GB of the full three-index tensor, and
independent of how many frequencies the CD quadrature has. Rebuilding W Bp in
the reverse pass costs one more factorization per frequency than holding it
would -- naux^3 each, against the sweep's M^2 (norb + naux) per tau -- and
removes the (nfreq, naux, norb) array.

At the chlorophyllide hexamer (M 117762, naux 28236, ntau 24, four states)
the reverse call's peak falls from 601 GB a rank with every block whole to
181 GB by rows at 8 ranks; the explicit residue route is refused there, C_ov
alone being 2197 GB.

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
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.Base.constants import (EXPLICIT_RESIDUE_MAX_GB, ISDF_TILE_GB,
                                LAPLACE_SCREENING_TOL, QP_CD_NEWTON_TOL,
                                SOP_N_POLES)
from src.Base.utils.mpi_grid import (agreement, allgather_rows,
                                     contiguous_block, current_comm, lockstep,
                                     mpi_map, ordered_chain_sum, ordered_sum,
                                     partition, reduce_sum)
from src.SingleReference.GW.contour_deformation import (
    cd_integral_weights, cd_pair_contraction_multi,
    cd_screening_contraction_multi, qp_energy_cd, residue_route_auto,
    screening_solver)
from src.SingleReference.GW.qp_states import scissor_route
from src.SingleReference.GW.sum_over_poles import (compressible,
                                                   pole_amplitudes,
                                                   qp_energy_sop, sop_from_wc)
from src.gradients.contour_deformation_adjoint import (
    ExplicitRealScreeningAdjoint as ExplicitRealScreening,
    LaplaceRealScreeningAdjoint as LaplaceRealScreening,
    integral_term_backward, pair_term_backward, residue_terms_backward)
from src.gradients.sum_over_poles_adjoint import sigma_sop_backward
from src.Solvers.qp_equation import qp_start, quasiparticle_weight
from src.Base.sliced_factors import GridTileRows, SlicedFactors, whole_factor
from src.Base.utils.timing import stage
from src.SingleReference.LinearResponse.space_time import (
    ProjRows, polarizability_projected_rows, three_index_ov, three_index_slice)
from src.gradients.space_time_adjoint import (
    polarizability_backward, polarizability_backward_rows,
    three_index_ov_backward, three_index_slice_backward,
    three_index_slice_backward_rows)

# The routes whose Sigma is analytic in omega: the pole model and the frozen
# scissor integrate nothing on the contour-deformation quadrature.
ANALYTIC_ROUTES = frozenset({'sop', 'scissor'})
#: The route a pole-model state Eq. (27) does not admit takes, resolved by
#: `fallback_route` from the problem's dimensions alone.
FALLBACK_ROUTE = 'fallback'
#: The residue routes the kernels take: a residue backend, the pole model,
#: 'none' (refuse a residue) or 'auto' (a backend decided per geometry).
KERNEL_RESIDUE_ROUTES = frozenset({'laplace', 'explicit', 'sop', 'none',
                                   'auto'})



def frozen_root(w0, p):
    """The root the reference geometry selected for orbital p -- w0 a scalar,
    or a mapping orbital -> energy holding the reference's roots -- or None
    before it is frozen. Where p is judged before its root is known
    (`frozen_start`), and the root followed by energy where no accepted
    geometry has named a branch (`followed_branch`): re-deciding by weight
    at every geometry jumps branches where a satellite crosses the
    quasiparticle, and f(w) = w - eps_p - Sigma(w) has a zero of vanishing
    pole strength beside every pole of Sigma."""
    if isinstance(w0, Mapping):
        got = w0.get(int(p))
        return None if got is None else float(got)
    return None if w0 is None else float(w0)


def frozen_start(w0, p, eps, nocc):
    """Where orbital p is judged before its root is known: the frozen root,
    else `qp_start` (eps_p pushed toward the chemical potential)."""
    root = frozen_root(w0, p)
    return qp_start(eps, p, nocc) if root is None else root


def rescans_beside(z0, p):
    """Whether a followed root of p below QP_WINDOW_Z_MIN has its window
    rescanned for the largest weight beside it (the branch warning; the root
    kept never depends on it): unless the reference's root of p, of pole
    strength z0 (a mapping orbital -> Z frozen with w0), was no quasiparticle
    either. A state with no quasiparticle at the reference has no
    quasiparticle branch to lose; its satellites trade the largest weight at
    every geometry, and the rescan of its whole window was most of the cost of
    a displaced contour-deformation solve."""
    z = z0.get(int(p)) if isinstance(z0, Mapping) else None
    return z is None or quasiparticle_weight(z)


def followed_branch(follow, w0, p):
    """(w, labels): the root orbital p follows at this geometry and the
    labels of the poles that bracketed it (None where the poles are
    unknown), or None before a root is frozen. follow: a mapping orbital ->
    (w, labels), the branch the last accepted geometry kept; an orbital it
    does not name follows its frozen reference root w0 (`frozen_root`) by
    energy. Whatever the weight, nothing is refused: the solve says where
    the largest weight has left the branch (`find_qp_root`)."""
    got = follow.get(int(p)) if isinstance(follow, Mapping) else None
    if got is not None:
        w, labels = got
        return float(w), (None if labels is None else tuple(labels))
    root = frozen_root(w0, p)
    return None if root is None else (root, None)


def static_term(xc_correction, si):
    """The static term <p|Sigma_x - v_xc|p> + Sigma^env_pp the si-th state of
    a set is solved with: one entry per state, or one scalar for all."""
    return (float(np.asarray(xc_correction).ravel()[si])
            if np.ndim(xc_correction) else float(xc_correction))


def require_static_terms(xc_correction, nstates):
    """Refuse per-state static terms that are not one per state of the set:
    `static_term` reads them by position, so a list left over from another
    set would put one orbital's correction on another."""
    if np.ndim(xc_correction) and np.size(xc_correction) != nstates:
        raise ValueError(f'{np.size(xc_correction)} static terms for a set '
                         f'of {nstates} states')


def frozen_on_pole_model(p, w0, sop_poles):
    """Whether the reference solve put orbital p on the pole model: True with
    its poles frozen, False with its root frozen and no poles, None before.

    Eq. (27) read at each geometry's start point is not one decision: the
    start is eps_p before the reference solve and the frozen root after it,
    and a state whose correction carries it across the wall (formaldehyde
    PBE0 orbitals 4, 5 and 10: reach 0.96 / 0.76 / 0.76 at eps_p, 1.35 /
    1.16 / 1.15 at the root) leaves the pole model for the quadrature at
    every displaced geometry, a step in the surface at the first one.
    """
    if isinstance(sop_poles, Mapping) and sop_poles.get(int(p)) is not None:
        return True
    if isinstance(w0, Mapping) and w0.get(int(p)) is not None:
        return False
    return None


def pole_model_route(residue_route, scissor, p, eps, nocc, w0,
                     sop_poles=None):
    """(route, shift, start) of orbital p before 'auto' is resolved.

    'sop' asks for the pole model; a state with a frozen scissor takes the
    shift instead, and one Eq. (27) does not admit at its start point falls
    back to FALLBACK_ROUTE (`fallback_route`: one residue backend for every
    geometry, where 'auto' would decide it per geometry and step the
    surface). Once the reference solve has frozen the state's root
    (w0) and its poles (sop_poles), that verdict is frozen with them
    (`frozen_on_pole_model`), as the scissor tiers are. Every other
    residue_route is taken as asked. Read off eps and the frozen start alone,
    never the grid, so a caller can tell before solving whether a set
    integrates anything on the quadrature.
    """
    start = frozen_start(w0, p, eps, nocc)
    route, shift = residue_route, None
    if route == 'sop':
        route, shift = scissor_route(scissor, p, eps, nocc, start)
        frozen = frozen_on_pole_model(p, w0, sop_poles)
        admitted = (compressible(start, eps, nocc)[0] if frozen is None
                    else frozen)
        if route == 'sop' and not admitted:
            route = FALLBACK_ROUTE
    return route, shift, start


def fallback_route(naux, nocc, nvir):
    """The residue backend of a pole-model state Eq. (27) does not admit:
    'explicit' where its blocks fit (`explicit_fits`), 'laplace' where they do
    not. Decided by the dimensions alone, so the whole walk takes one."""
    return 'explicit' if explicit_fits(naux, nocc, nvir) else 'laplace'


def explicit_fits(naux, nocc, nvir):
    """Whether the explicit residue backend's C_ov blocks fit in
    EXPLICIT_RESIDUE_MAX_GB."""
    return naux * nocc * nvir * 8 / 1e9 <= EXPLICIT_RESIDUE_MAX_GB


def carried_analytically(states, eps, nocc, residue_route, scissor, w0,
                         sop_poles=None):
    """Whether every state of the set takes the pole model or a frozen
    scissor (`ANALYTIC_ROUTES`), so that nothing of the set is integrated on
    the contour-deformation quadrature and the grid only feeds the pole fits
    (`ExcitedStateChain._fix_cd_grid` sizes it for that)."""
    return all(pole_model_route(residue_route, scissor, int(p), eps, nocc, w0,
                                sop_poles)[0] in ANALYTIC_ROUTES
               for p in np.atleast_1d(states))


def _stride_kw(sop_stride):
    """`stride=` for `sop_from_wc`, or nothing when the default is wanted.

    The poles are COMMON to all orbitals and placed from every stride-th
    column. Placement is a stronger lever on dSigma/domega than on Sigma, so
    this is a knob a gradient study needs and an energy one does not.
    """
    return {} if sop_stride is None else {'stride': int(sop_stride)}


def sop_model(wc, nu_points, eps, nocc, p, sop_poles, n_poles, sop_stride):
    """(poles, amplitudes) of orbital p's pole model, on the frozen poles if any.

    The pole positions are a realization of Sigma(omega), fitted by vector
    fitting on the reference geometry's wc and then held: the adjoint
    (`sigma_sop_backward`) differentiates A = F^+ wc at FIXED poles, so a set
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
    every one; serially the list it always was. comm None is `current_comm()`.

    A slice is O(M naux norb), one per state, so a window of hundreds of
    states built on every rank is a cost no rank count divides. Each slice is
    the same `three_index_slice` call on the same whole factors wherever it
    runs and travels verbatim (`allgather_rows`): every rank holds the serial
    slices bitwise, as views into one (nstates, naux, norb) array.
    """
    comm = current_comm() if comm is None else comm
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


def pole_fits_needed(states, eps, nocc, residue_route, scissor, w0,
                     sop_poles):
    """The positions in `states` of the pole-model states with no frozen
    poles: the fits pass 1 makes, in state order."""
    if residue_route != 'sop':
        return []
    return [si for si, p in enumerate(states)
            if (sop_poles is None or sop_poles.get(int(p)) is None)
            and pole_model_route(residue_route, scissor, int(p), eps, nocc,
                                 w0, sop_poles)[0] == 'sop']


def fit_poles_over_ranks(states, wcs, nu_points, eps, nocc, residue_route,
                         scissor, w0, sop_poles, n_poles, sop_stride,
                         comm=None):
    """`sop_poles` with the fit of every pole-model state it lacks added, the
    fits striped over the ranks (`mpi_map`) and gathered onto every one;
    comm None is `current_comm()`.

    The fit is a dense least squares of about norb^3 a state, the one cost of
    pass 1 that grows with the molecule. It reads only the state's wc, which
    is reduced and identical on every rank, so the owner's poles are the
    serial poles bitwise; `sop_model` then reads them as frozen poles, and
    the amplitudes are the call `sop_from_wc` would have made. The states
    fitted are those pass 1 sends to the pole model (route, scissor tier,
    compression condition at the start point) and has no frozen poles for.
    """
    need = pole_fits_needed(states, eps, nocc, residue_route, scissor, w0,
                            sop_poles)
    if not need:
        return sop_poles

    def fit(si):
        return sop_from_wc(wcs[si], nu_points, eps, nocc, n_poles=n_poles,
                           **_stride_kw(sop_stride))[0]

    comm = current_comm() if comm is None else comm
    out = dict(sop_poles or {})
    for si, poles in zip(need, mpi_map(fit, need, comm)):
        out[int(states[si])] = poles
    return out


def integral_term_reverse(state, WtB, k):
    """(Bp_bar, chi0_bar, eps_bar) of ONE contour-deformation frequency for one state.

    Sigma^int_p(w) = -(1/pi) sum_kq c[k,q] wc[k,q] with the weights of
    `cd_integral_weights`, so at a fixed frequency the adjoint on the
    screening is the weight vector c[k, :] on wc[k, :] -- which
    `integral_term_backward` pushes back onto the state's slice and onto
    chi0(i.nu_k) -- and the explicit eps dependence of the weights gives the
    third term. On the pole-model route the omega dependence is analytic and
    `wc_bar` already holds the whole weight, so that term is zero.

    state: one entry of the reverse pass's active set -- the state's slice
    `Bp`, its Z-weighted adjoint `zw`, the weights `c` and `dc` at its root
    (`integral_state`) and `wc_bar` or None.
    """
    Bp = state['Bp']
    if state['wc_bar'] is not None:
        bb, _, cb = integral_term_backward(Bp, WtB, state['wc_bar'][k])
        return bb, cb, 0.0
    coeff = -state['zw'] / np.pi
    bb, wck, cb = integral_term_backward(Bp, WtB, coeff * state['c'][k])
    return bb, cb, -coeff * wck * state['dc'][k]


def integral_state(Bp, zw, w_star, eps, nu_points, nu_weights, wc_bar=None,
                   **extra):
    """One entry of the reverse pass's active set (`integral_term_reverse`):
    the integral term's weights at the root are computed once, for every
    frequency, because the exact singular part puts the near-pole correction
    on the two smallest nodes whichever rank owns them."""
    state = dict(Bp=Bp, zw=zw, wc_bar=wc_bar, **extra)
    if wc_bar is None:
        state['c'], state['dc'] = cd_integral_weights(w_star - eps, nu_points,
                                                      nu_weights)
    return state

def require_explicit_fits(naux, nocc, nvir, p):
    """Refuse the explicit residue backend where its blocks cannot fit.

    C_ov is (naux, nocc*nvir) and the backend holds up to three such blocks at
    once on every rank (`EXPLICIT_RESIDUE_MAX_GB`); the Laplace backend and
    the pole model serve the same residues from proj(tau) without one.
    """
    gb = naux * nocc * nvir * 8 / 1e9
    if not explicit_fits(naux, nocc, nvir):
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
                           *, residue_route, mu=None, xc_correction=0.0,
                           w0=None, tol=QP_CD_NEWTON_TOL,
                           laplace_tol=LAPLACE_SCREENING_TOL,
                           want_grad=True, tile_gb=ISDF_TILE_GB,
                           follow=None, route_out=None,
                           n_poles=SOP_N_POLES, sop_stride=None,
                           scissor=None, sop_poles=None, comm=None, z0=None):
    """(eps^QP_p, Z, eps_bar, X_bar, D_bar), or (eps^QP_p, Z) with want_grad=False.

    `qp_set_gradient` of the one state p at weight one (zero without a
    gradient): the same routes, frozen arguments, partition over the ranks
    and bits. grid, residue_route, w0, z0, follow, sop_poles and comm are
    that function's; xc_correction is p's static term.
    route_out: a dict, if given, receives 'residue_route', 'residues',
               'weights', 'roots', 'labels', 'branches' and 'sop_poles'.
    """
    p = int(p)
    solved = {}
    roots, eps_bar, X_bar, D_bar = qp_set_gradient(
        X, D, eps, nocc, grid, nu_points, nu_weights, [p],
        [1.0 if want_grad else 0.0], mu=mu, xc_correction=xc_correction,
        w0=w0, tol=tol, residue_route=residue_route, laplace_tol=laplace_tol,
        tile_gb=tile_gb, follow=follow, route_out=solved,
        n_poles=n_poles, sop_stride=sop_stride, scissor=scissor,
        sop_poles=sop_poles, comm=comm, keep_tape=False, z0=z0)
    w_star, z = float(roots[0]), float(solved['z'][0])
    if route_out is not None:
        route_out['residue_route'] = solved['routes'][p]
        route_out['residues'] = solved['residues'][p]
        route_out['weights'] = solved['weights']
        route_out['roots'] = solved['roots']
        route_out['sop_poles'] = solved['sop_poles']
    if not want_grad:
        return w_star, z
    # a state with no adjoint on the factors (a frozen scissor) comes back as
    # read-only zeros; a caller adds into them
    if not X_bar.flags.writeable:
        X_bar, D_bar = np.zeros(X_bar.shape), np.zeros(D_bar.shape)
    return w_star, z, eps_bar, X_bar, D_bar


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
    applies to nothing. C_ov, the explicit residues' three-index block, is
    kept only while the solve that built it repeats (`drop_residues`).
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
    C_ov: object = None

    def release(self):
        """Drop what the tape holds."""
        self.X = self.D = self.proj_tau = self.Bps = self.wcs = None
        self.C_ov = None

    def drop_residues(self):
        """Drop C_ov, once the solve that built it no longer repeats."""
        self.C_ov = None

    def reads(self, X, D, eps, mu, tile_gb, states, grid):
        """(proj and Bps apply, wcs apply as well) for a call on these."""
        same = (self.proj_tau is not None and self.X is X and self.D is D
                and np.array_equal(self.eps, eps) and self.mu == mu
                and self.tile_gb == tile_gb
                and np.array_equal(self.states, states)
                and np.array_equal(self.tau_points, grid.tau_points))
        return same, same and np.array_equal(self.cosft_wt, grid.cosft_wt)


@dataclass(eq=False)
class QPSetPass:
    """One `qp_set_gradient` call: its arguments, locked to rank 0's on
    construction, and what each pass hands the next. The passes run in the
    order the methods are listed, every statement and collective in the order
    of the one-function form."""

    X: object
    D: object
    eps: object
    nocc: int
    grid: object
    nu_points: object
    nu_weights: object
    states: object
    weights: object
    residue_route: str
    mu: object
    xc_correction: object
    w0: object
    tol: float
    laplace_tol: float
    tile_gb: float
    follow: object
    route_out: object
    n_poles: int
    sop_stride: object
    scissor: object
    sop_poles: object
    comm: object
    rows_block: object
    tape: object
    timer: object
    pairs: object
    keep_tape: bool
    z0: object = None

    def __post_init__(self):
        (X, D, eps, nocc, grid, nu_points, nu_weights, states, weights,
         residue_route, mu, xc_correction, w0, tol, laplace_tol, tile_gb,
         follow, route_out, n_poles, sop_stride, scissor, sop_poles,
         comm, rows_block, tape, timer, pairs, keep_tape) = (
            self.X, self.D, self.eps, self.nocc, self.grid, self.nu_points,
            self.nu_weights, self.states, self.weights, self.residue_route,
            self.mu, self.xc_correction, self.w0, self.tol, self.laplace_tol,
            self.tile_gb, self.follow, self.route_out, self.n_poles,
            self.sop_stride, self.scissor, self.sop_poles, self.comm,
            self.rows_block, self.tape, self.timer, self.pairs,
            self.keep_tape)
        eps = np.asarray(eps, float)
        self.X_in, self.D_in = X, D
        self.paired = {s for i, j, _ in pairs for s in (int(i), int(j))}
        states = np.atleast_1d(states)
        weights = np.atleast_1d(weights)
        require_static_terms(xc_correction, len(states))
        nu_points = np.asarray(nu_points)
        if len(nu_points) != grid.cosft_wt.shape[0]:
            raise ValueError("the grid's frequency axis must be the CD "
                             "quadrature")
        comm = current_comm() if comm is None else comm
        rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                        else (0, 1))
        if nranks > 1:
            # Before the routes are chosen: they are read off eps, and with
            # the weights' zeros they set the active set whose adjoints pass 2
            # reduces as one buffer.
            eps, weights = lockstep((eps, weights), comm)
            agreement((_audited(X), _audited(D), xc_correction, w0,
                       follow, scissor, sop_poles, self.z0),
                      comm, audit_only=True, label='qp_set_gradient inputs')
        self.tau_mine = (partition(grid.ntau, rank, nranks) if nranks > 1
                         else None)
        self.nu_mine = (partition(len(nu_points), rank, nranks) if nranks > 1
                        else None)
        self.whole = {}
        (self.eps, self.nocc, self.grid, self.nu_points, self.nu_weights,
         self.states, self.weights, self.residue_route, self.mu,
         self.xc_correction, self.w0, self.tol, self.laplace_tol,
         self.tile_gb, self.follow, self.route_out, self.n_poles,
         self.sop_stride, self.scissor, self.sop_poles, self.comm,
         self.rows_block, self.tape, self.timer, self.pairs,
         self.keep_tape) = (eps, nocc, grid, nu_points, nu_weights, states,
                            weights, residue_route, mu, xc_correction, w0,
                            tol, laplace_tol, tile_gb, follow, route_out,
                            n_poles, sop_stride, scissor, sop_poles, comm,
                            rows_block, tape, timer, pairs, keep_tape)
        self.rank, self.nranks = rank, nranks

    def factors(self):
        """X_mo and D whole, gathered at the first call that reads them."""
        if not self.whole:
            self.whole['X'] = whole_factor(self.X_in, 'X_mo')
            self.whole['D'] = whole_factor(self.D_in, 'D')
        return self.whole['X'], self.whole['D']

    def forward_contraction(self):
        """proj(tau), the slices B_p and their screening contractions wc_p
        over the CD frequencies: read from the tape where it applies."""
        X_in, D_in, eps, mu, tile_gb, states, grid = (
            self.X_in, self.D_in, self.eps, self.mu, self.tile_gb,
            self.states, self.grid)
        tape, timer, comm, route_out = (self.tape, self.timer, self.comm,
                                        self.route_out)
        same, same_wc = ((False, False) if tape is None
                         else tape.reads(X_in, D_in, eps, mu, tile_gb, states,
                                         grid))
        if same:
            proj_tau, Bps = tape.proj_tau, tape.Bps
        else:
            with stage(timer, 'qp_set.tau_sweep'):
                X, D = self.factors()
                proj_tau = polarizability_projected_rows(
                    X, D, eps, self.nocc, grid.tau_points, mu=mu,
                    tile_memory_gb=tile_gb, comm=comm)
                Bps = slices_over_ranks(X, D, states, tile_gb, comm)
        self.naux = proj_tau.shape[-1]
        self.eye = np.eye(self.naux)
        self.eps_bar = np.zeros_like(eps)
        self.w_stars = np.zeros(len(states))
        self.routes, self.weights_of, self.roots, self.fitted = {}, {}, {}, {}
        self.labels, self.branches = {}, {}
        self.C_ov = None
        self.residues = {}
        self.z_of = np.zeros(len(states))

        if same_wc:
            wcs = tape.wcs
        else:
            with stage(timer, 'qp_set.frequency_pass'):
                wcs = cd_screening_contraction_multi(proj_tau, grid.cosft_wt,
                                                     Bps, tile_gb=tile_gb,
                                                     freq_indices=self.nu_mine)
                if self.nranks > 1:
                    wcs = list(reduce_sum(np.stack(wcs), comm))
        if route_out is not None and self.keep_tape:
            route_out['tape'] = QPSetTape(X_in, D_in, eps.copy(), mu, tile_gb,
                                          states.copy(),
                                          np.array(grid.tau_points),
                                          np.array(grid.cosft_wt), proj_tau,
                                          Bps, wcs)
        self.same, self.proj_tau, self.Bps, self.wcs = same, proj_tau, Bps, wcs

    def solve_states(self):
        """PASS 1 -- every state's route, root and residue set, and the
        states that carry an adjoint (`active`).

        Replicated: a root search on wc and, for a residue, a rank-one push into
        its backend. The pole fits are the exception, striped over the ranks
        a round at a time, one fit a rank, as the loop reaches them: the same
        owner and the same bits as one striped call, and a pass stopped at an
        unresolved root leaves the later rounds unfitted.
        """
        states, weights, eps, nocc, grid = (self.states, self.weights,
                                            self.eps, self.nocc, self.grid)
        Bps, wcs, nu_points, nu_weights = (self.Bps, self.wcs, self.nu_points,
                                           self.nu_weights)
        residue_route, scissor, w0, follow_of = (
            self.residue_route, self.scissor, self.w0, self.follow)
        sop_poles, nranks, naux, proj_tau = (self.sop_poles, self.nranks,
                                             self.naux, self.proj_tau)
        tol, laplace_tol, tile_gb = self.tol, self.laplace_tol, self.tile_gb
        route_out, tape, same = self.route_out, self.tape, self.same
        eps_bar, fitted, C_ov = self.eps_bar, self.fitted, self.C_ov
        self.lap = time.perf_counter()
        need = pole_fits_needed(states, eps, nocc, residue_route, scissor, w0,
                                sop_poles)
        fit_round = {si: need[i - i % nranks:i - i % nranks + nranks]
                     for i, si in enumerate(need)}
        active = []
        for si, (p, wgt) in enumerate(zip(states, weights)):
            p = int(p)
            Bp, wc = Bps[si], wcs[si]
            if si in fit_round and (sop_poles is None or p not in sop_poles):
                rnd = fit_round[si]
                sop_poles = fit_poles_over_ranks(
                    states[rnd], [wcs[i] for i in rnd], nu_points, eps, nocc,
                    residue_route, scissor, w0, sop_poles,
                    self.n_poles, self.sop_stride, self.comm)
            # PER STATE. <p|Sigma_x - v_xc|p> is a different number for every
            # orbital, so a scalar shared across the set would put the HOMO's
            # correction on the LUMO. A scalar is still accepted, and is the
            # right thing on Hartree-Fock, where every entry is zero.
            xc_p = static_term(self.xc_correction, si)
            # after the reference, every geometry follows the branch the last
            # accepted one kept, whatever its weight
            follow, branch = followed_branch(follow_of, w0, p), {}
            # A state Eq. (27) does not admit cannot be carried by the pole
            # model at any M, so it takes the residue route instead and the
            # record says so. On a valence quasiparticle set some states
            # always will.
            route, shift, start = pole_model_route(residue_route, scissor, p,
                                                   eps, nocc, w0, sop_poles)
            if route == FALLBACK_ROUTE:
                route = fallback_route(naux, nocc, len(eps) - nocc)
            if route == 'auto':
                route = residue_route_auto(grid, eps, nocc, start, laplace_tol)
            if route == 'laplace':
                rs = LaplaceRealScreening(proj_tau, grid, eps, nocc,
                                          tol=laplace_tol)
            elif route == 'explicit':
                if C_ov is None:
                    require_explicit_fits(naux, nocc, len(eps) - nocc, p)
                    C_ov = (tape.C_ov if same and tape.C_ov is not None else
                            three_index_ov(*self.factors(), eps, nocc,
                                           tile_gb=tile_gb))
                    if route_out is not None and self.keep_tape:
                        route_out['tape'].C_ov = C_ov
                rs = ExplicitRealScreening(C_ov, eps, nocc)
            elif route in ('sop', 'scissor', 'none'):
                rs = None
            else:
                raise ValueError(f'residue_route {route!r}')
            poles = amp = None
            if route == 'scissor':
                w_star, z, res = float(eps[p] + xc_p + shift), 1.0, []
            elif route == 'sop':
                poles, amp = sop_model(wc, nu_points, eps, nocc, p, sop_poles,
                                       self.n_poles, self.sop_stride)
                fitted[p] = poles
                w_star, z = qp_energy_sop(p, amp, poles, eps, nocc,
                                          xc_correction=xc_p, tol=tol,
                                          follow=follow, branch=branch)
                res = []
            else:
                w_star, z, res = qp_energy_cd(
                    p, Bp, eps, nocc, nu_points, nu_weights,
                    xc_correction=xc_p, tol=tol, follow=follow, wc=wc,
                    real_screening=rs, branch=branch,
                    rescan=rescans_beside(self.z0, p))
            self.w_stars[si] = w_star
            self.z_of[si] = z
            self.routes[p] = route
            self.weights_of[p] = float(z)
            self.roots[p] = float(w_star)
            self.labels[p] = branch.get('labels')
            if follow is not None:
                self.branches[p] = branch
            self.residues[p] = res
            if wgt == 0.0 and si not in self.paired:
                continue

            zw = z * wgt
            if route == 'scissor':
                if si in self.paired:
                    raise ValueError(f'orbital {p} carries a frozen scissor '
                                     f'and has no screening contraction to '
                                     f'pair')
                eps_bar[p] += zw
                continue
            wc_bar = None
            if route == 'sop':
                # The pole model's whole omega dependence is analytic, so its
                # adjoint on the imaginary-axis data is one transpose and
                # enters the screening exactly where the integral term's own
                # weights do.
                eps_sop, wc_bar, _ = sigma_sop_backward(w_star, amp, poles,
                                                        eps, nocc, nu_points,
                                                        sigma_bar=zw)
                eps_bar += eps_sop
            active.append(integral_state(Bp, zw, w_star, eps, nu_points,
                                         nu_weights, wc_bar, si=si, p=p,
                                         res=res, rs=rs, route=route))
        self.active, self.sop_poles, self.C_ov = active, sop_poles, C_ov

    def reverse_frequencies(self):
        """PASS 2 -- the integral term's reverse pass, frequency outside,
        over the frequencies this rank owns. Each frequency's addends reach
        every rank and are added in frequency order (`ordered_sum`): the
        one-rank bits."""
        active, eps, comm, rank = self.active, self.eps, self.comm, self.rank
        proj_tau, grid, naux, eye = (self.proj_tau, self.grid, self.naux,
                                     self.eye)
        pairs, Bps, paired = self.pairs, self.Bps, self.paired
        self.lap = _lap(self.timer, 'qp_set.routes', self.lap)
        Bp_bars = {a['si']: np.zeros_like(a['Bp']) for a in active}
        eps_bar_part = np.zeros_like(eps)
        proj_bar = None

        def chain_frequency(owner, terms):
            """Every state's B_p_bar += its term at one frequency, added by
            the frequency's owner and handed to every rank
            (`ordered_chain_sum`)."""
            for a in active:
                si = a['si']
                Bp_bars[si] = ordered_chain_sum(
                    {0: terms[si]} if owner == rank else {}, [owner], comm,
                    Bp_bars[si])

        if active:
            eps_terms = []
            proj_bar = proj_tau.zeros_like()
            for fb in proj_tau.blocks(grid.cosft_wt, self.tile_gb,
                                      self.nu_mine):
                owned = {}
                for m, k in enumerate(fb.ks):
                    solve = screening_solver(eye - fb.chi0[m])
                    chi0_bar = np.zeros((naux, naux))
                    terms, wtbs = {}, {}
                    for j, a in enumerate(active):
                        WtB = solve(a['Bp'])
                        bb, cb, e_k = integral_term_reverse(a, WtB, k)
                        terms[a['si']] = bb
                        chi0_bar += cb
                        eps_terms.append(((k, j), e_k))
                        if a['si'] in paired:
                            wtbs[a['si']] = WtB
                    # the off-diagonal contractions' adjoints, on the same
                    # solves
                    for i, j, wc_bar in pairs:
                        bi, bj, cb = pair_term_backward(
                            Bps[i], Bps[j], wtbs[i], wtbs[j], wc_bar[k])
                        terms[i] = terms[i] + bi
                        terms[j] = terms[j] + bj
                        chi0_bar += cb
                    fb.chi0[m] = chi0_bar
                    if fb.owners is None:
                        chain_frequency(rank, terms)
                    else:
                        owned[k] = terms
                    terms = wtbs = None
                # B_p_bar is (naux, nmo) a frequency, 0.34 GB at the
                # chlorophyllide dimer/cc-pVTZ: a round's terms join the
                # running sum in frequency order as soon as the round is
                # solved, so a rank holds one frequency's, whatever the number
                # of frequencies
                if fb.owners is not None:
                    for k, owner in zip(fb.block, fb.owners):
                        chain_frequency(int(owner), owned.get(k))
                owned = None
                proj_bar.fold(grid.cosft_wt, fb)
            fb = solve = chi0_bar = None
            eps_bar_part = ordered_sum(eps_terms, comm, onto=eps_bar_part)
            eps_terms = None
        self.eps_bar += eps_bar_part
        self.lap = _lap(self.timer, 'qp_set.frequency_reverse', self.lap)
        self.Bp_bars, self.proj_bar = Bp_bars, proj_bar

    def direct_terms(self):
        """PASS 3 -- per state: the direct term, the residues, the slice
        adjoint.

        Replicated, like pass 1. Each backend leaves with its state, so the
        explicit route holds one Cov_bar at a time; the Laplace backends keep
        only their recorded pushes, folded into projbar after the pass.
        """
        active, eps, nocc, comm = self.active, self.eps, self.nocc, self.comm
        rows_block, tile_gb, eps_bar = (self.rows_block, self.tile_gb,
                                        self.eps_bar)
        roots, route_out, Bp_bars = self.roots, self.route_out, self.Bp_bars
        rs = self.C_ov = None
        pushed = []
        # The factors' adjoints exist only where a state carries one: one
        # pair, every state's slice and residue adjoints added into it in
        # turn.
        X_bar = D_bar = None
        X = D = None
        if active:
            X, D = self.factors()
            if rows_block is None:
                X_bar, D_bar = np.zeros_like(X), np.zeros_like(D)
            else:
                X_bar = GridTileRows.zeros(X.shape[0], X.shape[1],
                                           rows_block, comm)
                D_bar = GridTileRows.zeros(D.shape[0], self.naux, rows_block,
                                           comm)
        for a in active:
            p, zw, Bp, res, route = (a['p'], a['zw'], a['Bp'], a['res'],
                                     a['route'])
            rs = a.pop('rs')
            Bp_bar = Bp_bars[a['si']]
            eps_bar[p] += zw
            if res:
                e_r, b_r, _ = residue_terms_backward(p, roots[p], Bp, eps,
                                                     nocc, res, zw, rs)
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
                three_index_slice_backward_rows(X, D, p, Bp_bar,
                                                (X_bar, D_bar))
        if active:
            fold_residue_adjoints(self.proj_bar, pushed)
        pushed = None
        del self.proj_tau
        if route_out is not None:
            route_out['routes'] = self.routes
            # The roots, their pole strengths and the labels of the poles
            # bracketing them, which the same caller hands back as `w0` and
            # `follow`: a displaced geometry follows each root's branch
            # (`followed_branch`). 'branches': per followed orbital, whether
            # its branch continued and the largest weight here
            # (`Solvers.qp_equation._note_branch`).
            route_out['weights'] = self.weights_of
            route_out['roots'] = roots
            route_out['labels'] = self.labels
            route_out['branches'] = self.branches
            # The pole positions each pole-model state used, handed back as
            # `sop_poles` so that the displaced geometries share the reference
            # fit.
            route_out['sop_poles'] = self.fitted
            route_out['residues'] = self.residues
        self.lap = _lap(self.timer, 'qp_set.direct_terms', self.lap)
        self.X, self.D, self.X_bar, self.D_bar = X, D, X_bar, D_bar

    def reverse_sweep(self):
        """The M^2 reverse sweep of projbar onto (X, D), and the call's
        (eps^QP values, eps_bar, X_bar, D_bar)."""
        active, rows_block, comm = self.active, self.rows_block, self.comm
        X, D, X_bar, D_bar = self.X, self.D, self.X_bar, self.D_bar
        eps, nocc, grid, mu = self.eps, self.nocc, self.grid, self.mu
        if active and rows_block is not None:
            # the sweep reads the gathered X_mo through column views of its
            # branches, so no copy of either stands beside it
            e2, Xs, Ds = polarizability_backward_rows(self.proj_bar, X, D, eps,
                                                      nocc, grid, mu=mu,
                                                      block=rows_block,
                                                      comm=comm)
            del self.proj_bar
            X_bar += Xs
            D_bar += Ds
            del Xs, Ds
        elif active:
            e2, Xs, Ds = polarizability_backward(self.proj_bar, X, D, eps,
                                                 nocc, grid, mu=mu,
                                                 tile_gb=self.tile_gb,
                                                 tau_indices=self.tau_mine)
            del self.proj_bar
            if self.nranks > 1:
                for arr in (e2, Xs, Ds):
                    reduce_sum(arr, comm)
            X_bar += Xs
            D_bar += Ds
            del Xs, Ds
        else:
            # Nothing carries a weight: the sweep would run on a zero projbar
            # and return exact zeros, so it is not run. The active set is
            # identical on every rank, so every rank skips the same
            # collectives. The factors' adjoints are exact zeros held as
            # read-only broadcasts of one zero: a forward solve returns no
            # adjoint pair.
            e2 = np.zeros_like(eps)
            X_bar = np.broadcast_to(np.zeros(()), _grid_shape(self.X_in,
                                                              'X_mo'))
            D_bar = np.broadcast_to(np.zeros(()), _grid_shape(self.D_in, 'D'))
        if active:
            _lap(self.timer, 'qp_set.reverse_sweep', self.lap)
        if self.route_out is not None:
            # The pole strengths, for callers that must weight a term the
            # Newton condition renormalizes. Anything added to the RIGHT of
            # w = eps_p + Delta_p + Sigma_c(w) reaches dw multiplied by Z, so
            # a caller carrying its own Delta (the Sigma_x - v_xc correction
            # on a Kohn-Sham reference) needs these and cannot reconstruct
            # them.
            self.route_out['z'] = self.z_of
            self.route_out['routes'] = self.routes
        out = self.w_stars, self.eps_bar + e2, X_bar, D_bar
        if self.nranks > 1:
            # grid tiles are each rank's own rows: what ranks share is the rest
            agreement(out if rows_block is None else out[:2], comm,
                      audit_only=True, label='qp_set_gradient outputs')
        return out


def _lap(timer, key, t0):
    """Add the seconds since t0 to timer[key]; the clock now, the next lap's
    start."""
    now = time.perf_counter()
    if timer is not None:
        timer[key] = timer.get(key, 0.0) + now - t0
    return now


def _audited(factor):
    """What an audited run digests of a factor: the array, or the grid
    points of `SlicedFactors` (their rows are this rank's alone)."""
    return factor.coords if isinstance(factor, SlicedFactors) else factor


def _grid_shape(factor, name):
    """The whole shape of a factor, (M, ncol), without gathering it."""
    if isinstance(factor, SlicedFactors):
        return (factor.npts, getattr(factor, name).shape[1])
    return np.shape(factor)


def pair_contractions(tape, grid, pairs, tile_gb=None, comm=None):
    """The off-diagonal contractions wc_ij (`cd_pair_contraction_multi`) of
    `pairs`, positions into the states of a `QPSetTape`, on its proj(tau) and
    slices; split over the contour-deformation frequencies as the diagonal
    ones are and reduced, so every rank holds the serial ones bitwise."""
    comm = current_comm() if comm is None else comm
    nranks = 1 if comm is None else comm.Get_size()
    mine = (partition(len(grid.cosft_wt), comm.Get_rank(), nranks)
            if nranks > 1 else None)
    out = cd_pair_contraction_multi(
        tape.proj_tau, grid.cosft_wt, tape.Bps, pairs,
        tile_gb=ISDF_TILE_GB if tile_gb is None else tile_gb,
        freq_indices=mine)
    if nranks > 1:
        out = list(reduce_sum(np.stack(out), comm))
    return out


def qp_set_gradient(X, D, eps, nocc, grid, nu_points, nu_weights, states,
                    weights, *, residue_route, mu=None, xc_correction=0.0,
                    w0=None, tol=QP_CD_NEWTON_TOL,
                    laplace_tol=LAPLACE_SCREENING_TOL,
                    tile_gb=ISDF_TILE_GB, follow=None, route_out=None,
                    n_poles=SOP_N_POLES, sop_stride=None,
                    scissor=None, sop_poles=None, comm=None,
                    rows_block=None, tape=None, timer=None, pairs=(),
                    keep_tape=True, z0=None):
    """(eps^QP values, eps_bar, X_bar, D_bar) for sum_p weights[p] eps^QP_p.

    A BSE@GW excitation energy carries a weight on EVERY quasiparticle on the
    BSE diagonal, so the naive chain runs the whole reverse pass once per
    orbital. Almost all of it is shared: proj(tau) is built once, and every
    state's adjoint on chi0 folds into ONE projbar, so the expensive sweep --
    `polarizability_backward`, the only (M x M) work in the route -- runs once
    for the entire set instead of once per state. What remains per state is a
    three-index slice, a screening contraction and a root search, all
    (naux, norb) at worst.

    This is the same weight-folding the dense pilot does in `bse_qb._qp_chain`,
    where hoisting the maps out of the orbital loop was worth 29.8x.

    Only states with a non-zero weight are solved at all, so a quasiparticle
    set smaller than the orbital basis costs proportionally less. Where no
    state carries an adjoint on the factors -- the forward solve, every weight
    zero -- X_bar and D_bar are read-only broadcasts of one zero, no pair of
    the factors' shape formed.

    THE REVERSE PASS RUNS FREQUENCY-OUTSIDE, STATES INSIDE, exactly as the
    forward contraction does: [1 - chi0(i.nu)] is factorized ONCE per
    frequency and every weighted state takes a triangular solve from it, and
    the states' adjoints on chi0(i.nu) are summed before the block is folded
    into projbar. A state loop outside would refactorize the same matrix once
    per state -- nstates naux^3 for nothing, the largest cost of the gradient
    at large naux. So the work is three passes: every
    state's root and route (replicated, cheap), the integral term's reverse
    over frequencies, then each state's direct term, residues and slice
    adjoint.

    residue_route: one of KERNEL_RESIDUE_ROUTES, required: 'auto' is not a
                 frozen mode, and anything else is refused before any work.

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
    the route, the followed root, the root search itself -- is decided from eps and
    from the ALREADY REDUCED wc, never from a rank's own partial, so the
    decisions cannot diverge once the inputs agree. They are what fixes the
    ACTIVE SET, and the active set fixes how many buffers pass 2 reduces:
    ranks reading different eps would call that collective with different
    shapes. Every transform over tau is one auxiliary row per call, a shape
    no rank count changes, so every wc, every root and projbar itself are the
    serial ones bitwise at every rank count; the adjoints are partial sums
    over the partitions, exact up to summation order.

    w0:          the reference's roots, one value or a mapping orbital ->
                 energy frozen by the caller: where each state is judged
                 before its root is known (`frozen_start`), and the root
                 followed where `follow` names none.
    z0:          the pole strengths of those roots, a mapping orbital -> Z
                 frozen with w0: a state whose reference root was no
                 quasiparticle has no window rescan beside its followed root
                 (`rescans_beside`); None rescans every one.
    follow:      a mapping orbital -> (w, labels), the root the last accepted
                 geometry kept and the labels of the poles bracketing it:
                 the branch each state follows here, whatever its weight,
                 instead of re-deciding the largest Z (`followed_branch`).
    sop_poles:   the pole model's positions, a mapping orbital -> poles frozen
                 by the caller (`sop_model`); None fits every state's here.
    route_out:   a dict, if given, receives 'routes', 'z', 'weights',
                 'roots', 'labels', 'branches', 'residues', 'sop_poles' and,
                 with keep_tape, 'tape' (`QPSetTape`).
    keep_tape:   False hands back no tape, so proj(tau) is dropped before
                 the reverse sweep instead of standing beside projbar.
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
    timer:       a dict, if given, accumulating this rank's seconds per pass
                 over every call: 'qp_set.tau_sweep' (proj(tau) and the
                 slices), 'qp_set.frequency_pass' (the forward contraction
                 over the CD frequencies), 'qp_set.routes' (pass 1: routes,
                 roots, pole fits), 'qp_set.frequency_reverse' (pass 2),
                 'qp_set.direct_terms' (pass 3) and 'qp_set.reverse_sweep'
                 (the M^2 reverse sweep).
    pairs:       (i, j, wc_bar) with i, j positions into `states`: an adjoint
                 on the off-diagonal contraction wc_ij of
                 `cd_pair_contraction_multi`, (nfreq, norb), folded into the
                 same reverse pass; both states are reversed even at a zero
                 weight, and must reach the factors through the integral term.
    """
    if residue_route not in KERNEL_RESIDUE_ROUTES:
        raise ValueError(f'residue_route {residue_route!r}: one of '
                         f'{sorted(KERNEL_RESIDUE_ROUTES)}')
    run = QPSetPass(X, D, eps, nocc, grid, nu_points, nu_weights, states,
                    weights, residue_route, mu, xc_correction, w0, tol,
                    laplace_tol, tile_gb, follow, route_out, n_poles,
                    sop_stride, scissor, sop_poles, comm, rows_block, tape,
                    timer, pairs, keep_tape, z0)
    run.forward_contraction()
    run.solve_states()
    run.reverse_frequencies()
    run.direct_terms()
    return run.reverse_sweep()
