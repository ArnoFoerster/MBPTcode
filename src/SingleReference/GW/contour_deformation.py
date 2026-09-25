"""Quasiparticle energies by contour deformation: no analytic continuation.

Rotating the omega' contour in Sigma_c = (i/2pi) int G(omega+omega') W_c(omega')
onto the imaginary axis, and collecting the poles of G swept over:

    Sigma^c_pp(omega) = Sigma^int + Sigma^res

    Sigma^int = -(1/pi) int_0^inf dnu sum_q  (omega - eps_q)
                       / [(omega - eps_q)^2 + nu^2]  W^c_pq,qp(i.nu)

    Sigma^res = - sum_{i occ,  eps_i > omega} W^c_pi,ip(eps_i - omega)
                + sum_{a virt, eps_a < omega} W^c_pa,ap(omega - eps_a)

with a half-weight where a pole sits exactly on the contour. The imaginary-axis
term needs the screening only on the quadrature grid, where it is smooth; each
RESIDUE needs W at a REAL frequency |eps_q - omega|, which is where this route
costs and where it can be wrong. The imaginary-axis routes reach the real axis
by Pade-continuing Sigma_c instead, and that map is not differentiable: forward
mode through Thiele's recursion gives a derivative of order 1e11, because it
divides by inverse differences that approach zero.

What the diagonal self-energy of one state needs of the three-index tensor is
its slice Bp = B[:, p, :], (naux, norb): every W^c_pq,qp above is Bp[:,q]^T
[W - I] Bp[:,q]. The full tensor is (naux, norb, norb) -- 90 GB at 150 atoms --
and never enters. The imaginary-axis screening comes in as W(i.nu) Bp (`wbp`)
or, better, already contracted to wc[nu, q] = Bp[:,q]^T [W(i.nu) - I] Bp[:,q]
(`wc`, nfreq x norb): nothing else of it is ever needed, and each later Sigma
evaluation then costs nfreq x norb. Built from proj(tau) by
`cd_screening_contraction` the whole route is O(N^3); built explicitly from
C_ov by `wc_explicit` it is O(naux^2 nocc nvirt) per frequency.

WHAT MAKES IT WRONG. A residue frequency landing on a particle-hole transition
sits on a real pole of chi0, and W and its frequency derivative there are
arbitrarily large: `residue_pole_distance` reports how close. And omega = eps_q
puts a pole of G ON the contour, where the integrand of the imaginary-axis term
collapses onto nu = 0 and no quadrature resolves it -- hence the pole guard of
the Newton (`Solvers.qp_equation.solve_qp_equation_newton_guarded`), the
warning when it has to be relaxed, and `root_pole_distance` as the measure of
the narrowest feature the quadrature was asked to carry.

The residue SET is a discrete choice, and the total is analytic where it
changes -- the integral term compensates the jump -- so a gradient freezes the
set at the reference geometry and differentiates one fixed branch; re-deciding
it per geometry would put a step in the surface. The adjoints themselves live
in `gradients.contour_deformation_adjoint`.

`solve_qp_energy_contour` is the driver `calc_qp_energy(mode='space-time',
continuation='cd'|'laplace'|'sop')` enters: one ISDF factorization, one
proj(tau) sweep, one contour grid sized against the root-to-pole distance, and
then either this module's Newton or the pole model of `sum_over_poles`. 'cd'
and 'laplace' solve the SAME equation and differ only in the backend the
residues read W from: the explicit O(N^4) chi0(w') of
`real_screening.ExplicitRealScreening`, or the O(N^3) cosh transform of the
proj(tau) the integral term already built. The cubic one exists only below the
particle-hole gap and REFUSES above it rather than fall back, because a
continuation that silently changes route returns a different functional under
the name asked for.
"""
import warnings

import numpy as np
import scipy.linalg

from src.Base.constants import (CD_NFREQ, CD_NFREQ_MAX, CD_NTAU,
                                CD_POLE_RESOLUTION, HARTREE_TO_EV,
                                ISDF_TILE_GB, QP_CD_NEWTON_MAX_ITER,
                                QP_CD_NEWTON_TOL, QP_POLE_OFFSET,
                                QP_POLE_OFFSET_MIN, QP_POLE_STRENGTH_MIN,
                                RESIDUE_FREQ_MARGIN, RESIDUE_ON_CONTOUR_TOL,
                                SOP_FIT_STRIDE, SOP_N_POLES)
from src.Base.pyscf_interface import get_orbital_energies
from src.Base.utils.grids import gap_scaled_w0, gauss_legendre_grid
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.GW.real_screening import (ExplicitRealScreening,
                                                   LaplaceRealScreening,
                                                   ov_energies, screening_aux)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.space_time import (
    laplace_representation_error, owned_frequency_blocks,
    polarizability_projected_sweep, three_index_ov, three_index_slice)
from src.Solvers.qp_equation import solve_qp_equation_newton_guarded


def _need(C_ov, what):
    if C_ov is None:
        raise ValueError(
            f"{what} needs C_ov = B[:, occ, virt] as (naux, nocc*nvirt) and it "
            f"was not supplied. A frontier state sweeps no residues and needs "
            f"none once the imaginary-axis screening comes in as wbp or wc.")


def residue_set(eps, nocc, omega, tol=RESIDUE_ON_CONTOUR_TOL):
    """[(q, weight)] for the poles of G the deformation sweeps over."""
    out = []
    for q in range(len(eps)):
        gap = eps[q] - omega
        if abs(gap) < tol:
            out.append((q, -0.5 if q < nocc else 0.5))
        elif q < nocc and gap > 0.0:
            out.append((q, -1.0))
        elif q >= nocc and gap < 0.0:
            out.append((q, 1.0))
    return out


def residue_backend(real_screening, C_ov, eps, nocc, eta):
    """The backend the residue term uses: the one given, else explicit from C_ov."""
    if real_screening is not None:
        return real_screening
    _need(C_ov, 'the residue term')
    return ExplicitRealScreening(C_ov, eps, nocc, eta)


def residue_pole_distance(eps, nocc, omega, residues):
    """How close each residue frequency sits to a particle-hole transition.

    The residue term evaluates W at REAL frequencies, where chi0 has a pole at
    every eps_a - eps_i. A residue landing near one makes both W and its
    derivative large. This returns the distances so a caller can see which
    regime it is in rather than discover it in the gradient.
    """
    d = ov_energies(eps, nocc)
    return np.array([np.abs(d - abs(eps[q] - omega)).min() for q, _ in residues])


def root_pole_distance(eps, roots, states):
    """min over p, q != p of |w*_p - eps_q|: the narrowest spike on the contour.

    Sigma^int weights W(i.nu) by (w - eps_q)/[(w - eps_q)^2 + nu^2], a
    Lorentzian of half-width d_q = w - eps_q sitting at nu = 0, so this is the
    smallest feature the imaginary-frequency quadrature has to resolve. The
    orbital's own pole is excluded: the Newton's guard holds the root away from
    eps_p by construction, while a root that has walked up to a NEIGHBOURING
    level is what makes the quadrature the limiting approximation.
    `residue_pole_distance` is the companion on the real axis, where the poles
    belong to chi0 rather than to G.
    """
    eps = np.asarray(eps, float)
    out = np.inf
    for w, p in zip(np.atleast_1d(roots), np.atleast_1d(states)):
        d = np.abs(float(w) - eps)
        d[int(p)] = np.inf
        out = min(out, float(d.min()))
    return out


def screening_applied(chi0, Bp):
    """W(i.nu) Bp for every frequency: [1 - chi0]^-1 applied, never formed.

    One factorization plus one norb-column solve per frequency, about three
    times fewer operations than the inverse, and (nfreq, naux, norb) instead of
    (nfreq, naux, naux). Takes the whole chi0 axis, so it is for a small grid
    or a reference; `cd_screening_contraction` streams.
    `LinearResponse.imaginary_frequency.solve_rpa_screening_df` is the route
    that does form W, from the DF factors rather than from a chi0 handed in,
    and is not interchangeable with this one: it inverts where this solves.
    """
    eye = np.eye(chi0.shape[-1])
    return np.linalg.solve(eye - chi0, np.broadcast_to(Bp, chi0.shape[:1]
                                                       + Bp.shape))


def screening_contraction(Bp, wbp):
    """wc[k, q] = sum_P Bp[P,q] [W(i.nu_k) Bp - Bp][P,q] -- NO omega in it.

    The imaginary-axis term's only omega dependence is the scalar
    (omega - eps_q) / [(omega - eps_q)^2 + nu^2], so this is built once and each
    later Sigma evaluation costs nfreq x norb instead of a full sweep over the
    screening.
    """
    return np.einsum('Pq,kPq->kq', Bp, wbp - Bp, optimize=True)


def wc_explicit(Bp, C_ov, d, nu_points):
    """The same contraction from the explicit particle-hole screening: O(N^4)
    per frequency, the reference the cubic route is checked against."""
    return np.stack([np.einsum('Pq,PQ,Qq->q', Bp,
                               screening_aux(C_ov, d, nu, True)[0], Bp,
                               optimize=True) for nu in nu_points])


def cd_screening_contraction(proj_tau, cosft_wt, Bp, tile_gb=ISDF_TILE_GB):
    """wc[k, q] = Bp[:,q]^T [W(i.nu_k) - I] Bp[:,q] from proj(tau), streamed.

    chi0(i.nu) comes out of proj(tau) one frequency block at a time; one LU and
    one norb-column solve per frequency, W never formed, no frequency outliving
    its block. `cd_screening_contraction_multi` does the same for several bra
    states at once and should be preferred when there is more than one.
    """
    return cd_screening_contraction_multi(proj_tau, cosft_wt, [Bp],tile_gb=tile_gb)[0]


def cd_screening_contraction_multi(proj_tau, cosft_wt, Bps,
                                   tile_gb=ISDF_TILE_GB, freq_indices=None):
    """`cd_screening_contraction` for a LIST of bra slices, sharing the work.

    The matrix [1 - chi0(i.nu)] depends on the frequency alone, not on which
    state is being screened, so rebuilding chi0 and factorizing it once per
    state is pure waste. Frequency runs OUTSIDE and states inside: one
    tensordot and one LU per frequency serve every state, each of which then
    costs only a triangular solve. For a BSE quasiparticle set of n states that
    is n times fewer factorizations and n times fewer chi0 rebuilds.

    freq_indices: the frequencies this rank computes; every other row of each
    wc stays zero, so the sum over ranks is the whole. A frequency's result is
    its own row and the LU is the cost, which makes this the reduction-free
    axis to split (nfreq x norb per state to gather, 8 MB at dodecacene).
    The chi0 rows are the serial block's rows on any BLAS
    (`owned_frequency_blocks`), so the sum over ranks, which adds exact
    zeros, is the serial wc bitwise. Serial (None) is bitwise unchanged.
    """
    nfreq, naux = cosft_wt.shape[0], proj_tau.shape[-1]
    eye = np.eye(naux)
    out = [np.zeros((nfreq, B.shape[1])) for B in Bps]
    for ks, blk in owned_frequency_blocks(proj_tau, cosft_wt, tile_gb,
                                          freq_indices):
        for m, k in enumerate(ks):
            lu = scipy.linalg.lu_factor(eye - blk[m])
            for B, wc in zip(Bps, out):
                WtB = scipy.linalg.lu_solve(lu, B)
                wc[k] = np.einsum('Pq,Pq->q', B, WtB - B)
    return out


def residue_route_auto(grid, eps, nocc, start, tol, margin=RESIDUE_FREQ_MARGIN):
    """'none' | 'laplace' | 'explicit' for the pole set at the Newton start.

    Laplace when the grid carries every residue frequency plus `margin` to the
    backend's tolerance; explicit otherwise. The rule is deliberately made
    from the START point so that finite differences of the energy, which re-run
    the decision at every displaced geometry, land on one branch.
    """
    expected = residue_set(eps, nocc, start)
    if not expected:
        return 'none'
    freqs = [abs(eps[q] - start) + margin for q, _ in expected]
    ok = all(laplace_representation_error(grid, eps, nocc, f) < tol for f in freqs)
    return 'laplace' if ok else 'explicit'


def sigma_cd(p, omega, Bp, eps, nocc, nu_points, nu_weights, residues=None,
             wbp=None, eta=0.0, wc=None, C_ov=None, real_screening=None):
    """Sigma^c_pp(omega) by contour deformation, from the state's slice Bp.

    residues:       the frozen pole set from `residue_set`; recomputed at omega
                    when omitted, which is right for a scan and wrong for a
                    gradient.
    wc / wbp:       the imaginary-axis screening, contracted
                    (`screening_contraction`) or as W(i.nu) Bp. With neither it
                    is built here from C_ov, explicitly, at O(N^4) per frequency.
    real_screening: the backend the residues take W(real freq) from (module
                    docstring); None with C_ov given means the explicit one.
    C_ov:           B[:, occ, virt] as (naux, nocc*nvirt), for the explicit
                    imaginary-axis screening and the explicit residue backend.
    """
    de = omega - eps
    if wc is None:
        if wbp is None:
            _need(C_ov, 'the imaginary-axis screening without wbp or wc')
            wc = wc_explicit(Bp, C_ov, ov_energies(eps, nocc), nu_points)
        else:
            wc = screening_contraction(Bp, wbp)
    g = de[None, :] / (de[None, :] ** 2 + np.asarray(nu_points)[:, None] ** 2)
    s_int = -float(np.asarray(nu_weights) @ np.einsum('kq,kq->k', g, wc)) / np.pi

    res = residue_set(eps, nocc, omega) if residues is None else residues
    s_res = 0.0
    if res:
        rs = residue_backend(real_screening, C_ov, eps, nocc, eta)
        for q, weight in res:
            b = Bp[:, q]
            y = rs.apply(abs(eps[q] - omega), b)
            s_res += weight * (b @ y - b @ b)
    return s_int + s_res


def sigma_cd_slope(p, omega, Bp, eps, nocc, nu_points, nu_weights, residues, wc,
                   eta=0.0, C_ov=None, real_screening=None):
    """dSigma^c_pp/domega at fixed (eps, Bp), residue set frozen -- closed form.

    The integral term's only omega dependence is g_q = (omega - eps_q) /
    [(omega - eps_q)^2 + nu^2], so its slope costs nfreq x norb given wc. A
    residue evaluates W at |eps_q - omega|, so its slope is the frequency
    derivative of the screening, W [dchi0/dfreq] W, contracted with Bp[:,q] on
    both sides: one W build per residue, as the value itself costs.

    This is the slope that sets Z, and it has to be exact: Z multiplies the
    whole gradient. A central-difference probe of Sigma has an O(dw^2)
    truncation error against a third derivative that a nearby particle-hole
    pole makes enormous; at 0.016 Ha clearance on water/cc-pVDZ it put 1.1e-5
    on Z and therefore on every component of the adjoint, with a step scan flat
    across three decades -- the signature of a constant factor, not of a
    differencing error.
    """
    de = omega - eps
    nu = np.asarray(nu_points)[:, None]
    dg = (nu ** 2 - de[None, :] ** 2) / (de[None, :] ** 2 + nu ** 2) ** 2
    slope = -float(np.asarray(nu_weights) @ np.einsum('kq,kq->k', dg, wc)) / np.pi
    if residues:
        rs = residue_backend(real_screening, C_ov, eps, nocc, eta)
        for q, weight in residues:
            gap = eps[q] - omega
            freq = abs(gap)
            y = rs.apply(freq, Bp[:, q])
            slope -= weight * rs.slope(freq, y) * np.sign(gap)
    return slope


def qp_energy_cd(p, Bp, eps, nocc, nu_points, nu_weights, xc_correction=0.0,
                 tol=QP_CD_NEWTON_TOL, max_iter=QP_CD_NEWTON_MAX_ITER, w0=None,
                 wbp=None, wc=None, C_ov=None, pole_offset=QP_POLE_OFFSET,
                 eta=0.0, real_screening=None, linearize_on_capture=True,
                 relax_offset=True, guard_out=None,
                 offset_min=QP_POLE_OFFSET_MIN, z_min=QP_POLE_STRENGTH_MIN,
                 linear_offset=QP_POLE_OFFSET):
    """(eps^QP_p, Z_p, residues) from w = eps_p + <Sigma_x - v_xc> + Sigma^c(w).

    The guarded Newton of `Solvers.qp_equation`, with the closed-form slope
    (`sigma_cd_slope`) and the residue set re-decided at every iterate and
    FROZEN at the converged root -- the gradient must differentiate one fixed
    analytic branch. Z = 1/(1 - dSigma/domega) comes from the same slope.
    xc_correction is zero on a Hartree-Fock reference.

    The imaginary-axis screening enters as wc (nfreq x norb, all that is ever
    needed of it), as wbp contracted once here, or -- with neither -- explicitly
    from C_ov, O(N^4) per frequency and contracted ONCE rather than once per
    Newton step.

    pole_offset / relax_offset / linearize_on_capture / guard_out / offset_min /
    z_min / linear_offset are the guard, and are documented on
    `solve_qp_equation_newton_guarded`: the iterate is held that far from every
    orbital energy, because at omega = eps_q the pole of G sits ON the contour
    and the imaginary-axis integrand collapses onto nu = 0.
    """
    if wc is None:
        if wbp is not None:
            wc = screening_contraction(Bp, wbp)
        else:
            _need(C_ov, 'the imaginary-axis screening without wbp or wc')
            wc = wc_explicit(Bp, C_ov, ov_energies(eps, nocc), nu_points)
    common = dict(eta=eta, C_ov=C_ov, real_screening=real_screening)

    def sigma(w):
        return sigma_cd(p, w, Bp, eps, nocc, nu_points, nu_weights,
                        residues=residue_set(eps, nocc, w), wc=wc, **common)

    def slope(w):
        return sigma_cd_slope(p, w, Bp, eps, nocc, nu_points, nu_weights,
                              residue_set(eps, nocc, w), wc, **common)

    w, z = solve_qp_equation_newton_guarded(
        sigma, slope, eps, p, nocc, xc_correction=xc_correction, w0=w0,
        tol=tol, max_iter=max_iter, pole_offset=pole_offset,
        relax_offset=relax_offset, linearize_on_capture=linearize_on_capture,
        guard_out=guard_out, offset_min=offset_min, z_min=z_min,
        linear_offset=linear_offset)
    return w, z, residue_set(eps, nocc, w)


def cd_grid_range(eps, nocc, e_min_below_gap=None):
    """(e_min, e_max) the contour's imaginary-time grid is fitted on.

    e_min reaches BELOW the particle-hole gap, to gap - e_min_below_gap and by
    default to half the gap, because a residue needs the cosh transform of
    proj(tau) at a real frequency w' and therefore the grid's 1/y fit down to
    gap - w'. A shift that eats the whole gap is refused here rather than
    handed to a quadrature as e^{-y tau} at y <= 0, which nothing represents.

    Separate from `cd_frequency_grid` so that a caller recording WHAT RANGE a
    surface was fitted on reads the same rule the grid was built from.
    """
    eps = np.asarray(eps, float)
    occ, virt = get_occ_virt_indices(eps, nocc)
    gap = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    shift = (gap_scaled_w0(eps, nocc) if e_min_below_gap is None
             else float(e_min_below_gap))
    if gap - shift <= 0:
        raise ValueError(f'e_min_below_gap {shift} exceeds the gap {gap:.4f}: '
                         f'the imaginary-time grid would be asked to represent '
                         f'e^{{-y tau}} at y <= 0, which no quadrature does.')
    return gap - shift, e_max


def cd_frequency_grid(eps, nocc, ntau=CD_NTAU, nfreq_cd=CD_NFREQ, w0_cd=None,
                      e_min_below_gap=None):
    """(nu, nu_weights, grid, w0) of the contour-deformation quadrature.

    The nu axis is Gauss-Legendre scaled to w0, which defaults to half the
    particle-hole gap: the integrand's Lorentzian (omega - eps_q) /
    [(omega - eps_q)^2 + nu^2] varies on the scale of the quasiparticle
    correction, and tying the grid to the gap makes one point count serve
    systems whose spectra differ by an order of magnitude.

    `grid` carries that axis as its frequency side, so chi0(i.nu) is a fixed
    linear combination of proj(tau) through `grid.cosft_wt`. Its tau axis
    reaches BELOW the gap, to e_min = gap - e_min_below_gap and by default to
    half the gap, because a residue needs the cosh transform of proj(tau) at a
    real frequency w' and therefore the grid's 1/y fit down to gap - w'. The
    cosine transform alone would want only [gap, e_max], and widening the range
    costs time points -- which is why `CD_NTAU` is 24 and not the 18 the
    imaginary axis alone converges at.
    """
    e_min, e_max = cd_grid_range(eps, nocc, e_min_below_gap)
    w0 = gap_scaled_w0(eps, nocc) if w0_cd is None else float(w0_cd)
    nu, nu_weights = gauss_legendre_grid(int(nfreq_cd), w0=w0)
    grid = TimeFrequencyGrid.minimax_split(int(ntau), e_min, e_max, nu,
                                           nu_weights, with_sine=False,
                                           with_inverse=False)
    return nu, nu_weights, grid, w0


def cd_grid_resolves(nu, eps, roots, states, resolution=CD_POLE_RESOLUTION):
    """Whether the quadrature's smallest node resolves the root-to-pole spike.

    A pole of G at a NEIGHBOURING orbital energy puts a Lorentzian of
    half-width d = |eps^QP_p - eps_q| on the imaginary-frequency integrand at
    nu = 0 (`root_pole_distance`). A grid whose smallest node sits at a
    comparable frequency integrates the wrong function there, and the Newton
    then converges on whatever zero the truncated self-energy has -- a
    satellite, or nothing at all.
    """
    return bool(float(np.min(nu)) * resolution
                <= root_pole_distance(eps, roots, states))


def newton_seeds(eps, nocc, states, pole_offset=None):
    """(seeds, guard band, whether it may still relax) for a fresh solve.

    eps_p sits exactly on a pole of G, so the Newton starts there pushed
    `pole_offset` toward the chemical potential -- the direction the
    quasiparticle correction takes. This is the convention
    `gradients.qp_space_time` applies to a state it has frozen nothing for, and
    the pole model reads the same start, both for its own Newton and for the
    compression test. A pole_offset given explicitly FREEZES the guard band, so
    that every geometry takes the same Newton path and the solver warns before
    leaving it; None lets each solve relax a band its root sits inside.
    """
    offset = QP_POLE_OFFSET if pole_offset is None else float(pole_offset)
    seeds = np.array([float(eps[int(p)]
                            + (offset if int(p) < nocc else -offset))
                      for p in np.atleast_1d(states)])
    return seeds, offset, pole_offset is None


def _laplace_representation_error(real_screening, eps, omega, residues):
    """The worst bare-quadrature error the cosh transform was read at.

    The measurable behind the route's validity: a residue's W comes out of
    proj(tau) only while the grid's 1/y fit still carries every pair energy
    d -/+ |eps_q - omega|, and this is that fit's residual at the frequencies
    the converged root actually asked for. None where no residue was swept.
    """
    if not residues:
        return None
    return max(real_screening.representation_error(abs(eps[q] - omega))
               for q, _ in residues)


def _contour_roots(continuation, states, seeds, admission, Bps, wcs, eps, nocc,
                   nu, nu_weights, xc_correction, offset, relax, n_poles,
                   sop_stride, real_screening):
    """(roots, pole strengths, per-state records) for one contour grid."""
    # cycle: `sum_over_poles` takes `residue_set` from this module.
    from src.SingleReference.GW.sum_over_poles import (qp_energy_sop,
                                                      sop_from_wc)
    roots = np.zeros(len(states))
    pole_strengths = np.zeros(len(states))
    records = []
    for i, p in enumerate(states):
        p, seed = int(p), float(seeds[i])
        xc_p = float(np.asarray(xc_correction).ravel()[i])
        guard = {'pole_offset': offset}
        error = None
        if continuation == 'sop':
            poles, amplitudes = sop_from_wc(wcs[i], nu, eps, nocc,
                                            n_poles=n_poles,
                                            stride=sop_stride)
            w_star, z = qp_energy_sop(p, amplitudes, poles, eps, nocc,
                                      xc_correction=xc_p, w0=seed)
            residues = []
        else:
            try:
                w_star, z, residues = qp_energy_cd(
                    p, Bps[i], eps, nocc, nu, nu_weights, xc_correction=xc_p,
                    w0=seed, wc=wcs[i], real_screening=real_screening,
                    pole_offset=offset, relax_offset=relax, guard_out=guard)
            except ValueError as refusal:
                # The cubic backend refuses a residue frequency its grid does
                # not carry, and names the frequency; the state is what the
                # caller asked for and only this loop knows it.
                if continuation != 'laplace':
                    raise
                raise ValueError(
                    f"continuation='laplace' cannot serve orbital {p}: its "
                    f"contour sweeps a residue the imaginary-time grid does "
                    f"not represent. {refusal}") from refusal
            if continuation == 'laplace':
                error = _laplace_representation_error(real_screening, eps,
                                                      w_star, residues)
        roots[i] = w_star
        pole_strengths[i] = z
        records.append({'state': p, 'newton_seed': seed,
                        'pole_offset': float(guard['pole_offset']),
                        'residues': len(residues),
                        'representation_error': error,
                        'sop_admits': bool(admission[i][0]),
                        'sop_reach': float(admission[i][1])})
    return roots, pole_strengths, records


def solve_qp_energy_contour(mf, mol, nocc, states, continuation='cd',
                            ntau=CD_NTAU, nfreq_cd=CD_NFREQ, w0_cd=None,
                            e_min_below_gap=None, pole_offset=None,
                            n_poles=SOP_N_POLES, sop_stride=SOP_FIT_STRIDE,
                            auxbasis=None, radii=None, counts=None,
                            factors=None, grid_accuracy=None, sigma_x='mf',
                            dm_correction=None, eps_anchor=None,
                            tile_gb=ISDF_TILE_GB):
    """(quasiparticle energies in eV, pole strengths, diagnostics), no continuation.

    GW@RPA on a restricted, density-fitted reference, the same quantity as
    `calc_qp_energy(selfenergy='GW', polarizability='RPA')` and the peer of
    `space_time.solve_qp_energy_space_time`, which reaches the real axis by
    Pade instead. Three ways off the imaginary axis live here:

      'cd'      the contour deformation of the module docstring -- exact for
                any state, and O(N^4) per residue because each one evaluates W
                at a real frequency.
      'laplace' the same contour, with each residue's W from the cosh
                transform of the proj(tau) the integral term already built:
                O(N^3), and defined only below the particle-hole gap, where
                the transform exists. A residue the tau grid does not carry is
                refused by name rather than served from the other backend.
      'sop'     W modelled by `n_poles` auxiliary poles, Sigma_c closed form,
                no real-frequency screening at all; refuses the states
                Eq. (27) excludes (`sum_over_poles.compressible`).

    Z = 1/(1 - dSigma/domega) comes from the same closed-form slope as the
    Newton step, so it is the derivative of the equation that was solved.

    states:      the orbitals to solve; one proj(tau), one contour grid and one
                 exchange build serve all of them.
    ntau:        imaginary-time points behind the grid (`cd_frequency_grid`).
    nfreq_cd:    contour-deformation quadrature points, DOUBLED up to
                 `CD_NFREQ_MAX` while the grid does not resolve the
                 root-to-pole spike (`cd_grid_resolves`).
    pole_offset: the Newton's guard band; None relaxes it per solve.
    sigma_x:     which K builds the static exchange, see
                 `qp_solve.static_exchange_diagonal`.
    eps_anchor:  the eps_p anchoring w = eps_p + <Sigma_x - v_xc> +
                 Re Sigma_c(w) when it differs from the spectrum that builds
                 G, P0 and W -- the evGW case. It shifts the right-hand side by
                 a constant and nothing else, so it enters as an addition to
                 the static correction and leaves the residue set and the pole
                 guard, which belong to the poles of G, on the mean field.
    """
    if continuation not in ('cd', 'laplace', 'sop'):
        raise ValueError(f"solve_qp_energy_contour runs continuation='cd', "
                         f"'laplace' or 'sop', not {continuation!r}")
    # cycle: `sum_over_poles` takes `residue_set` from this module.
    from src.SingleReference.GW.sum_over_poles import compressible

    eps = get_orbital_energies(mf, representation='spatial')
    states = np.atleast_1d(states).astype(int)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])

    # The Newton starts, the guard band, and the compression test taken on the
    # same start. Eq. (27) needs the spectrum alone, so a state the pole model
    # cannot carry is refused HERE -- discovering it after the tau sweep is the
    # whole route spent on a refusal.
    seeds, offset, relax = newton_seeds(eps, nocc, states,
                                        pole_offset=pole_offset)
    admission = [compressible(float(w), eps, nocc) for w in seeds]
    if continuation == 'sop':
        for p, seed, (admits, reach) in zip(states, seeds, admission):
            if not admits:
                raise ValueError(
                    f"continuation='sop' cannot serve orbital {int(p)}: its "
                    f'Newton starts at {seed:.4f} Ha, which sweeps a pole '
                    f'{reach:.3f} particle-hole gaps away, and Eq. (27) of the '
                    f'pole paper admits only reach < 1. Beyond it one '
                    f'excitation of W is resonant, the individual poles matter '
                    f'rather than their envelope and no number of them '
                    f"converges; use continuation='cd'.")

    X_mo, D = (tuple(factors)[:2] if factors is not None else
               separable_factors(mf, mol, auxbasis=auxbasis, radii=radii,
                                 counts=counts,
                                 grid_accuracy=grid_accuracy)[:2])

    # <Sigma_x - v_xc>_pp: one exchange build for the whole window, zero to
    # round-off on a gas-phase Hartree-Fock reference.
    xc_correction = static_exchange_diagonal(mf, mol, states,
                                             dm_correction=dm_correction,
                                             exchange=sigma_x)
    if eps_anchor is not None:
        xc_correction = xc_correction + (np.asarray(eps_anchor, float)[states]
                                         - eps[states])

    # The residue term's backend, built only where a residue can ask for it:
    # the pole model never evaluates W off the imaginary axis. The explicit one
    # holds C_ov, which depends on no state and on no grid, so it is built once
    # here; the cubic one reads proj(tau) and is rebuilt with it below.
    real_screening = None
    if continuation == 'cd':
        real_screening = ExplicitRealScreening(
            three_index_ov(X_mo, D, eps, nocc, tile_gb=tile_gb), eps, nocc)

    nfreq = int(nfreq_cd)
    while True:
        nu, nu_weights, grid, w0 = cd_frequency_grid(
            eps, nocc, ntau=ntau, nfreq_cd=nfreq, w0_cd=w0_cd,
            e_min_below_gap=e_min_below_gap)
        proj_tau = polarizability_projected_sweep(X_mo, D, eps, nocc,
                                                  grid.tau_points, mu=mu,
                                                  tile_memory_gb=tile_gb)
        Bps = [three_index_slice(X_mo, D, int(p), tile_gb=tile_gb)
               for p in states]
        wcs = cd_screening_contraction_multi(proj_tau, grid.cosft_wt, Bps,
                                             tile_gb=tile_gb)
        if continuation == 'laplace':
            # The residues read the SAME proj(tau) the imaginary-axis term did,
            # through cosh instead of cos; the backend keeps it alive.
            real_screening = LaplaceRealScreening(proj_tau, grid, eps, nocc)
        del proj_tau
        roots, pole_strengths, records = _contour_roots(
            continuation, states, seeds, admission, Bps, wcs, eps, nocc, nu,
            nu_weights, xc_correction, offset, relax, n_poles, sop_stride,
            real_screening)
        # Sized against the roots it just produced, and doubled at most to the
        # ceiling: the sizing needs the roots and the roots need a grid.
        resolved = cd_grid_resolves(nu, eps, roots, states)
        if resolved or nfreq >= CD_NFREQ_MAX:
            break
        nfreq = min(2 * nfreq, CD_NFREQ_MAX)
    if not resolved:
        warnings.warn(
            f'a quasiparticle root of this set lies '
            f'{root_pole_distance(eps, roots, states):.2e} Ha from a '
            f'neighbouring orbital energy, which the contour-deformation '
            f'quadrature does not resolve at its {CD_NFREQ_MAX}-point ceiling '
            f'(smallest frequency {np.min(nu):.2e} Ha). The self-energy of '
            f'that orbital is integrated over a spike the grid steps across.',
            RuntimeWarning, stacklevel=2)

    diagnostics = {'continuation': continuation, 'ntau': int(ntau),
                   'nfreq_cd': int(nfreq), 'w0_cd': float(w0),
                   # the residue treatment that RAN: 'cd' asks the
                   # explicit backend, the other two are their own
                   'residue_route_taken': ('explicit' if continuation == 'cd'
                                           else continuation),
                   'cd_grid_resolved': bool(resolved), 'states': records}
    return roots * HARTREE_TO_EV, pole_strengths, diagnostics
