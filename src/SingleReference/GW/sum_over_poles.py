"""The self-energy as a sum over M auxiliary poles: no contour, no residues.

Contour deformation reaches the real axis by splitting Sigma^c into an
imaginary-axis integral and a sum over the poles of G that the rotation sweeps
(`GW.contour_deformation`). Every one of those residues needs W at a REAL
frequency, and therefore its own naux^3 inversion of [1 - chi0(omega')]; the
ones beyond the particle-hole gap need an O(N^4) chi0(omega') as well, because
the cosh transform of the imaginary-time susceptibility diverges there.

Modelling W itself by M common poles removes the split instead of accelerating
it,

    Wt_pq,qp(z) ~= sum_m A^pq_m [1/(z - Om_m) - 1/(z + Om_m)] ,

which, put into the frequency convolution, integrates in closed form to

    Sigma^c_pp(omega) = sum_{i,m} A^pi_m / (omega - eps_i + Om_m)
                      + sum_{a,m} A^pa_m / (omega - eps_a - Om_m) .

The omega dependence is then analytic: no argument of W depends on omega, so
the screening is never evaluated off the imaginary axis, the Newton iteration
costs nothing after the fit, and there is no residue SET whose membership can
change along a nuclear coordinate.

What the fit consumes is exactly `GW.contour_deformation.screening_contraction`'s
wc[k, q] = Bp[:,q]^T [W(i.nu_k) - I] Bp[:,q], which the integral term already
builds, so it costs no screening evaluation of its own. With the Om_m FROZEN
after `fit_poles`, A = F^+ wc is a fixed LINEAR map of that data, which is what
makes the adjoint of the whole route that map transposed
(`gradients.sum_over_poles_adjoint`).

THE WALL. Eq. (27) of the pole paper: the denominators stay bounded away from
zero only while |eps_q - omega| < Om_1, the lowest true neutral excitation, for
which the dRPA guarantees the particle-hole gap E_g as a lower bound. Inside
it the self-energy sees the pole distribution of W through a smooth functional
and M poles reproduce it. Beyond it one excitation is resonant, the individual
Om_m matter rather than their envelope, and no M converges -- the error scatters
with M rather than falling. `compressible` is that condition, on E_g and on the
poles the contour would sweep, and it is what makes a number here believable:
a core or inner-valence state fails it permanently, as it does for the Laplace
residue route, and a fitted pole set that has crowded onto the evaluation point
fails `pole_clearance` instead.
"""
import warnings

import numpy as np

from src.Base.constants import (QP_CD_NEWTON_MAX_ITER, QP_CD_NEWTON_TOL, SOP_CLEARANCE_MIN, SOP_FIT_RCOND, SOP_FIT_STRIDE, SOP_N_POLES)
from src.SingleReference.GW.contour_deformation import residue_set
from src.SingleReference.GW.real_screening import ov_energies


def initial_poles(n_poles, gap, e_max):
    """Log-spaced starting poles on [gap, e_max], the naive common-pole set.

    Log spacing rather than linear because W varies on the scale of the gap at
    the bottom of the range and on the scale of the whole particle-hole
    spectrum at the top.
    """
    if not 0.0 < gap < e_max:
        raise ValueError(f'poles need 0 < gap < e_max, got {gap} and {e_max}')
    return np.logspace(np.log10(gap), np.log10(e_max), int(n_poles))


def pole_basis(nu_points, poles):
    """F[k, m] = 1/(i.nu_k - Om_m) - 1/(i.nu_k + Om_m), which is REAL.

    The two terms of the model combine to 2 Om/((i.nu)^2 - Om^2) =
    -2 Om/(nu^2 + Om^2): the model is even in z and the fit is real arithmetic
    throughout, which is why no complex least squares appears anywhere here.
    """
    nu = np.asarray(nu_points, float)[:, None]
    om = np.asarray(poles, float)[None, :]
    return -2.0 * om / (nu ** 2 + om ** 2)


def pole_pseudoinverse(nu_points, poles, rcond=SOP_FIT_RCOND):
    """F^+, the fixed linear map from imaginary-axis data to amplitudes.

    The forward fit and its adjoint must be the SAME operator or the gradient
    is the derivative of a slightly different function; both go through here so
    that a truncation of the least-squares spectrum cannot differ between them.
    """
    return np.linalg.pinv(pole_basis(nu_points, poles), rcond=rcond)


def pole_amplitudes(wc, poles, nu_points, rcond=SOP_FIT_RCOND):
    """A[m, q] = F^+ wc[:, q], one solve per orbital.

    The columns are independent and share F, so this is one factorization and
    a matrix product -- the linearity that makes the derivative a transpose.
    """
    return pole_pseudoinverse(nu_points, poles, rcond) @ np.asarray(wc, float)


def fit_poles(wc, nu_points, poles, n_iter=5, stride=1, bounds=None):
    """Relocate the poles by vector fitting, in the variable u = z^2.

    Gustavsen-Semlyen: solve the linear problem for the amplitudes of f and of
    a weight sigma sharing the same poles, then take the new poles as the zeros
    of sigma, the eigenvalues of diag(u_m) - 1 b^T. The model is even in z, so
    everything is done in u = z^2 -- samples at u = -nu^2 < 0, poles at
    u = Om^2 > 0 -- and stays real; a complex-conjugate pair in u would mean the
    fit had left the physical manifold, which the clip to `bounds` prevents.

    The relocation buys about a factor two in M (benzene/cc-pVDZ: M = 8 fitted
    matches M = 16 log-spaced, and M = 12 is within 0.012 meV of the CD value),
    and it is done ONCE, because the poles are then frozen for the gradient.

    stride: fit every stride-th column of wc. The poles are common to all
            orbitals, so they need only enough columns to be determined, and
            the least-squares problem grows linearly in the number kept.
    """
    wc = np.asarray(wc, float)
    om = np.asarray(poles, float).copy()
    n_poles = om.size
    lo, hi = bounds if bounds is not None else (om[0], om[-1])
    u_s = -np.asarray(nu_points, float) ** 2
    cols = range(0, wc.shape[1], max(int(stride), 1))
    f_cols = [wc[:, q] for q in cols]
    n_col = len(f_cols)
    for _ in range(int(n_iter)):
        kern = 1.0 / (u_s[:, None] - (om ** 2)[None, :])
        block = np.zeros((len(u_s) * n_col, n_col * n_poles + n_poles))
        rhs = np.concatenate(f_cols)
        for j, f in enumerate(f_cols):
            rows = slice(j * len(u_s), (j + 1) * len(u_s))
            block[rows, j * n_poles:(j + 1) * n_poles] = kern
            block[rows, -n_poles:] = -f[:, None] * kern
        x, *_ = np.linalg.lstsq(block, rhs, rcond=None)
        zeros = np.linalg.eigvals(np.diag(om ** 2)
                                  - np.ones((n_poles, 1)) @ x[-n_poles:][None, :])
        om = np.sqrt(np.sort(np.clip(np.real(zeros), lo ** 2, hi ** 2)))
    return om


def denominators(omega, poles, eps, nocc):
    """D[q, m] = omega - eps_q + Om_m for occupied q, - Om_m for virtual q."""
    sign = np.where(np.arange(len(eps)) < nocc, 1.0, -1.0)
    return (omega - np.asarray(eps, float))[:, None] + sign[:, None] * np.asarray(
        poles, float)[None, :]


def pole_clearance(omega, poles, eps, nocc):
    """min |D| / Om_1: how close the model is to a resonance of its own.

    Necessary and NOT sufficient: it passes a carbon 1s at 0.935 that is still
    32 meV out. `compressible` decides; this only catches a model whose poles
    have crowded onto the evaluation point.
    """
    return float(np.abs(denominators(omega, poles, eps, nocc)).min()
                 / np.min(poles))


def compressible(omega, eps, nocc):
    """(ok, reach) for the compression condition: is every pole the contour
    sweeps closer to omega than the particle-hole gap?

    THE THRESHOLD IS E_g, NOT THE LOWEST FITTED POLE. Om_1 of the condition is
    the lowest true neutral excitation and the dRPA guarantees Om_1 >= E_g; a
    fitted pole set is free to place its smallest member above the gap, which
    loosens the test where it has to bite -- a state 734 meV out passes at
    0.9936 against its own poles.

    On E_g this is "how many residues lie above the gap", so it needs no
    screening, no fit and no poles, and can be asked before fitting anything.

    Only the SWEPT poles matter: every other orbital reaches the model through a
    denominator bounded away from zero and an amplitude the fit reproduces in
    the aggregate. An amplitude-weighted measure of the self-energy coming from
    |eps_q - omega| >= Om_1 therefore ranks the states BACKWARDS -- frontier
    0.28 to 0.59, carbon 1s 0.015 -- and must not be used. The residue count
    separates them exactly.

    The same wall as `GW.real_screening.LaplaceRealScreening`: a state within
    one gap of the frontier is compressible AND has a cubic residue route,
    and a deeper one has neither.

    reach: the worst swept pole in units of E_g. ok is reach < 1.
    """
    limit = float(ov_energies(eps, nocc).min())
    swept = residue_set(eps, nocc, omega)
    reach = max((abs(eps[q] - omega) for q, _ in swept), default=0.0) / limit
    return bool(reach < 1.0), float(reach)


def sigma_sop(omega, amplitudes, poles, eps, nocc, check=True):
    """Sigma^c_pp(omega) from the pole model -- closed form, no screening call.

    check: refuse the states the compression condition excludes
           (`compressible`) and the near-resonant models `pole_clearance`
           catches. Beyond the wall the error scatters with M rather than
           falling, so a converged-looking value at one M means nothing.
    """
    if check:
        ok, reach = compressible(omega, eps, nocc)
        if not ok:
            warnings.warn(
                f'omega = {omega:.4f} sweeps a pole {reach:.2f} gaps '
                f'away, so Eq. (27) excludes this state from '
                f'the compression: the individual poles of W matter there and '
                f'no number of them converges. Use the real-axis route.',
                RuntimeWarning, stacklevel=2)
        clear = pole_clearance(omega, poles, eps, nocc)
        if clear < SOP_CLEARANCE_MIN:
            warnings.warn(
                f'the pole model is resonant at omega = {omega:.4f}: its '
                f'closest denominator is {clear:.2g} of the lowest auxiliary '
                f'pole.', RuntimeWarning, stacklevel=2)
    return float(np.sum(np.asarray(amplitudes, float).T
                        / denominators(omega, poles, eps, nocc)))


def sigma_sop_slope(omega, amplitudes, poles, eps, nocc):
    """dSigma^c_pp/domega at fixed (eps, A) -- the closed form that sets Z.

    Contour deformation pays a W build and a solve per residue for this; here
    it is the same sum with squared denominators.
    """
    return -float(np.sum(np.asarray(amplitudes, float).T
                         / denominators(omega, poles, eps, nocc) ** 2))


def qp_energy_sop(p, amplitudes, poles, eps, nocc, xc_correction=0.0,
                  tol=QP_CD_NEWTON_TOL, max_iter=QP_CD_NEWTON_MAX_ITER,
                  w0=None):
    """(eps^QP_p, Z_p) from w = eps_p + <Sigma_x - v_xc> + Sigma^c(w).

    Newton with the closed-form slope. There is no pole guard and no residue
    set to re-decide: the model has no pole ON the real axis where the
    quasiparticle lives, so nothing here has to be frozen for a gradient. What
    can still fail is the compression condition, which `pole_clearance` reports
    and which no iteration repairs.

    This is NOT `Solvers.qp_equation.solve_qp_equation_newton_guarded`: there is
    no pole on the real axis to guard the iterate against, and the residual is
    spelled in the model's own order, eps_p + xc + Sigma - w, which is that
    solver's negated step to within a last-bit rounding the frozen numbers of
    this route are pinned to.
    """
    w = float(eps[p] if w0 is None else w0)
    for _ in range(int(max_iter)):
        s = sigma_sop(w, amplitudes, poles, eps, nocc, check=False)
        slope = sigma_sop_slope(w, amplitudes, poles, eps, nocc)
        step = (eps[p] + xc_correction + s - w) / (1.0 - slope)
        w += step
        if abs(step) < tol:
            break
    else:
        warnings.warn(f'the quasiparticle Newton for orbital {p} did not '
                      f'converge to {tol:g} in {max_iter} steps',
                      RuntimeWarning, stacklevel=2)
    slope = sigma_sop_slope(w, amplitudes, poles, eps, nocc)
    return w, 1.0 / (1.0 - slope)


def sop_from_wc(wc, nu_points, eps, nocc, n_poles=SOP_N_POLES, relocate=True,
                stride=SOP_FIT_STRIDE, e_max=None):
    """(poles, amplitudes) for one state, from its imaginary-axis data.

    The one entry point a caller needs: the poles are fitted here and are then
    the frozen object a gradient differentiates through.
    """
    d = ov_energies(eps, nocc)
    gap, top = float(d.min()), float(e_max if e_max is not None else d.max())
    poles = initial_poles(n_poles, gap, top)
    if relocate:
        poles = fit_poles(wc, nu_points, poles, stride=stride,
                          bounds=(gap, top))
    return poles, pole_amplitudes(wc, poles, nu_points)
