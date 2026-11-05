"""The Franck-Condon weighted density of states of displaced harmonic
oscillators, and the checks its vibrational modes pass first.

`fc_weighted_dos` is the thermally averaged Franck-Condon density a golden-rule
rate (`src.photophysics.rates`) multiplies by the squared coupling, and the
generating function behind it gives the vibronic band shapes
(`src.properties.band_shape`). Its integrand that has not decayed by the end
of its time grid warns (`FCIntegrandNotDecayed`) and is recorded.
"""
import warnings
from dataclasses import dataclass

import numpy as np

from src.Base.numerics.constants import (BOLTZMANN_HARTREE_PER_KELVIN,
                                         FC_DECAYED_FRACTION, FC_DT_PER_PERIOD,
                                         FC_MAX_POINTS, FC_MIN_POINTS,
                                         FC_T_DECAY, FC_UNDERFLOW_EXPONENT,
                                         FC_WINDOWS, HARTREE_TO_CM,
                                         VIBRONIC_MODE_FLOOR_CM)


#: `np.trapezoid` on numpy >= 2.0, `np.trapz` below it.
TRAPEZOID = getattr(np, 'trapezoid', None) or np.trapz


def require_vibrational_modes(omega, where):
    """Refuse normal-mode frequencies (Hartree) that are not all
    vibrations of at least VIBRONIC_MODE_FLOOR_CM: an imaginary mode
    (negative from `vibronic.normal_modes`) is a saddle, where no
    displaced-oscillator quantity is defined, and a mode below the floor is a
    rigid-body remnant or a flat direction, whose Huang-Rhys factor has no
    bound. Dropping either would change the sum rule unseen."""
    w = np.atleast_1d(np.asarray(omega, float))
    low = w * HARTREE_TO_CM < VIBRONIC_MODE_FLOOR_CM
    if low.any():
        raise ValueError(
            f'{where}: modes {np.flatnonzero(low).tolist()} at '
            f'{np.round(w[low] * HARTREE_TO_CM, 2).tolist()} cm^-1 (negative = '
            f'imaginary) are below VIBRONIC_MODE_FLOOR_CM = '
            f'{VIBRONIC_MODE_FLOOR_CM} cm^-1: relax to a minimum, and pass the '
            f'vibrations only (rigid-body modes projected out)')


def thermal_energy(lambda_reorg, temperature):
    """k_B T in Hartree, with the guards the Gaussian bath needs."""
    if lambda_reorg <= 0.0:
        raise ValueError(f'the classical reorganization energy must be '
                         f'positive, got {lambda_reorg} Hartree; a bath with '
                         f'no width has a delta-function density of states and '
                         f'no golden-rule rate')
    if temperature <= 0.0:
        raise ValueError(f'temperature must be positive, got {temperature} K; '
                         f'the classical bath is thermally averaged and has no '
                         f'zero-temperature limit')
    return BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)


def bose(omega, kt):
    """n_bar(omega) = 1/(e^{omega/kT} - 1), elementwise."""
    x = np.asarray(omega, float) / kt
    out = np.empty_like(x)
    small = x < 1e-8
    with np.errstate(over='ignore'):        # omega >> kT: 1/inf is n = 0
        out[~small] = 1.0 / np.expm1(x[~small])
    # A mode softer than 1e-8 kT is classical; its occupation is kT/omega.
    out[small] = kt / np.maximum(np.asarray(omega, float)[small], 1e-30)
    return out


def _saddle(delta_e, s_k, omega_k, n_k, lam_cl, gauss):
    """tau at the stationary point of F(tau) = phi(-i tau) - dE tau, and F, F".

    F is CONVEX -- F'' is a sum of positive terms -- so F' is monotone
    increasing and a bisection on any bracket that straddles the root is both
    safe and sufficient. No scipy, no derivative-free search.

    Returns tau = None when F falls below FC_UNDERFLOW_EXPONENT on the way to
    its minimum: rho = e^F x (bell) is then zero in double precision -- the far
    side of a band at low temperature, where only a Gaussian tail reaches.
    """
    with np.errstate(divide='ignore'):
        log_n = np.log(n_k)         # -inf for a frozen mode, so never 0 x inf

    def hot(tau):
        return np.exp(log_n + omega_k * tau)

    def dF(tau):
        return (float((s_k * omega_k * (-(n_k + 1.0) * np.exp(-omega_k * tau)
                                        + hot(tau))).sum())
                - lam_cl + 2.0 * gauss * tau - delta_e)

    def F(tau):
        return (float((s_k * ((n_k + 1.0) * np.exp(-omega_k * tau)
                              + hot(tau) - (2.0 * n_k + 1.0))).sum())
                - lam_cl * tau + gauss * tau ** 2 - delta_e * tau)

    def d2F(tau):
        return (float((s_k * omega_k ** 2
                       * ((n_k + 1.0) * np.exp(-omega_k * tau)
                          + hot(tau))).sum())
                + 2.0 * gauss)

    # Bracket. The exponentials cap how far the shift can go before overflow;
    # 0.5/omega_max keeps every exponent under ~1 per doubling. Past 200 steps
    # the root sits in a Gaussian-dominated tail and the step grows
    # geometrically; F convex and still falling means its minimum is lower than
    # where the bracket stands.
    step = 0.5 / max(float(omega_k.max()), 1e-12)
    lo = hi = 0.0
    with np.errstate(over='ignore'):
        for it in range(400):
            if dF(lo) <= 0.0 <= dF(hi):
                break
            if min(F(lo), F(hi)) < FC_UNDERFLOW_EXPONENT:
                return None, -np.inf, 0.0
            if dF(hi) < 0.0:
                hi += step
            if dF(lo) > 0.0:
                lo -= step
            if it >= 200:
                step *= 2.0
        else:
            raise ValueError(
                f'no stationary point bracketed for dE = {delta_e:.6g} Ha. The '
                f'gap is far outside what this spectrum can absorb or emit; '
                f'there is no meaningful Franck-Condon density there.')
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if dF(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
        f_min = F(tau)
    if f_min < FC_UNDERFLOW_EXPONENT:
        return None, -np.inf, 0.0
    return tau, f_min, d2F(tau)


def log_generating_function(z, s_k, omega_k, n_k, log_n, lam_cl, gauss):
    """ln G(z) of `fc_weighted_dos` without the Lorentzian, at complex times z."""
    z = np.asarray(z, complex)
    phi = np.zeros_like(z)
    for sk, wk, nk, lnk in zip(s_k, omega_k, n_k, log_n):
        # n_k e^{i w_k z} as one exponential: e^{w_k tau} may overflow where a
        # frozen mode's n_k = 0
        phi += sk * ((nk + 1.0) * np.exp(-1j * wk * z)
                     + np.exp(lnk + 1j * wk * z) - (2.0 * nk + 1.0))
    return phi - 1j * lam_cl * z - gauss * z ** 2


class FCIntegrandNotDecayed(RuntimeWarning):
    """A Franck-Condon time integrand that had not decayed at the longest
    time `fc_weighted_dos` integrates to: rho may be truncated. The rate
    table names the densities it happened on ('fc_integrand_not_decayed' in
    its `numerical_flags`)."""


@dataclass(frozen=True)
class FCTimeGrid:
    """The one set of knobs of a real-time Franck-Condon integral
    (`fc_weighted_dos`, `duschinsky.duschinsky_density`): points per period
    of the fastest phase, the first window in widths of the band, and the
    cap on a grid's points (a larger grid is refused)."""
    dt_per_period: int = FC_DT_PER_PERIOD
    t_decay: float = FC_T_DECAY
    max_points: int = FC_MAX_POINTS


FC_TIME_GRID = FCTimeGrid()


def decayed(edge, g):
    """Whether |G| at a window's ends, `edge`, is below FC_DECAYED_FRACTION
    of the largest |G| on its grid `g`."""
    return edge <= FC_DECAYED_FRACTION * max(float(np.abs(g).max()), 1e-300)


def grown(integrate, t_max):
    """(result, t_max, decayed) of `integrate(t_max) -> (result, edge, g)`
    on windows from `t_max`, each twice the last, FC_WINDOWS at most,
    stopping at the first whose ends have decayed (`decayed`): a recurring
    integrand (one quantum mode recurs at multiples of 2 pi / omega) is
    grown past its recurrences, not sized from its first bell."""
    for k in range(FC_WINDOWS):
        if k:
            t_max *= 2.0
        result, edge, g = integrate(t_max)
        if decayed(edge, g):
            return result, t_max, True
    return result, t_max, False


def envelope_time(gauss, gamma):
    """The time after which the broadening alone keeps |G| below
    FC_DECAYED_FRACTION of G(0): |G| / G(0) <= 1 (a characteristic
    function), so e^{-gauss t^2} or e^{-gamma t} bounds it, whichever is
    given (inf with neither)."""
    decay = -np.log(FC_DECAYED_FRACTION)
    return min(np.sqrt(decay / gauss) if gauss > 0.0 else np.inf,
               decay / gamma if gamma > 0.0 else np.inf)


def real_axis_density(de, log_g, variance, lam_total, gamma, omega_max,
                       time_grid, t_envelope):
    """rho(dE) for every dE off ONE real-axis grid, with e^{-gamma |t|}.

    |G(t)| does not depend on dE, so neither does the window; G is Hermitian,
    G(-t) = G(t)*, so rho = (1/pi) Re int_0^inf e^{-i dE t} G(t) dt.

    THE GRID REACHES THE BROADENING'S ENVELOPE, `t_envelope`
    (`envelope_time`), where it is finite. A strongly displaced fast mode
    makes |G| dip below FC_DECAYED_FRACTION and revive one period later, so
    a decayed last point alone proves nothing. `time_grid` is the
    `FCTimeGrid`.
    """
    dt_per_period = time_grid.dt_per_period
    # THE STEP RESOLVES THE BAND, not only the fastest mode. The trapezoid
    # reads rho(dE) plus its aliases rho(dE +- 2 pi m / h), so 2 pi / h must
    # reach past the band's far edge, |lambda| + 12 standard deviations
    # beyond dE; a strongly displaced fast mode spreads it over many quanta
    # of that mode. Rounded up to a power of two, so that two routes to one
    # band (closed-form and numerical cumulants) take one grid.
    reach = (float(np.abs(de).max()) + abs(float(lam_total))
             + 12.0 * np.sqrt(variance))
    reach = 2.0 ** np.ceil(np.log2(max(reach, 1e-12)))
    fastest = max(omega_max, float(np.abs(de).max()), reach / dt_per_period,
                  1e-12)
    t_max = time_grid.t_decay / np.sqrt(variance + gamma ** 2)
    if np.isfinite(t_envelope):
        t_max = max(t_max, float(t_envelope))

    def integrate(t_max):
        # [0, t_max] of the [-t_max, t_max] grid G(-t) = G(t)* mirrors
        n_t = max(int(np.ceil(t_max * fastest * dt_per_period
                              / (2.0 * np.pi))) + 1, (FC_MIN_POINTS + 1) // 2)
        if 2 * n_t - 1 > time_grid.max_points:
            raise ValueError(
                f'the real-axis grid needs {2 * n_t - 1} points (t_max '
                f'{t_max:.3g} a.u., fastest phase {fastest:.3g} Ha) against a '
                f'cap of {time_grid.max_points}. Raise max_points, or widen '
                f'the Lorentzian -- a narrower line needs more time.')
        t = np.linspace(0.0, t_max, n_t)
        g = np.exp(log_g(t) - gamma * t)
        return (t, g), abs(g[-1]), g

    (t, g), t_max, done = grown(integrate, t_max)
    if not done:
        warnings.warn(f'the Franck-Condon integrand had not decayed at t_max '
                      f'= {t_max:.3g} a.u.; rho may be truncated.',
                      FCIntegrandNotDecayed)
    h = t[1] - t[0]
    gw = g * h
    gw[0] *= 0.5
    gw[-1] *= 0.5
    flat = de.ravel()
    rho = np.array([2.0 * float(np.real(np.exp(-1j * e * t) @ gw))
                    for e in flat])
    # Euler-Maclaurin at the kink of g = A e^{-gamma |t|}, A(0) = 1: g' jumps
    # by -2 gamma and g''' by -6 gamma A''(0) - 2 gamma^3, with
    # A''(0) = -(variance + (lambda + dE)^2); the trapezoid is left at O(h^6)
    a2 = -(variance + (lam_total + flat) ** 2)
    rho += (-h ** 2 * gamma / 6.0
            + h ** 4 * (6.0 * gamma * a2 + 2.0 * gamma ** 3) / 720.0)
    return (rho / (2.0 * np.pi)).reshape(de.shape)


def fc_weighted_dos(delta_e, s_k, omega_k, temperature, lambda_classical=0.0,
                    broadening=0.0, lorentzian=0.0, time_grid=FC_TIME_GRID):
    """rho_FC by the time-domain generating function -- EVERY mode explicitly.

        rho_FC(dE) = (1/2 pi) int dt e^{-i dE t} G(t)
        G(t) = exp[ -i lam_cl t - (lam_cl kT + sig^2/2) t^2 - gamma |t|
                    + sum_k S_k( (n_k+1) e^{-i w_k t} + n_k e^{+i w_k t}
                                 - (2 n_k + 1) ) ]

    Every mode enters at its own frequency with its own Bose factor, so
    unlike `marcus_levich_jortner_rate` there is no effective mode, no cutoff
    and no classical/quantum split: a mode with omega << kT becomes classical
    through n_k. `lambda_classical` is for what the mode list does not contain,
    the solvent or matrix, and is zero for a gas-phase molecule.

    The contour is shifted to t = t' - i tau through the stationary point of
    F(tau) = phi(-i tau) - dE tau. Intersystem crossing is deeply inverted,
    |dE| far above the reorganization energy, where rho_FC is 1e-20 or smaller;
    on the real axis that asks double precision for twenty digits of
    cancellation between oscillations of order one and returns the roundoff
    floor. On the shifted contour the exponential smallness is the real
    prefactor e^{F(tau)} and the integrand is a bell of width 1/sqrt(F'').

    broadening: sig, the standard deviation of an inhomogeneous Gaussian.
    lorentzian: gamma, the half width at half maximum of a homogeneous
        Lorentzian. e^{-gamma |t|} is not analytic, so with gamma > 0 the
        integral stays on the real axis; that is safe because the Lorentzian's
        algebraic tail, gamma / (pi dE^2), keeps rho far above the roundoff
        floor that forces the shift for a Gaussian-damped deep tail.

    time_grid: the integral's `FCTimeGrid`.

    delta_e may be a scalar or an array. Returns rho_FC in Hartree^-1.
    """
    s_k = np.atleast_1d(np.asarray(s_k, float))
    omega_k = np.atleast_1d(np.asarray(omega_k, float))
    if s_k.shape != omega_k.shape:
        raise ValueError(f'S has shape {s_k.shape} and omega {omega_k.shape}; '
                         f'they are one spectrum and must match')
    if np.any(omega_k <= 0.0):
        raise ValueError('every mode needs omega > 0; drop the rigid-body and '
                         'imaginary ones before calling (a saddle point has no '
                         'Franck-Condon density of states)')
    if np.any(s_k < 0.0):
        raise ValueError('Huang-Rhys factors are squares and cannot be negative')
    if lambda_classical < 0.0:
        raise ValueError(f'lambda_classical = {lambda_classical} < 0')
    if lorentzian < 0.0:
        raise ValueError(f'the Lorentzian half width {lorentzian} is negative')
    kt = thermal_energy(1.0, temperature)          # reuse only the T guard
    n_k = bose(omega_k, kt)
    with np.errstate(divide='ignore'):
        log_n = np.log(n_k)
    lam_cl = float(lambda_classical)
    gauss = lam_cl * kt + 0.5 * float(broadening) ** 2
    gamma = float(lorentzian)
    if gauss <= 0.0 and gamma <= 0.0 and not np.any(s_k > 0.0):
        raise ValueError(
            'nothing damps the time integral: the mode spectrum carries no '
            'displacement (all S_k = 0) and neither lambda_classical nor '
            'broadening was given. With no coupling there is no rate; with a '
            'bath, say which.')

    def log_g(z):
        return log_generating_function(z, s_k, omega_k, n_k, log_n, lam_cl,
                                        gauss)

    de = np.atleast_1d(np.asarray(delta_e, float))
    if gamma > 0.0:
        # the first two cumulants of the band, for the kink correction
        variance = (float((s_k * omega_k ** 2 * (2.0 * n_k + 1.0)).sum())
                    + 2.0 * gauss)
        lam_total = float((s_k * omega_k).sum()) + lam_cl
        rho = real_axis_density(de, log_g, variance, lam_total, gamma,
                                 float(omega_k.max()), time_grid,
                                 envelope_time(gauss, gamma))
        rho = np.maximum(rho, 0.0)
        return rho if np.ndim(delta_e) else float(rho[0])

    rho = np.empty(de.shape)
    for i, e in enumerate(de.ravel()):
        tau, f0, f2 = _saddle(float(e), s_k, omega_k, n_k, lam_cl, gauss)
        if tau is None:
            rho.ravel()[i] = 0.0
            continue
        fastest = max(float(omega_k.max()), abs(float(e)), 1e-12)

        def integrate(t_max):
            n_t = int(np.ceil(2.0 * t_max * fastest * time_grid.dt_per_period
                              / (2.0 * np.pi))) + 1
            if n_t > time_grid.max_points:
                raise ValueError(
                    f'the shifted grid needs {n_t} points (t_max '
                    f'{t_max:.3g} a.u., fastest phase {fastest:.3g} Ha) '
                    f'against a cap of {time_grid.max_points}. Raise '
                    f'max_points, or add broadening -- a wider line needs '
                    f'less time.')
            t = np.linspace(-t_max, t_max, max(n_t, FC_MIN_POINTS))
            z = t - 1j * tau
            # exponent measured FROM the saddle: the integrand is O(1) there
            g = np.exp(log_g(z) - 1j * float(e) * z - f0)
            return (float(np.real(TRAPEZOID(g, t))) / (2.0 * np.pi),
                    max(abs(g[0]), abs(g[-1])), g)

        # Grown until the integrand has died at its ends, not sized from the
        # saddle curvature: 1/sqrt(F'') spans only the first recurrence of a
        # quantum mode. A many-mode spectrum dephases and stops after one pass.
        value, t_max, done = grown(integrate, time_grid.t_decay / np.sqrt(f2))
        if not done:
            warnings.warn(
                f'the Franck-Condon integrand had not decayed at t_max = '
                f'{t_max:.3g} a.u. for dE = {e:.6g} Ha; rho may be truncated. '
                f'A spectrum of one or two modes recurs forever without a '
                f'bath -- give lambda_classical or broadening.',
                FCIntegrandNotDecayed)
        rho.ravel()[i] = np.exp(f0) * value
    rho = np.maximum(rho, 0.0)          # a density of states is non-negative
    return rho if np.ndim(delta_e) else float(rho[0])
