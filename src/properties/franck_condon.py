"""The Franck-Condon weighted density of states of displaced harmonic
oscillators, and the checks its vibrational modes pass first.

`fc_weighted_dos` is the thermally averaged Franck-Condon density a golden-rule
rate (`src.photophysics.rates`) multiplies by the squared coupling, and the
generating function behind it gives the vibronic band shapes
(`src.properties.band_shape`). Its integrand that has not decayed by the end
of its time grid warns (`FCIntegrandNotDecayed`) and is recorded.
"""
import warnings

import numpy as np

from src.Base.numerics.constants import (BOLTZMANN_HARTREE_PER_KELVIN,
                                         FC_UNDERFLOW_EXPONENT, HARTREE_TO_CM,
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


def _saddle(delta_e, s_k, omega_k, n_k, lam_cl, gauss, kt):
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


def _log_generating_function(z, s_k, omega_k, n_k, log_n, lam_cl, gauss):
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


def _real_axis_density(de, log_g, variance, lam_total, gamma, omega_max,
                       dt_per_period, t_decay, max_points):
    """rho(dE) for every dE off ONE real-axis grid, with e^{-gamma |t|}.

    |G(t)| does not depend on dE, so neither does the window; G is Hermitian,
    G(-t) = G(t)*, so rho = (1/pi) Re int_0^inf e^{-i dE t} G(t) dt.
    """
    fastest = max(omega_max, float(np.abs(de).max()), 1e-12)
    t_max = t_decay / np.sqrt(variance + gamma ** 2)
    for _ in range(13):
        n_t = max(int(np.ceil(t_max * fastest * dt_per_period
                              / (2.0 * np.pi))) + 1, 17)
        if 2 * n_t - 1 > max_points:
            raise ValueError(
                f'the real-axis grid needs {2 * n_t - 1} points (t_max '
                f'{t_max:.3g} a.u., fastest phase {fastest:.3g} Ha) against a '
                f'cap of {max_points}. Raise max_points, or widen the '
                f'Lorentzian -- a narrower line needs more time.')
        t = np.linspace(0.0, t_max, n_t)
        g = np.exp(log_g(t) - gamma * t)
        if abs(g[-1]) <= 1e-10 * np.abs(g).max():
            break
        t_max *= 2.0
    else:
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
                    broadening=0.0, dt_per_period=24, t_decay=9.0,
                    max_points=1 << 20, lorentzian=0.0):
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
        return _log_generating_function(z, s_k, omega_k, n_k, log_n, lam_cl,
                                        gauss)

    de = np.atleast_1d(np.asarray(delta_e, float))
    if gamma > 0.0:
        # the first two cumulants of the band, for the kink correction
        variance = (float((s_k * omega_k ** 2 * (2.0 * n_k + 1.0)).sum())
                    + 2.0 * gauss)
        lam_total = float((s_k * omega_k).sum()) + lam_cl
        rho = _real_axis_density(de, log_g, variance, lam_total, gamma,
                                 float(omega_k.max()), dt_per_period, t_decay,
                                 max_points)
        rho = np.maximum(rho, 0.0)
        return rho if np.ndim(delta_e) else float(rho[0])

    rho = np.empty(de.shape)
    for i, e in enumerate(de.ravel()):
        tau, f0, f2 = _saddle(float(e), s_k, omega_k, n_k, lam_cl, gauss, kt)
        if tau is None:
            rho.ravel()[i] = 0.0
            continue
        fastest = max(float(omega_k.max()), abs(float(e)), 1e-12)

        def integrate(t_max):
            n_t = int(np.ceil(2.0 * t_max * fastest * dt_per_period
                              / (2.0 * np.pi))) + 1
            if n_t > max_points:
                raise ValueError(
                    f'the shifted grid needs {n_t} points (t_max '
                    f'{t_max:.3g} a.u., fastest phase {fastest:.3g} Ha) '
                    f'against a cap of {max_points}. Raise max_points, or add '
                    f'broadening -- a wider line needs less time.')
            t = np.linspace(-t_max, t_max, max(n_t, 33))
            z = t - 1j * tau
            # exponent measured FROM the saddle: the integrand is O(1) there
            g = np.exp(log_g(z) - 1j * float(e) * z - f0)
            return float(np.real(TRAPEZOID(g, t))) / (2.0 * np.pi), g

        # Grown until the integrand has died at its ends, not sized from the
        # saddle curvature: a single quantum mode makes G recur at multiples
        # of 2 pi / omega, one recurrence per vibrational channel, and
        # 1/sqrt(F'') spans only the first. A many-mode spectrum dephases and
        # stops after one pass.
        t_max = t_decay / np.sqrt(f2)
        value, g = integrate(t_max)
        for _ in range(12):
            edge = max(abs(g[0]), abs(g[-1]))
            if edge <= 1e-10 * max(np.abs(g).max(), 1e-300):
                break
            t_max *= 2.0
            value, g = integrate(t_max)
        else:
            warnings.warn(
                f'the Franck-Condon integrand had not decayed at t_max = '
                f'{t_max:.3g} a.u. for dE = {e:.6g} Ha; rho may be truncated. '
                f'A spectrum of one or two modes recurs forever without a '
                f'bath -- give lambda_classical or broadening.',
                FCIntegrandNotDecayed)
        rho.ravel()[i] = np.exp(f0) * value
    rho = np.maximum(rho, 0.0)          # a density of states is non-negative
    return rho if np.ndim(delta_e) else float(rho[0])
