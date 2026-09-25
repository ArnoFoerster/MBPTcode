"""Non-radiative transfer rates from the quantities a surface layer produces.

    k = (2 pi / hbar) |V|^2 rho_FC

with V the electronic coupling between the two states -- a spin-orbit matrix
element <S1|H_SO|T1> for intersystem crossing, a transfer integral for electron
transfer -- and rho_FC the thermally averaged Franck-Condon weighted density of
states. Everything below is Fermi's golden rule with a different rho_FC:
Marcus-Levich-Jortner treats ONE effective quantum mode explicitly and the rest
of the bath classically, Marcus treats all of it classically.

This is the expression the TADF work has to produce (Samanta, Kim, Coropceanu
and Bredas, J. Am. Chem. Soc. 139, 4042 (2017); the quantum-mode form is
Jortner, J. Chem. Phys. 64, 4860 (1976)). The surface layer supplies four of the
five inputs -- the adiabatic gap from two relaxed minima, lambda by two
independent routes, the Huang-Rhys spectrum and its first-moment effective mode
-- and V is an argument, so the rate is one call the day a spin-orbit element
lands.

WHAT THE SCALE DEMANDS. Reverse intersystem crossing in a multiresonance emitter
runs at 10^6 s^-1 on a gap of 0.02 eV and a coupling of a few tenths of a cm^-1.
50 meV of error in the adiabatic gap is an order of magnitude in the rate, and
so is a factor of three in the coupling: these formulas are exponentially
sensitive to their inputs and are only meaningful on DIFFERENCES between states
of one molecule, where the method's absolute error cancels.

Every energy in and out of this module is in Hartree, the temperature in Kelvin
and the rate in s^-1.
"""
import warnings

import numpy as np

from src.Base.constants import (ATOMIC_TIME_SECONDS,
                                BOLTZMANN_HARTREE_PER_KELVIN,
                                SPEED_OF_LIGHT_AU)

#: numpy 2.0 RENAMED `trapz` to `trapezoid` and removed the old spelling. The
#: development environment here is numpy 1.23 and the cluster's is 2.x, so a
#: bare `np.trapz` runs everywhere it is tested and raises everywhere it is
#: used -- it took a five hour rate run to its last stage before failing.
TRAPEZOID = getattr(np, 'trapezoid', None) or np.trapz


def _thermal_energy(lambda_reorg, temperature):
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


def _golden_rule(coupling, rho_fc):
    """(2 pi / hbar) |V|^2 rho_FC in s^-1, from Hartree and Hartree^-1.

    hbar = 1 in atomic units, so the product is already a rate in inverse
    atomic time and only the unit of time has to be converted.
    """
    return (2.0 * np.pi * abs(coupling) ** 2 * rho_fc) / ATOMIC_TIME_SECONDS


def marcus_levich_jortner_rate(coupling, delta_e, lambda_m, s_eff, omega_eff,
                               temperature, n_max=50):
    """Marcus-Levich-Jortner rate in s^-1.

        rho_FC = sum_n e^-S S^n / n! (4 pi lambda_M k_B T)^-1/2
                 exp[ -(Delta_E + lambda_M + n hbar omega_eff)^2
                      / (4 lambda_M k_B T) ]

    Each term is the classical Marcus expression for a channel that also
    deposits n quanta of the effective high-frequency mode, weighted by the
    Poisson factor that mode's Huang-Rhys factor gives it. The quantum mode is
    what makes an activationless-looking process run at all in the inverted
    region, which classical Marcus cannot do.

    coupling: |V| in Hartree (a complex spin-orbit element is fine; only its
        modulus enters).
    delta_e: Delta_E = E_final - E_initial in Hartree, the reaction energy's own
        sign convention, so an exothermic step is NEGATIVE and the barrierless
        case is Delta_E + lambda_M = 0. For reverse intersystem crossing
        T1 -> S1 that makes Delta_E = +Delta_E_ST = E(S1) - E(T1) at the TRIPLET
        minimum, positive, which is why the process is thermally activated.
    lambda_m: the CLASSICAL reorganization energy, from the low-frequency modes
        only. The modes folded into (s_eff, omega_eff) must not be counted here
        as well -- double counting them lowers the barrier twice.
    s_eff, omega_eff: the effective quantum mode. `vibronic_analysis` reports
        both: S = sum_k S_k over the high-frequency modes and
        omega_eff = sum_k S_k omega_k / sum_k S_k, their Huang-Rhys-weighted
        first moment.
    n_max: quanta retained in the Poisson sum.
    """
    kt = _thermal_energy(lambda_m, temperature)
    if s_eff < 0.0 or omega_eff < 0.0:
        raise ValueError(f'the effective mode needs S >= 0 and omega >= 0, got '
                         f'S = {s_eff}, omega = {omega_eff} Hartree')
    n = np.arange(int(n_max) + 1)
    # e^-S S^n/n! by recursion rather than a gamma function, so that S = 0
    # collapses to the single classical channel exactly.
    poisson = np.empty(len(n))
    w = np.exp(-s_eff)
    for k in range(len(n)):
        poisson[k] = w
        w *= s_eff / (k + 1)
    # The Poisson mass is 1; a truncation that loses weight loses rate with it,
    # and at S of order n_max that is a factor rather than a rounding error.
    if 1.0 - poisson.sum() > np.sqrt(np.finfo(float).eps):
        raise ValueError(f'n_max = {n_max} keeps only {poisson.sum():.6f} of '
                         f'the Poisson weight at S = {s_eff}; raise n_max')
    activation = (float(delta_e) + lambda_m + n * omega_eff) ** 2
    rho_fc = ((poisson * np.exp(-activation / (4.0 * lambda_m * kt))).sum()
              / np.sqrt(4.0 * np.pi * lambda_m * kt))
    return _golden_rule(coupling, rho_fc)


def marcus_rate(coupling, delta_e, lambda_total, temperature):
    """Classical Marcus rate in s^-1: the S -> 0 limit of the expression above.

        k = (2 pi / hbar) |V|^2 (4 pi lambda k_B T)^-1/2
            exp[ -(Delta_E + lambda)^2 / (4 lambda k_B T) ]

    `lambda_total` is the WHOLE reorganization energy here, high-frequency modes
    included, because none of them is treated quantum mechanically. Same sign
    convention: Delta_E = E_final - E_initial.
    """
    kt = _thermal_energy(lambda_total, temperature)
    rho_fc = (np.exp(-(float(delta_e) + lambda_total) ** 2
                     / (4.0 * lambda_total * kt))
              / np.sqrt(4.0 * np.pi * lambda_total * kt))
    return _golden_rule(coupling, rho_fc)

def _bose(omega, kt):
    """n_bar(omega) = 1/(e^{omega/kT} - 1), elementwise."""
    x = np.asarray(omega, float) / kt
    out = np.empty_like(x)
    small = x < 1e-8
    out[~small] = 1.0 / np.expm1(x[~small])
    # A mode softer than 1e-8 kT is classical; its occupation is kT/omega.
    out[small] = kt / np.maximum(np.asarray(omega, float)[small], 1e-30)
    return out


def _saddle(delta_e, s_k, omega_k, n_k, lam_cl, gauss, kt):
    """tau at the stationary point of F(tau) = phi(-i tau) - dE tau, and F, F".

    F is CONVEX -- F'' is a sum of positive terms -- so F' is monotone
    increasing and a bisection on any bracket that straddles the root is both
    safe and sufficient. No scipy, no derivative-free search.
    """
    def dF(tau):
        return (float((s_k * omega_k * (-(n_k + 1.0) * np.exp(-omega_k * tau)
                                        + n_k * np.exp(omega_k * tau))).sum())
                - lam_cl + 2.0 * gauss * tau - delta_e)

    def F(tau):
        return (float((s_k * ((n_k + 1.0) * np.exp(-omega_k * tau)
                              + n_k * np.exp(omega_k * tau)
                              - (2.0 * n_k + 1.0))).sum())
                - lam_cl * tau + gauss * tau ** 2 - delta_e * tau)

    def d2F(tau):
        return (float((s_k * omega_k ** 2
                       * ((n_k + 1.0) * np.exp(-omega_k * tau)
                          + n_k * np.exp(omega_k * tau))).sum())
                + 2.0 * gauss)

    # Bracket. The exponentials cap how far the shift can go before overflow;
    # 0.5/omega_max keeps every exponent under ~1 per doubling.
    step = 0.5 / max(float(omega_k.max()), 1e-12)
    lo = hi = 0.0
    for _ in range(200):
        if dF(lo) <= 0.0 <= dF(hi):
            break
        if dF(hi) < 0.0:
            hi += step
        if dF(lo) > 0.0:
            lo -= step
    else:
        raise ValueError(
            f'no stationary point bracketed for dE = {delta_e:.6g} Ha. The gap '
            f'is far outside what this spectrum can absorb or emit; there is '
            f'no meaningful Franck-Condon density there.')
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dF(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    return tau, F(tau), d2F(tau)


def fc_weighted_dos(delta_e, s_k, omega_k, temperature, lambda_classical=0.0,
                    broadening=0.0, dt_per_period=24, t_decay=9.0,
                    max_points=1 << 20):
    """rho_FC by the time-domain generating function -- EVERY mode explicitly.

        rho_FC(dE) = (1/2 pi) int dt e^{-i dE t} G(t)
        G(t) = exp[ -i lam_cl t - (lam_cl kT + sig^2/2) t^2
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
    kt = _thermal_energy(1.0, temperature)          # reuse only the T guard
    n_k = _bose(omega_k, kt)
    lam_cl = float(lambda_classical)
    gauss = lam_cl * kt + 0.5 * float(broadening) ** 2
    if gauss <= 0.0 and not np.any(s_k > 0.0):
        raise ValueError(
            'nothing damps the time integral: the mode spectrum carries no '
            'displacement (all S_k = 0) and neither lambda_classical nor '
            'broadening was given. With no coupling there is no rate; with a '
            'bath, say which.')

    de = np.atleast_1d(np.asarray(delta_e, float))
    rho = np.empty(de.shape)
    for i, e in enumerate(de.ravel()):
        tau, f0, f2 = _saddle(float(e), s_k, omega_k, n_k, lam_cl, gauss, kt)
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
            phi = np.zeros_like(z)
            for sk, wk, nk in zip(s_k, omega_k, n_k):    # (n_t,) accumulator
                phi += sk * ((nk + 1.0) * np.exp(-1j * wk * z)
                             + nk * np.exp(1j * wk * z) - (2.0 * nk + 1.0))
            # exponent measured FROM the saddle: the integrand is O(1) there
            g = np.exp(phi - 1j * lam_cl * z - gauss * z ** 2
                       - 1j * float(e) * z - f0)
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
                f'bath -- give lambda_classical or broadening.', RuntimeWarning)
        rho.ravel()[i] = np.exp(f0) * value
    rho = np.maximum(rho, 0.0)          # a density of states is non-negative
    return rho if np.ndim(delta_e) else float(rho[0])


def spin_vibronic_rate(delta_e, v0, dv_dq, omega_promoting, rho_fc,
                       temperature):
    """Condon + Herzberg-Teller intersystem-crossing rate in s^-1.

        k = 2 pi [ |V_0|^2 rho(dE)
                   + sum_k (1/2)|dV/dq_k|^2 ( (n_k+1) rho(dE + w_k)
                                              + n_k rho(dE - w_k) ) ]

    The Condon term alone (Samanta, Kim, Coropceanu and Bredas, J. Am. Chem.
    Soc. 139, 4042 (2017)) is nearly zero for an El-Sayed-forbidden pair,
    pi-pi* against pi-pi*, and the derivative carries the process: the
    promoting mode lends the pair the symmetry the coupling needs. Its channel
    deposits one quantum of mode k, so the remaining bath absorbs dE + w_k and
    rho is evaluated at the shifted gap.

    q_k are DIMENSIONLESS normal coordinates, in which <1|q|0> = 1/sqrt 2 and
    the 1/2 above is that squared; `dv_dq` must be d|V|/dq in the same
    convention (`vibronic_soc.py` produces it). Both v0 and dv_dq are Hartree.

    rho_fc: callable dE -> Hartree^-1. Pass `fc_weighted_dos` bound to the
    spectrum, or a Marcus/MLJ rho if that is the comparison being made.

    Returns (k_total, k_condon, k_herzberg_teller) in s^-1, so the two can be
    reported separately -- which is the whole point of computing the second.
    """
    kt = _thermal_energy(1.0, temperature)
    dv = np.atleast_1d(np.asarray(dv_dq, float))
    w = np.atleast_1d(np.asarray(omega_promoting, float))
    if dv.shape != w.shape:
        raise ValueError(f'{dv.shape} derivatives against {w.shape} '
                         f'frequencies; one per promoting mode')
    n = _bose(w, kt)
    k_c = _golden_rule(v0, rho_fc(float(delta_e)))
    k_ht = 0.0
    for dvk, wk, nk in zip(dv, w, n):
        k_ht += _golden_rule(dvk * np.sqrt(0.5 * (nk + 1.0)),
                             rho_fc(float(delta_e) + float(wk)))
        k_ht += _golden_rule(dvk * np.sqrt(0.5 * nk),
                             rho_fc(float(delta_e) - float(wk)))
    return k_c + k_ht, k_c, k_ht


def internal_conversion_rate(delta_e, d_modes, omega, rho_fc, temperature):
    """Internal-conversion rate in s^-1 from the derivative coupling.

        k = 2 pi sum_k omega_k |d_k|^2 [ (1/2)(n_k+1) rho(dE + w_k)
                                         + (1/2) n_k   rho(dE - w_k) ]

    THE COUPLING IS NOT A SCALAR, which is the whole difference from
    intersystem crossing. The non-adiabatic operator is a nuclear DERIVATIVE,
    H' = -sum_k d_k d/dQ_k in mass-weighted coordinates, so every mode couples
    through its own momentum and the rate is a sum over channels rather than
    one |V|^2. With dimensionless q_k = sqrt(omega_k) Q_k the operator is
    -sum_k d_k sqrt(omega_k) d/dq_k, and <v+1|d/dq|v> = -sqrt((v+1)/2) has the
    same magnitude as <v+1|q|v> -- which is why this has exactly the shape of
    the Herzberg-Teller sum in `spin_vibronic_rate` and is evaluated by it,
    with no Condon term because there is none.

    d_modes: the coupling projected on the ground-state mass-weighted modes,
        `vibronic.project_coupling` of an (natm, 3) array in 1/Bohr.
    omega: those modes' frequencies in Hartree. Imaginary ones come back
        negative from `vibronic.normal_modes` and are dropped here; nothing in
        this model is meaningful at a saddle point.
    rho_fc: callable dE -> Hartree^-1, `fc_weighted_dos` bound to the
        Huang-Rhys spectrum. The promoting mode is NOT removed from the
        accepting bath, which is the usual approximation and is good while its
        own Huang-Rhys factor is small; at a mode that both promotes and
        accepts strongly this double counts one quantum.

    Returns (k_total, k_per_mode) so the channels that carry the process can be
    read off -- internal conversion is usually one or two high-frequency
    stretches, and a rate quoted without them says nothing about why.
    """
    d = np.atleast_1d(np.asarray(d_modes, float))
    w = np.atleast_1d(np.asarray(omega, float))
    if d.shape != w.shape:
        raise ValueError(f'{d.shape} couplings against {w.shape} frequencies; '
                         f'one per mode')
    real = w > 0
    k_per_mode = np.zeros_like(w)
    for k in np.flatnonzero(real):
        k_per_mode[k] = spin_vibronic_rate(delta_e, 0.0,
                                           [np.sqrt(w[k]) * d[k]], [w[k]],
                                           rho_fc, temperature)[0]
    return float(k_per_mode.sum()), k_per_mode


def radiative_rate(delta_e, dipole):
    """Spontaneous-emission rate in s^-1, the Einstein A coefficient.

        k_r = 4 omega^3 |mu|^2 / (3 c^3)

    the third electronic quantity, beside the spin-orbit element and the
    derivative coupling. `solve_bse_isdf` and `solve_bse_df` already return
    `transition_dipole` per root in the length gauge, so nothing new is
    computed here -- what this adds is the rate.

    EVALUATE IT AT THE RELAXED EXCITED-STATE GEOMETRY, with the EMISSION
    energy. The oscillator strength a vertical spectrum reports belongs to
    ABSORPTION: it is the dipole at the Franck-Condon point at the vertical
    energy, and k_r is cubic in that energy, so using it for emission
    overestimates the rate by (vertical/emission)^3 -- a factor of 1.5 at the
    0.4 eV relaxation typical of these molecules.

    delta_e: the emission energy in Hartree.
    dipole: <0|r|n> in atomic units, either the (3,) vector or its magnitude.

    Condon only. A weakly allowed transition -- which a charge-transfer TADF
    state usually is -- also emits through dmu/dQ, and that term is NOT here;
    it is the same Herzberg-Teller structure `spin_vibronic_rate` carries for
    intersystem crossing, and it needs the transition dipole's nuclear
    derivative.
    """
    mu2 = float(np.sum(np.asarray(dipole, float) ** 2))
    w = float(delta_e)
    if w <= 0:
        raise ValueError(f'emission energy {w:.6g} Ha is not positive; a '
                         'radiative rate needs the state above the one it '
                         'falls to')
    return (4.0 * w ** 3 * mu2 / (3.0 * SPEED_OF_LIGHT_AU ** 3)
            / ATOMIC_TIME_SECONDS)


def photoluminescence(k_r, k_isc, k_risc, k_nr_s=0.0, k_nr_t=0.0):
    """The observables a PL experiment actually reports, from the four rates.

    Two coupled levels, the singlet prepared by absorption and the triplet fed
    only by intersystem crossing:

        d[S]/dt = -(k_r + k_nr_s + k_isc) [S] + k_risc [T]
        d[T]/dt =  k_isc [S] - (k_nr_t + k_risc) [T]

    whose two decay constants are the PROMPT and DELAYED components of the
    fluorescence -- not k_r and not k_risc. Emission comes from the singlet in
    both, which is why delayed fluorescence has the prompt spectrum and a
    lifetime set by the triplet reservoir.

        lambda_+- = (k_S + k_T)/2 -+ sqrt((k_S - k_T)^2 + 4 k_isc k_risc)/2

    with k_S = k_r + k_nr_s + k_isc and k_T = k_nr_t + k_risc. The photon yield
    is k_r times the singlet's time integral, which the 2x2 inverse gives in
    closed form; splitting it over the two exponentials gives the prompt and
    delayed yields whose RATIO is what a time-resolved measurement reads off.

    Every rate in s^-1. Returns a dict: `k_prompt`, `k_delayed` (s^-1),
    `tau_prompt`, `tau_delayed` (s), `phi_prompt`, `phi_delayed`, `phi_total`.

    A triplet that neither decays nor returns (k_nr_t = k_risc = 0) traps every
    molecule that crosses, and the delayed component vanishes rather than
    diverging -- the branch is handled, not assumed away.
    """
    k_r, k_isc, k_risc = float(k_r), float(k_isc), float(k_risc)
    k_s = k_r + float(k_nr_s) + k_isc
    k_t = float(k_nr_t) + k_risc
    if min(k_r, k_isc, k_risc, k_nr_s, k_nr_t) < 0:
        raise ValueError('a rate cannot be negative')
    disc = np.sqrt((k_s - k_t) ** 2 + 4.0 * k_isc * k_risc)
    lam_p = 0.5 * (k_s + k_t + disc)
    det = k_s * k_t - k_isc * k_risc
    if det <= 0:
        # The triplet is a perfect trap: every molecule that crosses stays, so
        # only the prompt channel emits and the singlet integral is 1/k_S.
        return {'k_prompt': k_s, 'k_delayed': 0.0,
                'tau_prompt': 1.0 / k_s if k_s > 0 else float('inf'),
                'tau_delayed': float('inf'),
                'phi_prompt': k_r / k_s if k_s > 0 else 0.0,
                'phi_delayed': 0.0,
                'phi_total': k_r / k_s if k_s > 0 else 0.0}
    # THE SMALL ROOT COMES FROM THE PRODUCT, NOT FROM THE DIFFERENCE. The two
    # roots satisfy lam_+ lam_- = det exactly, and the subtracted form
    # 0.5 (k_S + k_T - disc) loses every significant digit once
    # k_isc k_risc << k_S^2 -- which is the ordinary case for a slow reverse
    # crossing. Measured on formaldehyde, where 4 k_isc k_risc / k_S^2 is
    # 1e-20: disc came back equal to k_S - k_T in double precision, lam_-
    # underflowed to exactly zero, and the reported delayed lifetime was
    # infinite with a yield of nan while the true lam_- was 7.9e-23 s^-1.
    lam_d = det / lam_p
    # THE DELAYED AMPLITUDE HAS THE SAME DISEASE ONE LEVEL UP. a_d = 1 - a_p
    # with a_p = (k_S - lam_-)/(lam_+ - lam_-) is 1 - (1 - 1e-21) in double
    # precision, so the delayed yield reads 0 and the two yields no longer sum
    # to the closed-form total. Both roots satisfy
    # (lambda - k_S)(lambda - k_T) = k_isc k_risc, which gives lam_+ - k_S
    # without subtracting two nearly equal numbers.
    gap = lam_p - k_t
    top = (k_isc * k_risc / gap) if gap > 0 else (lam_p - k_s)
    a_d = top / (lam_p - lam_d) if lam_p > lam_d else 0.0
    a_p = 1.0 - a_d
    return {'k_prompt': lam_p, 'k_delayed': lam_d,
            'tau_prompt': 1.0 / lam_p, 'tau_delayed': 1.0 / lam_d,
            'phi_prompt': k_r * a_p / lam_p,
            'phi_delayed': k_r * a_d / lam_d,
            'phi_total': k_r * k_t / det}
