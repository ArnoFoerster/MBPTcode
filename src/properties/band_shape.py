"""Vibronic absorption and emission band shapes from Huang-Rhys factors.

Displaced harmonic oscillators: both electronic states carry the ground state's
normal modes and frequencies, the excited one displaced along each mode by its
Huang-Rhys factor S_k (`vibronic.huang_rhys_from_gradient` or
`huang_rhys_from_displacement`). No Duschinsky rotation and no frequency change
between the states; within that model the band is exact at any temperature.

The Franck-Condon density is `rates.fc_weighted_dos`, the same generating
function the golden-rule rates use, evaluated at the electronic energy the
photon leaves to the nuclei:

    emission   S1 -> S0 + E:  FC_em(E)  = rho(E - E00)
    absorption S0 + E -> S1:  FC_abs(E) = rho(E00 - E)

so at T = 0 the emission is the Poisson progression e^-S S^m / m! at
E00 - m omega and the absorption its mirror image at E00 + m omega. Each has
unit area, mean E00 -+ lambda with lambda = sum_k S_k omega_k, and variance
sum_k S_k omega_k^2 coth(omega_k / 2kT) plus the broadening's.

CONVENTION OF THE SPECTRA (Condon approximation, constant transition dipole):

    absorption: the cross section,                  sigma(E) ~ E   FC_abs(E)
    emission:   the photon-emission rate per unit
                photon energy (Einstein A per line), I(E)     ~ E^3 FC_em(E)

each normalized to unit area on its axis. A spectrum recorded as energy flux
carries one more power of E; one recorded per unit wavelength carries the
Jacobian |dE/dlambda| = E^2 / hc, which is how `emission_per_nm` is formed.

Energies in Hartree, temperature in Kelvin, wavelengths in nm.
"""
import numpy as np

from src.Base.constants import (BOLTZMANN_HARTREE_PER_KELVIN,
                                GAUSSIAN_FWHM_PER_SIGMA,
                                HARTREE_WAVELENGTH_NM)
from src.properties.rates import TRAPEZOID, fc_weighted_dos


def gaussian_limit_fwhm(s_k, omega_k, temperature):
    """2 sqrt(2 ln 2 sum_k S_k omega_k^2 coth(omega_k / 2kT)), in Hartree.

    The band's width when it is Gaussian: exact for the second cumulant, and
    the whole FWHM only when many soft modes share the displacement so the
    higher cumulants vanish. A few stiff modes give a resolved progression
    whose FWHM this does not describe.
    """
    s_k, omega_k = np.asarray(s_k, float), np.asarray(omega_k, float)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)
    coth = 1.0 / np.tanh(omega_k / (2.0 * kt))
    return GAUSSIAN_FWHM_PER_SIGMA * np.sqrt(float((s_k * omega_k ** 2
                                                    * coth).sum()))


def fwhm(x, y):
    """Full width at half maximum of a band sampled on ascending x.

    The OUTERMOST half-maximum crossings, linearly interpolated: a progression
    whose second peak exceeds half the first is one band, and its width spans
    both. Refuses a band whose half maximum is not inside the grid.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    half = 0.5 * y.max()
    above = np.flatnonzero(y >= half)
    lo, hi = above[0], above[-1]
    if lo == 0 or hi == len(y) - 1:
        raise ValueError('the half maximum lies outside the grid; widen it')
    x_lo = x[lo - 1] + (half - y[lo - 1]) * (x[lo] - x[lo - 1]) / (y[lo] - y[lo - 1])
    x_hi = x[hi] + (half - y[hi]) * (x[hi + 1] - x[hi]) / (y[hi + 1] - y[hi])
    return float(x_hi - x_lo)


def peak(x, y):
    """Position of the maximum, refined by the parabola through its neighbours."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        raise ValueError('the maximum sits on the edge of the grid; widen it')
    a, b, _ = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    return float(-b / (2.0 * a)) if a < 0.0 else float(x[i])


def band_grid(s_k, omega_k, e00, temperature, *, gaussian_fwhm,
              lorentzian_fwhm, n_widths=8.0, points_per_width=20):
    """Uniform energy grid covering both bands, in Hartree.

    Spans E00 -+ (lambda + n_widths w) with w the band's standard deviation
    plus the Lorentzian FWHM, and resolves the narrowest of the band, the
    Gaussian and the Lorentzian widths by `points_per_width` points.
    """
    s_k, omega_k = np.asarray(s_k, float), np.asarray(omega_k, float)
    sigma_g = float(gaussian_fwhm) / GAUSSIAN_FWHM_PER_SIGMA
    sigma_band = np.hypot(gaussian_limit_fwhm(s_k, omega_k, temperature)
                          / GAUSSIAN_FWHM_PER_SIGMA, sigma_g)
    widths = [w for w in (sigma_band, sigma_g, 0.5 * float(lorentzian_fwhm))
              if w > 0.0]
    if not widths:
        raise ValueError('the band has no width: no displacement and no '
                         'broadening')
    half_span = (float((s_k * omega_k).sum())
                 + n_widths * (sigma_band + float(lorentzian_fwhm)))
    lo = float(e00) - half_span
    if lo <= 0.0:
        raise ValueError(f'the band reaches zero photon energy (E00 = {e00:.4g} '
                         f'Ha, half span {half_span:.4g} Ha); pass a grid')
    n = int(np.ceil((float(e00) + half_span - lo) * points_per_width
                    / min(widths))) + 1
    return np.linspace(lo, float(e00) + half_span, n)


def band_shape(s_k, omega_k, e00, temperature, energies, *, gaussian_fwhm,
               lorentzian_fwhm):
    """Normalized absorption and emission spectra and their widths.

    s_k, omega_k: Huang-Rhys factors and frequencies (Hartree) of the real
        modes; rigid-body and imaginary ones must be dropped.
    e00: the 0-0 energy, the adiabatic gap between the two minima (zero-point
        energies cancel in the equal-frequency model).
    energies: ascending photon energies, Hartree, all positive (`band_grid`).
    gaussian_fwhm: inhomogeneous width, FWHM in Hartree; 0 for none.
    lorentzian_fwhm: homogeneous width, FWHM in Hartree; 0 for none.
        Both are required: a computed width is only comparable with a measured
        one when the broadening added to it is stated.

    Returns a dict: `energies`; `fc_absorption`, `fc_emission`, the
    Franck-Condon densities in Hartree^-1; `absorption`, `emission`, the
    normalized spectra of the module's convention; their `peak_*` and
    `fwhm_*` in Hartree; `stokes_shift`, peak absorption minus peak emission;
    `wavelengths_nm`, ascending, with `emission_per_nm` normalized on that axis
    and its `peak_emission_nm` and `fwhm_emission_nm`; `reorganization`,
    lambda = sum_k S_k omega_k.
    """
    e = np.asarray(energies, float)
    if np.any(e <= 0.0) or np.any(np.diff(e) <= 0.0):
        raise ValueError('photon energies must be positive and ascending')
    if gaussian_fwhm < 0.0 or lorentzian_fwhm < 0.0:
        raise ValueError('a broadening FWHM cannot be negative')
    widths = dict(broadening=float(gaussian_fwhm) / GAUSSIAN_FWHM_PER_SIGMA,
                  lorentzian=0.5 * float(lorentzian_fwhm))
    fc_abs = fc_weighted_dos(float(e00) - e, s_k, omega_k, temperature,
                             **widths)
    fc_em = fc_weighted_dos(e - float(e00), s_k, omega_k, temperature,
                            **widths)
    absorption = e * fc_abs
    absorption /= TRAPEZOID(absorption, e)
    emission = e ** 3 * fc_em
    emission /= TRAPEZOID(emission, e)

    # per unit wavelength: I(lambda) = I(E) |dE/dlambda| = I(E) E^2 / hc
    wavelengths = HARTREE_WAVELENGTH_NM / e[::-1]
    per_nm = (emission * e ** 2)[::-1]
    per_nm /= TRAPEZOID(per_nm, wavelengths)

    peak_abs, peak_em = peak(e, absorption), peak(e, emission)
    return {'energies': e, 'fc_absorption': fc_abs, 'fc_emission': fc_em,
            'absorption': absorption, 'emission': emission,
            'peak_absorption': peak_abs, 'peak_emission': peak_em,
            'fwhm_absorption': fwhm(e, absorption),
            'fwhm_emission': fwhm(e, emission),
            'stokes_shift': peak_abs - peak_em,
            'wavelengths_nm': wavelengths, 'emission_per_nm': per_nm,
            'peak_emission_nm': peak(wavelengths, per_nm),
            'fwhm_emission_nm': fwhm(wavelengths, per_nm),
            'reorganization': float((np.asarray(s_k, float)
                                     * np.asarray(omega_k, float)).sum())}
