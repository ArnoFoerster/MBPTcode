"""Gates for src/properties/band_shape.py and the line-shape branch of
`rates.fc_weighted_dos`.

Every reference is independent of the generating function: the Poisson
progression convolved with scipy's Voigt profile, the displaced-oscillator
cumulants in closed form, the Gaussian width of the second cumulant, and a
change of variables for the wavelength axis. No electronic structure.
"""
import os
import sys
from math import comb

import numpy as np
import pytest
from scipy.special import voigt_profile
from scipy.stats import norm, poisson

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (BOLTZMANN_HARTREE_PER_KELVIN,
                                GAUSSIAN_FWHM_PER_SIGMA, HARTREE_TO_CM,
                                HARTREE_TO_EV, HARTREE_WAVELENGTH_NM)
from src.properties.band_shape import (band_grid, band_shape, fwhm,
                                       gaussian_limit_fwhm, peak)
from src.properties.rates import TRAPEZOID, fc_weighted_dos

EV = 1.0 / HARTREE_TO_EV
CM = 1.0 / HARTREE_TO_CM
E00 = 3.0 * EV
# one stiff mode, cold: n = exp(-1500 cm^-1 / k_B 1 K) underflows to exactly 0
STIFF = 1500.0 * CM
COLD = 1.0
# a small multimode spectrum at room temperature
MODES = np.array([180.0, 420.0, 760.0, 1240.0, 1610.0]) * CM
S_MODES = np.array([0.6, 0.35, 0.2, 0.45, 0.3])
ROOM = 300.0


def poisson_voigt(x, s, omega, sigma, gamma, n_max=60):
    """sum_m e^-S S^m/m! V(x - m omega): the T = 0 band, by sum over states."""
    m = np.arange(n_max)
    return sum(p * voigt_profile(x - k * omega, sigma, gamma)
               for p, k in zip(poisson.pmf(m, s), m))


def raw_moments(e00, sign, s_k, omega_k, temperature, sigma, k_max=4):
    """<E^k> of E = E00 + sign x, x the displaced-oscillator energy transfer.

    Cumulants of x: kappa_r = sum_k S_k omega_k^r (2 n_k + 1) for even r and
    sum_k S_k omega_k^r for odd r, plus the Gaussian broadening in kappa_2.
    """
    kt = BOLTZMANN_HARTREE_PER_KELVIN * temperature
    coth = 1.0 / np.tanh(omega_k / (2.0 * kt))
    k1 = (s_k * omega_k).sum()
    k2 = (s_k * omega_k ** 2 * coth).sum() + sigma ** 2
    k3 = (s_k * omega_k ** 3).sum()
    k4 = (s_k * omega_k ** 4 * coth).sum()
    m = [1.0, k1, k2 + k1 ** 2, k3 + 3 * k2 * k1 + k1 ** 3,
         k4 + 4 * k3 * k1 + 3 * k2 ** 2 + 6 * k2 * k1 ** 2 + k1 ** 4]
    return [sum(comb(k, j) * e00 ** (k - j) * sign ** j * m[j]
                for j in range(k + 1)) for k in range(k_max + 1)]


def spectrum(s_k, omega_k, temperature, gaussian_fwhm, lorentzian_fwhm,
             points_per_width=6, n_widths=8.0):
    e = band_grid(s_k, omega_k, E00, temperature, gaussian_fwhm=gaussian_fwhm,
                  lorentzian_fwhm=lorentzian_fwhm,
                  points_per_width=points_per_width, n_widths=n_widths)
    return band_shape(s_k, omega_k, E00, temperature, e,
                      gaussian_fwhm=gaussian_fwhm,
                      lorentzian_fwhm=lorentzian_fwhm)


# ------------------------------------------------- T = 0: the Poisson progression
@pytest.mark.parametrize('g_fwhm, l_fwhm, tol', [(0.05, 0.0, 1e-7),
                                                 (0.0, 0.04, 1e-6),
                                                 (0.05, 0.04, 1e-6)])
def test_single_cold_mode_is_the_poisson_progression(g_fwhm, l_fwhm, tol):
    """Gaussian (shifted contour), Lorentzian and Voigt (real axis) widths."""
    s = 1.3
    r = spectrum([s], [STIFF], COLD, g_fwhm * EV, l_fwhm * EV,
                 points_per_width=3)
    e = r['energies']
    sigma, gamma = g_fwhm * EV / GAUSSIAN_FWHM_PER_SIGMA, 0.5 * l_fwhm * EV
    ref_em = poisson_voigt(E00 - e, s, STIFF, sigma, gamma)
    ref_abs = poisson_voigt(e - E00, s, STIFF, sigma, gamma)
    scale = ref_em.max()
    assert np.abs(r['fc_emission'] - ref_em).max() < tol * scale
    assert np.abs(r['fc_absorption'] - ref_abs).max() < tol * scale


def test_far_tail_of_a_cold_band_is_zero_not_a_refusal():
    """Where only the Gaussian reaches, rho is its tail, then exactly 0."""
    s, sigma = 1.3, 0.02 * EV
    near = np.array([20.0, 30.0]) * sigma        # still a normal double
    rho = fc_weighted_dos(near, [s], [STIFF], COLD, broadening=sigma)
    ref = np.exp(-s) * norm.pdf(near, scale=sigma)
    assert np.allclose(rho, ref, rtol=1e-8, atol=0.0)
    assert fc_weighted_dos(60.0 * sigma, [s], [STIFF], COLD,
                           broadening=sigma) == 0.0


# ---------------------------------------------------------------------- moments
def test_first_moments_and_the_prefactors():
    """<E> of FC_abs, FC_em is E00 +- lambda; E and E^3 weights give m2/m1, m4/m3."""
    sig_fwhm = 0.03 * EV
    # 12 widths: the multi-quantum tail of a stiff mode is heavier than a
    # Gaussian's, and 8 widths leave 6e-8 of the weight outside
    r = spectrum(S_MODES, MODES, ROOM, sig_fwhm, 0.0, points_per_width=3,
                 n_widths=12.0)
    e = r['energies']
    lam = r['reorganization']
    assert lam == pytest.approx((S_MODES * MODES).sum(), rel=1e-14)
    sigma = sig_fwhm / GAUSSIAN_FWHM_PER_SIGMA
    for key, sign in (('fc_absorption', 1.0), ('fc_emission', -1.0)):
        rho = r[key]
        assert TRAPEZOID(rho, e) == pytest.approx(1.0, abs=1e-9)
        mean = TRAPEZOID(e * rho, e)
        assert mean - E00 == pytest.approx(sign * lam, rel=1e-8)
        m = raw_moments(E00, sign, S_MODES, MODES, ROOM, sigma)
        assert TRAPEZOID(e ** 2 * rho, e) == pytest.approx(m[2], rel=1e-9)
    m_abs = raw_moments(E00, 1.0, S_MODES, MODES, ROOM, sigma)
    m_em = raw_moments(E00, -1.0, S_MODES, MODES, ROOM, sigma)
    assert TRAPEZOID(e * r['absorption'], e) == pytest.approx(
        m_abs[2] / m_abs[1], rel=1e-9)
    assert TRAPEZOID(e * r['emission'], e) == pytest.approx(
        m_em[4] / m_em[3], rel=1e-9)


def test_many_soft_modes_reach_the_gaussian_limit():
    """FWHM -> 2 sqrt(2 ln2 sum S w^2 coth), peak separation -> 2 lambda.

    lambda is held at 0.2 eV and the modes are made softer against kT, which
    is what removes the third cumulant; the error must fall with them
    (measured 9.7e-3, 7.5e-4, 4.9e-5 in the width and 3.3e-2, 2.2e-3, 1.5e-4
    in the separation). Stiffer than 300 cm^-1 the band is a progression and
    its FWHM no longer tracks the second cumulant.
    """
    lam = 0.2 * EV
    errors = []
    for w_max in (300.0, 75.0, 20.0):
        w = np.linspace(0.25, 1.0, 40) * w_max * CM
        s = np.full(w.shape, lam / (len(w) * w.mean()))
        # a Gaussian a fifth of the band's width ends the recurrences of a
        # commensurate mode set; it adds to the second cumulant and nothing else
        g = 0.2 * gaussian_limit_fwhm(s, w, ROOM)
        r = spectrum(s, w, ROOM, g, 0.0)
        e = r['energies']
        target = np.hypot(gaussian_limit_fwhm(s, w, ROOM), g)
        err_w = abs(fwhm(e, r['fc_emission']) / target - 1.0)
        assert fwhm(e, r['fc_absorption']) == pytest.approx(
            fwhm(e, r['fc_emission']), rel=1e-9)
        sep = peak(e, r['fc_absorption']) - peak(e, r['fc_emission'])
        errors.append((err_w, abs(sep / (2.0 * lam) - 1.0)))
    assert errors[-1][0] < 2e-4 and errors[-1][1] < 5e-4
    for coarse, fine in zip(errors, errors[1:]):
        assert fine[0] < coarse[0] and fine[1] < coarse[1]


# ----------------------------------------------------- normalization and axes
def test_spectra_are_normalized_on_their_axes():
    r = spectrum(S_MODES, MODES, ROOM, 0.03 * EV, 0.01 * EV)
    e = r['energies']
    assert TRAPEZOID(r['absorption'], e) == pytest.approx(1.0, rel=1e-12)
    assert TRAPEZOID(r['emission'], e) == pytest.approx(1.0, rel=1e-12)
    assert TRAPEZOID(r['emission_per_nm'], r['wavelengths_nm']) == \
        pytest.approx(1.0, rel=1e-12)
    assert np.all(np.diff(r['wavelengths_nm']) > 0)


def test_wavelength_axis_is_a_change_of_variables():
    """<lambda> per nm equals hc <1/E> per energy only with the Jacobian."""
    r = spectrum(S_MODES, MODES, ROOM, 0.03 * EV, 0.0, points_per_width=8)
    e, lam = r['energies'], r['wavelengths_nm']
    by_nm = TRAPEZOID(lam * r['emission_per_nm'], lam)
    by_energy = HARTREE_WAVELENGTH_NM * TRAPEZOID(r['emission'] / e, e)
    assert by_nm == pytest.approx(by_energy, rel=1e-6)
    assert HARTREE_WAVELENGTH_NM * HARTREE_TO_EV == pytest.approx(1239.84198,
                                                                  rel=1e-8)


# ------------------------------------------------- agreement with the rate route
def test_band_shape_reads_the_rate_density_at_the_photon_gaps():
    """FC_em(E) = rho(E - E00), FC_abs(E) = rho(E00 - E), the golden-rule rho."""
    g, l = 0.03 * EV, 0.01 * EV
    r = spectrum(S_MODES, MODES, ROOM, g, l, points_per_width=2)
    e = r['energies']
    kw = dict(broadening=g / GAUSSIAN_FWHM_PER_SIGMA, lorentzian=0.5 * l)
    assert np.array_equal(r['fc_emission'],
                          fc_weighted_dos(e - E00, S_MODES, MODES, ROOM, **kw))
    assert np.array_equal(r['fc_absorption'],
                          fc_weighted_dos(E00 - e, S_MODES, MODES, ROOM, **kw))
    # the two are different functions: the band is not symmetric about E00
    assert np.abs(r['fc_emission'] - r['fc_absorption']).max() > \
        0.1 * r['fc_emission'].max()


def test_real_axis_branch_meets_the_shifted_contour():
    """A vanishing Lorentzian on the real axis = the saddle-shifted Gaussian."""
    sigma = 0.03 * EV / GAUSSIAN_FWHM_PER_SIGMA
    de = np.linspace(-0.6, 0.2, 9) * EV
    contour = fc_weighted_dos(de, S_MODES, MODES, ROOM, broadening=sigma)
    real_axis = fc_weighted_dos(de, S_MODES, MODES, ROOM, broadening=sigma,
                                lorentzian=1e-6 * sigma)
    assert np.abs(real_axis - contour).max() < 1e-5 * contour.max()


# --------------------------------------------------------------- the widths
@pytest.mark.parametrize('g_fwhm, l_fwhm', [(0.04, 0.0), (0.0, 0.04),
                                            (0.03, 0.02)])
def test_fwhm_of_a_bare_line_is_the_stated_broadening(g_fwhm, l_fwhm):
    """S = 0: the FC density is the broadening itself, Voigt FWHM by scipy."""
    g, l = g_fwhm * EV, l_fwhm * EV
    r = spectrum([0.0], [STIFF], ROOM, g, l, points_per_width=40)
    e = r['energies']
    x = np.linspace(-0.3, 0.3, 600001) * EV
    exact = fwhm(x, voigt_profile(x, g / GAUSSIAN_FWHM_PER_SIGMA, 0.5 * l))
    assert fwhm(e, r['fc_emission']) == pytest.approx(exact, rel=2e-4)
    if l_fwhm == 0.0:
        assert exact == pytest.approx(g, rel=1e-6)
    if g_fwhm == 0.0:
        assert exact == pytest.approx(l, rel=1e-6)


def test_fwhm_spans_a_resolved_progression():
    """S = 0.9: the 0-1 line is 0.9 of the 0-0 one, so the width spans both.

    Half of the 0-0 maximum is crossed on the 0-0 line's blue flank at
    sigma sqrt(2 ln 2) and on the 0-1 line's red flank at sigma sqrt(2 ln 1.8);
    the 0-2 line, 0.405 high, stays below.
    """
    s, g = 0.9, 0.02 * EV
    sigma = g / GAUSSIAN_FWHM_PER_SIGMA
    de = np.arange(-STIFF - 6.0 * sigma, 6.0 * sigma, sigma / 40.0)
    rho = fc_weighted_dos(de, [s], [STIFF], COLD, broadening=sigma)
    expected = STIFF + sigma * (np.sqrt(2.0 * np.log(2.0))
                                + np.sqrt(2.0 * np.log(2.0 * s)))
    assert fwhm(de, rho) == pytest.approx(expected, rel=2e-4)


def test_broadening_is_required_and_cannot_be_negative():
    e = np.linspace(2.5, 3.5, 11) * EV
    with pytest.raises(TypeError):
        band_shape([1.0], [STIFF], E00, ROOM, e)
    with pytest.raises(ValueError):
        band_shape([1.0], [STIFF], E00, ROOM, e, gaussian_fwhm=-1e-3,
                   lorentzian_fwhm=0.0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
