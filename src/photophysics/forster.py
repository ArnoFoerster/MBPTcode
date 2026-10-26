"""Förster (dipole-dipole) annihilation and transfer rates and radii.

A donor in an excited state D* gives its energy to an acceptor A through the
Coulomb coupling of two transition dipoles, J = kappa mu_D mu_A / (n^2 r^3).
The golden rule over every vibronic line of both molecules gives

    k(r) = C6 / r^6,
    C6   = (2 pi / hbar) <kappa^2> / n^4  int D(E) A(E) dE,

with D(E) = sum_d p_d |mu_d|^2 rho_d^em(E) the donor's emission density and
A(E) = sum_a p_a |mu_a|^2 rho_a^abs(E) the acceptor's absorption density at the
energy E the donor gives up, each rho of unit area (`band_shape.VibronicModel`;
a Herzberg-Teller model carries its own <|mu|^2> instead). The Förster radius
is where this rate equals the donor's own decay rate 1/tau:

    R_F^6 = C6 tau = eta C6 / k_r,

with eta the donor's emission quantum yield and k_r its radiative rate in the
medium, n times the vacuum rate (`donor_radiative_rate`). Only the shape of the
donor's band enters R_F: its dipole cancels between C6 and k_r.

ANNIHILATION. The acceptor already carries an exciton, so A(E) is an
excited-state absorption, T1 -> Tn or S1 -> Sn, from the moments of
`properties.state_moments` (`excited_absorption_lines`). For an emitter with
both S1 and T1 populated the channels are singlet-triplet annihilation (donor
S1 emission, acceptor T1 absorption), triplet-triplet annihilation (donor T1
phosphorescence, acceptor T1 absorption) and singlet-singlet annihilation
(donor S1, acceptor S1 absorption). The model follows van Hoesel, Chibueze,
Visscher, Coehoorn and Bobbert, J. Chem. Phys. 164, 174314 (2026): dipole-dipole
coupling only (their quadrupolar terms are small for annihilation),
<kappa^2> = 2/3 for dynamic isotropic averaging. Their Förster radii fitted to
time-resolved photoluminescence carry a correlation factor of about 1.06,
which is not applied here.

The spectroscopic form, from a measured or computed emission spectrum and
molar absorption coefficient, is `forster_radius_from_spectra`; on the same
bands it equals the golden-rule radius (the tests check it, and check both
against the textbook constant of the Förster equation).

Energies in Hartree, distances in Bohr unless a name says otherwise,
temperature in Kelvin, rates in s^-1.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

from src.Base.numerics.constants import (ATOMIC_TIME_SECONDS,
                                         BOLTZMANN_HARTREE_PER_KELVIN,
                                         BOHR_TO_ANGSTROM,
                                         FORSTER_MIN_POINTS_PER_WIDTH,
                                         FORSTER_POINTS_PER_WIDTH,
                                         GAUSSIAN_FWHM_PER_SIGMA,
                                         MOLAR_ABSORPTION_PER_BOHR2,
                                         SPEED_OF_LIGHT_AU)
from src.photophysics.rates import vibronic_radiative_rate
from src.properties.band_shape import VibronicModel
from src.properties.franck_condon import TRAPEZOID

#: <kappa^2> of two dipoles that rotate freely during the transfer.
KAPPA2_ISOTROPIC = 2.0 / 3.0


@dataclass(frozen=True)
class Line:
    """One electronic transition and its vibronic band.

    energy: the electronic adiabatic gap of the transition, Hartree, > 0
        (the 0-0 line sits there plus the model's zero-point shift).
    model: its `band_shape.VibronicModel`.
    strength: |mu|^2 in atomic units for a Condon model; None for a model
        that carries its own dipole (Herzberg-Teller), whose density has
        <|mu|^2> as its area.
    weight: the population of the transition's initial state.
    """
    energy: float
    model: VibronicModel
    strength: Optional[float] = None
    weight: float = 1.0

    def __post_init__(self):
        if not self.energy > 0.0:
            raise ValueError(f'a line needs a positive energy, got '
                             f'{self.energy!r} Ha')
        if self.weight < 0.0:
            raise ValueError(f'a population cannot be negative ({self.weight})')
        own = getattr(self.model, 'dipole', None) is not None
        if own == (self.strength is not None):
            raise ValueError('the transition dipole is given twice or not at '
                             'all: a Condon model takes `strength`, a '
                             'Herzberg-Teller model carries its own')
        if self.strength is not None and self.strength < 0.0:
            raise ValueError(f'|mu|^2 cannot be negative ({self.strength})')

    @property
    def dipole2(self):
        return 1.0 if self.strength is None else float(self.strength)


def thermal_populations(energies, temperature, degeneracy=None):
    """Boltzmann populations of initial states at `energies` (Hartree),
    normalized over the states given; `degeneracy` per state, default 1."""
    e = np.asarray(energies, float)
    g = np.ones_like(e) if degeneracy is None else np.asarray(degeneracy,
                                                              float)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)
    if kt <= 0.0:
        raise ValueError(f'temperature {temperature} K is not positive')
    w = g * np.exp(-(e - e.min()) / kt)
    return w / w.sum()


def excited_absorption_lines(e_initial, e_final, dipoles, model, temperature,
                             degeneracy=None):
    """The lines of absorption out of thermally populated excited states.

    e_initial: (n_i,) energies of the initial excited states (T1, T2, ...),
        e_final: (n_f,) those of the final states, both above the same ground
        state, Hartree.
    dipoles: (n_i, n_f, 3) or (n_i, n_f) moments <i|r|f>
        (`state_moments.excited_transition_dipoles`).
    model: one `VibronicModel` for every line, or an (n_i, n_f) nested
        sequence of them.

    Only upward transitions, e_f > e_i, are lines: the stimulated emission
    toward lower states is left out (it needs the initial state above the
    final one, which a thermal population of the lowest states rarely has).
    Populations are Boltzmann over `e_initial` (`thermal_populations`).
    """
    ei, ef = np.atleast_1d(np.asarray(e_initial, float)), \
        np.atleast_1d(np.asarray(e_final, float))
    mu = np.asarray(dipoles, float)
    mu2 = (mu ** 2).sum(-1) if mu.ndim == 3 else mu ** 2
    if mu2.shape != (ei.size, ef.size):
        raise ValueError(f'dipoles give {mu2.shape} pairs; the energies give '
                         f'{(ei.size, ef.size)}')
    pop = thermal_populations(ei, temperature, degeneracy)
    lines = []
    for i in range(ei.size):
        for f in range(ef.size):
            if ef[f] <= ei[i]:
                continue
            m = model if isinstance(model, VibronicModel) else model[i][f]
            lines.append(Line(float(ef[f] - ei[i]), m,
                              strength=float(mu2[i, f]), weight=float(pop[i])))
    if not lines:
        raise ValueError('no final state lies above an initial one')
    return lines


def _widths(gaussian_fwhm, lorentzian_fwhm):
    if gaussian_fwhm < 0.0 or lorentzian_fwhm < 0.0:
        raise ValueError('a broadening FWHM cannot be negative')
    return dict(broadening=float(gaussian_fwhm) / GAUSSIAN_FWHM_PER_SIGMA,
                lorentzian=0.5 * float(lorentzian_fwhm))


def emission_density(lines, energies, temperature, *, gaussian_fwhm,
                     lorentzian_fwhm):
    """D(E) = sum p |mu|^2 rho^em(E) at photon energies E, a.u. Hartree^-1."""
    e = np.asarray(energies, float)
    w = _widths(gaussian_fwhm, lorentzian_fwhm)
    return sum(ln.weight * ln.dipole2
               * ln.model.density(e - ln.energy, temperature, 'emission', **w)
               for ln in lines)


def absorption_density(lines, energies, temperature, *, gaussian_fwhm,
                       lorentzian_fwhm):
    """A(E) = sum p |mu|^2 rho^abs(E) at photon energies E, a.u. Hartree^-1."""
    e = np.asarray(energies, float)
    w = _widths(gaussian_fwhm, lorentzian_fwhm)
    return sum(ln.weight * ln.dipole2
               * ln.model.density(ln.energy - e, temperature, 'absorption',
                                  **w)
               for ln in lines)


def _narrowest_width(donor, acceptor, temperature, sigma_g, lorentzian_fwhm):
    """The smallest standard deviation of any band (with the Gaussian
    broadening) and of the broadenings themselves, Hartree."""
    widths = [sigma_g, 0.5 * float(lorentzian_fwhm)]
    for lines, direction in ((donor, 'emission'), (acceptor, 'absorption')):
        for ln in lines:
            _, var = ln.model.cumulants(temperature, direction)
            widths.append(float(np.sqrt(max(var, 0.0) + sigma_g ** 2)))
    widths = [w for w in widths if w > 0.0]
    if not widths:
        raise ValueError('the bands have no width: no displacement and no '
                         'broadening')
    return min(widths)


def overlap_grid(donor, acceptor, temperature, *, gaussian_fwhm,
                 lorentzian_fwhm, n_widths=12.0,
                 points_per_width=FORSTER_POINTS_PER_WIDTH):
    """Uniform photon-energy grid for the overlap integral, Hartree.

    It spans the donor's emission (the integrand vanishes outside it),
    n_widths standard deviations (plus the Lorentzian FWHM) beyond each
    line's mean, and resolves the narrowest width of any donor or acceptor
    band and of the broadenings by points_per_width points.
    """
    sigma_g = float(gaussian_fwhm) / GAUSSIAN_FWHM_PER_SIGMA
    lo, hi = np.inf, -np.inf
    for ln in donor:
        mean, var = ln.model.cumulants(temperature, 'emission')
        sig = float(np.sqrt(max(var, 0.0) + sigma_g ** 2))
        reach = n_widths * (sig + float(lorentzian_fwhm))
        lo = min(lo, ln.energy - mean - reach)
        hi = max(hi, ln.energy - mean + reach)
    if lo <= 0.0:
        raise ValueError(f'the donor band reaches zero photon energy (lower '
                         f'end {lo:.4g} Ha); pass a grid')
    narrow = _narrowest_width(donor, acceptor, temperature, sigma_g,
                              lorentzian_fwhm)
    n = int(np.ceil((hi - lo) * points_per_width / narrow)) + 1
    return np.linspace(lo, hi, n)


def _model_groups(lines):
    """Lines grouped by their model object, in first-seen order."""
    groups = {}
    for ln in lines:
        groups.setdefault(id(ln.model), []).append(ln)
    return list(groups.values())


def _line_overlaps(d, e, group, temperature, widths):
    """int D(E) rho^abs(E_l - E) dE for every line l of one model.

    On the uniform grid E_k = E_0 + k h the differences y_j - E_k between
    the lattice y_j = E_0 + j h and the grid are multiples of h, so the
    trapezoid sums C(y_j) = sum_k w_k D_k rho((j - k) h) for every j are ONE
    discrete convolution with rho sampled once on that lattice -- exact
    samples, no interpolation of the band. C, the donor band convolved with
    the acceptor's, is then interpolated (cubic) at each line energy; it is
    at least as wide as the narrowest band, which the grid resolves by
    `FORSTER_MIN_POINTS_PER_WIDTH` points or more, so the interpolation
    error is below about 1e-5 of C's peak there and 2e-7 at the default
    resolution.
    """
    h = e[1] - e[0]
    n = e.size
    el = np.array([ln.energy for ln in group])
    j_lo = int(np.floor((el.min() - e[0]) / h)) - 3
    j_hi = int(np.ceil((el.max() - e[0]) / h)) + 3
    m0 = j_lo - (n - 1)
    x = np.arange(m0, j_hi + 1) * h
    rho = group[0].model.density(x, temperature, 'absorption', **widths)
    wd = d * h
    wd[0] *= 0.5
    wd[-1] *= 0.5
    c = np.convolve(wd, rho)[n - 1:n - 1 + (j_hi - j_lo + 1)]
    y = e[0] + np.arange(j_lo, j_hi + 1) * h
    return CubicSpline(y, c)(el)


def forster_c6(donor, acceptor, temperature, *, gaussian_fwhm,
               lorentzian_fwhm, kappa2=KAPPA2_ISOTROPIC, refractive_index=1.0,
               energies=None):
    """C6 = r^6 k(r) of the dipole-dipole golden rule, in s^-1 Bohr^6.

    donor, acceptor: sequences of `Line` (the donor's emitting transitions,
        the acceptor's absorbing ones); broadenings as in `band_shape`,
        applied to every line of both. refractive_index: n of the medium,
        which screens the coupling by n^2 (the optical dielectric constant).
    energies: a uniform photon grid, default `overlap_grid`; a grid that
        resolves the narrowest band by fewer than
        `FORSTER_MIN_POINTS_PER_WIDTH` points is refused.

    COST. The donor band is evaluated on the grid; the acceptor's lines are
    grouped by model object, and each group costs one band evaluation on a
    lattice as long as the grid plus the span of its line energies
    (`_line_overlaps`). Hundreds of excited-state absorption lines that share
    one model (`excited_absorption_lines` with a single model) cost about
    one band, not one band each.

    Returns a dict: `c6`; `overlap`, int D A dE in a.u.; `line_overlaps`,
    int D rho_l dE per acceptor line in the order given (times its weight
    and |mu|^2 it is that line's share); `radiative_rate`, the donor's k_r
    in the medium, n 4/(3 c^3) int E^3 D dE in s^-1 on the same grid;
    `energies` and `emission` D(E).
    """
    if kappa2 < 0.0 or refractive_index <= 0.0:
        raise ValueError('<kappa^2> must be >= 0 and n > 0')
    if energies is None:
        energies = overlap_grid(donor, acceptor, temperature,
                                gaussian_fwhm=gaussian_fwhm,
                                lorentzian_fwhm=lorentzian_fwhm)
    e = np.asarray(energies, float)
    if e.size < 3 or np.any(e <= 0.0):
        raise ValueError('photon energies must be positive, three or more')
    h = (e[-1] - e[0]) / (e.size - 1)
    if not h > 0.0 or np.abs(np.diff(e) - h).max() > 1e-9 * h:
        raise ValueError('the photon grid must be uniform and ascending')
    widths = _widths(gaussian_fwhm, lorentzian_fwhm)
    narrow = _narrowest_width(donor, acceptor, temperature,
                              widths['broadening'], lorentzian_fwhm)
    if narrow / h < FORSTER_MIN_POINTS_PER_WIDTH:
        raise ValueError(f'the grid spacing {h:.3g} Ha resolves the narrowest '
                         f'band ({narrow:.3g} Ha) by {narrow / h:.3g} points; '
                         f'{FORSTER_MIN_POINTS_PER_WIDTH} or more are needed')
    d = emission_density(donor, e, temperature, gaussian_fwhm=gaussian_fwhm,
                         lorentzian_fwhm=lorentzian_fwhm)
    per_line = {}
    for group in _model_groups(acceptor):
        for ln, o in zip(group, _line_overlaps(d, e, group, temperature,
                                               widths)):
            per_line[id(ln)] = float(o)
    line_overlaps = np.array([per_line[id(ln)] for ln in acceptor])
    overlap = float(sum(ln.weight * ln.dipole2 * o
                        for ln, o in zip(acceptor, line_overlaps)))
    c6 = (2.0 * np.pi * float(kappa2) / float(refractive_index) ** 4
          * overlap / ATOMIC_TIME_SECONDS)
    k_r = (float(refractive_index) * 4.0 * float(TRAPEZOID(e ** 3 * d, e))
           / (3.0 * SPEED_OF_LIGHT_AU ** 3) / ATOMIC_TIME_SECONDS)
    return {'c6': c6, 'overlap': overlap, 'line_overlaps': line_overlaps,
            'radiative_rate': k_r, 'energies': e, 'emission': d}


def donor_radiative_rate(donor, temperature, *, gaussian_fwhm,
                         lorentzian_fwhm, refractive_index=1.0):
    """k_r = n sum p k_r^vac of the donor's lines, s^-1
    (`rates.vibronic_radiative_rate` per line)."""
    total = 0.0
    for ln in donor:
        dipole = None if ln.strength is None else np.sqrt(ln.strength)
        total += ln.weight * vibronic_radiative_rate(
            ln.model, ln.energy, temperature, gaussian_fwhm=gaussian_fwhm,
            lorentzian_fwhm=lorentzian_fwhm, dipole=dipole)
    return float(refractive_index) * total


def forster_radius(c6, lifetime):
    """R_F = (C6 tau)^(1/6) in Bohr, with tau the donor's lifetime in s."""
    if c6 < 0.0 or lifetime <= 0.0:
        raise ValueError('C6 must be >= 0 and the lifetime > 0')
    return float((c6 * lifetime) ** (1.0 / 6.0))


def forster_radius_yield(c6, radiative_rate, quantum_yield):
    """R_F = (eta C6 / k_r)^(1/6) in Bohr: tau = eta / k_r."""
    if not 0.0 < quantum_yield <= 1.0 or radiative_rate <= 0.0:
        raise ValueError('the quantum yield must lie in (0, 1] and k_r > 0')
    return forster_radius(c6, quantum_yield / radiative_rate)


def normalized_emission(lines, energies, temperature, *, gaussian_fwhm,
                        lorentzian_fwhm):
    """I(E) = E^3 D(E), normalized to unit area on E: the photon-emission
    rate per unit photon energy (`band_shape`'s emission convention)."""
    e = np.asarray(energies, float)
    i = e ** 3 * emission_density(lines, e, temperature,
                                  gaussian_fwhm=gaussian_fwhm,
                                  lorentzian_fwhm=lorentzian_fwhm)
    return i / TRAPEZOID(i, e)


def molar_absorption(lines, energies, temperature, *, gaussian_fwhm,
                     lorentzian_fwhm, refractive_index=1.0):
    """The molar absorption coefficient in M^-1 cm^-1.

        sigma(E) = 4 pi^2 E A(E) / (3 c n),   eps = sigma N_A / (1000 ln 10)

    with sigma the absorption cross section per molecule (Bohr^2 here)."""
    e = np.asarray(energies, float)
    a = absorption_density(lines, e, temperature,
                           gaussian_fwhm=gaussian_fwhm,
                           lorentzian_fwhm=lorentzian_fwhm)
    sigma = 4.0 * np.pi ** 2 * e * a / (3.0 * SPEED_OF_LIGHT_AU
                                         * float(refractive_index))
    return sigma * MOLAR_ABSORPTION_PER_BOHR2


def forster_radius_from_spectra(energies, emission, epsilon, quantum_yield, *,
                                kappa2=KAPPA2_ISOTROPIC,
                                refractive_index=1.0):
    """R_F in Bohr from spectra, the Förster equation on an energy axis:

        R_F^6 = 9 c^4 hbar^4 ln 10 <kappa^2> eta / (8 pi n^4 N_A)
                int I(E) eps(E) / E^4 dE

    energies: ascending photon energies, Hartree; emission: the donor's
    emission spectrum on them (normalized here to unit area); epsilon: the
    acceptor's molar absorption coefficient, M^-1 cm^-1.
    """
    e = np.asarray(energies, float)
    if np.any(e <= 0.0) or np.any(np.diff(e) <= 0.0):
        raise ValueError('photon energies must be positive and ascending')
    if not 0.0 < quantum_yield <= 1.0:
        raise ValueError('the quantum yield must lie in (0, 1]')
    i = np.asarray(emission, float)
    i = i / TRAPEZOID(i, e)
    sigma = np.asarray(epsilon, float) / MOLAR_ABSORPTION_PER_BOHR2
    r6 = (9.0 * SPEED_OF_LIGHT_AU ** 4 * float(kappa2) * quantum_yield
          / (8.0 * np.pi * float(refractive_index) ** 4)
          * TRAPEZOID(i * sigma / e ** 4, e))
    return float(r6 ** (1.0 / 6.0))


def bohr_to_nm(r):
    """A distance in Bohr, in nm."""
    return float(r) * BOHR_TO_ANGSTROM / 10.0
