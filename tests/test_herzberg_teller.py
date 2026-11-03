"""Gates for src/properties/herzberg_teller.py and the dipole-weighted
correlation function of src/properties/duschinsky.py, and for
`photophysics.rates.vibronic_radiative_rate` (Herzberg-Teller emission).

References: the Condon density times |mu0|^2, the one-quantum false origin of
a symmetry-forbidden transition in closed form, a sum over states whose
dipole matrix elements are integrated on a quadrature grid, and an analytic
linear dipole for the finite differences. No electronic structure.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.numerics.constants import (ATOMIC_TIME_SECONDS,
                                         BOLTZMANN_HARTREE_PER_KELVIN,
                                         GAUSSIAN_FWHM_PER_SIGMA,
                                         HARTREE_TO_CM, HARTREE_TO_EV,
                                         SPEED_OF_LIGHT_AU)
from src.photophysics.rates import radiative_rate, vibronic_radiative_rate
from src.properties.band_shape import (DisplacedOscillators, vibronic_band,
                                       vibronic_band_grid)
from src.properties.duschinsky import (DuschinskyModel, DuschinskyOscillators,
                                       HarmonicState, duschinsky_density)
from src.properties.franck_condon import TRAPEZOID, bose, fc_weighted_dos
from src.properties.herzberg_teller import (TransitionDipole,
                                            TransitionDipolePhase,
                                            dipole_derivative,
                                            dipole_from_cartesian)
from tests.test_duschinsky import TWO_MODE, overlap_lines
from tests.molecules import WATER_ROUNDED_LOW

CM = 1.0 / HARTREE_TO_CM
EV = 1.0 / HARTREE_TO_EV
SIGMA = 60.0 * CM
DE = np.linspace(-0.04, 0.04, 41)
MU0 = np.array([0.3, 0.0, -0.1])
DMU = np.array([[0.004, -0.002, 0.0], [0.0, 0.003, 0.006]])    # (2 modes, 3)


# ----------------------------------------------------------- the Condon limit
@pytest.mark.parametrize('temperature', [1.0, 300.0])
@pytest.mark.parametrize('direction', ['absorption', 'emission'])
def test_herzberg_teller_off_is_condon(temperature, direction):
    """dmu/dq = 0 through the whole dipole path: |mu0|^2 times the Condon
    density, to roundoff."""
    condon = duschinsky_density(TWO_MODE, DE, temperature, direction,
                                broadening=SIGMA)
    ht = DuschinskyOscillators(TWO_MODE, TransitionDipole.condon(MU0, 2))
    got = ht.density(DE, temperature, direction, broadening=SIGMA)
    assert np.abs(got - MU0 @ MU0 * condon).max() < 1e-13 * got.max()


def test_herzberg_teller_off_band_is_the_condon_band():
    e = vibronic_band_grid(DuschinskyOscillators(TWO_MODE), 3.0 * EV, 300.0,
                           gaussian_fwhm=0.02 * EV, lorentzian_fwhm=0.0,
                           points_per_width=3)
    kw = dict(gaussian_fwhm=0.02 * EV, lorentzian_fwhm=0.0)
    a = vibronic_band(DuschinskyOscillators(TWO_MODE), 3.0 * EV, 300.0, e,
                      **kw)
    b = vibronic_band(DuschinskyOscillators(
        TWO_MODE, TransitionDipole.condon(MU0, 2, 'excited')), 3.0 * EV,
        300.0, e, **kw)
    for key in ('absorption', 'emission', 'emission_per_nm'):
        assert np.abs(a[key] - b[key]).max() < 1e-12 * a[key].max()


# --------------------------------------------- a symmetry-forbidden transition
@pytest.mark.parametrize('temperature', [1.0, 300.0])
def test_forbidden_transition_borrows_through_one_quantum(temperature):
    """mu0 = 0 and dmu/dq only along an undisplaced mode b: the band is the
    displaced mode's Condon band moved by one quantum of b either way,

        rho_HT(dE) = |m|^2 / (2 w_b) [(n_b + 1) rho(dE + w_b) + n_b rho(dE - w_b)],

    and nothing at all without the derivative."""
    w = np.array([1500.0, 400.0]) * CM
    dq = np.array([1.1 / np.sqrt(w[0]), 0.0])
    model = DuschinskyModel.from_displacement(w, dq)
    m = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])
    dip = TransitionDipole(np.zeros(3), m, 'ground')
    s_a = w[0] * dq[0] ** 2 / 2.0
    n_b = bose(w[1:], BOLTZMANN_HARTREE_PER_KELVIN * temperature)[0]
    de = np.linspace(-0.04, 0.02, 31)
    for direction in ('absorption', 'emission'):
        got = DuschinskyOscillators(model, dip).density(
            de, temperature, direction, broadening=SIGMA)
        ref = (0.01 ** 2 / (2.0 * w[1])
               * ((n_b + 1.0) * fc_weighted_dos(de + w[1], [s_a], w[:1],
                                                temperature, broadening=SIGMA)
                  + n_b * fc_weighted_dos(de - w[1], [s_a], w[:1],
                                          temperature, broadening=SIGMA)))
        assert np.abs(got - ref).max() < 1e-9 * ref.max()
    with pytest.raises(ValueError, match='vanishes'):
        DuschinskyOscillators(model, TransitionDipole.condon(
            np.zeros(3), 2)).density(de, temperature, 'emission',
                                     broadening=SIGMA)


# ------------------------------------------------------------ sum over states
def dipole_on_grid(mu0, m, direction):
    """mu_a(q) on the brute-force grid, q the initial coordinates of
    `direction`, for a dipole expanded about the GROUND minimum: emission
    starts on the excited state, q_g = J^T (q_e - K)."""
    def op(q):
        if direction == 'absorption':
            qg = q
        else:
            qg = np.einsum('ji,jab->iab', TWO_MODE.J,
                           q - TWO_MODE.K[:, None, None])
        return [mu0[a] + np.einsum('k,kab->ab', m[:, a], qg) for a in range(3)]
    return op


@pytest.mark.parametrize('temperature, direction, n_init',
                         [(1.0, 'absorption', 1), (1.0, 'emission', 1),
                          (150.0, 'emission', 9)])
def test_rotated_model_with_interference_matches_the_sum_over_states(
        temperature, direction, n_init):
    """Condon and Herzberg-Teller terms interfere along both rotated,
    displaced, frequency-changed modes: the density from the dipole
    correlation function against sum P_n |<m| mu(q) |n>|^2."""
    dip = TransitionDipole(MU0, DMU, 'ground')
    x, w = overlap_lines(*TWO_MODE.oriented(direction), temperature, n_init,
                         40, operator=dipole_on_grid(MU0, DMU, direction))
    ref = np.array([(w * norm.pdf(-d - x, scale=SIGMA)).sum() for d in DE])
    got = DuschinskyOscillators(TWO_MODE, dip).density(
        DE, temperature, direction, broadening=SIGMA)
    assert np.abs(got - ref).max() < 1e-8 * ref.max()
    # the area is the thermal <|mu(q)|^2>
    assert dip.strength(TWO_MODE, temperature, direction) == pytest.approx(
        w.sum(), rel=1e-8)
    # and the cross term matters: flipping the derivative's sign moves it
    flipped = DuschinskyOscillators(TWO_MODE, TransitionDipole(
        MU0, -DMU, 'ground')).density(DE, temperature, direction,
                                       broadening=SIGMA)
    assert np.abs(flipped - got).max() > 1e-3 * got.max()


def test_expansion_point_does_not_change_the_band():
    """The same linear dipole about the ground or the excited minimum."""
    ground = TransitionDipole(MU0, DMU, 'ground')
    # mu = mu0 + m^T q_g, q_g = J^T (q_e - K): about the excited minimum
    excited = TransitionDipole(MU0 - DMU.T @ (TWO_MODE.J.T @ TWO_MODE.K),
                               TWO_MODE.J @ DMU, 'excited')
    for direction in ('absorption', 'emission'):
        a = DuschinskyOscillators(TWO_MODE, ground).density(
            DE, 300.0, direction, broadening=SIGMA)
        b = DuschinskyOscillators(TWO_MODE, excited).density(
            DE, 300.0, direction, broadening=SIGMA)
        assert np.abs(a - b).max() < 1e-12 * a.max()


# ----------------------------------------------------- the dipole derivative
def water_like_state():
    from pyscf import gto
    from src.properties.vibronic import normal_modes
    mol = gto.M(atom=WATER_ROUNDED_LOW,
                basis='sto-3g', verbose=0)
    rng = np.random.default_rng(1)
    a = rng.normal(size=(9, 9))
    h = a @ a.T * 0.05 + np.eye(9) * 0.3
    omega, modes, masses, _ = normal_modes(None, mol=mol, hess=h)
    return HarmonicState(mol, omega, modes, masses, 'model')


def test_finite_difference_of_a_linear_dipole_is_its_projection():
    state = water_like_state()
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 3, 3)) * 0.05
    x0 = state.mol.atom_coords()

    def dipole_at(mol):
        return MU0 + np.einsum('iac,ia->c', a, mol.atom_coords() - x0)

    fd = dipole_derivative(dipole_at, state, 'ground')
    assert fd.suspect_modes == ()
    exact = dipole_from_cartesian(MU0, a, state, 'ground')
    assert np.abs(fd.mu0 - MU0).max() == 0.0
    assert np.abs(fd.dmu_dq - exact.dmu_dq).max() < 1e-10 * np.abs(
        exact.dmu_dq).max()


def test_a_phase_flip_between_displaced_solves_warns():
    state = water_like_state()
    x0 = state.mol.atom_coords()
    calls = iter([1, 1, -1, 1, 1, 1, 1])

    def dipole_at(mol):
        mu = MU0 + 0.05 * (mol.atom_coords() - x0)[0]
        return next(calls) * mu

    with pytest.warns(TransitionDipolePhase, match=r'modes \[0\]'):
        dip = dipole_derivative(dipole_at, state, 'ground')
    assert dip.suspect_modes == (0,)


# ------------------------------------------- the radiative rate over the band
def test_vibronic_radiative_rate():
    """Condon: k_r = 4 |mu|^2 <E^3> / 3 c^3 with <E^3> the band's third raw
    moment in closed form; the Herzberg-Teller model with dmu/dq = 0 is the
    Condon one."""
    s, w = np.array([0.6, 0.3]), np.array([1200.0, 400.0]) * CM
    e_ad, temperature, g = 2.5 * EV, 300.0, 0.03 * EV
    sig = g / GAUSSIAN_FWHM_PER_SIGMA
    k = vibronic_radiative_rate(DisplacedOscillators(s, w), e_ad, temperature,
                                gaussian_fwhm=g, dipole=MU0)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * temperature
    coth = 1.0 / np.tanh(w / (2.0 * kt))
    k1, k2, k3 = (s * w).sum(), (s * w ** 2 * coth).sum() + sig ** 2, \
        (s * w ** 3).sum()
    mean = e_ad - k1                      # emission: E = E_ad - x
    m3 = mean ** 3 + 3.0 * mean * k2 - k3
    exact = (4.0 * (MU0 @ MU0) * m3 / (3.0 * SPEED_OF_LIGHT_AU ** 3)
             / ATOMIC_TIME_SECONDS)
    assert k == pytest.approx(exact, rel=1e-8)
    # a band is not its 0-0 line: the single-energy rate at E_ad is 7 % off
    assert abs(radiative_rate(e_ad, MU0) / k - 1.0) > 0.05
    model = DuschinskyModel.from_displacement(w, -np.sqrt(2.0 * s / w))
    ht = vibronic_radiative_rate(DuschinskyOscillators(
        model, TransitionDipole.condon(MU0, 2, 'excited')), e_ad, temperature,
        gaussian_fwhm=g)
    assert ht == pytest.approx(k, rel=1e-8)
    with pytest.raises(ValueError, match='twice or not at all'):
        vibronic_radiative_rate(DisplacedOscillators(s, w), e_ad, temperature,
                                gaussian_fwhm=g)


def test_refusals():
    with pytest.raises(ValueError, match='reference'):
        TransitionDipole(MU0, DMU, 'S1')
    with pytest.raises(ValueError, match='derivatives'):
        TransitionDipole(MU0, np.zeros((3, 3)), 'ground').in_initial(
            TWO_MODE, 'emission')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
