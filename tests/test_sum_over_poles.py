"""The pole model must reproduce contour deformation, not merely resemble it.

The sharp test needs no physics: give both routes a screening that IS a sum of
M poles. Then the fit is exact, the two are the same function of the same data,
and any disagreement is an error in the contour algebra of one of them -- the
convolution closed in the half plane free of the propagator pole
(`sum_over_poles`) against the integral-plus-residues split
(`contour_deformation`). A model screening also makes the failure mode testable:
put omega beyond the lowest auxiliary pole and the denominators of Eq. (25)
become resonant, which is the one thing no M repairs.

The physics gate at the end is the same comparison on water, where the fit is
no longer exact and the residual is the compression error.
"""
import warnings

import numpy as np
import pytest

from src.Base.constants import HARTREE_TO_EV
from src.Base.utils.grids import gauss_legendre_grid, gap_scaled_w0
from src.gradients.contour_deformation import (residue_set, sigma_cd,
                                               screening_contraction)
from src.gradients.sum_over_poles import (SOP_CLEARANCE_MIN, compressible,
                                          fit_poles, initial_poles,
                                          pole_amplitudes, pole_basis,
                                          pole_clearance, qp_energy_sop,
                                          sigma_sop, sigma_sop_backward,
                                          sigma_sop_slope, sop_from_wc)

EPS = np.array([-0.90, -0.62, -0.35, 0.18, 0.44, 0.83, 1.25, 1.70])
NOCC = 3
POLES = np.array([0.70, 1.50, 3.00])
WEIGHTS = np.array([0.050, 0.080, 0.120])       # the model's a_m
COUPLING = np.array([0.31, -0.24, 0.45, 0.18, -0.37, 0.22, 0.14, -0.29])


class ModelScreening:
    """W(freq) for the naux = 1 model: the residue backend `sigma_cd` expects.

    W(z) - 1 = sum_m a_m [1/(z - Om_m) - 1/(z + Om_m)] = sum_m 2 a_m Om_m /
    (z^2 - Om_m^2), so one scalar serves both axes and the pair element
    Bp[:,q]^T [W - 1] Bp[:,q] is c_q^2 (W - 1) exactly.
    """

    def __init__(self, poles=POLES, weights=WEIGHTS):
        self.poles, self.weights = np.asarray(poles), np.asarray(weights)

    def value(self, z):
        return 1.0 + float(np.sum(2.0 * self.weights * self.poles
                                  / (z ** 2 - self.poles ** 2)))

    def apply(self, freq, b):
        return self.value(freq) * np.asarray(b, float)


def model_wc(nu_points, poles=POLES, weights=WEIGHTS, coupling=COUPLING):
    """wc[k, q] = c_q^2 sum_m a_m F[k, m] -- the data both routes consume."""
    return np.outer(pole_basis(nu_points, poles) @ weights,
                    np.asarray(coupling) ** 2)


def model_amplitudes(poles=POLES, weights=WEIGHTS, coupling=COUPLING):
    return np.outer(weights, np.asarray(coupling) ** 2)


def grid(n=160):
    return gauss_legendre_grid(n, gap_scaled_w0(EPS, NOCC))


@pytest.mark.parametrize('p', [NOCC - 1, NOCC, NOCC - 3, NOCC + 2])
def test_sop_reproduces_contour_deformation(p):
    """Same screening, same omega: the closed form must equal int + residues."""
    nu, wt = grid()
    wc = model_wc(nu)
    amp = model_amplitudes()
    bp = COUPLING[None, :].copy()
    omega = float(EPS[p] + 0.02)
    cd = sigma_cd(p, omega, bp, EPS, NOCC, nu, wt,
                  residues=residue_set(EPS, NOCC, omega), wc=wc,
                  real_screening=ModelScreening())
    # check=False: p = NOCC + 2 sits 0.043 of a pole from a resonance, and the
    # identity holds there anyway. Clearance limits what a COMPRESSION of a
    # real W can reproduce, not the closed form of an exact model.
    sop = sigma_sop(omega, amp, POLES, EPS, NOCC, check=False)
    assert abs(sop - cd) < 1e-8, f'{sop:.12f} vs {cd:.12f}'


def test_the_test_can_fail():
    """A wrong sign in the virtual branch must not pass the gate above."""
    nu, wt = grid()
    wc, amp, bp = model_wc(nu), model_amplitudes(), COUPLING[None, :].copy()
    omega = float(EPS[NOCC - 1] + 0.02)
    cd = sigma_cd(NOCC - 1, omega, bp, EPS, NOCC, nu, wt,
                  residues=residue_set(EPS, NOCC, omega), wc=wc,
                  real_screening=ModelScreening())
    flipped = sigma_sop(omega, amp, POLES, EPS, NOCC + 1, check=False)
    assert abs(flipped - cd) > 1e-6


def test_amplitudes_are_recovered_exactly():
    """The fit of an M-pole screening on M poles is exact, so A comes back."""
    nu, _ = grid()
    got = pole_amplitudes(model_wc(nu), POLES, nu)
    assert np.allclose(got, model_amplitudes(), atol=1e-10)


def test_vector_fitting_finds_the_true_poles():
    """Poles that are not where the log-spaced guess puts them are found."""
    true = np.array([0.55, 1.90, 4.20])
    nu, _ = grid()
    wc = model_wc(nu, poles=true)
    start = initial_poles(3, 0.3, 6.0)
    assert np.abs(start - true).max() > 0.3, 'the guess must be wrong to test'
    found = fit_poles(wc, nu, start, bounds=(0.05, 20.0))
    assert np.abs(found - true).max() < 1e-6, f'{found} vs {true}'


def test_slope_matches_a_difference_of_the_value():
    nu, _ = grid()
    amp = model_amplitudes()
    omega, h = float(EPS[NOCC - 1] + 0.02), 1e-5
    fd = (sigma_sop(omega + h, amp, POLES, EPS, NOCC)
          - sigma_sop(omega - h, amp, POLES, EPS, NOCC)) / (2 * h)
    exact = sigma_sop_slope(omega, amp, POLES, EPS, NOCC)
    assert abs(exact - fd) < 1e-7 * max(abs(exact), 1.0)


def test_adjoint_is_the_transpose_of_the_fit():
    """dSigma through (wc, eps), against a central difference of the forward."""
    nu, _ = grid()
    wc = model_wc(nu)
    omega = float(EPS[NOCC - 1] + 0.02)
    amp = pole_amplitudes(wc, POLES, nu)
    eps_bar, wc_bar, _ = sigma_sop_backward(omega, amp, POLES, EPS, NOCC, nu)
    rng = np.random.default_rng(11)
    dwc = rng.standard_normal(wc.shape) * np.abs(wc).mean()
    deps = rng.standard_normal(EPS.shape) * 1e-2
    h = 1e-6

    def value(t):
        e = EPS + t * deps
        a = pole_amplitudes(wc + t * dwc, POLES, nu)
        return sigma_sop(omega, a, POLES, e, NOCC, check=False)

    fd = (value(h) - value(-h)) / (2 * h)
    exact = float(np.sum(wc_bar * dwc) + eps_bar @ deps)
    assert abs(exact - fd) < 1e-6 * max(abs(exact), 1.0), f'{exact} vs {fd}'


def test_the_wall_is_the_swept_poles_and_nothing_else():
    """Eq. (27) separates; the obvious alternatives do not.

    Calibrated against benzene/cc-pVDZ at M = 8, where the compression
    reproduces CD to 0.07 meV on every state with no swept pole beyond the gap
    and misses by 32 to 3528 meV on every state that has one. Two candidate
    guards were measured on the same states and rejected: `pole_clearance`
    alone misses the carbon 1s (0.935, and 32 meV out), and the share of
    |Sigma| carried by |eps_q - omega| >= Om_1 ranks them backwards, 0.015 for
    the 1s against 0.28 to 0.59 for the frontier states it gets right.
    """
    frontier = float(EPS[NOCC - 1] + 0.02)
    ok, reach = compressible(frontier, EPS, NOCC)
    assert ok and reach < 1.0, reach
    deep = -1.5                                   # below every occupied level
    ok, reach = compressible(deep, EPS, NOCC)
    assert not ok and reach > 1.0, reach
    with pytest.warns(RuntimeWarning, match='Eq. \\(27\\)'):
        sigma_sop(deep, model_amplitudes(), POLES, EPS, NOCC)


def test_resonant_denominator_is_refused():
    """Resonance is a CROSSING, not a distance: |eps_q - omega| ~= Om_m.

    Moving omega far away from every orbital energy makes the model safer, not
    more dangerous -- at 1.4 Ha below the lowest occupied level the clearance is
    0.14, comfortably inside the guard. What kills it is omega one auxiliary
    pole away from an orbital, where a denominator of Eq. (26) passes through
    zero and the sum is dominated by one term of a model that only ever had an
    envelope to offer.
    """
    amp = model_amplitudes()
    far = float(EPS[0] - 1.4)
    assert pole_clearance(far, POLES, EPS, NOCC) > SOP_CLEARANCE_MIN
    crossing = float(EPS[NOCC] + POLES[0] + 0.01)
    assert pole_clearance(crossing, POLES, EPS, NOCC) < SOP_CLEARANCE_MIN
    with pytest.warns(RuntimeWarning, match='resonant'):
        sigma_sop(crossing, amp, POLES, EPS, NOCC)
    frontier = float(EPS[NOCC - 1] + 0.02)
    assert pole_clearance(frontier, POLES, EPS, NOCC) > SOP_CLEARANCE_MIN


def test_quasiparticle_solve_lands_on_its_own_fixed_point():
    nu, _ = grid()
    amp = model_amplitudes()
    w, z = qp_energy_sop(NOCC - 1, amp, POLES, EPS, NOCC)
    residual = w - EPS[NOCC - 1] - sigma_sop(w, amp, POLES, EPS, NOCC)
    assert abs(residual) < 1e-10
    assert 0.0 < z <= 1.0


@pytest.mark.parametrize('n_poles, tol_mev', [(8, 5.0), (16, 0.5)])
def test_water_frontier_against_contour_deformation(n_poles, tol_mev):
    """The physics gate: a real screening, where the fit is no longer exact."""
    pyscf = pytest.importorskip('pyscf')
    from pyscf import gto, scf
    from src.Base.pyscf_interface import get_density_fitting_coefficients
    from src.SingleReference.base import get_occ_virt_indices
    from src.gradients.contour_deformation import (_ov_energies, _wc_explicit,
                                                   qp_energy_cd)
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-10)
    eps = np.asarray(mf.mo_energy, float)
    nocc = mol.nelectron // 2
    b = get_density_fitting_coefficients(mol, mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    c_ov = b[:, occ, :][:, :, virt].reshape(b.shape[0], -1)
    nu, wt = gauss_legendre_grid(48, gap_scaled_w0(eps, nocc))
    for p in (nocc - 1, nocc):
        bp = b[:, p, :]
        wc = _wc_explicit(bp, c_ov, _ov_energies(eps, nocc), nu)
        w_cd, _, _ = qp_energy_cd(p, bp, eps, nocc, nu, wt, wc=wc, C_ov=c_ov,
                                  relax_offset=False)
        poles, amp = sop_from_wc(wc, nu, eps, nocc, n_poles=n_poles, stride=3)
        ok, reach = compressible(w_cd, eps, nocc)
        assert ok, f'orbital {p} is outside the wall at reach {reach:.2f}'
        w_sop, _ = qp_energy_sop(p, amp, poles, eps, nocc, w0=w_cd)
        err = abs(w_sop - w_cd) * HARTREE_TO_EV * 1e3
        assert err < tol_mev, (f'orbital {p}: SOP {w_sop * HARTREE_TO_EV:.4f} '
                               f'vs CD {w_cd * HARTREE_TO_EV:.4f} eV, '
                               f'{err:.2f} meV at M = {n_poles}')


def test_the_guard_bounds_the_value_and_not_the_slope():
    """`compressible` is calibrated on Sigma, and dSigma/domega is an order of
    magnitude worse at the same reach.

    The consequence is a state that passes the guard, returns a quasiparticle
    energy good to micro-Hartree, and carries a FORCE that is wrong -- a silent
    failure, because every check is green.

    The ratio is MEASURED, not derived. The obvious reading, that
    dSigma/domega ~ -sum_m A_m / D_m^2 weights a small denominator more
    heavily than Sigma ~ sum_m A_m / D_m does, is not what happens here: the
    per-pole shares of the two sums agree to within a factor of two and the
    lowest pole carries 0.001 of either. Whatever sets the ratio, it is not
    one pole dominating the derivative, so do not tune a guard against that
    picture.

    Measured on the Toelle set: thioketene/aug-cc-pVTZ at its S0 structure
    agrees with the unapproximated route to 8 neV on the excitation energy and
    differs by 5.7e-03 Ha/Bohr, 15%, on the gradient, with no warning raised.
    """
    nu = np.linspace(0.01, 40.0, 96)
    fit_poles, fit_amp = sop_from_wc(model_wc(nu), nu, EPS, NOCC, n_poles=12)
    ex_amp = model_amplitudes()
    ratios = []
    for omega in np.linspace(0.30, 0.70, 9):
        ok, reach = compressible(omega, EPS, NOCC)
        assert ok, 'this sweep must stay inside the guard'
        value = abs(sigma_sop(omega, fit_amp, fit_poles, EPS, NOCC, check=False)
                    - sigma_sop(omega, ex_amp, POLES, EPS, NOCC, check=False))
        slope = abs(sigma_sop_slope(omega, fit_amp, fit_poles, EPS, NOCC)
                    - sigma_sop_slope(omega, ex_amp, POLES, EPS, NOCC))
        ratios.append(slope / value)
    assert min(ratios) > 10.0, \
        f'the slope should be the demanding one everywhere, got {min(ratios)}'
    # And it is worst where the guard is about to refuse, which is exactly
    # where a caller is most likely to trust a pass.
    ok, reach = compressible(0.70, EPS, NOCC)
    assert ok and reach > 0.95, 'that omega should be a NARROW pass'
    slope = abs(sigma_sop_slope(0.70, fit_amp, fit_poles, EPS, NOCC)
                - sigma_sop_slope(0.70, ex_amp, POLES, EPS, NOCC))
    assert slope > 1e-2, f'a passing state carries slope error {slope}'
