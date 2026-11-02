"""Gates for src/properties/state_moments.py and src/photophysics/forster.py.

The excited-to-excited dipoles are checked against an independent oracle: the
CIS states written as full-CI vectors (creation and annihilation operators
applied to the Hartree-Fock string, so every sign is pyscf's) and their
transition density from `fci.direct_spin1.trans_rdm1`, singlets and triplets.
The Förster code is checked against a closed form (Gaussian bands), against
the spectroscopic Förster equation on the same bands, and against the textbook
constant of that equation (R_F in nm = 0.02108 (kappa^2 eta n^-4 J)^(1/6), J
in M^-1 cm^-1 nm^4, the constant taken in full from its SI form), which shares
no code with either.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import fci, gto, scf, tdscf

from src.Base.numerics.constants import (ATOMIC_TIME_SECONDS,
                                         GAUSSIAN_FWHM_PER_SIGMA,
                                         HARTREE_TO_CM, HARTREE_TO_EV,
                                         HARTREE_WAVELENGTH_NM)
from src.gradients.gwbse.derivative_coupling import state_to_state_density
from src.photophysics.forster import (KAPPA2_ISOTROPIC, Line,
                                      absorption_density, bohr_to_nm,
                                      donor_radiative_rate,
                                      excited_absorption_lines, forster_c6,
                                      forster_radius_from_spectra,
                                      forster_radius_yield, molar_absorption,
                                      normalized_emission, overlap_grid)
from src.properties.band_shape import DisplacedOscillators
from src.properties.conformers import boltzmann_weights
from src.properties.franck_condon import TRAPEZOID
from src.properties.state_moments import excited_transition_dipoles

EV = 1.0 / HARTREE_TO_EV
CM = 1.0 / HARTREE_TO_CM
ROOM = 300.0


# ---------------------------------------------------------------- moments

@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    return mol, mf


def _cis(mf, singlet, nstates=6):
    td = tdscf.TDA(mf)
    td.singlet = singlet
    td.nstates = nstates
    td.conv_tol = 1e-10
    td.kernel()
    # pyscf normalizes <X|X> = 1/2; the solvers here use 1
    x = np.array([np.sqrt(2.0) * xy[0].ravel() for xy in td.xy]).T
    return td.e, x


def _fci_vectors(x, nocc, norb, singlet):
    """CIS states as full-CI vectors: sum_ia X_ia (E^a_i,alpha +- beta)/sqrt2."""
    nelec = (nocc, nocc)
    hf = np.zeros((fci.cistring.num_strings(norb, nocc),) * 2)
    hf[0, 0] = 1.0
    nvir = norb - nocc
    sign = 1.0 if singlet else -1.0
    out = []
    for col in x.T:
        v = np.zeros_like(hf)
        for i in range(nocc):
            for a in range(nvir):
                c = col[i * nvir + a]
                if c == 0.0:
                    continue
                p = nocc + a
                va = fci.addons.cre_a(fci.addons.des_a(hf, norb, nelec, i),
                                      norb, (nocc - 1, nocc), p)
                vb = fci.addons.cre_b(fci.addons.des_b(hf, norb, nelec, i),
                                      norb, (nocc, nocc - 1), p)
                v += c * (va + sign * vb) / np.sqrt(2.0)
        out.append(v)
    return out


def _oracle(mol, mf, x, singlet):
    nocc = mol.nelectron // 2
    norb = mf.mo_coeff.shape[1]
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ao = mol.intor_symmetric('int1e_r', comp=3)
    d = np.einsum('xmn,mp,nq->xpq', ao, mf.mo_coeff, mf.mo_coeff)
    vecs = _fci_vectors(x, nocc, norb, singlet)
    n = len(vecs)
    out = np.zeros((n, n, 3))
    for m in range(n):
        for k in range(n):
            # trans_rdm1 gives <m|q^+ p|k> as dm[p, q]
            dm = fci.direct_spin1.trans_rdm1(vecs[m], vecs[k], norb,
                                             (nocc, nocc))
            out[m, k] = np.einsum('xpq,qp->x', d, dm)
    return out, vecs


@pytest.mark.parametrize('singlet', [True, False])
def test_excited_dipoles_equal_full_ci_transition_density(water, singlet):
    mol, mf = water
    nocc = mol.nelectron // 2
    _, x = _cis(mf, singlet)
    got = excited_transition_dipoles(mol, mf.mo_coeff, nocc, x, x)
    ref, vecs = _oracle(mol, mf, x, singlet)
    # the oracle's states are normalized and orthogonal, as CIS vectors are
    s = np.array([[np.vdot(a, b) for b in vecs] for a in vecs])
    np.testing.assert_allclose(s, np.eye(len(vecs)), atol=1e-9)
    # the diagonal is the difference dipole: the oracle carries <0|r|0>
    ref_off = ref.copy()
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        mu0 = np.einsum('xmn,nm->x', mol.intor_symmetric('int1e_r', comp=3),
                        mf.make_rdm1())
    for m in range(len(vecs)):
        ref_off[m, m] -= mu0
    np.testing.assert_allclose(got, ref_off, atol=1e-10)
    # the moments are not all zero by symmetry: something is being checked
    off = got[~np.eye(len(vecs), dtype=bool)]
    assert np.abs(off).max() > 0.1


def test_excited_dipoles_share_the_state_to_state_density(water):
    mol, mf = water
    nocc = mol.nelectron // 2
    _, x = _cis(mf, False, nstates=4)
    got = excited_transition_dipoles(mol, mf.mo_coeff, nocc, x[:, 0], x)
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ao = mol.intor_symmetric('int1e_r', comp=3)
    d = np.einsum('xmn,mp,nq->xpq', ao, mf.mo_coeff, mf.mo_coeff)
    zero = np.zeros_like(x[:, 0])
    for n in range(x.shape[1]):
        goo, gvv = state_to_state_density(nocc, x[:, 0], zero, x[:, n], zero)
        ref = (np.einsum('ab,xab->x', gvv, d[:, nocc:, nocc:])
               - np.einsum('ij,xij->x', goo, d[:, :nocc, :nocc]))
        np.testing.assert_allclose(got[0, n], ref, atol=1e-12)


def test_excited_dipoles_refuse_y_and_unnormalized(water):
    mol, mf = water
    nocc = mol.nelectron // 2
    _, x = _cis(mf, True, nstates=2)
    with pytest.raises(ValueError, match='quadratic-response'):
        excited_transition_dipoles(mol, mf.mo_coeff, nocc, x, x,
                                   y_from=1e-3 * x)
    with pytest.raises(ValueError, match='normalized'):
        excited_transition_dipoles(mol, mf.mo_coeff, nocc, x / np.sqrt(2.0), x)
    # zero Y (a Tamm-Dancoff solve that returns it) is accepted
    excited_transition_dipoles(mol, mf.mo_coeff, nocc, x, x,
                               y_from=np.zeros_like(x), y_to=np.zeros_like(x))


# ---------------------------------------------------------------- Förster

MODES = np.array([180.0, 420.0, 760.0, 1240.0, 1610.0]) * CM
S_D = np.array([0.6, 0.35, 0.2, 0.45, 0.3])
S_A = np.array([0.3, 0.5, 0.1, 0.25, 0.4])
GAUSS = 0.1 * EV
LORENTZ = 0.0


def _bands():
    donor = [Line(2.6 * EV, DisplacedOscillators(S_D, MODES), strength=0.8)]
    acceptor = [Line(2.3 * EV, DisplacedOscillators(S_A, MODES),
                     strength=1.5, weight=0.7),
                Line(2.9 * EV, DisplacedOscillators(S_A, MODES),
                     strength=0.4, weight=0.3)]
    return donor, acceptor


def test_gaussian_bands_give_the_closed_form():
    """No displacement, Gaussian broadening only: each band is a Gaussian of
    standard deviation sig, and int D A dE is a Gaussian of variance 2 sig^2
    at the gap between the two lines."""
    flat = DisplacedOscillators([0.0], [1000.0 * CM])
    sig = GAUSS / GAUSSIAN_FWHM_PER_SIGMA
    e_d, e_a, mu_d2, mu_a2, n, k2 = 2.5 * EV, 2.42 * EV, 0.7, 1.3, 1.6, 0.4
    donor = [Line(e_d, flat, strength=mu_d2)]
    acceptor = [Line(e_a, flat, strength=mu_a2)]
    out = forster_c6(donor, acceptor, ROOM, gaussian_fwhm=GAUSS,
                     lorentzian_fwhm=0.0, kappa2=k2, refractive_index=n)
    gap = e_d - e_a
    overlap = (mu_d2 * mu_a2 * np.exp(-gap ** 2 / (4.0 * sig ** 2))
               / np.sqrt(4.0 * np.pi * sig ** 2))
    np.testing.assert_allclose(out['overlap'], overlap, rtol=1e-6)
    np.testing.assert_allclose(
        out['c6'], 2.0 * np.pi * k2 / n ** 4 * overlap / ATOMIC_TIME_SECONDS,
        rtol=1e-6)


@pytest.fixture(scope='module')
def bands_n17():
    """The two bands, their golden-rule C6 in n = 1.7 and the spectra on
    the same grid (the direct per-point absorption, which the lattice route
    of `forster_c6` does not use)."""
    donor, acceptor = _bands()
    kw = dict(gaussian_fwhm=GAUSS, lorentzian_fwhm=LORENTZ)
    out = forster_c6(donor, acceptor, ROOM, refractive_index=1.7, **kw)
    e = out['energies']
    a = [absorption_density([ln], e, ROOM, **kw) for ln in acceptor]
    return donor, acceptor, out, a


def test_lattice_line_overlaps_equal_the_direct_trapezoid(bands_n17):
    _, acceptor, out, a = bands_n17
    e, d = out['energies'], out['emission']
    for ln, a_l, got in zip(acceptor, a, out['line_overlaps']):
        direct = TRAPEZOID(d * a_l, e) / (ln.weight * ln.dipole2)
        np.testing.assert_allclose(got, direct, rtol=1e-6)
    # the second line sits 0.3 eV above the donor's origin: its overlap, 1e-4
    # of the first, is a tail, checked to the same relative accuracy
    assert out['line_overlaps'].min() > 1e-5 * out['line_overlaps'].max()


def test_radiative_rate_on_the_overlap_grid(bands_n17):
    donor, _, out, _ = bands_n17
    ref = donor_radiative_rate(donor, ROOM, gaussian_fwhm=GAUSS,
                               lorentzian_fwhm=LORENTZ, refractive_index=1.7)
    np.testing.assert_allclose(out['radiative_rate'], ref, rtol=1e-6)


def test_golden_rule_radius_equals_the_spectroscopic_one(bands_n17):
    donor, acceptor, out, _ = bands_n17
    eta, n = 0.6, 1.7
    kw = dict(gaussian_fwhm=GAUSS, lorentzian_fwhm=LORENTZ)
    r_golden = forster_radius_yield(out['c6'], out['radiative_rate'], eta)
    e = out['energies']
    em = normalized_emission(donor, e, ROOM, **kw)
    eps = molar_absorption(acceptor, e, ROOM, refractive_index=n, **kw)
    r_spec = forster_radius_from_spectra(e, em, eps, eta, refractive_index=n)
    np.testing.assert_allclose(r_golden, r_spec, rtol=1e-6)
    assert 1.0 < bohr_to_nm(r_golden) < 10.0


def test_forster_c6_refuses_a_coarse_or_uneven_grid():
    donor, acceptor = _bands()
    kw = dict(gaussian_fwhm=GAUSS, lorentzian_fwhm=LORENTZ)
    with pytest.raises(ValueError, match='resolves the narrowest'):
        forster_c6(donor, acceptor, ROOM,
                   energies=np.linspace(2.0, 3.0, 40) * EV, **kw)
    uneven = np.linspace(2.0, 3.0, 4000) ** 1.01 * EV
    with pytest.raises(ValueError, match='uniform'):
        forster_c6(donor, acceptor, ROOM, energies=uneven, **kw)


def test_many_lines_of_one_model_cost_about_one_band():
    """Sixty T1 -> Tn lines of one model: the result equals the sum of the
    lines taken one group each, and the shared group evaluates the band once."""
    donor, _ = _bands()
    model = DisplacedOscillators(S_A, MODES)
    calls = []

    class Counting(DisplacedOscillators):
        def density(self, *a, **k):
            calls.append(np.size(a[0]))
            return super().density(*a, **k)

    shared = Counting(S_A, MODES)
    ef = np.linspace(2.2, 5.0, 60) * EV
    mu = np.cos(np.arange(60.0))[None, :, None] * np.array([0.3, 0.2, 0.1])
    lines = excited_absorption_lines([0.0001 * EV], ef + 0.0001 * EV, mu,
                                     shared, ROOM)
    kw = dict(gaussian_fwhm=GAUSS, lorentzian_fwhm=LORENTZ)
    out = forster_c6(donor, lines, ROOM, **kw)
    assert len(calls) == 1
    e = out['energies']
    some = [0, 59]
    for i in some:
        own = Line(lines[i].energy, DisplacedOscillators(S_A, MODES),
                   strength=lines[i].strength)
        ref = forster_c6(donor, [own], ROOM, energies=e, **kw)
        np.testing.assert_allclose(out['line_overlaps'][i],
                                   ref["line_overlaps"][0], rtol=1e-7,
                                   atol=1e-12 * out['line_overlaps'].max())


def test_spectroscopic_radius_matches_the_textbook_constant():
    """R_F [nm] = 0.02108 (kappa^2 eta n^-4 J)^(1/6), J = int F(lam) eps(lam)
    lam^4 dlam in M^-1 cm^-1 nm^4 with F normalized on wavelength."""
    donor, acceptor = _bands()
    kw = dict(gaussian_fwhm=GAUSS, lorentzian_fwhm=LORENTZ)
    e = overlap_grid(donor, acceptor, ROOM, points_per_width=10, **kw)
    em = normalized_emission(donor, e, ROOM, **kw)
    eps = molar_absorption(acceptor, e, ROOM, refractive_index=1.5, **kw)
    eta, n = 0.8, 1.5
    r = bohr_to_nm(forster_radius_from_spectra(e, em, eps, eta,
                                               refractive_index=n))
    lam = HARTREE_WAVELENGTH_NM / e[::-1]
    f_lam = (em * e ** 2)[::-1]                 # I(lam) = I(E) |dE/dlam|
    f_lam /= TRAPEZOID(f_lam, lam)
    j = TRAPEZOID(f_lam * eps[::-1] * lam ** 4, lam)
    # 0.02108 nm to four digits; in full from the SI form of the equation,
    # R^6 = 9000 ln 10 kappa^2 eta J / (128 pi^5 N_A n^4) with J in cm^6/mol
    const_nm = 1e7 * (9000.0 * np.log(10.0) / (128.0 * np.pi ** 5
                                               * 6.02214076e23)
                      * 1e-28) ** (1.0 / 6.0)
    np.testing.assert_allclose(const_nm, 0.02108, rtol=1e-3)
    textbook = const_nm * (KAPPA2_ISOTROPIC * eta * n ** -4 * j) ** (1.0 / 6.0)
    np.testing.assert_allclose(r, textbook, rtol=1e-6)


def test_radius_does_not_depend_on_the_donor_dipole():
    donor, acceptor = _bands()
    kw = dict(gaussian_fwhm=GAUSS, lorentzian_fwhm=LORENTZ)
    radii = []
    for scale in (1.0, 1e-6):
        d = [Line(ln.energy, ln.model, strength=scale * ln.strength)
             for ln in donor]
        out = forster_c6(d, acceptor, ROOM, **kw)
        radii.append(forster_radius_yield(out['c6'], out['radiative_rate'],
                                          0.5))
    np.testing.assert_allclose(radii[0], radii[1], rtol=1e-10)


def test_isotropic_orientation_factor():
    """<kappa^2> over independent uniform orientations of both dipoles is
    2/3, kappa = u.v - 3 (u.r)(v.r)."""
    k = 4000
    i = np.arange(k) + 0.5
    z = 1.0 - 2.0 * i / k
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    u = np.stack([np.sqrt(1 - z ** 2) * np.cos(phi),
                  np.sqrt(1 - z ** 2) * np.sin(phi), z], 1)
    rng = np.random.default_rng(7)
    v = u[rng.permutation(k)]
    rhat = np.array([0.0, 0.0, 1.0])
    kappa = (u * v).sum(1) - 3.0 * (u @ rhat) * (v @ rhat)
    np.testing.assert_allclose((kappa ** 2).mean(), KAPPA2_ISOTROPIC,
                               rtol=2e-2)


def test_excited_absorption_lines_are_upward_and_thermal():
    model = DisplacedOscillators(S_A, MODES)
    ei = np.array([2.0, 2.05]) * EV
    ef = np.array([2.04, 3.0, 3.4]) * EV
    mu = np.arange(18, dtype=float).reshape(2, 3, 3) / 10.0
    lines = excited_absorption_lines(ei, ef, mu, model, ROOM)
    # 2.04 eV lies above 2.00 but below 2.05: five upward pairs
    assert len(lines) == 5
    pop = boltzmann_weights(ei, ROOM)
    np.testing.assert_allclose(pop.sum(), 1.0)
    np.testing.assert_allclose(pop[1] / pop[0],
                               np.exp(-0.05 * EV / (ROOM * 3.166811563e-6)))
    first = lines[0]
    np.testing.assert_allclose(first.energy, 0.04 * EV)
    np.testing.assert_allclose(first.strength, (mu[0, 0] ** 2).sum())
    np.testing.assert_allclose(first.weight, pop[0])
    with pytest.raises(ValueError, match='above'):
        excited_absorption_lines([3.5 * EV], ef, mu[:1], model, ROOM)
    with pytest.raises(ValueError, match='pairs'):
        excited_absorption_lines(ei, ef, mu[:, :2], model, ROOM)


def test_line_refuses_inconsistent_dipoles():
    model = DisplacedOscillators(S_A, MODES)
    with pytest.raises(ValueError, match='twice or not at all'):
        Line(2.0 * EV, model)
    with pytest.raises(ValueError, match='positive energy'):
        Line(-0.1, model, strength=1.0)
