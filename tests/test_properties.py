"""Gates for src/properties/ -- everything computed FROM a surface.

Two tiers. The cheap one needs no electronic structure at all: a hand-written
harmonic surface in the H-H distance of an H2 molecule exercises the protocol,
the optimizer, the finite-difference gradient wrapper and `refreeze` in
milliseconds, and the rate expressions are closed forms checked against a hand
evaluation. The expensive one is on real BSE@GW surfaces (CH2O/H2O, cc-pVDZ):
the normal modes against pyscf's own harmonic analysis, the two Huang-Rhys
routes bracketing the relaxation energy they must sum to, and the optimizer
converging on a state it can follow and diagnosing one it cannot.

Run the cheap tier alone with
    -k "toy or rate or rigid or align or sum_rule or adiabatic"
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.hessian import thermo

from src.Base.constants import (BOHR_TO_ANGSTROM, GEOM_OPT_CONV, HARTREE_TO_CM,
                                HARTREE_TO_EV)
from src.gradients.excited_state import ExcitedStateChain
from src.properties.optimize import optimize, translation_rotation_basis
from src.properties.rates import marcus_levich_jortner_rate, marcus_rate
from src.properties.surface import (FiniteDifferenceGradient,
                                    PotentialEnergySurface)
from src.properties.vibronic import (adiabatic_gap, align_to,
                                     huang_rhys_from_displacement,
                                     huang_rhys_from_gradient, normal_modes,
                                     relax_state,
                                     reorganization_from_huang_rhys)

BASIS = 'cc-pvdz'
CH2O = ('C 0.0000 0.0000 -0.5290; O 0.0000 0.0000 0.6746; '
        'H 0.0000 0.9376 -1.1188; H 0.0000 -0.9376 -1.1188')
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'


class HarmonicToySurface:
    """E = k (r - r_e)^2 / 2 in the bond length of a diatomic, and nothing else.

    A `PotentialEnergySurface` with no integrals behind it: the protocol's
    shape at the cost of a square root, so the optimizer, the finite-difference
    wrapper and `refreeze` can be gated without an SCF. The minimum is known
    exactly, which a real surface's never is.
    """

    def __init__(self, mol, k=0.4, r_e=1.6):
        self.mol0, self.k, self.r_e = mol, k, r_e

    def scf_factory(self, mol):
        """No mean field is involved; the surface is a function of the nuclei."""
        return None

    def _displacement(self, mol):
        mol = self.mol0 if mol is None else mol
        d = mol.atom_coords()[1] - mol.atom_coords()[0]
        r = float(np.linalg.norm(d))
        return r, d / r

    def total_energy(self, mol=None, mf=None):
        r, _ = self._displacement(mol)
        return 0.5 * self.k * (r - self.r_e) ** 2

    def total_gradient(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        r, u = self._displacement(mol)
        grad = np.zeros((mol.natm, 3))
        grad[1] = self.k * (r - self.r_e) * u
        grad[0] = -grad[1]
        return grad, self.total_energy(mol), {}

    def refreeze(self, mol):
        return type(self)(mol, k=self.k, r_e=self.r_e)

    def label(self):
        return f'harmonic toy, k = {self.k} Ha/Bohr^2, r_e = {self.r_e} Bohr'


def scf_factory(mol):
    """A mean field converged for GRADIENT work: conv_tol_grad 1e-11, because
    the Lagrangian assumes the occupied-virtual Fock block vanishes."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture
def h2_toy():
    """H2 stretched 0.4 Bohr inside a surface whose minimum is at 1.6 Bohr."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1.2', unit='Bohr', basis='sto-3g',
                verbose=0)
    return HarmonicToySurface(mol)


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    return mol, scf_factory(mol)


@pytest.fixture(scope='module')
def ch2o():
    mol = gto.M(atom=CH2O, basis=BASIS, verbose=0)
    return mol, scf_factory(mol)


# -------------------------------------------------------------- the protocol
def test_toy_surface_satisfies_the_protocol(h2_toy):
    """A surface is what has the five members, not what inherits from anything.

    The protocol is structural on purpose: the BSE chain, the dRPA chain and an
    embedded root are unrelated classes, and a property routine must not have to
    know which one it was handed.
    """
    assert isinstance(h2_toy, PotentialEnergySurface)
    assert isinstance(FiniteDifferenceGradient(h2_toy), PotentialEnergySurface)

    class NoRefreeze:
        mol0 = None

        def scf_factory(self, mol):
            return None

        def total_energy(self, mol=None, mf=None):
            return 0.0

        def total_gradient(self, mol=None, mf=None):
            return np.zeros((1, 3)), 0.0, {}

        def label(self):
            return 'incomplete'

    # An energy and a gradient are not enough: without `refreeze` the optimizer
    # cannot rebuild the frozen conventions and has no error bar on its minimum.
    assert not isinstance(NoRefreeze(), PotentialEnergySurface)


def test_toy_surface_refreeze_rebuilds_at_the_new_geometry(h2_toy):
    """`refreeze` returns the SAME class, at the new reference, same settings."""
    moved = h2_toy.mol0.copy()
    moved.set_geom_(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.55]]), unit='Bohr')
    moved.build(False, False)
    fresh = h2_toy.refreeze(moved)
    assert type(fresh) is type(h2_toy)
    assert fresh.mol0 is moved
    assert (fresh.k, fresh.r_e) == (h2_toy.k, h2_toy.r_e)
    # the energy is a property of the geometry, not of which object reports it
    assert fresh.total_energy() == pytest.approx(h2_toy.total_energy(moved),
                                                 rel=1e-14)


# ------------------------------------------------------------------ optimizer
def test_toy_optimizer_finds_the_known_minimum(h2_toy):
    """The analytic gradient walks an exactly quadratic surface to r = r_e."""
    opt, info = optimize(h2_toy, trust=0.3, max_cycle=40, verbose=False)
    assert info['converged'] and info['status'] == 'ok'
    assert info['omega'] is None, 'a ground-state surface has no excitation'
    r = np.linalg.norm(opt.atom_coords()[1] - opt.atom_coords()[0])
    assert r == pytest.approx(h2_toy.r_e, abs=1e-3)
    assert info['energy'] < h2_toy.total_energy()


def test_toy_optimizer_through_finite_differences(h2_toy):
    """The same minimum with the gradient differenced instead of derived.

    This is the route an ENERGY-ONLY surface takes -- the solvated chain today
    -- so it has to reach the same geometry, not merely run.
    """
    fd = FiniteDifferenceGradient(h2_toy, h=1e-3)
    g_fd, e_fd, diags = fd.total_gradient()
    g_exact = h2_toy.total_gradient()[0]
    assert e_fd == pytest.approx(h2_toy.total_energy(), rel=1e-14)
    assert diags['fd_step'] == 1e-3
    assert np.abs(g_fd - g_exact).max() < 1e-8

    opt, info = optimize(fd, trust=0.3, max_cycle=40, verbose=False)
    assert info['converged']
    r = np.linalg.norm(opt.atom_coords()[1] - opt.atom_coords()[0])
    assert r == pytest.approx(h2_toy.r_e, abs=1e-3)
    assert 'finite difference' in fd.label()


# ---------------------------------------------------------------------- rates
def test_rate_marcus_levich_jortner_against_a_hand_evaluation():
    """One parameter set, evaluated term by term outside the module.

    Poisson weights e^-S S^n/n! at S = 1 times a Gaussian activation factor per
    channel; the whole sum is a number, and pinning it catches a dropped 2 pi, a
    k_B in the wrong unit and an atomic-time conversion applied twice.
    """
    k = marcus_levich_jortner_rate(coupling=1e-6, delta_e=-0.02, lambda_m=0.01,
                                   s_eff=1.0, omega_eff=0.005, temperature=300.0)
    assert k == pytest.approx(1.0313579250e7, rel=1e-8)


def test_rate_reduces_to_marcus_when_the_quantum_mode_vanishes():
    """S -> 0 leaves one channel, and that channel IS the Marcus expression."""
    args = dict(coupling=2.5e-6, delta_e=-0.015, temperature=298.15)
    mlj = marcus_levich_jortner_rate(lambda_m=0.012, s_eff=0.0, omega_eff=0.006,
                                     **args)
    assert mlj == pytest.approx(marcus_rate(lambda_total=0.012, **args),
                                rel=1e-12)
    # A small but nonzero S must not reproduce it: the quantum channels are a
    # real physical effect, not a numerical correction.
    near = marcus_levich_jortner_rate(lambda_m=0.012, s_eff=1e-3,
                                      omega_eff=0.006, **args)
    assert near != pytest.approx(mlj, rel=1e-9)
    assert near == pytest.approx(mlj, rel=1e-2)


def test_rate_is_quadratic_in_the_coupling():
    """k scales with |V|^2, which is what makes a factor of three in a spin-orbit
    element an order of magnitude in the rate."""
    args = dict(delta_e=0.0008, lambda_m=0.01, s_eff=0.8, omega_eff=0.005,
                temperature=300.0)
    base = marcus_levich_jortner_rate(coupling=1e-6, **args)
    for scale in (2.0, 5.0, 0.5):
        assert marcus_levich_jortner_rate(coupling=scale * 1e-6,
                                          **args) == pytest.approx(
            scale ** 2 * base, rel=1e-12)
    # a complex spin-orbit element enters through its modulus only
    assert marcus_levich_jortner_rate(coupling=1j * 1e-6,
                                      **args) == pytest.approx(base, rel=1e-12)


def test_rate_refuses_a_truncated_poisson_sum():
    """A quantum mode with S = 40 needs more than 20 quanta; silently keeping
    half the Poisson weight would halve the rate."""
    with pytest.raises(ValueError, match='Poisson'):
        marcus_levich_jortner_rate(coupling=1e-6, delta_e=-0.02, lambda_m=0.01,
                                   s_eff=40.0, omega_eff=0.005,
                                   temperature=300.0, n_max=20)


# --------------------------------------------------------------- normal modes
def test_normal_modes_match_pyscf(ch2o):
    """Frequencies against pyscf's harmonic_analysis, same Hessian.

    Pinned because the mass convention is a silent offset: most-abundant-isotope
    masses instead of isotope-averaged raise every frequency, by up to 11.8
    cm^-1 here -- small, entirely systematic, and invisible without this.
    """
    mol, mf = ch2o
    hess = mf.Hessian().kernel()
    w, modes, _, _ = normal_modes(mf, mol, hess=hess)
    ref = np.sort(np.asarray(thermo.harmonic_analysis(mol, hess)
                             ['freq_wavenumber'], float).real)
    assert np.abs(np.sort(w * HARTREE_TO_CM) - ref).max() < 1e-2
    assert np.abs(modes.T @ modes - np.eye(modes.shape[1])).max() < 1e-10
    assert (w > 0).all(), 'the reference geometry should be a minimum'


def test_rigid_body_basis():
    """6 modes for a bent molecule, 5 for a linear one, orthonormal both times.

    An optimizer that does not project these out drifts and spins along them,
    and a normal-mode displacement projected without them picks up spurious
    weight on the softest modes -- the ones with the largest Huang-Rhys factors.
    """
    bent = np.array([[0, 0, 0.117], [0, 0.757, -0.468], [0, -0.757, -0.468]])
    lin = np.array([[0, 0, -1.0], [0, 0, 0.0], [0, 0, 1.0]])
    for coords, expect in ((bent, 6), (lin, 5)):
        b = translation_rotation_basis(coords)
        assert b.shape[0] == expect
        assert np.abs(b @ b.T - np.eye(expect)).max() < 1e-12


def test_align_to_undoes_a_rigid_motion():
    """Superposition must remove exactly a rotation plus a translation."""
    mol = gto.M(atom=CH2O, basis='sto-3g', verbose=0)
    th = 0.4
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0],
                    [0, 0, 1.0]])
    moved = mol.copy()
    moved.set_geom_(mol.atom_coords() @ rot.T + np.array([0.3, -0.2, 0.7]),
                    unit='Bohr')
    moved.build(False, False)
    back = align_to(mol, moved)
    assert np.abs(back.atom_coords() - mol.atom_coords()).max() < 1e-10


# ----------------------------------------------------------------- Huang-Rhys
def test_huang_rhys_sum_rule_on_a_displaced_oscillator():
    """S_k and lambda = sum_k S_k hbar omega_k on a case with a known answer.

    A harmonic surface displaced by a known amount: both routes must return the
    same S, because the linear model is EXACT when the curvature does not
    change. That is the control for the real-molecule case, where they differ
    and the difference measures the curvature change rather than an error.
    """
    n = 3
    omega = np.array([0.01, 0.02, 0.005])
    modes = np.eye(3 * n)[:, :3]
    masses = np.ones(n) * 1000.0
    dq = np.array([0.7, -0.3, 1.1])                     # mass-weighted
    s_ref = omega * dq ** 2 / 2
    grad = np.zeros(3 * n)
    grad[:3] = -(omega ** 2 * dq) / np.repeat(masses, 3)[:3] ** -0.5
    s_g, _ = huang_rhys_from_gradient(grad.reshape(n, 3), omega, modes, masses)
    assert np.allclose(s_g, s_ref, rtol=1e-10)
    assert reorganization_from_huang_rhys(s_ref, omega) == pytest.approx(
        float((s_ref * omega).sum()), rel=1e-12)


def test_huang_rhys_routes_bracket_the_relaxation_energy(ch2o):
    """On a real state the two routes BRACKET the true relaxation energy.

    lambda(gradient) uses the true gradient with ground-state curvature;
    lambda(displacement) uses the true displacement with ground-state
    curvature. When the excited surface is softer than the ground one the first
    under- and the second over-estimates, and the true relaxation energy --
    computed from two total energies, with no modes at all -- sits between.
    Measured on formaldehyde S1: 0.326 / 0.398 / 0.592 eV.
    """
    mol, mf = ch2o
    ch = ExcitedStateChain(mol, scf_factory, spin='singlet', mf=mf)
    w, modes, masses, _ = normal_modes(mf, mol)
    g_fc, e_vert, _ = ch.total_gradient()
    s_g, _ = huang_rhys_from_gradient(g_fc, w, modes, masses)
    lam_g = reorganization_from_huang_rhys(s_g, w)
    st = relax_state(ch, engine='cartesian', verbose=False)
    lam_relax = e_vert - st['e_total']
    s_d, _ = huang_rhys_from_displacement(mol, align_to(mol, st['mol']), w,
                                          modes, masses)
    lam_d = reorganization_from_huang_rhys(s_d, w)
    assert lam_relax > 0
    assert lam_g < lam_relax < lam_d
    # The C=O stretch must carry the progression for an n->pi* state. 1892
    # cm^-1 is that mode at THIS geometry (the fixture's, unrelaxed); it is
    # 2013 at the relaxed S0 minimum, and quoting the one into the other is
    # exactly the mistake this pin exists to catch.
    assert w[np.argmax(s_g)] * HARTREE_TO_CM == pytest.approx(1892, abs=60)


# --------------------------------------------------- optimizer on a real state
def test_optimizer_relaxes_and_lowers_the_energy(ch2o):
    """Formaldehyde S1: converges, lengthens C=O, never raises the energy."""
    mol, mf = ch2o
    ch = ExcitedStateChain(mol, scf_factory, spin='singlet', mf=mf)
    opt, info = optimize(ch, max_cycle=30, verbose=False)
    assert info['converged'] and info['status'] == 'ok'
    assert info['grad_max'] < GEOM_OPT_CONV['grad_max']
    e = [h['e'] for h in info['history']]
    assert e[-1] < e[0]
    rco = lambda m: (np.linalg.norm(m.atom_coords()[1] - m.atom_coords()[0])
                     * BOHR_TO_ANGSTROM)
    assert rco(opt) - rco(mol) == pytest.approx(0.085, abs=0.02)
    # an excited surface reports its excitation energy alongside the total
    assert info['omega'] * HARTREE_TO_EV > 0


def test_optimizer_reports_a_state_it_cannot_follow(water):
    """Water's S1 is DISSOCIATIVE; the run must end with a diagnosis, not a
    traceback, and must not claim convergence."""
    mol, mf = water
    ch = ExcitedStateChain(mol, scf_factory, spin='singlet', mf=mf)
    _, info = optimize(ch, max_cycle=12, trust=0.5, trust_max=0.8,
                       verbose=False)
    assert not info['converged']
    assert info['status'] != 'ok'      # either a diagnosis or the cycle limit
    assert info['history'], 'the geometries walked are still the useful output'


def test_adiabatic_gap_uses_each_state_own_minimum():
    """Delta-E_ST is a difference of TOTAL energies, so E_0 does not cancel."""
    a = {'e_total': -113.72080728}
    b = {'e_total': -113.74816714}
    assert adiabatic_gap(a, b) == pytest.approx(0.02735986, abs=1e-8)
