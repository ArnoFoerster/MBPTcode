"""The fragment partition of the folded ADC(2) singles operator, singlets and
triplets, and the analytic gradient of its diabatic matrix, against exact
references.

The offset stacked ethylene dimer of tests/test_fragment_diabatic.py (4 A
apart, slipped), cc-pVDZ, DF-RHF in cc-pVDZ-RI, ADC(2) with the doubles in
the same fit, no frozen core; one site state per monomer unless stated. The
partition and its gradient are the method-neutral code GW-BSE is gated on
(`src.fragments.partition`, `src.gradients.fragment_diabatic`); what is
ADC(2)'s own is the operator K(Omega) = M + C (Omega - D)^-1 C^T, its Omega
derivative, and the reverse pass (`src.gradients.adc.fragment_adc2`).

WHAT EACH GATE IS FOR:

- Loewdin-PM on Hartree-Fock orbitals still gives clean fragments: HF
  virtuals are more diffuse than Kohn-Sham ones, and the higher virtuals are
  where localization is weakest. Every local orbital, occupied and virtual,
  sits on one fragment to 0.998 (floor LOCALIZED_ASSIGNMENT_FLOOR), the
  virtuals split evenly.
- the matrix-free fold is the dense K(Omega_0), and its dK/dOmega is the
  central difference of K over Omega.
- the partition is exact at Omega_0: Sigma equals the dense Feshbach oracle
  on K(Omega_0), and the dense and the matrix-free iterative paths agree.
- dA_eff/dOmega is the central difference of the dense Feshbach
  elimination of K(Omega) over Omega_0, on the diabats of Omega_0. This is
  the gate for the fold's own slope u^T (dK/dOmega) u, which an
  energy-independent kernel does not have; the test also shows that slope is
  large here (the gate fails by orders of magnitude without it).
- SELF-CONSISTENT ROOTS: built AT a full ADC(2) root (Omega_0 = omega_n),
  A_eff(omega_n) has omega_n as an eigenvalue, and Z = [1 - c^T dA_eff/dOmega
  c]^-1 is the root's weight on P in the whole singles + doubles norm (its
  singles weight t1 times the P part of its unit singles). Eliminating the
  doubles and the other singles at once is exact only with the fold's slope.
- the ROOT gradient through the partition equals the supermolecular ADC(2)
  root's (`ee_gradient.adc2_force` less its HF + MP2 ground state), with the
  gauge and localization terms vanishing for it: the gate for the reverse
  pass of the pairs and the diabat responses.
- each ELEMENT's gradient along a random direction equals the relocalized
  finite difference (`tests/fragment_diabatic_fd.py`): the gate for the
  symmetric-gauge and the localization terms, singlets and triplets.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.fragments import partition as fragment_partition
from src.fragments.localization import FragmentOrbitals
from src.fragments.partition import FragmentPartition
from src.gradients.adc.ee_gradient import adc2_force, solve_root
from src.gradients.adc.fragment_adc2 import ADC2FragmentMethod, ADC2Operator
from src.gradients.fragment_diabatic import DiabaticGradient
from tests.feshbach_oracle import FeshbachOracle
from tests.fragment_diabatic_fd import derivative_ladder
from tests.test_fragment_diabatic import DIMER, FRAGMENTS, SITES, random_direction

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
BOTH_CT = {(0, 1): 1, (1, 0): 1}
#: Omega_0 per channel, below the lowest root. Full ADC(2) roots of the dimer
#: (Ha): singlets 0.305438, 0.325998, 0.329640; triplets 0.167330, 0.168960,
#: 0.314355. Q's lowest eigenvalue with one site state per monomer: 0.331265
#: (singlet), 0.320450 (triplet); the pole guard asks Omega_0 below it by
#: FRAGMENT_POLE_MARGIN (0.01 Ha), so the singlet root 0 and both triplet
#: roots can be Omega_0 themselves.
OMEGA0 = {'singlet': 0.29, 'triplet': 0.16}
#: The roots taken as Omega_0 in the self-consistency gate.
SC_ROOTS = {'singlet': (0,), 'triplet': (0, 1)}
SPINS = ('singlet', 'triplet')


def factory(mol):
    mf = scf.RHF(mol).density_fit(AUX)
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-13, 1e-11, 0
    mf.max_cycle = 300
    return mf.run()


def force_iterative(monkeypatch):
    """Every block, guard, resolvent and response iterative: both dense
    limits (`partition.dense_limit`) below the smallest block here."""
    monkeypatch.setattr(fragment_partition, 'FRAGMENT_DENSE_MAX', 10)
    monkeypatch.setattr(fragment_partition, 'FRAGMENT_MATRIX_FREE_DENSE_MAX', 10)


class Setup:
    """One spin channel's method, roots, orbitals and dense operator."""

    def __init__(self, spin):
        self.spin = spin
        self.mol0 = gto.M(atom=DIMER, basis=BASIS, verbose=0)
        self.method = ADC2FragmentMethod(self.mol0, factory, spin, AUX)
        self.mol, self.mf = self.method.mean_field()
        _, _, self.roots = solve_root(self.mol, self.mf, spin, 3, 0, AUX)
        self.orbitals = FragmentOrbitals.from_mf(self.mf, FRAGMENTS)
        self.omega0 = OMEGA0[spin]
        self.dense = ADC2Operator.from_mean_field(self.mol, self.mf, spin, AUX,
                                                  self.omega0, route='dense')
        self.partition = FragmentPartition.build(self.dense, self.orbitals,
                                                 SITES, omega0=self.omega0)

    def operator(self, omega, route='fold'):
        return ADC2Operator.from_mean_field(self.mol, self.mf, self.spin, AUX,
                                            omega, route=route)


SETUPS = {}


@pytest.fixture(scope='module', params=SPINS)
def setup(request):
    if request.param not in SETUPS:
        SETUPS[request.param] = Setup(request.param)
    return SETUPS[request.param]


def test_hartree_fock_loewdin_pm_gives_clean_fragments(setup):
    o = setup.orbitals
    s = setup.mol0.intor_symmetric('int1e_ovlp')
    for c in (o.c_occ, o.c_vir):
        assert np.abs(c.T @ s @ c - np.eye(c.shape[1])).max() < 1e-12
    assert max(g[0] for g in o.pm_gradient.values()) < 1e-8
    # 0.99893 for both spaces at this geometry
    assert min(o.occ_weights.min(), o.vir_weights.min()) > 0.998
    n_occ, n_vir = o.counts()
    assert list(n_occ) == [8, 8] and list(n_vir) == [40, 40]


def test_the_fold_is_the_dense_operator_and_its_omega_derivative(setup):
    fold = setup.operator(setup.omega0)
    x = np.random.default_rng(5).normal(size=(fold.n_ov, 3))
    k = setup.dense.dense
    assert np.abs(fold.apply(x) - k @ x).max() < 1e-12 * np.abs(k).max() * 10
    h = 1e-4
    plus, minus = (setup.operator(setup.omega0 + s * h) for s in (1.0, -1.0))
    fd = (plus.apply(x) - minus.apply(x)) / (2 * h)
    dk = fold.apply_domega(x)
    assert np.abs(dk - fd).max() < 1e-6 * np.abs(dk).max()
    # dK/dOmega = -C (Omega - D)^-2 C^T is negative semidefinite
    assert np.all(np.einsum('ik,ik->k', x, dk) < 0.0)


def test_operator_refusals(setup):
    with pytest.raises(ValueError, match='depends on Omega'):
        setup.operator(None)
    with pytest.raises(ValueError, match='lowest doubles energy'):
        setup.operator(setup.dense.fold.d_min)
    with pytest.raises(ValueError, match='another energy'):
        FragmentPartition.build(setup.dense, setup.orbitals, SITES,
                                omega0=setup.omega0 + 0.01)


def test_partition_is_the_exact_feshbach_elimination(setup, monkeypatch):
    part = setup.partition
    t = setup.orbitals.transition_rotation()
    k_loc = t.T @ setup.dense.dense @ t
    oracle = FeshbachOracle(k_loc, part.p_local)
    assert np.abs(oracle.sigma(part.omega0) - part.sigma).max() < 1e-12
    assert np.abs(oracle.a_pp - part.a_pp).max() < 1e-12
    # the matrix-free iterative path: Davidson diabats, guard, conjugate
    # gradients on the fold's products
    force_iterative(monkeypatch)
    iterative = FragmentPartition.build(setup.operator(setup.omega0),
                                        setup.orbitals, SITES,
                                        omega0=setup.omega0)
    for name in ('a_eff', 'dsigma'):
        a, b = getattr(iterative, name), getattr(part, name)
        assert np.abs(np.abs(a) - np.abs(b)).max() < 1e-8, name
    # The guard's one-root Davidson bounds Q's lowest eigenvalue from above.
    # Q's two lowest are a near-degenerate pair of local states, one per
    # monomer (triplets: 0.320171 and 0.320358 Ha),
    # and it may land on the upper one: the guard then errs by the splitting,
    # far inside FRAGMENT_POLE_MARGIN.
    assert -1e-8 < iterative.q_lowest - part.q_lowest < 2e-4


def dense_folds(setup, omegas):
    """[K(w) for w in omegas], whole, in ONE pass of the doubles per column
    (`RIFoldADC2.fold_images`: only the denominator changes with w)."""
    fold = setup.dense.fold
    n = setup.dense.n_ov
    out = [np.empty((n, n)) for _ in omegas]
    for c0 in range(0, n, 64):
        c1 = min(c0 + 64, n)
        cols = np.zeros((n, c1 - c0))
        cols[np.arange(c0, c1), np.arange(c1 - c0)] = 1.0
        m, f = fold.fold_images(cols, list(omegas))
        for k in range(len(omegas)):
            out[k][:, c0:c1] = m + f[k]
    return [0.5 * (k + k.T) for k in out]


def test_the_slope_is_the_omega_derivative_of_a_eff(setup):
    """dA_eff/dOmega, every element, against the central difference of the
    dense Feshbach elimination of K(Omega) at Omega_0 +- h, both on the
    diabats of Omega_0 (the slope holds P fixed, as the root's Z needs)."""
    part = setup.partition
    t = setup.orbitals.transition_rotation()
    h = 1e-3
    oms = (part.omega0 - 2 * h, part.omega0 - h, part.omega0 + h,
           part.omega0 + 2 * h)
    a = []
    for om, k in zip(oms, dense_folds(setup, oms)):
        o = FeshbachOracle(t.T @ k @ t, part.p_local)
        a.append(o.a_pp + o.sigma(om))
    # fourth-order central difference
    fd = (a[0] - 8.0 * a[1] + 8.0 * a[2] - a[3]) / (12.0 * h)
    assert np.abs(fd - part.dsigma).max() < 1e-8
    # the fold's own slope u^T K' u is a large part of it: leaving it out
    # fails the gate by orders of magnitude
    u = part.u_local
    du = setup.orbitals.to_local(setup.dense.apply_domega(
        setup.orbitals.to_canonical(u)))
    assert np.abs(np.diag(u.T @ du)).min() > 1e-3


@pytest.mark.parametrize('spin', SPINS)
def test_self_consistent_roots_are_the_full_adc2_roots(spin):
    s = SETUPS.get(spin) or SETUPS.setdefault(spin, Setup(spin))
    y_all = np.asarray(s.roots['y'])
    for n in SC_ROOTS[spin]:
        omega, t1 = float(s.roots['omega'][n]), float(s.roots['t1'][n])
        part = FragmentPartition.build(s.operator(omega), s.orbitals, SITES,
                                       omega0=omega)
        w, c = np.linalg.eigh(part.a_eff)
        k = int(np.argmin(np.abs(w - omega)))
        assert abs(w[k] - omega) < 1e-9, (spin, n)
        z = 1.0 / (1.0 - float(c[:, k] @ part.dsigma @ c[:, k]))
        y = s.orbitals.to_local(y_all[:, [n]])[:, 0]
        y /= np.linalg.norm(y)
        assert abs(z - t1 * np.sum((part.p_local.T @ y) ** 2)) < 1e-8, (spin, n)


def test_root_gradient_is_the_supermolecular_root_gradient():
    s = SETUPS.get('singlet') or SETUPS.setdefault('singlet', Setup('singlet'))
    omega = float(s.roots['omega'][0])
    ref, info = adc2_force(s.mf, s.mol, 'singlet', root=0, nroots=3,
                           auxbasis=AUX)
    assert abs(info['omega'] - omega) < 1e-10
    ref = ref - s.method.ground_gradient(s.mf)
    grad = DiabaticGradient(s.method, FRAGMENTS, SITES, omega0=omega)
    g, d = grad.root_gradient(omega)
    assert np.abs(g - ref).max() < 1e-8 * max(1.0, np.abs(ref).max())
    assert np.abs(d['canonical']).max() < 1e-10
    assert np.abs(d['localization']).max() < 1e-10


@pytest.mark.parametrize('spin, sites, ct', [('singlet', SITES, BOTH_CT),
                                             ('triplet', SITES, None)])
def test_element_gradients_match_finite_differences(spin, sites, ct):
    s = SETUPS.get(spin) or SETUPS.setdefault(spin, Setup(spin))
    direction = random_direction(s.mol0.natm, 7)
    grad = DiabaticGradient(s.method, FRAGMENTS, sites, ct, omega0=s.omega0)
    ladder = derivative_ladder(s.method, FRAGMENTS, sites, direction, ct,
                               omega0=s.omega0, steps=(1e-3, 5e-4))
    fd = ladder['richardson'][-1]
    scale = np.abs(fd).max()
    n = len(grad.partition.labels)
    for a in range(n):
        for b in range(a, n):
            g, d = grad.element(a, b)
            an = float((g * direction).sum())
            assert abs(an - fd[a, b]) < 2e-5 * scale, (spin, a, b, an, fd[a, b])
            assert np.abs(d['localization']).max() > 0.0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
