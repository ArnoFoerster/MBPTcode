"""The fragment-partitioned BSE, Tamm-Dancoff and full, and the analytic
gradient of its diabatic matrix, against exact references.

An offset stacked ethylene dimer (4 A apart, slipped so that no symmetry makes
a term vanish by accident), cc-pVDZ, RHF, BSE@G0W0 on the ISDF/space-time chain,
singlet; one site state per monomer unless stated. The Tamm-Dancoff gates come
first; the full-BSE ones repeat the decisive ones for the pencil form.

WHAT EACH GATE IS FOR:

- the localization is orthonormal and stationary, every local orbital sits on
  one fragment, and the diabatic matrix does not change when the local
  orbitals are rotated INSIDE a fragment: the diabats depend on the fragment
  subspaces only. That invariance is what lets the analytic gradient ignore
  the near-flat intra-fragment directions of the localization functional.
  It holds SIGNED, couplings included, and under a local orbital's sign
  flip too: the diabats' phase is their transition density's, which
  ethylene's pi* (equal and opposite coefficients on its two carbons) cannot
  flip.
- the partition is exact: Sigma and dSigma/dOmega equal a dense Feshbach
  oracle, the roots of A_eff(Omega) c = Omega c with complete Q equal the full
  TDA eigenvalues, and Z equals the exact P weight of the full eigenvector.
- the matrix-free ISDF action is the dense matrix, and the iterative path
  (Davidson, conjugate gradients) gives the dense path's partition.
- the ROOT gradient through the partition equals the supermolecular TDA root
  gradient. A root does not depend on the localization, so this gate tests the
  chain seed, the resolvent vectors and the diabat response without the
  orbital-rotation terms -- and checks that those terms vanish for it.
- each ELEMENT's gradient, projected on a random direction that distorts both
  monomers, equals the finite difference with relocalization at every
  displaced geometry. This is the gate for the canonical-gauge and the
  localization terms, which are several percent of the element's gradient
  here and which a root cannot see. The off-diagonal element is compared up to
  its sign, which is a convention of the diabats' phases.
- the same, in a field of fixed point charges, with both charge-transfer
  diabats explicit, for the dressed A_eff and the bare A_PP. The charges make
  the site-charge-transfer couplings and their localization terms large,
  which exposes an error in the localization response (a metric term
  counted twice shows in exactly those terms and nothing else).
- a charge-transfer diabat's SURFACE (ground state plus diabat), the object an
  optimizer relaxes, against a central difference of its own energy.
- the linear vibronic coupling model is the element gradients projected on
  the modes, symmetric in the two diabats.
- the iterative path (Davidson diabats, conjugate-gradient resolvent, MINRES
  responses) gives the dense path's gradient.
- TWO STATES IN ONE BLOCK, in both kernels: every element, including the
  coupling between the two states of a block, against finite differences.
  The diabats' rotation among themselves is part of the diabat response, and
  a single state per block cannot see it.

FULL BSE (the pencil K v = Omega S v, K = [[A, B], [B, A]], S = diag(1, -1)):

- the partition is exact: Sigma and dSigma/dOmega equal the dense oracle on
  the complement orthogonal in the metric, the roots of A_eff(Omega) with
  complete Q equal the full-BSE eigenvalues, and Z equals the exact P weight
  (X and Y alike) of the full eigenvector.
- the ISDF action is the dense [[A, B], [B, A]], and the iterative path
  ((A + B, A - B) Davidson diabats, conjugate-gradient resolvent, Newton on
  the pole guard) gives the dense partition.
- the root gradient through the partition equals the supermolecular full-BSE
  root gradient, with the orbital-rotation terms vanishing.
- the elements in point charges, both charge-transfer diabats explicit,
  against finite differences, and the iterative gradient against the dense
  one.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dataclasses

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base import constants
from src.Base.environment import PointCharges
from src.Base.fragment_localization import FragmentOrbitals
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.fragment_diabatic import (DiabaticGradient, DiabaticSurface,
                                             linear_vibronic_coupling)
from src.properties import fragment_bse
from src.properties.diabatic import displaced_along, diabatic_derivative, step_ladder
from src.gradients.bse_isdf import bse_solve
from src.properties.fragment_bse import (BSEOperator, FeshbachOracle,
                                         FragmentPartition, to_local)

ETHYLENE = [('C', (0, 0, 0.667)), ('C', (0, 0, -0.667)),
            ('H', (0, 0.923, 1.238)), ('H', (0, -0.923, 1.238)),
            ('H', (0, 0.923, -1.238)), ('H', (0, -0.923, -1.238))]
DIMER = ETHYLENE + [(s, (x + 4.0, y + 0.3, z + 0.2)) for s, (x, y, z) in ETHYLENE]
FRAGMENTS = [range(6), range(6, 12)]
SITES = {0: 1, 1: 1}
BOTH_CT = {(0, 1): 1, (1, 0): 1}
#: Charges (Angstrom) that pull one charge-transfer diabat down; Omega_0 is set
#: below every diabat, because in this field a site moves up past the mean.
CHARGES = ([[-5.0, 0, 0], [9.0, 0, 0], [2.0, 6.0, 0]], [0.3, -0.3, 0.2])
OMEGA0_CHARGED = 0.31
BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'


def factory(mol):
    mf = scf.RHF(mol).density_fit(AUX)
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-13, 1e-11, 0
    mf.max_cycle = 300
    return mf.run()


@pytest.fixture(scope='module')
def chain():
    mol = gto.M(atom=DIMER, basis=BASIS, verbose=0)
    return ExcitedStateChain(mol, factory, bse_tda=True, auxbasis=AUX)


@pytest.fixture(scope='module')
def charged_chain():
    mol = gto.M(atom=DIMER, basis=BASIS, verbose=0)
    return ExcitedStateChain(mol, factory, bse_tda=True, auxbasis=AUX,
                             environment=PointCharges(*CHARGES))


@pytest.fixture(scope='module')
def full_chain():
    mol = gto.M(atom=DIMER, basis=BASIS, verbose=0)
    return ExcitedStateChain(mol, factory, bse_tda=False, auxbasis=AUX)


@pytest.fixture(scope='module')
def charged_full_chain():
    mol = gto.M(atom=DIMER, basis=BASIS, verbose=0)
    return ExcitedStateChain(mol, factory, bse_tda=False, auxbasis=AUX,
                             environment=PointCharges(*CHARGES))


@pytest.fixture(scope='module')
def gradient(chain):
    return DiabaticGradient(chain, FRAGMENTS, SITES)


def force_iterative(monkeypatch):
    """Every block, guard, resolvent and response iterative: both dense
    limits (`fragment_bse.dense_limit`) below the smallest block here."""
    monkeypatch.setattr(fragment_bse, 'FRAGMENT_DENSE_MAX', 10)
    monkeypatch.setattr(fragment_bse, 'FRAGMENT_MATRIX_FREE_DENSE_MAX', 10)


def random_direction(natm, seed):
    d = np.random.default_rng(seed).normal(size=(natm, 3))
    return d / np.linalg.norm(d)


@pytest.fixture(scope='module')
def operator(chain):
    return BSEOperator.from_chain(chain, route='dense')[0]


@pytest.fixture(scope='module')
def orbitals(chain):
    return FragmentOrbitals.from_mf(chain.mf0, FRAGMENTS)


@pytest.fixture(scope='module')
def partition(operator, orbitals):
    return FragmentPartition.build(operator, orbitals, SITES)


def test_localization_is_orthonormal_stationary_and_assigned(chain, orbitals):
    s = chain.mol0.intor_symmetric('int1e_ovlp')
    for c in (orbitals.c_occ, orbitals.c_vir):
        assert np.abs(c.T @ s @ c - np.eye(c.shape[1])).max() < 1e-12
    # between fragments; inside one the functional is nearly flat and unused
    assert max(g[0] for g in orbitals.pm_gradient.values()) < 1e-8
    assert min(orbitals.occ_weights.min(), orbitals.vir_weights.min()) \
        >= constants.LOCALIZED_ASSIGNMENT_FLOOR
    n_occ, n_vir = orbitals.counts()
    assert list(n_occ) == [8, 8]


def test_diabats_depend_on_the_fragment_subspaces_only(operator, orbitals,
                                                       partition):
    rng = np.random.default_rng(3)

    def rotate(u, labels):
        u = u.copy()
        for f in np.unique(labels):
            idx = np.flatnonzero(labels == f)
            q = np.linalg.qr(rng.normal(size=(idx.size, idx.size)))[0]
            u[:, idx] = u[:, idx] @ q
        return u
    turned = dataclasses.replace(
        orbitals, u_occ=rotate(orbitals.u_occ, orbitals.occ_labels),
        u_vir=rotate(orbitals.u_vir, orbitals.vir_labels))
    # a local orbital's sign is what symmetry leaves to the last bits
    # (ethylene's pi*): fragment 0's virtuals flipped
    flipped = dataclasses.replace(
        orbitals, u_vir=orbitals.u_vir * np.where(orbitals.vir_labels == 0,
                                                  -1.0, 1.0))
    for case, orbs in (('rotated', turned), ('flipped', flipped)):
        other = FragmentPartition.build(operator, orbs, SITES,
                                        omega0=partition.omega0)
        # signed: the diabats' phase is their transition density's
        # (`fragment_bse.diabat_phase`), which no local orbital's sign moves
        for name, a, b in (('a_eff', other.a_eff, partition.a_eff),
                           ('sigma', other.sigma, partition.sigma),
                           ('dsigma', other.dsigma, partition.dsigma),
                           ('p', other.p_canonical(),
                            partition.p_canonical())):
            assert np.abs(a - b).max() < 1e-10, (case, name)


def test_partition_is_the_exact_feshbach_elimination(operator, orbitals,
                                                     partition):
    t = orbitals.transition_rotation()
    a_loc = t.T @ operator.dense @ t
    oracle = FeshbachOracle(a_loc, partition.p_local)
    assert np.abs(oracle.sigma(partition.omega0) - partition.sigma).max() < 1e-12
    assert np.abs(oracle.dsigma(partition.omega0)
                  - partition.dsigma).max() < 1e-12
    w, x = np.linalg.eigh(operator.dense)
    for guess in np.linalg.eigvalsh(partition.a_eff):
        om, _, z = oracle.root(guess)
        k = int(np.argmin(np.abs(w - om)))
        assert abs(om - w[k]) < 1e-10
        p_weight = np.sum((partition.p_local.T
                           @ orbitals.to_local(x[:, [k]])[:, 0]) ** 2)
        assert abs(z - p_weight) < 1e-10


def test_isdf_action_and_iterative_path_match_dense(chain, operator, orbitals,
                                                    partition, monkeypatch):
    isdf = BSEOperator.from_chain(chain, route='isdf')[0]
    x = np.random.default_rng(5).normal(size=(operator.n_ov, 3))
    assert np.abs(isdf.apply(x) - operator.dense @ x).max() < 1e-10
    force_iterative(monkeypatch)
    iterative = FragmentPartition.build(isdf, orbitals, SITES,
                                        omega0=partition.omega0)
    assert np.abs(np.diag(iterative.a_eff)
                  - np.diag(partition.a_eff)).max() < 1e-8
    assert np.abs(np.abs(iterative.a_eff)
                  - np.abs(partition.a_eff)).max() < 1e-8


def test_root_gradient_is_the_supermolecular_root_gradient(chain, operator):
    ref, info = chain.excitation_gradient()
    omega = np.linalg.eigvalsh(operator.dense)[0]
    assert abs(info['omega'] - omega) < 1e-10
    grad = DiabaticGradient(chain, FRAGMENTS, SITES, omega0=omega)
    g, d = grad.root_gradient(omega)
    assert np.abs(g - ref).max() < 1e-8 * max(1.0, np.abs(ref).max())
    assert np.abs(d['canonical']).max() < 1e-10
    assert np.abs(d['localization']).max() < 1e-10


def test_element_gradients_match_finite_differences(chain, gradient):
    direction = random_direction(chain.mol0.natm, 7)
    ladder = step_ladder(chain, FRAGMENTS, SITES, direction,
                         omega0=gradient.partition.omega0,
                         steps=(1e-3, 5e-4))
    fd = ladder['richardson'][-1]
    scale = np.abs(fd).max()
    for a, b in ((0, 0), (1, 1), (0, 1)):
        g, d = gradient.element(a, b)
        an = float((g * direction).sum())
        if a == b:
            assert abs(an - fd[a, b]) < 2e-5 * scale
        else:
            assert abs(abs(an) - abs(fd[a, b])) < 2e-5 * scale
        assert np.abs(d['localization']).max() > 0.0


@pytest.mark.parametrize('dressed', [True, False])
def test_elements_in_point_charges_match_finite_differences(charged_chain,
                                                            dressed):
    direction = random_direction(charged_chain.mol0.natm, 11)
    grad = DiabaticGradient(charged_chain, FRAGMENTS, SITES, BOTH_CT,
                            omega0=OMEGA0_CHARGED, dressed=dressed)
    key = 'a_eff' if dressed else 'a_pp'
    fd = [diabatic_derivative(charged_chain, FRAGMENTS, SITES, direction,
                              BOTH_CT, omega0=OMEGA0_CHARGED, step=h)[0][key]
          for h in (5e-4, 2.5e-4)]
    fd = (4.0 * fd[1] - fd[0]) / 3.0
    scale = np.abs(fd).max()
    n = len(grad.partition.labels)
    for a in range(n):
        for b in range(a, n):
            an = float((grad.element(a, b)[0] * direction).sum())
            assert abs(abs(an) - abs(fd[a, b])) < 1e-6 * scale, (a, b)


def test_charge_transfer_diabat_surface_gradient(charged_chain):
    direction = random_direction(charged_chain.mol0.natm, 13)
    surf = DiabaticSurface(charged_chain, FRAGMENTS, SITES, 'ct 0->1.0',
                           BOTH_CT, omega0=OMEGA0_CHARGED)
    g, e, _ = surf.total_gradient()
    mol0 = charged_chain.mol0

    def central(h):
        return (surf.total_energy(displaced_along(mol0, direction, h))
                - surf.total_energy(displaced_along(mol0, direction, -h))) / (2 * h)
    # Richardson over (1e-3, 5e-4): a single central difference leaves an h^2
    # error of a few 1e-5 relative on a direction where the slope is small
    fd = (4.0 * central(5e-4) - central(1e-3)) / 3.0
    an = float((g * direction).sum())
    assert abs(an - fd) < 1e-6 * np.abs(g).max()
    assert abs(e - surf.total_energy()) < 1e-10


def test_linear_vibronic_coupling_is_the_projected_gradients(chain, gradient):
    natm = chain.mol0.natm
    rng = np.random.default_rng(17)
    modes = np.linalg.qr(rng.normal(size=(3 * natm, 4)))[0]
    masses = np.ones(natm)
    lvc = linear_vibronic_coupling(gradient, modes, masses)
    assert np.allclose(lvc['energies'], gradient.partition.a_eff)
    assert np.abs(lvc['kappa'] - lvc['kappa'].transpose(1, 0, 2)).max() == 0.0
    g01 = gradient.element(0, 1)[0]
    assert np.abs(lvc['kappa'][0, 1] - modes.T @ g01.ravel()).max() < 1e-12


def test_iterative_gradient_matches_dense(chain, gradient, monkeypatch):
    dense = {ab: gradient.element(*ab)[0] for ab in ((0, 0), (0, 1))}
    force_iterative(monkeypatch)
    iterative = DiabaticGradient(chain, FRAGMENTS, SITES, route='isdf',
                                 omega0=gradient.partition.omega0)
    for ab, ref in dense.items():
        g = iterative.element(*ab)[0]
        assert min(np.abs(g - ref).max(), np.abs(g + ref).max()) \
            < 1e-7 * np.abs(ref).max()


#: Two states in each site block; Omega_0 near the lowest site states, since
#: the default (their mean) comes within the margin of Q's lowest state.
TWO_PER_SITE = {0: 2, 1: 2}
OMEGA0_TWO = 0.30


def check_elements(chain, sites, ct, omega0, seed, tol, steps=(1e-3, 5e-4)):
    direction = random_direction(chain.mol0.natm, seed)
    grad = DiabaticGradient(chain, FRAGMENTS, sites, ct, omega0=omega0)
    fd = [diabatic_derivative(chain, FRAGMENTS, sites, direction, ct,
                              omega0=omega0, step=h)[0]['a_eff']
          for h in steps]
    fd = (4.0 * fd[1] - fd[0]) / 3.0
    scale = np.abs(fd).max()
    n = len(grad.partition.labels)
    for a in range(n):
        for b in range(a, n):
            an = float((grad.element(a, b)[0] * direction).sum())
            assert abs(abs(an) - abs(fd[a, b])) < tol * scale, (a, b)


@pytest.mark.parametrize('tda', [True, False])
def test_two_states_in_one_block_match_finite_differences(chain, full_chain,
                                                          tda):
    check_elements(chain if tda else full_chain, TWO_PER_SITE, None,
                   OMEGA0_TWO, 7, 2e-5)


@pytest.fixture(scope='module')
def full_operator(full_chain):
    return BSEOperator.from_chain(full_chain, route='dense')[0]


@pytest.fixture(scope='module')
def full_partition(full_operator, orbitals):
    return FragmentPartition.build(full_operator, orbitals, SITES)


def test_full_bse_partition_is_the_exact_feshbach_elimination(
        full_operator, orbitals, full_partition):
    op, part = full_operator, full_partition
    n = op.n_ov
    t = orbitals.transition_rotation()
    zero = np.zeros_like(t)
    tt = np.block([[t, zero], [zero, t]])
    oracle = FeshbachOracle(tt.T @ op.dense @ tt, part.p_local, op.metric)
    assert np.abs(oracle.sigma(part.omega0) - part.sigma).max() < 1e-12
    assert np.abs(oracle.dsigma(part.omega0) - part.dsigma).max() < 1e-12
    w, x, y = bse_solve(op.dense[:n, :n], op.dense[:n, n:])
    for guess in np.linalg.eigvalsh(part.a_eff):
        om, _, z = oracle.root(guess)
        k = int(np.argmin(np.abs(w - om)))
        assert abs(om - w[k]) < 1e-10
        v = to_local(orbitals, np.vstack([x[:, [k]], y[:, [k]]]))[:, 0]
        assert abs(z - np.sum((part.p_local.T @ (op.metric * v)) ** 2)) < 1e-10


def test_full_bse_isdf_action_and_iterative_path_match_dense(
        full_chain, full_operator, orbitals, full_partition, monkeypatch):
    isdf = BSEOperator.from_chain(full_chain, route='isdf')[0]
    x = np.random.default_rng(5).normal(size=(full_operator.dim, 3))
    assert np.abs(isdf.apply(x) - full_operator.dense @ x).max() < 1e-10
    force_iterative(monkeypatch)
    iterative = FragmentPartition.build(isdf, orbitals, SITES,
                                        omega0=full_partition.omega0)
    assert np.abs(np.diag(iterative.a_eff)
                  - np.diag(full_partition.a_eff)).max() < 1e-8
    assert np.abs(np.abs(iterative.a_eff)
                  - np.abs(full_partition.a_eff)).max() < 1e-8
    assert abs(iterative.q_lowest - full_partition.q_lowest) < 1e-6


def test_full_bse_root_gradient_is_the_supermolecular_root_gradient(
        full_chain, full_operator):
    ref, info = full_chain.excitation_gradient()
    n = full_operator.n_ov
    omega = bse_solve(full_operator.dense[:n, :n],
                      full_operator.dense[:n, n:])[0][0]
    assert abs(info['omega'] - omega) < 1e-10
    grad = DiabaticGradient(full_chain, FRAGMENTS, SITES, omega0=omega)
    g, d = grad.root_gradient(omega)
    assert np.abs(g - ref).max() < 1e-8 * max(1.0, np.abs(ref).max())
    assert np.abs(d['canonical']).max() < 1e-10
    assert np.abs(d['localization']).max() < 1e-10


def test_full_bse_elements_in_point_charges_match_finite_differences(
        charged_full_chain):
    check_elements(charged_full_chain, SITES, BOTH_CT, OMEGA0_CHARGED, 11,
                   1e-6, steps=(5e-4, 2.5e-4))


def test_full_bse_iterative_gradient_matches_dense(full_chain, monkeypatch):
    dense = DiabaticGradient(full_chain, FRAGMENTS, SITES)
    ref = {ab: dense.element(*ab)[0] for ab in ((0, 0), (0, 1))}
    force_iterative(monkeypatch)
    iterative = DiabaticGradient(full_chain, FRAGMENTS, SITES, route='isdf',
                                 omega0=dense.partition.omega0)
    for ab, r in ref.items():
        g = iterative.element(*ab)[0]
        assert min(np.abs(g - r).max(), np.abs(g + r).max()) \
            < 1e-7 * np.abs(r).max()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
