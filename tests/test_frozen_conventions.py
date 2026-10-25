"""The frozen conventions keyed by orbital identity (`gradients.conventions`)
and the self-consistent quasiparticle Hamiltonian of a group
(`sum_over_poles.subspace_hamiltonian`).

  - the group's H is covariant: a rotation of the subspace's basis rotates H
    and leaves its eigenvalues to rounding; one orbital is the diagonal
    Newton w = e0 + Sigma^c(w); the adjoint through the fixed point matches
    central differences in e0, eps and the contractions;
  - the conventions are matched unit by unit: an orbital rotating inside its
    group, or among outside orbitals of one shift, keeps every index; two
    explicit orbitals that have crossed swap their conventions; one mixed
    half and half with another unit is refused; `commit` makes the matched
    orbitals the keys and carries a group's seed along;
  - a hole that shares a quasiparticle-Hamiltonian group with an explicit
    orbital is solved in it.
"""
import types
import warnings

import numpy as np
import pytest

import src.gradients.gwbse.conventions as conventions
from src.Base.numerics.constants import CONVENTION_SPAN_MIN
from src.SingleReference.GW.qp_selection import AdaptivePartition
from src.SingleReference.GW.sum_over_poles import (initial_poles,
                                                   pole_amplitudes,
                                                   pole_denominators,
                                                   qp_energy_sop,
                                                   subspace_hamiltonian,
                                                   subspace_residues)
from src.gradients.gwbse.conventions import ConventionsLost, FrozenConventions
from src.gradients.gwbse.sum_over_poles_adjoint import (subspace_hamiltonian_backward,
                                                        subspace_residues_backward)
from tests.test_qp_hamiltonian import chain

NORB, NOCC, NFREQ, NAUX = 12, 5, 20, 4
#: Ha: covariance and the one-orbital limit, against rounding
EXACT = 1e-12
#: the adjoint against a central difference of step FD_STEP, relative
FD_STEP = 1e-6
FD_TOL = 1e-7


def synthetic(n=3, seed=1):
    """(eps, nu, poles, slices, e0) of a small pole model with a subspace of
    n virtual orbitals: contractions wc_ab = f(nu) sum_P B_a B_b per column."""
    rng = np.random.default_rng(seed)
    eps = np.sort(rng.uniform(-1.5, 1.5, NORB))
    eps[NOCC:] += 0.4
    nu = np.geomspace(0.01, 20.0, NFREQ)
    poles = initial_poles(6, 0.5, 5.0)
    slices = rng.normal(size=(n, NORB, NAUX)) * 0.3
    members = list(range(NOCC, NOCC + n))
    coupling = 0.002 * (np.ones((n, n)) - np.eye(n))
    return eps, nu, poles, slices, np.diag(eps[members]) + coupling


def contractions(nu, slices):
    """{(a, b): wc_ab} of the slices, a <= b."""
    f = -1.0 / (1.0 + (nu / 0.8) ** 2)
    n = len(slices)
    return {(a, b): f[:, None] * np.einsum('sP,sP->s', slices[a],
                                           slices[b])[None, :]
            for a in range(n) for b in range(a, n)}


def solve(e0, slices, eps, nu, poles):
    amp = subspace_residues(contractions(nu, slices), poles, nu)
    d = pole_denominators(poles, eps, NOCC)
    h, _, converged = subspace_hamiltonian(e0, amp, d, e0)
    assert converged
    return h, amp, d


def test_the_group_hamiltonian_is_covariant():
    eps, nu, poles, slices, e0 = synthetic()
    h, _, _ = solve(e0, slices, eps, nu, poles)
    theta = 0.7
    u = np.eye(3)
    u[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]]
    turned = np.einsum('ab,bsP->asP', u.T, slices)
    h_r, _, _ = solve(u.T @ e0 @ u, turned, eps, nu, poles)
    assert np.abs(h_r - u.T @ h @ u).max() < EXACT


def test_one_orbital_is_the_diagonal_newton():
    eps, nu, poles, slices, e0 = synthetic(n=1)
    h, _, _ = solve(e0, slices, eps, nu, poles)
    amp = pole_amplitudes(contractions(nu, slices)[(0, 0)], poles, nu)
    w, _ = qp_energy_sop(NOCC, amp, poles, eps, NOCC,
                         xc_correction=e0[0, 0] - eps[NOCC])
    assert abs(h[0, 0] - w) < EXACT


def test_the_adjoint_through_the_fixed_point():
    eps, nu, poles, slices, e0 = synthetic()
    rng = np.random.default_rng(2)
    h_bar = rng.normal(size=(3, 3))
    h_bar = h_bar + h_bar.T

    def loss(e0_, slices_, eps_):
        return float(np.sum(h_bar * solve(e0_, slices_, eps_, nu, poles)[0]))

    h, amp, d = solve(e0, slices, eps, nu, poles)
    e0_bar, amp_bar, d_bar = subspace_hamiltonian_backward(h, amp, d, h_bar)
    wc_bar = subspace_residues_backward(amp_bar, poles, nu,
                                        list(contractions(nu, slices)))
    eps_bar = d_bar.reshape(NORB, -1).sum(axis=1)

    def central(f):
        return (f(FD_STEP) - f(-FD_STEP)) / (2 * FD_STEP)

    de0 = rng.normal(size=(3, 3))
    de0 = de0 + de0.T
    fd = central(lambda t: loss(e0 + t * de0, slices, eps))
    assert abs(fd - np.sum(e0_bar * de0)) < FD_TOL * abs(fd)
    deps = np.zeros(NORB)
    deps[NOCC + 4] = 1.0
    fd = central(lambda t: loss(e0, slices, eps + t * deps))
    assert abs(fd - eps_bar[NOCC + 4]) < FD_TOL * max(abs(fd), 1.0)
    dslices = rng.normal(size=slices.shape)
    fd = central(lambda t: loss(e0, slices + t * dslices, eps))
    dwc = {k: central(lambda t: contractions(nu, slices + t * dslices)[k])
           for k in wc_bar}
    assert abs(fd - sum(np.sum(wc_bar[k] * dwc[k]) for k in wc_bar)) \
        < FD_TOL * abs(fd)


# ----------------------------------------------------------------- matching
def frozen(groups=(), explicit=(1, 2, 3, 5, 6, 7), norb=10, nocc=4):
    """Conventions anchored at a stand-in geometry: `explicit` solved,
    `groups` its quasiparticle-Hamiltonian groups, every other orbital one
    shift per side."""
    mol = types.SimpleNamespace(atom_coords=lambda: np.zeros((1, 3)))
    c = FrozenConventions(key_mol=mol, key_mo=np.eye(norb), nocc=nocc)
    c.qp_set = np.array(explicit, int)
    c.qp_subspaces = (tuple(explicit), tuple(groups)) if groups else None
    c.outside_shift = {p: 0.1 if p < nocc else -0.2 for p in range(norb)
                       if p not in explicit}
    c.anchored = True
    return c


def displaced(monkeypatch, t):
    """A stand-in geometry whose orbitals overlap the keys by `t`."""
    monkeypatch.setattr(conventions, 'body_overlap', lambda *args: t)
    mol = types.SimpleNamespace(atom_coords=lambda: np.ones((1, 3)))
    return mol, types.SimpleNamespace(mo_coeff=np.eye(len(t)))


def rotation(norb, p, q, mixing):
    t = np.eye(norb)
    c, s = np.sqrt(1.0 - mixing), np.sqrt(mixing)
    t[np.ix_([p, q], [p, q])] = [[c, -s], [s, c]]
    return t


def test_a_rotation_inside_a_group_keeps_every_index(monkeypatch):
    c = frozen(groups=((5, 6),))
    view = c.at(*displaced(monkeypatch, rotation(10, 5, 6, 0.5)))
    assert view.perm is None
    assert view.rotations[(5, 6)] is not None


def test_outside_orbitals_of_one_shift_rotate_freely(monkeypatch):
    c = frozen()
    view = c.at(*displaced(monkeypatch, rotation(10, 8, 9, 0.5)))
    assert view.perm is None


def test_crossed_explicit_orbitals_swap_their_conventions(monkeypatch):
    c = frozen()
    c.qp_seeds.update({5: -1.0, 6: -2.0})
    view = c.at(*displaced(monkeypatch, rotation(10, 5, 6, 0.95)))
    assert view.index(5) == 6 and view.index(6) == 5
    assert view.qp_seeds[6] == -1.0 and view.qp_seeds[5] == -2.0


def test_an_orbital_mixed_with_another_unit_is_refused(monkeypatch):
    c = frozen()
    with pytest.raises(ConventionsLost, match='left its group'):
        c.at(*displaced(monkeypatch, rotation(10, 7, 8, 0.5)))
    assert CONVENTION_SPAN_MIN > 0.5


def test_commit_carries_the_keys_and_a_group_seed(monkeypatch):
    c = frozen(groups=((5, 6),))
    seed = np.array([[1.0, 0.1], [0.1, 2.0]])
    c.group_seeds[(5, 6)] = seed.copy()
    t = rotation(10, 5, 6, 0.3)
    mol, mf = displaced(monkeypatch, t)
    rot = c.at(mol, mf).rotations[(5, 6)]
    c.commit(mol, mf)
    assert c.key_mol is mol and c.commits == 1
    assert np.allclose(c.group_seeds[(5, 6)], rot.T @ seed @ rot)
    assert np.allclose(np.linalg.eigvalsh(c.group_seeds[(5, 6)]),
                       np.linalg.eigvalsh(seed))


# ---------------------------------------------------------------- promotion
def test_a_hole_in_a_group_with_an_explicit_orbital_is_promoted():
    warnings.simplefilter('ignore')
    window = list(range(4, 18))
    explicit = [p for p in window if p != 13]
    partition = AdaptivePartition(explicit=explicit, tier_of={13: (12,)},
                                  candidates=window, targets=(('singlet', 0),),
                                  tol_meV=1.0)
    plain = chain(qp_partition=partition)
    assert 13 not in set(int(p) for p in plain.resolved.qp_set)
    coupled = chain(qp_partition=partition,
                    qp_hamiltonian=True)
    assert 13 in set(int(p) for p in coupled.resolved.qp_set)
    assert 12 in coupled.conventions.group_promoted[13]
    assert coupled.resolved.qp_partition.tier_of == {}
