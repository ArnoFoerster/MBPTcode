"""The symmetry-breaking start of an excited-state walk and the saddle test.

A surface exactly invariant to rigid motion keeps the point group of a
walk's start: the force has no component that breaks it, so a walk from
planar formaldehyde stops at the planar S1 saddle, 79 meV above the
pyramidal minimum. Every excited-state walk therefore starts a seeded
internal displacement away (`start_nudge`), and a Hessian at a minimum is
tested for a negative mode and left along it (`escape_saddle`).

The surface is a toy with the same structure: an NH3-shaped spring network
(planar at rest) plus a double well in the pyramidalization, so the planar
point is a symmetric saddle and the pyramidal minima break the symmetry. Its
energy is even in the out-of-plane coordinate to the bit, as a rotation-
invariant chain's is.

Every check asserts: pytest discards a returned verdict and passes on False.
"""
import numpy as np
import pytest
from pyscf import gto

import src.gradients  # noqa: F401  cycle: src.properties imports src.gradients
from src.Base.constants import (SADDLE_CURVATURE_TOL, START_NUDGE_BOHR,
                                START_NUDGE_SEED)
from src.properties.hessian import surface_hessian
from src.properties.optimize import (escape_saddle, internal_modes,
                                     nudge_vector, optimize,
                                     optimize_geometric, saddle_test,
                                     start_nudge)

# planar NH3: N at the centre, three H at 2 Bohr, 120 degrees apart
PLANAR = np.array([[0.0, 0.0, 0.0]]
                  + [[2.0 * np.cos(t), 2.0 * np.sin(t), 0.0]
                     for t in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)])
SPRING = 0.5          # Ha/Bohr^2, every pair distance
WELL_A = 0.05         # Ha/Bohr^2: curvature -2 WELL_A at the plane
WELL_C = 0.15625      # Ha/Bohr^4: |h| = 0.4 at the minima without the springs
FD = 1e-5


def pyramidalization(x):
    """Signed distance of atom 0 from the plane of atoms 1-3."""
    n = np.cross(x[2] - x[1], x[3] - x[1])
    return float(n @ (x[0] - x[1]) / np.linalg.norm(n))


def well_energy(x):
    """Spring network at the planar rest distances plus the double well."""
    rest = np.linalg.norm(PLANAR[:, None] - PLANAR[None], axis=-1)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    h = pyramidalization(x)
    return (0.25 * SPRING * ((d - rest) ** 2).sum()
            - WELL_A * h ** 2 + WELL_C * h ** 4)


class GroundWell:
    """The double well as a ground-state surface, its central-difference
    force."""

    def __init__(self, mol):
        self.mol0 = mol

    def total_energy(self, mol=None, mf=None):
        return well_energy((mol or self.mol0).atom_coords())

    def total_gradient(self, mol=None, mf=None):
        x = (mol or self.mol0).atom_coords()
        g = np.zeros_like(x)
        for i in range(x.shape[0]):
            for k in range(3):
                xp, xm = x.copy(), x.copy()
                xp[i, k] += FD
                xm[i, k] -= FD
                g[i, k] = (well_energy(xp) - well_energy(xm)) / (2 * FD)
        return g, well_energy(x), {'omega': 0.1}

    def refreeze(self, mol):
        return self

    def label(self):
        return 'double well'


class DoubleWell(GroundWell):
    """The double well with an `excitation`: an excited-state surface to
    `start_nudge`."""

    def excitation(self, mol=None, mf=None):
        return 0.1


def molecule(coords):
    return gto.M(atom=[('N' if i == 0 else 'H', tuple(c))
                       for i, c in enumerate(coords)],
                 unit='Bohr', basis='sto-3g', verbose=0)


@pytest.fixture(scope='module')
def pyramidal_minimum():
    """(|h|, E) of the minimum, walked from a clearly pyramidal start."""
    x = PLANAR.copy()
    x[0, 2] = 0.4
    surface = DoubleWell(molecule(x))
    mol, info = optimize(surface, surface.mol0, max_cycle=200, verbose=False,
                         nudge=0, conv={'opt_grad_max': 1e-6,
                                        'grad_rms': 1e-6})
    assert info['converged']
    return abs(pyramidalization(mol.atom_coords())), info['energy']


@pytest.fixture
def planar_start():
    """Planar, the in-plane springs stretched, so the walk has to walk."""
    x = PLANAR.copy()
    x[1, 0] += 0.15
    x[2, 1] -= 0.1
    return molecule(x)


def test_the_nudge_is_reproducible_and_internal():
    x = PLANAR + 0.01 * np.arange(12).reshape(4, 3)
    d = nudge_vector(x)
    assert np.array_equal(d, nudge_vector(x)), 'same seed, same vector'
    assert np.linalg.norm(d, axis=1).max() == pytest.approx(START_NUDGE_BOHR,
                                                            rel=1e-12)
    # no net translation, no net rotation about the centroid
    assert np.abs(d.sum(axis=0)).max() < 1e-15
    c = x - x.mean(axis=0)
    assert np.abs(np.cross(c, d).sum(axis=0)).max() < 1e-15
    assert not np.array_equal(d, nudge_vector(x, seed=START_NUDGE_SEED + 1))
    assert not nudge_vector(x, amplitude=0.0).any()


def test_the_nudge_is_recorded_and_excited_only(planar_start):
    start, record = start_nudge(DoubleWell(planar_start), planar_start)
    d = start.atom_coords() - planar_start.atom_coords()
    assert np.allclose(d, record['displacement_bohr'], atol=1e-14)
    assert record['amplitude_bohr'] == START_NUDGE_BOHR
    assert record['seed'] == START_NUDGE_SEED
    # a ground-state surface, or the opt-out, starts exactly at the geometry
    for surface, nudge in ((GroundWell(planar_start), None),
                           (DoubleWell(planar_start), 0)):
        same, record = start_nudge(surface, planar_start, nudge)
        assert same is planar_start
        assert record['amplitude_bohr'] == 0.0
        assert record['displacement_bohr'] is None


@pytest.mark.parametrize('walk', [
    lambda s, m, **kw: optimize(s, m, max_cycle=200, verbose=False, **kw),
    lambda s, m, **kw: optimize_geometric(s, m, verbose=False, **kw)],
    ids=['cartesian', 'geometric'])
def test_a_symmetric_start_reaches_the_broken_minimum(planar_start, walk,
                                                      pyramidal_minimum):
    surface = DoubleWell(planar_start)
    stuck, info0 = walk(surface, planar_start, nudge=0)
    assert info0['converged']
    assert abs(pyramidalization(stuck.atom_coords())) < 1e-10, \
        'without the nudge the walk keeps the plane: the saddle'
    moved, info = walk(surface, planar_start)
    assert info['converged']
    h_min, e_min = pyramidal_minimum
    assert abs(pyramidalization(moved.atom_coords())) == pytest.approx(
        h_min, abs=1e-2)
    assert info['energy'] == pytest.approx(e_min, abs=1e-6)
    assert info0['energy'] - e_min > 1e-3, 'the plane is the saddle above it'
    assert info['start_nudge']['amplitude_bohr'] == START_NUDGE_BOHR
    assert info0['start_nudge']['amplitude_bohr'] == 0.0


def test_a_planted_negative_mode_is_found():
    x = PLANAR + np.array([[0.0, 0.0, 0.3], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    lam, modes = internal_modes(np.eye(12), x)
    planted = modes[:, 2]
    hess = np.eye(12) - 1.05 * np.outer(planted, planted)
    check = saddle_test(hess, x)
    assert check['saddle'] and check['negative_modes'] == 1
    assert check['lowest_curvature'] == pytest.approx(-0.05, abs=1e-12)
    assert abs(check['mode'].ravel() @ planted) == pytest.approx(1.0,
                                                                 abs=1e-12)
    # in pyscf's (natm, natm, 3, 3) layout the same
    flat = hess.reshape(4, 3, 4, 3).transpose(0, 2, 1, 3)
    assert saddle_test(flat, x)['lowest_curvature'] == pytest.approx(-0.05)
    # a negative curvature softer than the tolerance moves no energy a record
    # could see
    soft = np.eye(12) - (1 - 0.5 * SADDLE_CURVATURE_TOL) * np.outer(planted,
                                                                     planted)
    assert not saddle_test(soft, x)['saddle']


def walk_without_nudge(calls):
    def walk(surface, start):
        calls.append(start)
        return optimize(surface, start, max_cycle=200, verbose=False, nudge=0)
    return walk


def test_a_saddle_is_left_and_the_new_minimum_checked(planar_start,
                                                      pyramidal_minimum):
    surface = DoubleWell(planar_start)
    saddle, info = optimize(surface, planar_start, max_cycle=200,
                            verbose=False, nudge=0)
    hess = surface_hessian(surface, saddle)
    calls = []
    moved, again, new_hess, record = escape_saddle(
        surface, saddle, hess, walk_without_nudge(calls), surface_hessian)
    assert record['status'] == 'escaped'
    assert record['first']['saddle']
    assert record['first']['lowest_curvature'] < SADDLE_CURVATURE_TOL
    # the mode is the pyramidalization: atom 0 out of the plane
    mode = record['first']['mode']
    assert np.abs(mode[:, :2]).max() < 1e-6
    assert len(calls) == 1 and again['converged']
    h_min, e_min = pyramidal_minimum
    assert abs(pyramidalization(moved.atom_coords())) == pytest.approx(
        h_min, abs=1e-2)
    assert again['energy'] == pytest.approx(e_min, abs=1e-6)
    assert record['energy_drop'] == pytest.approx(
        record['energy_at_saddle'] - e_min, abs=1e-6)
    assert not record['second']['saddle']
    assert record['direction'] in (+1, -1) and len(record['energies']) == 2
    assert new_hess is not None


def test_a_true_minimum_passes_unchanged(planar_start):
    surface = DoubleWell(planar_start)
    minimum, _ = optimize(surface, planar_start, max_cycle=200, verbose=False)
    hess = surface_hessian(surface, minimum)
    calls = []
    same, again, kept, record = escape_saddle(
        surface, minimum, hess, walk_without_nudge(calls), surface_hessian)
    assert same is minimum and again is None and kept is hess
    assert record['status'] == 'minimum' and not calls
    assert record['first']['lowest_curvature'] > 0


def test_a_second_saddle_stops_the_check(planar_start):
    surface = DoubleWell(planar_start)
    saddle, _ = optimize(surface, planar_start, max_cycle=200, verbose=False,
                         nudge=0)
    hess = surface_hessian(surface, saddle)
    calls = []
    planted = -np.eye(12)
    _, _, _, record = escape_saddle(surface, saddle, hess,
                                    walk_without_nudge(calls),
                                    lambda s, m: planted)
    assert record['status'] == 'second saddle: stopped'
    assert len(calls) == 1, 'one escape, no loop'
    assert record['second']['saddle']
