"""The geomeTRIC walk retries a step shorter when its evaluation raises
`StepFailed`, as the Cartesian walker does.

A frozen-convention surface fails at a trial geometry too far from the last
accepted one (a lost quasiparticle root, a lost followed state,
`ConventionsLost`); the walk must then reject the step and retry it shorter
from the last accepted geometry, not end. The surface here is harmonic in the
interatomic distances and raises `ConventionsLost` whenever a distance moved
more than GATE from the geometry the walk last committed:

  - the gated walk reaches the minimum of an uninterrupted walk on the same
    surface without the gate, and counts every failed step as a rejection;
  - a step that fails at every radius ends the walk with the radius
    collapsed, at the start, and every shrink counted;
  - an exception that is not a `StepFailed` leaves the walk;
  - a checkpoint after a rejection carries the shrunk radius and the count,
    and a walk resumed from it reaches the same minimum.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.constants import WALK_STEP_SHRINK, WALK_TRUST_MIN_BOHR
from src.gradients.conventions import ConventionsLost
from src.properties.optimize import optimize_geometric

pytest.importorskip('geometric')

#: Bohr: the largest change of an interatomic distance from the last accepted
#: geometry that the gated surface still evaluates. geomeTRIC's default first
#: step moves the distances below by about 0.2 Bohr.
GATE = 0.05
#: Bohr: off the minimum, the start's displacement (not a rigid motion).
DISPLACEMENT = np.array([[0.0, 0.0, 0.0], [0.0, 0.25, 0.20],
                         [0.0, -0.10, 0.30]])
#: Bohr and Ha: two converged walks (geomeTRIC's GAU) agree within these.
SAME_DISTANCE, SAME_ENERGY = 2e-3, 1e-6


def distances(coords):
    """Interatomic distances, i < j."""
    coords = np.asarray(coords, float).reshape(-1, 3)
    i, j = np.triu_indices(len(coords), 1)
    return np.linalg.norm(coords[i] - coords[j], axis=1)


class GatedSpring:
    """E = sum_ij (r_ij - r*_ij)^2 / 2 over the distances of `minimum`;
    `failure(...)` is raised at a geometry whose distances moved more than
    `gate` from the last committed one."""

    def __init__(self, mol, minimum, gate=np.inf, failure=ConventionsLost):
        self.mol0 = mol
        self.target = distances(minimum)
        self.gate, self.failure = gate, failure
        self.anchor = None
        self.failures = 0

    def commit(self, mol):
        self.anchor = distances(mol.atom_coords())

    def total_gradient(self, mol=None, mf=None):
        coords = (self.mol0 if mol is None else mol).atom_coords()
        r = distances(coords)
        if self.anchor is not None and (np.abs(r - self.anchor).max()
                                        > self.gate):
            self.failures += 1
            raise self.failure(f'a distance moved '
                               f'{np.abs(r - self.anchor).max():.3f} Bohr')
        i, j = np.triu_indices(len(coords), 1)
        pull = ((r - self.target) / r)[:, None] * (coords[i] - coords[j])
        grad = np.zeros_like(coords)
        np.add.at(grad, i, pull)
        np.add.at(grad, j, -pull)
        return grad, 0.5 * float(((r - self.target) ** 2).sum()), {}

    def total_energy(self, mol=None, mf=None):
        return self.total_gradient(mol)[1]

    def refreeze(self, mol):
        return self


@pytest.fixture
def water():
    """(start, minimum geometry) of the spring surface."""
    minimum = gto.M(atom='O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11',
                    unit='Bohr', basis='sto-3g', verbose=0)
    start = minimum.copy()
    start.set_geom_(minimum.atom_coords() + DISPLACEMENT, unit='Bohr')
    start.build(False, False)
    return start, minimum.atom_coords()


def walk(surface, start, tmp_path, name, **kw):
    workdir = tmp_path / name
    workdir.mkdir()
    return optimize_geometric(surface, start, verbose=False, nudge=0,
                              workdir=str(workdir), **kw)


def test_a_failed_step_is_retried_shorter_to_the_same_minimum(water,
                                                              tmp_path):
    start, minimum = water
    free_mol, free = walk(GatedSpring(start, minimum), start, tmp_path,
                          'free')
    assert free['converged']
    gated = GatedSpring(start, minimum, gate=GATE)
    gated_mol, info = walk(gated, start, tmp_path, 'gated')
    assert info['converged'], info['status']
    assert info['status'] == 'ok'
    print(f'gated walk: {info["cycles"]} evaluations, {info["rejected"]} '
          f'rejected; free walk: {free["cycles"]} evaluations')
    assert free['rejected'] == 0
    # every failed evaluation was a rejected step, and each was retried
    assert gated.failures >= 1
    assert info['rejected'] == gated.failures == len(info['retries'])
    for retry in info['retries']:
        assert retry['status'].startswith('ConventionsLost: ')
        assert retry['trust_bohr'] == pytest.approx(
            WALK_STEP_SHRINK * retry['step_bohr'])
    assert info['history'][-1]['rejected'] == info['rejected']
    r_gated = distances(gated_mol.atom_coords())
    assert np.abs(r_gated - distances(free_mol.atom_coords())).max() \
        < SAME_DISTANCE
    assert np.abs(r_gated - distances(minimum)).max() < SAME_DISTANCE
    assert abs(info['energy'] - free['energy']) < SAME_ENERGY


def test_a_step_that_fails_at_every_radius_collapses_the_radius(water,
                                                                tmp_path):
    start, minimum = water
    stuck = GatedSpring(start, minimum, gate=0.0)
    mol_opt, info = walk(stuck, start, tmp_path, 'stuck')
    assert info['converged'] is False
    assert info['status'].startswith('trust radius collapsed at ')
    assert 'the state stopped evaluating (ConventionsLost: ' in info['status']
    assert info['rejected'] == stuck.failures == len(info['retries']) >= 2
    radii = [retry['trust_bohr'] for retry in info['retries']]
    assert radii[-1] < WALK_TRUST_MIN_BOHR <= min(radii[:-1])
    assert np.allclose(mol_opt.atom_coords(), start.atom_coords(), atol=1e-12)
    assert info['refreeze_shift'] is None


def test_any_other_evaluation_failure_leaves_the_walk(water, tmp_path):
    start, minimum = water
    broken = GatedSpring(start, minimum, gate=GATE, failure=ValueError)
    with pytest.raises(ValueError, match='a distance moved'):
        walk(broken, start, tmp_path, 'broken')
    assert broken.failures == 1


def test_a_walk_resumed_after_a_rejection_keeps_its_radius(water, tmp_path):
    start, minimum = water
    states = []
    _, info = walk(GatedSpring(start, minimum, gate=GATE), start, tmp_path,
                   'whole', checkpoint=states.append)
    after = [s for s in states if s['rejected']]
    assert after and info['converged']
    state = after[0]
    # the radius of the restart after the state's last rejection
    assert state['trust'] == pytest.approx(
        info['retries'][state['rejected'] - 1]['trust_bohr'])
    again = GatedSpring(start, minimum, gate=GATE)
    again.anchor = distances(state['coords'])
    mol_again, resumed = walk(again, start, tmp_path, 'resumed',
                              resume=state)
    assert resumed['converged'], resumed['status']
    assert resumed['rejected'] >= state['rejected']
    assert np.abs(distances(mol_again.atom_coords())
                  - distances(minimum)).max() < SAME_DISTANCE
