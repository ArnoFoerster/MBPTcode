"""A convention refreeze on a solvated (scheme-B) walk across a
near-degenerate quasiparticle-Hamiltonian block (`gradients.conventions`,
`gradients.solvated_surface`, `properties.optimize`).

The walk runs the real solvation loop, `SolvatedSurface`, `FrozenConventions`
and walkers on a stand-in state model: a harmonic spring in the interatomic
distances of water whose orbitals depend on the geometry and on the reaction
field. Leaving the start geometry, orbital 6 (in the group (5, 6)) mixes 0.3
with orbital 8 outside it, so the first step is refused with 8 its partner
and the walk refreezes at the start with 8 absorbed. Orbitals 7 and 8 are
near-degenerate: the field rotates them into each other by KAPPA radians per
unit of charge, as the reaction field rotates diffuse virtuals. S0's charges
(Q_S0) and the state's (Q_S1) differ enough to mix them by 0.5:

  - a group is a contiguous energy window: the absorbed (5, 6, 8) is taken as
    (5, 6, 7, 8), and a group with an orbital between its members outside it
    is cut there;
  - the refrozen surface freezes on S0's charges at the start, as a
    vertical there does, and its loop moves the field to the state's: the
    7-8 rotation stays inside the group (5, 6, 7, 8), so the refrozen surface
    evaluates at its own geometry and the walk converges (before: the group
    (5, 6, 8) refused the geometry it was frozen at with `ConventionsLost`,
    which left the walk as an exception);
  - a refrozen surface that does not evaluate at its own geometry ends the
    walk as a record, not converged, its status 'convention refreeze
    failed', on every rank (both walkers; geomeTRIC also on two ranks);
  - a retry refused again before any step is accepted refreezes at
    WALK_STEP_SHRINK times the radius;
  - a refrozen surface that evaluates at its own geometry but moves the
    state's energy there by more than REFREEZE_TOL_MEV (as DMABN's second
    refreeze moved S1 by 0.52 eV) starts a new round there, on the walk's
    starting radius and Hessian, whose minimum is the one returned; one
    within it continues the round; rounds that keep jumping past
    WALK_CONVENTION_REFREEZES_MAX end the walk flagged, on every rank.
"""
import json
import types
import warnings

import numpy as np
import pytest
from pyscf import gto

import src.gradients.gwbse.conventions as conventions
import src.properties.optimize as optimize_module
from src.Base.numerics.constants import (HARTREE_TO_MEV, REFREEZE_TOL_MEV,
                                         WALK_STEP_SHRINK)
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.GW.qp_selection import contiguous_runs
from src.gradients.gwbse.conventions import ConventionsLost, FrozenConventions
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.qp_set import QuasiparticleSet
from src.gradients.solvated_surface import SolvatedSurface
from src.photophysics.records import walk_fields
from src.properties.optimize import optimize, optimize_geometric
from tests.test_frozen_conventions import rotation
from tests.test_geometric_step_failed_retry import DISPLACEMENT, distances

NORB, NOCC = 10, 5
#: Ha: the mean-field energies; 7 and 8 lie 0.04 Ha above the group (5, 6),
#: beyond QP_HAMILTONIAN_WINDOW of it, and 0.1 mHa apart
EPS = np.array([-1.0, -0.8, -0.6, -0.5, -0.4, 0.100, 0.1005, 0.140, 0.1401,
                0.6])
EXPLICIT, GROUP = (3, 4, 5, 6), (5, 6)
#: the weight orbital 6 gives orbital 8 away from the start geometry
MIX = 0.3
#: radians of 7-8 rotation per unit of field charge
KAPPA = 0.571
#: dipoles of S0 and of the state: their equilibrium charges are -d / 2
D_S0, D_S1 = -0.2, -3.0
Q_S0, Q_S1 = -0.5 * D_S0, -0.5 * D_S1
#: Bohr and Ha: the walks agree with the minimum within these
SAME_DISTANCE, SAME_ENERGY = 2e-3, 1e-6
#: meV: the refrozen surface's lift at one geometry: DMABN's second refreeze,
#: and a change of conventions within REFREEZE_TOL_MEV
JUMP, DRIFT = 520.0, 0.5 * REFREEZE_TOL_MEV


def turn(p, q, theta):
    """The rotation by `theta` radians between orbitals p and q."""
    t = np.eye(NORB)
    c, s = np.cos(theta), np.sin(theta)
    t[np.ix_([p, q], [p, q])] = [[c, -s], [s, c]]
    return t


def orbitals(coords, start, field):
    """The orbitals at `coords` in the field charge `field`: the 7-8 rotation
    of the field, and away from the start geometry the 6-8 mixing in its
    orbitals."""
    f = turn(7, 8, KAPPA * field)
    if np.allclose(coords, start, rtol=0, atol=1e-10):
        return f
    return f @ rotation(NORB, 6, 8, MIX)


def spring(coords, target):
    coords = np.asarray(coords, float)
    r = distances(coords)
    i, j = np.triu_indices(len(coords), 1)
    pull = ((r - target) / r)[:, None] * (coords[i] - coords[j])
    grad = np.zeros_like(coords)
    np.add.at(grad, i, pull)
    np.add.at(grad, j, -pull)
    return 0.5 * float(((r - target) ** 2).sum()), grad


class Field:
    """One site: the field charge is the operator, the answer to a density
    is -d / 2, the polarization q^2 / 2; nothing depends on the nuclei."""

    nsites = 1

    def __init__(self, mol):
        self.mol = mol

    def operator(self, q):
        return np.array(q, float), 0.0

    def potential(self, dm):
        return np.array([np.asarray(dm, float).ravel()[0]])

    def charges(self, v, kind):
        return -0.5 * np.asarray(v, float)

    def response_solve(self, q, kind):
        return np.asarray(q, float)

    def force(self, dm):
        return np.zeros((self.mol.natm, 3))

    def transition_environment(self):
        return None

    def identity(self):
        return {'field': 'one site'}


class Ground:
    """S0: the spring, dipole D_S0, freezes nothing."""

    lr_environment = None

    def __init__(self, mol, target):
        self.mol, self.target = mol, target

    def __call__(self, h, e):
        energy, _ = spring(self.mol.atom_coords(), self.target)
        return types.SimpleNamespace(energy=energy, dm_eff=np.array([D_S0]),
                                     info={'flags': []})

    def identity(self):
        return {'model': 'ground'}


class Chain:
    """The frozen conventions of one surface, built on the orbitals of its
    first call. `contiguous` False builds the group as the walk built it
    before groups were contiguous windows: 7, between absorbed 8 and the
    group, stays outside."""

    def __init__(self, mol, mo, absorbed, contiguous):
        c = FrozenConventions(key_mol=mol, key_mo=mo.copy(), nocc=NOCC)
        c.qp_set = np.array(EXPLICIT, int)
        c.carried_subspaces = ()
        if contiguous:
            # the chain's own absorption and grouping
            ExcitedStateChain.absorb_groups(types.SimpleNamespace(
                conventions=c, qp_select=None, nocc=NOCC), absorbed)
            joined = ((GROUP,) + tuple(c.carried_subspaces))
            c.carried_subspaces = joined
            groups = QuasiparticleSet._window_subspaces(
                types.SimpleNamespace(chain=types.SimpleNamespace(
                    nocc=NOCC, resolved=c)),
                tuple(int(p) for p in c.qp_set), EPS)
        else:
            members = sorted(set(GROUP) | {p for g in absorbed for p in g})
            c.qp_set = np.array(sorted(set(EXPLICIT) | set(members)), int)
            c.absorbed = tuple(absorbed)
            groups = (tuple(members),)
        c.qp_subspaces = (tuple(int(p) for p in c.qp_set), tuple(groups))
        c.outside_shift = {p: 0.1 if p < NOCC else -0.2 for p in range(NORB)
                           if p not in set(int(q) for q in c.qp_set)}
        c.anchored = True
        self.conventions = c


class State:
    """The excited state: the spring, dipole `dipoles[generation]`, its
    conventions frozen on its first call and resolved at every call."""

    lr_environment = None
    make_scf = None

    def __init__(self, mol, target, start, log, chain=None, generation=0,
                 absorbed=(), dipoles=(D_S1,), contiguous=True, lifts=(0.0,)):
        self.mol, self.target, self.start, self.log = mol, target, start, log
        self.chain, self.generation = chain, generation
        self.absorbed, self.dipoles = absorbed, dipoles
        self.contiguous, self.lifts = contiguous, lifts
        self.mf = None

    @property
    def freezes_on_first_call(self):
        return self.chain is None

    @property
    def surface_key(self):
        return self.chain

    def __call__(self, h, e):
        q = float(np.asarray(h).ravel()[0])
        coords = self.mol.atom_coords()
        self.mf = types.SimpleNamespace(
            mo_coeff=orbitals(coords, self.start, q))
        if self.chain is None:
            self.log.append(('frozen', self.generation, q))
            self.chain = Chain(self.mol, self.mf.mo_coeff, self.absorbed,
                               self.contiguous)
        self.chain.conventions.at(self.mol, self.mf)
        energy, _ = spring(coords, self.target)
        energy += self.lifts[min(self.generation, len(self.lifts) - 1)]
        d = self.dipoles[min(self.generation, len(self.dipoles) - 1)]
        return types.SimpleNamespace(energy=energy, dm_eff=np.array([d]),
                                     info={'flags': []})

    def force(self, h, e):
        r = self(h, e)
        _, grad = spring(self.mol.atom_coords(), self.target)
        return grad, {'dm_eff': r.dm_eff, 'lr': [], 'flags': []}

    def identity(self):
        return {'model': 'stand-in', 'generation': self.generation}

    def at(self, mol, make_scf, lr_environment=None):
        return State(mol, self.target, self.start, self.log, self.chain,
                     self.generation, self.absorbed, self.dipoles,
                     self.contiguous, self.lifts)

    def refrozen(self, mol, lr_environment=None):
        return State(mol, self.target, self.start, self.log, None,
                     self.generation + 1,
                     self.chain.conventions.absorbed_at_keys(), self.dipoles,
                     self.contiguous, self.lifts)

    def commit(self, mol):
        self.chain.conventions.commit(mol, self.mf)


@pytest.fixture
def water(monkeypatch):
    """(start, minimum coordinates); body_overlap is the orbitals' own."""
    minimum = gto.M(atom='O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11',
                    unit='Bohr', basis='sto-3g', verbose=0)
    start = minimum.copy()
    start.set_geom_(minimum.atom_coords() + DISPLACEMENT, unit='Bohr')
    start.build(False, False)
    monkeypatch.setattr(conventions, 'body_overlap',
                        lambda km, kmo, m, mo: np.asarray(kmo).T @ mo)
    return start, minimum.atom_coords()


def surface(water, log, **kw):
    start, minimum = water
    target = distances(minimum)
    built = []

    def make_state(m, field):
        if built:
            return built[0].at(m, None)
        built.append(State(m, target, start.atom_coords(), log, **kw))
        return built[0]
    return SolvatedSurface(start, Field, make_state, label='stand-in S1',
                           make_ground=lambda m, f: Ground(m, target))


def run(walker, s, start, tmp_path):
    warnings.simplefilter('ignore')
    if walker == 'cartesian':
        return optimize(s, start, verbose=False, nudge=0)
    workdir = tmp_path / 'walk'
    workdir.mkdir(parents=True, exist_ok=True)
    return optimize_geometric(s, start, verbose=False, nudge=0,
                              workdir=str(workdir))


def at_minimum(mol, info, minimum, energy=0.0):
    """At the spring's minimum, whose G is `energy` (the polarization)."""
    assert np.abs(distances(mol.atom_coords()) - distances(minimum)).max() \
        < SAME_DISTANCE
    assert abs(info['energy'] - energy) < SAME_ENERGY


# ------------------------------------------------------- the group's window
def test_a_group_is_a_contiguous_window(water):
    """The absorbed (5, 6, 8) joins 7 between them; the group then holds the
    7-8 rotation the field makes, which the group (5, 6, 8) with 7 outside
    refuses at the geometry it was frozen at."""
    assert contiguous_runs((5, 6, 8, 9, 12)) == [(5, 6), (8, 9), (12,)]
    start, _ = water
    mo_s0 = orbitals(start.atom_coords(), start.atom_coords(), Q_S0)
    mf_s1 = types.SimpleNamespace(
        mo_coeff=orbitals(start.atom_coords(), start.atom_coords(), Q_S1))
    window = Chain(start, mo_s0, ((5, 6, 8),), contiguous=True).conventions
    assert window.group_slots() == ((5, 6, 7, 8),)
    assert window.absorbed == ((5, 6, 7, 8),)
    assert {5, 6, 7, 8} <= set(int(p) for p in window.qp_set)
    view = window.at(start, mf_s1)
    assert view.perm is None
    gapped = Chain(start, mo_s0, ((5, 6, 8),), contiguous=False).conventions
    assert gapped.group_slots() == ((5, 6, 8),)
    with pytest.raises(ConventionsLost):
        gapped.at(start, mf_s1)


def test_a_group_with_a_member_outside_its_window_is_cut():
    """The set (3, 4, 5, 6, 8) with (5, 6, 8) carried: 7, between, is not
    solved, so the group is cut into its runs and (8) is no group."""
    c = FrozenConventions(nocc=NOCC)
    c.qp_set = np.array((3, 4, 5, 6, 8), int)
    c.carried_subspaces = ((5, 6, 8),)
    groups = QuasiparticleSet._window_subspaces(
        types.SimpleNamespace(chain=types.SimpleNamespace(
            nocc=NOCC, resolved=c)), (3, 4, 5, 6, 8), EPS)
    assert groups == [(5, 6)]


# -------------------------------------------------------- the refreeze
@pytest.mark.parametrize('walker', ['cartesian', 'geometric'])
def test_the_refrozen_surface_evaluates_at_its_own_geometry(water, walker,
                                                             tmp_path):
    if walker == 'geometric':
        pytest.importorskip('geometric')
    log = []
    s = surface(water, log)
    mol, info = run(walker, s, water[0], tmp_path)
    assert info['converged'], info['status']
    events = info['convention_refreezes']
    assert len(events) == 1 and events[0]['refrozen']
    assert events[0]['absorbed'] == [[5, 6, 8]]
    frozen = [entry for entry in log if entry[0] == 'frozen']
    # both surfaces freeze on S0's charges, as a vertical does, and the
    # loop then rotates 7 into 8 by 0.5
    assert frozen[0][1:] == (0, pytest.approx(Q_S0))
    assert frozen[1][1:] == (1, pytest.approx(Q_S0))
    assert np.sin(KAPPA * (Q_S1 - Q_S0)) ** 2 > 0.25
    rebuilt = info['final_surface']
    assert rebuilt is not s
    # G at the minimum: the spring's zero less the polarization q^2 / 2
    at_minimum(mol, info, water[1], -0.5 * Q_S1 ** 2)


@pytest.mark.parametrize('walker', ['cartesian', 'geometric'])
def test_a_refrozen_surface_refusing_its_own_geometry_ends_the_walk(
        water, walker, tmp_path):
    """The loop of the refrozen surface moves the field enough to mix 7 and
    8, and its group (5, 6, 8) leaves 7 outside (as the walk built it before
    groups were contiguous windows): the refrozen surface refuses the
    geometry it was frozen at. The walk ends as a record."""
    if walker == 'geometric':
        pytest.importorskip('geometric')
    s = surface(water, [], contiguous=False)
    mol, info = run(walker, s, water[0], tmp_path)
    assert not info['converged']
    assert info['status'].startswith('convention refreeze failed: ')
    assert 'ConventionsLost' in info['status']
    events = info['convention_refreezes']
    assert len(events) == 1 and events[0]['refrozen'] is False
    assert events[0]['e_after'] is None
    assert events[0]['failed'].startswith('ConventionsLost: ')
    # the walk stays on the surface of its last accepted step
    assert info['final_surface'] is s
    assert np.allclose(mol.atom_coords(), water[0].atom_coords())


def test_every_rank_ends_the_walk_on_a_failed_refreeze(water, tmp_path):
    pytest.importorskip('geometric')

    def walk(comm):
        s = surface(water, [], contiguous=False)
        work = tmp_path / f'rank{comm.Get_rank()}'
        _, info = run('geometric', s, water[0], work)
        return info['converged'], info['status']

    out = run_simulated(walk, 2)
    assert out[0] == out[1]
    assert out[0][0] is False
    assert out[0][1].startswith('convention refreeze failed: ')


# ------------------------------------------------- the energy's continuity
@pytest.mark.parametrize('walker', ['cartesian', 'geometric'])
def test_a_refreeze_that_jumps_starts_a_new_round(water, walker, tmp_path):
    """The refrozen surface evaluates at its own geometry JUMP meV above the
    surface it was rebuilt from there: a new surface, so the walk starts a
    new round there on its starting radius (geomeTRIC: its own) and
    converges on that surface (before: the same round walked on, its radius
    and Hessian those of the old surface, nothing recorded)."""
    if walker == 'geometric':
        pytest.importorskip('geometric')
    s = surface(water, [], lifts=(0.0, JUMP / HARTREE_TO_MEV))
    mol, info = run(walker, s, water[0], tmp_path)
    assert info['converged'], info['status']
    events = info['convention_refreezes']
    assert len(events) == 1 and events[0]['refrozen']
    assert events[0]['round'] == 2 and info['rounds'] == 2
    assert events[0]['jump_meV'] == pytest.approx(JUMP, abs=1e-6)
    if walker == 'cartesian':
        assert events[0]['trust'] == info['history'][0]['trust']
    else:
        assert events[0]['trust'] is None
    at_minimum(mol, info, water[1], -0.5 * Q_S1 ** 2 + JUMP / HARTREE_TO_MEV)
    # the stage record keeps the rounds and the jump
    kept = json.loads(json.dumps(walk_fields(info)))
    assert kept['rounds'] == 2
    assert kept['convention_refreezes'][0]['round'] == 2
    assert kept['convention_refreezes'][0]['jump_meV'] == pytest.approx(JUMP)


@pytest.mark.parametrize('walker', ['cartesian', 'geometric'])
def test_a_refreeze_within_the_tolerance_continues_the_round(water, walker,
                                                             tmp_path):
    """A lift of DRIFT meV, within REFREEZE_TOL_MEV, is the old surface
    continued: the same round goes on and converges on the rebuilt one."""
    if walker == 'geometric':
        pytest.importorskip('geometric')
    s = surface(water, [], lifts=(0.0, DRIFT / HARTREE_TO_MEV))
    mol, info = run(walker, s, water[0], tmp_path)
    assert info['converged'], info['status']
    events = info['convention_refreezes']
    assert len(events) == 1 and events[0]['refrozen']
    assert events[0]['round'] is None and info['rounds'] == 1
    assert events[0]['trust'] is not None
    at_minimum(mol, info, water[1], -0.5 * Q_S1 ** 2 + DRIFT / HARTREE_TO_MEV)


@pytest.mark.parametrize('ranks', [1, 2])
def test_every_rank_ends_the_walk_when_the_rounds_run_out(water, tmp_path,
                                                          monkeypatch, ranks):
    """With one round allowed, a jump ends the walk flagged, on the surface
    and at the energy of its last accepted step, the same on every rank."""
    pytest.importorskip('geometric')
    monkeypatch.setattr(optimize_module, 'WALK_CONVENTION_REFREEZES_MAX', 1)

    def walk(comm):
        s = surface(water, [], lifts=(0.0, JUMP / HARTREE_TO_MEV))
        work = tmp_path / f'rank{comm.Get_rank()}'
        mol, info = run('geometric', s, water[0], work)
        events = info['convention_refreezes']
        return (info['converged'], info['status'], info['energy'],
                info['final_surface'] is s, len(events), events[0]['refrozen'],
                events[0]['failed'], events[0]['e_before'],
                np.allclose(mol.atom_coords(), water[0].atom_coords()))

    out = run_simulated(walk, ranks)
    assert all(o == out[0] for o in out)
    (converged, status, energy, kept, n, refrozen, failed, e_before,
     still) = out[0]
    assert converged is False
    assert status.startswith('convention refreeze failed: the rounds ')
    assert failed.startswith('RefreezeRoundsExhausted: ')
    assert kept and n == 1 and refrozen is False and still
    assert energy == pytest.approx(e_before, abs=1e-12)


# -------------------------------------------------- consecutive refreezes
class Crossing:
    """A spring whose orbitals mix, at every geometry other than the keys',
    6 with 7 on the first surface and 7 with 8 on the second: the retry
    after the first refreeze is refused again before any step is
    accepted."""

    def __init__(self, mol, target, groups=(GROUP,), explicit=EXPLICIT,
                 generation=0, lifts=(0.0,)):
        self.mol0, self.target, self.generation = mol, target, generation
        self.lifts = lifts
        c = FrozenConventions(key_mol=mol, key_mo=np.eye(NORB), nocc=NOCC)
        c.qp_set = np.array(explicit, int)
        c.qp_subspaces = (tuple(explicit), tuple(groups))
        c.outside_shift = {p: 0.1 if p < NOCC else -0.2 for p in range(NORB)
                           if p not in explicit}
        c.anchored = True
        self.conv = c

    def identity(self):
        return None

    def overlap(self, km, kmo, m, mo):
        moved = not np.array_equal(np.asarray(km.atom_coords()),
                                   np.asarray(m.atom_coords()))
        pair = ((6, 7), (7, 8), None)[min(self.generation, 2)]
        if not moved or pair is None:
            return np.eye(NORB)
        return rotation(NORB, *pair, MIX)

    def commit(self, mol):
        self.conv.commit(mol, types.SimpleNamespace(mo_coeff=np.eye(NORB)))

    def total_gradient(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        self.conv.at(mol, types.SimpleNamespace(mo_coeff=np.eye(NORB)))
        energy, grad = spring(mol.atom_coords(), self.target)
        energy += self.lifts[min(self.generation, len(self.lifts) - 1)]
        return grad, energy, {}

    def total_energy(self, mol=None, mf=None):
        return self.total_gradient(mol)[1]

    def refreeze(self, mol, mf=None):
        absorbed = self.conv.absorbed_at_keys()
        group = tuple(sorted({p for g in absorbed for p in g} | set(GROUP)))
        out = Crossing(mol, self.target, groups=(group,),
                       explicit=tuple(sorted(set(EXPLICIT) | set(group))),
                       generation=self.generation + 1, lifts=self.lifts)
        out.conv.absorbed = absorbed
        holder[0] = out
        return out


holder = [None]


@pytest.mark.parametrize('walker', ['cartesian', 'geometric'])
def test_a_retry_refused_again_refreezes_at_a_shorter_radius(
        water, walker, tmp_path, monkeypatch):
    if walker == 'geometric':
        pytest.importorskip('geometric')
    start, minimum = water
    holder[0] = Crossing(start, distances(minimum))
    monkeypatch.setattr(conventions, 'body_overlap',
                        lambda *a: holder[0].overlap(*a))
    mol, info = run(walker, holder[0], start, tmp_path)
    assert info['converged'], info['status']
    events = info['convention_refreezes']
    assert [e['absorbed'] for e in events] == [[[5, 6, 7]], [[5, 6, 7, 8]]]
    if walker == 'cartesian':
        assert events[1]['trust'] == pytest.approx(
            WALK_STEP_SHRINK * events[0]['trust'])
    else:
        # WALK_STEP_SHRINK times the refused retry's length, which
        # geomeTRIC (an RMS radius per atom) keeps near the first one's
        assert events[1]['trust'] < 2 * WALK_STEP_SHRINK * events[0]['trust']
    at_minimum(mol, info, minimum)


@pytest.mark.parametrize('walker', ['cartesian', 'geometric'])
def test_rounds_that_keep_jumping_end_the_walk_flagged(
        water, walker, tmp_path, monkeypatch):
    """Two refreezes, each JUMP meV up, with two rounds allowed: the first
    starts round 2, the second is past the rounds and ends the walk flagged
    on round 2's surface at its last accepted energy."""
    if walker == 'geometric':
        pytest.importorskip('geometric')
    monkeypatch.setattr(optimize_module, 'WALK_CONVENTION_REFREEZES_MAX', 2)
    start, minimum = water
    lift = JUMP / HARTREE_TO_MEV
    holder[0] = Crossing(start, distances(minimum), lifts=(0.0, lift, 2 * lift))
    monkeypatch.setattr(conventions, 'body_overlap',
                        lambda *a: holder[0].overlap(*a))
    mol, info = run(walker, holder[0], start, tmp_path)
    assert not info['converged']
    assert info['status'].startswith('convention refreeze failed: the rounds ')
    events = info['convention_refreezes']
    assert [e['refrozen'] for e in events] == [True, False]
    assert events[0]['round'] == 2 and info['rounds'] == 2
    assert events[1]['failed'].startswith('RefreezeRoundsExhausted: ')
    assert events[1]['jump_meV'] == pytest.approx(JUMP, abs=1e-6)
    assert info['final_surface'].generation == 1
    assert info['energy'] == pytest.approx(events[1]['e_before'], abs=1e-12)
