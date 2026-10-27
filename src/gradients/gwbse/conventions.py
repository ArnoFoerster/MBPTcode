"""Every discrete choice an excited-state surface freezes, in one object keyed
by orbital identity.

A BSE@GW surface is E(R; C) with C a set of discrete conventions fixed at a
reference geometry, because each is a step in the surface if re-decided per
geometry: the explicit quasiparticle set and the roots of it flagged at Z
outside (0, 1],
the adaptive partition and the probe each hole borrows, the roots each
state follows (the reference's, then each accepted geometry's, with the
labels of the poles bracketing them) and their pole strengths, pole-model
positions and verdicts, the inside scissor, the
truly degenerate blocks and the quasiparticle-Hamiltonian groups with their
common poles and seeds, the scissor outside the set (tiers or curve), and the
contour-deformation quadrature. `FrozenConventions` holds all of them.

ORBITAL IDENTITY, NOT INDEX. Every convention is stored per SLOT, the index
an orbital had at the reference geometry, and each slot carries its orbital
(the key). At a displaced geometry `at` matches the orbitals there to the
keys by overlap and returns the index view the kernels read
(`ResolvedConventions`). The match is by UNITS, not by orbitals one at a
time: a near-degenerate group of the explicit set (a subspace of the
quasiparticle Hamiltonian, or a truly degenerate block) is matched by its
span, the projector overlap of the orbitals there with the group's keys, so
orbitals rotating into each other inside a group change nothing; every other
explicit orbital is a unit of its own; the holes and outside orbitals that
carry one and the same frozen shift are one unit per side of the gap, since
how they rotate among themselves changes nothing either. An orbital of the
displaced geometry belongs to the unit holding most of it -- compared in
atomic orbitals that move and turn with their atoms (`body_overlap`), so
neither a rigid motion nor an orbital simply following its atoms moves
anything -- and the
conventions are REFUSED (`ConventionsLost`, a step too long) when a unit
does not get back as many orbitals as it has, or one of them holds less than
CONVENTION_SPAN_MIN of its weight in the unit's span: the orbital left its
group's span, mixed with another unit, and which convention it carries is
no longer a function of the geometry. Past the crossing, held again by one
unit, it carries that unit's conventions.

THE KEYS TRAVEL WITH THE WALK. `commit` (an accepted step) makes the orbitals
matched there the keys, so every match is between neighbouring geometries
and the overlaps stay near one over a long walk; a group's frozen
Hamiltonian seed is carried into the new keys' basis by the rotation between
them, and each quasiparticle's followed branch becomes the root its solve kept
there (`hold_qp_branch`). A permutation found at an accepted step stays with
the slots.

At the geometry the keys belong to, with the same orbitals, the view reads
the storage itself; at a displaced geometry where every unit is matched to
the indices it holds it reads the storage as well, so a finite difference
around the reference reads the conventions the reference froze, bitwise.

ONE WRITE PATH. The view is read-only. A convention is written by a method of
`FrozenConventions` that takes the view it was decided in, and only where
that view's indices are the slots (`perm` None): a convention decided where
an orbital holds another slot would be stored under the wrong orbital, so it
is refused (`ConventionsWriteRefused`) instead.
"""
import collections
import copy
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.Base.numerics.constants import (CONVENTION_SPAN_MIN, EVALUATION_CACHE_SIZE,
                                         HARTREE_TO_EV)
from src.Base.geometry.geometry import molecule_at
from src.Base.step_failed import StepFailed
from src.Base.utils.parallel.mpi_grid import lockstep
from src.SingleReference.GW.qp_selection import AdaptivePartition
from src.gradients.evaluation import geometry_hash
from src.gradients.state_identity import body_overlap


class ConventionsLost(StepFailed):
    """An orbital of the displaced geometry left the span of the frozen unit
    it belongs to, so the convention it carries cannot be told."""


class ConventionsWriteRefused(AssertionError):
    """A convention written through a view whose indices are not the slots,
    where it would be stored under another orbital."""


@dataclass(frozen=True)
class Unit:
    """Orbitals matched together: `slots` at the keys' geometry; `kind` is
    'group' (a near-degenerate group of the explicit set), 'explicit' (one
    explicit orbital) or 'shared' (holes and outside orbitals carrying one
    frozen treatment, matched as a whole)."""

    kind: str
    slots: tuple


@dataclass(eq=False)
class FrozenConventions:
    """The frozen conventions of one surface, per slot (module docstring).

    Fields named as the chain has always exposed them; the chain's
    attributes of the same names are views of these.
    """

    key_mol: object = None
    key_mo: np.ndarray = None
    nocc: int = None
    # the explicit set, and the candidates (or declared set) it came from
    qp_set: np.ndarray = None
    window: tuple = ()
    # explicit roots of pole strength outside (0, 1] at the reference, kept
    # as solved: {slot: record}
    qp_z_outside: dict = field(default_factory=dict)
    qp_set_settled: bool = False
    # the adaptive partition (holes and probes) and how it was chosen
    qp_partition: object = None
    qp_selection: dict = None
    # grow-only refreeze: the explicit set carried from the surface before
    qp_floor: tuple = None
    # the quasiparticle branch: each state's root at the reference, its pole
    # strength, the labels of the poles bracketing it (None where the poles
    # are unknown) and the pole positions; and the root and labels the last
    # accepted geometry kept, {slot: (w, labels)}, which the next geometry
    # follows (`commit`)
    qp_seed_weights: dict = field(default_factory=dict)
    qp_seeds: dict = field(default_factory=dict)
    qp_seed_brackets: dict = field(default_factory=dict)
    qp_follow: dict = field(default_factory=dict)
    sop_poles: dict = field(default_factory=dict)
    # inside scissor: frozen shifts of set members Eq. (27) excludes
    scissor_map: dict = field(default_factory=dict)
    # (set key, positions) of the truly degenerate blocks, and the record
    qp_blocks: tuple = None
    qp_block_record: dict = None
    # (set key, groups) of the quasiparticle Hamiltonian, the groups carried
    # from the surface before, and the record
    qp_subspaces: tuple = None
    carried_subspaces: tuple = ()
    qp_subspace_record: dict = None
    # per group (slot tuple): its common poles and its frozen Newton seed
    group_poles: dict = field(default_factory=dict)
    group_seeds: dict = field(default_factory=dict)
    # holes the quasiparticle Hamiltonian made explicit: {slot: group}
    group_promoted: dict = field(default_factory=dict)
    # outside the set: {slot: shift}, {slot: the probe slots it borrows
    # from}, {side: curve}, and what the surface before had
    outside_shift: dict = None
    outside_probe: dict = None
    outside_curve: dict = field(default_factory=dict)
    outside_shift_before: dict = None
    outside_curve_before: dict = None
    # the shape of the shift beyond the set (OUTSIDE_SCISSOR_MODES)
    outside_scissor: str = 'tiers'
    # the contour-deformation quadrature: its points, weights, the
    # imaginary-time grid it is the frequency axis of, w0 and the range the
    # tau axis is fitted on; its size, fixed or not, floor, and asked
    nu: np.ndarray = None
    wt: np.ndarray = None
    gw_grid: object = None
    w0_cd: float = None
    e_min_gw: float = None
    e_max: float = None
    nfreq_cd: int = None
    # the static W's tau count: the minimax count of the reference gap, or
    # the one a surface before this grew to and carried over exactly
    ntau_w: int = None
    cd_sized: bool = False
    nfreq_cd_floor: int = 0
    nfreq_cd_asked: int = None
    # whether a forward has frozen anything yet, and where (`geometry_hash`
    # of the reference geometry): before it the keys follow another mean
    # field at the keys' geometry
    anchored: bool = False
    anchored_at: str = None
    # every accepted step that moved a slot: {'step', 'moved': {slot: index}}
    moves: list = field(default_factory=list)
    commits: int = 0

    def __post_init__(self):
        # {geometry_hash: {slot: (w, labels)}}: the branch each solve kept at
        # the last geometries, until `commit` adopts one (`hold_qp_branch`)
        self._held_branch = collections.OrderedDict()

    def to_record(self):
        """Every field, the keys' molecule as its coordinates (Bohr): what
        `from_record` rebuilds this object from (`record_io` writes it)."""
        out = {f.name: copy.deepcopy(getattr(self, f.name))
               for f in fields(self) if f.name != 'key_mol'}
        out['key_coords'] = (None if self.key_mol is None else
                             np.array(self.key_mol.atom_coords(), float))
        return out

    @classmethod
    def from_record(cls, record, mol):
        """The conventions `to_record` wrote, the keys' molecule rebuilt
        from `mol` (basis and charge) at their coordinates; nothing is
        derived again."""
        held = dict(record)
        if held.pop('qp_demoted', None):
            # written when such a root left the set for the scissor: another
            # surface than one that keeps it
            raise ValueError(
                'these conventions demoted a quasiparticle root of pole '
                'strength outside (0, 1] to the scissor; such a root is now '
                'kept as solved, so the record is another surface: build it '
                'again')
        coords = held.pop('key_coords')
        # a record before the one root finder: its weights are no input now
        held.pop('pole_offsets', None)
        return cls(key_mol=None if coords is None else molecule_at(mol, coords),
                   **held)

    def live(self):
        """The view that IS the storage: the slots are this geometry's indices."""
        return ResolvedConventions(self, None, {})

    def at(self, mol, mf):
        """The `ResolvedConventions` at (mol, mf): the storage itself where
        every unit is matched to the indices it holds, a re-indexed copy
        where some unit moved, `ConventionsLost` where one cannot be matched.
        The view keeps the molecule it resolved (`ResolvedConventions.
        geometry`)."""
        view = self._at(mol, mf)
        view.mol = mol
        return view

    def _at(self, mol, mf):
        if self.key_mo is None:
            return self.live()
        mo = np.asarray(mf.mo_coeff, float)
        if not self.anchored and np.array_equal(
                np.asarray(mol.atom_coords()),
                np.asarray(self.key_mol.atom_coords())):
            self.key_mol, self.key_mo = mol, mo.copy()
            return self.live()
        # rank 0 decides, so every rank resolves through the same path
        if lockstep(np.array([np.array_equal(
                np.asarray(mol.atom_coords()),
                np.asarray(self.key_mol.atom_coords()))
                and np.array_equal(mo, self.key_mo)]))[0]:
            return self.live()
        t = lockstep(body_overlap(self.key_mol, self.key_mo, mol, mo))
        perm = self._matched(t)
        rotations = {g: _polar(t[np.ix_(g, perm[list(g)])])
                     for g in self.group_slots()}
        if np.array_equal(perm, np.arange(len(perm))):
            return ResolvedConventions(self, None, rotations)
        return ResolvedConventions(self, perm, rotations)

    def keyed_copy(self):
        """These conventions with keys of their own: the frozen values shared
        by reference, the keys, the group seeds carried with them and the
        record of moves owned, so a commit on one copy re-keys no other."""
        out = copy.copy(self)
        out.group_seeds = dict(self.group_seeds)
        out.qp_follow = dict(self.qp_follow)
        out.moves = list(self.moves)
        return out

    def anchor(self, mol):
        """Mark the conventions frozen, at the geometry of `mol`, once."""
        if not self.anchored:
            self.anchored, self.anchored_at = True, geometry_hash(mol)

    def commit(self, mol, mf):
        """Make the orbitals matched at (mol, mf) the keys: an accepted step.
        Each group's frozen seed is carried into the new keys' basis, and
        each quasiparticle follows from here the root its solve kept at
        (mol, mf) (`hold_qp_branch`); one no longer held keeps the branch it
        had."""
        if self.key_mo is None or not self.anchored:
            return
        view = self.at(mol, mf)
        if self._held_branch:
            self.qp_follow.update(
                self._held_branch.get(view.geometry, {}))
        perm = view.permutation()
        for g, rot in view.rotations.items():
            if g in self.group_seeds and rot is not None:
                self.group_seeds[g] = rot.T @ self.group_seeds[g] @ rot
        self.key_mol = mol
        self.key_mo = np.asarray(mf.mo_coeff, float)[:, perm].copy()
        self.commits += 1
        if view.perm is not None:
            self.moves.append({'step': self.commits, 'moved': {
                int(s): int(p) for s, p in enumerate(perm) if s != p}})

    # Every write below takes the view it was decided in and stores by slot;
    # `_writable` admits only a view whose indices are the slots, so each
    # value is stored as given.
    def _writable(self, view, what):
        """Refuse writing `what` through `view` unless its indices are the
        slots of this object."""
        if view.conventions is not self:
            raise ConventionsWriteRefused(
                f'{what}: the view resolves another conventions object')
        if view.perm is not None:
            raise ConventionsWriteRefused(
                f'{what} would be frozen at a geometry where orbitals hold '
                f'other slots (slot: index here '
                f'{view.moved(range(len(view.perm)))}), and stored under '
                f'the wrong orbital. Conventions are decided at the '
                f'reference geometry; refreeze the surface here instead.')

    def freeze_qp_branch(self, view, weights, roots, sop_poles, labels=None):
        """Keep each orbital's first root (the one displaced geometries
        follow until a step is accepted), its pole strength, the labels of
        the poles bracketing it and its pole set, {index of `view`: value};
        True when a root was frozen here for the first time."""
        labels = {} if labels is None else labels
        new = sorted({int(p) for held, got in
                      ((self.qp_seed_weights, weights),
                       (self.qp_seeds, roots), (self.sop_poles, sop_poles))
                      for p in got if view.slot(p) not in held})
        if not new:
            return False
        self._writable(view, f'the quasiparticle branch of orbitals {new}')
        froze = any(int(p) not in self.qp_seeds for p in roots)
        for p, z in weights.items():
            self.qp_seed_weights.setdefault(int(p), float(z))
        for p, w in roots.items():
            if int(p) not in self.qp_seeds:
                self.qp_seeds[int(p)] = float(w)
                self.qp_seed_brackets[int(p)] = _labels(labels.get(p))
        for p, poles in sop_poles.items():
            self.sop_poles.setdefault(int(p), np.array(poles, float))
        return froze

    def hold_qp_branch(self, view, roots, labels):
        """Hold the root each orbital's solve kept at the geometry of `view`
        and the labels of its poles, {index of `view`: value}, by slot, until
        `commit` makes them the branch the next geometries follow; the last
        EVALUATION_CACHE_SIZE geometries are held, as their evaluations are.
        A solve repeated there replaces its orbitals' entries."""
        if view.geometry is None or not roots:
            return
        held = self._held_branch.setdefault(view.geometry, {})
        self._held_branch.move_to_end(view.geometry)
        for p, w in roots.items():
            held[view.slot(p)] = (float(w), _labels(labels.get(p)))
        while len(self._held_branch) > EVALUATION_CACHE_SIZE:
            self._held_branch.popitem(last=False)

    def freeze_scissor(self, view, p, shift):
        """The inside scissor of set member `p`."""
        self._writable(view, f'the inside scissor of orbital {p}')
        self.scissor_map[int(p)] = float(shift)

    def flag_z_outside(self, view, p, why):
        """Record that set member `p`'s reference root has a pole strength
        outside (0, 1]; the root is kept."""
        self._writable(view, f'the pole-strength flag of orbital {p}')
        self.qp_z_outside[int(p)] = why

    def set_explicit(self, view, states):
        """The explicit set, as indices of `view`."""
        self._writable(view, 'the explicit set')
        self.qp_set = states

    def promote(self, view, promoted):
        """Holes the quasiparticle Hamiltonian made explicit, {hole: group}."""
        self._writable(view, f'the holes promoted into groups {promoted}')
        self.group_promoted.update(promoted)

    def freeze_blocks(self, view, key, blocks, record):
        """The truly degenerate blocks of the set `key`, positions into it."""
        self._writable(view, 'the degenerate blocks')
        self.qp_block_record = record
        self.qp_blocks = (key, blocks)

    def freeze_subspaces(self, view, key, groups, record):
        """The quasiparticle-Hamiltonian groups of the set `key`."""
        self._writable(view, 'the quasiparticle-Hamiltonian groups')
        self.qp_subspace_record = record
        self.qp_subspaces = (key, groups)

    def record_groups(self, view, groups):
        """What each group was where the groups were decided, once."""
        if 'groups' in self.qp_subspace_record:
            return
        self._writable(view, 'the record of the groups')
        self.qp_subspace_record['groups'] = groups

    def freeze_group_poles(self, view, group, poles):
        """The common poles of the group whose orbitals here are `group`."""
        self._writable(view, f'the poles of group {list(group)}')
        self.group_poles[view.group_slots(group)] = poles

    def freeze_group_seed(self, view, group, h):
        """The Newton seed of a group, H in its orbitals here, which must be
        the keys' orbitals (no rotation between them)."""
        g = view.group_slots(group)
        self._writable(view, f'the seed of group {list(group)}')
        if view.rotations.get(g) is not None:
            raise ConventionsWriteRefused(
                f'the seed of group {list(group)} would be frozen in '
                f'orbitals rotated away from its keys; conventions are '
                f'decided at the reference geometry')
        self.group_seeds[g] = h.copy()

    def freeze_outside_shift(self, view, shift, probes):
        """The scissor outside the set, {orbital: shift}, and the probes each
        orbital borrows it from."""
        self._writable(view, 'the outside scissor')
        self.outside_probe = probes
        self.outside_shift = shift

    def freeze_outside_curves(self, view, curves):
        """The outside curves, {side: `OutsideCurve`}."""
        self._writable(view, 'the outside curves')
        self.outside_curve = curves

    def freeze_partition(self, view, partition):
        """The adaptive partition, holes and probes as indices of `view`."""
        self._writable(view, 'the adaptive partition')
        self.qp_partition = partition

    def record_selection(self, view, record):
        """The record of how the explicit set was chosen."""
        self._writable(view, 'the selection record')
        self.qp_selection = record

    def reopen(self, view, explicit):
        """Unfreeze what a solve at the reference geometry froze, so the set
        `explicit` is solved there again from cold: the partition, the
        settling, the quasiparticle branch, the poles, both scissors and the
        quadrature's size (rebuilt by the caller from `nfreq_cd_asked`).
        Kept: the holes promoted into groups, which are why the set is what it
        is, and the blocks and groups, keyed by the set they were decided for.
        The pole-strength flags go: the solve again flags its own roots."""
        self._writable(view, 'reopening the explicit set')
        self.qp_z_outside = {}
        self.qp_partition = None
        self.qp_set = np.asarray(sorted(int(p) for p in explicit), int)
        self.qp_set_settled = False
        self.outside_shift = None
        self.outside_probe = None
        self.outside_curve = {}
        self.scissor_map = {}
        self.qp_seed_weights.clear()
        self.qp_seeds.clear()
        self.qp_seed_brackets.clear()
        self.qp_follow.clear()
        self._held_branch.clear()
        self.sop_poles.clear()
        self.group_poles.clear()
        self.group_seeds.clear()
        self.cd_sized = False

    def group_slots(self):
        """The quasiparticle-Hamiltonian groups, as slot tuples."""
        return tuple(self.qp_subspaces[1]) if self.qp_subspaces else ()

    def units(self):
        """The `Unit`s every orbital is matched in (module docstring)."""
        norb = self.key_mo.shape[1]
        explicit = [int(p) for p in self.qp_set]
        groups = list(self.group_slots())
        if not groups and self.qp_blocks:
            key, blocks = self.qp_blocks
            groups = [tuple(key[i] for i in b) for b in blocks]
        out = [Unit('group', tuple(g)) for g in groups if len(g) > 1]
        grouped = {p for u in out for p in u.slots}
        out += [Unit('explicit', (p,)) for p in explicit if p not in grouped]
        on_curve = {int(p): side for side, c in self.outside_curve.items()
                    for p in c.members}
        shared = {}
        solved = set(explicit)
        for p in range(norb):
            if p in solved:
                continue
            side = 'occupied' if p < self.nocc else 'virtual'
            if p in on_curve:
                what = ('curve', on_curve[p])
            else:
                shift = (None if self.outside_shift is None
                         else self.outside_shift.get(p))
                what = ('shift', None if shift is None else float(shift))
            shared.setdefault((side,) + what, []).append(p)
        out += [Unit('shared', tuple(m)) for _, m in sorted(
            shared.items(), key=lambda kv: kv[1][0])]
        return out

    def _matched(self, t):
        """perm[slot] = index at the displaced geometry, unit by unit, or
        `ConventionsLost` naming every unit that cannot be matched."""
        norb = t.shape[0]
        w = t ** 2
        perm = np.full(norb, -1)
        lost = []
        every = self.units()
        for occupied in (True, False):
            cols = (np.arange(self.nocc) if occupied
                    else np.arange(self.nocc, norb))
            units = [u for u in every if (u.slots[0] < self.nocc) == occupied]
            weight = np.array([w[list(u.slots)][:, cols].sum(axis=0)
                               for u in units])
            owner = np.argmax(weight, axis=0)
            for i, u in enumerate(units):
                got = cols[owner == i]
                held = weight[i, owner == i]
                if len(got) != len(u.slots) or (held.size and held.min()
                                                < CONVENTION_SPAN_MIN):
                    lost.append(_lost_unit(u, got, held, w, cols, units,
                                           owner, weight))
                    continue
                perm[list(u.slots)] = _paired(u, got, w)
        if lost:
            raise ConventionsLost(
                'the frozen conventions cannot be matched at this geometry: '
                + '; '.join(lost)
                + f' (each needs its own count of orbitals, each with at '
                f'least {CONVENTION_SPAN_MIN} in the span). The orbital '
                f'left its group; shorten the step or refreeze here.')
        return perm

    def rebuild(self, mol, mf):
        """What a surface rebuilt at (mol, mf) with the same settings starts
        from, in that geometry's indices: the window, the explicit set and
        the grow-only floor it carries, the groups it carries, and the
        outside scissor of this surface for the record of how far it
        moved."""
        view = self.at(mol, mf)
        return {'window': tuple(sorted(view.index(p) for p in self.window)),
                'explicit': tuple(int(p) for p in view.qp_set),
                'floor': view.qp_floor,
                'subspaces': tuple(tuple(sorted(view.index(p) for p in g))
                                   for g in self.group_slots()),
                'outside_shift_before': (None if view.outside_shift is None
                                         else dict(view.outside_shift)),
                'outside_curve_before': dict(view.outside_curve) or None}

    def outside_record(self):
        """What the orbitals outside the quasiparticle set carry, in eV, per
        slot: the shifts, and after a refreeze the ones the surface it was
        rebuilt from had, whose drift says whether the surface a walk ended
        on is the one the calibration would choose there."""
        out = {'outside': 'scissor'}
        if self.outside_scissor != 'tiers':
            out['outside_scissor'] = self.outside_scissor
            out['outside_curve_knots_eV'] = {
                s: c.as_record()['knots_eV']
                for s, c in sorted(self.outside_curve.items())}
        if self.outside_shift is None:
            return out
        now = {int(p): float(s) * HARTREE_TO_EV
               for p, s in sorted(self.outside_shift.items())}
        out['outside_shift_ev'] = now
        if self.outside_shift_before is not None:
            before = {int(p): float(s) * HARTREE_TO_EV
                      for p, s in sorted(self.outside_shift_before.items())}
            out['outside_shift_before_ev'] = before
            out['outside_shift_moved_ev'] = max(
                (abs(now[p] - before[p]) for p in now if p in before),
                default=0.0)
        return out

    def curve_record(self):
        """The frozen outside curves in eV, and the knots of the surface
        this one was refrozen from; None on tiers."""
        if self.outside_scissor == 'tiers':
            return None
        out = {'curves': {s: c.as_record()
                          for s, c in sorted(self.outside_curve.items())}}
        if self.outside_curve_before is not None:
            out['knots_before_eV'] = {
                s: c.as_record()['knots_eV']
                for s, c in sorted(self.outside_curve_before.items())}
        return out

    def record(self):
        """What is frozen, per slot, and what the walk did to the keys."""
        units = self.units() if self.key_mo is not None and \
            self.qp_set is not None else []
        return {
            'keyed_by': 'orbital overlap, keys carried along accepted steps',
            'explicit': [int(p) for p in self.qp_set] if self.qp_set
            is not None else None,
            'groups': [list(g) for g in self.group_slots()],
            'group_n_poles': {str(list(g)): int(len(p))
                              for g, p in self.group_poles.items()},
            'group_promoted': {int(p): list(g)
                               for p, g in sorted(self.group_promoted.items())},
            'units': {k: sum(u.kind == k for u in units)
                      for k in ('group', 'explicit', 'shared')},
            'span_min': CONVENTION_SPAN_MIN,
            'commits': self.commits,
            'moves': list(self.moves),
            'nfreq_cd': self.nfreq_cd, 'ntau_w': self.ntau_w}


@dataclass(eq=False)
class ResolvedConventions:
    """The conventions in the orbital indices of one geometry, read-only.

    perm None: the slots are this geometry's indices and every accessor
    reads the storage itself. Otherwise perm[slot] is the index here and
    the accessors return re-indexed copies. Writes go through
    `FrozenConventions`, which takes the view. rotations: {group slots: the
    rotation from the group's keys to its orbitals here, columns in the
    order perm pairs them with the slots}, empty at the keys' geometry.
    """

    conventions: FrozenConventions
    perm: np.ndarray = None
    rotations: dict = field(default_factory=dict)
    # the molecule resolved, None for a bare view
    mol: object = None

    @property
    def geometry(self):
        """The `geometry_hash` of the molecule resolved, None for a bare
        view: where a quasiparticle branch solved here is held
        (`FrozenConventions.hold_qp_branch`)."""
        return None if self.mol is None else geometry_hash(self.mol)

    def permutation(self):
        """perm, the identity where no unit moved."""
        if self.perm is not None:
            return self.perm
        return np.arange(self.conventions.key_mo.shape[1])

    def index(self, slot):
        """This geometry's index of a slot."""
        return int(slot) if self.perm is None else int(self.perm[int(slot)])

    def slot(self, index):
        """The slot of this geometry's index."""
        if self.perm is None:
            return int(index)
        return int(np.flatnonzero(self.perm == int(index))[0])

    def _keyed(self, mapping):
        if mapping is None:
            return None
        if self.perm is None:
            return MappingProxyType(mapping)
        return MappingProxyType({self.index(p): v for p, v in mapping.items()})

    @property
    def qp_set(self):
        """The explicit set, sorted."""
        c = self.conventions
        if c.qp_set is None:
            return None
        if self.perm is None:
            return _read_only(c.qp_set)
        return _read_only(np.array(sorted(self.index(p) for p in c.qp_set),
                                   int))

    @property
    def qp_seeds(self):
        """{orbital: frozen root, followed at displaced geometries}."""
        return self._keyed(self.conventions.qp_seeds)

    @property
    def qp_follow(self):
        """{orbital: (root, labels)}: the branch each state follows here,
        the root the last accepted geometry kept, else the reference's."""
        c = self.conventions
        held = {p: (w, c.qp_seed_brackets.get(p))
                for p, w in c.qp_seeds.items()}
        held.update(c.qp_follow)
        return self._keyed(held)

    @property
    def qp_seed_weights(self):
        """{orbital: pole strength of the frozen root}."""
        return self._keyed(self.conventions.qp_seed_weights)

    @property
    def sop_poles(self):
        """{orbital: frozen pole positions}."""
        return self._keyed(self.conventions.sop_poles)

    @property
    def scissor_map(self):
        """{orbital: frozen inside scissor}."""
        return self._keyed(self.conventions.scissor_map)

    @property
    def qp_z_outside(self):
        """{orbital: its reference root's record}, the explicit roots of
        pole strength outside (0, 1] (kept as solved)."""
        return self._keyed(self.conventions.qp_z_outside)

    @property
    def outside_shift(self):
        """{orbital: frozen outside scissor}, or None before calibration."""
        return self._keyed(self.conventions.outside_shift)

    @property
    def outside_probe(self):
        """{orbital: the explicit orbitals whose shift it borrows}."""
        probes = self.conventions.outside_probe
        if self.perm is None or probes is None:
            return self._keyed(probes)
        return MappingProxyType({self.index(p): tuple(self.index(q) for q in b)
                                 for p, b in probes.items()})

    @property
    def outside_curve(self):
        """{side: `OutsideCurve`}, members as this geometry's indices."""
        curves = self.conventions.outside_curve
        if self.perm is None:
            return MappingProxyType(curves)
        return MappingProxyType({
            s: replace(c, members=tuple(sorted(self.index(p)
                                               for p in c.members)))
            for s, c in curves.items()})

    def _set_key(self, key):
        return tuple(sorted(self.index(p) for p in key))

    @property
    def qp_blocks(self):
        """(set key, positions into the set) of the degenerate blocks."""
        blocks = self.conventions.qp_blocks
        if self.perm is None or blocks is None:
            return blocks
        key, found = blocks
        now = self._set_key(key)
        pos = {p: i for i, p in enumerate(now)}
        return (now, tuple(tuple(sorted(pos[self.index(key[i])] for i in b))
                           for b in found))

    @property
    def qp_subspaces(self):
        """(set key, groups) of the quasiparticle Hamiltonian, as indices."""
        subspaces = self.conventions.qp_subspaces
        if self.perm is None or subspaces is None:
            return subspaces
        key, groups = subspaces
        return (self._set_key(key),
                tuple(self._set_key(g) for g in groups))

    @property
    def carried_subspaces(self):
        """The groups carried from the surface before, as indices."""
        carried = self.conventions.carried_subspaces
        if self.perm is None:
            return carried
        return tuple(self._set_key(g) for g in carried)

    @property
    def qp_floor(self):
        """The explicit set carried grow-only, as indices."""
        floor = self.conventions.qp_floor
        if self.perm is None or floor is None:
            return floor
        return self._set_key(floor)

    @property
    def qp_partition(self):
        """The adaptive partition, holes and probes as indices."""
        part = self.conventions.qp_partition
        if self.perm is None or part is None:
            return part
        return AdaptivePartition(
            explicit=self._set_key(part.explicit),
            tier_of={self.index(h): self._set_key(b)
                     for h, b in part.tier_of.items()},
            candidates=self._set_key(part.candidates), targets=part.targets,
            tol_meV=part.tol_meV, provenance=part.provenance)

    def group_slots(self, indices):
        """The slot tuple of the group whose orbitals here are `indices`."""
        return tuple(sorted(self.slot(p) for p in indices))

    def group_poles(self, indices):
        """The frozen common poles of a group, or None."""
        return self.conventions.group_poles.get(self.group_slots(indices))

    def group_seed(self, indices):
        """The frozen Newton seed of a group in this geometry's basis of it,
        or None: the stored H carried by the rotation from the keys."""
        g = self.group_slots(indices)
        seed = self.conventions.group_seeds.get(g)
        rot = self.rotations.get(g)
        if seed is None or rot is None:
            return seed
        order = np.argsort([self.index(p) for p in g])
        return (rot.T @ seed @ rot)[np.ix_(order, order)]

    def moved(self, slots):
        """{slot: index here} for every slot of `slots` held by another index."""
        return {int(p): self.index(p) for p in slots
                if self.index(p) != int(p)}


def _labels(labels):
    """A root's pole labels as a tuple (lo, hi), None where unknown."""
    return None if labels is None else tuple(
        None if x is None else int(x) for x in labels)


def _read_only(a):
    """A read-only view of the array `a`."""
    out = np.asarray(a).view()
    out.flags.writeable = False
    return out


def _paired(unit, got, w):
    """The indices of `got` in the order of `unit.slots`: unchanged when the
    unit holds its own indices, else by the assignment of largest overlap
    for a group or an explicit orbital; a shared unit is uniform, sorted."""
    slots = list(unit.slots)
    got = np.sort(np.asarray(got, int))
    if set(int(p) for p in got) == set(slots):
        return np.array(slots, int)
    if unit.kind == 'shared':
        return got
    rows, cols = linear_sum_assignment(-w[np.ix_(slots, got)])
    out = np.empty(len(slots), int)
    out[rows] = got[cols]
    return out


def _lost_unit(unit, got, held, w, cols, units, owner, weight):
    """One unit's part of a `ConventionsLost` message: the orbitals it got
    here and how much of its span they hold, and for each slot whose largest
    overlap went to an orbital another unit took, that weight, the orbital
    and the unit that took it. A unit that got no orbital lost them by
    weight, its slot spread thinner over the orbitals here than the larger
    unit beside it."""
    i = units.index(unit)
    line = f'{unit.kind} {_runs(unit.slots)} gets orbitals {_runs(got)}'
    if not len(held):
        line += ' holding none of its span'
    elif len(unit.slots) == 1:
        line += f' holding {np.round(held, 3).tolist()} of its span'
    else:
        line += f' holding at least {held.min():.3f} of its span'
    gave = []
    for s in unit.slots:
        j = int(np.argmax(w[s, cols]))
        if owner[j] == i:
            continue
        other = units[owner[j]]
        gave.append(f'slot {s} is at most {w[s, cols[j]]:.3f} in one '
                    f'orbital here, {cols[j]}, of which {other.kind} '
                    f'{_runs(other.slots)} holds {weight[owner[j], j]:.3f}')
    return line + (f' ({"; ".join(gave)})' if gave else '')


def _runs(indices):
    """Orbital indices as sorted runs, '[22, 24-37]'."""
    out, idx = [], sorted(int(p) for p in indices)
    for p in idx:
        if out and p == out[-1][1] + 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return '[' + ', '.join(str(a) if a == b else f'{a}-{b}'
                           for a, b in out) + ']'


def _polar(o):
    """The orthogonal factor of o: the rotation closest to the overlap."""
    u, _, vt = np.linalg.svd(o)
    return u @ vt
