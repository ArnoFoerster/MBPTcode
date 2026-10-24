"""The frozen conventions have one write path, and it refuses a write that
would land under the wrong orbital.

WHAT BREAKS IF THIS FAILS. Writing the quasiparticle branch, the inside scissor, the
pole-strength flags and the outside curve THROUGH the resolved view, which is the
storage while every orbital holds its own slot and a re-indexed copy
otherwise, puts a first solve's seed at a geometry where two orbitals have
traded slots into a discarded copy: `freeze_qp_branch` says "frozen here
for the first time" on every repeat and `qp_set_solve`'s loop never ends.
The view is read-only, every write is a method of `FrozenConventions` that
takes the view, and a
write through a view whose indices are not the slots is refused.

Planted: six orbitals, slots 2 and 3 held by each other's index (`perm`).
"""
import numpy as np
import pytest

from src.Base.numerics.numerics import Numerics
from src.gradients.gwbse.conventions import FrozenConventions, ResolvedConventions
from src.gradients.gwbse.excited_state import ExcitedStateChain

NORB, NOCC = 6, 3
#: slot -> index here: the HOMO and LUMO slots traded places
PERM = np.array([0, 1, 3, 2, 4, 5])


def planted(perm=PERM):
    """A chain with nothing built but the conventions and their view."""
    conv = FrozenConventions(key_mo=np.eye(NORB), nocc=NOCC,
                             qp_set=np.array([2, 3]))
    chain = ExcitedStateChain.__new__(ExcitedStateChain)
    chain.settings = Numerics()
    chain.conventions = conv
    chain.resolved_view = ResolvedConventions(conv, perm, {})
    chain.scissor, chain.residue_route = None, 'cd'
    return chain


def test_a_first_seed_at_a_permuted_geometry_is_refused_not_lost():
    chain = planted()
    route_out = {'roots': {2: -0.31}, 'weights': {2: 0.9},
                 'sop_poles': {2: [0.5, 0.9]}}
    with pytest.raises(AssertionError, match='other slots'):
        chain.quasiparticles.freeze_qp_branch(route_out, np.zeros(NORB), 0.0, [2])
    assert not chain.conventions.qp_seeds
    assert not chain.conventions.qp_seed_weights


def test_a_seed_already_frozen_is_read_at_a_permuted_geometry():
    chain = planted()
    chain.conventions.qp_seeds[3] = -0.31
    chain.conventions.qp_seed_weights[3] = 0.9
    route_out = {'roots': {2: -0.30}, 'weights': {2: 0.8}}
    assert chain.quasiparticles.freeze_qp_branch(route_out, np.zeros(NORB), 0.0,
                                       [2]) is False
    assert chain.conventions.qp_seeds == {3: -0.31}
    assert dict(chain.resolved.qp_seeds) == {2: -0.31}


def test_the_view_is_read_only():
    view = planted(perm=None).resolved
    with pytest.raises(TypeError):
        view.qp_seeds[2] = -0.31
    with pytest.raises(TypeError):
        view.scissor_map[2] = 0.01
    with pytest.raises(AttributeError):
        view.qp_seed_weights.clear()
    with pytest.raises(ValueError):
        view.qp_set[0] = 1


@pytest.mark.parametrize('write', [
    lambda c, v: c.freeze_scissor(v, 2, 0.01),
    lambda c, v: c.flag_z_outside(v, 2, {'reason': 'planted'}),
    lambda c, v: c.set_explicit(v, np.array([3])),
    lambda c, v: c.freeze_outside_shift(v, {0: 0.1}, {0: (2,)}),
    lambda c, v: c.freeze_outside_curves(v, {}),
    lambda c, v: c.freeze_blocks(v, (2, 3), (), {}),
    lambda c, v: c.freeze_subspaces(v, (2, 3), ((2, 3),), {}),
    lambda c, v: c.freeze_group_poles(v, (2, 3), np.ones(2)),
    lambda c, v: c.freeze_group_seed(v, (2, 3), np.eye(2)),
    lambda c, v: c.promote(v, {4: (3, 4)}),
    lambda c, v: c.reopen(v, [2, 3, 4]),
], ids=['scissor', 'flag_z_outside', 'explicit', 'outside_shift', 'outside_curves',
        'blocks', 'subspaces', 'group_poles', 'group_seed', 'promote',
        'reopen'])
def test_every_index_keyed_write_is_refused_through_a_permuted_view(write):
    chain = planted()
    before = chain.conventions.to_record()
    with pytest.raises(AssertionError, match='other slots'):
        write(chain.conventions, chain.resolved)
    after = chain.conventions.to_record()
    assert before.keys() == after.keys()
    for k in before:
        assert repr(before[k]) == repr(after[k]), k


def test_a_view_of_another_object_is_refused():
    chain = planted(perm=None)
    other = FrozenConventions(key_mo=np.eye(NORB), nocc=NOCC)
    with pytest.raises(AssertionError, match='another conventions'):
        other.freeze_scissor(chain.resolved, 2, 0.01)


def test_writes_through_the_storage_view_land_in_the_storage():
    chain = planted(perm=None)
    route_out = {'roots': {2: -0.31}, 'weights': {2: 0.9}}
    assert chain.quasiparticles.freeze_qp_branch(route_out, np.zeros(NORB), 0.0, [2])
    assert chain.conventions.qp_seeds == {2: -0.31}
    assert chain.conventions.qp_seed_weights == {2: 0.9}
    assert not chain.quasiparticles.freeze_qp_branch(route_out, np.zeros(NORB), 0.0,
                                           [2])
