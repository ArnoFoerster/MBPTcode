"""A quasiparticle-Hamiltonian group carried from the previous surface keeps a
record of every member it loses, including one that is not in the
explicit set.

WHAT BREAKS IF THIS FAILS. `merged_blocks` joins only the members of the
explicit set, so a carried member that left the set (a refrozen selection
that made it a hole) would vanish from its group without a word, while a
member that left the pole model is recorded in
`qp_subspace_record['dropped']`. Both are recorded and warned about the same
way.

Planted: eight orbitals, four occupied, set 2-5, eigenvalues far apart (no
group of their own), carried groups (4, 5) and (5, 6); orbital 6 is not
explicit.
"""
import numpy as np
import pytest

from src.gradients.conventions import FrozenConventions
from src.gradients.excited_state import ExcitedStateChain

EPS = np.array([-20.0, -10.0, -5.0, -1.0, 1.0, 2.0, 3.0, 9.0])
NOCC = 4
SET = np.array([2, 3, 4, 5])
CARRIED = ((4, 5), (5, 6))


def planted():
    chain = ExcitedStateChain.__new__(ExcitedStateChain)
    chain.conventions = FrozenConventions(
        key_mo=np.eye(len(EPS)), nocc=NOCC, qp_set=SET,
        carried_subspaces=CARRIED)
    chain.nocc = NOCC
    return chain


def test_a_carried_member_outside_the_set_is_recorded():
    chain = planted()
    routes = {int(p): 'sop' for p in SET}
    with pytest.warns(RuntimeWarning, match=r'\b6\b'):
        groups = chain._frozen_qp_subspaces(EPS, routes)
    assert groups == ((4, 5),)
    dropped = chain.conventions.qp_subspace_record['dropped']
    lost = [d for d in dropped if d['orbital'] == 6]
    assert lost and lost[0]['carried'] and lost[0]['subspace'] == [5, 6]
    assert lost[0]['route'] is None
    assert 'explicit' in lost[0]['reason']


def test_nothing_is_recorded_when_every_carried_member_is_explicit():
    chain = planted()
    chain.conventions.carried_subspaces = ((4, 5),)
    routes = {int(p): 'sop' for p in SET}
    assert chain._frozen_qp_subspaces(EPS, routes) == ((4, 5),)
    assert chain.conventions.qp_subspace_record['dropped'] == []
