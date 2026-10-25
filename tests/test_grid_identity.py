"""One grid identity for an ISDF grid, carried by grid_record and checked on resume.

An ISDF level is a NAME for rows of the shipped table, and a table re-cut
under its names changes their content: at cc-pVDZ 'G3' is (24, 15, 9, 3),
where (32, 20, 12, 4) is 'GH'. A checkpoint or record stamped with the name
alone would resume or compares a run on another grid
without a word. `grid_identity` is the level with each element's counts and
the sha256 of the radii they place; `grid_record` carries it, and
`refuse_other_grid` refuses a stored identity other than the one a run would
use -- or none at all -- naming both.
"""
import json

import pytest
from pyscf import gto

from src.Base.declaration import GroundState, QPStates
from src.Base.numerics.numerics import numerics_of
from src.Base.factorization.separable_ri import grid_identity, refuse_other_grid
from src.properties.excitations import SurfaceSpec, grid_record

BASIS = 'cc-pvdz'
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def old_g3():
    """A stale record: cc-pVDZ 'GH' content under the level name 'G3'."""
    return dict(grid_identity('GH', BASIS, ['H', 'O']), level='G3')


def test_a_level_recut_under_its_name_is_another_identity():
    new = grid_identity('G3', BASIS, ['O', 'H', 'H'])
    assert new['level'] == 'G3'
    assert new['counts'] == {'H': [24, 15, 9, 3], 'O': [24, 15, 9, 3]}
    assert new != old_g3()
    assert new['radii_sha256'] != old_g3()['radii_sha256']
    # the same grid by explicit counts is the same content, without a level
    explicit = grid_identity(None, BASIS, ['H', 'O'],
                             counts={'A1': 24, 'A2': 15, 'A3': 9, 'B1': 3})
    assert explicit == dict(new, level=None)


def test_a_stored_identity_other_than_the_runs_is_refused_naming_both():
    want = grid_identity('G3', BASIS, ['H', 'O'])
    # what a checkpoint reads back is JSON
    refuse_other_grid(json.loads(json.dumps(want)), want, 'vert.json')
    with pytest.raises(RuntimeError) as err:
        refuse_other_grid(old_g3(), want, 'vert.json')
    message = str(err.value)
    assert 'vert.json' in message
    assert 'G3 (H 32,20,12,4' in message and 'G3 (H 24,15,9,3' in message
    with pytest.raises(RuntimeError, match="level name 'G3' alone"):
        refuse_other_grid({'level': 'G3'}, want, 'vert.json')
    with pytest.raises(RuntimeError, match='no grid identity'):
        refuse_other_grid(None, want, 'vert.json')


def spec(level):
    return SurfaceSpec(GroundState('rpa', 'hf'), chi0='space-time',
                       residues='sop', factorization='isdf',
                       qp_states=QPStates('frontier'),
                       numerics=numerics_of({'grid_accuracy': level}))


def test_grid_record_carries_the_identity():
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    g3, gh = grid_record(spec('G3'), mol), grid_record(spec('GH'), mol)
    assert g3['grid_identity'] == grid_identity('G3', BASIS, ['H', 'O'])
    assert gh['grid_identity'] == grid_identity('GH', BASIS, ['H', 'O'])
    assert g3['grid_identity']['radii_sha256'] \
        != gh['grid_identity']['radii_sha256']
