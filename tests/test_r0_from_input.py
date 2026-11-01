"""R0 as the input geometry (--r0 input), for a vertical on a frame.

Gates:
  * --r0 input makes the s0 stage record the input geometry with no SCF
    (the run builds no mean field for it), marks the record, and stamps 'r0'
    into every stage's stamp; --r0 walk (the default) stamps nothing new, so
    every walked run keeps its stamps;
  * a later stage takes the input R0 without --stages naming s0: formaldehyde
    LRC-wPBEh/cc-pVDZ, ISDF-K GH, the vertical stage on the continuation;
  * an R0 route outside R0_ROUTES is refused.
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.photophysics.declaration import DeclarationRefused
from src.photophysics.stages import ground_minimum, run_tadf
from tests.photophysics_cases import declaration, run_in

INPUT = ('--r0', 'input')
PADE = ('--residues', 'pade', '--qp-states', 'admitted')


def test_the_input_r0_is_recorded_with_no_scf(tmp_path):
    run = run_in(tmp_path, declaration(*INPUT))
    record = ground_minimum(run)
    assert run.histories == []
    assert record['r0'] == 'input' and record['e0_walk'] is None
    assert np.array_equal(np.asarray(record['coords_bohr']),
                          run.mol_in.atom_coords())
    assert run.store.has('s0')


def test_the_stamps(tmp_path):
    walk = run_in(tmp_path / 'w', declaration()).stamp.stamps
    given = run_in(tmp_path / 'i', declaration(*INPUT)).stamp.stamps
    assert all('r0' not in s for s in walk.values())
    assert all(s.get('r0') == 'input' for s in given.values())
    assert declaration().policies.r0 == 'walk'


def test_a_later_stage_takes_the_input_r0(tmp_path):
    d = declaration(*INPUT, *PADE)
    out = run_tadf(d, run_in(tmp_path, d).store, max_memory=4000,
                   stages=('vert',))
    assert out['s0']['r0'] == 'input'
    omega = out['vert']['props']['omega']
    assert 0 < omega['T1'] < omega['S1']


def test_another_r0_route_is_refused():
    d = declaration()
    with pytest.raises(DeclarationRefused, match='policies.r0'):
        replace(d, policies=replace(d.policies, r0='relaxed'))
