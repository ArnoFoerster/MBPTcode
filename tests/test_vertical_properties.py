"""The vertical stage's property set (--properties).

Gates:
  * --properties energies runs no spin-orbit pass and no coupling (both stubbed
    to fail), keeps the energies, oscillator strengths and dipoles, and its
    energies are the full property set's (formaldehyde LRC-wPBEh/cc-pVDZ,
    ISDF-K GH, the pole model, R0 the input geometry);
  * the vert stamp carries 'properties' only away from 'all', so every
    earlier record keeps its stamp;
  * a stage that reads the spin-orbit table or a coupling (the walks, vib,
    rate) is refused before any SCF under --properties energies.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.photophysics import stages
from src.photophysics.records import StageRefused
from src.photophysics.stages import run_tadf
from tests.photophysics_cases import declaration, refusing, run_in

ENERGIES = ('--properties', 'energies', '--r0', 'input', '--qp-states',
            'admitted')


def vert(tmp_path, *extra):
    d = declaration(*extra)
    return run_tadf(d, run_in(tmp_path, d).store, max_memory=4000,
                    stages=('vert',))['vert']


def test_energies_skip_the_spin_orbit_pass(tmp_path, monkeypatch):
    full = vert(tmp_path / 'all', '--r0', 'input', '--qp-states', 'admitted')
    monkeypatch.setattr(stages, 'spin_orbit_couplings',
                        refusing('the spin-orbit pass ran'))
    monkeypatch.setattr(stages, 'analytic_coupling',
                        refusing('a coupling ran'))
    monkeypatch.setattr(stages, 'analytic_ground_coupling',
                        refusing('a ground coupling ran'))
    rec = vert(tmp_path / 'energies', *ENERGIES)
    props = rec['props']
    for name in ('socme', 'soc_v0', 'd_ground', 'nac', 'soc_dv_cart'):
        assert props[name] is None, name
    assert np.allclose(props['f_osc'], full['props']['f_osc'], atol=1e-10)
    for name, w in full['props']['omega'].items():
        assert props['omega'][name] == pytest.approx(w, abs=1e-10), name


def test_the_stamp(tmp_path):
    full = run_in(tmp_path / 'a', declaration()).stamp.stamps['vert']
    part = run_in(tmp_path / 'e', declaration(*ENERGIES)).stamp.stamps['vert']
    assert 'properties' not in full and part['properties'] == 'energies'


@pytest.mark.parametrize('stage', ['s1', 't1', 'rate'])
def test_a_stage_reading_couplings_is_refused(tmp_path, stage):
    d = declaration(*ENERGIES)
    with pytest.raises(StageRefused, match='--properties energies'):
        run_tadf(d, run_in(tmp_path, d).store, max_memory=4000,
                 stages=('vert', stage))
