"""The ground state's PCM and the optical response sit on one cavity
(`SolventScreening.check_ground_state_continuum`): a mean field whose PCM was
converged on another radius table than the environment's is refused, also
when the environment keeps pyscf's default table (it was accepted then), and
the same table spelled out is accepted.

Water 6-31G (bent, non-symmetric), toluene, IEF-PCM.

Run: python -m pytest tests/test_ground_state_cavity_check.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pyscf import solvent as pyscf_solvent
from pyscf.data import radii as pyscf_radii
from pyscf.solvent import pcm as pyscf_pcm

from src.environments.solvent_screening import SolventScreening
from tests.adc_common import ATOMS, make_scf_tight, molecule


def ground_state_pcm(env, radii_table):
    mf = pyscf_solvent.PCM(make_scf_tight(env.mol))
    pcm = mf.with_solvent
    pcm.eps, pcm.method = env.eps_static, env.method
    pcm.lebedev_order = env.cavity_settings['lebedev_order']
    pcm.radii_table = radii_table
    return pcm


def test_a_ground_state_on_another_radius_table_is_refused():
    env = SolventScreening(molecule(ATOMS), solvent='toluene')
    with pytest.raises(ValueError, match='radii_table'):
        env.check_ground_state_continuum(ground_state_pcm(env, 1.1 * pyscf_radii.UFF))
    env.check_ground_state_continuum(ground_state_pcm(env, 1.2 * pyscf_pcm.modified_Bondi))
    env.check_ground_state_continuum(ground_state_pcm(env, None))
