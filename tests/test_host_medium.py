"""A continuum by any static constant and n^2 on one named cavity convention
(src.environments.medium), the same medium in every solvent loop.

Water 6-31G (bent, non-symmetric), frozen 1, cc-pvdz-ri / cc-pvdz-jkfit.

  1. a medium without both constants is refused (only eps_s, only n^2, n^2
     above eps_s), as are an unknown solvent name, a malformed spec and an
     unknown cavity convention; 'gas' is no medium.
  2. the record carries both constants, the name or label and the cavity
     (convention, radius set, scale); `describe` reads the same off the
     continuum a medium builds.
  3. the resume key (`ContinuumField.identity`) moves with eps_s, with n^2
     and with the cavity convention; a solvent by name and by its numbers is
     one key; the modified Bondi set at 1.2 is the key a continuum built
     before cavities were named carries.
  4. each convention's spheres: the radius of O and H is the set's radius
     times its scale; the COSMO set refuses an element it has no radius for.
  5. eps_s = n^2 = 1 on every cavity convention is the gas phase: no
     response, the SCF's energy, a zero optical kernel (scheme A), and the
     ADC(2) S1 loop (G_eq, cycle count, force) to a few ulp.

Run: python -m pytest tests/test_host_medium.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto
from pyscf.data import radii as pyscf_radii
from pyscf.solvent import pcm as pyscf_pcm

from src.Base.numerics.constants import BOHR_TO_ANGSTROM, COSMO_RADII_ANGSTROM
from src.environments import solvation_loop as sl
from src.environments.medium import (CAVITY_CONVENTIONS, DEFAULT_CAVITY, Medium,
                                     describe, parse)
from src.environments.reaction_field import ContinuumField, NoMedium
from src.environments.solvent_screening import SolventScreening
from src.gradients.adc.ee_state_model import ADC2State
from src.gradients.state_models import HFState, MP2State
from tests.adc_common import (ADC_TIGHT, ADC_NUMERICS, ATOMS, AUX, NF, make_scf_loop,
                              make_scf_tight, molecule)
from tests.pins import assert_same

MOL = molecule(ATOMS)
FILM = 'eps=5.0,n2=2.89,label=film'


@pytest.mark.parametrize('spec', ['eps=5.0', 'n2=2.89', 'eps=2.0,n2=2.5',
                                  'no-such-solvent', 'eps=5,n2=2,n3=1',
                                  'eps=5.0,eps=4.0,n2=2.0'])
def test_a_medium_without_both_constants_is_refused(spec):
    with pytest.raises(ValueError):
        parse(spec)


def test_an_unknown_cavity_is_refused_and_gas_is_no_medium():
    with pytest.raises(ValueError):
        parse('toluene', cavity='bondi-1.3')
    assert parse('gas') is None
    with pytest.raises(ValueError):
        Medium.constants(5.0, None)


def test_the_record_carries_the_medium():
    m = parse(FILM, cavity='uff-1.1')
    assert m.record() == {'name': 'film', 'solvent': None, 'label': 'film',
                          'eps_static': 5.0, 'n2': 2.89, 'cavity': 'uff-1.1',
                          'radius_set': 'uff', 'vdw_scale': 1.1}
    d = describe(m.screening(MOL))
    assert (d['eps_static'], d['n2'], d['cavity'], d['radius_set'], d['vdw_scale']) == \
        (5.0, 2.89, 'uff-1.1', 'uff', 1.1)
    t = parse('toluene')
    assert t.record()['cavity'] == DEFAULT_CAVITY
    assert describe(t.screening(MOL))['solvent'] == 'toluene'
    assert parse('2.25:2.37').record()['n2'] == 2.25
    assert describe(None) is None


def key(medium):
    return ContinuumField.from_environment(medium.screening(MOL)).identity()


def test_the_resume_key_moves_with_every_part_of_the_medium():
    base = key(Medium.constants(5.0, 2.89))
    assert key(Medium.constants(5.0, 2.89)) == base
    assert key(Medium.constants(6.0, 2.89)) != base
    assert key(Medium.constants(5.0, 2.5)) != base
    keys = {c: key(Medium.constants(5.0, 2.89, cavity=c)) for c in CAVITY_CONVENTIONS}
    assert len({repr(k) for k in keys.values()}) == len(CAVITY_CONVENTIONS)
    for c, k in keys.items():
        assert (k['cavity'].get('radius_set') is None) == (c == 'bondi-1.2')
    # a solvent by name and by its numbers: one continuum
    tol = Medium.named('toluene')
    assert key(tol) == key(Medium.constants(tol.eps_static, tol.eps_optical))
    # the key of a continuum built before cavities were named
    old = SolventScreening(MOL, solvent='toluene')
    assert ContinuumField.from_environment(old).identity() == key(tol)


@pytest.mark.parametrize('cavity', sorted(CAVITY_CONVENTIONS))
def test_each_convention_draws_its_spheres(cavity):
    radius_set, scale = CAVITY_CONVENTIONS[cavity]
    env = Medium.constants(5.0, 2.89, cavity=cavity).screening(MOL)
    _, r = env.cavity_spheres()
    unscaled = {'bondi': lambda z: pyscf_pcm.modified_Bondi[z],
                'uff': lambda z: pyscf_radii.UFF[z],
                'cosmo': lambda z: COSMO_RADII_ANGSTROM[('', 'H', '', '', '', '',
                                                         '', '', 'O')[z]]
                / BOHR_TO_ANGSTROM}[radius_set]
    want = [scale * unscaled(int(z)) for z in MOL.atom_charges()]
    assert np.allclose(r, want, rtol=0, atol=1e-14)
    # the PCM the SCF relaxes in sits on the same spheres
    env.check_ground_state_continuum(
        env.mean_field(MOL, lambda m: make_scf_tight(m).run()).with_solvent)


def test_cosmo_refuses_an_element_without_a_radius():
    missing = next(s for s in ('Li', 'Na', 'Mg', 'Al') if s not in COSMO_RADII_ANGSTROM)
    mol = gto.M(atom=f'{missing} 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
    with pytest.raises(ValueError, match=missing):
        Medium.constants(5.0, 2.89, cavity='cosmo').screening(mol)
    Medium.constants(5.0, 2.89, cavity='uff-1.1').screening(mol)


def adc2_s1(field):
    mp2 = MP2State(HFState(MOL, make_scf_loop), NF, AUX, force_route='dense',
                   numerics=ADC_NUMERICS)
    return ADC2State(mp2, 'singlet', 0, 3, field.transition_environment())


@pytest.fixture(scope='module')
def gas_s1():
    gas = NoMedium(MOL)
    state = adc2_s1(gas)
    r = sl.equilibrate(state, gas, **ADC_TIGHT)
    return r, sl.force(state, gas, r)


@pytest.mark.parametrize('cavity', sorted(CAVITY_CONVENTIONS))
def test_eps_one_is_the_gas_phase_on_every_cavity(cavity, gas_s1):
    env = Medium.constants(1.0, 1.0, cavity=cavity).screening(MOL)
    field = ContinuumField.from_environment(env)
    assert not field.response_matrix('static').any()
    assert not field.response_matrix('fast').any()
    # scheme A's optical kernel and the ground state's reaction field
    assert not np.any(env.kernel_ao(MOL))
    gas_scf = make_scf_tight(MOL).run()
    assert_same(env.mean_field(MOL, lambda m: make_scf_tight(m).run()).e_tot,
                gas_scf.e_tot)
    # the ADC(2) state-specific loop
    state = adc2_s1(field)
    r = sl.equilibrate(state, field, **ADC_TIGHT)
    rg, fg = gas_s1
    assert len(r.history) == len(rg.history)
    assert_same(r.energy, rg.energy)
    assert_same(sl.force(state, field, r), fg, quantity='gradient')


def test_a_spec_names_its_own_cavity():
    assert parse('toluene,cavity=uff-1.1').cavity == 'uff-1.1'
    assert parse('eps=5.0,n2=2.89,cavity=cosmo').cavity == 'cosmo'
    assert parse('2.25:2.37,cavity=cosmo').cavity == 'cosmo'
    assert parse('toluene').cavity == DEFAULT_CAVITY
    assert parse('toluene', cavity='cosmo').cavity == 'cosmo'
    assert parse('toluene,cavity=cosmo', cavity='cosmo').cavity == 'cosmo'
    for bad, cavity in (('toluene,cavity=cosmo', 'uff-1.1'), ('toluene,eps=3', None),
                        ('toluene,cavity=x', None)):
        with pytest.raises(ValueError):
            parse(bad, cavity)


def test_the_stage_pipeline_reads_the_medium_from_its_environment(tmp_path):
    from dataclasses import replace
    from src.Base.utils.record_io import read_record
    from src.photophysics import chain
    from src.photophysics.store import Store
    from tests.photophysics_cases import declaration
    gas = declaration()
    with pytest.raises(ValueError, match='n\\^2'):
        replace(gas, environment='eps=5.0')
    d = replace(gas, environment=FILM + ',cavity=cosmo', solvent_scheme='scheme-a')
    mol = gto.M(atom=d.system.atom, basis=d.system.basis, verbose=0)
    assert describe(chain.continuum(d, mol))['cavity'] == 'cosmo'
    store = Store(tmp_path, d)
    store.write('s0', {'x': 1.0}, {'environment': d.environment})
    held = read_record(store.path('s0'))
    assert held['medium'] == parse(FILM, 'cosmo').record()
    assert Store(tmp_path, gas).medium is None
