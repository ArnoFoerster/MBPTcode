"""The solvent loops in a medium that is not a named solvent, on a cavity that
is not the default (src.environments.medium): the gas limit of the GW-BSE
loop and finite-difference forces of both families.

  1. GW-BSE (scheme B): eps_s = n^2 = 1 on the COSMO cavity is the gas phase
     for S1, G_eq and its force to a few ulp (tests/pins.py assert_same).
  2. forces of G_eq along a random direction against central differences of
     converged loops (h = 5e-4 Bohr, each displaced point a whole loop on its
     own cavity), in eps_s 5.0, n^2 2.89 on the UFF x 1.1 cavity: the ADC(2)
     singlet (water 6-31G, its LR term included) and the GW-BSE singlet
     (water cc-pVDZ, the chain of test_gwbse_solvation_loop), 1e-6 Ha/Bohr.

Run: python -m pytest tests/test_host_medium_loops.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.environments import solvation_loop as sl
from src.environments.medium import Medium
from src.environments.reaction_field import ContinuumField, NoMedium
from src.gradients.adc.ee_state_model import ADC2State
from src.gradients.state_models import HFState, MP2State
from tests import test_gwbse_solvation_loop as gw
from tests.adc_common import (ATOMS, AUX, FD_STEP_SOLVATION, NF, ROOT_CONV_TOL,
                              SOLVATION_TIGHT, central, make_scf_tight, molecule)
from tests.pins import assert_same

FILM = Medium.constants(5.0, 2.89, cavity='uff-1.1', label='film')


def field_of(medium, mol):
    return ContinuumField.from_environment(medium.screening(mol))


def direction(natm):
    d = np.random.default_rng(17).standard_normal((natm, 3))
    return d / np.linalg.norm(d)


def test_gwbse_eps_one_on_the_cosmo_cavity_is_the_gas_phase():
    mol = gw.molecule()
    one = field_of(Medium.constants(1.0, 1.0, cavity='cosmo'), mol)
    _, s1 = gw.setup(mol, 'singlet', field=one)
    r1 = gw.loop(s1, one)
    gas = NoMedium(mol)
    _, sg = gw.setup(mol, 'singlet', field=gas)
    rg = gw.loop(sg, gas)
    assert len(r1.history) == len(rg.history)
    assert_same(r1.energy, rg.energy)
    assert_same(sl.force(s1, one, r1), sl.force(sg, gas, rg), quantity='gradient')


def adc2_singlet(mol, field):
    mp2 = MP2State(HFState(mol, make_scf_tight), NF, AUX)
    return ADC2State(mp2, 'singlet', 0, 3, field.transition_environment(),
                     conv_tol=ROOT_CONV_TOL)


def test_adc2_force_in_a_film_medium():
    mol = molecule(ATOMS)
    field = field_of(FILM, mol)
    state = adc2_singlet(mol, field)
    r = sl.equilibrate(state, field, **SOLVATION_TIGHT)
    assert r.converged
    d = direction(mol.natm)

    def G(shift):
        m = molecule(ATOMS, shift * d)
        f = field_of(FILM, m)
        out = sl.equilibrate(adc2_singlet(m, f), f, **SOLVATION_TIGHT)
        assert out.converged
        return out.energy
    an = float(np.sum(sl.force(state, field, r) * d))
    fd = central(G, FD_STEP_SOLVATION)
    assert abs(fd - an) < 1e-6, (an, fd)


def test_gwbse_force_in_a_film_medium():
    mol = gw.molecule()
    field = field_of(FILM, mol)
    _, state = gw.setup(mol, 'singlet', field=field)
    r = gw.loop(state, field)
    assert r.converged
    d = direction(mol.natm)

    def G(shift):
        m = gw.molecule(shift * d)
        f = field_of(FILM, m)
        out = gw.loop(state.at(m, gw.make_scf, f.transition_environment()), f)
        assert out.converged
        return out.energy
    an = float(np.sum(sl.force(state, field, r) * d))
    fd = central(G, FD_STEP_SOLVATION)
    assert abs(fd - an) < 1e-6, (an, fd)
