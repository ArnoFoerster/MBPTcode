"""A walk on a SolvatedSurface holds one step's state model at a time: the
previous geometry's model (its mean field, factors and fold operator) is
released before the next geometry's is built, and only the surface's own
carry-over (charges, followed root, fold start, surface key, flags) survives.
Water cc-pVDZ (bent, non-symmetric; above the dense limit, so the model holds
the RI fold operator a large walk does), frozen 1, gas, ADC(2) singlet S1, two
steps 0.01 Angstrom apart.

  1. while the second step builds its model, the first step's model and its
     fold operator are gone.
  2. the second step still starts from the first step's roots (fold_start),
     and `last` is the second step's model afterwards (commit, mean_field,
     refreeze read it).
  3. out of equilibrium (water 6-31G, toluene IEF-PCM, S0 Hartree-Fock, the
     ADC(2) triplet): while the second step builds S0's model, neither the
     first step's S0 model nor its triplet model is alive; `last_ground` is
     the second step's S0 model afterwards (the force reads it).

Run: python -m pytest -q tests/test_walk_state_release.py
"""
import os
import sys
import weakref

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.numerics import Numerics
from src.environments.reaction_field import ContinuumField, NoMedium
from src.environments.solvent_screening import SolventScreening
from src.gradients.adc.ee_state_model import ADC2State
from src.gradients.solvated_surface import SolvatedSurface
from src.gradients.state_models import HFState, MP2State
from src.SingleReference.ADC.eeADC import ee_ri_fold
from tests.adc_common import ATOMS, AUX, NF, make_scf_loop, make_scf_tight, molecule

NUMERICS = Numerics(adc={'conv_tol': 1e-10})
MOL1 = molecule(ATOMS, basis='cc-pvdz')
MOL2 = molecule(ATOMS, shift=np.full((len(ATOMS), 3), 0.01 / 0.529177210903),
                basis='cc-pvdz')


def walk():
    built, ops, alive = [], [], []
    init = ee_ri_fold.RIFoldADC2.__init__

    def recorded(op, *a, **k):
        ops.append(weakref.ref(op))
        return init(op, *a, **k)

    def make_state(m, field):
        # the models and fold operators built so far that are still referenced
        alive.append([ref() is not None for ref in built + ops])
        state = ADC2State(MP2State(HFState(m, make_scf_loop), NF, AUX, numerics=NUMERICS),
                          'singlet', 0, 1, None, follow='index')
        built.append(weakref.ref(state))
        return state

    ee_ri_fold.RIFoldADC2.__init__ = recorded
    try:
        surf = SolvatedSurface(MOL1, NoMedium, make_state, label='S1 gas')
        surf.total_gradient(MOL1)
        start = surf.fold_start
        n_ops = len(ops)
        surf.total_gradient(MOL2)
    finally:
        ee_ri_fold.RIFoldADC2.__init__ = init
    return surf, built, n_ops, alive, start


SURF, BUILT, N_OPS, ALIVE, START = walk()


def test_the_previous_model_is_gone_when_the_next_is_built():
    assert N_OPS >= 1, 'the first step built no RI fold operator'
    assert ALIVE == [[], [False] * (1 + N_OPS)], ALIVE


def test_the_carry_over_survives():
    assert START is not None
    assert SURF.last['state'] is BUILT[1]() and SURF.last['mol'] is MOL2


def toluene(mol):
    return ContinuumField.from_environment(SolventScreening(mol, solvent='toluene'))


def walk_noneq():
    grounds, states, alive = [], [], []

    def make_ground(m, field):
        # the S0 and triplet models built so far that are still referenced
        alive.append([ref() is not None for ref in grounds + states])
        model = HFState(m, make_scf_tight)
        grounds.append(weakref.ref(model))
        return model

    def make_state(m, field):
        state = ADC2State(MP2State(HFState(m, make_scf_tight), NF, AUX, numerics=NUMERICS),
                          'triplet', 0, 1, None, follow='index')
        states.append(weakref.ref(state))
        return state

    shift = np.full((len(ATOMS), 3), 0.01 / 0.529177210903)
    surf = SolvatedSurface(molecule(ATOMS), toluene, make_state, label='T1 toluene neq',
                           make_ground=make_ground, noneq=True)
    surf.total_gradient(molecule(ATOMS))
    surf.total_gradient(molecule(ATOMS, shift=shift))
    return surf, grounds, alive


def test_out_of_equilibrium_the_previous_models_are_gone():
    surf, grounds, alive = walk_noneq()
    assert alive == [[], [False, False]], alive
    assert surf.last_ground['state'] is grounds[1]()
