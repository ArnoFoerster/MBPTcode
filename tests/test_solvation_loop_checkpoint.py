"""Checkpoint restart of the solvation loop for the HF, MP2 and ADC(2) state models
(src.Base.solvation_loop.equilibrate(checkpoint=...)): water 6-31G in
toluene IEF-PCM, frozen 1, aux cc-pvdz-ri, JK cc-pvdz-jkfit, the loop at
tol_q 1e-10 e and tol_energy 1e-12 Ha. The GW-BSE twin is in
test_gwbse_solvation_loop.py.

  1. a loop stopped after three cycles and resumed by fresh state models and
     a fresh medium (as a new process builds them) gives the uninterrupted
     run's cycle history, G, charges, relaxed density and force of G_eq to
     a few ulp (the resumed cycles are computed again): HF, MP2 and the ADC(2) singlet with its LR term; and the
     non-equilibrium loop of the singlet in S0's slow charges.
  2. refusals: a state model without checkpoint() / restore() / identity(),
     and a checkpoint resumed by another loop, one axis at a time: slow
     charges, medium (water for toluene), tolerance, start, basis, and state
     (MP2 for HF, the ADC(2) triplet for the singlet).

Run: python -m pytest tests/test_solvation_loop_checkpoint.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

from src.Base import solvation_loop as sl
from src.Base.numerics import Numerics
from src.Base.reaction_field import ContinuumField
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.state_models import HFState, MP2State, StateResult
from src.SingleReference.ADC.eeADC.ee_state_model import ADC2State
from tests.adc_common import (AUX, SCF_TOL_TIGHT, SOLVATION_TIGHT, SOLVATION_TIGHT_LOOP,
                              WATER_C2V, Stop, df_rhf, stop_after)
from tests.pins import assert_same, assert_same_tree

#: The cycle after which the first run is stopped.
STOP_AFTER = 3


def make_scf(mol):
    return df_rhf(mol, SCF_TOL_TIGHT, 1e-9)


def build(kind, solvent='toluene', basis='6-31g'):
    """(field, state) on a fresh molecule, medium and model; kind 'hf', 'mp2',
    'adc2' (the singlet with its LR term) or 'adc2-triplet'."""
    mol = gto.M(atom=WATER_C2V, basis=basis, verbose=0)
    field = ContinuumField.from_environment(SolventScreening(mol, solvent=solvent))
    hf = HFState(mol, make_scf)
    if kind == 'hf':
        return field, hf
    mp2 = MP2State(hf, 1, AUX)
    if kind == 'mp2':
        return field, mp2
    if kind == 'adc2-triplet':
        return field, ADC2State(mp2, 'triplet', 0, 3, follow='index')
    return field, ADC2State(mp2, 'singlet', 0, 3, field.transition_environment(),
                            follow='index')


def resumed(kind, path, slow=None):
    """(uninterrupted, resumed) loops, each with its field and state."""
    field, state = build(kind)
    whole = sl.equilibrate(state, field, slow=slow, **SOLVATION_TIGHT)
    f1, s1 = build(kind)
    with pytest.raises(Stop):
        sl.equilibrate(s1, f1, slow=slow, checkpoint=path, log=stop_after(STOP_AFTER),
                       **SOLVATION_TIGHT)
    f2, s2 = build(kind)
    part = sl.equilibrate(s2, f2, slow=slow, checkpoint=path, **SOLVATION_TIGHT)
    return (field, state, whole), (f2, s2, part)


def same(a, b):
    assert len(a.history) == len(b.history) > STOP_AFTER
    assert_same_tree(a.history, b.history, label='history')
    assert_same(a.energy, b.energy, label='G')
    assert_same(a.state_energy, b.state_energy, label='E(q)')
    assert_same(a.q, b.q, unit='1', quantity='root', label='charges')
    assert_same(a.dm_eff, b.dm_eff, unit='1', quantity='vector',
                label='relaxed density')


@pytest.mark.parametrize('kind', ['hf', 'mp2', 'adc2'])
def test_a_resumed_loop_is_the_uninterrupted_one(kind, tmp_path):
    (field, state, whole), (f2, s2, part) = resumed(kind, str(tmp_path / 'c.json'))
    same(part, whole)
    assert_same(sl.force(s2, f2, part), sl.force(state, field, whole),
                quantity='gradient', label='force of G_eq')


def test_a_resumed_non_equilibrium_loop(tmp_path):
    field, s0 = build('mp2')
    slow = sl.slow_charges(field, sl.equilibrate(s0, field, **SOLVATION_TIGHT).dm_eff)
    (_, _, whole), (_, _, part) = resumed('adc2', str(tmp_path / 'neq.json'), slow)
    same(part, whole)


class NoCheckpoint:
    """A state model with no checkpoint(): the gas-phase HF energy."""

    def __init__(self, mol):
        self.mol, self.hf = mol, HFState(mol, make_scf)

    def __call__(self, h, e):
        r = self.hf(h, e)
        return StateResult(r.energy, r.dm_eff)


def test_a_model_without_checkpoint_is_refused(tmp_path):
    field, state = build('hf')
    with pytest.raises(ValueError, match='no checkpoint'):
        sl.equilibrate(NoCheckpoint(state.mol), field, checkpoint=str(tmp_path / 'x.json'),
                       **SOLVATION_TIGHT)


def stopped(kind, path, **kw):
    """A loop of `kind` (`build`) killed after one cycle, its checkpoint at path."""
    field, state = build(kind)
    with pytest.raises(Stop):
        sl.equilibrate(state, field, checkpoint=path, log=stop_after(1), **SOLVATION_TIGHT)


@pytest.mark.parametrize('axis', ['medium', 'tolerance', 'start', 'basis', 'model',
                                  'spin'])
def test_a_checkpoint_of_another_loop_is_refused(axis, tmp_path):
    """One axis changed at a time against the checkpoint's loop."""
    path = str(tmp_path / 'c.json')
    kind = 'adc2' if axis == 'spin' else 'hf'
    stopped(kind, path)
    kw = dict(SOLVATION_TIGHT)
    if axis == 'spin':
        field, state = build('adc2-triplet')
    elif axis == 'model':
        field, state = build('mp2')
    else:
        field, state = build('hf', solvent='water' if axis == 'medium' else 'toluene',
                             basis='6-31g*' if axis == 'basis' else '6-31g')
    if axis == 'tolerance':
        kw = dict(numerics=Numerics(solvent=dict(SOLVATION_TIGHT_LOOP, loop_tol_q=1e-9)))
    if axis == 'start':
        kw['q0'] = np.zeros(field.nsites)
    with pytest.raises(ValueError, match='another loop'):
        sl.equilibrate(state, field, checkpoint=path, **kw)


def test_a_checkpoint_of_slow_charges_is_not_an_equilibrium(tmp_path):
    path = str(tmp_path / 'neq.json')
    field, state = build('hf')
    slow = sl.slow_charges(field, sl.equilibrate(state, field, **SOLVATION_TIGHT).dm_eff)
    f1, s1 = build('hf')
    with pytest.raises(Stop):
        sl.equilibrate(s1, f1, slow=slow, checkpoint=path, log=stop_after(1),
                       **SOLVATION_TIGHT)
    f2, s2 = build('hf')
    with pytest.raises(ValueError, match='another loop'):
        sl.equilibrate(s2, f2, checkpoint=path, **SOLVATION_TIGHT)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
