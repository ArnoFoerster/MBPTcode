"""The memory budget of the solvated photophysics driver
(photophysics.solvated.memory_budget) and where it reaches. Water 6-31G, frozen
1, cc-pvdz-ri / -jkfit.

  1. --max-memory is the budget as given.
  2. inside SLURM, without --max-memory: this rank's share of the allocation
     (SLURM_MEM_PER_NODE over the tasks per node, ALLOCATION_MEMORY_FRACTION).
  3. with no option, allocation or PYSCF_MAX_MEMORY: pyscf's default, warned
     about and flagged 'memory_budget_unknown'.
  4. the budget set on the molecule reaches the tiled forces' memory check of
     the MP2 and the ADC(2) state models the driver builds (_factories), in the
     call and in the force.

Run: python -m pytest -q tests/test_solvated_memory_budget.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, lib

from src.Base.numerics.constants import ALLOCATION_MEMORY_FRACTION
from src.gradients import df_correlation_force_tiled as tcf
from src.gradients import state_models
from src.gradients.adc import ee_state_model
from src.environments.reaction_field import NoMedium
from src.photophysics import solvated
from tests.adc_common import ATOMS

SLURM = ('SLURM_JOB_ID', 'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_CPUS_PER_TASK',
         'SLURM_NTASKS_PER_NODE', 'SLURM_TASKS_PER_NODE', 'PYSCF_MAX_MEMORY')


@pytest.fixture
def clean_env(monkeypatch):
    for name in SLURM:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_asked_budget(clean_env):
    flags = []
    assert solvated.memory_budget(1234, flags) == 1234 and flags == []


def test_allocation_share(clean_env):
    clean_env.setenv('SLURM_JOB_ID', '1')
    clean_env.setenv('SLURM_MEM_PER_NODE', '240000')
    clean_env.setenv('SLURM_NTASKS_PER_NODE', '4')
    flags = []
    got = solvated.memory_budget(None, flags)
    assert got == int(60000 * ALLOCATION_MEMORY_FRACTION) and flags == []


def test_unknown_budget_is_flagged(clean_env):
    flags = []
    with pytest.warns(RuntimeWarning, match='PYSCF_MAX_MEMORY'):
        got = solvated.memory_budget(None, flags)
    assert got == lib.param.MAX_MEMORY
    assert [f['flag'] for f in flags] == ['memory_budget_unknown']


def test_budget_reaches_the_state_models(monkeypatch):
    seen = []
    refuse = tcf.refuse_over

    def recorded(est, max_memory_mb):
        seen.append(max_memory_mb)
        return refuse(est, max_memory_mb)
    monkeypatch.setattr(tcf, 'refuse_over', recorded)
    for module in (state_models, ee_state_model):
        monkeypatch.setattr(module, 'tiled_force_route', lambda route, mol: True)
    atoms = [[a, *xyz] for a, xyz in ATOMS]
    mol = solvated._mol_of(atoms, '6-31g', 1234)
    args = argparse.Namespace(method='adc2', jk_auxbasis='cc-pvdz-jkfit', xc=None,
                              frozen=1, auxbasis='cc-pvdz-ri', nroots=3, follow='index')
    args.tol_q, args.tol_energy = 1e-9, 1e-11
    _, make_state = solvated._factories(args, 'gas', {'numerics': solvated.adc2_numerics(args)})
    h = np.zeros((mol.nao, mol.nao))
    for kind in ('S0', 'singlet'):
        state = make_state(kind)(mol, NoMedium(mol))
        state(h, 0.0)
        state.force(h, 0.0)
    assert seen and set(seen) == {1234}, seen
