"""Forces of equilibrium-solvated states (src.environments.solvation_loop.force): the
state's force in the fixed charges + the medium's force at the relaxed density
+ the medium's part of the LR term. Water 6-31G (bent, non-symmetric), toluene
IEF-PCM, frozen 1, cc-pvdz-ri / cc-pvdz-jkfit.

  1. Hartree-Fock: the loop's force is pyscf's PCM-SCF gradient (the
     equilibrium HF state IS pyscf's PCM-SCF), to 1e-8 Ha/Bohr.
  2. G_eq of MP2 (S0), the ADC(2) singlet (LR term included; two directions)
     and triplet: the analytic force along random unit directions of the
     nuclear coordinates against central differences of the converged G_eq
     (h = 5e-4 Bohr, every displaced point a full loop on its own cavity), to
     1e-6 Ha/Bohr.

One pytest test per check (per state kind for 2); every loop is asserted
converged.

Run: python -m pytest -s tests/test_solvation_force.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.environments import solvation_loop as sl
from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.gradients.state_models import HFState, MP2State
from src.gradients.adc.ee_state_model import ADC2State
from tests.adc_common import (ATOMS, AUX, FD_STEP_SOLVATION, NF, ROOT_CONV_TOL,
                              SOLVATION_TIGHT, check, make_scf_tight, molecule)


def setup(mol, kind):
    env = SolventScreening(mol, solvent='toluene')
    field = ContinuumField.from_environment(env)
    hf = HFState(mol, make_scf_tight)
    if kind == 'hf':
        return field, hf
    mp2 = MP2State(hf, NF, AUX)
    if kind == 'mp2':
        return field, mp2
    lr = field.transition_environment() if kind == 'singlet' else None
    return field, ADC2State(mp2, kind, 0, 3, lr, conv_tol=ROOT_CONV_TOL)


def G(mol, kind):
    field, state = setup(mol, kind)
    r = sl.equilibrate(state, field, **SOLVATION_TIGHT)
    assert r.converged
    return r.energy


def test_hf_force_is_pyscf_pcm_gradient():
    mol = molecule(ATOMS)
    field, state = setup(mol, 'hf')
    r = sl.equilibrate(state, field, **SOLVATION_TIGHT)
    assert r.converged
    F = sl.force(state, field, r)
    env = SolventScreening(mol, solvent='toluene')
    mf = env.mean_field(mol, lambda m: make_scf_tight(m).run())
    g = mf.nuc_grad_method()
    g.verbose = 0
    ref = g.kernel()
    d = float(np.max(np.abs(F - ref)))
    assert check(d < 1e-8, "Hartree-Fock: the loop's force is pyscf's PCM-SCF "
                 'gradient', f'max diff {d:.1e} Ha/Bohr, max |F| '
                 f'{float(np.max(np.abs(ref))):.2e}')


DIRECTIONS = (('mp2', 1), ('singlet', 2), ('triplet', 1))


@pytest.mark.parametrize('kind', [k for k, _ in DIRECTIONS])
def test_directional(kind):
    rng = np.random.default_rng(17)
    mol = molecule(ATOMS)
    draws = {}
    for k, ndir in DIRECTIONS:          # one stream of directions, in this order
        draws[k] = [rng.standard_normal((mol.natm, 3)) for _ in range(ndir)]
    ok = True
    field, state = setup(mol, kind)
    r = sl.equilibrate(state, field, **SOLVATION_TIGHT)
    assert r.converged
    F = sl.force(state, field, r)
    for d in draws[kind]:
        d /= np.linalg.norm(d)
        fd = (G(molecule(ATOMS, FD_STEP_SOLVATION * d), kind) - G(molecule(ATOMS, -FD_STEP_SOLVATION * d), kind)) / (2 * FD_STEP_SOLVATION)
        an = float(np.sum(F * d))
        ok &= check(abs(fd - an) < 1e-6, f'{kind}: dG_eq along a random '
                    'direction = finite differences',
                    f'{an:+.8f} vs {fd:+.8f}, {fd - an:+.1e} Ha/Bohr')
    assert ok
