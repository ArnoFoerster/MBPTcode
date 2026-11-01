"""A walk's solvation loop tied to its gradient
(`src.gradients.solvated_surface`, `SolventNumerics.loop_force_fraction`).
Water 6-31G (bent, non-symmetric), toluene IEF-PCM, the ADC(2) singlet with
its linear-response term (frozen 1, cc-pvdz-ri / cc-pvdz-jkfit), loops to
1e-10 e / 1e-12 Ha at their tolerances.

  1. `ReactionField.charge_force` is the derivative of q . V_tot(dm) at fixed
     charges and density (central differences, sites moving with the atoms).
  2. the first gradient of a surface runs its loop to the tolerances and
     measures kappa >= 1 (the singlet's own response makes the direct
     estimate several times too small); the next gradient's loop stops early
     with kappa times its estimate at most 1/3 of the gradient, and that
     force differs from the fully converged one by at most 1/3 of the
     gradient; below the walk's convergence bar the loop runs to its
     tolerances.
  3. the ground state's walk (Hartree-Fock) with the tied loop and with the
     fixed loop reach the same minimum within the walk's step tolerance, the
     tied one in fewer loop cycles, and every gradient's loop is logged (the
     water singlet has no minimum to walk to).
  4. refusals: a fraction outside (0, 1), and a tied surface without the
     walk's bar.

Run: python -m pytest -s tests/test_solvation_loop_tied.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import GEOM_OPT_CONV, LOOP_FORCE_FRACTION
from src.Base.numerics.numerics import Numerics
from src.environments import solvation_loop as sl
from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.gradients.adc.ee_state_model import ADC2State
from src.gradients.solvated_surface import SolvatedSurface
from src.gradients.state_models import HFState, MP2State
from src.properties.optimize import optimize
from tests.adc_common import (ATOMS, AUX, NF, ROOT_CONV_TOL, SOLVATION_TIGHT_LOOP,
                              check, make_scf_tight, molecule)

FIXED = Numerics(solvent=SOLVATION_TIGHT_LOOP, adc={'conv_tol': ROOT_CONV_TOL})
TIED = Numerics(solvent=dict(SOLVATION_TIGHT_LOOP,
                             loop_force_fraction=LOOP_FORCE_FRACTION),
                adc={'conv_tol': ROOT_CONV_TOL})


def make_field(mol):
    return ContinuumField.from_environment(SolventScreening(mol, solvent='toluene'))


def make_singlet(numerics):
    """The singlet's model on the surface's one Numerics."""
    def build(mol, field):
        mp2 = MP2State(HFState(mol, make_scf_tight), NF, AUX, numerics=numerics)
        return ADC2State(mp2, 'singlet', 0, 3, field.transition_environment())
    return build


def surface(numerics, final_grad_max):
    return SolvatedSurface(molecule(ATOMS), make_field, make_singlet(numerics),
                           label='S1 toluene', final_grad_max=final_grad_max,
                           numerics=numerics)


def test_charge_force_is_the_derivative_of_the_charges_interaction():
    mol = molecule(ATOMS)
    dm = make_scf_tight(mol).run().make_rdm1()
    field = make_field(mol)
    q = np.random.default_rng(1).standard_normal(field.nsites) * 1e-3
    F = field.charge_force(dm, q)
    u = np.random.default_rng(2).standard_normal(F.shape)
    u /= np.linalg.norm(u)
    h = 1e-4

    def value(step):
        other = make_field(molecule_at(mol, mol.atom_coords() + step * u))
        assert other.nsites == field.nsites
        return float(q @ other.potential(dm))
    fd = (value(h) - value(-h)) / (2 * h)
    d = abs(float((F * u).sum()) - fd)
    assert check(d < 1e-9, 'charge_force is d/dR [q . V_tot(dm)]',
                 f'|analytic - fd| {d:.1e}')


def test_a_tied_loop_keeps_its_force_within_the_bar():
    mol = molecule(ATOMS)
    tied = surface(TIED, final_grad_max=1e-6)
    F0 = tied.total_gradient(mol)[0]
    first = tied.loop_log[-1]
    assert first['mode'] == 'calibration' and first['kappa'] >= 1.0
    print(f'kappa {first["kappa"]:.2f}, first loop {first["cycles"]} cycles, '
          f'|F|max {np.abs(F0).max():.2e}')
    moved = molecule_at(mol, mol.atom_coords()
                        + 0.03 * np.random.default_rng(3).standard_normal(F0.shape))
    F = tied.total_gradient(moved)[0]
    entry = tied.loop_log[-1]
    full = surface(FIXED, final_grad_max=None)
    F_full = full.total_gradient(moved)[0]
    g = float(np.max(np.abs(F)))
    miss = float(np.max(np.abs(F - F_full)))
    print(f'tied: {entry}, |F|max {g:.2e}, miss vs the converged loop {miss:.2e}, '
          f'full loop {len(full.last["result"].history)} cycles')
    assert entry['mode'] == 'tied'
    assert entry['estimate'] <= LOOP_FORCE_FRACTION * g
    assert entry['cycles'] < len(full.last['result'].history)
    assert check(miss <= LOOP_FORCE_FRACTION * g, 'the tied force is within '
                 '1/3 of the gradient of the converged one',
                 f'{miss:.1e} vs {LOOP_FORCE_FRACTION * g:.1e} Ha/Bohr')


def test_below_the_walk_bar_the_loop_runs_to_its_tolerances():
    mol = molecule(ATOMS)
    tied = surface(TIED, final_grad_max=10.0)
    tied.total_gradient(mol)
    moved = molecule_at(mol, mol.atom_coords() + 0.01)
    tied.total_gradient(moved)
    assert [e['mode'] for e in tied.loop_log] == ['calibration', 'full']
    assert not tied.last['result'].stopped_early


def test_an_untied_twin_runs_every_loop_to_its_tolerances():
    mol = molecule(ATOMS)
    tied = surface(TIED, final_grad_max=1e-6)
    tied.total_gradient(mol)
    q, logged = tied._q, len(tied.loop_log)
    twin = tied.untied()
    assert twin.loop_log is tied.loop_log and twin.flags is tied.flags
    moved = molecule_at(mol, mol.atom_coords() + 1e-3)
    F = twin.total_gradient(moved)[0]
    F_full = surface(FIXED, final_grad_max=None).total_gradient(moved)[0]
    assert not twin.last['result'].stopped_early
    assert len(tied.loop_log) == logged and tied._q is q
    miss = float(np.max(np.abs(F - F_full)))
    assert check(miss < 1e-8, "the untied twin's displaced force is the "
                 "fully converged loop's", f'{miss:.1e} Ha/Bohr')


def test_a_tied_walk_reaches_the_fixed_walks_minimum():
    mol = molecule(ATOMS)
    walks = {}
    for name, numerics in (('fixed', FIXED), ('tied', TIED)):
        s = SolvatedSurface(mol, make_field,
                            lambda m, f: HFState(m, make_scf_tight),
                            label='S0 toluene', numerics=numerics,
                            final_grad_max=GEOM_OPT_CONV['opt_grad_max'])
        cycles = []
        solve = s._solve

        def counted(m, calibrate=False, solve=solve, cycles=cycles):
            out = solve(m, calibrate)
            cycles.append(len(out[2].history))
            return out
        s._solve = counted
        mol_opt, info = optimize(s, mol, nudge=0, verbose=False)
        assert info['converged']
        walks[name] = (mol_opt.atom_coords(), info['energy'], sum(cycles), s)
        print(f'{name}: {info["cycles"]} steps, {sum(cycles)} loop cycles, '
              f'G {info["energy"]:.10f}')
    dx = float(np.max(np.abs(walks['tied'][0] - walks['fixed'][0])))
    dG = abs(walks['tied'][1] - walks['fixed'][1])
    modes = [e['mode'] for e in walks['tied'][3].loop_log]
    print(f'minima differ by {dx:.1e} Bohr, {dG:.1e} Ha; tied modes {modes}')
    assert 'tied' in modes and modes[0] == 'calibration'
    assert check(dx < GEOM_OPT_CONV['step_max'], 'one minimum within the walk '
                 'tolerance', f'max |dR| {dx:.1e} Bohr, |dG| {dG:.1e} Ha')
    assert walks['tied'][2] < walks['fixed'][2]


def test_a_fraction_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match='loop_force_fraction'):
        Numerics(solvent={'loop_force_fraction': 1.5})


def test_a_tied_surface_needs_the_walk_bar():
    s = surface(TIED, final_grad_max=None)
    with pytest.raises(ValueError, match='final_grad_max'):
        s.total_gradient(molecule(ATOMS))


def test_an_early_stop_keeps_no_checkpoint(tmp_path):
    mol = molecule(ATOMS)
    field = make_field(mol)
    with pytest.raises(ValueError, match='checkpoint'):
        sl.equilibrate(HFState(mol, make_scf_tight), field,
                       stop=lambda *a: True, checkpoint=str(tmp_path / 'c'))
