"""The gas-phase limit of state-specific solvation (src.environments.solvation_loop with
src.environments.reaction_field.ContinuumField): a continuum with eps_static = eps_inf
= 1 holds no charges, so every quantity must be the gas phase's (NoMedium, the
same loop with no sites) to a few ulp (`matches` of tests/pins.py: the two
loops stand on two sets of SCFs, and two SCF runs are never bitwise), and a
continuum at 1 + delta must leave it smoothly, linearly in delta. Water 6-31G
(bent, non-symmetric), frozen 1, cc-pvdz-ri / cc-pvdz-jkfit; MP2 (S0), ADC(2)
S1 (with its LR term) and T1.

  1. at eps = 1 both response matrices are exactly zero.
  2. G_eq, E(q), the relaxed density and the number of loop cycles at eps = 1
     equal NoMedium's (the cycle count exactly, the rest to a few ulp), for
     S0, S1 and T1.
  3. the force of G_eq (solvation_loop.force: the state's force, the
     medium's at the relaxed density, the LR term's) equals NoMedium's to a
     few ulp, for S0, S1 and T1.
  4. absorption and emission (vertical_energies) of S1 equal NoMedium's to
     a few ulp.
  5. continuity: at eps_s = 1 + delta (eps_inf = 1 + delta / 2) the solvent
     shift of G_eq (S0, S1) and of the S1 absorption scales with delta: the
     ratio of the shifts at delta = 1e-2 and 1e-3 lies within 10% of 10, and
     the shift at 1e-3 is nonzero (the check can fail).

Run: python tests/test_solvation_gas_limit.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.environments import solvation_loop as sl
from src.environments.reaction_field import ContinuumField, NoMedium
from src.environments.solvent_screening import SolventScreening
from src.gradients.state_models import HFState, MP2State
from src.gradients.adc.ee_state_model import ADC2State
from tests.adc_common import (ATOMS, AUX, NF, ROOT_CONV_TOL, SOLVATION_TIGHT,
                              check, make_scf_loop, molecule)
from tests.pins import RECORDING_BLAS, matches


MOL = molecule(ATOMS)


def continuum(eps_s, eps_inf):
    env = SolventScreening(MOL, eps=eps_inf, eps_static=eps_s, allow_static_eps=True)
    return ContinuumField.from_environment(env)


def state(kind, field):
    mp2 = MP2State(HFState(MOL, make_scf_loop), NF, AUX, force_route='dense')
    if kind == 'S0':
        return mp2
    lr = field.transition_environment() if kind == 'S1' else None
    return ADC2State(mp2, 'singlet' if kind == 'S1' else 'triplet', 0, 3, lr,
                     conv_tol=ROOT_CONV_TOL)


def same(a, b, quantity, unit='Ha'):
    """`a` and `b` are one `quantity`: two loops on two sets of SCFs agree to
    a few ulp (`matches`), never bitwise."""
    return matches(a, b, RECORDING_BLAS, unit, quantity=quantity)


def check_zero_response():
    f = continuum(1.0, 1.0)
    m = max(float(np.max(np.abs(f.response_matrix(r)))) for r in ('static', 'fast'))
    return check(m == 0.0, 'eps = 1: both response matrices are exactly zero',
                 f'max |Q| {m}')


def check_gas_phase():
    ok = True
    for kind in ('S0', 'S1', 'T1'):
        gas, cont = NoMedium(MOL), continuum(1.0, 1.0)
        s_gas, s_cont = state(kind, gas), state(kind, cont)
        r_gas = sl.equilibrate(s_gas, gas, **SOLVATION_TIGHT)
        r_cont = sl.equilibrate(s_cont, cont, **SOLVATION_TIGHT)
        ok &= check(same(r_cont.energy, r_gas.energy, 'energy')
                    and same(r_cont.state_energy, r_gas.state_energy, 'energy')
                    and same(r_cont.dm_eff, r_gas.dm_eff, 'vector', '1')
                    and len(r_cont.history) == len(r_gas.history),
                    f'{kind}: G_eq, E(q), relaxed density and cycles at eps = 1 '
                    'are the gas phase',
                    f'G {r_cont.energy!r} vs {r_gas.energy!r}, '
                    f'{len(r_cont.history)} vs {len(r_gas.history)} cycles')
        F_gas = sl.force(s_gas, gas, r_gas)
        F_cont = sl.force(s_cont, cont, r_cont)
        ok &= check(same(F_cont, F_gas, 'gradient'), f'{kind}: the force at '
                    'eps = 1 is the gas phase',
                    f'max |dF| {float(np.max(np.abs(F_cont - F_gas))):.1e}')
    return ok


def check_vertical():
    gas, cont = NoMedium(MOL), continuum(1.0, 1.0)
    r_gas = sl.vertical_energies(lambda: state('S0', gas), lambda: state('S1', gas),
                                 gas, **SOLVATION_TIGHT)
    r_cont = sl.vertical_energies(lambda: state('S0', cont), lambda: state('S1', cont),
                                  cont, **SOLVATION_TIGHT)
    return check(same(r_cont['absorption'], r_gas['absorption'], 'energy')
                 and same(r_cont['emission'], r_gas['emission'], 'energy')
                 and r_cont['G'].keys() == r_gas['G'].keys()
                 and all(same(r_cont['G'][k], r_gas['G'][k], 'energy')
                         for k in r_gas['G']),
                 'S1 absorption and emission at eps = 1 are the gas phase',
                 f"absorption {r_cont['absorption']!r} vs {r_gas['absorption']!r}")


def check_continuity():
    ok = True
    gas = NoMedium(MOL)
    G_gas = {k: sl.equilibrate(state(k, gas), gas, **SOLVATION_TIGHT).energy for k in ('S0', 'S1')}
    abs_gas = sl.vertical_energies(lambda: state('S0', gas), lambda: state('S1', gas),
                                   gas, **SOLVATION_TIGHT)['absorption']
    shift = {}
    for delta in (1e-2, 1e-3):
        f = continuum(1.0 + delta, 1.0 + 0.5 * delta)
        s = {k: sl.equilibrate(state(k, f), f, **SOLVATION_TIGHT).energy - G_gas[k]
             for k in ('S0', 'S1')}
        s['absorption'] = sl.vertical_energies(
            lambda: state('S0', f), lambda: state('S1', f), f, **SOLVATION_TIGHT)['absorption'] - abs_gas
        shift[delta] = s
    for k in ('S0', 'S1', 'absorption'):
        a, b = shift[1e-2][k], shift[1e-3][k]
        ratio = a / b if b else float('inf')
        ok &= check(b != 0.0 and abs(ratio - 10.0) < 1.0,
                    f'{k}: the solvent shift is linear in eps - 1',
                    f'{a:+.3e} at 1e-2, {b:+.3e} at 1e-3, ratio {ratio:.3f}')
    return ok


def run():
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for c in (check_zero_response, check_gas_phase, check_vertical, check_continuity):
            all_ok &= c()
    print('ALL PASSED' if all_ok else 'FAILURES DETECTED')
    return all_ok


def test_solvation_gas_limit():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
