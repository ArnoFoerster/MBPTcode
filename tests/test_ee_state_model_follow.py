"""Root picking of the ADC(2) state model (eeADC.ee_state_model.ADC2State,
`follow`). An optimization to the adiabatic S1 / T1 minimum needs an energy
that is a function of the geometry alone; following by singles overlap is not,
since the followed vector is carried from call to call (and across rejected
optimizer steps). Water 6-31G (bent, non-symmetric), frozen 1, cc-pvdz-ri /
cc-pvdz-jkfit, gas (no external field), roots 1e-12.

  1. follow='index' with a carried vector of root 1 (as after a crossing)
     still gives root 0: energy, relaxed density and force equal a fresh
     model's to a few ulp (`matches` of tests/pins.py: two SCFs and two
     solves are never bitwise), for the singlet (with no LR) and the triplet.
  2. follow='overlap' with the same carried vector gives root 1 (the
     history matters there: the check of 1 can fail).
  3. an unknown follow is refused.

Run: python tests/test_ee_state_model_follow.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.gradients.state_models import HFState, MP2State
from src.gradients.adc.ee_state_model import ADC2State
from tests.adc_common import ATOMS, AUX, NF, ROOT_CONV_TOL, check, make_scf_loop, molecule
from tests.pins import RECORDING_BLAS, matches

MOL = molecule(ATOMS)
H0 = np.zeros((MOL.nao, MOL.nao))


def model(spin, root=0, follow='overlap'):
    mp2 = MP2State(HFState(MOL, make_scf_loop), NF, AUX, force_route='dense')
    return ADC2State(mp2, spin, root, 3, None, conv_tol=ROOT_CONV_TOL, follow=follow)


def check_follow(spin):
    fresh = model(spin, follow='index')
    r0 = fresh(H0, 0.0)
    F0, _ = fresh.force(H0, 0.0)
    upper = model(spin, root=1)
    r1 = upper(H0, 0.0)
    ok = check(r1.info['index'] == 1 and r1.energy > r0.energy,
               f'{spin}: root 1 lies above root 0',
               f"omega {r0.info['omega']:.6f} / {r1.info['omega']:.6f} Ha")

    carried = model(spin, follow='index')
    carried.y_follow = upper.y_follow.copy()
    r = carried(H0, 0.0)
    carried.y_follow = upper.y_follow.copy()
    F, _ = carried.force(H0, 0.0)
    # two models, each with its own SCF and solve: to a few ulp, not bitwise
    ok &= check(r.info['index'] == 0
                and matches(r.energy, r0.energy, RECORDING_BLAS)
                and matches(r.dm_eff, r0.dm_eff, RECORDING_BLAS, unit='1',
                            quantity='vector')
                and matches(F, F0, RECORDING_BLAS, quantity='gradient'),
                f"{spin}, follow='index', carried root-1 vector: root 0, energy, "
                'density and force to a few ulp',
                f'max |dF| {float(np.max(np.abs(F - F0))):.1e}')

    over = model(spin, follow='overlap')
    over.y_follow = upper.y_follow.copy()
    r = over(H0, 0.0)
    ok &= check(r.info['index'] == 1
                and matches(r.energy, r1.energy, RECORDING_BLAS),
                f"{spin}, follow='overlap', same carried vector: root 1 "
                '(history-dependent)', f"index {r.info['index']}")
    return ok


def check_refuse():
    try:
        model('singlet', follow='adiabatic')
    except ValueError:
        return check(True, "an unknown follow is refused")
    return check(False, "an unknown follow is refused")


def run():
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for spin in ('singlet', 'triplet'):
            all_ok &= check_follow(spin)
        all_ok &= check_refuse()
    print('ALL PASSED' if all_ok else 'FAILURES DETECTED')
    return all_ok


def test_ee_state_model_follow():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
