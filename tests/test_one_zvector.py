"""The relaxed density of an ADC(2) state's call (its solvation loop's) and
of its force are one quantity: both solve the orbital response with
src.Base.pyscf.cphf, on the model's one Numerics. Water cc-pVDZ (bent,
non-symmetric), frozen 1, cc-pvdz-ri, roots 1e-12 with residual 1e-11 (the
call's root as converged as the force's polish), 3 roots, root 0; the
singlet with its LR term in toluene (IEF-PCM, zero charges) and in the gas
phase, on the dense and the tiled force route.

  1. the call and the force each reach `cphf.solve_operator`, and every one
     of those solves reads the model's Numerics object (no solve on another
     tolerance).
  2. the dipole of the call's density (the loop's) equals the dipole of the
     force Lagrangian's relaxed density to 1e-9 a.u.

Run: python tests/test_one_zvector.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.numerics import Numerics
from src.Base.pyscf import cphf
from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.gradients.adc.ee_state_model import ADC2State
from src.gradients.state_models import HFState, MP2State
from tests.adc_common import ATOMS, AUX, NF, ROOT_CONV_TOL, check, make_scf_loop, molecule

MOL = molecule(ATOMS, basis='cc-pvdz')
# the roots' residual: at the default the call's root differs from the
# force's polished one by 1e-8 a.u. in the dipole, which is not the Z-vector's
ROOT_RESIDUAL_TOL = 1e-11


class Solves:
    """Records the Numerics of every `cphf.solve_operator` call while installed."""

    def __enter__(self):
        self.seen = []
        self.fn = cphf.solve_operator

        def recorded(apply, rhs, denominators, numerics=None, flags=None):
            self.seen.append(numerics)
            return self.fn(apply, rhs, denominators, numerics, flags)
        cphf.solve_operator = recorded
        return self

    def __exit__(self, *exc):
        cphf.solve_operator = self.fn


def setup(solvent, route):
    """(model, h, e): the singlet, in toluene with its LR term or in the gas
    phase."""
    if solvent:
        field = ContinuumField.from_environment(SolventScreening(MOL, solvent='toluene'))
        h, e = field.operator(np.zeros(field.nsites))
        lr = field.transition_environment()
    else:
        h, e, lr = np.zeros((MOL.nao, MOL.nao)), 0.0, None
    mp2 = MP2State(HFState(MOL, make_scf_loop), NF, AUX, force_route=route,
                   numerics=Numerics(adc={'conv_tol': ROOT_CONV_TOL,
                                          'residual_tol': ROOT_RESIDUAL_TOL}))
    return ADC2State(mp2, 'singlet', 0, 3, lr), h, e


def dipole(dm):
    return -np.einsum('xmn,mn->x', MOL.intor('int1e_r'), np.asarray(dm, float))


def check_one_solve(solvent, route):
    where = f"{'toluene' if solvent else 'gas'}, {route}"
    model, h, e = setup(solvent, route)
    with Solves() as call:
        r = model(h, e)
    with Solves() as force:
        _, info = model.force(h, e)
    ok = check(len(call.seen) >= 1 and len(force.seen) >= 1
               and all(n is model.numerics for n in call.seen + force.seen),
               f"{where}: the call and the force solve with cphf on the model's "
               'Numerics', f'{len(call.seen)} + {len(force.seen)} solves')
    d = float(np.max(np.abs(dipole(r.dm_eff) - dipole(info['dm_eff']))))
    ok &= check(d < 1e-9, f"{where}: the loop's relaxed dipole is the force "
                "Lagrangian's", f'max |d mu| {d:.1e} a.u.')
    return ok


def run():
    ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for solvent in (True, False):
            for route in ('dense', 'tiled'):
                ok &= check_one_solve(solvent, route)
    print('ALL PASSED' if ok else 'FAILURES DETECTED')
    return ok


def test_one_zvector():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
