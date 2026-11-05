"""What one call and one force of the ADC(2) state model
(gradients.adc.ee_state_model.ADC2State) compute, and what they reuse. Water
6-31G (bent, non-symmetric), frozen 1, cc-pvdz-ri / cc-pvdz-jkfit, roots 1e-12,
3 roots, root 0; the singlet with its LR term in toluene (IEF-PCM, zero
charges) and the triplet in the gas phase.

  1. a call builds the orbital-relaxed density of the root it keeps and of no
     other (one Z-vector), and that density is the one the solve of every
     root's densities (solve_ee_adc(relaxed_density=True)) gives the root, bit
     for bit (its dipole change).
  2. a force at the call's field polishes the call's root on the call's
     operator: no SCF, no fold factors, at most a third of the product columns
     of a force that solves every root afresh, and the same root: omega within
     1e-10 Ha and the force within 1e-8 Ha/Bohr of the afresh force, on the
     dense route after a call to 1e-12 and to the default EE_ADC_CONV_TOL (the
     polish then takes Newton steps), and on the tiled route after the latter.
  3. a force at another field (the charges moved) solves afresh: factors built
     once, its SCF converged again.

Run: python tests/test_adc2_state_reuse.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.constants import EE_ADC_CONV_TOL
from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.gradients.adc.ee_state_model import ADC2State
from src.Base.numerics.numerics import Numerics
from src.gradients.state_models import HFState, MP2State
from src.SingleReference.ADC.eeADC import ee_relaxed_density as erd
from src.SingleReference.ADC.eeADC import ee_ri_fold
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from tests.adc_common import ATOMS, AUX, NF, ROOT_CONV_TOL, check, make_scf_loop, molecule, ee_numerics

MOL = molecule(ATOMS)


class Counter:
    """Counts the calls of a module function while installed."""

    def __init__(self, module, name):
        self.module, self.name, self.calls = module, name, 0
        self.fn = getattr(module, name)

    def __enter__(self):
        def counted(*a, **k):
            self.calls += 1
            return self.fn(*a, **k)
        setattr(self.module, self.name, counted)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.fn)


class Columns:
    """Counts the trial vectors the fold operator is applied to while installed."""

    def __enter__(self):
        self.n = 0
        R = ee_ri_fold.RIFoldADC2
        self.saved = {name: getattr(R, name) for name in ('apply', 'fold_images')}

        def wrap(fn):
            def counted(op, U, *a, **k):
                U = np.asarray(U)
                self.n += 1 if U.ndim == 1 else U.shape[1]
                return fn(op, U, *a, **k)
            return counted
        for name, fn in self.saved.items():
            setattr(R, name, wrap(fn))
        return self

    def __exit__(self, *exc):
        for name, fn in self.saved.items():
            setattr(ee_ri_fold.RIFoldADC2, name, fn)


def setup(spin, route='dense', conv_tol=ROOT_CONV_TOL):
    """(model, h, e): the singlet in toluene with its LR term, the triplet in
    the gas phase."""
    if spin == 'singlet':
        field = ContinuumField.from_environment(SolventScreening(MOL, solvent='toluene'))
        h, e = field.operator(np.zeros(field.nsites))
        lr = field.transition_environment()
    else:
        h, e, lr = np.zeros((MOL.nao, MOL.nao)), 0.0, None
    mp2 = MP2State(HFState(MOL, make_scf_loop), NF, AUX, force_route=route,
                   numerics=Numerics(adc={'conv_tol': conv_tol}))
    return ADC2State(mp2, spin, 0, 3, lr), h, e


def check_one_density(spin):
    model, h, e = setup(spin)
    with Counter(erd, 'relaxed_density') as c:
        r = model(h, e)
    mf = model.mp2.mf
    _, _, res = solve_ee_adc(mf, MOL, level='adc2', spin=spin, nroots=3, fold=True,
                             frozen=NF, auxbasis=AUX, environment=model.lr_environment,
                             numerics=ee_numerics(conv_tol=ROOT_CONV_TOL, tile_gb=model.tile_gb), return_fold=True,
                             relaxed_density=True)
    C = np.asarray(mf.mo_coeff)
    k = r.info['index']
    d_all = C @ np.asarray(res.delta_density_relaxed[k]) @ C.T
    mu = -np.einsum('xmn,mn->x', MOL.intor('int1e_r'), d_all)
    return check(c.calls == 1 and np.array_equal(r.info['delta_dipole'], mu),
                 f'{spin}: a call builds the relaxed density of its root alone, the '
                 "all-roots solve's to the bit",
                 f'{c.calls} relaxed density(ies), |d mu| '
                 f"{float(np.max(np.abs(r.info['delta_dipole'] - mu))):.1e}")


def check_polish(spin, route, conv_tol):
    model, h, e = setup(spin, route, conv_tol)
    model(h, e)
    mf = model.mp2.mf
    with Counter(ee_ri_fold, 'bare_factors') as fac, Counter(mf, 'kernel') as scf, \
            Columns() as cols:
        F, info = model.force(h, e)
    n_polish = cols.n
    model.solved = None                     # the afresh force
    with Columns() as cols:
        F0, info0 = model.force(h, e)
    dF = float(np.max(np.abs(F - F0)))
    dw = abs(info['omega'] - info0['omega'])
    return check(fac.calls == 0 and scf.calls == 0 and 3 * n_polish <= cols.n
                 and dw < 1e-10 and dF < 1e-8,
                 f"{spin}, {route}, call to {conv_tol:.0e}: the force polishes the call's root: no SCF, no "
                 'factors, a third of the columns or fewer, the afresh force',
                 f'{n_polish} vs {cols.n} columns, |d omega| {dw:.1e} Ha, '
                 f'max |dF| {dF:.1e} Ha/Bohr')


def check_moved(spin):
    model, h, e = setup(spin)
    model(h, e)
    h2 = h + 1e-4 * MOL.intor('int1e_r')[2]
    with Counter(ee_ri_fold, 'bare_factors') as fac, \
            Counter(model.mp2.mf, 'kernel') as scf:
        model.force(h2, e)
    return check(fac.calls == 1 and scf.calls == 1,
                 f'{spin}: a force at another field solves afresh',
                 f'{fac.calls} factor build(s), {scf.calls} SCF(s)')


def run():
    ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for spin in ('singlet', 'triplet'):
            ok &= check_one_density(spin)
            for route, tol in (('dense', ROOT_CONV_TOL), ('dense', EE_ADC_CONV_TOL),
                               ('tiled', EE_ADC_CONV_TOL)):
                ok &= check_polish(spin, route, tol)
            ok &= check_moved(spin)
    print('ALL PASSED' if ok else 'FAILURES DETECTED')
    return ok


def test_adc2_state_reuse():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
