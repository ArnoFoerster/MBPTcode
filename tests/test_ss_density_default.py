"""The state-specific density default is one constant, 'relaxed', for the
EE-ADC solve and the classical state-specific energy alike.

  1. solvent_vocabulary.SS_DENSITY_DEFAULT is 'relaxed', one of SS_DENSITIES,
     and is the default of solve_ee_adc, the folded driver, the vertical run's
     --ss-density, ClassicalStateSpecific and environment.solvent_scheme.
  2. state_specific='off' (the default) builds no relaxed density, on the
     default density; a density other than the default there is refused.
  3. a solvated state-specific EE-ADC run on defaults is the run with
     ss_density='relaxed' written out (formaldehyde cc-pVDZ, toluene, two
     solves: to a few ulp, `matches` of tests/pins.py).
"""
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from pyscf import gto, scf

from src.environments import environment, solvent_vocabulary as vocab
from src.environments.solvent_screening import SolventScreening
from src.environments.split_equilibrium import ClassicalStateSpecific
from src.SingleReference.ADC.eeADC import ee_driver, vertical
from src.SingleReference.ADC.eeADC import ee_relaxed_density as erd
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from tests.adc_common import H2CO, JKFIT, ee_numerics
from tests.pins import RECORDING_BLAS, matches


def _default(fn, name='ss_density'):
    return inspect.signature(fn).parameters[name].default


def test_one_default_for_every_entry_point():
    assert vocab.SS_DENSITY_DEFAULT == 'relaxed'
    assert vocab.SS_DENSITY_DEFAULT in vocab.SS_DENSITIES
    for fn in (solve_ee_adc, ee_driver.solve_folded_ri, ee_driver._state_specific,
               ClassicalStateSpecific.__init__, environment.solvent_scheme):
        assert _default(fn) is vocab.SS_DENSITY_DEFAULT, fn
    # the vertical run's parser is built inside main()
    assert "'--ss-density', default=SS_DENSITY_DEFAULT" in inspect.getsource(vertical.main)


@pytest.fixture(scope='module')
def solvated():
    mol = gto.M(atom=H2CO, basis='cc-pvdz', verbose=0)
    env = SolventScreening(mol, solvent='toluene')

    def factory(m):
        mf = scf.RHF(m).density_fit(auxbasis=JKFIT)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        return mf.run()

    return env, env.mean_field(mol, factory)


KW = dict(level='adc2', nroots=1, frozen=2, numerics=ee_numerics(conv_tol=1e-9), fold=True,
          spin='singlet')


def test_off_builds_no_relaxed_density(solvated, monkeypatch):
    env, sol = solvated

    def refuse(*a, **k):
        raise AssertionError('a relaxed density was built')

    monkeypatch.setattr(erd, 'relaxed_density', refuse)
    monkeypatch.setattr(ee_driver, 'relaxed_densities', refuse)
    e, _, res = solve_ee_adc(sol, environment=env, return_fold=True, **KW)
    assert res.omega.shape == (1,) and getattr(res, 'delta_density_relaxed', None) is None
    solve_ee_adc(sol, environment=env, state_specific='off', **KW)
    for dens in ('unrelaxed', 'relaxed+pcm'):
        with pytest.raises(ValueError, match='needs state_specific'):
            solve_ee_adc(sol, environment=env, ss_density=dens, **KW)


def test_default_run_is_the_relaxed_run(solvated):
    env, sol = solvated
    kw = dict(KW, environment=env, return_fold=True, state_specific='vertical')
    e_d, _, res_d = solve_ee_adc(sol, **kw)
    e_r, _, res_r = solve_ee_adc(sol, ss_density='relaxed', **kw)
    assert res_d.ss_density == 'relaxed'
    assert matches(e_d, e_r, RECORDING_BLAS, quantity='root')
    assert matches(res_d.ss_shifts['vertical'], res_r.ss_shifts['vertical'],
                   RECORDING_BLAS, quantity='energy')
    assert res_d.delta_density_relaxed is not None
