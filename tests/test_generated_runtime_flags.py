"""Numerical events of the generated charged-ADC solves warn and flag the
result, they do not refuse (inputs are refused, numerical events warned
about), and the
one-step on-shell scheme is not such an event. Each path is starved on purpose
(LiH/sto-3g, generated restricted ADC(3)/(4)):

  * the Davidson (max_cycle=1): RuntimeWarning, last_result['conv'] False;
  * the exact k=2 downfold's outer omega loop (max_outer=1): RuntimeWarning,
    conv False; with on_shell=True the same single step is the scheme: no
    warning, conv True, 'on_shell' recorded;
  * the diagonal downfold's omega is the one QP root finder's root of its
    pole sum (largest Z), checked against its own equation.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pyscf import gto, scf

from src.SingleReference.ADC import ADCSolver
from src.SingleReference.ADC import sigma_driver
from tests.generated_runtime import requires_runtime

pytestmark = requires_runtime(4)


@pytest.fixture(scope='module')
def lih():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-12)


def test_starved_davidson_warns_and_flags(lih):
    s = ADCSolver(lih, level='adc3', pdaggerq=True)
    with pytest.warns(RuntimeWarning, match='unconverged'):
        s.solve(tol=1e-12, max_cycle=1)
    assert not s.last_result['conv'].any()


def test_starved_outer_loop_warns_and_flags(lih):
    gen = ADCSolver(lih, level='adc4', pdaggerq=True, construction='twosector')._generated()
    with pytest.warns(RuntimeWarning, match='not self-consistent'):
        out = sigma_driver.solve_dyson_adc_downfolded(gen, gen.nocc - 1, max_outer=1, construction='twosector')
    assert out['conv'] is False and out['on_shell'] is False


def test_on_shell_is_one_step_by_definition(lih):
    s = ADCSolver(lih, level='adc4', pdaggerq=True, on_shell=True, construction='twosector')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        s.solve()
    assert s.last_result['on_shell'] is True and s.last_result['conv'] is True
    assert s.last_result['n_outer'] == 1


def test_diagonal_downfold_root_comes_from_the_one_qp_root_finder(lih):
    # the diagonal ADC(4) downfold is a pole-sum QP equation; its root is the
    # shared finder's (largest Z), so it solves omega = e3 + static + schur
    gen = ADCSolver(lih, level='adc3', pdaggerq=True)._generated()
    out = sigma_driver.solve_dyson_adc4_downfold(gen, gen.nocc - 1)
    rhs = out['e_adc3'] + out['c2'] + out['u3'] + out['schur']
    assert abs(out['e_downfold'] - rhs) < 1e-10
    assert 0.0 < out['Z'] <= 1.0 and out['conv'] is True
