"""The S0 Hessian's mean field is converged to the run's orbital-gradient
threshold or HESSIAN_SCF_GRAD_TOL, whichever is tighter, never to
SCF_DIFFERENTIABLE_GRAD_TOL by default: on augmented double-zeta bases with
density-fitted J and K on XC grid 4 DIIS stalls above that, and a continuum's
re-convergence is then refused, so every Hessian in a solvent failed.

Gates, on the tests' formaldehyde declaration (`photophysics_cases`):
  * the analytic route's density-fitted twin (`Run.analytic_df_hessian`) is
    built on numerics whose conv_tol_grad is min(run's, HESSIAN_SCF_GRAD_TOL):
    a walk-converged run (1e-9, the release's setting; 1e-8, the command
    line's default) hands its twin HESSIAN_SCF_GRAD_TOL, a run converged
    tighter (1e-11) keeps its own; a run on the release's command line (density-fitted at
    1e-9), in gas and toluene, hands its twin its own numerics (deriving nothing is
    refused, which aborted the Hessian stage at the release settings);
  * the numerical route's displaced forces (`cli.hessian_numerics_of`) the same.
"""
import pytest

from src.Base.numerics.constants import (HESSIAN_SCF_GRAD_TOL,
                                         SCF_DIFFERENTIABLE_GRAD_TOL)
from tests.photophysics_cases import declaration, run_of

CASES = [('1e-9', HESSIAN_SCF_GRAD_TOL), ('1e-8', HESSIAN_SCF_GRAD_TOL),
         ('1e-11', 1e-11)]


class _Built(Exception):
    """Stops the twin once its numerics are known."""


def test_the_tolerance_is_reachable():
    assert HESSIAN_SCF_GRAD_TOL > SCF_DIFFERENTIABLE_GRAD_TOL


@pytest.mark.parametrize('run_tol, twin_tol', CASES)
def test_the_analytic_twin_converges_to_a_reachable_threshold(tmp_path, monkeypatch,
                                                               run_tol, twin_tol):
    run = run_of(tmp_path, '--scf-grad-tol', run_tol)
    seen = {}

    def factory(numerics):
        seen['scf'] = numerics.scf
        raise _Built
    monkeypatch.setattr(run, 'mean_field_factory', factory)
    with pytest.raises(_Built):
        run.analytic_df_hessian()
    assert seen['scf'].route == 'df'
    assert seen['scf'].conv_tol_grad == twin_tol


@pytest.mark.parametrize('run_tol, hess_tol', CASES)
def test_the_numerical_hessian_converges_to_a_reachable_threshold(run_tol, hess_tol):
    d = declaration('--hessian', 'numerical', '--scf-grad-tol', run_tol)
    assert d.hessian_numerics.scf.conv_tol_grad == hess_tol


#: The release's command line beyond the tests' required options: the walk
#: route's DF J/K mean field, its SCF threshold, the excited-state recipe.
RELEASE = ['--scf', 'df', '--scf-grad-tol', '1e-9', '--xc-grid', '4', '--tda',
           '--residues', 'sop', '--outside-scissor', 'curve', '--grid-accuracy', 'G3']


@pytest.mark.parametrize('environment', ['gas', 'toluene'])
def test_a_density_fitted_run_at_the_ceiling_is_its_own_twin(tmp_path, monkeypatch, environment):
    # in a continuum the pipeline recipe, scheme A, by name
    scheme = () if environment == 'gas' else ('--solvent-scheme', 'scheme-a')
    run = run_of(tmp_path, *RELEASE, '--environment', environment, *scheme)
    assert run.declaration.numerics.scf.route == 'df'
    assert run.declaration.numerics.scf.grid_level is None
    seen = {}

    def factory(numerics):
        seen['numerics'] = numerics
        raise _Built
    monkeypatch.setattr(run, 'mean_field_factory', factory)
    with pytest.raises(_Built):
        run.analytic_df_hessian()
    assert seen['numerics'] is run.declaration.numerics
