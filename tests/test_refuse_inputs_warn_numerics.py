"""Refuse inputs, warn numerics.

- A ground-state relaxation that stopped before converging is a numerical
  event: it returns where it stopped, warns (`UnconvergedRelaxation`) and
  writes `relaxation_converged=False` with max |dE/dR| into
  `mf.relaxation`, on the geomeTRIC route (pyscf's driver, whose `optimize`
  returned the stop point as the minimum with no word) and on the Cartesian
  one. A converged relaxation carries the flag True.
- An ISDF grid with no row in the shipped radii table: re-optimized at run
  time with a warning that every number built on it is unvalidated. Now
  refused (`UntabulatedISDFGrid`) unless `IsdfNumerics.runtime_radii` (or the
  `runtime_radii` keyword) declares the run-time grid.

The relaxation refused or, through pyscf's driver, said nothing; the grid
warned and went on.
"""
import warnings

import pytest
from pyscf import gto, scf

import src.Base.factorization.separable_ri as sri
from src.Base.numerics.numerics import Numerics
from src.properties.optimize import (UnconvergedRelaxation,
                                     relax_ground_state)


def tight_rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    return mf.run()


STRETCHED_WATER = 'O 0 0 0.2; H 0 0.9 -0.5; H 0 -0.9 -0.5'


@pytest.mark.parametrize('engine', ['auto', 'cartesian'])
def test_an_unconverged_relaxation_warns_and_flags(engine):
    if engine == 'auto':
        pytest.importorskip('geometric')
    # a bent water stretched off its minimum, one step allowed
    mol = gto.M(atom=STRETCHED_WATER, basis='sto-3g', verbose=0)
    with pytest.warns(UnconvergedRelaxation, match='relaxation_converged'):
        opt, mf = relax_ground_state(mol, tight_rhf, engine=engine,
                                     numerics=Numerics(walk={'ground_max_cycle': 1}))
    flag = mf.relaxation
    assert flag['relaxation_converged'] is False
    assert flag['engine'] == ('geometric' if engine == 'auto'
                              else 'cartesian')
    # the stop point, not a minimum, and its residual force on record
    assert flag['opt_grad_max'] > 1e-3
    assert mf.converged and opt.natm == 3


def test_a_converged_relaxation_carries_the_flag():
    mol = gto.M(atom=STRETCHED_WATER, basis='sto-3g', verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UnconvergedRelaxation)
        _, mf = relax_ground_state(mol, tight_rhf, engine='cartesian')
    assert mf.relaxation['relaxation_converged'] is True
