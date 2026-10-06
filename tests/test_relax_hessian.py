"""A relaxation that starts from a given quasi-Newton Hessian and hands back
the one it ended with, on both engines.

The use is sequential: a triplet relaxation started at the singlet minimum
with the singlet run's final Hessian. What the handoff must do is measurable
without an excited state, on the Hartree-Fock ground state of a distorted
water: a walk restarted AT its own minimum with its own final Hessian has
nothing left to do (at most one step), and a planted wrong Hessian -- the
right one scaled up twentyfold, steps twenty times too short -- costs
measurably more steps than the right one from the same start, so a Hessian
that is silently dropped or misread cannot pass both.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.properties.optimize import MeanFieldSurface, relax  # noqa: E402

DISTORTED = 'O 0 0 0.25; H 0 0.92 -0.40; H 0 -0.70 -0.52'

#: The factor the planted Hessian is wrong by.
WRONG_SCALE = 20.0


def factory(mol):
    """Tight Hartree-Fock: an optimizer reads converged forces."""
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-12, 1e-8, 0
    return mf.run()


def water():
    """Water well away from its minimum, 6-31G."""
    return gto.M(atom=DISTORTED, basis='6-31g', verbose=0)


def walk(mol, engine, hess_init=None):
    """(minimum, info) of the Hartree-Fock relaxation from `mol`."""
    extra = {} if hess_init is None else {'hess_init': hess_init}
    if engine == 'cartesian':
        extra.update(max_cycle=60)
    return relax(MeanFieldSurface(mol, factory), mol, engine=engine,
                 verbose=False, **extra)


@pytest.mark.parametrize('engine', ['cartesian', 'geometric'])
def test_own_hessian_at_own_minimum_has_nothing_to_do(engine):
    """Restarted at its minimum with its own final Hessian: at most one step
    (two evaluations), and the Hessian handed back is (3N, 3N) symmetric."""
    if engine == 'geometric':
        pytest.importorskip('geometric')
    mol_min, info = walk(water(), engine)
    hessian = np.asarray(info['hessian'])
    n3 = 3 * mol_min.natm
    assert hessian.shape == (n3, n3)
    assert np.abs(hessian - hessian.T).max() < 1e-8
    _, again = walk(mol_min, engine, hess_init=hessian)
    assert again['cycles'] <= 2, again['cycles']


@pytest.mark.parametrize('engine', ['cartesian', 'geometric'])
def test_a_wrong_hessian_costs_steps(engine):
    """From the distorted start: the minimum's own Hessian takes fewer steps
    than the same Hessian scaled by WRONG_SCALE. (Against the engine's own
    guess it need not win: geomeTRIC's internal-coordinate guess for water
    is as good as a BFGS Hessian, 6 steps against 7.)"""
    if engine == 'geometric':
        pytest.importorskip('geometric')
    _, first = walk(water(), engine)
    hessian = np.asarray(first['hessian'])
    _, right = walk(water(), engine, hess_init=hessian)
    _, wrong = walk(water(), engine, hess_init=WRONG_SCALE * hessian)
    assert right['cycles'] < wrong['cycles'], (right['cycles'],
                                               wrong['cycles'])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
