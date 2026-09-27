"""The force of an ISDF mean field relaxed inside a PCM continuum.

The ISDF force is assembled by this code (`isdf_mean_field_gradient`), not by
pyscf's gradient class, so pyscf's PCM mixin never adds the reaction field's
fixed-density term. Without it the O z-force of water/cc-pVDZ in water is off
by 1.0e-2 Ha/Bohr against a finite difference of the reported energy.

  1. `mean_field_skeleton_force` -- the dispatch every correlated chain and
     `MeanFieldSurface` use -- matches the central difference of E_tot.
  2. `SolventScreening.mean_field` wrapping an ISDF factory re-binds the ISDF
     gradient to the wrapped object, so `mf.Gradients()` (what geometry
     optimizers call) gives the same force.
  3. A gradient still bound to the unwrapped, never-converged object is
     refused with a message instead of a TypeError from inside pyscf.

Run as a script (`python tests/test_isdf_pcm_force.py`) or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf, solvent

from src.Base.isdf_jk import isdf_jk, mean_field_skeleton_force
from src.Base.solvent_screening import SolventScreening

WATER = [['O', (0.0, 0.0, 0.117)], ['H', (0.0, 0.757, -0.469)],
         ['H', (0.0, -0.757, -0.469)]]
EPS_STATIC = 78.355
STEP_ANGSTROM = 1e-3
BOHR = 0.52917721092
#: the central difference's own step error at 1e-3 A is ~2e-7 Ha/Bohr here
FORCE_TOL = 1e-6


def _mol(atoms):
    return gto.M(atom=atoms, basis='cc-pvdz', unit='Angstrom', verbose=0)


def _isdf_pcm(atoms):
    mf = solvent.PCM(isdf_jk(scf.RHF(_mol(atoms)), auxbasis='cc-pvdz-ri'))
    mf.with_solvent.eps = EPS_STATIC
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def _displaced(atom, axis, sign):
    out = [[el, list(xyz)] for el, xyz in WATER]
    out[atom][1][axis] += sign * STEP_ANGSTROM
    return out


def _central_difference():
    return ((_isdf_pcm(_displaced(0, 2, +1)).e_tot
             - _isdf_pcm(_displaced(0, 2, -1)).e_tot)
            / (2 * STEP_ANGSTROM / BOHR))


@pytest.fixture(scope='module')
def reference():
    return _isdf_pcm(WATER), _central_difference()


def test_skeleton_force_carries_the_reaction_field(reference):
    mf, fd = reference
    g = mean_field_skeleton_force(mf)
    assert abs(g[0, 2] - fd) < FORCE_TOL, (g[0, 2], fd)


def test_mean_field_wrapping_rebinds_the_gradient(reference):
    _, fd = reference
    mol = _mol(WATER)
    env = SolventScreening(mol, eps=1.78, eps_static=EPS_STATIC, method='C-PCM')
    mf = env.mean_field(mol, lambda m: isdf_jk(scf.RHF(m),
                                               auxbasis='cc-pvdz-ri').run())
    g = mf.Gradients().kernel()
    assert np.all(np.isfinite(g))
    # a different cavity discretization than the pyscf-default reference, so
    # compare the wrapped gradient with its own skeleton force instead
    assert np.abs(g - mean_field_skeleton_force(mf)).max() < 1e-10


def test_stale_binding_is_refused(reference):
    mf, _ = reference
    with pytest.raises(RuntimeError, match='never converged'):
        mf.Gradients().kernel()


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
