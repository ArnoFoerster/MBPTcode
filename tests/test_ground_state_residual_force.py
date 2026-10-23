"""The ground-state residual force is the mean field's own, grid response
included.

WHAT BREAKS IF THIS FAILS. pyscf's `mf.Gradients().kernel()` leaves out
d(Becke weights)/dR by default: on a Kohn-Sham reference that force breaks
translational invariance by about 1e-5 Ha/Bohr, and run_uisc would print that
residual beside its Huang-Rhys data. With that force the residual below
differs from `mean_field_force`'s by the grid response.
"""
import numpy as np
import pytest
from pyscf import dft, gto

from src.properties.optimize import ground_state_residual_force
from src.gradients.mean_field_surface import mean_field_force


def test_the_residual_is_the_grid_response_force():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='6-31g', verbose=0)
    mf = dft.RKS(mol, xc='b3lyp')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    force = mean_field_force(mf)
    # the grid response is what keeps the force translationally invariant
    assert np.abs(force.sum(axis=0)).max() < 1e-9
    assert ground_state_residual_force(mf) == pytest.approx(
        float(np.abs(force).max()), abs=1e-12)
