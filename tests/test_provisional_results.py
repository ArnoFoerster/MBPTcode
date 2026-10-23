"""Provisional results warn and the run carries on; mixed pictures are
refused (refuse inputs, warn numerics).

- A Franck-Condon integrand that has not decayed at its largest t_max: rho
  may be truncated, a numerical event, warned about. One undamped mode
  recurs forever.
- The X2C spin-orbit operator on a mean field that is not
  scalar-relativistic, for a molecule with an element heavier than Kr: two
  relativistic pictures in one quantity, an input that is refused.
- The adaptive set over its budget at the end of a walk warns and records
  'exceeds_tol': tests/test_adaptive_posteriori.py. An unconverged
  ground-state relaxation: the relaxation's own tests.
"""
import numpy as np
import pytest
from pyscf import gto

from src.properties.franck_condon import fc_weighted_dos
from src.properties.spin_orbit import soc_operator_mo


def test_an_undecayed_franck_condon_integrand_warns():
    with pytest.warns(RuntimeWarning, match='had not decayed'):
        rho = fc_weighted_dos(-0.05, np.array([1.0]), np.array([0.01]), 300.0)
    assert np.isfinite(rho)


def test_x2c_on_a_non_relativistic_heavy_reference_is_refused():
    mol = gto.M(atom='I 0 0 0; H 0 0 1.6', basis='def2-svp', verbose=0)

    class MeanField:
        """A non-relativistic mean field: no `with_x2c`."""

    with pytest.raises(ValueError, match='scalar-relativistic'):
        soc_operator_mo(MeanField(), mol, one_electron='x2c')
