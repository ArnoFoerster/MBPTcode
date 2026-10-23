"""The fast twin of test_the_floor_grid_reproduces_the_fitted_frequencies in
tests/test_numerical_hessian.py.

The nightly gate builds formaldehyde's whole Hessian on the floor grid
(`ISDF_HESSIAN_MIN_GRID`), 24 to 48 forces, and compares its frequencies with
the density-fitted route's. What makes the coarse grids rough is the carbon,
whose interpolation frame turns fastest of the four atoms (2.8 rad/Bohr), so
this twin differences the carbon's forces alone, 6 forces on the frozen pair
layout `numerical_hessian` uses, and compares the carbon's three columns with
pyscf's density-fitted analytic Hessian on Hartree-Fock, the stricter case
(five times B3LYP's exact exchange).

The bound is the floor grid's own curvature error: 2.20e-2 Ha/Bohr^2
against a largest element of 0.782 (py310 workstation), the size
behind the nightly gate's 34.5 cm^-1 on Hartree-Fock. The 148-point grid
puts a 1421 cm^-1 mode at 2646, an error of order one in these
columns.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.numerics.constants import ISDF_HESSIAN_MIN_GRID, NUCLEAR_FD_STEP
from src.Base.factorization.isdf_jk import frozen_pair_layout
from src.properties.hessian import gradient_at, reference_pair_layout
from tests.test_numerical_hessian import FORMALDEHYDE, factory

CARBON = 0
#: max |H_isdf - H_df| over the carbon's columns, Ha/Bohr^2.
COLUMN_BOUND = 3e-2


def test_the_floor_grid_carbon_columns_follow_the_fitted_hessian():
    ch2o = gto.M(atom=FORMALDEHYDE, basis='cc-pvdz', verbose=0)
    build = factory('isdf', None, ISDF_HESSIAN_MIN_GRID)
    layout = reference_pair_layout(ch2o, build)
    columns = np.zeros((ch2o.natm, 3, 3))
    for y in range(3):
        g = {}
        for sign in (+1, -1):
            with frozen_pair_layout(layout):
                g[sign] = gradient_at(ch2o, build, CARBON, y, sign)
        columns[:, :, y] = (g[+1] - g[-1]) / (2.0 * NUCLEAR_FD_STEP)
    mf = factory('df', None)(ch2o)
    ref = np.asarray(mf.Hessian().kernel())[:, CARBON]
    miss = float(np.abs(columns - ref).max())
    print(f'\nformaldehyde/HF carbon columns, floor grid - DF analytic: '
          f'{miss:.2e} Ha/Bohr^2 (largest {np.abs(ref).max():.3f})')
    assert miss < COLUMN_BOUND, miss


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
