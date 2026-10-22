"""The fast twin of test_kohn_sham_reference in tests/test_derivative_coupling.py.

On a hybrid reference the Z-vector picks up the hybrid scaling and f_xc, the
kind of term that leaves the coupling the right shape and the wrong size. The
nightly gate differences the state overlap along all twelve coordinates of
formaldehyde (25 solves); this one along a seeded direction over every
coordinate (tests/directional_difference.py), three solves, with the same
building blocks `derivative_couplings` uses:

    d_01 . u = [ S_01(R0, R0 + h u) - S_01(R0, R0 - h u) ] / 2h,

against the analytic coupling projected on u, to 1e-4 of the analytic
coupling's largest component (the nightly gate's 1e-4 of the difference's).
"""
import os
import sys

import numpy as np
import pytest
from pyscf import dft, gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import NUCLEAR_FD_STEP
from src.gradients.derivative_coupling import analytic_coupling
from src.gradients.excited_state import ExcitedStateChain
from src.properties.nonadiabatic import (aligned_overlap, mo_overlap,
                                         spectrum_solve, state_overlap)
from src.Base.numerics import numerics_of
from tests import directional_difference as dd
from tests.isdf_test_grid import g1
from tests.test_derivative_coupling import CH2O


def ks(mol):
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-ri')
    mf.grids.level = 5
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-11
    mf.kernel()
    return mf


def test_kohn_sham_coupling_along_a_direction():
    mol = gto.M(atom=CH2O, basis='cc-pvdz', verbose=0)
    ch = ExcitedStateChain(mol, ks, state=0, numerics=numerics_of({'nroots': 5, 'auxbasis': 'cc-pvdz-ri', 'counts': g1(mol)}))
    ana, diags = analytic_coupling(ch, 0, 1)
    assert diags['branch_configuration'] > diags['branch_amplitude']

    solve = spectrum_solve(ch)
    mf0, _, x0, y0 = solve(mol)
    nocc = int(np.count_nonzero(np.asarray(mf0.mo_occ) > 0))
    x_ref, y_ref = x0[:, :2], y0[:, :2]
    u = dd.direction(mol)
    sides = []
    for sign in (-1.0, 1.0):
        mfd, _, xd, yd = solve(dd.displaced(mol, u, sign * NUCLEAR_FD_STEP))
        t = mo_overlap(mol, mf0.mo_coeff, mfd.mol, mfd.mo_coeff)
        s, _, weight = aligned_overlap(
            state_overlap(t, nocc, x_ref, y_ref, xd, yd), 2)
        assert weight > 0.9, weight
        sides.append(s[1, 2])
    fd = (sides[1] - sides[0]) / (2.0 * NUCLEAR_FD_STEP)
    miss = dd.along(ana, u) - fd
    print(f'\nformaldehyde/PBE0 d_01 along u: analytic {dd.along(ana, u):.6e}, '
          f'FD {fd:.6e}, miss {miss:.1e}')
    assert abs(miss) / np.abs(ana).max() < 1e-4, miss


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
