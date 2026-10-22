"""The fast twin of tests/test_bse_force_residual_tolerance.py.

A Davidson root converged only to BSE_FORCE_RESIDUAL_TOL still gives the force
of its own energy. Formaldehyde only (ethylene stays nightly), and one
difference along a seeded direction over every coordinate
(tests/directional_difference.py) instead of three components.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (BSE_DAVIDSON_CONV_TOL, BSE_FORCE_RESIDUAL_TOL,
                                ISDF_GRADIENT_FLOOR)
from src.Base.utils.mpi_grid import distributed
from tests import directional_difference as dd
from tests.test_bse_force_residual_tolerance import chain
from tests.test_one_fit_forces_follow_their_energy import BAR, STEP, molecule


def test_the_force_at_the_accepted_residual_follows_its_energy_along_a_direction():
    warnings.simplefilter('ignore')
    mol = molecule('formaldehyde')
    d = dd.direction(mol)
    with distributed(None):
        loose = chain(mol, BSE_FORCE_RESIDUAL_TOL)
        tight = chain(mol, BSE_DAVIDSON_CONV_TOL)
        g_loose, d_loose = loose.excitation_gradient()
        g_tight, d_tight = tight.excitation_gradient()
        fd = dd.difference(loose.excitation, mol, d, STEP)
    assert d_loose['omega'] != d_tight['omega']
    miss = dd.along(g_loose, d) - fd
    moved = float(np.abs(g_loose - g_tight).max())
    print(f'\nformaldehyde: omega moved '
          f'{d_loose["omega"] - d_tight["omega"]:.1e} Ha, force moved '
          f'{moved:.1e}, FD miss {miss:.1e} Ha/Bohr')
    assert abs(miss) <= BAR, miss
    assert moved <= ISDF_GRADIENT_FLOOR, moved


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
