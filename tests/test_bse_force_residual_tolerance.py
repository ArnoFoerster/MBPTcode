"""A Davidson root converged only to BSE_FORCE_RESIDUAL_TOL still gives the
force of its own energy: the residual a force accepts a read root at, where
its Davidson stopped above BSE_DAVIDSON_CONV_TOL.

Gated on formaldehyde and ethylene/cc-pVDZ with the seeded C1 distortion of
tests/test_one_fit_forces_follow_their_energy.py, the SOP route of the
production flag set (frontier states, outside scissor, grid BSE adjoint) on its
density-fitted Hartree-Fock reference, 5 roots, the default preconditioner:
  * the solve at BSE_FORCE_RESIDUAL_TOL stops at other vectors than the one
    at BSE_DAVIDSON_CONV_TOL (their S1 energies differ), so the gate can fail;
  * its S1 force lies within ISDF_GRADIENT_FLOOR of the force at
    BSE_DAVIDSON_CONV_TOL, and within the SOP route's bar of a 4-point
    central difference of the same chain's energy at three components:
    ISDF_GRADIENT_FLOOR at h = 1e-3 on formaldehyde, 1e-7 at h = 4e-3 on
    ethylene, whose energy carries the pole-basis rounding noise
    tests/test_one_fit_forces_follow_their_energy.py gates it at.
Water is not gated: its 95 pairs fill the trial space at every tolerance
from 1e-4 down, so both solves return the same vectors.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (BSE_DAVIDSON_CONV_TOL, BSE_FORCE_RESIDUAL_TOL,
                                ISDF_GRADIENT_FLOOR)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.utils.mpi_grid import distributed
from src.properties.excitations import SurfaceSpec, surface_of
from tests.test_one_fit_forces_follow_their_energy import (BAR, COMPONENTS,
                                                           SOP_NOISY, STEP, at,
                                                           molecule, sop_scf)

NAMES = ('formaldehyde', 'ethylene')


def chain(mol, tol):
    """The SOP chain of the production flag set at Davidson tolerance `tol`."""
    spec = SurfaceSpec(GroundState('dft', 'hf'), environment=None,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf', qp_states=QPStates(kind='frontier'),
                       numerics={'bse_adjoint': 'grid', 'bse_conv_tol': tol})
    return surface_of(spec, Excitation('singlet', root=1, kernel='bse'), mol,
                      sop_scf)


@pytest.mark.parametrize('name', NAMES)
def test_the_force_at_the_accepted_residual_follows_its_energy(name):
    warnings.simplefilter('ignore')
    h, bar = SOP_NOISY.get(name, (STEP, BAR))
    mol = molecule(name)
    with distributed(None):
        loose = chain(mol, BSE_FORCE_RESIDUAL_TOL)
        tight = chain(mol, BSE_DAVIDSON_CONV_TOL)
        g_loose, d_loose = loose.excitation_gradient()
        g_tight, d_tight = tight.excitation_gradient()
        fd = {}
        for atom, axis in COMPONENTS[name]:
            e = {k: loose.excitation(at(mol, atom, axis, k * h))
                 for k in (2, 1, -1, -2)}
            fd[atom, axis] = ((8.0 * (e[1] - e[-1]) - (e[2] - e[-2]))
                              / (12.0 * h))
    assert d_loose['omega'] != d_tight['omega']
    misses = np.array([g_loose[c] - v for c, v in fd.items()])
    moved = float(np.abs(g_loose - g_tight).max())
    print(f'\n{name}: omega moved {d_loose["omega"] - d_tight["omega"]:.1e} '
          f'Ha, force moved {moved:.1e}, FD misses '
          f'{np.array2string(misses, precision=2)} Ha/Bohr')
    assert np.abs(misses).max() <= bar, misses
    assert moved <= ISDF_GRADIENT_FLOOR, moved


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
