"""The fast twin of tests/test_one_fit_forces_follow_their_energy.py.

The same claims on the same fit: every route's force (dRPA, BSE@GW on the
contour-deformation residues, the SOP route) is the derivative of the energy it
reports, serially on the replicated fit and on the row fit under ranks, and the
ISDF-K mean-field force is the derivative of the ISDF-K SCF energy. Cheaper in
three ways:

  * formaldehyde/cc-pVDZ only (6 of 1444 pairs screened, so the screen is
    still gated); ethylene, the SOP route's noisy case, stays nightly;
  * one difference along a seeded direction over every coordinate
    (tests/directional_difference.py) instead of three components, four
    energies instead of twelve;
  * the row fit at 3 simulated ranks (formaldehyde's 10 tiles fall unevenly)
    instead of 2, 3 and 8.

Same bars as the nightly file, which takes about 11 minutes; this one: see
tools/ci/fast.txt.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.numerics.constants import ISDF_GRADIENT_FLOOR, ISDF_ROW_FORCE_ENERGY_TOL
from src.Base.factorization.distributed_df import distributed_mean_field
from src.Base.factorization.distributed_isdf_jk import distributed_isdf_jk
from src.Base.utils.parallel.mpi_grid import distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.isdf_derivatives import product_pairs
from src.gradients.isdf_mean_field import isdf_mean_field_gradient
from tests import directional_difference as dd
from tests.test_one_fit_forces_follow_their_energy import (BAR, MF_COMPONENTS,
                                                           ROUTES,
                                                           ROW_FIT, SCF_STEPS,
                                                           STEP, TILE,
                                                           isdf_scf, molecule)
from tests.pins import assert_same

NAME = 'formaldehyde'
SIZE = 3


@pytest.mark.parametrize('route', list(ROUTES))
def test_the_chain_force_follows_its_energy_along_a_direction(route):
    mol = molecule(NAME)
    d = dd.direction(mol)
    with distributed(None):
        serial = ROUTES[route](mol)
        assert serial.chain.fit == 'replicated' and not serial.chain.sliced
        screened = len(product_pairs(mol)[0]) - len(serial.chain.layout[0])
        assert screened > 0, 'the screen drops no pair: no gate of the fit'
        force = serial.force()
        fd = dd.difference(serial.energy, mol, d, STEP)
    miss = dd.along(force, d) - fd
    assert abs(miss) < BAR, (route, miss)

    def rank(comm):
        return ROUTES[route](molecule(NAME), **ROW_FIT).force()

    out = run_simulated(rank, SIZE)
    for f in out:
        # each rank assembles its own force: rank 0's to a few ulp
        assert_same(f, out[0], quantity='gradient', label='rank force')
    apart = float(np.abs(out[0] - force).max())
    rows_miss = dd.along(out[0], d) - fd
    print(f'\n{route} {NAME}: serial {miss:.1e}, rows at {SIZE} '
          f'{rows_miss:.1e}, apart {apart:.1e} Ha/Bohr')
    assert abs(rows_miss) < BAR, rows_miss
    assert apart < ISDF_GRADIENT_FLOOR, apart


def test_the_isdf_k_mean_field_force_follows_its_energy_along_a_direction():
    """Hartree-Fock: the row realization serially and the distributed ISDF-K
    SCF's own force at 3 simulated ranks, against a Richardson difference.

    Along the nightly file's two components, not a random direction: the
    carbon's y component carries a third derivative of ~2e3 Ha/Bohr^3 on the
    ISDF-K surface (its frame turns with the molecule), so the Richardson
    limit at SCF_STEPS misses it by 7e-6 although the central difference
    falls as h^2 to 1.9e-7 at h = 1e-5. A seeded direction picks that
    component up and fails on truncation, not on the force."""
    mf = isdf_scf(NAME, 'hf')
    d = np.zeros((mf.mol.natm, 3))
    for atom, axis in MF_COMPONENTS:
        d[atom, axis] = 1.0
    d /= np.linalg.norm(d)

    def energy(m):
        return isdf_scf(NAME, 'hf', m.atom_coords()).kernel()

    fd, _ = dd.richardson(energy, mf.mol, d, SCF_STEPS)
    with distributed(None):
        mf.kernel()
        force = isdf_mean_field_gradient(mf, fit='rows')
    miss = dd.along(force, d) - fd
    assert abs(miss) < ISDF_ROW_FORCE_ENERGY_TOL, miss

    def rank(comm):
        mf = isdf_scf(NAME, 'hf')
        distributed_isdf_jk(mf, comm, tile=TILE)
        distributed_mean_field(mf)
        return isdf_mean_field_gradient(mf)

    out = run_simulated(rank, SIZE)
    for f in out:
        # each rank assembles its own force: rank 0's to a few ulp
        assert_same(f, out[0], quantity='gradient', label='rank force')
    apart = float(np.abs(out[0] - force).max())
    print(f'\n{NAME}/HF ISDF-K mean field: serial {miss:.1e}, {SIZE} ranks '
          f'{dd.along(out[0], d) - fd:.1e}, apart {apart:.1e}')
    assert abs(dd.along(out[0], d) - fd) < ISDF_ROW_FORCE_ENERGY_TOL
    assert apart < ISDF_GRADIENT_FLOOR, apart


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
