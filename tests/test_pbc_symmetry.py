"""k-mesh symmetry (`pbc_symmetry`).

* Every operation carries a one-body operator at k onto the one at Sk: the
  overlap is checked when the group is built, so the kinetic energy, with d
  shells (gth-dzvp), is the independent check (time-reversed operations
  included; Si has no inversion among pyscf's symmorphic operations).
* The stars partition the mesh; the group average of a covariant field is the
  field; unfolding from the irreducible points rebuilds it.
* The group of a 2x2x1 mesh is only the part of the crystal's group that maps
  that mesh onto itself.
* The average over a group of ranks, split by tau and gathered (or whole on
  each rank for a field without tau), is the serial one bit for bit.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf.pbc import gto

from src.Base.numerics.constants import KMESH_SYMMETRY_TOL
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.Periodic.pbc_symmetry import KMeshSymmetry, kmesh_symmetry

SI_LATTICE = np.array([[0.0, 2.715, 2.715], [2.715, 0.0, 2.715],
                       [2.715, 2.715, 0.0]])


@pytest.fixture(scope='module')
def si_dzvp():
    cell = gto.M(a=SI_LATTICE, atom='Si 0 0 0; Si 1.3575 1.3575 1.3575',
                 basis='gth-dzvp', pseudo='gth-pade', verbose=0)
    kpts = cell.make_kpts([3, 3, 3])
    return cell, kpts, KMeshSymmetry(cell, kpts)


def test_operations_carry_the_kinetic_energy(si_dzvp):
    cell, kpts, sym = si_dzvp
    T = np.asarray(cell.pbc_intor('int1e_kin', hermi=1, kpts=kpts))
    assert sym.nop == 48 and sym.time_reversal.sum() == 24
    worst = max(np.abs(sym.rotate(s, k, T[k]) - T[sym.image[s, k]]).max()
                for s in range(sym.nop) for k in range(sym.nk))
    assert worst < KMESH_SYMMETRY_TOL


def test_stars_partition_the_mesh(si_dzvp):
    cell, kpts, sym = si_dzvp
    assert list(sym.irreducible) == sorted(set(sym.rep))
    assert sym.star_size[sym.irreducible].sum() == sym.nk == 27
    assert len(sym.irreducible) == 4
    assert all(sym.rep[sym.image[s, k]] == sym.rep[k]
               for s in range(sym.nop) for k in range(sym.nk))


def test_average_and_unfold_keep_a_covariant_field(si_dzvp):
    cell, kpts, sym = si_dzvp
    T = np.asarray(cell.pbc_intor('int1e_kin', hermi=1, kpts=kpts))
    assert np.abs(sym.symmetrize(T) - T).max() < KMESH_SYMMETRY_TOL
    assert np.abs(sym.unfold(T) - T).max() < KMESH_SYMMETRY_TOL


def test_a_lower_mesh_keeps_only_its_own_operations():
    cell = gto.M(a=SI_LATTICE, atom='Si 0 0 0; Si 1.3575 1.3575 1.3575',
                 basis='gth-szv', pseudo='gth-pade', verbose=0)
    sym = KMeshSymmetry(cell, cell.make_kpts([2, 2, 1]))
    assert sym.nop == 8
    assert sym.star_size[sym.irreducible].sum() == 4


def test_trivial_group_changes_nothing():
    sym = kmesh_symmetry(None, np.zeros((5, 3)), False)
    field = np.random.default_rng(0).normal(size=(5, 2, 3, 3))
    assert sym.symmetrize(field) is field
    assert list(sym.irreducible) == list(range(5)) and (sym.star_size == 1).all()


def test_a_shifted_mesh_is_refused():
    cell = gto.M(a=SI_LATTICE, atom='Si 0 0 0; Si 1.3575 1.3575 1.3575',
                 basis='gth-szv', pseudo='gth-pade', verbose=0)
    with pytest.raises(ValueError, match='Gamma-centred'):
        KMeshSymmetry(cell, cell.make_kpts([2, 2, 2], scaled_center=[0.25, 0.25, 0.25]))


@pytest.mark.parametrize('inner', [(5,), (2,), ()], ids=['tau5', 'tau2', 'no-tau'])
def test_average_over_ranks_is_the_serial_one(si_dzvp, inner):
    """Split over 3 ranks by tau (tau2 leaves one rank empty) and gathered, or
    whole on each rank without tau: the serial average bit for bit."""
    cell, kpts, sym = si_dzvp
    rng = np.random.default_rng(1)
    shape = (sym.nk,) + inner + (cell.nao_nr(),) * 2
    field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    serial = sym.symmetrize(field)
    for out in run_simulated(lambda comm: sym.symmetrize(field), 3):
        assert out.shape == serial.shape and out.flags.c_contiguous
        assert np.array_equal(out, serial)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
