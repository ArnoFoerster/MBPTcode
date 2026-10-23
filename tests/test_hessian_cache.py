"""The finite-difference Hessian keeps its forces on disk and refuses loudly.

`surface_hessian(..., cache_dir=d)` writes each displaced force to
d/<atom>_<axis>_<sign>.npz with the geometry, the step and the hash of the
surface's record, and reads one already there instead of computing it: a
Hessian whose cache is half filled computes only the missing forces and is
the same matrix to the bit. A file written for another step or another
surface is refused. A Hessian refused for its asymmetry leaves
d/asymmetry.json first, with the worst elements and their atom/axis pairs.
Water/STO-3G Hartree-Fock: the cache does not care what the force is of.
"""
import json

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import (HESSIAN_ASYMMETRY_WORST, NUCLEAR_FD_STEP,
                                         PIN_CROSS_BLAS_GRADIENT_BAR)
from src.properties.hessian import surface_hessian
from src.gradients.mean_field_surface import MeanFieldSurface
from tests.blas_bar import assert_same_within


class Counted:
    """A surface that counts its force evaluations."""

    def __init__(self, surface, record=None):
        self.surface, self.calls, self.record = surface, 0, record
        self.mol0 = surface.mol0

    def total_gradient(self, mol=None, mf=None):
        self.calls += 1
        return self.surface.total_gradient(mol)


class Recorded(Counted):
    """A counted surface that writes a record (`surface_record`)."""

    def to_record(self):
        return self.record

#: Hessian bar on a BLAS other than the pin files': a force's class bar over
#: the central difference's 2 * step.
BAR = PIN_CROSS_BLAS_GRADIENT_BAR / (2 * NUCLEAR_FD_STEP)


def water():
    """Water/STO-3G and its Hartree-Fock surface."""
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
                basis='sto-3g', verbose=0)

    def factory(m):
        mf = scf.RHF(m)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        return mf.run()
    return mol, MeanFieldSurface(mol, factory)


def test_a_half_filled_cache_computes_only_the_missing_forces(tmp_path):
    mol, surface = water()
    whole = Counted(surface)
    info = {}
    h = surface_hessian(whole, mol, cache_dir=tmp_path, info=info)
    n = 6 * mol.natm
    files = sorted(tmp_path.glob('*.npz'))
    assert whole.calls == n and len(files) == n
    assert info['forces_reused'] == 0
    for path in files[::2]:
        path.unlink()
    again = Counted(surface)
    info = {}
    h_again = surface_hessian(again, mol, cache_dir=tmp_path, info=info)
    assert again.calls == n // 2 and info['forces_reused'] == n // 2
    # The recomputed forces come from fresh SCF solves: the cached Hessian is
    # theirs bit for bit on the recording BLAS; on another every force moves by
    # the gradient class bar and the central difference divides that by 2 step.
    assert_same_within(h_again, h, BAR, label='recomputed against cached')
    # every force is on disk: nothing is computed, and the Hessian is the
    # previous one bit for bit (the same files, read back; not `h`, whose
    # missing half was recomputed by fresh SCF solves)
    third = Counted(surface)
    assert np.array_equal(surface_hessian(third, mol, cache_dir=tmp_path),
                          h_again)
    assert third.calls == 0


def test_a_force_of_another_hessian_is_refused(tmp_path):
    mol, surface = water()
    surface_hessian(Recorded(surface, {'conventions': 1}), mol,
                    cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match='another Hessian.*step'):
        surface_hessian(Recorded(surface, {'conventions': 1}), mol,
                        step=2 * NUCLEAR_FD_STEP, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match='another Hessian.*conventions'):
        surface_hessian(Recorded(surface, {'conventions': 2}), mol,
                        cache_dir=tmp_path)
    # forces on each geometry's own turning ISDF frames are forces of another
    # surface than the default stencil's, whose frames are frozen at mol
    with pytest.raises(RuntimeError, match='another Hessian.*frames'):
        surface_hessian(Recorded(surface, {'conventions': 1}), mol,
                        cache_dir=tmp_path, frames=None)


def test_a_cache_written_before_frames_were_frozen_is_refused(tmp_path):
    mol, surface = water()
    surface_hessian(Recorded(surface, {'conventions': 1}), mol,
                    cache_dir=tmp_path)
    for path in tmp_path.glob('*.npz'):
        with np.load(path) as held:
            old = {k: held[k] for k in held.files if k != 'frames'}
        np.savez(path, **old)
    with pytest.raises(RuntimeError, match='another Hessian.*frames'):
        surface_hessian(Recorded(surface, {'conventions': 1}), mol,
                        cache_dir=tmp_path)


def test_a_refused_hessian_leaves_its_asymmetry(tmp_path):
    mol, surface = water()
    info = {}
    with pytest.raises(RuntimeError, match='asymmetric'):
        surface_hessian(surface, mol, cache_dir=tmp_path, info=info,
                        asymmetry_tol=0.0)
    dump = json.loads((tmp_path / 'asymmetry.json').read_text())
    assert dump['asymmetry'] == info['asymmetry'] > 0.0
    assert dump['asymmetry_rel'] == info['asymmetry_rel']
    worst = dump['worst']
    assert len(worst) == HESSIAN_ASYMMETRY_WORST
    assert worst[0]['difference'] == dump['asymmetry']
    assert all(a['difference'] >= b['difference']
               for a, b in zip(worst, worst[1:]))
    first = worst[0]
    assert abs(first['h_ij'] - first['h_ji']) == first['difference']
    assert first['i'][1] in 'xyz' and 0 <= first['j'][0] < mol.natm
