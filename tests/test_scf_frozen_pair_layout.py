"""The ISDF-K mean field of a walk keeps its reference geometry's pair layout.

The SCF's fit screens its test co-density pairs against `pair_tol`, a
discrete choice. Screened again at every geometry, the ground-state energy
steps wherever a pair crosses the threshold, while the force differentiates
the fit on one column set and cannot see the step: a central difference
across the crossing then grows as the step shrinks instead of falling as
h^2. Formaldehyde/cc-pVDZ Hartree-Fock on the default grid drops one pair of
the SCF's screen when a hydrogen moves 0.0125170 Bohr along y; the reference
sits 3.7e-5 Bohr short of that, so every stencil below straddles it.

Measured (diff = analytic - central difference, Ha/Bohr, h = 5e-4, 2.5e-4,
1.25e-4): re-screened -2.1e-6, -9.8e-6, -2.1e-5; on the frozen layout
3.1e-6, 7.7e-7, 2.0e-7.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.distributed_df import distributed_mean_field, release_distributed
from src.Base import separable_ri
from src.Base.distributed_isdf_jk import distributed_isdf_jk, scf_pair_layout
from src.Base.isdf_jk import frozen_pair_layout, isdf_grid, isdf_jk
from src.Base.utils.mpi_grid import run_simulated
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.optimize import MeanFieldSurface

ATOM = 'C 0 0 0; O 0 0 1.205; H 0 0.94 -0.58; H 0 -0.94 -0.58'
AUXBASIS = 'cc-pvdz-ri'
#: Bohr, hydrogen 2 along y: 3.7e-5 short of the pair crossing.
SHIFT = 0.01248
MOVED = (2, 1)
STEPS = (5e-4, 2.5e-4, 1.25e-4)


def _at(mol, d):
    coords = mol.atom_coords().copy()
    coords[MOVED] += d
    out = mol.copy()
    out.set_geom_(coords, unit='Bohr')
    out.build(False, False)
    return out


def _reference():
    return _at(gto.M(atom=ATOM, basis='cc-pvdz', verbose=0), SHIFT)


def _unrun(mol):
    mf = isdf_jk(scf.RHF(mol), auxbasis=AUXBASIS)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-11, 200
    return mf


def _factory(mol):
    mf = _unrun(mol)
    mf.kernel()
    assert mf.converged
    return mf


def _pairs(layout):
    return set(zip(np.asarray(layout[0]).tolist(),
                   np.asarray(layout[1]).tolist()))


def _screen(mol):
    return _pairs(_layout(mol))


def _layout(mol):
    return separable_ri.test_set_layout(mol, isdf_grid(mol, auxbasis=AUXBASIS))


def test_every_stencil_straddles_a_pair_crossing():
    mol0 = _reference()
    s0 = _screen(mol0)
    for h in STEPS:
        assert _screen(_at(mol0, h)) != s0
        assert _screen(_at(mol0, -h)) == s0


def test_the_ground_state_force_follows_its_energy_across_the_crossing():
    mol0 = _reference()
    surface = MeanFieldSurface(mol0, _factory)
    g = surface.total_gradient(mol0)[0][MOVED]
    diffs = []
    for h in STEPS:
        fd = (surface.total_energy(_at(mol0, h))
              - surface.total_energy(_at(mol0, -h))) / (2 * h)
        diffs.append(abs(g - fd))
    # h^2: a factor of 4 per halving, where a step the force cannot see
    # would grow as 1/h
    assert diffs[1] < diffs[0] / 3 and diffs[2] < diffs[1] / 3, diffs
    assert diffs[2] < 3e-7, diffs


def test_a_displaced_mean_field_fits_on_the_reference_layout():
    mol0 = _reference()
    surface = MeanFieldSurface(mol0, _factory)
    e0 = surface.total_energy(mol0)
    frozen = surface.pair_layout()
    assert _pairs(frozen) == _screen(mol0)
    moved = _at(mol0, STEPS[0])
    mf = surface.mean_field(moved)[1]
    assert mf.with_df.pair_layout is frozen
    assert _pairs(frozen) != _screen(moved)
    # the reference's own layout, frozen, is its own screen's fit bit for bit
    assert surface.total_energy(_at(mol0, 0.0)) == e0


def test_refreeze_takes_the_layout_at_the_new_reference():
    mol0 = _reference()
    surface = MeanFieldSurface(mol0, _factory)
    surface.total_energy(mol0)
    moved = _at(mol0, STEPS[0])
    again = surface.refreeze(moved)
    again.total_energy(moved)
    assert _pairs(again.pair_layout()) == _screen(moved)
    assert _pairs(again.pair_layout()) != _pairs(surface.pair_layout())


def test_the_chain_mean_field_keeps_the_reference_layout():
    mol0 = _reference()
    chain = RPAGroundStateChain(mol0, _factory)
    assert _pairs(chain.scf_layout()) == _screen(mol0)
    mf = chain.mean_field(_at(mol0, STEPS[0]))[1]
    assert mf.with_df.pair_layout is chain.scf_layout()


def test_the_distributed_scf_fits_on_the_frozen_layout():
    mol0 = _reference()
    moved = _at(mol0, STEPS[0])
    frozen, own = _layout(mol0), _layout(moved)
    serial = _unrun(moved)
    with frozen_pair_layout(frozen):
        serial.kernel()
    mfs = [_unrun(moved) for _ in range(2)]

    def one(comm):
        mf = mfs[comm.Get_rank()]
        with frozen_pair_layout(frozen):
            handle = distributed_isdf_jk(mf, comm)
            distributed_mean_field(mf)
        out = (handle.fit_kept.copy(), scf_pair_layout(mf), mf.e_tot)
        release_distributed(mf)
        return out

    def live(comm):
        mf = _unrun(moved)
        distributed_isdf_jk(mf, comm)
        distributed_mean_field(mf)
        out = scf_pair_layout(mf)
        release_distributed(mf)
        return out

    for kept, layout, e in run_simulated(one, 2):
        assert _pairs(layout) == _pairs(frozen)
        assert kept.sum() == len(frozen[0]) != len(own[0])
        assert abs(e - serial.e_tot) < 1e-10
    for layout in run_simulated(live, 2):
        assert _pairs(layout) == _pairs(own)


def test_a_scanner_keeps_the_layout_it_was_given():
    """pyscf's geometry optimizers reset one mean field at every geometry;
    a layout set on its ISDFJK travels through the reset (`relax_ground_state`
    sets the start's)."""
    mol0 = _reference()
    mf = _factory(mol0)
    frozen = scf_pair_layout(mf)
    mf.with_df.pair_layout = frozen
    scanner = mf.as_scanner()
    moved = _at(mol0, STEPS[0])
    e = scanner(moved)
    assert scanner.with_df.pair_layout is frozen
    ref = _unrun(moved)
    with frozen_pair_layout(frozen):
        ref.kernel()
    assert abs(e - ref.e_tot) < 1e-9


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
