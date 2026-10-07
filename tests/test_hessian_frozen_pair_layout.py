"""A finite-difference Hessian fits every displaced ISDF-K mean field on the
reference geometry's pair layout.

The SCF's ISDF-K fit screens its AO pairs against `pair_tol`, a discrete
choice. `numerical_hessian` differences forces of independent SCFs at
R0 +/- h; screened again at each of them, a pair crossing the threshold
between R0 - h and R0 + h makes the two forces of one central difference
forces of two energies, and the column an O(1/h) error. Formaldehyde/cc-pVDZ
Hartree-Fock on the G3 grid (the Hessian floor) drops one pair of the SCF's
screen when hydrogen 2 moves CROSSING Bohr along y; the reference sits 3.7e-5
Bohr short of it, so every stencil along that coordinate straddles it (and,
the pair being shared, several other coordinates' stencils do too).

The same construction as test_scf_frozen_pair_layout, on the
equilibrium-like geometry of test_numerical_hessian, where the G3 surface is
smooth on the step's scale.

Measured (max |H - H^T| before symmetrizing, Ha/Bohr^2, h = 1e-3, 5e-4,
2.5e-4): each SCF on its own screen 2.1e-4, 4.2e-4, 8.4e-4, growing as 1/h;
on R0's layout 3.9e-5, 1.1e-5, 3.0e-6.
"""
import functools
import os
import sys

import numpy as np
import pytest
from pyscf import gto, lib, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base import separable_ri
from src.Base.constants import ISDF_GRID_ACCURACY, NUCLEAR_FD_STEP
from src.Base.distributed_isdf_jk import scf_pair_layout
from src.Base.isdf_jk import isdf_grid, isdf_jk
from src.Base.separable_ri import _SHELL_ORDER
from src.Base.utils.mpi_grid import mpi_map, run_simulated
from src.properties import hessian as H

ATOM = ('C 0.0 0.0 -0.6035; O 0.0 0.0 0.7349; '
        'H 0.0 0.9753 -1.1057; H 0.0 -0.9753 -1.1057')
AUXBASIS = 'cc-pvdz-ri'
COUNTS = dict(zip(_SHELL_ORDER, ISDF_GRID_ACCURACY['cc-pvdz']['G3']))
#: Bohr, hydrogen 2 along y from ATOM: one pair of the G3 screen drops here.
CROSSING = 0.008182855963706971
SHIFT = CROSSING - 3.7e-5
MOVED = (2, 1)
#: the moved coordinate's index in the (3 natm) flattening
COLUMN = 3 * MOVED[0] + MOVED[1]
STEPS = (NUCLEAR_FD_STEP, NUCLEAR_FD_STEP / 2, NUCLEAR_FD_STEP / 4)


class Spy:
    """`_factory`, recording the pair layout every mean field it built fits
    on (list.append is atomic, so rank threads may share one)."""

    def __init__(self):
        self.layouts = []

    def __call__(self, mol):
        mf = _factory(mol)
        self.layouts.append(mf.with_df.pair_layout)
        return mf


def _reference():
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    coords = mol.atom_coords().copy()
    coords[MOVED] += SHIFT
    mol.set_geom_(coords, unit='Bohr')
    mol.build(False, False)
    return mol


def _factory(mol):
    mf = isdf_jk(scf.RHF(mol), auxbasis=AUXBASIS, counts=COUNTS)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


def _screen(mol):
    lay = separable_ri.test_set_layout(
        mol, isdf_grid(mol, counts=COUNTS, auxbasis=AUXBASIS))
    return set(zip(lay[0].tolist(), lay[1].tolist()))


def _pairs_of(layout):
    return set(zip(np.asarray(layout[0]).tolist(),
                   np.asarray(layout[1]).tolist()))


def _same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def _flat(h):
    natm = h.shape[0]
    return np.asarray(h).transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)


def _raw(gradients, natm, step):
    """The unsymmetrized Hessian, rows the force, columns the displacement."""
    h = np.zeros((3 * natm, 3 * natm))
    for (ia, x, sign), g in gradients.items():
        if sign > 0:
            h[:, 3 * ia + x] = (np.ravel(g) - np.ravel(
                gradients[(ia, x, -1)])) / (2.0 * step)
    return h


@pytest.fixture(scope='module')
def mol0():
    return _reference()


@pytest.fixture(scope='module')
def reference(mol0):
    """R0's mean field and the layout of its ISDF-K fit."""
    mf = _factory(mol0)
    return mf, scf_pair_layout(mf)


@pytest.fixture(scope='module')
def serial(mol0):
    """{step: (Hessian, raw Hessian, info, layouts seen)} by
    `numerical_hessian`, its own reference SCF included. The 2h repeat is
    `check_step_scaling`'s business (test_numerical_hessian) and is off."""
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H, 'HESSIAN_FD_NOISE_REL', np.inf)
        gradient_at = H.gradient_at
        for step in STEPS:
            grads = {}

            def keep(m, factory, ia, x, sign, h):
                grads[(ia, x, sign)] = gradient_at(m, factory, ia, x, sign, h)
                return grads[(ia, x, sign)]

            mp.setattr(H, 'gradient_at', keep)
            spy, info = Spy(), {}
            hess = H.numerical_hessian(mol0, spy, step=step, info=info)
            out[step] = (hess, _raw(grads, mol0.natm, step), info,
                         spy.layouts)
    return out


def test_every_stencil_straddles_a_pair_crossing(mol0):
    s0 = _screen(mol0)
    for h in STEPS:
        assert _screen(H.displaced(mol0, *MOVED, +1, h)) != s0
        assert _screen(H.displaced(mol0, *MOVED, -1, h)) == s0


def test_every_displaced_mean_field_fits_on_the_reference_layout(
        mol0, reference, serial):
    frozen = reference[1]
    assert _pairs_of(frozen) == _screen(mol0)
    for step, (_, _, info, layouts) in serial.items():
        # the reference SCF screens its own pairs, every displaced one is
        # handed them
        assert layouts[0] is None
        assert len(layouts) == 1 + 6 * mol0.natm
        assert all(_same(lay, frozen) for lay in layouts[1:]), step
        assert info['frozen_pairs'] == len(frozen[0])


def test_the_crossed_column_is_symmetric_and_converges_as_h2(serial):
    """The column of the moved coordinate: its asymmetry against the row
    falls by 4 per halving of h, and so does the change of the symmetrized
    Hessian. Each displaced SCF on its own screen, the column carried the
    step's force difference over 2h on top (measured below)."""
    asym = [float(np.abs(raw[:, COLUMN] - raw[COLUMN, :]).max())
            for _, raw, _, _ in serial.values()]
    hs = [_flat(hess) for hess, _, _, _ in serial.values()]
    moves = [float(np.abs(b - a).max()) for a, b in zip(hs, hs[1:])]
    for a, b in zip(asym, asym[1:]):
        assert 3.0 < a / b < 5.0, asym
    assert 3.0 < moves[0] / moves[1] < 5.0, moves


def test_a_per_geometry_screen_puts_a_one_over_h_error_into_the_column(
        mol0, reference, serial):
    """The defect this guards against is present in this construction: the
    moved coordinate's column with each SCF on its own screen differs from
    the frozen one by a force step over 2h, growing as h shrinks."""
    gaps = []
    for step, (_, raw, _, _) in serial.items():
        own = [H.gradient_at(mol0, _factory, *MOVED, sign, step)
               for sign in (+1, -1)]
        column = (np.ravel(own[0]) - np.ravel(own[1])) / (2.0 * step)
        gaps.append(float(np.abs(column - raw[:, COLUMN]).max()))
    for a, b in zip(gaps, gaps[1:]):
        assert b > 1.5 * a, gaps


def test_ranks_difference_forces_on_one_layout(mol0, reference, serial,
                                              monkeypatch):
    """`mpi_map` over 2 simulated ranks: one rank converges the reference,
    every rank fits its displaced mean fields on that layout, and the Hessian
    is the serial one."""
    frozen = reference[1]
    step = STEPS[0]
    spy = Spy()
    # as in `serial`: here the asymmetry at 2h is 1.9x the one at h, past
    # this surface's h^2 range, and `check_step_scaling` would refuse it
    monkeypatch.setattr(H, 'HESSIAN_FD_NOISE_REL', np.inf)
    threads = lib.num_threads()
    lib.num_threads(1)
    try:
        ranked = run_simulated(
            lambda comm: H.numerical_hessian(
                mol0, spy, step=step,
                map_fn=functools.partial(mpi_map, comm=comm)), 2)
    finally:
        lib.num_threads(threads)
    assert len(spy.layouts) == 1 + 6 * mol0.natm
    assert sum(lay is None for lay in spy.layouts) == 1
    assert all(_same(lay, frozen) for lay in spy.layouts if lay is not None)
    assert np.array_equal(ranked[0], ranked[1])
    assert float(np.abs(ranked[0] - serial[step][0]).max()) < 1e-6


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
