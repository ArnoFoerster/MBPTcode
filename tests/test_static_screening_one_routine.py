"""The BSE kernel and the reaction field screen with ONE routine.

`LinearResponse.davidson.static_screening_matrix` on
`LinearResponse.davidson.static_screening_grid` is the single build of
W = [1 - chi0(i.omega = 0)]^-1 from the separable factors.
`isdf_bse_factors(screening='imaginary-time')` returns it as the BSE kernel's
W_aux and `gradients.reaction_field_adjoint.static_screening` returns it beside
the orbital densities A[Q,p] = sum_k D[k,Q] X[k,p]^2 that Duchemin, Guido,
Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18) contracts it against.
The two entry points must therefore hand back the SAME BITS, serially and off
the same tau partition, on the default grid and on one the caller sizes itself.

The last test keeps the dependency pointing one way: production never imports
`src.gradients`, so the shared routine lives on the production side and the
gradient package is the one that calls in.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.davidson import (
    isdf_bse_factors, minimax_points_for_bse, static_screening_grid,
    static_screening_matrix)
from src.gradients.reaction_field_adjoint import static_grid, static_screening

BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
SIZES = [2, 3]
#: chi0 accumulates its tau points as they arrive, so a tau partition
#: re-associates that sum: the ranks agree with the serial answer at the last
#: bits, not in them. Measured 2.2e-16 relative on water/cc-pVDZ.
RANK_REL = 2.3e-16
#: A grid the caller sizes itself, so the ntau argument of both entry points is
#: exercised and not only the shared default.
NTAU_FIXED = 12


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol, mf.conv_tol_grad = 1e-14, 1e-11
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis=AUXBASIS)
    return dict(mf=mf, mol=mol, nocc=nocc, factors=factors, X=factors[0],
                D=factors[1], eps=np.asarray(mf.mo_energy, float))


def test_the_entry_points_return_one_matrix(water):
    """Same W, bit for bit, on the shared default grid and on a fixed one."""
    w = water
    _, w_rf = static_screening(w['X'], w['D'], w['eps'], w['nocc'],
                               static_grid(w['eps'], w['nocc']))
    w_bse = isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                             factors=w['factors'])[2]
    assert np.array_equal(w_rf, w_bse)

    _, w_rf12 = static_screening(w['X'], w['D'], w['eps'], w['nocc'],
                                 static_grid(w['eps'], w['nocc'],
                                             ntau=NTAU_FIXED))
    w_bse12 = isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                               factors=w['factors'], ntau=NTAU_FIXED)[2]
    assert np.array_equal(w_rf12, w_bse12)
    assert not np.array_equal(w_rf, w_rf12), 'ntau must reach the grid'


def test_the_grids_are_one_object(water):
    """`static_grid` is `static_screening_grid`: the same axis, not a copy of
    its construction."""
    w = water
    a = static_grid(w['eps'], w['nocc'])
    b = static_screening_grid(w['eps'], w['nocc'])
    assert a.ntau == b.ntau == minimax_points_for_bse(w['eps'], w['nocc'])[0]
    assert a.nfreq == b.nfreq == 1
    for name in ('tau_points', 'tau_weights', 'omega_points', 'omega_weights',
                 'cosft_wt'):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    # a single input frequency cannot constrain the omega -> tau fit
    assert a.cosft_tw is None and b.cosft_tw is None
    assert np.array_equal(a.omega_points, np.zeros(1))


def test_static_screening_still_returns_the_orbital_densities(water):
    """A[Q,p] = sum_k D[k,Q] X[k,p]^2, the density Eq. (18) contracts W
    against, is this entry point's own second return value."""
    w = water
    a, mat = static_screening(w['X'], w['D'], w['eps'], w['nocc'])
    assert a.shape == (w['D'].shape[1], w['X'].shape[1])
    assert np.array_equal(a, w['D'].T @ (w['X'] ** 2))
    assert mat.shape == (w['D'].shape[1], w['D'].shape[1])
    assert np.array_equal(mat, static_screening_matrix(
        w['X'], w['D'], w['eps'], w['nocc'], static_grid(w['eps'], w['nocc'])))


@pytest.mark.parametrize('size', SIZES)
def test_the_entry_points_agree_under_ranks(water, size):
    """Off the same tau partition the two entry points are still one matrix,
    bitwise, and each sits at the re-associated sum's distance from serial."""
    w = water
    grid = static_grid(w['eps'], w['nocc'])
    a0, w_serial = static_screening(w['X'], w['D'], w['eps'], w['nocc'], grid)

    ranks_rf = run_simulated(
        lambda c: static_screening(w['X'], w['D'], w['eps'], w['nocc'], grid,
                                   comm=c), size)
    ranks_bse = run_simulated(
        lambda c: isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                                   factors=w['factors'], distribute=True,
                                   comm=c)[2], size)
    for (a, w_rf), w_bse in zip(ranks_rf, ranks_bse):
        assert np.array_equal(w_rf, w_bse)
        assert np.array_equal(a, a0)            # no sweep, so bitwise
        rel = np.abs(w_rf - w_serial).max() / np.abs(w_serial).max()
        assert rel <= RANK_REL, f'{size} ranks moved W by {rel:.2e}'


def test_production_does_not_import_the_gradient_package():
    """The shared routine lives in production, so importing it must not drag
    `src.gradients` in: the dependency runs the other way."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ('import sys; sys.path.insert(0, %r);'
            'import src.SingleReference.LinearResponse.davidson as d;'
            'assert hasattr(d, "static_screening_matrix");'
            'print([m for m in sys.modules if m.startswith("src.gradients")])'
            % root)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, cwd=root)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '[]', out.stdout
