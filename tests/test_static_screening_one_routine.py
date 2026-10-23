"""The BSE kernel and the reaction field screen with ONE routine.

`LinearResponse.space_time.static_screening_matrix` on
`LinearResponse.space_time.static_screening_grid` is the single build of
W = [1 - chi0(i.omega = 0)]^-1 from the separable factors.
`isdf_bse_factors(screening='imaginary-time')` returns it as the BSE kernel's
W_aux and `gradients.reaction_field_adjoint.static_screening` returns it beside
the orbital densities A[Q,p] = sum_k D[k,Q] X[k,p]^2 that Duchemin, Guido,
Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18) contracts it against.
The two entry points must therefore hand back the SAME BITS, serially and off
the same tau partition, on the default grid and on one the caller sizes itself.

Every other molecular route on the separable factors reads the same matrix:
the space-time GW stage's W(0) (in core, frequency-blocked and solvated, where
the Eq. (18) shift reads it), the dense ISDF BSE, and so the Davidson BSE
gives the same roots whether it runs its own GW stage or is handed the
diagonal. A GW stage that carried W(0) as an omega = 0 row of its own
axis (ntau 20 on water/cc-pVDZ against 14 here), or a dense ISDF BSE that
screened exactly on the implied three-index factor, would move the singlet S1
by 0.26 and 0.27 ueV against this routine.
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

from src.Base.numerics.constants import SCF_DIFFERENTIABLE_GRAD_TOL, SCF_TIGHT_CONV_TOL
from src.Base.environment_seam import attach_environment
from src.Base.numerics.numerics import Numerics
from src.environments.solvent_screening import SolventScreening
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_diagonal_space_time)
from src.SingleReference.BSE.solve import solve_bse
from src.SingleReference.BSE.solve import isdf_bse_factors
from src.SingleReference.LinearResponse.space_time import (
    minimax_points_for_bse, static_screening_grid, static_screening_matrix)
from src.gradients.gwbse.reaction_field_adjoint import static_grid, static_screening
from tests.isdf_test_grid import g1
from tests.pins import assert_same

BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
SIZES = [2, 3]
#: chi0 accumulates its tau points as they arrive, so a tau partition
#: re-associates that sum: the ranks agree with the serial answer at the last
#: bits, not in them. Measured 2.2e-16 relative on water/cc-pVDZ.
RANK_REL = 2.3e-16
#: A grid the caller sizes itself, so the ntau argument of both entry points is
#: exercised and not only the shared default.
NTAU_FIXED = 12
#: Optical dielectric constant of the continuum the solvated gate attaches.
EPS_SOLVENT = 1.78


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis=AUXBASIS, counts=g1(mol))
    return dict(mf=mf, mol=mol, nocc=nocc, factors=factors, X=factors[0],
                D=factors[1], eps=np.asarray(mf.mo_energy, float))


def test_the_entry_points_return_one_matrix(water):
    """Same W, to a few ulp, on the shared default grid and on a fixed one."""
    w = water
    _, w_rf = static_screening(w['X'], w['D'], w['eps'], w['nocc'],
                               static_grid(w['eps'], w['nocc']))
    w_bse = isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                             factors=w['factors'])[2]
    assert_same(w_rf, w_bse, quantity='root', label='default grid')

    _, w_rf12 = static_screening(w['X'], w['D'], w['eps'], w['nocc'],
                                 static_grid(w['eps'], w['nocc'],
                                             ntau=NTAU_FIXED))
    w_bse12 = isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                               factors=w['factors'], numerics=Numerics(quadrature={'ntau_w': NTAU_FIXED}))[2]
    assert_same(w_rf12, w_bse12, quantity='root', label='fixed grid')
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
        assert_same(getattr(a, name), getattr(b, name), unit='1',
                    quantity='root', label=name)
    # a single input frequency cannot constrain the omega -> tau fit
    assert a.cosft_tw is None and b.cosft_tw is None
    assert np.array_equal(a.omega_points, np.zeros(1))


def test_static_screening_still_returns_the_orbital_densities(water):
    """A[Q,p] = sum_k D[k,Q] X[k,p]^2, the density Eq. (18) contracts W
    against, is this entry point's own second return value."""
    w = water
    a, mat = static_screening(w['X'], w['D'], w['eps'], w['nocc'])
    assert a.shape == (w['D'].shape[1], w['X'].shape[1])
    assert_same(a, w['D'].T @ (w['X'] ** 2), unit='1', quantity='root',
                label='orbital densities')
    assert mat.shape == (w['D'].shape[1], w['D'].shape[1])
    assert_same(mat, static_screening_matrix(
        w['X'], w['D'], w['eps'], w['nocc'], static_grid(w['eps'], w['nocc'])),
        quantity='root', label='W')


@pytest.mark.parametrize('size', SIZES)
def test_the_entry_points_agree_under_ranks(water, size):
    """Off the same tau partition the two entry points are still one matrix,
    to a few ulp, and each sits at the re-associated sum's distance from
    serial."""
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
        assert_same(w_rf, w_bse, quantity='root', label='two entry points')
        # no sweep: the same local contraction, a few ulp
        assert_same(a, a0, unit='1', quantity='root', label='A')
        rel = np.abs(w_rf - w_serial).max() / np.abs(w_serial).max()
        assert rel <= RANK_REL, f'{size} ranks moved W by {rel:.2e}'


@pytest.mark.parametrize('route', [{}, {'freq_block': 4}])
def test_the_gw_stage_hands_over_the_one_matrix(water, route):
    """extras['w_static'] of the space-time GW stage, in core and frequency
    blocked, is the BSE kernel's W, to a few ulp."""
    w, extras = water, {}
    solve_qp_diagonal_space_time(w['mf'], w['mol'], w['nocc'],
                                 factors=w['factors'], extras=extras, **route)
    w_bse = isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                             factors=w['factors'])[2]
    assert_same(extras['w_static'], w_bse, quantity='root', label='GW stage W')
    assert extras['w_static_ntau'] == static_screening_grid(
        w['eps'], w['nocc']).ntau


def test_the_solvated_shift_reads_the_one_matrix(water):
    """In a continuum the Eq. (18) shift of the GW stage reads the dressed W
    of the one static build."""
    w = water
    mf = scf.RHF(w['mol']).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-12
    mf.kernel()
    attach_environment(mf, SolventScreening(w['mol'], eps=EPS_SOLVENT))
    factors = separable_factors(mf, w['mol'], auxbasis=AUXBASIS, counts=g1(w['mol']))
    extras = {}
    solve_qp_diagonal_space_time(mf, w['mol'], w['nocc'], factors=factors,
                                 extras=extras)
    w_bse = isdf_bse_factors(mf, w['mol'], w['nocc'], factors=factors)[2]
    assert_same(extras['w_static'], w_bse, quantity='root',
                label='dressed W')


@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_the_bse_roots_do_not_depend_on_who_built_w(water, spin):
    """qp='G0W0' and the same diagonal handed in give the same roots, on the
    Davidson and on the dense ISDF route alike."""
    w = water
    kw = dict(numerics=Numerics(bse={'nroots': 3}), integrals='isdf', spin=spin, factors=w['factors'])
    om, _, _, info = solve_bse(w['mf'], w['mol'], w['nocc'], qp='G0W0', **kw)
    om_q = solve_bse(w['mf'], w['mol'], w['nocc'], qp=info['eps'], **kw)[0]
    assert_same(om, om_q, quantity='root', label='roots')
    w_bse = isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                             factors=w['factors'])[2]
    for qp in ('G0W0', info['eps']):
        dense = solve_bse(w['mf'], w['mol'], w['nocc'], qp=qp,
                          solver='dense', **kw)
        assert_same(dense[3]['W_aux'], w_bse, quantity='root',
                    label='dense route W')


def test_production_does_not_import_the_gradient_package():
    """The shared routine lives in production, so importing it must not drag
    `src.gradients` in: the dependency runs the other way."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ('import sys; sys.path.insert(0, %r);'
            'import src.SingleReference.BSE.solve as d;'
            'assert hasattr(d, "static_screening_matrix");'
            'print([m for m in sys.modules if m.startswith("src.gradients")])'
            % root)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, cwd=root)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '[]', out.stdout
