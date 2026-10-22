"""One per-state contour solve, one start point, one static term.

The contour driver (`contour_deformation._contour_roots`) and the gradient
chain's pass 1 (`gradients.qp_space_time.qp_set_gradient`) each solved every
state with their own copy of the route branch (pole model or contour
equation), their own start point (`qp_starts` vs `frozen_start`) and their
own reading of the per-state static term. The forward holds the one version
of each (`contour_state_root`, `Solvers.qp_equation.qp_start`,
`static_term`); these tests pin the gradient's copies equal to them until it
calls them, and the two passes equal on a window of water/cc-pVDZ states.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inspect

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import CD_NTAU, HARTREE_TO_EV
import src.SingleReference.GW.contour_deformation as cd
from src.Solvers.qp_equation import qp_start
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.GW.space_time import separable_factors
import src.gradients.qp_space_time as qst
from tests.isdf_test_grid import g1
from tests.pins import assert_same

EPS = np.array([-0.95, -0.62, -0.48, 0.12, 0.31, 0.77])
NOCC = 3


@pytest.mark.parametrize('w0', [None, -0.5, {2: -0.47}, {4: 0.3}])
@pytest.mark.parametrize('p', [1, 2, 3, 4])
def test_one_start_point(w0, p):
    """The gradient's start point is the forward's for a state it froze
    nothing for, and the frozen root otherwise, for every kind of w0."""
    frozen = qst.frozen_root(w0, p)
    start = qst.frozen_start(w0, p, EPS, NOCC)
    assert start == (qp_start(EPS, p, NOCC) if frozen is None else frozen)
    assert cd.qp_starts(EPS, NOCC, [p])[0] == qst.frozen_start(None, p, EPS,
                                                               NOCC)


@pytest.mark.parametrize('xc', [0.0, 0.02, np.array([0.01, -0.03, 0.002])])
def test_one_static_term(xc):
    for si in range(3):
        assert qst.static_term(xc, si) == cd.static_term(xc, si)


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2
    x_mo, d = separable_factors(mf, mol, auxbasis='cc-pvdz-ri', counts=g1(mol))[:2]
    eps = np.asarray(mf.mo_energy, float)
    return dict(mol=mol, mf=mf, nocc=nocc, x_mo=x_mo, d=d, eps=eps,
                mu=0.5 * (eps[nocc - 1] + eps[nocc]))


@pytest.mark.skipif(
    'keep_tape' not in inspect.signature(qst.qp_set_gradient).parameters,
    reason='the gradient pass 1 this compares against (qp_set_gradient with '
           'keep_tape) is not available in this build')
@pytest.mark.parametrize('continuation,route', [('cd', 'explicit'),
                                                ('sop', 'sop')])
def test_both_passes_solve_every_state_alike(water, continuation, route):
    """The forward driver and the gradient's pass 1 return the same roots and
    pole strengths for a window of states on one grid: two solves of each
    root, so to a few ulp (`assert_same` of tests/pins.py), not bitwise."""
    w = water
    states = [w['nocc'] - 2, w['nocc'] - 1, w['nocc'], w['nocc'] + 1]
    energies, z_fwd, _ = cd.solve_qp_energy_contour(
        w['mf'], w['mol'], w['nocc'], states, continuation=continuation,
        factors=(w['x_mo'], w['d']))
    nu, wt, grid, _ = cd.cd_frequency_grid(w['eps'], w['nocc'], ntau=CD_NTAU)
    xc = static_exchange_diagonal(w['mf'], w['mol'], states)
    out = {}
    qst.qp_set_gradient(w['x_mo'], w['d'], w['eps'], w['nocc'], grid, nu, wt,
                        states, np.zeros(len(states)), residue_route=route,
                        mu=w['mu'], xc_correction=xc, route_out=out,
                        keep_tape=False)
    roots = np.array([out['roots'][p] for p in states])
    assert_same(roots * HARTREE_TO_EV, energies, unit='eV', quantity='energy',
                label='roots')
    assert_same(np.asarray(out['z']), z_fwd, unit='1', quantity='energy',
                label='Z')


def test_the_driver_solves_through_the_one_state_solve(monkeypatch):
    """`_contour_roots` reaches every state through `contour_state_root`."""
    seen = []

    def spy(route, p, *a, **k):
        seen.append((route, p))
        return float(EPS[p]), 1.0, []

    monkeypatch.setattr(cd, 'contour_state_root', spy)
    starts = cd.qp_starts(EPS, NOCC, [2, 3])
    roots, _, records = cd._contour_roots(
        'cd', [2, 3], starts, [(True, 0.1), (True, 0.1)], [None, None],
        [None, None], EPS, NOCC, None, None, np.zeros(2), 3, 1, None)
    assert seen == [('contour', 2), ('contour', 3)]
    assert roots.tolist() == EPS[[2, 3]].tolist()
    assert [r['qp_start'] for r in records] == starts.tolist()
