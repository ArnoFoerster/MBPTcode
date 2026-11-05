"""A state that crosses below a walk's roots and that the fold model does not see
is caught by the full completeness check, at every EE_FOLD_FULL_CHECK_EVERY-th
warm solve and at the walk's converged geometry.

  1. ee_fold, synthetic pieces (test_ee_adc_fold_solve_speed.crossing 'coupled':
     800 singles, a block of 50 coupled to one double, with the exact fold
     diagonal as the model), the block dropping below root 1 between two calls,
     nroots 1 and 3: the warm solve misses it (its check runs the seeded search
     alone); with full_check it is found, every root equal to the supermatrix's
     to 1e-8 Ha, and flagged 'fold_root_crossed'; check_roots finds it off the
     missed roots and nothing off the found ones; a cold solve does not flag.
  2. ADC2State (water aug-cc-pVDZ, frozen 1, aug-cc-pvdz-ri, gas singlet, 3 roots):
     of a cold call and 2 * EE_FOLD_FULL_CHECK_EVERY warm ones, the warm calls
     5 and 10 run full_check and no other; confirm(mol) after a seeded-only call
     runs check_roots once and returns no flag (water has no crossing), after a
     full call none, and at another geometry it is refused.
  3. the walk (properties.optimize, Cartesian and geomeTRIC): a surface whose
     confirm returns a flag at the converged geometry ends not converged, its
     status naming the flag, the flag in walk_flags, and no refreeze; without a
     flag, or without confirm, the walk converges as before.

Run: python -m pytest -q tests/test_ee_adc_fold_full_check.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

import src.properties.optimize as optimize_module
from src.Base.numerics.constants import EE_FOLD_FULL_CHECK_EVERY
from src.Base.numerics.numerics import Numerics
from src.gradients.adc import ee_state_model
from src.gradients.state_models import HFState, MP2State
from src.SingleReference.ADC.eeADC import ee_fold
from src.properties.optimize import relax
from tests.adc_common import ATOMS, NF, ROOT_CONV_TOL, make_scf_loop, molecule
from tests.test_ee_adc_fold_solve_speed import crossing

KW = dict(spin='singlet', dense_limit=0, solver='block', tol_omega=1e-10)


@pytest.mark.parametrize('nroots', [1, 3])
def test_full_check_finds_a_state_the_model_misses(nroots):
    p0, w0 = crossing('coupled', 0.6, True)
    r0 = ee_fold.solve_folded(p0, nroots, **KW)
    p1, w1 = crossing('coupled', 2.2, True)
    assert w1[0] < w0[0] - 0.1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        miss = ee_fold.solve_folded(p1, nroots, start=np.asarray(r0.y), **KW)
        full = ee_fold.solve_folded(p1, nroots, start=np.asarray(r0.y),
                                    full_check=True, **KW)
        cold = ee_fold.solve_folded(p1, nroots, **KW)
    # the seeded search alone misses a state spread over the block
    assert miss.omega[0] - w1[0] > 0.1 and not miss.flags
    assert np.max(np.abs(full.omega[:nroots] - w1[:nroots])) < 1e-8
    assert full.converged.all()
    assert [f['flag'] for f in full.flags] == ['fold_root_crossed']
    assert not cold.flags
    mu, top = ee_fold.check_roots(p1, 'singlet', miss.omega[:nroots],
                                  miss.y[:, :nroots], 1e-10, 1e-6)
    assert mu <= top
    mu, top = ee_fold.check_roots(p1, 'singlet', full.omega[:nroots],
                                  full.y[:, :nroots], 1e-10, 1e-6)
    assert mu > top


def test_state_model_times_the_full_check_and_confirms():
    mol = molecule(ATOMS, basis='aug-cc-pvdz')
    state = ee_state_model.ADC2State(
        MP2State(HFState(mol, make_scf_loop), NF, 'aug-cc-pvdz-ri',
                 numerics=Numerics(adc={'conv_tol': ROOT_CONV_TOL})), 'singlet',
        root=0, nroots=3)
    h, e = np.zeros((mol.nao, mol.nao)), 0.0
    asked, solve, checks, check = [], ee_state_model.solve_ee_adc, [], ee_fold.check_roots

    def solve_(*a, **k):
        asked.append((k['start'] is not None, k['full_check']))
        return solve(*a, **k)

    def check_(*a, **k):
        checks.append(1)
        return check(*a, **k)
    ee_state_model.solve_ee_adc, ee_fold.check_roots = solve_, check_
    try:
        n = 2 * EE_FOLD_FULL_CHECK_EVERY
        confirmed = []
        for call in range(n + 1):
            state(h, e)
            if call in (1, EE_FOLD_FULL_CHECK_EVERY):
                confirmed.append((state.confirm(mol), len(checks)))
        with pytest.raises(ValueError, match='another geometry'):
            state.confirm(molecule(ATOMS, shift=np.full((len(ATOMS), 3), 0.01),
                                   basis='aug-cc-pvdz'))
    finally:
        ee_state_model.solve_ee_adc, ee_fold.check_roots = solve, check
    assert asked[0] == (False, False)
    full = [i for i, (warm, f) in enumerate(asked) if f]
    assert all(warm for warm, _ in asked[1:])
    assert full == [EE_FOLD_FULL_CHECK_EVERY, 2 * EE_FOLD_FULL_CHECK_EVERY], asked
    # after warm call 1 (seeded only) confirm checks once; after warm call 5
    # (full) it checks nothing more
    assert confirmed == [([], 1), ([], 1)], confirmed


class Bond:
    """E = k (r - r_e)^2 / 2 in the bond length; confirm returns `flags` at
    the geometry it is asked about."""

    def __init__(self, mol, flags=None):
        self.mol0, self.flags, self.asked = mol, flags, []

    def total_gradient(self, mol=None, mf=None):
        here = mol.atom_coords()
        d = here[1] - here[0]
        r = float(np.linalg.norm(d))
        grad = np.zeros_like(here)
        grad[1] = 0.4 * (r - 2.2) * d / r
        grad[0] = -grad[1]
        return grad, 0.2 * (r - 2.2) ** 2, {}

    def refreeze(self, mol):
        return self

    def confirm(self, mol):
        self.asked.append(mol.atom_coords())
        return [] if self.flags is None else list(self.flags)


class Plain(Bond):
    confirm = None


@pytest.mark.parametrize('geometric', [False, True])
def test_walk_reports_an_unconfirmed_minimum_as_not_converged(geometric, monkeypatch):
    if geometric:
        pytest.importorskip('geometric')
    monkeypatch.setattr(optimize_module, 'HAVE_GEOMETRIC', geometric)
    n2 = gto.M(atom='N 0 0 0; N 0 0 2.0', unit='Bohr', basis='sto-3g', verbose=0)
    crossed = [{'flag': 'fold_root_crossed', 'message': 'planted'}]
    for surface, ok in ((Bond(n2), True), (Plain(n2), True), (Bond(n2, crossed), False)):
        mol, info = relax(surface, n2, engine='auto', verbose=False, nudge=0,
                          refreeze=1)
        assert info['converged'] is ok, (type(surface).__name__, info['status'])
        if surface.confirm is not None:
            # asked once, at the geometry the walk returns
            assert len(surface.asked) >= 1
            assert np.allclose(surface.asked[-1], mol.atom_coords())
        if not ok:
            assert 'fold_root_crossed' in info['status']
            assert crossed[0] in info['walk_flags']
            assert info['refreeze'] == 'not measured: the first pass did not converge'
