"""The geomeTRIC walk reports a walk that did not converge instead of dying.

geomeTRIC raises `GeomOptNotConvergedError` when its iteration cap is reached.
`optimize_geometric` turns that into a record, so a refrozen pass that runs
out of iterations keeps the passes that converged ahead of it, and a walk's
`converged` is what the driver reported. These tests stand in for geomeTRIC's driver (`run_optimizer`) with a stub that
calls the walk's engine at chosen geometries and then returns or raises, so no
real optimization runs:

  - a walk that hits the cap returns its last geometry and trace with
    `converged` False and a status naming the cap; any other failure of the
    driver is reported the same way, with its message;
  - `refreeze_passes` stops cleanly on such a pass;
  - each refrozen pass starts from the Hessian the walk before it ended with
    (not the first walk's starting guess) and writes under its own prefix.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto

import src.gradients  # noqa: F401  cycle: src.properties imports src.gradients
from src.Base.constants import BOHR_TO_ANGSTROM
from src.properties.optimize import optimize_geometric

geometric_optimize = pytest.importorskip('geometric.optimize')
from geometric.errors import GeomOptNotConvergedError  # noqa: E402

#: Bohr: where the stub's last evaluation stands, off the start.
STEP = 0.01
#: The Hessian, Ha/Bohr^2 on the diagonal, the stub's converged walk ends with.
H_END = 0.7
#: Bohr: where the stub's converged walk ends, off its start. Not a rigid
#: motion -- a refreeze shift is measured on superposed geometries, so a
#: walk that only translated would not have moved.
MOVE = 0.05 + np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, -0.02]])


class Harmonic:
    """E = |R - R*|^2 / 2 about a point 0.05 Bohr from the start; refreezing
    returns a new surface with the same minimum."""

    def __init__(self, mol):
        self.mol0 = mol
        self.minimum = mol.atom_coords() + 0.05

    def total_gradient(self, mol=None, mf=None):
        d = (self.mol0 if mol is None else mol).atom_coords() - self.minimum
        return d, 0.5 * float((d ** 2).sum()), {}

    def total_energy(self, mol=None, mf=None):
        return self.total_gradient(mol)[1]

    def refreeze(self, mol):
        out = Harmonic(mol)
        out.minimum = self.minimum
        return out


@pytest.fixture
def water():
    return gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59',
                 basis='sto-3g', verbose=0)


def stub(monkeypatch, outcomes):
    """Replace geomeTRIC's driver by one that plays `outcomes` in order.

    Each outcome is 'converge' (evaluate the start and the minimum, write
    H_END where the Cartesian Hessian is asked for, return the minimum),
    'cap' (evaluate the start and a point STEP off it, then raise the
    iteration-cap error) or an exception instance raised after one
    evaluation. Returns the list of keyword dicts each call received."""
    calls = []

    def run_optimizer(**kw):
        calls.append(kw)
        outcome = outcomes[len(calls) - 1]
        engine = kw['customengine']
        start = np.asarray(engine.M.xyzs[0]) / BOHR_TO_ANGSTROM
        engine.calc_new(start.ravel(), None)
        if outcome == 'converge':
            minimum = start + MOVE
            engine.calc_new(minimum.ravel(), None)
            n3 = start.size
            np.savetxt(kw['write_cart_hess'], H_END * np.eye(n3))
            return SimpleNamespace(xyzs=[minimum * BOHR_TO_ANGSTROM])
        if outcome == 'cap':
            engine.calc_new((start + STEP).ravel(), None)
            raise GeomOptNotConvergedError(
                'Optimizer.optimizeGeometry() failed to converge.')
        raise outcome

    monkeypatch.setattr(geometric_optimize, 'run_optimizer', run_optimizer)
    return calls


def test_the_iteration_cap_is_a_status_not_an_exception(monkeypatch, water):
    stub(monkeypatch, ['cap'])
    mol_opt, info = optimize_geometric(Harmonic(water), water, maxiter=7,
                                       verbose=False)
    assert info['converged'] is False
    assert info['status'] == 'maxiter (7) reached without convergence'
    assert info['cycles'] == 2 and len(info['history']) == 2
    assert np.allclose(mol_opt.atom_coords(), water.atom_coords() + STEP,
                       atol=1e-10)
    assert info['energy'] == info['history'][-1]['e']
    assert info['refreeze_shift'] is None
    assert info['refreeze'] == 'not measured'


def test_any_other_driver_failure_is_reported(monkeypatch, water):
    stub(monkeypatch, [RuntimeError('Gradient contains nan')])
    mol_opt, info = optimize_geometric(Harmonic(water), water, verbose=False)
    assert info['converged'] is False
    assert info['status'] == 'RuntimeError: Gradient contains nan'
    assert np.allclose(mol_opt.atom_coords(), water.atom_coords(), atol=1e-10)


def test_refreeze_not_attempted_after_an_unconverged_first_walk(monkeypatch,
                                                                water):
    calls = stub(monkeypatch, ['cap'])
    _, info = optimize_geometric(Harmonic(water), water, refreeze=2,
                                 verbose=False)
    assert len(calls) == 1
    assert info['refreeze'] == 'not measured: the first pass did not converge'


def test_refreeze_stops_cleanly_on_a_pass_that_hits_the_cap(monkeypatch,
                                                            water):
    calls = stub(monkeypatch, ['converge', 'cap'])
    mol_opt, info = optimize_geometric(Harmonic(water), water, refreeze=3,
                                       verbose=False)
    assert len(calls) == 2
    assert info['refreeze'] == 'measured'
    assert len(info['refreeze_jumps']) == 1
    assert info['refreeze_converged'] is False
    assert info['refreeze_info']['converged'] is False
    assert 'without convergence' in info['refreeze_info']['status']
    # the geometry returned is the last walk's, where it stopped
    assert np.allclose(mol_opt.atom_coords(),
                       water.atom_coords() + MOVE + STEP, atol=1e-10)


def test_each_pass_starts_from_the_last_hessian_under_its_own_prefix(
        monkeypatch, water):
    calls = stub(monkeypatch, ['converge', 'converge', 'converge'])
    n3 = 3 * water.natm
    optimize_geometric(Harmonic(water), water, refreeze=2, verbose=False,
                       hess_init=0.3 * np.eye(n3))
    assert len(calls) == 3
    assert np.allclose(calls[0]['hess_data'], 0.3 * np.eye(n3))
    for call in calls[1:]:
        assert np.allclose(call['hess_data'], H_END * np.eye(n3))
    prefixes = [call['prefix'] for call in calls]
    hessians = [call['write_cart_hess'] for call in calls]
    assert len(set(prefixes)) == 3 and len(set(hessians)) == 3
    assert len({os.path.dirname(p) for p in prefixes}) == 1
