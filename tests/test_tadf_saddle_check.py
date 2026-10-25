"""The saddle check and the nudged walks of the stages, on the toy double
well.

`Run.saddle_checked` differences the walked state's force into a Hessian at
the minimum and, at a saddle, leaves it along the negative mode and walks
again (`escape_saddle`). Driven here with the run's own `walk` (refreeze
passes, the production nudge) on the planar saddle of `test_walk_nudge`'s
surface.
"""
import numpy as np
import pytest

from src.properties.optimize import derived_walk
from src.Base.numerics.constants import HESSIAN_FD_ASYMMETRY_TOL, START_NUDGE_BOHR
from src.properties.hessian import HessianAsymmetric
from src.photophysics.records import walk_fields
from src.properties.excitations import WalkNotRelaxed, saddle_verdict
from src.properties import excitations
from src.properties.optimize import optimize
from tests.photophysics_cases import (PLANAR, DoubleWell, molecule,
                                      pyramidalization, run_of)


class Walked(DoubleWell):
    """The double well with the root a walk followed."""
    state = 0


class Noisy(Walked):
    """Its forces with seeded noise of `eps` Ha/Bohr on every element: a
    finite-difference Hessian of them is asymmetric."""

    def __init__(self, mol, eps):
        super().__init__(mol)
        self.rng, self.eps = np.random.default_rng(7), eps

    def total_gradient(self, mol=None, mf=None):
        g, e, info = super().total_gradient(mol, mf)
        return g + self.eps * self.rng.standard_normal(g.shape), e, info


def test_the_nudge_is_declared_and_stamped(tmp_path):
    run = run_of(tmp_path, '--engine', 'auto', '--nudge',
                 str(START_NUDGE_BOHR))
    assert run.declaration.policies.nudge_bohr == START_NUDGE_BOHR
    assert not run.declaration.handover
    assert run.stamp('s1')['nudge_bohr'] == START_NUDGE_BOHR
    assert run.stamp('t1')['nudge_bohr'] == START_NUDGE_BOHR
    off = run_of(tmp_path / 'off', '--engine', 'auto', '--nudge', '0')
    assert off.declaration.handover, ('an exact start takes the first point '
                                      'again')


def test_a_saddle_minimum_is_left_and_recorded(tmp_path):
    x = PLANAR.copy()
    x[1, 0] += 0.15
    surface = Walked(molecule(x))
    saddle, info = optimize(surface, surface.mol0,
                            numerics=derived_walk(surface, 'a capped test walk', max_cycle=200),
                            verbose=False, nudge=0)
    assert abs(pyramidalization(saddle.atom_coords())) < 1e-10
    run = run_of(tmp_path, '--engine', 'auto', '--saddle-check', '1')
    mol, again, root, final, record = run.saddle_checked(
        's1', saddle, info, 0, surface, 0.0)
    assert record['status'] == 'escaped'
    assert record['first']['lowest_curvature'] < 0
    assert record['saddle_walk']['converged']
    assert abs(pyramidalization(mol.atom_coords())) > 0.3
    assert again['start_nudge']['amplitude_bohr'] == START_NUDGE_BOHR
    assert root == 0 and final is surface
    assert record['hessian_asymmetry'] < 1e-6


def test_without_the_policy_the_walk_is_kept(tmp_path):
    run = run_of(tmp_path, '--engine', 'auto')
    out = run.saddle_checked('s1', 'mol', {'k': 1}, 2, 'surface', 0.0)
    assert out == ('mol', {'k': 1}, 2, 'surface', None)


def test_a_second_saddle_is_flagged_and_walked_on(tmp_path, monkeypatch):
    x = PLANAR.copy()
    x[1, 0] += 0.15
    surface = Walked(molecule(x))
    saddle, info = optimize(surface, surface.mol0,
                            numerics=derived_walk(surface, 'a capped test walk', max_cycle=200),
                            verbose=False, nudge=0)
    real = excitations.surface_hessian
    calls = []

    def planted(s, m, **kw):
        calls.append(m)
        # the first Hessian is the saddle's own; the second is planted
        return real(s, m, **kw) if len(calls) == 1 else -1.0 * real(s, m)
    monkeypatch.setattr(excitations, 'surface_hessian', planted)
    run = run_of(tmp_path, '--engine', 'auto', '--saddle-check', '1')
    # a numerical event: warned about, not refused
    with pytest.warns(WalkNotRelaxed, match='second saddle'):
        mol, again, root, final, record = run.saddle_checked(
            't1', saddle, info, 0, surface, 0.0)
    assert record['status'] == 'second saddle: stopped'
    assert again['numerical_flags'] == {
        'saddle_check': 'second saddle: stopped'}
    assert walk_fields(again)['numerical_flags'] == again['numerical_flags']
    assert root == 0 and final is surface


def test_an_asymmetric_hessian_is_flagged_on_the_walk_the_stage_keeps(tmp_path):
    """A Hessian above HESSIAN_FD_ASYMMETRY_TOL is warned about, kept, and
    named 'hessian_asymmetric' in the check's record, its verdict and the
    stage's numerical_flags, also where the check finds a minimum; a clean
    one carries no such flag."""
    x = PLANAR.copy()
    x[0, 2] += 0.3
    surface = Walked(molecule(x))
    minimum, info = optimize(surface, surface.mol0,
                             numerics=derived_walk(surface, 'a capped test walk', max_cycle=200),
                             verbose=False, nudge=0)
    run = run_of(tmp_path, '--engine', 'auto', '--saddle-check', '1')
    _, kept, _, _, record = run.saddle_checked('s1', minimum, info, 0, surface, 0.0)
    assert record['status'] == 'minimum' and record['hessian_asymmetric'] is None
    assert 'hessian_asymmetric' not in kept['numerical_flags']
    noisy = Noisy(minimum, 1e-5)
    run = run_of(tmp_path / 'noisy', '--engine', 'auto', '--saddle-check', '1')
    with pytest.warns(HessianAsymmetric):
        _, kept, _, _, record = run.saddle_checked('s1', minimum, info, 0, noisy, 0.0)
    assert record['status'] == 'minimum'
    rel = record['hessian_asymmetric']['first']
    assert rel == record['hessian_asymmetry_rel'] > 10 * HESSIAN_FD_ASYMMETRY_TOL
    assert saddle_verdict('s1', record) == {'hessian_asymmetric': {'first': rel}}
    assert kept['numerical_flags']['hessian_asymmetric'] == {'first': rel}


def test_an_unrelaxed_walk_is_not_saddle_checked(tmp_path):
    run = run_of(tmp_path, '--engine', 'auto', '--saddle-check', '1')
    info = {'numerical_flags': {'walk_not_converged': 'max cycles'}}
    mol, again, root, final, record = run.saddle_checked(
        's1', 'mol', info, 2, 'surface', 0.0)
    assert (mol, again, root, final) == ('mol', info, 2, 'surface')
    assert record == {'status': 'not run: the walk did not relax'}
