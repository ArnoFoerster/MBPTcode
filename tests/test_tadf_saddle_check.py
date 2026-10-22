"""The saddle check and the nudged walks of the stages, on the toy double
well.

`Run.saddle_checked` differences the walked state's force into a Hessian at
the minimum and, at a saddle, leaves it along the negative mode and walks
again (`escape_saddle`). Driven here with the run's own `walk` (refreeze
passes, the production nudge) on the planar saddle of `test_walk_nudge`'s
surface.
"""
import pytest

from src.Base.constants import START_NUDGE_BOHR
from src.photophysics.records import WalkNotRelaxed, walk_fields
from src.properties import excitations
from src.properties.optimize import optimize
from tests.photophysics_cases import (PLANAR, DoubleWell, molecule,
                                      pyramidalization, run_of)


class Walked(DoubleWell):
    """The double well with the root a walk followed."""
    state = 0


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
    saddle, info = optimize(surface, surface.mol0, max_cycle=200,
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
    saddle, info = optimize(surface, surface.mol0, max_cycle=200,
                            verbose=False, nudge=0)
    real = excitations.surface_hessian
    calls = []

    def planted(s, m, **kw):
        calls.append(m)
        # the first Hessian is the saddle's own; the second is planted
        return real(s, m, **kw) if len(calls) == 1 else -1.0 * real(s, m)
    monkeypatch.setattr(excitations, 'surface_hessian', planted)
    run = run_of(tmp_path, '--engine', 'auto', '--saddle-check', '1')
    # Refuse inputs, warn numerics
    with pytest.warns(WalkNotRelaxed, match='second saddle'):
        mol, again, root, final, record = run.saddle_checked(
            't1', saddle, info, 0, surface, 0.0)
    assert record['status'] == 'second saddle: stopped'
    assert again['numerical_flags'] == {
        'saddle_check': 'second saddle: stopped'}
    assert walk_fields(again)['numerical_flags'] == again['numerical_flags']
    assert root == 0 and final is surface


def test_an_unrelaxed_walk_is_not_saddle_checked(tmp_path):
    run = run_of(tmp_path, '--engine', 'auto', '--saddle-check', '1')
    info = {'numerical_flags': {'walk_not_converged': 'max cycles'}}
    mol, again, root, final, record = run.saddle_checked(
        's1', 'mol', info, 2, 'surface', 0.0)
    assert (mol, again, root, final) == ('mol', info, 2, 'surface')
    assert record == {'status': 'not run: the walk did not relax'}
