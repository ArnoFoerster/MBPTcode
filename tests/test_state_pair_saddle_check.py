"""`saddle_checked_state` at an excited minimum and the record it leaves.

`emission_block` (so `calc_adiabatic_excitation` with `saddle_check=True`)
differences the walked state's force into a Hessian at its minimum and, at a
saddle, leaves it along the negative mode and walks again with the first
walk's settings (`escape_saddle`). It is driven here on the toy double well
of `test_walk_nudge`, whose planar point is a symmetric saddle: a planted
saddle is left and recorded, a true minimum is kept as it is, and a second
saddle is refused.

Every check asserts: pytest discards a returned verdict and passes on False.
"""
from types import SimpleNamespace

import pytest

import src.properties.excitations as excitations
from src.properties.excitations import saddle_checked_state
from src.properties.vibronic import relax_state
from tests.test_walk_nudge import (PLANAR, DoubleWell, molecule,
                                   pyramidalization)

WALK = {'engine': 'cartesian', 'max_cycle': 200, 'verbose': False,
        'refreeze': 1}


class Relaxable(DoubleWell):
    """The double well with what `relax_state` reads off a surface: its own
    mean field (a stand-in carrying the total as e_tot) and its energies
    (total, excitation)."""

    def mean_field(self, mol):
        return mol, SimpleNamespace(e_tot=self.total_energy(mol))

    def energy(self, mol=None, mf=None):
        return self.total_energy(mol), self.excitation(mol)


@pytest.fixture
def at_saddle():
    """A converged walk that stopped at the planar saddle (no nudge)."""
    x = PLANAR.copy()
    x[1, 0] += 0.15
    surface = Relaxable(molecule(x))
    relaxed = relax_state(surface, surface.mol0, nudge=0, **WALK)
    assert relaxed['info']['converged']
    assert abs(pyramidalization(relaxed['mol'].atom_coords())) < 1e-10
    return relaxed


def test_a_planted_saddle_is_left_and_recorded(at_saddle):
    state, record = saddle_checked_state(at_saddle, **WALK)
    assert record['status'] == 'escaped'
    assert record['first']['lowest_curvature'] < 0
    assert record['second']['lowest_curvature'] > 0
    assert record['energy_drop'] > 0
    assert record['saddle_walk']['converged']
    assert record['hessian_asymmetry'] < 1e-6
    # the state is the second walk's, at a pyramidal minimum, nudged as the
    # first walk's settings say (the default START_NUDGE_BOHR)
    assert state is not at_saddle
    assert state['info']['converged']
    assert abs(pyramidalization(state['mol'].atom_coords())) > 0.3
    assert state['e_total'] < at_saddle['e_total']
    assert state['info']['start_nudge']['amplitude_bohr'] > 0


def test_a_true_minimum_is_kept():
    x = PLANAR.copy()
    x[0, 2] = 0.4
    surface = Relaxable(molecule(x))
    relaxed = relax_state(surface, surface.mol0, nudge=0, **WALK)
    state, record = saddle_checked_state(relaxed, **WALK)
    assert state is relaxed
    assert record['status'] == 'minimum'
    assert record['first']['lowest_curvature'] > 0
    assert record['second'] is None and 'saddle_walk' not in record


def test_a_second_saddle_is_refused(at_saddle, monkeypatch):
    real = excitations.surface_hessian
    calls = []

    def planted(surface, mol, **kw):
        calls.append(mol)
        # the first Hessian is the saddle's own; the second is planted
        h = real(surface, mol, **kw)
        return h if len(calls) == 1 else -1.0 * h
    monkeypatch.setattr(excitations, 'surface_hessian', planted)
    with pytest.raises(RuntimeError, match='second saddle'):
        saddle_checked_state(at_saddle, **WALK)

