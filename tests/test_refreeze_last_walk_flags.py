"""A refrozen relaxation's record describes the walk whose geometry it returns.

`refreeze_passes` takes the energy, Omega, Hessian and geometry from the last
walk, and so `converged`, `status`, `cycles` and `opt_grad_max` too: a first
walk that converged followed by a refrozen pass that ran out of cycles comes
back `converged: False`, standing on the second walk's unconverged geometry.
These tests pin:

  - the record's walk flags are the last walk's, the first walk's kept under
    `first_walk_*`, and `history` stays the first walk's (it starts at the
    input geometry, where `driving_force` reads it);
  - `refreeze_converged` stays the outer loop's own verdict;
  - `relaxation_fields` reports the last walk.
"""
import numpy as np
import pytest
from pyscf import gto

import src.gradients  # noqa: F401  cycle: src.properties imports src.gradients
from src.properties.excitations import relaxation_fields
from src.properties.optimize import refreeze_passes

class Rebuilt:
    """A surface whose refreeze hands back another one; nothing is evaluated."""

    def refreeze(self, mol):
        return Rebuilt()


def walk_record(converged, status, cycles, residual, energy, n3):
    return {'converged': converged, 'status': status, 'cycles': cycles,
            'history': [{'e': energy}] * cycles, 'energy': energy,
            'omega': 0.2, 'grad_max': residual, 'opt_grad_max': residual,
            'optimizer': 'internal', 'hessian': np.eye(n3)}


@pytest.fixture
def water():
    return gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59',
                 basis='sto-3g', verbose=0)


def refrozen(water):
    """A converged first walk and a refrozen pass that hits its cycle cap."""
    n3 = 3 * water.natm
    first = walk_record(True, 'ok', 7, 1e-4, -76.0, n3)
    capped = walk_record(False, 'max_cycle (50) reached without convergence',
                         50, 3e-3, -76.002, n3)
    moved = water.copy()
    moved.set_geom_(water.atom_coords() + 0.02, unit='Bohr')

    def walk(surface, mol, hessian):
        return moved, dict(capped)

    mol, info = refreeze_passes(walk, Rebuilt(), water, dict(first), 4,
                                verbose=False)
    return mol, info, first, capped, moved


def test_the_record_carries_the_last_walks_flags(water):
    mol, info, first, capped, moved = refrozen(water)
    assert mol is moved
    for key in ('converged', 'status', 'cycles', 'opt_grad_max', 'grad_max'):
        assert info[key] == capped[key], key
        assert info['first_walk_' + key] == first[key], key
    assert info['energy'] == capped['energy']
    assert info['history'] == first['history']
    assert info['refreeze_converged'] is False
    assert len(info['refreeze_jumps']) == 1


def test_relaxation_fields_report_the_last_walk(water):
    _, info, _, capped, _ = refrozen(water)
    fields = relaxation_fields(info)
    assert fields['converged'] is False
    assert fields['status'] == capped['status']
    assert fields['cycles'] == capped['cycles']
    assert fields['opt_grad_max'] == capped['opt_grad_max']
    assert fields['refreeze_converged'] is False
