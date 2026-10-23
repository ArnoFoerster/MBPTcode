"""A refrozen relaxation's record describes the walk whose geometry it returns.

The refreeze (`refreeze_pass`) updates the energy, Omega, Hessian, geometry,
`converged`, `status`, `cycles` and `opt_grad_max` all to the LAST walk's:
leaving the flags at the FIRST walk's would let a first walk that converged
followed by a refrozen pass that ran out of cycles come back
`converged: True, status: 'ok'` standing on the second walk's unconverged
geometry, and `relaxation_fields` would carry that into every record. These
tests pin:

  - the record's walk flags are the last walk's, the first walk's kept under
    `first_walk_*`, and `history` stays the first walk's (it starts at the
    input geometry, where `driving_force` reads it);
  - `refreeze_converged` stays the refreeze's own verdict, on its drift
    alone: the capped walk below drifts 0.54 meV, within REFREEZE_TOL_MEV
    (3 meV; a drift above it is flagged refreeze_not_converged, see
    test_optimize_refreeze), so it is True while `converged` is False -- the
    walk, not the conventions, failed;
  - `relaxation_fields` reports the last walk.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import REFREEZE_TOL_MEV
from src.properties.excitations import relaxation_fields
from src.properties.optimize import refreeze_pass

class Rebuilt:
    """A surface whose refreeze hands back another one; nothing is evaluated."""

    def refreeze(self, mol):
        return Rebuilt()


def walk_record(converged, status, cycles, residual, energy, n3,
                start=None):
    """A walk's record; `start` is the energy of its first evaluation."""
    return {'converged': converged, 'status': status, 'cycles': cycles,
            'history': ([{'e': energy if start is None else start}]
                        + [{'e': energy}] * (cycles - 1)), 'energy': energy,
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
    # the rebuilt surface at the first minimum: 0.54 meV above the old one,
    # within REFREEZE_TOL_MEV
    capped = walk_record(False, 'max_cycle (50) reached without convergence',
                         50, 3e-3, -76.002, n3, start=-76.0 + 2e-5)
    moved = water.copy()
    moved.set_geom_(water.atom_coords() + 0.02, unit='Bohr')

    def walk(surface, mol, hessian):
        return moved, dict(capped)

    mol, info = refreeze_pass(walk, Rebuilt(), water, dict(first),
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
    assert abs(info['refreeze_drift_meV']) < REFREEZE_TOL_MEV
    assert info['refreeze_converged'] is True
    assert info['refreeze_drift'] == pytest.approx(2e-5, abs=1e-12)


def test_relaxation_fields_report_the_last_walk(water):
    _, info, _, capped, _ = refrozen(water)
    fields = relaxation_fields(info)
    assert fields['converged'] is False
    assert fields['status'] == capped['status']
    assert fields['cycles'] == capped['cycles']
    assert fields['opt_grad_max'] == capped['opt_grad_max']
    assert fields['refreeze_converged'] is True
