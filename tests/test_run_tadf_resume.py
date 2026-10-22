"""The S1 stage resumes from what an earlier launch left in the store.

With s1_walked in the store (the converged walk and the record of the
surface it ended on) the stage neither evaluates R0 nor walks: it rebuilds
the surface from its record and goes on at the saddle check, at the walked
minimum to the bit, with the Hessian's force cache in hess_s1/. With only
s1_progress (a walk's last accepted step) it rebuilds that surface and
hands the walk the state to resume from, again without evaluating R0. The
walk gate's gas-phase production surface (formaldehyde/cc-pVDZ); everything
after the point under test is stubbed out.
"""
import pathlib

import numpy as np
import pytest

from src.Base.record_io import record_hash
from src.photophysics import stages
from src.Base.geometry import molecule_at
from src.properties.optimize import nudge_vector, surface_record
from src.properties.surfaces import reference_mean_field
from tests.photophysics_cases import Stop, declaration, refusing, run_at_r0

#: Bohr: how far from R0 the stand-in minimum lies.
MOVE = 0.02


def stage_run(tmp_path):
    """A Run of the S1 stage with the saddle check, R0 at the input
    geometry."""
    decl = declaration('--saddle-check', '1', '--relaxed-solvation',
                       'nonequilibrium', qp_states='admitted')
    return run_at_r0(tmp_path, decl)


def frozen_surface(run):
    """The S1 surface at R0, its conventions frozen by one evaluation a
    little away from R0, and that geometry."""
    mol0 = run.mol0
    mf0 = reference_mean_field(mol0, run.factory, run.relaxed_spec.environment)
    surface = run.excited_at(run.relaxed_spec, mol0, mf0)
    at = molecule_at(mol0, mol0.atom_coords()
                     + nudge_vector(mol0.atom_coords(), MOVE))
    e = surface.total_energy(at)
    return surface, at, e


refused = refusing('the resumed stage evaluated R0 or walked again')


def test_a_walked_stage_resumes_at_the_saddle_check(tmp_path, monkeypatch):
    run = stage_run(tmp_path)
    surface, minimum, e = frozen_surface(run)
    info = {'converged': True, 'cycles': 3, 'energy': e, 'omega': None,
            'history': [{'cycle': 0, 'e': e}], 'refreeze': 'measured',
            'refreeze_drift_meV': 0.0, 'refreeze_converged': True,
            'hessian': np.eye(3 * minimum.natm), 'status': 'ok',
            'opt_grad_max': 0.0}
    run.write_walked('s1', minimum, info, 0, surface)
    monkeypatch.setattr(run, 'reference', refused)
    monkeypatch.setattr(run, 'walk', refused)
    seen = {}

    def saddle_check(final, mol, walk, verbose=False, cache_dir=None):
        seen.update(final=final, mol=mol, cache_dir=cache_dir)
        raise Stop
    monkeypatch.setattr(stages, 'saddle_check', saddle_check)
    with pytest.raises(Stop):
        stages.s1_minimum(run)
    assert np.array_equal(seen['mol'].atom_coords(), minimum.atom_coords())
    assert seen['final'] is not surface
    assert (record_hash(surface_record(seen['final']))
            == record_hash(surface_record(surface)))
    assert pathlib.Path(seen['cache_dir']) == run.store.dir / 'hess_s1'


def test_a_walk_in_progress_resumes_without_r0(tmp_path, monkeypatch):
    run = stage_run(tmp_path)
    surface, at, e = frozen_surface(run)
    state = {'pass': 0, 'engine': 'internal', 'coords': at.atom_coords(),
             'energy': e, 'history': [{'cycle': 0, 'e': e}],
             'surface': surface_record(surface)}
    run.write('s1_progress', {'start_bohr': run.mol0.atom_coords(),
                              'state': state})
    monkeypatch.setattr(run, 'reference', refused)
    seen = {}

    def relax(surface, mol, resume=None, checkpoint=None, **kw):
        seen.update(surface=surface, mol=mol, resume=resume)
        raise Stop
    monkeypatch.setattr(stages, 'relax', relax)
    with pytest.raises(Stop):
        stages.s1_minimum(run)
    assert seen['resume']['pass'] == 0
    assert np.array_equal(seen['resume']['coords'], at.atom_coords())
    assert np.array_equal(seen['mol'].atom_coords(), run.mol0.atom_coords())
    assert (record_hash(surface_record(seen['surface']))
            == record_hash(state['surface']))
