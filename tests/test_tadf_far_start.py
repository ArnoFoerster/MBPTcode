"""A far start of the S1 walk builds its own chain AT the start.

The start is its own reference: the relaxed declaration on its own mean field
there, with the window, the adaptive selection and the frozen conventions
resolved there by one evaluation, and no orbital of R0's chain matched to the
start (R0's conventions refrozen at the start by orbital identity cannot carry a
far start: an orbital can leave R0's span there). The s1 record stamps that reference, the chain's record
(s1_start_chain) is kept, and a T1 walk in a later launch is rebuilt from it,
not from R0's.

The walk gate's gas-phase production surface (formaldehyde/cc-pVDZ) with the
default adaptive set; the walk itself and the property block are stubbed out.
"""
from dataclasses import replace

import numpy as np
import pytest
from pyscf import gto

from src.Base.constants import BOHR_TO_ANGSTROM
from src.Base.record_io import record_hash
from src.gradients.conventions import FrozenConventions
from src.gradients.excited_state import ExcitedStateChain
from src.photophysics import stages
from src.photophysics.declaration import FarStart
from src.photophysics.records import label
from src.properties.optimize import nudge_vector
from src.properties.surface import driven_chain
from tests.photophysics_cases import (FORMALDEHYDE, Stop, declaration, refusing,
                                      run_at_r0)

#: Bohr: how far from R0 the start lies, and from the start the minimum
START_MOVE = 0.1
MINIMUM_MOVE = 0.02


refused = refusing('R0 was evaluated, or its chain rebuilt, for a walk from '
                   '--far-start')


def load(tmp_path, monkeypatch, far):
    """A Run walking from the far start `far`, after s0 at R0."""
    decl = declaration()
    decl = declaration(policies=replace(decl.policies, far_start=far))
    run = run_at_r0(tmp_path, decl)
    monkeypatch.setattr(run, 'reference', refused)
    monkeypatch.setattr(run, 'rebuilt_at', refused)
    return run


def far_start(mol, coords):
    """The `FarStart` at `coords` (Bohr) in `mol`'s atom order."""
    rows = [f'{mol.atom_pure_symbol(ia)} {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}'
            for ia, c in enumerate(np.asarray(coords) * BOHR_TO_ANGSTROM)]
    return FarStart('; '.join(rows))


def test_a_far_start_builds_its_own_chain(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    r0_mol = gto.M(atom=FORMALDEHYDE, basis='cc-pvdz', verbose=0)
    r0 = r0_mol.atom_coords()
    far = far_start(r0_mol, r0 + nudge_vector(r0, START_MOVE, seed=11))
    run = load(out, monkeypatch, far)
    assert np.array_equal(run.mol0.atom_coords(), r0)
    # the start as the run reads it off the declaration
    start = run.start_molecule()

    matched = []
    at = FrozenConventions.at

    def spy(self, mol, mf):
        if self.key_mol is not None and self.anchored:
            matched.append(np.array(self.key_mol.atom_coords()))
        return at(self, mol, mf)
    monkeypatch.setattr(FrozenConventions, 'at', spy)
    walked = {}

    def walk(surface, mol, stage=None, hess_init=None, resume=None):
        chain = driven_chain(surface)
        walked.update(chain=chain, start=mol)
        minimum = run.molecule_at(mol.atom_coords() + nudge_vector(
            mol.atom_coords(), MINIMUM_MOVE, seed=12))
        info = {'converged': True, 'cycles': 1, 'status': 'ok',
                'opt_grad_max': 0.0, 'energy': 0.0, 'history': [],
                'refreeze': 'measured', 'refreeze_drift_meV': 0.0,
                'refreeze_converged': True}
        return minimum, info, 0, chain
    monkeypatch.setattr(run, 'walk', walk)

    def props(ev, s_key, t_key, n_states):
        return {'coords_bohr': ev.mol.atom_coords(),
                'energy': {label(s_key): float(ev.energy[s_key])}}
    monkeypatch.setattr(stages, 'geometry_properties', props)
    monkeypatch.setattr(stages, 'report_properties', lambda *a: None)
    stages.s1_minimum(run)

    chain = walked['chain']
    assert np.array_equal(walked['start'].atom_coords(), start.atom_coords())
    # built at the start: its reference, its keys and a fresh selection
    assert np.array_equal(chain.mol0.atom_coords(), start.atom_coords())
    assert chain.resolved.qp_partition is not None
    assert chain.resolved.qp_floor is None
    # no orbital of R0's chain was matched anywhere
    assert not any(np.array_equal(k, r0) for k in matched)

    s1 = run.read('s1')
    assert run.stamp('s1')['far_start'] == far.sha256()
    assert s1['chain_reference'] == {'built_at': 'start',
                                     'sha256': far.sha256(),
                                     'record': 's1_start_chain'}
    record = run.read('s1_start_chain')
    assert np.array_equal(record['start_bohr'], start.atom_coords())

    # a later launch walks T1: rebuilt from the start's chain, not R0's
    run = load(out, monkeypatch, far)
    seen = {}

    def refreeze(self, mol, factorization=None, qp_floor=None):
        seen.update(chain=self, mol=mol, floor=qp_floor)
        raise Stop
    monkeypatch.setattr(ExcitedStateChain, 'refreeze', refreeze)
    with pytest.raises(Stop):
        stages.t1_minimum(run)
    assert np.array_equal(seen['chain'].mol0.atom_coords(),
                          start.atom_coords())
    assert (record_hash(seen['chain'].to_record())
            == record_hash(record['surface']))
    assert np.array_equal(seen['mol'].atom_coords(), s1['props']['coords_bohr'])
    assert seen['floor'] == s1['qp_partition']['explicit']
