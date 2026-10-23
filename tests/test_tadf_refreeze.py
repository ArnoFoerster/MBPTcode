"""The S1 and T1 walks refreeze at their minimum.

A walk on the conventions frozen at R0 alone (`relax(..., refreeze=0)`) stands
every minimum on a surface built at another geometry. `Run.walk` hands
`relax` the declared refreeze (policies.refreeze True -> 1, False -> 0), takes
the root the walk followed off the surface the LAST walk ran on, and the
record keeps the drift of the conventions at the minimum in meV and whether
it is within REFREEZE_TOL_MEV; a refused drift, like a last walk that did
not converge, is warned about and flagged (refuse
inputs, warn numerics).
"""
import types

import pytest
from pyscf import gto

from src.Base.numerics.constants import HARTREE_TO_MEV, REFREEZE_TOL_MEV
from src.photophysics import stages
from src.photophysics.records import refreeze_summary, walk_fields
from src.properties.excitations import WalkNotRelaxed, relaxation_flags
from tests.photophysics_cases import run_of


def surface(state, follow):
    """A walked surface as `walk` reads it: a state and a follow log."""
    return types.SimpleNamespace(state=state, follow_log=list(follow))


def refrozen_info(final, drift_meV=0.04, refreeze='measured'):
    """A refrozen walk's record as `relax` returns it."""
    return {'converged': True, 'cycles': 7, 'history': [{}] * 7,
            'energy': -1.0, 'omega': 0.1, 'opt_grad_max': 1e-5,
            'status': 'ok', 'hessian': None, 'refreeze': refreeze,
            'refreeze_drift': drift_meV / HARTREE_TO_MEV,
            'refreeze_drift_meV': drift_meV,
            'refreeze_converged': abs(drift_meV) <= REFREEZE_TOL_MEV,
            'refreeze_shift': 2e-2, 'refreeze_rigid_shift': 3e-2,
            'refreeze_denergy': -3e-5,
            'refreeze_qp_growth': {'carried': [5, 6, 7], 'added': [4],
                                   'not_explicit': []},
            'first_walk_de_meV': 0.05,
            'refreeze_info': {'converged': True, 'cycles': 3,
                              'opt_grad_max': 2e-5, 'status': 'ok'},
            'final_surface': final}


def fake_relax(seen, final):
    """`relax` that records its keywords and returns a refrozen walk."""
    def relax(start, mol, **kw):
        seen.update(kw)
        start.follow_log.extend({'index': start.state} for _ in range(6))
        return mol, refrozen_info(final)
    return relax


@pytest.mark.parametrize('extra,passes', [((), 1), (('--refreeze', '0'), 0)])
def test_the_walks_refreeze_and_record_the_drift(tmp_path, monkeypatch,
                                                 extra, passes):
    seen = {}
    final = surface(0, [{'index': 1}])
    monkeypatch.setattr(stages, 'relax', fake_relax(seen, final))
    run = run_of(tmp_path, *extra)
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    start = surface(0, [{'index': 0}])
    _, info, root, last = run.walk(start, mol)
    assert seen['refreeze'] == passes
    assert seen['nudge'] == run.declaration.policies.nudge_bohr
    assert last is final, 'the walk hands back the surface it ended on'
    assert root == 1, 'the root is the last walk\'s'
    assert info['first_point_replayed']
    fields = walk_fields(info)
    assert fields['refreeze_drift_meV'] == 0.04
    assert fields['refreeze_converged'] is True
    assert fields['refreeze_denergy_meV'] == -3e-5 * HARTREE_TO_MEV
    assert fields['last_walk']['cycles'] == 3
    # grow-only adaptive set: the rebuild names the states it added
    assert fields['refreeze_qp_growth']['added'] == [4]
    assert 'drift at the minimum +0.0400 meV' in refreeze_summary(fields)
    assert 'final_surface' not in info


def test_a_last_walk_that_failed_is_flagged():
    info = {'converged': True, 'refreeze': 'measured',
            'refreeze_converged': False, 'refreeze_drift_meV': 0.3,
            'refreeze_info': {'converged': False, 'status': 'trust radius'}}
    with pytest.warns(WalkNotRelaxed, match='trust radius'):
        flags = relaxation_flags('s1', info)
    assert flags == {'walk_not_converged': 'trust radius'}
    info['refreeze_info']['converged'] = True
    assert relaxation_flags('s1', info) == {}


def test_a_refused_refreeze_flags_the_walk(tmp_path, monkeypatch):
    """`relax` returning a refreeze that did not converge comes back from
    `walk` as its record, and `relaxation_flags` warns and flags it."""
    final = surface(0, [{'index': 0}])

    def refusing(start, mol, **kw):
        return mol, refrozen_info(final, 2.5, 'not converged')

    monkeypatch.setattr(stages, 'relax', refusing)
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    _, info, _, last = run_of(tmp_path).walk(surface(0, []), mol)
    assert last is final and info['refreeze'] == 'not converged'
    with pytest.warns(WalkNotRelaxed, match='REFREEZE_TOL_MEV'):
        flags = relaxation_flags('s1', info)
    assert flags == {'refreeze_not_converged': 2.5}
    assert 'FLAGGED' in refreeze_summary(walk_fields(info))
