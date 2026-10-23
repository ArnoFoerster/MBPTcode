"""A walk killed after an accepted step and resumed from its checkpoint
takes the steps the uninterrupted walk took, in either pass.

`relax(..., checkpoint=f)` hands `f` the walk after its start and after
every accepted step (`walk_state`): geometry, energy, force, the BFGS
Hessian, the trust radius, the history, the rejected count and the record
of the surface, the conventions with the keys the accepted steps carried
and the followed state among them; in the walk on the refrozen surface also
the first walk's record and its minimum. `relax(..., resume=state)` goes on
from there on the surface rebuilt from that record. On the walk gate's
gas-phase production surface (formaldehyde/cc-pVDZ, the photophysics run's
factory and declaration, the Cartesian engine, one refreeze), with the convergence
thresholds ten times looser to keep the walk short:

  * a walk killed by its checkpoint callable after two accepted cycles and
    resumed from the state on disk reaches the uninterrupted walk's minimum
    and energy to 1e-10 Bohr / Ha (to the bit, in fact), and records how
    many evaluations it reused;
  * a walk resumed from a state of its refrozen pass does the same, with the
    refreeze drift the uninterrupted walk measured.

The same on a mean-field surface (water/STO-3G Hartree-Fock, which writes
no record: the walk goes on on the surface it is handed), and through
geomeTRIC, which restarts from the last accepted geometry and reaches the
same minimum to its convergence criteria.
"""

import numpy as np
import pytest
from pyscf import gto, scf

from tests.blas_bar import assert_same
from src.Base.numerics.constants import GEOM_OPT_CONV
from src.Base.utils.record_io import read_record, write_record
from src.Base.geometry.body_frame import aligned_displacement
from src.properties.optimize import relax
from src.gradients.mean_field_surface import MeanFieldSurface
from src.gradients.gwbse.surface_factory import reference_mean_field
from tests.photophysics_cases import run_of

#: The walk's thresholds: GEOM_OPT_CONV ten times looser.
LOOSE = {k: 10.0 * v for k, v in GEOM_OPT_CONV.items() if k != 'grad_max'}
#: Accepted cycles before the kill (the start is the first checkpoint).
KILLED_AFTER = 2
#: Bohr / Ha: how close the resumed walk must land.
SAME = 1e-10
#: Bohr / Ha: how close a restarted geomeTRIC walk must land, within the
#: reach of the GAU criteria both walks converge to.
GEOMETRIC_SAME = (1e-2, 1e-6)


class Killed(Exception):
    """The checkpoint callable's stand-in for a job killed by the queue."""


def production(tmp_path):
    """(the photophysics Run, a factory of the S1 surface at R0 it would
    walk) in the gas phase, the walk gate's configuration."""
    run = run_of(tmp_path / 'store', qp_states='admitted')
    mf0 = reference_mean_field(run.mol_in, run.factory,
                               run.relaxed_spec.environment)
    return run, lambda: run.excited_at(run.relaxed_spec, run.mol_in, mf0)


def kept(directory, kill_after=None):
    """A checkpoint callable writing each state to `directory`/<n>.json,
    raising `Killed` after `kill_after` of them; and the list of paths."""
    paths = []

    def checkpoint(state):
        path = directory / f'{len(paths)}.json'
        write_record(path, state)
        paths.append(path)
        if kill_after is not None and len(paths) >= kill_after:
            raise Killed(f'killed after {len(paths)} checkpoints')
    directory.mkdir()
    return checkpoint, paths


def walk(surface, mol, **kw):
    return relax(surface, mol, engine='cartesian', refreeze=1, conv=LOOSE,
                 verbose=False, **kw)


def same_walk(mol, info, mol_ref, info_ref):
    """Assert two walks end at one minimum with one energy and drift."""
    moved = float(np.abs(mol.atom_coords() - mol_ref.atom_coords()).max())
    de = abs(info['energy'] - info_ref['energy'])
    print(f'resumed in pass {info["resume"]["pass"]}, '
          f'{info["resume"]["evaluations_reused"]} evaluations reused: '
          f'minimum {moved:.1e} Bohr, energy {de:.1e} Ha from the '
          f'uninterrupted walk')
    assert moved <= SAME and de <= SAME
    # the drift is a difference of two energies of two independent SCF solves
    # (one per walk): bitwise on the recording BLAS, energy class bar on another
    if info['refreeze_drift'] is None or info_ref['refreeze_drift'] is None:
        assert info['refreeze_drift'] == info_ref['refreeze_drift']
    else:
        assert_same(info['refreeze_drift'], info_ref['refreeze_drift'],
                    label='refreeze drift')
    assert info['cycles'] == info_ref['cycles']


def test_a_killed_walk_resumes_where_it_stopped(tmp_path):
    run, fresh = production(tmp_path)
    mol0 = run.mol_in
    # the uninterrupted walk, every state kept
    every, states = kept(tmp_path / 'whole')
    mol_ref, info_ref = walk(fresh(), mol0, checkpoint=every)
    assert info_ref['converged'] and info_ref['refreeze'] == 'measured'
    passes = [read_record(p)['pass'] for p in states]
    print(f'uninterrupted: {len(states)} checkpoints, passes {passes}, '
          f'{info_ref["first_walk_cycles"]} + {info_ref["cycles"]} cycles')
    assert 1 in passes

    # killed in the first walk, resumed from the last state it wrote; the
    # relaunch hands the walk an unevaluated surface at R0
    killed, written = kept(tmp_path / 'killed', KILLED_AFTER + 1)
    try:
        walk(fresh(), mol0, checkpoint=killed)
    except Killed:
        pass
    else:
        raise AssertionError('the checkpoint callable did not stop the walk')
    state = read_record(written[-1])
    assert state['pass'] == 0
    mol, info = walk(fresh(), mol0, resume=state)
    assert info['resume'] == {'pass': 0,
                              'evaluations_reused': len(state['history'])}
    same_walk(mol, info, mol_ref, info_ref)

    # resumed in the walk on the refrozen surface
    state = read_record(next(p for p in states if read_record(p)['pass'] == 1))
    mol, info = walk(fresh(), mol0, resume=state)
    assert info['resume']['pass'] == 1
    same_walk(mol, info, mol_ref, info_ref)


def bent_water():
    """Water/STO-3G Hartree-Fock away from its minimum, and its factory."""
    mol = gto.M(atom='O 0 0 0.15; H 0 0.85 -0.45; H 0 -0.70 -0.50',
                basis='sto-3g', verbose=0)

    def factory(m):
        mf = scf.RHF(m)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        return mf.run()
    return mol, factory


def test_a_mean_field_walk_resumes_bitwise_in_either_pass(tmp_path):
    mol, factory = bent_water()
    every, states = kept(tmp_path / 'whole')
    mol_ref, info_ref = walk(MeanFieldSurface(mol, factory), mol,
                             checkpoint=every)
    passes = [read_record(p)['pass'] for p in states]
    assert info_ref['converged'] and passes.count(0) > KILLED_AFTER
    for k in (KILLED_AFTER, passes.index(1)):
        state = read_record(states[k])
        mol_again, info = walk(MeanFieldSurface(mol, factory), mol,
                               resume=state)
        same_walk(mol_again, info, mol_ref, info_ref)


def test_a_killed_geometric_walk_reaches_the_same_minimum(tmp_path):
    pytest.importorskip('geometric')
    mol, factory = bent_water()
    whole = tmp_path / 'whole'
    whole.mkdir()
    mol_ref, info_ref = relax(MeanFieldSurface(mol, factory), mol,
                              engine='geometric', verbose=False,
                              workdir=str(whole))
    killed, written = kept(tmp_path / 'killed', KILLED_AFTER + 1)
    with pytest.raises(Killed):
        relax(MeanFieldSurface(mol, factory), mol, engine='geometric',
              verbose=False, checkpoint=killed, workdir=str(tmp_path))
    state = read_record(written[-1])
    assert state['engine'] == 'geometric'
    mol_again, info = relax(MeanFieldSurface(mol, factory), mol,
                            engine='geometric', verbose=False, resume=state,
                            workdir=str(tmp_path))
    assert info['converged']
    assert info['resume'] == {'pass': 0,
                              'evaluations_reused': len(state['history']),
                              'geometric_restarted': True}
    moved = aligned_displacement(mol_ref.atom_coords(),
                                 mol_again.atom_coords())
    de = abs(info['energy'] - info_ref['energy'])
    print(f'geomeTRIC restarted after {len(state["history"])} evaluations: '
          f'{moved:.1e} Bohr, {de:.1e} Ha from the uninterrupted minimum')
    assert moved < GEOMETRIC_SAME[0] and de < GEOMETRIC_SAME[1]
