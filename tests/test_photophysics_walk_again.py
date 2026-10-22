"""A relaunch walks a stored walk that did not converge again,
as the solvated driver does (`_drop_unconverged`): the stage
whose record, or whose '<stage>_walked', is flagged 'walk_not_converged' is
set aside with every stage built on its minimum (`stages.walk_again`), into
unconverged/<stage>-<time>/, and the stage walks again from its start. A
converged walk, a stage this launch does not ask for and a far start's chain
stay. Records written by hand; no walk runs.
"""
import numpy as np

from src.photophysics import stages
from tests.photophysics_cases import run_of

FLAGGED = {'numerical_flags': {'walk_not_converged': 'max cycles'}}


def stored(run, *names, flagged=()):
    """Write a record under each of `names`, the ones in `flagged` with a
    walk that did not converge (a '<stage>_walked' in its info)."""
    for name in names:
        bad = name in flagged
        record = ({'info': FLAGGED if bad else {'numerical_flags': {}}}
                  if name.endswith('_walked') else
                  dict(FLAGGED if bad else {'numerical_flags': {}},
                       coords_bohr=np.zeros((2, 3))))
        run.write(name, record)


EVERY = ('s0', 'hess', 'vert', 's1', 's1_walked', 's1_progress',
         's1_start_chain', 't1', 't1_walked', 'rate')


def test_an_unconverged_s1_is_walked_again_with_what_is_built_on_it(
        tmp_path):
    run = run_of(tmp_path)
    stored(run, *EVERY, flagged=('s1',))
    out = stages.walk_again(run, stages.STAGE_ORDER)
    assert set(out) == {'s1'}
    gone = {n.split('.')[0] for n in out['s1']}
    assert gone == {'s1', 's1_walked', 's1_progress', 't1', 't1_walked',
                    'rate'}
    for name in ('s0', 'hess', 'vert', 's1_start_chain'):
        assert run.store.has(name), name
    for name in gone:
        assert not run.store.has(name), name
    aside = list((run.store.dir / 'unconverged').iterdir())
    assert len(aside) == 1 and aside[0].name.startswith('s1-')
    assert (aside[0] / 's1.json').exists()


def test_a_flagged_walked_record_alone_is_walked_again(tmp_path):
    run = run_of(tmp_path)
    stored(run, 's0', 't1_walked', 'rate', flagged=('t1_walked',))
    out = stages.walk_again(run, ('t1', 'rate'))
    assert set(out) == {'t1'}
    assert not run.store.has('t1_walked') and not run.store.has('rate')
    assert run.store.has('s0')


def test_an_unconverged_ground_walk_takes_every_stage(tmp_path):
    run = run_of(tmp_path)
    stored(run, *EVERY, flagged=('s0',))
    out = stages.walk_again(run, stages.STAGE_ORDER)
    assert set(out) == {'s0'}
    assert [n for n in EVERY if run.store.has(n)] == ['s1_start_chain']


def test_converged_and_unasked_walks_stay(tmp_path):
    run = run_of(tmp_path)
    stored(run, *EVERY, flagged=('t1',))
    assert stages.walk_again(run, ('s0', 's1')) == {}
    assert all(run.store.has(n) for n in EVERY)


def test_run_tadf_walks_again_before_the_stages(tmp_path, monkeypatch):
    run = run_of(tmp_path)
    stored(run, 's0', 's1', flagged=('s1',))
    seen = {}
    monkeypatch.setattr(stages, 'ground_minimum', lambda run_: {})
    monkeypatch.setitem(stages.STAGES, 's1', lambda run_: seen.setdefault(
        's1', run_.store.has('s1')))
    monkeypatch.setattr(stages, 'describe_run', lambda *a: None)
    monkeypatch.setattr(stages.Run, 'refuse_early', lambda self: None)
    stages.run_tadf(run.declaration, run.store, max_memory=4000,
                    stages=('s0', 's1'))
    assert seen == {'s1': False}, 'the flagged s1 was reused'
