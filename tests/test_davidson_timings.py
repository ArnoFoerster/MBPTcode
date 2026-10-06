"""Where the ISDF BSE Davidson spends its wall clock, serially and under a comm.

`info['timings']` carries the loop broken up: the block action, its
collectives included; the part spent in those collectives (the entry
lockstep of the trial vectors, the head's gather, the reductions) plus the
lockstep of the result, `davidson_comm`; the subspace work, which every rank
runs whole; the action's own build, paid once before the first trial vector;
what the locksteps moved, `davidson_lockstep_mb`, and what their digests
showed every rank already held, `davidson_lockstep_skipped_mb`; and the
counts (iterations, vectors applied, largest subspace) that say whether the
cost is the action or the number of times the solver calls it.

The block action is split further into its local pieces
(`davidson_action_<piece>`: the GEMMs, the Hadamard product, the Hartree
term, the owner-order write, the batch's element-wise terms), which with the
collectives inside it (`davidson_comm_gather`, `_reduce_grid`,
`_reduce_pairs`) say what of a vector's time a rank count divides. The
subspace work is split likewise where the trial space is cut by pair rows
(`davidson_subspace_<piece>`, zero serially), beside the bytes this rank's
holders take (`davidson_trial_space_mb`).

Gated on water/cc-pVDZ Hartree-Fock: the keys exist on every rank; the parts
do not exceed the whole; the counts equal the `stats` the same run reports;
every rank reports rank 0's counts and lockstep volume and a subspace of its
own; and the whole dict survives `json.dumps`.

The `qp` stage carries the same kind of breakdown from the space-time GW
solve `qp='G0W0'` (or `self_consistency='evGW'`) runs underneath: `qp_chi0`,
`qp_dyson`, `qp_sigma`, the exchange build `qp_static` and the per-state root
search `qp_states` (`GW/space_time.py`'s own `timings` dict, merged by
`_QPStageTimings`). Gated: the keys exist serially and under a comm; the
parts stay inside the `qp` stage; evGW's cycles sum into those keys, with
`qp_cycles` saying how many.

The timers move no bits: serially, the roots and vectors are bitwise those
of a pinned tree without the `qp_*` timers, extracted with `git archive` and
run in its own process with its W frequency axis moved onto rW, the range its
transform is fitted over. The pinned tree has no communicator, and a
distributed run sums in rank order, so the distributed gates compare every
rank with rank 0 instead.
"""
import copy
import json
import os
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.davidson import (ACTION_PIECES,
                                                         COMM_KINDS,
                                                         _QPStageTimings,
                                                         solve_bse_isdf)
from src.SingleReference.LinearResponse.trial_space import SUBSPACE_PIECES

SIZES = [2, 3]
NROOTS = 3
#: s. The parts are read with time.time() at the loop boundaries, so reading
#: the clock is itself outside some of them and the parts may exceed the whole
#: by that much.
TIMER_SLACK = 1e-3
#: Every rank fills these; `davidson_block_action_by_rank` and
#: `davidson_setup_by_rank` only under a comm.
TIMING_KEYS = ('davidson_block_action', 'davidson_subspace', 'davidson_comm',
               'davidson_setup', 'davidson_iterations',
               'davidson_subspace_max', 'davidson_vectors_applied',
               'davidson_lockstep_mb', 'davidson_lockstep_skipped_mb',
               'davidson_max_space', 'davidson_collapses',
               *(f'davidson_comm_{kind}' for kind in COMM_KINDS),
               *(f'davidson_action_{piece}' for piece in ACTION_PIECES),
               *(f'davidson_subspace_{piece}' for piece in SUBSPACE_PIECES),
               'davidson_trial_space_mb')
#: The collectives inside the block action; the lockstep kind also holds the
#: result's, which is outside it.
ACTION_COMM_KINDS = ('gather', 'reduce_grid', 'reduce_pairs')
#: Every rank fills these too: the space-time QP solve's chi0, sigma,
#: exchange build and root search all run, and are timed, on every rank.
QP_TIMING_KEYS = ('qp_chi0', 'qp_dyson', 'qp_sigma', 'qp_static', 'qp_states')

REPO = Path(__file__).resolve().parents[1]
#: A tree without the `qp_*` timers, so the bitwise gate below compares
#: against code that cannot move a bit for the reason tested here.
BASELINE_COMMIT = '3ae688706f409591b2304d9a7ef653122aa36be6'
#: The archived probe's thread caps, so it does not size itself against the
#: whole machine.
THREAD_CAPS = {name: '2' for name in
              ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
#: Built fresh in its own process against the extracted tree, on the same
#: molecule the `water` fixture below uses; serial, since the extracted tree
#: has no communicator.
BITWISE_PROBE = '''
import sys
import warnings

sys.path.insert(0, {archive!r})

import numpy as np
from pyscf import gto, scf

from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.davidson import solve_bse_isdf

import src.SingleReference.GW.space_time as _archived_space_time
from src.Base.utils.grids import minimax_frequency_grid as _minimax_frequency_grid
from src.Base.utils.time_frequency import SELF_ENERGY_PAD as _PAD

# the one change since the pinned commit: W's frequencies span rW, the range
# its transform back to tau is fitted over, not the bare window
_archived_space_time.minimax_frequency_grid = (
    lambda n, e_min, e_max: _minimax_frequency_grid(n, _PAD[0] * e_min,
                                                    _PAD[1] * e_max))

warnings.simplefilter('ignore')
mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
mf.kernel()
nocc = mol.nelectron // 2
factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
omega, X, Y, _ = solve_bse_isdf(mf, mol, nocc, nroots={nroots}, probe=True,
                                progress=False, factors=factors)
np.savez({out!r}, omega=np.asarray(omega), X=np.asarray(X), Y=np.asarray(Y))
'''


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    return dict(mf=mf, mol=mol, nocc=nocc, factors=factors)


def _rank_copy(w):
    """This rank's own mean field and factors: a distributed solve replicates
    rank 0's over them in place, and simulated ranks share one process."""
    mf = copy.copy(w['mf'])
    mf.mo_energy = np.asarray(w['mf'].mo_energy, float).copy()
    mf.mo_coeff = np.asarray(w['mf'].mo_coeff, float).copy()
    return mf, tuple(np.array(a, copy=True) for a in w['factors'])


def _check_iterating_rank(t, stats):
    """One rank's iteration: keys present, parts inside the whole, and the
    counts the same numbers `stats` reports."""
    for key in TIMING_KEYS:
        assert key in t, key
    assert t['davidson_block_action'] > 0.0
    # The action's build is paid inside the stage and outside every
    # per-iteration term, so it is a third summand rather than a part of one.
    assert (t['davidson_block_action'] + t['davidson_subspace']
            + t['davidson_setup'] <= t['davidson'] + TIMER_SLACK)
    assert t['davidson_subspace'] >= 0.0
    assert t['davidson_setup'] >= 0.0
    assert t['davidson_iterations'] >= 1
    assert t['davidson_vectors_applied'] >= NROOTS
    assert t['davidson_subspace_max'] >= NROOTS
    # the timings and the stats read the same counters
    assert t['davidson_iterations'] == stats['davidson_vind_calls']
    assert t['davidson_vectors_applied'] == stats['davidson_block_actions']
    assert t['davidson_block_action'] == stats['davidson_action_s']
    # A vind call applies an X block and a Y block, so the subspace is never
    # more than half the vectors the action saw.
    assert 2 * t['davidson_subspace_max'] <= t['davidson_vectors_applied']
    # The action's local pieces all ran (a singlet BSE has every one), and
    # with its own collectives they stay inside it.
    pieces = [t[f'davidson_action_{piece}'] for piece in ACTION_PIECES]
    assert min(pieces) > 0.0
    inside = sum(pieces) + sum(t[f'davidson_comm_{kind}']
                               for kind in ACTION_COMM_KINDS)
    assert inside <= t['davidson_block_action'] + TIMER_SLACK
    # The pair-row loop's pieces stay inside the subspace work, and zero
    # where real_eig runs whole; the holders take their 32 bytes per pair.
    sub = [t[f'davidson_subspace_{piece}'] for piece in SUBSPACE_PIECES]
    assert min(sub) >= 0.0
    assert sum(sub) <= t['davidson_subspace'] + TIMER_SLACK
    assert t['davidson_trial_space_mb'] >= 0.0
    json.dumps(t)                     # a caller may record it as JSON


def _check_qp_stage(t, against='qp'):
    """The `qp_*` keys exist and stay inside the stage they were merged from.

    `against` is the outer stage timer that bounds the sum: `'qp'` for
    `qp='G0W0'`'s single call, `'evgw'` for the whole eigenvalue loop, which
    also pays the DIIS/convergence bookkeeping around every cycle's call.
    """
    for key in QP_TIMING_KEYS:
        assert key in t, key
    assert (t['qp_chi0'] + t['qp_sigma'] + t['qp_static'] + t['qp_states']
            <= t[against] + TIMER_SLACK)
    json.dumps(t)                     # a caller may record it as JSON


def test_serial_davidson_timings(water):
    """Serially there is no traffic, and the loop is the action plus the
    subspace work."""
    w = water
    _, _, _, info = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=NROOTS,
                                   probe=True, progress=False,
                                   factors=w['factors'])
    t = info['timings']
    _check_iterating_rank(t, info['stats'])
    assert t['davidson_comm'] == 0.0
    assert t['davidson_lockstep_mb'] == 0.0
    assert t['davidson_lockstep_skipped_mb'] == 0.0
    assert t['davidson_setup'] > 0.0
    assert 'davidson_block_action_by_rank' not in t
    assert 'davidson_setup_by_rank' not in t
    _check_qp_stage(t)
    assert 'qp_cycles' not in t          # one call, no evGW loop to sum over


@pytest.mark.parametrize('size', SIZES)
def test_distributed_davidson_timings(water, size):
    """Under a comm every rank runs the same iteration on the lockstepped
    action output: each reports rank 0's counts and lockstep volume, a
    subspace of its own and time in collectives, and the gathered block-action
    times are the per-rank load the row split produced."""
    w = water

    def one_rank(comm):
        mf, factors = _rank_copy(w)
        om, _, _, info = solve_bse_isdf(mf, w['mol'], w['nocc'], nroots=NROOTS,
                                        probe=True, progress=False,
                                        factors=factors, distribute=True,
                                        comm=comm)
        return om, info

    out = run_simulated(one_rank, size)
    om0, info0 = out[0]
    t0 = info0['timings']
    # The trial vectors and the result go through checked locksteps, and
    # thread-ranks repeat each other's bits: all covered, nothing moved.
    assert t0['davidson_lockstep_skipped_mb'] > 0.0
    assert t0['davidson_lockstep_mb'] == 0.0
    by_rank = t0['davidson_block_action_by_rank']
    assert len(by_rank) == size
    setup_by_rank = t0['davidson_setup_by_rank']
    assert len(setup_by_rank) == size
    for r, (om, info) in enumerate(out):
        t = info['timings']
        assert np.abs(om - om0).max() == 0.0        # rank 0's roots, returned
        _check_iterating_rank(t, info['stats'])
        # The same iteration everywhere: the same cycles, vectors and largest
        # subspace as rank 0, and the same bytes through the locksteps.
        for key in ('davidson_iterations', 'davidson_vectors_applied',
                    'davidson_subspace_max', 'davidson_lockstep_mb',
                    'davidson_lockstep_skipped_mb'):
            assert t[key] == t0[key], key
        assert t['davidson_comm'] > 0.0             # the lockstep and reductions
        assert t['davidson_comm'] <= t['davidson'] + TIMER_SLACK
        # Split by collective, the kinds are the whole of it, and the ISDF
        # action's per-vector gather and grid-length reduce both ran.
        parts = [t[f'davidson_comm_{kind}'] for kind in COMM_KINDS]
        assert min(parts) >= 0.0
        assert abs(sum(parts) - t['davidson_comm']) <= TIMER_SLACK
        assert t['davidson_comm_gather'] > 0.0
        assert t['davidson_comm_reduce_grid'] > 0.0
        assert t['davidson_block_action_by_rank'] == by_rank
        assert by_rank[r] == t['davidson_block_action']
        assert t['davidson_setup'] > 0.0
        assert t['davidson_setup_by_rank'] == setup_by_rank
        assert setup_by_rank[r] == t['davidson_setup']
        # The GW stage runs on every rank too: each ran its own share of
        # chi0/sigma/the exchange build/the root search and timed it.
        _check_qp_stage(t)
        assert 'qp_cycles' not in t


def test_qp_stage_timings_class_sums_additive_keys_only():
    """`_QPStageTimings`, which evGW's cycles write into once each: the
    per-stage keys accumulate across repeated writes, everything else
    (`nranks`, `ntau_auto`, ...) takes the latest write. A caller-supplied
    dict is written into directly, not copied, so `gw_kwargs['timings']`
    carries the final numbers."""
    caller = {'t_chi0': 1.0}
    qs = _QPStageTimings(caller)
    assert qs.target is caller
    qs['t_chi0'] = 2.0
    qs['t_sigma'] = 0.5
    qs['t_sigma'] = 0.25
    qs['nranks'] = 2
    qs['nranks'] = 2
    qs['ntau_auto'] = 24
    assert caller['t_chi0'] == pytest.approx(3.0)      # 1.0 + 2.0: summed
    assert caller['t_sigma'] == pytest.approx(0.75)    # 0.5 + 0.25: summed
    assert caller['nranks'] == 2                        # latest write only
    assert caller['ntau_auto'] == 24
    # No caller dict: the wrapper starts its own, empty.
    assert _QPStageTimings().target == {}


def test_evgw_qp_stage_sums_over_cycles(water):
    """evGW re-enters the space-time QP solve once per cycle on the same
    `timings` object: the merged `qp_*` keys are the total over the cycles,
    and `qp_cycles` says how many. `tol=1e-12` never converges in two cycles,
    so the count is deterministic."""
    w = water
    _, _, _, info = solve_bse_isdf(
        w['mf'], w['mol'], w['nocc'], nroots=NROOTS, probe=False,
        progress=False, factors=w['factors'], self_consistency='evGW',
        gw_kwargs=dict(max_cycle=2, tol=1e-12))
    t = info['timings']
    assert info['evgw']['cycles'] == 2
    assert not info['evgw']['converged']
    assert t['qp_cycles'] == 2
    _check_qp_stage(t, against='evgw')


@pytest.fixture(scope='session')
def archive(tmp_path_factory):
    """The pinned tree without the `qp_*` timers, unpacked into a temporary
    directory."""
    out = tmp_path_factory.mktemp('qp_timings_baseline')
    tar = out.parent / f'{BASELINE_COMMIT}.tar'
    done = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                           '-o', str(tar), BASELINE_COMMIT],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with tarfile.open(tar) as fh:
        fh.extractall(out)
    assert (out / 'src' / 'SingleReference' / 'LinearResponse'
           / 'davidson.py').is_file()
    return out


def _archived_bse_roots(archive_dir, tmp_path):
    """(omega, X, Y) from the pinned tree, run as its own process."""
    script = tmp_path / 'probe.py'
    out = tmp_path / 'roots.npz'
    script.write_text(BITWISE_PROBE.format(archive=str(archive_dir),
                                           nroots=NROOTS, out=str(out)))
    env = dict(os.environ, **THREAD_CAPS)
    env.pop('PYTHONPATH', None)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(archive_dir),
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    data = np.load(out)
    return data['omega'], data['X'], data['Y']


def _bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
           and a.tobytes() == b.tobytes())


def test_qp_timings_move_no_bits(water, tmp_path_factory, archive):
    """The timers only read the clock: the pinned tree's roots and vectors,
    run as a subprocess on the extracted tree, are bitwise this tree's.

    This tree is asked for the bare diagonal preconditioner, the only one the
    pinned tree has; the default screened diagonal moves the roots' last bits.
    """
    tmp = tmp_path_factory.mktemp('qp_timings_bitwise')
    omega_old, X_old, Y_old = _archived_bse_roots(archive, tmp)
    omega_new, X_new, Y_new, _ = solve_bse_isdf(
        water['mf'], water['mol'], water['nocc'], nroots=NROOTS,
        probe=True, progress=False, factors=water['factors'],
        preconditioner='bare')
    assert _bitwise(omega_old, omega_new)
    assert _bitwise(X_old, X_new)
    assert _bitwise(Y_old, Y_new)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
