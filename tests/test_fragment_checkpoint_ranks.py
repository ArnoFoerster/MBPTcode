"""The fragment partition's checkpoint over ranks: rank 0 alone reads a stage
file, and every rank gets its verdict and its arrays bit for bit.

    OMP_NUM_THREADS=1 mpirun -n 2 python tests/test_fragment_checkpoint_ranks.py

Script and pytest module: under pytest every rank is a thread of one process
(`simulated_comm.run_simulated`, 1, 2 and 3 ranks), because MPI cannot
initialize inside a sandboxed test runner; run as a script under mpirun or
srun it checks the same over real ranks and exits non-zero if any check failed
on any rank.

`PartitionCheckpoint.load` (`src/fragments/checkpoint.py`) reads on rank 0 and
hands the verdict, then the arrays, to every rank through `lockstep`. The
operator and orbitals are synthetic (a random symmetric matrix, random
rotations; identical on every rank), so the checks are about the files and
the collectives, not a partition. Each rank works in its OWN directory, and
every rank but 0 holds a decoy: a valid file of the same stage and key with
other arrays. Checks, on every rank:
  * `save` writes on rank 0 only (no file appears in another rank's directory)
  * a hit -- the block eigenpairs (a Fortran-ordered matrix among them), the
    guard's 0-d value, the resolvent with an integer array -- returns rank 0's
    saved arrays bit for bit, shape and dtype, never the decoy
  * a rank whose operator differs from rank 0's still gets rank 0's verdict
    and arrays: only rank 0 compares the fingerprint
  * a miss gives None on every rank: no file, another key, another
    fingerprint, a torn file
  * `_read` ran on rank 0 once per load and never elsewhere, and every rank
    made the same number of `lockstep` calls (the collectives paired)
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils.parallel.mpi_grid import (distributed, grid_comm, lockstep,
                                              lockstep_stats)
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.fragments.checkpoint import PartitionCheckpoint

NOCC, NVIR = 3, 4
SITES = {0: 1, 1: 1}
CT = {(0, 1): 1}
SETTINGS = dict(tol=1e-10, margin=0.01, dense_max=3000)
BLOCK = dict(name='site 0', n=1, n_get=2, rows=6)
SOLVED = dict(omega0=0.3, p='digest')


class Operator:
    """K x for a fixed symmetric K: the members `PartitionCheckpoint.bind`
    reads."""
    tda, nocc, nvir, omega = True, NOCC, NVIR, None
    dim = NOCC * NVIR

    def __init__(self, shift=0.0):
        k = np.random.default_rng(1).standard_normal((self.dim, self.dim))
        self.k = k + k.T + shift * np.eye(self.dim)

    def apply(self, x):
        return self.k @ x


class Orbitals:
    def __init__(self):
        rng = np.random.default_rng(2)
        self.u_occ = np.linalg.qr(rng.standard_normal((NOCC, NOCC)))[0]
        self.u_vir = np.linalg.qr(rng.standard_normal((NVIR, NVIR)))[0]
        self.occ_labels = np.array([0, 0, 1])
        self.vir_labels = np.array([0, 0, 1, 1])


EPS = np.linspace(-0.5, 0.8, NOCC + NVIR)


def saved_arrays(seed):
    """The arrays rank 0 saves (seed 3) or a decoy holds (another seed)."""
    rng = np.random.default_rng(seed)
    return dict(
        block_0=dict(w=rng.standard_normal(2),
                     v=np.asfortranarray(rng.standard_normal((6, 2)))),
        guard=dict(q_low=np.asarray(0.33 + rng.uniform())),
        resolvent=dict(y=rng.standard_normal((Operator.dim, 3)),
                       rows=rng.integers(0, 99, 5)))


STAGE_EXTRA = dict(block_0=BLOCK, guard=SOLVED, resolvent=SOLVED)


def write_decoy(ckpt, stage, arrays):
    """A valid file of `stage` in this rank's own directory (`save` writes
    on rank 0 only, so a decoy is written here directly)."""
    os.makedirs(ckpt.directory, exist_ok=True)
    np.savez(os.path.join(ckpt.directory, f'{stage}.npz'),
             key=json.dumps(ckpt._key(STAGE_EXTRA[stage]), sort_keys=True),
             **{f'fp_{n}': a for n, a in ckpt.fingerprint.items()},
             **{f'out_{n}': a for n, a in arrays.items()})


def digest(arrays):
    if arrays is None:
        return None
    h = hashlib.sha256()
    for n in sorted(arrays):
        a = np.asarray(arrays[n])
        h.update(f'{n}{a.shape}{a.dtype.str}'.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def same_bits(got, want):
    return (got is not None and sorted(got) == sorted(want)
            and all(np.asarray(got[n]).shape == np.asarray(want[n]).shape
                    and np.asarray(got[n]).dtype == np.asarray(want[n]).dtype
                    and np.array_equal(got[n], want[n]) for n in want))


def counted(ckpt):
    """Count this checkpoint's file reads."""
    calls = [0]
    read = ckpt._read

    def wrapped(stage, extra):
        calls[0] += 1
        return read(stage, extra)
    ckpt._read = wrapped
    return calls


def run_checks(comm):
    """[(check, passed, detail)] on this rank; a collective."""
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1
    root = rank == 0
    out = []

    def check(name, ok, detail=''):
        out.append((name, bool(ok), str(detail)))

    base = lockstep(tempfile.mkdtemp(prefix='ckpt_ranks_') if root else '')
    mine = os.path.join(base, f'rank{rank}')
    os.makedirs(mine, exist_ok=True)
    lockstep_stats(reset=True)
    try:
        ckpt = PartitionCheckpoint(mine, tag='t')
        ckpt.bind(Operator(), Orbitals(), EPS, SITES, CT, SETTINGS)
        reads = counted(ckpt)
        loads = 0
        want = saved_arrays(3)
        for stage, arrays in want.items():
            ckpt.save(stage, extra=STAGE_EXTRA[stage], **arrays)
        written = sorted(os.listdir(mine))
        check('save writes on rank 0 only',
              written == (sorted(f'{s}.npz' for s in want) if root else []),
              written)
        if not root:
            for stage, arrays in saved_arrays(4 + rank).items():
                write_decoy(ckpt, stage, arrays)

        for stage, arrays in want.items():
            got = ckpt.load(stage, **STAGE_EXTRA[stage])
            loads += 1
            check(f'hit {stage}: rank 0 arrays, bit for bit',
                  same_bits(got, arrays), digest(got))
            seen = comm.allgather(digest(got)) if comm is not None else [digest(got)]
            check(f'hit {stage}: one set of bits on every rank',
                  len(set(seen)) == 1 and seen[0] == digest(arrays), seen)

        # rank 0 alone compares the fingerprint: a rank whose operator
        # differs still takes rank 0's verdict and arrays
        skewed = PartitionCheckpoint(mine, tag='t')
        skewed.bind(Operator(0.0 if root else 1.0), Orbitals(), EPS, SITES,
                    CT, SETTINGS)
        skew_reads = counted(skewed)
        got = skewed.load('block_0', **BLOCK)
        check('skewed operator off rank 0: rank 0 verdict and arrays',
              same_bits(got, want['block_0']), digest(got))

        # misses: None on every rank
        moved = PartitionCheckpoint(mine, tag='t')
        moved.bind(Operator(1e-6), Orbitals(), EPS, SITES, CT, SETTINGS)
        move_reads = counted(moved)
        if root:
            with open(os.path.join(mine, 'torn.npz'), 'wb') as f:
                f.write(b'not an npz')
        misses = [('no file', ckpt, 'absent', BLOCK),
                  ('another key', ckpt, 'block_0', dict(BLOCK, n=2)),
                  ('a torn file', ckpt, 'torn', BLOCK),
                  ('another fingerprint', moved, 'block_0', BLOCK)]
        for name, c, stage, extra in misses:
            got = c.load(stage, **extra)
            loads += c is ckpt
            seen = comm.allgather(got is None) if comm is not None else [got is None]
            check(f'miss ({name}): None on every rank', all(seen), seen)

        check('_read on rank 0 only, once per load',
              (reads[0], skew_reads[0], move_reads[0])
              == ((loads, 1, 1) if root else (0, 0, 0)),
              (reads[0], skew_reads[0], move_reads[0], loads))
        calls = lockstep_stats()['calls']
        seen = comm.allgather(calls) if comm is not None else [calls]
        check('every rank made the same lockstep calls',
              len(set(seen)) == 1, seen)
    finally:
        shutil.rmtree(mine, ignore_errors=True)
        if comm is not None:
            comm.allgather(None)
        if root:
            shutil.rmtree(base, ignore_errors=True)
    if size > 1:
        check('ran over ranks', comm.allgather(rank) == list(range(size)))
    return out


@pytest.mark.parametrize('size', [1, 2, 3])
def test_checkpoint_over_thread_ranks(size):
    for rank, checks in enumerate(run_simulated(run_checks, size)):
        failed = [(n, d) for n, ok, d in checks if not ok]
        assert not failed, f'rank {rank}: {failed}'


def main(comm):
    with distributed(comm):
        checks = run_checks(comm)
    size = comm.Get_size() if comm is not None else 1
    every = comm.allgather(checks) if comm is not None else [checks]
    if comm is None or comm.Get_rank() == 0:
        print(f'partition checkpoint over {size} rank(s)')
        for rank, rank_checks in enumerate(every):
            for name, ok, detail in rank_checks:
                print(f'  rank {rank} {"PASS" if ok else "FAIL"} {name}'
                      + ('' if ok else f': {detail}'))
    failed = any(not ok for rank_checks in every for _, ok, _ in rank_checks)
    if comm is None or comm.Get_rank() == 0:
        print('FAILURES DETECTED' if failed else 'ALL PASSED')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
