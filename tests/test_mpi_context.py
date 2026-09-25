"""`mpi_grid.distributed`, `current_comm` and `lockstep` under simulated ranks.

MPI cannot initialize in the sandboxed test runner, so the ranks are threads of
this process (`run_simulated`). The abort guard's real-communicator path is
driven through a stand-in that is not a `SimulatedComm` -- the rule every
routine in `mpi_grid` dispatches on -- and records `Abort` instead of ending
the process.

Gated:
  * the context is per rank-thread, the innermost block wins, and leaving a
    block (normally or by an exception) restores the enclosing one;
    `distributed(None)` is a serial region and `mpi_map` runs its items there;
  * `lockstep` leaves rank 0's bits on every rank: rank 2 one ulp off is
    repaired and the audit counts exactly one mismatched call and the calls'
    bytes; ranks that all differ all come back holding rank 0's;
  * a C-contiguous array is its own broadcast buffer; a strided or Fortran
    array goes through a C-ordered temporary copied back into it;
  * tuple, list, namedtuple, dict, None, scalars and small objects;
  * a shape, dtype or structure disagreement, and a read-only receiver, raise
    on EVERY rank with every buffer untouched;
  * serially and on one rank `lockstep` returns its argument, communicating
    nothing; without audit no comparison copy is made;
  * `lockstep(check=True)` on identical copies broadcasts nothing at 2, 3 and
    8 ranks (`bytes` stays 0, `skipped_bytes` counts the arrays) and
    snapshots nothing under audit; with rank 2's array one ulp off it
    broadcasts that array alone and repairs it, the audit counting one
    mismatch there, and every rank counts the same calls and bytes; in a
    container the agreeing arrays stay put while the scalars still travel;
    serially it is the plain no-op; a read-only receiver and a call checked
    on some ranks only raise on every rank;
  * a kernel inside `run_simulated` finds the comm with no `comm=`;
  * a SimulatedComm installs no hook and re-raises; a wire comm installs both
    hooks, aborts once with the traceback printed when an exception leaves
    the outermost block, and not when one is handled inside it.
"""
import copy
import os
import pickle
import sys
import threading
from collections import namedtuple
from fractions import Fraction

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils import mpi_grid
from src.Base.utils.mpi_grid import (SimulatedComm, current_comm, distributed,
                                     lockstep, lockstep_stats, mpi_map,
                                     partition, reduce_sum, run_simulated,
                                     simulated_world)

SIZE = 3
HANG_SECONDS = 60                              # the whole file runs in ~1 s

Pair = namedtuple('Pair', 'energy coeff')


class _WireComm:
    """An mpi4py communicator as `mpi_grid` sees one: not a SimulatedComm.

    `Abort` records its code rather than ending the process."""

    def __init__(self, size=2, rank=1):
        self.size, self.rank, self.aborted = size, rank, []

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Abort(self, code):
        self.aborted.append(code)


def _spy_collectives(monkeypatch):
    """Record, per rank, every buffer and object a SimulatedComm moves."""
    seen = {}
    lock = threading.Lock()

    def wrap(name):
        real = getattr(SimulatedComm, name)

        def recording(self, *args, **kwargs):
            with lock:
                seen.setdefault(self.Get_rank(), []).append((name, args[0]))
            return real(self, *args, **kwargs)
        monkeypatch.setattr(SimulatedComm, name, recording)

    for name in ('bcast_into', 'bcast', 'allgather'):
        wrap(name)
    return seen


def _broadcast_buffers(seen, rank):
    return [buf for name, buf in seen.get(rank, []) if name == 'bcast_into']


def _hooks():
    return sys.excepthook, threading.excepthook


def _ranks(fn, size, *args):
    """`run_simulated`, failing instead of hanging when the ranks deadlock --
    what a broken collective does rather than raise."""
    box = {}

    def run():
        try:
            box['out'] = run_simulated(fn, size, *args)
        except BaseException as exc:            # noqa: BLE001 -- re-raised
            box['exc'] = exc

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(HANG_SECONDS)
    if runner.is_alive():
        pytest.fail(f'{size} simulated ranks deadlocked')
    if 'exc' in box:
        raise box['exc']
    return box['out']


def _lockstep_refusal(exc):
    """The message of lockstep's own refusal; anything else is re-raised, so a
    rank that failed inside a collective ends the run instead of returning
    while its peers wait."""
    if 'lockstep (raised on rank' not in str(exc):
        raise exc
    return str(exc)


def _kernel(v, comm=None):
    """A kernel of the contract: comm=None falls back to the context, the input
    is lockstepped, each rank squares its round-robin share, the shares add."""
    comm = current_comm() if comm is None else comm
    v = lockstep(v, comm)
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    part = np.zeros_like(v)
    mine = partition(v.size, rank, size)
    part[mine] = v[mine] ** 2
    return reduce_sum(part, comm)


# ------------------------------------------------------------------ context
def test_each_rank_thread_reads_its_own_comm():
    def one_rank(comm):
        first = current_comm() is comm
        comm.allgather(None)                   # every rank inside at once
        return first, current_comm() is comm, current_comm().Get_rank()

    assert _ranks(one_rank, SIZE) == [(True, True, r) for r in range(SIZE)]
    assert current_comm() is None


def test_rank_threads_do_not_inherit_the_callers_comm():
    outer = simulated_world(2)[0]
    with distributed(outer) as got:
        assert got is outer
        seen = run_simulated(lambda comm: current_comm() is comm, 2)
        assert current_comm() is outer
    assert seen == [True, True]
    assert current_comm() is None


def test_the_innermost_block_wins_and_leaving_restores():
    a, b = simulated_world(2)
    assert current_comm() is None
    with distributed(a):
        assert current_comm() is a
        with distributed(b):
            assert current_comm() is b
            with distributed(None):
                assert current_comm() is None
            assert current_comm() is b
        assert current_comm() is a
    assert current_comm() is None


def test_an_exception_restores_the_enclosing_comm():
    a, b = simulated_world(2)
    with distributed(a):
        with pytest.raises(RuntimeError, match='inner'):
            with distributed(b):
                with distributed(None):
                    raise RuntimeError('inner')
        assert current_comm() is a
    with pytest.raises(KeyError):
        with distributed(a):
            with distributed(b):
                raise KeyError('both')
    assert current_comm() is None


def test_a_kernel_finds_the_comm_without_being_passed_one():
    v0 = np.linspace(0.5, 3.0, 11)
    serial = _kernel(v0.copy())

    def one_rank(comm):
        v = v0 + comm.Get_rank()               # diverged copies of one input
        return _kernel(v), v

    for out, v in _ranks(one_rank, SIZE):
        assert out.tobytes() == serial.tobytes()
        assert v.tobytes() == v0.tobytes()


def test_mpi_maps_items_run_in_a_serial_region():
    def one_rank(comm):
        out = mpi_map(lambda i: (i, current_comm() is None), range(5),
                      comm=comm)
        return out, current_comm() is comm

    for out, restored in _ranks(one_rank, SIZE):
        assert out == [(i, True) for i in range(5)]
        assert restored


# ----------------------------------------------------------------- lockstep
def test_lockstep_repairs_a_rank_one_ulp_off():
    ref = np.linspace(-1.0, 2.0, 12).reshape(3, 4)
    other = np.arange(5, dtype=np.int64)
    ulp = abs(np.nextafter(ref[1, 2], np.inf) - ref[1, 2])

    def one_rank(comm):
        x, y = ref.copy(), other.copy()
        if comm.Get_rank() == 2:
            x[1, 2] = np.nextafter(x[1, 2], np.inf)
        with distributed(comm, audit=True):
            assert lockstep(x) is x
            assert lockstep(y) is y
        return x, lockstep_stats()

    for rank, (x, stats) in enumerate(_ranks(one_rank, SIZE)):
        assert x.tobytes() == ref.tobytes()
        assert stats == {'calls': 2, 'bytes': ref.nbytes + other.nbytes,
                         'checked_calls': 0, 'skipped_bytes': 0,
                         'audited_calls': 2,
                         'mismatched_calls': 1 if rank == 2 else 0,
                         'max_abs_diff': ulp if rank == 2 else 0.0}


@pytest.mark.parametrize('size', [2, 3, 8])
def test_every_rank_comes_back_holding_rank_zeros_bits(size):
    def build(rank):
        rng = np.random.default_rng(100 + rank)
        return (rng.standard_normal((4, 5)),
                rng.standard_normal(3) + 1j * rng.standard_normal(3),
                rng.integers(0, 9, 6), rng.random(4) > 0.5,
                np.array(rng.random()))

    def one_rank(comm):
        mine = build(comm.Get_rank())
        out = lockstep(mine)
        return ([a.tobytes() for a in out],
                all(o is m for o, m in zip(out, mine)))

    want = [a.tobytes() for a in build(0)]
    for got, same_objects in _ranks(one_rank, size):
        assert got == want
        assert same_objects


def test_a_c_contiguous_array_is_its_own_broadcast_buffer(monkeypatch):
    seen = _spy_collectives(monkeypatch)

    def one_rank(comm):
        x = np.full((3, 4), float(comm.Get_rank()))
        return x, lockstep(x)

    for rank, (x, out) in enumerate(_ranks(one_rank, SIZE)):
        assert out is x
        buffers = _broadcast_buffers(seen, rank)
        assert len(buffers) == 1 and buffers[0] is x
        assert not x.any()


def test_a_strided_array_goes_through_a_temporary(monkeypatch):
    seen = _spy_collectives(monkeypatch)
    fortran0 = np.arange(6.0).reshape(2, 3)

    def one_rank(comm):
        r = comm.Get_rank()
        base = np.full((4, 6), float(r))
        view = base[:, ::2]
        fortran = np.asfortranarray(fortran0 + r)
        return base, view, fortran, lockstep((view, fortran))

    for rank, (base, view, fortran, out) in enumerate(_ranks(one_rank,
                                                             SIZE)):
        assert out[0] is view and out[1] is fortran
        assert not view.any()
        assert np.array_equal(base[:, 1::2], np.full((4, 3), float(rank)))
        assert np.array_equal(fortran, fortran0) and fortran.flags.f_contiguous
        buffers = _broadcast_buffers(seen, rank)
        assert len(buffers) == 2
        for buf in buffers:
            assert buf.flags.c_contiguous
            assert not np.shares_memory(buf, base)
            assert not np.shares_memory(buf, fortran)


def test_containers_come_back_as_the_same_type():
    def one_rank(comm):
        r = comm.Get_rank()
        a, b, c = (np.full(shape, float(r)) for shape in (3, (2, 2), 2))
        x = {'tuple': (a, r), 'list': [b, None], 'pair': Pair(1.5 * r, c),
             'label': f'rank{r}'}
        return (a, b, c), lockstep(x)

    for (a, b, c), out in _ranks(one_rank, SIZE):
        assert type(out) is dict
        assert list(out) == ['tuple', 'list', 'pair', 'label']
        assert type(out['tuple']) is tuple
        assert out['tuple'][0] is a and out['tuple'][1] == 0
        assert type(out['list']) is list
        assert out['list'][0] is b and out['list'][1] is None
        assert type(out['pair']) is Pair
        assert out['pair'].energy == 0.0 and out['pair'].coeff is c
        assert out['label'] == 'rank0'
        assert not (a.any() or b.any() or c.any())


@pytest.mark.parametrize('make', [
    lambda r: 7 + r,
    lambda r: 0.1 * (r + 1),
    lambda r: complex(1.0, r),
    lambda r: np.float64(r + 1) / 3,
    lambda r: Fraction(r + 1, 3),
    lambda r: 'x' * (r + 1),
    lambda r: {'ntau': 18 - r, 'grid': f'minimax{r}'},
], ids=['int', 'float', 'complex', 'numpy-scalar', 'object', 'str',
        'dict-of-scalars'])
def test_scalars_and_small_objects_are_rank_zeros(make):
    got = _ranks(lambda comm: lockstep(make(comm.Get_rank())), SIZE)
    assert got == [make(0)] * SIZE
    assert all(type(g) is type(make(0)) for g in got)


@pytest.mark.parametrize('case', ['shape', 'dtype', 'length', 'kind', 'keys'])
def test_a_disagreement_raises_on_every_rank_before_anything_moves(case):
    def build(rank):
        a = np.full((2, 3), float(rank))
        odd = rank == 1
        if case == 'shape':
            return np.full((3, 3), 1.0) if odd else a
        if case == 'dtype':
            return a.astype(np.float32) if odd else a
        if case == 'length':
            return (a, a.copy()) if odd else (a,)
        if case == 'kind':
            return (a, 1.0 if odd else np.ones(1))
        return {'mo_energy' if odd else 'mo_coeff': a}

    def one_rank(comm):
        mine = build(comm.Get_rank())
        before = pickle.dumps(copy.deepcopy(mine))
        try:
            lockstep(mine)
        except ValueError as exc:
            return _lockstep_refusal(exc), pickle.dumps(mine) == before
        return None, None

    for rank, (message, untouched) in enumerate(_ranks(one_rank, SIZE)):
        assert message is not None
        assert f'raised on rank {rank}' in message
        assert 'rank 1 holds' in message and 'rank 0 holds' in message
        assert untouched


def test_a_read_only_receiver_raises_on_every_rank():
    def one_rank(comm, frozen):
        x = np.full(4, float(comm.Get_rank()))
        x.flags.writeable = comm.Get_rank() != frozen
        try:
            lockstep(x)
        except ValueError as exc:
            return _lockstep_refusal(exc), x.copy()
        return None, x.copy()

    for rank, (message, x) in enumerate(_ranks(one_rank, SIZE, 2)):
        assert 'rank 2 holds a read-only array' in message
        assert np.array_equal(x, np.full(4, float(rank)))
    for message, x in _ranks(one_rank, SIZE, 0):
        assert message is None and not x.any()   # the source is only read


def test_serially_and_on_one_rank_lockstep_returns_its_argument(monkeypatch):
    seen = _spy_collectives(monkeypatch)
    lockstep_stats(reset=True)
    x = np.arange(4.0)
    objects = (x, (x, 3), {'x': x}, 5, None)
    assert all(lockstep(o) is o for o in objects)
    with distributed(None):
        assert all(lockstep(o) is o for o in objects)

    def one_rank(comm):
        return (all(lockstep(o) is o for o in objects)
                and all(lockstep(o, comm) is o for o in objects))

    assert _ranks(one_rank, 1) == [True]
    assert seen == {}
    assert lockstep_stats()['calls'] == 0
    assert np.array_equal(x, np.arange(4.0))


def test_no_comparison_copy_without_audit(monkeypatch):
    copies = []
    real = mpi_grid._audit_snapshot

    def counting(a):
        copies.append(a.shape)
        return real(a)
    monkeypatch.setattr(mpi_grid, '_audit_snapshot', counting)

    def one_rank(comm, audit):
        x = np.full(8, float(comm.Get_rank()))
        with distributed(comm, audit=audit):
            lockstep(x)
        return lockstep_stats()

    plain = _ranks(one_rank, SIZE, False)
    assert copies == []
    for stats in plain:
        assert stats['calls'] == 1 and stats['bytes'] == 64
        assert stats['audited_calls'] == 0 and stats['mismatched_calls'] == 0
    audited = _ranks(one_rank, SIZE, True)
    assert len(copies) == SIZE - 1             # rank 0 is the source
    assert [s['mismatched_calls'] for s in audited] == [0] + [1] * (SIZE - 1)


# ---------------------------------------------------------- checked lockstep
@pytest.mark.parametrize('size', [2, 3, 8])
def test_a_checked_lockstep_on_identical_arrays_moves_nothing(monkeypatch,
                                                              size):
    """Every rank holds rank 0's bytes (in its own buffers, one of them
    Fortran-ordered): one allgather of digests, no broadcast, the same
    objects back untouched, `bytes` unchanged and `skipped_bytes` the whole
    payload -- and under audit no comparison copy, since nothing moved."""
    seen = _spy_collectives(monkeypatch)
    snapshots = []
    real = mpi_grid._audit_snapshot
    monkeypatch.setattr(mpi_grid, '_audit_snapshot',
                        lambda a: snapshots.append(a.shape) or real(a))
    ref = np.random.default_rng(3).normal(size=(6, 7))
    counts = np.arange(9, dtype=np.int64)

    def one_rank(comm, audit):
        x, y = ref.copy(), np.asfortranarray(counts.reshape(3, 3))
        with distributed(comm, audit=audit):
            lockstep_stats(reset=True)
            out = lockstep((x, y), check=True)
            stats = lockstep_stats()
        return (out[0] is x and out[1] is y, x.tobytes() == ref.tobytes(),
                stats)

    payload = ref.nbytes + counts.nbytes
    for audit in (False, True):
        seen.clear()
        for rank, (same, intact, stats) in enumerate(_ranks(one_rank, size,
                                                            audit)):
            assert same and intact
            assert _broadcast_buffers(seen, rank) == []
            assert [name for name, _ in seen[rank]] == ['allgather']
            assert stats['calls'] == 1 and stats['checked_calls'] == 1
            assert stats['bytes'] == 0 and stats['skipped_bytes'] == payload
            assert stats['audited_calls'] == int(audit)
            assert stats['mismatched_calls'] == 0
    assert snapshots == []


@pytest.mark.parametrize('size', [3, 8])
def test_a_checked_lockstep_repairs_a_rank_one_ulp_off(monkeypatch, size):
    """Rank 2's first array one ulp off rank 0's: the digests differ, that
    array is broadcast and repaired in place on rank 2, the agreeing one is
    not sent, the audit counts one mismatch of one ulp on rank 2 alone, and
    every rank counts the same calls and bytes."""
    seen = _spy_collectives(monkeypatch)
    ref = np.linspace(-1.0, 2.0, 12).reshape(3, 4)
    other = np.arange(5, dtype=np.int64)
    ulp = abs(np.nextafter(ref[1, 2], np.inf) - ref[1, 2])

    def one_rank(comm):
        x, y = ref.copy(), other.copy()
        if comm.Get_rank() == 2:
            x[1, 2] = np.nextafter(x[1, 2], np.inf)
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            out = lockstep((x, y), check=True)
            stats = lockstep_stats()
        return out[0] is x, x, stats

    out = _ranks(one_rank, size)
    for rank, (same, x, stats) in enumerate(out):
        assert same and x.tobytes() == ref.tobytes()
        buffers = _broadcast_buffers(seen, rank)
        assert len(buffers) == 1 and buffers[0].shape == ref.shape
        assert stats == {'calls': 1, 'bytes': ref.nbytes,
                         'checked_calls': 1, 'skipped_bytes': other.nbytes,
                         'audited_calls': 1,
                         'mismatched_calls': 1 if rank == 2 else 0,
                         'max_abs_diff': ulp if rank == 2 else 0.0}
    counted = {tuple(stats[k] for k in ('calls', 'bytes', 'checked_calls',
                                        'skipped_bytes'))
               for _, _, stats in out}
    assert len(counted) == 1


def test_a_checked_container_keeps_its_arrays_and_sends_its_scalars(
        monkeypatch):
    """A dict of a tuple, a list, a namedtuple and a label: the agreeing
    arrays stay where they are and come back as the same objects; the strided
    one the last rank holds differently goes through a temporary and is
    written back; the rank-dependent scalars and the label are rank 0's."""
    seen = _spy_collectives(monkeypatch)
    base0 = np.arange(24.0).reshape(4, 6)

    def one_rank(comm):
        r = comm.Get_rank()
        a, b, c = np.full(3, 2.0), np.eye(2), np.ones(2)
        base = base0 + (r == comm.Get_size() - 1)
        view = base[:, ::2]
        x = {'tuple': (a, r), 'list': [b, None], 'pair': Pair(1.5 * r, c),
             'strided': view, 'label': f'rank{r}'}
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            out = lockstep(x, check=True)
            stats = lockstep_stats()
        return (a, b, c, base, view), out, stats

    for rank, ((a, b, c, base, view), out, stats) in enumerate(
            _ranks(one_rank, SIZE)):
        assert type(out) is dict
        assert list(out) == ['tuple', 'list', 'pair', 'strided', 'label']
        assert out['tuple'][0] is a and out['tuple'][1] == 0
        assert out['list'][0] is b and out['list'][1] is None
        assert type(out['pair']) is Pair
        assert out['pair'].energy == 0.0 and out['pair'].coeff is c
        assert out['strided'] is view and out['label'] == 'rank0'
        assert np.array_equal(view, base0[:, ::2])
        assert np.array_equal(base[:, 1::2],
                              base0[:, 1::2] + (rank == SIZE - 1))
        buffers = _broadcast_buffers(seen, rank)
        assert len(buffers) == 1 and buffers[0].shape == view.shape
        assert 'bcast' in [name for name, _ in seen[rank]]   # the scalars
        assert stats['bytes'] == view.nbytes
        assert stats['skipped_bytes'] == a.nbytes + b.nbytes + c.nbytes
        # every other rank's scalars differed from rank 0's
        assert stats['mismatched_calls'] == (1 if rank else 0)


def test_serially_a_checked_lockstep_is_the_plain_no_op(monkeypatch):
    seen = _spy_collectives(monkeypatch)
    lockstep_stats(reset=True)
    x = np.arange(4.0)
    objects = (x, (x, 3), {'x': x}, 5, None)
    assert all(lockstep(o, check=True) is o for o in objects)

    def one_rank(comm):
        return all(lockstep(o, comm, check=True) is o for o in objects)

    assert _ranks(one_rank, 1) == [True]
    assert seen == {}
    stats = lockstep_stats()
    assert stats['calls'] == stats['checked_calls'] == 0
    assert stats['skipped_bytes'] == 0


def test_a_checked_lockstep_refuses_what_the_plain_one_refuses():
    """A read-only receiver raises even where every digest agrees (else only
    a drifted run would find it), and so does a call checked on some ranks
    only; both on every rank, before any buffer moves."""
    def frozen(comm):
        x = np.full(4, 1.0)
        x.flags.writeable = comm.Get_rank() != 2
        try:
            lockstep(x, check=True)
        except ValueError as exc:
            return _lockstep_refusal(exc)
        return None

    for message in _ranks(frozen, SIZE):
        assert 'rank 2 holds a read-only array' in message

    def mixed(comm):
        x = np.full(4, float(comm.Get_rank()))
        try:
            lockstep(x, check=comm.Get_rank() != 1)
        except ValueError as exc:
            return _lockstep_refusal(exc), x.copy()
        return None, x.copy()

    for rank, (message, x) in enumerate(_ranks(mixed, SIZE)):
        assert 'asked for a checked call and the others did not' in message
        assert np.array_equal(x, np.full(4, float(rank)))


def test_audit_is_inherited_and_can_be_switched_off():
    def one_rank(comm):
        r = float(comm.Get_rank())
        with distributed(comm, audit=True):
            with distributed(comm):
                lockstep(np.full(2, r))
            with distributed(comm, audit=False):
                lockstep(np.full(2, r))
        return lockstep_stats()

    for rank, stats in enumerate(_ranks(one_rank, SIZE)):
        assert stats['calls'] == 2 and stats['audited_calls'] == 1
        assert stats['mismatched_calls'] == (1 if rank else 0)


# ------------------------------------------------------------- abort guard
def test_a_simulated_comm_installs_nothing_and_reraises():
    hooks = _hooks()
    for comm in (simulated_world(2)[1], None):
        with pytest.raises(ZeroDivisionError):
            with distributed(comm):
                assert _hooks() == hooks
                1 / 0
        assert _hooks() == hooks

    def one_rank(comm):
        assert _hooks() == hooks
        if comm.Get_rank() == 1:
            raise ZeroDivisionError('rank 1')
        comm.allgather(None)                   # rank 0 waits in a collective

    with pytest.raises(ZeroDivisionError, match='rank 1'):
        _ranks(one_rank, 2)
    assert _hooks() == hooks


def test_a_wire_comm_aborts_when_an_exception_leaves_the_region(capsys):
    comm, hooks = _WireComm(), _hooks()
    with pytest.raises(RuntimeError, match='diverged'):
        with distributed(comm):
            assert current_comm() is comm
            assert sys.excepthook is not hooks[0]
            assert threading.excepthook is not hooks[1]
            raise RuntimeError('diverged')
    assert comm.aborted == [1]
    err = capsys.readouterr().err
    assert 'rank 1 of 2' in err and 'RuntimeError: diverged' in err
    assert _hooks() == hooks and current_comm() is None


def test_the_installed_excepthook_aborts(capsys):
    comm = _WireComm()
    with distributed(comm):
        try:
            raise KeyError('uncaught at top level')
        except KeyError:
            sys.excepthook(*sys.exc_info())
    assert comm.aborted == [1]
    assert 'uncaught at top level' in capsys.readouterr().err


def test_a_thread_dying_inside_the_region_aborts(capsys):
    comm = _WireComm()
    with distributed(comm):
        worker = threading.Thread(target=lambda: 1 / 0)
        worker.start()
        worker.join()
    assert comm.aborted == [1]
    assert 'ZeroDivisionError' in capsys.readouterr().err


def test_an_exception_handled_inside_the_outermost_region_aborts_nothing():
    comm = _WireComm()
    with distributed(comm):
        try:
            with distributed(comm):
                raise ValueError('handled on every rank alike')
        except ValueError:
            pass
        assert current_comm() is comm
    assert comm.aborted == []


def test_a_clean_exit_is_not_a_failure():
    comm = _WireComm()
    with pytest.raises(SystemExit):
        with distributed(comm):
            sys.exit(0)
    assert comm.aborted == []
    with pytest.raises(SystemExit):
        with distributed(comm):
            sys.exit(3)
    assert comm.aborted == [1]


def test_a_one_rank_wire_comm_installs_nothing():
    comm, hooks = _WireComm(size=1, rank=0), _hooks()
    with pytest.raises(RuntimeError):
        with distributed(comm):
            assert _hooks() == hooks
            raise RuntimeError('alone')
    assert comm.aborted == []


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
