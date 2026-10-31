"""The photophysics record store under ranks: every rank sees the record that
rank 0 just wrote, whatever its own view of the filesystem.

    python tests/test_photophysics_store_mpi.py           # one rank
    OMP_NUM_THREADS=1 mpirun -n 2 python tests/test_photophysics_store_mpi.py
    OMP_NUM_THREADS=1 mpirun -n 3 python tests/test_photophysics_store_mpi.py

The same `main(comm)` runs over thread-ranks in pytest (`run_simulated`), and
over real ranks as a script that exits with a status; every rank's verdict is
gathered. Only rank 0 touches the disk: the other ranks get what it found, so a
rank that sees a stale or empty directory, or reads while rank 0 is replacing
the record, cannot fail.

Checks:
  * a record written and at once read, found and listed on every rank
  * the same with the other ranks' view of the directory empty (as on a
    network filesystem that has not yet shown the new file)
  * a record rewritten many times while the other ranks read slowly: no rank
    reads the arrays file that the rewrite replaced
  * a missing record raises on every rank, and so does a record of another
    stamp
  * `set_aside` and `write_xyz` agree on every rank
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import numpy as np
import pytest

import src.Base.utils.record_io as record_io
from src.Base.utils.parallel.mpi_grid import distributed, grid_comm, is_root
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.photophysics.store import Store, StoreRefused
from tests.test_distributed_fit_mpi import Gate

DECLARATION = types.SimpleNamespace(
    system=types.SimpleNamespace(name='h2'), environment_dir='gas',
    gas=True, environment=None)
STAMP = {'basis': 'sto-3g'}
OTHER_STAMP = {'basis': 'cc-pvdz'}
REWRITES = 6


class Molecule:
    natm = 2

    @staticmethod
    def atom_coords():
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])

    @staticmethod
    def atom_pure_symbol(ia):
        return 'H'


def blind(store, root):
    """`store` with this rank's own view of the run's directory empty, unless
    it is rank 0's."""
    if not is_root():
        store.dir = root / 'not_there'
    return store


def slow_arrays(delay):
    """`np.load` that waits before opening on every rank but 0: the window
    between a rank reading the record's tree and opening its arrays file."""
    load = np.load

    def slowly(*args, **kwargs):
        if not is_root():
            time.sleep(delay)
        return load(*args, **kwargs)
    return load, slowly


def main(comm, root, delay=0.05):
    gate = Gate(comm)
    store = Store(root, DECLARATION)
    record = {'e': np.arange(5.0), 'flags': {}}
    gate.section('write, then read on every rank')
    store.write('vert', record, STAMP)
    got = store.read('vert', STAMP)
    gate.check(np.array_equal(got['e'], record['e']), 'the record on this rank')
    gate.check(store.has('vert') and store.reusable('vert', STAMP)
               and 'vert.json' in store.names(), 'found and listed')

    gate.section("the other ranks' directory empty")
    eyes = blind(Store(root, DECLARATION), root)
    gate.check(eyes.has('vert') and eyes.reusable('vert', STAMP),
               'found through rank 0')
    gate.check(np.array_equal(eyes.read('vert', STAMP)['e'], record['e'])
               and 'vert.json' in eyes.names(), 'read and listed through rank 0')

    gate.section('rewritten while the other ranks read slowly')
    original, slowly = slow_arrays(delay)
    record_io.np.load = slowly          # rank 0 passes straight through
    try:
        for i in range(REWRITES):
            store.update('vert', STAMP, flags={'step': i})
            gate.check(store.read('vert', STAMP)['flags'] == {'step': i},
                       f'rewrite {i} seen on this rank')
    finally:
        store.barrier()
        record_io.np.load = original

    gate.section('refusals reach every rank')
    try:
        store.read('missing', STAMP)
        gate.check(False, 'a missing record raises')
    except FileNotFoundError:
        gate.check(True, 'a missing record raises')
    try:
        store.read('vert', OTHER_STAMP)
        gate.check(False, 'another stamp refuses')
    except StoreRefused:
        gate.check(True, 'another stamp refuses')

    gate.section('set_aside and write_xyz')
    store.write_xyz('s0.xyz', Molecule(), 'h2')
    moved = eyes.set_aside(['s0.xyz', 'absent'], 'old')
    gate.check(moved == ['s0.xyz'], 'the names moved are the same everywhere')
    gate.check(store.has('old/s0.xyz') or 's0.xyz' not in store.names(),
               's0.xyz no longer in the run directory')
    return gate.finish()


@pytest.mark.parametrize('size', [2, 3])
def test_store_read_after_write_over_ranks(tmp_path, size):
    assert run_simulated(main, size, tmp_path) == [0] * size


def test_store_read_after_write_one_rank(tmp_path):
    assert main(None, tmp_path) == 0


if __name__ == '__main__':
    import pathlib
    import tempfile
    comm = grid_comm()[0]
    made = tempfile.mkdtemp(prefix='store_ranks_') \
        if comm is None or comm.Get_rank() == 0 else None
    root = made if comm is None else comm.bcast(made)
    with distributed(comm):
        status = main(comm, pathlib.Path(root))
    sys.exit(status)
