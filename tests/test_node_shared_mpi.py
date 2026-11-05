"""Node-shared arrays built many times over real ranks hold no MPI contexts.

    python tests/test_node_shared_mpi.py                 # one rank
    OMP_NUM_THREADS=1 srun --mpi=pmi2 -n 4 python tests/test_node_shared_mpi.py

Script, not a pytest module: it initializes MPI. A long run (a geometry walk
over ranks, a solvent loop) builds a `NodeShared` array per fold call. Each
one used to split a new node communicator and leave it and its window
allocated, and an MPI process has only a few thousand contexts (MPICH: 2048),
so the run aborted after about a thousand calls. Here: more constructions than
that limit, each filled by the node's first rank and read on every rank; the
node communicator is split once per parent communicator; the windows held stay
as few as the arrays alive; a window whose array (or only a view of it) is
still held on some rank of the node is kept, with its values, until no rank
holds it; the same on a sub-communicator. Every rank's verdict is gathered.
Run it on at least two nodes with several ranks per node.
"""
import gc
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np                                           # noqa: E402

from src.Base.utils.parallel import mpi_grid                 # noqa: E402
from src.Base.utils.parallel.mpi_grid import NodeShared, grid_comm  # noqa: E402
from tests.test_distributed_fit_mpi import Gate              # noqa: E402

# more than MPICH's 2048 contexts per process, which the old construction
# (a communicator and a window each time) used up in about a thousand
BUILDS = 2500


def filled(shape, comm, value):
    """A NodeShared array of `shape` holding `value`, readable on every rank."""
    sh = NodeShared(shape, comm)
    if sh.fills:
        sh.array[...] = value
    sh.ready()
    return sh


def held(comm):
    """How many windows the node of `comm` still holds."""
    return len(mpi_grid._node_of(comm).windows)


def many(gate, comm, label):
    """BUILDS constructions through the public interface alone (before the fix
    MPI aborts in this loop), then the windows the node still holds."""
    gate.section(f'{BUILDS} constructions, {label}')
    first, wrong, same = None, 0, True
    for k in range(BUILDS):
        sh = filled((7, 3), comm, float(k))
        wrong += int(not np.all(sh.array == k))
        first = sh.node if first is None else first
        same = same and sh.node is first
        del sh
    gate.check(same, 'the node communicator is split once and reused')
    gate.check(wrong == 0, 'every rank reads the value its node\'s first rank wrote',
               f'{wrong} wrong')
    gate.check(held(comm) <= 1, 'only the window dropped last is still held',
               f'{held(comm)} held')


def kept(gate, comm):
    gate.section('arrays still held are kept')
    node = mpi_grid._node_of(comm).comm
    keep = filled((5, 4), comm, -1.0)
    view = keep.array[1:, 2]
    del keep
    for k in range(50):
        filled((3,), comm, float(k)).ready()
    gate.check(np.all(view == -1.0), 'a view of a dropped NodeShared keeps its memory')
    gate.check(held(comm) == 2, 'its window is held beside the last one dropped',
               f'{held(comm)} held')
    del view
    gc.collect()
    filled((3,), comm, 0.0).ready()
    gate.check(held(comm) == 1, 'once dropped everywhere its window is freed',
               f'{held(comm)} held')
    one = filled((6,), comm, 2.5)
    mine = one.array if node.Get_rank() == 0 else None
    del one
    for k in range(5):
        filled((3,), comm, float(k)).ready()
    seen = mine is None or bool(np.all(mine == 2.5))
    gate.check(seen and held(comm) == 2,
               'a window one rank of the node still views is kept on every rank',
               f'{held(comm)} held')
    del mine
    filled((3,), comm, 0.0).ready()
    gate.check(held(comm) == 1, 'and freed once that rank drops it', f'{held(comm)} held')


def main(comm):
    gate = Gate(comm)
    gate.section('serial')
    sh = NodeShared((2, 3), None)
    gate.check(sh.fills and sh.node is None and sh.array.shape == (2, 3),
               'without a communicator the array is private and this rank fills it')
    if comm is None or comm.Get_size() == 1:
        gate.say('one rank: no shared windows to build')
        return gate.finish()
    gate.info(f'{comm.Get_size()} ranks on '
              f'{len(set(gate.everyone(os.uname().nodename)))} node(s)')
    many(gate, comm, 'the world')
    node = mpi_grid._node_of(comm).comm
    kept(gate, comm)
    sub = comm.Split(comm.Get_rank() % 2, comm.Get_rank())
    if min(gate.everyone(sub.Get_size())) > 1:
        many(gate, sub, 'a sub-communicator of every other rank')
        gate.check(mpi_grid._node_of(sub).comm is not node,
                   'a sub-communicator has its own node communicator')
    else:
        gate.say('fewer than four ranks: no sub-communicator section')
    sub.Free()
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
