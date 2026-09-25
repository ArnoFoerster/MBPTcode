"""The row-split ISDF block action reproduces the serial one, rank by rank.

`isdf_block_action(comm=...)` gives each rank a contiguous block of the rows of
Zt and every term of the action follows that block: the two exchange terms and
the Hartree term are sums over the row index and ride one reduction of the
batch, while z X_v^T is built as an OUTPUT PARTITION over the same rows and
all-gathered, and the intermediate X_o^T (Zt * P) is reduced so that each rank
contracts only its own columns with X_v.

That mix is why the ranks are run here as THREADS of one process
(`mpi_grid.run_simulated`), with the real collectives, rather than through a
communicator whose all-reduce is a no-op: a gather has no partial to inspect,
and a rank handed nothing from the others computes with an uninitialized grid
rather than with a contribution that could be added up afterwards. (mpi4py's
MPI_Init cannot run under the sandbox these tests run in, so the real
multi-rank check is `tests/test_mpi_routes.py` under mpirun; this one gates the
algebra.)

The system is small and deliberately awkward: 37 grid rows against a tile of
six, so a rank's block spans several tiles and the last tile of each block is
short, and 37 rows do not divide evenly among 2, 3 or 5 ranks.

What is gated:
  * the serial action (comm=None) is bitwise what it was: `Az`, `Bz` for
    singlet and triplet BSE, TDHF and RPA against a fixed random system
    (tests/test_block_action_split.py, against the archived commit);
  * for 2, 3 and 5 simulated ranks, every rank's action agrees with the serial
    one at 1e-13 relative -- the row split re-associates the sums it reduces --
    for the BSE, the TDHF and the RPA kernel, singlet and triplet;
  * every rank returns the same bits, which a reduced and gathered result
    cannot fail to unless a rank is following a different iteration;
  * surplus ranks (more ranks than rows) own an empty block and still return
    the serial answer;
  * `contiguous_block` covers every row exactly once;
  * the DF action, which is not divided, runs whole on every rank under a
    rank count, passed or read off the `distributed` region around it, and
    returns the serial A and B bitwise.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils.mpi_grid import contiguous_block, run_simulated
from src.SingleReference.LinearResponse.davidson import (_block_action,
                                                         isdf_block_action)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

M, NORB, NOCC, NAUX, NVEC = 37, 9, 4, 7, 3
#: six grid rows per tile, so a rank's block spans several tiles and the last
#: tile of each block is short
TILE_GB = 6 * M * 8 / 1e9
REL = 1e-13
#: (lBSE, screened, spin) of every kernel this action serves.
KERNELS = [('bse', True, True, 'singlet'), ('bse', True, True, 'triplet'),
           ('tdhf', True, False, 'singlet'), ('rpa', False, False, 'singlet')]


def system(seed=5):
    rng = np.random.default_rng(seed)
    eps = np.sort(rng.normal(size=NORB))
    eps[NOCC:] += 2.0
    X = 0.3 * rng.normal(size=(M, NORB))
    D = 0.3 * rng.normal(size=(M, NAUX))
    W = rng.normal(size=(NAUX, NAUX))
    W = 0.5 * (W + W.T) + NAUX * np.eye(NAUX)
    z = rng.normal(size=(NVEC, NOCC, NORB - NOCC))
    lr = LinearResponseSolver(eps, spin_mode='restricted')
    return lr, X, D, W, z


def action(lr, X, D, W_aux, lBSE, spin, comm=None):
    apply_AB, _ = isdf_block_action(lr, NOCC, lBSE, W_aux, (X, D),
                                    tile_memory_gb=TILE_GB, spin=spin,
                                    comm=comm)
    return apply_AB


def relative(got, ref):
    scale = max(np.abs(ref).max(), 1e-300)
    return np.abs(np.asarray(got) - np.asarray(ref)).max() / scale


def test_contiguous_blocks_cover_every_row_once():
    for n in (1, 5, 37, 100):
        for size in (1, 2, 3, 5, 64):
            blocks = [contiguous_block(n, r, size) for r in range(size)]
            rows = np.concatenate([np.arange(a, b) for a, b in blocks])
            assert np.array_equal(rows, np.arange(n))
            assert all(b >= a for a, b in blocks)
            lengths = [b - a for a, b in blocks]
            assert max(lengths) - min(lengths) <= 1


@pytest.mark.parametrize('size', [2, 3, 5, M + 3])
@pytest.mark.parametrize('name,lBSE,screened,spin', KERNELS)
def test_row_split_reproduces_the_serial_action(size, name, lBSE, screened,
                                                spin):
    """Every rank's A and B against the serial ones, and against each other.

    The largest size is more ranks than grid rows: those ranks own an empty
    block, contribute nothing to any of the sums and still come back with the
    whole answer.
    """
    lr, X, D, W, z = system()
    W_aux = W if screened else None
    A0, B0 = action(lr, X, D, W_aux, lBSE, spin)(z)

    def one_rank(comm):
        return action(lr, X, D, W_aux, lBSE, spin, comm=comm)(z)

    out = run_simulated(one_rank, size)
    for A, B in out:
        assert relative(A, A0) < REL
        assert relative(B, B0) < REL
        assert np.array_equal(A, out[0][0]) and np.array_equal(B, out[0][1])


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('name,lBSE,screened,spin', KERNELS)
def test_df_action_runs_whole_on_every_rank(size, name, lBSE, screened, spin):
    """The DF action is not divided: under a comm, passed or read off the
    `distributed` region around it, every rank applies the serial action, so
    A and B are the serial ones bitwise -- nothing is re-associated."""
    lr, X, D, W, z = system()
    lr_df = LinearResponseSolver(lr.eps,
                                 coeff_df=np.einsum('kp,kq,kA->Apq', X, X, D),
                                 spin_mode='restricted')
    mode = {'bse': 'BSE', 'tdhf': 'TDHF', 'rpa': 'RPA'}[name]
    W_aux = W if screened else None
    A0, B0 = _block_action(lr_df, NOCC, mode, W_aux, None, spin=spin)[0](z)

    def one_rank(comm, explicit):
        act, _ = _block_action(lr_df, NOCC, mode, W_aux, None, spin=spin,
                               comm=comm if explicit else None)
        return act(z.copy())

    for explicit in (True, False):
        for A, B in run_simulated(one_rank, size, explicit):
            assert A.tobytes() == A0.tobytes() and B.tobytes() == B0.tobytes()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
