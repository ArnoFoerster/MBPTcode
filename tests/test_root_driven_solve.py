"""`krylov.root_driven_solve` runs its solver in a serial region.

Only rank 0 iterates; the other ranks wait in a broadcast for its next block.
A kernel inside the solver that falls back to `current_comm()` would pair its
collective with that wait. The shared Casida Davidson's pair-row loop does
(its serial `PairRows` locksteps the subspace solution on the current comm),
so before the serial region fragment_bse's full-BSE block solve
(`_lowest_pencil`) hung under real MPI ranks and failed under simulated ones.

Gated, at 2 and 3 simulated ranks inside `distributed(comm)`:

  * the Casida Davidson driven from rank 0 returns rank 0's roots on every
    rank, the serial roots bitwise (rank 0 runs the serial solve), and every
    rank applied the operator the same number of times;
  * `act` applies the operator in the caller's region: an operator that reads
    `current_comm()` sees the ranks' comm on every rank.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils.linear_algebra.krylov import root_driven_solve
from src.Base.utils.parallel.mpi_grid import current_comm, distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.LinearResponse.davidson import run_davidson

#: Occupied and virtual orbitals of the model Casida problem, and its roots.
NO, NV, NROOTS = 4, 12, 3
CONV_TOL = 1e-9


def model():
    """A = D + 2 V^T V, B = 2 V^T V on NO x NV pairs: (A - B) and (A + B)
    positive definite."""
    rng = np.random.default_rng(11)
    d = (np.linspace(0.4, 2.0, NV)[None, :]
         - np.linspace(-1.2, -0.3, NO)[:, None])
    v = 0.05 * rng.standard_normal((6, NO * NV))
    k = 2.0 * v.T @ v
    return d, np.diag(d.ravel()) + k, k


def solve(act, d, n_pair):
    """The Casida Davidson on a block action through `act`, (2 n_pair, k)
    stacked [x; y] to [A x + B y; B x + A y]."""
    def apply_ab(z):
        cols = np.ascontiguousarray(z.reshape(len(z), n_pair).T)
        out = act(np.vstack([cols, np.zeros_like(cols)]))
        return (out[:n_pair].T.reshape(z.shape),
                out[n_pair:].T.reshape(z.shape))
    om, x, y = run_davidson(apply_ab, d, NROOTS, CONV_TOL, 100, None,
                            preconditioner='bare')[:3]
    return om, x, y


def ranked(comm):
    d, a, b = model()
    n_pair = d.size
    k = np.block([[a, b], [b, a]])
    seen = []

    def apply(x):
        seen.append(current_comm() is comm)
        return k @ x
    with distributed(comm):
        om, x, y = root_driven_solve(lambda act: solve(act, d, n_pair), apply,
                                     2 * n_pair)
    return om, x, y, seen


@pytest.mark.parametrize('size', [2, 3])
def test_the_casida_davidson_driven_from_rank_zero(size):
    d, a, b = model()
    k = np.block([[a, b], [b, a]])
    om_s = solve(lambda x: k @ x, d, d.size)[0]
    out = run_simulated(ranked, size)
    om0 = out[0][0]
    assert np.array_equal(om0, om_s)
    for om, x, y, seen in out:
        assert np.array_equal(om, om0)
        assert np.array_equal(x, out[0][1]) and np.array_equal(y, out[0][2])
        assert len(seen) == len(out[0][3]) > 0
        assert all(seen)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
