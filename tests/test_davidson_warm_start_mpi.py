"""The warm-started Casida Davidson under real ranks.

    python tests/test_davidson_warm_start_mpi.py                 # one rank
    OMP_NUM_THREADS=1 mpirun -n 2 python tests/test_davidson_warm_start_mpi.py
    OMP_NUM_THREADS=1 mpirun -n 3 python tests/test_davidson_warm_start_mpi.py

Script, not a pytest module: it initializes MPI, which the sandboxed test
runner cannot, and it exits with a status. `main(comm)` takes the
communicator, so `run_simulated(main, n)` runs the same checks over
thread-ranks; tests/test_davidson_warm_start.py gates the same properties in
pytest over simulated ranks. Every rank's verdict is gathered.

Checks, on the Casida model of tests/test_distributed_trial_space.py, the
trial space cut by pair rows over the ranks:
  * from its own converged roots (the serial solve's) the solve ends in its
    first cycle, the guess kept whole, on every rank;
  * every rank returns the same roots, bit for bit, within the Davidson's
    resolution (`roots_resolution`) of the cold serial ones;
  * from a neighbouring problem's roots it takes fewer cycles than the cold
    serial solve of that problem, to its roots.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import numpy as np

from src.Base.utils.mpi_grid import distributed, grid_comm
from tests.test_davidson_warm_start import close, neighbour
from tests.test_distributed_fit_mpi import Gate, bitwise, serial
from tests.test_distributed_trial_space import NOCC, model_problem, root


def main(comm):
    """Every check on this rank of `comm` (None serially); 0 when every rank
    passed."""
    gate = Gate(comm)
    with distributed(comm):
        apply_AB, diag, eps, _ = model_problem()
        # the roots and the tracked Ritz pairs above them: a guess of the
        # roots alone leaves the pairs above to be found again
        held = {}
        om, X, Y, _, cycles, _ = serial(root, apply_AB, diag, tracked=held)
        X, Y = held['X'], held['Y']
        gate.section('warm start from its own roots')
        warm = root(apply_AB, diag, comm=comm, guess=(X, Y))
        gate.check(warm[3].all() and warm[4] == 1,
                   'converged in the first cycle', f'{warm[4]} of {cycles}')
        gate.check(warm[5]['warm_start'] == X.shape[1], 'the guess kept whole')
        gate.check(bitwise((warm[0],), (gate.everyone(warm[0])[0],)),
                   "rank 0's roots on this rank")
        gate.check(close(warm[0], om, eps, NOCC), 'the cold roots',
                   f'{np.linalg.norm(warm[0] - om) / np.linalg.norm(om):.1e}')
        gate.section('warm start on a neighbouring problem')
        moved, _ = neighbour()
        om_m, _, _, _, cold_m, _ = serial(root, moved, diag)
        warm = root(moved, diag, comm=comm, guess=(X, Y))
        gate.check(warm[3].all() and warm[4] < cold_m,
                   'fewer cycles than cold', f'{warm[4]} against {cold_m}')
        gate.check(close(warm[0], om_m, eps, NOCC), 'its cold roots')
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
