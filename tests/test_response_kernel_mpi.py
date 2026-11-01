"""The orbital-response kernel and the tiled ADC(2) force's Z-vector under
real ranks.

    python tests/test_response_kernel_mpi.py                          # one rank
    OMP_NUM_THREADS=1 mpirun -n 2 python tests/test_response_kernel_mpi.py
    OMP_NUM_THREADS=1 mpirun -n 3 python tests/test_response_kernel_mpi.py

Script, not a pytest module: it initializes MPI, which the sandboxed test
runner cannot, and it exits with a status; tests/test_response_kernel.py gates
the same properties over simulated ranks. Every rank's verdict is gathered.

Checks (water cc-pVDZ and 6-31G as tests/test_response_kernel.py):
  * the kernel's general and factored builds on this rank equal the serial
    ones bit for bit, from its own auxiliary blocks (round-robin);
  * the ADC(2) S1 tiled force with its Z-vector shared over the ranks equals
    the serial force to a few ulp on every rank.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import scf

from src.Base.pyscf.cphf import ResponseKernel
from src.Base.utils.parallel.mpi_grid import distributed, grid_comm
from src.gradients.adc import ee_gradient as eg
from tests.adc_common import ATOMS, AUX, NF, molecule
from tests.pins import RECORDING_BLAS, matches
from tests.test_distributed_fit_mpi import Gate, serial
from tests.test_response_kernel import BLOCK, MF, P, X


def force(mf, mol, comm):
    return eg.adc2_force_tiled(mf, mol, 'singlet', frozen=NF, auxbasis=AUX, comm=comm,
                               tile_gb=1e-9)[0]


def main(comm):
    """Every check on this rank of `comm` (None serially); 0 when every rank
    passed."""
    gate = Gate(comm)
    gate.section('response kernel')
    ref = ResponseKernel(MF, block=BLOCK)
    want = (ref(X), ref.factored(P))
    with distributed(comm):
        k = ResponseKernel(MF, comm, block=BLOCK)
        got = (k(X), k.factored(P))
    size = 1 if comm is None else comm.Get_size()
    gate.check(all(np.array_equal(a, b) for a, b in zip(got, want)),
               'general and factored G: the serial bits')
    gate.check(k.mine == list(range(gate.rank, k._nblocks(), size)),
               'this rank builds its own blocks', f'{k.mine}')
    gate.section('ADC(2) S1 tiled force, Z-vector over the ranks')
    mol = molecule(ATOMS)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-12, 1e-10, 0
    mf.kernel()
    F0 = serial(force, mf, mol, None)
    with distributed(comm):
        F = force(mf, mol, comm)
    gate.check(matches(F, F0, RECORDING_BLAS, quantity='gradient'),
               'the serial force', f'max |dF| {float(np.max(np.abs(F - F0))):.1e}')
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
