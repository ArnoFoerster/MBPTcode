"""The four-centre exchange skeleton, its atoms split over real ranks.

    python tests/test_exchange_skeleton_mpi.py                 # one rank
    OMP_NUM_THREADS=1 mpirun -n 2 python tests/test_exchange_skeleton_mpi.py
    OMP_NUM_THREADS=1 mpirun -n 3 python tests/test_exchange_skeleton_mpi.py

Script, not a pytest module: it initializes MPI, which the sandboxed test
runner cannot, and it exits with a status. `main(comm)` takes the
communicator; tests/test_exchange_skeleton_split.py gates the same properties
in pytest over thread-ranks. Every rank's verdict is gathered: a check that
fails on rank 2 fails the run.

Checks, formaldehyde and acrolein/aug-cc-pVDZ, random symmetric densities
(the same on every rank), omega 0 and 0.3:
  * `exchange_channel_skeleton` over the ranks against two separate
    `pyscf.grad.rhf.get_jk` passes on this rank: within 1e-12 relative (one
    pass screens on both densities at once, so a quartet near the screening
    bound can move the last bits);
  * the same bits on every rank.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np                                           # noqa: E402
from pyscf import gto                                        # noqa: E402
from pyscf.grad import rhf as grad_rhf                       # noqa: E402

from src.Base.utils.parallel.mpi_grid import distributed, grid_comm  # noqa: E402
from src.gradients.mean_field_response import exchange_channel_skeleton  # noqa: E402
from tests.test_distributed_fit_mpi import Gate, bitwise, serial  # noqa: E402

#: The bar against two separate pyscf passes, relative to the largest element.
SPLIT_REL_TOL = 1e-12

MOLECULES = {
    'formaldehyde': 'C 0 0 0; O 0 0 1.205; H 0 0.94 -0.59; H 0.02 -0.94 -0.6',
    'acrolein': ('C -0.1505 -0.7423 0; C 1.0937 -0.2357 0; C 2.2770 -1.1149 0; '
                 'O 3.4006 -0.6460 0; H -1.0412 -0.1263 0; H -0.2913 -1.8186 0; '
                 'H 1.2550 0.8399 0; H 2.0850 -2.2012 0'),
}


class _Mf:
    def __init__(self, mol):
        self.mol = mol


def densities(nao):
    rng = np.random.default_rng(11)
    a, b = rng.standard_normal((2, nao, nao))
    return a + a.T, b + b.T


def two_passes(mol, g_ao, D, omega, weight):
    with mol.with_range_coulomb(omega):
        vk_D = grad_rhf.get_jk(mol, D)[1]
        vk_g = grad_rhf.get_jk(mol, g_ao)[1]
    t = -weight * (np.einsum('ac,xac->xa', g_ao, vk_D)
                   + np.einsum('ad,xad->xa', D, vk_g))
    return np.array([t[:, p0:p1].sum(axis=1) for _, _, p0, p1 in mol.aoslice_by_atom()])


def main(comm):
    gate = Gate(comm)
    with distributed(comm):
        for name, atom in MOLECULES.items():
            mol = gto.M(atom=atom, basis='aug-cc-pvdz', verbose=0)
            D, g_ao = densities(mol.nao)
            for omega in (0.0, 0.3):
                gate.section(f'{name}, omega {omega}')
                split = exchange_channel_skeleton(_Mf(mol), g_ao, D, omega, 0.7)
                reference = serial(two_passes, mol, g_ao, D, omega, 0.7)
                rel = float(np.abs(split - reference).max() / np.abs(reference).max())
                gate.check(rel <= SPLIT_REL_TOL, 'two separate pyscf passes',
                           f'{rel:.1e} relative')
                everyone = gate.everyone(split)
                gate.check(all(bitwise([g], [everyone[0]]) for g in everyone),
                           'the same bits on every rank')
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
