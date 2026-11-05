"""The one-electron spin-orbit operator of the exact two-component (X2C)
Hamiltonian, built once over the ranks.

The decoupling is one complex generalized eigenproblem of four times the
uncontracted basis (pySCF's `SpinOrbitalX2CHelper`); the one-electron
integrals it starts from are under 1% of its time, so the step is not divided.
Under a communicator rank 0 runs it and `lockstep` sends the (3, nao, nao)
result to every rank, bit for bit. While it runs, rank 0's threads are spread
over the cores of every rank on its node and those ranks wait without spinning
(`mpi_grid.on_root_with_node_cpus`): the reduction to tridiagonal form is bound
by memory bandwidth, so the other NUMA domains' memory controllers are what
rank 0 gains, and the bits do not move, since its thread count does not.
Outside a communicator it is the plain serial build, the same bits.
"""
import numpy as np
from pyscf.x2c import sfx2c1e, x2c

from src.Base.utils.parallel.mpi_grid import (current_comm, lockstep,
                                              on_root_with_node_cpus)


def one_electron_term(mol, comm=None):
    """(3, nao, nao) real antisymmetric X2C spin-orbit operator, Hartree, the
    same bits on every rank.

    comm: defaults to `current_comm()`; serial where there is none.
    """
    comm = current_comm() if comm is None else comm
    if comm is None or comm.Get_size() == 1:
        return _x2c_build(mol)
    h = on_root_with_node_cpus(lambda: _x2c_build(mol), comm)
    if h is None:
        nao = mol.nao_nr()
        h = np.zeros((3, nao, nao))
    return lockstep(h, comm)


def _x2c_build(mol):
    """The X2C operator of `one_electron_term`, on this process."""
    nao = mol.nao_nr()
    soc = (x2c.SpinOrbitalX2CHelper(mol).get_hcore(mol)
           - np.kron(np.eye(2), sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)))
    h = np.array([soc[:nao, nao:].imag, soc[:nao, nao:].real,
                  soc[:nao, :nao].imag])
    # antisymmetric by construction; the X2C decoupling leaves ~1e-11 of noise
    # on that, and the vanishing diagonal is what lets the interstate density
    # skip its i = j, a = b mask
    return 0.5 * (h - h.transpose(0, 2, 1))
