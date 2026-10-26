"""The two-electron spin-orbit mean-field term under real ranks.

    python tests/test_spin_orbit_mpi.py                          # one rank
    OMP_NUM_THREADS=1 mpirun -n 2 python tests/test_spin_orbit_mpi.py
    OMP_NUM_THREADS=1 mpirun -n 3 python tests/test_spin_orbit_mpi.py

Script, not a pytest module: it initializes MPI, which the sandboxed test
runner cannot, and it exits with a status. `main(comm)` takes the
communicator, so `run_simulated(main, n)` runs the same checks over
thread-ranks (tests/test_spin_orbit_ranks.py does). Every rank's verdict is
gathered.

Checks, on formaldehyde (cc-pVDZ, RHF), for SOMF and for AMFI, with every
rank but 0 handed its own density (rank 0's plus 1e-9 r, the way two nodes'
SCFs differ in their last bits, here far beyond them):
  * the integral pass ran on rank 0 and on no other rank (one build for SOMF,
    one per atom for AMFI);
  * every rank holds rank 0's spin-orbit operator, bit for bit, and that is
    the serial operator of rank 0's density, bit for bit;
  * the operator in the orbital basis (`soc_operator_mo`) is the serial one
    on every rank, bit for bit, on rank 0's mean field (`lockstep_mean_field`,
    as a chain's kernels hold it; the transform reads each rank's orbitals).
"""
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.utils.parallel.mpi_grid import (distributed, grid_comm,
                                             lockstep_mean_field)
from src.gradients import spin_orbit_mean_field
from src.properties.spin_orbit import soc_operator, soc_operator_mo
from tests.test_distributed_fit_mpi import Gate, bitwise, serial

CH2O = ('C 0 0 -0.5296; O 0 0 0.6771; H 0 0.9343 -1.1163; '
        'H 0 -0.9343 -1.1163')

#: How far every rank r > 0 moves its density off rank 0's: 1e-9 r.
DENSITY_SKEW = 1e-9


def formaldehyde():
    """(mol, mf) at a converged RHF, serially on every rank."""
    mol = gto.M(atom=CH2O, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    with distributed(None):
        mf.kernel()
    return mol, mf


#: The integral pass and this rank's count of its calls (thread-local, so
#: that thread-ranks count their own).
_REAL_JK = spin_orbit_mean_field._soc_jk
_BUILDS = threading.local()


def _counted_jk(mol, dm):
    if getattr(_BUILDS, 'on', False):
        _BUILDS.n += 1
    return _REAL_JK(mol, dm)


def builds_of(fn, *args):
    """(fn(*args), the integral passes it ran on this rank)."""
    _BUILDS.on, _BUILDS.n = True, 0
    try:
        return fn(*args), _BUILDS.n
    finally:
        _BUILDS.on = False


def main(comm):
    """Every check on this rank of `comm` (None serially); 0 when every rank
    passed."""
    gate = Gate(comm)
    # installed by every rank, the same function each time: no rank undoes
    # another's
    spin_orbit_mean_field._soc_jk = _counted_jk
    with distributed(comm):
        mol, mf = formaldehyde()
        # rank 0's mean field on every rank, as the kernels of a chain make it
        lockstep_mean_field(mf)
        dm = mf.make_rdm1()
        mine = dm + DENSITY_SKEW * gate.rank
        for two_electron in ('somf', 'amfi'):
            gate.section(f'two_electron={two_electron!r}')
            reference = serial(soc_operator, mol, dm, two_electron)
            h, builds = builds_of(soc_operator, mol, mine, two_electron)
            want = (mol.natm if two_electron == 'amfi' else 1) \
                if gate.rank == 0 else 0
            per_rank = gate.everyone(builds)
            gate.check(builds == want, 'integral pass on rank 0 only',
                       f'builds per rank {per_rank}')
            gate.check(bitwise((h,), (gate.everyone(h)[0],)),
                       "rank 0's operator on this rank")
            gate.check(bitwise((h,), (reference,)),
                       "the serial operator of rank 0's density",
                       f'{abs(h - reference).max():.1e}')
            mo = soc_operator_mo(mf, mol, two_electron)
            mo_reference = serial(soc_operator_mo, mf, mol, two_electron)
            gate.check(bitwise((mo,), (mo_reference,)),
                       'the serial operator in the orbital basis')
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
