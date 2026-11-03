"""The quasiparticle solve's grid-tile sweeps over real MPI ranks, across nodes.

    python tests/test_gw_tiles_mpi.py                     # one rank
    OMP_NUM_THREADS=1 srun --mpi=pmi2 -N 2 -n 4 python tests/test_gw_tiles_mpi.py

Script, not a pytest module: it initializes MPI. The simulated-rank gate is
tests/test_gw_tiles_distributed.py; this one runs the same sweeps on the wire,
where tiles travel between processes and nodes:

  * W(0) (`static_screening_matrix`) from `SlicedFactors` is the one-rank
    W every rank computes on its own, bitwise;
  * the self-energy's branch sums (`self_energy_branch_sums_tiles`) from
    sliced factors are the one-rank sums, bitwise;
  * the in-core quasiparticle window on sliced factors is the window on the
    replicated factors at this rank count, bitwise, and every rank's is
    rank 0's;
  * the grid tiles a rank holds (`tile_ledger`) are at most 1.5 times its
    share of the factors.

Every rank's verdict is gathered; the per-rank peak resident memory is
printed beside the checks. Bits across nodes of different CPUs need a
conditional-numerical-reproducibility BLAS setting (MKL_CBWR) or nodes of one
type.
"""
import os
import resource
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np                                           # noqa: E402
from pyscf import gto, scf                                   # noqa: E402

from src.Base.utils.parallel.mpi_grid import (distributed, grid_comm,  # noqa: E402
                                              lockstep_mean_field)
from src.SingleReference.GW.space_time import separable_factors  # noqa: E402
from src.SingleReference.LinearResponse import space_time   # noqa: E402
from tests.isdf_test_grid import g1                          # noqa: E402
from tests.test_distributed_fit_mpi import Gate              # noqa: E402
from tests.test_gw_tiles_distributed import (                # noqa: E402
    AUXBASIS, BASIS, DIMER, NUM, TILE_GB, model_wt, sigma_sums, sliced)
from src.Base.factorization.sliced_factors import tile_ledger  # noqa: E402
from src.SingleReference.GW.space_time import solve_qp_energy_space_time  # noqa: E402


class _One:
    """A communicator of one rank for the serial references."""

    def Get_size(self):
        return 1

    def Get_rank(self):
        return 0


def main(comm):
    gate = Gate(comm)
    size = gate.size
    gate.section('grid tiles')
    mol = gto.M(atom=DIMER, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.kernel()
    nocc = mol.nelectron // 2
    with distributed(comm):
        if comm is not None:
            lockstep_mean_field(mf, comm)
        factors = separable_factors(mf, mol, auxbasis=AUXBASIS, counts=g1(mol))
    X, D = factors[:2]
    with distributed(None):
        W_one = space_time.static_screening_matrix(
            X, D, mf.mo_energy, nocc, comm=None, tile_memory_gb=TILE_GB)
        Wt, tau = model_wt(D.shape[1])
        sig_one = sigma_sums(_One(), factors, mf, nocc, Wt, tau, None, False)[0]
    if comm is None or size == 1:
        gate.say('one rank: the references alone')
        return gate.finish()
    gate.info(f'{size} ranks on {len(set(gate.everyone(os.uname().nodename)))}'
              ' node(s)')
    with distributed(comm):
        part = sliced(factors, comm)
        with tile_ledger() as led:
            W = space_time.static_screening_matrix(
                part, None, mf.mo_energy, nocc, tile_memory_gb=TILE_GB)
        gate.check(np.array_equal(W, W_one),
                   'W(0) from slices is the one-rank W, bitwise')
        sig = sigma_sums(comm, factors, mf, nocc, Wt, tau, None, True)[0]
        gate.check(np.array_equal(sig, sig_one),
                   'the self-energy branch sums are the one-rank sums, bitwise')
        num = NUM
        window = np.arange(nocc - 2, nocc + 2)
        qp_whole = solve_qp_energy_space_time(mf, mol, nocc, window,
                                              numerics=num, factors=factors)
        qp = solve_qp_energy_space_time(mf, mol, nocc, window, numerics=num,
                                        factors=part)
    gate.check(np.array_equal(qp, qp_whole),
               'the window on slices is the replicated window, bitwise')
    every = gate.everyone(qp)
    gate.check(all(np.array_equal(q, every[0]) for q in every),
               "every rank's window is rank 0's")
    share = (X.nbytes + D.nbytes) / size
    gate.check(led['tiles'] < 1.5 * share,
               'the tiles a rank holds are its share of the factors',
               f"{led['tiles']} B against a share of {share:.0f} B")
    peaks = gate.everyone(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    gate.info(f'peak resident memory per rank (ru_maxrss): {peaks}')
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
