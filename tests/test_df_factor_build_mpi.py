"""The tiled force's DF factors against the folded ADC(2) operator's, real ranks.

    python tests/test_df_factor_build_mpi.py                 # one rank
    OMP_NUM_THREADS=1 srun --mpi=pmi2 -n 2 python tests/test_df_factor_build_mpi.py

Script, not a pytest module: it initializes MPI. `DFFactors.build` (the
provider of the tiled MP2 force) and `ee_ri_fold.bare_factors` (the fold's
factors, which the tiled ADC(2) force reads) must hand the force the same
bits, so the B_ia, B_ij and B_ab chunks of the two builds are compared as
bytes, on every rank, with and without a frozen core, at the default chunk
width and at a small one that gives every rank several chunks. Also: the
frozen core's factors of both providers, the refusal of auxbasis None, and a
control that the comparison fails on a one-ulp difference. Every rank's
verdict is gathered.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np                                           # noqa: E402
from pyscf import gto, scf                                   # noqa: E402

from src.Base.utils.parallel.mpi_grid import (distributed, grid_comm,  # noqa: E402
                                              lockstep_mean_field)
from src.SingleReference.ADC.eeADC import ee_ri_fold         # noqa: E402
from src.SingleReference.ADC.eeADC.ee_relaxed_density import core_factors  # noqa: E402
from src.gradients import df_factors                         # noqa: E402
from tests.adc_common import ground_fold                     # noqa: E402
from tests.test_distributed_fit_mpi import Gate, bitwise     # noqa: E402

AUX = 'cc-pvdz-ri'
ATOM = 'O 0 0 0.12; H 0 0.76 -0.48; H 0.05 -0.74 -0.5'


def parts(f):
    """Every array of a factor set that the force reads, in a fixed order."""
    return ([f.LT, f.BooP, f.eo, f.ev]
            + [f.bvv_chunks[c] for c in sorted(f.bvv_chunks)])


def same_build(a, b):
    """The two factor sets hold the same chunks with the same bytes."""
    return (sorted(a.bvv_chunks) == sorted(b.bvv_chunks)
            and bitwise(parts(a), parts(b)))


def main(comm):
    gate = Gate(comm)
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    mf = lockstep_mean_field(scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run(),
                             comm)
    with distributed(comm):
        for frozen in (0, 1):
            for width in (None, 16):
                gate.section(f'frozen {frozen}, chunk width {width}')
                fold = ee_ri_fold.bare_factors(mol, mf, frozen=frozen, auxbasis=AUX,
                                               comm=comm, aux_chunk=width)
                prov = df_factors.DFFactors.build(mol, mf, frozen, AUX, comm=comm,
                                                  aux_chunk=width)
                gate.check(same_build(fold, prov.f),
                           'B_ia, B_ij and this rank\'s B_ab chunks are the fold\'s, bitwise',
                           f'{len(fold.bvv_chunks)} chunk(s) here')
                chunks, owners = ee_ri_fold.aux_chunks(fold.naux, fold.nv, gate.size, width)
                gate.check(prov.chunks == chunks and prov.owners == owners
                           and sorted(prov.f.bvv_chunks) == prov.my_chunks,
                           'the provider\'s chunks and owners are aux_chunks\'')
                if frozen:
                    gate.check(bitwise([prov.core_factors(mol, mf, frozen, AUX)],
                                       [core_factors(mol, mf, frozen, AUX)]),
                               'the frozen core\'s factors are the fold\'s, bitwise')
                    gnd = ground_fold(mol, mf, frozen, AUX, comm=comm,
                                      aux_chunk=width)
                    gate.check(same_build(gnd.f, prov.f),
                               'the fold provider\'s factors are the default provider\'s')
        gate.section('control and refusal')
        control = [a.copy() for a in parts(fold)]
        control[0].flat[0] = np.nextafter(control[0].flat[0], np.inf)
        gate.check(not bitwise(parts(fold), control),
                   'the comparison sees a one-ulp difference')
        try:
            df_factors.DFFactors.build(mol, mf, 0, None, comm=comm)
            refused = False
        except ValueError:
            refused = True
        gate.check(refused, 'auxbasis None is refused')
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
