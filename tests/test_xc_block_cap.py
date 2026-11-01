"""A rank's block of the distributed XC quadrature plans for at most
XC_BLOCK_MAX_MB, whatever max_memory the mean field carries.

WHAT BREAKS IF THIS FAILS. pyscf's block loop sizes its AO values by the
max_memory it is handed: on a large node rank 0's free share, several GB of AO
values on every rank at every rank count (3.8 GB at nao 990), and a blocking,
so an order of the sums, that follows the free memory. Capped, the block is
near 1 GB and the same in every run.

Water, cc-pVDZ, LRC-wPBEh, density-fitted, over 2 simulated ranks, with
max_memory 100 GB: every quadrature plans with the cap, and the potential is
the uncapped one's to the reduction's last bits.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.factorization import distributed_df
from src.Base.factorization.distributed_df import distributed_fock
from src.Base.numerics.constants import XC_BLOCK_MAX_MB
from tests.pins import assert_within_class
from tests.test_rotation_gradient_columns import on_ranks, symmetric_partial


def test_the_quadrature_plans_with_the_cap(monkeypatch):
    planned, partial = [], distributed_df.DistributedNumInt.partial_xc

    def recorded(self, method, mol, xc_code, dms, relativity=0, hermi=1,
                 max_memory=2000, verbose=None):
        planned.append(max_memory)
        return partial(self, method, mol, xc_code, dms, relativity, hermi,
                       max_memory, verbose)
    monkeypatch.setattr(distributed_df.DistributedNumInt, 'partial_xc', recorded)

    def veff(comm, mf):
        mf.max_memory = 100000
        dm = symmetric_partial(mf.mol.nao) * 1e-2 + mf.make_rdm1()
        with distributed_fock(mf, build=False):
            return np.asarray(mf.get_veff(mf.mol, dm))

    capped = on_ranks(veff, 2)
    assert planned and max(planned) <= XC_BLOCK_MAX_MB, sorted(set(planned))
    monkeypatch.setattr(distributed_df, 'XC_BLOCK_MAX_MB', 1e9, raising=False)
    planned.clear()
    whole = on_ranks(veff, 2)
    assert max(planned) > XC_BLOCK_MAX_MB
    for a, b in zip(capped, whole):
        assert_within_class(a, b, 'root', label='V with the capped blocks')
    assert np.array_equal(capped[0], capped[1])
