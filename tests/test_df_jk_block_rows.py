"""A rank's density-fitted J/K build holds its exchange buffers under a fixed
budget, set by nao alone (`distributed_df.exchange_block_rows`).

WHAT BREAKS IF THIS FAILS. pyscf's K loop holds (2, rows, nao, nao) with rows
= min(blockdim, free memory), blockdim 240: 4.6 GB on every rank at nao 1090
(CBP), whatever the rank count, and a block that follows the free memory sums
K in an order, so in bits, that changes from run to run. The rows here come
from DF_JK_BLOCK_GB and nao alone, for the general K and for the exchange read
on occupied columns.

Water, cc-pVDZ, LRC-wPBEh (bare and long-range rows), over 2 simulated ranks;
the budget lowered to give 4-row blocks, so the blocking shows on water.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import lib

from src.Base.factorization import distributed_df
from src.Base.factorization.distributed_df import distributed_fock
from src.Base.numerics.constants import BYTES_PER_GB, DF_JK_BLOCK_GB
from tests.pins import assert_within_class
from tests.test_rotation_gradient_columns import OMEGA, on_ranks, symmetric_partial


def test_rows_keep_the_buffer_under_the_budget():
    for nao in (300, 1090, 1780):
        rows = distributed_df.exchange_block_rows(nao)
        assert 4 <= rows <= 240
        assert 2 * rows * nao * nao * 8 <= DF_JK_BLOCK_GB * BYTES_PER_GB
    assert distributed_df.exchange_block_rows(4024) == 4


def test_both_exchange_loops_take_the_budget_rows(monkeypatch):
    """Every block the ranks' J/K loops take is the budget's, and K is the
    one of pyscf's 240-row blocks to the reduction's last bits."""
    def jk(comm, mf):
        # AO matrices, not MO ones: the two runs' SCFs need not end in the
        # same orbital signs
        nao = mf.mol.nao
        x = symmetric_partial(nao)
        Co = np.random.default_rng(5).standard_normal((nao, mf.mol.nelectron // 2))
        with distributed_fock(mf, build=False):
            out = [mf.get_k(mf.mol, x, hermi=1, omega=omega) for omega in (None, OMEGA)]
            out.append(mf.get_k(mf.mol, lib.tag_array(x, read_columns=Co), hermi=1) @ Co)
        return out

    whole = on_ranks(jk, 2)
    sizes, loop = [], distributed_df.DistributedDF.loop

    def recorded(self, blksize=None):
        sizes.append(blksize)
        return loop(self, blksize)
    monkeypatch.setattr(distributed_df.DistributedDF, 'loop', recorded)
    monkeypatch.setattr(distributed_df, 'DF_JK_BLOCK_GB', 4 * 16 * 24 ** 2 / BYTES_PER_GB,
                        raising=False)
    small = on_ranks(jk, 2)
    assert sizes and set(sizes) == {4}, sorted(set(sizes))
    for got, ref in zip(small, whole):
        for a, b in zip(got, ref):
            assert_within_class(a, b, 'root', label='K in 4-row blocks')
    for got in small:
        for a, b in zip(got, small[0]):
            assert np.array_equal(a, b)
