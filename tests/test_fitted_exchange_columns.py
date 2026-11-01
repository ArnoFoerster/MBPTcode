"""The fitted exchange skeleton solves its metric on blocks of occupied columns,
moving L and At once each instead of streaming every tile to every rank.

WHAT BREAKS IF THIS FAILS. `FittedFockSkeleton.prepare` broadcast each
auxiliary tile of L (naux x nocc nao) to every rank for the forward solve,
again for the backward solve, and Kt once more for <g Kt^P, Kt^Q>: every rank
received the whole tensor about three times a channel (CzBN1, 2 nodes: 85 GB
and some 87 s a rank). The columns of L are independent, so one exchange moves
it to fixed blocks of occupied orbitals, each block is solved where it lands,
and one exchange moves At back.

Water, cc-pVDZ / cc-pVDZ-JKFIT in 16-function auxiliary tiles, a random
symmetric partial and a density of 5 random occupied columns, Coulomb and
exchange halves, over 1, 2 and 3 simulated ranks:
  * every rank holds the one-rank skeleton bit for bit;
  * no tile is broadcast, two exchanges run, and a rank receives less than
    the whole of L through them.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto
from pyscf.df import addons

from src.Base.utils.parallel.mpi_grid import distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients import skeleton_tiles

WATER = 'O 0 0.05 0.1173; H 0.02 0.7572 -0.4692; H 0 -0.7672 -0.4592'
NOCC = 5


def inputs():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    auxmol = addons.make_auxmol(mol, 'cc-pvdz-jkfit')
    rng = np.random.default_rng(3)
    g = rng.standard_normal((mol.nao, mol.nao))
    Cd = rng.standard_normal((mol.nao, NOCC)) * 0.3
    return mol, auxmol, g + g.T, Cd


def skeleton(comm=None):
    mol, auxmol, g, Cd = inputs()
    return skeleton_tiles.fitted_fock_skeleton(mol, auxmol, g, Cd @ Cd.T, occ=Cd,
                                               exchange=0.5, tile=16)


@pytest.mark.parametrize('size', [2, 3])
def test_columns_move_once_and_keep_the_bits(size, monkeypatch):
    with distributed(None):
        serial = skeleton()
    mol, auxmol, _, _ = inputs()
    whole = auxmol.nao_nr() * NOCC * mol.nao * 8
    seen, lock = {'bcast': 0, 'exchange': 0, 'received': {}}, threading.Lock()
    bcast, exchange = skeleton_tiles.broadcast_rows, skeleton_tiles.exchange_blocks

    def counted_bcast(*a, **kw):
        with lock:
            seen['bcast'] += 1
        return bcast(*a, **kw)

    def counted_exchange(send, shapes, comm):
        got = exchange(send, shapes, comm)
        with lock:
            seen['exchange'] += 1
            name = threading.current_thread().name
            seen['received'][name] = seen['received'].get(name, 0) + sum(
                x.nbytes for r, x in enumerate(got) if r != comm.Get_rank())
        return got
    monkeypatch.setattr(skeleton_tiles, 'broadcast_rows', counted_bcast)
    monkeypatch.setattr(skeleton_tiles, 'exchange_blocks', counted_exchange)
    out = run_simulated(lambda comm: skeleton(comm), size)
    for grad in out:
        np.testing.assert_array_equal(grad, serial)
    assert seen['bcast'] == 0, seen['bcast']
    assert seen['exchange'] == 2 * size, seen['exchange']
    assert max(seen['received'].values()) < whole, (seen['received'], whole)
