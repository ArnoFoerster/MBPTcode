"""The serial ISDF-K build takes the same arithmetic whatever memory is free.

WHAT BREAKS IF THIS FAILS. `ISDFJK` with z_mode 'auto' chooses between
holding Z ('dense') and rebuilding its rows from L = V^1/2 M ('factored'), and
with no `layout.block` it sizes the row block of its J/K builds. Both are
different arithmetic, so the same input gives other last bits when either is
taken from the memory the process happens to have free: two runs of one
input, or one run before and after a large array is freed, stop agreeing and a
pin recorded on one machine moves on another. Both are read from max_memory
(the budget), never from the memory free at the moment.

Water/cc-pVDZ Hartree-Fock on the G1 grid, the core guess: J and K of two
fresh handles, one built while the process reports almost nothing free and one
while it reports far more than max_memory, are the same bits.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.factorization.isdf_jk import isdf_jk
from src.Base.utils.parallel import memory
from tests.isdf_test_grid import g1
from tests.molecules import WATER



def jk_with_free(monkeypatch, free_mb):
    """(J, K, z mode, block) of a fresh ISDF-K handle while the process
    reports `free_mb` MB of max_memory left."""
    monkeypatch.setattr(memory, 'left_mb', lambda max_memory: free_mb)
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = isdf_jk(scf.RHF(mol), auxbasis='cc-pvdz-ri', counts=g1(scf.RHF(mol)))
    dm = mf.get_init_guess(key='1e')
    vj, vk = mf.get_jk(mol, dm)
    with_df = mf.with_df
    return vj, vk, with_df._resolve_z_mode(), with_df._block()


def test_free_memory_does_not_choose_the_arithmetic(monkeypatch):
    scarce = jk_with_free(monkeypatch, 1.0)
    ample = jk_with_free(monkeypatch, 1e7)
    assert scarce[2:] == ample[2:], (scarce[2:], ample[2:])
    assert np.array_equal(scarce[0], ample[0])
    assert np.array_equal(scarce[1], ample[1])
