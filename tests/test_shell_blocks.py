"""The derivative-integral blocks must actually fit the memory they claim to.

`two_electron_skeleton` sweeps (grad mu nu|lam sig) in shell ranges chosen so
each range's block fits a byte budget. The cost per AO of the leading index is
3 nao^3 * 8 -- 3.6e7 bytes at 114 basis functions, 1.4e8 at 180 -- and
`mol.ao_loc_nr()` is int32, so the product overflows past 60 and 15 AOs
respectively. A wrapped negative compares as fitting, every candidate block
then passes, and the loop runs to the end of the molecule: the sweep asks for
the whole tensor instead of a chunk of it.

Nothing catches that downstream. The numbers stay right -- the blocks are a
partition either way -- so it costs memory while looking like a successful run,
and it gets worse as the molecule grows, which is exactly where the blocking
was supposed to help.

IT ONLY BITES ON NUMPY 2. Under NEP 50 a Python int is weak and int32 * int
stays int32; numpy 1 used value-based casting and promoted to int64, so the
same source is correct there and overflows here. That is why this went
unnoticed on a numpy 1.23 workstation and surfaced on a numpy 2 cluster. These
tests assert the partition is right rather than that the warning is absent, so
they specify the behaviour on both and can only FAIL where the bug is real.
"""
import warnings

import numpy as np
import pytest
from pyscf import gto

from src.Base.constants import DERIV_BLOCK_BYTES
from src.gradients.grad_engine import _shell_blocks

#: Big enough that per_ao * a whole-molecule block overflows int32; benzene at
#: cc-pVDZ is 114 basis functions and overflows past 60 of them.
BENZENE = ('C 0 0 1.39; C 1.20 0 0.695; C 1.20 0 -0.695; C 0 0 -1.39; '
           'C -1.20 0 -0.695; C -1.20 0 0.695; H 0 0 2.47; H 2.14 0 1.235; '
           'H 2.14 0 -1.235; H 0 0 -2.47; H -2.14 0 -1.235; H -2.14 0 1.235')


def blocks_and_widths(mol, budget):
    ao_loc = mol.ao_loc_nr()
    blocks = _shell_blocks(mol, budget)
    return blocks, [int(ao_loc[b] - ao_loc[a]) for a, b in blocks]


@pytest.mark.parametrize('basis', ('cc-pvdz', 'cc-pvtz'))
def test_no_block_is_chosen_by_an_overflowed_comparison(basis):
    mol = gto.M(atom=BENZENE, basis=basis, verbose=0)
    assert mol.ao_loc_nr().dtype == np.int32, (
        'this test exists because ao_loc is int32; if that changed, so did the '
        'failure mode')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        blocks, widths = blocks_and_widths(mol, DERIV_BLOCK_BYTES)
    overflow = [c for c in caught if 'overflow' in str(c.message)]
    assert not overflow, f'{basis}: {overflow[0].message}'
    assert blocks[0][0] == 0 and blocks[-1][1] == mol.nbas
    assert sum(widths) == mol.nao, 'the blocks must partition the AOs'


@pytest.mark.parametrize('basis', ('cc-pvdz', 'cc-pvtz'))
def test_a_block_fits_the_budget_unless_one_shell_already_exceeds_it(basis):
    """One shell is the smallest a block can be, so a single f shell at 264
    basis functions is over budget by construction. Every block WIDER than one
    shell has no such excuse."""
    mol = gto.M(atom=BENZENE, basis=basis, verbose=0)
    per_ao = 3 * mol.nao ** 3 * 8
    blocks, widths = blocks_and_widths(mol, DERIV_BLOCK_BYTES)
    for (sh0, sh1), width in zip(blocks, widths):
        if sh1 - sh0 == 1:
            continue
        assert width * per_ao <= DERIV_BLOCK_BYTES, (
            f'{basis}: shells {sh0}:{sh1} span {width} AOs = '
            f'{width * per_ao / 1024 ** 2:.0f} MB, over the '
            f'{DERIV_BLOCK_BYTES / 1024 ** 2:.0f} MB budget')


def test_a_tiny_budget_still_partitions_the_molecule():
    """The degenerate case: every block is one shell and nothing is dropped."""
    mol = gto.M(atom=BENZENE, basis='cc-pvdz', verbose=0)
    blocks, widths = blocks_and_widths(mol, 1)
    assert len(blocks) == mol.nbas
    assert sum(widths) == mol.nao
