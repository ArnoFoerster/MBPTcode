"""ONE orbital-energy separation decides degeneracy, and the correction
spread is a bar.

The window's degenerate blocks and the explicit set's shared-correction
blocks were decided by two constants of one value (QP_WINDOW_DEGENERACY_TOL,
QP_BLOCK_DEGENERACY_TOL) and two settings (`degeneracy_tol`, `block_tol`) that
could be given apart. They are one separation, `degeneracy_tol`
(QP_DEGENERACY_TOL). The spread of the members' own corrections within which
a block is a true degeneracy is a different quantity -- averaging accidental
near-degeneracies moved PXZ-TRZ's SOP T1 by -22 meV -- and stays its own
constant, QP_BLOCK_CORRECTION_TOL, a bar no run sets.
"""
from dataclasses import fields

import numpy as np

import src.Base.constants as constants
from src.Base.numerics import Numerics
from src.Base.numerics import QPNumerics
from src.gradients.conventions import FrozenConventions
from src.gradients.excited_state import ExcitedStateChain

#: Ha: three orbital energies, 1 and 2 two-thirds of the separation apart
EPS = np.array([-0.5, 0.1, 0.1, 0.3])
#: Ha: their own corrections, 1 and 2 agreeing far inside the bar
OWN = np.array([-0.02, 0.012, 0.012, 0.03])


def test_one_separation_one_bar():
    assert constants.QP_BLOCK_CORRECTION_TOL == constants.QP_DEGENERACY_TOL
    for gone in ('QP_WINDOW_DEGENERACY_TOL', 'QP_BLOCK_DEGENERACY_TOL'):
        assert not hasattr(constants, gone), gone
    names = {f.name for f in fields(QPNumerics)}
    assert 'degeneracy_tol' in names
    assert not names & {'block_tol', 'block_correction_tol'}


def test_the_set_blocks_read_the_one_separation():
    """A chain's degeneracy_tol is the energy gate of its blocks: two
    orbitals 3e-4 Ha apart are one block at 5e-4 and two at 1e-4."""
    eps = EPS.copy()
    eps[2] += 3e-4
    for tol, blocks in ((5e-4, ((1, 2),)), (1e-4, ())):
        ch = ExcitedStateChain.__new__(ExcitedStateChain)
        ch.settings = Numerics()
        ch.conventions = FrozenConventions(qp_set=np.arange(len(eps)))
        ch.qp_hamiltonian = False
        ch.degeneracy_tol = tol
        ch._block_shared(eps, eps + OWN)
        assert ch.resolved.qp_blocks[1] == blocks, tol
        rec = ch.conventions.qp_block_record
        assert rec['energy_tol_Ha'] == tol
        assert rec['correction_tol_Ha'] == constants.QP_BLOCK_CORRECTION_TOL
