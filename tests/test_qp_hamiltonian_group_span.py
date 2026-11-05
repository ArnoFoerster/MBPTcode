"""A quasiparticle-Hamiltonian group is a near-degenerate set, not a band.

WHAT BREAKS IF THIS FAILS. Neighbours chained one by one under
QP_HAMILTONIAN_WINDOW join a whole band of virtuals into one group (CBP's
139 orbitals spanning 7.6 eV); the group's dense solve then needs tens of GB
on every rank. A group holds only orbitals all within the window of each
other.

No SCF: synthetic spectra.
"""
import numpy as np

from src.Base.numerics.constants import QP_HAMILTONIAN_WINDOW
from src.SingleReference.GW.qp_selection import degenerate_blocks, window_blocks

W = QP_HAMILTONIAN_WINDOW


def test_a_band_of_chained_neighbours_is_not_one_group():
    eps = 0.6 * W * np.arange(40)            # every gap inside the window
    assert len(degenerate_blocks(eps, range(40), W)) == 1      # chained
    for block in window_blocks(eps, range(40), W):
        e = eps[list(block)]
        assert e.max() - e.min() < W
    assert sorted(p for b in window_blocks(eps, range(40), W) for p in b) \
        == list(range(40))


def test_a_near_degenerate_pair_stays_one_group_and_ties_stay_together():
    eps = np.array([-1.0, -0.5, -0.5, -0.5 + 0.2 * W, 0.0, 0.5])
    assert window_blocks(eps, range(6), W) == [(0,), (1, 2, 3), (4,), (5,)]
    # equal energies are never split, whatever the member order
    assert window_blocks(eps, [3, 2, 1, 5, 4, 0], W) == window_blocks(eps, range(6), W)


def test_a_gap_exactly_the_window_starts_a_new_group():
    eps = np.array([0.0, W, 2 * W])
    assert window_blocks(eps, range(3), W) == [(0,), (1,), (2,)]


def test_a_pair_at_the_window_edge_stays_one_group():
    # the window from the lowest member ends between the pair 2 mHa apart:
    # the block is cut at its wider gap, below the pair
    eps = np.array([0.0, 0.029, 0.031])
    assert window_blocks(eps, range(3), 0.03) == [(0,), (1, 2)]
    eps = np.array([0.0, 0.02, 0.028, 0.03, 0.05])
    for block in window_blocks(eps, range(5), 0.03):
        e = eps[list(block)]
        assert e.max() - e.min() < 0.03
    assert window_blocks(eps, range(5), 0.03) == [(0,), (1, 2, 3), (4,)]
