"""The quasiparticle-Hamiltonian pairs' reverse pass holds one subspace's
solves at a time (`QPSetPass._reverse_group`).

A pair joins two states of one subspace. Each paired state keeps its slice
and its screening solves at every held frequency until the last state of its
subspace has passed; that subspace's contractions are then taken and its
solves let go. The peak is the widest subspace's solves, not every paired
state's: on CBP (8 ranks, QP-H on) holding all of them raised the reverse
pass's peak by 8.5 GB a rank.

Formaldehyde/cc-pVDZ (C 0.1 Bohr out of plane), PBE0, SOP on orbitals 4-10
with the quasiparticle Hamiltonian on and a 0.1 Ha subspace window: the
subspaces (4, 5, 6) and (9, 10), as in tests/test_qp_hamiltonian.py.

Run: python -m pytest tests/test_qp_pairs_by_subspace.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))
import src.gradients.gwbse.adaptive_selection as adaptive_selection_module
import src.gradients.gwbse.qp_set as qp_set_module
import src.gradients.gwbse.qp_space_time as qp_space_time
from tests.test_qp_hamiltonian import (PBE0_SUBSPACE_WINDOW, PBE0_SUBSPACES,
                                       PBE0_WINDOW, chain, pbe0)


def test_one_subspace_is_held_at_a_time(monkeypatch):
    for module in (qp_set_module, adaptive_selection_module):
        monkeypatch.setattr(module, 'QP_HAMILTONIAN_WINDOW',
                            PBE0_SUBSPACE_WINDOW)
    held, contract = [], qp_space_time.pairs_chi0_backward

    def recorded(WtB, pairs):
        held.append(frozenset(WtB))
        return contract(WtB, pairs)
    monkeypatch.setattr(qp_space_time, 'pairs_chi0_backward', recorded)
    ch = chain(pbe0, qp_hamiltonian=True, qp_window=PBE0_WINDOW)
    grad = ch.excitation_gradient()[0]
    assert ch.resolved.qp_subspaces[1] == PBE0_SUBSPACES
    assert np.isfinite(grad).all()
    widths = {len(h) for h in held}
    assert held and widths <= {len(s) for s in PBE0_SUBSPACES}, held


def test_pairs_fall_into_their_subspaces():
    pairs = [(4, 4, 0), (9, 10, 0), (4, 6, 0), (5, 6, 0), (10, 10, 0)]
    assert qp_space_time.pair_subspaces(pairs) == [frozenset({4, 5, 6}),
                                                   frozenset({9, 10})]
