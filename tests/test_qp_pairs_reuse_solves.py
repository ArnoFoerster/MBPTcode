"""The quasiparticle-Hamiltonian pairs' reverse pass takes each paired state's
screening solves from the diagonal pass (`QPSetPass._reverse_group`).

Every held frequency's [1 - chi0(i.nu)]^-1 is applied once per state of the
reverse pass's active set, whether or not the state is in a pair: a pair's
off-diagonal adjoint reads the solves its two states already made, so a group
of g states costs g solves per frequency, not g + g(g + 1). A frequency's
chi0 adjoint of all the pairs is one product per state
(`pairs_chi0_backward`), the sum of the pairs' own one by one to round-off.

Formaldehyde/cc-pVDZ (C 0.1 Bohr out of plane), Hartree-Fock, SOP on orbitals
4-17 with the quasiparticle Hamiltonian on: the subspace (12, 13), as in
tests/test_qp_hamiltonian.py.

Run: python -m pytest tests/test_qp_pairs_reuse_solves.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))
import src.gradients.gwbse.qp_space_time as qp_space_time
from src.gradients.gwbse.contour_deformation_adjoint import (
    pair_term_backward, pairs_chi0_backward)
from tests.test_qp_hamiltonian import PAIR, chain


def test_each_state_solves_once_per_frequency(monkeypatch):
    solves, active = [], []
    factorize, group = qp_space_time.screening_solver, \
        qp_space_time.QPSetPass._reverse_group

    def counted_solver(eps_mat):
        solve, n = factorize(eps_mat), len(solves)
        solves.append(0)

        def counted(b):
            solves[n] += 1
            return solve(b)
        return counted

    def recorded(self, *a, **kw):
        active.append(len(self.active))
        return group(self, *a, **kw)
    monkeypatch.setattr(qp_space_time, 'screening_solver', counted_solver)
    monkeypatch.setattr(qp_space_time.QPSetPass, '_reverse_group', recorded)
    ch = chain(qp_hamiltonian=True)
    grad = ch.excitation_gradient()[0]
    assert ch.resolved.qp_subspaces[1] == (PAIR,)
    assert np.isfinite(grad).all()
    assert solves and set(active) == {active[0]}, (solves, active)
    assert set(solves) == {active[0]}, (set(solves), active[0])


def test_the_pairs_chi0_adjoint_is_the_sum_of_the_pairs():
    rng = np.random.default_rng(7)
    naux, norb, states = 30, 9, (2, 5, 6)
    B = {s: rng.standard_normal((naux, norb)) for s in states}
    W = {s: rng.standard_normal((naux, norb)) for s in states}
    pairs = [(i, j, rng.standard_normal(norb))
             for a, i in enumerate(states) for j in states[a:]]
    one_by_one = sum(pair_term_backward(B[i], B[j], W[i], W[j], g)[2]
                     for i, j, g in pairs)
    at_once = pairs_chi0_backward(W, pairs)
    assert np.allclose(at_once, one_by_one, rtol=1e-13, atol=1e-13)
    assert np.array_equal(at_once, at_once.T)
