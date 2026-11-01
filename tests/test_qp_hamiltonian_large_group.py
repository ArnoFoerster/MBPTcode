"""A large quasiparticle-Hamiltonian group is solved without its Jacobian.

WHAT BREAKS IF THIS FAILS. The dense Newton solve builds the n^2 x n^2
Jacobian of Sigma^c[H] (a group of 139 orbitals: 3.7e8 elements, 22 GB a rank
with its temporaries); a group above QP_SUBSPACE_DENSE_MAX orbitals is solved
by GMRES on the operator, which must reach the same H, the same pole strengths
and the same adjoint, on one rank or several, within a memory that does not
grow as n^4.

No SCF: a synthetic group, amplitudes small against the pole gaps so Newton
converges from the bare e0.
"""
import tracemalloc

import numpy as np

from src.SingleReference.GW import sum_over_poles as sop
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.gwbse.sum_over_poles_adjoint import subspace_hamiltonian_backward


def group(n, n_x, scale=3e-4, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_x, n, n))
    amp = scale * (a + a.transpose(0, 2, 1))
    e0 = np.diag(np.linspace(0.0, 0.01, n)) + 1e-3 * np.triu(rng.standard_normal((n, n)), 1)
    e0 = e0 + np.triu(e0, 1).T
    d = rng.uniform(0.3, 2.0, n_x) * rng.choice([-1.0, 1.0], n_x)
    return e0, amp, d


def test_matrix_free_solve_matches_the_dense_one(monkeypatch):
    e0, amp, d = group(8, 40)
    h_ref, _, ok = sop.subspace_hamiltonian(e0, amp, d, e0)
    z_ref = sop.subspace_pole_strengths(h_ref, amp, d)
    back_ref = subspace_hamiltonian_backward(h_ref, amp, d, np.diag(np.arange(8.0)))
    monkeypatch.setattr(sop, 'QP_SUBSPACE_DENSE_MAX', 3)
    h, _, ok_mf = sop.subspace_hamiltonian(e0, amp, d, e0)
    assert ok and ok_mf
    assert np.abs(h - h_ref).max() < 1e-12
    assert np.abs(sop.subspace_pole_strengths(h, amp, d) - z_ref).max() < 1e-12
    back = subspace_hamiltonian_backward(h, amp, d, np.diag(np.arange(8.0)))
    for got, want in zip(back, back_ref):
        assert np.abs(got - want).max() < 1e-10


def test_the_poles_split_over_ranks_give_the_same_h(monkeypatch):
    e0, amp, d = group(8, 41)
    monkeypatch.setattr(sop, 'QP_SUBSPACE_DENSE_MAX', 3)
    one, _, ok = sop.subspace_hamiltonian(e0, amp, d, e0)
    two = run_simulated(lambda comm: sop.subspace_hamiltonian(e0, amp, d, e0), 3)
    assert ok and all(t[2] for t in two)
    for h, _, _ in two:
        assert np.abs(h - one).max() < 1e-12
        assert np.array_equal(h, two[0][0])


def test_a_large_group_needs_no_n4_memory():
    n = 70                      # dense: a 4900 x 4900 Jacobian, 190 MB
    e0, amp, d = group(n, 30)
    tracemalloc.start()
    h, _, ok = sop.subspace_hamiltonian(e0, amp, d, e0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert ok
    assert peak < 40e6, peak
