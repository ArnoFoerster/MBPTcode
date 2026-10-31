"""A large quasiparticle-Hamiltonian group is one calculation on every rank.

WHAT BREAKS IF THIS FAILS. A group of more than QP_SUBSPACE_DENSE_MAX orbitals
(`sum_over_poles.PoleSlice`) sums each rank's share of the poles in H's
eigenbasis. If every rank takes its own eigh, two ranks whose eigenvectors
differ (a sign, a rotation of a degenerate pair: LAPACK on two nodes need not
agree) add partials of two different bases, and H, its pole strengths and its
adjoint are wrong on every rank. A GMRES that each rank iterated on its own
arithmetic could stop at different counts and leave the others in a
reduction; one that runs out of iterations is a numerical shortfall, flagged
with its last iterate kept, not an error.

Synthetic group, no SCF: 26 orbitals, random positive residues on 60 poles
well away from the energies. On rank 1 of two simulated ranks, eigh returns
its first eigenvector with the opposite sign, a valid eigendecomposition.
"""
import numpy as np
import pytest

from src.Base.numerical_flags import collecting
from src.Base.numerics.constants import QP_SUBSPACE_DENSE_MAX
from src.Base.utils.parallel.mpi_grid import current_comm
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.gwbse.sum_over_poles_adjoint import \
    subspace_hamiltonian_backward
from src.SingleReference.GW import sum_over_poles as sop

N = QP_SUBSPACE_DENSE_MAX + 2


def group():
    """(e0, amp, d, seed, h_bar) of a synthetic group of N orbitals."""
    rng = np.random.default_rng(7)
    e0 = np.diag(np.linspace(-0.6, -0.4, N))
    off = 1e-3 * rng.standard_normal((N, N))
    e0 = e0 + 0.5 * (off + off.T)
    b = 0.02 * rng.standard_normal((60, N, 3))
    amp = b @ b.transpose(0, 2, 1)
    d = np.concatenate([rng.uniform(-3.0, -1.5, 30), rng.uniform(1.0, 3.0, 30)])
    h_bar = rng.standard_normal((N, N))
    return e0, amp, d, np.diag(np.diag(e0)), 0.5 * (h_bar + h_bar.T)


def results():
    """(H, Z, e0_bar) of the group in the enclosing distribution."""
    e0, amp, d, seed, h_bar = group()
    h, _, ok = sop.subspace_hamiltonian(e0, amp, d, seed)
    assert ok
    z = sop.subspace_pole_strengths(h, amp, d)
    return h, z, subspace_hamiltonian_backward(h, amp, d, h_bar)[0]


def test_ranks_whose_eigh_differs_give_the_serial_group(monkeypatch):
    serial = results()
    eigh = np.linalg.eigh

    def rank_one_flips(a):
        e, u = eigh(a)
        comm = current_comm()
        if comm is not None and comm.Get_rank() == 1:
            u = u.copy()
            u[:, 0] *= -1.0
        return e, u
    monkeypatch.setattr(np.linalg, 'eigh', rank_one_flips)
    out = run_simulated(lambda comm: results(), 2)
    for got in out:
        for a, b in zip(got, out[0]):
            assert np.array_equal(a, b)
        for a, b in zip(got, serial):
            assert np.abs(a - b).max() < 1e-10, np.abs(a - b).max()


@pytest.mark.parametrize('ranks', [1, 2])
def test_a_capped_gmres_is_flagged_and_keeps_its_iterate(monkeypatch, ranks):
    monkeypatch.setattr(sop, 'QP_SUBSPACE_GMRES_RESTART', 2, raising=False)
    monkeypatch.setattr(sop, 'QP_SUBSPACE_GMRES_MAXITER', 1, raising=False)
    e0, amp, d, seed, _ = group()

    def one(comm):
        flags = []
        with collecting(flags):
            h, _, _ = sop.subspace_hamiltonian(e0, amp.copy(), d, seed,
                                               max_iter=2)
        return h, flags
    # warnings are caught process-wide, so the watch spans every rank's
    # thread; the flags are each rank's own
    with pytest.warns(sop.SubspaceGMRESShort):
        out = run_simulated(one, ranks)
    for h, flags in out:
        assert np.all(np.isfinite(h))
        assert np.array_equal(h, out[0][0])
        assert {f['flag'] for f in flags} == {'qp_subspace_gmres_not_converged'}
