"""The quasiparticle Hamiltonian's one-body term of the Casida action
(`with_one_body`) is the same bits on every rank.

The term is formed on every rank from blocks and products whose last bits
differ between ranks (threaded BLAS); a Davidson that read them would take
different decisions on different ranks (a trial batch of 9 vectors on one
rank, 8 on the others) and the ranks would fall apart. Two simulated ranks
whose Hamiltonian blocks differ by one ulp stand for that.
"""
import numpy as np

from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.LinearResponse.davidson import with_one_body

NO, NV, K = 5, 7, 3


def _blocks(rank):
    rng = np.random.default_rng(7)
    h_oo = rng.standard_normal((2, 2))
    h_vv = rng.standard_normal((3, 3))
    h_oo, h_vv = h_oo + h_oo.T, h_vv + h_vv.T
    np.fill_diagonal(h_oo, 0.0)
    np.fill_diagonal(h_vv, 0.0)
    if rank:       # this rank's products end in other bits
        h_vv = np.nextafter(h_vv, np.inf)
        h_oo = np.nextafter(h_oo, -np.inf)
    return (np.array([1, 3]), h_oo), (np.array([0, 2, 5]), h_vv)


def _action(z):
    z3 = np.asarray(z, float).reshape(-1, NO, NV)
    return 0.5 * z3, np.zeros_like(z3)


def _one_rank(comm):
    z = np.random.default_rng(11).standard_normal((K, NO, NV))
    coupled = with_one_body(_action, _blocks(comm.Get_rank()))
    return coupled(z)[0]


def test_the_one_body_action_is_rank_0s_on_every_rank():
    out = run_simulated(_one_rank, 2)
    assert np.array_equal(out[0], out[1])
    # and it is the term itself: A z = z/2 + the one-body blocks
    z = np.random.default_rng(11).standard_normal((K, NO, NV))
    (o, h_oo), (v, h_vv) = _blocks(0)
    expect = 0.5 * z
    expect[:, :, v] += z[:, :, v] @ h_vv
    expect[:, o, :] -= np.einsum('ij,kja->kia', h_oo, z[:, o, :])
    assert np.abs(out[0] - expect).max() < 1e-14
