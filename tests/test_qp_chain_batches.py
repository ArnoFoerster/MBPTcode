"""The QP-set reverse pass chains several states' B_p_bar terms in one sum
(`QPSetPass._reverse_group`).

WHAT BREAKS IF THIS FAILS. Each state's terms join its B_p_bar in frequency
order (`ordered_sliced_sum`), collectives of every rank per sum. One sum per
state made a collective of every rank per state (with the earlier hand-over
chain, nstates x nfreq of them a group: CzBN1, 2 nodes: 4471 calls, 55 s a
rank). Consecutive states of one slice owner share a sum as one stacked sum:
the same additions, element by element, so the same bits.

Water, cc-pVDZ, HF, G0W0-BSE S1 in fixed toluene charges on the row fit
(tests/test_force_after_density.py's model), over 2 simulated ranks: the
force is the one of a chain per state (a zero batch budget) bit for bit, and
fewer chains run than states are chained.
"""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.gradients.gwbse.qp_space_time as qp_space_time
from src.Base.utils.parallel.simulated_comm import run_simulated
from tests.test_force_after_density import model_in_charges, molecule, numerics


def force():
    mol = molecule()
    state, h, e = model_in_charges(mol, numerics(mol, rows=True))
    state(h, e)
    return state.chain.excitation_gradient(state.mol, state.mf)[0]


def test_states_share_their_chains(monkeypatch):
    calls, lock, chain = [], threading.Lock(), qp_space_time.ordered_sliced_sum

    def counted(terms, owners, comm, onto, root=None):
        if comm is not None and comm.Get_rank() == 0:
            with lock:
                calls.append(np.shape(onto)[0])
        return chain(terms, owners, comm, onto, root=root)
    monkeypatch.setattr(qp_space_time, 'ordered_sliced_sum', counted)
    out = run_simulated(lambda comm: force(), 2)
    batched = list(calls)
    calls.clear()
    monkeypatch.setattr(qp_space_time, 'QP_CHAIN_BATCH_GB', 0.0, raising=False)
    single = run_simulated(lambda comm: force(), 2)
    for grad, one in zip(out, single):
        np.testing.assert_array_equal(grad, one)
        np.testing.assert_array_equal(grad, out[0])
    assert set(calls) == {1}, calls
    assert batched and max(batched) > 1, batched
    assert len(batched) < sum(batched), batched
