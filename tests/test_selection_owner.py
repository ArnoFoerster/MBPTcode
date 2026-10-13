"""Only the chain that started the adaptive selection refines or verifies it.

A spin view is a shallow copy and shares `_selecting` by reference; the
selection's rounds belong to the chain that opened it. The owner is a token
per chain, not `id(self)`, which CPython hands to the next object allocated at
a freed chain's address. Here every identity is made equal, the reuse case at
its limit: a copy must still refuse to refine or verify what it did not start.
"""
import copy

from src.gradients import excited_state
from src.gradients.excited_state import ExcitedStateChain


def selecting_chain():
    """A chain holding a selection it opened, nothing else built."""
    chain = ExcitedStateChain.__new__(ExcitedStateChain)
    chain.__dict__.update(qp_partition=None, qp_set=[],
                          _selection_owner=object())
    chain._selecting = {'owner': chain._selection_owner,
                        'verify_pending': True, 'candidates': ()}
    return chain


def test_a_copy_does_not_take_over_the_selection(monkeypatch):
    monkeypatch.setattr(excited_state, 'id', lambda obj: 0, raising=False)
    chain = selecting_chain()
    view = copy.copy(chain)
    assert view._selecting is chain._selecting
    assert view._refine_selection(None, None, None) is False
    assert view.verify_selection(None, None) is False
    assert chain._owns_selection(chain._selecting)
