"""A manifold on a chain whose 'auto' solver resolves dense takes any root.

WHAT BREAKS IF THIS FAILS. `StateManifold._require_reachable` refusing every
root at or above `nroots` unless the chain's `solver` is literally 'dense'
would refuse for nothing: 'auto' that resolves dense on the chain's pair space
solves every root. The manifold below is accepted, not a ValueError.
"""
from types import SimpleNamespace

import numpy as np

from src.Base.numerics.numerics import Numerics
from src.gradients.gwbse.conventions import FrozenConventions
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.state_manifold import StateManifold

#: A small pair space: 5 occupied, 20 orbitals, which 'auto' solves densely.
NOCC, NMO = 5, 20


def chain(solver, nroots=2):
    """A chain with nothing built but what the manifold's check reads."""
    out = ExcitedStateChain.__new__(ExcitedStateChain)
    out.settings = Numerics()
    out.conventions = FrozenConventions()
    out.solver, out.nroots, out.state = solver, nroots, 0
    out.bse_tda = False
    out.nocc = NOCC
    out.mf0 = SimpleNamespace(mo_energy=np.zeros(NMO))
    out._owner = object()
    return out


def test_auto_resolving_dense_reaches_every_root():
    assert chain('auto').solver_used(NOCC * (NMO - NOCC)) == 'dense'
    StateManifold(chain('auto'), states=(0, 1, 2, 3))


def test_davidson_still_refuses_roots_it_does_not_solve():
    try:
        StateManifold(chain('davidson'), states=(0, 1, 2, 3))
    except ValueError as exc:
        assert 'nroots' in str(exc)
    else:
        raise AssertionError('a Davidson chain solving 2 roots took root 3')
