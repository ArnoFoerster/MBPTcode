"""A refrozen Cartesian walk starts at the trust radius the caller asked for.

WHAT BREAKS IF THIS FAILS. A refrozen walk started with the radius the walk
before it ENDED with, handed on by a closure over the mutated local, would
start after rejected steps at a fraction of the asked-for radius, and after a
collapse at `trust_min`, a refreeze pass that cannot move. Here the first walk
is held below the asked-for radius by `trust_max`; the refrozen walk starts at
the asked-for radius, not at that cap.
"""
import numpy as np
import pytest
from pyscf import gto

from src.properties.optimize import optimize

#: Bohr: the radius asked for, and the cap every accepted step clips it to.
TRUST = 0.1
TRUST_MAX = 0.05


class Bond:
    """E = k (r - r_e)^2 / 2 in the bond length."""

    def __init__(self, mol):
        self.mol0 = mol

    def total_gradient(self, mol=None, mf=None):
        here = mol.atom_coords()
        d = here[1] - here[0]
        r = float(np.linalg.norm(d))
        grad = np.zeros_like(here)
        grad[1] = 0.4 * (r - 2.2) * d / r
        grad[0] = -grad[1]
        return grad, 0.2 * (r - 2.2) ** 2, {}

    def refreeze(self, mol):
        return type(self)(mol)


def test_the_refrozen_walk_starts_at_the_asked_radius():
    mol = gto.M(atom='N 0 0 0; N 0 0 1.6', unit='Bohr', basis='sto-3g',
                verbose=0)
    _, info = optimize(Bond(mol), mol, verbose=False, nudge=0, trust=TRUST,
                       trust_max=TRUST_MAX, refreeze=1)
    assert info['history'][-1]['trust'] <= TRUST_MAX
    assert info['refreeze_info']['history'][0]['trust'] == pytest.approx(TRUST)
