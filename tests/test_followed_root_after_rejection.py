"""The root a walk followed is read at the geometry the walk returned.

`ExcitedStateChain.tracked_state` logs a root at every evaluation, and an
optimizer evaluates the steps it then rejects: after a trust radius collapses
the last entry of `follow_log` is the root of a geometry the walk did not stay
at. `followed_root`, which `refreeze` and `excitations.adaptive_check` read,
takes the entry evaluated at the returned geometry.

The surface here is a stub that logs as a tracked chain does: root 0 at the
start, root 1 wherever else it is evaluated, and an energy that rises along
every step, so every step is rejected and the walk returns its start.
"""
import numpy as np
from pyscf import gto

import src.gradients  # noqa: F401  cycle: src.properties imports src.gradients
from src.properties.optimize import optimize
from src.properties.surface import followed_root

class Crossing:
    """A walked state whose root changes off the start and whose energy rises
    along any step, though its force points downhill."""

    def __init__(self, mol):
        self.mol0, self.state = mol, 0
        self.start = mol.atom_coords().copy()
        self.follow_log, self.follow_coords = [], []

    def total_gradient(self, mol=None, mf=None):
        here = mol.atom_coords()
        moved = float(np.abs(here - self.start).max())
        self.follow_log.append({'index': 0 if moved == 0.0 else 1})
        self.follow_coords.append(here.copy())
        g = np.zeros_like(here)
        g[0, 2] = 1e-2
        return g, moved, {}


def water():
    return gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59',
                 basis='sto-3g', verbose=0)


def test_a_collapsed_walk_reads_the_root_where_it_stopped():
    mol = water()
    surface = Crossing(mol)
    mol_opt, info = optimize(surface, mol, verbose=False)
    assert 'collapsed' in info['status']
    assert np.array_equal(mol_opt.atom_coords(), surface.start)
    assert surface.follow_log[-1]['index'] == 1, 'the rejected step is last'
    assert followed_root(surface, mol_opt) == 0

