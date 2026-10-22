"""Evaluate once per geometry, by ONE mechanism: the chain's evaluation cache.

A manifold over a surface whose forward is not split at the Casida step runs
no forward pass rebound over every root: that pass would be solved for
whichever root the chain had selected last, so a higher root of the manifold
would read eigenvectors from a solve that never guaranteed it converged, and
it would be a second way of evaluating once beside the cache the roots
share. The roots go through the cache, highest first, so one forward pass
solved for the highest root serves them all.

The adaptive set's re-check on the production eigenvectors exists once
(`ExcitedStateChain.verified`), not in a loop of the manifold.

H2/cc-pVDZ Hartree-Fock on the G1 grid, three singlet roots.
"""
import pytest
from pyscf import gto

import src.gradients.state_manifold as state_manifold
from src.Base.numerics import Numerics
from src.Base.separable_ri import resolve_isdf_grid
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.state_manifold import StateManifold
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'
ROOTS = (0, 1, 2)


class Unsplit:
    """A surface holding an excited chain whose manifold does not share the
    split forward (no `composed`): E = E_KS + Omega through the chain."""

    def __init__(self, excited):
        self.excited, self.mol0 = excited, excited.mol0

    def scf_factory(self, mol):
        return self.excited.scf_factory(mol)

    def mean_field(self, mol=None, mf=None):
        return self.excited.mean_field(mol, mf)

    def total_energy(self, mol=None, mf=None):
        return self.excited.total_energy(mol, mf)


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom=H2, basis=BASIS, verbose=0)


@pytest.fixture
def chain(h2):
    counts = resolve_isdf_grid('G1', BASIS, ['H'])[0]
    return ExcitedStateChain(h2, rhf_df_differentiable_checked,
                             numerics=Numerics(isdf={'counts': counts}))


def test_pinned_forward_is_retired():
    assert not hasattr(state_manifold, 'pinned_forward')


def test_one_forward_solved_for_the_highest_root(h2, chain, monkeypatch):
    man = StateManifold(Unsplit(chain), states=ROOTS)
    assert not man.shares_forward
    solved_for = []
    original = ExcitedStateChain._forward_pass

    def spy(self, mol, mf):
        solved_for.append(int(self.state))
        return original(self, mol, mf)
    monkeypatch.setattr(ExcitedStateChain, '_forward_pass', spy)
    energies = man.energies(h2)
    assert solved_for == [max(ROOTS)], solved_for
    assert list(energies) == list(ROOTS)
    alone = {n: man.energy(n, h2) for n in ROOTS}
    assert solved_for == [max(ROOTS)], 'the roots were evaluated again'
    assert all(energies[n] == alone[n] for n in ROOTS)


def test_the_manifold_rechecks_the_set_through_the_chain(h2, chain,
                                                         monkeypatch):
    calls = []
    original = ExcitedStateChain.verified

    def spy(self, *args):
        calls.append(type(self).__name__)
        return original(self, *args)
    monkeypatch.setattr(ExcitedStateChain, 'verified', spy)
    chain.excitation(h2)
    StateManifold(chain, states=ROOTS).evaluate(h2)
    assert len(calls) == 2, calls
