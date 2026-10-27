"""A handed-over first point replays only on the conventions it was taken on,
and one that is discarded says so.

`StateManifold.surface(n, first_point=ev)` hands a surface the force `ev`
evaluated, which its first `total_gradient` returns without a forward pass.
A force is a force of the frozen conventions as well as of the geometry, the
spin and the root: on another set of conventions it is another surface's.
And a walk that starts elsewhere (a nudged start) drops the point, which
costs a whole force evaluation and must not pass unremarked.

H2/cc-pVDZ Hartree-Fock on the dense Casida route: the smallest chain a
manifold evaluates.
"""
import warnings
from dataclasses import replace

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.numerics import Numerics
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.state_manifold import StateManifold
from tests.isdf_test_grid import g1_numerics
from tests.reference_scf import rhf_df_differentiable

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'
H2_NUDGED = 'H 0 0 0; H 0 0 0.76'
S0, T0 = ('singlet', 0), ('triplet', 0)


def molecule(atom=H2):
    return gto.M(atom=atom, basis=BASIS, verbose=0)


@pytest.fixture
def handed_over(monkeypatch):
    """(manifold, its evaluation of S0 and T0, the forward passes counted)."""
    mol = molecule()
    chain = ExcitedStateChain(mol, rhf_df_differentiable, mf=rhf_df_differentiable(mol),
                              numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    man = StateManifold(chain, states=(S0, T0))
    ev = man.evaluate(gradients=(S0, T0))
    forwards, shared = [], ExcitedStateChain.shared_forward

    def counted(self, *args):
        forwards.append(1)
        return shared(self, *args)

    monkeypatch.setattr(ExcitedStateChain, 'shared_forward', counted)
    return man, ev, forwards


def test_the_point_replays_on_its_own_conventions(handed_over):
    man, ev, forwards = handed_over
    walk = man.surface(T0, first_point=ev)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        first = walk.total_gradient(molecule())
    assert forwards == [] and first[0] is ev.gradient[T0]


def test_other_conventions_do_not_replay_the_point(handed_over):
    man, ev, forwards = handed_over
    walk = man.surface(T0, first_point=ev)
    # the surface freezes something else than the force was taken on
    walk.conventions = replace(walk.conventions,
                               nfreq_cd_floor=walk.conventions.nfreq_cd + 1)
    with pytest.warns(UserWarning, match='first point.*conventions'):
        out = walk.total_gradient(molecule())
    assert forwards == [1], 'the force of other conventions was replayed'
    assert out[0] is not ev.gradient[T0]


def test_a_discarded_point_warns(handed_over):
    man, ev, forwards = handed_over
    walk = man.surface(T0, first_point=ev)
    with pytest.warns(UserWarning, match='first point.*geometry'):
        walk.total_gradient(molecule(H2_NUDGED))
    assert forwards == [1]
    # taken off either way: a later visit is evaluated afresh, silently
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        again = walk.total_gradient(molecule())
    assert forwards == [1, 1]
    assert np.abs(again[0] - ev.gradient[T0]).max() < 1e-9
