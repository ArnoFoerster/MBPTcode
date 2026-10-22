"""Copies of a chain own their conventions' keys and share its frozen values.

WHAT BREAKS IF THIS FAILS. A spin view, a root copy and the surface
`StateManifold.surface` hands a walk, if shallow copies sharing ONE
`FrozenConventions`, would re-key every sibling when one walks, since `commit`
moves its keys (and rotates the group seeds into them): the manifold, and a
second state walked in the same process, would match their next geometry
against keys from somebody else's walk. Each copy taken after the conventions
are anchored owns its keys (`keyed_copy`), and every
frozen value is the same object in all of them.

H2/cc-pVDZ Hartree-Fock on the dense Casida route.
"""
import copy

import numpy as np
from pyscf import gto

from src.Base.numerics import Numerics
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.state_manifold import StateManifold
from tests.isdf_test_grid import g1_numerics
from tests.reference_scf import rhf_df_differentiable

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'
H2_STEP = 'H 0 0 0; H 0 0 0.75'
S0, T0 = ('singlet', 0), ('triplet', 0)


def molecule(atom=H2):
    return gto.M(atom=atom, basis=BASIS, verbose=0)


def evaluated():
    mol = molecule()
    chain = ExcitedStateChain(mol, rhf_df_differentiable, mf=rhf_df_differentiable(mol),
                              numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    chain.total_energy(mol)
    return chain


def keys(chain):
    c = chain.conventions
    return c.key_mol, c.commits


def walk_one_step(surface):
    step = molecule(H2_STEP)
    surface.total_energy(step)
    surface.commit(step)
    return step


def test_walking_one_spin_view_does_not_re_key_its_sibling():
    chain = evaluated()
    singlet, triplet = chain.spin_view('singlet'), chain.spin_view('triplet')
    before = keys(singlet), keys(chain)
    step = walk_one_step(triplet)
    assert triplet.conventions.key_mol is step
    assert triplet.conventions.commits == 1
    assert (keys(singlet), keys(chain)) == before
    # the frozen values are one set of objects
    for name in ('qp_seeds', 'qp_seed_weights', 'sop_poles', 'outside_shift',
                 'scissor_map', 'qp_z_outside'):
        assert getattr(triplet.conventions, name) is getattr(
            chain.conventions, name), name


def test_walking_a_manifold_surface_does_not_re_key_the_manifold():
    chain = evaluated()
    man = StateManifold(chain, states=(S0, T0))
    man.evaluate()
    walk = man.surface(T0)
    other = man.surface(S0)
    before = keys(chain), keys(other)
    walk_one_step(walk)
    assert walk.conventions.commits == 1
    assert (keys(chain), keys(other)) == before


def test_a_copy_before_anchoring_shares_the_conventions_whole():
    mol = molecule()
    chain = ExcitedStateChain(mol, rhf_df_differentiable, mf=rhf_df_differentiable(mol),
                              numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    early = copy.copy(chain)
    assert early.conventions is chain.conventions
    early.total_energy(mol)
    assert chain.conventions.anchored
    assert np.array_equal(chain.conventions.key_mo, early.conventions.key_mo)
