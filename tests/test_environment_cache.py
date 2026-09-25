"""The reaction field a chain uses at each geometry, cached on CONTENT.

A cavity moves with the atoms, so every displaced geometry needs its own, and a
finite difference of the energy is the derivative of the real surface only if
two displaced points never share one.

WHAT THE CONTENT KEY ACTUALLY BUYS. Keyed on `id(mol)` the cache was safe, and
safe even when bounded, but only by accident of what the environments return:
a resident entry holds a rebuilt cavity, that cavity holds its own molecule, so
the address cannot be recycled while the entry lives, and collisions therefore
only ever land on evicted keys, which miss and rebuild correctly. A static
environment returning a singleton is safe for the unrelated reason that any hit
is the right object.

The key is content because that invariant is not one the class states or
enforces. An environment that is geometry-specific and does NOT keep its
molecule breaks it, and `test_a_non_pinning_environment_needs_the_content_key`
measures how badly. Content-keying also makes two molecule objects at one
geometry share one cavity, which identity-keying cannot.

The tests here drive `FactorChain.environment_at` directly on a stub holding a
real `SolventScreening`. That is the production method, and it avoids paying for
a whole chain's radii optimization to exercise ten lines of caching.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import collections
import gc

import numpy as np
import pytest
from pyscf import gto

from src.Base.constants import BOHR_TO_ANGSTROM, ENVIRONMENT_CACHE_SIZE
from src.Base.solvent_screening import SolventScreening
from src.gradients.factor_chain import FactorChain


class Holder:
    """The real method, with only the two attributes it touches."""

    environment_at = FactorChain.environment_at

    def __init__(self, environment):
        self.environment = environment
        self._environment_cache = collections.OrderedDict()


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                 basis='sto-3g', verbose=0)


@pytest.fixture
def holder(mol):
    return Holder(SolventScreening(mol, solvent='water'))


def displaced(mol, atom, comp, step):
    crd = mol.atom_coords() * BOHR_TO_ANGSTROM
    crd[atom, comp] += step * BOHR_TO_ANGSTROM
    m = mol.copy()
    m.set_geom_(crd, unit='Angstrom')
    m.build(False, False)
    return m


def test_every_geometry_gets_a_cavity_built_for_itself(holder, mol):
    """A finite-difference sweep with the cache bounded well below the number of
    geometries, so entries are evicted while it runs. Each returned cavity must
    sit on the atoms that asked for it.

    This holds for an identity key too, so it discriminates nothing on its own;
    it is here because it is the property that must never break, whatever the
    key.
    """
    worst = 0.0
    n = 0
    for atom in range(mol.natm):
        for comp in range(3):
            for step in (1e-3, -1e-3):
                m = displaced(mol, atom, comp, step)
                env = holder.environment_at(m)
                worst = max(worst, float(np.abs(
                    np.asarray(env.mol.atom_coords())
                    - np.asarray(m.atom_coords())).max()))
                n += 1
                del m
    gc.collect()
    assert n == 18
    assert worst == 0.0, (f'a cavity was returned for the wrong geometry, off '
                          f'by {worst:.3e} Bohr')
    assert len(holder._environment_cache) <= ENVIRONMENT_CACHE_SIZE


def test_a_non_pinning_environment_needs_the_content_key():
    """THE GATE THAT DISCRIMINATES, run against both keys in one test.

    An identity-keyed cache is safe only while every cached value keeps its own
    molecule alive. This environment is geometry-specific and stores only the
    coordinates, so nothing pins the key: a resident entry's molecule is
    collected, CPython reuses the address, and the next geometry reads the
    previous one's field. Measured here at 17 of 18 displaced geometries, wrong
    by the displacement itself -- and self-consistent, so no energy or gradient
    gate would show it.
    """
    class Lean:
        """Geometry-specific, and does NOT keep its molecule."""

        differentiable = True

        def __init__(self, coords=None):
            self.coords = coords

        def for_geometry(self, m):
            return Lean(np.asarray(m.atom_coords()).copy())

    def identity_keyed(self, mol):
        key = id(mol)
        hit = self._environment_cache.get(key)
        if hit is not None:
            self._environment_cache.move_to_end(key)
            return hit
        hit = self.environment.for_geometry(mol)
        self._environment_cache[key] = hit
        while len(self._environment_cache) > ENVIRONMENT_CACHE_SIZE:
            self._environment_cache.popitem(last=False)
        return hit

    base = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                 basis='sto-3g', verbose=0)

    def stale_hits(impl):
        holder = Holder(Lean())
        bound = impl.__get__(holder, Holder)
        bad = 0
        for atom in range(base.natm):
            for comp in range(3):
                for step in (1e-3, -1e-3):
                    m = displaced(base, atom, comp, step)
                    env = bound(m)
                    if float(np.abs(env.coords
                                    - np.asarray(m.atom_coords())).max()) > 0.0:
                        bad += 1
                    del m
        gc.collect()
        return bad

    assert stale_hits(identity_keyed) > 0, (
        'the identity key must be shown to fail here, or this test proves '
        'nothing about the content key')
    assert stale_hits(FactorChain.environment_at) == 0


def test_two_molecule_objects_at_one_geometry_share_a_cavity(holder, mol):
    """What the content key buys over identity: the same geometry is the same
    cavity, however many Mole objects describe it. Identity-keying rebuilt one
    per object."""
    calls = {'n': 0}
    original = type(holder.environment).for_geometry

    def counted(self, m):
        calls['n'] += 1
        return original(self, m)

    type(holder.environment).for_geometry = counted
    try:
        a = displaced(mol, 0, 2, 1e-3)
        b = displaced(mol, 0, 2, 1e-3)
        assert a is not b
        env_a = holder.environment_at(a)
        env_b = holder.environment_at(b)
    finally:
        type(holder.environment).for_geometry = original
    assert calls['n'] == 1, 'one geometry must be built once'
    assert env_a is env_b


def test_the_key_is_exact_bytes_not_rounded(holder, mol):
    """Two geometries a hair apart must NOT share a cavity. Rounding the key --
    to six decimals, say -- would merge neighbouring finite-difference
    displacements, and the resulting force is wrong in a way that is
    self-consistent and therefore invisible."""
    a = displaced(mol, 0, 2, 1e-3)
    b = displaced(mol, 0, 2, 1e-3 + 1e-12)
    assert not np.array_equal(a.atom_coords(), b.atom_coords())
    assert holder.environment_at(a) is not holder.environment_at(b)
    assert len(holder._environment_cache) == 2


def test_the_charges_are_in_the_key(mol):
    """A cavity follows the element radii, so coordinates alone do not
    determine one: the same three positions with different elements is a
    different surface."""
    swapped = gto.M(atom='N 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                    basis='sto-3g', spin=1, verbose=0)
    h = Holder(SolventScreening(mol, solvent='water'))
    assert not np.array_equal(mol.atom_charges(), swapped.atom_charges())
    assert np.array_equal(mol.atom_coords(), swapped.atom_coords())
    assert h.environment_at(mol) is not h.environment_at(swapped)


def test_the_cache_is_bounded(holder, mol):
    """Retention was the whole cost of the old version: every geometry a
    relaxation ever visited stayed alive, cavity and molecule both."""
    for i in range(6):
        holder.environment_at(displaced(mol, 1, 0, (i + 1) * 1e-3))
    assert len(holder._environment_cache) == ENVIRONMENT_CACHE_SIZE


def test_the_reference_geometry_is_still_the_environments_own_object(holder, mol):
    """`SolventScreening.for_geometry` returns `self` for the molecule it was
    built around, so the reference geometry must not get a rebuilt copy."""
    assert holder.environment_at(mol) is holder.environment
