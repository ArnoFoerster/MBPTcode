"""Every class that offers a nuclear gradient must be a WHOLE surface.

`src/properties` is written against `PotentialEnergySurface` and nothing else:
the optimizer, the Huang-Rhys spectrum, the adiabatic gaps and the Marcus rates
all reach for `total_energy`, `refreeze` and `label` on whatever they are
handed. A method that implements `total_gradient` alone is a gradient, not a
surface, and it fails at the first property routine rather than at import --
which is how `CRPAEmbeddingChain` shipped able to compute forces but unable to
be optimized.

The classes are DISCOVERED, not listed, so a surface added later is held to the
same bar without anyone remembering to add it here.

`mean_field(mol) -> (mol, mf)` is the protocol's accessor for the mean field a
surface evaluates its energy on, and is required of every surface here: a
solvated surface's own mean field is the environment's ground state, not the
raw `scf_factory`'s, and the two are 0.24 eV apart on water in PCM. A gas-phase
surface owes the accessor too, returning its factory's mean field, so that a
property routine never has to ask which kind it was handed.

`mol0` is an instance attribute on the chains -- `FactorChain` binds it in
`__init__` -- so it is checked on a constructed object, while the methods are
checked on the class.

A surface built through `potential_energy_surface` owes MORE than the protocol:
`physics` says which functional its E_0 is and which state sits on it,
`realization` says how that was computed, `numerics` the grids it resolved, and
`describe()` prints all three with E_0's terms evaluated. Without them a
property routine holding two surfaces cannot tell whether their energies may be
differenced, which is the 0.9 eV adiabatic disagreement this layer exists to
stop.
"""
import importlib
import inspect
import os
import pathlib
import pkgutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pyscf import dft, gto, scf

from src.Base.declaration import (ChargedExcitation, Excitation, GroundState,
                                  SurfacePhysics)
from src.properties.surfaces import potential_energy_surface

#: The protocol's methods. `mol0` is data, checked separately.
REQUIRED = ('total_energy', 'total_gradient', 'refreeze', 'label')
#: The accessor for the mean field the surface evaluates its energy on.
ACCESSOR = 'mean_field'
#: A class offering this is claiming to be a surface.
MARKER = 'total_gradient'
PACKAGES = ('src.gradients', 'src.properties')


def surface_classes():
    """Every class in the gradient and property packages that offers a gradient."""
    found = {}
    for package in PACKAGES:
        pkg = importlib.import_module(package)
        for info in pkgutil.iter_modules(pkg.__path__):
            mod = importlib.import_module(f'{package}.{info.name}')
            for name, obj in vars(mod).items():
                if (inspect.isclass(obj) and obj.__module__ == mod.__name__
                        and any(MARKER in vars(k) for k in obj.__mro__)):
                    found[f'{info.name}.{name}'] = obj
    return found


CLASSES = surface_classes()
#: The protocol itself is discovered too; it states the requirements rather
#: than meeting them.
CONCRETE = sorted(n for n, c in CLASSES.items()
                  if not getattr(c, '_is_protocol', False))


def test_the_search_finds_the_surfaces_we_know_about():
    """A discovery that silently found nothing would make every test below pass.

    `CRPAEmbeddingChain` is not among them: src.Embedding is out of scope for
    this tree, so its surface never ships here.
    """
    names = {n.split('.')[-1] for n in CLASSES}
    assert {'RPAGroundStateChain', 'ExcitedStateChain', 'DenseRPASurface',
            'QuasiparticleSurface', 'DenseBSESurface'} <= names


@pytest.mark.parametrize('name', sorted(CLASSES))
@pytest.mark.parametrize('member', REQUIRED)
def test_a_surface_carries_the_whole_protocol(name, member):
    cls = CLASSES[name]
    assert any(member in vars(k) for k in cls.__mro__), (
        f'{name} offers {MARKER} but not {member}: it is a gradient, not a '
        f'surface, and src/properties cannot step on it')


@pytest.mark.parametrize('name', CONCRETE)
def test_a_surface_names_the_mean_field_it_evaluates_on(name):
    """`mean_field` is the one accessor a property routine may use.

    A surface in an environment evaluates its energy on the environment's
    ground state, and `scf_factory`'s mean field is a different surface --
    0.24 eV of dRPA total energy on water in PCM. A gas-phase surface owes the
    accessor as well, returning `(mol, self.scf_factory(mol))`, so that
    `surface_mean_field` never has to guess which kind it holds.
    """
    cls = CLASSES[name]
    assert any(ACCESSOR in vars(k) for k in cls.__mro__), (
        f'{name} is a surface but does not define {ACCESSOR}(mol) -> '
        f'(mol, mf): a property routine reaching for the mean field it was '
        f'evaluated on falls back to the raw factory, which in an environment '
        f'is not this surface')


def water():
    """The one geometry the dispatched surfaces below are built at."""
    return gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                 basis='cc-pvdz', verbose=0)


def rhf(mol):
    """A reference converged tightly enough to differentiate."""
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    return mf


def pbe0(mol):
    """The Kohn-Sham starting point of the dft row."""
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.grids.level = 5
    mf.kernel()
    return mf


#: One dispatched surface per (ground-state kind, state type), covering both the
#: cubic space-time route and the dense quasi-boson oracle.
DISPATCHED = {
    'mean field': (pbe0, dict(ground_state=GroundState('dft', 'pbe0'))),
    'dRPA ground state': (rhf, dict(ground_state=GroundState('rpa', 'hf'))),
    'dense dRPA ground state': (
        rhf, dict(ground_state=GroundState('rpa', 'hf'), chi0='dense-qb',
                  factorization='four-index')),
    'dense BSE@GW': (
        rhf, dict(ground_state=GroundState('rpa', 'hf'),
                  excitation=Excitation('singlet'), chi0='dense-qb',
                  factorization='four-index')),
    'dense quasiparticle': (
        rhf, dict(ground_state=GroundState('rpa', 'hf'),
                  excitation=ChargedExcitation(4, -1), chi0='dense-qb',
                  factorization='four-index')),
}

#: What `potential_energy_surface` attaches on top of the protocol.
DECLARED = ('physics', 'realization', 'numerics', 'describe')

#: The realizing classes, each of which DECLARES its own `physics` and is
#: therefore built directly below as well as dispatched. Named rather than
#: derived, so that a row whose class gained no declaration shows up as a
#: missing case instead of as nothing at all.
DIRECT_NAMES = ('DenseBSESurface', 'DenseRPASurface', 'ExcitedStateChain',
                'MeanFieldSurface', 'QuasiparticleSurface', 'RPABSESurface',
                'RPAGroundStateChain', 'RPAQPSurface')


@pytest.fixture(scope='module')
def dispatched():
    """Every row of DISPATCHED, built once."""
    mol = water()
    return {name: potential_energy_surface(mol, factory, **kw)
            for name, (factory, kw) in DISPATCHED.items()}


@pytest.mark.parametrize('name', sorted(DISPATCHED))
@pytest.mark.parametrize('member', DECLARED)
def test_a_dispatched_surface_declares_its_physics(dispatched, name, member):
    assert hasattr(dispatched[name], member), (
        f'{name} came out of potential_energy_surface without {member}: what '
        f'it computes and how would again be implicit in the constructor call')


@pytest.fixture(scope='module')
def references():
    """The two mean fields the direct constructions below share."""
    mol = water()
    return mol, rhf(mol), pbe0(mol)


def direct_surfaces(mol, hf, ks):
    """One DIRECTLY constructed instance per realizing class, with the
    `SurfacePhysics` its constructor arguments mean.

    Built by hand, not through `potential_energy_surface`: the declaration is
    the CLASS's, and a class that only carries one because the entry point
    stamped it cannot be checked against anything -- which is what let a chain
    be handed settings the caller never asked for and still report the
    caller's label.
    """
    from src.gradients.dense_surfaces import (DenseBSESurface, DenseRPASurface,
                                              QuasiparticleSurface)
    from src.gradients.excited_state import ExcitedStateChain
    from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
    from src.gradients.rpa_ground_state import RPAGroundStateChain
    from src.properties.optimize import MeanFieldSurface

    rpa_hf, ks_pbe0 = GroundState('rpa', 'hf'), GroundState('dft', 'pbe0')
    homo = mol.nelectron // 2 - 1
    return {
        'MeanFieldSurface': (
            MeanFieldSurface(mol, pbe0, mf=ks),
            SurfacePhysics(ks_pbe0, None)),
        'RPAGroundStateChain': (
            RPAGroundStateChain(mol, rhf, mf=hf),
            SurfacePhysics(rpa_hf, None)),
        'ExcitedStateChain': (
            ExcitedStateChain(mol, pbe0, mf=ks, spin='triplet', state=1,
                              bse_tda=True),
            SurfacePhysics(ks_pbe0, Excitation('triplet', root=2,
                                               kernel='bse-tda'))),
        'RPABSESurface': (
            RPABSESurface(mol, rhf, mf=hf, state=0, spin='singlet'),
            SurfacePhysics(rpa_hf, Excitation('singlet', root=1))),
        'RPAQPSurface': (
            RPAQPSurface(mol, rhf, mf=hf, state=0),
            SurfacePhysics(rpa_hf, ChargedExcitation(homo, -1))),
        'DenseRPASurface': (
            DenseRPASurface(mol, rhf),
            SurfacePhysics(rpa_hf, None)),
        'DenseBSESurface': (
            DenseBSESurface(mol, scf=rhf),
            SurfacePhysics(rpa_hf, Excitation('singlet', root=1))),
        'QuasiparticleSurface': (
            QuasiparticleSurface(mol, rhf, charge_change=-1, orbital=homo),
            SurfacePhysics(rpa_hf, ChargedExcitation(homo, -1))),
    }


@pytest.fixture(scope='module')
def direct(references):
    return direct_surfaces(*references)


def test_every_dispatched_class_is_built_directly_here(direct):
    """A curated list that fell behind the dispatch table would test nothing."""
    from src.properties.surfaces import dispatch_table
    assert set(direct) == set(DIRECT_NAMES)
    assert {row.cls.__name__ for row in dispatch_table()} <= set(DIRECT_NAMES)


@pytest.mark.parametrize('name', sorted(DIRECT_NAMES))
def test_a_surface_declares_its_physics_without_being_dispatched(direct, name):
    """`physics` is the class's own, on an instance nobody dispatched.

    It says which functional E_0 is and which state sits on it, off the
    constructor arguments alone -- so `potential_energy_surface` can CHECK the
    declaration it built against the one the class carries instead of stamping
    its own over whatever was actually constructed.
    """
    surface, expected = direct[name]
    assert surface.physics == expected, (name, surface.physics, expected)
    assert isinstance(type(surface).physics, property), (
        f'{name}.physics is not a class-level declaration: an instance '
        f'attribute can be stamped from outside, which is the thing this '
        f'test exists to stop')


@pytest.mark.parametrize('name', sorted(DISPATCHED))
def test_the_physics_label_names_the_ground_state(dispatched, name):
    """The printable name of the total energy contains E_0's own name.

    A label that named only the state would put E_HF + Omega and
    E_HF + E_c^dRPA + Omega under one string, which is the 0.9 eV.
    """
    physics = dispatched[name].physics
    assert physics.ground_state.label() in physics.label()
