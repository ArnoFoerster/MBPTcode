"""WHICH MEAN FIELD a property routine evaluates a surface on.

`scf_factory` is the raw user factory. A surface standing in an environment
builds its mean field THROUGH that environment -- PCM at the static dielectric
for a continuum, the classical charges in h_core for a point-charge field -- so
the factory's mean field belongs to a different surface. Every routine here
that reached for the factory was therefore reporting a geometry relaxed in the
environment at its gas-phase energy: 0.25 eV on water/cc-pVDZ in
PCM(1.78/78.39) and 25 mHa in a pair of half-charges 3 Angstrom away, which is
larger than any reorganization energy such a record is used to compute.

Each case carries its NEGATIVE CONTROL -- the factory's energy, asserted to
differ -- so the gate cannot pass by the two mean fields happening to coincide.

Every check ASSERTS: pytest discards a returned verdict and passes on False.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.environment import PointCharges
from src.Base.solvent_screening import SolventScreening
from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.surface import (FiniteDifferenceGradient,
                                    surface_mean_field)
from src.properties.vibronic import energy_at, relax_state

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Water: eps at the optical frequency and the static one.
EPS, EPS_STATIC = 1.78, 78.39


def hf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


class RecordingSurface:
    """A surface whose mean field is deliberately NOT its factory's.

    It records which one every energy was evaluated on, which is the only way
    to gate the finite-difference wrapper without paying for 6 natm SCFs: the
    defect was in the DISPLACED energies, not in the reference one.
    """

    def __init__(self, mol):
        self.mol0 = mol
        self.seen = []

    def scf_factory(self, mol):
        return 'factory'

    def mean_field(self, mol=None, mf=None):
        return (self.mol0 if mol is None else mol), ('own' if mf is None else mf)

    def total_energy(self, mol=None, mf=None):
        mol, mf = self.mean_field(mol, mf)
        self.seen.append(mf)
        d = mol.atom_coords()[1] - mol.atom_coords()[0]
        return float(np.linalg.norm(d))

    def total_gradient(self, mol=None, mf=None):
        return np.zeros((self.mol0.natm, 3)), self.total_energy(mol, mf), {}

    def refreeze(self, mol):
        return type(self)(mol)

    def label(self):
        return 'recording toy surface'


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=H2O, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def solvated(water):
    """The dRPA ground state in the continuum, and its gas-phase control."""
    env = SolventScreening(water, eps=EPS, eps_static=EPS_STATIC)
    return RPAGroundStateChain(water, hf_factory, environment=env)


# ------------------------------------------------------------------ the helper
def test_the_helper_returns_the_surfaces_own_mean_field(water, solvated):
    """The continuum is in it, which `scf_factory` never puts there."""
    mf = surface_mean_field(solvated, water)
    assert hasattr(mf, 'with_solvent'), 'the mean field came back bare'
    assert abs(mf.with_solvent.eps - EPS_STATIC) < 1e-12
    assert not hasattr(solvated.scf_factory(water), 'with_solvent')


def test_the_helper_falls_back_to_the_factory(water):
    """A surface with no environment to build one through -- the toy surfaces
    and the dense route -- must still answer."""
    class NoMeanField:
        mol0 = None

        def scf_factory(self, mol):
            return 'factory'

    assert surface_mean_field(NoMeanField(), water) == 'factory'


def test_the_composed_surfaces_expose_their_mean_field():
    """Both halves are evaluated on ONE mean field, and the composed surface is
    what a property routine or a manifold is handed, so it has to be reachable
    from there rather than only from `ground`."""
    for cls in (RPABSESurface, RPAQPSurface):
        assert 'mean_field' in vars(cls), f'{cls.__name__} hides its mean field'


# ------------------------------------------------- the solvated ground state
def test_energy_at_is_on_the_solvated_surface(water, solvated):
    """`energy_at` is an off-diagonal entry of a four-point reorganization
    scheme, so it must be the same functional as the diagonal ones."""
    own = solvated.total_energy(water)
    assert energy_at(solvated, water) == pytest.approx(own, abs=1e-10)
    # the control: the factory's mean field is a different surface entirely
    bare = solvated.total_energy(water, solvated.scf_factory(water))
    assert abs(own - bare) > 1e-3, f'|own - factory| = {abs(own - bare):.2e} Ha'


def test_the_finite_difference_wrapper_carries_the_environment(water, solvated):
    """The wrapper is the route an energy-only solvated surface takes, so the
    mean field must not be lost at the wrap."""
    fd = FiniteDifferenceGradient(solvated)
    assert hasattr(surface_mean_field(fd, water), 'with_solvent')


def test_every_displaced_energy_builds_its_own_mean_field(water):
    """Counted on a toy surface rather than paid for: 6 natm SCFs would make
    this gate cost more than the defect."""
    rec = RecordingSurface(water)
    grad, e0, diags = FiniteDifferenceGradient(rec, h=1e-3).total_gradient()
    assert rec.seen, 'nothing was evaluated'
    assert set(rec.seen) == {'own'}, f'{rec.seen.count("factory")} energies on ' \
                                     f'the factory mean field'
    assert e0 == pytest.approx(rec.total_energy(), rel=1e-14)
    assert diags['fd_step'] == 1e-3
    assert grad.shape == (water.natm, 3)


def test_a_relaxed_record_is_on_the_surface_it_was_relaxed_on(water, solvated):
    """`max_cycle=0` costs ONE gradient and still walks the whole record path.

    The record's `mf` is what the vibronic analysis takes its normal modes and
    its `e_scf` from, so a bare one there puts the Huang-Rhys factors on a
    different surface from the geometry.
    """
    record = relax_state(solvated, water, engine='cartesian', max_cycle=0,
                         verbose=False)
    assert hasattr(record['mf'], 'with_solvent'), 'the record came back bare'
    assert record['e_scf'] == pytest.approx(float(record['mf'].e_tot), rel=1e-14)
    assert record['e_total'] == pytest.approx(
        solvated.total_energy(record['mol']), abs=1e-10)


# ------------------------------------------------- the composed BSE surface
def test_energy_at_is_on_the_solvated_bse_surface(water):
    """The composed surface answers `mean_field` itself; before it did not, and
    `energy_at` silently dropped the continuum from E_0 AND from Omega."""
    env = SolventScreening(water, eps=EPS, eps_static=EPS_STATIC)
    s = RPABSESurface(water, hf_factory, environment=env)
    assert s.mean_field(water)[1] is s.ground.mf0
    assert energy_at(s, water) == pytest.approx(s.total_energy(water), abs=1e-10)


# -------------------------------------------------------- a classical field
def test_point_charges_reach_the_property_layer(water):
    """The same hole, on an environment that enters the mean field ALONE:
    everything the charges do is in h_core, so the factory's mean field is the
    only place their effect could go missing."""
    charges = PointCharges([[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]], [0.5, -0.5])
    chain = RPAGroundStateChain(water, hf_factory, environment=charges)
    assert hasattr(surface_mean_field(chain, water), 'mm_mol')
    own = chain.total_energy(water)
    assert energy_at(chain, water) == pytest.approx(own, abs=1e-10)
    bare = chain.total_energy(water, chain.scf_factory(water))
    assert abs(own - bare) > 1e-3, f'|own - factory| = {abs(own - bare):.2e} Ha'
