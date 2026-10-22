"""The quasiparticle modes the excited-state surface does not take are refused
by
name, and the calibration fills only what a route reads.

  * a fixed `scissor` shift for the states the pole model excludes: the
    calibrated one ('calibrate') is the only shift left;
  * `residue_route='auto'`, a residue backend decided per state at every
    geometry, on the chain and at the entry point (`residues='auto'`);
  * `scissor='calibrate'` on a route other than the pole model: the shifts
    are read by the pole-model route alone (`scissor_route`), so none is
    frozen there.

Water/cc-pVDZ, the record's RHF (tests/test_outside_scissor.py).
"""
import pytest

from src.Base.numerics import Numerics
from src.Base.declaration import Excitation, GroundState
from src.gradients.excited_state import ExcitedStateChain
from src.properties.surfaces import potential_energy_surface

from test_outside_scissor import molecule, scf_factory
from tests.isdf_test_grid import g1_numerics


@pytest.fixture(scope='module')
def water():
    mol = molecule('water')
    return mol, scf_factory(mol)


def test_a_fixed_scissor_is_refused(water):
    mol, mf = water
    with pytest.raises(ValueError, match="'calibrate' or None"):
        ExcitedStateChain(
            mol, scf_factory, mf=mf,
            numerics=g1_numerics(mol, Numerics(qp={'residue_route': 'sop', 'scissor': 0.01})))


def test_a_residue_backend_per_geometry_is_refused(water):
    mol, mf = water
    with pytest.raises(ValueError, match="residue_route='auto'"):
        ExcitedStateChain(
            mol, scf_factory, mf=mf,
            numerics=g1_numerics(mol, Numerics(qp={'residue_route': 'auto'})))
    with pytest.raises(ValueError, match="residues 'auto' not in"):
        potential_energy_surface(mol, scf_factory,
                                 ground_state=GroundState('dft', 'hf'),
                                 excitation=Excitation('singlet'),
                                 chi0='space-time', residues='auto')


def test_the_calibration_freezes_nothing_off_the_pole_model(water):
    """Every explicit root took the explicit backend: no shift is frozen for
    a route that never reads one."""
    mol, mf = water
    chain = ExcitedStateChain(
        mol, scf_factory, mf=mf,
        numerics=g1_numerics(mol, Numerics(qp={'residue_route': 'explicit', 'scissor': 'calibrate'})))
    chain.excitation(mol, mf)
    assert chain.resolved.scissor_map == {}
    assert set(chain.qp_diagnostics['routes'].values()) == {'explicit'}
