"""E_0 of an adiabatic energy stands on the excited surface's own Numerics.

`ground_surface_of` builds the dRPA ground chain from the excited surface's
`Numerics`: ground-only keywords would be a different `Numerics` (without the
explicit tau count of the static screening, E_c^dRPA would sit on the minimax
count while the excited surface's own ground half used the explicit one). It
hands the excited surface's object to the ground chain, and the seam where E_0
meets the excited energy (`vertical_block`, `adiabatic_energy`) refuses two
surfaces of different identity before a relaxation starts.

H2/cc-pVDZ Hartree-Fock on the G1 grid, dRPA ground state.
"""
import pytest
from pyscf import gto

from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.numerics import (NumericsMismatch, numerics_identity,
                               surface_numerics)
from src.properties import excitations
from src.properties.excitations import (SurfaceSpec, adiabatic_energy,
                                        ground_surface_of, surface_of)
from src.properties.surfaces import potential_energy_surface
from tests.reference_scf import rhf_df_differentiable

BASIS = 'cc-pvdz'
NTAU_W = 14


def spec(ntau_w):
    return SurfaceSpec(GroundState('rpa', 'hf'), chi0='space-time',
                       factorization='isdf', qp_states=QPStates('all'),
                       numerics={'grid_accuracy': 'G1', 'ntau_w': ntau_w})


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom='H 0 0 0; H 0 0 0.74', basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def excited(h2):
    return surface_of(spec(NTAU_W), Excitation('singlet'), h2, rhf_df_differentiable)


def test_the_ground_surface_stands_on_the_excited_numerics(h2, excited):
    ground = ground_surface_of(spec(NTAU_W), excited, h2, rhf_df_differentiable)
    assert numerics_identity(ground) == numerics_identity(excited)
    assert surface_numerics(ground) is surface_numerics(excited)
    assert ground.ntau == NTAU_W


def test_adiabatic_energy_refuses_a_ground_surface_of_other_numerics(
        h2, excited, monkeypatch):
    def started(*args, **kwargs):
        raise AssertionError('a relaxation started')
    monkeypatch.setattr(excitations, 'emission_block', started)
    other = surface_of(spec(NTAU_W + 4), None, h2, rhf_df_differentiable)
    assert numerics_identity(other) != numerics_identity(excited)
    with pytest.raises(NumericsMismatch, match='E_0 against the excited'):
        adiabatic_energy(excited, other, h2)


def test_a_mean_field_ground_surface_holds_no_numerics_and_is_not_refused(h2):
    dft = SurfaceSpec(GroundState('dft', 'hf'), chi0='space-time',
                      factorization='isdf', qp_states=QPStates('all'),
                      numerics={'grid_accuracy': 'G1'})
    excited = surface_of(dft, Excitation('singlet'), h2, rhf_df_differentiable)
    ground = ground_surface_of(dft, excited, h2, rhf_df_differentiable)
    assert numerics_identity(ground) is None
    excitations.refuse_other_numerics(excited, ground)
    with pytest.raises(TypeError, match='does not stand on a given Numerics'):
        potential_energy_surface(h2, rhf_df_differentiable, ground_state=GroundState('dft', 'hf'),
                                 settings=surface_numerics(excited))


def test_a_given_numerics_on_another_grid_than_the_keywords_is_refused(
        h2, excited):
    with pytest.raises(NumericsMismatch, match='isdf.counts'):
        potential_energy_surface(
            h2, rhf_df_differentiable, ground_state=GroundState('rpa', 'hf'),
            settings=surface_numerics(excited), grid_accuracy='G2')
