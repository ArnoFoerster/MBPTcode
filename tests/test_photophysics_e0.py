"""The photophysics record's E_0 is the evaluation's own.

`StateManifold.evaluate` builds every state energy as E_0 + Omega with E_0
the declared ground state's: the mean field's on an excited chain alone,
E_HF + E_c^dRPA under `GroundState('rpa', ...)`. The record stored
`mf.e_tot` as 'e0', so under rpa the adiabatic energies and the gaps to S0
the rate table reads mixed two ground-state functionals (by E_c^dRPA, 45 mHa
here). The evaluation now carries its E_0 (`StateEvaluation.e0`) and the
record stores it.

H2/cc-pVDZ Hartree-Fock on the G1 grid, the space-time ISDF route, every
orbital explicit: the dRPA ground state, and the mean field's ('dft', the
photophysics default), whose E_0 stays mf.e_tot bit for bit.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.declaration import Excitation, GroundState, QPStates
from src.gradients.state_manifold import StateManifold
from src.photophysics import stages
from src.properties.excitations import SurfaceSpec, surface_of
from tests.reference_scf import rhf_df_differentiable

S0, T0 = ('singlet', 0), ('triplet', 0)


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)


def evaluation(h2, ground):
    spec = SurfaceSpec(GroundState(ground, 'hf'), chi0='space-time',
                       factorization='isdf', qp_states=QPStates('all'),
                       numerics={'grid_accuracy': 'G1'})
    excited = surface_of(spec, Excitation('singlet'), h2,
                         rhf_df_differentiable)
    return StateManifold(excited, states=(S0, T0)).evaluate(h2)


@pytest.fixture(scope='module')
def rpa(h2):
    return evaluation(h2, 'rpa')


def test_an_rpa_evaluation_carries_its_own_e0(rpa):
    for key in (S0, T0):
        assert rpa.e0 + rpa.omega[key] == rpa.energy[key]
    # E_c^dRPA apart from the mean field's energy
    assert rpa.e0 < rpa.mf.e_tot - 0.01


def test_a_mean_field_ground_state_is_the_scf_energy(h2):
    # 'dft' on a Hartree-Fock reference: E_0 is the mean field's, bit for bit
    ev = evaluation(h2, 'dft')
    assert ev.e0 == ev.mf.e_tot
    for key in (S0, T0):
        assert ev.e0 + ev.omega[key] == ev.energy[key]


def test_the_record_stores_the_evaluation_s_e0(rpa, monkeypatch):
    """`geometry_properties` with its property routines stubbed: 'e0' is the
    evaluation's, and e0 + omega is each state's energy in the record."""
    n = 1
    monkeypatch.setattr(stages, 'oscillator_strengths',
                        lambda *a: (np.zeros(n), np.zeros((n, 3))))
    monkeypatch.setattr(stages, 'spin_orbit_couplings',
                        lambda *a, **k: {'socme': np.zeros((n + 1, n)),
                                         'h_mo': None})
    monkeypatch.setattr(stages, 'analytic_ground_coupling',
                        lambda *a, **k: (np.zeros((2, 3)), {}))
    monkeypatch.setattr(stages, 'analytic_coupling',
                        lambda *a, **k: (np.zeros((2, 3)), {}))
    monkeypatch.setattr(stages, 'spin_vibronic_coupling',
                        lambda *a, **k: {'v0': 0.0, 'v0_vector': None,
                                         'dv_cart': None, 'paths': {}})
    props = stages.geometry_properties(rpa, S0, T0, n)
    assert props['e0'] == rpa.e0 != rpa.mf.e_tot
    for name in props['energy']:
        assert props['e0'] + props['omega'][name] == props['energy'][name]
