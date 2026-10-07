"""The vibronic analysis and the ground-state relaxation stand in the surface's
environment.

`relax_state` reports the relaxed energy on the surface's own mean field
(`surface_mean_field`), so `vibronic_analysis` takes the Franck-Condon
gradient and the vertical energy there too, not on the caller's ground-state
mean field. In a continuum the two are different mean fields (the surface's
is relaxed in the static reaction field, the factory's is the bare SCF), and
lambda_relaxation across them would be the solvation energy, not a
reorganization. `relax_ground_state(environment=)` relaxes the mean field in
the named environment.

Water/STO-3G RHF in PCM(water). Gates:

  - with the relaxed state AT the Franck-Condon geometry, lambda_relaxation is
    zero to SCF precision, and the Huang-Rhys gradient is the surface's own;
  - `relax_ground_state(environment=...)` returns a mean field in the
    continuum at a stationary point of the solvated surface, and without it
    the gas-phase relaxation.
"""
import numpy as np
import pytest
from pyscf import gto, scf

import src.gradients  # noqa: F401  cycle: src.properties imports src.gradients
from src.Base.constants import GEOM_OPT_CONV
from src.Base.solvent_screening import SolventScreening
from src.properties.optimize import MeanFieldSurface, relax_ground_state
from src.properties.surface import surface_mean_field
from src.properties.vibronic import (huang_rhys_from_gradient, normal_modes,
                                     vibronic_analysis)

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def factory(mol):
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-12, 1e-9, 0
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=WATER, basis='sto-3g', verbose=0)


@pytest.fixture(scope='module')
def solvated(water):
    return MeanFieldSurface(water, factory,
                            environment=SolventScreening(water,
                                                         solvent='water'))


def test_lambda_relaxation_is_on_the_surfaces_own_mean_field(water, solvated):
    mf_gas = factory(water)
    own = surface_mean_field(solvated, water)
    # the two mean fields this defect confused: tens of meV apart
    assert abs(own.e_tot - mf_gas.e_tot) > 1e-3
    state = {'mol': water, 'e_total': solvated.total_energy(water, own)}
    out = vibronic_analysis(solvated, state, water, mf_gas, verbose=False)
    assert out['lambda_relaxation'] == pytest.approx(0.0, abs=1e-8)
    omega, modes, masses, _ = normal_modes(mf_gas, water)
    g_own = solvated.total_gradient(water, own)[0]
    _, gk = huang_rhys_from_gradient(g_own, omega, modes, masses)
    assert np.allclose(out['gk'], gk, atol=1e-8)


def test_relax_ground_state_in_an_environment(water):
    env = SolventScreening(water, solvent='water')
    mol, mf = relax_ground_state(water, factory, engine='cartesian',
                                 environment=env)
    assert hasattr(mf, 'with_solvent')
    surface = MeanFieldSurface(mol, factory, environment=env)
    force = surface.total_gradient(mol, surface_mean_field(surface, mol))[0]
    assert np.abs(force).max() < GEOM_OPT_CONV['opt_grad_max']
    assert mf.e_tot == pytest.approx(surface_mean_field(surface, mol).e_tot,
                                     abs=1e-9)
    gas_mol, gas_mf = relax_ground_state(water, factory, engine='cartesian')
    assert not hasattr(gas_mf, 'with_solvent')
    assert abs(gas_mf.e_tot - mf.e_tot) > 1e-3
