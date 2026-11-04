"""The force of G_neq(X | S0), X in the slow charges of S0's own equilibrium
at every geometry (`solvation_loop.force_neq`, the non-equilibrium mode of
`src.gradients.solvated_surface.SolvatedSurface`). Water 6-31G (bent,
non-symmetric), toluene IEF-PCM, S0 Hartree-Fock, X the ADC(2) singlet (its
linear-response term at eps_inf) or triplet (frozen 1, cc-pvdz-ri /
cc-pvdz-jkfit), loops to 1e-10 e / 1e-12 Ha.

  1. `scf_loop_response`: Tr[h dD_0/dR] of a mean field (Hartree-Fock, PBE0,
     LRC-wPBEh) in its own continuum at eps_static (pyscf's PCM-SCF) and in
     the gas phase against five-point differences of the density (h = 1e-3
     Bohr), within 2e-9 (the XC grid's noise; |Tr[h dD/dR]| ~ 0.05).
  2. the singlet and the triplet: the force against five-point differences
     of G_neq along a random direction (h = 5e-4 Bohr, every point both loops
     on its own cavity), within 1e-8 Ha/Bohr; without the slow charges'
     motion (the lambda terms, the response of S0's loop among them) the
     miss is far above that, so the term is needed.
  3. X = S0: G_neq(S0 | S0) is G_eq(S0), energy and force.
  4. eps_static = eps_inf: no slow charges, and the force is that of G_eq.

Run: python -m pytest -s tests/test_solvation_force_neq.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.geometry.geometry import molecule_at
from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.gradients.adc.ee_state_model import ADC2State
from src.gradients.scf_loop_response import loop_density_response
from src.gradients.solvated_surface import SolvatedSurface
from src.gradients.state_models import HFState, MP2State
from tests.adc_common import (ADC_NUMERICS, ATOMS, AUX, NF, check, make_scf_tight,
                              molecule)

NUMERICS = ADC_NUMERICS
H = 5e-4
FORCE_BAR = 1e-8


def toluene(mol):
    return ContinuumField.from_environment(SolventScreening(mol, solvent='toluene'))


def optical_toluene(mol):
    """Toluene with eps_static set to its eps_inf: no slow charges."""
    env = SolventScreening(mol, solvent='toluene')
    return ContinuumField(mol, env.eps_optical, env.eps_optical, env.method,
                          {k: env.cavity_settings[k] for k in
                           ('lebedev_order', 'vdw_scale', 'r_probe', 'radii_table')},
                          transition=toluene(mol).transition_environment())


def ground(mol, field):
    return HFState(mol, make_scf_tight)


def excited(kind):
    def build(mol, field):
        if kind == 'S0':
            return HFState(mol, make_scf_tight)
        mp2 = MP2State(HFState(mol, make_scf_tight), NF, AUX, numerics=NUMERICS)
        lr = field.transition_environment() if kind == 'singlet' else None
        return ADC2State(mp2, kind, 0, 3, lr)
    return build


def surface(kind, make_field=toluene, noneq=True):
    return SolvatedSurface(molecule(ATOMS), make_field, excited(kind),
                           make_ground=ground, noneq=noneq, numerics=NUMERICS)


def direction(shape, seed=7):
    u = np.random.default_rng(seed).standard_normal(shape)
    return u / np.linalg.norm(u)


@pytest.mark.parametrize('xc', ['hf', 'pbe0', 'lrc-wpbeh'])
@pytest.mark.parametrize('solvated', [False, True])
def test_the_loop_density_response_is_the_density_derivative(xc, solvated):
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    rng = np.random.default_rng(5)
    h = rng.standard_normal((mol.nao, mol.nao))
    h = 0.05 * (h + h.T)
    u = direction((mol.natm, 3), seed=6)

    def solved(m):
        mf = (scf.RHF(m) if xc == 'hf' else dft.RKS(m, xc=xc)).density_fit()
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
        if solvated:
            mf = mf.PCM()
            mf.with_solvent = toluene(m).scf_continuum()
        mf.kernel()
        return mf
    mf = solved(mol)
    continuum = toluene(mol).scf_continuum() if solvated else None
    g = loop_density_response(mf, mf.get_hcore(), continuum, h, NUMERICS)
    step = 1e-3
    density = [np.sum(h * solved(molecule_at(mol, mol.atom_coords() + t * u))
                      .make_rdm1()) for t in (2 * step, step, -step, -2 * step)]
    fd = (-density[0] + 8 * density[1] - 8 * density[2] + density[3]) / (12 * step)
    d = abs(float((g * u).sum()) - fd)
    assert check(d < 2e-9, f'Tr[h dD/dR] ({xc}, {"PCM" if solvated else "gas"})',
                 f'|analytic - fd| {d:.1e}')


def g_neq(kind, step, u, make_field=toluene):
    mol = molecule(ATOMS)
    s = surface(kind, make_field)
    return s.total_energy(molecule_at(mol, mol.atom_coords() + step * u))


def lambda_terms(s):
    """The terms of force_neq that move the slow charges (module docstring
    of solvation_loop): what a force with q_s held would leave out."""
    field = s.last['field']
    g = s.last_ground
    d0 = np.asarray(g['result'].dm_eff, float)
    dx = np.asarray(s.last['result'].dm_eff, float)
    lam = field.electronic_potential(dx - d0)
    mu = field.charges(lam, 'static') - field.charges(lam, 'fast')
    v0 = field.potential(d0)
    return (field.response_force(lam, v0, 'static')
            - field.response_force(lam, v0, 'fast')
            + field.charge_force(d0, mu)
            + g['state'].loop_response(field.operator(mu)[0], field, NUMERICS))


@pytest.mark.parametrize('kind', ['singlet', 'triplet'])
def test_the_force_is_the_derivative_of_g_neq(kind):
    mol = molecule(ATOMS)
    s = surface(kind)
    F = s.total_gradient(mol)[0]
    assert s.last['result'].converged and s.last_ground['result'].converged
    held = F - lambda_terms(s)
    u = direction(F.shape)
    fd = (-g_neq(kind, 2 * H, u) + 8 * g_neq(kind, H, u)
          - 8 * g_neq(kind, -H, u) + g_neq(kind, -2 * H, u)) / (12 * H)
    d = abs(float((F * u).sum()) - fd)
    miss = abs(float((held * u).sum()) - fd)
    print(f'{kind}: dG_neq/dx.u {float((F * u).sum()):.12f}, differences '
          f'{fd:.12f}, |diff| {d:.2e}; with the slow charges held {miss:.2e}')
    assert miss > 100 * FORCE_BAR
    assert check(d < FORCE_BAR, f'{kind}: the force of G_neq(X | S0)',
                 f'|analytic - fd| {d:.1e} Ha/Bohr')


def test_the_ground_state_in_its_own_slow_charges_is_its_equilibrium():
    mol = molecule(ATOMS)
    neq, eq = surface('S0'), surface('S0', noneq=False)
    F_neq, G_neq = neq.total_gradient(mol)[:2]
    F_eq, G_eq = eq.total_gradient(mol)[:2]
    assert abs(G_neq - G_eq) < 1e-10
    assert float(np.max(np.abs(F_neq - F_eq))) < 1e-9


def test_without_slow_charges_the_force_is_that_of_g_eq():
    mol = molecule(ATOMS)
    neq = surface('singlet', optical_toluene)
    eq = surface('singlet', optical_toluene, noneq=False)
    F_neq, G_neq = neq.total_gradient(mol)[:2]
    F_eq, G_eq = eq.total_gradient(mol)[:2]
    assert abs(G_neq - G_eq) < 1e-9
    assert float(np.max(np.abs(F_neq - F_eq))) < 1e-8
