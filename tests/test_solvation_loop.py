"""Self-consistent state-specific solvation (src.environments.solvation_loop) with the
continuum adapter (src.environments.reaction_field.ContinuumField) and the state models
(src.gradients.state_models, eeADC.ee_state_model). Water 6-31G, toluene
IEF-PCM unless stated; frozen 1, aux cc-pvdz-ri, JK cc-pvdz-jkfit.

  1. the medium adapter: Hartree-Fock equilibrated by the generic loop is
     pyscf's equilibrium PCM-SCF (SolventScreening.mean_field), to 1e-10 Ha.
  2. the state models' contract: Tr(D_eff O) is the 5-point finite-field
     derivative of the energy in the field h + lam O (h the operator of fixed
     charges), O = mu_z and a random symmetric matrix, for MP2 and for the
     ADC(2) singlet (with its LR term) and triplet, to 1e-9 (SCF to a
     gradient of 1e-12, roots to 1e-12).
  3. stationarity: G(q) has no first derivative at the converged charges of an
     ADC(2) singlet and triplet (below 1e-7 Ha per unit |dq|) and has one at
     the start of the loop (above 1e-5: the check can fail).
  4. first-order limit: at eps_s 1.1 the equilibrium excitation energy is
     omega at the ground state's PTE-HF charges plus (1/2) dV Q dV +
     dV Q V(D_MP2) to within 10%, the remainder shrinking 1.5x or more from
     eps_s 1.2.
  5. a state in its own slow charges: G_neq(S0 | S0) = G_eq(S0) and
     G_neq(X | X) = G_eq(X) to 1e-9.
  6. eps_s = eps_inf: no slow charges, G_neq(X | S0) = G_eq(X) to 1e-9.
  7. G_neq is stationary in the fast charges at convergence (< 1e-7) and not
     at the start (> 1e-5).
  8. weak continuum (eps_s 1.1): absorption = omega(q_0) + (1/2) dV Q_f dV and
     absorption - emission = -dV (Q_s - Q_f) dV > 0, each to within 10%.

One pytest test per check (per model or spin where a check covers several);
every loop is asserted converged. The refusals are test_solvation_refusals.py.

Run: python -m pytest -s tests/test_solvation_loop.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

from src.environments import solvation_loop as sl
from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.gradients.state_models import HFState, MP2State
from src.gradients.adc.ee_state_model import ADC2State
from tests.adc_common import (AUX, ROOT_CONV_TOL, SCF_TOL_TIGHT,
                              SOLVATION_TIGHT, WATER_C2V, check, df_rhf,
                              five_point)


def make_scf(mol, grad=1e-9):
    return df_rhf(mol, SCF_TOL_TIGHT, grad)


def setup(eps_static=None, eps_inf=None):
    mol = gto.M(atom=WATER_C2V, basis='6-31g', verbose=0)
    if eps_static is None:
        env = SolventScreening(mol, solvent='toluene')
    else:
        env = SolventScreening(mol, eps=eps_inf or 1.0 + 0.5 * (eps_static - 1.0),
                               eps_static=eps_static, allow_static_eps=True)
    return mol, env, ContinuumField.from_environment(env)


def models(mol, field, tight=False):
    """Factories of fresh HF, MP2 and ADC(2) state models; tight converges
    the SCF to a gradient of 1e-12 and the roots to 1e-12 (finite fields:
    omega is not variational in the orbitals, so their error enters it
    linearly)."""
    hf = lambda: HFState(mol, (lambda m: make_scf(m, 1e-12)) if tight else make_scf)
    mp2 = lambda: MP2State(hf(), 1, AUX)
    adc = lambda spin: ADC2State(mp2(), spin, 0, 3,
                                 field.transition_environment()
                                 if spin == 'singlet' else None,
                                 conv_tol=ROOT_CONV_TOL if tight else 1e-8)
    return hf, mp2, adc


def test_hf_is_pyscf_pcm_scf():
    mol, env, field = setup()
    e_pcm = env.mean_field(mol, lambda m: make_scf(m).run()).e_tot
    r = sl.equilibrate(HFState(mol, make_scf), field, **SOLVATION_TIGHT)
    assert r.converged
    assert check(abs(r.energy - e_pcm) < 1e-10,
                 'generic loop + ContinuumField + HFState = pyscf PCM-SCF',
                 f'{r.energy - e_pcm:+.1e}, {len(r.history)} cycles')


@pytest.mark.parametrize('label', ['MP2', 'ADC(2) singlet', 'ADC(2) triplet'])
def test_contract(label):
    mol, env, field = setup()
    _, mp2, adc = models(mol, field, tight=True)
    q = field.charges(field.potential(HFState(mol, make_scf)(
        *field.operator(np.zeros(field.nsites))).dm_eff))
    h, e = field.operator(q)
    rng = np.random.default_rng(3)
    A = rng.standard_normal((mol.nao, mol.nao))
    ops = {'mu_z': mol.intor('int1e_r')[2], 'random': 0.05 * (A + A.T)}
    make = {'MP2': mp2, 'ADC(2) singlet': lambda: adc('singlet'),
            'ADC(2) triplet': lambda: adc('triplet')}[label]
    state = make()
    D = state(h, e).dm_eff
    ok = True
    for name, O in ops.items():
        fd = five_point(lambda lam: state(h + lam * O, e).energy)
        an = float(np.sum(D * O))
        ok &= check(abs(fd - an) < 1e-9, f'{label}: Tr(D_eff {name}) is '
                    'dE/d field', f'{an:+.10f} vs {fd:+.10f}, {fd - an:+.1e}')
    assert ok


@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_stationary(spin):
    mol, env, field = setup()
    _, _, adc = models(mol, field)
    rng = np.random.default_rng(11)
    draws = [rng.standard_normal(field.nsites) for _ in ('singlet', 'triplet')]
    state = adc(spin)
    r = sl.equilibrate(state, field, **SOLVATION_TIGHT)
    assert check(r.converged, f'{spin}: the loop converges',
                 f'{len(r.history)} cycles, G {r.energy:.10f}')
    d = draws[('singlet', 'triplet').index(spin)]
    d *= 1e-3 / np.linalg.norm(d)
    slope = (sl.free_energy(state, field, r.q + d)
             - sl.free_energy(state, field, r.q - d)) / 2e-3
    q0 = field.charges(field.potential(state(
        *field.operator(np.zeros(field.nsites))).dm_eff))
    slope0 = (sl.free_energy(state, field, q0 + d)
              - sl.free_energy(state, field, q0 - d)) / 2e-3
    ok = check(abs(slope) < 1e-7, f'{spin}: G is stationary at the '
               'converged charges', f'{slope:+.1e}')
    ok &= check(abs(slope0) > 1e-5, f'{spin}: and not at the start',
                f'{slope0:+.1e}')
    assert ok


def _first_order(eps_static):
    mol, env, field = setup(eps_static)
    hf, mp2, adc = models(mol, field)
    r_hf = sl.equilibrate(hf(), field, **SOLVATION_TIGHT)
    assert r_hf.converged
    q_hf = r_hf.q
    h, e = field.operator(q_hf)
    m = mp2()
    r0 = m(h, e)
    x = adc('singlet')
    rx = x(h, e)
    D2 = r0.dm_eff - m.hf(h, e).dm_eff
    dV = field.electronic_potential(rx.dm_eff - r0.dm_eff)
    V2 = field.electronic_potential(D2)
    Q = field.response_matrix('static')
    first = 0.5 * dV @ Q @ dV + dV @ Q @ V2
    g0 = sl.equilibrate(mp2(), field, **SOLVATION_TIGHT)
    gx = sl.equilibrate(adc('singlet'), field, **SOLVATION_TIGHT)
    assert g0.converged and gx.converged
    return (gx.energy - g0.energy) - rx.info['omega'], first


def test_first_order():
    ea, fa = _first_order(1.2)
    eb, fb = _first_order(1.1)
    ok = check(abs(eb - fb) < 0.1 * abs(fb), 'weak continuum: equilibrium = '
               'first order to within 10%', f'{eb:+.3e} vs {fb:+.3e} Ha')
    ok &= check(abs(ea - fa) > 1.5 * abs(eb - fb), 'and the remainder shrinks',
                f'{abs(ea - fa):.2e} at 1.2, {abs(eb - fb):.2e} at 1.1')
    assert ok


@pytest.mark.parametrize('label', ['S0', 'singlet'])
def test_own_slow(label):
    mol, env, field = setup()
    _, mp2, adc = models(mol, field)
    make = mp2 if label == 'S0' else (lambda: adc('singlet'))
    eq = sl.equilibrate(make(), field, **SOLVATION_TIGHT)
    neq = sl.equilibrate(make(), field, slow=sl.slow_charges(field, eq.dm_eff),
                         **SOLVATION_TIGHT)
    assert eq.converged and neq.converged
    assert check(abs(neq.energy - eq.energy) < 1e-9,
                 f'{label} in its own slow charges: G_neq = G_eq',
                 f'{neq.energy - eq.energy:+.1e}')


def test_no_slow():
    mol, env, field = setup(2.0, 2.0)
    _, mp2, adc = models(mol, field)
    r = sl.vertical_energies(mp2, lambda: adc('singlet'), field, **SOLVATION_TIGHT)
    d = r['G']['neq_state'] - r['G']['eq_state']
    assert r['converged']
    assert check(abs(d) < 1e-9,
                 'eps_s = eps_inf: G_neq(X | S0) = G_eq(X)', f'{d:+.1e}')


def test_fast_stationary():
    mol, env, field = setup()
    _, mp2, adc = models(mol, field)
    g0 = sl.equilibrate(mp2(), field, **SOLVATION_TIGHT)
    slow = sl.slow_charges(field, g0.dm_eff)
    state = adc('singlet')
    r = sl.equilibrate(state, field, slow=slow, **SOLVATION_TIGHT)
    assert g0.converged and r.converged
    d = np.random.default_rng(5).standard_normal(field.nsites)
    d *= 1e-3 / np.linalg.norm(d)
    q_start = slow[0] + field.charges(field.potential(state(
        *field.operator(np.zeros(field.nsites))).dm_eff), 'fast')
    G = lambda q: sl.free_energy(state, field, q, slow)
    slope = (G(r.q + d) - G(r.q - d)) / 2e-3
    slope0 = (G(q_start + d) - G(q_start - d)) / 2e-3
    ok = check(abs(slope) < 1e-7, 'G_neq is stationary in the fast charges',
               f'{slope:+.1e}')
    ok &= check(abs(slope0) > 1e-5, 'and not at the start', f'{slope0:+.1e}')
    assert ok


def test_vertical_first_order():
    mol, env, field = setup(1.1)
    _, mp2, adc = models(mol, field)
    r = sl.vertical_energies(mp2, lambda: adc('singlet'), field, **SOLVATION_TIGHT)
    assert r['converged']
    g0 = r['loops']['eq_ground']
    h, e = field.operator(g0.q)
    rx = adc('singlet')(h, e)
    dV = field.electronic_potential(rx.dm_eff - g0.dm_eff)
    Qs, Qf = field.response_matrix('static'), field.response_matrix('fast')
    abs_x, abs_1 = r['absorption'] - rx.info['omega'], 0.5 * dV @ Qf @ dV
    st_x, st_1 = r['absorption'] - r['emission'], -dV @ (Qs - Qf) @ dV
    ok = check(abs(abs_x - abs_1) < 0.1 * abs(abs_1),
               'weak continuum: absorption = omega(q_0) + (1/2) dV Q_f dV',
               f'{abs_x:+.3e} vs {abs_1:+.3e} Ha')
    ok &= check(st_x > 0 and abs(st_x - st_1) < 0.1 * abs(st_1),
                'and absorption - emission = -dV (Q_s - Q_f) dV > 0',
                f'{st_x:+.3e} vs {st_1:+.3e} Ha')
    assert ok
