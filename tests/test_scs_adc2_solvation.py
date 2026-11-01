"""SCS-ADC(2) in the self-consistent state-specific solvation loop
(src.environments.solvation_loop with ContinuumField): the state models of
MP2State(spin_scaling='scs') and ADC2State on it, their loop, relaxed density
and force. Water 6-31G (bent, non-symmetric), toluene IEF-PCM, frozen 1,
cc-pvdz-ri / cc-pvdz-jkfit; the singlet carries its LR term.

  1. the contract: Tr(D_eff O) is the 5-point finite-field derivative of the
     energy in the field h + lam O (h the operator of the HF reaction
     field's charges), O = mu_z and a random symmetric matrix, for SCS-MP2
     and the SCS-ADC(2) singlet and triplet, to 1e-9 (SCF to a gradient of
     1e-12, roots to 1e-12).
  2. stationarity: G(q) of the SCS-ADC(2) singlet and triplet has no first
     derivative at the converged charges (below 1e-7 Ha per unit |dq|) and
     has one at the start of the loop (above 1e-5: the check can fail).
  3. the force of G_eq (solvation_loop.force) of SCS-MP2 and the SCS-ADC(2)
     singlet and triplet along a random unit direction against the 5-point
     central difference of the converged G_eq (h = 2e-3 Bohr, every point a
     full loop on its own cavity), to 1e-8 Ha/Bohr.
  4. eps = 1 is the gas phase: G_eq, the relaxed density and the force of S0,
     S1 and T1 in a continuum at eps_s = eps_inf = 1 equal NoMedium's to a
     few ulp (`matches`), and the scaled G_eq is not ADC(2)'s (the check
     can fail).
  5. the one-shot state-specific term (solve_ee_adc(state_specific=...)) on
     a scaled root: with (c_os, c_ss) = (1, 1) and a relaxed density it is
     ADC(2)'s to a few ulp (shifts, relaxed densities, dipole changes);
     with 'scs' e = omega + the relaxed shift and the unrelaxed entries are
     None; the unrelaxed density is refused with a scaling.
  6. the driver (`python -m src.photophysics solvated --spin-scaling scs`):
     every state model of the run is scaled (MP2 and ADC(2), identity and
     resume key); --method gwbse refuses the option.

Run: python -m pytest -s tests/test_scs_adc2_solvation.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import ADC2_SPIN_SCALING
from src.environments import solvation_loop as sl
from src.environments.reaction_field import ContinuumField, NoMedium
from src.environments.solvent_screening import SolventScreening
from src.gradients.state_models import HFState, MP2State
from src.gradients.adc.ee_state_model import ADC2State
from src.photophysics import solvated
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from tests.adc_common import (ADC_TIGHT, ADC_NUMERICS, ATOMS, AUX, H2CO, JKFIT, NF, check,
                              five_point, make_scf_loop, make_scf_tight, molecule)
from tests.pins import RECORDING_BLAS, matches

SCS = 'scs'
FD_STEP = 2e-3          # Bohr, the 5-point difference of the converged G_eq
FD_BAR = 1e-8           # Ha/Bohr


def state(kind, field, mol, make_scf=make_scf_tight, scaling=SCS):
    """The SCS state model of `kind` ('S0', 'singlet', 'triplet') in field."""
    mp2 = MP2State(HFState(mol, make_scf), NF, AUX, spin_scaling=scaling,
                   force_route='dense', numerics=ADC_NUMERICS)
    if kind == 'S0':
        return mp2
    lr = field.transition_environment() if kind == 'singlet' else None
    return ADC2State(mp2, kind, 0, 3, lr)


def toluene(mol):
    return ContinuumField.from_environment(SolventScreening(mol, solvent='toluene'))


def G(kind, shift=None):
    mol = molecule(ATOMS, shift)
    field = toluene(mol)
    r = sl.equilibrate(state(kind, field, mol), field, **ADC_TIGHT)
    assert r.converged
    return r.energy


@pytest.mark.parametrize('kind', ['S0', 'singlet', 'triplet'])
def test_contract(kind):
    mol = molecule(ATOMS)
    field = toluene(mol)
    q = field.charges(field.potential(HFState(mol, make_scf_tight)(
        *field.operator(np.zeros(field.nsites))).dm_eff))
    h, e = field.operator(q)
    A = np.random.default_rng(3).standard_normal((mol.nao, mol.nao))
    ops = {'mu_z': mol.intor('int1e_r')[2], 'random': 0.05 * (A + A.T)}
    s = state(kind, field, mol)
    D = s(h, e).dm_eff
    ok = True
    for name, O in ops.items():
        fd = five_point(lambda lam: s(h + lam * O, e).energy)
        an = float(np.sum(D * O))
        ok &= check(abs(fd - an) < 1e-9, f'SCS {kind}: Tr(D_eff {name}) is dE/d field',
                    f'{an:+.10f} vs {fd:+.10f}, {fd - an:+.1e}')
    assert ok


@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_stationary(spin):
    mol = molecule(ATOMS)
    field = toluene(mol)
    s = state(spin, field, mol, make_scf_loop)
    r = sl.equilibrate(s, field, **ADC_TIGHT)
    assert check(r.converged, f'SCS {spin}: the loop converges',
                 f'{len(r.history)} cycles, G {r.energy:.10f}')
    d = np.random.default_rng(11).standard_normal(field.nsites)
    d *= 1e-3 / np.linalg.norm(d)
    slope = (sl.free_energy(s, field, r.q + d) - sl.free_energy(s, field, r.q - d)) / 2e-3
    q0 = field.charges(field.potential(s(*field.operator(np.zeros(field.nsites))).dm_eff))
    slope0 = (sl.free_energy(s, field, q0 + d) - sl.free_energy(s, field, q0 - d)) / 2e-3
    ok = check(abs(slope) < 1e-7, f'SCS {spin}: G is stationary at the converged '
               'charges', f'{slope:+.1e}')
    ok &= check(abs(slope0) > 1e-5, f'SCS {spin}: and not at the start', f'{slope0:+.1e}')
    assert ok


@pytest.mark.parametrize('kind', ['S0', 'singlet', 'triplet'])
def test_force(kind):
    mol = molecule(ATOMS)
    field = toluene(mol)
    s = state(kind, field, mol)
    r = sl.equilibrate(s, field, **ADC_TIGHT)
    assert r.converged
    F = sl.force(s, field, r)
    d = np.random.default_rng(5).standard_normal((mol.natm, 3))
    d /= np.linalg.norm(d)
    fd = (G(kind, -2 * FD_STEP * d) - 8 * G(kind, -FD_STEP * d)
          + 8 * G(kind, FD_STEP * d) - G(kind, 2 * FD_STEP * d)) / (12 * FD_STEP)
    an = float(np.sum(F * d))
    assert check(abs(fd - an) < FD_BAR, f'SCS {kind}: dG_eq along a random direction '
                 '= finite differences', f'{an:+.10f} vs {fd:+.10f}, {fd - an:+.1e} Ha/Bohr')


def test_gas_limit():
    mol = molecule(ATOMS)
    env = SolventScreening(mol, eps=1.0, eps_static=1.0, allow_static_eps=True)
    ok = True
    for kind in ('S0', 'singlet', 'triplet'):
        gas, cont = NoMedium(mol), ContinuumField.from_environment(env)
        s_gas = state(kind, gas, mol, make_scf_loop)
        s_cont = state(kind, cont, mol, make_scf_loop)
        r_gas = sl.equilibrate(s_gas, gas, **ADC_TIGHT)
        r_cont = sl.equilibrate(s_cont, cont, **ADC_TIGHT)
        F_gas, F_cont = sl.force(s_gas, gas, r_gas), sl.force(s_cont, cont, r_cont)
        ok &= check(matches(r_cont.energy, r_gas.energy, RECORDING_BLAS, quantity='energy')
                    and matches(r_cont.dm_eff, r_gas.dm_eff, RECORDING_BLAS, '1',
                                quantity='vector')
                    and matches(F_cont, F_gas, RECORDING_BLAS, quantity='gradient'),
                    f'SCS {kind}: G_eq, relaxed density and force at eps = 1 are the '
                    'gas phase', f'G {r_cont.energy!r} vs {r_gas.energy!r}, max |dF| '
                    f'{float(np.max(np.abs(F_cont - F_gas))):.1e}')
        r_plain = sl.equilibrate(state(kind, gas, mol, make_scf_loop, None), gas,
                                 **ADC_TIGHT)
        ok &= check(abs(r_plain.energy - r_gas.energy) > 1e-4,
                    f'SCS {kind}: and not ADC(2)', f'{r_gas.energy - r_plain.energy:+.2e} Ha')
    assert ok


def _toluene_scf(mol):
    env = SolventScreening(mol, solvent='toluene')

    def factory(m):
        mf = scf.RHF(m).density_fit(auxbasis=JKFIT)
        mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-13, 1e-10, 0
        return mf.run()
    return env, env.mean_field(mol, factory)


def test_one_shot_state_specific():
    mol = gto.M(atom=H2CO, basis='cc-pvdz', verbose=0)
    env, sol = _toluene_scf(mol)
    kw = dict(level='adc2', nroots=1, frozen=2, conv_tol=1e-10, fold=True,
              environment=env, return_fold=True, state_specific='equilibrium',
              ss_density='relaxed')
    ok = True
    for spin in ('singlet', 'triplet'):
        _, _, plain = solve_ee_adc(sol, spin=spin, **kw)
        _, _, unit = solve_ee_adc(sol, spin=spin, spin_scaling=(1.0, 1.0), **kw)
        ok &= check(matches(unit.ss_shifts['equilibrium'], plain.ss_shifts['equilibrium'],
                            RECORDING_BLAS, quantity='energy')
                    and matches(np.asarray(unit.delta_density_relaxed),
                                np.asarray(plain.delta_density_relaxed),
                                RECORDING_BLAS, '1', quantity='vector')
                    and np.allclose(unit.delta_dipole['relaxed'],
                                    plain.delta_dipole['relaxed'], rtol=0, atol=1e-10),
                    f'{spin}: (c_os, c_ss) = (1, 1) is ADC(2)\'s relaxed state-specific term',
                    f"{unit.ss_shifts['equilibrium'][0]:+.12f} vs "
                    f"{plain.ss_shifts['equilibrium'][0]:+.12f} Ha")
        e, _, res = solve_ee_adc(sol, spin=spin, spin_scaling=SCS, **kw)
        ok &= check(np.array_equal(e, res.omega + res.ss_shifts['equilibrium'])
                    and res.delta_density is None and res.ss_shifts_unrelaxed is None
                    and res.delta_dipole['unrelaxed'] is None
                    and np.all(np.isfinite(res.delta_dipole['relaxed'])),
                    f'{spin}: SCS e = omega + the relaxed shift, no unrelaxed entries',
                    f"shift {res.ss_shifts['equilibrium'][0] * 27211.386:+.1f} meV "
                    f"(ADC(2) {plain.ss_shifts['equilibrium'][0] * 27211.386:+.1f} meV)")
    with pytest.raises(ValueError, match='no unrelaxed'):
        solve_ee_adc(sol, spin='singlet', spin_scaling=SCS,
                     **dict(kw, ss_density='unrelaxed'))
    assert ok


def test_driver(tmp_path):
    xyz = tmp_path / 'w.xyz'
    xyz.write_text('3\n\n' + '\n'.join(f'{a} {x} {y} {z}' for a, (x, y, z) in ATOMS) + '\n')
    argv = ['--xyz', str(xyz), '--basis', '6-31g', '--solvent', 'toluene',
            '--out', str(tmp_path / 'r.json')]
    args = solvated._parse_args(argv + ['--spin-scaling', SCS])
    solvated._resolve_basis(args)
    mol = molecule(ATOMS)
    solvated._check_args(args, mol)
    make_field, make_state = solvated._factories(
        args, 'toluene', {'numerics': solvated.adc2_numerics(args)})
    field = make_field(mol)
    models = {k: make_state(k)(mol, field) for k in ('S0', 'singlet', 'triplet')}
    scaling = tuple(ADC2_SPIN_SCALING[SCS])
    assert models['S0'].scaling == scaling
    assert all(models[k].mp2.scaling == scaling for k in ('singlet', 'triplet'))
    assert models['singlet'].identity()['spin_scaling'] == list(scaling)
    rec = solvated._start_record(args)
    assert rec['spin_scaling'] == SCS and 'spin_scaling' in solvated.RESUME_KEY
    assert rec['method'].startswith('SCS-ADC(2)')
    plain = solvated._parse_args(argv)
    solvated._resolve_basis(plain)
    solvated._check_args(plain, mol)
    assert solvated._settings(plain)['spin_scaling'] is None
    gw = solvated._parse_args(argv + ['--method', 'gwbse', '--isdf-grid', 'G1',
                                      '--spin-scaling', SCS])
    solvated._resolve_basis(gw)
    with pytest.raises(SystemExit, match='takes no --spin-scaling'):
        solvated._check_args(gw, mol)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sys.exit(pytest.main(['-s', __file__]))
