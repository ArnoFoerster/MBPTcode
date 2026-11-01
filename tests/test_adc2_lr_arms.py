"""The two ADC(2) solvent arms of the state-specific loop (`python -m
src.photophysics solvated --solvent-arm`): 'ss', the loop alone (the
singles block on the bare interaction), and 'ss+lr', the loop with the
singlet's LR term at eps_inf. Water 6-31G (bent, non-symmetric), toluene IEF-PCM,
frozen 1, cc-pvdz-ri / cc-pvdz-jkfit; ADC(2) and SCS-ADC(2).

  1. the force of G_eq of the 'ss' singlet (no LR term), ADC(2) and
     SCS-ADC(2), along a random unit direction against the 5-point central
     difference of the converged G_eq (h = 2e-3 Bohr, every point a full
     loop on its own cavity), to 1e-8 Ha/Bohr; and the same for the 'ss+lr'
     ADC(2) singlet (the arm the campaign compares it with).
  2. the 'ss' singlet's G(q) is stationary at the converged charges (below
     1e-7 Ha per unit |dq|) and not at the start (above 1e-5).
  3. the arms differ where they should: the singlet's G_eq and absorption
     move by more than 1e-5 Ha between the arms (the LR term; the check can
     fail), the triplet's are the same bits.
  4. the driver: 'ss' is the default, a singlet of the 'ss' arm carries no
     LR environment and one of 'ss+lr' does, a triplet none in either; the arm
     is a setting of the record and its resume key (a record of the other arm
     is refused; one made before the option existed resumes as 'ss+lr' only;
     a gas-phase record has no arm), and --method gwbse refuses the option;
     the run's one Numerics reaches every state model (a non-default
     multiplier tolerance, the Z-vector's, and ADC solve tolerance) and the
     record carries it.

Run: python -m pytest -s tests/test_adc2_lr_arms.py
"""
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.environments import solvation_loop as sl
from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.gradients.state_models import HFState, MP2State
from src.gradients.adc.ee_state_model import ADC2State
from src.photophysics import solvated
from tests.adc_common import (ADC_TIGHT, ADC_NUMERICS, ATOMS, AUX, NF, check, make_scf_loop,
                              make_scf_tight, molecule)

FD_STEP = 2e-3          # Bohr, the 5-point difference of the converged G_eq
FD_BAR = 1e-8           # Ha/Bohr


def field_of(mol):
    return ContinuumField.from_environment(SolventScreening(mol, solvent='toluene'))


def state(kind, field, mol, arm, scaling=None, make_scf=make_scf_tight):
    mp2 = MP2State(HFState(mol, make_scf), NF, AUX, spin_scaling=scaling,
                   force_route='dense', numerics=ADC_NUMERICS)
    if kind == 'S0':
        return mp2
    lr = field.transition_environment() if kind == 'singlet' and arm == 'ss+lr' else None
    return ADC2State(mp2, kind, 0, 3, lr)


def G(arm, scaling, shift=None):
    mol = molecule(ATOMS, shift)
    f = field_of(mol)
    r = sl.equilibrate(state('singlet', f, mol, arm, scaling), f, **ADC_TIGHT)
    assert r.converged
    return r.energy


@pytest.mark.parametrize('arm,scaling', [('ss', None), ('ss', 'scs'), ('ss+lr', None)])
def test_force(arm, scaling):
    mol = molecule(ATOMS)
    f = field_of(mol)
    s = state('singlet', f, mol, arm, scaling)
    r = sl.equilibrate(s, f, **ADC_TIGHT)
    assert r.converged
    F = sl.force(s, f, r)
    d = np.random.default_rng(23).standard_normal((mol.natm, 3))
    d /= np.linalg.norm(d)
    fd = (G(arm, scaling, -2 * FD_STEP * d) - 8 * G(arm, scaling, -FD_STEP * d)
          + 8 * G(arm, scaling, FD_STEP * d) - G(arm, scaling, 2 * FD_STEP * d)) / (12 * FD_STEP)
    an = float(np.sum(F * d))
    label = f"{scaling.upper() + '-' if scaling else ''}ADC(2) singlet, arm {arm}"
    assert check(abs(fd - an) < FD_BAR, f'{label}: dG_eq along a random direction = '
                 'finite differences', f'{an:+.10f} vs {fd:+.10f}, {fd - an:+.1e} Ha/Bohr')


def test_stationary():
    mol = molecule(ATOMS)
    f = field_of(mol)
    s = state('singlet', f, mol, 'ss', make_scf=make_scf_loop)
    r = sl.equilibrate(s, f, **ADC_TIGHT)
    assert r.converged
    d = np.random.default_rng(11).standard_normal(f.nsites)
    d *= 1e-3 / np.linalg.norm(d)
    slope = (sl.free_energy(s, f, r.q + d) - sl.free_energy(s, f, r.q - d)) / 2e-3
    q0 = f.charges(f.potential(s(*f.operator(np.zeros(f.nsites))).dm_eff))
    slope0 = (sl.free_energy(s, f, q0 + d) - sl.free_energy(s, f, q0 - d)) / 2e-3
    ok = check(abs(slope) < 1e-7, "arm ss singlet: G is stationary at the converged "
               'charges', f'{slope:+.1e}')
    ok &= check(abs(slope0) > 1e-5, 'arm ss singlet: and not at the start', f'{slope0:+.1e}')
    assert ok


def test_arms_differ_where_they_should():
    mol = molecule(ATOMS)
    ok = True
    out = {}
    for arm in solvated.SOLVENT_ARMS:
        f = field_of(mol)
        out[arm] = {kind: sl.vertical_energies(
            lambda: state('S0', f, mol, arm, make_scf=make_scf_loop),
            lambda: state(kind, f, mol, arm, make_scf=make_scf_loop), f, **ADC_TIGHT)
            for kind in ('singlet', 'triplet')}
    a, b = out['ss+lr']['singlet'], out['ss']['singlet']
    ok &= check(abs(a['G']['eq_state'] - b['G']['eq_state']) > 1e-5
                and abs(a['absorption'] - b['absorption']) > 1e-5,
                'singlet: G_eq and absorption move with the LR term',
                f"G_eq {a['G']['eq_state'] - b['G']['eq_state']:+.2e}, absorption "
                f"{a['absorption'] - b['absorption']:+.2e} Ha (ss+lr - ss)")
    a, b = out['ss+lr']['triplet'], out['ss']['triplet']
    ok &= check(a['G']['eq_state'] == b['G']['eq_state'] and a['absorption'] == b['absorption'],
                'triplet: the same bits in both arms (no LR term)')
    assert ok


def test_driver(tmp_path, monkeypatch):
    xyz = tmp_path / 'w.xyz'
    xyz.write_text('3\n\n' + '\n'.join(f'{a} {x} {y} {z}' for a, (x, y, z) in ATOMS) + '\n')
    out = str(tmp_path / 'r.json')
    argv = ['--xyz', str(xyz), '--basis', '6-31g', '--solvent', 'toluene', '--out', out]
    mol = molecule(ATOMS)
    lr = {}
    for extra in ([], ['--solvent-arm', 'ss+lr']):
        args = solvated._parse_args(argv + extra)
        solvated._resolve_basis(args)
        solvated._check_args(args, mol)
        make_field, make_state = solvated._factories(
            args, 'toluene', {'numerics': solvated.adc2_numerics(args)})
        f = make_field(mol)
        lr[args.solvent_arm] = {k: make_state(k)(mol, f).lr_environment is not None
                                for k in ('singlet', 'triplet')}
        if args.solvent_arm == 'ss+lr':
            # the record of the default arm is on disk
            with pytest.raises(SystemExit, match='other settings'):
                solvated._start_record(args)
            args.out = str(tmp_path / 'lr.json')
        rec = solvated._start_record(args)
        assert rec['solvent_arm'] == args.solvent_arm
        assert ('singlets with the LR term' if args.solvent_arm == 'ss+lr'
                else 'no LR term') in rec['method']
        solvated._save(rec, args.out)
    assert lr == {'ss+lr': {'singlet': True, 'triplet': False},
                  'ss': {'singlet': False, 'triplet': False}}
    assert 'solvent_arm' in solvated.RESUME_KEY
    # a record made before the option existed ran 'ss+lr'
    older = solvated._start_record(args)
    older.pop('solvent_arm')
    solvated._save(older, args.out)
    assert solvated._start_record(args)['solvent_arm'] == 'ss+lr'
    default = solvated._parse_args(argv[:-1] + [args.out])
    solvated._resolve_basis(default)
    solvated._check_args(default, mol)
    with pytest.raises(SystemExit, match='other settings'):
        solvated._start_record(default)
    # the gas phase has no arm
    gas = solvated._parse_args(['--xyz', str(xyz), '--basis', '6-31g',
                                '--out', str(tmp_path / 'gas.json')])
    solvated._resolve_basis(gas)
    solvated._check_args(gas, mol)
    assert solvated._start_record(gas)['solvent_arm'] is None
    gw = solvated._parse_args(argv + ['--method', 'gwbse', '--isdf-grid', 'G1',
                                      '--solvent-arm', 'ss'])
    solvated._resolve_basis(gw)
    with pytest.raises(SystemExit, match='takes no --solvent-arm'):
        solvated._check_args(gw, mol)
    # the run's one Numerics reaches the state models and the record
    plain = solvated.adc2_numerics
    given = dict(scf={'multiplier_tol': 3e-9}, adc={'conv_tol': 2e-9})
    monkeypatch.setattr(solvated, 'adc2_numerics',
                        lambda a: plain(a).derive('a non-default tolerance', **given))
    args = solvated._parse_args(argv[:-1] + [str(tmp_path / 'num.json')])
    solvated._resolve_basis(args)
    solvated._check_args(args, mol)
    numerics = solvated.adc2_numerics(args)
    _, make_state = solvated._factories(args, 'toluene', {'numerics': numerics})
    for kind in ('S0', 'singlet', 'triplet'):
        s = make_state(kind)(mol, field_of(mol))
        mp2 = s if kind == 'S0' else s.mp2
        assert mp2.numerics is numerics and mp2.numerics.scf.multiplier_tol == 3e-9
        if kind != 'S0':
            assert s.conv_tol == 2e-9
    monkeypatch.setattr(solvated, '_run_medium',
                        lambda a, rec, *rest: solvated._save(rec, a.out))
    monkeypatch.setattr(solvated, '_print_summary', lambda *a: None)
    solvated.main(argv[:-1] + [str(tmp_path / 'num.json'), '--media', 'solvent'])
    with open(tmp_path / 'num.json') as fh:
        saved = json.load(fh)['numerics']
    assert saved['scf']['multiplier_tol'] == 3e-9 and saved['adc']['conv_tol'] == 2e-9


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sys.exit(pytest.main(['-s', __file__]))
