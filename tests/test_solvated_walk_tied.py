"""The solvated driver's walks tied to their gradient (`python -m
src.photophysics solvated --loop-tolerance gradient-tied`, SolvatedSurface's
tie). Water 6-31G (bent, non-symmetric), toluene IEF-PCM, frozen 1,
cc-pvdz-ri / cc-pvdz-jkfit; the ADC(2) route's ground state (MP2).

  1. the option: 'fixed' (the default) leaves SolventNumerics.loop_force_fraction
     unset, 'gradient-tied' sets it to LOOP_FORCE_FRACTION, for ADC(2) and
     GW-BSE alike; it is a setting of the record and its resume key (a record
     of the other choice is refused, one made before the option existed
     resumes as 'fixed' only).
  2. a tied S0 walk through the driver (`_minimum`): its record's loop log
     starts with the calibration loop, holds tied loops and ends on a full
     one (the converged gradient is a fully converged loop's); the saddle
     check's Hessian is taken on the untied twin, so it adds nothing to the
     log; and its minimum's G is the fixed walk's to 1e-7 Ha.

Run: python -m pytest -s tests/test_solvated_walk_tied.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.Base.numerics.constants import LOOP_FORCE_FRACTION
from src.photophysics import solvated
from tests.adc_common import ATOMS, check, molecule


def _xyz(tmp_path):
    xyz = tmp_path / 'w.xyz'
    xyz.write_text('3\n\n' + '\n'.join(f'{a} {x} {y} {z}' for a, (x, y, z) in ATOMS) + '\n')
    return str(xyz)


def _args(tmp_path, *extra, out='r.json'):
    argv = ['--xyz', _xyz(tmp_path), '--basis', '6-31g', '--solvent', 'toluene',
            '--frozen', '1', '--out', str(tmp_path / out), *extra]
    args = solvated._parse_args(argv)
    solvated._resolve_basis(args)
    solvated._check_args(args, molecule(ATOMS))
    return args


def test_the_option_sets_the_tie_and_is_a_resume_setting(tmp_path):
    fixed, tied = _args(tmp_path), _args(tmp_path, '--loop-tolerance', 'gradient-tied')
    assert fixed.loop_tolerance == 'fixed'
    assert solvated.adc2_numerics(fixed).solvent.loop_force_fraction is None
    assert solvated.adc2_numerics(tied).solvent.loop_force_fraction == LOOP_FORCE_FRACTION
    gw = solvated._parse_args(['--xyz', _xyz(tmp_path), '--basis', 'cc-pvdz',
                               '--method', 'gwbse', '--isdf-grid', 'G1', '--out',
                               str(tmp_path / 'gw.json'), '--loop-tolerance',
                               'gradient-tied'])
    solvated._resolve_basis(gw)
    solvated._check_args(gw, molecule(ATOMS, basis='cc-pvdz'))
    assert (solvated.gwbse_numerics(gw, molecule(ATOMS, basis='cc-pvdz')).solvent.loop_force_fraction
            == LOOP_FORCE_FRACTION)
    assert 'loop_tolerance' in solvated.RESUME_KEY
    solvated._save(solvated._start_record(fixed), fixed.out)
    with pytest.raises(SystemExit, match='other settings'):
        solvated._start_record(tied)
    older = solvated._start_record(fixed)
    older.pop('loop_tolerance')
    solvated._save(older, fixed.out)
    assert solvated._start_record(fixed)['loop_tolerance'] == 'fixed'
    with pytest.raises(SystemExit, match='other settings'):
        solvated._start_record(tied)


def _walk(args, monkeypatch=None, seen=None):
    """(the S0 minimum's record, the surface) of the driver's S0 walk."""
    loop_kw = dict(numerics=solvated.adc2_numerics(args))
    make_field, make_state = solvated._factories(args, 'toluene', loop_kw)
    if monkeypatch is not None:
        plain = solvated.saddle_check

        def watched(surface, m, walk, **kw):
            seen.append((surface._untied, len(surface.loop_log)))
            out = plain(surface, m, walk, **kw)
            seen.append((surface._untied, len(surface.loop_log)))
            return out
        monkeypatch.setattr(solvated, 'saddle_check', watched)
    mol = molecule(ATOMS)
    surf = solvated.SolvatedSurface(mol, make_field, make_state('S0'),
                                    label='S0 toluene',
                                    final_grad_max=solvated.WALK_GRAD_MAX, **loop_kw)
    return solvated._minimum(args, surf, mol, 0, 'S0')[1], surf


def test_a_tied_walk_logs_its_loops_and_checks_its_minimum_untied(tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        seen = []
        tied, surf = _walk(_args(tmp_path, '--loop-tolerance', 'gradient-tied'),
                           monkeypatch, seen)
        fixed, _ = _walk(_args(tmp_path, out='f.json'))
    modes = [e['mode'] for e in tied['loop_log']]
    print(f'tied walk: {tied["cycles"]} steps, loops {modes}')
    assert tied['converged'] and fixed['converged']
    assert modes[0] == 'calibration' and 'tied' in modes and modes[-1] == 'full'
    assert fixed['loop_log'] == []
    # the Hessian's forces on the untied twin: every loop full, none logged
    assert seen[0][0] is True and seen[0][1] == seen[1][1]
    assert tied['saddle_check']['status'] == fixed['saddle_check']['status']
    d = abs(tied['G'] - fixed['G'])
    assert check(d < 1e-7, "the tied walk's minimum has the fixed walk's G",
                 f'{d:.1e} Ha')


if __name__ == '__main__':
    sys.exit(pytest.main(['-s', __file__]))
