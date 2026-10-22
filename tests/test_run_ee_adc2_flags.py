"""tools/ee_adc/run_ee_adc2.py carries on past an SCF that did not converge,
warned and flagged (refuse inputs, warn numerics):
the record's scf.converged is False, each channel's
flags and dEST_flags name 'scf_not_converged'. Water STO-3G, an SCF asked
for an orbital gradient it cannot reach in its 200 cycles.
"""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def tool():
    spec = importlib.util.spec_from_file_location(
        'run_ee_adc2', os.path.join(ROOT, 'tools', 'ee_adc', 'run_ee_adc2.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_an_unconverged_scf_is_flagged_and_the_run_carries_on(tmp_path, monkeypatch):
    xyz = tmp_path / 'water.xyz'
    xyz.write_text('3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n')
    out = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', [
        'run_ee_adc2.py', '--xyz', str(xyz), '--basis', 'sto-3g',
        '--auxbasis', 'cc-pvdz-ri', '--jk-auxbasis', 'cc-pvdz-jkfit',
        '--nroots', '2', '--frozen', '0', '--scf-conv-tol', '1e-30',
        '--scf-conv-tol-grad', '1e-30', '--out-dir', str(out)])
    with pytest.warns(RuntimeWarning, match='SCF did not converge'):
        assert tool().main() == 0
    record = json.loads((out / 'record.json').read_text())
    assert record['scf']['converged'] is False
    assert 'scf_not_converged' in record['dEST_flags']
    for spin in ('singlet', 'triplet'):
        channel = json.loads((out / f'channel_{spin}.json').read_text())
        assert 'scf_not_converged' in [f['flag'] for f in channel['flags']]
