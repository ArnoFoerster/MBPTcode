"""The vertical ADC(2) run (`python -m src.SingleReference.ADC.eeADC`,
eeADC/vertical.py) carries on past an SCF that did not converge, warned and
flagged (refuse inputs, warn numerics):
the record's scf.converged is False, each channel's flags and
dEST_flags name 'scf_not_converged'. Water STO-3G, an SCF asked for an
orbital gradient it cannot reach in its 200 cycles. The module runs from the
command line.
"""
import json
import pathlib
import subprocess
import sys

import pytest

from src.SingleReference.ADC.eeADC import vertical


def test_an_unconverged_scf_is_flagged_and_the_run_carries_on(tmp_path):
    xyz = tmp_path / 'water.xyz'
    xyz.write_text('3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n')
    out = tmp_path / 'run'
    argv = ['--xyz', str(xyz), '--basis', 'sto-3g',
            '--auxbasis', 'cc-pvdz-ri', '--jk-auxbasis', 'cc-pvdz-jkfit',
            '--nroots', '2', '--frozen', '0', '--scf-conv-tol', '1e-30',
            '--scf-conv-tol-grad', '1e-30', '--out-dir', str(out)]
    with pytest.warns(RuntimeWarning, match='SCF did not converge'):
        assert vertical.main(argv) == 0
    record = json.loads((out / 'record.json').read_text())
    assert record['scf']['converged'] is False
    assert 'scf_not_converged' in record['dEST_flags']
    for spin in ('singlet', 'triplet'):
        channel = json.loads((out / f'channel_{spin}.json').read_text())
        assert 'scf_not_converged' in [f['flag'] for f in channel['flags']]


def test_the_module_runs_from_the_command_line():
    repo = pathlib.Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, '-m', 'src.SingleReference.ADC.eeADC',
                          '--help'], cwd=repo, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert 'python -m src.SingleReference.ADC.eeADC' in out.stdout
    assert '--state-specific' in out.stdout
