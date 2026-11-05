"""The two python examples of docs/methods/properties.md run as written.

Each block is read out of the page and run from the repository root, the way
the page tells a reader to run it: the Duschinsky band of formaldehyde S0 ->
T1 must print a positive width and peak, and the rate chain on DMABN's S1 and
T1 must print a positive ISC rate and one line per rung of the ablation
ladder.
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def blocks():
    """The python blocks of the page, in order."""
    with open(os.path.join(REPO, 'docs', 'methods', 'properties.md')) as f:
        text = f.read()
    return [part.split('```', 1)[0] for part in text.split('```python\n')[1:]]


def run(block):
    """stdout of `block`, run from the repository root."""
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''))
    with tempfile.TemporaryDirectory() as where:
        script = os.path.join(where, 'run.py')
        with open(script, 'w') as f:
            f.write(block)
        out = subprocess.run([sys.executable, script], cwd=REPO, env=env,
                             capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()


def test_the_duschinsky_band_example_runs():
    block, = [b for b in blocks() if 'DuschinskyOscillators' in b]
    fwhm, peak = (float(x) for x in run(block)[-1].split())
    assert fwhm > 0.0 and peak > 0.0


def test_the_rate_chain_example_runs():
    block, = [b for b in blocks() if 'ablation_ladder' in b]
    lines = run(block)
    assert float(lines[0]) > 0.0
    assert len(lines) > 5
