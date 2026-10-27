"""The MPI example in docs/parallel.md runs as written on one process.

The block is read out of the page and run with `MBPT_USE_MPI=0` from a foreign
working directory, the serial run the page promises is bit for bit the serial
code. It must finish and print five excitation energies.
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def parallel_block():
    """The python block under `## Running under MPI`."""
    with open(os.path.join(REPO, 'docs', 'parallel.md')) as f:
        text = f.read()
    section = text.split('## Running under MPI', 1)[1]
    return section.split('```python\n', 1)[1].split('```', 1)[0]


def test_parallel_example_runs_serially():
    env = dict(os.environ, MBPT_USE_MPI='0',
               PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''))
    with tempfile.TemporaryDirectory() as where:
        script = os.path.join(where, 'run.py')
        with open(script, 'w') as f:
            f.write(parallel_block())
        run = subprocess.run([sys.executable, script], cwd=where, env=env,
                             capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    last = run.stdout.strip().splitlines()[-1]
    omega = [float(x) for x in last.strip('[]').split()]
    assert len(omega) == 5 and all(0.0 < w < 1.0 for w in omega), last
