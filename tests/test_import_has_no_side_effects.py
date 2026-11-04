"""Importing the code sets no thread variable of the process.

WHAT BREAKS IF THIS FAILS. BLAS and OpenMP size their thread pools from
OMP_NUM_THREADS, MKL_NUM_THREADS and their kin when they first start. A module
that sets one of them at import (say to the core count less one) silently
gives every later calculation in that process its own thread count: ranks on
one node oversubscribe it, and a threaded GEMM adds its partials in another
order, so the bits of a pinned result move. The thread count is the caller's
(`Base/utils/parallel/threads`, a launcher's `set_threads`).

One subprocess, its environment stripped of every thread variable, imports
every module under src except `__main__` modules (which run a command line)
and `src.autograd_oracle`, whose scripts are processes of their own that fix
their own threads by design. A module whose optional dependency is not
installed is skipped.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')

PROBE = r'''
import importlib, json, os, pkgutil, sys
sys.path.insert(0, sys.argv[1])
import src
failed = []
for m in pkgutil.walk_packages(src.__path__, 'src.'):
    if m.name.endswith('__main__') or m.name.startswith('src.autograd_oracle'):
        continue
    try:
        importlib.import_module(m.name)
    except ModuleNotFoundError as err:
        if (err.name or '').split('.')[0] == 'src':
            failed.append((m.name, str(err)))
    except Exception as err:
        failed.append((m.name, f'{type(err).__name__}: {err}'))
    leaked = sorted(v for v in json.loads(sys.argv[2]) if v in os.environ)
    if leaked:
        print(json.dumps({'module': m.name, 'leaked': leaked}))
        sys.exit(0)
print(json.dumps({'module': None, 'leaked': [], 'failed': failed}))
'''


def test_importing_src_sets_no_thread_variable():
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARIABLES}
    out = subprocess.run([sys.executable, '-c', PROBE, str(REPO),
                          json.dumps(THREAD_VARIABLES)],
                         env=env, capture_output=True, text=True, check=True)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got['leaked'] == [], (
        f"importing {got['module']} set {got['leaked']}")
    assert got['failed'] == [], got['failed']
