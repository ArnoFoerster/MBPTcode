#!/usr/bin/env python
"""Does every production module still import?

One second, and it catches the class of breakage that no physics test does: a
module that references a name another module no longer defines. Every other
test in this directory imports only the few modules it exercises, so a rename
that misses one caller passes all of them and fails at the front door.

    python tests/test_imports.py            production packages, ~1 s
    python tests/test_imports.py --all      everything under src/, slower

The default set is the packages that change together -- Base, GW, BSE,
LinearResponse, Solvers, and the gradients and properties built on them. `--all`
adds ADC, CC and DensityMatrix, whose generated modules are large and slow to
parse. Under pytest the default set runs in its own process
(`test_all_core_modules_import`).
"""
import argparse
import importlib
import os
import pathlib
import subprocess
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORE = ['src/Base', 'src/Solvers', 'src/SingleReference/GW',
        'src/SingleReference/LinearResponse', 'src/gradients', 'src/properties']
SKIP = {'__pycache__', 'data'}


def modules(roots):
    for r in roots:
        for f in sorted((ROOT / r).rglob('*.py')):
            if any(p in SKIP for p in f.parts) or f.name.startswith('_'):
                continue
            yield '.'.join(f.relative_to(ROOT).with_suffix('').parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='every package under src/')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()

    roots = ['src'] if a.all else CORE
    bad, skipped = [], []
    n = 0
    for name in modules(roots):
        n += 1
        try:
            importlib.import_module(name)
            if a.verbose:
                print(f'  ok   {name}')
        except ModuleNotFoundError as exc:
            # An OPTIONAL third-party package (elpa, mpi4py ...) missing on this
            # machine is not a broken repo. A missing `src.` module is.
            missing = (exc.name or '')
            if missing.split('.')[0] == 'src':
                bad.append((name, exc))
                print(f'  FAIL {name}\n       {type(exc).__name__}: {exc}')
            else:
                skipped.append((name, missing))
                if a.verbose:
                    print(f'  skip {name}  (optional: {missing})')
        except Exception as exc:
            bad.append((name, exc))
            print(f'  FAIL {name}\n       {type(exc).__name__}: {exc}')
            if a.verbose:
                traceback.print_exc()

    if skipped:
        opt = sorted({m for _, m in skipped})
        print(f'\n  {len(skipped)} skipped, optional dependency absent: {", ".join(opt)}')
    print(f'{n - len(bad) - len(skipped)}/{n - len(skipped)} modules import')
    if bad:
        print('BROKEN')
        return 1
    print('ALL PASSED')
    return 0


def test_all_core_modules_import():
    """The import sweep, run as pytest collects it: in its own process, so a
    module that fails at import cannot take the rest of the session with it,
    and with mpi4py's initialization deferred, since only a module that starts
    MPI at import would need it here."""
    env = dict(os.environ, MPI4PY_RC_INITIALIZE='0')
    done = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())],
                          capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert done.returncode == 0, (
        f'import sweep failed (exit {done.returncode}):\n'
        f'{done.stdout[-4000:]}\n{done.stderr[-2000:]}')
    assert 'ALL PASSED' in done.stdout


if __name__ == '__main__':
    sys.exit(main())
