"""Run one tier of the test suite, optionally one shard of it.

    python tools/ci/run_tier.py fast --shard 1/4
    python tools/ci/run_tier.py slow --shard 2/3 --timeout 3600

A tier is a list in tools/ci/<tier>.txt, one `pytest <file>` or `script <file>`
per line; a `#` starts a comment (the reason a file is in triage.txt or
heavy.txt). Every test file is in exactly one of fast, slow, heavy, triage.
Files written as scripts print their own verdict and exit non-zero
(tests/README.md); pytest collects nothing from them, so they run as
`python tests/<file>`; a script that prints `FAILURES DETECTED` fails even if
it exits 0 (test_constants_registry does). Every file runs in its own process.
The run goes on past a failure and exits non-zero at the end if any file
failed or timed out.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def read_tier(name):
    entries = []
    with open(os.path.join(HERE, name + '.txt')) as f:
        for line in f:
            line = line.split('#')[0]
            if line.strip():
                mode, test = line.split()
                entries.append((mode, test))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tier')
    ap.add_argument('--shard', default='1/1', help='i/n: every n-th file from i')
    ap.add_argument('--timeout', type=int, default=900, help='seconds per file')
    args = ap.parse_args()

    i, n = (int(x) for x in args.shard.split('/'))
    entries = read_tier(args.tier)[i - 1::n]

    env = dict(os.environ)
    # The suite's thread policy (tests/conftest.py, tests/README.md).
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(var, '2')

    failed = []
    for mode, test in entries:
        path = os.path.join('tests', test)
        cmd = ([sys.executable, '-m', 'pytest', path, '-q', '-p', 'no:cacheprovider']
               if mode == 'pytest' else [sys.executable, path])
        t0 = time.time()
        print(f'::group::{test} ({mode})', flush=True)
        try:
            if mode == 'pytest':
                rc = subprocess.run(cmd, cwd=REPO, env=env,
                                    timeout=args.timeout).returncode
                marked = False
            else:
                run = subprocess.run(cmd, cwd=REPO, env=env, timeout=args.timeout,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, errors='replace')
                print(run.stdout, end='', flush=True)
                rc, marked = run.returncode, 'FAILURES DETECTED' in run.stdout
            status = ('ok' if rc == 0 and not marked else
                      f'FAILED (exit {rc})' if rc else 'FAILED (prints FAILURES DETECTED)')
        except subprocess.TimeoutExpired:
            status = f'TIMEOUT after {args.timeout} s'
        print('::endgroup::', flush=True)
        print(f'{test}: {status} in {time.time() - t0:.0f} s', flush=True)
        if status != 'ok':
            failed.append(f'{test}: {status}')

    print(f'\n{len(entries) - len(failed)} of {len(entries)} files passed '
          f'(tier {args.tier}, shard {args.shard})')
    for line in failed:
        print('  ' + line)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
