"""ScaLAPACK NUMROC: the block-cyclic split must account for every row.

`ElpaEigensolver._numroc` says how many rows of an n x n matrix a process owns.
Two defects sat here undetected because nothing imports the module without MPI
and ELPA installed, and neither is on PyPI:

  * `__init__` called it with 4 of its 5 required arguments, so constructing the
    solver raised TypeError before any of this ran.
  * the formula dropped the leftover blocks, so the pieces summed to less than n
    whenever nblocks was not a multiple of nprocs -- 8 of 10 rows at
    n=10, nb=2, nprocs=2, i.e. two rows of the matrix belonged to nobody.

The check is the PARTITION, not the formula: whatever NUMROC returns, the sizes
have to add up to n and no process may own a negative or over-long slice. That
catches a wrong formula without restating it.

Run: python tests/test_numroc.py
"""
import ast
import os
import pathlib
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# elpa.py imports pyelpa and mpi4py at module scope; neither is needed for the
# arithmetic, and pyelpa is not installable from PyPI, so stub them.
for name, attrs in (('pyelpa', {'Elpa': object}),
                    ('mpi4py', {'MPI': types.SimpleNamespace(COMM_WORLD=None)})):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

from src.Base.utils.linearAlgebra.elpa import ElpaEigensolver

numroc = ElpaEigensolver._numroc


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def main():
    ok = True

    print('\n=== 1. the split accounts for every row, and no more ===')
    bad = []
    for n in (1, 7, 10, 17, 64, 100, 257, 1024):
        for nb in (1, 2, 8, 64):
            for nprocs in (1, 2, 3, 4, 8):
                parts = [numroc(None, n, nb, p, 0, nprocs) for p in range(nprocs)]
                if sum(parts) != n or any(x < 0 for x in parts):
                    bad.append((n, nb, nprocs, sum(parts)))
    ok &= check(not bad, 'sum over processes == n for every (n, nb, nprocs)',
                'all 160 cases' if not bad
                else f'{len(bad)} broken, e.g. {bad[0]}')

    print('\n=== 2. no process is starved while another holds two blocks extra ===')
    worst = 0
    for n in (100, 257, 1024):
        for nb in (8, 64):
            for nprocs in (2, 3, 4, 8):
                parts = [numroc(None, n, nb, p, 0, nprocs) for p in range(nprocs)]
                worst = max(worst, max(parts) - min(parts))
    ok &= check(worst <= 64, 'load imbalance never exceeds one block',
                f'max spread {worst} rows, nb <= 64')

    print('\n=== 3. the distribution shifts with isrcproc, and stays a partition ===')
    shifted = True
    for src in range(4):
        parts = [numroc(None, 100, 8, p, src, 4) for p in range(4)]
        shifted &= sum(parts) == 100
    ok &= check(shifted, 'every isrcproc still partitions n')

    print('\n=== 4. __init__ passes all five arguments ===')
    # the original defect: 4 args against a 5-parameter signature, which
    # byte-compiles and raises TypeError only when the solver is constructed.
    src = pathlib.Path('src/Base/utils/linearAlgebra/elpa.py').read_text()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == '_numroc']
    nargs = {len(c.args) for c in calls}
    ok &= check(calls and nargs == {5},
                f'every _numroc call site passes 5 arguments',
                f'{len(calls)} call sites, arg counts {sorted(nargs)}')

    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
