"""EE-ADC Davidson: the subspace gets the memory the SCF object's budget leaves.

pyscf's davidson1 holds its subspace in memory up to max_memory MB and in HDF5
files beyond. The EE-ADC routes pass it mf.max_memory, the budget of the whole
process in pyscf's convention, less what the process holds when the solve starts.

Water in cc-pVDZ at ADC(2), density fitted, 1s frozen: its singlet channel (above
the dense limit) goes through the Davidson. With mf.max_memory = 1 MB the budget
reaching davidson1 is 0 and the subspace goes to disk; with 10**6 MB it is that
budget less the process's use, under 10 GB here. Both give the same roots to
1e-10 Eh, since only where the subspace is stored differs. solve_symmetric called
without max_memory hands davidson1 none, so its other callers keep pyscf's default.

Run: python tests/test_ee_adc_davidson_max_memory.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
from src.Solvers import davidson as dav

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
LARGE_MB = 10**6
TOL_EH = 1e-10      # the two solves differ only in where the subspace is stored


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def spy_davidson1(seen):
    """Wrap pyscf's davidson1 so each call records the max_memory it received
    ('absent' when the caller passed none, apart from an explicit None, which
    davidson1 cannot take); returns the undo."""
    davidson1 = dav.pyscf_lib.davidson1

    def spy(*args, **kw):
        seen.append(kw.get('max_memory', 'absent'))
        return davidson1(*args, **kw)
    dav.pyscf_lib.davidson1 = spy
    return lambda: setattr(dav.pyscf_lib, 'davidson1', davidson1)


def solve(mf, budget_mb, ncore):
    """Singlet channel, three roots, at mf.max_memory = budget_mb; returns the
    energies and the max_memory davidson1 received."""
    mf.max_memory = budget_mb
    seen = []
    undo = spy_davidson1(seen)
    try:
        e, _ = solve_ee_adc(mf, level='adc2', nroots=3, spin='singlet',
                            frozen=ncore, df=True)
    finally:
        undo()
    return np.sort(np.asarray(e)), seen


def test_ee_adc_budget():
    """The budget reaches davidson1, and the roots do not depend on it."""
    mol = gto.M(atom=WATER, basis='cc-pvdz', unit='Angstrom', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    ncore = int((mol.atom_charges() > 2).sum())
    e_small, seen_small = solve(mf, 1, ncore)
    e_large, seen_large = solve(mf, LARGE_MB, ncore)
    ok = check(seen_small == [0], 'mf.max_memory = 1 MB reaches davidson1 as 0',
               f'received {seen_small}')
    held = [LARGE_MB - m for m in seen_large if isinstance(m, (int, float))]
    ok &= check(len(held) == 1 and 0 < held[0] < 10**4,
                f'mf.max_memory = {LARGE_MB} MB reaches davidson1 less the '
                "process's use", f'received {seen_large}')
    d = np.abs(e_small - e_large).max()
    ok &= check(d < TOL_EH, 'the roots on disk and in memory agree',
                f'max |d| = {d:.1e} Eh')
    return ok


def test_default_untouched():
    """solve_symmetric without max_memory leaves davidson1's default alone."""
    rng = np.random.default_rng(0)
    n = 300
    a = rng.standard_normal((n, n))
    a = 0.01 * (a + a.T) + np.diag(np.arange(1.0, n + 1.0))
    seen, e, err = [], None, ''
    undo = spy_davidson1(seen)
    try:
        e, _, _ = dav.solve_symmetric(lambda x: a @ x, np.diag(a), nroots=2)
    except Exception as exc:
        err = f'; {type(exc).__name__}: {exc}'
    finally:
        undo()
    ok = check(seen == ['absent'], 'solve_symmetric without max_memory passes none',
               f'received {seen}{err}')
    d = np.inf if e is None else np.abs(np.asarray(e) - np.linalg.eigvalsh(a)[:2]).max()
    ok &= check(d < 1e-6, 'and still finds the two lowest eigenvalues',
                f'max |d| = {d:.1e}')
    return ok


if __name__ == '__main__':
    print('=== water / cc-pVDZ, ADC(2) singlet channel: the subspace budget ===')
    all_ok = test_ee_adc_budget()
    print('=== solve_symmetric without max_memory ===')
    all_ok &= test_default_untouched()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
