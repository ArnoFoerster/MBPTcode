"""EE-ADC spin channel: the lowest roots are found whatever their symmetry.

The operator commutes with the molecule's point group, so a Davidson search stays
inside the symmetry species its start vectors carry. Unit vectors on the lowest
diagonal entries can all miss the species of a low state; the channel solve then
returns a higher root in its place and reports convergence.

Benzene (regular D6h, C-C 1.39 and C-H 1.09 Angstrom) in cc-pVDZ at ADC(2), density
fitted on pyscf's default auxiliary set, 1s frozen: its third and fourth singlets are
a degenerate pair. The check is that the singlet channel at nroots=3 returns pyscf
RADC's three lowest singlets to 1e-3 eV, each with parity +1, and warns of nothing.

Reference: pyscf RADC, nroots=5, same reference and fitting set: 5.4443, 6.8020,
7.6729, 7.6729, 8.1812 eV (run 2026-09-28_benzene-radc-pyscf-5roots).

Before it, without an SCF: at max_subspace=0 and one root the seeds number one,
as solve_symmetric keeps room for, and the solve runs.

Run: python tests/test_ee_adc_channel_seeds.py
"""
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC.eeADC.ee_driver import _open_seeds, solve_ee_adc
from src.Solvers.davidson import solve_symmetric

HARTREE_TO_EV = 27.211386245988
REFERENCE_EV = np.array([5.4443, 6.8020, 7.6729])
TOL_EV = 1e-3       # the reference prints 4 decimals; a skipped root is 0.5 eV off


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def benzene():
    """Regular D6h ring in the xy plane, carbons then hydrogens."""
    atoms = []
    for i in range(6):
        a = math.pi * i / 3.0
        atoms.append(f'C {1.39 * math.cos(a):.6f} {1.39 * math.sin(a):.6f} 0')
        atoms.append(f'H {2.48 * math.cos(a):.6f} {2.48 * math.sin(a):.6f} 0')
    return '; '.join(atoms)


def check_zero_subspace():
    """max_subspace=0 at one root: solve_symmetric keeps room for one vector, so
    the seeds must number one too; more made davidson1 raise IndexError."""
    rng = np.random.default_rng(0)
    n = 20
    A = rng.standard_normal((n, n))
    A = 0.05 * (A + A.T) + np.diag(np.arange(1.0, n + 1.0))
    diag = np.diag(A).copy()
    x0 = _open_seeds(diag, 1, 0)
    ok = check(x0.shape == (n, 1), 'max_subspace=0: one seed for one root',
               f'shape {x0.shape}')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            solve_symmetric(lambda x: A @ x, diag, nroots=1, x0=x0,
                            max_subspace=0)
        ok &= check(True, 'max_subspace=0: the solve runs')
    except Exception as exc:
        ok &= check(False, 'max_subspace=0: the solve runs',
                    f'{type(exc).__name__}: {exc}')
    return ok


def check_benzene_singlets():
    """The singlet channel's three roots against pyscf's three lowest singlets."""
    mol = gto.M(atom=benzene(), basis='cc-pvdz', unit='Angstrom', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    ncore = int((mol.atom_charges() > 2).sum())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        e, _, parity = solve_ee_adc(mf, level='adc2', nroots=3, spin='singlet',
                                    frozen=ncore, df=True, return_parity=True)
    warned = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    e_ev = np.sort(np.asarray(e)) * HARTREE_TO_EV
    d = np.abs(e_ev - REFERENCE_EV).max()
    ok = check(d < TOL_EV, "the singlet channel's three roots are pyscf's three "
               'lowest singlets', f"{', '.join(f'{x:.4f}' for x in e_ev)} eV, "
               f'max |d| = {d:.1e} eV')
    ok &= check(np.allclose(parity, 1.0, atol=1e-6), 'every root has parity +1')
    ok &= check(not warned, 'the solve raises no RuntimeWarning',
                '; '.join(str(w.message)[:80] for w in warned))
    return ok


def run():
    print('=== the seed count at max_subspace=0, nroots=1 ===')
    all_ok = check_zero_subspace()
    print('\n=== benzene / cc-pVDZ, ADC(2) singlet channel, nroots=3 ===')
    all_ok &= check_benzene_singlets()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return all_ok


def test_ee_adc_channel_seeds_checks():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
