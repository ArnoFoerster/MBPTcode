"""The Casida normalization is a contract between a solver and its consumers.

<X|X> - <Y|Y> = 1 is what every solver in this repo returns and what the
spin-adaptation factors downstream are derived for -- the sqrt(2) in the
transition dipole. pySCF returns 1/2 instead.

The difference cancels out of every excitation energy, so nothing in a spectrum
reveals it, and the consumer is QUADRATIC in the vector: a factor of two in an
oscillator strength. It is therefore refused by name at the boundary rather
than rescaled, and `from_pyscf` converts in one visible line.

Run: python tests/test_casida_normalization.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.constants import CASIDA_NORM_TOL
from src.SingleReference.LinearResponse.davidson import oscillator_strengths
from src.SingleReference.LinearResponse.linear_response import (
    check_normalization, from_pyscf)


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build():
    """(mol, mf, td) with pySCF's own TDHF roots to convert and compare against."""
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    td = mf.TDHF()
    td.nstates = 3
    td.kernel()
    return mol, mf, td


def test_oscillator_strengths_refuse_a_pyscf_vector(mol, mf, td):
    """The mistake this catches is silent: f comes back a factor of two small
    and every root sits at the right energy."""
    nocc = mol.nelectron // 2
    raw = np.column_stack([np.asarray(xy[0]).ravel() for xy in td.xy])
    raw_y = np.column_stack([np.asarray(xy[1]).ravel() for xy in td.xy])
    try:
        oscillator_strengths(mf, mol, nocc, td.e, raw, raw_y)
        return check(False, "a pySCF vector is refused by name")
    except ValueError as exc:
        return check("pySCF's convention" in str(exc),
                     'a pySCF vector is refused by name', str(exc).split('--')[0].strip())


def test_the_converted_vector_reproduces_pyscf_oscillator_strengths(mol, mf, td):
    """THE ABSOLUTE GATE ON THE PAIR. An independent implementation with the
    other convention and its own factors agrees only if this repo's sqrt(2)
    goes with this repo's normalization; either one alone is unfalsifiable."""
    nocc = mol.nelectron // 2
    omega, x, y = from_pyscf(td)
    f, _ = oscillator_strengths(mf, mol, nocc, omega, x, y)
    d = np.abs(np.sort(f) - np.sort(td.oscillator_strength(gauge='length'))).max()
    return check(d < 1e-10, 'converted, f matches pySCF exactly', f'max |df| {d:.1e}')


def test_the_tolerance_admits_a_loose_root_and_never_one_half():
    """It has to straddle a real gap: a Davidson root at conv_tol 1e-5 carries
    a norm error far above machine precision, and 1/2 must never pass."""
    x = np.ones((6, 1)) / np.sqrt(6.0)
    check_normalization(np.sqrt(1.0 + 0.5 * CASIDA_NORM_TOL) * x)
    ok = check(True, 'a root just inside the tolerance passes',
               f'tol {CASIDA_NORM_TOL:g}')
    for bad, name in ((np.sqrt(0.5), "pySCF's 1/2"), (np.sqrt(2.0), 'twice'),
                      (1.0 + 10 * CASIDA_NORM_TOL, 'ten tolerances out')):
        try:
            check_normalization(bad * x)
            ok &= check(False, f'{name} is refused')
        except ValueError as exc:
            ok &= check('not 1 to' in str(exc), f'{name} is refused')
    return ok


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    mol, mf, td = build()
    all_ok = True
    print('\n-- the boundary refuses the other convention')
    all_ok &= test_oscillator_strengths_refuse_a_pyscf_vector(mol, mf, td)
    all_ok &= test_the_tolerance_admits_a_loose_root_and_never_one_half()
    print('\n-- converted, the two implementations agree')
    all_ok &= test_the_converted_vector_reproduces_pyscf_oscillator_strengths(mol, mf, td)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
