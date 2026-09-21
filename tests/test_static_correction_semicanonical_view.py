"""The semicanonical view of a Kohn-Sham reference keeps the direct-SCF optimizer.

_ks_semicanonical_setup and its UHF twin hand the MP2/MP3 static correction a
shallow copy of the mean field with semicanonical orbitals. A molecule too
large for in-core J/K takes PySCF's direct branch, which reads the optimizer
the mean field caches; a copy made through the pickle hooks arrives without it
and a J/K build on it raises. `max_memory = 0` with the cached ERI cleared
forces that branch on water and on the OH radical, and the view's V_Hxc must
equal the in-core one to 1e-10 Ha.

Run: python tests/test_static_correction_semicanonical_view.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import dft, gto

from src.SingleReference.ADC.static_correction import (_ks_semicanonical_setup,
                                                       _ks_semicanonical_setup_uhf)


def check(ok, label, detail=''):
    """Print an [ok]/[FAIL] verdict line for `label` and return `ok`."""
    print(f"  [{'ok' if ok else 'FAIL'}] {label}"
          + (f'   ({detail})' if detail else ''))
    return bool(ok)


def direct_veff_matches(view, mf):
    """(ok, detail): V_Hxc of `view` on the direct J/K path against `mf`'s
    in-core one, both on `mf`'s own density. The view shares `mf`'s cached ERI
    until it is cleared, and the in-core branch would store a fresh one, so an
    ERI still absent after the build is the witness that the direct branch ran."""
    dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    view.max_memory = 0
    view._eri = None
    try:
        d = np.abs(np.asarray(view.get_veff(mf.mol, dm))
                   - np.asarray(mf.get_veff(mf.mol, dm))).max()
        return (d < 1e-10 and view._eri is None,
                f'max |d V_Hxc| {d:.1e} Ha, ERI cached: {view._eri is not None}')
    except AttributeError as err:
        return False, f'direct get_veff raised: {err}'


def test_the_restricted_view():
    """Water, RKS PBE0: _ks_semicanonical_setup's view on the direct path."""
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    view = _ks_semicanonical_setup(mf, mol, nocc=mol.nelectron // 2, ncore=0)[0]
    ok, detail = direct_veff_matches(view, mf)
    return check(ok, 'RKS: the semicanonical view builds V_Hxc on the direct J/K path',
                 detail)


def test_the_unrestricted_view():
    """OH, UKS PBE0: _ks_semicanonical_setup_uhf's view on the direct path."""
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    view = _ks_semicanonical_setup_uhf(mf, mol, ncore=0)[0]
    ok, detail = direct_veff_matches(view, mf)
    return check(ok, 'UKS: the semicanonical view builds V_Hxc on the direct J/K path',
                 detail)


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    all_ok = True
    all_ok &= test_the_restricted_view()
    all_ok &= test_the_unrestricted_view()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
