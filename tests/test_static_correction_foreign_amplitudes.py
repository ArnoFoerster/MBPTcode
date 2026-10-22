"""Pins build_mp2_static_correction_restricted(amplitudes=...) -- the Sigma(inf)
partner of a CC-ADC(3) amplitude substitution.

The correction evaluates the SAME second-order Delta_gamma^(2) functional the
MP2 static uses, at foreign (CC) amplitudes instead of the MBPT T2^(1)/T1^(2)
it would otherwise generate. Because bra amplitudes are the ket adjoints (the
driver's own _to_l_restricted, the same helper cc_amplitudes_restricted uses),
this is an EXPECTATION-VALUE density -- deliberately matching how the CC-ADC(3)
supermatrix is built -- and NOT the Lambda-CCSD response 1-RDM that
build_ccsd_static_correction_restricted computes. The two are different objects
and this test does not claim otherwise.

What is actually pinned:

1. IDENTITY. Feeding the MBPT pool back in must reproduce the default path
   EXACTLY (dense) / to DF round-off (B_aa). This is the gate that the index
   conventions of cc_adc_amplitudes.py's pool and the density driver's t2_1/t1_2
   parameters agree, and that nothing in the functional was reimplemented.
   It holds for relax=False and relax=True (CPHF) alike.
2. DF/dense agreement at foreign amplitudes -- the DF route is the reason this
   parameter exists (the Lambda-CCSD builder forms the dense norb^4 ERI and so
   cannot run at aug-cc-pVQZ).
3. NON-TRIVIALITY. A CCSD pool must give a different correction from the MBPT
   one, or the parameter would be a silent no-op and tests 1-2 would pass
   vacuously.
4. The guard rails: amplitudes= is rejected with EN dressing, with a non-MP2
   kind, and on a shape mismatch, rather than silently computing something else.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals
from src.SingleReference.ADC.cc_adc_amplitudes import (cc_amplitudes_restricted,
                                                       mbpt_amplitudes_restricted)
from src.SingleReference.ADC.static_correction import (
    build_mp2_static_correction_restricted, build_static_correction)

SYSTEMS = [('H 0 0 0; F 0 0 0.917', '6-31g'),
           ('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', '6-31g')]
IDENTITY_TOL = 1e-12      # dense is exact; DF carries RI round-off only
DF_TOL = 1e-10


def _run(atom, basis):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.run()
    return mol, mf, mol.nelectron // 2


def main():
    ok = True
    for atom, basis in SYSTEMS:
        mol, mf, nocc = _run(atom, basis)
        label = f'{atom.split()[0]}.../{basis}'
        mbpt = mbpt_amplitudes_restricted(mf)
        ccsd = cc_amplitudes_restricted(mf, method='ccsd')
        B_aa = DFIntegrals.from_scf(mol, mf).B_aa

        # 1. identity: the MBPT pool reproduces the internally generated one
        for relax in (False, True):
            for tag, kw, tol in (('dense', {}, IDENTITY_TOL),
                                 ('df', {'B_aa': B_aa}, DF_TOL)):
                ref = build_mp2_static_correction_restricted(
                    mf, mol, nocc, relax=relax, **kw)
                got = build_mp2_static_correction_restricted(
                    mf, mol, nocc, relax=relax, amplitudes=mbpt, **kw)
                d = float(np.max(np.abs(ref - got)))
                good = d < tol
                ok &= good
                print(f'  [{"OK" if good else "FAIL"}] {label} identity '
                      f'relax={relax!s:5s} {tag:5s}  max|d| = {d:.2e}')

        # 2. DF vs dense at foreign amplitudes
        dense = build_mp2_static_correction_restricted(
            mf, mol, nocc, relax=True, amplitudes=ccsd)
        dfval = build_mp2_static_correction_restricted(
            mf, mol, nocc, relax=True, amplitudes=ccsd, B_aa=B_aa)
        d = float(np.max(np.abs(dense - dfval)))
        good = d < DF_TOL
        ok &= good
        print(f'  [{"OK" if good else "FAIL"}] {label} DF vs dense (CCSD pool) '
              f'  max|d| = {d:.2e}')

        # 3. non-triviality: CCSD amplitudes must actually change the answer
        base = build_mp2_static_correction_restricted(mf, mol, nocc, relax=True)
        d = float(np.max(np.abs(dense - base)))
        good = d > 1e-6
        ok &= good
        print(f'  [{"OK" if good else "FAIL"}] {label} CCSD pool differs from MBPT'
              f'   max|d| = {d:.2e}')

    # 4. guard rails
    mol, mf, nocc = _run(*SYSTEMS[0])
    ccsd = cc_amplitudes_restricted(mf, method='ccsd')
    en = {'hh': True, 'pp': True, 'singles': False, 'spin_adapted': True,
          'shift': 'sum'}
    checks = [
        ('EN dressing + amplitudes', ValueError,
         lambda: build_mp2_static_correction_restricted(
             mf, mol, nocc, relax=True, amplitudes=ccsd, u2_denom_dress=en)),
        ('non-MP2 kind + amplitudes', ValueError,
         lambda: build_static_correction(mf, mol, kind='ccsd', amplitudes=ccsd)),
        ('shape mismatch', ValueError,
         lambda: build_mp2_static_correction_restricted(
             mf, mol, nocc, relax=True,
             amplitudes=dict(ccsd, t1_aa=np.zeros((3, 3))))),
        ('missing key', KeyError,
         lambda: build_mp2_static_correction_restricted(
             mf, mol, nocc, relax=True,
             amplitudes={k: v for k, v in ccsd.items() if k != 't2_abab'})),
    ]
    for name, exc, fn in checks:
        try:
            fn()
        except exc:
            print(f'  [OK] rejected: {name}')
        except Exception as e:                                  # noqa: BLE001
            ok = False
            print(f'  [FAIL] {name}: raised {type(e).__name__}, expected {exc.__name__}')
        else:
            ok = False
            print(f'  [FAIL] {name}: did not raise')

    print('\nALL PASS' if ok else '\nFAILURES')
    return 0 if ok else 1


def test_main():
    assert not main(), 'main() returned a nonzero exit code'


if __name__ == '__main__':
    sys.exit(main())
