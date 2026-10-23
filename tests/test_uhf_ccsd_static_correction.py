"""Correctness oracle for the new UHF branch of build_ccsd_static_correction
(kind='ccsd' static self-energy from the UCCSD Lambda density).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC.static_correction import build_static_correction, build_ccsd_static_correction
from src.Base.pyscf.pyscf_interface import uhf_blockstacked_order


def interleaved_to_blockstacked_perm(nocc_a, nocc_b, norb):
    order = uhf_blockstacked_order(nocc_a, nocc_b, norb, norb)
    return np.array([2 * u if u < norb else 2 * (u - norb) + 1 for u in order])


if __name__ == '__main__':
    all_ok = True

    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.run()
    nocc = mol.nelectron // 2
    norb = mf.mo_coeff.shape[1]
    mfu = scf.addons.convert_to_uhf(mf)
    mfu.conv_tol = 1e-12
    mfu.run()
    perm = interleaved_to_blockstacked_perm(nocc, nocc, norb)

    sc_r = build_ccsd_static_correction(mf, mol)
    sc_u = build_ccsd_static_correction(mfu, mol)
    diff = np.abs(sc_r[np.ix_(perm, perm)] - sc_u).max()
    thresh = 1e-6
    ok = diff < thresh
    all_ok &= ok
    print(f"closed-shell H2O, ccsd static correction: "
          f"max|RHF branch - UHF branch| = {diff:.2e}: {'OK' if ok else 'FAIL'}")

    # en_dress must still be rejected for kind='ccsd' (no EN hook on a CC density)
    try:
        build_static_correction(mfu, mol, kind='ccsd', en_dress={'hh': True})
        print("en_dress rejection: FAIL (should have raised)")
        all_ok = False
    except ValueError:
        print("en_dress rejection: OK")

    # ccsdt must still raise for UHF (not implemented)
    try:
        build_static_correction(mfu, mol, kind='ccsdt')
        print("ccsdt UHF rejection: FAIL (should have raised)")
        all_ok = False
    except NotImplementedError:
        print("ccsdt UHF rejection: OK")

    print()
    print("ALL PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)
