"""
Validates the DF/RI variant of ADCSolverRestricted.build_supermatrix/
build_matrix_free_operator
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.SingleReference.ADC import ADCSolverRestricted
from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_two_electron_integrals_chemist, DFIntegrals


if __name__ == '__main__':
    all_ok = True

    for atom, basis in [('H 0 0 0; F 0 0 0.9', 'sto-3g'), ('O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', '6-31g')]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc = mol.nelectron // 2
        eps = get_orbital_energies(mf, representation='spatial')
        eri_chemist = get_two_electron_integrals_chemist(mol, mf)

        for level in ('adc2x', 'adc3'):
            dense = ADCSolverRestricted.from_arrays(eps, eri_chemist, level=level)
            H_dense = dense.build_supermatrix(nocc)
            poles_dense = np.linalg.eigvalsh(H_dense)

            dfi_exact = DFIntegrals.from_scf(mol, mf, exact=True)
            drv_exact = ADCSolverRestricted.from_arrays(eps, eri_chemist, level=level, B_aa=dfi_exact.B_aa)
            H_exact = drv_exact.build_supermatrix(nocc)
            diff_H = np.max(np.abs(H_exact - H_dense))
            ok = diff_H < 1e-8
            all_ok &= ok
            print(f"{level:6s} {basis:8s} {atom:20s}: DF (naux=norb^2 exact) build_supermatrix "
                  f"matches dense (max diff={diff_H:.2e}): {'OK' if ok else 'FAIL'}")

            poles_exact = np.linalg.eigvalsh(H_exact)
            diff_poles = np.max(np.abs(poles_exact - poles_dense))
            ok = diff_poles < 1e-7
            all_ok &= ok
            print(f"{level:6s} {basis:8s} {atom:20s}: DF (naux=norb^2 exact) pole spectrum "
                  f"matches dense (max diff={diff_poles:.2e}): {'OK' if ok else 'FAIL'}")

            # matrix-free operator: random trial-vector matvec cross-check
            aop_dense, diag_dense, dims_dense = dense.build_matrix_free_operator(nocc)
            aop_exact, diag_exact, dims_exact = drv_exact.build_matrix_free_operator(nocc)
            ok = dims_dense == dims_exact
            all_ok &= ok
            print(f"{level:6s} {basis:8s} {atom:20s}: DF matrix-free operator dims match dense: "
                  f"{'OK' if ok else 'FAIL'}")
            rng = np.random.default_rng(0)
            nH = H_dense.shape[0]
            for _ in range(3):
                z = rng.standard_normal(nH)
                Hz_dense = H_dense @ z
                Hz_aop_dense = aop_dense(z)
                Hz_aop_exact = aop_exact(z)
                d1 = np.max(np.abs(Hz_aop_dense - Hz_dense))
                d2 = np.max(np.abs(Hz_aop_exact - Hz_dense))
                ok1, ok2 = d1 < 1e-7, d2 < 1e-7
                all_ok &= ok1 & ok2
            print(f"{level:6s} {basis:8s} {atom:20s}: dense matrix-free operator == dense H@z "
                  f"(max diff={d1:.2e}): {'OK' if ok1 else 'FAIL'}")
            print(f"{level:6s} {basis:8s} {atom:20s}: DF (exact) matrix-free operator == dense H@z "
                  f"(max diff={d2:.2e}): {'OK' if ok2 else 'FAIL'}")

            # realistic RI (post-hoc DF on the same converged mf -- see
            # tests/test_mp2_density_df.py's own comment for why not a fresh
            # density_fit().run())
            mf.with_df = df.DF(mol)
            mf.with_df.build()
            dfi_ri = DFIntegrals.from_scf(mol, mf)
            drv_ri = ADCSolverRestricted.from_arrays(eps, eri_chemist, level=level, B_aa=dfi_ri.B_aa)
            H_ri = drv_ri.build_supermatrix(nocc)
            poles_ri = np.linalg.eigvalsh(H_ri)
            diff_ri = np.abs(poles_ri - poles_dense)

            ok_all = diff_ri.max() < 2e-3
            homo = eps[nocc - 1]
            valence_mask = np.abs(poles_dense - homo) < 1.0
            ok_valence = diff_ri[valence_mask].max() < 5e-4 if valence_mask.any() else True
            all_ok &= ok_all & ok_valence
            print(f"{level:6s} {basis:8s} {atom:20s}: DF (real JK-fit RI, naux={dfi_ri.naux_aa}) whole "
                  f"spectrum matches dense (max diff={diff_ri.max():.2e}): {'OK' if ok_all else 'FAIL'}")
            print(f"{level:6s} {basis:8s} {atom:20s}: DF (real JK-fit RI) valence-region (|E-HOMO|<1Ha) "
                  f"poles match dense (max diff={diff_ri[valence_mask].max() if valence_mask.any() else 0:.2e}): "
                  f"{'OK' if ok_valence else 'FAIL'}")
            mf.with_df = None

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
