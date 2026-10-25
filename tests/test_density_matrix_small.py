import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.SingleReference.DensityMatrix.density_matrix import compute_gw_density_matrix, GWDensityMatrixSolver
from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients

if __name__ == '__main__':
    all_ok = True

    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g')
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()
    S = mol.intor('int1e_ovlp')
    nelec = mol.nelectron
    nocc = mol.nelectron // 2

    for polarizability in ['RPA', 'BSE']:
        for relax in [False, True]:
            dm = compute_gw_density_matrix(mf, mol, polarizability=polarizability, df=True, relax=relax)
            n = np.trace(dm @ S)
            ok = abs(n - nelec) < 1e-6
            all_ok &= ok
            print(f"polarizability={polarizability:4s} relax={relax!s:5s}: N={n:.8f} (expect {nelec}) {'OK' if ok else 'FAIL'}")

    # Charge conservation at the block level: trace(oo) = -trace(vv) is exact
    # by construction (see GWDensityMatrixSolver.compute_unrelaxed_blocks),
    # independent of relaxation.
    eps = get_orbital_energies(mf, representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    solver = GWDensityMatrixSolver(eps, df_coeff=df_coeff, eta=1e-3)
    dgamma_oo, dgamma_ov, dgamma_vv = solver.compute_unrelaxed_blocks(nocc, mf, mol, polarizability='RPA')
    ok = abs(np.trace(dgamma_oo) + np.trace(dgamma_vv)) < 1e-8
    all_ok &= ok
    print(f"trace(dgamma_oo) == -trace(dgamma_vv): {'OK' if ok else 'FAIL'} "
          f"(sum={np.trace(dgamma_oo) + np.trace(dgamma_vv):.2e})")

    # relax=False vs relax=True must differ only in the ov block (oo/vv blocks
    # come from the same compute_unrelaxed_blocks call either way).
    dm_lin = compute_gw_density_matrix(mf, mol, polarizability='RPA', df=True, relax=False)
    dm_relaxed = compute_gw_density_matrix(mf, mol, polarizability='RPA', df=True, relax=True)
    ok = not np.allclose(dm_lin, dm_relaxed, atol=1e-8)
    all_ok &= ok
    print(f"relaxed density differs from linearized (CPHF actually ran): {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
