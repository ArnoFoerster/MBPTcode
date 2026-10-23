import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver

if __name__ == '__main__':
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g')
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()

    eps = get_orbital_energies(mf, representation='spatial')
    nocc = mol.nelectron // 2
    norb = len(eps)
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    eta = 1e-3

    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted', eta=eta)
    se = SelfEnergySolver(eps, df_coeff=df_coeff, spin_mode='restricted', eta=eta)

    A, B = lr.build_casida_matrices(nocc, lBSE=False)
    omega_rpa, X_rpa, Y_rpa = CasidaSolver(A, B).solve()
    chi_a_full = se.get_chi_a(nocc, X_rpa, Y_rpa, p_state=None)

    rng = np.random.default_rng(0)
    freq = eps + 0.01 * rng.standard_normal(norb)  # arbitrary, not on-shell

    batch = se.calculate_self_energy_diagonal_batch(freq, nocc, omega_rpa, chi_a_full, vertex_mode='GW')

    scalar = np.array([
        se.calculate_self_energy(p, freq[p], nocc, omega_rpa, chi_a_full, vertex_mode='GW')
        for p in range(norb)
    ])

    diff = np.max(np.abs(batch - scalar))
    ok = diff < 1e-12
    print(f"calculate_self_energy_diagonal_batch vs calculate_self_energy: maxdiff={diff:.2e}  {'OK' if ok else 'FAIL'}")
    print("\nALL PASSED" if ok else "\nFAILURES DETECTED")
    sys.exit(0 if ok else 1)
