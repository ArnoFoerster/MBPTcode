import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.Base.numerics import numerics_of

#: The residual the solves run to: benzene's matrix-free Casida breaks down
#: in round-off at the 1e-8 Ha a production solve converges to.
CONV_TOL = 1e-5
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson

BENZENE_GEOM = """
C  0.000  1.396  0.000
C  1.209  0.698  0.000
C  1.209 -0.698  0.000
C  0.000 -1.396  0.000
C -1.209 -0.698  0.000
C -1.209  0.698  0.000
H  0.000  2.479  0.000
H  2.147  1.240  0.000
H  2.147 -1.240  0.000
H  0.000 -2.479  0.000
H -2.147 -1.240  0.000
H -2.147  1.240  0.000
"""

if __name__ == '__main__':
    basis = sys.argv[1] if len(sys.argv) > 1 else '6-31g'
    nroots = 5

    mol = gto.M(atom=BENZENE_GEOM, basis=basis, symmetry=True)
    print(f"point group: groupname={mol.groupname}  topgroup={mol.topgroup}")
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()

    eps = get_orbital_energies(mf, representation='spatial')
    nocc = mol.nelectron // 2
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted', eta=1e-3)
    print(f"basis={basis}  nocc={nocc}  nvirt={len(eps) - nocc}  n_pair={nocc * (len(eps) - nocc)}")

    w_aux = lr.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]

    t0 = time.perf_counter()
    A_bse, B_bse = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w_aux)
    omega_dense = np.sort(CasidaSolver(A_bse, B_bse).solve()[0])[:nroots]
    t_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    omega_dav, X_dav, Y_dav = solve_casida_davidson(lr, nocc, numerics=numerics_of({'nroots': nroots, 'bse_conv_tol': CONV_TOL}), polarizability='BSE', W_aux=w_aux)
    omega_dav = np.sort(omega_dav)
    t_dav = time.perf_counter() - t0

    t0 = time.perf_counter()
    omega_dav_sym, X_dav_sym, Y_dav_sym = solve_casida_davidson(
        lr, nocc, numerics=numerics_of({'nroots': nroots, 'bse_conv_tol': CONV_TOL}), polarizability='BSE', W_aux=w_aux, orbsym=mf.orbsym)
    omega_dav_sym = np.sort(omega_dav_sym)
    t_dav_sym = time.perf_counter() - t0

    diff = np.max(np.abs(omega_dense - omega_dav))
    diff_sym = np.max(np.abs(omega_dense - omega_dav_sym))
    ok = diff < 1e-6
    ok_sym = diff_sym < 1e-6

    print(f"dense           (t={t_dense:8.2f}s): {omega_dense}")
    print(f"davidson        (t={t_dav:8.2f}s): {omega_dav}")
    print(f"davidson+orbsym (t={t_dav_sym:8.2f}s): {omega_dav_sym}")
    print(f"maxdiff={diff:.2e}  {'OK' if ok else 'FAIL'}   "
          f"maxdiff(orbsym)={diff_sym:.2e}  {'OK' if ok_sym else 'FAIL'}")
    print("\nALL PASSED" if ok and ok_sym else "\nFAILURES DETECTED")
    sys.exit(0 if ok and ok_sym else 1)
