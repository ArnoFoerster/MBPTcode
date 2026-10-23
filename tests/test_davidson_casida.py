import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.Base.numerics.numerics import numerics_of
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson

if __name__ == '__main__':
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g')
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()

    eps = get_orbital_energies(mf, representation='spatial')
    nocc = mol.nelectron // 2
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted', eta=1e-3)
    w_aux = lr.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]

    nroots = 4
    all_ok = True
    for mode, kwargs in [('RPA', {}), ('TDHF', {}), ('BSE', {'W_aux': w_aux})]:
        lBSE = mode != 'RPA'
        A_dense, B_dense = lr.build_casida_matrices(nocc, lBSE=lBSE, W_aux=kwargs.get('W_aux'))
        omega_dense = np.sort(CasidaSolver(A_dense, B_dense).solve()[0])[:nroots]

        omega_dav, X_dav, Y_dav = solve_casida_davidson(lr, nocc, numerics=numerics_of({'nroots': nroots}), polarizability=mode, **kwargs)
        omega_dav = np.sort(omega_dav)

        diff = np.max(np.abs(omega_dense - omega_dav))
        ok = diff < 1e-6
        all_ok &= ok
        print(f"{mode:5s}: dense={omega_dense}  davidson={omega_dav}  maxdiff={diff:.2e}  {'OK' if ok else 'FAIL'}")

    se = SelfEnergySolver(eps, df_coeff=df_coeff, spin_mode='restricted', eta=1e-3)
    p_state = nocc - 1
    omega, X, Y = solve_casida_davidson(lr, nocc, numerics=numerics_of({'nroots': nroots}), polarizability='BSE', W_aux=w_aux)
    chi_a = se.get_chi_a(nocc, X, Y, p_state=p_state)
    compat_ok = chi_a.shape == (nroots, len(eps)) and np.all(np.isfinite(chi_a))
    print(f"AmplitudeGenerator compatibility: {'OK' if compat_ok else 'FAIL'}")
    all_ok &= compat_ok

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
