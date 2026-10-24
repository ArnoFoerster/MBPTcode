import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df
from pyscf.gw.rpa import RPA as PyscfRPA

from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients, \
    get_two_electron_integrals_chemist
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import rpa_correlation_energy_casida, \
    rpa_correlation_energy_imaginary_axis

if __name__ == '__main__':
    all_ok = True

    # --- DF route: Casida-trace vs. imaginary-axis quadrature vs. pyscf's own
    # independent RPA implementation (pyscf.gw.rpa, N^4 dRPA via imaginary-frequency
    # AC quadrature -- same equations 26-28, different code base entirely). ---
    cases = [
        ('HF/6-31g', 'H 0 0 0; F 0 0 0.9', '6-31g'),
        ('Ne/cc-pvdz', 'Ne 0 0 0', 'cc-pvdz'),
    ]

    for label, atom, basis in cases:
        mol = gto.M(atom=atom, basis=basis)
        mf = scf.RHF(mol).density_fit()
        mf.with_df.auxbasis = df.make_auxbasis(mol)
        mf.run()
        nocc = mol.nelectron // 2

        eps = get_orbital_energies(mf, representation='spatial')
        df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted')

        e_casida = rpa_correlation_energy_casida(lr, nocc)
        e_ac = rpa_correlation_energy_imaginary_axis(lr, nocc, grid='minimax', numerics=Numerics(quadrature={'nfreq_rpa': 20}))

        rpa_ref = PyscfRPA(mf)
        rpa_ref.verbose = 0
        e_pyscf = rpa_ref.kernel(nw=40, x0=0.5)

        diff_casida_ac = abs(e_casida - e_ac)
        diff_casida_pyscf = abs(e_casida - e_pyscf)
        diff_ac_pyscf = abs(e_ac - e_pyscf)

        ok = diff_casida_ac < 1e-6 and diff_casida_pyscf < 1e-5 and diff_ac_pyscf < 1e-5
        all_ok &= ok
        print(f"{label:12s} Casida={e_casida:.8f}  AC(minimax20)={e_ac:.8f}  "
              f"pyscf={e_pyscf:.8f} Ha  "
              f"(Casida-AC={diff_casida_ac:.2e}, Casida-pyscf={diff_casida_pyscf:.2e}, "
              f"AC-pyscf={diff_ac_pyscf:.2e})  {'OK' if ok else 'FAIL'}")

    # --- gauss_legendre grid should agree with minimax and with the Casida route too ---
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g')
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted')

    e_casida = rpa_correlation_energy_casida(lr, nocc)
    e_ac_gl = rpa_correlation_energy_imaginary_axis(lr, nocc, grid='gauss_legendre', numerics=Numerics(quadrature={'nfreq_rpa': 30}))
    ok = abs(e_casida - e_ac_gl) < 1e-5
    all_ok &= ok
    print(f"HF/6-31g gauss_legendre(nfreq=30) vs Casida: AC={e_ac_gl:.8f}  "
          f"Casida={e_casida:.8f}  diff={abs(e_casida-e_ac_gl):.2e}  {'OK' if ok else 'FAIL'}")

    # --- full-ERI (non-DF) Casida-trace route should agree closely with the DF route ---
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    lr_full = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted')
    e_casida_full = rpa_correlation_energy_casida(lr_full, nocc)
    ok = abs(e_casida_full - e_casida) < 1e-4
    all_ok &= ok
    print(f"HF/6-31g full-ERI vs DF Casida-trace: full={e_casida_full:.8f}  "
          f"DF={e_casida:.8f}  diff={abs(e_casida_full-e_casida):.2e}  {'OK' if ok else 'FAIL'}")

    # --- sanity: RPA correlation energy must be negative ---
    ok = e_casida < 0.0 and e_ac < 0.0
    all_ok &= ok
    print(f"RPA correlation energy is negative: {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
