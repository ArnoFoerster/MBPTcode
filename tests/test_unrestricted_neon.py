import os
import sys
import numpy as np
from pyscf import gto, scf, df

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients

def run_tests():
    # Setup Neon atom
    mol = gto.Mole()
    mol.atom = "Ne 0 0 0"
    mol.basis = "cc-pvdz"
    mol.verbose = 0
    mol.build()

    # Restricted starting point
    mf_rhf = scf.RHF(mol).density_fit()
    mf_rhf.with_df.auxbasis = df.make_auxbasis(mol)
    mf_rhf.run()

    # Unrestricted starting point (with spin-symmetry broken initially, though it should find RHF solution)
    mf_uhf = scf.UHF(mol).density_fit()
    mf_uhf.with_df.auxbasis = df.make_auxbasis(mol)
    mf_uhf.run()

    print("="*80)
    print(" VERIFYING UNRESTRICTED SPIN MODE: NEON ATOM (DENSITY FITTING) ")
    print("="*80)

    methods = ["GW@RPA", "GW@BSE", "GWGammaInf", "PSD2"]

    for m in methods:
        print(f"Running {m} with Density Fitting...")
        val_res = calc_qp_energy(mf_rhf, selfenergy=m, polarizability="BSE" if m != "GW@RPA" else "RPA", df=True)
        val_unres = calc_qp_energy(mf_uhf, selfenergy=m, polarizability="BSE" if m != "GW@RPA" else "RPA", df=True, spin_channel='alpha')
        diff = abs(val_res - val_unres)
        print(f"  Restricted:   {val_res:12.6f} eV")
        print(f"  Unrestricted: {val_unres:12.6f} eV")
        print(f"  Difference:   {diff:12.3e} eV")
        threshold = 1.0e-5
        assert diff < threshold, f"Unrestricted spin mode differs for {m} (df=True): {diff:.3e} eV (threshold={threshold})"

    print("\n" + "="*80)
    print(" VERIFYING UNRESTRICTED SPIN MODE: NEON ATOM (FULL ERI) ")
    print("="*80)

    # Restricted starting point without DF
    mf_rhf_full = scf.RHF(mol)
    mf_rhf_full.run()

    # Unrestricted starting point without DF
    mf_uhf_full = scf.UHF(mol)
    mf_uhf_full.run()

    for m in methods:
        print(f"Running {m} with Full ERI...")
        val_res = calc_qp_energy(mf_rhf_full, selfenergy=m, polarizability="BSE" if m != "GW@RPA" else "RPA", df=False)
        val_unres = calc_qp_energy(mf_uhf_full, selfenergy=m, polarizability="BSE" if m != "GW@RPA" else "RPA", df=False, spin_channel='alpha')
        diff = abs(val_res - val_unres)
        print(f"  Restricted:   {val_res:12.6f} eV")
        print(f"  Unrestricted: {val_unres:12.6f} eV")
        print(f"  Difference:   {diff:12.3e} eV")
        threshold = 1.0e-5
        assert diff < threshold, f"Unrestricted spin mode differs for {m} (df=False): {diff:.3e} eV (threshold={threshold})"

    print("\n" + "="*80)
    print(" VERIFYING SPIN-FLIP CASIDA MATRICES ")
    print("="*80)

    eps = get_orbital_energies(mf_uhf, representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, mf_uhf, representation='spatial')
    nocc = mf_uhf.nelec

    lr_solver = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='unrestricted')
    w_aux = lr_solver.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]

    print("Building spin-flip Casida matrices for channel 'ab'...")
    A_sf_ab, B_sf_ab = lr_solver.build_spin_flip_casida_matrices(nocc, lBSE=True, W_aux=w_aux, channel='ab')
    print(f"  A shape: {A_sf_ab.shape}, B shape: {B_sf_ab.shape}")
    omega_ab, X_ab, Y_ab = CasidaSolver(A_sf_ab, B_sf_ab).solve()
    print(f"  Lowest excitation energy ('ab'): {omega_ab[0] * 27.2114:.6f} eV")

    print("\nBuilding spin-flip Casida matrices for channel 'ba'...")
    A_sf_ba, B_sf_ba = lr_solver.build_spin_flip_casida_matrices(nocc, lBSE=True, W_aux=w_aux, channel='ba')
    print(f"  A shape: {A_sf_ba.shape}, B shape: {B_sf_ba.shape}")
    omega_ba, X_ba, Y_ba = CasidaSolver(A_sf_ba, B_sf_ba).solve()
    print(f"  Lowest excitation energy ('ba'): {omega_ba[0] * 27.2114:.6f} eV")

    # Since Ne is closed shell, channels ab and ba should have identical eigenvalues
    diff_sf = np.linalg.norm(omega_ab - omega_ba)
    print(f"\n  Spin-flip channel difference: {diff_sf:.3e}")
    assert diff_sf < 1e-5, f"Spin-flip channels ab and ba differ: {diff_sf:.3e}"

    print("\nALL UNRESTRICTED SPIN MODE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
