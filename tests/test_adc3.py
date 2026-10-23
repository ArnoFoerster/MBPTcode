"""
Regression tests for the spin-orbital Dyson IP/EA-ADC(3) supermatrix (src/SingleReference/ADC3.py),
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC import (
    ADCSolver, ADCSolverUnrestricted, build_mp2_static_correction,
    ADCSolverRestricted, build_mp2_static_correction_restricted)
from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_two_electron_integrals_chemist, get_antisymmetrized_spin_eri

HARTREE_TO_EV = 27.211386245988

if __name__ == '__main__':
    all_ok = True

    for atom, basis, label in [
        ('H 0 0 0; H 0 0 0.74', 'sto-3g', 'H2/sto-3g'),
        ('O 0 0 0; H 0 0 0.96; H 0.96 0 0', 'sto-3g', 'H2O/sto-3g'),
    ]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc = mol.nelectron // 2
        eps_spin = get_orbital_energies(mf, representation='spin')
        eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
        g = get_antisymmetrized_spin_eri(eri)

        solver = ADCSolverUnrestricted.from_arrays(eps_spin, g)
        d = solver.dimensions(2 * nocc)
        H = solver.build_supermatrix(2 * nocc)

        # --- 1. Exact symmetry ---
        asym = np.max(np.abs(H - H.T))
        ok = asym < 1e-10
        all_ok &= ok
        print(f"{label:12s}: H exactly symmetric (max|H-H^T|={asym:.2e}): {'OK' if ok else 'FAIL'}")

        # --- 2. Dimension formula ---
        norb, nvirt = d['norb'], d['norb'] - 2 * nocc
        n2h1p_expect = (2 * nocc) * (2 * nocc - 1) // 2 * nvirt
        n2p1h_expect = nvirt * (nvirt - 1) // 2 * (2 * nocc)
        nH_expect = norb + n2h1p_expect + n2p1h_expect
        ok = d['nH'] == nH_expect
        all_ok &= ok
        print(f"{label:12s}: nH matches combinatorial formula ({d['nH']} == {nH_expect}): {'OK' if ok else 'FAIL'}")

        # --- 3/4. Sum rule and Z bounds ---
        eGF, Z, Reigv = solver.solve_dense(2 * nocc)
        ok = abs(np.sum(Z) - norb) < 1e-8
        all_ok &= ok
        print(f"{label:12s}: sum(Z) == norb ({np.sum(Z):.8f} == {norb}): {'OK' if ok else 'FAIL'}")

        ok = np.all(Z >= -1e-10) and np.all(Z <= 1 + 1e-10)
        all_ok &= ok
        print(f"{label:12s}: all Z in [0,1] (min={Z.min():.2e}, max={Z.max():.6f}): {'OK' if ok else 'FAIL'}")

        print()

    # --- 5. Core-orbital (weakly correlated) Koopmans check on water ---
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.96 0 0', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    eGF, Z = ADCSolver(mf, mol, spin='spinorbital', matrix_free=False).solve()
    eps_spin = get_orbital_energies(mf, representation='spin')

    core_pole = np.argmin(np.abs(eGF - eps_spin[0]))
    ok = Z[core_pole] > 0.5 and abs(eGF[core_pole] - eps_spin[0]) < 0.5
    all_ok &= ok
    print(f"H2O/sto-3g core pole near Koopmans (eGF={eGF[core_pole]:.6f} vs eps_1s={eps_spin[0]:.6f}, "
          f"Z={Z[core_pole]:.6f}): {'OK' if ok else 'FAIL'}\n")

    # --- 6. Decisive check: Ne/aug-cc-pVDZ IP(2p) vs externally supplied reference ---
    mol = gto.M(atom='Ne 0 0 0', basis='aug-cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2

    eGF, Z = ADCSolver(mf, mol, spin='spinorbital', matrix_free=False).solve()
    mask = Z > 0.5
    ip_2p_eV = -np.sort(eGF[mask & (eGF < 0)])[-1] * HARTREE_TO_EV
    ref_ip_2p_eV = 21.403
    ok = abs(ip_2p_eV - ref_ip_2p_eV) < 0.01
    all_ok &= ok
    print(f"Ne/aug-cc-pVDZ Dyson-ADC(3) IP(2p) matches reference "
          f"({ip_2p_eV:.4f} eV vs {ref_ip_2p_eV} eV): {'OK' if ok else 'FAIL'}\n")

    # --- 7. MP2-density static correction: symmetry and structural sum rules preserved ---
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.96 0 0', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    eps_spin = get_orbital_energies(mf, representation='spin')
    g = get_antisymmetrized_spin_eri(get_two_electron_integrals_chemist(mol, mf, representation='spatial'))
    solver = ADCSolverUnrestricted.from_arrays(eps_spin, g)

    for relax, rlabel in [(False, 'unrelaxed'), (True, 'relaxed')]:
        dF = build_mp2_static_correction(mf, mol, nocc, relax=relax)
        ok = np.max(np.abs(dF - dF.T)) < 1e-8
        all_ok &= ok
        print(f"H2O/sto-3g static correction ({rlabel}) is symmetric "
              f"(max|dF-dF.T|={np.max(np.abs(dF - dF.T)):.2e}): {'OK' if ok else 'FAIL'}")

        eGF_s, Z_s, _ = solver.solve_dense(2 * nocc, static_correction=dF)
        H_s = solver.build_supermatrix(2 * nocc, static_correction=dF)
        ok = np.max(np.abs(H_s - H_s.T)) < 1e-8 and abs(np.sum(Z_s) - len(eps_spin)) < 1e-8
        all_ok &= ok
        print(f"H2O/sto-3g H with static correction ({rlabel}) stays symmetric and sum(Z)==norb "
              f"(sum(Z)={np.sum(Z_s):.8f}): {'OK' if ok else 'FAIL'}")

    # --- 8. ADCSolverRestricted Test ---
    print("\n--- Testing ADCSolverRestricted ---")
    mol = gto.M(atom='Ne 0 0 0', basis='aug-cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2

    eGF_r, Z_r = ADCSolverRestricted(mf, mol, matrix_free=False).solve()
    mask = Z_r > 0.5
    ip_2p_eV_r = -np.sort(eGF_r[mask & (eGF_r < 0)])[-1] * HARTREE_TO_EV
    ref_ip_2p_eV = 21.403
    ok = abs(ip_2p_eV_r - ref_ip_2p_eV) < 0.01
    all_ok &= ok
    print(f"Ne/aug-cc-pVDZ Dyson-ADC(3) IP(2p) matches reference (restricted) "
          f"({ip_2p_eV_r:.4f} eV vs {ref_ip_2p_eV} eV): {'OK' if ok else 'FAIL'}\n")

    # --- 9. Restricted MP2 static correction: symmetry, sum rules, and cross-check
    #     against the (already-validated) spin-orbital build_mp2_static_correction ---
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.96 0 0', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    solver_r = ADCSolverRestricted.from_arrays(eps, eri_chemist)

    for relax, rlabel in [(False, 'unrelaxed'), (True, 'relaxed')]:
        dF_r = build_mp2_static_correction_restricted(mf, mol, nocc, relax=relax)
        ok = np.max(np.abs(dF_r - dF_r.T)) < 1e-8
        all_ok &= ok
        print(f"H2O/sto-3g restricted static correction ({rlabel}) is symmetric "
              f"(max|dF-dF.T|={np.max(np.abs(dF_r - dF_r.T)):.2e}): {'OK' if ok else 'FAIL'}")

        dF_so = build_mp2_static_correction(mf, mol, nocc, relax=relax)
        d_cross = np.max(np.abs(dF_r - dF_so[0::2, 0::2]))
        ok = d_cross < 1e-10
        all_ok &= ok
        print(f"H2O/sto-3g restricted static correction ({rlabel}) matches the spin-orbital "
              f"alpha-alpha block (max|diff|={d_cross:.2e}): {'OK' if ok else 'FAIL'}")

        eGF_rs, Z_rs, _ = solver_r.solve_dense(nocc, static_correction=dF_r)
        H_rs = solver_r.build_supermatrix(nocc, static_correction=dF_r)
        ok = np.max(np.abs(H_rs - H_rs.T)) < 1e-8 and abs(np.sum(Z_rs) - len(eps)) < 1e-8
        all_ok &= ok
        print(f"H2O/sto-3g restricted H with static correction ({rlabel}) stays symmetric and "
              f"sum(Z)==norb (sum(Z)={np.sum(Z_rs):.8f}): {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
