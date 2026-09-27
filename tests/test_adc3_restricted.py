"""Verify that ADCSolverRestricted exactly reproduces ADCSolver (spin-orbital)
eigenvalues: every eigenvalue of the restricted ADC(3) supermatrix of
water/STO-3G appears in the spin-orbital one.

Run as a script (`python tests/test_adc3_restricted.py`) or under pytest.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf
from src.Base.pyscf_interface import get_orbital_energies, get_two_electron_integrals_chemist, get_antisymmetrized_spin_eri
from src.SingleReference.ADC import ADCSolverUnrestricted, ADCSolverRestricted

#: Ha
TOL = 1e-10


def test_restricted_matches_spin_orbital():
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

    # Restricted (loop-based)
    solver_r = ADCSolverRestricted.from_arrays(eps, eri_chemist)
    H_r = solver_r.build_supermatrix(nocc)
    evals_r = np.sort(np.linalg.eigvalsh(H_r))

    # Unrestricted (spin-orbital)
    eps_spin = np.zeros(2 * len(eps))
    eps_spin[0::2] = eps; eps_spin[1::2] = eps
    g_anti = get_antisymmetrized_spin_eri(eri_chemist)
    solver_u = ADCSolverUnrestricted.from_arrays(eps_spin, g_anti)
    H_u = solver_u.build_supermatrix(2 * nocc)
    evals_u = np.sort(np.linalg.eigvalsh(H_u))

    # Match: every restricted eigenvalue must appear in the unrestricted set
    max_diff = 0.0
    for er in evals_r:
        idx = np.argmin(np.abs(evals_u - er))
        max_diff = max(max_diff, abs(evals_u[idx] - er))

    print(f"Water/STO-3G:  restricted dim={H_r.shape[0]}, unrestricted dim={H_u.shape[0]}")
    print(f"  Max eigenvalue difference: {max_diff:.2e}")
    assert max_diff < TOL, f"max diff = {max_diff}"


if __name__ == '__main__':
    test_restricted_matches_spin_orbital()
    print('  PASS')
    print('\nALL PASSED')
