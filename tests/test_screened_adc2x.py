"""Static-RPA screening of the C^(1) block (screened MCDE / screened ADC(2)-x,
Romaniello & Berger arXiv:2603.27329).

Checks, on H2O/cc-pVDZ:
  1. bare dense == bare matrix-free (the screening hook does not perturb the
     unscreened result)
  2. dense W_chemist screening runs and shifts the first IP
  3. DF matrix-free W_aux screening reproduces the dense W_chemist answer
     (the two representations of the same static W)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf.pyscf_interface import (DFIntegrals, get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.SingleReference.LinearResponse.linear_response import (
    static_screened_coulomb_aux, static_screened_coulomb_chemist)
from src.SingleReference.ADC import ADCSolver

HARTREE_EV = 27.211386


def first_ip(e, Z, z_min=0.1):
    """Highest-lying occupied-side pole carrying real 1p weight. Dense solves
    return every pole, so the lowest eigenvalue is not the first IP."""
    m = (e < 0) & (Z > z_min)
    k = np.argmax(e[m])
    return -e[m][k] * HARTREE_EV, Z[m][k]


def main():
    all_ok = True
    mol = gto.M(atom='O 0 0 0; H 0 0 0.958; H 0.926 0 -0.240',
                basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

    e, Z = ADCSolver(mf, level='adc2x', matrix_free=False).solve()
    ip_dense, _ = first_ip(e, Z)
    e, Z = ADCSolver(mf, level='adc2x').solve()
    ip_matfree = -e[0] * HARTREE_EV
    d = abs(ip_dense - ip_matfree)
    ok = d < 1e-6
    all_ok &= ok
    print(f"bare ADC(2)-x dense {ip_dense:.4f} eV == matrix-free {ip_matfree:.4f} eV "
          f"(diff={d:.2e}): {'OK' if ok else 'FAIL'}")

    W = static_screened_coulomb_chemist(eps, eri, nocc)
    e, Z = ADCSolver(mf, level='adc2x', matrix_free=False,
                     screening={'W_chemist': W}).solve()
    ip_scr, z_scr = first_ip(e, Z)
    ok = 0.05 < abs(ip_scr - ip_dense) < 2.0 and z_scr > 0.5
    all_ok &= ok
    print(f"dense W_chemist screening shifts the IP {ip_dense:.4f} -> {ip_scr:.4f} eV "
          f"(Z={z_scr:.3f}): {'OK' if ok else 'FAIL'}")

    mfd = scf.RHF(mol).density_fit().run()
    epsd = get_orbital_energies(mfd, representation='spatial')
    B = DFIntegrals.from_scf(mol, mfd).B_aa
    W_aux = static_screened_coulomb_aux(epsd, B, nocc)
    e, Z = ADCSolver(mfd, level='adc2x', df=True,
                     screening={'W_aux': W_aux}).solve()
    ip_aux = -e[0] * HARTREE_EV
    d = abs(ip_aux - ip_scr)
    ok = d < 5e-3
    all_ok &= ok
    print(f"DF W_aux screening {ip_aux:.4f} eV matches dense W_chemist {ip_scr:.4f} eV "
          f"(diff={d:.2e}): {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
