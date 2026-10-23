"""Validates MP4's T4^(2)_aaaaaaaa Laplace fusion
(generate_mpn_laplace.py -> generated_mpn_restricted/
mp4_laplace_restricted.py, wired into mpn_density_driver_restricted.py's
compute_delta_gamma4(laplace_ntau=...)) against the exact (materialized-T4)
result -- see that module's docstring for scope (only t2_3_aaaa_numerator/
t3_3_aaaaaa_numerator are de-materialized; m4_*_22/overlap4 still need
t4_2_aaaaaaaa built, since T4^(2) appears in both bra and ket roles there).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import MPnDensityDriverRestricted
from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_antisymmetrized_spin_block_eri


if __name__ == '__main__':
    all_ok = True
    for atom, basis in [('H 0 0 0; Li 0 0 1.6', 'sto-3g'),
                        ('O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', '6-31g')]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc = mol.nelectron // 2
        eps_a = get_orbital_energies(mf, representation='spatial')
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
        f_aa = np.diag(eps_a)
        d = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc)

        oo_ref, ov_ref, vv_ref = d.compute_delta_gamma4(laplace_ntau=None)
        oo_lap, ov_lap, vv_lap = d.compute_delta_gamma4(laplace_ntau=8)

        diff = max(np.max(np.abs(oo_ref - oo_lap)), np.max(np.abs(ov_ref - ov_lap)),
                  np.max(np.abs(vv_ref - vv_lap)))
        ok = diff < 1e-9
        all_ok &= ok
        print(f"{basis:8s} {atom:35s}: MP4 Laplace(ntau=8) vs exact (max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    print("ALL PASSED" if all_ok else "FAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
