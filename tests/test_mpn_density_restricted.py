"""Validates the restricted (spin-blocked) generic order-n MPn
density generator (generated_mpn_restricted/
+ mpn_density_driver_restricted.py) against MP2DensityMatrixSolverUnrestricted/
MP3DensityMatrixSolverUnrestricted (src/SingleReference/DensityMatrix/density_matrix.py --
themselves independently derived via the spin_labels mechanism, see that class's
docstring), on the alpha (aa) block for a closed-shell RHF reference, where
alpha and beta agree and off-diagonal spin blocks vanish exactly.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from tests.mpn_density_oracle import (MP2DensityMatrixSolverUnrestricted, MP3DensityMatrixSolverUnrestricted)
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import MPnDensityDriverRestricted
from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_antisymmetrized_spin_block_eri, get_two_electron_integrals_chemist


if __name__ == '__main__':
    all_ok = True

    for atom, basis in [('H 0 0 0; F 0 0 0.9', 'sto-3g'), ('H 0 0 0; Li 0 0 1.6', 'sto-3g'),
                       ('O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', '6-31g')]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc = mol.nelectron // 2

        eps_a = eps_b = get_orbital_energies(mf, representation='spatial')
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
        f_aa = np.diag(eps_a)

        ref2 = MP2DensityMatrixSolverUnrestricted(eps_a, eps_b, g_aaaa, g_bbbb, g_abab, nocc, nocc)
        oo_a2, oo_b2, ov_a2, ov_b2, vv_a2, vv_b2 = ref2.compute_blocks()

        ref3 = MP3DensityMatrixSolverUnrestricted(eps_a, eps_b, g_aaaa, g_bbbb, g_abab, nocc, nocc)
        oo_a3, oo_b3, ov_a3, ov_b3, vv_a3, vv_b3 = ref3.compute_gamma3_blocks()

        gen_driver = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc)
        gen_oo2, gen_ov2, gen_vv2 = gen_driver.compute_delta_gamma2()
        gen_oo3, gen_ov3, gen_vv3 = gen_driver.compute_delta_gamma3()

        diff2 = max(np.max(np.abs(gen_oo2 - oo_a2)), np.max(np.abs(gen_ov2 - ov_a2)),
                    np.max(np.abs(gen_vv2 - vv_a2)))
        ok2 = diff2 < 1e-8
        all_ok &= ok2
        print(f"{basis:8s} {atom:20s}: generic restricted MP2 (n=2) matches "
              f"Unrestricted-oracle alpha block (max diff={diff2:.2e}): {'OK' if ok2 else 'FAIL'}")

        diff3 = max(np.max(np.abs(gen_oo3 - oo_a3)), np.max(np.abs(gen_ov3 - ov_a3)),
                    np.max(np.abs(gen_vv3 - vv_a3)))
        ok3 = diff3 < 1e-8
        all_ok &= ok3
        print(f"{basis:8s} {atom:20s}: generic restricted MP3 (n=3) matches "
              f"Unrestricted-oracle alpha block (max diff={diff3:.2e}): {'OK' if ok3 else 'FAIL'}")

        # Laplace-accelerated path (mpn_density_driver_restricted.py's own
        # port of density_matrix.py's _laplace_aaa_contribution) against the
        # SAME Laplace-accelerated oracle -- both at ntau=8.
        oo_a3_lap, oo_b3_lap, ov_a3_lap, ov_b3_lap, vv_a3_lap, vv_b3_lap = ref3.compute_gamma3_blocks(laplace_ntau=8)
        gen_oo3_lap, gen_ov3_lap, gen_vv3_lap = gen_driver.compute_delta_gamma3(laplace_ntau=8)
        diff3_lap = max(np.max(np.abs(gen_oo3_lap - oo_a3_lap)), np.max(np.abs(gen_ov3_lap - ov_a3_lap)),
                        np.max(np.abs(gen_vv3_lap - vv_a3_lap)))
        ok3_lap = diff3_lap < 1e-8
        all_ok &= ok3_lap
        print(f"{basis:8s} {atom:20s}: generic restricted MP3 Laplace (ntau=8) matches "
              f"Unrestricted-oracle Laplace(ntau=8) (max diff={diff3_lap:.2e}): {'OK' if ok3_lap else 'FAIL'}")

    print("ALL PASSED" if all_ok else "FAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
