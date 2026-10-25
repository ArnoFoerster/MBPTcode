"""Validates the unrestricted (spin-blocked, separate alpha/beta) generic
MPn density pipeline (generated_mpn_unrestricted/ + mpn_density_driver_unrestricted.py) against
MP2DensityMatrixSolverUnrestricted/MP3DensityMatrixSolverUnrestricted
(src/SingleReference/DensityMatrix/density_matrix.py -- FCI-validated test
oracles, kept for exactly this purpose), on ALL six per-spin blocks:

- closed-shell RHF fed as UHF (LiH/sto-3g -- the molecule whose nontrivial
  virtual mixing historically exposed the mixed-block l-tag axis convention,
  see mpn_density_driver_restricted.py's module docstring; plus H2O/6-31g);
- genuine open-shell UHF (OH radical doublet, NH2 radical doublet) with
  nocc_a != nocc_b and separate alpha/beta orbitals.

MP3 oracle runs with laplace_ntau=None (exact double-loop reference).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from tests.mpn_density_oracle import (MP2DensityMatrixSolverUnrestricted, MP3DensityMatrixSolverUnrestricted)
from src.SingleReference.DensityMatrix.mpn_density_driver_unrestricted import MPnDensityDriverUnrestricted
from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_antisymmetrized_spin_block_eri


CASES = [
    ('H 0 0 0; Li 0 0 1.6', 'sto-3g', 0, 'LiH (RHF-as-UHF)'),
    ('O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', '6-31g', 0, 'H2O (RHF-as-UHF)'),
    ('O 0 0 0; H 0 0 0.97', '6-31g', 1, 'OH radical'),
    ('N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6', 'sto-3g', 1, 'NH2 radical'),
]

if __name__ == '__main__':
    all_ok = True

    for atom, basis, spin, label in CASES:
        mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
        mf = scf.UHF(mol).run()
        nocc_a, nocc_b = mf.nelec

        eps = get_orbital_energies(mf, representation='spatial')
        if isinstance(eps, tuple):
            eps_a, eps_b = eps
        else:
            eps_a = eps_b = eps
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)

        ref2 = MP2DensityMatrixSolverUnrestricted(eps_a, eps_b, g_aaaa, g_bbbb, g_abab, nocc_a, nocc_b)
        ref2_blocks = ref2.compute_blocks()
        ref3 = MP3DensityMatrixSolverUnrestricted(eps_a, eps_b, g_aaaa, g_bbbb, g_abab, nocc_a, nocc_b)
        ref3_blocks = ref3.compute_gamma3_blocks(laplace_ntau=None)

        driver = MPnDensityDriverUnrestricted(np.diag(eps_a), np.diag(eps_b),
                                              g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b)
        gen2_blocks = driver.compute_delta_gamma2()
        gen3_blocks = driver.compute_delta_gamma3(laplace_ntau=None)

        names = ('oo_a', 'oo_b', 'ov_a', 'ov_b', 'vv_a', 'vv_b')
        diff2 = max(np.max(np.abs(g - r)) for g, r in zip(gen2_blocks, ref2_blocks))
        ok2 = diff2 < 1e-9
        all_ok &= ok2
        print(f"{label:20s} (no_a={nocc_a}, no_b={nocc_b}): generic unrestricted MP2 matches "
              f"oracle all 6 blocks (max diff={diff2:.2e}): {'OK' if ok2 else 'FAIL'}")
        if not ok2:
            for n, g, r in zip(names, gen2_blocks, ref2_blocks):
                print(f"    {n}: {np.max(np.abs(g - r)):.2e}")

        diff3 = max(np.max(np.abs(g - r)) for g, r in zip(gen3_blocks, ref3_blocks))
        ok3 = diff3 < 1e-9
        all_ok &= ok3
        print(f"{label:20s} (no_a={nocc_a}, no_b={nocc_b}): generic unrestricted MP3 (n=3, exact) matches "
              f"oracle all 6 blocks (max diff={diff3:.2e}): {'OK' if ok3 else 'FAIL'}")
        if not ok3:
            for n, g, r in zip(names, gen3_blocks, ref3_blocks):
                print(f"    {n}: {np.max(np.abs(g - r)):.2e}")

        # Laplace mode (production default, laplace_ntau=6): the driver never
        # materializes any of the four O(nv^3*no^3) t3_2_* tensors -- see
        # mpn_density_driver_unrestricted.py's compute_delta_gamma3 docstring
        # and generate_mpn_laplace.py. Cross-checked against
        # the oracle's own laplace_ntau=8 path (same pattern as
        # test_mpn_density_restricted.py).
        ref3_lap_blocks = ref3.compute_gamma3_blocks(laplace_ntau=8)
        gen3_lap_blocks = driver.compute_delta_gamma3(laplace_ntau=8)
        diff3_lap = max(np.max(np.abs(g - r)) for g, r in zip(gen3_lap_blocks, ref3_lap_blocks))
        ok3_lap = diff3_lap < 1e-8
        all_ok &= ok3_lap
        print(f"{label:20s} (no_a={nocc_a}, no_b={nocc_b}): generic unrestricted MP3 Laplace (ntau=8) matches "
              f"oracle Laplace(ntau=8) (max diff={diff3_lap:.2e}): {'OK' if ok3_lap else 'FAIL'}")
        if not ok3_lap:
            for n, g, r in zip(names, gen3_lap_blocks, ref3_lap_blocks):
                print(f"    {n}: {np.max(np.abs(g - r)):.2e}")

    print("ALL PASSED" if all_ok else "FAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
