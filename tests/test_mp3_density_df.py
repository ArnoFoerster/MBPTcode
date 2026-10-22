"""Validates the DF/RI variant of MPnDensityDriverRestricted/Unrestricted's
compute_delta_gamma3 against the
existing dense pipeline, at two validation tiers -- see
tests/test_mp2_density_df.py's own docstring for the general two-tier strategy
(exact-plumbing naux=norb^2, then realistic RI).

This additionally exercises BOTH the production (laplace_ntau=8, the
Laplace-fused T3^(2) route -- restricted's hand-written
_laplace_aaaaaa_contribution_df plus the DF+Laplace-composed gen_lap3_df
pieces) and the non-Laplace oracle path (laplace_ntau=None, full T3^(2)
materialized via the DF-dressed t3_2_*_numerator_df functions), to confirm that
DF and Laplace fusion compose cleanly on the same raw term list, not just
that either works alone.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import MPnDensityDriverRestricted
from src.SingleReference.DensityMatrix.mpn_density_driver_unrestricted import MPnDensityDriverUnrestricted
from src.Base.pyscf_interface import get_orbital_energies, get_antisymmetrized_spin_block_eri, DFIntegrals


def _max_diff(blocks_a, blocks_b):
    return max(np.max(np.abs(a - b)) for a, b in zip(blocks_a, blocks_b))


if __name__ == '__main__':
    all_ok = True

    # ---------------- RHF ----------------
    for atom, basis in [('H 0 0 0; F 0 0 0.9', 'sto-3g'), ('H 0 0 0; Li 0 0 1.6', 'sto-3g'),
                       ('O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', '6-31g')]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc = mol.nelectron // 2
        eps_a = get_orbital_energies(mf, representation='spatial')
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
        f_aa = np.diag(eps_a)

        dense = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc)
        dense_lap = dense.compute_delta_gamma3(laplace_ntau=8)
        dense_exact = dense.compute_delta_gamma3(laplace_ntau=None)

        dfi_exact = DFIntegrals.from_scf(mol, mf, exact=True)
        drv_exact = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc,
                                               B_aa=dfi_exact.B_aa, B_bb=dfi_exact.B_bb)
        df_lap = drv_exact.compute_delta_gamma3_df(laplace_ntau=8)
        df_exact = drv_exact.compute_delta_gamma3_df(laplace_ntau=None)

        diff_lap = _max_diff(dense_lap, df_lap)
        ok = diff_lap < 1e-8
        all_ok &= ok
        print(f"RHF {basis:8s} {atom:20s}: DF (naux=norb^2 exact) Laplace-mode compute_delta_gamma3 "
              f"matches dense (max diff={diff_lap:.2e}): {'OK' if ok else 'FAIL'}")

        diff_exact = _max_diff(dense_exact, df_exact)
        ok = diff_exact < 1e-8
        all_ok &= ok
        print(f"RHF {basis:8s} {atom:20s}: DF (naux=norb^2 exact) non-Laplace compute_delta_gamma3 "
              f"matches dense (max diff={diff_exact:.2e}): {'OK' if ok else 'FAIL'}")

        # cross-check: laplace and non-laplace paths agree with EACH OTHER
        # in DF mode too (not just each matching its own dense twin)
        diff_lap_vs_exact = _max_diff(df_lap, df_exact)
        ok = diff_lap_vs_exact < 1e-7
        all_ok &= ok
        print(f"RHF {basis:8s} {atom:20s}: DF Laplace-mode matches DF non-Laplace "
              f"(max diff={diff_lap_vs_exact:.2e}): {'OK' if ok else 'FAIL'}")

        mf.with_df = df.DF(mol)
        mf.with_df.build()
        dfi_ri = DFIntegrals.from_scf(mol, mf)
        drv_ri = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc,
                                            B_aa=dfi_ri.B_aa, B_bb=dfi_ri.B_bb)
        df_ri_lap = drv_ri.compute_delta_gamma3_df(laplace_ntau=8)
        diff_ri = _max_diff(dense_lap, df_ri_lap)
        ok_ri = diff_ri < 1e-3
        all_ok &= ok_ri
        print(f"RHF {basis:8s} {atom:20s}: DF (real JK-fit RI) Laplace-mode matches dense "
              f"(max diff={diff_ri:.2e}, naux={dfi_ri.naux_aa}): {'OK' if ok_ri else 'FAIL'}")
        mf.with_df = None

    # ---------------- UHF (incl. genuine open-shell) ----------------
    UHF_CASES = [
        ('H 0 0 0; Li 0 0 1.6', 'sto-3g', 0, 'LiH (RHF-as-UHF)'),
        ('O 0 0 0; H 0 0 0.97', '6-31g', 1, 'OH radical'),
        ('N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6', 'sto-3g', 1, 'NH2 radical'),
    ]
    for atom, basis, spin, label in UHF_CASES:
        mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
        mf = scf.UHF(mol).run()
        nocc_a, nocc_b = mf.nelec
        eps = get_orbital_energies(mf, representation='spatial')
        eps_a, eps_b = eps if isinstance(eps, tuple) else (eps, eps)
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
        f_aa, f_bb = np.diag(eps_a), np.diag(eps_b)

        dense = MPnDensityDriverUnrestricted(f_aa, f_bb, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b)
        dense_lap = dense.compute_delta_gamma3(laplace_ntau=8)
        dense_exact = dense.compute_delta_gamma3(laplace_ntau=None)

        dfi_exact = DFIntegrals.from_scf(mol, mf, exact=True)
        drv_exact = MPnDensityDriverUnrestricted(f_aa, f_bb, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b,
                                                  B_aa=dfi_exact.B_aa, B_bb=dfi_exact.B_bb)
        df_lap = drv_exact.compute_delta_gamma3_df(laplace_ntau=8)
        df_exact = drv_exact.compute_delta_gamma3_df(laplace_ntau=None)

        diff_lap = _max_diff(dense_lap, df_lap)
        ok = diff_lap < 1e-8
        all_ok &= ok
        print(f"UHF {label:20s} (no_a={nocc_a}, no_b={nocc_b}): DF (naux=norb^2 exact) Laplace-mode "
              f"matches dense (max diff={diff_lap:.2e}): {'OK' if ok else 'FAIL'}")

        diff_exact = _max_diff(dense_exact, df_exact)
        ok = diff_exact < 1e-8
        all_ok &= ok
        print(f"UHF {label:20s} (no_a={nocc_a}, no_b={nocc_b}): DF (naux=norb^2 exact) non-Laplace "
              f"matches dense (max diff={diff_exact:.2e}): {'OK' if ok else 'FAIL'}")

        mf.with_df = df.DF(mol)
        mf.with_df.build()
        dfi_ri = DFIntegrals.from_scf(mol, mf)
        drv_ri = MPnDensityDriverUnrestricted(f_aa, f_bb, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b,
                                              B_aa=dfi_ri.B_aa, B_bb=dfi_ri.B_bb)
        df_ri_lap = drv_ri.compute_delta_gamma3_df(laplace_ntau=8)
        diff_ri = _max_diff(dense_lap, df_ri_lap)
        ok_ri = diff_ri < 1e-3
        all_ok &= ok_ri
        print(f"UHF {label:20s} (no_a={nocc_a}, no_b={nocc_b}): DF (real JK-fit RI) Laplace-mode "
              f"matches dense (max diff={diff_ri:.2e}, naux={dfi_ri.naux_aa}): {'OK' if ok_ri else 'FAIL'}")
        mf.with_df = None

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
