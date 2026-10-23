"""Validates the DF/RI variant of MPnDensityDriverRestricted/Unrestricted's
compute_delta_gamma2 against the existing dense pipeline,
at two validation tiers:

1. DFIntegrals.from_scf(..., exact=True) forces the
   naux=norb^2 eigh-decomposition fallback (not real DF), so the DF driver
   should reproduce the dense driver to ~1e-10 -- this isolates "did the
   codegen rewrite (dress_integral_factor) preserve exact semantics" from
   "how good is the RI approximation," independent of any real auxiliary
   basis.
2. Realistic-RI check: a real published JK-fit auxiliary basis
   (density_fit()'s default), compared to the dense pipeline at the
   RI-approximation-level tolerance
   (~1e-4 relative on a density-matrix element scale).

Covers both RHF (MPnDensityDriverRestricted) and UHF
(MPnDensityDriverUnrestricted), including genuine open-shell UHF
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import MPnDensityDriverRestricted
from src.SingleReference.DensityMatrix.mpn_density_driver_unrestricted import MPnDensityDriverUnrestricted
from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_antisymmetrized_spin_block_eri, DFIntegrals


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

        dense_blocks = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc).compute_delta_gamma2()

        dfi_exact = DFIntegrals.from_scf(mol, mf, exact=True)
        drv_exact = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc,
                                               B_aa=dfi_exact.B_aa, B_bb=dfi_exact.B_bb)
        df_exact_blocks = drv_exact.compute_delta_gamma2_df()
        diff_exact = _max_diff(dense_blocks, df_exact_blocks)
        ok = diff_exact < 1e-9
        all_ok &= ok
        print(f"RHF {basis:8s} {atom:20s}: DF (naux=norb^2 exact) matches dense "
              f"compute_delta_gamma2 (max diff={diff_exact:.2e}): {'OK' if ok else 'FAIL'}")

        # Post-hoc RI approximation of the SAME converged (non-DF) mf's
        # integrals -- attaching with_df does NOT re-run SCF or perturb
        # mo_coeff/mo_energy, so eps_a/f_aa/g_aaaa above and B_aa/B_bb below
        # are guaranteed to be in the exact same MO basis; only the integral
        # REPRESENTATION differs (dense vs. RI-fitted), isolating the RI
        # approximation error from any SCF-convergence difference a fresh
        # density_fit().run() would also introduce.
        mf.with_df = df.DF(mol)
        mf.with_df.build()
        dfi_ri = DFIntegrals.from_scf(mol, mf)
        drv_ri = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc,
                                            B_aa=dfi_ri.B_aa, B_bb=dfi_ri.B_bb)
        df_ri_blocks = drv_ri.compute_delta_gamma2_df()
        diff_ri = _max_diff(dense_blocks, df_ri_blocks)
        ok_ri = diff_ri < 1e-4
        all_ok &= ok_ri
        print(f"RHF {basis:8s} {atom:20s}: DF (real JK-fit RI) matches dense "
              f"compute_delta_gamma2 (max diff={diff_ri:.2e}, naux={dfi_ri.naux_aa}): "
              f"{'OK' if ok_ri else 'FAIL'}")

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

        dense_blocks = MPnDensityDriverUnrestricted(
            f_aa, f_bb, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b).compute_delta_gamma2()

        dfi_exact = DFIntegrals.from_scf(mol, mf, exact=True)
        drv_exact = MPnDensityDriverUnrestricted(f_aa, f_bb, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b,
                                                  B_aa=dfi_exact.B_aa, B_bb=dfi_exact.B_bb)
        df_exact_blocks = drv_exact.compute_delta_gamma2_df()
        diff_exact = _max_diff(dense_blocks, df_exact_blocks)
        ok = diff_exact < 1e-9
        all_ok &= ok
        print(f"UHF {label:20s} (no_a={nocc_a}, no_b={nocc_b}): DF (naux=norb^2 exact) matches "
              f"dense compute_delta_gamma2 (max diff={diff_exact:.2e}): {'OK' if ok else 'FAIL'}")

        # Post-hoc RI approximation of the same converged mf -- see the RHF
        # loop's comment above for why this (not a fresh density_fit().run())
        # is the correct way to isolate RI approximation error.
        mf.with_df = df.DF(mol)
        mf.with_df.build()
        dfi_ri = DFIntegrals.from_scf(mol, mf)
        drv_ri = MPnDensityDriverUnrestricted(f_aa, f_bb, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b,
                                              B_aa=dfi_ri.B_aa, B_bb=dfi_ri.B_bb)
        df_ri_blocks = drv_ri.compute_delta_gamma2_df()
        diff_ri = _max_diff(dense_blocks, df_ri_blocks)
        ok_ri = diff_ri < 1e-4
        all_ok &= ok_ri
        print(f"UHF {label:20s} (no_a={nocc_a}, no_b={nocc_b}): DF (real JK-fit RI) matches "
              f"dense compute_delta_gamma2 (max diff={diff_ri:.2e}, naux={dfi_ri.naux_aa}): "
              f"{'OK' if ok_ri else 'FAIL'}")

    # ---------------- streamed (rank-4-free) DF-MP2 density ----------------
    # compute_delta_gamma2_df_streamed must (a) reproduce the materialized
    # DF driver exactly, bare + EN hh/pp dressed, at several chunk sizes,
    # and (b) never touch any rank-4 producer -- guard-checked by making
    # the three producers on that path raise.
    from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import (
        compute_delta_gamma2_df_streamed)
    import src.SingleReference.DensityMatrix.mpn_density_driver_restricted as _drv_mod
    import src.SingleReference.DensityMatrix.generated_mpn_restricted.mpn_density_pieces_restricted_df as _gdf
    import src.SingleReference.ADC.static_correction as _sc
    import src.SingleReference.ADC.adc_r_utils as _ar
    from src.SingleReference.ADC import build_mp2_static_correction_restricted
    from src.SingleReference.EpsteinNesbet import (
        build_dressed_e_abij, build_dressed_e_ai, restricted_channel_shifts)

    mol6 = gto.M(atom='C 0 0 0; O 0 0 1.13', basis='sto-3g', verbose=0)
    mf6 = scf.RHF(mol6).run()
    nocc6 = mol6.nelectron // 2
    eps6 = get_orbital_energies(mf6, representation='spatial')
    B6 = DFIntegrals.from_scf(mol6, mf6, exact=True).B_aa
    norb6 = len(eps6)
    O6, V6 = nocc6, norb6 - nocc6
    vidx6 = np.arange(O6, norb6)

    for dname, dress in (("bare", None), ("hh+pp", {'hh': True, 'pp': True})):
        e_abij6 = e_ai6 = dh6 = dp6 = None
        if dress is not None:
            e_abij6 = build_dressed_e_abij(dress, eps6, O6, V6, B_aa=B6)
            e_ai6 = build_dressed_e_ai(dress, eps6, O6, V6, B_aa=B6)
            d_h6, d_p6, _ = restricted_channel_shifts(dress, B6, None, None, O6, vidx6)
            dh6, dp6 = d_h6[0], d_p6[0]
        ref6 = MPnDensityDriverRestricted(np.diag(eps6), None, None, None, nocc6,
                                          B_aa=B6, B_bb=B6, e_abij=e_abij6,
                                          e_ai=e_ai6).compute_delta_gamma2_df()
        worst = 0.0
        for chunk in (1, 3, O6):
            got6 = compute_delta_gamma2_df_streamed(B6, eps6, nocc6, dh=dh6, dp=dp6,
                                                    e_ai=e_ai6, chunk_size=chunk)
            worst = max(worst, _max_diff(ref6, got6))
        ok = worst < 1e-11
        all_ok &= ok
        print(f"streamed dgamma2 ({dname}) matches materialized DF driver, chunks 1/3/O "
              f"(worst diff={worst:.2e}): {'OK' if ok else 'FAIL'}")

    def _boom(*a, **k):
        raise AssertionError("rank-4 producer called on streamed MP2 path")
    _saved = (_gdf.t2_1_aaaa_numerator_df, _drv_mod.to_l, _sc.build_dressed_e_abij)
    _gdf.t2_1_aaaa_numerator_df = _boom
    _drv_mod.to_l = _boom
    _sc.build_dressed_e_abij = _boom
    try:
        build_mp2_static_correction_restricted(mf6, mol6, nocc6, relax=True, B_aa=B6,
                                               u2_denom_dress={'hh': True, 'pp': True})
        build_mp2_static_correction_restricted(mf6, mol6, nocc6, relax=True, B_aa=B6)
        guard_ok = True
    except AssertionError:
        guard_ok = False
    finally:
        _gdf.t2_1_aaaa_numerator_df, _drv_mod.to_l, _sc.build_dressed_e_abij = _saved
    all_ok &= guard_ok
    print(f"streamed MP2 static correction touches NO rank-4 producer "
          f"(t2 numerator/to_l/build_dressed_e_abij all guarded): "
          f"{'OK' if guard_ok else 'FAIL'}")

    # hp-dress must FALL BACK (still exact) rather than stream
    stat_hp = build_mp2_static_correction_restricted(
        mf6, mol6, nocc6, relax=True, B_aa=B6,
        u2_denom_dress={'hh': True, 'pp': True, 'hp': True})
    _ar_orig = _ar.dress_is_streamable
    _ar.dress_is_streamable = lambda d: False
    try:
        stat_hp_ref = build_mp2_static_correction_restricted(
            mf6, mol6, nocc6, relax=True, B_aa=B6,
            u2_denom_dress={'hh': True, 'pp': True, 'hp': True})
    finally:
        _ar.dress_is_streamable = _ar_orig
    d_hp = np.abs(stat_hp - stat_hp_ref).max()
    ok = d_hp < 1e-12
    all_ok &= ok
    print(f"hp-dressed MP2 static correction falls back to materialized path, exact "
          f"(diff={d_hp:.2e}): {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
