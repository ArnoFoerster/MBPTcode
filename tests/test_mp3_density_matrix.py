"""Regression tests for the unrelaxed MP3 1-particle density-matrix correction (src/SingleReference/DensityMatrix/density_matrix.py):

1. trace(dgamma_oo) == -trace(dgamma_vv) exactly by construction (the
   unrelaxed correction is particle-number conserving order by order).
2. dgamma_oo/dgamma_vv are symmetric; dgamma_ov (third-order singles
   response) is nonzero.
3. The third-order energy reconstructed from t_ij^{ab(2)}
   (E^(3) = 1/4 sum_ijab <ij||ab> t_ij^{ab(2)}) must match the standard,
   independent MP3 energy formula built only from t_ij^{ab(1)}
   (particle-particle ladder + hole-hole ladder + ring terms) -- this
   cross-checks every term of compute_t2_second_order without relying on
   any external MP3 implementation (pyscf/psi4 have none).
4. t_ijk^{abc(2)} (compute_t3_second_order) is fully antisymmetric under
   exchange of any two occupied or any two virtual indices -- a necessary
   condition that a sign/transcription error in any of its 8 permutation
   terms would very likely break. Needs a system with >=3 occupied and
   >=3 virtual spin-orbitals or the amplitude vanishes trivially by Pauli
   exclusion (sto-3g HF and 6-31g H2 both fail that requirement, hence LiH).
5. compute_mp3_density_matrix_ao (relax=False and relax=True) both integrate
   to n_electrons; the CPHF/CPKS-relaxed ov block must actually differ from
   the direct t_i^{a(3)} one (i.e. the Z-vector solve ran); and the
   relax=False AO density, transformed back to MO basis, must reproduce the
   spatial-MO blocks from MP3DensityMatrixSolver.compute_gamma3_blocks_spatial
   exactly (mirrors test_mp2_density_matrix.py's check 5).

Note: compute_t1_second_order/compute_t2_second_order were both derived from
a raw generated equation that includes a disconnected
'-0.5 * sum <kl||kl> * t^(lower order)' energy-shift term; that term is
dropped, since check 3 above shows it breaks agreement with the independent
E^(3) formula by orders of magnitude (see density_matrix.py docstrings).

6. MP3DensityMatrixSolverUnrestricted (spin-case-resolved aaaa/bbbb/abab and
   aaa/aab/abb/bbb sectors, ~16x fewer FLOPs than the interleaved spin-orbital
   solver above -- see density_matrix.py docstring) must reproduce every
   amplitude (t2_1, t1_2, t2_2, t3_2, t1_3) and the final gamma3 oo/ov/vv
   blocks exactly on a restricted reference (RHF-folding oracle), and against
   a genuinely asymmetric-alpha/beta UHF oracle built from random data (the
   only independent way to exercise open-shell density_matrix.py, which had
   no prior UHF oracle at all).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.DensityMatrix.density_matrix import (compute_mp3_density_matrix_ao)
from tests.mpn_density_oracle import (MP3DensityMatrixSolver, MP3DensityMatrixSolverUnrestricted)
from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies,
    get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri,
    get_antisymmetrized_spin_block_eri,
)


def _independent_mp3_energy(t2_1, g, nocc):
    """E^(3) purely from t_ij^{ab(1)}: pp-ladder + hh-ladder + ring (standard textbook formula)."""
    norb = g.shape[0]
    occ, virt = slice(0, nocc), slice(nocc, norb)
    term1 = 0.125 * np.einsum('abij,abcd,cdij->', t2_1, g[virt, virt, virt, virt], t2_1, optimize=True)
    term2 = 0.125 * np.einsum('abij,klij,abkl->', t2_1, g[occ, occ, occ, occ], t2_1, optimize=True)
    term3 = np.einsum('abij,kbcj,acik->', t2_1, g[occ, virt, virt, occ], t2_1, optimize=True)
    return term1 + term2 + term3


if __name__ == '__main__':
    all_ok = True

    for atom, basis in [('H 0 0 0; F 0 0 0.9', 'sto-3g'), ('H 0 0 0; H 0 0 0.74', '6-31g'),
                         ('Li 0 0 0; H 0 0 1.6', '6-31g')]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc = mol.nelectron // 2

        eps_spin = get_orbital_energies(mf, representation='spin')
        eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
        g_anti_spin = get_antisymmetrized_spin_eri(eri_chemist)

        solver = MP3DensityMatrixSolver(eps_spin, g_anti_spin, nocc_spin=2 * nocc)
        t2_1 = solver.compute_t2()
        dgamma_oo, dgamma_ov, dgamma_vv = solver.compute_gamma3_blocks()

        # --- 1. Particle-number conservation
        trace_sum = np.trace(dgamma_oo) + np.trace(dgamma_vv)
        ok = abs(trace_sum) < 1e-8
        all_ok &= ok
        print(f"{basis:8s}: trace(dgamma_oo) == -trace(dgamma_vv) (sum={trace_sum:.2e}): {'OK' if ok else 'FAIL'}")

        # --- 2. Symmetry / nonzero ov response
        ok = np.allclose(dgamma_oo, dgamma_oo.T, atol=1e-10) and np.allclose(dgamma_vv, dgamma_vv.T, atol=1e-10)
        all_ok &= ok
        print(f"{basis:8s}: dgamma_oo/dgamma_vv are symmetric: {'OK' if ok else 'FAIL'}")

        ok = np.max(np.abs(dgamma_ov)) > 1e-6
        all_ok &= ok
        print(f"{basis:8s}: dgamma_ov (third-order singles response) is nonzero: {'OK' if ok else 'FAIL'}")

        # --- 3. t2^(2)-based E^(3) matches the independent t2^(1)-only formula
        t2_2 = solver.compute_t2_second_order(t2_1)
        norb = len(eps_spin)
        occ, virt = slice(0, 2 * nocc), slice(2 * nocc, norb)
        e3_from_t2_2 = 0.25 * np.einsum('ijab,abij->', g_anti_spin[occ, occ, virt, virt], t2_2, optimize=True)
        e3_independent = _independent_mp3_energy(t2_1, g_anti_spin, 2 * nocc)
        diff = abs(e3_from_t2_2 - e3_independent)
        ok = diff < 1e-8
        all_ok &= ok
        print(f"{basis:8s}: E^(3) from t2^(2) matches independent ladder+ring formula "
              f"({e3_from_t2_2:.8f} vs {e3_independent:.8f}, diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

        # --- 4. Full antisymmetry of t_ijk^{abc(2)} (only meaningful with >=3 occ and >=3 virt)
        t3_2 = solver.compute_t3_second_order(t2_1)
        nv = norb - 2 * nocc
        if 2 * nocc >= 3 and nv >= 3:
            perms = {
                'a<->b': (1, 0, 2, 3, 4, 5), 'b<->c': (0, 2, 1, 3, 4, 5), 'a<->c': (2, 1, 0, 3, 4, 5),
                'i<->j': (0, 1, 2, 4, 3, 5), 'j<->k': (0, 1, 2, 3, 5, 4), 'i<->k': (0, 1, 2, 5, 4, 3),
            }
            scale = np.max(np.abs(t3_2))
            worst = max(np.max(np.abs(t3_2 + t3_2.transpose(p))) for p in perms.values())
            ok = scale > 1e-6 and worst < 1e-8
            all_ok &= ok
            print(f"{basis:8s}: t3^(2) fully antisymmetric under occ/virt exchange "
                  f"(scale={scale:.2e}, worst violation={worst:.2e}): {'OK' if ok else 'FAIL'}")

        # --- 5. Relaxed (CPHF/CPKS) AO density matrix
        S = mol.intor('int1e_ovlp')
        nelec = mol.nelectron
        dm_lin = compute_mp3_density_matrix_ao(mf, mol, relax=False)
        dm_relaxed = compute_mp3_density_matrix_ao(mf, mol, relax=True)

        for label, dm in [('relax=False', dm_lin), ('relax=True', dm_relaxed)]:
            n = np.trace(dm @ S)
            ok = abs(n - nelec) < 1e-6
            all_ok &= ok
            print(f"{basis:8s}: {label} AO density integrates to n_electrons "
                  f"(N={n:.8f}, expect {nelec}): {'OK' if ok else 'FAIL'}")

        ok = not np.allclose(dm_lin, dm_relaxed, atol=1e-8)
        all_ok &= ok
        print(f"{basis:8s}: relaxed density differs from unrelaxed (CPHF actually ran): {'OK' if ok else 'FAIL'}")

        # relax=False AO density, folded back to MO basis, must reproduce
        # MP3DensityMatrixSolver.compute_gamma3_blocks_spatial exactly
        from tests.mpn_density_oracle import (MP2DensityMatrixSolver)
        mp2_solver = MP2DensityMatrixSolver(eps_spin, g_anti_spin, nocc_spin=2 * nocc)
        oo2_so, ov2_so, vv2_so = mp2_solver.compute_blocks()
        oo2_sp = oo2_so[0::2, 0::2] * 2.0
        ov2_sp = ov2_so[0::2, 0::2] * 2.0
        vv2_sp = vv2_so[0::2, 0::2] * 2.0

        mo = mf.mo_coeff
        mo_inv = mo.T @ S
        dgamma_check = mo_inv @ (dm_lin - mf.make_rdm1()) @ mo_inv.T
        oo_spatial_direct, ov_spatial_direct, vv_spatial_direct = solver.compute_gamma3_blocks_spatial()
        diff = max(
            np.max(np.abs(dgamma_check[:nocc, :nocc] - oo2_sp - oo_spatial_direct)),
            np.max(np.abs(dgamma_check[:nocc, nocc:] - ov2_sp - ov_spatial_direct)),
            np.max(np.abs(dgamma_check[nocc:, nocc:] - vv2_sp - vv_spatial_direct)),
        )
        ok = diff < 1e-8
        all_ok &= ok
        print(f"{basis:8s}: relax=False AO density folds back to compute_gamma3_blocks_spatial "
              f"(max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

        # --- 6. MP3DensityMatrixSolverUnrestricted (RHF-folding oracle): every
        # amplitude and the final gamma3 blocks must match the interleaved
        # solver's native (undoubled) alpha spin-orbital sub-block exactly.
        eps = get_orbital_energies(mf, representation='spatial')
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
        usolver = MP3DensityMatrixSolverUnrestricted(eps, eps, g_aaaa, g_bbbb, g_abab, nocc, nocc)
        t2_1_aaaa, t2_1_bbbb, t2_1_abab = usolver.compute_t2_1()
        t1_2_aa, t1_2_bb = usolver.compute_t1_2(t2_1_aaaa, t2_1_bbbb, t2_1_abab)
        t2_2_aaaa, t2_2_bbbb, t2_2_abab = usolver.compute_t2_2(t2_1_aaaa, t2_1_bbbb, t2_1_abab)
        t3_2_aaa, t3_2_bbb, t3_2_aab, t3_2_abb = usolver.compute_t3_2(t2_1_aaaa, t2_1_bbbb, t2_1_abab)
        t1_3_aa, t1_3_bb = usolver.compute_t1_3(t1_2_aa, t1_2_bb, t2_2_aaaa, t2_2_bbbb, t2_2_abab,
                                                 t3_2_aaa, t3_2_bbb, t3_2_aab, t3_2_abb)
        oo_a, oo_b, ov_a, ov_b, vv_a, vv_b = usolver.compute_gamma3_blocks()

        diffs = {
            't2_1 aaaa': np.max(np.abs(t2_1_aaaa - t2_1[0::2, 0::2, 0::2, 0::2])),
            't2_1 abab': np.max(np.abs(t2_1_abab - t2_1[0::2, 1::2, 0::2, 1::2])),
            't1_2': np.max(np.abs(t1_2_aa - solver.compute_t1_second_order(t2_1)[0::2, 0::2])),
            't2_2 aaaa': np.max(np.abs(t2_2_aaaa - t2_2[0::2, 0::2, 0::2, 0::2])),
            't2_2 abab': np.max(np.abs(t2_2_abab - t2_2[0::2, 1::2, 0::2, 1::2])),
            'gamma3 oo': np.max(np.abs(oo_a - dgamma_oo[0::2, 0::2])),
            'gamma3 ov': np.max(np.abs(ov_a - dgamma_ov[0::2, 0::2])),
            'gamma3 vv': np.max(np.abs(vv_a - dgamma_vv[0::2, 0::2])),
        }
        if 2 * nocc >= 3 and nv >= 3:
            diffs['t3_2 aaa'] = np.max(np.abs(t3_2_aaa - t3_2[0::2, 0::2, 0::2, 0::2, 0::2, 0::2]))
            diffs['t3_2 aab'] = np.max(np.abs(t3_2_aab - t3_2[0::2, 0::2, 1::2, 0::2, 0::2, 1::2]))
            diffs['t3_2 abb'] = np.max(np.abs(t3_2_abb - t3_2[0::2, 1::2, 1::2, 0::2, 1::2, 1::2]))
            diffs['t1_3'] = np.max(np.abs(t1_3_aa - solver.compute_t1_third_order(
                solver.compute_t1_second_order(t2_1), t2_2, t3_2)[0::2, 0::2]))
        ok = all(d < 1e-8 for d in diffs.values())
        all_ok &= ok
        worst = max(diffs.values())
        print(f"{basis:8s}: MP3DensityMatrixSolverUnrestricted matches interleaved solver "
              f"exactly (worst diff={worst:.2e} over {list(diffs)}): {'OK' if ok else 'FAIL'}")

        print()

    # --- Genuine open-shell (UHF, asymmetric alpha/beta spaces) oracle: random
    # data, since building a matching block-stacked interleaved spin-orbital
    # system by hand is the only independent way to exercise UHF density_matrix.py
    # had no prior oracle for -- see MP3DensityMatrixSolverUnrestricted docstring.
    print("--- random UHF (asymmetric alpha/beta) oracle ---")
    rng = np.random.RandomState(11)
    no_a, nv_a, no_b, nv_b = 3, 4, 2, 3
    na, nb = no_a + nv_a, no_b + nv_b

    def antisym_phys(n):
        raw = rng.randn(n, n, n, n)
        raw = raw + raw.transpose(1, 0, 3, 2)
        raw = raw + raw.transpose(2, 3, 0, 1)
        return raw

    phys_aa, phys_bb = antisym_phys(na), antisym_phys(nb)
    g_aaaa_r = phys_aa - phys_aa.transpose(0, 1, 3, 2)
    g_bbbb_r = phys_bb - phys_bb.transpose(0, 1, 3, 2)
    g_abab_r = rng.randn(na, nb, na, nb)
    g_abab_r = g_abab_r + g_abab_r.transpose(2, 3, 0, 1)
    eps_a_r = np.sort(rng.randn(na)); eps_a_r[:no_a] -= 5.0
    eps_b_r = np.sort(rng.randn(nb)); eps_b_r[:no_b] -= 5.0

    urand = MP3DensityMatrixSolverUnrestricted(eps_a_r, eps_b_r, g_aaaa_r, g_bbbb_r, g_abab_r, no_a, no_b)
    t2_1_aaaa_r, t2_1_bbbb_r, t2_1_abab_r = urand.compute_t2_1()
    t1_2_aa_r, t1_2_bb_r = urand.compute_t1_2(t2_1_aaaa_r, t2_1_bbbb_r, t2_1_abab_r)
    t3_2_aaa_r, t3_2_bbb_r, t3_2_aab_r, t3_2_abb_r = urand.compute_t3_2(t2_1_aaaa_r, t2_1_bbbb_r, t2_1_abab_r)
    oo_a_r, oo_b_r, ov_a_r, ov_b_r, vv_a_r, vv_b_r = urand.compute_gamma3_blocks()

    # block-stacked (not interleaved) full spin-orbital system: order = [a-occ, b-occ, a-virt, b-virt]
    nso, nocc_so = na + nb, no_a + no_b
    order = np.concatenate([np.arange(0, no_a), np.arange(na, na + no_b),
                             np.arange(no_a, na), np.arange(na + no_b, nso)])
    g_so = np.zeros((nso, nso, nso, nso))
    g_so[0:na, 0:na, 0:na, 0:na] = g_aaaa_r
    g_so[na:, na:, na:, na:] = g_bbbb_r
    g_so[0:na, na:, 0:na, na:] = g_abab_r
    g_so[na:, 0:na, na:, 0:na] = g_abab_r.transpose(1, 0, 3, 2)
    g_so[0:na, na:, na:, 0:na] = -g_abab_r.transpose(0, 1, 3, 2)
    g_so[na:, 0:na, 0:na, na:] = -g_abab_r.transpose(1, 0, 2, 3)
    g_so_ord = g_so[np.ix_(order, order, order, order)]
    eps_so_ord = np.concatenate([eps_a_r, eps_b_r])[order]

    old_rand = MP3DensityMatrixSolver(eps_so_ord, g_so_ord, nocc_spin=nocc_so)
    t2_1_old_r = old_rand.compute_t2()
    t3_2_old_r = old_rand.compute_t3_second_order(t2_1_old_r)
    oo_old_r, ov_old_r, vv_old_r = old_rand.compute_gamma3_blocks()
    oa_idx, va_idx = slice(0, no_a), slice(0, nv_a)
    ob_idx, vb_idx = slice(no_a, nocc_so), slice(nv_a, nv_a + nv_b)

    diff_t3 = np.max(np.abs(t3_2_aaa_r - t3_2_old_r[np.ix_(range(0, nv_a), range(0, nv_a), range(0, nv_a),
                                                   range(0, no_a), range(0, no_a), range(0, no_a))]))
    diff_oo_a = np.max(np.abs(oo_a_r - oo_old_r[oa_idx, oa_idx]))
    diff_ov_a = np.max(np.abs(ov_a_r - ov_old_r[oa_idx, va_idx]))
    diff_vv_a = np.max(np.abs(vv_a_r - vv_old_r[va_idx, va_idx]))
    diff_oo_b = np.max(np.abs(oo_b_r - oo_old_r[ob_idx, ob_idx]))
    diff_ov_b = np.max(np.abs(ov_b_r - ov_old_r[ob_idx, vb_idx]))
    diff_vv_b = np.max(np.abs(vv_b_r - vv_old_r[vb_idx, vb_idx]))

    print(f"    diff_t3  : {diff_t3:.2e}")
    print(f"    diff_oo_a: {diff_oo_a:.2e}")
    print(f"    diff_ov_a: {diff_ov_a:.2e}")
    print(f"    diff_vv_a: {diff_vv_a:.2e}")
    print(f"    diff_oo_b: {diff_oo_b:.2e}")
    print(f"    diff_ov_b: {diff_ov_b:.2e}")
    print(f"    diff_vv_b: {diff_vv_b:.2e}")

    diff = max(diff_t3, diff_oo_a, diff_ov_a, diff_vv_a, diff_oo_b, diff_ov_b, diff_vv_b)
    ok = diff < 1e-8
    all_ok &= ok
    print(f"UHF (no_a={no_a},nv_a={nv_a},no_b={no_b},nv_b={nv_b}): matches block-stacked "
          f"interleaved oracle (max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    # --- Real open-shell UHF molecule: unrelaxed AO density integrates to n_electrons
    mol_uhf = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.9 0 -0.3', basis='6-31g', spin=2, verbose=0)
    mf_uhf = scf.UHF(mol_uhf).run()
    S_uhf = mol_uhf.intor('int1e_ovlp')
    dm_a, dm_b = compute_mp3_density_matrix_ao(mf_uhf, mol_uhf, relax=False)
    n_uhf = np.trace((dm_a + dm_b) @ S_uhf)
    ok = abs(n_uhf - mol_uhf.nelectron) < 1e-6
    all_ok &= ok
    print(f"UHF OH radical: relax=False AO density integrates to n_electrons "
          f"(N={n_uhf:.8f}, expect {mol_uhf.nelectron}): {'OK' if ok else 'FAIL'}")

    print("ALL PASSED" if all_ok else "FAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
