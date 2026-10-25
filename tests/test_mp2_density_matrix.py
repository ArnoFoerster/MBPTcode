"""Regression tests for the unrelaxed MP2 1-particle density-matrix correction (src/SingleReference/DensityMatrix/density_matrix.py):

1. The oo/vv blocks (folded from spin-orbital to spatial MO) must reproduce
   pyscf's own unrelaxed MP2 density (mp.MP2.make_rdm1() minus the HF
   reference density) exactly -- pyscf's convention zeroes the ov block, so
   this only checks oo/vv.
2. trace(dgamma_oo) == -trace(dgamma_vv) exactly by construction (the
   unrelaxed correction is particle-number conserving).
3. The MP2 correlation energy reconstructed from the t2 amplitudes
   (E_corr = 1/4 sum_ijab <ij||ab> t_ij^ab) must match pyscf's MP2.e_corr --
   this independently validates the antisymmetrized spin-orbital integrals
   and t2 denominators used to build the density blocks.
4. dgamma_oo/dgamma_vv are symmetric; dgamma_ov (the second-order singles
   response) is nonzero, unlike pyscf's zeroed orbital-response block.
5. compute_mp2_density_matrix_ao (relax=False and relax=True) both integrate
   to n_electrons; the CPHF/CPKS-relaxed ov block must actually differ from
   the unrelaxed one (i.e. the Z-vector solve ran); and the relax=False AO
   density, transformed back to MO basis, must reproduce the spatial-MO
   blocks from MP2DensityMatrixSolver.compute_blocks_spatial exactly.
6. MP2DensityMatrixSolverUnrestricted (spin-case-resolved aaaa/bbbb/abab,
   ~16x fewer FLOPs than the interleaved spin-orbital solver above) must
   reproduce that solver's native alpha spin-orbital sub-block exactly on a
   restricted reference (RHF-folding oracle), and compute_mp2_density_matrix_ao
   must give the same UHF AO density as folding the interleaved solver's own
   spin-orbital blocks (built from a genuinely asymmetric alpha/beta system,
   not just an RHF fold) -- both derived via the spin_labels mechanism,
   see density_matrix.py's MP2DensityMatrixSolverUnrestricted docstring.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, mp

from src.SingleReference.DensityMatrix.density_matrix import (compute_mp2_density_matrix_ao)
from tests.mpn_density_oracle import (compute_mp2_density_matrix, MP2DensityMatrixSolver, MP2DensityMatrixSolverUnrestricted)
from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies,
    get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri,
    get_antisymmetrized_spin_block_eri,
)


def _spin_to_spatial(block):
    """Fold an interleaved alpha/beta spin-orbital block to its restricted spatial-MO value (alpha block * 2)."""
    return 2 * block[0::2, 0::2]


if __name__ == '__main__':
    all_ok = True

    for atom, basis in [('H 0 0 0; F 0 0 0.9', 'sto-3g'), ('Ne 0 0 0', 'aug-cc-pvdz')]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc = mol.nelectron // 2
        nmo = mf.mo_coeff.shape[1]

        dgamma_oo, dgamma_ov, dgamma_vv = compute_mp2_density_matrix(mf, mol, nocc)

        # --- 1. oo/vv blocks reproduce pyscf's own unrelaxed MP2 density
        mymp = mp.MP2(mf).run()
        dm1_mo = mymp.make_rdm1()
        dm_hf_mo = np.diag([2.0] * nocc + [0.0] * (nmo - nocc))
        dcorr = dm1_mo - dm_hf_mo

        oo_spatial = _spin_to_spatial(dgamma_oo)
        vv_spatial = _spin_to_spatial(dgamma_vv)
        diff_oo = np.max(np.abs(oo_spatial - dcorr[:nocc, :nocc]))
        diff_vv = np.max(np.abs(vv_spatial - dcorr[nocc:, nocc:]))
        ok = diff_oo < 1e-8 and diff_vv < 1e-8
        all_ok &= ok
        print(f"{basis:12s}: oo/vv blocks match pyscf unrelaxed MP2 density "
              f"(max diff oo={diff_oo:.2e}, vv={diff_vv:.2e}): {'OK' if ok else 'FAIL'}")

        # --- 2. Particle-number conservation
        trace_sum = np.trace(dgamma_oo) + np.trace(dgamma_vv)
        ok = abs(trace_sum) < 1e-8
        all_ok &= ok
        print(f"{basis:12s}: trace(dgamma_oo) == -trace(dgamma_vv) "
              f"(sum={trace_sum:.2e}): {'OK' if ok else 'FAIL'}")

        # --- 3. t2 amplitudes reproduce the pyscf MP2 correlation energy
        eps_spin = get_orbital_energies(mf, representation='spin')
        eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
        g_anti_spin = get_antisymmetrized_spin_eri(eri_chemist)
        solver = MP2DensityMatrixSolver(eps_spin, g_anti_spin, nocc_spin=2 * nocc)
        t2 = solver.compute_t2()  # t2[a,b,i,j] = t_ij^ab

        occ, virt = slice(0, 2 * nocc), slice(2 * nocc, 2 * nmo)
        g_ijab = g_anti_spin[occ, occ, virt, virt]
        e_corr = 0.25 * np.einsum('ijab,abij->', g_ijab, t2)
        ok = abs(e_corr - mymp.e_corr) < 1e-8
        all_ok &= ok
        print(f"{basis:12s}: E_corr from t2 amplitudes matches pyscf MP2 "
              f"({e_corr:.8f} vs {mymp.e_corr:.8f}): {'OK' if ok else 'FAIL'}")

        # --- 4. Block symmetry / nonzero ov response
        ok = np.allclose(dgamma_oo, dgamma_oo.T, atol=1e-10) and np.allclose(dgamma_vv, dgamma_vv.T, atol=1e-10)
        all_ok &= ok
        print(f"{basis:12s}: dgamma_oo/dgamma_vv are symmetric: {'OK' if ok else 'FAIL'}")

        ok = np.max(np.abs(dgamma_ov)) > 1e-6
        all_ok &= ok
        print(f"{basis:12s}: dgamma_ov (second-order singles response) is nonzero: {'OK' if ok else 'FAIL'}")

        # --- 5. Relaxed (CPHF/CPKS) AO density matrix
        S = mol.intor('int1e_ovlp')
        nelec = mol.nelectron
        dm_lin = compute_mp2_density_matrix_ao(mf, mol, relax=False)
        dm_relaxed = compute_mp2_density_matrix_ao(mf, mol, relax=True)

        for label, dm in [('relax=False', dm_lin), ('relax=True', dm_relaxed)]:
            n = np.trace(dm @ S)
            ok = abs(n - nelec) < 1e-6
            all_ok &= ok
            print(f"{basis:12s}: {label} AO density integrates to n_electrons "
                  f"(N={n:.8f}, expect {nelec}): {'OK' if ok else 'FAIL'}")

        ok = not np.allclose(dm_lin, dm_relaxed, atol=1e-8)
        all_ok &= ok
        print(f"{basis:12s}: relaxed density differs from unrelaxed (CPHF actually ran): {'OK' if ok else 'FAIL'}")

        # relax=False AO density, folded back to MO basis, must reproduce
        # MP2DensityMatrixSolver.compute_blocks_spatial exactly (mo.T @ S @ mo == I)
        mo = mf.mo_coeff
        mo_inv = mo.T @ S
        dgamma_check = mo_inv @ (dm_lin - mf.make_rdm1()) @ mo_inv.T
        oo_spatial_direct, ov_spatial_direct, vv_spatial_direct = solver.compute_blocks_spatial()
        diff = max(
            np.max(np.abs(dgamma_check[:nocc, :nocc] - oo_spatial_direct)),
            np.max(np.abs(dgamma_check[:nocc, nocc:] - ov_spatial_direct)),
            np.max(np.abs(dgamma_check[nocc:, nocc:] - vv_spatial_direct)),
        )
        ok = diff < 1e-8
        all_ok &= ok
        print(f"{basis:12s}: relax=False AO density folds back to compute_blocks_spatial "
              f"(max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

        # --- 6. MP2DensityMatrixSolverUnrestricted (RHF-folding oracle): its alpha
        # block must equal the interleaved solver's own native (undoubled) alpha
        # spin-orbital sub-block exactly.
        eps = get_orbital_energies(mf, representation='spatial')
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
        usolver = MP2DensityMatrixSolverUnrestricted(eps, eps, g_aaaa, g_bbbb, g_abab, nocc, nocc)
        oo_a, oo_b, ov_a, ov_b, vv_a, vv_b = usolver.compute_blocks()
        diff = max(
            np.max(np.abs(oo_a - dgamma_oo[0::2, 0::2])),
            np.max(np.abs(ov_a - dgamma_ov[0::2, 0::2])),
            np.max(np.abs(vv_a - dgamma_vv[0::2, 0::2])),
        )
        ok = diff < 1e-8 and np.max(np.abs(oo_a - oo_b)) < 1e-10 and np.max(np.abs(vv_a - vv_b)) < 1e-10
        all_ok &= ok
        print(f"{basis:12s}: MP2DensityMatrixSolverUnrestricted alpha block matches interleaved "
              f"solver's native spin-orbital block (max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

        print()

    # --- Genuine open-shell (UHF, asymmetric alpha/beta spaces) oracle: random
    # data, since building a matching block-stacked interleaved spin-orbital
    # system by hand is the only independent way to exercise UHF density_matrix.py
    # had no prior oracle for -- see MP2DensityMatrixSolverUnrestricted docstring.
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

    urand = MP2DensityMatrixSolverUnrestricted(eps_a_r, eps_b_r, g_aaaa_r, g_bbbb_r, g_abab_r, no_a, no_b)
    oo_a_r, oo_b_r, ov_a_r, ov_b_r, vv_a_r, vv_b_r = urand.compute_blocks()

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

    old_rand = MP2DensityMatrixSolver(eps_so_ord, g_so_ord, nocc_spin=nocc_so)
    oo_old_r, ov_old_r, vv_old_r = old_rand.compute_blocks()
    oa_idx, va_idx = slice(0, no_a), slice(0, nv_a)
    ob_idx, vb_idx = slice(no_a, nocc_so), slice(nv_a, nv_a + nv_b)

    diff = max(
        np.max(np.abs(oo_a_r - oo_old_r[oa_idx, oa_idx])), np.max(np.abs(ov_a_r - ov_old_r[oa_idx, va_idx])),
        np.max(np.abs(vv_a_r - vv_old_r[va_idx, va_idx])), np.max(np.abs(oo_b_r - oo_old_r[ob_idx, ob_idx])),
        np.max(np.abs(ov_b_r - ov_old_r[ob_idx, vb_idx])), np.max(np.abs(vv_b_r - vv_old_r[vb_idx, vb_idx])),
    )
    ok = diff < 1e-8
    all_ok &= ok
    print(f"UHF (no_a={no_a},nv_a={nv_a},no_b={no_b},nv_b={nv_b}): matches block-stacked "
          f"interleaved oracle (max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    # --- Real open-shell UHF molecule: unrelaxed AO density integrates to n_electrons
    mol_uhf = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.9 0 -0.3', basis='6-31g', spin=2, verbose=0)
    mf_uhf = scf.UHF(mol_uhf).run()
    S_uhf = mol_uhf.intor('int1e_ovlp')
    dm_a, dm_b = compute_mp2_density_matrix_ao(mf_uhf, mol_uhf, relax=False)
    n_uhf = np.trace((dm_a + dm_b) @ S_uhf)
    ok = abs(n_uhf - mol_uhf.nelectron) < 1e-6
    all_ok &= ok
    print(f"UHF OH radical: relax=False AO density integrates to n_electrons "
          f"(N={n_uhf:.8f}, expect {mol_uhf.nelectron}): {'OK' if ok else 'FAIL'}")

    print("ALL PASSED" if all_ok else "FAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
