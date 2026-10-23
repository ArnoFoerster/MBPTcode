"""Correctness oracle for solve_cphf_relaxation_uhf (the open-shell mp2_relaxed
route), the UHF counterpart of tests/test_mp3_finite_field.py's checks 1/2.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, mp

from src.Base.pyscf.pyscf_interface import (get_orbital_energies,
                                            get_antisymmetrized_spin_block_eri,
                                            uhf_blockstacked_order)
from src.SingleReference.ADC.static_correction import build_mp2_static_correction
from src.SingleReference.DensityMatrix.mpn_density_driver_unrestricted import (
    MPnDensityDriverUnrestricted)
from src.SingleReference.DensityMatrix.density_matrix import (
    solve_cphf_relaxation_uhf, build_dgamma_block)


def interleaved_to_blockstacked_perm(nocc_a, nocc_b, norb):
    """Permutation p mapping the RHF branch's interleaved alpha/beta spin-orbital
    ordering onto the UHF branch's blockstacked one: M_block = M_inter[ix_(p, p)]."""
    order = uhf_blockstacked_order(nocc_a, nocc_b, norb, norb)
    return np.array([2 * u if u < norb else 2 * (u - norb) + 1 for u in order])


def make_field_uhf(mol, field_z):
    """Converged UHF under a uniform field along z (electronic coupling only,
    h1 -> h1 + field_z * z_AO -- same sign convention as test_mp3_finite_field)."""
    mf = scf.UHF(mol)
    h1 = scf.hf.get_hcore(mol)
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    mf.get_hcore = lambda *args, **kwargs: h1 + field_z * ao_dip[2]
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.run()
    assert mf.converged
    return mf


def relaxed_mp2_dm_ao_uhf(mf, mol, relax):
    """gamma_UHF + Delta_gamma^MP2 in the AO basis (both spins summed),
    replicating build_mp2_static_correction's UHF-branch density build."""
    nocc_a, nocc_b = mf.nelec
    eps_a, eps_b = get_orbital_energies(mf, representation='spatial')
    g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
    driver = MPnDensityDriverUnrestricted(np.diag(eps_a), np.diag(eps_b),
                                          g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b)
    oo_a, oo_b, ov_a, ov_b, vv_a, vv_b = driver.compute_delta_gamma2()
    if relax:
        dg_a, dg_b = solve_cphf_relaxation_uhf(
            mf, nocc_a, nocc_b, oo_a, ov_a, vv_a, oo_b, ov_b, vv_b)
    else:
        dg_a = build_dgamma_block(len(eps_a), nocc_a, oo_a, ov_a, vv_a)
        dg_b = build_dgamma_block(len(eps_b), nocc_b, oo_b, ov_b, vv_b)
    mo_a, mo_b = mf.mo_coeff
    dm_hf_a = np.diag((mf.mo_occ[0] > 0).astype(float))
    dm_hf_b = np.diag((mf.mo_occ[1] > 0).astype(float))
    return (mo_a @ (dm_hf_a + dg_a) @ mo_a.T
            + mo_b @ (dm_hf_b + dg_b) @ mo_b.T)


if __name__ == '__main__':
    all_ok = True

    # --- 1. Closed-shell reduction: UHF branch must reproduce the RHF branch
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.run()
    nocc = mol.nelectron // 2
    norb = mf.mo_coeff.shape[1]
    mfu = scf.addons.convert_to_uhf(mf)
    perm = interleaved_to_blockstacked_perm(nocc, nocc, norb)

    for relax, thresh in ((False, 1e-12), (True, 1e-8)):
        sc_r = build_mp2_static_correction(mf, mol, nocc, relax=relax)
        sc_u = build_mp2_static_correction(mfu, mol, relax=relax)
        diff = np.abs(sc_r[np.ix_(perm, perm)] - sc_u).max()
        ok = diff < thresh
        all_ok &= ok
        tag = 'relaxed' if relax else 'unrelaxed (mapping control)'
        print(f"closed-shell H2O, {tag}: max|RHF branch - UHF branch| = "
              f"{diff:.2e}: {'OK' if ok else 'FAIL'}")

    # --- 2. Open-shell finite-field dipole sum rule (OH doublet)
    mol2 = gto.M(atom='O 0 0 0; H 0 0 0.970', basis='6-31g', spin=1, verbose=0)
    mf0 = make_field_uhf(mol2, 0.0)
    ao_dip_z = mol2.intor_symmetric('int1e_r', comp=3)[2]

    h = 1.0e-3
    mf_p, mf_m = make_field_uhf(mol2, h), make_field_uhf(mol2, -h)

    # harness sanity: HF-only sum rule before trusting the MP2 comparison
    mu_hf_analytic = -np.einsum('pq,qp->', mf0.make_rdm1()[0] + mf0.make_rdm1()[1],
                                ao_dip_z)
    mu_hf_numeric = -(mf_p.e_tot - mf_m.e_tot) / (2 * h)
    diff = abs(mu_hf_analytic - mu_hf_numeric)
    ok = diff < 1e-5
    all_ok &= ok
    print(f"OH UHF-only harness sanity: analytic mu_z={mu_hf_analytic:.8f} vs "
          f"finite-field {mu_hf_numeric:.8f} (diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    e_p = mf_p.e_tot + mp.UMP2(mf_p).run(verbose=0).e_corr
    e_m = mf_m.e_tot + mp.UMP2(mf_m).run(verbose=0).e_corr
    mu_numeric = -(e_p - e_m) / (2 * h)

    dm_relaxed = relaxed_mp2_dm_ao_uhf(mf0, mol2, relax=True)
    mu_relaxed = -np.einsum('pq,qp->', dm_relaxed, ao_dip_z)
    diff = abs(mu_relaxed - mu_numeric)
    ok = diff < 1e-5
    all_ok &= ok
    print(f"OH UMP2 relaxed: analytic mu_z={mu_relaxed:.8f} vs finite-field "
          f"{mu_numeric:.8f} (diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    # discriminator: the unrelaxed density must NOT satisfy the sum rule
    dm_unrelaxed = relaxed_mp2_dm_ao_uhf(mf0, mol2, relax=False)
    mu_unrelaxed = -np.einsum('pq,qp->', dm_unrelaxed, ao_dip_z)
    diff = abs(mu_unrelaxed - mu_numeric)
    ok = diff > 1e-3
    all_ok &= ok
    print(f"OH UMP2 unrelaxed disagrees with finite-field (diff={diff:.2e}, "
          f"expected large): {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
