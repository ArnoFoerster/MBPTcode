"""Validation: explicit-Coulomb / AUTO-damped DF integrals feeding the full
periodic pipeline (through PSD1).

The damping (Forster et al., JCTC 2025) is Option A -- it modifies the DF
Coulomb metric itself (pbc_damped_integrals.build_dfintegrals_coulG), so every
downstream stage (Pi0, W^q, chi_a, chi_b, plain + vertex self-energy) inherits
it. Checks:
 1. Bare-kernel consistency: the RPA correlation energy from the periodic
    Casida pipeline (plasmon formula, built on the G-space L) equals the independent standalone
    pbc_rpa.ri_rpa_ecorr (logdet formula) to ~1e-14 -- proves the G-space L
    feeds the whole pipeline correctly.
 2. AUTO-damped: the pipeline runs end to end and PSD1 self-energy is finite;
    the damped q=0 treatment shifts the result vs bare (the damping replaces the
    bare q=0/G=0 exxdiv value with a finite, k-grid-dependent one -- so bare and
    damped are NOT expected to coincide; they converge together only as the
    k-grid -> TDL, which is a multi-grid property not checked here).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto, scf

from src.SingleReference.Periodic.pbc_damped_integrals import build_dfintegrals_coulG
from src.SingleReference.Periodic.pbc_rpa import ri_rpa_ecorr
from src.SingleReference.Periodic.pbc_rpa_damping import nyquist_params, make_coulG_damped
from src.SingleReference.Periodic.pbc_casida import build_rpa_matrices, build_static_W_all_Q
from src.SingleReference.Periodic import pbc_self_energy as pse
from src.SingleReference.LinearResponse.casida import CasidaSolver


def rpa_ecorr_plasmon(dfints, mo_energy):
    """RPA E_c = (1/2 Nk) sum_q [ sum_S Omega_S - Tr A^q ] from the Casida blocks."""
    nk = dfints.nkpts
    ec = 0.0
    for q in range(nk):
        A, B = build_rpa_matrices(dfints, dfints.kconserv_pair[q], mo_energy=mo_energy)
        Om, _, _ = CasidaSolver(A, B).solve()
        ec += 0.5 * (np.sum(Om.real) - np.trace(A).real)
    return ec / nk


if __name__ == '__main__':
    all_ok = True

    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 1.1 1.1 1.1'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 3.2
    cell.verbose = 0
    cell.mesh = [15, 15, 15]
    cell.build()
    kmesh = [2, 2, 1]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts, exxdiv='ewald').density_fit()
    mf.kernel()
    e = np.asarray(mf.mo_energy)

    # ---- 1. bare-kernel RPA Ec: Casida plasmon (via G-space L) == ri_rpa_ecorr ----
    df_bare = build_dfintegrals_coulG(mf)
    ec_mine = rpa_ecorr_plasmon(df_bare, e)
    ec_ref = ri_rpa_ecorr(mf, nw=32)
    d = abs(ec_mine - ec_ref)
    ok = d < 1e-10
    print(f"1. RPA Ec: Casida-pipeline (G-space L) vs ri_rpa_ecorr: "
          f"mine={ec_mine:.8f} ref={ec_ref:.8f} diff={d:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- 2. AUTO-damped pipeline runs through PSD1; damping shifts the result ----
    r0, beta, Rc = nyquist_params(cell, kmesh)
    df_damp = build_dfintegrals_coulG(mf, coulG_fn=make_coulG_damped(r0, beta))
    nocc = df_damp.nocc[0]

    def psd1_homo(df):
        W = build_static_W_all_Q(df, mo_energy=e)
        eig = pse.solve_bse_all_q(df, W, mo_energy=e)
        return pse.sigma_vertex_diag(df, eig, W, 0, nocc - 1, e[0][nocc - 1].real,
                                     vertex_mode='PSD1', mo_energy=e)

    s_bare = psd1_homo(df_bare)
    s_damp = psd1_homo(df_damp)
    ok = np.isfinite(s_bare) and np.isfinite(s_damp) and abs(s_bare - s_damp) > 1e-4
    print(f"2. PSD1 Sigma(HOMO) runs bare & AUTO-damped (r0={r0:.2f}): "
          f"bare={s_bare:+.5f} damped={s_damp:+.5f} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- 3. direction-aware Nyquist: r0 grows with the k-grid in SAMPLED dirs ----
    chain = gto.Cell()
    chain.atom = 'H 0 0 0; H 0 0 1.2'
    chain.basis = 'gth-szv'
    chain.pseudo = 'gth-pade'
    chain.a = np.diag([8.0, 8.0, 2.4])   # 8 A vacuum in x,y; periodic in z
    chain.verbose = 0
    chain.build()
    r0s = [nyquist_params(chain, [1, 1, Nz])[0] for Nz in (2, 4, 8, 16)]
    grows = all(r0s[i + 1] > r0s[i] - 1e-9 for i in range(len(r0s) - 1)) and r0s[-1] > 2 * r0s[0]
    # 3D bulk with all directions sampled is unchanged (reduces to old all-dir formula)
    amax = np.array([np.linalg.norm(cell.lattice_vectors()[i]) for i in range(3)])
    r_iso = nyquist_params(cell, [2, 2, 2])[0]
    r_all = 0.5 * 0.5 * (np.array([2, 2, 2]) * amax).min()
    iso_ok = abs(r_iso - r_all) < 1e-9
    ok = grows and iso_ok
    print(f"3. direction-aware Nyquist: chain r0 grows with Nz {[round(x,2) for x in r0s]}; "
          f"3D-bulk unchanged={iso_ok}: {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
