"""Validation of the periodic GW transition amplitudes chi_a and chi_b.

At the Gamma-only (nk=1, q=0) limit the periodic chi_a must reduce exactly to
the molecular GW.transition_amplitudes.AmplitudeGenerator.get_chi_a fed the same
DF factor and the same Casida eigenvectors -- an external check against the
trusted molecular code. (The end-to-end oracle for the momentum-resolved
amplitude is the plain-GW self-energy vs krgw_ac, test_pbc_self_energy.py.)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto, scf

from src.SingleReference.Periodic.pbc_integrals import PBCDFIntegrals
from src.SingleReference.Periodic import pbc_casida as pc
from src.SingleReference.Periodic import pbc_amplitudes as pa
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.GW.transition_amplitudes import AmplitudeGenerator
from src.SingleReference.base import get_occ_virt_indices


class _MolAmp(AmplitudeGenerator):
    """Minimal molecular AmplitudeGenerator harness (restricted, DF)."""
    def __init__(self, eps, df_coeff):
        self.spin_mode = 'restricted'
        self.eps = eps
        self.df_coeff = df_coeff
        self.naux = df_coeff.shape[0]
        self.block_size = 10000

    def _get_occ_virt_indices(self, eps, nocc):
        return get_occ_virt_indices(eps, nocc)


def _pbc_setup(atom, basis, a, kmesh):
    from src.SingleReference.Periodic import pbc_casida as pc
    cell = gto.Cell()
    cell.atom = atom
    cell.basis = basis
    cell.pseudo = 'gth-pade'
    cell.a = a
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.kernel()
    dfints = PBCDFIntegrals.from_scf(cell, mf)
    return cell, mf, dfints


if __name__ == '__main__':
    all_ok = True

    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 1.2; He 1.5 0 0.6'
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = np.diag([4.0, 4.0, 3.6])
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([1, 1, 1])
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.kernel()
    dfints = PBCDFIntegrals.from_scf(cell, mf)
    e = np.asarray(mf.mo_energy)
    nocc = dfints.nocc[0]
    L = dfints.L[0][0].real
    eps = e[0].real

    # RPA eigenvectors at q=0
    kcon = dfints.kconserv_pair[0]
    A, B = pc.build_rpa_matrices(dfints, kcon, mo_energy=e)
    Omega, X, Y = CasidaSolver(A, B).solve()
    X = X.real
    Y = Y.real

    # periodic chi_a (q=0, target kp=0)
    rho = pa.project_XpY(dfints, kcon, X, Y)
    chi_a_pbc = pa.get_chi_a(dfints, 0, rho, 0)          # (nstates, nmo, nmo)

    # molecular oracle
    mol = _MolAmp(eps, L)
    chi_a_mol = mol.get_chi_a(nocc, X, Y)                # (nstates, norb, norb)

    d = np.abs(chi_a_pbc - chi_a_mol).max()
    ok = d < 1e-10
    print(f"1. chi_a q=0 vs molecular AmplitudeGenerator.get_chi_a: maxerr={d:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- chi_b vertex q=0 vs molecular get_chi_b_vertex ----
    W_all = pc.build_static_W_all_Q(dfints, mo_energy=e)
    A_bse, B_bse = pc.build_bse_matrices(dfints, 0, W_all, mo_energy=e)
    Om_b, Xb, Yb = CasidaSolver(A_bse, B_bse).solve()
    Xb, Yb = Xb.real, Yb.real
    chi_b_pbc = pa.get_chi_b_vertex(dfints, 0, W_all, Xb, Yb, 0)
    chi_b_mol = mol.get_chi_b_vertex(nocc, Xb, Yb, eri_w=W_all[0].real)
    db = np.abs(chi_b_pbc - chi_b_mol).max()
    ok = db < 1e-10
    print(f"1b. chi_b vertex q=0 vs molecular get_chi_b_vertex: maxerr={db:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # structural: momentum helper round-trips (km with kconserv_pair[q,km]=kp)
    nk_prim = 3
    prim = gto.Cell()
    prim.atom = 'H 0 0 0; H 0.9 0.9 0.9'
    prim.basis = 'gth-szv'
    prim.pseudo = 'gth-pade'
    prim.a = np.eye(3) * 2.5
    prim.verbose = 0
    prim.build()
    kpts3 = prim.make_kpts([1, 1, nk_prim])
    mf3 = scf.KRHF(prim, kpts3).density_fit()
    mf3.kernel()
    dfints3 = PBCDFIntegrals.from_scf(prim, mf3)
    map_ok = True
    for q in range(nk_prim):
        for kp in range(nk_prim):
            km = pa.kpoint_minus_q(dfints3, q, kp)
            if dfints3.kconserv_pair[q, km] != kp:
                map_ok = False
    print(f"2. k-q momentum helper consistent for all (q,kp): {'OK' if map_ok else 'FAIL'}")
    all_ok &= map_ok

    # ---- finite-q smoke test: chi_a and chi_b run for all (q, kp), finite, right shape ----
    W_all3 = pc.build_static_W_all_Q(dfints3, mo_energy=np.asarray(mf3.mo_energy))
    e3 = np.asarray(mf3.mo_energy)
    nmo3 = dfints3.nmo
    smoke_ok = True
    for q in range(nk_prim):
        A3, B3 = pc.build_bse_matrices(dfints3, q, W_all3, mo_energy=e3)
        Om3, X3, Y3 = CasidaSolver(A3, B3).solve()
        rho3 = pa.project_XpY(dfints3, dfints3.kconserv_pair[q], X3, Y3)
        for kp in range(nk_prim):
            ca = pa.get_chi_a(dfints3, q, rho3, kp)
            cb = pa.get_chi_b_vertex(dfints3, q, W_all3, X3, Y3, kp)
            if ca.shape != (len(Om3), nmo3, nmo3) or cb.shape != (len(Om3), nmo3, nmo3):
                smoke_ok = False
            if not (np.all(np.isfinite(ca)) and np.all(np.isfinite(cb))):
                smoke_ok = False
    print(f"3. chi_a/chi_b finite-q run+shape+finite for all (q,kp) on 1x1x3: "
          f"{'OK' if smoke_ok else 'FAIL'}  (end-to-end finite-q oracle: test_pbc_self_energy)")
    all_ok &= smoke_ok

    print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
