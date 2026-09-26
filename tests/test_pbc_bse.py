"""Validation of the full (non-TDA) BSE per momentum transfer q.

Two independent oracles, both required:
 1. q=0 limit vs the molecular LinearResponseSolver BSE (lBSE) fed the same DF
    factor and static W -- validates the PHYSICS of A and B (which screened
    terms, correct at q=0 where momentum is trivial).
 2. Supercell folding: an N-point k-mesh on the primitive cell, unioned over q,
    must reproduce the N-fold supercell solved at Gamma -- validates the finite-q
    MOMENTUM bookkeeping, including B's antiresonant time-reversal (Kresse
    Eq. 58). This is the non-negotiable periodic-code oracle; the q=0 check
    alone cannot catch a finite-q momentum error.

Also checks A, B Hermitian and A-B positive definite (Kresse structural
requirements) at every q.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto, scf

from src.SingleReference.Periodic.pbc_integrals import PBCDFIntegrals
from src.SingleReference.Periodic import pbc_casida as pc
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver


def bse_spectrum(cell, kmesh):
    kpts = cell.make_kpts(kmesh)
    nk = len(kpts)
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.kernel()
    dfints = PBCDFIntegrals.from_scf(cell, mf)
    e = np.asarray(mf.mo_energy)
    W_all = pc.build_static_W_all_Q(dfints, mo_energy=e)
    allom = []
    herm = 0.0
    pd_ok = True
    for q in range(nk):
        A, B = pc.build_bse_matrices(dfints, q, W_all, mo_energy=e)
        herm = max(herm, np.abs(A - A.conj().T).max(), np.abs(B - B.conj().T).max())
        if np.linalg.eigvalsh(A - B).min() <= 0:
            pd_ok = False
        allom.append(CasidaSolver(A, B).solve()[0].real)
    return np.sort(np.concatenate(allom)), herm, pd_ok


if __name__ == '__main__':
    all_ok = True

    # ---- 1. q=0 vs molecular BSE (bigger basis so nvirt > 1 distinguishes B) ----
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
    W_all = pc.build_static_W_all_Q(dfints, mo_energy=e)

    lr = LinearResponseSolver(eps, coeff_df=L, spin_mode='restricted', eta=1e-3)
    Ar, Br = lr.build_casida_matrices(nocc, lBSE=False)
    omr, Xr, Yr = CasidaSolver(Ar, Br).solve()
    Waux = lr.solve_rpa_spectral_df([0.0], nocc, omr, Xr + Yr, is_imaginary=True)[0]
    Am, Bm = lr.build_casida_matrices(nocc, lBSE=True, W_aux=Waux)
    sp_mol = np.sort(CasidaSolver(Am, Bm).solve()[0].real)

    A, B = pc.build_bse_matrices(dfints, 0, W_all, mo_energy=e)
    sp_pbc = np.sort(CasidaSolver(A, B).solve()[0].real)
    d = np.abs(sp_pbc - sp_mol).max()
    ok = d < 1e-8
    print(f"1. q=0 full BSE vs molecular LinearResponseSolver (nvirt={dfints.nmo-nocc}): "
          f"maxerr={d:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- 2. supercell folding of the full non-TDA BSE ----
    prim = gto.Cell()
    prim.atom = 'H 0 0 0; H 0 0 1.2'
    prim.basis = 'gth-szv'
    prim.pseudo = 'gth-pade'
    prim.a = np.diag([3.0, 3.0, 2.4])
    prim.verbose = 0
    prim.build()
    sup = gto.Cell()
    sup.atom = 'H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6'
    sup.basis = 'gth-szv'
    sup.pseudo = 'gth-pade'
    sup.a = np.diag([3.0, 3.0, 4.8])
    sup.verbose = 0
    sup.build()

    sp_p, herm_p, pd_p = bse_spectrum(prim, [1, 1, 2])
    sp_s, herm_s, pd_s = bse_spectrum(sup, [1, 1, 1])
    n = min(len(sp_p), len(sp_s))
    dfold = np.abs(sp_p[:n] - sp_s[:n]).max()
    ok = dfold < 1e-8
    print(f"2. full BSE supercell folding (1x1x2 union-q vs 2x-supercell Gamma): "
          f"maxerr={dfold:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- 3. structural: A,B Hermitian, A-B PD ----
    ok = max(herm_p, herm_s) < 1e-10 and pd_p and pd_s
    print(f"3. A,B Hermitian (max={max(herm_p, herm_s):.1e}) and A-B positive definite: "
          f"{'OK' if ok else 'FAIL'}")
    all_ok &= ok

    print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
