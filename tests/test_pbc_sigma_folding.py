"""Supercell folding of the periodic correlation self-energy Sigma_c.

THE GAP THIS FILLS. Otherwise the periodic Sigma_c has no machine-precision
oracle at nk > 1: it is validated by reducing EXACTLY to the molecular
SelfEnergySolver at nk=1, plus one 1x1x2 comparison with krgw_ac at the
analytic-continuation level (test_pbc_self_energy.py). An nk=1 oracle is blind
to ANY wrong power of nkpts -- the factor is 1 -- and a single nk=2 comparison
barely constrains nk-scaling. That matters because Sigma_c is observed to grow
strongly with k-density (3D LiH 0.44 -> 1.87 from 2x2x2 to 3x3x3), and the
q=Gamma sampling explanation for that growth was refuted -- which is part of
why the long-wavelength limit is treated in the KERNEL (pbc_rpa_damping,
pbc_wav) rather than by a special q-grid.

Supercell folding is the standard periodic-code oracle (test_pbc_bse.py applies
it to the BSE): an N-point k-mesh on
the primitive cell must reproduce the N-fold supercell solved at Gamma. It is an
nk > 1 test with an exact expected answer, so unlike every existing Sigma_c check
it CAN see a wrong nkpts scaling.

Sigma_nn is an intensive diagonal matrix element, so no volume factor enters:
the folded states pair up one-to-one by orbital energy, and each pair's Sigma_c
-- evaluated at its own orbital energy, which folding makes equal -- must agree.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto, scf

from src.SingleReference.Periodic.pbc_integrals import PBCDFIntegrals
from src.SingleReference.Periodic import pbc_self_energy as pse


def sigma_table(cell, kmesh):
    """(eps, Sigma_c(eps)) for every (k, orbital), sorted by orbital energy."""
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    dfints = PBCDFIntegrals.from_scf(cell, mf)
    e = np.asarray(mf.mo_energy)
    eig = pse.solve_rpa_all_q(dfints, mo_energy=e)

    rows = []
    for kn in range(dfints.nkpts):
        for n in range(dfints.nmo):
            w = e[kn][n].real
            s = float(pse.sigma_c_diag(dfints, eig, kn, n, w, mo_energy=e))
            rows.append((w, s))
    rows.sort(key=lambda r: r[0])
    return np.array(rows)


if __name__ == '__main__':
    all_ok = True

    # Same primitive/supercell pair as the validated BSE folding test.
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

    tp = sigma_table(prim, [1, 1, 2])
    ts = sigma_table(sup, [1, 1, 1])

    n = min(len(tp), len(ts))
    d_eps = np.abs(tp[:n, 0] - ts[:n, 0]).max()
    ok_eps = d_eps < 1e-6
    print(f"1. orbital energies fold (1x1x2 primitive vs 2x supercell Gamma): "
          f"maxerr={d_eps:.2e} {'OK' if ok_eps else 'FAIL'}")
    all_ok &= ok_eps

    d_sig = np.abs(tp[:n, 1] - ts[:n, 1]).max()
    ok_sig = d_sig < 1e-6
    print(f"2. Sigma_c folds (same pairing, evaluated at each state's own eps): "
          f"maxerr={d_sig:.2e} {'OK' if ok_sig else 'FAIL'}")
    all_ok &= ok_sig

    if not ok_sig:
        print("\n   state-by-state (eps, Sigma_c primitive, Sigma_c supercell, diff):")
        for i in range(n):
            print(f"     {tp[i,0]: .6f}  {tp[i,1]: .6f}  {ts[i,1]: .6f}  "
                  f"{tp[i,1]-ts[i,1]: .6f}")
        ratio = tp[:n, 1] / np.where(np.abs(ts[:n, 1]) > 1e-12, ts[:n, 1], np.nan)
        print(f"   primitive/supercell ratio: mean={np.nanmean(ratio):.4f} "
              f"(a clean constant would point at an nkpts power)")

    print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
