"""Validation of the static/dynamic screened interaction W^q in the aux basis.

The screened interaction is built two independent ways and must agree:
 - inversion:  W = (I - Pi0(omega))^{-1}, Pi0 from build_chi0_aux (itself
   validated == pyscf get_rho_response in test_pbc_casida.py).
 - spectral:   W = I - (4/nkpts) sum_S rho_S rho_S^H Omega/(nu^2+Omega^2), from
   the Kresse RPA eigenpairs.

Agreement at ALL frequencies validates the eigenpairs (values AND vectors)
jointly and locks the -4/nkpts spectral normalization. Also checks the
imaginary-axis vs real-axis forms coincide in the static (omega=0) limit.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto, scf

from src.SingleReference.Periodic.pbc_integrals import PBCDFIntegrals
from src.SingleReference.Periodic.pbc_casida import (
    solve_rpa_spectral, build_W_aux_spectral, build_W_aux_inversion)

if __name__ == '__main__':
    all_ok = True

    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0.9 0.9 0.9'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 2.5
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts([1, 1, 3])
    nk = len(kpts)
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.kernel()
    dfints = PBCDFIntegrals.from_scf(cell, mf)
    e = np.asarray(mf.mo_energy)

    # ---- 1. spectral W == inversion W, all q, several imaginary frequencies ----
    max_w = 0.0
    for q in range(nk):
        kcon = dfints.kconserv_pair[q]
        Omega, rho = solve_rpa_spectral(dfints, kcon, mo_energy=e)
        for nu in (0.0, 0.3, 1.0, 3.0):
            W_spec = build_W_aux_spectral(dfints, Omega, rho, omega=nu, imaginary=True)
            W_inv = build_W_aux_inversion(dfints, kcon, nu, mo_energy=e, imaginary=True)
            max_w = max(max_w, np.abs(W_spec - W_inv).max())
    ok = max_w < 1e-10
    print(f"1. spectral W == inversion W (all q, 4 imag freqs): maxerr={max_w:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- 2. Hermiticity of the static (imaginary-axis) W ----
    max_herm = 0.0
    for q in range(nk):
        kcon = dfints.kconserv_pair[q]
        Omega, rho = solve_rpa_spectral(dfints, kcon, mo_energy=e)
        W = build_W_aux_spectral(dfints, Omega, rho, omega=0.0, imaginary=True)
        max_herm = max(max_herm, np.abs(W - W.conj().T).max())
    ok = max_herm < 1e-10
    print(f"2. static W Hermitian (all q): maxerr={max_herm:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- 3. imaginary-axis and real-axis spectral forms agree at omega=0 ----
    max_static = 0.0
    for q in range(nk):
        kcon = dfints.kconserv_pair[q]
        Omega, rho = solve_rpa_spectral(dfints, kcon, mo_energy=e)
        W_im = build_W_aux_spectral(dfints, Omega, rho, omega=0.0, imaginary=True)
        W_re = build_W_aux_spectral(dfints, Omega, rho, omega=0.0, imaginary=False, eta=0.0)
        max_static = max(max_static, np.abs(W_im - W_re).max())
    ok = max_static < 1e-10
    print(f"3. imag-axis vs real-axis spectral W at omega=0 (all q): maxerr={max_static:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
