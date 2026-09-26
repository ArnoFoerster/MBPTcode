"""Validation of the periodic RPA polarizability + complex CasidaSolver.

Oracles:
 1. build_chi0_aux (RPA polarizability Pi0^q in the aux basis) vs
    pyscf.pbc.gw.krgw_ac.get_rho_response, all momentum transfers, several
    frequencies -- the trustworthy EXTERNAL finite-q check of the momentum
    bookkeeping and the direct-Coulomb contraction. (pyscf's tdscf get_ab is
    buggy at finite kshift -- it disagrees with its own gen_vind -- so it is
    deliberately NOT used as an oracle here.)
 2. build_rpa_matrices structure: A, B Hermitian; A - B diagonal and positive
    (Kresse: guaranteed for RPA); the complex CasidaSolver returns a real,
    positive spectrum at every q.
 3. Solver correctness on periodic RPA blocks: the squaring-trick spectrum
    equals a direct generalized-eigenproblem solve of the same Kresse blocks
    [[A,B],[B,A]] v = Omega [[I,0],[0,-I]] v (scipy.linalg.eig), all q.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg as sla
from pyscf.pbc import gto, scf
from pyscf.pbc.gw import krgw_ac
from pyscf.pbc.gw.krgw_ac import get_rho_response
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos

from src.SingleReference.Periodic.pbc_integrals import PBCDFIntegrals
from src.SingleReference.Periodic.pbc_casida import build_chi0_aux, build_rpa_matrices
from src.SingleReference.LinearResponse.casida import CasidaSolver


def _generalized_eig_spectrum(A, B):
    """Positive eigenvalues of [[A,B],[B,A]] v = w [[I,0],[0,-I]] v."""
    n = A.shape[0]
    H = np.block([[A, B], [B, A]])
    S = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), -np.eye(n)]])
    w = sla.eig(H, S, right=False)
    w = w.real[np.abs(w.imag) < 1e-8]
    return np.sort(w[w > 1e-9])


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
    mo = np.asarray(mf.mo_coeff)
    nmo = mo.shape[-1]
    nao = cell.nao_nr()
    mo_energy = np.asarray(mf.mo_energy)
    kscaled = cell.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    dfints = PBCDFIntegrals.from_scf(cell, mf)
    gw = krgw_ac.KRGWAC(mf)

    def build_Lij_pyscf(ki, kj):
        Lpq = []
        for R, I, s in mf.with_df.sr_loop([kpts[ki], kpts[kj]], compact=False):
            Lpq.append(R + I * 1j)
        Lpq = np.vstack(Lpq).reshape(-1, nao * nao)
        moij, ijslice = _conc_mos(mo[ki], mo[kj])[2:]
        return _ao2mo.r_e2(Lpq, moij, ijslice, [], None).reshape(-1, nmo, nmo)

    # ---- 1. Pi0 vs pyscf get_rho_response, all q, several frequencies ----
    max_pi = 0.0
    for kL in range(nk):
        kidx = np.zeros(nk, dtype=int)
        Lij = []
        for i in range(nk):
            for j in range(nk):
                kc = -kscaled[i] + kscaled[j] + kscaled[kL]
                if np.linalg.norm(np.round(kc) - kc) < 1e-8:
                    kidx[i] = j
            Lij.append(build_Lij_pyscf(i, kidx[i]))
        Lij = np.asarray(Lij)
        for omega in (0.0, 0.5, 2.0):
            Pi_ref = get_rho_response(gw, omega, mo_energy, Lij, kL, kidx)
            Pi_mine = build_chi0_aux(dfints, kidx, omega, mo_energy=mo_energy, imaginary=True)
            max_pi = max(max_pi, np.abs(Pi_mine - Pi_ref).max())
    ok = max_pi < 1e-10
    print(f"1. Pi0^q vs pyscf get_rho_response (all q, 3 freqs): maxerr={max_pi:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- 2. Kresse RPA blocks: structure + real positive spectrum ----
    herm = 0.0
    amb_min = np.inf
    spec_ok = True
    for q in range(nk):
        kconserv = dfints.kconserv_pair[q]
        A, B = build_rpa_matrices(dfints, kconserv, mo_energy=mo_energy)
        herm = max(herm, np.abs(A - A.conj().T).max(), np.abs(B - B.conj().T).max())
        AmB = A - B
        offdiag = np.abs(AmB - np.diag(np.diag(AmB))).max()
        amb_min = min(amb_min, np.diag(AmB).real.min())
        omega, X, Y = CasidaSolver(A, B).solve()
        if omega.min() < -1e-8 or np.abs(omega.imag).max() > 1e-8:
            spec_ok = False
        if offdiag > 1e-10:
            spec_ok = False
    ok = herm < 1e-10 and amb_min > 0 and spec_ok
    print(f"2. Kresse RPA blocks Hermitian={herm:.1e}, A-B diagonal & min={amb_min:.3f}>0, "
          f"real+ spectrum: {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- 3. Squaring-trick spectrum vs direct generalized eig, all q ----
    max_solver = 0.0
    for q in range(nk):
        kconserv = dfints.kconserv_pair[q]
        A, B = build_rpa_matrices(dfints, kconserv, mo_energy=mo_energy)
        omega, _, _ = CasidaSolver(A, B).solve()
        w_ref = _generalized_eig_spectrum(A, B)
        w = np.sort(omega.real)
        n = min(len(w), len(w_ref))
        max_solver = max(max_solver, np.abs(w[:n] - w_ref[:n]).max())
    ok = max_solver < 1e-9
    print(f"3. CasidaSolver squaring-trick vs generalized eig (all q): maxerr={max_solver:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
