import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import (CasidaInstability,
                                                       CasidaSolver)

#: |min eig| below which A + B or A - B counts as singular, Ha: a channel
#: there may solve or be refused. TDHF (Sz=0) on this reference carries a
#: zero mode at -5.8e-8 Ha.
SINGULAR_TOL = 1e-6

if __name__ == '__main__':
    all_ok = True

    # BN, triplet ground state (2 unpaired electrons) -- forces a genuine
    # unrestricted (UHF) reference, needed to exercise the spin-flip
    # (Sz=+-1, "triplet" in the open-shell sense) Casida machinery.
    mol = gto.M(atom='B 0 0 0; N 0 0 1.28', basis='cc-pvdz', spin=2)
    mf = scf.UHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()
    ok = mf.converged
    all_ok &= ok
    print(f"BN/cc-pvdz UHF (spin=2) converged: {'OK' if ok else 'FAIL'}  E={mf.e_tot:.6f}")

    nocc = mf.nelec  # (nocc_a, nocc_b)
    eps = get_orbital_energies(mf, representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    mo_a, mo_b = mf.mo_coeff
    S_AO = mol.intor_symmetric('int1e_ovlp')
    S_ab = mo_a.T @ S_AO @ mo_b
    df_a, df_b = df_coeff
    df_ab = np.einsum('ia, pik -> pka', S_ab, df_a)
    df_coeff = (df_a, df_b, df_ab)

    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='unrestricted', eta=1e-3)
    w_aux = lr.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]

    def check(label, A, B):
        """Positive roots where A + B and A - B are positive definite, a
        CasidaInstability where either is not."""
        lowest_pd = min(np.linalg.eigvalsh(A + B)[0],
                        np.linalg.eigvalsh(A - B)[0])
        try:
            omega, _, _ = CasidaSolver(A, B).solve()
        except CasidaInstability:
            ok = lowest_pd < SINGULAR_TOL
            print(f"{label:32s}: refused, min eig(A +- B) = {lowest_pd:+.3e} Ha  {'OK' if ok else 'FAIL'}")
            return ok
        lowest = np.min(omega)
        ok = (np.all(np.isfinite(omega)) and lowest > 0
              and lowest_pd > -SINGULAR_TOL)
        print(f"{label:32s}: lowest excitation = {lowest:10.6f} Ha  (n_states={len(omega)})  {'OK' if ok else 'FAIL'}")
        return ok

    # Spin-conserving (Sz=0) channel -- for an unrestricted reference this is
    # a single combined alpha/beta problem (no separate singlet/triplet split
    # the way a closed-shell restricted reference has; that spin-adaptation
    # only exists for RHF references).
    all_ok &= check('RPA (Sz=0)', *lr.build_casida_matrices(nocc, lBSE=False))
    all_ok &= check('TDHF (Sz=0, bare exchange)',
                    *lr.build_casida_matrices(nocc, lBSE=True, W_aux=None))
    all_ok &= check('BSE (Sz=0, screened exchange)',
                    *lr.build_casida_matrices(nocc, lBSE=True, W_aux=w_aux))

    # Spin-flip (Sz=+-1, the open-shell analog of a "triplet" channel): one
    # problem coupling ab excitations (occ_alpha -> virt_beta) with ba
    # de-excitations (occ_beta -> virt_alpha) and back. TDHF carries the zero
    # mode of a global spin rotation here.
    all_ok &= check('RPA spin-flip',
                    *lr.build_spin_flip_casida_matrices(nocc, lBSE=False))
    all_ok &= check('TDHF spin-flip',
                    *lr.build_spin_flip_casida_matrices(nocc, lBSE=True, W_aux=None))
    all_ok &= check('BSE spin-flip',
                    *lr.build_spin_flip_casida_matrices(nocc, lBSE=True, W_aux=w_aux))

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
