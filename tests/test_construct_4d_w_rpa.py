import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf.pyscf_interface import get_orbital_energies, get_two_electron_integrals_chemist
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW.qp_energy import calc_qp_energy

if __name__ == '__main__':
    all_ok = True

    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g')
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2

    eps = get_orbital_energies(mf, representation='spatial')
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted', eta=1e-3)

    W_singlet, W_triplet = lr.construct_4d_w_rpa(nocc)
    norb = len(eps)
    ok = W_singlet.shape == (norb, norb, norb, norb) and W_triplet is W_singlet
    all_ok &= ok
    print(f"construct_4d_w_rpa shape/restricted-triplet-aliasing: {'OK' if ok else 'FAIL'}")

    ok = np.allclose(W_singlet, W_singlet.transpose(1, 0, 3, 2), atol=1e-8)
    all_ok &= ok
    print(f"construct_4d_w_rpa has chemist 8-fold-adjacent (pq|rs)=(qp|rs) symmetry: {'OK' if ok else 'FAIL'}")

    ok = np.all(np.isfinite(W_singlet))
    all_ok &= ok
    print(f"construct_4d_w_rpa is finite: {'OK' if ok else 'FAIL'}")

    # End-to-end: the full-ERI (df=False, exercises construct_4d_w_rpa via
    # calc_qp_energy) and DF (df=True) routes for a vertex-corrected method
    # should agree closely for this tiny basis.
    qp_full = calc_qp_energy(mf, selfenergy='PSD1', polarizability='BSE', df=False, state='homo')
    mf_df = scf.RHF(mol).density_fit()
    from pyscf import df as pyscf_df
    mf_df.with_df.auxbasis = pyscf_df.make_auxbasis(mol)
    mf_df.run()
    qp_df = calc_qp_energy(mf_df, selfenergy='PSD1', polarizability='BSE', df=True, state='homo')
    ok = abs(qp_full - qp_df) < 0.05
    all_ok &= ok
    print(f"full-ERI vs DF PSD1 QP energies agree to <50 meV: {'OK' if ok else 'FAIL'} "
          f"(full={qp_full:.6f} eV, df={qp_df:.6f} eV, diff={(qp_full-qp_df)*1000:.2f} meV)")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
