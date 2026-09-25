"""Static second-order GW kernel (BSE2@GW) on the DF Casida route.

`static_second_order_kernel_df` returns the Theta^GW blocks of Monino and Loos,
J. Chem. Phys. 159, 034105 (2023), eq 72, to add to the BSE@GW Casida matrices.
H2O/cc-pVDZ at QuAcK's geometry, HF start, exact three-index factors (no with_df,
so the eigh/Cholesky factorisation of the full ERI tensor). Four checks:

1. The DF contraction equals a dense transcription of the six terms on the norb^4
   W tensor built from the same factors, at the default block size and at one that
   leaves a partial last block over c.
2. With QuAcK's G0W0@HF energies on the diagonal (graphical roots, full-RPA W,
   eta = 0) the lowest singlet and triplet of BSE@GW and BSE2@GW, full and TDA,
   reproduce QuAcK master 2236bfc, whose RGW_phBSE2_static_kernel_A/B is added to
   both manifolds with the HF energies in the denominators: run
   2026-09-25_quack-xcheck-h2o on WSL, files full_bse, tda_fullw_bse, full_bse2,
   tda_fullw_bse2 (same-input match to 1e-6 eV in
   2026-09-25_h2o-bse2-vs-quack-sameqp).
3. Shifting the HF energies by -0.3 Ha puts the LUMO below zero, so a
   particle-particle denominator eps_c + eps_d changes sign inside its sum: the
   routine warns once, and stays silent on the unshifted set.
4. H2/STO-3G, nocc = nvirt = 1: the size-one block shape against the dense form.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, static_second_order_kernel_df)

# QuAcK mol/H2O.xyz, Angstrom
GEOM = 'O 0.0000 0.0000 0.0000; H 0.7571 0.0000 0.5861; H -0.7571 0.0000 0.5861'
# QuAcK G0W0@RHF e_QP, eV, all 24 orbitals (full_bse.log of the run above)
QUACK_QP_EV = np.array([
    -547.096903359, -33.376695376, -18.558315280, -14.436803319, -12.158825764,
    4.708293937, 6.656989854, 20.360279319, 21.783506682, 30.406097434,
    31.317121452, 32.403102704, 38.226762239, 39.069729540, 44.458840280,
    49.498548274, 50.876928601, 65.959485567, 67.738625861, 88.744413957,
    90.503059309, 95.157864951, 103.217832913, 112.841177294])
# (S1, T1) in eV per (kernel, tda), the static rows of the QuAcK summaries
QUACK_EV = {('bse', False): (8.450045, 7.664144), ('bse', True): (8.484507, 7.698575),
            ('bse2', False): (8.736268, 7.802672), ('bse2', True): (8.759333, 8.010908)}
TOL_EV = 1e-5


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}"
          + (f'   ({detail})' if detail else ''))
    return bool(ok)


def dense_theta(eps, coeff, W_aux, nocc):
    """The six terms of the Notes of static_second_order_kernel_df on the dense
    W_pq,rs = sum_PQ B_P,pq W_PQ B_Q,rs, as (n_pair, n_pair) blocks in (ia, jb)."""
    W = np.einsum('Ppq,PQ,Qrs->pqrs', coeff, W_aux, coeff)
    o, v = slice(0, nocc), slice(nocc, None)
    e = eps
    d_kc = 1.0 / (e[v][None, :] - e[o][:, None])
    s_kl = 1.0 / (e[o][:, None] + e[o][None, :])
    s_cd = 1.0 / (e[v][:, None] + e[v][None, :])
    A = (4.0 * np.einsum('ijkc,kc,abkc->iajb', W[o, o, o, v], d_kc, W[v, v, o, v])
         + 2.0 * np.einsum('akjl,kl,kilb->iajb', W[v, o, o, o], s_kl, W[o, o, o, v])
         - 2.0 * np.einsum('acjd,cd,cidb->iajb', W[v, v, o, v], s_cd, W[v, o, v, v]))
    B = (4.0 * np.einsum('ibkc,kc,ajkc->iajb', W[o, v, o, v], d_kc, W[v, o, o, v])
         + 2.0 * np.einsum('akbl,kl,kilj->iajb', W[v, o, v, o], s_kl, W[o, o, o, o])
         - 2.0 * np.einsum('acbd,cd,cidj->iajb', W[v, v, v, v], s_cd, W[v, o, v, o]))
    n_pair = A.shape[0] * A.shape[1]
    return A.reshape(n_pair, n_pair), B.reshape(n_pair, n_pair)


if __name__ == '__main__':
    mol = gto.M(atom=GEOM, basis='cc-pvdz', unit='Angstrom', verbose=0)
    nocc = mol.nelectron // 2
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    eps_hf = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    w_aux = LinearResponseSolver(eps_hf, coeff_df=coeff,
                                 spin_mode='restricted').static_screening_aux(nocc)
    all_ok = True

    # 1. DF contraction against the dense transcription, two block sizes
    A_ref, B_ref = dense_theta(eps_hf, coeff, w_aux, nocc)
    scale = max(np.abs(A_ref).max(), np.abs(B_ref).max())
    for blksize in (8, 4):
        A_df, B_df = static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc,
                                                   blksize=blksize)
        dev = max(np.abs(A_df - A_ref).max(), np.abs(B_df - B_ref).max()) / scale
        all_ok &= check(dev < 1e-10, f'DF Theta^A/Theta^B vs dense, blksize={blksize}',
                        f'max rel dev {dev:.1e}')

    # 2. BSE@GW and BSE2@GW energies on QuAcK's QP energies vs QuAcK
    eps_qp = QUACK_QP_EV / HARTREE_TO_EV
    theta = static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc)
    lr = LinearResponseSolver(eps_qp, coeff_df=coeff, spin_mode='restricted')
    for kernel in ('bse', 'bse2'):
        for tda in (False, True):
            got = []
            for triplet in (False, True):
                A, B = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w_aux,
                                                triplet=triplet)
                if kernel == 'bse2':
                    A, B = A + theta[0], B + theta[1]
                got.append(float(CasidaSolver(A, B).solve(tda=tda)[0].min())
                           * HARTREE_TO_EV)
            ref = QUACK_EV[(kernel, tda)]
            dev = max(abs(got[0] - ref[0]), abs(got[1] - ref[1]))
            all_ok &= check(dev < TOL_EV,
                            f"{kernel}@G0W0 {'TDA' if tda else 'full'} S1/T1 vs QuAcK",
                            f'{got[0]:.6f}/{got[1]:.6f} eV, dev {dev * 1e6:.1f} ueV')

    # 3. Energies not measured from the chemical potential: a ladder denominator
    #    changes sign inside its sum and the routine says so; the HF set is silent.
    for shift, expect in ((0.0, 0), (-0.3, 1)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            static_second_order_kernel_df(eps_hf + shift, coeff, w_aux, nocc)
        hits = [w for w in caught if 'changes sign' in str(w.message)]
        all_ok &= check(len(hits) == expect,
                        f'sign-change warning with eps shifted by {shift} Ha',
                        f'{len(hits)} warning(s), expected {expect}')

    # 4. The degenerate block shape, nocc = nvirt = 1 (H2/STO-3G): every reshape
    #    and transpose of the routine on size-one axes, against the dense form.
    mol1 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', unit='Angstrom',
                 verbose=0)
    mf1 = scf.RHF(mol1)
    mf1.conv_tol = 1e-10
    mf1.kernel()
    eps1 = np.asarray(get_orbital_energies(mf1, representation='spatial'),
                      float)
    coeff1 = get_density_fitting_coefficients(mol1, mf1, representation='spatial')
    w1 = LinearResponseSolver(eps1, coeff_df=coeff1,
                              spin_mode='restricted').static_screening_aux(1)
    A1, B1 = static_second_order_kernel_df(eps1, coeff1, w1, 1)
    A1_ref, B1_ref = dense_theta(eps1, coeff1, w1, 1)
    dev = max(abs(A1 - A1_ref).max(), abs(B1 - B1_ref).max())
    all_ok &= check(A1.shape == (1, 1) and dev < 1e-12 * max(abs(A1_ref).max(), 1.0),
                    'nocc = nvirt = 1 block shape vs dense', f'abs dev {dev:.1e}')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
