import os
import sys
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def reference_ab(eps, coeff, nocc, factor, W):
    """A, B from the DF factors by einsum: A = diag(d) + f V - W_dir, B = f V - W_swap,
    V[ia,jb] = sum_P C[P,i,a] C[P,j,b], W_dir[ia,jb] = sum_PQ C[P,i,j] W[P,Q] C[Q,a,b],
    W_swap[ia,jb] = sum_PQ C[P,i,b] W[P,Q] C[Q,j,a]. W=None: no exchange terms (RPA);
    W='bare': W = identity (TDHF)."""
    norb = coeff.shape[1]
    occ, virt = np.arange(nocc), np.arange(nocc, norb)
    n_pair = len(occ) * len(virt)
    C_ov = coeff[:, occ[:, None], virt]
    d = (eps[virt][None, :] - eps[occ][:, None]).ravel()
    V = np.einsum('Pia,Pjb->iajb', C_ov, C_ov).reshape(n_pair, n_pair)
    A = np.diag(d) + factor * V
    B = factor * V
    if W is None:
        return A, B
    Wm = np.eye(coeff.shape[0]) if isinstance(W, str) else W
    C_oo = coeff[:, occ[:, None], occ]
    C_vv = coeff[:, virt[:, None], virt]
    W_dir = np.einsum('Pij,PQ,Qab->iajb', C_oo, Wm, C_vv).reshape(n_pair, n_pair)
    W_swap = np.einsum('Pib,PQ,Qja->iajb', C_ov, Wm, C_ov).reshape(n_pair, n_pair)
    return A - W_dir, B - W_swap


if __name__ == '__main__':
    all_ok = True
    systems = [('HF/6-31g', 'H 0 0 0; F 0 0 0.9', '6-31g'),
               ('H2/6-31g (nocc=1)', 'H 0 0 0; H 0 0 0.74', '6-31g'),
               ('HF/sto-3g (nvirt=1)', 'H 0 0 0; F 0 0 0.9', 'sto-3g')]
    for label, atom, basis in systems:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).density_fit()
        mf.with_df.auxbasis = df.make_auxbasis(mol)
        mf.run()
        eps = get_orbital_energies(mf, representation='spatial')
        coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        nocc = mol.nelectron // 2
        lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted')
        w_aux = lr.static_screening_aux(nocc)
        w_copy = w_aux.copy()
        cases = [('RPA singlet', dict(lBSE=False), 2.0, None),
                 ('RPA triplet', dict(lBSE=False, triplet=True), 0.0, None),
                 ('TDHF singlet', dict(lBSE=True, W_aux=None), 2.0, 'bare'),
                 ('TDHF triplet', dict(lBSE=True, W_aux=None, triplet=True), 0.0, 'bare'),
                 ('BSE singlet', dict(lBSE=True, W_aux=w_aux), 2.0, w_aux),
                 ('BSE triplet', dict(lBSE=True, W_aux=w_aux, triplet=True), 0.0, w_aux)]
        for name, kw, factor, W in cases:
            A, B = lr.build_casida_matrices(nocc, **kw)
            A_ref, B_ref = reference_ab(eps, coeff, nocc, factor, W)
            scale = max(1.0, np.max(np.abs(A_ref)))
            dA = np.max(np.abs(A - A_ref)) / scale
            dB = np.max(np.abs(B - B_ref)) / scale
            all_ok &= check(dA < 1e-12 and dB < 1e-12, f'{label} {name}: A, B vs einsum reference',
                            f'dA={dA:.1e} dB={dB:.1e}')
        all_ok &= check(np.array_equal(w_aux, w_copy), f'{label}: W_aux untouched by the builds')

    # --- tracemalloc ratchet on the BSE-DF build, N_pair = 2100, in units of one N_pair^2 array ---
    rng = np.random.default_rng(0)
    naux, norb, nocc = 400, 100, 30
    n_pair = nocc * (norb - nocc)
    coeff = rng.standard_normal((naux, norb, norb))
    coeff = coeff + coeff.transpose(0, 2, 1)
    eps = np.sort(rng.uniform(-1.0, 1.0, norb))
    Wm = rng.standard_normal((naux, naux)) * 0.01
    Wm = Wm @ Wm.T + np.eye(naux)
    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted')
    unit = 8 * n_pair * n_pair
    RATCHET = 4.3                                 # pinned in Task 2 step 4
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    A, B = lr.build_casida_matrices(nocc, lBSE=True, W_aux=Wm)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    over = (peak - base) / unit
    all_ok &= check(over <= RATCHET, f'BSE-DF build peak <= {RATCHET} arrays', f'{over:.2f} arrays')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
