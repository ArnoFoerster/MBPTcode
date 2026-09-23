import os
import sys
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf_interface import (
    get_orbital_energies, get_density_fitting_coefficients)
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, gram_product)


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}"
          + (f'   ({detail})' if detail else ''))
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


def df_static_screening(eps, coeff, nocc, eta):
    """W = (1 - χ₀)⁻¹ in the auxiliary basis at ω = 0, independent of src:
    χ₀ = 2 C_ov diag(f) C_ovᵀ with f = -2d / (d² + η²)."""
    norb = coeff.shape[1]
    occ, virt = np.arange(nocc), np.arange(nocc, norb)
    C_ov = coeff[:, occ[:, None], virt].reshape(coeff.shape[0], -1)
    d = (eps[virt][None, :] - eps[occ][:, None]).ravel()
    f = -2.0 * d / (d**2 + eta**2)
    return np.linalg.inv(np.eye(coeff.shape[0]) - 2.0 * (C_ov * f) @ C_ov.T)


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
                 ('TDHF triplet', dict(lBSE=True, W_aux=None, triplet=True),
                  0.0, 'bare'),
                 ('BSE singlet', dict(lBSE=True, W_aux=w_aux), 2.0, w_aux),
                 ('BSE triplet', dict(lBSE=True, W_aux=w_aux, triplet=True),
                  0.0, w_aux)]
        for name, kw, factor, W in cases:
            A, B = lr.build_casida_matrices(nocc, **kw)
            A_ref, B_ref = reference_ab(eps, coeff, nocc, factor, W)
            scale = max(1.0, np.max(np.abs(A_ref)))
            dA = np.max(np.abs(A - A_ref)) / scale
            dB = np.max(np.abs(B - B_ref)) / scale
            all_ok &= check(
                dA < 1e-12 and dB < 1e-12,
                f'{label} {name}: A, B vs einsum reference',
                f'dA={dA:.1e} dB={dB:.1e}')
        all_ok &= check(np.array_equal(w_aux, w_copy),
                        f'{label}: W_aux untouched by the builds')

        # --- restricted full-ERI twin: same eps/coeff, chemist eri built
        # from the DF factors ---
        eri = np.einsum('Ppq,Prs->pqrs', coeff, coeff)
        lr_full = LinearResponseSolver(eps, eri_chemist=eri,
                                       spin_mode='restricted')
        full_cases = [
            ('RPA singlet', dict(lBSE=False), 2.0, None),
            ('RPA triplet', dict(lBSE=False, triplet=True), 0.0, None),
            ('TDHF singlet', dict(lBSE=True, W_aux=None), 2.0, 'bare'),
            ('TDHF triplet', dict(lBSE=True, W_aux=None, triplet=True),
             0.0, 'bare'),
        ]
        for name, kw, factor, W in full_cases:
            A, B = lr_full.build_casida_matrices(nocc, **kw)
            A_ref, B_ref = reference_ab(eps, coeff, nocc, factor, W)
            scale = max(1.0, np.max(np.abs(A_ref)))
            dA = np.max(np.abs(A - A_ref)) / scale
            dB = np.max(np.abs(B - B_ref)) / scale
            all_ok &= check(
                dA < 1e-12 and dB < 1e-12,
                f'{label} full {name}: A, B vs einsum reference',
                f'dA={dA:.1e} dB={dB:.1e}')

        # BSE, full ERI: one W_aux reused for singlet then triplet, each
        # compared to a build given a fresh copy taken before either build.
        w_full = lr_full.static_screening_aux(nocc)
        w_full_before = w_full.copy()
        w_full_copy_s = w_full.copy()
        w_full_copy_t = w_full.copy()
        A_s, B_s = lr_full.build_casida_matrices(nocc, lBSE=True, W_aux=w_full)
        A_t, B_t = lr_full.build_casida_matrices(
            nocc, lBSE=True, W_aux=w_full, triplet=True)
        A_s_ref, B_s_ref = lr_full.build_casida_matrices(
            nocc, lBSE=True, W_aux=w_full_copy_s)
        A_t_ref, B_t_ref = lr_full.build_casida_matrices(
            nocc, lBSE=True, W_aux=w_full_copy_t, triplet=True)
        all_ok &= check(
            np.array_equal(A_s, A_s_ref) and np.array_equal(B_s, B_s_ref),
            f'{label} full BSE singlet: reused vs fresh-copy W_aux')
        all_ok &= check(
            np.array_equal(A_t, A_t_ref) and np.array_equal(B_t, B_t_ref),
            f'{label} full BSE triplet: reused vs fresh-copy W_aux')
        all_ok &= check(np.array_equal(w_full, w_full_before),
                        f'{label} full: W_aux untouched by the builds')

        # BSE, full ERI, values. With eri = C·C the push-through identity
        # C_ovᵀ (1 - C_ov χ₀ C_ovᵀ)⁻¹ C_ov = V (1 - χ₀ V)⁻¹ makes the full-ERI
        # kernel equal to the DF one, so the DF einsum reference applies.
        # B takes W_aux from the caller (imaginary axis, η = 0). A screens its
        # direct term inside the builder at ω = 0 on the real axis, which
        # carries the solver's η, at the eps_screen gaps when those are given.
        w_swap_df = df_static_screening(eps, coeff, nocc, 0.0)
        w_dir_df = df_static_screening(eps, coeff, nocc, lr_full.eta)
        norb = len(eps)
        eps_qp = (eps + np.where(np.arange(norb) < nocc, -0.03, 0.05)
                  + 1e-3 * np.arange(norb))
        lr_full_qp = LinearResponseSolver(eps_qp, eri_chemist=eri,
                                          spin_mode='restricted')
        value_cases = [
            ('BSE', lr_full, eps, dict()),
            ('BSE@QP with eps_screen', lr_full_qp, eps_qp, dict(eps_screen=eps)),
        ]
        for name, solver, eps_diag, extra in value_cases:
            for spin, triplet, factor in (('singlet', False, 2.0),
                                          ('triplet', True, 0.0)):
                A, B = solver.build_casida_matrices(
                    nocc, lBSE=True, W_aux=w_full.copy(), triplet=triplet,
                    **extra)
                A_ref = reference_ab(eps_diag, coeff, nocc, factor, w_dir_df)[0]
                B_ref = reference_ab(eps_diag, coeff, nocc, factor, w_swap_df)[1]
                scale = max(1.0, np.max(np.abs(A_ref)))
                dA = np.max(np.abs(A - A_ref)) / scale
                dB = np.max(np.abs(B - B_ref)) / scale
                all_ok &= check(
                    dA < 1e-12 and dB < 1e-12,
                    f'{label} full {name} {spin}: A, B vs DF einsum reference',
                    f'dA={dA:.1e} dB={dB:.1e}')

    # --- gram_product against the einsum reference, in ragged row blocks ---
    rng = np.random.default_rng(1)
    C = rng.standard_normal((5, 37))
    V_ref = np.einsum('Pi,Pj->ij', C, C)
    for block_elems in (64, 100, 2**27):
        V = gram_product(C, block_elems=block_elems)
        dV = np.abs(V - V_ref).max()
        all_ok &= check(dV < 1e-13 and V.flags.c_contiguous,
                        f'gram_product block_elems={block_elems}: C^T C by row blocks',
                        f'dV={dV:.1e}')

    # --- tracemalloc ratchet on the BSE-DF build, N_pair = 2100, in units
    # of one N_pair^2 array ---
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
    RATCHET = 4.3
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    A, B = lr.build_casida_matrices(nocc, lBSE=True, W_aux=Wm)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    over = (peak - base) / unit
    all_ok &= check(over <= RATCHET,
                    f'BSE-DF build peak <= {RATCHET} arrays',
                    f'{over:.2f} arrays')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
