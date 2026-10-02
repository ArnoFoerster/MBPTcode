"""Particle-particle RPA and its Riccati (ladder-CCD) form.

The ppRPA spectrum is not symmetric: n_vv positive-norm N+2 energies and n_oo
negative-norm N-2 energies, solved here as a shifted Hermitian pencil. The
checks tie that solve to three things it could get wrong silently:

- the pencil against a direct non-Hermitian eig of eta*H (inertia, energies,
  norms), and the two sectors sharing ONE amplitude matrix, T_rem = T_add^T;
- the Riccati amplitudes (Newton, and the DF-streamed spin-free ladder CCD
  that the production Faddeev-ADC(3) route uses) against T = Y X^{-1} from
  the eigenvectors, and the three expressions of the ppRPA correlation
  energy, sum Omega^+ - Tr A = -sum Omega^- - Tr C = Tr(B T);
- the closed-shell singlet/triplet matrices against the spin-orbital ones:
  singlet plus three copies of the triplet is the spin-orbital spectrum.

Run: python tests/test_pp_rpa.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg as la
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_antisymmetrized_spin_eri
from src.SingleReference.ADC.adc_r_faddeev import (
    _alpha_beta_ladder, solve_ladder_amplitudes_restricted)
from src.SingleReference.LinearResponse.pp_rpa import (
    PPRPASolver, build_pprpa_matrices, build_pprpa_matrices_restricted)
from src.SingleReference.LinearResponse.riccati import (
    amplitudes_from_phonons, solve_riccati)


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build():
    """H2O/6-31G with a DF mean field; the dense tensor is the DF one, so the
    spin-orbital and restricted builds see identical integrals."""
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    B_aa = DFIntegrals.from_scf(mol, mf).B_aa
    eri = np.einsum('Qpq,Qrs->pqrs', B_aa, B_aa, optimize=True)
    return mol, mf.mo_energy, B_aa, eri


def check_pencil_against_direct_eig(eps, eri, nocc):
    g = get_antisymmetrized_spin_eri(eri)
    es = np.repeat(eps, 2)
    A, B, C = build_pprpa_matrices(es, g, 2 * nocc)
    solver = PPRPASolver(A, B, C)
    r = solver.solve()
    H, eta = solver.hamiltonian(), solver.metric()
    w, V = la.eig(eta[:, None] * H)
    order = np.argsort(w.real)
    w, V = w.real[order], V.real[:, order]
    norm = np.einsum('i,ij,ij->j', eta, V, V)
    ok = check(np.count_nonzero(norm > 0) == len(r.omega_add),
               'n_vv positive-norm solutions', f'{len(r.omega_add)} N+2, '
               f'{len(r.omega_rem)} N-2')
    d_add = np.abs(np.sort(w[norm > 0]) - r.omega_add).max()
    d_rem = np.abs(np.sort(w[norm < 0]) - r.omega_rem).max()
    ok &= check(max(d_add, d_rem) < 1e-10, 'pencil energies == eig(eta H)',
                f'max dev {max(d_add, d_rem):.1e}')
    n_add = np.abs(r.X_add.T @ r.X_add - r.Y_add.T @ r.Y_add
                   - np.eye(len(r.omega_add))).max()
    n_rem = np.abs(r.Y_rem.T @ r.Y_rem - r.X_rem.T @ r.X_rem
                   - np.eye(len(r.omega_rem))).max()
    ok &= check(max(n_add, n_rem) < 1e-10, 'X^T X - Y^T Y = +1 / -1',
                f'{max(n_add, n_rem):.1e}')
    T = amplitudes_from_phonons(r.X_add, r.Y_add)
    d = np.abs(r.X_rem @ la.inv(r.Y_rem) - T.T).max()
    ok &= check(d < 1e-10, 'N-2 amplitudes are the N+2 ones transposed', f'{d:.1e}')
    Tn, info = solve_riccati(A, B, C)
    d = np.abs(Tn - T).max()
    ok &= check(d < 1e-9, 'Newton Riccati T == Y X^{-1}',
                f'{d:.1e}, {info["niter"]} steps')
    e1 = solver.correlation_energy(r)
    e2 = -np.sum(r.omega_rem) - np.trace(C)
    e3 = np.sum(B * T.T)
    d = max(abs(e1 - e2), abs(e1 - e3))
    ok &= check(d < 1e-9, 'E_c: sum Omega+ - Tr A = -sum Omega- - Tr C = Tr(B T)',
                f'E_c = {e1:.10f}, spread {d:.1e}')
    rt = solver.solve(tda=True)
    d = np.abs(rt.omega_add - la.eigvalsh(A)).max()
    ok &= check(d < 1e-12 and not rt.Y_add.any(), 'tda=True is eig(A), Y = 0')
    return ok, r, e1


def check_restricted_spectrum(eps, eri, B_aa, nocc, r_so, ec_so):
    ok = True
    res = {}
    for spin in ('singlet', 'triplet'):
        M_df = build_pprpa_matrices_restricted(eps, nocc, spin, B_aa=B_aa)
        M_d = build_pprpa_matrices_restricted(eps, nocc, spin, eri_chemist=eri)
        d = max(np.abs(a - b).max() for a, b in zip(M_df, M_d))
        ok &= check(d < 1e-12, f'{spin}: DF build == dense build', f'{d:.1e}')
        res[spin] = (PPRPASolver(*M_df), PPRPASolver(*M_df).solve())
    add = np.sort(np.r_[res['singlet'][1].omega_add,
                        np.repeat(res['triplet'][1].omega_add, 3)])
    rem = np.sort(np.r_[res['singlet'][1].omega_rem,
                        np.repeat(res['triplet'][1].omega_rem, 3)])
    d = max(np.abs(add - r_so.omega_add).max(), np.abs(rem - r_so.omega_rem).max())
    ok &= check(d < 1e-10, 'singlet + 3 x triplet == spin-orbital spectrum',
                f'{d:.1e}')
    ec = (res['singlet'][0].correlation_energy(res['singlet'][1])
          + 3 * res['triplet'][0].correlation_energy(res['triplet'][1]))
    ok &= check(abs(ec - ec_so) < 1e-10, 'E_c(singlet) + 3 E_c(triplet) == '
                'spin-orbital E_c', f'{abs(ec - ec_so):.1e}')
    return ok, res


def check_df_ladder_amplitudes(eps, eri, B_aa, nocc, res):
    """The production amplitudes: no eigenvectors, no pair-space matrix of
    the virtuals; must be the phonons' T in the alpha-beta pair form."""
    lad = solve_ladder_amplitudes_restricted(eps, nocc, B_aa=B_aa)
    lad_d = solve_ladder_amplitudes_restricted(eps, nocc, eri_chemist=eri)
    ok = check(np.abs(lad.t - lad_d.t).max() < 1e-12, 'DF ladder == dense ladder',
               f"{lad.info['niter']} Jacobi/DIIS iterations")
    pp = {spin: {'T': amplitudes_from_phonons(res[spin][1].X_add, res[spin][1].Y_add)} for spin in res}
    t_ph = _alpha_beta_ladder(pp, nocc, len(eps) - nocc)
    d = np.abs(lad.t - t_ph).max()
    ok &= check(d < 1e-9, 'ladder t == phonon T (singlet+triplet, alpha-beta form)',
                f'{d:.1e}')
    nv = len(eps) - nocc
    # T A through the ladder term against the explicit alpha-beta A
    o, v = slice(0, nocc), slice(nocc, None)
    A_ab = np.einsum('ecfd->efcd', eri[v, v, v, v]).reshape(nv * nv, nv * nv)
    A_ab += np.diag((eps[v][:, None] + eps[v][None, :]).ravel())
    TA = (lad.t.reshape(nocc * nocc, -1) @ A_ab).reshape(lad.t.shape)
    d = np.abs(TA - lad.TA).max()
    ok &= check(d < 1e-12, 'returned T A == t times the alpha-beta A', f'{d:.1e}')
    return ok


def run():
    warnings.simplefilter('ignore')
    mol, eps, B_aa, eri = build()
    nocc = mol.nelectron // 2
    all_ok = True
    print('\n-- spin-orbital ppRPA: pencil, amplitudes, correlation energy')
    ok, r_so, ec_so = check_pencil_against_direct_eig(eps, eri, nocc)
    all_ok &= ok
    print('\n-- closed shell: singlet and triplet channels')
    ok, res = check_restricted_spectrum(eps, eri, B_aa, nocc, r_so, ec_so)
    all_ok &= ok
    print('\n-- the DF-streamed ladder amplitudes')
    all_ok &= check_df_ladder_amplitudes(eps, eri, B_aa, nocc, res)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return all_ok


def test_pp_rpa_and_riccati():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
