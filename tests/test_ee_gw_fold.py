"""Eq 66 of Monino and Loos 2023 (BSE@GW, one doubles set) as fold level 'gw'.

Checks, water (RHF, DF factors), basis per check:

  1. build_pieces_gw: M is the gf2 builder's A^HF plus eq 70 on the aaaa and bbbb
     blocks only; D has shape (no, nv, nm); Vt is V's transpose.
  2. fold against unfolded (6-31G, TDA and RPA screening): every folded root, singlet,
     triplet and spin=None, is an eigenvalue of the dense spin-orbital supermatrix
     [[M, V^T], [V, D]] assembled from the same pieces, to 3e-10 Ha (1e-8 eV), and its
     T1 is that eigenvector's singles weight to 1e-8.
  3. the fold and the builder leave D, W and the input vector untouched; an unknown
     screening raises ValueError, and so do gw pieces without dnorm2; the Davidson
     branch equals the dense one (6-31G, TDA, 1e-10 Ha and 1e-8 on T1).
  4. eq 66 transcribed from QuAcK's loops (RGW_phBSE_upfolded_sym.f90 lines 86-216
     at QuAcK 2236bfc) on CasidaSolver's modes, both screenings: M per channel and
     the full spectrum to 1e-10 Ha; this pins sqrt(2), 1/2, the signs, the spin, and
     QPqb's normalisation.
  5. with TDA screening and without eq 70, the spectrum equals Bintrim and
     Berkelbach's H~ (bse_upfolded.build_hamiltonian_familiar), STO-3G, 1e-10 Ha.
  6. QuAcK's RGW_phBSE_upfolded_sym on H2O/cc-pVDZ at QuAcK's geometry, four-index
     HF, exact factors: S1, T1 and their 1h1p weight, TDA_W on and off, to 1e-5 eV and
     1e-5 (QuAcK master 2236bfc, github.com/pfloos/QuAcK, its upfolded phBSE switched
     on in RGW_phBSE; the printed values are QUACK_UPF below).
  7. H2CO / STO-3G, TDA screening, where the fold reorders the seeds of M past
     nroots: the singlet's 3 and the triplet's 4 lowest roots equal the lowest
     eigenvalues of check 4's supermatrix below min D, on both branches, to 1e-9 Ha,
     without a warning.

Run: python tests/test_ee_gw_fold.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.SingleReference.ADC.eeADC import ee_fold, ee_gw_pieces, ee_r_sigma_df
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def water(basis='6-31g', exact=False):
    """(mf, eps, B, no): RHF water with DF factors B[Q, p, q] in the MO basis."""
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf, exact=exact).B_aa
    return mf, eps, B, mol.nelectron // 2


def dense_supermatrix(P):
    """[[M, V^T], [V, D]] over the full spin-orbital singles (aa then bb) and the
    doubles (the aa then the bb block of (k, c, m)), from the pieces by unit
    vectors. Returns (H, ns), ns the number of singles."""
    no, nv = P['no'], P['nv']
    m0, _, diag_s, _, _ = ee_fold.folded_operator(P, None, spin=None)
    ns = diag_s.size
    M = ee_fold.dense_effective(m0, ns)
    cols = []
    for k in range(ns):
        Y = P['V'](ee_fold.singles_flat_to_sb(np.eye(ns)[:, k], no, nv))
        cols.append(np.concatenate([Y.get('aa').ravel(), Y.get('bb').ravel()]))
    V = np.column_stack(cols)
    D = np.concatenate([P['D'].ravel(), P['D'].ravel()])
    return np.block([[M, V.T], [V, np.diag(D)]]), ns


def check_pieces(eps, B, no):
    """M = A^HF + eq 70 on aaaa and bbbb only; D's shape; Vt is V's transpose."""
    ok = True
    nv = len(eps) - no
    P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening='tda')
    _, _, _, gf2 = ee_r_sigma_df.build_operator(eps, B, no, level='gf2', pieces=True)
    Abar = ee_gw_pieces.static_term(eps, P['W'], P['omega'], no)
    d_same = max(float(np.max(np.abs(P['M'].get(k) - gf2['M'].get(k) - Abar)))
                 for k in ('aaaa', 'bbbb'))
    d_cross = max(float(np.max(np.abs(P['M'].get(k) - gf2['M'].get(k))))
                  for k in ('aabb', 'bbaa'))
    ok &= check(d_same < 1e-14 and d_cross < 1e-14,
                'M = A^HF + eq 70 on the same-spin blocks, A^HF across',
                f'{d_same:.1e}, {d_cross:.1e}')
    nm = P['omega'].size
    ok &= check(P['D'].shape == (no, nv, nm) and P['level'] == 'gw',
                "D has shape (no, nv, nm); level 'gw'", str(P['D'].shape))
    H, ns = dense_supermatrix(P)
    rng = np.random.default_rng(7)
    Yr = SB({s: rng.standard_normal(P['D'].shape) for s in ('aa', 'bb')})
    wt = ee_fold.singles_sb_to_flat(P['Vt'](Yr), no, nv)
    yf = np.concatenate([Yr.get('aa').ravel(), Yr.get('bb').ravel()])
    d = float(np.max(np.abs(wt - H[:ns, ns:] @ yf)))
    ok &= check(d < 1e-12, 'Vt is the transpose of V', f'{d:.1e}')
    return ok


def check_fold_vs_unfolded(eps, B, no):
    """Every folded root is an eigenpair of the dense supermatrix of its pieces."""
    ok = True
    for screening in ('tda', 'rpa'):
        P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening)
        H, ns = dense_supermatrix(P)
        w, v = np.linalg.eigh(H)
        for spin in ('singlet', 'triplet', None):
            res = ee_fold.solve_folded(P, 3, spin=spin, tol_omega=1e-12,
                                       tol_residual=1e-10)
            dw, dt = [], []
            for r in range(res.omega.size):
                k = int(np.argmin(np.abs(w - res.omega[r])))
                dw.append(abs(w[k] - res.omega[r]))
                dt.append(abs(float(v[:ns, k] @ v[:ns, k]) - res.t1[r]))
            ok &= check(max(dw) < 3e-10 and max(dt) < 1e-8
                        and bool(res.converged.all()),
                        f'{screening} {spin}: fold roots and T1 are unfolded '
                        'eigenpairs', f'max |dw| {max(dw):.1e} Ha, '
                        f'max |dT1| {max(dt):.1e}')
    return ok


def check_davidson_branch(eps, B, no):
    """The Davidson branch (dense_limit=0) equals the dense one at level gw: the
    only branch a production size reaches, which the test size never does."""
    ok = True
    P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening='tda')
    for spin in ('singlet', 'triplet'):
        kw = dict(spin=spin, tol_omega=1e-10, tol_residual=1e-9)
        a = ee_fold.solve_folded(P, 3, dense_limit=2000, **kw)
        b = ee_fold.solve_folded(P, 3, dense_limit=0, **kw)
        dw = float(np.max(np.abs(a.omega[:3] - b.omega[:3])))
        dt = float(np.max(np.abs(a.t1[:3] - b.t1[:3])))
        ok &= check(dw < 1e-10 and dt < 1e-8 and bool(b.converged.all()),
                    f'tda {spin}: the Davidson branch equals the dense one',
                    f'max |dw| {dw:.1e} Ha, max |dT1| {dt:.1e}')
    return ok


def check_boundaries(eps, B, no):
    """Inputs untouched; an unknown screening refused."""
    ok = True
    P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening='tda')
    D0, W0 = P['D'].copy(), P['W'].copy()
    matvec, _, diag_s, _, _ = ee_fold.folded_operator(P, 0.3, spin='singlet')
    u = np.linspace(-1.0, 1.0, diag_s.size)
    u0 = u.copy()
    matvec(u)
    ee_fold.solve_folded(P, 1, spin='singlet')
    ok &= check(np.array_equal(u, u0) and np.array_equal(P['D'], D0)
                and np.array_equal(P['W'], W0),
                'matvec and solve_folded leave u, D and W untouched')
    try:
        ee_gw_pieces.build_pieces_gw(eps, B, no, screening='bogus')
        ok &= check(False, 'an unknown screening raises ValueError')
    except ValueError as exc:
        ok &= check('screening' in str(exc), 'an unknown screening raises ValueError',
                    str(exc)[:60])
    # the adc2 flat layout reads none of the (k, c, m) blocks: T1 would be 1
    Q = {k: v for k, v in P.items() if k != 'dnorm2'}
    try:
        ee_fold.folded_operator(Q, 0.3, spin='singlet')
        ok &= check(False, "gw pieces without dnorm2 raise ValueError")
    except ValueError as exc:
        ok &= check('dnorm2' in str(exc), "gw pieces without dnorm2 raise ValueError",
                    str(exc)[:60])
    return ok


def quack_supermatrix(eps, B, no, spin, screening):
    """Eq 66 spin-adapted, transcribed from QuAcK's RGW_phBSE_upfolded_sym.f90
    (lines 86-216) and RGW_excitation_density.f90 (lines 40-50), on CasidaSolver's
    modes renormalised to (X - Y)^T (X + Y) = 1. QuAcK's ERI(p,q,r,s) is
    <pq|rs> = (pr|qs); eri here is (pq|rs)."""
    from src.SingleReference.LinearResponse.casida import CasidaSolver
    norb = len(eps)
    nv = norb - no
    o, v = slice(0, no), slice(no, norb)
    eri = np.einsum('Qpq,Qrs->pqrs', B, B, optimize=True)
    eo, ev = eps[o], eps[v]
    ns = no * nv
    # phRLR on eHF, singlet direct RPA: A = de + 2 (ia|jb), B = 2 (ia|jb) or 0
    iajb = eri[o, v, o, v].reshape(ns, ns)
    A = np.diag((ev[None, :] - eo[:, None]).ravel()) + 2.0 * iajb
    Bw = np.zeros_like(A) if screening == 'tda' else 2.0 * iajb
    Om, X, Y = CasidaSolver(A, Bw).solve(tda=(screening == 'tda'))
    XpY = (X + Y) / np.sqrt(np.einsum('Im,Im->m', X - Y, X + Y))[None, :]
    nm = Om.size
    # rho(p,q,m) = sum_jb ERI(p,j,q,b) XpY(jb,m) = sum_jb (pq|jb) XpY(jb,m)
    rho = np.einsum('pqI,Im->pqm', eri[:, :, o, v].reshape(norb, norb, ns), XpY,
                    optimize=True)
    H = np.zeros((ns + ns * nm, ns + ns * nm))
    kappa = 2.0 if spin == 'singlet' else 0.0
    for i in range(no):
        for a in range(nv):
            ia = i * nv + a
            for j in range(no):
                for b in range(nv):
                    jb = j * nv + b
                    # (eHF(a) - eHF(i)) d_ij d_ab + kappa ERI(i,b,a,j) - ERI(i,b,j,a)
                    h = kappa * eri[i, no + a, j, no + b] - eri[i, j, no + a, no + b]
                    if i == j and a == b:
                        h += ev[a] - eo[i]
                    if i == j:
                        # + rho(a,k,m) rho(b,k,m) [1/(e_a - e_k + Om) + 1/(e_b - ...)]
                        rr = rho[no + a, o, :] * rho[no + b, o, :]
                        h += np.sum(rr / (ev[a] - eo[:, None] + Om[None, :])
                                    + rr / (ev[b] - eo[:, None] + Om[None, :]))
                    if a == b:
                        # - rho(i,c,m) rho(j,c,m) [1/(e_i - e_c - Om) + 1/(e_j - ...)]
                        rr = rho[i, v, :] * rho[j, v, :]
                        h -= np.sum(rr / (eo[i] - ev[:, None] - Om[None, :])
                                    + rr / (eo[j] - ev[:, None] - Om[None, :]))
                    H[ia, jb] = h
            # Jph + Kph = - sqrt(2) d_ac rho(i,k,m) + sqrt(2) d_ik rho(a,c,m)
            blk = np.zeros((no, nv, nm))
            blk[:, a, :] -= np.sqrt(2.0) * rho[i, o, :]
            blk[i, :, :] += np.sqrt(2.0) * rho[no + a, v, :]
            H[ia, ns:] = blk.ravel()
            H[ns:, ia] = blk.ravel()
    # C2h2p(iam) = Om(m) + eHF(a) - eHF(i)
    H[ns:, ns:] = np.diag((Om[None, None, :] + ev[None, :, None]
                           - eo[:, None, None]).ravel())
    return H


def check_eq66_transcription(eps, B, no):
    """The pieces against QuAcK's loops, with modes from CasidaSolver: the singles
    block element by element per channel, and the full spectrum."""
    ok = True
    nv = len(eps) - no
    ns = no * nv
    for screening in ('tda', 'rpa'):
        P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening)
        Hq = {s: quack_supermatrix(eps, B, no, s, screening)
              for s in ('singlet', 'triplet')}
        Ma, Mx = P['M'].get('aaaa'), P['M'].get('aabb')
        for spin, sgn in (('singlet', 1.0), ('triplet', -1.0)):
            Mch = (Ma + sgn * Mx).reshape(ns, ns)
            d = float(np.max(np.abs(Mch - Hq[spin][:ns, :ns])))
            ok &= check(d < 1e-10, f'{screening} {spin}: M equals QuAcK singles block',
                        f'max |d| {d:.1e} Ha')
        H, _ = dense_supermatrix(P)
        ref = np.sort(np.concatenate([np.linalg.eigvalsh(Hq[s]) for s in Hq]))
        got = np.linalg.eigvalsh(H)
        d = float(np.max(np.abs(got - ref))) if got.size == ref.size else np.inf
        ok &= check(d < 1e-10, f'{screening}: the full spectrum equals QuAcK form',
                    f'max |d| {d:.1e} Ha over {ref.size}')
    return ok


def check_bintrim_berkelbach():
    """Eq 66 minus eq 70 with TDA screening is Bintrim and Berkelbach's symmetric H~
    (eq 12, BSE.bse_upfolded.build_hamiltonian_familiar) with its inner pair rotated
    onto the modes: the two spectra agree. Water / STO-3G."""
    from src.SingleReference.BSE import bse_upfolded
    ok = True
    _, eps, B, no = water('sto-3g')
    norb = len(eps)
    P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening='tda')
    _, _, _, gf2 = ee_r_sigma_df.build_operator(eps, B, no, level='gf2', pieces=True)
    H, _ = dense_supermatrix(dict(P, M=gf2['M']))
    eri = np.einsum('Qpq,Qrs->pqrs', B, B, optimize=True)
    gb = bse_upfolded.eri_blocks(eri, no, norb)
    ref = np.sort(np.concatenate([np.linalg.eigvalsh(
        bse_upfolded.build_hamiltonian_familiar(eps, gb, no, norb - no, spin=s))
        for s in ('singlet', 'triplet')]))
    got = np.linalg.eigvalsh(H)
    d = float(np.max(np.abs(got - ref))) if got.size == ref.size else np.inf
    ok &= check(d < 1e-10, "tda pieces without eq 70 have H~'s spectrum",
                f'max |d| {d:.1e} Ha over {ref.size}')
    return ok


QUACK_H2O = 'O 0.0000 0.0000 0.0000; H 0.7571 0.0000 0.5861; H -0.7571 0.0000 0.5861'
# (screening, spin): (omega eV, Z) of QuAcK's lowest 1h1p-dominated upfolded root
QUACK_UPF = {('tda', 'singlet'): (8.427847, 0.964662),
             ('tda', 'triplet'): (7.632712, 0.966154),
             ('rpa', 'singlet'): (8.632125, 0.971539),
             ('rpa', 'triplet'): (7.804563, 0.972781)}


def check_quack_h2o():
    """Check 6: the fold against QuAcK's printed roots, 1e-5 eV and 1e-5 on
    T1 (QuAcK prints 1e-6)."""
    mol = gto.M(atom=QUACK_H2O, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    no = mol.nelectron // 2
    B = DFIntegrals.from_scf(mol, mf, exact=True).B_aa
    ok = True
    for screening in ('tda', 'rpa'):
        P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening)
        for spin in ('singlet', 'triplet'):
            w_q, z_q = QUACK_UPF[(screening, spin)]
            res = ee_fold.solve_folded(P, 1, spin=spin, tol_omega=1e-10)
            d = res.omega[0] * HARTREE_TO_EV - w_q
            ok &= check(abs(d) < 1e-5, f'{screening} W, {spin}: omega = QuAcK',
                        f'{d:+.1e} eV')
            ok &= check(abs(res.t1[0] - z_q) < 1e-5, f'{screening} W, {spin}: '
                        'T1 = QuAcK Z', f'{res.t1[0] - z_q:+.1e}')
    return ok


H2CO = 'C 0 0 0; O 0 0 1.205; H 0 0.943 -0.587; H 0 -0.943 -0.587'


def check_seeding_gap():
    """H2CO / STO-3G, TDA screening: the fold pulls a state of M from above its
    seeds below them (singlet at nroots 3, triplet at nroots 4), so the lowest roots
    come from the count check; they equal the lowest eigenvalues of the channel's
    QuAcK-form supermatrix below min D."""
    ok = True
    mol = gto.M(atom=H2CO, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    no = mol.nelectron // 2
    P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening='tda')
    dmin = float(np.min(P['D']))
    for spin, k in (('singlet', 3), ('triplet', 4)):
        w = np.linalg.eigvalsh(quack_supermatrix(eps, B, no, spin, 'tda'))
        w = w[w < dmin][:k]
        for dense_limit, route in ((2000, 'dense'), (0, 'Davidson')):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                res = ee_fold.solve_folded(P, k, spin=spin, tol_omega=1e-10,
                                           tol_residual=1e-9, dense_limit=dense_limit)
            d = float(np.max(np.abs(res.omega[:k] - w))) if res.omega.size >= k \
                else np.inf
            ok &= check(d < 1e-9 and not caught,
                        f'{route} {spin}: the {k} lowest roots equal the supermatrix',
                        f'max |dw| {d:.1e} Ha, nout {res.omega.size}, '
                        f'{len(caught)} warning(s)')
    return ok


def main():
    """Run every check; print ALL PASSED or FAILURES DETECTED; exit 0 or 1."""
    all_ok = True
    mf, eps, B, no = water('6-31g')
    all_ok &= check_pieces(eps, B, no)
    all_ok &= check_fold_vs_unfolded(eps, B, no)
    all_ok &= check_davidson_branch(eps, B, no)
    all_ok &= check_eq66_transcription(eps, B, no)
    all_ok &= check_bintrim_berkelbach()
    all_ok &= check_quack_h2o()
    all_ok &= check_boundaries(eps, B, no)
    all_ok &= check_seeding_gap()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
