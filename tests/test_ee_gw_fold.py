"""Eq 66 of Monino and Loos 2023 (BSE@GW, one doubles set) as fold level 'gw'.

Checks, water (RHF, DF factors), basis per check:

  1. build_pieces_gw: M is the gf2 builder's A^HF plus eq 70 on the aaaa and bbbb
     blocks only; D has shape (no, nv, nm); W is C-contiguous; Vt is V's transpose.
  2. fold against unfolded (6-31G, TDA and RPA screening): every folded root, singlet,
     triplet and spin=None, is an eigenvalue of the dense spin-orbital supermatrix
     [[M, V^T], [V, D]] assembled from the same pieces, to 3e-10 Ha (1e-8 eV), and its
     T1 is that eigenvector's singles weight to 1e-8.
  3. the fold and the builder leave D, W and the input vector untouched; an unknown
     screening raises ValueError, and so do gw pieces without dnorm2; the Davidson
     branch equals the dense one (6-31G, TDA, 1e-10 Ha and 1e-8 on T1); on the
     (k, c, m) layout dmatvec(u, vector=True) is dA_eff/domega u (its quadratic
     form -dmatvec(u) to 1e-12 relative, the central difference to 1e-6).
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
  8. build_pieces_gw's pieces['K'], the kernel streamed over slabs of the pair's
     occupied index k (6-31G, TDA and RPA screening), of the W pieces and of the
     df=True pieces (kernel_df, through B and rho), equals the V, divide, Vt path
     at 0.3 Ha for powers (1, 2) and () to 1e-12 relative, its norm dnorm2's, on one
     slab, on three of width 2 that the budget _STREAM_GB sets, and on those three
     with two in flight; the folded singlets with and without it agree to 1e-9 Ha;
     a singles vector with a flip parity raises NotImplementedError; a slab holds
     at most _K_LIVE slab blocks per slab index (the slope of tracemalloc's peak
     from width 1 to 2, water cc-pVTZ, TDA).
  9. build_pieces_gw(..., laplace=True) (6-31G, TDA and RPA screening): W is None;
     M, and Vt (D - ω)^-1 V and its norm through the DF forms of V and Vt, equal the
     W pieces' to 1e-12 relative; pieces['K'], kernel_laplace, equals the W pieces'
     V, divide, Vt at 0.3, -0.2 and 0.05 Ha and 3e-3 Ha below min D for powers
     (1, 2) and its norm to 1e-8 relative, on the grid minimax_points_for_accuracy
     gives for relative error _LAPLACE_TARGET (1e-9); the folded singlets and
     triplets equal the W pieces' to 1e-6 eV; 1e-7 Ha below min D, where
     no tabulated grid reaches the target, and above min D, K is kernel_df's exact
     kernel (1e-12 relative); a flip parity raises NotImplementedError;
     gw_modes_df holds at most 6 (nm, nm) float64 arrays at once (traced peak,
     water cc-pVTZ, both screenings); static_term_df in slabs of three modes, on
     one and on three workers, equals its one-slab form to 1e-12 relative.

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
    W = P['W']
    ok &= check(W.flags.c_contiguous
                and np.shares_memory(W[no:, no:, :].reshape(nv, nv * nm), W),
                "W is C-contiguous, so the couplings reshape W's slices without a copy")
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
    matvec, dmv, diag_s, _, _ = ee_fold.folded_operator(P, 0.3, spin='singlet')
    u = np.linspace(-1.0, 1.0, diag_s.size)
    u0 = u.copy()
    matvec(u)
    ee_fold.solve_folded(P, 1, spin='singlet')
    ok &= check(np.array_equal(u, u0) and np.array_equal(P['D'], D0)
                and np.array_equal(P['W'], W0),
                'matvec and solve_folded leave u, D and W untouched')
    # the (k, c, m) layout: the vector derivative's quadratic form is the dnorm2 scalar
    h = 1e-4
    g = dmv(u, vector=True)
    fd = (ee_fold.folded_operator(P, 0.3 + h, spin='singlet')[0](u)
          - ee_fold.folded_operator(P, 0.3 - h, spin='singlet')[0](u)) / (2 * h)
    d1 = abs(float(u @ g) + dmv(u)) / abs(dmv(u))
    d2 = float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))
    ok &= check(d1 < 1e-12 and d2 < 1e-6,
                'gw: dmatvec(u, vector=True) is dA_eff/domega u', f'{d1:.1e}, {d2:.1e}')
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


def traced_peak(fn):
    """Bytes tracemalloc saw allocated at once while fn() ran, above its start."""
    import tracemalloc
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


def check_streamed_kernel(eps, B, no):
    """pieces['K'], the eq 66 kernel streamed over slabs of k, of the W pieces and of
    the df pieces (kernel_df), against the same pieces' V, divide, Vt path at omega
    0.3 Ha, powers (1, 2) and (), both screenings, to 1e-12 relative per block, its
    norm against dnorm2; on one slab, on three of width 2, and on those three two at
    a time; the folded singlets with and without K to 1e-9 Ha; a y1 with a flip
    parity refused; at most _K_LIVE slab blocks per slab index."""
    from src.SingleReference.ADC.eeADC import ee_spin_blocks as sb
    ok = True
    nv = len(eps) - no
    om = 0.3
    rng = np.random.default_rng(13)
    y1 = SB({s: rng.standard_normal((no, nv)) for s in ('aa', 'bb')})

    def rel(a, b):
        return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))

    live = ee_gw_pieces._K_LIVE
    for screening, df in (('tda', False), ('rpa', False), ('tda', True),
                          ('rpa', True)):
        tag = f'{screening}, df' if df else screening
        P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening, df=df)
        be, nm = P['be'], P['omega'].size
        Yt = be.divide(P['V'](y1), P['D'] - om)
        ref = [P['Vt'](Yt), P['Vt'](be.divide(Yt, P['D'] - om))]
        n2_ref = P['dnorm2'](Yt)
        per = nv * nm

        def budget(fit):
            # the _STREAM_GB that holds `fit` indices k in flight
            return (fit + 0.5) * 8 * per * live / 1e9

        runs = (('one slab', sb._STREAM_GB, 1), ('3 slabs', budget(2), 1),
                ('3 slabs, 2 workers', budget(4), 2))
        for label, gb, workers in runs:
            keep_gb, keep_w = sb._STREAM_GB, sb._stream_workers
            sb._STREAM_GB, sb._stream_workers = gb, (lambda w=workers: w)
            try:
                plan = sb._slab_plan(no, per, live)
                ws, n2 = P['K'](y1, om, (1, 2))
                ws0, n20 = P['K'](y1, om, ())
            finally:
                sb._STREAM_GB, sb._stream_workers = keep_gb, keep_w
            d = max(rel(w.get(s), r.get(s)) for w, r in zip(ws, ref)
                    for s in ('aa', 'bb'))
            dn = max(abs(n2 - n2_ref), abs(n20 - n2_ref)) / n2_ref
            ok &= check(len(ws) == 2 and ws0 == [] and d < 1e-12 and dn < 1e-12,
                        f"{tag}, {label}: pieces['K'] equals V, divide, Vt",
                        f'{d:.1e}, norm {dn:.1e} relative, (width, workers) {plan}')
        a = ee_fold.solve_folded(P, 3, spin='singlet', dense_limit=0)
        b = ee_fold.solve_folded(dict(P, K=None), 3, spin='singlet', dense_limit=0)
        dw = float(np.max(np.abs(a.omega - b.omega)))
        ok &= check(a.omega.size == 3 and dw < 1e-9,
                    f"{tag}: the folded singlets with and without pieces['K']",
                    f'max |dw| {dw:.1e} Ha')
        try:
            P['K'](SB(y1.blocks, parity=1.0), om, (1,))
            ok &= check(False, f"{tag}: pieces['K'] refuses a flip parity")
        except NotImplementedError as exc:
            ok &= check('parity' in str(exc),
                        f"{tag}: pieces['K'] refuses a flip parity",
                        str(exc)[:60])
    # slab blocks alive per slab index: the slope of K's traced peak from width 1 to
    # width 2 on water cc-pVTZ, where a block outgrows NumPy's ufunc buffers and the
    # fixed overhead cancels
    _, e3, B3, n3 = water('cc-pvtz')
    v3 = len(e3) - n3
    y3 = SB({s: rng.standard_normal((n3, v3)) for s in ('aa', 'bb')})
    for df in (False, True):
        P3 = ee_gw_pieces.build_pieces_gw(e3, B3, n3, screening='tda', df=df)
        per3 = v3 * P3['omega'].size
        held = {}
        for width in (1, 2):
            keep_gb, keep_w = sb._STREAM_GB, sb._stream_workers
            sb._STREAM_GB = (width + 0.5) * 8 * per3 * live / 1e9
            sb._stream_workers = lambda: 1
            try:
                held[width] = traced_peak(lambda: P3['K'](y3, om, (1, 2)))
            finally:
                sb._STREAM_GB, sb._stream_workers = keep_gb, keep_w
        slope = (held[2] - held[1]) / (8 * per3)
        ok &= check(slope <= live,
                    f"{'df: ' if df else ''}a slab of pieces['K'] holds at most "
                    f'_K_LIVE = {live} slab blocks per slab index',
                    f'{slope:.2f}, water cc-pVTZ, tda')
    return ok


def check_laplace_kernel(eps, B, no):
    """build_pieces_gw(..., laplace=True) against the W pieces, both screenings: W
    None; M and the DF V and Vt to 1e-12 relative; kernel_laplace at omega 0.3,
    -0.2 and 0.05 Ha and 3e-3 Ha below min D, powers (1, 2) and (), to 1e-8
    relative with its norm; the folded singlets and triplets to 1e-6 eV; the exact
    kernel 1e-7 Ha below min D and above it, to 1e-12; a flip
    parity refused; gw_modes_df's traced peak at most 6 (nm, nm) arrays;
    static_term_df in slabs of three modes equals its one-slab form."""
    from src.Base.utils.time_frequency import minimax_points_for_accuracy
    ok = True
    nv = len(eps) - no
    om = 0.3
    rng = np.random.default_rng(17)
    y1 = SB({s: rng.standard_normal((no, nv)) for s in ('aa', 'bb')})

    def rel(a, b):
        return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))

    for screening in ('tda', 'rpa'):
        P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening)
        PL = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening,
                                          laplace=True)
        be = P['be']
        dM = max(rel(PL['M'].get(k), P['M'].get(k)) for k in P['M'].blocks)
        ok &= check(PL['W'] is None and dM < 1e-12,
                    f'{screening}, laplace: W is None, M equals the W pieces\'',
                    f'{dM:.1e} relative')
        # static_term_df's slabs over the modes: three modes per slab, so 14 slabs
        # with a short last one, on one and on three workers, against one slab
        from src.SingleReference.ADC.eeADC import ee_spin_blocks as sb
        omega_m, rho = ee_gw_pieces.gw_modes_df(eps, B, no, screening)
        one = ee_gw_pieces.static_term_df(eps, B, rho, omega_m, no)
        for workers in (1, 3):
            keep_gb, keep_w = sb._STREAM_GB, sb._stream_workers
            sb._STREAM_GB = 8 * no * nv * 2 * 3 * workers * 1.0001 / 1e9
            sb._stream_workers = lambda w=workers: w
            try:
                plan = sb._slab_plan(omega_m.size, no * nv, 2)
                slabbed = ee_gw_pieces.static_term_df(eps, B, rho, omega_m, no)
            finally:
                sb._STREAM_GB, sb._stream_workers = keep_gb, keep_w
            d = rel(slabbed, one)
            ok &= check(plan == (3, workers) and d < 1e-12,
                        f'{screening}: static_term_df in slabs of 3 modes on {workers} '
                        'worker(s) equals one slab', f'plan {plan}, {d:.1e} relative')
        Yt = be.divide(P['V'](y1), P['D'] - om)
        ref = [P['Vt'](Yt), P['Vt'](be.divide(Yt, P['D'] - om))]
        n2_ref = P['dnorm2'](Yt)
        YL = be.divide(PL['V'](y1), PL['D'] - om)
        d = max(rel(PL['Vt'](YL).get(s), ref[0].get(s)) for s in ('aa', 'bb'))
        dn = abs(PL['dnorm2'](YL) - n2_ref) / n2_ref
        ok &= check(d < 1e-12 and dn < 1e-12,
                    f'{screening}, laplace: the DF V and Vt give the W pieces\' '
                    'Vt (D - w)^-1 V', f'{d:.1e}, norm {dn:.1e} relative')
        # PL's own min D: its modes come from another product of the same factors;
        # towards min D the grid grows (about 30 points 3e-3 Ha below it)
        d_min = float(np.min(PL['D']))
        for w_at in (om, -0.2, 0.05, d_min - 3e-3):
            Yt = be.divide(P['V'](y1), P['D'] - w_at)
            ref = [P['Vt'](Yt), P['Vt'](be.divide(Yt, P['D'] - w_at))]
            n2_ref = P['dnorm2'](Yt)
            ntau, err = minimax_points_for_accuracy(
                d_min - w_at, float(np.max(PL['D'])) - w_at,
                ee_gw_pieces._LAPLACE_TARGET, relative=True)
            ws, n2 = PL['K'](y1, w_at, (1, 2))
            ws0, n20 = PL['K'](y1, w_at, ())
            d = [max(rel(w.get(s), r.get(s)) for s in ('aa', 'bb'))
                 for w, r in zip(ws, ref)]
            dn = max(abs(n2 - n2_ref), abs(n20 - n2_ref)) / n2_ref
            ok &= check(len(ws) == 2 and ws0 == [] and max(d) < 1e-8 and dn < 1e-8,
                        f'{screening}: kernel_laplace equals V, divide, Vt at omega '
                        f'{w_at:.4f} Ha',
                        f'power 1 {d[0]:.1e}, power 2 {d[1]:.1e}, norm {dn:.1e} '
                        f'relative, ntau {ntau} (grid error {err:.1e})')
        for spin in ('singlet', 'triplet'):
            a = ee_fold.solve_folded(P, 3, spin=spin, dense_limit=0)
            b = ee_fold.solve_folded(PL, 3, spin=spin, dense_limit=0)
            dw = float(np.max(np.abs(a.omega - b.omega))) * HARTREE_TO_EV
            ok &= check(b.omega.size == 3 and dw < 1e-6,
                        f'{screening}: the folded {spin}s with and without laplace',
                        f'max |dw| {dw:.1e} eV')
        # the exact fallback: 1e-7 Ha below min D, where no tabulated grid reaches
        # the target, and above min D, mid-way across the widest gap of its 50
        # lowest D_K
        lowest = np.sort(PL['D'], axis=None)[:50]
        g = int(np.argmax(np.diff(lowest)))
        for label, w_at in (('1e-7 Ha below min D', d_min - 1e-7),
                            ('above min D', 0.5 * (lowest[g] + lowest[g + 1]))):
            # PL's own V, divide, Vt: a min D one ulp off would move 1/(D - w)
            Yh = be.divide(PL['V'](y1), PL['D'] - w_at)
            r1, n2h_ref = PL['Vt'](Yh), PL['dnorm2'](Yh)
            wsh, n2h = PL['K'](y1, w_at, (1,))
            d = max(rel(wsh[0].get(s), r1.get(s)) for s in ('aa', 'bb'))
            dn = abs(n2h - n2h_ref) / n2h_ref
            ok &= check(d < 1e-12 and dn < 1e-12,
                        f'{screening}: kernel_laplace {label} is the exact kernel',
                        f'{d:.1e}, norm {dn:.1e} relative, omega {w_at:.7f} Ha')
        try:
            PL['K'](SB(y1.blocks, parity=1.0), om, (1,))
            ok &= check(False, f'{screening}: kernel_laplace refuses a flip parity')
        except NotImplementedError as exc:
            ok &= check('parity' in str(exc),
                        f'{screening}: kernel_laplace refuses a flip parity',
                        str(exc)[:60])
    # the modes' memory in units of one (nm, nm) float64 array, water cc-pVTZ: the
    # RPA class's dense matrix functions held about 20 of them at once
    _, e3, B3, n3 = water('cc-pvtz')
    unit = 8 * (n3 * (len(e3) - n3)) ** 2
    for screening in ('tda', 'rpa'):
        units = traced_peak(lambda: ee_gw_pieces.gw_modes_df(e3, B3, n3,
                                                             screening)) / unit
        ok &= check(units <= 6, f'{screening}: gw_modes_df holds at most 6 (nm, nm) '
                    'arrays at once', f'{units:.1f}, water cc-pVTZ')
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
    all_ok &= check_streamed_kernel(eps, B, no)
    all_ok &= check_laplace_kernel(eps, B, no)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
