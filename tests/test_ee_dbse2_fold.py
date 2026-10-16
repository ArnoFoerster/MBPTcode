"""Eq 71 of Monino and Loos 2023 on the signed fold (dBSE2@GW doubles, level 'dbse2').

Checks, water (RHF, DF factors), basis per check. The GW energies are a deterministic
stand-in (gw_like: occupied +0.03 Ha, virtual -0.05 Ha on the HF energies), so a swapped
energy set is visible; the W lines are screened at the HF energies throughout.

  1. build_pieces_dbse2 (6-31G): M per channel equals the Casida A of
     LinearResponseSolver at the GW energies, lBSE=True with the bare metric for
     singles='eq62' and with the static W_aux for 'eq62+eq59', to 1e-10 Ha; D is
     e_c + e_d - e_k - e_l, shape (no, no, nv, nv); V(y) has the eight signed blocks
     '+aaaa' ... '-baba' of that shape; Vt is the eta-transpose of V (V dense by unit
     singles, Vt on random doubles, eta = +1 on the '+' blocks, -1 on the '-' blocks) to
     1e-12; dnorm2 is |Y+|^2 - |Y-|^2, positive on a '+'-only Y and negative on a
     '-'-only Y; the HF energies pass as eps_gw, a closed GW gap raises ValueError, a
     flip parity raises NotImplementedError; matvec and solve_folded leave u, D and B
     untouched.
  2. fold against unfolded: the dense signed supermatrix [[M, V+^T, -V-^T], [V+, D, 0],
     [V-, 0, D]] from the pieces by unit vectors, eig not eigh; STO-3G the whole matrix,
     6-31G per channel on the eigenspaces of the alpha<->beta flip; both kernels; every
     root of solver 'newton' and 'davidson' (singlet, triplet, and spin=None on STO-3G)
     is a real eigenvalue to 3e-10 Ha (1e-8 eV) and its T1 the eigenvector's
     |y|^2 / (|y|^2 + |Z+|^2 - |Z-|^2) to 1e-8; every real eigenvalue below min D is a
     root (completeness); the two solvers agree to 1e-8 Ha and 1e-8 on T1, their matvec
     counts printed side by side; the count of complex eigenvalues is printed.
  3. eq 74 as printed (kernel 'printed', I.J + K.L - I.K - K.I) and Yamada's kernel
     ('eq71', I.J + K.L - I.L - L.I) transcribed on spin orbitals at omega = 0.3 Ha
     against A_eff(omega) - M from the pieces, singlet and triplet, 6-31G, to 1e-11 Ha;
     eq 71's triplet is zero to 1e-12, the printed triplet carries the loop terms.
  4. the static limit of the pieces' terms (static_limit_pieces) against
     static_second_order_kernel_df(..., kernel=, channel=) at the GW energies, both
     kernels, both channels, 6-31G, to 1e-10 Ha.
  5. boundaries: 'dbse2' is a fold level; the dressed W lines reproduce
     sum_PQ B_P,pq W_PQ B_Q,rs to 1e-12; coupling_pairs('eq71', 'triplet') is empty and
     'printed' keeps its pair; an unknown kernel, channel, sets or singles raises
     ValueError; a B of the wrong shape raises ValueError; pieces at level 'dbse2'
     without dnorm2 raise ValueError in folded_operator.

Run: python tests/test_ee_dbse2_fold.py
"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.SingleReference.ADC.eeADC import ee_dbse2_pieces, ee_fold
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, static_screened_coulomb_aux, static_second_order_kernel_df)

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'
# the signed doubles blocks of one pair, the order the dense forms below stack them in
KEYS = ('+aaaa', '-aaaa', '+bbbb', '-bbbb', '+abab', '-abab', '+baba', '-baba')


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def water(basis='6-31g'):
    """(eps, B, no): RHF water with DF factors B[Q, p, q] in the MO basis."""
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    return eps, B, mol.nelectron // 2


def gw_like(eps, no):
    """A deterministic stand-in for the G0W0 energies: the HF gap closed by 0.08 Ha."""
    e = np.array(eps, float)
    e[:no] += 0.03
    e[no:] -= 0.05
    return e


def check_boundaries(eps, B, no):
    """The level name, the dressed W lines, the pair lists and the refusals."""
    ok = True
    ok &= check('dbse2' in ee_fold.FOLD_LEVELS, "'dbse2' is a fold level")
    L, Bt = ee_dbse2_pieces.dressed_factors(eps, B, no)
    W_aux = static_screened_coulomb_aux(eps, B, no)
    rng = np.random.default_rng(11)
    n = len(eps)
    d = 0.0
    for p, q, r, s in rng.integers(0, n, size=(4, 4)):
        # w(pq|rs) = sum_P Bt_P,pq Bt_P,rs against sum_PQ B_P,pq W_PQ B_Q,rs
        d = max(d, abs(float(Bt[:, p, q] @ Bt[:, r, s])
                       - float(B[:, p, q] @ W_aux @ B[:, r, s])))
    ok &= check(d < 1e-12 and np.allclose(L @ L.T, W_aux, atol=1e-12),
                'the dressed W lines reproduce the static W_aux', f'{d:.1e}')
    ok &= check(ee_dbse2_pieces.coupling_pairs('eq71', 'triplet') == [],
                "coupling_pairs('eq71', 'triplet') is empty")
    pr = ee_dbse2_pieces.coupling_pairs('printed', 'triplet')
    ok &= check(len(pr) == 1 and len(pr[0]) == 2 and pr[0][0].shape == (4,),
                "coupling_pairs('printed', 'triplet') keeps its pair")
    for kw, word in ((dict(kernel='eq74', channel='singlet'), 'kernel'),
                     (dict(kernel='eq71', channel='doublet'), 'channel')):
        try:
            ee_dbse2_pieces.coupling_pairs(**kw)
            ok &= check(False, f'an unknown {word} raises ValueError')
        except ValueError as exc:
            ok &= check(word in str(exc), f'an unknown {word} raises ValueError',
                        str(exc)[:60])
    try:
        ee_dbse2_pieces.dressed_factors(eps, B[:, :, :3], no)
        ok &= check(False, 'a B of the wrong shape raises ValueError')
    except ValueError as exc:
        ok &= check('shape' in str(exc), 'a B of the wrong shape raises ValueError',
                    str(exc)[:60])
    try:
        ee_fold.folded_operator({'level': 'dbse2'}, 0.3)
        ok &= check(False, "dbse2 pieces without dnorm2 raise ValueError")
    except ValueError as exc:
        ok &= check('dnorm2' in str(exc),
                    'dbse2 pieces without dnorm2 raise ValueError', str(exc)[:60])
    return ok


def flat_doubles(Z):
    """SB doubles on KEYS -> one vector, the blocks in KEYS order."""
    return np.concatenate([Z.get(k).ravel() for k in KEYS])


def dense_coupling(P):
    """(V, eta): V (8 n_d, 2 ns) the coupling on the full spin-orbital singles (aa then
    bb) by unit vectors, rows the doubles in KEYS order; eta (8 n_d,) the metric, +1 on
    the '+' blocks and -1 on the '-' blocks."""
    no, nv = P['no'], P['nv']
    ns = no * nv
    E = np.eye(2 * ns)
    cols = [flat_doubles(P['V'](ee_fold.singles_flat_to_sb(E[:, k], no, nv)))
            for k in range(2 * ns)]
    n_d = no * no * nv * nv
    eta = np.concatenate([np.full(n_d, 1.0 if k[0] == '+' else -1.0) for k in KEYS])
    return np.column_stack(cols), eta


def check_pieces(eps, B, no):
    """M per channel against the Casida A at the GW energies; D; the eight blocks; Vt
    the eta-transpose of V; dnorm2 signed; the refusals; inputs untouched."""
    ok = True
    nv = len(eps) - no
    ns = no * nv
    eps_gw = gw_like(eps, no)
    W_aux = static_screened_coulomb_aux(eps, B, no)
    lr = LinearResponseSolver(eps_gw, coeff_df=B, spin_mode='restricted')
    # the Casida A with the bare metric is eq 62 (lBSE=False drops the exchange: the
    # direct RPA block), with the static W_aux it is eq 62 + eq 59
    for singles, metric in (('eq62', np.eye(B.shape[0])), ('eq62+eq59', W_aux)):
        P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel='eq71',
                                               singles=singles)
        Ma, Mx = P['M'].get('aaaa'), P['M'].get('aabb')
        for spin, sgn, triplet in (('singlet', 1.0, False), ('triplet', -1.0, True)):
            A, _ = lr.build_casida_matrices(no, lBSE=True, W_aux=metric,
                                            triplet=triplet)
            Mch = (Ma + sgn * Mx).reshape(ns, ns)
            d = float(np.max(np.abs(Mch - np.asarray(A))))
            ok &= check(d < 1e-10, f'{singles} {spin}: M equals the Casida A at the GW '
                        'energies', f'max |d| {d:.1e} Ha')
    eo, ev = eps_gw[:no], eps_gw[no:]
    Dref = (ev[None, None, :, None] + ev[None, None, None, :]
            - eo[:, None, None, None] - eo[None, :, None, None])
    ok &= check(P['D'].shape == (no, no, nv, nv) and np.array_equal(P['D'], Dref)
                and P['level'] == 'dbse2',
                "D is e_c + e_d - e_k - e_l, shape (no, no, nv, nv); level 'dbse2'",
                str(P['D'].shape))
    Y = P['V'](ee_fold.singles_flat_to_sb(np.eye(2 * ns)[:, 0], no, nv))
    ok &= check(sorted(Y.keys()) == sorted(KEYS)
                and all(Y.get(k).shape == (no, no, nv, nv) for k in KEYS),
                'V(y) has the eight signed blocks of shape (no, no, nv, nv)',
                ' '.join(sorted(Y.keys())))
    V, eta = dense_coupling(P)
    rng = np.random.default_rng(7)
    Zr = SB({k: rng.standard_normal((no, no, nv, nv)) for k in KEYS})
    zf = flat_doubles(Zr)
    wt = ee_fold.singles_sb_to_flat(P['Vt'](Zr), no, nv)
    d = float(np.max(np.abs(wt - V.T @ (eta * zf))))
    ok &= check(d < 1e-12, 'Vt is the eta-transpose of V', f'{d:.1e}')
    dn, ref = float(P['dnorm2'](Zr)), float((eta * zf) @ zf)
    ok &= check(abs(dn - ref) < 1e-12 * abs(ref), 'dnorm2 is |Y+|^2 - |Y-|^2',
                f'{dn:.6f} vs {ref:.6f}')
    Zp = SB({k: Zr.get(k) * (1.0 if k[0] == '+' else 0.0) for k in KEYS})
    Zm = SB({k: Zr.get(k) * (1.0 if k[0] == '-' else 0.0) for k in KEYS})
    ok &= check(P['dnorm2'](Zp) > 0 and P['dnorm2'](Zm) < 0,
                "dnorm2 > 0 on a '+'-only Y and < 0 on a '-'-only Y")
    ee_dbse2_pieces.build_pieces_dbse2(eps, eps, B, no)
    ok &= check(True, 'the HF energies pass as eps_gw')
    bad = eps_gw.copy()
    bad[no - 1], bad[no] = bad[no], bad[no - 1]
    try:
        ee_dbse2_pieces.build_pieces_dbse2(bad, eps, B, no)
        ok &= check(False, 'a closed GW gap raises ValueError')
    except ValueError as exc:
        ok &= check('gap' in str(exc), 'a closed GW gap raises ValueError',
                    str(exc)[:60])
    y1 = ee_fold.singles_flat_to_sb(np.eye(2 * ns)[:, 0], no, nv)
    try:
        P['V'](SB({'aa': y1.get('aa')}, parity=1.0))
        ok &= check(False, 'a flip parity raises NotImplementedError')
    except NotImplementedError:
        ok &= check(True, 'a flip parity raises NotImplementedError')
    for kw, word in ((dict(sets=('eq61',)), 'sets'), (dict(singles='eq59'), 'singles')):
        try:
            ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, **kw)
            ok &= check(False, f'an unknown {word} raises ValueError')
        except ValueError as exc:
            ok &= check(word in str(exc), f'an unknown {word} raises ValueError',
                        str(exc)[:60])
    D0, B0 = P['D'].copy(), B.copy()
    matvec, _, diag_s, _, _ = ee_fold.folded_operator(P, 0.3, spin='singlet')
    u = np.linspace(-1.0, 1.0, diag_s.size)
    u0 = u.copy()
    matvec(u)
    ee_fold.solve_folded(P, 1, spin='singlet')
    ok &= check(np.array_equal(u, u0) and np.array_equal(P['D'], D0)
                and np.array_equal(B, B0),
                'matvec and solve_folded leave u, D and B untouched')
    return ok


def transcription(Bt, eps_gw, no, omega, kernel):
    """Theta(omega) of ee_dbse2_pieces' table on spin orbitals (interleaved spins, the
    W lines zero across spin in a pair), as the (ns, ns) singlet and triplet blocks in
    (ia, jb): the vertices and products of eq74_fold_identity.py with w from the dressed
    factors and the GW energies in the doubles' denominators."""
    n_sp = eps_gw.size
    # w(pq|rs) = sum_P Bt_P,pq Bt_P,rs on spatial orbitals, then on spin orbitals
    w_sp = np.einsum('Ppq,Prs->pqrs', Bt, Bt, optimize=True)
    n = 2 * n_sp
    idx = np.arange(n)
    sp, sg = idx // 2, idx % 2
    same = sg[:, None] == sg[None, :]
    w = w_sp[np.ix_(sp, sp, sp, sp)] * same[:, :, None, None] * same[None, None, :, :]
    eps = eps_gw[sp]
    no2, nv2 = 2 * no, 2 * (n_sp - no)
    o, v = slice(0, no2), slice(no2, n)
    eo, ev = eps[o], eps[v]
    D = (ev[None, None, :, None] + ev[None, None, None, :]
         - eo[:, None, None, None] - eo[None, :, None, None])
    R = 1.0 / (omega - D)

    def vertex(kind, y):
        # t_kc,ld = sum_i w(ik|ld) y_ic ; u_kc,ld = sum_a w(ac|ld) y_ka
        t = np.einsum('ikld,ic->kcld', w[o, o, o, v], y)
        u = np.einsum('acld,ka->kcld', w[v, v, o, v], y)
        if kind == 'I':
            return t.transpose(0, 2, 1, 3)              # Y_klcd = t_kc,ld
        if kind == 'J':
            return t.transpose(2, 0, 3, 1)              # Y_klcd = t_ld,kc
        if kind == 'K':
            return u.transpose(0, 2, 1, 3)
        return u.transpose(2, 0, 3, 1)

    ns2 = no2 * nv2
    E = np.eye(ns2).reshape(ns2, no2, nv2)
    VX = {X: np.stack([vertex(X, E[k]) for k in range(ns2)]).reshape(ns2, -1)
          for X in 'IJKL'}

    def product(X, Y):
        # sum_K (X e_ia)_K (Y e_jb)_K / (omega - D_K)
        T = np.einsum('xK,K,yK->xy', VX[X], R.ravel(), VX[Y])
        return T.reshape(no2, nv2, no2, nv2)

    if kernel == 'printed':
        T = (product('I', 'J') + product('K', 'L') - product('I', 'K')
             - product('K', 'I'))
    else:
        T = (product('I', 'J') + product('K', 'L') - product('I', 'L')
             - product('L', 'I'))
    S = T[::2, ::2, ::2, ::2] + T[::2, ::2, 1::2, 1::2]
    Tt = T[::2, ::2, ::2, ::2] - T[::2, ::2, 1::2, 1::2]
    ns = S.shape[0] * S.shape[1]
    return {'singlet': S.reshape(ns, ns), 'triplet': Tt.reshape(ns, ns)}


def check_transcription(eps, B, no):
    """Theta(omega) from the pieces, A_eff(omega) - M per channel, against the
    spin-orbital transcription of each kernel; eq 71's triplet zero, the printed one's
    carrying the loop terms; the other kernel's transcription is the negative
    control."""
    ok = True
    nv = len(eps) - no
    ns = no * nv
    eps_gw = gw_like(eps, no)
    omega = 0.3
    got, ref = {}, {}
    for kernel in ee_dbse2_pieces.KERNELS:
        P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel=kernel)
        ref[kernel] = transcription(P['Bt'], eps_gw, no, omega, kernel)
        Ma, Mx = P['M'].get('aaaa'), P['M'].get('aabb')
        for spin, sgn in (('singlet', 1.0), ('triplet', -1.0)):
            mv_w = ee_fold.folded_operator(P, omega, spin)[0]
            mv_0 = ee_fold.folded_operator(P, None, spin)[0]
            A_w = ee_fold.dense_effective(mv_w, ns)
            A_0 = ee_fold.dense_effective(mv_0, ns)
            d0 = float(np.max(np.abs(A_0 - (Ma + sgn * Mx).reshape(ns, ns))))
            got[kernel, spin] = A_w - A_0
            d = float(np.max(np.abs(got[kernel, spin] - ref[kernel][spin])))
            ok &= check(d < 1e-11 and d0 < 1e-12,
                        f'{kernel} {spin}: Theta(omega) from the pieces equals the '
                        'transcription', f'max |d| {d:.1e} Ha, M {d0:.1e}')
    d1 = float(np.max(np.abs(ref['eq71']['triplet'])))
    d2 = float(np.max(np.abs(got['eq71', 'triplet'])))
    ok &= check(d1 < 1e-12 and d2 < 1e-12, "eq71: Theta has no triplet part",
                f'transcription {d1:.1e}, pieces {d2:.1e} Ha')
    d3 = float(np.max(np.abs(ref['printed']['triplet'])))
    ok &= check(d3 > 1e-3, 'printed: the triplet carries the loop terms',
                f'max {d3:.1e} Ha')
    d4 = float(np.max(np.abs(got['eq71', 'singlet'] - ref['printed']['singlet'])))
    ok &= check(d4 > 1e-3,
                "control: eq71's pieces differ from the printed transcription",
                f'max |d| {d4:.1e} Ha')
    return ok


def check_static_limit(eps, B, no):
    """The static limit of the pieces' terms against the tree's static kernel at the GW
    energies, both kernels, both channels."""
    ok = True
    eps_gw = gw_like(eps, no)
    W_aux = static_screened_coulomb_aux(eps, B, no)
    for kernel in ee_dbse2_pieces.KERNELS:
        for channel in ('singlet', 'triplet'):
            got = ee_dbse2_pieces.static_limit_pieces(eps_gw, eps, B, no, kernel=kernel,
                                                      channel=channel)
            ref, _ = static_second_order_kernel_df(eps_gw, B, W_aux, no, eta=0.0,
                                                   channel=channel, kernel=kernel)
            d = float(np.max(np.abs(got - ref)))
            ok &= check(d < 1e-10, f'{kernel} {channel}: the static limit equals the '
                        'static kernel',
                        f'max |d| {d:.1e} Ha, |Theta| {np.abs(ref).max():.1e}')
    return ok


def dense_signed_supermatrix(P):
    """H = [[M, (eta V)^T], [V, D]] over the full spin-orbital singles (aa then bb) and
    the doubles in KEYS order, from the pieces by unit vectors; eta is +1 on the '+'
    blocks and -1 on the '-' blocks, so Z = (omega - D)^-1 V y gives the module's
    A_eff. Returns (H, ns, eta_full) with eta_full +1 on the singles."""
    no, nv = P['no'], P['nv']
    m0, _, diag_s, _, _ = ee_fold.folded_operator(P, None, spin=None)
    ns = diag_s.size
    M = ee_fold.dense_effective(m0, ns)
    V, eta = dense_coupling(P)
    D = np.tile(P['D'].ravel(), len(KEYS))
    H = np.block([[M, (eta[:, None] * V).T], [V, np.diag(D)]])
    return H, ns, np.concatenate([np.ones(ns), eta])


def flip_partner(no, nv):
    """The alpha<->beta flip on the supermatrix's basis as an involution perm (n,):
    singles aa <-> bb, doubles '+aaaa' <-> '+bbbb', '+abab' <-> '+baba', the '-' ones
    alike, the (k, l, c, d) order untouched."""
    ns, n_d = no * nv, no * no * nv * nv
    perm = np.empty(2 * ns + len(KEYS) * n_d, int)
    perm[:ns] = np.arange(ns, 2 * ns)
    perm[ns:2 * ns] = np.arange(ns)
    tr = str.maketrans('ab', 'ba')
    off = {k: 2 * ns + i * n_d for i, k in enumerate(KEYS)}
    for k in KEYS:
        o = off[k.translate(tr)]
        perm[off[k]:off[k] + n_d] = np.arange(o, o + n_d)
    return perm


def channel_block(H, perm, sign):
    """E^T H E on the sign eigenspace of the flip, E the orthonormal pairs
    (e_i + sign e_perm(i))/sqrt2 over the representatives i < perm(i); (H_s, reps)."""
    reps = np.flatnonzero(np.arange(perm.size) < perm)
    part = perm[reps]
    Hr = H[reps] + sign * H[part]
    return 0.5 * (Hr[:, reps] + sign * Hr[:, part]), reps


def solve_counted(P, nroots, **kw):
    """solve_folded with its trace captured; (result, matvec count)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = ee_fold.solve_folded(P, nroots, verbose=1, **kw)
    mv = re.findall(r'mv = (\d+)', buf.getvalue())
    return res, int(mv[-1]) if mv else -1


def check_fold_vs_unfolded(basis):
    """Every folded root is a real eigenvalue of the dense signed supermatrix of its
    pieces and its T1 the eigenvector's |y|^2 / (|y|^2 + |Z+|^2 - |Z-|^2); the lowest
    roots complete; Newton against the Davidson; the Hermitian misfold as control."""
    ok = True
    eps, B, no = water(basis)
    nv = len(eps) - no
    ns = no * nv
    eps_gw = gw_like(eps, no)
    perm = flip_partner(no, nv)
    for kernel in ee_dbse2_pieces.KERNELS:
        P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel=kernel,
                                               singles='eq62+eq59')
        H, ns_full, eta_full = dense_signed_supermatrix(P)
        d_min = float(P['D'].min())
        blocks = {}
        for spin, sign in (('singlet', 1.0), ('triplet', -1.0)):
            Hs, reps = channel_block(H, perm, sign)
            blocks[spin] = (Hs, eta_full[reps], ns)
        if basis == 'sto-3g':
            blocks[None] = (H, eta_full, ns_full)
        for spin, (Hs, wts, nsing) in blocks.items():
            w, v = scipy.linalg.eig(Hs)
            real = np.abs(w.imag) < 1e-9
            wr, vr = w[real].real, v[:, real].real
            below = np.sort(wr[wr < d_min])
            print(f'    {kernel} {basis} {spin}: {int((~real).sum())} complex '
                  f'eigenvalues of {w.size}, {below.size} real below min D = '
                  f'{d_min:.3f} Ha')
            nroots = int(min(below.size, 6))
            results = {}
            for label, solver, dl in (('newton', 'newton', 0),
                                      ('davidson', 'davidson', 0),
                                      ('default', 'davidson', None)):
                kw = dict(spin=spin, solver=solver, tol_omega=1e-12, tol_residual=1e-10)
                if dl is not None:
                    kw['dense_limit'] = dl
                res, mv = solve_counted(P, nroots, **kw)
                results[label] = res
                dw, dt = [], []
                for r in range(res.omega.size):
                    k = int(np.argmin(np.abs(wr - res.omega[r])))
                    dw.append(abs(wr[k] - res.omega[r]))
                    u = vr[:, k]
                    t1 = float(u[:nsing] @ u[:nsing]) / float((wts * u) @ u)
                    dt.append(abs(t1 - res.t1[r]))
                n = min(res.omega.size, nroots)
                dc = float(np.max(np.abs(np.sort(res.omega)[:n] - below[:n])))
                ok &= check(max(dw) < 3e-10 and max(dt) < 1e-8 and dc < 3e-10
                            and n == nroots and bool(res.converged.all()),
                            f'{kernel} {basis} {spin} {label}: roots and T1 are '
                            f'eigenpairs of the signed supermatrix, lowest {nroots} '
                            'complete', f'max |dw| {max(dw):.1e} Ha, max |dT1| '
                            f'{max(dt):.1e}, {mv} matvecs')
            a, b = results['newton'], results['davidson']
            n = min(a.omega.size, b.omega.size, nroots)
            dw = float(np.max(np.abs(a.omega[:n] - b.omega[:n])))
            dt = float(np.max(np.abs(a.t1[:n] - b.t1[:n])))
            ok &= check(dw < 1e-8 and dt < 1e-8,
                        f'{kernel} {basis} {spin}: newton and davidson agree',
                        f'max |dw| {dw:.1e} Ha, max |dT1| {dt:.1e}')
        if basis == 'sto-3g':
            # the control: the Hermitian misfold [[M, V^T], [V, D]] has other roots
            Hm = H.copy()
            Hm[:ns_full, ns_full:] *= eta_full[ns_full:][None, :]
            wm = np.sort(np.linalg.eigvalsh(Hm))
            res = ee_fold.solve_folded(P, 1, spin='singlet', tol_omega=1e-12)
            d = float(np.min(np.abs(wm - res.omega[0])))
            ok &= check(d > 1e-6, f'{kernel}: control, the Hermitian misfold differs',
                        f'nearest {d:.1e} Ha')
    return ok


def main():
    """Run every check; print ALL PASSED or FAILURES DETECTED; exit 0 or 1."""
    all_ok = True
    eps, B, no = water('6-31g')
    all_ok &= check_boundaries(eps, B, no)
    all_ok &= check_pieces(eps, B, no)
    all_ok &= check_transcription(eps, B, no)
    all_ok &= check_static_limit(eps, B, no)
    all_ok &= check_fold_vs_unfolded('sto-3g')
    all_ok &= check_fold_vs_unfolded('6-31g')
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
