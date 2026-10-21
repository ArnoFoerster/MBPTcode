"""Eqs 61 and 71 of Monino and Loos 2023 on the signed fold (dBSE2@GW doubles, level
'dbse2'), the two sets of doubles selected by `sets`.

Checks, water (RHF, DF factors), basis per check. The GW energies are a deterministic
stand-in (gw_like: occupied +0.03 Ha, virtual -0.05 Ha on the HF energies), so a swapped
energy set is visible; the W lines and the modes are screened at the HF energies
throughout. The doubles of every set lie on the flat per-spin layout: keys 'a' and 'b'
by the spin of the (k, c) pair, pieces['segments'][s] the (name, sign, slice) list of
key s in the order eq61+, eq61- (no nv nm each), eq71+ssss, eq71-ssss, eq71+ss'ss',
eq71-ss'ss' (no^2 nv^2 each), one flat D shared by both keys.

  1. build_pieces_dbse2 (6-31G): M per channel equals the Casida A of
     LinearResponseSolver at the GW energies, lBSE=True with the bare metric for
     singles='eq62' and with the static W_aux for 'eq62+eq59', to 1e-10 Ha. On every
     set of SETS: V(y) has the keys 'a' and 'b' of the length the segments say; D has
     that length, D[slice] = (Omega_m + e_c - e_k).ravel() on the eq 61 segments and
     (e_c + e_d - e_k - e_l).ravel() on the eq 71 ones; Vt is the eta-transpose of V (V
     dense by unit singles, Vt on random doubles, eta the segments' signs) to 1e-12;
     dnorm2 is sum_segments sign |Z[slice]|^2, positive on a '+'-only Z and negative on
     a '-'-only Z; the HF energies pass as eps_gw, a closed GW gap raises ValueError, a
     flip parity raises NotImplementedError; matvec and solve_folded leave u, D, M and
     B untouched; pieces['modes'] is (omega, rho) of gw_modes_df on a mode set, None
     otherwise, and a `modes` argument is taken as given.
  2. fold against unfolded: the dense signed supermatrix [[M, (eta V)^T], [V, D]] from
     the pieces by unit vectors, eig not eigh; STO-3G the whole matrix, 6-31G per
     channel on the eigenspaces of the alpha<->beta flip; both kernels on STO-3G and
     the eq 71 kernel on 6-31G, every set of SETS;
     every root of solver 'newton' and 'davidson' (singlet, triplet, and spin=None
     on STO-3G), and of 'block' on the mode sets, is a real eigenvalue to 3e-10 Ha
     (1e-8 eV) and its T1 the eigenvector's |y|^2 / (|y|^2 + |Z+|^2 - |Z-|^2) to 1e-8;
     at least six real eigenvalues lie below min D and the lowest six are the roots
     (completeness to the sixth); newton and davidson agree to 1e-8 Ha and 1e-8 on T1,
     their matvec counts and solver loops per root printed side by side (the 6-31G
     Davidson leg of both kernels on sets=('eq71',) keeps its model, asserted: the
     printed kernel's roots have T1 > 1, and the model's bracket follows the
     rising branch); the count of complex eigenvalues and the lowest
     complex real part against min D are printed. Regression guard: on sets=('eq71',)
     the two lowest newton roots per channel equal M0_ROOTS, recorded from this
     code, to 1e-8 Ha.
  3. eq 74 as printed (kernel 'printed', I.J + K.L - I.K - K.I) and Yamada's kernel
     ('eq71', I.J + K.L - I.L - L.I) transcribed on spin orbitals at omega = 0.3 Ha
     against A_eff(omega) - M from the pieces, singlet and triplet, 6-31G, to 1e-11 Ha;
     eq 71's triplet is zero to 1e-12, the printed triplet carries the loop terms. Eq
     61's signed cross J.K + K.J, transcribed from gw_modes' dense W with
     J_ia,kcm = -delta_ac W^m_ik, K_ia,kcm = +delta_ik W^m_ac and D_kcm = Omega_m +
     e_c - e_k at the GW energies, against sets=('eq61',) to 1e-11 Ha in both
     channels, the two channels' Theta^61 equal to 1e-12 (no exchange in eq 61's
     couplings); the two-set Theta is the sum of the two sets' to 1e-12. The builder's
     stored dressed factors equal dressed_factors' (check 5 pins those to W_aux). The
     product table is the spec's, re-derived by transcription_printed_eq74, eq 74's
     four printed terms as einsums, to 1e-11.
  4. the static limit of the pieces' terms (static_limit_pieces) against
     static_second_order_kernel_df(..., kernel=, channel=) at the GW energies, both
     kernels, both channels, 6-31G, to 1e-10 Ha; eq 71's triplet leg compares zero
     with zero.
  5. boundaries: 'dbse2' is a fold level; the dressed W lines reproduce
     sum_PQ B_P,pq W_PQ B_Q,rs to 1e-12; coupling_pairs('eq71', 'triplet') is empty and
     'printed' keeps its pair; an unknown kernel, channel or singles raises
     ValueError; sets outside SETS, ('eq66',) among them, raise ValueError naming
     `sets`; modes whose rho has naux + 1 rows raise naming `naux`; screening='x'
     raises naming `screening`; a B of the wrong shape raises ValueError; eps_gw or
     eps_hf in eV (a HOMO below -2), mixed units (a GW gap outside 0.2 to 5 times the
     HF gap), eps_gw of another length than B and nocc = norb raise ValueError naming
     the input; pieces at level 'dbse2' without dnorm2 raise ValueError in
     folded_operator.
  6. fold_step, H2O/cc-pVDZ at QuAcK's G0W0 energies, printed kernel: static, step,
     correction (eV) and zeta equal QuAcK's TDA dBSE2 singlet and (QuAcK's triplet
     carries the singlet's Theta) the 2026-10-18 triplet record to 1e-5; control: eq
     71's step is not QuAcK's; identities in static and |x|; stationary at the root;
     on hand-built signed pieces with 1 + dmatvec(x) < 0 one RuntimeWarning, no raise.

Run: python tests/test_ee_dbse2_fold.py
"""
import collections
import contextlib
import io
import math
import os
import re
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
from pyscf import gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.pyscf.pyscf_interface import DFIntegrals, get_orbital_energies
from src.SingleReference.ADC.eeADC import ee_dbse2_pieces, ee_fold, ee_gw_pieces
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, static_screened_coulomb_aux, static_second_order_kernel_df)

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'
# the two lowest roots (Ha) of check 2's pieces on sets=('eq71',), singles='eq62+eq59',
# gw_like_uniform's energies, solver newton at tol_omega 1e-12
M0_ROOTS = {
    ('sto-3g', 'eq71', 'singlet'): (0.442968065932, 0.527190875894),
    ('sto-3g', 'eq71', 'triplet'): (0.370338072713, 0.445326163321),
    ('sto-3g', 'printed', 'singlet'): (0.459080179849, 0.548624376509),
    ('sto-3g', 'printed', 'triplet'): (0.389912129426, 0.459391713355),
    ('6-31g', 'eq71', 'singlet'): (0.301434169674, 0.380377092199),
    ('6-31g', 'eq71', 'triplet'): (0.272469376582, 0.334739864551),
    ('6-31g', 'printed', 'singlet'): (0.313226018723, 0.392848106734),
    ('6-31g', 'printed', 'triplet'): (0.286815570557, 0.347614467992),
}
# QuAcK mol/H2O.xyz, Angstrom; QuAcK's G0W0@RHF e_QP, eV, all 24 orbitals, full_bse.log
# of 2026-09-25_quack-xcheck-h2o (the constant of test_bse2_static_kernel.py)
QUACK_H2O = 'O 0.0000 0.0000 0.0000; H 0.7571 0.0000 0.5861; H -0.7571 0.0000 0.5861'
QUACK_QP_EV = np.array([
    -547.096903359, -33.376695376, -18.558315280, -14.436803319, -12.158825764,
    4.708293937, 6.656989854, 20.360279319, 21.783506682, 30.406097434,
    31.317121452, 32.403102704, 38.226762239, 39.069729540, 44.458840280,
    49.498548274, 50.876928601, 65.959485567, 67.738625861, 88.744413957,
    90.503059309, 95.157864951, 103.217832913, 112.841177294])
# QuAcK's TDA dBSE2 on those energies (tda T, tda_w F, bse2 T, dbse T, dynamical TDA),
# singlet row 1 of 'First-order dynamical correction' in
# 2026-10-17_quack-xcheck-h2o-bse2dyn/tda_fullw_bse2_dyn.log: static, dynamic,
# correction (eV), renormalisation factor
QUACK_STEP_S1 = (8.759333, 8.616588, -0.142745, 1.012870)
# the triplet with its own kernel, the inline formula's record of 2026-10-18
# (2026-10-18_dbse2dyn-vs-quack-h2o-ccpvdz-eq61eq71-printed/result.json; QuAcK's triplet
# carries the singlet's Theta, the recorded convention mismatch): the same four
RECORD_STEP_T1 = (8.131480, 7.927939, -0.203541, 1.015818)
# the mode sets: the ones that take the 'block' solver in check 2 and have no M0 row
MODE_SETS = (('eq61',), ('eq61', 'eq71'))


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail, flush=True)
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
    """A deterministic stand-in for the G0W0 energies: the HF gap closed by 0.08 Ha,
    with an orbital-dependent part, so that a route confusing per-orbital GW energies
    with a shifted HF set cannot pass."""
    e = np.array(eps, float)
    e[:no] += 0.03 + 0.002 * np.arange(no)
    e[no:] -= 0.05 - 0.001 * np.arange(e.size - no)
    return e


def gw_like_uniform(eps, no):
    """The uniform shift M0_ROOTS was recorded with (the HF gap closed by 0.08 Ha, no
    orbital-dependent part): the regression guard keeps its input."""
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
    ev = 27.211386
    for label, word, args in (
            ('eps_gw in eV', 'eV', (eps * ev, eps, B, no)),
            ('eps_hf in eV', 'eV', (eps, eps * ev, B, no)),
            ('both energies in eV', 'eV', (eps * ev, eps * ev, B, no)),
            ('eps_gw at a tenth of eps_hf', 'gap', (eps * 0.1, eps, B, no)),
            ('eps_gw of another length than B', 'shape', (eps[:-1], eps, B, no)),
            ('nocc = norb', 'nocc', (eps, eps, B, len(eps)))):
        try:
            ee_dbse2_pieces.build_pieces_dbse2(*args)
            ok &= check(False, f'{label} raises ValueError')
        except ValueError as exc:
            ok &= check(word in str(exc), f'{label} raises ValueError', str(exc)[:60])
    try:
        ee_fold.folded_operator({'level': 'dbse2'}, 0.3)
        ok &= check(False, "dbse2 pieces without dnorm2 raise ValueError")
    except ValueError as exc:
        ok &= check('dnorm2' in str(exc),
                    'dbse2 pieces without dnorm2 raise ValueError', str(exc)[:60])
    eps_gw = gw_like(eps, no)
    om, rho = ee_gw_pieces.gw_modes_df(eps, B, no, 'rpa')
    bad = np.vstack([rho, rho[:1]])
    for label, word, kw in (
            ("sets=('eq66',)", 'sets', dict(sets=('eq66',))),
            ("sets=('eq61', 'eq71', 'x')", 'sets', dict(sets=('eq61', 'eq71', 'x'))),
            ("sets=('eq61', 'eq61')", 'sets', dict(sets=('eq61', 'eq61'))),
            ('modes with naux + 1 rows', 'naux', dict(sets=('eq61',), modes=(om, bad))),
            ('modes with omega of another length', 'modes',
             dict(sets=('eq61',), modes=(om[:-1], rho))),
            ("screening='x'", 'screening', dict(sets=('eq61',), screening='x'))):
        try:
            ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, **kw)
            ok &= check(False, f'{label} raises ValueError')
        except ValueError as exc:
            ok &= check(word in str(exc), f'{label} raises ValueError naming {word}',
                        str(exc)[:70])
    return ok


def flat_doubles(Z):
    """SB doubles on the flat layout -> one vector, key 'a' then key 'b'."""
    return np.concatenate([Z.get('a'), Z.get('b')])


def eta_of(P):
    """The metric (2 n_flat,) of the flat layout: the segments' signs, key 'a' then
    'b'."""
    n_flat = P['D'].size
    eta = np.empty(2 * n_flat)
    for i, s in enumerate('ab'):
        for _, sign, sl in P['segments'][s]:
            eta[i * n_flat + sl.start:i * n_flat + sl.stop] = sign
    return eta


def dense_coupling(P):
    """(V, eta): V (2 n_flat, 2 ns) the coupling on the full spin-orbital singles (aa
    then bb) by unit vectors, rows the flat doubles of key 'a' then key 'b'; eta
    (2 n_flat,) the segments' signs."""
    no, nv = P['no'], P['nv']
    ns = no * nv
    E = np.eye(2 * ns)
    cols = [flat_doubles(P['V'](ee_fold.singles_flat_to_sb(E[:, k], no, nv)))
            for k in range(2 * ns)]
    return np.column_stack(cols), eta_of(P)


def segment_lengths(no, nv, nm, sets):
    """The (name, length) list of one key's segments in the layout's order, key 'a'."""
    out = []
    if 'eq61' in sets:
        out += [('eq61+', no * nv * nm), ('eq61-', no * nv * nm)]
    if 'eq71' in sets:
        n71 = no * no * nv * nv
        out += [('eq71+aaaa', n71), ('eq71-aaaa', n71), ('eq71+abab', n71),
                ('eq71-abab', n71)]
    return out


def check_layout(P, eps_gw, no, nv, sets, om):
    """One set's flat layout: the keys, the segments, D per segment, Vt the
    eta-transpose of V, dnorm2 signed."""
    ok = True
    ns = no * nv
    tag = '+'.join(sets)
    nm = om.size
    want = segment_lengths(no, nv, nm, sets)
    n_flat = sum(n for _, n in want)
    segs = P['segments']
    got = [(name, sl.stop - sl.start) for name, _, sl in segs['a']]
    ends = [sl.stop for _, _, sl in segs['a']]
    ok &= check(got == want and ends == list(np.cumsum([n for _, n in want]))
                and [n for n, _, _ in segs['b']]
                == [n.translate(str.maketrans('ab', 'ba')) for n, _, _ in segs['a']]
                and [(s, sl) for _, s, sl in segs['b']]
                == [(s, sl) for _, s, sl in segs['a']],
                f"{tag}: the segments of key 'a' in the layout's order, key 'b' the "
                'spin-flipped names on the same slices', ' '.join(n for n, _ in got))
    eo, ev = eps_gw[:no], eps_gw[no:]
    D71 = (ev[None, None, :, None] + ev[None, None, None, :]
           - eo[:, None, None, None] - eo[None, :, None, None])
    C = om[None, None, :] + ev[None, :, None] - eo[:, None, None]
    d = 0.0
    for name, _, sl in segs['a']:
        ref = C if name.startswith('eq61') else D71
        d = max(d, float(np.max(np.abs(P['D'][sl] - ref.ravel()))))
    ok &= check(P['D'].shape == (n_flat,) and d == 0.0 and P['level'] == 'dbse2',
                f'{tag}: D is flat of length {n_flat}, Omega_m + e_c - e_k on the '
                'eq 61 segments and e_c + e_d - e_k - e_l on the eq 71 ones',
                f'max |d| {d}')
    Y = P['V'](ee_fold.singles_flat_to_sb(np.eye(2 * ns)[:, 0], no, nv))
    ok &= check(sorted(Y.keys()) == ['a', 'b']
                and all(Y.get(k).shape == (n_flat,) for k in 'ab'),
                f"{tag}: V(y) has the keys 'a' and 'b' of length {n_flat}",
                ' '.join(f'{k} {Y.get(k).shape}' for k in sorted(Y.keys())))
    V, eta = dense_coupling(P)
    rng = np.random.default_rng(7)
    Zr = SB({k: rng.standard_normal(n_flat) for k in 'ab'})
    zf = flat_doubles(Zr)
    wt = ee_fold.singles_sb_to_flat(P['Vt'](Zr), no, nv)
    d = float(np.max(np.abs(wt - V.T @ (eta * zf))))
    ok &= check(d < 1e-12, f'{tag}: Vt is the eta-transpose of V', f'{d:.1e}')
    dn, ref = float(P['dnorm2'](Zr)), float((eta * zf) @ zf)
    ok &= check(abs(dn - ref) < 1e-12 * abs(ref),
                f'{tag}: dnorm2 is sum_segments sign |Z[slice]|^2',
                f'{dn:.6f} vs {ref:.6f}')
    # the mutant: the unsigned sum |Z|^2 is not dnorm2 wherever a '-' segment holds
    # a nonzero Z, as every segment of Zr does
    unsigned = float(zf @ zf)
    ok &= check(abs(dn - unsigned) > 1e-6 * unsigned,
                f'{tag}: control, the unsigned |Z|^2 is not dnorm2',
                f'{dn:.6f} vs {unsigned:.6f}')
    plus = SB({k: Zr.get(k) * (eta[:n_flat] > 0) for k in 'ab'})
    minus = SB({k: Zr.get(k) * (eta[:n_flat] < 0) for k in 'ab'})
    ok &= check(P['dnorm2'](plus) > 0 and P['dnorm2'](minus) < 0,
                f"{tag}: dnorm2 > 0 on a '+'-only Z and < 0 on a '-'-only Z")
    return ok


def check_pieces(eps, B, no):
    """M per channel against the Casida A at the GW energies; the flat layout of every
    set (check_layout); the modes; the refusals; inputs untouched."""
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
            # the control: the other metric (W_aux for eq 62, the identity for eq 62
            # plus eq 59) builds a different A
            other = W_aux if singles == 'eq62' else np.eye(B.shape[0])
            Ao, _ = lr.build_casida_matrices(no, lBSE=True, W_aux=other,
                                             triplet=triplet)
            do = float(np.max(np.abs(Mch - np.asarray(Ao))))
            ok &= check(do > 1e-4, f'{singles} {spin}: control, M differs from the '
                        'Casida A on the other metric', f'max |d| {do:.1e} Ha')
    om, rho = ee_gw_pieces.gw_modes_df(eps, B, no, 'rpa')
    for sets in ee_dbse2_pieces.SETS:
        P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, sets=sets)
        ok &= check_layout(P, eps_gw, no, nv, sets, om)
        if 'eq61' in sets:
            dm = max(float(np.max(np.abs(P['modes'][0] - om))),
                     float(np.max(np.abs(np.abs(P['modes'][1]) - np.abs(rho)))))
            ok &= check(dm < 1e-10, f"{'+'.join(sets)}: pieces['modes'] is "
                        "gw_modes_df's (omega, rho) at the HF energies, RPA screening",
                        f'{dm:.1e}')
            Pg = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, sets=sets,
                                                    modes=(om, rho))
            ok &= check(Pg['modes'][1] is rho and np.array_equal(Pg['D'], P['D']),
                        f"{'+'.join(sets)}: a modes argument is taken as given")
        else:
            ok &= check(P['modes'] is None,
                        f"{'+'.join(sets)}: pieces['modes'] is None")
    P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel='eq71',
                                           singles='eq62+eq59')
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
    for kw, word in ((dict(sets=('eq72',)), 'sets'), (dict(singles='eq59'), 'singles')):
        try:
            ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, **kw)
            ok &= check(False, f'an unknown {word} raises ValueError')
        except ValueError as exc:
            ok &= check(word in str(exc), f'an unknown {word} raises ValueError',
                        str(exc)[:60])
    D0, B0 = P['D'].copy(), B.copy()
    M0 = {k: v.copy() for k, v in P['M'].items()}
    matvec, _, diag_s, _, _ = ee_fold.folded_operator(P, 0.3, spin='singlet')
    u = np.linspace(-1.0, 1.0, diag_s.size)
    u0 = u.copy()
    matvec(u)
    ee_fold.solve_folded(P, 1, spin='singlet')
    ok &= check(np.array_equal(u, u0) and np.array_equal(P['D'], D0)
                and np.array_equal(B, B0)
                and all(np.array_equal(P['M'].get(k), v) for k, v in M0.items()),
                'matvec and solve_folded leave u, D, M and B untouched')
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


def transcription_printed_eq74(Bt, eps_gw, no, omega):
    """Theta(omega) of the printed kernel straight from eq 74 as printed, W_pq,rs =
    w(pr|qs), the four terms of eq74_fold_identity.py's eq74() as einsums on spin
    orbitals (the same spin-orbital w as transcription), as the singlet and triplet
    blocks in (ia, jb): the vertex table's independent check."""
    n_sp = eps_gw.size
    w_sp = np.einsum('Ppq,Prs->pqrs', Bt, Bt, optimize=True)
    n = 2 * n_sp
    idx = np.arange(n)
    sp, sg = idx // 2, idx % 2
    same = sg[:, None] == sg[None, :]
    w = w_sp[np.ix_(sp, sp, sp, sp)] * same[:, :, None, None] * same[None, None, :, :]
    eps = eps_gw[sp]
    no2 = 2 * no
    o, v = slice(0, no2), slice(no2, n)
    eo, ev = eps[o], eps[v]
    # - W_ac,bk W_jk,ic / (w - (e_a + e_c - e_k - e_j))
    R1 = 1.0 / (omega - (ev[:, None, None, None] + ev[None, :, None, None]
                         - eo[None, None, :, None] - eo[None, None, None, :]))
    t1 = -np.einsum('abck,jikc,ackj->iajb', w[v, v, v, o], w[o, o, o, v], R1)
    # - W_ak,bc W_ki,cj / (w - (e_c + e_b - e_i - e_k))
    t2 = -np.einsum('abkc,kcij,cbik->iajb', w[v, v, o, v], w[o, v, o, o], R1)
    # + W_aj,cd W_cd,ib / (w - (e_c + e_d - e_j - e_i))
    t3 = np.einsum('acjd,cidb,cdji->iajb', w[v, v, o, v], w[v, o, v, v], R1)
    # + W_aj,kl W_kl,ib / (w - (e_a + e_b - e_k - e_l))
    t4 = np.einsum('akjl,kilb,abkl->iajb', w[v, o, o, o], w[o, o, o, v], R1)
    T = t1 + t2 + t3 + t4
    S = T[::2, ::2, ::2, ::2] + T[::2, ::2, 1::2, 1::2]
    Tt = T[::2, ::2, ::2, ::2] - T[::2, ::2, 1::2, 1::2]
    ns = S.shape[0] * S.shape[1]
    return {'singlet': S.reshape(ns, ns), 'triplet': Tt.reshape(ns, ns)}


def transcription_eq61(eps, eps_gw, B, no, omega):
    """Theta^61(omega) = J.K + K.J on one spin block, (ns, ns) in (ia, jb), from
    gw_modes' dense W at the HF energies: J_ia,kcm = -delta_ac W^m_ik, K_ia,kcm =
    +delta_ik W^m_ac as dense (ns, no nv nm) matrices by the deltas, X.Y = sum_K X_ia,K
    Y_jb,K / (omega - D_K) with D_kcm = Omega_m + e_c - e_k at the GW energies."""
    nv = len(eps) - no
    ns = no * nv
    om, W = ee_gw_pieces.gw_modes(eps, B, no, 'rpa')
    nm = om.size
    eo, ev = eps_gw[:no], eps_gw[no:]
    J = np.zeros((no, nv, no, nv, nm))
    K = np.zeros((no, nv, no, nv, nm))
    for a in range(nv):
        # J_ia,kam = -W^m_ik
        J[:, a, :, a, :] = -W[:no, :no, :]
    for i in range(no):
        # K_ia,icm = +W^m_ac
        K[i, :, i, :, :] = W[no:, no:, :]
    J = J.reshape(ns, no * nv * nm)
    K = K.reshape(ns, no * nv * nm)
    D = (om[None, None, :] + ev[None, :, None] - eo[:, None, None]).ravel()
    R = 1.0 / (omega - D)
    return J @ (R[:, None] * K.T) + K @ (R[:, None] * J.T)


def theta_from_pieces(P, omega, spin, ns):
    """A_eff(omega) - A_eff(None) of the pieces in one channel, (ns, ns), and the
    distance of A_eff(None) from M's channel matrix."""
    Ma, Mx = P['M'].get('aaaa'), P['M'].get('aabb')
    sgn = 1.0 if spin == 'singlet' else -1.0
    A_w = ee_fold.dense_effective(ee_fold.folded_operator(P, omega, spin)[0], ns)
    A_0 = ee_fold.dense_effective(ee_fold.folded_operator(P, None, spin)[0], ns)
    d0 = float(np.max(np.abs(A_0 - (Ma + sgn * Mx).reshape(ns, ns))))
    return A_w - A_0, d0


def check_transcription(eps, B, no):
    """Theta(omega) from the pieces, A_eff(omega) - M per channel, against the
    spin-orbital transcription of each kernel; eq 71's triplet zero, the printed one's
    carrying the loop terms; the other kernel's transcription is the negative
    control; eq 61's signed cross against its transcription, equal in both channels;
    the two-set Theta the sum of the sets'."""
    ok = True
    nv = len(eps) - no
    ns = no * nv
    eps_gw = gw_like(eps, no)
    omega = 0.3
    got, ref = {}, {}
    _, Bt = ee_dbse2_pieces.dressed_factors(eps, B, no)
    for kernel in ee_dbse2_pieces.KERNELS:
        P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel=kernel)
        dB = float(np.max(np.abs(P['Bt'] - Bt)))
        ok &= check(dB < 1e-14, f"{kernel}: the builder's dressed factors are "
                    "dressed_factors' (check 5 pins those to W_aux)", f'{dB:.1e}')
        ref[kernel] = transcription(Bt, eps_gw, no, omega, kernel)
        for spin in ('singlet', 'triplet'):
            got[kernel, spin], d0 = theta_from_pieces(P, omega, spin, ns)
            d = float(np.max(np.abs(got[kernel, spin] - ref[kernel][spin])))
            ok &= check(d < 1e-11 and d0 < 1e-12,
                        f'{kernel} {spin}: Theta(omega) from the pieces equals the '
                        'transcription', f'max |d| {d:.1e} Ha, M {d0:.1e}')
    # the vertex table against eq 74 as printed, term by term
    paper = transcription_printed_eq74(Bt, eps_gw, no, omega)
    for spin in ('singlet', 'triplet'):
        d = float(np.max(np.abs(ref['printed'][spin] - paper[spin])))
        ok &= check(d < 1e-11, f"printed {spin}: the vertex table's Theta equals eq "
                    "74's four terms as printed", f'max |d| {d:.1e} Ha')
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
    d4b = float(np.max(np.abs(got['printed', 'singlet'] - ref['eq71']['singlet'])))
    ok &= check(d4b > 1e-3,
                "control: the printed pieces differ from eq 71's transcription",
                f'max |d| {d4b:.1e} Ha')
    # eq 61: the signed cross J.K + K.J of the modes, no cross-spin coupling
    T61 = transcription_eq61(eps, eps_gw, B, no, omega)
    P61 = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, sets=('eq61',))
    for spin in ('singlet', 'triplet'):
        got['eq61', spin], d0 = theta_from_pieces(P61, omega, spin, ns)
        d = float(np.max(np.abs(got['eq61', spin] - T61)))
        ok &= check(d < 1e-11 and d0 < 1e-12, f'eq61 {spin}: Theta(omega) from the '
                    'pieces equals the transcription J.K + K.J',
                    f'max |d| {d:.1e} Ha, |Theta| {np.abs(T61).max():.1e}, M {d0:.1e}')
    d5 = float(np.max(np.abs(got['eq61', 'singlet'] - got['eq61', 'triplet'])))
    ok &= check(d5 < 1e-12, 'eq61: the singlet and triplet Theta are equal (no '
                "exchange in eq 61's couplings)", f'max |d| {d5:.1e} Ha')
    d6 = float(np.max(np.abs(got['eq61', 'singlet'] - got['eq71', 'singlet'])))
    ok &= check(d6 > 1e-3, "control: eq 61's Theta differs from eq 71's",
                f'max |d| {d6:.1e} Ha')
    P2 = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, sets=('eq61', 'eq71'))
    for spin in ('singlet', 'triplet'):
        T2, d0 = theta_from_pieces(P2, omega, spin, ns)
        d = float(np.max(np.abs(T2 - got['eq61', spin] - got['eq71', spin])))
        ok &= check(d < 1e-12 and d0 < 1e-12, f'eq61+eq71 {spin}: Theta is the sum of '
                    "the two sets' Theta", f'max |d| {d:.1e} Ha')
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
    the flat doubles of key 'a' then key 'b', from the pieces by unit vectors; eta is
    the segments' signs, so Z = (omega - D)^-1 V y gives the module's A_eff. Returns
    (H, ns, eta_full) with eta_full +1 on the singles."""
    m0, _, diag_s, _, _ = ee_fold.folded_operator(P, None, spin=None)
    ns = diag_s.size
    M = ee_fold.dense_effective(m0, ns)
    V, eta = dense_coupling(P)
    D = np.tile(P['D'], 2)
    H = np.block([[M, (eta[:, None] * V).T], [V, np.diag(D)]])
    return H, ns, np.concatenate([np.ones(ns), eta])


def flip_partner(ns, n_flat):
    """The alpha<->beta flip on the supermatrix's basis as an involution perm (n,):
    singles aa <-> bb, the flat doubles of key 'a' <-> those of key 'b' at the same
    position (the segments of the two keys are the spin-flipped names on the same
    slices, their inner order untouched)."""
    perm = np.empty(2 * ns + 2 * n_flat, int)
    perm[:ns] = np.arange(ns, 2 * ns)
    perm[ns:2 * ns] = np.arange(ns)
    perm[2 * ns:2 * ns + n_flat] = np.arange(2 * ns + n_flat, 2 * ns + 2 * n_flat)
    perm[2 * ns + n_flat:] = np.arange(2 * ns, 2 * ns + n_flat)
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
    roots complete; Newton against the Davidson (and the block solver on the mode
    sets); the regression guard on sets=('eq71',); the Hermitian misfold as control."""
    ok = True
    eps, B, no = water(basis)
    nv = len(eps) - no
    ns = no * nv
    eps_gw = gw_like_uniform(eps, no)
    for sets in ee_dbse2_pieces.SETS:
        tag = '+'.join(sets)
        solvers = [('newton', 'newton', 0), ('davidson', 'davidson', 0),
                   ('default', 'davidson', None)]
        if sets in MODE_SETS:
            solvers.append(('block', 'block', 0))
        # 6-31G runs the eq 71 kernel alone: its supermatrix is 6440 per channel (an
        # eig of minutes per kernel and spin) and 9640 on the two-set level; the
        # kernel axis is STO-3G's rows here and check 3's rows
        kernels = ('eq71',) if basis == '6-31g' else ee_dbse2_pieces.KERNELS
        for kernel in kernels:
            t0 = time.perf_counter()
            # the level's singles block: eq 62 beside eq 61's set, whose doubles carry
            # the static W; eq 62 plus eq 59 on eq 71's set alone
            singles = 'eq62' if 'eq61' in sets else 'eq62+eq59'
            P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel=kernel,
                                                   sets=sets, singles=singles)
            H, ns_full, eta_full = dense_signed_supermatrix(P)
            perm = flip_partner(ns, P['D'].size)
            d_min = float(P['D'].min())
            blocks = {}
            for spin, sign in (('singlet', 1.0), ('triplet', -1.0)):
                Hs, reps = channel_block(H, perm, sign)
                blocks[spin] = (Hs, eta_full[reps], ns)
            if basis == 'sto-3g':
                blocks[None] = (H, eta_full, ns_full)
            for spin, (Hs, wts, nsing) in blocks.items():
                t1 = time.perf_counter()
                w, v = scipy.linalg.eig(Hs)
                t_eig = time.perf_counter() - t1
                real = np.abs(w.imag) < 1e-9
                wr, vr = w[real].real, v[:, real].real
                below = np.sort(wr[wr < d_min])
                cplx = w[~real].real
                above = float(cplx.min() - d_min) if cplx.size else float('nan')
                print(f'    {tag} {kernel} {basis} {spin}: {int((~real).sum())} '
                      f'complex eigenvalues of {w.size}, {below.size} real below min D '
                      f'= {d_min:.3f} Ha, lowest complex Re {above:+.3f} Ha from min '
                      f'D; eig {t_eig:.1f} s')
                ok &= check(below.size >= 6, f'{tag} {kernel} {basis} {spin}: at least '
                            'six real eigenvalues below min D', f'{below.size}')
                nroots = int(min(below.size, 6))
                results = {}
                for label, solver, dl in solvers:
                    kw = dict(spin=spin, solver=solver, tol_omega=1e-12,
                              tol_residual=1e-10)
                    if dl is not None:
                        kw['dense_limit'] = dl
                    res, mv = solve_counted(P, nroots, **kw)
                    results[label] = res
                    loops = dict(collections.Counter(res.loop))
                    if res.omega.size == 0:
                        ok &= check(False, f'{tag} {kernel} {basis} {spin} {label}: '
                                    'no roots')
                        continue
                    if (basis, label, sets) == ('6-31g', 'davidson', ('eq71',)):
                        # the model's bracket follows a rising branch (the printed
                        # kernel's T1 > 1 roots) as well as a falling one
                        ok &= check(all(l == 'davidson' for l in res.loop),
                                    f'{tag} {kernel} {basis} {spin}: the Davidson '
                                    'model keeps its roots', f'loops {loops}')
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
                                f'{tag} {kernel} {basis} {spin} {label}: roots and T1 '
                                'are eigenpairs of the signed supermatrix, lowest '
                                f'{nroots} complete', f'max |dw| {max(dw):.1e} Ha, max '
                                f'|dT1| {max(dt):.1e}, {mv} matvecs, loops {loops}')
                a, b = results['newton'], results['davidson']
                n = min(a.omega.size, b.omega.size, nroots)
                dw = float(np.max(np.abs(a.omega[:n] - b.omega[:n]))) if n else np.inf
                dt = float(np.max(np.abs(a.t1[:n] - b.t1[:n]))) if n else np.inf
                ok &= check(dw < 1e-8 and dt < 1e-8,
                            f'{tag} {kernel} {basis} {spin}: newton and davidson agree',
                            f'max |dw| {dw:.1e} Ha, max |dT1| {dt:.1e}')
                if sets == ('eq71',) and spin is not None:
                    # the regression guard: the recorded roots
                    ref = np.array(M0_ROOTS[basis, kernel, spin])
                    dg = float(np.max(np.abs(np.sort(a.omega)[:2] - ref)))
                    ok &= check(dg < 1e-8, f'{tag} {kernel} {basis} {spin}: the two '
                                'lowest newton roots equal the recorded values',
                                f'max |dw| {dg:.1e} Ha')
            if basis == 'sto-3g':
                # the control: the Hermitian misfold [[M, V^T], [V, D]] has other roots,
                # on the singles it was calibrated with (eq 62 plus eq 59 on every set,
                # the nearest misfold root 5e-3 Ha or more away; the bare singles bring
                # the printed two-set root within 1e-4 Ha of one of them)
                if singles == 'eq62+eq59':
                    Pc, Hc, nsc, etac = P, H, ns_full, eta_full
                else:
                    Pc = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no,
                                                            kernel=kernel, sets=sets,
                                                            singles='eq62+eq59')
                    Hc, nsc, etac = dense_signed_supermatrix(Pc)
                Hm = Hc.copy()
                Hm[:nsc, nsc:] *= etac[nsc:][None, :]
                wm = np.sort(np.linalg.eigvalsh(Hm))
                res = ee_fold.solve_folded(Pc, 1, spin='singlet', tol_omega=1e-12)
                d = float(np.min(np.abs(wm - res.omega[0])))
                ok &= check(d > 1e-3, f'{tag} {kernel}: control, the Hermitian misfold '
                            'differs', f'nearest {d:.1e} Ha')
            print(f'    {tag} {kernel} {basis}: {time.perf_counter() - t0:.1f} s')
    return ok


def check_step():
    """fold_step on H2O/cc-pVDZ at QuAcK's G0W0 energies, eqs 61 and 71 with the bare
    singles: the printed kernel's step against QuAcK's singlet and the 2026-10-18
    triplet record; the eq 71 kernel's step as the control; the identities in static
    and in the norm of x; stationarity at the fold's root; the refusals."""
    ev = HARTREE_TO_EV
    mol = gto.M(atom=QUACK_H2O, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    no = mol.nelectron // 2
    B = DFIntegrals.from_scf(mol, mf, exact=True).B_aa
    eps_gw = QUACK_QP_EV / HARTREE_TO_EV
    w_aux = static_screened_coulomb_aux(eps, B, no)
    lr = LinearResponseSolver(eps_gw, coeff_df=B, spin_mode='restricted')
    pins = {'singlet': QUACK_STEP_S1, 'triplet': RECORD_STEP_T1}
    ok = True
    rows = {}
    for kernel in ('printed', 'eq71'):
        P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel=kernel,
                                               sets=('eq61', 'eq71'), singles='eq62')
        for spin in ('singlet', 'triplet'):
            A = lr.build_casida_matrices(no, lBSE=True, W_aux=w_aux,
                                         triplet=spin == 'triplet')[0]
            # the static Theta at the HF energies: QuAcK master's static convention
            theta = static_second_order_kernel_df(eps, B, w_aux, no, eta=0.0,
                                                  channel=spin, kernel=kernel)[0]
            w0, X0 = np.linalg.eigh(A + theta)
            om0, x = float(w0[0]), X0[:, 0]
            om, zeta, delta = ee_fold.fold_step(P, om0, spin, x)
            rows[kernel, spin] = (P, om0, x, om, zeta, delta)
            if kernel != 'printed':
                continue
            got = (om0 * ev, om * ev, zeta * delta * ev, zeta)
            d = max(abs(g - r) for g, r in zip(got, pins[spin]))
            ok &= check(d < 1e-5, f'printed {spin}: static, step, correction (eV) and '
                        'zeta equal the record', f'max |d| {d:.1e}, step '
                        f'{om * ev:.6f} eV, zeta {zeta:.6f}')
    dc = abs(rows['eq71', 'singlet'][3] * ev - QUACK_STEP_S1[1])
    ok &= check(dc > 1e-3, "eq71 singlet: control, the crossed kernel's step is not "
                "QuAcK's", f'{dc:.1e} eV away')
    P, om0, x, om, zeta, delta = rows['printed', 'singlet']
    same = ee_fold.fold_step(P, om0, 'singlet', x, static=om0)
    ok &= check(all(math.isclose(s, r, rel_tol=1e-15, abs_tol=0.0)
                    for s, r in zip(same, (om, zeta, delta))),
                'printed singlet: static=omega0 is the default, to 1e-15 relative')
    d = abs(ee_fold.fold_step(P, om0, 'singlet', x, static=om0 + 0.01)[0]
            - (om - zeta * 0.01))
    ok &= check(d < 1e-12, 'printed singlet: static raised by 0.01 Ha moves the step '
                'by -zeta 0.01', f'{d:.1e} Ha')
    om2, z2, d2 = ee_fold.fold_step(P, om0, 'singlet', 2.0 * x, static=4.0 * om0)
    r1 = abs(d2 - 4.0 * delta) / abs(4.0 * delta)
    g4 = 4.0 * (1.0 / zeta - 1.0)
    r2 = abs((1.0 / z2 - 1.0) - g4) / abs(g4)
    ok &= check(max(r1, r2) < 1e-12, 'printed singlet: x doubled, static times 4: '
                'delta and 1/zeta - 1 scale by 4', f'relative {r1:.1e}, {r2:.1e}')
    res = ee_fold.solve_folded(P, 1, spin='singlet', tol_omega=1e-10)
    om_f, y = float(res.omega[0]), res.y[:, 0]
    om_n, z_n, d_n = ee_fold.fold_step(P, om_f, 'singlet', y)
    dw, dz = abs(om_n - om_f), abs(z_n - res.t1[0])
    ok &= check(dw < 1e-9 and abs(d_n) < 1e-9 and dz < 1e-10, 'printed singlet: '
                "the step is stationary at the fold's root, zeta its T1",
                f'|dw| {dw:.1e} Ha, |delta| {abs(d_n):.1e} Ha, |dzeta| {dz:.1e}')
    try:
        ee_fold.fold_step(P, float(P['D'].min()) + 0.1, 'singlet', x)
        ok &= check(False, 'omega0 above min D raises ValueError')
    except ValueError as exc:
        ok &= check('omega0' in str(exc), 'omega0 above min D raises ValueError '
                    'naming omega0', str(exc)[:70])
    try:
        ee_fold.fold_step(P, om0, 'singlet', np.ones(x.size + 1))
        ok &= check(False, 'x of another length raises ValueError')
    except ValueError as exc:
        ok &= check(True, 'x of another length raises ValueError', str(exc)[:70])
    return ok & check_step_signed()


def check_step_signed():
    """fold_step on hand-built signed pieces, no = nv = 1, one '-' copy of one double
    with V_K,jb = 1 and D_K = 1 Ha: at omega0 = 0.5 Ha, x = (1, 0),
    1 + dmatvec(x) = 1 - 1 / (1 - 0.5)^2 = -3, so exactly one RuntimeWarning and no
    exception."""
    from src.SingleReference.ADC.eeADC import ee_equations
    Ma = np.full((1, 1, 1, 1), 0.4)
    M = SB({'aaaa': Ma, 'bbbb': Ma.copy(), 'aabb': np.zeros_like(Ma),
            'bbaa': np.zeros_like(Ma)})

    def V(y1):
        # Z_K = sum_jb V_K,jb y_jb on each spin block, V = 1
        return SB({s: y1.get(s).ravel().copy() for s in ('aa', 'bb')})

    def Vt(Z):
        # the eta-transpose of the '-' copy: w_jb = -sum_K V_K,jb Z_K
        return SB({s: -Z.get(s).reshape(1, 1) for s in ('aa', 'bb')})

    def dnorm2(Z):
        # the signed norm -|Z-|^2 of the '-' copy
        return -float(sum(Z.get(s) @ Z.get(s) for s in ('aa', 'bb')))

    P = {'M': M, 'V': V, 'Vt': Vt, 'D': np.array([1.0]), 'dnorm2': dnorm2,
         'level': 'dbse2', 'no': 1, 'nv': 1, 'be': ee_equations.SPIN_BLOCKED}
    x = np.array([1.0, 0.0])
    g = 1.0 + ee_fold.folded_operator(P, 0.5, None)[1](x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            om, zeta, delta = ee_fold.fold_step(P, 0.5, None, x)
            err = ''
        except Exception as exc:      # the row reports any raise
            om = zeta = delta = None
            err = f'{type(exc).__name__}: {exc}'
    n_rt = sum(issubclass(w.category, RuntimeWarning) for w in caught)
    return check(g <= 0.0 and not err and len(caught) == 1 and n_rt == 1,
                 'signed pieces, 1 + dmatvec(x) <= 0: one RuntimeWarning, no '
                 'exception', f'1 + dmatvec {g:+.6f}, {len(caught)} warnings, '
                 f'zeta {zeta}, step {om}' + (f', {err}' if err else ''))


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
    all_ok &= check_step()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
