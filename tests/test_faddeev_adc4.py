"""Faddeev-ADC(4): the Faddeev pencil with the second-order parquet kernel,
dense spin-orbital (ADC/adc_u_faddeev4.py), against diagrammatic Dyson ADC(4)
(sigma_driver.build_sigma_operator, construction 'diagrammatic').

What pins what:

1. The k1 basis. The spin-orbital ADC(3) supermatrix of the generated
   operator equals adc_u_dense_full's, so the Faddeev blocks (built on
   adc_u_dense_full's configurations) sit in the ADC(4) operator's basis.
2. E3. The ADC(4) k1-k2 block equals <J|V_N|P> by Slater-Condon (the
   emission terms of the three-particle propagator), up to the phases of the
   configurations, in both sectors.
3. The pieces. The static line dressing equals its loop definition (the
   spin-orbital formulas element by element), and the coupling remainder
   N^{-1/2} U_F - U_an^(<=3) goes as lambda^4.
4. The lambda^5 limit. v -> lambda v at fixed
   orbital energies, H2O and LiH in STO-3G: the dynamic self-energy of the
   pencil differs from ADC(4)'s as lambda^5 (ratio -> 32 per halving), on the
   RPA ('phonon') and the CCD route. Controls go as lambda^4 (ratio -> 16):
   the pencil without dC, and Faddeev-ADC(3) alone.
   The pencil is taken in its symmetric form: k1 block
   N^{-1/2}(H_F + dC)N^{-1/2}, k1-k2 coupling N^{-1/2} E3^+. The form
   N^{-1/2} H_F N^{-1/2} + dC with the unscaled E3 differs from it at
   O(v^5) in Sigma, so both pass this check.
5. At lambda = 1 the pencil is symmetric (its principal IPs are printed
   next to ADC(4)'s, not gated: they differ by the resummation).

Run: python tests/test_faddeev_adc4.py, or under pytest.
"""
import itertools as it
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.pyscf.pyscf_interface import get_antisymmetrized_spin_eri
from src.SingleReference.ADC import ADCSolverUnrestricted
from src.SingleReference.ADC import adc_u_dense_full
from src.SingleReference.ADC import adc_u_faddeev as FU
from src.SingleReference.ADC import adc_u_faddeev4 as F4
from src.SingleReference.ADC.precursor_classes import class_configs
from src.SingleReference.ADC.sigma_driver import build_sigma_operator
from tests.generated_runtime import requires_runtime
from tests.test_faddeev_adc3 import ccd_spin_orbital

pytestmark = requires_runtime(4)

MOLS = {'h2o': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
        'lih': 'Li 0 0 0; H 0 0 1.595'}
LAMBDAS = (0.08, 0.04, 0.02)
OMEGAS = np.array([-0.6 + 0.4j, -0.2 + 0.4j, 0.4 + 0.4j, 1.0 + 0.4j])
RATIO_5 = 28.0                 # last halving, lambda^5 (32 asymptotically)
RATIO_4 = (13.0, 20.0)         # last halving, lambda^4 (16)
_CACHE = {}
_ADC4 = {}


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def system(name):
    if name not in _CACHE:
        mol = gto.M(atom=MOLS[name], basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-12, 300
        mf.run()
        eri = mol.ao2mo(mf.mo_coeff, compact=False).reshape((mol.nao,) * 4)
        _CACHE[name] = (np.repeat(mf.mo_energy, 2), get_antisymmetrized_spin_eri(eri),
                        mol.nelectron)
    return _CACHE[name]


def solver(eps, g, nocc):
    return ADCSolverUnrestricted.from_arrays(eps, g, nocc=nocc)


def adc4(name, lam):
    """The dense ADC(4) supermatrix of `name` at v -> lam v, built once."""
    if (name, lam) not in _ADC4:
        eps, g, nocc = system(name)
        _ADC4[(name, lam)] = F4.adc4_supermatrix(solver(eps, lam * g, nocc), nocc)
    return _ADC4[(name, lam)]


def sigma_dyn(M, norb, omegas=OMEGAS):
    """U^+ (omega - M_sat)^{-1} U on the orbital block, per omega."""
    U, S = M[norb:, :norb], M[norb:, norb:]
    w, V = np.linalg.eigh(S)
    UV = V.T @ U
    return np.array([(UV.T / (z - w)) @ UV for z in omegas])


# ---------------------------------------------------------------------------
# 1. the k1 basis
# ---------------------------------------------------------------------------

def check_basis(eps, g, nocc):
    s = solver(eps, g, nocc)
    op, _, _ = build_sigma_operator(s, nocc, 'adc3')
    M = F4._dense(op)
    ref = adc_u_dense_full.build_supermatrix(s, nocc)
    d = np.abs(M - ref).max()
    return check(d < 1e-12, 'generated ADC(3) == adc_u_dense_full ADC(3) (k1 basis)', f'{d:.1e}')


# ---------------------------------------------------------------------------
# 2. E3 by Slater-Condon
# ---------------------------------------------------------------------------

def _excite(ops, orbs):
    """Apply a+/a (('+'|'-', p), rightmost first) to the sorted determinant
    `orbs`; (sign, tuple) or None."""
    sgn, orbs = 1, list(orbs)
    for kind, p in reversed(ops):
        k = sum(1 for q in orbs if q < p)
        if kind == '-':
            if p not in orbs:
                return None
            orbs.remove(p)
        else:
            if p in orbs:
                return None
            orbs.insert(k, p)
        sgn *= (-1) ** k
    return sgn, tuple(orbs)


def slater_condon_coupling(g, nocc, norb, sector):
    """<J|H|P> for the k1 configurations P and k2 configurations J of
    `sector` (determinants a+..a..|HF> in class_configs order), J != P, with
    canonical HF orbitals: a single replacement p -> q gives
    sum_{m in P minus HF} <qm||pm> - sum_{m in HF minus P} <qm||pm>, a double
    <qr||ps>. Rows k2, columns k1."""
    hf = tuple(range(nocc))

    def dets(k):
        cfg = class_configs(nocc, norb, k, sector)
        nh = k + 1 if sector == 'IP' else k
        out = []
        for c in zip(*cfg):
            holes, parts = c[:nh], c[nh:]
            ops = [('+', p) for p in parts] + [('-', h) for h in holes]
            out.append(_excite(ops, hf))
        return out

    P, J = dets(1), dets(2)
    E = np.zeros((len(J), len(P)))
    for c, (sp, dp) in enumerate(P):
        sP = set(dp)
        for r, (sj, dj) in enumerate(J):
            sJ = set(dj)
            out_, in_ = sorted(sP - sJ), sorted(sJ - sP)
            if len(in_) == 1:
                p, q = out_[0], in_[0]
                val = (sum(g[q, m, p, m] for m in sP - set(hf))
                       - sum(g[q, m, p, m] for m in set(hf) - sP))
                ph = _excite([('+', q), ('-', p)], dp)
            elif len(in_) == 2:
                (p, s_), (q, r_) = out_, in_
                val = g[q, r_, p, s_]
                ph = _excite([('+', q), ('+', r_), ('-', s_), ('-', p)], dp)
            else:
                continue
            assert ph[1] == dj
            E[r, c] = sp * sj * ph[0] * val
    return E


def same_up_to_phases(A, B, tol=1e-12):
    """max |A - D_r B D_c| over diagonal sign matrices fitted on the nonzero
    pattern (np.inf if no consistent signs exist)."""
    if np.abs(np.abs(A) - np.abs(B)).max() > tol:
        return np.abs(np.abs(A) - np.abs(B)).max()
    nz = np.abs(A) > tol
    dr = np.zeros(A.shape[0])
    dc = np.zeros(A.shape[1])
    for start in range(A.shape[1]):
        if dc[start] or not nz[:, start].any():
            continue
        dc[start] = 1.0
        stack = [('c', start)]
        while stack:
            kind, k = stack.pop()
            if kind == 'c':
                for r in np.nonzero(nz[:, k])[0]:
                    want = np.sign(A[r, k] * B[r, k]) * dc[k]
                    if dr[r] == 0:
                        dr[r] = want
                        stack.append(('r', r))
                    elif dr[r] != want:
                        return np.inf
            else:
                for c in np.nonzero(nz[k])[0]:
                    want = np.sign(A[k, c] * B[k, c]) * dr[k]
                    if dc[c] == 0:
                        dc[c] = want
                        stack.append(('c', c))
                    elif dc[c] != want:
                        return np.inf
    dr[dr == 0] = 1.0
    dc[dc == 0] = 1.0
    return np.abs(A - dr[:, None] * B * dc[None, :]).max()


def check_e3(eps, g, nocc, M4):
    s = solver(eps, g, nocc)
    d = s.dimensions_adc4(nocc)
    norb = len(eps)
    o1, o2, o3 = norb, norb + d['n2h1p'], d['nH']
    ok = True
    for sector, k1, k2 in (('IP', slice(o1, o2), slice(o3, o3 + d['n3h2p'])),
                           ('EA', slice(o2, o3), slice(o3 + d['n3h2p'], None))):
        E = slater_condon_coupling(g, nocc, norb, sector)
        dE = same_up_to_phases(M4[k2, k1], E)
        ok &= check(dE < 1e-12, f'E3 ({sector}) == <J|V_N|P> up to configuration phases',
                    f'{dE:.1e}')
    return ok


# ---------------------------------------------------------------------------
# 3. the pieces
# ---------------------------------------------------------------------------

def static_line_dressing_loops(eps, g, nocc):
    """dC element by element from the formulas of adc_u_faddeev4's docstring."""
    n = len(eps)
    nv = n - nocc
    o, v = slice(0, nocc), slice(nocc, n)
    eo, ev = eps[o], eps[v]
    sl = np.zeros((nv, nv))
    sg = np.zeros((nocc, nocc))
    for p, q in it.product(range(nv), repeat=2):
        for c, k, l in it.product(range(nv), range(nocc), range(nocc)):
            num = g[nocc + p, nocc + c, k, l] * g[k, l, nocc + q, nocc + c]
            sl[p, q] += 0.25 * num * (1 / (ev[p] + ev[c] - eo[k] - eo[l])
                                      + 1 / (ev[q] + ev[c] - eo[k] - eo[l]))
    for p, q in it.product(range(nocc), repeat=2):
        for k, c, dd in it.product(range(nocc), range(nv), range(nv)):
            num = g[p, k, nocc + c, nocc + dd] * g[nocc + c, nocc + dd, q, k]
            sg[p, q] += 0.25 * num * (1 / (eo[p] + eo[k] - ev[c] - ev[dd])
                                      + 1 / (eo[q] + eo[k] - ev[c] - ev[dd]))
    iu, ju = np.triu_indices(nocc, 1)
    au, bu = np.triu_indices(nv, 1)
    II = np.zeros((nocc * len(au),) * 2)
    for (i, x), (j, y) in it.product(it.product(range(nocc), range(len(au))), repeat=2):
        a, b, c, dd = au[x], bu[x], au[y], bu[y]
        val = 0.0
        if i == j:
            val += ((b == dd) * sl[a, c] - (a == dd) * sl[b, c]
                    - (b == c) * sl[a, dd] + (a == c) * sl[b, dd])
        if a == c and b == dd:
            val -= sg[i, j]
        II[i * len(au) + x, j * len(au) + y] = val
    I = np.zeros((len(iu) * nv,) * 2)
    for (x, a), (y, b) in it.product(it.product(range(len(iu)), range(nv)), repeat=2):
        i, j, k, l = iu[x], ju[x], iu[y], ju[y]
        val = 0.0
        if a == b:
            val += ((j == l) * sg[i, k] - (j == k) * sg[i, l]
                    - (i == l) * sg[j, k] + (i == k) * sg[j, l])
        if i == k and j == l:
            val -= sl[a, b]
        I[x * nv + a, y * nv + b] = val
    return I, II


def check_line_dressing(eps, g, nocc):
    dI, dII = F4.static_line_dressing(eps, g, nocc)
    rI, rII = static_line_dressing_loops(eps, g, nocc)
    d = max(np.abs(dI - rI).max(), np.abs(dII - rII).max())
    return check(d < 1e-14, 'static line dressing == its element-wise definition',
                 f'{d:.1e}, |dC| {np.abs(rII).max():.2e}')


# ---------------------------------------------------------------------------
# 4. the lambda^5 limit
# ---------------------------------------------------------------------------

def _ratios(rows, col):
    return [rows[n - 1][col] / rows[n][col] for n in range(1, len(rows))]


def lambda_scan(name, route):
    eps, g0, nocc = system(name)
    norb = len(eps)
    rows = []
    for lam in LAMBDAS:
        g = lam * g0
        s = solver(eps, g, nocc)
        t = ccd_spin_orbital(eps, g, nocc, tol=1e-15)[0] if route == 'ccd' else None
        A4 = adc4(name, lam)
        P = F4.build_supermatrix(s, nocc, route=route, t=t, adc4=A4)
        P0 = F4.build_supermatrix(s, nocc, route=route, t=t, adc4=A4, line_dressing=False)
        A3 = FU.build_supermatrix(s, nocc, route=route,
                                  channels=FU.pair_channels(eps, g, nocc, route=route, t=t))
        d = s.dimensions(nocc)
        k1 = slice(norb, d['nH'])
        rem = np.abs(P[k1, :norb] - A4[k1, :norb]).max()
        sR = sigma_dyn(A4, norb)
        diff = lambda M: np.abs(sigma_dyn(M, norb) - sR).max()
        rows.append((lam, diff(P), diff(P0), diff(A3), rem))
    print(f"   route {route!r}:  lambda  |dS| pencil  |dS| no dC  |dS| F-ADC(3)  |remainder|")
    for r in rows:
        print(f"      {r[0]:.3f}  {r[1]:.3e}  {r[2]:.3e}  {r[3]:.3e}  {r[4]:.2e}")
    return rows


def check_gate(name, route):
    rows = lambda_scan(name, route)
    rp, r0, r3, rr = (_ratios(rows, c) for c in (1, 2, 3, 4))
    print(f"   ratios: pencil {np.round(rp, 1)}  no dC {np.round(r0, 1)}  "
          f"F-ADC(3) {np.round(r3, 1)}  remainder {np.round(rr, 1)}")
    ok = check(rp[-1] > RATIO_5, f'{name} {route}: pencil - ADC(4) ~ lambda^5', f'{rp[-1]:.1f}')
    ok &= check(RATIO_4[0] < r0[-1] < RATIO_4[1], f'{name} {route}: without dC ~ lambda^4',
                f'{r0[-1]:.1f}')
    ok &= check(RATIO_4[0] < r3[-1] < RATIO_4[1], f'{name} {route}: Faddeev-ADC(3) ~ lambda^4',
                f'{r3[-1]:.1f}')
    # the coupling difference to ADC(4) is the remainder, O(v^4)
    ok &= check(RATIO_4[0] < rr[-1] < RATIO_4[1], f'{name} {route}: coupling remainder ~ lambda^4',
                f'{rr[-1]:.1f}')
    return ok


# ---------------------------------------------------------------------------
# 5. lambda = 1
# ---------------------------------------------------------------------------

def check_full_strength(name):
    eps, g, nocc = system(name)
    s = solver(eps, g, nocc)
    A4 = adc4(name, 1.0)
    P = F4.build_supermatrix(s, nocc, route='phonon', adc4=A4)
    norb = len(eps)
    ok = check(np.abs(P - P.T).max() < 1e-12, f'{name}: pencil symmetric')

    def ips(M):
        w, V = np.linalg.eigh(M)
        st = np.sum(V[:norb] ** 2, axis=0)
        keep = (w < 0) & (st > 0.5)
        return np.sort(w[keep])[::-1][:3]

    print(f"   {name} principal IPs (Eh): Faddeev-ADC(4) {np.round(ips(P), 6)}  "
          f"ADC(4) {np.round(ips(A4), 6)}")
    return ok


def run(names=('h2o', 'lih')):
    warnings.simplefilter('ignore')
    ok = True
    for name in names:
        eps, g, nocc = system(name)
        print(f"\n== {name}/sto-3g spin orbitals: norb {len(eps)}, nocc {nocc}")
        ok &= check_basis(eps, g, nocc)
        ok &= check_line_dressing(eps, g, nocc)
        ok &= check_e3(eps, g, nocc, adc4(name, 1.0))
        for route in ('phonon', 'ccd'):
            ok &= check_gate(name, route)
        ok &= check_full_strength(name)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


@pytest.mark.parametrize('name', ['h2o', 'lih'])
def test_faddeev_adc4_spin_orbital(name):
    assert run((name,))


if __name__ == '__main__':
    sys.exit(0 if run(tuple(sys.argv[1:]) or ('h2o', 'lih')) else 1)
