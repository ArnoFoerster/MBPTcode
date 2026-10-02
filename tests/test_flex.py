"""One-shot FLEX on HF orbitals (ADC/adc_r_flex.py) against its definition.

1. The particle-hole term is the single eh pair in the ordered three-particle
   space: its restricted spin sums (singlet 1/2, triplet 3/2) must equal the
   spin-orbital single eh pair built from the spin-orbital TDHF phonons (the
   independent reference: FLEX evaluates the term through MBPTcode's PSD2),
   and PSD2 on its own dense-integral Casida solve.
2. FLEX = Sigma_eh + Sigma_pp - 2 Sigma(2) must be exact through third order:
   under v -> lam v its difference to ADC(3) and to Faddeev-ADC(3) must scale
   as lam^4. A wrong double-counting coefficient shows up as lam^2, a wrong
   third-order piece as lam^3.
3. The diagonal QP equation converges and returns a quasiparticle.

Run: python tests/test_flex.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_antisymmetrized_spin_eri
from src.SingleReference.ADC import ADCSolverRestricted
from src.SingleReference.ADC import adc_r_flex as FX
from src.SingleReference.ADC import adc_r_sigma_df
from src.SingleReference.ADC import adc_u_faddeev as FU
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from scipy.sparse.linalg import LinearOperator, minres


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build(basis='6-31g'):
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf.mo_energy, DFIntegrals.from_scf(mol, mf).B_aa


def eh_pair_spin_orbital(eps, B, nocc, p, w):
    """The single eh pair from the spin-orbital TDHF phonons, spatial p on its alpha row."""
    eri = np.einsum('Qpq,Qrs->pqrs', B, B, optimize=True)
    g = get_antisymmetrized_spin_eri(eri)
    es = np.repeat(eps, 2)
    no = 2 * nocc
    nv = len(es) - no
    A, Bm = FU.build_tdhf_matrices(es, g, no)
    om, X, Y = CasidaSolver(A, Bm).solve()
    o, v = slice(0, no), slice(no, None)
    P = 2 * p
    X3, Y3 = X.reshape(no, nv, -1), Y.reshape(no, nv, -1)
    M = (np.einsum('iab,ian->bn', g[P, o, v, v], X3)
         + np.einsum('jcb,jcn->bn', g[P, v, o, v].transpose(1, 0, 2), Y3))
    s = np.sum(M ** 2 / (w - es[v][:, None] - om[None, :]))
    M = (np.einsum('iaj,ian->jn', g[P, v, o, o].transpose(1, 0, 2), X3)
         + np.einsum('kcj,kcn->jn', g[v, o, P, o].transpose(2, 0, 1), Y3))
    return s + np.sum(M ** 2 / (w - es[o][:, None] + om[None, :]))


def psd2(eps, B, nocc, p, w):
    eri = np.einsum('Qpq,Qrs->pqrs', B, B, optimize=True)
    lr = LinearResponseSolver(eps, eri_chemist=eri)
    se = SelfEnergySolver(eps, eri_chemist=eri, eta=1e-12)
    om, X, Y = CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=True, W_aux=None,
                                                      triplet=False)).solve()
    omt, Xt, Yt = CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=True, W_aux=None,
                                                         triplet=True)).solve()
    return se.calculate_self_energy(
        p, w, nocc, om, se.get_chi_a(nocc, X, Y), se.get_chi_b_vertex(nocc, X, Y),
        omt, se.get_chi_b_vertex(nocc, Xt, Yt), vertex_mode='PSD2')


def sigma_adc3(s, nocc, p, w):
    aop, _, d = adc_r_sigma_df.build_operator(s, nocc)
    norb, nH = s.norb, d['nH']
    e = np.zeros(nH)
    e[p] = 1.0
    u = aop(e)[norb:]
    zero = np.zeros(norb)
    op = LinearOperator((nH - norb,) * 2, dtype=float,
                        matvec=lambda x: w * x - aop(np.concatenate([zero, x]))[norb:])
    x, info = minres(op, u, rtol=1e-13, maxiter=4000)
    assert info == 0
    return float(u @ x)


def check_eh_term(eps, B, nocc):
    ok = True
    s = ADCSolverRestricted.from_arrays(eps, B_aa=B, nocc=nocc)
    fx = FX.FlexSelfEnergy(s, nocc)
    for p in (nocc - 1, nocc - 2, nocc):
        w = eps[p] - 0.35
        se, _ = fx.sigma_eh(p, w)
        so = eh_pair_spin_orbital(eps, B, nocc, p, w)
        ps = psd2(eps, B, nocc, p, w)
        ok &= check(abs(se - so) < 1e-10 and abs(se - ps) < 1e-10,
                    f'p={p}: restricted eh term == spin-orbital eh pair == PSD2 (TDHF)',
                    f'{se:+.10f}; d {se - so:+.1e}, {se - ps:+.1e}')
    return ok


def check_third_order(eps, B0, nocc):
    ok = True
    p = nocc - 1
    rows = []
    for lam in (0.025, 0.05, 0.1):
        B = np.sqrt(lam) * B0
        s = ADCSolverRestricted.from_arrays(eps, B_aa=B, nocc=nocc)
        fx = FX.FlexSelfEnergy(s, nocc)
        w = eps[p] - 0.35
        f, _ = fx.sigma_flex(p, w)
        fa, _ = fx.sigma_operator(('faddeev', 'rpa'), p, w)
        a3 = sigma_adc3(s, nocc, p, w)
        rows.append((lam, f, abs(f - a3), abs(f - fa)))
    sl_a = np.log(rows[-1][2] / rows[0][2]) / np.log(rows[-1][0] / rows[0][0])
    sl_f = np.log(rows[-1][3] / rows[0][3]) / np.log(rows[-1][0] / rows[0][0])
    for lam, f, da, df in rows:
        print(f"      lam {lam:5.3f}: FLEX {f:+.4e}  |FLEX-ADC(3)| {da:.3e}  "
              f"|FLEX-Faddeev| {df:.3e}")
    ok &= check(sl_a > 3.7, 'FLEX - ADC(3) ~ lam^4 (exact through third order)',
                f'slope {sl_a:.2f}')
    ok &= check(sl_f > 3.7, 'FLEX - Faddeev-ADC(3) ~ lam^4', f'slope {sl_f:.2f}')
    return ok


def check_qp(eps, B, nocc):
    s = ADCSolverRestricted.from_arrays(eps, B_aa=B, nocc=nocc)
    fx = FX.FlexSelfEnergy(s, nocc)
    r = fx.solve_qp(nocc - 1)
    return check(r['converged'] and 0.5 < r['Z'] < 1.0,
                 'diagonal FLEX QP converges to a quasiparticle (HOMO)',
                 f"IP {-r['e'] * 27.211386245988:.4f} eV, Z {r['Z']:.3f}, "
                 f"{r['niter']} Newton steps")


def run():
    warnings.simplefilter('ignore')
    mol, eps, B = build()
    nocc = mol.nelectron // 2
    all_ok = True
    print('\n-- the eh term is the single eh pair (and PSD2 on TDHF phonons)')
    all_ok &= check_eh_term(eps, B, nocc)
    print('\n-- exact through third order')
    all_ok &= check_third_order(eps, B, nocc)
    print('\n-- the quasiparticle equation')
    all_ok &= check_qp(eps, B, nocc)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return all_ok


def test_flex_against_its_definition():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
