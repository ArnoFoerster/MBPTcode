"""The second-order self-energy Sigma(2) (ADC/second_order.py) against its
definition.

1. The restricted D + X equals the spin-orbital Sigma(2) from the
   antisymmetrized integrals,
       1/2 sum_iab |<pi||ab>|^2 / (w + eps_i - eps_a - eps_b)
     + 1/2 sum_ija |<pa||ij>|^2 / (w + eps_a - eps_i - eps_j),
   for occupied and virtual p; and its slope equals a finite difference.
2. The DF and the dense-integral routes agree on the same (DF) integrals.
3. It is the GW machinery on the bare pairs: SelfEnergySolver's GWGammaInf
   (GW + SOSEX) with X = 1, Y = 0 is Sigma(2) exactly.
4. D alone is the direct term: the spin-orbital sum with the exchange
   integrals dropped.

Run: python tests/test_sigma2.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_antisymmetrized_spin_eri
from src.SingleReference.ADC.second_order import sigma2, sigma2_poles
from src.SingleReference.GW.self_energy import SelfEnergySolver


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    B = DFIntegrals.from_scf(mol, mf).B_aa
    return mol, mf.mo_energy, B, np.einsum('Qpq,Qrs->pqrs', B, B, optimize=True)


def sigma2_spin_orbital(eps, eri, nocc, p, w, exchange=True):
    """Sigma(2) on the alpha row of spatial p from the spin-orbital
    integrals: 1/2 |<pi||ab>|^2 + 1/2 |<pa||ij>|^2 terms, or with
    exchange=False the direct term alone, |<pi|ab>|^2 + |<pa|ij>|^2."""
    es = np.repeat(eps, 2)
    if exchange:
        g, f = get_antisymmetrized_spin_eri(eri), 0.5
    else:
        sp, sg = np.arange(len(es)) // 2, np.arange(len(es)) % 2
        g = (eri.transpose(0, 2, 1, 3)[np.ix_(sp, sp, sp, sp)]            # <pq|rs>
             * (sg[:, None, None, None] == sg[None, None, :, None])
             * (sg[None, :, None, None] == sg[None, None, None, :]))
        f = 1.0
    o, v = slice(0, 2 * nocc), slice(2 * nocc, len(es))
    q = 2 * p
    gp = g[q, o, v, v]                              # <pi||ab> or <pi|ab>
    gh = g[q, v, o, o]                              # <pa||ij> or <pa|ij>
    dp = w + es[o][:, None, None] - es[v][None, :, None] - es[v][None, None, :]
    dh = w + es[v][:, None, None] - es[o][None, :, None] - es[o][None, None, :]
    return f * (np.sum(gp ** 2 / dp) + np.sum(gh ** 2 / dh))


def check_against_definition(eps, B, eri, nocc):
    ok = True
    for p in (nocc - 1, 0, nocc):
        w = eps[p] - 0.37
        s, ds = sigma2(eps, nocc, p, w, B_aa=B)
        ref = sigma2_spin_orbital(eps, eri, nocc, p, w)
        h = 1e-5
        fd = (sigma2(eps, nocc, p, w + h, B_aa=B)[0]
              - sigma2(eps, nocc, p, w - h, B_aa=B)[0]) / (2 * h)
        ok &= check(abs(s - ref) < 1e-12 and abs(ds - fd) < 1e-7,
                    f'p={p}: restricted D + X == spin-orbital Sigma(2); slope',
                    f'{s:+.10f}; d {s - ref:+.1e}, slope d {ds - fd:+.1e}')
        rD, rX, pol = sigma2_poles(eps, nocc, p, B_aa=B)
        direct = float(np.sum(rD / (w - pol)))
        ref_D = sigma2_spin_orbital(eps, eri, nocc, p, w, exchange=False)
        ok &= check(abs(direct - ref_D) < 1e-12, f'p={p}: D == the direct term',
                    f'd {direct - ref_D:+.1e}')
    return ok


def check_routes(eps, B, eri, nocc):
    ok = True
    for p in (nocc - 1, nocc):
        a = sigma2_poles(eps, nocc, p, B_aa=B)
        b = sigma2_poles(eps, nocc, p, eri_chemist=eri)
        d = max(np.abs(x - y).max() for x, y in zip(a, b))
        ok &= check(d < 1e-12, f'p={p}: DF == dense route', f'{d:.1e}')
    return ok


def check_gw_machinery(eps, B, nocc):
    ok = True
    n = nocc * (len(eps) - nocc)
    X, Y = np.eye(n), np.zeros((n, n))
    D = (eps[None, nocc:] - eps[:nocc, None]).ravel()
    se = SelfEnergySolver(eps, df_coeff=B, eta=0.0)
    for p in (nocc - 1, nocc):
        w = eps[p] - 0.37
        ev = se.self_energy_evaluator(p, nocc, D, se.get_chi_a(nocc, X, Y, p_state=p),
                                      se.get_chi_b_vertex(nocc, X, Y, p_state=p),
                                      vertex_mode='GWGammaInf')
        res, pol = ev.poles()
        gw = float(np.sum(res / (w - pol)))
        s, _ = sigma2(eps, nocc, p, w, B_aa=B)
        ok &= check(abs(gw - s) < 1e-12, f'p={p}: GWGammaInf on bare pairs == Sigma(2)',
                    f'd {gw - s:+.1e}')
    return ok


def run():
    warnings.simplefilter('ignore')
    mol, eps, B, eri = build()
    nocc = mol.nelectron // 2
    all_ok = True
    print('\n-- against the spin-orbital definition')
    all_ok &= check_against_definition(eps, B, eri, nocc)
    print('\n-- DF and dense routes')
    all_ok &= check_routes(eps, B, eri, nocc)
    print('\n-- the GW machinery on bare pairs')
    all_ok &= check_gw_machinery(eps, B, nocc)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return all_ok


def test_sigma2_against_definition():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
