"""On TDHF screening the self-energy vertex is the bare v the TDHF kernel uses.

PSD2@TDHF with a bare vertex is the single electron-hole pair of the Faddeev
expansion: its quasiparticle energy from the Casida route, on an RHF and on a
UHF reference, must equal the one from the eh pair built with spin-orbital
TDHF excitations, which hold every singlet and all three components of every
triplet (water/6-31G, density fitting with cc-pvdz-jkfit, tight SCF).

    python tests/test_tdhf_vertex_is_bare.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.pyscf_interface import DFIntegrals, get_antisymmetrized_spin_eri
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.Solvers.qp_equation import solve_qp_equation_pole_strength

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def check(ok, label, detail=''):
    msg = f"  [{'ok' if ok else 'FAIL'}] {label}"
    print(msg + (f'   ({detail})' if detail else ''))
    return bool(ok)


def water():
    """(mol, rhf, uhf): the UHF started from the RHF density, both tight."""
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    rhf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    uhf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    for mf in (rhf, uhf):
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    rhf.kernel()
    dm = rhf.make_rdm1()
    uhf.kernel(np.array([dm, dm]) / 2)
    return mol, rhf, uhf


def eh_pair_quasiparticle(mf, nocc, p):
    """Largest-Z root of w = eps_p + Sigma_eh(w), in eV, Sigma_eh from
    spin-orbital TDHF excitations."""
    eps = mf.mo_energy
    B = DFIntegrals.from_scf(mf.mol, mf).B_aa
    g = get_antisymmetrized_spin_eri(np.einsum('Qpq,Qrs->pqrs', B, B, optimize=True))
    es = np.repeat(eps, 2)
    no = 2 * nocc
    nv = len(es) - no
    n = no * nv
    o, v = slice(0, no), slice(no, None)
    A = g[v, o, o, v].transpose(2, 0, 1, 3).reshape(n, n).copy()
    A[np.diag_indices(n)] += (es[None, v] - es[o, None]).ravel()
    Bm = g[v, v, o, o].transpose(2, 0, 3, 1).reshape(n, n).copy()
    om, X, Y = CasidaSolver(A, Bm).solve()
    X3, Y3 = X.reshape(no, nv, -1), Y.reshape(no, nv, -1)
    P = 2 * p
    Mv = (np.einsum('iab,ian->bn', g[P, o, v, v], X3)
          + np.einsum('jcb,jcn->bn', g[P, v, o, v].transpose(1, 0, 2), Y3))
    Mo = (np.einsum('iaj,ian->jn', g[P, v, o, o].transpose(1, 0, 2), X3)
          + np.einsum('kcj,kcn->jn', g[v, o, P, o].transpose(2, 0, 1), Y3))

    def sigma(w):
        return (np.sum(Mv ** 2 / (w - es[v][:, None] - om[None, :]))
                + np.sum(Mo ** 2 / (w - es[o][:, None] + om[None, :])))
    return solve_qp_equation_pole_strength(lambda w: w - eps[p] - sigma(w),
                                           eps[p]) * HARTREE_TO_EV


def check_psd2_on_tdhf_is_the_eh_pair(mol, rhf, uhf):
    nocc = mol.nelectron // 2
    states = [nocc - 1, nocc]
    kw = dict(selfenergy='PSD2', polarizability='TDHF', mode='casida', df=True,
              state=states, eta=1e-10)
    qr = calc_qp_energy(rhf, **kw)
    qu = calc_qp_energy(uhf, spin_channel='alpha', **kw)
    ok = True
    for p in states:
        ref = eh_pair_quasiparticle(rhf, nocc, p)
        ok &= check(abs(qr[p]['PSD2'] - ref) < 1e-6, f'RHF PSD2@TDHF = eh pair, p={p}',
                    f"{qr[p]['PSD2'] - ref:+.1e} eV")
        ok &= check(abs(qu[p]['PSD2'] - ref) < 1e-6, f'UHF PSD2@TDHF = eh pair, p={p}',
                    f"{qu[p]['PSD2'] - ref:+.1e} eV")
    return ok


def run():
    return check_psd2_on_tdhf_is_the_eh_pair(*water())


def test_tdhf_vertex_is_bare():
    assert run()


if __name__ == '__main__':
    passed = run()
    print('ALL PASSED' if passed else 'FAILURES DETECTED')
    sys.exit(0 if passed else 1)
