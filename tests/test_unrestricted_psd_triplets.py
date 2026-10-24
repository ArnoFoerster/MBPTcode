"""UHF and RHF give one PSD2/4/7/9 quasiparticle energy on a closed shell.

The triplet vertex modes sum over every triplet component. In spin orbitals,
PSD2 is the single electron-hole pair (Faddeev restricted to the eh channel):
the spin-conserving excitons carry the singlets and the M_s = 0 triplets, and
the two spin-flip channels carry M_s = +1 and -1. At fixed omega, with the bare
vertex and TDHF excitons, the restricted and unrestricted PSD2 self-energies
both equal that eh pair built from spin-orbital TDHF. calc_qp_energy (screened
vertex, the default) then gives one UHF and one RHF energy for all four modes,
with and without density fitting. Water/6-31G, tight SCF.

    python tests/test_unrestricted_psd_triplets.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.GW.self_energy import METHOD_REGISTRY
from src.Base.pyscf.pyscf_interface import (get_antisymmetrized_spin_eri,
                                            get_two_electron_integrals_chemist)
from src.SingleReference.GW import qp_energy
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.self_energy import SelfEnergySolver, TRIPLET_VERTEX_MODES
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def check(ok, label, detail=''):
    msg = f"  [{'ok' if ok else 'FAIL'}] {label}"
    print(msg + (f'   ({detail})' if detail else ''))
    return bool(ok)


def water(df):
    """(mol, rhf, uhf): the UHF started from the RHF density, both tight."""
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    rhf, uhf = scf.RHF(mol), scf.UHF(mol)
    if df:
        rhf = rhf.density_fit(auxbasis='cc-pvdz-jkfit')
        uhf = uhf.density_fit(auxbasis='cc-pvdz-jkfit')
    for mf in (rhf, uhf):
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    rhf.kernel()
    dm = rhf.make_rdm1()
    uhf.kernel(np.array([dm, dm]) / 2)
    return mol, rhf, uhf


def eh_pair_spin_orbital(eps, eri, nocc):
    """Sigma_pp(w) of the eh pair over spin-orbital TDHF: every singlet and
    all three components of every triplet."""
    g = get_antisymmetrized_spin_eri(eri)
    es = np.repeat(eps, 2)
    no = 2 * nocc
    nv = len(es) - no
    n = no * nv
    o, v = slice(0, no), slice(no, None)
    A = g[v, o, o, v].transpose(2, 0, 1, 3).reshape(n, n).copy()
    A[np.diag_indices(n)] += (es[None, v] - es[o, None]).ravel()
    B = g[v, v, o, o].transpose(2, 0, 3, 1).reshape(n, n).copy()
    om, X, Y = CasidaSolver(A, B).solve()
    X, Y = X.reshape(no, nv, -1), Y.reshape(no, nv, -1)

    def sigma(p, w):
        P = 2 * p
        M = (np.einsum('iab,ian->bn', g[P, o, v, v], X)
             + np.einsum('jcb,jcn->bn', g[P, v, o, v].transpose(1, 0, 2), Y))
        s = np.sum(M**2 / (w - es[v][:, None] - om[None, :]))
        M = (np.einsum('iaj,ian->jn', g[P, v, o, o].transpose(1, 0, 2), X)
             + np.einsum('kcj,kcn->jn', g[v, o, P, o].transpose(2, 0, 1), Y))
        return s + np.sum(M**2 / (w - es[o][:, None] + om[None, :]))
    return sigma


def check_psd2_is_the_eh_pair(mol, rhf, uhf):
    """Bare vertex, TDHF excitons, exact ERIs: RHF and UHF PSD2 Sigma at
    omega = eps_p - 0.35 and eps_p + 0.20 Ha against the spin-orbital eh pair."""
    nocc = mol.nelectron // 2
    eps = rhf.mo_energy
    eri = get_two_electron_integrals_chemist(mol, rhf, representation='spatial')
    reference = eh_pair_spin_orbital(eps, eri, nocc)

    lr = LinearResponseSolver(eps, eri_chemist=eri)
    se = SelfEnergySolver(eps, eri_chemist=eri, eta=1e-12)
    om_s, X_s, Y_s = CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=True, triplet=False)).solve()
    om_t, X_t, Y_t = CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=True, triplet=True)).solve()

    _, eri_u = qp_energy._two_electron_integrals(mol, uhf, False, True)
    eps_u = tuple(uhf.mo_energy)
    lr_u = LinearResponseSolver(eps_u, eri_chemist=eri_u, spin_mode='unrestricted')
    se_u = SelfEnergySolver(eps_u, eri_chemist=eri_u, spin_mode='unrestricted', eta=1e-12)
    infos = {'PSD2': METHOD_REGISTRY['PSD2']}
    spectrum = qp_energy.casida_spectrum(lr_u, uhf.nelec, 'TDHF', None, False, infos,
                                          ['PSD2'], True, False)
    ok = True
    for p in (nocc - 1, nocc):
        om_u, chi_a_u, chi_b_u, om_t_u, chi_b_t_u = qp_energy._self_energy_amplitudes(
            se_u, uhf.nelec, spectrum, infos, ['PSD2'], 'alpha', p, None, None, True,
            False)['PSD2']
        for dw in (-0.35, 0.20):
            w = eps[p] + dw
            ref = reference(p, w)
            s_r = se.calculate_self_energy(
                p, w, nocc, om_s, se.get_chi_a(nocc, X_s, Y_s),
                se.get_chi_b_vertex(nocc, X_s, Y_s), om_t,
                se.get_chi_b_vertex(nocc, X_t, Y_t), vertex_mode='PSD2')
            s_u = se_u.calculate_self_energy(
                p, w, uhf.nelec, om_u, chi_a_u, chi_b_u, eigenvalues_casida_t=om_t_u,
                chiXYb_t=chi_b_t_u, spin_channel='alpha', vertex_mode='PSD2')
            s_e = se_u.self_energy_evaluator(
                p, uhf.nelec, om_u, chi_a_u, chi_b_u, eigenvalues_casida_t=om_t_u,
                chiXYb_t=chi_b_t_u, spin_channel='alpha', vertex_mode='PSD2')(w)
            ok &= check(abs(s_r - ref) < 1e-10, f'RHF PSD2 = eh pair, p={p}, w=eps{dw:+.2f}',
                        f'{s_r - ref:+.1e} Ha')
            ok &= check(abs(s_u - ref) < 1e-7, f'UHF PSD2 = eh pair, p={p}, w=eps{dw:+.2f}',
                        f'{s_u - ref:+.1e} Ha')
            ok &= check(abs(s_e - s_u) < 1e-10, f'UHF evaluator = scalar, p={p}, w=eps{dw:+.2f}',
                        f'{s_e - s_u:+.1e} Ha')
    return ok


def check_rhf_equals_uhf(mol, rhf, uhf, df):
    """calc_qp_energy, PSD2/4/7/9, TDHF and BSE screening, HOMO and LUMO."""
    nocc = mol.nelectron // 2
    states = [nocc - 1, nocc]
    ok = True
    for polarizability in ('TDHF', 'BSE'):
        kw = dict(selfenergy=list(TRIPLET_VERTEX_MODES), polarizability=polarizability,
                  mode='casida', df=df, state=states)
        r = calc_qp_energy(rhf, **kw)
        u = calc_qp_energy(uhf, spin_channel='alpha', **kw)
        for p in states:
            for m in TRIPLET_VERTEX_MODES:
                d = u[p][m] - r[p][m]
                ok &= check(abs(d) < 1e-6, f'{m}@{polarizability} df={df} p={p}: UHF = RHF',
                            f'{d:+.1e} eV')
    return ok


def run():
    ok = True
    mol, rhf, uhf = water(df=False)
    ok &= check_psd2_is_the_eh_pair(mol, rhf, uhf)
    ok &= check_rhf_equals_uhf(mol, rhf, uhf, df=False)
    mol, rhf, uhf = water(df=True)
    ok &= check_rhf_equals_uhf(mol, rhf, uhf, df=True)
    return ok


def test_unrestricted_psd_triplets():
    assert run()


if __name__ == '__main__':
    passed = run()
    print('ALL PASSED' if passed else 'FAILURES DETECTED')
    sys.exit(0 if passed else 1)
