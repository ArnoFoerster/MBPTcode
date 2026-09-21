"""`solve_bse_df`: the density-fitted BSE twin of `solve_bse_isdf`.

Gated on what makes the two interchangeable: the matrix-free solve is the
dense Casida solve on the same factor, the quasiparticle diagonal IS the
Casida GW route's, the two factorizations of one interaction agree on the
spectrum to the fit error, evGW screens at its own fixed point, and a
continuum reaches the diagonal and the kernel the same way on both routes.

The last of those is the one worth reading twice. Non-equilibrium solvation is
two dielectric constants: the ground state relaxes at eps_static and only the
RESPONSE to the excitation is optical. `env.mean_field` applies the first,
`attach_environment` the second, and the test separates the two halves because
they are separately wrong in different ways.

Run: python tests/test_bse_df_driver.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import dft, gto

from src.Base.constants import HARTREE_TO_EV
from src.Base.environment import attach_environment
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.davidson import (solve_bse_df,
                                                         solve_bse_isdf)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

ATOM = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUXBASIS, XC = 'cc-pvdz', 'cc-pvdz-ri', 'pbe0'
NROOTS = 3
#: eV. Two factorizations of one interaction: the interpolative fit at its
#: default grid sits within this of pyscf's Coulomb fit on water's BSE roots.
FIT_TOL = 0.04
#: eV. The response half of a solvatochromic shift is a difference of two
#: solves on the same factor, so the fit error cancels to a few meV.
RESPONSE_TOL = 0.01


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def factory(mol):
    mf = dft.RKS(mol).density_fit(auxbasis=AUXBASIS)
    mf.xc = XC
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


def test_bse_at_the_mean_field_is_the_dense_casida_solve(mol, mf, nocc):
    """Same factor, same static W: Davidson and the dense [[A,B],[B,A]] agree
    to round-off, which fixes the kernel's gauge and normalization at once."""
    omega, _, _, info = solve_bse_df(mf, mol, nocc, nroots=NROOTS, qp=False,
                                     probe=False)
    coeff = get_density_fitting_coefficients(mol, mf)
    lr = LinearResponseSolver(get_orbital_energies(mf, representation='spatial'),
                              coeff_df=coeff, spin_mode='restricted')
    w = lr.static_screening_aux(nocc)
    a, b = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w)
    dense = np.sort(CasidaSolver(a, b).solve()[0])[:NROOTS]
    d = np.abs(np.sort(omega) - dense).max()
    return check(d < 1e-9 and np.abs(info['W_aux'] - w).max() < 1e-12,
                 'matrix-free == dense Casida on the same DF factor',
                 f'max |dw| {d:.1e} Ha')


def test_the_g0w0_diagonal_is_the_casida_gw_route(mol, mf, nocc):
    """Every orbital's quasiparticle energy on the diagonal is the one
    `calc_qp_energy(mode='casida')` returns, and GW opens the Kohn-Sham gap."""
    _, _, _, info = solve_bse_df(mol=mol, mf=mf, nocc=nocc, nroots=NROOTS,
                                 probe=False)
    n = len(info['eps'])
    out = calc_qp_energy(mf, mode='casida', state=list(range(n)))
    ref = np.array([out[p]['GW'] for p in range(n)]) / HARTREE_TO_EV
    d = np.abs(info['eps'] - ref).max()
    gap_ks = info['eps_mf'][nocc] - info['eps_mf'][nocc - 1]
    gap_qp = info['eps'][nocc] - info['eps'][nocc - 1]
    ok = check(d < 1e-10, 'the diagonal IS the Casida GW route', f'{d:.1e} Ha')
    return ok & check(gap_qp > gap_ks + 3.0 / HARTREE_TO_EV,
                      'and GW opens the Kohn-Sham gap',
                      f'{gap_ks * HARTREE_TO_EV:.2f} -> '
                      f'{gap_qp * HARTREE_TO_EV:.2f} eV')


def test_the_two_factorizations_agree_on_the_spectrum(mol, mf, nocc):
    """DF and ISDF are two fits of one interaction on one mean field; the
    BSE@G0W0 roots agree to the interpolative fit's error and no more."""
    om_df = np.sort(solve_bse_df(mf, mol, nocc, nroots=NROOTS, probe=False)[0])
    om_isdf = np.sort(solve_bse_isdf(mf, mol, nocc, nroots=NROOTS,
                                     auxbasis=AUXBASIS, probe=False)[0])
    d = np.abs(om_df - om_isdf).max() * HARTREE_TO_EV
    return check(d < FIT_TOL, 'the DF and ISDF routes agree to the fit error',
                 f'{d * 1e3:.1f} meV against a {FIT_TOL * 1e3:.0f} meV bar')


def test_evgw_converges_and_screens_at_its_fixed_point(mol, mf, nocc):
    """The loop runs on the Casida route; its fixed point opens the gap past
    G0W0 from a PBE0 start, sits on the diagonal, and rebuilds W."""
    _, _, _, g0w0 = solve_bse_df(mf, mol, nocc, nroots=NROOTS, probe=False)
    om, _, _, info = solve_bse_df(mf, mol, nocc, nroots=NROOTS, probe=False,
                                  self_consistency='evGW')

    def gap(e):
        return (e[nocc] - e[nocc - 1]) * HARTREE_TO_EV

    ok = check(info['evgw']['converged'] and gap(info['eps']) > gap(g0w0['eps']),
               'evGW converges and opens the gap past G0W0',
               f"{info['evgw']['cycles']} cycles, "
               f"{gap(g0w0['eps']):.3f} -> {gap(info['eps']):.3f} eV")
    ok &= check(np.abs(info['W_aux'] - g0w0['W_aux']).max() > 1e-4,
                'and W is rebuilt at the fixed point, not left at the mean field',
                f"max |dW| {np.abs(info['W_aux'] - g0w0['W_aux']).max():.1e}")
    om_isdf = solve_bse_isdf(mf, mol, nocc, nroots=NROOTS, auxbasis=AUXBASIS,
                             probe=False, self_consistency='evGW')[0]
    d = np.abs(np.sort(om) - np.sort(om_isdf)).max() * HARTREE_TO_EV
    ok &= check(d < FIT_TOL, 'both routes reach the same evGW-BSE roots',
                f'{d * 1e3:.1f} meV')
    # the kernel screens at the fixed point, evGW's W; evGW0 would keep another
    try:
        solve_bse_df(mf, mol, nocc, nroots=NROOTS, probe=False,
                     self_consistency='evGW', gw_kwargs={'screening': 'fixed'})
        ok &= check(False, "a 'screening' in gw_kwargs is refused")
    except ValueError as e:
        ok &= check('screening' in str(e), "a 'screening' in gw_kwargs is refused")
    return ok


def test_a_continuum_reaches_the_diagonal_and_the_kernel(mol, mf_gas, nocc):
    """Both halves of the reaction field, on both routes: the ground state at
    eps_static opens the roots (the frozen-polarization half), the response at
    eps_inf then moves them by the same amount through the two factorizations."""
    env = SolventScreening(mol, solvent='water')
    frozen = env.mean_field(mol, factory)
    solvated = env.mean_field(mol, factory)
    attach_environment(solvated, env)
    om_gas = np.sort(solve_bse_df(mf_gas, mol, nocc, nroots=NROOTS, probe=False)[0])
    om_0, _, _, info_0 = solve_bse_df(frozen, mol, nocc, nroots=NROOTS,
                                      probe=False)
    om, _, _, info = solve_bse_df(solvated, mol, nocc, nroots=NROOTS,
                                  probe=False)
    om_0, om = np.sort(om_0), np.sort(om)

    # water's lowest excitations are local and its virtuals diffuse: the
    # ground-state field opens them by several tenths of an eV
    ok = check((om_0 - om_gas).min() * HARTREE_TO_EV > 0.3,
               'the ground-state field at eps_static opens the roots',
               'shifts ' + ' '.join(f'{d * HARTREE_TO_EV:+.3f}'
                                    for d in om_0 - om_gas) + ' eV')
    # Eq. (18) sits on the diagonal: the solvated quasiparticle gap differs
    # from the frozen one while the two mean fields are the same
    d_gap = abs((info['eps'][nocc] - info['eps'][nocc - 1])
                - (info_0['eps'][nocc] - info_0['eps'][nocc - 1]))
    ok &= check(np.abs(info['eps_mf'] - info_0['eps_mf']).max() < 1e-10
                and d_gap > 1e-3,
                'and Eq. (18) then moves the diagonal on an unchanged mean field',
                f'quasiparticle gap by {d_gap * HARTREE_TO_EV:.3f} eV')

    om_0_isdf = np.sort(solve_bse_isdf(frozen, mol, nocc, nroots=NROOTS,
                                       auxbasis=AUXBASIS, probe=False)[0])
    om_isdf = np.sort(solve_bse_isdf(solvated, mol, nocc, nroots=NROOTS,
                                     auxbasis=AUXBASIS, probe=False)[0])
    response_df = (om - om_0) * HARTREE_TO_EV
    response_isdf = (om_isdf - om_0_isdf) * HARTREE_TO_EV
    d = np.abs(response_df - response_isdf).max()
    return ok & check(d < RESPONSE_TOL,
                      'the response half is the same on both factorizations',
                      'DF ' + ' '.join(f'{v:+.4f}' for v in response_df)
                      + ' eV, max |d| ' + f'{d * 1e3:.2f} meV')


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    mol = gto.M(atom=ATOM, basis=BASIS, verbose=0)
    mf = factory(mol)
    nocc = mol.nelectron // 2
    print(f'\n=== water / {BASIS} @ {XC}, nocc={nocc}, {NROOTS} roots ===')
    all_ok = True
    print('\n-- 1. the solve, and its diagonal')
    all_ok &= test_bse_at_the_mean_field_is_the_dense_casida_solve(mol, mf, nocc)
    all_ok &= test_the_g0w0_diagonal_is_the_casida_gw_route(mol, mf, nocc)
    all_ok &= test_the_two_factorizations_agree_on_the_spectrum(mol, mf, nocc)
    print('\n-- 2. evGW-BSE')
    all_ok &= test_evgw_converges_and_screens_at_its_fixed_point(mol, mf, nocc)
    print('\n-- 3. a continuum, both halves')
    all_ok &= test_a_continuum_reaches_the_diagonal_and_the_kernel(mol, mf, nocc)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
