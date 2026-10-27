"""Diagnostic checks for solve_qp_energy_scgw.

Test 1 (Sigma = 0): with the correlation self-energy switched off, the scGW
cycle reduces to self-consistent HF, so its Galitskii-Migdal energy must equal
the HF total energy, whether the cycle starts from HF or from DFT orbitals.

Test 2 (start independence): the full scGW cycle is self-consistent, so
starting it from HF or from DFT orbitals must give the same Galitskii-Migdal
energy (on a shared IR grid).

Sigma is switched off by monkeypatching: scgw.calc_new_sigma_ir is replaced
by a wrapper that returns a zero Sigma, and put back afterwards. Nothing in
scgw.py itself changes.

    python run_diag_tests.py
"""
import os
import sys
from contextlib import contextmanager

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyscf import gto, scf

import src.SingleReference.GW.scgw as scgw
from src.SingleReference.GW.space_time import separable_factors

# The cycle stops at max |delta gamma| = 1e-6. On H2/cc-pVDZ the E_GM of an
# HF and a PBE start then agree to 2e-8 Ha (their one-body and correlation
# parts to ~1.5e-7 each), so this leaves margin for larger systems.
E_GM_TOL = 1e-6


@contextmanager
def sigma_switched_off():
    """Inside this block, scgw.calc_new_sigma_ir returns Sigma = 0.

    solve_qp_energy_scgw looks calc_new_sigma_ir up in the scgw module at
    every call, so replacing the module attribute is enough for the whole
    cycle to see the new function. The original is restored on exit, also
    when the run raises.
    """
    original = scgw.calc_new_sigma_ir

    def zero_sigma(G, grid_f, grid_b, X_mo, D):
        return np.zeros_like(original(G, grid_f, grid_b, X_mo, D))

    scgw.calc_new_sigma_ir = zero_sigma
    try:
        yield
    finally:
        scgw.calc_new_sigma_ir = original


def check_sigma_zero_gives_hf(mol, mf_start, e_hf, name):
    """With Sigma = 0 the cycle is self-consistent HF, so E_GM must be e_hf.

    E_GM then reduces to E_nuc + Tr[gamma (h + F)], the HF total energy of
    the converged density. Both an HF and a DFT start are compared to the
    HF energy. The peaks of A(w) are not used: not every level shows up as
    a separate peak (e.g. on N2), while the energy checks the converged
    state as a whole.
    """
    nocc = mol.nelectron // 2
    factors = separable_factors(mf_start, mol, auxbasis='cc-pvdz-ri', grid_accuracy='G2')
    try:
        with sigma_switched_off():
            _, _, _, E_gm, _ = scgw.solve_qp_energy_scgw(mf_start, mol, nocc,
                                                      factors=factors)
    except RuntimeError as err:
        print(f'\n=== {name} ===\n{err}\n{name}: FAIL')
        return False

    diff = E_gm - e_hf
    print(f'\n=== {name} ===')
    print(f'E_HF             = {e_hf:.10f} Ha')
    print(f'E_GM (Sigma = 0) = {E_gm:.10f} Ha')
    ok = abs(diff) < E_GM_TOL
    print(f'|diff| = {abs(diff):.2e} Ha (tol {E_GM_TOL:.0e})')
    print(f'{name}: {"PASS" if ok else "FAIL"}')
    return ok


def grid_params(mf, nocc):
    """(beta, omega_max) as solve_qp_energy_scgw derives them from mf."""
    eps = np.sort(mf.mo_energy)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    dG = np.abs(eps - mu)
    e_max = eps[nocc:].max() - eps[:nocc].min()
    return 25.0 / dG.min(), 3.0 * (dG.max() + e_max)


def shared_grid_params(mfs, nocc):
    """One (beta, omega_max) that is adequate for every start in mfs.

    The smallest gap needs the largest beta and the widest spectrum the
    largest omega_max, so the maximum of each is taken.
    """
    betas, omega_maxes = zip(*(grid_params(mf, nocc) for mf in mfs))
    return max(betas), max(omega_maxes)


def converged_gm_energy(mol, mf_start, **grid):
    """Full scGW (real Sigma) from the orbitals of mf_start.

    Returns the Galitskii-Migdal energy of the converged G and Sigma. It is a
    total energy, so unlike F it does not depend on the MO basis of the start
    and can be compared directly. grid: optional beta / omega_max, passed on
    to override the IR grid.
    """
    nocc = mol.nelectron // 2
    factors = separable_factors(mf_start, mol, auxbasis='cc-pvdz-ri', grid_accuracy='G2')
    _, _, _, E_gm, _ = scgw.solve_qp_energy_scgw(mf_start, mol, nocc,
                                              factors=factors, **grid)
    # _, _, _, E_gm = scgw.solve_qp_energy_scgw(mf_start, mol, nocc,
    #                                           factors=factors)
    return E_gm


def check_start_independence(mol, mf_hf, mf_dft, name):
    """scGW is self-consistent, so its result must not depend on the start.

    Compares the Galitskii-Migdal energy from an HF and a DFT start.
    The peaks of A(w) are not used: the Pade continuation amplifies tiny
    differences in Sigma far from mu, so they can differ while the converged
    state is the same. Left to itself, each run builds its IR grid from its
    own start's orbital energies, so both get the same, shared beta and
    omega_max: a remaining difference then comes from the cycle, not from
    the grids.
    """
    beta, omega_max = shared_grid_params([mf_hf, mf_dft], mol.nelectron // 2)
    print(f'\nShared IR grid: beta = {beta:.4g}, omega_max = {omega_max:.4g}')
    try:
        E_hf = converged_gm_energy(mol, mf_hf, beta=beta, omega_max=omega_max)
        E_dft = converged_gm_energy(mol, mf_dft, beta=beta, omega_max=omega_max)
    except RuntimeError as err:
        print(f'\n=== {name} ===\n{err}\n{name}: FAIL')
        return False

    diff = E_dft - E_hf
    print(f'\n=== {name} ===')
    print(f'E_GM (HF start)  = {E_hf:.10f} Ha')
    print(f'E_GM (DFT start) = {E_dft:.10f} Ha')
    ok = abs(diff) < E_GM_TOL
    print(f'|diff| = {abs(diff):.2e} Ha (tol {E_GM_TOL:.0e})')
    print(f'{name}: {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    # mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
    mol = gto.M(atom='N 0 0 0; N 0 0 1.0977', basis='cc-pvdz', verbose=0)
    # mol = gto.M(atom='C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276',
    #              basis = 'cc-pvdz', verbose = 0)

    mf_hf = scf.RHF(mol)
    mf_hf.conv_tol = 1e-10
    mf_hf.run()

    mf_dft = scf.RKS(mol)
    mf_dft.xc = 'pbe'
    mf_dft.conv_tol = 1e-10
    mf_dft.run()

    results = {
        'Sigma=0, HF start': check_sigma_zero_gives_hf(
            mol, mf_hf, mf_hf.e_tot, 'Sigma=0, HF start'),
        'Sigma=0, DFT start': check_sigma_zero_gives_hf(
            mol, mf_dft, mf_hf.e_tot, 'Sigma=0, DFT start'),
        'scGW, HF vs DFT start': check_start_independence(
            mol, mf_hf, mf_dft, 'scGW, HF vs DFT start'),
    }

    print('\n=== Summary ===')
    for name, ok in results.items():
        print(f'{name}: {"PASS" if ok else "FAIL"}')
    sys.exit(0 if all(results.values()) else 1)


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)


@pytest.fixture(scope='module')
def mf_hf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    return mf.run()


@pytest.fixture(scope='module')
def mf_dft(mol):
    mf = scf.RKS(mol)
    mf.xc = 'pbe'
    mf.conv_tol = 1e-10
    return mf.run()


def test_sigma_zero_hf_start(mol, mf_hf):
    assert check_sigma_zero_gives_hf(mol, mf_hf, mf_hf.e_tot, 'Sigma=0, HF start')


def test_sigma_zero_dft_start(mol, mf_hf, mf_dft):
    assert check_sigma_zero_gives_hf(mol, mf_dft, mf_hf.e_tot, 'Sigma=0, DFT start')


def test_start_independence(mol, mf_hf, mf_dft):
    assert check_start_independence(mol, mf_hf, mf_dft, 'scGW, HF vs DFT start')
