"""Diagnostic checks for solve_qp_energy_scgw.

Test 1 (Sigma = 0): with the correlation self-energy switched off, the scGW
cycle reduces to self-consistent HF, so the peaks of A(w) must sit at the HF
orbital energies, whether the cycle starts from HF or from DFT orbitals.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyscf import gto, scf
from pyscf.data.elements import chemcore

import src.SingleReference.GW.scgw as scgw
from src.SingleReference.GW.space_time import separable_factors

# Peaks are found on a grid of step eta/10 = 1e-3 Ha and refined with a
# parabola, so they match the HF levels to well below this.
QPE_TOL = 1e-3

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
        G_tau, chi_omega, W_omega, sigma = original(G, grid_f, grid_b, X_mo, D)
        return G_tau, chi_omega, W_omega, np.zeros_like(sigma)

    scgw.calc_new_sigma_ir = zero_sigma
    try:
        yield
    finally:
        scgw.calc_new_sigma_ir = original


def expected_hf_peaks(mol, eps_hf):
    """HF levels that should show up as peaks of A(w).

    The real-axis window starts at the first valence level (chemcore), and
    degenerate levels give a single peak, so those are dropped/merged.
    """
    levels = np.sort(eps_hf)[chemcore(mol):]
    unique = [levels[0]]
    for e in levels[1:]:
        if e - unique[-1] > 1e-5:
            unique.append(e)
    return np.array(unique)


def compare_peaks(name, qpes, expected):
    """Check that qpes has one peak within QPE_TOL of each expected level."""
    print(f'\n=== {name} ===')
    print(f'{"reference":>14} {"scGW peak":>14} {"diff":>12}')
    ok = len(qpes) == len(expected)
    if not ok:
        print(f'Number of peaks differs: {len(qpes)} found, {len(expected)} expected')
    for e in expected:
        nearest = qpes[np.argmin(np.abs(qpes - e))] if len(qpes) else np.nan
        diff = nearest - e
        ok &= abs(diff) < QPE_TOL
        print(f'{e:14.6f} {nearest:14.6f} {diff:12.2e}')
    print(f'{name}: {"PASS" if ok else "FAIL"}')
    return ok


def test_sigma_zero_gives_hf(mol, mf_start, eps_hf, name):
    nocc = mol.nelectron // 2
    factors = separable_factors(mf_start, mol, auxbasis='cc-pvdz-ri')
    with sigma_switched_off():
        qpes, A, w_real, _ = scgw.solve_qp_energy_scgw(mf_start, mol, nocc,
                                                       factors=factors)
    return compare_peaks(name, qpes, expected_hf_peaks(mol, eps_hf))


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
    factors = separable_factors(mf_start, mol, auxbasis='cc-pvdz-ri')
    _, _, _, E_gm = scgw.solve_qp_energy_scgw(mf_start, mol, nocc,
                                              factors=factors, **grid)
    return E_gm


def test_start_independence(mol, mf_hf, mf_dft, name):
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
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)

    mf_hf = scf.RHF(mol)
    mf_hf.conv_tol = 1e-10
    mf_hf.run()

    mf_dft = scf.RKS(mol)
    mf_dft.xc = 'pbe'
    mf_dft.conv_tol = 1e-10
    mf_dft.run()

    results = {
        'Sigma=0, HF start': test_sigma_zero_gives_hf(
            mol, mf_hf, mf_hf.mo_energy, 'Sigma=0, HF start'),
        'Sigma=0, DFT start': test_sigma_zero_gives_hf(
            mol, mf_dft, mf_hf.mo_energy, 'Sigma=0, DFT start'),
        'scGW, HF vs DFT start': test_start_independence(
            mol, mf_hf, mf_dft, 'scGW, HF vs DFT start'),
    }

    print('\n=== Summary ===')
    for name, ok in results.items():
        print(f'{name}: {"PASS" if ok else "FAIL"}')
    sys.exit(0 if all(results.values()) else 1)
