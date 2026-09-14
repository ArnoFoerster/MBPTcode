"""The matrix-free Davidson block action at kappa = 0, against the dense triplet.

Spin enters the Casida blocks through one scalar: A and B carry kappa (ia|jb)
with kappa = 2 for a singlet and 0 for a triplet, while the screened term W is
spin-independent. A triplet is therefore the same action with the bare-exchange
contraction absent -- not scaled to zero, absent, since it is two of the
contractions in each step.

The reference is `LinearResponseSolver.build_casida_matrices(triplet=True)`,
which is the dense builder and owes nothing to the code under test.

Run: python tests/test_davidson_triplet.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.davidson import (isdf_bse_factors,
                                                         isdf_block_action,
                                                         solve_casida_davidson)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

NROOTS = 3


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build():
    # cc-pVDZ, not sto-3g: the separable-RI factorization needs a <basis>-ri
    # auxiliary set and sto-3g has none.
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis='cc-pvdz',
                verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-13
    mf.kernel()
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted',
                              eta=1e-3)
    return mol, mf, nocc, lr


def test_matrix_free_matches_the_dense_blocks(mol, mf, nocc, lr):
    ok = True
    for mode in ('RPA', 'TDHF'):
        for spin in ('singlet', 'triplet'):
            A, B = lr.build_casida_matrices(nocc, lBSE=(mode != 'RPA'),
                                            triplet=(spin == 'triplet'))
            dense = np.sort(CasidaSolver(A, B).solve()[0])[:NROOTS]
            om, _, _ = solve_casida_davidson(lr, nocc, nroots=NROOTS,
                                             polarizability=mode,
                                             conv_tol=1e-10, spin=spin)
            d = np.abs(np.sort(om)[:NROOTS] - dense).max()
            ok &= check(d < 1e-8, f'{mode} {spin}: matrix-free == dense blocks',
                        f'max |dw| {d:.1e} Ha')
    return ok


def test_the_triplet_is_a_different_number_and_lies_below(mol, mf, nocc, lr):
    """What a kappa silently left at 2 would fail.

    Dropping the 2(ia|jb) bare exchange can only lower a root, so the triplet
    sits below its singlet -- the sign is as much a check as the magnitude.
    """
    roots = {}
    for spin in ('singlet', 'triplet'):
        om, _, _ = solve_casida_davidson(lr, nocc, nroots=NROOTS,
                                         polarizability='RPA', conv_tol=1e-10,
                                         spin=spin)
        roots[spin] = np.sort(om)[:NROOTS]
    return check((roots['triplet'] < roots['singlet']).all()
                 and (roots['singlet'] - roots['triplet']).min() > 1e-3,
                 'every triplet lies below its singlet',
                 'splittings ' + ' '.join(
                     f'{d:.4f}' for d in roots['singlet'] - roots['triplet'])
                 + ' Ha')


def test_isdf_action_drops_the_bare_term_from_both_blocks(mol, mf, nocc, lr):
    """kappa = 0 must remove the Hartree term from A AND B, not just from A."""
    nvirt = len(mf.mo_energy) - nocc
    X_mo, D, W_aux = isdf_bse_factors(mf, mol, nocc)[:3]
    z = np.random.default_rng(1).standard_normal((2, nocc, nvirt))

    act_s, _ = isdf_block_action(lr, nocc, True, W_aux, (X_mo, D),
                                 spin='singlet')
    act_t, _ = isdf_block_action(lr, nocc, True, W_aux, (X_mo, D),
                                 spin='triplet')
    Az_s, Bz_s = act_s(z)
    Az_t, Bz_t = act_t(z)
    ok = check(np.abs(Az_s - Az_t).max() > 1e-6
               and np.abs(Bz_s - Bz_t).max() > 1e-6,
               'kappa = 0 changes both blocks',
               f'|dA| {np.abs(Az_s - Az_t).max():.2e}, '
               f'|dB| {np.abs(Bz_s - Bz_t).max():.2e}')
    # The difference is EXACTLY the singlet's Hartree term, which enters A and
    # B identically -- so the two differences must coincide.
    d = np.abs((Az_s - Az_t) - (Bz_s - Bz_t)).max()
    return ok & check(d < 1e-10,
                      'and by the same amount, since Hartree enters A and B alike',
                      f'{d:.1e}')


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    mol, mf, nocc, lr = build()
    print(f'\n=== water / cc-pVDZ, nocc={nocc} ===')
    all_ok = True
    all_ok &= test_matrix_free_matches_the_dense_blocks(mol, mf, nocc, lr)
    all_ok &= test_the_triplet_is_a_different_number_and_lies_below(mol, mf, nocc, lr)
    all_ok &= test_isdf_action_drops_the_bare_term_from_both_blocks(mol, mf, nocc, lr)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
