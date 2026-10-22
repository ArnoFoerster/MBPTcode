"""The separable-RI factors on orbitals that carry near-null AO combinations.

Water in aug-cc-pVDZ plus four tightly spaced diffuse s and p shells per atom
(exponents 0.03 x 1.15^k): 89 AOs, smallest overlap eigenvalue 3e-12. The
Duchemin-Blase fit is made on AO-pair densities, so an orbital that takes a
near-null combination with a coefficient of order w^-1/2 receives the fit's
error amplified by up to 1/w. Hartree-Fock in the canonical span above 1e-8
(83 orbitals, coefficient norms to 3.9e3) leaves the worst virtual density
69% wrong, and the ISDF BSE puts the third singlet 153 meV below the
density-fitted BSE on the same orbitals; in the represented span (49 of 89
directions kept) the three lowest roots of each spin agree to 1.3 meV.

Gates:
  * `orbital_density_residual` sees it (the occupied densities are fitted to
    1e-7, the worst virtual one not at all) and `refuse_unrepresented_orbitals`
    refuses those orbitals;
  * `represented_span` keeps the overlap eigenfunctions the fit represents,
    and the mean field solved in that span (`isdf_jk.solve_in_span`) passes
    the refusal and gives the ISDF BSE roots of DF on the same orbitals to a
    few meV.
"""
import numpy as np
import pytest
from pyscf import df as pyscf_df, gto, scf

from src.Base.constants import (HARTREE_TO_EV, ISDF_DENSITY_RESIDUAL_TOL,
                                ISDF_ORBITAL_RESIDUAL_TOL)
from src.Base.isdf_jk import canonical_basis, solve_in_span
from src.Base.numerics import Numerics
from src.Base.separable_ri import (orbital_density_residual,
                                   refuse_unrepresented_orbitals,
                                   represented_span, shipped_radii_lookup)
from src.SingleReference.GW.space_time import (separable_factors,
                                                unrefused_separable_factors)
from src.SingleReference.LinearResponse.davidson import (solve_bse_df,
                                                         solve_bse_isdf)

#: The extra diffuse shells: first exponent, ratio, count.
EXTRA = (0.03, 1.15, 4)
#: ISDF - DF on the lowest roots of the canonical span, eV (measured 1-2 meV).
ROOT_TOL_EV = 5e-3
#: Canonical cut of the unprotected reference: the smallest at which its SCF
#: converges (with no cut |g| stalls at 3e-6 while |ddm| jumps between 45 and
#: 240, exact density fitting and no ISDF anywhere).
REFERENCE_CUT = 1e-8


def _basis(element):
    base = gto.basis.load('aug-cc-pvdz', element)
    a0, ratio, n = EXTRA
    for k in range(n):
        base = base + [[0, [a0 * ratio**k, 1.0]], [1, [a0 * ratio**k, 1.0]]]
    return base


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis={'O': _basis('O'), 'H': _basis('H')}, verbose=0)
    auxmol = pyscf_df.addons.make_auxmol(mol, 'aug-cc-pvdz-ri')
    radii = {el: shipped_radii_lookup(el, 'aug-cc-pvdz', 'aug-cc-pvdz-ri',
                                      {'A1': 24, 'A2': 15, 'A3': 9, 'B1': 3})[0]
             for el in ('O', 'H')}
    mf = _mean_field(mol, REFERENCE_CUT)
    mf.kernel()
    assert mf.converged
    return mol, auxmol, radii, mf


def _mean_field(mol, threshold):
    """Hartree-Fock, J and K fitted on jkfit, solved in the canonical span
    above `threshold`."""
    mf = solve_in_span(scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-jkfit'),
                       canonical_basis(mol.intor('int1e_ovlp'), threshold))
    mf.conv_tol = 1e-10
    return mf


def _factors(mf, mol, radii, build=separable_factors):
    X_mo, D, X_ao, _ = build(mf, mol, auxbasis='aug-cc-pvdz-ri', radii=radii)
    return X_mo, D, X_ao


def test_near_null_orbitals_are_seen_and_refused(water):
    mol, auxmol, radii, mf = water
    # unmeasured: the factor build itself refuses them by default
    _, D, X_ao = _factors(mf, mol, radii, unrefused_separable_factors)
    nocc = mol.nelectron // 2
    rel, _ = orbital_density_residual(mol, auxmol, mf.mo_coeff, X_ao, D)
    assert rel[:nocc].max() < 1e-6
    assert rel[nocc:].max() > 10 * ISDF_ORBITAL_RESIDUAL_TOL
    with pytest.raises(RuntimeError, match='near-null'):
        refuse_unrepresented_orbitals(mol, auxmol, mf.mo_coeff, X_ao, D)


def test_represented_span_restores_the_roots(water):
    mol, auxmol, radii, mf0 = water
    _, D, X_ao = _factors(mf0, mol, radii, unrefused_separable_factors)
    basis, record = represented_span(mol, auxmol, X_ao, D)
    assert 0 < record['n_dropped'] < mol.nao_nr() // 2
    mf = solve_in_span(scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-jkfit'),
                       basis)
    mf.conv_tol = 1e-10
    mf.kernel(dm0=mf0.make_rdm1())
    assert mf.converged
    assert mf.mo_coeff.shape[1] == record['n_kept']
    X_mo, D, X_ao = _factors(mf, mol, radii)
    rel, _ = orbital_density_residual(mol, auxmol, mf.mo_coeff, X_ao, D)
    assert rel.max() <= ISDF_ORBITAL_RESIDUAL_TOL
    refuse_unrepresented_orbitals(mol, auxmol, mf.mo_coeff, X_ao, D)
    nocc = mol.nelectron // 2
    for spin in ('singlet', 'triplet'):
        w_is, _, _, info = solve_bse_isdf(mf, mol, nocc, numerics=Numerics(bse={'nroots': 3}), qp=None,
                                          factors=(X_mo, D, X_ao), spin=spin,
                                          probe=True)
        w_df, _, _, _ = solve_bse_df(mf, mol, nocc, numerics=Numerics(bse={'nroots': 3}), qp=None,
                                     spin=spin)
        err = (np.sort(w_is) - np.sort(w_df)) * HARTREE_TO_EV
        assert np.abs(err).max() < ROOT_TOL_EV, (spin, err)
        assert info['min_eig_amb'] > 0


def test_canonical_basis_spans_the_kept_overlap_eigenvectors(water):
    mol = water[0]
    S = mol.intor('int1e_ovlp')
    X = canonical_basis(S, REFERENCE_CUT)
    assert X.shape[1] == int((np.linalg.eigvalsh(S) > REFERENCE_CUT).sum())
    assert np.allclose(X.T @ S @ X, np.eye(X.shape[1]), atol=1e-8)
    assert water[3].mo_coeff.shape[1] == X.shape[1]


def test_density_tolerance_is_tighter_than_the_orbital_one():
    assert 0 < ISDF_DENSITY_RESIDUAL_TOL <= ISDF_ORBITAL_RESIDUAL_TOL < 1
