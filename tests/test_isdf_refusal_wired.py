"""Every builder of separable-RI factors for a mean field refuses them on an
orbital density they miss.

The fit misses the densities of near-null AO combinations, the virtual
orbitals of an augmented basis carry them, and a BSE on those orbitals binds
a POSITIVE-DEFINITE spurious root that no other check refuses (CBP gas
aug-cc-pVDZ: S1 0.94 eV against DF 3.95 eV). Gated here: `separable_factors`
(every non-chain ISDF route: `solve_bse`, `solve_bse_isdf`, the space-time
GW) and the chain's `FactorChain.factors_at` refuse the factors, naming the
orbitals with their residuals, at the bar ISDF_ORBITAL_RESIDUAL_TOL, which no
signature takes as a keyword; factors held as grid rows over ranks give one
verdict on every rank, the residuals of the whole factors.

The unrepresented reference is the water of `test_isdf_near_null_orbitals`:
aug-cc-pVDZ plus four diffuse s and p shells, Hartree-Fock in the canonical
span above 1e-8, whose worst virtual density the (24, 15, 9, 3) aug-cc-pVDZ
factors miss by 69 %.
"""

import numpy as np
import pytest
from pyscf import df as pyscf_df, gto, scf

from src.Base.numerics.constants import ISDF_ORBITAL_RESIDUAL_TOL
from src.Base.factorization.isdf_jk import canonical_basis, solve_in_span
from src.Base.numerics.numerics import Numerics
from src.Base.factorization.separable_ri import (UnrepresentedOrbitals,
                                                 aux_metric_sqrt, orbital_density_residual,
                                                 refuse_unrepresented_factors,
                                                 shipped_radii_lookup)
from src.Base.factorization.sliced_factors import SlicedFactors
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.GW.space_time import (separable_factors,
                                                unrefused_separable_factors)
from src.SingleReference.BSE.solve import solve_bse
from src.gradients.gwbse.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1_numerics
from tests.pins import assert_same

#: The extra diffuse shells (first exponent, ratio, count) and the canonical
#: cut of the unrepresented reference, as in test_isdf_near_null_orbitals.
EXTRA = (0.03, 1.15, 4)
REFERENCE_CUT = 1e-8
COUNTS = {'A1': 24, 'A2': 15, 'A3': 9, 'B1': 3}
H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def _basis(element):
    base = gto.basis.load('aug-cc-pvdz', element)
    a0, ratio, n = EXTRA
    for k in range(n):
        base = base + [[0, [a0 * ratio**k, 1.0]], [1, [a0 * ratio**k, 1.0]]]
    return base


def _span_rhf(mol):
    mf = solve_in_span(scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-jkfit'),
                       canonical_basis(mol.intor('int1e_ovlp'), REFERENCE_CUT))
    mf.conv_tol = 1e-10
    return mf


@pytest.fixture(scope='module')
def diffuse_water():
    """(mol, auxmol, radii, mf): the reference whose virtuals the factors miss."""
    mol = gto.M(atom=H2O, basis={'O': _basis('O'), 'H': _basis('H')},
                verbose=0)
    auxmol = pyscf_df.addons.make_auxmol(mol, 'aug-cc-pvdz-ri')
    radii = {el: shipped_radii_lookup(el, 'aug-cc-pvdz', 'aug-cc-pvdz-ri',
                                      COUNTS)[0] for el in ('O', 'H')}
    mf = _span_rhf(mol)
    mf.kernel()
    assert mf.converged
    return mol, auxmol, radii, mf


@pytest.fixture(scope='module')
def water():
    """Water/cc-pVDZ Hartree-Fock: every orbital density represented."""
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    assert mf.converged
    return mol, mf


def scf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


def test_separable_factors_refuse_unrepresented_orbitals(diffuse_water):
    """The front end every non-chain ISDF route builds its factors with
    refuses them by default, naming the refused orbitals (all virtual here)
    with their residuals."""
    mol, auxmol, radii, mf = diffuse_water
    nocc = mol.nelectron // 2
    with pytest.raises(UnrepresentedOrbitals) as refused:
        separable_factors(mf, mol, auxbasis='aug-cc-pvdz-ri', radii=radii)
    exc = refused.value
    assert exc.orbitals and min(exc.orbitals) >= nocc
    assert len(exc.residuals) == mf.mo_coeff.shape[1]
    worst = int(np.argmax(exc.residuals))
    assert exc.orbitals[0] == worst
    assert exc.residuals[worst] > 10 * ISDF_ORBITAL_RESIDUAL_TOL
    assert f'{worst} virtual' in str(exc) and 'Refused orbitals' in str(exc)
    # the residuals are the measurement's own
    X_mo, D, X_ao, _ = unrefused_separable_factors(
        mf, mol, auxbasis='aug-cc-pvdz-ri', radii=radii)
    rel, _ = orbital_density_residual(mol, auxmol, mf.mo_coeff, X_ao, D)
    assert np.allclose(rel, exc.residuals, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize('solver', ['dense', 'davidson'])
def test_solve_bse_refuses_before_any_gw(diffuse_water, solver):
    """`solve_bse` on the ISDF integrals refuses at the factors, on both
    solvers, before a self-energy is evaluated."""
    mol, _, radii, mf = diffuse_water
    with pytest.raises(UnrepresentedOrbitals):
        solve_bse(mf, mol, numerics=Numerics(bse={'nroots': 2}), solver=solver, integrals='isdf',
                  auxbasis='aug-cc-pvdz-ri', radii=radii)


def test_the_chain_refuses_at_its_factors(diffuse_water):
    """`FactorChain.factors_at`, which every chain's targets start from,
    refuses the mean field's orbitals."""
    mol, _, radii, mf = diffuse_water
    numerics = Numerics(isdf={'auxbasis': 'aug-cc-pvdz-ri', 'radii': radii},
                        bse={'solver': 'dense'})
    chain = ExcitedStateChain(mol, _span_rhf, mf=mf,
                              numerics=g1_numerics(mol, numerics))
    with pytest.raises(UnrepresentedOrbitals):
        chain.factors_at(mol, mf)


def test_the_chain_measures_a_represented_reference_once(water):
    """On water/cc-pVDZ the chain's factors pass, the residuals are kept per
    mean field, and nothing is measured twice."""
    mol, mf = water
    chain = ExcitedStateChain(mol, scf_factory, mf=mf,
                              numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    chain.factors_at(mol, mf)
    rel = chain.orbital_residuals_of(mol, mf, None, None, None)
    assert rel is not None and rel.max() < 1e-3
    assert chain.orbital_residuals_of(mol, mf, None, None, None) is rel


def _sliced_verdict(comm, mol, auxmol, mf, whole, root):
    rows = SlicedFactors.from_whole(whole, comm)
    try:
        refuse_unrepresented_factors(mol, auxmol, mf.mo_coeff, rows, None,
                                     root=root)
    except UnrepresentedOrbitals as exc:
        return exc.orbitals, exc.residuals
    return None


def test_grid_rows_over_ranks_give_one_verdict(diffuse_water):
    """Factors held as each rank's grid rows are summed over the ranks: two
    ranks refuse the same orbitals with the whole factors' residuals."""
    mol, auxmol, radii, mf = diffuse_water
    whole = unrefused_separable_factors(mf, mol, auxbasis='aug-cc-pvdz-ri',
                                        radii=radii, comm=None)
    rel, _ = orbital_density_residual(mol, auxmol, mf.mo_coeff, whole[2],
                                      whole[1])
    root = aux_metric_sqrt(auxmol)
    out = run_simulated(_sliced_verdict, 2, mol, auxmol, mf, whole, root)
    assert out[0] is not None and out[1] is not None
    assert out[0][0] == out[1][0]
    assert_same(out[0][1], out[1][1], unit='1', quantity='root',
                label='residuals on the two ranks')
    assert np.allclose(out[0][1], rel, rtol=1e-8, atol=1e-14)
