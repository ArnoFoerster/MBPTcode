"""A static W handed to the ISDF BSE must be declared in the separable factors' gauge.

pyscf's cderi is L^-1-whitened and the separable fit V^-1/2-whitened: on the
same auxiliary basis the DF route's W has the factors' naux, so the shape
check alone let it through. It is refused; a W built on the factors and
declared separable gives the roots the solve builds for itself.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import SEPARABLE_AUX_GAUGE
from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.BSE.solve import isdf_bse_factors, solve_bse_isdf
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from tests.isdf_test_grid import g1
from tests.pins import assert_same

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, counts=g1(mol))
    eps = get_orbital_energies(mf, representation='spatial')
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    W_df = LinearResponseSolver(eps, coeff_df=coeff,
                                spin_mode='restricted').static_screening_aux(nocc)
    return dict(mol=mol, mf=mf, nocc=nocc, factors=factors, eps=eps, W_df=W_df)


def test_the_df_route_w_passes_the_shape_check(water):
    naux = water['factors'][1].shape[1]
    assert np.shape(water['W_df']) == (naux, naux)


@pytest.mark.parametrize('gauge', [None, 'cderi'])
def test_a_df_route_w_is_refused(water, gauge):
    with pytest.raises(ValueError, match='W_aux_gauge'):
        solve_bse_isdf(water['mf'], water['mol'], water['nocc'], numerics=Numerics(bse={'nroots': 1}),
                       qp=water['eps'], factors=water['factors'],
                       W_aux=water['W_df'], W_aux_gauge=gauge, probe=False,
                       progress=False)


def test_a_separable_w_declared_so_gives_the_built_roots(water):
    w = water
    _, _, W_sep = isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                                   factors=w['factors'])
    built = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], numerics=Numerics(bse={'nroots': 2}), qp=w['eps'],
                           factors=w['factors'], probe=False, progress=False)[0]
    handed = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], numerics=Numerics(bse={'nroots': 2}),
                            qp=w['eps'], factors=w['factors'], W_aux=W_sep,
                            W_aux_gauge=SEPARABLE_AUX_GAUGE, probe=False,
                            progress=False)[0]
    # two Davidson solves: the same roots to a few ulp, not bitwise
    assert_same(handed, built, quantity='root', label='separable W')


def test_a_gauge_without_a_w_is_refused(water):
    with pytest.raises(TypeError, match='W_aux_gauge'):
        solve_bse_isdf(water['mf'], water['mol'], water['nocc'], numerics=Numerics(bse={'nroots': 1}),
                       qp=water['eps'], factors=water['factors'],
                       W_aux_gauge=SEPARABLE_AUX_GAUGE, probe=False,
                       progress=False)
