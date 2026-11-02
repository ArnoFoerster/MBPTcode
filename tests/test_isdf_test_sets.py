"""The ISDF fit of each test set, chosen by name (`src.Base.factorization.test_sets`).

On C1 water/cc-pVDZ at 150 points pivoted from the G1 parent: `replicated_fit`
and `row_fit` give, bit for bit, the fit of the test set they are named
(`ao_product_fit` / `ao_product_fit_rows` for 'ao_products', the collocation
with `fit_M_streaming` / `fit_rows` for 'duchemin_blase'), a name outside
ISDF_TEST_SETS is refused, and the GW row factors read the Duchemin-Blase fit
when no test set is named.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.factorization import ao_product_fit as apf
from src.Base.factorization.separable_ri import ao_values, fit_M_streaming, fit_rows
from src.Base.factorization.test_sets import is_ao_products, replicated_fit, row_fit
from src.Base.numerics.constants import ISDF_AO_PRODUCTS, ISDF_DUCHEMIN_BLASE
from src.SingleReference.GW.space_time import _row_factors
from tests.test_ao_product_fit import pivoted_water

TILE = 40


@pytest.fixture(scope='module')
def water():
    mol, auxmol, pts, _, frames = pivoted_water()
    return mol, auxmol, apf.placed(mol, pts, frames)


def test_an_unknown_test_set_is_refused():
    with pytest.raises(ValueError, match='test_set'):
        is_ao_products('legacy')


def test_each_name_runs_its_own_fit(water):
    mol, auxmol, coords = water
    X, M = replicated_fit(ISDF_AO_PRODUCTS, mol, auxmol, coords)
    want = apf.ao_product_fit(mol, auxmol, coords)
    assert np.array_equal(X, want[0]) and np.array_equal(M, want[1])
    X, M = replicated_fit(ISDF_DUCHEMIN_BLASE, mol, auxmol, coords)
    assert np.array_equal(X, ao_values(mol, coords))
    assert np.array_equal(M, fit_M_streaming(mol, auxmol, coords))
    for name, direct in ((ISDF_AO_PRODUCTS, apf.ao_product_fit_rows),
                         (ISDF_DUCHEMIN_BLASE, fit_rows)):
        got = row_fit(name, mol, auxmol, coords, block=TILE).mt
        want = direct(mol, auxmol, coords, block=TILE).mt
        assert sorted(got) == sorted(want)
        assert all(np.array_equal(got[t], want[t]) for t in got), name


def test_the_row_factors_default_to_the_duchemin_blase_fit(water):
    """`_row_factors` with no test set named: the rows of the Duchemin-Blase
    row fit, as before the test sets were chosen by name."""
    mol, auxmol, coords = water
    mf = scf.RHF(mol)
    mf.mo_coeff = np.eye(mol.nao_nr())
    X_mo, D, X_ao, _ = _row_factors(mf, mol, auxmol, coords, None, None, None,
                                    TILE, 0.0)
    fit = fit_rows(mol, auxmol, coords, block=TILE)
    assert np.array_equal(X_mo, fit.mo_rows(mf.mo_coeff))
    assert np.array_equal(D, fit.metric_root_rows(auxmol, None))
    assert np.array_equal(X_ao, fit.ao_rows())
