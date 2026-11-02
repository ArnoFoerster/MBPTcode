"""The ISDF fit of each test set, chosen in one place.

An ISDF fit is trained on a test set: 'duchemin_blase' (ISDF_DUCHEMIN_BLASE),
the screened AO pairs of Duchemin and Blase's scheme (`separable_ri`), or
'ao_products' (ISDF_AO_PRODUCTS), every AO product in the Coulomb metric on
pivoted points (`ao_product_fit`). Each has a replicated fit and a fit in
grid-row tiles. `replicated_fit` and `row_fit` take the test set by name and
run that fit, so the routes that build factors (the ISDF-K mean field, its
distributed twin, the post-SCF factors, the tiled exchange skeleton) dispatch
in one place.
"""
import time

from src.Base.factorization.ao_product_fit import (ao_product_fit,
                                                   ao_product_fit_rows)
from src.Base.factorization.separable_ri import ao_values, fit_M_streaming
from src.Base.numerics.constants import ISDF_AO_PRODUCTS, ISDF_TEST_SETS


def is_ao_products(test_set):
    """Whether `test_set` (a name of ISDF_TEST_SETS) is the AO-product test
    set; any other name is refused."""
    if test_set not in ISDF_TEST_SETS:
        raise ValueError(f'test_set={test_set!r}: one of {ISDF_TEST_SETS}')
    return test_set == ISDF_AO_PRODUCTS


def replicated_fit(test_set, mol, auxmol, coords, numerics=None, comm=None,
                   timings=None, progress=False, layout=None):
    """(X, M): the collocation (nk, nao) and the fit's auxiliary coefficients
    (naux, nk) of `test_set` at `coords`, whole on every rank of `comm`
    (None: the current one). On the Duchemin-Blase test set `progress`,
    `layout` (a frozen pair layout, or a zero-argument callable giving it,
    read on that test set only) and `timings` are `fit_M_streaming`'s, and
    timings['fit_assembly'] the collocation's seconds; the AO-product fit
    screens no pair and reads none of them."""
    if is_ao_products(test_set):
        X, M, _ = ao_product_fit(mol, auxmol, coords, numerics, comm)
        return X, M
    M = fit_M_streaming(mol, auxmol, coords, numerics=numerics,
                        progress=progress, comm=comm, timings=timings,
                        layout=layout() if callable(layout) else layout)
    _t = time.time()
    X = ao_values(mol, coords)
    if timings is not None:
        timings['fit_assembly'] = time.time() - _t
    return X, M


def row_fit(test_set, mol, auxmol, coords, numerics=None, comm=None,
            block=None, tiles=None, timings=None, progress=False,
            layout=None):
    """The `separable_ri.RowFit` of `test_set` at `coords` in grid-row tiles
    of `block` points over the ranks of `comm`: `ao_product_fit_rows` or
    `fit_rows`. tiles: a `FitTiles` another stage holds, read where it is
    this fit. `progress` and `layout` (as in `replicated_fit`) are the
    Duchemin-Blase fit's."""
    if is_ao_products(test_set):
        return ao_product_fit_rows(mol, auxmol, coords, numerics=numerics,
                                   comm=comm, block=block, tiles=tiles,
                                   timings=timings)
    return fit_M_streaming(mol, auxmol, coords, numerics=numerics,
                           progress=progress, comm=comm, timings=timings,
                           fit='rows', block=block, tiles=tiles,
                           layout=layout() if callable(layout) else layout)
