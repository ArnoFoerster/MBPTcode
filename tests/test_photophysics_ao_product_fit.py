"""The photophysics command line on the AO-product fit (--isdf-test-set
ao_products, --pivot-points).

Gates:
  * the options reach the declaration's one Numerics: the AO-product test set
    and the pivot count on the --grid-accuracy level, fitted whole (the row
    fit realizes the Duchemin-Blase set only), the factors still sliced and
    the adjoint still the grid's; without them the factors are the row fit
    as before;
  * a pivot count without the AO-product set is refused by name;
  * the chain the command line declares runs on that fit: water/cc-pVDZ
    Hartree-Fock, 250 points pivoted from the G3 parent, over two and three
    simulated ranks, its S1 force on the sliced factors with the grid adjoint
    the whole layout's with the explicit adjoint within 1e-9 Ha/Bohr (two
    adjoints, which agree to rounding; measured 1.3e-12), on every rank rank
    0's force, both chains on the 250 AO-product points.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.gwbse.excited_state import ExcitedStateChain
from tests.photophysics_cases import declaration
from tests.test_chain_sliced_factors import (H2O_DISPLACED,
                                             one_mean_field_per_geometry,
                                             own_water)
from tests.threads_helper import pyscf_one_thread  # noqa: F401  (fixture)

AO_FIT = ('--isdf-test-set', 'ao_products', '--pivot-points', '250')
#: Ha/Bohr: the grid adjoint against the explicit one on one fit.
ADJOINT_TOL = 1e-9


def test_the_options_select_the_ao_product_fit():
    n = declaration(*AO_FIT).numerics
    assert (n.isdf.level, n.isdf.test_set, n.isdf.pivot_points) == ('G3', 'ao_products', 250)
    assert n.isdf.fit is None and n.layout.sliced and n.bse.adjoint == 'grid'
    plain = declaration().numerics
    assert (plain.isdf.fit, plain.isdf.test_set, plain.isdf.pivot_points) == ('rows', None, None)
    assert n.identity() != plain.identity()


@pytest.mark.parametrize('extra', [('--pivot-points', '250'),
                                   ('--isdf-test-set', 'duchemin_blase',
                                    '--pivot-points', '250')])
def test_a_pivot_count_needs_the_ao_product_set(extra):
    with pytest.raises(SystemExit, match='--isdf-test-set ao_products'):
        declaration(*extra)


@pytest.mark.parametrize('size', [2, 3])
def test_the_declared_chain_runs_on_the_ao_product_fit(size, pyscf_one_thread):
    production = declaration(*AO_FIT).numerics
    whole = production.derive('the whole layout and the explicit adjoint',
                              layout={'sliced': None}, bse={'adjoint': 'explicit'})

    def rank(comm):
        out = {}
        scf_factory = one_mean_field_per_geometry()
        for tag, numerics in (('production', production), ('whole', whole)):
            mol = own_water()
            chain = ExcitedStateChain(mol, scf_factory, mf=scf_factory(mol),
                                      numerics=numerics)
            g, d = chain.excitation_gradient(own_water(H2O_DISPLACED))
            fz = chain.factorization
            out[tag] = (np.asarray(g), float(d['omega']), (fz.test_set, fz.M))
        return out

    res = run_simulated(rank, size)
    for r, out in enumerate(res):
        miss = float(np.abs(out['production'][0] - out['whole'][0]).max())
        print(f'[info] {size} ranks, rank {r}: grid-sliced minus explicit-whole '
              f'S1 force {miss:.2e} Ha/Bohr, omega {out["production"][1]:.10f}')
        assert miss <= ADJOINT_TOL
        assert abs(out['production'][1] - out['whole'][1]) <= 1e-10
        assert out['production'][2] == out['whole'][2] == ('ao_products', 250)
        assert np.array_equal(out['production'][0], res[0]['production'][0])
