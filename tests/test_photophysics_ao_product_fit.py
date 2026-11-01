"""The photophysics command line on the AO-product fit (--pivot-points,
--isdf-test-set).

Gates:
  * the options reach the declaration's one Numerics: with no test set named
    the --grid-accuracy level is the AO-product fit (on --pivot-points, else
    the default count), with --isdf-test-set duchemin_blase the Duchemin-Blase
    set, every fit by rows, the factors sliced and the adjoint the grid's;
  * a pivot count beside the Duchemin-Blase set is refused by name;
  * the chain the command line declares runs on that fit: water/cc-pVDZ
    Hartree-Fock, 250 points pivoted from the G3 parent, over two and three
    simulated ranks, its S1 force on the sliced row fit with the grid adjoint
    the whole layout's replicated fit with the explicit adjoint within 1e-9
    Ha/Bohr (two realizations and two adjoints, which agree to rounding), on
    every rank rank 0's force, both chains on the 250 AO-product points.
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
    assert n.isdf.fit == 'rows' and n.layout.sliced and n.bse.adjoint == 'grid'
    assert declaration('--pivot-points', '250').numerics.identity() == n.identity()
    plain = declaration().numerics
    assert (plain.isdf.fit, plain.isdf.pivot_points) == ('rows', None)
    assert plain.isdf.test_set_named() == 'ao_products'
    duchemin_blase = declaration('--isdf-test-set', 'duchemin_blase').numerics
    assert (duchemin_blase.isdf.fit, duchemin_blase.isdf.test_set) == ('rows', 'duchemin_blase')
    assert len({n.identity(), plain.identity(), duchemin_blase.identity()}) == 3


def test_a_pivot_count_beside_the_duchemin_blase_set_is_refused():
    with pytest.raises(SystemExit, match='duchemin_blase'):
        declaration('--isdf-test-set', 'duchemin_blase', '--pivot-points', '250')


@pytest.mark.parametrize('size', [2, 3])
def test_the_declared_chain_runs_on_the_ao_product_fit(size, pyscf_one_thread):
    production = declaration(*AO_FIT).numerics
    whole = production.derive('the replicated fit, the whole layout and the '
                              'explicit adjoint', isdf={'fit': 'replicated'},
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
