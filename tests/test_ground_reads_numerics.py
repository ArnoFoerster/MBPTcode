"""The dRPA ground chain, the cRPA chain and a composed surface's ground half
read the one `Numerics`.

A chain with numerical keywords of its own, and composed surfaces building
their ground half from the factorization keywords alone, would give the
ground half its constructor's tile cap, fold and quadrature whatever the
surface was handed. Every factor chain takes `numerics` and holds it as
`settings`, a
composed surface hands its ground half the same object, and the ground
quadrature is derived from the excited one -- the static screening's tau
count -- unless a `derive` gives it its own.

H2/cc-pVDZ Hartree-Fock on the G1 grid; nothing is solved.
"""
import inspect

import pytest
from pyscf import gto

from src.Base.numerics.constants import RPA_ENERGY_NFREQ
from src.Base.numerics.numerics import Numerics
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.gradients.rpa.factor_chain import FactorChain
from src.gradients.gwbse.rpa_bse_surface import RPABSESurface
from src.gradients.rpa.ground_state import RPAGroundStateChain
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'
#: Keywords the chains do not take, each a field of `Numerics`.
RETIRED = ('basis', 'auxbasis', 'counts', 'n_start', 'radii', 'sliced', 'fit',
           'fit_block', 'grid_accuracy', 'ntau', 'nfreq', 'tile_gb', 'fold')


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom=H2, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def g1():
    return Numerics(isdf={'counts': resolve_isdf_grid('G1', BASIS, ['H'])[0]})


@pytest.mark.parametrize('cls', [FactorChain, RPAGroundStateChain])
def test_the_chains_take_the_object(cls):
    params = inspect.signature(cls.__init__).parameters
    assert 'numerics' in params
    assert not set(RETIRED) & set(params), sorted(set(RETIRED) & set(params))


def test_the_ground_half_reads_what_the_surface_was_handed(h2, g1):
    """The tile cap reached the excited half only."""
    numerics = g1.derive('a tile cap', layout={'tile_gb': 1.5})
    surface = RPABSESurface(h2, rhf_df_differentiable_checked, numerics=numerics)
    assert surface.ground.settings is numerics
    assert surface.ground.tile_gb == 1.5 == surface.excited.tile_gb


def test_the_ground_quadrature_follows_the_excited_one(h2, g1):
    """O6: the tau count is the static screening's, the minimax count of the
    gap where neither names one, an explicit one where the excited half's
    is explicit, and its own only where derived."""
    plain = RPABSESurface(h2, rhf_df_differentiable_checked, numerics=g1)
    assert plain.ground.ntau == plain.excited.ntau_w
    assert len(plain.ground.grid.omega_points) == RPA_ENERGY_NFREQ
    asked = g1.derive('a finer screening', quadrature={'ntau_w': 12})
    surface = RPABSESurface(h2, rhf_df_differentiable_checked, numerics=asked)
    assert surface.ground.ntau == 12 == surface.excited.ntau_w
    own = asked.derive('the ground state on its own grid',
                       quadrature={'ntau_rpa': 10, 'nfreq_rpa': 20})
    ground = RPAGroundStateChain(h2, rhf_df_differentiable_checked, numerics=own)
    assert ground.ntau == 10 and len(ground.grid.omega_points) == 20


def test_a_ground_refreeze_carries_the_resolved_quadrature(h2, g1):
    ground = RPAGroundStateChain(h2, rhf_df_differentiable_checked, numerics=g1)
    fresh = ground.refreeze(h2)
    assert fresh.ntau == ground.ntau
    assert len(fresh.grid.omega_points) == RPA_ENERGY_NFREQ
    assert fresh.settings is g1
