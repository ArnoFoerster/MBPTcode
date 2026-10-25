"""Every factor chain refreezes through ONE signature and ONE rebuild of its
factorization, and the composed surfaces through one body.

A refreeze carries every choice of the chain to the new geometry and rebuilds
only what depends on it. Every chain rebuilds its factorization by
`FrozenFactorization.rebuilt_at`, not from a hand-copied keyword list, and
`RPAQPSurface` and `RPABSESurface` share the one body of the composed
refreeze. Two paths given different keywords diverge silently: a keyword list
that omits `orbital_tol` (a bar), or a setting added to the factorization,
would be dropped the same way.

H2/cc-pVDZ Hartree-Fock on the G1 grid: the smallest system every chain
accepts. Nothing here compares a number; what is checked is which path ran.
"""
import inspect

import pytest
from pyscf import gto

from src.Base.numerics.numerics import Numerics, numerics_of
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.rpa.factor_chain import FactorChain, FrozenFactorization
from src.gradients.gwbse.rpa_bse_surface import RPABSESurface, RPAQPSurface
from src.gradients.rpa.ground_state import RPAGroundStateChain
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'
#: Bohr the second hydrogen moves along the bond.
STEP = 0.05
#: The chain signature every factor chain's refreeze has.
SIGNATURE = ('mol', 'factorization', 'mf', 'walked', 'qp_floor')


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom=H2, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def moved(h2):
    m = h2.copy()
    c = h2.atom_coords().copy()
    c[1, 2] += STEP
    m.set_geom_(c, unit='Bohr')
    m.build(False, False)
    return m


@pytest.fixture(scope='module')
def counts():
    return resolve_isdf_grid('G1', BASIS, ['H'])[0]


def ground(mol, counts, **kw):
    return RPAGroundStateChain(mol, rhf_df_differentiable_checked,
                               numerics=numerics_of(dict(kw, counts=counts)))


def excited(mol, counts):
    return ExcitedStateChain(mol, rhf_df_differentiable_checked,
                             numerics=Numerics(isdf={'counts': counts}))


BUILDERS = {'ground': ground}


@pytest.mark.parametrize('cls', [FactorChain, RPAGroundStateChain,
                                 ExcitedStateChain],
                         ids=lambda c: c.__name__)
def test_every_factor_chain_refreezes_through_one_signature(cls):
    params = tuple(inspect.signature(cls.refreeze).parameters)[1:]
    assert params == SIGNATURE, f'{cls.__name__}.refreeze{params}'


def test_the_composed_surfaces_share_one_refreeze():
    assert RPABSESurface.refreeze is RPAQPSurface.refreeze


@pytest.mark.parametrize('which', sorted(BUILDERS) + ['excited'])
def test_a_refreeze_rebuilds_the_factorization_by_rebuilt_at(
        h2, moved, counts, which, monkeypatch):
    """One rebuild: a factorization re-derived anywhere else drops whatever
    `rebuilt_at` carries that the other path does not name."""
    chain = (excited(h2, counts) if which == 'excited'
             else BUILDERS[which](h2, counts))
    rebuilt = []
    original = FrozenFactorization.rebuilt_at

    def spy(self, mol):
        out = original(self, mol)
        rebuilt.append(out)
        return out
    monkeypatch.setattr(FrozenFactorization, 'rebuilt_at', spy)
    fresh = chain.refreeze(moved)
    assert len(rebuilt) == 1 and fresh.factorization is rebuilt[0]


@pytest.mark.parametrize('which', sorted(BUILDERS))
def test_a_chain_without_a_quasiparticle_set_refuses_a_floor(h2, moved,
                                                             counts, which):
    chain = BUILDERS[which](h2, counts)
    with pytest.raises(ValueError, match='qp_floor'):
        chain.refreeze(moved, qp_floor=(0,))


def test_a_shared_factorizations_recipe_is_not_refused(h2, moved, counts):
    """A composed surface whose factorization came from a grid level holds
    that level's radii recipe (n_start ISDF_GRID_N_START, not 1); its halves
    refreeze on the rebuilt factorization without restating the recipe, and
    a chain with no opinion on it accepts the factorization's."""
    surface = RPABSESurface(h2, rhf_df_differentiable_checked, numerics=Numerics(isdf={'counts': counts, 'n_start': 8}))
    assert surface.ground.factorization.n_start == 8
    fresh = surface.refreeze(moved)
    assert fresh.ground.factorization is fresh.excited.factorization
    assert fresh.ground.factorization.n_start == 8
