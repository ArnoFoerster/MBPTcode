"""The excited chain and the composed surfaces are handed ONE `Numerics`.

Two records stored beside a `Numerics` built from them would leave every
setting with a stored copy beside the object the kernels read. The
constructors take one `numerics`, the chain holds that object as `settings`,
its record carries it, and the driver builds it (`surfaces.excited_kwargs`).

H2/cc-pVDZ Hartree-Fock on the G1 grid; nothing is solved.
"""
import inspect

import pytest
from pyscf import gto

import src.Base.chain_settings as chain_settings
from src.Base.numerics import Numerics
from src.Base.separable_ri import resolve_isdf_grid
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom=H2, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def numerics():
    counts = resolve_isdf_grid('G1', BASIS, ['H'])[0]
    return Numerics(isdf={'counts': counts}, bse={'nroots': 5},
                    qp={'residue_route': 'sop'})


def test_the_records_are_retired():
    assert not hasattr(chain_settings, 'ChainNumerics')
    assert not hasattr(chain_settings, 'ChainRealization')
    assert not hasattr(Numerics, 'from_chain_records')


@pytest.mark.parametrize('cls', [ExcitedStateChain, RPABSESurface,
                                 RPAQPSurface])
def test_one_object_is_handed_in(cls):
    params = inspect.signature(cls.__init__).parameters
    assert 'numerics' in params and 'realization' not in params


def test_the_chain_holds_the_object_it_was_handed(h2, numerics):
    chain = ExcitedStateChain(h2, rhf_df_differentiable_checked, numerics=numerics)
    assert chain.settings is numerics
    record = chain.to_record()
    assert record['settings'] is numerics
    assert 'chain_numerics' not in record
    assert 'chain_realization' not in record
    assert not any(k.startswith('chain_') for k in vars(chain))


def test_a_composed_surface_hands_its_halves_one_object(h2, numerics):
    surface = RPABSESurface(h2, rhf_df_differentiable_checked, numerics=numerics)
    assert surface.excited.settings is numerics
    assert surface.ground.factorization is surface.excited.factorization
