"""The chain's settings live in ONE place: its `Numerics`.

Settings copied onto flat attributes at construction, which the kernels read,
while `refreeze`, `to_record` and the force identity read a record, would let
a write to a flat attribute after construction -- `spin_view(track=)`,
`_wider_solve`'s `nroots`, a test -- move the kernels and not the record, so
the refrozen surface and the checkpoint would quietly go back to the
constructor's value. The chain holds one `Numerics` (`chain.settings`), every
setting is read off it, and a write replaces it.

H2/cc-pVDZ Hartree-Fock on the G1 grid.
"""
import pytest
from pyscf import gto

from src.Base.numerics import FLAT_FIELDS, Numerics
from src.Base.separable_ri import resolve_isdf_grid
from src.gradients.excited_state import ExcitedStateChain
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'

#: The flat names the kernels read, each a field of the chain's `Numerics`.
FLAT = ('ntau_gw', 'e_min_below_gap', 'n_poles', 'sop_stride', 'nroots',
        'bse_conv_tol', 'probe', 'degeneracy_tol', 'tile_gb',
        'residue_route', 'scissor', 'outside_scissor', 'qp_hamiltonian',
        'solver', 'bse_adjoint', 'track', 'qp_window')


def held(chain, name):
    section, key = FLAT_FIELDS[name].split('.')
    return getattr(getattr(chain.settings, section), key)


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom=H2, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def counts():
    return resolve_isdf_grid('G1', BASIS, ['H'])[0]


@pytest.fixture
def chain(h2, counts):
    return ExcitedStateChain(h2, rhf_df_differentiable_checked,
                             numerics=Numerics(isdf={'counts': counts}))


def test_the_flat_names_are_the_numerics(chain):
    """Every flat name the kernels read is a field of the chain's
    `Numerics`, not a copy."""
    for name in FLAT:
        assert name not in chain.__dict__, name
        assert getattr(chain, name) == held(chain, name)


def test_a_write_moves_the_numerics(chain):
    chain.nroots = 7
    chain.track = 'overlap'
    assert chain.settings.bse.nroots == 7
    assert chain.settings.bse.track == 'overlap'
    assert chain.to_record()['settings'].bse.track == 'overlap'


def test_a_write_is_validated_like_the_record(chain):
    with pytest.raises(ValueError, match='track'):
        chain.track = 'nearest'


def test_a_refreeze_carries_a_write(h2, chain):
    """The refrozen chain is built from the `Numerics`, so a setting written
    after construction travels."""
    chain.track = 'overlap'
    chain.bse_conv_tol = 1e-7
    fresh = chain.refreeze(h2)
    assert fresh.track == 'overlap'
    assert fresh.bse_conv_tol == 1e-7
