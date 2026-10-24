"""ONE function says what a force is a force of: `force_identity`.

Three hashes could answer it, each its own way: `conventions_hash` (the
excited half's conventions record, which a FirstPoint is matched on),
the ForceCache header's (the surface's whole record, walk history included,
or the halves' records, or a factorization's) and `optimize.surface_record` (the record a walk
checkpoint keeps, None for a composed surface). So three hashes would let a
FirstPoint handed between
two chains differing only in their numerics replay one's force on the
other, refuse a cached force once the surface had walked, and resume a
composed surface's walk on a freshly built surface with its committed root
following lost.

The identity is the conventions, the grid (the factorization's record
without its wall times), the settings (the physics and the chain's records)
and the environment, per half of a composed surface. Not the walk's history.

H2/cc-pVDZ Hartree-Fock on the G1 grid, as in test_one_refreeze.
"""
import numpy as np
import pytest
from pyscf import gto

from src.gradients.rpa.chain_settings import with_setting
from src.Base.numerics.numerics import Numerics
from src.Base.utils.record_io import record_hash
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.gradients.gwbse.excited_state import ExcitedStateChain, conventions_hash
from src.gradients.gwbse.rpa_bse_surface import RPABSESurface
from src.gradients.rpa.ground_state import RPAGroundStateChain
from src.gradients.surface_identity import force_identity
from src.properties.optimize import restored_surface, surface_record
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom=H2, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def counts():
    return resolve_isdf_grid('G1', BASIS, ['H'])[0]


@pytest.fixture(scope='module')
def mf0(h2):
    return rhf_df_differentiable_checked(h2)


def excited(h2, mf0, counts, numerics=None):
    numerics = Numerics() if numerics is None else numerics
    return ExcitedStateChain(h2, rhf_df_differentiable_checked, mf=mf0,
                             numerics=with_setting(numerics, 'isdf.counts',
                                                   counts))


def composed(h2, mf0, counts):
    return RPABSESurface(h2, rhf_df_differentiable_checked, mf=mf0,
                         numerics=Numerics(isdf={'counts': counts}))


def same(a, b):
    return record_hash(a) == record_hash(b)


def test_the_three_answers_are_one(h2, mf0, counts):
    chain = excited(h2, mf0, counts)
    identity = force_identity(chain)
    assert conventions_hash(chain) == record_hash(identity)
    surface = composed(h2, mf0, counts)
    assert conventions_hash(surface) == record_hash(force_identity(surface))
    ground = RPAGroundStateChain(h2, rhf_df_differentiable_checked, mf=mf0, numerics=Numerics(isdf={'counts': counts}))
    assert force_identity(ground) is not None


def test_settings_are_part_of_the_identity(h2, mf0, counts):
    """Two chains on one mean field and one grid whose conventions agree and
    whose numerics do not are forces of two surfaces."""
    a = excited(h2, mf0, counts)
    b = excited(h2, mf0, counts, numerics=Numerics(bse={'conv_tol': 1e-7}))
    assert same(a.conventions.to_record(), b.conventions.to_record())
    assert conventions_hash(a) != conventions_hash(b)


def test_the_walks_history_is_not(h2, mf0, counts):
    chain = excited(h2, mf0, counts)
    before = record_hash(force_identity(chain))
    chain.follow_log.append({'root': 0, 'overlap': 1.0})
    chain.follow_coords.append(np.array(h2.atom_coords()))
    assert record_hash(force_identity(chain)) == before
    assert conventions_hash(chain) == before


def test_a_composed_surface_checkpoint_restores_its_surface(h2, mf0, counts):
    """5.10: a composed walk's checkpoint stores its surface, and resume
    rebuilds it on ONE factorization with the excited half's record."""
    surface = composed(h2, mf0, counts)
    record = surface_record(surface)
    assert record is not None
    fresh = RPABSESurface(h2, rhf_df_differentiable_checked, mf=mf0,
                          numerics=Numerics(isdf={'counts': counts}),
                          state=1)
    restored = restored_surface(fresh, record)
    assert type(restored) is RPABSESurface and restored.state == 0
    assert restored.excited.factorization is restored.ground.factorization
    assert same(surface_record(restored), record)
    assert conventions_hash(restored) == conventions_hash(surface)


class Frames:
    """Frozen SCF frames as `FactorChain.scf_frames` holds them."""

    def __init__(self, reference):
        self.reference = reference

    def to_record(self):
        return {'convention': 'reference-body-fixed',
                'reference': self.reference}


def test_the_scf_frames_layout_and_span_are_in_it(h2, mf0, counts):
    """What every SCF of a walk is held to -- the frames its ISDF-K points
    sit on, the pair layout it fits on, the span it is solved in -- says
    which surface a force is of."""
    import copy
    chain = excited(h2, mf0, counts)
    base = conventions_hash(chain)
    framed = copy.copy(chain)
    framed.scf_frames = Frames([[0.0, 0.0, 0.0]])
    other = copy.copy(chain)
    other.scf_frames = Frames([[0.0, 0.0, 0.1]])
    assert len({base, conventions_hash(framed), conventions_hash(other)}) == 3
    laid = copy.copy(chain)
    laid.hold_scf_layout(np.arange(3))
    assert conventions_hash(laid) != base
    spanned = copy.copy(chain)
    spanned.span_basis = np.eye(2)
    assert conventions_hash(spanned) != base


def test_a_force_names_its_numerics_by_identity_and_not_by_layout(h2, mf0, counts):
    """Two chains differing in a choice (the tau count) are two surfaces; two
    differing only in the layout (the tile cap), which the identity leaves out, are one. The settings carry the object as its hash."""
    base = excited(h2, mf0, counts)
    other = excited(h2, mf0, counts,
                    Numerics(quadrature={'ntau_gw': base.settings.quadrature.ntau_gw + 2}))
    tiled = excited(h2, mf0, counts, Numerics(layout={'tile_gb': 1.5}))
    ident = force_identity(base)
    assert ident['settings']['settings'] == {'numerics': base.settings.identity()}
    assert record_hash(force_identity(other)) != record_hash(ident)
    assert record_hash(force_identity(tiled)) == record_hash(ident)
