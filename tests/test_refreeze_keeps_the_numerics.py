"""A refreeze and a checkpoint restore leave the `Numerics` exactly as asked.

A refrozen chain starts from the grids the chain before it resolved -- the
static W's tau count exactly, the contour-deformation size as a floor, an
explicit quasiparticle set as its orbitals, the dRPA quadrature -- and those
travel as `CarriedGrids` and in the conventions, not inside the `Numerics`.
So its identity is one across a refreeze and across a restore, which is what
lets a seam that combines two products refuse a numerics mismatch, and the
resolved values stay named in the conventions record and the force identity.

H2/cc-pVDZ Hartree-Fock on the G1 grid; nothing is solved.
"""
import pytest
from pyscf import gto

from src.gradients.rpa import chain_settings
from src.Base.numerics.numerics import Numerics
from src.Base.utils.record_io import record_hash
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.rpa_bse_surface import RPABSESurface
from src.gradients.rpa.ground_state import RPAGroundStateChain
from src.gradients.surface_identity import force_identity
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'
#: The grown quadratures a carried chain is handed.
NTAU_W, NFREQ_CD = 14, 96


def h2(bond):
    return gto.M(atom=f'H 0 0 0; H 0 0 {bond}', basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def g1():
    return Numerics(isdf={'counts': resolve_isdf_grid('G1', BASIS, ['H'])[0]})


def test_an_excited_refreeze_keeps_the_numerics_as_asked(g1):
    chain = ExcitedStateChain(h2(0.74), rhf_df_differentiable_checked, numerics=g1)
    fresh = chain.refreeze(h2(0.80))
    assert fresh.settings is g1
    assert fresh.settings.identity() == chain.settings.identity()
    assert fresh.settings.quadrature.ntau_w is None
    assert fresh.settings.quadrature.nfreq_cd is None


def test_the_resolved_grids_live_in_the_conventions(g1):
    chain = ExcitedStateChain(h2(0.74), rhf_df_differentiable_checked, numerics=g1)
    fresh = chain.refreeze(h2(0.80))
    record = fresh.conventions.to_record()
    assert record['ntau_w'] == chain.ntau_w == fresh.ntau_w
    assert fresh.conventions.nfreq_cd_floor == chain.conventions.nfreq_cd


def test_a_carried_quadrature_is_never_stepped_back_from(g1):
    """A chain handed the grids a chain before it grew to starts there, and
    its own refreeze hands them on, whatever the gap at the new geometry."""
    carried = chain_settings.CarriedGrids(ntau_w=NTAU_W, nfreq_cd=NFREQ_CD)
    chain = ExcitedStateChain(h2(0.74), rhf_df_differentiable_checked, numerics=g1, carried=carried)
    assert chain.ntau_w == NTAU_W
    assert chain.conventions.nfreq_cd == NFREQ_CD
    assert chain.conventions.nfreq_cd_floor == NFREQ_CD
    assert chain.settings is g1
    fresh = chain.refreeze(h2(1.20))
    assert fresh.ntau_w == NTAU_W
    assert fresh.conventions.nfreq_cd_floor == NFREQ_CD
    assert fresh.settings.identity() == g1.identity()


def test_the_force_identity_names_the_resolved_grids(g1):
    plain = ExcitedStateChain(h2(0.74), rhf_df_differentiable_checked, numerics=g1)
    grown = ExcitedStateChain(
        h2(0.74), rhf_df_differentiable_checked, numerics=g1,
        carried=chain_settings.CarriedGrids(ntau_w=NTAU_W, nfreq_cd=NFREQ_CD))
    assert plain.settings.identity() == grown.settings.identity()
    assert (record_hash(force_identity(plain))
            != record_hash(force_identity(grown)))


def test_a_restored_chain_holds_the_numerics_as_asked(g1):
    chain = ExcitedStateChain(h2(0.74), rhf_df_differentiable_checked, numerics=g1,
                              carried=chain_settings.CarriedGrids(ntau_w=NTAU_W,
                                                   nfreq_cd=NFREQ_CD))
    fresh = chain.refreeze(h2(0.80))
    back = ExcitedStateChain.from_record(fresh.to_record(), fresh.mol0, rhf_df_differentiable_checked)
    assert back.settings.identity() == g1.identity()
    assert back.ntau_w == NTAU_W
    assert back.conventions.nfreq_cd_floor == NFREQ_CD


def test_a_ground_refreeze_and_restore_keep_the_numerics_as_asked(g1):
    ground = RPAGroundStateChain(h2(0.74), rhf_df_differentiable_checked, numerics=g1)
    fresh = ground.refreeze(h2(0.80))
    assert fresh.settings is g1
    assert fresh.ntau == ground.ntau
    back = RPAGroundStateChain.from_record(fresh.to_record(), fresh.mol0, rhf_df_differentiable_checked)
    assert back.settings.identity() == g1.identity()
    assert back.ntau == ground.ntau
    assert len(back.grid.omega_points) == len(ground.grid.omega_points)


def test_a_composed_refreeze_keeps_both_halves_on_one_numerics(g1):
    surface = RPABSESurface(h2(0.74), rhf_df_differentiable_checked, numerics=g1)
    fresh = surface.refreeze(h2(0.80))
    assert (fresh.ground.settings.identity()
            == fresh.excited.settings.identity() == g1.identity())
