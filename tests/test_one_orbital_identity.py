"""ONE orbital identity: the followed state is matched by the conventions'
`body_overlap`, and its record says so.

"The same orbital at two geometries" had two definitions: the
frozen conventions matched orbitals in Loewdin AOs turned with the molecule
(`conventions.body_overlap`, because the spatial overlap
refused core orbitals on a 0.2 Bohr DMABN step), while root following
matched the state through the spatial cross overlap (`nonadiabatic.
mo_overlap`) and patched the lost core by dividing by det(T_oo)^2. A state can
then be the same one to the conventions and another one to the follower.

The discriminating step is a rigid rotation: every orbital, every root and
every frozen convention is the same, and the conventions see it, but the
spatial overlap of an orbital with its own turned image is small (a p
orbital turned by 90 degrees is orthogonal to itself), so the spatial
follower loses the state it never left. The spatial overlap stays only where
it is physics, the derivative coupling's <Psi_I(R)|Psi_J(R+h)>.

Water/cc-pVDZ Hartree-Fock on the G1 grid, S1 followed by overlap.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics import Numerics
from src.Base.constants import ROOT_FOLLOW_WEIGHT_MIN
from src.Base.geometry import molecule_at
from src.Base.separable_ri import resolve_isdf_grid
from src.gradients.conventions import body_overlap
from src.gradients.excited_state import ExcitedStateChain
from src.properties.nonadiabatic import mo_overlap
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: Degrees the molecule turns, rigidly, about the x axis through its centroid.
TURN = 90.0


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=WATER, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def turned(water):
    a = np.deg2rad(TURN)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                    [0, np.sin(a), np.cos(a)]])
    c = water.atom_coords()
    centre = c.mean(axis=0)
    return molecule_at(water, (c - centre) @ rot.T + centre)


def test_the_spatial_overlap_loses_a_turned_orbital(water, turned):
    """The premise: the two identities disagree on this step."""
    mo0, mo1 = rhf_df_differentiable_checked(water).mo_coeff, rhf_df_differentiable_checked(turned).mo_coeff
    spatial = np.abs(mo_overlap(water, mo0, turned, mo1)).max(axis=1)
    body = np.abs(body_overlap(water, mo0, turned, mo1)).max(axis=1)
    assert body.min() > 0.99
    assert spatial.min() < ROOT_FOLLOW_WEIGHT_MIN


def test_the_followed_state_survives_a_rigid_turn(water, turned):
    counts = resolve_isdf_grid('G1', BASIS, ['H', 'O'])[0]
    chain = ExcitedStateChain(water, rhf_df_differentiable_checked, state=0, numerics=Numerics(isdf={'counts': counts},
                                                                     bse={'track': 'overlap'}))
    om0 = chain.excitation(water)
    chain.commit(water)
    om1 = chain.excitation(turned)
    step = chain.follow_log[-1]
    assert step['identity'] == 'body_overlap'
    assert all(s['identity'] == 'body_overlap' for s in chain.follow_log)
    assert step['index'] == 0 and step['weight'] > 0.99, step
    assert abs(om1 - om0) < 1e-6
