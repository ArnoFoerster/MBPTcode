"""Root following reads the overlap of the excitations, not of the cores.

`state_overlap` multiplies every element by D = det(T^oo)^2, the overlap of
the two reference determinants, and D falls with the displacement of the
nuclei whatever the states do: the core electrons ride on them. A rigid
translation leaves every state exactly what it was, yet formaldehyde/6-31G
moved 0.3 Bohr has D = 0.075, and the follower read that as the state's own
overlap and refused the step (`RootFollowingLost`). DMABN/cc-pVDZ, eleven heavy
atoms, reached the same verdict on the first step of its S1 walk (0.19 Bohr
rms, no rigid motion): D = 0.019, the overlap over D 0.996.

The translated geometry keeps the MO coefficients and the Casida vectors of
the reference unchanged, which is exact for a rigid translation: the basis
moves with the atoms.
"""
import numpy as np
import pytest
from pyscf import gto, scf, tdscf

from src.Base.numerics import Numerics
from src.Base.constants import ROOT_FOLLOW_WEIGHT_MIN
from src.gradients.conventions import FrozenConventions
from src.gradients.evaluation import GeometryCache
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.state_following import ForwardPieces
from src.properties.nonadiabatic import (mo_overlap, relative_overlap,
                                         state_overlap)
from src.SingleReference.LinearResponse.linear_response import from_pyscf

#: Rigid translation, Bohr: far enough that D is well below the weight floor.
SHIFT = 0.3
NROOTS = 4


@pytest.fixture(scope='module')
def reference():
    mol = gto.M(atom='C 0 0 0; O 0 0 1.205; H 0 0.94 -0.587; '
                     'H 0 -0.94 -0.587', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    td = tdscf.TDHF(mf)
    td.nstates = NROOTS
    td.kernel()
    _, x, y = from_pyscf(td)
    return mol, mf, x, y


def follower(mol):
    """An overlap-following chain with nothing built but what following reads."""
    chain = ExcitedStateChain.__new__(ExcitedStateChain)
    chain.settings = Numerics()
    chain.conventions = FrozenConventions()
    chain.track, chain.state, chain.nroots = 'overlap', 0, NROOTS
    chain.nocc = mol.nelectron // 2
    chain._followed = chain._anchor = None
    chain._evaluations, chain._owner = GeometryCache(), object()
    chain.follow_log, chain.follow_coords = [], []
    chain.timer = None
    return chain


def pieces(mol, mf, x, y):
    body = [None] * 17
    body[0], body[1], body[10], body[11] = mol, mf, x, y
    return ForwardPieces(tuple(body))


def translated(mol, shift):
    return mol.set_geom_(mol.atom_coords() + np.array([shift, 0.0, 0.0]),
                         unit='Bohr', inplace=False)


def test_the_reference_overlap_alone_falls_below_the_floor(reference):
    """The defect's size: D, not the excitation, takes the raw overlap down."""
    mol, mf, x, y = reference
    moved = translated(mol, SHIFT)
    s = state_overlap(mo_overlap(mol, mf.mo_coeff, moved, mf.mo_coeff),
                      mol.nelectron // 2, x, y, x, y)
    assert s[0, 0] < 0.1
    assert abs(s[1, 1]) < ROOT_FOLLOW_WEIGHT_MIN
    assert abs(relative_overlap(s)[1, 1]) > 0.9


@pytest.mark.parametrize('shift', [1e-3, SHIFT])
def test_a_translated_state_is_followed(reference, shift):
    mol, mf, x, y = reference
    chain = follower(mol)
    om = np.arange(NROOTS, dtype=float)
    chain.tracked_root(mol, mf, om, pieces(mol, mf, x, y))
    moved = translated(mol, shift)
    # without the wider re-solve: the state is among these roots
    root = chain.tracked_state(moved, mf, om, pieces(moved, mf, x, y))
    step = chain.follow_log[-1]
    assert root == 0
    assert step['weight'] > (1.0 - 1e-5 if shift < 1e-2 else 0.9)
    assert step['anchor'] > (1.0 - 1e-5 if shift < 1e-2 else 0.9)
