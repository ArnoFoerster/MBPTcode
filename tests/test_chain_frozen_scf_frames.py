"""A chain's ISDF-K mean field at every geometry places its points on the
atomic frames of the chain's reference, turned with the molecule
(`isdf_jk.reference_frames`, `FactorChain.scf_frames`), and its force is the
slope of that energy away from the reference.

Why: a covariant frame near a degenerate neighbour tensor turns at 1/gap
(the ladder formaldehyde's carbon at 18.9 rad/Bohr), and the ISDF energy
depends on each cloud's orientation, so between two refreezes a walk's
surface dimples on a 1e-3 Bohr scale. Frozen at the anchor it is smooth; a
refreeze builds the next chain at its own reference and re-anchors there.

Water/cc-pVDZ Hartree-Fock on an ISDF-K mean field, the dRPA ground state,
the gradient 0.01 Bohr off the reference against a central
difference of the energy.
"""
import numpy as np
from pyscf import gto, scf

from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.Base.numerics.numerics import Numerics
from src.Base.geometry.geometry import molecule_at
from src.gradients.rpa.ground_state import RPAGroundStateChain
from tests.isdf_test_grid import g1 as g1_counts

H = 1e-3


def isdf_rhf(mol):
    mf = isdf_jk_with_force(scf.RHF(mol), auxbasis='cc-pvdz-ri', counts=g1_counts(mol))
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-11
    mf.kernel()
    return mf


def test_the_chain_scf_stands_on_the_reference_frames_and_its_slope():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='cc-pvdz', verbose=0)
    chain = RPAGroundStateChain(mol, isdf_rhf,
                                numerics=Numerics(isdf={'counts': g1_counts(mol)}))
    frames = chain.scf_frames
    assert frames is not None
    assert np.array_equal(frames.reference, mol.atom_coords())
    assert chain.mf0.with_df.frozen_frames is frames
    away = mol.atom_coords().copy()
    away[1, 1] += 1e-2
    m1 = molecule_at(mol, away)
    assert chain.mean_field(m1)[1].with_df.frozen_frames is frames
    g = np.asarray(chain.total_gradient(m1)[0])
    for ia, x in ((0, 2), (1, 1)):
        e = []
        for sign in (+1, -1):
            c = away.copy()
            c[ia, x] += sign * H
            e.append(chain.total_energy(molecule_at(mol, c)))
        assert abs(g[ia, x] - (e[0] - e[1]) / (2 * H)) < 5e-7
