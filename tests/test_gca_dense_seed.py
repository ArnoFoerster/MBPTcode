"""The dense quasiparticle surface's root at a geometry does not depend on
what was evaluated before it.

`QuasiparticleSurface` starts its Newton at a seed. Carried over from the last
evaluation, the root near the reference after a far probe started from the
far root and the root there alone from the reference one: two numbers for one
geometry. Water/cc-pVDZ Hartree-Fock, ionization from the inner-valence
orbital 1: after a probe FAR Bohr along one O-H stretch, the Newton at NEAR
Bohr on the other side landed on another branch, 48 mHa from the root it
reaches alone. The seed is the reference root, frozen at construction as
`DenseBSESurface` freezes its roots, and the root at NEAR is the same either
way, for the orbital handed to the surface and for the one it chooses: to a
few ulp (`assert_same` of tests/pins.py), since the two surfaces each solve
their own reference, so not bitwise.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.geometry import molecule_at
from src.gradients.dense_surfaces import QuasiparticleSurface, tight_rhf
from tests.pins import assert_same

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: Bohr along the stretch: a far probe, then a geometry near the reference
FAR, NEAR = 0.6, -0.02
INNER_VALENCE = 1


@pytest.fixture(scope='module')
def geometries():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    move = np.zeros((mol.natm, 3))
    move[1, 1] = 1.0
    far = molecule_at(mol, mol.atom_coords() + FAR * move)
    near = molecule_at(mol, mol.atom_coords() + NEAR * move)
    return mol, far, near, tight_rhf(far), tight_rhf(near)


@pytest.mark.parametrize('orbital', [INNER_VALENCE, None])
def test_root_at_a_geometry_is_independent_of_the_call_history(geometries,
                                                               orbital):
    mol, far, near, mf_far, mf_near = geometries

    def surface():
        return QuasiparticleSurface(mol, tight_rhf, -1, orbital=orbital)

    alone = surface().quasiparticle_energy(near, mf_near)
    walked = surface()
    walked.quasiparticle_energy(far, mf_far)
    after = walked.quasiparticle_energy(near, mf_near)
    assert_same(after, alone, quantity='energy', label='root at NEAR')
