"""The ISDF collocation evaluates spherical functions, so a molecule with
cart=True is refused by name where the AOs are first put on a grid
(`separable_ri.ao_values`) instead of failing later on a shape. A spherical
molecule gets pySCF's own values.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.factorization.separable_ri import ao_values

ATOM = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
POINTS = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.5]])


def test_a_cartesian_basis_is_refused_by_name():
    mol = gto.M(atom=ATOM, basis='cc-pvdz', cart=True, verbose=0)
    with pytest.raises(ValueError, match='cart=True'):
        ao_values(mol, POINTS)


def test_a_spherical_basis_gets_pyscfs_values():
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    got = ao_values(mol, POINTS)
    assert got.shape == (len(POINTS), mol.nao_nr())
    assert np.array_equal(got, mol.eval_gto('GTOval_sph', POINTS))
