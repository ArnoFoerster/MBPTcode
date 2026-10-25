"""No ISDF entry point picks a grid nobody asked for.

A call naming neither counts, a level nor radii is refused by every entry
point (`separable_ri.require_grid`): a grid nobody validated is a different
factorization, not a coarse one, so no grid is a default.
"""
import pytest
from pyscf import gto, scf

from src.Base.factorization.isdf_jk import ISDFJK, isdf_grid
from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.Base.factorization.separable_ri import optimize_atomic_radii
from src.SingleReference.GW.space_time import separable_factors
from src.gradients.rpa.factor_chain import FrozenFactorization

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom=WATER, basis='cc-pvdz', verbose=0)


@pytest.mark.parametrize('call', [
    lambda mol: isdf_grid(mol),
    lambda mol: ISDFJK(mol),
    lambda mol: isdf_jk_with_force(scf.RHF(mol)),
    lambda mol: separable_factors(scf.RHF(mol), mol),
    lambda mol: FrozenFactorization(mol),
    lambda mol: optimize_atomic_radii('O', 'cc-pvdz', 'cc-pvdz-ri'),
], ids=['isdf_grid', 'ISDFJK', 'isdf_jk', 'separable_factors',
        'FrozenFactorization', 'optimize_atomic_radii'])
def test_a_call_without_a_grid_is_refused(mol, call):
    with pytest.raises(ValueError, match='no ISDF grid asked for'):
        call(mol)

