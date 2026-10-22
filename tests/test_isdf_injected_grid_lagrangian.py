"""The Lagrangian half of an ISDF-K force refuses points handed in from outside.

A factorization given `coords` without the radii that placed them has no
atom-local clouds, so the point chain of its exchange skeleton cannot be
built. The mean-field force refused it (`require_isdf_gradient_support`); the
excitation gradient reached the row fit's exchange skeleton, which silently
placed a grid of its own instead and differentiated that.
"""
import pytest
from pyscf import gto, scf

from src.Base.numerics import Numerics
from src.Base.isdf_jk import ISDFJK, isdf_jk
from src.gradients.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1, g1_numerics

pytestmark = pytest.mark.filterwarnings('ignore::RuntimeWarning')

AUXBASIS = 'cc-pvdz-ri'
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def injected_rhf(mol):
    """ISDF-K Hartree-Fock on points placed by another factorization and
    handed in as bare coordinates."""
    built = ISDFJK(mol, auxbasis=AUXBASIS, check_tol=None, counts=g1(mol))
    built.build()
    mf = isdf_jk(scf.RHF(mol), auxbasis=AUXBASIS, counts=g1(scf.RHF(mol)))
    mf.with_df.coords = built.coords
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 200
    mf.kernel()
    assert mf.with_df.grid_radii is None
    return mf


def test_excitation_gradient_refuses_injected_points():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    chain = ExcitedStateChain(mol, injected_rhf, state=0,
                              numerics=g1_numerics(mol, Numerics(isdf={'auxbasis': AUXBASIS},
                                                                 bse={'nroots': 3})))
    with pytest.raises(NotImplementedError, match='coords'):
        chain.excitation_gradient()
