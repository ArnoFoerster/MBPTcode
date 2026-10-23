"""The ISDF forward reads the grid and the fit from the one Numerics it is handed.

- `separable_factors` with a Numerics alone builds the grid the Numerics names
  (it once refused: no grid asked for), bit for bit the grid of the keyword;
- a keyword naming another grid than the Numerics is refused (counts= once
  silently won over numerics.isdf.counts);
- `solve_bse_isdf` fits its factors on the Numerics it stamps on its result
  (it once fitted at the default fit settings and stamped the Numerics it was
  given, B0b): its roots equal the roots on factors fitted on that Numerics;
- `ISDFJK` reads numerics.scf.grid_level and refuses counts naming another grid.

Water/cc-pVDZ, RHF, G1; BSE@HF (qp=None), so no GW stage runs.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.factorization.isdf_jk import ISDFJK
from src.Base.numerics.numerics import Numerics
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.BSE.solve import solve_bse_isdf

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: a fit setting off its default, which moves the factors' bits
REGULARIZATION = 1e-6


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf


def counts_of(mol, level):
    return resolve_isdf_grid(level, mol.basis, ('O', 'H'))[0]


def test_a_numerics_alone_names_the_grid(water):
    mol, mf = water
    by_numerics = separable_factors(mf, mol, numerics=Numerics(isdf={'level': 'G1'}))
    by_keyword = separable_factors(mf, mol, grid_accuracy='G1')
    for a, b in zip(by_numerics, by_keyword):
        assert np.array_equal(a, b)


def test_a_keyword_naming_another_grid_is_refused(water):
    mol, mf = water
    numerics = Numerics(isdf={'counts': counts_of(mol, 'G1')})
    with pytest.raises(ValueError, match='two settings'):
        separable_factors(mf, mol, counts=counts_of(mol, 'G2'),
                          numerics=numerics)
    with pytest.raises(ValueError, match='two grids'):
        separable_factors(mf, mol, counts=counts_of(mol, 'G2'),
                          numerics=Numerics(isdf={'level': 'G1'}))
    # the same grid twice is one request
    separable_factors(mf, mol, counts=counts_of(mol, 'G1'), numerics=numerics)


def test_the_bse_fits_on_the_numerics_it_stamps(water):
    mol, mf = water
    nocc = mol.nelectron // 2
    numerics = Numerics(isdf={'regularization': REGULARIZATION},
                        bse={'nroots': 3, 'conv_tol': 1e-10})
    default_fit = separable_factors(mf, mol, grid_accuracy='G1')
    own_fit = separable_factors(mf, mol, grid_accuracy='G1', numerics=numerics)
    assert not np.array_equal(default_fit[1], own_fit[1])
    omega, _, _, info = solve_bse_isdf(mf, mol, nocc, numerics=numerics,
                                       qp=None, grid_accuracy='G1')
    omega_own = solve_bse_isdf(mf, mol, nocc, numerics=numerics, qp=None,
                               factors=own_fit)[0]
    assert info['numerics_identity'] == numerics.identity()
    assert np.array_equal(info['factors'][1], own_fit[1])
    assert np.allclose(omega, omega_own, rtol=0, atol=1e-12)


def test_the_isdf_k_mean_field_reads_its_grid_level(water):
    mol, _ = water
    numerics = Numerics(scf={'route': 'isdf', 'grid_level': 'G1'})
    jk = ISDFJK(mol, numerics=numerics)
    assert jk.counts == counts_of(mol, 'G1')
    with pytest.raises(ValueError, match='two grids'):
        ISDFJK(mol, counts=counts_of(mol, 'G2'), numerics=numerics)
    with pytest.raises(ValueError, match='two settings'):
        ISDFJK(mol, j_route='isdf',
               numerics=Numerics(scf={'j_route': 'df-direct'}))
