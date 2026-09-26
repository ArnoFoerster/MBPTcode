"""Gamma-point periodic ISDF.

The point of these tests is NOT accuracy -- it is to pin the Coulomb
convention while there is no k-index to hide a mistake in. Two things are
therefore asserted separately:

  * the grid quadrature itself, against pyscf's FFTDF (machine precision);
  * the ISDF fit on top of it, which must become EXACT once the number of
    interpolation points reaches the algebraic rank of the pair densities.

The second is the sharpest available test of the factorization: for nmo real
orbitals the pair densities rho_ij = rho_ji span exactly nmo(nmo+1)/2
dimensions, the pivoted Cholesky must find that rank and stop there, and at
that rank interpolation is not an approximation at all. Anything short of
~1e-9 there is a bug in the fit, not a resolution problem.
"""
import os
import sys

import numpy as np
import pytest
from pyscf.pbc import df, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.SingleReference.Periodic.pbc_isdf import (build_isdf_gamma, collocation,
                                                   thc_eri, uniform_grid)


def _diamond(basis='gth-szv'):
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
    cell.a = np.array([[0., 1.7834, 1.7834],
                       [1.7834, 0., 1.7834],
                       [1.7834, 1.7834, 0.]])
    cell.basis, cell.pseudo, cell.verbose = basis, 'gth-pade', 0
    cell.build()
    return cell


@pytest.fixture(scope='module')
def gamma_scf():
    cell = _diamond()
    mf = scf.KRHF(cell, kpts=cell.make_kpts([1, 1, 1])).density_fit()
    mf.kernel()
    assert mf.converged
    mo = mf.mo_coeff[0]
    nmo = mo.shape[1]
    eri = df.FFTDF(cell).ao2mo(mo, compact=False).real.reshape([nmo] * 4)
    return cell, mf, mo, eri


def test_collocation_is_orthonormal(gamma_scf):
    """The grid and its scalar weight reproduce the MO overlap.

    Necessary, and famously not sufficient: this check passed to 8e-13 while
    the Coulomb matrix was carrying an extra factor of vol/N.
    """
    cell, mf, mo, _ = gamma_scf
    coords, w = uniform_grid(cell)
    Phi = collocation(cell, coords, mo)
    S = (Phi.conj().T @ Phi).real * w
    assert abs(S - np.eye(mo.shape[1])).max() < 1e-10


def test_grid_quadrature_matches_fftdf(gamma_scf):
    """Our Coulomb convention against pyscf's, with no ISDF in between."""
    from pyscf.pbc import tools
    cell, mf, mo, eri_ref = gamma_scf
    nmo = mo.shape[1]
    coords, w = uniform_grid(cell)
    mesh, ngrid = cell.mesh, np.prod(cell.mesh)
    Phi = collocation(cell, coords, mo)
    coulG = tools.get_coulG(cell, k=np.zeros(3), mesh=mesh, Gv=cell.get_Gv(mesh))
    rho = np.einsum('ri,rj->ijr', Phi.conj(), Phi,
                    optimize=True).reshape(nmo * nmo, ngrid)
    rhoG = tools.fft(np.ascontiguousarray(rho), mesh)
    eri = (((rhoG.conj() * coulG) @ rhoG.T).real * (w / ngrid)).reshape([nmo] * 4)
    assert abs(eri - eri_ref).max() < 1e-12


def test_cholesky_finds_the_pair_density_rank(gamma_scf):
    """Point selection must terminate at nmo(nmo+1)/2, not before or after."""
    cell, mf, mo, _ = gamma_scf
    nmo = mo.shape[1]
    _, _, info = build_isdf_gamma(cell, mo, npoints=4 * nmo * nmo)
    assert len(info['points']) == nmo * (nmo + 1) // 2
    residuals = info['residuals']
    assert np.all(np.diff(residuals) <= 1e-12), "Schur residuals must decrease"


def test_isdf_is_exact_at_full_rank(gamma_scf):
    """At the full pair-density rank the ERI must be reproduced exactly."""
    cell, mf, mo, eri_ref = gamma_scf
    nmo = mo.shape[1]
    X, V, info = build_isdf_gamma(cell, mo, npoints=nmo * (nmo + 1) // 2)
    assert abs(thc_eri(X, V).real - eri_ref).max() < 1e-8


def test_error_decreases_with_rank(gamma_scf):
    """Below full rank the error must fall monotonically with npoints."""
    cell, mf, mo, eri_ref = gamma_scf
    nmo = mo.shape[1]
    errs = []
    for npts in (16, 24, 32):
        X, V, _ = build_isdf_gamma(cell, mo, npoints=npts)
        errs.append(abs(thc_eri(X, V).real - eri_ref).max())
    assert errs[0] > errs[1] > errs[2]


def test_gdf_is_not_an_exact_oracle(gamma_scf):
    """GDF differs from the converged grid ERI by ~2e-3 on this system.

    Pinned deliberately: the k-point oracles are all GDF-based
    (`PBCDFIntegrals`, `pbc_rpa`, `pbc_self_energy`), and if a later ISDF
    comparison is held to a tighter bar than this it will be measuring the
    auxiliary basis rather than the factorization.
    """
    from src.SingleReference.Periodic.pbc_integrals import PBCDFIntegrals
    cell, mf, mo, eri_ref = gamma_scf
    L = PBCDFIntegrals.from_scf(cell, mf).L[0][0]
    eri_gdf = np.einsum('Lij,Lkl->ijkl', L, L, optimize=True).real
    dev = abs(eri_gdf - eri_ref).max()
    assert 1e-4 < dev < 1e-2, f"GDF vs FFTDF deviation moved: {dev:.3e}"

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
