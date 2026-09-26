"""THC/ISDF on a two-dimensional cell.

Surface chemistry is the application. Unguarded, the ISDF route does not raise
on a slab either: there are TWO ways a low-dimensional cell corrupts `V^q` in
silence, and both are guarded here.

  1. THE NEGATIVE HEAD. `tools.get_coulG` returns v(G=0) = -2 pi L_z^2 for
     cell.dimension == 2 with the default ft type, and it is the largest entry
     in magnitude. V^q = sum_G v ahat ahat^* is PSD only while v >= 0, so the
     Coulomb matrix goes indefinite. The DF route catches this one step later
     on the RI-V metric; the ISDF route has no RI metric, so the negative head
     would propagate into the RPA logdet where the only guard is a
     determinant sign that fires on a symptom.
  2. `low_dim_ft_type='inf_vacuum'`. `get_Gv_weights` then returns a
     non-uniform Gauss-Chebyshev base along the vacuum axis while the ISDF
     Coulomb build FFTs on the uniform grid. Because pyscf forces an even
     vacuum mesh the LENGTHS still match, so coulG is paired with the wrong
     ahat and nothing raises. (That setting also breaks pyscf's own periodic
     SCF -- E(H2 slab) = -3830 Ha against -0.84.)

THE ORACLE IS NOT FFTDF HERE, and that matters. On a slab the physical kernel
is the DAMPED one, and `FFTDF.ao2mo` would build its ERIs with the bare
(negative-head) kernel -- a different object, so the comparison would measure
the kernel choice rather than the factorization. The oracle is instead the
exact grid quadrature with THE SAME kernel and no ISDF in between, as in
test_pbc_isdf_gamma.py: remove the factorization first, so a scale error
cannot hide behind a fitting error.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf.pbc import gto, scf, tools

from src.SingleReference.Periodic.pbc_isdf import (build_isdf_gamma,
                                                   build_isdf_kpts,
                                                   collocation, thc_eri,
                                                   uniform_grid)
from src.SingleReference.Periodic.pbc_isdf_rpa import rpa_ecorr_thc
from src.SingleReference.Periodic.pbc_rpa_damping import (make_coulG_damped,
                                                          nyquist_params)

KMESH = [2, 2, 1]
MESH = [13, 13, 72]


def _h2_slab(mesh=MESH, low_dim_ft_type=None, basis='gth-dzvp'):
    """A thin gapped slab: 2D, vacuum along z.

    gth-dzvp rather than szv on purpose: H2/szv has nmo = 2, so the pair
    densities span only nmo(nmo+1)/2 = 3 dimensions and the pivoted Cholesky
    terminates at rank 3 for every alpha >= 2. Every build is then at FULL
    rank, where interpolation is not an approximation -- the ERI comes back
    exact and the rank series is perfectly flat, which tests nothing about the
    fit. dzvp gives nmo = 10 and a real rank structure to converge.
    """
    cell = gto.Cell()
    cell.atom = 'H 0 0 -0.37; H 0 0 0.37'
    cell.a = np.diag([4.0, 4.0, 24.0])
    cell.basis, cell.pseudo = basis, 'gth-pade'
    cell.dimension = 2
    if mesh is not None:
        cell.mesh = list(mesh)
    cell.verbose = 0
    if low_dim_ft_type is not None:
        cell.low_dim_ft_type = low_dim_ft_type
    cell.build()
    return cell


@pytest.fixture(scope='module')
def slab():
    cell = _h2_slab()
    mf = scf.KRHF(cell, cell.make_kpts(KMESH), exxdiv=None).density_fit()
    mf.kernel()
    assert mf.converged
    r0, beta, _ = nyquist_params(cell, KMESH)
    return cell, mf, make_coulG_damped(r0, beta)


def _exact_grid_eri(cell, mo, coulG_fn, mesh=None):
    """(ij|kl) at Gamma by grid quadrature, with NO ISDF -- the oracle.

    Same convention as `coulomb_matrix`: one quadrature weight, not two.
    """
    mesh = cell.mesh if mesh is None else mesh
    ngrid = int(np.prod(mesh))
    coords, w = uniform_grid(cell, mesh)
    nmo = mo.shape[1]
    Phi = collocation(cell, coords, mo)
    Gv = cell.get_Gv(mesh)
    coulG = coulG_fn(cell, np.zeros(3), Gv)
    rho = np.einsum('ri,rj->ijr', Phi.conj(), Phi,
                    optimize=True).reshape(nmo * nmo, ngrid)
    rhoG = tools.fft(np.ascontiguousarray(rho), mesh)
    return (((rhoG.conj() * coulG) @ rhoG.T).real * (w / ngrid)).reshape([nmo] * 4)


# --- the two silent corruptions -------------------------------------------

def test_bare_2d_kernel_is_refused(slab):
    """[1] pyscf's default 2D kernel has a negative head; V^q would be indefinite."""
    cell, mf, _ = slab
    mo = [np.asarray(c) for c in mf.mo_coeff]
    nmo = mo[0].shape[1]
    # the premise, so this test cannot pass for the wrong reason
    coulG = tools.get_coulG(cell, k=np.zeros(3), mesh=cell.mesh,
                            Gv=cell.get_Gv(cell.mesh))
    assert coulG.min() < 0, 'expected a negative 2D head from pyscf'

    with pytest.raises(ValueError, match='negative'):
        build_isdf_kpts(cell, mo, cell.make_kpts(KMESH), 4 * nmo)


def test_inf_vacuum_is_refused(slab):
    """[2] G-ordering mismatch: same lengths, wrong correspondence, no error."""
    _, mf, cg = slab
    cell_iv = _h2_slab(low_dim_ft_type='inf_vacuum')
    mo = [np.asarray(c) for c in mf.mo_coeff]
    with pytest.raises(ValueError, match='inf_vacuum'):
        build_isdf_kpts(cell_iv, mo, cell_iv.make_kpts(KMESH),
                        4 * mo[0].shape[1], coulG_fn=cg)


def test_damping_support_is_checked(slab):
    """[3] A damped kernel whose support does not fit the vacuum must raise.

    This is `check_low_dim_support`, which the four DF builders call and the
    ISDF route -- a fifth consumer of the same coulG_fn seam -- did not.
    """
    cell, mf, _ = slab
    mo = [np.asarray(c) for c in mf.mo_coeff]
    r0, beta, _ = nyquist_params(cell, KMESH)
    # 8x, not 4x: at 4x the support is 16.3 bohr and still fits inside the
    # 22.7 bohr half-vacuum, so the guard is right not to fire there.
    too_wide = make_coulG_damped(r0=8.0 * r0, beta=beta)
    with pytest.raises(ValueError):
        build_isdf_kpts(cell, mo, cell.make_kpts(KMESH), 4 * mo[0].shape[1],
                        coulG_fn=too_wide)


# --- the factorization itself, on a slab ----------------------------------

def test_thc_eri_matches_the_exact_grid_on_a_slab(slab):
    """The Gamma ERI through THC against the same integral with no ISDF.

    Both sides use the damped kernel, so what is measured is the
    factorization and nothing else. The bar tracks the bulk case: the rank
    here is deliberately generous (alpha = 12) because the point of the test
    is the 2D plumbing, not the rank convergence.
    """
    cell, mf, cg = slab
    mo = np.asarray(mf.mo_coeff)[0]
    nmo = mo.shape[1]
    ref = _exact_grid_eri(cell, mo, cg)

    X, V, info = build_isdf_gamma(cell, mo, 12 * nmo, coulG_fn=cg)
    eri = thc_eri(X, V).real
    rel = abs(eri - ref).max() / abs(ref).max()
    print(f"  slab THC vs exact grid: max rel {rel:.3e} at npoints={len(info['points'])}")
    assert rel < 5e-2, rel


def test_slab_eri_improves_with_rank(slab):
    """Guards the previous test against being insensitive to the fit."""
    cell, mf, cg = slab
    mo = np.asarray(mf.mo_coeff)[0]
    nmo = mo.shape[1]
    ref = _exact_grid_eri(cell, mo, cg)
    errs = []
    for a in (1, 2, 4):
        X, V, _ = build_isdf_gamma(cell, mo, a * nmo, coulG_fn=cg)
        errs.append(abs(thc_eri(X, V).real - ref).max() / abs(ref).max())
    print(f"  slab ERI error vs alpha (1, 2, 4): "
          f"{errs[0]:.2e} {errs[1]:.2e} {errs[2]:.2e}")
    assert errs[0] > errs[1] > errs[2], errs


def test_ke_cutoff_selects_the_grid_and_the_cell_agrees(slab):
    """`ke_cutoff` must rebuild the cell, not just override the loop bound.

    A bare `mesh=` override would leave `cell.mesh` stale, so anything
    reading the cell would disagree with the grid actually used. The ISDF cost is
    LINEAR in N_r and pyscf's default slab mesh is set by the vacuum, so this
    knob is a first-order cost decision rather than tuning.
    """
    cell, mf, cg = slab
    mo = np.asarray(mf.mo_coeff)[0]
    _, _, info = build_isdf_gamma(cell, mo, 4 * mo.shape[1], coulG_fn=cg,
                                  ke_cutoff=60.0)
    expect = tools.cutoff_to_mesh(cell.lattice_vectors(), 60.0)
    assert list(info['mesh']) == list(expect)
    assert info['ngrid'] == int(np.prod(expect))

    # And the point of the knob: pyscf's DEFAULT mesh for this slab is set by
    # the VACUUM, not by the physics. Measured [83, 83, 493] = 3,396,277
    # points against 135,401 at ke_cutoff = 60 -- a factor of 25 on a build
    # that is LINEAR in N_r, so inheriting the default silently is a
    # first-order cost decision. (The fixture pins a small mesh of its own, so
    # this asks a default-mesh cell what it would have chosen.)
    default_mesh = _h2_slab(mesh=None).mesh
    ratio = int(np.prod(default_mesh)) / info['ngrid']
    print(f"  default slab mesh {list(default_mesh)} = {int(np.prod(default_mesh))} "
          f"pts, vs {info['ngrid']} at ke_cutoff=60 -- {ratio:.0f}x")
    assert ratio > 10, (default_mesh, info['mesh'], ratio)


def test_slab_correlation_energy_is_flat_in_ke_cutoff(slab):
    """The whole RPA path on a slab, and the grid converged by measurement.

    This is the slab gate: THC-RPA runs end to end on a two-dimensional cell
    with the damped kernel, and E_c is flat in the one knob that sets the
    cost. Measured (H2/gth-dzvp, 2x2x1): -0.05561961, -0.05557750,
    -0.05557847, -0.05557934 Ha at ke_cutoff 40/60/80/120, i.e. 0.042 mHa
    from 40 to 60 and then ~0.001 mHa -- converged to well under 1 mHa by 60,
    where the grid is 135k points against the 3.4M pyscf would have chosen.
    """
    cell, mf, cg = slab
    nmo = np.asarray(mf.mo_coeff).shape[-1]
    ec = []
    for ke in (60.0, 120.0):
        e, _ = rpa_ecorr_thc(cell, mf, 8 * nmo, nw=24, route='frequency',
                             coulG_fn=cg, ke_cutoff=ke)
        ec.append(e)
    print(f"  slab E_c: ke=60 {ec[0]:.8f}  ke=120 {ec[1]:.8f}  "
          f"d={abs(ec[1] - ec[0]) * 1e3:.4f} mHa")
    assert abs(ec[1] - ec[0]) < 1e-3, ec          # << 1 mHa
    assert ec[0] < 0, ec                          # and it is a correlation energy


def test_mesh_and_ke_cutoff_together_are_refused(slab):
    cell, mf, cg = slab
    mo = np.asarray(mf.mo_coeff)[0]
    with pytest.raises(ValueError, match='not both'):
        build_isdf_gamma(cell, mo, 4 * mo.shape[1], coulG_fn=cg,
                         mesh=[9, 9, 40], ke_cutoff=60.0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
