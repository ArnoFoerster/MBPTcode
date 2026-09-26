"""q-resolved ISDF with k-points.

The Gamma tests (test_pbc_isdf_gamma.py) pinned the Coulomb convention. What
is new here is momentum bookkeeping, and the two things that can go silently
wrong are the ones tested hardest:

  * zeta^{-q} = conj(zeta^q), which the ERI needs because the two pair
    densities in (ij|kl) carry opposite momenta. This is EXACT by relabelling
    the k-sum -- not an assumption about time reversal -- and is asserted at
    machine precision so that a future change to the k-sum (an IBZ
    restriction, say) cannot break it quietly;
  * the q=0 interpolation points reused at every q, which is an empirical
    claim and the load-bearing one for k-separability.

Oracle is FFTDF, not GDF -- see test_pbc_isdf_gamma.test_gdf_is_not_an_exact_oracle.
"""
import itertools
import os
import sys

import numpy as np
import pytest
from pyscf.pbc import df, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.SingleReference.Periodic.pbc_integrals import get_momentum_transfer_map
from src.SingleReference.Periodic.pbc_isdf import (build_isdf_kpts,
                                                   build_isdf_kpts_per_q,
                                                   collocation_kpts,
                                                   interpolating_vectors_q,
                                                   kpoint_minus_map,
                                                   select_points_cholesky_kpts,
                                                   thc_eri_kpts,
                                                   thc_eri_kpts_per_q,
                                                   uniform_grid)

KMESH = [2, 1, 1]


def _diamond():
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
    cell.a = np.array([[0., 1.7834, 1.7834],
                       [1.7834, 0., 1.7834],
                       [1.7834, 1.7834, 0.]])
    cell.basis, cell.pseudo, cell.verbose = 'gth-szv', 'gth-pade', 0
    cell.build()
    return cell


def _quadruples(cell, kpts):
    """All (k1,k2,k3,k4) with k1 - k2 + k3 - k4 = 0 (mod G)."""
    ks = cell.get_scaled_kpts(kpts)
    out = []
    for k1, k2, k3 in itertools.product(range(len(kpts)), repeat=3):
        d = ks - (ks[k1] - ks[k2] + ks[k3])
        m = np.where(np.linalg.norm(np.round(d) - d, axis=1) < 1e-8)[0]
        if len(m) == 1:
            out.append((k1, k2, k3, int(m[0])))
    return out


@pytest.fixture(scope='module')
def kscf():
    cell = _diamond()
    kpts = cell.make_kpts(KMESH)
    mf = scf.KRHF(cell, kpts=kpts).density_fit()
    mf.kernel()
    assert mf.converged
    mo = [np.asarray(c) for c in mf.mo_coeff]
    nmo = mo[0].shape[1]
    quads = _quadruples(cell, kpts)
    fftdf = df.FFTDF(cell, kpts)
    ref = {k: fftdf.ao2mo([mo[i] for i in k], kpts=[kpts[i] for i in k],
                          compact=False).reshape([nmo] * 4) for k in quads}
    return cell, kpts, mo, nmo, quads, ref


def test_kpoint_minus_map_inverts_the_transfer_map(kscf):
    cell, kpts, *_ = kscf
    kplus = get_momentum_transfer_map(cell, kpts)
    kminus = kpoint_minus_map(cell, kpts)
    for q in range(len(kpts)):
        assert np.array_equal(kminus[q, kplus[q]], np.arange(len(kpts)))


def test_zeta_at_minus_q_is_the_conjugate(kscf):
    """C^{-q} = conj(C^q) exactly, hence zeta^{-q} = conj(zeta^q).

    Relabelling k -> k+q in the k-sum is a bijection over the full mesh, so
    this holds with no time-reversal assumption -- which matters, because
    assuming phi^{-k} = phi^{k*} is only true up to a rotation inside
    degenerate blocks, and degeneracies live at exactly the high-symmetry
    k-points.
    """
    cell, kpts, mo, nmo, *_ = kscf
    coords, _ = uniform_grid(cell)
    Phi = collocation_kpts(cell, coords, mo, kpts)
    kminus = kpoint_minus_map(cell, kpts)
    points, _ = select_points_cholesky_kpts(Phi, 4 * nmo)
    for q in range(len(kpts)):
        mq = kminus[0, q]                       # index of -kpts[q]
        z_q = interpolating_vectors_q(Phi, kminus, q, points)
        z_mq = interpolating_vectors_q(Phi, kminus, mq, points)
        assert abs(z_mq - z_q.conj()).max() < 1e-10


def test_coulomb_matrix_is_hermitian(kscf):
    cell, kpts, mo, nmo, *_ = kscf
    _, V, _ = build_isdf_kpts(cell, mo, kpts, 4 * nmo)
    for q, Vq in enumerate(V):
        assert abs(Vq - Vq.conj().T).max() < 1e-10 * abs(Vq).max()


def test_eri_matches_fftdf_over_all_quadruples(kscf):
    """Every momentum-conserving quadruple, at the numerical rank."""
    cell, kpts, mo, nmo, quads, ref = kscf
    X, V, info = build_isdf_kpts(cell, mo, kpts, 16 * nmo)
    worst = max(abs(thc_eri_kpts(X, V, info['kminus'], *k) - ref[k]).max()
                for k in quads)
    assert worst < 1e-5, f"max ERI deviation {worst:.3e} over {len(quads)} quadruples"


def test_eri_error_decreases_with_rank(kscf):
    cell, kpts, mo, nmo, quads, ref = kscf
    errs = []
    for a in (3, 4, 6):
        X, V, info = build_isdf_kpts(cell, mo, kpts, a * nmo)
        errs.append(max(abs(thc_eri_kpts(X, V, info['kminus'], *k) - ref[k]).max()
                        for k in quads))
    assert errs[0] > errs[1] > errs[2]


def test_q0_points_cost_little_against_per_q_points(kscf):
    """Measured: reusing the q=0 points is what keeps X free of a q
    index; if it were expensive, k-separability would not be affordable.

    Measured penalty in max ERI error at alpha=8: 1.62x on a 2x2x1 mesh,
    1.03x on 2x2x2 -- i.e. it gets CHEAPER as the mesh grows, since the
    k-summed Gram matrix averages over more of the BZ and becomes less
    q-sensitive. A factor of 4 here would mean the assumption had failed.
    """
    cell, kpts, mo, nmo, quads, ref = kscf
    X, V, info = build_isdf_kpts(cell, mo, kpts, 4 * nmo)
    e0 = max(abs(thc_eri_kpts(X, V, info['kminus'], *k) - ref[k]).max() for k in quads)
    Xq, Vq, iq = build_isdf_kpts_per_q(cell, mo, kpts, 4 * nmo)
    eq = max(abs(thc_eri_kpts_per_q(Xq, Vq, iq['kminus'], *k) - ref[k]).max()
             for k in quads)
    assert e0 < 4 * eq, f"q=0 points cost {e0/eq:.2f}x -- assumption degraded"


def test_reduces_to_the_gamma_path(kscf):
    """nk=1 through the k-point code must reproduce the Gamma code exactly."""
    from src.SingleReference.Periodic.pbc_isdf import build_isdf_gamma, thc_eri
    cell = _diamond()
    kpts = cell.make_kpts([1, 1, 1])
    mf = scf.KRHF(cell, kpts=kpts).density_fit()
    mf.kernel()
    mo = [np.asarray(c) for c in mf.mo_coeff]
    nmo = mo[0].shape[1]
    npts = 4 * nmo

    Xk, Vk, ik = build_isdf_kpts(cell, mo, kpts, npts)
    Xg, Vg, ig = build_isdf_gamma(cell, mo[0], npts)
    assert np.array_equal(ik['points'], ig['points'])
    eri_k = thc_eri_kpts(Xk, Vk, ik['kminus'], 0, 0, 0, 0)
    eri_g = thc_eri(Xg, Vg)
    assert abs(eri_k - eri_g).max() < 1e-10

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
