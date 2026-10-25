"""Left/right pairing of the EOM-EE transition densities (src/SingleReference/
CC/eom.py) on degenerate states, and the reference component of the right
vectors.

- _pair_left_right: within a degenerate eigenvalue the left and right solves
  return unrelated bases, and the right solve may hold only part of the
  cluster; paired left vectors satisfy <L'_m|R_n> = delta_mn whenever the left
  vectors span the cluster's left eigenspace, and a cluster they do not span is
  left unpaired.
- transition_densities: a state without a left partner warns
  (UnpairedEOMStates), is listed in `unpaired_states` and gets zero densities.
- G0W@CC on He/6-31G with the whole EE manifold equals G0W@HF screened by the
  FCI response (exact for two electrons), which needs the r0 = -<Lambda|R_n>
  term of the totally symmetric states.

Synthetic matrices, H2/STO-3G and He/6-31G; each check runs in a few seconds.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.SingleReference.CC import eom
from src.SingleReference.GW.qp_energy import calc_qp_energy
from tests.fci_screened_gw import fci_screened_g0w_ip


def degenerate_eigensystem(seed=3):
    """A non-symmetric 6x6 matrix with a 3-fold eigenvalue: (omega, right
    eigenvectors as columns, left eigenvectors as rows)."""
    rng = np.random.default_rng(seed)
    omega = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 4.0])
    P = rng.normal(size=(6, 6))
    return omega, P, np.linalg.inv(P)


def test_a_partial_right_cluster_pairs_with_a_full_left_one():
    omega, P, Pinv = degenerate_eigensystem()
    rng = np.random.default_rng(7)
    # right: the eigenvalue 1 and two mixtures of the 3-fold cluster
    mix_r = rng.normal(size=(3, 2))
    R = [P[:, 0]] + list((P[:, 1:4] @ mix_r).T)
    omega_r = np.array([1.0, 2.0, 2.0])
    # left: a mixed basis of the whole cluster, plus the other eigenvalues
    mix_l = rng.normal(size=(3, 3))
    L = list(Pinv[[4, 0, 5]]) + list((mix_l @ Pinv[1:4]))
    omega_l = np.array([3.0, 1.0, 4.0, 2.0, 2.0, 2.0])
    paired = eom._pair_left_right(omega_r, R, omega_l, L, dual=lambda x: x)
    overlap = np.array([[lp @ r for r in R] for lp in paired])
    assert np.allclose(overlap, np.eye(3), atol=1e-10)


def test_a_cluster_the_left_vectors_do_not_span_stays_unpaired():
    omega, P, Pinv = degenerate_eigensystem()
    R = [P[:, 1], P[:, 2], P[:, 0]]
    omega_r = np.array([2.0, 2.0, 1.0])
    # two left vectors at the eigenvalue 2, one of them dual to neither right
    # vector: <L|R> has rank 1
    L = [Pinv[1], Pinv[3], Pinv[0]]
    omega_l = np.array([2.0, 2.0, 1.0])
    paired = eom._pair_left_right(omega_r, R, omega_l, L, dual=lambda x: x)
    assert paired[0] is None and paired[1] is None
    assert paired[2] @ R[2] == pytest.approx(1.0)


@pytest.fixture(scope='module')
def h2_sto3g():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    return eom.EOMCC(scf.RHF(mol).run(conv_tol=1e-12), level='ccsdt',
                     t_stopping_eps=1e-11)


def test_an_unpaired_state_warns_and_is_flagged(h2_sto3g):
    right = h2_sto3g.kernel('ee', nroots=4)
    left = h2_sto3g.kernel('ee', nroots=4, left=True)
    # the lowest cluster is a spin triplet; keep one of its three left vectors
    triplet = np.flatnonzero(abs(left.omega - right.omega[0]) < 1e-6)
    assert len(triplet) == 3
    keep = [n for n in range(len(left.omega)) if n not in triplet[1:]]
    partial = eom.EOMResult(h2_sto3g, 'ee', True, left.omega[keep],
                            [left.vectors[n] for n in keep], left.conv[keep])
    with pytest.warns(eom.UnpairedEOMStates, match='3 of 4 states'):
        rho, rho_star = right.transition_densities(left_result=partial)
    assert right.unpaired_states == [0, 1, 2]
    assert not rho[:3].any() and not rho_star[:3].any()
    assert abs(rho[3]).max() > 1e-3


def test_every_state_pairs_with_its_own_left_solve(h2_sto3g):
    right = h2_sto3g.kernel('ee', nroots=4)
    right.transition_densities()
    assert right.unpaired_states == []


def test_g0wcc_equals_the_fci_screened_g0w_on_he():
    mol = gto.M(atom='He 0 0 0', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    nstates = eom.EOMCC(mf, level='ccsdt').manifold_dim('ee')
    ip_cc = -calc_qp_energy(mf, selfenergy='GW', polarizability='CCSDT',
                            state='homo', nroots=nstates)
    assert ip_cc == pytest.approx(fci_screened_g0w_ip(mf), abs=1e-6)
