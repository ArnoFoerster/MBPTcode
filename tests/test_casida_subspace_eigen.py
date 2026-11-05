"""The Casida Davidson's projected eigenproblem is solved to round-off however
the trial pairs are conditioned.

Each cycle of `trial_space.casida_rows` solves the projected pencil
[[a, b], [b, a]] z = omega [[sigma, pi], [-pi, -sigma]] z. pyscf's
`TDDFT_subspace_eigen_solver` inverts the LU factors of the scaled
sigma - pi, and its error grows as eps cond(sigma - pi)^2. In the basis
p = v + w, q = v - w of the trial pairs, sigma - pi is P^T Q, so a trial
pair whose p and q are almost orthogonal -- |x| close to |y| -- makes it
ill-conditioned. On water/cc-pVDZ BSE@LRC-wPBEh (density-fitted, S1 on the
sum-over-poles scissor chain) the 80th trial pair of one solve was such a
pair: cond 1.2e3, S1 2.2e-9 Ha too high, the solve stopped there, and the
excitation force missed its own 4-point difference by 1.5e-6 Ha/Bohr at
h = 1e-3 (tests/test_sop_force_follows_its_energy.py) while the force itself
was right.

Gated here, on a model pencil whose last trial pair has |p . q| = EPS:

  * pyscf's solver misses the generalized eigenvalues by more than
    PYSCF_MISS (the premise, so the gate can fail);
  * `subspace_eigen` meets them to ROOT_TOL, its solution has a residual
    under DAVIDSON_SUBSPACE_RESIDUAL_TOL and z^T M z = 1, each vector has
    pyscf's sign, and it says it re-solved;
  * on a well-conditioned pencil it returns pyscf's solution bitwise and
    says it did not.
"""
import os
import sys

import numpy as np
import pytest
import scipy.linalg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf.tdscf._lr_eig import TDDFT_subspace_eigen_solver   # noqa: E402

from src.Base.numerics.constants import DAVIDSON_SUBSPACE_RESIDUAL_TOL  # noqa: E402
from src.SingleReference.LinearResponse.trial_space import subspace_eigen  # noqa: E402

#: pairs, trial pairs, roots read
N_PAIR, M_TRIAL, NROOTS = 40, 12, 3
#: |p . q| of the last trial pair: cond(sigma - pi) = 1 / EPS. pyscf's
#: solver missed by 5.6e-10 to 1.1e-9 on three seeds here (workstation).
EPS = 1e-4
PYSCF_MISS = 1e-10
#: the re-solved roots against QZ on the same pencil
ROOT_TOL = 1e-12


def pencil(eps, seed=0):
    """(a, b, sigma, pi) of a stable Casida model projected on M_TRIAL
    orthonormal trial pairs whose last one has p . q = eps."""
    rng = np.random.default_rng(seed)
    d = np.sort(rng.uniform(0.3, 2.0, N_PAIR))
    k = rng.normal(size=(N_PAIR, N_PAIR)) * 0.02
    k = k + k.T
    big_a, big_b = np.diag(d) + k, 0.5 * k
    p = np.linalg.qr(rng.normal(size=(N_PAIR, M_TRIAL + 1)))[0]
    p, outside = p[:, :M_TRIAL], p[:, M_TRIAL]
    q = p.copy()
    q[:, -1] = eps * q[:, -1] + np.sqrt(1 - eps**2) * outside
    v, w = (p + q) / 2, (p - q) / 2
    a = v.T @ big_a @ v + w.T @ big_a @ w + v.T @ big_b @ w + w.T @ big_b @ v
    b = v.T @ big_b @ v + w.T @ big_b @ w + v.T @ big_a @ w + w.T @ big_a @ v
    sigma = v.T @ v - w.T @ w
    pi = v.T @ w - w.T @ v
    return (a + a.T) / 2, (b + b.T) / 2, (sigma + sigma.T) / 2, (pi - pi.T) / 2


def matrices(a, b, sigma, pi):
    return (np.block([[a, b], [b, a]]),
            np.block([[sigma, pi], [-pi, -sigma]]))


def qz_roots(a, b, sigma, pi):
    """The lowest NROOTS positive generalized eigenvalues, by QZ."""
    h, metric = matrices(a, b, sigma, pi)
    ev = scipy.linalg.eigvals(h, metric)
    ev = ev.real[(ev.real > 0) & (np.abs(ev.imag) < 1e-8)]
    return np.sort(ev)[:NROOTS]


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_an_ill_conditioned_pencil_is_solved_again(seed):
    blocks = pencil(EPS, seed)
    exact = qz_roots(*blocks)
    om_pyscf, x_pyscf, y_pyscf = TDDFT_subspace_eigen_solver(*blocks, NROOTS)
    assert np.abs(om_pyscf - exact).max() > PYSCF_MISS, \
        'the premise: pyscf\'s solver misses on this pencil'
    omega, x, y, refined = subspace_eigen(*blocks, NROOTS)
    assert refined
    assert np.abs(omega - exact).max() < ROOT_TOL, omega - exact
    h, metric = matrices(*blocks)
    z = np.vstack([x, y])
    residual = np.abs(h @ z - (metric @ z) * omega).max(axis=0)
    assert np.all(residual <= DAVIDSON_SUBSPACE_RESIDUAL_TOL
                  * np.abs(h).max() * np.abs(z).max(axis=0)), residual
    norm = np.einsum('ik,ij,jk->k', z, metric, z)
    assert np.abs(norm - 1).max() < 1e-12, norm
    overlap = np.sum(z * np.vstack([x_pyscf, y_pyscf]), axis=0)
    assert np.all(overlap > 0.5), overlap


def test_a_well_conditioned_pencil_keeps_pyscfs_bits():
    blocks = pencil(0.5)
    ref = TDDFT_subspace_eigen_solver(*blocks, NROOTS)
    omega, x, y, refined = subspace_eigen(*blocks, NROOTS)
    assert not refined
    for got, want in zip((omega, x, y), ref):
        assert np.array_equal(got, want)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
