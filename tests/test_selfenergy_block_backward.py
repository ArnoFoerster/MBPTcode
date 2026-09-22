"""The adjoint of the self-energy MATRIX against finite differences.

`selfenergy_block` is Sigma^c_pq with both indices free and an optional mask on
the summed intermediate state -- with the environment orbitals of an active
window that mask makes it the embedding self-energy Sigma_c^{G^E W}, which is
what an embedded excited root's cubic gradient differentiates. Its adjoint has
to be right in eps, X and D separately, and the masked branch is a different
code path from the unmasked one, so both are gated.

A DIRECTIONAL derivative is the test, not an element-by-element Jacobian: the
adjoint returns one gradient per input, and contracting it with a random
direction must equal the central difference of the same scalar functional along
that direction. Central differences are O(h^2), so agreement is checked at a
few times that, never at machine precision.

The factorization here is random and small rather than a real molecule's. That
is deliberate: this gates the reverse-mode ALGEBRA, and a physical X, D would
only make the same test slower and its failures harder to localize. Whether the
factors mean anything is `test_downfolded.py`'s business.
"""
import numpy as np
import pytest

from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.gradients.space_time_adjoint import (selfenergy_block,
                                              selfenergy_block_backward,
                                              sigma_transforms)

NORB, NOCC, NAUX, NTAU, NFREQ = 8, 3, 6, 6, 6
STATES = [2, 3, 4]
#: Central-difference step. Small enough that O(h^2) truncation is under the
#: tolerance, large enough that cancellation in the difference is not.
H = 1e-5
#: The O(h^2) floor of a 2-point derivative of this functional, with room for
#: the conditioning of [1 - chi0]^-1.
TOL = 2e-6


def system(seed=7):
    """A small, well-conditioned (eps, X, D) and the grid that goes with it.

    The factors are scaled so chi0 stays well inside the radius where
    [1 - chi0]^-1 exists; a random X, D at full size puts the screened
    interaction next to its pole and the finite difference then measures the
    conditioning rather than the derivative.
    """
    rng = np.random.default_rng(seed)
    eps = np.sort(rng.normal(size=NORB))
    eps[NOCC:] += 2.0                       # a gap, so the branches separate
    X = 0.25 * rng.normal(size=(NAUX, NORB))
    D = 0.25 * rng.normal(size=(NAUX, NAUX))
    D = 0.5 * (D + D.T)                     # the auxiliary metric is symmetric
    gap = eps[NOCC] - eps[NOCC - 1]
    e_max = eps[-1] - eps[0]
    nu, wt = gauss_legendre_grid(NFREQ, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(NTAU, gap, e_max, nu, wt,
                                           with_sine=True, with_inverse=False)
    mu = 0.5 * (eps[NOCC - 1] + eps[NOCC])
    return eps, X, D, grid, mu


def functional(eps, X, D, grid, mu, transforms, a_re, a_im, intermediate):
    """A real scalar of Sigma: <a_re, Re Sigma> + <a_im, Im Sigma>."""
    sigma, _ = selfenergy_block(X, D, eps, NOCC, grid, STATES, transforms, mu,
                                intermediate=intermediate)
    return float((a_re * sigma.real).sum() + (a_im * sigma.imag).sum())


def setup(intermediate, seed=7):
    eps, X, D, grid, mu = system(seed)
    transforms = sigma_transforms(eps, NOCC, grid.tau_points, grid.omega_points,
                                  grid.omega_points, mu=mu,
                                  intermediate=intermediate)
    rng = np.random.default_rng(seed + 1)
    shape = (len(grid.omega_points), len(STATES), len(STATES))
    # Sigma_pq is symmetric, so only a symmetric adjoint is meaningful; an
    # antisymmetric part would be silently projected out and the finite
    # difference would disagree for a reason that is not a defect.
    a_re = rng.normal(size=shape)
    a_im = rng.normal(size=shape)
    a_re = 0.5 * (a_re + np.swapaxes(a_re, 1, 2))
    a_im = 0.5 * (a_im + np.swapaxes(a_im, 1, 2))
    return eps, X, D, grid, mu, transforms, a_re, a_im


def central(f, x, direction):
    """(f(x + h v) - f(x - h v)) / 2h, the O(h^2) directional derivative."""
    return (f(x + H * direction) - f(x - H * direction)) / (2.0 * H)


@pytest.mark.parametrize('masked', (False, True))
def test_eps_adjoint_matches_finite_difference(masked):
    inter = [0, 1, 5, 6, 7] if masked else None
    eps, X, D, grid, mu, tr, a_re, a_im = setup(inter)
    sigma, cache = selfenergy_block(X, D, eps, NOCC, grid, STATES, tr, mu,
                                    intermediate=inter)
    eps_bar, _, _ = selfenergy_block_backward(a_re, a_im, X, D, eps, NOCC, grid,
                                              STATES, tr, mu, cache,
                                              intermediate=inter)
    rng = np.random.default_rng(99)
    v = rng.normal(size=eps.shape)
    # The transforms are built FROM eps and are frozen parameters of the
    # surface, so they are held fixed here exactly as the adjoint assumes.
    num = central(lambda e: functional(e, X, D, grid, mu, tr, a_re, a_im, inter),
                  eps, v)
    assert abs(eps_bar @ v - num) < TOL * max(1.0, abs(num))


@pytest.mark.parametrize('masked', (False, True))
def test_x_adjoint_matches_finite_difference(masked):
    inter = [0, 1, 5, 6, 7] if masked else None
    eps, X, D, grid, mu, tr, a_re, a_im = setup(inter)
    sigma, cache = selfenergy_block(X, D, eps, NOCC, grid, STATES, tr, mu,
                                    intermediate=inter)
    _, X_bar, _ = selfenergy_block_backward(a_re, a_im, X, D, eps, NOCC, grid,
                                            STATES, tr, mu, cache,
                                            intermediate=inter)
    rng = np.random.default_rng(101)
    v = rng.normal(size=X.shape)
    num = central(lambda x: functional(eps, x, D, grid, mu, tr, a_re, a_im, inter),
                  X, v)
    assert abs((X_bar * v).sum() - num) < TOL * max(1.0, abs(num))


@pytest.mark.parametrize('masked', (False, True))
def test_d_adjoint_matches_finite_difference(masked):
    inter = [0, 1, 5, 6, 7] if masked else None
    eps, X, D, grid, mu, tr, a_re, a_im = setup(inter)
    sigma, cache = selfenergy_block(X, D, eps, NOCC, grid, STATES, tr, mu,
                                    intermediate=inter)
    _, _, D_bar = selfenergy_block_backward(a_re, a_im, X, D, eps, NOCC, grid,
                                            STATES, tr, mu, cache,
                                            intermediate=inter)
    rng = np.random.default_rng(103)
    v = rng.normal(size=D.shape)
    num = central(lambda d: functional(eps, X, d, grid, mu, tr, a_re, a_im, inter),
                  D, v)
    assert abs((D_bar * v).sum() - num) < TOL * max(1.0, abs(num))


def test_the_mask_actually_changes_the_answer():
    """Otherwise the masked tests above would pass on the unmasked code path.

    The threshold is a FRACTION of Sigma's own scale. An absolute one says
    nothing: these factors are deliberately small, so |Sigma| here is ~4e-4 and
    any absolute bound above that passes whether the mask did anything or not.
    """
    eps, X, D, grid, mu, tr, a_re, a_im = setup(None)
    inter = [0, 1, 5, 6, 7]
    tr_m = sigma_transforms(eps, NOCC, grid.tau_points, grid.omega_points,
                            grid.omega_points, mu=mu, intermediate=inter)
    full, _ = selfenergy_block(X, D, eps, NOCC, grid, STATES, tr, mu)
    part, _ = selfenergy_block(X, D, eps, NOCC, grid, STATES, tr_m, mu,
                               intermediate=inter)
    assert np.abs(full - part).max() > 0.1 * np.abs(full).max()


def test_the_block_diagonal_is_the_diagonal_self_energy():
    """`selfenergy_block` and `selfenergy_diag` are one quantity, so p == q of
    the matrix must be the vector, or the two routes have drifted apart."""
    from src.gradients.space_time_adjoint import selfenergy_diag
    eps, X, D, grid, mu = system()
    tr = sigma_transforms(eps, NOCC, grid.tau_points, grid.omega_points,
                          grid.omega_points, mu=mu)
    block, _ = selfenergy_block(X, D, eps, NOCC, grid, STATES, tr, mu)
    diag, _ = selfenergy_diag(X, D, eps, NOCC, grid, STATES, tr, mu)
    got = np.stack([np.diagonal(block[k]) for k in range(block.shape[0])])
    assert np.abs(got - diag.T).max() < 1e-12
