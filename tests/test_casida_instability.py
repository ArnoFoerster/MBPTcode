"""The dense Casida solver refuses an unstable problem instead of repairing it.

`CasidaSolver.solve` reduces [[A, B], [B, A]] to omega^2 through (A + B) and
(A - B), which needs both positive definite. Shifting an indefinite
A + B by its lowest eigenvalue plus eta before the Cholesky factorization and
clipping omega^2 < 0 to eta^2 would return roots near sqrt(eta) for an
unstable reference with nothing said. The Davidson route
(`davidson._CasidaBreakdown`) and the quasi-boson BSE refuse the same case;
the dense one raises
`CasidaInstability`, a ValueError and a `StepFailed`.

Each planted problem is a symmetric pair with one instability, on both
branches of the solver: a non-diagonal A - B (Cholesky of A + B) and a
diagonal one (the orbital-energy-difference form). The healthy pair is the
same construction with every eigenvalue positive.
"""
import numpy as np
import pytest

from src.Base.step_failed import StepFailed
from src.SingleReference.LinearResponse.casida import (CasidaInstability,
                                                       CasidaSolver)

N = 6


def symmetric(eigenvalues, seed):
    """Q diag(eigenvalues) Q^T on a random orthogonal Q."""
    q = np.linalg.qr(np.random.default_rng(seed).standard_normal((N, N)))[0]
    return (q * np.asarray(eigenvalues, float)) @ q.T


def pair(apb, amb):
    """(A, B) with the given A + B and A - B."""
    return 0.5 * (apb + amb), 0.5 * (apb - amb)


HEALTHY = np.linspace(0.3, 1.2, N)
INDEFINITE = np.r_[-0.2, np.linspace(0.3, 1.2, N - 1)]


def test_the_exception_is_a_step_failure_and_a_value_error():
    assert issubclass(CasidaInstability, StepFailed)
    assert issubclass(CasidaInstability, ValueError)


def test_indefinite_a_plus_b_is_refused():
    A, B = pair(symmetric(INDEFINITE, 0), symmetric(HEALTHY, 1))
    with pytest.raises(CasidaInstability, match=r'\(A \+ B\) is not positive'):
        CasidaSolver(A, B).solve()


def test_indefinite_a_minus_b_is_refused_through_omega_squared():
    A, B = pair(symmetric(HEALTHY, 0), symmetric(INDEFINITE, 1))
    with pytest.raises(CasidaInstability, match=r'omega\^2 = .* <= 0'):
        CasidaSolver(A, B).solve()


def test_diagonal_a_minus_b_with_a_negative_entry_is_refused():
    A, B = pair(symmetric(HEALTHY, 0), np.diag(INDEFINITE))
    with pytest.raises(CasidaInstability, match=r'\(A - B\) is diagonal'):
        CasidaSolver(A, B).solve()


def test_diagonal_a_minus_b_with_indefinite_a_plus_b_is_refused():
    A, B = pair(symmetric(INDEFINITE, 0), np.diag(HEALTHY))
    with pytest.raises(CasidaInstability, match=r'omega\^2 = .* <= 0'):
        CasidaSolver(A, B).solve()


@pytest.mark.parametrize('diagonal', (False, True))
def test_a_stable_pair_solves_its_casida_equation(diagonal):
    """Where both halves are positive definite nothing is refused and the
    roots solve [[A, B], [B, A]] [X, Y] = omega [X, -Y]."""
    amb = np.diag(HEALTHY) if diagonal else symmetric(HEALTHY, 1)
    A, B = pair(symmetric(HEALTHY, 0), amb)
    omega, X, Y = CasidaSolver(A, B).solve()
    assert omega.min() > 0.0
    assert np.abs(A @ X + B @ Y - X * omega).max() < 1e-12
    assert np.abs(B @ X + A @ Y + Y * omega).max() < 1e-12
