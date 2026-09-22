"""The orbital-response (Z-vector) solve shared by every gradient route.

Both gradient engines close their Lagrangian on the same equation: the
multiplier Lambda of the orthonormality/canonical constraint, symmetric with a
zero diagonal, that makes the total orbital-rotation gradient symmetric,

    antisym(Y_E + Y[fold(Lambda)]) = 0.

What separates the routes is only how Y[fold(Lambda)] is applied -- from the
four-index MO integral tensor in `grad_engine`, from Coulomb/exchange builds
through the cubic ISDF route's own Fock-partial map elsewhere -- so that map is
the argument and everything around it (the antisymmetric packing, the
degenerate-pair projection, the energy-denominator preconditioner, the lgmres
call and the residual acceptance) lives here once.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres

from src.Base.constants import (ORBITAL_MULTIPLIER_DEGENERACY_TOL,
                                ORBITAL_MULTIPLIER_MAX_ITER,
                                ORBITAL_MULTIPLIER_RESIDUAL_TOL,
                                ORBITAL_MULTIPLIER_TOL)


def antisym_to_vec(Mmat, iu):
    """The strict upper triangle of M - M^T, the equation's own coordinates."""
    A = Mmat - Mmat.T
    return A[iu]


def vec_to_sym(v, norb, iu):
    """The symmetric zero-diagonal matrix whose upper triangle is v."""
    Lam = np.zeros((norb, norb))
    Lam[iu] = v
    return Lam + Lam.T


def solve_orbital_multipliers(matvec, Y_E, eps, *,
                              tol=ORBITAL_MULTIPLIER_TOL,
                              max_iter=ORBITAL_MULTIPLIER_MAX_ITER,
                              degeneracy_tol=ORBITAL_MULTIPLIER_DEGENERACY_TOL,
                              residual_tol=ORBITAL_MULTIPLIER_RESIDUAL_TOL,
                              verbose=False):
    """Solve antisym(Y_E + Y[fold(Lambda)]) = 0 for symmetric zero-diag Lambda.

    matvec: Lambda (symmetric, zero diagonal) -> Y[fold(Lambda)], the route's
        own orbital-rotation gradient of the folded multiplier. It is the ONLY
        thing the two engines do differently; nothing else about the solve is
        parameterized, because nothing else about it differs.
    Y_E:   the target's orbital-rotation gradient, whose antisymmetric part is
        the right-hand side.
    eps:   orbital energies, which set both the pair basis (p < q) and the
        preconditioner 1/(2(eps_p - eps_q)).

    lgmres on the pair-space operator, preconditioned by the energy
    denominator, which is the diagonal of the same operator at the
    Hartree-Fock limit and carries the whole conditioning of the problem.

    Exactly degenerate pairs are projected out -- the operator is replaced by
    the identity there and the pair is zeroed out of the solution -- because a
    rotation inside a degenerate set does not change the energy and the
    equation cannot determine it. Their right-hand side vanishes by that same
    symmetry, so a non-vanishing one is not a solvable case that was dropped
    but a broken assumption, and it refuses rather than return a Lambda that
    ignores part of the gradient. `residual_tol` bounds that too: it is the
    amount of gradient this solve is willing to discard, which is the same
    amount it is willing to leave in the residual.

    The residual is recomputed from the returned solution and checked against
    `residual_tol` because lgmres reports info=0 on stagnation as well as on
    convergence: the multiplier enters the force linearly, so a solve that
    stopped short leaves a gradient that is wrong by a smooth, plausible
    amount rather than one that visibly fails.

    Returns (Lambda, residual norm).
    """
    eps = np.asarray(eps, float)
    norb = len(eps)
    iu = np.triu_indices(norb, k=1)
    rhs = -antisym_to_vec(Y_E, iu)

    de = eps[iu[1]] - eps[iu[0]]           # eps_q - eps_p for p<q (row p, col q)
    deg = np.abs(de) < degeneracy_tol
    if deg.any():
        bad = np.abs(rhs[deg]).max() if rhs[deg].size else 0.0
        if bad > residual_tol:
            raise RuntimeError(f"degenerate orbital pair carries gradient "
                               f"{bad:.2e}; symmetry-adapted treatment needed")
        rhs = np.where(deg, 0.0, rhs)

    def pair_matvec(v):
        v = np.where(deg, 0.0, v)
        out = antisym_to_vec(matvec(vec_to_sym(v, norb, iu)), iu)
        return np.where(deg, v, out)

    def precond(v):
        return np.where(deg, v, v / (2.0 * np.where(deg, 1.0, -de)))

    n = rhs.size
    op = LinearOperator((n, n), matvec=pair_matvec)
    Mop = LinearOperator((n, n), matvec=precond)
    x, info = lgmres(op, rhs, M=Mop, rtol=0.0,
                     atol=tol * max(1.0, np.linalg.norm(rhs)), maxiter=max_iter)
    res = np.linalg.norm(pair_matvec(x) - rhs)
    if verbose:
        print(f"    [multipliers] n={n} info={info} |res|={res:.3e}")
    if info != 0 or res > residual_tol * max(1.0, np.linalg.norm(rhs)):
        raise RuntimeError(f"multiplier solve failed: info={info}, "
                           f"res={res:.3e}")
    return vec_to_sym(np.where(deg, 0.0, x), norb, iu), res
