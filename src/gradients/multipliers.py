"""The orbital-response (Z-vector) solve shared by every gradient route.

Both gradient engines close their Lagrangian on the same equation for the
multiplier Lambda of the orthonormality/canonical constraint (symmetric, zero
diagonal):

    antisym(Y_E + Y[fold(Lambda)]) = 0.

The routes differ only in how Y[fold(Lambda)] is applied (from the four-index
MO tensor in `grad_engine`, or from Coulomb/exchange builds through the ISDF
route's Fock-partial map), so that map is the argument; the antisymmetric
packing, degenerate-pair projection, energy-denominator preconditioner, lgmres
call and residual check live here.
"""
import numpy as np

from src.Base.constants import (ORBITAL_MULTIPLIER_DEGENERACY_TOL,
                                ORBITAL_MULTIPLIER_MAX_ITER,
                                ORBITAL_MULTIPLIER_RESIDUAL_TOL,
                                ORBITAL_MULTIPLIER_TOL)
from src.Base.utils.krylov import lgmres_root_driven
from src.Base.utils.mpi_grid import lockstep


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

    matvec: Lambda -> Y[fold(Lambda)], the route's orbital-rotation gradient
        of the folded multiplier; the only route-specific part of the solve.
    Y_E:   the target's orbital-rotation gradient; its antisymmetric part is
        the right-hand side.
    eps:   orbital energies, which set the pair basis (p < q) and the
        preconditioner 1/(2(eps_p - eps_q)).

    lgmres on the pair-space operator, preconditioned by the energy
    denominator (the operator's diagonal at the Hartree-Fock limit). Under
    ranks the solve is root-driven (`lgmres_root_driven`): rank 0 iterates
    and every rank ends on its Lambda.

    Exactly degenerate pairs are projected out (identity operator, zero
    solution): a rotation inside a degenerate set does not change the energy.
    Their right-hand side vanishes by symmetry, so a non-vanishing one above
    `residual_tol` raises. The residual is recomputed from the solution and
    checked against `residual_tol` because lgmres reports info=0 on
    stagnation too, and a short solve gives a plausible but wrong force.

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

    # rank 0 iterates under ranks: the matvec's Fock builds are collectives
    x, info = lgmres_root_driven(pair_matvec, rhs, precond,
                                 atol=tol * max(1.0, np.linalg.norm(rhs)),
                                 maxiter=max_iter)
    res = lockstep(np.linalg.norm(pair_matvec(x) - rhs))
    if verbose:
        print(f"    [multipliers] n={rhs.size} info={info} |res|={res:.3e}")
    if info != 0 or res > residual_tol * max(1.0, np.linalg.norm(rhs)):
        raise RuntimeError(f"multiplier solve failed: info={info}, "
                           f"res={res:.3e}")
    return vec_to_sym(np.where(deg, 0.0, x), norb, iu), res
