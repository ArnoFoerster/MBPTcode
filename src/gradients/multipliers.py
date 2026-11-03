"""The orbital-response (Z-vector) solve shared by every gradient route.

Every gradient closes its Lagrangian on the same equation for the multiplier
Lambda of the orthonormality/canonical constraint (symmetric, zero
diagonal):

    antisym(Y_E + Y[fold(Lambda)]) = 0.

The routes differ only in how Y[fold(Lambda)] is applied (from the four-index
MO tensor in `grad_engine`, or from Coulomb/exchange builds through
`rotation_gradient`, which the cubic GW-BSE route and the density-fitted
correlation forces share), so that map is the argument; the antisymmetric
packing, degenerate-pair projection and energy-denominator preconditioner live
here. A route whose Lagrangian is stationary in some pairs by construction
solves only the others (`free`). The solve itself is `Base.pyscf.cphf`'s, the
one every relaxed density solves too: its tolerances are the `scf` section's
`multiplier_*` fields of the caller's `Numerics`, and a solve that ends above
its residual tolerance is warned about and flagged there
('orbital_response_not_converged', appended to `flags` where the caller hands
a list). A degenerate pair that carries gradient is an input the equation does
not cover (a symmetry the caller has to adapt to) and is refused.

Every route then closes its Lagrangian the same way (`close_lagrangian`):
the stationary gradient Y = Y_E + Y[fold(Lambda)], the relaxed density
gamma + Lambda and the energy-weighted matrix M = -(Y + Y^T)/4, the
overlap's partial, whose one-electron skeleton is `lagrangian_skeleton`.
"""
from typing import NamedTuple

import numpy as np
from pyscf import lib

from src.Base.utils.parallel.mpi_grid import current_comm
from src.Base.numerics.constants import ORBITAL_MULTIPLIER_DEGENERACY_TOL
from src.Base.numerics.numerics import required_numerics
from src.Base.pyscf import cphf
from src.gradients.one_electron import one_electron_skeleton


def antisym_to_vec(Mmat, iu):
    """The strict upper triangle of M - M^T, the equation's own coordinates."""
    A = Mmat - Mmat.T
    return A[iu]


def vec_to_sym(v, norb, iu):
    """The symmetric zero-diagonal matrix whose upper triangle is v."""
    Lam = np.zeros((norb, norb))
    Lam[iu] = v
    return Lam + Lam.T


def rotation_gradient(C, fock, gamma, nocc, kernel):
    """Orbital-rotation gradient of Tr[gamma F], gamma a symmetric MO partial:

        Y[u,p] = 2 (F gamma)[u,p] + 4 delta_{p occupied} (C^T G(gamma_ao) C)[u,p]

    with gamma_ao = C gamma C^T, fock the MO Fock matrix and kernel the
    closed-shell response kernel G (AO density -> AO potential). The same
    expression serves a target's own partial and the multiplier's; it needs a
    Coulomb and an exchange build and no norb^4 tensor. A gamma without a
    virtual-virtual block goes to the kernel's `factored` where it has one
    (Base.pyscf.cphf.ResponseKernel). Any other gamma goes to the kernel
    tagged `read_columns` = Co: only G Co is read here, and a distributed JK
    fit builds the exchange of a tagged density as K Co
    (`DistributedDF._exchange_on_columns`), nocc / nao of a general K; a
    kernel that does not read the tag returns the whole G."""
    gs = 0.5 * (gamma + gamma.T)
    factored = getattr(kernel, 'factored', None)
    if factored is not None and not np.any(gs[nocc:, nocc:]):
        # no virtual-virtual block: C gs C^T = Co P^T + P Co^T
        P = C @ gs[:, :nocc] - 0.5 * C[:, :nocc] @ gs[:nocc, :nocc]
        G = factored(P)
    else:
        G = kernel(lib.tag_array(C @ gs @ C.T, read_columns=C[:, :nocc]))
    Y = 2.0 * (fock @ gs)
    Y[:, :nocc] += 4.0 * (C.T @ G @ C)[:, :nocc]
    return Y


def solve_orbital_multipliers(matvec, Y_E, eps, *, numerics, free=None,
                              degeneracy_tol=ORBITAL_MULTIPLIER_DEGENERACY_TOL,
                              verbose=False, flags=None):
    """Solve antisym(Y_E + Y[fold(Lambda)]) = 0 for symmetric zero-diag Lambda.

    matvec: Lambda -> Y[fold(Lambda)], the route's orbital-rotation gradient
        of the folded multiplier; the only route-specific part of the solve.
    Y_E:   the target's orbital-rotation gradient; its antisymmetric part is
        the right-hand side.
    eps:   orbital energies, which set the pair basis (p < q) and the
        preconditioner 1/(2(eps_p - eps_q)).
    numerics: the run's `Numerics`, required (None is refused); its
        `scf.multiplier_tol`, `multiplier_max_iter` and
        `multiplier_residual_tol` set the solve.
    free:  (norb, norb) booleans, read on p < q: the pairs solved. The others
        keep Lambda = 0 and their right-hand side is left as it is (a route
        whose Lagrangian is stationary there by construction reads it as a
        check). None solves every pair.

    `cphf.solve_operator` on the pair-space operator, preconditioned by the
    energy denominator (the operator's diagonal at the Hartree-Fock limit).
    Under ranks the solve is root-driven: rank 0 iterates and every rank
    ends on its Lambda.

    Exactly degenerate pairs are projected out (identity operator, zero
    solution): a rotation inside a degenerate set does not change the energy.
    Their right-hand side vanishes by symmetry, so a non-vanishing one above
    the residual tolerance is refused. A short solve gives a plausible but
    wrong force: it is warned about and flagged (`cphf.solve_operator`), the
    force carrying the residual's error.

    Returns (Lambda, residual norm).
    """
    residual_tol = required_numerics(
        numerics, 'solve_orbital_multipliers').scf.multiplier_residual_tol
    eps = np.asarray(eps, float)
    norb = len(eps)
    iu = np.triu_indices(norb, k=1)
    rhs = -antisym_to_vec(Y_E, iu)

    de = eps[iu[1]] - eps[iu[0]]           # eps_q - eps_p for p<q (row p, col q)
    deg = np.abs(de) < degeneracy_tol
    if free is not None:
        deg = deg & np.asarray(free, bool)[iu]
    if deg.any():
        bad = np.abs(rhs[deg]).max() if rhs[deg].size else 0.0
        if bad > residual_tol:
            raise RuntimeError(f"degenerate orbital pair carries gradient "
                               f"{bad:.2e}; symmetry-adapted treatment needed")
    fixed = deg if free is None else deg | ~np.asarray(free, bool)[iu]
    if fixed.any():
        rhs = np.where(fixed, 0.0, rhs)

    def pair_matvec(v):
        v = np.where(fixed, 0.0, v)
        out = antisym_to_vec(matvec(vec_to_sym(v, norb, iu)), iu)
        return np.where(fixed, v, out)

    # a fixed pair's operator is the identity: its denominator 1
    denominators = np.where(fixed, 1.0, 2.0 * -de)
    # rank 0 iterates under ranks: the matvec's Fock builds are collectives
    x, res, info = cphf.solve_operator(pair_matvec, rhs, denominators,
                                       numerics, flags)
    if verbose and (current_comm() is None or current_comm().Get_rank() == 0):
        print(f"    [multipliers] n={rhs.size} info={info} |res|={res:.3e}")
    return vec_to_sym(np.where(fixed, 0.0, x), norb, iu), res


class ClosedLagrangian(NamedTuple):
    """A Lagrangian closed on its multiplier (`close_lagrangian`), MO basis.

    gamma: the relaxed density, the target's partial plus Lambda.
    M: the energy-weighted matrix -(Y + Y^T)/4, the overlap's partial: with
        antisym(Y) = 0 the coefficient of a rotation dC = C X is
        Y_up + M_pu + M_up = 0.
    Y: the stationary orbital-rotation gradient Y_E + Y[fold(Lambda)].
    lam, res: the multiplier and its solve's residual norm.
    """
    gamma: np.ndarray
    M: np.ndarray
    Y: np.ndarray
    lam: np.ndarray
    res: float

    @property
    def stationarity(self):
        """max |Y - Y^T|, which the solve drives to its tolerance."""
        return float(np.abs(self.Y - self.Y.T).max())


def close_lagrangian(matvec, Y_E, eps, gamma, *, numerics, free=None,
                     flags=None, verbose=False, solved=None, total=None):
    """The `ClosedLagrangian` of a target with MO partial gamma (symmetric)
    and orbital-rotation gradient Y_E.

    matvec, eps, numerics, free, flags, verbose: `solve_orbital_multipliers`'s,
        which solves Lambda unless `solved` = (Lambda, residual) holds an
        earlier solve of the same right-hand side; numerics is required.
    total: Lambda -> Y where a route forms its stationary gradient in one
        build of the whole density (Y_E being that build at Lambda = 0);
        None takes Y_E + matvec(Lambda).
    """
    if solved is None:
        lam, res = solve_orbital_multipliers(matvec, Y_E, eps, numerics=numerics,
                                             free=free, verbose=verbose,
                                             flags=flags)
    else:
        lam, res = solved
    Y = Y_E + matvec(lam) if total is None else total(lam)
    return ClosedLagrangian(gamma + lam, -0.25 * (Y + Y.T), Y, lam, res)


def lagrangian_skeleton(mol, mf, closed):
    """(len(closed), natm, 3): each closed Lagrangian's one-electron skeleton,
    Tr[gamma dh] + Tr[M dS] in the AO basis of mf's orbitals
    (`one_electron.one_electron_skeleton`)."""
    C = np.asarray(mf.mo_coeff)
    return one_electron_skeleton(mol, mf, [C @ c.gamma @ C.T for c in closed],
                                 [C @ c.M @ C.T for c in closed])
