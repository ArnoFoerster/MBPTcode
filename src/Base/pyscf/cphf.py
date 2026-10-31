"""The orbital-response (CPHF / Z-vector) linear solve, the one of every
relaxed density and every force.

Every relaxed density and every gradient Lagrangian in this code ends in one
equation over orbital rotations,

    diag * x + coupling(x) = rhs,

diag the orbital-energy differences and coupling the response of the Fock
matrix to the rotation (pyscf's fvind). A relaxed density and the force of
the same state therefore solve the same equation the same way: `solve` takes
that form, `solve_operator` the whole operator of a caller that applies it
in one piece (`src.gradients.multipliers`, over the antisymmetric pairs).

The solver is scipy's lgmres on the operator, preconditioned by the energy
denominators (diag, or `precond` where given), to an absolute residual of
`multiplier_tol` times max(1, |rhs|) with at most `multiplier_max_iter`
iterations. Under ranks it is root-driven (`krylov.lgmres_root_driven`):
rank 0 iterates, every rank applies the operator with it (a collective
coupling stays one) and ends on rank 0's x. A solve whose true residual is
already below the acceptance bar ends at the first outer cycle that lowers
it by less than ORBITAL_MULTIPLIER_STALL_RATIO: there the operator's
round-off floor lies above `multiplier_tol`, and lgmres would restart on it
to its ceiling.

The true residual |rhs - A x| is recomputed from the solution, since lgmres
reports success on stagnation too, and checked against
`multiplier_residual_tol` times max(1, |rhs|). A solve that ends above it is
a numerical event (refuse inputs, warn numerics): its last x is returned,
CphfNotConverged is warned and the flag 'orbital_response_not_converged' is
raised (`numerical_flags.flag`: inside `numerical_flags.collecting(flags)`
it reaches that list, and `flags` where the caller hands one), which a state
model copies into its record (src.gradients.state_models).

numerics: a `src.Base.numerics.numerics.Numerics`; its scf section's
multiplier_tol, multiplier_max_iter and multiplier_residual_tol set every
solve. None takes them at their defaults.

precond: the denominators of the preconditioner, default diag. A floor on
small gaps (sign(d) max(|d|, shift)) speeds a hard case up without changing
the answer: the operator itself always carries the exact diag.
"""

import numpy as np
from pyscf import lib, scf

from src.Base.numerics.constants import (ORBITAL_MULTIPLIER_STALL_RATIO,
                                         SCF_KERNEL_AUX_BLOCK)
from src.Base.numerics.numerics import numerics_or_default
from src.Base.numerical_flags import flag
from src.Base.utils.linear_algebra.krylov import lgmres_root_driven
from src.Base.utils.parallel.mpi_grid import lockstep, ordered_sum

class CphfNotConverged(RuntimeWarning):
    """An orbital-response solve that ended above its true-residual tolerance
    (its last iterate returned)."""


def solve_operator(apply, rhs, denominators, numerics=None, flags=None):
    """(x, true residual norm, lgmres info) of apply(x) = rhs, flat vectors,
    preconditioned by division by `denominators` (module docstring). Under
    ranks a collective: every rank calls it with the same rhs, and `apply`
    is called on every rank with rank 0's vector."""
    s = numerics_or_default(numerics).scf
    m = np.asarray(denominators, float)
    scale = max(1.0, np.linalg.norm(rhs))
    x, info = lgmres_root_driven(
        apply, rhs, lambda v: v / m, atol=s.multiplier_tol * scale,
        maxiter=s.multiplier_max_iter,
        stall=(s.multiplier_residual_tol * scale,
               ORBITAL_MULTIPLIER_STALL_RATIO))
    res = lockstep(np.linalg.norm(apply(x) - rhs))
    if info != 0 or res > s.multiplier_residual_tol * scale:
        not_converged(f'orbital-response solve not converged: info={info}, '
                      f'true residual {res:.3e} (tol '
                      f'{s.multiplier_residual_tol:.0e} relative); its last '
                      f'iterate is returned', flags)
    return x, res, info


def solve(coupling, diag, rhs, numerics=None, precond=None, flags=None):
    """x of diag * x + coupling(x) = rhs, shaped like rhs (module docstring).
    coupling maps an array shaped like rhs to its response, either shape."""
    rhs = np.asarray(rhs, float)
    shape = rhs.shape
    d = np.asarray(diag, float).ravel()
    m = d if precond is None else np.asarray(precond, float).ravel()

    def apply(x):
        x = np.asarray(x, float).ravel()
        return d * x + np.asarray(coupling(x.reshape(shape)), float).ravel()
    x, _, _ = solve_operator(apply, rhs.ravel(), m, numerics, flags)
    return np.asarray(x).reshape(shape)


def not_converged(text, flags=None):
    """Warn `CphfNotConverged` with `text` and raise the named flag
    'orbital_response_not_converged' (`numerical_flags.flag`), also into
    `flags` where it is a list: the one failure policy of every
    orbital-response solve."""
    flag('orbital_response_not_converged', text, CphfNotConverged,
         stacklevel=3, into=flags)


class ResponseKernel:
    """G(x) = J[x] - K[x]/2 of mf's own interaction (its JK fit, or the exact
    integrals when it has none), no environment, for symmetric AO x.

    With a JK fit the auxiliary blocks of `block` functions (default
    SCF_KERNEL_AUX_BLOCK) are dealt over comm's ranks (default none: serial)
    and each block's J and K summed in block order (ordered_sum): the same
    bits on every rank and at every rank count, a collective that every rank
    calls with the same x.
    `factored(P)` is G of x = Co P^T + P Co^T (Co the occupied orbitals, P
    (nao, nocc)), the shape of every orbital-rotation density without a
    virtual-virtual block: its exchange from the occupied half-transforms
    (L_Q Co)(L_Q P)^T, nocc / nao of the cost of a general x. Without a fit,
    pyscf's exact get_jk, serial on every rank."""

    def __init__(self, mf, comm=None, block=SCF_KERNEL_AUX_BLOCK):
        self.mf, self.comm, self.block = mf, comm, int(block)
        self.with_df = getattr(mf, 'with_df', None)
        nocc = mf.mol.nelectron // 2
        self.Co = np.ascontiguousarray(np.asarray(mf.mo_coeff)[:, :nocc])
        self._half = {}
        if self.with_df is None:
            self.m = scf.RHF(mf.mol)
            return
        if self.with_df._cderi is None:
            self.with_df.build()
        rank = 0 if comm is None else comm.Get_rank()
        size = 1 if comm is None else comm.Get_size()
        self.mine = [b for b in range(self._nblocks()) if b % size == rank]

    def _nblocks(self):
        return -(-self.with_df.get_naoaux() // self.block)

    def _blocks(self):
        """(b, L_b (nb, nao, nao)) of this rank's auxiliary blocks."""
        nao = self.mf.mol.nao
        for b, blk in enumerate(self.with_df.loop(self.block)):
            if b in self.mine:
                yield b, lib.unpack_tril(blk).reshape(-1, nao, nao)

    def _total(self, parts):
        nao = self.mf.mol.nao
        tot = ordered_sum(parts, self.comm, shape=(2, nao, nao))
        return tot[0] - 0.5 * tot[1]

    def __call__(self, x):
        x = np.asarray(x, float)
        if self.with_df is None:
            vj, vk = self.m.get_jk(self.mf.mol, x, hermi=1)
            return vj - 0.5 * vk
        nao = x.shape[0]
        parts = []
        for b, L in self._blocks():
            Lf = L.reshape(len(L), -1)
            J = (Lf @ x.ravel()) @ Lf
            T = L @ x                                  # (L_Q x)_{m l}
            K = T.transpose(1, 0, 2).reshape(nao, -1) @ L.reshape(-1, nao)
            parts.append((b, np.stack([J.reshape(nao, nao), K])))
        return self._total(parts)

    def factored(self, P):
        """G of x = Co P^T + P Co^T (class docstring)."""
        P = np.ascontiguousarray(P, dtype=float)
        if self.with_df is None:
            return self(self.Co @ P.T + P @ self.Co.T)
        nao = P.shape[0]
        parts = []
        for b, L in self._blocks():
            X = self._half.get(b)
            if X is None:
                X = self._half[b] = L @ self.Co        # (L_Q Co) (nb, nao, nocc)
            Y = L @ P
            rho = 2.0 * np.einsum('qmi,mi->q', X, P)
            J = (rho @ L.reshape(len(L), -1)).reshape(nao, nao)
            K = (X.transpose(1, 0, 2).reshape(nao, -1)
                 @ Y.transpose(1, 0, 2).reshape(nao, -1).T)
            parts.append((b, np.stack([J, K + K.T])))
        return self._total(parts)
