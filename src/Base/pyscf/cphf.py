"""The orbital-response (CPHF / Z-vector) linear solve, to a true residual.

Every relaxed density in this code ends in one equation over the
occupied-virtual rotations,

    diag * x + coupling(x) = rhs,

diag the orbital-energy differences e_a - e_i and coupling the response of
the Fock matrix to the rotation (pyscf's fvind). pyscf solves it with
lib.krylov (scf.cphf.solve, scf.ucphf.solve), which stops as soon as its
Krylov basis turns linearly dependent and returns what it has, whatever tol
is asked for: the stop is the dependence test, not the tolerance.

`solve` keeps pyscf's solver and refines: it solves for the remaining
residual again, the right-hand side scaled to unit norm (krylov's dependence
test is absolute, so a small residual would otherwise stop it at once),
until the true residual max |rhs - diag x - coupling(x)| is below tol. A
solve that ends above tol is a numerical event (refuse inputs, warn numerics):
it returns its last x, warns
(CphfNotConverged) and, inside `numerical_flags.collecting(flags)`, appends the flag
'orbital_response_not_converged' to that list, which a state model copies into its
record (src.gradients.state_models).

numerics: a `src.Base.numerics.numerics.Numerics`; its scf section holds the true
residual to reach (cphf_tol), the Krylov cycles of one solve
(cphf_max_cycle) and the rounds of refinement (cphf_refinements). None takes
them at their defaults.

precond: the denominators of the preconditioner, default diag. A floor on
small gaps (sign(d) max(|d|, shift)) speeds a hard case up without changing
the answer: the operator itself always carries the exact diag.
"""

import numpy as np
from pyscf import lib, scf

from src.Base.numerics.constants import (CPHF_KRYLOV_LINDEP, CPHF_KRYLOV_TOL,
                                         SCF_KERNEL_AUX_BLOCK)
from src.Base.numerics.numerics import numerics_or_default
from src.Base.numerical_flags import flag
from src.Base.utils.parallel.mpi_grid import ordered_sum

class CphfNotConverged(RuntimeWarning):
    """An orbital-response solve that ended above its true-residual tolerance
    (its last iterate returned)."""


def _krylov(aop, b, scf_numerics):
    """pyscf's lib.krylov of (1 + aop) x = b, at most scf_numerics
    .cphf_max_cycle cycles; at that cap, where pyscf
    raises, the solution in the subspace it built (the same projected solve
    pyscf ends with), so the refinement goes on and the true residual
    decides (module docstring)."""
    built = {}

    def keep(cycle, xs, ax):
        built['xs'], built['ax'] = xs, ax
    try:
        return lib.krylov(aop, b, tol=CPHF_KRYLOV_TOL,
                          max_cycle=scf_numerics.cphf_max_cycle,
                          lindep=CPHF_KRYLOV_LINDEP, callback=keep)
    except RuntimeError as exc:
        if 'failed to converge' not in str(exc) or 'xs' not in built:
            raise
    xs = np.array([np.asarray(x) for x in built['xs']])
    ax = np.array([np.asarray(a) for a in built['ax']])
    # pyscf's basis is orthogonal, not normalized: <x_i|x_i> on the diagonal
    h = xs.conj() @ ax.T + np.diag(np.einsum('ij,ij->i', xs.conj(), xs).real)
    c = np.linalg.solve(h, xs.conj() @ b.ravel())
    return (c @ xs).reshape(b.shape)


def solve(coupling, diag, rhs, numerics=None, precond=None):
    """x of diag * x + coupling(x) = rhs, shaped like rhs (module docstring).
    coupling maps an array shaped like rhs (or flat) to its response, either
    shape."""
    s = numerics_or_default(numerics).scf
    tol, refinements = s.cphf_tol, s.cphf_refinements
    rhs = np.asarray(rhs, float)
    shape = rhs.shape
    d = np.asarray(diag, float).reshape(shape)
    m = d if precond is None else np.asarray(precond, float).reshape(shape)
    shift = d - m

    def response(x):
        return np.asarray(coupling(x), float).reshape(shape)

    def aop(xs):
        xs = np.asarray(xs, float).reshape((-1,) + shape)
        out = [(shift * x + response(x)) / m for x in xs]
        return np.array(out).reshape(xs.shape[0], -1)

    x = np.zeros(shape)
    r = rhs.copy()
    for _ in range(refinements):
        if np.abs(r).max() < tol:
            return x
        b = (r / m).reshape(1, -1)
        nb = np.linalg.norm(b)
        dx = _krylov(aop, b / nb, s)
        x = x + nb * dx.reshape(shape)
        r = rhs - d * x - response(x)
    res = np.abs(r).max()
    if res >= tol:
        text = (f'CPHF not converged: true residual {res:.1e} after {refinements} '
                f'refinements (tol {tol:.0e}); its last iterate is returned')
        flag('orbital_response_not_converged', text, CphfNotConverged)
    return x


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
