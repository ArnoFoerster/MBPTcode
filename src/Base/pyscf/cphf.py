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
'cphf_not_converged' to that list, which a state model copies into its
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
from pyscf import lib

from src.Base.numerics.constants import CPHF_KRYLOV_LINDEP, CPHF_KRYLOV_TOL
from src.Base.numerics.numerics import numerics_or_default
from src.Base.numerical_flags import flag

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
        flag('cphf_not_converged', text, CphfNotConverged)
    return x
