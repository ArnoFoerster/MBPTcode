"""The one fixed-point + DIIS loop of the CC solvers.

Every amplitude and Lambda solve here (amplitudes.kernel, the restricted T and
Lambda solves, the spin-orbital Lambda solve) is the same
iteration: build a Jacobi update of a list of blocks, extrapolate the flattened
blocks with DIIS (error vector = change between cycles), stop on a caller's
test. Only the update and the test differ, so they are passed in.

A singular DIIS B-matrix (tiny systems can converge so fast the error vectors
become exactly degenerate) keeps the undamped update for that step. Running out
of iterations is a numerical shortfall, not a bad input: the loop warns
(`CCNotConverged`) and returns the last iterate with `converged=False`.
"""
import warnings

import numpy as np

from .diis import DIIS


class CCNotConverged(RuntimeWarning):
    """A CC amplitude or Lambda iteration hit max_iter: the last iterate is
    returned and the caller's flag (`converged`) says so."""


def jacobi_step(residual, x, e, fock_e):
    """The diagonal update x + R / D, written (R + D x) / D: `e` is the stored
    1/D (integrals.energy_denominators) and `fock_e` = D its reciprocal."""
    return (residual + fock_e * x) * e


def iterate(step, blocks, is_converged, max_iter, diis_size=0, diis_start_cycle=2,
            what='CC iteration'):
    """Iterate `blocks` (a list of arrays) to a fixed point.

    step(blocks) -> (new_blocks, info): one Jacobi update; `info` is whatever
        the convergence test needs from before the extrapolation.
    is_converged(it, blocks, info) -> bool: called on the iterate after DIIS.
    diis_size: DIIS subspace size; 0 or None is plain fixed-point iteration.

    Returns (blocks, converged, n_iter). When max_iter is reached it warns
    `CCNotConverged` and returns the last iterate with converged False.
    """
    shapes = [b.shape for b in blocks]
    bounds = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    use_diis = bool(diis_size)
    if use_diis:
        diis = DIIS(diis_size, start_iter=diis_start_cycle)
        old_vec = np.hstack([b.ravel() for b in blocks])

    for it in range(max_iter):
        new_blocks, info = step(blocks)
        if use_diis:
            vec = np.hstack([b.ravel() for b in new_blocks])
            # drop the separate blocks now that they live inside `vec`: one
            # fewer full-size (rank-6 at CCSDT) array resident at the peak
            new_blocks = None
            error_vec = old_vec - vec
            try:
                vec = diis.compute_new_vec(vec, error_vec)
            except np.linalg.LinAlgError:
                pass
            old_vec = vec
            blocks = [vec[bounds[k]:bounds[k + 1]].reshape(s) for k, s in enumerate(shapes)]
        else:
            blocks = list(new_blocks)
        if is_converged(it, blocks, info):
            return blocks, True, it + 1

    warnings.warn(f'{what} did not converge in {max_iter} iterations; the last '
                  f'iterate is returned (converged=False)', CCNotConverged, stacklevel=3)
    return blocks, False, max_iter


#: per excitation rank, the transpose from the T/residual layout (nv,..,no,..)
#: to the Lambda layout (no,..,nv,..)
TO_L_LAYOUT = {1: (1, 0), 2: (2, 3, 0, 1), 3: (3, 4, 5, 0, 1, 2)}


def lambda_denominators(ranks, e_ai, e_abij, e_abcijk):
    """(e, fock_e) per Lambda block of the given ranks, in the Lambda layout."""
    e = {1: e_ai, 2: e_abij, 3: e_abcijk}
    return [(e[r].transpose(TO_L_LAYOUT[r]), np.reciprocal(e[r]).transpose(TO_L_LAYOUT[r]))
            for r in ranks]


def solve_lambda(residuals, l_blocks, denominators, names, max_iter, stopping_eps,
                 diis_size, diis_start_cycle, verbose, what):
    """The Lambda fixed point shared by the spin-orbital and restricted solves.

    residuals(l_blocks) -> list of Lambda residuals in the l-block layout;
    denominators: per block (e, fock_e) in that layout, see jacobi_step;
    names: one label per block for the iteration table.
    Converged when every block's change |dL| (before DIIS) is below stopping_eps.
    Returns (l_blocks, converged, n_iter).
    """
    def step(ls):
        rs = residuals(ls)
        new = [jacobi_step(r, l, e, fe) for r, l, (e, fe) in zip(rs, ls, denominators)]
        return new, [np.linalg.norm(n - l) for n, l in zip(new, ls)]

    def is_converged(it, ls, ds):
        if verbose:
            print(f"     {it:4d}   " + "      ".join(f"{d:.3e}" for d in ds), flush=True)
        return all(d < stopping_eps for d in ds)

    if verbose:
        # flush=True: same batch-scheduler buffering reason as amplitudes.kernel
        print(f"    ==> {what} amplitude equations (DIIS) <==", flush=True)
        print("     Iter   " + "".join(f"{'|d' + n + '|':15s}" for n in names), flush=True)
    return iterate(step, l_blocks, is_converged, max_iter, diis_size=diis_size,
                   diis_start_cycle=diis_start_cycle, what=what)
