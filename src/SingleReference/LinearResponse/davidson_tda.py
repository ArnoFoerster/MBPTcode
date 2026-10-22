"""The Tamm-Dancoff Davidson: a Hermitian block iteration on the A half of the block action."""
import time

import numpy as np
from pyscf.lib import param

from src.Base.constants import (DAVIDSON_GUESS_FACTOR, DAVIDSON_LINDEP,
                                DAVIDSON_PRECONDITIONER,
                                DAVIDSON_SPACE_FRACTION, DAVIDSON_SPACE_GB)
from src.Base.utils.linearAlgebra.orthogonalization import extend_orthonormal
from src.Base.utils.memory import allocation_max_memory_mb
from src.SingleReference.LinearResponse.davidson_shared import (
    _action_working_bytes, _correction, _guess_indices,
    _preconditioner_diagonal, _real_eig_space, _tracked_roots,
    _trial_space_memory)


def _tda_davidson_root(apply_AB, diag_d, nroots, conv_tol, max_cycle,
                       guess_factor=DAVIDSON_GUESS_FACTOR, stats=None, counters=None,
                       max_memory=None, preconditioner=DAVIDSON_PRECONDITIONER,
                       guess=None, tracked=None):
    """(omega, X, Y, converged, cycles, None, unconfirmed): the `nroots`
    lowest roots of the Tamm-Dancoff problem A X = omega X, Y = 0,
    <X|X> = 1, by a Hermitian block Davidson on the A half of the block
    action.

    A is real symmetric, so the Rayleigh-Ritz step is one eigh of V A V^T on
    an orthonormal trial space and the roots are bounded from above by
    their Ritz values; no (A-B) definiteness enters, so the solve has no
    breakdown regime. Each correction r / (d - omega) is normalized before
    its part outside the space is measured, and kept while that part's
    squared norm exceeds DAVIDSON_LINDEP; corrections are orthogonalized
    twice (classical Gram-Schmidt). The space is held within the trial
    pairs `_run_davidson`'s budget gives real_eig (`_trial_space_memory`),
    which holds twice the vectors per pair this one does, and collapses
    onto its 2 * nroots lowest Ritz vectors when a cycle would overrun it.
    Every decision is dense arithmetic on the action's output, which an
    MPI action locksteps, so every rank runs the same iteration.
    `guess` is (X, Y): its X columns lead the unit vectors of the first
    space.

    The iteration tracks `_tracked_roots` Ritz pairs, more than `nroots`: a
    root whose Ritz value starts above the next root up is otherwise never
    corrected and the solve converges past it. An extra pair is corrected
    until it converges or theta - |r|, below which no eigenvalue it
    approximates can lie, clears the highest requested root; `unconfirmed`
    marks the requested roots above the theta - |r| of an extra pair still
    unsettled where the solve stopped (cycle cap, no new direction).
    `tracked`, a dict, receives 'omega', 'X' and 'Y' of every tracked pair.
    """
    no, nv = diag_d.shape
    n_pair = no * nv
    t0 = time.time()
    precond_d = _preconditioner_diagonal(apply_AB, diag_d, preconditioner)
    t_precond = time.time() - t0 if precond_d is not diag_d else 0.0
    hdiag = np.asarray(precond_d, float).ravel()
    order = _guess_indices(diag_d, nroots, guess_factor, counters=counters)
    space_inc = _real_eig_space(param.MAX_MEMORY, nroots, n_pair)[0]
    if max_memory is None:
        max_memory = allocation_max_memory_mb()
    working = _action_working_bytes(apply_AB, max(len(order), space_inc),
                                    n_pair, False)
    max_space = _real_eig_space(
        _trial_space_memory(nroots, n_pair, None, max_memory, working),
        nroots, n_pair)[1]
    track = _tracked_roots(nroots, n_pair)
    keep = min(max(2 * nroots, track), n_pair)   # Ritz vectors kept by a collapse
    t_action = [0.0]
    nvec = [0]

    def action(V):
        t = time.time()
        out = apply_AB(V.reshape(len(V), no, nv))[0].reshape(len(V), n_pair)
        t_action[0] += time.time() - t
        nvec[0] += len(V)
        return out

    x0 = np.zeros((len(order), n_pair))
    x0[np.arange(len(order)), order] = 1.0
    warm = 0
    if guess is not None:
        gx = np.asarray(guess[0], float).reshape(n_pair, -1).T
        x0 = np.vstack([gx, x0])
    V = _extend_rows(np.zeros((0, n_pair)), x0)[:max_space]
    if guess is not None:
        warm = min(gx.shape[0], len(V))
    t_loop0 = time.time()
    AV = action(V)
    h = V @ AV.T
    cycles, collapses, space_max = 1, 0, len(V)
    while True:
        h = 0.5 * (h + h.T)
        w, c = np.linalg.eigh(h)
        k = min(track, len(w))
        omega = w[:k]
        X = c[:, :k].T @ V
        R = c[:, :k].T @ AV - omega[:, None] * X
        r = np.linalg.norm(R, axis=1)
        converged = r <= conv_tol
        # an extra pair is settled once no eigenvalue within |r| of it can lie
        # below the highest requested root
        settled = converged.copy()
        settled[nroots:] |= omega[nroots:] - r[nroots:] > omega[:nroots].max()
        if settled.all() or len(V) == n_pair or cycles >= max_cycle:
            break
        todo = np.flatnonzero(~settled)
        t = _correction(R[todo], hdiag, omega[todo])
        if len(V) + len(todo) > max_space:
            # thick restart on the lowest Ritz vectors, A applied to them by
            # the same combination, so no action is spent
            m = min(keep, len(w))
            V, AV = c[:, :m].T @ V, c[:, :m].T @ AV
            h = np.diag(w[:m])
            collapses += 1
        new = _extend_rows(V, t)
        if not len(new):
            break                   # no direction survived the lindep test
        A_new = action(new)
        cycles += 1
        h = np.block([[h, V @ A_new.T], [new @ AV.T, new @ A_new.T]])
        V, AV = np.vstack([V, new]), np.vstack([AV, A_new])
        space_max = max(space_max, len(V))
    t_loop = time.time() - t_loop0         # the iteration, action included
    if stats is not None:
        stats.update({'davidson_vind_calls': cycles,
                      'davidson_block_actions': nvec[0],
                      'davidson_action_s': t_action[0],
                      'davidson_vind_s': t_action[0],
                      'davidson_dense_completion': False})
    if counters is not None:
        counters['action_s'] = t_action[0]
        counters['loop_s'] = t_loop
        counters['precond_s'] = t_precond
        counters['warm_start'] = warm
        counters['iterations'] = cycles
        counters['vectors'] = nvec[0]
        counters['space_max'] = space_max
        counters['space_bound'] = max_space
        counters['collapses'] = collapses
        counters['space_bytes'] = 16 * n_pair * space_max
        counters['space_budget_bytes'] = (
            DAVIDSON_SPACE_GB * 1e9 if max_memory is None
            else DAVIDSON_SPACE_FRACTION * max_memory * 1e6 - working)
        counters['working_bytes'] = working
    unsettled = np.flatnonzero(~settled[nroots:]) + nroots   # extra pairs
    lower = (np.inf if len(V) == n_pair or not unsettled.size
             else np.min(omega[unsettled] - r[unsettled]))
    if tracked is not None:
        tracked.update(omega=omega.copy(), X=X.T.copy(),
                       Y=np.zeros((n_pair, len(omega))))
    X = X[:nroots].T.copy()
    return (omega[:nroots], X, np.zeros_like(X), converged[:nroots], cycles,
            None, omega[:nroots] > lower)


def _extend_rows(V, T):
    """extend_orthonormal on rows: the rows of T made orthonormal to the
    orthonormal rows of V and to each other, kept while the relative norm left
    is above DAVIDSON_LINDEP^(1/2); (k, n)."""
    return np.ascontiguousarray(
        extend_orthonormal(V.T, T.T, rtol=DAVIDSON_LINDEP ** 0.5).T)
