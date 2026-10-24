"""The (A - B) probe: the lowest eigenvalue of A - B, whose sign decides whether the Casida omega^2 reduction is valid.

The one symmetric Davidson (`Solvers.davidson.solve_symmetric`) on the block
action's A - B, the same iteration serially and over ranks, run cold from 1/d or
started from a Davidson's converged roots (`_lowest_amb_local`).
"""
import time
from types import SimpleNamespace

import numpy as np

from src.Base.numerics.constants import (AMB_PROBE_SIGN_TOLS, AMB_PROBE_TOL,
                                         PROBE_START_MIX)
from src.Base.utils.parallel.mpi_grid import current_comm, lockstep
from src.Solvers.davidson import diagonal_seeds, solve_symmetric
from src.SingleReference.LinearResponse.block_actions import block_action


class AmbProbeNotConverged(RuntimeError):
    """The (A-B) Davidson left a root above its residual bar."""


def lowest_amb_eigenvalue(lr_solver, nocc, polarizability='BSE', W_aux=None,
                          isdf_factors=None, k=1, tol=AMB_PROBE_TOL, sign_only=False,
                          stats=None, comm=None):
    """Lowest eigenvalue(s) of (A - B): the sign that decides whether the
    Casida omega^2 reduction is valid at all.

    Matrix-free (the symmetric Davidson on z -> (A-B)z through the same block
    action the solver uses), so it runs at any size the action runs at: the
    instability check where nothing dense fits. min eig < 0 confirms the
    regime `solve_casida_davidson` refuses; comparing the DF and ISDF actions'
    values at a size where both fit tests whether the factorization, not the
    physics, pushed a near-zero mode negative.

    pyscf's mf.stability() is not a proxy for this: 90-degree twisted ethene
    reports internally stable there while (A-B) here is indefinite
    (-0.005 Ha); the SCF Hessian under real singlet rotations is a different
    condition from response-metric positive definiteness.

    The probe is a property of the reference, not of the molecule. Twisted
    ethene has (at least) two converged RHF solutions 31 mHa apart, and
    pyscf's default minao guess can land on the higher one, where min eig(A-B)
    is far more negative (TDHF -0.067 vs -0.005 on the lowest solution); on
    both, screening moves the minimum upward. Which guess finds the lowest
    solution is system-dependent (on the acenes, with spreads up to 107 mHa,
    minao lands on the lowest of three and '1e' is the outlier). So compare
    E_SCF across initial guesses, probe the reference and kernel you will
    use, and if the probe comes back negative, establish that the SCF is the
    lowest solution before concluding the system itself is unstable.

    Spin does not enter: kappa (ia|jb) is the same term in A and B, so it cancels
    out of the difference and a triplet has the same (A - B) as its singlet.

    comm: the communicator the ISDF action divides its rows over, by default
    the current `distributed` region's; every rank then runs the same Davidson
    and returns rank 0's value (`lowest_amb_from_action`).

    Standalone the probe has no converged roots to try first, so it starts
    cold, from 1/d; `solve_bse_isdf` runs it after its Davidson instead, on
    the Davidson's own action, and under 'sign' tries the roots first.
    """
    comm = current_comm() if comm is None else comm
    apply_AB, diag_d = block_action(lr_solver, nocc, polarizability, W_aux,
                                     isdf_factors, comm=comm)
    return lowest_amb_from_action(apply_AB, diag_d, k=k, tol=tol,
                                   sign_only=sign_only, stats=stats, comm=comm)


def lowest_amb_from_action(apply_AB, diag_d, k=1, tol=AMB_PROBE_TOL, sign_only=False,
                            stats=None, comm=None, start=None):
    """The (A-B) probe, the same iteration on every rank, rank 0's value on each.

    The Davidson is a function of the operator's outputs alone, so ranks handed
    the distributed action's lockstepped output take the same steps, threads of
    one process (`run_simulated`) included. The value is lockstepped at the
    end; ranks whose iterations had fallen apart meet that lockstep against
    another rank's block-action lockstep and raise ValueError together instead
    of deadlocking.

    start: the converged Casida roots' X - Y, (n_pair, nroots) with the
    lowest root first, or one root's (n_pair,), the same on every rank; read
    by the sign-only probe alone (`_lowest_amb_local`).
    """
    replicated = comm is not None and comm.Get_size() > 1
    out = _lowest_amb_local(apply_AB, diag_d, k, tol, sign_only, stats,
                            comm if replicated else None, start)
    return lockstep(out, comm) if replicated else out


def _lowest_amb_local(apply_AB, diag_d, k=1, tol=AMB_PROBE_TOL, sign_only=False,
                      stats=None, comm=None, start=None):
    """The probe itself: lowest eigenvalue(s) of (A - B) by the symmetric
    Davidson; the sign-only probe after a Casida solve tries its converged
    roots (`start`) first.

    What the roots prove: Rayleigh-Ritz of (A - B) on the span of the roots'
    X - Y gives a Ritz value theta and residual r, and |lambda - theta| <= r
    holds for some eigenvalue lambda of (A - B): theta - r > 0 proves that
    one positive. theta is at least the minimum over the span, so the bound
    reaches the minimum only when the span does; a minimum in a pair irrep no
    root lies in is invisible to it. The general probe is the Davidson from
    1/d, which has weight in every irrep: the escalating passes below and the
    converged probe start there.

    stats receives `probe_matvecs` (every block action the probe applied,
    one per root in the span), `probe_action_s` and `probe_source`: 'roots'
    where the span's certificate held, 'davidson-warm' for the pass started
    from the lowest root, 'davidson-cold' for one started from 1/d.
    """
    replicated = comm is not None
    no, nv = diag_d.shape
    nmv = [0]
    t_action = [0.0]

    def block(Z):
        """(A - B) on the rows of Z, (m, n_pair), as one batch."""
        t0 = time.time()
        Az, Bz = apply_AB(np.asarray(Z, dtype=float).reshape(-1, no, nv))
        t_action[0] += time.time() - t0
        nmv[0] += Az.shape[0]
        return (Az - Bz).reshape(Az.shape[0], -1)

    def matvec(z):
        return block(z)[0]

    # (A - B) for `solve_symmetric`: one vector, or a batch as columns
    Operator = SimpleNamespace(matvec=matvec,
                               matmat=lambda X: block(np.asarray(X).T).T)

    # A fixed start vector, a function of the data alone, so the probe is
    # reproducible. It must carry weight in every pair irrep:
    # (A - B) is block diagonal over Gamma_i x Gamma_a and a unit vector would
    # confine the Krylov space to one block, returning that block's minimum as
    # if it were the global one. 1/d has both properties and leans on the low
    # pairs.
    cold = 1.0 / diag_d.ravel()
    cold /= np.linalg.norm(cold)
    source = 'davidson-cold'
    diag = diag_d.ravel()

    def lowest(nroots, tol_used, vectors, v0=cold, relative=False):
        """The nroots lowest eigenvalues from the start v0 (with the lowest
        pairs' unit vectors beside it for nroots > 1), to |r| <= tol_used
        Hartree, or with `relative` to |r| <= tol_used |v0^T (A - B) v0| (a
        sign pass: a residual below |theta| is what certifies the sign, so the
        bar follows the value's own size), with their vectors as columns if
        asked for."""
        x0 = np.column_stack([v0] + diagonal_seeds(diag, nroots - 1)
                             if nroots > 1 else [v0])
        if relative:
            tol_used = tol_used * max(abs(float(v0 @ matvec(v0))),
                                      float(np.finfo(float).eps))
        w, x, conv = solve_symmetric(Operator, diag, nroots=nroots, x0=x0,
                                     tol_residual=tol_used,
                                     max_cycle=max(200, diag.size),
                                     warn_unconverged=False,
                                     label='the (A - B) probe')
        if not conv.all():
            raise AmbProbeNotConverged(
                f'the (A - B) Davidson left {int((~conv).sum())} of {conv.size} '
                f'roots above |r| <= {tol_used:g}')
        return (w, x) if vectors else w

    def _record(extra=None):
        if stats is not None:
            stats.update({'probe_matvecs': nmv[0],
                          'probe_action_s': t_action[0],
                          'probe_source': source, **(extra or {})})

    if sign_only and k == 1 and no * nv > 1:
        # The sign is the answer, so stop once it is proven, which a Ritz
        # value alone does not do. Ritz values from a Krylov subspace bracket
        # the spectrum from the inside, so theta >= lambda_min and theta > 0
        # proves nothing on its own. The residual closes it: for a symmetric
        # operator |lambda_i - theta| <= ||r||, so theta - ||r|| > 0 is a
        # certificate, and theta + ||r|| < 0 is the certificate for the other
        # sign. Escalate the tolerance only until one of them holds, which for
        # a value far from zero is the first pass and a fraction of the
        # iterations a converged 'SA' solve would take.
        passes = [(cold, 'davidson-cold', tol_try)
                  for tol_try in (*AMB_PROBE_SIGN_TOLS, tol)]
        if start is not None:
            roots = np.asarray(start, dtype=float).reshape(no * nv, -1)
            # Rayleigh-Ritz on the roots' span, one block action per root.
            Q = np.linalg.qr(roots)[0]
            W = block(Q.T).T                        # (A - B) Q
            s = np.linalg.eigh(0.5 * (Q.T @ W + W.T @ Q))[1][:, 0]
            x, w = Q @ s, W @ s
            theta = (x @ w) / (x @ x)
            cert = np.array([theta, np.linalg.norm(w - theta * x)
                             / np.linalg.norm(x)])
            if replicated:
                cert = lockstep(cert, comm, check=True)   # rank 0's decision
            theta, resid = float(cert[0]), float(cert[1])
            if theta - resid > 0.0 or theta + resid < 0.0:
                source = 'roots'
                _record({'probe_sign_proven': True, 'probe_tol_used': None,
                         'probe_residual': resid})
                return np.array([theta])
            # The first pass starts warm: the lowest root with a
            # PROBE_START_MIX share of 1/d, which reaches the irreps it lacks.
            warm = (roots[:, 0] / np.linalg.norm(roots[:, 0])
                    + PROBE_START_MIX * cold)
            passes[0] = (warm / np.linalg.norm(warm), 'davidson-warm',
                         passes[0][2])
        for v0, source, tol_try in passes:
            w, v = lowest(1, tol_try, True, v0, relative=True)
            theta = float(w[0])
            vec = v[:, 0]
            resid = float(np.linalg.norm(matvec(vec) - theta * vec))
            if theta - resid > 0.0 or theta + resid < 0.0:
                _record({'probe_sign_proven': True, 'probe_tol_used': tol_try,
                         'probe_residual': resid})
                return np.array([theta])
        # No certificate at the tightest tolerance: the value sits inside its
        # own error bar, i.e. near zero. Say so rather than returning
        # a sign the arithmetic does not support.
        _record({'probe_sign_proven': False, 'probe_tol_used': tol,
                 'probe_residual': resid})
        return np.array([theta])

    # A Ritz value is good to |r|^2 / gap, so a residual of tol leaves the
    # value far below it.
    out = np.sort(lowest(min(k, no * nv - 1), tol, False))
    _record()
    return out

