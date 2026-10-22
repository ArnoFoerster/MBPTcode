"""The (A - B) probe: the lowest eigenvalue of A - B, whose sign decides whether the Casida omega^2 reduction is valid.

A symmetric Lanczos on the block action's A - B, ARPACK serially and a
distributed Lanczos over ranks (`_lanczos_lowest`), run cold from 1/d or
started from a Davidson's converged roots (`_lowest_amb_local`).
"""
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from src.Base.constants import (AMB_LANCZOS_MAXITER_PER_DIM, AMB_LANCZOS_NCV,
                                PROBE_START_MIX)
from src.Base.utils.mpi_grid import current_comm, lockstep
from src.SingleReference.LinearResponse.block_actions import _block_action


class AmbProbeNotConverged(RuntimeError):
    """The (A-B) Lanczos (`_lanczos_lowest`) ran out of operator applications
    with a root above its residual bar; ARPACK's serial counterpart is
    `ArpackNoConvergence`."""


def lowest_amb_eigenvalue(lr_solver, nocc, polarizability='BSE', W_aux=None,
                          isdf_factors=None, k=1, tol=1e-6, sign_only=False,
                          stats=None, comm=None):
    """Lowest eigenvalue(s) of (A - B): the sign that decides whether the
    Casida omega^2 reduction is valid at all.

    Matrix-free (a symmetric Lanczos on z -> (A-B)z through the same block
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
    the current `distributed` region's; every rank then runs the same Lanczos
    and returns rank 0's value (`_lowest_amb_from_action`).

    Standalone the probe has no converged roots to try first, so it starts
    cold, from 1/d; `solve_bse_isdf` runs it after its Davidson instead, on
    the Davidson's own action, and under 'sign' tries the roots first.
    """
    comm = current_comm() if comm is None else comm
    apply_AB, diag_d = _block_action(lr_solver, nocc, polarizability, W_aux,
                                     isdf_factors, comm=comm)
    return _lowest_amb_from_action(apply_AB, diag_d, k=k, tol=tol,
                                   sign_only=sign_only, stats=stats, comm=comm)


def _lowest_amb_from_action(apply_AB, diag_d, k=1, tol=1e-6, sign_only=False,
                            stats=None, comm=None, start=None):
    """The (A-B) probe, the same iteration on every rank, rank 0's value on each.

    Serially the Lanczos is ARPACK's. Under any rank count it is
    `_lanczos_lowest` on every rank, over the distributed action's lockstepped
    output: eigsh holds one process-wide lock across its whole iteration,
    matvecs included, so over ranks that are threads of one process
    (`run_simulated`) the first rank inside waits in its matvec's collective
    for a rank that waits for the lock -- and simulated ranks have to run the
    algorithm MPI ranks run. The value is lockstepped at the end;
    ranks whose iterations had fallen apart meet that lockstep against another
    rank's block-action lockstep and raise ValueError together instead of
    deadlocking.

    start: the converged Casida roots' X - Y, (n_pair, nroots) with the
    lowest root first, or one root's (n_pair,), the same on every rank; read
    by the sign-only probe alone (`_lowest_amb_local`).
    """
    replicated = comm is not None and comm.Get_size() > 1
    out = _lowest_amb_local(apply_AB, diag_d, k, tol, sign_only, stats,
                            comm if replicated else None, start)
    return lockstep(out, comm) if replicated else out


def _lowest_amb_local(apply_AB, diag_d, k=1, tol=1e-6, sign_only=False,
                      stats=None, comm=None, start=None):
    """The Lanczos itself: lowest eigenvalue(s) of (A - B), by ARPACK, or by
    `_lanczos_lowest` where the action is distributed over `comm`; the
    sign-only probe after a Davidson tries the converged roots (`start`)
    first.

    What the roots prove: Rayleigh-Ritz of (A - B) on the span of the roots'
    X - Y gives a Ritz value theta and residual r, and |lambda - theta| <= r
    holds for some eigenvalue lambda of (A - B): theta - r > 0 proves that
    one positive. theta is at least the minimum over the span, so the bound
    reaches the minimum only when the span does; a minimum in a pair irrep no
    root lies in is invisible to it. The general probe is the Lanczos from
    1/d, which has weight in every irrep: the escalating passes below and the
    converged probe start there.

    stats receives `probe_matvecs` (every block action the probe applied,
    one per root in the span), `probe_action_s` and `probe_source`: 'roots'
    where the span's certificate held, 'lanczos-warm' for the pass started
    from the lowest root, 'lanczos-cold' for one started from 1/d.
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

    op = LinearOperator((no * nv, no * nv), matvec=matvec, dtype=float)
    # A fixed start vector, not ARPACK's random one: ARPACK draws its start
    # from process-global state, which makes the answer depend on what else
    # in the process has drawn from it; the start must be a function of the
    # data alone for the probe to be reproducible. It must also carry weight
    # in every pair irrep:
    # (A - B) is block diagonal over Gamma_i x Gamma_a and a unit vector would
    # confine the Krylov space to one block, returning that block's minimum as
    # if it were the global one. 1/d has both properties and leans on the low
    # pairs.
    cold = 1.0 / diag_d.ravel()
    cold /= np.linalg.norm(cold)
    source = 'lanczos-cold'

    def lowest(nroots, tol_used, vectors, v0=cold):
        """The nroots lowest Ritz values from the start v0, with their vectors
        as columns if asked for."""
        if not replicated:
            return eigsh(op, k=nroots, which='SA', tol=tol_used, v0=v0,
                         return_eigenvectors=vectors)
        w, x = _lanczos_lowest(matvec, v0, nroots, tol_used)
        return (w, x.T) if vectors else w

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
        passes = [(cold, 'lanczos-cold', tol_try)
                  for tol_try in (1e-2, 1e-3, 1e-4, tol)]
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
            passes[0] = (warm / np.linalg.norm(warm), 'lanczos-warm',
                         passes[0][2])
        for v0, source, tol_try in passes:
            w, v = lowest(1, tol_try, True, v0)
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

    # A Ritz value is good to |r|^2 / gap, so the replicated Lanczos converges
    # to |r| <= sqrt(eps) |theta| rather than stopping at tol: at water/cc-pVDZ
    # BSE@HF that is 88 matvecs where ARPACK takes 81, for a value 3e-15 Ha
    # from the dense eigenvalue where ARPACK's sits 7e-14 from it.
    if replicated:
        tol = min(tol, float(np.sqrt(np.finfo(float).eps)))
    out = np.sort(lowest(min(k, no * nv - 1), tol, False))
    _record()
    return out


def _lanczos_lowest(matvec, v0, k, tol):
    """(theta, x): the k lowest eigenpairs of the symmetric operator
    `matvec`, the vectors as rows, from the start vector `v0`.

    A Lanczos with full reorthogonalization, thick-restarted on the lower
    half of its Ritz vectors whenever the basis reaches eigsh's own size
    (Wu and Simon, SIAM J. Matrix Anal. Appl. 22, 602 (2000)), converged at
    ARPACK's test |r| <= tol |theta|. The Rayleigh quotient is formed from the
    stored products, never from a recurrence, so the iteration is a function
    of the operator's outputs alone and ranks handed the same outputs take the
    same steps; every array is this call's own, so rank threads can run it
    side by side. One start vector spans one direction of a degenerate
    eigenspace, so for k > 1 a repeated eigenvalue can appear once, as in
    ARPACK; the lowest value itself is found either way.
    """
    n = v0.size
    ncv = min(n, max(2 * k + 1, AMB_LANCZOS_NCV))
    V = np.zeros((ncv, n))
    AV = np.zeros((ncv, n))
    q = v0 / np.linalg.norm(v0)
    m = 0
    for _ in range(AMB_LANCZOS_MAXITER_PER_DIM * n):
        V[m] = q
        AV[m] = matvec(V[m])                 # a distributed action locksteps V[m]
        m += 1
        H = V[:m] @ AV[:m].T
        theta, s = np.linalg.eigh(0.5 * (H + H.T))
        nk = min(k, m)
        x = s[:, :nk].T @ V[:m]
        r = s[:, :nk].T @ AV[:m] - theta[:nk, None] * x
        rnorm = np.linalg.norm(r, axis=1)
        if m == n or (m >= k and np.all(rnorm <= tol * np.abs(theta[:k]))):
            return theta[:nk], x
        # In a Lanczos basis every Ritz residual is the same next direction;
        # the longest is its most accurate copy.
        q = r[np.argmax(rnorm)]
        if m == ncv:
            m = max(k, ncv // 2)
            V[:m] = s[:, :m].T @ V
            AV[:m] = s[:, :m].T @ AV
        q -= (V[:m] @ q) @ V[:m]
        q -= (V[:m] @ q) @ V[:m]             # twice is enough (Kahan, Parlett)
        norm = np.linalg.norm(q)
        if norm <= np.finfo(float).eps * np.abs(theta).max():
            return theta[:nk], x             # an invariant subspace
        q /= norm
    raise AmbProbeNotConverged(
        f'the (A - B) Lanczos left roots above |r| <= {tol:g} |theta| after '
        f'{AMB_LANCZOS_MAXITER_PER_DIM * n} operator applications')
