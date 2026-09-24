"""Symmetric Davidson-Liu eigensolver: the k lowest eigenpairs of a matrix-free
operator, or (follow=True) the k Ritz vectors closest to a reference guess.

Distinct from ADC's own solve.py::davidson_follow, which is built on pyscf's
davidson1 and root-follows a single state (the HOMO IP/EA) by overlap with a
reference vector.
"""
import warnings

import numpy as np


def davidson(A, diag, k=1, v0=None, follow=False, tol=1e-8,
             max_iter=200, max_subspace=None, return_vectors=True):
    """Symmetric Davidson-Liu eigensolver with a K-diagonal preconditioner and
    a Jacobi-Davidson projected correction.

    A: a scipy LinearOperator or a callable matvec (n-vector -> n-vector).
    diag: (n,) the operator's (approximate) diagonal -- the zeroth-order K
        diagonal -- used BOTH for the preconditioner (t <- r/(diag-theta)) and,
        when v0 is None, for the starting guesses (unit vectors on the lowest
        diagonal entries).
    k: number of eigenpairs wanted.
    v0: optional (n,) or (n, m) initial guess column(s); e.g. a unit vector on
        a target orbital to seed / follow a specific (HOMO) root.
    follow: if True (needs v0), select at each step the k Ritz vectors with the
        largest overlap onto the ORIGINAL guess space span(v0) -- state /
        root-following, for targeting a specific near-extreme root (the HOMO IP
        the dynamical ADC(4) correction wants) rather than the algebraically
        lowest. If False, the k LOWEST-algebraic eigenpairs (lowest IP/EA roots).
        (Robust interior targeting of a blind energy shift is NOT provided --
        these operators have clustered/degenerate interior spectra; use follow
        with a physically-motivated guess instead.)
    tol: residual-norm convergence threshold (all k roots). Roots still above
        it when the iterations end are named in a RuntimeWarning.
    Returns (w,) or (w, X): eigenvalues ascending and, if return_vectors, the
        (n, k) eigenvectors.
    """
    matvec = A.matvec if hasattr(A, 'matvec') else A
    diag = np.asarray(diag, float)
    n = diag.size
    k = min(k, n)
    if max_subspace is None:
        max_subspace = min(n, max(6 * k + 20, 40))
    max_subspace = min(max_subspace, n)
    restart_keep = min(2 * k + 2, max_subspace)   # Ritz vectors kept at restart

    # ---- initial subspace (seed richer than k so degenerate/clustered lowest
    #      roots are all captured; a bare-k seed can collapse a degenerate pair) --
    want = min(n, 2 * k + 4)
    if v0 is None:
        cols = np.argsort(diag)[:want]
        V = np.zeros((n, len(cols)))
        V[cols, np.arange(len(cols))] = 1.0
        V, _ = np.linalg.qr(V)
        Q0 = None
    else:
        V = np.asarray(v0, float)
        V = V.reshape(n, -1) if V.ndim > 1 else V.reshape(n, 1)
        V, _ = np.linalg.qr(V)
        Q0 = V.copy() if follow else None   # frozen guess space (before padding)
        if V.shape[1] < want:               # pad with lowest-diagonal seeds so
            cols = np.argsort(diag)[:want]  # the subspace has >= k start vectors
            seeds = np.zeros((n, len(cols)))
            seeds[cols, np.arange(len(cols))] = 1.0
            V = np.linalg.qr(np.column_stack([V, seeds]))[0][:, :want]
    AV = np.column_stack([matvec(V[:, j]) for j in range(V.shape[1])])

    theta = np.zeros(k)
    X = V[:, :k].copy()
    rnorm = np.full(k, np.inf)
    for _ in range(max_iter):
        H = V.T @ AV
        w, S = np.linalg.eigh(0.5 * (H + H.T))
        if follow:
            # order Ritz vectors by weight in the frozen guess space
            sel = np.argsort(-np.linalg.norm(Q0.T @ (V @ S), axis=0))
        else:
            sel = np.argsort(w)                       # lowest-algebraic
        idx = sel[:k]
        theta, Y = w[idx], S[:, idx]
        X = V @ Y
        AX = AV @ Y
        R = AX - X * theta[None, :]
        rnorm = np.linalg.norm(R, axis=0)
        if np.all(rnorm < tol):
            break

        # ---- Jacobi-Davidson preconditioned corrections ----
        new = []
        for i in range(k):
            if rnorm[i] < tol:
                continue
            denom = diag - theta[i]
            denom[np.abs(denom) < 1e-8] = 1e-8
            # precondition AND project orthogonal to the Ritz vector u:
            #   t = K^-1 r - alpha K^-1 u,  alpha = (u.K^-1 r)/(u.K^-1 u).
            Kr = R[:, i] / denom
            u = X[:, i]
            Ku = u / denom
            uKu = u @ Ku
            if abs(uKu) > 1e-300:
                Kr = Kr - ((u @ Kr) / uKu) * Ku
            # normalised first, so the 1e-9 below tests linear dependence, not size:
            # a large diagonal shrinks Kr under it while r is still above tol
            t = Kr / np.linalg.norm(Kr)
            # double modified Gram-Schmidt against subspace + accepted dirs
            for _ in range(2):
                t -= V @ (V.T @ t)
                if new:
                    Nd = np.column_stack(new)
                    t -= Nd @ (Nd.T @ t)
            nrm = np.linalg.norm(t)
            if nrm > 1e-9:
                new.append(t / nrm)
        if not new:
            break

        if V.shape[1] + len(new) > max_subspace:
            # restart: collapse to the best `restart_keep` Ritz vectors (not
            # just the k wanted -- keeping near-degenerate/clustered partners is
            # what lets the lowest roots converge). Exact + cheap: S is
            # orthonormal so V@S[:,keep] is orthonormal and AV@S[:,keep] is its
            # true image, no matvec recompute.
            keep_idx = sel[:restart_keep]
            V = V @ S[:, keep_idx]
            AV = AV @ S[:, keep_idx]
            keep = []
            for t in new:
                for _ in range(2):
                    t -= V @ (V.T @ t)
                    if keep:
                        Nd = np.column_stack(keep)
                        t -= Nd @ (Nd.T @ t)
                if np.linalg.norm(t) > 1e-9:
                    keep.append(t / np.linalg.norm(t))
            new = keep
            if not new:
                continue
        Nd = np.column_stack(new)
        AN = np.column_stack([matvec(Nd[:, j]) for j in range(Nd.shape[1])])
        V = np.column_stack([V, Nd])
        AV = np.column_stack([AV, AN])

    bad = np.flatnonzero(rnorm >= tol)
    if bad.size:
        # an unconverged Ritz pair is otherwise indistinguishable from a
        # converged one to the caller
        warnings.warn(
            f"davidson: {bad.size} of {k} roots still above tol={tol:g} when "
            f"the iterations stopped (max_iter={max_iter}): " + ", ".join(
                f"{theta[i]:.8g} (residual {rnorm[i]:.2e})" for i in bad),
            RuntimeWarning, stacklevel=2)
    order = np.argsort(theta)
    theta, X = theta[order], X[:, order]
    return (theta, X) if return_vectors else theta
