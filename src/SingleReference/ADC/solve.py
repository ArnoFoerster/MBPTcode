"""Solver machinery shared by every ADC route: root-following Davidson,
dense diagonalization, the downfolded seed builder, and the
Lanczos/continued-fraction spectral solver."""

import numpy as np

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import INV_SQRT_LANCZOS_MAX_STEPS, INV_SQRT_LANCZOS_TOL
from src.Base.utils.linear_algebra.diagonalization import eigh_symmetric
from src.Base.utils.linear_algebra.matrix_functions import eigh_sym, invsqrtm_sym
from src.Solvers.davidson import overlap_pick, solve_symmetric


class ADCNotConverged(RuntimeWarning):
    """An iteration inside an ADC route (CC or ladder amplitudes, a Lanczos
    resolvent or inverse square root) stopped before its tolerance. The last
    iterate is used and the result carries converged=False."""


def koopmans_seed(n, homo_index, v0=None, space='the operator space'):
    """The Davidson guess of an IP/EA root, one length-n vector: v0
    normalized, or the Koopmans unit vector at homo_index (never an n x n
    identity: the operators are matrix-free because n is large). A v0 of
    another length is refused, naming `space`."""
    if v0 is None:
        x0 = np.zeros(n)
        x0[homo_index] = 1.0
        return x0
    x0 = np.asarray(v0, dtype=float)
    if x0.shape[0] != n:
        raise ValueError(f"v0 has length {x0.shape[0]}, expected {n} ({space})")
    return x0 / np.linalg.norm(x0)


def follow_roots(op, diag, x0, nroots=1, tol=1e-8, follow=True, label='ADC Davidson',
                 max_cycle=200, max_subspace=None, max_memory=None, verbose=0):
    """(w, X, conv) of a matrix-free ADC operator by MBPTcode's one symmetric
    Davidson, src.Solvers.davidson.solve_symmetric, started from x0: with
    follow=True the nroots roots whose Ritz vectors overlap most with
    span(x0) (overlap_pick, ties broken by energy), the targeted
    quasiparticle or satellite; with follow=False the nroots lowest. tol is a
    RESIDUAL NORM (the solver's tolerance contract), not pyscf's eigenvalue
    `tol`, from which pyscf derives sqrt(tol). max_cycle, max_subspace (the
    memory knob of the large ADC(4)/(5) operators), max_memory and verbose go
    to solve_symmetric. An unconverged root warns (RuntimeWarning) and comes
    back with conv False; the caller records the flag."""
    if follow and x0 is None:
        raise ValueError("follow_roots(follow=True) needs x0, the guess to follow")
    pick = overlap_pick(x0) if follow else None
    return solve_symmetric(op, diag, nroots=nroots, x0=x0, pick=pick,
                           tol_residual=tol, label=label, max_cycle=max_cycle,
                           max_subspace=max_subspace, max_memory=max_memory,
                           verbose=verbose)


def davidson_follow(aop_vec, diag, nH, norb, homo_index, ref_vec, nroots,
                    tol=1e-6, max_cycle=200, max_subspace=40, verbose=0):
    """Root-following Davidson (follow_roots) from ref_vec, or the Koopmans
    unit vector at homo_index (koopmans_seed). Returns (eGF, Z, Reigv, conv)
    sorted ascending in eGF, Z the orbital-block weight of each root."""
    x0 = koopmans_seed(nH, homo_index, ref_vec)
    eGF, Reigv, conv = follow_roots(aop_vec, diag, x0, nroots=nroots, tol=tol,
                                    max_cycle=max_cycle, max_subspace=max_subspace,
                                    verbose=verbose)
    Z = np.sum(Reigv[:norb, :] ** 2, axis=0)
    return eGF, Z, Reigv, conv


def solve_handwritten(s, static_correction, nroots, homo_index, ref_vec, tol,
                      threshold, verbose, max_cycle):
    """The hand-written routes of either solver class: dense eigh
    (benchmarking) or root-following Davidson on s.handwritten_operator
    (production). Returns (e, Z); vectors and the convergence flag 'conv' on
    s.last_result."""
    nocc = s.nocc
    if not s.matrix_free:
        H = s.build_supermatrix(nocc, static_correction)
        e, Z, vec = diag_dense(H, s.norb, threshold=threshold)
        # dense: eigh either returns every root or raises, so this one
        # really is all-True rather than a stand-in for an unread flag
        s.last_result = {'vec': vec, 'conv': np.ones_like(e, dtype=bool)}
        return e, Z
    aop, diag, nH = s.handwritten_operator(nocc, static_correction)
    homo = homo_index if homo_index is not None else nocc - 1
    e, Z, vec, conv = davidson_follow(aop, diag, nH, s.norb, homo, ref_vec, nroots,
                                      tol=tol, max_cycle=max_cycle, verbose=verbose)
    # the solver's own flag, not a placeholder: a Davidson that ran out of
    # cycles still returns an energy for every root it was asked for
    s.last_result = {'vec': vec, 'conv': conv}
    return e, Z


def downfolded_seed_vectors(aop_vec, diag, nH, norb, orbital_window, omega0,
                            eta=0.01):
    """Mixed-orbital Davidson seeds from a one-shot (on-shell) static
    downfold of the supermatrix: H_eff = F_ww + U_w (omega0 - K_d)^{-1} U_w^T
    over the orbitals in orbital_window, with one aop_vec matvec per window
    orbital supplying the F/U columns and K approximated by its diagonal
    (`diag`, the Davidson preconditioner). The orbital block is NOT diagonal-
    approximated -- the off-diagonal Sigma_pq(omega0) elements are what rotate
    near-degenerate orbitals (pi/pi*) into mixed quasiparticle seeds that a
    single Koopmans unit vector cannot represent. eta (Ha) regularizes
    near-resonant K denominators: 1/(w-K) -> (w-K)/((w-K)^2 + eta^2).

    Returns (e_eff, seeds): e_eff ascending (nw,), seeds (nH, nw) unit
    vectors carrying the H_eff eigenvectors on their orbital-window rows
    (zeros elsewhere) -- ref_vec candidates for davidson_follow."""
    window = np.asarray(orbital_window, dtype=int)
    cols = np.empty((window.size, nH))
    for k, p in enumerate(window):
        e_p = np.zeros(nH)
        e_p[p] = 1.0
        cols[k] = aop_vec(e_p)
    U_w = cols[:, norb:]
    d = omega0 - diag[norb:]
    H_eff = cols[:, window] + (U_w * (d / (d * d + eta * eta))) @ U_w.T
    H_eff = 0.5 * (H_eff + H_eff.T)
    e_eff, C = np.linalg.eigh(H_eff)
    seeds = np.zeros((nH, window.size))
    seeds[window] = C
    return e_eff, seeds


def diag_dense(H, norb, threshold=5000):
    """Dense diagonalization on the shared backend (ELPA-distributed above
    threshold); (eGF, Z, Reigv) sorted ascending. Only rank 0 gets Reigv/Z
    when the solve was distributed."""
    eGF, Reigv, _ = eigh_symmetric(H, threshold=threshold)
    if Reigv is None:
        return eGF, None, None      # not this rank's eigenvectors
    Z = np.sum(Reigv[:norb, :] ** 2, axis=0)
    order = np.argsort(eGF)
    return eGF[order], Z[order], Reigv[:, order]


# ---- Lanczos/continued-fraction spectral solver ----
class LanczosState:
    """Incremental Lanczos state so lanczos_spectral can extend the
    tridiagonalization in blocks instead of restarting from scratch each
    time it checks convergence. keep_vectors=False stores no Krylov basis
    (V is None): only the tridiagonal, for runs that never reorthogonalize
    or form a Ritz vector."""
    __slots__ = ('V', 'a', 'b', 'v_prev', 'v_curr', 'beta', 'breakdown')

    def __init__(self, v0, keep_vectors=True):
        nrm0 = np.linalg.norm(v0)
        if nrm0 < 1e-300:
            raise ValueError("lanczos_spectral: zero starting vector")
        self.V = [] if keep_vectors else None
        self.a = []
        self.b = []          # b[k-1] = beta_k, the k-1<->k off-diagonal
        self.v_prev = np.zeros_like(v0)
        self.v_curr = v0 / nrm0
        self.beta = 0.0
        self.breakdown = False

    def arrays(self):
        a = np.asarray(self.a, float)
        b = np.concatenate([[0.0], np.asarray(self.b, float)])
        return a, b


def lanczos_extend(matvec, state, nsteps, reorth=True):
    """Run up to `nsteps` more Lanczos iterations on `state` in place (fewer
    if an invariant subspace is hit -- state.breakdown is then set True)."""
    if reorth and state.V is None:
        raise ValueError("lanczos_extend: reorth needs a state that keeps "
                         "its vectors")
    for _ in range(nsteps):
        if state.breakdown:
            break
        if state.V is not None:
            state.V.append(state.v_curr)
        w = matvec(state.v_curr)
        alpha = float(np.dot(state.v_curr, w))
        state.a.append(alpha)
        w = w - alpha * state.v_curr - state.beta * state.v_prev
        if reorth:
            Vmat = np.asarray(state.V)
            w -= Vmat.T @ (Vmat @ w)
        beta = np.linalg.norm(w)
        if beta < 1e-12:
            state.breakdown = True
            break
        state.v_prev = state.v_curr
        state.v_curr = w / beta
        state.b.append(beta)
        state.beta = beta
    return state


def apply_inverse_sqrt(matvec, v, tol=INV_SQRT_LANCZOS_TOL,
                       max_steps=INV_SQRT_LANCZOS_MAX_STEPS):
    """N^{-1/2} v for a symmetric positive-definite N given only as a
    matvec: Lanczos on N from v (lanczos_extend, reorthogonalized),
    N^{-1/2} v ~ |v| V_k f(T_k) e_1 with f(x) = x^{-1/2}, extended until two
    successive approximations differ by less than tol*|v|. For the Faddeev
    metric N = 1 + dN, dN small, a handful of steps suffice.

    Raises when a Ritz value of N is not positive (N indefinite: the input is
    not a metric). An approximation that has not settled in max_steps is a
    numerical shortfall: it warns (ADCNotConverged) and returns the last
    approximation."""
    v = np.asarray(v, dtype=float)
    vnorm = np.linalg.norm(v)
    if vnorm == 0.0:
        return np.zeros_like(v)
    state = LanczosState(v)
    prev = None
    for _ in range(max_steps):
        lanczos_extend(matvec, state, 1)
        a = np.asarray(state.a)
        b = np.asarray(state.b[:len(a) - 1])
        th, S = np.linalg.eigh(np.diag(a) + np.diag(b, 1) + np.diag(b, -1))
        if th.min() <= 0.0:
            raise np.linalg.LinAlgError(
                f"apply_inverse_sqrt: N is not positive definite (Ritz value "
                f"{th.min():.3e})")
        cur = np.asarray(state.V).T @ (vnorm * (S @ (S[0, :] / np.sqrt(th))))
        if state.breakdown or (prev is not None
                               and np.linalg.norm(cur - prev) < tol * vnorm):
            return cur
        prev = cur
    flag('inverse_sqrt_not_settled', f"apply_inverse_sqrt: not settled to {tol:g} in {max_steps} Lanczos "
                  "steps; the last approximation is returned", ADCNotConverged, stacklevel=2)
    return cur


def inverse_sqrt_psd(N, what):
    """N^{-1/2} of a dense symmetric positive-definite N
    (Base.utils.linearAlgebra.matrix_functions.invsqrtm_sym); raises
    LinAlgError for an indefinite N, naming it by `what`."""
    w, V = eigh_sym(N)
    if w.min() <= 0.0:
        raise np.linalg.LinAlgError(f"{what} is not positive definite (min "
                                    f"eigenvalue {w.min():.3e})")
    return invsqrtm_sym(N, eig=(w, V))


def _continued_fraction(omega, a, b, eta):
    """G(omega+i*eta) via the Lanczos continued fraction, evaluated
    bottom-up. a: (n,) diagonal (alpha_k). b: (n,) off-diagonal (beta_k),
    b[0] unused/0 by convention -- matches LanczosState.arrays()."""
    z = np.asarray(omega, dtype=complex) + 1j * eta
    n = len(a)
    cf = 1.0 / (z - a[n - 1])
    for k in range(n - 2, -1, -1):
        cf = 1.0 / (z - a[k] - b[k + 1] ** 2 * cf)
    return cf


def lanczos_spectral(A, diag, v0, omega_range, eta=None, npts=2000,
                     min_steps=40, max_steps=800, step_block=40,
                     peak_tol=1e-3, z_threshold=1e-5, reorth=True):
    """Matrix-free Lanczos/continued-fraction spectral solver -- an
    alternative to davidson_follow() that computes the FULL spectral function
    for a
    single starting channel v0 (main peak + every satellite) in ONE pass,
    converged over a requested frequency window, instead of root-following
    one state at a time. No seed-quality dependence, no energy-window
    guessing for individual roots -- the window IS the input.

    A: scipy LinearOperator or a callable matvec (n-vector -> n-vector) --
        SAME convention as davidson_follow()'s operator. Reused as-is across
        every ADC level/DF configuration: whatever (op, diag) pair the
        adc_r_sigma_* / adc_u_sigma_* build_operator functions produce works
        unchanged here (DF dispatch happens inside those builders, not here
        -- this function has no level/DF awareness at all).
    diag: (n,) diagonal -- SAME convention as solve_symmetric(); not used
        by Lanczos itself (no preconditioner needed), kept purely so callers
        can build (op, diag) once and pass the identical pair to either
        solver.
    v0: (n,) starting vector. PHYSICALLY MEANINGFUL here, unlike the
        Davidson's x0 (a root-following seed/guess): G_v0(omega) = <v0|(omega-A)^-1|v0>
        is the EXACT spectral function for whatever channel v0 represents,
        e.g. a unit vector on one orbital row = that orbital's complete
        removal spectrum (main peak + satellites), or any other physically
        motivated combination.
    omega_range: (omega_lo, omega_hi), SAME sign/unit convention as diag/the
        operator's eigenvalues elsewhere in this module (Hartree; negative
        for IP removal energies) -- the window to converge the continued
        fraction over.
    eta: broadening (Ha) for the continued-fraction spectral density;
        default 3x the omega_range grid spacing so peaks are resolved
        without being needlessly sharp (and without demanding pathologically
        many Lanczos steps to resolve a delta function).
    npts: frequency-grid resolution within omega_range.
    min_steps/max_steps/step_block: Lanczos is extended incrementally in
        blocks of step_block steps; the spectral function on omega_range is
        recomputed after each block and compared to the previous block (max
        abs diff, relative to the running peak height) -- stops when that
        stabilizes below peak_tol, an invariant subspace is hit
        (state.breakdown -- Lanczos has then captured everything reachable
        from v0 exactly), or max_steps is reached.
    z_threshold: peaks below this (eta-regularized weight estimate) are
        dropped from the returned peak list.
    reorth: full reorthogonalization (default True; the O(nsteps^2 * n) cost
        is fine at the k<=1/full-manifold operator sizes this targets).

    Returns a dict:
      'omega': (npts,) frequency grid
      'spectral': (npts,) A(omega) = -Im[G(omega+i*eta)]/pi
      'peak_omega', 'peak_weight': (npeaks,) local maxima above
          z_threshold; weight estimated as A(peak)*pi*eta (exact for an
          isolated simple pole broadened by eta, approximate if peaks
          overlap -- resolve overlapping peaks by shrinking eta/widening
          npts, not by trusting this estimate blindly)
      'nsteps': Lanczos steps actually run
      'converged': whether step-to-step stability (or exact breakdown) was
          reached before max_steps
      'a', 'b': the raw Lanczos coefficients (re-evaluate
          _continued_fraction at any other omega/eta without rerunning
          Lanczos -- b padded with a leading 0, matching _continued_fraction)
    """
    matvec = A.matvec if hasattr(A, 'matvec') else A
    v0 = np.asarray(v0, float)
    omega_lo, omega_hi = omega_range
    if omega_hi <= omega_lo:
        raise ValueError(f"omega_range=({omega_lo},{omega_hi}): need hi > lo")
    omega_grid = np.linspace(omega_lo, omega_hi, npts)
    if eta is None:
        eta = 3.0 * (omega_hi - omega_lo) / npts

    state = LanczosState(v0)
    prev_spectral = None
    converged = False
    while len(state.a) < max_steps:
        block = min(step_block, max_steps - len(state.a))
        lanczos_extend(matvec, state, block, reorth=reorth)
        a, b = state.arrays()
        spectral = -_continued_fraction(omega_grid, a, b, eta).imag / np.pi
        if state.breakdown:
            converged = True
            prev_spectral = spectral
            break
        if len(a) >= min_steps and prev_spectral is not None:
            scale = max(spectral.max(), 1e-300)
            if np.max(np.abs(spectral - prev_spectral)) < peak_tol * scale:
                converged = True
                prev_spectral = spectral
                break
        prev_spectral = spectral

    spectral = prev_spectral
    a, b = state.arrays()
    peak_idx = [i for i in range(1, len(spectral) - 1)
               if spectral[i] > spectral[i - 1] and spectral[i] > spectral[i + 1]]
    peak_weight = np.array([spectral[i] * np.pi * eta for i in peak_idx])
    peak_omega = np.array([omega_grid[i] for i in peak_idx])
    keep = peak_weight > z_threshold
    order = np.argsort(peak_omega[keep])
    return {
        'omega': omega_grid,
        'spectral': spectral,
        'peak_omega': peak_omega[keep][order],
        'peak_weight': peak_weight[keep][order],
        'nsteps': len(a),
        'converged': converged,
        'a': a, 'b': b, 'eta': eta,
    }

