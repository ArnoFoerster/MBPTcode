"""Solver machinery shared by every ADC route: root-following Davidson,
dense diagonalization, the downfolded/satellite seed builders, and the
Lanczos/continued-fraction spectral solver."""
import numpy as np
from pyscf import lib as pyscf_lib

from src.Base.utils.linearAlgebra.diagonalization import diagonalize_matrix

def davidson_follow(aop_vec, diag, nH, norb, homo_index, ref_vec, nroots,
                    conv_tol=1e-6, max_cycle=100, max_space=30, verbose=0):
    """Root-following Davidson on a matrix-free operator: Ritz pairs are
    picked by |overlap| with the reference vector (Koopmans unit vector at
    homo_index unless ref_vec is given). Returns (eGF, Z, Reigv) sorted
    ascending in eGF."""
    if ref_vec is None:
        x0 = np.zeros(nH)
        x0[homo_index] = 1.0
    else:
        x0 = np.asarray(ref_vec, dtype=float)
        x0 = x0 / np.linalg.norm(x0)
    ref = x0.copy()

    def precond(dx, e, x0_):
        d = diag - e
        d[np.abs(d) < 1e-8] = 1e-8
        return dx / d

    def pick(w, v, nroots_, envs):
        xs = envs['xs']
        ref_coeff = np.array([np.dot(ref, x) for x in xs])
        overlap = np.abs(ref_coeff @ v)
        idx = np.argsort(-overlap)[:nroots_]
        order = idx[np.argsort(w[idx])]
        return w[order], v[:, order], order

    def aop(xs):
        return [aop_vec(x) for x in xs]

    conv, e, c = pyscf_lib.davidson1(
        aop, x0, precond, nroots=nroots, pick=pick,
        tol=conv_tol, max_cycle=max_cycle, max_space=max_space,
        verbose=verbose)

    e = np.atleast_1d(np.asarray(e, dtype=float))
    c = np.asarray(c)
    if c.ndim == 1:
        c = c[None, :]
    order = np.argsort(e)
    eGF = e[order]
    Reigv = c[order].T
    Z = np.sum(Reigv[:norb, :] ** 2, axis=0)
    return eGF, Z, Reigv


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


def select_resonant_configs(diag, norb, omega0, tol=0.05, max_candidates=20):
    """2h1p/2p1h configuration indices (>= norb) whose zeroth-order (diagonal
    K) energy lies within `tol` Ha of `omega0` -- candidate host
    configurations for a satellite at that energy, for satellite_seed_vectors.
    Same units/convention as downfolded_seed_vectors's omega0 (Hartree,
    compared directly against `diag`). Returns indices sorted by |diag-omega0|
    ascending, capped at max_candidates."""
    idx = np.arange(norb, diag.size)
    dist = np.abs(diag[idx] - omega0)
    order = np.argsort(dist)
    sel = idx[order][dist[order] < tol]
    return sel[:max_candidates]


def satellite_seed_vectors(aop_vec, diag, nH, norb, orbital_window,
                           satellite_indices, omega0, eta=0.01):
    """Mixed-orbital + 2h1p/2p1h Davidson seeds for SATELLITE root-following.

    downfolded_seed_vectors folds the *entire* 2h1p+ manifold into an
    effective correction on the orbital block, so its seeds carry zero
    weight beyond the window rows by construction -- useless for a satellite,
    whose defining feature is non-negligible weight surviving on a 2h1p+ row.
    This variant instead keeps a chosen RESONANT subset of 2h1p+
    configurations (satellite_indices, e.g. from select_resonant_configs)
    fully explicit -- only the remaining, off-resonant continuum gets folded
    away, exactly as before. Diagonalizing the resulting small
    (window + satellite_indices)-dimensional block can therefore produce
    genuinely mixed 1h/2h1p eigenvectors: the renormalized-QP root(s) stay
    ~100% on the window rows, while the satellite root picks up real weight
    on the satellite_indices rows -- use dominant_satellite_seed to select it.

    Returns (e_eff, seeds, full_idx): e_eff ascending (m,) where
    m = len(orbital_window) + len(satellite_indices); seeds (nH, m) unit
    vectors carrying the H_block eigenvectors on the window+satellite rows
    (zeros elsewhere); full_idx (m,) = concatenate([orbital_window,
    satellite_indices]), the row order seeds/e_eff are indexed by."""
    window = np.asarray(orbital_window, dtype=int)
    sat = np.asarray(satellite_indices, dtype=int)
    full_idx = np.concatenate([window, sat])
    nw = window.size

    cols = np.empty((full_idx.size, nH))
    for k, p in enumerate(full_idx):
        e_p = np.zeros(nH)
        e_p[p] = 1.0
        cols[k] = aop_vec(e_p)

    # off-resonant remainder: the 2h1p+ block minus the explicit satellite set
    remainder_mask = np.ones(nH - norb, dtype=bool)
    remainder_mask[sat - norb] = False
    R = np.arange(norb, nH)[remainder_mask]

    U_w = cols[:nw, R]  # window -> off-resonant coupling, same as
                        # downfolded_seed_vectors's U_w but with R excluding
                        # the satellite set
    d = omega0 - diag[R]
    fold = (U_w * (d / (d * d + eta * eta))) @ U_w.T  # (nw, nw)

    H_block = cols[:, full_idx]  # direct (m, m): window+satellite, undownfolded
    H_block[:nw, :nw] += fold    # only the window-window sub-block is corrected
    H_block = 0.5 * (H_block + H_block.T)

    e_eff, C = np.linalg.eigh(H_block)
    seeds = np.zeros((nH, full_idx.size))
    seeds[full_idx] = C
    return e_eff, seeds, full_idx


def dominant_satellite_seed(e_eff, seeds, full_idx, window_size):
    """From satellite_seed_vectors' output, pick the eigenvector with the
    largest weight on the explicit satellite rows (full_idx[window_size:])
    -- the one to feed as ref_vec into davidson_follow so it tracks the
    satellite instead of re-locking onto a renormalized-QP root. Returns
    (e_guess, seed_vec)."""
    sat_rows = full_idx[window_size:]
    weight_on_sat = np.sum(seeds[sat_rows, :] ** 2, axis=0)
    k = np.argmax(weight_on_sat)
    return e_eff[k], seeds[:, k]


def diag_dense(H, norb, threshold=5000):
    """Dense diagonalization (scalapack-distributed above threshold);
    (eGF, Z, Reigv) sorted ascending. Only rank 0 gets Reigv/Z distributed."""
    eGF, Reigv, is_distributed, solver, comm = diagonalize_matrix(H, threshold=threshold)
    if is_distributed:
        # Reigv came back whole on rank 0; the other ranks are done.
        solver.destroy()
        if comm.Get_rank() != 0:
            return eGF, None, None
    Z = np.sum(Reigv[:norb, :] ** 2, axis=0)
    order = np.argsort(eGF)
    return eGF[order], Z[order], Reigv[:, order]


# ---- Lanczos/continued-fraction spectral solver ----
class _LanczosState:
    """Incremental Lanczos state so lanczos_spectral can extend the
    tridiagonalization in blocks instead of restarting from scratch each
    time it checks convergence."""
    __slots__ = ('V', 'a', 'b', 'v_prev', 'v_curr', 'beta', 'breakdown')

    def __init__(self, v0):
        nrm0 = np.linalg.norm(v0)
        if nrm0 < 1e-300:
            raise ValueError("lanczos_spectral: zero starting vector")
        self.V = []
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


def _lanczos_extend(matvec, state, nsteps, reorth=True):
    """Run up to `nsteps` more Lanczos iterations on `state` in place (fewer
    if an invariant subspace is hit -- state.breakdown is then set True)."""
    for _ in range(nsteps):
        if state.breakdown:
            break
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


def _continued_fraction(omega, a, b, eta):
    """G(omega+i*eta) via the Lanczos continued fraction, evaluated
    bottom-up. a: (n,) diagonal (alpha_k). b: (n,) off-diagonal (beta_k),
    b[0] unused/0 by convention -- matches _LanczosState.arrays()."""
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
    alternative to davidson() that computes the FULL spectral function for a
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
    diag: (n,) diagonal -- SAME convention as davidson(); not used by
        Lanczos itself (no preconditioner needed), kept purely so callers
        can build (op, diag) once and pass the identical pair to either
        solver.
    v0: (n,) starting vector. PHYSICALLY MEANINGFUL here, unlike davidson's
        v0 (a root-following seed/guess): G_v0(omega) = <v0|(omega-A)^-1|v0>
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

    state = _LanczosState(v0)
    prev_spectral = None
    converged = False
    while len(state.a) < max_steps:
        block = min(step_block, max_steps - len(state.a))
        _lanczos_extend(matvec, state, block, reorth=reorth)
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

