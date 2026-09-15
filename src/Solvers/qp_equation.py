"""Unified dynamical quasiparticle-equation solvers: root finders for f(w) = w - eps - Sigma(w) = 0.

Sigma is any frequency-dependent quantity (GW/PSD self-energy, ADC static
correction, embedding self-energy). Every root search in the codebase goes
through these routines.
"""
import numpy as np
from src.Base.constants import (
    QP_BISECTION_TOL, QP_BISECTION_MAX_ITER,
    QP_NEWTON_TOL, QP_NEWTON_MAX_ITER,
    QP_GRAPHICAL_TOL, QP_GRAPHICAL_N_OMEGA, QP_GRAPHICAL_MAX_BISECTION,
    QP_Z_MIN, QP_Z_DERIV_STEP,
)


def solve_qp_equation(func, e_start, method='pole_strength', **kwargs):
    """Single entry point dispatching to the pole-strength/graphical/newton/
    bisection root finders below. func(w) must return f(w) = w - eps - Sigma(w) (or any
    function whose root is the sought quasiparticle/eigenvalue energy);
    e_start is the zeroth-order guess (e.g. the KS/HF eigenvalue).
    vectorized=True (pole_strength, graphical): func accepts and returns
    arrays; the search grid is one call."""
    if method == 'graphical':
        return solve_qp_equation_graphical(func, e_start, **kwargs)
    if method == 'pole_strength':
        return solve_qp_equation_pole_strength(func, e_start, **kwargs)
    if method == 'newton':
        return solve_qp_equation_newton(func, e_start, **kwargs)
    if method == 'bisection':
        w_min = kwargs.pop('w_min', e_start - 0.5)
        w_max = kwargs.pop('w_max', e_start + 0.5)
        return solve_qp_equation_bisection(func, w_min, w_max, **kwargs)
    raise ValueError(f"Unknown QP-equation method '{method}' "
                     "(expected 'pole_strength', 'graphical', 'newton', "
                     "or 'bisection')")


def solve_qp_equation_bisection(func, w_min, w_max, tol=QP_BISECTION_TOL, max_iter=QP_BISECTION_MAX_ITER):
    """Solve f(w) = w - eps - Sigma(w) = 0 by bisection."""
    a, b = w_min, w_max
    fa = func(a)
    fb = func(b)

    if np.isnan(fa) or np.isnan(fb):
        raise ValueError(f"Endpoints failed to evaluate: f({a})={fa}, f({b})={fb}")

    if fa * fb > 0:
        # Same sign: try expanding the interval
        for _ in range(5):
            a -= 0.5
            b += 0.5
            fa, fb = func(a), func(b)
            if fa * fb < 0:
                break
        else:
            raise ValueError(f"Same sign at endpoints: f({a})={fa}, f({b})={fb}. No root guaranteed.")
            
    for i in range(max_iter):
        c = 0.5 * (a + b)
        fc = func(c)
        
        if abs(fc) < tol or 0.5 * (b - a) < tol:
            return c
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
    return 0.5 * (a + b)

def solve_qp_equation_newton(func, e_start, deriv_func=None, tol=QP_NEWTON_TOL, max_iter=QP_NEWTON_MAX_ITER, damping=1.0):
    """Solve f(w) = w - eps - Sigma(w) = 0 by Newton-Raphson. deriv_func defaults to numerical finite differences."""
    w = e_start
    for i in range(max_iter):
        fw = func(w)
        if abs(fw) < tol:
            return w
            
        if deriv_func is not None:
            dfw = deriv_func(w)
        else:
            h = 1e-5
            dfw = (func(w + h) - func(w - h)) / (2.0 * h)

        if abs(dfw) < 1e-12:
            dfw = 1e-12

        step = fw / dfw
        w_next = w - damping * step

        # Halve damping if the step overshoots (function value grows instead of shrinking).
        if abs(func(w_next)) > 2.0 * abs(fw):
            damping *= 0.5
            w = w - damping * step
        else:
            w = w_next
            damping = min(1.0, damping * 1.5)

    return w


def solve_qp_equation_newton_batch(func, e_start, deriv_func=None, tol=QP_NEWTON_TOL,
                                   max_iter=QP_NEWTON_MAX_ITER, damping=1.0):
    """Vectorized Newton-Raphson for N independent QP equations f_p(w_p)=0, batched so func can use one vectorized self-energy call.

    Same update rule as solve_qp_equation_newton, applied elementwise with
    per-element damping. Returns the converged root for every p.
    """
    w = np.array(e_start, dtype=float, copy=True)
    damp = np.full_like(w, damping)
    for _ in range(max_iter):
        fw = func(w)
        if np.max(np.abs(fw)) < tol:
            return w

        if deriv_func is not None:
            dfw = deriv_func(w)
        else:
            h = 1e-5
            dfw = (func(w + h) - func(w - h)) / (2.0 * h)
        dfw = np.where(np.abs(dfw) < 1e-12, 1e-12, dfw)

        step = fw / dfw
        w_next = w - damp * step
        overshoot = np.abs(func(w_next)) > 2.0 * np.abs(fw)
        damp = np.where(overshoot, damp * 0.5, np.minimum(1.0, damp * 1.5))
        w = np.where(overshoot, w - damp * step, w_next)

    return w


def calculate_z_factor(dsigma_dw):
    """Quasiparticle renormalization Z = 1 / (1 - dSigma/dw)."""
    return 1.0 / (1.0 - dsigma_dw)

def spectral_function(w_grid, eps_hf, sigma_func, eta):
    """Spectral function A(w) = -1/pi * Imag(G(w)), G(w) = 1/(w - eps_hf - Sigma(w))."""
    a_grid = []
    for w in w_grid:
        sig = sigma_func(w)
        denom = w - eps_hf - sig + 1j * eta
        g = 1.0 / denom
        a = -1.0 / np.pi * g.imag
        a_grid.append(a)
    return np.array(a_grid)

def _qp_search_window(eigKS):
    """Grid bounds used by the graphical/pole-strength scans (widened for deep states)."""
    omegaMin = -0.5
    if abs(eigKS) > 1.5:
        omegaMin = -1.0
    if abs(eigKS) > 4.0:
        omegaMin = -2.0
    return omegaMin + eigKS, -omegaMin + eigKS


def _refine_root(func, a, b, tol, max_bisection):
    """Bisect a bracketing interval [a, b] down to `tol`; f(a) is
    evaluated once per move."""
    fa = func(a)
    for _ in range(max_bisection):
        c = 0.5 * (a + b)
        fc = func(c)
        if fa * fc <= 0.0:
            b = c
        else:
            a = c
            fa = fc
        if abs(b - a) <= tol:
            break
    return 0.5 * (a + b)


def pole_strength(func, w, h=QP_Z_DERIV_STEP):
    """Z = 1/(1 - dSigma/dw) at w, for func(w) = w - eps - Sigma(w).

    f'(w) = 1 - dSigma/dw, so Z is just 1/f'(w) and no separate Sigma evaluation
    (or self-energy object) is needed.
    """
    deriv = (func(w + h) - func(w - h)) / (2.0 * h)
    if deriv == 0.0 or not np.isfinite(deriv):
        return np.nan
    return 1.0 / deriv


def _grid_shifts(func, omega_grid, vectorized):
    """Evaluate f on the search grid, scalar loop or one vectorized call.

    Parameters
    ----------
    func : callable
        f(w) = w - eps - Sigma(w); scalar-to-scalar, or (if `vectorized`)
        ndarray-to-ndarray of the same shape as `omega_grid`.
    omega_grid : ndarray, shape (nOmega,)
        Frequencies to evaluate f at.
    vectorized : bool
        If True, call func once on the whole grid; otherwise loop over it.

    Returns
    -------
    shifts : ndarray, shape (nOmega,)
        f(omega_grid), evaluated elementwise.
    """
    if not vectorized:
        return np.array([func(w) for w in omega_grid])
    shifts = np.asarray(func(omega_grid), dtype=float)
    if shifts.shape != omega_grid.shape:
        raise ValueError(
            f'vectorized func must return shape {omega_grid.shape}, '
            f'got {shifts.shape}')
    return shifts


def solve_qp_equation_pole_strength(func, eigKS, tol=QP_GRAPHICAL_TOL,
                                    nOmega=QP_GRAPHICAL_N_OMEGA,
                                    max_bisection=QP_GRAPHICAL_MAX_BISECTION,
                                    z_min=QP_Z_MIN, return_diagnostics=False,
                                    vectorized=False):
    """Solve f(w) = w - eps - Sigma(w) = 0, returning the root with the largest
    quasiparticle weight Z = 1/f'(w) rather than the one nearest eigKS.

    For a shallow state these agree, but a deep valence or semicore state has
    low-weight satellite crossings sitting *between* eps_HF and the true QP pole,
    and the nearest-root rule then returns a satellite whose Z is ~0.03 --
    numerically a root, physically not the quasiparticle.  Roots with Z <= z_min
    or Z > 1 (the latter is unphysical for a Dyson QP equation) are discarded;
    if nothing survives, fall back to the nearest-root answer so the caller
    always gets the previous behaviour rather than an exception.

    return_diagnostics=True additionally returns the full list of
    (root, Z) pairs found, for inspecting the satellite structure.
    """
    omegaMin, omegaMax = _qp_search_window(eigKS)
    omega_grid = np.linspace(omegaMin, omegaMax, nOmega)
    shifts = _grid_shifts(func, omega_grid, vectorized)

    sign_change_idx = np.where(shifts[1:] * shifts[:-1] < 0.0)[0] + 1
    if len(sign_change_idx) == 0:
        root = solve_qp_equation_newton(func, eigKS, tol=tol)
        return (root, []) if return_diagnostics else root

    found = []
    for i in sign_change_idx:
        root = _refine_root(func, omega_grid[i], omega_grid[i - 1], tol, max_bisection)
        found.append((root, pole_strength(func, root)))

    physical = [(r, z) for r, z in found if np.isfinite(z) and z_min < z <= 1.0]
    if physical:
        best = max(physical, key=lambda rz: rz[1])[0]
    else:
        best = min(found, key=lambda rz: abs(rz[0] - eigKS))[0]

    return (best, found) if return_diagnostics else best


def solve_qp_equation_graphical(func, eigKS, tol=QP_GRAPHICAL_TOL,
                                nOmega=QP_GRAPHICAL_N_OMEGA,
                                max_bisection=QP_GRAPHICAL_MAX_BISECTION,
                                vectorized=False):
    """Solve f(w) = w - eps - Sigma(w) = 0 by grid search, then bisect only the sign-changing interval closest to eigKS.

    Only that closest root is ever returned: bisection can't move a root out
    of its grid interval, so the closest-on-the-coarse-grid interval is also
    the one whose refined root is closest -- refining every crossing would
    waste O(max_bisection) evaluations per extra pole.
    """
    omegaMin = -0.5
    if abs(eigKS) > 1.5:
        omegaMin = -1.0
    if abs(eigKS) > 4.0:
        omegaMin = -2.0
    omegaMax = -omegaMin

    omega_grid = np.linspace(omegaMin + eigKS, omegaMax + eigKS, nOmega)
    shifts = _grid_shifts(func, omega_grid, vectorized)

    sign_change_idx = np.where(shifts[1:] * shifts[:-1] < 0.0)[0] + 1
    if len(sign_change_idx) == 0:
        return solve_qp_equation_newton(func, eigKS, tol=tol)

    los = np.minimum(omega_grid[sign_change_idx], omega_grid[sign_change_idx - 1])
    his = np.maximum(omega_grid[sign_change_idx], omega_grid[sign_change_idx - 1])
    # Lower bound on each bracket's possible distance to eigKS (0 if eigKS is
    # inside it); once the best root found beats every remaining bound, stop.
    lower_bound = np.where((eigKS >= los) & (eigKS <= his), 0.0,
                            np.minimum(np.abs(los - eigKS), np.abs(his - eigKS)))
    order = np.argsort(lower_bound)

    best_root, best_dist = None, np.inf
    for k in order:
        if best_dist <= lower_bound[k]:
            break
        i = sign_change_idx[k]
        a, b = omega_grid[i], omega_grid[i - 1]
        for ibisection in range(max_bisection):
            c = (a + b) / 2.0
            shift_a = func(a)
            shift_c = func(c)
            if shift_a * shift_c <= 0.0:
                b = c
            else:
                a = c
            if abs(b - a) <= tol:
                break
        root = (a + b) / 2.0
        d = abs(root - eigKS)
        if d < best_dist:
            best_dist, best_root = d, root
    return best_root
