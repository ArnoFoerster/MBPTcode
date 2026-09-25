"""Unified dynamical quasiparticle-equation solvers: root finders for f(w) = w - eps - Sigma(w) = 0.

Sigma is any frequency-dependent quantity (GW/PSD self-energy, ADC static
correction, embedding self-energy). Every root search in the codebase goes
through these routines.
"""
import warnings

import numpy as np
from src.Base.constants import (
    QP_BISECTION_TOL, QP_BISECTION_MAX_ITER,
    QP_CD_NEWTON_MAX_ITER, QP_CD_NEWTON_TOL,
    QP_NEWTON_TOL, QP_NEWTON_MAX_ITER,
    QP_GRAPHICAL_TOL, QP_GRAPHICAL_N_OMEGA, QP_GRAPHICAL_MAX_BISECTION,
    QP_POLE_OFFSET, QP_POLE_OFFSET_MIN, QP_POLE_STRENGTH_MIN,
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


def solve_qp_equation_newton_guarded(sigma, slope, eps, p, nocc,
                                     xc_correction=0.0, w0=None,
                                     tol=QP_CD_NEWTON_TOL,
                                     max_iter=QP_CD_NEWTON_MAX_ITER,
                                     pole_offset=QP_POLE_OFFSET,
                                     relax_offset=True,
                                     linearize_on_capture=True, guard_out=None,
                                     offset_min=QP_POLE_OFFSET_MIN,
                                     z_min=QP_POLE_STRENGTH_MIN,
                                     linear_offset=QP_POLE_OFFSET):
    """(w, Z) of w = eps_p + xc_correction + Sigma(w), Newton HELD OFF THE POLES.

    f(w) = w - eps_p - xc_correction - Sigma(w) has a pole at every orbital
    energy, because the self-energy does: a quasiparticle route that deforms
    the frequency contour (`GW.contour_deformation`) puts the pole of G at
    omega = eps_q ON the contour, where the imaginary-axis integrand collapses
    onto nu = 0 and no quadrature resolves it. The iterate is therefore kept
    `pole_offset` away from every eps_q, and the three things that can go wrong
    around such a pole are each answered here rather than returned as a number.

    sigma, slope : callbacks Sigma(w) and dSigma/dw at fixed everything else.
                   Each decides its own residue set, so the branch of Sigma
                   being solved is the one the caller's route defines.
    xc_correction: <Sigma_x - v_xc>, zero on a Hartree-Fock reference.
    w0 :           the start; the default eps_p pushed `pole_offset` toward the
                   chemical potential, the direction the correction takes, since
                   eps_p is itself exactly on a pole.
    relax_offset:  whether a root INSIDE the guard may shrink it. The guard and
                   the Newton step then fight -- the step carries the iterate
                   off the guard and the guard puts it back, forever -- and the
                   tell is the same pushed value twice. True halves the band
                   toward `offset_min`, which is decided per call and so per
                   geometry; False holds a band the caller froze and warns
                   before leaving it, so a surface is differentiated along one
                   Newton path rather than whichever one each geometry took.
    linearize_on_capture:
                   what to do when the iteration ends on a POLE instead of a
                   root. Newton near a pole takes ever smaller steps and
                   converges onto it, and the pole strength there goes to zero
                   because dSigma/dw diverges. Two fallbacks share this flag:
                   an iterate pinned against q != p, and a converged root with
                   Z < `z_min`. Both are replaced by ONE Newton step from a
                   point a full `linear_offset` away from every pole, which
                   cannot be captured -- a different approximation from the
                   self-consistent root, and it says so. False refuses instead.
    guard_out:     a dict, if given, receives 'pole_offset', the band actually
                   in force -- what a caller has to record to pin this path.

    Z = 1/(1 - dSigma/dw) comes from the same slope as the step, so it is the
    exact derivative of the equation that was solved and not a difference of it.
    """
    w = float(eps[p] + (pole_offset if p < nocc else -pole_offset)
              if w0 is None else w0)
    offset = float(pole_offset)
    frozen = not relax_offset
    if guard_out is not None:
        guard_out['pole_offset'] = offset
    last_push = None
    blocker = None
    was_pinned = False
    for _ in range(max_iter):
        gaps = np.abs(eps - w)
        pinned = gaps.min() < offset
        if pinned:
            q = int(np.argmin(gaps))
            w_push = float(eps[q] + np.sign(w - eps[q] or 1.0) * offset)
            # A root INSIDE the band makes this a two-cycle, not a fixed
            # point: the Newton step carries the iterate off the guard and the
            # guard puts it back, forever. The tell is the SAME pushed value
            # twice. Yield the margin rather than the answer -- shrinking costs
            # quadrature accuracy, cycling costs the whole calculation.
            if (last_push is not None and abs(w_push - last_push) < tol
                    and offset > offset_min):
                if frozen:
                    # A FROZEN GUARD THAT DOES NOT REACH THE ROOT IS NEWS. The
                    # caller pinned it so every geometry takes the same Newton
                    # path; shrinking it here is still the only way to the
                    # root, but doing it silently would move this geometry onto
                    # another path while the surface looked frozen.
                    warnings.warn(
                        f'the frozen {offset:.1e} Ha pole guard of orbital {p} '
                        f'does not reach its quasiparticle root, so this '
                        f'geometry is not on the frozen Newton path: the guard '
                        f'is relaxed from here as an unpinned one would be.',
                        RuntimeWarning, stacklevel=2)
                    frozen = False
                offset = max(0.5 * offset, offset_min)
                if guard_out is not None:
                    guard_out['pole_offset'] = offset
                w_push = float(eps[q] + np.sign(w - eps[q] or 1.0) * offset)
            last_push = w_push
            blocker = q
            w = w_push
        was_pinned = pinned
        s = sigma(w)
        sp = slope(w)
        step = -(w - eps[p] - xc_correction - s) / (1.0 - sp)
        w += step
        # A SMALL STEP FROM A PINNED POINT IS NOT CONVERGENCE. The guard
        # overwrites w at the top of the iteration and the step is measured
        # from there, so an iterate the guard is holding against a pole
        # produces a tiny step and looks converged -- while sitting exactly
        # `offset` from eps_q, which is the guard's position and not a root of
        # f. Keep shrinking instead; the floor is what decides when to give up.
        if abs(step) < tol and not was_pinned:
            break
    else:
        # Name the orbital that actually blocked it. The guard fires on the
        # NEAREST eps to the iterate, which is p only at the very first step;
        # after that the root is usually walking through OTHER levels. Blaming
        # eps_p reads as "the quasiparticle correction is tiny", which for a
        # frontier orbital is both false and misleading -- the real statement
        # is that the root is nearly degenerate with a different level, which
        # is a property of the SPECTRUM and says which one to look at.
        where = ('' if blocker is None else
                 f' The iterate was pinned against orbital {blocker} at '
                 f'eps={eps[blocker]:.6f} Ha with the root near {w:.6f} Ha, '
                 f'{abs(w - eps[blocker]):.1e} Ha away'
                 + (' -- the same orbital.' if blocker == p else
                    f', not p={p} (eps={eps[p]:.6f}).'))
        if blocker is not None and blocker != p and linearize_on_capture:
            # CAPTURE BY A POLE, NOT A ROOT. f(w) = w - eps_p - xc - Sigma(w)
            # has a pole at every eps_q, where it runs to +-infinity; Newton
            # near one takes ever smaller steps and converges ONTO it. The tell
            # is a pin against q != p: a genuine quasiparticle root of orbital
            # p has no reason to sit 1e-4 Ha from a DIFFERENT orbital energy,
            # and the pole strength there goes to zero because dSigma/dw
            # diverges.
            #
            # The linearized solution is one Newton step from a point a full
            # guard away from every pole, so it cannot be captured. It is a
            # different approximation from the self-consistent root and says
            # so; the caller records which orbitals used it.
            # The DEFAULT guard fixes that point, not a band a caller pinned
            # smaller to hold one Newton path: that one sits on the pole, where
            # the slope diverges and one step from it means nothing.
            w_lin = float(eps[p] + (linear_offset if p < nocc
                                    else -linear_offset))
            s_lin = sigma(w_lin)
            sp_lin = slope(w_lin)
            w = w_lin - (w_lin - eps[p] - xc_correction - s_lin) / (1.0 - sp_lin)
            warnings.warn(
                f'the CD Newton for orbital {p} was captured by the pole of '
                f'the self-energy at orbital {blocker} (eps='
                f'{eps[blocker]:.6f} Ha), not by a root of its own. Falling '
                f'back to the LINEARIZED quasiparticle energy '
                f'{w:.6f} Ha, one step from eps_{p}+/-{linear_offset:.0e}. That '
                f'is a different approximation from the self-consistent root '
                f'and is not interchangeable with the other orbitals here.',
                RuntimeWarning, stacklevel=2)
            sp = slope(w)
            return w, 1.0 / (1.0 - sp)
        raise RuntimeError(
            f"CD quasiparticle Newton for p={p} did not converge in "
            f"{max_iter} steps; the last pole guard was {offset:.1e} Ha."
            + where +
            f" A root within {offset_min:.0e} Ha of an orbital energy "
            f"cannot be resolved by this quadrature and no offset resolves "
            f"both.")
    if offset != pole_offset:
        warnings.warn(
            f'the quasiparticle root of orbital {p} lies inside the '
            f'{pole_offset:.0e} Ha pole guard, so the guard was relaxed to '
            f'{offset:.1e} Ha to reach it. The self-energy quadrature is less '
            f'accurate that close to the pole; the root is still converged to '
            f'{tol:.0e}.', RuntimeWarning, stacklevel=2)
    sp = slope(w)
    z = 1.0 / (1.0 - sp)
    if z < z_min and linearize_on_capture:
        # CONVERGENCE IS NOT EVIDENCE OF THE RIGHT ROOT. f has a zero just to
        # either side of every pole and Newton is drawn to them; those are
        # satellites, and Z collapses because dSigma/dw diverges there. This
        # catches the case the guard does NOT -- where the iteration converged
        # cleanly onto one instead of cycling against it, which is the same
        # situation resolved by rounding rather than by the physics.
        w_lin = float(eps[p] + (linear_offset if p < nocc
                                else -linear_offset))
        s_lin = sigma(w_lin)
        sp_lin = slope(w_lin)
        w_new = w_lin - (w_lin - eps[p] - xc_correction - s_lin) / (1.0 - sp_lin)
        warnings.warn(
            f'the CD Newton for orbital {p} converged on a root with pole '
            f'strength Z={z:.3f}, below {z_min}: that is a '
            f'satellite at {w:.6f} Ha, not the quasiparticle. Falling back to '
            f'the LINEARIZED quasiparticle energy {w_new:.6f} Ha. That is a '
            f'different approximation from the self-consistent root and is '
            f'not interchangeable with the other orbitals here.',
            RuntimeWarning, stacklevel=2)
        w = w_new
        sp = slope(w)
        z = 1.0 / (1.0 - sp)
    return w, z


def solve_fixed_point(step, x0, carry=None, tol=QP_NEWTON_TOL, max_iter=QP_NEWTON_MAX_ITER):
    """Direct iteration x_{n+1} = step(x_n) for a self-consistent eigenvalue problem.

    The root search for an operator that depends on the energy it is evaluated at:
    build H(x), diagonalize, take the target eigenvalue as the next x. Used by the
    dynamical downfolded Hamiltonians (solve_dynamical_hamiltonian).

    step(x, carry) -> (x_new, carry_new). `carry` threads auxiliary state between
    iterations -- typically the eigenvector, so the next step can follow the SAME
    root by overlap instead of re-picking by eigenvalue proximity -- and whatever
    the last step returned comes back to the caller.

    Returns (x, carry, converged, n_iter). n_iter is the number of steps taken;
    converged is False if max_iter was exhausted.
    """
    x, it, converged = x0, 0, False
    for it in range(1, max_iter + 1):
        x_new, carry = step(x, carry)
        if not np.isfinite(x_new):
            return x_new, carry, False, it
        if abs(x_new - x) < tol:
            return x_new, carry, True, it
        x = x_new
    return x, carry, converged, it


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
