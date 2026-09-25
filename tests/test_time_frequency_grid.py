"""TimeFrequencyGrid: one container, two backends, and the honest diagnostics.

Checks:
  1. The ported minimax cosine transform reproduces the model pair it is fitted
     for: Pi(i.tau) = e^{-x tau}  ->  Pi(i.omega) = 2x/(x^2 + omega^2).
  2. The minimax forward/backward pair is NOT a dual -- the property that makes
     four separate matrices necessary rather than one plus an inverse.
  3. The IR backend, by contrast, DOES round-trip, because its transform goes
     through basis coefficients.
  4. gauss_legendre carries a frequency axis only and refuses to transform
     rather than returning nonsense.
  5. Both backends present the identical interface, so a consumer never
     branches on `method`.
  6. The transform fit is a pseudo-inverse: an exactly zero singular value
     contributes zero instead of 0/0, every other one is inverted bitwise as
     before, and a row that still comes out non-finite is refused.
  7. A grid size is an explicit count or the 'auto' sentinel resolved by the
     widest range, never a sentinel forwarded into a constructor.

Usage: python tests/test_time_frequency_grid.py; every check also asserts, so
the functions are pytest-collectable and fail there as well.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.constants import TRANSFORM_FIT_RCOND
from src.Base.utils import time_frequency
from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.time_frequency import (COSINE_TW, COSINE_WT,
                                           DEFAULT_TAU_TARGET,
                                           TimeFrequencyGrid,
                                           minimax_convergence_floor,
                                           minimax_points_for_accuracy,
                                           minimax_points_for_ranges,
                                           minimax_transform_weights,
                                           resolve_grid_size)

E_MIN, E_MAX = 0.4, 40.0
#: The grid of the pseudo-inverse check: water's own energy range on the
#: 8-point tau axis and 24-point frequency axis the self-energy tests use.
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
NTAU, NFREQ = 8, 24


def check(ok, label, detail=''):
    """Report a condition AND enforce it.

    pytest DISCARDS whatever a test function returns, so a suite built out of
    `return ok` passes whether ok is True or False -- verified on pytest 9.1.1.
    Asserting here fixes every call site at once, and makes the script path
    stop at a genuine failure instead of printing FAIL and exiting 0.
    """
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'  ({detail})' if detail else ''))
    assert ok, f"{label}" + (f'  ({detail})' if detail else '')
    return bool(ok)


def test_minimax_transforms_the_model_pair():
    g = TimeFrequencyGrid.minimax(14, E_MIN, E_MAX)
    ok = True
    worst = 0.0
    for x in (0.5, 2.0, 11.0, 37.0):
        got = g.to_omega(np.exp(-x * g.tau_points), parity='even')
        want = 2 * x / (x**2 + g.omega_points**2)
        rel = np.abs((got - want) / want).max()
        worst = max(worst, rel)
    ok &= check(worst < 5e-2, 'cosine tau->omega reproduces 2x/(x^2+w^2)',
                f'worst rel err {worst:.2e} over x in [e_min, e_max]')
    ok &= check(max(g.fit_errors.values()) < 1e-3,
                'every transform fit converged',
                ', '.join(f'{k}={v:.1e}' for k, v in g.fit_errors.items()))
    return ok


def test_transform_matrices_are_not_inverses():
    """Why four matrices -- and why |A B - I| is the wrong acceptance test."""
    ok = True
    duality = {n: TimeFrequencyGrid.minimax(n, E_MIN, E_MAX).duality_error()
               for n in (10, 14, 18)}
    ok &= check(all(e > 1e-3 for e in duality.values()),
                'as MATRICES the forward/backward pair is not an inverse pair',
                ', '.join(f'n={k}: {v:.2e}' for k, v in duality.items()))
    g = TimeFrequencyGrid.minimax(14, E_MIN, E_MAX)
    sv = np.linalg.svd(g.cosft_wt, compute_uv=False)
    ok &= check(sv.max() / sv.min() > 1e2, 'nor is either one orthogonal',
                f'singular values {sv.min():.2e} .. {sv.max():.2e}')
    # ...but on the subspace every physical Pi(i.tau) lives in, it round-trips.
    rt = {n: TimeFrequencyGrid.minimax(n, E_MIN, E_MAX).roundtrip_error()
          for n in (10, 14, 18)}
    ok &= check(all(e < 1e-4 for e in rt.values()),
                'yet the round trip on sums of exponentials is accurate',
                ', '.join(f'n={k}: {v:.2e}' for k, v in rt.items()))
    return ok


def test_minimax_more_points_never_hurt():
    """A transform that misses is UNDER-resolved: more points, never fewer.

    Below the narrowest tabulated Remez column GreenX slides that column onto
    the requested range, (tau, omega) -> (tau e_ratio, omega / e_ratio), and
    the two axes carry OPPOSITE powers of e_ratio (`grids._table_row`).
    Dividing on both -- which is what copying the frequency grid's rescaling
    onto the tau one does -- scales every product tau*omega by 1/e_ratio^2 and
    the fit collapses above 20 points instead of saturating. It is silent: the
    units still look right and only the residual moves. So the property gated
    here is monotonicity, at the narrow ranges where the stretch is active.
    """
    ok = True

    floors = [minimax_convergence_floor(n) for n in (14, 20, 24, 30, 34)]
    ok &= check(floors == sorted(floors) and floors[0] < floors[-1],
                'the tabulated Remez floor rises monotonically with n',
                ' -> '.join(f'{f:g}' for f in floors))

    for rng in (1e2, 1e3, 1e4):
        errs = []
        for n in (14, 20, 24, 30, 34):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                errs.append(TimeFrequencyGrid.minimax(
                    n, E_MIN, E_MIN * rng).fit_errors['cosft_wt'])
        # Past ~1e-6 the fit sits on the tabulated coefficients' own precision
        # and stops improving, so the claim is that it never DEGRADES by more
        # than that floor -- not that the sequence is strictly decreasing.
        ok &= check(max(errs[1:]) < max(errs[0], 1e-6),
                    f'range={rng:.0e}: no point count above 14 is worse',
                    ' '.join(f'n={n}:{e:.1e}'
                             for n, e in zip((14, 20, 24, 30, 34), errs)))

    # ... and too FEW points for a wide range is still flagged
    cases = [(14, 1e2, False), (20, 1e2, False), (30, 1e4, False),
             (14, 1e4, True)]
    for n, rng, want_warn in cases:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            g = TimeFrequencyGrid.minimax(n, E_MIN, E_MIN * rng)
        got_warn = any('minimax transform fit' in str(w.message) for w in caught)
        ok &= check(got_warn == want_warn,
                    f'n={n:2d}, range={rng:.0e}: '
                    f"{'flagged' if want_warn else 'accepted'}",
                    f'fit err {g.fit_errors["cosft_wt"]:.1e}')
    return ok


def test_ir_does_round_trip():
    g = TimeFrequencyGrid.ir(beta=50.0, omega_max=5.0, eps=1e-10, statistics='boson')
    ok = check(g.roundtrip_error() < 1e-6,
               'IR round-trips through basis coefficients',
               f'roundtrip = {g.roundtrip_error():.2e}, L = {g.meta["size"]}')
    # a bosonic Pi(i.tau) = e^{-x tau} + e^{-x (beta - tau)} round-trips
    beta, x = 50.0, 1.3
    f = np.exp(-x * g.tau_points) + np.exp(-x * (beta - g.tau_points))
    back = g.to_tau(g.to_omega(f, 'even'), 'even')
    rel = np.abs(back - f).max() / np.abs(f).max()
    ok &= check(rel < 1e-6, 'and a bosonic model function survives the round trip',
                f'max rel err {rel:.2e}')
    return ok


IR_SWEEP = [(beta, wmax, eps) for beta in (100.0, 200.0, 800.0)
            for wmax in (2.0, 4.0) for eps in (1e-8, 1e-10)]


def test_ir_duality_is_the_basis_round_trip():
    """`duality_error` must be SMALL on an IR grid, for both parities.

    Its docstring always said so; the code returned max|A B - I| instead, which
    for IR is a projector of rank min(L_sector, nfreq) against the identity on
    nfreq dimensions and so is O(1) whenever the sampling set is bigger than
    the sector -- i.e. normally. Measured 7.5e-01 to 9.1e-01 on grids whose
    transforms were perfectly good.

    Worse, the raw product was ANTI-CORRELATED with the truth: the only way to
    make A B the identity is to starve the sampling set until nfreq <= L_sector,
    which is exactly when the omega -> tau fit goes rank deficient. Three grids
    in this sweep reported 1.6e-14 .. 1.9e-14 while their round trip was wrong
    by 0.22 to 0.48. Both halves are gated here: the diagnostic is small AND
    the round trip is small, on every grid.
    """
    ok = True
    for beta, wmax, eps in IR_SWEEP:
        g = TimeFrequencyGrid.ir(beta, wmax, eps=eps, statistics='boson')
        de, do, rt = g.duality_error('even'), g.duality_error('odd'), g.roundtrip_error()
        ok &= check(max(de, do) < 1e-9,
                    f'beta={beta:g} wmax={wmax:g} eps={eps:.0e}: IR is two-sided '
                    f'on both parities',
                    f'even {de:.1e}, odd {do:.1e} (L={g.meta["size"]}, '
                    f'nfreq={g.nfreq})')
        ok &= check(rt < 1e-8,
                    f'beta={beta:g} wmax={wmax:g} eps={eps:.0e}: and the model '
                    f'round trip agrees',
                    f'{rt:.1e}')
    return ok


def test_ir_sector_split_is_a_clean_partition():
    """Each sector must have no more functions than there are sampling points.

    The split was `|Re uhat_l| > 1e-8 max|uhat|`, which measures one function
    against the largest of all of them while the part that should vanish is
    that function's own truncation floor -- growing with l and with eps. At
    beta = 100, omega_max = 2, eps = 1e-10 it put l = 35 (|Re| = 4.5e-08 vs
    |Im| = 0.399) in the even sector, taking it to 20 functions on 19 points:
    rank deficient, and `duality_error` 5.8e-01. Comparing each function's own
    two parts splits the same basis 19/18.

    A parity split should also come out nearly even, so the imbalance is worth
    pinning directly -- it is the cheap signal that a function has been
    misfiled.
    """
    ok = True
    for beta, wmax, eps in IR_SWEEP:
        g = TimeFrequencyGrid.ir(beta, wmax, eps=eps, statistics='boson')
        L, n_even = g.meta['size'], g.meta['n_even_sector']
        n_odd = L - n_even
        ok &= check(max(n_even, n_odd) <= g.nfreq,
                    f'beta={beta:g} wmax={wmax:g} eps={eps:.0e}: every sector is '
                    f'determined',
                    f'sectors {n_even}/{n_odd} on nfreq={g.nfreq}')
        ok &= check(abs(n_even - n_odd) <= 1,
                    f'beta={beta:g} wmax={wmax:g} eps={eps:.0e}: and the parity '
                    f'split is even',
                    f'{n_even} vs {n_odd} of L={L}')
    return ok


def test_minimax_duality_is_still_the_raw_product():
    """The IR branch must not have changed what minimax reports.

    `duality_error` on a minimax grid is GreenX's own `cosft_duality_error` and
    is expected to be O(1) -- the module docstring tabulates it and
    test_transform_matrices_are_not_inverses reads it. Recomputed here against
    max|A B - I| directly so the branch cannot quietly capture minimax too.
    """
    ok = True
    for n in (6, 14, 20):
        g = TimeFrequencyGrid.minimax(n, E_MIN, E_MAX)
        raw = float(np.abs(g.cosft_wt @ g.cosft_tw - np.eye(g.nfreq)).max())
        ok &= check(g.duality_error('even') == raw,
                    f'n={n}: minimax still reports max|A B - I| exactly',
                    f'{g.duality_error("even"):.3e}')
    return ok


def test_gauss_legendre_refuses_to_transform():
    g = TimeFrequencyGrid.gauss_legendre(20, w0=0.5)
    ok = check(g.nfreq == 20 and g.ntau == 0,
               'gauss_legendre is frequency-only', f'{g!r}')
    try:
        g.to_omega(np.zeros(20))
        ok &= check(False, 'to_omega raises instead of returning nonsense')
    except ValueError as exc:
        ok &= check('carries no' in str(exc),
                    'to_omega raises instead of returning nonsense',
                    str(exc)[:60])
    return ok


def test_identical_interface():
    grids = [TimeFrequencyGrid.minimax(14, E_MIN, E_MAX),
             TimeFrequencyGrid.ir(beta=50.0, omega_max=5.0, statistics='boson')]
    fields = ('tau_points', 'tau_weights', 'omega_points', 'omega_weights',
              'cosft_wt', 'cosft_tw', 'sinft_wt', 'sinft_tw')
    ok = True
    for g in grids:
        present = all(getattr(g, f) is not None and len(np.shape(getattr(g, f)))
                      for f in fields)
        shapes_ok = (g.cosft_wt.shape == (g.nfreq, g.ntau)
                     and g.cosft_tw.shape == (g.ntau, g.nfreq))
        ok &= check(present and shapes_ok,
                    f"method={g.method!r} exposes the full interface with consistent shapes",
                    f'ntau={g.ntau} nfreq={g.nfreq}')
    return ok


def _weights(kind, grid, e_min, e_max, rcond=None):
    """The transform, optionally with the pseudo-inverse cutoff DISABLED.

    A negative rcond keeps every singular value, which is the expression this
    routine carried before the cutoff existed -- so the two calls are the
    before and after of the change, on the same grid, in the same process.
    """
    saved = time_frequency.TRANSFORM_FIT_RCOND
    try:
        if rcond is not None:
            time_frequency.TRANSFORM_FIT_RCOND = rcond
        return minimax_transform_weights(kind, grid.tau_points,
                                         grid.omega_points, e_min, e_max)
    finally:
        time_frequency.TRANSFORM_FIT_RCOND = saved


def test_transform_pseudo_inverse():
    """A zero singular value contributes zero; everything else is untouched.

    The fit is a per-point least squares through an SVD, and below 20 points
    it carries no Tikhonov term: the filter is 1/S. A minimax tau point large
    enough that exp(-x tau) underflows over the whole node range leaves an
    exactly zero COLUMN in the design matrix, hence an exactly zero singular
    value, hence 0/0 -- a NaN row in the transform. It is silent twice over:
    the fit error of a NaN row is NaN, and `max(x, nan)` returns x, so the
    fit-error warning never sees it. That is the warning a production run
    raised.

    BITWISE, and that is the point. The smallest relative singular value the
    grids here reach is 2.4e-17, and the cutoff sits at 1e-100: small values
    are still inverted, exactly as GreenX inverts them -- cutting at the
    SVD's backward error instead would have moved this grid's omega -> tau
    weights in the sixth digit, which is a change to the physics and not a
    NaN fix.
    """
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(NFREQ, w0=gap)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grid = TimeFrequencyGrid.minimax_split(NTAU, 0.5 * gap,
                                               eps[-1] - eps[0], nu, wt,
                                               with_sine=False,
                                               with_inverse=False)
    e_min = 0.5 * (eps[nocc] - eps[nocc - 1])
    e_max = eps[-1] - eps[0]
    ok = True
    for kind in (COSINE_TW, COSINE_WT):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cut, err_cut = _weights(kind, grid, e_min, e_max)
            uncut, err_uncut = _weights(kind, grid, e_min, e_max, rcond=-1.0)
        ok &= check(np.array_equal(cut, uncut) and err_cut == err_uncut,
                    f'kind {kind}: the cutoff leaves a production grid bitwise',
                    f'fit error {err_cut:.2e}')

    # The two ways 1/S breaks, in the arithmetic itself, and why it is silent.
    with np.errstate(invalid='ignore', divide='ignore'):
        ok &= check(np.isnan(np.float64(0.0) / np.float64(0.0) ** 2)
                    and np.isinf(np.float64(5e-296) / np.float64(5e-296) ** 2),
                    '0/0 is NaN and a singular value squaring to zero is inf')
    ok &= check(max(0.0, float('nan')) == 0.0, 'a NaN never wins the max')

    # A tau point so large that exp(-x tau) underflows for every node leaves
    # an exactly zero COLUMN, and a singular value whose square is zero.
    tau = np.array([0.1, 1.0, 10.0, 2.0e3])
    omega = np.array([0.5, 1.0, 2.0, 4.0])
    x = e_min * (e_max / e_min) ** (np.arange(400) / 399.0)
    psi, A = time_frequency._psi_and_matrix(COSINE_TW, tau, omega, 0, x)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    ok &= check(np.linalg.norm(A[:, -1]) == 0.0 and S[-1] ** 2 == 0.0
                and S[-1] < TRANSFORM_FIT_RCOND * S[0],
                'an underflowed tau column is an exactly zero singular value',
                f'S {S[0]:.2e} .. {S[-1]:.2e}')
    with np.errstate(invalid='ignore', divide='ignore'):
        old = Vt.T @ ((S / S**2) * (U.T @ psi))
    ok &= check(not np.isfinite(old).any(), 'and 1/S turns the row non-finite')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        W, err = minimax_transform_weights(COSINE_TW, tau, omega, e_min, e_max)
    ok &= check(np.isfinite(W).all() and err > 0.0 and np.isfinite(err)
                and any(issubclass(c.category, RuntimeWarning) for c in caught),
                'with the cutoff the row is finite and its error is seen, and '
                'warned', f'fit error {err:.2e}')
    # Without the cutoff the same call is refused rather than returning NaN --
    # and with the warning switched off too, since `warn` governs the warning
    # and not the refusal.
    for warn in (True, False):
        saved = time_frequency.TRANSFORM_FIT_RCOND
        refused = False
        try:
            time_frequency.TRANSFORM_FIT_RCOND = -1.0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                minimax_transform_weights(COSINE_TW, tau, omega, e_min, e_max,
                                          warn=warn)
        except FloatingPointError:
            refused = True
        finally:
            time_frequency.TRANSFORM_FIT_RCOND = saved
        ok &= check(refused, f'without the cutoff a non-finite row is refused '
                             f'(warn={warn})')
    # and warn=False still silences the under-resolved fit it was made for
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, err_quiet = minimax_transform_weights(COSINE_TW, tau, omega, e_min,
                                                 e_max, warn=False)
    ok &= check(err_quiet == err and not any(
                    issubclass(c.category, RuntimeWarning) for c in caught),
                'warn=False returns the same error and warns nothing')
    return ok


def test_grid_size_resolution():
    """'auto' or None resolves over EVERY ratio, the widest binding; an
    explicit count passes through with nothing measured."""
    ratios = (40.0, 486.0, 3000.0)
    n, worst = minimax_points_for_ranges(ratios)
    per = [minimax_points_for_accuracy(1.0, R, target=DEFAULT_TAU_TARGET)
           for R in ratios]
    ok = check(n == max(p[0] for p in per)
               and worst == max(p[1] for p in per),
               'the widest ratio sets the count and the reported error',
               f'n={n}, worst={worst:.1e}, per ratio '
               + ', '.join(f'{p[0]}' for p in per))
    ok &= check(resolve_grid_size('auto', ratios) == (n, worst)
                and resolve_grid_size(None, ratios) == (n, worst)
                and resolve_grid_size('AUTO', ratios) == (n, worst),
                "'auto', 'AUTO' and None are the same sentinel")
    ok &= check(resolve_grid_size(18, ratios) == (18, None)
                and resolve_grid_size('22', ratios) == (22, None),
                'an explicit count passes through unmeasured')
    missed = minimax_points_for_ranges(ratios, target=1e-30)
    ok &= check(np.isfinite(missed[1]) and missed[1] > 1e-30,
                'a target nothing reaches returns the best count WITH the error '
                'it actually obtained', f'{missed}')
    none = minimax_points_for_ranges(ratios, npoints_max=4)
    ok &= check(none == (4, float('inf')),
                'no tabulated size in reach: the ceiling and an infinite error',
                f'{none}')
    return ok


if __name__ == '__main__':
    all_ok = True
    print('\n-- 1. minimax reproduces its model pair')
    all_ok &= test_minimax_transforms_the_model_pair()
    print('\n-- 2. matrices are not inverses, but the round trip works')
    all_ok &= test_transform_matrices_are_not_inverses()
    print('\n-- 2b. more points never hurt the transform fit')
    all_ok &= test_minimax_more_points_never_hurt()
    print('\n-- 3. IR does round-trip')
    all_ok &= test_ir_does_round_trip()
    print('\n-- 3b. the IR duality diagnostic is the basis round trip')
    all_ok &= test_ir_duality_is_the_basis_round_trip()
    all_ok &= test_ir_sector_split_is_a_clean_partition()
    all_ok &= test_minimax_duality_is_still_the_raw_product()
    print('\n-- 4. gauss_legendre is frequency-only')
    all_ok &= test_gauss_legendre_refuses_to_transform()
    print('\n-- 5. one interface, both backends')
    all_ok &= test_identical_interface()
    print('\n-- 6. the transform fit is a pseudo-inverse')
    all_ok &= test_transform_pseudo_inverse()
    print('\n-- 7. a grid size is a count or a resolved sentinel')
    all_ok &= test_grid_size_resolution()
    print('\n' + ('All TimeFrequencyGrid checks passed.' if all_ok else 'FAILURES above.'))
    sys.exit(0 if all_ok else 1)
