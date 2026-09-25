"""Validates src/Base/utils/grids.py::minimax_time_grid (the GreenX minimax
imaginary-time grid, tabulated in src/Base/utils/minimax_tau_data.json from
GreenX GX-TimeFrequency/src/minimax_tau.F90) by checking the Laplace-transform
identity it exists to provide:

    1/x ~= sum_k w_k * exp(-x * tau_k)      for x in [e_min, e_max]

This is the standard quadrature used in Laplace-transformed MP2 to factorize
an energy denominator into separable per-particle exponentials; the same
trick generalizes directly to MP3's six-index (three-particle) denominator
D_ijk^abc = (e_a+e_b+e_c) - (e_i+e_j+e_k), since
exp(-D*tau) = exp(-e_a*tau)*exp(-e_b*tau)*exp(-e_c*tau)*exp(e_i*tau)*exp(e_j*tau)*exp(e_k*tau)
is fully separable.

Both of the tau grid's rescalings are the INVERSE of minimax_frequency_grid's,
because tau carries 1/energy where omega carries energy: the physical grid
divides by e_min (minimax_grids.F90:142) and the narrow-range stretch
multiplies by e_ratio (minimax_tau.F90:2933, against minimax_omega.F90:2217).
Copying either one from the frequency grid is silent -- the units still look
right -- so both are locked in here, the second by the branch cases below and
by the invariance of every product tau_k * omega_k. The GreenX line numbers
cite GX-TimeFrequency/src/.

Runs as a script (exit code is the verdict) and under pytest.
"""
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from src.Base.utils.grids import (_load_minimax_data, _load_minimax_tau_data,
                                  _table_row, minimax_frequency_grid,
                                  minimax_time_grid, minimax_tau_supported_sizes)
from src.Base.utils.time_frequency import COSINE_TW, minimax_transform_weights

# (grid_size, e_min, e_max, tolerance) on ranges each grid_size is tabulated for
IN_TABLE = [
    (6, 1.0, 3.0, 1e-6),
    (8, 1.0, 12.0, 1e-5),
    (14, 0.3, 30.0, 1e-6),
    (20, 0.3, 30.0, 1e-9),
    (30, 1.0, 4000.0, 1e-8),
]

# Ranges NARROWER than the narrowest tabulated one at more than 20 points, so
# GreenX's e_ratio stretch fires. Nothing here is out of table: the stretch
# slides an existing column onto the narrower range exactly.
STRETCHED = [
    (22, 1.0, 100.0, 1e-9),      # e_ratio 2.0     (tabulated floor 300)
    (24, 1.0, 100.0, 1e-9),      # e_ratio 4.7     (floor 700)
    (28, 1.0, 100.0, 1e-9),      # e_ratio 10.3    (floor 1545)
    (30, 1.0, 100.0, 1e-9),      # e_ratio 19.4    (floor 2906)
    (34, 1.0, 100.0, 1e-9),      # e_ratio 64.3    (floor 9649)
    (34, 0.3, 30.0, 1e-9),       # same ratio off e_min = 1, so both scalings act
    (30, 2.0, 60.0, 1e-9),
]


def relative_laplace_error(ntau, e_min, e_max, nsample=2001):
    """max |1 - x sum_k w_k exp(-x tau_k)| over x in [e_min, e_max]."""
    tau, w = minimax_time_grid(ntau, e_min, e_max)
    assert tau.shape == (ntau,) and w.shape == (ntau,)
    x = np.geomspace(e_min, e_max, nsample)
    return float(np.abs(1.0 - x * (np.exp(-np.outer(x, tau)) @ w)).max())


@pytest.mark.parametrize('ntau,e_min,e_max,tol', IN_TABLE)
def test_the_laplace_identity_holds_on_a_tabulated_range(ntau, e_min, e_max, tol):
    assert relative_laplace_error(ntau, e_min, e_max) < tol


@pytest.mark.parametrize('ntau,e_min,e_max,tol', STRETCHED)
def test_the_narrow_range_stretch_preserves_the_laplace_identity(ntau, e_min, e_max, tol):
    """The stretch is an exact rescaling, so it costs nothing.

    Applying it with the frequency grid's sign instead divides where it should
    multiply, and these same fits reach only 4.9e-05 (ntau=22) to 7.4e-01
    (ntau=34).
    """
    assert relative_laplace_error(ntau, e_min, e_max) < tol


def test_more_points_never_make_the_fit_worse():
    """The residual falls with ntau and then sits on the tabulated coefficients'
    own precision -- there is no upper edge to the usable point count.

    With the stretch inverted the residual instead TURNS OVER at 20 points and
    climbs to 7.4e-01 by 34, which is the whole of the reported "minimax does
    not converge above ~20 points at molecular energy ranges".
    """
    e_min, e_max = 1.0, 100.0
    err = {n: relative_laplace_error(n, e_min, e_max)
           for n in minimax_tau_supported_sizes()}
    below = max(err[n] for n in err if n <= 20)
    above = max(err[n] for n in err if n > 20)
    assert above < below, f'{above:.2e} at ntau > 20 against {below:.2e} at or below'
    assert above < 1e-9, f'worst residual above 20 points is {above:.2e}'


def test_the_transform_between_the_axes_also_keeps_improving():
    """The Remez fit the two axes are actually used through, not just the grid.

    chi0(i.tau) -> chi0(i.omega) is what the space-time routes run, and it is
    where a mis-stretched tau axis shows up: it fits to 1.8e-07 at 20 points
    and 8.2e-08 at 34, but with the stretch inverted it turns over instead and
    reaches 9.2e-03 at 24 and 5.6e-01 at 34.
    """
    e_min, e_max = 0.4, 40.0
    err = {}
    for n in minimax_tau_supported_sizes():
        tau = 0.5 * minimax_time_grid(n, e_min, e_max)[0]
        omega = minimax_frequency_grid(n, e_min, e_max)[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            err[n] = minimax_transform_weights(COSINE_TW, tau, omega,
                                               e_min, e_max)[1]
    best_below = min(err[n] for n in err if n <= 20)
    assert all(err[n] <= best_below for n in err if n > 20), \
        ' '.join(f'{n}:{err[n]:.1e}' for n in sorted(err))


@pytest.mark.parametrize('ntau,e_min,e_max', [(22, 1.0, 100.0), (28, 0.3, 30.0),
                                              (34, 2.0, 60.0)])
def test_the_two_axes_carry_opposite_powers_of_the_stretch(ntau, e_min, e_max):
    """Every product tau_k * omega_k comes back to the bare tabulated one.

    The transform matrices between the axes are built out of cos(tau_k omega_l),
    so this product is what the pair has to preserve. e_min cancels between the
    axes whatever the stretch does, and the stretch cancels only if the two
    carry opposite powers of it; carrying the same power moves every product by
    e_ratio^2 and the transform fit collapses with it.
    """
    tau_row, e_ratio = _table_row(_load_minimax_tau_data()[str(ntau)],
                                  ntau, e_min, e_max)
    omega_row, _ = _table_row(_load_minimax_data()[str(ntau)],
                              ntau, e_min, e_max)
    assert e_ratio > 1.0, 'this case does not exercise the stretch'
    tau = minimax_time_grid(ntau, e_min, e_max)[0]
    omega = minimax_frequency_grid(ntau, e_min, e_max)[0]
    assert np.allclose(tau * omega, tau_row[:ntau] * omega_row[:ntau],
                       rtol=1e-12, atol=0)


def main():
    print(f'Supported minimax tau grid sizes: {minimax_tau_supported_sizes()}')
    failures = 0
    for label, cases in (('tabulated', IN_TABLE), ('stretched', STRETCHED)):
        for ntau, e_min, e_max, tol in cases:
            err = relative_laplace_error(ntau, e_min, e_max)
            ok = err < tol
            failures += not ok
            print(f'  [{label:9s}] ntau={ntau:3d} range=({e_min},{e_max})  '
                  f'max relerr={err:.3e} (tol {tol:.0e})  [{"OK" if ok else "FAIL"}]')
    for fn in (test_more_points_never_make_the_fit_worse,
               test_the_transform_between_the_axes_also_keeps_improving):
        try:
            fn()
            print(f'  [{fn.__name__}] OK')
        except AssertionError as exc:
            failures += 1
            print(f'  [{fn.__name__}] FAIL: {exc}')
    for ntau, e_min, e_max in [(22, 1.0, 100.0), (28, 0.3, 30.0), (34, 2.0, 60.0)]:
        try:
            test_the_two_axes_carry_opposite_powers_of_the_stretch(ntau, e_min, e_max)
            print(f'  [tau*omega ] ntau={ntau:3d} invariant  [OK]')
        except AssertionError:
            failures += 1
            print(f'  [tau*omega ] ntau={ntau:3d}  [FAIL]')
    print('\n' + ('All minimax_time_grid checks passed.'
                  if not failures else f'{failures} FAILURES above.'))
    return failures


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
