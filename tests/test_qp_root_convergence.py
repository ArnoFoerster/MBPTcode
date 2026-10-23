"""Every quasiparticle root finder says when it has no root it can vouch for.

The finder is global: it scans f(w) over the window of the state's depth
and then every wider tier, finds every root in it and picks by weight; no
local Newton stands in when the scan finds nothing.

A planted f(w) = 1 + tanh(w - eps)/2 has no real root: each finder warns by
default and raises QPRootNotConverged under refuse_unconverged, and
full_output returns the flag False. A planted f with only a low-weight root
has no quasiparticle: returned, flagged, never warned or refused.
"""
import warnings

import numpy as np
import pytest

from src.Base.numerics.constants import QP_WINDOW_Z_MIN
from src.Base.step_failed import StepFailed
from src.Solvers.qp_equation import (QPRootNotConverged, solve_qp_equation,
                                     solve_qp_equation_pole_strength)

EPS = -0.4


def no_root(w):
    """f(w) in (0.5, 1.5): no zero, so Newton runs off and no window brackets."""
    return 1.0 + 0.5 * np.tanh(w - EPS)


def quasiparticle(w):
    """f(w) = w - eps - Sigma(w) with a single root at eps + 0.01, Z = 1."""
    return w - EPS - 0.01


def satellite_only(w):
    """f(w) whose only root in the window has Z = 1/f' = 0.02."""
    return 50.0 * (w - EPS - 0.05)


@pytest.mark.parametrize('method', ['pole_strength'])
def test_no_root_in_the_window_warns_by_default(method):
    with pytest.warns(RuntimeWarning, match='no root'):
        root = solve_qp_equation(no_root, EPS, method=method)
    assert np.isnan(root)


@pytest.mark.parametrize('method', ['pole_strength'])
def test_no_root_in_the_window_is_refused_when_asked(method):
    with pytest.raises(QPRootNotConverged) as err:
        solve_qp_equation(no_root, EPS, method=method, refuse_unconverged=True)
    assert isinstance(err.value, StepFailed)
    assert err.value.reason == 'no_root'


@pytest.mark.parametrize('method', ['pole_strength'])
def test_full_output_carries_the_flag(method):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _, ok = solve_qp_equation(no_root, EPS, method=method,
                                  full_output=True)
    assert ok is False or ok == np.False_
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        root, ok = solve_qp_equation(quasiparticle, EPS, method=method,
                                     full_output=True)
    assert ok
    assert abs(root - (EPS + 0.01)) < 1e-6


@pytest.mark.parametrize('method', ['graphical', 'bisection', 'newton'])
def test_the_retired_rules_are_refused(method):
    """One rule serves every route: the nearest-root and fixed-bracket rules
    are refused by name, not solved by a second finder."""
    with pytest.raises(ValueError, match='retired'):
        solve_qp_equation(quasiparticle, EPS, method=method)


def test_a_satellite_alone_is_returned_flagged_and_quiet():
    """A state whose only root has Z = 0.02 has no quasiparticle; that is no
    failure (high virtuals often have none), so the root comes back with the
    flag False and no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        root, ok = solve_qp_equation_pole_strength(satellite_only, EPS,
                                                   full_output=True)
    assert abs(root - (EPS + 0.05)) < 1e-6 and not ok


def test_a_root_below_half_weight_is_flagged():
    """Z = 0.4 passes nothing: QP_WINDOW_Z_MIN is the one threshold."""
    weak = lambda w: 2.5 * (w - EPS - 0.02)
    root, ok = solve_qp_equation_pole_strength(weak, EPS, full_output=True)
    assert abs(root - (EPS + 0.02)) < 1e-6 and not ok
    assert QP_WINDOW_Z_MIN == 0.5


def test_a_converged_root_is_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert abs(solve_qp_equation_pole_strength(quasiparticle, EPS)
                   - (EPS + 0.01)) < 1e-6


def test_a_root_beyond_the_first_window_is_found_by_widening():
    """A quasiparticle 0.8 Ha off eps lies outside the 0.5 Ha window of a
    shallow state; the scan widens to it instead of handing over to Newton."""
    far = lambda w: w - EPS - 0.8
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert abs(solve_qp_equation(far, EPS) - (EPS + 0.8)) < 1e-6


def test_the_quasiparticle_beyond_a_satellite_window_wins():
    """Sigma with one pole 0.06 Ha above eps: inside the first window only a
    satellite beside it (Z ~ 1e-3), the quasiparticle (Z ~ 1) 0.7 Ha out. The
    largest-weight rule widens to it rather than return the satellite."""
    def f(w):
        x = w - EPS
        return x - 0.7 - 5e-4 / (x - 0.06)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        root = solve_qp_equation_pole_strength(f, EPS)
    assert abs(root - EPS - 0.70078) < 1e-4
