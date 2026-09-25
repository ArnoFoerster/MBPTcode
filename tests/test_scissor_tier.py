"""The states the pole model cannot carry take a frozen shift, not a solve.

`compressible` splits the quasiparticle set at reach = 1, and reach is not the
same cut as "core": on benzene the six carbon 1s states sit at reach 22.5 but
three inner-valence states at 18-22 eV sit at 1.19 and 1.71, outside the wall
and nowhere near the core. Those three are also where the leverage is -- a
20 eV error on the core moves a frontier quasiparticle by 0.05 meV, the inner
valence by 3.6-10 meV per eV -- so the shift they take has to be calibrated on
explicit roots rather than guessed, and frozen so the tier boundary cannot move
between geometries.
"""
import types

import numpy as np
import pytest

from src.gradients.excited_state import ExcitedStateChain
from src.gradients.qp_space_time import (calibrate_scissor, frozen_scissor,
                                         scissor_route)

#: A core state, two inner-valence ones, then the frontier.
EPS = np.array([-11.0, -1.35, -1.20, -0.62, -0.35, 0.18, 0.44])
NOCC = 5


def test_a_scalar_applies_to_every_excluded_state():
    assert frozen_scissor(0.1, 3) == pytest.approx(0.1)
    assert frozen_scissor(0.1, 0) == pytest.approx(0.1)


def test_a_mapping_applies_per_orbital_and_may_omit_one():
    assert frozen_scissor({3: 0.2}, 3) == pytest.approx(0.2)
    assert frozen_scissor({1: 0.2}, 3) is None
    assert frozen_scissor(None, 3) is None


def test_calibration_takes_the_nearest_probe_in_orbital_energy():
    """One probe per tier: the core takes the core's shift, the inner valence
    the inner valence's, and neither borrows the other's."""
    roots = {0: -11.6, 1: -1.50}
    shifts = calibrate_scissor(EPS, NOCC, roots, excluded=[0, 1, 2])
    assert shifts[0] == pytest.approx(-0.6)          # core probe
    assert shifts[1] == pytest.approx(-0.15)         # inner-valence probe
    assert shifts[2] == pytest.approx(-0.15), 'orbital 2 is nearest orbital 1'


def test_a_probe_is_required():
    with pytest.raises(ValueError, match='at least one'):
        calibrate_scissor(EPS, NOCC, {}, excluded=[0])


def test_the_shift_is_frozen_not_recomputed():
    """The same shifts must come back for a DISPLACED geometry's eigenvalues.

    That is the whole point of freezing: the tier assignment and the shift are
    decided once, so a state cannot change tier as the geometry moves and put a
    step in the surface.
    """
    roots = {0: -11.6, 1: -1.50}
    shifts = calibrate_scissor(EPS, NOCC, roots, excluded=[0, 1, 2])
    displaced = EPS + np.array([0.02, -0.03, 0.01, 0.0, 0.0, 0.0, 0.0])
    again = {p: frozen_scissor(shifts, p) for p in (0, 1, 2)}
    assert again == pytest.approx(shifts)
    assert not np.allclose(displaced, EPS), 'the probe geometry must differ'


def test_the_excluded_set_is_read_off_reach_not_off_depth():
    """`compressible` decides, and it is not a depth threshold in disguise."""
    from src.SingleReference.GW.sum_over_poles import compressible
    eps = np.array([-11.0, -0.9, -0.62, -0.35, 0.18, 0.44, 0.83])
    nocc = 4
    gap = (eps[nocc:][None, :] - eps[:nocc][:, None]).min()
    frontier_ok, frontier_reach = compressible(float(eps[nocc - 1] + 0.01),
                                               eps, nocc)
    deep_ok, deep_reach = compressible(float(eps[0] + 0.01), eps, nocc)
    assert frontier_ok and not deep_ok
    assert deep_reach > 1.0 > frontier_reach
    assert gap > 0


def test_a_mapping_is_the_tier_and_outranks_the_reach_test():
    """The boundary case that made this necessary: benzene's orbital 9 reads
    reach 1.04 at the Newton start and 0.87 at its converged root, so a
    per-geometry test can put it on either side. A state named in the mapping
    is shifted regardless; one not named takes the pole model regardless."""
    eps = np.array([-11.0, -0.9, -0.62, -0.35, 0.18, 0.44, 0.83])
    nocc = 4
    frontier = float(eps[nocc - 1] + 0.01)
    route, shift = scissor_route({nocc - 1: 0.07}, nocc - 1, eps, nocc, frontier)
    assert route == 'scissor' and shift == pytest.approx(0.07), (
        'a named state is shifted even though its reach passes')
    route, shift = scissor_route({0: 0.5}, 1, eps, nocc, float(eps[1] + 0.01))
    assert route == 'sop' and shift is None, 'an unnamed state is not shifted'


def test_a_scalar_still_auto_detects():
    eps = np.array([-11.0, -0.9, -0.62, -0.35, 0.18, 0.44, 0.83])
    nocc = 4
    assert scissor_route(0.3, nocc - 1, eps, nocc,
                         float(eps[nocc - 1] + 0.01))[0] == 'sop'
    assert scissor_route(0.3, 0, eps, nocc, float(eps[0] + 0.01))[0] == 'scissor'


def test_a_string_scissor_defers_rather_than_being_parsed_as_a_number():
    """'calibrate' is a REQUEST, carried until the reference geometry has
    solved the excluded states. Treating it as a shift would raise, and
    treating it as a mapping would silently shift nothing."""
    eps = np.array([-11.0, -0.9, -0.62, -0.35, 0.18, 0.44, 0.83])
    nocc = 4
    assert frozen_scissor('calibrate', 0) is None
    route, shift = scissor_route('calibrate', 0, eps, nocc, float(eps[0] + 0.01))
    assert (route, shift) == ('sop', None), (
        'an uncalibrated request must leave the state to the fallback, not '
        'shift it by an unknown amount')


def test_a_calibrated_shift_is_the_reference_root_minus_the_eigenvalue():
    """What the surface builds from the reference geometry's own roots."""
    eps = np.array([-11.0, -1.35, -1.20, -0.62, -0.35, 0.18, 0.44])
    roots = {0: -11.62, 1: -1.47}
    shifts = {p: roots[p] - eps[p] for p in roots}
    assert shifts[0] == pytest.approx(-0.62)
    assert shifts[1] == pytest.approx(-0.12)
    assert frozen_scissor(shifts, 1) == pytest.approx(-0.12)
    assert scissor_route(shifts, 1, eps, 5, -1.35)[0] == 'scissor'


def _stub_surface(scissor, excluded=(1,)):
    """An ExcitedStateChain carrying only what a quasiparticle set solve
    reads, over a `qp_set_gradient` that reports which route each state took.

    The stub sends `excluded` to the real axis while no shift is on file for
    it, and to the scissor once one is -- which is the whole behaviour the
    calibration depends on and none of the arithmetic it does not.
    """
    surface = object.__new__(ExcitedStateChain)
    surface.qp_set = np.array([1, 3])
    surface.nocc = NOCC
    surface.gw_grid = surface.nu = surface.wt = None
    surface.residue_route = 'sop'
    surface.pole_offsets, surface.qp_seeds, surface.scissor_map = {}, {}, {}
    surface.scissor = scissor
    surface.n_poles = 12
    surface.sop_stride = None
    surface.tile_gb = None
    surface.mf0 = types.SimpleNamespace(mo_energy=EPS)
    surface._grow_cd_grid = lambda *a, **k: False
    return surface


def _stub_solve(monkeypatch, surface, excluded=(1,), root=-1.50):
    calls = []

    def fake(*args, **kw):
        scissor = kw['scissor']
        routes, roots = {}, {}
        for p in (1, 3):
            on_file = frozen_scissor(scissor, p) is not None
            routes[p] = ('scissor' if on_file else 'laplace') \
                if p in excluded else 'sop'
            roots[p] = root if p in excluded else float(EPS[p])
        calls.append(dict(routes))
        kw['route_out'].update(routes=routes, roots=roots, pole_offsets={})
        return (np.zeros(2), np.zeros(2))

    monkeypatch.setattr('src.gradients.excited_state.qp_set_gradient', fake)
    return calls


def test_a_calibrated_solve_is_repeated_so_the_gradient_reads_the_shift(
        monkeypatch):
    """The pass that MEASURES a shift is the pass that solved the state on the
    real axis, so it cannot also be the pass that uses it.

    Without the repeat the map is written after the last solve and the
    gradient -- and every displaced build seeded from it -- silently runs the
    fallback, which is bit-for-bit the uncalibrated answer.
    """
    surface = _stub_surface('calibrate')
    calls = _stub_solve(monkeypatch, surface)
    surface._qp_set_solve(None, None, EPS, 0.0, np.zeros(2), np.zeros(2))
    assert len(calls) == 2, 'the solve must run again once a shift is on file'
    assert calls[0][1] == 'laplace', 'the first pass measures it on the axis'
    assert calls[1][1] == 'scissor', 'the second pass must SPEND it'
    assert surface.scissor_map[1] == pytest.approx(-1.50 - EPS[1])
    assert 3 not in surface.scissor_map, 'a compressible state is not tiered'


def test_calibration_tiers_what_the_route_excluded_not_what_reach_rereads(
        monkeypatch):
    """A state the pole model carried is never tiered, whatever a second
    reading of reach at its root would say."""
    surface = _stub_surface('calibrate', excluded=())
    calls = _stub_solve(monkeypatch, surface, excluded=())
    surface._qp_set_solve(None, None, EPS, 0.0, np.zeros(2), np.zeros(2))
    assert len(calls) == 1, 'nothing to calibrate, so nothing to repeat'
    assert surface.scissor_map == {}


def test_an_uncalibrated_surface_never_repeats_the_solve(monkeypatch):
    surface = _stub_surface(None)
    calls = _stub_solve(monkeypatch, surface)
    surface._qp_set_solve(None, None, EPS, 0.0, np.zeros(2), np.zeros(2))
    assert len(calls) == 1
    assert surface.scissor_map == {}


def test_nothing_is_frozen_from_a_pass_that_grew_the_grid(monkeypatch):
    """The roots and guard bands are frozen by `setdefault`, so freezing on a
    pass that then doubled the quadrature locks in the under-resolved ones.

    Guards the short circuit in `_qp_set_solve`: hoisting the freeze out of
    the `or` into its own name passes every other gate here and silently seeds
    the whole chain from a grid that was thrown away.
    """
    surface = _stub_surface(None)
    roots = iter((-9.9, -1.50))          # coarse grid first, then the real one
    calls = []

    def fake(*args, **kw):
        root = next(roots)
        calls.append(root)
        kw['route_out'].update(routes={1: 'sop', 3: 'sop'},
                               roots={1: root, 3: float(EPS[3])},
                               pole_offsets={1: 0.1 * len(calls)})
        return (np.zeros(2), np.zeros(2))

    monkeypatch.setattr('src.gradients.excited_state.qp_set_gradient', fake)
    grows = iter((True, False))
    surface._grow_cd_grid = lambda *a, **k: next(grows)
    surface._qp_set_solve(None, None, EPS, 0.0, np.zeros(2), np.zeros(2))
    assert len(calls) == 2, 'the grid grew once, so the solve runs twice'
    assert surface.qp_seeds[1] == pytest.approx(-1.50), \
        'the seed must come from the settled grid, not the discarded one'
    assert surface.pole_offsets[1] == pytest.approx(0.2)
