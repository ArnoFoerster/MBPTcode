"""Every quasiparticle equation is solved by one routine, `find_qp_root`.

The pole model (`sum_over_poles.qp_energy_sop`), the quasi-boson finders
(`QPqb.solve_diag`, `largest_weight_root`, `bracketed_root`,
`qp_energy_general`), the contour deformation (`qp_energy_cd`), the analytic
continuation (`solve_qp_equation_pole_strength`), the FLEX self-energy and the
periodic root scan all bracket, refine inside the bracket and keep the largest
valid Z (or follow a reference root) through `Solvers.qp_equation.
find_qp_root`; on a self-energy that is a sum of known poles they return the
same root and Z, and they read the one start point (`qp_start`), weight test
(`quasiparticle_weight`) and selection rule (`select_qp_root`).
"""
import inspect
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import QP_POLE_OFFSET, QP_WINDOW_Z_MIN
import src.Solvers.qp_equation as qpe
from src.Solvers.qp_equation import (largest_weight_root, qp_start,
                                     quasiparticle_weight, select_qp_root,
                                     solve_qp_equation_pole_strength,
                                     solve_qp_equation_scan)
import src.SingleReference.GW.quasi_boson as qb
import src.SingleReference.GW.sum_over_poles as sop
import src.SingleReference.GW.contour_deformation as cd

EPS = np.array([-0.95, -0.62, -0.48, 0.12, 0.31, 0.77])
NOCC = 3
POLES = np.array([0.45, 0.8, 1.6])


def model(seed=7):
    """Positive pole-model amplitudes A[m, q] and the positions d[q, m] of the
    poles of Sigma^c they give (`pole_denominators`)."""
    rng = np.random.default_rng(seed)
    amp = 2e-3 * rng.random((len(POLES), len(EPS)))
    return amp, sop.pole_denominators(POLES, EPS, NOCC)


def sigma(w, amp, d):
    return float(np.sum(amp.T / (w - d)))


@pytest.mark.parametrize('p', [1, 2, 3, 4])
@pytest.mark.parametrize('xc', [0.0, 0.031])
def test_every_engine_returns_the_same_root(p, xc):
    """The pole model's global solve, the quasi-boson secular finder, the
    subspace Newton of one orbital and the pole-strength scan agree on root
    and Z."""
    amp, d = model()
    w_sop, z_sop = sop.qp_energy_sop(p, amp, POLES, EPS, NOCC,
                                     xc_correction=xc)
    w_lw, z_lw = largest_weight_root(amp.T.ravel(), d.ravel(), EPS[p] + xc)
    h, _, ok = sop.subspace_hamiltonian(np.array([[EPS[p] + xc]]),
                                        amp.T.ravel()[:, None, None], d.ravel(),
                                        np.array([[w_sop]]))
    f = lambda w: w - EPS[p] - xc - sigma(w, amp, d)
    w_ps = solve_qp_equation_pole_strength(f, EPS[p])
    assert ok
    assert abs(w_sop - w_lw) < 1e-10 and abs(z_sop - z_lw) < 1e-10
    assert abs(h[0, 0] - w_sop) < 1e-10
    assert abs(w_ps - w_sop) < 1e-6


def test_every_route_solves_through_the_one_finder(monkeypatch):
    """The pole model, the quasi-boson finders, the contour equation and the
    pointwise scan each reach `find_qp_root` once per root, and the
    quasi-boson finders are the solver's own."""
    calls = []
    real = qpe.find_qp_root

    def spy(source, a, **k):
        calls.append(type(source).__name__)
        return real(source, a, **k)

    monkeypatch.setattr(qpe, 'find_qp_root', spy)
    amp, d = model()
    sop.qp_energy_sop(2, amp, POLES, EPS, NOCC)
    largest_weight_root(amp.T.ravel(), d.ravel(), EPS[2])
    qpe.bracketed_root(amp.T.ravel(), d.ravel(), EPS[2], EPS[2])
    f = lambda w: w - EPS[2] - sigma(w, amp, d)
    solve_qp_equation_pole_strength(f, EPS[2])
    solve_qp_equation_scan(f, lambda w: 1.0 + float(
        np.sum(amp.T / (w - d) ** 2)), EPS[2])
    assert calls == ['PoleSum'] * 3 + ['SampledEquation'] * 2
    for name in ('largest_weight_root', 'bracketed_root'):
        assert getattr(qb, name) is getattr(qpe, name), name
    assert 'solve_qp_equation_scan' in inspect.getsource(cd.qp_energy_cd)
    assert 'bracketed_root' in inspect.getsource(qb.qp_energy_general)


def test_no_second_root_finder_is_left():
    """The seeded and guarded Newtons, the nearest-root scan, the fixed
    bracket, the batch scan, the rank follower and the arrowhead eigensolve
    are gone."""
    for name in ('solve_qp_equation_newton_guarded', 'newton_seed',
                 'solve_qp_equation_graphical', 'solve_qp_equation_bisection',
                 'solve_qp_equation_pole_strength_batch', 'PoleGuardRefusal',
                 'ranked_root'):
        assert not hasattr(qpe, name), name
    src = inspect.getsource(qpe)
    assert 'eigvals' not in src and 'np.linalg.eig' not in src


def test_signed_amplitudes_still_give_every_root():
    """A fitted amplitude may be negative; the bracketed solve does not need
    positive weights, and its roots solve the equation."""
    num = np.array([3e-3, -1e-4, 2e-3])
    pol = np.array([-0.9, 2.5, 1.2])
    w, z = qpe.solve_qp_equation_poles(num, pol, 0.1)
    assert abs(w - 0.1 - np.sum(num / (w - pol))) < 1e-12
    assert quasiparticle_weight(z)
    # a negative weight beside the level leaves only Z > 1: no valid weight,
    # so the root nearest the level comes back and its Z says so
    _, z = qpe.solve_qp_equation_poles(np.array([-1e-3]), np.array([0.5]), 0.1)
    assert not quasiparticle_weight(z, 0.0)


def _arrowhead_roots(num, pol, a):
    """Every real root of the pole-sum equation, as eigenvalues of the
    arrowhead matrix: the reference the graphical scan must reproduce."""
    n = num.size
    arrow = np.zeros((n + 1, n + 1))
    arrow[0, 0], arrow[0, 1:], arrow[1:, 0] = a, num, 1.0
    arrow[1:, 1:] = np.diag(pol)
    lam = np.linalg.eigvals(arrow)
    return np.sort(lam.real[np.abs(lam.imag) < 1e-10])


def test_an_uncoupled_pole_is_no_pole_of_the_equation():
    """A pole of zero weight (a span-cut orbital, a symmetry-forbidden
    pair) beside the quasiparticle gave the arrowhead route a root ON that
    pole and 0/0 in its slope, so Z = NaN. The bracketed solve drops it and
    returns the root and Z of the equation without it, with no warning."""
    num = np.array([3e-3, 0.0, 2e-3, 0.0])
    pol = np.array([-0.9, 0.1, 1.2, 0.1])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        w, z = qpe.solve_qp_equation_poles(num, pol, 0.1)
    w_ref, z_ref = qpe.solve_qp_equation_poles(num[[0, 2]], pol[[0, 2]], 0.1)
    assert np.isfinite(z) and quasiparticle_weight(z)
    assert (w, z) == (w_ref, z_ref)
    assert abs(w - 0.1 - np.sum(num / (w - pol))) < 1e-12


def test_two_crossings_between_neighbouring_poles(monkeypatch):
    """Mixed-sign amplitudes: f is +inf at both ends of the interval between
    a negative-weight pole and the positive-weight pole above it, and dips
    below zero in between, so the interval holds two roots and no sample sees
    the sign change. The turning point, found by Newton on f', splits it;
    with no interior samples at all, both roots are still found."""
    a = 0.0
    num = np.array([-1e-8, 1e-8])
    pol = np.array([a - 1.1e-3, a - 0.9e-3])
    monkeypatch.setattr(qpe, 'QP_POLE_INTERVAL_SAMPLES', 0)
    source = qpe.PoleSum(num, pol, a, 1e-13)
    found = [w for w, _ in source.solve(0)]
    ref = _arrowhead_roots(num, pol, a)
    ref = ref[(ref > pol[0]) & (ref < pol[1])]
    assert ref.size == 2
    assert np.allclose(np.sort(found), ref, rtol=0, atol=1e-12)


def test_the_signed_bounds_hold(monkeypatch):
    """Every bound on Z the signed pole sum skips intervals by is a bound:
    f' - 1 sampled densely in each certified interval stays above it, and g'
    (f without the bracketing poles) above its own (`_interval_bounds`,
    `_abel_bounds`, `_far_slack`). The model is the reproducer's: 12 poles,
    a fifth of the amplitudes negative."""
    rng = np.random.default_rng(3)
    nmo, nocc = 96, 12
    eps = np.sort(np.concatenate([rng.uniform(-30, -0.3, nocc),
                                  rng.uniform(0.02, 40, nmo - nocc)]))
    amp = rng.uniform(1e-6, 1e-3, (12, nmo))
    amp *= np.where(rng.random(amp.shape) < 0.2, -0.2, 1.0)
    d = sop.pole_denominators(np.sort(rng.uniform(0.2, 3.0, 12)), eps, nocc)
    ps = qpe.PoleSum(amp.T.ravel(), d.ravel(), eps[nocc - 1])
    num, pol = ps.num, ps.pol
    bound, net, certified = qpe._interval_bounds(num, pol, ps.a)
    for m in np.flatnonzero(certified):
        x = pol[m] + (pol[m + 1] - pol[m]) * np.linspace(1e-4, 1 - 1e-4, 64)
        terms = num[None, :] / (x[:, None] - pol[None, :]) ** 2
        assert terms.sum(1).min() > -1.0, m            # f' > 0: one root
        rest = np.delete(terms, [m, m + 1], axis=1).sum(1)
        assert rest.min() >= -net[m] - 1e-9 * abs(net[m]), m
        w, z = ps.solve(m)[0]
        assert z <= bound[m] * (1 + 1e-12), m


def test_the_largest_weight_is_global_on_signed_sums():
    """The largest valid Z over every root, as the arrowhead's eigenvalues
    give them all, also where no root is a quasiparticle (the exhaustive
    pass): 40 random signed pole sums."""
    rng = np.random.default_rng(5)
    for _ in range(40):
        nmo, nocc = int(rng.integers(6, 40)), 3
        eps = np.sort(np.concatenate([rng.uniform(-3, -0.2, nocc),
                                      rng.uniform(0.0, 4, nmo - nocc)]))
        amp = rng.uniform(1e-5, 5e-3, (6, nmo))
        amp *= np.where(rng.random(amp.shape) < 0.2, -0.3, 1.0)
        num = amp.T.ravel()
        pol = sop.pole_denominators(np.sort(rng.uniform(0.2, 3, 6)), eps,
                                    nocc).ravel()
        a = float(eps[int(rng.integers(nmo))])
        roots = _arrowhead_roots(num, pol, a)
        with np.errstate(all='ignore'):
            z = np.array([1.0 / qpe.secular(num, pol, a, w)[1]
                          for w in roots])
        k = select_qp_root(roots, z)
        w, zw = qpe.solve_qp_equation_poles(num, pol, a)
        if k is not None:
            assert abs(w - roots[k]) < 1e-9 and abs(zw - z[k]) < 1e-7


def test_no_dense_eigensolve(monkeypatch):
    """The quasiparticle root of a pole sum costs O(poles) per
    refinement step and bound, not an eigensolve of the (poles + 1)^2
    arrowhead; 12 poles x 2048 orbitals (24576 poles of Sigma^c, a fifth of
    the amplitudes negative) is solved without one."""
    def refuse(*a, **k):
        raise AssertionError('dense eigensolve on the quasiparticle path')
    monkeypatch.setattr(np.linalg, 'eigvals', refuse)
    monkeypatch.setattr(np.linalg, 'eig', refuse)
    rng = np.random.default_rng(0)
    nmo, nocc = 2048, 256
    eps = np.sort(np.concatenate([rng.uniform(-30, -0.3, nocc),
                                  rng.uniform(0.02, 40, nmo - nocc)]))
    poles = np.sort(rng.uniform(0.2, 3.0, 12))
    amp = rng.uniform(1e-6, 1e-3, (12, nmo))
    amp *= np.where(rng.random(amp.shape) < 0.2, -0.2, 1.0)
    w, z = sop.qp_energy_sop(nocc - 1, amp, poles, eps, nocc)
    d = sop.pole_denominators(poles, eps, nocc)
    assert abs(w - eps[nocc - 1] - sigma(w, amp, d)) < 1e-10
    assert np.isfinite(z)


def test_one_start_point():
    """The contour driver's start points are the solver's `qp_start`."""
    states = [1, 2, 3, 4]
    starts = cd.qp_starts(EPS, NOCC, states)
    assert starts.tolist() == [qp_start(EPS, p, NOCC) for p in states]
    assert qp_start(EPS, 2, NOCC) == EPS[2] + QP_POLE_OFFSET
    assert qp_start(EPS, 2, NOCC, 0.0) == EPS[2]


def test_one_weight_rule():
    """(z_min, 1], finite: the bounds every route tests against."""
    assert quasiparticle_weight(0.9) and quasiparticle_weight(1.0)
    assert not quasiparticle_weight(1.0 + 1e-12)
    assert not quasiparticle_weight(QP_WINDOW_Z_MIN)
    assert not quasiparticle_weight(np.nan)
    assert select_qp_root([0.1, 0.2, 0.3], [0.04, 0.6, 0.3]) == 1
    assert select_qp_root([0.1, 0.2, 0.3], [0.04, 0.6, 0.3], near=0.31) == 2
    assert select_qp_root([0.1], [0.01]) == 0
    assert select_qp_root([0.1], [0.01], z_min=QP_WINDOW_Z_MIN) is None
    assert not quasiparticle_weight(0.4) and quasiparticle_weight(0.4, 0.0)


def test_a_converged_root_is_returned_with_its_weight():
    """A root is returned as it is, whatever its Z (here 2, no root of a
    Dyson equation), and the weight test says it is no quasiparticle; no
    linearized energy of another approximation stands in for it."""
    # Sigma rising through the root: dSigma/dw = +0.5, Z = 2
    f = lambda w: w - EPS[3] - (0.5 * (w - EPS[3]) - 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        w, z = solve_qp_equation_scan(f, lambda w: 0.5, EPS[3])
    assert abs(w - (EPS[3] - 0.02)) < 1e-10 and abs(z - 2.0) < 1e-12
    assert not quasiparticle_weight(z)


def test_a_pole_hidden_beside_the_root_is_found():
    """A grid cell holding the quasiparticle and a pole of Sigma just above
    it has one sign at both ends; its secant falls below
    QP_SCAN_SECANT_MIN, the cell is halved and the quasiparticle found, and
    the sign change across the pole is told from a root by |f| growing:
    exactly the two roots come back, never the pole."""
    a, p0 = 0.0, 0.0033
    sig = lambda w: 1e-6 / (w - p0)
    f = lambda w: w - a - 0.003 - sig(w)
    fp = lambda w: 1.0 + 1e-6 / (w - p0) ** 2
    grid = np.linspace(a - 0.5, a + 0.5, qpe.QP_GRAPHICAL_N_OMEGA)
    cell = (grid[:-1] < 0.003) & (grid[1:] > p0)
    assert cell.any() and f(grid[:-1][cell][0]) * f(grid[1:][cell][0]) > 0
    # the pole splits the level 0.003 into two roots, Z 0.574 and 0.426
    roots = np.sort(np.roots([1.0, -(0.003 + p0), 0.003 * p0 - 1e-6]))
    w, z = solve_qp_equation_scan(f, fp, a)
    assert abs(w - roots[0]) < 1e-12 and abs(z - 1.0 / fp(roots[0])) < 1e-9
    assert z > QP_WINDOW_Z_MIN
    source = qpe.SampledEquation(f, fp)
    found = [rz for br in source.brackets(a - 0.5, a + 0.5)[0]
             for rz in source.solve(br)]
    assert np.allclose(sorted(r for r, _ in found), roots, atol=1e-12)


def test_qp_energy_general_follows_through_the_one_finder():
    """qp_energy_general takes its root from the solver's pole-sum finders
    and warns at a followed root below the one threshold, no literal cut."""
    src = inspect.getsource(qb.qp_energy_general)
    assert 'bracketed_root' in src and 'largest_weight_root' in src
    assert 'QP_WINDOW_Z_MIN' in src and '0.2' not in src
    assert 'eigh(H)' not in src
