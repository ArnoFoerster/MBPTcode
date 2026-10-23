"""Gates for `outside_scissor='curve'`: the frozen Delta(eps) beyond the window.

The frozen scissor with the default `outside_scissor='tiers'` lends every
orbital beyond the explicit window the correction of the explicit orbital
nearest in energy, a constant. 'curve' gives each of them Delta(eps_p) of a
frozen `OutsideCurve` per side, placed at the reference geometry on
the analytic continuation of every orbital and anchored on the window edge's
root; it reads
the orbital's current eigenvalue, so the force carries dDelta/deps.

Water/cc-pVDZ, the record's RHF, the default window (3-6): the occupied curve
runs from +0.98 eV at the edge to +7.7 eV on O 1s, the virtual one from
-0.32 eV at the edge to -3 eV, where the tiers lend +0.98 and -0.32 eV.

The curve is a natural cubic spline, C2 at every knot and beyond them: at
the reference geometry a one-member bin puts a knot on that member's
eigenvalue, where a C1 interpolant's curvature jump made the surface itself
C1 (a central difference O(h) there, the production walk gate's miss).
"""
import copy

import numpy as np
import pytest

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.GW.qp_states import (CurveOvershoots, OutsideCurve,
                                              curve_sides, fit_outside_curve)
from src.gradients.gwbse import qp_set
from src.gradients.evaluation import GeometryCache
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.properties.excitations import qp_bookkeeping

from test_outside_scissor import (FD_STEP, FD_TOL, RECORD, at_coords,
                                  molecule, scf_factory)
from tests.isdf_test_grid import g1_numerics
from tests.pins import assert_same

#: The record's displaced water, where the force is differenced.
DISPLACED = RECORD['surfaces']['water']['ExcitedStateChain[singlet]'][
    'geometries']['R0+0.05x_atom0']['atom_coords_bohr']


@pytest.fixture(scope='module')
def water():
    """The record's water, with the mean field every chain below reads."""
    mol = molecule('water')
    return mol, scf_factory(mol)


@pytest.fixture(scope='module')
def curved(water):
    """A curve chain evaluated once at the reference geometry."""
    mol, mf = water
    chain = ExcitedStateChain(
        mol, scf_factory, mf=mf,
        numerics=g1_numerics(mol, Numerics(qp={'outside_scissor': 'curve'})))
    shared, solution = chain.forward(mol, mf)
    return chain, solution, shared


# ------------------------------------------------------------- the curve alone
def test_the_curve_starts_on_the_anchor_and_continues_its_tangent():
    """Delta(eps_edge) is the edge's own correction; beyond the knots the
    curve is the end tangent, so value and slope are continuous there."""
    eps = np.array([-20.0, -10.0, -5.0, -1.0, 0.5, 2.0, 4.0, 8.0])
    ac = {5: 0.1, 6: 0.3, 7: 0.2}
    z = {5: 0.9, 6: 0.8, 7: 0.95}
    curve = fit_outside_curve(eps, 'virtual', 4, -0.05, (5, 6, 7), ac, z)
    assert curve.knots[0] == (0.5, -0.05)
    assert [k[0] for k in curve.knots] == sorted(k[0] for k in curve.knots)
    delta, slope = curve(np.array([0.5, 8.0]))
    assert delta[0] == -0.05 and delta[1] == pytest.approx(0.2)
    h = 1e-7
    for e in (0.5, 8.0):
        lo, _ = curve(np.array([e - h]))
        hi, _ = curve(np.array([e + h]))
        assert (hi - lo)[0] / (2 * h) == pytest.approx(curve(np.array([e]))[1][0],
                                                       rel=1e-5, abs=1e-9)


def test_the_curve_is_c2_at_every_knot():
    """The one-sided second differences meet at every knot and at both ends,
    where the curve continues as its tangent.

    FAILS on the PCHIP interpolant the curve had: its curvature jumps at the
    inner knots by about its own size (-1.87 vs -0.85, 0.90 vs 4.16 here).
    """
    curve = OutsideCurve('virtual', (1, 2, 3), 0,
                         ((0.0, 0.0), (0.1, 0.01), (0.2, 0.012), (0.3, 0.03)))
    h = 1e-4
    f = lambda e: curve(np.array([e]))[0][0]
    scale = 0.0
    jumps = []
    for k, _ in curve.knots:
        left = (f(k) - 2 * f(k - h) + f(k - 2 * h)) / h ** 2
        right = (f(k + 2 * h) - 2 * f(k + h) + f(k)) / h ** 2
        scale = max(scale, abs(left), abs(right))
        jumps.append(abs(right - left))
    assert max(jumps) < 1e-2 * scale, jumps


def test_a_curve_that_overshoots_at_a_member_is_refused():
    """A spline swinging past its two bracketing knots by more than
    OUTSIDE_CURVE_OVERSHOOT_EV at a member's eigenvalue is refused at build
    time: a C2 interpolant gives up the no-overshoot of a shape-preserving
    one. Orbitals 2 and 6 are members between knots, where the spline reads
    -2.4 and -3.8 Ha."""
    eps = np.array([-1.0, 0.0, 0.5, 1.0, 1.2, 1.4, 2.0, 3.0])
    ac = {3: 0.0, 4: 2.0, 5: 0.0, 7: 2.0}
    z = {p: 0.9 for p in ac}
    members = (2, 3, 4, 5, 6, 7)
    with pytest.raises(CurveOvershoots, match='overshoots its knot values'):
        fit_outside_curve(eps, 'virtual', 1, 0.0, members, ac, z,
                          bin_ev=0.1 * HARTREE_TO_EV)
    # with no member in the swing the same knots are a curve
    curve = fit_outside_curve(eps, 'virtual', 1, 0.0, (3, 4, 5, 7), ac, z,
                              bin_ev=0.1 * HARTREE_TO_EV)
    assert curve.overshoot(eps[[3, 4, 5, 7]]) < 1e-12   # on its knots


def test_a_probe_the_continuation_cannot_judge_is_dropped():
    """A failed continuation, a Z outside (0, 1] and a probe in the edge's
    own bin place no knot, and the record says why."""
    eps = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 3.01])
    ac = {2: np.nan, 3: 0.4, 4: 0.5, 5: 0.7}
    z = {2: 0.9, 3: 1.04, 4: 0.9, 5: 0.9}
    curve = fit_outside_curve(eps, 'virtual', 1, 0.0, (2, 3, 4, 5), ac, z)
    assert 'continuation failed' in curve.probes[2]['dropped']
    assert 'outside (0, 1]' in curve.probes[3]['dropped']
    assert curve.probes[4]['used'] and curve.probes[5]['used']
    # 4 and 5 share a bin: one knot at their mean eps, their median shift
    assert curve.knots[0] == (0.0, 0.0) and len(curve.knots) == 2
    assert curve.knots[1] == pytest.approx((3.005, 0.6))
    near = fit_outside_curve(eps, 'virtual', 1, 0.0, (2,), {2: 0.3},
                             {2: 0.9}, bin_ev=2.0 * HARTREE_TO_EV)
    assert near.probes[2]['dropped'] == 'within the window edge bin'
    assert near.knots == ((0.0, 0.0),)
    flat, slope = near(np.array([-3.0, 5.0]))
    assert np.all(flat == 0.0) and np.all(slope == 0.0)


def test_the_sides():
    """Members lie beyond the outermost explicit orbital of their side."""
    eps = np.arange(10, dtype=float)
    sides = curve_sides(eps, 4, explicit=(2, 3, 5), outside=(0, 1, 4, 6, 7,
                                                             8, 9))
    assert sides['occupied'] == (2, (0, 1))
    assert sides['virtual'] == (5, (6, 7, 8, 9))


# ------------------------------------------------------------- on the chain
def test_the_curve_carries_the_members_and_meets_the_edge(water, curved):
    """eps^QP = eps + Delta(eps) on every member, the explicit roots as the
    tiers have them, and Delta(eps_edge) equal to the edge's own correction:
    the window boundary has no jump."""
    mol, mf = water
    chain, _, shared = curved
    tiers = ExcitedStateChain(mol, scf_factory, mf=mf, numerics=g1_numerics(mol))
    shared_t = tiers.forward(mol, mf)[0]
    eps, eps_qp, eps_qp_t = shared.eps, shared.eps_qp, shared_t.eps_qp
    explicit = [int(p) for p in chain.resolved.qp_set]
    # two chains solve the explicit roots apart: to a few ulp, not bitwise
    assert_same(eps_qp[explicit], eps_qp_t[explicit], quantity='energy',
                label='explicit roots')
    assert set(chain.resolved.outside_curve) == {'occupied', 'virtual'}
    on_curve = set()
    for curve in chain.resolved.outside_curve.values():
        idx = np.asarray(curve.members, int)
        on_curve |= set(curve.members)
        assert np.array_equal(eps_qp[idx], eps[idx] + curve(eps[idx])[0])
        edge = curve.anchor
        assert curve(eps[[edge]])[0][0] == pytest.approx(
            eps_qp[edge] - eps[edge], rel=0.0, abs=1e-15)
    assert not on_curve & set(chain.resolved.outside_shift)
    assert on_curve | set(chain.resolved.outside_shift) | set(explicit) == set(
        range(len(eps)))


def test_tiers_place_no_curve_and_solve_no_continuation(water, monkeypatch):
    """The default never calls the continuation and keeps every outside
    orbital on a tier: the pinned gates of tests/test_outside_scissor.py
    hold its numbers bitwise."""
    mol, mf = water

    def refuse(*args, **kwargs):
        raise AssertionError('tiers solved a continuation')

    monkeypatch.setattr(qp_set, 'ac_quasiparticle_shifts', refuse)
    chain = ExcitedStateChain(mol, scf_factory, mf=mf, numerics=g1_numerics(mol))
    assert chain.outside_scissor == 'tiers'
    chain.excitation(mol, mf)
    assert chain.resolved.outside_curve == {}
    assert chain.conventions.curve_record() is None
    assert 'outside_scissor' not in chain.conventions.outside_record()


def test_the_curve_is_frozen_and_its_force_follows_the_energy(water, curved):
    """The knots do not move with the geometry, and dE/dR at a displaced
    geometry is the finite difference of E on that frozen curve."""
    mol, _ = water
    chain, _, _ = curved
    knots = {s: c.knots for s, c in chain.resolved.outside_curve.items()}
    moved = at_coords(mol, DISPLACED)
    grad, _, diags = chain.total_gradient(moved)
    assert {s: c.knots for s, c in chain.resolved.outside_curve.items()} == knots
    assert diags['outside_scissor'] == 'curve'
    worst = 0.0
    for ia in range(mol.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                step = np.zeros((mol.natm, 3))
                step[ia, x] = k * FD_STEP
                m = moved.copy()
                m.set_geom_(moved.atom_coords() + step, unit='Bohr')
                m.build(False, False)
                v.append(chain.energy(m)[0])
            fd = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * FD_STEP)
            worst = max(worst, abs(fd - grad[ia, x]))
    assert worst / np.abs(grad).max() < FD_TOL
    assert {s: c.knots for s, c in chain.resolved.outside_curve.items()} == knots


def test_the_slope_term_is_what_the_force_needs(water, curved):
    """Without dDelta/deps on the members' seeds the force misses a term the
    finite difference sees: on water the slope's part is 3.3e-5 of the force,
    against 3.5e-8 for the whole force's finite-difference residual."""
    mol, _ = water
    chain, _, _ = curved
    moved = at_coords(mol, DISPLACED)
    grad = chain.total_gradient(moved)[0]
    flat = copy.copy(chain)
    # another surface: its own evaluations, not the chain's cached force
    flat._evaluations = GeometryCache()
    full = flat.quasiparticles._curve_at
    flat.quasiparticles._curve_at = lambda eps: [(i, d, 0.0 * s) for i, d, s in full(eps)]
    missing = np.abs(grad - flat.total_gradient(moved)[0]).max()
    assert missing / np.abs(grad).max() > 10 * FD_TOL


def test_refreeze_rebuilds_the_curve(water, curved):
    """A refrozen chain places its own curve at the new geometry and records
    the knots it replaced."""
    mol, _ = water
    chain, _, _ = curved
    fresh = chain.refreeze(at_coords(mol, DISPLACED))
    assert fresh.outside_scissor == 'curve'
    assert fresh.resolved.outside_curve == {}
    fresh.excitation()
    for side, curve in fresh.resolved.outside_curve.items():
        assert curve.knots != chain.resolved.outside_curve[side].knots
    record = fresh.conventions.curve_record()
    assert set(record['knots_before_eV']) == set(chain.resolved.outside_curve)


def test_the_record_names_the_mode_and_the_curve(water, curved):
    """qp_bookkeeping carries the mode, the knots and every probe."""
    mol, _ = water
    chain, _, _ = curved
    book = qp_bookkeeping(chain, mol)
    assert book['outside_scissor'] == 'curve'
    curves = book['scissor_curve']['curves']
    assert set(curves) == {'occupied', 'virtual'}
    for side, rec in curves.items():
        assert rec['knots_eV'][0][1] == pytest.approx(
            chain.resolved.outside_curve[side].knots[0][1] * HARTREE_TO_EV)
        assert set(rec['probes']) <= set(rec['members'])


def test_the_orbital_mode_is_retired(water):
    """outside_scissor='orbital' (each member's own continuation correction
    frozen as a number) is retired: the curve and the tiers remain."""
    mol, mf = water
    with pytest.raises(ValueError, match="'orbital' not in"):
        ExcitedStateChain(
            mol, scf_factory, mf=mf,
            numerics=g1_numerics(mol, Numerics(qp={'outside_scissor': 'orbital'})))


def test_an_unknown_curve_mode_is_refused(water):
    """A shape of the frozen scissor that is not one of the modes."""
    mol, mf = water
    with pytest.raises(ValueError, match='not in'):
        ExcitedStateChain(
            mol, scf_factory, mf=mf,
            numerics=g1_numerics(mol, Numerics(qp={'outside_scissor': 'spline'})))


def test_an_unused_curve_class_is_a_constant():
    """A side whose every probe was dropped keeps the edge's shift: tiers."""
    curve = OutsideCurve('occupied', (0,), 1, ((-1.0, 0.25),))
    delta, slope = curve(np.array([-3.0, -2.0]))
    assert np.all(delta == 0.25) and np.all(slope == 0.0)
