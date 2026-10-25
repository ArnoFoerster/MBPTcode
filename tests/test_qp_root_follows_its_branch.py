"""Along a walk the quasiparticle root is followed, not re-decided.

The equation is
solved in full at every geometry; the largest Z selects the root at the
reference, and every later geometry keeps the root continuing the one the
previous geometry kept (`Solvers.qp_equation.find_qp_root(follow=)`): for a
pole sum the root between the same two LABELLED poles, for a sampled
self-energy the root of the scan nearest the previous one. Where the largest
weight moves to another root the followed one is kept, even below
QP_WINDOW_Z_MIN, and the caller is told (`QPRoot.flipped`); nothing is
refused. Re-deciding by weight at every step jumps branches where a satellite
crosses the quasiparticle: the surface steps there.

The model: Sigma = 0.05/(w + 2) + 1e-3/(w - s), a satellite pole s walked
down through the level a = 0 in steps of 2 mHa, standing in for a geometry
coordinate. The quasiparticle branch passes it by an avoided crossing (gap
about 2 sqrt(1e-3) = 63 mHa), its weight handed to the other branch.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import QP_WINDOW_Z_MIN
import src.Solvers.qp_equation as qpe
from src.Solvers.qp_equation import PoleSum, SampledEquation, find_qp_root
import src.gradients.gwbse.qp_space_time as qst

A = 0.0
STEPS = 0.1 - 2e-3 * np.arange(101)         # the satellite pole, 0.1 .. -0.1


def poles(s):
    return np.array([0.05, 1e-3]), np.array([-2.0, s])


def source(kind, s):
    num, pol = poles(s)
    if kind == 'poles':
        return PoleSum(num, pol, A, 1e-13)
    f = lambda w: w - A - float(np.sum(num / (w - pol)))
    fp = lambda w: 1.0 + float(np.sum(num / (w - pol) ** 2))
    return SampledEquation(f, fp, tol=1e-12)


def walk(kind, follow, steps=STEPS):
    """[QPRoot] along the walk: re-decided by the largest Z, or the previous
    step's root followed (its labels with it)."""
    out, prev = [], None
    for s in steps:
        if not follow or prev is None:
            got = find_qp_root(source(kind, s), A)
        else:
            got = find_qp_root(source(kind, s), A,
                               follow=(prev.w, prev.labels))
        prev = got
        out.append(got)
    return out


@pytest.mark.parametrize('kind', ['poles', 'scan'])
def test_redeciding_by_weight_jumps_branches(kind):
    """The largest-weight rule at every step changes root where the two
    branches share the weight: one step moves it by the crossing's gap."""
    path = walk(kind, follow=False)
    jumps = np.abs(np.diff([g.w for g in path]))
    assert jumps.max() > 0.03
    assert jumps.max() > 10 * np.median(jumps)


@pytest.mark.parametrize('kind', ['poles', 'scan'])
def test_the_followed_root_stays_on_its_branch_whatever_its_weight(kind):
    """Followed, the root moves continuously (no step above 2 mHa, the
    pole's own step) through the whole walk while its Z falls far below
    QP_WINDOW_Z_MIN; nothing is refused. From the first geometry where the
    other branch carries more weight, the root says so (`flipped`, with the
    largest weight beside it), and the walk goes on on its branch."""
    path = walk(kind, follow=True)
    roots = np.array([g.w for g in path])
    z = np.array([g.z for g in path])
    assert np.abs(np.diff(roots)).max() < 2e-3
    assert np.all(np.diff(z) < 0)
    assert path[0].z > 0.85 and z[-1] < 0.1
    assert path[0].followed is None and all(g.followed for g in path[1:])
    tol = 1e-10
    flipped = np.array([g.flipped(tol) for g in path])
    first = int(np.argmax(flipped))
    assert flipped[first:].all() and not flipped[:first].any()
    # it flips where the other branch overtakes it, not where Z crosses 1/2
    assert path[first].largest[1] > path[first].z
    assert path[first - 1].largest == (path[first - 1].w, path[first - 1].z)
    assert not path[-1].quasiparticle


def test_both_sources_follow_the_same_branch():
    """The labelled poles and the pointwise scan follow one branch to the
    same roots and weights, step by step, through the crossing."""
    got = {k: np.array([(g.w, g.z) for g in walk(k, follow=True)])
           for k in ('poles', 'scan')}
    assert np.allclose(got['poles'], got['scan'], rtol=0, atol=1e-10)


def test_a_pole_sum_follows_its_labelled_poles_not_the_nearest_energy():
    """One long step that carries the satellite pole across the root: the
    labelled poles keep the root between the same two poles (the adiabatic
    branch, pushed below the satellite with little weight); by energy, the
    previous root lies above the pole now and the root there is taken. The
    labels are each pole's index in the weights as given."""
    ref = find_qp_root(source('poles', 0.1), A)
    assert ref.labels == (0, 1)              # between -2 and the satellite
    far = source('poles', -0.05)
    by_label = find_qp_root(far, A, follow=(ref.w, ref.labels))
    by_energy = find_qp_root(far, A, follow=(ref.w, None))
    assert -2.0 < by_label.w < -0.05 < by_energy.w
    assert by_label.labels == (0, 1) and by_energy.labels == (1, None)
    assert by_label.z < QP_WINDOW_Z_MIN < by_energy.z
    assert by_label.flipped(1e-10) and not by_energy.flipped(1e-10)


def test_the_root_kept_does_not_depend_on_where_it_was_followed_from():
    """Every root is refined from its own bracket, never from the previous
    root: following from anywhere on the branch returns the same bits."""
    for kind in ('poles', 'scan'):
        here = source(kind, 0.02)
        ref = find_qp_root(here, A)
        for w in (ref.w, ref.w + 1e-3, ref.w - 2e-3):
            got = find_qp_root(source(kind, 0.02), A, follow=(w, ref.labels))
            assert got.w == ref.w and got.z == ref.z


def test_a_branch_that_has_gone_is_picked_again_by_weight():
    """A labelled pole that is no pole here (its weight zero, or merged
    away) or labels out of order: the branch is gone, the largest weight is
    kept again (followed False, flipped), and nothing is refused."""
    num, pol = np.array([0.05, 0.0]), np.array([-2.0, 0.1])
    got = find_qp_root(PoleSum(num, pol, A, 1e-13), A, follow=(0.01, (0, 1)))
    assert got.followed is False and got.flipped(1e-10)
    assert (got.w, got.z) == got.largest
    ref = find_qp_root(PoleSum(num, pol, A, 1e-13), A)
    assert got.w == ref.w
    swapped = find_qp_root(source('poles', 0.1), A, follow=(0.01, (1, 0)))
    assert swapped.followed is False


def test_the_entry_points_warn_and_hand_the_branch_over():
    """Without an `out` dict a flipped branch is a RuntimeWarning (a seed
    below the satellite pole, by energy); with one the branch is handed over
    and nothing is warned."""
    ref = find_qp_root(source('poles', 0.1), A)
    num, pol = poles(-0.05)
    with pytest.warns(RuntimeWarning, match='branch is kept'):
        w, z = qpe.bracketed_root(num, pol, A, -0.06)
    assert w < -0.05 and z < QP_WINDOW_Z_MIN
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        w2, z2 = qpe.solve_qp_equation_poles(num, pol, A, tol=1e-13,
                                             follow=(ref.w, ref.labels),
                                             out=out)
    assert out['flipped'] and out['followed'] and out['labels'] == (0, 1)
    assert z2 < QP_WINDOW_Z_MIN


def test_the_chain_follows_the_last_accepted_branch():
    """The kernels follow the branch the conventions hand them
    (`followed_branch`): the last accepted geometry's root and labels, else
    the frozen reference root by energy, else nothing before the reference
    solve; a scalar or a mapping w0, as the conventions hand them."""
    assert qst.frozen_root(None, 3) is None
    assert qst.frozen_root(-0.4, 3) == -0.4
    assert qst.frozen_root({3: -0.4}, 3) == -0.4
    assert qst.frozen_root({2: -0.4}, 3) is None
    assert qst.followed_branch(None, None, 3) is None
    assert qst.followed_branch({}, {3: -0.4}, 3) == (-0.4, None)
    assert qst.followed_branch({3: (-0.41, [7, 9])}, {3: -0.4}, 3) == (
        -0.41, (7, 9))
    assert qst.followed_branch({3: (-0.41, None)}, None, 3) == (-0.41, None)
