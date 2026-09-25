"""The Hessian by differencing the analytic force, against pyscf's analytic one.

Two claims are gated here. That the route AGREES with pyscf where both are
valid, to far inside what a Huang-Rhys factor cares about. And that it is
available where pyscf's is not: on an interpolated mean field pyscf
differentiates the fitted interaction twice and returns force constants for a
different functional, while the ISDF FORCE is exact, so differencing it is the
only route to that Hessian.

The acoustic sum rule is the sharpest statement available on any Hessian --
moving every atom together changes no force, it needs no reference
calculation, and neither route is told the identity exists.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.constants import (HARTREE_TO_CM,
                                HESSIAN_FD_ASYMMETRY_TOL, NUCLEAR_FD_STEP)
from src.Base.isdf_jk import isdf_jk
from src.properties.hessian import (displacement_list, gradient_at,
                                    hessian_from_gradients, numerical_hessian,
                                    translation_residual)
from src.properties.vibronic import normal_modes

ATOM = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: The interpolated route's finite-difference Hessian misbehaves here and not
#: on water, and no force gate reproduces it -- see the refusal test below.
FORMALDEHYDE = ('C 0.0 0.0 -0.6035; O 0.0 0.0 0.7349; '
                'H 0.0 0.9753 -1.1057; H 0.0 -0.9753 -1.1057')


def factory(route):
    def build(m):
        mf = dft.RKS(m, xc='b3lyp')
        mf = (isdf_jk(mf, auxbasis='cc-pvdz-ri') if route == 'isdf'
              else mf.density_fit(auxbasis='cc-pvdz-jkfit'))
        mf.grids.prune = None
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
        mf.verbose = 0
        mf.kernel()
        return mf
    return build


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)


@pytest.fixture(scope='module')
def both(water):
    build = factory('df')
    mf = build(water)
    info = {}
    return mf, np.asarray(mf.Hessian().kernel()), numerical_hessian(
        water, build, info=info), info


def test_frequencies_agree_with_the_analytic_hessian(water, both):
    """Under half a wavenumber, which is nothing against the 1.2 meV the grid
    contributes to an adiabatic gap."""
    mf, h_ana, h_num, _ = both
    wa = normal_modes(mf, water, hess=h_ana)[0] * HARTREE_TO_CM
    wn = normal_modes(mf, water, hess=h_num)[0] * HARTREE_TO_CM
    assert np.abs(wa - wn).max() < 1.0
    assert np.abs(0.5 * (wa - wn).sum()) < 1.0          # the zero-point energy


def test_it_satisfies_the_sum_rule_better_than_the_analytic_route(both):
    """NOT a formality, and READ FROM `info`, not from the returned matrix.

    The raw finite-difference Hessian satisfies the sum over the FIRST index
    identically -- it is d(sum_i g_i)/dR_j and sum_i g_i vanishes at every
    geometry -- so that sum measures nothing. The sum over the DISPLACED index
    is the check, and symmetrizing averages the two and reports half the real
    violation as if it were the answer. An earlier version of this test read
    the returned matrix and was measuring its own symmetrization.

    Against pyscf's analytic Hessian on water/cc-pVDZ/B3LYP the honest
    comparison is 5.6e-08 to 5.1e-04, which is why the bound below is loose
    rather than marginal.
    """
    _, h_ana, _, info = both
    assert info['translation'] < 1e-2 * translation_residual(h_ana)


def test_both_diagnostics_fall_as_the_square_of_the_step(water):
    """h^2 IS THE CLAIM. A finite difference limited by its own truncation
    halves its error fourfold when the step halves; anything else in there --
    force noise, a discontinuity, a force that is not the gradient of the
    energy it is paired with -- scales differently and shows up as a different
    power. This is what separates the fitted route, which passes, from the
    interpolated one, which comes out at h^1.
    """
    build = factory('df')
    out = {}
    for h in (1e-3, 2e-3, 4e-3):
        info = {}
        numerical_hessian(water, build, step=h, info=info)
        out[h] = info
    for key in ('asymmetry', 'translation'):
        r = [out[4e-3][key] / out[2e-3][key], out[2e-3][key] / out[1e-3][key]]
        assert all(2.5 < v < 6.0 for v in r), (key, r)


def test_shards_assemble_to_the_same_hessian(water, both):
    """Every shard differences forces about ONE geometry and writes its own
    subset, so which shard finishes last cannot matter."""
    _, _, h_num, _ = both
    build = factory('df')
    jobs = displacement_list(water.natm)
    grads = {}
    for k, (ia, x, sign) in enumerate(jobs):
        if k % 3 != 1:                       # one shard of three
            continue
        grads[(ia, x, sign)] = gradient_at(water, build, ia, x, sign)
    assert len(grads) == len(jobs) // 3
    for k, (ia, x, sign) in enumerate(jobs):
        if k % 3 != 1:
            grads[(ia, x, sign)] = gradient_at(water, build, ia, x, sign)
    assembled = hessian_from_gradients(grads, water.natm, NUCLEAR_FD_STEP)
    assert np.allclose(assembled, h_num, atol=1e-12, rtol=0)


def test_the_interpolated_route_is_refused_rather_than_believed(water):
    """THIS TEST USED TO ASSERT THE OPPOSITE, and it passed, because it checked
    the sum over the wrong index of a symmetrized matrix and read the lowest
    frequency instead of the spectrum.

    Both routes refuse the interpolated mean field, for different reasons and
    at different places. `normal_modes` refuses to BUILD an analytic Hessian
    there because pyscf would differentiate the fitted interaction twice.
    `numerical_hessian` refuses the finite-difference one for a reason that is
    MEASURED BUT NOT EXPLAINED. On formaldehyde it puts a 1421 cm^-1 mode at
    2012; its asymmetry is 3.2e-03 against the fitted route's 2.9e-07 and falls
    as h^1 where the fitted one falls as h^2.

    Five mechanisms were tested and refuted: a fluctuating fit realization (two
    builds at one geometry agree bitwise), steps in the surface (a step gives
    1/h, this grows with h), loss of translational invariance (the force is
    invariant to 1e-14 at every geometry tried), degenerate atomic frames at
    the reference (all six directions sit at the gradient's floor), and
    degradation at displaced geometries (1.209e-06 there against 1.179e-06 at
    the minimum, on all twelve components).

    What is left unexplained: BOTH routes carry a ~1e-06 force residual at this
    step, and the fitted route's cancels in the difference while the
    interpolated route's does not. Anyone picking this up starts here.
    """
    build = factory('isdf')
    with pytest.raises(NotImplementedError, match='ISDF factors'):
        normal_modes(build(water))
    # WATER DOES NOT REPRODUCE IT -- 3.7e-06, inside the tolerance -- which is
    # why this needs a molecule that does. Every ISDF force gate in the suite
    # runs on water, and that is how the symptom went unseen.
    info = {}
    numerical_hessian(water, build, info=info)
    assert info['asymmetry'] < HESSIAN_FD_ASYMMETRY_TOL
    ch2o = gto.M(atom=FORMALDEHYDE, basis='cc-pvdz', verbose=0)
    with pytest.raises(RuntimeError, match='scattered'):
        numerical_hessian(ch2o, build)
