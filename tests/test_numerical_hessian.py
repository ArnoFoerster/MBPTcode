"""The Hessian by differencing the analytic force, against pyscf's analytic one.

Three claims are gated here. That the route AGREES with pyscf where both are
valid, to far inside what a Huang-Rhys factor cares about. That it is
available where pyscf's is not: on an interpolated mean field pyscf
differentiates the fitted interaction twice and returns force constants for a
different functional, while the ISDF FORCE is exact, so differencing it is the
only route to that Hessian. And that on the interpolated route it is built
only on a grid fine enough for the ISDF energy to be smooth on the step's
scale (`ISDF_HESSIAN_MIN_GRID`), and refused below it.

The acoustic sum rule is the sharpest statement available on any Hessian --
moving every atom together changes no force, it needs no reference
calculation, and neither route is told the identity exists.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.constants import (HARTREE_TO_CM, HESSIAN_FD_ASYMMETRY_TOL,
                                HESSIAN_FD_NOISE_REL, HESSIAN_FD_STEP_RATIO,
                                ISDF_GRID_ACCURACY,
                                ISDF_HESSIAN_MIN_GRID, NUCLEAR_FD_STEP)
from src.Base.isdf_jk import (FrozenFrames, frozen_frames, isdf_grid,
                              isdf_jk)
from src.Base.separable_ri import (_SHELL_ORDER, atomic_points,
                                   resolve_isdf_grid)
from src.properties.hessian import (check_step_scaling, displacement_list,
                                    gradient_at, hessian_from_gradients,
                                    numerical_hessian, translation_residual)
from src.properties.optimize import mean_field_force
from src.properties.vibronic import normal_modes
from tests.pins import assert_same

ATOM = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Its carbon's frame turns fastest of the four atoms (2.8 rad/Bohr), which is
#: what makes the coarse-grid ISDF surface rough here and not on water.
FORMALDEHYDE = ('C 0.0 0.0 -0.6035; O 0.0 0.0 0.7349; '
                'H 0.0 0.9753 -1.1057; H 0.0 -0.9753 -1.1057')


def factory(route, xc='b3lyp', level=None):
    """Converged mean fields: `level` names an ISDF_GRID_ACCURACY row (the
    interpolated route has no default grid)."""
    counts = (None if level is None else
              dict(zip(_SHELL_ORDER, ISDF_GRID_ACCURACY['cc-pvdz'][level])))

    def build(m):
        mf = dft.RKS(m, xc=xc) if xc else scf.RHF(m)
        mf = (isdf_jk(mf, auxbasis='cc-pvdz-ri', counts=counts)
              if route == 'isdf'
              else mf.density_fit(auxbasis='cc-pvdz-jkfit'))
        if xc:
            mf.grids.prune = None
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
        mf.verbose = 0
        mf.kernel()
        return mf
    return build


def _at(mol, coords):
    """`mol` moved to `coords` (Bohr)."""
    return mol.set_geom_(coords, unit='Bohr', inplace=False)


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
    violation as if it were the answer. Reading the returned matrix
    measures its own symmetrization.

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
    energy it is paired with, a surface rough on the step's scale -- scales
    differently and shows up as a different power.
    """
    build = factory('df')
    out = {}
    for h in (1e-3, 2e-3, 4e-3):
        info = {}
        numerical_hessian(water, build, step=h, info=info)
        out[h] = info
    lo, hi = HESSIAN_FD_STEP_RATIO
    for key in ('asymmetry', 'translation'):
        r = [out[4e-3][key] / out[2e-3][key], out[2e-3][key] / out[1e-3][key]]
        assert all(lo < v < hi for v in r), (key, r)


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


def test_the_interpolated_route_is_refused_below_the_floor_grid(water):
    """Both routes refuse a coarse interpolated mean field, for different
    reasons. `normal_modes` refuses to BUILD an analytic Hessian there because
    pyscf would differentiate the fitted interaction twice. `numerical_hessian`
    refuses a grid below `ISDF_HESSIAN_MIN_GRID`: the force is the exact
    derivative of the ISDF energy at every grid, but that energy depends on
    the orientation of each atom's interpolation cloud and the atomic frames
    turn the clouds as the atoms move, so a coarse grid's surface is rough on
    the step's scale. On formaldehyde at 148 points per atom the asymmetry is
    7.6e-02 and the h -> 0 force constants are themselves wrong (a 1421 cm^-1
    mode at 2646); at the 16-point-per-shell G1 grid it is 405 cm^-1.

    The refusal comes at the first displaced mean field, before any force,
    and on water too, whose own asymmetry would pass: the floor is a property
    of the grid, not of the molecule it happens to be tried on.
    """
    build = factory('isdf', level='G1')
    with pytest.raises(NotImplementedError, match='ISDF factors'):
        normal_modes(build(water))
    with pytest.raises(NotImplementedError, match=ISDF_HESSIAN_MIN_GRID):
        numerical_hessian(water, build)
    ch2o = gto.M(atom=FORMALDEHYDE, basis='cc-pvdz', verbose=0)
    with pytest.raises(NotImplementedError, match=r'\(16, 10, 6, 2\)'):
        numerical_hessian(ch2o, factory('isdf', level='G1'))


@pytest.mark.parametrize('xc, bound', [('b3lyp', 5.0), (None, 10.0)])
def test_the_floor_grid_reproduces_the_fitted_frequencies(xc, bound):
    """On the floor grid formaldehyde's Hessian is smooth on the step's scale
    and its frequencies follow the density-fitted route's.

    With the frames frozen at the reference (`numerical_hessian`'s default)
    the largest difference is 1.6 cm^-1 on B3LYP and 4.6 on Hartree-Fock, and
    the asymmetry 4.5e-07 and 5.8e-07 of the largest force constant, below
    the noise floor. On frames turning with each geometry it is 11.2 and
    34.5 cm^-1 and 7.1e-06 and 3.3e-05: the orientation dependence of the
    ISDF energy turned into curvature, five times larger on Hartree-Fock
    with its five times the exact exchange. If the asymmetry is above the
    noise floor the forces are repeated at 2h and must grow it fourfold.
    """
    ch2o = gto.M(atom=FORMALDEHYDE, basis='cc-pvdz', verbose=0)
    info = {}
    h_isdf = numerical_hessian(ch2o, factory('isdf', xc, ISDF_HESSIAN_MIN_GRID),
                               info=info)
    assert info['asymmetry_rel'] < HESSIAN_FD_ASYMMETRY_TOL
    if 'step_ratio' in info:
        lo, hi = HESSIAN_FD_STEP_RATIO
        assert lo <= info['step_ratio'] <= hi
    mf = factory('df', xc)(ch2o)
    w_df = normal_modes(mf, ch2o, hess=np.asarray(mf.Hessian().kernel()))[0]
    w_isdf = normal_modes(mf, ch2o, hess=h_isdf)[0]
    assert np.abs(w_isdf - w_df).max() * HARTREE_TO_CM < bound


def test_the_step_check_refuses_a_surface_rough_on_the_step_scale():
    """The asymmetries measured on formaldehyde/cc-pVDZ/B3LYP at h = 1e-3 and
    2e-3. The 148-point grid grows it by 2.38, a surface still rough at the
    step; Hartree-Fock on the floor grid by 3.85, the h^2 truncation. The
    check reads two numbers, so it is gated on them directly.
    """
    coarse = {'step': 1e-3, 'asymmetry': 1.365e-1, 'asymmetry_2h': 3.252e-1}
    with pytest.raises(RuntimeError, match='grows by 2.38'):
        check_step_scaling(coarse)
    floor = {'step': 1e-3, 'asymmetry': 2.493e-5, 'asymmetry_2h': 9.600e-5}
    check_step_scaling(floor)
    assert abs(floor['step_ratio'] - 3.85) < 0.01


#: The force ladder's formaldehyde. Its carbon's two in-plane neighbour-tensor
#: eigenvalues lie 3.2e-2 apart (relative), so the covariant frame turns its
#: cloud at 18.9 rad/Bohr under an in-plane carbon displacement, against 2.8
#: on `FORMALDEHYDE`: the same situation as DMABN's planar amine nitrogen
#: (1.0e-2, 16.2 rad/Bohr).
FORMALDEHYDE_LADDER = ('C 0 0 -0.5296; O 0 0 0.6742; '
                       'H 0 0.9429 -1.1123; H 0 -0.9429 -1.1123')


def test_frozen_frames_are_the_reference_points_turned_with_the_molecule():
    """At the reference the frozen frames place the covariant frames' points
    to a few ulp; under a rigid motion the cloud moves with the molecule; under
    an in-plane carbon step of 1e-3 Bohr the carbon cloud only translates
    (and turns with the body frame: 2.1e-5 Bohr at most), where the
    covariant frame swings its outer points by 0.30 Bohr."""
    mol = gto.M(atom=FORMALDEHYDE_LADDER, basis='cc-pvdz', verbose=0)
    counts, n_start = resolve_isdf_grid(ISDF_HESSIAN_MIN_GRID, 'cc-pvdz',
                                        ['C', 'H', 'O'],
                                        auxbasis='cc-pvdz-ri')
    frames = FrozenFrames(mol)

    def grid(m, f):
        return isdf_grid(m, counts=counts, auxbasis='cc-pvdz-ri',
                         n_start=n_start, frames=f)

    turning = grid(mol, None)
    assert_same(grid(mol, frames), turning, unit='1', quantity='root',
                label='frozen frames at the reference')

    c = mol.atom_coords()
    t = np.array([0.3, -0.2, 0.1])
    u = np.linalg.qr(np.array([[1.0, 0.2, 0.1], [0.3, 1.0, -0.2],
                               [0.1, 0.4, 1.0]]))[0]
    u *= np.sign(np.linalg.det(u))
    moved = grid(_at(mol, c @ u + t), frames)
    assert np.abs(moved - (turning @ u + t)).max() < 1e-10

    c[0, 1] += NUCLEAR_FD_STEP
    step = _at(mol, c)
    _, radii, origins = isdf_grid(mol, counts=counts, auxbasis='cc-pvdz-ri',
                                  n_start=n_start, return_info=True)
    carbon = slice(0, len(atomic_points(radii['C'], origin=origins['C'])))
    shift = np.array([0.0, NUCLEAR_FD_STEP, 0.0])
    frozen_move = np.abs(grid(step, frames)[carbon] - turning[carbon]
                         - shift).max()
    turning_move = np.abs(grid(step, None)[carbon] - turning[carbon]
                          - shift).max()
    assert frozen_move < 1e-4
    assert turning_move > 0.1


def test_the_force_on_frozen_frames_is_the_slope_of_their_energy():
    """The point chain of a mean field on frozen frames turns its clouds with
    the body frame alone (`mean_field_point_chain`), and its force is then the
    derivative of the energy those frames give. Frames frozen at the ladder
    geometry, the force taken 0.01 Bohr away along the carbon's in-plane axis
    (where the body frame is not the identity and the force is not zero by
    symmetry), its carbon in-plane and oxygen axial components against a
    central difference of the energy at h = 1e-4."""
    mol = gto.M(atom=FORMALDEHYDE_LADDER, basis='cc-pvdz', verbose=0)
    build = factory('isdf', level='G1')
    frames = FrozenFrames(mol)
    away = mol.atom_coords().copy()
    away[0, 1] += 1e-2
    h = 1e-4
    with frozen_frames(frames):
        force = np.asarray(mean_field_force(build(_at(mol, away))))
        for ia, x in ((0, 1), (1, 2)):
            energy = []
            for sign in (+1, -1):
                c = away.copy()
                c[ia, x] += sign * h
                energy.append(build(_at(mol, c)).e_tot)
            slope = (energy[0] - energy[1]) / (2 * h)
            assert abs(force[ia, x]) > 1e-3
            assert abs(force[ia, x] - slope) < 1e-7


def test_the_floor_grid_hessian_is_symmetric_where_a_frame_turns_fast():
    """With every displaced mean field on the reference's frames the floor
    grid's Hessian of the ladder formaldehyde is symmetric to the force noise
    (2.2e-07 of the largest force constant) and its frequencies follow the
    density-fitted route's (1.3 cm^-1). On turning frames the
    same Hessian is refused, asymmetric by 2.0e-02, all of it in the carbon's
    in-plane rows, whose force constant came out -0.088 Ha/Bohr^2 against
    density fitting's 0.619 on LRC-wPBEh: an imaginary 1610 cm^-1 mode at a
    geometry density fitting puts no saddle at. 0.02 Bohr off the symmetric
    geometry it is the same (2.2e-02): the cause is the near-degenerate
    neighbour tensor, not the symmetry."""
    ch2o = gto.M(atom=FORMALDEHYDE_LADDER, basis='cc-pvdz', verbose=0)
    info = {}
    h_isdf = numerical_hessian(ch2o, factory('isdf', 'b3lyp',
                                             ISDF_HESSIAN_MIN_GRID), info=info)
    assert info['frozen_frames']['convention'] == 'reference-body-fixed'
    assert info['asymmetry_rel'] < HESSIAN_FD_NOISE_REL
    mf = factory('df', 'b3lyp')(ch2o)
    w_df = normal_modes(mf, ch2o, hess=np.asarray(mf.Hessian().kernel()))[0]
    w_isdf = normal_modes(mf, ch2o, hess=h_isdf)[0]
    assert np.abs(w_isdf - w_df).max() * HARTREE_TO_CM < 5.0
