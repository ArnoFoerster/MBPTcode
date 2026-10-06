"""Nuclear Hessian by central differences of the analytic gradient.

Two reasons this exists next to pyscf's analytic Hessian rather than instead
of it.

THE INTERPOLATED ROUTE HAS NO ANALYTIC HESSIAN. pyscf differentiates the
fitted interaction twice and knows nothing of the interpolation points, so on
an ISDF mean field its force constants belong to a different functional -- the
same defect `refuse_isdf_jk_gradient` exists for, one derivative further, and
silent because nothing raises. The ISDF FORCE is exact
(`isdf_mean_field_gradient`), so differencing it gives the Hessian of the
energy the SCF actually reported.

IT PARALLELIZES WHERE THE ANALYTIC ONE DOES NOT. pyscf's Hessian is one task
holding a whole allocation, and it dominates a surface scan at the cost of
tens of SCFs. The 6N displaced gradients here are independent, so 6N/w of
them run at a time and the wall time falls with the worker count even though
the total work rises.

The cost ratio is 6N (SCF + gradient) against one analytic Hessian; the wall
time is that divided by the workers.

WHAT IT IS WORTH, measured on formaldehyde/cc-pVDZ/B3LYP with density-fitted
exchange at NUCLEAR_FD_STEP: frequencies within 0.68 cm^-1 of pyscf's analytic
Hessian, and an acoustic sum rule of 5.6e-08 against pyscf's 5.1e-04. Both
diagnostics fall as h^2, which is the finite difference's own truncation and
says there is nothing else left in them.

ON THE INTERPOLATED ROUTE THE GRID DECIDES. The force is the exact derivative
of the ISDF energy at every grid, but that energy depends on the orientation
of each atom's interpolation cloud, and the covariant atomic frames turn the
clouds as the atoms move. On a coarse grid the surface is then rough on a
1e-3 Bohr scale: the step falls outside its Taylor radius, the asymmetry grows
with h, and even the h -> 0 force constants are wrong (formaldehyde at 148
points per atom: a 1421 cm^-1 mode at 2646). Below `ISDF_HESSIAN_MIN_GRID` the
Hessian is refused; at it formaldehyde's frequencies sit within 11 cm^-1 of
the density-fitted route's, and the asymmetry falls as h^2.
"""
import numpy as np

from src.Base.constants import (BOHR_TO_ANGSTROM,
                                HESSIAN_FD_ASYMMETRY_TOL,
                                HESSIAN_FD_NOISE_REL,
                                HESSIAN_FD_STEP_RATIO,
                                ISDF_GRID_ACCURACY,
                                ISDF_HESSIAN_MIN_GRID,
                                NUCLEAR_FD_STEP)
from src.Base.isdf_jk import ISDFJK
from src.Base.separable_ri import _SHELL_ORDER
from src.properties.optimize import mean_field_force

#: Why a finite-difference Hessian of an ISDF-K mean field has a grid floor.
ROUGH_GRID = ('the ISDF energy depends on the orientation of each atom\'s '
              'interpolation cloud and the atomic frames turn the clouds as '
              'the atoms move, so on a coarse grid the surface is rough on '
              'the step\'s scale and its force constants are wrong even though '
              'the force is exact (formaldehyde/cc-pVDZ/B3LYP: 1362 cm^-1 off '
              'at 148 points per atom, 405 at G2, 11 at G3)')


def displacement_list(natm, step=NUCLEAR_FD_STEP):
    """Every (atom, component, sign) central-difference displacement.

    Ordered so that the two members of a pair are adjacent, which is what lets
    a shard hold both and difference them without a join.
    """
    return [(ia, x, sign)
            for ia in range(natm) for x in range(3) for sign in (+1, -1)]


def displaced(mol, ia, x, sign, step=NUCLEAR_FD_STEP):
    """`mol` with atom `ia` moved by `sign * step` Bohr along axis `x`."""
    out = mol.copy()
    coords = np.asarray(mol.atom_coords()).copy()
    coords[ia, x] += sign * step
    out.set_geom_(coords, unit='Bohr')
    out.build(False, False)
    return out


def isdf_grid_counts(with_df):
    """{element: (A1, A2, A3, B1) shell counts} of the grid an ISDFJK placed,
    or None when its points were injected from outside."""
    radii = with_df.grid_radii if with_df.grid_radii is not None else with_df.radii
    if radii is not None:
        return {el: tuple(len(np.atleast_1d(r.get(sh, []))) for sh in _SHELL_ORDER)
                for el, r in radii.items()}
    if with_df.coords is not None:
        return None
    mol = with_df.mol
    counts = tuple(int(with_df.counts[sh]) for sh in _SHELL_ORDER)
    return {mol.atom_pure_symbol(i): counts for i in range(mol.natm)}


def require_hessian_grid(mf, level=ISDF_HESSIAN_MIN_GRID):
    """Raise if `mf` is an ISDF-K mean field on a grid below `level`."""
    with_df = getattr(mf, 'with_df', None)
    if not isinstance(with_df, ISDFJK):
        return
    got = isdf_grid_counts(with_df)
    if got is None:
        raise NotImplementedError(
            'this ISDF-K mean field was handed its interpolation points from '
            f'outside, so its grid cannot be checked against the {level} '
            f'floor a finite-difference Hessian needs: {ROUGH_GRID}.')
    require_hessian_counts(mf.mol.basis, got, level)


def require_hessian_counts(basis, counts, level=ISDF_HESSIAN_MIN_GRID):
    """Raise unless every element's (A1, A2, A3, B1) shell counts at `basis`
    reach the `level` grid a finite-difference ISDF-K Hessian needs.

    The check of `require_hessian_grid` on the counts alone, so a driver can
    refuse a Hessian grid before it converges the first SCF.
    counts: {element: (A1, A2, A3, B1)}.
    """
    floor = ISDF_GRID_ACCURACY.get(str(basis).lower(), {}).get(level)
    if floor is None:
        raise NotImplementedError(
            f'no ISDF grid is validated at {level} for {basis}, so a '
            f'finite-difference Hessian of this ISDF-K mean field has no grid '
            f'it may be built on: {ROUGH_GRID}.')
    short = {el: tuple(c) for el, c in counts.items()
             if any(n < f for n, f in zip(c, floor))}
    if short:
        listed = ', '.join(f'{el} {c}' for el, c in sorted(short.items()))
        raise NotImplementedError(
            f'a finite-difference Hessian of an ISDF-K mean field needs the '
            f'{level} grid {tuple(floor)} at {basis} or more on every '
            f'element, and this one has {listed}: {ROUGH_GRID}.')


def gradient_at(mol, scf_factory, ia, x, sign, step=NUCLEAR_FD_STEP):
    """(natm, 3) analytic force at one displaced geometry.

    Through `mean_field_force`, not `mf.Gradients()`, so an ISDF mean field
    gets the force of ITS energy rather than of the fitted one -- the whole
    point of differencing rather than calling pyscf's Hessian. An ISDF mean
    field below `ISDF_HESSIAN_MIN_GRID` is refused here.
    """
    mf = scf_factory(displaced(mol, ia, x, sign, step))
    require_hessian_grid(mf)
    return np.asarray(mean_field_force(mf))


def hessian_from_gradients(gradients, natm, step=NUCLEAR_FD_STEP, info=None):
    """(natm, natm, 3, 3) from {(ia, x, sign): gradient}, pyscf's layout.

    H[i, j, x, y] = dE^2 / dR_ix dR_jy, which is what `normal_modes` and
    `pyscf.hessian` both expect. Returned SYMMETRIZED.

    info: a dict, filled with `asymmetry` -- max |H_ixjy - H_jyix| BEFORE the
        symmetrization, which is the one number only the raw matrix carries.
        The two halves of a central difference of an exact force differ by
        the force noise over the step plus the O(h^2) truncation, which is
        what `check_hessian_noise` and `check_step_scaling` read. Diagnosing it
        after the fact is impossible: the returned matrix is symmetric by
        construction and reports zero.
    """
    h = np.zeros((natm, natm, 3, 3))
    for ja in range(natm):
        for y in range(3):
            gp = np.asarray(gradients[(ja, y, +1)])
            gm = np.asarray(gradients[(ja, y, -1)])
            # the displaced atom is the SECOND index: one pair fills a column
            h[:, ja, :, y] = (gp - gm) / (2.0 * step)
    flat = h.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
    if info is not None:
        info['asymmetry'] = float(np.abs(flat - flat.T).max())
        # THE SUM RULE MUST BE READ BEFORE SYMMETRIZING, and over the DISPLACED
        # index. Summing over the first index is identically zero here whatever
        # the forces do -- it is d(sum_i g_i)/dR_j and sum_i g_i vanishes at
        # every geometry -- so it measures nothing. Summing over the second
        # asks whether the forces actually respond correctly to a uniform
        # displacement, which nothing guarantees. Symmetrizing averages the two
        # and reports half of the real violation as if it were the check.
        info['translation'] = float(np.abs(h.sum(axis=1)).max())
    flat = 0.5 * (flat + flat.T)
    return flat.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)


def translation_residual(h):
    """max |sum_j H_ixjy| -- the acoustic sum rule over the DISPLACED index.

    Moving every atom together changes no force. For a SYMMETRIC Hessian the
    two sums are the same and either will do, which is the case for pyscf's
    analytic one.

    FOR A RAW FINITE-DIFFERENCE HESSIAN THEY ARE NOT THE SAME, and only this
    one is a check. Summing over the first index gives d(sum_i g_i)/dR_j, and
    sum_i g_i vanishes at every geometry for any translation-invariant energy,
    so that sum is zero however badly the forces behave. `numerical_hessian`
    records the informative one in `info` before symmetrizing; reading it off
    the returned matrix afterwards mixes the two and halves the violation.
    """
    return float(np.abs(np.asarray(h).sum(axis=1)).max())


def numerical_hessian(mol, scf_factory, step=NUCLEAR_FD_STEP, map_fn=None,
                      verbose=False, info=None,
                      asymmetry_tol=HESSIAN_FD_ASYMMETRY_TOL):
    """(natm, natm, 3, 3) Hessian by central differences of analytic forces.

    map_fn: anything with `map`'s signature, so the 6N independent displaced
        solves can be handed to a process pool. The default runs them in
        order, which is correct and slow.

    The step is the property layer's `NUCLEAR_FD_STEP`. Differencing a GRADIENT
    rather than an energy is what makes 1e-3 Bohr comfortable here: the noise
    being divided by h is the force's, which on the interpolated route sits at
    its ~1e-9 Ha/Bohr reproducibility floor, so the Hessian inherits ~1e-6 --
    five orders below a typical force constant.

    An asymmetry above the noise floor (`HESSIAN_FD_NOISE_REL`) is either h^2
    truncation or a surface rough on the step's scale, and one step cannot
    tell them apart: the 6N forces are then repeated at 2h and the asymmetry
    must grow by `HESSIAN_FD_STEP_RATIO` (`check_step_scaling`). The returned
    Hessian is the one at `step` either way.
    """
    natm = mol.natm
    runner = map if map_fn is None else map_fn

    def forces_at(h):
        jobs = displacement_list(natm, h)

        def one(item):
            k, (ia, x, sign) = item
            g = gradient_at(mol, scf_factory, ia, x, sign, h)
            if verbose:
                print(f'  [hess {k + 1:4d}/{len(jobs)}] atom {ia:3d} '
                      f'{"xyz"[x]} {"+" if sign > 0 else "-"}'
                      f'{h * BOHR_TO_ANGSTROM:.5f} A  |g|max '
                      f'{np.abs(g).max():.3e}', flush=True)
            return (ia, x, sign), g

        return dict(runner(one, list(enumerate(jobs))))

    gradients = forces_at(step)
    seen = {} if info is None else info
    h = hessian_from_gradients(gradients, natm, step, info=seen)
    seen['step'] = step
    seen['gradients'] = len(gradients)
    check_hessian_noise(h, seen, tol=asymmetry_tol)
    if seen['asymmetry_rel'] > HESSIAN_FD_NOISE_REL:
        twice = {}
        hessian_from_gradients(forces_at(2.0 * step), natm, 2.0 * step,
                               info=twice)
        seen['asymmetry_2h'] = twice['asymmetry']
        seen['gradients'] += 6 * natm
        check_step_scaling(seen)
    return h


def check_hessian_noise(h, info, tol=HESSIAN_FD_ASYMMETRY_TOL):
    """Raise if the asymmetry is too large for the difference to mean anything.

    A SYMMETRIC MATRIX IS NOT EVIDENCE: `hessian_from_gradients` symmetrizes,
    so the asymmetry it measured beforehand is the only record of how far the
    two halves were apart, and nothing downstream can recover it. Left
    unchecked it arrives as a low-frequency mode of the wrong sign -- the
    softest modes are the ones it reaches first, and they are the ones
    carrying the Huang-Rhys weight. Records `asymmetry_rel` in `info`.
    """
    scale = max(float(np.abs(np.asarray(h)).max()), 1e-30)
    rel = info['asymmetry'] / scale
    info['asymmetry_rel'] = rel
    if rel <= tol:
        return
    raise RuntimeError(
        f'the finite-difference Hessian is asymmetric by {info["asymmetry"]:.2e} '
        f'Ha/Bohr^2 ({rel:.2e} of the largest force constant), above '
        f'{tol:.1e}, at a step of {info["step"]:.1e} Bohr. Either the forces '
        f'are not reproducible at this geometry -- build the same mean field '
        f'twice and difference the forces -- or the surface is not smooth on '
        f'the scale of the step, which on an ISDF-K mean field means its grid '
        f'is too coarse. A Hessian built from them puts imaginary modes at '
        f'real minima.')


def check_step_scaling(info, ratio=HESSIAN_FD_STEP_RATIO):
    """Raise unless the asymmetry at 2h is the one at h times an h^2 ratio.

    Truncation of a smooth surface grows the asymmetry fourfold per doubling;
    a surface rough on the step's scale grows it less, force noise shrinks it.
    Records `step_ratio` in `info`.
    """
    r = info['asymmetry_2h'] / max(info['asymmetry'], 1e-300)
    info['step_ratio'] = r
    lo, hi = ratio
    if lo <= r <= hi:
        return
    raise RuntimeError(
        f'the Hessian asymmetry grows by {r:.2f} from h = {info["step"]:.1e} '
        f'to 2h ({info["asymmetry"]:.2e} -> {info["asymmetry_2h"]:.2e} '
        f'Ha/Bohr^2), outside the [{lo}, {hi}] an O(h^2) truncation gives: the '
        f'step is outside the surface\'s Taylor radius or the forces are '
        f'noisy, and the force constants at this step are not converged.')
