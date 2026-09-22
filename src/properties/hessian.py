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
holding a whole allocation, and it dominates a surface scan: measured on
azobenzene/cc-pVDZ at 8 threads, the SCF is 33.4 s, the gradient 10.6 s and
the Hessian 1389.5 s -- forty-two SCFs. The 6N displaced gradients here are
independent, so 6N/w of them run at a time and the wall time falls with the
worker count even though the total work rises.

The cost ratio is 6N (SCF + gradient) against one analytic Hessian; the wall
time is that divided by the workers.

WHAT IT IS WORTH, measured on formaldehyde/cc-pVDZ/B3LYP with density-fitted
exchange at NUCLEAR_FD_STEP: frequencies within 0.68 cm^-1 of pyscf's analytic
Hessian, and an acoustic sum rule of 5.6e-08 against pyscf's 5.1e-04. Both
diagnostics fall as h^2, which is the finite difference's own truncation and
says there is nothing else left in them.

IT DOES NOT WORK ON THE INTERPOLATED ROUTE and is refused there. The same
molecule gives frequencies up to 590 cm^-1 away -- a 1421 cm^-1 mode comes out
at 2012 -- and the diagnostics fall as h^1 rather than h^2, which is the
signature of a force that is not the derivative of its own energy at the
displaced geometries even though it gates at 7e-09 at the reference. That is a
statement about the ISDF gradient and not about Hessians.
"""
import numpy as np

from src.Base.constants import (BOHR_TO_ANGSTROM,
                                HESSIAN_FD_ASYMMETRY_TOL,
                                NUCLEAR_FD_STEP)
from src.properties.optimize import mean_field_force


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


def gradient_at(mol, scf_factory, ia, x, sign, step=NUCLEAR_FD_STEP):
    """(natm, 3) analytic force at one displaced geometry.

    Through `mean_field_force`, not `mf.Gradients()`, so an ISDF mean field
    gets the force of ITS energy rather than of the fitted one -- the whole
    point of differencing rather than calling pyscf's Hessian.
    """
    return np.asarray(mean_field_force(scf_factory(displaced(mol, ia, x, sign,
                                                             step))))


def hessian_from_gradients(gradients, natm, step=NUCLEAR_FD_STEP, info=None):
    """(natm, natm, 3, 3) from {(ia, x, sign): gradient}, pyscf's layout.

    H[i, j, x, y] = dE^2 / dR_ix dR_jy, which is what `normal_modes` and
    `pyscf.hessian` both expect. Returned SYMMETRIZED.

    info: a dict, filled with `asymmetry` -- max |H_ixjy - H_jyix| BEFORE the
        symmetrization, which is the one number only the raw matrix carries. A
        central difference makes the two halves differ solely through the
        gradient's own noise divided by the step, so this measures the noise
        floor at this step and NOT the truncation error. Diagnosing it after
        the fact is impossible: the returned matrix is symmetric by
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
    """
    natm = mol.natm
    jobs = displacement_list(natm, step)
    runner = map if map_fn is None else map_fn

    def one(item):
        k, (ia, x, sign) = item
        g = gradient_at(mol, scf_factory, ia, x, sign, step)
        if verbose:
            print(f'  [hess {k + 1:4d}/{len(jobs)}] atom {ia:3d} '
                  f'{"xyz"[x]} {"+" if sign > 0 else "-"}'
                  f'{step * BOHR_TO_ANGSTROM:.5f} A  |g|max {np.abs(g).max():.3e}',
                  flush=True)
        return (ia, x, sign), g

    gradients = dict(runner(one, list(enumerate(jobs))))
    seen = {} if info is None else info
    h = hessian_from_gradients(gradients, natm, step, info=seen)
    seen['step'] = step
    seen['gradients'] = len(jobs)
    check_hessian_noise(h, seen, tol=asymmetry_tol)
    return h


def check_hessian_noise(h, info, tol=HESSIAN_FD_ASYMMETRY_TOL):
    """Raise if the forces scattered too much for the difference to mean anything.

    A SYMMETRIC MATRIX IS NOT EVIDENCE: `hessian_from_gradients` symmetrizes,
    so the asymmetry it measured beforehand is the only record of how far the
    two halves were apart, and nothing downstream can recover it. Left
    unchecked the noise arrives as a low-frequency mode of the wrong sign --
    the softest modes are the ones it reaches first, and they are the ones
    carrying the Huang-Rhys weight.
    """
    scale = max(float(np.abs(np.asarray(h)).max()), 1e-30)
    rel = info['asymmetry'] / scale
    if rel <= tol:
        return
    raise RuntimeError(
        f'the displaced forces scattered by {info["asymmetry"]:.2e} Ha/Bohr^2 '
        f'({rel:.2e} of the largest force constant), above {tol:.1e}: at a '
        f'step of {info["step"]:.1e} Bohr that is a force reproducibility of '
        f'{info["asymmetry"] * info["step"]:.1e} Ha/Bohr, and the difference '
        f'of two such forces carries more noise than curvature. A Hessian '
        f'built from them puts imaginary modes at real minima. Either the '
        f'force is not reproducible at this geometry -- build the same mean '
        f'field twice and difference the forces -- or the step is too small '
        f'for it.')
