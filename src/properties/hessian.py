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
of each atom's interpolation cloud. On a coarse grid the surface is rough on
a 1e-3 Bohr scale: the step falls outside its Taylor radius, the asymmetry
grows with h, and even the h -> 0 force constants are wrong (formaldehyde at
148 points per atom: a 1421 cm^-1 mode at 2646). Below `ISDF_HESSIAN_MIN_GRID`
the Hessian is refused.

AND THE FRAMES ARE FROZEN AT THE REFERENCE. The covariant atomic frames turn
each cloud with its atom's environment, at 1/gap of the neighbour tensor's
eigenvalues, and an sp2 atom with three similar neighbours turns its cloud
fast even on the floor grid: formaldehyde's carbon at 18.9 rad/Bohr, DMABN's
planar amine nitrogen at 16.2. Turning the orientation dependence that fast
puts it into the curvature: formaldehyde LRC-wPBEh/cc-pVDZ on GH, h = 1e-3
Bohr, is asymmetric by 2.85e-2 Ha/Bohr^2 (2.7 percent, 2.3 percent 0.02 Bohr
off the symmetric geometry: it is the tensor gap, not the symmetry), planar
DMABN by 3.46 percent. Every displaced mean field therefore places its clouds
on the reference geometry's frames turned with the whole molecule
(`isdf_jk.frozen_frames`, `FrozenFrames`), the way it fits on the
reference's pair layout: the same Hessian is then asymmetric by 2.4e-7, the
force noise, and its frequencies are within 4.1 cm^-1 of density fitting's,
where the turning frames' put an imaginary 1610 cm^-1 mode. At the reference
the points are the turning frames' bit for bit; the force constants are
those of the energy with the frames held, not of the orientation artifact's
turning.

EVERY DISPLACED MEAN FIELD FITS ON THE REFERENCE GEOMETRY'S PAIR LAYOUT. The
ISDF-K fit screens its AO pairs against `pair_tol`, a discrete choice. If
each displaced SCF screened its own pairs, a pair crossing the threshold
between R0 - h and R0 + h would make the two forces of one central difference
forces of two different energies, and their difference over 2h an O(1/h)
error in that column of the Hessian. `numerical_hessian` therefore takes the
layout of the factory's mean field at R0 (`reference_pair_layout`) and builds
every displaced mean field inside `frozen_pair_layout`, the way a walk does.
"""
import json
import os
import pathlib

import numpy as np

from src.Base.numerics.constants import (BOHR_TO_ANGSTROM, FD_LADDER_RATIO_FLOOR,
                                         HESSIAN_ASYMMETRY_WORST,
                                         HESSIAN_FD_ASYMMETRY_TOL,
                                         HESSIAN_FD_NOISE_REL,
                                         HESSIAN_FD_STEP_RATIO,
                                         ISDF_GRID_ACCURACY,
                                         ISDF_HESSIAN_MIN_GRID,
                                         NUCLEAR_FD_STEP)
from src.Base.factorization.distributed_isdf_jk import scf_pair_layout
from src.Base.factorization.isdf_jk import (ISDFJK, FrozenFrames, frozen_frames,
                                            frozen_pair_layout)
from src.Base.utils.record_io import record_hash
from src.Base.geometry.geometry import displaced
from src.Base.factorization.separable_ri import (SHELL_ORDER, frame_turning_record,
                                                 shell_tuple)
from src.Base.utils.parallel.mpi_grid import current_comm, lockstep
from src.Base.numerics.numerics import numerics_identity
from src.gradients.surface_identity import force_identity
from src.properties.optimize import cartesian_hessian
from src.gradients.mean_field_surface import mean_field_force
from src.gradients.surface import evaluate

#: `numerical_hessian`'s default `pair_layout`: the layout of the factory's
#: mean field at the reference geometry (`reference_pair_layout`).
REFERENCE_LAYOUT = 'reference'

#: `numerical_hessian`'s and `surface_hessian`'s default `frames`: every
#: displaced ISDF-K mean field places its points on the atomic frames of the
#: reference geometry, turned with the molecule (`isdf_jk.FrozenFrames`).
REFERENCE_FRAMES = 'reference'

#: Why a finite-difference Hessian of an ISDF-K mean field has a grid floor.
ROUGH_GRID = ('the ISDF energy depends on the orientation of each atom\'s '
              'interpolation cloud, so on a coarse grid the surface is rough on '
              'the step\'s scale and its force constants are wrong even though '
              'the force is exact (formaldehyde/cc-pVDZ/B3LYP on frames turning '
              'with each geometry: 1362 cm^-1 off at 148 points per atom, 405 '
              'at (16, 10, 6, 2), 11 at (32, 20, 12, 4), 1.6 there on frames '
              'frozen at the reference)')


def fd_hessian_stamp(numerics=None):
    """What a finite-difference Hessian froze, the one stamp every checkpoint
    of one writes and compares: the pair layout and the frames of the
    reference, and -- for a Hessian of a surface -- the identity of the
    `Numerics` it was taken on."""
    stamp = {'pair_layout': REFERENCE_LAYOUT, 'frames': REFERENCE_FRAMES}
    if numerics is not None:
        stamp['numerics'] = numerics.identity()
    return stamp


def stencil_frames(mol, frames):
    """The `FrozenFrames` of a stencil about `mol` for `frames`
    (`REFERENCE_FRAMES`, a `FrozenFrames`, or None for each geometry's own
    covariant frames)."""
    if isinstance(frames, str):
        if frames != REFERENCE_FRAMES:
            raise ValueError(f'frames={frames!r}: pass {REFERENCE_FRAMES!r}, '
                             f'a FrozenFrames or None')
        return FrozenFrames(mol)
    return frames


class ForceCache:
    """The displaced forces of one finite-difference Hessian on disk, one
    file per displacement, each carrying what it is a force of: the
    geometry the Hessian is taken at (Bohr), the step and the hash of the
    surface's conventions (`cached_conventions`). A file whose header
    differs is refused, not overwritten: it is a force of another Hessian."""

    def __init__(self, directory, mol, step, surface, frames=None):
        self.dir = pathlib.Path(directory)
        record = cached_conventions(surface)
        self.header = {'coords': np.array(mol.atom_coords(), float),
                       'step': np.array(float(step)),
                       'conventions': np.array(
                           '' if record is None else record_hash(record)),
                       # a force on the turning frames is a force of another
                       # surface, and a file written before frames were frozen
                       # carries no such key
                       'frames': np.array(
                           'turning' if frames is None
                           else frames.to_record()['convention'])}
        if writes():
            self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, ia, x, sign):
        return self.dir / f'{ia}_{"xyz"[x]}_{"+" if sign > 0 else "-"}.npz'

    def get(self, ia, x, sign):
        """The force kept for this displacement, or None."""
        path = self.path(ia, x, sign)
        # rank 0 decides, so every rank computes or reads the same forces
        if not lockstep(np.array([path.exists()]))[0]:
            return None
        with np.load(path) as held:
            for key, want in self.header.items():
                if key not in held or not np.array_equal(held[key], want):
                    raise RuntimeError(
                        f'{path} holds a force of another Hessian (its {key} '
                        f'differs from this one\'s): move the directory '
                        f'aside or point the cache elsewhere')
            return np.array(held['gradient'])

    def put(self, ia, x, sign, gradient):
        """Keep one force, atomically, from rank 0."""
        if not writes():
            return
        path = self.path(ia, x, sign)
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'wb') as handle:
            np.savez(handle, gradient=np.asarray(gradient, float),
                     **self.header)
        os.replace(tmp, path)


def cached_conventions(surface):
    """What a cached force is a force of: `force_identity`, the one answer
    the walk's first point and checkpoint give too -- each half's
    conventions, grid, settings, environment, mean field, SCF layout, frames
    and span, not the walk's history; None for a surface that freezes
    nothing and writes no record, which a dense surface is."""
    return force_identity(surface)


def component_step(mol, ia, x, step):
    """(natm, 3) Bohr: `step` on atom `ia`'s axis `x`, zero elsewhere -- the
    one-component displacement `geometry.displaced` moves a molecule by."""
    delta = np.zeros((mol.natm, 3))
    delta[ia, x] = step
    return delta


def ladder_ratios(differences):
    """d_j / d_(j+1) over the successive differences of a ladder of halved
    finite-difference steps, the one convergence read of every ladder here
    (4 for a clean h^2 central difference, 2 for an error first order in h): a
    denominator within FD_LADDER_RATIO_FLOOR of zero is read as the floor."""
    d = np.asarray(differences, float)
    den = d[1:]
    return d[:-1] / np.where(np.abs(den) > FD_LADDER_RATIO_FLOOR, den,
                             FD_LADDER_RATIO_FLOOR)


def displacement_list(natm, step=NUCLEAR_FD_STEP):
    """Every (atom, component, sign) central-difference displacement.

    Ordered so that the two members of a pair are adjacent, which is what lets
    a shard hold both and difference them without a join.
    """
    return [(ia, x, sign)
            for ia in range(natm) for x in range(3) for sign in (+1, -1)]


def isdf_grid_counts(with_df):
    """{element: (A1, A2, A3, B1) shell counts} of the grid an ISDFJK placed,
    or None when its points were injected from outside."""
    radii = with_df.grid_radii if with_df.grid_radii is not None else with_df.radii
    if radii is not None:
        return {el: tuple(len(np.atleast_1d(r.get(sh, []))) for sh in SHELL_ORDER)
                for el, r in radii.items()}
    if with_df.coords is not None:
        return None
    mol = with_df.mol
    counts = shell_tuple(with_df.counts)
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
    mf = scf_factory(displaced(mol, component_step(mol, ia, x, sign * step)))
    require_hessian_grid(mf)
    return np.asarray(mean_field_force(mf))


def reference_pair_layout(mol, scf_factory, map_fn=None):
    """The AO-pair columns (mu, nu, weight) the ISDF-K fit of
    `scf_factory`'s mean field at `mol` keeps (`scf_pair_layout`), which every
    displaced mean field of a finite-difference Hessian about `mol` fits on;
    None for a mean field without an ISDF-K fit.

    One SCF, run as the single item of `map_fn`: under `mpi_map` one rank
    converges it, inside `distributed(None)` like every displaced one, and
    every rank receives that rank's arrays, so all ranks difference forces on
    one layout.
    """
    runner = map if map_fn is None else map_fn

    def one(_):
        return scf_pair_layout(scf_factory(mol))

    return list(runner(one, [0]))[0]


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
    flat = cartesian_hessian(h, natm)
    if info is not None:
        info['asymmetry'] = float(np.abs(flat - flat.T).max())
        info['asymmetry_worst'] = worst_asymmetry(flat)
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


def worst_asymmetry(flat, count=HESSIAN_ASYMMETRY_WORST):
    """The `count` largest |H_ixjy - H_jyix| of a raw (3N, 3N) Hessian, each
    as {'i': [atom, axis], 'j': [atom, axis], 'h_ij', 'h_ji', 'difference'}
    in Ha/Bohr^2, largest first; each pair once."""
    diff = np.triu(np.abs(flat - flat.T), 1)
    order = np.argsort(diff, axis=None)[::-1][:count]
    out = []
    for k in order:
        a, b = np.unravel_index(k, diff.shape)
        out.append({'i': [int(a // 3), 'xyz'[a % 3]],
                    'j': [int(b // 3), 'xyz'[b % 3]],
                    'h_ij': float(flat[a, b]), 'h_ji': float(flat[b, a]),
                    'difference': float(diff[a, b])})
    return out


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
                      asymmetry_tol=HESSIAN_FD_ASYMMETRY_TOL,
                      pair_layout=REFERENCE_LAYOUT, frames=REFERENCE_FRAMES):
    """(natm, natm, 3, 3) Hessian by central differences of analytic forces.

    map_fn: anything with `map`'s signature, so the 6N independent displaced
        solves can be handed to a process pool. The default runs them in
        order, which is correct and slow.
    pair_layout: the ISDF-K pair layout every displaced mean field fits on
        (`frozen_pair_layout`). By default the one of the factory's mean field
        at `mol` (`reference_pair_layout`, one more SCF); a caller holding
        that mean field passes `scf_pair_layout(mf)` instead. None lets each
        displaced mean field screen its own pairs, which puts an O(1/h) error
        into every column across which a pair crosses `pair_tol`. Recorded in
        `info` as `frozen_pairs` (None where no ISDF-K fit was frozen).
    frames: the atomic frames every displaced ISDF-K mean field places its
        points on: by default the reference geometry's, turned with the
        molecule (`isdf_jk.FrozenFrames`); None lets each turn its own, which
        puts the orientation dependence of the ISDF energy into the force
        constants wherever a frame turns fast. Recorded in `info` as
        `frozen_frames`, with the covariant frames' turning at the reference
        as `frame_turning` (`separable_ri.frame_turning_record`).

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
    if isinstance(pair_layout, str) and pair_layout == REFERENCE_LAYOUT:
        pair_layout = reference_pair_layout(mol, scf_factory, map_fn)
    frames = stencil_frames(mol, frames)

    def forces_at(h):
        jobs = displacement_list(natm, h)

        def one(item):
            k, (ia, x, sign) = item
            # inside the job, so the block holds on whichever thread or rank
            # runs it
            with frozen_pair_layout(pair_layout), frozen_frames(frames):
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
    seen['frozen_pairs'] = (None if pair_layout is None
                            else int(len(pair_layout[0])))
    record_frames(seen, frames, mol)
    check_hessian_noise(h, seen, tol=asymmetry_tol)
    if seen['asymmetry_rel'] > HESSIAN_FD_NOISE_REL:
        twice = {}
        hessian_from_gradients(forces_at(2.0 * step), natm, 2.0 * step,
                               info=twice)
        seen['asymmetry_2h'] = twice['asymmetry']
        seen['gradients'] += 6 * natm
        check_step_scaling(seen)
    return h


def surface_hessian(surface, mol, step=NUCLEAR_FD_STEP, verbose=False,
                    info=None, asymmetry_tol=HESSIAN_FD_ASYMMETRY_TOL,
                    cache_dir=None, frames=REFERENCE_FRAMES):
    """(natm, natm, 3, 3) Hessian of a surface's total energy at `mol` by
    central differences of its force, on the conventions the surface froze.

    The 6N displaced forces go through `evaluate`, one after another, each
    over every rank of the region it runs in. `info` takes the asymmetry
    and the sum rule (`hessian_from_gradients`), refused above
    `asymmetry_tol` (`check_hessian_noise`, which writes `asymmetry.json`
    into `cache_dir` first).

    cache_dir: each displaced force is kept there as it is computed
    (`ForceCache`: `<atom>_<axis>_<sign>.npz`, with the geometry, the step
    and a hash of the surface's conventions), and one already there is read
    instead of computed; one written for another geometry, step or surface
    is refused. Rank 0 writes; every rank reads. `info['forces_reused']`
    counts what was read.
    frames: the atomic frames the surface's ISDF-K mean field places its
        points on at each displaced geometry, as `numerical_hessian`'s, and in
        the cache's header; the surface's own factors keep the frames they
        froze.
    """
    jobs = displacement_list(mol.natm, step)
    frames = stencil_frames(mol, frames)
    cache = (None if cache_dir is None
             else ForceCache(cache_dir, mol, step, surface, frames))
    gradients, reused = {}, 0
    for k, (ia, x, sign) in enumerate(jobs):
        g = None if cache is None else cache.get(ia, x, sign)
        if g is None:
            with frozen_frames(frames):
                g, _, _ = evaluate(surface, displaced(
                    mol, component_step(mol, ia, x, sign * step)))
            g = np.asarray(g)
            if cache is not None:
                cache.put(ia, x, sign, g)
        else:
            reused += 1
        gradients[(ia, x, sign)] = g
        if verbose:
            print(f'  [hess {k + 1:4d}/{len(jobs)}] atom {ia:3d} '
                  f'{"xyz"[x]} {"+" if sign > 0 else "-"} |g|max '
                  f'{np.abs(g).max():.3e}', flush=True)
    seen = {} if info is None else info
    h = hessian_from_gradients(gradients, mol.natm, step, info=seen)
    seen['step'] = step
    seen['gradients'] = len(gradients)
    seen['forces_reused'] = reused
    record_frames(seen, frames, mol)
    seen['fd_hessian'] = dict(fd_hessian_stamp(),
                              numerics=numerics_identity(surface))
    check_hessian_noise(h, seen, tol=asymmetry_tol, dump_dir=cache_dir)
    return h


def record_frames(info, frames, mol):
    """Write into `info` the frozen atomic frames (None where none were
    frozen) and the covariant frames' turning at the reference, which
    freezing them takes out of the force constants."""
    info['frozen_frames'] = None if frames is None else frames.to_record()
    info['frame_turning'] = frame_turning_record(mol)


def check_hessian_noise(h, info, tol=HESSIAN_FD_ASYMMETRY_TOL, dump_dir=None):
    """Raise if the asymmetry is too large for the difference to mean anything.

    dump_dir: where `asymmetry.json` is written before the refusal (rank 0):
    the asymmetry, absolute and relative, the step and the worst elements
    with their atom/axis pairs (`worst_asymmetry`).

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
    if dump_dir is not None and writes():
        dump = pathlib.Path(dump_dir)
        dump.mkdir(parents=True, exist_ok=True)
        (dump / 'asymmetry.json').write_text(json.dumps({
            'asymmetry': info['asymmetry'], 'asymmetry_rel': rel,
            'tolerance': tol, 'step': info.get('step'),
            'largest_force_constant': scale,
            'worst': info.get('asymmetry_worst')}, indent=1))
    raise RuntimeError(
        f'the finite-difference Hessian is asymmetric by {info["asymmetry"]:.2e} '
        f'Ha/Bohr^2 ({rel:.2e} of the largest force constant), above '
        f'{tol:.1e}, at a step of {info["step"]:.1e} Bohr. Either the forces '
        f'are not reproducible at this geometry -- build the same mean field '
        f'twice and difference the forces -- or the surface is not smooth on '
        f'the scale of the step, which on an ISDF-K mean field means its grid '
        f'is too coarse. A Hessian built from them puts imaginary modes at '
        f'real minima.')


def writes():
    """Whether this rank writes files: rank 0, or a serial run."""
    comm = current_comm()
    return comm is None or comm.Get_rank() == 0


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
