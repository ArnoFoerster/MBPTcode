"""Geometry optimization on any `PotentialEnergySurface`.

A Cartesian trust-region quasi-Newton method with no dependency, geomeTRIC's
internal coordinates when they are installed, and the mean-field ground-state
relaxation the vibronic quantities are defined about.

A cubic chain fixes five discrete choices at a reference geometry (the
quasiparticle set, the frame orientation, the interpolation pair layout, the
Newton branch and the residue backend), since each is otherwise a
discontinuity in the surface. An optimizer moves far enough that the layout
chosen at the start is not the one the fit would choose at the end, but
refreezing every step makes the energy discontinuous between steps, which
breaks a line search or a BFGS history. Both optimizers therefore freeze once,
so the optimization runs on one smooth surface, and offer an outer loop:
`surface.refreeze` at the converged geometry and optimize again. How far the
refreeze moved the geometry is reported as the error bar on the minimum.

Inside `with distributed(comm):` every rank runs the walk, and every
evaluation goes through `surface.evaluate`, which locks the geometry to rank
0's on the way in and the energy, force and diagnostics on the way out. Every
decision (step, convergence test, trust radius, rejection, refreeze) is
therefore taken on every rank from the same numbers, and the result is locked
to rank 0's once more at the end (`rank_zero_walk`). Outside a distributed
region every lockstep is a no-op and the walk is bit for bit the serial one.
"""
import itertools
import os
import tempfile
import warnings

import numpy as np
from pyscf import grad  # noqa: F401  registers mf.Gradients/nuc_grad_method

from src.Base.body_frame import aligned_displacement
from src.Base.constants import (BOHR_TO_ANGSTROM, GEOM_OPT_CONV,
                                GEOMETRIC_START_TOL, HARTREE_TO_EV,
                                HARTREE_TO_MEV, REFREEZE_TOL_FLAG_MEV,
                                REFREEZE_TOL_MEV, SADDLE_CURVATURE_TOL,
                                SADDLE_ESCAPE_STEP_BOHR, START_NUDGE_BOHR,
                                START_NUDGE_SEED)
from src.Base.declaration import SurfacePhysics
from src.Base.distributed_isdf_jk import scf_pair_layout
from src.Base.environment import environment_label, resolve_environment
from src.Base.isdf_jk import (ISDFJK, frozen_pair_layout,
                              mean_field_skeleton_force)
from src.Base.utils.mpi_grid import lockstep, lockstep_mean_field
from src.SingleReference.LinearResponse.rpa_energy import declared_ground_state
from src.properties.surface import (driven_chain, evaluate,
                                    lockstep_geometry)


def translation_rotation_basis(coords, masses=None):
    """Orthonormal basis of the 6 (5 if linear) rigid-body directions.

    Not projected out, they are null directions of the gradient along which
    the BFGS inverse Hessian blows up and the molecule drifts and spins. Mass
    weighting is optional because the projector only has to span the same
    subspace.
    """
    n = len(coords)
    m = np.ones(n) if masses is None else np.asarray(masses, float)
    c = coords - (m[:, None] * coords).sum(0) / m.sum()
    v = np.zeros((6, n, 3))
    for k in range(3):
        v[k, :, k] = 1.0
    v[3, :, 1], v[3, :, 2] = -c[:, 2], c[:, 1]
    v[4, :, 2], v[4, :, 0] = -c[:, 0], c[:, 2]
    v[5, :, 0], v[5, :, 1] = -c[:, 1], c[:, 0]
    v = v.reshape(6, 3 * n)
    # Gram-Schmidt, dropping the rotation that vanishes for a linear molecule.
    out = []
    for row in v:
        for q in out:
            row = row - (row @ q) * q
        nrm = np.linalg.norm(row)
        if nrm > 1e-8:
            out.append(row / nrm)
    return np.array(out)


def _rfo_step(hess, grad, trust, proj):
    """Rational-function step, projected and trust-radius limited.

    RFO rather than plain Newton because the Hessian is only approximate early
    on and a Newton step through a small or negative eigenvalue is enormous;
    the augmented eigenproblem shifts it automatically.
    """
    h = proj.T @ hess @ proj
    g = proj.T @ grad
    n = len(g)
    aug = np.zeros((n + 1, n + 1))
    aug[:n, :n], aug[:n, n], aug[n, :n] = h, g, g
    w, v = np.linalg.eigh(0.5 * (aug + aug.T))
    vec = v[:, 0]
    if abs(vec[n]) < 1e-10:                       # degenerate: fall back to SD
        step = -g
    else:
        step = vec[:n] / vec[n]
    nrm = np.linalg.norm(step)
    if nrm > trust:
        step *= trust / nrm
    return proj @ step


def cartesian_hessian(hess, natm):
    """A Hessian as (3N, 3N) in Ha/Bohr^2, from that shape or pyscf's
    (natm, natm, 3, 3)."""
    h = np.asarray(hess, float)
    if h.ndim == 4:
        h = h.transpose(0, 2, 1, 3)
    return h.reshape(3 * natm, 3 * natm)


def internal_modes(hess, coords):
    """(curvatures ascending, Ha/Bohr^2; their unit Cartesian modes as
    columns) of a Cartesian Hessian in the internal space at `coords`, the
    rigid-body directions projected out."""
    coords = np.asarray(coords, float).reshape(-1, 3)
    n3 = 3 * len(coords)
    h = cartesian_hessian(hess, len(coords))
    tr = translation_rotation_basis(coords)
    w, v = np.linalg.eigh(np.eye(n3) - tr.T @ tr)
    q = v[:, w > 0.5]
    lam, u = np.linalg.eigh(q.T @ (0.5 * (h + h.T)) @ q)
    return lam, q @ u


def walk_resolution(grad, hess, coords):
    """How much energy a walk leaves undetermined where it stopped.

    dE_walk = 1/2 g^T H^-1 g: the energy between the walk's last point and
    the minimum of its own quadratic model, from its last gradient g (Ha/Bohr)
    and the approximate Hessian H it ended with (Ha/Bohr^2), both Cartesian,
    taken in the internal space (the rigid-body directions projected out).
    A refreeze pass that moves the energy by less than this has not measured
    the conventions; it has measured where the optimizer stopped.

    Only the positive-definite part of H is a quadratic model. Where H has a
    non-positive internal mode the estimate is the safe bound 1/2 |g|^2 /
    lambda_min, every gradient component at the softest positive curvature,
    and `bound` says so; with no positive curvature at all there is no
    estimate (`de_meV` None).

    Returns {'de_meV', 'bound', 'nonpositive_modes', 'softest_curvature'}
    ('quadratic' or 'safe bound', the count of internal modes with curvature
    <= 0, the smallest positive one in Ha/Bohr^2), or None without a Hessian.
    """
    if hess is None or grad is None:
        return None
    lam, modes = internal_modes(hess, coords)
    c = modes.T @ np.asarray(grad, float).ravel()
    positive = lam > 1e-12 * max(1.0, float(np.abs(lam).max()))
    nonpositive = int((~positive).sum())
    if not positive.any():
        return {'de_meV': None, 'bound': 'no positive curvature',
                'nonpositive_modes': nonpositive, 'softest_curvature': None}
    softest = float(lam[positive].min())
    if nonpositive:
        de, bound = 0.5 * float(c @ c) / softest, 'safe bound'
    else:
        de, bound = 0.5 * float((c ** 2 / lam).sum()), 'quadratic'
    return {'de_meV': de * HARTREE_TO_MEV, 'bound': bound,
            'nonpositive_modes': nonpositive, 'softest_curvature': softest}


def resolved_conv(conv):
    """GEOM_OPT_CONV with a caller's overrides, either spelling of the force.

    The force threshold is `opt_grad_max`, the residual at the converged
    geometry the convergence test compares against. The deprecated spelling
    `grad_max` sets the same threshold with a DeprecationWarning; the two
    given together and different raise rather than pick. The resolved dict
    carries both spellings of the one number.
    """
    asked = dict(conv or {})
    old = asked.pop('grad_max', None)
    if old is not None:
        if 'opt_grad_max' not in asked:
            warnings.warn(
                "conv={'grad_max': ...} names the geometry optimizer's force "
                "threshold under a superseded name; it is 'opt_grad_max', the "
                'residual at the converged geometry', DeprecationWarning,
                stacklevel=3)
            asked['opt_grad_max'] = old
        elif asked['opt_grad_max'] != old:
            raise ValueError(
                f'conv gives both spellings of the force threshold and they '
                f"disagree: opt_grad_max={asked['opt_grad_max']!r}, "
                f'grad_max={old!r}. They are one threshold under two names')
    out = dict(GEOM_OPT_CONV, **asked)
    out['grad_max'] = out['opt_grad_max']
    return out


def at_geometry(mol, coords):
    """`mol` moved to the Cartesian coordinates `coords` in Bohr, rebuilt."""
    m = mol.copy()
    m.set_geom_(np.asarray(coords).reshape(-1, 3), unit='Bohr')
    m.build(False, False)
    return m


def rank_zero_walk(mol_opt, info):
    """(mol, info) of a finished walk, rank 0's on every rank.

    Every rank took its steps from the same evaluated numbers, but the
    optimizer's own arithmetic (the `eigh` of the augmented RFO matrix, the
    BFGS update) is not guaranteed bitwise across ranks, so the result is
    locked.
    """
    return lockstep_geometry(mol_opt), lockstep(info)


def with_final_surface(surface, mol_opt, info):
    """`rank_zero_walk`'s (geometry, record) with `info['final_surface']`, the
    surface the walk ended on: `surface` itself, or the last refrozen one.

    Every rank built its own copy of that surface in lockstep, so it is kept
    beside the record and not broadcast with it.
    """
    final = info.pop('final_surface', surface)
    mol_opt, info = rank_zero_walk(mol_opt, info)
    info['final_surface'] = final
    return mol_opt, info


def excited_surface(surface):
    """Whether `surface` is an excited state's: it reports an excitation
    energy beside its total."""
    return hasattr(surface, 'excitation') or hasattr(surface,
                                                     'excitation_energy')


def nudge_vector(coords, amplitude=START_NUDGE_BOHR, seed=START_NUDGE_SEED):
    """(natm, 3) Bohr: a seeded random displacement with rigid motion
    projected out, its largest atomic displacement `amplitude`; zero where
    the geometry has no internal coordinate.

    Rank 0's bits on every rank, so every rank starts its walk at one
    geometry.
    """
    coords = np.asarray(coords, float).reshape(-1, 3)
    raw = np.random.default_rng(seed).standard_normal(coords.size)
    tr = translation_rotation_basis(coords)
    d = (raw - tr.T @ (tr @ raw)).reshape(-1, 3)
    largest = float(np.linalg.norm(d, axis=1).max())
    if amplitude == 0.0 or largest <= 1e-8 * np.sqrt(coords.size):
        return np.zeros_like(coords)
    return lockstep(d * (amplitude / largest))


def start_nudge(surface, mol, nudge=None):
    """(the geometry a walk starts at, the record of how it was moved).

    An exactly rotation-invariant surface keeps the point group of the walk's
    start, so a walk from a symmetric geometry can only reach the symmetric
    stationary point, a saddle where the state breaks the symmetry. An
    excited-state walk therefore starts `nudge_vector` away from `mol`.

    nudge: the largest atomic displacement in Bohr; None is
        START_NUDGE_BOHR on an excited-state surface (`excited_surface`) and
        0 on a ground-state one, 0 starts exactly at `mol`.

    The record keeps the amplitude, the seed, the displacement (natm, 3) in
    Bohr and its root-mean-square atomic displacement.
    """
    amplitude = (float(nudge) if nudge is not None
                 else START_NUDGE_BOHR if excited_surface(surface) else 0.0)
    d = nudge_vector(mol.atom_coords(), amplitude)
    record = {'amplitude_bohr': amplitude, 'seed': START_NUDGE_SEED,
              'norm': 'largest atomic displacement',
              'displacement_bohr': d,
              'rms_bohr': float(np.sqrt((d ** 2).sum(axis=1).mean()))}
    if not d.any():
        record.update(amplitude_bohr=0.0, seed=None, displacement_bohr=None)
        return mol, record
    return at_geometry(mol, mol.atom_coords() + d), record


def saddle_test(hessian, coords, tol=SADDLE_CURVATURE_TOL):
    """Whether a minimum's Hessian marks a saddle: its lowest internal
    curvature (Ha/Bohr^2, rigid motion projected out) below `tol`.

    Returns {'saddle', 'lowest_curvature', 'negative_modes' (curvatures
    below `tol`), 'tolerance', 'mode' (natm, 3), the unit Cartesian mode of
    the lowest curvature}.
    """
    coords = np.asarray(coords, float).reshape(-1, 3)
    lam, modes = internal_modes(hessian, coords)
    return {'saddle': bool(lam[0] < tol),
            'lowest_curvature': float(lam[0]),
            'negative_modes': int((lam < tol).sum()), 'tolerance': float(tol),
            'mode': modes[:, 0].reshape(coords.shape)}


def escape_saddle(surface, mol, hessian, walk, hessian_at,
                  step=SADDLE_ESCAPE_STEP_BOHR, tol=SADDLE_CURVATURE_TOL):
    """(minimum, the re-walk's record or None, its Hessian, the check's
    record) after the saddle test of a walk's minimum `mol` on `surface`.

    A minimum whose Hessian passes `saddle_test` is returned as it is. A
    saddle is left along its lowest mode by `step` Bohr: both directions are
    evaluated (at a stationary point the cubic term decides which is
    downhill), `walk(surface, start) -> (mol, info)` relaxes from the lower,
    on `info['final_surface']` where the walk names one, and
    `hessian_at(surface, mol)` is the Hessian at the new minimum, which is
    tested again. A second saddle is reported and not left: the check stops.

    The record: 'first' and 'second' (`saddle_test` of each Hessian), 'status'
    ('minimum', 'escaped', 'escape walk failed: ...' or 'second saddle:
    stopped'), and after an escape 'step_bohr', the 'direction' taken (+1 or
    -1 along the first mode), 'energies' at both displaced starts and the
    re-walk's 'energy_drop' (Hartree, saddle minus new minimum).
    """
    first = saddle_test(hessian, mol.atom_coords(), tol)
    record = {'first': first, 'second': None, 'status': 'minimum'}
    if not first['saddle']:
        return mol, None, hessian, record
    x = mol.atom_coords()
    starts = [at_geometry(mol, x + sign * step * first['mode'])
              for sign in (+1, -1)]
    energies = [float(lockstep(surface.total_energy(lockstep_geometry(m))))
                for m in starts]
    k = int(np.argmin(energies))
    at_saddle = float(lockstep(surface.total_energy(lockstep_geometry(mol))))
    moved, info = walk(surface, starts[k])
    record.update(step_bohr=float(step), direction=(+1, -1)[k],
                  energies=energies, energy_at_saddle=at_saddle)
    if not info.get('converged'):
        record['status'] = f'escape walk failed: {info.get("status")}'
        return moved, info, None, record
    record['energy_drop'] = at_saddle - float(info['energy'])
    final = info.get('final_surface', surface)
    hess = hessian_at(final, moved)
    second = saddle_test(hess, moved.atom_coords(), tol)
    record['second'] = second
    record['status'] = ('second saddle: stopped' if second['saddle']
                        else 'escaped')
    return moved, info, hess, record


def optimize(surface, mol=None, max_cycle=50, trust=0.1, trust_max=0.5,
             trust_min=2e-3, hess_init=None, conv=None, refreeze=0,
             verbose=True, nudge=None):
    """Relax `surface`'s state. Returns (mol, info).

    nudge: the walk starts `start_nudge(surface, mol, nudge)` away from
        `mol`: by default START_NUDGE_BOHR on an excited-state surface, so a
        symmetric start can reach a symmetry-broken minimum; 0 starts
        exactly at `mol`. `info['start_nudge']` records it. The refreeze
        passes start at the previous minimum and are not nudged.

    A trust-region quasi-Newton method: RFO step, BFGS Hessian, translations
    and rotations projected out; steps that turn out badly, or fail to
    evaluate, are rejected and retried smaller.

    `trust` starts at 0.1 Bohr rather than the 0.3 a ground-state optimizer
    would use: the first step is taken on a guessed Hessian, and a longer one
    can carry a quasiparticle root past where the frozen Newton seed finds it.
    Without rejection an excited-state surface can walk into nonsense (on
    water's dissociative S1, oscillation at the maximum radius until the
    Casida problem loses positive-definiteness).

    hess_init: an initial Cartesian Hessian, or None for a scaled identity.
        Passing the ground-state analytic Hessian (`normal_modes` builds one
        anyway for the Huang-Rhys factors) typically halves the cycle count:
        the excited surface has similar curvature to the ground one, which is
        what a quasi-Newton method needs and cannot guess. Another walk's
        final `info['hessian']` -- the singlet's at its minimum, for the
        triplet started there -- is the same kind of guess.

    `info['hessian']` is the BFGS Hessian the walk ended with, (3N, 3N) in
    Ha/Bohr^2, the refrozen walk's after a refreeze.
    refreeze: after convergence, rebuild the frozen conventions at the new
        geometry (`surface.refreeze`) and optimize again, up to this many
        times, until the energy has converged in them (`refreeze_passes`).
        0 reports the single-surface minimum with the drift unmeasured
        and marks the record `refreeze: 'not measured'`, so a null
        `refreeze_shift` can only have come from an explicit 0; 1 measures it:
        `info['refreeze_shift']` is how far the geometry moved, the error bar
        on the minimum: the largest atomic move once the refrozen minimum is
        superposed on the first (`body_frame.aligned_displacement`), since
        the surface is invariant to rigid motion; `refreeze_rigid_shifts`
        keeps the raw Cartesian moves. `info['final_surface']` is the surface the
        last walk ran on, where the reported energy belongs, and `converged`,
        `status`, `cycles` and `opt_grad_max` are that walk's
        (`refreeze_passes`).

    `info['opt_grad_max']` is max |dE/dR| at the converged geometry R*, the
    residual the convergence test was applied to -- not the driving force at
    the input geometry R0, which is what `grad_max` means wherever a gradient
    is reported at a fixed geometry. `info['grad_max']` is the same float, kept
    for callers that already read it.

    conv: thresholds overriding GEOM_OPT_CONV. Its force entry is
        `opt_grad_max`; the deprecated spelling `grad_max` sets the same
        threshold (`resolved_conv`).

    A state that stops existing along the path (dissociative, or a reference
    that goes singlet/triplet unstable) is reported as `info['status']`, not
    raised: the geometries up to that point are still the useful output.

    Under ranks every rank runs this walk on evaluations locked to rank 0's
    (`surface.evaluate`), and every rank comes back holding rank 0's geometry
    and record.
    """
    mol = surface.mol0 if mol is None else mol
    start, nudged = start_nudge(surface, mol, nudge)
    mol_opt, info = with_final_surface(surface, *_trust_region_walk(
        surface, start, max_cycle, trust, trust_max, trust_min, hess_init,
        conv, refreeze, verbose))
    info['start_nudge'] = nudged
    return mol_opt, info


def _trust_region_walk(surface, mol, max_cycle, trust, trust_max,
                       trust_min, hess_init, conv, refreeze, verbose):
    """The method `optimize` documents."""
    conv = resolved_conv(conv)
    n3 = 3 * mol.natm
    hess = (np.eye(n3) * 0.5 if hess_init is None
            else np.asarray(hess_init, float).reshape(n3, n3).copy())
    tr = translation_rotation_basis(mol.atom_coords())
    proj = np.eye(n3) - tr.T @ tr

    def at(xvec):
        m = at_geometry(mol, xvec)
        g, e, d = evaluate(surface, m)
        return proj @ g.ravel(), e, d, m

    x = mol.atom_coords().ravel().copy()
    try:
        g, e, diags, m_cur = at(x)
    except Exception as exc:
        return mol, {'converged': False, 'cycles': 0, 'history': [],
                     'status': f'{type(exc).__name__}: {exc}'}

    history, converged, status = [], False, 'ok'
    rejected = 0
    for cycle in range(max_cycle):
        gmax, grms = np.abs(g).max(), np.sqrt((g ** 2).mean())
        step = _rfo_step(hess, g, trust, proj)
        smax, srms = np.abs(step).max(), np.sqrt((step ** 2).mean())
        # `omega` is None on a ground-state surface, which has no excitation.
        omega = diags.get('omega')
        history.append({'cycle': cycle, 'e': e, 'omega': omega,
                        'grad_max': gmax, 'grad_rms': grms,
                        'step_max': smax, 'trust': trust, 'rejected': rejected})
        if verbose:
            om_txt = ('' if omega is None
                      else f'  Om = {omega * HARTREE_TO_EV:8.4f} eV')
            print(f'  [opt {cycle:3d}] E = {e:.10f}{om_txt}  '
                  f'|g|max {gmax:.2e} rms {grms:.2e}  step {smax:.2e}  '
                  f'trust {trust:.3f}', flush=True)
        if (gmax < conv['opt_grad_max'] and grms < conv['grad_rms']
                and smax < conv['step_max'] and srms < conv['step_rms']):
            converged = True
            break

        pred = g @ step + 0.5 * step @ (hess @ step)
        failed = None
        try:
            g_new, e_new, d_new, m_new = at(x + step)
        except Exception as exc:
            # A step that does not evaluate was too long, not a dead run: a
            # large displacement can carry a quasiparticle root past where the
            # reference-branch Newton seed finds it, or cost the Casida problem
            # its positive-definiteness. Both recover with a smaller radius;
            # only a radius that collapses is a failure.
            failed = f'{type(exc).__name__}: {exc}'
        ratio = 0.0 if failed else (
            (e_new - e) / pred if abs(pred) > 1e-14 else 0.0)

        if failed or (e_new > e and pred < 0):
            trust *= 0.25
            rejected += 1
            if trust < trust_min:
                status = (f'trust radius collapsed at {trust:.1e} Bohr; '
                          + (f'the state stopped evaluating ({failed})'
                             if failed else
                             'the surface is not locally quadratic here (a '
                             'crossing, or a state losing its identity)'))
                break
            history[-1]['rejected'] = rejected
            continue

        dx, dg = step, g_new - g
        sy = dg @ dx
        if sy > 1e-10 * np.linalg.norm(dx) * np.linalg.norm(dg):
            hdx = hess @ dx
            hess += np.outer(dg, dg) / sy - np.outer(hdx, hdx) / (dx @ hdx)
        trust = (min(trust * 1.3, trust_max) if ratio > 0.75
                 else trust * 0.5 if ratio < 0.25 else trust)
        trust = max(trust, trust_min)
        x, g, e, diags, m_cur = x + step, g_new, e_new, d_new, m_new

    if status == 'ok' and not converged:
        # Exhausting the cycle budget is not a clean finish: the last geometry
        # is not a minimum.
        status = f'max_cycle ({max_cycle}) reached without convergence'
    residual = float(np.abs(g).max())
    info = {'converged': converged, 'cycles': len(history), 'history': history,
            'energy': e, 'omega': diags.get('omega'),
            'grad_max': residual, 'opt_grad_max': residual,
            'rejected': rejected, 'status': status, 'optimizer': 'internal',
            'hessian': hess.copy(),
            'walk_resolution': (walk_resolution(g, hess, m_cur.atom_coords())
                                if converged else None)}

    if refreeze and converged:
        return refreeze_passes(
            lambda s, m, h: _trust_region_walk(s, m, max_cycle, trust,
                                               trust_max, trust_min, h, conv,
                                               0, verbose),
            surface, m_cur, info, refreeze, verbose)
    # A null shift says why it is null, so it is not read as a refreeze that
    # ran and found nothing; a run that never converged had no minimum to
    # refreeze at.
    info.update(refreeze_shift=None,
                refreeze=('not measured' if not refreeze else
                          'not measured: the first pass did not converge'))
    return m_cur, info


# ---------------------------------------------------------------------------
# geomeTRIC: internal coordinates, when it is installed
# ---------------------------------------------------------------------------

def optimize_geometric(surface, mol=None, maxiter=100, converge='GAU',
                       coordsys='tric', refreeze=0, verbose=True,
                       workdir=None, hess_init=None, nudge=None):
    """Relax `surface`'s state through geomeTRIC. Returns (mol, info).

    nudge: the symmetry-breaking start, as in `optimize` (`start_nudge`).

    Preferred over `optimize` beyond a handful of atoms: the Cartesian
    optimizer's step count grows with the system because Cartesians couple
    every internal degree of freedom to every other, while geomeTRIC's TRIC
    coordinates (bonds, angles, torsions plus rigid-body fragments) need far
    fewer cycles, each of which is a full BSE@GW gradient.

    It is an optional dependency and not in the environment by default:

        pip install --no-deps --target <dir> geometric   # numpy/scipy already present
        PYTHONPATH=<dir> python ...

    `--no-deps` matters: a plain install pulls numpy 2.x, which shadows the
    environment's 1.23 and breaks the compiled pyscf against it.

    geomeTRIC is imported, never vendored: its licence is BSD 3-clause with an
    added clause (no inclusion in machine-learning training datasets), and
    redistributing its source would take on the notice obligation and that
    restriction, while depending on it imposes nothing. Hence the import guard
    and the dependency-free Cartesian `optimize` fallback.

    The engine reports `surface.total_gradient` through `surface.evaluate`,
    so the optimization runs on one smooth surface, and `refreeze` measures
    its drift from the one the fit would choose at the end.

    refreeze: after convergence, rebuild the frozen conventions at the new
        geometry (`surface.refreeze`) and optimize again, up to this many
        times until the energy has converged in them, as `optimize` does, the
        shift measured on superposed geometries. 0 marks the record
        `refreeze: 'not measured'`, so a null `refreeze_shift` can only have
        come from an explicit 0.

    `info['opt_grad_max']` is max |dE/dR| at the converged geometry R*, the
    residual, and not the driving force at R0 that `grad_max` means elsewhere;
    `info['grad_max']` is the same float, kept for callers that read it.

    hess_init: a starting Cartesian Hessian in Ha/Bohr^2 (geomeTRIC's
        `hess_data`, transformed to its internal coordinates), or None for
        geomeTRIC's own guess; another walk's `info['hessian']` is one.
        `info['hessian']` is the approximate Cartesian Hessian geomeTRIC
        ends with (`write_cart_hess`), the refrozen walk's after a refreeze,
        and None after a walk that did not converge (geomeTRIC writes none
        then); a refrozen walk starts from the Hessian the walk before it
        ended with, as the Cartesian optimizer's does.

    A walk that does not converge is a record, not an exception, as in
    `optimize`: geomeTRIC's iteration cap (`GeomOptNotConvergedError`) comes
    back as `converged` False with status 'maxiter (N) reached without
    convergence', and any other failure of the driver or of an evaluation as
    `converged` False with its message. The geometry and energy are then the
    last ones evaluated, which is not a minimum, and no refreeze follows.
    Each walk (the first and every refrozen pass) writes its files under its
    own prefix in `workdir` (a fresh temporary directory when None).

    Under ranks every rank runs geomeTRIC on evaluations locked to rank 0's
    (`surface.evaluate`, in the engine's `calc_new` callback), and every rank
    comes back holding rank 0's geometry and record. geomeTRIC's files then go
    to one `workdir` from every rank, so give each rank its own there or leave
    it None, which makes a fresh directory per call.
    """
    # Probed before the walk starts, so a missing package raises before any
    # geometry is evaluated rather than partway through.
    geometric_engine()
    mol = surface.mol0 if mol is None else mol
    start, nudged = start_nudge(surface, mol, nudge)
    mol_opt, info = with_final_surface(surface, *_geometric_walk(
        surface, start, maxiter, converge, coordsys, refreeze, verbose,
        workdir, hess_init))
    info['start_nudge'] = nudged
    return mol_opt, info


def geometric_engine():
    """geomeTRIC's engine base, molecule, driver and the error its driver
    raises at the iteration cap, or a refusal naming the dependency-free
    alternative.

    An optional dependency, absent from the environment by default: importing
    it at module level would make every property routine need it.
    """
    try:
        # optional dependency, absent from the environment by default
        from geometric.engine import Engine
        from geometric.errors import GeomOptNotConvergedError
        from geometric.molecule import Molecule as GeoMolecule
        from geometric.optimize import run_optimizer
    except ImportError as exc:
        raise ImportError(
            'geomeTRIC is not importable; use `optimize` for the '
            'dependency-free Cartesian optimizer, or install geomeTRIC as in '
            '`optimize_geometric`\'s docstring.') from exc
    return Engine, GeoMolecule, run_optimizer, GeomOptNotConvergedError


def _geometric_walk(surface, mol, maxiter, converge, coordsys, refreeze,
                    verbose, workdir, hess_init=None, tag='es'):
    """The engine `optimize_geometric` documents; `tag` prefixes this walk's
    files in `workdir`."""
    Engine, GeoMolecule, run_optimizer, NotConverged = geometric_engine()

    gm = GeoMolecule()
    gm.elem = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    gm.xyzs = [np.asarray(mol.atom_coords()) * BOHR_TO_ANGSTROM]
    gm.build_topology()

    trace = []
    # the molecules `trace` was evaluated at: where a walk that did
    # not converge stopped
    visited = []
    # the gradient of the last evaluation, which is the converged geometry's
    last_gradient = []

    start = np.asarray(mol.atom_coords(), float).ravel()

    class _Surface(Engine):
        def calc_new(self, coords, dirname):
            coords = np.asarray(coords, float).ravel()
            if np.abs(coords - start).max() < GEOMETRIC_START_TOL:
                coords = start
            m = at_geometry(mol, coords)
            g, e, d = evaluate(surface, m)
            omega = d.get('omega')
            visited.append(m)
            last_gradient[:] = [np.asarray(g, float).ravel()]
            trace.append({'e': float(e),
                          'omega': None if omega is None else float(omega),
                          'grad_max': float(np.abs(g).max())})
            if verbose:
                om_txt = ('' if omega is None
                          else f'  Om = {omega * HARTREE_TO_EV:8.4f} eV')
                print(f'  [geomeTRIC {len(trace):3d}] E = {e:.10f}{om_txt}  '
                      f'|g|max {np.abs(g).max():.2e}', flush=True)
            return {'energy': float(e), 'gradient': np.asarray(g).ravel()}

    engine = _Surface(gm)
    # geomeTRIC's files go under an absolute prefix rather than a chdir: the
    # working directory belongs to the process, which the rank threads of
    # `run_simulated` share, and a chdir on one would move the others' files.
    tmp = workdir or tempfile.mkdtemp(prefix='esopt_')
    # the approximate Cartesian Hessian geomeTRIC ends with, written here
    hess_out = os.path.join(tmp, f'{tag}_final_hessian.txt')
    # geomeTRIC tests `hess_data` for truth, which an array refuses; a nested
    # list in Ha/Bohr^2 is its documented form, and the frequency analysis it
    # would run on a given Hessian is not wanted
    given = ({} if hess_init is None else
             {'hess_data': np.asarray(hess_init, float).tolist(),
              'frequency': False})
    try:
        out = run_optimizer(customengine=engine, coordsys=coordsys,
                            maxiter=maxiter, convergence_set=converge,
                            input=tag, prefix=os.path.join(tmp, tag), check=0,
                            write_cart_hess=hess_out, **given)
    except NotConverged:
        # geomeTRIC raises at its iteration cap rather than returning; the
        # walk up to there is still the output, as `optimize`'s is
        converged = False
        status = f'maxiter ({maxiter}) reached without convergence'
    except Exception as exc:
        # a step that failed to evaluate, or the driver itself: reported as
        # `optimize` reports a start that fails to evaluate
        converged = False
        status = f'{type(exc).__name__}: {exc}'
    else:
        converged, status = True, 'ok'

    if not trace:
        return mol, {'converged': False, 'cycles': 0, 'history': [],
                     'status': status, 'engine': 'geometric',
                     'optimizer': 'geometric', 'coordsys': coordsys,
                     'hessian': None, 'refreeze_shift': None,
                     'refreeze': ('not measured' if not refreeze else
                                  'not measured: the first pass did not '
                                  'converge')}
    if converged:
        mol_opt = at_geometry(mol, np.asarray(out.xyzs[-1]) / BOHR_TO_ANGSTROM)
    else:
        mol_opt = visited[-1]
    residual = trace[-1]['grad_max']
    hessian = (np.loadtxt(hess_out)
               if converged and os.path.exists(hess_out) else None)
    info = {'converged': converged, 'cycles': len(trace), 'history': trace,
            'energy': trace[-1]['e'], 'omega': trace[-1]['omega'],
            'grad_max': residual, 'opt_grad_max': residual, 'status': status,
            'engine': 'geometric', 'optimizer': 'geometric',
            'coordsys': coordsys, 'hessian': hessian,
            'walk_resolution': (walk_resolution(
                last_gradient[0], hessian, visited[-1].atom_coords())
                if converged and hessian is not None else None)}

    if refreeze and converged:
        passes = itertools.count(1)
        return refreeze_passes(
            lambda s, m, h: _geometric_walk(
                s, m, maxiter, converge, coordsys, 0, verbose, tmp, h,
                tag=f'{tag}_refreeze{next(passes)}'),
            surface, mol_opt, info, refreeze, verbose)
    info.update(refreeze_shift=None,
                refreeze=('not measured' if not refreeze else
                          'not measured: the first pass did not converge'))
    return mol_opt, info


#: What a single walk's record says about that walk alone: after a refreeze
#: these are the last walk's, the one whose geometry is returned.
WALK_FLAGS = ('converged', 'status', 'cycles', 'opt_grad_max', 'grad_max',
              'rejected', 'walk_resolution')


def refreeze_growth(surface):
    """What the refreeze that built `surface` did to an adaptive explicit
    set (the chain's `qp_growth`: the states carried from the surface before
    and the states added), or None for a surface without one."""
    half = driven_chain(getattr(surface, 'driven', surface))
    return getattr(half, 'qp_growth', None)


def resolved_de(info):
    """A walk record's dE_walk in meV (`walk_resolution`), or None."""
    res = info.get('walk_resolution')
    return None if res is None else res.get('de_meV')


def refreeze_passes(walk, surface, mol, info, passes, verbose):
    """(geometry, record) after rebuilding the frozen conventions at the
    minimum and relaxing again, until the energy has converged in them.

    A pass refreezes the last surface at the last minimum and walks
    (`walk(surface, mol, hessian)`, no refreeze of its own). The loop stops
    when the energy moved by no more than the pass's tolerance from the
    surface before, when the walk did not move (a surface rebuilt where it
    was built is the same surface; GEOMETRIC_START_TOL), when a walk did not
    converge, or after `passes`.

    The tolerance is what the walks resolve: max(REFREEZE_TOL_MEV, the
    larger dE_walk of the pass's walk and the walk before it), dE_walk = 1/2 g^T H^-1 g on the walk's own last gradient and
    Hessian (`walk_resolution`). Per pass the record keeps
    `refreeze_walk_de_meV` (the pass's walk; the first walk's is
    `first_walk_de_meV`), `refreeze_walk_bound` ('quadratic', 'safe bound'
    where the Hessian has a non-positive mode, None without one),
    `refreeze_tol_meV` and `refreeze_walk_decided` (the energy rule was met
    only through dE_walk). A converged loop whose last tolerance exceeds
    REFREEZE_TOL_FLAG_MEV sets `refreeze_tol_flag` and warns.

    `refreeze_shift` and `refreeze_denergy` are the first pass's: how far the
    conventions frozen at the start were from those at the minimum, in Bohr
    and Hartree. A shift is the largest atomic move once the walk's minimum is
    superposed on its start (`body_frame.aligned_displacement`): the surface
    is invariant to rigid motion, which an internal-coordinate walk is free
    to make, so a rigid move is not a move, and the stop on a walk that did
    not move reads the same number; `refreeze_rigid_shifts` keeps the raw
    Cartesian moves. `refreeze_shifts` and `refreeze_jumps` list every pass,
    and `refreeze_converged` says whether the loop met the stopping rule on a
    walk that converged. `energy`, `omega` and the returned geometry are the
    last surface's, which is `final_surface`: every energy a relaxation
    reports is read there.

    The record describes the walk whose geometry it returns. `converged`,
    `status`, `cycles`, `opt_grad_max` (and `grad_max`, `rejected`) are the
    last walk's, so `converged` False means the geometry returned is not a
    minimum of the surface it stands on; the first walk's are kept as
    `first_walk_converged`, `first_walk_status`, ... Two separate verdicts,
    then: `converged`, whether the last walk reached a minimum, and
    `refreeze_converged`, whether the conventions stopped moving there.
    `history` stays the first walk's, the one that starts at the input
    geometry; the last walk's is in `refreeze_info`.

    An adaptive explicit set grows only (`ExcitedStateChain.refreeze`), and
    `refreeze_qp_growth` lists, per pass, the states it carried and added
    (`refreeze_growth`; None for a surface without one).
    """
    shifts, rigid, jumps, growth = [], [], [], []
    walk_de, bounds, tols, decided = [], [], [], []
    first_de = resolved_de(info)
    last, current, done = info, surface, False
    for _ in range(int(passes)):
        current = current.refreeze(mol)
        moved, walked = walk(current, mol, last['hessian'])
        shifts.append(aligned_displacement(mol.atom_coords(),
                                           moved.atom_coords()))
        rigid.append(float(np.abs(moved.atom_coords()
                                  - mol.atom_coords()).max()))
        jumps.append(float(walked['energy'] - last['energy']))
        growth.append(refreeze_growth(current))
        before, now = (resolved_de(last), resolved_de(walked))
        res = walked.get('walk_resolution')
        walk_de.append(now)
        bounds.append(None if res is None else res['bound'])
        resolved = max((d for d in (before, now) if d is not None),
                       default=None)
        tols.append(max(REFREEZE_TOL_MEV, resolved or 0.0))
        jump = abs(jumps[-1]) * HARTREE_TO_MEV
        on_floor = jump < REFREEZE_TOL_MEV
        decided.append(bool(not on_floor and resolved is not None
                            and jump <= resolved))
        last, mol = walked, moved
        if verbose:
            added = ('' if growth[-1] is None else
                     f', explicit states added {growth[-1]["added"]}')
            de_txt = ('' if now is None else
                      f', dE_walk {now:.3f} meV, tolerance {tols[-1]:.3f} meV')
            print(f'  [refreeze {len(jumps)}] geometry moved {shifts[-1]:.2e} '
                  f'Bohr superposed ({rigid[-1]:.2e} raw), energy by '
                  f'{jumps[-1]:+.2e} Ha{de_txt}{added}', flush=True)
        # a walk that stayed within GEOMETRIC_START_TOL of where its surface
        # was rebuilt stands where the next one would be rebuilt
        done = (on_floor or decided[-1]
                or shifts[-1] <= GEOMETRIC_START_TOL)
        if done or not last['converged']:
            break
    converged = bool(done and last['converged'])
    flagged = bool(converged and tols[-1] > REFREEZE_TOL_FLAG_MEV)
    if flagged:
        warnings.warn(
            f'the refreeze passes converged only to {tols[-1]:.3f} meV, what '
            f'the walks resolve (dE_walk), above REFREEZE_TOL_FLAG_MEV = '
            f'{REFREEZE_TOL_FLAG_MEV} meV: an energy at this minimum is no '
            f'better than that', RuntimeWarning, stacklevel=2)
    flags = [key for key in WALK_FLAGS if key in last]
    info.update({f'first_walk_{key}': info.get(key) for key in flags})
    info.update({key: last[key] for key in flags})
    info.update(refreeze_shift=shifts[0], refreeze_denergy=jumps[0],
                refreeze_shifts=shifts, refreeze_rigid_shifts=rigid,
                refreeze_jumps=jumps,
                refreeze_qp_growth=growth,
                first_walk_de_meV=first_de,
                refreeze_walk_de_meV=walk_de, refreeze_walk_bound=bounds,
                refreeze_tol_meV=tols, refreeze_walk_decided=decided,
                refreeze_tol_flag=flagged,
                refreeze_converged=converged,
                refreeze_info=last, energy=last['energy'],
                omega=last['omega'], refreeze='measured',
                hessian=last['hessian'], final_surface=current)
    return mol, info


def relax(surface, mol=None, engine='auto', **kw):
    """Relax a state, preferring internal coordinates when available.

    engine: 'geometric', 'cartesian', or 'auto' (geomeTRIC if importable).
    Either starts an excited-state walk nudged off `mol` unless `nudge=0`
    (`start_nudge`).

    `refreeze` reaches either optimizer: both measure the drift of the frozen
    conventions the same way, so which engine ran does not change what the
    number means. Which one ran is in `info['optimizer']`, written by the
    engine itself, since the two converge to different residuals on the same
    minimum.
    """
    if engine == 'cartesian':
        mol_opt, info = optimize(surface, mol, **kw)
    elif engine == 'geometric':
        mol_opt, info = optimize_geometric(surface, mol, **kw)
    else:
        try:
            # availability probe: the two take different keywords otherwise
            import geometric      # noqa: F401
        except ImportError:
            mol_opt, info = optimize(
                surface, mol, **{k: v for k, v in kw.items()
                                 if k not in ('coordsys', 'converge',
                                              'maxiter', 'workdir')})
        else:
            mol_opt, info = optimize_geometric(surface, mol, **kw)
    info.setdefault('optimizer', engine)
    return mol_opt, info


def mean_field_force(mf):
    """dE/dR of the energy this mean field reported, whatever built its exchange.

    The definition is `isdf_jk.mean_field_skeleton_force`, which keeps
    `grid_response`: pyscf defaults it to False, dropping d(becke weights)/dR,
    and that force breaks translational invariance by 8.1e-06 Ha/Bohr on
    water/cc-pVDZ/B3LYP against 4.2e-15 with it.

    Rank 0's force on every rank: pyscf's threaded GEMM adds its partial sums
    in thread-arrival order and its density-fitted gradient blocks by each
    process's free memory, so every rank's own call differs in its last bits.
    """
    return lockstep(mean_field_skeleton_force(mf))


class MeanFieldSurface:
    """The mean field's own energy as a `PotentialEnergySurface`.

    The ground state needs no excited-state machinery and no frozen
    conventions, so `refreeze` is the identity. It lets `optimize` relax a
    ground state with neither geomeTRIC nor pyberny installed.

    The SCF is the only stage, and it is what the ranks divide: a factory that
    hands back a mean field built but not run leaves the convergence to
    `converged_factory`, where every rank runs pyscf's driver against the
    reduced J/K and quadrature, contributing its block of the auxiliary index
    and of the grid. A factory that converges its own is used as it is, its
    orbitals locked to rank 0's. Each rank forms the force from rank 0's
    orbitals and `mean_field_force` hands every rank rank 0's; the energy is
    the locked mean field's own.

    In a continuum (`environment`, a `SolventScreening`) every geometry's SCF
    is the environment's ground state, PCM at eps_static on a cavity rebuilt
    around that geometry (`for_geometry`), and the force is pyscf's PCM
    gradient of it, or the ISDF one completed with the reaction field's term.
    This is the surface for e.g. a UKS geometry of an ion in solvent, where no
    correlated surface reaches.
    """

    def __init__(self, mol, scf_factory, mf=None, environment=None,
                 own=False):
        """`mf` is the reference mean field where the caller already has one.

        The declaration (which functional E_0 is, a property of the reference
        and not of a geometry) reads it, so a surface built without one
        converges its own the first time it is asked. With `own` it is also
        the mean field this surface evaluates at `mol` (the gas phase, where
        the factory's mean field is the reference), handed back by
        `mean_field(mol)` rather than converged again.
        """
        self.mol0 = mol
        self._scf = scf_factory
        self._mf0 = mf
        self.environment = resolve_environment(environment, None)
        self._own = bool(own) and mf is not None
        # The ISDF-K pair layout of the factory's mean field at mol0, which
        # the mean field at every geometry of this surface fits on: [] until
        # read, then [layout], None for a mean field without an ISDF-K fit.
        self._layout = []

    def _mean_field(self, mol):
        """The factory's mean field at `mol` in this surface's environment,
        converged over the ranks if it was not already, rank 0's on every
        rank, its ISDF-K fit on the pair layout of mol0's (`pair_layout`)."""
        at_reference = np.array_equal(mol.atom_coords(),
                                      self.mol0.atom_coords())
        layout = (None if at_reference and not self._layout
                  else self.pair_layout())
        with frozen_pair_layout(layout):
            mf = self._converged(mol)
        if not self._layout:
            self._layout.append(scf_pair_layout(mf))
        return mf

    def _converged(self, mol):
        """The factory's mean field at `mol` in this environment, converged
        over the ranks, rank 0's on every rank."""
        # cycle: src.gradients -> src.properties.__init__ -> this module
        from src.gradients.factor_chain import converged_factory

        return lockstep_mean_field(
            self.environment.for_geometry(mol).mean_field(
                mol, converged_factory(self._scf)))

    def pair_layout(self):
        """The AO-pair columns the ISDF-K fit of the mean field at mol0
        kept, which every geometry of this surface fits on, so that its
        energy is one function of the nuclei and the force its derivative;
        a surface asked elsewhere first converges mol0's for it. None for a
        mean field without an ISDF-K fit. `refreeze` takes it again at the
        new reference."""
        if not self._layout:
            mf = (self._mf0 if self._own else
                  self._converged(self.mol0))
            self._layout.append(scf_pair_layout(mf))
        return self._layout[0]

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the mean field this surface evaluates on: the factory's
        in this surface's environment, `mf` as given, or the reference
        geometry's own reference where the surface owns one."""
        mol = self.mol0 if mol is None else mol
        if mf is None and self._own and mol is self.mol0:
            return mol, self._mf0
        return mol, (self._mean_field(mol) if mf is None else mf)

    def scf_factory(self, mol):
        return self._scf(mol)

    @property
    def physics_ground_state(self):
        """E_0 = E_KS[xc], the mean field's own energy; 'hf' for Hartree-Fock."""
        if self._mf0 is None:
            self._mf0 = self._mean_field(self.mol0)
        return declared_ground_state(self._mf0, 'dft')

    @property
    def physics(self):
        """What this surface computes: E_0 alone, with no state on it, in
        the environment its SCF is converged in."""
        return SurfacePhysics(self.physics_ground_state, None,
                              environment_label(self.environment))

    def total_energy(self, mol=None, mf=None):
        mol = mol if mol is not None else self.mol0
        return float((mf or self._mean_field(mol)).e_tot)

    def total_gradient(self, mol=None, mf=None):
        mol = mol if mol is not None else self.mol0
        mf = mf or self._mean_field(mol)
        return np.asarray(mean_field_force(mf)), float(mf.e_tot), {}

    def refreeze(self, mol):
        return MeanFieldSurface(mol, self._scf, environment=self.environment)

    def label(self):
        return 'mean-field ground state'


def relax_ground_state(mol, scf_factory, engine='auto', maxsteps=100,
                       verbose=False, converge=None, environment=None):
    """Relax the mean-field ground state. Returns (mol, mf).

    The vibronic quantities are defined about this geometry, not the input
    one: Huang-Rhys factors expand both surfaces in the ground state's normal
    modes, which are modes only at a stationary point, and a reorganization
    energy from a non-stationary reference includes the ground state's own
    relaxation (on formaldehyde/cc-pVDZ the unrelaxed input moves the total
    Huang-Rhys factor from 1.278 to 0.896).

    environment: what the ground state stands in, None for the gas phase. The
    gas phase relaxes the factory's own mean field, through pyscf's geomopt
    drivers where they are installed. An environment relaxes the mean field
    `MeanFieldSurface` builds in it (PCM at eps_static for a continuum)
    through this package's optimizers, since pyscf's drivers step on the
    factory's bare SCF: geomeTRIC's internal coordinates when it is
    importable and `engine` is not 'cartesian', else the Cartesian trust
    region. The mean field returned is that surface's own at the minimum.
    """
    if environment is not None:
        surface = MeanFieldSurface(mol, scf_factory, environment=environment)
        opt, info = None, None
        if engine != 'cartesian':
            try:
                opt, info = optimize_geometric(
                    surface, mol, maxiter=maxsteps,
                    converge='GAU' if converge is None else converge,
                    verbose=verbose)
            except ImportError as exc:
                if verbose:
                    print(f'{exc}; using the Cartesian trust-region optimizer')
        if info is None:
            opt, info = optimize(surface, mol, max_cycle=maxsteps, trust=0.3,
                                 verbose=verbose)
        warn_unrelaxed(info)
        final = info.get('final_surface', surface)
        return opt, final.mean_field(opt)[1]
    mf = scf_factory(mol)
    # pyscf's scanner resets this mean field at each geometry, and its ISDF-K
    # fit then keeps the start's pair layout (`ISDFJK.reset`)
    if isinstance(getattr(mf, 'with_df', None), ISDFJK):
        mf.with_df.pair_layout = scf_pair_layout(mf)
    tried = []
    for name, mod in (('geometric', 'geometric_solver'),
                      ('pyberny', 'berny_solver')):
        if engine == 'cartesian':
            break
        try:
            solver = __import__(f'pyscf.geomopt.{mod}', fromlist=[mod])
        except ImportError as exc:
            # geomeTRIC needs numpy, scipy, networkx and six; a --no-deps
            # install that forgets one fails like an absent package.
            tried.append(f'{name} ({exc})')
            continue
        kw = {} if converge is None else {'convergence_set': converge}
        opt = solver.optimize(mf, maxsteps=maxsteps, **kw)
        return opt, scf_factory(opt)
    # Neither external optimizer is installed: fall back to the Cartesian
    # trust-region method, which needs nothing outside src/.
    if verbose:
        print(f'no external optimizer ({"; ".join(tried)}); using the '
              f'Cartesian trust-region optimizer')
    opt, info = optimize(MeanFieldSurface(mol, scf_factory), mol,
                         max_cycle=maxsteps, trust=0.3, verbose=verbose)
    warn_unrelaxed(info)
    return opt, scf_factory(opt)


def warn_unrelaxed(info):
    """Warn that a ground-state relaxation's record says it did not converge."""
    if not info.get('converged'):
        residual = info.get('grad_max')
        warnings.warn(
            f'the ground-state relaxation stopped after {info.get("cycles")} '
            f'cycles without converging ({info.get("status")}; max |dE/dR| = '
            f'{float("nan") if residual is None else residual:.2e}); every '
            f'vibronic quantity is defined about a STATIONARY point, so treat '
            f'what follows as provisional.', RuntimeWarning, stacklevel=3)


def ground_state_residual_force(mf, mol=None):
    """max |dE_0/dR| -- how far the reference is from its own minimum.

    Worth printing next to any Huang-Rhys spectrum: it is the size of the term
    the harmonic expansion takes as zero.
    """
    return float(np.abs(mf.Gradients().kernel()).max())
