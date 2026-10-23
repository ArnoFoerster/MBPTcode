"""Geometry optimization on any `PotentialEnergySurface`.

A Cartesian trust-region quasi-Newton method with no dependency, geomeTRIC's
internal coordinates when they are installed, and the mean-field ground-state
relaxation the vibronic quantities are defined about.

THE FROZEN CONVENTIONS AND A MOVING GEOMETRY. A cubic chain fixes five discrete
choices at a reference geometry -- the quasiparticle set, the frame
orientation, the interpolation pair layout, the Newton branch and the residue
backend -- because each is a discontinuity in the surface otherwise. An
optimizer moves far enough that the *layout* chosen at the start is no longer
the one the fit would choose at the end. Both cannot be had at once: refreezing
each step makes every step's gradient exact for its own surface and makes the
ENERGY discontinuous between steps, which is fatal to a line search or a BFGS
history.

So both optimizers freeze and keep it for a whole walk, which makes each walk
one smooth surface, and refreeze in an OUTER LOOP: `surface.refreeze` at the
converged geometry R* rebuilds the conventions this geometry would choose,
the drift E_new(R*) - E_old(R*) is measured there, at one geometry, and the
state is relaxed again on the rebuilt surface (`refreeze_pass`). A drift
within REFREEZE_TOL_MEV ends the loop; above it the next pass refreezes at
the new minimum. A fixed-point iteration of a geometry and a discrete
convention set has no guarantee of converging, so the loop stops after
REFREEZE_MAX_PASSES; a drift still above the tolerance is a numerical event
(refuse inputs, warn numerics): warned about
(RefreezeNotConverged), the last pass's geometry kept, the record flagged.

ONE WALK, HOWEVER MANY RANKS. Inside `with distributed(comm):` every rank runs
the walk below, and every evaluation goes through `surface.evaluate`, which
locks the geometry to rank 0's on the way in and the energy, force and
diagnostics on the way out. Every decision -- the step, the convergence test,
the trust radius, whether a step was rejected, where the refreeze happens --
is therefore taken on every rank from the same numbers, and the walk's result
is locked to rank 0's once more at the end, so every rank returns one geometry
and one record. Outside a distributed region every lockstep is a no-op and the
walk is bit for bit the serial one.
"""
import inspect
import os
import tempfile
import warnings
from dataclasses import dataclass

import numpy as np
from pyscf import grad  # noqa: F401  registers mf.Gradients/nuc_grad_method

try:  # optional: only `optimize_geometric` needs geomeTRIC
    from geometric.engine import Engine
    from geometric.errors import GeomOptNotConvergedError
    from geometric.molecule import Molecule as GeoMolecule
    from geometric.optimize import run_optimizer
    HAVE_GEOMETRIC = True
except ImportError:
    HAVE_GEOMETRIC = False

from src.Base.geometry.body_frame import aligned_displacement, centred, kabsch_rotation
from src.Base.numerics.constants import (BOHR_TO_ANGSTROM, GEOM_OPT_CONV,
                                         GEOMETRIC_START_TOL, HARTREE_TO_EV,
                                         HARTREE_TO_MEV, REFREEZE_MAX_PASSES,
                                         REFREEZE_TOL_MEV,
                                         SADDLE_CURVATURE_TOL,
                                         SADDLE_ESCAPE_STEP_BOHR, START_NUDGE_BOHR,
                                         START_NUDGE_SEED, WALK_STEP_SHRINK,
                                         WALK_TRUST_MIN_BOHR)
from src.Base.factorization.distributed_isdf_jk import scf_pair_layout
from src.Base.geometry.geometry import molecule_at
from src.Base.utils.record_io import record_class, record_hash
from src.Base.factorization.isdf_jk import ISDFJK
from src.Base.numerics.numerics import Numerics, surface_numerics
from src.Base.step_failed import StepFailed
from src.Base.utils.parallel.mpi_grid import lockstep
from src.gradients.mean_field_surface import MeanFieldSurface, mean_field_force
from src.gradients.surface_identity import force_identity
from src.gradients.surface import evaluate, lockstep_geometry
from src.gradients.surface import commit, driven_chain, provides


class NonFiniteEvaluation(StepFailed):
    """A surface returned a NaN or infinite energy or force at a geometry (a
    root its solve did not return: the surface's flags say which). Raised
    inside the walkers only, so the walk rejects the step and retries it
    shorter as for any StepFailed; the surface itself returns the NaN,
    flagged."""


def walk_evaluate(surface, mol):
    """surface.evaluate for a walk: a non-finite energy or force is a
    rejected step (NonFiniteEvaluation)."""
    g, e, d = evaluate(surface, mol)
    if not (np.isfinite(e) and np.isfinite(g).all()):
        names = getattr(surface, 'flags_at', lambda m: [])(mol)
        raise NonFiniteEvaluation(
            f'the surface returned energy {e} and a force with '
            f'{int((~np.isfinite(g)).sum())} non-finite entries'
            + (f' (flags {", ".join(names)})' if names else ''))
    return g, e, d


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
    A diagnostic of the walk alone; the refreeze drift is measured at one
    geometry and does not read it.

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
    energy beside its total (the `excitation` ingredient)."""
    return provides(surface, 'excitation')


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
    return molecule_at(mol, mol.atom_coords() + d), record


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
                  step=SADDLE_ESCAPE_STEP_BOHR, tol=SADDLE_CURVATURE_TOL,
                  cache_dir=None):
    """(minimum, the re-walk's record or None, its Hessian, the check's
    record) after the saddle test of a walk's minimum `mol` on `surface`.

    A minimum whose Hessian passes `saddle_test` is returned as it is. A
    saddle is left along its lowest mode by `step` Bohr: both directions are
    evaluated (at a stationary point the cubic term decides which is
    downhill), `walk(surface, start) -> (mol, info)` relaxes from the lower,
    on `info['final_surface']` where the walk names one, and
    `hessian_at(surface, mol)` is the Hessian at the new minimum, which is
    tested again. A second saddle is reported and not left: the check stops.
    cache_dir: the first Hessian's force cache (`surface_hessian`); the
    second Hessian keeps its forces in its own subdirectory 'second',
    `hessian_at(surface, mol, cache_dir=...)`.

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
    starts = [molecule_at(mol, x + sign * step * first['mode'])
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
    hess = (hessian_at(final, moved) if cache_dir is None else
            hessian_at(final, moved,
                       cache_dir=os.path.join(cache_dir, 'second')))
    second = saddle_test(hess, moved.atom_coords(), tol)
    record['second'] = second
    record['status'] = ('second saddle: stopped' if second['saddle']
                        else 'escaped')
    return moved, info, hess, record


def optimize(surface, mol=None, max_cycle=None, trust=None, trust_max=None,
             trust_min=None, hess_init=None, conv=None,
             refreeze=0, verbose=True, nudge=None, checkpoint=None,
             resume=None):
    """Relax `surface`'s state. Returns (mol, info).

    max_cycle, trust, trust_max, trust_min: the walk's settings; a keyword not
        given is the surface's `Numerics.walk` (`surface_numerics`), the
        `WalkNumerics` defaults on a surface that holds none.

    nudge: the walk starts `start_nudge(surface, mol, nudge)` away from
        `mol`: by default START_NUDGE_BOHR on an excited-state surface, so a
        symmetric start can reach a symmetry-broken minimum; 0 starts
        exactly at `mol`. `info['start_nudge']` records it. The refreeze
        passes start at the previous minimum and are not nudged.

    A trust-region quasi-Newton method: RFO step, BFGS Hessian, translations
    and rotations projected out, and steps that turn out badly -- or whose
    evaluation raises `StepFailed` -- are REJECTED and retried smaller rather
    than merely followed by a smaller radius. Any other exception an
    evaluation raises is not about the step and leaves the walk.

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
    refreeze: 1 runs the refreeze loop: rebuild the frozen conventions at
        the converged geometry (`surface.refreeze`), measure the drift there,
        relax on the rebuilt surface, and again while the drift is above
        REFREEZE_TOL_MEV, at most REFREEZE_MAX_PASSES passes
        (`refreeze_pass`; refused beyond). 0 reports the single-surface
        minimum with the drift unmeasured and marks the record
        `refreeze: 'not measured'` (`refreeze_asked`); a first pass that did
        not converge is marked 'not measured: the first pass did not
        converge'. Both leave `refreeze_drift` and `refreeze_shift` null.
        `info['final_surface']` is the surface the last walk ran on, where
        the reported energy belongs, and `converged`, `status`, `cycles` and
        `opt_grad_max` are that walk's.

    `info['opt_grad_max']` is max |dE/dR| at the converged geometry R*, the
    residual the convergence test was applied to -- not the driving force at
    the input geometry R0, which is what `grad_max` means wherever a gradient
    is reported at a fixed geometry. `info['grad_max']` is the same float, kept
    for callers that already read it.

    conv: thresholds overriding GEOM_OPT_CONV. Its force entry is
        `opt_grad_max`; the deprecated spelling `grad_max` sets the same
        threshold (`resolved_conv`).

    A state that stops existing along the path -- dissociative, or a reference
    that goes singlet/triplet unstable, each a `StepFailed` at every radius --
    is reported as `info['status']`, not raised: the geometries up to that
    point are still the useful output.

    The start and every accepted step are committed to the surface's root
    following (`surface.commit`); a rejected step never is.

    checkpoint: `checkpoint(state)` after the start and every accepted
        step of either pass, `state` the walk at that point (`walk_state`);
        resume: such a state, read back, from which the walk goes on instead
        of starting at `mol` (in the refrozen pass, `resumed_refreeze`), on
        the surface rebuilt from its record (`restored_surface`), taking the
        steps the uninterrupted walk would have taken to the bit.
        `info['resume']` says which pass it resumed in and how many
        evaluations it reused.

    Under ranks every rank runs this walk on evaluations locked to rank 0's
    (`surface.evaluate`), and every rank comes back holding rank 0's geometry
    and record.
    """
    walk = (surface_numerics(surface) or Numerics()).walk
    max_cycle = walk.max_cycle if max_cycle is None else max_cycle
    trust = walk.trust if trust is None else trust
    trust_max = walk.trust_max if trust_max is None else trust_max
    trust_min = walk.trust_min if trust_min is None else trust_min
    mol = surface.mol0 if mol is None else mol
    refreeze = refreeze_asked(refreeze)
    start, nudged = start_nudge(surface, mol, nudge)
    mol_opt, info = with_final_surface(surface, *_trust_region_walk(
        surface, start, max_cycle, trust, trust_max, trust_min, hess_init,
        conv, refreeze, verbose, checkpoint=checkpoint, resume=resume))
    info['start_nudge'] = nudged
    return mol_opt, info


def refrozen_walker(walk, **fixed):
    """`walk_again(surface, mol, hessian, checkpoint, resume)` for the refreeze
    pass: `walk` run as a refrozen walk on `fixed`, from the given Hessian."""
    def walk_again(s, m, h, checkpoint=None, resume=None):
        return walk(s, m, hess_init=h, refrozen=True, checkpoint=checkpoint,
                    resume=resume, **fixed)
    return walk_again


@dataclass
class Trial:
    """One trial step of the Cartesian walk: its evaluation (`failed` holds
    the `StepFailed` message instead), the model-predicted energy change and
    the actual-to-predicted ratio."""
    failed: object
    g: object
    e: object
    d: object
    m: object
    pred: float
    ratio: float


class CartesianWalk:
    """The state of one Cartesian trust-region walk: the geometry, energy,
    projected gradient and BFGS Hessian at the last accepted point, the
    trust radius and the rejection count, and the record of the cycles."""

    def __init__(self, surface, mol, hess_init, trust, trust_max, trust_min,
                 refrozen, checkpoint):
        self.surface, self.mol = surface, mol
        self.trust, self.trust_max, self.trust_min = trust, trust_max, trust_min
        self.refrozen, self.checkpoint = refrozen, checkpoint
        n3 = 3 * mol.natm
        self.hess = (np.eye(n3) * 0.5 if hess_init is None
                     else np.asarray(hess_init, float).reshape(n3, n3).copy())
        tr = translation_rotation_basis(mol.atom_coords())
        self.proj = np.eye(n3) - tr.T @ tr
        self.history, self.converged, self.status = [], False, 'ok'
        self.rejected = 0

    def at(self, xvec):
        """(projected gradient, energy, diagnostics, molecule) at `xvec`."""
        m = molecule_at(self.mol, xvec)
        g, e, d = walk_evaluate(self.surface, m)
        return self.proj @ g.ravel(), e, d, m

    def save(self):
        """The checkpoint of the walk as it stands."""
        if self.checkpoint is not None:
            self.checkpoint(walk_state(
                self.surface, 1 if self.refrozen else 0, 'internal',
                self.m_cur.atom_coords(), self.e, self.g, self.diags,
                self.hess, self.history, trust=self.trust,
                rejected=self.rejected))

    def start_or_resume(self, resume):
        """Evaluate the start, or take the walk up as it stood after its last
        accepted step (no evaluation). Returns the record of a start that
        fails to evaluate, else None."""
        if resume is not None:
            self.x = np.asarray(resume['coords'], float).ravel().copy()
            self.g = np.asarray(resume['gradient'], float)
            self.e = resume['energy']
            self.diags = {'omega': resume['omega']}
            self.hess = resume['hessian'].copy()
            self.trust, self.rejected = resume['trust'], resume['rejected']
            self.history = list(resume['history'])
            self.m_cur = molecule_at(self.mol, self.x)
            return None
        self.x = self.mol.atom_coords().ravel().copy()
        try:
            self.g, self.e, self.diags, self.m_cur = self.at(self.x)
        except StepFailed as exc:
            # a refrozen surface that does not evaluate at the minimum of the
            # surface it was rebuilt from is broken, not a walk that went
            # nowhere
            if self.refrozen:
                raise
            return {'converged': False, 'cycles': 0, 'history': [],
                    'status': f'{type(exc).__name__}: {exc}'}
        commit(self.surface, self.m_cur)
        self.save()
        return None

    def plan_step(self, cycle, verbose):
        """The RFO step of this cycle; its cycle is recorded and reported."""
        g, e, trust = self.g, self.e, self.trust
        self.gmax, self.grms = np.abs(g).max(), np.sqrt((g ** 2).mean())
        step = _rfo_step(self.hess, g, trust, self.proj)
        self.smax = np.abs(step).max()
        self.srms = np.sqrt((step ** 2).mean())
        # `omega` is None on a ground-state surface, which has no excitation.
        omega = self.diags.get('omega')
        self.history.append({'cycle': cycle, 'e': e, 'omega': omega,
                             'grad_max': self.gmax, 'grad_rms': self.grms,
                             'step_max': self.smax, 'trust': trust,
                             'rejected': self.rejected})
        if verbose:
            om_txt = ('' if omega is None
                      else f'  Om = {omega * HARTREE_TO_EV:8.4f} eV')
            print(f'  [opt {cycle:3d}] E = {e:.10f}{om_txt}  '
                  f'|g|max {self.gmax:.2e} rms {self.grms:.2e}  '
                  f'step {self.smax:.2e}  trust {trust:.3f}', flush=True)
        return step

    def has_converged(self, conv):
        """Whether the gradient and the planned step are within `conv`."""
        return bool(self.gmax < conv['opt_grad_max']
                    and self.grms < conv['grad_rms']
                    and self.smax < conv['step_max']
                    and self.srms < conv['step_rms'])

    def take_step(self, step):
        """Evaluate the trial geometry `x + step` as a `Trial`."""
        pred = self.g @ step + 0.5 * step @ (self.hess @ step)
        failed = g_new = e_new = d_new = m_new = None
        try:
            g_new, e_new, d_new, m_new = self.at(self.x + step)
        except StepFailed as exc:
            # a root lost, a Casida problem broken or a followed state gone at
            # the trial geometry: the step was too long, and only a radius
            # that collapses is a real failure
            failed = f'{type(exc).__name__}: {exc}'
        ratio = 0.0 if failed else (
            (e_new - self.e) / pred if abs(pred) > 1e-14 else 0.0)
        return Trial(failed, g_new, e_new, d_new, m_new, pred, ratio)

    def is_rejected(self, trial):
        """A failed evaluation, or an energy rise against a predicted fall."""
        return bool(trial.failed
                    or (trial.e > self.e and trial.pred < 0))

    def reject_step(self, trial):
        """Shrink the radius after a rejected step; True when it has
        collapsed below `trust_min` and the walk ends."""
        self.trust *= WALK_STEP_SHRINK
        self.rejected += 1
        if self.trust < self.trust_min:
            failed = trial.failed
            self.status = (
                f'trust radius collapsed at {self.trust:.1e} Bohr; '
                + (f'the state stopped evaluating ({failed})'
                   if failed else
                   'the surface is not locally quadratic here (a '
                   'crossing, or a state losing its identity)'))
            return True
        self.history[-1]['rejected'] = self.rejected
        return False

    def accept_step(self, step, trial):
        """BFGS-update the Hessian, adapt the radius, move, commit, save."""
        dx, dg = step, trial.g - self.g
        sy = dg @ dx
        if sy > 1e-10 * np.linalg.norm(dx) * np.linalg.norm(dg):
            hdx = self.hess @ dx
            self.hess += (np.outer(dg, dg) / sy
                          - np.outer(hdx, hdx) / (dx @ hdx))
        trust, ratio = self.trust, trial.ratio
        trust = (min(trust * 1.3, self.trust_max) if ratio > 0.75
                 else trust * 0.5 if ratio < 0.25 else trust)
        self.trust = max(trust, self.trust_min)
        self.x, self.g, self.e = self.x + step, trial.g, trial.e
        self.diags, self.m_cur = trial.d, trial.m
        commit(self.surface, self.m_cur)
        self.save()

    def descend(self, budget, conv, verbose):
        """Cycle until converged, the radius collapses or the budget ends."""
        for cycle in range(len(self.history), budget):
            step = self.plan_step(cycle, verbose)
            if self.has_converged(conv):
                self.converged = True
                break
            trial = self.take_step(step)
            if self.is_rejected(trial):
                if self.reject_step(trial):
                    break
                continue
            self.accept_step(step, trial)
        if self.status == 'ok' and not self.converged:
            # Exhausting the cycle budget is not a clean finish: the last
            # geometry is not a minimum.
            self.status = (f'max_cycle ({budget}) reached without '
                           f'convergence')

    def record(self, resume):
        """The walk's record, `final_surface` on a walk that is not refrozen."""
        residual = float(np.abs(self.g).max())
        info = {'converged': self.converged, 'cycles': len(self.history),
                'history': self.history, 'energy': self.e,
                'omega': self.diags.get('omega'),
                'grad_max': residual, 'opt_grad_max': residual,
                'rejected': self.rejected, 'status': self.status,
                'optimizer': 'internal', 'hessian': self.hess.copy(),
                'walk_resolution': (walk_resolution(
                    self.g, self.hess, self.m_cur.atom_coords())
                    if self.converged else None)}
        if resume is not None:
            info['resume'] = {'pass': 1 if self.refrozen else 0,
                              'evaluations_reused': len(resume['history'])}
        if not self.refrozen:
            # the surface walked, which a resumed walk rebuilt from its record
            info['final_surface'] = self.surface
        return info


def finish_walk(info, refreeze):
    """`info` marked with why its refreeze drift is null (never measured)."""
    # a refreeze asked for on a run that never converged had no minimum to
    # refreeze at, which is not the same as one that ran and found nothing
    info.update(refreeze_drift=None, refreeze_shift=None,
                refreeze=('not measured' if not refreeze else
                          'not measured: the first pass did not converge'))
    return info


def _trust_region_walk(surface, mol, max_cycle, trust, trust_max, trust_min,
                       hess_init, conv, refreeze, verbose, refrozen=False,
                       checkpoint=None, resume=None):
    """The method `optimize` documents; a `refrozen` walk raises when its start
    fails to evaluate, and every walk raises what is not a `StepFailed`."""
    conv = resolved_conv(conv)
    # a refrozen walk is a new surface and starts at the constructor's radius
    walk_again = refrozen_walker(
        _trust_region_walk, max_cycle=max_cycle, trust=trust,
        trust_max=trust_max, trust_min=trust_min, conv=conv, refreeze=0,
        verbose=verbose)
    if resume is not None and not refrozen:
        surface = restored_surface(surface, resume['surface'],
                                   resume.get('identity'))
        if resume['pass'] >= 1:
            return resumed_refreeze(walk_again, surface, mol, resume,
                                    verbose, checkpoint)
    walk = CartesianWalk(surface, mol, hess_init, trust, trust_max,
                         trust_min, refrozen, checkpoint)
    failed_start = walk.start_or_resume(resume)
    if failed_start is not None:
        return mol, failed_start
    walk.descend(max_cycle, conv, verbose)
    info = walk.record(resume)
    if refreeze and walk.converged:
        return refreeze_pass(walk_again, surface, walk.m_cur, info, verbose,
                             checkpoint=checkpoint)
    return walk.m_cur, finish_walk(info, refreeze)


# ---------------------------------------------------------------------------
# geomeTRIC: internal coordinates, when it is installed
# ---------------------------------------------------------------------------

def optimize_geometric(surface, mol=None, maxiter=100, converge='GAU',
                       coordsys='tric', refreeze=0, verbose=True,
                       workdir=None, hess_init=None, nudge=None,
                       checkpoint=None, resume=None,
                       trust_min=WALK_TRUST_MIN_BOHR):
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

    refreeze: 0 or 1, exactly as `optimize` (`refreeze_pass`).

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

    A trial step whose evaluation raises `StepFailed` is REJECTED and retried
    shorter, as in `optimize`: geomeTRIC is restarted at the last accepted
    geometry, its first point that geometry's stored energy and force (no
    evaluation), on `hess_init` and with a trust radius of WALK_STEP_SHRINK
    times the failed step (the norm of its superposed 3N-vector, in Bohr;
    geomeTRIC's own radius is the per-atom RMS in Angstrom, converted). A
    radius below `trust_min` ends the walk with status 'trust radius
    collapsed'. `info['rejected']` counts the rejected steps, each history
    entry the count when it was evaluated, and `info['retries']` holds each
    rejection's failed step, new radius and message. geomeTRIC's own rejection
    of a poor step is its business and is not counted. The iteration budget
    `maxiter` is shared by the restarts: each gets what the accepted steps
    before it left.

    A walk that does not converge is a record, not an exception, as in
    `optimize`: geomeTRIC's iteration cap (`GeomOptNotConvergedError`) comes
    back as `converged` False with status 'maxiter (N) reached without
    convergence', a start that fails to evaluate as `converged` False with
    its message. The geometry and energy are then the last ones evaluated,
    which is not a minimum, and no refreeze follows. Every other exception,
    of an evaluation or of the driver, leaves the walk.
    Each walk -- the first and every refrozen pass -- writes its files under
    its own prefix in `workdir` (a fresh temporary directory when None).

    checkpoint, resume: as in `optimize`, with one difference. geomeTRIC
    keeps its internal-coordinate Hessian and step history to itself, so a
    resumed walk is a NEW geomeTRIC run from the last accepted geometry on
    the restored surface, its first point the stored energy and force (no
    evaluation), started on `hess_init` and on the trust radius and rejected
    count the checkpoint holds: it reaches the same minimum to the
    convergence criteria, not along the same steps.

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
    refreeze = refreeze_asked(refreeze)
    start, nudged = start_nudge(surface, mol, nudge)
    mol_opt, info = with_final_surface(surface, *_geometric_walk(
        surface, start, maxiter, converge, coordsys, refreeze, verbose,
        workdir, hess_init, checkpoint=checkpoint, resume=resume,
        trust_min=trust_min))
    info['start_nudge'] = nudged
    return mol_opt, info


def geometric_engine():
    """geomeTRIC's engine base, molecule, driver and the error its driver
    raises at the iteration cap, or a refusal naming the dependency-free
    alternative."""
    if not HAVE_GEOMETRIC:
        raise ImportError(
            'geomeTRIC is not importable; use `optimize` for the '
            'dependency-free Cartesian optimizer, or install geomeTRIC as in '
            '`optimize_geometric`\'s docstring.')
    return Engine, GeoMolecule, run_optimizer, GeomOptNotConvergedError


class GeometricWalk:
    """The state of one geomeTRIC walk: what its engine evaluated (`trace`,
    `visited`, `last_gradient`), the last accepted point a rejected step is
    retried from, and the trust radius and rejections of the restarts."""

    def __init__(self, surface, mol, resume, refrozen, checkpoint, verbose,
                 trust_min):
        self.surface, self.mol = surface, mol
        self.refrozen, self.checkpoint = refrozen, checkpoint
        self.verbose, self.trust_min = verbose, trust_min
        self.trace = [] if resume is None else list(resume['history'])
        # the molecules `trace` was evaluated at: where a walk that did
        # not converge stopped
        self.visited = []
        # the gradient of the last evaluation, which is the converged
        # geometry's
        self.last_gradient = []
        # a run's first point, held: (energy, gradient) at its start
        self.replay = ([] if resume is None else
                       [(resume['energy'],
                         np.asarray(resume['gradient'], float))])
        # (molecule, energy, gradient) at the last accepted geometry, where a
        # rejected step is retried from
        self.accepted = []
        # the trial geometry whose evaluation raised `StepFailed`
        self.failed_at = []
        # Bohr, the radius the current geomeTRIC run starts on (None: its own)
        self.radius = None if resume is None else resume.get('trust')
        self.rejected = 0 if resume is None else resume.get('rejected', 0)
        self.retries = []
        self.steps = []

    def save(self):
        """The checkpoint of the walk as it stands."""
        if self.checkpoint is not None:
            self.checkpoint(walk_state(
                self.surface, 1 if self.refrozen else 0, 'geometric',
                self.visited[-1].atom_coords(), self.trace[-1]['e'],
                self.last_gradient[0], {'omega': self.trace[-1]['omega']},
                None, self.trace, trust=self.radius,
                rejected=self.rejected))

    def engine(self, Engine, GeoMolecule):
        """The geomeTRIC engine that evaluates this walk's surface."""
        walk = self

        class _Surface(Engine):
            def __init__(self, here):
                gm = GeoMolecule()
                gm.elem = [here.atom_pure_symbol(i) for i in range(here.natm)]
                gm.xyzs = [np.asarray(here.atom_coords()) * BOHR_TO_ANGSTROM]
                gm.build_topology()
                super().__init__(gm)
                self.start = np.asarray(here.atom_coords(), float).ravel()

            def save_guess_files(self, dirname):
                # geomeTRIC's hook on an accepted step, whose geometry is the
                # last one evaluated
                walk.accept_last()

            def calc_new(self, coords, dirname):
                coords = np.asarray(coords, float).ravel()
                if np.abs(coords - self.start).max() < GEOMETRIC_START_TOL:
                    coords = self.start
                return walk.evaluate_at(coords, coords is self.start)

        return _Surface

    def accept_last(self):
        """geomeTRIC accepted its last evaluated geometry: commit and save."""
        commit(self.surface, self.visited[-1])
        self.accepted[:] = [self.visited[-1], self.trace[-1]['e'],
                            self.last_gradient[0]]
        self.steps.append(len(self.trace))
        self.save()

    def evaluate_at(self, coords, at_start):
        """geomeTRIC's energy and gradient at `coords`: the held start values
        on a restart, else an evaluation that is traced."""
        surface, verbose, trace = self.surface, self.verbose, self.trace
        m = molecule_at(self.mol, coords)
        if self.replay and at_start:
            e, g = self.replay.pop()
            self.visited.append(m)
            self.last_gradient[:] = [g]
            self.accepted[:] = [m, e, g]
            return {'energy': float(e), 'gradient': g}
        try:
            g, e, d = walk_evaluate(surface, m)
        except StepFailed:
            self.failed_at[:] = [coords]
            raise
        omega = d.get('omega')
        self.visited.append(m)
        self.last_gradient[:] = [np.asarray(g, float).ravel()]
        first = not self.visited[:-1]
        if first:
            commit(surface, m)
            self.accepted[:] = [m, e, self.last_gradient[0]]
        trace.append({'e': float(e),
                      'omega': None if omega is None else float(omega),
                      'grad_max': float(np.abs(g).max()),
                      'rejected': self.rejected})
        if verbose:
            om_txt = ('' if omega is None
                      else f'  Om = {omega * HARTREE_TO_EV:8.4f} eV')
            print(f'  [geomeTRIC {len(trace):3d}] E = {e:.10f}{om_txt}  '
                  f'|g|max {np.abs(g).max():.2e}', flush=True)
        if first:
            self.save()
        return {'energy': float(e), 'gradient': np.asarray(g).ravel()}

    def reject_step(self, status):
        """Rejection of the trial step that raised `StepFailed`: geomeTRIC is
        to be restarted shorter at the last accepted geometry. Returns None
        then, else the status of a collapsed radius."""
        # a trial step too long for the frozen branch: rejected, and retried
        # shorter from the last accepted geometry
        base = self.accepted[0].atom_coords()
        trial = self.failed_at[0].reshape(-1, 3)
        q = kabsch_rotation(base, trial)
        step = float(np.linalg.norm(centred(trial) @ q.T - centred(base)))
        self.radius = WALK_STEP_SHRINK * step
        radius = self.radius
        self.rejected += 1
        self.retries.append({'cycle': len(self.trace), 'step_bohr': step,
                             'trust_bohr': radius, 'status': status})
        if self.verbose:
            print(f'  [geomeTRIC] step of {step:.3e} Bohr rejected '
                  f'({status}); restarting at the last accepted '
                  f'geometry, trust {radius:.3e} Bohr', flush=True)
        if radius >= self.trust_min:
            self.replay[:] = [(self.accepted[1], self.accepted[2])]
            return None
        return (f'trust radius collapsed at {radius:.1e} Bohr; '
                f'the state stopped evaluating ({status})')

    def run(self, tag, tmp, maxiter, converge, coordsys, hess_init):
        """Run geomeTRIC, restarting after each rejected step. Returns
        (converged, status)."""
        Engine, GeoMolecule, run_optimizer, NotConverged = geometric_engine()
        engine = self.engine(Engine, GeoMolecule)
        # the approximate Cartesian Hessian geomeTRIC ends with, written here
        self.hess_out = os.path.join(tmp, f'{tag}_final_hessian.txt')
        # geomeTRIC tests `hess_data` for truth, which an array refuses; a
        # nested list in Ha/Bohr^2 is its documented form, and the frequency
        # analysis it would run on a given Hessian is not wanted
        given = ({} if hess_init is None else
                 {'hess_data': np.asarray(hess_init, float).tolist(),
                  'frequency': False})
        while True:
            here = self.accepted[0] if self.accepted else self.mol
            self.failed_at.clear()
            # geomeTRIC's trust radius is the per-atom RMS step in Angstrom
            trust = ({} if self.radius is None else
                     {'trust': (self.radius / np.sqrt(self.mol.natm)
                                * BOHR_TO_ANGSTROM)})
            rejected = self.rejected
            prefix = tag if not rejected else f'{tag}_retry{rejected}'
            budget = maxiter - len(self.steps)
            if budget < 1:
                return False, f'maxiter ({maxiter}) reached without convergence'
            try:
                run_optimizer(customengine=engine(here), coordsys=coordsys,
                              maxiter=budget, convergence_set=converge,
                              input=tag, prefix=os.path.join(tmp, prefix),
                              check=0, write_cart_hess=self.hess_out,
                              **given, **trust)
            except NotConverged:
                # geomeTRIC raises at its iteration cap rather than returning;
                # the walk up to there is still the output, as `optimize`'s is
                return False, f'maxiter ({maxiter}) reached without convergence'
            except StepFailed as exc:
                # a refrozen surface that does not evaluate at the minimum of
                # the surface it was rebuilt from is broken, not a walk that
                # went nowhere
                if self.refrozen and not self.trace:
                    raise
                status = f'{type(exc).__name__}: {exc}'
                if self.failed_at and self.accepted:
                    collapsed = self.reject_step(status)
                    if collapsed is None:
                        continue
                    status = collapsed
                return False, status
            return True, 'ok'

    def unevaluated_record(self, status, coordsys, refreeze):
        """The record of a walk that evaluated nothing."""
        return {'converged': False, 'cycles': 0, 'history': [],
                'status': status, 'engine': 'geometric',
                'optimizer': 'geometric', 'coordsys': coordsys,
                'hessian': None, 'rejected': self.rejected,
                'retries': self.retries, 'refreeze_drift': None,
                'refreeze_shift': None,
                'refreeze': ('not measured' if not refreeze else
                             'not measured: the first pass did not '
                             'converge')}

    def record(self, converged, status, coordsys, resume):
        """The walk's record; its geometry is the last one evaluated."""
        trace, visited = self.trace, self.visited
        residual = trace[-1]['grad_max']
        hessian = (np.loadtxt(self.hess_out)
                   if converged and os.path.exists(self.hess_out) else None)
        info = {'converged': converged, 'cycles': len(trace),
                'history': trace, 'energy': trace[-1]['e'],
                'omega': trace[-1]['omega'],
                'grad_max': residual, 'opt_grad_max': residual,
                'status': status, 'engine': 'geometric',
                'optimizer': 'geometric', 'coordsys': coordsys,
                'hessian': hessian, 'rejected': self.rejected,
                'retries': self.retries,
                'walk_resolution': (walk_resolution(
                    self.last_gradient[0], hessian, visited[-1].atom_coords())
                    if converged and hessian is not None else None)}
        if resume is not None:
            info['resume'] = {'pass': 1 if self.refrozen else 0,
                              'evaluations_reused': len(resume['history']),
                              'geometric_restarted': True}
        if not self.refrozen:
            info['final_surface'] = self.surface
        return info


def _geometric_walk(surface, mol, maxiter, converge, coordsys, refreeze,
                    verbose, workdir, hess_init=None, tag='es', refrozen=False,
                    checkpoint=None, resume=None,
                    trust_min=WALK_TRUST_MIN_BOHR):
    """The engine `optimize_geometric` documents; `tag` prefixes this walk's
    files in `workdir`, and a `refrozen` walk raises when its start fails to
    evaluate."""
    geometric_engine()
    # geomeTRIC's files go under an absolute prefix rather than a chdir: the
    # working directory belongs to the process, which the rank threads of
    # `run_simulated` share.
    tmp = workdir or tempfile.mkdtemp(prefix='esopt_')
    walk_again = refrozen_walker(
        _geometric_walk, maxiter=maxiter, converge=converge,
        coordsys=coordsys, refreeze=0, verbose=verbose, workdir=tmp,
        tag=f'{tag}_refreeze', trust_min=trust_min)
    if resume is not None and not refrozen:
        surface = restored_surface(surface, resume['surface'],
                                   resume.get('identity'))
        if resume['pass'] >= 1:
            return resumed_refreeze(walk_again, surface, mol, resume,
                                    verbose, checkpoint)
    if resume is not None:
        mol = molecule_at(mol, resume['coords'])
    walk = GeometricWalk(surface, mol, resume, refrozen, checkpoint, verbose,
                         trust_min)
    converged, status = walk.run(tag, tmp, maxiter, converge, coordsys,
                                 hess_init)
    if not walk.trace:
        return mol, walk.unevaluated_record(status, coordsys, refreeze)
    # the converged geometry is the last one evaluated: the molecule itself,
    # not its Angstrom round trip, so every later question about it reads
    # the evaluation held there
    mol_opt = walk.visited[-1]
    if converged:
        # geomeTRIC stops on convergence before its accepted-step hook
        commit(surface, mol_opt)
    info = walk.record(converged, status, coordsys, resume)
    if refreeze and converged:
        return refreeze_pass(walk_again, surface, mol_opt, info, verbose,
                             checkpoint=checkpoint)
    return mol_opt, finish_walk(info, refreeze)


#: What a single walk's record says about that walk alone: after a refreeze
#: these are the last walk's, the one whose geometry is returned.
WALK_FLAGS = ('converged', 'status', 'cycles', 'opt_grad_max', 'grad_max',
              'rejected', 'walk_resolution')


class RefreezeNotConverged(RuntimeWarning):
    """The conventions rebuilt at a minimum still move its energy by more
    than REFREEZE_TOL_MEV after REFREEZE_MAX_PASSES, or a refrozen walk did
    not converge: the geometry and record are kept, flagged (`refreeze_pass`)."""


def refreeze_asked(refreeze):
    """`refreeze` as 0 or 1, refusing a count of passes: the loop's passes
    are decided by the drift, at most REFREEZE_MAX_PASSES."""
    if refreeze not in (0, 1):
        raise ValueError(
            f'refreeze={refreeze!r}: a relaxation runs the refreeze loop (1) '
            f'or does not refreeze (0); the drift decides how many passes, at '
            f'most REFREEZE_MAX_PASSES = {REFREEZE_MAX_PASSES}')
    return int(refreeze)


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


def refreeze_pass(walk, surface, mol, info, verbose, checkpoint=None,
                  resume=None, rebuilt=None, npass=1):
    """(geometry, record) after rebuilding the frozen conventions at the
    minimum `mol` and relaxing on the rebuilt surface
    (`walk(surface, mol, hessian)`, starting at `mol` unnudged, with
    `checkpoint` and `resume` keywords where they are given), pass `npass`
    of the refreeze loop.

    checkpoint: the walk's, handed to the refrozen walk with the first
    walk's record and its minimum added to every state, so a relaunch
    resumes in this pass (`resumed_refreeze`); resume, rebuilt: such a
    state and the surface rebuilt from its record, which then stand in for
    the rebuild.

    THE DRIFT IS MEASURED AT ONE GEOMETRY. `refreeze_drift` = E_new(R*) -
    E_old(R*) in Hartree (`refreeze_drift_meV` in meV): the rebuilt surface's
    first evaluation, at R*, minus the old surface's last, at the same R*. It
    holds the change of the conventions alone; the walk that follows is
    descent on one surface and is not in it. |drift| <= REFREEZE_TOL_MEV sets
    `refreeze_converged`, ends the loop, and the drift is the minimum's error
    bar. Above it, a converged walk is refrozen again at its own minimum, the
    next pass; after REFREEZE_MAX_PASSES, or when the walk did not converge,
    the last pass's geometry and this record are returned with a warning
    (`RefreezeNotConverged`), `refreeze` = 'not converged' and the flag
    'refreeze_not_converged' in `walk_flags` ({'flag', 'message'} each;
    refuse inputs, warn numerics). `refreeze_passes` lists
    every pass: its drift, shift and whether its walk converged.

    `refreeze_shift` is how far the walk on the rebuilt surface moved the
    geometry, the largest atomic move once its minimum is superposed on R*
    (`body_frame.aligned_displacement`: the surface is invariant to rigid
    motion, which an internal-coordinate walk is free to make);
    `refreeze_rigid_shift` keeps the raw Cartesian move; both are the last
    pass's. `refreeze_denergy`
    is E_new(R*_new) - E_old(R*), drift and descent together, kept as a
    record and deciding nothing. `refreeze_qp_growth` is what the rebuild
    did to an adaptive explicit set (`refreeze_growth`, grow-only).

    THE RECORD DESCRIBES THE WALK WHOSE GEOMETRY IT RETURNS. `converged`,
    `status`, `cycles`, `opt_grad_max` (and `grad_max`, `rejected`,
    `walk_resolution`) are the refrozen walk's, so `converged` False means
    the geometry returned is not a minimum of the surface it stands on; the
    first walk's are kept as `first_walk_converged`, ... `history` stays the
    first walk's, the one that starts at the input geometry; the refrozen
    walk's is in `refreeze_info`. `energy`, `omega` and `hessian` are the
    rebuilt surface's, which is `final_surface`.
    """
    first_de = resolved_de(info)
    if rebuilt is None:
        rebuilt = surface.refreeze(mol)
    first = {k: v for k, v in info.items() if k != 'final_surface'}
    # a walk that is not checkpointed is called as `walk(surface, mol, h)`
    kw = {} if resume is None else {'resume': resume}
    if checkpoint is not None:
        kw['checkpoint'] = lambda state: checkpoint(dict(
            state, first_walk=first, minimum=np.array(mol.atom_coords()))
            | {'pass': npass})
    moved, walked = walk(rebuilt, mol, info['hessian'], **kw)
    drift = float(walked['history'][0]['e'] - info['energy'])
    drift_mev = drift * HARTREE_TO_MEV
    shift = aligned_displacement(mol.atom_coords(), moved.atom_coords())
    rigid = float(np.abs(moved.atom_coords() - mol.atom_coords()).max())
    growth = refreeze_growth(rebuilt)
    if verbose:
        added = ('' if growth is None else
                 f', explicit states added {growth["added"]}')
        print(f'  [refreeze {npass}] drift at the minimum {drift_mev:+.4f} '
              f'meV; the '
              f'walk on the rebuilt surface moved {shift:.2e} Bohr superposed '
              f'({rigid:.2e} raw){added}', flush=True)
    flags = [key for key in WALK_FLAGS if key in walked]
    for key in flags:
        info.setdefault(f'first_walk_{key}', info.get(key))
    info.update({key: walked[key] for key in flags})
    passes = list(info.get('refreeze_passes') or []) + [{
        'pass': npass, 'drift_meV': drift_mev, 'shift': shift,
        'converged': bool(walked.get('converged'))}]
    info.update(refreeze_drift=drift, refreeze_drift_meV=drift_mev,
                refreeze_converged=bool(abs(drift_mev) <= REFREEZE_TOL_MEV),
                refreeze_shift=shift, refreeze_rigid_shift=rigid,
                refreeze_denergy=float(walked['energy'] - info['energy']),
                refreeze_qp_growth=growth, refreeze_passes=passes,
                refreeze_info=walked, energy=walked['energy'],
                omega=walked['omega'], refreeze='measured',
                hessian=walked['hessian'], final_surface=rebuilt)
    if 'resume' in walked:
        info['resume'] = dict(walked['resume'], evaluations_reused=(
            walked['resume']['evaluations_reused'] + len(first['history'])))
    info.setdefault('first_walk_de_meV', first_de)
    if info['refreeze_converged']:
        return moved, info
    if walked.get('converged') and npass < REFREEZE_MAX_PASSES:
        return refreeze_pass(walk, rebuilt, moved, info, verbose,
                             checkpoint=checkpoint, npass=npass + 1)
    info['refreeze'] = 'not converged'
    why = (f'after {npass} passes (REFREEZE_MAX_PASSES)'
           if walked.get('converged') else
           f'and the walk of pass {npass} did not converge')
    text = (f'the conventions rebuilt at the minimum move its energy by '
            f'{drift_mev:+.3f} meV, above REFREEZE_TOL_MEV = {REFREEZE_TOL_MEV} '
            f'meV, {why}: the geometry of the last pass is kept and flagged')
    warnings.warn(text, RefreezeNotConverged, stacklevel=2)
    info['walk_flags'] = list(info.get('walk_flags') or []) + [
        {'flag': 'refreeze_not_converged', 'message': text}]
    return moved, info


def resumed_refreeze(walk, rebuilt, mol, resume, verbose, checkpoint):
    """`refreeze_pass` taken up from a state of its refrozen walk: the first
    walk's record and minimum from the state, the rebuilt surface from its
    record (`rebuilt`), and the refrozen walk from where it stood. A state
    with no surface record (a surface that writes none) rebuilds the
    conventions at the minimum again."""
    minimum = molecule_at(mol, resume['minimum'])
    info = dict(resume['first_walk'])
    return refreeze_pass(walk, rebuilt, minimum, info, verbose,
                         checkpoint=checkpoint, resume=resume,
                         rebuilt=None if resume['surface'] is None else rebuilt,
                         npass=int(resume['pass']))


def walk_state(surface, stage, engine, coords, energy, gradient, diags,
               hessian, history, **extra):
    """What a walk checkpoint holds after an accepted step: the pass
    (0 the first walk, k the walk of the refreeze loop's k-th pass), the
    engine, the
    geometry (Bohr), energy, projected gradient and excitation energy there,
    the optimizer's Hessian (None where the engine keeps its own), the
    history, the engine's own state (`extra`: the trust radius and the
    rejected count), the surface's record (`surface_record`) and the hash
    of what its forces are forces of (`surface_identity`), which a resume
    checks the restored surface against."""
    return dict(extra) | {
        'pass': stage, 'engine': engine,
        'coords': np.array(coords, float).reshape(-1, 3),
        'energy': energy, 'gradient': np.array(gradient, float),
        'omega': diags.get('omega'),
        'hessian': None if hessian is None else np.array(hessian, float),
        'history': list(history), 'surface': surface_record(surface),
        'identity': surface_identity(surface)}


def surface_identity(surface):
    """sha256 of `force_identity(surface)`, or None for a surface that
    freezes nothing."""
    identity = force_identity(surface)
    return None if identity is None else record_hash(identity)


def surface_record(surface):
    """The record of `surface` a checkpoint keeps (`to_record`): a chain's,
    or a composed surface's with both halves (`ComposedSurface`); None for
    a surface that writes none. Restored, it is the surface the force
    identity (`surface_identity.force_identity`) was taken of."""
    if not provides(surface, 'to_record'):
        return None
    return surface.to_record()


def surface_from_record(record, mol, scf_factory, environment=None, mf=None):
    """The surface `surface_record` wrote, rebuilt by its class's
    `from_record` on `mol`'s basis with the declaration's factory and
    environment."""
    return record_class(record['class']).from_record(
        record, mol, scf_factory, environment=environment, mf=mf)


def restored_surface(surface, record, identity=None):
    """`surface` rebuilt from `record` on its own factory and environment;
    `surface` itself where there is no record or it is that record's.

    identity: the `surface_identity` the checkpoint stamped, if any; a
    surface rebuilt to another one is refused, since the walk would go on
    with forces of a surface it never took a step on."""
    if record is None:
        return surface
    own = surface_record(surface)
    if own is not None and record_hash(own) == record_hash(record):
        return surface
    out = surface_from_record(
        record, surface.mol0, surface.scf_factory,
        environment=getattr(driven_chain(surface), 'environment', None))
    if identity is not None and surface_identity(out) != identity:
        raise RuntimeError(
            'the surface rebuilt from the checkpoint is not the one the walk '
            'stood on (its force identity differs): resume with the '
            'declaration the checkpoint was written under')
    return out


def relax(surface, mol=None, engine='auto', **kw):
    """Relax a state, preferring internal coordinates when available.

    engine: 'geometric', 'cartesian', or 'auto' (geomeTRIC if importable).
    Either starts an excited-state walk nudged off `mol` unless `nudge=0`
    (`start_nudge`).

    `refreeze` reaches either optimizer: both measure the drift of the frozen
    conventions the same way, so which engine ran does not change what the
    number means. Which one did run is in `info['optimizer']`, written by the
    engine itself and passed through untouched -- the record has to say which
    method produced the geometry, because the two converge to different
    residuals on the same minimum.

    With 'auto' the caller does not know which engine runs, so each takes
    the keywords it has and drops the other's (`engine_keywords`); a keyword
    neither has is refused. A named engine takes `kw` as it is.
    """
    if engine == 'cartesian':
        mol_opt, info = optimize(surface, mol, **kw)
    elif engine == 'geometric':
        mol_opt, info = optimize_geometric(surface, mol, **kw)
    elif HAVE_GEOMETRIC:
        mol_opt, info = optimize_geometric(
            surface, mol, **engine_keywords(optimize_geometric, kw))
    else:
        mol_opt, info = optimize(surface, mol, **engine_keywords(optimize, kw))
    info.setdefault('optimizer', engine)
    return mol_opt, info


def engine_keywords(walker, kw):
    """The keywords of `kw` that `walker` takes; one that neither optimizer
    takes is refused rather than dropped."""
    taken = {name: set(inspect.signature(f).parameters)
             for name, f in (('optimize', optimize),
                             ('optimize_geometric', optimize_geometric))}
    unknown = set(kw) - set.union(*taken.values())
    if unknown:
        raise TypeError(f'relax: no optimizer takes {sorted(unknown)}')
    return {k: v for k, v in kw.items() if k in taken[walker.__name__]}


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

    A relaxation that stops before converging returns where it stopped, with
    a warning (`UnconvergedRelaxation`); `mf.relaxation` is its record flag
    (`relaxation_flag`) on every route, converged or not.
    """
    if environment is not None:
        surface = MeanFieldSurface(mol, scf_factory, environment=environment)
        opt, info, name = None, None, 'cartesian'
        if engine != 'cartesian':
            try:
                opt, info = optimize_geometric(
                    surface, mol, maxiter=maxsteps,
                    converge='GAU' if converge is None else converge,
                    verbose=verbose)
                name = 'geometric'
            except ImportError as exc:
                if verbose:
                    print(f'{exc}; using the Cartesian trust-region optimizer')
        if info is None:
            opt, info = optimize(surface, mol, max_cycle=maxsteps, trust=0.3,
                                 verbose=verbose)
        final = info.get('final_surface', surface).mean_field(opt)[1]
        final.relaxation = relaxation_flag(info, name)
        return opt, final
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
        # `kernel`, not `optimize`: `optimize` drops the converged flag
        converged, opt = solver.kernel(mf, maxsteps=maxsteps, **kw)
        final = scf_factory(opt)
        final.relaxation = relaxation_flag({
            'converged': bool(converged), 'cycles': None,
            'status': 'converged' if converged else f'stopped at maxsteps '
                                                    f'{maxsteps}',
            'opt_grad_max': ground_state_residual_force(final, opt)}, name)
        return opt, final
    # Neither external optimizer is installed: fall back to the Cartesian
    # trust-region method, which needs nothing outside src/.
    if verbose:
        print(f'no external optimizer ({"; ".join(tried)}); using the '
              f'Cartesian trust-region optimizer')
    opt, info = optimize(MeanFieldSurface(mol, scf_factory), mol,
                         max_cycle=maxsteps, trust=0.3, verbose=verbose)
    final = scf_factory(opt)
    final.relaxation = relaxation_flag(info, 'cartesian')
    return opt, final


class UnconvergedRelaxation(RuntimeWarning):
    """A ground-state relaxation that stopped before converging: every
    vibronic quantity built on its geometry is about a non-stationary point
    and carries the record flag (`relaxation_flag`)."""


def relaxation_flag(info, engine):
    """The record flag of a ground-state relaxation: relaxation_converged,
    opt_grad_max (max |dE/dR| where it stopped, Ha/Bohr), cycles, status and
    engine; warns (`UnconvergedRelaxation`) where it did not converge."""
    flag = {'relaxation_converged': bool(info.get('converged')),
            'opt_grad_max': info.get('opt_grad_max'),
            'cycles': info.get('cycles'), 'status': info.get('status'),
            'engine': engine}
    if not flag['relaxation_converged']:
        residual = flag['opt_grad_max']
        warnings.warn(
            f'the ground-state relaxation ({engine}) stopped without '
            f'converging after {flag["cycles"]} cycles ({flag["status"]}; '
            f'max |dE/dR| = '
            f'{float("nan") if residual is None else residual:.2e} Ha/Bohr); '
            f'every vibronic quantity built on this geometry is about a '
            f'non-stationary point (flag relaxation_converged=False)',
            UnconvergedRelaxation, stacklevel=3)
    return flag


def ground_state_residual_force(mf, mol=None):
    """max |dE_0/dR| -- how far the reference is from its own minimum.

    A number worth printing next to any Huang-Rhys spectrum: it is the size of
    the term the harmonic expansion is pretending is zero. The force is
    `mean_field_force`'s, grid response included.
    """
    return float(np.abs(mean_field_force(mf)).max())
