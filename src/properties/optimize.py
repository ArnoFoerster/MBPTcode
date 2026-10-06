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
import os
import tempfile
import warnings

import numpy as np
from pyscf import grad  # noqa: F401  registers mf.Gradients/nuc_grad_method

from src.Base.constants import (BOHR_TO_ANGSTROM, GEOM_OPT_CONV,
                                GEOMETRIC_START_TOL, HARTREE_TO_EV)
from src.Base.declaration import SurfacePhysics
from src.Base.environment import environment_label, resolve_environment
from src.Base.isdf_jk import mean_field_skeleton_force
from src.Base.utils.mpi_grid import lockstep, lockstep_mean_field
from src.SingleReference.LinearResponse.rpa_energy import declared_ground_state
from src.properties.surface import evaluate, lockstep_geometry


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


def optimize(surface, mol=None, max_cycle=50, trust=0.1, trust_max=0.5,
             trust_min=2e-3, hess_init=None, conv=None, refreeze=0,
             verbose=True):
    """Relax `surface`'s state. Returns (mol, info).

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
        times. 0 reports the single-surface minimum with the drift unmeasured
        and marks the record `refreeze: 'not measured'`, so a null
        `refreeze_shift` can only have come from an explicit 0; 1 measures it:
        `info['refreeze_shift']` is how far the geometry moved, the error bar
        on the minimum.

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
    return rank_zero_walk(*_trust_region_walk(
        surface, mol, max_cycle, trust, trust_max, trust_min, hess_init, conv,
        refreeze, verbose))


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
            'hessian': hess.copy()}

    if refreeze and converged:
        mol2, info2 = _trust_region_walk(surface.refreeze(m_cur), m_cur,
                                         max_cycle, trust, trust_max,
                                         trust_min, hess, conv, refreeze - 1,
                                         verbose)
        shift = float(np.abs(mol2.atom_coords() - m_cur.atom_coords()).max())
        info.update(refreeze_shift=shift,
                    refreeze_denergy=float(info2['energy'] - info['energy']),
                    refreeze_info=info2, energy=info2['energy'],
                    omega=info2['omega'], refreeze='measured',
                    hessian=info2['hessian'])
        if verbose:
            print(f'  [refreeze] geometry moved {shift:.2e} Bohr, energy by '
                  f'{info2["energy"] - e:+.2e} Ha')
        return mol2, info
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
                       workdir=None, hess_init=None):
    """Relax `surface`'s state through geomeTRIC. Returns (mol, info).

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
        times, as `optimize` does. 0 marks the record
        `refreeze: 'not measured'`, so a null `refreeze_shift` can only have
        come from an explicit 0.

    `info['opt_grad_max']` is max |dE/dR| at the converged geometry R*, the
    residual, and not the driving force at R0 that `grad_max` means elsewhere;
    `info['grad_max']` is the same float, kept for callers that read it.

    hess_init: a starting Cartesian Hessian in Ha/Bohr^2 (geomeTRIC's
        `hess_data`, transformed to its internal coordinates), or None for
        geomeTRIC's own guess; another walk's `info['hessian']` is one.
        `info['hessian']` is the approximate Cartesian Hessian geomeTRIC
        ends with (`write_cart_hess`), the refrozen walk's after a refreeze;
        a refrozen walk starts from `hess_init` as the first did.

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
    return rank_zero_walk(*_geometric_walk(surface, mol, maxiter, converge,
                                           coordsys, refreeze, verbose,
                                           workdir, hess_init))


def geometric_engine():
    """geomeTRIC's engine base, molecule and driver, or a refusal naming the
    dependency-free alternative.

    An optional dependency, absent from the environment by default: importing
    it at module level would make every property routine need it.
    """
    try:
        # optional dependency, absent from the environment by default
        from geometric.engine import Engine
        from geometric.molecule import Molecule as GeoMolecule
        from geometric.optimize import run_optimizer
    except ImportError as exc:
        raise ImportError(
            'geomeTRIC is not importable; use `optimize` for the '
            'dependency-free Cartesian optimizer, or install geomeTRIC as in '
            '`optimize_geometric`\'s docstring.') from exc
    return Engine, GeoMolecule, run_optimizer


def _geometric_walk(surface, mol, maxiter, converge, coordsys, refreeze,
                    verbose, workdir, hess_init=None):
    """The engine `optimize_geometric` documents."""
    Engine, GeoMolecule, run_optimizer = geometric_engine()

    gm = GeoMolecule()
    gm.elem = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    gm.xyzs = [np.asarray(mol.atom_coords()) * BOHR_TO_ANGSTROM]
    gm.build_topology()

    trace = []

    start = np.asarray(mol.atom_coords(), float).ravel()

    class _Surface(Engine):
        def calc_new(self, coords, dirname):
            coords = np.asarray(coords, float).ravel()
            if np.abs(coords - start).max() < GEOMETRIC_START_TOL:
                coords = start
            m = at_geometry(mol, coords)
            g, e, d = evaluate(surface, m)
            omega = d.get('omega')
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
    hess_out = os.path.join(tmp, 'es_final_hessian.txt')
    # geomeTRIC tests `hess_data` for truth, which an array refuses; a nested
    # list in Ha/Bohr^2 is its documented form, and the frequency analysis it
    # would run on a given Hessian is not wanted
    given = ({} if hess_init is None else
             {'hess_data': np.asarray(hess_init, float).tolist(),
              'frequency': False})
    out = run_optimizer(customengine=engine, coordsys=coordsys,
                        maxiter=maxiter, convergence_set=converge,
                        input='es', prefix=os.path.join(tmp, 'es'), check=0,
                        write_cart_hess=hess_out, **given)

    xyz = np.asarray(out.xyzs[-1]) / BOHR_TO_ANGSTROM
    mol_opt = mol.copy()
    mol_opt.set_geom_(xyz, unit='Bohr')
    mol_opt.build(False, False)
    residual = trace[-1]['grad_max']
    info = {'converged': True, 'cycles': len(trace), 'history': trace,
            'energy': trace[-1]['e'], 'omega': trace[-1]['omega'],
            'grad_max': residual, 'opt_grad_max': residual, 'status': 'ok',
            'engine': 'geometric', 'optimizer': 'geometric',
            'coordsys': coordsys,
            'hessian': (np.loadtxt(hess_out) if os.path.exists(hess_out)
                        else None)}

    if refreeze:
        mol2, info2 = _geometric_walk(surface.refreeze(mol_opt), mol_opt,
                                      maxiter, converge, coordsys,
                                      refreeze - 1, verbose, workdir,
                                      hess_init)
        shift = float(np.abs(mol2.atom_coords() - mol_opt.atom_coords()).max())
        info.update(refreeze_shift=shift,
                    refreeze_denergy=float(info2['energy'] - info['energy']),
                    refreeze_info=info2, energy=info2['energy'],
                    omega=info2['omega'], refreeze='measured',
                    hessian=info2['hessian'])
        if verbose:
            print(f'  [refreeze] geometry moved {shift:.2e} Bohr, energy by '
                  f'{info["refreeze_denergy"]:+.2e} Ha')
        return mol2, info
    info.update(refreeze_shift=None, refreeze='not measured')
    return mol_opt, info


def relax(surface, mol=None, engine='auto', **kw):
    """Relax a state, preferring internal coordinates when available.

    engine: 'geometric', 'cartesian', or 'auto' (geomeTRIC if importable).

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

    def _mean_field(self, mol):
        """The factory's mean field at `mol` in this surface's environment,
        converged over the ranks if it was not already, and rank 0's on every
        rank."""
        # cycle: src.gradients -> src.properties.__init__ -> this module
        from src.gradients.factor_chain import converged_factory

        return lockstep_mean_field(
            self.environment.for_geometry(mol).mean_field(
                mol, converged_factory(self._scf)))

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
                       verbose=False, converge=None):
    """Relax the mean-field ground state. Returns (mol, mf).

    The vibronic quantities are defined about this geometry, not the input
    one: Huang-Rhys factors expand both surfaces in the ground state's normal
    modes, which are modes only at a stationary point, and a reorganization
    energy from a non-stationary reference includes the ground state's own
    relaxation (on formaldehyde/cc-pVDZ the unrelaxed input moves the total
    Huang-Rhys factor from 1.278 to 0.896).
    """
    mf = scf_factory(mol)
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
    if not info.get('converged'):
        warnings.warn(
            f'the ground-state relaxation stopped after {info.get("cycles")} '
            f'cycles without converging (max |dE/dR| = '
            f'{info.get("grad_max", float("nan")):.2e}); every vibronic '
            f'quantity is defined about a STATIONARY point, so treat what '
            f'follows as provisional.', RuntimeWarning, stacklevel=2)
    return opt, scf_factory(opt)


def ground_state_residual_force(mf, mol=None):
    """max |dE_0/dR| -- how far the reference is from its own minimum.

    Worth printing next to any Huang-Rhys spectrum: it is the size of the term
    the harmonic expansion takes as zero.
    """
    return float(np.abs(mf.Gradients().kernel()).max())
