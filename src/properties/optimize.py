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

So both optimizers freeze once and keep it, which makes the whole optimization
one smooth surface, and then offer an OUTER loop: `surface.refreeze` at the
converged geometry and optimize again, until refreezing stops moving it. That
converts an uncontrolled approximation into a measured one -- each reports how
far the refreeze moved the geometry, and that number is the honest error bar on
the minimum.

ONE WALK, HOWEVER MANY RANKS. Inside `with distributed(comm):` every rank runs
the walk below, and every evaluation goes through `surface.evaluate`, which
locks the geometry to rank 0's on the way in and the energy, force and
diagnostics on the way out. Every decision -- the step, the convergence test,
the trust radius, whether a step was rejected, where the refreeze happens --
is therefore taken on every rank from the same numbers, and the walk's result
is locked to rank 0's once more at the end (`rank_zero_walk`), so every rank
returns one geometry and one record. Outside a distributed region every
lockstep is a no-op and the walk is bit for bit the serial one.
"""
import os
import tempfile
import warnings

import numpy as np
from pyscf import grad  # noqa: F401  registers mf.Gradients/nuc_grad_method

from src.Base.constants import BOHR_TO_ANGSTROM, GEOM_OPT_CONV, HARTREE_TO_EV
from src.Base.declaration import SurfacePhysics
from src.Base.utils.mpi_grid import lockstep, lockstep_mean_field
from src.SingleReference.LinearResponse.rpa_energy import declared_ground_state
from src.properties.surface import evaluate, lockstep_geometry


def translation_rotation_basis(coords, masses=None):
    """Orthonormal basis of the 6 (5 if linear) rigid-body directions.

    An optimizer that does not project these out wanders along them: the
    gradient is exactly zero there, so BFGS sees a null direction, the inverse
    Hessian blows up along it and the molecule drifts and spins while the
    energy does nothing. Mass weighting is optional here because the projector
    only has to span the same subspace.
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

    The force threshold is `opt_grad_max`: the residual at the converged
    geometry, which is what the convergence test compares against. It was
    spelled `grad_max` before that word was split from the driving force at a
    fixed geometry, so a caller's dict written under the old name still sets
    the same threshold, for one release. The two given TOGETHER and DIFFERENT
    is a contradiction, not a precedence question -- one of them is not the
    threshold the caller believes is in force -- so it raises rather than
    picking. The resolved dict carries both spellings of the one number, so a
    reader of either sees what actually ran.
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

    Every rank took its steps from the same evaluated numbers, so the walks
    agree as far as the optimizer's own arithmetic is bitwise across nodes --
    the `eigh` of the augmented RFO matrix, the BFGS update -- and that is not
    guaranteed there. Locking the result makes the returned geometry and
    record one by construction, whatever that arithmetic did.
    """
    return lockstep_geometry(mol_opt), lockstep(info)


def optimize(surface, mol=None, max_cycle=50, trust=0.1, trust_max=0.5,
             trust_min=2e-3, hess_init=None, conv=None, refreeze=0,
             verbose=True):
    """Relax `surface`'s state. Returns (mol, info).

    A trust-region quasi-Newton method: RFO step, BFGS Hessian, translations
    and rotations projected out, and steps that turn out badly -- or that fail
    to evaluate at all -- are REJECTED and retried smaller rather than merely
    followed by a smaller radius.

    `trust` starts at 0.1 Bohr rather than the 0.3 a ground-state optimizer
    would use, because the first step is taken on a guessed Hessian and an
    excited surface is more fragile than a ground-state one: at formaldehyde a
    0.23 Bohr first step carried a quasiparticle root past where the frozen
    Newton seed could find it. Without
    rejection an excited-state surface walks itself into nonsense -- measured on
    water, whose S1 is dissociative, the energy oscillated over 40 mHa for
    twenty cycles at the maximum radius and then the Casida problem lost
    positive-definiteness altogether.

    hess_init: an initial Cartesian Hessian, or None for a scaled identity.
        Passing the ground-state analytic Hessian (`normal_modes` builds one
        anyway for the Huang-Rhys factors) typically halves the cycle count:
        the excited surface has similar curvature to the ground one, which is
        what a quasi-Newton method needs and cannot guess.
    refreeze: after convergence, rebuild the frozen conventions at the new
        geometry (`surface.refreeze`) and optimize again, up to this many
        times. 0 reports the single-surface minimum with the drift unmeasured
        and marks the record `refreeze: 'not measured'`, so a null
        `refreeze_shift` can only have come from an explicit 0; 1 MEASURES it
        -- `info['refreeze_shift']` is how far the geometry moved and is the
        honest error bar on the minimum.

    `info['opt_grad_max']` is max |dE/dR| at the CONVERGED geometry R*, the
    residual the convergence test was applied to -- not the driving force at
    the input geometry R0, which is what `grad_max` means wherever a gradient
    is reported at a fixed geometry. `info['grad_max']` is the same float, kept
    for callers that already read it.

    conv: thresholds overriding GEOM_OPT_CONV. Its force entry is
        `opt_grad_max` for the same reason; the superseded spelling `grad_max`
        is accepted for one release and sets the same threshold
        (`resolved_conv`).

    A state that stops existing along the path -- dissociative, or a reference
    that goes singlet/triplet unstable -- is reported as `info['status']`, not
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
            # A STEP THAT DOES NOT EVALUATE IS A STEP THAT WAS TOO LONG, not a
            # dead run. The quasiparticle Newton is seeded from the reference
            # branch and a large displacement can carry a root past where that
            # seed still finds it; the Casida problem can lose
            # positive-definiteness the same way. Both recover by halving the
            # radius, and only a radius that collapses is a real failure.
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
        # Exhausting the cycle budget is a DISTINCT outcome from finishing
        # cleanly, and calling it 'ok' invites a caller to read the last
        # geometry as a minimum. It is not one.
        status = f'max_cycle ({max_cycle}) reached without convergence'
    residual = float(np.abs(g).max())
    info = {'converged': converged, 'cycles': len(history), 'history': history,
            'energy': e, 'omega': diags.get('omega'),
            'grad_max': residual, 'opt_grad_max': residual,
            'rejected': rejected, 'status': status, 'optimizer': 'internal'}

    if refreeze and converged:
        mol2, info2 = _trust_region_walk(surface.refreeze(m_cur), m_cur,
                                         max_cycle, trust, trust_max,
                                         trust_min, hess, conv, refreeze - 1,
                                         verbose)
        shift = float(np.abs(mol2.atom_coords() - m_cur.atom_coords()).max())
        info.update(refreeze_shift=shift,
                    refreeze_denergy=float(info2['energy'] - info['energy']),
                    refreeze_info=info2, energy=info2['energy'],
                    omega=info2['omega'], refreeze='measured')
        if verbose:
            print(f'  [refreeze] geometry moved {shift:.2e} Bohr, energy by '
                  f'{info2["energy"] - e:+.2e} Ha')
        return mol2, info
    # A record whose shift is null says WHY it is null: the drift was never
    # measured. Without the marker a caller cannot tell that from a refreeze
    # that ran and found nothing, and a refreeze asked for on a run that never
    # converged is a third case again -- there was no minimum to refreeze at.
    info.update(refreeze_shift=None,
                refreeze=('not measured' if not refreeze else
                          'not measured: the first pass did not converge'))
    return m_cur, info


# ---------------------------------------------------------------------------
# geomeTRIC: internal coordinates, when it is installed
# ---------------------------------------------------------------------------

def optimize_geometric(surface, mol=None, maxiter=100, converge='GAU',
                       coordsys='tric', refreeze=0, verbose=True,
                       workdir=None):
    """Relax `surface`'s state through geomeTRIC. Returns (mol, info).

    PREFER THIS OVER `optimize` AT SIZE. The Cartesian optimizer above carries
    no dependency and is fine for a handful of atoms, but the number of
    quasi-Newton steps it needs grows with the system because Cartesians couple
    every internal degree of freedom to every other. geomeTRIC's TRIC
    coordinates are built from bonds, angles and torsions plus rigid-body
    fragments, which is the difference between a dozen cycles and a hundred on
    a 60-atom emitter -- and every cycle here is a full BSE@GW gradient.

    It is an optional dependency and not in the environment by default:

        pip install --no-deps --target <dir> geometric   # numpy/scipy already present
        PYTHONPATH=<dir> python ...

    `--no-deps` is not incidental -- a plain install pulls numpy 2.x, which
    shadows the environment's 1.23 and breaks the compiled pyscf against it.

    LICENCE, and why this is IMPORTED and never vendored. geomeTRIC is BSD
    3-clause with a fourth clause added: its source may not be placed in a
    machine-learning training dataset. Clauses 1-2 bind anyone who
    REDISTRIBUTES the source or a binary -- they must carry the notice -- so
    copying it into this tree would take on that obligation and the added
    restriction along with it. Depending on it does not: using a BSD library
    imposes nothing on the calling code. Hence the import guard below and the
    Cartesian `optimize` fallback, which keeps every result reachable with no
    dependency at all.

    The engine below reports `surface.total_gradient`, through
    `surface.evaluate`, so everything the frozen conventions imply still
    holds: the whole optimization is one smooth surface, and `refreeze` is
    still the way to measure the drift of that surface from the one the fit
    would choose at the end.

    refreeze: after convergence, rebuild the frozen conventions at the new
        geometry (`surface.refreeze`) and optimize again, up to this many
        times, exactly as `optimize` does. 0 marks the record
        `refreeze: 'not measured'`, so a null `refreeze_shift` can only have
        come from an explicit 0.

    `info['opt_grad_max']` is max |dE/dR| at the CONVERGED geometry R*, the
    residual, and not the driving force at R0 that `grad_max` means elsewhere;
    `info['grad_max']` is the same float, kept for callers that read it.

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
                                           workdir))


def geometric_engine():
    """geomeTRIC's engine base, molecule and driver, or a refusal naming the
    dependency-free alternative.

    An optional dependency, absent from the environment by default: importing
    it at module level would make every property routine need it.
    """
    try:
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
                    verbose, workdir):
    """The engine `optimize_geometric` documents."""
    Engine, GeoMolecule, run_optimizer = geometric_engine()

    gm = GeoMolecule()
    gm.elem = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    gm.xyzs = [np.asarray(mol.atom_coords()) * BOHR_TO_ANGSTROM]
    gm.build_topology()

    trace = []

    class _Surface(Engine):
        def calc_new(self, coords, dirname):
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
    out = run_optimizer(customengine=engine, coordsys=coordsys,
                        maxiter=maxiter, convergence_set=converge,
                        input='es', prefix=os.path.join(tmp, 'es'), check=0)

    xyz = np.asarray(out.xyzs[-1]) / BOHR_TO_ANGSTROM
    mol_opt = mol.copy()
    mol_opt.set_geom_(xyz, unit='Bohr')
    mol_opt.build(False, False)
    residual = trace[-1]['grad_max']
    info = {'converged': True, 'cycles': len(trace), 'history': trace,
            'energy': trace[-1]['e'], 'omega': trace[-1]['omega'],
            'grad_max': residual, 'opt_grad_max': residual, 'status': 'ok',
            'engine': 'geometric', 'optimizer': 'geometric',
            'coordsys': coordsys}

    if refreeze:
        mol2, info2 = _geometric_walk(surface.refreeze(mol_opt), mol_opt,
                                      maxiter, converge, coordsys,
                                      refreeze - 1, verbose, workdir)
        shift = float(np.abs(mol2.atom_coords() - mol_opt.atom_coords()).max())
        info.update(refreeze_shift=shift,
                    refreeze_denergy=float(info2['energy'] - info['energy']),
                    refreeze_info=info2, energy=info2['energy'],
                    omega=info2['omega'], refreeze='measured')
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
    number means. Which one did run is in `info['optimizer']`, written by the
    engine itself and passed through untouched -- the record has to say which
    method produced the geometry, because the two converge to different
    residuals on the same minimum.
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
    """dE/dR of the energy THIS mean field reported, whatever built its exchange.

    ONE DEFINITION, in `isdf_jk`. This was a second copy of it that omitted
    `grid_response`, and the omission is not small: pyscf defaults it to False,
    which drops d(becke weights)/dR, and the resulting force breaks
    translational invariance by 8.1e-06 Ha/Bohr on water/cc-pVDZ/B3LYP where
    the correct one sits at 4.2e-15. A finite-difference Hessian differencing
    that force inherits the violation, and an optimizer walks downhill on a
    surface whose minimum is not the one whose energy it is printing.

    Rank 0's force on every rank: pyscf's threaded GEMM adds its partial sums
    in thread-arrival order and its density-fitted gradient blocks by each
    process's free memory, so every rank's own call differs in its last bits.
    """
    # cycle: the gradient package imports the surface protocol this module defines
    from src.Base.isdf_jk import mean_field_skeleton_force

    return lockstep(mean_field_skeleton_force(mf))


class MeanFieldSurface:
    """The mean field's own energy as a `PotentialEnergySurface`.

    The ground state needs no excited-state machinery and no frozen
    conventions, so `refreeze` is the identity. This exists so `optimize` --
    which needs nothing outside this repository -- can relax a ground state on
    a machine with neither geomeTRIC nor pyberny installed.

    THE SCF IS THE ONLY STAGE THERE IS, and it is what the ranks divide. There
    is no post-SCF step to split over them, so a factory that hands back a
    mean field it has BUILT AND NOT RUN leaves the convergence to
    `converged_factory`: every rank runs pyscf's driver against the reduced
    J/K and the reduced quadrature, contributing its block of the auxiliary
    index and of the grid. A factory that converges its own is used as it is,
    its orbitals locked to rank 0's, and outside a distributed region this is
    the call the factory would have made. Each rank forms the force from rank
    0's orbitals with its own node's arithmetic and `mean_field_force` hands
    every rank rank 0's; the energy is the locked mean field's own.
    """

    def __init__(self, mol, scf_factory, mf=None):
        """`mf` is the reference mean field where the caller already has one.

        Only the DECLARATION reads it -- which functional E_0 is, which is a
        property of the reference and not of a geometry -- so a surface built
        without one converges its own the first time it is asked.
        """
        self.mol0 = mol
        self._scf = scf_factory
        self._mf0 = mf

    def _mean_field(self, mol):
        """The factory's mean field at `mol`, converged over the ranks if it
        was not already, and rank 0's on every rank."""
        # cycle: the gradient package imports the surface protocol this module defines
        from src.gradients.factor_chain import converged_factory

        return lockstep_mean_field(converged_factory(self._scf)(mol))

    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the mean field this surface evaluates on -- a gas-phase
        surface, so the factory's own; `mf` given is used as it is."""
        mol = self.mol0 if mol is None else mol
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
        """What this surface computes: E_0 alone, with no state on it.

        No post-SCF step, so no environment of its own: whatever the factory
        attached is already inside `mf.e_tot`, and `SurfacePhysics`'s default
        is what a surface that resolves none declares.
        """
        return SurfacePhysics(self.physics_ground_state, None)

    def total_energy(self, mol=None, mf=None):
        mol = mol if mol is not None else self.mol0
        return float((mf or self._mean_field(mol)).e_tot)

    def total_gradient(self, mol=None, mf=None):
        mol = mol if mol is not None else self.mol0
        mf = mf or self._mean_field(mol)
        return np.asarray(mean_field_force(mf)), float(mf.e_tot), {}

    def refreeze(self, mol):
        return MeanFieldSurface(mol, self._scf)

    def label(self):
        return 'mean-field ground state'


def relax_ground_state(mol, scf_factory, engine='auto', maxsteps=100,
                       verbose=False, converge=None):
    """Relax the mean-field ground state. Returns (mol, mf).

    THE VIBRONIC QUANTITIES ARE DEFINED ABOUT THIS GEOMETRY, not about whatever
    the input file happened to contain. Huang-Rhys factors expand both surfaces
    in the ground state's normal modes, and normal modes are only modes at a
    stationary point; a reorganization energy measured from a non-stationary
    reference silently includes the ground state's own relaxation. Measured on
    formaldehyde/cc-pVDZ, taking the unrelaxed input as the reference moved the
    total Huang-Rhys factor from 1.278 to 0.896 and the effective mode from
    2057 to 2003 cm^-1 -- a 30% error in S, from a geometry that looked fine.
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
            # geomeTRIC's own dependencies count here: it needs numpy, scipy,
            # networkx AND six, and a --no-deps install that forgets one fails
            # exactly like an absent package.
            tried.append(f'{name} ({exc})')
            continue
        kw = {} if converge is None else {'convergence_set': converge}
        opt = solver.optimize(mf, maxsteps=maxsteps, **kw)
        return opt, scf_factory(opt)
    # Neither external optimizer is installed. Fall back to this repository's
    # own Cartesian trust-region method rather than refusing: it needs nothing
    # outside src/, and a ground-state minimum is the easy case for it.
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

    A number worth printing next to any Huang-Rhys spectrum: it is the size of
    the term the harmonic expansion is pretending is zero.
    """
    return float(np.abs(mf.Gradients().kernel()).max())
