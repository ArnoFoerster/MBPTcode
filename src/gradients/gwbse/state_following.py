"""The Casida step of the excited-state chain and the following of its states.

One forward shared by several states (`SharedForward`), the dense or Davidson
Casida solve on it (`CasidaSolution`), which together are everything a reverse
pass reads, and the root-following that keeps one excited state on one surface
along a walk. `ExcitedStateChain` inherits these methods; `forward`,
`shared_forward`, `solve_casida` and `tracked_root` are its public stages.
"""
import warnings
from dataclasses import dataclass

import numpy as np

from src.Base.factorization.sliced_factors import GridTileRows, SlicedFactors
from src.Base.step_failed import StepFailed
from src.Base.numerics.constants import (ROOT_FOLLOW_MARGIN_MIN,
                                         ROOT_FOLLOW_RESOLVE_FACTOR,
                                         ROOT_FOLLOW_WEIGHT_MIN, BSE_FORCE_RESIDUAL_TOL,
                                         CASIDA_PHASE_TIE_TOL)
from src.Base.numerical_flags import collecting, gathered
from src.Base.utils.parallel.mpi_grid import current_comm
from src.SingleReference.GW.qp_solve import add_reaction_field
from src.SingleReference.GW.qp_states import frozen_scissor
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.gradients.gwbse.isdf_bse_adjoint import (
    isdf_bse_backward, isdf_bse_backward_rows, isdf_interstate_backward)
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)
from src.gradients.gwbse.bse_isdf import (bse_backward, bse_blocks, bse_cache,
                                          bse_solve, interstate_backward)
from src.gradients.evaluation import Evaluation
from src.gradients.gwbse.reaction_field_adjoint import static_screening
from src.gradients.state_identity import ORBITAL_IDENTITY
from src.gradients.state_identity import follow_state, transported_overlap


@dataclass(eq=False)
class SplitResponse:
    """The static half of a split equilibrium continuum at one geometry
    (`split_equilibrium`): D_s, its static screening W_s, the metric roots
    R_inf and R_s, and T = R_inf^-1 R_s, which carries W_s into the optical
    gauge of the kernel's factor as T W_s T^T."""

    d: np.ndarray
    root_inf: np.ndarray
    root_s: np.ndarray
    t: np.ndarray
    environment: object
    w: np.ndarray = None


@dataclass(eq=False)
class SharedForward:
    """What one geometry's forward pass holds before the Casida step: the
    factors, eps, eps^QP, the static W_aux, mu, the bare factor, the Eq. (18)
    shift and its screening pair, the quasiparticle set's tape and the static
    correction it read, and the conventions in this geometry's indices. No
    spin and no root enters any of them."""

    mol: object
    mf: object
    auxmol: object
    crd: object
    x_mo: object
    d: object
    eps: np.ndarray
    eps_qp: np.ndarray
    w_aux: np.ndarray
    mu: float
    d_bare: object
    shift: object
    screening: object
    qp_tape: object
    xc_correction: object
    split: object = None
    qp_h: object = None
    resolved: object = None

    def release(self):
        """Drop the quasiparticle tape's proj(tau) rows and slices."""
        if self.qp_tape is not None:
            self.qp_tape.release()


@dataclass(eq=False)
class CasidaSolution:
    """One spin's Casida step on a `SharedForward`: every excitation energy
    solved, ascending (`omega`), the eigenvectors X and Y (n_ov, nroots),
    phase-fixed (`casida_phase`), and the explicit adjoint's block cache,
    filled at the first reverse call where the solve left it empty."""

    omega: np.ndarray
    x: np.ndarray
    y: np.ndarray
    cache: dict


class RootFollowingLost(StepFailed):
    """The followed excited state has no counterpart among the roots solved
    at this geometry, even after solving for more of them."""


class StateFollower:
    """The forward, the Casida solve and the following of states."""

    def __init__(self, chain):
        self.chain = chain

    def _casida(self, x_mo, d, eps_qp, w_aux, qp_h=None, at=None):
        """(Omega, X, Y, cache): every root densely, or the lowest `nroots`
        matrix-free on the SAME static W, sorted.

        at: (mol, mf) of the geometry solved. The matrix-free solve then
        starts from the roots of the last geometry a walk accepted, carried
        into these orbitals through the MO overlap (`davidson_warm_start`, a
        `DavidsonWarmStart`; None: always cold), and offers its own roots for
        `commit` to hold. A single point, which commits nothing, is cold.

        The cache is what the explicit adjoint reads. The dense route's blocks
        are the ones its Casida matrices were built from and ride along on the
        explicit route; the matrix-free route returns it empty and
        `casida_seeds` fills it at the first reverse call, so an energy
        never builds a three-index block. On the grid route it stays empty.
        """
        n_ov = self.chain.nocc * (len(eps_qp) - self.chain.nocc)
        one_body = None if qp_h is None else qp_h.one_body(self.chain.nocc)
        if self.chain.solver_used(n_ov) == 'dense':
            a, b, cache = bse_blocks(x_mo, d, eps_qp, w_aux, self.chain.nocc,
                                     spin=self.chain.spin, bse_tda=self.chain.bse_tda)
            if one_body is not None:
                a = dense_one_body(a, one_body, self.chain.nocc,
                                   len(eps_qp) - self.chain.nocc)
            om, xn, yn = casida_phase(*bse_solve(a, b))
            return om, xn, yn, (cache if self.chain.bse_adjoint == 'explicit'
                                else {})
        if self.chain.nroots <= self.chain.state:
            raise ValueError(f'nroots={self.chain.nroots} does not reach state {self.chain.state}')
        lr = LinearResponseSolver(np.asarray(eps_qp, float), spin_mode='restricted')
        # Sliced, the block action reads the rows themselves: D and X_o are
        # gathered once per build, never per trial vector.
        factors = x_mo if isinstance(x_mo, SlicedFactors) else (x_mo, d)
        # Hellmann-Feynman reads the eigenvectors, so an unconverged root is a
        # wrong force by its residual -- but only a root the force reads (the
        # state and those below it, every root when the state is followed by
        # overlap), and only above the residual the force needs,
        # BSE_FORCE_RESIDUAL_TOL. Such a root is kept as solved, warned about
        # and logged in `casida_log` (the policy: refuse inputs, warn
        # numerics); the timings always ride along, since the per-root
        # residuals the verdict reads are recorded there.
        solve = ({'stats': {}, 'timings': {}} if self.chain.timer is not None
                 else {'stats': None, 'timings': {}})
        if self.chain.timer is not None:
            self.chain.davidson_solves.append(solve)
        read = None if self.chain.track is not None else self.chain.state + 1
        read_tol = max(self.chain.bse_conv_tol, BSE_FORCE_RESIDUAL_TOL)
        holder = self.chain.__dict__.get('davidson_warm_start')
        warm = holder is not None and at is not None
        guess = (holder.guess(at[0], at[1].mo_coeff, self.chain.nocc, self.chain.spin)
                 if warm else None)
        tracked = {} if warm else None
        om, xn, yn = solve_casida_davidson(
            lr, self.chain.nocc, numerics=self.chain.settings, polarizability='BSE',
            W_aux=w_aux, isdf_factors=factors, spin=self.chain.spin,
            stats=solve['stats'], timings=solve['timings'],
            refuse_unconverged=False, read_roots=read, read_tol=read_tol,
            tda=self.chain.bse_tda, one_body=one_body, probe=self.chain.probe,
            guess=guess, tracked=tracked)
        if warm:
            holder.offer(at[0], at[1].mo_coeff, self.chain.nocc, self.chain.spin,
                         tracked.get('X', xn), tracked.get('Y', yn))
        self._flag_casida(solve['timings'], read, read_tol)
        order = np.argsort(om)
        om, xn, yn = casida_phase(om[order], xn[:, order], yn[:, order])
        return om, xn, yn, {}

    def _flag_casida(self, timings, read, read_tol):
        """Warn about, and log in `casida_log`, the roots of one matrix-free
        Casida solve the force reads (the lowest `read`, None: all) that
        Davidson left unusable for a force, from the flags and residuals it
        recorded in `timings`: 'not_converged', a read root above `read_tol`
        (its |r| recomputed from the returned vectors), or 'unconfirmed', a
        read root an unsettled extra Ritz pair may lie below (Tamm-Dancoff;
        its residual within conv_tol). Once per kind, roots, spin, state and
        geometry; the roots and vectors are kept as solved."""
        flags = timings.get('davidson_converged')
        if flags is None or all(flags):
            return
        every = timings.get('davidson_residuals')
        # each root's own unconfirmed flag, where Davidson records it apart
        # (`_record_verdict`): a root both unconverged and unconfirmed is
        # then logged as unconfirmed whatever its residual
        doubted = timings.get('davidson_unconfirmed')
        n = len(flags) if read is None else min(int(read), len(flags))
        events = {'not_converged': [], 'unconfirmed': []}
        for k in range(n):
            if flags[k]:
                continue
            r = None if every is None else float(every[k])
            if doubted is not None:
                if doubted[k]:
                    events['unconfirmed'].append((k, r))
                if r is not None and r > read_tol:
                    events['not_converged'].append((k, r))
            elif r is None or r <= self.chain.bse_conv_tol:
                # converged in the loop, yet flagged: unconfirmed
                events['unconfirmed'].append((k, r))
            elif r > read_tol:
                events['not_converged'].append((k, r))
        geometry = self.chain.resolved.geometry
        log = self.chain.__dict__.setdefault('casida_log', [])
        for kind, hit in events.items():
            if not hit:
                continue
            entry = {'kind': kind, 'spin': self.chain.spin, 'state': int(self.chain.state),
                     'roots': [k for k, _ in hit],
                     'residuals': [r for _, r in hit],
                     'read_tol': float(read_tol), 'geometry': geometry}
            if any(all(e.get(key) == entry[key] for key in
                       ('kind', 'spin', 'state', 'roots', 'geometry'))
                   for e in log):
                continue
            log.append(entry)
            detail = (
                'left above read_tol = %g at |r| = %s' % (
                    read_tol, ', '.join(f'{r:.2e}' for _, r in hit))
                if kind == 'not_converged' else
                'not confirmed as the lowest (an extra Ritz pair was still '
                'unsettled)')
            warnings.warn(
                f'excited-state chain ({self.chain.spin}, state {self.chain.state}): the '
                f'Casida roots {entry["roots"]} the force reads were {detail}; '
                'kept as solved, so the energy and force carry their error, '
                f'and logged in casida_log (geometry {str(geometry)[:12]})',
                RuntimeWarning, stacklevel=3)

    def forward(self, mol, mf):
        """(`SharedForward`, `CasidaSolution`) at `mol` on `mf`: the
        evaluation held there when it is whole, else one forward pass, held
        in its place.

        Only the newest geometry keeps its forward: an older one keeps its
        roots and forces, and asked for its forward again it is evaluated
        again on the same mean field.
        """
        # the roots a matrix-free solve checks for a force: all of them for
        # a followed state, else the state and those below it (`_casida`)
        read = None if self.chain.track is not None else self.chain.state + 1
        held = self.chain.evaluations.get(mol)
        if (held is not None and held.mf is mf and held.shared is not None
                and (held.read is None or (read is not None
                                           and read <= held.read))):
            return held.shared, held.solution
        raised = []
        with collecting(raised):
            shared, solution = self._forward_pass(mol, mf)
        self._log_flags(gathered(raised))
        for other in self.chain.evaluations.values():
            other.shared = None
        if held is None or held.mf is not mf:
            held = self.chain.evaluations.put(mol, Evaluation(mf))
        held.shared, held.solution, held.read = shared, solution, read
        return shared, solution

    def _log_flags(self, raised):
        """Log in `numerical_flag_log` the named flags (`numerical_flags`)
        a forward pass raised on any thread or rank, each once, at the
        geometry it was first raised at; each was warned about where it
        was raised."""
        geometry = self.chain.resolved.geometry
        log = self.chain.__dict__.setdefault('numerical_flag_log', [])
        held = {(e['flag'], e['message']) for e in log}
        log.extend(dict(e, geometry=geometry) for e in raised
                   if (e['flag'], e['message']) not in held)

    def _forward_pass(self, mol, mf):
        """One forward pass: the shared forward and this spin's Casida step,
        with an adaptive set checked on the production eigenvectors."""
        shared = self.chain.shared_forward(mol, mf)
        shared, solved = self.chain.verified(
            mol, mf, shared, {self.chain.spin: self.chain.solve_casida(shared)},
            lambda fresh: {self.chain.spin: self.chain.solve_casida(fresh)})
        return shared, solved[self.chain.spin]

    def verified(self, mol, mf, shared, solved, solve):
        """(shared, solved) once the adaptive set holds: the set is checked
        on the production eigenvectors `solved` ({spin: `CasidaSolution`}) at
        the reference geometry (`verify_selection`), and while it grows the
        shared forward is built again at (mol, mf) and `solve(shared)` solves
        it again. The one re-check for a chain's own pass and a manifold's."""
        while self.chain.selection.verify_selection(shared, solved):
            shared.release()
            shared = self.chain.shared_forward(mol, mf)
            solved = solve(shared)
        return shared, solved

    def solve_casida(self, shared, warm=True):
        """The `CasidaSolution` of this chain's spin on `shared`.

        warm: the matrix-free solve starts from the roots of the last
        geometry a walk committed (`_casida(at=)`); False solves cold, as a
        second spin read off another spin's forward does.
        """
        with self.chain.phase('t_casida'):
            om, xn, yn, cache = self._casida(
                shared.x_mo, shared.d, shared.eps_qp, shared.w_aux,
                **coupled(shared.qp_h),
                at=(shared.mol, shared.mf) if warm else None)
        return CasidaSolution(om, xn, yn, cache)

    def shared_forward(self, mol, mf):
        """The `SharedForward` at one geometry: everything before the Casida
        step, which reads neither `spin` nor `state`, on the conventions
        resolved in this geometry's orbital indices; a first one elsewhere
        than the reference geometry is preceded by the reference's
        (`anchored_first`). The quasiparticle set's tape rides with it: the
        reverse solve reads proj(tau) and the slices from it instead of
        sweeping them again, bitwise. A quantity that needs the kernel but
        not the supermolecular roots -- the fragment-diabatic elements
        (`fragment_bse`) -- starts here, and so does its gradient
        (`fold_to_nuclei` reads nothing from the roots). With no roots there
        is no eigenvector check of an adaptive set (`verify_selection`)."""
        self.anchored_first(mol, lambda: self.chain.forward(self.chain.mol0, self.chain.mf0))
        resolved = self.chain.resolve_at(mol, mf)
        x_mo, d, eps, mu, auxmol, crd, _, d_bare = self.chain.factors_for(mol, mf)
        # A SPLIT EQUILIBRIUM continuum screens the density terms -- the
        # direct W and Eq. (18) -- with the static-dressed D_s, while the
        # kernel's exchange keeps the optical D (`split_equilibrium`).
        split = self.chain.split_factor(mol, auxmol, crd)
        if split is not None:
            split = SplitResponse(*split)
        d_w = d if split is None else split.d
        # ONE STATIC SCREENING. The BSE kernel's W and the dressed half of
        # Eq. (18) are the same [1 - chi0(0)]^-1 on the same axis; the orbital
        # densities are Eq. (18)'s alone.
        with self.chain.phase('t_screening'):
            dressed = static_screening(x_mo, d_w, eps, self.chain.nocc, self.chain.w_grid,
                                       densities=d_bare is not None)
            w_aux = dressed[1]
            pair = (None if d_bare is None else
                    (dressed, static_screening(x_mo, d_bare, eps, self.chain.nocc,
                                               self.chain.w_grid)))
        if split is not None:
            split.w = w_aux
            w_aux = split.t @ w_aux @ split.t.T
        # THE SELF-ENERGY SCREENS BARE. Eq. (18) is the static approximation to
        # Sigma[W_solv] - Sigma[W_gas], so screening Sigma dynamically as well
        # counts the reaction field twice; the kernel below keeps D.
        shift, screening = self.chain.reaction_field(x_mo, d_w, d_bare, eps,
                                                screening=pair)
        d_sigma = d if d_bare is None else d_bare
        qp_h = None
        with self.chain.phase('t_qp'):
            if (self.chain.qp_select is not None and self.chain.selecting is None
                    and self.chain.resolved.qp_partition is None):
                self.chain.selection.select_qp_set(mol, mf, x_mo, d, d_sigma, eps,
                                    w_aux, shift)
            while True:
                xc = self.chain.xc_correction(mf, self.chain.resolved.qp_set, shift)
                out, route_out = self.chain.quasiparticles.qp_set_solve(
                    x_mo, d_sigma, eps, mu, xc,
                    np.zeros(len(self.chain.resolved.qp_set)))
                # the set the roots belong to: settling may have shrunk it
                xc = route_out['xc_correction']
                # G0W0: Eq. (18) whole on each root, never Z x shift inside
                ws = add_reaction_field(np.asarray(out[0], float), shift,
                                        self.chain.resolved.qp_set)
                if not self.chain.selection.refine_selection(eps, ws, shift):
                    break
                tape = route_out.get('tape')
                if tape is not None:
                    tape.release()
        qp_tape = route_out.get('tape')
        eps_qp = eps.copy()
        eps_qp[self.chain.resolved.qp_set] = self.chain.quasiparticles.block_shared(eps, ws)
        if self.chain.qp_hamiltonian:
            qp_h = self.chain.quasiparticles.qp_hamiltonian(
                mf, eps, ws, route_out,
                None if shift is None else (x_mo, d_w, d_bare, screening,
                                            shift))
        # inside a group the quasiparticle energy is H's diagonal
        ws = np.array(ws, float)
        pos = {int(p): i for i, p in enumerate(self.chain.resolved.qp_set)}
        for group in (qp_h.groups if qp_h is not None else ()):
            diagonal = np.diag(group.hamiltonian)
            eps_qp[list(group.orbitals)] = diagonal
            ws[[pos[p] for p in group.orbitals]] = diagonal
        # OUTSIDE THE SET, THE FROZEN SCISSOR. Calibrated on these roots
        # at the reference geometry and spent unchanged afterwards; the
        # environment's static term below is added on top of it, not
        # instead of it.
        if self.chain.outside_scissor != 'tiers':
            self.chain.quasiparticles.freeze_outside_curve(
                mol, mf, x_mo, d_sigma, eps, eps_qp, shift)
        self.chain.quasiparticles.freeze_outside_shift(eps, ws, shift)
        for p in np.flatnonzero(self.chain.quasiparticles.outside_window(len(eps))):
            scissor_p = frozen_scissor(self.chain.resolved.outside_shift, p)
            if scissor_p is not None:
                eps_qp[p] = eps[p] + scissor_p
        # beyond the window, Delta(eps_p) at this geometry's eps_p
        for idx, delta, _ in self.chain.quasiparticles.curve_at(eps):
            eps_qp[idx] = eps[idx] + delta
        env_outside = self.chain.quasiparticles.env_static_outside(shift, len(eps))
        if env_outside is not None:
            eps_qp = eps_qp + env_outside
        if self.chain.resolved.qp_partition is not None:
            # what the check at a walk's end reads at this geometry
            self.chain.last_forward = {'coords': np.array(mol.atom_coords()),
                                 'eps': eps, 'eps_qp': eps_qp, 'shift': shift}
        self.chain.conventions.anchor(mol)
        return SharedForward(mol, mf, auxmol, crd, x_mo, d, eps, eps_qp, w_aux,
                             mu, d_bare, shift, screening, qp_tape, xc, split,
                             qp_h, resolved)

    def anchored_first(self, mol, at_reference):
        """Run `at_reference`, an evaluation at (mol0, mf0), before the first
        one at `mol` when that is another geometry: what the first forward
        freezes -- the set, the scissor, the quadrature, the seeds -- is the
        reference's, never that of where a walk happens to start (a nudged
        start is 0.02 Bohr away)."""
        if self.chain.conventions.anchored or np.array_equal(
                np.asarray(mol.atom_coords()),
                np.asarray(self.chain.mol0.atom_coords())):
            return
        at_reference()

    def tracked(self, mol, mf):
        """(root, `SharedForward`, `CasidaSolution`) of the followed state at
        `mol` on `mf`, off the evaluation held there (`forward`); a wider
        re-solve (`tracked_root`) replaces the solution held."""
        shared, solution = self.chain.forward(mol, mf)
        root, wider = self.chain.tracked_root(mol, mf, shared, solution)
        held = self.held(mol, mf)
        if wider is not solution and held is not None:
            held.solution = wider
        return root, shared, wider

    def held(self, mol, mf):
        """The `Evaluation` held at `mol` on `mf`, or None."""
        held = self.chain.evaluations.get(mol)
        return held if held is not None and held.mf is mf else None

    def _matched_at(self, mol, mf, followed):
        """Hold `followed`, the state this chain matched at `mol` on `mf`,
        in that geometry's `Evaluation` until `commit` adopts it."""
        held = self.held(mol, mf)
        if held is None:
            held = self.chain.evaluations.put(mol, Evaluation(mf))
        held.followed[self.chain.selection_owner] = followed

    def tracked_state(self, mol, mf, solution):
        """Which root of THIS geometry is the state being followed, among the
        roots `solution` holds; `tracked_root` without the wider re-solve."""
        return self.chain.tracked_root(mol, mf, None, solution, widen=False)[0]

    def tracked_root(self, mol, mf, shared, solution, widen=True):
        """(root, `CasidaSolution`): the state being followed at THIS
        geometry, and the solve it was found in (`shared` is what a wider
        solve is taken on).

        `self.state` is an index into an energy-ordered manifold, and an
        optimizer that keeps it follows whatever is n-th at each step rather
        than one state. Following the overlap with the previous step keeps the
        identity through a crossing, where the index is exactly what changes.

        Propagates from ACCEPTED geometry to accepted geometry rather than
        against the first one: the overlap with a distant reference decays,
        and a relaxation moves far. Every evaluation is matched against the
        last geometry `commit` accepted and moves nothing itself, so a
        rejected trial step, a Hessian displacement or an energy read at
        another geometry is never what the next step is matched against.
        The first evaluation of a chain that follows nothing yet is its
        anchor, committed as it is made. The price of propagating is that a
        sequence of small misassignments can ratchet onto a different state,
        so the overlap with the ORIGINAL anchor is logged beside each match
        and is what shows that if it happens.

        A best overlap below ROOT_FOLLOW_WEIGHT_MIN means the state is not
        among the roots solved. With `widen` the same forward is solved again
        for ROOT_FOLLOW_RESOLVE_FACTOR times the roots (at most the pair
        space) and matched again, and the wider solve is what comes back;
        still below, `RootFollowingLost` is raised. Continuing on the best of
        the wrong roots puts another state's force into the walk. A best
        overlap within ROOT_FOLLOW_MARGIN_MIN of the runner-up is two mixed
        roots: warned about, and flagged as 'mixed' in `follow_log`; the walk
        carries on.
        """
        xn = solution.x
        if self.chain.track is None:
            return self.chain.state, solution
        here = (mol, np.asarray(mf.mo_coeff, float))
        if self.chain.followed is None:
            self.chain.followed = here + (xn[:, [self.chain.state]].copy(),
                                     solution.y[:, [self.chain.state]].copy())
            self.chain.anchor = self.chain.followed
            self._matched_at(mol, mf, self.chain.followed)
            self.chain.follow_log.append(dict(index=self.chain.state, weight=1.0,
                                        margin=1.0, anchor=1.0,
                                        identity=ORBITAL_IDENTITY))
            self.chain.follow_coords.append(np.array(mol.atom_coords(), float))
            return self.chain.state, solution
        index, weight, margin = self._match_followed(here, solution)
        solved = xn.shape[1]
        if weight < ROOT_FOLLOW_WEIGHT_MIN and widen and solved < xn.shape[0]:
            solution = self._wider_solve(shared, solution)
            index, weight, margin = self._match_followed(here, solution)
        xn, yn = solution.x, solution.y
        if weight < ROOT_FOLLOW_WEIGHT_MIN:
            raise RootFollowingLost(
                f'the tracked state overlaps its own previous step by only '
                f'{weight:.2f} among the {xn.shape[1]} roots solved here '
                f'(first {solved}): it is not in the manifold, and the walk '
                f'is refused rather than continued on another state. Shorten '
                f'the step or raise nroots.')
        mixed = bool(margin < ROOT_FOLLOW_MARGIN_MIN)
        if mixed:
            warnings.warn(
                f'two roots share the tracked state: overlaps differ by only '
                f'{margin:.2f}. They have mixed, so which one is followed is '
                f'decided by that difference and the state carried past this '
                f'geometry may not be the one asked for.',
                RuntimeWarning, stacklevel=2)
        anchor = float(np.abs(transported_overlap(
            self.chain.anchor[:2], here, self.chain.nocc, self.chain.anchor[2],
            self.chain.anchor[3], xn[:, [index]], yn[:, [index]])[1, 1]))
        self._matched_at(mol, mf, here + (xn[:, [index]].copy(),
                                          yn[:, [index]].copy()))
        step = dict(index=int(index), weight=weight, margin=margin,
                    anchor=anchor, omega=float(solution.omega[index]),
                    identity=ORBITAL_IDENTITY)
        if xn.shape[1] > solved:
            step['widened_to'] = int(xn.shape[1])
        if mixed:
            step['mixed'] = True
        self.chain.follow_log.append(step)
        self.chain.follow_coords.append(np.array(mol.atom_coords(), float))
        return int(index), solution

    def following(self):
        """(followed, anchor) -- the last committed state and the first, each
        (mol, mo_coeff, X, Y) -- that a surface built afresh carries on
        (`hand_over`); None when this chain follows nothing (by index, or
        nothing committed yet)."""
        if self.chain.track is None or self.chain.followed is None:
            return None
        return self.chain.followed, self.chain.anchor

    def hand_over(self, held):
        """Follow on from `held`, another chain's `following()`: its first
        evaluation is matched against that state by overlap, not by index."""
        self.chain.followed, self.chain.anchor = held

    def commit(self, mol):
        """Accept `mol`: the state matched there is what every later geometry
        is matched against, and the orbitals the conventions were matched to
        there become their keys (`FrozenConventions.commit`), both read off
        the evaluation held at `mol` whatever was evaluated since, and the
        Davidson roots solved there are what the next geometry's solve starts
        from (`DavidsonWarmStart.accept`). True when
        a followed state was adopted, False for a chain that follows none
        yet or follows by index.

        A chain following a state with no match held at `mol` -- none made
        there, or dropped from the cache since -- refuses (`RuntimeError`):
        walking on would match the next step against a geometry the walk
        left. Committing the geometry already followed adopts nothing."""
        held = self.chain.evaluations.get(mol)
        following = self.chain.track is not None and self.chain.followed is not None
        matched = None if held is None else held.followed.get(self.chain.selection_owner)
        if following and matched is None and not np.array_equal(
                np.asarray(self.chain.followed[0].atom_coords()),
                np.asarray(mol.atom_coords())):
            raise RuntimeError(
                'commit: no state matched at this geometry is held (it was '
                'not evaluated here with the state followed, or is older '
                f'than the last {self.chain.evaluations.size} geometries); the '
                'followed state and the keys stay where they were last '
                'committed')
        if held is not None:
            self.chain.conventions.commit(mol, held.mf)
        holder = self.chain.__dict__.get('davidson_warm_start')
        if holder is not None:
            # the next geometry's Davidson starts from the roots solved here
            holder.accept(mol)
        if not following or matched is None:
            return False
        self.chain.followed = matched
        return True

    def _match_followed(self, here, solution):
        """(index, weight, margin) of the last committed state among the
        roots of `solution` (`follow_state`) by the one rule for the same
        state at another geometry (`transported_overlap`): its orbitals
        matched by `body_overlap`, in which the core rides with its atoms,
        relative to the overlap of the two reference determinants, which it
        keeps near 1."""
        mol0, mo0, x0, y0 = self.chain.followed
        return follow_state(transported_overlap(
            (mol0, mo0), here, self.chain.nocc, x0, y0, solution.x, solution.y), 0)

    def _wider_solve(self, shared, solution):
        """The `CasidaSolution` of the same forward's Casida step solved for
        ROOT_FOLLOW_RESOLVE_FACTOR times the roots of `solution`, at most
        all (cold, as before)."""
        solved = solution.x.shape[1]
        nroots, self.chain.nroots = self.chain.nroots, min(
            ROOT_FOLLOW_RESOLVE_FACTOR * solved, solution.x.shape[0])
        try:
            return self.chain.solve_casida(shared, warm=False)
        finally:
            self.chain.nroots = nroots

    def casida_seeds(self, shared, solution, n, m=None):
        """(eps_qp_bar, X_bar, D_bar, W_aux_bar) of Omega_n, or of the
        interstate element <m| dH |n>, by the realization `bse_adjoint` names.

        The explicit route fills the forward pass's cache in place the first
        time it is asked, so every root reversed off one held forward pass
        reads the same blocks and builds them once.
        """
        x_mo, d, eps_qp, w_aux = shared.x_mo, shared.d, shared.eps_qp, shared.w_aux
        nocc, cache, xn, yn = self.chain.nocc, solution.cache, solution.x, solution.y
        block = self.chain.reverse.adjoint_rows(x_mo)
        if self.chain.bse_adjoint == 'grid' and block is not None:
            # the grid adjoint's own rows, moved into the adjoint's tiles:
            # X_bar and D_bar are never whole on any rank
            comm = current_comm()
            e_bar, x_rows, d_rows, w_bar = isdf_bse_backward_rows(
                n, x_mo, d, eps_qp, w_aux, nocc, xn, yn, spin=self.chain.spin,
                bse_tda=self.chain.bse_tda, bra=m, comm=comm)
            npts = (x_mo.npts if isinstance(x_mo, SlicedFactors)
                    else len(x_mo))
            return (e_bar,
                    GridTileRows.from_blocks(x_rows, npts, block, comm),
                    GridTileRows.from_blocks(d_rows, npts, block, comm),
                    w_bar)
        if self.chain.bse_adjoint == 'grid':
            kw = dict(spin=self.chain.spin, bse_tda=self.chain.bse_tda)
            if m is None:
                return isdf_bse_backward(n, x_mo, d, eps_qp, w_aux, nocc, xn,
                                         yn, **kw)
            return isdf_interstate_backward(m, n, x_mo, d, eps_qp, w_aux, nocc,
                                            xn, yn, **kw)
        if not cache:
            cache.update(bse_cache(x_mo, d, eps_qp, nocc,
                                   spin=self.chain.spin, bse_tda=self.chain.bse_tda))
        if m is None:
            return bse_backward(n, x_mo, d, eps_qp, w_aux, nocc, cache, xn, yn,
                                **self.chain.quasiparticles.tile_kw())
        return interstate_backward(m, n, x_mo, d, eps_qp, w_aux, nocc, cache,
                                   xn, yn, **self.chain.quasiparticles.tile_kw())


def coupled(qp_h):
    """`_casida`'s keyword for a quasiparticle Hamiltonian, none without one."""
    return {} if qp_h is None else {'qp_h': qp_h}


def dense_one_body(a, one_body, no, nv):
    """The dense A with a quasiparticle Hamiltonian's off-diagonal one-body
    blocks added, A_ia,jb += delta_ij H_ab - delta_ab H_ij (`with_one_body`)."""
    (o, h_oo), (v, h_vv) = one_body
    full_o = np.zeros((no, no))
    full_o[np.ix_(o, o)] = h_oo
    full_v = np.zeros((nv, nv))
    full_v[np.ix_(v, v)] = h_vv
    return a + np.kron(np.eye(no), full_v) - np.kron(full_o, np.eye(nv))


def casida_phase(om, xn, yn):
    """(Omega, X, Y) with each root's sign fixed by its largest |X| element,
    made positive; elements within CASIDA_PHASE_TIE_TOL of it are a tie,
    broken by the lowest index.

    An eigensolver returns each root with whatever sign its last Rayleigh-Ritz
    step produced, and a Davidson's path follows the last bits of its input:
    on water/cc-pVDZ Hartree-Fock, 34 of 48 one-ulp moves of a single eps_QP
    flip at least one of four roots. Energies and state gradients do not see
    it; a coupling between two roots, d<m|H|n>/dR and the derivative
    coupling, and a spin-orbit element and its derivative change sign with
    it, so the sign is a convention of the vector alone.
    """
    xn, yn = np.array(xn, copy=True), np.array(yn, copy=True)
    for k in range(xn.shape[1]):
        mag = np.abs(xn[:, k])
        lead = int(np.flatnonzero(mag >= (1.0 - CASIDA_PHASE_TIE_TOL)
                                  * mag.max())[0])
        if xn[lead, k] < 0.0:
            xn[:, k] = -xn[:, k]
            yn[:, k] = -yn[:, k]
    return om, xn, yn
