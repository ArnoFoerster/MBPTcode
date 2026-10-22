"""The Casida step of the excited-state chain and the following of its states.

One forward shared by several states (`SharedForward`), the dense or Davidson
Casida solve, the pieces a reverse pass reads (`ForwardPieces`), and the
root-following that keeps one excited state on one surface along a walk.
`ExcitedStateChain` inherits these methods.
"""
import warnings
from dataclasses import dataclass

import numpy as np

from src.Base.sliced_factors import GridTileRows, SlicedFactors
from src.Base.step_failed import StepFailed
from src.Base.constants import (ROOT_FOLLOW_MARGIN_MIN,
                                ROOT_FOLLOW_RESOLVE_FACTOR,
                                ROOT_FOLLOW_WEIGHT_MIN, BSE_FORCE_RESIDUAL_TOL,
                                CASIDA_PHASE_TIE_TOL)
from src.Base.utils.mpi_grid import current_comm
from src.SingleReference.GW.qp_solve import add_reaction_field
from src.SingleReference.GW.qp_states import frozen_scissor
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.LinearResponse.isdf_bse_adjoint import (
    isdf_bse_backward, isdf_bse_backward_rows, isdf_interstate_backward)
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)
from src.gradients.bse_isdf import (bse_backward, bse_blocks, bse_cache,
                                    bse_solve, interstate_backward)
from src.gradients.evaluation import Evaluation
from src.gradients.reaction_field_adjoint import static_screening
from src.gradients.conventions import ORBITAL_IDENTITY, body_overlap
from src.properties.nonadiabatic import (follow_state, relative_overlap,
                                         state_overlap)


class ForwardPieces(tuple):
    """The seventeen pieces of `ExcitedStateChain._forward`, with the static
    correction <p|Sigma_x - v_xc|p> the quasiparticle solve read
    on the set beside them (`xc_correction`, None at the mean field), so the
    reverse solve reads the forward's own bits; `qp_h`, the quasiparticle
    Hamiltonian's groups (`QPHamiltonian`), None on the diagonal route; and
    `resolved`, the conventions in this geometry's orbital indices
    (`ResolvedConventions`) the forward read and the reverse reads again."""

    def __new__(cls, pieces, xc_correction=None, split=None, qp_h=None,
                resolved=None):
        out = super().__new__(cls, pieces)
        out.xc_correction = xc_correction
        out.split = split
        out.qp_h = qp_h
        out.resolved = resolved
        return out


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

    def pieces(self, cache, xn, yn):
        """The `ForwardPieces` of one Casida solve on this forward."""
        return ForwardPieces(
            (self.mol, self.mf, self.auxmol, self.crd, self.x_mo, self.d,
             self.eps, self.eps_qp, self.w_aux, cache, xn, yn, self.mu,
             self.d_bare, self.shift, self.screening, self.qp_tape),
            self.xc_correction, self.split, self.qp_h, self.resolved)

    def release(self):
        """Drop the quasiparticle tape's proj(tau) rows and slices."""
        if self.qp_tape is not None:
            self.qp_tape.release()


class RootFollowingLost(StepFailed):
    """The followed excited state has no counterpart among the roots solved
    at this geometry, even after solving for more of them."""


class StateFollowingMixin:
    """The forward, the Casida solve and the following of states."""

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
        `_casida_seeds` fills it at the first reverse call, so an energy
        never builds a three-index block. On the grid route it stays empty.
        """
        n_ov = self.nocc * (len(eps_qp) - self.nocc)
        one_body = None if qp_h is None else qp_h.one_body(self.nocc)
        if self.solver_used(n_ov) == 'dense':
            a, b, cache = bse_blocks(x_mo, d, eps_qp, w_aux, self.nocc,
                                     spin=self.spin, bse_tda=self.bse_tda)
            if one_body is not None:
                a = dense_one_body(a, one_body, self.nocc,
                                   len(eps_qp) - self.nocc)
            om, xn, yn = casida_phase(*bse_solve(a, b))
            return om, xn, yn, (cache if self.bse_adjoint == 'explicit'
                                else {})
        if self.nroots <= self.state:
            raise ValueError(f'nroots={self.nroots} does not reach state {self.state}')
        lr = LinearResponseSolver(np.asarray(eps_qp, float), spin_mode='restricted')
        # Sliced, the block action reads the rows themselves: D and X_o are
        # gathered once per build, never per trial vector.
        factors = x_mo if isinstance(x_mo, SlicedFactors) else (x_mo, d)
        # Hellmann-Feynman reads the eigenvectors, so an unconverged root is a
        # wrong force by its residual -- but only a root the force reads (the
        # state and those below it, every root when the state is followed by
        # overlap), and only above the residual the force needs,
        # BSE_FORCE_RESIDUAL_TOL. Such a root is kept as solved, warned about
        # and logged in `casida_log` (refuse inputs,
        # warn numerics); the timings always ride along, since the per-root
        # residuals the verdict reads are recorded there.
        solve = ({'stats': {}, 'timings': {}} if self.timer is not None
                 else {'stats': None, 'timings': {}})
        if self.timer is not None:
            self.davidson_solves.append(solve)
        read = None if self.track is not None else self.state + 1
        read_tol = max(self.bse_conv_tol, BSE_FORCE_RESIDUAL_TOL)
        holder = self.__dict__.get('davidson_warm_start')
        warm = holder is not None and at is not None
        guess = (holder.guess(at[0], at[1].mo_coeff, self.nocc, self.spin)
                 if warm else None)
        tracked = {} if warm else None
        om, xn, yn = solve_casida_davidson(
            lr, self.nocc, numerics=self.settings, polarizability='BSE',
            W_aux=w_aux, isdf_factors=factors, spin=self.spin,
            stats=solve['stats'], timings=solve['timings'],
            refuse_unconverged=False, read_roots=read, read_tol=read_tol,
            tda=self.bse_tda, one_body=one_body, probe=self.probe,
            guess=guess, tracked=tracked)
        if warm:
            holder.offer(at[0], at[1].mo_coeff, self.nocc, self.spin,
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
            elif r is None or r <= self.bse_conv_tol:
                # converged in the loop, yet flagged: unconfirmed
                events['unconfirmed'].append((k, r))
            elif r > read_tol:
                events['not_converged'].append((k, r))
        geometry = self.resolved.geometry
        log = self.__dict__.setdefault('casida_log', [])
        for kind, hit in events.items():
            if not hit:
                continue
            entry = {'kind': kind, 'spin': self.spin, 'state': int(self.state),
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
                f'excited-state chain ({self.spin}, state {self.state}): the '
                f'Casida roots {entry["roots"]} the force reads were {detail}; '
                'kept as solved, so the energy and force carry their error, '
                f'and logged in casida_log (geometry {str(geometry)[:12]})',
                RuntimeWarning, stacklevel=3)

    def _forward(self, mol, mf):
        """(Omega, `ForwardPieces`) at `mol` on `mf`: the evaluation held
        there when it is whole, else one forward pass, held in its place.

        Only the newest geometry keeps its pieces: an older one keeps its
        roots and forces, and asked for its pieces again it is evaluated
        again on the same mean field.
        """
        # the roots a matrix-free solve checks for a force: all of them for
        # a followed state, else the state and those below it (`_casida`)
        read = None if self.track is not None else self.state + 1
        held = self._evaluations.get(mol)
        if (held is not None and held.mf is mf and held.pieces is not None
                and (held.read is None or (read is not None
                                           and read <= held.read))):
            return held.om, held.pieces
        om, pieces = self._forward_pass(mol, mf)
        for other in self._evaluations.values():
            other.pieces = None
        if held is None or held.mf is not mf:
            held = self._evaluations.put(mol, Evaluation(mf))
        held.om, held.pieces, held.read = om, pieces, read
        return om, pieces

    def _forward_pass(self, mol, mf):
        """One forward pass: the shared forward and this spin's Casida step,
        with an adaptive set checked on the production eigenvectors."""
        shared = self._shared_forward(mol, mf)
        _, solved = self.verified(
            mol, mf, shared, {self.spin: self._casida_forward(shared)},
            lambda fresh: {self.spin: self._casida_forward(fresh)})
        return solved[self.spin]

    def verified(self, mol, mf, shared, solved, solve):
        """(shared, solved) once the adaptive set holds: the set is checked
        on the production eigenvectors `solved` ({spin: (Omega, pieces)}) at
        the reference geometry (`verify_selection`), and while it grows the
        shared forward is built again at (mol, mf) and `solve(shared)` solves
        it again. The one re-check for a chain's own pass and a manifold's."""
        while self.verify_selection(shared, solved):
            shared.release()
            shared = self._shared_forward(mol, mf)
            solved = solve(shared)
        return shared, solved

    def kernel_pieces(self, mol, mf):
        """Everything `_forward` builds before the Casida solve, in its layout.

        The same seventeen-long `ForwardPieces` `_forward` returns, with an
        empty cache and no roots (`xn`, `yn` are None): the quasiparticle
        energies, the static screening and the factors the BSE matrix is made
        of. A quantity that needs the kernel but not the supermolecular roots
        -- MBPTcode's fragment-diabatic elements (`properties.fragment_bse`)
        -- starts here, and so does its gradient: `_fold_to_nuclei` reads
        nothing from the roots. With no roots there is no eigenvector check
        of an adaptive quasiparticle set (`verify_selection`): the set is the
        one the chain holds.
        """
        return self._shared_forward(mol, mf).pieces({}, None, None)

    def _casida_forward(self, shared):
        """(Omega, `ForwardPieces`): this chain's spin solved on `shared`."""
        with self.phase('t_casida'):
            om, xn, yn, cache = self._casida(shared.x_mo, shared.d,
                                             shared.eps_qp, shared.w_aux,
                                             **coupled(shared.qp_h),
                                             at=(shared.mol, shared.mf))
        # The quasiparticle set's tape rides with the pieces: the reverse
        # solve reads proj(tau) and the slices from it instead of sweeping
        # them again, bitwise.
        return om, shared.pieces(cache, xn, yn)

    def _shared_forward(self, mol, mf):
        """The `SharedForward` at one geometry: everything before the Casida
        step, which reads neither `spin` nor `state`, on the conventions
        resolved in this geometry's orbital indices; a first one elsewhere
        than the reference geometry is preceded by the reference's
        (`_anchored_first`)."""
        self._anchored_first(mol, lambda: self._forward(self.mol0, self.mf0))
        resolved = self._resolve_at(mol, mf)
        x_mo, d, eps, mu, auxmol, crd, _, d_bare = self._factors_for(mol, mf)
        # A SPLIT EQUILIBRIUM continuum screens the density terms -- the
        # direct W and Eq. (18) -- with the static-dressed D_s, while the
        # kernel's exchange keeps the optical D (`split_equilibrium`).
        split = self.split_factor(mol, auxmol, crd)
        if split is not None:
            split = SplitResponse(*split)
        d_w = d if split is None else split.d
        # ONE STATIC SCREENING. The BSE kernel's W and the dressed half of
        # Eq. (18) are the same [1 - chi0(0)]^-1 on the same axis; the orbital
        # densities are Eq. (18)'s alone.
        with self.phase('t_screening'):
            dressed = static_screening(x_mo, d_w, eps, self.nocc, self.w_grid,
                                       densities=d_bare is not None)
            w_aux = dressed[1]
            pair = (None if d_bare is None else
                    (dressed, static_screening(x_mo, d_bare, eps, self.nocc,
                                               self.w_grid)))
        if split is not None:
            split.w = w_aux
            w_aux = split.t @ w_aux @ split.t.T
        # THE SELF-ENERGY SCREENS BARE. Eq. (18) is the static approximation to
        # Sigma[W_solv] - Sigma[W_gas], so screening Sigma dynamically as well
        # counts the reaction field twice; the kernel below keeps D.
        shift, screening = self._reaction_field(x_mo, d_w, d_bare, eps,
                                                screening=pair)
        d_sigma = d if d_bare is None else d_bare
        qp_h = None
        with self.phase('t_qp'):
            if (self.qp_select is not None and self._selecting is None
                    and self.resolved.qp_partition is None):
                self._select_qp_set(mol, mf, x_mo, d, d_sigma, eps,
                                    w_aux, shift)
            while True:
                xc = self._xc_correction(mf, self.resolved.qp_set, shift)
                out, route_out = self._qp_set_solve(
                    x_mo, d_sigma, eps, mu, xc,
                    np.zeros(len(self.resolved.qp_set)))
                # the set the roots belong to: settling may have shrunk it
                xc = route_out['xc_correction']
                # G0W0: Eq. (18) whole on each root, never Z x shift inside
                ws = add_reaction_field(np.asarray(out[0], float), shift,
                                        self.resolved.qp_set)
                if not self._refine_selection(eps, ws, shift):
                    break
                tape = route_out.get('tape')
                if tape is not None:
                    tape.release()
        qp_tape = route_out.get('tape')
        eps_qp = eps.copy()
        eps_qp[self.resolved.qp_set] = self._block_shared(eps, ws)
        if self.qp_hamiltonian:
            qp_h = self._qp_hamiltonian(
                mf, eps, ws, route_out,
                None if shift is None else (x_mo, d_w, d_bare, screening,
                                            shift))
        # inside a group the quasiparticle energy is H's diagonal
        ws = np.array(ws, float)
        pos = {int(p): i for i, p in enumerate(self.resolved.qp_set)}
        for group in (qp_h.groups if qp_h is not None else ()):
            diagonal = np.diag(group.hamiltonian)
            eps_qp[list(group.orbitals)] = diagonal
            ws[[pos[p] for p in group.orbitals]] = diagonal
        # OUTSIDE THE SET, THE FROZEN SCISSOR. Calibrated on these roots
        # at the reference geometry and spent unchanged afterwards; the
        # environment's static term below is added on top of it, not
        # instead of it.
        if self.outside_scissor != 'tiers':
            self._freeze_outside_curve(
                mol, mf, x_mo, d_sigma, eps, eps_qp, shift)
        self._freeze_outside_shift(eps, ws, shift)
        for p in np.flatnonzero(self._outside_window(len(eps))):
            scissor_p = frozen_scissor(self.resolved.outside_shift, p)
            if scissor_p is not None:
                eps_qp[p] = eps[p] + scissor_p
        # beyond the window, Delta(eps_p) at this geometry's eps_p
        for idx, delta, _ in self._curve_at(eps):
            eps_qp[idx] = eps[idx] + delta
        env_outside = self._env_static_outside(shift, len(eps))
        if env_outside is not None:
            eps_qp = eps_qp + env_outside
        if self.resolved.qp_partition is not None:
            # what the check at a walk's end reads at this geometry
            self.last_forward = {'coords': np.array(mol.atom_coords()),
                                 'eps': eps, 'eps_qp': eps_qp, 'shift': shift}
        self.conventions.anchor(mol)
        return SharedForward(mol, mf, auxmol, crd, x_mo, d, eps, eps_qp, w_aux,
                             mu, d_bare, shift, screening, qp_tape, xc, split,
                             qp_h, resolved)

    def _anchored_first(self, mol, at_reference):
        """Run `at_reference`, an evaluation at (mol0, mf0), before the first
        one at `mol` when that is another geometry: what the first forward
        freezes -- the set, the scissor, the quadrature, the seeds -- is the
        reference's, never that of where a walk happens to start (a nudged
        start is 0.02 Bohr away)."""
        if self.conventions.anchored or np.array_equal(
                np.asarray(mol.atom_coords()),
                np.asarray(self.mol0.atom_coords())):
            return
        at_reference()

    def _tracked(self, mol, mf):
        """(root, Omega, pieces) of the followed state at `mol` on `mf`, off
        the evaluation held there (`_forward`); a wider re-solve
        (`tracked_root`) replaces what is held."""
        om, pieces = self._forward(mol, mf)
        root, om_t, pieces_t = self.tracked_root(mol, mf, om, pieces)
        held = self._held(mol, mf)
        if pieces_t is not pieces and held is not None:
            held.om, held.pieces = om_t, pieces_t
        return root, om_t, pieces_t

    def _held(self, mol, mf):
        """The `Evaluation` held at `mol` on `mf`, or None."""
        held = self._evaluations.get(mol)
        return held if held is not None and held.mf is mf else None

    def _matched_at(self, mol, mf, followed):
        """Hold `followed`, the state this chain matched at `mol` on `mf`,
        in that geometry's `Evaluation` until `commit` adopts it."""
        held = self._held(mol, mf)
        if held is None:
            held = self._evaluations.put(mol, Evaluation(mf))
        held.followed[self._owner] = followed

    def tracked_state(self, mol, mf, om, pieces):
        """Which root of THIS geometry is the state being followed, among the
        roots `om` holds; `tracked_root` without the wider re-solve."""
        return self.tracked_root(mol, mf, om, pieces, widen=False)[0]

    def tracked_root(self, mol, mf, om, pieces, widen=True):
        """(root, Omega, pieces): the state being followed at THIS geometry,
        and the solve it was found in.

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
        xn = pieces[10]
        if self.track is None:
            return self.state, om, pieces
        here = (mol, np.asarray(mf.mo_coeff, float))
        if self._followed is None:
            self._followed = here + (xn[:, [self.state]].copy(),
                                     pieces[11][:, [self.state]].copy())
            self._anchor = self._followed
            self._matched_at(mol, mf, self._followed)
            self.follow_log.append(dict(index=self.state, weight=1.0,
                                        margin=1.0, anchor=1.0,
                                        identity=ORBITAL_IDENTITY))
            self.follow_coords.append(np.array(mol.atom_coords(), float))
            return self.state, om, pieces
        index, weight, margin = self._match_followed(here, pieces)
        solved = xn.shape[1]
        if weight < ROOT_FOLLOW_WEIGHT_MIN and widen and solved < xn.shape[0]:
            om, pieces = self._wider_solve(pieces, solved)
            index, weight, margin = self._match_followed(here, pieces)
        xn, yn = pieces[10], pieces[11]
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
        ta = body_overlap(self._anchor[0], self._anchor[1], mol, here[1])
        anchor = float(np.abs(relative_overlap(state_overlap(
            ta, self.nocc, self._anchor[2], self._anchor[3],
            xn[:, [index]], yn[:, [index]]))[1, 1]))
        self._matched_at(mol, mf, here + (xn[:, [index]].copy(),
                                          yn[:, [index]].copy()))
        step = dict(index=int(index), weight=weight, margin=margin,
                    anchor=anchor, omega=float(om[index]),
                    identity=ORBITAL_IDENTITY)
        if xn.shape[1] > solved:
            step['widened_to'] = int(xn.shape[1])
        if mixed:
            step['mixed'] = True
        self.follow_log.append(step)
        self.follow_coords.append(np.array(mol.atom_coords(), float))
        return int(index), om, pieces

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
        held = self._evaluations.get(mol)
        following = self.track is not None and self._followed is not None
        matched = None if held is None else held.followed.get(self._owner)
        if following and matched is None and not np.array_equal(
                np.asarray(self._followed[0].atom_coords()),
                np.asarray(mol.atom_coords())):
            raise RuntimeError(
                'commit: no state matched at this geometry is held (it was '
                'not evaluated here with the state followed, or is older '
                f'than the last {self._evaluations.size} geometries); the '
                'followed state and the keys stay where they were last '
                'committed')
        if held is not None:
            self.conventions.commit(mol, held.mf)
        holder = self.__dict__.get('davidson_warm_start')
        if holder is not None:
            # the next geometry's Davidson starts from the roots solved here
            holder.accept(mol)
        if not following or matched is None:
            return False
        self._followed = matched
        return True

    def _match_followed(self, here, pieces):
        """(index, weight, margin) of the last committed state among the
        roots of `pieces` (`follow_state`), its orbitals matched by the one
        orbital identity the conventions use (`body_overlap`), in which the
        core rides with its atoms, relative to the overlap of the two
        reference determinants (`relative_overlap`), which it keeps near 1."""
        mol0, mo0, x0, y0 = self._followed
        t = body_overlap(mol0, mo0, here[0], here[1])
        return follow_state(relative_overlap(state_overlap(
            t, self.nocc, x0, y0, pieces[10], pieces[11])), 0)

    def _wider_solve(self, pieces, solved):
        """(Omega, pieces) of the same forward's Casida step solved for
        ROOT_FOLLOW_RESOLVE_FACTOR times `solved` roots, at most all."""
        x_mo, d, eps_qp, w_aux = pieces[4], pieces[5], pieces[7], pieces[8]
        nroots, self.nroots = self.nroots, min(
            ROOT_FOLLOW_RESOLVE_FACTOR * solved, pieces[10].shape[0])
        try:
            with self.phase('t_casida'):
                om, xn, yn, cache = self._casida(
                    x_mo, d, eps_qp, w_aux,
                    **coupled(getattr(pieces, 'qp_h', None)))
        finally:
            self.nroots = nroots
        return om, ForwardPieces(
            tuple(pieces[:9]) + (cache, xn, yn) + tuple(pieces[12:]),
            getattr(pieces, 'xc_correction', None),
            getattr(pieces, 'split', None), getattr(pieces, 'qp_h', None),
            getattr(pieces, 'resolved', None))

    def _casida_args(self, pieces):
        """(X_mo, D, eps_qp, W_aux, nocc, cache, Xn, Yn) out of `_forward`.

        The positional tail every Casida-level adjoint takes, named once so a
        seed cannot be wired to the wrong member of a seventeen-long tuple.
        """
        (_, _, _, _, x_mo, d, _, eps_qp, w_aux, cache, xn, yn,
         _, _, _, _, _) = pieces
        return x_mo, d, eps_qp, w_aux, self.nocc, cache, xn, yn

    def _casida_seeds(self, pieces, n, m=None):
        """(eps_qp_bar, X_bar, D_bar, W_aux_bar) of Omega_n, or of the
        interstate element <m| dH |n>, by the realization `bse_adjoint` names.

        The explicit route fills the forward pass's cache in place the first
        time it is asked, so every root reversed off one held forward pass
        reads the same blocks and builds them once.
        """
        x_mo, d, eps_qp, w_aux, nocc, cache, xn, yn = self._casida_args(pieces)
        block = self._adjoint_rows(x_mo)
        if self.bse_adjoint == 'grid' and block is not None:
            # the grid adjoint's own rows, moved into the adjoint's tiles:
            # X_bar and D_bar are never whole on any rank
            comm = current_comm()
            e_bar, x_rows, d_rows, w_bar = isdf_bse_backward_rows(
                n, x_mo, d, eps_qp, w_aux, nocc, xn, yn, spin=self.spin,
                bse_tda=self.bse_tda, bra=m, comm=comm)
            npts = (x_mo.npts if isinstance(x_mo, SlicedFactors)
                    else len(x_mo))
            return (e_bar,
                    GridTileRows.from_blocks(x_rows, npts, block, comm),
                    GridTileRows.from_blocks(d_rows, npts, block, comm),
                    w_bar)
        if self.bse_adjoint == 'grid':
            kw = dict(spin=self.spin, bse_tda=self.bse_tda)
            if m is None:
                return isdf_bse_backward(n, x_mo, d, eps_qp, w_aux, nocc, xn,
                                         yn, **kw)
            return isdf_interstate_backward(m, n, x_mo, d, eps_qp, w_aux, nocc,
                                            xn, yn, **kw)
        if not cache:
            cache.update(bse_cache(x_mo, d, eps_qp, w_aux, nocc,
                                   spin=self.spin, bse_tda=self.bse_tda))
        if m is None:
            return bse_backward(n, x_mo, d, eps_qp, w_aux, nocc, cache, xn, yn,
                                **self._tile_kw())
        return interstate_backward(m, n, x_mo, d, eps_qp, w_aux, nocc, cache,
                                   xn, yn, **self._tile_kw())


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
