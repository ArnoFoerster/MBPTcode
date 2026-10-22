"""The quasiparticle set of the excited-state chain.

The frozen contour grid, the quasiparticle branches (roots followed), the groups and
subspaces of near-degenerate orbitals, the quasiparticle Hamiltonian and its
backward, and the scissor or curve the orbitals outside the window carry.
`ExcitedStateChain` inherits these methods; they read the chain's settings
and frozen conventions through `self`.
"""
import warnings
from dataclasses import dataclass

import numpy as np

from src.Base.sliced_factors import SlicedFactors
from src.Base.step_failed import StepFailed
from src.Base.constants import (QP_BLOCK_CORRECTION_TOL, QP_HAMILTONIAN_WINDOW,
                                CD_NFREQ, CD_NFREQ_SOP)
from src.Base.utils.mpi_grid import lockstep
from src.SingleReference.GW.contour_deformation import (cd_frequency_grid,
                                                        cd_grid_range)
from src.SingleReference.GW.qp_selection import (ac_quasiparticle_shifts,
                                                 degenerate_blocks,
                                                 merged_blocks,
                                                 true_degenerate_blocks,
                                                 window_edge_holes)
from src.SingleReference.GW.qp_states import (calibrate_scissor,
                                              calibrate_scissor_tiers,
                                              curve_sides, fit_outside_curve,
                                              frontier_qp_states,
                                              is_quasiparticle_root,
                                              nearest_probe)
from src.gradients.isdf_derivatives import qp_xc_correction_matrix
from src.SingleReference.GW.reaction_field import (offdiagonal_reaction_field,
                                                   pair_densities)
from src.SingleReference.GW.sum_over_poles import (pole_denominators,
                                                   sop_from_wc,
                                                   subspace_hamiltonian,
                                                   subspace_pole_strengths,
                                                   subspace_residues)
from src.gradients.sum_over_poles_adjoint import (
    subspace_hamiltonian_backward, subspace_residues_backward)
from src.gradients.qp_space_time import (carried_analytically,
                                         pair_contractions, qp_set_gradient,
                                         static_term)
from src.Solvers.qp_equation import quasiparticle_weight


@dataclass(eq=False)
class QPGroup:
    """One group's quasiparticle Hamiltonian at one geometry
    (`subspace_hamiltonian`): `orbitals` its members, sorted, `positions`
    theirs in the set, `h` the solved H, `amp` and `d` its residues and their
    positions on the group's common `poles`, `pairs` the (a, b), a <= b,
    whose contractions built `amp`, `z` the pole strength of each eigenvector
    and `iterations` the Newton steps taken. In a continuum `h` is solved
    without the reaction field and `env` is its Eq. (18) block (diagonal and
    `offdiagonal_reaction_field`), added after the solve like every G0W0
    root's; `hamiltonian` is the two together, zero `env` in the gas phase."""

    orbitals: tuple
    positions: tuple
    h: np.ndarray
    amp: np.ndarray
    d: np.ndarray
    poles: np.ndarray
    pairs: list
    z: np.ndarray
    iterations: int
    env: np.ndarray

    @property
    def hamiltonian(self):
        """H with the reaction field: what the BSE reads."""
        return self.h + self.env


@dataclass(eq=False)
class QPHamiltonian:
    """The quasiparticle Hamiltonian of the set's frozen groups at one
    geometry: inside each group H solved to self-consistency on the group's
    common poles (`QPGroup`), invariant to how the group's orbitals rotate
    into each other. Its diagonal is the members' eps^QP; `h` is (norb, norb)
    with H's off-diagonal elements inside the groups and zero elsewhere;
    `pairs` lists (i, j, p, q), i, j positions into the set and p < q
    orbitals of one group.
    """

    h: np.ndarray
    pairs: list
    groups: list

    def one_body(self, nocc):
        """((o, H_oo), (v, H_vv)) for the Casida action (`with_one_body`):
        the coupled occupied and virtual orbitals, as pair-space positions."""
        members = np.flatnonzero(np.abs(self.h).sum(axis=1) > 0.0)
        o = members[members < nocc]
        v = members[members >= nocc]
        return ((o, self.h[np.ix_(o, o)]),
                (v - nocc, self.h[np.ix_(v, v)]))


class QuasiparticleRootLost(StepFailed):
    """A quasiparticle-Hamiltonian group has nothing to stand on at a
    displaced geometry: it lost members before its poles or seed were
    frozen, or its Newton ended on a non-finite H. A pole strength outside
    (0, 1] or a Newton that stopped short of QP_SUBSPACE_TOL on a finite H is
    not one: warned about and logged (`qp_hamiltonian_log`)."""


class QPSetMixin:
    """The frozen quasiparticle set and the scissor outside it."""

    # ------------------------------------------------------------------ setup
    def _build_cd_grid(self, nfreq_cd, eps):
        """The contour-deformation quadrature and the imaginary-time grid whose
        frequency axis it is, at `nfreq_cd` points.

        Production's `cd_frequency_grid` builds it, so the surface a gradient
        walks and the quadrature an energy route integrates on are one choice
        made in one place -- including the refusal when e_min_below_gap eats
        the whole gap, which no imaginary-time grid represents.

        ntau_gw defaults to CD_NTAU: a residue's cosh transform reaches below
        the gap, where the imaginary axis alone would be converged at fewer
        points (see CD_NTAU)."""
        self.conventions.nfreq_cd = int(nfreq_cd)
        # the range the tau axis is fitted on, recorded beside the grid it
        # built: an audit reads the range a surface was frozen at
        grid = self.conventions
        grid.e_min_gw, grid.e_max = cd_grid_range(eps, self.nocc,
                                                  self.e_min_below_gap)
        grid.nu, grid.wt, grid.gw_grid, grid.w0_cd = cd_frequency_grid(
            eps, self.nocc, ntau=self.ntau_gw, nfreq_cd=grid.nfreq_cd,
            e_min_below_gap=self.e_min_below_gap)

    def _fix_cd_grid(self, states, eps):
        """Fix the contour-deformation grid for the set, once: `CD_NFREQ_SOP`
        points when the pole model carries it whole, `CD_NFREQ` otherwise,
        never below an `nfreq_cd` the caller asked for (a refrozen chain does
        not step back).

        The grid is never resized against the roots. The Lorentzian a pole of
        G puts on the imaginary-frequency integrand at omega = eps_q -- a
        neighbouring orbital's, or the root's own -- is integrated in closed
        form (`contour_deformation.cd_integral_weights`), so a root beside an
        orbital energy asks nothing more of the quadrature than any other.
        A set the pole model carries whole has a Sigma analytic in omega, and
        its grid only feeds the fits, which converge at fewer points.
        """
        if self.conventions.cd_sized:
            return
        kw = self._qp_kw()
        analytic = carried_analytically(
            states, eps, self.nocc, kw['residue_route'], kw['scissor'],
            kw['w0'], kw['sop_poles'])
        nfreq = max(CD_NFREQ_SOP if analytic else CD_NFREQ,
                    self.conventions.nfreq_cd_floor)
        if self.conventions.nfreq_cd != nfreq:
            self._build_cd_grid(nfreq, eps)
        self.conventions.cd_sized = True

    @staticmethod
    def _qp_set(eps, nocc, window, tol):
        """The quasiparticle set: a SEQUENCE of orbital indices -- the set a
        `QPStates` declaration resolved to -- taken as given, else the
        frontier set of half-width `window` (or 'all') widened over
        degenerate blocks (`frontier_qp_states`)."""
        if np.ndim(window) == 1:
            return np.sort(np.asarray(window, int))
        return frontier_qp_states(eps, nocc, window, tol)

    def _log_qp_branches(self, view, route_out):
        """Warn about, and log in `qp_branch_log`, each followed root whose
        branch the largest weight has left at this geometry (`QPRoot.flipped`),
        once per orbital (by slot) and geometry: the walk keeps the branch."""
        weights, roots = route_out.get('weights', {}), route_out.get('roots', {})
        log = self.__dict__.setdefault('qp_branch_log', [])
        for p, branch in sorted(route_out.get('branches', {}).items()):
            if not branch.get('flipped'):
                continue
            largest = branch.get('largest')
            entry = {'orbital': view.slot(p), 'index': int(p),
                     'geometry': view.geometry,
                     'root': float(roots[p]), 'z': float(weights[p]),
                     'quasiparticle': quasiparticle_weight(weights[p]),
                     'followed': bool(branch.get('followed')),
                     'largest_root': None if largest is None
                     else float(largest[0]),
                     'largest_z': None if largest is None
                     else float(largest[1])}
            if any(e['orbital'] == entry['orbital']
                   and e['geometry'] == entry['geometry']
                   for e in log):
                continue
            log.append(entry)
            warnings.warn(
                f'quasiparticle {p} (slot {entry["orbital"]}): '
                + ('the followed branch has gone; the root of largest weight '
                   'is kept again, ' if not entry['followed'] else
                   'the root on the followed branch is kept, ')
                + f'{entry["root"]:.10f} Ha, Z = {entry["z"]:.4f}'
                + ('' if largest is None else
                   f'; the largest weight here is {entry["largest_z"]:.4f} '
                   f'at {entry["largest_root"]:.10f} Ha')
                + ('' if entry['quasiparticle'] else
                   '; no quasiparticle by weight')
                + f' (geometry {str(view.geometry)[:12]}). The walk goes on '
                  f'on this branch.', RuntimeWarning, stacklevel=3)

    def _qp_kw(self):
        # plain dicts: the kernels audit their inputs across ranks
        view = self.resolved
        kw = {'residue_route': self.residue_route,
              'follow': dict(view.qp_follow),
              'w0': dict(view.qp_seeds),
              'z0': dict(view.qp_seed_weights),
              'sop_poles': dict(view.sop_poles),
              'n_poles': self.n_poles,
              'sop_stride': self.sop_stride,
              'scissor': dict(view.scissor_map) or self.scissor}
        if self.tile_gb is not None:
            kw['tile_gb'] = self.tile_gb
        return kw

    def _freeze_qp_branch(self, route_out, eps, xc_correction, states):
        """Keep the root, its pole strength, its pole labels and the pole
        set each quasiparticle solve resolved to, once. True when the solve
        has to be repeated: a root was frozen here for the first time, or a
        calibrated shift was added and is read by the next solve.

        The first solve is the reference geometry's in every path that then
        displaces it, so `setdefault` freezes that one: a displaced geometry
        reports what it used, follows the branch of the last accepted
        geometry (the reference's until a step is accepted), evaluates the
        pole model on the reference poles and changes none of them. Every
        solve's roots are held for `commit` (`hold_qp_branch`), and a
        followed root the largest weight has left is warned about and logged
        once per orbital and geometry (`_log_qp_branches`): the walk goes on
        on its branch, whatever its weight.

        THE REFERENCE ROOT IS THE ONE ITS FOLLOWED BRANCH REACHES. The first
        solve selects by the largest Z over every bracket; every later one at
        this geometry solves the same brackets and keeps the root nearest the
        one it froze, between the same labelled poles: the same root, refined
        from the same bracket, bitwise. Repeating the freezing solve as a
        followed one makes every evaluation at the reference geometry the
        same arithmetic, forward and reverse.

        eps, xc_correction, states: what the solve read, xc_correction per
        state of `states` (or one scalar), so the calibrated shift is taken
        off the same numbers the 'scissor' route adds it back to.

        THE CALIBRATED SHIFT IS Sigma_c,pp(w_p) ALONE. The 'scissor' route
        returns eps_p + xc_p + s_p, xc_p = <p|Sigma_x - v_xc|p> the static
        term (Eq. (18) is added to the root after the solve, outside this
        equation), and the root solved here is w_p = eps_p + xc_p +
        Sigma_c,pp(w_p), so s_p = w_p - (eps_p + xc_p) puts the state on its
        own root at this geometry. w_p - eps_p carried xc_p twice: formaldehyde
        orbital 0 36.5 eV low on PBE0, 1.8 eV high on HF in toluene.
        """
        view = self.resolved
        roots = route_out.get('roots', {})
        labels = route_out.get('labels', {})
        froze = self.conventions.freeze_qp_branch(
            view, route_out.get('weights', {}), roots,
            route_out.get('sop_poles', {}), labels)
        self.conventions.hold_qp_branch(view, roots, labels)
        self._log_qp_branches(view, route_out)
        # the shifts are read on the pole-model route alone (`scissor_route`)
        if self.scissor != 'calibrate' or self.residue_route != 'sop':
            return froze
        # Tier the states the route ACTUALLY sent to the real axis. A second
        # reading of Eq. (27) would test the root where the route tested the
        # start, and disagree on exactly the marginal states.
        static = {int(p): static_term(xc_correction, si)
                  for si, p in enumerate(np.atleast_1d(states))}
        grew = False
        for p, route in self._routes_taken(route_out).items():
            if route in ('sop', 'scissor') or p in view.scissor_map:
                continue
            self.conventions.freeze_scissor(
                view, p, float(roots[p]) - float(eps[p] + static[p]))
            grew = True
        return grew or froze

    def _flag_z_outside(self, route_out, states):
        """Warn about, and at the reference geometry record, the roots of the
        set whose pole strength Z lies outside (0, 1] (`is_quasiparticle_root`).

        Such a root is KEPT: it is neither refused nor moved to the scissor
        (it can happen, rarely, and is often harmless; a
        warning, not a refusal -- the one exception to refusing every shortfall).
        A declaration names the orbitals before anything is solved, so Z is
        first known here; the reference geometry's flagged roots go to
        `qp_z_outside` with the orbital's Z, root and route, frozen with the
        rest of the conventions, and a displaced geometry warns on its own
        roots. The pole strengths are rank 0's on every rank (`lockstep`).
        Members of a quasiparticle-Hamiltonian group are judged on their
        group's eigenvectors (`_qp_hamiltonian`) instead.

        Z is read off a continuous self-energy: the Lorentzian a pole of G
        puts beside an orbital energy is integrated in closed form
        (`cd_integral_weights`), so the slope is the self-energy's own and a
        Z judged here is a verdict on the root, not on the quadrature.
        """
        z_of = route_out.get('z')
        if z_of is None:
            return
        z_of = lockstep(np.array(z_of, float))
        flagged = {int(p): float(z) for p, z in zip(states, z_of)
                   if not is_quasiparticle_root(z)}
        roots = route_out.get('roots', {})
        if self.conventions.qp_set_settled and self.resolved.qp_subspaces:
            grouped = {p for b in self.resolved.qp_subspaces[1] for p in b}
            flagged = {p: z for p, z in flagged.items() if p not in grouped}
        if not flagged:
            return
        where = ('a displaced geometry' if self.conventions.qp_set_settled
                 else 'the reference geometry')
        if not self.conventions.qp_set_settled:
            routes = self._routes_taken(route_out)
            for p, z in flagged.items():
                self.conventions.flag_z_outside(self.resolved, p, {
                    'reason': 'pole strength Z outside (0, 1]', 'z': z,
                    'root': float(roots[p]), 'route': routes.get(p)})
        warnings.warn(
            f'quasiparticle roots with a pole strength outside (0, 1] at '
            f'{where}, orbital: Z {flagged}, roots (Ha) '
            f'{ {p: roots.get(p) for p in flagged} }. They are kept as '
            f'solved.', RuntimeWarning, stacklevel=3)

    @staticmethod
    def _routes_taken(route_out):
        """{orbital: route} from either route_out shape.

        A set solve reports 'routes' per orbital; a single-orbital solve
        reports one 'residue_route' and one root.
        """
        routes = route_out.get('routes')
        if routes is not None:
            return {int(p): r for p, r in routes.items()}
        route = route_out.get('residue_route')
        return {int(p): route for p in route_out.get('roots', {})}

    def _frozen_qp_blocks(self, eps, ws):
        """The set's truly degenerate blocks, positions into `qp_set`: two or
        more orbitals closer than `degeneracy_tol` whose own corrections
        ws_q - eps_q agree within the bar QP_BLOCK_CORRECTION_TOL.

        Decided on the eigenvalues and roots of the first forward of a set,
        which is the reference geometry's, and frozen with it: a block that
        formed or broke per geometry would switch an orbital between its own
        correction and a shared one along the walk.
        """
        key = tuple(int(p) for p in self.resolved.qp_set)
        blocks = self.resolved.qp_blocks
        if blocks is None or blocks[0] != key:
            pos = {p: i for i, p in enumerate(key)}
            own = {p: float(ws[i] - eps[p]) for i, p in enumerate(key)}
            found, record = true_degenerate_blocks(
                eps, own, key, self.degeneracy_tol, QP_BLOCK_CORRECTION_TOL)
            if self.qp_hamiltonian:
                # the quasiparticle Hamiltonian couples degenerate members
                for b in record['kept']:
                    b['why'] = 'qp_hamiltonian: coupled in its subspace'
                record = {'kept': [], 'rejected': (record['rejected']
                                                   + record['kept'])}
                found = ()
            self.conventions.freeze_blocks(
                self.resolved, key,
                tuple(tuple(pos[p] for p in b) for b in found),
                dict(record, energy_tol_Ha=float(self.degeneracy_tol),
                     correction_tol_Ha=float(QP_BLOCK_CORRECTION_TOL)))
        return self.resolved.qp_blocks[1]

    def _frozen_qp_subspaces(self, eps, routes):
        """The set's subspaces of two or more orbitals the quasiparticle
        Hamiltonian couples, as orbital tuples: neighbours on one side of the
        gap closer than QP_HAMILTONIAN_WINDOW, joined with the subspaces
        carried from the surface this one was refrozen from.

        Decided at the first forward of a set, the reference geometry's, and
        frozen with it. Generous on purpose: inside a subspace H is invariant
        to how its orbitals rotate into each other, and a pair that does not
        mix keeps its own corrections to second order in Sigma_pq / gap.

        Sigma^c_pq exists only on the pole model, so a member whose root took
        another route at this forward (`routes`, {orbital: route}) is dropped
        from its subspace and keeps its own diagonal correction; a subspace
        left with one member is none. Pole-model admission is decided afresh
        at a refrozen geometry, so a carried member can lose it there.
        `qp_subspace_record` lists every drop with the subspace it left and
        the route it took.
        """
        key = tuple(int(p) for p in self.resolved.qp_set)
        subspaces = self.resolved.qp_subspaces
        if subspaces is None or subspaces[0] != key:
            groups = []
            for side in ([p for p in key if p < self.nocc],
                         [p for p in key if p >= self.nocc]):
                groups += degenerate_blocks(eps, side, QP_HAMILTONIAN_WINDOW)
            before = self.resolved.carried_subspaces
            carried = {int(p) for b in before for p in b}
            # a carried member the explicit set no longer holds leaves its
            # group (`merged_blocks` joins members of the set only)
            dropped = [{'orbital': int(p), 'subspace': [int(q) for q in b],
                        'route': None, 'carried': True,
                        'reason': 'not in the explicit set',
                        'remaining': [int(q) for q in b if q in set(key)]}
                       for b in before for p in b if int(p) not in set(key)]
            kept = []
            for b in merged_blocks(key, groups, before):
                if len(b) < 2:
                    continue
                pole = tuple(p for p in b if routes.get(p) == 'sop')
                dropped += [{'orbital': p, 'subspace': list(b),
                             'route': routes.get(p), 'carried': p in carried,
                             'reason': 'off the pole model',
                             'remaining': list(pole) if len(pole) > 1 else []}
                            for p in b if p not in pole]
                if len(pole) > 1:
                    kept.append(pole)
            if dropped:
                warnings.warn(
                    f'quasiparticle Hamiltonian: orbitals '
                    f'{[d["orbital"] for d in dropped]} leave their '
                    f'subspaces {[d["subspace"] for d in dropped]} at this '
                    f'geometry ({[d["reason"] for d in dropped]}; routes '
                    f'{[d["route"] for d in dropped]})',
                    RuntimeWarning, stacklevel=3)
            self.conventions.freeze_subspaces(self.resolved, key, tuple(kept), {
                'subspaces': [list(b) for b in kept],
                'carried': [list(b) for b in self.resolved.carried_subspaces],
                'dropped': dropped})
        return self.resolved.qp_subspaces[1]

    def _qp_hamiltonian(self, mf, eps, ws, route_out, continuum=None):
        """The `QPHamiltonian` of this forward, or None where the set has no
        group of two: each frozen group's H solved to self-consistency
        (`subspace_hamiltonian`) on the group's common poles.

        H = e0 + Sigma^c[H] with e0 = diag(eps_p + <p|Sigma_x - v_xc|p>)
        plus the off-diagonal <p|Sigma_x - v_xc|q>, and Sigma^c
        the pole model of every element B_p^T [W - 1] B_q of the group,
        diagonal ones off the quasiparticle tape and off-diagonal ones on the
        same screening (`pair_contractions`). ONE set of poles per group,
        fitted at the reference geometry to the group's trace sum_p wc_pp and
        frozen with it, makes the amplitudes one linear map of the operator;
        with Sigma^c taken at H's own eigenvalues the group's H is the same
        operator whichever way its orbitals rotate into each other. The
        group's Newton starts from its frozen seed, the reference H carried
        into this geometry's orbitals of the group (`group_seed`).
        continuum: (X_mo, D, D_bare, screening, shift) of Eq. (18); None in
        the gas phase. Its block Sigma^env (shift_p on the diagonal,
        `offdiagonal_reaction_field` off it) is added to the solved H whole,
        as on every G0W0 root (`add_reaction_field`); inside e0 the pole
        model would renormalize it. `ws`, the diagonal roots, carry it, so
        the seed takes it off.

        In a continuum the reaction-field part is not an operator: Eq. (18)
        is quartic in the orbital, and its off-diagonal partner is first
        order in a rotation, so there H is invariant to first order only.
        """
        routes = self._routes_taken(route_out)
        frozen = self._frozen_qp_subspaces(eps, routes)
        self.qp_subspace_drops = {
            'coords': np.array(mf.mol.atom_coords(), float), 'dropped': []}
        if not frozen:
            return None
        pos = {int(p): i for i, p in enumerate(self.resolved.qp_set)}
        # A frozen member off the pole model at this geometry leaves its
        # group here, as at the freeze (`_frozen_qp_subspaces`), and keeps its
        # own diagonal root; the rest keep the group's frozen poles and seed.
        off = {p for b in frozen for p in b if routes.get(p) != 'sop'}
        kept = [(f, tuple(p for p in f if p not in off)) for f in frozen]
        kept = [(f, b) for f, b in kept if len(b) > 1]
        if off:
            self._dropped_here(frozen, kept, off, routes, route_out, pos)
        if not kept:
            return None
        subspaces = [b for _, b in kept]
        pairs = [(pos[p], pos[q], p, q) for b in subspaces
                 for a, p in enumerate(b) for q in b[a + 1:]]
        tape = route_out['tape']
        wc_off = dict(zip([(p, q) for _, _, p, q in pairs], pair_contractions(
            tape, self.conventions.gw_grid, [(i, j) for i, j, _, _ in pairs],
            self.tile_gb)))
        members = sorted({p for b in subspaces for p in b})
        static = np.zeros((len(eps), len(eps)))
        if self.is_ks:
            static[np.ix_(members, members)] = qp_xc_correction_matrix(
                mf, members)
        # Sigma^env over the groups, added to each solved H
        env = np.zeros((len(eps), len(eps)))
        if continuum is not None:
            x_mo, d, d_bare, screening, shift = continuum
            orbitals = [(p, q) for _, _, p, q in pairs]
            offdiagonal = offdiagonal_reaction_field(
                screening, [pair_densities(x_mo, f, orbitals)
                            for f in (d, d_bare)], orbitals, self.nocc)
            for (p, q), e in zip(orbitals, offdiagonal):
                env[p, q] += e
                env[q, p] += e
            env[members, members] = np.asarray(shift, float)[members]
        xc = route_out['xc_correction']
        h = np.zeros((len(eps), len(eps)))
        groups = []
        for f, b in kept:
            idx = list(b)
            n = len(idx)
            wc = {(i, i): tape.wcs[pos[p]] for i, p in enumerate(idx)}
            wc.update({(i, j): wc_off[(idx[i], idx[j])]
                       for i in range(n) for j in range(i + 1, n)})
            poles = self._group_poles(b, wc, eps, of=f)
            amp = subspace_residues(wc, poles, self.conventions.nu)
            d_x = pole_denominators(poles, eps, self.nocc).ravel()
            e0 = static[np.ix_(idx, idx)].copy()
            np.fill_diagonal(e0, [float(eps[p]) + static_term(xc, pos[p])
                                  for p in idx])
            env_g = env[np.ix_(idx, idx)]
            hg, it = self._group_newton(
                b, e0, amp, d_x,
                [ws[pos[p]] - env_g[i, i] for i, p in enumerate(idx)], of=f)
            z = subspace_pole_strengths(hg, amp, d_x)
            if not all(is_quasiparticle_root(zk) for zk in z):
                # kept as solved, as a diagonal root of Z outside (0, 1] is
                # (refuse inputs, warn numerics)
                self._log_qp_hamiltonian(
                    'z_outside', b, f, z=[float(v) for v in z], newton_steps=it)
            h[np.ix_(idx, idx)] = hg + env_g
            groups.append(QPGroup(tuple(idx), tuple(pos[p] for p in idx), hg,
                                  amp, d_x, poles, sorted(wc), z, it, env_g))
        np.fill_diagonal(h, 0.0)
        # what each group was at the geometry the groups were decided at
        self.conventions.record_groups(self.resolved, [
            {'orbitals': list(g.orbitals), 'n_poles': len(g.poles),
             'z': [float(v) for v in g.z], 'newton_steps': int(g.iterations)}
            for g in groups])
        return QPHamiltonian(h, pairs, groups)

    def _dropped_here(self, frozen, kept, off, routes, route_out, pos):
        """Record, warn about and Z-check the members leaving their frozen
        groups at this forward: those off the pole model and any left alone.
        Each carries its own diagonal root, so the surface is the coupled one
        on one side of the drop and the diagonal one on the other, as at a
        drop at the freeze: a numerical event, warned about and flagged in the
        record (`qp_subspace_drops`, `qp_bookkeeping`'s 'dropped_here'), the
        run carries on (refuse inputs, warn
        numerics); a diagonal root with Z outside (0, 1] is kept and flagged
        ('z_outside' on its entry) as well."""
        remaining = {p: b for _, b in kept for p in b}
        z_of = route_out.get('z')
        z_of = None if z_of is None else lockstep(np.array(z_of, float))
        dropped = []
        for f in frozen:
            for p in f:
                if p in remaining:
                    continue
                z = None if z_of is None else float(z_of[pos[p]])
                dropped.append({'orbital': int(p), 'subspace': list(f),
                                'route': routes.get(p), 'z': z,
                                'remaining': [int(q) for q in f
                                              if q in remaining]})
        for d in dropped:
            d['z_outside'] = (d['z'] is not None
                              and not is_quasiparticle_root(d['z']))
        bad = [(d['orbital'], d['z']) for d in dropped if d['z_outside']]
        self.qp_subspace_drops['dropped'] = dropped
        warnings.warn(
            f'quasiparticle Hamiltonian: orbitals '
            f'{[d["orbital"] for d in dropped]} leave their frozen subspaces '
            f'{[d["subspace"] for d in dropped]} at this geometry (routes '
            f'{[d["route"] for d in dropped]}) and carry their own roots'
            + ('' if not bad else
               f'; kept with pole strength outside (0, 1]: {bad}'),
            RuntimeWarning, stacklevel=4)

    def _group_poles(self, group, wc, eps, of=None):
        """The group's common poles: frozen, or fitted here, at the reference
        geometry, to the trace of its diagonal contractions, the same at
        every rank. of: the frozen group `group` is what remains of here,
        whose poles it keeps."""
        view = self.resolved
        if of is not None and tuple(of) != tuple(group):
            poles = view.group_poles(of)
            if poles is None:
                raise QuasiparticleRootLost(
                    f'group {list(of)} lost members before its poles were '
                    f'frozen')
            return poles
        poles = view.group_poles(group)
        if poles is not None:
            return poles
        trace = sum(wc[(i, i)] for i in range(len(group)))
        kw = {} if self.sop_stride is None else {'stride': int(self.sop_stride)}
        poles = lockstep(sop_from_wc(trace, self.conventions.nu, eps,
                                     self.nocc, n_poles=self.n_poles,
                                     **kw)[0])
        self.conventions.freeze_group_poles(view, group, poles)
        return poles

    def _group_newton(self, group, e0, amp, d, roots, of=None):
        """(H, Newton steps) of one group from its frozen seed. The first
        solve, at the reference geometry, starts from the members' diagonal
        roots, freezes its H as the seed and is repeated from it, so every
        evaluation there is the same arithmetic. of: the frozen group `group`
        is what remains of here, whose seed's block of `group` it starts
        from."""
        view = self.resolved
        if of is not None and tuple(of) != tuple(group):
            whole = view.group_seed(of)
            if whole is None:
                raise QuasiparticleRootLost(
                    f'group {list(of)} lost members before its seed was '
                    f'frozen')
            order = sorted(of)
            k = [order.index(p) for p in group]
            h, it, ok = subspace_hamiltonian(e0, amp, d, whole[np.ix_(k, k)])
            if not ok:
                self._unconverged_group(group, of, e0, amp, d, h, it)
            return h, it
        seed = view.group_seed(group)
        fresh = seed is None
        if fresh:
            seed = e0 - np.diag(np.diag(e0)) + np.diag(roots)
        h, it, ok = subspace_hamiltonian(e0, amp, d, seed)
        if fresh and not ok:
            # the reference H frozen as it ended, so the seed is still the
            # reference geometry's convention
            self._unconverged_group(group, group, e0, amp, d, h, it)
        if fresh:
            self.conventions.freeze_group_seed(view, group, h)
            h, it, ok = subspace_hamiltonian(e0, amp, d, h)
        if not ok:
            self._unconverged_group(group, group, e0, amp, d, h, it)
        return h, it

    def _unconverged_group(self, group, of, e0, amp, d, h, it):
        """A group's Newton that ended above QP_SUBSPACE_TOL: on a finite H a
        numerical event, warned about and logged (`qp_hamiltonian_log`, with
        the size of one more Newton step), the run carrying on with that H
        (refuse inputs, warn numerics); on a
        non-finite one there is no H to carry on with and the step is
        refused (`QuasiparticleRootLost`)."""
        if not np.all(np.isfinite(h)):
            raise QuasiparticleRootLost(
                f'the quasiparticle Hamiltonian of group {list(group)} ended '
                f'non-finite after {it} Newton steps')
        step = float(np.abs(subspace_hamiltonian(e0, amp, d, h, max_iter=1)[0]
                            - h).max())
        self._log_qp_hamiltonian('not_self_consistent', group, of,
                                 newton_steps=it, last_step=step)

    def _log_qp_hamiltonian(self, kind, group, of, **what):
        """Warn about, and log in `qp_hamiltonian_log`, a numerical event of
        one quasiparticle-Hamiltonian group at this geometry, once per kind,
        group and geometry: 'z_outside' (eigenvectors of pole strength
        outside (0, 1], kept as solved) or 'not_self_consistent' (its Newton
        stopped above QP_SUBSPACE_TOL; `last_step`, the largest element of
        one more Newton step, says by how much). `of`: the frozen group
        `group` is what remains of."""
        geometry = self.resolved.geometry
        entry = dict({'kind': kind, 'orbitals': [int(p) for p in group],
                      'of': [int(p) for p in of], 'geometry': geometry},
                     **what)
        log = self.__dict__.setdefault('qp_hamiltonian_log', [])
        if any(e['kind'] == kind and e['orbitals'] == entry['orbitals']
               and e['geometry'] == geometry for e in log):
            return
        log.append(entry)
        detail = (f'eigenvectors of pole strength '
                  f'{np.round(what["z"], 4).tolist()} outside (0, 1], kept'
                  if kind == 'z_outside' else
                  f'no self-consistency after {what["newton_steps"]} Newton '
                  f'steps (the next step {what["last_step"]:.2e} Ha), '
                  f'carried on as it ended')
        warnings.warn(
            f'the quasiparticle Hamiltonian of group {entry["orbitals"]}'
            + ('' if entry['of'] == entry['orbitals'] else
               f' (left of {entry["of"]})')
            + f': {detail} (geometry {str(geometry)[:12]})',
            RuntimeWarning, stacklevel=3)

    def _one_body_seed(self, pieces, n, m=None):
        """(norb, norb) adjoint of Omega_n (of <m|dH|n>) on the quasiparticle
        Hamiltonian's off-diagonal elements, symmetric, or None without one:
        dOmega = sum C^A dA with C^A = x x^T + y y^T, through
        A_ia,jb += delta_ij H_ab - delta_ab H_ij."""
        qp_h = getattr(pieces, 'qp_h', None)
        if qp_h is None:
            return None
        no = self.nocc
        nv = len(pieces[6]) - no
        out = np.zeros_like(qp_h.h)
        for bra, ket, wt in (((n, n, 1.0),) if m is None else
                             ((m, n, 0.5), (n, m, 0.5))):
            xm = pieces[10][:, bra].reshape(no, nv)
            ym = pieces[11][:, bra].reshape(no, nv)
            xk = pieces[10][:, ket].reshape(no, nv)
            yk = pieces[11][:, ket].reshape(no, nv)
            out[no:, no:] += wt * (xm.T @ xk + ym.T @ yk)
            out[:no, :no] -= wt * (xm @ xk.T + ym @ yk.T)
        # the coupled pairs alone: an element zero by symmetry here still
        # has a derivative along a displacement that breaks it
        coupled = np.zeros(out.shape, bool)
        for _, _, p, q in qp_h.pairs:
            coupled[p, q] = coupled[q, p] = True
        return np.where(coupled, 0.5 * (out + out.T), 0.0)

    def _qp_hamiltonian_backward(self, qp_h, h_bar, eqp_bar, eps):
        """(positions of the groups' members in the set, eps_bar, pair
        adjoints for `qp_set_gradient`, Lambda) of the BSE's adjoints on the
        groups: eqp_bar on each H's diagonal and h_bar (`_one_body_seed`, or
        None) on its off-diagonal elements.

        Through each group's fixed point (`subspace_hamiltonian_backward`):
        Lambda = (1 - J^T)^-1 H_bar is the adjoint on e0 -- eps on its
        diagonal and the static terms, (norb, norb), zero outside the groups,
        whose diagonal the static chain takes in place of Z times the seed;
        the reaction field, added after the solve, takes H_bar itself --
        and the residues' adjoint goes back through A = F^+ wc onto every
        contraction of the group, diagonal ones included, and through the
        poles' positions onto eps.
        """
        eps_bar = np.zeros(len(eps))
        lam = np.zeros((len(eps), len(eps)))
        pairs, members = [], []
        for g in qp_h.groups:
            idx = list(g.orbitals)
            seed = (np.zeros((len(idx), len(idx))) if h_bar is None
                    else np.array(h_bar[np.ix_(idx, idx)], float))
            np.fill_diagonal(seed, eqp_bar[idx])
            e0_bar, amp_bar, d_bar = subspace_hamiltonian_backward(
                g.h, g.amp, g.d, seed)
            lam[np.ix_(idx, idx)] += e0_bar
            eps_bar[idx] += np.diag(e0_bar)
            eps_bar += d_bar.reshape(len(eps), -1).sum(axis=1)
            for (a, b), w in subspace_residues_backward(
                    amp_bar, g.poles, self.conventions.nu, g.pairs).items():
                pairs.append((g.positions[a], g.positions[b], w))
            members += list(g.positions)
        return members, eps_bar, pairs, lam

    def _block_shared(self, eps, ws):
        """eps^QP over the set with ONE correction per truly degenerate block.

        Each member keeps its own eps_p and takes the mean of the block's
        eps^QP_q - eps_q. Symmetric in the members, so two orbitals that swap
        character across an avoided crossing swap nothing on the BSE diagonal;
        diagonal G0W0 gives each index its own Sigma_pp, which steps there.
        """
        blocks = self._frozen_qp_blocks(eps, ws)
        if not blocks:
            return ws
        states = np.asarray(self.resolved.qp_set, int)
        out = np.array(ws, float)
        for b in blocks:
            b = list(b)
            out[b] = eps[states[b]] + np.mean(out[b] - eps[states[b]])
        return out

    def _block_seeds(self, eqp_bar):
        """(seed on the set's roots, seed on eps) of an adjoint on eps^QP.

        eps^QP_p = eps_p + mean_q (w_q - eps_q) over p's block, so every root
        of a block takes the block's mean seed, and eps_p takes its own seed
        less that mean; outside a block both reduce to (eqp_bar_p, 0).
        """
        states = np.asarray(self.resolved.qp_set, int)
        seed = np.array(eqp_bar[states], float)
        direct = np.zeros(len(eqp_bar))
        blocks = self.resolved.qp_blocks
        for b in (blocks[1] if blocks else ()):
            b = list(b)
            mean = np.mean(seed[b])
            direct[states[b]] = seed[b] - mean
            seed[b] = mean
        return seed, direct

    def _qp_set_solve(self, x_mo, d_sigma, eps, mu, xc_correction, weights,
                      states=None, tape=None, rows_block=None, pairs=()):
        """((eps^QP values, adjoints), route_out) for a quasiparticle set, on
        the contour-deformation grid fixed for it before the first pass
        (`_fix_cd_grid`), never resized against the roots.

        tape: an earlier solve's `QPSetTape` on the same factors, and each
        repeat reads the one before it, so proj(tau) and the slices are
        swept once however often the solve repeats (`QPSetTape.reads`).
        rows_block: the adjoints in grid tiles over the ranks
        (`qp_set_gradient`).

        A solve of the whole set (states=None) settles it; a root whose pole
        strength lies outside (0, 1] is kept, warned about and, at the
        reference geometry, recorded (`_flag_z_outside`).
        route_out['xc_correction'] is the correction the returned roots were
        solved with.
        """
        whole_set = states is None
        states = self.resolved.qp_set if whole_set else states
        self._fix_cd_grid(states, eps)
        while True:
            route_out = {}
            grid = self.conventions
            out = qp_set_gradient(
                x_mo, d_sigma, eps, self.nocc, grid.gw_grid, grid.nu,
                grid.wt, states, weights, mu=mu,
                xc_correction=xc_correction, route_out=route_out,
                tape=tape, rows_block=rows_block, timer=self.timer,
                pairs=pairs, **self._qp_kw())
            tape = route_out.get('tape', tape)
            if whole_set:
                self._flag_z_outside(route_out, states)
            # A root frozen here, or a calibrated shift, is read by the solve
            # AFTER the one that froze it, so either triggers one repeat.
            if not self._freeze_qp_branch(route_out, eps, xc_correction,
                                          states):
                break
        if whole_set:
            self.conventions.qp_set_settled = True
        if route_out.get('tape') is not None:
            route_out['tape'].drop_residues()
        route_out['xc_correction'] = xc_correction
        # What the last solve took, state by state, for the record: the route
        # and the pole strength Z, which the route does not keep on the chain.
        # A solve that reports no Z (a stand-in route) records None for it.
        z_of = route_out.get('z')
        self.qp_diagnostics = {
            'routes': self._routes_taken(route_out),
            'z': None if z_of is None else
            {int(p): float(z) for p, z in zip(states, z_of)}}
        return out, route_out

    def _tile_kw(self):
        return {} if self.tile_gb is None else {'tile_gb': self.tile_gb}

    # -------------------------------------------------------- forward/reverse
    def _outside_window(self, norb):
        """Mask of the orbitals with no quasiparticle equation of their own,
        everything outside the set."""
        mask = np.ones(norb, bool)
        mask[self.resolved.qp_set] = False
        return mask

    def _freeze_outside_shift(self, eps, ws, env=None):
        """Calibrate the scissor the orbitals outside the set carry, once.

        `calibrate_scissor` gives every outside orbital the quasiparticle
        correction of the explicitly solved orbital NEAREST it in orbital
        energy, so the occupied ones take an occupied probe's shift and the
        virtual ones a virtual probe's -- two tiers, which is where the
        sensitivity splits: a 20 eV error on a core state moves a frontier
        quasiparticle by 0.05 meV, the inner valence by 3.6-10 meV per eV.

        Frozen at the FIRST forward, which is the reference geometry's in
        every path that then displaces it, and held from then on like the
        quasiparticle branch and the quadrature. A shift re-derived per geometry
        steps the surface wherever the nearest probe changes, and it would put
        a d(shift)/dR into a force that has no term for it -- the shift being
        a constant is exactly what lets `_fold_to_nuclei` send the seed of an
        outside orbital straight to the mean-field eigenvalue.

        A HOLE of an adaptive set (a candidate left unsolved) takes its frozen
        probe's shift instead.

        IN A CONTINUUM THE SHIFT IS THE PROBE'S GW CORRECTION ALONE. The
        probe's root is w' + Sigma^env_qq, w' solving w' = eps_q + <q|Sigma_x
        - v_xc|q> + Sigma_c,qq(w'), and every orbital outside the set gets its
        own Sigma^env_pp on the diagonal (`_env_static_outside`), so the
        scissor transfers w - eps_q - Sigma^env_qq = w' - eps_q, exactly the
        gas-like GW correction, never the probe's reaction field
        on top of the orbital's own (Duchemin, Jacquemin and Blase, J. Chem.
        Phys. 144, 164106 (2016), Eq. (18)). Lent twice, it put every
        scissored orbital of formaldehyde in toluene 1.7-1.8 eV off and S1/T1
        17/23 meV low. Gas phase: `env` is None, the root minus eps, bitwise.
        """
        if self.resolved.outside_shift is not None:
            return
        roots = {int(p): float(w) for p, w in zip(self.resolved.qp_set, ws)}
        free = self._env_free(roots, env)
        outside = np.flatnonzero(self._outside_window(len(eps)))
        on_curve = {int(p) for c in self.resolved.outside_curve.values()
                    for p in c.members} if self.outside_scissor == 'curve' \
            else set()
        if on_curve:
            outside = np.array([p for p in outside if int(p) not in on_curve],
                               int)
        part = self.resolved.qp_partition
        # the probes each orbital takes its shift from, frozen beside it
        probes = {int(p): (nearest_probe(eps, free, p),) for p in outside}
        if part is None or not part.tier_of:
            self.conventions.freeze_outside_shift(
                self.resolved, calibrate_scissor(eps, self.nocc, free,
                                                 outside), probes)
            return
        # Beyond the window the nearest explicit orbital, which is a window
        # edge; inside it each hole's frozen probe (`calibrate_scissor_tiers`).
        holes = {p: b for p, b in part.tier_of.items() if p in set(outside)}
        probes.update({int(p): tuple(int(q) for q in np.atleast_1d(b))
                       for p, b in holes.items()})
        shift = calibrate_scissor(eps, self.nocc, free,
                                  [p for p in outside if int(p) not in holes])
        shift.update(calibrate_scissor_tiers(eps, free, holes))
        self.conventions.freeze_outside_shift(self.resolved, shift, probes)

    def _freeze_outside_curve(self, mol, mf, x_mo, d_sigma, eps, eps_qp,
                              env):
        """Place the frozen `OutsideCurve` of each side, once, at the first
        forward: the continuation on every orbital (one self-energy, every
        diagonal element continued at once), on this chain's bare factors and
        static term, each member's correction without its own Eq. (18) term
        (lent like the tiers'), anchored on the edge's block-shared root."""
        if self.resolved.outside_shift is not None:
            return
        outside = np.flatnonzero(self._outside_window(len(eps)))
        part = self.resolved.qp_partition
        holes = (set(part.tier_of) if part is not None and part.tier_of
                 else set())
        if holes:
            # a window edge within QP_HAMILTONIAN_WINDOW of the outside is on
            # the curve with the outside orbitals it rotates with
            holes -= set(window_edge_holes(eps, self.nocc, part.candidates,
                                           self.degeneracy_tol))
        outside = [int(p) for p in outside if int(p) not in holes]
        sides = curve_sides(eps, self.nocc, self.resolved.qp_set, outside)
        curves = {}
        if not sides:
            self.conventions.freeze_outside_curves(self.resolved, curves)
            return
        a, z, failed = self._ac_shifts(mol, mf, x_mo, d_sigma,
                                       np.arange(len(eps)), env,
                                       'qp_outside_curve.ac')
        free = self._env_free({p: v for p, v in a.items()}, env)
        for side, (edge, members) in sides.items():
            mine = set(members)
            edge_shift = self._env_free(
                {edge: float(eps_qp[edge]) - float(eps[edge])}, env)[edge]
            curves[side] = fit_outside_curve(
                eps, side, edge, edge_shift, members,
                {p: free[p] for p in mine}, {p: z[p] for p in mine},
                failures=failed)
        self.conventions.freeze_outside_curves(self.resolved, curves)

    def _ac_shifts(self, mol, mf, x_mo, d_sigma, states, env, phase):
        """({p: a_p}, {p: Z_p}, {p: 'Class: message'}) of the
        analytic-continuation GW on `states`, the last naming the exception
        of each state whose continuation failed, the same on every rank: Sigma screened bare (`d_sigma`; sliced, those
        rows) with this chain's static term. The one continuation the
        adaptive selection, its end-of-walk check and the outside curve read.
        """
        with self.phase(phase):
            xc = self._xc_correction(mf, np.asarray(states, int), env)
            # W_solv would count the continuum again beside Eq. (18)
            factors = (d_sigma if isinstance(d_sigma, SlicedFactors)
                       else (x_mo, d_sigma))
            failed = {}
            a, z = ac_quasiparticle_shifts(mf, mol, self.nocc, states,
                                           factors=factors, xc_diagonal=xc,
                                           numerics=self.settings,
                                           failures=failed,
                                           reaction_field=env)
            return lockstep((a, z, failed))

    def _curve_at(self, eps):
        """[(members, Delta, dDelta/deps)] of each frozen curve at `eps`;
        none unless outside_scissor='curve'."""
        out = []
        if self.outside_scissor != 'curve':
            return out
        for side in sorted(self.resolved.outside_curve):
            curve = self.resolved.outside_curve[side]
            idx = np.asarray(curve.members, int)
            delta, slope = curve(np.asarray(eps, float)[idx])
            out.append((idx, delta, slope))
        return out

    def _env_static_outside(self, shift, norb):
        """<p|Sigma^env|p> on the orbitals the quasiparticle window LEFT OUT,
        zero on the ones inside it (whose roots already carry it whole).

        The scissor outside the window approximates GW there, never the
        reaction field: Sigma^env is ~ +lambda/2 on EVERY occupied orbital and
        ~ -lambda/2 on EVERY virtual one, and the direct kernel's vtilde
        reaches EVERY pair; the two cancel for a compact electron-hole pair
        only if both halves reach the same orbitals. Without this term the
        local roots the window leaves out keep a whole lambda of spurious BLUE
        shift (C2H4...F2/cc-pVDZ at eps = 2.5, window 2: the local 11 -> LUMO
        root +2186 meV without it, -43 meV with it; the charge-transfer root
        inside the window 5 meV either way). Eq. (18) is one vector over the
        whole spectrum, so it costs nothing; the gas phase passes None.
        """
        if shift is None:
            return None
        return np.where(self._outside_window(norb), np.asarray(shift, float),
                        0.0)
