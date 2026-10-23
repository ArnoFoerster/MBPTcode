"""Adaptive selection of the explicit quasiparticle set, and its records.

At the first forward one analytic-continuation GW over the admitted window
decides which candidates are solved explicitly; the selection is verified
against the production eigenvectors, frozen for the walk, and re-opened
grow-only by `refreeze`. `ExcitedStateChain` inherits these methods.
"""
import warnings

import numpy as np

from src.Base.factorization.sliced_factors import whole_factor
from src.Base.numerics.constants import (ADAPTIVE_AC_CALIBRATION_MAX_EV,
                                         ADAPTIVE_AC_SHIFT_MAX_EV, ADAPTIVE_HOLE_MAX_EV,
                                         ADAPTIVE_MAX_ROUNDS, ADAPTIVE_QP_TOL_MEV,
                                         ADAPTIVE_ROOT_MIX_EV, ADAPTIVE_SELECT_MARGIN,
                                         HARTREE_TO_EV, HARTREE_TO_MEV,
                                         QP_HAMILTONIAN_WINDOW)
from src.SingleReference.GW.qp_selection import (AdaptivePartition,
                                                 ac_unjudgeable, calibrate_ac,
                                                 calibrated_selection,
                                                 capped_why,
                                                 casida_instability,
                                                 degenerate_blocks,
                                                 error_terms,
                                                 first_order_weights,
                                                 frontier_residual,
                                                 hole_errors,
                                                 mandatory_members,
                                                 near_root_targets,
                                                 select_explicit,
                                                 window_edge_holes)
from src.gradients.gwbse.bse_isdf import bse_blocks, bse_solve


class QuasiparticleSelection:
    """The adaptive explicit set: selection, verification and records."""

    def __init__(self, chain):
        self.chain = chain

    def _owns_selection(self, sel):
        """Whether the selection state `sel` was started by this chain."""
        return sel is not None and sel['owner'] is self.chain._owner

    def _init_adaptive(self, qp_select, qp_partition):
        """The adaptive explicit set's state: what selects, what was selected.

        qp_partition, when given, is the set already: `qp_set` becomes its
        explicit orbitals and nothing is selected. Otherwise `qp_set` holds the
        candidates until the first forward selects.
        """
        self.chain.qp_select = qp_select
        self.chain.conventions.qp_partition = qp_partition
        # the quadrature asked for, which a re-solve of the selection starts
        # from again
        self.chain.conventions.nfreq_cd_asked = self.chain.conventions.nfreq_cd
        # the record of how the partition was chosen (`selection_record`)
        self.chain.conventions.qp_selection = None
        # the selection's working state at the reference geometry
        self.chain._selecting = None
        if qp_select is None and qp_partition is None:
            return
        if qp_partition is not None:
            window = [int(p) for p in self.chain.resolved.qp_set]
            if tuple(window) != qp_partition.candidates:
                raise ValueError(
                    'a frozen partition belongs to its candidates: qp_window '
                    f'{window} is not {list(qp_partition.candidates)}')
            frozen = self.chain.conventions
            frozen.qp_partition = self._promoted(
                qp_partition, np.asarray(self.chain.mf0.mo_energy, float))
            frozen.qp_set = np.asarray(frozen.qp_partition.explicit, int)
            frozen.qp_selection = dict(qp_partition.provenance) or None

    def _promoted(self, partition, eps, barred=()):
        """`partition` with the holes the quasiparticle Hamiltonian must
        solve made explicit, recorded in `group_promoted`; unchanged on the
        diagonal route.

        A group (neighbours on one side of the gap closer than
        QP_HAMILTONIAN_WINDOW, as the Hamiltonian's subspaces are cut) is
        invariant to how its orbitals rotate into each other only if they are
        all solved in it: a hole crossing an explicit member trades a frozen
        tier shift for an own correction within the crossing's width (DMABN
        gas S1 minimum: orbital 49 explicit, 50 a hole of tier 52, index 49's
        correction 558 -> 596 meV across their crossing). So a hole sharing a
        group with an explicit orbital, or with a hole of another probe,
        becomes explicit; holes sharing one probe in a group of holes keep
        it, the same shift for every rotation among them. A rejected root
        and a window edge within QP_HAMILTONIAN_WINDOW of the outside
        (`barred`) stay holes: the edge rotates with the outside orbitals,
        whose shift it borrows, not with the group inside.
        """
        if not self.chain.qp_hamiltonian or partition is None:
            return partition
        explicit = set(partition.explicit)
        promoted = {}
        for side in ([p for p in partition.candidates if p < self.chain.nocc],
                     [p for p in partition.candidates if p >= self.chain.nocc]):
            for group in degenerate_blocks(eps, side, QP_HAMILTONIAN_WINDOW):
                holes = [p for p in group
                         if p not in explicit and p not in set(barred)]
                if len(group) < 2 or not holes:
                    continue
                if explicit & set(group) or len(
                        {partition.tier_of[p] for p in holes}) > 1:
                    promoted.update({int(p): tuple(group) for p in holes})
        if not promoted:
            return partition
        self.chain.conventions.promote(self.chain.resolved, promoted)
        return AdaptivePartition(
            explicit=sorted(explicit | set(promoted)),
            tier_of={h: b for h, b in partition.tier_of.items()
                     if h not in promoted},
            candidates=partition.candidates, targets=partition.targets,
            tol_meV=partition.tol_meV, provenance=partition.provenance)

    # --------------------------------------------- the adaptive explicit set
    def _target_states(self):
        """The (spin, root) pairs the selection budget holds for."""
        spec = self.chain.qp_select
        if spec is not None and spec.targets:
            return tuple(spec.targets)
        return ((self.chain.spin, int(self.chain.state)),)

    def _tol_mev(self):
        spec = self.chain.qp_select
        if spec is not None and spec.tol_meV is not None:
            return float(spec.tol_meV)
        if self.chain.resolved.qp_partition is not None:
            return float(self.chain.resolved.qp_partition.tol_meV)
        return ADAPTIVE_QP_TOL_MEV

    @staticmethod
    def _env_free(values, env):
        """{p: v_p - <p|Sigma^env|p>}: a shift without its orbital's own
        Eq. (18) term, which every orbital outside the set gets on its own."""
        if env is None:
            return dict(values)
        return {int(p): float(v) - float(env[int(p)])
                for p, v in values.items()}

    def _window_weights(self, x_mo, d, eps, eps_window, w_aux, window, spins):
        """({(spin, root): n over every orbital}, {spin: Omega}, {spin: the
        matrix that was not positive definite, or None}) of the BSE
        restricted to the pairs inside `window`, on `eps_window` (the AC
        energies): the first-order weights the selection reads before any
        explicit root exists. Pairs with a partner outside the window are
        missing here; the production eigenvectors check that afterwards.
        Where the window's full BSE is unstable the weights are Tamm-Dancoff
        (`casida_instability`)."""
        occ = [int(p) for p in window if p < self.chain.nocc]
        cols = occ + [int(p) for p in window if p >= self.chain.nocc]
        xw = whole_factor(x_mo, 'X_mo')[:, cols]
        dw = whole_factor(d, 'D')
        norb = len(eps)
        weights, omegas, unstable = {}, {}, {}
        for spin in spins:
            a, b, _ = bse_blocks(xw, dw, np.asarray(eps_window)[cols], w_aux,
                                 len(occ), spin=spin, bse_tda=self.chain.bse_tda)
            # Tamm-Dancoff weights where the window's full BSE is unstable
            unstable[spin] = casida_instability(a, b)
            om, xn, yn = bse_solve(a, None if unstable[spin] else b)
            n = first_order_weights(xn, yn, len(occ), len(cols))
            omegas[spin] = om
            for k in range(len(om)):
                full = np.zeros(norb)
                full[cols] = n[k]
                weights[(spin, k)] = full
        return weights, omegas, unstable

    def _floor_in(self, cands):
        """The carried states (`qp_floor`) among the candidates, sorted."""
        floor = self.chain.resolved.qp_floor
        if floor is None:
            return []
        cands = set(int(p) for p in cands)
        return sorted(int(p) for p in floor if int(p) in cands)

    @property
    def qp_growth(self):
        """What a grow-only refreeze did to the explicit set, once this
        surface's selection is frozen: {'carried': the explicit set of the
        surface it was refrozen from, 'added': the states explicit here and
        not there, 'not_explicit': carried states that are holes here, which
        only a window edge within QP_HAMILTONIAN_WINDOW of the outside here
        (`window_edge_holes`) makes}. None on
        a surface that was not refrozen from a selected one."""
        view = self.chain.resolved
        if view.qp_floor is None or view.qp_partition is None:
            return None
        floor = [int(p) for p in view.qp_floor]
        explicit = set(int(p) for p in view.qp_partition.explicit)
        return {'carried': floor,
                'added': sorted(explicit - set(floor)),
                'not_explicit': sorted(set(floor) - explicit)}

    def _select_qp_set(self, mol, mf, x_mo, d, d_sigma, eps, w_aux, shift):
        """Choose the explicit set out of the candidates, before any root.

        One analytic-continuation GW over the window on this chain's own
        factors and static term (`ac_quasiparticle_shifts`), the
        window-restricted BSE on those energies for the first-order weights
        of every target (and of every root within ADAPTIVE_ROOT_MIX_EV of
        one), then the greedy rule on the raw AC shifts to
        ADAPTIVE_SELECT_MARGIN of the budget. `qp_set` becomes that set; the
        roots solved next calibrate the continuation (`_refine_selection`).
        """
        cands = np.asarray(self.chain.resolved.qp_set, int)
        a, z, failed = self.chain.quasiparticles.continuation_shifts(mol, mf, x_mo, d_sigma, cands, shift,
                                       'qp_select.ac')
        bad = ac_unjudgeable(a, z, ADAPTIVE_AC_SHIFT_MAX_EV, failed)
        why = mandatory_members(eps, self.chain.nocc, cands, self.chain.degeneracy_tol,
                                unjudgeable=bad)
        # a window edge within QP_HAMILTONIAN_WINDOW of the outside is a hole
        # of the next probe inward, whatever the surface before solved
        edge = window_edge_holes(eps, self.chain.nocc, cands, self.chain.degeneracy_tol)
        # grow-only: the surface this one was refrozen from solved these
        for p in self._floor_in(cands):
            if p not in edge:
                why.setdefault(p, 'carried')
        targets = self._target_states()
        with self.chain.phase('qp_select.bse'):
            eps_ac = np.array(eps, float)
            for p in cands:
                if int(p) not in bad:
                    eps_ac[p] += a[int(p)]
            weights, omegas, unstable = self._window_weights(
                x_mo, d, eps, eps_ac, w_aux, cands,
                tuple(dict.fromkeys(t[0] for t in targets)))
        targets = near_root_targets(omegas, targets, ADAPTIVE_ROOT_MIX_EV)
        free = self._env_free(a, shift)
        sel_weights = {t: weights[t] for t in targets if t in weights}
        first = select_explicit(eps, self.chain.nocc, cands, free, sel_weights,
                                self._tol_mev() * ADAPTIVE_SELECT_MARGIN,
                                start=why, degeneracy_tol=self.chain.degeneracy_tol)
        for i, (block, target, term) in enumerate(first.order):
            for p in block:
                why.setdefault(p, {'reason': 'budget', 'order': i,
                                   'target': list(target),
                                   'term_meV': term * HARTREE_TO_MEV})
        for p, w in capped_why(first.capped).items():
            why.setdefault(p, w)
        # a carried state's degenerate partners, if its block widened here
        for p in first.explicit:
            why.setdefault(int(p), 'carried')
        self.chain._selecting = {
            'owner': self.chain._owner, 'candidates': tuple(int(p) for p in cands),
            'ac': a, 'z': z, 'ac_free': free, 'unjudgeable': bad, 'why': why,
            'targets': targets, 'weights': sel_weights, 'omegas': omegas,
            'window_bse': unstable, 'rounds': 0, 'fell_back': None,
            'edge_holes': edge,
            'verify_pending': False,
            'selection_budget': {t: b * HARTREE_TO_MEV
                                 for t, b in first.budget.items()}}
        self.chain.conventions.set_explicit(self.chain.resolved,
                                      np.asarray(first.explicit, int))

    def _refine_selection(self, eps, ws, shift):
        """After a solve of the selected set at the reference geometry:
        calibrate the continuation on the explicit roots and either grow the
        set (True: solve again) or freeze the partition (False).

        A frontier residual beyond ADAPTIVE_AC_CALIBRATION_MAX_EV means the
        continuation is unfit to select, and the whole window is solved.
        """
        sel = self.chain._selecting
        if (self.chain.resolved.qp_partition is not None
                or not self._owns_selection(sel)):
            return False
        cands = sel['candidates']
        explicit = [int(p) for p in self.chain.resolved.qp_set]
        edge = set(sel['edge_holes'])
        roots = {p: float(w) for p, w in zip(explicit, ws)}
        s_free = self._env_free({p: roots[p] - float(eps[p])
                                 for p in explicit}, shift)
        tol = self._tol_mev()
        # the production eigenvectors' weights once a check has read them
        weights = (sel['verification']['weights'] if 'verification' in sel
                   else sel['weights'])
        selection, calibration = calibrated_selection(
            eps, self.chain.nocc, cands, sel['ac_free'], s_free, weights,
            tol * ADAPTIVE_SELECT_MARGIN, degeneracy_tol=self.chain.degeneracy_tol,
            gate_ev=ADAPTIVE_AC_CALIBRATION_MAX_EV)
        if selection.fell_back == 'calibration':
            sel['fell_back'] = sel['fell_back'] or 'calibration'
        grown = [p for p in selection.explicit if p not in set(explicit)]
        if grown:
            sel['rounds'] += 1
            if sel['rounds'] > ADAPTIVE_MAX_ROUNDS:
                sel['fell_back'] = sel['fell_back'] or 'window'
                grown = [p for p in cands if p not in set(explicit)
                         and p not in edge]
            for p, w in capped_why(selection.capped,
                                   after='calibration').items():
                sel['why'].setdefault(p, w)
            for i, p in enumerate(grown):
                sel['why'].setdefault(
                    p, 'calibration_fallback' if sel['fell_back']
                    else {'reason': 'budget', 'order': i,
                          'after': 'calibration'})
            self._reopen_selection(sorted(set(explicit) | set(grown)))
            return True
        blocks = degenerate_blocks(eps, cands, self.chain.degeneracy_tol)
        tier_of = {p: b for p, b in selection.tier_of.items()}
        sel.update(calibration=calibration, s_free=s_free, blocks=blocks,
                   tier_of=tier_of, verify_pending=True,
                   budget_meV={t: b * HARTREE_TO_MEV
                               for t, b in selection.budget.items()},
                   signed_meV={t: b * HARTREE_TO_MEV
                               for t, b in selection.signed.items()})
        partition = AdaptivePartition(
            explicit=explicit, tier_of=tier_of, candidates=cands,
            targets=sel['targets'], tol_meV=tol)
        promoted = self._promoted(partition, eps, edge)
        if promoted is not partition:
            # solved inside their group: the set grows and is solved again
            for p in set(promoted.explicit) - set(explicit):
                sel['why'].setdefault(p, 'group')
            self._reopen_selection(promoted.explicit)
            return True
        self.chain.conventions.freeze_partition(self.chain.resolved, partition)
        self.chain.conventions.record_selection(self.chain.resolved,
                                          self.selection_record())
        return False

    def _reopen_selection(self, explicit):
        """Unfreeze what the last solve of the reference geometry froze, so
        the set `explicit` is solved there again from cold: the quadrature
        from its asked size, the Newton branch, the poles and the scissors.
        A warm repeat would put the earlier roots' last bits into this one,
        and the surface would then depend on how many rounds selected it."""
        self.chain.conventions.reopen(self.chain.resolved, explicit)
        self.chain.quasiparticles._build_cd_grid(self.chain.conventions.nfreq_cd_asked,
                            np.asarray(self.chain.mf0.mo_energy, float))

    def verify_selection(self, shared, solved):
        """Check a just-frozen adaptive set on the production eigenvectors at
        the reference geometry; True when the set grew and the forward must
        be repeated.

        solved: {spin: `CasidaSolution`} of the Casida solves on `shared`; a
        target spin not among them is solved once here on the same forward.
        The exact Hellmann-Feynman weights of every target give its budget
        B and signed first-order error; every other solved root's are
        recorded. A target over ADAPTIVE_QP_TOL_MEV adds the blocks that
        break it, at most ADAPTIVE_MAX_ROUNDS times, then the whole window.
        """
        sel = self.chain._selecting
        if not self._owns_selection(sel) or not sel.get('verify_pending'):
            return False
        with self.chain.phase('qp_select.verify'):
            solved = dict(solved)
            for spin in dict.fromkeys(t[0] for t in sel['targets']):
                if spin not in solved:
                    solved[spin] = self.chain.spin_view(spin).solve_casida(shared)
            spectra = {spin: (s.omega, s.x, s.y) for spin, s in solved.items()}
            check = self._exact_budgets(shared.eps, spectra, sel)
        sel['verify_pending'] = False
        sel['verification'] = check
        tol = self._tol_mev()
        over = [t for t in sel['targets']
                if check['budget'].get(t, 0.0) * HARTREE_TO_MEV > tol]
        if not over:
            self.chain.conventions.record_selection(self.chain.resolved,
                                              self.selection_record())
            return False
        cands = sel['candidates']
        explicit = [int(p) for p in self.chain.resolved.qp_set]
        sel['rounds'] += 1
        if sel['rounds'] > ADAPTIVE_MAX_ROUNDS:
            sel['fell_back'] = sel['fell_back'] or 'verification'
            grown = [p for p in cands if p not in set(explicit)
                     and p not in set(sel['edge_holes'])]
        else:
            cal = sel['calibration']
            more = select_explicit(
                shared.eps, self.chain.nocc, cands, cal.a_tilde, check['weights'],
                tol * ADAPTIVE_SELECT_MARGIN, start=explicit,
                degeneracy_tol=self.chain.degeneracy_tol,
                probe_shift=sel['s_free'], u=cal.u)
            grown = [p for p in more.explicit if p not in set(explicit)]
        if not grown:
            self.chain.conventions.record_selection(self.chain.resolved,
                                              self.selection_record())
            return False
        for p in grown:
            sel['why'].setdefault(p, 'verification')
        self._reopen_selection(sorted(set(explicit) | set(grown)))
        return True

    def _exact_budgets(self, eps, spectra, sel):
        """The budget and signed error of every solved root, and the exact
        weights of the targets, on the production eigenvectors."""
        norb, cands = len(eps), set(sel['candidates'])
        cal, part = sel['calibration'], self.chain.resolved.qp_partition
        out = {'weights': {}, 'budget': {}, 'signed': {}, 'outside': {},
               'omega': {}}
        for spin, (om, xn, yn) in spectra.items():
            # the reported roots and every target, not a dense solve's all
            top = max([int(self.chain.nroots)] + [int(t[1]) + 1
                                            for t in sel['targets']
                                            if t[0] == spin])
            om, xn, yn = om[:top], xn[:, :top], yn[:, :top]
            n_all = np.atleast_2d(first_order_weights(xn, yn, self.chain.nocc, norb))
            weights = {(spin, k): n_all[k] for k in range(len(om))}
            signed, budget, _ = error_terms(weights, cal.a_tilde, cal.u,
                                            sel['s_free'], part.tier_of,
                                            sel['blocks'])
            for k, t in enumerate(weights):
                out['budget'][t], out['signed'][t] = budget[t], signed[t]
                out['omega'][t] = float(om[k])
                out['outside'][t] = float(sum(abs(n_all[k][p])
                                              for p in range(norb)
                                              if p not in cands))
                if t in sel['targets']:
                    out['weights'][t] = n_all[k]
        return out

    def selection_record(self):
        """`qp_bookkeeping['adaptive']`: the candidates, the explicit set and
        why each member is in it, the holes with their probes, the
        continuation and its calibration, every target's budget, and the
        fallback, energies in eV and errors in meV. None for another set."""
        sel, part = self.chain._selecting, self.chain.resolved.qp_partition
        if sel is None:
            return self.chain.conventions.qp_selection
        ev, mev = HARTREE_TO_EV, HARTREE_TO_MEV
        cands = list(sel['candidates'])
        explicit = [int(p) for p in self.chain.resolved.qp_set]
        holes = [p for p in cands if p not in set(explicit)]
        cal = sel.get('calibration')
        record = {
            'candidates': cands, 'n_candidates': len(cands),
            'explicit': explicit, 'n_explicit': len(explicit),
            'why': {int(p): w for p, w in sorted(sel['why'].items())
                    if p in set(explicit)},
            'holes': holes,
            # edges within QP_HAMILTONIAN_WINDOW of the outside: {hole: probe}
            'window_edge_holes': {int(p): int(q) for p, q in
                                  sorted(sel.get('edge_holes', {}).items())},
            'ac': {'route': 'space-time pade',
                   'shift_eV': {p: sel['ac'][p] * ev for p in cands},
                   'z': {p: sel['z'][p] for p in cands},
                   'unjudgeable': dict(sel['unjudgeable'])},
            'tol_meV': self._tol_mev(), 'margin': ADAPTIVE_SELECT_MARGIN,
            'rounds': sel['rounds'], 'fell_back': sel['fell_back'],
            # the states solved because, as holes, their own error |delta| + u
            # would have exceeded the cap whatever their weight
            'hole_cap': {'max_eV': ADAPTIVE_HOLE_MAX_EV, 'kept_explicit': [
                p for p in explicit if isinstance(sel['why'].get(p), dict)
                and sel['why'][p].get('reason') == 'hole_cap']},
            # the form of the window BSE the selection weights came from:
            # Tamm-Dancoff where the full one was unstable (casida_instability)
            'window_bse': {
                spin: {'weights': 'tda' if self.chain.bse_tda or bad else 'full',
                       'tda_fallback': bad is not None,
                       'not_positive_definite': bad}
                for spin, bad in sel.get('window_bse', {}).items()}}
        if part is not None:
            record['tier_of'] = {p: list(b) for p, b in part.tier_of.items()}
            # what a later stage of the same run reads the partition back from
            record['partition'] = part.as_record()
        if cal is not None:
            front = [max((p for p in explicit if p < self.chain.nocc), default=None),
                     min((p for p in explicit if p >= self.chain.nocc),
                         default=None)]
            record['holes_detail'] = {
                p: {'tier': list(sel['tier_of'].get(p, ())),
                    'delta_eV': (float(np.mean(
                        [sel['s_free'][q] for q in sel['tier_of'][p]]))
                        - cal.a_tilde[p]) * ev if p in sel['tier_of'] else None,
                    'u_eV': cal.u.get(p, 0.0) * ev} for p in holes}
            record['calibration'] = {
                'residual_eV': {q: r * ev for q, r in cal.residual.items()},
                'frontier_max_abs_eV': max(
                    (abs(cal.residual[q]) * ev for q in front
                     if q is not None and q in cal.residual), default=None),
                'gate_eV': ADAPTIVE_AC_CALIBRATION_MAX_EV,
                'passed': sel['fell_back'] != 'calibration'}
            eps = np.asarray(self.chain.mf0.mo_energy, float)
            record['hole_margin_meV'] = min(
                (abs(eps[p] - eps[q]) * mev for p in holes for q in explicit),
                default=None)
            record['hole_cap']['largest_hole_meV'] = max(
                (e * mev for e in hole_errors(
                    cal.a_tilde, cal.u, sel['s_free'], sel['tier_of'],
                    sel['blocks']).values()), default=None)
        check = sel.get('verification')
        targets = []
        for t in sel['targets']:
            row = {'spin': t[0], 'root': int(t[1]),
                   'selection_budget_meV': sel['selection_budget'].get(t, 0.0),
                   'calibrated_budget_meV': sel.get('budget_meV', {}).get(t)}
            if check is not None and t in check['budget']:
                row.update(omega_eV=check['omega'][t] * ev,
                           predicted_error_meV=check['signed'][t] * mev,
                           budget_meV=check['budget'][t] * mev,
                           weight_outside_window=check['outside'][t])
            targets.append(row)
        record['targets'] = targets
        if check is not None:
            other = {}
            for t in check['budget']:
                if t in sel['targets']:
                    continue
                other.setdefault(t[0], []).append(
                    {'root': int(t[1]), 'omega_eV': check['omega'][t] * ev,
                     'predicted_error_meV': check['signed'][t] * mev,
                     'budget_meV': check['budget'][t] * mev})
            record['other_roots'] = other
        record['verified'] = check is not None
        # grow-only refreeze: what this pass carried and what it added
        record['refreeze_growth'] = self.qp_growth
        return record

    def posteriori_check(self, mol, mf, spectra, targets):
        """The selection's first-order budget at the end of a walk.

        One AC GW at `mol` on this geometry's factors (`last_forward`, which
        the last evaluation here left), calibrated on this geometry's
        explicit roots, and the exact weights of the walked `targets` from
        `spectra` ({spin: (Omega, X, Y)}): delta_p = a~_p(R*) - s_t(p)(R*)
        with the frozen map. Recorded ('exceeds_tol', the named flag) and
        warned about, never acted on. Also
        whether the overlap with the reference orbitals maps any window
        orbital onto another, which a frozen hole could not follow.
        """
        part = self.chain.resolved.qp_partition
        last = getattr(self.chain, 'last_forward', None)
        if part is None:
            return None
        if last is None or not np.array_equal(last['coords'],
                                              mol.atom_coords()):
            raise RuntimeError('the a posteriori check reads the last forward '
                               'at this geometry: evaluate the surface here '
                               'first')
        eps, eps_qp, env = last['eps'], last['eps_qp'], last['shift']
        cands = np.asarray(part.candidates, int)
        x_mo, d, _, _, _, _, _, d_bare = self.chain._factors_for(mol, mf)
        d_sigma = d if d_bare is None else d_bare
        a, z, _ = self.chain.quasiparticles.continuation_shifts(mol, mf, x_mo, d_sigma, cands, env,
                                  'qp_select.ac')
        explicit = [int(p) for p in self.chain.resolved.qp_set]
        # an explicit root carries its Eq. (18) term once, whole, added
        # after its solve
        s_free = self._env_free({q: float(eps_qp[q] - eps[q])
                                 for q in explicit}, env)
        cal = calibrate_ac(eps, self.chain.nocc, self._env_free(a, env), s_free,
                           cands)
        gate = frontier_residual(cal, eps, self.chain.nocc, explicit,
                                 self.chain.degeneracy_tol) * HARTREE_TO_EV
        blocks = degenerate_blocks(eps, cands, self.chain.degeneracy_tol)
        norb = len(eps)
        weights = {}
        for spin, root in targets:
            om, xn, yn = spectra[spin]
            weights[(spin, int(root))] = first_order_weights(
                xn[:, int(root)], yn[:, int(root)], self.chain.nocc, norb)
        signed, budget, terms = error_terms(weights, cal.a_tilde, cal.u,
                                            s_free, part.tier_of, blocks)
        tol = self._tol_mev()
        moved = self._orbital_map_changed(mol, mf, cands)
        out = {'ac_shift_eV': {int(p): a[int(p)] * HARTREE_TO_EV
                               for p in cands},
               'targets': [], 'orbital_map_changed': moved,
               'calibration': {
                   'residual_eV': {q: r * HARTREE_TO_EV
                                   for q, r in cal.residual.items()},
                   'frontier_max_abs_eV': gate,
                   'passed': bool(gate <= ADAPTIVE_AC_CALIBRATION_MAX_EV)}}
        exceeds = False
        for t in weights:
            worst = sorted(terms[t].items(), key=lambda bt: -bt[1])[:3]
            row = {'spin': t[0], 'root': t[1],
                   'budget_meV': budget[t] * HARTREE_TO_MEV,
                   'predicted_error_meV': signed[t] * HARTREE_TO_MEV,
                   'largest_terms_meV': [[list(b), v * HARTREE_TO_MEV]
                                         for b, v in worst]}
            out['targets'].append(row)
            if budget[t] * HARTREE_TO_MEV > tol:
                exceeds = True
                warnings.warn(
                    f'the adaptive set frozen at the reference geometry is '
                    f'over its {tol} meV budget at the end of the walk: '
                    f'{t[0]} root {t[1]} B = {budget[t] * HARTREE_TO_MEV:.2f} '
                    f'meV, largest terms {row["largest_terms_meV"]}',
                    RuntimeWarning, stacklevel=2)
        out['exceeds_tol'] = exceeds
        return out

    def _orbital_map_changed(self, mol, mf, cands):
        """{p: q} for every candidate p of the reference geometry that the
        conventions match at `mol` to another index q, or {}."""
        view = self.chain.conventions.at(mol, mf)
        return view.moved(self.chain.conventions.window)
