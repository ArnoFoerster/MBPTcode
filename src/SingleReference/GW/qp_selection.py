"""Which orbitals of the admitted window a surface solves explicitly: the
adaptive explicit set.

The admitted window C is the outer bound. Inside it, an orbital whose
quasiparticle equation is not solved (a hole) carries a frozen scissor, the
explicit shift of one solved orbital (its probe). That costs each excitation
Omega_s, to first order in the shift error delta_p of each hole,

    dOmega_s = sum_p n_p^s delta_p,     n_p^s = dOmega_s / d eps_p

and the Hellmann-Feynman weights of the full BSE (X^T X - Y^T Y = 1, the
orbital energies entering A alone) are

    n_a^s = + sum_i (X_ia^2 + Y_ia^2)      (virtual a)
    n_i^s = - sum_a (X_ia^2 + Y_ia^2)      (occupied i).

The set is grown until the absolute budget B_s = sum_p |n_p^s| (|delta_p| +
u_p) is below the tolerance for EACH target state: a signed sum cancels
inside one state, and a criterion on a gap cancels between two.

Where delta comes from. One analytic-continuation (AC) GW over the window at
the reference geometry gives a shift a_p for every candidate. It selects only:
it decides which orbitals are solved and which probe a hole borrows, and no
AC number ever reaches an energy or a force. Its error is calibrated on the
explicit roots, r_q = a_q - s_q, interpolated in eps between the explicit
states of the hole's side (occupied states continue on the other branch);
a uniform AC offset cancels from the calibrated a~_p = a_p - r^_p entirely,
and u_p, half the spread of r over the hole's bracket, carries what the
interpolation cannot know.

The probe of a hole is the explicit block of its own side whose shift is
nearest the hole's calibrated a~_p (AC-matched tiers). That minimizes |delta_p|
per hole, so adding a probe never raises any term and the greedy budget is
monotone. A degenerate block is never split -- as a member, a hole or a
probe -- and a probe block lends the mean of its members' shifts, which does
not depend on the rotation inside it.

The hole cap. The budget weighs a hole by the targets' weight on it, so a
hole no target reaches could carry any error, and a root that is not a target
at the reference geometry -- but becomes the followed state along a walk --
would carry all of it. So no hole may carry more than ADAPTIVE_HOLE_MAX_EV of
its own, |delta_p| + u_p, whatever its weight: that bounds every root's
first-order error, not only the targets'. The partition is still frozen at
the reference geometry, so the surface stays smooth; the cost is fewer holes.

Pure numpy, except `ac_quasiparticle_shifts`, the one call into a GW route.
Everything here is in Hartree; tolerances are taken in the units of their
constants (meV, eV).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from src.Base.constants import (ADAPTIVE_HOLE_MAX_EV, HARTREE_TO_EV,
                                HARTREE_TO_MEV)
from src.Base.environment import NoEnvironment, attached_environment
from src.SingleReference.GW.space_time import solve_qp_energy_space_time


@dataclass(frozen=True)
class Calibration:
    """The AC shifts corrected on the explicit roots, Hartree.

    a_tilde:  {p: calibrated AC shift} over the candidates; s_q itself on an
              explicit q.
    residual: {q: a_q - s_q} over the explicit states.
    u:        {p: half the spread of r over p's bracket} over the holes.
    """
    a_tilde: Dict[int, float]
    residual: Dict[int, float]
    u: Dict[int, float]


@dataclass(frozen=True)
class Selection:
    """One run of the greedy rule.

    explicit:  the selected set, sorted.
    tier_of:   {hole: probe block} for every hole.
    signed:    {target: first-order dOmega}, Hartree.
    budget:    {target: B}, Hartree.
    order:     [(block, target, term in Hartree)], the blocks the rule added,
               in order, each with the target and the term that drove it.
    fell_back: 'window' when the rule had to take the whole window, else None.
    capped:    [(block, |delta| + u in Hartree)], the blocks the hole cap
               added, in order, each with its own error as a hole then.
    """
    explicit: Tuple[int, ...]
    tier_of: Dict[int, Tuple[int, ...]]
    signed: Dict[tuple, float]
    budget: Dict[tuple, float]
    order: list
    fell_back: Optional[str] = None
    capped: list = field(default_factory=list)


@dataclass(frozen=True)
class AdaptivePartition:
    """The frozen outcome of a selection: which candidates are explicit, which
    probe each hole borrows, and what it was selected for.

    explicit:   E, sorted; a subset of `candidates`.
    tier_of:    {hole: probe block} over C minus E.
    candidates: C, the admitted window, the outer bound.
    targets:    the (spin, root) pairs the budget held for.
    tol_meV:    the budget per target.
    provenance: the record of how it was selected (`qp_bookkeeping`).
    """
    explicit: Tuple[int, ...]
    tier_of: Dict[int, Tuple[int, ...]] = field(hash=False)
    candidates: Tuple[int, ...] = ()
    targets: Tuple[Tuple[str, int], ...] = ()
    tol_meV: float = 0.0
    provenance: dict = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        explicit = tuple(sorted(int(p) for p in self.explicit))
        candidates = tuple(sorted(int(p) for p in self.candidates))
        tier_of = {int(p): tuple(sorted(int(q) for q in np.atleast_1d(b)))
                   for p, b in self.tier_of.items()}
        if not set(explicit) <= set(candidates):
            raise ValueError('an adaptive set is a subset of its candidates: '
                             f'{sorted(set(explicit) - set(candidates))} are not')
        holes = set(candidates) - set(explicit)
        if set(tier_of) != holes:
            raise ValueError(
                f'every hole needs a probe and only holes have one: holes '
                f'{sorted(holes)}, tiers for {sorted(tier_of)}')
        for p, block in tier_of.items():
            if not set(block) <= set(explicit):
                raise ValueError(f'hole {p} borrows from {block}, which is '
                                 'not explicit')
        object.__setattr__(self, 'explicit', explicit)
        object.__setattr__(self, 'candidates', candidates)
        object.__setattr__(self, 'tier_of', tier_of)
        object.__setattr__(self, 'targets', tuple(
            (str(s), int(r)) for s, r in self.targets))
        object.__setattr__(self, 'tol_meV', float(self.tol_meV))

    @property
    def holes(self):
        """C minus E, sorted."""
        return tuple(sorted(self.tier_of))

    def as_record(self):
        """The partition as JSON: what a later stage reads it back from."""
        return {'explicit': list(self.explicit),
                'tier_of': {str(p): list(b) for p, b in self.tier_of.items()},
                'candidates': list(self.candidates),
                'targets': [list(t) for t in self.targets],
                'tol_meV': self.tol_meV}

    @classmethod
    def from_record(cls, record):
        """The partition `as_record` wrote."""
        return cls(explicit=record['explicit'],
                   tier_of={int(p): tuple(b)
                            for p, b in record['tier_of'].items()},
                   candidates=record['candidates'],
                   targets=tuple(tuple(t) for t in record['targets']),
                   tol_meV=record['tol_meV'])


def first_order_weights(xn, yn, nocc, norb, occ=None, virt=None):
    """n_p = dOmega/deps_p of each root (Hellmann-Feynman on the full BSE).

    xn, yn: (n_ov,) for one root or (n_ov, nroots); yn None is Tamm-Dancoff.
    occ, virt: the orbitals the pair space runs over, default every occupied
               and every virtual one; a window-restricted BSE passes its own.
    Returns (norb,) for one root, (nroots, norb) otherwise; zero off the pairs.
    """
    occ = np.arange(nocc) if occ is None else np.asarray(occ, int)
    virt = np.arange(nocc, norb) if virt is None else np.asarray(virt, int)
    x = np.asarray(xn, float)
    single = x.ndim == 1
    x = x.reshape(x.shape[0], -1)
    y = np.zeros_like(x) if yn is None else np.asarray(yn, float).reshape(x.shape)
    norm = (x ** 2).sum(0) - (y ** 2).sum(0)
    w = ((x ** 2 + y ** 2) / norm).reshape(len(occ), len(virt), -1)
    n = np.zeros((w.shape[2], norb))
    n[:, occ] = -w.sum(1).T
    n[:, virt] = w.sum(0).T
    return n[0] if single else n


def _positive_definite(m):
    """Whether the symmetric part of m admits a Cholesky factor."""
    try:
        np.linalg.cholesky(0.5 * (m + m.T))
    except np.linalg.LinAlgError:
        return False
    return True


def casida_instability(a, b):
    """'A - B' or 'A + B', the first that is not positive definite, or None.

    The full Casida problem has a real positive spectrum only when both are.
    On the AC energies of a window that can fail where the production solve
    does not: a window leaves out the pairs that stabilize it, and a state
    the continuation cannot judge keeps its mean-field energy there (a PBE0
    LUMO, eV below its quasiparticle). The selection then reads its
    first-order weights in the Tamm-Dancoff approximation (b None), which
    has no such condition; the production solve is untouched.
    """
    if b is None:
        return None
    for name, m in (('A - B', a - b), ('A + B', a + b)):
        if not _positive_definite(m):
            return name
    return None


def degenerate_blocks(eps, members, tol):
    """`members` grouped into blocks of neighbours in energy closer than `tol`."""
    eps = np.asarray(eps, float)
    ordered = sorted((int(p) for p in members), key=lambda p: (eps[p], p))
    blocks, current = [], ordered[:1]
    for p in ordered[1:]:
        if eps[p] - eps[current[-1]] < tol:
            current.append(p)
        else:
            blocks.append(tuple(sorted(current)))
            current = [p]
    if current:
        blocks.append(tuple(sorted(current)))
    return blocks


def _block_map(blocks):
    """{orbital: its block}."""
    return {p: b for b in blocks for p in b}


def ac_unjudgeable(ac_shift, z, shift_max_ev):
    """{p: why} for the states the continuation cannot judge.

    A Pade or Newton that failed (a non-finite shift), a pole strength outside
    (0, 1], or a shift beyond `shift_max_ev`: such a state is never scissored
    on the continuation's word.
    """
    out = {}
    for p, a in ac_shift.items():
        zp = None if z is None else z.get(int(p))
        if a is None or not np.isfinite(a):
            out[int(p)] = 'continuation failed'
        elif zp is None or not np.isfinite(zp) or not 0.0 < zp <= 1.0:
            out[int(p)] = f'Z_AC {zp} outside (0, 1]'
        elif abs(a) * HARTREE_TO_EV > shift_max_ev:
            out[int(p)] = (f'|a| {abs(a) * HARTREE_TO_EV:.3f} eV > '
                           f'{shift_max_ev} eV')
    return out


def mandatory_members(eps, nocc, candidates, degeneracy_tol, unjudgeable=()):
    """{p: reason} of the orbitals every adaptive set solves.

    The two window edges keep the treatment outside the window the admitted
    one's (`calibrate_scissor` resolves to them) and bracket every hole; the
    frontier pair is where the continuation is most reliable and so where the
    calibration is anchored; a state the continuation cannot judge is solved.
    Each widened over its degenerate block.
    """
    candidates = sorted(int(p) for p in candidates)
    blocks = _block_map(degenerate_blocks(eps, candidates, degeneracy_tol))
    why = {}
    for edge in (candidates[0], candidates[-1]):
        for p in blocks[edge]:
            why.setdefault(p, 'edge')
    occ = [p for p in candidates if p < nocc]
    vir = [p for p in candidates if p >= nocc]
    for p in ([max(occ)] if occ else []) + ([min(vir)] if vir else []):
        for q in blocks[p]:
            why.setdefault(q, 'frontier')
    for p in unjudgeable:
        for q in blocks[int(p)]:
            why.setdefault(q, 'ac_unjudgeable')
    return why


def calibrate_ac(eps, nocc, ac_shift, explicit_shift, candidates):
    """The AC shifts calibrated on the explicit ones (`Calibration`).

    explicit_shift: {q: s_q = root_q - eps_q} on the explicit states.
    Each hole takes the residual r = a - s interpolated linearly in eps
    between the explicit states of its side that bracket it, and u, half the
    residual's spread over that bracket; beyond the outermost explicit state
    of its side, that state's residual and half the step to the next one.
    """
    eps = np.asarray(eps, float)
    residual = {int(q): float(ac_shift[q]) - float(s)
                for q, s in explicit_shift.items()}
    a_tilde, u = {}, {}
    for side in (True, False):
        # an explicit state the continuation failed on calibrates nothing
        probes = sorted((q for q, r in residual.items()
                         if (q < nocc) == side and np.isfinite(r)),
                        key=lambda q: (eps[q], q))
        for p in (int(p) for p in candidates):
            if (p < nocc) != side:
                continue
            if p in residual:
                a_tilde[p] = float(explicit_shift[p])
                continue
            if not probes:
                a_tilde[p], u[p] = float(ac_shift[p]), 0.0
                continue
            below = [q for q in probes if eps[q] <= eps[p]]
            above = [q for q in probes if eps[q] >= eps[p]]
            if below and above:
                lo, hi = below[-1], above[0]
                span = eps[hi] - eps[lo]
                t = 0.0 if span <= 0.0 else (eps[p] - eps[lo]) / span
                r_hat = (1.0 - t) * residual[lo] + t * residual[hi]
                u[p] = 0.5 * abs(residual[hi] - residual[lo])
            else:
                # beyond the outermost probe: its residual, and half the
                # step to the one inside it
                near, nxt = ((probes[-1], probes[-2:-1]) if below
                             else (probes[0], probes[1:2]))
                nxt = nxt[0] if nxt else None
                r_hat = residual[near]
                u[p] = (0.0 if nxt is None
                        else 0.5 * abs(residual[near] - residual[nxt]))
            a_tilde[p] = float(ac_shift[p]) - r_hat
    return Calibration(a_tilde=a_tilde, residual=residual, u=u)


def matched_tiers(eps, nocc, a_tilde, probe_shift, explicit, blocks):
    """{hole: probe block}: the explicit block of the hole's own side whose
    shift is nearest the hole block's mean calibrated AC shift.

    probe_shift: {q: s_q}, the shift each explicit q lends.
    Ties go to the nearest in energy, then to the lower index.
    """
    eps = np.asarray(eps, float)
    explicit = set(int(p) for p in explicit)
    probes = [b for b in blocks if b[0] in explicit]
    s_of = {b: float(np.mean([probe_shift[q] for q in b])) for b in probes}
    tier_of = {}
    for b in blocks:
        if b[0] in explicit:
            continue
        side = b[0] < nocc
        mine = [pb for pb in probes
                if (pb[0] < nocc) == side and np.isfinite(s_of[pb])]
        if not mine:
            raise ValueError(
                f'hole block {b} has no explicit probe on its side of the gap')
        a = float(np.mean([a_tilde[p] for p in b]))
        e = float(np.mean(eps[list(b)]))
        best = min(mine, key=lambda pb: (abs(a - s_of[pb]),
                                         abs(e - float(np.mean(eps[list(pb)]))),
                                         pb[0]))
        for p in b:
            tier_of[p] = best
    return tier_of


def error_terms(weights, a_tilde, u, probe_shift, tier_of, blocks):
    """({target: dOmega}, {target: B}, {target: {hole block: term}}), Hartree.

    delta_p = s_t(p) - a~_p, the shift a hole carries minus the one its own
    root would have (a~ averaged over the hole's degenerate block), so
    dOmega_s = sum_p n_p^s delta_p is the selected surface's Omega minus the
    whole window's to first order (signed, recorded) and
    B_s = sum_p |n_p^s| (|delta_p| + u_p) the budget (enforced).
    """
    u = {} if u is None else u
    signed, budget, terms = {}, {}, {}
    hole_blocks = [b for b in blocks if b[0] in tier_of]
    for target, n in weights.items():
        n = np.asarray(n, float)
        signed[target] = budget[target] = 0.0
        terms[target] = {}
        for b in hole_blocks:
            delta = _hole_delta(b, a_tilde, probe_shift, tier_of)
            t_signed = sum(n[p] * delta for p in b)
            t_abs = sum(abs(n[p]) * (abs(delta) + u.get(p, 0.0)) for p in b)
            signed[target] += t_signed
            budget[target] += t_abs
            terms[target][b] = t_abs
    return signed, budget, terms


def _hole_delta(block, a_tilde, probe_shift, tier_of):
    """delta of a hole block: its probe's mean shift minus its mean a~."""
    a = float(np.mean([a_tilde[p] for p in block]))
    s = float(np.mean([probe_shift[q] for q in tier_of[block[0]]]))
    return s - a


def hole_errors(a_tilde, u, probe_shift, tier_of, blocks):
    """{hole block: max over its members of |delta_p| + u_p}, Hartree: the
    first-order error the hole puts on a root of unit weight on it."""
    u = {} if u is None else u
    return {b: abs(_hole_delta(b, a_tilde, probe_shift, tier_of))
            + max(u.get(p, 0.0) for p in b)
            for b in blocks if b[0] in tier_of}


def capped_why(capped, **extra):
    """{orbital: why it is explicit} over the blocks the hole cap added."""
    return {int(p): {'reason': 'hole_cap', 'error_meV': e * HARTREE_TO_MEV,
                     **extra} for b, e in capped for p in b}


def select_explicit(eps, nocc, candidates, a_tilde, weights, tol_meV, *,
                    start, degeneracy_tol, probe_shift=None, u=None,
                    barred=(), hole_max_ev=ADAPTIVE_HOLE_MAX_EV):
    """The greedy rule (`Selection`).

    Starting from `start` (the mandatory members), first add the hole block
    whose own error |delta| + u is largest while one exceeds hole_max_ev (eV;
    None: no cap), then the hole block whose largest per-target term is
    largest until every target's budget is at most tol_meV. probe_shift:
    {q: s_q} of the states already solved; a member not yet solved lends its
    calibrated AC shift a~_q, the prediction of its root. Ties go to the
    larger weight, then the lower index. Deterministic, so ranks that hold
    the same inputs select the same set. tol_meV = 0 takes the whole window.
    barred: orbitals never added (a root the surface already rejected), whose
    blocks stay holes whatever they cost.
    """
    tol = float(tol_meV) / HARTREE_TO_MEV
    cap = None if hole_max_ev is None else float(hole_max_ev) / HARTREE_TO_EV
    candidates = sorted(int(p) for p in candidates)
    blocks = degenerate_blocks(eps, candidates, degeneracy_tol)
    bmap = _block_map(blocks)
    barred = set(int(p) for p in barred)
    chosen = set()
    for p in start:
        chosen.update(bmap[int(p)])
    known = {} if probe_shift is None else {int(q): float(s)
                                           for q, s in probe_shift.items()}
    order, capped = [], []
    while True:
        shift = {q: known.get(q, a_tilde[q]) for q in chosen}
        tier_of = matched_tiers(eps, nocc, a_tilde, shift, chosen, blocks)
        signed, budget, terms = error_terms(weights, a_tilde, u, shift,
                                            tier_of, blocks)
        holes = [b for b in blocks if b[0] in tier_of
                 and not set(b) & barred]
        if cap is not None:
            errors = hole_errors(a_tilde, u, shift, tier_of, blocks)
            # one at a time: each new probe can bring other holes under it
            over = [b for b in holes if errors[b] > cap]
            if over:
                pick = max(over, key=lambda b: (errors[b], -b[0]))
                capped.append((pick, errors[pick]))
                chosen.update(pick)
                continue
        if not tier_of or (tol > 0.0
                           and max(budget.values(), default=0.0) <= tol):
            break

        def key(b):
            worst = max((terms[t][b], t) for t in terms)
            heavy = max(abs(float(np.asarray(weights[t])[p]))
                        for t in weights for p in b)
            return (worst[0], heavy, -b[0])

        if not holes:
            break
        pick = max(holes, key=key)
        driver = max(terms, key=lambda t: terms[t][pick])
        order.append((pick, driver, terms[driver][pick]))
        chosen.update(pick)
    explicit = tuple(sorted(chosen))
    return Selection(
        explicit=explicit, tier_of=tier_of, signed=signed, budget=budget,
        order=order, capped=capped,
        fell_back='window' if (order or capped) and len(explicit) + len(
            barred & set(candidates) - set(explicit)) == len(candidates)
        else None)


def near_root_targets(omegas, targets, mix_ev):
    """`targets` plus every root of the same spin within mix_ev of one.

    omegas: {spin: ascending excitation energies, Hartree}. Along a walk such
    a root can mix with, or swap onto, the followed state.
    """
    out = list(dict.fromkeys((str(s), int(r)) for s, r in targets))
    for spin, root in list(out):
        om = np.asarray(omegas.get(spin, ()), float)
        if root >= len(om):
            continue
        for k in np.flatnonzero(np.abs(om - om[root]) * HARTREE_TO_EV
                                <= mix_ev):
            if (spin, int(k)) not in out:
                out.append((spin, int(k)))
    return tuple(out)


def frontier_residual(calibration, eps, nocc, explicit, degeneracy_tol):
    """max |r_q| over the frontier blocks (HOMO, LUMO), Hartree: the gate
    on whether the continuation is fit to select at all."""
    blocks = _block_map(degenerate_blocks(eps, explicit, degeneracy_tol))
    occ = [p for p in explicit if p < nocc]
    vir = [p for p in explicit if p >= nocc]
    front = ([max(occ)] if occ else []) + ([min(vir)] if vir else [])
    # a frontier state the continuation failed on fails the gate
    return max((abs(calibration.residual[q]) if np.isfinite(
        calibration.residual[q]) else np.inf
                for p in front for q in blocks[p]), default=0.0)


def calibrated_selection(eps, nocc, candidates, ac_shift, explicit_shift,
                         weights, tol_meV, *, degeneracy_tol, gate_ev,
                         start=(), barred=(),
                         hole_max_ev=ADAPTIVE_HOLE_MAX_EV):
    """(`Selection`, `Calibration`) once explicit roots exist.

    The continuation is calibrated on the explicit shifts; if its residual
    on the frontier exceeds gate_ev it is broken here and the whole window is
    solved (fell_back='calibration'). Otherwise the greedy rule runs on the
    calibrated shifts with u, from the explicit set and `start`, under the
    same hole cap.
    """
    candidates = tuple(sorted(int(p) for p in candidates))
    calibration = calibrate_ac(eps, nocc, ac_shift, explicit_shift,
                               candidates)
    gate = frontier_residual(calibration, eps, nocc, sorted(explicit_shift),
                             degeneracy_tol)
    if not np.isfinite(gate) or gate * HARTREE_TO_EV > gate_ev:
        return Selection(explicit=candidates, tier_of={}, signed={},
                         budget={}, order=[], fell_back='calibration'), \
            calibration
    selection = select_explicit(
        eps, nocc, candidates, calibration.a_tilde, weights, tol_meV,
        start=set(explicit_shift) | set(int(p) for p in start),
        degeneracy_tol=degeneracy_tol, probe_shift=explicit_shift,
        u=calibration.u, barred=barred, hole_max_ev=hole_max_ev)
    return selection, calibration


def ac_quasiparticle_shifts(mf, mol, nocc, states, *, factors, xc_diagonal,
                            comm=None, timings=None):
    """({p: a_p}, {p: Z_AC}): the analytic-continuation quasiparticle shift
    a_p = omega_p - eps_p and pole strength of each state, Hartree. For the
    selection only: no number returned here may reach an energy or a force.

    The conventional space-time GW (`solve_qp_energy_space_time`) on the
    caller's own factors and its own static term, so that a - s, against the
    caller's explicit shift s, is the correlation part alone:
    factors:     the factors the caller's self-energy screens with (the bare
                 gauge in a continuum), as (X_mo, D) or `SlicedFactors`;
    xc_diagonal: the caller's <p|Sigma_x - v_xc + Sigma^env|p> on `states`,
                 Eq. (18) included, which replaces the route's own build.
    The mean field is evaluated with no environment attached: the factors
    are already bare and the environment's static term is in xc_diagonal.
    A state whose continuation or root search fails comes back NaN.
    """
    states = np.asarray(states, int)
    with attached_environment(mf, NoEnvironment()):
        roots, z = solve_qp_energy_space_time(
            mf, mol, nocc, states, factors=factors,
            xc_diagonal=np.asarray(xc_diagonal, float), with_z=True,
            comm=comm, timings=timings)
    eps = np.asarray(mf.mo_energy, float)
    return ({int(p): float(w) - float(eps[p]) for p, w in zip(states, roots)},
            {int(p): float(zz) for p, zz in zip(states, z)})
