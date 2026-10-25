"""The quasiparticle equation f(w) = w - eps - Sigma(w) = 0 and its one root finder.

Sigma is any frequency-dependent quantity (GW/PSD self-energy, ADC static
correction, embedding self-energy). Every scalar quasiparticle equation in the
code base is solved by ONE routine, `find_qp_root`: it brackets the roots,
refines inside each bracket, weighs every root by its pole strength
Z = 1/f'(w) and keeps the largest valid one (`select_qp_root`); along a walk it
solves the equation in full at every geometry and keeps the root continuing
the one the previous geometry kept, whatever its weight, saying so when the
largest weight has moved to another root. Where the brackets come from is the self-energy's
business: a sum of known poles brackets its roots between neighbouring poles
(`PoleSum`), a self-energy known pointwise on a pole-aware grid
(`SampledEquation`). No root is found by a Newton from a seed or by a dense
eigensolve; Newton steps only refine inside a bracket, which bounds them.
"""
import warnings
from typing import NamedTuple

import numpy as np
from src.Base.numerics.constants import (
    QP_ROOT_MAX_ITER, QP_ROOT_TOL,
    QP_SEARCH_WINDOW_TIERS,
    QP_GRAPHICAL_TOL, QP_GRAPHICAL_N_OMEGA, QP_GRAPHICAL_MAX_BISECTION,
    QP_POLE_OFFSET, QP_SUBSPACE_TOL, QP_SUBSPACE_MAX_ITER,
    QP_WINDOW_Z_MIN, QP_Z_DERIV_STEP,
    QB_QP_END_BLOCK_BYTES, QB_QP_ROOT_TOL, QP_POLE_INTERVAL_SAMPLES,
    QP_SCAN_SECANT_MIN, QP_SCAN_SPLIT_DEPTH, QP_WEIGHT_SUM_SLACK,
)
from src.Base.numerical_flags import flag
from src.Base.step_failed import StepFailed


class QPRootNotConverged(StepFailed):
    """A quasiparticle root finder ended without a root it can vouch for.

    reason: 'no_root' (no root in the widest scan window); last: the value
    the finder would have returned. A root without a quasiparticle weight is
    no failure: it is returned and flagged, and neither is a followed root
    whose weight another root has overtaken (`find_qp_root`).
    """

    def __init__(self, message, reason, last):
        super().__init__(message)
        self.reason = reason
        self.last = last


class QPNoRoot(RuntimeWarning):
    """A quasiparticle equation without a root in its widest scan window:
    the root is NaN, the named flag 'no_root' is raised, and a caller's `out`
    dict carries 'no_root' (refuse inputs, warn numerics: a number that does
    not exist is a flagged NaN)."""


class QPWeightOutside(RuntimeWarning):
    """A quasiparticle root kept with Z outside (0, 1], which no root of a
    positive-weight self-energy has: a numerical event, kept with this
    warning, the named flag 'z_outside', and 'z_outside' in a caller's `out`
    dict."""


def _report(message, reason, last, refuse):
    """Raise QPRootNotConverged when the caller refuses this case, warn
    (`QPNoRoot`) and raise the named flag 'no_root' otherwise."""
    if refuse:
        raise QPRootNotConverged(message, reason, last)
    flag('no_root', message, QPNoRoot, stacklevel=3)


def z_outside(got):
    """Whether a root was found and its pole strength lies outside (0, 1]."""
    return bool(np.isfinite(got.w) and not 0.0 < got.z <= 1.0)


def _flag_outside(got, label):
    """Warn (`QPWeightOutside`) and raise the named flag 'z_outside' for a
    root kept with Z outside (0, 1]; the root stays as solved."""
    if z_outside(got):
        flag('z_outside',
             f'the quasiparticle root of {label or "the quasiparticle equation"} '
             f'{got.w!r} Ha has Z = {got.z!r}, outside (0, 1], which no root '
             f'of a positive-weight self-energy has: kept', QPWeightOutside,
             stacklevel=3)
    return got


def qp_start(eps, p, nocc, offset=QP_POLE_OFFSET):
    """The point a state is judged at before its root is known: eps_p pushed
    `offset` toward the chemical potential, the direction the quasiparticle
    correction takes. Eq. (27) of the pole model and the residue backend's
    frequency margin are read here; no root search starts from it."""
    return float(eps[p] + (offset if p < nocc else -offset))


def quasiparticle_weight(z, z_min=QP_WINDOW_Z_MIN):
    """Whether a root's pole strength Z = 1/(1 - dSigma/dw) is a
    quasiparticle's: finite and in (z_min, 1]. A positive-weight self-energy
    has dSigma/dw <= 0, so Z > 1 is no root of a Dyson equation; at or below
    QP_WINDOW_Z_MIN the root carries less than half the spectral weight and
    the state has no quasiparticle. z_min = 0 is the bare validity test."""
    z = float(z)
    return bool(np.isfinite(z) and z_min < z <= 1.0)


def select_qp_root(roots, weights, z_min=0.0, near=None):
    """Index of the root of largest weight among those passing
    `quasiparticle_weight(z, z_min)`, or with `near` the one closest to it
    among them (a root followed from a reference); None if none passes. The
    default z_min = 0 admits every valid weight, so the caller decides from the
    returned Z whether it is a quasiparticle."""
    ok = [i for i, z in enumerate(weights) if quasiparticle_weight(z, z_min)]
    if not ok:
        return None
    if near is None:
        return max(ok, key=lambda i: weights[i])
    return min(ok, key=lambda i: abs(roots[i] - near))


# ---------------------------------------------------------------------------
# The one root finder
# ---------------------------------------------------------------------------

class QPRoot(NamedTuple):
    """What `find_qp_root` returns: the root w, its pole strength z, whether z
    passes `quasiparticle_weight(z, z_min)`, and every (root, Z) it weighed in
    the window it ended on, in increasing order of the brackets.

    labels: (lo, hi), the labels of the poles bracketing w (`PoleSum`: each
        pole's index in the weights as given, None for no pole beyond an
        outer interval); None where the poles are unknown.
    Following a root (`find_qp_root` with `follow`):
    largest: (w, Z) of the largest valid weight at this geometry, None if the
        unfollowed search found no root;
    followed: True when the previous root's branch continued here, False when
        it had gone and w is the largest weight picked again.
    """
    w: float
    z: float
    quasiparticle: bool
    found: tuple
    labels: tuple = None
    largest: tuple = None
    followed: bool = None

    def flipped(self, tol):
        """Whether a followed root is not the root of largest weight here:
        the branch was re-picked, or another root (more than `tol` away)
        carries more weight than the one followed."""
        if self.followed is None:
            return False
        return not self.followed or (self.largest is not None and abs(
            self.largest[0] - self.w) > tol and self.largest[1] > self.z)


def find_qp_root(source, a, follow=None, z_min=QP_WINDOW_Z_MIN, label='',
                 refuse_unconverged=False):
    """The quasiparticle root of the equation `source` brackets, about the
    bare level a.

    Without `follow`, in each window of `source.windows(a)`, narrowest first,
    the brackets are refined in decreasing order of the upper bound on Z the
    source gives each, until no bound exceeds the largest valid weight found
    (`_weighed_roots`); a source whose weights sum to one (a positive-weight
    Sigma) has its grid walked outward from a instead, until the weights
    found leave no more for the rest (`_weighed_outward`). `select_qp_root`
    keeps that root. A weight above z_min
    ends the search; otherwise the window widens, and the widest keeps its
    largest valid weight (the root nearest a if none is valid), flagged as no
    quasiparticle. A source whose bounds are not all exact (`PoleSum` with
    negative weights) is searched again without them first, so that answer
    is exact too; one that could not evaluate f somewhere
    (`SampledEquation`'s refusals) raises that refusal instead, since the
    root may lie there. No root at all is 'no_root': refused under
    refuse_unconverged, NaN with a `QPNoRoot` warning and the named flag
    'no_root' otherwise. A root kept with Z outside (0, 1] stays as solved,
    with a `QPWeightOutside` warning and the named flag 'z_outside', on
    every route.

    With `follow` = (w, labels), the root the previous geometry of a walk
    kept and the labels of the poles that bracketed it (None where the
    poles are unknown, or to bracket it by energy): the root kept is the one
    continuing that branch (`source.follow_root`): between the same two
    labelled poles of a pole sum, else the root nearest w on the scan of the
    narrowest window about a that holds it. Every root is refined from its
    own bracket, never from w, so the bits of the root kept do not depend on
    the walk that reached this geometry, only which root it is. Beside it,
    the root of largest weight here (`QPRoot.largest`): for a pole sum the
    search without `follow`; for a scan the followed root itself where its
    Z exceeds QP_WINDOW_Z_MIN (no other root of a positive-weight
    self-energy can carry more), else the largest among the brackets of the
    window followed, a bracket refined only where its weight, estimated
    from one slope at its secant root, can beat the best so far
    (`SampledEquation.follow_root`).
    Re-deciding by weight at every geometry jumps branches where a satellite
    crosses the quasiparticle; the followed root stays on its branch through
    the avoided crossing, whatever its weight. Where the branch has gone (no
    valid root between its poles, poles gone or reordered), the largest
    weight is kept again (followed=False). Nothing is refused for a weight:
    `QPRoot.flipped` tells the caller that the largest weight has moved.
    """
    if follow is None:
        return _flag_outside(_largest_root(source, a, z_min, label,
                                           refuse_unconverged), label)
    w_prev, labels = (follow if isinstance(follow, (tuple, list))
                      else (follow, None))
    got = source.follow_root(float(w_prev), labels, a)
    if got is not None and got[3] is not None:
        # () is a largest weight the source was told not to search for
        w, z, labels, largest = got
        return _flag_outside(QPRoot(w, z, quasiparticle_weight(z, z_min), (),
                                    labels, largest or None, True), label)
    try:
        # no root of largest weight beside a followed one is no event
        best, failed = _largest_root(source, a, z_min, label,
                                     refuse_unconverged, report=False), None
    except tuple(getattr(source, 'refusals', ())) as refusal:
        best, failed = None, refusal
    largest = None if best is None else (best.w, best.z)
    if got is None:
        if failed is not None:
            raise failed
        if not np.isfinite(best.w):
            _report(f'{label or "the quasiparticle equation"}: the branch '
                    f'followed from {float(w_prev):.6f} Ha has gone and no '
                    f'root lies in the widest scan window about {a:.6f} Ha.',
                    'no_root', np.nan, refuse_unconverged)
        return _flag_outside(best._replace(largest=largest, followed=False),
                             label)
    w, z, labels, _ = got
    found = () if best is None else best.found
    return _flag_outside(QPRoot(w, z, quasiparticle_weight(z, z_min), found,
                                labels, largest, True), label)


def _largest_root(source, a, z_min, label, refuse_unconverged, report=True):
    """`find_qp_root` without `follow`: the root of largest valid weight;
    with `report` False, no root at all is the NaN root alone, for the
    caller to report."""
    found, k, window = [], None, None
    windows = list(source.windows(a))
    for i, window in enumerate(windows):
        # a window widened past only has to show it holds no quasiparticle
        found = _weighed_roots(source, *window, a, z_min if i + 1 < len(
            windows) else -np.inf)
        k = select_qp_root([w for w, _, _ in found], [z for _, z, _ in found])
        if k is not None and quasiparticle_weight(found[k][1], z_min):
            return _rooted(found, k, True)
    refused = getattr(source, 'refused', None)
    if refused is not None:
        # no quasiparticle where Sigma could be evaluated: the root may lie
        # where it could not, so the backend's refusal stands
        raise refused
    exact = getattr(source, 'exhaustive', None)
    if exact is not None and exact() is not None:
        # no quasiparticle: every bound that is not exact is dropped
        return _largest_root(exact(), a, z_min, label, refuse_unconverged,
                             report)
    if not found:
        if report:
            _report(f'{label or "the quasiparticle equation"} has no root in the '
                    f'widest scan window [{window[0]:.6f}, {window[1]:.6f}] '
                    f'Ha about {a:.6f} Ha.', 'no_root', np.nan,
                    refuse_unconverged)
        return QPRoot(np.nan, np.nan, False, ())
    if k is None:
        k = min(range(len(found)), key=lambda i: abs(found[i][0] - a))
    return _rooted(found, k, False)


def _rooted(found, k, quasiparticle):
    """The `QPRoot` of root k among `found` [(root, Z, labels)]."""
    return QPRoot(found[k][0], found[k][1], quasiparticle,
                  tuple((w, z) for w, z, _ in found), found[k][2])


def _weighed_roots(source, lo, hi, a, enough=-np.inf):
    """[(root, Z, labels)] of the brackets of [lo, hi] that can hold the root
    of largest valid weight, in increasing order of the brackets.

    Brackets go in decreasing order of their bound on Z, in batches that
    double, so the best Z found rises before the source's costlier tightening
    (`source.tighten`) reaches the long tail of weak brackets; a bracket is
    passed over only when a bound puts it at or below a valid Z already found.
    No bound under-estimates, so the largest valid weight is exact. A source
    with `weight_sum` is searched by `_weighed_outward` from a, which ends
    once no root left can carry more than `enough` either.
    """
    if getattr(source, 'weight_sum', None) is not None:
        return _weighed_outward(source, lo, hi, a, enough)
    brackets, bound = source.brackets(lo, hi)
    bound = np.asarray(bound, float)
    order = np.argsort(-bound, kind='stable')
    solved, best = {}, -np.inf
    start, size = 0, 1
    while start < order.size and bound[order[start]] > best:
        batch = order[start:start + size]
        batch = batch[bound[batch] > best]
        tight = np.minimum(bound[batch], source.tighten(brackets, batch))
        for k in np.argsort(-tight, kind='stable'):
            if tight[k] <= best:
                break
            got = source.solve(brackets[batch[k]])
            labels = source.labels_of(brackets[batch[k]])
            solved[int(batch[k])] = [(w, z, labels) for w, z in got]
            for _, z in got:
                if quasiparticle_weight(z, 0.0) and z > best:
                    best = z
        start, size = start + size, 2 * size
    return [rz for i in sorted(solved) for rz in solved[i]]


def _weighed_outward(source, lo, hi, a, enough=-np.inf):
    """`_weighed_roots` for a source whose roots' weights sum to
    `source.weight_sum` (a positive-weight Sigma: one): the brackets of the
    grid cells of [lo, hi], nearest a first (`source.outward`), each refined,
    until what the valid weights found leave of that sum, plus
    QP_WEIGHT_SUM_SLACK for the error in each Z, is at most the largest of
    them, or than `enough` (a window `_largest_root` widens past where it
    holds no weight above z_min, which it then need not search for): no
    root not yet found can then carry more. A quasiparticle near
    the bare level ends the walk within a few cells; a state without one
    walks the whole window, every bracket refined, as the full scan would.
    The cells and their brackets are the full scan's, so each root found is
    the same bits; only the roots no longer needed go unrefined."""
    found, best, weighed = [], -np.inf, 0.0
    for brackets in source.outward(lo, hi, a):
        for bracket in brackets:
            got = source.solve(bracket)
            labels = source.labels_of(bracket)
            found.append((bracket[0], [(w, z, labels) for w, z in got]))
            for _, z in got:
                if quasiparticle_weight(z, 0.0):
                    weighed += z
                    best = max(best, z)
        if (source.weight_sum - weighed + QP_WEIGHT_SUM_SLACK
                <= max(best, enough)):
            break
    found.sort(key=lambda lr: lr[0])
    return [rz for _, got in found for rz in got]


def _note_branch(got, label, tol, out):
    """Hand a followed root's branch to `out` (a dict: 'labels', 'largest',
    'followed', 'flipped', 'no_root' for a NaN root and 'z_outside' for a
    root kept with Z outside (0, 1]), or, with no
    `out`, warn where the largest weight has moved to another root
    (`QPRoot.flipped`)."""
    flipped = got.flipped(tol)
    if out is not None:
        out.update(labels=got.labels, largest=got.largest,
                   followed=got.followed, flipped=flipped,
                   no_root=not np.isfinite(got.w), z_outside=z_outside(got))
        return
    if flipped:
        warnings.warn(branch_warning(label, got), RuntimeWarning,
                      stacklevel=3)


def branch_warning(label, got):
    """The warning for a followed root the largest weight has left."""
    what = label or 'the quasiparticle equation'
    if not got.followed:
        return (f'{what}: the branch the previous geometry kept has gone; '
                f'the root of largest weight is kept again, {got.w:.10f} Ha '
                f'(Z = {got.z:.4f}).')
    big = ('' if got.largest is None else
           f' the largest weight is now {got.largest[1]:.4f} at '
           f'{got.largest[0]:.10f} Ha;')
    low = ('' if quasiparticle_weight(got.z) else
           f' Z is at or below {QP_WINDOW_Z_MIN}, no quasiparticle by '
           f'weight;')
    return (f'{what}: the followed root {got.w:.10f} Ha has Z = '
            f'{got.z:.4f};{big}{low} the branch is kept.')


def _bracketed_newton(fval, l, r, sl, tol, max_iter, to_last_bit=False):
    """The root of f in (l, r), f of sign sl at l and -sl at r, by Newton on
    the slope fval returns with f, safeguarded by the bracket: a step that
    would leave it is a bisection. An end may be a pole, where f is never
    evaluated. sl = 0 is a root at l.

    to_last_bit, for a cheap f (`PoleSum`): every second step that has not
    halved the bracket is a bisection too, and a converged step (below
    `tol`) ends the loop only strictly inside the bracket; one that rounds
    onto the end it has just moved has the root bisected to its last bit.
    Otherwise, for a costly f (`SampledEquation`, each evaluation a
    self-energy): a step that is not under half the step before the last (a
    stalled or oscillating Newton) is a bisection, Newton steps that converge
    are never interrupted, and a step below `tol` that stays in the closed
    bracket ends it (the last point evaluated is an end of the bracket, and
    the step may round to it), so a root costs a few evaluations of f."""
    if sl == 0.0:
        return float(l)
    w = 0.5 * (l + r)
    width = step = old = r - l
    for it in range(int(max_iter)):
        f, fp = fval(w)
        if f == 0.0:
            return float(w)
        if np.sign(f) == sl:
            l = w
        else:
            r = w
        dw = -f / fp
        if abs(dw) < tol and ((l < w + dw < r) if to_last_bit
                              else (l <= w + dw <= r)):
            return float(w + dw)
        mid = 0.5 * (l + r)
        if not l < mid < r:
            break                                 # bracket one ulp wide
        if to_last_bit:
            newton = l < w + dw < r and np.isfinite(dw)
            if it % 2 and r - l > 0.5 * width:
                newton = False
            if it % 2:
                width = r - l
        else:
            newton = (np.isfinite(dw) and l < w + dw < r
                      and abs(dw) <= 0.5 * abs(old))
            old, step = step, (dw if newton else mid - w)
        w = w + dw if newton else mid
    return float(w)


# ---------------------------------------------------------------------------
# Brackets of a sum of known poles
# ---------------------------------------------------------------------------

class PoleSum:
    """f(w) = w - a - sum_i num_i/(w - pol_i), its poles known: the pole
    model of `GW.sum_over_poles` (fitted amplitudes of either sign) and the
    quasi-boson supermatrix (positive weights). One window, the whole axis:
    the brackets are the intervals between neighbouring poles, every one of
    them, and the bounds on Z that `_interval_bounds` and `end_value_bounds`
    give skip those that cannot beat the best Z found.

    Between two poles of positive weight with no negative weight near enough
    to bend f, f rises from -inf to +inf and the interval holds exactly one
    root, refined by `root_between`. Any other interval (a negative weight at
    an end, or close to it) holds 0, 1, 2 or more, found by sampling it with
    the poles' known limits as nodes (`_interval_brackets`).

    prune: drop the weakest couplings while their total norm stays below tol
    (Weyl's inequality: no root moves more than tol); positive weights only.
    Zero weights are no poles of f and are dropped; with signed weights,
    coinciding poles are merged into one, whose sign sets f's limits there.
    exact: search every interval at a negative weight whole, not only those
    in the narrowest window about a (`_interval_bounds`, `exhaustive`).

    Each pole kept is labelled by its index in num as given (`label`; a
    merged pole by its first), so a walk names an interval by the two poles
    that bound it wherever they have moved (`labels_of`, `follow_root`).
    """

    def __init__(self, num, pol, a, tol=QB_QP_ROOT_TOL, prune=False,
                 exact=False):
        self.exact = exact
        num = np.asarray(num, float).ravel()
        pol = np.asarray(pol, float).ravel()
        label = np.arange(num.size)
        self.signed = bool((num < 0.0).any())
        if self.signed:
            num, pol, label = _merged_poles(num, pol, labels=True)
        else:
            if prune:
                weakest = np.argsort(num, kind='stable')
                keep = np.ones(num.size, bool)
                keep[weakest[np.cumsum(num[weakest]) <= tol * tol]] = False
                num, pol, label = num[keep], pol[keep], label[keep]
            order = np.argsort(pol, kind='stable')
            num, pol, label = num[order], pol[order], label[order]
            num, pol, label = num[num > 0.0], pol[num > 0.0], label[num > 0.0]
        self.num, self.pol, self.a, self.tol = num, pol, float(a), float(tol)
        self.label = label
        self.n = num.size
        self._bounds = None

    def windows(self, a):
        return [(-np.inf, np.inf)]

    def exhaustive(self):
        """This sum with every interval at a negative weight searched whole,
        the one bound that is not exact dropped; None if it has none."""
        if not self.signed or self.exact:
            return None
        other = PoleSum.__new__(PoleSum)
        other.__dict__.update(self.__dict__)
        other.exact, other._bounds = True, None
        return other

    def brackets(self, lo, hi):
        """Interval m = -1 .. n-1, (pol[m], pol[m+1]), and its bound on Z;
        the two unbounded outer intervals are always solved."""
        n = self.n
        bound = np.full(max(n + 1, 1), np.inf)
        if n > 1:
            bound[1:-1] = self._interval_data()[0]
        return list(range(-1, n)) or [-1], bound

    def tighten(self, brackets, batch):
        """`end_value_bounds` on the certified inner intervals of the batch."""
        ms = np.asarray(brackets)[batch]
        out = np.full(ms.size, np.inf)
        if self.n < 2:
            return out
        _, slack, certified = self._interval_data()
        inner = (ms >= 0) & (ms < self.n - 1)
        inner[inner] = certified[ms[inner]] & (slack[ms[inner]] < 1.0)
        if inner.any():
            out[inner] = end_value_bounds(
                self.num, self.pol, self.a, ms[inner],
                slack=slack[ms[inner]] if self.signed else None)
        return out

    def solve(self, m):
        """[(root, Z)] in interval m."""
        if self.n == 0:
            return [(self.a, 1.0)]
        if self._certified(m):
            w, z = root_between(self.num, self.pol, self.a, m, self.tol)
            return [(float(w), float(z))]
        return [(w, float(1.0 / secular(self.num, self.pol, self.a, w)[1]))
                for w in self._interval_roots(m)]

    def labels_of(self, m):
        """(lo, hi): the labels of the poles bounding interval m, None for
        the unbounded side of an outer one."""
        return (None if m < 0 else int(self.label[m]),
                None if m + 1 >= self.n else int(self.label[m + 1]))

    def follow_root(self, follow, labels, a):
        """(root, Z, labels, None) of valid weight nearest `follow` between the
        poles labelled `labels` -- every interval between them where poles
        have moved in -- or None where they are gone, out of order, or
        bound no root of valid weight. labels None: the interval holding
        `follow`, the branch decided by energy."""
        if self.n == 0:
            return self.a, 1.0, (None, None), None
        span = self._span(follow, labels)
        if span is None:
            return None
        got = [(w, z, m) for m in span for w, z in self.solve(m)
               if quasiparticle_weight(z, 0.0)]
        if not got:
            return None
        w, z, m = min(got, key=lambda wzm: abs(wzm[0] - follow))
        return w, z, self.labels_of(m), None

    def _span(self, follow, labels):
        """The intervals m between the poles labelled (lo, hi), or the one
        holding `follow` when labels is None; None if a label is no pole
        here or the two are out of order."""
        if labels is None:
            return [int(np.searchsorted(self.pol, follow)) - 1]
        ends = []
        for side, lab in zip((-1, self.n), labels):
            if lab is None:
                ends.append(side)
                continue
            at = np.flatnonzero(self.label == int(lab))
            if at.size != 1:
                return None
            ends.append(int(at[0]))
        if ends[0] >= ends[1]:
            return None
        return range(ends[0], ends[1])

    def _interval_data(self):
        """(bound on Z, bound on the negative weights' slope, whether one root
        is certified) of the inner intervals, computed once; intervals with a
        negative end inside the narrowest search window about a are searched
        whole (`_interval_bounds`)."""
        if self._bounds is None:
            self._bounds = _interval_bounds(
                self.num, self.pol, self.a,
                window=((-np.inf, np.inf) if self.exact
                        else _qp_search_windows(self.a)[0]))
        return self._bounds

    def _certified(self, m):
        """Whether interval m holds exactly one root, rising: both ends
        poles of positive weight (or an unbounded end), and the negative
        weights unable to turn f' to zero inside."""
        if not self.signed:
            return True
        num, pol, n = self.num, self.pol, self.n
        if m < 0 or m == n - 1:
            j = 0 if m < 0 else n - 1
            if num[j] <= 0.0:
                return False
            neg = num < 0.0
            return float((-num[neg] / (pol[neg] - pol[j]) ** 2).sum()) < 1.0
        return bool(self._interval_data()[2][m])

    def _outer_end(self, side):
        """A finite end beyond which no root lies, and where f has the sign
        of `side`: |w - a| = |sum num/(w - pol)| <= sum|num| / dist needs
        |w - a| dist <= sum|num|, false once both exceed sqrt(sum|num|)."""
        reach = np.sqrt(np.abs(self.num).sum()) * (1.0 + 1e-9) + 1e-12
        if side < 0:
            return min(self.a, self.pol[0]) - reach
        return max(self.a, self.pol[-1]) + reach

    def _interval_roots(self, m):
        """Every root in interval m, by `_interval_brackets` and
        `_bracketed_newton` to the last bit on the closed-form slope."""
        num, pol, a, n = self.num, self.pol, self.a, self.n
        fval = lambda w: secular(num, pol, a, w)
        if m >= 0:
            l, sl, spl = pol[m], -np.sign(num[m]), np.sign(num[m])
        else:
            l = self._outer_end(-1)
            sl, spl = -1.0, np.sign(fval(l)[1])
        if m + 1 < n:
            r, sr, spr = pol[m + 1], np.sign(num[m + 1]), np.sign(num[m + 1])
        else:
            r = self._outer_end(+1)
            sr, spr = 1.0, np.sign(fval(r)[1])
        k = (QP_POLE_INTERVAL_SAMPLES if 0 <= m < n - 1
             else QP_GRAPHICAL_N_OMEGA)
        x = l + (r - l) * np.arange(1, k + 1) / (k + 1)
        f, fp = _pole_sum_values(num, pol, a, x)
        x = np.concatenate([[l], x, [r]])
        sf = np.concatenate([[sl], np.sign(f), [sr]])
        sfp = np.concatenate([[spl], np.sign(fp), [spr]])
        curve = lambda w: _secular_curvature(num, pol, a, w)
        return [_bracketed_newton(fval, lb, rb, s, self.tol,
                                  QP_GRAPHICAL_MAX_BISECTION, to_last_bit=True)
                for lb, rb, s in _interval_brackets(curve, x, sf, sfp,
                                                    self.tol)]


def _interval_brackets(curve, x, sf, sfp, tol):
    """[(l, r, sign of f at l)] bracketing the roots between the nodes x,
    given the signs of f (sf) and f' (sfp) there; curve(w) -> (f, f', f'').

    A sign change of f between neighbouring nodes is a bracket. Between two
    nodes of one sign whose slopes say f turns toward zero (against the sign
    at the left, with it at the right), the turning point f' = 0 is found by
    Newton on f' safeguarded by that bracket, and where f has the other sign
    on the way the pair splits into two brackets there: the 0 or 2 crossings
    of mixed-sign amplitudes.
    """
    brackets = []
    for i in range(x.size - 1):
        l, r = x[i], x[i + 1]
        if sf[i] == 0.0:
            brackets.append((l, l, 0.0))
            continue
        if sf[i] != sf[i + 1]:
            if sf[i + 1] != 0.0:
                brackets.append((l, r, sf[i]))
            continue
        if not (sfp[i] == -sf[i] and sfp[i + 1] == sf[i]):
            continue
        w, width = 0.5 * (l + r), r - l
        for it in range(QP_GRAPHICAL_MAX_BISECTION):
            fm, fpm, fppm = curve(w)
            if np.sign(fm) != sf[i]:
                brackets += [(x[i], w, sf[i]), (w, x[i + 1], -sf[i])]
                break
            if np.sign(fpm) == sfp[i]:
                l = w
            else:
                r = w
            dw = -fpm / fppm if fppm != 0.0 else np.inf
            if abs(dw) < tol or not l < 0.5 * (l + r) < r:
                break                     # the turning point: f keeps its sign
            newton = l < w + dw < r
            if it % 2 and r - l > 0.5 * width:
                newton = False
            if it % 2:
                width = r - l
            w = w + dw if newton else 0.5 * (l + r)
    return brackets


def _secular_curvature(num, pol, a, w):
    """(f, f', f'') of the quasiparticle equation at w."""
    inv = 1.0 / (w - pol)
    r = num * inv
    ri = r * inv
    return w - a - r.sum(), 1.0 + ri.sum(), -2.0 * (ri * inv).sum()


def _merged_poles(num, pol, labels=False):
    """(num, pol) in increasing order of pol, coinciding poles summed into one
    and poles of zero weight dropped; with labels, also each pole's index in
    num as given (a merged pole's first in that order)."""
    num = np.asarray(num, float).ravel()
    pol = np.asarray(pol, float).ravel()
    order = np.argsort(pol, kind='stable')
    pol, first = np.unique(pol[order], return_index=True)
    num = np.add.reduceat(num[order], first) if num.size else num
    keep = num != 0.0
    if labels:
        return num[keep], pol[keep], order[first][keep]
    return num[keep], pol[keep]


def _pole_sum_values(num, pol, a, w):
    """(f, f') of the quasiparticle equation on the points w, in blocks of
    QB_QP_END_BLOCK_BYTES."""
    w = np.asarray(w, float)
    f, fp = np.empty(w.size), np.empty(w.size)
    rows = max(1, QB_QP_END_BLOCK_BYTES // (8 * max(pol.size, 1)))
    for r0 in range(0, w.size, rows):
        inv = 1.0 / (w[r0:r0 + rows, None] - pol[None, :])
        r = num[None, :] * inv
        f[r0:r0 + rows] = w[r0:r0 + rows] - a - r.sum(1)
        fp[r0:r0 + rows] = 1.0 + (r * inv).sum(1)
    return f, fp


def secular(num, pol, a, w):
    """(f, f') of the quasiparticle equation at w."""
    r = num / (w - pol)
    return w - a - r.sum(), 1.0 + (r / (w - pol)).sum()


def root_between(num, pol, a, m, tol):
    """(w, Z) of the one root between pol[m] and pol[m+1] (outer for m = -1, n-1).

    Each step solves a model of f that keeps the bracketing poles exactly and
    the rest of Sigma to first order about the iterate (`model_root`), so the
    singular part costs no iterations; bisection on the sign of f safeguards
    it. A root closer to a weak pole than w resolves is left with the bracket
    one ulp wide and that pole still at its edge; its Z is then taken from
    f = 0 itself, num/delta^2 = g^2/num with g = f without that pole, instead
    of from a delta that is round-off. Needs f rising from -inf to +inf across
    the interval (`PoleSum._certified`).
    """
    n = num.size
    lo = pol[m] if m >= 0 else -np.inf
    hi = pol[m + 1] if m + 1 < n else np.inf
    step = 1.0
    while not np.isfinite(lo):
        trial = hi - step
        lo = trial if secular(num, pol, a, trial)[0] < 0 else -np.inf
        step *= 2.0
    step = 1.0
    while not np.isfinite(hi):
        trial = lo + step
        hi = trial if secular(num, pol, a, trial)[0] > 0 else np.inf
        step *= 2.0
    edge = [j for j in (m, m + 1) if 0 <= j < n]
    lo0, hi0 = lo, hi
    w = 0.5 * (lo + hi)
    width = hi - lo
    while True:
        r = num / (w - pol)
        rd = r / (w - pol)
        f = w - a - r.sum()
        fp = 1.0 + rd.sum()
        if f == 0.0:
            return w, 1.0 / fp
        if f > 0:
            hi = w
        else:
            lo = w
        dw = -f / fp
        if abs(dw) < tol and lo < w + dw < hi:
            w += dw
            return w, 1.0 / secular(num, pol, a, w)[1]
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            break                                      # bracket one ulp wide
        far = r.sum() - r[edge].sum()
        far_slope = rd[edge].sum() - rd.sum()
        trial = model_root(num[edge], pol[edge], a + far - far_slope * w,
                           1.0 - far_slope, lo, hi, tol)
        # the model step only while it at least halves the bracket, so the
        # loop ends within twice the bisection count that resolves w
        take = lo < trial < hi and hi - lo <= 0.5 * width
        width = hi - lo
        w = trial if take else mid
    j = m if (m >= 0 and lo == lo0) else m + 1 if (m + 1 < n and hi == hi0) \
        else None
    if j is None:
        return w, 1.0 / secular(num, pol, a, w)[1]
    d = w - pol
    d[j] = 1.0
    r = num / d
    r[j] = 0.0
    g = w - a - r.sum()
    return w, 1.0 / (1.0 + (r / d).sum() + g * g / num[j])


def model_root(num, pol, c, slope, lo, hi, tol):
    """The root in (lo, hi) of slope*x - c - sum_j num_j/(x - pol_j), rising
    there, by Newton safeguarded by bisection; the bracketing poles' share of
    f with the rest linear in x."""
    x = 0.5 * (lo + hi)
    while True:
        g, gp = slope * x - c, slope
        for nj, pj in zip(num.tolist(), pol.tolist()):
            g -= nj / (x - pj)
            gp += nj / (x - pj) ** 2
        if g > 0:
            hi = x
        else:
            lo = x
        dx = -g / gp
        mid = 0.5 * (lo + hi)
        if abs(dx) < tol or not lo < mid < hi:
            return x + dx if lo < x + dx < hi else x
        x = x + dx if lo < x + dx < hi else mid


def end_value_bounds(num, pol, a, intervals, slack=None):
    """Upper bound on Z of the root in each of `intervals` (pol[m], pol[m+1]).

    On interval m, g(w) = f(w) + num_m/(w - pol_m) + num_m+1/(w - pol_m+1) has
    no pole and rises, so |g| >= G, its smaller end value when both ends share
    a sign (0 otherwise). At the root g is the two bracketing terms, and
    Cauchy-Schwarz on them gives 1/Z >= 1 + G^2/(num_m + num_m+1). Far from
    a zero of g the root hugs a pole with a Z orders of magnitude below the
    neighbour bound, which is most of a continuum orbital's intervals.
    An end that coincides with an excluded pole bounds nothing.

    slack: with negative weights off the interval's ends, the most they can
    lower g' there net of the positive neighbours (`_interval_bounds`); below
    one, g' >= 1 - slack still rises, and 1/Z >= 1 - slack + G^2/(...).
    """
    ends = np.unique(np.concatenate([intervals, intervals + 1]))
    value = np.empty(ends.size)
    rows = max(1, QB_QP_END_BLOCK_BYTES // (8 * pol.size))
    for r0 in range(0, ends.size, rows):
        at = ends[r0:r0 + rows]
        gap = pol[at][:, None] - pol[None, :]
        gap[np.arange(at.size), at] = np.inf           # f without its own pole
        with np.errstate(divide='ignore', invalid='ignore'):
            value[r0:r0 + rows] = pol[at] - a - (num[None, :] / gap).sum(1)
    reg = dict(zip(ends.tolist(), value.tolist()))
    lo = np.array([reg[m] for m in intervals.tolist()])
    hi = np.array([reg[m + 1] for m in intervals.tolist()])
    width = pol[intervals + 1] - pol[intervals]
    g_lo = lo - num[intervals + 1] / width              # g at pol[m]
    g_hi = hi - num[intervals] / width                  # g at pol[m+1]
    floor = np.where(g_lo > 0, g_lo, np.where(g_hi < 0, -g_hi, 0.0))
    floor[~(np.isfinite(g_lo) & np.isfinite(g_hi))] = 0.0
    lift = floor ** 2 / (num[intervals] + num[intervals + 1])
    if slack is None:
        return 1.0 / (1.0 + lift)
    return 1.0 / np.maximum(1.0, 1.0 - slack + lift)


def _interval_bounds(num, pol, a, neighbours=16, window=None):
    """(bound on Z, slack, certified) of each interval (pol[m], pol[m+1]).

    1/Z = f' = 1 + S+ - S- at the root, S+/- the sums of |num_i|/(w - pol_i)^2
    over the positive and negative weights. S+ is bounded below three ways:
    the two bracketing poles at their joint minimum, (a^1/3 + b^1/3)^3/L^2;
    `neighbours` more on each side at their farthest from the interval; and
    Cauchy-Schwarz on f(w) = 0, (w - a)^2 <= sum|num| (S+ + S-), which
    confines a large Z to near the bare level a. S- off the interval's ends is
    bounded above (the slack) by the negative neighbours at their nearest and
    every farther negative weight, block by block, at its block's nearest
    pole (`_far_slack`): f' >= 1 + S+ - slack, and >= 1 + (w - a)^2/sum|num|
    - 2 slack.

    Certified: both ends of positive weight and 1 + S+ - slack > 0 with S+ at
    its bound, so f' > 0 on the whole interval and it holds exactly one root
    (`root_between`); its bound is exact. The slack returned is net of the
    positive neighbours, what bounds g' = 1 + S+ - S- without the two
    bracketing poles from below (`end_value_bounds`).

    An interval with a negative weight at an end holds 0 or 2 roots where f'
    runs to -inf, and a root hugging that pole has Z < 0; the other is a root
    of f without the negative weights, perturbed. Inside `window` (the
    narrowest search window about a) such an interval is searched whole
    (bound one); outside it, it is bounded as that unperturbed root, the
    positive weights alone: a weak fitted amplitude raises Z only within
    about sqrt(|num|/f') of its pole, and far from the bare level that is no
    quasiparticle's place. This is the one bound that is not exact, and
    `PoleSum.exhaustive` drops it where no quasiparticle is found. An
    interval with both ends positive but f' not bounded above zero keeps its
    bound and is searched whole.
    """
    n = num.size
    width = np.diff(pol)
    pos = np.maximum(num, 0.0)
    neg = np.maximum(-num, 0.0)
    signed = bool(neg.any())
    c = np.cbrt(pos)
    slack = np.zeros(n - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (c[:-1] + c[1:]) ** 3 / width ** 2
        s_bracket = s.copy() if signed else None
        idx = np.arange(n - 1)
        for k in range(1, neighbours + 1):
            left = idx - k
            ok = left >= 0
            s[ok] += pos[left[ok]] / (pol[idx[ok] + 1] - pol[left[ok]]) ** 2
            if signed:
                slack[ok] += neg[left[ok]] / (pol[idx[ok]]
                                              - pol[left[ok]]) ** 2
            right = idx + 1 + k
            ok = right < n
            s[ok] += pos[right[ok]] / (pol[right[ok]] - pol[idx[ok]]) ** 2
            if signed:
                slack[ok] += neg[right[ok]] / (pol[right[ok]]
                                               - pol[idx[ok] + 1]) ** 2
        gap = np.maximum(np.maximum(pol[:-1] - a, a - pol[1:]), 0.0)
        if signed:
            far = _far_slack(neg, pol, neighbours)
            slack += far
            touching = (num[:-1] < 0.0) | (num[1:] < 0.0)
            lower = s - slack
            net = slack - (s - s_bracket)
            # Abel where the bound apart does not certify the interval or
            # leaves g unbounded
            weak = ~touching & ((1.0 + lower <= 0.0) | (net >= 1.0))
            if weak.any():
                abel_f, abel_g = _abel_bounds(num, pol, neighbours,
                                              np.flatnonzero(weak))
                lower[weak] = np.maximum(lower[weak], abel_f - far[weak])
                net[weak] = np.minimum(net[weak], far[weak] - abel_g)
            rising = 1.0 + lower > 0.0
            s_all = np.maximum(lower, gap ** 2 / np.abs(num).sum()
                               - 2.0 * slack)
            # an interval at a negative pole: the root of f without the
            # negative weights, outside the window (below)
            s_all[touching] = np.maximum(s[touching],
                                         gap[touching] ** 2 / pos.sum())
            bound = 1.0 / (1.0 + np.maximum(s_all, 0.0))
        else:
            s = np.maximum(s, gap ** 2 / num.sum())
            bound = 1.0 / (1.0 + s)
            rising, net = True, slack
    certified = (width > 0) & (num[:-1] > 0.0) & (num[1:] > 0.0) & rising
    if signed and window is not None:
        inside = (pol[1:] >= window[0]) & (pol[:-1] <= window[1])
        bound[touching & inside] = 1.0
    bound[~(width > 0)] = 0.0
    return bound, net, certified


def _abel_bounds(num, pol, neighbours, m=None):
    """Lower bounds on sum_i num_i/(w - pol_i)^2 over each interval (pol[m],
    pol[m+1]) from the poles within `neighbours` of it on each side: of f'-1
    (bracketing poles included) and of g'-1 (g = f without them).

    Taken a side at a time, nearest pole first, by Abel summation: with W_k
    the weight of the k nearest and d_k their distances, the side's sum is
    sum_k W_k (1/d_k^2 - 1/d_k+1^2) + W_K/d_K^2, every bracket positive, so a
    term with W_k >= 0 is bounded below at the interval's far end and one with
    W_k < 0 at its near end. A negative weight beside a stronger positive one
    then lowers the sum only by their difference, where bounding the two
    apart (`_interval_bounds`' slack) subtracts it whole. A negative
    bracketing pole is left out of f's bound (its own term runs to -inf at
    it): what remains bounds the root of f without it. m: the intervals
    to bound (all by default).
    """
    n = num.size
    m = np.arange(n - 1) if m is None else np.asarray(m)
    out_f, out_g = np.zeros(m.size), np.zeros(m.size)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for side in (-1, 1):
            near_x = pol[m] if side < 0 else pol[m + 1]
            far_x = pol[m + 1] if side < 0 else pol[m]
            first = m if side < 0 else m + 1
            for out, k0 in ((out_f, 0), (out_g, 1)):
                w = np.zeros(m.size)
                for k in range(k0, neighbours + 1):
                    i = first + side * k
                    ok = (i >= 0) & (i < n)
                    if not ok.any():
                        break
                    ii = np.clip(i, 0, n - 1)
                    add = num[ii]
                    if k == 0:
                        add = np.where(add < 0.0, 0.0, add)   # parent bound
                    w = np.where(ok, w + add, w)
                    j = first + side * (k + 1)
                    nxt = ok & (j >= 0) & (j < n) & (k < neighbours)
                    jj = np.clip(j, 0, n - 1)

                    def term(x):
                        d = np.abs(x - pol[ii])
                        t = 1.0 / d ** 2
                        far = 1.0 / np.abs(x - pol[jj]) ** 2
                        return np.where(nxt, t - far, t)
                    t_far, t_near = term(far_x), term(near_x)
                    val = np.where(w >= 0.0, w * t_far, w * t_near)
                    out += np.where(ok, np.nan_to_num(val, nan=0.0,
                                                      neginf=-np.inf), 0.0)
    return out_f, out_g


def _far_slack(neg, pol, neighbours, block=64):
    """Upper bound on sum |num_i|/(w - pol_i)^2 over the negative weights more
    than `neighbours` poles away from each interval (pol[m], pol[m+1]), for w
    in it: the poles in blocks of `block`, each block's negative weight at the
    distance of its nearest pole to the interval, or, for a block reaching
    into the neighbours, of the nearest pole beyond them. O(n^2 / block)."""
    n = pol.size
    starts = np.arange(0, n, block)
    ends = np.minimum(starts + block, n) - 1
    weight = np.add.reduceat(neg, starts)
    keep = weight > 0.0
    starts, ends, weight = starts[keep], ends[keep], weight[keep]
    out = np.zeros(n - 1)
    if not weight.size:
        return out
    rows = max(1, QB_QP_END_BLOCK_BYTES // (8 * weight.size))
    for r0 in range(0, n - 1, rows):
        m = np.arange(r0, min(r0 + rows, n - 1))[:, None]
        lo_reach, hi_reach = m - neighbours, m + 1 + neighbours
        # nearest pole of each block counted from the interval's side
        left = np.minimum(ends[None, :], lo_reach - 1)
        right = np.maximum(starts[None, :], hi_reach + 1)
        is_left = starts[None, :] < lo_reach
        is_right = ends[None, :] > hi_reach
        d_left = np.where(is_left & (left >= 0),
                          pol[m] - pol[np.clip(left, 0, n - 1)], np.inf)
        d_right = np.where(is_right & (right < n),
                           pol[np.clip(right, 0, n - 1)] - pol[m + 1], np.inf)
        with np.errstate(divide='ignore'):
            out[r0:r0 + rows] = ((weight / d_left ** 2).sum(1)
                                 + (weight / d_right ** 2).sum(1))
    return out


# ---------------------------------------------------------------------------
# Brackets of a self-energy known pointwise
# ---------------------------------------------------------------------------

class SampledEquation:
    """f(w) = w - eps - Sigma(w) for a self-energy known only pointwise:
    the contour deformation's quadrature, an analytic continuation. Its poles
    are unknown, so its brackets are the sign changes of f on a grid over each
    window of `_qp_search_windows(a)`, narrowest first, pole-aware twice over:

    - a sign change where |f| blows up under refinement is a pole of Sigma,
      not a root (`solve` returns nothing for it); and
    - a cell whose secant slope is below QP_SCAN_SECANT_MIN hides a pole
      (f rises with slope >= 1 between the poles of a positive-weight Sigma)
      and with it possibly a root with no sign change at the cell's ends, so
      it is halved until they show (`_cells`).

    fprime: f'(w) in closed form, which refines each bracket by Newton
            safeguarded by it to `tol` and gives Z = 1/f'; None bisects to
            `tol` and takes Z from a central difference (`pole_strength`).
    vectorized: func accepts and returns arrays; the grid is one call.
    windows: [(lo, hi)] scanned in place of the tiers about a.
    refusals: exception types with which func or fprime refuse a point they
        cannot evaluate (a residue backend outside its grid): such a point
        counts as unknown, no bracket ends on it, and the first refusal is
        kept (`refused`). A quasiparticle found elsewhere stands -- a root of
        Z > 1/2 is the largest of a positive-weight Sigma -- otherwise
        `find_qp_root` raises the refusal.
    positive_weight: Sigma is a sum of poles of positive weight (the contour
        deformation's), so f rises between its poles and a bracket where f
        falls (f(l) > 0 > f(r)) holds a pole: it is passed over unrefined.
        Refined, it would keep f(l) > 0 > f(r) at its ends down to one ulp,
        which only a pole does when f rises, and be dropped there (`solve`)
        after about fifty evaluations of f.
    rescan: False skips `follow_root`'s rescan of the window for the
        largest weight beside a root below QP_WINDOW_Z_MIN; the root's
        `QPRoot.largest` is then None (not searched), and no warning is
        raised for it.
    No bound on one bracket's Z is known. With positive_weight the weights
    of all roots sum to one (`weight_sum`): the grid is walked outward from
    the bare level (`outward`) and ends once the weights found leave no
    more for the rest (`_weighed_outward`), mostly at the quasiparticle a
    few cells away; otherwise every bracket of a window is solved.
    """

    def __init__(self, func, fprime=None, vectorized=False,
                 n_omega=QP_GRAPHICAL_N_OMEGA, tol=QP_GRAPHICAL_TOL,
                 max_iter=QP_GRAPHICAL_MAX_BISECTION, windows=None,
                 refusals=(), positive_weight=False, rescan=True):
        self.positive_weight, self.rescan = positive_weight, rescan
        self.weight_sum = 1.0 if positive_weight else None
        self.vectorized = vectorized
        self.n_omega, self.tol, self.max_iter = int(n_omega), tol, max_iter
        self._windows = None if windows is None else list(windows)
        self.refusals, self.refused = tuple(refusals), None
        self.func = self._guarded(func) if refusals else func
        self.fprime = (self._guarded(fprime) if refusals and fprime is not None
                       else fprime)

    def _guarded(self, g):
        """g, NaN where it refuses the point (`refusals`), the first refusal
        kept."""
        def call(w):
            try:
                return g(w)
            except self.refusals as refusal:
                if self.refused is None:
                    self.refused = refusal
                return np.nan
        return call

    def windows(self, a):
        return (_qp_search_windows(a) if self._windows is None
                else self._windows)

    def brackets(self, lo, hi):
        grid = np.linspace(lo, hi, self.n_omega)
        f = _grid_shifts(self.func, grid, self.vectorized)
        out = []
        for i in range(1, grid.size):
            out += self._cells(grid[i - 1], grid[i], f[i - 1], f[i],
                               QP_SCAN_SPLIT_DEPTH)
        return out, np.full(len(out), np.inf)

    def tighten(self, brackets, batch):
        return np.full(len(batch), np.inf)

    def outward(self, lo, hi, a, f=None):
        """The brackets of each cell of the grid `brackets` scans on [lo, hi]
        (a list per cell, as `_cells` gives them), the cells in increasing
        distance from the one holding a; f is evaluated on a grid point when
        a cell first reaches it. f: the grid's values known already ({index:
        f} or, vectorized, the whole grid's), filled in place."""
        grid = np.linspace(lo, hi, self.n_omega)
        if f is None:
            f = _grid_shifts(self.func, grid, True) if self.vectorized else {}
        value = (f.__getitem__ if self.vectorized else
                 lambda j: f[j] if j in f else f.setdefault(
                     j, float(self.func(grid[j]))))
        c0 = min(max(int(np.searchsorted(grid, a)) - 1, 0), grid.size - 2)
        for d in range(grid.size - 1):
            for c in sorted({c0 - d, c0 + d}):
                if 0 <= c < grid.size - 1:
                    yield self._cells(grid[c], grid[c + 1], value(c),
                                      value(c + 1), QP_SCAN_SPLIT_DEPTH)

    def solve(self, bracket):
        """[(root, Z)] of a bracket (l, r, f(l), f(r)); empty for a pole."""
        l, r, fl, fr = bracket
        if l == r:
            return [(float(l), self._weight(l))]
        if self.positive_weight and fl > 0.0 > fr:
            return []                         # f falls: a pole
        if self.fprime is None:
            w = _refine_root(self.func, r, l, self.tol, self.max_iter)
            fw = float(self.func(w))
        else:
            fval = lambda x: (float(self.func(x)), float(self.fprime(x)))
            w = _bracketed_newton(fval, l, r, np.sign(fl), self.tol,
                                  self.max_iter)
            fw = fval(w)[0]
        if not abs(fw) <= max(abs(fl), abs(fr)):
            return []                         # |f| grew: a pole, not a root
                                              # (or a point refused: NaN)
        return [(float(w), self._weight(w))]

    def labels_of(self, bracket):
        """No labels: the poles of a sampled self-energy are unknown."""
        return None

    def follow_root(self, follow, labels, a):
        """(root, Z, None, (w, Z) of the largest weight) of valid weight
        nearest `follow`, or None.

        On the grid of the narrowest window about a that holds `follow` (a
        wider one while the nearest root is farther than the window's edge),
        the cells are taken outward from the one holding `follow`, each
        bracketed as `brackets` brackets it and every bracket refined, until
        the next cell lies farther than the nearest root found. The grid is
        fixed by a and the window and each root refined from its own
        bracket, so the root kept does not depend on `follow`'s bits; the
        grid points are evaluated as the cells reach them.

        The largest weight: a root of Z above QP_WINDOW_Z_MIN is the largest
        of a positive-weight self-energy, the weights of its roots summing
        to one. Below, the window's other brackets are refined where the
        weight estimated at their secant root can beat the best so far
        (`_largest_beside`): with positive_weight cell by cell outward from
        a on the grid points evaluated already, until the weights refined
        leave no more for the rest; otherwise the whole window bracketed.
        Unless `rescan` is off: () then,
        not searched (`find_qp_root` records no largest weight and runs no
        search of its own)."""
        windows = self.windows(a)
        for i, (lo, hi) in enumerate(windows):
            if not lo <= follow <= hi:
                continue
            grid = np.linspace(lo, hi, self.n_omega)
            f = (_grid_shifts(self.func, grid, True) if self.vectorized
                 else {})
            value = (f.__getitem__ if self.vectorized else
                     lambda j: f[j] if j in f else f.setdefault(
                         j, float(self.func(grid[j]))))
            c0 = min(max(int(np.searchsorted(grid, follow)) - 1, 0),
                     grid.size - 2)
            solved, best = {}, None
            for d in range(grid.size - 1):
                cells = [c for c in {c0 - d, c0 + d} if 0 <= c < grid.size - 1]
                if not cells:
                    break
                near = min(max(grid[c] - follow, follow - grid[c + 1], 0.0)
                           for c in cells)
                if best is not None and near > abs(best[0] - follow):
                    break
                for c in sorted(cells):
                    for br in self._cells(grid[c], grid[c + 1], value(c),
                                          value(c + 1), QP_SCAN_SPLIT_DEPTH):
                        solved[br] = self.solve(br)
                        for w, z in solved[br]:
                            if quasiparticle_weight(z, 0.0) and (
                                    best is None or abs(w - follow)
                                    < abs(best[0] - follow)):
                                best = (w, z)
            if best is None:
                continue
            # a root beyond the edge may be nearer: widen first
            if (abs(best[0] - follow) <= min(follow - lo, hi - follow)
                    or i == len(windows) - 1):
                largest = best
                if not quasiparticle_weight(best[1]):
                    cells = (self.outward(lo, hi, a, f)
                             if self.weight_sum is not None
                             else [self.brackets(lo, hi)[0]])
                    largest = (self._largest_beside(cells, solved, best)
                               if self.rescan else ())
                return best[0], best[1], None, largest
        return None

    def _largest_beside(self, cells, solved, best):
        """(w, Z) of the largest valid weight among the brackets of `cells`
        (lists of brackets): those in `solved` ({bracket: roots}) as found,
        every other refined where the weight estimated at its secant root
        (1/f' there) exceeds the largest so far. With `weight_sum`, the
        cells end once the weights refined, plus QP_WEIGHT_SUM_SLACK, leave
        no more of that sum than the largest of them."""
        top, weighed = best, 0.0
        for got in solved.values():
            for w, z in got:
                if quasiparticle_weight(z, 0.0):
                    weighed += z
                    if z > top[1]:
                        top = (w, z)
        for brackets in cells:
            if self.weight_sum is not None and (
                    self.weight_sum - weighed + QP_WEIGHT_SUM_SLACK
                    <= top[1]):
                break
            for l, r, fl, fr in brackets:
                if (l, r, fl, fr) in solved:
                    continue
                x = l if l == r else l - fl * (r - l) / (fr - fl)
                z = self._weight(x)
                if not quasiparticle_weight(z, 0.0) or z <= top[1]:
                    continue
                for w, z in self.solve((l, r, fl, fr)):
                    if quasiparticle_weight(z, 0.0):
                        weighed += z
                        if z > top[1]:
                            top = (w, z)
        return top

    def _cells(self, l, r, fl, fr, depth):
        """The brackets (l, r, f(l), f(r)) in [l, r]: the cell itself on a
        sign change; halves of it while it rises too slowly to be free of a
        pole (secant slope below QP_SCAN_SECANT_MIN), or while one end is a
        refused point and the other is not (toward the edge of where f is
        known), at most `depth` times."""
        if fl * fr < 0.0:
            return [(l, r, fl, fr)]
        edge = np.isnan(fl) != np.isnan(fr)
        if depth == 0 or fl == 0.0 or fr == 0.0 or not (
                edge or fr - fl < QP_SCAN_SECANT_MIN * (r - l)):
            return []
        m = 0.5 * (l + r)
        fm = float(self.func(m))
        return (self._cells(l, m, fl, fm, depth - 1)
                + self._cells(m, r, fm, fr, depth - 1))

    def _weight(self, w):
        if self.fprime is None:
            return float(pole_strength(self.func, w))
        fp = float(self.fprime(w))
        return 1.0 / fp if fp != 0.0 else np.nan


def _qp_search_windows(eigKS):
    """Scan windows of a self-energy known pointwise, narrowest first: the
    tier of the state's depth, then every wider tier."""
    tier = 0
    for k, (depth, _) in enumerate(QP_SEARCH_WINDOW_TIERS):
        if abs(eigKS) > depth:
            tier = k
    return [(eigKS - half, eigKS + half)
            for _, half in QP_SEARCH_WINDOW_TIERS[tier:]]


def _grid_shifts(func, omega_grid, vectorized):
    """Evaluate f on the search grid, scalar loop or one vectorized call.

    Parameters
    ----------
    func : callable
        f(w) = w - eps - Sigma(w); scalar-to-scalar, or (if `vectorized`)
        ndarray-to-ndarray of the same shape as `omega_grid`.
    omega_grid : ndarray, shape (nOmega,)
        Frequencies to evaluate f at.
    vectorized : bool
        If True, call func once on the whole grid; otherwise loop over it.

    Returns
    -------
    shifts : ndarray, shape (nOmega,)
        f(omega_grid), evaluated elementwise.
    """
    if not vectorized:
        return np.array([func(w) for w in omega_grid])
    shifts = np.asarray(func(omega_grid), dtype=float)
    if shifts.shape != omega_grid.shape:
        raise ValueError(
            f'vectorized func must return shape {omega_grid.shape}, '
            f'got {shifts.shape}')
    return shifts


def _refine_root(func, a, b, tol, max_bisection):
    """Bisect a bracketing interval [a, b] down to `tol`; f(a) is
    evaluated once per move."""
    fa = func(a)
    for _ in range(max_bisection):
        c = 0.5 * (a + b)
        fc = func(c)
        if fa * fc <= 0.0:
            b = c
        else:
            a = c
            fa = fc
        if abs(b - a) <= tol:
            break
    return 0.5 * (a + b)


def pole_strength(func, w, h=QP_Z_DERIV_STEP):
    """Z = 1/(1 - dSigma/dw) at w, for func(w) = w - eps - Sigma(w).

    f'(w) = 1 - dSigma/dw, so Z is just 1/f'(w) and no separate Sigma evaluation
    (or self-energy object) is needed.
    """
    deriv = (func(w + h) - func(w - h)) / (2.0 * h)
    if deriv == 0.0 or not np.isfinite(deriv):
        return np.nan
    return 1.0 / deriv


# ---------------------------------------------------------------------------
# The routes' entry points, each one build of `find_qp_root`
# ---------------------------------------------------------------------------


def solve_qp_equation_pole_strength(func, eigKS, tol=QP_GRAPHICAL_TOL, nOmega=QP_GRAPHICAL_N_OMEGA,
                                    max_bisection=QP_GRAPHICAL_MAX_BISECTION,
                                    z_min=QP_WINDOW_Z_MIN,
                                    return_diagnostics=False,
                                    refuse_unconverged=False,
                                    full_output=False, vectorized=False,
                                    fprime=None, follow=None, out=None):
    """Solve f(w) = w - eps - Sigma(w) = 0, returning the root with the largest
    quasiparticle weight Z = 1/f'(w) rather than the one nearest eigKS:
    `find_qp_root` on a `SampledEquation`.

    For a shallow state these agree, but a deep valence or semicore state has
    low-weight satellite crossings sitting *between* eps_HF and the true QP pole,
    and the nearest-root rule then returns a satellite whose Z is ~0.03 --
    numerically a root, physically not the quasiparticle. Every root in the
    window of the state's depth is found and the one of largest valid weight
    (0 < Z <= 1) kept; while that weight fails `quasiparticle_weight(Z, z_min)`
    the scan widens through the wider tiers of QP_SEARCH_WINDOW_TIERS. A state
    whose best root never passes keeps the largest-weight root of the widest
    window (nearest eigKS if no weight is valid), flagged False under
    full_output: a high virtual often has no quasiparticle, and that is no
    failure. A widest window with no root at all: refused under
    refuse_unconverged, NaN with a warning otherwise.

    fprime: f'(w) in closed form (refinement by bracketed Newton, exact Z);
    follow: `find_qp_root`'s, a walk's previous root followed (a warning
    where the largest weight has left it, `branch_warning`); out: a dict
    that receives its branch instead (`_note_branch`: 'flipped' is the
    named flag a record copies).
    return_diagnostics=True additionally returns the full list of
    (root, Z) pairs found, for inspecting the satellite structure;
    full_output=True appends whether the returned root is a quasiparticle.
    """
    source = SampledEquation(func, fprime=fprime, vectorized=vectorized,
                             n_omega=nOmega, tol=tol, max_iter=max_bisection)
    got = find_qp_root(source, eigKS, follow=follow, z_min=z_min,
                       refuse_unconverged=refuse_unconverged)
    _note_branch(got, '', tol, out)
    res = (got.w, list(got.found)) if return_diagnostics else (got.w,)
    res = res + (got.quasiparticle,) if full_output else res
    return res if len(res) > 1 else res[0]


def solve_qp_equation_scan(residual, fprime, a, follow=None,
                           tol=QP_ROOT_TOL,
                           max_iter=QP_ROOT_MAX_ITER, label='',
                           refusals=(), out=None, positive_weight=False,
                           rescan=True):
    """(w*, Z) of residual(w) = w - a - Sigma(w) = 0 for a self-energy known
    pointwise with a closed-form slope, fprime(w) = 1 - dSigma/dw: the
    contour deformation's equation. `find_qp_root` on a `SampledEquation`
    scanned about the bare level a, each bracket refined by Newton
    safeguarded by it to `tol`, Z = 1/fprime at the root: the exact
    derivative of the equation solved. No root at all is (NaN, NaN) with a
    `QPNoRoot` warning and 'no_root' in `out`; refusals: `SampledEquation`'s,
    the exceptions that mark a point unevaluable.
    follow: a walk's previous root, (w, None) or w (`find_qp_root`); out: a
    dict that receives its branch (`_note_branch`), else a warning where the
    largest weight has left it. positive_weight: `SampledEquation`'s, a
    bracket where f falls is a pole of Sigma and is not refined; rescan:
    `SampledEquation`'s, False reports no largest weight beside a followed
    root below QP_WINDOW_Z_MIN."""
    source = SampledEquation(residual, fprime=fprime, tol=tol,
                             max_iter=max_iter, refusals=refusals,
                             positive_weight=positive_weight, rescan=rescan)
    got = find_qp_root(source, a, follow=follow, label=label)
    _note_branch(got, label, tol, out)
    return float(got.w), float(got.z)


def solve_qp_equation_poles(num, pol, a, tol=QP_ROOT_TOL, label='',
                            follow=None, out=None):
    """(w*, Z) of f(w) = w - a - sum_i num_i/(w - pol_i) = 0, the root of
    largest weight, for weights num of either sign: `find_qp_root` on a
    `PoleSum`, every interval between the poles a bracket, those no bound
    lets beat the best Z found skipped. With `follow` = (w, labels), the
    root continuing the previous geometry's, between the same labelled poles
    (`find_qp_root`); out: a dict that receives its branch and the labels of
    the root kept (`_note_branch`), else a warning where the largest weight
    has left it. Each root costs O(poles) per
    refinement step, never the (poles + 1)^3 of an eigensolve of the
    arrowhead matrix.

    The largest valid weight is kept (the root nearest a if none is valid);
    the caller reads from Z whether it is a quasiparticle. No real root at all
    is (NaN, NaN) with a `QPNoRoot` warning and 'no_root' in `out`.
    """
    got = find_qp_root(PoleSum(num, pol, a, tol), a, follow=follow,
                       label=label)
    _note_branch(got, label, tol, out)
    return float(got.w), float(got.z)


def largest_weight_root(num, pol, a, label='', tol=QB_QP_ROOT_TOL, out=None):
    """(w*, Z) of f(w) = w - a - sum_i num_i/(w - pol_i) = 0 with the largest Z,
    for positive weights (the quasi-boson supermatrix): `find_qp_root` on a
    `PoleSum` that drops the weakest couplings while their norm stays below
    `tol`, which by Weyl's inequality moves no root by more than `tol`.

    f rises from -inf to +inf between neighbouring poles, so each interval
    holds one root, Z = 1/f'(w) there, and the weights of all roots sum to
    one. Most intervals are excluded without a solve by `_interval_bounds`,
    most of the rest by `end_value_bounds`; the survivors are solved in
    decreasing order of their bound until no bound exceeds the best Z found.
    Exact: no bound under-estimates.

    Every root of a positive-weight self-energy has 0 < Z <= 1; one outside
    is a numerical event, kept with a `QPWeightOutside` warning and the named
    flag 'z_outside' (`find_qp_root`), and 'z_outside' True in `out` (a dict,
    when given), as is no root at all (NaN, `QPNoRoot`, 'no_root').
    """
    got = find_qp_root(PoleSum(num, pol, a, tol, prune=True), a,
                       label=label or 'the secular equation')
    if out is not None:
        out.update(z_outside=z_outside(got), no_root=not np.isfinite(got.w))
    return float(got.w), float(got.z)


def bracketed_root(num, pol, a, w0, labels=None, tol=QB_QP_ROOT_TOL, out=None):
    """(w, Z) of the root of f(w) = w - a - sum_i num_i/(w - pol_i) = 0 that
    a reference seed names, decided exactly from the poles (`find_qp_root`
    with `follow`): between the poles labelled `labels` (`bracket_labels` of
    the reference root) wherever they have moved, or, labels None, in the
    interval holding w0. A warning where the largest weight has left it;
    out: a dict that receives its branch instead (`_note_branch`,
    'flipped')."""
    got = find_qp_root(PoleSum(num, pol, a, tol), a, follow=(w0, labels))
    _note_branch(got, '', tol, out)
    return float(got.w), float(got.z)


def bracket_labels(num, pol, a, w):
    """(lo, hi): the labels of the poles bounding the root w of the pole sum
    (`PoleSum.labels_of`), which `bracketed_root` follows at other
    geometries; None for the unbounded side of an outer interval."""
    poles = PoleSum(num, pol, a)
    return poles.labels_of(int(np.searchsorted(poles.pol, w)) - 1)


def solve_qp_hamiltonian(newton_step, seed, tol=QP_SUBSPACE_TOL,
                         max_iter=QP_SUBSPACE_MAX_ITER):
    """(H, iterations, converged) of the matrix quasiparticle equation
    H = e0 + Sigma^c[H] on a subspace, by Newton from `seed`.

    newton_step(H) -> (H after one Newton step, max |residual| at H); the step
    that brings the residual to tol is taken and the iteration stops. The
    matrix equation has no bracket; the seed is the frozen reference
    Hamiltonian, whose diagonal the scalar roots of `find_qp_root` set. The
    caller decides what an unconverged H means.
    """
    h = np.array(seed, float)
    for it in range(1, int(max_iter) + 1):
        h, res = newton_step(h)
        if res <= tol:
            return h, it, True
    return h, int(max_iter), False
