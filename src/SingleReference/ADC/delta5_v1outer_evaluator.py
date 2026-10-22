"""PRODUCTION (rank<=4) evaluator for the OUTER v=1 singles rows
-- the diagram side of the X(0)+X(3) group.

SCOPE (the cancellation economy, not an approximation): the eight
all-w3/w5 signatures

    w3.w3.w3.w3  w3.w3.w5.w3  w3.w5.w3.w3  w3.w5.w5.w3
    w3.w3.w3.f4  w3.w3.w5.f4  w3.w5.w3.f4  w3.w5.w5.f4

are EXACTLY equal, at the symmetrized level, to their resolved-
manifold ISR sub-pieces (gates 1e-13..1e-15 at both
geometries), so they cancel identically in the physical group and are
never evaluated here -- their ISR partners are dropped with them.
`SIG_EXACT_VS_ISR` names them; `needed_signatures()` returns the rest
(13 exotic w7 families + 18 free-sector families, 8240 orderings).
This is the same accounting that reduced the v>=2 groups to brackets
B1-B5; it is also what removes every rank-8 denominator from the
problem (all four rank-8 entries live
in the dropped set).

PER-CUT DISPATCH (census-driven, all four mechanisms pre-existing):
  cut3 free (f4)      -> ABSORB: a free 4-line cut3 is always exactly
                         the last vertex's own four lines (every line
                         of the last vertex crosses it, and self-loops
                         do not exist), so 1/E3 folds elementwise into
                         that vertex's factor -- no operand, no grid.
  netted denominator
    identically zero  -> EXCLUDED (that ordering sits in a deeper
                         v>=2 coincidence class; its v=1 residue is 0).
    rank <= 4         -> DENSE operand, 1/D masked to 0 where |D|<ZERO
                         (that mask IS the v=1 off-pole selection);
                         any sign -- plain division, no grid.
    definite          -> LAPLACE, 1/D = -sum_k w_k e^{D tau_k}
                         (D<0), exponentials factorize per label onto
                         the owning vertex factor.  At most TWO grids
                         are ever simultaneously live (measured).
    else              -> DENSE with BATCHING: loop over the excess
                         axes (worst case 2) so the operand is rank<=4
                         inside the loop.
Every intermediate is rank<=4, asserted by the contraction planner.

Gate: an external gate script (not in this repository) -- vs the dense
prototype rows (delta5_f_v1outer_{r1,r1.6}.npz) restricted to the same
signature set, and vs the cached fine2 pair target, both geometries.
"""
from itertools import combinations, product

import numpy as np

from src.Base.pdaggerq_interface import OCC_LETTERS
from src.Base.utils.grids import minimax_tau_supported_sizes
from src.CodeGeneration.diagrammatic import goldstone as GS
from src.CodeGeneration.diagrammatic.emit import (
    _du3_strip_factor)
from src.CodeGeneration.diagrammatic.skeleton_enum import (
    enumerate_skeletons)
from src.SingleReference.ADC.delta5_w77_evaluator import _contraction_order
from src.SingleReference.ADC.delta5_w37_evaluator import (
    _good_bounds, _net, tau_grid_good)

ZERO = 1e-8

#: signatures whose symmetrized diagram content equals its resolved
#: ISR sub-piece exactly -- dropped from BOTH sides
SIG_EXACT_VS_ISR = frozenset({
    'w3.w3.w3.w3', 'w3.w3.w5.w3', 'w3.w5.w3.w3', 'w3.w5.w5.w3',
    'w3.w3.w3.f4', 'w3.w3.w5.f4', 'w3.w5.w3.f4', 'w3.w5.w5.f4'})


def _grid_escalate(eps, nocc, dlists, ntau):
    """Smallest tabulated grid >= ntau whose own self-gate passes.

    The definite-denominator span is geometry dependent -- at H4 r=1.6
    it is [0.0146, 2.83] (ratio ~194) where ntau=14 fails its gate at
    3.2e-07.  Escalating is safe precisely BECAUSE `tau_grid_good`
    self-gates: an accepted grid has certified accuracy, never an
    assumed one.  Returns (taus, wts, ntau_used)."""
    sizes = [n for n in minimax_tau_supported_sizes() if n >= ntau]
    last = None
    for n in sizes:
        try:
            taus, wts = tau_grid_good(eps, nocc, dlists, n)
            return taus, wts, n
        except ValueError as exc:
            last = exc
    raise ValueError(f"no tabulated minimax grid >= {ntau} passes its "
                     f"self-gate; last failure: {last}")


def _plan_multibatch(subs, out, cap=4):
    """(plan, batch_chars): the fewest non-output chars to loop over so
    that every operand is rank<=cap after their removal AND a rank<=cap
    pairwise plan exists.  Generalizes delta5_w77_evaluator's
    single-char `_plan_with_batching` (needed depth here is <=2)."""
    inner = sorted({c for s in subs for c in s} - set(out))
    for nb in range(len(inner) + 1):
        for combo in combinations(inner, nb):
            subs2 = [''.join(c for c in s if c not in combo) for s in subs]
            if any(len(s) > cap for s in subs2):
                continue
            plan = _contraction_order(subs2, out, cap)
            if plan is not None:
                return plan, combo
    raise AssertionError(f"no rank<={cap} plan even with batching: {subs}")


def needed_signatures(eps, nocc):
    """The v=1-position-0 signatures this evaluator covers."""
    return sorted({s for s, _ in _enumerate(eps, nocc, sig_only=True)})


def _enumerate(eps, nocc, sig_only=False):
    """Walk every v=1-position-0 ordering of a NEEDED signature,
    yielding (signature, spec) with the per-cut dispatch resolved."""
    eps = np.asarray(eps, float)
    for block in GS._BLOCKS:
        for skel in enumerate_skeletons(5, block):
            for ordering in skel.orderings:
                cuts = GS._ordering_cuts(skel, ordering)
                d = [c[0] for c in cuts]
                nl = [len(c[1]) + len(c[2]) for c in cuts]
                if d[0] == 0 or nl[0] != 3:
                    continue
                sig = '.'.join(('w' if d[i] else 'f') + str(nl[i])
                               for i in range(4))
                if sig in SIG_EXACT_VS_ISR:
                    continue
                if sig_only:
                    yield sig, None
                    continue
                sec = GS.ordering_sector(skel, ordering)
                w = (skel.coeff / skel.n_ext) * GS.ordering_sign(skel, sec)
                vstar, vlast = ordering[0], ordering[-1]
                strip_lbl = (skel.bra_label if vstar == skel.bra_vertex
                             else skel.ket_label)
                ext_lbl = (skel.ket_label if vstar == skel.bra_vertex
                           else skel.bra_label)
                ext_fam = (block[1] if vstar == skel.bra_vertex
                           else block[0])
                xi, cfg_lbls = _du3_strip_factor(
                    skel.factors[vstar], strip_lbl, sec)
                cfg_kinds = tuple(l in OCC_LETTERS for l in cfg_lbls)

                rest = sorted({ln.idx for ln in skel.lines} - set(cfg_lbls))
                sub = {cfg_lbls[t]: chr(97 + t) for t in range(3)}
                sub[ext_lbl] = 'd'
                for i, l in enumerate(rest):
                    sub[l] = chr(101 + i)
                facs, vlast_fi = [], None
                for vv, (cr, an) in enumerate(skel.factors):
                    if vv == vstar:
                        continue
                    ls = list(cr) + list(an)
                    if vv == vlast:
                        vlast_fi = len(facs)
                    facs.append((''.join(sub[l_] for l_ in ls),
                                 tuple(l_ in OCC_LETTERS for l_ in ls)))

                def dl(ci, minus0):
                    out = []
                    for sgn, ls in ((+1.0, cuts[ci][1]), (-1.0, cuts[ci][2])):
                        for l_ in ls:
                            out.append((sub[l_], l_ in OCC_LETTERS, sgn))
                    if minus0:
                        for sgn, ls in ((-1.0, cuts[0][1]),
                                        (+1.0, cuts[0][2])):
                            for l_ in ls:
                                out.append((sub[l_], l_ in OCC_LETTERS, sgn))
                    return out

                # cut3 free == the last vertex's own four lines -> absorb
                absorb = None
                if d[3] == 0:
                    cut3_lbls = set(cuts[3][1]) | set(cuts[3][2])
                    vl_cr, vl_an = skel.factors[vlast]
                    assert set(list(vl_cr) + list(vl_an)) == cut3_lbls, \
                        f"free cut3 != last vertex's lines: {skel.raw}"
                    absorb = _net(dl(3, False))

                dense, lap, dead = [], [], False
                for i in (1, 2, 3):
                    if i == 3 and d[3] == 0:
                        continue                       # absorbed above
                    nd = _net(dl(i, d[i] != 0))
                    if not nd:
                        # omega cut pinned to the same pole -> the v=1
                        # residue of this ordering is identically zero
                        assert d[i] != 0, f"free cut nets to zero: {skel.raw}"
                        dead = True
                        break
                    axes = [(ch, isocc, cf) for (ch, isocc), cf in nd.items()]
                    lo, hi = _good_bounds(
                        [(ch, io, c) for ch, io, c in axes], eps, nocc)
                    # _good_bounds returns the span of -D; definite iff
                    # it stays strictly positive (i.e. D strictly < 0)
                    if len(axes) > 4 and lo > 0.0:
                        lap.append([(ch, io, c) for ch, io, c in axes])
                    else:
                        dense.append(axes)
                if dead:
                    continue

                subs_all = [f[0] for f in facs] + \
                    [''.join(sorted(ch for ch, _, _ in ax)) for ax in dense]
                plan, batch = _plan_multibatch(subs_all, 'abcd')
                kindof = {}
                for s_, kinds in facs:
                    for ax, ch in enumerate(s_):
                        kindof.setdefault(ch, kinds[ax])
                yield sig, dict(
                    facs=facs, vlast_fi=vlast_fi, absorb=absorb,
                    dense=dense, lap=lap, plan=plan, batch=batch,
                    sec=sec, cfg_kinds=cfg_kinds, ext_fam=ext_fam,
                    kindof=kindof, pref=w * xi * d[0])


def _axis_array(axes, eps, nocc, norb, order, bound=None):
    """Dense array of sum_i coeff_i eps(label_i), laid out over the
    chars of `order`.  Chars listed in `bound` (char -> relative index
    within their own occ/vir slice) contribute a SCALAR instead of an
    axis -- that is what keeps a batched operand rank<=4: the full
    high-rank array is never built, only its slice."""
    sl = {True: slice(0, nocc), False: slice(nocc, norb)}
    bound = bound or {}
    kind = {ch: isocc for ch, isocc, _ in axes}
    free = [ch for ch in order if ch not in bound]
    pos = {ch: i for i, ch in enumerate(free)}
    shape = [len(eps[sl[kind[ch]]]) for ch in free]
    D = np.zeros(shape if shape else [])
    for ch, isocc, coeff in axes:
        if ch in bound:
            D = D + coeff * eps[sl[isocc]][bound[ch]]
        else:
            sh = [1] * len(free)
            sh[pos[ch]] = -1
            D = D + coeff * eps[sl[isocc]].reshape(sh)
    return D, ''.join(free)


def v1outer_rows(eps, g, nocc, ntau=14, report=False):
    """(rows_p, rows_m): rank<=4 OUTER v=1 singles rows over the needed
    signature set.  Row normalization matches the dense prototype (the dense prototype): symfold at 1/2 gives the p0+p3 sum."""
    eps = np.asarray(eps, float)
    g = np.asarray(g)
    norb = len(eps)
    nvir = norb - nocc
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}

    specs = [sp for _, sp in _enumerate(eps, nocc)]
    all_lap = [dlv for sp in specs for dlv in sp['lap']]
    if all_lap:
        taus, wts, ntau = _grid_escalate(eps, nocc, all_lap, ntau)
    else:
        taus, wts = np.zeros(0), np.zeros(0)
    if report:
        ng = {}
        for sp in specs:
            ng[len(sp['lap'])] = ng.get(len(sp['lap']), 0) + 1
        nb = {}
        for sp in specs:
            nb[len(sp['batch'])] = nb.get(len(sp['batch']), 0) + 1
        print(f"v1outer: {len(specs)} orderings; grids/ordering {ng}; "
              f"batch-depth {nb}; ntau={ntau}", flush=True)

    grids = {+1: None, -1: None}
    for sp in specs:
        base = []
        for fi, (s_, kinds) in enumerate(sp['facs']):
            op = g[tuple(sl[k] for k in kinds)]
            if fi == sp['vlast_fi'] and sp['absorb'] is not None:
                E3, _ = _axis_array(
                    [(ch, io, c) for (ch, io), c in sp['absorb'].items()],
                    eps, nocc, norb, s_)
                op = op / E3
            base.append(op)
        subs_f = [s_ for s_, _ in sp['facs']]
        dense_orders = [''.join(sorted(ch for ch, _, _ in ax))
                        for ax in sp['dense']]

        owner = {}
        for fi, (s_, kinds) in enumerate(sp['facs']):
            for ax, ch in enumerate(s_):
                owner.setdefault((ch, kinds[ax]), (fi, ax))

        nlap = len(sp['lap'])
        acc = None
        for kcombo in (product(range(ntau), repeat=nlap) if nlap
                       else [()]):
            wfac = sp['pref']
            texp = {}
            for gi, k in enumerate(kcombo):
                wfac = wfac * (-wts[k])
                for ch, isocc, cf in sp['lap'][gi]:
                    texp[(ch, isocc)] = texp.get((ch, isocc), 0.0) \
                        + cf * taus[k]
            ops = list(base)
            for key, tv in texp.items():
                if tv == 0.0:
                    continue
                fi, ax = owner[key]
                s_, kinds = sp['facs'][fi]
                sh = [1] * len(s_)
                sh[ax] = -1
                ops[fi] = ops[fi] * np.exp(
                    tv * eps[sl[kinds[ax]]]).reshape(sh)

            def run(subs_l, ops_l):
                cur = {i: (subs_l[i], ops_l[i]) for i in range(len(ops_l))}
                nxt = len(ops_l)
                for (i, j, res) in sp['plan']:
                    si, ai = cur[i]
                    sj, aj = cur[j]
                    assert len(res) <= 4
                    cur[nxt] = (res, np.einsum(f"{si},{sj}->{res}",
                                               ai, aj, optimize=True))
                    nxt += 1
                s_fin, a_fin = cur[nxt - 1]
                return np.einsum(f"{s_fin}->abcd", a_fin)

            def build(bound):
                """factor + denominator operands with `bound` chars
                fixed; every returned operand is rank<=4."""
                subs_l, ops_l = [], []
                for s_, op_ in zip(subs_f, ops):
                    s2 = s_
                    for ch, vb in bound.items():
                        while ch in s2:
                            op_ = np.take(op_, vb, axis=s2.index(ch))
                            s2 = s2.replace(ch, '', 1)
                    subs_l.append(s2)
                    ops_l.append(op_)
                for axes, order in zip(sp['dense'], dense_orders):
                    D, red = _axis_array(axes, eps, nocc, norb, order,
                                         bound=bound)
                    bad = np.abs(D) < ZERO
                    subs_l.append(red)
                    ops_l.append(np.where(bad, 0.0,
                                          1.0 / np.where(bad, 1.0, D)))
                assert all(len(s_) <= 4 for s_ in subs_l)
                return subs_l, ops_l

            if not sp['batch']:
                val = run(*build({}))
            else:
                dims = [nocc if sp['kindof'][ch] else nvir
                        for ch in sp['batch']]
                val = 0.0
                for vals in product(*[range(n) for n in dims]):
                    val = val + run(*build(dict(zip(sp['batch'], vals))))
            acc = wfac * val if acc is None else acc + wfac * val

        sec = sp['sec']
        if grids[sec] is None:
            od = nocc if sp['cfg_kinds'][0] else nvir
            pd = nvir if sp['cfg_kinds'][0] else nocc
            grids[sec] = np.zeros((od, pd, pd, norb))
        colsl = o if sp['ext_fam'] == 'o' else v
        grids[sec][:, :, :, colsl] += acc

    def pack(T, sector):
        A = T - T.transpose(0, 2, 1, 3)
        rows = []
        if sector == +1:
            for j in range(nocc):
                for k in range(nvir):
                    for l_ in range(k + 1, nvir):
                        rows.append((-1.0) ** j * A[j, k, l_])
        else:
            for j in range(nvir):
                for k in range(nocc):
                    for l_ in range(k + 1, nocc):
                        rows.append((-1.0) ** (k + l_ + nocc) * A[j, k, l_])
        return np.array(rows)

    Tp, Tm = grids[+1], grids[-1]
    rp = pack(Tp, +1) if Tp is not None else np.zeros(0)
    rm = pack(Tm, -1) if Tm is not None else np.zeros(0)
    return rp, rm
