"""PRODUCTION evaluator for the w57/w75 survivor rows (order-5
ingredients #2/#3).

Same mechanism as delta5_w77_evaluator.py's w3.w7.w7.w3 class, applied
to the two remaining fully-definite X(0,3) signatures: w3.w5.w7.w3
((X,Y)=(5,7)) and w3.w7.w5.w3 ((X,Y)=(7,5)).  A per-(X,Y) shell-definiteness scan found that D1=E1-E0 and
D2=E2-E0 are BOTH strictly negative on-shell whenever the corresponding
cut's linecount is in {5,7} (a genuine k2-or-deeper intermediate is
always energetically "further" than the config's own k1 energy -- the
w77 shell mechanism) -- true for (5,5),(5,7),(7,5),(7,7) alike, and
gated exact (front-strip/2 == cached d_..._diag at 5-6e-18, both
geometries) with NO mirror-pairing needed.  Only signatures touching a
3-line rest cut ((3,3),(3,5),(3,7),(5,3),(7,3)... i.e. any (X,Y) with a
3 in it) are genuinely indefinite (comparing the config's OWN labels to
a freely-summed same-kind rest pair -- no shell/gap argument applies)
and remain open (w37/w73, m35/m53).

    rows[A, q] = 1/2 sum_{orderings, sigma} w xi
                 * sum_inner N_sigma(A, q, inner)
                 * (-1/(d0 d3)) [ d1/(D1^2 D2) + d2/(D2^2 D1) ],

    1/(Da^2 Db) = sum_{k,l} w_k tau_k w_l e^{Da tau_k + Db tau_l}
    (Da, Db strictly negative; repeated-denominator rule, adc_codegen.tex).

Generalizes w77's grid (hardcoded to a "2h2p" span) to an ARBITRARY
D-list energy span computed directly from the label/sign composition
(conservative bound: sums each label's independent min/max
contribution, which over-covers the true range whenever a label is
shared between cuts[ci] and cuts[0] and partially cancels -- safe, and
self-gated exactly like tau_grid_2h2p so an inadequate ntau raises
rather than silently under-converging).

Reuses delta5_w77_evaluator's rank<=4 contraction-search machinery
unchanged (_contraction_order/_plan_with_batching) -- single source of
truth for that search, not duplicated.

Gate: an external gate script (not in this repository) -- machine agreement
vs the dense front-strip prototype (the dense prototype) and the cached
delta5_x03_{r}.npz Laurent tables, both geometries.  ALSO regression-
gates (X,Y)=(7,7) against delta5_w77_evaluator.w77_rows_laplace exactly
(same physics, independently generalized code path).
"""
from itertools import permutations

import numpy as np

from src.Base.pdaggerq_interface import OCC_LETTERS
from src.Base.utils.grids import minimax_time_grid
from src.CodeGeneration.diagrammatic import goldstone as GS
from src.CodeGeneration.diagrammatic.emit import (
    _du3_strip_factor)
from src.CodeGeneration.diagrammatic.skeleton_enum import (
    enumerate_skeletons)
from src.SingleReference.ADC.delta5_w77_evaluator import (
    _contraction_order, _plan_with_batching)


def _energy_bounds(dlist, eps, nocc):
    """Conservative [min, max] span of D = sum_i sgn_i * eps(label_i).

    A label can appear TWICE in dlist (once from cuts[ci], once from
    the -cuts[0] flip) when the same physical line crosses both cuts;
    those entries must be NET-SUMMED per (axis, kind) key FIRST (same
    accumulation the evaluator's own `texp` dict performs) before
    taking independent per-key extremes -- treating raw (unmerged)
    entries as independent overestimates the span and can even cross
    zero for an exactly-cancelling shared label (measured: naive
    per-entry bound gave [-4.36, 0.7], crossing zero, for a signature
    whose true D is strictly negative)."""
    eo = np.sort(np.asarray(eps, float)[:nocc])
    ev = np.sort(np.asarray(eps, float)[nocc:])
    net = {}
    for ch, isocc, sgn in dlist:
        key = (ch, isocc)
        net[key] = net.get(key, 0.0) + sgn
    lo = hi = 0.0
    for (ch, isocc), c in net.items():
        if c == 0.0:
            continue
        e = eo if isocc else ev
        if c > 0:
            lo += c * e[0]
            hi += c * e[-1]
        else:
            lo += c * e[-1]
            hi += c * e[0]
    return lo, hi


def tau_grid_general(eps, nocc, dlists, ntau, gate=1e-8):
    """Self-gated minimax grid spanning the union of every dlist's
    conservative energy bound (one shared grid for D1 AND D2, both
    signatures).

    minimax_time_grid represents 1/x for x > 0 (GreenX convention,
    matching tau_grid_2h2p's positive `2*ev-2*eo`-type bounds); the
    evaluator's own 1/D = -sum w_k e^{D tau_k} identity (D < 0) needs
    the grid built for x = -D, i.e. the POSITIVE range [-hi, -lo] of
    each dlist's [lo, hi] D-span, not [lo, hi] itself."""
    bounds = [_energy_bounds(dl, eps, nocc) for dl in dlists]
    negs = [(-hi, -lo) for lo, hi in bounds]
    e_min = min(a for a, _ in negs)
    e_max = max(b for _, b in negs)
    if e_min <= 0.0:
        raise ValueError(f"dlist not sign-definite: -D spans down to "
                         f"{e_min:.3g} <= 0 (grid requires D < 0 always)")
    taus, wts = minimax_time_grid(ntau, e_min, e_max)
    xs = np.linspace(e_min, e_max, 101)
    err = np.max(np.abs([1.0 / x - np.sum(wts * np.exp(-x * taus))
                         for x in xs]) * xs)
    if err > gate:
        raise ValueError(f"minimax grid ntau={ntau} invalid on "
                         f"[{e_min:.3g}, {e_max:.3g}]: rel err {err:.2e}")
    return taus, wts


def w57_term_specs(X, Y):
    """Enumerate the diagonal (X,Y) terms (block-resolved); cached per
    (X, Y) on first call.  Structurally identical to
    delta5_w77_evaluator.w77_term_specs, generalized off the [3,7,7,3]
    filter -- see that module for the per-field documentation."""
    if not hasattr(w57_term_specs, '_cache'):
        w57_term_specs._cache = {}
    if (X, Y) in w57_term_specs._cache:
        return w57_term_specs._cache[(X, Y)]
    specs = []
    for block in GS._BLOCKS:
        for skel in enumerate_skeletons(5, block):
            for ordering in skel.orderings:
                cuts = GS._ordering_cuts(skel, ordering)
                if any(c[0] == 0 for c in cuts):
                    continue
                if [len(c[1]) + len(c[2]) for c in cuts] != [3, X, Y, 3]:
                    continue
                sec = GS.ordering_sector(skel, ordering)
                d = [c[0] for c in cuts]
                w = (skel.coeff / skel.n_ext) * GS.ordering_sign(skel, sec)
                vstar = ordering[0]
                strip_lbl = (skel.bra_label if vstar == skel.bra_vertex
                             else skel.ket_label)
                ext_lbl = (skel.ket_label if vstar == skel.bra_vertex
                           else skel.bra_label)
                ext_fam = (block[1] if vstar == skel.bra_vertex
                           else block[0])
                xi, cfg_lbls = _du3_strip_factor(
                    skel.factors[vstar], strip_lbl, sec)
                oth_lbls = list(cuts[3][1]) + list(cuts[3][2])
                kinds_c = [l in OCC_LETTERS for l in cfg_lbls]
                kinds_o = [l in OCC_LETTERS for l in oth_lbls]
                sigmas = []
                for per in permutations(range(3)):
                    ok = True
                    for t in range(3):
                        lo = oth_lbls[per[t]]
                        if kinds_o[per[t]] != kinds_c[t] \
                                or (lo in cfg_lbls and lo != cfg_lbls[t]) \
                                or (cfg_lbls[t] in oth_lbls
                                    and lo != cfg_lbls[t]):
                            ok = False
                            break
                    if ok:
                        sigmas.append(per)
                rest = sorted({ln.idx for ln in skel.lines}
                              - set(cfg_lbls) - set(oth_lbls))
                for per in sigmas:
                    sub = {cfg_lbls[t]: chr(97 + t) for t in range(3)}
                    sub[ext_lbl] = 'd'
                    for i, l in enumerate(rest):
                        sub[l] = chr(101 + i)
                    for t in range(3):
                        sub[oth_lbls[per[t]]] = sub[cfg_lbls[t]]
                    facs = []
                    for vv, (cr, an) in enumerate(skel.factors):
                        if vv == vstar:
                            continue
                        ls = list(cr) + list(an)
                        facs.append((''.join(sub[l_] for l_ in ls),
                                     tuple(l_ in OCC_LETTERS
                                           for l_ in ls)))

                    def dlist(ci):
                        out = []
                        for sgn, ls in ((+1.0, cuts[ci][1]),
                                        (-1.0, cuts[ci][2])):
                            for l_ in ls:
                                out.append((sub[l_], l_ in OCC_LETTERS,
                                            sgn))
                        for sgn, ls in ((-1.0, cuts[0][1]),
                                        (+1.0, cuts[0][2])):
                            for l_ in ls:
                                out.append((sub[l_], l_ in OCC_LETTERS,
                                            sgn))
                        return out
                    plan, batch = _plan_with_batching(
                        [f[0] for f in facs], 'abcd')
                    specs.append(dict(
                        facs=facs, plan=plan, batch=batch, sec=sec,
                        kinds_c=tuple(kinds_c), ext_fam=ext_fam,
                        d1=dlist(1), d2=dlist(2),
                        pref=-w * xi / (d[0] * d[3]),
                        dmid=(d[1], d[2])))
    w57_term_specs._cache[(X, Y)] = specs
    return specs


def _rows_laplace_xy(X, Y, eps, g, nocc, ntau, taus, wts):
    """Shared per-(X,Y) accumulation loop (same structure as
    delta5_w77_evaluator.w77_rows_laplace's inner term loop, factored so
    (5,7) and (7,5) can share ONE grid pass)."""
    norb = len(eps)
    nvir = norb - nocc
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}
    grids = {+1: None, -1: None}
    for spec in w57_term_specs(X, Y):
        base = [g[tuple(sl[k] for k in kinds)]
                for _, kinds in spec['facs']]
        subs = [s_ for s_, _ in spec['facs']]
        owner = {}
        for fi, (s_, kinds) in enumerate(spec['facs']):
            for ax, ch in enumerate(s_):
                owner.setdefault((ch, kinds[ax]), (fi, ax))
        for term_i, (dsq, dlin) in enumerate(
                ((spec['d1'], spec['d2']), (spec['d2'], spec['d1']))):
            dmid = spec['dmid'][term_i]
            acc = None
            for k in range(ntau):
                for l_ in range(ntau):
                    wfac = wts[k] * taus[k] * wts[l_] * (-1.0) \
                        * spec['pref'] * dmid
                    texp = {}
                    for c2, isocc, sgn in dsq:
                        key = (c2, isocc)
                        texp[key] = texp.get(key, 0.0) + sgn * taus[k]
                    for c2, isocc, sgn in dlin:
                        key = (c2, isocc)
                        texp[key] = texp.get(key, 0.0) + sgn * taus[l_]
                    ops = [b for b in base]
                    for key, tv in texp.items():
                        if tv == 0.0:
                            continue
                        fi, ax = owner[key]
                        s_, kinds = spec['facs'][fi]
                        sh = [1] * len(s_)
                        sh[ax] = -1
                        ops[fi] = ops[fi] * np.exp(
                            tv * eps[sl[kinds[ax]]]).reshape(sh)

                    def run_plan(subs_l, ops_l):
                        cur = {i: (subs_l[i], ops_l[i])
                               for i in range(len(ops_l))}
                        nxt = len(ops_l)
                        for (i, j, res) in spec['plan']:
                            si, ai = cur[i]
                            sj, aj = cur[j]
                            assert len(res) <= 4
                            cur[nxt] = (res, np.einsum(
                                f"{si},{sj}->{res}", ai, aj,
                                optimize=True))
                            nxt += 1
                        s_fin, a_fin = cur[nxt - 1]
                        return np.einsum(f"{s_fin}->abcd", a_fin)

                    if spec['batch'] is None:
                        a_fin = run_plan(subs, ops)
                    else:
                        ch = spec['batch']
                        subs_b = [s_.replace(ch, '') for s_ in subs]
                        fi0 = next(i2 for i2, s_ in enumerate(subs)
                                   if ch in s_)
                        kb = spec['facs'][fi0][1][subs[fi0].index(ch)]
                        nb = nocc if kb else nvir
                        a_fin = 0.0
                        for vb in range(nb):
                            ops_b = []
                            for (s_, _k), op_ in zip(spec['facs'], ops):
                                s2 = s_
                                while ch in s2:
                                    op_ = np.take(op_, vb,
                                                  axis=s2.index(ch))
                                    s2 = s2.replace(ch, '', 1)
                                ops_b.append(op_)
                            a_fin = a_fin + run_plan(subs_b, ops_b)
                    if acc is None:
                        acc = wfac * a_fin
                    else:
                        acc = acc + wfac * a_fin
            sec = spec['sec']
            if grids[sec] is None:
                od = nocc if spec['kinds_c'][0] else nvir
                pd = nocc if spec['kinds_c'][1] else nvir
                grids[sec] = np.zeros((od, pd, pd, norb))
            colsl = o if spec['ext_fam'] == 'o' else v
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
        return np.array(rows) / 2.0          # G1 stripped-factor law

    zvec_p = np.zeros(0)
    zvec_m = np.zeros(0)
    Tp, Tm = grids[+1], grids[-1]
    rp = pack(Tp, +1) if Tp is not None else zvec_p
    rm = pack(Tm, -1) if Tm is not None else zvec_m
    return rp, rm


def w57_rows_laplace(eps, g, nocc, ntau=14):
    """(rows_p, rows_m) x {(5,7), (7,5)}: the two remaining fully
    denominator-definite X(0,3) survivor classes, rank <= 4.  Returns
    dict {(X,Y): (rows_p, rows_m)}."""
    eps = np.asarray(eps, float)
    g = np.asarray(g)
    dlists = []
    for X, Y in ((5, 7), (7, 5)):
        for spec in w57_term_specs(X, Y):
            dlists.append(spec['d1'])
            dlists.append(spec['d2'])
    taus, wts = tau_grid_general(eps, nocc, dlists, ntau)
    out = {}
    for X, Y in ((5, 7), (7, 5)):
        out[(X, Y)] = _rows_laplace_xy(X, Y, eps, g, nocc, ntau, taus, wts)
    return out
