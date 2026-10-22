"""PRODUCTION evaluator for the w77 survivor rows (order-5 ingredient #1).

The w3.w7.w7.w3 double-outer-coincidence class is the first proven
pure-diagram survivor of the order-5 defect (no lambda^5 ISR chain
partner), with machine-certified strictly-negative on-shell
denominators and a closed diagonal row form gated at 1e-18 against the
independent Laurent tables (an external gate script (not in this repository),
both H4 geometries).  This module evaluates those rows at the
PRODUCTION floor: every intermediate rank <= 4 (asserted per term at
runtime), no full index grids, denominators by minimax Laplace.

    rows[A, q] = 1/2 sum_{orderings, sigma} w xi
                 * sum_inner N_sigma(A, q, inner)
                 * (-1/(d0 d3)) [ d1/(D1^2 D2) + d2/(D2^2 D1) ],

    1/(Da^2 Db) = sum_{k,l} w_k tau_k w_l e^{Da tau_k + Db tau_l}
    (Da, Db strictly negative 2h2p sums; repeated-denominator rule,
     adc_codegen.tex; every exponential factorizes index-wise).

Term specs are enumerated at runtime from the committed route-B
skeleton machinery (seconds; pdaggerq-free); the contraction order per
term is searched greedily and REJECTED if any intermediate would
exceed rank 4 -- the production floor is enforced by construction, not
by convention.  NOTE: this is ONE ingredient of the order-5 dressing
(the complete delta_u5_rows additionally needs the remaining X(P)
survivors + ISR-piece rows + cross rows).

Gate: an external gate script (not in this repository) --
machine agreement vs the dense prototype rows at both geometries.
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


def tau_grid_2h2p(eps, nocc, ntau, gate=1e-8):
    """Self-gated minimax grid spanning the 2h2p spectrum
    [2 lowest-gap, 2 widest-gap] (the D1/D2 composition)."""
    eo = np.sort(np.asarray(eps, float)[:nocc])
    ev = np.sort(np.asarray(eps, float)[nocc:])
    e_min = 2.0 * ev[0] - 2.0 * eo[-1]
    e_max = 2.0 * ev[-1] - 2.0 * eo[0]
    taus, wts = minimax_time_grid(ntau, e_min, e_max)
    xs = np.linspace(e_min, e_max, 101)
    err = np.max(np.abs([1.0 / x - np.sum(wts * np.exp(-x * taus))
                         for x in xs]) * xs)
    if err > gate:
        raise ValueError(f"minimax grid ntau={ntau} invalid on "
                         f"[{e_min:.3g}, {e_max:.3g}]: rel err {err:.2e}")
    return taus, wts


def _contraction_order(subs, out, cap=4):
    """Exhaustive pairwise contraction search with every intermediate at
    rank <= cap; returns the step list or None."""
    def rec(cur, live, steps):
        if len(live) == 1:
            return steps
        live_l = sorted(live)
        for ii in range(len(live_l)):
            for jj in range(ii + 1, len(live_l)):
                i, j = live_l[ii], live_l[jj]
                other = ''.join(cur[k] for k in live if k not in (i, j))
                keep = set(other) | set(out)
                res = ''.join(sorted((set(cur[i]) | set(cur[j])) & keep))
                if len(res) > cap:
                    continue
                nxt = max(cur) + 1
                cur2 = dict(cur)
                cur2[nxt] = res
                got = rec(cur2, (live - {i, j}) | {nxt},
                          steps + [(i, j, res)])
                if got is not None:
                    return got
        return None
    return rec({i: s for i, s in enumerate(subs)},
               set(range(len(subs))), [])


def _plan_with_batching(subs, out):
    """(plan, batch_char): a rank<=4 pairwise plan, batching over one
    non-output axis when no direct plan exists (memory floor preserved;
    flops scale by that axis' dimension)."""
    plan = _contraction_order(subs, out)
    if plan is not None:
        return plan, None
    inner = sorted({c for s in subs for c in s} - set(out))
    for ch in inner:
        subs2 = [s.replace(ch, '') for s in subs]
        plan = _contraction_order(subs2, out)
        if plan is not None:
            return plan, ch
    raise AssertionError(f"no rank<=4 plan even with batching: {subs}")


def w77_term_specs():
    """Enumerate the diagonal w77 terms (block-resolved); cached on
    first call.  Each spec: dict with factor label lists + families,
    axis map, cfg kinds, D1/D2 label lists, weight, sector, ext
    family, contraction plan."""
    if hasattr(w77_term_specs, '_cache'):
        return w77_term_specs._cache
    specs = []
    for block in GS._BLOCKS:
        for skel in enumerate_skeletons(5, block):
            for ordering in skel.orderings:
                cuts = GS._ordering_cuts(skel, ordering)
                if any(c[0] == 0 for c in cuts):
                    continue
                if [len(c[1]) + len(c[2]) for c in cuts] != [3, 7, 7, 3]:
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
                    # D lists as (axis char, is_occ, sign) per label,
                    # relative to E0 (the cfg cut): D_i = E_i - E_0
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
    w77_term_specs._cache = specs
    return specs


def w77_rows_laplace(eps, g, nocc, ntau=14):
    """(rows_p, rows_m) of the diagonal w77 survivor class, rank <= 4.

    Cost: n_terms * ntau^2 chained small contractions; one-time static.
    Rows in the SectorSpaces det convention (delta_u3_blocks twin)."""
    eps = np.asarray(eps, float)
    g = np.asarray(g)
    norb = len(eps)
    nvir = norb - nocc
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}
    taus, wts = tau_grid_2h2p(eps, nocc, ntau)
    exo = np.exp(np.multiply.outer(taus, eps[:nocc]))     # e^{+eps_o tau}
    exv = np.exp(-np.multiply.outer(taus, eps[nocc:]))    # e^{-eps_v tau}

    grids = {+1: None, -1: None}
    for spec in w77_term_specs():
        base = [g[tuple(sl[k] for k in kinds)]
                for _, kinds in spec['facs']]
        subs = [s_ for s_, _ in spec['facs']]
        # one OWNER slot per (axis char, kind): identified axes carry the
        # same index value, so the aggregated weight is applied exactly
        # once (double application on repeated dummies would square it)
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
                    # -1: 1/Db = -sum w e^{Db tau} for Db < 0
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

    return pack(grids[+1], +1), pack(grids[-1], -1)
