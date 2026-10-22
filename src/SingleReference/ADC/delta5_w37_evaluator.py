"""PRODUCTION evaluator attempt for the w37/w73 survivor rows.

w37=(X,Y)=(3,7) and w73=(7,3) are the two X(0,3) signatures where ONE
of D1=E1-E0 / D2=E2-E0 is genuinely sign-indefinite : comparing a row-config's OWN fixed label(s)
to a freely-summed same-kind "rest" quantity, with no shell/gap
argument to protect the sign.  The mirror-pair-summed-orderings idea
(App-A precedent) does NOT resolve this -- this
module tries a DIFFERENT mechanism instead.

KEY STRUCTURAL FACT: the
INDEFINITE axis, after net-summing repeated/cancelling labels, depends
on AT MOST 2 genuinely free ("rest") dummy labels, plus up to 3 of the
row's own config labels -- NEVER more (a graph-topological consequence
of X or Y == 3: the cut differs from the config's own 3-line boundary
cut by moving exactly one internal vertex across it, which can only
ever expose <= 2 new lines).  24/136 terms (per (X,Y), H4) have the bad
axis IDENTICALLY ZERO (the "structural zero" case -- config axes' own
net ALSO cancels, not just the free part): these terms belong to a
DEEPER v>=3 coincidence class and must be EXCLUDED from the (0,3)-only
bracket entirely (contribute 0), not divided through.

CONSEQUENCE: the indefinite factor never needs a Laplace/exponential
decomposition at all (which is what makes a sign-indefinite denominator
hard) -- it can be materialized DIRECTLY as a small (<=5, typically
2-3-axis) DENSE array via plain elementwise division (numpy handles any
sign correctly), independent of system size, and fed into the SAME
rank<=4 contraction-plan search as one more low-rank operand.  Only the
OTHER (definite) axis needs the standard Laplace treatment, and only a
SINGLE grid, not delta5_w77_evaluator's double (k,l) grid (which pairs
TWO Laplace-treated denominators): term A = d1/(D1^2 D2) becomes
[dense 1/D1^2] * [single-Laplace 1/D2]; term B = d2/(D2^2 D1) becomes
[single-Laplace-repeated 1/D2^2] * [dense 1/D1] -- D1 never touches a
tau grid, so this evaluator is O(ntau), not O(ntau^2).

Gate: an external gate script (not in this repository) -- vs the dense
front-strip prototype (delta5_e_mirror.py, WITH its on-pole/structural-
zero masking fix), both geometries.
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

ZERO_TOL = 1e-10


def _net(dlist):
    net = {}
    for ch, isocc, sgn in dlist:
        net[(ch, isocc)] = net.get((ch, isocc), 0.0) + sgn
    return {k: v for k, v in net.items() if abs(v) > 1e-12}


def w37_term_specs(X, Y, bad_idx, good_idx):
    """Enumerate diagonal (X,Y) terms, classifying the `bad_idx` cut
    (indefinite: E{bad_idx}-E0) into EXCLUDED (structural zero) or a
    small set of `bad_axes` (ch, isocc, coeff); the `good_idx` cut
    (definite) keeps its full dlist for standard Laplace treatment.
    Structurally identical to w57_term_specs otherwise -- see that
    module / delta5_w77_evaluator for the per-field documentation."""
    cache_key = (X, Y, bad_idx, good_idx)
    if not hasattr(w37_term_specs, '_cache'):
        w37_term_specs._cache = {}
    if cache_key in w37_term_specs._cache:
        return w37_term_specs._cache[cache_key]
    specs = []
    n_excluded = 0
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

                    dbad = dlist(bad_idx)
                    net_bad = _net(dbad)
                    if not net_bad:
                        n_excluded += 1
                        continue        # structural zero: excluded
                    dgood = dlist(good_idx)

                    facs = []
                    for vv, (cr, an) in enumerate(skel.factors):
                        if vv == vstar:
                            continue
                        ls = list(cr) + list(an)
                        facs.append((''.join(sub[l_] for l_ in ls),
                                     tuple(l_ in OCC_LETTERS
                                           for l_ in ls)))
                    bad_axes = [(ch, isocc, cf)
                               for (ch, isocc), cf in net_bad.items()]
                    bad_sub = ''.join(sorted(ch for ch, _, _ in bad_axes))
                    # the dense bad-D array is ONE MORE operand in the
                    # SAME rank<=4 search (small: <=2 rest axes + <=3
                    # config axes, typically 1-3 total after netting)
                    all_subs = [f[0] for f in facs] + [bad_sub]
                    plan, batch = _plan_with_batching(all_subs, 'abcd')
                    specs.append(dict(
                        facs=facs, plan=plan, batch=batch, sec=sec,
                        kinds_c=tuple(kinds_c), ext_fam=ext_fam,
                        bad_axes=bad_axes, bad_sub=bad_sub, dgood=dgood,
                        pref=-w * xi / (d[0] * d[3]),
                        dmid=(d[1], d[2]), bad_idx=bad_idx))
    w37_term_specs._cache[cache_key] = specs
    print(f"w37_term_specs({X},{Y}): {len(specs)} kept, "
          f"{n_excluded} structural-zero excluded", flush=True)
    return specs


def _good_bounds(dlist, eps, nocc):
    """Conservative [-hi, -lo] positive span for the GOOD (definite)
    axis, same net-then-bound logic as delta5_w57_evaluator's
    _energy_bounds/tau_grid_general (duplicated narrowly to keep this
    module import-self-contained; identical algorithm)."""
    eo = np.sort(np.asarray(eps, float)[:nocc])
    ev = np.sort(np.asarray(eps, float)[nocc:])
    net = {}
    for ch, isocc, sgn in dlist:
        net[(ch, isocc)] = net.get((ch, isocc), 0.0) + sgn
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
    return -hi, -lo


def tau_grid_good(eps, nocc, dlists, ntau, gate=1e-8):
    bounds = [_good_bounds(dl, eps, nocc) for dl in dlists]
    e_min = min(a for a, _ in bounds)
    e_max = max(b for _, b in bounds)
    if e_min <= 0.0:
        raise ValueError(f"good-axis dlist not definite: spans down to "
                         f"{e_min:.3g} <= 0")
    taus, wts = minimax_time_grid(ntau, e_min, e_max)
    xs = np.linspace(e_min, e_max, 101)
    err = np.max(np.abs([1.0 / x - np.sum(wts * np.exp(-x * taus))
                         for x in xs]) * xs)
    if err > gate:
        raise ValueError(f"minimax grid ntau={ntau} invalid on "
                         f"[{e_min:.3g}, {e_max:.3g}]: rel err {err:.2e}")
    return taus, wts


def _dense_bad_array(bad_axes, eps, nocc, norb, ax_shape):
    """Dense array of D_bad = sum_i coeff_i * eps(label_i), broadcast
    over ax_shape's axes (numpy division handles any sign -- this is
    the whole point: no Laplace decomposition needed for a <=2-free-
    dummy quantity)."""
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}
    ndim = len(ax_shape)
    D = np.zeros([1] * ndim)
    for ch, isocc, coeff in bad_axes:
        ax = ax_shape[ch]
        sh = [1] * ndim
        sh[ax] = -1
        D = D + coeff * eps[sl[isocc]].reshape(sh)
    return D


def w37_rows_laplace(X, Y, bad_idx, good_idx, eps, g, nocc, ntau=14):
    """(rows_p, rows_m) of the diagonal (X,Y) survivor class, rank<=4,
    via the dense-indefinite-factor + single-grid-Laplace-good-factor
    construction (see module docstring)."""
    eps = np.asarray(eps, float)
    g = np.asarray(g)
    norb = len(eps)
    nvir = norb - nocc
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}
    specs = w37_term_specs(X, Y, bad_idx, good_idx)
    taus, wts = tau_grid_good(eps, nocc, [s['dgood'] for s in specs], ntau)

    grids = {+1: None, -1: None}
    for spec in specs:
        base = [g[tuple(sl[k] for k in kinds)]
                for _, kinds in spec['facs']]
        subs = [s_ for s_, _ in spec['facs']]
        owner = {}
        for fi, (s_, kinds) in enumerate(spec['facs']):
            for ax, ch in enumerate(s_):
                owner.setdefault((ch, kinds[ax]), (fi, ax))

        # dense bad-D array over its own (small) axis set
        bad_sub = spec['bad_sub']
        ax_shape = {ch: i for i, ch in enumerate(bad_sub)}
        Dbad = _dense_bad_array(spec['bad_axes'], eps, nocc, norb, ax_shape)
        zero = np.abs(Dbad) < ZERO_TOL
        Dbad_safe = np.where(zero, 1.0, Dbad)
        bad_inv1 = np.where(zero, 0.0, 1.0 / Dbad_safe)
        bad_inv2 = np.where(zero, 0.0, 1.0 / Dbad_safe ** 2)
        bad_shape = [1 if ch not in ax_shape
                     else (nocc if dict((c2, io) for c2, io, _ in
                                        spec['bad_axes'])[ch] else nvir)
                     for ch in bad_sub]

        # dmid convention matches delta5_w77_evaluator: term_i=0 uses
        # (dsq=D_{bad_idx}, dlin=D_{good_idx}) i.e. term A =
        # d[bad_idx]/(Dbad^2 Dgood); term_i=1 uses (dsq=D_{good_idx},
        # dlin=D_{bad_idx}) i.e. term B = d[good_idx]/(Dgood^2 Dbad)
        for term_i, dmid in enumerate(spec['dmid']):
            bad_power = bad_inv2 if term_i == 0 else bad_inv1
            acc = None
            for k in range(ntau):
                # term_i=0: good axis LINEAR, 1/Dgood = -sum w_k e^{...}
                # term_i=1: good axis SQUARED, 1/Dgood^2 = +sum w_k tau_k
                #           e^{...} (derivative-of-1/x trick; no minus)
                if term_i == 0:
                    wfac = wts[k] * (-1.0) * spec['pref'] * dmid
                else:
                    wfac = wts[k] * taus[k] * spec['pref'] * dmid
                texp = {}
                for c2, isocc, sgn in spec['dgood']:
                    key = (c2, isocc)
                    texp[key] = texp.get(key, 0.0) + sgn * taus[k]
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
                            f"{si},{sj}->{res}", ai, aj, optimize=True))
                        nxt += 1
                    s_fin, a_fin = cur[nxt - 1]
                    return np.einsum(f"{s_fin}->abcd", a_fin)

                subs_all = subs + [bad_sub]
                ops_all = ops + [bad_power.reshape(bad_shape)]
                if spec['batch'] is None:
                    a_fin = run_plan(subs_all, ops_all)
                else:
                    ch = spec['batch']
                    subs_b = [s_.replace(ch, '') for s_ in subs_all]
                    fi0 = next(i2 for i2, s_ in enumerate(subs_all)
                              if ch in s_)
                    if fi0 < len(spec['facs']):
                        kb = spec['facs'][fi0][1][subs[fi0].index(ch)]
                    else:
                        kb = dict((c2, io) for c2, io, _ in
                                 spec['bad_axes'])[ch]
                    nb = nocc if kb else nvir
                    a_fin = 0.0
                    for vb in range(nb):
                        ops_b = []
                        for s_, op_ in zip(subs_all, ops_all):
                            s2 = s_
                            while ch in s2:
                                op_ = np.take(op_, vb, axis=s2.index(ch))
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
        return np.array(rows) / 2.0

    Tp, Tm = grids[+1], grids[-1]
    rp = pack(Tp, +1) if Tp is not None else np.zeros(0)
    rm = pack(Tm, -1) if Tm is not None else np.zeros(0)
    return rp, rm
