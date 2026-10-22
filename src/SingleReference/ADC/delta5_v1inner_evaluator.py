"""PRODUCTION (rank<=4) evaluator for the INNER v=1 singles group
-- the diagram side of X(1)+X(2).

WHAT THIS EMITS, AND WHY IT IS THE FIELD AND NOT ROWS.  Earlier attempts
looked for per-config rows here and found (correctly) that the inner
content is row-shaped only as a whole: the on-pole cut sits at position
1, so its lines come from TWO vertices and the left block is not
proportional to U1.  The group is closed by inverting the rows
ansatz over the whole on-pole family instead of per config:

    G S + S G = U M U'   (Sylvester),   X = G^+ (U M - S U),
    U = the on-pole coupling family (m x norb),  G = U U'

which needs only U M and U M U' -- i.e. only the pole RESIDUE MATRIX
M(kappa), never a config-resolved object.  So the production diagram
ingredient is exactly the per-pole residue field this module emits, and
the config-space bookkeeping (packing, orbit phases, the antisymmetric
projection needed for the per-config read-off) drops out of
the production path entirely.

    delta5_i_inner_rows.py = the same physics at validation scale
    (it additionally carries the per-config route as a diagnostic).

PER-CUT DISPATCH (the step-9 census rule, unchanged from
`delta5_v1outer_evaluator`):
  free 4-line cut at either end -> ABSORB into that end vertex's own
      factor (asserted per ordering: every line of the first/last vertex
      crosses the adjacent cut and self-loops do not exist).
  netted denominator identically zero -> EXCLUDED (deeper v>=2
      coincidence class; its v=1 residue vanishes).
  <=4 axes, any sign -> DENSE operand, 1/D masked to 0 where |D| < ZERO
      (that mask IS the v=1 off-pole selection).
  >4 axes and definite -> LAPLACE, exponentials factorized per label
      onto the owning vertex factor; `_grid_escalate` picks the size.
  else -> DENSE with BATCHING over the fewest non-output chars.
SIDE SPLIT (what makes this affordable).  Contracting all five vertices
straight to (p, q) with the on-pole selector as one more operand DOES
plan at rank<=4, but the selector's config axes push the batch depth to
4 on 320 orderings and 2 on 6160 -- ~1e5 plan runs per pole, measured
unusable.  Instead build the two SIDE factors once per ordering,

    L[cfg, p]  = vertices left of cut 1  / cut-0 denominator
    R[cfg, q]  = vertices right of cut 1 / cut-2, cut-3 denominators

(the config labels are shared, every other label is internal to its own
side -- a line crossing cut 1 IS a config line), and then each pole is a
single cheap  einsum('...p,...q->pq', L*mask, R).  At k1 both sides are
rank 4 outright; at k2 (five config labels) TWO config labels are bound
in an outer loop, which is what keeps that channel rank<=4 as well --
even though the k2 rows the field feeds are themselves an O(N^5) x O(N)
object.  Per-side batching only ever touches labels internal to that
side, so it can never sum away a config index.

Gate: an external gate script (not in this repository) -- vs the
dense prototype field (`delta5_i_inner_rows.diagram_mu`) and vs the
cached `delta5_fine2_*` position-1 class total, both geometries.
"""
from itertools import combinations, product

import numpy as np

from src.Base.pdaggerq_interface import OCC_LETTERS
from src.CodeGeneration.diagrammatic import goldstone as GS
from src.CodeGeneration.diagrammatic.skeleton_enum import (
    enumerate_skeletons)
from src.SingleReference.ADC.delta5_v1outer_evaluator import (
    _axis_array, _grid_escalate, _plan_multibatch)
from src.SingleReference.ADC.delta5_w37_evaluator import _good_bounds, _net

ZERO = 1e-8
TOL = 1e-9


def _enumerate(eps, nocc, sig_only=False):
    """Walk every v=1-position-1 ordering, yielding (signature, spec)
    with the per-cut dispatch resolved."""
    eps = np.asarray(eps, float)
    for block in GS._BLOCKS:
        for skel in enumerate_skeletons(5, block):
            for ordering in skel.orderings:
                cuts = GS._ordering_cuts(skel, ordering)
                d = [c[0] for c in cuts]
                nl = [len(c[1]) + len(c[2]) for c in cuts]
                if d[1] == 0 or nl[1] not in (3, 5):
                    continue
                sig = '.'.join(('w' if d[i] else 'f') + str(nl[i])
                               for i in range(4))
                if sig_only:
                    yield sig, None
                    continue
                sec = GS.ordering_sector(skel, ordering)
                w = (skel.coeff / skel.n_ext) * GS.ordering_sign(skel, sec)
                # config labels in the SectorSpaces order (holes first for
                # sector +1, the reverse for -1); kappa = -E_1/d_1
                cfg = (list(cuts[1][1]) + list(cuts[1][2])) if sec == +1 \
                    else (list(cuts[1][2]) + list(cuts[1][1]))
                sub = {cfg[t]: chr(97 + t) for t in range(len(cfg))}
                sub[skel.bra_label] = 'p'
                sub[skel.ket_label] = 'q'
                rest = sorted({ln.idx for ln in skel.lines} - set(cfg))
                for i, l_ in enumerate(rest):
                    sub[l_] = chr(114 + i)      # 'r','s',... no collision

                def dl(ci, minus_cfg):
                    out = []
                    for sgn, ls in ((+1.0, cuts[ci][1]), (-1.0, cuts[ci][2])):
                        for l_ in ls:
                            out.append((sub[l_], l_ in OCC_LETTERS, sgn))
                    if minus_cfg:
                        for sgn, ls in ((-1.0, cuts[1][1]),
                                        (+1.0, cuts[1][2])):
                            for l_ in ls:
                                out.append((sub[l_], l_ in OCC_LETTERS, sgn))
                    return out

                # free 4-line end cuts are the end vertex's own lines
                absorb = {}
                for ci, vend in ((0, ordering[0]), (3, ordering[-1])):
                    if d[ci] != 0 or nl[ci] != 4:
                        continue
                    cr, an = skel.factors[vend]
                    assert set(list(cr) + list(an)) == \
                        set(cuts[ci][1]) | set(cuts[ci][2]), \
                        f"free end cut != end vertex's lines: {skel.raw}"
                    absorb[vend] = _net(dl(ci, False))

                cfgsub = ''.join(chr(97 + t) for t in range(len(cfg)))
                # two config labels are bound in an outer loop at k2 so
                # that each side's OUTPUT (cfg + one external) is rank<=4
                cbatch = cfgsub[:max(0, len(cfg) + 1 - 4)]
                free_cfg = ''.join(c for c in cfgsub if c not in cbatch)

                sides, dead = [], False
                for vlist, cutlist, extch in (
                        (ordering[:2], (0,), 'p' if skel.bra_vertex in
                         set(ordering[:2]) else 'q'),
                        (ordering[2:], (2, 3), 'q' if skel.bra_vertex in
                         set(ordering[:2]) else 'p')):
                    dense, lap = [], []
                    for ci in cutlist:
                        if ci in (0, 3) and d[ci] == 0 and nl[ci] == 4:
                            continue                    # absorbed above
                        nd = _net(dl(ci, d[ci] != 0))
                        if not nd:
                            assert d[ci] != 0, \
                                f"free cut nets to zero: {skel.raw}"
                            dead = True                 # deeper v>=2 class
                            break
                        axes = [(ch, io, cf) for (ch, io), cf in nd.items()]
                        lo, _ = _good_bounds(axes, eps, nocc)
                        if len(axes) > 4 and lo > 0.0:
                            lap.append(axes)
                        else:
                            dense.append(axes)
                    if dead:
                        break
                    facs = []
                    for vv in vlist:
                        cr, an = skel.factors[vv]
                        ls = list(cr) + list(an)
                        facs.append((''.join(sub[l_] for l_ in ls),
                                     tuple(l_ in OCC_LETTERS for l_ in ls),
                                     vv))
                    subs_all = [f[0] for f in facs] + \
                        [''.join(sorted(ch for ch, _, _ in ax))
                         for ax in dense]
                    subs_b = [''.join(c for c in s_ if c not in cbatch)
                              for s_ in subs_all]
                    plan, batch = _plan_multibatch(subs_b, free_cfg + extch)
                    assert not (set(batch) & set(cfgsub)), \
                        "per-side batching may not touch a config label"
                    kindof = {}
                    for s_, kinds, _ in facs:
                        for ax, ch in enumerate(s_):
                            kindof.setdefault(ch, kinds[ax])
                    sides.append(dict(facs=facs, dense=dense, lap=lap,
                                      plan=plan, batch=batch,
                                      kindof=kindof, ext=extch,
                                      out=free_cfg + extch))
                if dead:
                    continue

                yield sig, dict(
                    sides=sides, absorb=absorb, sec=sec, cfgsub=cfgsub,
                    cbatch=cbatch, free_cfg=free_cfg,
                    cfg_kinds=tuple(l_ in OCC_LETTERS for l_ in cfg),
                    block=block, pref=w / d[1], dcut=d[1])


def needed_signatures(eps, nocc):
    return sorted({s for s, _ in _enumerate(eps, nocc, sig_only=True)})


def v1inner_field(eps, g, nocc, kappas, ntau=14, report=False):
    """(len(kappas), norb, norb) per-pole residue field of the INNER
    v=1 position-1 group -- the diagram ingredient of X(1); X(2) is its
    transpose.  Every intermediate is rank<=4."""
    eps = np.asarray(eps, float)
    g = np.asarray(g)
    kappas = np.asarray(kappas, float)
    norb = len(eps)
    nvir = norb - nocc
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}
    ranges = {'o': np.arange(nocc), 'v': np.arange(nocc, norb)}

    specs = [sp for _, sp in _enumerate(eps, nocc)]
    all_lap = [dlv for sp in specs for side in sp['sides']
               for dlv in side['lap']]
    if all_lap:
        taus, wts, ntau = _grid_escalate(eps, nocc, all_lap, ntau)
    else:
        taus, wts = np.zeros(0), np.zeros(0)
    if report:
        ng, nb, nc = {}, {}, {}
        for sp in specs:
            gtot = sum(len(sd['lap']) for sd in sp['sides'])
            btot = max(len(sd['batch']) for sd in sp['sides'])
            ng[gtot] = ng.get(gtot, 0) + 1
            nb[btot] = nb.get(btot, 0) + 1
            nc[len(sp['cbatch'])] = nc.get(len(sp['cbatch']), 0) + 1
        print(f"v1inner: {len(specs)} orderings; grids/ordering {ng}; "
              f"per-side batch-depth {nb}; config-batch {nc}; "
              f"ntau={ntau}", flush=True)

    out = np.zeros((len(kappas), norb, norb))
    for sp in specs:
        ncfg = len(sp['cfg_kinds'])
        E1 = np.zeros([1] * ncfg)
        for t, isocc in enumerate(sp['cfg_kinds']):
            sh = [1] * ncfg
            sh[t] = -1
            E1 = E1 + (1.0 if isocc else -1.0) * eps[sl[isocc]].reshape(sh)
        kapgrid = -E1 / sp['dcut']
        hits = [ik for ik in range(len(kappas))
                if np.any(np.abs(kapgrid - kappas[ik]) < TOL)]
        if not hits:
            continue
        win = np.ix_(ranges[sp['block'][0]], ranges[sp['block'][1]])
        cdims = [nocc if sp['cfg_kinds'][sp['cfgsub'].index(c)] else nvir
                 for c in sp['cbatch']]

        for cvals in product(*[range(n) for n in cdims]):
            cbind = dict(zip(sp['cbatch'], cvals))
            kg = kapgrid
            for t, c in enumerate(sp['cbatch']):
                kg = np.take(kg, cvals[t], axis=sp['cfgsub'].index(c) - t)
            side_vals = []
            for side in sp['sides']:
                base = []
                for s_, kinds, vv in side['facs']:
                    op = g[tuple(sl[k] for k in kinds)]
                    if vv in sp['absorb']:
                        E, _ = _axis_array(
                            [(ch, io, c) for (ch, io), c
                             in sp['absorb'][vv].items()],
                            eps, nocc, norb, s_)
                        op = op / E
                    base.append(op)
                subs_f = [s_ for s_, _, _ in side['facs']]
                dense_orders = [''.join(sorted(ch for ch, _, _ in ax))
                                for ax in side['dense']]
                owner = {}
                for fi, (s_, kinds, _) in enumerate(side['facs']):
                    for ax, ch in enumerate(s_):
                        owner.setdefault((ch, kinds[ax]), (fi, ax))
                nlap = len(side['lap'])
                acc = 0.0
                for kcombo in (product(range(ntau), repeat=nlap) if nlap
                               else [()]):
                    wfac = 1.0
                    texp = {}
                    for gi, k in enumerate(kcombo):
                        wfac = wfac * (-wts[k])
                        for ch, isocc, cf in side['lap'][gi]:
                            texp[(ch, isocc)] = texp.get((ch, isocc), 0.0) \
                                + cf * taus[k]
                    ops = list(base)
                    for key, tv in texp.items():
                        if tv == 0.0:
                            continue
                        fi, ax = owner[key]
                        s_, kinds, _ = side['facs'][fi]
                        sh = [1] * len(s_)
                        sh[ax] = -1
                        ops[fi] = ops[fi] * np.exp(
                            tv * eps[sl[kinds[ax]]]).reshape(sh)

                    def run(subs_l, ops_l):
                        cur = {i: (subs_l[i], ops_l[i])
                               for i in range(len(ops_l))}
                        nxt = len(ops_l)
                        for (i, j, res) in side['plan']:
                            si, ai = cur[i]
                            sj, aj = cur[j]
                            assert len(res) <= 4, \
                                f"rank {len(res)} intermediate"
                            cur[nxt] = (res, np.einsum(
                                f"{si},{sj}->{res}", ai, aj, optimize=True))
                            nxt += 1
                        s_fin, a_fin = cur[nxt - 1]
                        return np.einsum(f"{s_fin}->{side['out']}", a_fin)

                    def build(bound):
                        subs_l, ops_l = [], []
                        for s_, op_ in zip(subs_f, ops):
                            s2 = s_
                            for ch, vb in bound.items():
                                while ch in s2:
                                    op_ = np.take(op_, vb,
                                                  axis=s2.index(ch))
                                    s2 = s2.replace(ch, '', 1)
                            subs_l.append(s2)
                            ops_l.append(op_)
                        for axes, order in zip(side['dense'], dense_orders):
                            D, red = _axis_array(axes, eps, nocc, norb,
                                                 order, bound=bound)
                            bad = np.abs(D) < ZERO
                            subs_l.append(red)
                            ops_l.append(np.where(
                                bad, 0.0, 1.0 / np.where(bad, 1.0, D)))
                        assert all(len(s_) <= 4 for s_ in subs_l)
                        return subs_l, ops_l

                    if not side['batch']:
                        val = run(*build(dict(cbind)))
                    else:
                        dims = [nocc if side['kindof'][ch] else nvir
                                for ch in side['batch']]
                        val = 0.0
                        for vals in product(*[range(n) for n in dims]):
                            bd = dict(cbind)
                            bd.update(dict(zip(side['batch'], vals)))
                            val = val + run(*build(bd))
                    acc = acc + wfac * val
                side_vals.append(acc)
            L, R = side_vals
            # output is always (bra, ket) = (p, q); which side carries
            # which external only changes the operand subscripts
            specL = (f"{sp['sides'][0]['out']},{sp['sides'][1]['out']}"
                     f"->pq")
            for ik in hits:
                m = (np.abs(kg - kappas[ik]) < TOL)
                if not m.any():
                    continue
                out[ik][win] += sp['pref'] * np.einsum(
                    specL, L * m[..., None], R, optimize=True)
    return out
