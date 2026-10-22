"""PRODUCTION (rank<=4) evaluator for the cross-pattern w7
brackets: the diagram sides of

  B5  = sym X(013) survivor      [3,3,7,3] classes, v=3 at (0,1,3)
  B1  = the (0,1) w7 bracket     [3,3,7,3] classes, v=2 at (0,1)
  B3  = the (0,2) w7 bracket     [3,7,3,3] classes, v=2 at (0,2)
  B2  = the (0,1) f4 survivor    [3,3,7,f4] classes, v=2 at (0,1)
  B4  = the (0,2) f4 survivor    [3,7,3,f4] classes, v=2 at (0,2)

(sym X(01) = B1 + B2 - C2piece(01); sym X(02) = B3 + B4 - C2piece(02);
sym X(013) = B5 -- see an external gate script (not in this repository).)

MECHANISMS (all inherited, no new formalism):

* B1/B3/B5 extend delta5_x03_evaluator's pair-loop: bind the row
  config A (cut0 strip) AND the far boundary config B (cut3) and
  evaluate the per-pair term dense over the remaining <=4 rest axes
  (any sign -- plain division; rank<=4 by construction).  For B5, B
  runs over A's OWN K1-degenerate class (energy match, bounded); for
  B1/B3 it runs over ALL tabulated configs with the scalar boundary
  resolvent 1/D3, D3 = cutE3(B) - cutE0(A) (skipping |D3|<ZERO, which
  is exactly the deeper B5 class).  The adjacent on-pole cut of the
  pattern enters as a dense 0/1 MASK over the rest axes (|D|<ZERO
  selection, no division -- sign irrelevant).

* B2/B4 use delta5_w37_evaluator's single-tau-grid pattern: the free
  cut3 is ALWAYS exactly the last (slot-4) vertex's own 4 lines (2h2p,
  strictly negative), so 1/E3f is absorbed elementwise into that
  vertex's factor (the delta_u3_laplace "extreme vertex is a t2-like
  amplitude" trick); the definite 7-line denominator D^2 gets the
  single minimax grid (1/D^2 = +sum_k w_k tau_k e^{D tau_k}, D<0); the
  adjacent on-pole cut is the same dense 0/1 mask operand; rank<=4
  enforced by the shared _plan_with_batching contraction search.

Gate: an external gate script (not in this repository) -- row-level vs the
dense scan (delta5_f_w7dia_{r1,r1.6}.npz) + fold vs the cached fine2
pair targets, both geometries.
"""
import numpy as np

from src.Base.pdaggerq_interface import OCC_LETTERS
from src.CodeGeneration.diagrammatic import goldstone as GS
from src.CodeGeneration.diagrammatic.emit import (
    _du3_strip_factor)
from src.CodeGeneration.diagrammatic.skeleton_enum import (
    enumerate_skeletons)
from src.SingleReference.ADC.delta5_w77_evaluator import (
    _plan_with_batching)
from src.SingleReference.ADC.delta5_w37_evaluator import (
    _net, tau_grid_good)
from src.SingleReference.ADC.delta5_x03_evaluator import (
    config_k1_groups, _bind_facs, _bound_D)

ZERO = 1e-8


def xp_term_specs(shape):
    """Per-ordering term metadata for one all-omega [3,X,Y,3] shape,
    with UNBOUND (A, B) templates -- delta5_x03_evaluator.x03_term_specs
    plus the d3 (far-boundary) dlist.  shape = (3,3,7,3) or (3,7,3,3)."""
    if not hasattr(xp_term_specs, '_cache'):
        xp_term_specs._cache = {}
    if shape in xp_term_specs._cache:
        return xp_term_specs._cache[shape]
    specs = []
    for block in GS._BLOCKS:
        for skel in enumerate_skeletons(5, block):
            for ordering in skel.orderings:
                cuts = GS._ordering_cuts(skel, ordering)
                if any(c[0] == 0 for c in cuts):
                    continue
                if tuple(len(c[1]) + len(c[2]) for c in cuts) != shape:
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
                if sec == +1:
                    oth_lbls = tuple(list(cuts[3][1]) + list(cuts[3][2]))
                else:
                    oth_lbls = tuple(list(cuts[3][2]) + list(cuts[3][1]))
                cfg_kinds = tuple(l in OCC_LETTERS for l in cfg_lbls)
                oth_kinds = tuple(l in OCC_LETTERS for l in oth_lbls)
                assert oth_kinds == cfg_kinds, skel.raw
                shared_cfg_pos, shared_oth_pos = [], []
                for to, lo in enumerate(oth_lbls):
                    if lo in cfg_lbls:
                        shared_oth_pos.append(to)
                        shared_cfg_pos.append(cfg_lbls.index(lo))
                rest = sorted({ln.idx for ln in skel.lines}
                              - set(cfg_lbls) - set(oth_lbls))
                sub = {cfg_lbls[t]: chr(97 + t) for t in range(3)}
                sub[ext_lbl] = 'd'
                free_oth = [t for t in range(3) if t not in shared_oth_pos]
                for i, t in enumerate(free_oth):
                    sub[oth_lbls[t]] = chr(72 + i)
                for i, l in enumerate(rest):
                    sub[l] = chr(101 + i)
                facs = []
                for vv, (cr, an) in enumerate(skel.factors):
                    if vv == vstar:
                        continue
                    ls = list(cr) + list(an)
                    facs.append((''.join(sub[l_] for l_ in ls),
                                 tuple(l_ in OCC_LETTERS for l_ in ls)))

                def dlist(ci):
                    out = []
                    for sgn, ls in ((+1.0, cuts[ci][1]), (-1.0, cuts[ci][2])):
                        for l_ in ls:
                            out.append((sub[l_], l_ in OCC_LETTERS, sgn))
                    for sgn, ls in ((-1.0, cuts[0][1]), (+1.0, cuts[0][2])):
                        for l_ in ls:
                            out.append((sub[l_], l_ in OCC_LETTERS, sgn))
                    return out

                specs.append(dict(
                    facs=facs, sec=sec, ext_fam=ext_fam, wxi=w * xi,
                    d0=d[0], cfg_kinds=cfg_kinds,
                    shared_cfg_pos=tuple(shared_cfg_pos),
                    shared_oth_pos=tuple(shared_oth_pos),
                    free_oth=tuple(free_oth),
                    d1=dlist(1), d2=dlist(2), d3=dlist(3)))
    xp_term_specs._cache[shape] = specs
    return specs


def _pack(T, nocc, norb, sector):
    A = T - T.transpose(0, 2, 1, 3)
    nvir = norb - nocc
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


def xp_pair_rows(bracket, eps, g, nocc, shape=None):
    """(rows_p, rows_m) for bracket in {'B5','B1','B3','V1'} via the
    bound (A, B) pair loop.  Row normalization matches the dense scan
    the dense prototype (pack WITHOUT /2; symfold at 1/2 gives the
    mirror-pair target).

    'V1': the v=1-position-0 residue of ONE all-omega
    (3,X,Y,3) shape (pass `shape`) -- covers the outer-singles all-w
    signatures.  All three non-vanishing cuts are OFF-pole: the far
    boundary enters as the scalar resolvent 1/D3 (all-configs B loop,
    |D3|<ZERO skipped), the two middle cuts as dense any-sign division
    over the <=4 bounded rest axes (definiteness irrelevant here --
    the rule that 3-line cuts and bound-pair rest content
    never need Laplace).  val = d0 * num / (D1 * D2 * D3)."""
    eps = np.asarray(eps, float)
    g = np.asarray(g)
    norb = len(eps)
    nvir = norb - nocc
    if bracket == 'V1':
        assert shape is not None and shape[0] == 3 and shape[3] == 3
    else:
        shape = (3, 3, 7, 3) if bracket in ('B5', 'B1') else (3, 7, 3, 3)
    specs = xp_term_specs(shape)
    groups_cache = {}

    def groups_for(odd_occ):
        if odd_occ not in groups_cache:
            groups_cache[odd_occ] = config_k1_groups(eps, nocc, odd_occ)
        return groups_cache[odd_occ]

    grids = {+1: None, -1: None}
    for spec in specs:
        odd_occ = spec['cfg_kinds'][0]
        groups = groups_for(odd_occ)
        all_members = [m for ms in groups.values() for m in ms]
        shared_cfg = spec['shared_cfg_pos']
        shared_oth = spec['shared_oth_pos']
        free_oth = spec['free_oth']
        free_chars = [chr(72 + i) for i in range(len(free_oth))]
        rest_chars = sorted({ch for s_, _ in spec['facs'] for ch in s_}
                            - set('abcd') - set(free_chars))
        for key, members in groups.items():
            for ja, ka, la, KA in members:
                A = (ja, ka, la)
                bindA = {'a': ja, 'b': ka, 'c': la}
                bmembers = members if bracket == 'B5' else all_members
                for jb, kb, lb, KB in bmembers:
                    B = (jb, kb, lb)
                    ok = True
                    for sc, so in zip(shared_cfg, shared_oth):
                        if A[sc] != B[so]:
                            ok = False
                            break
                    if not ok:
                        continue
                    bind = dict(bindA)
                    for i, t in enumerate(free_oth):
                        bind[free_chars[i]] = B[t]
                    ndim = 1 + len(rest_chars)
                    rest_axis = {ch: 1 + i for i, ch in
                                 enumerate(rest_chars)}
                    D3 = _bound_D(spec['d3'], bind, eps, nocc, norb,
                                  ndim, rest_axis)
                    D3c = float(np.asarray(D3).ravel()[0])
                    if bracket == 'B5':
                        if abs(D3c) >= ZERO:
                            continue
                    else:                     # B1/B3/V1: far cut off-pole
                        if abs(D3c) < ZERO:
                            continue
                    subs, ops = _bind_facs(spec['facs'], bind, nocc,
                                           norb, g)
                    num = np.einsum(
                        ','.join(subs) + '->d' + ''.join(rest_chars),
                        *ops, optimize=True)
                    D1 = _bound_D(spec['d1'], bind, eps, nocc, norb,
                                  ndim, rest_axis)
                    D2 = _bound_D(spec['d2'], bind, eps, nocc, norb,
                                  ndim, rest_axis)
                    if bracket == 'B5':
                        sel = (np.abs(D1) < ZERO) & (np.abs(D2) >= ZERO)
                        D2s = np.where(np.abs(D2) < ZERO, 1.0, D2)
                        fac = spec['d0'] / D2s ** 3
                    elif bracket == 'B1':
                        sel = (np.abs(D1) < ZERO) & (np.abs(D2) >= ZERO)
                        D2s = np.where(np.abs(D2) < ZERO, 1.0, D2)
                        fac = (-spec['d0'] * (1.0 / D2s + 1.0 / D3c)
                               / (D2s * D3c))
                    elif bracket == 'B3':
                        sel = (np.abs(D2) < ZERO) & (np.abs(D1) >= ZERO)
                        D1s = np.where(np.abs(D1) < ZERO, 1.0, D1)
                        fac = (-spec['d0'] * (1.0 / D1s + 1.0 / D3c)
                               / (D1s * D3c))
                    else:                       # V1: v=1 at cut0 only
                        sel = ((np.abs(D1) >= ZERO)
                               & (np.abs(D2) >= ZERO))
                        D1s = np.where(np.abs(D1) < ZERO, 1.0, D1)
                        D2s = np.where(np.abs(D2) < ZERO, 1.0, D2)
                        fac = spec['d0'] / (D1s * D2s * D3c)
                    val = np.where(sel, fac, 0.0)
                    VU = (num * val).sum(axis=tuple(range(1, ndim)))
                    sec = spec['sec']
                    if grids[sec] is None:
                        od = nocc if odd_occ else nvir
                        pd = nvir if odd_occ else nocc
                        grids[sec] = np.zeros((od, pd, pd, norb))
                    row_idx = (ja if odd_occ else ja - nocc,
                               ka - (nocc if odd_occ else 0),
                               la - (nocc if odd_occ else 0))
                    colsl = (slice(0, nocc) if spec['ext_fam'] == 'o'
                             else slice(nocc, norb))
                    grids[sec][row_idx[0], row_idx[1], row_idx[2],
                               colsl] += spec['wxi'] * VU
    Tp, Tm = grids[+1], grids[-1]
    rp = _pack(Tp, nocc, norb, +1) if Tp is not None else np.zeros(0)
    rm = _pack(Tm, nocc, norb, -1) if Tm is not None else np.zeros(0)
    return rp, rm


# ---------------------------------------------------------------------
# B2/B4: free-cut3 classes, single-tau-grid Laplace on the definite
# 7-line denominator, E3f absorbed into the slot-4 vertex factor
# ---------------------------------------------------------------------

def xp_free_specs(shape):
    """shape = (3,3,7,4) [B2] or (3,7,3,4) [B4]: cuts 0-2 omega, cut3
    FREE (== the slot-4 vertex's own 4 lines, asserted)."""
    if not hasattr(xp_free_specs, '_cache'):
        xp_free_specs._cache = {}
    if shape in xp_free_specs._cache:
        return xp_free_specs._cache[shape]
    specs = []
    for block in GS._BLOCKS:
        for skel in enumerate_skeletons(5, block):
            for ordering in skel.orderings:
                cuts = GS._ordering_cuts(skel, ordering)
                d = [c[0] for c in cuts]
                if not (d[0] != 0 and d[1] != 0 and d[2] != 0
                        and d[3] == 0):
                    continue
                if tuple(len(c[1]) + len(c[2]) for c in cuts) != shape:
                    continue
                sec = GS.ordering_sector(skel, ordering)
                w = (skel.coeff / skel.n_ext) * GS.ordering_sign(skel, sec)
                vstar = ordering[0]
                vlast = ordering[-1]
                strip_lbl = (skel.bra_label if vstar == skel.bra_vertex
                             else skel.ket_label)
                ext_lbl = (skel.ket_label if vstar == skel.bra_vertex
                           else skel.bra_label)
                ext_fam = (block[1] if vstar == skel.bra_vertex
                           else block[0])
                xi, cfg_lbls = _du3_strip_factor(
                    skel.factors[vstar], strip_lbl, sec)
                cfg_kinds = tuple(l in OCC_LETTERS for l in cfg_lbls)
                cut3_lbls = set(cuts[3][1]) | set(cuts[3][2])
                vl_cr, vl_an = skel.factors[vlast]
                vl_lbls = list(vl_cr) + list(vl_an)
                assert set(vl_lbls) == cut3_lbls, \
                    f"cut3 lines != slot-4 vertex lines: {skel.raw}"
                rest = sorted({ln.idx for ln in skel.lines}
                              - set(cfg_lbls))
                sub = {cfg_lbls[t]: chr(97 + t) for t in range(3)}
                sub[ext_lbl] = 'd'
                for i, l in enumerate(rest):
                    sub[l] = chr(101 + i)
                facs = []
                vlast_fi = None
                for vv, (cr, an) in enumerate(skel.factors):
                    if vv == vstar:
                        continue
                    ls = list(cr) + list(an)
                    if vv == vlast:
                        vlast_fi = len(facs)
                    facs.append((''.join(sub[l_] for l_ in ls),
                                 tuple(l_ in OCC_LETTERS for l_ in ls)))

                def dlist(ci, flip0=True):
                    out = []
                    for sgn, ls in ((+1.0, cuts[ci][1]), (-1.0, cuts[ci][2])):
                        for l_ in ls:
                            out.append((sub[l_], l_ in OCC_LETTERS, sgn))
                    if flip0:
                        for sgn, ls in ((-1.0, cuts[0][1]),
                                        (+1.0, cuts[0][2])):
                            for l_ in ls:
                                out.append((sub[l_], l_ in OCC_LETTERS,
                                            sgn))
                    return out

                dmask = dlist(1 if shape[1] == 3 else 2)
                dgood = dlist(2 if shape[1] == 3 else 1)
                dfree = dlist(3, flip0=False)     # E3f: cut3 alone
                mask_net = _net(dmask)
                mask_sub = ''.join(sorted(ch for ch, _ in mask_net))
                mask_axes = [(ch, isocc, cf)
                             for (ch, isocc), cf in mask_net.items()]
                all_subs = [f[0] for f in facs] + \
                    ([mask_sub] if mask_sub else [])
                plan, batch = _plan_with_batching(all_subs, 'abcd')
                specs.append(dict(
                    facs=facs, plan=plan, batch=batch, sec=sec,
                    cfg_kinds=cfg_kinds, ext_fam=ext_fam,
                    vlast_fi=vlast_fi, dfree=dfree, dgood=dgood,
                    mask_axes=mask_axes, mask_sub=mask_sub,
                    pref=-w * xi * d[0]))
    xp_free_specs._cache[shape] = specs
    return specs


def _axis_array(axes, eps, nocc, norb, ax_shape):
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}
    ndim = len(ax_shape)
    D = np.zeros([1] * ndim)
    for ch, isocc, coeff in axes:
        sh = [1] * ndim
        sh[ax_shape[ch]] = -1
        D = D + coeff * eps[sl[isocc]].reshape(sh)
    return D


def xp_free_rows(bracket, eps, g, nocc, ntau=14):
    """(rows_p, rows_m) for bracket in {'B2','B4'}: rank<=4 via
    E3f-absorption + single-grid Laplace on the squared definite
    denominator + dense on-pole mask operand."""
    eps = np.asarray(eps, float)
    g = np.asarray(g)
    norb = len(eps)
    nvir = norb - nocc
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}
    shape = (3, 3, 7, 4) if bracket == 'B2' else (3, 7, 3, 4)
    specs = xp_free_specs(shape)
    taus, wts = tau_grid_good(eps, nocc, [s['dgood'] for s in specs], ntau)

    grids = {+1: None, -1: None}
    for spec in specs:
        base = []
        for fi, (s_, kinds) in enumerate(spec['facs']):
            op = g[tuple(sl[k] for k in kinds)]
            if fi == spec['vlast_fi']:
                # absorb 1/E3f into the slot-4 vertex factor (2h2p,
                # strictly negative -- the t2-like amplitude trick)
                axs = {ch: i for i, ch in enumerate(s_)}
                E3f = _axis_array(
                    [(ch, isocc, sgn) for ch, isocc, sgn in spec['dfree']],
                    eps, nocc, norb, axs)
                op = op / E3f
            base.append(op)
        subs = [s_ for s_, _ in spec['facs']]
        owner = {}
        for fi, (s_, kinds) in enumerate(spec['facs']):
            for ax, ch in enumerate(s_):
                owner.setdefault((ch, kinds[ax]), (fi, ax))

        mask_sub = spec['mask_sub']
        if mask_sub:
            # every mask_sub char comes from mask_axes, so Dm already
            # has full extent on every axis -- no broadcast needed
            ax_shape = {ch: i for i, ch in enumerate(mask_sub)}
            Dm = _axis_array(spec['mask_axes'], eps, nocc, norb, ax_shape)
            mask_op = (np.abs(Dm) < ZERO).astype(float)
        acc = None
        for k in range(ntau):
            wfac = wts[k] * taus[k] * spec['pref']
            texp = {}
            for c2, isocc, sgn in spec['dgood']:
                key = (c2, isocc)
                texp[key] = texp.get(key, 0.0) + sgn * taus[k]
            ops = [b for b in base]
            for key2, tv in texp.items():
                if tv == 0.0:
                    continue
                fi, ax = owner[key2]
                s_, kinds = spec['facs'][fi]
                sh = [1] * len(s_)
                sh[ax] = -1
                ops[fi] = ops[fi] * np.exp(
                    tv * eps[sl[kinds[ax]]]).reshape(sh)

            def run_plan(subs_l, ops_l):
                cur = {i: (subs_l[i], ops_l[i]) for i in range(len(ops_l))}
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

            subs_all = subs + ([mask_sub] if mask_sub else [])
            ops_all = ops + ([mask_op] if mask_sub else [])
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
                    kb = {c2: io for c2, io, _ in
                          spec['mask_axes']}[ch]
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
            od = nocc if spec['cfg_kinds'][0] else nvir
            pd = nvir if spec['cfg_kinds'][0] else nocc
            grids[sec] = np.zeros((od, pd, pd, norb))
        colsl = o if spec['ext_fam'] == 'o' else v
        grids[sec][:, :, :, colsl] += acc

    Tp, Tm = grids[+1], grids[-1]
    rp = _pack(Tp, nocc, norb, +1) if Tp is not None else np.zeros(0)
    rm = _pack(Tm, nocc, norb, -1) if Tm is not None else np.zeros(0)
    return rp, rm
