"""PRODUCTION (rank<=4) evaluator for the COMPLETE X(0,3) rows: all six original brackets
(w37, w73, w57, w75, m35, m53) plus w77, diag+off content together, in
one unified construction.

MECHANISM (the dense off-diagonal finding, generalized here off
the dense grid): delta5_w77_evaluator's diagonal construction
IDENTIFIES the far external vertex's own labels (oth_lbls) with the
row config's own labels (cfg_lbls) -- stronger than the physics needs,
which is only cutE(oth_lbls) == cutE(cfg_lbls) (same k1 pole energy,
not the same specific config).

RANK<=4 HARDENING (the new piece here): the dense prototype
materializes oth_lbls as a FULL grid over its kind-consistent range
(nocc*nvir^2-ish) and masks by energy match -- correct but not
production-scale. The key realization: for a FIXED row config A, the
set of A2 with K1(A2) == K1(A) is NOT the full config space -- it's a
(typically small, system-size-INDEPENDENT) DEGENERATE SHELL, bounded
by the intrinsic spin/symmetry multiplicity of the orbital spectrum,
not by norb. So instead of masking a dense grid, GROUP the full config
space by K1 value ONCE per (kind pattern, sector) -- reusable across
every diagram term of that (X,Y,sector) -- and for each row A, loop
over ONLY its own (small) degenerate class, applying the EXISTING
per-term rank<=4 Laurent contraction (D1,D2 machinery unchanged from
delta5_w77_evaluator/delta5_w37_evaluator) to each (A, A2) pair
explicitly. No new tensor axis is ever summed over the full virtual/
occupied space -- the "extra" sum is a plain Python loop over a
bounded set, not a rank concern.

A shared line between cut0 and cut3 (present in most terms -- e.g. the
line directly connecting the two external vertices when they're
adjacent in the skeleton) additionally constrains which A2 are valid
for a GIVEN A: whichever of oth_lbls' 3 positions coincides with one of
cfg_lbls' positions (same symbolic label) must take the SAME VALUE as
that cfg position, not just satisfy the energy match. Handled via an
explicit per-position consistency filter (see `_valid_a2_for_a`).

Gate: an external gate script (not in this repository) -- vs the
dense front-strip-with-energy-mask prototype (the dense prototype) and the cached diag+off Laurent tables,
both geometries.
"""
from itertools import permutations

import numpy as np

from src.Base.pdaggerq_interface import OCC_LETTERS
from src.CodeGeneration.diagrammatic import goldstone as GS
from src.CodeGeneration.diagrammatic.emit import (
    _du3_strip_factor)
from src.CodeGeneration.diagrammatic.skeleton_enum import (
    enumerate_skeletons)

ZERO = 1e-8
KTOL = 1e-8


def config_k1_groups(eps, nocc, odd_occ):
    """All (odd, p0, p1) configs of one kind pattern, grouped by exact
    K1 value.  odd_occ=True: sector+1 (odd hole, pair virtual pair,
    K1=eps[p0]+eps[p1]-eps[odd]); odd_occ=False: sector-1 (odd virtual,
    pair hole pair, K1=eps[odd]-eps[p0]-eps[p1]).  Matches
    adc_form.SectorSpaces's own K1 convention exactly (cross-checked by
    this module's gate).

    BOTH pair orderings (p0,p1) and (p1,p0) are included as SEPARATE
    entries (no p1>p0 canonicalization) -- required for pack()'s
    T - T.transpose(0,2,1,3) antisymmetrization downstream to see both
    fill positions, exactly matching how the dense per-term einsum
    (delta5_w77_evaluator etc.) populates a FULL (nocc,nvir,nvir,norb)
    grid, not just the k<l upper triangle.  Returns dict: rounded-K1 ->
    list of (odd,p0,p1,K1)."""
    norb = len(eps)
    odd_range = range(nocc) if odd_occ else range(nocc, norb)
    pair_range = range(nocc, norb) if odd_occ else range(nocc)
    groups = {}
    for j in odd_range:
        for k in pair_range:
            for l in pair_range:
                if l == k:
                    continue
                K1 = (eps[k] + eps[l] - eps[j] if odd_occ
                      else eps[j] - eps[k] - eps[l])
                key = round(float(K1), 8)
                groups.setdefault(key, []).append((j, k, l, K1))
    return groups


def x03_term_specs(X, Y):
    """Enumerate diag+off terms for one (X,Y) signature, WITHOUT
    binding cfg/oth to specific values -- returns per-term metadata
    (which positions of oth are shared with cfg, the vertex factor
    subscript templates) so the evaluator can plug in specific (A, A2)
    pairs later.  Structurally the front-strip half of
    delta5_e_offdiag_dense.full_rows_dense, factored so the (A, A2)
    binding happens at evaluation time, not enumeration time."""
    if not hasattr(x03_term_specs, '_cache'):
        x03_term_specs._cache = {}
    if (X, Y) in x03_term_specs._cache:
        return x03_term_specs._cache[(X, Y)]
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
                # oth_lbls in the SAME (odd, pair0, pair1) convention
                # _du3_strip_factor uses for cfg_lbls: odd is the
                # singleton opposite-kind label (occ iff sec==+1), NOT
                # simply "holes then particles" (which only happens to
                # put odd first for sec==+1 -- backwards for sec==-1,
                # caught by the kind-pattern assertion below)
                if sec == +1:
                    oth_lbls = tuple(list(cuts[3][1]) + list(cuts[3][2]))
                else:
                    oth_lbls = tuple(list(cuts[3][2]) + list(cuts[3][1]))
                cfg_kinds = tuple(l in OCC_LETTERS for l in cfg_lbls)
                oth_kinds = tuple(l in OCC_LETTERS for l in oth_lbls)
                if oth_kinds != cfg_kinds:
                    raise AssertionError(
                        f"oth kind pattern != cfg kind pattern: {skel.raw}")
                shared_cfg_pos, shared_oth_pos = [], []
                for to, lo in enumerate(oth_lbls):
                    if lo in cfg_lbls:
                        shared_oth_pos.append(to)
                        shared_cfg_pos.append(cfg_lbls.index(lo))

                rest = sorted({ln.idx for ln in skel.lines}
                              - set(cfg_lbls) - set(oth_lbls))
                sub = {cfg_lbls[t]: chr(97 + t) for t in range(3)}
                sub[ext_lbl] = 'd'
                # oth's OWN (non-shared) axes get fresh chars too; shared
                # ones already point at cfg's axis via `sub`
                free_oth = [t for t in range(3) if t not in shared_oth_pos]
                for i, t in enumerate(free_oth):
                    sub[oth_lbls[t]] = chr(72 + i)   # 'H','I' -- distinct pool
                for i, l in enumerate(rest):
                    sub[l] = chr(101 + i)

                facs = []
                for vv, (cr, an) in enumerate(skel.factors):
                    if vv == vstar:
                        continue
                    ls = list(cr) + list(an)
                    facs.append((''.join(sub[l_] for l_ in ls),
                                 tuple(l_ in OCC_LETTERS for l_ in ls)))
                out_sub = 'abcd' + ''.join(chr(72 + i)
                                          for i in range(len(free_oth))) \
                    + ''.join(chr(101 + i) for i in range(len(rest)))

                # D1/D2 as (axis_char, is_occ, sign) lists, EXACTLY the
                # delta5_w37_evaluator convention (cuts[ci] native sign,
                # -cuts[0] flipped sign) -- axis chars now span the
                # BOUND (a,b,c / H,I,...) pool since oth is no longer
                # identified with cfg
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
                    facs=facs, sec=sec, ext_fam=ext_fam,
                    pref=-w * xi / (d[0] * d[3]), dmid=(d[1], d[2]),
                    cfg_kinds=cfg_kinds, oth_kinds=oth_kinds,
                    shared_cfg_pos=tuple(shared_cfg_pos),
                    shared_oth_pos=tuple(shared_oth_pos),
                    free_oth=tuple(free_oth), out_sub=out_sub,
                    d1=dlist(1), d2=dlist(2)))
    x03_term_specs._cache[(X, Y)] = specs
    return specs


# ---------------------------------------------------------------------
# binding: reduce a term's vertex factors + D1/D2 lists once specific
# (A, A2) orbital values are plugged into the a,b,c / H,I char pool
# ---------------------------------------------------------------------

def _bind_facs(facs, bind, nocc, norb, g):
    """facs: [(subscript, kinds)]; bind: char -> ABSOLUTE orbital index
    (only for bound chars).  Returns (subs, ops) with bound axes
    indexed away -- only 'd' (output) and rest chars survive."""
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}
    out_subs, out_ops = [], []
    for s_, kinds in facs:
        op = g[tuple(sl[k] for k in kinds)]
        chars = list(s_)
        kk = list(kinds)
        idx = 0
        while idx < len(chars):
            ch = chars[idx]
            if ch in bind:
                val = bind[ch] - (0 if kk[idx] else nocc)
                op = np.take(op, val, axis=idx)
                del chars[idx]
                del kk[idx]
            else:
                idx += 1
        out_subs.append(''.join(chars))
        out_ops.append(op)
    return out_subs, out_ops


def _bound_D(dlist, bind, eps, nocc, norb, ndim, rest_axis):
    """D = sum sgn*eps(label) over a dlist, split into a SCALAR (bound
    chars) plus a broadcastable array over the surviving `rest_axis`
    chars (same ndim/axis convention as the bound facs arrays)."""
    o, v = slice(0, nocc), slice(nocc, norb)
    sl = {True: o, False: v}
    const = 0.0
    arr = np.zeros([1] * ndim)
    for ch, isocc, sgn in dlist:
        if ch in bind:
            const += sgn * eps[bind[ch]]
        else:
            ax = rest_axis[ch]
            sh = [1] * ndim
            sh[ax] = -1
            arr = arr + sgn * eps[sl[isocc]].reshape(sh)
    return const + arr


# ---------------------------------------------------------------------
# main evaluator: per term, per k1-degenerate class, explicit (A, A2)
# pairs within the (small) class -- no axis is ever summed over the
# full virtual/occupied space
# ---------------------------------------------------------------------

def x03_rows_full(X, Y, eps, g, nocc, diag_only=False):
    """(rows_p, rows_m): COMPLETE (diag+off) X(0,3)-family rows for one
    (X,Y) signature, rank<=4 per term (the class loop is a plain Python
    loop over a system-size-independent bounded set, not a tensor
    axis).  SectorSpaces row convention (odd,pair0,pair1) tuple order,
    packed via the SAME antisymmetrized pack() every other evaluator in
    this program uses."""
    eps = np.asarray(eps, float)
    g = np.asarray(g)
    norb = len(eps)
    nvir = norb - nocc
    specs = x03_term_specs(X, Y)
    groups_cache = {}

    def groups_for(odd_occ):
        if odd_occ not in groups_cache:
            groups_cache[odd_occ] = config_k1_groups(eps, nocc, odd_occ)
        return groups_cache[odd_occ]

    grids = {+1: None, -1: None}
    for spec in specs:
        odd_occ = spec['cfg_kinds'][0]
        groups = groups_for(odd_occ)
        shared_cfg = spec['shared_cfg_pos']
        shared_oth = spec['shared_oth_pos']
        free_oth = spec['free_oth']
        free_chars = [chr(72 + i) for i in range(len(free_oth))]
        rest_chars = sorted(set(spec['out_sub']) - set('abcd')
                            - set(free_chars))
        for key, members in groups.items():
            m = len(members)
            for ia in range(m):
                ja, ka, la, _ = members[ia]
                A = (ja, ka, la)
                bindA = {'a': ja, 'b': ka, 'c': la}
                for ib in ((ia,) if diag_only else range(m)):
                    jb, kb, lb, _ = members[ib]
                    # config_k1_groups already enumerates BOTH pair
                    # orderings as separate members, so B is used
                    # as-is here (no separate swap search needed --
                    # the swapped member is already a distinct ib)
                    B = (jb, kb, lb)
                    ok = True
                    for sc, so in zip(shared_cfg, shared_oth):
                        if A[sc] != B[so]:
                            ok = False
                            break
                    if ok:
                        bind = dict(bindA)
                        for i, t in enumerate(free_oth):
                            bind[free_chars[i]] = B[t]

                        subs, ops = _bind_facs(spec['facs'], bind, nocc,
                                               norb, g)
                        ndim = 1 + len(rest_chars)  # 'd' + rest
                        rest_axis = {ch: 1 + i for i, ch in
                                    enumerate(rest_chars)}
                        out_sub_bound = 'd' + ''.join(rest_chars)
                        num = np.einsum(
                            ','.join(subs) + '->' + out_sub_bound, *ops,
                            optimize=True)

                        D1 = _bound_D(spec['d1'], bind, eps, nocc, norb,
                                     ndim, rest_axis)
                        D2 = _bound_D(spec['d2'], bind, eps, nocc, norb,
                                     ndim, rest_axis)
                        bad = (np.abs(D1) < ZERO) | (np.abs(D2) < ZERO)
                        D1s = np.where(bad, 1.0, D1)
                        D2s = np.where(bad, 1.0, D2)
                        d1c, d2c = spec['dmid']
                        val = np.where(
                            bad, 0.0,
                            spec['pref'] * (d1c / (D1s ** 2 * D2s)
                                            + d2c / (D2s ** 2 * D1s)))
                        VU = (num * val).sum(axis=tuple(range(1, ndim)))

                        sec = spec['sec']
                        if grids[sec] is None:
                            od = nocc if odd_occ else nvir
                            pd = nocc if not odd_occ else nvir
                            grids[sec] = np.zeros((od, pd, pd, norb))
                        row_idx = (ja if odd_occ else ja - nocc,
                                  ka - (nocc if odd_occ else 0),
                                  la - (nocc if odd_occ else 0))
                        colsl = (slice(0, nocc) if spec['ext_fam'] == 'o'
                                else slice(nocc, norb))
                        grids[sec][row_idx[0], row_idx[1], row_idx[2],
                                  colsl] += VU

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
