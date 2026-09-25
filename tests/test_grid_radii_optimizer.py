"""What `optimize_atomic_radii` has to get right, against the published tables.

Duchemin & Blase's SI Tables S8-S11 are ground truth for H, C, N, O at
cc-pVTZ, so the optimizer can be held to them rather than to its own output.
Two properties are checked there and the rest on the search itself.

The objective is multi-modal in the radii and the only lever measured to help
is start diversity. Carbon/cc-pVDZ at 148 counts, one descent per starting
shape, everything else at production settings:

    7.59e-02   4.50e-04   3.03e-03   1.03e-01   1.47e-02   2.61e-02

The first is the geometric ladder the single descent uses, and it is the worst
of the six -- it returns carbon's A3 shell as (0.199, 0.202), two coincident
radii duplicating 12 of its 36 points. Best-of-four is ~163x better for four
times the offline cost, paid once per (element, basis, counts) and cached.

Three things that look like fixes are not. A monotone reparametrization making
coincident radii unreachable was no better than the free one. Widening r_max
acts only by moving where the starts land: nothing sits on the bound, and the
result is not monotone in it. And sampling the nuclear cusp, which is worth
2.2-5.5x on the published tables, is worth nothing at 148 counts -- best-of-six
reaches 1.40e-03 with it against 4.50e-04 without, because a grid that has no
cusp point already spends a radius near zero doing that job.

Absolute fit errors are NOT pinned. The optimizer is a numerically
differentiated descent under threaded BLAS, so its result moves with the thread
count -- the same reason `shipped_radii` exists. Everything here is a ratio or
a structural property.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import df, gto
from threadpoolctl import threadpool_limits

from src.Base.separable_ri import (ELEMENT_R_MAX, LEGACY_R_MAX, _START_SHAPES,
                                   _shipped_key, _start_radii, atomic_points,
                                   fit_error_coulomb, lebedev_subshells,
                                   optimize_atomic_radii, published_grids,
                                   search_r_max, shipped_radii)

#: The production grid: 148 points/atom, which is where the collapse showed up.
COUNTS_148 = {'A1': 8, 'A2': 5, 'A3': 3, 'B1': 1}


def _atom(el, basis):
    mol = gto.M(atom=f'{el} 0 0 0', basis=basis, verbose=0,
                spin=gto.charge(el) % 2)
    return mol, df.addons.make_auxmol(mol, auxbasis=f'{basis}-ri')


def _min_log_gap(radii):
    """Smallest adjacent ratio - 1 over all shells; 0 means two radii coincide."""
    worst = np.inf
    for r in radii.values():
        v = np.sort(np.atleast_1d(r))
        if len(v) > 1:
            worst = min(worst, float((v[1:] / v[:-1] - 1.0).min()))
    return worst


if __name__ == '__main__':
    lebedev_subshells()
    all_ok = True

    # 1. The nuclear cusp point is worth having, on the grids optimized around
    #    one. Every published table starts with a bare 0 0 0 entry; the
    #    fallback path omits it, and the optimizer's objective has to be able to
    #    see it or it cannot reproduce those tables.
    print('1. the cusp point on published tables (origin=True vs False):')
    ok1, worst_gain = True, np.inf
    pub = published_grids()
    for el in ('H', 'C', 'N', 'O'):
        radii, _ = pub[el]
        mol, aux = _atom(el, 'cc-pvtz')
        e = {f: fit_error_coulomb(mol, aux, atomic_points(radii, origin=f))
             for f in (True, False)}
        worst_gain = min(worst_gain, e[False] / e[True])
        print(f'   {el:2s} {len(atomic_points(radii, origin=True)):4d} pts   '
              f'with {e[True]:.3e}   without {e[False]:.3e}   '
              f'{e[False] / e[True]:5.1f}x')
    ok1 = worst_gain > 2.0
    all_ok &= ok1
    print(f'   smallest gain {worst_gain:.1f}x   {"OK" if ok1 else "FAIL"}')

    # 2. n_start=1 must BE the old single geometric descent, or every cached and
    #    shipped grid silently changes meaning.
    print('\n2. n_start=1 is the plain geometric ladder:')
    s0 = _start_radii(COUNTS_148, 0.05, 5.0, 0)
    ref = {n: np.geomspace(0.05 * 2, 5.0 * 0.8, k)
           for n, k in COUNTS_148.items() if k}
    d = max(float(np.abs(s0[n] - ref[n]).max()) for n in ref)
    ok2 = d < 1e-12
    all_ok &= ok2
    print(f'   max |shape 0 - geomspace(2 r_min, 0.8 r_max)| = {d:.2e}   '
          f'{len(_START_SHAPES)} shapes available   {"OK" if ok2 else "FAIL"}')

    # 3. Start diversity, at PRODUCTION settings -- origin=False and the
    #    default box. Both of the tempting variations hide the defect: with the
    #    cusp sampled, or with r_max widened to 8, carbon's single start lands
    #    in a good basin by itself and there is nothing left to rescue.
    #
    #    Both calls must run the LIVE search, bypassing the shipped table AND
    #    the git-ignored radii cache; `return_candidates=True` is the
    #    documented bypass of both, since a caller that wants the runners-up
    #    has to run the search. The table holds ONE grid per physical request,
    #    so a lookup answers n_start=1 and n_start=4 with the same row and the
    #    comparison measures nothing (it reported 1.0x before the bypass); a
    #    stale cache silently reports its own thread count's minimum (see
    #    optimize_atomic_radii). One BLAS thread, since the objective is a
    #    numerically differentiated descent under threaded BLAS and a
    #    different reduction order moves L-BFGS-B onto a different local
    #    minimum. The ratio still varies by machine even from an empty cache
    #    (measured 15x-44x across CPUs/OpenBLAS kernels at one thread), so the
    #    bound is kept loose -- this checks that multi-start helps at all, not
    #    by how much.
    print('\n3. multi-start rescues carbon at 148 counts, production settings:')
    with threadpool_limits(limits=1, user_api='blas'):
        e_one = optimize_atomic_radii('C', 'cc-pvdz', 'cc-pvdz-ri',
                                      counts=COUNTS_148, maxiter=110, n_start=1,
                                      return_candidates=True)[1]
        r_many, e_many, _ = optimize_atomic_radii('C', 'cc-pvdz', 'cc-pvdz-ri',
                                                  counts=COUNTS_148, maxiter=110,
                                                  n_start=4,
                                                  return_candidates=True)
    ok3 = e_many < e_one / 5.0
    all_ok &= ok3
    print(f'   1 start {e_one:.3e}   4 starts {e_many:.3e}   '
          f'{e_one / e_many:.1f}x   {"OK" if ok3 else "FAIL"}')

    # 4. And the multi-start optimum must not be a collapsed grid: coincident
    #    radii duplicate whole Lebedev shells, which spends points on nothing.
    print('\n4. no shell of the optimum has coincident radii:')
    gap = _min_log_gap(r_many)
    ok4 = gap > 0.02
    all_ok &= ok4
    for name in ('A1', 'A2', 'A3', 'B1'):
        v = np.sort(np.atleast_1d(r_many[name]))
        print(f'   {name}: {np.array2string(v, precision=3, floatmode="fixed")}')
    print(f'   smallest adjacent ratio - 1 = {gap:.3f}   {"OK" if ok4 else "FAIL"}')

    # 5. The table is keyed on the PHYSICS and on nothing else, so one physical
    #    grid has one row. Keying on the optimizer recipe as well is what left
    #    up to nine rows per request, chosen between at run time by
    #    `search_r_max` -- a heuristic that picked the worse box in 272 of 612
    #    cases. There must be no way to spell a recipe into a key any more.
    print('\n5. the table key is the physics alone:')
    key = _shipped_key('C', 'cc-pvdz', 'cc-pvdz-ri', COUNTS_148)
    table = shipped_radii()
    reachable = all(
        _shipped_key(row['element'], row['basis'], row['auxbasis'],
                     row['counts']) == k
        for k, row in table.items())
    # Each row must name a grid whose radii match its counts, or a lookup hands
    # back a shape the fit cannot place.
    shaped = all(
        all(len(row['radii'][n]) == row['counts'][n]
            for n in ('A1', 'A2', 'A3', 'B1') if n in row['radii'])
        for row in table.values())
    # A cusp-sampled row must refuse to answer a caller who asked for no cusp:
    # the point counts differ by one, so substituting silently is a different
    # grid. The transcribed Duchemin & Blase rows are the ones that set it.
    cusp_guarded = False
    try:
        optimize_atomic_radii('C', 'cc-pvtz', 'cc-pvtz-ri',
                              counts={'A1': 9, 'A2': 9, 'A3': 7, 'B1': 4},
                              origin=False)
    except ValueError:
        cusp_guarded = True
    ok5 = (key == 'C|cc-pvdz|cc-pvdz-ri|8,5,3,1' and reachable and shaped
           and cusp_guarded)
    all_ok &= ok5
    print(f'   key spells the physics: {key}')
    print(f'   every row reachable from its own fields: {reachable}   '
          f'radii match counts: {shaped}   cusp row guarded: {cusp_guarded}   '
          f'{"OK" if ok5 else "FAIL"}')

    # 6. The box rule: one descent keeps the legacy box for every element, so
    #    nothing cached changes meaning; a multi-start search widens it only
    #    where the density asks for it.
    print('\n6. search box: legacy for one descent, per element otherwise:')
    ok6 = (all(search_r_max(el, 1) == LEGACY_R_MAX for el in ELEMENT_R_MAX)
           and search_r_max('C', 8) == LEGACY_R_MAX
           and search_r_max('Mg', 8) == 16.0
           and search_r_max('Mg', 8, r_max=7.0) == 7.0
           and search_r_max('Xx', 8) == 12.0)
    all_ok &= ok6
    print(f'   n_start=1 -> {LEGACY_R_MAX} for all {len(ELEMENT_R_MAX)} elements; '
          f'n_start=8 -> C {search_r_max("C", 8)}, Mg {search_r_max("Mg", 8)}; '
          f'explicit wins; unlisted -> 12.0   {"OK" if ok6 else "FAIL"}')

    # 7. return_candidates must survive a CACHE HIT. The table and the cache
    #    hold only the best-by-fit grid, so the early returns cannot serve a
    #    caller who wants every local minimum.
    print('\n7. return_candidates on a cold call and on a cache hit:')
    kw = dict(counts=COUNTS_148, n_start=2, maxiter=15)
    first = optimize_atomic_radii('H', 'cc-pvdz', 'cc-pvdz-ri',
                                  return_candidates=True, **kw)
    second = optimize_atomic_radii('H', 'cc-pvdz', 'cc-pvdz-ri',
                                   return_candidates=True, **kw)
    plain = optimize_atomic_radii('H', 'cc-pvdz', 'cc-pvdz-ri', **kw)
    ok7 = (len(first) == 3 and len(second) == 3 and len(plain) == 2
           and len(first[2]) == 2
           and abs(first[1] - min(f for _, f in first[2])) < 1e-15)
    all_ok &= ok7
    print(f'   cold: {len(first)} values, {len(first[2])} candidates; cache hit: '
          f'{len(second)} values; plain call: {len(plain)} values; best == min '
          f'over candidates   {"OK" if ok7 else "FAIL"}')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
