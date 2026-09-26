"""Band paths and Fourier interpolation (pbc_kpath).

Checks, in the order they matter:
  1. The interpolant is EXACT at the mesh points. It is an invertible square
     transform there, so this is interpolation and not a fit; if it were only
     approximate the whole construction would be the wrong one.
  2. It is exact EVERYWHERE for a band-limited function -- one that is already
     a trigonometric polynomial on the Born-von Karman lattice. That is the
     real statement of what the method does: it reconstructs exactly what it
     can represent, and the error on anything else is the part outside that
     span, not a tuning failure.
  3. It works on a 2D mesh (n x n x 1), which is the slab case and the one
     where a symmetry-based scheme would have needed the mesh subgroup rather
     than the space group. This method needs no operations at all.
  4. Damping trades exactness for smoothness, and does so visibly.
  5. Path segments are proportioned by TRUE reciprocal length, not by equal
     points per segment.
"""
import sys
import numpy as np
from pyscf.pbc import gto as pgto

sys.path.insert(0, __file__.rsplit('/tests/', 1)[0])
from src.SingleReference.Periodic.pbc_kpath import (        # noqa: E402
    band_path, bvk_vectors, fourier_interpolate, HIGH_SYMMETRY)


def check(ok, label, detail=''):
    """Report a condition AND enforce it.

    pytest DISCARDS whatever a test function returns, so a suite built out of
    `return ok` passes whether ok is True or False -- verified on pytest 9.1.1.
    Asserting here fixes every call site at once, and makes the script path
    stop at a genuine failure instead of printing FAIL and exiting 0.
    """
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    assert ok, f"{label}{(' -- ' + detail) if detail else ''}"
    return ok


def _al(dim=3, vac=20.0):
    a = 4.05
    cell = pgto.Cell()
    if dim == 3:
        h = a / 2
        cell.atom = [('Al', (0, 0, 0))]
        cell.a = np.array([[0, h, h], [h, 0, h], [h, h, 0]])
    else:
        ap = a / np.sqrt(2)
        cell.atom = [('Al', (0, 0, 0))]
        cell.a = np.array([[ap, 0, 0], [ap / 2, ap * np.sqrt(3) / 2, 0],
                           [0, 0, vac]])
    cell.basis, cell.pseudo, cell.dimension = 'gth-szv', 'gth-pade', dim
    cell.verbose = 0
    cell.build()
    return cell


def test_exact_at_mesh_points():
    cell = _al()
    ok = True
    for km in ([2, 2, 2], [3, 3, 3], [4, 4, 4]):
        kpts = cell.make_kpts(km)
        rng = np.random.default_rng(0)
        vals = rng.standard_normal((len(kpts), 3))       # 3 fake bands
        back = fourier_interpolate(cell, km, kpts, vals, kpts)
        err = np.abs(back - vals).max()
        ok &= check(err < 1e-10,
                    f'{km[0]}^3: exact at the mesh points it was built from',
                    f'{err:.1e}')
    return ok


def test_exact_for_band_limited():
    """A trigonometric polynomial on the BvK lattice is reproduced EVERYWHERE."""
    cell = _al()
    km = [4, 4, 4]
    kpts = cell.make_kpts(km)
    a = cell.lattice_vectors()
    R = [a[0], a[1], a[0] + a[1], a[2] - a[0]]
    w = [2.0, -3.0, 1.5, 0.75]

    def f(k):
        return sum(wi * np.cos(k @ Ri) for wi, Ri in zip(w, R))

    rng = np.random.default_rng(1)
    target = (rng.random((40, 3)) - 0.5) @ cell.reciprocal_vectors()
    got = fourier_interpolate(cell, km, kpts, f(kpts), target)
    err = np.abs(got - f(target)).max()
    ok = check(err < 1e-10,
               'and exact BETWEEN mesh points for a band-limited function',
               f'{err:.1e} at 40 random k')

    # ... while something outside the span is only approximated, which is the
    # honest statement of the method's limit rather than a hidden failure.
    g = lambda k: np.exp(-np.linalg.norm(k, axis=-1) ** 2)
    err2 = np.abs(fourier_interpolate(cell, km, kpts, g(kpts), target)
                  - g(target)).max()
    ok &= check(err2 > 1e-6,
                'and NOT exact for something outside that span -- the error is '
                'the unrepresentable part, not a tuning failure',
                f'{err2:.1e}')
    return ok


def test_real_input_gives_real_output():
    """Real data in, real array out -- on EVEN meshes especially.

    The centred BvK index set is lopsided for an even mesh: it holds the
    Nyquist component at +N/2 with no -N/2 partner, so the raw sum is complex
    between mesh points even for real input. Measured max|imag| 0.84 at 2^3 and
    1.27 at 4^3, against 1.5e-16 at 3^3 where the set IS symmetric.

    The old tests missed this because they only compared magnitudes, and
    |complex - real| is a perfectly good number. The consequence downstream was
    not cosmetic: json.dump raises partway through a complex array, so every
    attempt to write a band structure produced a TRUNCATED file.
    """
    cell = _al()
    ok = True
    rng = np.random.default_rng(7)
    for km in ([2, 2, 2], [3, 3, 3], [4, 4, 4], [6, 6, 6]):
        kpts = cell.make_kpts(km)
        vals = rng.standard_normal(len(kpts))
        out = fourier_interpolate(cell, km, kpts, vals, kpts)
        parity = 'even' if km[0] % 2 == 0 else 'odd'
        ok &= check(not np.iscomplexobj(out),
                    f'{km[0]}^3 ({parity}): real input gives a REAL array',
                    f'dtype {out.dtype}')
        ok &= check(np.abs(out - vals).max() < 1e-12,
                    f'{km[0]}^3: and taking Re() keeps it exact at the mesh',
                    f'{np.abs(out - vals).max():.1e}')
    # and it must round-trip through JSON, which is where this actually bit
    import json
    kpts = cell.make_kpts([4, 4, 4])
    out = fourier_interpolate(cell, [4, 4, 4], kpts,
                              rng.standard_normal(len(kpts)), kpts)
    try:
        json.dumps(out.tolist())
        ok &= check(True, 'and the result is JSON-serialisable')
    except TypeError as e:
        ok &= check(False, 'and the result is JSON-serialisable', str(e)[:50])
    return ok


def test_two_dimensional_mesh():
    cell = _al(dim=2)
    km = [4, 4, 1]
    kpts = cell.make_kpts(km)
    R, idx = bvk_vectors(cell, km)
    ok = check(len(R) == 16 and set(idx[:, 2]) == {0},
               'a 2D mesh gives one BvK vector per k-point, none out of plane',
               f'{len(R)} vectors, n3 in {sorted(set(idx[:, 2]))}')
    rng = np.random.default_rng(2)
    vals = rng.standard_normal(len(kpts))
    err = np.abs(fourier_interpolate(cell, km, kpts, vals, kpts) - vals).max()
    ok &= check(err < 1e-10, 'and interpolation is exact on it', f'{err:.1e}')
    return ok


def test_damping_trades_exactness():
    cell = _al()
    km = [3, 3, 3]
    kpts = cell.make_kpts(km)
    rng = np.random.default_rng(3)
    vals = rng.standard_normal(len(kpts))
    e0 = np.abs(fourier_interpolate(cell, km, kpts, vals, kpts) - vals).max()
    e1 = np.abs(fourier_interpolate(cell, km, kpts, vals, kpts,
                                    damping=1.0) - vals).max()
    return (check(e0 < 1e-10 and e1 > 1e-3,
                  'damping visibly gives up exactness at the mesh points',
                  f'{e0:.1e} undamped vs {e1:.1e} damped'))


def test_path_geometry():
    cell = _al()
    kpts, x, tick_x, labels = band_path(cell, 'fcc', 'G X W K G', npoints=120)
    ok = check(len(kpts) == len(x), 'one distance per k-point',
               f'{len(kpts)}')
    ok &= check(len(tick_x) == len(labels) == 5,
                'one tick per label', f'{len(tick_x)} ticks, {labels}')
    ok &= check(np.all(np.diff(x) >= -1e-12), 'distance is monotonic')

    b = cell.reciprocal_vectors()
    want = [np.linalg.norm((np.array(HIGH_SYMMETRY['fcc'][labels[i + 1]])
                            - np.array(HIGH_SYMMETRY['fcc'][labels[i]])) @ b)
            for i in range(len(labels) - 1)]
    got = np.diff(tick_x)
    ok &= check(np.abs(got - want).max() < 1e-10,
                'and segment lengths are the TRUE reciprocal distances',
                f'{np.abs(got - want).max():.1e}')
    # points are distributed by length, so density is roughly uniform
    dens = [np.sum((x >= tick_x[i]) & (x < tick_x[i + 1])) / want[i]
            for i in range(len(want))]
    ok &= check(max(dens) / min(dens) < 1.35,
                'points are spread by length, not equally per segment',
                f'density spread {max(dens) / min(dens):.2f}x')
    try:
        band_path(cell, 'fcc', 'G Z', npoints=10)
    except ValueError as e:
        ok &= check('unknown high-symmetry' in str(e),
                    'an unknown label is refused')
    else:
        ok &= check(False, 'an unknown label is refused')
    return ok


if __name__ == '__main__':
    all_ok = True
    print('\n-- 1. exactness at the mesh')
    all_ok &= test_exact_at_mesh_points()
    print('\n-- 2. exactness for a band-limited function')
    all_ok &= test_exact_for_band_limited()
    print('\n-- 2b. real input, real output')
    all_ok &= test_real_input_gives_real_output()
    print('\n-- 3. a 2D (slab) mesh')
    all_ok &= test_two_dimensional_mesh()
    print('\n-- 4. damping')
    all_ok &= test_damping_trades_exactness()
    print('\n-- 5. path geometry')
    all_ok &= test_path_geometry()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
