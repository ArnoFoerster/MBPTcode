"""Mini-BZ averaging of the interaction (W-av), the neighbourhood-valued route.

src/SingleReference/Periodic/pbc_wav.py. Every published route through the
long-wavelength limit is a different answer to what the momentum-transfer sum
should do near the origin, and this is the one that acts on the whole zone: the
interaction at each grid point is replaced by its average over the mini-BZ cell
around THAT point, with a uniform grid and no region-dependent parameter.

Checks:
  1. The quadrature: weights sum to 1, the polygon area is reproduced, a
     constant integrand is exact, and -- the cross-validation that matters --
     the 2D quadrature of 2pi/(q+kappa) agrees with pbc_smallq's CLOSED FORM
     for the same polygon to machine precision. Two independent routes to the
     same number, one numerical and one analytic.
  2. Convergence in the quadrature order, so `order` is a convergeable
     parameter rather than a tuned one.
  3. It acts at EVERY grid point, not only at Gamma. The damped kernel RINGS
     -- it is the transform of a sharply truncated theta(r)/r, with period
     2pi/r0, which under the AUTO scheme is EXACTLY 4 grid spacings at every
     mesh -- so the correction is substantial far from the origin and its SIGN
     follows the local curvature. Checked against the second-order prediction
     (1/2)<u^2> lap(v), sign changes included. That ringing is also why
     refining the k-grid never improves the sampling, and hence why both this
     scheme's and the constant approximation's head ratios come out
     mesh-invariant.
  4. The size of the correction, and the contrast with the constant
     approximation. Both are mesh-invariant under the AUTO scheme (q * r0 is),
     but W-av trims the Gamma head by 3.5% where the constant approximation
     cuts 19.4% -- because the constant approximation reads v at the
     NEIGHBOURING GRID POINT, which lies outside the Gamma cell entirely,
     while W-av averages over the cell the Gamma term actually stands for.
  5. A singular kernel is REFUSED, naming the interpolated auxiliary function
     that case needs, rather than being quadratured badly and silently.
  6. Structural: the kernel stays non-negative, the damping tag survives so the
     damping-support check still reaches it, and the builders accept it.
  8. THE INTERPOLANT HALF (`wav_bz_average`), for a quantity known only at
     grid points. Three properties define it and all three are pinned:
       * the mini-BZ cells TILE the zone, sum_q I0 = int_BZ f -- which needs
         each cell clipped to the zone and its images added back, because on
         an EVEN mesh a transfer sits on the zone boundary and f(|q|) is not
         periodic (before the fix the sum was 10% low at 2x2, 2.5% at 4x4, and
         exact at 3x3 -- the parity signature of exactly that);
       * EXACT when F is proportional to f, to machine precision at every mesh
         and for a singular law, where the uniform sum is 19-62% out;
       * identical to the uniform average when f is constant.
     Plus the accuracy gain on F = f * smooth, and the round trip through real
     per-q RPA correlation energies.
  7. IN-PLANE SUPERCELL FOLDING. A 2x2 primitive mini-BZ and a 1x1 supercell
     mini-BZ are the SAME region of absolute k-space, so an exactly averaged
     kernel must fold exactly as the bare one does. This is the sharpest
     available check that the averaging region is constructed in absolute
     momentum and not in some mesh-relative convention.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto as pgto, scf as pscf, tools

from src.SingleReference.Periodic.pbc_rpa import ri_rpa_ecorr_from_dfints
from src.SingleReference.Periodic.pbc_rpa_damping import (
    nyquist_params, make_coulG_damped, assert_damping_fits)
from src.SingleReference.Periodic.pbc_damped_integrals import build_dfintegrals_coulG
from src.SingleReference.Periodic.pbc_smallq import (
    minibz_polygon, polygon_area, head_average_polygon, smallest_transfer,
    make_coulG_constant_head, polygon_radial_moments, inplane_cell_area)
from src.SingleReference.Periodic.pbc_wav import (
    triangulate, polygon_quadrature, minibz_quadrature,
    make_coulG_minibz_averaged, transfer_grid, cell_moments, wav_bz_average,
    auxiliary_function, values_on_transfer_grid)


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


def make_slab(nrep, kmesh, mesh, scf=True):
    atoms = []
    for i in range(nrep):
        for j in range(nrep):
            atoms += [f'H {4.0 * i} {4.0 * j} -0.37', f'H {4.0 * i} {4.0 * j} 0.37']
    cell = pgto.Cell()
    cell.atom = '; '.join(atoms)
    cell.a = np.diag([4.0 * nrep, 4.0 * nrep, 24.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = mesh
    cell.verbose = 0
    cell.build()
    if not scf:
        return cell, None
    mf = pscf.KRHF(cell, cell.make_kpts(kmesh), exxdiv=None).density_fit()
    mf.kernel()
    return cell, mf


# ------------------------------------------------------------ 1/2. quadrature

def check_quadrature():
    cell, _ = make_slab(1, None, [13, 13, 72], scf=False)
    poly = minibz_polygon(cell, [2, 2, 1])

    ok = check(len(triangulate(poly)) == len(poly),
               'the centroid fan gives one triangle per edge')
    pts, wts = polygon_quadrature(poly, order=8)
    ok &= check(abs(wts.sum() - polygon_area(poly)) < 1e-14 * polygon_area(poly),
                'the quadrature weights sum to the polygon area',
                f'{wts.sum():.12f}')
    for order in (4, 8, 12):
        _, w = minibz_quadrature(cell, [2, 2, 1], order=order)
        ok &= check(abs(w.sum() - 1.0) < 1e-14,
                    f'order {order}: mini-BZ weights are normalised')

    # Cross-validation: the numerical 2D average of 2pi/(q+kappa) over the
    # mini-BZ against pbc_smallq's closed form for the same polygon. The two
    # share no code -- one integrates the radial part analytically per edge,
    # the other is a Duffy-mapped tensor Gauss rule on triangles.
    off, w = minibz_quadrature(cell, [2, 2, 1], order=16)
    q = np.linalg.norm(off, axis=1)
    for kappa in (0.5, 2.0, 10.0):
        num = float(np.dot(w, 2.0 * np.pi / (q + kappa)))
        ana = head_average_polygon(poly, kappa)
        ok &= check(abs(num / ana - 1.0) < 1e-12,
                    f'quadrature == closed form for kappa = {kappa}',
                    f'{num:.10f} vs {ana:.10f}')

    # A constant is exact, and a linear function integrates to its centre value.
    ok &= check(abs(float(np.dot(w, np.ones(len(w)))) - 1.0) < 1e-14,
                'a constant integrand is exact')
    ok &= check(abs(float(np.dot(w, off[:, 0]))) < 1e-14,
                'the cell average of a linear function is its centre value')

    # Convergence in `order` on the actual kernel.
    r0, beta, _ = nyquist_params(cell, [2, 2, 1])
    base = make_coulG_damped(r0, beta)
    ref = None
    errs = []
    for order in (2, 4, 8, 16):
        avg = make_coulG_minibz_averaged(base, cell, [2, 2, 1], order=order)
        val = float(avg(cell, np.zeros(3), np.zeros((1, 3)))[0])
        if order == 16:
            ref = val
        errs.append(val)
    errs = [abs(e / ref - 1.0) for e in errs[:-1]]
    ok &= check(errs[0] > errs[-1] and errs[-1] < 1e-6,
                'the averaged kernel converges in the quadrature order',
                ' -> '.join(f'{e:.1e}' for e in errs))
    return ok


# --------------------------------------------- 3/4. what the averaging does

def check_correction_profile():
    cell, _ = make_slab(1, None, [13, 13, 72], scf=False)
    ok = True
    print(f"       {'kmesh':>6} {'v(0)':>11} {'W-av':>11} {'ratio':>9} "
          f"{'CA':>11} {'ratio':>9}")
    wav_ratios, ca_ratios = [], []
    for n in (2, 3, 4):
        kmesh = [n, n, 1]
        r0, beta, _ = nyquist_params(cell, kmesh)
        base = make_coulG_damped(r0, beta)
        avg = make_coulG_minibz_averaged(base, cell, kmesh, order=8)
        ca = make_coulG_constant_head(base, smallest_transfer(cell, kmesh))
        head = np.zeros((1, 3))
        v0 = float(base(cell, np.zeros(3), head)[0])
        vw = float(avg(cell, np.zeros(3), head)[0])
        vc = float(ca(cell, np.zeros(3), head)[0])
        wav_ratios.append(vw / v0)
        ca_ratios.append(vc / v0)
        print(f"       {n}x{n}x1 {v0:11.4f} {vw:11.4f} {vw / v0:9.6f} "
              f"{vc:11.4f} {vc / v0:9.6f}")

    wav_ratios, ca_ratios = np.array(wav_ratios), np.array(ca_ratios)
    ok &= check(wav_ratios.ptp() < 1e-6,
                'the W-av head ratio is mesh-invariant, like the CA one '
                '(q * r0 is)', f'{wav_ratios[0]:.6f}, spread {wav_ratios.ptp():.1e}')
    ok &= check(wav_ratios[0] < 1.0, 'W-av trims the damped Gamma head',
                f'{100 * (1 - wav_ratios[0]):.1f}%')
    over = (1 - ca_ratios[0]) / (1 - wav_ratios[0])
    ok &= check(over > 4.0,
                'the constant approximation cuts far deeper, because it reads '
                'v at the NEIGHBOURING GRID POINT, outside the Gamma cell',
                f'{100 * (1 - ca_ratios[0]):.1f}% vs {100 * (1 - wav_ratios[0]):.1f}%, '
                f'{over:.1f}x')

    # The damped kernel RINGS: it is the transform of a sharply truncated
    # theta(r)/r, with period 2pi/r0. Under the AUTO scheme that is exactly 4
    # grid spacings at EVERY mesh, so refining the grid never resolves the
    # oscillation better -- which is why the ratios above are mesh-invariant,
    # and why averaging is not only a long-wavelength fix here.
    b1 = np.linalg.norm(cell.reciprocal_vectors()[0])
    for n in (2, 3, 4, 6):
        r0, _, _ = nyquist_params(cell, [n, n, 1])
        per_point = (2 * np.pi / r0) / (b1 / n)
        ok &= check(abs(per_point - 4.0) < 1e-9,
                    f'{n}x{n}: the ringing period is exactly 4 grid spacings',
                    f'{per_point:.6f}')

    # It is not a Gamma-point patch: the average differs from the kernel far
    # from the origin too, and it agrees with the second-order prediction
    # (1/2)<u^2> lap(v) -- sign changes included, which is the ringing.
    kmesh = [2, 2, 1]
    r0, beta, _ = nyquist_params(cell, kmesh)
    base = make_coulG_damped(r0, beta)
    avg = make_coulG_minibz_averaged(base, cell, kmesh, order=12)
    off, w = minibz_quadrature(cell, kmesh, order=12)
    m2 = float(np.dot(w, off[:, 0] ** 2))

    ks = np.array([0.6, 1.0, 2.2, 3.0])
    h = 2e-3
    pts = []
    for k in ks:                       # one array, so the table is built once
        pts += [[k, 0, 0], [k + h, 0, 0], [k - h, 0, 0], [k, h, 0], [k, -h, 0]]
    pts = np.array(pts, dtype=float)
    v = np.asarray(base(cell, np.zeros(3), pts), dtype=float)
    vb = np.asarray(avg(cell, np.zeros(3), pts), dtype=float)

    print(f"       {'|k|':>6} {'v':>10} {'delta':>11} {'Taylor':>11} {'ratio':>8}")
    signs, ratios = [], []
    for i, k in enumerate(ks):
        v0, vpx, vmx, vpy, vmy = v[5 * i:5 * i + 5]
        lap = (vpx - 2 * v0 + vmx) / h ** 2 + (vpy - 2 * v0 + vmy) / h ** 2
        pred = 0.5 * m2 * lap
        delta = vb[5 * i] - v0
        signs.append(np.sign(delta))
        ratios.append(delta / pred)
        print(f"       {k:6.2f} {v0:10.4f} {delta:11.6f} {pred:11.6f} "
              f"{delta / pred:8.4f}")
    ok &= check(all(abs(r - 1.0) < 0.15 for r in ratios),
                'the average matches (1/2)<u^2> lap(v) at every probe',
                ' '.join(f'{r:.3f}' for r in ratios))
    ok &= check(len(set(signs)) == 2,
                'and its SIGN changes with the ringing, so this is not a '
                'monotone long-wavelength patch',
                ' '.join(f'{int(s):+d}' for s in signs))
    ok &= check((vb > 0).all(), 'the averaged kernel stays positive')
    return ok


# --------------------------------------------------------- 5/6. structural

def check_guards_and_tags():
    cell, _ = make_slab(1, None, [13, 13, 72], scf=False)
    kmesh = [2, 2, 1]
    r0, beta, _ = nyquist_params(cell, kmesh)
    base = make_coulG_damped(r0, beta)
    avg = make_coulG_minibz_averaged(base, cell, kmesh, order=8)

    ok = check(getattr(avg, 'damping', None) == (r0, beta),
               'the damping tag survives, so the support check reaches it')
    ok &= check(assert_damping_fits(cell, cell.make_kpts(kmesh), avg) == [],
                'assert_damping_fits runs on the averaged kernel')
    ok &= check(getattr(avg, 'minibz_average', None) == ((2, 2, 1), 8),
                'the kernel records the mesh and order it was averaged with')

    bare = lambda c, q, Gv: tools.get_coulG(c, k=q, mesh=c.mesh, Gv=Gv)
    try:
        make_coulG_minibz_averaged(bare, cell, kmesh, order=8)
    except ValueError as err:
        ok &= check('SINGULAR' in str(err) and 'law=' in str(err),
                    'a singular kernel with no law is refused, naming the '
                    'parameter that lifts it',
                    str(err).split(',')[0][:70])
    else:
        ok &= check(False, 'a singular kernel is refused')
    return ok



# ------------------------------------- 6b. the singular kernel, once given f

def check_singular_kernel_with_law():
    """A singular kernel becomes averageable once its small-k form is supplied.

    This is the auxiliary-function half applied to the kernel wrapper. Only ONE
    grid point ever needs it -- |q+G| vanishes only at q = Gamma with G = 0 --
    so the test checks three separate things: that the head is right, that
    everywhere else is untouched, and that a law which does not match the
    kernel is rejected rather than absorbed into a plausible-looking head.

    The head is checked against `head_average_polygon`, which integrates
    2pi/(k+kappa) over the same polygon by its OWN quadrature. Two independent
    routes -- signed-triangle radial moments here, direct polygon quadrature
    there -- so agreement is evidence and not a restatement.
    """
    cell, _ = make_slab(1, None, [13, 13, 72], scf=False)
    kmesh = [2, 2, 1]
    law = _law(0.0)                       # 2pi/k, the bare 2D head

    def bare2d(c, q, Gv):
        """A kernel that IS its own law, so A == 1 and the scheme is exact."""
        k = np.linalg.norm(np.asarray(q)[None, :] + np.asarray(Gv), axis=1)
        return 2.0 * np.pi / np.maximum(k, 1e-300)

    try:
        make_coulG_minibz_averaged(bare2d, cell, kmesh, order=8)
    except ValueError as err:
        ok = check('SINGULAR' in str(err) and 'law=' in str(err),
                   'without a law the singular kernel is still refused')
    else:
        ok = check(False, 'without a law the singular kernel is refused')

    avg = make_coulG_minibz_averaged(bare2d, cell, kmesh, order=8, law=law)
    got = float(avg(cell, np.zeros(3), np.zeros((1, 3)))[0])
    expect = head_average_polygon(minibz_polygon(cell, kmesh), kappa=0.0,
                                  nquad=256)
    ok &= check(np.isfinite(got),
                'the head is finite where the kernel itself is not',
                f'{got:.8f}')
    ok &= check(abs(got - expect) / abs(expect) < 1e-6,
                'and equals the exact cell average of the law, by an '
                'independent quadrature',
                f'{got:.8f} vs {expect:.8f}')

    # Everywhere else must be bit-for-bit what plain quadrature already gave:
    # those cells sit a full |G| from the origin.
    offs, wts = minibz_quadrature(cell, kmesh, order=8)
    pad = np.zeros((len(offs), 3))
    pad[:, :2] = offs
    Gv = np.array([[0.4, 0.0, 0.0], [0.0, 0.55, 0.0], [0.3, 0.3, 0.0]])
    manual = np.array([wts @ bare2d(cell, np.zeros(3), g[None, :] + pad)
                       for g in Gv])
    # Relative, not exact: the wrapper contracts a whole block of G against the
    # weights in one matmul while this loop does one G at a time, so the sum is
    # reassociated and the last bit can differ. One ulp is agreement here; an
    # exact-equality bar would be testing BLAS, not the code.
    off = np.abs(avg(cell, np.zeros(3), Gv) - manual).max() / np.abs(manual).max()
    ok &= check(off < 1e-14,
                'off-Gamma points are plain quadrature, untouched by the law',
                f'{off:.1e} relative')

    # A law that is not this kernel's singularity: v/f is not constant over the
    # cell, and integrating f exactly would not be integrating v.
    wrong = lambda k: 4.0 * np.pi / np.maximum(
        np.asarray(k, dtype=float), 1e-300) ** 2
    try:
        bad = make_coulG_minibz_averaged(bare2d, cell, kmesh, order=8, law=wrong)
        bad(cell, np.zeros(3), np.zeros((1, 3)))
    except ValueError as err:
        ok &= check('varies' in str(err) and 'law' in str(err),
                    'a mismatched law is rejected, not absorbed into a '
                    'plausible head', str(err).split(',')[0][:60])
    else:
        ok &= check(False, 'a mismatched law is rejected')

    # And a FINITE kernel must be entirely unaffected by the new parameter.
    r0, beta, _ = nyquist_params(cell, kmesh)
    damped = make_coulG_damped(r0, beta)
    Gt = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.45, 0.0]])
    a = make_coulG_minibz_averaged(damped, cell, kmesh, order=8)
    b = make_coulG_minibz_averaged(damped, cell, kmesh, order=8, law=law)
    ok &= check(np.abs(a(cell, np.zeros(3), Gt)
                       - b(cell, np.zeros(3), Gt)).max() == 0.0,
                'a finite kernel ignores the law entirely -- bit-identical')
    ok &= check(getattr(a, 'singular_law', 'x') is None
                and getattr(avg, 'singular_law', None) is law,
                'and the tag records which route the kernel took')
    return ok


# ------------------------------------------------------- 8. the interpolant

def _law(kappa):
    """f(k) = 2pi/(k + kappa); kappa = 0 is the singular 2D head."""
    if kappa == 0.0:
        return lambda k: 2.0 * np.pi / np.maximum(np.asarray(k, dtype=float), 1e-300)
    return lambda k: 2.0 * np.pi / (np.asarray(k, dtype=float) + kappa)


def _bz_integral(cell, law):
    """int_BZ f d^2k, by the same signed-triangle route on the whole zone."""
    return polygon_radial_moments(minibz_polygon(cell, [1, 1, 1]), law,
                                  nang=96, nrad=96)[0]


def check_interpolant():
    cell, _ = make_slab(1, None, [13, 13, 72], scf=False)
    area_bz = (2 * np.pi) ** 2 / inplane_cell_area(cell)
    ok = True

    # (a) the cells tile the zone -- including the EVEN meshes, which is what
    # the clip-and-image-back construction is for.
    for n in (2, 3, 4, 5):
        for kappa in (0.0, 2.0):
            I0, _ = cell_moments(cell, [n, n, 1], _law(kappa), nang=48, nrad=48)
            ref = _bz_integral(cell, _law(kappa))
            ok &= check(abs(I0.sum() / ref - 1.0) < 1e-12,
                        f'{n}x{n} kappa={kappa}: the mini-BZ cells tile the zone',
                        f'{abs(I0.sum() / ref - 1.0):.1e}')

    # (b) exact when F is proportional to f -- the defining property.
    print(f"       {'mesh':>5} {'kappa':>6} {'W-av':>10} {'uniform':>10}")
    for n in (2, 3, 4, 6):
        for kappa in (0.0, 2.0):
            law = _law(kappa)
            q = transfer_grid(cell, [n, n, 1])
            qabs = np.linalg.norm(q, axis=-1)
            F = np.where(qabs > 1e-10, law(np.maximum(qabs, 1e-300)),
                         0.0 if kappa == 0.0 else float(law(np.zeros(1))[0]))
            got = wav_bz_average(F, cell, [n, n, 1], law, nang=48, nrad=48)
            exact = _bz_integral(cell, law) / area_bz
            rel, urel = abs(got / exact - 1.0), abs(F.mean() / exact - 1.0)
            print(f"       {n}x{n} {kappa:6.1f} {rel:10.1e} {urel:10.1e}")
            ok &= check(rel < 1e-12,
                        f'{n}x{n} kappa={kappa}: EXACT for F proportional to f')
            if kappa == 0.0:
                ok &= check(urel > 0.1,
                            f'{n}x{n}: and the uniform sum is not, by a long way',
                            f'{100 * urel:.0f}% out')

    # (c) a constant law reduces to the plain uniform average.
    one = lambda k: np.ones_like(np.asarray(k, dtype=float))
    rng = np.random.default_rng(0)
    for n in (2, 3, 4):
        F = rng.normal(size=(n, n))
        got = wav_bz_average(F, cell, [n, n, 1], one, nang=32, nrad=32)
        ok &= check(abs(got - F.mean()) < 1e-13,
                    f'{n}x{n}: a constant law reduces to the uniform average',
                    f'{abs(got - F.mean()):.1e}')

    # (d) the accuracy gain on F = f * smooth, against a fine quadrature.
    law = _law(2.0)
    smooth = lambda v: 1.0 + 0.3 * np.cos(v[..., 0] * 2.0) + 0.2 * v[..., 1] ** 2
    pts, wts = polygon_quadrature(minibz_polygon(cell, [1, 1, 1]), order=60)
    ref = float(np.dot(wts, law(np.linalg.norm(pts, axis=1)) * smooth(pts)) / area_bz)
    print(f"       {'mesh':>5} {'uniform':>10} {'W-av':>10} {'gain':>7}")
    gains = []
    for n in (2, 3, 4, 6, 8):
        q = transfer_grid(cell, [n, n, 1])
        qabs = np.linalg.norm(q, axis=-1)
        F = np.where(qabs > 1e-10, law(np.maximum(qabs, 1e-300)) * smooth(q),
                     float(law(np.zeros(1))[0]) * float(smooth(np.zeros((1, 2)))[0]))
        got = wav_bz_average(F, cell, [n, n, 1], law, nang=32, nrad=32)
        eu, ew = abs(F.mean() / ref - 1.0), abs(got / ref - 1.0)
        gains.append(eu / ew)
        print(f"       {n}x{n} {eu:10.2e} {ew:10.2e} {eu / ew:6.1f}x")
    ok &= check(min(gains) > 2.0,
                'W-av beats the uniform sum at every mesh on f * smooth',
                f'{min(gains):.1f}x to {max(gains):.1f}x')

    # (e) the Gamma auxiliary value comes from the neighbour shell when the law
    # is singular -- which is the whole reason the auxiliary function exists.
    law = _law(0.0)
    q = transfer_grid(cell, [3, 3, 1])
    qabs = np.linalg.norm(q, axis=-1)
    F = np.where(qabs > 1e-10, law(np.maximum(qabs, 1e-300)) * 2.5, -999.0)
    A = auxiliary_function(F, cell, [3, 3, 1], law)
    gamma = np.unravel_index(np.argmin(qabs), qabs.shape)
    ok &= check(abs(A[gamma] - 2.5) < 1e-12,
                'A(Gamma) is extrapolated from the neighbour shell, not read '
                'off the code\'s own Gamma convention', f'{A[gamma]:.6f}')
    return ok


def check_interpolant_on_real_data():
    """Round trip through actual per-q RPA correlation energies."""
    kmesh = [2, 2, 1]
    cell, mf = make_slab(1, kmesh, [13, 13, 72])
    r0, beta, _ = nyquist_params(cell, kmesh)
    coulG = make_coulG_damped(r0, beta)
    dfints = build_dfintegrals_coulG(mf, coulG_fn=coulG)
    ec, per_q = ri_rpa_ecorr_from_dfints(dfints, nw=24, return_per_q=True)

    grid = values_on_transfer_grid(per_q, cell, mf.kpts, kmesh)
    ok = check(abs(grid.mean() - ec) < 1e-12,
               'per-q values map onto the transfer grid and re-average to Ec',
               f'{grid.mean():.10f} vs {ec:.10f}')

    one = lambda k: np.ones_like(np.asarray(k, dtype=float))
    flat = wav_bz_average(grid, cell, kmesh, one, nang=24, nrad=24)
    ok &= check(abs(flat - ec) < 1e-12,
                'and a constant law reproduces the plain q-sum exactly',
                f'{flat:.10f}')

    # With the damped kernel's OWN radial profile as the law -- the sensible
    # choice here, since that is what makes Ec(q) vary rapidly (it rings) --
    # the estimate moves. This exercises the seam on real data; the size of the
    # move is not a claim about accuracy, which needs a metal.
    law = lambda k: np.asarray(
        coulG(cell, np.zeros(3), np.stack([np.asarray(k, dtype=float),
                                           np.zeros(np.size(k)),
                                           np.zeros(np.size(k))], axis=1)),
        dtype=float)
    shifted = wav_bz_average(grid, cell, kmesh, law, nang=24, nrad=24)
    ok &= check(np.isfinite(shifted) and abs(shifted - ec) > 1e-8,
                'a real small-q law moves the q-sum',
                f'Ec {ec:.8f} -> {shifted:.8f} ({100 * (shifted / ec - 1):+.2f}%)')

    try:
        values_on_transfer_grid(per_q[:-1], cell, mf.kpts, kmesh)
    except ValueError:
        ok &= check(True, 'a mismatched per-q array is refused')
    else:
        ok &= check(False, 'a mismatched per-q array is refused')
    return ok


# ------------------------------------------------------------- 7. folding

def check_inplane_folding():
    """A 2x2 primitive mini-BZ and a 1x1 supercell mini-BZ are the same region
    of absolute k-space, so the averaged kernel must fold exactly."""
    prim, mf_p = make_slab(1, [2, 2, 1], [13, 13, 72])
    sup, mf_s = make_slab(2, [1, 1, 1], [26, 26, 72])

    poly_p = minibz_polygon(prim, [2, 2, 1])
    poly_s = minibz_polygon(sup, [1, 1, 1])
    ok = check(abs(polygon_area(poly_p) / polygon_area(poly_s) - 1.0) < 1e-12,
               'the folded pair share one mini-BZ region',
               f'{polygon_area(poly_p):.8f} vs {polygon_area(poly_s):.8f}')

    r0p, bp, _ = nyquist_params(prim, [2, 2, 1])
    r0s, bs, _ = nyquist_params(sup, [1, 1, 1])
    ok &= check(abs(r0p - r0s) < 1e-9, 'and one damping radius')
    ok &= check(abs(mf_p.e_tot - mf_s.e_tot / 4) < 1e-6, 'mean field folds',
                f'{abs(mf_p.e_tot - mf_s.e_tot / 4):.1e}')

    rel = {}
    for tag in ('plain', 'W-av'):
        ec = []
        for cell, mf, km, r0, beta, n in ((prim, mf_p, [2, 2, 1], r0p, bp, 1),
                                          (sup, mf_s, [1, 1, 1], r0s, bs, 4)):
            cg = make_coulG_damped(r0, beta)
            if tag == 'W-av':
                cg = make_coulG_minibz_averaged(cg, cell, km, order=8)
            dfints = build_dfintegrals_coulG(mf, coulG_fn=cg)
            ec.append(ri_rpa_ecorr_from_dfints(dfints, nw=24) / n)
        rel[tag] = abs(ec[0] - ec[1]) / abs(ec[0])
        print(f'       {tag:6s}: Ec/cell primitive {ec[0]:.10f}  '
              f'supercell {ec[1]:.10f}')
    ok &= check(rel['W-av'] < 3 * max(rel['plain'], 1e-12),
                'the AVERAGED kernel folds as exactly as the plain one',
                f"W-av {rel['W-av']:.1e} vs plain {rel['plain']:.1e}")
    return ok


def run():
    all_ok = True
    print('\n-- 1/2. the mini-BZ quadrature')
    all_ok &= check_quadrature()
    print('\n-- 3/4. what the averaging does, and how it compares to CA')
    all_ok &= check_correction_profile()
    print('\n-- 5/6. guards and tags')
    all_ok &= check_guards_and_tags()
    print('\n-- 6b. a singular kernel, once given its law')
    all_ok &= check_singular_kernel_with_law()
    print('\n-- 8. the interpolant half')
    all_ok &= check_interpolant()
    print('\n-- 8b. the interpolant on real per-q RPA data')
    all_ok &= check_interpolant_on_real_data()
    print('\n-- 7. in-plane supercell folding')
    all_ok &= check_inplane_folding()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return all_ok


def test_pbc_wav_checks():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
