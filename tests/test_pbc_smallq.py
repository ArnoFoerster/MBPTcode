"""The small-q limits of a 2D metal, and the mini-BZ measure to apply them.

src/SingleReference/Periodic/pbc_smallq.py. A slab is not a small 3D solid in
the long-wavelength limit: the bare interaction is 2pi/q, the dielectric head
diverges as 1/q rather than 1/q^2, the screened head is FINITE, and the plasmon
disperses as sqrt(q). Porting a 3D expression gives numbers that look
reasonable and are wrong, so every form here is checked against a published
limit rather than against itself.

Checks:
  1. Stern's 2D electron gas (PRL 18, 546 (1967)) as the reference limit:
     the static polarizability is CONSTANT below q = 2k_F, has the sqrt cusp
     at 2k_F, and falls as 1/q^2 above it. Feeding it through
     eps = 1 - v Pi reproduces 1 + kappa/q with kappa = 2 EXACTLY -- which is
     what fixes KAPPA_2DEG_FREE_ELECTRON and validates thomas_fermi_kappa_2d.
     The screened head v/eps is then 2pi/(q + kappa), finite at the origin.
  2. The mini-BZ measure: the disc average against numerical 2D quadrature,
     its two closed-form limits, and the mesh scaling of the disc radius.
  3. The disc average is NOT the 3D sphere average. Pinned deliberately: the
     published Gaussian-basis finite-size correction integrates over a sphere,
     which for a slab is the wrong domain, and the two differ by more than a
     convention -- they do not even have the same units of R.
  4. The sqrt(q) plasmon, and that it tends to ZERO rather than to the 3D
     constant.
  4b. The EXACT mini-BZ (Wigner-Seitz) average, against the disc it replaces
     and against 2D quadrature, plus an anisotropy check: a direction-dependent
     kappa is NOT the same as its
     angular mean, which is what makes an isotropic sqrt(q) form "the right
     scaling law with the wrong coefficient".
  5. The constant approximation kernel: it moves ONLY the head, keeps the
     kernel non-negative, carries the damping tag through so the
     damping-support check still reaches it, and reduces the AUTO-damped v(0) by a factor that is
     the SAME at every k-mesh -- because q_min * r0 = pi * frac_r0 is
     mesh-independent under the AUTO scheme. That invariance is the point: the
     constant approximation is a fixed redefinition of the Gamma head, not a
     correction that vanishes in the thermodynamic limit.
  6. Admissibility on a real slab, which is the result the module exists to
     record: with the reaction field present, the head channel admits a value
     only up to a geometric ceiling. The bare mini-BZ head and the constant
     approximation both exceed it and drive the whitened metric indefinite;
     the METAL-SCREENED head does not. The ceiling is unchanged by the z-mesh,
     so it is not the "number of G_z points" the screening module's docstring
     blamed.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.df import ft_ao

from src.SingleReference.Periodic.pbc_rpa import make_auxcell, _bz_index
from src.SingleReference.Periodic.pbc_rpa_damping import (
    nyquist_params, make_coulG_damped, assert_damping_fits)
from src.SingleReference.Periodic.pbc_solvent_screening import (
    SlabDielectricEnvironment, build_dfintegrals_screened, solvent_cohsex_kpts,
    reaction_field_form, _planar_grid)
from src.SingleReference.Periodic.pbc_damped_integrals import build_dfintegrals_coulG
from src.SingleReference.Periodic.pbc_smallq import (
    KAPPA_2DEG_FREE_ELECTRON, WING_EXPONENTS_2D_METAL, inplane_cell_area,
    minibz_disc_radius, head_average_2d, sphere_average_3d,
    thomas_fermi_kappa_2d, polarizability_2deg, density_2deg,
    dielectric_head_2d, w_head_2d, plasmon_frequency_2d, slab_head_value,
    make_coulG_constant_head, smallest_transfer, minibz_polygon,
    polygon_area, head_average_polygon)

from src.Base.constants import HARTREE_TO_EV
from src.Base.constants import BOHR_TO_ANGSTROM


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


def make_slab(kmesh, mesh):
    """A layer of H2 units with vacuum along z."""
    cell = pgto.Cell()
    cell.atom = 'H 0 0 -0.37; H 0 0 0.37'
    cell.a = np.diag([4.0, 4.0, 24.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = mesh
    cell.verbose = 0
    cell.build()
    mf = pscf.KRHF(cell, cell.make_kpts(kmesh), exxdiv=None).density_fit()
    mf.kernel()
    return cell, mf


# ---------------------------------------------------------------- 1. Stern

def test_stern_reference_limits():
    kf = 0.7
    dos = 1.0 / np.pi                       # 2D free-electron DOS, a.u.

    below = np.array([1e-4, 0.05, 0.3, 0.9, 1.399])
    Pi = polarizability_2deg(below, kf)
    ok = check(np.abs(Pi + dos).max() < 1e-14,
               'Stern Pi is exactly -N(0) below q = 2k_F',
               f'max dev {np.abs(Pi + dos).max():.1e}')

    # eps = 1 - v Pi with v = 2pi/q must be EXACTLY 1 + 2/q there. This is what
    # fixes kappa = 2 a.u. and it is density-independent, because the 2D
    # density of states is.
    eps = 1.0 - (2.0 * np.pi / below) * Pi
    ok &= check(np.abs(eps - dielectric_head_2d(below, 2.0)).max() < 1e-13,
                'eps = 1 - v Pi reproduces 1 + kappa/q with kappa = 2')
    ok &= check(thomas_fermi_kappa_2d(dos) == KAPPA_2DEG_FREE_ELECTRON,
                'thomas_fermi_kappa_2d(1/pi) == KAPPA_2DEG_FREE_ELECTRON')
    # Density-independence, explicitly -- but only where every k_F keeps the
    # point below its own 2k_F, since that is where Pi is constant at all.
    others = (0.3, 1.5)
    shared = below[below < 2.0 * min(others + (kf,))]
    eps_shared = 1.0 - (2.0 * np.pi / shared) * polarizability_2deg(shared, kf)
    for kf2 in others:
        e2 = 1.0 - (2.0 * np.pi / shared) * polarizability_2deg(shared, kf2)
        ok &= check(np.abs(e2 - eps_shared).max() < 1e-13,
                    f'kappa does not depend on the density (k_F = {kf2})',
                    f'{len(shared)} points below 2k_F for every k_F')

    # v/eps is the screened head, finite at the origin.
    ok &= check(np.abs((2.0 * np.pi / below) / eps - w_head_2d(below, 2.0)).max() < 1e-13,
                'v/eps == 2pi/(q + kappa), the screened head')
    ok &= check(abs(w_head_2d(0.0, 2.0) - np.pi) < 1e-14,
                'the screened head is FINITE at q = 0', f'W(0) = {w_head_2d(0.0, 2.0):.6f}')

    # The 2k_F cusp: Pi is continuous but its deviation grows as sqrt(delta),
    # so the derivative is infinite. Check the exponent, not just continuity.
    deltas = np.array([1e-6, 1e-5, 1e-4, 1e-3])
    jump = np.abs(polarizability_2deg(2 * kf + deltas, kf) + dos)
    slope = np.polyfit(np.log(deltas), np.log(jump), 1)[0]
    ok &= check(abs(slope - 0.5) < 1e-3, 'Pi has a sqrt cusp at q = 2k_F',
                f'exponent {slope:.5f}')

    # Large q: Pi -> -(1/pi) 2 k_F^2/q^2.
    big = np.array([50.0, 100.0, 200.0]) * kf
    ratio = polarizability_2deg(big, kf) / (-dos * 2.0 * kf ** 2 / big ** 2)
    ok &= check(np.abs(ratio - 1.0).max() < 2e-3,
                'Pi falls as 1/q^2 above 2k_F', f'ratio {ratio[-1]:.6f}')

    ok &= check(abs(density_2deg(kf) - kf ** 2 / (2 * np.pi)) < 1e-15,
                'density_2deg is k_F^2/2pi (both spins)')
    return ok


# ------------------------------------------------------- 2/3. the BZ measure

def test_minibz_measure():
    R = 0.234486

    ok = check(abs(head_average_2d(R, 0.0) - 4.0 * np.pi / R) < 1e-13,
               'the bare disc average is 4pi/R')

    # Numerical 2D quadrature over the disc, as an independent route.
    for kappa in (0.0, 0.5, 2.0, 10.0):
        q = np.linspace(1e-9, R, 200001)
        integrand = (2.0 * np.pi / (q + kappa)) * 2.0 * np.pi * q
        num = np.trapz(integrand, q) / (np.pi * R ** 2)
        ana = head_average_2d(R, kappa)
        ok &= check(abs(num - ana) / abs(ana) < 1e-6,
                    f'disc average == 2D quadrature (kappa = {kappa})',
                    f'{ana:.6f} vs {num:.6f}')

    ok &= check(abs(head_average_2d(R, 1e-12) - head_average_2d(R, 0.0))
                / head_average_2d(R, 0.0) < 1e-9,
                'the kappa -> 0 limit is continuous')
    big = 1e6
    ok &= check(abs(head_average_2d(R, big) - 2.0 * np.pi / big)
                / (2.0 * np.pi / big) < 1e-5,
                'the kappa -> inf limit is the saturated head 2pi/kappa')

    # The 3D sphere average is a DIFFERENT object -- not a convention change.
    # It scales as 1/R^2 where the 2D disc average scales as 1/R, so no
    # constant relates them: the ratio moves by 2x when R does.
    r1, r2 = 0.2, 0.4
    ratio1 = sphere_average_3d(r1) / head_average_2d(r1, 0.0)
    ratio2 = sphere_average_3d(r2) / head_average_2d(r2, 0.0)
    ok &= check(abs(ratio1 / ratio2 - 2.0) < 1e-12,
                'the 3D sphere average is not the 2D disc average '
                '(different power of R)', f'ratio changes {ratio1 / ratio2:.4f}x')

    # Disc radius against an explicit BZ-area construction, and its 1/N scaling.
    cell, _ = _bare_cell()
    area = inplane_cell_area(cell)
    ok &= check(abs(area - 4.0 / BOHR_TO_ANGSTROM * 4.0 / BOHR_TO_ANGSTROM) < 1e-6,
                'inplane_cell_area is |a1 x a2|', f'{area:.4f} bohr^2')
    for n in (2, 3, 4):
        want = np.sqrt((2 * np.pi) ** 2 / (area * n * n) / np.pi)
        ok &= check(abs(minibz_disc_radius(cell, [n, n, 1]) - want) < 1e-14,
                    f'mini-BZ disc radius at {n}x{n}', f'{want:.6f} bohr^-1')
    r2_, r4_ = minibz_disc_radius(cell, [2, 2, 1]), minibz_disc_radius(cell, [4, 4, 1])
    ok &= check(abs(r2_ / r4_ - 2.0) < 1e-12, 'the disc radius scales as 1/N')

    ok &= check(WING_EXPONENTS_2D_METAL['gz_odd'] == 0.5
                and WING_EXPONENTS_2D_METAL['gz_even'] == 1.0,
                'the wing exponents record the sqrt(q)/q split by G_z parity')
    return ok


def _bare_cell():
    cell = pgto.Cell()
    cell.atom = 'H 0 0 -0.37; H 0 0 0.37'
    cell.a = np.diag([4.0, 4.0, 24.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = [13, 13, 72]
    cell.verbose = 0
    cell.build()
    return cell, None


# ------------------------------------------------------------- 4. plasmon

def test_plasmon_dispersion():
    n = density_2deg(0.7)
    q = np.array([0.01, 0.04, 0.16, 0.64])
    w = plasmon_frequency_2d(q, n)
    ratios = w[1:] / w[:-1]
    ok = check(np.abs(ratios - 2.0).max() < 1e-12,
               'omega_p doubles when q quadruples (sqrt(q) dispersion)',
               f'{ratios}')
    ok &= check(plasmon_frequency_2d(0.0, n) == 0.0,
                'the 2D plasmon tends to ZERO at q -> 0, not to a constant')
    ok &= check(abs(plasmon_frequency_2d(0.25, n) - np.sqrt(2 * np.pi * n * 0.25)) < 1e-15,
                'omega_p = sqrt(2 pi n q / m)')
    return ok


# ------------------------------------------------ 4b. the exact mini-BZ cell

def _cell_with_lattice(a1, a2):
    cell = pgto.Cell()
    cell.atom = 'H 0 0 -0.37; H 0 0 0.37'
    cell.a = np.array([list(a1) + [0.0], list(a2) + [0.0], [0.0, 0.0, 24.0]])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = [13, 13, 72]
    cell.verbose = 0
    cell.build()
    return cell


def test_minibz_polygon():
    lattices = {'square': ((4.0, 0.0), (0.0, 4.0)),
                'hexagonal': ((2.46, 0.0), (-1.23, 2.13042)),
                'rect 2:1': ((4.0, 0.0), (0.0, 8.0))}
    ok = True
    for name, (a1, a2) in lattices.items():
        cell = _cell_with_lattice(a1, a2)
        poly = minibz_polygon(cell, [2, 2, 1])
        want = (2 * np.pi) ** 2 / (inplane_cell_area(cell) * 4)
        ok &= check(abs(polygon_area(poly) - want) / want < 1e-12,
                    f'{name}: the Wigner-Seitz cell has the mini-BZ area',
                    f'{len(poly)} vertices, A = {polygon_area(poly):.6f}')

    # The WS cell is a property of the LATTICE, not of the basis chosen for it:
    # (0,8), (4,8), (12,8) and (28,8) all generate the same lattice with (4,0).
    # The far-from-reduced ones need more neighbour shells, which is exactly
    # what would fail silently -- an under-clipped cell is too LARGE, so the
    # head would come out too small.
    heads = [head_average_polygon(
                 minibz_polygon(_cell_with_lattice((4.0, 0.0), (dx, 8.0)), [2, 2, 1]))
             for dx in (0.0, 4.0, 12.0, 28.0)]
    ok &= check(max(abs(h / heads[0] - 1.0) for h in heads) < 1e-12,
                'the mini-BZ average is basis-independent, however oblique',
                f'{heads[0]:.6f} across 4 bases of one lattice')
    try:
        minibz_polygon(_cell_with_lattice((4.0, 0.0), (12.0, 8.0)), [2, 2, 1], nmax=2)
    except RuntimeError:
        ok &= check(True, 'and too few neighbour shells is REFUSED, not '
                          'silently under-clipped')
    else:
        ok &= check(False, 'too few neighbour shells is refused')

    # A regular n-gon of area pi R^2 tends to the disc formula as n grows.
    R = 0.234486
    for n, tol in ((8, 2e-2), (64, 4e-4), (512, 1e-5)):
        ang = 2 * np.pi * np.arange(n) / n
        rad = R * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))   # equal area
        poly = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        rel = abs(head_average_polygon(poly) / head_average_2d(R, 0.0) - 1.0)
        ok &= check(rel < tol, f'a {n}-gon approaches the disc average',
                    f'{rel:.2e}')

    # Independent 2D quadrature on the hexagonal (graphite) mini-BZ.
    hexcell = _cell_with_lattice((2.46, 0.0), (-1.23, 2.13042))
    poly = minibz_polygon(hexcell, [2, 2, 1])
    ana = head_average_polygon(poly, 2.0)
    num = _quadrature_over_polygon(poly, 2.0)
    ok &= check(abs(num - ana) / ana < 1e-5,
                'hexagonal mini-BZ average == 2D quadrature (kappa=2)',
                f'{ana:.6f} vs {num:.6f}')
    # At kappa = 0 the integrand has the integrable 1/q singularity at the
    # origin, which a midpoint grid resolves slowly -- so check that the
    # QUADRATURE converges onto the analytic value rather than demanding it
    # already agree. The analytic route has no such error: it integrates the
    # radial part in closed form, which is the point of doing it that way.
    ana0 = head_average_polygon(poly, 0.0)
    errs0 = [abs(_quadrature_over_polygon(poly, 0.0, n) - ana0) / ana0
             for n in (400, 800, 1600)]
    ok &= check(errs0[0] > errs0[1] > errs0[2] and errs0[-1] < 3e-4,
                'and converges onto it at kappa = 0, where the grid must '
                'resolve a 1/q singularity',
                ' -> '.join(f'{e:.1e}' for e in errs0))

    # How much the equal-area disc costs: little when the cell is round, not
    # little when it is not.
    print(f"       {'lattice':12s} {'exact':>10s} {'disc':>10s} {'disc error':>11s}")
    errs = {}
    for name, (a1, a2) in lattices.items():
        cell = _cell_with_lattice(a1, a2)
        ex = slab_head_value(cell, [2, 2, 1])
        di = slab_head_value(cell, [2, 2, 1], shape='disc')
        errs[name] = abs(di / ex - 1.0)
        print(f"       {name:12s} {ex:10.4f} {di:10.4f} {100 * errs[name]:10.2f}%")
    ok &= check(errs['hexagonal'] < 2e-3,
                'the disc is a good approximation for a hexagonal cell')
    ok &= check(errs['rect 2:1'] > 20 * errs['hexagonal'],
                'and a much worse one for an anisotropic cell',
                f"{100 * errs['rect 2:1']:.2f}% vs {100 * errs['hexagonal']:.2f}%")

    # Anisotropy: a direction-dependent kappa is NOT its angular mean. Modulating
    # kappa with zero mean still moves the head, so an isotropic form cannot be
    # rescued by averaging kappa -- it is the wrong coefficient, not a shift.
    sq_cell = _cell_with_lattice((4.0, 0.0), (0.0, 4.0))
    poly = minibz_polygon(sq_cell, [2, 2, 1])
    iso = head_average_polygon(poly, 2.0)
    aniso = head_average_polygon(poly, lambda phi: 2.0 + 1.0 * np.cos(2 * phi))
    ok &= check(abs(aniso / iso - 1.0) > 0.05,
                'an angle-dependent kappa is not its angular mean',
                f'{iso:.4f} -> {aniso:.4f}, {100 * (aniso / iso - 1):+.1f}%')
    ok &= check(slab_head_value(sq_cell, [2, 2, 1],
                                lambda phi: 2.0 + 0.0 * phi) == iso,
                'a constant callable reproduces the scalar kappa')
    try:
        slab_head_value(sq_cell, [2, 2, 1], lambda phi: phi, shape='disc')
    except ValueError:
        ok &= check(True, "shape='disc' refuses an angle-dependent kappa")
    else:
        ok &= check(False, "shape='disc' refuses an angle-dependent kappa")

    try:
        head_average_polygon(poly[::-1], 0.0)
    except ValueError:
        ok &= check(True, 'clockwise vertices are refused')
    else:
        ok &= check(False, 'clockwise vertices are refused')
    try:
        head_average_polygon(poly + np.array([10.0, 0.0]), 0.0)
    except ValueError:
        ok &= check(True, 'a polygon not containing the origin is refused')
    else:
        ok &= check(False, 'a polygon not containing the origin is refused')
    return ok


def _quadrature_over_polygon(poly, kappa, n=1200):
    """<2pi/(q+kappa)> by brute-force Cartesian quadrature, as an independent
    route: sample a bounding box, keep the points inside the convex polygon."""
    v = np.asarray(poly)
    lim = np.abs(v).max() * 1.001
    grid = (np.arange(n) + 0.5) / n * 2 * lim - lim
    X, Y = np.meshgrid(grid, grid, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    inside = np.ones(len(pts), dtype=bool)
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        edge = b - a
        inside &= (edge[0] * (pts[:, 1] - a[1]) - edge[1] * (pts[:, 0] - a[0])) >= 0
    q = np.linalg.norm(pts[inside], axis=1)
    cellarea = (2 * lim / n) ** 2
    return float(np.sum(2 * np.pi / (q + kappa)) * cellarea / (inside.sum() * cellarea))


# ------------------------------------------ 5. the constant-approximation kernel

def test_constant_head_kernel():
    cell, _ = _bare_cell()
    Gv, _, _ = cell.get_Gv_weights(cell.mesh)
    ok = True

    ratios = []
    for n in (2, 3, 4):
        kmesh = [n, n, 1]
        r0, beta, _ = nyquist_params(cell, kmesh)
        base = make_coulG_damped(r0, beta)
        q_ref = smallest_transfer(cell, kmesh)
        ca = make_coulG_constant_head(base, q_ref)

        v_base = base(cell, np.zeros(3), Gv)
        v_ca = ca(cell, np.zeros(3), Gv)
        head = np.linalg.norm(Gv, axis=1) < 1e-8
        ok &= check(head.sum() == 1, f'{n}x{n}: exactly one head point on the grid')
        ok &= check(np.array_equal(v_base[~head], v_ca[~head]),
                    f'{n}x{n}: every non-head point is bit-identical')
        ok &= check((v_ca >= 0).all(), f'{n}x{n}: the kernel stays non-negative')
        ok &= check(v_ca[head][0] < v_base[head][0],
                    f'{n}x{n}: the head is reduced',
                    f'{v_base[head][0]:.3f} -> {v_ca[head][0]:.3f}')
        ratios.append(v_ca[head][0] / v_base[head][0])
        ok &= check(getattr(ca, 'damping', None) == (r0, beta),
                    f'{n}x{n}: the damping tag survives, so the support check reaches it')
        ok &= check(assert_damping_fits(cell, cell.make_kpts(kmesh), ca) == [],
                    f'{n}x{n}: assert_damping_fits runs on the CA kernel')

    # q_min * r0 = pi * frac_r0 is mesh-independent under the AUTO scheme, so
    # the damped kernel is sampled at the SAME dimensionless point on every
    # mesh and the constant approximation is a fixed rescaling of the Gamma
    # head -- it does not vanish as the grid grows.
    ratios = np.array(ratios)
    ok &= check(ratios.ptp() < 1e-6,
                'the CA head ratio is the same at every k-mesh '
                '(q_min * r0 is mesh-independent)',
                f'{ratios[0]:.6f}, spread {ratios.ptp():.1e}')

    for n in (2, 4):
        kmesh = [n, n, 1]
        r0, _, _ = nyquist_params(cell, kmesh)
        ok &= check(abs(smallest_transfer(cell, kmesh) * r0 - np.pi * 0.5) < 1e-10,
                    f'{n}x{n}: q_min * r0 == pi * frac_r0')

    try:
        make_coulG_constant_head(make_coulG_damped(1.0, 1.0), 0.0)
    except ValueError:
        ok &= check(True, 'a non-positive reference momentum is refused')
    else:
        ok &= check(False, 'a non-positive reference momentum is refused')
    try:
        smallest_transfer(cell, [1, 1, 1])
    except ValueError:
        ok &= check(True, 'a Gamma-only mesh has no reference transfer')
    else:
        ok &= check(False, 'a Gamma-only mesh has no reference transfer')
    return ok


# ------------------------------------------------- 6. admissibility on a slab

class _GammaHead:
    """J(Gamma) and dJ(Gamma) split at head_value = 0 and 1.

    dJ is AFFINE in head_value (the head enters rank2_matrix linearly and
    nothing else depends on it), so two builds give the whole sweep and the
    ceiling can be bisected without rebuilding integrals.
    """

    def __init__(self, cell, mf, env, coulG):
        aux = make_auxcell(cell, 'weigend')
        naux = aux.nao_nr()
        kpts = np.asarray(mf.kpts)
        kscaled = cell.get_scaled_kpts(kpts)
        kscaled -= kscaled[0]
        b = cell.reciprocal_vectors()
        Gv, _, kws = cell.get_Gv_weights(cell.mesh)
        kws = np.asarray(kws)
        gpar, gz, wz, _ = _planar_grid(cell, cell.mesh)
        area = cell.vol / np.linalg.norm(cell.a[2])

        J = np.zeros((naux, naux), dtype=np.complex128)
        d0 = np.zeros((naux, naux), dtype=np.complex128)
        d1 = np.zeros((naux, naux), dtype=np.complex128)
        for ki in range(len(kpts)):
            kj, G0 = _bz_index(kscaled, kscaled[ki])
            qtrue = (kscaled[kj] + G0 - kscaled[ki]).dot(b)
            auxG = ft_ao.ft_ao(aux, Gv, kpt=qtrue)
            vG = coulG(cell, qtrue, Gv) * kws
            J += np.einsum('gP,gQ->PQ', auxG.conj() * vG[:, None], auxG)
            d0 += reaction_field_form(env, auxG, auxG, qtrue, gpar, gz, wz, area, 0.0)
            d1 += reaction_field_form(env, auxG, auxG, qtrue, gpar, gz, wz, area, 1.0)
        nk = len(kpts)
        self.J, self.const, self.slope = J / nk, d0 / nk, (d1 - d0) / nk
        self.nGz = len(gz)

    def rel_min_eig(self, head):
        M = self.J + self.const + head * self.slope
        e = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
        return e.min() / e.max()

    def ceiling(self):
        lo, hi = 0.0, 1.0
        while self.rel_min_eig(hi) > 0 and hi < 1e6:
            lo, hi = hi, 2 * hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.rel_min_eig(mid) > 0:
                lo = mid
            else:
                hi = mid
        return lo


def test_head_admissibility():
    kmesh = [2, 2, 1]
    cell, mf = make_slab(kmesh, [13, 13, 72])
    r0, beta, _ = nyquist_params(cell, kmesh)
    coulG = make_coulG_damped(r0, beta)
    env = SlabDielectricEnvironment(cell, solvent='water', eps_bot=np.inf,
                                    z_center=0.0, z_half_width=4.0)
    gh = _GammaHead(cell, mf, env, coulG)
    ceiling = gh.ceiling()
    ok = check(ceiling > 0, 'the head channel has a positive admissible ceiling',
               f'{ceiling:.4f}')

    bare = slab_head_value(cell, kmesh)
    scr = slab_head_value(cell, kmesh, KAPPA_2DEG_FREE_ELECTRON)
    ca_head = 2.0 * np.pi / smallest_transfer(cell, kmesh)
    print(f"       ceiling {ceiling:8.4f} | bare {bare:8.4f} | "
          f"constant-approx {ca_head:8.4f} | metal-screened {scr:8.4f}")

    ok &= check(bare > ceiling, 'the BARE mini-BZ head exceeds the ceiling',
                f'{bare / ceiling:.2f}x')
    ok &= check(ca_head > ceiling,
                'the constant-approximation head also exceeds it',
                f'{ca_head / ceiling:.2f}x')
    ok &= check(scr < ceiling,
                'the METAL-SCREENED head fits -- the head becomes admissible '
                'exactly when screening is included', f'{scr / ceiling:.2f}x')
    ok &= check(gh.rel_min_eig(scr) > 0 and gh.rel_min_eig(bare) < 0,
                'and the metric agrees: screened positive definite, bare not',
                f'{gh.rel_min_eig(scr):.2e} vs {gh.rel_min_eig(bare):.2e}')

    # The ceiling is geometric, not a z-mesh artifact: refine mesh[2] and both
    # the ceiling and the head channel's whole contribution stay put. The
    # screening module's docstring blamed "roughly the number of G_z points".
    cell2, mf2 = make_slab(kmesh, [13, 13, 144])
    env2 = SlabDielectricEnvironment(cell2, solvent='water', eps_bot=np.inf,
                                     z_center=0.0, z_half_width=4.0)
    gh2 = _GammaHead(cell2, mf2, env2, coulG)
    c2 = gh2.ceiling()
    s1, s2 = np.linalg.norm(gh.slope), np.linalg.norm(gh2.slope)
    ok &= check(gh2.nGz == 2 * gh.nGz, 'the z-mesh really doubled',
                f'{gh.nGz} -> {gh2.nGz}')
    ok &= check(abs(c2 - ceiling) / ceiling < 2e-3,
                'the ceiling is unchanged by doubling the z-mesh',
                f'{ceiling:.4f} -> {c2:.4f}')
    ok &= check(abs(s2 - s1) / s1 < 1e-4,
                'so is the head channel itself: |d(dJ)/d(head)| is converged',
                f'{s1:.5f} -> {s2:.5f}')

    # End to end: an admissible head is not a no-op -- it moves Sigma^solv.
    bare_ints = build_dfintegrals_coulG(mf, coulG_fn=coulG)
    dropped = build_dfintegrals_screened(mf, env, coulG_fn=coulG, head_value=0.0)
    screened = build_dfintegrals_screened(mf, env, coulG_fn=coulG, head_value=scr)
    s_drop = solvent_cohsex_kpts(bare_ints, dropped)
    s_head = solvent_cohsex_kpts(bare_ints, screened)
    nocc = bare_ints.nocc[0]
    occ_drop = np.array([np.diag(s_drop[k]).real[:nocc] for k in range(bare_ints.nkpts)]).mean()
    occ_head = np.array([np.diag(s_head[k]).real[:nocc] for k in range(bare_ints.nkpts)]).mean()
    ok &= check(occ_head > occ_drop > 0,
                'a screened head screens MORE than dropping the channel',
                f'occupied shift {occ_drop * HARTREE_TO_EV:+.4f} -> '
                f'{occ_head * HARTREE_TO_EV:+.4f} eV')

    try:
        build_dfintegrals_screened(mf, env, coulG_fn=coulG, head_value=bare)
    except ValueError as err:
        ok &= check('not positive definite' in str(err),
                    'the builder refuses the bare head rather than returning '
                    'a silently over-screened L')
    else:
        ok &= check(False, 'the builder refuses the bare head')
    return ok


if __name__ == '__main__':
    all_ok = True
    print('\n-- 1. Stern 2DEG reference limits')
    all_ok &= test_stern_reference_limits()
    print('\n-- 2/3. the mini-BZ measure, and the 3D average it is not')
    all_ok &= test_minibz_measure()
    print('\n-- 4. the sqrt(q) plasmon')
    all_ok &= test_plasmon_dispersion()
    print('\n-- 4b. the exact mini-BZ cell and the anisotropy hook')
    all_ok &= test_minibz_polygon()
    print('\n-- 5. the constant-approximation kernel')
    all_ok &= test_constant_head_kernel()
    print('\n-- 6. head admissibility on a slab')
    all_ok &= test_head_admissibility()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
