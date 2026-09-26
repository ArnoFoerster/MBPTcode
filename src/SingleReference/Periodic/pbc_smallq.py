"""Small-q limits of the screened interaction for a SLAB.

The long-wavelength limit is where a two-dimensional system stops resembling a
three-dimensional one, and it is the accuracy-limiting step for a metallic
slab. This module owns the analytic forms and the Brillouin-zone measure that
the q->0 channel needs; it computes no integrals of its own, so every function
here is cheap and separately testable against a published limit.

Why 2D is qualitatively different
---------------------------------
For a two-dimensional electron gas the bare interaction is v(q) = 2pi/q, not
4pi/q^2. Stern's static polarizability (Phys. Rev. Lett. 18, 546 (1967)) is a
CONSTANT -N(0) = -m/(pi hbar^2) below q = 2 k_F, so

    eps(q)  = 1 - v(q) Pi(q) = 1 + kappa/q ,     kappa = 2 pi N(0) ,
    W(q)    = v(q)/eps(q)    = 2pi/(q + kappa) ,

i.e. the DIELECTRIC HEAD diverges as 1/q (not 1/q^2) and the SCREENED
interaction head is FINITE at q = 0, saturating at 2pi/kappa. In a.u. the free
electron gas gives kappa = 2 exactly, independent of density, because the 2D
density of states is. The dynamic counterpart is the sqrt(q) plasmon
omega_p = sqrt(2 pi n q / m) rather than the 3D constant, and it is that
non-analyticity at the origin -- not the metallicity as such -- that no
three-dimensional expression can be Taylor-expanded into.

Consequence for this code: a metallic slab's small-q form is NOT the textbook
3D Drude expression, and porting the 3D one gives numbers that look reasonable
and are wrong.

The Brillouin-zone measure, and the one that must NOT be used
------------------------------------------------------------
The momentum-transfer sum sum_q w_q F(q) with w_q = 1/N_k stands for the BZ
average (1/A_BZ) int F(q) d^2q, so the q = Gamma term stands for the average of
F over the MINI-BZ CELL around the origin, not for F(0). For an in-plane cell
of area A the mini-BZ has area A_mBZ = (2 pi)^2/(A N_par); replacing it by the
disc of equal area, radius R = sqrt(A_mBZ/pi), gives in closed form

    <2pi/(q+kappa)>_disc = (4 pi / R^2) [ R - kappa ln(1 + R/kappa) ] ,
    <2pi/q>_disc         = 4 pi / R                     (the kappa -> 0 limit).

Both integrals converge because d^2q = q dq d(phi) kills one power of 1/q --
which is exactly the property a 3D construction does not have. The Gaussian
basis periodic-GW literature obtains its finite-size head and wing corrections
by integrating the long-wavelength contribution over a SPHERE of radius q0
around the origin; that is intrinsically three-dimensional, because it rests on
a 3D volume element. For a slab the domain is this 2D disc and the divergence
structure differs, so the spherical expression must not be ported: it would
look like a principled correction and be wrong. `sphere_average_3d` is provided
only so a test can pin that the two differ.

What is measured, and what it means for `head_value`
----------------------------------------------------
pbc_solvent_screening drops the single in-plane channel with q_par + G_par = 0
(`head_value=0.0`), because the reaction field vtilde is negative and a head
that is too large drives the whitened metric J + dJ indefinite. That ceiling is
real and it is GEOMETRIC, not a discretization artifact: measured on the H2
slab (gth-szv, dimension=2, water above and a metal electrode below, cavity
half-width 4 bohr), the largest admissible head is 6.07 and it is unchanged --
along with the head channel's whole contribution, |d(dJ)/d(head)| = 14.7716 --
as the z-mesh goes 72 -> 108 -> 144 -> 216 points. (The module docstring there
attributed the over-screening to "roughly the number of G_z points"; the G-space
quadrature weight already carries the 1/nGz, so the channel is converged in the
z-mesh and that is not the mechanism.) The ceiling instead tracks the reaction
field's strength: 13.94 for water alone, 6.07 once a conductor sits below, and
it falls with the cavity width (7.92 / 6.07 / 5.63 / 5.56 at half-widths
3/4/5/6 bohr).

Set against that ceiling, on a 2x2x1 mesh:

    head treatment                     value    value/ceiling
    dropped                            0.00     0.00   (current default)
    2pi/q_min   (constant approx.)    15.12     2.49   INDEFINITE
    <2pi/q>     mini-BZ, bare         53.30     8.78   INDEFINITE
    <2pi/(q+kappa)>, kappa = 2         2.91     0.48   admissible

(mini-BZ averages over the true square cell; the equal-area disc gives 53.59
and 2.92, a 0.55% and 0.12% difference -- see slab_head_value.)

**So the head becomes admissible exactly when metallic screening is included.**
The bare and constant-approximation heads are inadmissible not because of a
bug but because they describe an UNSCREENED long-wavelength interaction, which
a slab reaction field cannot support. That tracks the physics rather than being
numerology: a two-dimensional INSULATOR has eps -> 1 as q -> 0 (its
polarizability enters as 1 + 2 pi alpha q), so its head genuinely diverges like
the bare 2pi/q and there is nothing finite to put there; only a metal's
eps = 1 + kappa/q saturates W at 2pi/kappa. Note the test slab is itself an H2
layer, i.e. gapped -- the metallic kappa there probes ADMISSIBILITY, not that
slab's own screening. Refining the k-mesh does not rescue them
-- the bare head grows as 1/R ~ N_k while the ceiling grows more slowly (6.07 /
11.92 / 18.58 at 2x2/3x3/4x4, so the ratio only falls 8.8 -> 6.7 -> 5.8) -- and
the part of that growth which does help comes from the AUTO-damped v(0) ~ r0^2
rising with the grid, which is itself a Gamma over-screening. The screened head
instead saturates at 2pi/kappa and so becomes MORE comfortably admissible as
the mesh is refined (0.48 / 0.25 / 0.16 of the ceiling).

Scope
-----
These are the q -> 0 forms and the measure to apply them with. They do not by
themselves make a metal computable: with integer occupations there is no
intraband response at all, so kappa cannot be measured from an
integer-occupation polarizability and must be supplied (or taken from the
free-electron value); the intraband response is the occupation-weighted one of
`pbc_occupations`. What this module does supply is the head law that response
has to be checked against, the mini-BZ measure any neighbourhood-valued scheme needs, and
the constant-approximation kernel that gives the Gamma-convergence study a
defensible baseline.
"""
import numpy as np
from pyscf.pbc import tools

#: Thomas-Fermi screening wavevector of a free 2D electron gas, in a.u.
#: kappa = 2 pi N(0) with N(0) = m/(pi hbar^2); density-independent because the
#: 2D density of states is. Stern, PRL 18, 546 (1967).
KAPPA_2DEG_FREE_ELECTRON = 2.0

#: Small-q scaling exponents of the inverse-dielectric wings for a 2D metal,
#: tabulated by Sesti et al. and reproduced here only as the classification an
#: interpolant must respect: in-plane G and ODD out-of-plane G_z carry sqrt(q),
#: even G_z carry q. The sqrt(q) branch is non-analytic at the origin, which is
#: the formal reason a 3D expression cannot be expanded into this regime.
WING_EXPONENTS_2D_METAL = {'in_plane': 0.5, 'gz_odd': 0.5, 'gz_even': 1.0}


def inplane_cell_area(cell):
    """Area of the in-plane unit cell, |a1 x a2|, in bohr^2."""
    a = cell.lattice_vectors()
    return float(np.linalg.norm(np.cross(a[0], a[1])))


def minibz_disc_radius(cell, kmesh):
    """Radius R of the disc with the same area as one mini-BZ cell, in bohr^-1.

    The in-plane BZ has area (2 pi)^2/A; an N1 x N2 in-plane mesh divides it
    into N1*N2 mini-BZ cells. Replacing that cell by an equal-area disc is what
    makes the head average isotropic and analytic; see the module docstring for
    why the anisotropic case matters and this does not cover it.
    """
    n_par = int(kmesh[0]) * int(kmesh[1])
    if n_par < 1:
        raise ValueError(f"in-plane k-mesh must be at least 1x1, got {kmesh}")
    area_mbz = (2.0 * np.pi) ** 2 / (inplane_cell_area(cell) * n_par)
    return float(np.sqrt(area_mbz / np.pi))


def head_average_2d(radius, kappa=0.0):
    """<2pi/(q + kappa)> over a disc of radius `radius`, in the 2D measure.

    The mini-BZ average of the screened 2D head -- the value the Gamma term of
    the momentum-transfer sum actually stands for. kappa = 0 gives the bare
    head's average, 4 pi / radius. Exact, from

        (1/(pi R^2)) int_0^R [2pi/(q+k)] 2 pi q dq
            = (4 pi / R^2) [ R - k ln(1 + R/k) ] .
    """
    R = float(radius)
    if R <= 0:
        raise ValueError(f"disc radius must be positive, got {R}")
    k = float(kappa)
    if k < 0:
        raise ValueError(f"kappa must be non-negative, got {k}")
    if k == 0.0:
        return 4.0 * np.pi / R
    return (4.0 * np.pi / R ** 2) * (R - k * np.log1p(R / k))


def minibz_polygon(cell, kmesh, nmax=None):
    """Vertices of the in-plane mini-BZ, counter-clockwise, in bohr^-1.

    The Wigner-Seitz cell of the momentum-transfer lattice (primitive vectors
    b1/N1, b2/N2), built by intersecting the perpendicular-bisector half-planes
    of its near neighbours. This is the EXACT domain the Gamma term of the
    q-sum stands for; `minibz_disc_radius` replaces it by an equal-area disc,
    which is isotropic and therefore cannot carry a direction dependence.

    Only the in-plane components are used, so the cell's third lattice vector
    must be perpendicular to the other two -- the same requirement the slab
    kernel already imposes.

    How many neighbour shells are needed depends on how reduced the basis is:
    the same lattice written as (4,0),(0,8) closes at one shell and as
    (4,0),(12,8) does not. `nmax=None` escalates until the cell tiles, which is
    checked exactly; pass an integer to pin it.
    """
    b = cell.reciprocal_vectors()
    a = cell.lattice_vectors()
    if (abs(np.dot(a[2], a[0])) > 1e-8 * np.linalg.norm(a[2]) * np.linalg.norm(a[0])
            or abs(np.dot(a[2], a[1])) > 1e-8 * np.linalg.norm(a[2]) * np.linalg.norm(a[1])):
        raise ValueError("the mini-BZ polygon needs a3 perpendicular to the "
                         "in-plane lattice vectors")
    c1 = b[0][:2] / int(kmesh[0])
    c2 = b[1][:2] / int(kmesh[1])
    want = abs(float(np.cross(c1, c2)))

    shells = [nmax] if nmax is not None else [2, 3, 5, 8, 12]
    got = None
    for shell in shells:
        out = _wigner_seitz_2d(c1, c2, shell)
        got = polygon_area(out)
        # A Wigner-Seitz cell tiles its lattice, so its area is EXACTLY one
        # mini-BZ. Under-clipping leaves a too-large domain and hence a
        # too-small head, silently, so check rather than trust the range.
        if abs(got - want) <= 1e-9 * want:
            return out
    raise RuntimeError(
        f"mini-BZ construction is not a Wigner-Seitz cell after {shells[-1]} "
        f"neighbour shells: area {got:.8g} vs the lattice's {want:.8g}. The "
        f"in-plane basis is far from reduced; reduce it, or pass a larger "
        f"nmax.")


def _wigner_seitz_2d(c1, c2, nmax):
    """Clip a bounding box by the perpendicular bisectors of the neighbours
    within +/- nmax of the origin in each lattice direction."""
    scale = 4.0 * max(np.linalg.norm(c1), np.linalg.norm(c2)) * nmax
    poly = [np.array([-scale, -scale]), np.array([scale, -scale]),
            np.array([scale, scale]), np.array([-scale, scale])]
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            if i == 0 and j == 0:
                continue
            R = i * c1 + j * c2
            poly = _clip_halfplane(poly, R, 0.5 * float(R @ R))
            if len(poly) < 3:
                raise RuntimeError("mini-BZ construction collapsed")
    return _sort_ccw(np.array(poly))


def _clip_halfplane(poly, normal, offset):
    """Sutherland-Hodgman clip of a convex polygon to {x : x . normal <= offset}."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp = float(p @ normal) - offset
        dq = float(q @ normal) - offset
        if dp <= 0:
            out.append(p)
        if (dp < 0 < dq) or (dq < 0 < dp):
            out.append(p + (q - p) * (dp / (dp - dq)))
    return out


def _sort_ccw(vertices):
    """Order polygon vertices counter-clockwise about their centroid, dropping
    duplicates left behind by successive clips."""
    keep = [vertices[0]]
    for v in vertices[1:]:
        if min(np.linalg.norm(v - k) for k in keep) > 1e-10:
            keep.append(v)
    keep = np.array(keep)
    ang = np.arctan2(keep[:, 1] - keep[:, 1].mean(), keep[:, 0] - keep[:, 0].mean())
    return keep[np.argsort(ang)]


def polygon_area(vertices):
    """Shoelace area of a simple polygon."""
    v = np.asarray(vertices, dtype=float)
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def head_average_polygon(vertices, kappa=0.0, nquad=64):
    """<2pi/(q + kappa)> over a convex polygon containing the origin.

    The exact counterpart of `head_average_2d`, for the real mini-BZ rather
    than an equal-area disc, and the hook for ANISOTROPY: `kappa` may be a
    scalar or a callable kappa(phi) taking the in-plane angle in radians. A
    lithiated-graphite slab with a lithium superstructure has reduced in-plane
    symmetry, so its small-q response is genuinely direction-dependent, and an
    isotropic form there is the right scaling law with the wrong coefficient.

    In polar coordinates about the origin the 2D measure kills one power of
    1/q, leaving a single angular integral over the polygon's support:

        int f d^2q = 2 pi int dphi [ rho(phi) - k ln(1 + rho(phi)/k) ] ,

    with rho(phi) the boundary distance (rho itself for k -> 0). Each edge
    contributes a smooth angular range, so Gauss-Legendre per edge is exact to
    machine precision rather than merely convergent -- a uniform grid would
    have a kink at every vertex.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        raise ValueError("need at least three 2D vertices")
    kappa_fn = kappa if callable(kappa) else (lambda _phi, k=float(kappa): k)
    nodes, weights = np.polynomial.legendre.leggauss(int(nquad))

    total = 0.0
    for i in range(len(v)):
        p, q = v[i], v[(i + 1) % len(v)]
        edge = q - p
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-14:
            continue
        normal = normal / norm
        d = float(p @ normal)
        if d <= 0:                       # normal points inward: flip
            normal, d = -normal, -d
        if d <= 1e-14:
            raise ValueError("the polygon does not contain the origin strictly")
        phi_n = np.arctan2(normal[1], normal[0])
        phi_p = np.arctan2(p[1], p[0])
        phi_q = np.arctan2(q[1], q[0])
        sweep = (phi_q - phi_p) % (2 * np.pi)
        if sweep > np.pi:                # vertices given clockwise
            raise ValueError("polygon vertices must be counter-clockwise")
        lo, hi = phi_p, phi_p + sweep
        phi = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        rho = d / np.cos(phi - phi_n)
        k = np.asarray([kappa_fn(float(x)) for x in phi], dtype=float)
        inner = np.where(k > 0, rho - k * np.log1p(rho / np.where(k > 0, k, 1.0)), rho)
        total += 0.5 * (hi - lo) * float(np.dot(weights, inner))
    return 2.0 * np.pi * total / polygon_area(v)


def sphere_average_3d(radius):
    """<4pi/q^2> over a BALL of radius `radius` -- the THREE-dimensional head
    average, provided so a test can pin that it is a different object.

    (1/(4/3 pi R^3)) int_0^R (4pi/q^2) 4 pi q^2 dq = 12 pi / R^2.

    Do not use this for a slab. It integrates over a 3D volume element, which
    is what makes the published Gaussian-basis finite-size correction
    inapplicable here; the slab's domain is the 2D disc of `head_average_2d`.
    """
    R = float(radius)
    if R <= 0:
        raise ValueError(f"ball radius must be positive, got {R}")
    return 12.0 * np.pi / R ** 2


def polygon_radial_moments(vertices, g, nang=24, nrad=24):
    """(I0, I1) = (int_P g(|k|) d^2k, int_P k g(|k|) d^2k) over ANY simple polygon.

    `head_average_polygon` assumes the origin is inside the polygon, which is
    true only of the mini-BZ cell at Gamma. Every other cell of the transfer
    grid sits away from the origin, and W-av needs the same integrals there, so
    this generalizes by SIGNED TRIANGLES: the polygon equals the signed sum of
    the triangles (0, a_i, b_i) over its edges, whichever side of it the origin
    lies on, exactly as the shoelace formula does for the area.

    Within one triangle the boundary distance is rho(phi) = d/cos(phi - phi_n)
    with d the origin's distance to the edge line, so the integrals reduce to

        I0 = sum_edges int dphi int_0^rho g(k) k dk ,
        I1 = sum_edges int dphi (cos phi, sin phi) int_0^rho g(k) k^2 dk ,

    the sweep's sign carrying the orientation. The radial integrand carries the
    2D measure factor k, which is what makes an integrable 1/k singularity
    harmless, so Gauss-Legendre in both directions is accurate with no special
    case at the origin. `g` is called with a flat array of |k|.

    Returns I0 (float) and I1 (2-vector), both UNNORMALIZED -- divide by the
    polygon area for an average, and note I1/area is the g-weighted centroid.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        raise ValueError("need at least three 2D vertices")
    an, aw = np.polynomial.legendre.leggauss(int(nang))
    rn, rw = np.polynomial.legendre.leggauss(int(nrad))

    I0 = 0.0
    I1 = np.zeros(2)
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        if abs(float(a[0] * b[1] - a[1] * b[0])) < 1e-14:
            continue                       # origin on the line: degenerate
        edge = b - a
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        d = float(a @ normal)
        if d < 0:
            normal, d = -normal, -d
        if d < 1e-14:
            continue
        phi_n = np.arctan2(normal[1], normal[0])
        phi_a = np.arctan2(a[1], a[0])
        phi_b = np.arctan2(b[1], b[0])
        sweep = (phi_b - phi_a + np.pi) % (2 * np.pi) - np.pi     # into (-pi, pi]
        lo, hi = phi_a, phi_a + sweep
        phi = 0.5 * (hi - lo) * an + 0.5 * (hi + lo)
        wphi = 0.5 * (hi - lo) * aw
        rho = d / np.cos(phi - phi_n)

        k = 0.5 * rho[:, None] * (rn[None, :] + 1.0)
        wk = 0.5 * rho[:, None] * rw[None, :]
        gk = np.asarray(g(k.ravel()), dtype=float).reshape(k.shape)
        m1 = np.sum(wk * gk * k, axis=1)              # int_0^rho g k dk
        m2 = np.sum(wk * gk * k ** 2, axis=1)         # int_0^rho g k^2 dk
        I0 += float(np.dot(wphi, m1))
        I1 += np.array([float(np.dot(wphi * np.cos(phi), m2)),
                        float(np.dot(wphi * np.sin(phi), m2))])
    return I0, I1


def thomas_fermi_kappa_2d(dos_ef):
    """kappa = 2 pi N(0) from the 2D density of states at the Fermi level.

    N(0) is per unit area and per unit energy, both spins included. The free
    electron gas has N(0) = m/(pi hbar^2) = 1/pi in a.u., giving
    KAPPA_2DEG_FREE_ELECTRON.
    """
    if dos_ef < 0:
        raise ValueError(f"density of states must be non-negative, got {dos_ef}")
    return 2.0 * np.pi * float(dos_ef)


def polarizability_2deg(q, kf):
    """Stern's static polarizability Pi(q, omega=0) of a 2D electron gas, a.u.

    Phys. Rev. Lett. 18, 546 (1967):

        Pi(q) = -N(0) [ 1 - theta(q - 2k_F) sqrt(1 - (2 k_F/q)^2) ] ,

    with N(0) = 1/pi in a.u. Constant below 2 k_F -- which is what makes the
    dielectric head exactly 1 + kappa/q there -- and falling as 1/q^2 above it.
    Negative by convention (a density response). This is the reference limit an
    occupation-weighted response (`pbc_occupations`) has to reproduce for a
    free 2D metal.
    """
    q = np.asarray(q, dtype=float)
    kf = float(kf)
    if kf <= 0:
        raise ValueError(f"Fermi wavevector must be positive, got {kf}")
    dos = 1.0 / np.pi
    above = q > 2.0 * kf
    safe_q = np.where(above, q, 1.0)
    reduction = np.where(above, np.sqrt(np.maximum(1.0 - (2.0 * kf / safe_q) ** 2, 0.0)), 0.0)
    out = -dos * (1.0 - reduction)
    return out if out.ndim else float(out)


def density_2deg(kf):
    """Areal density n = k_F^2/(2 pi) of a 2D electron gas (both spins), a.u."""
    return float(kf) ** 2 / (2.0 * np.pi)


def dielectric_head_2d(q, kappa=KAPPA_2DEG_FREE_ELECTRON):
    """eps(q) = 1 + kappa/q, the 2D-metal static dielectric head.

    Diverges as 1/q, NOT as the 3D 1/q^2. Undefined at q = 0; use
    `head_average_2d` for the Gamma channel.
    """
    q = np.asarray(q, dtype=float)
    if np.any(q <= 0):
        raise ValueError("eps(q) has a pole at q = 0; use head_average_2d for "
                         "the Gamma channel of the momentum-transfer sum.")
    return 1.0 + float(kappa) / q


def w_head_2d(q, kappa=KAPPA_2DEG_FREE_ELECTRON):
    """W(q) = 2pi/(q + kappa), the 2D-metal screened interaction head.

    FINITE at q = 0, saturating at 2pi/kappa -- the whole reason a metallic
    slab's head is treatable at all. Equals v(q)/eps(q) with v = 2pi/q and
    `dielectric_head_2d`.
    """
    q = np.asarray(q, dtype=float)
    k = float(kappa)
    if k <= 0:
        raise ValueError(f"kappa must be positive for the screened head, got {k}")
    if np.any(q < 0):
        raise ValueError("q must be non-negative")
    return 2.0 * np.pi / (q + k)


def plasmon_frequency_2d(q, n2d, mass=1.0):
    """omega_p(q) = sqrt(2 pi n q / m), the sqrt(q) 2D plasmon dispersion, a.u.

    The dynamic counterpart of the 1/q dielectric head, and the dispersion a
    plasmon-pole model cannot represent for a doped layered material: it tends
    to ZERO at q -> 0 rather than to the 3D constant sqrt(4 pi n/m).
    """
    q = np.asarray(q, dtype=float)
    if np.any(q < 0):
        raise ValueError("q must be non-negative")
    if n2d < 0 or mass <= 0:
        raise ValueError("need n2d >= 0 and mass > 0")
    return np.sqrt(2.0 * np.pi * float(n2d) * q / float(mass))


def slab_head_value(cell, kmesh, kappa=None, shape='cell'):
    """The number to pass as `head_value` to build_dfintegrals_screened.

    The mini-BZ average of the 2D head at the given screening wavevector.
    `kappa=None` gives the BARE average, which is the measure-correct value for
    an unscreened head and is MEASURED TO BE INADMISSIBLE on every mesh tried
    (module docstring) -- it is available so that fact stays reproducible, not
    because it should be used. Pass the metallic kappa for a metallic slab;
    kappa may be a callable kappa(phi) when `shape='cell'`.

    shape='cell' (default) averages over the true mini-BZ polygon; 'disc' uses
    the equal-area disc, which is isotropic and has a closed form. Measured
    difference in the bare head: 0.09% for a hexagonal cell (graphite), 0.55%
    for a square one, but 4.2% for a 2:1 rectangle -- small where the cell is
    round, not small where it is not, which is the same reason the response's
    own anisotropy cannot be ignored either.
    """
    k = 0.0 if kappa is None else kappa
    if shape == 'cell':
        return head_average_polygon(minibz_polygon(cell, kmesh), k)
    if shape == 'disc':
        if callable(k):
            raise ValueError("an angle-dependent kappa needs shape='cell'; "
                             "the equal-area disc is isotropic by construction")
        return head_average_2d(minibz_disc_radius(cell, kmesh), k)
    raise ValueError(f"unknown shape {shape!r}, expected 'cell' or 'disc'")


def sa_head_normalization(cell):
    """(L, finite_part) for the Sundararaman-Arias G -> 0 limit, in bohr.

    THE NORMALIZATION, derived rather than guessed, because it is the one
    place the head substitution goes wrong. pyscf's `dimension=2` kernel is

        v_SA(G) = 4 pi / |G|^2 [ 1 - cos(G_z L/2) exp(-G_par L/2) ]

    and it has two DIFFERENT G -> 0 limits:

        along G_z  (G_par = 0):  + pi L^2 / 2                     finite
        along G_par (G_z = 0):   2 pi L / G_par  -  pi L^2 / 2    divergent

    pyscf takes the G_par branch and keeps only its finite part, so
    `coulG[G=0] = -pi L^2 / 2 = -2 pi (L/2)^2` -- verified here to a ratio of
    1.000000 at L = 24, 30 and 40 A, and it is the ONLY negative entry in the
    whole kernel. The discarded divergence is `L * (2 pi / q)`: the 2D head
    `2 pi / q` that `head_average_2d` integrates over the mini-BZ in closed
    form, carrying ONE factor of the cell height because a 2D density in a 3D
    cell of height L has its 3D Fourier coefficient scaled by 1/L. Measured:
    `[v_SA(q) + pi L^2/2] * q / (2 pi L) -> 1.0000000086` at q = 1e-5.

    So the value the Gamma term actually stands for is

        coulG[G=0]  =  L * <2 pi / q>_miniBZ  -  pi L^2 / 2

    which is what `make_coulG_2d_head` substitutes.
    """
    a = cell.lattice_vectors()
    if abs(a[2, 0]) > 1e-10 or abs(a[2, 1]) > 1e-10:
        raise ValueError("the vacuum axis a3 must be along z for the 2D head; "
                         f"got a3 = {a[2]}.")
    L = float(a[2, 2])
    return L, -np.pi * L ** 2 / 2.0


def make_coulG_2d_head(cell, kmesh, kappa=0.0, shape='cell', base_coulG_fn=None):
    """pyscf's 2D kernel with the mini-BZ-averaged head RESTORED at G = 0.

    The point: pyscf discards a divergence rather than integrating it, and the
    leftover finite part is negative, which is what makes the RI-V metric
    indefinite and is the only reason the AUTO spherical damping was adopted
    for slabs. Damping fixes it at a price -- its real-space support must fit
    inside half the cell height, so refining the in-plane k-mesh FORCES a taller
    vacuum (measured: MoS2 needs 24.7 A at 12x12 and 61.6 A at 30x30), and
    since the z FFT mesh grows with L_z the grid-touching steps then scale as
    N_k^3 in-plane instead of N_k^2. Restoring the head instead removes the
    reason for damping, and with it that coupling.

    NOT UNCONDITIONALLY POSITIVE, and the condition is worth knowing because it
    runs OPPOSITE to damping's. The substituted value is positive iff

        <2 pi/q> * L  >  pi L^2 / 2   <=>   R < 8 / L     (bare head, disc)

    with R the mini-BZ radius -- i.e. this route wants the mesh FINE relative
    to the vacuum, where damping wants the vacuum TALL relative to the mesh.
    Production wants fine meshes, so the trade is favourable, but a coarse mesh
    on a tall cell still fails: measured, graphene in 24 A of vacuum is
    negative at 4x4 (-449) and positive from 6x6 (+942). It raises there rather
    than handing back an indefinite metric.

    `kappa > 0` uses the metal-screened head <2pi/(q+kappa)>, since a slab head
    is admissible only when metal-screened (module docstring); `shape='cell'`
    integrates the exact mini-BZ polygon, `'disc'` its equal-area disc.
    """
    L, finite = sa_head_normalization(cell)
    head = slab_head_value(cell, kmesh, kappa=kappa, shape=shape)
    value = L * head + finite
    if value <= 0.0:
        R = minibz_disc_radius(cell, kmesh)
        raise ValueError(
            f"restoring the 2D head still leaves v(G=0) = {value:.4g} <= 0 "
            f"(L<2pi/q> = {L * head:.4g}, finite part = {finite:.4g}). The "
            f"mini-BZ is too large for this vacuum: R = {R:.4g} against the "
            f"R < 8/L = {8.0 / L:.4g} this needs. Refine the in-plane k-mesh, "
            f"or shorten the vacuum -- note this is the OPPOSITE of what the "
            f"damped kernel would ask for.")

    def coulG_2d_head(cell_, q, Gv):
        Gv = np.asarray(Gv)
        base = (tools.get_coulG(cell_, k=np.asarray(q), Gv=Gv)
                if base_coulG_fn is None else base_coulG_fn(cell_, q, Gv))
        v = np.array(base, dtype=float, copy=True)
        at_gamma = np.linalg.norm(Gv + np.asarray(q), axis=1) < 1e-8
        if at_gamma.any():
            v[at_gamma] = value
        return v

    coulG_2d_head.head_2d = (float(head), float(value), tuple(kmesh), shape)
    coulG_2d_head.singular_law = None
    return coulG_2d_head


def make_coulG_constant_head(coulG_fn, q_ref):
    """Leon's constant approximation: v(q=0) := v(|q_ref|), everything else kept.

    The cheap, parameter-free baseline of the three published routes through the
    long-wavelength limit (Leon et al.): set the head at zero momentum equal to
    its value at the smallest momentum the grid actually represents, instead of
    evaluating the kernel at q = 0. It has no parameters and costs nothing.

    For the AUTO-damped kernel this matters because v(0) = 4 pi int r theta(r) dr
    grows as r0^2 with the k-grid -- a Gamma over-screening -- while v(|q_ref|)
    does not. Substituting at the
    single G = 0 point is a POINT-valued fix and therefore cannot represent the
    sqrt(q) wings of a 2D metal (that needs a neighbourhood-valued scheme); its
    role is to give the Gamma-convergence study a defensible baseline.

    The substitution value is read off the base kernel along +x, so it assumes
    the kernel is ISOTROPIC in |q+G| -- true of both the damped kernel and
    pyscf's get_coulG, and checked for the damped one by the mesh-invariance
    test. An anisotropic kernel would need a direction-resolved reference.

    The returned kernel carries over any `damping` tag, so the damping-support
    check (`pbc_rpa_damping.assert_damping_fits`) still reaches it.
    """
    q_ref = float(q_ref)
    if q_ref <= 0:
        raise ValueError(f"reference momentum must be positive, got {q_ref}")
    cache = {}

    def coulG_constant_head(cell, q, Gv):
        Gv = np.asarray(Gv)
        v = np.array(coulG_fn(cell, q, Gv), dtype=float, copy=True)
        kabs = np.linalg.norm(Gv + np.asarray(q), axis=1)
        head = kabs < 1e-8
        if head.any():
            key = id(cell)
            if key not in cache:
                probe = np.zeros((1, 3))
                probe[0, 0] = q_ref
                cache[key] = float(coulG_fn(cell, np.zeros(3), probe)[0])
            v[head] = cache[key]
        return v

    damping = getattr(coulG_fn, 'damping', None)
    if damping is not None:
        coulG_constant_head.damping = damping
    coulG_constant_head.constant_head_q = q_ref
    return coulG_constant_head


def smallest_transfer(cell, kmesh):
    """|q| of the smallest non-zero in-plane momentum transfer on `kmesh`.

    The reference momentum for `make_coulG_constant_head`. Uses the minimum
    image convention, so it is the true shortest transfer rather than the
    shortest fractional coordinate.
    """
    b = cell.reciprocal_vectors()
    grids = [np.arange(n) / n for n in kmesh]
    frac = np.array(np.meshgrid(*grids, indexing='ij')).reshape(3, -1).T
    frac = (frac + 0.5) % 1.0 - 0.5
    qabs = np.linalg.norm(frac.dot(b), axis=1)
    nonzero = qabs[qabs > 1e-8]
    if not len(nonzero):
        raise ValueError("a Gamma-only k-mesh has no non-zero momentum "
                         "transfer to take the constant approximation from.")
    return float(nonzero.min())
