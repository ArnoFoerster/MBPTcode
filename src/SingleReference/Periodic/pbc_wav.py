"""W-av: mini-Brillouin-zone averaging of the interaction.

`pbc_self_energy.q_grid` returns a uniform momentum grid with equal weights and
`sigma_c_diag` sums over it directly. For a three-dimensional semiconductor
that is defensible. For a metallic slab, where the integrand carries a sqrt(q)
non-analyticity, it is the accuracy-limiting step -- and every published route
through the long-wavelength limit (an explicit Drude head, the constant
approximation, self-energy fitting, shifted-grid averaging) is a different
answer to the same question: what should that sum do near the origin.

This module implements the answer that acts on the whole zone rather than on
one point. Following Guandalini and Sesti, the interaction at each grid point
is replaced by its average over the mini-BZ cell SURROUNDING that point,

    vbar(k) = (1/A_mBZ) int_{mBZ} v(k + u) d^2u ,

with u running over the in-plane mini-BZ (2D, because only the in-plane
directions are sampled). Three properties matter, and they are the reason this
is the target rather than a Gamma-point patch:

  * The average is taken around EVERY grid point, so there is no discontinuity
    between a specially-treated region and the rest of the zone.
  * The grid stays uniform, so no new convergence parameter is introduced. The
    contrast the original authors draw is with sub-sampling schemes, whose
    specially-sampled region shrinks as the grid is refined, making results
    from different grids incomparable. That matters here more than it does for
    them: our AUTO damping radius r0 grows with the in-plane k-mesh, so
    refining the grid already changes the kernel as well as the sampling, and a
    scheme whose special region also moves would be very hard to interpret.
  * It is neighbourhood-valued. A measurement here found independently that
    the head cannot be treated pointwise in the mixed representation at all
    (pbc_smallq), so a point-valued fix -- a Drude head, or the constant
    approximation -- cannot represent the wings whatever value it is given.

Why this matters here beyond the head: the damped kernel RINGS
--------------------------------------------------------------
The AUTO-damped kernel is the Fourier transform of a sharply truncated
theta(r)/r, so it does not decay monotonically -- it oscillates, with period
2 pi / r0. Under the AUTO scheme r0 = frac_r0 * 0.5 * N * |a|, so that period
is EXACTLY 4 grid spacings, at every k-mesh:

    2 pi / r0 = 8 pi / (N |a|) = 4 * (2 pi / (N |a|)) = 4 * |b| / N .

Refining the k-grid therefore never improves the sampling of that oscillation:
the q-sum always sees the same four points per period. Measured on the H2 slab
the mini-BZ average moves the kernel by several percent at momenta far from the
origin, with the SIGN tracking the local curvature of the ringing (-1.32 at
|k| = 0.6, +0.26 at 1.0, -0.16 at 2.2, +0.07 at 3.0), and it agrees with the
second-order prediction (1/2)<u^2> lap(v) to within 15%. So in this
architecture mini-BZ averaging is not only a long-wavelength fix -- it
integrates over a ringing the grid cannot resolve at any mesh. That also
explains why the head ratios of both this scheme and the constant approximation
come out mesh-invariant: nothing about the sampling of that oscillation changes
with N.

What is here, and what is not
-----------------------------
`make_coulG_minibz_averaged` wraps any FINITE kernel -- in practice the
AUTO-damped one, whose v(0) = 4 pi int r theta(r) dr is finite by construction
-- and returns a coulG_fn that is its mini-BZ average. It is exact for a
constant kernel, converges in the quadrature order, and needs no small-q law,
because a damped kernel has no singularity to expand around. Dropping it into
any of the four builders is the whole integration: the kernel is the only place
the rapid q-dependence lives in an RI-V build.

The interpolant half of the published scheme is here too. Guandalini and Sesti
average an auxiliary function constructed to be smoother in momentum than the
interaction, parametrized bilinearly over nearest neighbours, which is what
lets them handle a genuinely SINGULAR interaction and what carries the sqrt(q)
wing behaviour of a 2D metal. `auxiliary_function`, `cell_moments` and
`wav_bz_average` build it on the in-plane transfer grid for a supplied small-q
`law`, and `make_coulG_minibz_averaged(law=...)` applies it to a singular
kernel; WITHOUT a law a singular kernel is refused rather than silently
averaged badly. The sqrt(q) regime itself needs an intraband response, which
an integer-occupation polarizability does not have; that is the
occupation-weighted response of `pbc_occupations`.
"""
import numpy as np

from src.SingleReference.Periodic.pbc_smallq import (inplane_cell_area,
                                                     minibz_polygon, polygon_area,
                                                     polygon_radial_moments)


def triangulate(vertices):
    """Fan-triangulate a convex polygon from its centroid.

    The centroid fan (rather than a fan from vertex 0) keeps every triangle
    well-shaped, which matters because the quadrature below is mapped through a
    Duffy transform whose accuracy degrades on slivers.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        raise ValueError("need at least three 2D vertices")
    c = v.mean(axis=0)
    return [(c, v[i], v[(i + 1) % len(v)]) for i in range(len(v))]


def polygon_quadrature(vertices, order=8):
    """(points, weights) integrating any smooth function over a convex polygon.

    Each triangle is mapped from the unit square by the Duffy transform
        x(s,t) = p0 + s (p1 - p0) + s t (p2 - p1),   |J| = 2 A s ,
    so a tensor-product Gauss-Legendre rule in (s, t) converges spectrally for a
    smooth integrand instead of at the fixed order a symmetric triangle rule
    would give. `order` is the number of Gauss points per direction per
    triangle; the weights sum to the polygon's area.
    """
    nodes, gw = np.polynomial.legendre.leggauss(int(order))
    s = 0.5 * (nodes + 1.0)
    ws = 0.5 * gw
    S, T = np.meshgrid(s, s, indexing='ij')
    W = np.outer(ws, ws)

    pts, wts = [], []
    for p0, p1, p2 in triangulate(vertices):
        area2 = float(np.cross(p1 - p0, p2 - p0))          # 2 * signed area
        xy = (p0[None, None, :] + S[..., None] * (p1 - p0)[None, None, :]
              + (S * T)[..., None] * (p2 - p1)[None, None, :])
        pts.append(xy.reshape(-1, 2))
        wts.append((W * S * abs(area2)).ravel())
    return np.concatenate(pts), np.concatenate(wts)


def minibz_quadrature(cell, kmesh, order=8):
    """(offsets, weights) averaging over one mini-BZ cell; weights sum to 1.

    The offsets are in-plane displacements about the cell centre, so the same
    set applies around EVERY grid point -- which is the property that keeps the
    grid uniform and introduces no region-dependent parameter.
    """
    poly = minibz_polygon(cell, kmesh)
    centre = poly.mean(axis=0)
    pts, wts = polygon_quadrature(poly, order=order)
    return pts - centre, wts / wts.sum()


def make_coulG_minibz_averaged(coulG_fn, cell, kmesh, order=8, max_ratio=4.0,
                               law=None, nang=24, nrad=24, law_rtol=1e-3):
    """Return a coulG_fn giving the mini-BZ average of `coulG_fn`.

    vbar(q+G) = (1/A_mBZ) int_{mBZ} v(q+G+u) d^2u, with u in-plane, evaluated by
    the fixed quadrature of `minibz_quadrature`. The same offsets are used at
    every grid point, so this is not a Gamma-point special case.

    A FINITE kernel (the damped one, whose v(0) = 4 pi int r theta(r) dr) needs
    nothing else: quadrature is accurate on a smooth integrand and `law` is
    ignored. A SINGULAR kernel is the case the auxiliary-function half of the
    module exists for, and `law` is what enables it -- see below. Without a
    law a singular kernel is still refused, because quadrature on a divergent
    integrand converges to whatever the nodes happened to miss.

    Singularity is detected by probing the SCALING at the origin rather than
    the magnitude -- shrinking |k| by 16 and asking whether the kernel grows by
    more than `max_ratio` -- because pyscf returns v(G=0) = 0 with exxdiv=None
    while diverging either side of it, so finiteness at the origin identifies
    nothing. The margins are wide: a damped kernel grows by ~1 over that range,
    pyscf's 2D-truncated kernel by 16 and a 3D 4pi/k^2 by 256.

    The singular case, and why only one point needs it
    --------------------------------------------------
    `law` is f(|k|), the known small-k form of the kernel (2 pi / k for a bare
    2D interaction, 2 pi / (k + kappa) screened, 4 pi / k^2 in 3D), called with
    a flat array of |k| exactly as `auxiliary_function` and
    `polygon_radial_moments` call it. Write v = f * A with A smooth by design.
    Then over one cell

        int_cell v  =  int_cell f A  ~  Abar * int_cell f  =  Abar * I0 ,

    with I0 from `polygon_radial_moments`, whose 2D measure k dk absorbs an
    integrable 1/k, and Abar the quadrature mean of v/f over the cell. This is
    EXACT whenever the kernel is its own law (A == 1), which is the case that
    matters: the head of a bare Coulomb interaction.

    Only ONE entry ever needs this. |q+G| vanishes only when q is a reciprocal
    lattice vector, i.e. at q = Gamma with G = 0; every other grid point sits a
    full |G| ~ 2 pi / a from the origin while the cell radius is ~ pi / (a N),
    so its integrand is smooth and plain quadrature is already right. The
    special case is therefore a single row, detected numerically rather than
    assumed, and the rest of the grid is untouched.

    A wrong `law` is caught rather than absorbed: if v/f is not constant over
    the cell to `law_rtol` then f is not the kernel's actual singularity and
    the zeroth-order form is not justified, so it raises. That check is the
    reason a mismatched law cannot silently produce a plausible head.

    Carries the `damping` tag through, so the damping-support check
    (`pbc_rpa_damping.assert_damping_fits`) still reaches the wrapped kernel.
    """
    offsets, weights = minibz_quadrature(cell, kmesh, order=order)
    pad = np.zeros((len(offsets), 3))
    pad[:, :2] = offsets

    nq = len(pad)

    # Identify a SINGULAR kernel before wrapping it. Finiteness alone does not:
    # pyscf's get_coulG returns 0 at G = 0 with exxdiv=None while diverging as
    # 4pi/k^2 either side of it. Probe the scaling instead -- halve |k| twice
    # and see whether the kernel grows like a power.
    eps = 1e-3 * np.linalg.norm(pad, axis=1).max()
    probe = np.zeros((3, 3))
    probe[:, 0] = [eps, eps / 4.0, eps / 16.0]
    pv = np.abs(np.asarray(coulG_fn(cell, np.zeros(3), probe), dtype=float))
    growth = pv[2] / max(pv[0], 1e-300)
    singular = (not np.isfinite(growth)) or growth > max_ratio

    head_mean_f, f_nodes = None, None
    if singular:
        if law is None:
            raise ValueError(
                f"the kernel grows by {growth:.3g} as |k| falls from {eps:.2e} "
                f"to {eps / 16:.2e}, i.e. it is SINGULAR at the origin, and a "
                f"quadrature average of a singular kernel converges to "
                f"whatever the nodes happened to miss. Either average a damped "
                f"kernel (pbc_rpa_damping.make_coulG_damped, whose v(0) is "
                f"finite by construction), or pass law=f giving the kernel's "
                f"known small-k form -- 2*pi/k bare 2D, 2*pi/(k+kappa) "
                f"screened, 4*pi/k**2 in 3D -- which routes the one divergent "
                f"grid point through the auxiliary-function form.")
        poly = minibz_polygon(cell, kmesh)
        I0, _ = polygon_radial_moments(poly, law, nang=nang, nrad=nrad)
        head_mean_f = float(I0 / polygon_area(poly))
        f_nodes = np.asarray(law(np.linalg.norm(offsets, axis=1)), dtype=float)
        if not np.all(np.isfinite(f_nodes)) or np.any(f_nodes == 0.0):
            raise ValueError(
                "`law` must be finite and nonzero at every quadrature node of "
                "the mini-BZ cell; it is divided into the kernel there.")

    # Exact zero in practice, but a tolerance on the mini-BZ scale rather than
    # an == 0 test, since q+G is assembled by floating-point addition.
    origin_tol = 1e-6 * np.linalg.norm(offsets, axis=1).max()

    def coulG_averaged(cell_, q, Gv, blk=4096):
        Gv = np.asarray(Gv, dtype=float)
        q = np.asarray(q, dtype=float)
        acc = np.empty(len(Gv))
        # One base-kernel call per block of G, over ALL quadrature nodes at
        # once. Calling per node instead would be correct but pathological: a
        # tabulated kernel keys its table on the block's largest |q+G|, which
        # a per-node call perturbs, so every node would rebuild the table.
        for p0 in range(0, len(Gv), blk):
            g = Gv[p0:p0 + blk]
            shifted = (g[:, None, :] + pad[None, :, :]).reshape(-1, 3)
            v = np.asarray(coulG_fn(cell_, q, shifted), dtype=float)
            v = v.reshape(len(g), nq)

            at_origin = (np.linalg.norm(q[None, :] + g, axis=1) < origin_tol
                         if singular else np.zeros(len(g), dtype=bool))
            if not np.all(np.isfinite(v[~at_origin])):
                raise ValueError(
                    "the kernel is not finite on the shifted mini-BZ grid, so "
                    "it cannot be averaged by quadrature. Use a damped kernel "
                    "(pbc_rpa_damping.make_coulG_damped), whose v(0) is finite "
                    "by construction, or pass law=f to route the divergent "
                    "point through the auxiliary-function form.")
            acc[p0:p0 + blk] = v @ weights

            if at_origin.any():
                # v = f A: integrate f exactly over the cell, carry A by its
                # quadrature mean. Exact when the kernel IS its own law.
                A = v[at_origin] / f_nodes[None, :]
                if not np.all(np.isfinite(A)):
                    raise ValueError(
                        "v/law is not finite at the mini-BZ quadrature nodes "
                        "of the divergent point, so the kernel diverges faster "
                        "than the law supplied.")
                Abar = A @ weights
                spread = (np.abs(A - Abar[:, None]).max(axis=1)
                          / np.maximum(np.abs(Abar), 1e-300))
                if np.any(spread > law_rtol):
                    raise ValueError(
                        f"v/law varies by {spread.max():.3g} over the mini-BZ "
                        f"cell at the divergent point, above law_rtol="
                        f"{law_rtol:g}, so `law` is not this kernel's actual "
                        f"singularity and averaging f exactly would not be "
                        f"averaging v. Supply the matching small-k form.")
                acc[np.flatnonzero(at_origin) + p0] = Abar * head_mean_f
        return acc

    damping = getattr(coulG_fn, 'damping', None)
    if damping is not None:
        coulG_averaged.damping = damping
    coulG_averaged.minibz_average = (tuple(int(n) for n in kmesh), int(order))
    coulG_averaged.singular_law = law if singular else None
    return coulG_averaged


# ---------------------------------------------------------------------------
# The interpolant half: averaging a quantity known ONLY on the transfer grid.
# ---------------------------------------------------------------------------
#
# Averaging the KERNEL needs no interpolation, because a coulG_fn can be called
# anywhere. The published scheme's other half is for the case where it cannot:
# the screened interaction, or any per-q summand, exists only at grid points.
# There the construction is
#
#     F(q) = f(q) A(q) ,      f = the known small-q law, A smooth by design,
#
# with A parametrized over neighbouring grid points and f integrated exactly
# over each mini-BZ cell. Expanding A to first order about the cell centre,
#
#     int_{cell(q)} F = A(q) I0(q) + grad A(q) . [ I1(q) - q I0(q) ] + O(u^2) ,
#
# where I0 = int f d^2k and I1 = int k f d^2k over that cell come from
# pbc_smallq.polygon_radial_moments -- which is why that had to work for cells
# NOT containing the origin. Two exactness properties fall out and are pinned
# in the tests: for F = f the scheme is EXACT (A = 1, grad A = 0, and the cells
# tile the zone), and for f = 1 it reduces to the plain uniform average (I0 is
# the cell area and the first moment about the centre vanishes by the cell's
# inversion symmetry).


def transfer_grid(cell, kmesh):
    """Minimum-image in-plane transfer points of a Gamma-centred MP mesh.

    Shape (N1, N2, 2), in bohr^-1. Minimum image matters: the mini-BZ cells
    tile the first Brillouin zone only if each q is its own shortest
    representative, and the small-q law is a function of |q|, not periodic.
    """
    n1, n2 = int(kmesh[0]), int(kmesh[1])
    b = cell.reciprocal_vectors()
    f1 = ((np.arange(n1) / n1) + 0.5) % 1.0 - 0.5
    f2 = ((np.arange(n2) / n2) + 0.5) % 1.0 - 0.5
    F1, F2 = np.meshgrid(f1, f2, indexing='ij')
    return (F1[..., None] * b[0][:2] + F2[..., None] * b[1][:2])


def _clip_to_convex(poly, clipper):
    """Sutherland-Hodgman clip of a convex polygon against a convex clipper,
    both counter-clockwise. Returns [] if they do not overlap."""
    out = [np.asarray(v, dtype=float) for v in poly]
    n = len(clipper)
    for i in range(n):
        a, b = clipper[i], clipper[(i + 1) % n]
        edge = b - a
        normal = np.array([edge[1], -edge[0]])          # outward for CCW
        offset = float(a @ normal)
        clipped = []
        m = len(out)
        for j in range(m):
            p, q = out[j], out[(j + 1) % m]
            dp = float(p @ normal) - offset
            dq = float(q @ normal) - offset
            if dp <= 1e-14:
                clipped.append(p)
            if (dp < -1e-14 < 1e-14 < dq) or (dq < -1e-14 < 1e-14 < dp):
                clipped.append(p + (q - p) * (dp / (dp - dq)))
        out = clipped
        if len(out) < 3:
            return []
    return out


def cell_moments(cell, kmesh, law, nang=24, nrad=24, nimg=1):
    """(I0, I1c) per transfer point: int f d^2k over its mini-BZ, and the first
    moment of f about that cell's own centre.

    The cells must TILE the first Brillouin zone, because the identity the
    scheme rests on -- exactness when F is proportional to f -- is
    sum_q I0(q) = int_BZ f. On an ODD mesh the minimum-image cells already
    tile it. On an EVEN one they do not: a transfer at fractional 1/2 sits on
    the zone boundary, so its cell straddles the edge, and since f(|k|) is a
    function of |k| rather than a periodic function, the half that hangs
    outside is NOT the half that is missing on the opposite edge. Each cell is
    therefore clipped to the zone and its periodic images are added back, which
    makes the tiling exact for every mesh. (Measured before the fix: the sum
    was 10% low at 2x2 and 2.5% low at 4x4, and exact at 3x3 -- the parity
    signature of precisely this.)
    """
    poly = minibz_polygon(cell, kmesh)
    bz = list(minibz_polygon(cell, [1, 1, 1]))
    b = cell.reciprocal_vectors()
    q = transfer_grid(cell, kmesh)
    n1, n2 = q.shape[:2]
    shifts = [i * b[0][:2] + j * b[1][:2]
              for i in range(-nimg, nimg + 1) for j in range(-nimg, nimg + 1)]

    I0 = np.zeros((n1, n2))
    I1c = np.zeros((n1, n2, 2))
    for i in range(n1):
        for j in range(n2):
            centre = q[i, j]
            for G in shifts:
                piece = _clip_to_convex(poly + centre + G, bz)
                if len(piece) < 3:
                    continue
                piece = np.array(piece)
                if polygon_area(piece) < 1e-14:
                    continue
                m0, m1 = polygon_radial_moments(piece, law, nang=nang, nrad=nrad)
                I0[i, j] += m0
                I1c[i, j] += m1 - (centre + G) * m0
    return I0, I1c


def values_on_transfer_grid(values, cell, kpts, kmesh):
    """Reshape a per-q array in k-point order onto the (N1, N2) transfer grid.

    `ri_rpa_ecorr_from_dfints(..., return_per_q=True)` and the self-energy
    q-loop both hand back one number per momentum transfer in the k-point
    list's own order; `wav_bz_average` needs them laid out on the mesh. Refuses
    anything that is not the stated regular grid rather than silently
    misplacing a transfer, which would scramble the neighbour differences.
    """
    values = np.asarray(values, dtype=float)
    n1, n2 = int(kmesh[0]), int(kmesh[1])
    if values.shape != (len(kpts),):
        raise ValueError(f"expected one value per k-point ({len(kpts)}), "
                         f"got shape {values.shape}")
    scaled = cell.get_scaled_kpts(np.asarray(kpts))
    scaled = scaled - scaled[0]
    out = np.full((n1, n2), np.nan)
    for v, s in zip(values, scaled):
        i = int(round((s[0] % 1.0) * n1)) % n1
        j = int(round((s[1] % 1.0) * n2)) % n2
        if abs(s[0] % 1.0 * n1 - round(s[0] % 1.0 * n1)) > 1e-6 or \
           abs(s[1] % 1.0 * n2 - round(s[1] % 1.0 * n2)) > 1e-6:
            raise ValueError("the k-points are not on the stated "
                             f"{n1}x{n2} Monkhorst-Pack grid")
        if abs(s[2] % 1.0) > 1e-8:
            raise ValueError("the slab transfer grid needs k_z = 0 throughout")
        out[i, j] = v
    if np.isnan(out).any():
        raise ValueError(f"the k-point list does not cover the {n1}x{n2} grid")
    return out


def auxiliary_function(values, cell, kmesh, law, singular_at_gamma=None):
    """A(q) = F(q)/f(|q|) on the transfer grid, the function W-av interpolates.

    `values` is F on the (N1, N2) in-plane transfer grid. Where the law
    diverges at Gamma, F(Gamma)/f(Gamma) is not the limit of A -- the code's
    own Gamma value is whatever convention it uses there, not the physical
    divergence -- so A(Gamma) is EXTRAPOLATED from the nearest neighbour shell
    instead. That extrapolation is the whole reason the auxiliary function is
    introduced: it is what lets the scheme carry a singular law at all.

    `singular_at_gamma=None` decides by probing the law.
    """
    values = np.asarray(values, dtype=float)
    q = transfer_grid(cell, kmesh)
    qabs = np.linalg.norm(q, axis=-1)
    if values.shape != qabs.shape:
        raise ValueError(f"values has shape {values.shape}, expected "
                         f"{qabs.shape} for kmesh {kmesh}")

    gamma = np.unravel_index(np.argmin(qabs), qabs.shape)
    if qabs[gamma] > 1e-10:
        raise ValueError("the transfer grid has no Gamma point")
    if singular_at_gamma is None:
        probe = np.array([1e-4, 1e-5])
        fv = np.abs(np.asarray(law(probe), dtype=float))
        singular_at_gamma = not np.isfinite(fv[1]) or fv[1] > 4.0 * max(fv[0], 1e-300)

    safe = np.where(qabs > 1e-10, qabs, 1.0)
    f = np.asarray(law(safe.ravel()), dtype=float).reshape(qabs.shape)
    A = np.empty_like(values)
    A[:] = values / f
    if singular_at_gamma:
        shell = _nearest_shell(qabs, gamma)
        if not shell.any():
            raise ValueError("a Gamma-only mesh has no neighbour shell to "
                             "extrapolate the auxiliary function from")
        A[gamma] = float(A[shell].mean())
    else:
        A[gamma] = values[gamma] / float(np.asarray(law(np.zeros(1)))[0])
    return A


def _nearest_shell(qabs, gamma):
    """Boolean mask of the grid points at the smallest non-zero |q|."""
    nonzero = qabs[qabs > 1e-10]
    if not nonzero.size:
        return np.zeros_like(qabs, dtype=bool)
    rmin = nonzero.min()
    return (qabs > 1e-10) & (qabs < rmin * (1.0 + 1e-6))


def grid_gradient(values, cell, kmesh):
    """Cartesian gradient of a PERIODIC grid function, shape (N1, N2, 2).

    Central differences along the two mesh directions give the derivatives with
    respect to the fractional index; the Cartesian gradient then solves
    grad . (b_i/N_i) = dF/ds_i, the reciprocal-metric conversion the two
    directions need whenever they are not orthogonal.
    """
    F = np.asarray(values, dtype=float)
    n1, n2 = F.shape
    b = cell.reciprocal_vectors()
    d1 = (0.5 * (np.roll(F, -1, axis=0) - np.roll(F, 1, axis=0))
          if n1 > 2 else np.zeros_like(F))
    d2 = (0.5 * (np.roll(F, -1, axis=1) - np.roll(F, 1, axis=1))
          if n2 > 2 else np.zeros_like(F))
    M = np.array([b[0][:2] / int(kmesh[0]), b[1][:2] / int(kmesh[1])])
    rhs = np.stack([d1, d2], axis=-1)
    if n1 > 1 and n2 > 1 and abs(np.linalg.det(M)) > 1e-30:
        return rhs @ np.linalg.inv(M).T
    return rhs @ np.linalg.pinv(M).T


def auxiliary_gradient(A, cell, kmesh):
    """grad A on the transfer grid, by differencing A ITSELF.

    The obvious alternative -- use the product rule with the analytic f'(|q|)
    -- is wrong for this purpose, and measurably so. The scheme's defining
    property is that it is EXACT when F is proportional to f, which needs
    grad A to vanish identically when A is constant. Differencing F against an
    ANALYTIC f' does not give that: grad F comes from a coarse stencil while f'
    does not, so the two fail to cancel and the exactness is lost (measured:
    5.5e-2 relative at 3x3 where the correct construction gives 1e-16).
    Differencing A itself cancels the stencil error between numerator and
    denominator by construction.

    The price is at the zone boundary, where A = F/f is not periodic -- F is,
    the law f(|q|) is not -- so the wrapped difference there carries an error
    of order the variation of f across the zone. That is a correction to a
    correction, and the alternative loses the leading exactness, so this is the
    right trade; the tests bound what it costs.

    At Gamma the gradient multiplies a first moment that vanishes identically
    by the cell's inversion symmetry, so its value there is never used.
    """
    A = np.asarray(A, dtype=float)
    q = transfer_grid(cell, kmesh)
    gA = grid_gradient(A, cell, kmesh)
    gA[np.linalg.norm(q, axis=-1) <= 1e-10] = 0.0
    return gA


def wav_bz_average(values, cell, kmesh, law, nang=24, nrad=24,
                   with_gradient=True, return_terms=False):
    """The mini-BZ-averaged Brillouin-zone average of a grid function.

        (1/A_BZ) int_BZ F d^2q  ~  (1/A_BZ) sum_q [ A(q) I0(q)
                                                    + grad A(q) . I1c(q) ]

    against the plain uniform (1/N_k) sum_q F(q). `values` is F on the
    (N1, N2) in-plane transfer grid (see `transfer_grid`), `law` is f.

    Exact when F is proportional to f -- the defining property, since then
    A is constant and the cells tile the zone -- and identical to the uniform
    average when f is constant. `with_gradient=False` keeps only the
    zeroth-order term, which is the reweighting alone.
    """
    A = auxiliary_function(values, cell, kmesh, law)
    I0, I1c = cell_moments(cell, kmesh, law, nang=nang, nrad=nrad)
    zeroth = A * I0
    first = np.zeros_like(zeroth)
    if with_gradient:
        first = np.einsum('ijc,ijc->ij', auxiliary_gradient(A, cell, kmesh), I1c)
    area_bz = (2.0 * np.pi) ** 2 / inplane_cell_area(cell)
    total = float((zeroth + first).sum() / area_bz)
    if return_terms:
        return total, dict(A=A, I0=I0, I1c=I1c, zeroth=zeroth, first=first,
                           area_bz=area_bz)
    return total
