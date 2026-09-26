"""
AUTO Fermi-Dirac Coulomb damping for periodic RI-RPA
(Forster et al., JCTC 2025, 21, 9347; eqs 24-26).

The damped kernel is v_damped(r) = theta(r)/r with the spherical Fermi-Dirac
cutoff theta(r) = 1/(1+exp(beta*(r-r0))). Its 3D Fourier transform is radial:

    vtilde(k)  = (4 pi / k) int_0^inf theta(r) sin(k r) dr          (k > 0)
    vtilde(0)  = 4 pi int_0^inf r theta(r) dr                        (finite!)

so the Gamma divergence is removed and replaced by a finite value that grows
with r0 (hence with the k-grid) -- the source of the paper's clean convergence.

AUTO scheme: r0 is tied to the k-grid Nyquist radius Rc (eq 24), so damping ->
none as the grid grows, recovering the true (undamped) RPA correlation energy.

Returns a `coulG_fn(cell, q, Gv)` suitable for pbc_rpa.ri_rpa_ecorr.

LOW-DIMENSIONAL SYSTEMS (slabs, wires). No separate "2D-truncated" kernel is
needed, and none is implemented, because the damping is SPHERICAL and therefore
self-truncating. Per the paper: eq 25's theta depends on the full distance
|r2 - r1| in any dimensionality, and eq 24's Nyquist parallelepiped is
n-dimensional (Figure 1's Rc is the largest CIRCLE inscribed in it for a 2D
system). So a slab differs from bulk in exactly one respect -- Rc is set by the
IN-PLANE Nyquist distances only -- which `nyquist_params` already handles by
ignoring directions sampled at a single k-point.

What a slab DOES require is a validity constraint that bulk does not:
`check_low_dim_support` below. The G-space kernel is the analytic radial FT of
the infinite-space function theta(r)/r; using it in a periodic cell implies a
lattice sum of that function, so its real-space support must FIT INSIDE the cell
along every non-sampled direction. Otherwise neighbouring images overlap, the
implied potential is not the intended one, and the error is SILENT. Because r0
grows with the in-plane k-grid while the vacuum is fixed, this constraint is
eventually violated by refining the k-mesh alone -- e.g. for MgO(001) with
|a| = 2.977 A, a 9x9 grid needs ~19 A of cell height but a 17x17 grid needs ~35 A.
Check it per k-mesh; do not assume a vacuum chosen for a coarse grid still works.
"""
import numpy as np
from src.Base.constants import BOHR_TO_ANGSTROM


def fermi_dirac_theta(r, r0, beta):
    return 1.0 / (1.0 + np.exp(np.clip(beta * (r - r0), -500, 500)))


def damping_support(r0, beta, tol=1e-3):
    """Radius beyond which theta < tol, i.e. the real-space support of the damped
    Coulomb kernel. Inverts theta(R) = tol."""
    return r0 + np.log(1.0 / tol - 1.0) / beta


def check_low_dim_support(cell, kmesh, r0, beta, tol=1e-3, raise_on_fail=True):
    """Verify the damped kernel's support fits the cell along non-sampled directions.

    For every NON-PERIODIC direction (the vacuum of a slab or wire: axis index
    >= cell.dimension, which pyscf always samples at one k-point), the spherical
    damping must have decayed to `tol` within HALF the cell length, so that the
    periodic images implied by the G-space kernel do not overlap. Returns a list
    of (axis, need, have) for the failing directions; raises by default, since
    the resulting error is silent and looks like poor k-convergence.

    A periodic direction is NOT checked, even when it carries a single k-point.
    There the lattice sum is over real neighbours and overlapping images are the
    physics, not an artifact -- a 3D bulk cell run at [2,2,1] is merely coarse
    along a3, not a slab. (This distinction was in the original code as
    `i >= cell.dimension or norm(a[i]) > 0`, but the second clause is true for
    every real lattice vector, so the gate never applied. It never mattered
    while nothing called this function; wiring it into the integral builders
    made it matter, and made a legitimate 3D test fail.)
    """
    a = cell.lattice_vectors()                    # Bohr
    BOHR = BOHR_TO_ANGSTROM
    R = damping_support(r0, beta, tol)            # Bohr (r0/beta are in Bohr)
    bad = []
    for i in range(3):
        if kmesh[i] > 1:
            continue                              # sampled: Rc already accounts for it
        if i < getattr(cell, 'dimension', 3):
            continue                              # periodic: images are physical
        half = 0.5 * np.linalg.norm(a[i])
        if half < R:
            bad.append((i, R * BOHR, half * BOHR))
    if bad and raise_on_fail:
        msg = "; ".join(f"axis {i}: damping reaches {need:.1f} A but half the cell "
                        f"is only {have:.1f} A" for i, need, have in bad)
        raise ValueError(
            f"damped Coulomb kernel wraps around the cell ({msg}). The spherical "
            f"damping is self-truncating ONLY if its support fits the cell along "
            f"non-periodic directions. Increase the vacuum to > "
            f"{2 * R * BOHR:.1f} A, or coarsen the k-grid (r0 grows with it).")
    return bad


def nyquist_params(cell, kmesh, frac_r0=0.5, tail_frac=1.3, tail_level=1e-3):
    """Return (r0, beta, Rc) from the k-grid via the Nyquist construction (eq 24).

    Rmax_i = N_i * |a_i| is the max representable real-space distance along
    lattice direction i; Rc = 0.5*min Rmax over the *k-sampled* directions
    (inscribed radius). r0 = frac_r0 * Rc, and beta is set so
    theta(tail_frac*r0) = tail_level.

    Direction-aware: only directions sampled at more than one k-point (N_i > 1)
    grow their Nyquist distance as the grid is refined, so only they may set the
    damping radius. A direction with a single k-point -- a transverse vacuum box
    in a slab/chain, or an unconverged direction -- has an effectively fixed
    real-space extent and must NOT pin Rc; otherwise r0 saturates and the AUTO
    scheme cannot converge to the thermodynamic limit as the k-grid grows. Falls
    back to all three directions only for a pure Gamma-point calculation.
    """
    a = cell.lattice_vectors()
    amax = np.array([np.linalg.norm(a[i]) for i in range(3)])
    Rmax = np.array(kmesh, dtype=float) * amax
    sampled = np.asarray(kmesh) > 1
    dirs = sampled if sampled.any() else np.ones(3, dtype=bool)
    Rc = 0.5 * Rmax[dirs].min()
    r0 = frac_r0 * Rc
    beta = np.log(1.0 / tail_level - 1.0) / ((tail_frac - 1.0) * r0)
    return r0, beta, Rc


def kmesh_from_kpts(cell, kpts):
    """The Monkhorst-Pack mesh behind a k-point list, as a 3-tuple.

    check_low_dim_support and nyquist_params are both phrased in terms of the
    mesh, but the integral builders are handed kpts, so wiring the check into
    them needs this inverse. Raises on anything that is not a regular grid --
    a silent wrong mesh here would defeat the check it feeds.
    """
    scaled = cell.get_scaled_kpts(np.asarray(kpts))
    mesh = []
    for i in range(3):
        vals = np.mod(np.round(scaled[:, i], 8), 1.0)
        mesh.append(len(np.unique(np.round(vals, 6))))
    if int(np.prod(mesh)) != len(kpts):
        raise ValueError(
            f"kpts is not a regular Monkhorst-Pack grid: {len(kpts)} points do "
            f"not factor as {mesh[0]}x{mesh[1]}x{mesh[2]}. Pass the mesh "
            f"explicitly to check_low_dim_support.")
    return tuple(mesh)


def assert_damping_fits(cell, kpts, coulG_fn, tol=1e-3):
    """Run check_low_dim_support for a damped kernel, given only the kernel.

    A no-op for any other coulG_fn (a bare or user-supplied kernel implies no
    real-space damping envelope, so there is nothing to fit). make_coulG_damped
    tags its return value with the (r0, beta) it was built from, which is what
    makes this reachable from inside an integral builder that never saw them.

    This is the wiring the check needs: its violation is SILENT and looks
    like poor k-convergence.
    """
    damping = getattr(coulG_fn, 'damping', None)
    if damping is None:
        return None
    r0, beta = damping
    return check_low_dim_support(cell, kmesh_from_kpts(cell, kpts), r0, beta, tol=tol)


def make_coulG_damped(r0, beta, nk_tab=4000, nr=20000):
    """Return coulG_fn(cell, q, Gv) -> vtilde(|q+G|) for the damped 3D Coulomb.

    Precomputes vtilde on a dense |k| table (the kernel is a pure function of
    |q+G|, so table lookups are consistent across k-meshes and supercells, which
    preserves the k-mesh <-> supercell identity exactly at fixed r0).
    """
    Rint = r0 + 25.0 / beta          # theta ~ 0 beyond this
    r = np.linspace(0.0, Rint, nr)
    th = fermi_dirac_theta(r, r0, beta)
    dr = r[1] - r[0]
    w = np.ones(nr); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0; w *= dr / 3.0   # Simpson
    v0 = 4.0 * np.pi * np.sum(w * r * th)         # vtilde(0), finite

    def _table(kmax):
        ktab = np.linspace(0.0, kmax * 1.02 + 1e-6, nk_tab)
        vtab = np.empty_like(ktab)
        vtab[0] = v0
        chunk = 200
        for p0 in range(1, nk_tab, chunk):
            p1 = min(p0 + chunk, nk_tab)
            kk = ktab[p0:p1][:, None]
            Ik = np.sum(w[None, :] * th[None, :] * np.sin(kk * r[None, :]), axis=1)
            vtab[p0:p1] = 4.0 * np.pi / ktab[p0:p1] * Ik
        return ktab, vtab

    cache = {}

    def coulG_fn(cell, q, Gv):
        kabs = np.linalg.norm(Gv + q, axis=1)
        key = round(kabs.max(), 3)
        if key not in cache:
            cache[key] = _table(max(kabs.max(), 1.0))
        ktab, vtab = cache[key]
        return np.interp(kabs, ktab, vtab)

    # Carry the damping parameters on the function itself, so a builder that
    # is handed only a coulG_fn can still run assert_damping_fits.
    coulG_fn.damping = (r0, beta)
    return coulG_fn
