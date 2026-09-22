"""The classical induced-dipole response, against its own limits.

There is no reference implementation to check against here -- the point of
writing it was to avoid the cppe dependency -- so it is gated on physics that
has to hold rather than on numbers from elsewhere:

  isolated sites      T -> 0 with separation, so B -> diag(alpha)
  two sites, analytic on the axis the interaction is diagonal in, where the
                      coupled polarizability is alpha/(1 -/+ 2 alpha/r^3) and
                      can be written down
  boundedness         damped, T stays FINITE as the sites close in where
                      undamped it goes as r^-3. That is what the damping buys,
                      and it is less than it sounds: the coupled matrix is
                      still indefinite below ~2.2 Bohr for alpha = 5 either
                      way, so the claim under test is boundedness, not
                      stability
  stability            positive definite at separations sites actually occupy
  symmetry            B = B^T, since T is
  energy sign         polarizing an environment lowers the energy, always
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.polarizable_sites import (THOLE_FACTOR,
                                        dipole_interaction_matrix,
                                        induced_dipoles, polarization_energy,
                                        response_matrix)

ALPHA = 5.0          # Bohr^3, near a heavy-atom polarizability
FAR = 400.0          # Bohr; T falls as r^-3, so this is ~1e-8 of alpha^-1


def two_sites(r):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]), np.array([ALPHA, ALPHA])


def test_isolated_sites_are_uncoupled():
    """Far apart, each site polarizes on its own and B is diagonal."""
    coords, alphas = two_sites(FAR)
    B = response_matrix(coords, alphas)
    assert np.abs(B - np.diag(np.repeat(alphas, 3))).max() < 1e-6


def test_two_sites_match_the_analytic_coupled_polarizability():
    """On the axis joining them T is diagonal, so each direction decouples and
    the coupled response is alpha / (1 -/+ 2 alpha / r^3) along z and
    alpha / (1 -/+ alpha / r^3) across it -- the classic result, and the sign
    of the interaction differs between the two.

    Undamped, because that is the regime the closed form describes; the damping
    is checked separately below.
    """
    r = 12.0                                   # far enough that damping ~ 1
    coords, alphas = two_sites(r)
    B = response_matrix(coords, alphas, thole=None)
    a3 = ALPHA / r ** 3
    # symmetric/antisymmetric combinations diagonalize a two-site problem
    for comp, coupling in ((2, 2.0), (0, -1.0)):
        got = np.linalg.eigvalsh(B[np.ix_([comp, comp + 3], [comp, comp + 3])])
        want = sorted([ALPHA / (1.0 - coupling * a3),
                       ALPHA / (1.0 + coupling * a3)])
        assert np.allclose(sorted(got), want, rtol=1e-10)


def test_the_response_is_symmetric():
    coords = np.array([[0., 0., 0.], [3., 0., 0.], [0., 4., 1.]])
    B = response_matrix(coords, np.array([3.0, 5.0, 8.0]))
    assert np.abs(B - B.T).max() < 1e-12


@pytest.mark.parametrize('r', (2.5, 3.0, 6.0))
def test_the_response_is_stable_where_sites_actually_sit(r):
    """Positive definite at separations a real site list uses. 2.5 Bohr is
    1.3 Angstrom, already tighter than any polarizable centre pair."""
    coords, alphas = two_sites(r)
    assert np.linalg.eigvalsh(response_matrix(coords, alphas)).min() > 0.0


def test_damping_bounds_the_coupling_where_undamped_it_diverges():
    """The claim is about SCALING, not a fixed ratio. Quartering the separation
    multiplies the undamped coupling by 64; the damped one must stay bounded.

    A fixed-ratio assertion is what this test had first, and it fails: the
    ratio is 35 at 0.5 Bohr and 5 at 1.0, because the damping is only partial
    there. Bounded-versus-divergent is the property; "always ten times smaller"
    is not.

    Nor does the damping buy positive definiteness -- (alpha^-1 - T) is
    indefinite for this pair below ~2.2 Bohr either way.
    """
    def peak(r, thole):
        return np.abs(dipole_interaction_matrix(*two_sites(r), thole=thole)).max()
    assert peak(0.25, None) / peak(1.0, None) > 50.0        # ~64, i.e. r^-3
    assert peak(0.25, THOLE_FACTOR) < 2.0 * peak(1.0, THOLE_FACTOR)
    assert max(peak(r, THOLE_FACTOR) for r in (0.25, 0.5, 1.0)) < 1.0


def test_without_damping_the_coupling_diverges_as_r_cubed():
    """Halving the separation must multiply the undamped coupling by eight."""
    big = np.abs(dipole_interaction_matrix(*two_sites(1.0), thole=None)).max()
    small = np.abs(dipole_interaction_matrix(*two_sites(0.5), thole=None)).max()
    assert 7.5 < small / big < 8.5


def test_a_dipole_follows_the_field_it_is_given():
    coords, alphas = two_sites(FAR)
    field = np.zeros((2, 3))
    field[0, 2] = 0.01
    mu = induced_dipoles(coords, alphas, field)
    assert abs(mu[0, 2] - ALPHA * 0.01) < 1e-8
    # The far site feels the FIRST dipole's field, not nothing: to leading
    # order alpha^2 * 2/r^3 * E. Asserting a bound instead of this value would
    # pass on a coupling of the wrong sign or size.
    assert abs(mu[1, 2] - ALPHA ** 2 * 2.0 / FAR ** 3 * 0.01) < 1e-12


def test_polarizing_an_environment_lowers_the_energy():
    coords = np.array([[0., 0., 0.], [3., 0., 0.], [0., 4., 1.]])
    alphas = np.array([3.0, 5.0, 8.0])
    rng = np.random.default_rng(3)
    for _ in range(5):
        e = polarization_energy(coords, alphas, rng.normal(size=(3, 3)))
        assert e < 0.0


def test_coincident_sites_are_refused():
    coords = np.array([[0., 0., 0.], [0., 0., 1e-9]])
    with pytest.raises(ValueError, match='no damping makes that finite'):
        dipole_interaction_matrix(coords, np.array([1.0, 1.0]))


def test_a_negative_polarizability_is_refused():
    with pytest.raises(ValueError, match='must be positive'):
        response_matrix(np.array([[0., 0., 0.]]), np.array([-1.0]))


# --------------------------------------------------------------------- the
# continuum limit: discrete sites against exact dielectric electrostatics
# ---------------------------------------------------------------------------

def cm_ball(radius, per_axis, eps):
    """A cubic lattice clipped to a ball, at the Clausius-Mossotti density for `eps`.

        (eps - 1)/(eps + 2) = (4 pi / 3) n alpha

    so a ball of such sites IS a dielectric sphere of that permittivity, and
    its collective polarizability must be the textbook R^3 (eps-1)/(eps+2).
    """
    g = (np.arange(per_axis) + 0.5) / per_axis * 2 * radius - radius
    pts = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    pts = pts[np.linalg.norm(pts, axis=1) <= radius]
    density = len(pts) / (4.0 / 3.0 * np.pi * radius ** 3)
    cm = (eps - 1.0) / (eps + 2.0)
    return pts, np.full(len(pts), 3.0 / (4.0 * np.pi * density) * cm)


def collective_polarizability(coords, alphas, thole=None):
    """Total induced dipole per unit uniform field, averaged over the axes."""
    B = response_matrix(coords, alphas, thole=thole)
    out = []
    for x in range(3):
        field = np.zeros((len(coords), 3))
        field[:, x] = 1.0
        out.append((B @ field.reshape(-1)).reshape(-1, 3)[:, x].sum())
    return float(np.mean(out))


@pytest.mark.parametrize('eps,per_axis', [(1.5, 6), (1.5, 10), (2.0, 6), (2.0, 10)])
def test_a_clausius_mossotti_ball_is_a_dielectric_sphere(eps, per_axis):
    """The continuum limit, and the strongest statement available about B.

    An isolated-site test fixes the scale of alpha but says nothing about the
    dipole-dipole coupling; here the coupling IS the answer, since a ball of
    weakly polarizable points only reaches R^3 (eps-1)/(eps+2) because the
    dipoles depolarize one another. It is the same classical electrostatics a
    continuum solver discretizes, approached from the discrete side.

    Agreement is ~0.1-0.4% and PLATEAUS rather than vanishing with site count:
    Clausius-Mossotti is exact for an infinite lattice and a ball has a surface
    layer. Asserting convergence to zero would be asserting something false.

    NOTE WHAT THIS DOES NOT TEST. sum(alpha) equals the analytic sphere
    ALGEBRAICALLY under Clausius-Mossotti, so this passes with T = 0 -- better,
    in fact, than with the true coupling. It fixes the SCALE of alpha and says
    nothing about the dipole-dipole interaction;
    `test_a_spheroid_reproduces_its_depolarization_factors` is what tests that.
    """
    radius = 10.0
    coords, alphas = cm_ball(radius, per_axis, eps)
    got = collective_polarizability(coords, alphas)
    exact = radius ** 3 * (eps - 1.0) / (eps + 2.0)
    assert abs(got - exact) / exact < 0.01


def cm_spheroid(a, b, spacing, eps):
    """A prolate spheroid (semi-axes a > b = c) of Clausius-Mossotti sites.

    The lattice must be CUBIC -- one spacing for all three axes, clipped to the
    body. A lattice with different spacings per axis has its own anisotropic
    local field, which contaminates the shape anisotropy this is built to
    measure, and Clausius-Mossotti does not hold on it at all.
    """
    long_ax = np.arange(-a, a + 1e-9, spacing)
    short = np.arange(-b, b + 1e-9, spacing)
    pts = np.array(np.meshgrid(short, short, long_ax, indexing='ij')).reshape(3, -1).T
    pts = pts[(pts[:, 0] / b) ** 2 + (pts[:, 1] / b) ** 2 + (pts[:, 2] / a) ** 2 <= 1.0]
    volume = 4.0 / 3.0 * np.pi * a * b * b
    cm = (eps - 1.0) / (eps + 2.0)
    return pts, np.full(len(pts), 3.0 * volume / (4.0 * np.pi * len(pts)) * cm), volume


def depolarization_prolate(a, b):
    """(L_parallel, L_perpendicular) of a prolate spheroid, Landau-Lifshitz."""
    e = np.sqrt(1.0 - b * b / (a * a))
    l_par = (1 - e * e) / (e * e) * (np.log((1 + e) / (1 - e)) / (2 * e) - 1.0)
    return l_par, 0.5 * (1.0 - l_par)


def test_a_spheroid_reproduces_its_depolarization_factors():
    """THIS is the test with teeth, and the sphere above is not.

    Clausius-Mossotti makes sum(alpha) equal R^3 (eps-1)/(eps+2) ALGEBRAICALLY,
    so the sphere test passes with T = 0 -- and passes better than with the
    correct coupling, 0.00% against 0.35%. A sphere is precisely the shape
    whose depolarization cancels.

    A prolate spheroid does not cancel: its analytic polarizability is
    anisotropic through L_par != L_perp, while sum(alpha) is isotropic by
    construction. So every bit of the anisotropy is produced by the dipole
    coupling, and switching it off misses by 6-12% where keeping it lands
    inside 1%.
    """
    a, b, eps = 12.0, 6.0, 2.0
    coords, alphas, volume = cm_spheroid(a, b, 1.5, eps)
    l_par, l_perp = depolarization_prolate(a, b)
    for axis, depol in ((2, l_par), (0, l_perp)):
        exact = volume * (eps - 1.0) / (4.0 * np.pi * (1.0 + depol * (eps - 1.0)))
        with_t = _axis_polarizability(coords, alphas, axis)
        assert abs(with_t - exact) / exact < 0.02
        # the counter-check: no coupling gives the isotropic sum, which is wrong
        # for BOTH axes and by far more than the tolerance above
        assert abs(alphas.sum() - exact) / exact > 0.05


def _axis_polarizability(coords, alphas, axis, thole=None):
    """Collective polarizability along ONE axis, the shape-sensitive quantity."""
    B = response_matrix(coords, alphas, thole=thole)
    field = np.zeros((len(coords), 3))
    field[:, axis] = 1.0
    return float((B @ field.reshape(-1)).reshape(-1, 3)[:, axis].sum())
