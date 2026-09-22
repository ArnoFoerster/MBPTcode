"""The contour-deformation pole guard as a FROZEN convention.

The quasiparticle Newton is held `QP_POLE_OFFSET` away from every orbital
energy, because at omega = eps_q the pole of G sits on the contour and the
imaginary-axis integrand collapses onto nu = 0. A root inside that band makes
the guard and the Newton step fight, and the iteration escapes by halving the
band -- a decision taken per call, and so per geometry, on a surface whose
every other discrete choice is fixed at the reference geometry. `qp_energy_cd`
now takes the band as an input a caller can PIN and reports the band actually
used; `qp_set_gradient`/`qp_gradient_space_time` pass one per state and return
them in `route_out['pole_offsets']`; `ExcitedStateChain` records what the first
solve resolved and hands it to every later geometry.

WHAT THE BAND DOES NOT DO IS MOVE THE ANSWER. A Newton that converges returns a
root of f free of the guard, so the band changes the path and not the root:
measured on the solvated surface below, bands spanning 1e-3 down to 1.2e-4 Ha
give the same total energy to 1e-14 Ha. What the pinning buys is that the PATH
-- and with it which branch the capture and satellite fallbacks pick, each a
different approximation from the self-consistent root -- is the same at every
geometry rather than re-chosen at each.

The second-difference gate below is therefore a smoothness FLOOR, not a
before/after: on B3LYP water in a continuum the surface is analytic through the
scan (second differences 0.3306, 0.3319, 0.3440 and 0.5204 Ha/Bohr^2 at
h = 5e-4, 1e-3, 2e-3 and 4e-3 -- converging, where a step in E' would diverge
like 1/h). Its enormous higher derivatives come from the HOMO's quasiparticle
root skimming the G pole at eps_(HOMO-1), 8.1e-4 Ha away at the reference and
4.0e-4 at dz = -4e-3; 8 mBohr out it crosses, and there the 64-point CD
quadrature loses the spike (3.6e-4 Ha against a 128-point grid) and the Newton
lands on a satellite with Z = -0.093. That is the limit the guard exists to
announce, and no choice of band removes it.
"""
import warnings

import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.constants import QP_POLE_OFFSET
from src.Base.solvent_screening import SolventScreening
from src.gradients.contour_deformation import qp_energy_cd
from src.gradients.qp_space_time import frozen_pole_offset
from src.gradients.rpa_bse_surface import RPABSESurface

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'

#: A flat imaginary-axis screening scales the quasiparticle shift: 2e-3 puts
#: the root 5.1e-4 Ha from eps_p, inside the default band and outside its floor.
EPS = np.array([-0.9, -0.6, -0.35, 0.15, 0.4, 0.8])
NOCC = 3
NU = np.array([0.05, 0.3, 1.0, 4.0])
WT = np.array([0.1, 0.3, 0.8, 3.0])


def solve(p=NOCC - 1, screening=2e-3, **kw):
    """(root, the band that was in force, the warnings) for a flat screening."""
    Bp = np.zeros((3, len(EPS)))            # a frontier state sweeps no residues
    wc = np.full((len(NU), len(EPS)), screening)
    guard = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w = qp_energy_cd(p, Bp, EPS, NOCC, NU, WT, wc=wc, guard_out=guard,
                         **kw)[0]
    return w, guard['pole_offset'], [str(c.message) for c in caught]


def ks_factory(mol):
    """The reference of the solvated gate: B3LYP, converged for gradient work."""
    mf = dft.RKS(mol, xc='b3lyp').density_fit(auxbasis=BASIS + '-ri')
    mf.grids.prune = None
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=H2O, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def solvated(water):
    """(surface, E at the reference) with the guard already frozen there.

    The band is resolved by the first quasiparticle solve, so the fixture
    performs that solve: every test below reads or displaces ONE frozen
    surface, which is what an optimizer holds.
    """
    env = SolventScreening(water, eps=1.78, eps_static=78.39)
    s = RPABSESurface(water, ks_factory, environment=env)
    return s, s.total_energy(water)


def energy_at(surface, water, dz, record=False):
    """E of the frozen surface with the oxygen moved dz Bohr along z."""
    m = water.copy()
    c = water.atom_coords().copy()
    c[0, 2] += dz
    m.set_geom_(c, unit='Bohr')
    m.build(False, False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        e = surface.total_energy(m)
    return (e, [str(w.message) for w in caught]) if record else e


# ------------------------------------------------------- the band as an input
def test_the_relaxed_band_is_reported_and_not_only_warned():
    """The default path is untouched -- it still halves the band to reach a root
    inside it -- and now says which band it ended on, which is exactly what a
    caller has to record to take the same path at the next geometry."""
    w, band, notes = solve()
    assert band == pytest.approx(QP_POLE_OFFSET / 2)
    assert any('relaxed to' in n for n in notes), notes
    assert abs(w - EPS[NOCC - 1]) > band


def test_a_pinned_band_is_the_one_used_and_reaches_the_same_root():
    """Pinning the band the reference geometry resolved costs nothing: the root
    is the same root, reached without relaxing anything and without a warning."""
    w_relaxed = solve()[0]
    w, band, notes = solve(pole_offset=QP_POLE_OFFSET / 2, relax_offset=False)
    assert band == pytest.approx(QP_POLE_OFFSET / 2)
    assert not notes, notes
    assert w == pytest.approx(w_relaxed, abs=1e-12)


def test_a_pinned_band_that_misses_the_root_says_so_and_still_answers():
    """A pinned band no root fits inside is not silently kept and not fatal: it
    warns that this geometry left the frozen path, then relaxes exactly as an
    unpinned band does, and reports the band it ended on."""
    w_relaxed = solve()[0]
    w, band, notes = solve(pole_offset=QP_POLE_OFFSET, relax_offset=False)
    assert any('frozen' in n and 'does not reach' in n for n in notes), notes
    assert band == pytest.approx(QP_POLE_OFFSET / 2)
    assert w == pytest.approx(w_relaxed, abs=1e-12)


def test_a_band_the_root_sits_outside_is_left_alone():
    """The common case must pay nothing for the rare one."""
    w, band, notes = solve(screening=5e-3)
    assert band == QP_POLE_OFFSET
    assert not notes, notes
    assert abs(w - EPS[NOCC - 1]) > QP_POLE_OFFSET


def test_a_frozen_band_reads_as_a_scalar_a_mapping_or_a_gap_in_one():
    """A scalar pins every state, a mapping pins the states it names, and an
    orbital it does not name keeps the default -- which is how the gas phase,
    where nothing is ever relaxed, stays on the default path."""
    assert frozen_pole_offset(None, 4) == (QP_POLE_OFFSET, True)
    assert frozen_pole_offset(QP_POLE_OFFSET / 2, 4) == (QP_POLE_OFFSET / 2,
                                                         False)
    assert frozen_pole_offset({4: QP_POLE_OFFSET / 2}, 4) == (
        QP_POLE_OFFSET / 2, False)
    assert frozen_pole_offset({3: QP_POLE_OFFSET / 2}, 4) == (QP_POLE_OFFSET,
                                                              True)
    assert frozen_pole_offset({}, 4) == (QP_POLE_OFFSET, True)


# --------------------------------------------- the band as a frozen convention
def test_the_reference_geometry_freezes_the_band_of_every_state(solvated):
    """One entry per quasiparticle, resolved where the surface was frozen.

    Eq. (18) raises this HOMO by 1.67 eV and lands its root 8.1e-4 Ha from
    eps_(HOMO-1), inside the default band, so its band is the halved one while
    every other state in the window keeps the default.
    """
    s, _ = solvated
    off = s.excited.pole_offsets
    assert sorted(off) == sorted(int(p) for p in s.excited.qp_set)
    homo = s.excited.nocc - 1
    assert off[homo] == pytest.approx(QP_POLE_OFFSET / 2)
    assert all(off[int(p)] == QP_POLE_OFFSET for p in s.excited.qp_set
               if int(p) != homo)


def test_a_displaced_geometry_uses_the_frozen_band_and_decides_nothing(solvated,
                                                                       water):
    """Two displacements, and the recorded bands come back untouched: the
    surface an optimizer walks is the one frozen at its reference geometry."""
    s, _ = solvated
    frozen = dict(s.excited.pole_offsets)
    for dz in (-4e-3, 4e-3):
        energy_at(s, water, dz)
    assert s.excited.pole_offsets == frozen


def test_the_frozen_band_cannot_move_the_surface(solvated, water):
    """The band pins the PATH, not the root: a converged Newton returns the
    root of f whatever band it walked around.

    Measured by re-running two displaced geometries with the HOMO's band pinned
    four times smaller than the 5e-4 Ha the reference resolved, and once with a
    band ten times the default -- so wide that no root fits inside it, which is
    the fallback branch of `test_a_pinned_band_that_misses_the_root...` on the
    real surface. All three agree with the frozen surface to 1e-10 Ha.
    """
    s, _ = solvated
    frozen = dict(s.excited.pole_offsets)
    homo = s.excited.nocc - 1
    before = {dz: energy_at(s, water, dz) for dz in (-4e-3, 2e-3)}
    try:
        s.excited.pole_offsets[homo] = frozen[homo] / 4
        narrow = {dz: energy_at(s, water, dz) for dz in (-4e-3, 2e-3)}
        s.excited.pole_offsets[homo] = 10 * QP_POLE_OFFSET
        wide, notes = energy_at(s, water, 2e-3, record=True)
    finally:
        s.excited.pole_offsets.clear()
        s.excited.pole_offsets.update(frozen)
    for dz, e in before.items():
        assert narrow[dz] == pytest.approx(e, abs=1e-10), (dz, e, narrow[dz])
    assert wide == pytest.approx(before[2e-3], abs=1e-10)
    # the wide band was really used, and really reported leaving itself
    assert any('frozen' in n and 'does not reach' in n for n in notes), notes


def test_the_energy_scan_is_smooth_at_both_step_sizes(solvated, water):
    """Second differences of E along the oxygen z at h = 2e-3 and 4e-3 Bohr.

    A step in E' of size delta makes (E+ - 2E0 + E-)/h^2 diverge like delta/h,
    so the two step sizes would disagree by a factor of two per halving and
    without limit. They come out 0.3440 and 0.5204 Ha/Bohr^2: a factor 1.51,
    which is the near-pole curvature of this surface (E'''' ~ 2e4) and not a
    step -- the same second difference is 0.3319 at h = 1e-3 and 0.3306 at
    5e-4, i.e. converging. A factor of three is loose against 1.51 and tight
    against the 1/h a step would give at these steps.
    """
    s, e0 = solvated
    e = {dz: energy_at(s, water, dz) for dz in (-4e-3, -2e-3, 2e-3, 4e-3)}
    d2 = {h: (e[h] - 2 * e0 + e[-h]) / h ** 2 for h in (2e-3, 4e-3)}
    ratio = d2[4e-3] / d2[2e-3]
    assert 1 / 3 < ratio < 3, (
        f'second differences {d2[2e-3]:.6e} (h=2e-3) and {d2[4e-3]:.6e} '
        f'(h=4e-3) differ by {ratio:.2f}: that is a step in the surface, not '
        f'its curvature')


def test_refreeze_re_derives_the_band_at_the_new_geometry(solvated, water):
    """A frozen convention belongs to ITS geometry, so `refreeze` must not
    carry the reference band forward.

    8 mBohr out the HOMO's root sits 1.6e-3 Ha from eps_(HOMO-1), outside the
    default band, so the refrozen surface resolves the default there where the
    reference resolved half of it -- carrying the old value would have pinned a
    band this geometry has no reason to use.
    """
    s, _ = solvated
    moved = water.copy()
    c = water.atom_coords().copy()
    c[0, 2] += 8e-3
    moved.set_geom_(c, unit='Bohr')
    moved.build(False, False)
    r = s.refreeze(moved)
    assert r.excited.pole_offsets == {}, 'the reference band was carried over'
    r.total_energy(moved)
    homo = r.excited.nocc - 1
    assert sorted(r.excited.pole_offsets) == sorted(int(p) for p
                                                    in r.excited.qp_set)
    assert r.excited.pole_offsets[homo] == QP_POLE_OFFSET
    assert s.excited.pole_offsets[homo] < QP_POLE_OFFSET
