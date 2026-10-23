"""The quasiparticle branch as a frozen convention.

The reference geometry selects each state's root by the largest Z
(`Solvers.qp_equation.find_qp_root`); `ExcitedStateChain` records that root,
its pole strength and the labels of the poles bracketing it
(`FrozenConventions.qp_seeds`, `qp_seed_weights`, `qp_seed_brackets`). Every
later geometry solves the equation in full and keeps the root continuing the
one the last ACCEPTED geometry kept (`qp_follow`, the reference's until a step
is committed; `qp_space_time.followed_branch`), whatever its weight. A
displaced geometry that is not committed (a trial step, a finite difference)
changes none of it. Re-deciding by weight at every geometry would step the
surface where a satellite crosses the quasiparticle.

The surface: water in a continuum on a BLYP hybrid with 22 % exact exchange
(`XC`), whose HOMO-1 quasiparticle root skims eps_(HOMO-2): 8.1e-4 Ha away at
the reference, 1.8e-3 at dz = -4e-3, 3.0e-4 at +2e-3 and 2.1e-4 at +4e-3; it
crosses it between +4 and +8 mBohr. The self-energy is continuous there
(`cd_integral_weights`), so the crossing is smooth. The retired pole guard
held the Newton 1e-3 Ha off that level and had to be frozen too; the
bracketed search needs no such band.
"""
import warnings

import pytest
from pyscf import dft, gto

from src.Base.numerics.constants import QP_WINDOW_Z_MIN
from src.environments.solvent_screening import SolventScreening
from src.gradients.evaluation import geometry_hash
from src.gradients.gwbse.rpa_bse_surface import RPABSESurface
from tests.isdf_test_grid import g1_numerics

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: A BLYP hybrid with 22 % exact exchange: the fraction that puts the HOMO-1
#: root of the solvated surface within 1e-3 Ha of eps_(HOMO-2).
XC = '0.22*HF + 0.78*B88, LYP'


def ks_factory(mol):
    """The reference of the solvated gate: `XC`, converged for gradient work."""
    mf = dft.RKS(mol, xc=XC).density_fit(auxbasis=BASIS + '-ri')
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
    """(surface, E at the reference) with the branch already frozen there:
    every test below reads or displaces ONE frozen surface, which is what an
    optimizer holds."""
    env = SolventScreening(water, eps=1.78, eps_static=78.39)
    s = RPABSESurface(water, ks_factory, environment=env,
                      numerics=g1_numerics(water))
    return s, s.total_energy(water)


def displaced(water, dz):
    """Water with the oxygen moved dz Bohr along z."""
    m = water.copy()
    c = water.atom_coords().copy()
    c[0, 2] += dz
    m.set_geom_(c, unit='Bohr')
    m.build(False, False)
    return m


def energy_at(surface, water, dz):
    """E of the frozen surface with the oxygen moved dz Bohr along z."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return surface.total_energy(displaced(water, dz))


def test_the_reference_freezes_the_root_and_weight_of_every_state(solvated):
    """One root and one pole strength per quasiparticle, resolved where the
    surface was frozen, and the branch each follows is that root until a
    step is accepted; the HOMO-1 beside eps_(HOMO-2) is a quasiparticle."""
    s, _ = solvated
    view = s.excited.resolved
    states = sorted(int(p) for p in view.qp_set)
    assert sorted(view.qp_seeds) == states
    assert sorted(view.qp_seed_weights) == states
    assert {p: w for p, (w, _) in view.qp_follow.items()} == dict(
        view.qp_seeds)
    homo = s.excited.nocc - 1
    assert view.qp_seed_weights[homo - 1] > QP_WINDOW_Z_MIN
    assert all(0.0 < z <= 1.0 for z in view.qp_seed_weights.values())


def test_a_displaced_geometry_follows_and_decides_nothing(solvated, water):
    """Two displacements, and the frozen roots, weights and followed
    branches come back untouched: a geometry nobody accepted is never what
    the next one follows."""
    s, _ = solvated
    seeds = dict(s.excited.resolved.qp_seeds)
    weights = dict(s.excited.resolved.qp_seed_weights)
    follow = dict(s.excited.resolved.qp_follow)
    for dz in (-4e-3, 4e-3):
        energy_at(s, water, dz)
    assert s.excited.resolved.qp_seeds == seeds
    assert s.excited.resolved.qp_seed_weights == weights
    assert dict(s.excited.resolved.qp_follow) == follow


def test_the_energy_scan_is_smooth_at_both_step_sizes(solvated, water):
    """Second differences of E along the oxygen z at h = 2e-3 and 4e-3 Bohr.

    A step in E' of size delta makes (E+ - 2E0 + E-)/h^2 diverge like delta/h,
    so the two step sizes would disagree by a factor of two per halving and
    without limit; a smooth surface gives nearly the same value at both,
    and a factor of three is the bound.
    """
    s, e0 = solvated
    e = {dz: energy_at(s, water, dz) for dz in (-4e-3, -2e-3, 2e-3, 4e-3)}
    d2 = {h: (e[h] - 2 * e0 + e[-h]) / h ** 2 for h in (2e-3, 4e-3)}
    ratio = d2[4e-3] / d2[2e-3]
    assert 1 / 3 < ratio < 3, (
        f'second differences {d2[2e-3]:.6e} (h=2e-3) and {d2[4e-3]:.6e} '
        f'(h=4e-3) differ by {ratio:.2f}: that is a step in the surface, not '
        f'its curvature')


def test_refreeze_re_derives_the_branch_at_the_new_geometry(solvated, water):
    """A frozen convention belongs to ITS geometry, so `refreeze` must not
    carry the reference roots forward: the refrozen surface selects its own
    by the largest Z, 8 mBohr out where the HOMO-1 has crossed
    eps_(HOMO-2)."""
    s, _ = solvated
    moved = water.copy()
    c = water.atom_coords().copy()
    c[0, 2] += 8e-3
    moved.set_geom_(c, unit='Bohr')
    moved.build(False, False)
    r = s.refreeze(moved)
    assert r.excited.resolved.qp_seeds == {}, 'the reference roots came along'
    assert r.excited.resolved.qp_seed_weights == {}
    r.total_energy(moved)
    homo = r.excited.nocc - 1
    assert sorted(r.excited.resolved.qp_seeds) == sorted(
        int(p) for p in r.excited.resolved.qp_set)
    homo_root = r.excited.resolved.qp_seeds[homo]
    assert homo_root != s.excited.resolved.qp_seeds[homo]


def test_an_accepted_step_is_what_the_next_geometry_follows(solvated,
                                                            water):
    """Committing a displaced geometry makes the roots its solve kept the
    branch every later geometry follows (`FrozenConventions.commit`); the
    reference roots and weights stay as they were frozen. Last in this file:
    it moves the module's surface."""
    s, _ = solvated
    seeds = dict(s.excited.resolved.qp_seeds)
    m = displaced(water, 2e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s.total_energy(m)
    held = dict(s.excited.conventions._held_branch[geometry_hash(m)])
    s.excited.commit(m)
    follow = s.excited.conventions.qp_follow
    assert follow == held and sorted(follow) == sorted(seeds)
    assert all(follow[p][0] != seeds[p] for p in seeds)
    assert dict(s.excited.resolved.qp_seeds) == seeds
