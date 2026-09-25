"""E_nu = E_HF + E_c^dRPA + Omega, and its analytic gradient.

Toelle/Kitsaras/Loos (arXiv:2507.02160) Eq. (14)-(15). `ExcitedStateChain`
returns `mf.e_tot + Omega`, which is the letter's E_0 only when the correlation
part of the plasmon formula vanishes -- i.e. under TDA screening. Everywhere
else it is missing E_c^dRPA, which on water/cc-pVDZ is 6.3 eV and, being a
functional of the geometry, moves the surface rather than its zero.

Every check below ASSERTS. A test that merely returns its verdict is discarded
by pytest and passes on False.
"""
import re

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.constants import CD_NFREQ, CD_POLE_RESOLUTION, HARTREE_TO_EV
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.GW.contour_deformation import root_pole_distance
from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
from src.gradients.isdf_derivatives import (exx_double_counting,
                                            exx_double_counting_Y,
                                            exx_double_counting_skeleton)
from src.gradients.rpa_ground_state import RPAGroundStateChain

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'


def hf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


def ks_factory(mol):
    mf = dft.RKS(mol, xc='b3lyp').density_fit(auxbasis=BASIS + '-ri')
    mf.grids.prune = None
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=H2O, basis=BASIS, verbose=0)


def test_energy_is_the_plasmon_ground_state_plus_omega(water):
    """E_nu = E_0 + Omega, and E_0 carries E_c -- which the chain alone drops."""
    s = RPABSESurface(water, hf_factory)
    e_nu, e_0, e_hf, e_c, omega = s.energy(water)
    assert abs(e_nu - (e_0 + omega)) < 1e-12
    assert abs(e_0 - (e_hf + e_c)) < 1e-12
    assert abs(e_hf - s.ground.mf0.e_tot) < 1e-12, 'HF reference: E_0^HF is mf.e_tot'
    # the omitted term is not a rounding error
    assert e_c * HARTREE_TO_EV < -1.0, f'E_c^dRPA = {e_c * HARTREE_TO_EV:.3f} eV'
    chain_only = s.ground.mf0.e_tot + omega
    assert abs(chain_only - e_nu - (-e_c)) < 1e-12


def test_gradient_matches_finite_difference(water):
    """The gate. 4-point FD of the SAME frozen surface; one step is worthless."""
    s = RPABSESurface(water, hf_factory)
    g, _, _ = s.total_gradient(water)
    x0 = water.atom_coords()

    def energy_at(ia, k, dx):
        m = water.copy()
        c = x0.copy()
        c[ia, k] += dx
        m.set_geom_(c, unit='Bohr')
        m.build(False, False)
        return s.total_energy(m)

    worst = 0.0
    for ia, k in ((0, 2), (1, 1)):
        for h in (2e-3, 4e-3):
            f = [energy_at(ia, k, d * h) for d in (1, -1, 2, -2)]
            fd = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)
            worst = max(worst, abs(fd - g[ia, k]))
    assert worst < 1e-7, f'worst |analytic - FD| = {worst:.2e} Ha/Bohr'


def test_a_kohn_sham_reference_reaches_the_same_functional(water):
    """E_0 is E_HF + E_c^dRPA from ANY starting point.

    E_KS + E_c would double count: E_xc already carries a correlation piece and
    an approximate exchange. The chain adds E_x^exact - E_xc, so the KS
    reference lands on the same functional the HF one does -- not the same
    NUMBER, since the two are evaluated at different densities, but the same
    energy expression, and E_HF[rho] must be far from E_KS.
    """
    chain = RPAGroundStateChain(water, ks_factory)
    mf = chain.mf0
    _, e_hf, e_c = chain.energy(water)
    assert chain.is_kohn_sham(mf)
    assert abs(e_hf - mf.e_tot) * HARTREE_TO_EV > 1.0, 'E_HF[rho_KS] is not E_KS'
    # E_HF[rho_KS] must be an HF-shaped energy: within ~0.1 Ha of the HF minimum,
    # and ABOVE it, since HF orbitals are the ones that minimize it.
    hf_min = RPAGroundStateChain(water, hf_factory).mf0.e_tot
    assert hf_min - 1e-9 <= e_hf <= hf_min + 0.1, (e_hf, hf_min)
    assert e_c < 0.0


@pytest.mark.parametrize('factory', [hf_factory, ks_factory],
                         ids=['hf', 'b3lyp'])
def test_ground_state_gradient_from_any_starting_point(water, factory):
    """The gate that makes the starting point free: dE_0/dR against a 4-point
    FD, on Hartree-Fock and on a HYBRID -- a pure functional would hide an
    a_x bookkeeping error, so the hybrid is the one that matters."""
    chain = RPAGroundStateChain(water, factory)
    g, _, _ = chain.total_gradient(water)
    x0 = water.atom_coords()

    def energy_at(ia, k, dx):
        m = water.copy()
        c = x0.copy()
        c[ia, k] += dx
        m.set_geom_(c, unit='Bohr')
        m.build(False, False)
        return chain.total_energy(m)

    worst = 0.0
    for ia, k in ((0, 2), (1, 1)):
        f = [energy_at(ia, k, d * 3e-3) for d in (1, -1, 2, -2)]
        fd = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * 3e-3)
        worst = max(worst, abs(fd - g[ia, k]))
    assert worst < 1e-6, f'worst |analytic - FD| = {worst:.2e} Ha/Bohr'


@pytest.mark.parametrize('xc', ['b3lyp', 'pbe0', 'pbe'])
def test_exx_double_counting_energy_is_exact(water, xc):
    """E_KS + (E_x^exact - E_xc) == E_HF[rho], the identity the whole scheme
    rests on. Hybrids included, where E_xc already carries a_x E_x."""
    mf = dft.RKS(water, xc=xc)
    mf.grids.prune = None
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    e_hf = scf.RHF(water).energy_tot(dm=mf.make_rdm1())
    assert abs(mf.e_tot + exx_double_counting(mf) - e_hf) < 1e-10


def test_exx_double_counting_vanishes_on_hartree_fock(water):
    """It must, and its gradient pieces with it -- otherwise every HF number
    in the repo would move."""
    mf = hf_factory(water)
    assert abs(exx_double_counting(mf)) < 1e-12
    assert np.abs(exx_double_counting_skeleton(mf)).max() < 1e-12
    assert np.abs(exx_double_counting_Y(mf, water.nelectron // 2)).max() < 1e-12


@pytest.mark.parametrize('surface_cls', [RPABSESurface, RPAQPSurface],
                         ids=['bse', 'qp'])
def test_both_halves_share_one_factorization(water, surface_cls):
    """A composed surface is ONE functional, so it gets ONE factorization.

    `separable_factors` is deterministic given its settings, so two chains
    handed matching keywords already agree bitwise and merely pay twice --
    measured, at a DISPLACED geometry, not the reference where they agree by
    construction. Holding one object makes that structural instead of a
    property of this constructor remembering to pass matching keywords.
    """
    s = surface_cls(water, hf_factory, state=0)
    assert s.ground.factorization is s.excited.factorization


@pytest.mark.parametrize('surface_cls', [RPABSESurface, RPAQPSurface],
                         ids=['bse', 'qp'])
def test_refreeze_rebuilds_one_shared_factorization(water, surface_cls):
    """Refreezing must build a NEW factorization and give it to BOTH halves.

    New, because a factorization is frozen AT a geometry and carrying the old
    one keeps the old radii, points and frames -- the discontinuity refreeze
    exists to remove. Shared, because otherwise the surface reverts to two
    factorizations after the first refreeze and the sharing quietly stops
    holding for the rest of the optimization, which is a regression no energy
    or gradient gate would show.
    """
    s = surface_cls(water, hf_factory, state=0)
    moved = water.copy()
    c = water.atom_coords().copy()
    c[1, 1] += 0.05
    moved.set_geom_(c, unit='Bohr')
    moved.build(False, False)
    r = s.refreeze(moved)

    assert r.ground.factorization is r.excited.factorization, 'halves diverged'
    assert r.ground.factorization is not s.ground.factorization, 'carried the old one'
    # frozen at the NEW geometry: its points are not the old object's there
    assert not np.array_equal(s.ground.factorization.coords(moved),
                              r.ground.factorization.coords(moved))


def test_the_ground_state_relaxes_in_the_static_field(water):
    """`mean_field` applies PCM(eps_static); the caller does not get a choice."""
    env = SolventScreening(water, eps=1.78, eps_static=78.39)
    mf = env.mean_field(water, ks_factory)
    assert hasattr(mf, 'with_solvent'), 'the ground state came back bare'
    assert abs(mf.with_solvent.eps - 78.39) < 1e-12
    # the reaction field is a real stabilization, not a gauge shift
    assert mf.e_tot < ks_factory(water).e_tot - 1e-4
    # and it is applied ONCE: a mean field that already carries one is kept
    assert env.mean_field(water, lambda m: mf) is mf


def test_solvated_gradient_matches_finite_difference(water):
    """The force must answer the WHOLE energy, ground-state reaction field
    included -- through the mean field's own force, through the exchange
    substitution, and through the relaxed density, which owes V_PCM a skeleton
    term of the same kind as the Coulomb one. Leaving any of the three out is
    stationary for neither functional: 4.1e-03, 4.2e-04 and 4.2e-04 Ha/Bohr
    respectively, against a surface whose forces are 2e-02."""
    env = SolventScreening(water, eps=1.78, eps_static=78.39)
    s = RPAGroundStateChain(water, ks_factory, environment=env)
    g, _, _ = s.total_gradient(water)
    x0 = water.atom_coords()

    def energy_at(ia, k, dx):
        m = water.copy()
        c = x0.copy()
        c[ia, k] += dx
        m.set_geom_(c, unit='Bohr')
        m.build(False, False)
        return s.total_energy(m)

    worst = 0.0
    for ia, k in ((0, 2), (1, 1)):
        h = 4e-3
        f = [energy_at(ia, k, d * h) for d in (1, -1, 2, -2)]
        fd = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)
        worst = max(worst, abs(fd - g[ia, k]))
    assert worst < 1e-6, f'worst |analytic - FD| = {worst:.2e} Ha/Bohr'


def test_solvated_excited_surface_gradient_matches_finite_difference(water):
    """The same three reaction-field entries, on E_0^dRPA + Omega, plus Eq. (18).

    `excitation_gradient` folds through the same `nuclear_gradient` as the
    ground-state chain, so the Fock skeleton's continuum term reaches the
    excited surface too; this asserts it rather than inferring it. The
    reaction field's own quasiparticle shift rides the factors instead, its
    adjoint on (eps, X, D) alongside the self-energy's.

    THE STEP IS THE ORDINARY ONE AND THE QUADRATURE IS WHAT MAKES IT SO.
    Eq. (18) raises this HOMO by 1.67 eV, which brings its quasiparticle root
    to 8.1e-4 Ha of eps_(HOMO-1), where the pole of G puts a Lorentzian of that
    half-width on the imaginary-frequency integrand at nu = 0. A 64-point
    Gauss-Legendre grid starts at 6.2e-5 Ha and steps across it, and the
    residual then behaves like nothing: 1.2e-4 Ha/Bohr at h = 2e-3 against
    3.0e-7 at 5e-4. `_grow_cd_grid` measures that distance on the frozen
    quasiparticle set and doubles the grid to 128 points, whose first frequency
    is 1.6e-5 Ha, and the residual becomes flat in h -- 5.8e-8 at 2e-3, 8.1e-8
    at 1e-3, 7.8e-8 at 5e-4 -- which is the fit adjoint's own floor.
    """
    env = SolventScreening(water, eps=1.78, eps_static=78.39)
    s = RPABSESurface(water, ks_factory, environment=env)
    g, _, _ = s.total_gradient(water)
    x0 = water.atom_coords()

    def energy_at(dx):
        m = water.copy()
        c = x0.copy()
        c[0, 2] += dx
        m.set_geom_(c, unit='Bohr')
        m.build(False, False)
        return s.total_energy(m)

    assert s.excited.nfreq_cd == 128, 'the CD grid was not grown to the pole'
    h = 2e-3
    f = [energy_at(d * h) for d in (1, -1, 2, -2)]
    fd = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)
    assert abs(fd - g[0, 2]) < 1e-6, \
        f'|analytic - FD| = {abs(fd - g[0, 2]):.2e} Ha/Bohr'


def test_the_cd_grid_is_sized_from_the_root_to_pole_distance(water):
    """The quadrature grows only where a root sits close to a pole of G.

    Gas phase: the water/B3LYP quasiparticle roots stay 1.2e-2 Ha from their
    nearest neighbouring orbital energy, 204 times the 64-point grid's first
    frequency, so nothing is resized and every gas-phase number is the one it
    was. In the continuum Eq. (18) closes that to 8.1e-4 Ha, 13 times the first
    frequency, and the grid doubles once.
    """
    gas = RPABSESurface(water, ks_factory)
    gas.total_energy(water)
    assert gas.excited.cd_sized
    assert gas.excited.nfreq_cd == CD_NFREQ, 'the gas phase was resized'

    env = SolventScreening(water, eps=1.78, eps_static=78.39)
    s = RPABSESurface(water, ks_factory, environment=env)
    s.total_energy(water)
    assert s.excited.nfreq_cd == 2 * CD_NFREQ
    assert len(s.excited.nu) == 2 * CD_NFREQ
    assert s.excited.gw_grid.cosft_wt.shape[0] == 2 * CD_NFREQ, \
        'the imaginary-time grid kept the old frequency axis'
    # and the rule it was sized by holds at the frozen roots
    eps = np.asarray(s.excited.mf0.mo_energy, float)
    d = root_pole_distance(eps, [s.excited.qp_seeds[int(p)]
                                 for p in s.excited.qp_set], s.excited.qp_set)
    assert s.excited.nu.min() * CD_POLE_RESOLUTION <= d


def test_the_solvated_surface_has_no_step_over_the_stencil(water):
    """Second differences of the solvated surface, which is where the step was.

    The energy is sampled every 1 mBohr from -10 to +10 and differenced twice,
    so a kink shows as a spike rather than as a slope. Over the span the 2 mBohr
    stencil reaches, the curvature is flat to 1.3 % of its 0.406 Ha/Bohr^2; on
    the 64-point grid it ran to 3.75 and 2.25 Ha/Bohr^2 at -4 and -3 mBohr,
    which is what put 1.2e-4 Ha/Bohr into a finite difference that reads those
    points.

    A CROSSING SURVIVES FURTHER OUT and no quadrature removes it: 8 mBohr along
    -z the HOMO root sits 6e-6 Ha from eps_(HOMO-1), inside even the 128-point
    grid's first frequency of 1.6e-5 Ha, and the residue term and the integral
    term cancel exactly only in the unquadratured integral. What is left there
    is one point displaced by 0.9 meV -- a second difference of 1.8 meV, down
    from 21.6 on the 64-point grid.
    """
    env = SolventScreening(water, eps=1.78, eps_static=78.39)
    s = RPABSESurface(water, ks_factory, environment=env)
    x0 = water.atom_coords()

    def energy_at(dx):
        m = water.copy()
        c = x0.copy()
        c[0, 2] += dx
        m.set_geom_(c, unit='Bohr')
        m.build(False, False)
        return s.total_energy(m)

    h = 1e-3
    e = np.array([energy_at(k * h) for k in range(-10, 11)])
    d2 = (e[2:] - 2 * e[1:-1] + e[:-2]) / h ** 2
    med = np.median(d2)
    stencil = d2[5:14]                      # the second differences at |dz| <= 4 mBohr
    assert np.abs(stencil - med).max() < 0.02 * abs(med), \
        f'over the stencil {np.array2string(stencil, precision=3)}'
    # what the worst second difference anywhere in the scan is worth in energy
    kink = float(np.abs(d2 - med).max()) * h ** 2 * HARTREE_TO_EV
    assert kink < 3e-3, (f'a {kink * 1e3:.2f} meV step at the pole crossing: '
                         f'{np.array2string(d2, precision=2)}')


@pytest.mark.parametrize('surface_cls', (RPABSESurface, RPAQPSurface))
def test_the_docstring_warns_off_an_absolute_solvated_total_energy(surface_cls):
    """A surface that composes a dRPA ground state must say what its total
    energy is worth in a continuum, because nothing in the number says it.

    The exact block fold of the ACFDT log-determinant keeps the BARE
    interaction in the linear counter-term while the code dresses it, and the
    dressing throws away the leading solute-solvent dispersion term -- tens of
    kcal/mol, i.e. larger than the whole mean-field solvation energy. A reader
    who takes an absolute solvated total energy off these surfaces has to be
    told; a reader taking an excitation energy, a quasiparticle level or any
    fixed-geometry difference has to be told that those are untouched, or the
    warning costs more than it saves.

    The CLAIMS are asserted, not a sentence, so a rewording does not fail here.
    """
    doc = surface_cls.__doc__.lower()
    assert 'total energy' in doc, 'the warning must name what is untrustworthy'
    assert 'trustworth' in doc, 'and say that it is not'
    assert 'counter-term' in doc, 'the cause: the linear counter-term'
    assert 'bare' in doc and 'dispersion' in doc, 'and what dressing it costs'
    for spared in ('excitation', 'quasiparticle', 'fixed geometry'):
        assert spared in doc, f'the warning must spare {spared}'
    assert re.search(r'gas[- ]phase', doc), 'and spare the gas phase outright'
