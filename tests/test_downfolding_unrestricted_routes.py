"""The RPA downfolding of an unrestricted (UHF/UKS) reference on the
imagfrequency route against the casida route.

The mean fields are density-fitted UHF calculations, so the three-index
factors are the mean field's own fit, a real one with a fitting error: the
OH radical (93 auxiliary functions) and triplet CH2 (116) in 6-31G. casida
with df=True runs on the same factors, so the two routes solve the same
problem and differ only by what the route adds: the frequency quadrature of
the correlation energies and the continuation of Sigma^E. The spaces have
different, non-contiguous orbitals per spin, N^A_alpha != N^A_beta, unequal
cores, a channel without an active pair, and no active pair at all.

mode='imagfrequency' keeps the static polarizability P(0) = chi0_alpha(0) +
chi0_beta(0) of the whole system in the fitting basis, takes E_c from the
two-spin imaginary-axis energy and keeps its frequency grid, which covers
the transitions of both spins. The screening correction is P - P^A and one
Dyson step in the fitting basis, P^A(0) the static response of the active
pairs of both spins, evaluated between each spin's own active-pair factors:
the casida route's deletion and Cholesky solve on the same factors written
another way, so the three blocks agree per element to rounding. E_c^A[v_eff]
is the energy's quadrature on the energy's own grid with the bare response
of the active pairs of both spins side by side, so with every orbital
active the E_c pair cancels to rounding, also on a coarse grid where the
quadrature error is visible. Through the driver every energy term, H_eff
and the FCI energy follow casida.

With sigma, S^E of each spin is sampled on the line mu_s + i w, mu_s that
spin's own mid-gap, from the Lorentzian fit of the one W of both spins and
that spin's factors and orbital energies, continued by Pade and evaluated
per element; Sigma_bar^E, E_0 and the FCI energy follow casida at
srg_s = None to far below the 1e-4 target, for spaces whose active
orbitals lie inside both pole-free windows by more than the continuation
margin. Omega_min of the windows is the lowest root of the two-spin
problem, or above the dense limit its lower bound, the smallest gap.

mode='space-time' does the same on separable (ISDF) factors in imaginary
time: one collocation per spin, X_s = X_ao C_s, and one Coulomb factor D.
The mean fields there are density-fitted UHF calculations of OH and of
triplet CH2 in cc-pVDZ (6-31G has no validated ISDF grid), on the G1 grid.
The reference is the casida route on the SAME ISDF factors, built here as
the three-index objects B^s_Q,pq = sum_P D_PQ X^s_Pp X^s_Pq, so the two
differ by the route's quadratures alone and not by a fit. That reference
cannot see the ISDF fit itself: a poor fit of one spin's orbital products
would pass. So one test compares space-time with casida on the mean field's
ordinary density fit (cc-pvdz-ri), where the difference is the ISDF fit's
error, per spin block.
"""
import dataclasses
import re
import warnings
from functools import cache
import numpy as np
import pytest
from pyscf import gto, scf
from scipy.sparse.linalg import ArpackNoConvergence
from src.Base.factorization.sliced_factors import unpack_factors
from src.Base.numerical_flags import collecting
from src.Base.numerics.numerics import (Numerics, numerics_or_default,
                                        rpa_quadrature)
from src.Base.orbital_spaces import particle_hole_window, transition_range
from src.Base.utils.grids.analytic_continuation import pade_nodes
from src.Base.utils.grids.grids import gap_scaled_w0, gauss_legendre_grid
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.zero_body import Energies
from src.EffectiveHamiltonian.Downfolding.methods import rpa
from src.EffectiveHamiltonian.active_space import ActiveSpace, active_pair_mask
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import (
    imaginary_axis_energy_grid, rpa_correlation_energy_imaginary_axis)
from src.SingleReference.LinearResponse.space_time import (
    rpa_correlation_energy_space_time, static_screening_grid)
from src.EffectiveHamiltonian.solvers.solve import solve
from tests.downfolding_oracle import (CH2, OH, drpa_so,
                                      fitted_spin_orbital_integrals,
                                      sigma_env_uhf, spin_orbital_integrals,
                                      unrestricted, water)
from tests.gw_density_routes import fitted
from tests.gw_density_routes import water as gw_water
from tests.isdf_test_grid import g1_numerics

REFS = ('oh', 'ch2')
SCREENINGS = ('bare', 'crpa', 'crpa_ph')
# the fields of Energies that are records, not energies
RECORDS = ('diagnostics', 'numerical_flags')
MOLECULES = {'oh': (OH, 1), 'ch2': (CH2, 2)}
# (occupied, virtual) active orbitals of alpha, then of beta. OH has 5 alpha
# and 4 beta occupied orbitals of 11, CH2 5 and 3 of 13.
SPACES = {
    'oh': {
        'asymmetric': (([2, 4], [5, 7]), ([3], [4, 6, 7])),
        'unequal_cores': (([3, 4], [5, 6]), ([2, 3], [4, 5])),
        'no_beta_pair': (([3, 4], [5]), ([], [4, 5, 6])),
        'occupied_only': (([3, 4], []), ([2, 3], [])),
    },
    'ch2': {
        'asymmetric': (([2, 4], [5, 7]), ([2], [3, 5, 6])),
        'unequal_cores': (([3, 4], [5, 6]), ([1, 2], [3, 4])),
        'no_beta_pair': (([3, 4], [5]), ([], [3, 4, 5])),
        'occupied_only': (([3, 4], []), ([1, 2], [])),
    },
}


@cache
def _mean_field(ref):
    """A density-fitted UHF of the OH radical or of triplet CH2 in 6-31G."""
    atom, spin = MOLECULES[ref]
    return unrestricted(atom, spin, density_fit=True)


@cache
def _low(ref, mode):
    """The RPA low-level result of `ref` on `mode`; casida with df=True runs
    on the mean field's own fit, the three-index factors of imagfrequency."""
    return run_low_level(_mean_field(ref), method='rpa', mode=mode)


@cache
def _low_infinite_s(ref, mode):
    """`_low` with srg_s = None (s = inf: Eq. 42 on casida), which is what
    'srg' is on imagfrequency."""
    return run_low_level(_mean_field(ref), method='rpa', mode=mode,
                         numerics=Numerics(downfolding={'srg_s': None}))


@pytest.fixture(scope='module', autouse=True)
def _release_cached_mean_fields():
    """Empty the caches when the module is done (a mean field held at
    interpreter shutdown leaves pyscf's temporary chkfile behind)."""
    yield
    for cached in (_low_infinite_s, _low, _mean_field):
        cached.cache_clear()


def _spaces(ref, which):
    """The pair (alpha, beta) of ActiveSpace objects of SPACES[ref][which]."""
    mf = _mean_field(ref)
    nmo = mf.mo_coeff.shape[-1]
    nocc = [int(np.count_nonzero(occ)) for occ in mf.mo_occ]
    return tuple(ActiveSpace.from_indices(nmo, n, occ, virt)
                 for n, (occ, virt) in zip(nocc, SPACES[ref][which]))


@pytest.mark.parametrize('ref', REFS)
def test_full_rpa_is_the_imaginary_axis_energy(ref):
    """E_c of the low-level result is the two-spin imaginary-axis energy on
    the mean field's factors, and equals the casida route's plasmon formula
    on the same factors to the quadrature error. Kept for the active spaces:
    the energy's own frequency grid, over the transitions of both spins, and
    P(0), the sum of the two spins' static responses,
    -2 sum_ia B_ia B_ia^T / delta_ia each. Measured: E_c against casida
    4.8e-13 (OH) and 3.6e-13 (CH2) on the 20-point grid, P(0) 6.9e-16."""
    low, cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    assert low.unrestricted and low.mode == 'imagfrequency'
    assert low.omega is None and low.xpy is None and low.eri is None
    assert len(low.factors) == 2
    assert all(np.array_equal(x, y) for x, y in zip(low.factors, cas.factors))
    lr = LinearResponseSolver(low.eps, coeff_df=low.factors,
                              spin_mode='unrestricted')
    assert low.e_c_full == rpa_correlation_energy_imaginary_axis(lr, low.nocc)
    assert abs(low.e_c_full - cas.e_c_full) < 1e-9
    points, weights = imaginary_axis_energy_grid(low.eps, low.nocc)
    assert np.array_equal(low.cache['grid'][0], points)
    assert np.array_equal(low.cache['grid'][1], weights)
    chi0 = 0.0
    for b, eps, nocc in zip(low.factors, low.eps, low.nocc):
        b_ov = b[:, :nocc, nocc:].reshape(b.shape[0], -1)
        delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
        chi0 = chi0 - 2.0 * (b_ov / delta) @ b_ov.T
    assert low.cache['p0'].shape == (low.factors[0].shape[0],) * 2
    assert np.abs(low.cache['p0'] - chi0).max() < 1e-12


def _parts(ref, which):
    """The pair of ActiveSpaces of SPACES[ref][which]."""
    return tuple(space for space, eps in zip(
        _spaces(ref, which), _mean_field(ref).mo_energy))


def _all_active(ref):
    mf = _mean_field(ref)
    nmo = mf.mo_coeff.shape[-1]
    return tuple(ActiveSpace.from_indices(nmo, n, range(n), range(n, nmo))
                 for n in (int(np.count_nonzero(occ)) for occ in mf.mo_occ))


@pytest.mark.parametrize('which', sorted(SPACES['oh']))
@pytest.mark.parametrize('ref', REFS)
def test_vtilde_matches_casida(ref, which):
    """P - P^A and one Dyson step give the casida route's correction per
    element in all three blocks, for crpa and crpa_ph, also with a channel
    without an active pair and with no active pair at all (nothing removed:
    the fully screened W(0) - v); 'bare' is zero. Measured over four spaces:
    1.9e-16 (OH), 1.0e-16 (CH2), on blocks of at least 0.05."""
    low_if, low_cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    parts = _parts(ref, which)
    for screening in ('crpa', 'crpa_ph'):
        vt = rpa.screening_correction(low_if, parts, screening)
        want = rpa.screening_correction(low_cas, parts, screening)
        assert isinstance(vt, tuple) and len(vt) == 3
        for ours, casida in zip(vt, want):
            assert np.abs(ours - casida).max() < 1e-8
        if screening == 'crpa':
            assert min(np.abs(block).max() for block in want) > 1e-2
            assert np.abs(want[0] - want[2]).max() > 1e-3
        terms = rpa.downfold(low_if, parts, screening, diagnostics=())
        assert all(np.array_equal(x, y) for x, y in zip(terms.vtilde, vt))
    bare = rpa.screening_correction(low_if, parts, 'bare')
    assert len(bare) == 3 and not any(block.any() for block in bare)


@pytest.mark.parametrize('ref', REFS)
def test_active_pair_factors_and_polarizability(ref):
    """The factors of every active pair of each spin, and P^A(0), the static
    response of the active particle-hole pairs of both spins,
    -2 sum_ia B_ia B_ia^T / delta_ia over each spin's active pairs: the
    other spin's alone when a channel has none, zero when neither has one,
    and P(0) itself when every orbital is active (then vtilde is zero)."""
    low = _low(ref, 'imagfrequency')
    naux = low.factors[0].shape[0]

    def response(parts):
        out = np.zeros((naux, naux))
        for b, eps, nocc, active_space in zip(low.factors, low.eps, low.nocc,
                                              parts):
            mask = active_pair_mask(active_space)
            b_a = b[:, :nocc, nocc:].reshape(naux, -1)[:, mask]
            delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()[mask]
            out -= 2.0 * (b_a / delta) @ b_a.T
        return out

    parts = _parts(ref, 'asymmetric')
    b_act = rpa.active_pair_factors(low, parts)
    assert isinstance(b_act, tuple) and len(b_act) == 2
    for ours, b, active_space in zip(b_act, low.factors, parts):
        act = active_space.act
        assert np.array_equal(ours, b[:, act][:, :, act])
    assert not np.array_equal(b_act[0], b_act[1])
    for which in ('asymmetric', 'unequal_cores', 'no_beta_pair'):
        parts = _parts(ref, which)
        p_a = rpa.active_polarizability_static(low, parts)
        assert np.abs(p_a - response(parts)).max() < 1e-13
        assert np.abs(p_a).max() > 1e-3
    none = rpa.active_polarizability_static(low, _parts(ref, 'occupied_only'))
    assert none.shape == (naux, naux) and not none.any()
    mf = _mean_field(ref)
    every = tuple(space for space, eps in zip(
        _all_active(ref), mf.mo_energy))
    assert np.abs(rpa.active_polarizability_static(low, every)
                  - low.cache['p0']).max() < 1e-13
    vt = rpa.screening_correction(low, every, 'crpa')
    assert len(vt) == 3 and not any(block.any() for block in vt)


def _solve(g_eff, space):
    return solve('pyscf_fci', g_eff, space['n_orbs'], space['n_elec_active'],
                 spin_free=space['spin_free'])[0]


@pytest.mark.parametrize('screening', SCREENINGS)
@pytest.mark.parametrize('ref', REFS)
def test_all_active_ec_pair_cancels(ref, screening):
    """Every orbital of both spins active: Pi^A U is the full chi0 v on the
    same grid, so E_c^A = E_c[v] and vtilde = 0. Measured: 7.3e-14 (OH,
    E_c = -0.1057), 1.1e-14 (CH2, E_c = -0.0768)."""
    low = _low(ref, 'imagfrequency')
    every = tuple(space for space, eps in zip(
        _all_active(ref), _mean_field(ref).mo_energy))
    terms = rpa.downfold(low, every, screening)
    assert terms.e_c_full < -1e-2
    assert abs(terms.e_c_full - terms.e_c_active) < 1e-10
    assert not any(block.any() for block in terms.vtilde)


@pytest.mark.parametrize('screening', SCREENINGS)
def test_all_active_ec_pair_cancels_on_a_coarse_grid(screening):
    """OH on the coarsest minimax grid (6 points), where the quadrature
    error of E_c[v] against casida is visible: the pair still cancels, which
    it does only if both energies are integrated on the one grid of both
    spins' transitions. Measured: the quadrature error of E_c is 2.7e-6,
    the pair cancels to 3.2e-15."""
    mf = _mean_field('oh')
    low = run_low_level(mf, method='rpa', mode='imagfrequency',
                        numerics=Numerics(quadrature={'nfreq_rpa': 6}))
    assert len(low.cache['grid'][0]) == 6
    assert abs(low.e_c_full - _low('oh', 'casida').e_c_full) > 1e-7
    every = tuple(space for space, eps in zip(
        _all_active('oh'), mf.mo_energy))
    terms = rpa.downfold(low, every, screening)
    assert abs(terms.e_c_full - terms.e_c_active) < 1e-12


@pytest.mark.parametrize('screening', SCREENINGS)
@pytest.mark.parametrize('which', sorted(SPACES['oh']))
@pytest.mark.parametrize('ref', REFS)
def test_e0_terms_match_casida(ref, which, screening):
    """Every Energies field, E_0, h_eff of both spins, the three blocks of
    v_eff and the FCI energy of H_eff agree between imagfrequency and
    casida on the same factors. The E_c pair differs by the quadrature
    error, the rest by rounding. Measured over four spaces and three
    screenings: every field, E_0 and the FCI energy 4.8e-13 (OH), 3.6e-13
    (CH2), all of it from E_c[v] (E_c^A 4.4e-14); h_eff 8.9e-16, v_eff
    2.2e-16."""
    spaces = _spaces(ref, which)
    ham = build_effective_hamiltonian(
        _low(ref, 'imagfrequency'), spaces, screening=screening, sigma=None)
    g, space, en = ham.g_eff, ham.space, ham.energies
    ham_c = build_effective_hamiltonian(
        _low(ref, 'casida'), spaces, screening=screening, sigma=None)
    g_c, en_c = ham_c.g_eff, ham_c.energies
    for field in dataclasses.fields(Energies):
        if field.name not in RECORDS:
            assert abs(getattr(en, field.name)
                       - getattr(en_c, field.name)) < 1e-8, field.name
    assert abs(en.e0 - en_c.e0) < 1e-8
    for ours, casida in zip(g[1] + g[2], g_c[1] + g_c[2]):
        assert np.abs(ours - casida).max(initial=0.0) < 1e-8
    assert abs(_solve(g, space) - _solve(g_c, space)) < 1e-8
    if which != 'occupied_only':
        assert en_c.e_c_active < -1e-5


@pytest.mark.parametrize('ref', REFS)
def test_static_defect_matches_casida(ref):
    """W(0) from the kept P(0) between each spin's active-pair factors
    against the active RPA of the fitted v + vtilde: zero for crpa, and the
    casida route's defect for bare and crpa_ph. Measured: the crpa defect
    4.4e-16; bare and crpa_ph against casida 7.2e-16."""
    low_if, low_cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    for which in ('asymmetric', 'no_beta_pair'):
        spaces = _spaces(ref, which)
        ham = build_effective_hamiltonian(low_if, spaces,
                                          screening='crpa')
        en = ham.energies
        assert ham.diagnostics['static_defect'] < 1e-10
        for screening in ('bare', 'crpa_ph'):
            ham = build_effective_hamiltonian(low_if, spaces,
                                              screening=screening)
            en = ham.energies
            ham_c = build_effective_hamiltonian(low_cas, spaces,
                                                screening=screening)
            en_c = ham_c.energies
            assert ham.diagnostics['static_defect'] > 1e-3
            assert abs(ham.diagnostics['static_defect']
                       - ham_c.diagnostics['static_defect']) < 1e-10


@pytest.mark.parametrize('screening', SCREENINGS)
@pytest.mark.parametrize('ref', REFS)
def test_ec_active_casida_diagnostic(ref, screening):
    """'ec_active_casida' records the dense two-spin Casida E_c^A on the
    same v_eff and the quadrature's difference from it; the dense value is
    the casida route's E_c^A. Measured |difference|: 4.4e-14 (OH), 3.2e-14
    (CH2). With the restricted spin factor 2 in the quadrature this test,
    the all-active one and the E_0 one fail (tried)."""
    spaces = _spaces(ref, 'asymmetric')
    ham = build_effective_hamiltonian(_low(ref, 'imagfrequency'), spaces,
                                      screening=screening, sigma=None)
    en = ham.energies
    record = ham.diagnostics['ec_active_casida']
    assert set(record) == {'casida', 'difference'}
    assert record['difference'] == en.e_c_active - record['casida']
    assert abs(record['difference']) < 1e-10
    ham_c = build_effective_hamiltonian(_low(ref, 'casida'), spaces,
                                        screening=screening, sigma=None)
    en_c = ham_c.energies
    assert abs(record['casida'] - en_c.e_c_active) < 1e-10
    assert set(ham.diagnostics) == {'static_defect', 'ec_active_casida'}
    ham = build_effective_hamiltonian(
        _low(ref, 'imagfrequency'), _spaces(ref, 'occupied_only'),
        screening=screening, sigma=None)
    en = ham.energies
    assert ham.diagnostics['ec_active_casida'] == {'casida': 0.0,
                                                  'difference': 0.0}


@pytest.mark.parametrize('ref', REFS)
def test_lowest_excitation_of_two_spins(ref):
    """Omega_min of the pole-free window on imagfrequency, which
    diagonalizes nothing: the lowest root of the two-spin problem on the
    stacked pairs of the route's factors, found without building it, i.e.
    the casida route's lowest Omega; the windows of both spins follow.
    Measured: 1.1e-16 and 4.2e-15 (0.662550 Ha for OH with 58 pairs,
    0.674525 for CH2 with 70). No Casida matrix is built for it."""
    low, cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    om = rpa.lowest_excitation(low)
    assert abs(om - cas.omega.min()) < 1e-10
    assert low.cache['omega_min'] == om
    parts = _parts(ref, 'unequal_cores')
    for ours, casida in zip(rpa.pole_window(low, parts),
                            rpa.pole_window(cas, parts)):
        assert np.abs(np.subtract(ours, casida)).max() < 1e-10
    fresh = run_low_level(_mean_field(ref), method='rpa',
                          mode='imagfrequency')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rpa, 'CasidaSolver', _refuse_a_casida_problem)
        assert rpa.lowest_excitation(fresh) == om


def _refuse_a_casida_problem(*args, **kwargs):
    raise AssertionError('a Casida problem was built for Omega_min')


def _no_convergence(*args, **kwargs):
    """Stands in for the Lanczos solver: the iteration gives up."""
    raise ArpackNoConvergence('ARPACK error -1: No convergence (stood in)',
                              np.zeros(0), np.zeros((0, 0)))


def test_gap_bound_when_the_lanczos_fails(monkeypatch):
    """When the iteration for Omega_min does not finish, Omega_min is
    replaced by the exact lower bound of the direct RPA, the smallest
    orbital-energy gap over both spins. The window is then narrower than the
    true one, so the near-pole warning can come too often, never too rarely.
    A SigmaNearPoleWarning says so once, and the named flag
    'sigma_pole_window_bound' is with every later use of the cached value.
    Measured for OH: the bound is the beta gap, 0.6295 Ha, 0.033 below the
    true Omega_min (CH2: the alpha gap, 0.015 below)."""
    monkeypatch.setattr(rpa, 'eigsh', _no_convergence)
    mf = _mean_field('oh')
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    gap = min(eps[nocc] - eps[nocc - 1]
              for eps, nocc in zip(mf.mo_energy, low.nocc))
    true = _low('oh', 'casida').omega.min()
    parts = _parts('oh', 'unequal_cores')
    raised = []
    with collecting(raised), pytest.warns(
            rpa.SigmaNearPoleWarning,
            match='did not finish.*smallest orbital-energy gap') as rec:
        windows = rpa.pole_window(low, parts)
    assert len(rec) == 1
    assert [entry['flag'] for entry in raised] == ['sigma_pole_window_bound']
    om = rpa.lowest_excitation(low)
    assert om == gap and 1e-3 < true - om < 0.1
    for (lo, hi), active_space, eps in zip(windows, parts, mf.mo_energy):
        assert lo == eps[active_space.env_occ].max() - gap
        assert hi == eps[active_space.env_virt].min() + gap
    again = []
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with collecting(again):
            assert rpa.pole_window(low, parts) == windows
    assert again == raised


# the spaces whose active orbitals are inside both pole-free windows by more
# than the continuation margin (0.26 Ha)
INSIDE = ('unequal_cores', 'no_beta_pair')


@pytest.mark.parametrize('ref', REFS)
def test_sigma_env_samples_against_the_pole_sum(ref):
    """S^E of each spin on the Pade nodes, on the line through that spin's
    own mid-gap: the Lorentzian fit of the one W of both spins against the
    oracle's exact pole sum in spin orbitals on the same fitted v. Measured
    at the first, middle and last of the 16 nodes: at most 1.3e-8, on
    samples of 4e-4 (the last node, w = 188) to 3e-2. The two spins have
    different mid-gaps (OH -0.173 and -0.189, CH2 -0.079 and -0.214), so
    near the real axis a sample on the other spin's line is off by 1.2e-4
    (OH) to 1.7e-3 (CH2). W is built on the first request and kept."""
    mf = _mean_field(ref)
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    parts = _parts(ref, 'unequal_cores')
    nmo, n = mf.mo_coeff.shape[-1], parts[0].n_act
    nodes = pade_nodes(numerics_or_default(low.numerics).quadrature)
    mus = [particle_hole_window(eps, nocc)[2]
           for eps, nocc in zip(low.eps, low.nocc)]
    assert abs(mus[0] - mus[1]) > 1e-2
    assert 'w_grid' not in low.cache
    got = rpa.environment_self_energy_samples(low, parts, nodes)
    assert isinstance(got, tuple) and len(got) == 2
    w_grid = low.cache['w_grid']
    assert w_grid[1].shape[1:] == (low.factors[0].shape[0],) * 2
    eri_fit = fitted_spin_orbital_integrals(mf)
    eps_so = np.concatenate(mf.mo_energy)
    occ_so = list(range(low.nocc[0])) + [nmo + i for i in range(low.nocc[1])]
    act_so = list(parts[0].act) + [nmo + q for q in parts[1].act]
    blocks = (slice(0, n), slice(n, 2 * n))
    for spin in (0, 1):
        assert got[spin].shape == (len(nodes), n, n)
        for k in (0, len(nodes) // 2, len(nodes) - 1):
            exact = sigma_env_uhf(mus[spin] + 1j * nodes[k], eri_fit, eps_so,
                                  occ_so, act_so)[blocks[spin], blocks[spin]]
            assert np.abs(exact).max() > 1e-4
            assert np.abs(got[spin][k] - exact).max() < 1e-6
        other = sigma_env_uhf(mus[1 - spin] + 1j * nodes[0], eri_fit, eps_so,
                              occ_so, act_so)[blocks[spin], blocks[spin]]
        assert np.abs(got[spin][0] - other).max() > 5e-5
    rpa.environment_self_energy_samples(low, parts, nodes)
    assert low.cache['w_grid'] is w_grid


@pytest.mark.parametrize('sigma', ('srg', 'averaged'))
@pytest.mark.parametrize('which', INSIDE)
@pytest.mark.parametrize('ref', REFS)
def test_sigma_bar_matches_casida(ref, which, sigma):
    """Sigma_bar^E of both spins on imagfrequency (the fit of W, Pade from
    each spin's line to z_tu or to the real axis) against casida on the same
    factors at srg_s = None, where 'srg' is Eq. 42 on both. Target 1e-4 Ha
    per element; measured 6.3e-7 and 9.5e-7 (OH), 5.7e-10 and 2.3e-8 (CH2),
    on blocks of 0.007 to 0.019 Ha whose alpha and beta differ by 0.006 to
    0.028. No near-pole warning: the active orbitals are at least 0.68 Ha
    inside both windows."""
    low_if = _low_infinite_s(ref, 'imagfrequency')
    low_cas = _low_infinite_s(ref, 'casida')
    parts = _parts(ref, which)
    with warnings.catch_warnings():
        warnings.simplefilter('error', rpa.SigmaNearPoleWarning)
        got = rpa.environment_self_energy(low_if, parts, sigma)
    want = rpa.environment_self_energy(low_cas, parts, sigma)
    assert isinstance(got, tuple) and len(got) == 2
    for ours, casida in zip(got, want):
        assert np.abs(casida).max() > 1e-3
        assert np.abs(ours - casida).max() < 1e-5
    assert np.abs(got[0] - got[1]).max() > 1e-3


@pytest.mark.parametrize('screening', SCREENINGS)
@pytest.mark.parametrize('ref', REFS)
def test_route_agreement_with_sigma(ref, screening):
    """With sigma='srg' (srg_s = None) every Energies field, E_0, h_eff of
    both spins and the FCI energy agree between imagfrequency and casida:
    the Sigma^E terms of E_0 and h_eff follow Sigma_bar^E. Target 1e-4;
    measured, OH and CH2: every field 3.2e-8 and 2.4e-10, h_eff 6.3e-7 and
    5.7e-10, FCI 3.3e-9 and 1.6e-12 (Tr[Sigma_bar gamma] at least 0.045)."""
    spaces = _spaces(ref, 'unequal_cores')
    ham = build_effective_hamiltonian(
        _low_infinite_s(ref, 'imagfrequency'), spaces, screening=screening,
        sigma='srg')
    g, space, en = ham.g_eff, ham.space, ham.energies
    ham_c = build_effective_hamiltonian(
        _low_infinite_s(ref, 'casida'), spaces, screening=screening,
        sigma='srg')
    g_c, en_c = ham_c.g_eff, ham_c.energies
    assert abs(en.e_sigma_trace) > 1e-3
    for field in dataclasses.fields(Energies):
        if field.name not in RECORDS:
            assert abs(getattr(en, field.name)
                       - getattr(en_c, field.name)) < 1e-5, field.name
    for ours, casida in zip(g[1], g_c[1]):
        assert np.abs(ours - casida).max() < 1e-5
    assert abs(_solve(g, space) - _solve(g_c, space)) < 1e-5


def test_sigma_records_of_two_spins():
    """The records of Sigma^E on imagfrequency for two spins: the pole-free
    window, the pole distances, Z and the continuation spread are pairs
    (alpha, beta); no damping is applied (s = inf). Measured: the spread is
    at most 2.8e-5 (alpha) and 9.9e-7 (beta), Z at least 0.990, the pole
    distances at least 0.68 and 1.28 Ha."""
    low = _low('oh', 'imagfrequency')
    spaces = _spaces('oh', 'unequal_cores')
    parts = _parts('oh', 'unequal_cores')
    n = parts[0].n_act
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ham = build_effective_hamiltonian(low, spaces, sigma='srg')
        en = ham.energies
    assert ham.numerical_flags == []
    assert set(ham.diagnostics) == {'static_defect', 'ec_active_casida',
                                   'sigma_poles', 'sigma_continuation'}
    poles = ham.diagnostics['sigma_poles']
    assert poles['window'] == rpa.pole_window(low, parts)
    assert poles['srg_s'] is None and poles['srg_damping'] == 0.0
    spread = ham.diagnostics['sigma_continuation']
    assert len(spread) == len(poles['z']) == len(poles['pole_distance']) == 2
    for s, z, dist in zip(spread, poles['z'], poles['pole_distance']):
        assert s.shape == (n, n) and z.shape == dist.shape == (n,)
        assert s.max() < 1e-4 and dist.min() > 0.26
        assert np.all((z > 0.9) & (z <= 1.0))
    for dist, (lo, hi), active_space, eps in zip(poles['pole_distance'],
                                         poles['window'], parts, low.eps):
        own = np.minimum(eps[active_space.act] - lo,
                         hi - eps[active_space.act])
        assert np.array_equal(dist, own)
    assert not np.array_equal(*poles['pole_distance'])


def test_near_pole_warning_on_imagfrequency_names_the_spin():
    """In the asymmetric OH space the beta orbitals 6 and 7 lie above the
    beta window (0.20 and 0.27 Ha outside): both forms warn on
    imagfrequency, naming their spin and pointing to casida, and Sigma_bar
    is computed all the same. The continuation check warns too, of the
    element (6, 6) of the beta spin, where the two Pade fits differ by
    9.0e-4 Ha. Both warnings point at the caller and both flags are in
    h.numerical_flags."""
    low = _low('oh', 'imagfrequency')
    spaces = _spaces('oh', 'asymmetric')
    for sigma in ('srg', 'averaged'):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            ham = build_effective_hamiltonian(low, spaces, sigma=sigma)
            g, en = ham.g_eff, ham.energies
        assert [w.category for w in rec] == [rpa.SigmaNearPoleWarning,
                                             rpa.SigmaContinuationWarning]
        assert all(w.filename == __file__ for w in rec)
        near, spread = (str(w.message) for w in rec)
        assert re.search(r'beta orbital 6 \(.*beta orbital 7 \(', near)
        assert "mode='casida'" in near and 'alpha orbital' not in near
        assert 'beta (6, 6)' in spread and 'alpha (' not in spread
        assert all(np.all(np.isfinite(h)) for h in g[1])
        assert len(ham.numerical_flags) == 2
        assert {'flag': 'sigma_near_pole',
                'message': near} in ham.numerical_flags
        assert {'flag': 'sigma_continuation_spread',
                'message': spread} in ham.numerical_flags


def test_the_bound_reaches_every_build(monkeypatch):
    """With Omega_min replaced by its bound, the flag is in the
    numerical_flags of every Hamiltonian built with Sigma^E on that
    low-level result, and the warning is issued by the first alone."""
    monkeypatch.setattr(rpa, 'eigsh', _no_convergence)
    low = run_low_level(_mean_field('oh'), method='rpa', mode='imagfrequency')
    spaces = _spaces('oh', 'unequal_cores')
    with pytest.warns(rpa.SigmaNearPoleWarning,
                      match='smallest orbital-energy gap'):
        first = build_effective_hamiltonian(low, spaces, sigma='srg')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        second = build_effective_hamiltonian(
            low, _spaces('oh', 'no_beta_pair'), sigma='srg')
    for ham in (first, second):
        assert [f['flag'] for f in ham.numerical_flags] == [
            'sigma_pole_window_bound']


@pytest.mark.parametrize('sigma', (None, 'srg', 'averaged'))
def test_imagfrequency_closed_shell_limit(sigma):
    """A closed-shell determinant on a density-fitted mean field, as RHF and
    as the same determinant in a UHF object: on imagfrequency the
    two-channel code gives the restricted H_eff, every energy term and the
    FCI energy (the same grids, since both spins have the one spectrum).
    Not compared: the pole-free window, narrower for two channels. Measured
    (116 auxiliary functions), with and without Sigma^E: E_c 0, v_eff 0,
    h_eff 2.2e-16, the energy terms 7.6e-15, FCI 1.4e-14."""
    rhf = water().density_fit().run()
    uhf = rhf.to_uhf()
    assert getattr(uhf, 'with_df', None) is not None
    low_r = run_low_level(rhf, method='rpa', mode='imagfrequency')
    low_u = run_low_level(uhf, method='rpa', mode='imagfrequency')
    assert abs(low_u.e_c_full - low_r.e_c_full) < 1e-10
    one = ActiveSpace.from_counts(rhf.mo_energy, 5, 2, 2)
    for screening in SCREENINGS:
        ham_r = build_effective_hamiltonian(low_r, one,
                                            screening=screening,
                                            sigma=sigma)
        g_r, en_r = ham_r.g_eff, ham_r.energies
        ham_u = build_effective_hamiltonian(
            low_u, (one, one), screening=screening, sigma=sigma)
        g_u, space, en_u = ham_u.g_eff, ham_u.space, ham_u.energies
        for h in g_u[1]:
            assert np.abs(h - g_r[1]).max() < 1e-8
        for v in g_u[2]:
            assert np.abs(v - g_r[2]).max() < 1e-10
        for field in dataclasses.fields(Energies):
            if field.name not in RECORDS:
                assert abs(getattr(en_u, field.name)
                           - getattr(en_r, field.name)) < 1e-8, field.name
        e_r = solve('pyscf_fci', g_r, 4, 4, spin_free=True)[0]
        assert abs(_solve(g_u, space) - e_r) < 1e-8


def test_mrpa_and_residues_stay_casida_only():
    low = _low('oh', 'imagfrequency')
    parts = _parts('oh', 'unequal_cores')
    with pytest.raises(NotImplementedError, match="mode='casida'"):
        rpa.screening_correction(low, parts, 'mrpa')
    with pytest.raises(NotImplementedError, match="mode='casida'"):
        rpa.screened_residues(low, parts)
    with pytest.raises(NotImplementedError, match='imagfrequency'):
        rpa.environment_self_energy_samples(_low('oh', 'casida'), parts, [0.5])


def test_numerical_routes_refuse_a_spin_without_occupied_or_virtual_orbitals():
    """Triplet H2 in 6-31G has both electrons in alpha orbitals: the beta
    channel has no occupied orbital, so no transition and no mid-gap. casida
    takes it, the RPA then holding the alpha pairs alone (E_c against the
    oracle's spin-orbital RPA on exact integrals, measured 4.4e-16 on
    E_c = -0.0081); the numerical routes refuse it, naming the spin and
    casida."""
    mf = unrestricted('H 0 0 0; H 0 0 0.74', 2)
    assert [int(np.count_nonzero(occ)) for occ in mf.mo_occ] == [2, 0]
    low = run_low_level(mf, method='rpa', df=False)
    assert low.nocc == (2, 0) and len(low.omega) == 4
    nmo = mf.mo_coeff.shape[-1]
    _, eri_so = spin_orbital_integrals(mf.mol, *mf.mo_coeff)
    e_c = drpa_so(eri_so, np.concatenate(mf.mo_energy), [0, 1])[0]
    assert e_c < -1e-3 and abs(low.e_c_full - e_c) < 1e-10
    for mode in ('imagfrequency', 'space-time'):
        with pytest.raises(NotImplementedError,
                           match="beta channel has 0 occupied of 4.*"
                                 "mode='casida'"):
            run_low_level(mf, method='rpa', mode=mode)


# --- mode='space-time': separable (ISDF) factors, imaginary time ---------------

@cache
def _st_mean_field(ref):
    """A density-fitted UHF (cc-pvdz-ri) of OH or of triplet CH2 in cc-pVDZ."""
    atom, spin = MOLECULES[ref]
    mol = gto.M(atom=atom, basis='cc-pvdz', spin=spin, cart=False, verbose=0)
    mf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


@cache
def _st_low(ref, quadrature=()):
    """The space-time low-level result of `ref` on the G1 grid, with the
    quadrature settings `quadrature` (pairs) and srg_s = None."""
    mf = _st_mean_field(ref)
    numerics = g1_numerics(mf, Numerics(quadrature=dict(quadrature),
                                        downfolding={'srg_s': None}))
    return run_low_level(mf, method='rpa', mode='space-time',
                         numerics=numerics)


def _isdf_three_index(low):
    """(B_alpha, B_beta), B^s_Q,pq = sum_P D_PQ X^s_Pp X^s_Pq over every
    orbital pair: the fitted v of the space-time result as three-index
    objects."""
    x, d = low.cache['factors']
    nmo = x.shape[-1]
    return tuple(
        (d.T @ (x_s[:, :, None] * x_s[:, None, :]).reshape(len(x_s), -1)
         ).reshape(d.shape[1], nmo, nmo) for x_s in x)


@cache
def _st_casida(ref):
    """A casida low-level result on the space-time result's own ISDF
    factors, solved here as the two-spin Casida problem: the same fitted v
    with explicit poles, at srg_s = None."""
    low = _st_low(ref)
    factors = _isdf_three_index(low)
    lr = LinearResponseSolver(low.eps, coeff_df=factors,
                              spin_mode='unrestricted')
    a, b = lr.build_casida_matrices(low.nocc, lBSE=False)
    trace = np.trace(a)
    omega, x_amp, y_amp = CasidaSolver(a, b).solve()
    return rpa.RPALowLevel(
        mf=low.mf, mode='casida', df=True, eps=low.eps, nocc=low.nocc,
        e_c_full=float(0.5 * (omega.sum() - trace)), omega=omega,
        xpy=x_amp + y_amp, factors=factors, eri=None,
        numerics=Numerics(downfolding={'srg_s': None}))


@pytest.fixture(scope='module', autouse=True)
def _release_space_time_caches():
    yield
    for cached in (_st_casida_on_the_density_fit, _st_casida, _st_low,
                   _st_mean_field):
        cached.cache_clear()


def _st_spaces(ref, which):
    """SPACES[ref][which] on the cc-pVDZ mean field (the same occupied
    counts as in 6-31G)."""
    mf = _st_mean_field(ref)
    nmo = mf.mo_coeff.shape[-1]
    nocc = [int(np.count_nonzero(occ)) for occ in mf.mo_occ]
    return tuple(ActiveSpace.from_indices(nmo, n, occ, virt)
                 for n, (occ, virt) in zip(nocc, SPACES[ref][which]))


def _st_parts(ref, which):
    return tuple(space for space, eps in zip(
        _st_spaces(ref, which), _st_mean_field(ref).mo_energy))


@pytest.mark.parametrize('ref', REFS)
def test_space_time_full_rpa(ref):
    """run(mode='space-time') on a UHF mean field keeps the ISDF factors of
    `separable_factors`, X (2, M, nmo) one collocation per spin and D
    (M, naux) shared; the static tau grid over the transitions of both
    spins with P(0) = chi0_alpha(0) + chi0_beta(0) on it, each spin's pairs
    with the grid's transform of -exp(-delta tau); and the energy's grid,
    minimax tau to Gauss-Legendre frequencies scaled to the smallest gap of
    the two spins. e_c_full is the two-spin space-time energy on it, and
    the plasmon value of the casida problem on the same factors to the
    quadrature error. Measured: OH 400 points, 70 auxiliary functions, 130
    pairs, grids 14 and 14 x 40; CH2 600, 84, 158, 12 and 12 x 40; P(0)
    2.2e-16; E_c against casida 1.0e-8 (OH), 1.0e-9 (CH2)."""
    low, cas = _st_low(ref), _st_casida(ref)
    mf = low.mf
    assert low.unrestricted and low.mode == 'space-time' and low.df is True
    assert low.omega is None and low.xpy is None
    assert low.factors is None and low.eri is None
    x, d = low.cache['factors']
    want = unpack_factors(separable_factors(mf, mf.mol, numerics=low.numerics,
                                            comm=None))
    assert x.shape == (2, d.shape[0], mf.mo_coeff.shape[-1])
    assert np.array_equal(x, want[0]) and np.array_equal(d, want[1])
    eps, nocc = low.eps, low.nocc
    e_min, e_max = transition_range(eps, nocc)
    q = low.numerics.quadrature
    static = low.cache['static_grid']
    ref_grid = static_screening_grid(eps, nocc, ntau=q.ntau_w,
                                     tau_target=q.tau_target)
    assert np.array_equal(static.tau_points, ref_grid.tau_points)
    assert np.array_equal(static.cosft_wt, ref_grid.cosft_wt)
    p0 = np.zeros((d.shape[1],) * 2)
    for x_s, eps_s, n in zip(x, eps, nocc):
        pairs = (x_s[:, :n, None] * x_s[:, None, n:]).reshape(len(x_s), -1)
        b_ov = d.T @ pairs
        delta = (eps_s[n:][None, :] - eps_s[:n][:, None]).ravel()
        pi0 = -np.exp(-np.outer(delta, static.tau_points)) @ static.cosft_wt[0]
        p0 += (b_ov * pi0) @ b_ov.T
    assert np.abs(low.cache['p0'] - p0).max() < 1e-12
    grid = low.cache['energy_grid']
    ntau, nfreq = rpa_quadrature(low.numerics)
    assert ntau is None and grid.ntau == static.ntau
    nu, wt = gauss_legendre_grid(nfreq, gap_scaled_w0(gap=e_min))
    assert np.array_equal(grid.omega_points, nu)
    assert np.array_equal(grid.omega_weights, wt)
    assert low.e_c_full == rpa_correlation_energy_space_time(
        x, d, eps, nocc, grid, comm=None)
    assert abs(low.e_c_full - cas.e_c_full) < 1e-6


def test_space_time_refusals_of_the_run():
    """The row-fit realization of the ISDF fit builds one collocation and
    cannot take an unrestricted reference: refused, naming the one that
    can. Without an ISDF grid in numerics, and with df=False, run refuses as
    for a restricted reference."""
    mf = _st_mean_field('oh')
    rows = g1_numerics(mf).derive('the row fit', isdf={'fit': 'rows'})
    with pytest.raises(NotImplementedError, match="fit='replicated'"):
        run_low_level(mf, method='rpa', mode='space-time', numerics=rows)
    with pytest.raises(ValueError, match='numerics.isdf'):
        run_low_level(mf, method='rpa', mode='space-time')
    with pytest.raises(ValueError, match="df=False.*mode='casida'"):
        run_low_level(mf, method='rpa', mode='space-time', df=False,
                      numerics=g1_numerics(mf))


def _st_all_active(ref):
    mf = _st_mean_field(ref)
    nmo = mf.mo_coeff.shape[-1]
    return tuple(ActiveSpace.from_indices(nmo, n, range(n), range(n, nmo))
        for n, eps in zip((int(np.count_nonzero(occ)) for occ in mf.mo_occ),
                          mf.mo_energy))


@pytest.mark.parametrize('ref', REFS)
def test_space_time_active_part_cancels(ref):
    """Every orbital of both spins active: P^A, built per spin on the stored
    static grid with the active columns of each spin's X and summed as
    P(0) was, is P(0) itself, so P - P^A leaves nothing to screen and
    vtilde is zero; E_c^A, with exp(-delta tau) of both spins' pairs pushed
    through the energy's own transform, is E_c[v]. Measured: |P - P^A| and
    vtilde exactly 0, the E_c pair 2.0e-13 (OH) and 2.6e-14 (CH2), the
    static defect 2.7e-15."""
    low = _st_low(ref)
    every = _st_all_active(ref)
    p_a = rpa.active_polarizability_static(low, every)
    assert np.abs(low.cache['p0'] - p_a).max() < 1e-12
    for screening in SCREENINGS:
        terms = rpa.downfold(low, every, screening)
        assert len(terms.vtilde) == 3
        assert max(np.abs(block).max() for block in terms.vtilde) < 1e-10
        assert abs(terms.e_c_full - terms.e_c_active) < 1e-10
        assert terms.diagnostics['static_defect'] < 1e-8


@pytest.mark.parametrize('screening', SCREENINGS)
def test_space_time_all_active_cancels_on_a_coarse_grid(screening):
    """The same on 10 tau points and 6 Gauss-Legendre frequencies, where
    E_c[v] is visibly off its value on the default grid: the pair still
    cancels, which needs the energy's own points, weights and transform on
    both sides. Measured: E_c moves by 3.3e-3, the pair cancels to 2.3e-14.
    With the restricted spin factor in the quadrature the all-active test
    and the E_0 test fail (tried)."""
    low = _st_low('oh', (('ntau_rpa', 10), ('nfreq_rpa', 6)))
    grid = low.cache['energy_grid']
    assert grid.ntau == 10 and grid.nfreq == 6
    assert abs(low.e_c_full - _st_low('oh').e_c_full) > 1e-7
    terms = rpa.downfold(low, _st_all_active('oh'), screening)
    assert abs(terms.e_c_full - terms.e_c_active) < 1e-10


@pytest.mark.parametrize('ref', REFS)
def test_space_time_active_pair_factors_and_polarizability(ref):
    """The factors of every active pair of each spin are D^T (X^s_t X^s_u),
    the slice of that spin's three-index object; P^A(0) is the static
    grid's transform of -sum_ia B_ia B_ia^T exp(-delta_ia tau) over the
    active pairs of each spin: the other spin's alone when a channel has
    none, zero when neither has one."""
    low = _st_low(ref)
    three = _isdf_three_index(low)
    naux = three[0].shape[0]
    static = low.cache['static_grid']

    def response(parts):
        out = np.zeros((naux, naux))
        for b, eps, nocc, active_space in zip(three, low.eps, low.nocc, parts):
            mask = active_pair_mask(active_space)
            b_a = b[:, :nocc, nocc:].reshape(naux, -1)[:, mask]
            delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()[mask]
            pi0 = -np.exp(-np.outer(delta, static.tau_points)) \
                @ static.cosft_wt[0]
            out += (b_a * pi0) @ b_a.T
        return out

    parts = _st_parts(ref, 'asymmetric')
    b_act = rpa.active_pair_factors(low, parts)
    assert isinstance(b_act, tuple) and len(b_act) == 2
    for ours, b, active_space in zip(b_act, three, parts):
        act = active_space.act
        assert np.abs(ours - b[:, act][:, :, act]).max() < 1e-12
    for which in ('asymmetric', 'unequal_cores', 'no_beta_pair'):
        parts = _st_parts(ref, which)
        p_a = rpa.active_polarizability_static(low, parts)
        assert np.abs(p_a - response(parts)).max() < 1e-12
        assert np.abs(p_a).max() > 1e-3
    none = rpa.active_polarizability_static(low,
                                            _st_parts(ref, 'occupied_only'))
    assert none.shape == (naux, naux) and not none.any()


@pytest.mark.parametrize('which', sorted(SPACES['oh']))
@pytest.mark.parametrize('ref', REFS)
def test_space_time_vtilde_matches_casida(ref, which):
    """vtilde in the three blocks against casida on the same ISDF factors.
    The two differ only by the static grid's transform of the pairs'
    response, -sum_tau w exp(-delta tau) in place of the exact -2/delta.
    Measured over four spaces: 5.9e-8 (OH), 3.9e-8 (CH2)."""
    low, cas = _st_low(ref), _st_casida(ref)
    parts = _st_parts(ref, which)
    for screening in ('crpa', 'crpa_ph'):
        vt = rpa.screening_correction(low, parts, screening)
        want = rpa.screening_correction(cas, parts, screening)
        assert isinstance(vt, tuple) and len(vt) == 3
        for ours, casida in zip(vt, want):
            assert np.abs(ours - casida).max() < 1e-6
        if screening == 'crpa':
            assert min(np.abs(block).max() for block in want) > 1e-2
    bare = rpa.screening_correction(low, parts, 'bare')
    assert len(bare) == 3 and not any(block.any() for block in bare)


@pytest.mark.parametrize('screening', SCREENINGS)
@pytest.mark.parametrize('which', sorted(SPACES['oh']))
@pytest.mark.parametrize('ref', REFS)
def test_space_time_e0_terms_match_casida(ref, which, screening):
    """Every Energies field, E_0, h_eff of both spins, the three blocks of
    v_eff and the FCI energy of H_eff agree between space-time and casida
    on the same ISDF factors to the route's quadratures; the dense-Casida
    E_c^A on the same v_eff is the quadrature's value. Measured over four
    spaces and three screenings, OH and CH2: every field 4.2e-7 and 1.8e-7
    (E_c[v] 1.0e-8 and 1.0e-9, E_c^A 7.9e-11 and 7.7e-10), h_eff 1.7e-7
    and 9.3e-8, v_eff 5.9e-8 and 3.9e-8, FCI 1.0e-8 and 1.1e-9, the
    quadrature against the dense E_c^A 3.7e-13, the crpa static defect
    7.8e-16, the other defects against casida 6.0e-8."""
    spaces = _st_spaces(ref, which)
    ham = build_effective_hamiltonian(_st_low(ref), spaces,
                                      screening=screening, sigma=None)
    g, space, en = ham.g_eff, ham.space, ham.energies
    ham_c = build_effective_hamiltonian(_st_casida(ref), spaces,
                                        screening=screening, sigma=None)
    g_c, en_c = ham_c.g_eff, ham_c.energies
    for field in dataclasses.fields(Energies):
        if field.name not in RECORDS:
            assert abs(getattr(en, field.name)
                       - getattr(en_c, field.name)) < 5e-6, field.name
    for ours, casida in zip(g[1] + g[2], g_c[1] + g_c[2]):
        assert np.abs(ours - casida).max(initial=0.0) < 5e-6
    assert abs(_solve(g, space) - _solve(g_c, space)) < 1e-6
    assert abs(ham.diagnostics['ec_active_casida']['difference']) < 1e-10
    if screening == 'crpa':
        assert ham.diagnostics['static_defect'] < 1e-10
    else:
        assert abs(ham.diagnostics['static_defect']
                   - ham_c.diagnostics['static_defect']) < 1e-6


@pytest.mark.parametrize('ref', REFS)
def test_space_time_lowest_excitation(ref):
    """Omega_min on space-time: the lowest root of the two-spin problem on
    the ISDF factors, applied through each spin's collocation, i.e. the
    lowest Omega of the casida reference on those factors. Measured:
    1.3e-15 and 2.1e-15 (0.667597 Ha for OH, 0.615249 for CH2)."""
    low, cas = _st_low(ref), _st_casida(ref)
    om = rpa.lowest_excitation(low)
    assert abs(om - cas.omega.min()) < 1e-10
    parts = _st_parts(ref, 'unequal_cores')
    for ours, casida in zip(rpa.pole_window(low, parts),
                            rpa.pole_window(cas, parts)):
        assert np.abs(np.subtract(ours, casida)).max() < 1e-10


@pytest.mark.parametrize('ref', REFS)
def test_space_time_sigma_env_samples_against_the_pole_sum(ref):
    """S^E of each spin on the Pade nodes, sampled on space-time from the
    fit of the route's own W (chi0 of both spins on the ISDF factors) with
    B^s_Q,tr = sum_P D_PQ X^s_Pt X^s_Pr, against the oracle's exact pole sum
    in spin orbitals on the same ISDF v, on each spin's own mid-gap line.
    Measured at the first, middle and last node: 2.9e-8 (OH), 1.0e-8
    (CH2)."""
    mf = _st_mean_field(ref)
    low = run_low_level(mf, method='rpa', mode='space-time',
                        numerics=g1_numerics(mf))
    parts = _st_parts(ref, 'unequal_cores')
    nmo, n = mf.mo_coeff.shape[-1], parts[0].n_act
    nodes = pade_nodes(numerics_or_default(low.numerics).quadrature)
    mus = [particle_hole_window(eps, nocc)[2]
           for eps, nocc in zip(low.eps, low.nocc)]
    assert 'w_grid' not in low.cache
    got = rpa.environment_self_energy_samples(low, parts, nodes)
    assert isinstance(got, tuple) and len(got) == 2
    three = _isdf_three_index(low)
    eri_so = np.zeros((2 * nmo,) * 4)
    for s in (0, 1):
        for t in (0, 1):
            eri_so[s * nmo:(s + 1) * nmo, s * nmo:(s + 1) * nmo,
                   t * nmo:(t + 1) * nmo, t * nmo:(t + 1) * nmo] = np.einsum(
                'Qpq,Qrs->pqrs', three[s], three[t], optimize=True)
    eps_so = np.concatenate(low.eps)
    occ_so = list(range(low.nocc[0])) + [nmo + i for i in range(low.nocc[1])]
    act_so = list(parts[0].act) + [nmo + q for q in parts[1].act]
    blocks = (slice(0, n), slice(n, 2 * n))
    for spin in (0, 1):
        assert got[spin].shape == (len(nodes), n, n)
        for k in (0, len(nodes) // 2, len(nodes) - 1):
            exact = sigma_env_uhf(mus[spin] + 1j * nodes[k], eri_so, eps_so,
                                  occ_so, act_so)[blocks[spin], blocks[spin]]
            assert np.abs(exact).max() > 1e-4
            assert np.abs(got[spin][k] - exact).max() < 1e-6
    w_grid = low.cache['w_grid']
    rpa.environment_self_energy_samples(low, parts, nodes)
    assert low.cache['w_grid'] is w_grid


@pytest.mark.parametrize('sigma', ('srg', 'averaged'))
@pytest.mark.parametrize('which', INSIDE)
@pytest.mark.parametrize('ref', REFS)
def test_space_time_sigma_bar_matches_casida(ref, which, sigma):
    """Sigma_bar^E of both spins on space-time against casida on the same
    ISDF factors at srg_s = None. No near-pole warning: the active orbitals
    are at least 0.67 Ha inside both windows. Measured: 5.9e-8 (OH), 1.9e-8
    (CH2), the same for 'srg' and 'averaged'."""
    low, cas = _st_low(ref), _st_casida(ref)
    parts = _st_parts(ref, which)
    with warnings.catch_warnings():
        warnings.simplefilter('error', rpa.SigmaNearPoleWarning)
        got = rpa.environment_self_energy(low, parts, sigma)
    want = rpa.environment_self_energy(cas, parts, sigma)
    assert isinstance(got, tuple) and len(got) == 2
    for ours, casida in zip(got, want):
        assert np.abs(casida).max() > 1e-3
        assert np.abs(ours - casida).max() < 1e-6
    assert np.abs(got[0] - got[1]).max() > 1e-3


@pytest.mark.parametrize('screening', SCREENINGS)
@pytest.mark.parametrize('ref', REFS)
def test_space_time_route_agreement_with_sigma(ref, screening):
    """With sigma='srg' every Energies field, h_eff of both spins and the
    FCI energy agree between space-time and casida on the same ISDF
    factors; the records are pairs where they belong to a spin. Measured,
    OH and CH2: every field 4.2e-7 and 1.7e-7, h_eff 2.2e-7 and 1.2e-7,
    FCI 2.2e-8 and 1.0e-9; the continuation spread is at most 1.3e-5."""
    spaces = _st_spaces(ref, 'unequal_cores')
    ham = build_effective_hamiltonian(
        _st_low(ref), spaces, screening=screening, sigma='srg')
    g, space, en = ham.g_eff, ham.space, ham.energies
    ham_c = build_effective_hamiltonian(
        _st_casida(ref), spaces, screening=screening, sigma='srg')
    g_c, en_c = ham_c.g_eff, ham_c.energies
    assert abs(en.e_sigma_trace) > 1e-3
    for field in dataclasses.fields(Energies):
        if field.name not in RECORDS:
            assert abs(getattr(en, field.name)
                       - getattr(en_c, field.name)) < 5e-6, field.name
    for ours, casida in zip(g[1], g_c[1]):
        assert np.abs(ours - casida).max() < 5e-6
    assert abs(_solve(g, space) - _solve(g_c, space)) < 1e-6
    assert ham.numerical_flags == []
    poles = ham.diagnostics['sigma_poles']
    spread = ham.diagnostics['sigma_continuation']
    assert len(poles['window']) == len(poles['z']) == len(spread) == 2
    assert poles['srg_s'] is None and poles['srg_damping'] == 0.0
    assert max(s.max() for s in spread) < 1e-4


def test_space_time_near_pole_warning_names_the_spin():
    """In the asymmetric OH space the beta orbitals 6 and 7 are within the
    continuation margin of the beta window's upper edge or above it (0.06
    Ha inside and 0.31 Ha outside): both forms warn on space-time, naming
    the spin and pointing to casida, and Sigma_bar is computed all the
    same. The continuation check warns as well, of the elements (7, 7) of
    both spins (1.5e-4 and 2.6e-4 Ha)."""
    low = _st_low('oh')
    spaces = _st_spaces('oh', 'asymmetric')
    for sigma in ('srg', 'averaged'):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            ham = build_effective_hamiltonian(low, spaces, sigma=sigma)
            g, en = ham.g_eff, ham.energies
        near = [w for w in rec if w.category is rpa.SigmaNearPoleWarning]
        assert len(near) == 1 and near[0].filename == __file__
        assert all(w.category in (rpa.SigmaNearPoleWarning,
                                  rpa.SigmaContinuationWarning) for w in rec)
        text = str(near[0].message)
        assert re.search(r'beta orbital 6 \(.*beta orbital 7 \(', text)
        assert "mode='casida'" in text and 'alpha orbital' not in text
        assert all(np.all(np.isfinite(h)) for h in g[1])
        assert {'flag': 'sigma_near_pole',
                'message': text} in ham.numerical_flags


@pytest.mark.parametrize('sigma', (None, 'srg'))
def test_space_time_closed_shell_limit(sigma):
    """A closed shell (water/cc-pVDZ, density fitted, G1 grid) as RHF and
    as the same determinant in a UHF object: on space-time the two-channel
    code gives the restricted H_eff, energy terms and FCI energy. The two
    spins' collocations are the restricted one, and the grids are the
    restricted grids. Measured: the factors and E_c identical, h_eff
    4.4e-16, v_eff 0, the energy terms 2.4e-14, FCI 1.4e-14."""
    rhf = fitted(scf.RHF(gw_water()))
    uhf = rhf.to_uhf()
    numerics = g1_numerics(rhf)
    low_r = run_low_level(rhf, method='rpa', mode='space-time',
                          numerics=numerics)
    low_u = run_low_level(uhf, method='rpa', mode='space-time',
                          numerics=numerics)
    x_r, d_r = low_r.cache['factors']
    x_u, d_u = low_u.cache['factors']
    assert np.abs(d_u - d_r).max() < 1e-10
    assert all(np.abs(x_s - x_r).max() < 1e-10 for x_s in x_u)
    assert abs(low_u.e_c_full - low_r.e_c_full) < 1e-10
    one = ActiveSpace.from_counts(rhf.mo_energy, 5, 2, 2)
    for screening in SCREENINGS:
        ham_r = build_effective_hamiltonian(
            low_r, one, screening=screening, sigma=sigma)
        g_r, en_r = ham_r.g_eff, ham_r.energies
        ham_u = build_effective_hamiltonian(
            low_u, (one, one), screening=screening, sigma=sigma)
        g_u, space, en_u = ham_u.g_eff, ham_u.space, ham_u.energies
        for h in g_u[1]:
            assert np.abs(h - g_r[1]).max() < 1e-8
        for v in g_u[2]:
            assert np.abs(v - g_r[2]).max() < 1e-10
        for field in dataclasses.fields(Energies):
            if field.name not in RECORDS:
                assert abs(getattr(en_u, field.name)
                           - getattr(en_r, field.name)) < 1e-8, field.name
        e_r = solve('pyscf_fci', g_r, 4, 4, spin_free=True)[0]
        assert abs(_solve(g_u, space) - e_r) < 1e-8


def test_mrpa_stays_casida_only_on_space_time():
    with pytest.raises(NotImplementedError, match="mode='casida'"):
        rpa.screening_correction(_st_low('oh'),
                                 _st_parts('oh', 'unequal_cores'), 'mrpa')


@cache
def _st_casida_on_the_density_fit(ref):
    """The casida low-level result on the mean field's own density fit
    (df=True: the RI factors of cc-pvdz-ri), at srg_s = None."""
    return run_low_level(_st_mean_field(ref), method='rpa', mode='casida',
                         numerics=Numerics(downfolding={'srg_s': None}))


@pytest.mark.parametrize('ref', REFS)
def test_space_time_against_casida_on_the_density_fit(ref):
    """space-time (ISDF factors, one collocation per spin, G1 grid) against
    casida on the mean field's ordinary density fit: here the two differ by
    the ISDF fit itself, which the tests on the same ISDF factors cannot
    see. Checked per spin block, so that a poor fit of one spin's orbital
    products would show: the fitted bare v on the active orbitals, vtilde,
    v_eff, h_eff and Sigma_bar of each spin; and E_c, every energy term and
    the FCI energy of H_eff. Targets as for a restricted reference: 1e-5,
    5e-5 on the terms that sum the vtilde error over the occupied active
    orbitals for crpa (E_HF,ee[vtilde], E_0, h_eff), 1e-4 on Sigma_bar.

    Measured, OH and CH2: the bare v of the two fits differs by at most
    8.8e-6 and 2.2e-5 (aa, ab, bb: 2.9e-6, 5.3e-6, 8.8e-6 for OH; 2.2e-5,
    2.1e-5, 8.8e-6 for CH2); E_c 6.7e-7 and 1.5e-7; vtilde 1.3e-6 and
    3.7e-6; the energy terms 2.2e-6 and 1.6e-5 (E_HF,ee[vtilde] for crpa),
    h_eff 2.6e-6 and 1.2e-5, v_eff 1.3e-6 and 3.7e-6, the FCI energy 6.3e-7
    and 1.2e-8; Sigma_bar 5.0e-7 and 2.2e-6. With the alpha collocation used
    for both spins this test fails and the 60 tests on the same ISDF
    factors pass (tried)."""
    low, cas = _st_low(ref), _st_casida_on_the_density_fit(ref)
    assert low.cache['factors'][1].shape[1] == cas.factors[0].shape[0]
    parts = _st_parts(ref, 'unequal_cores')
    spaces = _st_spaces(ref, 'unequal_cores')
    n = parts[0].n_act

    # the bare v of the two fits between the active orbitals of each spin
    isdf = [b[:, p.act][:, :, p.act].reshape(len(b), n * n)
            for b, p in zip(_isdf_three_index(low), parts)]
    ri = [b[:, p.act][:, :, p.act].reshape(len(b), n * n)
          for b, p in zip(cas.factors, parts)]
    for first, second in ((0, 0), (0, 1), (1, 1)):
        fit_error = np.abs(isdf[first].T @ isdf[second]
                           - ri[first].T @ ri[second]).max()
        assert 1e-9 < fit_error < 1e-4

    assert abs(low.e_c_full - cas.e_c_full) < 1e-5
    for screening in ('crpa', 'crpa_ph'):
        vt = rpa.screening_correction(low, parts, screening)
        want = rpa.screening_correction(cas, parts, screening)
        for ours, casida in zip(vt, want):
            assert np.abs(ours - casida).max() < 1e-5
    for screening in SCREENINGS:
        ham = build_effective_hamiltonian(low, spaces,
                                          screening=screening)
        g, space, en = ham.g_eff, ham.space, ham.energies
        ham_c = build_effective_hamiltonian(cas, spaces,
                                            screening=screening)
        g_c, en_c = ham_c.g_eff, ham_c.energies
        tol = 5e-5 if screening == 'crpa' else 1e-5
        for field in dataclasses.fields(Energies):
            if field.name not in RECORDS:
                assert abs(getattr(en, field.name)
                           - getattr(en_c, field.name)) < tol, field.name
        for ours, casida in zip(g[1], g_c[1]):
            assert np.abs(ours - casida).max() < tol
        for ours, casida in zip(g[2], g_c[2]):
            assert np.abs(ours - casida).max() < 1e-5
        assert abs(_solve(g, space) - _solve(g_c, space)) < 1e-5
    for sigma in ('srg', 'averaged'):
        got = rpa.environment_self_energy(low, parts, sigma)
        want = rpa.environment_self_energy(cas, parts, sigma)
        for ours, casida in zip(got, want):
            assert np.abs(casida).max() > 1e-3
            assert np.abs(ours - casida).max() < 1e-4
