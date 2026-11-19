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
"""
import dataclasses
import re
import warnings
from functools import cache

import numpy as np
import pytest

from src.Base.numerical_flags import collecting
from src.Base.numerics.numerics import Numerics, numerics_or_default
from src.Base.orbital_spaces import particle_hole_window
from src.Base.utils.grids.analytic_continuation import pade_nodes
from src.MultiReference.Downfolding import (build_effective_hamiltonian,
                                            run_low_level)
from src.MultiReference.Downfolding.energy import Energies
from src.MultiReference.Downfolding.methods import rpa
from src.MultiReference.Downfolding.partition import (active_pair_mask,
                                                      partition)
from src.SingleReference.common.active_space import ActiveSpace
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import (
    imaginary_axis_energy_grid, rpa_correlation_energy_imaginary_axis)
from src.Solvers.active_space import solve
from tests.downfolding_oracle import (CH2, OH, drpa_so,
                                      fitted_spin_orbital_integrals,
                                      sigma_env_uhf, spin_orbital_integrals,
                                      unrestricted, water)

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
    """The pair of Partitions of SPACES[ref][which]."""
    return tuple(partition(space, eps) for space, eps in zip(
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
        terms = rpa.terms(low_if, parts, screening, diagnostics=())
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
        for b, eps, nocc, part in zip(low.factors, low.eps, low.nocc, parts):
            mask = active_pair_mask(part)
            b_a = b[:, :nocc, nocc:].reshape(naux, -1)[:, mask]
            delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()[mask]
            out -= 2.0 * (b_a / delta) @ b_a.T
        return out

    parts = _parts(ref, 'asymmetric')
    b_act = rpa.active_pair_factors(low, parts)
    assert isinstance(b_act, tuple) and len(b_act) == 2
    for ours, b, part in zip(b_act, low.factors, parts):
        assert np.array_equal(ours, b[:, part.act][:, :, part.act])
    assert not np.array_equal(b_act[0], b_act[1])
    for which in ('asymmetric', 'unequal_cores', 'no_beta_pair'):
        parts = _parts(ref, which)
        p_a = rpa.active_polarizability_static(low, parts)
        assert np.abs(p_a - response(parts)).max() < 1e-13
        assert np.abs(p_a).max() > 1e-3
    none = rpa.active_polarizability_static(low, _parts(ref, 'occupied_only'))
    assert none.shape == (naux, naux) and not none.any()
    mf = _mean_field(ref)
    every = tuple(partition(space, eps) for space, eps in zip(
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
    every = tuple(partition(space, eps) for space, eps in zip(
        _all_active(ref), _mean_field(ref).mo_energy))
    terms = rpa.terms(low, every, screening)
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
    every = tuple(partition(space, eps) for space, eps in zip(
        _all_active('oh'), mf.mo_energy))
    terms = rpa.terms(low, every, screening)
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
    g, space, en = build_effective_hamiltonian(
        _low(ref, 'imagfrequency'), spaces, screening=screening)
    g_c, _, en_c = build_effective_hamiltonian(
        _low(ref, 'casida'), spaces, screening=screening)
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
        _, _, en = build_effective_hamiltonian(low_if, spaces,
                                               screening='crpa')
        assert en.diagnostics['static_defect'] < 1e-10
        for screening in ('bare', 'crpa_ph'):
            _, _, en = build_effective_hamiltonian(low_if, spaces,
                                                   screening=screening)
            _, _, en_c = build_effective_hamiltonian(low_cas, spaces,
                                                     screening=screening)
            assert en.diagnostics['static_defect'] > 1e-3
            assert abs(en.diagnostics['static_defect']
                       - en_c.diagnostics['static_defect']) < 1e-10


@pytest.mark.parametrize('screening', SCREENINGS)
@pytest.mark.parametrize('ref', REFS)
def test_ec_active_casida_diagnostic(ref, screening):
    """'ec_active_casida' records the dense two-spin Casida E_c^A on the
    same v_eff and the quadrature's difference from it; the dense value is
    the casida route's E_c^A. Measured |difference|: 4.4e-14 (OH), 3.2e-14
    (CH2). With the restricted spin factor 2 in the quadrature this test,
    the all-active one and the E_0 one fail (tried)."""
    spaces = _spaces(ref, 'asymmetric')
    _, _, en = build_effective_hamiltonian(_low(ref, 'imagfrequency'), spaces,
                                           screening=screening)
    record = en.diagnostics['ec_active_casida']
    assert set(record) == {'casida', 'difference'}
    assert record['difference'] == en.e_c_active - record['casida']
    assert abs(record['difference']) < 1e-10
    _, _, en_c = build_effective_hamiltonian(_low(ref, 'casida'), spaces,
                                             screening=screening)
    assert abs(record['casida'] - en_c.e_c_active) < 1e-10
    assert set(en.diagnostics) == {'static_defect', 'ec_active_casida'}
    _, _, en = build_effective_hamiltonian(
        _low(ref, 'imagfrequency'), _spaces(ref, 'occupied_only'),
        screening=screening)
    assert en.diagnostics['ec_active_casida'] == {'casida': 0.0,
                                                  'difference': 0.0}


@pytest.mark.parametrize('ref', REFS)
def test_lowest_excitation_of_two_spins(ref):
    """Omega_min of the pole-free window on imagfrequency, which
    diagonalizes nothing: the lowest root of the dense two-spin Casida
    problem on the route's factors, i.e. the casida route's lowest Omega;
    the windows of both spins follow. Measured: 0 difference (0.662550 Ha
    for OH with 58 pairs, 0.674525 for CH2 with 70)."""
    low, cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    om = rpa.lowest_excitation(low)
    assert abs(om - cas.omega.min()) < 1e-9
    assert low.cache['omega_min'] == om
    assert rpa.lowest_excitation(low, solver='dense') == om
    parts = _parts(ref, 'unequal_cores')
    for ours, casida in zip(rpa.pole_window(low, parts),
                            rpa.pole_window(cas, parts)):
        assert np.abs(np.subtract(ours, casida)).max() < 1e-9
    with pytest.raises(ValueError, match="'dense', 'bound'"):
        rpa.lowest_excitation(low, solver='davidson')


def test_gap_bound_above_the_dense_limit(monkeypatch):
    """Above the dense limit there is no two-spin Davidson, and Omega_min is
    replaced by the exact lower bound of the direct RPA, the smallest
    orbital-energy gap over both spins. The window is then narrower than the
    true one, so the near-pole warning can come too often, never too rarely.
    A SigmaNearPoleWarning says so once, and the named flag
    'sigma_pole_window_bound' is with every later use of the cached value.
    Measured for OH: the bound is the beta gap, 0.6295 Ha, 0.033 below the
    true Omega_min (CH2: the alpha gap, 0.015 below)."""
    monkeypatch.setattr(rpa, 'DOWNFOLDING_DENSE_OMEGA_MAX_PAIRS', 10)
    mf = _mean_field('oh')
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    gap = min(eps[nocc] - eps[nocc - 1]
              for eps, nocc in zip(mf.mo_energy, low.nocc))
    true = _low('oh', 'casida').omega.min()
    parts = _parts('oh', 'unequal_cores')
    raised = []
    with collecting(raised), pytest.warns(
            rpa.SigmaNearPoleWarning,
            match='smallest orbital-energy gap') as rec:
        windows = rpa.pole_window(low, parts)
    assert len(rec) == 1
    assert [entry['flag'] for entry in raised] == ['sigma_pole_window_bound']
    om = rpa.lowest_excitation(low)
    assert om == gap and 1e-3 < true - om < 0.1
    for (lo, hi), part, eps in zip(windows, parts, mf.mo_energy):
        assert lo == eps[part.inactive].max() - gap
        assert hi == eps[part.env_virt].min() + gap
    again = []
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with collecting(again):
            assert rpa.pole_window(low, parts) == windows
    assert again == raised
    # the bound, asked for by name, whatever the size
    assert rpa.lowest_excitation(_low('oh', 'imagfrequency'),
                                 solver='dense') > om
    with pytest.warns(rpa.SigmaNearPoleWarning):
        assert rpa.lowest_excitation(_low('oh', 'imagfrequency'),
                                     solver='bound') == gap


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
    got = rpa.sigma_env_imag_axis(low, parts, nodes)
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
    rpa.sigma_env_imag_axis(low, parts, nodes)
    assert low.cache['w_grid'] is w_grid


@pytest.mark.parametrize('sigma', ('srg', 'mode_a'))
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
        got = rpa.embedding_self_energy(low_if, parts, sigma)
    want = rpa.embedding_self_energy(low_cas, parts, sigma)
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
    g, space, en = build_effective_hamiltonian(
        _low_infinite_s(ref, 'imagfrequency'), spaces, screening=screening,
        sigma='srg')
    g_c, _, en_c = build_effective_hamiltonian(
        _low_infinite_s(ref, 'casida'), spaces, screening=screening,
        sigma='srg')
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
        _, _, en = build_effective_hamiltonian(low, spaces, sigma='srg')
    assert en.numerical_flags == []
    assert set(en.diagnostics) == {'static_defect', 'ec_active_casida',
                                   'sigma_poles', 'sigma_continuation'}
    poles = en.diagnostics['sigma_poles']
    assert poles['window'] == rpa.pole_window(low, parts)
    assert poles['srg_s'] is None and poles['srg_damping'] == 0.0
    spread = en.diagnostics['sigma_continuation']
    assert len(spread) == len(poles['z']) == len(poles['pole_distance']) == 2
    for s, z, dist in zip(spread, poles['z'], poles['pole_distance']):
        assert s.shape == (n, n) and z.shape == dist.shape == (n,)
        assert s.max() < 1e-4 and dist.min() > 0.26
        assert np.all((z > 0.9) & (z <= 1.0))


def test_near_pole_warning_on_imagfrequency_names_the_spin():
    """In the asymmetric OH space the beta orbitals 6 and 7 lie above the
    beta window (0.20 and 0.27 Ha outside): both forms warn on
    imagfrequency, naming their spin and pointing to casida, and Sigma_bar
    is computed all the same. The continuation check warns too, of the
    element (6, 6) of the beta spin, where the two Pade fits differ by
    9.0e-4 Ha. Both warnings point at the caller and both flags are in
    energies.numerical_flags."""
    low = _low('oh', 'imagfrequency')
    spaces = _spaces('oh', 'asymmetric')
    for sigma in ('srg', 'mode_a'):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            g, _, en = build_effective_hamiltonian(low, spaces, sigma=sigma)
        assert [w.category for w in rec] == [rpa.SigmaNearPoleWarning,
                                             rpa.SigmaContinuationWarning]
        assert all(w.filename == __file__ for w in rec)
        near, spread = (str(w.message) for w in rec)
        assert re.search(r'beta orbital 6 \(.*beta orbital 7 \(', near)
        assert "mode='casida'" in near and 'alpha orbital' not in near
        assert 'beta (6, 6)' in spread and 'alpha (' not in spread
        assert all(np.all(np.isfinite(h)) for h in g[1])
        assert len(en.numerical_flags) == 2
        assert {'flag': 'sigma_near_pole',
                'message': near} in en.numerical_flags
        assert {'flag': 'sigma_continuation_spread',
                'message': spread} in en.numerical_flags


def test_the_bound_reaches_every_build(monkeypatch):
    """With Omega_min replaced by its bound, the flag is in the
    numerical_flags of every Hamiltonian built with Sigma^E on that
    low-level result, and the warning is issued by the first alone."""
    monkeypatch.setattr(rpa, 'DOWNFOLDING_DENSE_OMEGA_MAX_PAIRS', 10)
    low = run_low_level(_mean_field('oh'), method='rpa', mode='imagfrequency')
    spaces = _spaces('oh', 'unequal_cores')
    with pytest.warns(rpa.SigmaNearPoleWarning,
                      match='smallest orbital-energy gap'):
        _, _, first = build_effective_hamiltonian(low, spaces, sigma='srg')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _, _, second = build_effective_hamiltonian(
            low, _spaces('oh', 'no_beta_pair'), sigma='srg')
    for en in (first, second):
        assert [f['flag'] for f in en.numerical_flags] == [
            'sigma_pole_window_bound']


@pytest.mark.parametrize('sigma', (None, 'srg', 'mode_a'))
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
        g_r, _, en_r = build_effective_hamiltonian(low_r, one,
                                                   screening=screening,
                                                   sigma=sigma)
        g_u, space, en_u = build_effective_hamiltonian(
            low_u, (one, one), screening=screening, sigma=sigma)
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
        rpa.sigma_env_imag_axis(_low('oh', 'casida'), parts, [0.5])


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
