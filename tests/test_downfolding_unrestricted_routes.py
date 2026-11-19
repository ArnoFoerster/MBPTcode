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
"""
import dataclasses
import warnings
from functools import cache

import numpy as np
import pytest

from src.Base.numerical_flags import collecting
from src.Base.numerics.numerics import Numerics
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
                                      spin_orbital_integrals, unrestricted)

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


@pytest.fixture(scope='module', autouse=True)
def _release_cached_mean_fields():
    """Empty the caches when the module is done (a mean field held at
    interpreter shutdown leaves pyscf's temporary chkfile behind)."""
    yield
    for cached in (_low, _mean_field):
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
