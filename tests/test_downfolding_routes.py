"""The RPA downfolding on the imagfrequency and space-time routes against the
casida route.

mode='imagfrequency' keeps the static polarizability P(0) of the whole system
in the normalized density-fitting basis, takes E_RPA,corr from the existing
imaginary-axis energy and keeps its frequency grid. The screening correction
is P - P^A and one Dyson step, vt = B_act^T ([1 - (P - P^A)]^-1 - 1) B_act,
which is the casida route's deletion and Cholesky solve on the same
three-index object written another way, so the two agree per element to
rounding (the static inverse involves no quadrature). E_c^A[v_eff] is the
quadrature of Tr[ln(1 - Pi^A U) + Pi^A U] on the full system's grid, Pi^A =
2 f(i omega) = -4 delta/(delta^2 + omega^2) the bare response of the active
ph pairs (the spin factor and frequency factor of the full chi0); with every
orbital active it is the full energy, so the E_c pair cancels to rounding
(also on the coarsest grid, where the quadrature error is visible). With
sigma, Sigma_bar^E (Lorentzian fit of W, Pade continuation) and the E_0 and
FCI energies follow casida at srg_s = None to far below the 1e-4 target.

Mean fields: water/6-31G RHF and PBE without density fitting (the
three-index object is then the exact eigen-fit of the MO integrals) and a
density-fitted RHF; the contiguous (4,4) active space, a non-contiguous one
and one without active virtual orbitals. Measured values in each test's
docstring.

mode='space-time' does the same on the separable (ISDF) factors in imaginary
time: P(0) and P^A(0) on one static tau grid (P^A with the active columns of
X), E_c^A with exp(-delta tau) pushed through the energy's own tau -> omega
transform. With every orbital active P - P^A and the E_c pair vanish to
rounding (also on a coarse grid); against casida on the mean field's RI
factors the routes differ by the ISDF fit (NH3/cc-pVDZ, HF and PBE0, on the
G1 grid; water/cc-pVDZ for the all-active cases). With sigma, Sigma_bar^E
(the fit of W on the ISDF factors with the environment's G, Pade as on
imagfrequency) and the E_0 and FCI energies follow casida and
imagfrequency to the same ISDF error, and the near-pole space warns.
"""
import dataclasses
import warnings
from functools import cache
import numpy as np
import pytest
from pyscf import dft, gto, scf
from src.Base.factorization.sliced_factors import unpack_factors
from src.Base.numerics.numerics import (Numerics, numerics_or_default,
                                        rpa_quadrature)
from src.Base.orbital_spaces import ov_energies, particle_hole_window
from src.Base.utils.grids.grids import (gap_scaled_w0, gauss_legendre_grid,
                                        minimax_frequency_grid)
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.energy import Energies
from src.EffectiveHamiltonian.Downfolding.methods import rpa
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import (
    rpa_correlation_energy_imaginary_axis, solve_polarizability_imaginary_axis)
from src.SingleReference.LinearResponse.space_time import (
    rpa_correlation_energy_space_time, static_screening_grid,
    static_screening_matrix)
from src.EffectiveHamiltonian.solvers.solve import solve
from tests.downfolding_oracle import h2, water
from tests.gw_density_routes import fitted
from tests.gw_density_routes import mean_field as nh3_mean_field
from tests.gw_density_routes import water as gw_water
from tests.isdf_test_grid import g1_numerics

# the fields of Energies that are records, not energies
RECORDS = ('diagnostics', 'numerical_flags')
REFS = ('hf', 'pbe', 'df_hf')
ASPACES = ('contiguous', 'non_contiguous', 'no_virtual')
SCREENINGS_1 = ('bare', 'crpa', 'crpa_ph')


@cache
def _mean_field(ref):
    """Water/6-31G (spherical): RHF, RKS/PBE, or a density-fitted RHF."""
    if ref == 'hf':
        return water()
    mol = water().mol
    mf = dft.RKS(mol, xc='pbe') if ref == 'pbe' else scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


@cache
def _low(ref, mode):
    """The RPA low-level result of `ref` on `mode` (casida with df=True, the
    same three-index object as imagfrequency)."""
    return run_low_level(_mean_field(ref), method='rpa', mode=mode)


@cache
def _low_infinite_s(ref, mode):
    """`_low` with srg_s = None (s = inf: Eq. 42 on casida)."""
    return run_low_level(_mean_field(ref), method='rpa', mode=mode,
                         numerics=Numerics(downfolding={'srg_s': None}))


@pytest.fixture(scope='module', autouse=True)
def _release_cached_mean_fields():
    """Empty the caches when the module is done: a mean field still held at
    interpreter shutdown leaves pyscf's temporary chkfile to be finalized
    after the tempfile module, which warns (ResourceWarning)."""
    yield
    for cached in (_low_infinite_s, _low, _mean_field):
        cached.cache_clear()


def _aspace(mf, which):
    nocc = mf.mol.nelectron // 2
    if which == 'contiguous':
        return ActiveSpace.from_counts(mf.mo_energy, nocc, 2, 2)
    if which == 'non_contiguous':
        return ActiveSpace.from_indices(len(mf.mo_energy), nocc, [2, 4],
                                        [5, 7])
    return ActiveSpace.from_indices(len(mf.mo_energy), nocc, [3, 4], [])


def _solve(g, space):
    return solve('pyscf_fci', g, space['n_orbs'], space['n_elec_active'],
                 spin_free=True)[0]


def test_full_rpa_is_the_imaginary_axis_energy():
    """run(mode='imagfrequency') keeps the existing imaginary-axis energy
    (equal, the same function), its minimax grid and P(0) = 1 - W(0)^-1 =
    chi0(0) = -4 B_ov diag(1/delta) B_ov^T in the normalized DF basis
    (measured 5.6e-16); E_RPA,corr is the casida one to the quadrature error
    (measured 6.2e-13)."""
    low = _low('hf', 'imagfrequency')
    cas = _low('hf', 'casida')
    assert low.mode == 'imagfrequency' and low.df is True
    assert low.omega is None and low.xpy is None and low.eri is None
    eps, nocc, b = low.eps, low.nocc, low.factors
    lr = LinearResponseSolver(eps, coeff_df=b, spin_mode='restricted')
    assert low.e_c_full == rpa_correlation_energy_imaginary_axis(lr, nocc)
    nfreq = rpa_quadrature(numerics_or_default(None), 'minimax')[1]
    points, weights = minimax_frequency_grid(
        nfreq, *particle_hole_window(eps, nocc)[:2])
    assert np.array_equal(low.cache['grid'][0], points)
    assert np.array_equal(low.cache['grid'][1], weights)
    assert np.array_equal(low.cache['p0'], solve_polarizability_imaginary_axis(
        lr, nocc, [0.0])[0])
    b_ov = b[:, :nocc, nocc:].reshape(b.shape[0], -1)
    chi0 = -4.0 * (b_ov / ov_energies(eps, nocc)) @ b_ov.T
    assert np.abs(low.cache['p0'] - chi0).max() < 1e-12
    assert np.array_equal(cas.factors, low.factors)
    assert abs(low.e_c_full - cas.e_c_full) < 1e-9


def test_full_rpa_quadrature_on_a_wider_window():
    """HCl/6-31G, whose Cl 1s makes the particle-hole window five times as
    wide as water's (span/gap 165 against 32): the minimax energy still
    matches casida on the same three-index object (measured 1.1e-13), and
    so do E_0 of the contiguous (2,2) crpa space (1.1e-13) and the active
    quadrature against the dense value (1.5e-16). Water/cc-pVTZ (span/gap
    52, not run here: 9 s) measured 1.5e-12."""
    mol = gto.M(atom='H 0 0 0; Cl 0 0 1.2746', basis='6-31g', cart=False,
                verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    mf.run()
    low_if = run_low_level(mf, mode='imagfrequency')
    low_cas = run_low_level(mf, mode='casida')
    assert abs(low_if.e_c_full - low_cas.e_c_full) < 1e-9
    a = ActiveSpace.from_counts(mf.mo_energy, low_if.nocc, 2, 2)
    _, _, en = build_effective_hamiltonian(low_if, a)
    _, _, en_c = build_effective_hamiltonian(low_cas, a)
    assert abs(en.e0 - en_c.e0) < 1e-6
    assert abs(en.diagnostics['ec_active_casida']['difference']) < 1e-9


@pytest.mark.parametrize('aspace', ASPACES)
@pytest.mark.parametrize('ref', REFS)
def test_vtilde_matches_casida(ref, aspace):
    """P - P^A and one Dyson step give the casida route's deletion and
    Cholesky solve per element, crpa and crpa_ph, also with no active
    virtual orbital (nothing deleted: the fully screened W(0) - v). Measured
    at most 2.2e-16 (tolerance 1e-8)."""
    low_if, low_cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    mf = low_if.mf
    part = _aspace(mf, aspace)
    for s in ('crpa', 'crpa_ph'):
        vt = rpa.screening_correction(low_if, part, s)
        ref_vt = rpa.screening_correction(low_cas, part, s)
        if s == 'crpa' or part.n_act > part.n_act_occ:
            # crpa_ph has no ph element without an active virtual orbital
            assert np.abs(ref_vt).max() > 1e-3
        assert np.abs(vt - ref_vt).max() < 1e-8
        t = rpa.terms(low_if, part, s, diagnostics=())
        assert all(np.array_equal(x, vt) for x in t.vtilde)


def test_active_pair_factors_and_polarizability():
    """active_pair_factors are the three-index factors of every active pair
    (oo, ov, vv), (tu|vw) = sum_Q B_Q,tu B_Q,vw; P^A(0) is chi0(0) of the
    active ph pairs alone, so the full P(0) minus it is chi0 of the rest."""
    low = _low('hf', 'imagfrequency')
    mf = low.mf
    part = _aspace(mf, 'non_contiguous')
    b = low.factors
    b_act = rpa.active_pair_factors(low, part)
    assert b_act.shape == (b.shape[0], 4, 4)
    assert np.array_equal(b_act, b[:, part.act][:, :, part.act])
    p_a = rpa.active_polarizability_static(low, part)
    occ, vir = part.act[:2], part.act[2:]
    b_a = b[:, occ][:, :, vir].reshape(b.shape[0], -1)
    d_a = (low.eps[vir][None, :] - low.eps[occ][:, None]).ravel()
    assert np.abs(p_a - (-4.0 * (b_a / d_a) @ b_a.T)).max() < 1e-14
    all_act = ActiveSpace.from_indices(len(mf.mo_energy), 5,
                                                 range(5), range(5, 13))
    assert np.abs(rpa.active_polarizability_static(low, all_act)
                  - low.cache['p0']).max() < 1e-12


def test_pair_factor_functions_need_the_three_index_factors():
    """A casida result on the exact four-index integrals (df=False) keeps no
    three-index factors, so active_pair_factors and
    active_polarizability_static refuse it by name and point to df=True."""
    mf = _mean_field('hf')
    low = run_low_level(mf, method='rpa', mode='casida', df=False)
    part = _aspace(mf, 'contiguous')
    for function in (rpa.active_pair_factors,
                     rpa.active_polarizability_static):
        with pytest.raises(NotImplementedError,
                           match='three-index factors.*df=True'):
            function(low, part)


def test_naive_w_difference_is_wrong():
    """Screening is not additive: 'W^r' = W - W^A + v, W^A the RPA of the
    active pairs alone, misses the cRPA vtilde by 0.015 (contiguous (4,4)),
    while P - P^A and Dyson hit it."""
    low = _low('hf', 'imagfrequency')
    mf = low.mf
    part = _aspace(mf, 'contiguous')
    naux, n = low.factors.shape[0], part.n_act
    b_act = rpa.active_pair_factors(low, part).reshape(naux, n * n)
    eye = np.eye(naux)
    w = np.linalg.inv(eye - low.cache['p0'])
    w_a = np.linalg.inv(eye - rpa.active_polarizability_static(low, part))
    naive = (b_act.T @ (w - w_a) @ b_act).reshape((n,) * 4)
    vt = rpa.screening_correction(low, part, 'crpa')
    assert np.abs(naive - vt).max() > 1e-4


@pytest.mark.parametrize('s', SCREENINGS_1)
def test_all_active_ec_pair_cancels(s):
    """H2/6-31G with every orbital active: Pi^A U is the full chi0 v on the
    same grid, so E_c^A = E_c[v] (measured 1.1e-14 for each screening; the
    full energy by slogdet in the auxiliary basis, the active one in the ph
    basis)."""
    low = run_low_level(h2(), mode='imagfrequency')
    part = ActiveSpace.from_indices(4, 1, [0], [1, 2, 3])
    t = rpa.terms(low, part, s)
    assert abs(t.e_c_full - t.e_c_active) < 1e-10
    assert np.abs(t.vtilde[0]).max() == 0.0


@pytest.mark.parametrize('s', SCREENINGS_1)
def test_all_active_ec_pair_cancels_on_a_coarse_grid(s):
    """Water/6-31G with every orbital active on the coarsest minimax grid
    (nfreq_rpa = 6): the quadrature error of E_c[v] against casida is
    measured 2.0e-6, the E_c pair cancels to 1.3e-14. Only if
    both energies are integrated on one grid does the pair cancel below
    1e-12, so this fails when E_c^A leaves the full energy's grid; at the
    default 20 points the quadrature error (6.2e-13) is too small to tell."""
    mf = _mean_field('hf')
    low = run_low_level(mf, mode='imagfrequency',
                        numerics=Numerics(quadrature={'nfreq_rpa': 6}))
    assert len(low.cache['grid'][0]) == 6
    assert abs(low.e_c_full - _low('hf', 'casida').e_c_full) > 1e-7
    part = ActiveSpace.from_indices(13, 5, range(5), range(5, 13))
    t = rpa.terms(low, part, s)
    assert abs(t.e_c_full - t.e_c_active) < 1e-12


@pytest.mark.parametrize('s', SCREENINGS_1)
@pytest.mark.parametrize('aspace', ('contiguous', 'non_contiguous'))
@pytest.mark.parametrize('ref', REFS)
def test_e0_terms_match_casida(ref, aspace, s):
    """Every Energies field, E_0, h_eff, v_eff and the FCI energy of H_eff
    agree between imagfrequency and casida on the same three-index object.
    The E_c pair differs by the quadrature error, the rest only by rounding:
    measured |E_0 difference| at most 7.4e-12 (PBE; HF 6.3e-13, DF-HF
    6.0e-13; target 1e-6), all of it from E_c[v] (the largest field
    difference, 7.4e-12; E_c^A at most 3.6e-14), h_eff and v_eff 8.9e-16,
    FCI 7.4e-12."""
    low_if, low_cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    a = _aspace(low_if.mf, aspace)
    g, space, en = build_effective_hamiltonian(low_if, a, screening=s)
    g_c, _, en_c = build_effective_hamiltonian(low_cas, a, screening=s)
    for f in dataclasses.fields(Energies):
        if f.name not in RECORDS:
            assert abs(getattr(en, f.name) - getattr(en_c, f.name)) < 1e-6
    assert abs(en.e0 - en_c.e0) < 1e-6
    assert np.abs(g[1] - g_c[1]).max() < 1e-8
    assert np.abs(g[2] - g_c[2]).max() < 1e-8
    assert abs(_solve(g, space) - _solve(g_c, space)) < 1e-6


@pytest.mark.parametrize('ref', ('hf', 'df_hf'))
def test_static_defect_zero_for_crpa(ref):
    """W(0) from the cache's P(0), projected on the active pairs, against
    the active dRPA of the fitted v + vtilde: zero for crpa
    (measured 1.1e-16 / 5.0e-16), and the casida route's defect for bare
    and crpa_ph (measured 3.9e-15)."""
    low_if, low_cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    mf = low_if.mf
    a = _aspace(mf, 'contiguous')
    part = a
    _, _, en = build_effective_hamiltonian(low_if, a, screening='crpa')
    assert en.diagnostics['static_defect'] < 1e-8
    for s in ('bare', 'crpa_ph'):
        _, _, en = build_effective_hamiltonian(low_if, a, screening=s)
        _, _, en_c = build_effective_hamiltonian(low_cas, a, screening=s)
        assert en.diagnostics['static_defect'] > 1e-4
        assert abs(en.diagnostics['static_defect']
                   - en_c.diagnostics['static_defect']) < 1e-8
    assert rpa.static_defect(low_if, ActiveSpace.from_indices(13, 5, [], []),
        np.zeros((0,) * 4)) == 0.0


@pytest.mark.parametrize('s', SCREENINGS_1)
def test_ec_active_casida_diagnostic(s):
    """'ec_active_casida' records the dense-Casida E_c^A on the same v_eff
    and the quadrature's difference from it; the minimax grid of the full
    window resolves the narrower active window (measured |difference| at
    most 1.2e-14; 3.6e-14 on PBE orbitals). The dense value is the casida route's E_c^A (to the 1e-8
    agreement of vtilde)."""
    low = _low('hf', 'imagfrequency')
    a = _aspace(low.mf, 'contiguous')
    _, _, en = build_effective_hamiltonian(low, a, screening=s)
    rec = en.diagnostics['ec_active_casida']
    assert set(rec) == {'casida', 'difference'}
    assert rec['difference'] == en.e_c_active - rec['casida']
    assert abs(rec['difference']) < 1e-9
    _, _, en_c = build_effective_hamiltonian(_low('hf', 'casida'), a,
                                             screening=s)
    assert abs(rec['casida'] - en_c.e_c_active) < 1e-8
    assert set(en.diagnostics) == {'static_defect', 'ec_active_casida'}
    _, _, en = build_effective_hamiltonian(low, a, screening=s,
                                           diagnostics=('ec_active_casida',))
    assert list(en.diagnostics) == ['ec_active_casida']


def _same(a, b):
    (g1, _, e1), (g2, _, e2) = a, b
    assert abs(g1[0] - g2[0]) < 1e-12
    assert np.abs(g1[1] - g2[1]).max() < 1e-12
    assert np.abs(g1[2] - g2[2]).max() < 1e-12
    for f in dataclasses.fields(Energies):
        if f.name not in RECORDS:
            assert abs(getattr(e1, f.name) - getattr(e2, f.name)) < 1e-12
    assert e1.diagnostics.keys() == e2.diagnostics.keys()
    assert e1.numerical_flags == e2.numerical_flags


MOVED = 'O 0 0 0.1173; H 0 0.9 -0.55; H 0 -0.9 -0.55'


@pytest.mark.parametrize('mode', ['casida', 'imagfrequency'])
def test_a_mean_field_changed_after_the_run_is_refused(mode):
    """A low-level result holds the screening and E_c of the orbitals it was
    run on, and the build reads the orbitals, the core Hamiltonian and E_nuc
    from the mean-field object itself. When that object is converged again
    at another geometry, a build on the old result is refused (ValueError)
    and not a mixture of the two; a new run on the changed mean field
    builds. (Before the check, casida returned E_0 = -69.783 Ha without a
    word; the new run gives -69.083 Ha.)"""
    mf = water()
    space = _aspace(mf, 'contiguous')
    low = run_low_level(mf, method='rpa', mode=mode)
    build_effective_hamiltonian(low, space)
    mf.mol.set_geom_(MOVED)
    mf.kernel()
    with pytest.raises(ValueError,
                       match='mean field has changed.*run_low_level'):
        build_effective_hamiltonian(low, space)
    _, _, en = build_effective_hamiltonian(
        run_low_level(mf, method='rpa', mode=mode), space)
    assert np.isfinite(en.e0)


def test_low_reuse():
    """One imagfrequency low-level result, used for crpa, crpa_ph, bare and
    a second active space in turn, gives each the same H_eff as a fresh
    one, and its cache keeps only what run put there."""
    mf = _mean_field('hf')
    low = run_low_level(mf, mode='imagfrequency')
    p0 = low.cache['p0'].copy()
    a = _aspace(mf, 'contiguous')
    other = _aspace(mf, 'non_contiguous')
    calls = [(a, 'crpa'), (a, 'crpa_ph'), (a, 'bare'), (other, 'crpa'),
             (a, 'crpa')]
    shared = [build_effective_hamiltonian(low, x, screening=s)
              for x, s in calls]
    for (x, s), got in zip(calls, shared):
        fresh = run_low_level(mf, mode='imagfrequency')
        _same(got, build_effective_hamiltonian(fresh, x, screening=s))
    assert set(low.cache) == {'grid', 'p0'}
    assert np.array_equal(low.cache['p0'], p0)


@pytest.mark.parametrize('sigma', ('srg', 'mode_a'))
@pytest.mark.parametrize('aspace', ('contiguous', 'no_virtual'))
@pytest.mark.parametrize('ref', REFS)
def test_sigma_bar_matches_casida(ref, aspace, sigma):
    """Sigma_bar^E on imagfrequency (the Lorentzian fit of W on the minimax W
    grid, Pade from the imaginary axis to z_tu or to the real axis) against
    casida on the same three-index object at srg_s = None, where 'srg' is
    Eq. 42 on both. Target 1e-4 Eh per element; measured max 5.8e-9 (HF,
    DF-HF), 9.3e-8 (PBE), the largest on the space without active virtual
    orbitals; equal for 'srg' and 'mode_a' (a diagonal element).
    (The non-contiguous space here has virtual 7 active over the
    environment's 6, beyond the pole-free window: refused on imagfrequency,
    tests/test_downfolding_sigma.py.)"""
    low_if = _low_infinite_s(ref, 'imagfrequency')
    low_cas = _low_infinite_s(ref, 'casida')
    part = _aspace(low_if.mf, aspace)
    got = rpa.embedding_self_energy(low_if, part, sigma)
    want = rpa.embedding_self_energy(low_cas, part, sigma)
    assert np.abs(want).max() > 1e-3
    assert np.abs(got - want).max() < 1e-4


@pytest.mark.parametrize('s', SCREENINGS_1)
@pytest.mark.parametrize('ref', REFS)
def test_route_agreement_with_sigma(ref, s):
    """With sigma='srg' (srg_s = None) E_0, every Energies field and the FCI
    energy of H_eff agree between imagfrequency and casida, contiguous (4,4):
    the Sigma^E terms of E_0 and h_eff follow Sigma_bar^E. Target 1e-4;
    measured max |E_0 difference| 2.5e-10 (every field 2.6e-10, h_eff
    2.8e-9 on PBE), FCI 1.4e-11."""
    low_if = _low_infinite_s(ref, 'imagfrequency')
    low_cas = _low_infinite_s(ref, 'casida')
    a = _aspace(low_if.mf, 'contiguous')
    g, space, en = build_effective_hamiltonian(low_if, a, screening=s,
                                               sigma='srg')
    g_c, _, en_c = build_effective_hamiltonian(low_cas, a, screening=s,
                                               sigma='srg')
    assert abs(en.e_sigma_trace) > 1e-3
    for f in dataclasses.fields(Energies):
        if f.name not in RECORDS:
            assert abs(getattr(en, f.name) - getattr(en_c, f.name)) < 1e-4
    assert np.abs(g[1] - g_c[1]).max() < 1e-4
    assert abs(_solve(g, space) - _solve(g_c, space)) < 1e-4


def test_mrpa_refused_here():
    """mRPA is casida only and its refusal points to casida.
    df=False has no meaning here (the route is density fitted). The residues
    of W (`screened_residues`) need the Casida eigenvectors and refuse here
    too, pointing to casida."""
    low = _low('hf', 'imagfrequency')
    a = _aspace(low.mf, 'contiguous')
    with pytest.raises(NotImplementedError, match="mode='casida'"):
        build_effective_hamiltonian(low, a, screening='mrpa')
    with pytest.raises(NotImplementedError, match="mode='casida'"):
        rpa.screened_residues(low, a)
    with pytest.raises(ValueError, match="df=False.*mode='casida'"):
        run_low_level(low.mf, mode='imagfrequency', df=False)


# The space-time route: ISDF factors (X, D) on the G1 grid of NH3/cc-pVDZ
# against casida on the mean field's own density fit (cc-pvdz-ri, the basis
# the ISDF fits too), as tests/gw_density_routes.py; water/cc-pVDZ for the
# all-active cases (6-31G has no validated ISDF grid).

ST_REFS = ('HF', 'PBE0')


@cache
def _st_mean_field(ref):
    """NH3/cc-pVDZ, HF or PBE0, density fitted on cc-pvdz-ri ('water': the
    HF of water/cc-pVDZ, fitted the same way)."""
    if ref == 'water':
        return fitted(scf.RHF(gw_water()))
    return nh3_mean_field(ref)


@cache
def _st_low(ref, mode, quadrature=(), downfolding=()):
    """The RPA low-level result of `ref` (`_st_mean_field`) on `mode`, with
    the quadrature and downfolding settings `quadrature`, `downfolding`
    (pairs); on space-time on the G1 grid."""
    mf = _st_mean_field(ref)
    numerics = Numerics(quadrature=dict(quadrature),
                        downfolding=dict(downfolding))
    if mode == 'space-time':
        numerics = g1_numerics(mf, numerics)
    return run_low_level(mf, method='rpa', mode=mode, numerics=numerics)


#: srg_s = None (s = inf), where 'srg' is Eq. 42 on casida too.
INFINITE_S = (('srg_s', None),)


@pytest.fixture(scope='module', autouse=True)
def _release_space_time_caches():
    """As `_release_cached_mean_fields`, for the space-time caches."""
    yield
    for cached in (_st_low, _st_mean_field):
        cached.cache_clear()


def _st_aspace(mf, which):
    """NH3 (nocc 5; 2, 3 and 6, 7 and 8, 9 degenerate): (6,6) contiguous,
    a non-contiguous (5,4) one and one without active virtual orbitals."""
    n = len(mf.mo_energy)
    if which == 'contiguous':
        return ActiveSpace.from_indices(n, 5, [2, 3, 4], [5, 6, 7])
    if which == 'non_contiguous':
        return ActiveSpace.from_indices(n, 5, [1, 4], [5, 8, 9])
    return ActiveSpace.from_indices(n, 5, [2, 3, 4], [])


def _all_active(low):
    n = len(low.eps)
    return ActiveSpace.from_indices(n, low.nocc, range(low.nocc),
                                              range(low.nocc, n))


def test_space_time_full_rpa():
    """run(mode='space-time') keeps the ISDF factors (X, D) of
    `separable_factors` on numerics.isdf's grid, the static tau grid
    (`static_screening_grid` at quadrature.ntau_w) with P(0) built on it,
    which is 1 - W(0)^-1 of `static_screening_matrix` (measured 9.4e-16),
    and the energy's grid: minimax tau (the static count rule) to a
    Gauss-Legendre frequency axis of `rpa_quadrature`, the grid of the RPA
    ground-state chain. e_c_full is `rpa_correlation_energy_space_time` on
    it (equal) and the casida one to the ISDF error (measured 8.5e-7, target
    1e-5)."""
    low, cas = _st_low('HF', 'space-time'), _st_low('HF', 'casida')
    mf = low.mf
    assert low.mode == 'space-time' and low.df is True
    assert low.omega is None and low.xpy is None
    assert low.factors is None and low.eri is None
    x, d = low.cache['factors']
    want = unpack_factors(separable_factors(mf, mf.mol, numerics=low.numerics,
                                            comm=None))
    assert np.array_equal(x, want[0]) and np.array_equal(d, want[1])
    eps, nocc = low.eps, low.nocc
    q = low.numerics.quadrature
    g0 = low.cache['static_grid']
    ref0 = static_screening_grid(eps, nocc, ntau=q.ntau_w,
                                 tau_target=q.tau_target)
    assert np.array_equal(g0.tau_points, ref0.tau_points)
    assert np.array_equal(g0.cosft_wt, ref0.cosft_wt)
    w0 = static_screening_matrix(x, d, eps, nocc, g0, comm=None)
    p0 = low.cache['p0']
    assert np.abs(p0 - (np.eye(len(p0)) - np.linalg.inv(w0))).max() < 1e-12
    grid = low.cache['energy_grid']
    ntau, nfreq = rpa_quadrature(low.numerics)
    assert ntau is None and grid.ntau == g0.ntau
    nu, wt = gauss_legendre_grid(nfreq, gap_scaled_w0(eps, nocc))
    assert np.array_equal(grid.omega_points, nu)
    assert np.array_equal(grid.omega_weights, wt)
    assert low.e_c_full == rpa_correlation_energy_space_time(
        x, d, eps, nocc, grid, comm=None)
    assert abs(low.e_c_full - cas.e_c_full) < 1e-5


def test_active_part_cancels_exactly():
    """With every orbital active P^A, built on the stored static grid with
    the active columns of X, is the full P(0) (measured max |P - P^A| 0.0),
    so the Dyson step of P - P^A leaves nothing to screen and vtilde == 0
    for every screening (water/cc-pVDZ)."""
    low = _st_low('water', 'space-time')
    part = _all_active(low)
    p_a = rpa.active_polarizability_static(low, part)
    assert np.abs(low.cache['p0'] - p_a).max() < 1e-10
    n = part.n_act
    b_act = rpa.active_pair_factors(low, part).reshape(-1, n * n)
    w_r = np.linalg.inv(np.eye(len(p_a)) - (low.cache['p0'] - p_a))
    w_r -= np.eye(len(p_a))
    assert np.abs(b_act.T @ w_r @ b_act).max() < 1e-10
    for s in SCREENINGS_1:
        t = rpa.terms(low, part, s, diagnostics=())
        assert np.abs(t.vtilde[0]).max() == 0.0


def test_space_time_active_pair_factors_and_polarizability():
    """active_pair_factors are D^T (X_t X_u) of every active pair, so (tu|vw)
    = sum_Q B_Q,tu B_Q,vw is the ISDF v (measured 4.4e-16 against the
    einsum); P^A(0) is the static grid's transform of -2 sum_ia B_ia B_ia^T
    exp(-delta_ia tau) over the active ph pairs (3.8e-17 against the pair
    sum), whose pi(0) is -4/delta to the transform's 2.0e-6 (relative)."""
    low = _st_low('HF', 'space-time')
    part = _st_aspace(low.mf, 'non_contiguous')
    x, d = low.cache['factors']
    act = part.act
    b_act = rpa.active_pair_factors(low, part)
    want = np.einsum('PQ,Pt,Pu->Qtu', d, x[:, act], x[:, act])
    assert b_act.shape == (d.shape[1], 5, 5)
    assert np.abs(b_act - want).max() < 1e-12
    g0 = low.cache['static_grid']
    occ, vir = act[:2], act[2:]
    b_a = want[:, :2, 2:].reshape(d.shape[1], -1)
    delta = (low.eps[vir][None, :] - low.eps[occ][:, None]).ravel()
    pi0 = -2.0 * np.exp(-np.outer(delta, g0.tau_points)) @ g0.cosft_wt[0]
    p_a = rpa.active_polarizability_static(low, part)
    assert np.abs(p_a - (b_a * pi0) @ b_a.T).max() < 1e-12


@pytest.mark.parametrize('s', SCREENINGS_1)
def test_all_active_ec_pair_cancels_space_time(s):
    """Water/cc-pVDZ, every orbital active: Pi^A(i w_k), e^(-delta tau)
    pushed through the energy's own tau -> omega transform, makes Pi^A U the
    full chi0 v at every frequency, so E_c^A = E_c[v] (measured 4.3e-13)."""
    low = _st_low('water', 'space-time')
    t = rpa.terms(low, _all_active(low), s)
    assert abs(t.e_c_full - t.e_c_active) < 1e-10


@pytest.mark.parametrize('s', SCREENINGS_1)
def test_all_active_ec_pair_cancels_on_a_coarse_space_time_grid(s):
    """The same on a coarse grid, 10 tau points (the fewest whose transform
    fit does not warn here) and 6 Gauss-Legendre frequencies: E_c[v] moves
    from the default grid's (14, 40) by 3.0e-3, the quadrature error made
    visible, and the pair still cancels (measured 6.7e-15). E_c^A on the
    exact Lorentzian -4 delta/(delta^2 + w^2) at the same frequencies would
    miss by 1.4e-6 (1.8e-10 at the default grid, too small to tell): only
    the energy's own points, weights and transform make the pair cancel."""
    low = _st_low('water', 'space-time', (('ntau_rpa', 10), ('nfreq_rpa', 6)))
    grid = low.cache['energy_grid']
    assert grid.ntau == 10 and grid.nfreq == 6
    assert abs(low.e_c_full - _st_low('water', 'space-time').e_c_full) > 1e-7
    t = rpa.terms(low, _all_active(low), s)
    assert abs(t.e_c_full - t.e_c_active) < 1e-10


@pytest.mark.parametrize('aspace', ASPACES)
@pytest.mark.parametrize('ref', ST_REFS)
def test_space_time_vtilde_matches_casida(ref, aspace):
    """P - P^A on the static tau grid and one Dyson step, on the ISDF
    factors, against casida's deletion on the RI factors of the same
    auxiliary basis: per element within the ISDF error, crpa and crpa_ph,
    also with no active virtual orbital. Target 1e-5; measured at most
    3.6e-6 (HF) and 4.7e-6 (PBE0), on the non-contiguous space. The ISDF's
    bare v itself differs from the RI v on the contiguous space's orbitals
    by 7.6e-6 (G1), 9.6e-6 (G3): the residual is the fit's, not the static
    tau quadrature's (unchanged from 14 to 30 tau points)."""
    low_st, low_cas = _st_low(ref, 'space-time'), _st_low(ref, 'casida')
    mf = low_st.mf
    part = _st_aspace(mf, aspace)
    for s in ('crpa', 'crpa_ph'):
        vt = rpa.screening_correction(low_st, part, s)
        ref_vt = rpa.screening_correction(low_cas, part, s)
        if s == 'crpa' or part.n_act > part.n_act_occ:
            assert np.abs(ref_vt).max() > 1e-3
        assert np.abs(vt - ref_vt).max() < 1e-5
        t = rpa.terms(low_st, part, s, diagnostics=())
        assert all(np.array_equal(x, vt) for x in t.vtilde)


@pytest.mark.parametrize('s', SCREENINGS_1)
@pytest.mark.parametrize('aspace', ('contiguous', 'non_contiguous'))
@pytest.mark.parametrize('ref', ST_REFS)
def test_space_time_e0_terms_match_casida(ref, aspace, s):
    """Every Energies field, E_0, h_eff, v_eff and the FCI energy of H_eff
    on space-time against casida: the ISDF error of E_c[v] and vtilde plus
    the quadrature's. Target 1e-5. Measured: E_c[v] 8.5e-7 (HF) / 1.4e-6
    (PBE0), E_c^A at most 5.7e-7; for bare and crpa_ph every field and E_0
    at most 1.4e-6; for crpa E_HF,ee[vtilde] = 1/2 Tr[t^DC rho^A,0] sums the
    vtilde error over the active occupied orbitals, 2.6e-5, and E_0 follows
    (2.5e-5), as does h_eff = f - t^DC (1.6e-5): the target missed by 2.6x,
    hence 5e-5 on those for crpa alone (the same at G3, 2.2e-5,
    so the ISDF fit's), 1e-5 for bare and crpa_ph. The two cancel in
    the energy of H_eff: FCI at most 1.1e-6, v_eff 4.7e-6. The route's
    E_c^A against the dense one on the same v_eff ('ec_active_casida'):
    1.9e-11."""
    low_st, low_cas = _st_low(ref, 'space-time'), _st_low(ref, 'casida')
    a = _st_aspace(low_st.mf, aspace)
    g, space, en = build_effective_hamiltonian(low_st, a, screening=s)
    g_c, _, en_c = build_effective_hamiltonian(low_cas, a, screening=s)
    tol = 5e-5 if s == 'crpa' else 1e-5
    for f in dataclasses.fields(Energies):
        if f.name not in RECORDS:
            assert abs(getattr(en, f.name) - getattr(en_c, f.name)) < tol
    assert abs(en.e0 - en_c.e0) < tol
    assert np.abs(g[1] - g_c[1]).max() < tol
    assert np.abs(g[2] - g_c[2]).max() < 1e-5
    assert abs(_solve(g, space) - _solve(g_c, space)) < 1e-5
    assert abs(en.diagnostics['ec_active_casida']['difference']) < 1e-9


@pytest.mark.parametrize('ref', ST_REFS)
def test_space_time_static_defect_zero_for_crpa(ref):
    """W(0) = [1 - P(0)]^-1 of the static grid, projected on the active
    pairs, against the active dRPA of the ISDF v + vtilde with the ph pairs'
    static response on the same grid: an identity for crpa
    (measured 4.4e-16; with the exact -4/delta in W^A it would be the
    transform's error instead), and the casida route's defect for bare and
    crpa_ph to the ISDF error (target 1e-5; measured at most 3.7e-6 on
    this space, 8.6e-6 on the non-contiguous one)."""
    low_st, low_cas = _st_low(ref, 'space-time'), _st_low(ref, 'casida')
    a = _st_aspace(low_st.mf, 'contiguous')
    _, _, en = build_effective_hamiltonian(low_st, a, screening='crpa')
    assert en.diagnostics['static_defect'] < 1e-8
    for s in ('bare', 'crpa_ph'):
        _, _, en = build_effective_hamiltonian(low_st, a, screening=s)
        _, _, en_c = build_effective_hamiltonian(low_cas, a, screening=s)
        assert en.diagnostics['static_defect'] > 1e-4
        assert abs(en.diagnostics['static_defect']
                   - en_c.diagnostics['static_defect']) < 1e-5


def test_space_time_low_reuse():
    """One space-time low-level result, used for crpa, crpa_ph, bare and a
    second active space in turn, gives each the same H_eff as a fresh one,
    and its cache keeps only what run put there."""
    mf = _st_mean_field('HF')
    numerics = g1_numerics(mf)
    low = run_low_level(mf, mode='space-time', numerics=numerics)
    p0 = low.cache['p0'].copy()
    a = _st_aspace(mf, 'contiguous')
    other = _st_aspace(mf, 'non_contiguous')
    calls = [(a, 'crpa'), (a, 'crpa_ph'), (a, 'bare'), (other, 'crpa'),
             (a, 'crpa')]
    shared = [build_effective_hamiltonian(low, x, screening=s)
              for x, s in calls]
    for (x, s), got in zip(calls, shared):
        fresh = run_low_level(mf, mode='space-time', numerics=numerics)
        _same(got, build_effective_hamiltonian(fresh, x, screening=s))
    assert set(low.cache) == {'factors', 'static_grid', 'p0', 'energy_grid'}
    assert np.array_equal(low.cache['p0'], p0)


def test_space_time_refusals():
    """mRPA is casida only and points there; df=False has no meaning on the
    ISDF route; without an ISDF grid in numerics run says where to name one;
    Sigma^E near the poles is not refused but warns, as on imagfrequency:
    on the non-contiguous space occupied 1 and virtual 8, 9 lie 0.108 and
    0.165 Ha from the pole-free window's edges, inside the continuation
    margin; both forms name them (not orbital 4), point to mode='casida',
    and the Hamiltonian is built with 'sigma_near_pole' among its flags.
    A casida result has no imaginary-axis
    S^E to sample and points to the numerical routes."""
    low = _st_low('HF', 'space-time')
    mf = low.mf
    a = _st_aspace(mf, 'contiguous')
    with pytest.raises(NotImplementedError, match="mode='casida'"):
        build_effective_hamiltonian(low, a, screening='mrpa')
    with pytest.raises(ValueError, match="df=False.*mode='casida'"):
        run_low_level(mf, mode='space-time', df=False,
                      numerics=low.numerics)
    with pytest.raises(ValueError, match='numerics.isdf'):
        run_low_level(mf, mode='space-time')
    near = _st_aspace(mf, 'non_contiguous')
    for sigma in ('srg', 'mode_a'):
        with pytest.warns(rpa.SigmaNearPoleWarning,
                          match=r'orbital 1 \(.*orbital 8 \('
                          r".*orbital 9 \(.*continuation_margin"
                          r".*mode='casida'") as rec:
            # this close to the poles the two Pade fits disagree as well
            warnings.simplefilter('ignore', rpa.SigmaContinuationWarning)
            g, _, en = build_effective_hamiltonian(low, near, sigma=sigma)
        text = str([w.message for w in rec
                    if w.category is rpa.SigmaNearPoleWarning][0])
        assert 'orbital 4 ' not in text
        assert np.all(np.isfinite(g[1]))
        assert {'flag': 'sigma_near_pole',
                'message': text} in en.numerical_flags
    part = a
    with pytest.raises(NotImplementedError,
                       match="mode='imagfrequency' or mode='space-time'"):
        rpa.sigma_env_imag_axis(_st_low('HF', 'casida'), part, [1.0])


@pytest.mark.parametrize('sigma', ('srg', 'mode_a'))
@pytest.mark.parametrize('aspace', ('contiguous', 'no_virtual'))
@pytest.mark.parametrize('ref', ST_REFS)
def test_space_time_sigma_bar_matches_casida(ref, aspace, sigma):
    """Sigma_bar^E on space-time (S^E(mu + i w) of the imaginary-time kernel
    with the environment's G on the Pade nodes, continued as on
    imagfrequency) against casida on the mean field's RI factors at
    srg_s = None: the ISDF fit's error, the quadratures' and the
    continuation's. Target 1e-4 Eh per element; measured max 2.6e-6 (HF) /
    3.5e-6 (PBE0) on the contiguous (6,6), 1.4e-6 / 3.1e-6 without active
    virtual orbitals (|Sigma_bar| up to 3.9e-2), equal for both forms; the
    imagfrequency route on the same RI factors is within 3.1e-7 of casida,
    so the residual is the ISDF fit's, as for vtilde."""
    low_st = _st_low(ref, 'space-time', downfolding=INFINITE_S)
    low_cas = _st_low(ref, 'casida', downfolding=INFINITE_S)
    part = _st_aspace(low_st.mf, aspace)
    got = rpa.embedding_self_energy(low_st, part, sigma)
    want = rpa.embedding_self_energy(low_cas, part, sigma)
    assert np.array_equal(got, got.T)
    assert np.abs(want).max() > 1e-2
    assert np.abs(got - want).max() < 1e-4


@pytest.mark.parametrize('s', SCREENINGS_1)
@pytest.mark.parametrize('ref', ST_REFS)
def test_space_time_route_agreement_with_sigma(ref, s):
    """With sigma='srg' (srg_s = None) every Energies field, E_0, h_eff and
    the FCI energy of H_eff on the three routes, contiguous (6,6): space-time
    against casida and against imagfrequency (the mean field's RI factors).
    Target 1e-4 on the fields, E_0 and h_eff, 1e-5 on FCI. Measured
    space-time - casida: fields at most 1.4e-5 (bare, crpa_ph) and 2.4e-5
    (crpa, E_HF,ee[vtilde] as without sigma), E_0 at most 3.8e-5 (HF crpa),
    h_eff 1.1e-5, FCI 1.2e-6; imagfrequency - casida 2.3e-8 (E_0), 7.4e-11
    (FCI), so space-time - imagfrequency is the same ISDF error. The
    'sigma_continuation' spread is 3.1e-6 (HF) / 8.4e-6 (PBE0), without a
    warning, and 'sigma_poles' records s = inf and the window of the
    route's Omega_min, casida's to the ISDF error (2.7e-6)."""
    lows = [_st_low(ref, m, downfolding=INFINITE_S)
            for m in ('space-time', 'casida', 'imagfrequency')]
    a = _st_aspace(lows[0].mf, 'contiguous')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        (g, space, en), *others = [
            build_effective_hamiltonian(low, a, screening=s, sigma='srg')
            for low in lows]
    assert abs(en.e_sigma_trace) > 1e-2
    e_fci = _solve(g, space)
    for g_o, _, en_o in others:
        for f in dataclasses.fields(Energies):
            if f.name not in RECORDS:
                assert abs(getattr(en, f.name) - getattr(en_o, f.name)) < 1e-4
        assert np.abs(g[1] - g_o[1]).max() < 1e-4
        assert abs(e_fci - _solve(g_o, space)) < 1e-5
    assert en.diagnostics['sigma_continuation'].max() < 1e-4
    poles = en.diagnostics['sigma_poles']
    ref_poles = others[0][2].diagnostics['sigma_poles']
    assert poles['srg_s'] is None and poles['srg_damping'] == 0.0
    assert np.allclose(poles['window'], ref_poles['window'], atol=1e-5,
                       rtol=0)


# --- the grids shared with GW and the RPA ground-state chain ------------------

def test_sigma_uses_the_gw_screening_fit_grid(monkeypatch):
    """S^E on imagfrequency samples W on the points of `screening_fit_grid`
    and fits it with its poles, and so does the GW density's
    `self_energy_matrix_imaginary_axis`: one recipe, which both call, not a
    copy of it."""
    from src.SingleReference.GW import imaginary_axis
    mf = water()
    part = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    shared = imaginary_axis.screening_fit_grid
    calls, fitted = [], []

    def counted(eps, nocc, numerics=None):
        calls.append(numerics)
        return shared(eps, nocc, numerics)

    fit = rpa.fitted_self_energy_matrix_imaginary_axis

    def spy(b, eps_mu, freq_points, w_grid, query, poles, tikhonov):
        fitted.append((freq_points, poles))
        return fit(b, eps_mu, freq_points, w_grid, query, poles, tikhonov)

    monkeypatch.setattr(rpa, 'screening_fit_grid', counted)
    monkeypatch.setattr(imaginary_axis, 'screening_fit_grid', counted)
    monkeypatch.setattr(rpa, 'fitted_self_energy_matrix_imaginary_axis', spy)
    coarse = Numerics(imagfreq={'nfreq': 26})
    low = run_low_level(mf, method='rpa', mode='imagfrequency',
                        numerics=coarse)
    rpa.sigma_env_imag_axis(low, part, [0.5])
    # W's points, then the fit's poles, both on the run's numerics
    assert calls == [coarse, coarse]
    points, _, poles = shared(low.eps, low.nocc, coarse)
    assert len(points) == 26
    assert np.array_equal(low.cache['w_grid'][0], points)
    assert np.array_equal(fitted[0][0], points)
    assert np.array_equal(fitted[0][1], poles)
    imaginary_axis.self_energy_matrix_imaginary_axis(
        mf, mf.mol, low.nocc, np.array([0.5]), coarse)
    assert calls == [coarse] * 3


def test_imagfrequency_energy_grid_is_shared(monkeypatch):
    """The imagfrequency E_c and the grid kept for E_c^A both come from
    `imaginary_axis_energy_grid`, on the run's numerics: one recipe, which
    `rpa_correlation_energy_imaginary_axis` and downfolding both call, so
    the E_c pair is one quadrature by construction."""
    from src.SingleReference.LinearResponse import rpa_energy
    mf = water()
    shared = rpa_energy.imaginary_axis_energy_grid
    calls = []

    def counted(eps, nocc, numerics=None, grid='minimax'):
        calls.append((numerics, grid))
        return shared(eps, nocc, numerics, grid)

    monkeypatch.setattr(rpa, 'imaginary_axis_energy_grid', counted)
    monkeypatch.setattr(rpa_energy, 'imaginary_axis_energy_grid', counted)
    numerics = Numerics(quadrature={'nfreq_rpa': 20})
    low = run_low_level(mf, method='rpa', mode='imagfrequency',
                        numerics=numerics)
    assert calls == [(numerics, 'minimax')] * 2   # E_c, then the kept grid
    points, weights = shared(low.eps, low.nocc, numerics)
    assert len(points) == 20
    assert np.array_equal(low.cache['grid'][0], points)
    assert np.array_equal(low.cache['grid'][1], weights)
