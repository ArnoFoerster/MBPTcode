"""The RPA downfolding on the imagfrequency route against the casida route.

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
orbital active it is the full energy, so the E_c pair cancels to rounding.

Mean fields: water/6-31G RHF and PBE without density fitting (the
three-index object is then the exact eigen-fit of the MO integrals) and a
density-fitted RHF; the contiguous (4,4) active space, a non-contiguous one
and one without active virtual orbitals. Measured values in each test's
docstring.
"""
import dataclasses
from functools import cache

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.numerics.numerics import numerics_or_default, rpa_quadrature
from src.Base.orbital_spaces import ov_energies, particle_hole_window
from src.Base.utils.grids.grids import minimax_frequency_grid
from src.MultiReference.Downfolding import (build_effective_hamiltonian,
                                            run_low_level)
from src.MultiReference.Downfolding.energy import Energies
from src.MultiReference.Downfolding.methods import rpa
from src.MultiReference.Downfolding.partition import partition
from src.SingleReference.common.active_space import ActiveSpace
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import (
    rpa_correlation_energy_imaginary_axis, solve_polarizability_imaginary_axis)
from src.Solvers.active_space import solve
from tests.downfolding_oracle import h2, water

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
    part = partition(_aspace(mf, aspace), mf.mo_energy)
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
    part = partition(_aspace(mf, 'non_contiguous'), mf.mo_energy)
    b = low.factors
    b_act = rpa.active_pair_factors(low, part)
    assert b_act.shape == (b.shape[0], 4, 4)
    assert np.array_equal(b_act, b[:, part.act][:, :, part.act])
    p_a = rpa.active_polarizability_static(low, part)
    occ, vir = part.act[:2], part.act[2:]
    b_a = b[:, occ][:, :, vir].reshape(b.shape[0], -1)
    d_a = (low.eps[vir][None, :] - low.eps[occ][:, None]).ravel()
    assert np.abs(p_a - (-4.0 * (b_a / d_a) @ b_a.T)).max() < 1e-14
    all_act = partition(ActiveSpace.from_indices(len(mf.mo_energy), 5,
                                                 range(5), range(5, 13)),
                        mf.mo_energy)
    assert np.abs(rpa.active_polarizability_static(low, all_act)
                  - low.cache['p0']).max() < 1e-12


def test_naive_w_difference_is_wrong():
    """Screening is not additive: 'W^r' = W - W^A + v, W^A the RPA of the
    active pairs alone, misses the cRPA vtilde by 0.015 (contiguous (4,4)),
    while P - P^A and Dyson hit it."""
    low = _low('hf', 'imagfrequency')
    mf = low.mf
    part = partition(_aspace(mf, 'contiguous'), mf.mo_energy)
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
    part = partition(ActiveSpace.from_indices(4, 1, [0], [1, 2, 3]),
                     low.eps)
    t = rpa.terms(low, part, s)
    assert abs(t.e_c_full - t.e_c_active) < 1e-10
    assert np.abs(t.vtilde[0]).max() == 0.0


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
        if f.name != 'diagnostics':
            assert abs(getattr(en, f.name) - getattr(en_c, f.name)) < 1e-6
    assert abs(en.e0 - en_c.e0) < 1e-6
    assert np.abs(g[1] - g_c[1]).max() < 1e-8
    assert np.abs(g[2] - g_c[2]).max() < 1e-8
    assert abs(_solve(g, space) - _solve(g_c, space)) < 1e-6


@pytest.mark.parametrize('ref', ('hf', 'df_hf'))
def test_static_defect_zero_for_crpa(ref):
    """W(0) from the cache's P(0), projected on the active pairs, against
    the active dRPA of the fitted v + vtilde (R3, R4): zero for crpa
    (measured 1.1e-16 / 5.0e-16), and the casida route's defect for bare
    and crpa_ph (measured 3.9e-15)."""
    low_if, low_cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    mf = low_if.mf
    a = _aspace(mf, 'contiguous')
    part = partition(a, mf.mo_energy)
    _, _, en = build_effective_hamiltonian(low_if, a, screening='crpa')
    assert en.diagnostics['static_defect'] < 1e-8
    for s in ('bare', 'crpa_ph'):
        _, _, en = build_effective_hamiltonian(low_if, a, screening=s)
        _, _, en_c = build_effective_hamiltonian(low_cas, a, screening=s)
        assert en.diagnostics['static_defect'] > 1e-4
        assert abs(en.diagnostics['static_defect']
                   - en_c.diagnostics['static_defect']) < 1e-8
    assert rpa.static_defect(low_if, partition(
        ActiveSpace.from_indices(13, 5, [], []), mf.mo_energy),
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
        if f.name != 'diagnostics':
            assert abs(getattr(e1, f.name) - getattr(e2, f.name)) < 1e-12
    assert e1.diagnostics.keys() == e2.diagnostics.keys()


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


def test_mrpa_refused_here():
    """mRPA is casida only (spec §3.5); Sigma^E on imagfrequency is not built
    yet; both refusals point to casida. df=False has no meaning here (the
    route is density fitted) and space-time is not built."""
    low = _low('hf', 'imagfrequency')
    a = _aspace(low.mf, 'contiguous')
    with pytest.raises(NotImplementedError, match="mode='casida'"):
        build_effective_hamiltonian(low, a, screening='mrpa')
    for sigma in ('srg', 'mode_a'):
        with pytest.raises(NotImplementedError, match="mode='casida'"):
            build_effective_hamiltonian(low, a, sigma=sigma)
    with pytest.raises(ValueError, match="df=False.*mode='casida'"):
        run_low_level(low.mf, mode='imagfrequency', df=False)
    with pytest.raises(NotImplementedError, match="'imagfrequency'"):
        run_low_level(low.mf, mode='space-time')
