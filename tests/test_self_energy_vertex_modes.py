"""The diagonal self-energy serves the vertex modes it can build and refuses the rest.

RHF and UHF give one quasiparticle energy on a closed shell for every mode
(water/6-31G, tight SCF). For the triplet modes PSD2/4/7/9 that holds because
both paths count every triplet component of the spin-orbital eh pair: at fixed
omega, bare vertex and TDHF excitons, RHF and UHF PSD2 equal the eh pair built
from spin-orbital TDHF (Faddeev restricted to the eh channel). The batch
self-energy has no triplet sum and refuses PSD2/4/7/9 by name; both paths
refuse a vertex mode whose amplitudes are missing instead of returning GW.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import METHOD_REGISTRY
from src.Base.pyscf_interface import (get_antisymmetrized_spin_eri,
                                      get_density_fitting_coefficients,
                                      get_orbital_energies,
                                      get_two_electron_integrals_chemist)
from src.SingleReference.GW import qp_energy
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.self_energy import (TRIPLET_VERTEX_MODES,
                                                SelfEnergySolver)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
SINGLET_ONLY = [m for m, info in METHOD_REGISTRY.items()
                if m in ('GW', 'GWGammaInf') or (m.startswith('PSD')
                                                 and not info['needs_triplet'])]


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    rhf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    rhf.conv_tol, rhf.conv_tol_grad = 1e-13, 1e-9
    rhf.kernel()
    uhf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    uhf.conv_tol, uhf.conv_tol_grad = 1e-13, 1e-9
    dm = rhf.make_rdm1()
    uhf.kernel(np.array([dm, dm]) / 2)
    return mol, rhf, uhf


def test_rhf_and_uhf_agree_on_a_closed_shell(water):
    mol, rhf, uhf = water
    nocc = mol.nelectron // 2
    states = [nocc - 1, nocc]
    kw = dict(selfenergy=SINGLET_ONLY, polarizability='TDHF', mode='casida',
              df=True, state=states)
    r = calc_qp_energy(rhf, **kw)
    u = calc_qp_energy(uhf, spin_channel='alpha', **kw)
    for p in states:
        for m in SINGLET_ONLY:
            assert abs(r[p][m] - u[p][m]) < 1e-7, (p, m, r[p][m], u[p][m])


@pytest.mark.parametrize('polarizability', ['TDHF', 'BSE'])
def test_rhf_and_uhf_agree_on_the_triplet_modes(water, polarizability):
    mol, rhf, uhf = water
    nocc = mol.nelectron // 2
    states = [nocc - 1, nocc]
    kw = dict(selfenergy=list(TRIPLET_VERTEX_MODES), polarizability=polarizability,
              mode='casida', df=True, state=states)
    r = calc_qp_energy(rhf, **kw)
    for spin_channel in ('alpha', 'beta'):
        u = calc_qp_energy(uhf, spin_channel=spin_channel, **kw)
        for p in states:
            for m in TRIPLET_VERTEX_MODES:
                assert abs(r[p][m] - u[p][m]) < 1e-7, (spin_channel, p, m, r[p][m], u[p][m])


@pytest.fixture(scope='module')
def water_exact():
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    rhf = scf.RHF(mol)
    rhf.conv_tol, rhf.conv_tol_grad = 1e-13, 1e-9
    rhf.kernel()
    uhf = scf.UHF(mol)
    uhf.conv_tol, uhf.conv_tol_grad = 1e-13, 1e-9
    dm = rhf.make_rdm1()
    uhf.kernel(np.array([dm, dm]) / 2)
    return mol, rhf, uhf


def _eh_pair_spin_orbital(eps, eri, nocc):
    """Sigma_pp(w) of the eh pair over spin-orbital TDHF: every singlet and all three triplet components."""
    g = get_antisymmetrized_spin_eri(eri)
    es = np.repeat(eps, 2)
    no = 2 * nocc
    nv = len(es) - no
    n = no * nv
    o, v = slice(0, no), slice(no, None)
    A = g[v, o, o, v].transpose(2, 0, 1, 3).reshape(n, n).copy()
    A[np.diag_indices(n)] += (es[None, v] - es[o, None]).ravel()
    B = g[v, v, o, o].transpose(2, 0, 3, 1).reshape(n, n).copy()
    om, X, Y = CasidaSolver(A, B).solve()
    X, Y = X.reshape(no, nv, -1), Y.reshape(no, nv, -1)

    def sigma(p, w):
        P = 2 * p
        M = (np.einsum('iab,ian->bn', g[P, o, v, v], X)
             + np.einsum('jcb,jcn->bn', g[P, v, o, v].transpose(1, 0, 2), Y))
        s = np.sum(M**2 / (w - es[v][:, None] - om[None, :]))
        M = (np.einsum('iaj,ian->jn', g[P, v, o, o].transpose(1, 0, 2), X)
             + np.einsum('kcj,kcn->jn', g[v, o, P, o].transpose(2, 0, 1), Y))
        return s + np.sum(M**2 / (w - es[o][:, None] + om[None, :]))
    return sigma


def test_psd2_is_the_spin_orbital_eh_pair(water_exact):
    mol, rhf, uhf = water_exact
    nocc = mol.nelectron // 2
    eps = rhf.mo_energy
    eri = get_two_electron_integrals_chemist(mol, rhf, representation='spatial')
    reference = _eh_pair_spin_orbital(eps, eri, nocc)

    lr = LinearResponseSolver(eps, eri_chemist=eri)
    se = SelfEnergySolver(eps, eri_chemist=eri, eta=1e-12)
    om_s, X_s, Y_s = CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=True, triplet=False)).solve()
    om_t, X_t, Y_t = CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=True, triplet=True)).solve()

    _, eri_u = qp_energy._two_electron_integrals(mol, uhf, False, True)
    eps_u = tuple(uhf.mo_energy)
    lr_u = LinearResponseSolver(eps_u, eri_chemist=eri_u, spin_mode='unrestricted')
    se_u = SelfEnergySolver(eps_u, eri_chemist=eri_u, spin_mode='unrestricted', eta=1e-12)
    infos = {'PSD2': METHOD_REGISTRY['PSD2']}
    spectrum = qp_energy._casida_spectrum(lr_u, uhf.nelec, 'TDHF', None, False, infos,
                                          ['PSD2'], True, False)
    for p in (nocc - 1, nocc):
        om_u, chi_a_u, chi_b_u, om_t_u, chi_b_t_u = qp_energy._self_energy_amplitudes(
            se_u, uhf.nelec, spectrum, infos, ['PSD2'], 'alpha', p, None, None, True,
            False)['PSD2']
        for dw in (-0.35, 0.20):
            w = eps[p] + dw
            ref = reference(p, w)
            s_r = se.calculate_self_energy(
                p, w, nocc, om_s, se.get_chi_a(nocc, X_s, Y_s),
                se.get_chi_b_vertex(nocc, X_s, Y_s), om_t,
                se.get_chi_b_vertex(nocc, X_t, Y_t), vertex_mode='PSD2')
            s_u = se_u.calculate_self_energy(
                p, w, uhf.nelec, om_u, chi_a_u, chi_b_u, eigenvalues_casida_t=om_t_u,
                chiXYb_t=chi_b_t_u, spin_channel='alpha', vertex_mode='PSD2')
            assert abs(s_r - ref) < 1e-10, (p, dw, s_r, ref)
            assert abs(s_u - ref) < 1e-7, (p, dw, s_u, ref)


@pytest.mark.parametrize('polarizability', ['TDHF', 'BSE'])
def test_rhf_and_uhf_agree_on_the_triplet_modes_without_df(water_exact, polarizability):
    mol, rhf, uhf = water_exact
    nocc = mol.nelectron // 2
    states = [nocc - 1, nocc]
    kw = dict(selfenergy=list(TRIPLET_VERTEX_MODES), polarizability=polarizability,
              mode='casida', df=False, state=states)
    r = calc_qp_energy(rhf, **kw)
    u = calc_qp_energy(uhf, spin_channel='alpha', **kw)
    for p in states:
        for m in TRIPLET_VERTEX_MODES:
            assert abs(r[p][m] - u[p][m]) < 1e-6, (p, m, r[p][m], u[p][m])


@pytest.fixture(scope='module')
def batch(water):
    mol, rhf, _ = water
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(rhf, representation='spatial')
    coeff = get_density_fitting_coefficients(mol, rhf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted')
    se = SelfEnergySolver(eps, df_coeff=coeff, spin_mode='restricted')
    omega, X, Y = CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=False)).solve()
    chi_a = se.get_chi_a(nocc, X, Y, p_state=None)
    chi_b = se.get_chi_b_vertex(nocc, X, Y, p_state=None)
    return se, nocc, eps, omega, chi_a, chi_b


def test_the_triplet_modes_are_the_registry_ones():
    assert TRIPLET_VERTEX_MODES == ('PSD2', 'PSD4', 'PSD7', 'PSD9')


@pytest.mark.parametrize('mode', ['PSD2', 'PSD4', 'PSD7', 'PSD9', 'PSD3'])
def test_the_batch_refuses_what_it_cannot_build(batch, mode):
    se, nocc, eps, omega, chi_a, chi_b = batch
    with pytest.raises(ValueError, match=mode):
        se.calculate_self_energy_diagonal_batch(eps, nocc, omega, chi_a, chi_b,
                                                vertex_mode=mode)


def test_a_vertex_mode_without_its_amplitudes_is_refused(batch):
    se, nocc, eps, omega, chi_a, chi_b = batch
    with pytest.raises(ValueError, match='chiXYb'):
        se.calculate_self_energy_diagonal_batch(eps, nocc, omega, chi_a,
                                                vertex_mode='PSD1')
    with pytest.raises(ValueError, match='chiXYb'):
        se.calculate_self_energy(0, eps[0], nocc, omega, chi_a,
                                 vertex_mode='GWGammaInf')
    with pytest.raises(ValueError, match='triplet'):
        se.calculate_self_energy(0, eps[0], nocc, omega, chi_a, chi_b,
                                 vertex_mode='PSD2')


@pytest.mark.parametrize('mode', ['GW', 'GWGammaInf', 'PSD1', 'PSD5', 'PSD6',
                                  'PSD8'])
def test_the_batch_matches_the_scalar_on_what_it_serves(batch, mode):
    se, nocc, eps, omega, chi_a, chi_b = batch
    freq = eps + 0.01
    b = se.calculate_self_energy_diagonal_batch(freq, nocc, omega, chi_a, chi_b,
                                                vertex_mode=mode)
    s = [se.calculate_self_energy(p, freq[p], nocc, omega, chi_a, chi_b,
                                  vertex_mode=mode) for p in range(len(eps))]
    assert np.allclose(b, s, rtol=0, atol=1e-12)
