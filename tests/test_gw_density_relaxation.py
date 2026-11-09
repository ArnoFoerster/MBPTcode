"""The GW density's spin factors, its static gKS term and the J/K backend
of its HF-kernel ('dyson') relaxation. Its keywords are in
test_gw_density_keywords.

- The static ov term is the HF Fock gradient at the gKS density over the gap:
  <i|Sigma_x - v_xc|a> = F_HF[gamma_KS]_ia, since F_KS_ia = 0.
- Spin factors: per channel the dynamic terms carry the restricted
  w-normalization 2 and the static term none; the spatial density is the spin
  sum 2 of the per-channel blocks, so the static term enters it at 2x (it was
  4x before the fix).
- 'dyson' and 'zvector' coincide on an HF reference (the HF kernel is the
  reference's own), on exact and on density-fitted integrals: the HF-kernel
  object shares the mean field's J/K backend.
- 'dyson' on a PBE0 reference is Eq. 23 of the iterated-Dyson paper, checked
  against a dense solve, and differs from 'zvector' there.
- The HF-kernel object refuses a reference whose Hamiltonian it cannot carry
  (UHF, open-shell ROHF, an environment, a thermal smearing) rather than
  dropping part of it.

Water/cc-pVDZ (DF on an explicit auxiliary basis) and water/6-31G (exact
integrals) for the dense check.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy

import numpy as np
import pytest
from pyscf import dft, scf

from src.Base import numerical_flags
from src.Base.environment_seam import attach_environment
from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.Base.factorization.isdf_jk import isdf_jk
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.SingleReference.DensityMatrix.density_matrix import hf_kernel_mean_field
from src.SingleReference.GW.density_matrix import (casida_screening,
                                                    casida_unrelaxed_blocks,
                                                    compute_gw_density_matrix,
                                                    screened_coulomb_residues,
                                                    static_ov_term_unrelaxed)
from src.environments.solvent_screening import SolventScreening
from tests.gw_density_routes import AUXBASIS, converged, fitted, mo_dgamma, water


@pytest.fixture(scope='module')
def hf_df():
    return fitted(scf.RHF(water()))


@pytest.fixture(scope='module')
def pbe0():
    return fitted(dft.RKS(water(), xc='pbe0'))


def test_static_term_is_hf_fock_gradient(pbe0):
    """<i|Sigma_x - v_xc|a> / (e_i - e_a) == F_HF[gamma_KS]_ia / (e_i - e_a)."""
    ks = pbe0
    mol = ks.mol
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(ks, representation='spatial')
    C = ks.mo_coeff
    F_hf = C.T @ hf_kernel_mean_field(ks).get_fock(dm=ks.make_rdm1()) @ C
    ref = F_hf[:nocc, nocc:] / (eps[:nocc, None] - eps[None, nocc:])
    got = static_ov_term_unrelaxed(ks, mol, eps, nocc)
    err, size = np.abs(got - ref).max(), np.abs(ref).max()
    print(f'static ov term: max |diff| {err:.2e}, max |term| {size:.2e}')
    assert size > 1e-4, 'the static term vanishes: nothing tested'
    assert err < 1e-10


def test_spin_factor(pbe0):
    """The spatial density is 2x the per-channel blocks; per channel the
    dynamic ov carries the w-normalization 2, and the static term enters the
    spatial ov at 2x (not 4x, as before the fix)."""
    ks = pbe0
    mol = ks.mol
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(ks, representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, ks, representation='spatial')
    omega, X, Y = casida_screening(eps, nocc, df_coeff=df_coeff)
    oo, ov, vv = casida_unrelaxed_blocks(ks, mol, eps, nocc, screening=(omega, X, Y))
    static = static_ov_term_unrelaxed(ks, mol, eps, nocc)

    # The dynamic ov term written out with the raw (un-normalized) w of
    # get_chi_a; the restricted w-normalization 2 is explicit here.
    w = screened_coulomb_residues(eps, nocc, X, Y, df_coeff=df_coeff)
    o, v = slice(0, nocc), slice(nocc, None)
    ei, ea = eps[o], eps[v]
    t_ic = w[:, o, v] / (ei[None, :, None] - ea[None, None, :] - omega[:, None, None])
    t_ak = w[:, v, o] / (ea[None, :, None] - ei[None, None, :] + omega[:, None, None])
    dyn_raw = (np.einsum('sik,sbk->ib', w[:, o, o], t_ak)
               + np.einsum('sic,sbc->ib', t_ic, w[:, v, v])) / (ei[:, None] - ea[None, :])

    total = mo_dgamma(ks, compute_gw_density_matrix(ks, mol, relaxation='none'))
    for block, sl, per_channel in (('ov', (o, v), ov), ('oo', (o, o), oo),
                                   ('vv', (v, v), vv)):
        assert np.allclose(total[sl], 2 * per_channel, atol=1e-10, rtol=0), \
            f'spatial {block} != 2 * per-channel {block}'
    assert np.allclose(ov - static, 2 * dyn_raw, atol=1e-10, rtol=0), \
        'per-channel dynamic ov != 2 * raw-w dynamic ov'
    static_in_total = total[o, v] - 2 * (2 * dyn_raw)
    err2 = np.abs(static_in_total - 2 * static).max()
    err4 = np.abs(static_in_total - 4 * static).max()
    print(f'static term in the spatial ov: |vs 2x| {err2:.2e}, |vs 4x| {err4:.2e}')
    assert err2 < 1e-10 and err4 > 1e-4


def test_dyson_equals_zvector_on_hf_exact_integrals():
    hf = converged(scf.RHF(water()))
    numerics = Numerics()
    dyson = compute_gw_density_matrix(hf, df=False, relaxation='dyson', numerics=numerics)
    zvec = compute_gw_density_matrix(hf, df=False, relaxation='zvector', numerics=numerics)
    err = np.abs(dyson - zvec).max()
    print(f"HF, exact integrals: max |'dyson' - 'zvector'| {err:.2e}")
    assert err < 1e-8


def test_dense_eq23():
    """Eq. 23 of the iterated-Dyson paper, real restricted orbitals,
    spin-summed Delta gamma, solved densely: 'dyson' on PBE0 (water/6-31G,
    exact integrals) reproduces it and differs from 'zvector'."""
    ks = converged(dft.RKS(water('6-31g'), xc='pbe0'))
    mol = ks.mol
    nocc = mol.nelectron // 2
    numerics = Numerics()
    eps = get_orbital_energies(ks, representation='spatial')
    nmo = len(eps)
    o, v = slice(0, nocc), slice(nocc, None)
    nv = nmo - nocc
    eri = get_two_electron_integrals_chemist(mol, ks, representation='spatial')

    g = mo_dgamma(ks, compute_gw_density_matrix(ks, df=False, relaxation='none'))
    dyson = mo_dgamma(ks, compute_gw_density_matrix(ks, df=False, relaxation='dyson',
                                                    numerics=numerics))
    zvec = mo_dgamma(ks, compute_gw_density_matrix(ks, df=False, relaxation='zvector',
                                                   numerics=numerics))

    e_ia = eps[v][None, :] - eps[o][:, None]
    A = (4 * eri[o, v, o, v]
         - eri[o, o, v, v].transpose(0, 2, 1, 3)
         - eri[o, v, o, v].transpose(0, 3, 2, 1))
    A = A.reshape(nocc * nv, nocc * nv) + np.diag(e_ia.ravel())
    g_mn = np.zeros_like(g)
    g_mn[o, o], g_mn[v, v] = g[o, o], g[v, v]
    coupling = (np.einsum('iamn,mn->ia', eri[o, v, :, :], g_mn)
                - 0.5 * np.einsum('iman,mn->ia', eri[o, :, v, :], g_mn))
    Yv = e_ia * g[o, v] - 2 * coupling
    x = np.linalg.solve(A, Yv.ravel()).reshape(nocc, nv)

    err = np.abs(x - dyson[o, v]).max()
    gap = np.abs(x - zvec[o, v]).max()
    print(f"PBE0: max |Eq. 23 - 'dyson'| {err:.2e}, max |Eq. 23 - 'zvector'| {gap:.2e}")
    assert err < 1e-7, "dense Eq. 23 != 'dyson' ov block"
    assert gap > 1e-5, "dense Eq. 23 does not differ from 'zvector'"


def test_dyson_backend(hf_df):
    """The HF-kernel object keeps the mean field's J/K backend: exact _eri,
    the DF object, the ISDF J/K; 'dyson' == 'zvector' on density-fitted HF."""
    exact = scf.RHF(hf_df.mol).run()
    assert hf_kernel_mean_field(exact)._eri is exact._eri
    assert hf_kernel_mean_field(hf_df).with_df is hf_df.with_df
    numerics = Numerics()
    dyson = compute_gw_density_matrix(hf_df, relaxation='dyson', numerics=numerics)
    zvec = compute_gw_density_matrix(hf_df, relaxation='zvector', numerics=numerics)
    err = np.abs(dyson - zvec).max()
    print(f"HF, density fitted: max |'dyson' - 'zvector'| {err:.2e}")
    assert err < 1e-8

    # An ISDF mean field: the HF kernel is its own interpolated J - K/2.
    mol = hf_df.mol
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', ['O', 'H'], auxbasis=AUXBASIS)
    isdf = isdf_jk(scf.RHF(mol), auxbasis=AUXBASIS, counts=counts, n_start=n_start)
    converged(isdf)
    hf = hf_kernel_mean_field(isdf)
    assert hf.with_df is isdf.with_df
    rng = np.random.default_rng(7)
    dm = rng.standard_normal((mol.nao, mol.nao))
    dm = dm + dm.T
    vj, vk = isdf.get_jk(mol, dm, hermi=1)
    err = np.abs(hf.gen_response(hermi=1)(dm) - (vj - 0.5 * vk)).max()
    print(f'ISDF: max |HF-kernel response - (J - K/2)| {err:.2e}')
    assert err < 1e-10


def test_dyson_on_pbe_flags_or_converges():
    """On PBE (no exact exchange) the 'dyson' solve either flags itself not
    converged or converges to a finite density with Tr(gamma S) = N."""
    pbe = fitted(dft.RKS(water(), xc='pbe'))
    flags = []
    with numerical_flags.collecting(flags):
        dm = compute_gw_density_matrix(pbe, relaxation='dyson', numerics=Numerics())
    names = [f['flag'] for f in flags]
    if 'orbital_response_not_converged' in names:
        print("PBE 'dyson': solve flagged not converged")
        return
    n = np.trace(dm @ pbe.mol.intor('int1e_ovlp'))
    print(f"PBE 'dyson': flags {names}, N {n:.10f}")
    assert not names
    assert np.all(np.isfinite(dm))
    assert abs(n - pbe.mol.nelectron) < 1e-6


def _uhf(mf):
    return scf.UHF(mf.mol)


def _rohf_cation(mf):
    cation = mf.mol.copy()
    cation.charge, cation.spin = 1, 1
    cation.build()
    return scf.ROHF(cation)


def _solvated(mf):
    return attach_environment(copy.copy(mf),
                              SolventScreening(mf.mol, eps=2.0, eps_static=2.0))


def _smeared(mf):
    smeared = copy.copy(mf)
    smeared.sigma, smeared.smearing_method = 0.01, 'fermi'
    return smeared


@pytest.mark.parametrize('make, word', [(_uhf, 'restricted'),
                                        (_rohf_cation, 'closed-shell'),
                                        (_solvated, 'gas phase'),
                                        (_smeared, 'T = 0')])
def test_hf_kernel_refuses_out_of_scope_references(hf_df, make, word):
    with pytest.raises(NotImplementedError) as err:
        hf_kernel_mean_field(make(hf_df))
    print(f'{make.__name__}: {err.value}')
    assert word in str(err.value)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-v']))
