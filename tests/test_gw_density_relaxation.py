"""The GW density's keywords (`mode`, `relaxation`), its spin factors, its
static gKS term and the J/K backend of its HF-kernel ('dyson') relaxation.

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

Water/cc-pVDZ (DF on an explicit auxiliary basis) and water/6-31G (exact
integrals) for the dense check.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import dft, gto, scf

from src.Base import numerical_flags
from src.Base.numerics.constants import DEFAULT_BROADENING_ETA
from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.Base.factorization.isdf_jk import isdf_jk
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.SingleReference.DensityMatrix.density_matrix import hf_kernel_mean_field
from src.SingleReference.GW.density_matrix import (GWDensityMatrixSolver,
                                                    compute_gw_density_matrix,
                                                    static_ov_term)
from tests.molecules import WATER

AUXBASIS = 'cc-pvdz-ri'


def water(basis):
    return gto.M(atom=WATER, basis=basis, verbose=0)


def tight(mf):
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
    return mf.run()


def fitted(mf):
    mf = mf.density_fit(auxbasis=AUXBASIS)
    return tight(mf)


def mo_dgamma(mf, dm):
    """C^T S (dm - gamma_mf) S C: the MO-basis density correction."""
    S = mf.mol.intor('int1e_ovlp')
    C = mf.mo_coeff
    return C.T @ S @ (dm - mf.make_rdm1()) @ S @ C


def report(name, ok, detail=''):
    print(f"{name}: {'OK' if ok else 'FAIL'} {detail}")
    return ok


def check_static_term_is_hf_fock_gradient(ks):
    mol = ks.mol
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(ks, representation='spatial')
    C = ks.mo_coeff
    F_hf = C.T @ hf_kernel_mean_field(ks).get_fock(dm=ks.make_rdm1()) @ C
    ref = F_hf[:nocc, nocc:] / (eps[:nocc, None] - eps[None, nocc:])
    got = static_ov_term(ks, mol, eps, nocc)
    err = np.abs(got - ref).max()
    return report('static ov term == F_HF[gamma_KS]_ia / (e_i - e_a)',
                  err < 1e-10 and np.abs(ref).max() > 1e-4,
                  f'(max |diff| {err:.2e}, max |term| {np.abs(ref).max():.2e})')


def check_spin_factor(ks):
    mol = ks.mol
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(ks, representation='spatial')
    solver = GWDensityMatrixSolver(
        eps, df_coeff=get_density_fitting_coefficients(mol, ks, representation='spatial'),
        eta=DEFAULT_BROADENING_ETA)
    omega, X, Y = solver.build_screening(nocc)
    oo, ov, vv = solver.compute_unrelaxed_blocks(nocc, ks, mol, omega=omega, X=X, Y=Y)
    static = static_ov_term(ks, mol, eps, nocc)

    # The dynamic ov term written out with the raw (un-normalized) w of
    # get_chi_a; the restricted w-normalization 2 is explicit here.
    w = solver.get_chi_a(nocc, X, Y, p_state=None)
    o, v = slice(0, nocc), slice(nocc, None)
    ei, ea = eps[o], eps[v]
    t_ic = w[:, o, v] / (ei[None, :, None] - ea[None, None, :] - omega[:, None, None])
    t_ak = w[:, v, o] / (ea[None, :, None] - ei[None, None, :] + omega[:, None, None])
    dyn_raw = (np.einsum('sik,sbk->ib', w[:, o, o], t_ak)
               + np.einsum('sic,sbc->ib', t_ic, w[:, v, v])) / (ei[:, None] - ea[None, :])

    total = mo_dgamma(ks, compute_gw_density_matrix(ks, mol, relaxation='none'))
    ok = True
    ok &= report('spatial ov == 2 * per-channel ov',
                 np.allclose(total[o, v], 2 * ov, atol=1e-10, rtol=0))
    ok &= report('spatial oo == 2 * per-channel oo',
                 np.allclose(total[o, o], 2 * oo, atol=1e-10, rtol=0))
    ok &= report('spatial vv == 2 * per-channel vv',
                 np.allclose(total[v, v], 2 * vv, atol=1e-10, rtol=0))
    ok &= report('per-channel dynamic ov == 2 * raw-w dynamic ov',
                 np.allclose(ov - static, 2 * dyn_raw, atol=1e-10, rtol=0))
    static_in_total = total[o, v] - 2 * (2 * dyn_raw)
    err2 = np.abs(static_in_total - 2 * static).max()
    err4 = np.abs(static_in_total - 4 * static).max()
    ok &= report('static term enters the spatial ov at 2x (not 4x)',
                 err2 < 1e-10 and err4 > 1e-4,
                 f'(|vs 2x| {err2:.2e}, |vs 4x| {err4:.2e})')
    return ok


def check_dyson_equals_zvector_on_hf(hf):
    numerics = Numerics()
    dyson = compute_gw_density_matrix(hf, df=False, relaxation='dyson', numerics=numerics)
    zvec = compute_gw_density_matrix(hf, df=False, relaxation='zvector', numerics=numerics)
    err = np.abs(dyson - zvec).max()
    return report("HF, exact integrals: 'dyson' == 'zvector'", err < 1e-8,
                  f'(max |diff| {err:.2e})')


def check_dense_eq23(ks):
    """Eq. 23 of the iterated-Dyson paper, real restricted orbitals,
    spin-summed Delta gamma, solved densely."""
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
    ok = report("PBE0: dense Eq. 23 == 'dyson' ov block", err < 1e-7,
                f'(max |diff| {err:.2e})')
    ok &= report("PBE0: dense Eq. 23 differs from 'zvector'", gap > 1e-5,
                 f'(max |diff| {gap:.2e})')
    return ok


def check_dyson_backend(hf_df):
    numerics = Numerics()
    ok = report('HF-kernel object shares the DF object',
                hf_kernel_mean_field(hf_df).with_df is hf_df.with_df)
    dyson = compute_gw_density_matrix(hf_df, relaxation='dyson', numerics=numerics)
    zvec = compute_gw_density_matrix(hf_df, relaxation='zvector', numerics=numerics)
    err = np.abs(dyson - zvec).max()
    ok &= report("HF, density fitted: 'dyson' == 'zvector'", err < 1e-8,
                 f'(max |diff| {err:.2e})')

    # An ISDF mean field: the HF kernel is its own interpolated J - K/2.
    mol = hf_df.mol
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', ['O', 'H'], auxbasis=AUXBASIS)
    isdf = isdf_jk(scf.RHF(mol), auxbasis=AUXBASIS, counts=counts, n_start=n_start)
    tight(isdf)
    hf = hf_kernel_mean_field(isdf)
    ok &= report('HF-kernel object shares the ISDF J/K object', hf.with_df is isdf.with_df)
    rng = np.random.default_rng(7)
    dm = rng.standard_normal((mol.nao, mol.nao))
    dm = dm + dm.T
    vj, vk = isdf.get_jk(mol, dm, hermi=1)
    err = np.abs(hf.gen_response(hermi=1)(dm) - (vj - 0.5 * vk)).max()
    ok &= report('ISDF HF-kernel response == J - K/2 of the ISDF mean field', err < 1e-10,
                 f'(max |diff| {err:.2e})')
    return ok


def check_dyson_on_pbe_flags_or_converges(pbe):
    flags = []
    with numerical_flags.collecting(flags):
        dm = compute_gw_density_matrix(pbe, relaxation='dyson', numerics=Numerics())
    names = [f['flag'] for f in flags]
    if 'orbital_response_not_converged' in names:
        return report("PBE 'dyson': solve flagged not converged", True)
    n = np.trace(dm @ pbe.mol.intor('int1e_ovlp'))
    return report("PBE 'dyson': converged, finite, Tr(gamma S) = N",
                  not names and np.all(np.isfinite(dm))
                  and abs(n - pbe.mol.nelectron) < 1e-6,
                  f'(flags {names}, N {n:.10f})')


def raises(exc, words, call):
    try:
        call()
    except exc as err:
        return all(w in str(err) for w in words)
    return False


def check_keyword_values(hf):
    numerics = Numerics()
    ok = report("relaxation='Dyson' == 'dyson'", np.array_equal(
        compute_gw_density_matrix(hf, relaxation='Dyson', numerics=numerics),
        compute_gw_density_matrix(hf, relaxation='dyson', numerics=numerics)))
    ok &= report("relaxation='relaxed' refused with the choices", raises(
        ValueError, ['none', 'dyson', 'zvector'],
        lambda: compute_gw_density_matrix(hf, relaxation='relaxed', numerics=numerics)))
    ok &= report("mode='casida2' refused with the choices", raises(
        ValueError, ['casida', 'imagfrequency', 'space-time'],
        lambda: compute_gw_density_matrix(hf, mode='casida2', numerics=numerics)))
    dm = compute_gw_density_matrix(hf, mode='casida', relaxation='none', numerics=None)
    ok &= report("mode='casida', relaxation='none' runs without numerics",
                 dm.shape == (hf.mol.nao, hf.mol.nao))
    ok &= report("auxbasis with mode='casida' refused", raises(
        ValueError, ['auxbasis'],
        lambda: compute_gw_density_matrix(hf, mode='casida', auxbasis=AUXBASIS,
                                          numerics=numerics)))
    ok &= report("screen_r_cut with mode='casida' refused", raises(
        ValueError, ['screen_r_cut'],
        lambda: compute_gw_density_matrix(hf, mode='casida', screen_r_cut=5.0,
                                          numerics=numerics)))
    ok &= report('relax= is gone', raises(
        TypeError, [], lambda: compute_gw_density_matrix(hf, relax=True, numerics=numerics)))
    return ok


def run():
    hf_df = fitted(scf.RHF(water('cc-pvdz')))
    pbe0 = fitted(dft.RKS(water('cc-pvdz'), xc='pbe0'))
    pbe = fitted(dft.RKS(water('cc-pvdz'), xc='pbe'))
    hf_exact = tight(scf.RHF(water('cc-pvdz')))
    pbe0_small = tight(dft.RKS(water('6-31g'), xc='pbe0'))

    results = [
        check_static_term_is_hf_fock_gradient(pbe0),
        check_spin_factor(pbe0),
        check_dyson_equals_zvector_on_hf(hf_exact),
        check_dense_eq23(pbe0_small),
        check_dyson_backend(hf_df),
        check_dyson_on_pbe_flags_or_converges(pbe),
        check_keyword_values(hf_df),
    ]
    return all(results)


def test_gw_density_relaxation():
    assert run()


if __name__ == '__main__':
    ok = run()
    print("\nALL PASSED" if ok else "\nFAILURES DETECTED")
    sys.exit(0 if ok else 1)
