"""Solvent-screened Coulomb kernel v -> v + vtilde (Duchemin, Jacquemin and
Blase, J. Chem. Phys. 144, 164106 (2016), doi:10.1063/1.4946778).

src.Base.solvent_screening folds the solvent's reducible polarizability into
the Coulomb kernel (the paper's Eqs. (11)-(16)) and hands the result to the two
places this code gets two-electron integrals from, so every self-energy picks
it up at once.

Checks:
  1. Analytic limit: for a unit point charge at the centre of a SPHERICAL
     cavity the reaction potential is exactly -(1 - 1/eps)/a. All four PCM
     flavours must reproduce it -- the paper's own supplementary-material
     validation of vtilde_{beta beta'}.
  2. Vacuum invariant: eps = 1 leaves both the dense ERIs and the DF factor
     bit-identical (guarantees the substitution touches nothing it shouldn't).
  3. vtilde is symmetric and negative semi-definite in the pair space --
     screening can only lower the interaction.
  4. DF == dense: B -> T B reproduces the dense vtilde to within the RI error
     of the same B, and T = I when vtilde = 0.
  5. UHF: the same holds for the block-stacked spin-orbital factor.
  6. Static COHSEX reaction field: symmetric, occupied levels up / virtual
     levels down (the image-charge sign structure), so the IP falls and the EA
     rises.
  7. End to end: ADC(3) and GW IPs both drop, and the DF and dense routes
     agree on the shift.
  8. An eps large enough to be a *static* constant is rejected (the screening
     of an added electron or hole is optical).
  9. Route universality: the static reaction field reaches the imaginary-axis
     routes through static_exchange_matrix, and the space-time route screens
     its ISDF auxiliary metric, so all three GW modes (casida, imagfrequency,
     space-time) agree on the solvent shift of the IP.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import (get_two_electron_integrals_chemist,
                                      get_density_fitting_coefficients,
                                      get_uhf_spin_orbital_arrays_blockstacked,
                                      get_uhf_spin_orbital_df_factor_blockstacked)
from src.Base.solvent_screening import (SolventScreening, attach_solvent_screening,
                                        detach_solvent_screening,
                                        solvent_static_selfenergy,
                                        solvent_dielectrics)
from src.SingleReference.ADC import ADCSolver, build_static_correction
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.qp_solve import static_exchange_matrix
from src.Base.separable_ri import optimize_atomic_radii

from src.Base.constants import HARTREE_TO_EV
WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'


def check(ok, label, detail=''):
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def test_analytic_sphere():
    """Unit point charge at the centre of a sphere of radius a: the reaction
    potential it feels is -(1 - 1/eps)/a, the Born/conductor result, which is
    the exact solution of Poisson's equation for this geometry."""
    a = 4.0
    radii = np.full(120, a)
    mol = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
    source = gto.fakemol_for_charges(np.zeros((1, 3)), expnt=1e10)
    all_ok = True
    for method, tol in (('C-PCM', 1e-10), ('COSMO', 1e-10),
                        ('IEF-PCM', 1e-5), ('SS(V)PE', 1e-5)):
        for eps in (1.7764, 78.355):
            screening = SolventScreening(mol, eps=eps, method=method,
                                         radii_table=radii, allow_static_eps=True)
            v_grid = gto.mole.intor_cross('int2c2e', source, screening._fakemol())
            got = (v_grid @ screening.response_matrix() @ v_grid.T).item()
            # COSMO's f(eps) = (eps-1)/(eps+1/2) is a deliberate approximation
            # to the conductor limit, so it is checked against its own f.
            f = ((eps - 1) / (eps + 0.5) if method == 'COSMO' else 1 - 1 / eps)
            exact = -f / a
            err = abs(got / exact - 1)
            all_ok &= check(err < tol, f'analytic sphere {method} eps={eps}',
                            f'{got:.10f} vs {exact:.10f}, rel {err:.1e}')
    return all_ok


def test_vacuum_invariant(mol, mf):
    """eps = 1: no dielectric, so not one bit of any integral may move."""
    eri_gas = get_two_electron_integrals_chemist(mol, mf)
    B_gas = get_density_fitting_coefficients(mol, mf)
    attach_solvent_screening(mf, eps=1.0)
    d_eri = np.abs(get_two_electron_integrals_chemist(mol, mf) - eri_gas).max()
    d_B = np.abs(get_density_fitting_coefficients(mol, mf) - B_gas).max()
    detach_solvent_screening(mf)
    return (check(d_eri == 0.0, 'eps=1 leaves the dense ERIs bit-identical', f'{d_eri:.1e}')
            & check(d_B == 0.0, 'eps=1 leaves the DF factor bit-identical', f'{d_B:.1e}'))


def test_kernel_is_negative_semidefinite(mol, mf, eri_gas):
    attach_solvent_screening(mf, solvent='water')
    v_tilde = get_two_electron_integrals_chemist(mol, mf) - eri_gas
    detach_solvent_screening(mf)
    norb = eri_gas.shape[0]
    pair = v_tilde.reshape(norb * norb, norb * norb)
    asym = np.abs(pair - pair.T).max()
    scale = np.abs(pair).max()
    w_max = np.linalg.eigvalsh(pair).max()
    return (check(asym < 1e-12 * scale, 'vtilde is symmetric in the pair space', f'{asym:.1e}')
            & check(w_max < 1e-10 * scale, 'vtilde is negative semi-definite',
                    f'largest eigenvalue {w_max:.1e}, |vtilde|max {scale:.3f}'))


def test_df_matches_dense(mol, mf, eri_gas, B_gas):
    """B -> T B must carry the same vtilde the dense route adds, to within the
    RI error the same B already has on the bare interaction."""
    ri_error_bare = np.abs(np.einsum('Qpq,Qrs->pqrs', B_gas, B_gas) - eri_gas).max()
    attach_solvent_screening(mf, solvent='water')
    eri_scr = get_two_electron_integrals_chemist(mol, mf)
    B_scr = get_density_fitting_coefficients(mol, mf)
    transform = mf.with_screening.whitened_transform(mol, mf)
    detach_solvent_screening(mf)

    df_v_tilde = (np.einsum('Qpq,Qrs->pqrs', B_scr, B_scr)
                  - np.einsum('Qpq,Qrs->pqrs', B_gas, B_gas))
    diff = np.abs(df_v_tilde - (eri_scr - eri_gas)).max()
    ok = check(diff < ri_error_bare, 'DF vtilde matches dense vtilde within the RI error',
               f'{diff:.2e} vs RI error {ri_error_bare:.2e}')

    # T is the whitened screened Coulomb metric: symmetric, and its eigenvalues
    # lie in (0, 1] because screening only ever weakens the interaction.
    w = np.linalg.eigvalsh(transform)
    ok &= check(np.abs(transform - transform.T).max() < 1e-12, 'T is symmetric')
    ok &= check(0.0 < w.min() and w.max() < 1.0 + 1e-10,
                'T eigenvalues in (0, 1]', f'[{w.min():.4f}, {w.max():.4f}]')

    attach_solvent_screening(mf, eps=1.0)
    t_vac = mf.with_screening.whitened_transform(mol, mf)
    detach_solvent_screening(mf)
    d = np.abs(t_vac - np.eye(t_vac.shape[0])).max()
    return ok & check(d < 1e-12, 'T = I at eps = 1', f'{d:.1e}')


def test_uhf():
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='sto-3g', spin=1, verbose=0)
    mf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()
    _, g_gas, _ = get_uhf_spin_orbital_arrays_blockstacked(mol, mf)
    B_gas = get_uhf_spin_orbital_df_factor_blockstacked(mol, mf)
    attach_solvent_screening(mf, solvent='water')
    _, g_scr, _ = get_uhf_spin_orbital_arrays_blockstacked(mol, mf)
    B_scr = get_uhf_spin_orbital_df_factor_blockstacked(mol, mf)
    detach_solvent_screening(mf)

    def anti(B):
        return (np.einsum('Qpr,Qqs->pqrs', B, B)
                - np.einsum('Qps,Qqr->pqrs', B, B))

    err_gas = np.abs(anti(B_gas) - g_gas).max()
    err_scr = np.abs(anti(B_scr) - g_scr).max()
    shift = np.abs(g_scr - g_gas).max()
    return (check(shift > 1e-3, 'UHF: screening moves the spin-orbital integrals',
                  f'max |dg| = {shift:.3f}')
            & check(err_scr < 3 * max(err_gas, 1e-12),
                    'UHF: DF and dense screened tensors agree',
                    f'{err_scr:.1e} vs bare {err_gas:.1e}'))


def test_static_cohsex(mol, mf):
    """The first-order reaction field: image-charge sign structure."""
    attach_solvent_screening(mf, solvent='water')
    sigma = solvent_static_selfenergy(mf, mol)
    detach_solvent_screening(mf)
    nocc = mol.nelectron // 2
    diag = np.diag(sigma)
    ok = check(np.abs(sigma - sigma.T).max() < 1e-12, 'Sigma^solv is symmetric')
    ok &= check((diag[:nocc] > 0).all(), 'occupied levels shift up (IP falls)',
                f'{diag[:nocc].min() * HARTREE_TO_EV:+.2f} .. '
                f'{diag[:nocc].max() * HARTREE_TO_EV:+.2f} eV')
    ok &= check((diag[nocc:] < 0).all(), 'virtual levels shift down (EA rises)',
                f'{diag[nocc:].min() * HARTREE_TO_EV:+.2f} .. '
                f'{diag[nocc:].max() * HARTREE_TO_EV:+.2f} eV')
    return ok


def test_end_to_end(mol):
    """ADC(3) and GW must both report a lower IP in solvent, by the same amount
    through the DF and the dense route."""
    def ips(mf, df):
        ip_gw = -calc_qp_energy(mf, selfenergy='GW', df=df, state='homo')
        solver = ADCSolver(mf, level='adc3', df=df)
        static = build_static_correction(mf, kind='mp2_relaxed',
                                         B_aa=solver.B_aa if df else None)
        e_gf, _ = solver.solve(static_correction=static)
        return ip_gw, -e_gf[0] * HARTREE_TO_EV

    shifts = {}
    for tag, mf, df in (('DF', scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run(), True),
                        ('dense', scf.RHF(mol).run(), False)):
        gas = ips(mf, df)
        attach_solvent_screening(mf, solvent='water')
        sol = ips(mf, df)
        detach_solvent_screening(mf)
        shifts[tag] = (sol[0] - gas[0], sol[1] - gas[1])
        print(f'       {tag:5s}: IP(GW) {gas[0]:7.3f} -> {sol[0]:7.3f} eV, '
              f'IP(ADC3) {gas[1]:7.3f} -> {sol[1]:7.3f} eV')

    ok = check(all(s < -0.1 for pair in shifts.values() for s in pair),
               'solvent lowers both the GW and the ADC(3) IP')
    d_gw = abs(shifts['DF'][0] - shifts['dense'][0])
    d_adc = abs(shifts['DF'][1] - shifts['dense'][1])
    return ok & check(max(d_gw, d_adc) < 0.02,
                      'DF and dense agree on the shift',
                      f'GW {d_gw * 1000:.1f} meV, ADC(3) {d_adc * 1000:.1f} meV')


def test_all_gw_routes(mol):
    """Every GW route carries the solvent: same static reaction field through
    static_exchange_matrix, and the same v -> v + vtilde in Sigma_c (whitened
    DF congruence for the imaginary-frequency route, dressed ISDF metric for
    space-time), so the three modes must agree on the IP shift."""
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()

    # The mechanism itself: attaching the screening moves static_exchange_matrix
    # by exactly the static COHSEX operator.
    sx_gas = static_exchange_matrix(mf, mol)
    attach_solvent_screening(mf, solvent='water')
    sx_sol = static_exchange_matrix(mf, mol)
    sigma = solvent_static_selfenergy(mf, mol)
    detach_solvent_screening(mf)
    d = np.abs(sx_sol - sx_gas - sigma).max()
    ok = check(d < 1e-12, 'static_exchange_matrix picks up exactly Sigma^solv',
               f'{d:.1e}')

    # Optimize the ISDF radii once; the fit depends only on element/basis.
    auxbasis = str(mol.basis) + '-ri'
    radii = {el: optimize_atomic_radii(el, mol.basis, auxbasis)[0]
             for el in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})}

    def ip(mode):
        kw = {'radii': radii} if mode == 'space-time' else {}
        return -calc_qp_energy(mf, selfenergy='GW', df=True, state='homo',
                               mode=mode, **kw)

    shifts = {}
    for mode in ('casida', 'imagfrequency', 'space-time'):
        gas = ip(mode)
        attach_solvent_screening(mf, solvent='water')
        sol = ip(mode)
        detach_solvent_screening(mf)
        shifts[mode] = sol - gas
        print(f'       {mode:14s}: IP {gas:7.3f} -> {sol:7.3f} eV '
              f'(shift {sol - gas:+.3f})')

    ok &= check(all(s < -0.1 for s in shifts.values()),
                'every route reports a lower IP in solvent')
    d_imag = abs(shifts['imagfrequency'] - shifts['casida'])
    d_st = abs(shifts['space-time'] - shifts['casida'])
    return ok & check(max(d_imag, d_st) < 0.02,
                      'all three routes agree on the shift',
                      f'imagfrequency {d_imag * 1000:.1f} meV, '
                      f'space-time {d_st * 1000:.1f} meV vs casida')


def test_optical_eps_guard(mol):
    eps_opt, eps_static = solvent_dielectrics('water')
    ok = check(abs(eps_opt - 1.3328 ** 2) < 1e-9 and abs(eps_static - 78.355) < 1e-9,
               'water dielectrics from pyscf SMD table',
               f'optical {eps_opt:.4f}, static {eps_static:.3f}')
    try:
        SolventScreening(mol, eps=eps_static)
    except ValueError:
        ok &= check(True, 'a static eps is rejected unless forced')
    else:
        ok &= check(False, 'a static eps is rejected unless forced')
    try:
        SolventScreening(mol, eps=1.78, solvent='water')
    except ValueError:
        ok &= check(True, 'eps and solvent are mutually exclusive')
    else:
        ok &= check(False, 'eps and solvent are mutually exclusive')
    return ok


if __name__ == '__main__':
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()
    eri_gas = get_two_electron_integrals_chemist(mol, mf)
    B_gas = get_density_fitting_coefficients(mol, mf)

    all_ok = True
    print('\n-- 1. analytic spherical-cavity limit')
    all_ok &= test_analytic_sphere()
    print('\n-- 2. vacuum invariant')
    all_ok &= test_vacuum_invariant(mol, mf)
    print('\n-- 3. vtilde structure')
    all_ok &= test_kernel_is_negative_semidefinite(mol, mf, eri_gas)
    print('\n-- 4. DF route == dense route')
    all_ok &= test_df_matches_dense(mol, mf, eri_gas, B_gas)
    print('\n-- 5. UHF')
    all_ok &= test_uhf()
    print('\n-- 6. static COHSEX reaction field')
    all_ok &= test_static_cohsex(mol, mf)
    print('\n-- 7. end to end (ADC(3), GW)')
    all_ok &= test_end_to_end(mol)
    print('\n-- 8. optical-eps guards')
    all_ok &= test_optical_eps_guard(mol)
    print('\n-- 9. all GW routes carry the solvent')
    all_ok &= test_all_gw_routes(mol)

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
