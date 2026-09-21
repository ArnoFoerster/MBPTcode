"""qsGW and qsGW0: the orbitals and the eigenvalues reinjected until the static
Hermitian self-energy stops moving.

Gate (a): the blocked static self-energy equals the dense matrix routine and,
on its diagonal, the per-state self-energy at eps_p; its SRG-regularized form,
the one the loop runs, equals a dense evaluation of its formula. Gate (b): PBE
and PBE0 starts land on one fixed point. Gate (e): the converged point is a
fixed point of the evGW0 eigenvalue map, in one step. The two mixings land on
one fixed point.

Run: python tests/test_qsgw.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import df, dft, gto

from src.Base.constants import (EVGW_TOL, HARTREE_TO_EV, QSGW_SRG_FLOW,
                                get_method_info)
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.SingleReference.GW.evGW import evgw_eigenvalues, rotated_mean_field
from src.SingleReference.GW.qsGW import qsgw_eigenvalues
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
import src.SingleReference.GW.qp_energy as qpe

GEOMETRY = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}"
          + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build_reference(xc='pbe0'):
    """Water, cc-pVDZ, exact-JK SCF, the RI set attached for W and Sigma."""
    mol = gto.M(atom=GEOMETRY, basis='cc-pvdz', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.kernel()
    mf.with_df = df.DF(mol, auxbasis='cc-pvdz-ri')
    return mf


def mean_field_spectrum(mf):
    """(eps, coeff, nocc, omega, X, Y): the RPA Casida solution of `mf`."""
    mol = mf.mol
    nocc = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted')
    spectrum = qpe._casida_spectrum(lr, nocc, 'RPA', None, False,
                                    {'GW': get_method_info('GW')}, ['GW'], False, True)
    omega, X, Y = spectrum['singlet']
    return eps, coeff, nocc, omega, X, Y


def test_the_blocked_static_self_energy_is_the_dense_one(mf):
    """Gate (a). The blocked builder, forced to one excitation per chunk,
    reproduces calculate_self_energy_matrix at round-off, and its diagonal is
    calculate_self_energy(p, eps_p) per state: the matrix routine's
    `tmp + tmp.T` with prefactor 1.0 is the per-state prefactor 2.0 on the
    diagonal, which is what makes mode A's diagonal the G0W0 self-energy."""
    eps, coeff, nocc, omega, X, Y = mean_field_spectrum(mf)
    se = SelfEnergySolver(eps, df_coeff=coeff, spin_mode='restricted')
    rho = se._rho_a_df(nocc, X, Y)
    chi = se.get_chi_a(nocc, X, Y)
    dense = se.calculate_self_energy_matrix(nocc, omega, chi)
    blocked = se.static_self_energy_matrix(nocc, omega, rho, block_elems=1)
    whole = se.static_self_energy_matrix(nocc, omega, rho)
    diag = np.array([se.calculate_self_energy(p, eps[p], nocc, omega, chi)
                     for p in range(len(eps))])
    d_dense = np.abs(blocked - dense).max()
    d_whole = np.abs(whole - dense).max()
    d_diag = np.abs(np.diag(whole) - diag).max()
    ok = check(d_dense < 1e-11, 'chunked builder equals the dense matrix routine',
               f'max |d Sigma| {d_dense:.1e} Ha, one excitation per chunk')
    ok &= check(d_whole < 1e-11, 'one-chunk builder equals the dense matrix routine',
                f'max |d Sigma| {d_whole:.1e} Ha')
    ok &= check(d_diag < 1e-11, 'its diagonal is Sigma_pp(eps_p) of every state',
                f'max |d Sigma_pp| {d_diag:.1e} Ha')
    ok &= check(np.abs(whole - whole.T).max() < 1e-14, 'it is symmetric')
    return ok


def test_the_srg_static_self_energy(mf):
    """Gate (a), the regularized form. Marie and Loos's SRG-qsGW self-energy
    (arXiv:2303.05984, eq. 44; JCTC 2023, doi 10.1021/acs.jctc.3c00281),

        Sigma_pq(s) = 2 sum_{S,r} chi_Srp chi_Srq K(a_Srp, a_Srq),
        K(a, b) = (a + b) / (a^2 + b^2) [1 - exp(-(a^2 + b^2) s)],

    the 2 from the restricted spin sum. The blocked builder, forced to one
    (S, r) pair per chunk, equals a dense einsum of the formula; s = 0 is the
    Hartree-Fock limit, zero. On the HOMO and LUMO rows every denominator
    exceeds 0.25 Ha, so there the diagonal equals mode A's at eta = 1 mHa to
    O(eta^2 / a^2) ~ 1e-5 relative, below EVGW_TOL: that pins the prefactor."""
    eps, coeff, nocc, omega, X, Y = mean_field_spectrum(mf)
    se = SelfEnergySolver(eps, df_coeff=coeff, spin_mode='restricted')
    rho = se._rho_a_df(nocc, X, Y)
    chi = se.get_chi_a(nocc, X, Y)
    flow = QSGW_SRG_FLOW
    sign = np.where(np.arange(len(eps)) < nocc, 1.0, -1.0)
    a = (eps[None, None, :] - eps[None, :, None]
         + (sign[None, :] * omega[:, None])[:, :, None])
    lam = a[..., :, None]**2 + a[..., None, :]**2
    kern = (a[..., :, None] + a[..., None, :]) / lam * -np.expm1(-flow * lam)
    dense = 2.0 * np.einsum('Srp, Srq, Srpq -> pq', chi, chi, kern)
    blocked = se.static_self_energy_matrix(nocc, omega, rho, flow=flow,
                                           block_elems=1)
    whole = se.static_self_energy_matrix(nocc, omega, rho, flow=flow)
    zero = se.static_self_energy_matrix(nocc, omega, rho, flow=0.0)
    mode_a = se.static_self_energy_matrix(nocc, omega, rho)
    front = [nocc - 1, nocc]
    d_blocked = np.abs(blocked - dense).max()
    d_whole = np.abs(whole - dense).max()
    d_front = np.abs(np.diag(whole)[front] - np.diag(mode_a)[front]).max()
    ok = check(d_blocked < 1e-11, 'chunked SRG builder equals eq. 44, dense',
               f'max |d Sigma| {d_blocked:.1e} Ha, one (S, r) pair per chunk')
    ok &= check(d_whole < 1e-11, 'one-chunk SRG builder equals eq. 44, dense',
                f'max |d Sigma| {d_whole:.1e} Ha')
    ok &= check(np.abs(zero).max() == 0.0, 's = 0 is the Hartree-Fock limit, zero')
    ok &= check(np.abs(whole - whole.T).max() == 0.0, 'it is symmetric')
    ok &= check(d_front < EVGW_TOL, 'HOMO and LUMO diagonals equal mode A\'s',
                f'max |d Sigma_pp| {d_front:.1e} Ha at s = {flow:g}')
    return ok


def test_the_rotated_view_carries_the_orbitals_into_the_df_factors(mf):
    """The view hands the rotated orbitals to get_density_fitting_coefficients,
    so B' = U^T B U with U the rotation, and leaves the original untouched."""
    eps, coeff, nocc, omega, X, Y = mean_field_spectrum(mf)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((len(eps), len(eps)))
    U, _ = np.linalg.qr(A)
    view = rotated_mean_field(mf, eps + 0.01, mf.mo_coeff @ U)
    coeff_view = get_density_fitting_coefficients(mf.mol, view,
                                                  representation='spatial')
    rotated = np.einsum('rq, Prs, sp -> Pqp', U, coeff, U)
    d = np.abs(coeff_view - rotated).max()
    ok = check(d < 1e-10, "the view's DF factors are U^T B U", f'max |d B| {d:.1e}')
    ok &= check(view.with_df is mf.with_df,
                'the view shares the DF object of the original')
    ok &= check(np.abs(np.asarray(mf.mo_energy) - eps).max() == 0.0
                and np.abs(mf.mo_coeff - (view.mo_coeff @ U.T)).max() < 1e-12,
                'the original keeps its eigenvalues and orbitals')
    ok &= check(np.array_equal(view.mo_occ, mf.mo_occ), 'the occupations are kept')
    return ok


def test_a_preset_transition_density_feeds_the_amplitudes(mf):
    """A solver built on rotated DF factors contracts an injected rho, the mean
    field's, instead of forming its own: the qsGW0 amplitudes chi' = rho^T B'."""
    eps, coeff, nocc, omega, X, Y = mean_field_spectrum(mf)
    se = SelfEnergySolver(eps, df_coeff=coeff, spin_mode='restricted')
    rho = se._rho_a_df(nocc, X, Y)
    rng = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.standard_normal((len(eps), len(eps))))
    coeff_rot = np.einsum('rq, Prs, sp -> Pqp', U, coeff, U)
    se_rot = SelfEnergySolver(eps, df_coeff=coeff_rot, spin_mode='restricted')
    se_rot.preset_transition_density(nocc, X, Y, rho)
    chi_rot = se_rot.get_chi_a(nocc, X, Y, p_state=3)
    expect = rho.T @ np.ascontiguousarray(coeff_rot[:, :, 3])
    d = np.abs(chi_rot - expect).max()
    ok = check(d < 1e-12, "get_chi_a uses the preset rho with the solver's factors",
               f'max |d chi| {d:.1e}')
    own = SelfEnergySolver(eps, df_coeff=coeff_rot, spin_mode='restricted')
    own_chi = own.get_chi_a(nocc, X, Y, p_state=3)
    ok &= check(np.abs(own_chi - chi_rot).max() > 1e-6,
                'without the preset the solver forms a different rho from its factors')
    return ok


def gap_ev(eps, nocc):
    return (eps[nocc] - eps[nocc - 1]) * HARTREE_TO_EV


def test_both_flavors_converge_and_open_the_gap(mf):
    """qsGW and qsGW0 converge on water within EVGW_TOL on HOMO and LUMO and
    QSGW_DM_TOL on the density; both gaps lie above the Casida G0W0 gap of the
    same mean field, as every self-consistent flavor's does on water (Kaplan
    2016, Table 1: qsGW IP 12.95 eV against G0W0@PBE 11.87 eV). qsGW0 solves
    the Casida problem once; qsGW once per cycle."""
    nocc = mf.mol.nelectron // 2
    builds = []
    original = qpe._casida_spectrum

    def counted(*args, **kw):
        builds.append(1)
        return original(*args, **kw)

    qpe._casida_spectrum = counted
    try:
        eps_qs, c_qs, qs = qsgw_eigenvalues(mf, screening='updated')
        n_qs = len(builds)
        builds.clear()
        eps_qs0, c_qs0, qs0 = qsgw_eigenvalues(mf, screening='fixed')
        n_qs0 = len(builds)
    finally:
        qpe._casida_spectrum = original
    states = list(range(len(eps_qs)))
    g0w0 = qpe.calc_qp_energy(mf, mode='casida', state=states)
    g0w0 = np.array([g0w0[p]['GW'] for p in states]) / HARTREE_TO_EV
    ok = check(qs['converged'] and qs0['converged'], 'qsGW and qsGW0 converge',
               f"{qs['cycles']} and {qs0['cycles']} cycles")
    ok &= check(qs['history'][-1] < EVGW_TOL and qs['dm_history'][-1] < 1e-6,
                'the last qsGW step meets both criteria',
                f"d eps {qs['history'][-1]:.1e} Ha, d D {qs['dm_history'][-1]:.1e}")
    ok &= check(n_qs == qs['cycles'] and n_qs0 == 1,
                'one Casida solve per qsGW cycle, one for the whole qsGW0 loop',
                f'{n_qs} in {qs["cycles"]} cycles vs {n_qs0}')
    ok &= check(gap_ev(g0w0, nocc) < gap_ev(eps_qs0, nocc)
                and gap_ev(g0w0, nocc) < gap_ev(eps_qs, nocc),
                'G0W0 gap below both self-consistent gaps',
                f'{gap_ev(g0w0, nocc):.3f} | qsGW0 {gap_ev(eps_qs0, nocc):.3f} '
                f'| qsGW {gap_ev(eps_qs, nocc):.3f} eV')
    ov = mf.mol.intor_symmetric('int1e_ovlp')
    ortho = np.abs(c_qs.T @ ov @ c_qs - np.eye(len(eps_qs))).max()
    ok &= check(ortho < 1e-10, 'the returned orbitals are S-orthonormal',
                f'max |C^T S C - 1| {ortho:.1e}')
    naux = qs['df_coeff'].shape[0]
    ok &= check(qs['w_aux'].shape == (naux, naux)
                and qs0['w_aux'].shape == (naux, naux)
                and qs['df_coeff'].shape == (naux, len(eps_qs), len(eps_qs)),
                'info carries the static W and the DF factors of the result')
    return ok


def test_the_two_mixings_land_on_one_fixed_point(mf):
    """CDIIS on the AO Hamiltonian and Kaplan's linear mixing are two paths to
    the same fixed point: HOMO and LUMO agree within 10 EVGW_TOL, each run
    being converged to EVGW_TOL on its own."""
    nocc = mf.mol.nelectron // 2
    ok = True
    for screening in ('updated', 'fixed'):
        e_d, _, d = qsgw_eigenvalues(mf, screening=screening, mixing='diis')
        e_l, _, l = qsgw_eigenvalues(mf, screening=screening, mixing='linear')
        delta = np.abs(e_d[[nocc - 1, nocc]] - e_l[[nocc - 1, nocc]]).max()
        ok &= check(d['converged'] and l['converged'] and delta < 10 * EVGW_TOL,
                    f"{screening}: DIIS and linear mixing agree on HOMO and LUMO",
                    f"{delta * HARTREE_TO_EV * 1e3:.3f} meV, "
                    f"{d['cycles']} vs {l['cycles']} cycles")
    return ok


def test_the_refusals(mf):
    """Bad input and unsupported branches are named, not silently served."""
    ok = True
    for kw, exc, text in (({'screening': 'never'}, ValueError, 'screening'),
                          ({'mixing': 'never'}, ValueError, 'mixing'),
                          ({'converge_on': [10**6]}, ValueError, 'converge_on')):
        try:
            qsgw_eigenvalues(mf, max_cycle=1, **kw)
            ok &= check(False, f'{kw} is refused')
        except exc as e:
            ok &= check(text in str(e), f'{kw} is refused with {exc.__name__}')
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=0)
    umf = dft.UKS(mol)
    umf.xc = 'pbe0'
    umf.kernel()
    try:
        qsgw_eigenvalues(umf, max_cycle=1)
        ok &= check(False, 'an unrestricted reference is refused')
    except NotImplementedError as e:
        ok &= check('restricted' in str(e), 'an unrestricted reference is refused')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, _, info = qsgw_eigenvalues(mf, max_cycle=1)
    ok &= check(not info['converged'] and any('did not converge' in str(w.message)
                                              for w in caught),
                'a capped loop warns and returns the last iterate')
    return ok


def test_calc_qp_energy_drives_the_loop(mf):
    """The front door: self_consistency='qsGW' returns the loop's HOMO, a list
    of states the dict with the loop's info and orbitals; qsGW0 likewise; any
    other route, a vertex and a density correction are refused."""
    nocc = mf.mol.nelectron // 2
    eps_qs, c_qs, _ = qsgw_eigenvalues(mf, screening='updated')
    homo = qpe.calc_qp_energy(mf, mode='casida', self_consistency='qsGW')
    ok = check(abs(homo - eps_qs[nocc - 1] * HARTREE_TO_EV) < 1e-8,
               "self_consistency='qsGW' returns the qsGW HOMO")
    both = qpe.calc_qp_energy(mf, mode='casida', self_consistency='qsGW0',
                              state=[nocc - 1, nocc])
    eps_qs0, c_qs0, _ = qsgw_eigenvalues(mf, screening='fixed')
    # an eigenvector is fixed up to its sign, and eigh's choice varies between
    # runs, so each column is aligned with the direct run's before comparing
    c_front = both['qsgw_info']['mo_coeff']
    c_front = c_front * np.sign(np.einsum('mp, mp -> p', c_front, c_qs0))
    ok &= check(abs(both[nocc]['GW'] - eps_qs0[nocc] * HARTREE_TO_EV) < 1e-8
                and np.abs(c_front - c_qs0).max() < 1e-8,
                "self_consistency='qsGW0' with a state list carries the orbitals")
    try:
        qpe.calc_qp_energy(mf, mode='space-time', self_consistency='qsGW')
        ok &= check(False, 'qsGW on the space-time route is refused')
    except NotImplementedError as e:
        ok &= check('casida' in str(e).lower(),
                    'qsGW on the space-time route is refused')
    try:
        qpe.calc_qp_energy(mf, mode='casida', self_consistency='qsGW',
                           selfenergy='GWGammaInf')
        ok &= check(False, 'a vertex under qsGW is refused')
    except NotImplementedError:
        ok &= check(True, 'a vertex under qsGW is refused')
    try:
        qpe.calc_qp_energy(mf, mode='casida', self_consistency='qsGW',
                           dm_correction=mf.make_rdm1())
        ok &= check(False, 'a density correction under qsGW is refused')
    except NotImplementedError as e:
        ok &= check('dm_correction' in str(e),
                    'a density correction under qsGW is refused')
    return ok


def kohn_sham_fock_diagonal(mf, mo_coeff):
    """<h + v_Hxc[D]>_pp in the basis `mo_coeff`, D its closed-shell density."""
    dm = mf.make_rdm1(mo_coeff, mf.mo_occ)
    fock = mf.get_hcore() + mf.get_veff(mf.mol, dm)
    return np.einsum('mp, mn, np -> p', mo_coeff, fock, mo_coeff)


def test_the_converged_point_is_a_fixed_point_of_the_evgw0_map(mf):
    """Gate (e). One application of the Casida eigenvalue map at the converged
    (eps', C'), anchored on the Kohn-Sham Fock diagonal of the rotated density,
    returns eps' on HOMO and LUMO within EVGW_TOL, and the evGW0 loop started
    there stops at cycle 1. For qsGW the step builds W from (eps', C') itself;
    for qsGW0 the mean field's spectrum and transition density are injected.
    The per-state root scan and the static part are independent code from the
    blocked builder, so this is a check of the loop and not of itself. The map
    broadens with eta, the loop regularizes with the SRG flow; on the frontier
    rows the two diagonals differ by about 5e-8 Ha (gate (a)), far inside the
    tolerance."""
    nocc = mf.mol.nelectron // 2
    ok = True
    for screening in ('updated', 'fixed'):
        # converged two orders tighter than the gate, so the gate reads the
        # fixed point and not the last step of the loop
        eps_qs, c_qs, info = qsgw_eigenvalues(mf, screening=screening,
                                              keep_spectrum=True, tol=1e-7,
                                              dm_tol=1e-8)
        view = rotated_mean_field(mf, eps_qs, c_qs)
        anchor = kohn_sham_fock_diagonal(mf, c_qs)
        step_kw = {'eps_anchor': anchor}
        if screening == 'fixed':
            step_kw.update(fixed_spectrum=info['spectrum'], fixed_rho=info['rho'])
        step = qpe.casida_evgw_step(view, mf.mol, screening, **step_kw)
        back = step(eps_qs)
        d_front = np.abs((back - eps_qs)[[nocc - 1, nocc]]).max()
        d_all = np.abs(back - eps_qs).max()
        label = 'qsGW' if screening == 'updated' else 'qsGW0'
        ok &= check(info['converged'] and d_front < EVGW_TOL,
                    f'{label}: one step of the evGW0 map returns the fixed point',
                    f'HOMO/LUMO {d_front * HARTREE_TO_EV * 1e3:.3f} meV, '
                    f'all states {d_all * HARTREE_TO_EV * 1e3:.3f} meV')
        _, loop = evgw_eigenvalues(view, mf.mol, mode='casida', screening=screening,
                                   eps_init=eps_qs, **step_kw)
        ok &= check(loop['converged'] and loop['cycles'] == 1,
                    f'{label}: the evGW0 loop started there stops at cycle 1',
                    f"{loop['cycles']} cycles, residual "
                    f"{loop['history'][-1] * HARTREE_TO_EV * 1e3:.3f} meV")
    return ok


def test_the_step_keywords_leave_evgw_unchanged(mf):
    """Without the three keywords the step is the one evGW0 has always used,
    and a spectrum without its transition density is refused."""
    eps_fixed, fixed = evgw_eigenvalues(mf, mf.mol, mode='casida', screening='fixed')
    step = qpe.casida_evgw_step(mf, mf.mol, 'fixed')
    eps0 = np.asarray(mf.mo_energy, float)
    first = step(eps0)
    states = list(range(len(eps0)))
    g0w0 = qpe.calc_qp_energy(mf, mode='casida', state=states)
    g0w0 = np.array([g0w0[p]['GW'] for p in states]) / HARTREE_TO_EV
    d = np.abs(first - g0w0).max()
    ok = check(d < 1e-10 and fixed['converged'],
               'the default step is still the Casida G0W0 and evGW0 still converges',
               f'max |d eps| {d:.1e} Ha')
    for kw, text in (({'fixed_spectrum': {}}, 'fixed_rho'),
                     ({'fixed_rho': np.zeros((1, 1))}, "screening='fixed'")):
        screening = 'updated' if 'fixed_rho' in kw else 'fixed'
        try:
            qpe.casida_evgw_step(mf, mf.mol, screening, **kw)
            ok &= check(False, f'{sorted(kw)} under {screening!r} is refused')
        except ValueError as e:
            ok &= check(text in str(e),
                        f'{sorted(kw)} under {screening!r} is refused')
    return ok


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    mf = build_reference()
    all_ok = True
    print('\n-- 1. the static self-energy, gate (a)')
    all_ok &= test_the_blocked_static_self_energy_is_the_dense_one(mf)
    all_ok &= test_the_srg_static_self_energy(mf)
    print('\n-- 2. the rotated view and the injected transition density')
    all_ok &= test_the_rotated_view_carries_the_orbitals_into_the_df_factors(mf)
    all_ok &= test_a_preset_transition_density_feeds_the_amplitudes(mf)
    print('\n-- 3. the loop')
    all_ok &= test_both_flavors_converge_and_open_the_gap(mf)
    all_ok &= test_the_two_mixings_land_on_one_fixed_point(mf)
    all_ok &= test_the_refusals(mf)
    print('\n-- 4. the front door')
    all_ok &= test_calc_qp_energy_drives_the_loop(mf)
    print('\n-- 5. the refeed, gate (e)')
    all_ok &= test_the_converged_point_is_a_fixed_point_of_the_evgw0_map(mf)
    all_ok &= test_the_step_keywords_leave_evgw_unchanged(mf)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
