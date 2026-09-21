"""qsGW and qsGW0: the orbitals and the eigenvalues reinjected until the static
Hermitian self-energy stops moving.

The blocked static self-energy equals the dense matrix routine and, on its
diagonal, the per-state self-energy at eps_p; its SRG-regularized form, the one
the loop runs, equals a dense evaluation of its formula. PBE and PBE0 starts
land on one fixed point. The converged point is a fixed point of the evGW0
eigenvalue map, in one step. The two mixings land on one fixed point.

Run: python tests/test_qsgw.py
"""
import os
import sys
import types
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
from pyscf import df, dft, gto, scf

from src.Base.constants import (EVGW_TOL, HARTREE_TO_EV, QSGW_DM_TOL,
                                QSGW_SRG_FLOW, QSGW_SRG_QUAD_TOL, get_method_info)
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.SingleReference.GW.evGW import evgw_eigenvalues, rotated_mean_field
from src.SingleReference.GW.qsGW import qsgw_eigenvalues
from src.SingleReference.GW.self_energy import (SelfEnergySolver,
                                                _srg_laplace_quadrature)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
import src.SingleReference.GW.qp_energy as qpe

GEOMETRY = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def check(ok, label, detail=''):
    """Print an [ok]/[FAIL] verdict line for `label` and return `ok`."""
    print(f"  [{'ok' if ok else 'FAIL'}] {label}"
          + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build_reference(xc='pbe0', atom=GEOMETRY, basis='cc-pvdz', symmetry=False):
    """Water, cc-pVDZ, exact-JK SCF, the RI set attached for W and Sigma."""
    mol = gto.M(atom=atom, basis=basis, symmetry=symmetry, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.kernel()
    mf.with_df = df.DF(mol, auxbasis='cc-pvdz-ri')
    return mf


def mean_field_casida(mf):
    """(eps, coeff, nocc, spectrum, w_aux): the RPA Casida solution of `mf` as
    _casida_spectrum returns it, a dict with 'singlet' and 'rpa', and the
    static RPA W in the auxiliary basis of its DF factors `coeff`."""
    mol = mf.mol
    nocc = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted')
    spectrum = qpe._casida_spectrum(lr, nocc, 'RPA', None, False,
                                    {'GW': get_method_info('GW')}, ['GW'], False, True)
    return eps, coeff, nocc, spectrum, lr.static_screening_aux(nocc)


def mean_field_spectrum(mf):
    """(eps, coeff, nocc, omega, X, Y): the RPA Casida solution of `mf`."""
    eps, coeff, nocc, spectrum, _ = mean_field_casida(mf)
    omega, X, Y = spectrum['singlet']
    return eps, coeff, nocc, omega, X, Y


def test_the_blocked_static_self_energy_is_the_dense_one(mf):
    """The blocked builder, forced to one excitation per chunk,
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


def test_the_laplace_quadrature():
    """The rule behind the SRG kernel: sum_n w_n exp(-mu x_n) against
    int_0^1 exp(-mu x) dx = (1 - exp(-mu)) / mu, on a grid five times denser
    than the rule's own check. Relative error below QSGW_SRG_QUAD_TOL for every
    mu in [0, mu_max], from 10 to 1e9, the 2 s a_max^2 of an all-electron
    basis on a heavy atom; a tolerance below machine precision is refused."""
    ok = True
    for mu_max in (10.0, 1e4, 1e6, 1e9):
        x, w = _srg_laplace_quadrature(mu_max, QSGW_SRG_QUAD_TOL)
        mu = np.concatenate([[0.0], np.logspace(-6, np.log10(mu_max), 20000)])
        exact = np.ones_like(mu)
        exact[1:] = -np.expm1(-mu[1:]) / mu[1:]
        err = np.abs(np.exp(-np.outer(mu, x)) @ w / exact - 1).max()
        ok &= check(err < QSGW_SRG_QUAD_TOL and 0 < x.min() and x.max() <= 1,
                    f'mu_max = {mu_max:g}: the rule meets its tolerance',
                    f'{len(x)} nodes, max rel err {err:.1e}')
    try:
        _srg_laplace_quadrature(1e6, 1e-17)
        ok &= check(False, 'a tolerance below machine precision is refused')
    except ValueError as e:
        ok &= check('tol' in str(e), 'a tolerance below machine precision is refused')
    return ok


def test_the_srg_static_self_energy(mf):
    """The regularized form. Marie and Loos's SRG-qsGW self-energy
    (arXiv:2303.05984, eq. 44; JCTC 2023, doi 10.1021/acs.jctc.3c00281),

        Sigma_pq(s) = 2 sum_{S,r} chi_Srp chi_Srq K(a_Srp, a_Srq),
        K(a, b) = (a + b) / (a^2 + b^2) [1 - exp(-(a^2 + b^2) s)],

    the 2 from the restricted spin sum. The builder evaluates K through a
    quadrature of relative error below QSGW_SRG_QUAD_TOL, so against a dense
    einsum of the formula each element may miss by that tolerance times
    2 sum |chi chi K| over its terms, plus round-off; blocked, one-chunk and
    threaded builds all meet it, and threads change only the summation order.
    s = 0 is the Hartree-Fock limit, zero. On the HOMO and LUMO rows every
    denominator exceeds 0.25 Ha, so there the diagonal equals mode A's at
    eta = 1 mHa to O(eta^2 / a^2) ~ 1e-5 relative, below EVGW_TOL: that pins
    the prefactor."""
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
    bound = (QSGW_SRG_QUAD_TOL * 2.0
             * np.einsum('Srp, Srq, Srpq -> pq', np.abs(chi), np.abs(chi),
                         np.abs(kern)) + 1e-14)
    nmo = len(eps)
    blocked = se.static_self_energy_matrix(nocc, omega, rho, flow=flow,
                                           block_elems=1)
    whole = se.static_self_energy_matrix(nocc, omega, rho, flow=flow)
    threaded = se.static_self_energy_matrix(nocc, omega, rho, flow=flow,
                                            block_elems=10 * nmo * nmo,
                                            n_workers=4)
    zero = se.static_self_energy_matrix(nocc, omega, rho, flow=0.0)
    mode_a = se.static_self_energy_matrix(nocc, omega, rho)
    front = [nocc - 1, nocc]
    ok = True
    for label, sigma in (('one excitation per chunk', blocked),
                         ('one chunk', whole),
                         ('four threads, twelve chunks each', threaded)):
        ratio = (np.abs(sigma - dense) / bound).max()
        ok &= check(ratio <= 1.0,
                    f'SRG builder equals eq. 44 to its tolerance, {label}',
                    f'max |d Sigma| {np.abs(sigma - dense).max():.1e} Ha, '
                    f'{ratio:.2f} of the bound')
    d_thr = np.abs(threaded - whole).max() / np.abs(whole).max()
    ok &= check(d_thr < 1e-12, 'threads change only the summation order',
                f'max relative {d_thr:.1e}')
    ok &= check(np.abs(zero).max() == 0.0, 's = 0 is the Hartree-Fock limit, zero')
    ok &= check(np.abs(whole - whole.T).max() == 0.0, 'it is symmetric')
    d_front = np.abs(np.diag(whole)[front] - np.diag(mode_a)[front]).max()
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
    """HOMO-LUMO gap in eV of the spectrum `eps`, shape (nmo,), in Hartree."""
    return (eps[nocc] - eps[nocc - 1]) * HARTREE_TO_EV


def test_both_flavors_converge_and_open_the_gap(mf):
    """qsGW and qsGW0 converge on water within EVGW_TOL on HOMO and LUMO and
    QSGW_DM_TOL on the density; both gaps lie above the Casida G0W0 gap of the
    same mean field, as every self-consistent flavor's does on water (Kaplan
    2016, Table 1: qsGW IP 12.95 eV against G0W0@PBE 11.87 eV). qsGW0 solves
    the Casida problem once; qsGW once per cycle. `info` carries the DF
    factors with_df builds for the returned orbitals and the static W each
    flavor screens with, the result's for qsGW and the start's for qsGW0, to
    round-off."""
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
    ok &= check(qs['history'][-1] < EVGW_TOL and qs['dm_history'][-1] < QSGW_DM_TOL,
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
    # the factors rebuilt from the returned orbitals through with_df, and the W
    # each flavor screens with: qsGW's from those factors, qsGW0's the start's
    eps0, coeff0, _, _, w0 = mean_field_casida(mf)
    d_b, d_w = [], []
    for eps_x, c_x, info, screening in ((eps_qs, c_qs, qs, 'updated'),
                                        (eps_qs0, c_qs0, qs0, 'fixed')):
        rebuilt = get_density_fitting_coefficients(
            mf.mol, rotated_mean_field(mf, eps_x, c_x), representation='spatial')
        w_ref = w0 if screening == 'fixed' else LinearResponseSolver(
            eps_x, coeff_df=rebuilt, spin_mode='restricted').static_screening_aux(nocc)
        d_b.append(np.abs(info['df_coeff'] - rebuilt).max())
        d_w.append(np.abs(info['w_aux'] - w_ref).max() / np.abs(w_ref).max())
    ok &= check(max(d_b) < 1e-10 and max(d_w) < 1e-10,
                'info carries the DF factors of the result and the W it screens with',
                f'max |d B| {max(d_b):.1e}, max relative |d W| {max(d_w):.1e}')
    return ok


def test_the_two_mixings_land_on_one_fixed_point(mf):
    """CDIIS on the AO Hamiltonian and Kaplan's linear mixing are two paths to
    the same fixed point: HOMO and LUMO agree within 10 EVGW_TOL, each run
    being converged to EVGW_TOL on its own. The linear step is Kaplan et al.'s
    eq. 21 (J. Chem. Theory Comput. 12, 2528 (2016)), lambda of the new
    Hamiltonian and 1 - lambda of the last iterate's: at the first cycle the
    mean field's Fock matrix, here PySCF's own get_fock, so one cycle at
    lambda = 0.3 gives the spectrum of 0.3 H_new + 0.7 F_KS, H_new rebuilt from
    the cycle's Sigma~, within 1e-6 Ha: get_fock rebuilds F from the converged
    density, which differs from the SCF's last eigenpairs at its gradient
    tolerance, sqrt(conv_tol). A first cycle left unmixed misses by 0.5 Ha. At
    Kaplan's lambda the linear loop needs 25 to 30 cycles on water, so its runs
    here get 60."""
    nocc = mf.mol.nelectron // 2
    ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e_one, _, one = qsgw_eigenvalues(mf, mixing='linear', mixing_lambda=0.3,
                                         max_cycle=1)
    ovlp = mf.get_ovlp()
    dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    cs0 = ovlp @ mf.mo_coeff
    h_new = (mf.get_hcore() + scf.RHF(mf.mol).get_veff(mf.mol, dm0)
             + cs0 @ one['sigma_static'] @ cs0.T)
    e_ref = scipy.linalg.eigh(0.3 * h_new + 0.7 * mf.get_fock(), ovlp)[0]
    d_one = np.abs(e_one - e_ref).max()
    ok &= check(d_one < 1e-6, "one linear cycle is Kaplan's eq. 21 from the mean field",
                f'max |d eps| {d_one:.1e} Ha')
    for screening in ('updated', 'fixed'):
        e_d, _, d = qsgw_eigenvalues(mf, screening=screening, mixing='diis')
        e_l, _, l = qsgw_eigenvalues(mf, screening=screening, mixing='linear',
                                     max_cycle=60)
        delta = np.abs(e_d[[nocc - 1, nocc]] - e_l[[nocc - 1, nocc]]).max()
        ok &= check(d['converged'] and l['converged'] and delta < 10 * EVGW_TOL,
                    f"{screening}: DIIS and linear mixing agree on HOMO and LUMO",
                    f"{delta * HARTREE_TO_EV * 1e3:.3f} meV, "
                    f"{d['cycles']} vs {l['cycles']} cycles")
    return ok


def test_a_mean_field_without_density_fitting(mf):
    """Without with_df the DF factors are an eigendecomposition of the MO-basis
    ERI, whose auxiliary index follows the orbitals. The loop rotates the mean
    field's factors, B'_P,pq = sum_mn U_mp B_P,mn U_nq with U = C0^T S C, so the
    transition density qsGW0 builds once stays in their auxiliary basis. On the
    same SCF with with_df removed both flavors converge and land on the RI runs'
    HOMO and LUMO within 5 meV, the cc-pvdz-ri fitting error being about 1 meV
    (the qsGW pair, reported). The evGW0 step refuses a transition density on
    such a mean field, which has no auxiliary basis to keep it in."""
    nocc = mf.mol.nelectron // 2
    front = [nocc - 1, nocc]
    plain = mf.copy()
    plain.with_df = None
    ok = True
    d_front = {}
    for screening in ('updated', 'fixed'):
        e_ri, _, _ = qsgw_eigenvalues(mf, screening=screening)
        e_ex, c_ex, info = qsgw_eigenvalues(plain, screening=screening)
        d_front[screening] = np.abs(e_ex[front] - e_ri[front]).max() * HARTREE_TO_EV
        rot = mf.mo_coeff.T @ mf.get_ovlp() @ c_ex
        coeff0 = get_density_fitting_coefficients(mf.mol, plain,
                                                  representation='spatial')
        d_b = np.abs(info['df_coeff']
                     - np.einsum('Pmn, mp, nq -> Ppq', coeff0, rot, rot)).max()
        ok &= check(info['converged'] and d_b < 1e-10,
                    f'{screening}: converges, df_coeff in the mean field\'s '
                    f'auxiliary basis', f"{info['cycles']} cycles, "
                    f'max |df_coeff - U^T B0 U| {d_b:.1e}')
    ok &= check(d_front['fixed'] < 5e-3,
                'qsGW0 without with_df lands on the RI HOMO and LUMO',
                f"{d_front['fixed'] * 1e3:.3f} meV, qsGW {d_front['updated'] * 1e3:.3f}"
                ' meV')
    eps, coeff, nocc, omega, X, Y = mean_field_spectrum(mf)
    rho = SelfEnergySolver(eps, df_coeff=coeff, spin_mode='restricted')._rho_a_df(
        nocc, X, Y)
    spectrum = {'singlet': (omega, X, Y)}
    try:
        qpe.casida_evgw_step(plain, plain.mol, 'fixed', fixed_spectrum=spectrum,
                             fixed_rho=rho)
        ok &= check(False, 'fixed_rho on a mean field without with_df is refused')
    except NotImplementedError as e:
        ok &= check('density_fit' in str(e),
                    'fixed_rho on a mean field without with_df is refused')
    return ok


def test_the_refusals(mf):
    """Bad input and unsupported branches are named, not silently served."""
    ok = True
    for kw, text in (({'screening': 'never'}, 'screening'),
                     ({'mixing': 'never'}, 'mixing'),
                     ({'converge_on': [10**6]}, 'converge_on'),
                     ({'converge_on': [4, -1]}, 'converge_on'),
                     ({'mixing': 'linear', 'mixing_lambda': 0.0}, 'mixing_lambda'),
                     ({'max_cycle': 0}, 'max_cycle'),
                     ({'diis_size': 0}, 'diis_size'),
                     ({'flow': float('nan')}, 'flow')):
        try:
            qsgw_eigenvalues(mf, **{'max_cycle': 1, **kw})
            ok &= check(False, f'{kw} is refused')
        except ValueError as e:
            ok &= check(text in str(e), f'{kw} is refused with ValueError')
    nocc = mf.mol.nelectron // 2
    swapped = mf.copy()
    swapped.mo_occ = mf.mo_occ.copy()
    swapped.mo_occ[[nocc - 1, nocc]] = swapped.mo_occ[[nocc, nocc - 1]]
    try:
        qsgw_eigenvalues(swapped, max_cycle=1)
        ok &= check(False, 'a closed-shell but non-aufbau occupation is refused')
    except NotImplementedError as e:
        ok &= check('aufbau' in str(e),
                    'a closed-shell but non-aufbau occupation is refused')
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=0)
    umf = dft.UKS(mol)
    umf.xc = 'pbe0'
    umf.kernel()
    try:
        qsgw_eigenvalues(umf, max_cycle=1)
        ok &= check(False, 'an unrestricted reference is refused')
    except NotImplementedError as e:
        ok &= check('restricted' in str(e), 'an unrestricted reference is refused')
    romf = dft.ROKS(mol)
    romf.xc = 'pbe0'
    romf.kernel()
    try:
        qsgw_eigenvalues(romf, max_cycle=1)
        ok &= check(False, 'an open-shell restricted (ROKS) reference is refused')
    except NotImplementedError as e:
        ok &= check('closed-shell' in str(e),
                    'an open-shell restricted (ROKS) reference is refused')
    # canonical orthogonalization forced on water: S eigenvalues below 0.05 dropped
    lmf = scf.addons.remove_linear_dep_(dft.RKS(mf.mol), threshold=0.05, lindep=1.0)
    lmf.xc = 'pbe0'
    lmf.kernel()
    try:
        qsgw_eigenvalues(lmf, max_cycle=1)
        ok &= check(False, 'a mean field with fewer MOs than AOs is refused')
    except NotImplementedError as e:
        ok &= check('AO space' in str(e) and lmf.mo_coeff.shape[1] < mf.mol.nao,
                    'a mean field with fewer MOs than AOs is refused',
                    f'{lmf.mo_coeff.shape[1]} of {mf.mol.nao} orbitals kept')
    solvated = mf.copy()
    solvated.with_screening = types.SimpleNamespace(screens=True)
    for label, call, exc, text in (
            ('no density fitting', lambda: qsgw_eigenvalues(mf, df=False),
             NotImplementedError, 'DF'),
            ('an attached environment', lambda: qsgw_eigenvalues(solvated),
             NotImplementedError, 'gas phase'),
            ('a negative flow', lambda: qsgw_eigenvalues(mf, flow=-1.0, max_cycle=1),
             ValueError, 'flow')):
        try:
            call()
            ok &= check(False, f'{label} is refused')
        except exc as e:
            ok &= check(text in str(e), f'{label} is refused with {exc.__name__}')
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
    other route, a vertex, a density correction and an anchor are refused."""
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
    for label, kw, text in (
            ('a vertex', {'selfenergy': 'GWGammaInf'}, 'GW@RPA only'),
            ('a BSE polarizability', {'polarizability': 'BSE'}, 'GW@RPA only'),
            ('a density correction', {'dm_correction': mf.make_rdm1()},
             'dm_correction'),
            ('an anchor', {'eps_anchor': np.asarray(mf.mo_energy, float)},
             'eps_anchor')):
        try:
            qpe.calc_qp_energy(mf, mode='casida', self_consistency='qsGW', **kw)
            ok &= check(False, f'{label} under qsGW is refused')
        except NotImplementedError as e:
            ok &= check(text in str(e), f'{label} under qsGW is refused')
    return ok


def kohn_sham_fock_diagonal(mf, mo_coeff):
    """<h + v_Hxc[D]>_pp in the basis `mo_coeff`, D its closed-shell density."""
    dm = mf.make_rdm1(mo_coeff, mf.mo_occ)
    fock = mf.get_hcore() + mf.get_veff(mf.mol, dm)
    return np.einsum('mp, mn, np -> p', mo_coeff, fock, mo_coeff)


def test_the_converged_point_is_a_fixed_point_of_the_evgw0_map(mf):
    """One application of the Casida eigenvalue map at the converged
    (eps', C'), anchored on the Kohn-Sham Fock diagonal of the rotated density,
    returns eps' on HOMO and LUMO within EVGW_TOL, and the evGW0 loop started
    there stops at cycle 1. For qsGW the step builds W from (eps', C') itself;
    for qsGW0 the mean field's spectrum and transition density, built here and
    not handed back by the loop, are injected, and the density the loop kept
    must equal them.
    The per-state root scan and the static part are independent code from the
    blocked builder, so this is a check of the loop and not of itself. The map
    broadens with eta, the loop regularizes with the SRG flow; on the frontier
    rows the two diagonals differ by about 5e-8 Ha
    (test_the_srg_static_self_energy), far inside the tolerance."""
    nocc = mf.mol.nelectron // 2
    ok = True
    for screening in ('updated', 'fixed'):
        # converged two orders tighter than the check, so the check reads the
        # fixed point and not the last step of the loop
        eps_qs, c_qs, info = qsgw_eigenvalues(mf, screening=screening,
                                              keep_spectrum=True, tol=1e-7,
                                              dm_tol=1e-8)
        view = rotated_mean_field(mf, eps_qs, c_qs)
        anchor = kohn_sham_fock_diagonal(mf, c_qs)
        step_kw = {'eps_anchor': anchor}
        if screening == 'fixed':
            eps0, coeff0, _, spectrum0, _ = mean_field_casida(mf)
            _, X0, Y0 = spectrum0['singlet']
            rho0 = SelfEnergySolver(eps0, df_coeff=coeff0,
                                    spin_mode='restricted')._rho_a_df(nocc, X0, Y0)
            # an eigenvector is fixed up to its sign, so compare
            # sum_S rho_PS rho_QS / Omega_S, the combination Sigma reads
            omega0 = spectrum0['singlet'][0]
            ref = (rho0 / omega0) @ rho0.T
            d_rho = (np.abs((info['rho'] / omega0) @ info['rho'].T - ref).max()
                     / np.abs(ref).max())
            ok &= check(d_rho < 1e-10,
                        "qsGW0: the loop's transition density is the mean field's",
                        f'max relative |d sum_S rho rho / Omega| {d_rho:.1e}')
            step_kw.update(fixed_spectrum=spectrum0, fixed_rho=rho0)
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
    """Without the three keywords the step is the one evGW0 has always used;
    a spectrum without its transition density, or the reverse, is refused."""
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
    for screening, kw, text in (
            ('fixed', {'fixed_spectrum': {}}, 'fixed_rho'),
            ('fixed', {'fixed_rho': np.zeros((1, 1))}, 'fixed_spectrum'),
            ('updated', {'fixed_rho': np.zeros((1, 1))}, "screening='fixed'")):
        try:
            qpe.casida_evgw_step(mf, mf.mol, screening, **kw)
            ok &= check(False, f'{sorted(kw)} under {screening!r} is refused')
        except ValueError as e:
            ok &= check(text in str(e),
                        f'{sorted(kw)} under {screening!r} is refused')
    return ok


def test_qsgw_forgets_its_starting_point():
    """qsGW from PBE and from PBE0 lands on one fixed point. Both runs converge
    two orders tighter than the default, tol = 1e-7 Ha and dm_tol = 1e-8, so
    the comparison reads the fixed points and not the loops' last steps: HOMO
    and LUMO within EVGW_TOL, every state within 1 meV. The all-state bound is
    the one that tells two self-consistent branches apart: where the high
    virtuals of the two starts settle on different ones, the frontier can
    still agree to a meV while they sit eV apart. qsGW0 keeps the start's W
    and is start-dependent by construction; its two end points are reported."""
    nocc = None
    ends = {}
    for xc in ('pbe', 'pbe0'):
        mf = build_reference(xc)
        nocc = mf.mol.nelectron // 2
        ends[xc] = {'updated': qsgw_eigenvalues(mf, tol=1e-7, dm_tol=1e-8),
                    'fixed': qsgw_eigenvalues(mf, screening='fixed')}
    e_pbe, c_pbe, i_pbe = ends['pbe']['updated']
    e_pbe0, c_pbe0, i_pbe0 = ends['pbe0']['updated']
    front = [nocc - 1, nocc]
    d_qs = np.abs(e_pbe[front] - e_pbe0[front]).max()
    d_all = np.abs(e_pbe - e_pbe0).max()
    d_dm = np.linalg.norm(2 * c_pbe[:, :nocc] @ c_pbe[:, :nocc].T
                          - 2 * c_pbe0[:, :nocc] @ c_pbe0[:, :nocc].T) / len(e_pbe)
    ok = check(i_pbe['converged'] and i_pbe0['converged'],
               'qsGW from PBE and from PBE0 converges at tol 1e-7, dm_tol 1e-8',
               f"{i_pbe['cycles']} and {i_pbe0['cycles']} cycles")
    ok &= check(d_qs < EVGW_TOL, 'the two agree on HOMO and LUMO',
                f'{d_qs * HARTREE_TO_EV * 1e3:.4f} meV, |dD|/nmo {d_dm:.1e}')
    ok &= check(d_all * HARTREE_TO_EV < 1e-3, 'and on every state, one branch',
                f'{d_all * HARTREE_TO_EV * 1e3:.3f} meV')
    e0_pbe = ends['pbe']['fixed'][0]
    e0_pbe0 = ends['pbe0']['fixed'][0]
    d_qs0 = np.abs(e0_pbe[front] - e0_pbe0[front]).max()
    print(f'  qsGW0 from PBE vs PBE0 on HOMO/LUMO: {d_qs0 * HARTREE_TO_EV:.4f} eV '
          f'(reported, start-dependent by construction)')
    return ok


def test_the_boundaries(mf):
    """Shapes and switches the other checks never visit. TDA screening
    converges to its own fixed point, its gap more than 0.1 eV from full RPA's,
    so the switch reaches the Casida solver. A Mole with point-group symmetry
    on reaches the same fixed point as without, within 10 EVGW_TOL on HOMO and
    LUMO, each run converged to EVGW_TOL. One occupied orbital (H2) and one
    virtual (HF in STO-3G), where a reshape of an (nocc, nvirt) block can
    return a view instead of a copy: both flavors converge with a positive
    gap, leave the mean field untouched, and one excitation per chunk on one
    thread gives the default build's spectrum within EVGW_TOL, each run
    converged to it."""
    nocc = mf.mol.nelectron // 2
    ok = True
    e_tda, _, i_tda = qsgw_eigenvalues(mf, tda=True)
    e_full, _, _ = qsgw_eigenvalues(mf)
    ok &= check(i_tda['converged']
                and abs(gap_ev(e_tda, nocc) - gap_ev(e_full, nocc)) > 0.1,
                'TDA screening converges to its own fixed point',
                f"{i_tda['cycles']} cycles, gap {gap_ev(e_tda, nocc):.3f} eV "
                f"against {gap_ev(e_full, nocc):.3f} eV with full RPA")
    e_sym, _, i_sym = qsgw_eigenvalues(build_reference(symmetry=True))
    d_sym = np.abs(e_sym[[nocc - 1, nocc]] - e_full[[nocc - 1, nocc]]).max()
    ok &= check(i_sym['converged'] and d_sym < 10 * EVGW_TOL,
                'point-group symmetry on reaches the same fixed point',
                f'{d_sym * HARTREE_TO_EV * 1e3:.4f} meV')
    for label, atom, basis in (('H2, nocc = 1', 'H 0 0 0; H 0 0 0.74', 'cc-pvdz'),
                               ('HF in STO-3G, nvirt = 1', 'F 0 0 0; H 0 0 0.917',
                                'sto-3g')):
        small = build_reference(atom=atom, basis=basis)
        n = small.mol.nelectron // 2
        eps_mf = np.asarray(small.mo_energy, float).copy()
        c_mf = np.asarray(small.mo_coeff, float).copy()
        for screening in ('updated', 'fixed'):
            e, _, info = qsgw_eigenvalues(small, screening=screening)
            e_one, _, i_one = qsgw_eigenvalues(small, screening=screening,
                                               block_elems=1, n_workers=1)
            d_one = np.abs(e_one - e).max()
            kept = (np.array_equal(small.mo_energy, eps_mf)
                    and np.array_equal(small.mo_coeff, c_mf))
            ok &= check(info['converged'] and i_one['converged'] and e[n] > e[n - 1]
                        and kept and d_one < EVGW_TOL,
                        f'{label}, {screening}: converges with a positive gap, '
                        f'blocking-independent, mean field kept',
                        f"{info['cycles']} cycles, gap {gap_ev(e, n):.3f} eV, "
                        f'one-element blocks {d_one * HARTREE_TO_EV * 1e3:.4f} meV')
    return ok


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    mf = build_reference()
    all_ok = True
    print('\n-- 1. the static self-energy')
    all_ok &= test_the_blocked_static_self_energy_is_the_dense_one(mf)
    all_ok &= test_the_laplace_quadrature()
    all_ok &= test_the_srg_static_self_energy(mf)
    print('\n-- 2. the rotated view and the injected transition density')
    all_ok &= test_the_rotated_view_carries_the_orbitals_into_the_df_factors(mf)
    all_ok &= test_a_preset_transition_density_feeds_the_amplitudes(mf)
    print('\n-- 3. the loop')
    all_ok &= test_both_flavors_converge_and_open_the_gap(mf)
    all_ok &= test_the_two_mixings_land_on_one_fixed_point(mf)
    all_ok &= test_a_mean_field_without_density_fitting(mf)
    all_ok &= test_the_refusals(mf)
    print('\n-- 4. the front door')
    all_ok &= test_calc_qp_energy_drives_the_loop(mf)
    print('\n-- 5. the converged point is a fixed point of the evGW0 map')
    all_ok &= test_the_converged_point_is_a_fixed_point_of_the_evgw0_map(mf)
    all_ok &= test_the_step_keywords_leave_evgw_unchanged(mf)
    print('\n-- 6. starting-point independence')
    all_ok &= test_qsgw_forgets_its_starting_point()
    print('\n-- 7. the boundaries')
    all_ok &= test_the_boundaries(mf)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
