"""The GW density on the imaginary-frequency axis (mode='imagfrequency', N^4).

- The full-matrix correlation self-energy Sigma_c,pq(mu + i w): its pair
  weights w^m_pq are the QP route's (their diagonal under the plain W-grid
  quadrature is `self_energy_imaginary_axis`'s Sigma_pp); its inner integral
  is a Tikhonov-regularized Lorentzian fit of W_c on the W grid convolved
  exactly with G (the one-pole kernel against scipy quad), and it matches the
  Sigma summed over the Casida RPA poles.
- 'imagfrequency' reproduces 'casida' to 1e-5 per element in every block
  (oo, ov, vv) of Delta gamma_MO, on HF and PBE0, unrelaxed and 'dyson'
  relaxed, at the default grids; both routes screen with the same full RPA on
  the same three-index object (the mean field's density fit, or the exact
  integrals' decomposition on a mean field without one).
- The fit's regularization: lambda 1e-10 -> 1e-8 moves Delta gamma < 1e-6.
- Refusals: a polarizability other than RPA, df=False, an environment, no
  numerics, a thermally smeared mean field, an unrestricted reference.

Water/cc-pVDZ, DF on the explicit auxiliary basis cc-pvdz-ri.

Results at the defaults (W grid minimax 20, outer Gauss-Legendre 48), max
|imagfrequency - casida| per block:

    HF   'none'   oo 1.97e-9  ov 2.64e-10  vv 1.59e-10
    HF   'dyson'  oo 1.97e-9  ov 7.76e-10  vv 1.59e-10
    PBE0 'none'   oo 7.49e-7  ov 1.26e-9   vv 4.20e-10
    PBE0 'dyson'  oo 7.49e-7  ov 4.02e-7   vv 4.20e-10
    lambda 1e-10 -> 1e-8: max |d dgamma| HF 5.9e-9, PBE0 1.3e-8

GRID STUDY (sets GW_DENSITY_NFREQ), worst block over 'none' and 'dyson', W
grid minimax 20 (minimax 34 gives the same to ~1e-9):

    outer             16      24      32      48      64
    GL   HF        1.7e-4  1.5e-5  2.9e-7  2.0e-9  3.6e-10
    GL   PBE0      5.0e-4  1.7e-4  1.8e-5  7.5e-7  5.4e-9
    minimax HF     1.4e-4  3.6e-10 3.6e-10   -       -
    minimax PBE0   3.0e-4  4.9e-10 4.7e-10   -       -

48 is the smallest Gauss-Legendre size within 3e-6 on both. The tabulated
outer minimax grid reaches the floor already at 24 points; the default stays
Gauss-Legendre (reported, not switched). Before the fit, the plain W-grid
quadrature of the QP route left 3e-4 (HF) and 1e-3 (PBE0) at these grids and
never reached 1e-5 below inner Gauss-Legendre 256.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import dft, gto, scf
from scipy.integrate import quad

from src.Base.environment_seam import attach_environment
from src.Base.numerics.constants import DEFAULT_BROADENING_ETA
from src.Base.numerics.numerics import Numerics
from src.Base.orbital_spaces import particle_hole_window
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies)
from src.Base.utils.grids.grids import minimax_frequency_grid
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW import imaginary_axis
from src.SingleReference.GW.density_matrix import (GWDensityMatrixSolver,
                                                   compute_gw_density_matrix,
                                                   outer_frequency_grid)
from src.SingleReference.GW.imaginary_axis import (
    correlation_self_energy_matrix_imaginary_axis, one_pole_convolution,
    screened_pair_weights, self_energy_imaginary_axis)
from src.environments.solvent_screening import SolventScreening
from tests.molecules import WATER

AUXBASIS = 'cc-pvdz-ri'
TOL = 1e-5


def water(basis='cc-pvdz'):
    return gto.M(atom=WATER, basis=basis, verbose=0)


def tight(mf):
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
    return mf.run()


def fitted(mf):
    return tight(mf.density_fit(auxbasis=AUXBASIS))


def mo_dgamma(mf, dm):
    """C^T S (dm - gamma_mf) S C: the MO-basis density correction."""
    S = mf.mol.intor('int1e_ovlp')
    C = mf.mo_coeff
    return C.T @ S @ (dm - mf.make_rdm1()) @ S @ C


def report(name, ok, detail=''):
    print(f"{name}: {'OK' if ok else 'FAIL'} {detail}")
    return ok


def block_errors(mf, relaxation, numerics=None):
    """max |imagfrequency - casida| per block (oo, ov, vv) of Delta gamma_MO,
    and the largest element of the casida Delta gamma."""
    numerics = Numerics() if numerics is None else numerics
    nocc = mf.mol.nelectron // 2
    casida = mo_dgamma(mf, compute_gw_density_matrix(
        mf, mode='casida', relaxation=relaxation, numerics=numerics))
    imag = mo_dgamma(mf, compute_gw_density_matrix(
        mf, mode='imagfrequency', relaxation=relaxation, numerics=numerics))
    o, v = slice(0, nocc), slice(nocc, None)
    diff = np.abs(imag - casida)
    return ({'oo': diff[o, o].max(), 'ov': diff[o, v].max(), 'vv': diff[v, v].max()},
            np.abs(casida).max())


def check_pair_weights_diagonal_matches_qp_sigma(mf):
    """screened_pair_weights is the QP convolution's weight: its diagonal p = q,
    under the plain quadrature, is self_energy_imaginary_axis's Sigma_pp."""
    mol = mf.mol
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    e_min, e_max, mu = particle_hole_window(eps, nocc)
    points, weights = minimax_frequency_grid(Numerics().imagfreq.nfreq, e_min, e_max)
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted',
                              eta=DEFAULT_BROADENING_ETA)
    W = lr.solve_rpa_screening(points, nocc, is_imaginary=True)
    query = np.array([0.0, 0.05, 0.3, 1.1, 4.0])
    em = eps - mu
    full = np.zeros((len(query), len(eps), len(eps)), dtype=complex)
    for k, wk in enumerate(points):
        g = 1.0 / (1j * (query[:, None] + wk) - em) + 1.0 / (1j * (query[:, None] - wk) - em)
        full += weights[k] * np.einsum('qm,mpr->qpr', g, screened_pair_weights(coeff, W[k]))
    full *= -1.0 / (2.0 * np.pi)
    err = max(np.abs(full[:, p, p] - self_energy_imaginary_axis(
        coeff, em, nocc, p, points, weights, W, query)).max() for p in range(len(eps)))
    return report('pair weights: plain-quadrature diagonal == QP Sigma_pp (5 points)',
                  err < 1e-12, f'(max |diff| {err:.2e})')


def check_one_pole_kernel():
    """-(1/2pi) int G(i(w + w')) 2 Om/(w'^2 + Om^2) dw' against scipy quad."""
    worst = 0.0
    for e in (0.5, -0.3, 2.0, -7.0):
        for om in (0.1, 0.7, 5.0):
            for w in (0.0, 0.3, 4.0):
                def f(x, part):
                    return part(1.0 / (1j * (w + x) - e)) * 2 * om / (x * x + om * om)
                num = -(quad(f, -np.inf, np.inf, args=(np.real,), limit=500,
                             epsabs=1e-13, epsrel=1e-12)[0]
                        + 1j * quad(f, -np.inf, np.inf, args=(np.imag,), limit=500,
                                    epsabs=1e-13, epsrel=1e-12)[0]) / (2 * np.pi)
                ana = one_pole_convolution(np.array([e]), np.array([om]), w)[0, 0, 0]
                worst = max(worst, abs(num - ana) / abs(ana))
    return report('one-pole kernel == direct integral (36 cases)', worst < 1e-9,
                  f'(max rel diff {worst:.2e})')


def check_fitted_sigma_matches_casida_poles(mf, name):
    """The fitted Sigma_pq(mu + i w) on the outer nodes against the exact
    sum over the Casida RPA poles of the same screening."""
    mol = mf.mol
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    points, _ = outer_frequency_grid(eps, nocc, Numerics())
    sigma, mu = correlation_self_energy_matrix_imaginary_axis(mf, mol, nocc, points, Numerics())
    solver = GWDensityMatrixSolver(
        eps, df_coeff=get_density_fitting_coefficients(mol, mf, representation='spatial'))
    omega, X, Y = solver.build_screening(nocc)
    w = solver.get_chi_a(nocc, X, Y, p_state=None)
    shift = np.where(np.arange(len(eps)) < nocc, 1.0, -1.0)
    exact = np.array([2.0 * np.einsum(
        'spm,sqm,sm->pq', w, w,
        1.0 / (1j * x - (eps - mu)[None, :] + shift[None, :] * omega[:, None]))
        for x in points])
    err = np.abs(sigma - exact).max()
    return report(f'{name}: fitted Sigma_pq == Casida-pole Sigma_pq on the outer grid',
                  err < 1e-4, f'(max |diff| {err:.2e}, max |Sigma| {np.abs(exact).max():.2e})')


def check_matches_casida(mf, relaxation, name):
    errs, scale = block_errors(mf, relaxation)
    ok = all(e <= TOL for e in errs.values())
    return report(f"{name} {relaxation!r}: imagfrequency == casida per block",
                  ok, '(' + ', '.join(f'{k} {e:.2e}' for k, e in errs.items())
                  + f'; max |dgamma| {scale:.2e})')


def check_non_df_mean_field(hf_exact):
    errs, scale = block_errors(hf_exact, 'none')
    ok = all(e <= TOL for e in errs.values())
    return report("RHF without density fitting: imagfrequency == casida per block",
                  ok, '(' + ', '.join(f'{k} {e:.2e}' for k, e in errs.items()) + ')')


def check_tikhonov_robustness(mf, name):
    """The density moves < 1e-6 between the fit's lambda = 1e-10 and 1e-8."""
    def dgamma(relaxation):
        return mo_dgamma(mf, compute_gw_density_matrix(
            mf, mode='imagfrequency', relaxation=relaxation, numerics=Numerics()))
    ok = True
    for relaxation in ('none', 'dyson'):
        base = dgamma(relaxation)
        saved = imaginary_axis.GW_DENSITY_W_FIT_TIKHONOV
        imaginary_axis.GW_DENSITY_W_FIT_TIKHONOV = 1e-8
        try:
            moved = dgamma(relaxation)
        finally:
            imaginary_axis.GW_DENSITY_W_FIT_TIKHONOV = saved
        err = np.abs(moved - base).max()
        ok &= report(f'{name} {relaxation!r}: lambda 1e-10 -> 1e-8 moves dgamma < 1e-6',
                     err < 1e-6, f'(max |diff| {err:.2e})')
    return ok


def raises(exc, words, call):
    try:
        call()
    except exc as err:
        if all(w in str(err) for w in words):
            return True
        print(f'   wrong message: {err}')
        return False
    except Exception as err:  # noqa: BLE001 -- report the wrong type
        print(f'   wrong exception: {type(err).__name__}: {err}')
        return False
    return False


def check_refusals(hf):
    numerics = Numerics()

    def imag(mf, **kw):
        kw.setdefault('numerics', numerics)
        return lambda: compute_gw_density_matrix(mf, mode='imagfrequency',
                                                 relaxation='none', **kw)
    ok = report("polarizability='BSE' refused", raises(
        NotImplementedError, ['RPA'], imag(hf, polarizability='BSE')))
    ok &= report('df=False refused', raises(
        NotImplementedError, ['df=False'], imag(hf, df=False)))
    solvated = attach_environment(copy.copy(hf),
                                  SolventScreening(hf.mol, eps=2.0, eps_static=2.0))
    ok &= report('an environment refused', raises(
        NotImplementedError, ['environment'], imag(solvated)))
    ok &= report('missing numerics refused', raises(
        TypeError, ['Numerics'], imag(hf, numerics=None)))
    smeared = copy.copy(hf)
    smeared.sigma, smeared.smearing_method = 0.01, 'fermi'
    ok &= report('a thermally smeared mean field refused', raises(
        NotImplementedError, ['T = 0'], imag(smeared)))
    ok &= report('UHF refused', raises(
        NotImplementedError, ['restricted'], imag(scf.UHF(hf.mol))))
    return ok


def run():
    hf = fitted(scf.RHF(water()))
    pbe0 = fitted(dft.RKS(water(), xc='pbe0'))
    hf_exact = tight(scf.RHF(water()))
    results = [check_pair_weights_diagonal_matches_qp_sigma(hf), check_one_pole_kernel()]
    for mf, name in ((hf, 'HF'), (pbe0, 'PBE0')):
        results.append(check_fitted_sigma_matches_casida_poles(mf, name))
        for relaxation in ('none', 'dyson'):
            results.append(check_matches_casida(mf, relaxation, name))
        results.append(check_tikhonov_robustness(mf, name))
    results.append(check_non_df_mean_field(hf_exact))
    results.append(check_refusals(hf))
    return all(results)


def test_gw_density_imagfrequency():
    assert run()


if __name__ == '__main__':
    ok = run()
    print("\nALL PASSED" if ok else "\nFAILURES DETECTED")
    sys.exit(0 if ok else 1)
