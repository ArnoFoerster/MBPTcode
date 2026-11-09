"""The GW density on the imaginary-frequency axis (mode='imagfrequency', N^4).

- The full-matrix correlation self-energy Sigma_c,pq(mu + i w): its pair
  weights w^m_pq are the QP route's (their diagonal under the plain W-grid
  quadrature is `self_energy_diagonal_imaginary_axis`'s Sigma_pp); its inner integral
  is a Tikhonov-regularized Lorentzian fit of W_c on the W grid convolved
  exactly with G (the one-pole kernel against scipy quad), and it matches the
  Sigma summed over the Casida RPA poles.
- On a mean field without density fitting (the exact integrals'
  decomposition), 'imagfrequency' reproduces 'casida' to 1e-5 per element in
  every block (oo, ov, vv) of Delta gamma_MO: both routes screen with the
  same full RPA on the same three-index object. The density-fitted
  comparison, per relaxation on all three routes, is in test_gw_density_none,
  test_gw_density_dyson and test_gw_density_zvector.
- The fit's regularization: lambda 1e-10 -> 1e-8 moves Delta gamma < 1e-6.
- Refusals are in test_gw_density_keywords.

- The default outer grid (minimax 34) reproduces 'casida' to 1e-6 on HBr,
  a wide particle-hole window, where Gauss-Legendre 48 does not.
- The minimax outer grid is fitted to [gap/2, span], the window its
  integrand has: minimax 24 on HBr, which reads its table row by the window
  rather than the stretched first row the default 34 reads there, reaches
  the same 1e-7 (it was 8.8e-5 on [gap, span]).

Water/cc-pVDZ, DF on the explicit auxiliary basis cc-pvdz-ri; HBr/cc-pVDZ
for the wide window.

Results at the defaults (W grid minimax 20, outer minimax 34), max
|imagfrequency - casida| per block (the comparison itself runs in the
per-relaxation files):

    HF   'none'   oo 3.62e-10  ov 2.54e-10  vv 1.65e-10
    HF   'dyson'  oo 3.62e-10  ov 3.20e-10  vv 1.65e-10
    PBE0 'none'   oo 3.87e-10  ov 2.97e-10  vv 3.59e-10
    PBE0 'dyson'  oo 3.87e-10  ov 4.83e-10  vv 3.59e-10
    lambda 1e-10 -> 1e-8: max |d dgamma| HF 5.9e-9, PBE0 1.3e-8

FIRST GRID STUDY (water), worst block over 'none' and 'dyson', W grid
minimax 20 (minimax 34 gives the same to ~1e-9):

    outer             16      24      32      48      64
    GL   HF        1.7e-4  1.5e-5  2.9e-7  2.0e-9  3.6e-10
    GL   PBE0      5.0e-4  1.7e-4  1.8e-5  7.5e-7  5.4e-9
    minimax HF     1.4e-4  3.6e-10 3.6e-10   -       -
    minimax PBE0   3.0e-4  4.9e-10 4.7e-10   -       -

It set Gauss-Legendre 48 as the first default. Before the fit, the plain
W-grid quadrature of the QP route left 3e-4 (HF) and 1e-3 (PBE0) at these
grids and never reached 1e-5 below inner Gauss-Legendre 256.

OUTER-GRID STUDY (sets GW_DENSITY_NFREQ and GW_DENSITY_GRID): max
|imagfrequency - casida| over the whole Delta gamma_MO, 'none', W grid
minimax 20, against the ratio span/gap of the particle-hole window; minimax
on [gap/2, span] (scripts and full output: design notes,
runs/2026-11-06-gw-density-outer-grid and -outer-window):

    system (cc-pVDZ)     ratio   GL48     GL96     MM16     MM24     MM34
    H2O        HF          36    2.0e-9   3.6e-10  1.2e-9   3.6e-10  3.6e-10
    H2O        PBE0        62    7.5e-7   3.9e-10  1.2e-9   3.8e-10  3.9e-10
    C6H6       PBE0        53    1.8e-8   5.6e-10  9.0e-10  5.6e-10  5.4e-10
    C10H8      PBE0        78    2.0e-6   1.8e-9   1.5e-8   1.8e-9   1.8e-9
    NH3 (aug-cc-pVTZ) PBE0 111   1.3e-6   2.9e-9   1.9e-9   2.0e-9   2.0e-9
    H2S        HF         174    5.4e-6   6.5e-9   1.4e-8   6.5e-9   6.5e-9
    H2S        PBE0       296    1.1e-5   2.9e-8   1.0e-7   8.5e-9   8.5e-9
    HBr        HF         881    3.3e-6   7.3e-7   1.6e-8   1.6e-8   1.6e-8
    HBr        PBE0      1529    1.9e-5   5.8e-7   1.2e-7   1.3e-7   1.3e-7
    Br2        HF        1195    6.1e-6   5.4e-7   1.7e-7   1.7e-7   1.7e-7
    Br2        PBE0      2937    2.0e-5   3.7e-6   7.0e-7   7.1e-7   7.1e-7

The ratio sets the grid, not the system size; the Gauss-Legendre error sits
in the oo block (the deep cores, far above its scale w0 = gap/2). Minimax
24 reaches the floor everywhere (16 within 1e-7). On HBr and Br2 that floor
is the W grid's: minimax 32 there takes HBr to 4e-10 and Br2 HF to 3e-10,
and Br2 PBE0 reaches 3.9e-10 at W and outer minimax 34. Run on [gap, span]
instead, minimax 24 left 2.1e-5 (HBr HF) and 8.8e-5 (HBr PBE0): the
integrand's frontier Lorentzians have width |eps - mu| = gap/2. At 34
points every system here is below the table's narrowest ratio, where the
stretched first row ignores e_min, so the window does not change the
default's numbers. The default is 34, the largest tabulated size, at the
cost of minimax 24 and Gauss-Legendre 48 (naphthalene PBE0: 5.4 s, 5.4 s,
5.5 s; the space-time route alike). Not 32: most rows of GreenX's 32-point
frequency table miss their own target by 2-11 % (rows serving
e_max/e_min = 5846-6665 and >= 10044); on Br2 PBE0, where [gap/2, span]
reaches 5875, outer minimax 32 gives 3.7e-7 where 24 and 34 give ~1e-9 (W
grid minimax 34).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf
from scipy.integrate import quad

from src.Base.numerics.constants import DEFAULT_BROADENING_ETA
from src.Base.numerics.numerics import Numerics
from src.Base.orbital_spaces import particle_hole_window
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies)
from src.Base.utils.grids.grids import minimax_frequency_grid
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW import imaginary_axis
from src.SingleReference.GW.density_matrix import (casida_screening,
                                                   compute_gw_density_matrix,
                                                   outer_frequency_grid,
                                                   screened_coulomb_residues)
from src.SingleReference.GW.imaginary_axis import (
    self_energy_matrix_imaginary_axis, one_pole_convolution,
    screened_pair_weights, self_energy_diagonal_imaginary_axis)
from tests.gw_density_routes import converged, fitted, mo_dgamma, water

TOL = 1e-5


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


@pytest.fixture(scope='module')
def hf():
    return fitted(scf.RHF(water()))


@pytest.fixture(scope='module', params=['HF', 'PBE0'])
def reference(request, hf):
    """The density-fitted water mean field of each reference, with its name."""
    mf = hf if request.param == 'HF' else fitted(dft.RKS(water(), xc='pbe0'))
    return request.param, mf


def test_pair_weights_diagonal_matches_qp_sigma(hf):
    """screened_pair_weights is the QP convolution's weight: its diagonal p = q,
    under the plain quadrature, is self_energy_diagonal_imaginary_axis's Sigma_pp."""
    mf = hf
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
    err = max(np.abs(full[:, p, p] - self_energy_diagonal_imaginary_axis(
        coeff, em, nocc, p, points, weights, W, query)).max() for p in range(len(eps)))
    print(f'plain-quadrature diagonal vs QP Sigma_pp (5 points): max |diff| {err:.2e}')
    assert err < 1e-12


def test_one_pole_kernel():
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
    print(f'one-pole kernel vs direct integral (36 cases): max rel diff {worst:.2e}')
    assert worst < 1e-9


def test_fitted_sigma_matches_casida_poles(reference):
    """The fitted Sigma_pq(mu + i w) on the outer nodes against the exact
    sum over the Casida RPA poles of the same screening."""
    name, mf = reference
    mol = mf.mol
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    points, _ = outer_frequency_grid(eps, nocc, Numerics())
    sigma, mu = self_energy_matrix_imaginary_axis(mf, mol, nocc, points, Numerics())
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    omega, X, Y = casida_screening(eps, nocc, df_coeff=df_coeff)
    w = screened_coulomb_residues(eps, nocc, X, Y, df_coeff=df_coeff)
    shift = np.where(np.arange(len(eps)) < nocc, 1.0, -1.0)
    exact = np.array([2.0 * np.einsum(
        'spm,sqm,sm->pq', w, w,
        1.0 / (1j * x - (eps - mu)[None, :] + shift[None, :] * omega[:, None]))
        for x in points])
    err = np.abs(sigma - exact).max()
    print(f'{name}: fitted vs Casida-pole Sigma_pq on the outer grid: max |diff| '
          f'{err:.2e}, max |Sigma| {np.abs(exact).max():.2e}')
    assert err < 1e-4


def test_non_df_mean_field():
    """Without density fitting, imagfrequency == casida per block."""
    hf_exact = converged(scf.RHF(water()))
    errs, _ = block_errors(hf_exact, 'none')
    print('RHF without density fitting: '
          + ', '.join(f'{k} {e:.2e}' for k, e in errs.items()))
    assert all(e <= TOL for e in errs.values()), errs


@pytest.mark.parametrize('relaxation', ['none', 'dyson'])
def test_tikhonov_robustness(reference, relaxation):
    """The density moves < 1e-6 between the fit's lambda = 1e-10 and 1e-8."""
    name, mf = reference

    def dgamma():
        return mo_dgamma(mf, compute_gw_density_matrix(
            mf, mode='imagfrequency', relaxation=relaxation, numerics=Numerics()))
    base = dgamma()
    saved = imaginary_axis.GW_DENSITY_W_FIT_TIKHONOV
    imaginary_axis.GW_DENSITY_W_FIT_TIKHONOV = 1e-8
    try:
        moved = dgamma()
    finally:
        imaginary_axis.GW_DENSITY_W_FIT_TIKHONOV = saved
    err = np.abs(moved - base).max()
    print(f'{name} {relaxation!r}: lambda 1e-10 -> 1e-8 moves dgamma {err:.2e}')
    assert err < 1e-6


#: HBr/cc-pVDZ, span/gap 881 (HF) and 1529 (PBE0): a wide window. Reached at
#: the default 1.6e-8 (HF), 1.3e-7 (PBE0); minimax 24 on [gap/2, span] the
#: same, where on [gap, span] it left 2.1e-5 and 8.8e-5.
WIDE_WINDOW_BAR = 1e-6


@pytest.fixture(scope='module', params=['HF', 'PBE0'])
def hbr(request):
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.414', basis='cc-pvdz', verbose=0)
    mf = fitted(scf.RHF(mol) if request.param == 'HF' else dft.RKS(mol, xc='pbe0'))
    casida = mo_dgamma(mf, compute_gw_density_matrix(mf, mode='casida', relaxation='none'))
    return request.param, mf, casida


@pytest.mark.parametrize('nfreq', [None, 24])
def test_minimax_outer_grid_on_a_wide_window(hbr, nfreq):
    """The default outer grid, and minimax 24, whose table row (unlike the
    default's stretched first row) depends on the window's lower end."""
    reference, mf, casida = hbr
    numerics = Numerics() if nfreq is None else Numerics(gwdensity={'nfreq': nfreq})
    imag = mo_dgamma(mf, compute_gw_density_matrix(
        mf, mode='imagfrequency', relaxation='none', numerics=numerics))
    err = np.abs(imag - casida).max()
    print(f"HBr {reference}, outer minimax {numerics.gwdensity.nfreq}: "
          f"max |imagfrequency - casida| {err:.2e} (bar {WIDE_WINDOW_BAR:.0e})")
    assert err < WIDE_WINDOW_BAR


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-v']))
