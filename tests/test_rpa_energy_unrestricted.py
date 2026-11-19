"""The direct RPA correlation energy on the imaginary-frequency axis for two
spin channels (an unrestricted reference), and the frequency grids it and
the screened interaction use.

    E_c = (1/2 pi) sum_k w_k {log det[1 - Z(i w_k)] + Tr Z(i w_k)},

Z = chi0_alpha + chi0_beta in the auxiliary basis, on a grid over the
transition range of both spins: the smallest gap and the largest span of
the two spectra.

The mean fields here are density-fitted UHF calculations (the OH radical and
triplet CH2 in 6-31G), so the three-index factors are those of a real fit,
the mean field's own. The energy is compared with the plasmon formula
1/2 (sum Omega - Tr A) of the Casida problem on the same factors, which has
no quadrature, and with pyscf's URPA on the same density-fitting object; on
a closed shell given as two equal channels it is the restricted energy.

The same energy in imaginary time, on separable (ISDF) factors with one
collocation per spin and one Coulomb factor D: chi0 is the sum of the two
spins' responses on a minimax time grid over the transitions of both. It is
compared with the plasmon formula of the Casida problem on the same ISDF
factors (OH/cc-pVDZ, density-fitted UHF, G1 grid), and with the restricted
function on a closed shell given as two equal channels. The static
(omega = 0) grid of two spins spans the same range.
"""
import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.gw import urpa

from src.Base.factorization.sliced_factors import unpack_factors
from src.Base.numerics.constants import DEFAULT_TAU_TARGET
from src.Base.numerics.numerics import Numerics, rpa_quadrature
from src.Base.orbital_spaces import (particle_hole_window, response_range,
                                     transition_range)
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies)
from src.Base.utils.grids.grids import (gap_scaled_w0, gauss_legendre_grid,
                                        minimax_frequency_grid)
from src.Base.utils.grids.time_frequency import (TimeFrequencyGrid,
                                                 minimax_points_for_accuracy)
from src.SingleReference.GW.imaginary_axis import screening_fit_grid
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import (
    imaginary_axis_energy_grid, rpa_correlation_energy_casida,
    rpa_correlation_energy_imaginary_axis,
    solve_polarizability_imaginary_axis)
from src.SingleReference.LinearResponse.space_time import (
    rpa_correlation_energy_space_time, static_screening_grid)
from tests.downfolding_oracle import CH2, OH, unrestricted, water
from tests.gw_density_routes import fitted
from tests.gw_density_routes import water as gw_water
from tests.isdf_test_grid import g1_numerics


def _two_spin(atom, spin):
    """(mf, lr, eps, nocc) of a density-fitted UHF: the unrestricted solver
    on the mean field's own three-index factors."""
    mf = unrestricted(atom, spin, density_fit=True)
    eps = tuple(np.asarray(e) for e in get_orbital_energies(mf))
    nocc = tuple(int(np.count_nonzero(occ)) for occ in mf.mo_occ)
    factors = get_density_fitting_coefficients(mf.mol, mf, 'spatial')
    lr = LinearResponseSolver(eps, coeff_df=factors, spin_mode='unrestricted')
    return mf, lr, eps, nocc


@pytest.fixture(scope='module')
def oh():
    return _two_spin(OH, 1)


@pytest.fixture(scope='module')
def ch2():
    return _two_spin(CH2, 2)


@pytest.mark.parametrize('name', ['oh', 'ch2'])
def test_two_spin_energy_matches_the_plasmon_formula(request, name):
    """The quadrature against the plasmon formula on the same factors, on
    the default minimax grid (20 points) and on Gauss-Legendre grids (the
    default 40 points and 100). Measured, OH (93 auxiliary functions) and
    CH2 (116): minimax 4.8e-13 and 3.6e-13; Gauss-Legendre 8.6e-9 and
    2.0e-10 with 40 points, 4.8e-13 and 3.2e-12 with 100."""
    _, lr, _, nocc = request.getfixturevalue(name)
    plasmon = rpa_correlation_energy_casida(lr, nocc)
    assert plasmon < -1e-2
    e_c = rpa_correlation_energy_imaginary_axis(lr, nocc)
    assert abs(e_c - plasmon) < 1e-9
    coarse = rpa_correlation_energy_imaginary_axis(lr, nocc,
                                                   grid='gauss_legendre')
    assert abs(coarse - plasmon) < 1e-6
    fine = rpa_correlation_energy_imaginary_axis(
        lr, nocc, Numerics(quadrature={'nfreq_rpa': 100}),
        grid='gauss_legendre')
    assert abs(fine - plasmon) < 1e-9


def test_two_spin_energy_matches_pyscf_urpa(oh):
    """pyscf's URPA integrates the same integrand on the same with_df with
    its own Gauss-Legendre grid. Measured: 1.0e-12 against its 80 points
    (7.0e-10 against its default 40, pyscf's own quadrature error)."""
    mf, lr, _, nocc = oh
    ref = urpa.URPA(mf)
    ref.verbose = 0
    ref.kernel(nw=80)
    assert abs(rpa_correlation_energy_imaginary_axis(lr, nocc)
               - ref.e_corr) < 1e-9


def test_polarizability_is_the_sum_of_the_two_spins(oh):
    """Z(i w) = chi0_alpha + chi0_beta, each spin's
    -2 sum_ia B_ia B_ia^T d_ia / (d_ia^2 + w^2) with its own factors and
    orbital energies and no further spin factor. Measured: 8.3e-16."""
    _, lr, eps, nocc = oh
    points = [0.0, 0.3, 2.0]
    z = solve_polarizability_imaginary_axis(lr, nocc, points)
    for k, w in enumerate(points):
        ref = 0.0
        for b, e, n in zip((lr.coeff_a, lr.coeff_b), eps, nocc):
            b_ov = b[:, :n, n:].reshape(b.shape[0], -1)
            d = (e[n:][None, :] - e[:n][:, None]).ravel()
            ref = ref + (b_ov * (-2.0 * d / (d**2 + w**2))) @ b_ov.T
        assert np.max(np.abs(z[k] - ref)) < 1e-10


def test_closed_shell_two_spin_energy_is_the_restricted_one():
    """A closed shell as two equal channels: the same grid, the same Z, the
    same energy as the restricted call. Measured: 0 on both grids (the
    factors here are the exact integrals' decomposition; water has no
    density fit)."""
    mf = water()
    eps, nocc = np.asarray(mf.mo_energy), 5
    b = get_density_fitting_coefficients(mf.mol, mf, 'spatial')
    restricted = LinearResponseSolver(eps, coeff_df=b, spin_mode='restricted')
    two = LinearResponseSolver((eps, eps), coeff_df=(b, b),
                               spin_mode='unrestricted')
    for grid in ('minimax', 'gauss_legendre'):
        one = rpa_correlation_energy_imaginary_axis(restricted, nocc,
                                                    grid=grid)
        both = rpa_correlation_energy_imaginary_axis(two, (nocc, nocc),
                                                     grid=grid)
        assert abs(both - one) < 1e-10
        for ours, ref in zip(
                imaginary_axis_energy_grid((eps, eps), (nocc, nocc),
                                           grid=grid),
                imaginary_axis_energy_grid(eps, nocc, grid=grid)):
            assert np.array_equal(ours, ref)
    for ours, ref in zip(screening_fit_grid((eps, eps), (nocc, nocc)),
                         screening_fit_grid(eps, nocc)):
        assert np.array_equal(ours, ref)


def test_two_spin_grids(oh):
    """The grids of two spin channels span the transition range of both:
    the smallest gap and the largest span over the two spectra, which here
    come from different spins (OH: gaps 0.768 alpha, 0.630 beta; spans
    22.138 alpha, 22.143 beta)."""
    _, _, eps, nocc = oh
    gaps = [particle_hole_window(e, n)[:2] for e, n in zip(eps, nocc)]
    e_min, e_max = min(g[0] for g in gaps), max(g[1] for g in gaps)
    assert (e_min, e_max) == transition_range(eps, nocc)
    assert response_range(eps, nocc) == (e_min, e_max)
    assert response_range(eps[0], nocc[0]) == gaps[0]
    assert gaps[0][0] != gaps[1][0] and gaps[0][1] != gaps[1][1]

    numerics = Numerics(quadrature={'nfreq_rpa': 20})
    nfreq = rpa_quadrature(numerics, 'minimax')[1]
    assert nfreq == 20
    points, weights = imaginary_axis_energy_grid(eps, nocc, numerics)
    ref = minimax_frequency_grid(nfreq, e_min, e_max)
    assert np.array_equal(points, ref[0]) and np.array_equal(weights, ref[1])
    one_spin = imaginary_axis_energy_grid(eps[0], nocc[0], numerics)
    assert not np.array_equal(points, one_spin[0])

    points, weights = imaginary_axis_energy_grid(eps, nocc, numerics,
                                                 grid='gauss_legendre')
    ref = gauss_legendre_grid(20, w0=0.5 * e_min)
    assert np.array_equal(points, ref[0]) and np.array_equal(weights, ref[1])

    coarse = Numerics(imagfreq={'nfreq': 26})
    points, weights, poles = screening_fit_grid(eps, nocc, coarse)
    ref = minimax_frequency_grid(26, e_min, e_max)
    assert np.array_equal(points, ref[0]) and np.array_equal(weights, ref[1])
    assert len(poles) == 26
    assert abs(poles[0] / e_min - screening_fit_grid(
        eps[0], nocc[0], coarse)[2][0] / gaps[0][0]) < 1e-12
    assert np.all(np.diff(poles) > 0)


# --- imaginary time, separable (ISDF) factors ----------------------------------

@pytest.fixture(scope='module')
def oh_isdf():
    """(x, d, eps, nocc) of OH/cc-pVDZ on a density-fitted UHF (cc-pvdz-ri):
    the separable factors on the G1 grid, x (2, M, nmo) one collocation per
    spin and d (M, naux) shared, so that the fitted v between spins s and t
    is B^s B^t with B^s_Q,pq = sum_P d_PQ x^s_Pp x^s_Pq."""
    mol = gto.M(atom=OH, basis='cc-pvdz', spin=1, cart=False, verbose=0)
    mf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    x, d = unpack_factors(separable_factors(mf, mol, numerics=g1_numerics(mf),
                                            comm=None))[:2]
    eps = tuple(np.asarray(e) for e in mf.mo_energy)
    nocc = tuple(int(np.count_nonzero(occ)) for occ in mf.mo_occ)
    return x, d, eps, nocc


def _energy_grid(eps, nocc, nfreq):
    """A minimax time grid over the transitions of `eps` (one spectrum or
    the pair) with a Gauss-Legendre frequency axis scaled to the smallest
    gap, as the space-time energy takes it."""
    e_min, e_max = response_range(eps, nocc)
    ntau = minimax_points_for_accuracy(e_min, e_max, DEFAULT_TAU_TARGET)[0]
    nu, wt = gauss_legendre_grid(nfreq, gap_scaled_w0(gap=e_min))
    return TimeFrequencyGrid.minimax_split(ntau, e_min, e_max, nu, wt,
                                           with_sine=False,
                                           with_inverse=False)


def test_two_spin_space_time_energy_matches_the_plasmon_formula(oh_isdf):
    """chi0_alpha + chi0_beta in imaginary time against the plasmon formula
    1/2 (sum Omega - Tr A) of the two-spin Casida problem built here on the
    same ISDF factors (A = delta + K, B = K, K = B_ov^T B_ov over the
    stacked pairs), with 40 and with 80 frequency points. Measured (400
    grid points, 70 auxiliary functions, 130 pairs, 14 time points; E_c =
    -0.1845): 1.0e-8 with 40 frequencies, 1.1e-12 with 80."""
    x, d, eps, nocc = oh_isdf
    assert x.shape[0] == 2 and x.shape[1] == d.shape[0]
    b_ov, delta = [], []
    for x_s, eps_s, n in zip(x, eps, nocc):
        pairs = (x_s[:, :n, None] * x_s[:, None, n:]).reshape(len(x_s), -1)
        b_ov.append(d.T @ pairs)
        delta.append((eps_s[n:][None, :] - eps_s[:n][:, None]).ravel())
    b_ov = np.concatenate(b_ov, axis=1)
    k = b_ov.T @ b_ov
    a = k + np.diag(np.concatenate(delta))
    plasmon = 0.5 * (CasidaSolver(a, k.copy()).solve()[0].sum() - np.trace(a))
    assert plasmon < -0.1
    coarse = rpa_correlation_energy_space_time(
        x, d, eps, nocc, _energy_grid(eps, nocc, 40), comm=None)
    assert abs(coarse - plasmon) < 1e-6
    fine = rpa_correlation_energy_space_time(
        x, d, eps, nocc, _energy_grid(eps, nocc, 80), comm=None)
    assert abs(fine - plasmon) < 1e-9


def test_closed_shell_two_spin_space_time_energy_is_the_restricted_one():
    """A closed shell (water/cc-pVDZ, density fitted, G1 grid) given as two
    equal channels: the restricted function's energy on the same grid
    (measured difference 0), and no mu is taken for two channels."""
    mf = fitted(scf.RHF(gw_water()))
    x, d = unpack_factors(separable_factors(mf, mf.mol,
                                            numerics=g1_numerics(mf),
                                            comm=None))[:2]
    eps, nocc = np.asarray(mf.mo_energy), mf.mol.nelectron // 2
    grid = _energy_grid(eps, nocc, 40)
    one = rpa_correlation_energy_space_time(x, d, eps, nocc, grid, comm=None)
    flags = {}
    both = rpa_correlation_energy_space_time(
        (x, x), d, (eps, eps), (nocc, nocc), grid, comm=None, flags=flags)
    assert one < -0.1 and abs(both - one) < 1e-10
    assert flags == {}
    with pytest.raises(ValueError, match='mid-gap'):
        rpa_correlation_energy_space_time(
            (x, x), d, (eps, eps), (nocc, nocc), grid, mu=0.0, comm=None)


def test_two_spin_static_grid(oh_isdf):
    """The one-point (omega = 0) minimax grid of a static screening for two
    spin channels: over the smallest gap and the largest span of the two
    spectra, with the time points asked for or, by default, the minimax
    count of that range at the tau target. For two equal spectra it is the
    restricted grid."""
    _, _, eps, nocc = oh_isdf
    e_min, e_max = transition_range(eps, nocc)
    grid = static_screening_grid(eps, nocc)
    assert grid.nfreq == 1 and grid.omega_points[0] == 0.0
    ntau = minimax_points_for_accuracy(e_min, e_max, DEFAULT_TAU_TARGET)[0]
    assert grid.ntau == ntau
    ref = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max, [0.0], [1.0],
                                          with_sine=False, with_inverse=False)
    assert np.array_equal(grid.tau_points, ref.tau_points)
    assert np.array_equal(grid.cosft_wt, ref.cosft_wt)
    assert static_screening_grid(eps, nocc, ntau=10).ntau == 10
    one = static_screening_grid(eps[0], nocc[0])
    same = static_screening_grid((eps[0], eps[0]), (nocc[0], nocc[0]))
    assert np.array_equal(same.tau_points, one.tau_points)
    assert np.array_equal(same.cosft_wt, one.cosft_wt)
    assert not np.array_equal(grid.tau_points, one.tau_points)
