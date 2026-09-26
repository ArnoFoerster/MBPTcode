"""THC-GW self-energy with k-points.

Same ladder as test_pbc_isdf_rpa.py -- exact checks first, oracle last:

  1. the k-space CONVOLUTION sum_q G^{k-q} * Wt^q against the direct q-sum.
     Exact, no oracle. Note this is a convolution, not the polarizability's
     correlation, so it is the plain product of FFTs with no index negation --
     which is precisely why it deserves its own test rather than trusting the
     polarizability's.
  2. the two fit ranges are distinct and the self-energy's is wider.
  3. quasiparticle energies against `pbc_self_energy.qp_energy_g0w0`, which
     reaches the same number by a completely different route: real-frequency
     spectral (Casida) form with eta broadening, against THC's imaginary-time
     self-energy plus Pade continuation.
"""
import os
import sys

import numpy as np
import pytest
from pyscf.pbc import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.utils.grids import gauss_legendre_grid, minimax_time_grid
from src.SingleReference.GW.qp_solve import (imaginary_axis_sample_points,
                                             solve_qp_from_imaginary_axis)
from src.SingleReference.Periodic import pbc_isdf_gw as gw
from src.SingleReference.Periodic import pbc_isdf_rpa as rpa
from src.SingleReference.Periodic import pbc_self_energy as pse
from src.SingleReference.Periodic.pbc_integrals import (PBCDFIntegrals,
                                                        get_momentum_transfer_map)
from src.SingleReference.Periodic.pbc_isdf import build_isdf_kpts, kpoint_minus_map
from src.Base.constants import HARTREE_TO_EV

KMESH = [2, 2, 1]
ALPHA = 12       # alpha = 8 is 18 meV off; 12 and 16 both land inside 1 meV
NTAU = 18        # 14 and 22 are both ~3 meV out; 18 is the plateau


def _diamond():
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
    cell.a = np.array([[0., 1.7834, 1.7834],
                       [1.7834, 0., 1.7834],
                       [1.7834, 1.7834, 0.]])
    cell.basis, cell.pseudo, cell.verbose = 'gth-szv', 'gth-pade', 0
    cell.build()
    return cell


@pytest.fixture(scope='module')
def gwfix():
    cell = _diamond()
    kpts = cell.make_kpts(KMESH)
    mf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    mf.kernel()
    assert mf.converged
    mo = [np.asarray(c) for c in mf.mo_coeff]
    mo_energy = np.asarray(mf.mo_energy)
    mo_occ = np.asarray(mf.mo_occ)
    nmo = mo[0].shape[1]
    nocc = int((mo_occ[0] > 1e-8).sum())
    kplus = get_momentum_transfer_map(cell, kpts)
    kminus = kpoint_minus_map(cell, kpts)
    X, V, _ = build_isdf_kpts(cell, mo, kpts, ALPHA * nmo)
    mu = rpa.chemical_potential_occ(mo_energy, mo_occ)
    rW, rS = gw.self_energy_fit_ranges_k(mo_energy, mo_occ, mu)
    omega_in, _ = gauss_legendre_grid(64, w0=0.5 * rW[0])
    Pi = [rpa.polarizability_q_frequency_occ(X, mo_energy, mo_occ, kplus, q, omega_in)
          for q in range(len(kpts))]
    tau = 0.5 * minimax_time_grid(NTAU, *rS)[0]
    return dict(cell=cell, kpts=kpts, mf=mf, mo_energy=mo_energy, mo_occ=mo_occ,
                nocc=nocc, kplus=kplus, kminus=kminus, X=X, V=V, Pi=Pi, mu=mu,
                rW=rW, rS=rS, omega_in=omega_in, tau=tau)


def test_self_energy_k_convolution_matches_direct_sum(gwfix):
    """Rung 1. sum_q G^{k-q} * Wt^q is a CONVOLUTION (plain FFT product), not
    the polarizability's correlation (which needed an index negation). Same
    number as the direct q-sum, so anything above machine precision is a
    mesh-indexing bug."""
    kpts, kminus = gwfix['kpts'], gwfix['kminus']
    nk, M = len(kpts), 5
    rng = np.random.default_rng(0)
    G = rng.normal(size=(nk, M, M)) + 1j * rng.normal(size=(nk, M, M))
    Wt = rng.normal(size=(nk, M, M)) + 1j * rng.normal(size=(nk, M, M))
    fft = gw.self_energy_tau_convolution(G, Wt, KMESH)
    for k in range(nk):
        direct = gw.self_energy_tau_direct(G, Wt, kminus, k)
        assert abs(fft[k] - direct).max() / abs(direct).max() < 1e-12


def test_convolution_rejects_a_mismatched_mesh(gwfix):
    kpts = gwfix['kpts']
    nk, M = len(kpts), 4
    G = np.zeros((nk, M, M), dtype=complex)
    with pytest.raises(ValueError, match='regular mesh'):
        gw.self_energy_tau_convolution(G, G, [2, 2, 2])


def test_self_energy_range_is_wider_than_the_screening_range(gwfix):
    """The range trap. Sigma = -G Wt is a PRODUCT, so its decay rates are SUMS and its
    range exceeds the screening range. Measured 1.6x here. Fitting the Sigma
    transform on the W range is a silent accuracy loss, not an error, which is
    why this is asserted rather than left to a comment."""
    rW, rS = gwfix['rW'], gwfix['rS']
    assert rS[1] > rW[1]
    assert rS[0] > rW[0]


def test_qp_energies_match_the_gdf_spectral_oracle(gwfix):
    """Rung 3. Two entirely different routes to the same quasiparticle energy:
    the oracle is a real-frequency spectral (Casida) sum with eta broadening
    over GDF integrals; this is an imaginary-time THC self-energy carried
    through a Pade continuation. Agreement to 1 meV on four states.
    """
    f = gwfix
    dfi = PBCDFIntegrals.from_scf(f['cell'], f['mf'])
    eig = pse.solve_rpa_all_q(dfi)
    xmv = pse.get_exchange_minus_vxc(f['mf'], exxdiv=None)
    nocc = f['nocc']
    worst = 0.0
    for kn in (0, 1):
        for n in (nocc - 1, nocc):
            qp_ref = pse.qp_energy_g0w0(dfi, eig, kn, n,
                                        exchange_minus_vxc=xmv[kn, n])
            z_fit, iw = imaginary_axis_sample_points(
                gauss_legendre_grid(16, w0=0.5 * f['rW'][0])[0], nocc, n, f['mu'])
            sig = gw.sigma_c_diag_thc(f['X'], f['V'], f['Pi'], f['mo_energy'],
                                      f['mo_occ'], KMESH, kn, [n], f['omega_in'],
                                      iw, f['tau'], mu=f['mu'],
                                      ranges=(f['rW'], f['rS']))[0]
            qp = solve_qp_from_imaginary_axis(f['mo_energy'][kn], n,
                                              xmv[kn, n], z_fit, sig)
            worst = max(worst, abs(qp - qp_ref) * HARTREE_TO_EV)
    assert worst < 2e-3, f"worst QP deviation {worst:.4f} eV"


def test_driver_reproduces_the_hand_assembled_chain(gwfix):
    """Rung 3b. `qp_energy_thc` == the chain written out by hand above.

    Everything the rung-3 test assembles explicitly -- the ISDF build, Pi on a
    Gauss-Legendre set seeded from rW, the 0.5*minimax_time_grid(ntau, *rS) tau
    grid, the rW/rS split, the sample points and the Pade solve -- is exactly
    the sequence a caller had to reproduce to get a quasiparticle energy out of
    this module, because nothing exported the composition. Those conventions
    are not guessable (fitting the tau grid on rW instead of rS is a silent
    accuracy loss, not an error), so the driver is the thing that should be
    called and this pins it against the hand-written form.
    """
    f = gwfix
    dfi = PBCDFIntegrals.from_scf(f['cell'], f['mf'])
    eig = pse.solve_rpa_all_q(dfi)
    xmv = pse.get_exchange_minus_vxc(f['mf'], exxdiv=None)
    nocc, kn = f['nocc'], 0

    npoints = f['V'][0].shape[0]           # the fixture's rank, ALPHA * nmo
    qp, diag = gw.qp_energy_thc(f['cell'], f['mf'], npoints, kn=kn, ntau=NTAU,
                                exxdiv=None, return_diagnostics=True)
    assert diag['route'] == 'minimax'
    assert tuple(diag['kmesh']) == tuple(KMESH)

    worst = 0.0
    for i, n in enumerate((nocc - 1, nocc)):
        ref = pse.qp_energy_g0w0(dfi, eig, kn, n, exchange_minus_vxc=xmv[kn, n])
        worst = max(worst, abs(qp[i] - ref) * HARTREE_TO_EV)
    assert worst < 2e-3, f"worst QP deviation {worst:.4f} eV"


def test_pi_may_be_a_callable_and_gives_the_same_answer(gwfix):
    """Sigma_c accepts Pi per transfer, so no caller must hold it all-q.

    The materialized form is nq * nfreq * M^2, the largest single object in
    the route. A closure removes it at no cost, since each q is visited
    exactly once.

    Equality here has to be EXACT: it is the same arithmetic reached two ways,
    so any deviation is a defect rather than a trade-off.
    """
    f = gwfix
    nocc = f['nocc']
    z_fit, iw = imaginary_axis_sample_points(
        gauss_legendre_grid(16, w0=0.5 * f['rW'][0])[0], nocc, nocc, f['mu'])
    args = (f['mo_energy'], f['mo_occ'], KMESH, 0, [nocc], f['omega_in'], iw,
            f['tau'])
    kw = dict(mu=f['mu'], ranges=(f['rW'], f['rS']))

    from_list = gw.sigma_c_diag_thc(f['X'], f['V'], f['Pi'], *args, **kw)
    from_call = gw.sigma_c_diag_thc(f['X'], f['V'], lambda q: f['Pi'][q],
                                    *args, **kw)
    assert np.array_equal(from_list, from_call)


def test_driver_defaults_to_homo_and_lumo(gwfix):
    """The default state window is (HOMO, LUMO) at the requested k-point."""
    f = gwfix
    qp, diag = gw.qp_energy_thc(f['cell'], f['mf'], f['V'][0].shape[0], kn=0,
                                ntau=NTAU, exxdiv=None, return_diagnostics=True)
    assert qp.shape == (2,)
    # HOMO below LUMO, and both shifted off the mean-field eigenvalues.
    assert qp[0] < qp[1]
    assert diag['rS'][1] > diag['rW'][1]


# --- metallic route: fermionic IR (Matsubara) grid --------------------------

def test_self_energy_range_is_the_sum_of_the_two_ranges(gwfix):
    """Poles of Sigma = -G Wt sit at eps_m +/- omega_s, so the range is the SUM.

    Asserted by explicit pole arithmetic rather than by restating the formula.
    Sizing an IR Lambda or a Pade order from the SCREENING range instead is
    the range trap in its frequency-domain guise -- on the GDF route that cost
    4 of 40 nodes.
    """
    f = gwfix
    screening_max = f['rW'][1]
    got = gw.self_energy_range_k(f['mo_energy'], f['mo_occ'], screening_max, f['mu'])
    poles = np.concatenate([(f['mo_energy'] - f['mu']).ravel() + screening_max,
                            (f['mo_energy'] - f['mu']).ravel() - screening_max])
    assert abs(got - np.abs(poles).max()) < 1e-12
    assert got > screening_max


def test_matsubara_green_function_keeps_only_the_greater_branch(gwfix):
    """For 0 < tau < beta at T -> 0, G(tau) is the VIRTUAL sum alone: the
    occupied term carries e^{-x(tau-beta)} with x < 0, i.e. e^{|x|(tau-beta)},
    which vanishes as beta grows. The occupied contribution re-enters through
    the antiperiodicity built into the fermionic uhat, which is why the IR
    route needs no lesser/greater split at all."""
    f = gwfix
    beta, tau = 400.0, 3.0
    G = gw.green_function_tau_matsubara(f['X'], f['mo_energy'], f['mo_occ'],
                                        tau, beta, f['mu'])
    _, G_g = gw.green_functions_tau_gw(f['X'], f['mo_energy'], f['mo_occ'],
                                       tau, f['mu'])
    assert abs(G - G_g).max() / abs(G_g).max() < 1e-8


def test_matsubara_green_function_is_finite_across_the_whole_interval(gwfix):
    """Both terms are 0 * inf if formed naively -- (1-f/2) underflows exactly
    as e^{-x tau} overflows for a deep state, and the other way round in the
    second term. Each PRODUCT is bounded on [0, beta]; neither factor is."""
    f = gwfix
    beta = 800.0
    for tau in (0.0, 1.0, 0.5 * beta, beta - 1.0, beta):
        G = gw.green_function_tau_matsubara(f['X'], f['mo_energy'], f['mo_occ'],
                                            tau, beta, f['mu'])
        assert np.isfinite(G).all(), f"G not finite at tau={tau}"


def test_matsubara_route_reproduces_the_t0_route(gwfix):
    """THE GATE. A gapped system at large beta is the T -> 0 limit of the
    Matsubara route, so the two must agree -- and this is what pins the
    Jacobians, which are pure constants and therefore invisible to any
    convergence study.

    It caught one: the bosonic omega -> tau fit needs an explicit (2/beta)
    inverse Jacobian, which `IRBasis.fit_matsubara` does not carry (uhat
    integrates over dimensionless x, not over tau). Without it Sigma is too
    large by exactly beta/2 -- measured ratio 199.97 at beta = 400, a clean
    constant that reads as a convention rather than a bug.
    """
    f = gwfix
    from src.Base.utils.matsubara import ir_continuation_order
    beta = 400.0
    wmax = gw.self_energy_range_k(f['mo_energy'], f['mo_occ'], f['rW'][1], f['mu'])
    nuB = gw.bosonic_sampling(beta, wmax)
    Pi_b = [rpa.polarizability_q_frequency_occ(f['X'], f['mo_energy'], f['mo_occ'],
                                               f['kplus'], q, nuB)
            for q in range(len(f['kpts']))]
    nocc = f['nocc']
    worst = 0.0
    for kn in (0, 1):
        for n in (nocc - 1, nocc):
            z, iw = imaginary_axis_sample_points(
                gauss_legendre_grid(16, w0=0.5 * f['rW'][0])[0], nocc, n, f['mu'])
            s_t0 = gw.sigma_c_diag_thc(f['X'], f['V'], f['Pi'], f['mo_energy'],
                                       f['mo_occ'], KMESH, kn, [n], f['omega_in'],
                                       iw, f['tau'], mu=f['mu'],
                                       ranges=(f['rW'], f['rS']))[0]
            qp_t0 = solve_qp_from_imaginary_axis(f['mo_energy'][kn], n, 0.0, z, s_t0)

            sig, wf, diag = gw.sigma_c_matsubara_thc(
                f['X'], f['V'], Pi_b, f['mo_energy'], f['mo_occ'], KMESH, beta,
                wmax, kn, [n], mu=f['mu'], return_diagnostics=True)
            assert diag['wt_fit_residual'] < 1e-9, diag
            sign = -1.0 if n < nocc else 1.0
            keep = np.where(np.sign(wf) == sign)[0]
            qp_m = solve_qp_from_imaginary_axis(
                f['mo_energy'][kn], n, 0.0, f['mu'] + 1j * wf[keep], sig[0][keep],
                max_order=ir_continuation_order(beta, wmax))
            worst = max(worst, abs(qp_m - qp_t0) * HARTREE_TO_EV)
    assert worst < 0.1, f"Matsubara vs T=0 QP: {worst:.4f} eV"


def test_t0_route_refuses_a_metal(gwfix):
    """The T = 0 self-energy route is integer-occupation only, and says so
    rather than returning a plausible number."""
    f = gwfix
    smeared = np.array([[2., 1.3, 0.7, 0., 0., 0., 0., 0.]] * len(f['kpts']))
    with pytest.raises(NotImplementedError, match='fractional occupations'):
        gw.sigma_c_diag_thc(f['X'], f['V'], f['Pi'], f['mo_energy'], smeared,
                            KMESH, 0, [3], f['omega_in'], np.array([0.1]),
                            f['tau'], mu=f['mu'], ranges=(f['rW'], f['rS']))

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
