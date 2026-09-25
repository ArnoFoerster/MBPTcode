"""The auxiliary-boson expansion must BE the exact boson set when complete.

That is the sharp gate, and it needs no physics: build three-index factors of
full rank, so the auxiliary basis spans the whole particle-hole space, and the
compressed dRPA must then reproduce the full Casida one to machine precision.
Anything else is an error in the projection, not an approximation.

The second gate is the one that distinguishes this route from a fit. Truncating
the EXACT bosons to the M lowest is hopeless -- measured on water, 48 of 95
still leave the self-energy 37 meV out -- because the coupling weight is spread
across the spectrum. The AB basis is not a subset of those bosons but a smaller
space the whole dRPA is re-solved in, and its truncation converges smoothly.
"""
import numpy as np
import pytest

from src.SingleReference.GW.auxiliary_bosons import (ab_basis, ab_bosons,
                                                     ab_couplings,
                                                     ab_from_factors,
                                                     exact_bosons,
                                                     exact_from_factors)
from src.SingleReference.GW.sum_over_poles import sigma_sop

EPS = np.array([-0.90, -0.62, -0.35, 0.18, 0.44, 0.83, 1.25, 1.70])
NOCC = 3
NVIRT = len(EPS) - NOCC
N_OV = NOCC * NVIRT


def factors(naux, seed=5, scale=0.05):
    """C_ov of shape (naux, n_ov); full row rank when naux <= n_ov."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((naux, N_OV)) * scale


def ph_energies():
    return (EPS[NOCC:][None, :] - EPS[:NOCC][:, None]).ravel()


def test_the_basis_is_orthonormal():
    c_ab = ab_basis(factors(N_OV + 4))
    assert np.abs(c_ab.T @ c_ab - np.eye(c_ab.shape[1])).max() < 1e-10


def test_a_complete_basis_reproduces_the_full_casida_solve():
    """naux >= n_ov and full rank: the AB space IS the particle-hole space."""
    c_ov, d = factors(N_OV + 6), ph_energies()
    c_ab = ab_basis(c_ov)
    assert c_ab.shape[1] == N_OV, f'basis is not complete: {c_ab.shape[1]}'
    om_ab, _ = ab_bosons(c_ov, d, c_ab)
    om_ex, _ = exact_bosons(c_ov, d)
    assert np.abs(np.sort(om_ab) - np.sort(om_ex)).max() < 1e-10
    bp = factors(N_OV + 6, seed=9)[:, :len(EPS)]
    omega = -0.5
    s_ab = sigma_sop(omega, *reversed(ab_from_factors(bp, c_ov, d, c_ab=c_ab)),
                     eps=EPS, nocc=NOCC, check=False)
    s_ex = sigma_sop(omega, *reversed(exact_from_factors(bp, c_ov, d)),
                     eps=EPS, nocc=NOCC, check=False)
    assert abs(s_ab - s_ex) < 1e-10, f'{s_ab} vs {s_ex}'


def test_the_gate_can_fail():
    """A basis one direction short must NOT reproduce the full solve."""
    c_ov, d = factors(N_OV + 6), ph_energies()
    short = ab_basis(c_ov, n_bosons=N_OV - 1)
    om_short, _ = ab_bosons(c_ov, d, short)
    om_ex, _ = exact_bosons(c_ov, d)
    assert om_short.size == N_OV - 1
    assert np.abs(np.sort(om_short) - np.sort(om_ex)[:N_OV - 1]).max() > 1e-8


def test_truncation_converges_rather_than_scattering():
    """The distinguishing property: more bosons is monotonically better.

    Keeping the M lowest EXACT bosons does not have it -- benzene goes 11 meV
    out at M = 8 and 103 meV at M = 48 -- which is why a compression has to
    re-solve in a smaller space rather than select from the big one.
    """
    c_ov, d = factors(N_OV + 6), ph_energies()
    bp = factors(N_OV + 6, seed=9)[:, :len(EPS)]
    omega = -0.5
    exact = sigma_sop(omega, *reversed(exact_from_factors(bp, c_ov, d)),
                      eps=EPS, nocc=NOCC, check=False)
    errs = []
    for n in (3, 6, 9, 15):
        poles, amp = ab_from_factors(bp, c_ov, d, n_bosons=n)
        assert poles.size == n
        errs.append(abs(sigma_sop(omega, amp, poles, EPS, NOCC, check=False)
                        - exact))
    assert all(b <= a * 1.5 for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < errs[0] / 10.0, errs


@pytest.mark.parametrize('n_bosons', [6, 12])
def test_amplitudes_are_squared_couplings_in_the_sop_layout(n_bosons):
    """`sigma_sop` must serve this route without knowing which one it is."""
    c_ov, d = factors(N_OV + 6), ph_energies()
    bp = factors(N_OV + 6, seed=9)[:, :len(EPS)]
    c_ab = ab_basis(c_ov, n_bosons=n_bosons)
    om, xy = ab_bosons(c_ov, d, c_ab)
    w = ab_couplings(bp, c_ov, c_ab, xy)
    poles, amp = ab_from_factors(bp, c_ov, d, c_ab=c_ab)
    assert amp.shape == (n_bosons, len(EPS))
    assert np.abs(amp - (w ** 2).T).max() < 1e-12
    assert np.all(amp >= 0.0), 'squared couplings cannot be negative'


def test_a_core_state_is_reached_where_a_fit_is_not():
    """The one real advantage over the fitted route, on the real thing."""
    pyscf = pytest.importorskip('pyscf')
    from pyscf import gto, scf
    from src.Base.constants import HARTREE_TO_EV
    from src.Base.pyscf_interface import get_density_fitting_coefficients
    from src.Base.utils.grids import gauss_legendre_grid, gap_scaled_w0
    from src.SingleReference.base import get_occ_virt_indices
    from src.SingleReference.GW.contour_deformation import (qp_energy_cd,
                                                            wc_explicit)
    from src.SingleReference.GW.real_screening import ov_energies
    from src.SingleReference.GW.sum_over_poles import (compressible,
                                                       qp_energy_sop,
                                                       sop_from_wc)
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-10)
    eps = np.asarray(mf.mo_energy, float)
    nocc = mol.nelectron // 2
    b = get_density_fitting_coefficients(mol, mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    c_ov = b[:, occ, :][:, :, virt].reshape(b.shape[0], -1)
    d = ov_energies(eps, nocc)
    nu, wt = gauss_legendre_grid(48, gap_scaled_w0(eps, nocc))
    p = 0                                              # the oxygen 1s
    bp = b[:, p, :]
    wc = wc_explicit(bp, c_ov, d, nu)
    w_cd, _, _ = qp_energy_cd(p, bp, eps, nocc, nu, wt, wc=wc, C_ov=c_ov,
                              relax_offset=False)
    ok, reach = compressible(w_cd, eps, nocc)
    assert not ok and reach > 10.0, f'the 1s should be far outside: {reach}'
    poles, amp = ab_from_factors(bp, c_ov, d)
    w_ab, _ = qp_energy_sop(p, amp, poles, eps, nocc, w0=w_cd)
    assert abs(w_ab - w_cd) * HARTREE_TO_EV * 1e3 < 5.0, (
        f'AB on the core state: {(w_ab - w_cd) * HARTREE_TO_EV * 1e3:.1f} meV')
    sop_poles, sop_amp = sop_from_wc(wc, nu, eps, nocc, n_poles=24)
    w_sop, _ = qp_energy_sop(p, sop_amp, sop_poles, eps, nocc, w0=w_cd)
    assert abs(w_sop - w_cd) * HARTREE_TO_EV * 1e3 > 100.0, (
        'the fitted route is supposed to FAIL here; if it stopped failing, the '
        'wall has moved and the guard needs re-measuring')
