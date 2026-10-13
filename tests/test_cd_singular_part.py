"""The contour-deformation self-energy is continuous through every orbital energy.

Sigma_c has its poles at eps_q -/+ Omega_s, never at eps_q, but the split into
an imaginary-axis integral and residues has a step there: the integral jumps by
W^c_pq,qp(0) as omega crosses eps_q and the residue term jumps back. The
integral's jump comes from the Lorentzian (omega - eps_q) / [(omega - eps_q)^2
+ nu^2], whose half-width falls below any grid's smallest node; a plain
quadrature leaves a step of W^c(0) in Sigma with a slope of order +100 beside
it. `cd_integral_weights` takes the singular part in closed form.

WHAT EACH GATE IS FOR:

- the weights against a closed form. For wc(nu) = Omega^2 / (Omega^2 + nu^2)
  the integral of the Lorentzian times wc is (pi/2) sign(de) Omega /
  (Omega + |de|) exactly, at every width. The weights meet it from
  |de| = 1e-9 to 1; plain quadrature weights are off by order one below the
  smallest node, so the gate can fail. Inside the residue term's on-contour band the
  integral takes the midpoint, as the half-weight residue does.
- their derivative, which sets Z and the eps adjoint, against the same closed
  form. What remains is the remainder's nu^2 term at widths near the smallest
  node, bounded here; Omega runs from the gap's scale up, where W^c(i.nu)
  varies.
- on water/cc-pVDZ with explicit screening: Sigma(eps_q + s) - Sigma(eps_q - s)
  goes to zero like 2 s Sigma', and the closed-form slope equals a finite
  difference of the value beside the orbital energy.
- the reverse pass at a frequency 1e-5 Ha from an orbital energy equals a
  finite difference of the value along a random direction in eps.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, lib, scf

from src.Base.constants import CD_NFREQ
from src.Base.utils.grids import gap_scaled_w0, gauss_legendre_grid
from src.SingleReference.GW.contour_deformation import (cd_integral_weights,
                                                        residue_set, sigma_cd,
                                                        sigma_cd_slope,
                                                        wc_explicit)
from src.SingleReference.GW.real_screening import (ExplicitRealScreening,
                                                   ov_energies)
from src.gradients.contour_deformation_adjoint import sigma_cd_backward

#: Distances from the orbital energy, both sides, from just outside the
#: residue term's on-contour band (1e-10) to where nothing is singular.
WIDTHS = np.concatenate([-np.logspace(-9, 0, 37), np.logspace(-9, 0, 37)])
#: Screening scales of the closed-form model, from the gap's scale up.
OMEGAS = np.array([0.3, 1.0, 4.0])


def closed_form(de, omega):
    """int_0^inf de/(de^2 + nu^2) Omega^2/(Omega^2 + nu^2) dnu and d/d(de)."""
    value = 0.5 * np.pi * np.sign(de) * omega / (omega + np.abs(de))
    slope = -0.5 * np.pi * omega / (omega + np.abs(de)) ** 2
    return value, slope


@pytest.fixture(scope='module')
def grid():
    return gauss_legendre_grid(CD_NFREQ, w0=0.17)


def model_wc(nu):
    return OMEGAS[None, :] ** 2 / (OMEGAS[None, :] ** 2 + nu[:, None] ** 2)


def test_the_weights_integrate_the_lorentzian_at_every_width(grid):
    nu, wt = grid
    wc = model_wc(nu)
    for de in WIDTHS:
        c, _ = cd_integral_weights(np.full(len(OMEGAS), de), nu, wt)
        got = np.einsum('kq,kq->q', c, wc)
        want, _ = closed_form(de, OMEGAS)
        assert np.abs(got - want).max() < 1e-8, (de, got - want)


def test_on_the_contour_the_integral_takes_the_midpoint(grid):
    """|de| inside RESIDUE_ON_CONTOUR_TOL: the residue counts the pole with
    half weight, so the integral must carry neither side of the jump."""
    nu, wt = grid
    for de in (0.0, 3e-11, -3e-11):
        c, _ = cd_integral_weights(np.full(len(OMEGAS), de), nu, wt)
        assert np.abs(np.einsum('kq,kq->q', c, model_wc(nu))).max() < 1e-9


def test_the_plain_quadrature_fails_where_the_weights_do_not(grid):
    """The gate above can fail: without the closed-form part the same sum
    misses the jump once |de| is below the smallest node."""
    nu, wt = grid
    wc = model_wc(nu)
    de = 1e-8
    plain = (wt[:, None] * de / (de ** 2 + nu[:, None] ** 2) * wc).sum(axis=0)
    want, _ = closed_form(de, OMEGAS)
    assert np.abs(plain - want).min() > 1.0


def test_the_weights_derivative_is_the_closed_form_one(grid):
    nu, wt = grid
    wc = model_wc(nu)
    for de in WIDTHS:
        _, dc = cd_integral_weights(np.full(len(OMEGAS), de), nu, wt)
        got = np.einsum('kq,kq->q', dc, wc)
        _, want = closed_form(de, OMEGAS)
        # the remainder's nu^2 term, worst at |de| near the smallest node
        assert np.abs(got - want).max() < 3e-4, (de, got - want)


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = 1e-12
    mf.run()
    C, eps = mf.mo_coeff, mf.mo_energy
    nocc, nmo = mol.nelectron // 2, len(eps)
    naux = mf.with_df.get_naoaux()
    B = np.empty((naux, nmo, nmo))
    p0 = 0
    for blk in mf.with_df.loop():
        B[p0:p0 + blk.shape[0]] = lib.einsum('Pmn,mi,nj->Pij',
                                             lib.unpack_tril(blk), C, C)
        p0 += blk.shape[0]
    C_ov = B[:, :nocc, nocc:].reshape(naux, -1)
    nu, wt = gauss_legendre_grid(CD_NFREQ, w0=gap_scaled_w0(eps, nocc))
    return dict(B=B, eps=eps, nocc=nocc, C_ov=C_ov, nu=nu, wt=wt)


def water_sigma(w, p, sys_, eps=None, residues=None):
    eps = sys_['eps'] if eps is None else eps
    Bp = sys_['B'][:, p, :]
    wc = wc_explicit(Bp, sys_['C_ov'], ov_energies(eps, sys_['nocc']),
                     sys_['nu'])
    rs = ExplicitRealScreening(sys_['C_ov'], eps, sys_['nocc'])
    res = residue_set(eps, sys_['nocc'], w) if residues is None else residues
    return (sigma_cd(p, w, Bp, eps, sys_['nocc'], sys_['nu'], sys_['wt'],
                     residues=res, wc=wc, real_screening=rs),
            sigma_cd_slope(p, w, Bp, eps, sys_['nocc'], sys_['nu'],
                           sys_['wt'], res, wc, real_screening=rs))


@pytest.mark.parametrize('p, q', [(4, 3), (4, 5), (5, 6)])
def test_sigma_is_continuous_through_an_orbital_energy(water, p, q):
    """HOMO across HOMO-1 and across the LUMO, LUMO across LUMO+1, where a
    plain quadrature steps by 7e-3, 3.5e-3 and 1e-2 Ha."""
    eq = water['eps'][q]
    _, slope = water_sigma(eq + 1e-6, p, water)
    for s in (1e-5, 1e-7, 1e-9):
        up, _ = water_sigma(eq + s, p, water)
        down, _ = water_sigma(eq - s, p, water)
        assert abs(up - down - 2 * s * slope) < 1e-10, (s, up - down)


@pytest.mark.parametrize('s', [1e-5, -1e-5, 1e-3])
def test_the_slope_is_the_derivative_beside_an_orbital_energy(water, s):
    """A plain quadrature's slope there is +99, which drives Z negative."""
    p, q = 4, 3
    w = water['eps'][q] + s
    _, slope = water_sigma(w, p, water)
    h = 1e-7
    fd = (water_sigma(w + h, p, water)[0] - water_sigma(w - h, p, water)[0]) / (2 * h)
    assert abs(slope - fd) < 1e-7, (slope, fd)
    assert abs(slope) < 0.2


def test_the_reverse_pass_is_the_derivative_beside_an_orbital_energy(water):
    """eps_bar and omega_bar of `sigma_cd_backward`, 1e-5 Ha from eps_(HOMO-1),
    against central differences with the residue set frozen."""
    p, q = 4, 3
    eps0 = water['eps']
    w = eps0[q] + 1e-5
    res = residue_set(eps0, water['nocc'], w)
    Bp = water['B'][:, p, :]
    eps_bar, _, _, omega_bar, _ = sigma_cd_backward(
        p, w, Bp, eps0, water['nocc'], water['nu'], water['wt'], res,
        C_ov=water['C_ov'])
    direction = np.random.default_rng(3).standard_normal(len(eps0))
    h = 1e-7
    fd = (water_sigma(w, p, water, eps0 + h * direction, res)[0]
          - water_sigma(w, p, water, eps0 - h * direction, res)[0]) / (2 * h)
    assert abs(eps_bar @ direction - fd) < 1e-7, (eps_bar @ direction, fd)
    fd_w = (water_sigma(w + h, p, water, residues=res)[0]
            - water_sigma(w - h, p, water, residues=res)[0]) / (2 * h)
    assert abs(omega_bar - fd_w) < 1e-7, (omega_bar, fd_w)
