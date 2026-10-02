"""Vibronic absorption and emission bands of formaldehyde S1, and their widths.

The chain a vibronic band needs, end to end on one cheap
surface (BSE@G0W0@HF, cc-pVDZ, density fitted):

    S0 minimum and its normal modes      mean field, analytic Hessian
    S1 minimum                           ExcitedStateChain, Cartesian optimizer
    Huang-Rhys factors, both routes      vibronic_analysis
    0-0 energy                           E_S1(R_S1) - E_S0(R_S0)
    band shapes, FWHM, Stokes shift      band_shape

The displaced-oscillator band's Stokes shift is 2 lambda by construction; the
four-point vertical difference E_abs(R_S0) - E_em(R_S1) carries the two
surfaces' own curvatures and is printed beside it as the check on that model.
Formaldehyde's S1 <- S0 is n -> pi*, electric-dipole forbidden in C2v, so the
Condon band printed here is the Franck-Condon envelope only; the measured
spectrum borrows its intensity through out-of-plane modes (Herzberg-Teller).

    python examples/20_vibronic_band_shape.py
"""
import os
import sys

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.constants import HARTREE_TO_CM, HARTREE_TO_EV
from src.gradients.excited_state import ExcitedStateChain
from src.properties.band_shape import (band_grid, band_shape,
                                       gaussian_limit_fwhm)
from src.properties.optimize import MeanFieldSurface
from src.properties.vibronic import relax_state, vibronic_analysis

BASIS = 'cc-pvdz'
TEMPERATURE = 300.0
# the stated broadening: an inhomogeneous Gaussian, no homogeneous width
GAUSSIAN_FWHM = 0.05 / HARTREE_TO_EV
LORENTZIAN_FWHM = 0.0
CH2O = ('C 0.0000 0.0000 -0.5290; O 0.0000 0.0000 0.6746; '
        'H 0.0000 0.9376 -1.1188; H 0.0000 -0.9376 -1.1188')


def scf_factory(mol):
    """Converged for gradient work: the Lagrangian assumes F_ov = 0."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


mol0 = gto.M(atom=CH2O, basis=BASIS, verbose=0)
s0 = relax_state(MeanFieldSurface(mol0, scf_factory), mol0, engine='cartesian',
                 verbose=False)
mol_gs, mf_gs = s0['mol'], s0['mf']
chain = ExcitedStateChain(mol_gs, scf_factory, spin='singlet', mf=mf_gs)
s1 = relax_state(chain, mol_gs, engine='cartesian', verbose=False)
print(f'S0 relaxed: {s0["info"].get("status")}, S1 relaxed: '
      f'{s1["info"].get("status")}')
vib = vibronic_analysis(chain, s1, mol_gs, mf_gs)

e00 = s1['e_total'] - s0['e_total']
e_abs_vertical = chain.total_energy(mol_gs, mf_gs) - s0['e_total']
e_em_vertical = s1['e_total'] - scf_factory(s1['mol']).e_tot
print(f'\nE00 {e00 * HARTREE_TO_EV:.4f} eV, vertical absorption '
      f'{e_abs_vertical * HARTREE_TO_EV:.4f} eV, vertical emission '
      f'{e_em_vertical * HARTREE_TO_EV:.4f} eV, four-point Stokes shift '
      f'{(e_abs_vertical - e_em_vertical) * HARTREE_TO_EV:.4f} eV')
print(f'broadening: Gaussian FWHM {GAUSSIAN_FWHM * HARTREE_TO_EV:.3f} eV, '
      f'Lorentzian FWHM {LORENTZIAN_FWHM * HARTREE_TO_EV:.3f} eV, '
      f'T = {TEMPERATURE:.0f} K')

omega = vib['omega_cm'] / HARTREE_TO_CM
real = omega > 0
for route in ('gradient', 'displacement'):
    s_k = vib['s_' + route][real]
    w_k = omega[real]
    grid = band_grid(s_k, w_k, e00, TEMPERATURE, gaussian_fwhm=GAUSSIAN_FWHM,
                     lorentzian_fwhm=LORENTZIAN_FWHM)
    band = band_shape(s_k, w_k, e00, TEMPERATURE, grid,
                      gaussian_fwhm=GAUSSIAN_FWHM,
                      lorentzian_fwhm=LORENTZIAN_FWHM)
    g_lim = gaussian_limit_fwhm(s_k, w_k, TEMPERATURE)
    print(f'\nHuang-Rhys by the {route} route: sum S = {s_k.sum():.3f}, '
          f'lambda = {band["reorganization"] * HARTREE_TO_EV:.4f} eV')
    print(f'  absorption peak {band["peak_absorption"] * HARTREE_TO_EV:.4f} eV, '
          f'FWHM {band["fwhm_absorption"] * HARTREE_TO_EV:.4f} eV')
    print(f'  emission   peak {band["peak_emission"] * HARTREE_TO_EV:.4f} eV '
          f'({band["peak_emission_nm"]:.1f} nm), FWHM '
          f'{band["fwhm_emission"] * HARTREE_TO_EV:.4f} eV = '
          f'{band["fwhm_emission"] * HARTREE_TO_CM:.0f} cm^-1 = '
          f'{band["fwhm_emission_nm"]:.1f} nm')
    print(f'  Stokes shift (band peaks) {band["stokes_shift"] * HARTREE_TO_EV:.4f}'
          f' eV, 2 lambda {2.0 * band["reorganization"] * HARTREE_TO_EV:.4f} eV')
    print(f'  Gaussian-limit FWHM (no broadening) {g_lim * HARTREE_TO_EV:.4f} eV')
    top = np.argsort(-s_k)[:4]
    print('  largest S_k: ' + ', '.join(f'{w_k[i] * HARTREE_TO_CM:.0f} cm^-1 '
                                        f'S = {s_k[i]:.3f}' for i in top))
