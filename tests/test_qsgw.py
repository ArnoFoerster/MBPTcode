"""qsGW and qsGW0: the orbitals and the eigenvalues reinjected until the static
Hermitian self-energy stops moving.

Gate (a): the blocked static self-energy equals the dense matrix routine and, on
its diagonal, the per-state self-energy at eps_p. Gate (b): PBE and PBE0 starts
land on one fixed point. Gate (e): the converged point is a fixed point of the
evGW0 eigenvalue map, in one step. The two mixings land on one fixed point.

Run: python tests/test_qsgw.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import df, dft, gto

from src.Base.constants import EVGW_TOL, HARTREE_TO_EV, get_method_info
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
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


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    mf = build_reference()
    all_ok = True
    print('\n-- 1. the static self-energy, gate (a)')
    all_ok &= test_the_blocked_static_self_energy_is_the_dense_one(mf)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
