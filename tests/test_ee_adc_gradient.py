"""Nuclear forces of DF-MP2 and folded ADC(2) states (ee_gradient, the dense
reference). Water 6-31G (a bent, non-symmetric geometry so that every
coordinate moves), one frozen core orbital, correlation fit cc-pvdz-ri, JK fit
cc-pvdz-jkfit, gas phase.

  1. the RI factors ee_gradient whitens are the operator's (ee_ri_fold) to
     1e-12: the derivative of Gamma assumes that basis; its MP2 energy is
     pyscf DF-MP2's to 1e-12.
  2. the Lagrangian is stationary: Y = Y^T to 1e-8 (MP2, ADC(2) singlet and
     triplet).
  3. the relaxed MP2 density of the engine is pyscf DF-MP2's (refined CPHF,
     state_models.mp2_relaxed) to 1e-8, and the ADC(2) part is
     ee_relaxed_density's to 1e-8.
  4. the force of E_HF + E_MP2 and of E_HF + E_MP2 + omega (singlet and
     triplet, lowest root) against central differences of the energy on every
     coordinate (h = 1e-4 Bohr, SCF to a gradient of 1e-12, roots to 1e-12),
     to 1e-6 Ha/Bohr.

Run: python tests/test_ee_adc_gradient.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.mp import dfmp2_native

from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
from src.SingleReference.ADC.eeADC import ee_gradient as eg
from src.SingleReference.ADC.eeADC import ee_relaxed_density as erd
from src.SingleReference import df_correlation_force as dcf
from src.SingleReference.state_models import mp2_relaxed
from tests.adc_common import (ATOMS, AUX, FD_STEP_FORCE, NF, check,
                              make_scf_tight, molecule)


def scf_tight(mol):
    mf = make_scf_tight(mol)
    mf.kernel()
    return mf


def energies(mol):
    """(E_HF + E_MP2, E_HF + E_MP2 + omega_singlet, ... + omega_triplet), the
    MP2 energy from pyscf's DF-MP2 (independent of ee_gradient)."""
    mf = scf_tight(mol)
    mp = dfmp2_native.DFRMP2(mf, frozen=NF, auxbasis=AUX)
    mp.verbose = 0
    out = [mf.e_tot + mp.kernel()]
    mp.delete()
    f = ee_ri_fold.bare_factors(mol, mf, frozen=NF, auxbasis=AUX)
    for spin in ('singlet', 'triplet'):
        op = ee_ri_fold.RIFoldADC2(f, spin, level='adc2')
        r = ee_fold.solve_folded(op.pieces(), 3, spin=spin, dense_limit=0,
                                 tol_omega=1e-12, tol_residual=1e-11)
        out.append(out[0] + r.omega[0])
    return np.array(out)


def check_factors():
    mol = molecule(ATOMS)
    mf = scf_tight(mol)
    ri = dcf.RIFactors(mol, mf, AUX)
    f = ee_ri_fold.bare_factors(mol, mf, frozen=NF, auxbasis=AUX)
    nocc = mol.nelectron // 2
    L = f.LT.transpose(1, 2, 0)
    d = float(np.max(np.abs(L - ri.B[NF:nocc, nocc:])))
    ok = check(d < 1e-12, "the engine's RI factors are the operator's",
               f'max |dB| {d:.1e}')
    e2 = dcf.mp2_pieces(ri.B, mf.mo_energy, NF, nocc)[0]
    mp = dfmp2_native.DFRMP2(mf, frozen=NF, auxbasis=AUX)
    mp.verbose = 0
    ref = mp.kernel()
    mp.delete()
    ok &= check(abs(e2 - ref) < 1e-12, "the engine's MP2 energy is pyscf's",
                f'{e2:.12f} vs {ref:.12f}')
    return ok


def check_densities_and_symmetry():
    ok = True
    mol = molecule(ATOMS)
    mf = scf_tight(mol)
    C = mf.mo_coeff
    g0, info0 = eg.adc2_force(mf, mol, None, frozen=NF, auxbasis=AUX, ground=True)
    ok &= check(info0['asymmetry'] < 1e-8, 'MP2: the Lagrangian is stationary',
                f"max |Y - Y^T| {info0['asymmetry']:.1e}")
    _, D2 = mp2_relaxed(mf, NF, AUX)
    d = float(np.max(np.abs(info0['dm_corr'] - D2)))
    ok &= check(d < 1e-8, 'MP2: relaxed density = pyscf DF-MP2 (refined CPHF)',
                f'{d:.1e}')
    for spin in ('singlet', 'triplet'):
        g1, info = eg.adc2_force(mf, mol, spin, frozen=NF, auxbasis=AUX)
        ok &= check(info['asymmetry'] < 1e-8, f'{spin}: the Lagrangian is '
                    'stationary', f"max |Y - Y^T| {info['asymmetry']:.1e}")
        f = ee_ri_fold.bare_factors(mol, mf, frozen=NF, auxbasis=AUX)
        op = ee_ri_fold.RIFoldADC2(f, spin, level='adc2')
        r = ee_fold.solve_folded(op.pieces(), 3, spin=spin, dense_limit=0,
                                 tol_omega=1e-12, tol_residual=1e-11)
        pc = erd.dense_pieces(erd.dense_inputs(op), r.y[:, 0], r.omega[0], spin)
        Bc = erd.core_factors(mol, mf, NF, AUX)
        Drd = C @ erd.relaxed_density(pc, op, mf, mol, NF, r.t1[0], Bc=Bc) @ C.T
        d = float(np.max(np.abs((info['dm_corr'] - info0['dm_corr']) - Drd)))
        ok &= check(d < 1e-8, f"{spin}: the engine's omega density = "
                    "ee_relaxed_density's", f'{d:.1e}')
    return ok


def check_forces():
    ok = True
    mol = molecule(ATOMS)
    mf = scf_tight(mol)
    analytic = [eg.adc2_force(mf, mol, None, frozen=NF, auxbasis=AUX, ground=True)[0]]
    for spin in ('singlet', 'triplet'):
        analytic.append(eg.adc2_force(mf, mol, spin, frozen=NF, auxbasis=AUX)[0])
    fd = np.zeros((3, mol.natm, 3))
    for a in range(mol.natm):
        for x in range(3):
            d = np.zeros((mol.natm, 3))
            d[a, x] = FD_STEP_FORCE
            fd[:, a, x] = ((energies(molecule(ATOMS, d)) - energies(molecule(ATOMS, -d)))
                           / (2 * FD_STEP_FORCE))
    for k, label in enumerate(('E_HF + E_MP2', 'ADC(2) singlet', 'ADC(2) triplet')):
        err = float(np.max(np.abs(analytic[k] - fd[k])))
        ok &= check(err < 1e-6, f'{label}: analytic force = finite differences',
                    f'max error {err:.1e} Ha/Bohr, max |F| '
                    f'{float(np.max(np.abs(fd[k]))):.2e}')
    return ok


def run():
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for c in (check_factors, check_densities_and_symmetry, check_forces):
            all_ok &= c()
    print('ALL PASSED' if all_ok else 'FAILURES DETECTED')
    return all_ok


def test_ee_adc_gradient():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
