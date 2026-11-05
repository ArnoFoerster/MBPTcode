"""Nuclear forces of SCS-MP2 and SCS-ADC(2) states on the density-fitted fold
(ee_gradient with spin_scaling; ee_relaxed_density's scaled pieces). The
standard factors (c_os, c_ss) = (6/5, 1/3) of ADC2_SPIN_SCALING['scs'], the
ground state SCS-MP2 with the same factors; the triplet is the Ms = 0
component, the fold's spin-adapted triplet. Water (a bent, non-symmetric
geometry so that every coordinate moves), one frozen core orbital,
correlation fit cc-pvdz-ri, JK fit cc-pvdz-jkfit, gas phase.

  1. the unscaled limit: (c_os, c_ss) = (1, 1) is the ADC(2) force bit for bit
     (S0, S1, T1; dense on 6-31G, tiled on cc-pVDZ), and its energy.
  2. the energies: the force's E_HF + E_SCS-MP2 + omega is the mean field's
     energy plus the scaled operator's SCS-MP2 energy (ee_ri_fold, which
     test_ee_adc_scs pins to pyscf's c_os E_os + c_ss E_ss) plus its folded
     root, to 1e-10 Ha; the state models (ADC2State on MP2State with
     spin_scaling='scs') call the same energy, and their force is the
     dense force.
  3. the dense SCS forces of S0, S1 and T1 against five-point central
     differences of that energy on every coordinate (water 6-31G, h = 1e-3
     Bohr, SCF to a gradient of 1e-12, roots to 1e-12), to 1e-8 Ha/Bohr.
  4. the tiled SCS forces (one occupied orbital per tile, auxiliary chunks of
     20) are the dense ones to 1e-10 Ha/Bohr (water cc-pVDZ, S1 and T1), and
     on 2 simulated ranks every rank has the serial force to a few ulp.
  5. the relaxed density of a scaled root (solve_ee_adc(spin_scaling='scs',
     relaxed_density=True)) and the SCS-MP2 relaxed density are the
     five-point finite-field derivatives of omega and of E_SCS-MP2 (mu_z,
     field steps 1e-3 au) to 1e-8.
  6. a ground state scaled otherwise than the operator is refused.

Run: python tests/test_scs_adc2_gradient.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.constants import ADC2_SPIN_SCALING
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients import df_correlation_force as dcf
from src.gradients.adc import ee_gradient as eg
from src.gradients.adc.ee_state_model import ADC2State
from src.Base.numerics.numerics import Numerics
from src.gradients.state_models import HFState, MP2State
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from tests.adc_common import (ee_numerics, ATOMS, AUX, FD_STEP_DENSITY, NF, SCF_GRAD_TILED,
                              SCF_TOL_TIGHT, check, df_rhf, five_point, make_scf_tight,
                              molecule)
from tests.pins import RECORDING_BLAS, matches

SCS = ADC2_SPIN_SCALING['scs']
FD_STEP = 1e-3                     # five-point nuclear derivative, Bohr
TILED = dict(tile_gb=1e-9, block_gb=1e-4, width=20)
AUX_CHUNK = 20
SPINS = ('singlet', 'triplet')


def scf_tight(mol, h1=None, dm0=None):
    mf = make_scf_tight(mol)
    if h1 is not None:
        mf.get_hcore = lambda *a, **k: h1
    mf.kernel(dm0)
    return mf


def fold_root(mol, mf, spin, scaling=SCS):
    """(operator, omega) of the lowest scaled root, independent of the force."""
    f = ee_ri_fold.bare_factors(mol, mf, frozen=NF, auxbasis=AUX)
    op = ee_ri_fold.RIFoldADC2(f, spin, level='adc2', scaling=scaling)
    r = ee_fold.solve_folded(op.pieces(), 3, spin=spin, dense_limit=0,
                             tol_omega=1e-12, tol_residual=1e-11)
    return op, float(r.omega[0])


def energies(mol, h1=None, dm0=None):
    """(E_HF + E_SCS-MP2 + omega_S1, ... + omega_T1, E_HF + E_SCS-MP2): the
    mean field, the scaled operator's SCS-MP2 energy and its folded roots."""
    mf = scf_tight(mol, h1, dm0)
    out = []
    for spin in SPINS:
        op, w = fold_root(mol, mf, spin)
        out.append(mf.e_tot + op.e_mp2 + w)
    return np.array(out + [mf.e_tot + op.e_mp2])


def check_unscaled_limit():
    """Check 1."""
    ok = True
    mol = molecule(ATOMS)
    mf = scf_tight(mol)
    for spin in (None,) + SPINS:
        kw = dict(frozen=NF, auxbasis=AUX, ground=spin is None)
        F0, i0 = eg.adc2_force(mf, mol, spin, **kw, numerics=Numerics())
        F1, i1 = eg.adc2_force(mf, mol, spin, spin_scaling=(1.0, 1.0), **kw, numerics=Numerics())
        same = (np.array_equal(F0, F1) and np.array_equal(i0['dm_corr'], i1['dm_corr'])
                and i0['energy'] == i1['energy'])
        ok &= check(same, f'dense {spin or "S0"}: (1, 1) is ADC(2) bit for bit',
                    f'max |dF| {float(np.max(np.abs(F0 - F1))):.0e}')
    mol = molecule(ATOMS, basis='cc-pvdz')
    mf = df_rhf(mol, SCF_TOL_TIGHT, SCF_GRAD_TILED)
    mf.kernel()
    for spin in SPINS:
        kw = dict(frozen=NF, auxbasis=AUX, aux_chunk=AUX_CHUNK, **TILED)
        F0, i0 = eg.adc2_force_tiled(mf, mol, spin, **kw, numerics=Numerics())
        F1, i1 = eg.adc2_force_tiled(mf, mol, spin, spin_scaling=(1.0, 1.0), **kw,
                                     numerics=Numerics())
        same = np.array_equal(F0, F1) and np.array_equal(i0['dm_corr'], i1['dm_corr'])
        ok &= check(same, f'tiled {spin}: (1, 1) is ADC(2) bit for bit',
                    f'max |dF| {float(np.max(np.abs(F0 - F1))):.0e}')
    return ok


def check_energies_and_state_models():
    """Check 2."""
    ok = True
    mol = molecule(ATOMS)
    mf = scf_tight(mol)
    ref = energies(mol)
    forces = {}
    for k, spin in enumerate(SPINS + (None,)):
        F, info = eg.adc2_force(mf, mol, spin, frozen=NF, auxbasis=AUX,
                                ground=spin is None, spin_scaling='scs', numerics=Numerics())
        forces[spin] = F
        d = abs(info['energy'] - ref[k])
        ok &= check(d < 1e-10 and info['asymmetry'] < 1e-8,
                    f'{spin or "S0"}: the force\'s energy is E_HF + E_SCS-MP2 + omega '
                    'of the scaled operator, the Lagrangian stationary',
                    f"{info['energy']:.12f} vs {ref[k]:.12f} Ha, |Y - Y^T| "
                    f"{info['asymmetry']:.0e}")
    for spin in SPINS:
        hf = HFState(mol, make_scf_tight)
        mp2 = MP2State(hf, NF, AUX, force_route='dense', spin_scaling='scs',
                       numerics=Numerics(adc={'conv_tol': 1e-12}))
        state = ADC2State(mp2, spin, 0, 3, follow='index')
        r = state(np.zeros((mol.nao, mol.nao)), 0.0)
        F, _ = state.force(np.zeros((mol.nao, mol.nao)), 0.0)
        k = SPINS.index(spin)
        dE, dF = abs(r.energy - ref[k]), float(np.max(np.abs(F - forces[spin])))
        ok &= check(dE < 1e-9 and dF < 1e-9,
                    f'{spin}: the SCS state model calls that energy and its force',
                    f'energy {dE:.0e} Ha, force {dF:.0e} Ha/Bohr; identity '
                    f"{state.identity()['spin_scaling']}")
    return ok


def check_finite_differences():
    """Check 3."""
    ok = True
    mol = molecule(ATOMS)
    mf = scf_tight(mol)
    dm0 = mf.make_rdm1()
    analytic = [eg.adc2_force(mf, mol, spin, frozen=NF, auxbasis=AUX,
                              ground=spin is None, spin_scaling='scs', numerics=Numerics())[0]
                for spin in SPINS + (None,)]
    fd = np.zeros((3, mol.natm, 3))
    for a in range(mol.natm):
        for x in range(3):
            def at(s):
                d = np.zeros((mol.natm, 3))
                d[a, x] = s
                return energies(molecule(ATOMS, d), dm0=dm0)
            fd[:, a, x] = (at(-2 * FD_STEP) - 8 * at(-FD_STEP) + 8 * at(FD_STEP)
                           - at(2 * FD_STEP)) / (12 * FD_STEP)
    for k, label in enumerate(('SCS-ADC(2) S1', 'SCS-ADC(2) T1', 'SCS-MP2 S0')):
        err = float(np.max(np.abs(analytic[k] - fd[k])))
        ok &= check(err < 1e-8, f'{label}: analytic force = finite differences',
                    f'max error {err:.1e} Ha/Bohr, max |F| '
                    f'{float(np.max(np.abs(fd[k]))):.2e}')
    return ok


def check_tiled():
    """Check 4."""
    ok = True
    mol = molecule(ATOMS, basis='cc-pvdz')
    mf = df_rhf(mol, SCF_TOL_TIGHT, SCF_GRAD_TILED)
    mf.kernel()
    for spin in SPINS:
        op, _, out = eg.solve_root(mol, mf, spin, 3, NF, AUX, tile_gb=1e-9,
                                   aux_chunk=AUX_CHUNK, scaling=SCS)
        rt = eg._pick(out, 0, None)[1]
        Fd, (_, _, gd, _, _, _) = eg.dense_force(mf, mol, NF, AUX, op, rt, scaling=SCS,
                                                 numerics=Numerics())
        Ft, (_, _, gt, _, _, _) = eg.tiled_force(mf, mol, op, NF, AUX, rt, **TILED,
                                                 numerics=Numerics())
        e, eD = float(np.max(np.abs(Fd - Ft))), float(np.max(np.abs(gd - gt)))
        ok &= check(e < 1e-10 and eD < 1e-10, f'SCS {spin}: tiled force = dense',
                    f'force {e:.1e} Ha/Bohr, relaxed density {eD:.1e}, max |F| '
                    f'{float(np.max(np.abs(Fd))):.2e}')

    def triplet(comm):
        F, info = eg.adc2_force_tiled(mf, mol, 'triplet', frozen=NF, auxbasis=AUX,
                                      comm=comm, aux_chunk=AUX_CHUNK,
                                      spin_scaling='scs', **TILED, numerics=Numerics())
        return [F, info['dm_corr'], np.array([info['omega']])]

    ser = triplet(None)
    outs = run_simulated(triplet, 2)
    same = [all(matches(a, b, RECORDING_BLAS, quantity=q)
                for a, b, q in zip(o, ser, ('gradient', 'vector', 'root')))
            for o in outs]
    ok &= check(all(same) and len(same) == 2, 'SCS T1, 2 simulated ranks: every rank '
                'has the serial force to a few ulp',
                'ranks ' + ''.join('=' if s else 'x' for s in same))
    return ok


def check_field_derivatives():
    """Check 5."""
    ok = True
    mol = molecule(ATOMS)
    mf = scf_tight(mol)
    dm0 = mf.make_rdm1()
    h0 = mf.get_hcore()
    mu = mol.intor('int1e_r')[2]
    C = np.asarray(mf.mo_coeff)
    for spin in SPINS:
        _, _, res = solve_ee_adc(mf, mol, level='adc2', spin=spin, nroots=3, fold=True,
                                 frozen=NF, auxbasis=AUX, numerics=ee_numerics(conv_tol=1e-12),
                                 return_fold=True, spin_scaling='scs',
                                 relaxed_density=True)
        an = float(np.sum((C @ res.delta_density_relaxed[0] @ C.T) * mu))
        fd = five_point(lambda lam: fold_root(mol, scf_tight(mol, h0 + lam * mu, dm0),
                                              spin)[1], FD_STEP_DENSITY)
        ok &= check(abs(an - fd) < 1e-8, f'SCS {spin}: Tr(rho_rel mu_z) is d omega / d field',
                    f'{an:+.10f} vs {fd:+.10f}, {fd - an:+.1e}')
    e2, gam = dcf.mp2_density(mf, mol, NF, AUX, scaling=SCS, numerics=Numerics())
    an = float(np.sum((C @ gam @ C.T) * mu))

    def e_scs(lam):
        m = scf_tight(mol, h0 + lam * mu, dm0)
        return fold_root(mol, m, 'singlet')[0].e_mp2
    fd = five_point(e_scs, FD_STEP_DENSITY)
    ok &= check(abs(an - fd) < 1e-8, 'SCS-MP2: Tr(D_rel mu_z) is d E / d field',
                f'{an:+.10f} vs {fd:+.10f}, {fd - an:+.1e}')
    return ok


def check_refusal():
    """Check 6."""
    mol = molecule(ATOMS)
    mf = scf_tight(mol)
    op, _, out = eg.solve_root(mol, mf, 'singlet', 3, NF, AUX, scaling=SCS)
    rt = eg._pick(out, 0, None)[1]
    try:
        eg.dense_force(mf, mol, NF, AUX, op, rt, numerics=Numerics())
        msg = ''
    except ValueError as err:
        msg = str(err)
    return check('same factors' in msg, 'an unscaled ground state under a scaled '
                 'operator is refused', msg[:70])


def run():
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for c in (check_unscaled_limit, check_energies_and_state_models,
                  check_finite_differences, check_tiled, check_field_derivatives,
                  check_refusal):
            all_ok &= c()
    print('ALL PASSED' if all_ok else 'FAILURES DETECTED')
    return all_ok


def test_scs_adc2_gradient():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
