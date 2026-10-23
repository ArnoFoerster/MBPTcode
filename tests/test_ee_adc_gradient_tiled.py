"""The tiled, distributed nuclear force of DF-MP2 and folded ADC(2) states
(src.gradients.df_correlation_force_tiled through ee_gradient) against
the dense reference (ee_gradient.adc2_force). Water and formaldehyde
cc-pVDZ (bent, non-symmetric geometries so that every coordinate moves),
frozen core, correlation fit cc-pvdz-ri, JK fit cc-pvdz-jkfit. Tiles of one
occupied orbital, AO shell blocks of a few functions and auxiliary chunks of
20 functions, so every sum runs over many tiles, chunks and blocks.

  1. tiled = dense to 1e-10 Ha/Bohr, the same root handed to both: MP2 (S0),
     the triplet (gas), and through the state models (force_route 'dense'
     vs 'tiled') the singlet with the LR term in a fixed toluene IEF-PCM
     field, its medium force and LR transition force included; the relaxed
     densities to 1e-10.
  2. 2 and 3 simulated ranks give every rank the serial force, relaxed
     density and transition density to a few ulp (MP2, the singlet with the LR
     term, the triplet with its root solved over the ranks).
  3. the tiled singlet force (gas) against central differences of E_HF +
     E_MP2 + omega on every coordinate (h = 1e-4 Bohr, oracle independent of
     the force: pyscf DF-MP2 and the fold's omega), to 1e-6 Ha/Bohr.
  4. the per-rank memory estimate is refused (MemoryError) below the peak,
     before any work, and passes above it.

Run: python tests/test_ee_adc_gradient_tiled.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.mp import dfmp2_native

from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients import df_correlation_force_tiled as tcf
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
from src.gradients.adc import ee_gradient as eg
from src.gradients.adc.ee_state_model import ADC2State
from src.gradients.state_models import HFState, MP2State
from tests.adc_common import (ATOMS, AUX, FD_STEP_FORCE, JKFIT, SCF_GRAD_TILED,
                              SCF_TOL_TIGHT, check, df_rhf, molecule)
from tests.pins import RECORDING_BLAS, matches

FORMALDEHYDE = [('C', (0.01, 0.0, -0.53)), ('O', (0.0, 0.02, 0.68)),
                ('H', (0.0, 0.93, -1.12)), ('H', (0.02, -0.95, -1.10))]
BASIS = 'cc-pvdz'
# one occupied orbital per tile, a few AO functions per block, 20 auxiliary
# functions per chunk of the fold and of the force
TILED = dict(tile_gb=1e-9, block_gb=1e-4, width=20)
AUX_CHUNK = 20


def make_scf(mol):
    return df_rhf(mol, SCF_TOL_TIGHT, SCF_GRAD_TILED)


def scf_tight(mol):
    mf = make_scf(mol)
    mf.kernel()
    return mf


def frozen_of(mol):
    return sum(1 for a in range(mol.natm) if mol.atom_charge(a) > 2)


def triplet_root(mol, mf, nf, comm=None):
    op, _, out = eg.solve_root(mol, mf, 'triplet', 3, nf, AUX, tile_gb=1e-9,
                               comm=comm, aux_chunk=AUX_CHUNK)
    return op, eg._pick(out, 0, None)[1]


def check_against_dense():
    ok = True
    for name, atoms in (('water', ATOMS), ('formaldehyde', FORMALDEHYDE)):
        mol = molecule(atoms, basis=BASIS)
        nf = frozen_of(mol)
        mf = scf_tight(mol)
        # MP2, gas
        op = tcf.ground_operator(mol, mf, nf, AUX, 1e-9, aux_chunk=AUX_CHUNK)
        Fd, (_, _, gd, _, _, _) = eg.dense_force(mf, mol, nf, AUX)
        Ft, (_, _, gt, _, at, _) = eg.tiled_force(mf, mol, op, nf, AUX, **TILED)
        e, eD = float(np.max(np.abs(Fd - Ft))), float(np.max(np.abs(gd - gt)))
        ok &= check(e < 1e-10 and eD < 1e-10, f'{name} MP2: tiled force = dense',
                    f'force {e:.1e} Ha/Bohr, relaxed density {eD:.1e}, |Y - Y^T| '
                    f'{at:.0e}, max |F| {float(np.max(np.abs(Fd))):.2e}')
        # the triplet, gas, the same root to both routes
        op, rt = triplet_root(mol, mf, nf)
        Fd, (_, _, gd, _, _, _) = eg.dense_force(mf, mol, nf, AUX, op, rt)
        Ft, (_, _, gt, _, _, _) = eg.tiled_force(mf, mol, op, nf, AUX, rt, **TILED)
        e, eD = float(np.max(np.abs(Fd - Ft))), float(np.max(np.abs(gd - gt)))
        ok &= check(e < 1e-10 and eD < 1e-10, f'{name} T1: tiled force = dense',
                    f'force {e:.1e} Ha/Bohr, relaxed density {eD:.1e}, max |F| '
                    f'{float(np.max(np.abs(Fd))):.2e}')
        # the singlet with the LR term in a fixed toluene field, state models
        env = SolventScreening(mol, solvent='toluene')
        field = ContinuumField.from_environment(env)
        out = {}
        for route in ('dense', 'tiled'):
            hf = HFState(mol, make_scf)
            mp2 = MP2State(hf, nf, AUX, force_route=route, tile_gb=1e-9)
            state = ADC2State(mp2, 'singlet', 0, 3, field.transition_environment(),
                              conv_tol=1e-12)
            h, c = field.operator(field.charges(field.potential(hf(0.0, 0.0).dm_eff)))
            F, info = state.force(h, c)
            F = F + field.force(info['dm_eff'])
            for w, rho in info['lr']:
                F = F + w * field.transition_force(rho)
            out[route] = (F, info)
        (Fd, idn), (Ft, itl) = out['dense'], out['tiled']
        e = float(np.max(np.abs(Fd - Ft)))
        eD = float(np.max(np.abs(idn['dm_eff'] - itl['dm_eff'])))
        eT = float(np.max(np.abs(idn['lr'][0][1] - itl['lr'][0][1])))
        ok &= check(e < 1e-10 and eD < 1e-10 and eT < 1e-12 and len(itl['lr']) == 1,
                    f'{name} S1 in toluene (LR term), state models: tiled = dense',
                    f'force {e:.1e} Ha/Bohr, dm_eff {eD:.1e}, transition density '
                    f'{eT:.0e}, max |F| {float(np.max(np.abs(Fd))):.2e}')
    return ok


def check_ranks():
    ok = True
    mol = molecule(ATOMS, basis=BASIS)
    nf = frozen_of(mol)
    mf = scf_tight(mol)
    env = SolventScreening(mol, solvent='toluene')
    tenv = ContinuumField.from_environment(env).transition_environment()
    rs = eg.solve_root(mol, mf, 'singlet', 3, nf, AUX, env=tenv, tile_gb=1e-9,
                       aux_chunk=AUX_CHUNK)[2]
    root_s = eg._pick(rs, 0, None)[1]

    def mp2(comm):
        op = tcf.ground_operator(mol, mf, nf, AUX, 1e-9, comm, AUX_CHUNK)
        F, (_, _, g, _, _, _) = eg.tiled_force(mf, mol, op, nf, AUX, **TILED)
        return [F, g]

    def singlet(comm):
        f = ee_ri_fold.bare_factors(mol, mf, frozen=nf, auxbasis=AUX, comm=comm,
                                    aux_chunk=AUX_CHUNK)
        rf = ee_ri_fold.TransitionReactionField.from_environment(tenv, mol, mf, nf)
        op = ee_ri_fold.RIFoldADC2(f, 'singlet', reaction_field=rf, tile_gb=1e-9,
                                   comm=comm, aux_chunk=AUX_CHUNK)
        F, (_, _, g, lr, _, _) = eg.tiled_force(mf, mol, op, nf, AUX, root_s, tenv,
                                                **TILED)
        return [F, g, lr[0][1]]

    def triplet(comm):
        F, info = eg.adc2_force_tiled(mf, mol, 'triplet', frozen=nf, auxbasis=AUX,
                                      comm=comm, aux_chunk=AUX_CHUNK, **TILED)
        return [F, info['dm_corr'], np.array([info['omega']])]

    for label, run, kinds in (
            ('MP2', mp2, ('gradient', 'vector')),
            ('S1 with the LR term', singlet, ('gradient', 'vector', 'vector')),
            ('T1, root solved over the ranks', triplet,
             ('gradient', 'vector', 'root'))):
        ser = run(None)
        for size in (2, 3):
            outs = run_simulated(run, size)
            same = [all(matches(a, b, RECORDING_BLAS, quantity=q)
                        for a, b, q in zip(o, ser, kinds)) for o in outs]
            ok &= check(all(same) and len(same) == size,
                        f'{label}, {size} simulated ranks: every rank has the serial '
                        'force and densities to a few ulp',
                        'ranks ' + ''.join('=' if s else 'x' for s in same))
    return ok


def energy_singlet(mol):
    """E_HF + E_MP2 + omega of the lowest singlet, independent of the force:
    pyscf's DF-MP2 and the fold's root."""
    mf = scf_tight(mol)
    nf = frozen_of(mol)
    mp = dfmp2_native.DFRMP2(mf, frozen=nf, auxbasis=AUX)
    mp.verbose = 0
    e = mf.e_tot + mp.kernel()
    mp.delete()
    f = ee_ri_fold.bare_factors(mol, mf, frozen=nf, auxbasis=AUX)
    op = ee_ri_fold.RIFoldADC2(f, 'singlet', level='adc2')
    r = ee_fold.solve_folded(op.pieces(), 3, spin='singlet', dense_limit=0,
                             tol_omega=1e-12, tol_residual=1e-11)
    return e + r.omega[0]


def check_finite_differences():
    mol = molecule(ATOMS, basis=BASIS)
    nf = frozen_of(mol)
    F, _ = eg.adc2_force_tiled(scf_tight(mol), mol, 'singlet', frozen=nf, auxbasis=AUX,
                               aux_chunk=AUX_CHUNK, **TILED)
    fd = np.zeros_like(F)
    for a in range(mol.natm):
        for x in range(3):
            d = np.zeros((mol.natm, 3))
            d[a, x] = FD_STEP_FORCE
            fd[a, x] = (energy_singlet(molecule(ATOMS, d, basis=BASIS))
                        - energy_singlet(molecule(ATOMS, -d, basis=BASIS))) / (2 * FD_STEP_FORCE)
    err = float(np.max(np.abs(F - fd)))
    return check(err < 1e-6, 'water S1: the tiled force = central differences',
                 f'max error {err:.1e} Ha/Bohr, max |F| {float(np.max(np.abs(fd))):.2e}')


def check_memory_refusal():
    ok = True
    mol = molecule(ATOMS, basis=BASIS)
    nf = frozen_of(mol)
    mf = scf_tight(mol)
    called = []
    real = tcf.ground_operator
    est = tcf.estimate(mol, nf, AUX, JKFIT, excited=False, block_gb=TILED['block_gb'])

    def spy(*a, **k):
        called.append(1)
        return real(*a, **k)

    tcf.ground_operator = spy
    try:
        try:
            eg.adc2_force_tiled(mf, mol, None, frozen=nf, auxbasis=AUX, ground=True,
                                max_memory=0.5 * est['peak_gb'] * 1024, **TILED)
            ok &= check(False, 'a max_memory below the estimate is refused')
        except MemoryError as err:
            ok &= check(not called, 'a max_memory below the estimate is refused '
                        'before any work', str(err)[:70])
        F, info = eg.adc2_force_tiled(mf, mol, None, frozen=nf, auxbasis=AUX,
                                      ground=True, max_memory=4000, **TILED)
        peak = info['estimate']['peak_gb']
        ok &= check(bool(called) and 0.0 < peak * 1024 < 4000,
                    'above the estimate it runs', f'estimated peak {peak * 1024:.1f} MB')
    finally:
        tcf.ground_operator = real
    return ok


def run():
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for c in (check_against_dense, check_ranks, check_finite_differences,
                  check_memory_refusal):
            all_ok &= c()
    print('ALL PASSED' if all_ok else 'FAILURES DETECTED')
    return all_ok


def test_ee_adc_gradient_tiled():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
