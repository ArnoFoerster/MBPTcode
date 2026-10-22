"""The orbital-relaxed ADC(2) difference density (ee_relaxed_density) and the
state-specific term built from it.

The relaxed density of a folded root is d omega / d h: its contraction with
any one-electron operator O equals the finite-field derivative of omega when
O is added to the core Hamiltonian, the SCF re-converged and the root
re-solved. That is the gate; the rest pins the pieces it is made of.

  1. Phi = y^T A_eff(omega) y of the dense reference equals the folded
     operator's own product (ee_ri_fold.RIFoldADC2.apply) to 1e-12,
     ADC(1) and ADC(2), singlet and triplet, with the optical reaction field.
  2. every derivative of the dense reference against central differences of
     Phi: the orbital energies (the diagonal of dPhi/df), random elements of
     B^P_ia, B^P_ij and B^P_ab, to 1e-9 (water 6-31G, singlet and triplet).
  3. the tiled pieces equal the dense ones to 1e-12 relative (formaldehyde
     cc-pVDZ, frozen core, at the default tile and one occupied orbital per
     tile, singlet and triplet).
  4. gas phase, water 6-31G (DF): Tr(Delta rho_rel O) against the 5-point
     finite-field derivative of omega (h = 1e-3, SCF to a gradient of
     1e-12), O = mu_z and a random symmetric matrix, ADC(1) singlet and
     ADC(2) singlet and triplet with one frozen core orbital, to 1e-8.
  5. in toluene (PTE): the ground state's surface charges q0 frozen in the
     core Hamiltonian, the singlet carrying its optical LR term; the same
     gate with the relaxed density of the solvated SCF, to 1e-8.
  6. pcm_response: the same gate against a finite field in which the
     continuum responds (the PCM SCF re-converged with O to an orbital
     gradient of 1e-12; h = 2e-3, the responding continuum's higher orders
     against the SCF's noise), to 1e-8; and the
     frozen-charge density misses that finite field by more than 1e-6 (the
     two choices differ, and the check can tell them apart).
  7. solve_ee_adc(ss_density='relaxed'): e = omega + the relaxed shift; the
     unrelaxed shifts and both dipole changes are recorded; the relaxed SS
     shift is smaller in magnitude than the unrelaxed one (formaldehyde
     cc-pVDZ toluene, singlet and triplet); unrelaxed is the default (two
     solves: to a few ulp, `matches` of tests/pins.py); refusals
     (unknown density, relaxed without state_specific).
  8. over simulated MPI ranks (2 and 3, one occupied orbital per tile) the
     pieces and the relaxed density are the serial ones to a few ulp.
  9. a check that can fail: the pieces with the B^P_ab derivative (the
     doubles' and the exchange term's) left out miss the finite field of
     check 4 by more than 1e-5.

Run: python tests/test_ee_adc_relaxed_density.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.solvent_screening import SolventScreening
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
from src.SingleReference.ADC.eeADC import ee_relaxed_density as rd
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
from tests.adc_common import (FD_STEP_DENSITY, H2CO, five_point, JKFIT, SCF_GRAD_TIGHT,
                              SCF_MAX_CYCLE, WATER_C2V, check, converge_newton)
from tests.pins import RECORDING_BLAS, matches


def gas_scf(mol, h1=None, dm0=None):
    mf = scf.RHF(mol).density_fit(auxbasis=JKFIT)
    if h1 is not None:
        mf.get_hcore = lambda *a, **k: h1
    return converge_newton(mf, dm0)


def root(mol, mf, spin, level='adc2', frozen=0, env=None):
    """(operator, omega, y, t1, reaction field) of the lowest root."""
    f = ee_ri_fold.bare_factors(mol, mf, frozen=frozen)
    rf = (None if env is None else
          ee_ri_fold.TransitionReactionField.from_environment(env, mol, mf, frozen))
    op = ee_ri_fold.RIFoldADC2(f, spin, level=level, reaction_field=rf)
    if level == 'adc1':
        e, X = np.linalg.eigh(ee_fold.dense_effective(
            ee_ri_fold.FoldedMatvec(op, None), op.n))
        return op, e[0], X[:, 0], 1.0, rf
    r = ee_fold.solve_folded(op.pieces(), 1, spin=spin, dense_limit=0,
                             tol_omega=1e-13, tol_residual=1e-11)
    return op, r.omega[0], r.y[:, 0], r.t1[0], rf


def operators(mol):
    rng = np.random.default_rng(7)
    A = rng.standard_normal((mol.nao, mol.nao))
    return {'mu_z': mol.intor('int1e_r')[2], 'random': 0.05 * (A + A.T)}


def analytic(mol, mf, spin, level, frozen, env=None, pcm_response=False,
             drop_vv=False):
    """Delta rho_rel in the AO basis from the dense pieces; drop_vv leaves
    the B^P_ab derivative out (check 9)."""
    op, w, y, t1, rf = root(mol, mf, spin, level, frozen, env)
    pc = rd.dense_pieces(rd.dense_inputs(op), y, w, spin, level=level, rf=rf)
    if drop_vv:
        pc.Xvv = np.zeros_like(pc.Xvv)
    Bc = rd.core_factors(mol, mf, frozen) if frozen else None
    D = rd.relaxed_density(pc, op, mf, mol, frozen, t1, Bc=Bc, env=env,
                           pcm_response=pcm_response)
    C = np.asarray(mf.mo_coeff)
    return C @ D @ C.T


def check_phi():
    """Check 1."""
    ok = True
    mol = gto.M(atom=WATER_C2V, basis='6-31g', verbose=0)
    env = SolventScreening(mol, solvent='toluene')
    mf = gas_scf(mol)
    for level in ('adc1', 'adc2'):
        for spin in ('singlet', 'triplet'):
            op, w, y, _, rf = root(mol, mf, spin, level, env=env)
            phi = rd.dense_phi(rd.dense_inputs(op), y, w, spin, level, rf)
            Ay = op.apply(y[:, None], None if level == 'adc1' else np.array([w]))[0]
            d = abs(phi - y @ Ay[:, 0])
            ok &= check(d < 1e-12, f'{level} {spin}: dense Phi is y^T A_eff y',
                        f'{d:.1e}')
    return ok


def check_derivatives():
    """Check 2."""
    ok = True
    mol = gto.M(atom=WATER_C2V, basis='6-31g', verbose=0)
    mf = gas_scf(mol)
    rng = np.random.default_rng(1)
    h = 1e-5
    for spin in ('singlet', 'triplet'):
        op, w, y, _, _ = root(mol, mf, spin)
        inp = rd.dense_inputs(op)
        pc = rd.dense_pieces(inp, y, w, spin)
        no, nv = op.no, op.nv

        def fd(k, idx):
            def at(s):
                arrs = [a.copy() for a in inp]
                arrs[k][idx] += s
                return rd.dense_phi(tuple(arrs), y, w, spin)
            return (at(h) - at(-h)) / (2.0 * h)

        Gvv = np.einsum('ka,kbP->abP', pc.y, pc.Xvv.reshape(no, nv, -1))
        errs = {
            'e_occ': max(abs(fd(0, (i,)) - pc.g_oo[i, i]) for i in range(no)),
            'e_vir': max(abs(fd(1, (a,)) - pc.g_vv[a, a]) for a in range(nv)),
            'B_ia': max(abs(fd(2, (i, a, P)) - pc.GL.reshape(no, nv, -1)[i, a, P])
                        for i, a, P in zip(rng.integers(0, no, 8),
                                           rng.integers(0, nv, 8),
                                           rng.integers(0, op.f.naux, 8))),
            'B_ij': max(abs(fd(3, (i, j, P)) - pc.Goo[i, j, P])
                        for i, j, P in zip(rng.integers(0, no, 8),
                                           rng.integers(0, no, 8),
                                           rng.integers(0, op.f.naux, 8))),
            'B_ab': max(abs(fd(4, (a, b, P)) - Gvv[a, b, P])
                        for a, b, P in zip(rng.integers(0, nv, 8),
                                           rng.integers(0, nv, 8),
                                           rng.integers(0, op.f.naux, 8)))}
        for name, e in errs.items():
            ok &= check(e < 1e-9, f'{spin}: dPhi/d{name} against finite '
                        'differences', f'{e:.1e}')
    return ok


def check_tiled():
    """Check 3."""
    ok = True
    mol = gto.M(atom=H2CO, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    for spin in ('singlet', 'triplet'):
        op, w, y, _, _ = root(mol, mf, spin, frozen=2)
        d = rd.dense_pieces(rd.dense_inputs(op), y, w, spin)
        for tg in (rd.EE_RI_TILE_GB, 1e-9):
            t = rd.tiled_pieces(op, y, w, tile_gb=tg)
            err = max(np.abs(getattr(t, k) - getattr(d, k)).max()
                      / np.abs(getattr(d, k)).max()
                      for k in ('g_oo', 'g_vv', 'GL', 'Goo', 'Xvv'))
            ok &= check(err < 1e-12, f'{spin}: tiled pieces equal the dense '
                        f'ones (tile_gb {tg:g})', f'{err:.1e} relative')
    return ok


def check_gas_field():
    """Checks 4 and 9."""
    ok = True
    mol = gto.M(atom=WATER_C2V, basis='6-31g', verbose=0)
    mf0 = gas_scf(mol)
    dm0 = mf0.make_rdm1()
    h0 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    for level, spin, frozen in (('adc1', 'singlet', 0), ('adc2', 'singlet', 1),
                                ('adc2', 'triplet', 1)):
        P = analytic(mol, mf0, spin, level, frozen)
        for name, O in operators(mol).items():
            fd = five_point(lambda lam: root(mol, gas_scf(mol, h0 + lam * O, dm0),
                                             spin, level, frozen)[1])
            an = float(np.sum(P * O))
            ok &= check(abs(fd - an) < 1e-8, f'gas {level} {spin} frozen '
                        f'{frozen}: Tr(rho_rel {name}) is d omega / d field',
                        f'{an:+.10f} vs {fd:+.10f}, {fd - an:+.1e}')
            if level == 'adc2' and spin == 'singlet' and name == 'mu_z':
                bad = float(np.sum(analytic(mol, mf0, spin, level, frozen,
                                            drop_vv=True) * O))
                ok &= check(abs(fd - bad) > 1e-5, 'without the B_ab '
                            'derivative the gate fails', f'{fd - bad:+.1e}')
    return ok


def _toluene(mol, h1=None, dm0=None, conv_tol_grad=1e-9):
    env = SolventScreening(mol, solvent='toluene')

    def factory(m):
        mf = scf.RHF(m).density_fit(auxbasis=JKFIT)
        if h1 is not None:
            mf.get_hcore = lambda *a, **k: h1
        # The SCF's orbital gradient moves omega (5.6e-11 between two solves
        # at 1.5e-10 and 3.3e-10), and the five-point derivative divides that
        # by h and sums it with weights of 1.5: at conv_tol_grad 1e-9, where
        # pyscf's PCM SCF stopped on a cluster's MKL, it is the 1.5e-8 miss. The
        # finite-field solves pass SCF_GRAD_TIGHT, which pyscf's PCM SCF reaches
        # from a converged density (5e-13, workstation; two solves agree to
        # 1e-14) but not within pyscf's default 50 cycles from scratch.
        mf.conv_tol, mf.conv_tol_grad = 1e-13, conv_tol_grad
        mf.max_cycle = SCF_MAX_CYCLE
        mf.kernel(dm0=dm0)
        return mf

    return env, env.mean_field(mol, factory)


def check_continuum_field():
    """Checks 5 and 6."""
    ok = True
    mol = gto.M(atom=WATER_C2V, basis='6-31g', verbose=0)
    env, sol = _toluene(mol)
    dm = sol.make_rdm1()
    pcm = sol.with_solvent
    pcm._get_vind(dm)
    v0 = pcm._get_vmat(pcm._intermediates['q_sym'].copy())[0]
    h0 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    ops = operators(mol)
    for spin in ('singlet', 'triplet'):
        P = analytic(mol, sol, spin, 'adc2', 1, env=env)
        for name, O in ops.items():
            fd = five_point(lambda lam: root(
                mol, gas_scf(mol, h0 + v0 + lam * O, dm), spin, 'adc2', 1, env)[1])
            an = float(np.sum(P * O))
            ok &= check(abs(fd - an) < 1e-8, f'toluene, q0 frozen, {spin}: '
                        f'Tr(rho_rel {name}) is d omega / d field',
                        f'{an:+.10f} vs {fd:+.10f}, {fd - an:+.1e}')
    O = ops['mu_z']
    # the responding continuum's higher orders want a small step (h^4:
    # 7.9e-10 at 2e-3, 1.8e-8 at 4e-3, 2.9e-7 at 8e-3), the SCF's noise a
    # larger one (the displaced SCFs converge to SCF_GRAD_TIGHT)
    fd = five_point(lambda lam: root(
        mol, _toluene(mol, h0 + lam * O, dm, SCF_GRAD_TIGHT)[1],
        'singlet', 'adc2', 0, env)[1], step=2e-3)
    an = float(np.sum(analytic(mol, sol, 'singlet', 'adc2', 0, env=env,
                               pcm_response=True) * O))
    frozen_q = float(np.sum(analytic(mol, sol, 'singlet', 'adc2', 0, env=env) * O))
    ok &= check(abs(fd - an) < 1e-8, 'pcm_response: Tr(rho_rel mu_z) is the '
                'field derivative with the continuum responding',
                f'{an:+.10f} vs {fd:+.10f}, {fd - an:+.1e}')
    ok &= check(abs(fd - frozen_q) > 1e-6, 'the frozen-charge density misses '
                'that field derivative', f'{fd - frozen_q:+.1e}')
    return ok


def check_driver():
    """Check 7."""
    ok = True
    mol = gto.M(atom=H2CO, basis='cc-pvdz', verbose=0)
    env, sol = _toluene(mol)
    kw = dict(level='adc2', nroots=1, frozen=2, conv_tol=1e-9, fold=True,
              environment=env, return_fold=True, state_specific='vertical')
    for spin in ('singlet', 'triplet'):
        e, _, res = solve_ee_adc(sol, spin=spin, ss_density='relaxed', **kw)
        e_u, _, res_u = solve_ee_adc(sol, spin=spin, **kw)
        ok &= check(np.array_equal(e, res.omega + res.ss_shifts['vertical']),
                    f'{spin}: e = omega + the relaxed shift')
        # two solves of the state: its shift to a few ulp
        ok &= check(matches(res.ss_shifts_unrelaxed['vertical'],
                            res_u.ss_shifts['vertical'], RECORDING_BLAS),
                    f'{spin}: the unrelaxed shift is recorded alongside')
        sr, su = res.ss_shifts['vertical'][0], res.ss_shifts_unrelaxed['vertical'][0]
        mr, mu = res.delta_dipole['relaxed'][0], res.delta_dipole['unrelaxed'][0]
        ok &= check(su < sr < 0, f'{spin}: relaxation shrinks the SS shift',
                    f'relaxed {sr * 27211.386:+.1f} meV, unrelaxed '
                    f'{su * 27211.386:+.1f} meV; d mu_z relaxed {mr[2]:+.3f}, '
                    f'unrelaxed {mu[2]:+.3f} a.u.')
    e_d, _ = solve_ee_adc(sol, spin='singlet', **dict(kw, return_fold=False))
    e_x, _ = solve_ee_adc(sol, spin='singlet', ss_density='unrelaxed',
                          **dict(kw, return_fold=False))
    ok &= check(matches(e_d, e_x, RECORDING_BLAS, quantity='root'),
                'unrelaxed is the default (two solves: to a few ulp)')
    for label, args in (('an unknown density', dict(kw, ss_density='orbital')),
                        ('relaxed without state_specific',
                         dict(kw, state_specific='off', ss_density='relaxed'))):
        try:
            solve_ee_adc(sol, spin='singlet', **args)
            ok &= check(False, f'{label} raises')
        except ValueError as err:
            ok &= check(True, f'{label} raises', str(err)[:60])
    return ok


def check_ranks():
    """Check 8."""
    ok = True
    mol = gto.M(atom=WATER_C2V, basis='cc-pvdz', verbose=0)
    env, sol = _toluene(mol)
    for spin in ('singlet', 'triplet'):
        op, w, y, t1, _ = root(mol, sol, spin, frozen=1, env=env)
        Bc = rd.core_factors(mol, sol, 1)

        def run(comm, spin=spin):
            f = ee_ri_fold.bare_factors(mol, sol, frozen=1, comm=comm)
            rf = ee_ri_fold.TransitionReactionField.from_environment(env, mol, sol, 1)
            o = ee_ri_fold.RIFoldADC2(f, spin, reaction_field=rf, tile_gb=1e-9,
                                      threads=1, comm=comm)
            p = rd.tiled_pieces(o, y, w, tile_gb=1e-9)
            D = rd.relaxed_density(p, o, sol, mol, 1, t1, Bc=Bc, env=env)
            return [p.g_oo, p.g_vv, p.GL, p.Goo, p.Xvv, D]

        ser = run(None)
        for size in (2, 3):
            outs = run_simulated(run, size)
            # a rank split against the serial run: to a few ulp
            same = all(all(matches(a, b, RECORDING_BLAS, quantity='root')
                           for a, b in zip(o, ser)) for o in outs)
            ok &= check(same, f'{spin}, {size} simulated ranks: the serial '
                        'pieces and relaxed density to a few ulp')
    return ok


def run():
    """Every check; True when all passed."""
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        all_ok &= check_phi()
        all_ok &= check_derivatives()
        all_ok &= check_tiled()
        all_ok &= check_gas_field()
        all_ok &= check_continuum_field()
        all_ok &= check_driver()
        all_ok &= check_ranks()
    return all_ok


def test_ee_adc_relaxed_density():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
