"""ISR-ADC(2) difference densities and state-specific continuum solvation.

ee_density writes the unrelaxed ISR-ADC(2) difference density once against
the tensor backends; ee_ri_density evaluates it on the density-fitted tiles
of the folded route; ee_state_specific turns it into the reaction-field
energy (1/2) dV Q dV added to a root. Checks:

  1. the dense reference (the spin-free route's full eigenvector, spin
     blocks) equals pyscf's ISR-ADC(2) excited-state density, minus the
     reference and T1 times the MP2 ground-state correlation density, for
     water/6-31G singlets (radc_ee) and Ms = 0 triplets (uadc_ee on the same
     reference) to 1e-6; and leaving out any ONE of its 20 terms misses that
     density by more than 1e-5 (each term matters).
  2. the folded RI route equals the dense reference term by term to 1e-8
     (water and formaldehyde, cc-pVDZ, the same DF factors, singlets and
     triplets, two roots each), at the default tile and one occupied orbital
     per tile; the density is symmetric and traceless to 1e-12.
  3. the SS energy: (1/2) dV Q dV equals pyscf's own PCM energy functional
     (solvent.pcm _get_vind) as the second difference
     [E(P0 + dP) + E(P0 - dP) - 2 E(P0)] / 2 at eps_inf and at eps_static,
     to 1e-10 Ha; in the CIS limit (fold at adc1, toluene) the shift equals
     that functional on the difference density of pyscf's own TDA-PCM
     vector, to 1e-8 Ha.
  4. solve_ee_adc(fold=True, state_specific=...) in toluene, formaldehyde
     aug-cc-pVDZ: e = omega + shift; the shifts are negative, the
     equilibrium one the larger; a triplet's SS shift is non-zero while its
     LR term is zero (the eigenvalues equal those without the reaction
     field to 1e-10); the singlet's LR term is not.
  5. off is the default bit for bit, in the gas phase and in the continuum;
     refusals (no continuum, not fold, a bad mode).
  6. over simulated MPI ranks (2 and 3, one tile per orbital) the density is
     the serial one bit for bit, on every rank.
  7. ADC(2) in a continuum, end to end and independently of this code: in the
     PTE scheme the ground state's reaction field is a FIXED set of surface
     charges q0. Formaldehyde cc-pVDZ in toluene: the fold (LR and SS off) on
     the solvated SCF equals pyscf's own ADC(2) (radc singlets, uadc Ms = 0
     triplets) on an RHF/UHF whose core Hamiltonian carries q0 at the cavity
     points (pyscf's own smeared-charge potential), re-converged from scratch,
     the same DF fit on both sides, to 1e-7 Ha; q0 scaled by 1.01 misses by
     more than 1e-6 Ha (the check can fail).

Run: python tests/test_ee_adc_state_specific.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import adc, gto, scf, tdscf
from pyscf.adc import radc, uadc
from pyscf.solvent import pcm
from pyscf.solvent._attach_solvent import _for_tdscf

from src.Base.environment import NoEnvironment
from src.Base.pyscf_interface import (DFIntegrals, get_orbital_energies,
                                      get_two_electron_integrals_chemist)
from src.Base.solvent_screening import SolventScreening
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.ADC.eeADC import (ee_density, ee_fold, ee_ri_density,
                                           ee_ri_fold, ee_state_specific)
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
H2CO = 'C 0 0 0; O 0 0 1.205; H 0 0.94 -0.587; H 0 -0.94 -0.587'
PARITY = {'singlet': +1, 'triplet': -1}


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def rhf(atom, basis, df=False):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    if df:
        mf = mf.density_fit()
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


def pyscf_targets(mf, nroots):
    """{spin: [(omega, Delta rho)]} from pyscf's ISR-ADC(2) densities:
    the excited-state density minus rho_HF minus T1 rho_0^(2)."""
    no = mf.mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    hf = np.diag([2.0] * no + [0.0] * (nmo - no))
    out = {}
    a = adc.ADC(mf)
    a.method, a.method_type, a.conv_tol, a.tol_residual = 'adc(2)', 'ee', 1e-12, 1e-10
    e, _, _, _ = a.kernel(nroots=nroots)
    P = a.make_rdm1(with_frozen=False)
    p0 = radc.make_ref_rdm1(a, with_frozen=False) - hf
    out['singlet'] = (e, P, p0)
    ua = adc.ADC(scf.addons.convert_to_uhf(mf))
    ua.method, ua.method_type, ua.conv_tol, ua.tol_residual = 'adc(2)', 'ee', 1e-12, 1e-10
    eu, _, _, _ = ua.kernel(nroots=2 * nroots + 2)
    Pu = ua.make_rdm1(with_frozen=False)
    ru = uadc.make_ref_rdm1(ua, with_frozen=False)
    out['triplet'] = (eu, [Pu[0][k] + Pu[1][k] for k in range(len(eu))],
                      ru[0] + ru[1] - hf)
    return out, hf


def check_against_pyscf():
    """Check 1."""
    ok = True
    mf = rhf(WATER, '6-31g')
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    nv = len(eps) - no
    V = get_two_electron_integrals_chemist(mol, mf, representation='spatial'
                                           ).transpose(0, 2, 1, 3)
    targets, hf = pyscf_targets(mf, 3)
    for spin in ('singlet', 'triplet'):
        e, Z = solve_ee_adc(mf, level='adc2', spin=spin, nroots=2, conv_tol=1e-11)
        ep, P, p0 = targets[spin]
        for k in range(2):
            j = int(np.argmin(np.abs(ep - e[k])))
            T1 = 2.0 * float(Z[:no * nv, k] @ Z[:no * nv, k])
            target = P[j] - hf - T1 * p0
            terms = ee_density.spin_free_difference_density(
                eps, V, no, Z[:, k], parity=PARITY[spin], terms=True)
            err = float(np.abs(ee_density.assemble(terms, no, nv) - target).max())
            ok &= check(abs(ep[j] - e[k]) < 1e-8 and err < 1e-6,
                        f'{spin} {k}: dense density equals pyscf ISR-ADC(2)',
                        f'omega {abs(ep[j] - e[k]):.1e} Ha, density {err:.1e}')
            miss = {n: float(np.abs(ee_density.assemble(terms, no, nv, drop=(n,))
                                    - target).max())
                    for n in ee_density.ALL_TERMS}
            weakest = min(miss, key=miss.get)
            ok &= check(miss[weakest] > 1e-5,
                        f'{spin} {k}: every one of the 20 terms is needed',
                        f'weakest {weakest}: {miss[weakest]:.1e}')
    return ok


def check_ri_against_dense(atom):
    """Check 2 on one molecule."""
    ok = True
    mf = rhf(atom, 'cc-pvdz', df=True)
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    nv = len(eps) - no
    B = DFIntegrals.from_scf(mol, mf).B_aa
    V = np.einsum('Ppq,Prs->pqrs', B, B, optimize=True).transpose(0, 2, 1, 3)
    F = ee_ri_fold.RIFactors.from_full(eps, B, no)
    for spin in ('singlet', 'triplet'):
        e, Z = solve_ee_adc(mf, level='adc2', df=True, spin=spin, nroots=2,
                            conv_tol=1e-10)
        op = ee_ri_fold.RIFoldADC2(F, spin)
        r = ee_fold.solve_folded(op.pieces(), 2, spin=spin, tol_omega=1e-11,
                                 tol_residual=1e-10, dense_limit=0, solver='newton')
        ok &= check(np.abs(r.omega - e).max() < 1e-9,
                    f'{spin}: fold and full Davidson roots agree',
                    f'{np.abs(r.omega - e).max():.1e} Ha')
        for k in range(2):
            ref = ee_density.spin_free_difference_density(
                eps, V, no, Z[:, k], parity=PARITY[spin], terms=True)
            got = ee_ri_density.difference_density(op, r.y[:, k], r.omega[k],
                                                   r.t1[k], terms=True)
            tiny = ee_ri_density.difference_density(op, r.y[:, k], r.omega[k],
                                                    r.t1[k], terms=True,
                                                    tile_gb=1e-9)
            err = max(float(np.abs(ref[n] - got[n]).max())
                      for n in ee_density.ALL_TERMS)
            errt = max(float(np.abs(tiny[n] - got[n]).max())
                       for n in ee_density.ALL_TERMS)
            D = ee_density.assemble(got, no, nv)
            ok &= check(err < 1e-8 and errt < 1e-12,
                        f'{spin} {k}: tiled density equals the dense one, '
                        'term by term', f'{err:.1e}, tiles {errt:.1e}')
            ok &= check(abs(np.trace(D)) < 1e-12 and
                        np.abs(D - D.T).max() < 1e-12,
                        f'{spin} {k}: traceless and symmetric',
                        f'trace {np.trace(D):.1e}')
    return ok


def _pcm_like(env, mol, eps):
    """pyscf's own PCM object on env's cavity at dielectric constant eps."""
    p = pcm.PCM(mol)
    p.eps = eps
    p.method = env.method
    for k in ('lebedev_order', 'vdw_scale', 'r_probe', 'radii_table'):
        setattr(p, k, env._cavity[k])
    p.verbose = 0
    p.build()
    return p


def _second_difference(p, dm0, dP):
    """[E(P0 + dP) + E(P0 - dP) - 2 E(P0)] / 2 of pyscf's PCM energy."""
    f = [p._get_vind(dm)[0] for dm in (dm0 + dP, dm0 - dP, dm0)]
    return 0.5 * (f[0] + f[1] - 2.0 * f[2])


def toluene(atom, basis):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    env = SolventScreening(mol, solvent='toluene')

    def factory(m):
        mf = scf.RHF(m).density_fit()
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-8
        mf.kernel()
        return mf

    return mol, env, env.mean_field(mol, factory)


def check_energy_formula():
    """Check 3."""
    ok = True
    mol, env, mf = toluene(H2CO, 'cc-pvdz')
    C = np.asarray(mf.mo_coeff)
    dm0 = mf.make_rdm1()
    # a generic traceless density: one root of the fold at adc2
    e, _, res = solve_ee_adc(mf, level='adc2', spin='singlet', nroots=1,
                             fold=True, environment=env, conv_tol=1e-9,
                             return_fold=True, state_specific='vertical')
    dP = C @ res.delta_density[0] @ C.T
    for mode, eps in (('vertical', env.eps), ('equilibrium', env.eps_static)):
        ref = _second_difference(_pcm_like(env, mol, eps), dm0, dP)
        got = ee_state_specific.state_specific_energies(
            env, mol, C, res.delta_density, (mode,))[mode][0]
        ok &= check(abs(got - ref) < 1e-10 and got < 0,
                    f'{mode}: (1/2) dV Q dV is pyscf\'s PCM energy functional',
                    f'{got:.10f} vs {ref:.10f} Ha')
        ok &= check(abs(res.ss_shifts[mode][0] - got) < 1e-14,
                    f'{mode}: the driver records that shift')
    ok &= check(abs(e[0] - res.omega[0] - res.ss_shift[0]) < 1e-14,
                'e = omega + the vertical shift')
    # the CIS limit: pyscf's TDA-PCM vectors give the density independently
    fast = mf.with_solvent.copy()
    fast.reset()
    fast.eps = env.eps
    fast.equilibrium_solvation = False
    fast.build()
    td = _for_tdscf(tdscf.rhf.TDA(mf), solvent_obj=fast)
    # four states: pyscf's Davidson on two lands on spurious roots here
    td.nstates, td.conv_tol = 4, 1e-12
    td.kernel()
    e1, _, r1 = solve_ee_adc(mf, level='adc1', spin='singlet', nroots=2,
                             fold=True, environment=env, conv_tol=1e-10,
                             return_fold=True, state_specific='vertical',
                             dense_limit=0)
    no = mol.nelectron // 2
    p = _pcm_like(env, mol, env.eps)
    for k in range(2):
        # pyscf's singlet X has sum X^2 = 1/2, the alpha amplitudes of a
        # unit vector; the spin-summed CIS difference density is twice
        # (-X X^T, X^T X)
        X = np.asarray(td.xy[k][0])
        Dmo = np.zeros_like(dm0)
        Dmo[:no, :no] = -2.0 * X @ X.T
        Dmo[no:, no:] = 2.0 * X.T @ X
        ref = _second_difference(p, dm0, C @ Dmo @ C.T)
        got = r1.ss_shifts['vertical'][k]
        ok &= check(abs(td.e[k] - r1.omega[k]) < 1e-8 and abs(got - ref) < 1e-8,
                    f'CIS limit root {k}: shift from pyscf TDA-PCM density',
                    f'{got:.10f} vs {ref:.10f} Ha')
    return ok


def check_driver():
    """Check 4."""
    ok = True
    mol, env, mf = toluene(H2CO, 'aug-cc-pvdz')
    kw = dict(level='adc2', nroots=2, frozen=2, auxbasis='aug-cc-pvdz-ri',
              conv_tol=1e-8, fold=True, return_fold=True)
    for spin in ('singlet', 'triplet'):
        e, _, res = solve_ee_adc(mf, spin=spin, environment=env,
                                 state_specific='equilibrium', **kw)
        sv, se = res.ss_shifts['vertical'], res.ss_shifts['equilibrium']
        ok &= check(np.array_equal(e, res.omega + se),
                    f'{spin}: e = omega + the equilibrium shift')
        ok &= check(np.all(sv < -1e-6) and np.all(se < sv),
                    f'{spin}: SS shifts negative, equilibrium the larger',
                    'vertical ' + ', '.join(f'{x * 27.211386:+.4f}' for x in sv)
                    + ' eV; equilibrium '
                    + ', '.join(f'{x * 27.211386:+.4f}' for x in se) + ' eV')
        e_off, _ = solve_ee_adc(mf, spin=spin, environment=env,
                                **dict(kw, return_fold=False))
        e_no, _ = solve_ee_adc(mf, spin=spin, environment=NoEnvironment(),
                               **dict(kw, return_fold=False))
        lr = float(np.abs(e_off - e_no).max())
        if spin == 'triplet':
            ok &= check(lr < 1e-10, 'triplet: the LR term is zero, the SS '
                        'term is not', f'LR {lr:.1e} Ha')
        else:
            ok &= check(lr > 1e-6, 'singlet: the LR term is not zero',
                        f'{lr * 27.211386:.4f} eV')
        ok &= check(np.array_equal(e_off, res.omega),
                    f'{spin}: off is the eigenvalue itself')
    return ok


def check_default_and_refusals():
    """Check 5."""
    ok = True
    gas = rhf(WATER, 'cc-pvdz', df=True)
    kw = dict(level='adc2', spin='singlet', nroots=2, fold=True, conv_tol=1e-8)
    a, _ = solve_ee_adc(gas, **kw)
    b, _ = solve_ee_adc(gas, state_specific='off', **kw)
    ok &= check(np.array_equal(a, b), 'gas phase: off is the default bit for bit')
    _, env, mf = toluene(WATER, 'cc-pvdz')
    a, _ = solve_ee_adc(mf, environment=env, **kw)
    b, _ = solve_ee_adc(mf, environment=env, state_specific='off', **kw)
    ok &= check(np.array_equal(a, b), 'continuum: off is the default bit for bit')
    cases = [('state_specific in the gas phase', ValueError,
              dict(kw, state_specific='vertical'), gas),
             ('state_specific off the fold', NotImplementedError,
              dict(kw, fold=False, state_specific='vertical',
                   environment=env), mf),
             ('an unknown mode', ValueError,
              dict(kw, state_specific='relaxed', environment=env), mf)]
    for label, exc, args, m in cases:
        try:
            solve_ee_adc(m, **args)
            ok &= check(False, f'{label} raises')
        except exc as err:
            ok &= check(True, f'{label} raises', str(err)[:60])
    return ok


def check_ranks():
    """Check 6."""
    mf = rhf(WATER, 'cc-pvdz', df=True)
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    F = ee_ri_fold.RIFactors.from_full(eps, DFIntegrals.from_scf(mol, mf).B_aa, no)
    op = ee_ri_fold.RIFoldADC2(F, 'triplet')
    r = ee_fold.solve_folded(op.pieces(), 1, spin='triplet', dense_limit=0,
                             solver='newton')
    args = (r.y[:, 0], r.omega[0], r.t1[0])

    def rank(comm):
        o = ee_ri_fold.RIFoldADC2(F, 'triplet', tile_gb=1e-9, threads=1,
                                  comm=comm)
        return ee_ri_density.difference_density(o, *args, tile_gb=1e-9)

    ser = rank(None)
    ok = True
    for size in (2, 3):
        outs = run_simulated(rank, size)
        ok &= check(all(np.array_equal(o, ser) for o in outs),
                    f'{size} simulated ranks: the serial density bit for bit')
    return ok


def _frozen_charge_omegas(mol, sol, scale):
    """pyscf ADC(2) (S1, S2 from radc; T1, T2 from uadc) on references whose
    core Hamiltonian holds `scale` times the solvated SCF's apparent surface
    charges, frozen."""
    pcm = sol.with_solvent
    dm = sol.make_rdm1()
    pcm._get_vind(dm)
    q0 = pcm._intermediates['q_sym'].copy()
    v = pcm._get_vmat(scale * q0)[0]
    out = {}
    for spin, cls, n in (('singlet', scf.RHF, 3), ('triplet', scf.UHF, 6)):
        m = cls(mol).density_fit(auxbasis=sol.with_df.auxbasis)
        h = m.get_hcore() + v
        m.get_hcore = lambda *a, h=h, **k: h
        m.conv_tol, m.conv_tol_grad = 1e-13, 1e-9
        m.kernel(dm0=dm if cls is scf.RHF else np.array([dm / 2, dm / 2]))
        a = adc.ADC(m)
        a.method, a.method_type, a.conv_tol, a.tol_residual = 'adc(2)', 'ee', 1e-12, 1e-10
        out[spin] = np.asarray(a.kernel(nroots=n)[0])
    return out


def check_frozen_charges():
    """Check 7."""
    ok = True
    mol, _, sol = toluene(H2CO, 'cc-pvdz')
    ours = {spin: solve_ee_adc(sol, level='adc2', spin=spin, nroots=2,
                               fold=True, environment=NoEnvironment(),
                               conv_tol=1e-10)[0]
            for spin in ('singlet', 'triplet')}
    for scale in (1.0, 1.01):
        ref = _frozen_charge_omegas(mol, sol, scale)
        for spin in ('singlet', 'triplet'):
            # uadc interleaves singlets and triplets: the nearest root
            d = max(float(np.min(np.abs(ref[spin] - x))) for x in ours[spin])
            if scale == 1.0:
                ok &= check(d < 1e-7, f'{spin}: fold in the continuum equals '
                            'pyscf ADC(2) on the frozen charges q0', f'{d:.1e} Ha')
            else:
                ok &= check(d > 1e-6, f'{spin}: q0 x 1.01 is caught',
                            f'{d:.1e} Ha')
    return ok


def run():
    """Every check; True when all passed."""
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        all_ok &= check_against_pyscf()
        all_ok &= check_ri_against_dense(WATER)
        all_ok &= check_ri_against_dense(H2CO)
        all_ok &= check_energy_formula()
        all_ok &= check_driver()
        all_ok &= check_default_and_refusals()
        all_ok &= check_ranks()
        all_ok &= check_frozen_charges()
    return all_ok


def test_ee_adc_state_specific():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
