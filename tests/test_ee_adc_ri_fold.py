"""Tiled density-fitted ADC(2) fold (ee_ri_fold) and EE-ADC in a continuum.

ee_ri_fold builds A_eff(w) = M - C (D - w)^-1 C^T of one spin channel from the
DF factors one occupied tile at a time, never holding an array of doubles size;
solve_ee_adc(fold=True) runs ee_fold's solver on it. A continuum enters the
singles block as the linear-response term 2 (ia|vtilde|jb) of singlets, every
second-order term keeping the bare interaction. Checks:

  1. the operator equals ee_fold.folded_operator on ee_r_sigma_df's pieces
     (same factors), singlets and triplets, at w = None, 0.3 and 0.9 Ha: the
     product, the doubles norm, doubles_dot, the block product, to 1e-12; its
     diagonal is the diagonal of its dense matrix; one tile per occupied
     orbital gives the same numbers as the default tile.
  2. its MP2 energy equals pyscf's DF-MP2 on the same factors.
  3. bare_factors equals the slices of DFIntegrals.from_scf (gas phase).
  4. solve_ee_adc(fold=True) equals solve_folded on the explicit pieces and the
     full spin-free Davidson (frozen core, a separate RI basis) to 1e-8 eV,
     singles weights in (0, 1].
  5. ADC(1) in toluene, the tiled route and the full spin-free route, equals
     pyscf's TDA with a non-equilibrium (optical) PCM response on the SCF
     relaxed at eps_static to 1e-6 eV (singlets and triplets); a triplet
     equals the same mean field without the reaction field.
  6. ADC(2) in toluene: the fold equals the full spin-free route on the same
     solvated reference to 1e-6 eV; an environment attached to the mean field
     gives the numbers of the one passed explicitly; the singlet shift against
     the gas phase is negative and under 0.3 eV for formaldehyde's n->pi*.
  7. the gas phase: an explicit NoEnvironment is the default bit for bit.
  8. refusals: a continuum with eps_static on a gas-phase mean field, fold
     with spin None, fold at adc3, a continuum on the spin-orbital route.
  9. over simulated MPI ranks (2 and 3, one tile per occupied orbital) the
     operator's set-up, its products and the folded roots driven from rank 0
     are the serial ones bit for bit, on every rank.

Run: python tests/test_ee_adc_ri_fold.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, mp, scf, tdscf

from src.Base.constants import HARTREE_TO_EV
from src.Base.environment import NoEnvironment, attach_environment
from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.ADC.eeADC import ee_fold, ee_r_sigma_df, ee_ri_fold
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'
H2CO = 'C 0 0 0; O 0 0 1.205; H 0 0.94 -0.587; H 0 -0.94 -0.587'


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def rhf_df(atom, basis):
    """A tightly converged DF-RHF mean field."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    return mf


def check_operator(mf):
    """Checks 1 and 2."""
    ok = True
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    F = ee_ri_fold.RIFactors.from_full(eps, B, no)
    rng = np.random.default_rng(3)
    for spin in ee_ri_fold.SPINS:
        op = ee_ri_fold.RIFoldADC2(F, spin, threads=2)
        tiny = ee_ri_fold.RIFoldADC2(F, spin, tile_gb=1e-9, threads=3)
        ok &= check(tiny.kt == 1 and op.kt == no, f'{spin}: tile sizes',
                    f'{tiny.kt}, {op.kt}')
        for om in (None, 0.3, 0.9):
            mv, dmv, _, _, _ = ee_fold.folded_operator(P, om, spin)
            u = rng.standard_normal(op.n)
            a = mv(u)
            b, nb = op.apply(u[:, None], None if om is None else [om])
            c, nc = tiny.apply(u[:, None], None if om is None else [om])
            ref = dmv(u)
            err = float(np.abs(a - b[:, 0]).max())
            errn = abs(ref - nb[0]) / max(1.0, abs(ref))
            errt = max(float(np.abs(b - c).max()), abs(nb[0] - nc[0]))
            ok &= check(err < 1e-12 and errn < 1e-12 and errt < 1e-12,
                        f'{spin} w={om}: product and doubles norm',
                        f'{err:.1e}, {errn:.1e}, tiles {errt:.1e}')
            if om is not None:
                d = op.doubles_dot(u, om, u, om)
                v = rng.standard_normal(op.n)
                blk = ee_ri_fold.FoldedMatvec(op, om).matmat(np.column_stack([u, v]))
                errb = float(np.abs(blk[:, 0] - a).max())
                ok &= check(abs(d - ref) < 1e-12 * max(1, ref) and errb < 1e-12,
                            f'{spin} w={om}: doubles_dot and block product',
                            f'{abs(d - ref):.1e}, {errb:.1e}')
        A = np.column_stack([op.apply(np.eye(op.n)[:, k][:, None])[0][:, 0]
                             for k in range(op.n)])
        ok &= check(np.abs(np.diag(A) - op.diag).max() < 1e-12,
                    f'{spin}: diag is the operator diagonal')
    e_ref = mp.MP2(mf).run().e_corr
    ok &= check(abs(op.e_mp2 - e_ref) < 1e-10, 'MP2 energy equals pyscf DF-MP2',
                f'{op.e_mp2:.12f} vs {e_ref:.12f}')
    # check 3: the bare factors
    G = ee_ri_fold.bare_factors(mol, mf)
    errf = max(float(np.abs(G.L - F.L).max()), float(np.abs(G.Boo - F.Boo).max()),
               float(np.abs(G.Bvv - F.Bvv).max()))
    ok &= check(errf < 1e-12, 'bare_factors equals DFIntegrals slices', f'{errf:.1e}')
    return ok


def check_solves(mf):
    """Check 4."""
    ok = True
    mol = mf.mol
    for spin in ee_ri_fold.SPINS:
        e, Z, res = solve_ee_adc(mf, level='adc2', spin=spin, nroots=3, fold=True,
                                 frozen=1, auxbasis='cc-pvdz-ri', conv_tol=1e-9,
                                 return_fold=True)
        er, _ = solve_ee_adc(mf, level='adc2', df=True, spin=spin, nroots=3,
                             frozen=1, auxbasis='cc-pvdz-ri', conv_tol=1e-9)
        d = float(np.abs(e - er).max()) * HARTREE_TO_EV
        ok &= check(d < 1e-8 and res.converged.all() and
                    np.all((res.t1 > 0) & (res.t1 <= 1)),
                    f'{spin}: fold=True equals the full Davidson', f'{d:.1e} eV')
        no = mol.nelectron // 2 - 1
        ok &= check(Z.shape == (2 * no * (mol.nao - no - 1), e.size),
                    f'{spin}: Z is the flat singles of the active space')
    return ok


def toluene(atom, basis):
    """(mol, env, mf): the SCF relaxed at toluene's eps_static."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    env = SolventScreening(mol, solvent='toluene')

    def factory(m):
        mf = scf.RHF(m).density_fit()
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-8
        mf.kernel()
        return mf

    return mol, env, env.mean_field(mol, factory)


def pyscf_tda_pcm(mf, eps_optical, spin, nroots):
    """pyscf TDA on the solvated SCF with a non-equilibrium PCM response."""
    from pyscf.solvent._attach_solvent import _for_tdscf
    fast = mf.with_solvent.copy()
    fast.reset()
    fast.eps = eps_optical
    fast.equilibrium_solvation = False
    fast.build()
    td = tdscf.rhf.TDA(mf)
    td.singlet = spin == 'singlet'
    td = _for_tdscf(td, solvent_obj=fast)
    td.nstates = nroots
    td.conv_tol = 1e-12
    td.kernel()
    return np.asarray(td.e)


def check_cis_pcm():
    """Check 5."""
    ok = True
    _, env, mf = toluene(H2CO, 'aug-cc-pvdz')
    ok &= check(abs(env.eps - 2.2383) < 1e-3 and abs(env.eps_static - 2.3741) < 1e-4,
                "toluene's dielectric constants", f'{env.eps}, {env.eps_static}')
    for spin in ee_ri_fold.SPINS:
        ref = pyscf_tda_pcm(mf, env.eps, spin, 4)
        e1, _ = solve_ee_adc(mf, level='adc1', df=True, spin=spin, nroots=4,
                             environment=env, conv_tol=1e-9)
        e2, _ = solve_ee_adc(mf, level='adc1', spin=spin, nroots=4, fold=True,
                             environment=env, conv_tol=1e-9, dense_limit=0)
        d1 = float(np.abs(e1 - ref).max()) * HARTREE_TO_EV
        d2 = float(np.abs(e2 - ref).max()) * HARTREE_TO_EV
        ok &= check(d1 < 1e-6 and d2 < 1e-6,
                    f'{spin}: ADC(1) in toluene equals pyscf TDA-PCM',
                    f'full {d1:.1e} eV, tiled {d2:.1e} eV')
        if spin == 'triplet':
            e0, _ = solve_ee_adc(mf, level='adc1', df=True, spin=spin, nroots=4,
                                 conv_tol=1e-9, environment=NoEnvironment())
            ok &= check(float(np.abs(e0 - e1).max()) * HARTREE_TO_EV < 1e-8,
                        'triplet: no transition reaction field')
    return ok


def check_adc2_pcm():
    """Check 6."""
    ok = True
    mol, env, mf = toluene(H2CO, 'aug-cc-pvdz')
    gas = rhf_df(H2CO, 'aug-cc-pvdz')
    kw = dict(level='adc2', nroots=2, frozen=2, auxbasis='aug-cc-pvdz-ri',
              conv_tol=1e-8)
    for spin in ee_ri_fold.SPINS:
        ef, _ = solve_ee_adc(mf, spin=spin, fold=True, environment=env, **kw)
        er, _ = solve_ee_adc(mf, spin=spin, df=True, environment=env, **kw)
        d = float(np.abs(ef - er).max()) * HARTREE_TO_EV
        ok &= check(d < 1e-6, f'{spin}: ADC(2) in toluene, fold equals full',
                    f'{d:.1e} eV')
        eg, _ = solve_ee_adc(gas, spin=spin, fold=True, **kw)
        shift = (ef[0] - eg[0]) * HARTREE_TO_EV
        print(f'     {spin} S1/T1 gas {eg[0] * HARTREE_TO_EV:.4f} eV, toluene '
              f'{ef[0] * HARTREE_TO_EV:.4f} eV, shift {shift:+.4f} eV')
        if spin == 'singlet':
            # n->pi*: the lone pair loses its reaction field, a blue shift in
            # polar solvents; toluene is nearly non-polar, so a few 10 meV
            ok &= check(abs(shift) < 0.3, 'singlet n->pi* shift is small',
                        f'{shift:+.4f} eV')
    attach_environment(mf, env)
    ea, _ = solve_ee_adc(mf, spin='singlet', fold=True, **kw)
    ee_, _ = solve_ee_adc(mf, spin='singlet', fold=True, environment=env, **kw)
    ok &= check(np.array_equal(ea, ee_), 'attached environment equals explicit one')
    return ok


def check_gas_default(mf):
    """Check 7."""
    a, _ = solve_ee_adc(mf, level='adc2', df=True, spin='singlet', nroots=2)
    b, _ = solve_ee_adc(mf, level='adc2', df=True, spin='singlet', nroots=2,
                        environment=NoEnvironment())
    return check(np.array_equal(a, b), 'gas phase: NoEnvironment is the default')


def check_refusals(mf):
    """Check 8."""
    ok = True
    env = SolventScreening(mf.mol, solvent='toluene')
    cases = [
        ('continuum on a gas-phase mean field', ValueError,
         dict(level='adc1', spin='singlet', environment=env)),
        ('fold with spin None', ValueError,
         dict(level='adc2', fold=True)),
        ('fold at adc3', ValueError,
         dict(level='adc3', spin='singlet', fold=True)),
        ('continuum on the spin-orbital route', NotImplementedError,
         dict(level='adc1', route='spinorbital', environment=env)),
    ]
    for label, exc, kw in cases:
        try:
            solve_ee_adc(mf, **kw)
            ok &= check(False, f'{label} raises')
        except exc as err:
            ok &= check(True, f'{label} raises', str(err)[:60])
    return ok


def check_ranks(mf):
    """Check 9."""
    from src.Base.utils.mpi_grid import run_simulated
    ok = True
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    F = ee_ri_fold.RIFactors.from_full(eps, DFIntegrals.from_scf(mol, mf).B_aa, no)
    u = np.random.default_rng(5).standard_normal((F.no * F.nv, 2))

    def solve(o):
        r = ee_fold.solve_folded(o.pieces(), 3, spin='triplet', dense_limit=0)
        return dict(omega=r.omega, t1=r.t1)

    def rank(comm):
        op = ee_ri_fold.RIFoldADC2(F, 'triplet', tile_gb=1e-9, threads=1,
                                   comm=comm)
        prod = op.apply(u, [0.4, 0.7])
        return ee_ri_fold.drive(op, solve), op.diag, op.e_mp2, prod

    ser = rank(None)
    for size in (2, 3):
        outs = run_simulated(rank, size)
        same = all(np.array_equal(o['omega'], ser[0]['omega'])
                   and np.array_equal(o['t1'], ser[0]['t1'])
                   and np.array_equal(d, ser[1]) and e == ser[2]
                   and np.array_equal(p[0], ser[3][0])
                   and np.array_equal(p[1], ser[3][1])
                   for o, d, e, p in outs)
        ok &= check(same, f'{size} simulated ranks equal the serial run bit for bit')
    return ok


def run():
    """Every check; True when all passed."""
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mf = rhf_df(WATER, 'cc-pvdz')
        all_ok &= check_operator(mf)
        all_ok &= check_solves(mf)
        all_ok &= check_cis_pcm()
        all_ok &= check_adc2_pcm()
        all_ok &= check_gas_default(mf)
        all_ok &= check_refusals(mf)
        all_ok &= check_ranks(mf)
    return all_ok


def test_ee_adc_ri_fold():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
