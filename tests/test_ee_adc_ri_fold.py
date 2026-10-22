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
  7. the gas phase: an explicit NoEnvironment is the default (two solves:
     to a few ulp, `matches` of tests/pins.py).
  8. refusals: a continuum with eps_static on a gas-phase mean field, fold
     with spin None, fold at adc3, a continuum on the spin-orbital route.
  9. over simulated MPI ranks (2 and 3, one tile per occupied orbital) the
     operator's set-up, its products and the folded roots driven from rank 0,
     by solver 'newton' and by the default's Davidson, are the serial ones to
     a few ulp, on every rank.
 10. B_vv split over the ranks: bare_factors on rank r of 3 keeps exactly its
     auxiliary chunks, to a few ulp the slices of the whole factor; over 2 and 3
     simulated ranks each holding only its chunks, the products (M U, A_eff(w)
     U with |Y|^2, fold_images, doubles_dot) and the block solver's roots are
     the serial whole-factor ones to a few ulp, on every rank.
 11. the fold-diagonal model the block solver seeds with: its Laplace sums equal
     the direct particle and hole sums to 1e-4 relative, and, its mean matched, it
     follows the exact diagonal of the folded doubles (correlation above 0.7,
     rms under 0.8 of the diagonal's spread), singlets and triplets; zeros at
     ADC(1).
 12. the operator's dA_eff/dw images, fold_images with a derivative node, equal
     folded_operator's dmatvec(u, vector=True) on ee_r_sigma_df's pieces (same
     factors) to 1e-12 relative, singlets and triplets at w = 0.3 and 0.9 Ha, and
     FoldedMatvec.both gives A_eff U and dA_eff/dw U in one pass, the same to
     1e-12; the root-wise Davidson (solve_folded's default) on the operator
     returns solver 'newton''s roots to 1e-8 Ha.

Run: python tests/test_ee_adc_ri_fold.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, mp, tdscf

from src.Base.constants import (EE_RI_FOLD_DIAG_HOLE, EE_RI_FOLD_DIAG_PARTICLE,
                                HARTREE_TO_EV)
from src.Base.environment import NoEnvironment, attach_environment
from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.Base.solvent_screening import SolventScreening
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.ADC.eeADC import ee_fold, ee_r_sigma_df, ee_ri_fold
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
from tests.adc_common import H2CO, WATER_ORIGIN, check, df_rhf_converged, rhf_df
from tests.pins import RECORDING_BLAS, matches


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
    # B_vv comes in its auxiliary chunks, all on this one rank
    Gvv = np.concatenate([G.bvv(c, p0, p1) for c, (p0, p1) in
                          enumerate(ee_ri_fold.aux_chunks(G.naux, G.nv, 1)[0])], axis=1)
    errf = max(float(np.abs(G.L - F.L).max()), float(np.abs(G.Boo - F.Boo).max()),
               float(np.abs(Gvv - F.Bvv).max()))
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

    return mol, env, env.mean_field(mol, df_rhf_converged)


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
    # two solves: the same roots to a few ulp, not bitwise
    ok &= check(matches(ea, ee_, RECORDING_BLAS, quantity='root'),
                'attached environment equals explicit one')
    return ok


def check_gas_default(mf):
    """Check 7."""
    a, _ = solve_ee_adc(mf, level='adc2', df=True, spin='singlet', nroots=2)
    b, _ = solve_ee_adc(mf, level='adc2', df=True, spin='singlet', nroots=2,
                        environment=NoEnvironment())
    # two solves: the same roots to a few ulp, not bitwise
    return check(matches(a, b, RECORDING_BLAS, quantity='root'),
                 'gas phase: NoEnvironment is the default')


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
    ok = True
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    F = ee_ri_fold.RIFactors.from_full(eps, DFIntegrals.from_scf(mol, mf).B_aa, no)
    u = np.random.default_rng(5).standard_normal((F.no * F.nv, 2))

    def solve(o):
        r = ee_fold.solve_folded(o.pieces(), 3, spin='triplet', dense_limit=0,
                                 solver='newton')
        d = ee_fold.solve_folded(o.pieces(), 3, spin='triplet', dense_limit=0)
        return dict(omega=r.omega, t1=r.t1, omega_d=d.omega)

    def rank(comm):
        op = ee_ri_fold.RIFoldADC2(F, 'triplet', tile_gb=1e-9, threads=1,
                                   comm=comm)
        prod = op.apply(u, [0.4, 0.7])
        return ee_ri_fold.drive(op, solve), op.diag, op.e_mp2, prod

    ser = rank(None)
    for size in (2, 3):
        outs = run_simulated(rank, size)
        # a rank split against the serial run: to a few ulp
        same = all(matches(o['omega'], ser[0]['omega'], RECORDING_BLAS,
                           quantity='root')
                   and matches(o['t1'], ser[0]['t1'], RECORDING_BLAS, unit='1',
                               quantity='vector')
                   and matches(o['omega_d'], ser[0]['omega_d'], RECORDING_BLAS,
                               quantity='root')
                   and matches(d, ser[1], RECORDING_BLAS, quantity='root')
                   and matches(e, ser[2], RECORDING_BLAS)
                   and matches(p[0], ser[3][0], RECORDING_BLAS, quantity='root')
                   and matches(p[1], ser[3][1], RECORDING_BLAS, quantity='root')
                   for o, d, e, p in outs)
        ok &= check(same, f'{size} simulated ranks equal the serial run to a few ulp')
    return ok


class _RankStub:
    """Get_rank/Get_size of one rank, for a build that only asks its place."""

    def __init__(self, rank, size):
        self.rank, self.size = rank, size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size


def check_split_bvv(mf):
    """Check 10."""
    ok = True
    mol = mf.mol
    no = mol.nelectron // 2
    # chunks of 16 auxiliary functions: several per rank at water/cc-pVDZ
    width = 16
    whole = ee_ri_fold.bare_factors(mol, mf, auxbasis='cc-pvdz-ri')
    # the one-rank factor's B_vv, whole, from its own chunks
    wvv = np.concatenate([whole.bvv(c, p0, p1) for c, (p0, p1) in enumerate(
        ee_ri_fold.aux_chunks(whole.naux, whole.nv, 1)[0])], axis=1)
    chunks, owners = ee_ri_fold.aux_chunks(whole.naux, whole.nv, 3, width)
    for r in range(3):
        part = ee_ri_fold.bare_factors(mol, mf, auxbasis='cc-pvdz-ri',
                                       comm=_RankStub(r, 3), aux_chunk=width)
        mine = [c for c, o in enumerate(owners) if o == r]
        # each chunk built again on the rank: to a few ulp
        same = (part.Bvv is None and sorted(part.bvv_chunks) == mine and all(
            matches(part.bvv_chunks[c], wvv[:, slice(*chunks[c])],
                    RECORDING_BLAS, unit='1', quantity='root')
            for c in mine) and matches(part.L, whole.L, RECORDING_BLAS,
                                       unit='1', quantity='root'))
        ok &= check(same, f'rank {r} of 3 keeps its B_vv chunks {mine}, to a few ulp',
                    f'{part.nbytes() / whole.nbytes():.2f} of the whole factors')
    eps = np.asarray(get_orbital_energies(mf), float)
    F = ee_ri_fold.RIFactors.from_full(eps, DFIntegrals.from_scf(mol, mf).B_aa, no)
    U = np.random.default_rng(9).standard_normal((F.no * F.nv, 2))

    def solve(o):
        r = ee_fold.solve_folded(o.pieces(), 3, spin='singlet', dense_limit=0,
                                 solver='block')
        return dict(omega=r.omega, t1=r.t1)

    def products(comm):
        size = 1 if comm is None else comm.Get_size()
        rank = 0 if comm is None else comm.Get_rank()
        ch, ow = ee_ri_fold.aux_chunks(F.naux, F.nv, size, width)
        G = F if comm is None else ee_ri_fold.RIFactors(
            F.eo, F.ev, F.L, F.Boo, None,
            {c: np.ascontiguousarray(F.Bvv[:, slice(*ch[c])])
             for c, o in enumerate(ow) if o == rank})
        op = ee_ri_fold.RIFoldADC2(G, 'singlet', tile_gb=1e-9, threads=1, comm=comm,
                                   aux_chunk=width)
        m0 = op.apply(U, None)[0]
        a, nrm = op.apply(U, [0.4, 0.7])
        img = op.fold_images(U, [0.3, 0.5])[1]
        dd = op.doubles_dot(U[:, 0], 0.4, U[:, 1], 0.6)
        return [m0, a, nrm, img, np.array([dd]), op.diag], ee_ri_fold.drive(op, solve)

    ser, ser_roots = products(None)
    for size in (2, 3):
        outs = run_simulated(products, size)
        same = all(all(matches(x, y, RECORDING_BLAS, quantity='root')
                       for x, y in zip(o, ser))
                   and matches(r['omega'], ser_roots['omega'], RECORDING_BLAS,
                               quantity='root')
                   and matches(r['t1'], ser_roots['t1'], RECORDING_BLAS,
                               unit='1', quantity='vector') for o, r in outs)
        ok &= check(same, f'{size} simulated ranks, B_vv split: products and block '
                    'roots equal the serial whole-factor run to a few ulp')
    return ok


def check_fold_diagonal(mf):
    """Check 11."""
    ok = True
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    F = ee_ri_fold.RIFactors.from_full(eps, B, no)
    eo, ev = F.eo, F.ev
    # (ac|ld) as [a, c, l, d] and (ki|ld) as [k, i, l, d]
    vvov = np.einsum('aPc,ldP->acld', F.Bvv, F.L.reshape(no, F.nv, -1))
    ooov = np.einsum('kiP,ldP->kild', F.Boo, F.L.reshape(no, F.nv, -1))
    for spin in ee_ri_fold.SPINS:
        op = ee_ri_fold.RIFoldADC2(F, spin, threads=2)
        w = 0.5 * op.d_min
        dp = np.array([np.einsum('acld,cld->a', vvov ** 2, 1.0 / (
            w + eo[i] + eo[None, :, None] - ev[:, None, None] - ev[None, None, :]))
            for i in range(no)])
        dh = np.array([np.einsum('kild,kld->i', ooov ** 2, 1.0 / (
            w - ev[a] + eo[:, None, None] + eo[None, :, None] - ev[None, None, :]))
            for a in range(F.nv)]).T
        model = op.fold_diagonal(w)
        direct = (EE_RI_FOLD_DIAG_PARTICLE * dp + EE_RI_FOLD_DIAG_HOLE * dh).ravel()
        err = float(np.abs(model - direct).max() / np.abs(direct).max())
        ok &= check(err < 1e-4, f'{spin}: fold_diagonal is the Laplace form of the '
                    'direct sums', f'{err:.1e} relative')
        I = np.eye(op.n)
        exact = np.einsum('pk,pk->k', I, op.apply(I, np.full(op.n, w))[0]
                          - op.apply(I, None)[0])
        shifted = model - model.mean() + exact.mean()
        corr = float(np.corrcoef(model, exact)[0, 1])
        rms = float(np.sqrt(np.mean((shifted - exact) ** 2)))
        ok &= check(corr > 0.7 and rms < 0.8 * exact.std(),
                    f'{spin}: fold_diagonal follows the exact folded diagonal',
                    f'corr {corr:.2f}, rms {rms:.4f} vs spread {exact.std():.4f} Ha')
    op1 = ee_ri_fold.RIFoldADC2(F, 'singlet', level='adc1', threads=2)
    ok &= check(not np.any(op1.fold_diagonal(0.3)), 'adc1: no fold diagonal')
    return ok


def check_derivative(mf):
    """Check 12."""
    ok = True
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    F = ee_ri_fold.RIFactors.from_full(eps, B, no)
    rng = np.random.default_rng(7)

    def rel(a, b):
        return float(np.abs(a - b).max() / np.abs(b).max())

    for spin in ee_ri_fold.SPINS:
        op = ee_ri_fold.RIFoldADC2(F, spin, threads=2)
        for om in (0.3, 0.9):
            dmv = ee_fold.folded_operator(P, om, spin)[1]
            dmo = ee_fold.folded_operator(op.pieces(), om, spin)[1]
            U = rng.standard_normal((op.n, 2))
            ref = np.column_stack([dmv(u, vector=True) for u in U.T])
            got = np.column_stack([dmo(u, vector=True) for u in U.T])
            fm = ee_ri_fold.FoldedMatvec(op, om)
            A0, A1 = fm.both(U)
            e = (rel(got, ref), rel(A1, ref), rel(A0, fm.matmat(U)))
            ok &= check(max(e) < 1e-12,
                        f'{spin} w={om}: the operator\'s dA_eff/dw images, and '
                        'both() in one pass', ', '.join(f'{x:.1e}' for x in e))
        a = ee_fold.solve_folded(op.pieces(), 3, spin=spin, dense_limit=0)
        b = ee_fold.solve_folded(op.pieces(), 3, spin=spin, dense_limit=0,
                                 solver='newton')
        d = (float(np.abs(a.omega - b.omega).max())
             if a.omega.shape == b.omega.shape else np.inf)
        ok &= check(d < 1e-8 and set(a.loop) == {'davidson'},
                    f'{spin}: the default\'s Davidson on the operator returns '
                    "solver 'newton''s roots", f'{d:.1e} Ha, loop {a.loop}')
    return ok


def run():
    """Every check; True when all passed."""
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mf = rhf_df(WATER_ORIGIN, 'cc-pvdz')
        all_ok &= check_operator(mf)
        all_ok &= check_solves(mf)
        all_ok &= check_cis_pcm()
        all_ok &= check_adc2_pcm()
        all_ok &= check_gas_default(mf)
        all_ok &= check_refusals(mf)
        all_ok &= check_ranks(mf)
        all_ok &= check_split_bvv(mf)
        all_ok &= check_fold_diagonal(mf)
        all_ok &= check_derivative(mf)
    return all_ok


def test_ee_adc_ri_fold():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
