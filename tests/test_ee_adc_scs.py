"""Spin-component scaled ADC(2) on the density-fitted fold (ee_ri_fold).

SCS/SOS-ADC(2) weights every doubles configuration by its spin class, c_os for
an opposite-spin pair and c_ss for a same-spin pair, in the MP1 amplitudes of
the second-order singles block and in the folded coupling. Checks:

  1. the spin-orbital arbiter: the ADC(2) supermatrix built on full
     spin-orbital arrays from the same DF factors (ee_u_dense_full), its
     first-order amplitudes scaled by the pair's spin class and its coupling
     columns by sqrt(c), has every folded root to 1e-9 Ha as an eigenvalue
     whose singles are spin-conserving, for SCS, SOS and an arbitrary
     (c_os, c_ss); unscaled, the arbiter has the ADC(2) fold's roots, triplets
     three times (which checks the arbiter itself). Scaled, the triplet's
     Ms = +-1 components (spin-flip singles) lie elsewhere: spin-component
     scaling is not spin-rotation invariant, and the fold's triplet is the
     Ms = 0 component.
  2. (c_os, c_ss) = (1, 1) equals ADC(2) to 1e-12: M U, A_eff(w) U with |Y|^2,
     fold_images, doubles_dot, the diagonal and the MP2 energy, singlets and
     triplets.
  3. the scaled MP2 energy of the operator is pyscf's SCS-MP2 on the same
     factors, c_os E_os + c_ss E_ss.
  4. over 2 and 3 simulated MPI ranks a scaled operator's set-up, products and
     roots are the serial ones to a few ulp.
  5. solve_ee_adc(fold=True, spin_scaling='scs') gives the operator's roots;
     refusals: an unknown name, without fold, at ADC(1), with state_specific.
  6. with CuPy importable, the GPU doubles kernel equals the CPU path to 1e-10
     (scaled, singlets and triplets).

Run: python tests/test_ee_adc_scs.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import mp

from src.Base.numerics.constants import ADC2_SPIN_SCALING
from src.Base.pyscf.pyscf_interface import DFIntegrals, get_orbital_energies
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold, ee_u_dense_full, ee_utils
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
from tests.adc_common import WATER_ORIGIN, check, rhf_df
from tests.pins import RECORDING_BLAS, matches, same_solution

SCALINGS = (ADC2_SPIN_SCALING['scs'], ADC2_SPIN_SCALING['sos'], (0.7, 1.9))


def spin_orbital(eps, B, no):
    """(eps, <pq||rs>, spin) on spin orbitals ordered occ alpha, occ beta,
    virt alpha, virt beta, from the MO factor B (naux, n, n)."""
    n = eps.size
    nv = n - no
    spat = np.concatenate([np.arange(no), np.arange(no),
                           np.arange(no, n), np.arange(no, n)])
    spin = np.concatenate([np.zeros(no, int), np.ones(no, int),
                           np.zeros(nv, int), np.ones(nv, int)])
    eri = np.einsum('Ppq,Prs->pqrs', B, B, optimize=True)
    E = eri[np.ix_(spat, spat, spat, spat)]              # (pq|rs) on spin orbitals
    same = spin[:, None] == spin[None, :]
    # <pq|rs> = (pr|qs) delta(p,r) delta(q,s)
    phys = E.transpose(0, 2, 1, 3) * (same[:, None, :, None] & same[None, :, None, :])
    return eps[spat], phys - phys.transpose(0, 1, 3, 2), spin


def arbiter_roots(eps, B, no, scaling):
    """Eigenvalues of the spin-orbital ADC(2) supermatrix with its doubles
    scaled by spin class (scaling None: ADC(2))."""
    e_so, g, spin = spin_orbital(eps, B, no)
    nocc, norb = 2 * no, e_so.size
    amps = ee_utils.mp_amplitudes(e_so, g, nocc, norb, order=2)
    iu, ju, au, bu = ee_utils.pair_indices(nocc, norb)
    c_pair = np.ones(len(iu))
    if scaling is not None:
        c_os, c_ss = scaling
        so = spin[:nocc]
        cls = np.where(so[:, None] == so[None, :], c_ss, c_os)    # (nocc, nocc)
        amps = dict(amps, t2_1=amps['t2_1'] * cls[:, :, None, None])
        c_pair = cls[iu, ju]
    H = ee_u_dense_full.build_supermatrix(e_so, g, nocc, level='adc2', amps=amps)
    n_s = nocc * (norb - nocc)
    # one doubles row per (occupied pair, virtual pair), occupied pair major
    root = np.sqrt(np.repeat(c_pair, len(au)))
    H[:n_s, n_s:] *= root[None, :]
    H[n_s:, :n_s] *= root[:, None]
    w, V = np.linalg.eigh(H)
    so, sv = spin[:nocc], spin[nocc:]
    flip = (so[:, None] != sv[None, :]).ravel()
    # each root's spin-flip singles weight: Ms = +-1 where it dominates
    return w, np.sum(V[:n_s][flip] ** 2, axis=0)


def fold_roots(F, spin, scaling, nroots=3, comm=None):
    op = ee_ri_fold.RIFoldADC2(F, spin, scaling=scaling, comm=comm)
    r = ee_fold.solve_folded(op.pieces(), nroots, spin=spin, tol_omega=1e-11,
                             tol_residual=1e-9)
    return np.asarray(r.omega)


def check_arbiter(eps, B, no, F):
    """Check 1."""
    ok = True
    for scaling in (None,) + SCALINGS:
        ws, flip = arbiter_roots(eps, B, no, scaling)
        for spin in ('singlet', 'triplet'):
            om = fold_roots(F, spin, scaling)
            near = [np.flatnonzero(np.abs(ws - w) < 1e-9) for w in om]
            dev = max(float(np.min(np.abs(ws - w))) for w in om)
            ms0 = all(len(k) and np.min(flip[k]) < 1e-8 for k in near)
            counts = [len(k) for k in near]
            want = 3 if (spin == 'triplet' and scaling is None) else None
            # unscaled, the degenerate triplet components mix in eigh: count them
            ok &= check((ms0 if want is None else all(c == want for c in counts)),
                        f'{scaling}: {spin} roots are spin-conserving roots of '
                        'the spin-orbital matrix',
                        f'counts {counts}, max |dw| {dev:.1e} Ha')
        if scaling is not None:
            om = fold_roots(F, 'triplet', scaling, nroots=1)[0]
            ms1 = ws[flip > 0.5]
            print(f'      Ms = +-1 component of T1: {ms1[0]:.8f} Ha, Ms = 0 '
                  f'{om:.8f} Ha')
    return ok


def check_unit_scaling(F, mf):
    """Checks 2 and 3."""
    ok = True
    rng = np.random.default_rng(11)
    for spin in ee_ri_fold.SPINS:
        a = ee_ri_fold.RIFoldADC2(F, spin, tile_gb=1e-9)
        b = ee_ri_fold.RIFoldADC2(F, spin, tile_gb=1e-9, scaling=(1.0, 1.0))
        U = rng.standard_normal((a.n, 2))
        errs = [float(np.abs(a.diag - b.diag).max()), abs(a.e_mp2 - b.e_mp2)]
        for om in (None, [0.4, 0.8]):
            pa, na = a.apply(U, om)
            pb, nb = b.apply(U, om)
            errs += [float(np.abs(pa - pb).max()), float(np.abs(na - nb).max())]
        fa, fb = a.fold_images(U, [0.3, 0.9])[1], b.fold_images(U, [0.3, 0.9])[1]
        errs.append(float(np.abs(fa - fb).max()))
        errs.append(abs(a.doubles_dot(U[:, 0], 0.5, U[:, 1], 0.6)
                        - b.doubles_dot(U[:, 0], 0.5, U[:, 1], 0.6)))
        ok &= check(max(errs) < 1e-12, f'{spin}: (1, 1) equals ADC(2)',
                    f'{max(errs):.1e}')
    pt = mp.MP2(mf).run()
    for scaling in SCALINGS:
        op = ee_ri_fold.RIFoldADC2(F, 'singlet', scaling=scaling)
        ref = scaling[0] * pt.e_corr_os + scaling[1] * pt.e_corr_ss
        ok &= check(abs(op.e_mp2 - ref) < 1e-10, f'{scaling}: scaled MP2 energy',
                    f'{op.e_mp2:.12f} vs {ref:.12f}')
    return ok


def check_ranks(F):
    """Check 4."""
    ok = True
    u = np.random.default_rng(5).standard_normal((F.no * F.nv, 2))
    scaling = ADC2_SPIN_SCALING['scs']
    for spin in ee_ri_fold.SPINS:
        def solve(o, spin=spin):
            r = ee_fold.solve_folded(o.pieces(), 3, spin=spin, dense_limit=0)
            return dict(omega=r.omega, t1=r.t1)

        def rank(comm, spin=spin, solve=solve):
            op = ee_ri_fold.RIFoldADC2(F, spin, tile_gb=1e-9, threads=1,
                                       comm=comm, scaling=scaling)
            prod = op.apply(u, [0.4, 0.7])
            img = op.fold_images(u, [0.3, 0.6])[1]
            return ee_ri_fold.drive(op, solve), op.diag, op.e_mp2, prod, img

        ser = rank(None)
        for size in (2, 3):
            outs = run_simulated(rank, size)
            same = all(same_solution((o['omega'], o['t1']),
                                     (ser[0]['omega'], ser[0]['t1']))
                       and matches(d, ser[1], RECORDING_BLAS, quantity='root')
                       and matches(e, ser[2], RECORDING_BLAS)
                       and matches(p[0], ser[3][0], RECORDING_BLAS, quantity='root')
                       and matches(p[1], ser[3][1], RECORDING_BLAS, quantity='root')
                       and matches(im, ser[4], RECORDING_BLAS, quantity='root')
                       for o, d, e, p, im in outs)
            ok &= check(same, f'{spin}: {size} simulated ranks equal the serial '
                        'run to a few ulp')
    return ok


def check_driver(mf, F):
    """Check 5."""
    ok = True
    for spin in ee_ri_fold.SPINS:
        e, _ = solve_ee_adc(mf, level='adc2', spin=spin, nroots=3, fold=True,
                            conv_tol=1e-10, spin_scaling='scs')
        ref = fold_roots(F, spin, ADC2_SPIN_SCALING['scs'])
        ok &= check(np.abs(np.asarray(e) - ref).max() < 1e-8,
                    f'{spin}: solve_ee_adc spin_scaling=scs', f'{e}')
    for kw, label in ((dict(fold=True, spin_scaling='scss'), 'an unknown name'),
                      (dict(fold=False, spin_scaling='scs'), 'without fold'),
                      (dict(fold=True, level='adc1', spin_scaling='scs'), 'at ADC(1)'),
                      (dict(fold=True, spin_scaling='scs', state_specific='vertical'),
                       'with state_specific')):
        args = dict(level='adc2', spin='singlet', nroots=1)
        args.update(kw)
        try:
            solve_ee_adc(mf, **args)
            refused = False
        except (ValueError, NotImplementedError):
            refused = True
        ok &= check(refused, f'refused: {label}')
    return ok


def check_gpu(F):
    """Check 6."""
    if ee_ri_fold.cp is None:
        print('  [skip] GPU kernel: CuPy is not importable')
        return True
    ok = True
    U = np.random.default_rng(7).standard_normal((F.no * F.nv, 2))
    for spin in ee_ri_fold.SPINS:
        c = ee_ri_fold.RIFoldADC2(F, spin, scaling=ADC2_SPIN_SCALING['scs'])
        g = ee_ri_fold.RIFoldADC2(F, spin, scaling=ADC2_SPIN_SCALING['scs'],
                                  device='gpu')
        pc, nc = c.apply(U, [0.4, 0.7])
        pg, ng = g.apply(U, [0.4, 0.7])
        fc, fg = c.fold_images(U, [0.3])[1], g.fold_images(U, [0.3])[1]
        err = max(float(np.abs(pc - pg).max()), float(np.abs(nc - ng).max()),
                  float(np.abs(fc - fg).max()))
        ok &= check(err < 1e-10, f'{spin}: GPU kernel equals the CPU path',
                    f'{err:.1e}')
    return ok


def main():
    mf = rhf_df(WATER_ORIGIN, '6-31g')
    no = mf.mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    B = DFIntegrals.from_scf(mf.mol, mf).B_aa
    F = ee_ri_fold.RIFactors.from_full(eps, B, no)
    ok = True
    print('1. spin-orbital arbiter')
    ok &= check_arbiter(eps, B, no, F)
    print('2-3. unit scaling and the scaled MP2 energy')
    ok &= check_unit_scaling(F, mf)
    print('4. simulated MPI ranks')
    ok &= check_ranks(F)
    print('5. driver')
    ok &= check_driver(mf, F)
    print('6. GPU kernel')
    ok &= check_gpu(F)
    print('ALL OK' if ok else 'SOME CHECKS FAILED')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
