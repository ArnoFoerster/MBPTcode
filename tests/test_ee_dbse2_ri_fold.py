"""The RI operator of eq 71's doubles (ee_dbse2_ri_fold.RIFoldDBSE2) against the pieces
route (ee_dbse2_pieces.build_pieces_dbse2), water 6-31G (RHF, DF factors). The GW
energies are the deterministic stand-in of test_ee_dbse2_fold (gw_like: occupied +0.03
Ha, virtual -0.05 Ha on the HF energies); the W lines are screened at the HF energies.

Checks:
  1. the operator equals ee_fold.folded_operator on the pieces, both kernels, both
     singles blocks, both channels: the product at w = None, 0.3 and 0.5 Ha, the signed
     norm, the derivative image, fold_images at a node, the diagonal, doubles_dot at two
     frequencies, each to 1e-12 relative; d_min is min D exactly; one occupied per tile
     gives the same products. Eq 71's triplet has no doubles, so its doubles rows
     compare zero with zero at 1e-14 absolute. Refusals: energies in eV, bare and
     dressed factors at different energies, an unknown kernel, singles or device, each
     ValueError naming the input.
  2. solve_folded on the operator's pieces() against the explicit pieces, solvers
     'newton', 'davidson' and 'block' at tol_omega 1e-9, two roots per channel, both
     kernels: roots to 1e-8 Ha, T1 to 1e-8, every root converged (eq 71's triplet is M
     alone on both routes).
  3. two and three simulated ranks (mpi_grid.run_simulated, one occupied per tile)
     equal the serial run bit for bit: products and signed norms at two frequencies,
     the diagonal, two Newton roots driven through ee_ri_fold.drive.
  6. form='laplace' against form='exact' (both kernels, both channels): products and
     signed norms at 0.3 and 0.5 Ha, fold_images at three nodes with a derivative node
     between, doubles_dot at two frequencies and at one, each to 1e-8 relative (the
     quadrature's 1e-9 target; the achieved error printed); a node at min D + 0.05 Ha
     falls back to the exact tile (equal to 1e-12, laplace_fallbacks counted); two
     Newton roots within 1e-8 Ha of the exact form's; an unknown form raises
     ValueError. Checks 4 and 5 are the device path's (later tasks).
Run: python tests/test_ee_dbse2_ri_fold.py
"""
import os
import sys
import warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.ADC.eeADC import ee_dbse2_pieces, ee_dbse2_ri_fold, ee_fold
from src.SingleReference.ADC.eeADC.ee_ri_fold import RIFactors, drive

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'
HARTREE_TO_EV = 27.211386245988
KERNELS = ('eq71', 'printed')
SINGLES = ('eq62', 'eq62+eq59')
SPINS = ('singlet', 'triplet')


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail, flush=True)
    return bool(ok)


def water(basis='6-31g'):
    """(eps, B, no): RHF water with DF factors B[Q, p, q] in the MO basis."""
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    return eps, B, mol.nelectron // 2


def gw_like(eps, no):
    """A deterministic stand-in for the G0W0 energies: the HF gap closed by 0.08 Ha."""
    e = np.array(eps, float)
    e[:no] += 0.03
    e[no:] -= 0.05
    return e


def rel(a, b):
    """max |a - b| over max |b|, the GPU check's measure."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def close(a, b, zero):
    """1e-12 relative, or 1e-14 absolute where the reference is identically zero."""
    if zero:
        return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max()) < 1e-14
    return rel(a, b) < 1e-12


def check_operator(eps, B, no):
    """Check 1."""
    ok = True
    eps_gw = gw_like(eps, no)
    n = no * (eps.size - no)
    U = np.random.default_rng(7).standard_normal((n, 3))
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    for kernel in KERNELS:
        for singles in SINGLES:
            P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel=kernel,
                                                   singles=singles)
            for spin in SPINS:
                tag = f'{kernel} {singles} {spin}'
                # eq 71 has no triplet part: its doubles rows compare zero with zero
                zero = kernel == 'eq71' and spin == 'triplet'
                op = make(eps_gw, eps, B, no, spin, kernel=kernel, singles=singles)
                Bm03 = None
                for omega in (None, 0.3, 0.5):
                    mv, dmv, diag_s, embed, _ = ee_fold.folded_operator(P, omega, spin)
                    A = np.column_stack([mv(U[:, j]) for j in range(3)])
                    Bm, nrm = op.apply(U, None if omega is None else np.full(3, omega))
                    ok &= check(rel(Bm, A) < 1e-12, f'{tag} w={omega}: product',
                                f'{rel(Bm, A):.1e}')
                    if omega is None:
                        continue
                    if omega == 0.3:
                        Bm03 = Bm
                    ref = np.array([dmv(U[:, j]) for j in range(3)])
                    ok &= check(close(nrm, ref, zero), f'{tag} w={omega}: signed norm',
                                f'{np.abs(nrm - ref).max():.1e} abs')
                    F1 = op.fold_images(U, [omega], derivative=[True])[1][0]
                    dref = np.column_stack([dmv(U[:, j], vector=True)
                                            for j in range(3)])
                    ok &= check(close(F1, dref, zero),
                                f'{tag} w={omega}: derivative image',
                                f'{np.abs(F1 - dref).max():.1e} abs')
                    MU, F = op.fold_images(U, [omega, 0.5 * omega],
                                           derivative=[False, False])
                    ok &= check(rel(MU + F[0], A) < 1e-12,
                                f'{tag} w={omega}: fold_images at a node is apply',
                                f'{rel(MU + F[0], A):.1e}')
                # the pieces route hands the solver the aaaa block's diagonal at the
                # channel's representatives (a preconditioner); the operator's diag is
                # the channel matrix's own diagonal, as RIFoldADC2's
                mv, _, _, embed, _ = ee_fold.folded_operator(P, None, spin)
                dense = np.column_stack([mv(e) for e in np.eye(n)])
                ok &= check(rel(op.diag, np.diag(dense)) < 1e-12,
                            f'{tag}: diagonal is the dense channel matrix diagonal',
                            f'{rel(op.diag, np.diag(dense)):.1e}')
                Ya = ee_fold._doubles_image(P, U[:, 0], 0.3, embed)
                Yb = ee_fold._doubles_image(P, U[:, 1], 0.5, embed)
                d = op.doubles_dot(U[:, 0], 0.3, U[:, 1], 0.5)
                dref = ee_fold._doubles_dot(P, Ya, Yb)
                ok &= check(close(d, dref, zero),
                            f'{tag}: doubles_dot at two frequencies',
                            f'{d:.6e} against {dref:.6e}')
                ok &= check(op.pieces()['d_min'] == float(P['D'].min()),
                            f'{tag}: d_min is min D exactly')
                tiny = make(eps_gw, eps, B, no, spin, kernel=kernel, singles=singles,
                            tile_gb=1e-9)
                e = rel(tiny.apply(U, np.full(3, 0.3))[0], Bm03)
                ok &= check(tiny.kt == 1 and e < 1e-12,
                            f'{tag}: one occupied per tile gives the same products',
                            f'kt {tiny.kt}, {e:.1e}')
    return ok


def check_refusals(eps, B, no):
    """Refusals, each a ValueError naming the input."""
    ok = True
    eps_gw = gw_like(eps, no)
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    bare_hf = RIFactors.from_full(eps, B, no)
    Bt = ee_dbse2_pieces.dressed_factors(eps, B, no)[1]
    dressed_gw = RIFactors.from_full(eps_gw, Bt, no)
    cases = (
        ('eV', lambda: make(eps_gw * HARTREE_TO_EV, eps * HARTREE_TO_EV, B, no,
                            'singlet')),
        ('energies',
         lambda: ee_dbse2_ri_fold.RIFoldDBSE2(bare_hf, dressed_gw, 'singlet')),
        ('kernel', lambda: make(eps_gw, eps, B, no, 'singlet', kernel='x')),
        ('singles', lambda: make(eps_gw, eps, B, no, 'singlet', singles='x')),
        ('device', lambda: make(eps_gw, eps, B, no, 'singlet', device='tpu')),
    )
    for word, build in cases:
        try:
            build()
            ok &= check(False, f'{word}: raises ValueError')
        except ValueError as err:
            ok &= check(word in str(err), f'{word}: raises ValueError naming it',
                        str(err)[:70])
    return ok


def check_solves(eps, B, no):
    """Check 2."""
    ok = True
    eps_gw = gw_like(eps, no)
    tight = dict(dense_limit=0, tol_omega=1e-9, tol_residual=1e-8)
    for kernel in KERNELS:
        P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel=kernel)
        for spin in SPINS:
            op = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo(eps_gw, eps, B, no, spin,
                                                      kernel=kernel)
            for solver in ('newton', 'davidson', 'block'):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    ref = ee_fold.solve_folded(P, 2, spin=spin, solver=solver, **tight)
                    r = ee_fold.solve_folded(op.pieces(), 2, spin=spin, solver=solver,
                                             **tight)
                same_shape = r.omega.shape == ref.omega.shape
                d = float(np.abs(r.omega - ref.omega).max()) if same_shape else np.inf
                dt = float(np.abs(r.t1 - ref.t1).max()) if same_shape else np.inf
                ok &= check(d < 1e-8 and dt < 1e-8 and r.converged.all(),
                            f'{kernel} {spin} {solver}: roots and T1 on the operator '
                            'equal the pieces', f'{d:.1e} Ha, {dt:.1e} on T1')
    return ok


def check_ranks(eps, B, no):
    """Check 3."""
    ok = True
    eps_gw = gw_like(eps, no)
    Bt = ee_dbse2_pieces.dressed_factors(eps, B, no)[1]
    bare = RIFactors.from_full(eps_gw, B, no)
    dressed = RIFactors.from_full(eps_gw, Bt, no)
    u = np.random.default_rng(5).standard_normal((no * (eps.size - no), 2))

    def solve(o):
        r = ee_fold.solve_folded(o.pieces(), 2, spin='singlet', dense_limit=0,
                                 solver='newton')
        return dict(omega=r.omega, t1=r.t1)

    def rank(comm):
        op = ee_dbse2_ri_fold.RIFoldDBSE2(bare, dressed, 'singlet', kernel='printed',
                                          tile_gb=1e-9, threads=1, comm=comm)
        prod = op.apply(u, [0.3, 0.5])
        return drive(op, solve), op.diag, prod

    ser = rank(None)
    for size in (2, 3):
        outs = run_simulated(rank, size)
        same = all(np.array_equal(o['omega'], ser[0]['omega'])
                   and np.array_equal(o['t1'], ser[0]['t1'])
                   and np.array_equal(d, ser[1])
                   and np.array_equal(p[0], ser[2][0])
                   and np.array_equal(p[1], ser[2][1])
                   for o, d, p in outs)
        ok &= check(same, f'{size} simulated ranks equal the serial run bit for bit')
    return ok


def check_laplace(eps, B, no):
    """Check 6."""
    ok = True
    eps_gw = gw_like(eps, no)
    n = no * (eps.size - no)
    U = np.random.default_rng(11).standard_normal((n, 3))
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    tight = dict(dense_limit=0, tol_omega=1e-9, tol_residual=1e-8)

    def within(a, b, zero):
        """(ok, detail): 1e-8 relative, or 1e-12 absolute on a zero reference."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if zero:
            e = float(np.abs(a - b).max())
            return e < 1e-12, f'{e:.1e} abs'
        return rel(a, b) < 1e-8, f'{rel(a, b):.1e}'

    for kernel in KERNELS:
        for spin in SPINS:
            tag = f'{kernel} {spin} laplace'
            zero = kernel == 'eq71' and spin == 'triplet'
            ope = make(eps_gw, eps, B, no, spin, kernel=kernel)
            opl = make(eps_gw, eps, B, no, spin, kernel=kernel, form='laplace')
            for omega in (0.3, 0.5):
                om = np.full(3, omega)
                Ae, ne = ope.apply(U, om)
                Al, nl = opl.apply(U, om)
                good, detail = within(Al, Ae, False)
                ok &= check(good, f'{tag} w={omega}: product', detail)
                good, detail = within(nl, ne, zero)
                ok &= check(good, f'{tag} w={omega}: signed norm', detail)
            nodes, deriv = [0.3, 0.4, 0.5], [False, True, False]
            Me, Fe = ope.fold_images(U, nodes, derivative=deriv)
            Ml, Fl = opl.fold_images(U, nodes, derivative=deriv)
            good = rel(Ml, Me) < 1e-12
            details = []
            for q in range(3):
                g, d = within(Fl[q], Fe[q], zero)
                good &= g
                details.append(d)
            ok &= check(good, f'{tag}: fold_images, a derivative node between two '
                        'plain ones', ', '.join(details))
            for wr, ws in ((0.3, 0.5), (0.3, 0.3)):
                de = ope.doubles_dot(U[:, 0], wr, U[:, 1], ws)
                dl = opl.doubles_dot(U[:, 0], wr, U[:, 1], ws)
                good, detail = within(dl, de, zero)
                ok &= check(good, f'{tag}: doubles_dot at ({wr}, {ws})', detail)
            # the rows above ran on grids (no fallback); the three columns at a node
            # inside the continuum are the only fallbacks
            inside = opl.d_min + 0.05
            Ae = ope.apply(U, np.full(3, inside))[0]
            Al = opl.apply(U, np.full(3, inside))[0]
            ok &= check(rel(Al, Ae) < 1e-12 and opl.laplace_fallbacks == 3,
                        f'{tag}: a node inside the doubles continuum falls back to the '
                        'exact tile, the rows before it did not',
                        f'{rel(Al, Ae):.1e}, {opl.laplace_fallbacks} fallbacks')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                re_ = ee_fold.solve_folded(ope.pieces(), 2, spin=spin, solver='newton',
                                           **tight)
                rl = ee_fold.solve_folded(opl.pieces(), 2, spin=spin, solver='newton',
                                          **tight)
            d = (float(np.abs(rl.omega - re_.omega).max())
                 if rl.omega.shape == re_.omega.shape else np.inf)
            ok &= check(d < 1e-8 and rl.converged.all(),
                        f'{tag}: two Newton roots equal the exact form', f'{d:.1e} Ha')
    try:
        make(eps_gw, eps, B, no, 'singlet', form='x')
        ok &= check(False, 'form: raises ValueError')
    except ValueError as err:
        ok &= check('form' in str(err), 'form: raises ValueError naming it',
                    str(err)[:70])
    return ok


def main():
    eps, B, no = water('6-31g')
    ok = True
    print('check 1: the operator against the pieces route (6-31G)')
    ok &= check_operator(eps, B, no)
    print('refusals')
    ok &= check_refusals(eps, B, no)
    print('check 2: the three solvers on the operator against the pieces')
    ok &= check_solves(eps, B, no)
    print('check 3: simulated ranks')
    ok &= check_ranks(eps, B, no)
    print('check 6: the Laplace form against the exact form')
    ok &= check_laplace(eps, B, no)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
