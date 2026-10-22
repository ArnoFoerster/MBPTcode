"""ISDF (separable-RI) matrix-free BSE Davidson against the dense Casida solve.

Two DIFFERENT errors live in this comparison and keeping them apart is the whole
point of the test:

  1. ALGEBRA. The separable factorization defines its own three-index factor
     B[A,p,q] = sum_k X[k,p] X[k,q] D[k,A] exactly. Hand THAT B to
     build_casida_matrices and the dense CasidaSolver and you get precisely the
     operator the Fock-like matrix-vector products claim to apply, so the two
     must agree to Davidson's convergence threshold -- the same bar the DF
     matrix-free path already meets. Nothing about the grid enters this check.

  2. FACTORIZATION. Against pyscf's own DF the ISDF route additionally carries
     the separable fit's error. That is a property of the grid, not of the
     solver, so it is REPORTED separately instead of being folded into the
     tolerance of check 1. It behaves like one: benzene/cc-pVDZ moves from
     74.6 meV on a 148-point grid to 6.5 meV at 244 points (--dense-grid),
     while check 1 sits at 1e-11 throughout. Water is at 2.7 meV on the
     148-point grid. The published Duchemin-Blase tables are cc-pVTZ only, so a
     cc-pVDZ run falls back on one L-BFGS-B descent per element, which
     separable_ri.py already documents as landing above the published grids'
     accuracy.

Both silent traps are made loud here as well:

  * The auxiliary GAUGE. pyscf's cderi is L^-1-whitened, a separable RI fits
    with the symmetric V^-1/2, and the two differ by an orthogonal rotation of
    the auxiliary index. The negative control pairs the ISDF factors with the
    cderi-gauge W_aux and reports how far that moves the spectrum -- far beyond
    the factorization error, with nothing anywhere looking wrong.
  * Comparing gauge-carrying quantities ELEMENTWISE is meaningless. The two
    W_aux are the same operator in rotated bases: their eigenvalues agree to the
    fit error while their entries do not agree at all.

Run: python tests/test_davidson_isdf_bse.py [molecule ...] [--dense-grid]
     (default: both molecules, default grid)
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.davidson import (solve_casida_davidson,
                                                         isdf_bse_factors,
                                                         isdf_df_coefficients)
from src.SingleReference.LinearResponse.block_actions import isdf_block_action
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.Base.constants import (BSE_DAVIDSON_CONV_TOL, HARTREE_TO_EV,
                                HARTREE_TO_MEV)
from src.Base.numerics import numerics_of
from src.Base.separable_ri import resolve_isdf_grid
from src.SingleReference.GW.space_time import unrefused_separable_factors

GEOMETRIES = {
    'water': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
    'benzene': """
        C  0.000  1.396  0.000
        C  1.209  0.698  0.000
        C  1.209 -0.698  0.000
        C  0.000 -1.396  0.000
        C -1.209 -0.698  0.000
        C -1.209  0.698  0.000
        H  0.000  2.479  0.000
        H  2.147  1.240  0.000
        H  2.147 -1.240  0.000
        H  0.000 -2.479  0.000
        H -2.147 -1.240  0.000
        H -2.147  1.240  0.000
    """,
}

BASIS = 'cc-pvdz'
AUXBASIS = 'cc-pvdz-ri'
#: Lebedev sub-shell replicas per atom: the validated G1 level, and G3 for the
#: --dense-grid run that shows the factorization error following the grid.
COUNTS = resolve_isdf_grid('G1', BASIS)[0]
DENSE_COUNTS = resolve_isdf_grid('G3', BASIS)[0]
#: Benzene at NROOTS=5 is also the regression pin for the Davidson GUESS. Its
#: unit-vector guess of one vector per requested root skips roots whenever the
#: orbital-energy differences come in degenerate sets: benzene RPA at five
#: roots would report converged=True on all five while missing four of the true
#: lowest eight. That is not a property of either block action -- the DF and
#: ISDF actions agree to 1.6e-16 on this system and return the SAME skipped
#: set. `davidson.GUESS_FACTOR` avoids it at the source, and comparing exactly
#: the roots asked for holds it there.
NROOTS = 5
#: Every solve asks for the production residual. At the solver's default 1e-5
#: the 95-pair water RPA solve on the shipped grid grows its trial
#: space to 93 of the 95 pair directions, where round-off takes the positive
#: definiteness of the projected A-B block (its Cholesky factor fails).
CONV_TOL = BSE_DAVIDSON_CONV_TOL
SOLVER_NUMERICS = numerics_of({'nroots': NROOTS, 'bse_conv_tol': CONV_TOL})
#: An eigenvalue converges quadratically in the residual; 1e-8 is the level the
#: DF path already meets.
ALGEBRA_TOL = 1e-8


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def run(name, counts):
    print(f"\n=== {name} / {BASIS}, ISDF counts {counts} ===")
    mol = gto.M(atom=GEOMETRIES[name], basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.kernel()
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    nvirt = len(eps) - nocc

    t0 = time.perf_counter()
    # benzene's virtual 86 is missed on G1 and the factors would be refused;
    # this script gates the solver against the dense eigensolve, not the fit
    X_mo, D, W_isdf = isdf_bse_factors(
        mf, mol, nocc, factors=unrefused_separable_factors(
            mf, mol, auxbasis=AUXBASIS, counts=counts))
    t_fac = time.perf_counter() - t0
    print(f"nao={mol.nao_nr()}  nocc={nocc}  nvirt={nvirt}  n_pair={nocc * nvirt}  "
          f"naux={D.shape[1]}  grid points M={D.shape[0]}   (factorization {t_fac:.1f}s)")

    B_isdf = isdf_df_coefficients(X_mo, D)
    lr_isdf = LinearResponseSolver(eps, coeff_df=B_isdf, spin_mode='restricted', eta=1e-3)
    coeff_df = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr_df = LinearResponseSolver(eps, coeff_df=coeff_df, spin_mode='restricted', eta=1e-3)
    W_df = lr_df.static_screening_aux(nocc)

    ok = True

    # 0. The static W the whole thing runs on is built in imaginary time, so
    #    that nothing quartic survives at production sizes; the DF sum over
    #    occupied-virtual pairs is its reference, in the same gauge.
    _, _, W_ref = isdf_bse_factors(mf, mol, nocc, factors=(X_mo, D),
                                   screening='df')
    d_w = np.abs(W_isdf - W_ref).max() / np.abs(W_ref).max()
    ok &= check(d_w < 1e-6, 'static W from imaginary time == from the DF pair sum',
                f'rel {d_w:.1e}')

    # 1. ALGEBRA: same operator, dense against Fock-like matrix-free.
    for mode, w_isdf in (('RPA', None), ('TDHF', None), ('BSE', W_isdf)):
        lBSE = mode != 'RPA'
        A, Bm = lr_isdf.build_casida_matrices(nocc, lBSE=lBSE, W_aux=w_isdf)
        omega_dense = np.sort(CasidaSolver(A, Bm).solve()[0])[:NROOTS]
        kw = {'W_aux': w_isdf} if mode == 'BSE' else {}
        t0 = time.perf_counter()
        omega_mf = np.sort(solve_casida_davidson(lr_isdf, nocc, numerics=SOLVER_NUMERICS,
                                                 polarizability=mode,
                                                 isdf_factors=(X_mo, D),
                                                 **kw)[0])
        t_mf = time.perf_counter() - t0
        d = np.max(np.abs(omega_dense - omega_mf))
        ok &= check(d < ALGEBRA_TOL,
                    f'{mode:4s} ISDF matrix-free == dense Casida in the ISDF gauge',
                    f'max |dw| = {d:.2e} Ha, davidson {t_mf:.2f}s')
        if mode == 'BSE':
            omega_bse_isdf = omega_mf

    # The row tiling is what keeps the (M, M) Hadamard product from ever being
    # formed at production sizes, so it had better not change the answer. One
    # row per block against one block for everything.
    act_1, _ = isdf_block_action(lr_isdf, nocc, True, W_isdf, (X_mo, D))
    act_n, _ = isdf_block_action(lr_isdf, nocc, True, W_isdf, (X_mo, D),
                                 tile_memory_gb=1e-9)
    zt = np.random.default_rng(0).standard_normal((3, nocc, nvirt))
    a1, b1 = act_1(zt)
    an, bn = act_n(zt)
    d_t = max(np.abs(a1 - an).max() / np.abs(a1).max(),
              np.abs(b1 - bn).max() / np.abs(b1).max())
    ok &= check(d_t < 1e-12, 'block action is invariant to the row tiling',
                f'rel {d_t:.1e}')

    # 2. FACTORIZATION error, reported and not folded into the tolerance above.
    A_df, B_df = lr_df.build_casida_matrices(nocc, lBSE=True, W_aux=W_df)
    omega_bse_df = np.sort(CasidaSolver(A_df, B_df).solve()[0])[:NROOTS]
    err = (omega_bse_isdf - omega_bse_df) * HARTREE_TO_MEV
    print(f"  BSE excitation energies (eV)")
    print(f"    pyscf DF, dense : {np.array2string(omega_bse_df * HARTREE_TO_EV, precision=5)}")
    print(f"    ISDF, matrix-free: {np.array2string(omega_bse_isdf * HARTREE_TO_EV, precision=5)}")
    print(f"    ISDF factorization error: max |dw| = {np.abs(err).max():.2f} meV, "
          f"mean {err.mean():+.2f} meV")

    # The gauge, both ways round.
    ev_isdf = np.sort(np.linalg.eigvalsh(W_isdf))
    ev_df = np.sort(np.linalg.eigvalsh(W_df))
    d_spec = np.abs(ev_isdf - ev_df).max() / np.abs(ev_df).max()
    d_elem = np.abs(W_isdf - W_df).max() / np.abs(W_df).max()
    ok &= check(d_spec < 1e-2, 'W_aux SPECTRA agree (gauge-invariant)',
                f'rel {d_spec:.1e}')
    ok &= check(d_elem > 1e-1, 'W_aux ENTRIES do not, and must not be compared',
                f'rel {d_elem:.1e}')

    # Negative control: the ISDF factors with the cderi-gauge W_aux.
    omega_mixed = np.sort(solve_casida_davidson(lr_isdf, nocc, numerics=SOLVER_NUMERICS,
                                                polarizability='BSE', W_aux=W_df,
                                                isdf_factors=(X_mo, D))[0])
    shift = np.abs(omega_mixed - omega_bse_isdf).max() * HARTREE_TO_MEV
    ok &= check(shift > 3 * np.abs(err).max(),
                'mixing gauges corrupts the spectrum far beyond the fit error',
                f'{shift:.0f} meV vs {np.abs(err).max():.2f} meV')

    return ok


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    args = sys.argv[1:]
    counts = DENSE_COUNTS if '--dense-grid' in args else COUNTS
    names = [a for a in args if not a.startswith('--')] or list(GEOMETRIES)
    all_ok = True
    for n in names:
        all_ok &= run(n, counts)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
