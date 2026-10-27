"""The RI operator of the dBSE2 doubles (ee_dbse2_ri_fold.RIFoldDBSE2) against the
pieces route (ee_dbse2_pieces.build_pieces_dbse2), water 6-31G (RHF, DF factors), on
every set of SETS: eq 71's doubles alone, eq 61's alone, both. The GW energies are the
deterministic stand-in of test_ee_dbse2_fold (gw_like: occupied +0.03 Ha, virtual -0.05
Ha on the HF energies); the W lines and the modes are screened at the HF energies.

Checks:
  1. the operator equals ee_fold.folded_operator on the pieces, every set, both
     kernels, both singles blocks, both channels: the product at w = None, 0.3 and 0.5
     Ha, the signed norm, the derivative image, fold_images at a node, the diagonal,
     doubles_dot at two frequencies, each to 1e-12 relative; d_min is min D exactly (the
     lower of the sets' minima on the two-set level); the product at d_min + 0.05 Ha, a
     node inside the lower set's continuum, against the pieces; one occupied per tile
     gives the same products. Eq 71's triplet has no doubles, so on sets=('eq71',) its
     doubles rows compare zero with zero at 1e-14 absolute. Refusals: energies in eV,
     bare and dressed factors at different energies, an unknown kernel, singles or
     device, sets=('eq71',) without dressed factors, a mode set without modes, modes
     whose rho has naux + 1 rows (naming naux), sets outside SETS, each ValueError
     naming the input.
  2. solve_folded on the operator's pieces() against the explicit pieces, every set,
     solvers 'newton', 'davidson' and 'block' at tol_omega 1e-9, two roots per channel,
     both kernels and both forms: roots to 1e-8 Ha, T1 to 1e-8, every root converged
     (eq 71's triplet is M alone on both routes), and fold_step from the newton root's
     vector at 0.9 of its frequency and 0.05 Ha above, static fixed at the 0.9 value
     on every call, gives the same step, zeta and delta on both routes to 1e-10, delta
     moving by more than DELTA_MOVE between the two (exactly zero without doubles).
  3. two and three simulated ranks (mpi_grid.run_simulated, one occupied per tile)
     equal the serial run bit for bit on sets=('eq61', 'eq71'), kernel printed, both
     spins: products and signed norms at two frequencies, the diagonal, two Newton
     roots driven through ee_ri_fold.drive.
  7. the modes: from_mo on sets=('eq61',) equals the constructor given gw_modes_df's
     modes at the HF energies to 1e-12 on the product at 0.3 Ha and differs from the
     same with modes at the GW energies by more than 1e-6 relative (review focus 1);
     form='laplace' on sets=('eq61', 'eq71') equals form='exact' to 1e-8 relative on
     the product at 0.3 Ha with no fallback (both sets on the quadrature).
  10. sets=('eq66',), level gw through the operator (spec check 6's second clause):
     from_mo(eps, eps, B, no, spin, sets=('eq66',), screening=) against
     ee_fold.folded_operator on build_pieces_gw(eps, B, no, screening, df=True), both
     screenings, both spins: the product at None, 0.3 and 0.5 Ha and the signed norm to
     1e-12 relative, the diagonal against the dense channel matrix to 1e-12;
     solve_folded ('davidson', two roots, dense_limit 0, tol_omega 1e-9) on the
     operator's pieces against the W pieces (df=False): roots to 1e-8 Ha, T1 to 1e-8,
     every root converged (the roots test_ee_gw_fold.py pins to QuAcK's supermatrix);
     pieces()['level'] is 'gw'; static_term_pq's blocks assembled by einsum equal
     static_term_df to 1e-14; refusals: ('eq66',) with dressed factors, with
     singles='eq62+eq59', and ('eq61', 'eq66'), each ValueError naming the argument.
  6. form='laplace' against form='exact' (both kernels, both channels): products and
     signed norms at 0.3 and 0.5 Ha, fold_images at three nodes with a derivative node
     between, doubles_dot at two frequencies and at one, each to 1e-8 relative (the
     quadrature's 1e-9 target; the achieved error printed); doubles_dot at two
     frequencies 1e-2, 1e-5 and 1e-9 Ha apart (the duplicate check's regime); a node at
     min D + 0.05 Ha falls back to the exact tile (equal to 1e-12, laplace_fallbacks
     counted), also between two grid nodes in one fold_images call with the derivative
     on it; one grid serves the image and its derivative at one frequency; fold_images
     counts its M columns in matvecs; two Newton roots within 1e-8 Ha of the exact
     form's; an unknown form raises ValueError; the default form resolves to 'laplace'
     on one rank with every B_vv whole and to 'exact' with one UserWarning naming the
     reason otherwise (a chunked bare B_vv; two simulated ranks).
  9. the tile budget (_tile_bytes) bounds the peak traced by tracemalloc over one
     _tile_core call at two nodes and one _tile_overlap of two vectors, the Laplace
     budget (_laplace_bytes) the peak of one Laplace node, the mode budget
     (_mode_bytes) the peak of one _mode_tile at two nodes and of one _mode_overlap of
     two vectors (eq 61's cross and eq 66's square), the mode set's Laplace budget
     (_mode_laplace_bytes) the peak of one of its nodes (eq 61, image and derivative),
     each within a factor two; the device class's need carries the budgets and the
     residents of the sets in use, both budgets on sets=('eq61', 'eq71'), and the mode
     set's Laplace residents and node on that level's Laplace form, the node traced as
     eq 61's cross and as eq 66's square.
  4. linear_response.device_memory_check, the static kernel's pre-flight lifted for the
     device path: need above free raises ValueError naming the label and the GiB, need
     within free returns None.
  5. the device class on the numpy stand-in (_DoublesDBSE2GPU(op, xp=np)) against the
     host path, both forms, both kernels, both channels on sets=('eq71',), and on the
     mode sets (('eq61',) with both forms, ('eq61', 'eq71') with both kernels and forms,
     ('eq66',) with both screenings and forms): the product and the signed norm at 0.3
     Ha, fold_images at three
     nodes with a derivative node between, doubles_dot of two vectors and of one, each
     to 1e-12 relative; the Laplace form's fallback node through the device tile and,
     on the mode cells, through the device mode tile; two
     simulated ranks with the stand-in, one occupied per tile, equal the serial run bit
     for bit, on ('eq71',) and on ('eq61', 'eq71'); a device reporting one byte free is
     refused before the upload (ValueError naming GiB), both forms and on ('eq61',
     'eq71') with the label naming the modes; device='gpu' without CuPy raises
     ImportError naming CuPy; the Laplace form on two ranks or without the whole B~_vv
     raises NotImplementedError naming one rank, and on a mode set with the bare B_vv
     in auxiliary chunks one naming the bare B_vv.
  14. the mode sets' Laplace form (Task 9): form='laplace' against form='exact' on
     ('eq61',), ('eq61', 'eq71') with both kernels and ('eq66',) with both screenings,
     both channels: products and signed norms at 0.3 and 0.5 Ha, fold_images at three
     nodes with a derivative node between, doubles_dot at two frequencies and at one
     and at 1e-2, 1e-5 and 1e-9 Ha apart, each to 1e-8 relative, the overlap of two
     vectors against the largest of it and the two self-overlaps, since the signed
     overlap of two random vectors cancels (both errors printed); two Davidson roots
     within 1e-8 Ha; a node at the mode set's minimum + 0.05 Ha falls back to the mode
     tile (equal to 1e-12, the fallbacks counted per set); the construction on a mode
     set alone succeeds.
Run: python tests/test_ee_dbse2_ri_fold.py
"""
import os
import sys
import tracemalloc
import types
import warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.pyscf_interface import DFIntegrals, get_orbital_energies
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.ADC.eeADC import (ee_dbse2_pieces, ee_dbse2_ri_fold, ee_fold,
                                           ee_gw_pieces)
from src.SingleReference.ADC.eeADC.ee_ri_fold import RIFactors, aux_chunks, drive
from src.SingleReference.LinearResponse import linear_response
from tests.pins import same_bar

# the run choice that puts the device paths on the GPU
ON_GPU = Numerics(layout={'device': 'gpu'})
WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'
HARTREE_TO_EV = 27.211386245988
KERNELS = ('eq71', 'printed')
SINGLES = ('eq62', 'eq62+eq59')
SPINS = ('singlet', 'triplet')
# the sets of the pieces route, the operator's SETS without eq 66's level
SETS = (('eq71',), ('eq61',), ('eq61', 'eq71'))
# Ha: check 2's control, |delta(omega0) - delta(omega0 + 0.05)| at static=omega0, one
# decade below the smallest row with doubles, 7.59e-5 on eq71 eq71 singlet
DELTA_MOVE = 7.5e-6


def zero_doubles(kernel, spin, sets):
    """True where the channel has no doubles: eq 71's triplet on eq 71's set alone."""
    return kernel == 'eq71' and spin == 'triplet' and sets == ('eq71',)


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


def h2(basis='sto-3g'):
    """(eps, B, no) of RHF H2: nocc = nvirt = 1, the smallest shapes of every layout."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    return eps, B, 1


def gw_like(eps, no):
    """A deterministic stand-in for the G0W0 energies: the HF gap closed by 0.08 Ha,
    with an orbital-dependent part, so that a route confusing per-orbital GW energies
    with a shifted HF set cannot pass."""
    e = np.array(eps, float)
    e[:no] += 0.03 + 0.002 * np.arange(no)
    e[no:] -= 0.05 - 0.001 * np.arange(e.size - no)
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


def check_operator(eps, B, no, sets):
    """Check 1 on one set."""
    ok = True
    eps_gw = gw_like(eps, no)
    n = no * (eps.size - no)
    U = np.random.default_rng(7).standard_normal((n, 3))
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    for kernel in KERNELS:
        for singles in SINGLES:
            P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel=kernel,
                                                   singles=singles, sets=sets)
            for spin in SPINS:
                tag = f"{'+'.join(sets)} {kernel} {singles} {spin}"
                # eq 71 has no triplet part: on its set alone the doubles rows compare
                # zero with zero
                zero = zero_doubles(kernel, spin, sets)
                op = make(eps_gw, eps, B, no, spin, kernel=kernel, singles=singles,
                          sets=sets, form='exact')
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
                d_min, d_ref = op.pieces()['d_min'], float(P['D'].min())
                # two roundings of one difference of orbital energies
                ok &= check(abs(d_min - d_ref) <= same_bar(d_min, d_ref),
                            f'{tag}: d_min is min D to PIN_SAME_ULPS ulp',
                            f"{op.d_min:.6f} Ha, sets' minima "
                            + ', '.join(f'{P["D"][sl].min():.6f}' for _, _, sl
                                        in P['segments']['a'][::2]))
                # a node inside the lower set's continuum: the fold is a plain divide
                # there on both routes
                inside = op.d_min + 0.05
                mv = ee_fold.folded_operator(P, inside, spin)[0]
                A = np.column_stack([mv(U[:, j]) for j in range(3)])
                Bm = op.apply(U, np.full(3, inside))[0]
                ok &= check(rel(Bm, A) < 1e-12, f'{tag} w=d_min+0.05: product inside '
                            "the lower set's continuum", f'{rel(Bm, A):.1e}')
                tiny = make(eps_gw, eps, B, no, spin, kernel=kernel, singles=singles,
                            tile_gb=1e-9, sets=sets, form='exact')
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
    bare_gw = RIFactors.from_full(eps_gw, B, no)
    Bt = ee_dbse2_pieces.dressed_factors(eps, B, no)[1]
    dressed_gw = RIFactors.from_full(eps_gw, Bt, no)
    om, rho = ee_gw_pieces.gw_modes_df(eps, B, no, 'rpa')
    bad_rho = np.vstack([rho, rho[:1]])
    dressed_bad = RIFactors.from_full(eps_gw, np.vstack([Bt, Bt[:1]]), no)
    direct = ee_dbse2_ri_fold.RIFoldDBSE2
    cases = (
        ('eV', lambda: make(eps_gw * HARTREE_TO_EV, eps * HARTREE_TO_EV, B, no,
                            'singlet')),
        ('energies', lambda: direct(bare_hf, dressed_gw, 'singlet')),
        ('kernel', lambda: make(eps_gw, eps, B, no, 'singlet', kernel='x')),
        ('singles', lambda: make(eps_gw, eps, B, no, 'singlet', singles='x')),
        ('device', lambda: make(eps_gw, eps, B, no, 'singlet',
                                numerics=Numerics(layout={'device': 'tpu'}))),
        ('dressed', lambda: direct(bare_gw, None, 'singlet')),
        ('modes', lambda: direct(bare_gw, None, 'singlet', sets=('eq61',))),
        ('naux', lambda: direct(bare_gw, None, 'singlet', sets=('eq61',),
                                modes=(om, bad_rho))),
        ('sets', lambda: make(eps_gw, eps, B, no, 'singlet',
                              sets=('eq61', 'eq71', 'x'))),
        ('omega', lambda: direct(bare_gw, None, 'singlet', sets=('eq61',),
                                 modes=(om[:-1], rho))),
        ('auxiliary', lambda: direct(bare_gw, dressed_bad, 'singlet')),
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
    for sets in SETS:
        for kernel in KERNELS:
            P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, kernel=kernel,
                                                   sets=sets)
            for spin, form in ((s, f) for s in SPINS for f in ee_dbse2_ri_fold.FORMS):
                op = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo(eps_gw, eps, B, no, spin,
                                                          kernel=kernel, sets=sets,
                                                          form=form)
                for solver in ('newton', 'davidson', 'block'):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        ref = ee_fold.solve_folded(P, 2, spin=spin, solver=solver,
                                                   **tight)
                        r = ee_fold.solve_folded(op.pieces(), 2, spin=spin,
                                                 solver=solver, **tight)
                    same_shape = r.omega.shape == ref.omega.shape
                    d = (float(np.abs(r.omega - ref.omega).max()) if same_shape
                         else np.inf)
                    dt = float(np.abs(r.t1 - ref.t1).max()) if same_shape else np.inf
                    ok &= check(d < 1e-8 and dt < 1e-8 and r.converged.all(),
                                f"{'+'.join(sets)} {kernel} {form} {spin} {solver}: "
                                'roots and T1 on the operator equal the pieces',
                                f'{d:.1e} Ha, {dt:.1e} on T1, '
                                + ' '.join(f'{w:.6f}' for w in r.omega))
                    if solver == 'newton':
                        om0, x = 0.9 * float(ref.omega[0]), ref.y[:, 0]
                        # static=om0 throughout: a[2] - c[2] is the doubles'
                        # omega-dependence of x.A_eff(omega).x alone
                        a, b, c, e = (ee_fold.fold_step(Q, w, spin, x, static=om0)
                                      for w in (om0, om0 + 0.05)
                                      for Q in (P, op.pieces()))
                        d3 = [max(abs(a[k] - b[k]), abs(c[k] - e[k])) for k in range(3)]
                        moved = abs(a[2] - c[2])
                        if zero_doubles(kernel, spin, sets):
                            moves = moved == 0.0
                        else:
                            moves = moved > DELTA_MOVE
                        ok &= check(max(d3) < 1e-10 and moves,
                                    f"{'+'.join(sets)} {kernel} {form} {spin}: "
                                    'fold_step on the operator equals the pieces at '
                                    'omega0 and omega0 + 0.05 Ha, its delta moves '
                                    'with omega0 (zero without doubles)',
                                    f'{d3[0]:.1e} Ha, {d3[1]:.1e} on zeta, '
                                    f'{d3[2]:.1e} on delta, |a[2] - c[2]| '
                                    f'{moved:.3e} Ha per 0.05')
    return ok


def check_ranks(eps, B, no):
    """Check 3."""
    ok = True
    eps_gw = gw_like(eps, no)
    Bt = ee_dbse2_pieces.dressed_factors(eps, B, no)[1]
    bare = RIFactors.from_full(eps_gw, B, no)
    dressed = RIFactors.from_full(eps_gw, Bt, no)
    modes = ee_gw_pieces.gw_modes_df(eps, B, no, 'rpa')
    u = np.random.default_rng(5).standard_normal((no * (eps.size - no), 2))

    for spin in SPINS:
        def solve(o, spin=spin):
            r = ee_fold.solve_folded(o.pieces(), 2, spin=spin, dense_limit=0,
                                     solver='newton')
            return dict(omega=r.omega, t1=r.t1)

        def rank(comm, spin=spin, solve=solve):
            op = ee_dbse2_ri_fold.RIFoldDBSE2(bare, dressed, spin, kernel='printed',
                                              form='exact', sets=('eq61', 'eq71'),
                                              modes=modes, tile_gb=1e-9, threads=1,
                                              comm=comm)
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
            ok &= check(same, f'eq61+eq71 printed {spin}: {size} simulated ranks '
                        'equal the serial run bit for bit')
    # eq 66's level on two ranks: the mode tile's rank loop with its square flag
    bare_hf = RIFactors.from_full(eps, B, no)

    def rank66(comm):
        op = ee_dbse2_ri_fold.RIFoldDBSE2(bare_hf, None, 'singlet', sets=('eq66',),
                                          modes=modes, tile_gb=1e-9, threads=1,
                                          comm=comm, form='exact')
        return op.diag, op.apply(u, [0.3, 0.5])

    ser66 = rank66(None)
    same = all(np.array_equal(d, ser66[0]) and np.array_equal(p[0], ser66[1][0])
               and np.array_equal(p[1], ser66[1][1])
               for d, p in run_simulated(rank66, 2))
    ok &= check(same, 'eq66 singlet: 2 simulated ranks equal the serial run bit for '
                'bit')
    return ok


def check_modes(eps, B, no):
    """Check 7: the modes at the HF energies (review focus 1); the Laplace form on the
    two-set level."""
    ok = True
    eps_gw = gw_like(eps, no)
    n = no * (eps.size - no)
    U = np.random.default_rng(19).standard_normal((n, 3))
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    bare = RIFactors.from_full(eps_gw, B, no)
    for spin in SPINS:
        op = make(eps_gw, eps, B, no, spin, sets=('eq61',), form='exact')
        hf = ee_dbse2_ri_fold.RIFoldDBSE2(bare, None, spin, form='exact',
                                          sets=('eq61',),
                                          modes=ee_gw_pieces.gw_modes_df(eps, B, no,
                                                                         'rpa'))
        gw = ee_dbse2_ri_fold.RIFoldDBSE2(bare, None, spin, form='exact',
                                          sets=('eq61',),
                                          modes=ee_gw_pieces.gw_modes_df(eps_gw, B, no,
                                                                         'rpa'))
        A = op.apply(U, np.full(3, 0.3))[0]
        d_hf, d_gw = rel(hf.apply(U, np.full(3, 0.3))[0], A), rel(
            gw.apply(U, np.full(3, 0.3))[0], A)
        ok &= check(d_hf < 1e-12 and d_gw > 1e-6 and op.screening == 'rpa',
                    f'eq61 {spin}: from_mo builds the modes at the HF energies, not '
                    'the GW ones', f'HF modes {d_hf:.1e}, GW modes {d_gw:.1e}')
        ok &= check(op.nm == no * (eps.size - no)
                    and op.rho.shape == (B.shape[0], op.nm)
                    and op.om.shape == (op.nm,) and op.L2b.shape == (B.shape[0], n),
                    f'eq61 {spin}: nm, om, rho and L2b carry the modes and the bare '
                    'factor', f'nm {op.nm}, rho {op.rho.shape}, L2b {op.L2b.shape}')
        # the screening of the modes reaches both routes: 'tda' against the pieces
        # at 'tda', apart from the 'rpa' modes
        opt = make(eps_gw, eps, B, no, spin, form='exact', sets=('eq61',),
                   screening='tda')
        Pt = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, sets=('eq61',),
                                                screening='tda')
        mv = ee_fold.folded_operator(Pt, 0.3, spin)[0]
        At = np.column_stack([mv(U[:, j]) for j in range(3)])
        prod_t = opt.apply(U, np.full(3, 0.3))[0]
        ok &= check(rel(prod_t, At) < 1e-12 and rel(prod_t, A) > 1e-6
                    and opt.screening == 'tda',
                    f"eq61 {spin}: screening='tda' reaches the operator's modes and "
                    "the pieces' alike", f'pieces {rel(prod_t, At):.1e}, against rpa '
                    f'{rel(prod_t, A):.1e}')
        ope = make(eps_gw, eps, B, no, spin, sets=('eq61', 'eq71'), form='exact')
        opl = make(eps_gw, eps, B, no, spin, sets=('eq61', 'eq71'), form='laplace')
        Ae, ne = ope.apply(U, np.full(3, 0.3))
        Al, nl = opl.apply(U, np.full(3, 0.3))
        ok &= check(rel(Al, Ae) < 1e-8 and rel(nl, ne) < 1e-8
                    and opl.laplace_fallbacks == 0,
                    f'eq61+eq71 {spin}: the Laplace form of both sets equals the '
                    'exact form', f'{rel(Al, Ae):.1e}, norm '
                    f'{rel(nl, ne):.1e}, {opl.laplace_fallbacks} fallbacks')
    return ok


def check_minimal():
    """Check 11: nocc = nvirt = nm = 1 (H2/STO-3G), every set, the operator's product
    against the pieces route's at the smallest shapes of the flat layout and the
    mode tile; the Laplace form there too, where each set's span is one point."""
    ok = True
    eps, B, no = h2()
    eps_gw = gw_like(eps, no)
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    U = np.random.default_rng(11).standard_normal((1, 2))
    for sets in SETS + (('eq66',),):
        square = sets == ('eq66',)
        if square:
            P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening='rpa', df=True)
        else:
            P = ee_dbse2_pieces.build_pieces_dbse2(eps_gw, eps, B, no, sets=sets)
        for spin in SPINS:
            op = make(eps if square else eps_gw, eps, B, no, spin, form='exact',
                      sets=sets)
            opl = make(eps if square else eps_gw, eps, B, no, spin, form='laplace',
                       sets=sets)
            for omega in (None, 0.3):
                mv = ee_fold.folded_operator(P, omega, spin)[0]
                A = np.column_stack([mv(U[:, j]) for j in range(2)])
                Bm = op.apply(U, None if omega is None else np.full(2, omega))[0]
                ok &= check(rel(Bm, A) < 1e-12 and Bm.shape == (1, 2),
                            f"{'+'.join(sets)} {spin} w={omega}: H2/STO-3G product",
                            f'{rel(Bm, A):.1e}')
                if omega is not None:
                    Bl = opl.apply(U, np.full(2, omega))[0]
                    ok &= check(rel(Bl, A) < 1e-8
                                and opl.laplace_fallbacks == 2 * len(sets),
                                f"{'+'.join(sets)} {spin} w={omega}: H2/STO-3G "
                                'Laplace form against the pieces route, every node '
                                'a counted fallback',
                                f'{rel(Bl, A):.1e}, {opl.laplace_fallbacks} fallbacks, '
                                f'{2 * len(sets)} expected')
    return ok


def check_eq66(eps, B, no):
    """Check 10: level gw through the operator against build_pieces_gw's pieces."""
    ok = True
    nv = eps.size - no
    n = no * nv
    U = np.random.default_rng(23).standard_normal((n, 3))
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    tight = dict(dense_limit=0, tol_omega=1e-9, tol_residual=1e-8)
    for screening in ('tda', 'rpa'):
        P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening, df=True)
        Pw = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening)
        for spin in SPINS:
            tag = f'eq66 {screening} {spin}'
            op = make(eps, eps, B, no, spin, form='exact', sets=('eq66',),
                      screening=screening)
            for omega in (None, 0.3, 0.5):
                mv, dmv, _, _, _ = ee_fold.folded_operator(P, omega, spin)
                A = np.column_stack([mv(U[:, j]) for j in range(3)])
                Bm, nrm = op.apply(U, None if omega is None else np.full(3, omega))
                ok &= check(rel(Bm, A) < 1e-12, f'{tag} w={omega}: product',
                            f'{rel(Bm, A):.1e}')
                if omega is not None:
                    ref = np.array([dmv(U[:, j]) for j in range(3)])
                    ok &= check(rel(nrm, ref) < 1e-12, f'{tag} w={omega}: signed norm',
                                f'{rel(nrm, ref):.1e}')
            mv = ee_fold.folded_operator(P, None, spin)[0]
            dense = np.column_stack([mv(e) for e in np.eye(n)])
            ok &= check(rel(op.diag, np.diag(dense)) < 1e-12,
                        f'{tag}: diagonal is the dense channel matrix diagonal',
                        f'{rel(op.diag, np.diag(dense)):.1e}')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                ref = ee_fold.solve_folded(Pw, 2, spin=spin, solver='davidson', **tight)
                r = ee_fold.solve_folded(op.pieces(), 2, spin=spin, solver='davidson',
                                         **tight)
            same_shape = r.omega.shape == ref.omega.shape
            d = float(np.abs(r.omega - ref.omega).max()) if same_shape else np.inf
            dt = float(np.abs(r.t1 - ref.t1).max()) if same_shape else np.inf
            ok &= check(d < 1e-8 and dt < 1e-8 and r.converged.all()
                        and op.pieces()['level'] == 'gw',
                        f"{tag}: two Davidson roots and T1 equal the W pieces'; level "
                        "'gw'", f'{d:.1e} Ha, {dt:.1e} on T1, '
                        + ' '.join(f'{w:.6f}' for w in r.omega))
        om, rho = P['modes'] if 'modes' in P else ee_gw_pieces.gw_modes_df(
            eps, B, no, screening)
        Pb, Qb = ee_gw_pieces.static_term_pq(eps, B[:, :no, no:], rho, om, no)
        Abar = (np.einsum('ij,ab->iajb', np.eye(no), Pb + Pb.T)
                - np.einsum('ab,ij->iajb', np.eye(nv), Qb + Qb.T))
        d = float(np.abs(Abar - ee_gw_pieces.static_term_df(eps, B, rho, om, no)).max())
        ok &= check(Pb.shape == (nv, nv) and Qb.shape == (no, no) and d < 1e-14,
                    f'eq66 {screening}: static_term_pq assembled equals static_term_df',
                    f'{d:.1e}')
    bare = RIFactors.from_full(eps, B, no)
    Bt = ee_dbse2_pieces.dressed_factors(eps, B, no)[1]
    dressed = RIFactors.from_full(eps, Bt, no)
    modes = ee_gw_pieces.gw_modes_df(eps, B, no, 'rpa')
    direct = ee_dbse2_ri_fold.RIFoldDBSE2
    cases = (
        ('dressed', lambda: direct(bare, dressed, 'singlet', sets=('eq66',),
                                   modes=modes)),
        ('singles', lambda: make(eps, eps, B, no, 'singlet', sets=('eq66',),
                                 singles='eq62+eq59')),
        ('sets', lambda: make(eps, eps, B, no, 'singlet', sets=('eq61', 'eq66'))),
        # level gw runs at the reference energies: GW energies beside HF ones would
        # build a hybrid level, D and A-bar at the GW energies with the HF modes
        ('energies', lambda: make(gw_like(eps, no), eps, B, no, 'singlet',
                                  sets=('eq66',))),
    )
    for word, build in cases:
        try:
            build()
            ok &= check(False, f'eq66 with {word}: raises ValueError')
        except ValueError as err:
            ok &= check(word in str(err), f'eq66 with {word}: raises ValueError naming '
                        'it', str(err)[:70])
    return ok


def check_memory():
    """Check 4: the device memory pre-flight lifted from the static kernel."""
    ok = True
    try:
        linear_response.device_memory_check(2 ** 31, 2 ** 30, 'x')
        ok &= check(False, 'need above free raises ValueError')
    except ValueError as err:
        ok &= check(str(err).startswith('x need') and 'GiB' in str(err),
                    'need above free raises ValueError naming the label and GiB',
                    str(err)[:70])
    ok &= check(linear_response.device_memory_check(1, 2, 'x') is None,
                'need within free returns None')
    return ok


class _TinyDevice:
    """numpy behind a cuda attribute and a memory pool that report one byte free."""

    def __init__(self):
        self.cuda = types.SimpleNamespace(
            Device=lambda: types.SimpleNamespace(mem_info=(1, 1)))

    def get_default_memory_pool(self):
        return types.SimpleNamespace(free_bytes=lambda: 0)

    def __getattr__(self, name):
        return getattr(np, name)


def check_device(eps, B, no):
    """Check 5: the device class on the numpy stand-in against the host path."""
    ok = True
    eps_gw = gw_like(eps, no)
    n = no * (eps.size - no)
    U = np.random.default_rng(13).standard_normal((n, 3))
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    device = ee_dbse2_ri_fold._DoublesDBSE2GPU
    nodes, deriv = [0.3, 0.4, 0.5], [False, True, False]
    for form in ee_dbse2_ri_fold.FORMS:
        for kernel in KERNELS:
            for spin in SPINS:
                tag = f'{kernel} {spin} {form} stand-in'
                zero = kernel == 'eq71' and spin == 'triplet'
                oph = make(eps_gw, eps, B, no, spin, kernel=kernel, form=form)
                opd = make(eps_gw, eps, B, no, spin, kernel=kernel, form=form)
                opd.gpu = device(opd, xp=np)
                Ah, nh = oph.apply(U, np.full(3, 0.3))
                Ad, nd = opd.apply(U, np.full(3, 0.3))
                ok &= check(close(Ad, Ah, False) and close(nd, nh, zero),
                            f'{tag}: product and signed norm at w=0.3',
                            f'{rel(Ad, Ah):.1e}, {rel(nd, nh):.1e}')
                Mh, Fh = oph.fold_images(U, nodes, derivative=deriv)
                Md, Fd = opd.fold_images(U, nodes, derivative=deriv)
                good = close(Md, Mh, False)
                for q in range(3):
                    good &= close(Fd[q], Fh[q], zero)
                ok &= check(good, f'{tag}: fold_images, a derivative node between two '
                            'plain ones', f'{rel(Fd, Fh):.1e}')
                good, details = True, []
                for r_, s_, wr, ws in ((0, 1, 0.3, 0.5), (0, 0, 0.3, 0.3)):
                    dh = oph.doubles_dot(U[:, r_], wr, U[:, s_], ws)
                    dd = opd.doubles_dot(U[:, r_], wr, U[:, s_], ws)
                    good &= close(dd, dh, zero)
                    details.append(f'{rel(dd, dh):.1e}')
                ok &= check(good, f'{tag}: doubles_dot of two vectors at (0.3, 0.5) '
                            'and of one vector at (0.3, 0.3)', ', '.join(details))
                if form == 'laplace':
                    inside = np.full(3, opd.d_min + 0.05)
                    Ah = oph.apply(U, inside)[0]
                    Ad = opd.apply(U, inside)[0]
                    ok &= check(close(Ad, Ah, False) and opd.laplace_fallbacks == 3
                                and oph.laplace_fallbacks == 3,
                                f'{tag}: the fallback node through the device tile',
                                f'{rel(Ad, Ah):.1e}, {opd.laplace_fallbacks} fallbacks')
        op = make(eps_gw, eps, B, no, 'singlet', kernel='printed', form=form)
        try:
            device(op, xp=_TinyDevice())
            ok &= check(False, f'{form}: a device with one byte free is refused')
        except ValueError as err:
            ok &= check('GiB' in str(err), f'{form}: a device with one byte free is '
                        'refused before the upload', str(err)[:70])
    # the mode sets on the stand-in: eq 61 alone, both sets (both kernels and forms),
    # eq 66 (both screenings)
    cells = [(('eq61',), 'eq71', f, None) for f in ee_dbse2_ri_fold.FORMS]
    cells += [(('eq61', 'eq71'), k, f, None) for k in KERNELS
              for f in ee_dbse2_ri_fold.FORMS]
    cells += [(('eq66',), 'eq71', f, s) for s in ('tda', 'rpa')
              for f in ee_dbse2_ri_fold.FORMS]
    for sets, kernel, form, screening in cells:
        for spin in SPINS:
            tag = f"{'+'.join(sets)} {kernel} {form} {screening or ''} {spin} stand-in"
            kw = dict(kernel=kernel, form=form, sets=sets)
            if screening is None:
                oph, opd = make(eps_gw, eps, B, no, spin, **kw), make(
                    eps_gw, eps, B, no, spin, **kw)
            else:
                kw['screening'] = screening
                oph, opd = make(eps, eps, B, no, spin, **kw), make(eps, eps, B, no,
                                                                    spin, **kw)
            opd.gpu = device(opd, xp=np)
            Ah, nh = oph.apply(U, np.full(3, 0.3))
            Ad, nd = opd.apply(U, np.full(3, 0.3))
            ok &= check(close(Ad, Ah, False) and close(nd, nh, False),
                        f'{tag}: product and signed norm at w=0.3',
                        f'{rel(Ad, Ah):.1e}, {rel(nd, nh):.1e}')
            Mh, Fh = oph.fold_images(U, nodes, derivative=deriv)
            Md, Fd = opd.fold_images(U, nodes, derivative=deriv)
            good = close(Md, Mh, False)
            for q in range(3):
                good &= close(Fd[q], Fh[q], False)
            ok &= check(good, f'{tag}: fold_images, a derivative node between two '
                        'plain ones', f'{rel(Fd, Fh):.1e}')
            good, details = True, []
            for r_, s_, wr, ws in ((0, 1, 0.3, 0.5), (0, 0, 0.3, 0.3)):
                dh = oph.doubles_dot(U[:, r_], wr, U[:, s_], ws)
                dd = opd.doubles_dot(U[:, r_], wr, U[:, s_], ws)
                good &= close(dd, dh, False)
                details.append(f'{rel(dd, dh):.1e}')
            ok &= check(good, f'{tag}: doubles_dot of two vectors at (0.3, 0.5) and '
                        'of one vector at (0.3, 0.3)', ', '.join(details))
            if form == 'laplace':
                # a node inside the mode set's continuum falls back through the device
                # mode tile; on the two-set level eq 71's set too at or above its own
                # minimum; both minima read off the energies and the modes, not the
                # operator under test
                e_gw = eps_gw if screening is None else eps
                om_min = float(ee_gw_pieces.gw_modes_df(eps, B, no,
                                                        screening or 'rpa')[0].min())
                gap = float(e_gw[no:].min() - e_gw[:no].max())
                inside = om_min + gap + 0.05
                expect = 3 * (1 + ('eq71' in sets and inside >= 2.0 * gap))
                Ah = oph.apply(U, np.full(3, inside))[0]
                Ad = opd.apply(U, np.full(3, inside))[0]
                ok &= check(close(Ad, Ah, False) and opd.laplace_fallbacks == expect
                            and oph.laplace_fallbacks == expect,
                            f'{tag}: the fallback node through the device mode tile',
                            f'{rel(Ad, Ah):.1e}, {opd.laplace_fallbacks} fallbacks, '
                            f'{expect} expected')
    for form in ee_dbse2_ri_fold.FORMS:
        op = make(eps_gw, eps, B, no, 'singlet', sets=('eq61', 'eq71'), form=form)
        try:
            device(op, xp=_TinyDevice())
            ok &= check(False, f'eq61+eq71 {form}: a device with one byte free is '
                        'refused')
        except ValueError as err:
            ok &= check('GiB' in str(err) and 'modes' in str(err),
                        f'eq61+eq71 {form}: a device with one byte free is refused '
                        'before the upload, the label naming the modes', str(err)[:70])
    # the exact form on two simulated ranks, one occupied per tile, each rank with the
    # stand-in device, equals the serial stand-in run bit for bit (check 3's shape), on
    # eq 71's set and on both sets; the Laplace form refuses more than one rank and a
    # streamed B~_vv
    Bt = ee_dbse2_pieces.dressed_factors(eps, B, no)[1]
    bare = RIFactors.from_full(eps_gw, B, no)
    dressed = RIFactors.from_full(eps_gw, Bt, no)
    modes = ee_gw_pieces.gw_modes_df(eps, B, no, 'rpa')

    def rank(comm, form='exact', sets=('eq71',)):
        op = ee_dbse2_ri_fold.RIFoldDBSE2(bare, dressed, 'singlet', kernel='printed',
                                          form=form, sets=sets, modes=modes,
                                          tile_gb=1e-9, threads=1, comm=comm)
        op.gpu = device(op, xp=np)
        return (op.apply(U[:, :2], [0.3, 0.5]),
                op.doubles_dot(U[:, 0], 0.3, U[:, 1], 0.5))

    for sets in (('eq71',), ('eq61', 'eq71')):
        ser = rank(None, sets=sets)
        same = all(np.array_equal(p[0], ser[0][0]) and np.array_equal(p[1], ser[0][1])
                   and d == ser[1] for p, d in run_simulated(rank, 2, sets=sets))
        ok &= check(same, f"{'+'.join(sets)} exact stand-in: two simulated ranks, one "
                    'occupied per tile, equal the serial run bit for bit')
    streamed = RIFactors.from_full(eps_gw, Bt, no)
    streamed.Bvv = None
    # the bare factor in auxiliary chunks, as one rank holds it over MPI
    chunked = RIFactors.from_full(eps_gw, B, no)
    chunks = aux_chunks(chunked.naux, chunked.ev.size, 1)[0]
    chunked.bvv_chunks = {c: np.ascontiguousarray(chunked.Bvv[:, p0:p1])
                          for c, (p0, p1) in enumerate(chunks)}
    chunked.Bvv = None
    modes = ee_gw_pieces.gw_modes_df(eps, B, no, 'rpa')
    cases = (('two ranks', 'one rank',
              lambda: run_simulated(rank, 2, form='laplace')),
             ('no whole B~_vv', 'one rank', lambda: ee_dbse2_ri_fold.RIFoldDBSE2(
                 bare, streamed, 'singlet', form='laplace')),
             ('no whole bare B_vv', 'bare B_vv', lambda: ee_dbse2_ri_fold.RIFoldDBSE2(
                 chunked, None, 'singlet', sets=('eq61',), modes=modes,
                 form='laplace')))
    for word, named, build in cases:
        try:
            build()
            ok &= check(False, f'laplace with {word}: raises NotImplementedError')
        except NotImplementedError as err:
            ok &= check(named in str(err), f'laplace with {word}: raises '
                        f'NotImplementedError naming {named}', str(err)[:70])
    if ee_dbse2_ri_fold.cp is None:
        try:
            make(eps_gw, eps, B, no, 'singlet', numerics=ON_GPU)
            ok &= check(False, "device='gpu' without CuPy raises ImportError")
        except ImportError as err:
            ok &= check('CuPy' in str(err), "device='gpu' without CuPy raises "
                        'ImportError naming CuPy', str(err)[:70])
    else:
        print("  [skip] device='gpu' without CuPy: CuPy is importable here")
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
            ope = make(eps_gw, eps, B, no, spin, kernel=kernel, form='exact')
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
            # two frequencies close together, where solve_folded's duplicate check
            # calls it: the partial fraction's conditioning
            good, details = True, []
            for d in (1e-2, 1e-5, 1e-9):
                de = ope.doubles_dot(U[:, 0], 0.3, U[:, 1], 0.3 + d)
                dl = opl.doubles_dot(U[:, 0], 0.3, U[:, 1], 0.3 + d)
                g, detail = within(dl, de, zero)
                good &= g
                details.append(detail)
            ok &= check(good, f'{tag}: doubles_dot at frequencies 1e-2, 1e-5 and 1e-9 '
                        'Ha apart', ', '.join(details))
            mv = ope.matvecs
            ope.fold_images(U, nodes)
            ok &= check(ope.matvecs - mv == 3, f'{tag}: fold_images counts its M '
                        'columns in matvecs', f'{ope.matvecs - mv} for 3 columns')
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
            # a fallback node between two grid nodes in one call, the derivative on it
            nodes3, deriv3 = [0.3, inside, 0.5], [False, True, False]
            Fe = ope.fold_images(U, nodes3, derivative=deriv3)[1]
            before = opl.laplace_fallbacks
            Fl = opl.fold_images(U, nodes3, derivative=deriv3)[1]
            good = close(Fl[1], Fe[1], zero)
            details = [f'{rel(Fl[1], Fe[1]):.1e}']
            for q in (0, 2):
                g, detail = within(Fl[q], Fe[q], zero)
                good &= g
                details.append(detail)
            ok &= check(good and opl.laplace_fallbacks - before == 3,
                        f'{tag}: fold_images with a fallback node between two grid '
                        'nodes, the derivative on it',
                        f"{', '.join(details)}, {opl.laplace_fallbacks - before} "
                        'fallbacks')
            # the image and its derivative at one frequency share one grid and one pass
            calls = []
            grid = opl._grid
            opl._grid = lambda *a: (calls.append(a), grid(*a))[1]
            opl.fold_images(U[:, :1], [0.3, 0.3], derivative=[False, True])
            del opl._grid
            ok &= check(len(calls) == 1, f'{tag}: one grid for the image and its '
                        'derivative at one frequency', f'{len(calls)} grids built')
    try:
        make(eps_gw, eps, B, no, 'singlet', form='x')
        ok &= check(False, 'form: raises ValueError')
    except ValueError as err:
        ok &= check('form' in str(err), 'form: raises ValueError naming it',
                    str(err)[:70])
    ok &= check(make(eps_gw, eps, B, no, 'singlet').form == 'laplace',
                'form: laplace by default')
    # form=None where the Laplace form cannot run: 'exact' with one UserWarning naming
    # the reason; on simulated ranks the form alone (the warning is raised on their
    # threads and ignored there)
    chunked = RIFactors.from_full(eps_gw, B, no)
    chunks = aux_chunks(chunked.naux, chunked.ev.size, 1)[0]
    chunked.bvv_chunks = {c: np.ascontiguousarray(chunked.Bvv[:, p0:p1])
                          for c, (p0, p1) in enumerate(chunks)}
    chunked.Bvv = None
    modes = ee_gw_pieces.gw_modes_df(eps, B, no, 'rpa')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        opc = ee_dbse2_ri_fold.RIFoldDBSE2(chunked, None, 'singlet', sets=('eq61',),
                                           modes=modes)
    ok &= check(opc.form == 'exact' and len(caught) == 1
                and issubclass(caught[0].category, UserWarning)
                and 'bare B_vv' in str(caught[0].message),
                "form: None with the bare B_vv in chunks resolves to 'exact' with one "
                'warning naming it',
                str(caught[0].message)[:70] if caught else 'no warning')
    # the factors once, on the main thread: a from_mo build per simulated rank
    # segfaults on Snellius's wheel stack
    bare = RIFactors.from_full(eps_gw, B, no)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # emitted on the ranks' threads
        forms = run_simulated(lambda comm: ee_dbse2_ri_fold.RIFoldDBSE2(
            bare, None, 'singlet', sets=('eq61',), modes=modes, comm=comm).form, 2)
    ok &= check(forms == ['exact', 'exact'],
                "form: None on two simulated ranks resolves to 'exact'", str(forms))
    return ok


def check_laplace_shifted(eps, B, no):
    """Check 12: the Laplace form on the two-set level when eq 61's minimum lies below
    eq 71's (the modes shifted down by 0.2 Ha, as at a reference whose Omega_min lies
    below the GW gap): each set's Laplace sums take their own span and exponent shift
    (eq 71's own minimum for eq 71's, the mode set's extremes for the modes'), so the
    products, the signed norms and the two-vector overlap equal the exact form's."""
    ok = True
    eps_gw = gw_like(eps, no)
    n = no * (eps.size - no)
    U = np.random.default_rng(23).standard_normal((n, 3))
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    om, rho = ee_gw_pieces.gw_modes_df(eps, B, no, 'rpa')
    modes = (om - 0.2, rho)
    d71 = 2.0 * float(eps_gw[no:].min() - eps_gw[:no].max())
    for spin in SPINS:
        ope = make(eps_gw, eps, B, no, spin, form='exact', sets=('eq61', 'eq71'),
                   modes=modes)
        opl = make(eps_gw, eps, B, no, spin, sets=('eq61', 'eq71'), modes=modes,
                   form='laplace')
        ok &= check(ope.d_min < d71 - 1e-6, f'eq61+eq71 {spin}: the shifted modes put '
                    "the level's minimum below eq 71's",
                    f'd_min {ope.d_min:.4f}, eq 71 {d71:.4f} Ha')
        for omega in (0.3, 0.5):
            w = np.full(3, omega)
            Ae, ne = ope.apply(U, w)
            Al, nl = opl.apply(U, w)
            ok &= check(rel(Al, Ae) < 1e-8 and rel(nl, ne) < 1e-8
                        and opl.laplace_fallbacks == 0,
                        f'eq61+eq71 {spin} w={omega}: Laplace equals exact with the '
                        "level's minimum below eq 71's", f'{rel(Al, Ae):.1e}, norm '
                        f'{rel(nl, ne):.1e}, {opl.laplace_fallbacks} fallbacks')
        # between the two minima the mode set falls back to its tile and eq 71's set
        # stays on its quadrature (the ordering check 14 cannot reach); the mode
        # minimum off the shifted modes and the energies
        dm_min = float(modes[0].min() + eps_gw[no:].min() - eps_gw[:no].max())
        mid = 0.5 * (dm_min + d71)
        before = opl.laplace_fallbacks
        Ae = ope.apply(U, np.full(3, mid))[0]
        Al = opl.apply(U, np.full(3, mid))[0]
        fell = opl.laplace_fallbacks - before
        ok &= check(rel(Al, Ae) < 1e-8 and fell == 3,
                    f'eq61+eq71 {spin}: a node between the two minima falls back on '
                    'the mode set alone',
                    f'{rel(Al, Ae):.1e}, {fell} fallbacks, 3 expected')
        de = ope.doubles_dot(U[:, 0], 0.3, U[:, 1], 0.5)
        dl = opl.doubles_dot(U[:, 0], 0.3, U[:, 1], 0.5)
        ok &= check(abs(dl - de) < 1e-8 * max(abs(de), 1.0),
                    f'eq61+eq71 {spin}: the two-vector overlap, Laplace against exact',
                    f'{abs(dl - de):.1e}')
    return ok


def check_mode_laplace(eps, B, no):
    """Check 14: the mode sets' Laplace form against the exact mode tile."""
    ok = True
    eps_gw = gw_like(eps, no)
    n = no * (eps.size - no)
    U = np.random.default_rng(29).standard_normal((n, 3))
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    tight = dict(dense_limit=0, tol_omega=1e-9, tol_residual=1e-8)

    def within(a, b):
        return rel(a, b) < 1e-8, f'{rel(a, b):.1e}'

    cells = [(('eq61',), 'eq71', None)]
    cells += [(('eq61', 'eq71'), k, None) for k in KERNELS]
    cells += [(('eq66',), 'eq71', s) for s in ('tda', 'rpa')]
    for sets, kernel, screening in cells:
        for spin in SPINS:
            tag = f"{'+'.join(sets)} {kernel} {screening or ''} {spin} mode laplace"
            kw = dict(kernel=kernel, sets=sets)
            energies = (eps_gw, eps)
            if screening is not None:
                kw['screening'] = screening
                energies = (eps, eps)
            ope = make(*energies, B, no, spin, **kw, form='exact')
            opl = make(*energies, B, no, spin, form='laplace', **kw)
            for omega in (0.3, 0.5):
                om = np.full(3, omega)
                Ae, ne = ope.apply(U, om)
                Al, nl = opl.apply(U, om)
                good, detail = within(Al, Ae)
                ok &= check(good, f'{tag} w={omega}: product', detail)
                good, detail = within(nl, ne)
                ok &= check(good, f'{tag} w={omega}: signed norm', detail)
            nodes, deriv = [0.3, 0.4, 0.5], [False, True, False]
            Me, Fe = ope.fold_images(U, nodes, derivative=deriv)
            Ml, Fl = opl.fold_images(U, nodes, derivative=deriv)
            good, details = rel(Ml, Me) < 1e-12, []
            for q in range(3):
                g, d = within(Fl[q], Fe[q])
                good &= g
                details.append(d)
            ok &= check(good, f'{tag}: fold_images, a derivative node between two '
                        'plain ones', ', '.join(details))
            # the signed overlap of two random vectors cancels, so its error is read
            # against the largest of it and the two self-overlaps
            good, details = True, []
            for wr, ws in ((0.3, 0.5), (0.3, 0.3), (0.3, 0.31), (0.3, 0.30001),
                           (0.3, 0.300000001)):
                de = ope.doubles_dot(U[:, 0], wr, U[:, 1], ws)
                dl = opl.doubles_dot(U[:, 0], wr, U[:, 1], ws)
                scale = max(abs(de), abs(ope.doubles_dot(U[:, 0], wr, U[:, 0], wr)),
                            abs(ope.doubles_dot(U[:, 1], ws, U[:, 1], ws)))
                err = abs(dl - de) / scale
                good &= err < 1e-8
                details.append(f'{err:.1e} ({rel(dl, de):.1e} plain)')
            ok &= check(good, f'{tag}: doubles_dot at two frequencies, at one, and at '
                        '1e-2, 1e-5 and 1e-9 Ha apart', ', '.join(details))
            ok &= check(opl.laplace_fallbacks == 0, f'{tag}: no fallback so far',
                        f'{opl.laplace_fallbacks}')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                re_ = ee_fold.solve_folded(ope.pieces(), 2, spin=spin,
                                           solver='davidson', **tight)
                rl = ee_fold.solve_folded(opl.pieces(), 2, spin=spin,
                                          solver='davidson', **tight)
            d = (float(np.abs(rl.omega - re_.omega).max())
                 if rl.omega.shape == re_.omega.shape else np.inf)
            ok &= check(d < 1e-8 and rl.converged.all(),
                        f'{tag}: two Davidson roots equal the exact form',
                        f'{d:.1e} Ha')
            # a node inside the mode set's continuum falls back to the mode tile; on
            # the two-set level eq 71's set falls back too where its grid is None
            # both minima read off the energies and the modes, not the operator: the
            # mode set's min Omega + min eps_v - max eps_o, eq 71's 2 (min eps_v - max
            # eps_o), at the GW energies
            om_min = float(ee_gw_pieces.gw_modes_df(energies[1], B, no,
                                                    screening or 'rpa')[0].min())
            gap = float(energies[0][no:].min() - energies[0][:no].max())
            dm_min, d71 = om_min + gap, 2.0 * gap
            ok &= check(abs(opl.dm_min - dm_min) < 1e-12, f"{tag}: the operator's mode"
                        ' minimum equals min Omega + min eps_v - max eps_o',
                        f'{abs(opl.dm_min - dm_min):.1e} Ha')
            inside = dm_min + 0.05
            expect = 3 * (1 + ('eq71' in sets and inside >= d71))
            before = opl.laplace_fallbacks
            Ae = ope.apply(U, np.full(3, inside))[0]
            Al = opl.apply(U, np.full(3, inside))[0]
            fell = opl.laplace_fallbacks - before
            ok &= check(rel(Al, Ae) < 1e-12 and fell == expect,
                        f"{tag}: a node inside the mode set's continuum falls back to "
                        'the mode tile',
                        f'{rel(Al, Ae):.1e}, {fell} fallbacks, {expect} expected')
            if 'eq71' in sets and d71 < dm_min:
                # between the two minima: eq 71's set falls back, the mode set stays on
                # its quadrature
                mid = 0.5 * (d71 + dm_min)
                before = opl.laplace_fallbacks
                Ae = ope.apply(U, np.full(3, mid))[0]
                Al = opl.apply(U, np.full(3, mid))[0]
                fell = opl.laplace_fallbacks - before
                ok &= check(rel(Al, Ae) < 1e-8 and fell == 3,
                            f"{tag}: a node between the two minima falls back in eq "
                            "71's set alone", f'{rel(Al, Ae):.1e}, {fell} fallbacks')
    return ok


def check_tiles(eps, B, no):
    """Check 13: tiles of two occupied rows with an uneven last tile (water's five rows
    as 2, 2, 1), DABNA-1's tiling, against one tile: products, signed norms and the
    two-vector overlap on the two-set level."""
    ok = True
    eps_gw = gw_like(eps, no)
    nv = eps.size - no
    n = no * nv
    U = np.random.default_rng(29).standard_normal((n, 2))
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    one = make(eps_gw, eps, B, no, 'singlet', sets=('eq61', 'eq71'), form='exact')
    per_k = (ee_dbse2_ri_fold._tile_bytes(1, no, nv, B.shape[0])
             + ee_dbse2_ri_fold._mode_bytes(1, nv, one.nm, B.shape[0]))
    two = make(eps_gw, eps, B, no, 'singlet', sets=('eq61', 'eq71'),
               tile_gb=2.5 * per_k / 1024 ** 3, form='exact')
    last = two.tiles[-1][1] - two.tiles[-1][0]
    ok &= check(one.kt == no and two.kt == 2 and last == 1,
                'kt 2 gives tiles of 2, 2, 1 rows against the one tile',
                f'one {one.tiles}, two {two.tiles}')
    A1, n1 = one.apply(U, np.array([0.3, 0.5]))
    A2, n2 = two.apply(U, np.array([0.3, 0.5]))
    ok &= check(rel(A2, A1) < 1e-12 and rel(n2, n1) < 1e-12,
                'products and signed norms on three tiles equal one tile',
                f'{rel(A2, A1):.1e}, norm {rel(n2, n1):.1e}')
    d1 = one.doubles_dot(U[:, 0], 0.3, U[:, 1], 0.5)
    d2 = two.doubles_dot(U[:, 0], 0.3, U[:, 1], 0.5)
    ok &= check(abs(d2 - d1) < 1e-12 * max(abs(d1), 1.0),
                'the two-vector overlap on three tiles equals one tile',
                f'{abs(d2 - d1):.1e}')
    return ok


def check_budget(eps, B, no):
    """Check 9: the budgets bound the peaks tracemalloc traces over one call."""
    ok = True
    eps_gw = gw_like(eps, no)
    make = ee_dbse2_ri_fold.RIFoldDBSE2.from_mo
    op = make(eps_gw, eps, B, no, 'triplet', kernel='printed', form='laplace')
    nv, naux = op.nv, op.f.naux
    u = np.random.default_rng(17).standard_normal((2, op.n))
    Zr, Zs = op._vertex_factors(u[0]), op._vertex_factors(u[1])
    (k0, k1), = op.tiles
    kt = k1 - k0

    def peak(fn):
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn()
        _, top = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return top

    def bounded(measured, budget, label):
        return check(0.5 * budget < measured <= budget, label,
                     f'{measured / 1e3:.0f} kB traced, {budget / 1e3:.0f} kB budgeted')

    nodes, deriv = np.array([0.3, 0.4]), np.array([False, True])
    core = peak(lambda: ee_dbse2_ri_fold._tile_core(
        np, op.L2, op.f.eo, op.f.ev, k0, k1, Zr[0], Zr[1], op.coupling, op.spin, nodes,
        deriv))
    ok &= bounded(core, ee_dbse2_ri_fold._tile_bytes(kt, no, nv, naux, 2),
                  'the tile budget bounds _tile_core at two nodes (triplet)')
    overlap = peak(lambda: ee_dbse2_ri_fold._tile_overlap(
        np, op.L2, op.f.eo, op.f.ev, k0, k1, Zr, 0.3, Zs, 0.5, op.coupling, op.spin))
    tile = ee_dbse2_ri_fold._tile_bytes(kt, no, nv, naux)
    ok &= bounded(overlap, tile, 'the tile budget bounds _tile_overlap of two vectors')
    # the printed singlet holds every closure, the largest Laplace node
    opl = make(eps_gw, eps, B, no, 'singlet', kernel='printed', form='laplace')
    Zl = opl._vertex_factors(u[0])
    grid = opl._grid(0.3)
    node = peak(lambda: opl._laplace(Zl, grid, [lambda t, w: w, lambda t, w: w * t]))
    lap = ee_dbse2_ri_fold._laplace_bytes(no, nv, naux)
    ok &= bounded(node, lap, 'the Laplace budget bounds one Laplace node (printed '
                  'singlet, image and derivative)')
    residents = op.L2.nbytes + op.Boo_i.nbytes + op.Bvv_a.nbytes
    op.gpu = ee_dbse2_ri_fold._DoublesDBSE2GPU(op, xp=np)
    ok &= check(op.gpu.need >= residents + 2 * op.L2.nbytes + lap + tile,
                "the Laplace device's need carries the residents, the uploaded "
                'factors, one Laplace node and the fallback tile',
                f'{op.gpu.need / 1e6:.1f} MB')
    ope = make(eps_gw, eps, B, no, 'triplet', kernel='printed', form='exact')
    ope.gpu = ee_dbse2_ri_fold._DoublesDBSE2GPU(ope, xp=np)
    ok &= check(ope.gpu.need >= 3 * ope.L2.nbytes + tile,
                "the exact device's need carries the factor, the uploaded factors and "
                'one tile', f'{ope.gpu.need / 1e6:.1f} MB')
    # the mode tile: eq 61's cross and eq 66's square, the overlap of two vectors
    for sets, square, args in ((('eq61',), False, (eps_gw, eps)),
                               (('eq66',), True, (eps, eps))):
        opm = make(*args, B, no, 'triplet', sets=sets, form='exact')
        fb = opm.m_op.f
        (k0, k1), = opm.tiles
        Zr = opm._vertex_factors(u[0], fb)
        Zs = opm._vertex_factors(u[1], fb)
        core = peak(lambda: ee_dbse2_ri_fold._mode_tile(
            np, opm.rho, opm.om, fb.eo, fb.ev, k0, k1, Zr[0], Zr[1], nodes, deriv,
            square))
        ok &= bounded(core, ee_dbse2_ri_fold._mode_bytes(k1 - k0, nv, opm.nm, naux, 2),
                      f"{'+'.join(sets)}: the mode budget bounds _mode_tile at two "
                      'nodes')
        overlap = peak(lambda: ee_dbse2_ri_fold._mode_overlap(
            np, opm.rho, opm.om, fb.eo, fb.ev, k0, k1, Zr, 0.3, Zs, 0.5, square))
        # without the Gam pair: at water naux exceeds nm, so the pair's share would
        # cover an overlap that outgrows the tile arrays' budget at production sizes
        ok &= bounded(overlap, ee_dbse2_ri_fold._mode_bytes(k1 - k0, nv, opm.nm, 0),
                      f"{'+'.join(sets)}: the mode budget's tile arrays alone bound "
                      '_mode_overlap of two vectors')
    # the mode set's Laplace node: eq 61's cross, image and derivative
    opml = make(eps_gw, eps, B, no, 'singlet', sets=('eq61',), form='laplace')
    Zb = opml._vertex_factors(u[0], opml.m_op.f)
    grid = opml._grid(0.3, span=(opml.dm_min, opml.dm_max))
    node = peak(lambda: opml._mode_laplace(Zb, grid, [lambda t, w: w,
                                                  lambda t, w: w * t]))
    mlap = ee_dbse2_ri_fold._mode_laplace_bytes(no, nv, naux, opml.nm)
    ok &= bounded(node, mlap, "the mode set's Laplace budget bounds one of its "
                  'nodes (eq 61, image and derivative)')
    opms = make(eps, eps, B, no, 'singlet', sets=('eq66',), form='laplace')
    Zbs = opms._vertex_factors(u[0], opms.m_op.f)
    grids = opms._grid(0.3, span=(opms.dm_min, opms.dm_max))
    node_sq = peak(lambda: opms._mode_laplace(Zbs, grids, [lambda t, w: w,
                                                           lambda t, w: w * t]))
    ok &= bounded(node_sq, ee_dbse2_ri_fold._mode_laplace_bytes(no, nv, naux, opms.nm),
                  "the mode set's Laplace budget bounds one of its nodes (eq 66, the "
                  'square with its ZK - ZI copy)')
    op3 = make(eps_gw, eps, B, no, 'singlet', sets=('eq61', 'eq71'), form='laplace')
    op3.gpu = ee_dbse2_ri_fold._DoublesDBSE2GPU(op3, xp=np)
    need3 = (3 * op3.L2.nbytes + op3.Boo_i.nbytes + op3.Bvv_a.nbytes + lap + tile
             + op3.rho.nbytes + op3.om.nbytes + 2 * op3.L2b.nbytes + op3.Boo_ib.nbytes
             + op3.Bvv_ab.nbytes + mlap
             + ee_dbse2_ri_fold._mode_bytes(op3.kt, nv, op3.nm, naux))
    ok &= check(op3.gpu.need >= need3,
                "the two-set Laplace device's need carries both sets' residents, "
                'nodes and fallback tiles', f'{op3.gpu.need / 1e6:.1f} MB')
    op2 = make(eps_gw, eps, B, no, 'singlet', sets=('eq61', 'eq71'), form='exact')
    op2.gpu = ee_dbse2_ri_fold._DoublesDBSE2GPU(op2, xp=np)
    mode = ee_dbse2_ri_fold._mode_bytes(op2.kt, nv, op2.nm, naux)
    ok &= check(op2.gpu.need >= (3 * op2.L2.nbytes + tile + op2.rho.nbytes
                                 + op2.om.nbytes + 2 * op2.L2b.nbytes + mode),
                "the two-set device's need carries the dressed factor, the modes, the "
                'uploaded factors of both sets and both tiles',
                f'{op2.gpu.need / 1e6:.1f} MB, mode tile {mode / 1e6:.1f} MB')
    # the two-vector dot uploads both vectors' factor images per set: counted on the
    # stand-in, they bound need beside the residents and the tiles
    uploaded = []
    plain = op2.gpu.upload
    op2.gpu.upload = lambda Z: (uploaded.append(sum(int(z.nbytes) for z in Z)),
                                plain(Z))[1]
    op2.doubles_dot(u[0], 0.3, u[1], 0.5)
    op2.gpu.upload = plain
    ok &= check(op2.gpu.need >= (sum(uploaded) + op2.L2.nbytes + op2.rho.nbytes
                                 + op2.om.nbytes + tile + mode),
                "need covers the two-vector dot's uploads beside the residents and "
                'both tiles', f'uploaded {sum(uploaded) / 1e6:.1f} MB in '
                f'{len(uploaded)} calls, need {op2.gpu.need / 1e6:.1f} MB')
    return ok


def main():
    """Run the checks in order on water/6-31G and print the file's verdict."""
    eps, B, no = water('6-31g')
    ok = True
    print('check 1: the operator against the pieces route (6-31G)')
    for sets in SETS:
        ok &= check_operator(eps, B, no, sets)
    print('refusals')
    ok &= check_refusals(eps, B, no)
    print('check 2: the three solvers on the operator against the pieces')
    ok &= check_solves(eps, B, no)
    print('check 3: simulated ranks')
    ok &= check_ranks(eps, B, no)
    print('check 7: the modes at the HF energies; the Laplace form on the two sets')
    ok &= check_modes(eps, B, no)
    print("check 10: sets=('eq66',), level gw through the operator")
    ok &= check_eq66(eps, B, no)
    print('check 11: nocc = nvirt = nm = 1 (H2/STO-3G)')
    ok &= check_minimal()
    print('check 4: the device memory pre-flight')
    ok &= check_memory()
    print('check 5: the device class on the numpy stand-in')
    ok &= check_device(eps, B, no)
    print('check 6: the Laplace form against the exact form')
    ok &= check_laplace(eps, B, no)
    print('check 9: the budgets against traced peaks')
    ok &= check_budget(eps, B, no)
    print("check 12: the Laplace form with the level's minimum below eq 71's")
    ok &= check_laplace_shifted(eps, B, no)
    print('check 13: tiles of two rows with an uneven last tile')
    ok &= check_tiles(eps, B, no)
    print("check 14: the mode sets' Laplace form against the exact mode tile")
    ok &= check_mode_laplace(eps, B, no)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
