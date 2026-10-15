"""Folded EE-ADC solver (ee_fold) and the GF2 block order.

The doubles block of the EE-ADC operator is exactly diagonal at block order o_dd = 0
(levels adc2 and gf2), so the doubles can be eliminated without approximation and the
singles problem A_eff(w) y = w y, A_eff(w) = M - V^T (D - w)^-1 V, solved at each
root's own w. Checks, on water / cc-pVDZ (RHF, DF factors) unless stated:

  1. build_operator(pieces=True) hands out M, V, Vt, D that rebuild aop(vec) on a
     random vector to 1e-12, at adc2; pieces=True raises ValueError at adc1, adc2x,
     adc3.
  2. the gf2 block order (1, 1, 0): its M equals adc1's singles block, its V and Vt
     equal adc2's on a random vector, its D equals d_ijab, its dimensions equal
     adc2's.
  3. folded_operator at the full solve's root w reproduces that root on the singles part
     of the full eigenvector (residual < 1e-8); the dense path equals the matrix-free
     one; wrong length and complex dtype raise ValueError.
  4. solve_folded at adc2 reproduces the full spin-free ADC(2) Davidson roots, three
     singlets and three triplets, to 1e-5 eV, with T1 against the full eigenvector's
     singles weight to 1e-4, every root under eight Newton steps; t_min=1.0 forces the
     fixed-point loop to the same roots; H2/STO-3G triplets warn on a short channel
     and the one root equals the full solve.
  5. CH4 / cc-pVDZ (not NH3: the rounded C3v geometry splits the E pair by 2e-4 eV):
     the degenerate T2 singlet triple is followed without a skip, its partners
     orthonormal, on the dense and on the Davidson branch; one level at one omega,
     equal T1, x1, a cut level returned whole, an exhausted level unconverged.
  6. the Davidson branch equals the dense one on water; spin=None equals the full
     solve over both channels.
  7. gf2: the fold equals the full gf2 channel solve; the dense integral route
     equals the DF route with exact factors; the unrestricted route on an RHF
     reference equals the spin-free one (6-31G), and so does the spin-orbital route,
     singlets and triplets, whose Davidson equals its dense supermatrix (STO-3G).
  8. pieces=True with parity and en_dress at gf2 raise ValueError; an exhausted
     step budget reports converged False with a warning; no input is mutated.
  9. eq 53 as printed (Monino and Loos 2023: eq 54a plus the six terms of eq 56,
     transcribed in spin orbitals on the same DF factors) equals the gf2 A_eff
     element by element at w = None and 0.25 Ha to 1e-10 Ha; with eq 57 added it
     equals the adc2 A_eff.
  10. 'gw' is a fold level; pieces['dnorm2'] sets the doubles norm, the adc2 path
      unchanged.
  11. the level and duplicate helpers on constructed values.
  12. a collapsing distinct pair is reported, the kept root exact; a level split in
      A_eff warns and each root is a supermatrix eigenvalue; distinct roots cost the
      duplicate check no doubles image.
  13. CH4 / 6-31G triplet, where the fold puts M's lowest state (A1) above the T2
      triple: nroots = 1 returns the triple, equal to the full solve, on both
      branches and without a warning; a root above a doubles energy skips the
      count with a warning. Check 12's collapsing pair drops its copy and gets the
      missed root back by branch index, on the Davidson branch too at tol_omega
      far below tol_residual.
  14. the count where M misleads it: a state M's diagonal ranks last but folded
      lowest, a missing branch mixed with a found root at the cut (25 points), and
      one count solve per call when nothing is missing; a bracket narrower than
      tol_omega stops the index solve at once.
  15. dense_limit=None takes the pieces' own default: none at adc2 (Davidson), 40
      at gw (dense on water/6-31G, n 40); an explicit value overrides.
  16. the count's window: a pair of M split by 2e-9 Ha at the cut, tol_omega 1e-10,
      is two roots to the default, which returns the two lowest exact, a state
      folded below the pair among them, and none warns, since the count's window
      (1e-8 Ha) holds the partner; a distinct partner 1e-7 or 1e-6 Ha above the
      cut, not asked for, costs no warning and one count, widened once, at default
      tolerances.
  17. the count's check off the span of its own vectors: on the Davidson branch at
      n 800, a state last on M's diagonal folded lowest, where the roots found and
      M's seeds are exact eigenvectors of A_eff, is found at nroots 1 and 3, also
      inside a coupled block, without a warning.
  18. the count's paths that warn or fall back: two Newton steps leave every root
      unconverged and the surplus warns; a pair closer than tol_omega returns one
      root and says so; with no round left the missing root is named; an index
      step that leaves its bracket bisects it; a missing group whose roots differ
      is re-solved per branch, on both branches.
  19. a state folded to within a found root's window, that root not the cut, below
      or above it: the found root's branch comes from its own vector and the missed
      state is solved, both roots exact, no warning, on both branches and both
      solvers (the repair is the count's, reached through solver='newton'; the
      default's Davidson finds the state itself).
  20. the index solve keeps the cut's vectors in every start: a state folded
      between two found roots is solved on the Davidson branch from a start of
      exact eigenvectors (n 4) and from one symmetry block of two (n 200), without
      a warning.
  21. a root found past min D, the lowest doubles energy, as the nroots-th: the
      count runs at the highest root found below min D, so a state folded between
      the seeds is solved on both branches, without a warning.
  22. the Davidson start keeps every independent vector offered (n 4: two found
      roots, the random vector, four diagonal seeds), orthonormal, the found ones
      first.
  23. an unknown level and an unknown spin raise ValueError; a fixed-point loop
      out of steps warns, and the count solves its root again by index.
  24. an exactly degenerate pair the seeds miss, with M's seeds exact eigenvectors
      (spin=None on pieces without a cross-spin block, n 20 and 200, Davidson):
      the count's check finds both partners, no warning.
  25. the progress trace (water singlets, nroots 3, Davidson): verbose 1 returns
      the roots of verbose 0 to 1e-10 Ha, verbose 0 prints nothing; at verbose 1 the
      start, seeds, count round, check and closing lines appear, every line ends on
      its elapsed time and matvec count, the closing count equals every
      application of the operator (with solver='newton' the matvecs
      solve_symmetric received), a zero heartbeat interval prints heartbeats, and
      no trace outlives a call, one that raises included.
  26. ee_spin_blocks._ew, the elementwise pass split over the thread pin, equals the
      plain ufunc bit for bit at a size it splits (2.1e6 elements, 4 threads): a
      difference with an (a, b)-transposed operand, an in-place sum, a scalar
      product, a quotient, and a broadcast shape it leaves to NumPy.
  27. ee_spin_blocks.ooov_ia and ovvv_ia, their products summed over j-slabs on 4
      threads, equal the block-wise einsum on anti4's blocks for an X without a
      flip parity and with each one, to 1e-12 relative.
  28. dmatvec(u, vector=True) is dA_eff/domega u: its quadratic form is -dmatvec(u)
      to 1e-12 relative and it equals the central difference of matvec at
      h = 1e-4 Ha to 1e-6 relative; at omega None it is zeros; an optional
      pieces['K'] replaces the V, divide, Vt path of matvec and of both dmatvec
      forms, bitwise, called once per application with powers (1,), () and (2,).
  29. _model_roots, the subspace model H0 + (w - w0) H1 solved per branch: with
      H1 = 0 its roots are the eigenvalues of H0, with H1 = -I they are
      (w_j + w0) / 2, a root at or above min D is dropped, and an exact crossing
      of two branches (check 20's pieces at w0 = 0.5) gives no false root.
  30. _davidson_omega, one Davidson whose roots carry their own w (water adc2
      singlets and triplets, nroots 3, from the seeds of M, max_subspace 40 and 4):
      the model converges, its roots lie within 1e-3 Ha of the Newton roots, and
      the per-root Newton polish from them, at most three steps a root, returns
      the Newton roots to 1e-7 Ha; a collapse at one centre images only the new
      vectors; two cycles leave it unconverged, warned.
  31. solver='davidson' is the default of solve_folded: on water singlets, nroots
      3, its roots equal solver='newton''s to 1e-8 Ha and its T1 to 1e-6, loop
      reads 'davidson', steps counts the polish alone, solver='bogus' raises
      ValueError; on CH4 / 6-31G
      triplets, nroots 1 (check 13's reordered case), it returns Newton's T2
      triple without an index solve and closes on fewer matvecs.
  32. the default's polish and duplicate check: a model root one polish step
      cannot fix (strong coupling, the centre held at M's root) comes back
      converged False with a warning, and so does a 4-fold level, every partner;
      a root the Davidson hands over twice is reported, the copy dropped, the
      displaced root solved by index; max_newton 0 still returns the root;
      seeds left unconverged are one note line on the default, not a warning, and
      solve_symmetric's warning with solver='newton'.

The machinery of the seeds' Newton path (step budgets, a seed collapse, a seed
level split in A_eff, the count's repairs, a cut past min D) is checked through
solver='newton' in checks 4, 5, 8, 12, 13, 18, 19, 21 and 23, where the default meets
the same inputs without it, as their added lines check.
  33. ee_spin_blocks._stream runs fn once per slab of an index range, the last
      slab shorter than the width, with more workers than slabs, and sums the
      outputs exactly; slabs that split their own _ew over as many threads as
      there are workers return; _slab_plan gives one slab when the budget holds
      every index, else workers x width within it, never below one index.
  34. the slab forms ovvv_ijab_pij_slab, ovvv_ia_slab and ooov_ia_slab (no 5, nv 7,
      slabs of width 2 over the second occupied index) equal (1 - P_ij) ovvv_ijab
      restricted to the slab, ovvv_ia and ooov_ia summed over the slabs, to 1e-13
      relative; one slab over every j takes one product against <ic|ab>, both
      spins at once; _ksum splits a product on the main thread and keeps it whole
      off it.
  35. build_operator's pieces['K'] at adc2 and gf2, the kernel streamed over slabs of
      the doubles' second occupied index, equals the V, divide, Vt path at 0.3 Ha
      for powers (1, 2) and () to 1e-12 relative, its doubles norm the flat
      layout's, on one slab, on three of width 2 that the budget _STREAM_GB sets,
      and on those three with two in flight; the folded singlets with and without
      it agree to 1e-9 Ha; a singles vector with a flip parity raises
      NotImplementedError; a slab holds at most _K_LIVE slab blocks per slab index
      (the slope of tracemalloc's peak from width 1 to 2, water cc-pVTZ).
      Checks 10 and 28 take dict(P, K=None), the generic path, as their baseline.
  36. ee_spin_blocks._STREAM_GB follows MBPT_STREAM_GB of the environment at
      import (a subprocess): 0.0001 GB gives one index per slab, unset 16 GB holds
      all 44 indices in one slab; its parser _stream_gb takes unset and a positive
      number silently and an empty, non-numeric, nan, inf or non-positive value as
      16 GB with one warning naming it.

Run: python tests/test_ee_adc_fold.py
"""
import os
import subprocess
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.SingleReference.ADC.eeADC import ee_r_sigma, ee_r_sigma_df
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB

HARTREE_TO_EV = 27.211386245988
WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def water_df(basis='cc-pvdz'):
    """(mf, eps, B, no): RHF water with DF factors B[Q, p, q] in the MO basis."""
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    return mf, eps, B, mol.nelectron // 2


def check_pieces(eps, B, no):
    """Checks 1 and 2: the fold's pieces at adc2 and the gf2 block order."""
    ok = True
    aop, diag, dims, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2',
                                                      pieces=True)
    nv = len(eps) - no
    rng = np.random.default_rng(7)
    vec = rng.standard_normal(dims['nH'])
    y1, Y = ee_r_sigma.to_blocks(vec, no, nv, 'adc2')
    be = P['be']
    # the fold's reading of the operator: [M y + Vt Y ; V y + D * Y]
    w1 = be.ein('iajb,jb->ia', P['M'], y1) + P['Vt'](Y)
    W = P['V'](y1) + be.scale(Y, P['D'])
    rebuilt = ee_r_sigma.from_blocks(w1, W, no, nv, 'adc2')
    err = float(np.max(np.abs(rebuilt - aop(vec))))
    ok &= check(err < 1e-12, 'pieces rebuild aop(vec) at adc2', f'max |diff| {err:.1e}')
    ok &= check(P['D'].shape == (no, no, nv, nv), 'D has shape (no, no, nv, nv)')
    for level in ('adc1', 'adc2x', 'adc3'):
        try:
            ee_r_sigma_df.build_operator(eps, B, no, level=level, pieces=True)
            ok &= check(False, f'pieces=True raises at {level}')
        except ValueError as exc:
            ok &= check(level in str(exc), f'pieces=True raises at {level}',
                        str(exc)[:60])
    return ok


def check_gf2(eps, B, no):
    """eq 53 of Monino-Loos 2023: ADC(2) with the singles block cut back to A^HF."""
    ok = True
    nv = len(eps) - no
    _, _, d_gf2, P2 = ee_r_sigma_df.build_operator(eps, B, no, level='gf2',
                                                   pieces=True)
    _, _, d_adc2, P1 = ee_r_sigma_df.build_operator(eps, B, no, level='adc2',
                                                    pieces=True)
    aop1, _, _ = ee_r_sigma_df.build_operator(eps, B, no, level='adc1')
    ok &= check(d_gf2 == d_adc2, 'gf2 dimensions equal adc2')
    ok &= check(ee_r_sigma._BLOCK_ORDERS['gf2'] == (1, 1, 0),
                'gf2 block order (1, 1, 0)')
    be = P2['be']
    rng = np.random.default_rng(11)
    vec = rng.standard_normal(d_adc2['nH'])
    y1, Y = ee_r_sigma.to_blocks(vec, no, nv, 'adc2')
    # singles block: gf2's M on y1 equals the adc1 operator on the singles part
    w_gf2 = ee_r_sigma.from_blocks(be.ein('iajb,jb->ia', P2['M'], y1), SB(), no, nv,
                                   'adc1')
    w_adc1 = aop1(vec[:2 * no * nv])
    err = float(np.max(np.abs(w_gf2 - w_adc1)))
    ok &= check(err < 1e-12, 'gf2 singles block equals adc1 (A^HF)', f'{err:.1e}')
    # couplings and doubles diagonal: identical to adc2's
    ok &= check(_sb_close(P2['V'](y1), P1['V'](y1)), 'gf2 coupling V equals adc2')
    ok &= check(_sb_close(P2['Vt'](Y), P1['Vt'](Y)), 'gf2 coupling Vt equals adc2')
    ok &= check(np.allclose(P2['D'], P1['D'], atol=0, rtol=0), 'gf2 D equals d_ijab')
    return ok


def check_folded_operator(mf, eps, B, no):
    """Check 3: A_eff(w) at the full solve's roots."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    nv = len(eps) - no
    n_s = 2 * no * nv
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    # the full solve's lowest singlet: Davidson, since the channel (6365) is far
    # above DENSE_LIMIT; its residual 1e-10 sits well under every tolerance here
    e_full, Z = solve_ee_adc(mf, level='adc2', nroots=1, df=True, spin='singlet',
                             conv_tol=1e-10)
    w = float(e_full[0])
    matvec, dmatvec, diag_s, embed, restrict = ee_fold.folded_operator(P, w,
                                                                       spin='singlet')
    # singles part of the full eigenvector (solve_ee_adc returns it embedded),
    # in the channel basis
    z = np.asarray(Z[:, 0])
    y = restrict(z[:n_s])
    y = y / np.linalg.norm(y)
    res = float(np.linalg.norm(matvec(y) - w * y))
    ok &= check(res < 1e-8, 'A_eff(w) y = w y on the full root', f'residual {res:.1e}')
    # dense build equals the matrix-free action
    n = diag_s.size
    A = ee_fold.dense_effective(matvec, n)
    err = float(np.max(np.abs(A @ y - matvec(y))))
    ok &= check(err < 1e-12, 'dense_effective equals matvec', f'{err:.1e}')
    ok &= check(np.allclose(A, A.T, atol=1e-10), 'A_eff(w) is symmetric')
    # omega=None is the bare singles block
    m0, _, _, _, _ = ee_fold.folded_operator(P, None, spin='singlet')
    err0 = float(np.linalg.norm(m0(y) - restrict(
        ee_fold.singles_sb_to_flat(P['be'].ein('iajb,jb->ia', P['M'],
                                               ee_fold.singles_flat_to_sb(
                                                   embed(y), no, nv)), no, nv))))
    ok &= check(err0 < 1e-12, 'omega=None gives M alone', f'{err0:.1e}')
    # boundary
    for bad, label in ((np.ones(n + 1), 'wrong length'),
                       (np.ones(n, dtype=complex), 'complex dtype')):
        try:
            matvec(bad)
            ok &= check(False, f'{label} raises ValueError')
        except ValueError:
            ok &= check(True, f'{label} raises ValueError')
    return ok


def check_solve_folded(mf, eps, B, no):
    """Check 4: solve_folded at adc2 against the full Davidson roots."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    nv = len(eps) - no
    n_s = 2 * no * nv
    ev = HARTREE_TO_EV
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    for spin in ('singlet', 'triplet'):
        e_full, Z = solve_ee_adc(mf, level='adc2', nroots=3, df=True, spin=spin,
                                 conv_tol=1e-10)
        res = ee_fold.solve_folded(P, 3, spin=spin)
        res_n = ee_fold.solve_folded(P, 3, spin=spin, solver='newton')
        for r in range(3):
            d = abs(res.omega[r] - e_full[r]) * ev
            ok &= check(d < 1e-5, f'{spin} root {r}: folded equals full ADC(2)',
                        f'{res.omega[r] * ev:.6f} vs {e_full[r] * ev:.6f} eV, '
                        f'|d| {d:.1e} eV')
            z = np.asarray(Z[:, r])
            t1_full = float(z[:n_s] @ z[:n_s]) / float(z @ z)
            ok &= check(abs(res.t1[r] - t1_full) < 1e-4, f'{spin} root {r}: T1',
                        f'{res.t1[r]:.5f} vs {t1_full:.5f}')
            newton = (res_n.converged[r] and res_n.steps[r] <= 8
                      and res_n.loop[r] == 'newton')
            ok &= check(newton, f'{spin} root {r}: Newton under eight steps',
                        f'{res_n.steps[r]} steps, loop {res_n.loop[r]}')
        # the fixed-point loop reaches the same roots, from the seeds and as the
        # default's polish
        res_fp = ee_fold.solve_folded(P, 3, spin=spin, t_min=1.0, solver='newton')
        d = float(np.max(np.abs(res_fp.omega - e_full[:3]))) * ev
        ok &= check(d < 1e-5 and all(lp == 'fixed' for lp in res_fp.loop),
                    f'{spin}: fixed-point loop reaches the same roots',
                    f'|d| {d:.1e} eV')
        res_fp = ee_fold.solve_folded(P, 3, spin=spin, t_min=1.0)
        d = float(np.max(np.abs(res_fp.omega - e_full[:3]))) * ev
        ok &= check(d < 1e-5 and bool(res_fp.converged.all()),
                    f"{spin}: the default's polish in the fixed-point loop too",
                    f'|d| {d:.1e} eV')
    # short channel: H2/STO-3G has one triplet single
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf2 = scf.RHF(mol).density_fit()
    mf2.conv_tol = 1e-12
    mf2.kernel()
    eps2 = np.asarray(get_orbital_energies(mf2, representation='spatial'), float)
    B2 = DFIntegrals.from_scf(mol, mf2).B_aa
    _, _, _, P2 = ee_r_sigma_df.build_operator(eps2, B2, 1, level='adc2',
                                               pieces=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res2 = ee_fold.solve_folded(P2, 3, spin='triplet')
    fewer = any('fewer' in str(w.message) for w in caught)
    ok &= check(res2.omega.size == 1 and fewer,
                'short channel warns and returns what it has',
                f'{res2.omega.size} root(s), {len(caught)} warning(s)')
    # nocc = nvirt = 1: every reshape of the layout is degenerate. HeH+, not H2,
    # whose g/u symmetry cuts the one single from the one double
    mol3 = gto.M(atom='He 0 0 0; H 0 0 0.77', basis='sto-3g', charge=1, verbose=0)
    mf3 = scf.RHF(mol3).density_fit()
    mf3.conv_tol = 1e-12
    mf3.kernel()
    eps3 = np.asarray(get_orbital_energies(mf3, representation='spatial'), float)
    B3 = DFIntegrals.from_scf(mol3, mf3).B_aa
    _, _, _, P3 = ee_r_sigma_df.build_operator(eps3, B3, 1, level='adc2',
                                               pieces=True)
    res3 = ee_fold.solve_folded(P3, 1, spin='singlet')
    e3, _ = solve_ee_adc(mf3, level='adc2', nroots=1, df=True, spin='singlet',
                         conv_tol=1e-10)
    d = abs(float(res3.omega[0]) - float(e3[0])) * ev
    ok &= check(d < 1e-5 and res3.t1[0] < 1.0,
                'HeH+ singlet (nocc = nvirt = 1): folded equals the full solve',
                f'|d| {d:.1e} eV, T1 {res3.t1[0]:.6f}')
    return ok


# exact Td: every coordinate is +-0.6276, so the T2 triple is degenerate to
# machine precision (a rounded C3v NH3 splits its E pair by 2e-4 eV)
CH4 = ('C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; '
       'H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276')


def check_degenerate_set():
    """CH4 / cc-pVDZ: the degenerate T2 singlet triple is followed without a
    skip, each partner its own vector."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    ev = HARTREE_TO_EV
    mol = gto.M(atom=CH4, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    no = mol.nelectron // 2
    e_full, _ = solve_ee_adc(mf, level='adc2', nroots=4, df=True, spin='singlet',
                             conv_tol=1e-10)
    e_full = np.asarray(e_full)
    gaps = np.abs(np.diff(e_full)) * ev
    ok &= check(gaps[0] < 1e-6 and gaps[1] < 1e-6,
                'a degenerate singlet triple among the four lowest',
                f'gaps {gaps[0]:.1e}, {gaps[1]:.1e} eV')
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    res = ee_fold.solve_folded(P, 4, spin='singlet')
    # a level cut by nroots comes back whole, so nout may exceed four
    d = np.abs(res.omega[:4] - e_full) * ev
    ok &= check(float(d.max()) < 1e-5, 'all four folded roots equal the full solve',
                f'max |d| {d.max():.1e} eV, nout {res.omega.size}')
    for dense_limit, route in ((2000, 'dense'), (0, 'Davidson')):
        # the partners' T1 agree to the vectors' accuracy, set by tol_residual
        res = ee_fold.solve_folded(P, 4, spin='singlet', dense_limit=dense_limit,
                                   tol_residual=1e-9)
        d = np.abs(res.omega[:4] - e_full) * ev
        ok &= check(float(d.max()) < 1e-5,
                    f'{route}: all four folded roots equal the full solve',
                    f'max |d| {d.max():.1e} eV, nout {res.omega.size}')
        # within the triple; distinct roots sit at different w, so their singles
        # parts need not be orthogonal
        S = res.y[:, :3].T @ res.y[:, :3]
        ov = float(np.max(np.abs(S - np.eye(3))))
        ok &= check(ov < 1e-10, f'{route}: the triple partners are orthonormal',
                    f'max |y^T y - 1| {ov:.1e}')
        ok &= check(np.ptp(res.omega[:3]) == 0.0
                    and len(set(res.level[:3].tolist())) == 1
                    and res.level[3] != res.level[0],
                    f'{route}: the triple is one level at one omega',
                    f'levels {res.level.tolist()}')
        ok &= check(float(np.ptp(res.t1[:3])) < 1e-10,
                    f'{route}: the partners share T1',
                    f'spread {np.ptp(res.t1[:3]):.1e}')
        ok &= check(np.allclose(res.x1, res.y * np.sqrt(res.t1)[None, :],
                                atol=1e-15, rtol=0), f'{route}: x1 = sqrt(T1) y')
        cut = ee_fold.solve_folded(P, 2, spin='singlet', dense_limit=dense_limit)
        dcut = np.abs(cut.omega[:3] - res.omega[:3]) * ev if cut.omega.size >= 3 \
            else [np.inf]
        ok &= check(cut.omega.size == 3 and float(np.max(dcut)) < 1e-5,
                    f'{route}: nroots = 2 returns the whole triple',
                    f'nout {cut.omega.size}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            short = ee_fold.solve_folded(P, 3, spin='singlet', max_newton=1,
                                         dense_limit=dense_limit, solver='newton')
        lv = any('level' in str(x.message) and 'not converged' in str(x.message)
                 for x in caught)
        ok &= check(lv and not short.converged[:3].any(),
                    f'{route}: an exhausted level leaves every partner unconverged')
    return ok


def check_routes_and_channels(mf, eps, B, no):
    """The Davidson branch of the fold (dense_limit=0) against the dense one, and
    spin=None against the full solve over both channels."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    ev = HARTREE_TO_EV
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    for spin in ('singlet', 'triplet'):
        a = ee_fold.solve_folded(P, 3, spin=spin, dense_limit=2000)
        b = ee_fold.solve_folded(P, 3, spin=spin, dense_limit=0)
        d = float(np.max(np.abs(a.omega - b.omega))) * ev
        ok &= check(d < 1e-7 and b.converged.all(),
                    f'{spin}: Davidson branch equals the dense branch',
                    f'|d| {d:.1e} eV')
    e_full, _ = solve_ee_adc(mf, level='adc2', nroots=6, df=True, conv_tol=1e-10)
    res = ee_fold.solve_folded(P, 6, spin=None)
    d = float(np.max(np.abs(res.omega - np.asarray(e_full)))) * ev
    ok &= check(d < 1e-5, 'spin=None: six folded roots equal the full solve',
                f'|d| {d:.1e} eV')
    return ok


def check_gf2_solves(mf, eps, B, no):
    """gf2 end to end: the fold against the full gf2 channel solve, and the dense
    integral route against the DF route with exact factors."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    ev = HARTREE_TO_EV
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='gf2', pieces=True)
    for spin in ('singlet', 'triplet'):
        e_full, _ = solve_ee_adc(mf, level='gf2', nroots=3, df=True, spin=spin,
                                 conv_tol=1e-10)
        res = ee_fold.solve_folded(P, 3, spin=spin)
        d = float(np.max(np.abs(res.omega - np.asarray(e_full)))) * ev
        ok &= check(d < 1e-5 and res.converged.all(),
                    f'gf2 {spin}: folded equals the full solve', f'|d| {d:.1e} eV')
    mol = mf.mol
    mf0 = scf.RHF(mol)
    mf0.conv_tol = 1e-12
    mf0.kernel()
    e_dense, _ = solve_ee_adc(mf0, level='gf2', nroots=3, df=False, conv_tol=1e-10)
    e_exact, _ = solve_ee_adc(mf0, level='gf2', nroots=3, df=True, auxbasis='exact',
                              conv_tol=1e-10)
    d = float(np.max(np.abs(np.asarray(e_dense) - np.asarray(e_exact)))) * ev
    ok &= check(d < 1e-6, 'gf2: dense integral route equals DF with exact factors',
                f'|d| {d:.1e} eV')
    mf6 = water_df('6-31g')[0]
    e_sf, _ = solve_ee_adc(mf6, level='gf2', nroots=4, df=True, conv_tol=1e-10)
    e_u, _ = solve_ee_adc(mf6.to_uhf(), level='gf2', nroots=4, df=True,
                          route='unrestricted', conv_tol=1e-10)
    d = float(np.max(np.abs(np.asarray(e_sf) - np.asarray(e_u)))) * ev
    ok &= check(d < 1e-6, 'gf2: the unrestricted route on an RHF reference equals the '
                'spin-free one, both channels (6-31G)', f'|d| {d:.1e} eV')
    mf_s = scf.RHF(gto.M(atom=WATER, basis='sto-3g', verbose=0))
    mf_s.conv_tol = 1e-12
    mf_s.kernel()
    for spin in ('singlet', 'triplet'):
        # with spin set the spin-orbital route projects its dense supermatrix
        e_sf, _ = solve_ee_adc(mf_s, level='gf2', nroots=3, spin=spin,
                               matrix_free=False)
        e_so, _ = solve_ee_adc(mf_s, level='gf2', nroots=3, spin=spin,
                               route='spinorbital')
        d = float(np.max(np.abs(np.sort(e_so) - np.sort(e_sf)))) * ev
        ok &= check(d < 1e-6, f'gf2 {spin}: the spin-orbital route equals the '
                    'spin-free one (STO-3G)', f'|d| {d:.1e} eV')
    # spin=None and dense_limit 0: the matrix-free ee_u_sigma_full by Davidson;
    # 10 roots end on whole levels (Ms = 0, +-1 triplets come in threes)
    e_d, _ = solve_ee_adc(mf_s, level='gf2', nroots=10, route='spinorbital',
                          matrix_free=False)
    e_m, _ = solve_ee_adc(mf_s, level='gf2', nroots=10, route='spinorbital',
                          dense_limit=0, conv_tol=1e-10)
    d = float(np.max(np.abs(np.sort(e_m) - np.sort(e_d)))) * ev
    ok &= check(d < 1e-6, 'gf2: the spin-orbital Davidson equals its dense '
                'supermatrix, ten lowest of both channels (STO-3G)', f'|d| {d:.1e} eV')
    return ok


def check_boundaries(mf, eps, B, no):
    """The refusals and the step budget; inputs left as they were handed in."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    try:
        ee_r_sigma_df.build_operator(eps, B, no, level='adc2', parity=1.0,
                                     pieces=True)
        ok &= check(False, 'pieces=True with parity raises ValueError')
    except ValueError as exc:
        ok &= check('parity' in str(exc), 'pieces=True with parity raises ValueError')
    try:
        solve_ee_adc(mf, level='gf2', nroots=1, df=True, en_dress=True)
        ok &= check(False, 'en_dress at gf2 raises ValueError')
    except ValueError as exc:
        ok &= check('en_dress' in str(exc), 'en_dress at gf2 raises ValueError',
                    str(exc)[:60])
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    D0, M0 = P['D'].copy(), P['M'].get('aaaa').copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(P, 1, spin='singlet', max_newton=1,
                                   solver='newton')
    budget = any('not converged' in str(w.message) for w in caught)
    ok &= check(not res.converged[0] and budget,
                'an exhausted step budget reports converged False and warns',
                f'converged {res.converged[0]}, steps {res.steps[0]}')
    named = any(f'{res.omega[0]:.8f}' in str(w.message) for w in caught)
    ok &= check(named, 'the budget warning names the root by its omega in res.omega')
    matvec, _, diag_s, _, _ = ee_fold.folded_operator(P, 0.3, spin='singlet')
    u = np.linspace(-1.0, 1.0, diag_s.size)
    u0 = u.copy()
    matvec(u)
    same = (np.array_equal(u, u0) and np.array_equal(P['D'], D0)
            and np.array_equal(P['M'].get('aaaa'), M0))
    ok &= check(same, 'matvec and solve_folded leave their inputs untouched')
    empty = ee_fold.solve_folded(P, 0, spin='singlet')
    ok &= check(empty.omega.size == 0 and empty.y.shape == (diag_s.size, 0),
                'nroots = 0 returns an empty result', f'y {empty.y.shape}')
    try:
        ee_fold.solve_folded(P, -1, spin='singlet')
        ok &= check(False, 'nroots < 0 raises ValueError')
    except ValueError as exc:
        ok &= check('nroots' in str(exc), 'nroots < 0 raises ValueError', str(exc)[:60])
    return ok


def check_level_crossing():
    """The level path warns when its partners leave their previous span: the
    smallest singular value of sum_p Z_pk Y_pc below 0.5, silent when aligned."""
    from src.SingleReference.ADC.eeADC import ee_fold
    m = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    P = _synthetic_pieces(m, np.zeros((1, 6)), np.array([2.0]))
    spread = np.zeros((6, 2))
    spread[0, 0] = 1.0
    spread[1:, 1] = 1.0 / np.sqrt(5.0)               # |overlap| 0.45 with each state
    got = {}
    for name, Y in (('spread', spread), ('aligned', np.eye(6)[:, :2])):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ee_fold._eig_level(P, 0.3, 'singlet', Y, True, 1e-9, 'test')
        got[name] = any('crossing' in str(x.message) for x in caught)
    return check(got['spread'] and not got['aligned'],
                 'a level leaving its span warns; an aligned level does not',
                 str(got))


def check_diis_step():
    """The fixed-point extrapolation is well posed at every history length: a
    1e-14 relative change of the residuals moves it by rounding only, and on a
    residual linear in w it lands on the root."""
    from src.SingleReference.ADC.eeADC.ee_fold import _diis_step
    ok = True
    w_star = 0.3
    om = np.array([0.40, 0.36, 0.33, 0.315, 0.308, 0.304])
    for m in range(2, 7):
        o = om[:m]
        e = (w_star - o) + 0.3 * (w_star - o) ** 2        # a curved residual
        a = _diis_step(list(o), list(e))
        b = _diis_step(list(o), list(e * (1.0 + 1e-14 * np.arange(1, m + 1))))
        lin = _diis_step(list(o), list(0.7 * (w_star - o)))
        ok &= check(abs(a - b) < 1e-10 and abs(lin - w_star) < 1e-12,
                    f'fixed-point step well posed at history {m}',
                    f'|d| perturbed {abs(a - b):.1e}, linear {abs(lin - w_star):.1e}')
    return ok


def _spin_orbital_asym(eps, B, no):
    """<pq||rs>, shape (nso,)*4, index order (p, q, r, s), and ε, shape (nso,), in
    spin orbitals ordered occ alpha, occ beta, vir alpha, vir beta, from B[Q, p, q]."""
    nmo = eps.size
    nv = nmo - no
    occ, vir = np.arange(no), np.arange(no, nmo)
    sp = np.r_[occ, occ, vir, vir]
    sg = np.r_[np.zeros(no), np.ones(no), np.zeros(nv), np.ones(nv)]
    eri = np.einsum('Qpq,Qrs->pqrs', B, B)[np.ix_(sp, sp, sp, sp)]      # (pq|rs)
    same = (sg[:, None] == sg[None, :]).astype(float)
    # <pq|rs> = (pr|qs) δ(σp σr) δ(σq σs)
    phys = (eri.transpose(0, 2, 1, 3) * same[:, None, :, None]
            * same[None, :, None, :])
    return phys - phys.transpose(0, 1, 3, 2), eps[sp]


def _eq53_effective(g, e, No, w, with_eq57):
    """A^HF (eq 54a) [+ Abar (eq 57)] [+ Xi(w) (eq 56)] of Monino and Loos, shape
    (No, Nv, No, Nv), index order (i, a, j, b); w None leaves out eq 56."""
    def ein(sub, *ops):
        return np.einsum(sub, *ops, optimize=True)
    o, v = slice(0, No), slice(No, None)
    eo, ev = e[o], e[v]
    Nv = ev.size
    ooov, vovv, oovv, vvoo = g[o, o, o, v], g[v, o, v, v], g[o, o, v, v], g[v, v, o, o]
    # A_ia,jb = (ε_a - ε_i) δ_ij δ_ab + <ib||aj>
    A = ein('ibaj->iajb', g[o, v, v, o]).copy()
    for i in range(No):
        A[i, :, i, :] += np.diag(ev - eo[i])
    if with_eq57:
        # δ_ij ¼ sum_klc <ac||kl><kl||bc> [1/(ε_a - ε_k + ε_c - ε_l) + (a -> b)]
        r = 1.0 / (ev[:, None, None, None] - eo[None, :, None, None]
                   + ev[None, None, :, None] - eo[None, None, None, :])     # [a,k,c,l]
        t = 0.25 * (ein('ackl,klbc,akcl->ab', vvoo, oovv, r)
                    + ein('ackl,klbc,bkcl->ab', vvoo, oovv, r))
        for i in range(No):
            A[i, :, i, :] += t
        # δ_ab ¼ sum_kcd <cd||ik><jk||cd> [1/(ε_c - ε_i + ε_d - ε_k) + (i -> j)]
        r = 1.0 / (ev[None, :, None, None] - eo[:, None, None, None]
                   + ev[None, None, :, None] - eo[None, None, None, :])     # [i,c,d,k]
        t = 0.25 * (ein('cdik,jkcd,icdk->ij', vvoo, oovv, r)
                    + ein('cdik,jkcd,jcdk->ij', vvoo, oovv, r))
        for a in range(Nv):
            A[:, a, :, a] += t
        # - ½ sum_kc <ac||ik><jk||bc> [1/(ε_a - ε_i + ε_c - ε_k) + (ia -> jb)]
        r = 1.0 / (ev[None, :, None, None] - eo[:, None, None, None]
                   + ev[None, None, None, :] - eo[None, None, :, None])     # [i,a,k,c]
        A -= 0.5 * (ein('acik,jkbc,iakc->iajb', vvoo, oovv, r)
                    + ein('acik,jkbc,jbkc->iajb', vvoo, oovv, r))
    if w is None:
        return A
    # δ_ab ½ sum_klc <kl||ic><kl||jc> / (w - (ε_a + ε_c - ε_k - ε_l)), r[a,k,l,c]
    r = 1.0 / (w - (ev[:, None, None, None] + ev[None, None, None, :]
                    - eo[None, :, None, None] - eo[None, None, :, None]))
    t = 0.5 * ein('klic,kljc,aklc->ija', ooov, ooov, r)
    for a in range(Nv):
        A[:, a, :, a] += t[:, :, a]
    # δ_ij ½ sum_kcd <ak||cd><bk||cd> / (w - (ε_c + ε_d - ε_k - ε_i)), r[i,k,c,d]
    r = 1.0 / (w - (ev[None, None, :, None] + ev[None, None, None, :]
                    - eo[None, :, None, None] - eo[:, None, None, None]))
    t = 0.5 * ein('akcd,bkcd,ikcd->iab', vovv, vovv, r)
    for i in range(No):
        A[i, :, i, :] += t[i]
    # - sum_kc <jc||ik><ka||cb> / (w - (ε_b + ε_c - ε_k - ε_i)), r[i,b,k,c];
    # - sum_kc <jk||ic><ca||kb> / (w - (ε_a + ε_c - ε_k - ε_j)), the same r at [j,a,k,c]
    r = 1.0 / (w - (ev[None, :, None, None] + ev[None, None, None, :]
                    - eo[None, None, :, None] - eo[:, None, None, None]))
    A -= ein('jcik,kacb,ibkc->iajb', g[o, v, o, o], g[o, v, v, v], r)
    A -= ein('jkic,cakb,jakc->iajb', ooov, g[v, v, o, v], r)
    # + ½ sum_kl <aj||kl><lk||bi> / (w - (ε_a + ε_b - ε_k - ε_l)), r[a,b,k,l]
    r = 1.0 / (w - (ev[:, None, None, None] + ev[None, :, None, None]
                    - eo[None, None, :, None] - eo[None, None, None, :]))
    A += 0.5 * ein('ajkl,lkbi,abkl->iajb', g[v, o, o, o], g[o, o, v, o], r)
    # + ½ sum_cd <aj||cd><dc||bi> / (w - (ε_c + ε_d - ε_i - ε_j)), r[i,j,c,d]
    r = 1.0 / (w - (ev[None, None, :, None] + ev[None, None, None, :]
                    - eo[:, None, None, None] - eo[None, :, None, None]))
    A += 0.5 * ein('ajcd,dcbi,ijcd->iajb', vovv, g[v, v, v, o], r)
    return A


def _ms0(A4, no, nv):
    """(No, Nv, No, Nv) spin-orbital block -> the fold's flat singles layout, shape
    (2 no nv, 2 no nv): the M_s = 0 singles, aa then bb, (i, a) order."""
    occ = np.r_[np.repeat(np.arange(no), nv), np.repeat(np.arange(no) + no, nv)]
    vir = np.r_[np.tile(np.arange(nv), no), np.tile(np.arange(nv) + nv, no)]
    idx = occ * 2 * nv + vir
    n = A4.shape[0] * A4.shape[1]
    return A4.reshape(n, n)[np.ix_(idx, idx)]


def check_eq53_transcription(eps, B, no):
    """Eq 53 of Monino and Loos, JCP 159, 034105 (2023), as printed: eq 54a plus the
    six terms of eq 56, transcribed in spin orbitals on the same DF factors, against
    the gf2 A_eff element by element; with eq 57 added, against the adc2 A_eff."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    g, e = _spin_orbital_asym(eps, B, no)
    nv = eps.size - no
    for level in ('gf2', 'adc2'):
        _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level=level, pieces=True)
        for w in (None, 0.25):
            A = ee_fold.dense_effective(ee_fold.folded_operator(P, w)[0], 2 * no * nv)
            ref = _ms0(_eq53_effective(g, e, 2 * no, w, level == 'adc2'), no, nv)
            d = float(np.max(np.abs(A - ref)))
            terms = ('eq 54a' + (' + eq 57' if level == 'adc2' else '')
                     + ('' if w is None else ' + eq 56'))
            tag = 'M' if w is None else f'w = {w} Ha'
            ok &= check(d < 1e-10, f'{level} {tag}: A_eff equals {terms} transcribed',
                        f'max |d| {d:.1e} Ha')
    return ok


def check_gw_hook(eps, B, no):
    """'gw' is a fold level; an optional pieces['dnorm2'] is the doubles norm dmatvec
    takes, and without it the adc2 norm is the flat layout's, bitwise."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    nv = B.shape[1] - no
    ok &= check('gw' in ee_fold.FOLD_LEVELS, "'gw' is a fold level")
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    P = dict(P, K=None)                     # the generic path, not the streamed kernel
    _, dmv, diag_s, embed, _ = ee_fold.folded_operator(P, 0.3, spin='singlet')
    u = np.linspace(-1.0, 1.0, diag_s.size)
    u /= np.linalg.norm(u)
    y1 = ee_fold.singles_flat_to_sb(embed(u), no, nv)
    Yf = ee_fold.doubles_sb_to_flat(P['be'].divide(P['V'](y1), P['D'] - 0.3), no, nv)
    ok &= check(dmv(u) == float(Yf @ Yf), 'adc2: dmatvec is the flat-layout norm')
    calls = []

    def dnorm2(Y):
        calls.append(1)
        f = ee_fold.doubles_sb_to_flat(Y, no, nv)
        return float(f @ f)

    _, dmv2, _, _, _ = ee_fold.folded_operator(dict(P, dnorm2=dnorm2), 0.3,
                                               spin='singlet')
    ok &= check(dmv2(u) == dmv(u) and len(calls) == 1,
                "pieces['dnorm2'] is the norm dmatvec takes")
    return ok


def check_level_helpers():
    """_levels groups by spread, max minus min, without chaining; _duplicates flags
    two copies of one root and passes distinct roots whatever their singles overlap."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    e = np.array([0.1, 0.1 + 4e-9, 0.1 + 8e-9, 0.1 + 1.2e-8, 0.3])
    runs = ee_fold._levels(e)
    ok &= check(runs == [(0, 3), (3, 4), (4, 5)],
                'levels: spread at most 1e-8 Ha, no chaining', str(runs))

    def dot(a, b):
        return float(a @ b)

    t1 = np.array([0.5, 0.5])
    # distinct roots, T1 = 0.5 each: singles overlap -0.8, doubles overlap +0.8
    y = np.array([[1.0, -0.8], [0.0, 0.6]])
    Yt = [np.array([1.0, 0.0]), np.array([0.8, 0.6])]
    got = ee_fold._duplicates(np.array([0.2, 0.2 + 1e-7]), y, t1, Yt, dot, 1e-6)
    ok &= check(got == [], 'distinct roots with singles overlap 0.8 are no duplicate')
    y2 = np.array([[1.0, -1.0], [0.0, 0.0]])
    Yt2 = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
    got = ee_fold._duplicates(np.array([0.2, 0.2 + 1.5e-6]), y2, t1, Yt2, dot, 1e-6)
    ok &= check(got == [(0, 1)],
                'two copies of one root, 1.5 tol_omega apart, are one', str(got))
    got = ee_fold._duplicates(np.array([0.2, 0.3]), y2, t1, Yt2, dot, 1e-6)
    ok &= check(got == [], 'copies at distant omega are not compared')
    return ok


def _synthetic_pieces(m, C, D):
    """Fold pieces with no = 1, nv = len(m): M = diag(m) (or m itself, when m is a
    symmetric matrix) on both spin blocks, no cross-spin block; V = C (doubles x
    singles) per spin block; D the doubles diagonal; level 'gw' with its plain
    doubles norm. In the singlet channel the supermatrix is
    [[diag(m), C^T], [C, diag(D)]]."""
    from src.SingleReference.ADC.eeADC import ee_equations
    nv = len(m)
    Ma = np.zeros((1, nv, 1, nv))
    Ma[0, :, 0, :] = np.diag(m) if np.ndim(m) == 1 else m
    M = SB({'aaaa': Ma, 'bbbb': Ma.copy(), 'aabb': np.zeros_like(Ma),
            'bbaa': np.zeros_like(Ma)})

    def V(y1):
        return SB({s: C @ y1.get(s).ravel() for s in ('aa', 'bb')})

    def Vt(Y):
        return SB({s: (C.T @ Y.get(s)).reshape(1, nv) for s in ('aa', 'bb')})

    def dnorm2(Y):
        return float(sum(Y.get(s) @ Y.get(s) for s in ('aa', 'bb')))

    # the dense branch unless a check passes dense_limit: these pieces are tiny
    return {'M': M, 'V': V, 'Vt': Vt, 'D': np.asarray(D, float), 'dnorm2': dnorm2,
            'level': 'gw', 'no': 1, 'nv': nv, 'be': ee_equations.SPIN_BLOCKED,
            'dense_limit': 2000}


def _synthetic_exact(m, C, D):
    """Eigenpairs (w, v) of the singlet-channel supermatrix of _synthetic_pieces."""
    ns, nd = len(m), len(D)
    H = np.zeros((ns + nd, ns + nd))
    H[:ns, :ns] = np.diag(m) if np.ndim(m) == 1 else m
    H[ns:, :ns] = C
    H[:ns, ns:] = C.T
    H[ns:, ns:] = np.diag(D)
    return np.linalg.eigh(H)


def check_collapsing_pair():
    """A distinct pair 7.05e-7 Ha apart whose seeds lie 0.22 Ha apart in M: both
    seeds converge onto the lower root. The fold reports the duplicate, drops the
    copy, keeps one exact root and solves the missed one by its branch index (one
    root short of nroots); on the Davidson branch with tol_omega far below
    tol_residual the duplicate window tied to tol_residual catches it."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    D = np.array([1.0, 1.4])
    C = np.array([[0.3, 0.3, 0.0], [0.3, -0.54, 0.0]])
    s = np.sum(C ** 2 / (D[:, None] - 0.5), axis=0)
    m = np.array([0.5 + s[0], 0.5 + s[1] + 1.2e-6, 1.5])
    w, v = _synthetic_exact(m, C, D)
    pair = np.sort(w[np.abs(w - 0.5) < 1e-3])
    ok &= check(pair.size == 2 and 5e-7 < pair[1] - pair[0] < 1e-6,
                'the synthetic pair is distinct and closer than 1e-6 Ha',
                f'split {pair[1] - pair[0]:.2e} Ha')
    P = _synthetic_pieces(m, C, D)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(P, 2, spin='singlet', tol_omega=1e-12,
                                   solver='newton')
    dup = any('landed on one root' in str(x.message) for x in caught)
    ok &= check(dup and res.omega.size == 2 and bool(res.converged.all()),
                'the collapse is reported: a warning, the copy dropped',
                f'nout {res.omega.size}, converged {res.converged.tolist()}')
    u = v[:3, int(np.argmin(np.abs(w - res.omega[0])))]
    err = 1.0 - abs(float(u @ res.y[:, 0])) / np.linalg.norm(u)
    ok &= check(abs(res.omega[0] - pair[0]) < 1e-10 and err < 1e-8,
                'the kept root is the exact lower root, vector included',
                f'|dw| {abs(res.omega[0] - pair[0]):.1e} Ha, 1-|y.u| {err:.1e}')
    # one root short of nroots: the second branch is solved by its index
    d = abs(res.omega[-1] - pair[1])
    ok &= check(d < 1e-10 and res.loop[-1] == 'index',
                'the missed upper root is solved by its branch index',
                f'|dw| {d:.1e} Ha, loop {res.loop}')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(P, 2, spin='singlet', tol_omega=1e-12,
                                   dense_limit=0, solver='newton')
    dup = any('landed on one root' in str(x.message) for x in caught)
    d = np.abs(res.omega - pair).max() if res.omega.size == 2 else np.inf
    ok &= check(dup and d < 1e-8,
                'Davidson, tol_omega 1e-12 at tol_residual 1e-6: the collapse is '
                'caught and both roots come back', f'max |dw| {d:.1e} Ha')
    for dl in (2000, 0):
        # the root-wise Davidson follows both branches: no seed collapses
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ee_fold.solve_folded(P, 2, spin='singlet', tol_omega=1e-12,
                                       dense_limit=dl)
        d = np.abs(res.omega - pair).max() if res.omega.size == 2 else np.inf
        ok &= check(d < 1e-8 and not caught,
                    f'dense_limit {dl}: the default finds both roots of the pair, '
                    'no warning', f'max |dw| {d:.1e} Ha, {len(caught)} warning(s)')
    return ok


def check_split_level():
    """A pair degenerate in M that V splits by 7.8e-6 Ha in A_eff, along
    directions fixed in omega at 45 degrees to M's seeds: one level at the start;
    the fold warns, re-solves per root, and both roots are supermatrix eigenvalues."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    ca, cb, r2 = 0.2, np.sqrt(0.04 - 5e-6), 1.0 / np.sqrt(2.0)
    C = np.array([[ca * r2, ca * r2, 0.0], [cb * r2, -cb * r2, 0.0]])
    D = np.array([1.0, 1.0])
    m = np.array([0.5, 0.5, 1.5])
    w, _ = _synthetic_exact(m, C, D)
    pair = np.sort(w[(w > 0.3) & (w < 0.5)])
    P = _synthetic_pieces(m, C, D)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(P, 2, spin='singlet', tol_omega=1e-12,
                                   solver='newton')
    split = any('not degenerate' in str(x.message) for x in caught)
    ok &= check(split and pair.size == 2,
                'a level split in A_eff warns and is re-solved per root',
                f'split {pair[1] - pair[0]:.1e} Ha')
    d = np.abs(np.sort(res.omega)[:2] - pair) if res.omega.size >= 2 else [np.inf]
    ok &= check(float(np.max(d)) < 1e-10 and bool(res.converged.all()),
                'both roots are the supermatrix eigenvalues',
                f'max |dw| {np.max(d):.1e} Ha')
    lv = res.level[:2].tolist()
    ok &= check(len(lv) == 2 and lv[0] != lv[1],
                'the split roots are levels of their own', f'level {lv}')
    # the joint solve alone takes 4 steps from M's seed, each root 2 more
    ok &= check(res.steps.size >= 2 and int(res.steps[:2].min()) > 2,
                "the split roots' steps include the joint solve's",
                f'steps {res.steps.tolist()}')
    # the default: two roots of the model, never one level
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(P, 2, spin='singlet', tol_omega=1e-12)
    d = np.abs(np.sort(res.omega)[:2] - pair) if res.omega.size >= 2 else [np.inf]
    lv = res.level[:2].tolist()
    ok &= check(float(np.max(d)) < 1e-10 and len(lv) == 2 and lv[0] != lv[1]
                and not caught,
                'the default: both roots, levels of their own, no warning',
                f'max |dw| {np.max(d):.1e} Ha, level {lv}, {len(caught)} warning(s)')
    return ok


def check_duplicate_drop():
    """Of two copies of one root the converged one is kept: the earlier is marked
    only when the later alone converged."""
    from src.SingleReference.ADC.eeADC import ee_fold
    got = [ee_fold._duplicate_drop(0, 1, np.array(c))
           for c in ([True, True], [False, True], [True, False], [False, False])]
    return check(got == [1, 0, 1, 1],
                 'a duplicate marks the stalled copy, else the later one', str(got))


def check_duplicate_images(eps, B, no):
    """The duplicate check builds a doubles image only for a root within
    2 tol_omega of another: none for water's distinct roots."""
    from src.SingleReference.ADC.eeADC import ee_fold
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    calls = []
    orig = ee_fold._doubles_image

    def counted(*args):
        calls.append(1)
        return orig(*args)

    ee_fold._doubles_image = counted
    try:
        res = ee_fold.solve_folded(P, 3, spin='singlet')
    finally:
        ee_fold._doubles_image = orig
    return check(len(calls) == 0 and bool(res.converged.all()),
                 'distinct roots: the duplicate check builds no doubles image',
                 f'{len(calls)} image(s)')


def _sb_close(A, Bk, tol=1e-12):
    keys = set(A.keys()) | set(Bk.keys())
    return all(np.allclose(A.get(k) if A.get(k) is not None else 0.0,
                           Bk.get(k) if Bk.get(k) is not None else 0.0,
                           atol=tol, rtol=0) for k in keys)


def check_seeding_gap():
    """CH4 / 6-31G triplet: M's lowest state folds above the T2 triple, so the
    lowest root is seeded by the count check, not by M."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    ev = HARTREE_TO_EV
    mol = gto.M(atom=CH4, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, mol.nelectron // 2,
                                              level='adc2', pieces=True)
    e_full, _ = solve_ee_adc(mf, level='adc2', nroots=4, df=True, spin='triplet',
                             conv_tol=1e-10)
    e_full = np.asarray(e_full)
    for dense_limit, route in ((2000, 'dense'), (0, 'Davidson')):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ee_fold.solve_folded(P, 1, spin='triplet', dense_limit=dense_limit,
                                       tol_residual=1e-9)
        d = np.abs(res.omega - e_full[:res.omega.size]) * ev
        ok &= check(res.omega.size == 3 and float(d.max()) < 1e-5 and not caught,
                    f'{route}: nroots = 1 returns the T2 triple of the full solve',
                    f'nout {res.omega.size}, max |d| {d.max():.1e} eV, '
                    f'{len(caught)} warning(s)')
    # a doubles energy below the root: A_eff has a pole under it, so the count of
    # roots below the cut does not hold and is skipped with a warning
    m, C, D = np.array([0.5, 0.8, 1.1]), np.array([[0.05, 0.02, 0.0]]), [0.3]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), 1, spin='singlet')
    w, _ = _synthetic_exact(m, C, np.asarray(D))
    pole = any('not checked' in str(x.message) for x in caught)
    ok &= check(pole and abs(res.omega[0] - w[1]) < 1e-10,
                'a root above a doubles energy: the count is skipped with a warning',
                f'{len(caught)} warning(s), |dw| {abs(res.omega[0] - w[1]):.1e} Ha')
    return ok


def check_index_solve(eps, B, no):
    """The count where M misleads it: a state M's diagonal ranks last, folded to
    the bottom; a missing branch mixed 30 to 60 degrees with a found root at the
    cut (25 points); and one count solve per call on water when nothing is
    missing."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    m = 0.5 + 0.02 * np.arange(12)
    m[11] = 1.0
    C = np.zeros((1, 12))
    C[0, 11] = 0.7
    w, _ = _synthetic_exact(m, C, np.array([1.2]))
    for dense_limit, route in ((2000, 'dense'), (0, 'Davidson')):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ee_fold.solve_folded(_synthetic_pieces(m, C, [1.2]), 1,
                                       spin='singlet', dense_limit=dense_limit,
                                       tol_omega=1e-10)
        d = abs(res.omega[0] - w[0])
        ok &= check(d < 1e-9 and not caught,
                    f'{route}: a state last on M diagonal, folded lowest, is found',
                    f'|dw| {d:.1e} Ha, {len(caught)} warning(s)')
    worst, bad = 0.0, 0
    for t in (0.005, 0.01, 0.02, 0.04, 0.08):
        for delta in (-0.03, -0.015, 0.0, 0.015, 0.03):
            Mf = np.array([[0.50, 0.0, t], [0.0, 0.60, 0.0], [t, 0.0, 0.90]])
            Cq = np.array([[0.0, 0.0, np.sqrt((0.4 - delta) * 0.4)]])
            w, _ = _synthetic_exact(Mf, Cq, np.array([1.0]))
            for dense_limit in (2000, 0):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    res = ee_fold.solve_folded(_synthetic_pieces(Mf, Cq, [1.0]), 2,
                                               spin='singlet', tol_omega=1e-10,
                                               dense_limit=dense_limit)
                d = np.abs(res.omega[:2] - w[:2]).max() if res.omega.size >= 2 \
                    else np.inf
                worst, bad = max(worst, d), bad + int(d >= 1e-8 or bool(caught))
    ok &= check(bad == 0, 'mixed at the cut: 25 points x 2 branches, the two '
                'lowest roots exact', f'{bad} bad, max |dw| {worst:.1e} Ha')
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    calls, solve = [], ee_fold.solve_symmetric

    def counted(*a, **k):
        calls.append(k.get('label', ''))
        return solve(*a, **k)
    ee_fold.solve_symmetric = counted
    try:
        ee_fold.solve_folded(P, 3, spin='singlet', dense_limit=0)
    finally:
        ee_fold.solve_symmetric = solve
    n_count = sum(c.endswith(' count') for c in calls)
    ok &= check(n_count == 1, 'water, nothing missing: one count solve',
                f'{n_count} count solve(s)')
    return ok


def check_count_window():
    """The count's window against a near pair and against an unrequested partner:
    a pair of M split by 2e-9 Ha, two roots at tol_omega 1e-10, is the cut, with
    and without a state that folds below it: the two lowest come back exact and the
    partner, inside the count's 1e-8 Ha window, costs no warning; a distinct
    partner 1e-7 or 1e-6 Ha above the cut, not asked for, at default tolerances:
    the partner lies in the window, so the count's Davidson widens once past it."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    m, D = np.array([0.5, 0.5 + 2e-9, 0.9, 1.5]), np.array([1.6])
    for hidden in (True, False):
        C = np.array([[0.0, 0.0, 0.0, 1.149 if hidden else 0.0]])
        w, _ = _synthetic_exact(m, C, D)
        nw = 2
        for dense_limit, route in ((2000, 'dense'), (0, 'Davidson')):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), 2,
                                           spin='singlet', tol_omega=1e-10,
                                           dense_limit=dense_limit)
            d = np.abs(res.omega - w[:nw]).max() if res.omega.size == nw else np.inf
            what = 'the state folded below it is found' if hidden else 'none warns'
            ok &= check(d < 4e-10 and not caught,
                        f'{route}: a pair split by 2e-9 Ha at the cut, tol_omega '
                        f'1e-10: the two lowest exact, the partner in the count\'s '
                        f'window; {what}',
                        f'nout {res.omega.size}, max |dw| {d:.1e} Ha, '
                        f'{len(caught)} warning(s)')
    calls, solve = [], ee_fold.solve_symmetric

    def counted(*a, **k):
        calls.append(k.get('label', ''))
        return solve(*a, **k)
    ee_fold.solve_symmetric = counted
    try:
        for delta in (1e-7, 1e-6):
            m = np.array([0.5, 0.5 + delta, 0.7, 1.5])
            C = np.array([[0.0, 0.0, 0.0, 0.3]])
            for dense_limit, route in ((2000, 'dense'), (0, 'Davidson')):
                calls.clear()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), 1,
                                               spin='singlet', dense_limit=dense_limit)
                n_count = sum(c.endswith(' count') for c in calls)
                ok &= check(res.omega.size == 1 and abs(res.omega[0] - 0.5) < 1e-9
                            and not caught and n_count <= 2,
                            f'{route}: a partner {delta:.0e} Ha above the cut, not '
                            'asked for, costs no warning and one count (widened once)',
                            f'omega {res.omega}, {len(caught)} warning(s), '
                            f'{n_count} count solve(s)')
    finally:
        ee_fold.solve_symmetric = solve
    return ok


def check_hidden_branch():
    """The count's independent check, on the Davidson branch: a state last on M's
    diagonal folded lowest at n 800, where the roots found and M's diagonal seeds
    are exact eigenvectors of A_eff, so the count's own Davidson converges at once
    without it; the same with the folded state inside a coupled block of 50."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    n = 800
    m = 0.5 + 0.5 * np.arange(n) / n
    m[-1] = 3.0
    C = np.zeros((1, n))
    C[0, -1] = 2.75
    D = np.array([3.1])
    rng = np.random.default_rng(7)
    na, nb = n - 50, 50
    Mf = np.zeros((n, n))
    Mf[:na, :na] = np.diag(0.5 + 0.5 * np.arange(na) / na)
    Bm = rng.standard_normal((nb, nb)) * 2e-3
    Mf[na:, na:] = np.diag(1.5 + 0.5 * np.arange(nb) / nb) + 0.5 * (Bm + Bm.T)
    Cb = np.zeros((1, n))
    Cb[0, na:] = rng.standard_normal(nb)
    Cb *= 2.2 / np.linalg.norm(Cb)
    calls, solve = [], ee_fold.solve_symmetric

    def counted(*a, **k):
        calls.append(k.get('label', ''))
        return solve(*a, **k)
    ee_fold.solve_symmetric = counted
    try:
        for tag, mm, CC, DD in (('diagonal', m, C, D),
                                ('coupled block', Mf, Cb, [2.2])):
            w, _ = _synthetic_exact(mm, CC, np.asarray(DD))
            for nroots in (1, 3):
                calls.clear()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    res = ee_fold.solve_folded(_synthetic_pieces(mm, CC, DD), nroots,
                                               spin='singlet', dense_limit=0,
                                               tol_omega=1e-10)
                d = np.abs(res.omega[:nroots] - w[:nroots]).max() \
                    if res.omega.size >= nroots else np.inf
                n_check = sum(c.endswith(' check') for c in calls)
                ok &= check(d < 1e-8 and not caught and n_check <= 2,
                            f'Davidson, n {n}, {tag}: a state last on M folded '
                            f'lowest is found at nroots {nroots}, by one check that '
                            'finds it and one that confirms',
                            f'max |dw| {d:.1e} Ha, loop {res.loop}, '
                            f'{len(caught)} warning(s), {n_check} check solve(s)')
    finally:
        ee_fold.solve_symmetric = solve
    return ok


def check_index_bracket():
    """The index solve stops once its bracket is narrower than tol_omega, not at the
    step budget: a stub A_eff whose λ sits 1e-4 above every ω (the eigensolver's
    noise at a root pinned by a zero-width bracket) costs one step; a consistent
    λ(ω) = 5 - ω/2 with T1 = 2/3 still converges by Newton."""
    from src.SingleReference.ADC.eeADC import ee_fold

    def stub(slope, offset, dmv_value):
        def lowest_at(pieces, omega, spin, k, dense, tol_residual, label, cols,
                      tol=None):
            return (np.array([offset + slope * omega]), np.ones((1, 1)),
                    lambda v: dmv_value)
        return lowest_at

    orig = ee_fold._lowest_at
    out = {}
    try:
        for tag, f, om, lo, hi in (('pinned', stub(1.0, 1e-4, 0.0), 0.5, 0.5, 0.5),
                                   ('newton', stub(-0.5, 5.0, 0.5), 2.0, 2.0, 4.0)):
            ee_fold._lowest_at = f
            out[tag] = ee_fold._index_solve(None, None, 0, 1, om, lo, hi,
                                            np.ones((1, 1)), 0,
                                            1e-6, 1e-6, 42, 'test', 0)
    finally:
        ee_fold._lowest_at = orig
    k, conv = out['pinned'][3], out['pinned'][4]
    ok = check(k == 1 and not conv, 'index solve: a bracket narrower than tol_omega '
               'stops it at once, unconverged', f'{k} step(s)')
    lam, k, conv = out['newton'][0], out['newton'][3], out['newton'][4]
    ok &= check(conv and k <= 3 and abs(lam[0] - 10 / 3) < 1e-6,
                'index solve: a consistent branch converges by Newton',
                f'{k} steps, lambda {lam[0]:.8f}')
    return ok


def check_dense_default():
    """dense_limit=None takes pieces['dense_limit']: adc2 declares none and runs
    Davidson, gw declares 40 and builds A_eff densely on water/6-31G (n 40); an
    explicit dense_limit overrides."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_gw_pieces import build_pieces_gw
    _, eps, B, no = water_df('6-31g')
    P2 = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)[3]
    Pg = build_pieces_gw(eps, B, no, 'tda')
    calls = []
    orig = ee_fold.dense_effective

    def counted(*args):
        calls.append(1)
        return orig(*args)

    ee_fold.dense_effective = counted
    built = {}
    try:
        for tag, P, kw in (('adc2', P2, {}), ('gw', Pg, {}),
                           ('adc2 at 2000', P2, {'dense_limit': 2000})):
            calls.clear()
            ee_fold.solve_folded(P, 3, spin='singlet', **kw)
            built[tag] = len(calls)
    finally:
        ee_fold.dense_effective = orig
    ok = check('dense_limit' not in P2 and Pg.get('dense_limit') == 40,
               "pieces['dense_limit']: unset at adc2, 40 at gw",
               str(Pg.get('dense_limit')))
    ok &= check(built['adc2'] == 0 and built['gw'] > 0 and built['adc2 at 2000'] > 0,
                'dense_limit=None: adc2 runs Davidson, gw (n 40) builds A_eff, an '
                'explicit 2000 builds it at adc2', str(built))
    return ok


def check_warning_paths(eps, B, no):
    """The count's paths that warn or fall back: a surplus of unconverged roots, an
    index solve that lands on a found root, the round limit, the index solve's
    bisection, and a missing group re-solved per branch."""
    from src.SingleReference.ADC.eeADC import ee_fold

    def run(P, nroots, **kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ee_fold.solve_folded(P, nroots, spin='singlet', solver='newton',
                                       **kw)
        return res, [str(x.message) for x in caught]

    def pair(eps_m):
        # check 12's pair, the second state of M raised by eps_m
        D = np.array([1.0, 1.4])
        C = np.array([[0.3, 0.3, 0.0], [0.3, -0.54, 0.0]])
        s = np.sum(C ** 2 / (D[:, None] - 0.5), axis=0)
        m = np.array([0.5 + s[0], 0.5 + s[1] + eps_m, 1.5])
        return _synthetic_pieces(m, C, D), np.sort(_synthetic_exact(m, C, D)[0])

    P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)[3]
    res, msgs = run(P, 5, max_newton=2, max_fixed=0)
    ok = check(any('where A_eff has' in x for x in msgs) and not res.converged.any(),
               'two Newton steps: every root unconverged, and the count warns of the '
               'surplus', f'converged {res.converged.tolist()}')
    # split 5.9e-9 Ha, below the default tol_omega: one root at that resolution
    Pp, w = pair(1e-8)
    res, msgs = run(Pp, 2)
    ok &= check(any('landed on roots already found' in x
                    and 'missing from the result' in x for x in msgs)
                and res.omega.size == 1 and abs(res.omega[0] - w[0]) < 2e-6,
                'a pair closer than tol_omega: the branch solved by index lands on '
                'the found root, and the count says its root is missing',
                f'nout {res.omega.size}')
    Pp, w = pair(1.2e-6)
    keep = ee_fold._COUNT_ROUNDS
    ee_fold._COUNT_ROUNDS = 0
    try:
        res, msgs = run(Pp, 2, tol_omega=1e-12)
    finally:
        ee_fold._COUNT_ROUNDS = keep
    ok &= check(any('1 of the 2 lowest roots not found after 0 round' in x
                    for x in msgs) and res.omega.size == 1,
                'no count round left: the missing root is named, not solved',
                f'nout {res.omega.size}')

    def lowest_at(pieces, omega, spin, k, dense, tol_residual, label, cols,
                  tol=None):
        # λ(ω) = 5 - 3ω with T1 = 1: every Newton step leaves the bracket [1, 2]
        return np.array([5.0 - 3.0 * omega]), np.ones((1, 1)), lambda v: 0.0

    orig = ee_fold._lowest_at
    ee_fold._lowest_at = lowest_at
    try:
        lam, _, _, k, conv, _ = ee_fold._index_solve(None, None, 0, 1, 2.0, 1.0, 2.0,
                                                     np.ones((1, 1)), 0, 1e-6, 1e-10,
                                                     42, 'test', 0)
    finally:
        ee_fold._lowest_at = orig
    ok &= check(conv and k == 3 and abs(lam[0] - 1.25) < 1e-12,
                'index solve: a step that leaves the bracket bisects it',
                f'{k} steps, lambda {lam[0]:.12f}')
    # two states last on M, folded through different doubles to λ = 0.3 at
    # ω = 0.5: one missing group at the cut whose roots differ
    m = np.array([0.5, 0.6, 0.7, 1.5, 1.5])
    C = np.zeros((2, 5))
    C[0, 3], C[1, 4] = np.sqrt(1.2 * 1.1), np.sqrt(1.2 * 2.0)
    D = np.array([1.6, 2.5])
    w = np.sort(_synthetic_exact(m, C, D)[0])
    for dl in (2000, 0):
        res, msgs = run(_synthetic_pieces(m, C, D), 1, tol_omega=1e-10,
                        dense_limit=dl)
        d = abs(res.omega[0] - w[0])
        ok &= check(any('re-solved per branch' in x for x in msgs) and d < 1e-10
                    and res.loop == ['index'],
                    f'dense_limit {dl}: a missing group split at its roots is '
                    're-solved per branch, the lowest root exact', f'|dw| {d:.1e} Ha')
    return ok


def _fold_run(m, C, D, nroots, **kw):
    """(roots, max |w - exact| over the nroots lowest, warnings) of solve_folded on
    _synthetic_pieces(m, C, D), singlet."""
    from src.SingleReference.ADC.eeADC import ee_fold
    w = np.sort(_synthetic_exact(m, C, D)[0])[:nroots]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), nroots, spin='singlet',
                                   **kw)
    got = np.sort(res.omega)[:nroots]
    d = float(np.abs(got - w).max()) if got.size == nroots else np.inf
    return got, d, [str(x.message) for x in caught]


def check_window_branch():
    """A state M ranks third, folded to within delta of the found root 0.5 (inside
    that root's count window, the root not the cut): the branch index of the found
    root comes from its own vector, so the missed state is solved, not the found one
    again, with delta below or above 0.5, on both branches, without a warning."""
    ok = True
    for delta, tol_omega in ((1e-6, 1e-6), (-1e-6, 1e-6), (5e-9, 1e-10)):
        om = 0.5 - delta
        C = np.array([[0.0, 0.0, np.sqrt((0.9 - om) * (1.2 - om))]])
        for dl, solver in ((2000, 'davidson'), (0, 'davidson'), (2000, 'newton'),
                           (0, 'newton')):
            got, d, msgs = _fold_run(np.array([0.5, 0.6, 0.9]), C, np.array([1.2]), 2,
                                     tol_omega=tol_omega, dense_limit=dl,
                                     solver=solver)
            # the default tells two model roots apart down to tol_omega; Newton's
            # seeds within 1e-8 Ha are one level at its mean
            bound = 1e-8 if solver == 'newton' else min(1e-8, 4 * tol_omega)
            ok &= check(d < bound and not msgs,
                        f'delta {delta:+.0e}, tol_omega {tol_omega:.0e}, dense_limit '
                        f'{dl}, {solver}: both roots, no warning',
                        f'got {np.round(got, 11).tolist()}, max |dw| {d:.1e} Ha, '
                        f'{len(msgs)} warning(s)')
    return ok


def check_index_start():
    """The index solve keeps the cut's vectors in every Davidson start: a state last
    on M, folded to 0.55 between the found roots 0.5 and 0.6, is lost by a start of
    exact eigenvectors (n 4) or of one symmetry block (n 200, A_eff block diagonal
    in two irreps) unless it rides along; Davidson branch, no warning."""
    ok = True
    om = 0.55
    C = np.array([[0.0, 0.0, 0.0, np.sqrt((1.0 - om) * (0.8 - om))]])
    got, d, msgs = _fold_run(np.array([0.5, 0.6, 0.9, 1.0]), C, np.array([0.8]), 2,
                             dense_limit=0)
    ok &= check(d < 1e-8 and not msgs, 'n 4, exact eigenvector starts: 0.5 and 0.55',
                f'got {np.round(got, 9).tolist()}, {len(msgs)} warning(s)')
    n, rng = 200, np.random.default_rng(3)
    irrep = np.r_[0, 0, np.arange(n - 3) % 2, 1]
    R = rng.standard_normal((n, n)) * 1e-2
    R = 0.5 * (R + R.T) * (irrep[:, None] == irrep[None, :])
    m = np.diag(np.r_[0.5, 0.6, np.linspace(0.62, 1.5, n - 3), 1.6]) + R
    C = np.zeros((1, n))
    C[0, -1] = np.sqrt((1.6 - om) * (0.8 - om))
    got, d, msgs = _fold_run(m, C, np.array([0.8]), 2, dense_limit=0)
    ok &= check(d < 1e-7 and not msgs,
                'n 200, two irreps: the steep state of the other irrep is found',
                f'|dw| {d:.1e} Ha, {len(msgs)} warning(s)')
    return ok


def check_cut_above_doubles():
    """M = diag(0.5, 0.6, 0.9, 1.0), the last state folded through a double at 0.8
    to 0.55: at nroots 3 the third root found (0.9) lies past min D, where the
    count does not hold; it runs at 0.6 instead and finds 0.55."""
    ok = True
    om = 0.55
    C = np.array([[0.0, 0.0, 0.0, np.sqrt((1.0 - om) * (0.8 - om))]])
    for dl in (2000, 0):
        got, d, msgs = _fold_run(np.array([0.5, 0.6, 0.9, 1.0]), C, np.array([0.8]),
                                 3, dense_limit=dl, solver='newton')
        ok &= check(d < 1e-8 and not msgs,
                    f'dense_limit {dl}: nroots 3 past min D returns 0.5, 0.55, 0.6',
                    f'got {np.round(got, 9).tolist()}, {len(msgs)} warning(s)')
        # the default never finds 0.9: its polish stops at |lambda - w| < 1e-6
        got, d, msgs = _fold_run(np.array([0.5, 0.6, 0.9, 1.0]), C, np.array([0.8]),
                                 3, dense_limit=dl)
        ok &= check(d < 1e-7 and not msgs,
                    f'dense_limit {dl}: the default returns 0.5, 0.55, 0.6',
                    f'max |dw| {d:.1e} Ha, {len(msgs)} warning(s)')
        # nroots 4 reaches past min D, where the model holds three roots: the
        # default returns the seeds' 0.9 and its one warning, never a lambda at a
        # closed index bracket
        got, d, msgs = _fold_run(np.array([0.5, 0.6, 0.9, 1.0]), C, np.array([0.8]),
                                 4, dense_limit=dl)
        ok &= check(d < 1e-7 and len(msgs) == 1 and 'not checked' in msgs[0],
                    f'dense_limit {dl}: nroots 4, the default returns 0.5, 0.55, '
                    '0.6, 0.9 and warns once', f'got {np.round(got, 9).tolist()}, '
                    f'{len(msgs)} warning(s)')
    return ok


def check_guess_start():
    """_guess(cols, diag, width) keeps every independent column it is offered:
    n 4, cols e0 and e1, whose diagonal seeds repeat them."""
    from src.SingleReference.ADC.eeADC import ee_fold
    cols = np.eye(4)[:, :2]
    Q = ee_fold._guess(cols, np.array([0.5, 0.6, 0.9, 1.0]), 8)
    orth = Q.shape[1] == 4 and np.allclose(Q.T @ Q, np.eye(4), atol=1e-12)
    first = np.allclose(np.abs(Q[:, :2].T @ cols), np.eye(2), atol=1e-12)
    return check(orth and first, 'the start spans the space, the found vectors first',
                 f'{Q.shape[1]} of 4 columns')


def check_refusals():
    """A level outside FOLD_LEVELS and a spin outside singlet/triplet/None raise
    ValueError; a fixed-point loop (t_min above every T1) out of steps warns, and the
    count, which finds that root's branch uncovered, solves it again by index."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    m, C, D = np.array([0.5, 0.6]), np.array([[0.4, 0.0]]), np.array([0.9])
    for tag, P, spin in (('level adc3', dict(_synthetic_pieces(m, C, D),
                                             level='adc3'), 'singlet'),
                         ('spin quintet', _synthetic_pieces(m, C, D), 'quintet')):
        try:
            ee_fold.solve_folded(P, 1, spin=spin)
            raised = False
        except ValueError:
            raised = True
        ok &= check(raised, f'{tag}: ValueError')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), 1, spin='singlet',
                                   t_min=0.999, max_newton=1, max_fixed=1,
                                   solver='newton')
    msgs = [str(x.message) for x in caught]
    d = abs(res.omega[0] - np.sort(_synthetic_exact(m, C, D)[0])[0])
    ok &= check(any('(fixed)' in x for x in msgs) and res.loop == ['index']
                and bool(res.converged[0]) and d < 2e-6,
                'the fixed-point loop out of steps warns; the count solves the root '
                'by index', f'loop {res.loop}, |dw| {d:.1e} Ha')
    return ok


def check_degenerate_hidden():
    """spin=None doubles every state of _synthetic_pieces (no cross-spin block), so a
    state folded below the seeds is an exact pair; the seeds are exact eigenvectors
    (M and C diagonal in one basis). The count's check, one random vector per span,
    finds the second partner after the first."""
    ok = True
    for name, om, D0, n in (('below 0.5', 0.5 - 1e-6, 1.2, 20),
                            ('between 0.5 and 0.6', 0.55, 0.8, 20),
                            ('below 0.5', 0.5 - 1e-6, 1.2, 200)):
        m = np.r_[0.5, 0.6, np.linspace(0.62, 1.5, n - 3), 1.6]
        C = np.zeros((1, n))
        C[0, -1] = np.sqrt((1.6 - om) * (D0 - om))
        w = np.repeat(np.sort(_synthetic_exact(m, C, np.array([D0]))[0])[:2], 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = _spin_none_solve(m, C, D0)
        got = np.sort(res.omega)[:4]
        d = float(np.abs(got - w).max()) if got.size == 4 else np.inf
        ok &= check(d < 1e-8 and not caught,
                    f'n {n}, a pair {name}: both partners, no warning',
                    f'|dw| {d:.1e} Ha, {len(caught)} warning(s)')
    return ok


def check_trace(eps, B, no):
    """verbose 1 against verbose 0 on water singlets, nroots 3: the same roots bit
    for bit, the phase lines with their stamps, the closing matvec count against
    every application of the operator (the Davidson's images included) and, with
    solver='newton', against the matvecs solve_symmetric received, heartbeats at a
    zero interval, and the trace reset after every call, one that raises included."""
    import contextlib
    import io
    import re
    from src.SingleReference.ADC.eeADC import ee_fold
    P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)[3]
    out0 = io.StringIO()
    with contextlib.redirect_stdout(out0):
        res0 = ee_fold.solve_folded(P, 3, spin='singlet')
    calls, solve, beat = [0], ee_fold.solve_symmetric, ee_fold._HEARTBEAT
    apps, fop = [0], ee_fold.folded_operator

    def counted(matvec, diag, **k):
        def mv(u):
            calls[0] += 1
            return matvec(u)
        return solve(mv, diag, **k)

    def applied(pieces, omega, spin=None):
        # every matvec and every vector dmatvec, inside solve_symmetric or not
        matvec, dmatvec, diag_s, embed, restrict = fop(pieces, omega, spin)

        def mv(u):
            apps[0] += 1
            return matvec(u)

        def dmv(u, vector=False):
            apps[0] += bool(vector)
            return dmatvec(u, vector=vector)
        return mv, dmv, diag_s, embed, restrict
    ee_fold.solve_symmetric, ee_fold._HEARTBEAT = counted, 0.0
    ee_fold.folded_operator = applied
    out1, outn = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out1):
            res1 = ee_fold.solve_folded(P, 3, spin='singlet', verbose=1)
        n_calls, n_apps, calls[0] = calls[0], apps[0], 0
        with contextlib.redirect_stdout(outn):
            ee_fold.solve_folded(P, 3, spin='singlet', verbose=1, solver='newton')
    finally:
        ee_fold.solve_symmetric, ee_fold._HEARTBEAT = solve, beat
        ee_fold.folded_operator = fop
    lines = out1.getvalue().splitlines()
    stamp = re.compile(r'\[t = \d+\.\d s, mv = (\d+)\]$')
    d = float(np.abs(res0.omega - res1.omega).max()) \
        if res0.omega.shape == res1.omega.shape else np.inf
    ok = check(d < 1e-10 and not out0.getvalue(),
               'verbose 1 returns the roots of verbose 0; verbose 0 prints nothing',
               f'max |dw| {d:.1e} Ha, {len(out0.getvalue())} characters at verbose 0')
    tags = ('start:', 'seeds:', 'count round 0', 'check:', 'step 1', 'running',
            'returns')
    seen = {t: any(t in s for s in lines) for t in tags}
    unstamped = [s for s in lines if not stamp.search(s)]
    last = stamp.search(lines[-1]) if lines else None
    ok &= check(all(seen.values()) and lines and not unstamped,
                'verbose 1: start, seeds, count round, check, step, heartbeat and '
                'closing lines, each ending on its elapsed time and matvec count',
                f'{len(lines)} lines, missing {[t for t in tags if not seen[t]]}, '
                f'{len(unstamped)} unstamped')
    ok &= check(last is not None and 'returns' in lines[-1]
                and int(last.group(1)) == n_apps and n_calls < n_apps,
                'the closing line counts every application, the Davidson images '
                'beside the matvecs solve_symmetric received',
                f'closing {last.group(1) if last else None}, applications {n_apps}, '
                f'received {n_calls}')
    last = stamp.search(outn.getvalue().splitlines()[-1])
    ok &= check(last is not None and int(last.group(1)) == calls[0],
                "solver='newton': the closing line counts the matvecs "
                'solve_symmetric received',
                f'closing {last.group(1) if last else None}, received {calls[0]}')
    try:
        ee_fold.solve_folded(P, 3, spin='quintet', verbose=1)
        raised = False
    except ValueError:
        raised = True
    ok &= check(raised and ee_fold._TRACE.get() is None,
                'no trace outlives a call, one that raises included')
    return ok


def check_threaded_elementwise():
    """_ew against the plain ufunc, bit for bit, with the pin at 4 threads and an
    array above the size where the pass is split."""
    from src.SingleReference.ADC.eeADC import ee_spin_blocks as sb
    rng = np.random.default_rng(3)
    a = rng.standard_normal((64, 32, 32, 32))
    b = rng.standard_normal(a.shape)
    d = 1.0 + rng.random(a.shape)
    keep = os.environ.get('OMP_NUM_THREADS')
    os.environ['OMP_NUM_THREADS'] = '4'
    try:
        split = a.size >= sb._EW_MIN and sb._threads() == 4
        z = sb._ew(np.subtract, a, b.transpose(0, 1, 3, 2))
        s = a.copy()
        sb._ew(np.add, s, b, out=s)
        cases = {'difference, transposed operand': (z, a - b.transpose(0, 1, 3, 2)),
                 'in-place sum': (s, a + b),
                 'scalar product': (sb._ew(np.multiply, a, -1.0), -a),
                 'quotient': (sb._ew(np.true_divide, a, d), a / d),
                 'broadcast, left to NumPy': (sb._ew(np.multiply, a, d[0]), a * d[0])}
    finally:
        if keep is None:
            del os.environ['OMP_NUM_THREADS']
        else:
            os.environ['OMP_NUM_THREADS'] = keep
    bad = [k for k, (x, y) in cases.items() if not np.array_equal(x, y)]
    return check(split and not bad,
                 'the threaded elementwise pass equals the plain ufunc bit for bit',
                 f'split {split}, differing {bad}')


def check_split_contractions():
    """ooov_ia and ovvv_ia, summed over j-slabs by _ksum with the pin at 4 threads,
    against the block-wise einsum on anti4's blocks, for an X without a flip
    parity and with each one, to 1e-12 relative."""
    from src.SingleReference.ADC.eeADC import ee_equations as eq
    from src.SingleReference.ADC.eeADC import ee_spin_blocks as sb
    rng = np.random.default_rng(5)
    no, nv = 6, 9
    G = sb.anti4(rng.standard_normal((no, no, no, nv)),
                 rng.standard_normal((no, no, no, nv)))
    S = rng.standard_normal((nv, no, nv, nv))
    ovvv = sb.anti4(S.transpose(1, 0, 2, 3))
    keys = ('aaaa', 'bbbb', 'abab', 'baba', 'abba', 'baab')
    X0 = sb.SB({k: rng.standard_normal((no, no, nv, nv)) for k in keys})
    half = {k: v for k, v in X0.items() if k[0] == 'a'}
    Xs = {'none': X0} | {f'parity {p:+d}': sb.SB(sb._with_flip(half, p), parity=p)
                         for p in (1, -1)}
    keep, floor = os.environ.get('OMP_NUM_THREADS'), sb._KSUM_MIN
    os.environ['OMP_NUM_THREADS'] = '4'
    sb._KSUM_MIN = 0                        # these blocks sit below the split's floor
    ein = eq.SPIN_BLOCKED.ein
    try:
        worst = 0.0
        for X in Xs.values():
            for new, ref in ((sb.ooov_ia(G, X), ein('jkib,jkab->ia', G, X)),
                             (sb.ovvv_ia(S, X), ein('jabc,ijbc->ia', ovvv, X))):
                for s in ('aa', 'bb'):
                    r = ref.get(s)
                    worst = max(worst, float(np.max(np.abs(new.get(s) - r))
                                             / np.max(np.abs(r))))
    finally:
        sb._KSUM_MIN = floor
        if keep is None:
            del os.environ['OMP_NUM_THREADS']
        else:
            os.environ['OMP_NUM_THREADS'] = keep
    return check(worst < 1e-12,
                 'ooov_ia and ovvv_ia over j-slabs equal the block-wise einsum',
                 f'{worst:.1e} relative, X {list(Xs)}')


def check_kernel_hook(eps, B, no):
    """dmatvec(u, vector=True) is dA_eff/domega applied to u: its quadratic form is
    -dmatvec(u) to 1e-12 relative, the central difference of matvec at h = 1e-4 Ha
    to 1e-6 relative, zeros at omega None; an optional pieces['K'] replaces the V,
    divide, Vt path of matvec and of both dmatvec forms, bitwise, called once per
    application with the powers (1,), () and (2,)."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    nv = B.shape[1] - no
    P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)[3]
    P = dict(P, K=None)                     # the generic path, not the streamed kernel
    om, h = 0.3, 1e-4
    mv, dmv, diag_s, _, _ = ee_fold.folded_operator(P, om, spin='singlet')
    mvp = ee_fold.folded_operator(P, om + h, spin='singlet')[0]
    mvm = ee_fold.folded_operator(P, om - h, spin='singlet')[0]
    u = np.random.default_rng(2).standard_normal(diag_s.size)
    u /= np.linalg.norm(u)
    g = dmv(u, vector=True)
    fd = (mvp(u) - mvm(u)) / (2 * h)
    d1 = abs(float(u @ g) + dmv(u)) / abs(dmv(u))
    d2 = float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))
    ok &= check(d1 < 1e-12 and d2 < 1e-6,
                'dmatvec(u, vector=True) is dA_eff/domega u', f'{d1:.1e}, {d2:.1e}')
    dmv0 = ee_fold.folded_operator(P, None, spin='singlet')[1]
    ok &= check(dmv0(u) == 0.0
                and np.array_equal(dmv0(u, vector=True), np.zeros(u.size)),
                'omega None: dmatvec is 0.0 and its vector form zeros')
    calls = []

    def K(y1, omega, powers):
        calls.append(tuple(powers))
        denom = P['D'] - omega
        Y = P['be'].divide(P['V'](y1), denom)
        ws = [P['Vt'](Y if p == 1 else P['be'].divide(Y, denom)) for p in powers]
        Yf = ee_fold.doubles_sb_to_flat(Y, no, nv)
        return ws, float(Yf @ Yf)

    mvk, dmvk, _, _, _ = ee_fold.folded_operator(dict(P, K=K), om, spin='singlet')
    same = (np.array_equal(mvk(u), mv(u)) and dmvk(u) == dmv(u)
            and np.array_equal(dmvk(u, vector=True), g))
    ok &= check(same and calls == [(1,), (), (2,)],
                "pieces['K'] is the kernel matvec and both dmatvec forms apply",
                str(calls))
    return ok


def check_model_roots():
    """_model_roots on closed forms: H1 = 0 gives the eigenvalues of H0 below d_min;
    H1 = -I gives lambda_j(omega) = w_j - (omega - om0), root (w_j + om0) / 2."""
    from src.SingleReference.ADC.eeADC import ee_fold
    rng = np.random.default_rng(11)
    A = rng.standard_normal((6, 6))
    H0 = 0.5 * (A + A.T)
    w = np.linalg.eigvalsh(H0)
    om0 = float(w[0]) - 0.3
    oms, C = ee_fold._model_roots(H0, np.zeros((6, 6)), om0, 3, float(w[-1]), 1e-12)
    ok = check(np.allclose(oms, w[:3], atol=1e-10) and C.shape == (6, 3),
               'H1 = 0: the roots are the eigenvalues of H0', f'{oms - w[:3]}')
    oms, C = ee_fold._model_roots(H0, -np.eye(6), om0, 6, 1e9, 1e-12)
    ok &= check(np.allclose(oms, 0.5 * (w + om0), atol=1e-10),
                'H1 = -I: roots (w_j + om0) / 2', f'{oms - 0.5 * (w + om0)}')
    oms, _ = ee_fold._model_roots(H0, np.zeros((6, 6)), om0, 6, float(w[2]), 1e-12)
    ok &= check(oms.size == 2, 'roots at or above d_min are dropped', f'{oms.size}')
    # check 20's pieces imaged at w0 = 0.5: the last diagonal 0.625 - 1.25 (w - 0.5)
    # crosses 0.6 and 0.5 exactly, a kink in branch 1, whose root is 5 / 9; a
    # Newton step that jumps the kink cycles between 0.5 and 0.6
    c2 = (1.0 - 0.55) * (0.8 - 0.55)
    P = _synthetic_pieces(np.array([0.5, 0.6, 0.9, 1.0]), np.array([[0, 0, 0, c2]])
                          ** 0.5, np.array([0.8]))
    mv, dmv, diag_s, _, _ = ee_fold.folded_operator(P, 0.5, 'singlet')
    Q = np.eye(diag_s.size)
    H0 = Q.T @ np.column_stack([mv(q) for q in Q.T])
    H1 = Q.T @ np.column_stack([dmv(q, vector=True) for q in Q.T])
    oms, _ = ee_fold._model_roots(H0, H1, 0.5, 4, 0.8, 1e-7)
    ref = np.array([0.5, 5.0 / 9.0, 0.6])
    ok &= check(oms.size == 3 and np.allclose(oms, ref, atol=1e-6),
                'an exact branch crossing: one root per branch, none false',
                f'{oms}')
    return ok


def check_davidson_omega(eps, B, no):
    """_davidson_omega on water adc2 singlets and triplets, nroots 3, from the seeds
    of M, at max_subspace 40 and 4 (a collapse every cycle): the model converges,
    its roots lie within 1e-3 Ha of the Newton roots (its first-order error away
    from the centre), and the per-root Newton polish from them, at most three
    steps a root, returns the Newton roots to 1e-7 Ha; a collapse at one centre
    images only the new vectors; two cycles leave the model unconverged, with a
    warning."""
    import contextlib
    import io
    import re
    from src.SingleReference.ADC.eeADC import ee_fold
    P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)[3]
    d_min = float(np.min(P['D']))
    ok = True
    for spin in ('singlet', 'triplet'):
        ref = ee_fold.solve_folded(P, 3, spin=spin, tol_omega=1e-9, tol_residual=1e-8)
        m0, _, diag_s, _, _ = ee_fold.folded_operator(P, None, spin)
        _, X0, _ = ee_fold._seed_levels(m0, diag_s, diag_s.size, 3, False, 1e-8, 't')
        for cap in (40, 4):
            oms, X, conv, cyc = ee_fold._davidson_omega(
                P, spin, X0, 3, d_min, 1e-8, 1e-9, cap, 200, 'test')
            dm = float(np.max(np.abs(oms - ref.omega))) if oms.size == 3 else np.inf
            lams, steps = [], []
            for r in range(oms.size):
                # the solve's own per-root polish: Newton on λ(ω) = ω from the root
                step = ee_fold._root_step(P, spin, X[:, r], False, 1e-8, 'test')
                lam, _, _, k, _, c, _ = ee_fold._iterate(
                    step, float(oms[r]), 1e-9, 0.3, 12, 30, 'test', 0)
                lams.append(lam if c else np.inf)
                steps.append(k)
            d = float(np.max(np.abs(np.array(lams) - ref.omega))) \
                if oms.size == 3 else np.inf
            ok &= check(conv.all() and dm < 1e-3 and d < 1e-7 and max(steps) <= 3,
                        f'{spin}, max_subspace {cap}: the Davidson roots, polished, '
                        'equal the Newton roots',
                        f'model |d| {dm:.1e} Ha, polished |d| {d:.1e} Ha, {cyc} '
                        f'cycles, polish steps {steps}')
        # max_subspace 4, a collapse every cycle: at an unchanged centre a cycle
        # images only its new vectors, the kept ones' images rotated
        out = io.StringIO()
        token = ee_fold._TRACE.set(ee_fold._Trace('t'))
        try:
            with contextlib.redirect_stdout(out):
                ee_fold._davidson_omega(P, spin, X0, 3, d_min, 1e-8, 1e-9, 4, 200, 't')
        finally:
            ee_fold._TRACE.reset(token)
        pat = r'cycle \d+: (\d+) vectors, centre ([0-9.]+),.*mv = (\d+)\]'
        rows = [(int(a), b, int(c)) for a, b, c in re.findall(pat, out.getvalue())]
        same = [(nv, mv - mv0) for (_, c0, mv0), (nv, c, mv) in zip(rows, rows[1:])
                if c == c0]
        ok &= check(same and all(d < 2 * nv for nv, d in same),
                    f'{spin}, max_subspace 4: a collapse at one centre images only the '
                    'new vectors', f'(vectors, mv) per cycle {same[:6]}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, _, conv, cyc = ee_fold._davidson_omega(
                P, spin, X0, 3, d_min, 1e-8, 1e-9, 40, 2, 'test')
        msgs = [str(x.message) for x in caught]
        ok &= check(any('not converged' in x for x in msgs) and not conv.all()
                    and cyc == 2,
                    f'{spin}: a model out of cycles warns, its roots unconverged',
                    f'{len(msgs)} warning(s), conv {conv.tolist()}, {cyc} cycles')
    # the centre follows the nroots lowest roots, never the roots followed past
    # them: a new centre is the median of the previous cycle's nroots lowest
    pat = r'davidson cycle \d+: \d+ vectors, centre ([0-9.]+), roots ([0-9. ]+), \|r'
    for nr in (1, 2):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ee_fold.solve_folded(P, nr, spin='singlet', verbose=1)
        rows = [(float(c), [float(x) for x in r.split()])
                for c, r in re.findall(pat, out.getvalue())]
        moves = [(c, float(np.median(r0[:nr])))
                 for (c0, r0), (c, _) in zip(rows, rows[1:]) if c != c0]
        dev = max((abs(c - m) for c, m in moves), default=np.inf)
        ok &= check(dev < 2e-6, f'singlets, nroots {nr}: every new centre is the '
                    'median of the nroots lowest roots',
                    f'{len(moves)} recentre(s), max |centre - median| {dev:.1e} Ha')
    return ok


def _closing_mv(P, nroots, spin, **kw):
    """(FoldResult, closing mv) of solve_folded at verbose 1, its stdout captured."""
    import contextlib
    import io
    import re
    from src.SingleReference.ADC.eeADC import ee_fold
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = ee_fold.solve_folded(P, nroots, spin=spin, verbose=1, **kw)
    last = re.search(r'mv = (\d+)\]$', out.getvalue().splitlines()[-1])
    return res, int(last.group(1)) if last else -1


def check_solver_default(eps, B, no):
    """solver='davidson' is the default: on water adc2 singlets, nroots 3, its roots
    equal solver='newton''s to 1e-8 Ha and its T1 to 1e-6, loop reads 'davidson'
    for every root, steps counts the polish alone, solver='bogus' raises
    ValueError; on CH4 / 6-31G triplets,
    nroots 1, where the fold puts M's lowest state above the T2 triple (check 13),
    the default returns Newton's triple without the count's index solve and closes
    on fewer matvecs."""
    from src.SingleReference.ADC.eeADC import ee_fold
    P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)[3]
    dav = ee_fold.solve_folded(P, 3, spin='singlet')
    new = ee_fold.solve_folded(P, 3, spin='singlet', solver='newton')
    same = dav.omega.shape == new.omega.shape
    d = float(np.max(np.abs(dav.omega - new.omega))) if same else np.inf
    dt = float(np.max(np.abs(dav.t1 - new.t1))) if same else np.inf
    ok = check(d < 1e-8 and dt < 1e-6 and dav.converged.all(),
               "solver 'davidson' and 'newton' give the same roots and T1",
               f'|dw| {d:.1e} Ha, |dT1| {dt:.1e}')
    ok &= check(dav.loop == ['davidson'] * dav.omega.size
                and 'davidson' not in new.loop,
                "loop reads 'davidson' for every root of the default",
                f'{dav.loop}, {new.loop}')
    # the polish from the model's roots takes at most three steps (check 30)
    ok &= check(int(dav.steps.max()) <= 3,
                "the default's steps count its polish, not the Davidson's cycles",
                f'steps {dav.steps.tolist()}')
    try:
        ee_fold.solve_folded(P, 3, spin='singlet', solver='bogus')
        raised = False
    except ValueError:
        raised = True
    ok &= check(raised, "solver='bogus' raises ValueError")
    mol = gto.M(atom=CH4, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    e4 = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B4 = DFIntegrals.from_scf(mol, mf).B_aa
    P4 = ee_r_sigma_df.build_operator(e4, B4, mol.nelectron // 2, level='adc2',
                                      pieces=True)[3]
    r_dav, mv_dav = _closing_mv(P4, 1, 'triplet')
    r_new, mv_new = _closing_mv(P4, 1, 'triplet', solver='newton')
    same = r_dav.omega.shape == r_new.omega.shape == (3,)
    d = float(np.max(np.abs(r_dav.omega - r_new.omega))) if same else np.inf
    ok &= check(same and d < 1e-8 and 'index' in r_new.loop
                and r_dav.loop == ['davidson'] * 3,
                'CH4 / 6-31G triplets, nroots 1: the default finds the T2 triple '
                'that Newton reaches by index', f'|dw| {d:.1e} Ha, {r_new.loop}')
    ok &= check(0 < mv_dav < mv_new,
                'there the default closes on fewer matvecs than Newton',
                f'{mv_dav} against {mv_new}')
    return ok


def check_default_polish():
    """The default's exact polish and its duplicate check: a model root far from
    its exact root (strong coupling, the centre never moved: _RECENTRE 10 Ha) with
    one polish step comes back converged False with the solve's warning, never at
    the model's value as converged, and a 4-fold level so held comes back whole,
    every partner unconverged; two roots the Davidson hands over as copies of one
    are reported, the copy dropped, and the root it displaced solved by index;
    max_newton 0 leaves the Davidson a budget and returns the root."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    m, C, D = np.array([0.5, 0.9]), np.array([[0.25, 0.0]]), np.array([0.8])
    w = np.sort(_synthetic_exact(m, C, D)[0])
    keep = ee_fold._RECENTRE
    ee_fold._RECENTRE = 10.0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), 1, spin='singlet',
                                       dense_limit=0, max_newton=1, t_min=0.0)
    finally:
        ee_fold._RECENTRE = keep
    warned = any('not converged' in str(x.message) for x in caught)
    off = abs(float(res.omega[0]) - w[0])
    ok &= check(warned and not res.converged[0] and off > 1e-6,
                'a model root one polish step cannot fix: converged False, warned',
                f'|w - exact| {off:.1e} Ha, converged {res.converged.tolist()}')
    m, C, D = np.array([0.5, 0.6, 0.9]), np.array([[0.1, 0.1, 0.0]]), np.array([1.2])
    w = np.sort(_synthetic_exact(m, C, D)[0])[:2]
    orig = ee_fold._davidson_omega

    def twice(*a):
        # the lowest root handed over twice, 1e-6 Ha apart, one vector
        oms, X, conv, cycles = orig(*a)
        return (np.array([oms[0], oms[0] + 1e-6]), X[:, [0, 0]], conv[[0, 0]],
                cycles)
    ee_fold._davidson_omega = twice
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), 2, spin='singlet',
                                       dense_limit=0)
    finally:
        ee_fold._davidson_omega = orig
    dup = any('landed on one root' in str(x.message) for x in caught)
    d = float(np.max(np.abs(res.omega - w))) if res.omega.size == 2 else np.inf
    ok &= check(dup and d < 1e-8 and res.loop == ['davidson', 'index'],
                'a copy from the Davidson is dropped and the displaced root solved '
                'by index', f'max |dw| {d:.1e} Ha, loop {res.loop}')
    # k copies of a coupled block, each with its own double: every level k-fold,
    # M's seeds no eigenvectors of A_eff; spin=None doubles it again
    rng = np.random.default_rng(17)
    A = rng.standard_normal((3, 3)) * 0.03
    Bk = np.diag([0.50, 0.58, 0.70]) + 0.5 * (A + A.T)

    def kfold(k):
        """(m, C) of k copies of the block Bk, each coupled to its own double."""
        m = np.zeros((3 * k + 3, 3 * k + 3))
        C = np.zeros((k, 3 * k + 3))
        for i in range(k):
            m[3 * i:3 * i + 3, 3 * i:3 * i + 3] = Bk
            C[i, 3 * i:3 * i + 3] = [0.15, -0.10, 0.08]
        m[3 * k:, 3 * k:] = np.diag([0.9, 1.1, 1.3])
        return m, C
    for k, spin in ((4, 'singlet'), (5, 'singlet'), (2, None)):
        m, C = kfold(k)
        w0 = np.sort(_synthetic_exact(m, C, np.ones(k))[0])[0]
        g = k * (2 if spin is None else 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ee_fold.solve_folded(_synthetic_pieces(m, C, np.ones(k)), 1,
                                       spin=spin, dense_limit=0)
        got = int(np.count_nonzero(np.abs(res.omega - w0) < 1e-7))
        ok &= check(got == g and not caught,
                    f'a {g}-fold level cut by nroots 1 comes back whole, spin {spin}',
                    f'{got} of {g} partners, {len(caught)} warning(s)')
    # the 4-fold level, its model root far from the exact one (the centre held at
    # M's root), and one polish step: the level comes back whole and unconverged
    m, C = kfold(4)
    keep = ee_fold._RECENTRE
    ee_fold._RECENTRE = 10.0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ee_fold.solve_folded(_synthetic_pieces(m, C, np.ones(4)), 1,
                                       spin='singlet', dense_limit=0, max_newton=1,
                                       t_min=0.0)
    finally:
        ee_fold._RECENTRE = keep
    lv = any('level' in str(x.message) and 'not converged' in str(x.message)
             for x in caught)
    ok &= check(lv and res.omega.size == 4 and not res.converged.any(),
                'a level one polish step cannot fix: every partner converged False, '
                'the level warned', f'nout {res.omega.size}, converged '
                f'{res.converged.tolist()}, loop {res.loop}')
    # max_newton 0: the Newton budget spent at once, the Davidson still runs
    m, C, D = np.array([0.5, 0.9]), np.array([[0.1, 0.0]]), np.array([1.2])
    w0 = np.sort(_synthetic_exact(m, C, D)[0])[0]
    try:
        res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), 1, spin='singlet',
                                   dense_limit=0, max_newton=0)
        d, why = abs(float(res.omega[0]) - w0), f'loop {res.loop}'
    except Exception as err:
        d, why = np.inf, f'{type(err).__name__}: {err}'
    ok &= check(d < 1e-6, 'max_newton 0 on the default returns the root',
                f'|dw| {d:.1e} Ha, {why}')
    # the seeds' solve left unconverged: a start only on the default, so one note
    # line and no warning; Newton keeps solve_symmetric's warning
    import contextlib
    import io
    orig_solve, seen = ee_fold._solve, {}

    def unconverged(matvec, diag, **k):
        e, X, conv = orig_solve(matvec, diag, **k)
        if k.get('label', '').endswith(' seeds'):
            seen[k.get('warn_unconverged', True)] = True
            conv = np.zeros_like(conv)
        return e, X, conv
    ee_fold._solve = unconverged
    try:
        out = io.StringIO()
        with warnings.catch_warnings(record=True) as caught, \
                contextlib.redirect_stdout(out):
            warnings.simplefilter('always')
            res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), 1, spin='singlet',
                                       dense_limit=0, verbose=1)
        warn_dav = sorted(seen)
        seen.clear()
        with warnings.catch_warnings(record=True), \
                contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('always')
            ee_fold.solve_folded(_synthetic_pieces(m, C, D), 1, spin='singlet',
                                 dense_limit=0, solver='newton')
        warn_new = sorted(seen)
    finally:
        ee_fold._solve = orig_solve
    noted = [ln for ln in out.getvalue().splitlines()
             if 'seeds:' in ln and 'unconverged' in ln]
    ok &= check(len(noted) == 1 and not caught and warn_dav == [False]
                and warn_new == [True] and abs(float(res.omega[0]) - w0) < 1e-6,
                'seeds left unconverged: one note line and no warning on the default, '
                "solve_symmetric's warning kept with solver='newton'",
                f'{len(noted)} note line(s), {len(caught)} warning(s), '
                f'warn_unconverged {warn_dav} / {warn_new}')
    # the dense branch seeds M in full; the Davidson takes the levels it follows
    n = 200
    C = np.zeros((2, n))
    C[0, :3], C[1, 3:6] = 0.1, 0.1
    P = _synthetic_pieces(0.5 + 0.5 * np.arange(n) / n, C, np.array([1.2, 1.4]))
    sizes, orig = [], ee_fold._model_roots

    def sized(H0, *a):
        sizes.append(H0.shape[0])
        return orig(H0, *a)
    ee_fold._model_roots = sized
    try:
        res = ee_fold.solve_folded(P, 3, spin='singlet', dense_limit=2000)
    finally:
        ee_fold._model_roots = orig
    ref = ee_fold.solve_folded(P, 3, spin='singlet', dense_limit=2000,
                               solver='newton')
    d = float(np.max(np.abs(res.omega - ref.omega))) \
        if res.omega.shape == ref.omega.shape else np.inf
    ok &= check(max(sizes) <= 30 and d < 1e-8,
                'dense branch, n 200: the Davidson models at most 30 vectors',
                f'largest model {max(sizes)}, |dw| {d:.1e} Ha against Newton')
    return ok


def check_stream():
    """_stream sums fn over slabs exactly once each, with a last slab shorter than
    the width and more workers than slabs; a slab whose own _ew pass splits over
    as many threads as there are workers returns (on row_map's pool it hangs)."""
    from src.SingleReference.ADC.eeADC import ee_spin_blocks as sb
    seen = []

    def fn(j0, j1):
        seen.append((j0, j1))
        return [sb.SB({'aa': np.full((2, 2), float(j1 - j0))})], float(j1 - j0)

    out, tot = sb._stream(fn, 5, 2, workers=24)
    ok = check(sorted(seen) == [(0, 2), (2, 4), (4, 5)] and tot == 5.0
               and np.array_equal(out[0].get('aa'), np.full((2, 2), 5.0)),
               '_stream: slabs (0,2),(2,4),(4,5) once each, sums exact',
               f'{sorted(seen)}, {tot}')
    a = np.ones((3, 1 << 19))

    def fn_ew(j0, j1):
        return [sb._ew(np.multiply, a, 2.0)], float(j1 - j0)

    keep = os.environ.get('OMP_NUM_THREADS')
    os.environ['OMP_NUM_THREADS'] = '3'
    try:
        out, tot = sb._stream(fn_ew, 3, 1, workers=3)
    finally:
        if keep is None:
            del os.environ['OMP_NUM_THREADS']
        else:
            os.environ['OMP_NUM_THREADS'] = keep
    ok &= check(tot == 3.0 and np.array_equal(out[0], np.full(a.shape, 6.0)),
                '_stream: slabs that split their own _ew over 3 threads, 3 workers',
                f'{tot}')
    # _slab_plan(44 indices, 1000 elements per index, 10 blocks per slab), 3 workers
    keep_gb, keep_w = sb._STREAM_GB, sb._stream_workers
    sb._stream_workers = lambda: 3
    try:
        plans = {}
        for fit in (12500, 10, 4, 0):
            sb._STREAM_GB = (fit + 0.5) * 8 * 1000 * 10 / 1e9
            plans[fit] = sb._slab_plan(44, 1000, 10)
    finally:
        sb._STREAM_GB, sb._stream_workers = keep_gb, keep_w
    ok &= check(plans == {12500: (44, 1), 10: (3, 3), 4: (1, 3), 0: (1, 1)},
                '_slab_plan: one slab when the budget holds all, else workers x width '
                'within it, never below one index', str(plans))
    # the budget follows MBPT_STREAM_GB of the environment, read at import; the
    # child finds src through the repository root this file put on sys.path
    code = ('from src.SingleReference.ADC.eeADC import ee_spin_blocks as sb; '
            'print(sb._STREAM_GB, sb._slab_plan(44, 1000, 10))')
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.pathsep.join(p for p in (root, os.environ.get('PYTHONPATH', '')) if p)
    knob = dict(os.environ, MBPT_STREAM_GB='0.0001', PYTHONPATH=path)   # 0.1 MB
    base = dict({k: v for k, v in os.environ.items() if k != 'MBPT_STREAM_GB'},
                PYTHONPATH=path)
    runs = [subprocess.run([sys.executable, '-c', code], env=e, capture_output=True,
                           text=True) for e in (knob, base)]
    got = [r.stdout.split() for r in runs]
    ok &= check(got[0] == ['0.0001', '(1,', '1)'] and got[1] == ['16.0', '(44,', '1)'],
                '_STREAM_GB: MBPT_STREAM_GB=0.0001 gives one index per slab, unset '
                '16 GB holds all 44', str(got) + ''.join(r.stderr[-120:] for r in runs))
    # the parser behind it
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            silent = [sb._stream_gb(t) for t in (None, '2.5', '1e-3')]
        except RuntimeWarning as w:
            silent = str(w)
    ok &= check(silent == [16.0, 2.5, 1e-3],
                '_stream_gb: unset is 16 GB and a positive number itself, silently',
                str(silent))
    loud = {}
    for text in ('', 'abc', 'nan', 'inf', '-1', '0'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            gb = sb._stream_gb(text)
        loud[text] = (gb, len(caught),
                      bool(caught) and repr(text) in str(caught[0].message))
    ok &= check(all(v == (16.0, 1, True) for v in loud.values()),
                '_stream_gb: an empty, non-numeric, nan, inf or non-positive value is '
                '16 GB with one warning naming it', str(loud))
    return ok


def check_slab_contractions():
    """The slab helpers against the full contractions restricted to the slab (no 5,
    nv 7, width 2, so the last slab is shorter), to 1e-13 relative per block."""
    from src.SingleReference.ADC.eeADC import ee_spin_blocks as sb
    rng = np.random.default_rng(11)
    no, nv, width = 5, 7, 2
    keys = ('aaaa', 'bbbb', 'abab', 'baba', 'abba', 'baab')
    S = rng.standard_normal((nv, no, nv, nv))
    G = sb.anti4(rng.standard_normal((no, no, no, nv)),
                 rng.standard_normal((no, no, no, nv)))
    x = sb.SB({s: rng.standard_normal((no, nv)) for s in ('aa', 'bb')})
    Y = sb.SB({k: rng.standard_normal((no, no, nv, nv)) for k in keys})
    full = sb.ovvv_ijab(S, x)
    full = full - full.transpose(1, 0, 2, 3)

    def rel(a, b):
        return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))

    d_ijab, ia_ovvv, ia_ooov = 0.0, None, None
    for j0 in range(0, no, width):
        j1 = min(j0 + width, no)
        X = sb.ovvv_ijab_pij_slab(S, x, j0, j1)
        d_ijab = max([d_ijab] + [rel(X.get(k), full.get(k)[:, j0:j1]) for k in keys])
        Ys = sb.SB({k: v[:, j0:j1] for k, v in Y.items()})
        a, b = sb.ovvv_ia_slab(S, Ys, j0, j1), sb.ooov_ia_slab(G, Ys, j0, j1)
        ia_ovvv = a if ia_ovvv is None else ia_ovvv + a
        ia_ooov = b if ia_ooov is None else ia_ooov + b
    ref_ovvv, ref_ooov = sb.ovvv_ia(S, Y), sb.ooov_ia(G, Y)
    d_ovvv = max(rel(ia_ovvv.get(s), ref_ovvv.get(s)) for s in ('aa', 'bb'))
    d_ooov = max(rel(ia_ooov.get(s), ref_ooov.get(s)) for s in ('aa', 'bb'))
    ok = check(d_ijab < 1e-13 and sorted(X.keys()) == sorted(keys),
               '(1 - P_ij) ovvv_ijab on slabs of j equals the full one restricted',
               f'{d_ijab:.1e} relative')
    ok &= check(d_ovvv < 1e-13, 'ovvv_ia_slab summed over the slabs equals ovvv_ia',
                f'{d_ovvv:.1e} relative')
    ok &= check(d_ooov < 1e-13, 'ooov_ia_slab summed over the slabs equals ooov_ia',
                f'{d_ooov:.1e} relative')

    class Counted(np.ndarray):
        """S that counts the products taken against it."""
        products = 0

        def __rmatmul__(self, other):
            Counted.products += 1
            return np.matmul(other, self.view(np.ndarray))

    X = sb.ovvv_ijab_pij_slab(S.view(Counted), x, 0, no)
    d_one = max(rel(np.asarray(X.get(k)), full.get(k)) for k in keys)
    ok &= check(d_one < 1e-13 and Counted.products == 1,
                'one slab over every j forms T once, both spins in one product',
                f'{Counted.products} products against S, {d_one:.1e} relative')
    import threading
    calls = {'main': [], 'pool': []}

    def part(r0, r1, who):
        calls[who].append((r0, r1))
        return np.zeros(1)

    keep = os.environ.get('OMP_NUM_THREADS')
    os.environ['OMP_NUM_THREADS'] = '4'
    try:
        sb._ksum(lambda r0, r1: part(r0, r1, 'main'), 8, 1 << 30)
        th = threading.Thread(
            target=lambda: sb._ksum(lambda r0, r1: part(r0, r1, 'pool'), 8, 1 << 30))
        th.start()
        th.join()
    finally:
        if keep is None:
            del os.environ['OMP_NUM_THREADS']
        else:
            os.environ['OMP_NUM_THREADS'] = keep
    ok &= check(len(calls['main']) == 4 and calls['pool'] == [(0, 8)],
                '_ksum splits on the main thread, keeps a product whole off it',
                str(calls))
    return ok


def traced_peak(fn):
    """Bytes tracemalloc saw allocated at once while fn() ran, above its start."""
    import tracemalloc
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


def check_streamed_kernel(eps, B, no):
    """pieces['K'] at adc2 and gf2 against the generic V, divide, Vt path at omega
    0.3 Ha, powers (1, 2) and (), to 1e-12 relative per block, its doubles norm
    against the flat layout's; on one slab, on three of width 2 from the budget, and
    on those three two at a time; the folded roots with and without K to 1e-9 Ha; a
    y1 with a flip parity refused; at most _K_LIVE slab blocks per slab index."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC import ee_spin_blocks as sb
    ok = True
    nv = B.shape[1] - no
    om = 0.3
    rng = np.random.default_rng(13)
    y1 = SB({s: rng.standard_normal((no, nv)) for s in ('aa', 'bb')})

    def rel(a, b):
        return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))

    per, live = no * nv * nv, ee_r_sigma_df._K_LIVE

    def budget(fit):
        # the _STREAM_GB that holds `fit` indices of the doubles in flight
        return (fit + 0.5) * 8 * per * live / 1e9

    runs = (('one slab', sb._STREAM_GB, 1), ('3 slabs', budget(2), 1),
            ('3 slabs, 2 workers', budget(4), 2))
    for level in ('adc2', 'gf2'):
        P = ee_r_sigma_df.build_operator(eps, B, no, level=level, pieces=True)[3]
        be = P['be']
        Yt = be.divide(P['V'](y1), P['D'] - om)
        ref = [P['Vt'](Yt), P['Vt'](be.divide(Yt, P['D'] - om))]
        Yf = ee_fold.doubles_sb_to_flat(Yt, no, nv)
        n2_ref = float(Yf @ Yf)
        dn = abs(ee_r_sigma_df._doubles_norm2(Yt) - n2_ref) / n2_ref
        ok &= check(dn < 1e-12, f'{level}: _doubles_norm2 is the flat-layout norm',
                    f'{dn:.1e} relative')
        for label, gb, workers in runs:
            keep_gb, keep_w = sb._STREAM_GB, sb._stream_workers
            sb._STREAM_GB, sb._stream_workers = gb, (lambda w=workers: w)
            try:
                plan = sb._slab_plan(no, per, live)
                ws, n2 = P['K'](y1, om, (1, 2))
                ws0, n20 = P['K'](y1, om, ())
            finally:
                sb._STREAM_GB, sb._stream_workers = keep_gb, keep_w
            d = max(rel(w.get(s), r.get(s)) for w, r in zip(ws, ref)
                    for s in ('aa', 'bb'))
            dn = max(abs(n2 - n2_ref), abs(n20 - n2_ref)) / n2_ref
            ok &= check(len(ws) == 2 and ws0 == [] and d < 1e-12 and dn < 1e-12,
                        f"{level}, {label}: pieces['K'] equals V, divide, Vt",
                        f'{d:.1e}, norm {dn:.1e} relative, (width, workers) {plan}')
        a = ee_fold.solve_folded(P, 3, spin='singlet', dense_limit=0)
        b = ee_fold.solve_folded(dict(P, K=None), 3, spin='singlet', dense_limit=0)
        dw = float(np.max(np.abs(a.omega - b.omega)))
        ok &= check(a.omega.size == 3 and dw < 1e-9,
                    f"{level}: the folded singlets with and without pieces['K']",
                    f'max |dw| {dw:.1e} Ha')
        try:
            P['K'](SB(y1.blocks, parity=1.0), om, (1,))
            ok &= check(False, f"{level}: pieces['K'] refuses a flip parity")
        except NotImplementedError as exc:
            ok &= check('parity' in str(exc),
                        f"{level}: pieces['K'] refuses a flip parity", str(exc)[:60])
    # slab blocks alive per slab index: the slope of K's traced peak from width 1 to
    # width 2 on water cc-pVTZ, where a block outgrows NumPy's ufunc buffers and the
    # fixed overhead cancels
    _, e3, B3, n3 = water_df('cc-pvtz')
    P3 = ee_r_sigma_df.build_operator(e3, B3, n3, level='adc2', pieces=True)[3]
    v3 = len(e3) - n3
    per3 = n3 * v3 * v3
    y3 = SB({s: rng.standard_normal((n3, v3)) for s in ('aa', 'bb')})
    held = {}
    for width in (1, 2):
        keep_gb, keep_w = sb._STREAM_GB, sb._stream_workers
        sb._STREAM_GB = (width + 0.5) * 8 * per3 * live / 1e9
        sb._stream_workers = lambda: 1
        try:
            held[width] = traced_peak(lambda: P3['K'](y3, om, (1, 2)))
        finally:
            sb._STREAM_GB, sb._stream_workers = keep_gb, keep_w
    slope = (held[2] - held[1]) / (8 * per3)
    ok &= check(slope <= live,
                f"adc2: a slab of pieces['K'] holds at most _K_LIVE = {live} slab "
                'blocks per slab index', f'{slope:.2f}, water cc-pVTZ')
    return ok


def _spin_none_solve(m, C, D0):
    """solve_folded(nroots 4, spin=None, Davidson, tol_omega 1e-10, the precision the
    check asks of it) on _synthetic_pieces(m, C, D0)."""
    from src.SingleReference.ADC.eeADC import ee_fold
    return ee_fold.solve_folded(_synthetic_pieces(m, C, np.array([D0])), 4, spin=None,
                                dense_limit=0, tol_omega=1e-10)


def main():
    """Run every check; print ALL PASSED or FAILURES DETECTED; exit 0 or 1."""
    all_ok = True
    mf, eps, B, no = water_df()
    all_ok &= check_pieces(eps, B, no)
    all_ok &= check_gf2(eps, B, no)
    all_ok &= check_folded_operator(mf, eps, B, no)
    all_ok &= check_gw_hook(eps, B, no)
    all_ok &= check_solve_folded(mf, eps, B, no)
    all_ok &= check_degenerate_set()
    all_ok &= check_routes_and_channels(mf, eps, B, no)
    all_ok &= check_gf2_solves(mf, eps, B, no)
    all_ok &= check_boundaries(mf, eps, B, no)
    all_ok &= check_diis_step()
    all_ok &= check_level_helpers()
    all_ok &= check_collapsing_pair()
    all_ok &= check_split_level()
    all_ok &= check_duplicate_drop()
    all_ok &= check_duplicate_images(eps, B, no)
    all_ok &= check_level_crossing()
    all_ok &= check_eq53_transcription(eps, B, no)
    all_ok &= check_seeding_gap()
    all_ok &= check_index_solve(eps, B, no)
    all_ok &= check_index_bracket()
    all_ok &= check_count_window()
    all_ok &= check_hidden_branch()
    all_ok &= check_dense_default()
    all_ok &= check_warning_paths(eps, B, no)
    all_ok &= check_window_branch()
    all_ok &= check_index_start()
    all_ok &= check_cut_above_doubles()
    all_ok &= check_guess_start()
    all_ok &= check_refusals()
    all_ok &= check_degenerate_hidden()
    all_ok &= check_trace(eps, B, no)
    all_ok &= check_threaded_elementwise()
    all_ok &= check_split_contractions()
    all_ok &= check_kernel_hook(eps, B, no)
    all_ok &= check_model_roots()
    all_ok &= check_davidson_omega(eps, B, no)
    all_ok &= check_solver_default(eps, B, no)
    all_ok &= check_default_polish()
    all_ok &= check_stream()
    all_ok &= check_slab_contractions()
    all_ok &= check_streamed_kernel(eps, B, no)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
