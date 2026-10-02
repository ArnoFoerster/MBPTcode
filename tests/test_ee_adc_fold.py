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
  16. the count's window: a level split by 2e-9 Ha at the cut, tol_omega 1e-10,
      counts whole, so a state folded below it is found and none warns; a
      distinct partner 1e-7 or 1e-6 Ha above the cut, not asked for, costs no
      warning and one count, widened once, at default tolerances.
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
      state is solved, both roots exact, no warning, on both branches.
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

Run: python tests/test_ee_adc_fold.py
"""
import os
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
        for r in range(3):
            d = abs(res.omega[r] - e_full[r]) * ev
            ok &= check(d < 1e-5, f'{spin} root {r}: folded equals full ADC(2)',
                        f'{res.omega[r] * ev:.6f} vs {e_full[r] * ev:.6f} eV, '
                        f'|d| {d:.1e} eV')
            z = np.asarray(Z[:, r])
            t1_full = float(z[:n_s] @ z[:n_s]) / float(z @ z)
            ok &= check(abs(res.t1[r] - t1_full) < 1e-4, f'{spin} root {r}: T1',
                        f'{res.t1[r]:.5f} vs {t1_full:.5f}')
            newton = (res.converged[r] and res.steps[r] <= 8
                      and res.loop[r] == 'newton')
            ok &= check(newton, f'{spin} root {r}: Newton under eight steps',
                        f'{res.steps[r]} steps, loop {res.loop[r]}')
        # the fixed-point loop reaches the same roots
        res_fp = ee_fold.solve_folded(P, 3, spin=spin, t_min=1.0)
        d = float(np.max(np.abs(res_fp.omega - e_full[:3]))) * ev
        ok &= check(d < 1e-5 and all(lp == 'fixed' for lp in res_fp.loop),
                    f'{spin}: fixed-point loop reaches the same roots',
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
                                         dense_limit=dense_limit)
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
        res = ee_fold.solve_folded(P, 1, spin='singlet', max_newton=1)
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
        res = ee_fold.solve_folded(P, 2, spin='singlet', tol_omega=1e-12)
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
                                   dense_limit=0)
    dup = any('landed on one root' in str(x.message) for x in caught)
    d = np.abs(res.omega - pair).max() if res.omega.size == 2 else np.inf
    ok &= check(dup and d < 1e-8,
                'Davidson, tol_omega 1e-12 at tol_residual 1e-6: the collapse is '
                'caught and both roots come back', f'max |dw| {d:.1e} Ha')
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
        res = ee_fold.solve_folded(P, 2, spin='singlet', tol_omega=1e-12)
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
    """The count's window against a level and against an unrequested partner: a
    pair of M split by 2e-9 Ha (one level, solved at its mean) is the cut at
    tol_omega 1e-10, with and without a state that folds below it; a distinct
    partner 1e-7 or 1e-6 Ha above the cut, not asked for, at default tolerances:
    the partner lies in the window, so the count's Davidson widens once past it."""
    from src.SingleReference.ADC.eeADC import ee_fold
    ok = True
    m, D = np.array([0.5, 0.5 + 2e-9, 0.9, 1.5]), np.array([1.6])
    for hidden in (True, False):
        C = np.array([[0.0, 0.0, 0.0, 1.149 if hidden else 0.0]])
        w, _ = _synthetic_exact(m, C, D)
        nw = 3 if hidden else 2
        for dense_limit, route in ((2000, 'dense'), (0, 'Davidson')):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                res = ee_fold.solve_folded(_synthetic_pieces(m, C, D), 2,
                                           spin='singlet', tol_omega=1e-10,
                                           dense_limit=dense_limit)
            d = np.abs(res.omega - w[:nw]).max() if res.omega.size == nw else np.inf
            what = 'a state folded below it is found' if hidden else 'none warns'
            ok &= check(d < 2e-9 and not caught,
                        f'{route}: a level split by 2e-9 Ha at the cut, tol_omega '
                        f'1e-10, counts whole; {what}',
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
            res = ee_fold.solve_folded(P, nroots, spin='singlet', **kw)
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
        for dl in (2000, 0):
            got, d, msgs = _fold_run(np.array([0.5, 0.6, 0.9]), C, np.array([1.2]), 2,
                                     tol_omega=tol_omega, dense_limit=dl)
            ok &= check(d < 1e-8 and not msgs,
                        f'delta {delta:+.0e}, tol_omega {tol_omega:.0e}, dense_limit '
                        f'{dl}: both roots, no warning',
                        f'got {np.round(got, 9).tolist()}, {len(msgs)} warning(s)')
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
                                 3, dense_limit=dl)
        ok &= check(d < 1e-8 and not msgs,
                    f'dense_limit {dl}: nroots 3 past min D returns 0.5, 0.55, 0.6',
                    f'got {np.round(got, 9).tolist()}, {len(msgs)} warning(s)')
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
                                   t_min=0.999, max_newton=1, max_fixed=1)
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
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
