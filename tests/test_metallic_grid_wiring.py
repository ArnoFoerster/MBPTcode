"""The finite-temperature grids, wired into the routes that consume them.

src/Base/utils/matsubara.py built `thermal_e_min`, `IRBasis` and
`ir_continuation_order` and nothing called them. This is the wiring: `beta`
threaded through to grid construction in both the molecular
(GW/imaginary_axis) and periodic (Periodic/pbc_rpa) routes, plus the
IR-derived cap on the Pade continuation order.

The obvious reading of the remaining risk is that the grids.py guards raise, so
any metallic run stops until a beta is threaded through. Check 2 shows that is
only half of it, and which half you get is an accident of the
basis: `_transition_window` applies one integer occupied count at every
k-point, and on a Li monolayer that gives a NEGATIVE e_min in gth-dzvp (the
guard fires, the run stops) and a POSITIVE, plausible, fictitious one in
gth-szv (nothing fires, the grid is quietly mis-scaled). Both are tested.

Checks:
  1. `beta_from_mf` reads a Fermi-Dirac smearing width off the mean field, and
     refuses a Gaussian one -- a broadening parameter is not a temperature.
  2. BOTH FAILURE MODES of the integer window, on the same metal in two
     bases: gth-dzvp gives e_min = -0.0418 Ha and the guard fires; gth-szv
     gives +0.0888 Ha against an honest 0.0673 and nothing fires. mo_occ
     repairs the silent one, mo_occ + beta the loud one.
  3. The T = 0 path is untouched: on a gapped system thermal_e_min returns the
     gap itself for any beta whose pi/beta is below it, so grids, and every
     number downstream, are bit-identical.
  4. The Pade order cap: truncation happens after the greedy ordering, is a
     no-op when the cap exceeds the sample count, and refuses a cap too small
     to fit.
  5. End to end: the periodic RPA correlation energy of a METAL, which is the
     run the whole package exists to make possible, and its dependence on the
     grid scale that the wiring fixes.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto as mgto, scf as mscf
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.scf import addons as pbc_addons

from src.Base.utils.matsubara import (beta_from_mf, ir_continuation_order,
                                      self_energy_range,
                                      thermal_e_min)
from src.SingleReference.GW.qp_solve import solve_qp_from_imaginary_axis
from src.SingleReference.Periodic.pbc_rpa import (build_freq_grid,
                                                  ri_rpa_ecorr_from_dfints)
from src.SingleReference.Periodic.pbc_rpa_damping import (nyquist_params,
                                                          make_coulG_damped)
from src.SingleReference.Periodic.pbc_damped_integrals import build_dfintegrals_coulG
from src.SingleReference.Periodic.pbc_occupations import transition_window


def check(ok, label, detail=''):
    """Report a condition AND enforce it.

    pytest DISCARDS whatever a test function returns, so a suite built out of
    `return ok` passes whether ok is True or False -- verified on pytest 9.1.1.
    Asserting here fixes every call site at once, and makes the script path
    stop at a genuine failure instead of printing FAIL and exiting 0.
    """
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    assert ok, f"{label}{(' -- ' + detail) if detail else ''}"
    return ok


def li_layer(kmesh=(3, 3, 1), sigma=0.02, basis='gth-dzvp'):
    cell = pgto.Cell()
    cell.atom = 'Li 0 0 0; Li 1.75 1.75 0'
    cell.a = np.diag([3.5, 3.5, 16.0])
    cell.basis = basis
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = [15, 15, 60]
    cell.verbose = 0
    cell.build()
    mf = pscf.KRHF(cell, cell.make_kpts(list(kmesh)), exxdiv=None).density_fit()
    mf = pbc_addons.smearing_(mf, sigma=sigma, method='fermi')
    mf.conv_tol = 1e-9
    mf.kernel()
    return cell, mf


def h2_slab(kmesh=(2, 2, 1)):
    cell = pgto.Cell()
    cell.atom = 'H 0 0 -0.37; H 0 0 0.37'
    cell.a = np.diag([4.0, 4.0, 24.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = [13, 13, 72]
    cell.verbose = 0
    cell.build()
    mf = pscf.KRHF(cell, cell.make_kpts(list(kmesh)), exxdiv=None).density_fit()
    mf.kernel()
    return cell, mf


# ------------------------------------------------------------ 1. beta_from_mf

@pytest.fixture(scope='module')
def _li():
    return li_layer()


@pytest.fixture(scope='module')
def cell(_li):
    return _li[0]


@pytest.fixture(scope='module')
def mf(_li):
    return _li[1]


@pytest.fixture(scope='module')
def mf_metal(_li):
    return _li[1]


def test_beta_from_mf(mf_metal):
    ok = check(abs(beta_from_mf(mf_metal) - 1.0 / mf_metal.sigma) < 1e-12,
               'beta_from_mf reads 1/sigma off a Fermi-smeared mean field',
               f'sigma {mf_metal.sigma} -> beta {beta_from_mf(mf_metal):.1f}')

    mol = mgto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
    plain = mscf.RHF(mol)
    plain.kernel()
    ok &= check(beta_from_mf(plain) is None,
                'and returns None for an unsmeared one, so T=0 stays T=0')
    ok &= check(beta_from_mf(plain, default=7.0) == 7.0,
                'honouring an explicit default')

    class _Gaussian:
        sigma = 0.01
        smearing_method = 'gaussian'
    try:
        beta_from_mf(_Gaussian())
    except ValueError as err:
        ok &= check('not a temperature' in str(err) or 'Fermi' in str(err),
                    'a Gaussian smearing width is refused -- a broadening '
                    'parameter has no 1/kT reading')
    else:
        ok &= check(False, 'a Gaussian smearing width is refused')
    return ok


# ------------------------------------------------- 2. both failure modes

def test_both_failure_modes():
    """The integer window fails loudly on one basis and silently on another."""
    from src.SingleReference.Periodic.pbc_rpa import _transition_window
    ok = True
    results = {}
    print(f"       {'basis':10s} {'per-k nocc':>14} {'integer e_min':>14} "
          f"{'honest e_min':>13} {'guard':>7}")
    for basis in ('gth-dzvp', 'gth-szv'):
        cell, mf = li_layer(basis=basis)
        e = np.asarray(mf.mo_energy)
        f = np.asarray(mf.mo_occ, dtype=float)
        per_k = (f > 1e-8).sum(axis=1)
        n0 = int(per_k[0])
        e_int, _ = _transition_window(list(e), n0)
        honest, _ = transition_window(e, f)
        try:
            build_freq_grid(list(e), n0, 16)
            fired = False
        except ValueError:
            fired = True
        results[basis] = (e_int, honest, fired, e, f, n0, mf)
        print(f"       {basis:10s} {str(sorted(set(per_k.tolist()))):>14} "
              f"{e_int:+14.6f} {honest:13.6f} {'RAISES' if fired else 'silent':>7}")

    e_int, honest, fired, e, f, n0, mf = results['gth-dzvp']
    ok &= check(e_int < 0 and fired,
                'gth-dzvp: the ragged collapse gives a NEGATIVE e_min and the '
                'guard fires -- the loud case', f'{e_int:+.6f} Ha')

    e_int, honest, fired, e, f, n0, mf = results['gth-szv']
    ok &= check(e_int > 0 and not fired,
                'gth-szv: the same construction gives a POSITIVE fictitious '
                'gap and nothing fires at all', f'{e_int:+.6f} Ha')
    ok &= check(honest < e_int,
                'which is larger than the honest occupation-weighted scale',
                f'{e_int:.6f} vs {honest:.6f} Ha')

    # mo_occ alone repairs the silent case; beta alone repairs the loud one.
    ref = build_freq_grid(list(e), n0, 16)
    fixed = build_freq_grid(list(e), n0, 16, mo_occ=f)
    ok &= check(fixed[0].min() < ref[0].min(),
                'passing mo_occ moves the silent case onto the honest scale',
                f'{ref[0].min():.4e} -> {fixed[0].min():.4e}')

    e_int, honest, fired, e, f, n0, mf = results['gth-dzvp']
    beta = beta_from_mf(mf)
    grid = build_freq_grid(list(e), n0, 16, mo_occ=f, beta=beta)
    ok &= check(np.all(np.isfinite(grid[0])) and grid[0].min() > 0,
                'and mo_occ + beta make the loud case build a grid at all',
                f'{len(grid[0])} points, lowest {grid[0].min():.4e}')
    ok &= check(thermal_e_min(beta, honest) == thermal_e_min(beta, 0.0),
                'there the temperature, not the gap, sets the scale',
                f'honest {honest:.6f} < pi/beta {thermal_e_min(beta, 0.0):.6f}')
    return ok


# --------------------------------------------------- 3. T = 0 stays untouched

def test_gapped_is_untouched():
    cell, mf = h2_slab()
    e = np.asarray(mf.mo_energy)
    f = np.asarray(mf.mo_occ, dtype=float)
    nocc = int((f > 1e-8).sum(axis=1)[0])
    occ = np.concatenate([ee[:nocc] for ee in e])
    vir = np.concatenate([ee[nocc:] for ee in e])
    gap = vir.min() - occ.max()

    ref = build_freq_grid(list(e), nocc, 16)
    ok = check(gap > 0, 'the slab is gapped', f'{gap:.4f} Ha')
    ok &= check(beta_from_mf(mf) is None, 'and unsmeared, so beta is None')

    # Any beta whose pi/beta lies below the gap must leave the grid alone.
    for beta in (10.0, 50.0, 1e4):
        if np.pi / beta >= gap:
            continue
        got = build_freq_grid(list(e), nocc, 16, beta=beta)
        ok &= check(np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1]),
                    f'beta={beta:g}: the grid is bit-identical to the T=0 one')
    # And mo_occ is a no-op when the occupations are integers.
    got = build_freq_grid(list(e), nocc, 16, mo_occ=f)
    ok &= check(np.array_equal(got[0], ref[0]),
                'passing integer mo_occ is a no-op')

    # A small beta (high temperature) SHOULD move it -- otherwise the knob is
    # not connected at all.
    hot = build_freq_grid(list(e), nocc, 16, beta=2.0)
    ok &= check(not np.array_equal(hot[0], ref[0]),
                'but a temperature above the gap does move it',
                f'pi/beta = {np.pi / 2.0:.3f} vs gap {gap:.3f}')
    return ok


# ------------------------------------------------------- 4. the Pade order cap

def test_pade_order_cap():
    # A synthetic Sigma_c with two poles, sampled on a vertical line.
    z = 0.0 + 1j * np.linspace(0.05, 4.0, 24)
    sigma = 1.0 / (z - (-0.4 + 0.05j)) + 0.5 / (z - (0.9 + 0.05j))
    eps = np.array([-0.3, 0.2])

    full = solve_qp_from_imaginary_axis(eps, 0, 0.0, z, sigma, greedy=True)
    capped = solve_qp_from_imaginary_axis(eps, 0, 0.0, z, sigma, greedy=True,
                                          max_order=len(z))
    ok = check(abs(full - capped) < 1e-12,
               'a cap at the sample count is a no-op', f'{abs(full - capped):.1e}')

    tight = solve_qp_from_imaginary_axis(eps, 0, 0.0, z, sigma, greedy=True,
                                         max_order=8)
    ok &= check(np.isfinite(tight) and abs(tight - full) < 0.5,
                'and a real cap still solves, near the uncapped root',
                f'{full:.6f} vs {tight:.6f} with 8 of {len(z)} nodes')

    try:
        solve_qp_from_imaginary_axis(eps, 0, 0.0, z, sigma, max_order=1)
    except ValueError as err:
        ok &= check('too few points' in str(err),
                    'a cap below two nodes is refused')
    else:
        ok &= check(False, 'a cap below two nodes is refused')

    for beta, wmax in ((50.0, 3.0), (200.0, 3.0), (50.0, 30.0)):
        n = ir_continuation_order(beta, wmax)
        ok &= check(2 < n < 200,
                    f'ir_continuation_order(beta={beta:g}, wmax={wmax:g}) is a '
                    f'usable node count', f'{n}')
    ok &= check(ir_continuation_order(200.0, 3.0) > ir_continuation_order(50.0, 3.0),
                'and grows with Lambda = beta*wmax, as the basis does')
    return ok



# --------------------------------------------- 4b. what the cap is sized FROM

def test_pade_cap_is_sized_from_sigma_not_from_w():
    """The continuation acts on Sigma_c, so its range -- not W's -- sets the cap.

    e_max is the largest particle-hole TRANSITION energy: the range of W, and
    the quantity the frequency grid is built for. It is the natural thing to
    reach for and it is wrong here, because Sigma = -G Wt is a product whose
    poles sit at eps_m +/- omega_s. Passing e_max under-resolves Sigma, and
    because the IR size grows only logarithmically in Lambda the error is a few
    nodes rather than a blow-up -- small enough to never announce itself.
    """
    eps = np.array([-0.90, -0.55, -0.30, 0.25, 0.60, 1.40])
    mu = 0.5 * (eps[2] + eps[3])
    e_max = eps.max() - eps.min()          # widest transition = W's range
    w_sigma = self_energy_range(eps, mu, e_max)

    # Definitional: the outermost pole of a convolution of G (poles at eps_m)
    # with W (poles out to e_max) sits at max|eps - mu| + e_max. Built here by
    # explicit pole arithmetic rather than asserted.
    poles = [abs((e - mu) + s * w) for e in eps for w in (0.0, e_max)
             for s in (+1, -1)]
    ok = check(abs(max(poles) - w_sigma) < 1e-12,
               'self_energy_range is where the outermost Sigma pole actually is',
               f'{max(poles):.6f} vs {w_sigma:.6f}')
    ok &= check(w_sigma > e_max,
                'and it exceeds the screening range it is built from',
                f'{w_sigma:.4f} against e_max = {e_max:.4f}, '
                f'ratio {w_sigma / e_max:.2f}')

    beta = 100.0
    n_w = ir_continuation_order(beta, e_max)
    n_sigma = ir_continuation_order(beta, w_sigma)
    ok &= check(n_sigma > n_w,
                'so sizing from e_max discards resolvable Pade nodes',
                f'{n_w} nodes from W against {n_sigma} from Sigma, '
                f'{n_sigma - n_w} lost')

    # The wiring itself: capture the wmax solve_qp_energy_imaginary_axis passes.
    # Without the fix this is e_max and the test fails on the value.
    from pyscf import gto, scf
    import src.SingleReference.GW.imaginary_axis as ia

    mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    seen = []
    real_order = ia.ir_continuation_order
    ia.ir_continuation_order = lambda b, w, **kw: (seen.append((b, w)),
                                                   real_order(b, w, **kw))[1]
    try:
        ia.solve_qp_energy_imaginary_axis(mf, mol, mol.nelectron // 2,
                                          mol.nelectron // 2 - 1, nfreq=12,
                                          beta=100.0)
    finally:
        ia.ir_continuation_order = real_order

    ok &= check(len(seen) == 1, 'the cap is sized exactly once', f'{len(seen)}')
    if seen:
        eps_mf = mf.mo_energy
        n_occ = mol.nelectron // 2
        mu_mf = 0.5 * (eps_mf[n_occ - 1] + eps_mf[n_occ])
        w_screen = eps_mf[n_occ:].max() - eps_mf[:n_occ].min()
        expect = self_energy_range(eps_mf, mu_mf, w_screen)
        got = seen[0][1]
        ok &= check(abs(got - expect) < 1e-10,
                    "and from SIGMA's range, not from e_max",
                    f'wmax = {got:.4f}; Sigma range {expect:.4f}, '
                    f'screening range {w_screen:.4f}')
    return ok


# ------------------------------------------------------------- 5. end to end

def test_metallic_rpa_runs(cell, mf):
    """The run the package exists to make possible."""
    kmesh = [3, 3, 1]
    r0, beta_d, _ = nyquist_params(cell, kmesh)
    dfints = build_dfintegrals_coulG(mf, coulG_fn=make_coulG_damped(r0, beta_d))
    beta = beta_from_mf(mf)

    ec_th = ri_rpa_ecorr_from_dfints(dfints, nw=16, beta=beta)
    ok = check(np.isfinite(ec_th) and ec_th < 0,
               'RPA Ec of a metal runs on the thermal grid and is negative',
               f'{ec_th:.8f} Ha')

    # The naive route -- integer window, no temperature -- runs too, and gives
    # a different number. That is the silent failure this package removes.
    ec_naive = ri_rpa_ecorr_from_dfints(dfints, nw=16, mo_occ=None)
    ok &= check(np.isfinite(ec_naive),
                'the naive integer-window route also runs, without complaint',
                f'{ec_naive:.8f} Ha')
    ok &= check(abs(ec_th - ec_naive) > 1e-6,
                'and the two disagree, so the grid scale is not a detail',
                f'{100 * abs(ec_th / ec_naive - 1):.2f}% apart')

    # Convergence, not a fixed spread, is the honest claim here. A metal on a
    # thermal grid has structure all the way down to pi/beta, so it needs more
    # frequency points than a gapped system does, and at these sizes it is
    # still converging. What must be true is that it IS converging: successive
    # refinements have to shrink.
    ns = (12, 16, 24, 32)
    vals = [ri_rpa_ecorr_from_dfints(dfints, nw=n, beta=beta) for n in ns]
    steps = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    print('       nw ' + '/'.join(str(n) for n in ns) + ' on the thermal grid: '
          + '  '.join(f'{v:.8f}' for v in vals))
    print('       successive changes: ' + '  '.join(f'{s:.2e}' for s in steps))
    ok &= check(all(steps[i + 1] < steps[i] for i in range(len(steps) - 1)),
                'and the frequency integration converges -- each refinement '
                'moves it less than the last',
                ' -> '.join(f'{s:.1e}' for s in steps))
    ok &= check(steps[-1] < 3e-3 * abs(vals[-1]),
                'to well under a percent by nw = 32',
                f'{100 * steps[-1] / abs(vals[-1]):.3f}% of Ec')
    return ok


if __name__ == '__main__':
    all_ok = True
    print('\n-- 3. the T=0 path is untouched')
    all_ok &= test_gapped_is_untouched()

    print('\n-- 4. the Pade order cap')
    all_ok &= test_pade_order_cap()
    all_ok &= test_pade_cap_is_sized_from_sigma_not_from_w()

    cell_m, mf_m = li_layer()
    print('\n-- 1. beta_from_mf')
    all_ok &= test_beta_from_mf(mf_m)

    print('\n-- 2. both failure modes of the integer window')
    all_ok &= test_both_failure_modes()

    print('\n-- 5. a metallic RPA correlation energy, end to end')
    all_ok &= test_metallic_rpa_runs(cell_m, mf_m)

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
