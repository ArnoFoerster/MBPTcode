"""Frequency grids that survive a metal.

Every imaginary-axis grid in Base/utils/grids.py is a T = 0 construction scaled
by the smallest transition energy -- the gap. For a metal there isn't one, and
the failures are silent. src/Base/utils/matsubara.py supplies the two things
that fixes: the thermal e_min that makes the existing quadratures well defined
again, and the intermediate representation, which has no gap in its
construction at all.

References: Shinaoka, Otsuki, Ohzeki, Yoshimi, PRB 96, 035147 (2017) for the
IR basis; Li, Wallerberger, Chikano, Yeh, Gull, Shinaoka, PRB 101, 035144
(2020) for sparse sampling -- Sec. II B for the tau points and Sec. II C for
the Matsubara points.

Checks:
  1. The T = 0 grids now REFUSE a vanishing, zero or inverted gap instead of
     returning a collapsed grid, negative frequencies and negative weights.
     This is the test that fails without the guard.
  2. thermal_e_min restores them: at finite temperature a metal gets a valid,
     strictly positive grid with positive weights.
  3. The IR basis is correct as a basis: u and v orthonormal, the kernel
     reconstructed to its truncation level, and the singular values decaying
     exponentially.
  4. Its size converges with the quadrature and grows like log(Lambda) -- the
     property that fails if the imaginary-time panels are refined toward the
     middle instead of toward tau = 0 and tau = beta.
  5. uhat matches the paper's parity rule: purely imaginary for even l, purely
     real for odd l.
  6. Fit-and-predict: sampling a Green's function at the sparse points and
     evaluating it OFF that set reproduces it to the basis tolerance -- for a
     gapped spectrum and, identically, for metallic ones with weight down to
     omega = 0. That equality is the whole point.
  7. Sampling counts are O(L) and the fit is well conditioned, with the
     real-coefficient stacking that the alternating real/imaginary columns of
     uhat require.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.utils.grids import (minimax_frequency_grid, minimax_time_grid,
                                  gauss_legendre_grid, gap_scaled_w0)
from src.Base.utils.matsubara import (IRBasis, thermal_e_min, matsubara_frequencies,
                                      ir_continuation_order, _log_kernel)

WMAX = 10.0


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


def green(n, beta, poles, weights):
    iw = 1j * matsubara_frequencies(n, beta)
    return sum(w / (iw - p) for p, w in zip(poles, weights))


def test_t0_grids_refuse_a_metal():
    # a degenerate Fermi level, which is what a metal looks like to these
    # helpers: eps[nocc] - eps[nocc-1] is ~0 and can come out negative
    eps = np.array([-1.0, -0.5, -0.2, -0.2000001, 0.3, 0.9])
    cases = (('gap_scaled_w0, inverted gap', lambda: gap_scaled_w0(eps, 3)),
             ('minimax_frequency_grid, gap = 0', lambda: minimax_frequency_grid(16, 0.0, 2.0)),
             ('minimax_time_grid, gap < 0', lambda: minimax_time_grid(16, -1e-7, 2.0)),
             ('gauss_legendre_grid, w0 < 0', lambda: gauss_legendre_grid(8, w0=-5e-8)))
    ok = True
    for label, fn in cases:
        try:
            fn()
        except ValueError:
            ok &= check(True, f'refused: {label}')
        else:
            ok &= check(False, f'refused: {label}', 'returned a grid instead')
    return ok


def test_thermal_e_min_restores_them():
    beta = 316.0                                  # ~1000 K
    e_min = thermal_e_min(beta, gap=0.0)
    ok = check(abs(e_min - np.pi / beta) < 1e-12,
               'thermal_e_min falls back on the first Matsubara frequency',
               f'{e_min:.6f} Ha')
    fp, fw = minimax_frequency_grid(16, e_min, 2.0)
    ok &= check((fp > 0).all() and (fw > 0).all() and len(fp) == 16,
                'a metal then gets a valid minimax grid',
                f'omega in [{fp.min():.4f}, {fp.max():.3f}], all weights > 0')
    ok &= check(thermal_e_min(beta, gap=5.0) == 5.0,
                'a gapped system at low T keeps its own gap')
    return ok


def test_basis_is_a_basis():
    b = IRBasis(1000.0, eps=1e-10)
    gu = (b.u * b.wx) @ b.u.T
    gv = (b.v * b.wy) @ b.v.T
    ok = check(max(np.abs(gu - np.eye(b.size)).max(),
                   np.abs(gv - np.eye(b.size)).max()) < 1e-12,
               'u and v are orthonormal under their quadratures')
    kernel = np.exp(_log_kernel(b.x, b.y, b.lambda_))
    rec = (b.u.T * b._s_abs) @ b.v
    ok &= check(np.abs(kernel - rec).max() < 1e-9,
                'the SVD reconstructs the kernel', f'{np.abs(kernel - rec).max():.1e}')
    ratios = b.s[1:] / b.s[:-1]
    ok &= check((b.s > 0).all() and (ratios < 1.0).all() and b.s[-1] < 1e-9,
                'singular values decay monotonically and exponentially',
                f's[0]=1, s[-1]={b.s[-1]:.1e}')
    return ok


def test_size_converges_and_scales():
    sizes = {}
    for lam in (100.0, 1000.0, 10000.0, 100000.0):
        per_quad = [IRBasis(lam, eps=1e-8, order=o, n_extra=e).size
                    for o, e in ((16, 2), (24, 4), (32, 6))]
        sizes[lam] = per_quad
    ok = check(all(len(set(v)) == 1 for v in sizes.values()),
               'the basis size is converged w.r.t. the quadrature',
               '; '.join(f'{int(k)}:{v}' for k, v in sizes.items()))
    got = [sizes[k][0] for k in (100.0, 1000.0, 10000.0, 100000.0)]
    steps = np.diff(got)
    ok &= check((steps > 0).all() and steps.std() < 0.25 * steps.mean(),
                'and grows linearly in log(Lambda)',
                f'sizes {got}, per-decade {steps.tolist()}')
    return ok


def test_uhat_parity():
    b = IRBasis(1000.0, eps=1e-10)
    uh = b.uhat(np.arange(0, 40))
    ok = True
    for l in range(4):
        r, i = np.abs(uh[l].real).max(), np.abs(uh[l].imag).max()
        want_imag = (l % 2 == 0)
        got_imag = i > r
        ok &= check(got_imag == want_imag and min(r, i) < 1e-12 * max(r, i),
                    f'uhat_{l} is purely {"imaginary" if want_imag else "real"}',
                    f'|Re|={r:.1e} |Im|={i:.1e}')
    return ok


def test_fit_and_predict():
    """The gapped and the metallic spectra must reconstruct equally well."""
    rng = np.random.default_rng(0)
    spectra = [('gapped, single pole at 3.0', [3.0], [1.0]),
               ('metallic, poles at +-0.05', [0.05, -0.05], [0.5, 0.5]),
               ('metallic continuum, 200 poles incl. ~0',
                list(rng.uniform(-9.9, 9.9, 200)), list(np.full(200, 1 / 200)))]
    n_test = np.arange(-6000, 6000, 11)
    ok = True
    for beta, eps, tol in ((100.0, 1e-6, 1e-4), (100.0, 1e-10, 1e-7),
                           (1000.0, 1e-10, 1e-6)):
        b = IRBasis(beta * WMAX, eps=eps)
        n_s = b.default_matsubara_sampling()
        errs = {}
        for label, poles, wts in spectra:
            coeffs = b.fit_matsubara(n_s, green(n_s, beta, poles, wts))
            exact = green(n_test, beta, poles, wts)
            errs[label] = (np.abs(b.evaluate_matsubara(coeffs, n_test) - exact).max()
                           / np.abs(exact).max())
        worst = max(errs.values())
        ok &= check(worst < tol, f'beta={beta:g} eps={eps:.0e}: all spectra to tolerance',
                    f'L={b.size} npts={len(n_s)} worst={worst:.1e}')
        gapped = errs['gapped, single pole at 3.0']
        metallic = errs['metallic continuum, 200 poles incl. ~0']
        ok &= check(metallic < 50 * max(gapped, 1e-15),
                    f'beta={beta:g} eps={eps:.0e}: metal is no worse than the gapped case',
                    f'{metallic:.1e} vs {gapped:.1e}')
    return ok


def test_tau_transport_jacobian():
    """fit_matsubara's coefficients are dimensionless: u_at needs (2/beta).

    `uhat` integrates over x in [-1, 1] rather than tau in [0, beta], so a
    Matsubara fit transported onto the tau axis is long by dtau/dx = beta/2.
    The reason this needs a test rather than a comment is the second check
    below: the fit -> evaluate_matsubara round trip is self-consistent, so the
    factor cancels there and the obvious test passes while the mixed path is
    wrong. It cost a self-energy calculation a factor of 200 at beta = 400
    before it was found -- a clean constant, invisible to a beta scan because
    the error is beta-dependent but not beta-convergent.
    """
    rng = np.random.default_rng(0)
    ok = True
    for beta, wmax in ((100.0, 4.0), (200.0, 4.0), (400.0, 2.0), (800.0, 1.0)):
        b = IRBasis(beta * wmax, eps=1e-8, statistics='fermion')
        c = rng.standard_normal(b.size)
        n = b.default_matsubara_sampling(positive_only=True)
        # the PHYSICAL transform carries dtau = (beta/2) dx
        c_fit = b.fit_matsubara(n, (beta / 2.0) * (c @ b.uhat(n)))
        ratio = float(np.median(c_fit / c))
        # 1e-10, not 1e-12: the Jacobian is exact but `fit_matsubara` is a
        # least-squares solve, and its conditioning is the floor. Individual
        # coefficients scatter by up to 4e-5 here; the median lands within
        # 1.4e-12, so a 1e-12 bound is asking the round-off to fall the right
        # way. Measured worst case 1.4e-12 over beta = 100..800.
        ok &= check(abs(ratio - beta / 2) / (beta / 2) < 1e-10,
                    f'beta={beta:g}: the mixed path is off by exactly beta/2',
                    f'{ratio:.6f} vs {beta / 2:.1f}')

    b = IRBasis(400.0, eps=1e-8)
    c = rng.standard_normal(b.size)
    n = b.default_matsubara_sampling(positive_only=True)
    back = b.fit_matsubara(n, b.evaluate_matsubara(c, n))
    # 1e-3, not 1e-10. What is under test is that NO beta/2 appears here: a
    # missing Jacobian shows as an error of order 200 at beta = 400, so any
    # bound far below 1 settles it. The fit's own floor at eps=1e-8 is ~3e-6
    # (measured over lambda = 100..1600), and 1e-10 was never reachable -- the
    # test simply had never run.
    ok &= check(np.abs(back - c).max() < 1e-3,
                'while fit -> evaluate_matsubara needs NO Jacobian, which is '
                'why the round-trip test cannot catch this',
                f'{np.abs(back - c).max():.1e}')
    return ok


def test_sampling_is_sparse_and_conditioned():
    ok = True
    for beta in (10.0, 100.0, 1000.0):
        b = IRBasis(beta * WMAX, eps=1e-10)
        n_s = b.default_matsubara_sampling()
        t_s = b.default_tau_sampling()
        cond = b.condition_number(n_s)
        ok &= check(len(n_s) <= 3 * b.size and cond < 1e12,
                    f'beta={beta:g}: Matsubara sampling is O(L) and conditioned',
                    f'L={b.size} npts={len(n_s)} ({len(n_s) / b.size:.1f}L) cond={cond:.1e}')
        ok &= check(len(t_s) == b.size and (np.abs(t_s) < 1).all(),
                    f'beta={beta:g}: tau sampling gives exactly L interior points',
                    f'{len(t_s)} for L={b.size}')
    # the real-coefficient stacking is what makes one-sided sampling usable
    b = IRBasis(1000.0, eps=1e-10)
    n_s = b.default_matsubara_sampling()
    ok &= check(b.condition_number(n_s, real=False) > 1e4 * b.condition_number(n_s),
                'a complex fit over the same one-sided points is far worse conditioned',
                f'{b.condition_number(n_s, real=False):.1e} vs {b.condition_number(n_s):.1e}')
    return ok


SAMPLING_SWEEP = [(lam, eps) for lam in (100, 200, 400, 800, 1600)
                  for eps in (1e-6, 1e-8, 1e-10, 1e-12)]


def test_sampling_determines_the_fit():
    """Every (Lambda, eps) must give a sampling set that PINS the coefficients.

    This is the gate on the defect the module was shipped with. The paper's
    count -- one point per sign change of uhat_{L-1}, halved by positive_only
    -- is marginal by construction: it lands on 2 (L/2) = L against L unknowns
    and tips either way. Ten of these twenty bases came back underdetermined,
    with the worst coefficient out by 0.05 to 0.89 and no warning, because
    `lstsq` returns the minimum-norm solution and it reproduces the sampled
    values exactly.

    Recovering random coefficients is the sharp form of the test: an arbitrary
    c in R^L probes the whole space, where a physical Green's function may sit
    in the part of it that happens to survive.

    Tolerances are the measured floor after the fix, worst over the sweep:
    1.2e-12 on the coefficients and cond 5.8e3 (both at Lambda = 1600,
    eps = 1e-12), against 8.9e-01 and 6.0e+09 before it.
    """
    rng = np.random.default_rng(0)
    ok = True
    for lam, eps in SAMPLING_SWEEP:
        b = IRBasis(lam, eps=eps)
        c = rng.standard_normal(b.size)
        n = b.default_matsubara_sampling(positive_only=True)
        determined, nrows, cond = b.sampling_is_determined(n, real=True)
        err = float(np.abs(b.fit_matsubara(n, b.evaluate_matsubara(c, n)) - c).max())
        ok &= check(determined and err < 1e-11,
                    f'lambda={lam} eps={eps:.0e}: the one-sided set determines the fit',
                    f'L={b.size} npts={len(n)} rows={nrows} cond={cond:.1e} err={err:.1e}')
        # the row count is the criterion the shipped set failed, so pin it
        ok &= check(2 * len(n) >= b.size,
                    f'lambda={lam} eps={eps:.0e}: 2 n_pos >= L',
                    f'2*{len(n)} = {2 * len(n)} vs L = {b.size}')
    return ok


def test_sampling_stays_sparse():
    """The augmentation must not buy determinacy with points.

    Each Matsubara point is a self-energy evaluation, so a fix that doubled the
    count would be a different regression. Measured worst 1.01 L over the sweep
    plus Lambda = 1e4 -- FEWER than the shipped set produced in the bases where
    it read the wrong part of uhat and picked up far-tail crossings (up to
    2.6 L).
    """
    ok = True
    for lam, eps in SAMPLING_SWEEP + [(10000, 1e-10), (10000, 1e-12)]:
        b = IRBasis(lam, eps=eps)
        n = b.default_matsubara_sampling(positive_only=True)
        ok &= check(len(n) <= 1.5 * b.size,
                    f'lambda={lam} eps={eps:.0e}: sampling stays O(L)',
                    f'{len(n)} points for L={b.size} ({len(n) / b.size:.2f}L)')
    return ok


def test_sampling_reads_the_carrying_part_of_uhat():
    """uhat_l's parity ALTERNATES with l, so the part to read is not fixed.

    The shipped code took `top.imag` for every fermionic basis, while its own
    docstring says uhat_{L-1} is real for even L-1. In 12 of 24 bases that read
    the truncation floor -- |Re| = 5e-12 .. 7e-6 against |Im| = 0.23 .. 0.47 --
    so the sign changes were round-off, the sampling points landed in the far
    tail, and the design matrix reached cond 2e10. The count rule above cannot
    see this: reading noise INFLATES the number of crossings, so those bases
    satisfied 2 n_pos >= L and were wrong anyway.
    """
    ok = True
    for lam, eps in SAMPLING_SWEEP:
        for stat in ('fermion', 'boson'):
            b = IRBasis(lam, eps=eps, statistics=stat)
            n_max = int(4 * max(b.size, 4) + b.lambda_ / (2 * np.pi))
            top = b.uhat(np.arange(-n_max, n_max + 1))[-1]
            part = b._sampling_part(top)
            re, im = np.abs(top.real).max(), np.abs(top.imag).max()
            ok &= check(np.abs(part).max() == max(re, im),
                        f'lambda={lam} eps={eps:.0e} {stat}: reads the part that '
                        f'carries uhat_(L-1)',
                        f'|Re|={re:.1e} |Im|={im:.1e}, read {np.abs(part).max():.1e}')
    return ok


def test_fit_refuses_an_underdetermined_set():
    """A starved set must raise, not return the minimum-norm vector.

    `default_matsubara_sampling` now guarantees its own output, so this guards
    the other door: a caller's hand-rolled set. The message must not send them
    to positive_only=False, which does NOT repair a real fit -- see the next
    check.
    """
    b = IRBasis(800.0, eps=1e-12)
    starved = b.default_matsubara_sampling(positive_only=True)[:b.size // 4]
    c = np.random.default_rng(0).standard_normal(b.size)
    try:
        b.fit_matsubara(starved, b.evaluate_matsubara(c, starved))
        return check(False, 'a starved sampling set raises',
                     f'{len(starved)} points for L={b.size} returned silently')
    except ValueError as exc:
        ok = check('UNDERDETERMINED' in str(exc),
                   'a starved sampling set raises instead of returning min-norm',
                   f'{len(starved)} points for L={b.size}')
        ok &= check('does NOT help' in str(exc),
                    'and the message says mirroring to n < 0 will not help',
                    str(exc)[-90:])
    return ok


def test_mirroring_does_not_repair_a_real_fit():
    """The negative half is redundant, so positive_only=False is not the cure.

    uhat_l(-n-1) = conj(uhat_l(n)) makes the mirrored rows duplicates up to a
    sign: measured rank 54 for both the one-sided and the symmetric set of a
    basis with L = 57, and an identical error of 0.46. That is why the fix is
    to augment with the next basis function rather than to mirror -- worth a
    test because mirroring is the obvious move and it looks like it works
    (twice the points, same rank).
    """
    ok = True
    for lam, eps in ((800, 1e-12), (1600, 1e-12), (200, 1e-6)):
        b = IRBasis(lam, eps=eps)
        n_max = int(4 * max(b.size, 4) + b.lambda_ / (2 * np.pi))
        n = np.arange(-n_max, n_max + 1)
        # the SHIPPED recipe: peaks of the top function only, no augmentation
        bare = np.array(b._peaks_per_group(n, b.uhat(n)[-1]))
        pos, sym = bare[bare >= 0], bare
        rk = [np.linalg.matrix_rank(np.vstack([(A := b.uhat(m).T).real, A.imag]))
              for m in (pos, sym)]
        ok &= check(rk[1] <= rk[0] + 1 and len(sym) >= 2 * len(pos) - 1,
                    f'lambda={lam} eps={eps:.0e}: mirroring doubles the points and '
                    f'not the rank',
                    f'{len(pos)}->{len(sym)} points, rank {rk[0]}->{rk[1]}')
    return ok


def test_symmetric_set_determines_the_complex_fit():
    """The positive_only=False path is marginal in the same way, and is used.

    A complex fit (real=False, e.g. of a periodic Wt^q, which is complex
    Hermitian at q != 0) needs n >= L rather than 2 n >= L -- and the paper's
    ~L points sit on that boundary too.
    Measured before the augmentation: worst coefficient out by 8.6e-01 at
    Lambda = 1600, eps = 1e-12. It is a separate check because a fix aimed at
    the one-sided set would not have touched it.
    """
    rng = np.random.default_rng(1)
    ok = True
    for lam, eps in ((100, 1e-10), (400, 1e-6), (400, 1e-10), (1600, 1e-12)):
        b = IRBasis(lam, eps=eps)
        c = rng.standard_normal(b.size) + 1j * rng.standard_normal(b.size)
        n = b.default_matsubara_sampling(positive_only=False)
        back = b.fit_matsubara(n, b.evaluate_matsubara(c, n), real=False)
        err = float(np.abs(back - c).max())
        ok &= check(err < 1e-11,
                    f'lambda={lam} eps={eps:.0e}: the symmetric set determines the '
                    f'complex fit',
                    f'L={b.size} npts={len(n)} err={err:.1e}')
    return ok


def test_symmetric_set_is_closed_under_reflection():
    """positive_only=False must be two-sided, which a complex fit asks for by name.

    Its complex fit of Wt^q wants "a sampling set symmetric under n -> -n-1",
    and the set only had that property by accident: the axis
    arange(-n_max, n_max+1) is symmetric under n -> -n, not under n -> -n-1, so
    for a fermion the outermost group loses a point on one side. Measured 4
    unmatched indices at +-n_max, none in the interior.
    """
    ok = True
    for lam, eps in ((200, 1e-8), (800, 1e-12), (1600, 1e-10)):
        for stat, zeta in (('fermion', 1), ('boson', 0)):
            b = IRBasis(lam, eps=eps, statistics=stat)
            n = set(int(m) for m in b.default_matsubara_sampling(positive_only=False))
            unmatched = sorted(n ^ set(-m - zeta for m in n))
            ok &= check(not unmatched,
                        f'lambda={lam} eps={eps:.0e} {stat}: symmetric set is closed '
                        f'under n -> -n-{zeta}',
                        f'{len(n)} points, unmatched {unmatched}')
    return ok


def test_continuation_order():
    orders = [ir_continuation_order(100.0, WMAX, eps=e) for e in (1e-4, 1e-8, 1e-12)]
    return check(orders == sorted(orders) and len(set(orders)) == 3,
                 'ir_continuation_order grows as the tolerance tightens',
                 f'{orders} for eps = 1e-4, 1e-8, 1e-12')


if __name__ == '__main__':
    all_ok = True
    print('\n-- 1. the T=0 grids refuse a metal')
    all_ok &= test_t0_grids_refuse_a_metal()
    print('\n-- 2. thermal_e_min restores them')
    all_ok &= test_thermal_e_min_restores_them()
    print('\n-- 3. the IR basis is a basis')
    all_ok &= test_basis_is_a_basis()
    print('\n-- 4. size converges and scales as log(Lambda)')
    all_ok &= test_size_converges_and_scales()
    print('\n-- 5. uhat parity (Li et al. Sec. II C)')
    all_ok &= test_uhat_parity()
    print('\n-- 6. fit and predict, gapped vs metallic')
    all_ok &= test_fit_and_predict()
    print('\n-- 7. sparse sampling and conditioning')
    all_ok &= test_tau_transport_jacobian()
    all_ok &= test_sampling_is_sparse_and_conditioned()
    print('\n-- 7b. the sampling set determines the fit')
    all_ok &= test_sampling_determines_the_fit()
    all_ok &= test_sampling_stays_sparse()
    all_ok &= test_sampling_reads_the_carrying_part_of_uhat()
    all_ok &= test_fit_refuses_an_underdetermined_set()
    all_ok &= test_mirroring_does_not_repair_a_real_fit()
    all_ok &= test_symmetric_set_determines_the_complex_fit()
    all_ok &= test_symmetric_set_is_closed_under_reflection()
    print('\n-- 8. continuation order')
    all_ok &= test_continuation_order()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
