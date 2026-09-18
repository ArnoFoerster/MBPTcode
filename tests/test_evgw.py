"""evGW: the spectrum reinjected into G and P0 until it stops moving.

The two properties worth gating are an IDENTITY and a DIVERGENCE.

The identity: the first cycle screens with the mean field's own eigenvalues, so
it IS G0W0. If it ever stops being, the loop is not starting where it claims to.

The divergence: the loop converges only because the quasiparticle equation stays
anchored on the mean field while the screening follows the iterate. Anchor it on
the iterate instead and each cycle adds its own correction a second time. That
version still runs, still prints plausible numbers and never converges, so the
test asserts it FAILS -- otherwise nothing here would notice the anchor being
dropped.

Run: python tests/test_evgw.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import dft, gto

from src.Base.constants import HARTREE_TO_EV
from src.SingleReference.GW.evGW import (MODES, evgw_eigenvalues,
                                         shifted_mean_field)
from src.SingleReference.GW.imaginary_time import (DEFAULT_TAU_TARGET,
                                                   minimax_points_for_gw)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.space_time import solve_qp_energy_space_time
import src.SingleReference.GW.space_time as gwst
import src.SingleReference.LinearResponse.davidson as dv


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build_reference():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    m = dft.RKS(mol)
    m.xc = 'pbe0'
    m.conv_tol = 1e-12
    m.kernel()
    return m


def test_the_first_cycle_is_g0w0(mf):
    """Cycle one screens with the mean field's own spectrum, so it must
    reproduce the single-shot route: the same grid, the same anchor, the same
    call, to the round-off of the one unit conversion the dispatcher makes."""
    eps0 = np.asarray(mf.mo_energy, float)
    nocc = mf.mol.nelectron // 2
    states = np.arange(len(eps0))

    _, info = evgw_eigenvalues(mf, mf.mol, max_cycle=1, tol=0.0)
    one_cycle = info['eps_mean_field'] + info['shift']
    g0w0 = np.asarray(solve_qp_energy_space_time(mf, mf.mol, nocc, states), float)
    d = np.abs(one_cycle[states] - g0w0).max()
    return check(d < 1e-12 * max(np.abs(g0w0).max(), 1.0),
                 'the first evGW cycle IS G0W0', f'max |d eps| {d:.1e} Ha')


def test_it_converges_and_opens_the_gap_beyond_g0w0(mf):
    """evGW screens with the OPENED gap, so P0 is less polarizable, W less
    screening, and the gap ends above the G0W0 one. A loop that came back below
    it would have the feedback backwards."""
    nocc = mf.mol.nelectron // 2
    ks_gap = (mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * HARTREE_TO_EV
    eps, info = evgw_eigenvalues(mf, mf.mol, max_cycle=20)
    ev_gap = (eps[nocc] - eps[nocc - 1]) * HARTREE_TO_EV
    first = info['eps_mean_field'] + np.asarray(
        evgw_eigenvalues(mf, mf.mol, max_cycle=1, tol=0.0)[1]['shift'])
    g0w0_gap = (first[nocc] - first[nocc - 1]) * HARTREE_TO_EV

    ok = check(info['converged'], 'the loop converges',
               f"{info['cycles']} cycles, last residual "
               f"{info['history'][-1] * HARTREE_TO_EV:.2e} eV")
    ok &= check(ev_gap > g0w0_gap > ks_gap,
                'KS < G0W0 < evGW on the gap',
                f'{ks_gap:.3f} < {g0w0_gap:.3f} < {ev_gap:.3f} eV')
    ok &= check(info['history'][-1] < info['history'][0],
                'the residual falls')
    return ok


def test_the_high_virtuals_do_not_decide_convergence(mf):
    """They are discretised continuum, not quasiparticles: the solver picks a
    different root for them from one cycle to the next, so a criterion over the
    whole spectrum never converges even though the gap is stationary. The
    residual they carry is reported rather than hidden."""
    nocc = mf.mol.nelectron // 2
    _, info = evgw_eigenvalues(mf, mf.mol, max_cycle=20)
    ok = check(info['converged'] and set(info['converge_on']) == {nocc - 1, nocc},
               'convergence is decided on the HOMO and LUMO alone',
               f"converge_on={sorted(info['converge_on'])}")
    ok &= check(info['residual_untested'] > info['history'][-1],
                'and the rest of the spectrum reports its own, larger residual',
                f"{info['residual_untested'] * HARTREE_TO_EV:.3e} eV vs "
                f"{info['history'][-1] * HARTREE_TO_EV:.3e}")
    all_states = np.arange(len(np.asarray(mf.mo_energy)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, wide = evgw_eigenvalues(mf, mf.mol, converge_on=all_states,
                                   max_cycle=8)
    ok &= check(not wide['converged']
                and any('did not converge' in str(w.message) for w in caught),
                'converging on every orbital does not, and says so')
    return ok


def test_the_anchor_is_what_makes_it_converge(mf):
    """THE DISCRIMINATING TEST. Drive the same loop by hand with the equation
    anchored on the ITERATE and it must diverge, opening the gap by a
    near-constant amount every cycle. Without this, dropping `eps_anchor`
    would leave every other test here passing."""
    eps0 = np.asarray(mf.mo_energy, float)
    nocc = mf.mol.nelectron // 2
    states = np.arange(len(eps0))
    # the grid the loop freezes: sized from the mean field
    ntau = minimax_points_for_gw(eps0, nocc, mu=0.5 * (eps0[nocc - 1] + eps0[nocc]),
                                 target=DEFAULT_TAU_TARGET)[0]

    eps = eps0.copy()
    gaps = []
    for _ in range(4):
        # no eps_anchor: the route uses the iterate for BOTH the screening and
        # the eps_p of w = eps_p + ... , which is the mistake
        w = np.asarray(solve_qp_energy_space_time(
            shifted_mean_field(mf, eps), mf.mol, nocc, states, ntau=ntau), float)
        eps = eps.copy()
        eps[states] = w
        gaps.append(eps[nocc] - eps[nocc - 1])

    steps = np.diff(gaps)
    _, good = evgw_eigenvalues(mf, mf.mol, max_cycle=20)
    ok = check((steps > 0).all() and steps.min() * HARTREE_TO_EV > 1.0,
               'anchored on the ITERATE the gap runs away, > 1 eV per cycle',
               ' '.join(f'{s * HARTREE_TO_EV:+.2f}' for s in steps) + ' eV')
    ok &= check(good['converged'] and good['history'][-1] < good['history'][1] / 5.0,
                'anchored on the MEAN FIELD it converges')
    return ok


def test_the_quadrature_is_frozen_on_the_anchor(mf):
    """A grid re-derived per cycle shrinks as the gap opens, so the fixed point
    would depend on the schedule that reached it. The space-time route sizes
    its grid from the anchor when one is given: on a spectrum whose gap is
    opened by 30 eV it keeps the mean field's count, and without the anchor it
    wants fewer points for the narrower ratio e_max/e_min."""
    eps0 = np.asarray(mf.mo_energy, float)
    nocc = mf.mol.nelectron // 2
    states = np.arange(len(eps0))
    opened = eps0.copy()
    opened[nocc:] += 30.0 / HARTREE_TO_EV
    t_mf, t_free, t_anchored = {}, {}, {}
    solve_qp_energy_space_time(mf, mf.mol, nocc, states, timings=t_mf)
    solve_qp_energy_space_time(shifted_mean_field(mf, opened), mf.mol, nocc,
                               states, timings=t_free)
    solve_qp_energy_space_time(shifted_mean_field(mf, opened), mf.mol, nocc,
                               states, eps_anchor=eps0, timings=t_anchored)
    return check(t_anchored['ntau_auto'] == t_mf['ntau_auto']
                 and t_free['ntau_auto'] < t_mf['ntau_auto'],
                 'the tau grid follows the anchor, not the iterate',
                 f"mean field {t_mf['ntau_auto']}, anchored "
                 f"{t_anchored['ntau_auto']}, free {t_free['ntau_auto']}")


def test_every_orbital_is_updated(mf):
    """G and P0 are built from the whole spectrum, so a partial update leaves
    the rest screening at mean-field values and the result is not evGW."""
    _, info = evgw_eigenvalues(mf, mf.mol, max_cycle=2, tol=0.0)
    n = len(np.asarray(mf.mo_energy))
    return check(np.array_equal(info['states'], np.arange(n))
                 and (np.abs(info['shift']) > 0).all(),
                 'every orbital moves, so every orbital screens',
                 f"smallest |shift| {np.abs(info['shift']).min() * HARTREE_TO_EV:.3f} eV")


def test_diis_reaches_the_same_fixed_point_in_fewer_cycles(mf):
    """A fixed point reached faster must be the SAME fixed point; an
    accelerator that moved the answer would be extrapolating a different map.
    The comparison is against the damped linear mixing DIIS replaces, since on
    a two-electron system the undamped iteration already converges in five."""
    plain_eps, plain = evgw_eigenvalues(mf, mf.mol, max_cycle=40, diis_size=0,
                                        damping=0.5)
    diis_eps, diis = evgw_eigenvalues(mf, mf.mol, max_cycle=40)
    nocc = mf.mol.nelectron // 2
    worst = max(abs(plain_eps[p] - diis_eps[p]) * HARTREE_TO_EV
                for p in (nocc - 1, nocc))
    ok = check(plain['converged'] and diis['converged']
               and diis['cycles'] < plain['cycles'],
               'DIIS converges in fewer cycles than damped mixing',
               f"{diis['cycles']} vs {plain['cycles']}")
    ok &= check(worst < 1e-3, 'to the same fixed point', f'{worst:.1e} eV')
    return ok


def test_the_shift_view_does_not_move_the_original(mf):
    """The loop builds one of these per cycle; mutating the caller's mean field
    would leave it holding a spectrum from an abandoned iterate."""
    before = np.asarray(mf.mo_energy, float).copy()
    view = shifted_mean_field(mf, before + 0.1)
    return check(np.array_equal(np.asarray(mf.mo_energy, float), before)
                 and np.allclose(np.asarray(view.mo_energy, float), before + 0.1)
                 and view.mo_coeff is mf.mo_coeff and view.mol is mf.mol,
                 'the shifted view shares everything but the spectrum')


def test_the_shift_view_builds_j_and_k_on_the_direct_path(mf):
    """A molecule too large for in-core J/K takes PySCF's direct branch, which
    reads the optimizer the mean field caches; a view made through the pickle
    hooks arrives without it and the static correction of every cycle raises.
    `max_memory = 0` forces that branch on the small reference."""
    view = shifted_mean_field(mf, np.asarray(mf.mo_energy, float))
    view.max_memory = 0
    dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    try:
        v_direct = view.get_veff(mf.mol, dm)
        ok = np.allclose(v_direct, mf.get_veff(mf.mol, dm), atol=1e-10)
        detail = 'direct and in-core V_Hxc agree'
    except AttributeError as err:
        ok, detail = False, f'direct get_veff raised: {err}'
    return check(ok, 'the shifted view builds V_Hxc on the direct J/K path',
                 detail)


def test_g0w0_is_unchanged_when_the_anchor_is_not_given(mf):
    """`eps_anchor=None` must leave the single-shot route BITWISE as it was, or
    threading the keyword moved every existing G0W0 number."""
    nocc = mf.mol.nelectron // 2
    states = np.arange(len(np.asarray(mf.mo_energy)))
    plain = np.asarray(solve_qp_energy_space_time(mf, mf.mol, nocc, states,
                                                  ntau=14), float)
    given = np.asarray(solve_qp_energy_space_time(
        mf, mf.mol, nocc, states, ntau=14,
        eps_anchor=np.asarray(mf.mo_energy, float)), float)
    return check(np.array_equal(plain, given),
                 'eps_anchor=None leaves G0W0 bitwise unchanged')


def test_the_anchor_reaches_every_route_through_the_dispatcher(mf):
    """The loop hands `eps_anchor` to `calc_qp_energy`, so on a spectrum that
    is not the mean field's the anchored root must differ from the unanchored
    one by about the shift -- on every route, or that route would compound."""
    eps0 = np.asarray(mf.mo_energy, float)
    nocc = mf.mol.nelectron // 2
    opened = eps0.copy()
    opened[:nocc] -= 0.5 / HARTREE_TO_EV
    opened[nocc:] += 0.5 / HARTREE_TO_EV
    view = shifted_mean_field(mf, opened)
    ok = True
    for mode in MODES:
        free = calc_qp_energy(view, mode=mode, state=nocc - 1)
        anchored = calc_qp_energy(view, mode=mode, state=nocc - 1, eps_anchor=eps0)
        ok &= check(0.2 < anchored - free < 0.8,
                    f'{mode}: the anchor moves the root by about the shift',
                    f'{anchored - free:+.3f} eV of the 0.5 eV applied')
    return ok


def test_calc_qp_energy_drives_every_route(mf):
    """`self_consistency='evGW'` must work on all three GW routes, and all
    three must reach the same fixed point: they differ only in how chi0 is
    built, so a disagreement beyond the quadratures would be a bug in one."""
    ok = True
    got = []
    for mode in MODES:
        g0w0 = calc_qp_energy(mf, mode=mode)
        evgw = calc_qp_energy(mf, mode=mode, self_consistency='evGW')
        got.append(evgw)
        ok &= check(-0.5 < evgw - g0w0 < -0.1,
                    f'{mode}: evGW pushes the occupied level down',
                    f'G0W0 {g0w0:.4f} -> evGW {evgw:.4f} eV')
    ok &= check(max(got) - min(got) < 0.01,
                'three screening constructions, one fixed point',
                f'spread {(max(got) - min(got)) * 1e3:.1f} meV')
    return ok


def test_the_refusals(mf):
    """A vertex or a non-RPA screening would need its own fixed point rather
    than riding on this one; an unknown mode or switch is a typo, not a
    fallback."""
    ok = True
    for kw in ({'selfenergy': 'PSD1'}, {'polarizability': 'BSE'}):
        try:
            calc_qp_energy(mf, self_consistency='evGW', **kw)
            ok &= check(False, f'{kw} is refused')
        except NotImplementedError as exc:
            ok &= check('GW@RPA only' in str(exc), f'{kw} is refused')
    try:
        evgw_eigenvalues(mf, mf.mol, mode='nonsense')
        ok &= check(False, 'an unknown mode is refused')
    except ValueError as exc:
        ok &= check('choose one of' in str(exc), 'an unknown mode is refused')
    try:
        dv.solve_bse_isdf(mf, mf.mol, mf.mol.nelectron // 2, nroots=1,
                          probe=False, self_consistency='qsGW')
        ok &= check(False, "an unknown self_consistency is refused")
    except ValueError as exc:
        ok &= check("choose 'G0W0' or 'evGW'" in str(exc),
                    'an unknown self_consistency is refused')
    try:
        dv.solve_bse_isdf(mf, mf.mol, mf.mol.nelectron // 2, nroots=1,
                          probe=False, qp=np.asarray(mf.mo_energy, float),
                          screen_at='whatever')
        ok &= check(False, 'an unknown screen_at is refused')
    except ValueError as exc:
        ok &= check("choose 'qp' or 'mean-field'" in str(exc),
                    'an unknown screen_at is refused')
    return ok


def test_w_is_rebuilt_from_the_updated_spectrum_every_cycle(mf):
    """The point of the loop. chi0 must be evaluated once per cycle on the
    CURRENT eigenvalues, and W is its Dyson inverse -- a screening cached on
    the geometry instead of the spectrum would leave every cycle re-solving
    against the mean field's W and converge to G0W0 with extra steps.

    The norm falling is the physics: the gap opens, chi0 is less polarizable,
    W is less screened.
    """
    seen = []
    original = gwst.chi0_imaginary_frequency

    def counted(X, D, eps, nocc_, grid, **kw):
        out = original(X, D, eps, nocc_, grid, **kw)
        chi0 = out[0] if isinstance(out, tuple) else out
        seen.append((float(eps[nocc_] - eps[nocc_ - 1]),
                     float(np.linalg.norm(chi0))))
        return out

    gwst.chi0_imaginary_frequency = counted
    try:
        evgw_eigenvalues(mf, mf.mol, max_cycle=4, tol=0.0)
    finally:
        gwst.chi0_imaginary_frequency = original

    gaps = [g for g, _ in seen]
    norms = [n for _, n in seen]
    ok = check(len(seen) == 4 and len(set(gaps)) == 4 and len(set(norms)) == 4,
               'chi0 is rebuilt on a different spectrum every cycle',
               f'{len(seen)} builds, {len(set(gaps))} distinct gaps')
    ok &= check(gaps[1] > gaps[0] and norms[1] < norms[0],
                'a wider gap gives a smaller chi0',
                f'gap {gaps[0]:.4f} -> {gaps[1]:.4f} Ha, '
                f'|chi0| {norms[0]:.4f} -> {norms[1]:.4f}')
    return ok


def test_screen_at_selects_which_spectrum_builds_w(mf):
    """The convention is the LEVEL OF THEORY: G0W0 screens W0 at the mean
    field, evGW at its fixed point. An explicit `qp` array therefore defaults
    to the standard G0W0 split, and `screen_at='qp'` is the opt-in for an array
    that IS a converged spectrum. The difference is tens of meV between the DF
    and full-integral routes, so the knob has to reach the W build."""
    eps_qp, _ = evgw_eigenvalues(mf, mf.mol, max_cycle=3, tol=0.0)
    nocc = mf.mol.nelectron // 2
    seen = {}
    original = dv.isdf_bse_factors

    def spy(mf_in, mol, nocc_, **kw):
        e = np.asarray(mf_in.mo_energy, float)
        seen[len(seen)] = float(e[nocc_] - e[nocc_ - 1])
        return original(mf_in, mol, nocc_, **kw)

    mfe = np.asarray(mf.mo_energy, float)
    wanted = {'qp': eps_qp[nocc] - eps_qp[nocc - 1],
              'mean-field': mfe[nocc] - mfe[nocc - 1]}
    dv.isdf_bse_factors = spy
    ok = True
    try:
        for mode in ('qp', 'mean-field', None):
            seen.clear()
            kw = {} if mode is None else {'screen_at': mode}
            dv.solve_bse_isdf(mf, mf.mol, nocc, nroots=1, probe=False,
                              qp=eps_qp, **kw)
            gap = list(seen.values())[0]
            ok &= check(abs(gap - wanted[mode or 'mean-field']) < 1e-12,
                        f"screen_at={mode!r} builds W at that gap"
                        + (' (the default)' if mode is None else ''),
                        f'{gap:.6f} Ha')
    finally:
        dv.isdf_bse_factors = original
    return ok


def test_an_unrestricted_reference_is_driven_channel_by_channel():
    """Both spin channels are updated and converged together on the Casida
    route, and the dispatcher's evGW answer for one channel is that channel's
    fixed point. The alpha HOMO of a radical is its ionization, so GW pushes
    it down from the Kohn-Sham value."""
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.conv_tol = 1e-11
    mf.kernel()

    eps, info = evgw_eigenvalues(mf, mol, mode='casida')
    na, nb = mf.nelec
    ok = check(info['converged'] and eps.shape == (2, mf.mo_coeff[0].shape[1]),
               'both channels converge together',
               f"{info['cycles']} cycles, spectrum {eps.shape}")
    ok &= check(eps[0, na - 1] < mf.mo_energy[0][na - 1] - 1.0 / HARTREE_TO_EV
                and eps[1, nb - 1] < mf.mo_energy[1][nb - 1] - 1.0 / HARTREE_TO_EV,
                'GW pushes both spin HOMOs down from Kohn-Sham',
                f'alpha {(eps[0, na - 1] - mf.mo_energy[0][na - 1]) * HARTREE_TO_EV:+.2f}, '
                f'beta {(eps[1, nb - 1] - mf.mo_energy[1][nb - 1]) * HARTREE_TO_EV:+.2f} eV')
    for label, ch, n, row in (('alpha', 'alpha', na, 0), ('beta', 'beta', nb, 1)):
        got = calc_qp_energy(mf, mode='casida', self_consistency='evGW',
                             spin_channel=ch)
        ok &= check(abs(got - eps[row, n - 1] * HARTREE_TO_EV) < 1e-8,
                    f'the dispatcher returns the {label} fixed point')
    try:
        evgw_eigenvalues(mf, mol, mode='space-time')
        ok &= check(False, 'the imaginary-axis routes refuse it themselves')
    except NotImplementedError as exc:
        ok &= check('restricted-spin only' in str(exc),
                    'the imaginary-axis routes refuse it themselves')
    return ok


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    mf = build_reference()
    all_ok = True
    print('\n-- 1. the loop starts at G0W0 and ends past it')
    all_ok &= test_the_first_cycle_is_g0w0(mf)
    all_ok &= test_it_converges_and_opens_the_gap_beyond_g0w0(mf)
    all_ok &= test_every_orbital_is_updated(mf)
    print('\n-- 2. the anchor, which is what makes it converge')
    all_ok &= test_the_anchor_is_what_makes_it_converge(mf)
    all_ok &= test_the_quadrature_is_frozen_on_the_anchor(mf)
    all_ok &= test_g0w0_is_unchanged_when_the_anchor_is_not_given(mf)
    all_ok &= test_the_anchor_reaches_every_route_through_the_dispatcher(mf)
    print('\n-- 3. the screening really follows the iterate')
    all_ok &= test_w_is_rebuilt_from_the_updated_spectrum_every_cycle(mf)
    all_ok &= test_screen_at_selects_which_spectrum_builds_w(mf)
    print('\n-- 4. convergence machinery')
    all_ok &= test_the_high_virtuals_do_not_decide_convergence(mf)
    all_ok &= test_diis_reaches_the_same_fixed_point_in_fewer_cycles(mf)
    all_ok &= test_the_shift_view_does_not_move_the_original(mf)
    all_ok &= test_the_shift_view_builds_j_and_k_on_the_direct_path(mf)
    print('\n-- 5. the front door, on every route')
    all_ok &= test_calc_qp_energy_drives_every_route(mf)
    all_ok &= test_the_refusals(mf)
    print('\n-- 6. an unrestricted reference')
    all_ok &= test_an_unrestricted_reference_is_driven_channel_by_channel()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
