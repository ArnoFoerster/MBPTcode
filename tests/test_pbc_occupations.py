"""Occupation-weighted response with (f_m - f_n) and smearing.

src/SingleReference/Periodic/pbc_occupations.py. Everything in Periodic/ built
its transition space from an INTEGER occupied count, which for a metal is not a
loss of accuracy but of the response itself: on the Li monolayer below, eight of
nine k-points round to a completely filled band, so the whole low-energy
response vanishes.

A metal forces one decision -- Hermitian with a regularization, or
non-Hermitian Casida -- because with fractional occupations sqrt(f_i - f_a)
approaches zero and can change sign within the partially filled band. Check 1 settles it by measurement rather than argument: the sign
change belongs to the FIXED occ/virt partition, and ordering each transition so
f_m > f_n removes it. What then decides the formulation is whether e_n > e_m on
that same set, since the squaring trick needs A - B positive definite, and that
is exactly monotonicity of f in e -- which Fermi-Dirac at one chemical
potential gives identically.

Checks:
  1. THE DECISION, on a real metal: f monotone in e with zero violations, and
     zero ordered pairs with e_n <= e_m across all k-pairs; A - B accordingly
     positive on the retained space and the Casida solve well posed. Plus the
     guard raising on occupations that are not monotone.
  2. EXACT REDUCTION to the integer-occupation path -- chi0 at several
     frequencies and transfers, the RPA A/B blocks, and the diagonal
     self-energy -- to machine precision. This is what makes the new path safe
     to prefer: it is the same code path for a gapped system.
  3. What the integer collapse costs on the metal, and that the new path does
     not consult `dfints.nocc` at all.
  4. Internal consistency on the metal: the spectral W from the
     occupation-weighted eigenpairs equals the direct (1 - Pi0)^{-1}, which is
     what fixes the 2/nk constant that sqrt(f_m - f_n) in the amplitudes
     changes.
  5. The threshold regularization: results are unchanged below 1e-4 while the
     pair count falls, so it is a sparsity mechanism as much as a safeguard.
  6. The smearing limit: as sigma -> 0 on a GAPPED system the occupation
     weighting returns the integer answer.
  7. CONTINUITY of the correlation energy as a metal is deformed, which is the
     property every other check here misses. All of the above are pointwise --
     a reduction, a consistency relation, a limit at fixed geometry. The
     integer collapse fails in a way none of them can see: nocc changes in
     STEPS as the cell is strained, so E_c jumps. That was found by an
     equation-of-state scan of BCC Li and not by any unit test, so it gets one
     here.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.scf import addons as pbc_addons

from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.Periodic.pbc_rpa_damping import (nyquist_params,
                                                          make_coulG_damped)
from src.SingleReference.Periodic.pbc_damped_integrals import build_dfintegrals_coulG
from src.SingleReference.Periodic import pbc_casida as pc
from src.SingleReference.Periodic import pbc_self_energy as pse
from src.SingleReference.Periodic.pbc_rpa import ri_rpa_ecorr_from_dfints
from src.SingleReference.Periodic.pbc_isdf_rpa import chemical_potential_occ
from src.SingleReference.Periodic.pbc_occupations import (
    DEFAULT_OCCUPATION_THRESHOLD, fermi_level, fermi_level_from_mf,
    is_integer_occupation,
    check_occupation_monotonic, transition_pairs, transition_layout,
    build_chi0_aux_occ, build_rpa_matrices_occ, solve_rpa_spectral_occ,
    build_W_aux_spectral_occ, solve_rpa_all_q_occ, sigma_c_diag_occ,
    qp_energy_g0w0_occ, qp_energy_root_occ)
from src.Base.utils.matsubara import beta_from_mf
from pyscf.pbc import scf as pbcscf


def check(ok, label, detail=''):
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return bool(ok)


def h2_slab(kmesh=(2, 2, 1), mesh=(13, 13, 72)):
    """The gapped slab used throughout Periodic/ -- integer occupations."""
    cell = pgto.Cell()
    cell.atom = 'H 0 0 -0.37; H 0 0 0.37'
    cell.a = np.diag([4.0, 4.0, 24.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = list(mesh)
    cell.verbose = 0
    cell.build()
    mf = pscf.KRHF(cell, cell.make_kpts(list(kmesh)), exxdiv=None).density_fit()
    mf.kernel()
    return cell, mf


def li_layer(kmesh=(3, 3, 1), sigma=0.02, mesh=(15, 15, 60)):
    """A metallic Li monolayer: two-dimensional, fractionally occupied."""
    cell = pgto.Cell()
    cell.atom = 'Li 0 0 0; Li 1.75 1.75 0'
    cell.a = np.diag([3.5, 3.5, 16.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = list(mesh)
    cell.verbose = 0
    cell.build()
    mf = pscf.KRHF(cell, cell.make_kpts(list(kmesh)), exxdiv=None).density_fit()
    mf = pbc_addons.smearing_(mf, sigma=sigma, method='fermi')
    mf.conv_tol = 1e-9
    mf.kernel()
    return cell, mf


def integrals(cell, mf, kmesh):
    r0, beta, _ = nyquist_params(cell, list(kmesh))
    return build_dfintegrals_coulG(mf, coulG_fn=make_coulG_damped(r0, beta))


# --- fixtures -------------------------------------------------------------
#
# The checks below are written to run BOTH ways: `pytest` collects them, and
# `python tests/test_pbc_occupations.py` runs them in order with a summary.
# The fixtures exist so the first of those works at all -- without them pytest
# reads `cell`, `mf` and `dfints_*` as missing fixtures and ERRORS on every
# test in the file, which is a silent hole in an unattended run rather than a
# failure anyone would notice.
#
# Module-scoped because each one is an SCF: the metal is built once and shared.


@pytest.fixture(scope='module')
def _metal():
    return li_layer()


@pytest.fixture(scope='module')
def cell(_metal):
    return _metal[0]


@pytest.fixture(scope='module')
def mf(_metal):
    return _metal[1]


@pytest.fixture(scope='module')
def dfints_metal(_metal):
    return integrals(_metal[0], _metal[1], (3, 3, 1))


@pytest.fixture(scope='module')
def dfints_gapped():
    cell_h, mf_h = h2_slab()
    return integrals(cell_h, mf_h, (2, 2, 1))


# --------------------------------------------------- 0. the chemical potential

def test_fermi_level_reproduces_pyscfs_own_occupations(mf):
    """The root-find must recover the mu that GENERATED mo_occ.

    Self-consistency rather than a reference value: Fermi-Dirac at the returned
    mu has to reproduce pyscf's occupations, which pins the inversion without
    needing an external number. Measured to 5e-15.
    """
    e = np.asarray(mf.mo_energy)
    f = np.asarray(mf.mo_occ, dtype=float)
    mu = fermi_level_from_mf(mf)
    assert mu is not None
    fd = 2.0 / (1.0 + np.exp((e - mu) / mf.sigma))
    assert np.abs(fd - f).max() < 1e-12, np.abs(fd - f).max()


def test_get_fermi_is_not_the_smearing_chemical_potential(mf):
    """`mf.get_fermi()` is NOT this quantity, and reaching for it is the
    natural mistake -- measured 0.1 to 1.5 eV away on this system. Pinned so
    that anyone tempted to 'simplify' fermi_level into a get_fermi call sees
    the number first."""
    mu = fermi_level_from_mf(mf)
    assert abs(float(mf.get_fermi()) - mu) > 1e-3, (mf.get_fermi(), mu)


def test_the_heuristic_degrades_where_it_matters():
    """chemical_potential_occ is good with NO partial occupations and bad with
    them -- i.e. worst in the regime it exists for.

    Synthetic so the two regimes can be exhibited side by side without two
    SCFs: build occupations from a known mu, then ask each estimator to
    recover it. With a gap the heuristic falls back to midgap, which is where
    a Fermi-Dirac mu does sit; once bands cross the level, the mean of the
    partially occupied eigenvalues is simply a different quantity.
    """
    rng = np.random.default_rng(0)
    sigma, mu_true = 0.02, 0.15

    # (a) a clear gap around mu: nothing partially occupied
    e_gap = np.array([[-0.4, -0.3, 0.7, 0.8]])
    f_gap = 2.0 / (1.0 + np.exp((e_gap - mu_true) / sigma))
    nelec = f_gap.sum()
    assert not ((f_gap > 0.5) & (f_gap < 1.5)).any()
    mu_root = fermi_level(e_gap, sigma, nelec, nkpts=1)
    assert abs(mu_root - mu_true) < 1e-6
    assert abs(chemical_potential_occ(e_gap, f_gap) - mu_true) < 0.05

    # (b) bands straddling mu: partial occupations, and the heuristic drifts
    e_met = mu_true + np.sort(rng.normal(scale=0.05, size=(4, 6)), axis=1)
    f_met = 2.0 / (1.0 + np.exp((e_met - mu_true) / sigma))
    mu_root = fermi_level(e_met, sigma, f_met.sum() / 4, nkpts=4)
    assert abs(mu_root - mu_true) < 1e-6, mu_root
    assert ((f_met > 0.5) & (f_met < 1.5)).any()
    heur_err = abs(chemical_potential_occ(e_met, f_met) - mu_true)
    assert heur_err > 10 * abs(mu_root - mu_true), heur_err


def test_fermi_level_refuses_a_gaussian_width():
    """A broadening parameter is not a temperature -- same refusal as
    matsubara.beta_from_mf, for the same reason."""
    with pytest.raises(NotImplementedError, match='Fermi-Dirac'):
        fermi_level(np.zeros((1, 4)), 0.01, 2.0, nkpts=1, method='gauss')


# --------------------------------------------------------------- 1. decision

def test_the_decision(cell, mf):
    e = np.asarray(mf.mo_energy)
    f = np.asarray(mf.mo_occ, dtype=float)
    nk, nmo = e.shape

    ok = check(not is_integer_occupation(f),
               'the test metal really has fractional occupations',
               f'range [{f.min():.4f}, {f.max():.4f}], '
               f'{int(((f > 1e-8) & (f < 2 - 1e-8)).sum())} of {f.size} fractional')

    bad = check_occupation_monotonic(e, f, raise_on_fail=False)
    ok &= check(not bad, 'f is a monotone function of e -- zero violations',
                f'{len(bad)} rising steps out of {f.size - 1}')

    # The condition the Hermitian formulation needs, over ALL k-pairs, which is
    # what a finite momentum transfer couples.
    total = worst = 0
    for k1 in range(nk):
        for k2 in range(nk):
            w = f[k1][:, None] - f[k2][None, :]
            gap = e[k2][None, :] - e[k1][:, None]
            keep = w > 1e-12
            total += int(keep.sum())
            worst += int(((gap <= 0) & keep).sum())
    ok &= check(worst == 0 and total > 0,
                'every ordered pair with f_m > f_n also has e_n > e_m, so '
                'A - B stays positive definite', f'{worst} bad of {total} pairs')

    # And the guard fires when that is not true.
    broken = f.copy()
    lo = np.unravel_index(np.argmin(e), e.shape)
    hi = np.unravel_index(np.argmax(e), e.shape)
    broken[lo], broken[hi] = 0.0, 2.0
    try:
        check_occupation_monotonic(e, broken)
    except ValueError as err:
        ok &= check('not a monotone function' in str(err),
                    'the guard refuses non-monotone occupations, naming the '
                    'non-Hermitian alternative')
    else:
        ok &= check(False, 'the guard refuses non-monotone occupations')
    assert ok


def test_casida_is_well_posed(dfints_metal):
    """A - B positive and the solve returning real positive frequencies."""
    ok = True
    for q in (0, 1):
        A, B = build_rpa_matrices_occ(dfints_metal, dfints_metal.kconserv_pair[q])
        AmB = A - B
        offdiag = np.sqrt(max(np.linalg.norm(AmB) ** 2
                              - np.linalg.norm(np.diag(AmB)) ** 2, 0.0))
        d = np.diag(AmB).real
        ok &= check(offdiag < 1e-12 * max(np.linalg.norm(AmB), 1.0),
                    f'q={q}: A - B is diagonal', f'offdiag {offdiag:.1e}')
        ok &= check(d.min() > 0, f'q={q}: and positive', f'min {d.min():.4e}')
        Omega, X, Y = CasidaSolver(A, B).solve()
        ok &= check(np.all(np.isreal(Omega)) and Omega.min() > 0,
                    f'q={q}: the Casida solve gives real positive frequencies',
                    f'{len(Omega)} states, min {Omega.min():.4e}')
    assert ok


# ------------------------------------------------------- 2. exact reduction

def test_reduction_to_integer_path(dfints_gapped):
    ok = check(is_integer_occupation(dfints_gapped.mo_occ),
               'the gapped slab has integer occupations')
    worst_chi = worst_A = worst_B = 0.0
    for q in range(dfints_gapped.nkpts):
        kc = dfints_gapped.kconserv_pair[q]
        for omega in (0.0, 0.3, 1.7):
            a = pc.build_chi0_aux(dfints_gapped, kc, omega)
            b = build_chi0_aux_occ(dfints_gapped, kc, omega)
            worst_chi = max(worst_chi,
                            np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))
        A0, B0 = pc.build_rpa_matrices(dfints_gapped, kc)
        A1, B1 = build_rpa_matrices_occ(dfints_gapped, kc)
        ok &= check(A0.shape == A1.shape,
                    f'q={q}: the transition spaces have the same size',
                    f'{A0.shape[0]}')
        worst_A = max(worst_A, np.abs(A0 - A1).max())
        worst_B = max(worst_B, np.abs(B0 - B1).max())
    ok &= check(worst_chi < 1e-13,
                'chi0 reduces to build_chi0_aux at every q and frequency',
                f'{worst_chi:.1e}')
    ok &= check(worst_A < 1e-13 and worst_B < 1e-13,
                'and the Casida blocks reduce to build_rpa_matrices',
                f'A {worst_A:.1e}, B {worst_B:.1e}')

    nocc = dfints_gapped.nocc[0]
    eps = np.asarray(dfints_gapped.mo_energy)[0][nocc - 1].real
    freqs = [eps, eps + 0.2]
    old = pse.sigma_c_diag(dfints_gapped, pse.solve_rpa_all_q(dfints_gapped), 0, nocc - 1, freqs)
    new = sigma_c_diag_occ(dfints_gapped, solve_rpa_all_q_occ(dfints_gapped), 0, nocc - 1, freqs)
    ok &= check(np.abs(old - new).max() < 1e-13,
                'and the diagonal self-energy reduces to sigma_c_diag',
                f'{np.abs(old - new).max():.1e}, Sigma_c = {old[0]:.8f}')

    # The QP energy on top of it. This is the reduction that matters for the
    # wrapper's existence: pointing a metal at the integer entry point returns
    # a NUMBER, because sigma_c_diag labels the internal orbital by a binary
    # occupied/virtual flag rather than failing on fractional occupations.
    qp_old = pse.qp_energy_g0w0(dfints_gapped, pse.solve_rpa_all_q(dfints_gapped), 0, nocc - 1)
    qp_new, Z = qp_energy_g0w0_occ(dfints_gapped, solve_rpa_all_q_occ(dfints_gapped),
                                   0, nocc - 1)
    ok &= check(abs(qp_old - qp_new) < 1e-12,
                'and the linearized QP energy reduces to qp_energy_g0w0',
                f'{qp_old:.8f} vs {qp_new:.8f} Ha, Z = {Z:.4f}')

    # The root solve, checked HERE because this is where it can be checked:
    # Z = 0.99 on this gapped slab, so the linearization is trustworthy and the
    # two must agree. That agreement is what licenses the root solve on a metal,
    # where Z goes negative and there is nothing to compare against.
    root = qp_energy_root_occ(dfints_gapped, solve_rpa_all_q_occ(dfints_gapped), 0, nocc - 1)
    ok &= check(root['bracketed'] and root['nroots'] == 1,
                'the QP equation has exactly ONE root near eps when gapped',
                f"nroots={root['nroots']}, roots={[f'{r:.6f}' for r in root['roots']]}")
    ok &= check(abs(root['qp'] - qp_new) < 5e-3,
                'and the root solve agrees with the linearization where Z ~ 1',
                f"root {root['qp']:.8f} vs linearized {qp_new:.8f} Ha, "
                f"Z_root = {root['Z']:.4f}")

    # The pole-strength floor. A root ON a pole of Sigma_c solves the equation
    # with Z -> 0 and carries no spectral weight -- it is not a quasiparticle.
    # Bulk Al at 6x6x6 returned exactly that: a single bracketed root with
    # z_min = -3.2e-06, clean in every other column. Driving z_min above this
    # system's honest Z proves the filter FIRES without needing a pathological
    # system to hand.
    strict = qp_energy_root_occ(dfints_gapped,
                                solve_rpa_all_q_occ(dfints_gapped), 0,
                                nocc - 1, z_min=0.999)
    ok &= check(not strict['bracketed'] and strict['n_rejected'] >= 1,
                'a root below the pole-strength floor is REJECTED, not '
                'returned as the least-bad option',
                f"n_rejected={strict['n_rejected']}, "
                f"reason={strict.get('reject_reason', '')[:44]}")
    ok &= check(np.isnan(strict['qp']),
                'and the caller gets NaN rather than a plausible number')
    assert ok


# --------------------------------------------- 3. what the collapse costs

def test_integer_collapse_costs(dfints_metal):
    f = np.asarray(dfints_metal.mo_occ, dtype=float)
    nocc = (f > 1e-8).sum(axis=1)
    kc = dfints_metal.kconserv_pair[0]
    n_int = pc._transition_layout(dfints_metal, kc)[1]
    _, _, n_occ = transition_layout(dfints_metal, kc)

    dead = int((nocc >= dfints_metal.nmo).sum())
    ok = check(dead > 0,
               'the integer collapse fills the band completely at some k, '
               'leaving NO transitions there',
               f'{dead} of {len(nocc)} k-points, nocc = {nocc.tolist()}')
    ok &= check(n_occ > 5 * max(n_int, 1),
                'so the occupation-weighted transition space is far larger',
                f'{n_int} integer vs {n_occ} occupation-weighted at q=0')

    # The new path does not consult dfints.nocc at all.
    A1, _ = build_rpa_matrices_occ(dfints_metal, kc)
    saved = np.array(dfints_metal.nocc, copy=True)
    try:
        dfints_metal.nocc = np.maximum(saved - 1, 0)
        A2, _ = build_rpa_matrices_occ(dfints_metal, kc)
    finally:
        dfints_metal.nocc = saved
    ok &= check(np.abs(A1 - A2).max() == 0.0,
                'and it ignores the integer nocc entirely',
                f'{np.abs(A1 - A2).max():.1e} when nocc is perturbed')
    assert ok


# ------------------------------------------------ 4. internal consistency

def test_spectral_w_matches_inversion(dfints_metal):
    kc = dfints_metal.kconserv_pair[0]
    Omega, rho = solve_rpa_spectral_occ(dfints_metal, kc)
    ok = True
    for omega in (0.0, 0.5, 2.0):
        Wsp = build_W_aux_spectral_occ(dfints_metal, Omega, rho, omega=omega)
        Pi = build_chi0_aux_occ(dfints_metal, kc, omega)
        Wdir = np.linalg.inv(np.eye(Pi.shape[0]) - Pi)
        rel = np.abs(Wsp - Wdir).max() / np.abs(Wdir).max()
        ok &= check(rel < 1e-12,
                    f'omega={omega}: spectral W == (1 - Pi0)^-1, which is what '
                    f'fixes the 2/nk constant', f'{rel:.1e}')
    assert ok


# ------------------------------------------------------------ 5. threshold

def test_threshold(dfints_metal):
    kc = dfints_metal.kconserv_pair[0]
    base = build_chi0_aux_occ(dfints_metal, kc, 0.0, threshold=1e-12)
    scale = np.abs(base).max()
    ok = True
    counts = {}
    print(f"       {'threshold':>10} {'pairs':>7} {'chi0 change':>13}")
    for th in (1e-2, 1e-4, 1e-6, 1e-8):
        x = build_chi0_aux_occ(dfints_metal, kc, 0.0, threshold=th)
        counts[th] = transition_layout(dfints_metal, kc, threshold=th)[2]
        rel = np.abs(x - base).max() / scale
        print(f"       {th:10.0e} {counts[th]:7d} {rel:13.2e}")
        if th <= DEFAULT_OCCUPATION_THRESHOLD:
            ok &= check(rel < 1e-12,
                        f'threshold {th:.0e} leaves chi0 unchanged')
    ok &= check(counts[1e-2] < counts[1e-8],
                'a looser threshold really does prune pairs',
                f'{counts[1e-8]} -> {counts[1e-2]}')
    assert ok


# ----------------------------------------------------- 6. the smearing limit

def test_smearing_limit():
    """On a GAPPED system, occupation weighting must return the integer answer
    for any smearing small compared with the gap."""
    cell, mf = h2_slab()
    dfints = integrals(cell, mf, (2, 2, 1))
    kc = dfints.kconserv_pair[0]
    ref = pc.build_chi0_aux(dfints, kc, 0.0)

    e = np.asarray(mf.mo_energy)
    nocc = dfints.nocc[0]
    gap = e[:, nocc:].min() - e[:, :nocc].max()
    ok = check(gap > 0, 'the slab is gapped', f'{gap:.4f} Ha')

    print(f"       {'sigma (Ha)':>11} {'max |f-round|':>14} {'chi0 rel':>10}")
    prev = None
    for sigma in (0.05, 0.02, 0.005):
        mu = 0.5 * (e[:, nocc:].min() + e[:, :nocc - 1 + 1].max())
        occ = 2.0 / (1.0 + np.exp((e - mu) / sigma))
        chi = build_chi0_aux_occ(dfints, kc, 0.0, mo_occ=occ, threshold=1e-12)
        rel = np.abs(chi - ref).max() / np.abs(ref).max()
        dev = np.abs(occ - np.round(occ / 2.0) * 2.0).max()
        print(f"       {sigma:11.4f} {dev:14.2e} {rel:10.2e}")
        if prev is not None:
            ok &= check(rel < prev,
                        f'sigma {sigma}: closer to the integer answer than the '
                        f'previous smearing', f'{rel:.2e} < {prev:.2e}')
        prev = rel
    ok &= check(prev < 1e-6,
                'and converges onto it for a smearing well below the gap',
                f'{prev:.2e}')
    assert ok


# ------------------------------------------------- 7. continuity under strain

def _bcc_li(a_ang, kmesh=(2, 2, 2), sigma=0.01):
    """PBE orbitals, not HF -- the mean field is part of the specification.

    The integer-occupation STEP LOCATION is a property of the mean field as
    well as the geometry: the counts here were measured with KRKS/PBE, and an
    otherwise identical KRHF cell puts no step between 3.50 and 3.70 at all,
    so the test would sample three points on a smooth stretch and prove
    nothing. Caught by the guard below, which is why that guard exists.
    """
    cell = pgto.Cell()
    cell.atom = f'Li 0 0 0; Li {a_ang / 2} {a_ang / 2} {a_ang / 2}'
    cell.a = np.diag([a_ang] * 3)
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.build()
    mf = pscf.KRKS(cell, cell.make_kpts(list(kmesh))).density_fit()
    mf.xc = 'pbe'
    mf = pbc_addons.smearing_(mf, sigma=sigma, method='fermi')
    mf.conv_tol = 1e-9
    mf.kernel()
    # A PURE functional needs no exact exchange, so pyscf builds the GDF with
    # j_only=True: only the DIAGONAL k-pair blocks exist, and from_scf dies with
    # KeyError 'j3c/1' when it asks for an off-diagonal one. The RPA driver
    # never sees this because it evaluates exact exchange first, which forces
    # the full build. Clearing _cderi is required as well -- build() returns
    # immediately if it is already set, so asking again without clearing is a
    # silent no-op that leaves the half-built file in place.
    mf.with_df._cderi = None
    mf.with_df.build(j_only=False)
    return cell, mf


def test_correlation_energy_is_continuous_under_strain():
    """E_c must not jump when the integer nocc does.

    BCC Li at 2x2x2 crosses an integer-occupation step between a = 3.50 and
    3.60 A: nocc at one k-point goes 3 -> 6 and the integer transition space
    509 -> 536. The occupation-weighted response has no such step, because the
    weights that appear go to zero continuously.

    Two bars, and the SIGN one is the sharp one. The integer path does not
    merely wobble across the step, it reverses: measured d1 = -0.01088789 then
    +0.00429695, so consistency of sign catches it with no threshold at all.
    The magnitude bar is then expressed as a SHAPE ratio |d2|/min|d1| rather
    than an absolute tolerance, because an absolute one is arbitrary and does
    not travel: on a wide volume scan the physical curvature alone reaches 44%
    of the smallest first difference.

    The ratio is 0.069 for THIS cell. Do not confuse that with the
    cross-implementation check, which is a SEPARATE measurement: on the
    Li equation-of-state setup an independent THC-ISDF implementation got
    0.063 against 0.064 for this GDF code, agreeing to 1.6%. Same
    three lattice constants, but a different exxdiv convention and FFT mesh, so
    E_c differs by ~0.007 Ha and the ratio moves ~8%. The ratio travels between
    IMPLEMENTATIONS at a fixed setup; it does not travel between setups, and
    the band below is sized for that. The sign check above is the sharp bar.
    """
    from src.SingleReference.Periodic.pbc_integrals import PBCDFIntegrals
    lattices = (3.50, 3.60, 3.70)
    ec, noccs = [], []
    for a_ang in lattices:
        cell, mf = _bcc_li(a_ang)
        f = np.asarray(mf.mo_occ)
        noccs.append((f > 1e-8).sum(axis=1).tolist())
        dfints = PBCDFIntegrals.from_scf(cell, mf)
        ec.append(ri_rpa_ecorr_from_dfints(dfints, nw=16,
                                           beta=beta_from_mf(mf)))
    d1 = [ec[1] - ec[0], ec[2] - ec[1]]
    d2 = abs(d1[1] - d1[0])
    print(f"       a = {lattices}")
    print(f"       nocc collapse: {noccs}")
    print(f"       E_c = " + '  '.join(f'{x:.8f}' for x in ec))
    print(f"       first differences {d1[0]:+.8f} {d1[1]:+.8f}, second {d2:.2e}")

    ok = check(noccs[0] != noccs[-1],
               'the integer nocc really does step across this range -- '
               'otherwise the test proves nothing', f'{noccs[0]} -> {noccs[-1]}')
    ok &= check(all(x < 0 for x in ec), 'E_c is negative throughout')
    ok &= check(d1[0] * d1[1] > 0,
                'E_c keeps its SIGN across the step -- the integer path '
                'reverses here (-0.0109 then +0.0043)',
                f'{d1[0]:+.8f} {d1[1]:+.8f}')
    shape = d2 / min(abs(d1[0]), abs(d1[1]))
    ok &= check(abs(shape - 0.069) < 0.02,
                'and the SHAPE ratio |d2|/min|d1| matches the value recorded '
                'for this cell',
                f'{shape:.3f} against 0.069 recorded')
    assert ok


def _run(label, fn, *args):
    """Run one check, report, and KEEP GOING.

    The tests assert (so pytest sees a real pass/fail) rather than returning a
    bool, but the script mode is worth preserving as it was: one failure should
    not hide the other six, since which checks fail together is the diagnosis.
    """
    print(f'\n-- {label}')
    try:
        fn(*args)
        return True
    except AssertionError as err:
        print(f'   FAILED: {err}' if str(err) else '   FAILED')
        return False


if __name__ == '__main__':
    all_ok = True

    cell_h, mf_h = h2_slab()
    dfints_h = integrals(cell_h, mf_h, (2, 2, 1))
    all_ok &= _run('2. exact reduction to the integer path (gapped slab)',
                   test_reduction_to_integer_path, dfints_h)
    all_ok &= _run('6. the smearing limit', test_smearing_limit)

    cell_m, mf_m = li_layer()
    all_ok &= _run('1. the decision, measured on a metal',
                   test_the_decision, cell_m, mf_m)
    dfints_m = integrals(cell_m, mf_m, (3, 3, 1))
    all_ok &= _run('1b. the Casida solve is well posed',
                   test_casida_is_well_posed, dfints_m)
    all_ok &= _run('3. what the integer collapse costs',
                   test_integer_collapse_costs, dfints_m)
    all_ok &= _run('4. spectral W against the direct inversion',
                   test_spectral_w_matches_inversion, dfints_m)
    all_ok &= _run('5. the threshold regularization', test_threshold, dfints_m)
    all_ok &= _run('7. E_c is continuous when the integer nocc steps',
                   test_correlation_energy_is_continuous_under_strain)

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
