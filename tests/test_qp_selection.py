"""Gates for the pure selection layer of the adaptive explicit set
(`src/SingleReference/GW/qp_selection.py`): the first-order weights, the
calibration of the continuation, the AC-matched tiers and the greedy rule.

No SCF: every spectrum here is synthetic, so the numbers each gate checks are
known exactly and the file runs in under a second.
"""
import numpy as np
import pytest

from src.Base.constants import (ADAPTIVE_HOLE_MAX_EV, HARTREE_TO_EV,
                                HARTREE_TO_MEV)
from src.Base.declaration import QPStates
from src.SingleReference.GW.qp_selection import (AdaptivePartition,
                                                 ac_unjudgeable,
                                                 calibrate_ac,
                                                 calibrated_selection,
                                                 casida_instability,
                                                 degenerate_blocks,
                                                 error_terms,
                                                 first_order_weights,
                                                 hole_errors,
                                                 mandatory_members,
                                                 matched_tiers,
                                                 near_root_targets,
                                                 select_explicit)
from src.SingleReference.GW.qp_states import (calibrate_scissor_tiers,
                                              resolve_qp_states)
from src.gradients.bse_isdf import bse_solve

#: Hellmann-Feynman against a central difference in each eps (Ha per Ha).
HF_FD_STEP = 1e-5
HF_FD_TOL = 1e-7
DEG = 1e-4

EV = 1.0 / HARTREE_TO_EV


def casida(eps, nocc, k_a, k_b):
    """(Omega, X, Y) of A = diag(eps_a - eps_i) + K_A, B = K_B."""
    nvir = len(eps) - nocc
    d = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    return bse_solve(np.diag(d) + k_a, k_b)


def spectrum():
    """Ten orbitals, five occupied, a window of all ten."""
    eps = np.array([-1.20, -0.90, -0.62, -0.50, -0.40,
                    0.05, 0.12, 0.30, 0.55, 0.90])
    return eps, 5, tuple(range(10))


def test_weights_are_the_hellmann_feynman_derivative():
    """n from X and Y is dOmega/deps_p of every root, occupied and virtual."""
    rng = np.random.default_rng(7)
    eps = np.array([-0.8, -0.5, 0.2, 0.4, 0.7])
    nocc, n_ov = 2, 6
    k = rng.normal(scale=0.03, size=(n_ov, n_ov))
    k_a, k_b = 0.5 * (k + k.T), None
    kb = rng.normal(scale=0.01, size=(n_ov, n_ov))
    k_b = 0.5 * (kb + kb.T)
    om, xn, yn = casida(eps, nocc, k_a, k_b)
    n = first_order_weights(xn, yn, nocc, len(eps))
    for p in range(len(eps)):
        up, dn = eps.copy(), eps.copy()
        up[p] += HF_FD_STEP
        dn[p] -= HF_FD_STEP
        fd = (casida(up, nocc, k_a, k_b)[0]
              - casida(dn, nocc, k_a, k_b)[0]) / (2 * HF_FD_STEP)
        assert np.max(np.abs(fd - n[:, p])) < HF_FD_TOL, (p, fd, n[:, p])
    # one root alone is the same row
    assert np.array_equal(first_order_weights(xn[:, 0], yn[:, 0], nocc,
                                              len(eps)), n[0])


def synthetic():
    """A window with known shifts, weights and two targets."""
    eps, nocc, cand = spectrum()
    a = {p: s * EV for p, s in zip(cand, [1.6, 1.1, 0.9, 0.7, 0.5,
                                          -0.4, -0.3, -0.1, 0.2, 0.6])}
    n_s = np.array([-0.01, -0.05, -0.20, -0.30, -0.44,
                    0.60, 0.25, 0.10, 0.04, 0.01])
    n_t = np.array([-0.02, -0.02, -0.50, -0.10, -0.36,
                    0.30, 0.50, 0.15, 0.03, 0.02])
    return eps, nocc, cand, a, {('singlet', 0): n_s, ('triplet', 0): n_t}


def test_greedy_stops_at_the_budget_and_is_monotone():
    eps, nocc, cand, a, w = synthetic()
    tol = 30.0
    start = mandatory_members(eps, nocc, cand, DEG)
    sel = select_explicit(eps, nocc, cand, a, w, tol, start=start,
                          degeneracy_tol=DEG, hole_max_ev=None)
    assert max(sel.budget.values()) * HARTREE_TO_MEV <= tol
    assert sel.order, 'the synthetic budget must need at least one addition'
    blocks = degenerate_blocks(eps, cand, DEG)
    chosen = set(start)
    path = []
    for block, _, _ in [(None, None, None)] + sel.order:
        if block is not None:
            chosen |= set(block)
        shift = {q: a[q] for q in chosen}
        tiers = matched_tiers(eps, nocc, a, shift, chosen, blocks)
        path.append(max(error_terms(w, a, None, shift, tiers,
                                    blocks)[1].values()))
    assert all(path[i + 1] <= path[i] for i in range(len(path) - 1)), path
    # it stops at the FIRST set inside the budget
    assert all(b * HARTREE_TO_MEV > tol for b in path[:-1])
    assert set(sel.explicit) == chosen


def test_the_tolerance_is_per_target_not_on_the_gap():
    """Two targets with the same signed error: their gap is exact, each
    state is not, and the rule still solves the hole."""
    eps, nocc, cand, a, _ = synthetic()
    n = np.zeros(len(eps))
    n[2] = -0.5
    w = {('singlet', 0): n, ('triplet', 0): n.copy()}
    start = mandatory_members(eps, nocc, cand, DEG)
    shift = {q: a[q] for q in start}
    blocks = degenerate_blocks(eps, cand, DEG)
    tiers = matched_tiers(eps, nocc, a, shift, start, blocks)
    signed, budget, _ = error_terms(w, a, None, shift, tiers, blocks)
    assert signed[('singlet', 0)] == signed[('triplet', 0)] != 0.0
    tol = 0.5 * budget[('singlet', 0)] * HARTREE_TO_MEV
    sel = select_explicit(eps, nocc, cand, a, w, tol, start=start,
                          degeneracy_tol=DEG)
    assert 2 in sel.explicit


def test_edges_and_frontier_are_always_explicit():
    eps, nocc, cand, a, w = synthetic()
    why = mandatory_members(eps, nocc, cand, DEG)
    assert why == {0: 'edge', 9: 'edge', 4: 'frontier', 5: 'frontier'}
    sel = select_explicit(eps, nocc, cand, a, w, 1e9, start=why,
                          degeneracy_tol=DEG, hole_max_ev=None)
    assert sel.explicit == (0, 4, 5, 9) and sel.fell_back is None


def test_unjudgeable_ac_forces_explicit():
    eps, nocc, cand, a, w = synthetic()
    a = dict(a)
    a[7] = float('nan')                       # a Pade failure
    a[2] = 5.0 * EV                           # beyond the shift ceiling
    z = {p: 0.9 for p in cand}
    z[6] = 1.3                                # not a quasiparticle
    bad = ac_unjudgeable(a, z, shift_max_ev=3.0)
    assert set(bad) == {2, 6, 7}
    why = mandatory_members(eps, nocc, cand, DEG, unjudgeable=bad)
    assert {why[p] for p in (2, 6, 7)} == {'ac_unjudgeable'}


def degenerate_case():
    """Orbitals 2 and 3 degenerate (a hole pair), 6 and 7 degenerate."""
    eps = np.array([-1.20, -0.90, -0.60, -0.60 + 1e-6, -0.40,
                    0.05, 0.30, 0.30 + 2e-6, 0.55, 0.90])
    nocc, cand = 5, tuple(range(10))
    a = {p: s * EV for p, s in zip(cand, [1.6, 1.1, 0.9, 0.92, 0.5,
                                          -0.4, -0.3, -0.28, 0.2, 0.6])}
    n = np.array([-0.01, -0.05, -0.20, -0.30, -0.44,
                  0.60, 0.25, 0.10, 0.04, 0.01])
    return eps, nocc, cand, a, {('singlet', 0): n}


def test_degenerate_blocks_are_never_split():
    eps, nocc, cand, a, w = degenerate_case()
    blocks = degenerate_blocks(eps, cand, DEG)
    assert (2, 3) in blocks and (6, 7) in blocks
    # as a hole: one tier for the pair
    start = mandatory_members(eps, nocc, cand, DEG)
    shift = {q: a[q] for q in start}
    tiers = matched_tiers(eps, nocc, a, shift, start, blocks)
    assert tiers[2] == tiers[3]
    # in the greedy order: the pair comes in together or not at all
    for tol in (0.0, 1.0, 5.0, 20.0, 80.0):
        sel = select_explicit(eps, nocc, cand, a, w, tol, start=start,
                              degeneracy_tol=DEG)
        for b in blocks:
            assert set(b) <= set(sel.explicit) or not set(b) & set(sel.explicit)
    # as a probe: the block lends the mean of its members' shifts
    roots = {6: eps[6] + 0.010, 7: eps[7] + 0.014}
    got = calibrate_scissor_tiers(eps, roots, {8: (6, 7)})
    assert got[8] == pytest.approx(0.012, abs=1e-15)


def test_calibration_removes_a_uniform_offset():
    eps, nocc, cand, a, w = synthetic()
    explicit = {0: a[0] - 0.05 * EV, 4: a[4] + 0.02 * EV,
                5: a[5] - 0.01 * EV, 9: a[9] + 0.04 * EV}
    shifted = {p: v + 0.2 * EV for p, v in a.items()}
    one = calibrate_ac(eps, nocc, a, explicit, cand)
    two = calibrate_ac(eps, nocc, shifted, explicit, cand)
    for p in cand:
        assert two.a_tilde[p] == pytest.approx(one.a_tilde[p], abs=1e-14)
    s1, _ = calibrated_selection(eps, nocc, cand, a, explicit, w, 5.0,
                                 degeneracy_tol=DEG, gate_ev=1.0)
    s2, _ = calibrated_selection(eps, nocc, cand, shifted, explicit, w, 5.0,
                                 degeneracy_tol=DEG, gate_ev=1.0)
    assert s1.explicit == s2.explicit and s1.tier_of == s2.tier_of
    # the calibration is exact on the explicit states and interpolates between
    assert one.a_tilde[4] == explicit[4]
    r0, r4 = a[0] - explicit[0], a[4] - explicit[4]
    t = (eps[2] - eps[0]) / (eps[4] - eps[0])
    assert one.a_tilde[2] == pytest.approx(a[2] - ((1 - t) * r0 + t * r4),
                                           abs=1e-15)
    assert one.u[2] == pytest.approx(0.5 * abs(r4 - r0), abs=1e-15)


def test_calibration_gate_falls_back_to_the_window():
    eps, nocc, cand, a, w = synthetic()
    explicit = {0: a[0], 4: a[4] - 0.2 * EV, 5: a[5], 9: a[9]}
    sel, cal = calibrated_selection(eps, nocc, cand, a, explicit, w, 5.0,
                                    degeneracy_tol=DEG, gate_ev=0.1)
    assert sel.fell_back == 'calibration' and sel.explicit == cand
    assert cal.residual[4] == pytest.approx(0.2 * EV, abs=1e-15)


def test_whole_window_is_the_admitted_set():
    eps, nocc, cand, a, w = synthetic()
    start = mandatory_members(eps, nocc, cand, DEG)
    sel = select_explicit(eps, nocc, cand, a, w, 0.0, start=start,
                          degeneracy_tol=DEG)
    assert sel.explicit == cand and sel.fell_back == 'window'
    assert sel.tier_of == {}


def test_resolve_adaptive_returns_the_admitted_candidates():
    eps = np.array([-20.0, -1.4, -0.9, -0.6, -0.45, -0.35,
                    0.05, 0.1, 0.3, 0.6, 1.2, 4.0, 30.0])
    nocc = 6
    adm = resolve_qp_states(QPStates('admitted'), eps, nocc,
                            degeneracy_tol=DEG)
    ada = resolve_qp_states(QPStates('adaptive', tol_meV=2.0), eps, nocc,
                            degeneracy_tol=DEG)
    assert ada.explicit == adm.explicit and ada.outside == adm.outside
    assert 0 < len(ada.explicit) < len(eps)
    assert ada.label.startswith('adaptive(admitted(gap)) candidates:')


def test_the_declaration_carries_its_tolerance_and_targets():
    spec = QPStates('adaptive', tol_meV=1.0,
                    targets=[('singlet', 0), ['triplet', 0]])
    assert spec.targets == (('singlet', 0), ('triplet', 0))
    with pytest.raises(ValueError):
        QPStates('admitted', tol_meV=1.0)
    with pytest.raises(ValueError):
        QPStates('adaptive', targets=(('quartet', 0),))
    with pytest.raises(ValueError):
        QPStates('adaptive', tol_meV=-1.0)


def test_the_partition_round_trips_and_refuses_an_inconsistent_map():
    part = AdaptivePartition(explicit=(0, 4, 5, 9), tier_of={
        1: (0,), 2: (4,), 3: (4,), 6: (5,), 7: (5,), 8: (9,)},
        candidates=range(10), targets=(('singlet', 0),), tol_meV=1.0)
    back = AdaptivePartition.from_record(part.as_record())
    assert back == part and back.tier_of == part.tier_of
    assert part.holes == (1, 2, 3, 6, 7, 8)
    with pytest.raises(ValueError):
        AdaptivePartition(explicit=(0, 4), tier_of={1: (0,)},
                          candidates=range(5), targets=(), tol_meV=1.0)


def test_near_roots_join_the_targets():
    om = {'singlet': np.array([3.00, 3.05, 3.50]) * EV,
          'triplet': np.array([2.50, 2.70]) * EV}
    got = near_root_targets(om, (('singlet', 0), ('triplet', 0)), 0.1)
    assert got == (('singlet', 0), ('triplet', 0), ('singlet', 1))


def small_casida():
    """(A, B) of a stable synthetic pair space: two occupied, three virtual."""
    rng = np.random.default_rng(7)
    eps = np.array([-0.8, -0.5, 0.2, 0.4, 0.7])
    nocc, n_ov = 2, 6
    k = rng.normal(scale=0.03, size=(n_ov, n_ov))
    kb = rng.normal(scale=0.01, size=(n_ov, n_ov))
    d = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    return np.diag(d) + 0.5 * (k + k.T), 0.5 * (kb + kb.T)


def test_a_stable_window_keeps_the_full_bse():
    """Both A - B and A + B positive definite: no fallback."""
    a, b = small_casida()
    assert casida_instability(a, b) is None
    assert casida_instability(a, None) is None


@pytest.mark.parametrize('flip, name', [(+1, 'A - B'), (-1, 'A + B')])
def test_an_unstable_window_is_named(flip, name):
    """A - B or A + B not positive definite: the full solve has no real
    Omega, and the matrix that failed is named, so the selection reads
    Tamm-Dancoff weights instead."""
    a, _ = small_casida()
    # B = +-(A + 0.1) sends A -+ B negative on the whole diagonal
    b = flip * (a + 0.1 * np.eye(len(a)))
    with pytest.raises((np.linalg.LinAlgError, ValueError)):
        bse_solve(a, b)
    assert casida_instability(a, b) == name


def test_a_hole_no_target_reaches_is_still_capped():
    """No target has weight on orbitals 1-3 and 6-8, so the budget alone
    leaves them all holes, whatever their errors (up to 0.5 eV here). The cap
    solves the worst first, one at a time: with 0.25 eV, solving 1 (0.5 eV
    off) brings 2 (0.4 eV off until then) to 0.2 eV, so 2 stays a hole; on the
    virtual side 8 (0.4 eV) and then 7 (0.3 eV once 8 is a probe) are solved.
    Every hole left is within the cap; no cap is the budget rule alone."""
    eps, nocc, cand, a, _ = synthetic()
    n = np.zeros(len(eps))
    n[4], n[5] = -1.0, 1.0                     # a pure frontier transition
    w = {('singlet', 0): n, ('triplet', 0): n.copy()}
    start = mandatory_members(eps, nocc, cand, DEG)
    bare = select_explicit(eps, nocc, cand, a, w, 1.0, start=start,
                           degeneracy_tol=DEG, hole_max_ev=None)
    assert bare.explicit == (0, 4, 5, 9) and not bare.capped
    assert max(bare.budget.values()) == 0.0
    sel = select_explicit(eps, nocc, cand, a, w, 1.0, start=start,
                          degeneracy_tol=DEG, hole_max_ev=0.25)
    assert sel.explicit == (0, 1, 4, 5, 7, 8, 9) and not sel.order
    assert [b for b, _ in sel.capped] == [(1,), (8,), (7,)]
    assert [e * HARTREE_TO_EV for _, e in sel.capped] == pytest.approx(
        [0.5, 0.4, 0.3], abs=1e-12)
    shift = {q: a[q] for q in sel.explicit}
    left = hole_errors(a, None, shift, sel.tier_of,
                       degenerate_blocks(eps, cand, DEG))
    assert set(left) == {(2,), (3,), (6,)}
    assert max(left.values()) * HARTREE_TO_EV <= 0.25
    # u counts against the cap: hole 6 is 0.1 eV off, 0.2 eV with u
    near = select_explicit(eps, nocc, cand, a, w, 1.0, start=start,
                           degeneracy_tol=DEG, hole_max_ev=0.25,
                           u={6: 0.2 * EV})
    assert 6 in near.explicit


def test_the_default_cap_is_the_constant_and_spares_a_barred_root():
    """Without a hole_max_ev the constant applies, so no hole is left over
    ADAPTIVE_HOLE_MAX_EV; a root the surface rejected stays a hole anyway."""
    eps, nocc, cand, a, _ = synthetic()
    w = {('singlet', 0): np.zeros(len(eps))}
    start = mandatory_members(eps, nocc, cand, DEG)
    sel = select_explicit(eps, nocc, cand, a, w, 1.0, start=start,
                          degeneracy_tol=DEG)
    shift = {q: a[q] for q in sel.explicit}
    blocks = degenerate_blocks(eps, cand, DEG)
    assert sel.capped and all(
        e * HARTREE_TO_EV <= ADAPTIVE_HOLE_MAX_EV
        for e in hole_errors(a, None, shift, sel.tier_of, blocks).values())
    barred = select_explicit(eps, nocc, cand, a, w, 1.0, start=start,
                             degeneracy_tol=DEG, barred=(2,))
    assert 2 not in barred.explicit and 2 in barred.tier_of
