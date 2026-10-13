"""The quasiparticle Newton must reach a root that lies inside its pole guard.

The iteration is kept `QP_POLE_OFFSET` away from every orbital energy.
Sigma is continuous there (`cd_integral_weights`), so the band only sets the
Newton's start and the path a frozen surface records. When the quasiparticle
correction is smaller than the band, the guard and the Newton step fight: the
step carries the iterate off the guard, the guard puts it back, and the two
form a cycle that runs to max_iter at a point which is not the root. The band
therefore yields, down to `QP_POLE_OFFSET_MIN`; a root on an orbital energy is
refused.

The flat-screening models carry the physical sign, W^c_pp(0) < 0: Sigma^int
beside eps_HOMO is -W^c(0)/2 > 0, which keeps the HOMO's root inside the
window no residue reaches.

The capture and satellite fallbacks answer a Sigma with a genuine pole at an
orbital energy, which a contour-deformation Sigma does not have, so they are
exercised on the generic Newton with a model self-energy s0 + A / (w - eps_q).
"""
import warnings

import numpy as np
import pytest

from src.Base.constants import (QP_CD_NEWTON_TOL, QP_POLE_OFFSET,
                                QP_POLE_OFFSET_MIN, QP_POLE_STRENGTH_MIN)
from src.SingleReference.GW.contour_deformation import qp_energy_cd
from src.Solvers.qp_equation import (PoleGuardRefusal,
                                     solve_qp_equation_newton_guarded)

#: A flat imaginary-axis screening scales the quasiparticle shift, which is how
#: the root is placed relative to the guard: -4e-3 puts it 5.1e-4 from the
#: pole, inside the 1e-3 guard, and -1e-2 1.3e-3 away, outside it.
EPS = np.array([-0.9, -0.6, -0.35, 0.15, 0.4, 0.8])
NOCC = 3
NU = np.array([0.05, 0.3, 1.0, 4.0])
WT = np.array([0.1, 0.3, 0.8, 3.0])
INSIDE, OUTSIDE = -4e-3, -1e-2


def solve(screening, p=NOCC - 1):
    """(root, Z, warnings) for a flat screening of the given size."""
    Bp = np.zeros((3, len(EPS)))          # frontier state: no residues swept
    wc = np.full((len(NU), len(EPS)), screening)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w, z, _ = qp_energy_cd(p, Bp, EPS, NOCC, NU, WT, wc=wc)
    return w, z, [str(c.message) for c in caught if 'pole guard' in str(c.message)]


def test_a_root_inside_the_guard_is_reached_not_cycled():
    w, _, notes = solve(INSIDE)
    shift = abs(w - EPS[NOCC - 1])
    assert QP_POLE_OFFSET_MIN < shift < QP_POLE_OFFSET, shift
    assert notes and 'relaxed to' in notes[0]


def test_the_guard_is_untouched_when_the_root_is_outside_it():
    """The common case must not pay for the rare one: no relaxation, no warning."""
    w, _, notes = solve(OUTSIDE)
    assert abs(w - EPS[NOCC - 1]) > QP_POLE_OFFSET
    assert not notes


def test_a_root_on_the_pole_is_refused_rather_than_answered():
    """No screening puts the root exactly at eps_p, where no offset reaches
    it: a limit of the guard, and it says so."""
    with pytest.raises(RuntimeError, match='cannot be resolved by this'):
        solve(0.0)


def test_the_refusal_names_the_orbital_that_actually_blocked_it():
    """The guard fires on the nearest eps to the ITERATE, which is p only at
    the first step. Naming p regardless reads as "the quasiparticle correction
    is tiny" -- for a frontier orbital that is false, and it hides the real
    statement, that the root is nearly degenerate with some OTHER level."""
    with pytest.raises(RuntimeError) as exc:
        solve(0.0)
    message = str(exc.value)
    assert 'pinned against orbital' in message
    # This construction really does pin against p itself, so it must say so
    # rather than pointing at a neighbour.
    assert 'the same orbital' in message, message


def test_the_floor_is_below_the_guard():
    assert 0 < QP_POLE_OFFSET_MIN < QP_POLE_OFFSET


def test_a_healthy_root_is_left_alone():
    """The guard must not fire on the ordinary case: no warning, Z near one."""
    w, z, notes = solve(OUTSIDE)
    assert z > 0.5, z
    assert not any('satellite' in n or 'captured' in n for n in notes)


# A self-energy with a genuine pole at a neighbour's orbital energy. Orbital 4
# sits 0.05 Ha above orbital 3 and the constant pulls it down by exactly that;
# with A < 0 there is no root near the pole, and the Newton is drawn onto it.
MODEL_EPS = np.array([-0.9, -0.6, -0.35, 0.15, 0.20, 0.8])
MODEL_P, MODEL_Q = 4, 3


def model_newton(s0, A, **kw):
    """(w, Z, warnings) of the guarded Newton on Sigma = s0 + A/(w - eps_q)."""
    def sigma(w):
        return s0 + A / (w - MODEL_EPS[MODEL_Q])

    def slope(w):
        return -A / (w - MODEL_EPS[MODEL_Q]) ** 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w, z = solve_qp_equation_newton_guarded(sigma, slope, MODEL_EPS,
                                                MODEL_P, NOCC, **kw)
    return w, z, [str(c.message) for c in caught]


CAPTURE = dict(s0=-0.05, A=-1e-6)
#: A root 2e-3 above the pole with dSigma/dw = -10 there, so Z = 1/11; the
#: start is placed beside it, as a seed from another geometry would be.
SATELLITE = dict(s0=-0.068, A=4e-5, w0=0.1525)


def test_capture_by_another_orbitals_pole_is_named_and_recovered():
    """Newton near a pole takes ever smaller steps and converges onto it
    instead of onto a root. A pin against q != p is the tell: orbital p's own
    root has no reason to sit 1e-4 Ha from a different orbital energy."""
    w, _, notes = model_newton(**CAPTURE)
    hit = [n for n in notes if 'captured by the pole' in n]
    assert hit, notes
    assert 'orbital 3' in hit[0] and 'not by a root of its own' in hit[0]
    assert 'LINEARIZED' in hit[0], hit[0]
    assert min(abs(w - MODEL_EPS)) > QP_POLE_OFFSET_MIN, w


def test_capture_can_be_refused_instead():
    """The linearized value is a DIFFERENT approximation from the
    self-consistent root, so refusing has to stay available."""
    with pytest.raises(RuntimeError, match='pinned against orbital 3'):
        model_newton(**CAPTURE, linearize_on_capture=False)


def test_a_satellite_root_is_rejected_even_when_it_converged():
    """The failure mode that converges: a clean convergence is no evidence of
    the right root, only Z is."""
    _, z_raw, _ = model_newton(**SATELLITE, linearize_on_capture=False,
                               z_min=0.0)
    assert 0.0 < z_raw < QP_POLE_STRENGTH_MIN, 'the model must reach the satellite'
    _, z, notes = model_newton(**SATELLITE)
    assert z >= QP_POLE_STRENGTH_MIN, (
        f'a root returned with Z={z:.3f} carries no spectral weight')
    assert any('LINEARIZED' in n for n in notes)


def test_a_pinned_iterate_is_not_accepted_as_converged():
    """The step is measured from the point the guard just pushed to, so a tiny
    step there looks like convergence while the iterate sits exactly `offset`
    from eps_q -- the guard's position, not a root of f. Whatever comes back
    must be a real root: free of the guard, or refused."""
    for floor in (1e-4, 1e-6):
        try:
            w, _, _ = model_newton(**CAPTURE, linearize_on_capture=False,
                                   offset_min=floor)
        except RuntimeError:
            continue                       # refusing is the other valid answer
        # Not sitting on any guard position: a returned root is a root.
        gaps = np.abs(MODEL_EPS - w)
        assert gaps.min() > floor, (
            f'floor {floor:.0e}: returned w={w:.8f}, {gaps.min():.2e} from '
            f'orbital {int(np.argmin(gaps))} -- that is the guard, not a root')


# A root inside the guard band of its own orbital, reached across eps_p: the
# root sits 3.5e-4 Ha above eps_p with dSigma/dw = -0.074, and a slope that
# misleads near eps_p sends the Newton across the pole. The guard pushes it out
# on the other side, the step from there lands outside the band, and the next
# step falls back in: pushes at -band, +band, -band, ... that never repeat
# consecutively. The model is that step map: f(x) = x - Sigma is
# 1.074 (x - x*) everywhere, and only the slope misleads, where it does.
OWN_EPS = np.array([-0.69, -0.65, -0.5358, -0.4407, 0.1197, 0.1472])
OWN_NOCC, OWN_P, OWN_ROOT = 4, 2, 3.5e-4


def own_pole_cycle():
    """(sigma, slope) whose Newton cycles across eps_p at the default guard."""
    def f_and_slope(w):
        x = w - OWN_EPS[OWN_P]
        f = 1.074 * (x - OWN_ROOT)
        if 4e-4 < x <= 1.5e-3:
            fp = f / (1.1 * x)               # lands at -x/10: across eps_p
        elif x < 0.0:
            fp = f / (x - 2e-3)              # lands at +2e-3: outside the band
        else:
            fp = 1.074                       # resolved: lands on the root
        return f, fp

    def sigma(w):
        return w - OWN_EPS[OWN_P] - f_and_slope(w)[0]

    def slope(w):
        return 1.0 - f_and_slope(w)[1]
    return sigma, slope


def test_a_cycle_across_the_own_pole_still_relaxes_the_guard():
    """A pushed value that recurs within one band is the deadlock, not only
    the same one twice in a row. With the two-cycle tell alone: 100 steps,
    band never shrunk, refused."""
    sigma, slope = own_pole_cycle()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w, z = solve_qp_equation_newton_guarded(sigma, slope, OWN_EPS, OWN_P,
                                                OWN_NOCC)
    assert abs(w - OWN_EPS[OWN_P] - OWN_ROOT) < QP_CD_NEWTON_TOL
    assert abs(z - 1.0 / 1.074) < QP_CD_NEWTON_TOL, z
    assert any('relaxed to' in str(c.message) for c in caught)


def test_a_refusal_above_the_floor_does_not_claim_the_floor():
    """A band above the floor says so rather than claiming a root within the
    floor, and the refusal carries the orbital, the blocker and the band."""
    sigma, slope = own_pole_cycle()
    with pytest.raises(PoleGuardRefusal) as exc:
        solve_qp_equation_newton_guarded(sigma, slope, OWN_EPS, OWN_P,
                                         OWN_NOCC, max_iter=6)
    refusal = exc.value
    assert refusal.orbital == OWN_P and refusal.blocker == OWN_P
    assert refusal.band > QP_POLE_OFFSET_MIN
    message = str(refusal)
    assert 'within' not in message and 'floor' in message, message
    # at the floor, the floor is the statement
    with pytest.raises(PoleGuardRefusal, match='cannot be resolved by this') \
            as exc:
        solve(0.0)
    assert exc.value.band == QP_POLE_OFFSET_MIN
