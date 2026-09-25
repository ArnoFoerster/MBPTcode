"""The quasiparticle Newton must reach a root that lies inside its pole guard.

The iteration is kept `QP_POLE_OFFSET` away from every orbital energy, because
at omega = eps_q the imaginary-axis integrand collapses onto nu = 0 and no
quadrature resolves it. When the quasiparticle CORRECTION is smaller than that
margin, the guard and the Newton step fight: the step carries the iterate off
the guard, the guard puts it back, and the two form a cycle that runs to
max_iter at a point which is not the root.

Seen first on naphthalene/def2-SVP at orbital nocc-2, whose root sits 9.5e-4
from its own orbital energy; it took out fourteen cases of the gradient
campaign at once. The margin is numerical rather than physical, so it yields --
the pinned point's residual there (~6e-5 Ha) is far worse than the quadrature
accuracy given up by halving the offset. A root ON the pole is a different
thing and is still refused.
"""
import warnings

import numpy as np
import pytest

from src.Base.constants import (QP_POLE_OFFSET, QP_POLE_OFFSET_MIN,
                                QP_POLE_STRENGTH_MIN)
from src.SingleReference.GW.contour_deformation import qp_energy_cd

#: A flat imaginary-axis screening scales the quasiparticle shift, which is how
#: the root is placed relative to the guard: 2e-3 puts it 5.1e-4 from the pole,
#: inside the 1e-3 guard and outside the 1e-4 floor -- naphthalene's case.
EPS = np.array([-0.9, -0.6, -0.35, 0.15, 0.4, 0.8])
NOCC = 3
NU = np.array([0.05, 0.3, 1.0, 4.0])
WT = np.array([0.1, 0.3, 0.8, 3.0])


def solve(screening, p=NOCC - 1):
    """(root, Z, warnings) for a flat screening of the given size."""
    Bp = np.zeros((3, len(EPS)))          # frontier state: no residues swept
    wc = np.full((len(NU), len(EPS)), screening)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w, z, _ = qp_energy_cd(p, Bp, EPS, NOCC, NU, WT, wc=wc)
    return w, z, [str(c.message) for c in caught if 'pole guard' in str(c.message)]


def test_a_root_inside_the_guard_is_reached_not_cycled():
    w, _, notes = solve(2e-3)
    shift = abs(w - EPS[NOCC - 1])
    assert QP_POLE_OFFSET_MIN < shift < QP_POLE_OFFSET, shift
    assert notes and 'relaxed to' in notes[0]


def test_the_guard_is_untouched_when_the_root_is_outside_it():
    """The common case must not pay for the rare one: no relaxation, no warning."""
    w, _, notes = solve(5e-3)
    assert abs(w - EPS[NOCC - 1]) > QP_POLE_OFFSET
    assert not notes


def test_a_root_on_the_pole_is_refused_rather_than_answered():
    """No screening puts the root exactly at eps_p, where no offset resolves
    both the pole and the root -- that is a limit, not a bug, and it says so."""
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



# A root driven onto a NEIGHBOUR's orbital energy. Reproducing this needs
# residues -- C_ov -- because dragging a root across another level is exactly
# what sweeps one; a frontier state that sweeps none cannot be captured this
# way, which is why the model above could not show it. Orbital 4 sits 0.05 Ha
# above orbital 3 and the screening pulls it down by almost exactly that, the
# same shape as acetaldehyde/cam-B3LYP where orbital 13 was pulled 0.058 Ha
# down onto orbital 12.
CAPTURE_EPS = np.array([-0.9, -0.6, -0.35, 0.15, 0.20, 0.8])
CAPTURE_SCREENING = 0.16


def capture_model():
    """(p, Bp, C_ov, wc) whose Newton is captured by the pole at orbital 3."""
    rng = np.random.default_rng(7)
    b = rng.normal(scale=0.15, size=(3, len(CAPTURE_EPS), len(CAPTURE_EPS)))
    b = 0.5 * (b + b.transpose(0, 2, 1))
    c_ov = b[:, :NOCC, NOCC:].reshape(3, NOCC * (len(CAPTURE_EPS) - NOCC))
    wc = np.full((len(NU), len(CAPTURE_EPS)), CAPTURE_SCREENING)
    return 4, b[:, 4, :], c_ov, wc


def solve_capture(linearize=True):
    p, Bp, c_ov, wc = capture_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w, z, _ = qp_energy_cd(p, Bp, CAPTURE_EPS, NOCC, NU, WT, wc=wc,
                               C_ov=c_ov, linearize_on_capture=linearize)
    return w, z, [str(c.message) for c in caught]


def test_capture_by_another_orbitals_pole_is_named_and_recovered():
    """f(w) = w - eps_p - Sigma(w) has a POLE at every eps_q, and Newton near
    one takes ever smaller steps and converges onto it instead of onto a root.
    A pin against q != p is the tell: orbital p's own root has no reason to sit
    1e-4 Ha from a different orbital energy."""
    w, z, notes = solve_capture()
    hit = [n for n in notes if 'captured by the pole' in n]
    assert hit, notes
    assert 'orbital 3' in hit[0] and 'not by a root of its own' in hit[0]
    assert 'LINEARIZED' in hit[0], hit[0]
    # A recovered root must be a quasiparticle, not the pole: at a pole
    # dSigma/dw diverges and the pole strength collapses to zero.
    assert 0.0 < z <= 1.0, z
    assert min(abs(w - CAPTURE_EPS)) > QP_POLE_OFFSET_MIN, w


def test_capture_can_be_refused_instead():
    """The linearized value is a DIFFERENT approximation from the
    self-consistent root, so refusing has to stay available."""
    with pytest.raises(RuntimeError, match='pinned against orbital 3'):
        solve_capture(linearize=False)


def test_a_satellite_root_is_rejected_even_when_it_converged():
    """The failure mode that CONVERGES. f has a zero just to either side of
    every pole and Newton is drawn to them, so a clean convergence is no
    evidence of the right root -- only Z is. Acetaldehyde/cam-B3LYP showed both
    halves of this: on one machine orbital 13 cycled against the pole at
    orbital 12 and refused, on another it converged onto it and returned a
    number, from identical orbital energies."""
    p, Bp, c_ov, wc = capture_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, z, _ = qp_energy_cd(p, Bp, CAPTURE_EPS, NOCC, NU, WT, wc=wc,
                               C_ov=c_ov)
    assert z >= QP_POLE_STRENGTH_MIN, (
        f'a root returned with Z={z:.3f} carries no spectral weight')
    assert any('LINEARIZED' in str(c.message) for c in caught)


def test_a_healthy_root_is_left_alone():
    """The guard must not fire on the ordinary case: no warning, Z near one."""
    w, z, notes = solve(5e-3)
    assert z > 0.5, z
    assert not any('satellite' in n or 'captured' in n for n in notes)


def test_a_pinned_iterate_is_not_accepted_as_converged():
    """The step is measured from the point the guard just pushed to, so a tiny
    step there looks like convergence while the iterate sits exactly `offset`
    from eps_q -- the guard's position, not a root of f. Whatever comes back
    must be a real root: free of the guard, or refused."""
    p_idx, Bp, c_ov, wc = capture_model()
    for floor in (1e-4, 1e-6):
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                w, _, _ = qp_energy_cd(p_idx, Bp, CAPTURE_EPS, NOCC, NU, WT,
                                       wc=wc, C_ov=c_ov,
                                       linearize_on_capture=False,
                                       offset_min=floor)
        except RuntimeError:
            continue                       # refusing is the other valid answer
        # Not sitting on any guard position: a returned root is a root.
        gaps = np.abs(CAPTURE_EPS - w)
        assert gaps.min() > floor, (
            f'floor {floor:.0e}: returned w={w:.8f}, {gaps.min():.2e} from '
            f'orbital {int(np.argmin(gaps))} -- that is the guard, not a root')
