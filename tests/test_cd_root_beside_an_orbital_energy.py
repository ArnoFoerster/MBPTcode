"""The contour root search needs no distance from the orbital energies.

The contour deformation's Sigma is continuous through every orbital energy:
the Lorentzian a pole of G puts on the contour is integrated in closed form
(`cd_integral_weights`). The retired guarded Newton held its iterate
QP_POLE_OFFSET away from them, and a root inside that band deadlocked it,
relaxed it or was refused; the one root finder of `Solvers.qp_equation`
brackets the root on a pole-aware scan and refines inside the bracket, so a
root beside an orbital energy, or on it, is found like any other.

The flat-screening models carry the physical sign, W^c_pp(0) < 0: Sigma^int
beside eps_HOMO is -W^c(0)/2 > 0, which keeps the HOMO's root inside the
window no residue reaches. The pole and satellite cases are a Sigma with a
genuine pole at an orbital energy, s0 + A / (w - eps_q), which a
contour-deformation Sigma does not have, on the same finder
(`solve_qp_equation_scan`).
"""
import warnings

import numpy as np
import pytest

from src.Base.numerics.constants import (QP_CD_NEWTON_TOL, QP_POLE_OFFSET,
                                         QP_WINDOW_Z_MIN)
from src.SingleReference.GW.contour_deformation import qp_energy_cd
from src.SingleReference.GW.real_screening import ResidueRefused
from src.Solvers.qp_equation import (QPNoRoot, SampledEquation,
                                     quasiparticle_weight,
                                     solve_qp_equation_scan)

#: A flat imaginary-axis screening scales the quasiparticle shift: -4e-3 puts
#: the root 5.1e-4 from the pole, inside the retired 1e-3 guard, and -1e-2
#: 1.3e-3 away, outside it.
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
    return w, z, [str(c.message) for c in caught]


@pytest.mark.parametrize('screening', [INSIDE, OUTSIDE])
def test_a_root_beside_its_orbital_energy_is_found_quietly(screening):
    """Inside or outside the old band: the root, its Z near one, no
    warning, and the next scan from it (a displaced geometry) the same."""
    w, z, notes = solve(screening)
    shift = abs(w - EPS[NOCC - 1])
    assert (shift < QP_POLE_OFFSET) == (screening == INSIDE), shift
    assert quasiparticle_weight(z) and not notes, notes
    Bp = np.zeros((3, len(EPS)))
    wc = np.full((len(NU), len(EPS)), screening)
    w_f, z_f, _ = qp_energy_cd(NOCC - 1, Bp, EPS, NOCC, NU, WT, wc=wc,
                               follow=w)
    assert abs(w_f - w) < QP_CD_NEWTON_TOL and abs(z_f - z) < 1e-9


def test_a_root_on_its_orbital_energy_is_not_answered_without_its_residue():
    """No screening puts the root exactly at eps_p, where the contour takes
    the pole of G at half weight, so the residue is needed; without a
    particle-hole block the backend refuses it, and the refusal comes back
    rather than a point the old guard sat on."""
    with pytest.raises(ResidueRefused):
        solve(0.0)


# A self-energy with a genuine pole at a neighbour's orbital energy. Orbital 4
# sits 0.05 Ha above orbital 3 and the constant pulls it down by exactly that;
# with A < 0 there is no root near the pole.
MODEL_EPS = np.array([-0.9, -0.6, -0.35, 0.15, 0.20, 0.8])
MODEL_P, MODEL_Q = 4, 3


def model(s0, A):
    """(f, f') of w = eps_p + s0 + A/(w - eps_q)."""
    def f(w):
        return w - MODEL_EPS[MODEL_P] - s0 - A / (w - MODEL_EPS[MODEL_Q])

    def fprime(w):
        return 1.0 + A / (w - MODEL_EPS[MODEL_Q]) ** 2
    return f, fprime


CAPTURE = dict(s0=-0.05, A=-1e-6)
#: A root 2e-3 above the pole with dSigma/dw = -10 there, Z = 1/11.
SATELLITE = dict(s0=-0.068, A=4e-5)


def every_root(f, fprime, a):
    source = SampledEquation(f, fprime, tol=QP_CD_NEWTON_TOL)
    return [rz for lo, hi in source.windows(a)[:1]
            for br in source.brackets(lo, hi)[0] for rz in source.solve(br)]


def test_a_pole_of_sigma_is_no_root():
    """Newton near a pole takes ever smaller steps and converges onto it (the
    retired guard then fell back to a linearized energy of another
    approximation). Here f = x + 1e-6/x, x = w - eps_q, has no real root at
    all: the scan sees the sign change across the pole, |f| growing under
    refinement tells it from a root, and the search returns a NaN flagged
    'no_root' with a QPNoRoot warning rather than the pole."""
    f, fprime = model(**CAPTURE)
    a = MODEL_EPS[MODEL_P] + CAPTURE['s0']
    out = {}
    with pytest.warns(QPNoRoot, match='no root'):
        w, z = solve_qp_equation_scan(f, fprime, a, out=out)
    assert np.isnan(w) and np.isnan(z) and out['no_root']
    assert every_root(f, fprime, a) == []


def test_a_residue_the_backend_refuses_is_skipped_or_raised():
    """With no particle-hole block every residue is refused
    (`ResidueRefused`): the frontier quasiparticle in the window no residue
    reaches stands; a state whose root needs the residues raises the
    refusal, not another root."""
    w, z, _ = solve(OUTSIDE)
    assert quasiparticle_weight(z)
    Bp = np.zeros((3, len(EPS)))
    wc = np.full((len(NU), len(EPS)), 0.3)          # pushes the root down
    with pytest.raises(ResidueRefused):
        qp_energy_cd(0, Bp, EPS, NOCC, NU, WT, wc=wc)


def test_the_satellite_loses_to_the_quasiparticle():
    """A satellite (Z = 1/11) beside the pole and the quasiparticle further
    out: every root of the window is weighed, the largest Z returned, and the
    satellite only by following it, with its weight saying what it is."""
    f, fprime = model(**SATELLITE)
    a = MODEL_EPS[MODEL_P] + SATELLITE['s0']
    roots = every_root(f, fprime, a)
    best = max((rz for rz in roots if quasiparticle_weight(rz[1], 0.0)),
               key=lambda rz: rz[1])
    w, z = solve_qp_equation_scan(f, fprime, a)
    assert (w, z) == best and z > QP_WINDOW_Z_MIN
    sat = min(roots, key=lambda rz: abs(rz[0] - 0.1525))
    assert 0.0 < sat[1] < QP_WINDOW_Z_MIN
    w_s, z_s = solve_qp_equation_scan(f, fprime, a, follow=0.1525)
    assert abs(w_s - sat[0]) < 1e-10 and not quasiparticle_weight(z_s)


def test_a_misleading_slope_cannot_leave_the_bracket():
    """f(x) = 1.074 (x - x*), x* 3.5e-4 above eps_p, with a slope that
    misleads near eps_p and sent the retired Newton across the pole into a
    cycle: inside a bracket every step that leaves it is a bisection, so the
    root and its Z come back exact."""
    p, root = 2, 3.5e-4
    eps = np.array([-0.69, -0.65, -0.5358, -0.4407, 0.1197, 0.1472])

    def f(w):
        return 1.074 * (w - eps[p] - root)

    def fprime(w):
        x = w - eps[p]
        if 4e-4 < x <= 1.5e-3:
            return f(w) / (1.1 * x)
        if x < 0.0:
            return f(w) / (x - 2e-3)
        return 1.074

    w, z = solve_qp_equation_scan(f, fprime, float(eps[p]))
    assert abs(w - eps[p] - root) < QP_CD_NEWTON_TOL
    assert abs(z - 1.0 / 1.074) < QP_CD_NEWTON_TOL
