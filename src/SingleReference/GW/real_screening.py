"""The screened interaction at a REAL frequency, by two routes.

    chi0(w) = 2 (C_ov f(d, w)) C_ov^T,    W(w) = [1 - chi0(w)]^-1

with C_ov = B[:, occ, virt] reshaped to (naux, nocc*nvirt) and d the pair
energies eps_a - eps_i. One frequency costs O(naux^2 nocc nvirt) and holds
C_ov: the N^4 route (`ExplicitRealScreening`), and the only one above the
particle-hole gap, where chi0 has real poles and no imaginary-time form.

Below the gap the same W comes out of the imaginary-time polarizability at
O(N^3) (`LaplaceRealScreening`). The particle-hole factor at a real frequency is

    -2d/(d^2 - w^2) = -int_0^inf 2 cosh(w tau) e^{-d tau} dtau   (d > w)

so chi0(w) = int_0^inf 2 cosh(w tau) proj(tau) dtau: the imaginary-axis form
with cos -> cosh, on the SAME proj(tau) the contour's integral term already
built, and no O(N^4) object anywhere. The transform exists only while the
grid's bare Laplace quadrature still carries every d -/+ w, which is what
`laplace_representation_error` measures and what fails at the gap.

This is what a contour-deformation residue evaluates: `GW.contour_deformation`
sweeps a pole of G at eps_q and asks for W(|eps_q - omega|) there. A residue
frequency landing on a particle-hole transition d makes W and its frequency
derivative arbitrarily large -- that is a real pole of chi0, not a numerical
accident, and `residue_pole_distance` reports the distance to the nearest one.

Both classes serve one protocol -- apply(freq, b) = W(freq) b and
slope(freq, y) = y^T [dchi0/dfreq] y -- and neither records anything;
`gradients.contour_deformation_adjoint` subclasses each with the reverse half,
push/adjoints.

The spin factor 2 is the restricted-singlet convention of
`LinearResponse.imaginary_frequency.solve_rpa_screening_df`, which forms the
same chi0 from an assembled solver object over a whole frequency grid; this
module takes the factors themselves, one frequency at a time, and is not
otherwise a second route to a different number.
"""
import numpy as np

from src.Base.constants import LAPLACE_SCREENING_TOL
from src.SingleReference.LinearResponse.imaginary_frequency import \
    frequency_factor
from src.SingleReference.LinearResponse.space_time import \
    laplace_representation_error
from src.SingleReference.base import get_occ_virt_indices


class ExplicitRealScreening:
    """W at a real frequency from the particle-hole block C_ov, explicitly.

    The reference backend of the contour deformation's residue term, serving
    apply(freq, b) = W(freq) b and slope(freq, y) = y^T [dchi0/dfreq] y. It is
    the route the cubic backend is checked against, and the only route above
    the particle-hole gap. eta broadens the particle-hole poles -- biased, off
    by default, see `imaginary_frequency.frequency_factor`.

    `gradients.contour_deformation_adjoint.ExplicitRealScreeningAdjoint` adds
    the reverse half of the protocol, push/adjoints; nothing here records
    anything.
    """

    def __init__(self, C_ov, eps, nocc, eta=0.0):
        self.C_ov = C_ov
        self.eps = np.asarray(eps, float)
        self.nocc = nocc
        self.eta = eta
        self.d = ov_energies(self.eps, nocc)

    def apply(self, freq, b):
        return screening_aux(self.C_ov, self.d, freq, False, self.eta)[1] @ b

    def slope(self, freq, y):
        # d/dfreq [b^T (W - I) b] = y^T (dchi0/dfreq) y,  dchi0 = 2 C df C^T
        z = self.C_ov.T @ y
        return 2.0 * float(frequency_factor(self.d, freq, False, self.eta,
                                            slopes=True)[2] @ z ** 2)


class LaplaceRealScreening:
    """W at a real frequency below the particle-hole gap, from proj(tau): cubic.

    Serves the residue term of `contour_deformation` through the same protocol
    as `ExplicitRealScreening` at one (ntau, naux^2) transform and one naux^3
    solve per call, instead of an O(naux^2 nocc nvirt) chi0 build from C_ov.

    Validity is checked on every call against the pair energies themselves
    (`laplace_representation_error` < tol), which fails exactly when d_min -
    freq drops below the grid's e_min. Above the gap chi0 has real poles and no
    imaginary-time form: that regime belongs to the explicit backend, and this
    one REFUSES rather than integrate a fit it has failed.

    `gradients.contour_deformation_adjoint.LaplaceRealScreeningAdjoint` adds
    the reverse half, push/adjoints, which lands the residues' adjoint on
    proj(tau) so they ride the same `polarizability_backward` sweep as the
    integral term.
    """

    def __init__(self, proj_tau, grid, eps, nocc, tol=LAPLACE_SCREENING_TOL):
        self.proj_tau = proj_tau
        self.grid = grid
        self.eps = np.asarray(eps, float)
        self.nocc = nocc
        self.tol = tol
        self.eye = np.eye(proj_tau.shape[-1])

    def representation_error(self, freq):
        return laplace_representation_error(self.grid, self.eps, self.nocc, freq)

    def _weights(self, freq):
        err = self.representation_error(freq)
        if not err < self.tol:
            occ, virt = get_occ_virt_indices(self.eps, self.nocc)
            gap = self.eps[virt].min() - self.eps[occ].max()
            raise ValueError(
                f"real frequency {freq:.4f} Ha is not carried by the tau grid: "
                f"bare-quadrature error {err:.1e} against tol {self.tol:.0e}; "
                f"gap - freq = {gap - freq:.4f} Ha, the grid's e_min is "
                f"{self.grid.meta.get('e_min', float('nan')):.4f}. Build the grid "
                f"with e_min <= gap - freq, or use the explicit backend.")
        return real_frequency_weights(self.grid, freq)

    def apply(self, freq, b):
        c, _ = self._weights(freq)
        chi0 = np.tensordot(c, self.proj_tau, axes=(0, 0))
        return np.linalg.solve(self.eye - chi0, b)

    def slope(self, freq, y):
        _, dc = self._weights(freq)
        return float(y @ np.tensordot(dc, self.proj_tau, axes=(0, 0)) @ y)


def ov_energies(eps, nocc):
    """d_(i,a) = eps_a - eps_i on the flattened pair index of C_ov."""
    occ, virt = get_occ_virt_indices(eps, nocc)
    return (eps[virt][None, :] - eps[occ][:, None]).ravel()


def screening_aux(C_ov, d, omega, imaginary, eta=0.0):
    """(W - I, W) in the auxiliary basis: W = [1 - chi0(omega)]^-1.

    chi0 = 2 (C_ov f) C_ov^T is the restricted-singlet convention of
    `LinearResponse.imaginary_frequency.solve_rpa_screening_df`; the spin factor
    lives here and nowhere else. Serves both axes, since only the frequency
    factor knows which one it is on.
    """
    f = frequency_factor(d, omega, imaginary, eta)
    chi0 = 2.0 * (C_ov * f) @ C_ov.T
    W = np.linalg.inv(np.eye(C_ov.shape[0]) - chi0)
    return W - np.eye(C_ov.shape[0]), W


def real_frequency_weights(grid, omega):
    """(c, dc): chi0(omega) = sum_k c_k proj(tau_k) at a REAL omega below the
    particle-hole gap, and its exact omega derivative.

    On the grid's bare Laplace quadrature -- tau_weights, which carry
    sum_k w_k e^{-y tau_k} = 1/y over [e_min, e_max] -- the cosh transform of
    the module docstring is c_k = 2 w_k cosh(w tau_k), exact to the quadrature's
    accuracy as long as every d - w lies inside the fitted range: the grid's
    e_min must be at most gap - w. It is the caller's grid, frozen per geometry
    like the residue set.

    The omega dependence is closed-form, so dc_k = 2 w_k tau_k sinh(w tau_k) is
    the exact derivative of the function actually represented; weights refitted
    per frequency would make the slope the derivative of a pseudo-inverse and
    disagree with the value at the level a gradient gate sees.

    Measured on water/cc-pVDZ: with e_min lowered by 0.5 Ha at ntau = 18 the
    residue value matches the explicit chi0 to 1e-12 for w up to 0.3 Ha; on the
    unextended grid it is 1e-6 .. 5e-5. Raising ntau to 24 buys the same thing
    without touching e_min, because below the narrowest tabulated range the grid
    stretches and its fitted interval reaches further down than the gap
    (`laplace_representation_error` 6.8e-03 -> 1.5e-11 at w = 0.3 Ha).
    """
    t = grid.tau_points
    w = 2.0 * grid.tau_weights
    return w * np.cosh(omega * t), w * t * np.sinh(omega * t)
