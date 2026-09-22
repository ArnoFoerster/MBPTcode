"""Reverse mode of the sum-over-poles quasiparticle self-energy.

The forward half is `SingleReference.GW.sum_over_poles`. With the auxiliary
poles Om_m FROZEN after the fit, the amplitudes are a fixed LINEAR map of the
imaginary-axis data, A = F^+ wc, so the adjoint of the whole route is that map
transposed -- no unrolled vector fitting, no residue set, and no screening at a
real frequency anywhere in the reverse pass. What comes back is an adjoint on
wc, which is the same object the contour deformation's integral term already
consumes (`contour_deformation_adjoint.integral_term_backward`), so the chain
down to Bp, chi0 or C_ov is the one that route already owns.
"""
import numpy as np

from src.Base.constants import SOP_FIT_RCOND
from src.SingleReference.GW.real_screening import ov_energies, screening_aux
from src.SingleReference.GW.sum_over_poles import (denominators,
                                                   pole_pseudoinverse)
from src.SingleReference.base import get_occ_virt_indices
from src.gradients.contour_deformation_adjoint import (integral_term_backward,
                                                       screening_chain)


def sigma_sop_backward(omega, amplitudes, poles, eps, nocc, nu_points,
                       sigma_bar=1.0, rcond=None):
    """(eps_bar, wc_bar, omega_bar) of sigma_bar * Sigma^c_pp.

    With the poles frozen, A = F^+ wc, so the adjoint on the imaginary-axis
    data is one transpose:

        wc_bar[k, q] = sigma_bar sum_m (F^+)[m, k] / D[q, m] ,

    and wc is exactly what the integral term's chain already consumes
    (`contour_deformation_adjoint.integral_term_backward`), so nothing new is
    needed downstream to reach Bp, chi0 or C_ov. eps enters only through D, and
    omega is held fixed here as it is in `sigma_cd_backward` -- the
    quasiparticle chain rule multiplies by Z outside.
    """
    den = denominators(omega, poles, eps, nocc)
    inv2 = 1.0 / den ** 2
    amp = np.asarray(amplitudes, float).T                      # (norb, M)
    eps_bar = sigma_bar * np.sum(amp * inv2, axis=1)
    omega_bar = -sigma_bar * float(np.sum(amp * inv2))
    f_pinv = pole_pseudoinverse(nu_points, poles,
                                SOP_FIT_RCOND if rcond is None else rcond)
    wc_bar = sigma_bar * (f_pinv.T @ (1.0 / den).T)            # (nfreq, norb)
    return eps_bar, wc_bar, omega_bar


def sop_partials(omega, amplitudes, poles, eps, nocc, nu_points, Bp, C_ov,
                 sigma_bar=1.0):
    """(eps_bar, Bp_bar, Cov_bar) of sigma_bar * Sigma^c_pp, all the way down.

    `sigma_sop_backward` stops at wc, and wc enters the screening exactly as
    the integral term's own weights do -- both terms are
    sum_q weight_q Bp[:,q]^T [W(i.nu_k) - 1] Bp[:,q] -- so this is
    `sigma_cd_backward`'s explicit branch with weights = wc_bar[k] in place of
    -w_k g_k/pi, and with no residue block at all. That absence is the whole
    point: there is no frozen pole set in the reverse pass, no dW/domega' at a
    real frequency, and no backend.

    The O(N^4) screening rebuild here is the REFERENCE route, the same one
    `sigma_cd_backward` takes when handed C_ov. A cubic caller passes chi0
    adjoints instead; the wc_bar this consumes is identical either way.
    """
    eps_bar, wc_bar, _ = sigma_sop_backward(omega, amplitudes, poles, eps, nocc,
                                            nu_points, sigma_bar)
    d = ov_energies(eps, nocc)
    Bp_bar = np.zeros_like(Bp)
    Cov_bar = np.zeros_like(C_ov)
    d_bar = np.zeros_like(d)
    for k, nu in enumerate(nu_points):
        wtb = screening_aux(C_ov, d, nu, True)[1] @ Bp
        bb, _, _ = integral_term_backward(Bp, wtb, wc_bar[k], want_chi0=False)
        Bp_bar += bb
        cvb, db, _ = screening_chain(wtb, wc_bar[k], C_ov, d, nu, True)
        Cov_bar += cvb
        d_bar += db
    occ, virt = get_occ_virt_indices(eps, nocc)
    d4 = d_bar.reshape(len(occ), len(virt))
    eps_bar[virt] += d4.sum(axis=0)
    eps_bar[occ] -= d4.sum(axis=1)
    return eps_bar, Bp_bar, Cov_bar
