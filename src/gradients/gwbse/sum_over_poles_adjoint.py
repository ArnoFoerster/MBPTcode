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

from src.Base.numerics.constants import QP_SUBSPACE_POLE_CHUNK, SOP_FIT_RCOND
from src.Base.utils.parallel.mpi_grid import current_comm
from src.SingleReference.GW.sum_over_poles import (
    PoleSlice, denominators, large_group, lockstep_group, pole_pseudoinverse,
    subspace_self_energy_jacobian)


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


def subspace_hamiltonian_backward(h, amp, d, h_bar):
    """(e0_bar, amp_bar, d_bar) of <h_bar, H> through the fixed point
    H = e0 + Sigma^c[H] of `subspace_hamiltonian`.

    The implicit adjoint: (1 - J^T) Lambda = h_bar, J the derivative of
    Sigma^c[H] = 1/2 sum_x [R_x A_x + A_x R_x], R_x = (H - d_x)^-1, on H
    (`subspace_self_energy_jacobian`). Then e0_bar = Lambda, A_x's adjoint is
    (R_x Lambda + Lambda R_x)/2 and d_x's tr(Lambda R_x^2 A_x). Everything is
    taken in H's eigenbasis, where R_x is diagonal. A large group takes
    rank 0's inputs and eigenbasis on every rank, as its forward solve does.
    """
    n = len(h)
    if large_group(n):
        comm = current_comm()
        h, amp, d, h_bar = lockstep_group(comm, h, amp, np.ravel(d), h_bar)
        poles = PoleSlice(h, amp, d, comm)
        e, u = poles.e, poles.u
    else:
        e, u = np.linalg.eigh(h)
    d = np.asarray(d, float).ravel()
    r = 1.0 / (e[None, :] - d[:, None])
    hb = u.T @ (0.5 * (h_bar + np.asarray(h_bar).T)) @ u
    if large_group(n):
        # no n^2 x n^2 Jacobian: GMRES on the operator, the amplitudes
        # transformed a chunk of poles at a time
        lam = poles.solve(hb, transpose=True)
        lam = 0.5 * (lam + lam.T)
        amp_bar = np.empty_like(amp)
        d_bar = np.empty(len(d))
        for a in range(0, len(d), QP_SUBSPACE_POLE_CHUNK):
            part = slice(a, a + QP_SUBSPACE_POLE_CHUNK)
            rc = r[part]
            amp_bar[part] = u @ (0.5 * lam[None] * (rc[:, :, None] + rc[:, None, :])) @ u.T
            d_bar[part] = np.einsum('kl,xl,xlk->x', lam, rc ** 2,
                                    u.T @ amp[part] @ u)
        return u @ lam @ u.T, amp_bar, d_bar
    at = u.T @ amp @ u
    jac = subspace_self_energy_jacobian(r, at)
    lam = np.linalg.solve(np.eye(n * n) - jac.T, hb.ravel()).reshape(n, n)
    lam = 0.5 * (lam + lam.T)
    amp_bar = u @ (0.5 * lam[None] * (r[:, :, None] + r[:, None, :])) @ u.T
    d_bar = np.einsum('kl,xl,xlk->x', lam, r ** 2, at)
    return u @ lam @ u.T, amp_bar, d_bar


def subspace_residues_backward(amp_bar, poles, nu_points, pairs):
    """{(a, b): wc_ab_bar} for `subspace_residues`' pairs a <= b, each
    (nfreq, norb): the transpose of A = F^+ wc, an off-diagonal pair taking
    both of its symmetric elements."""
    n_poles = len(poles)
    pinv = pole_pseudoinverse(nu_points, poles)
    out = {}
    for a, b in pairs:
        g = amp_bar[:, a, b] if a == b else amp_bar[:, a, b] + amp_bar[:, b, a]
        out[(a, b)] = pinv.T @ g.reshape(-1, n_poles).T
    return out
