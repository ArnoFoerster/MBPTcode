"""Riccati form of a pair-channel RPA problem.

A pair channel is the metric eigenproblem

    [[A, B], [B^+, D]] (X; Y) = Omega diag(1, -1) (X; Y),

with A (nA x nA) and D (nD x nD) Hermitian and B (nA x nD). Its nA
positive-norm solutions, X^+ X - Y^+ Y = 1, define the amplitudes
T = Y X^{-1} (nD x nA), which solve the Riccati equation

    B^+ + D T + T A + T B T = 0

without the eigenvectors. The particle-hole channel (A, B, A) gives the
ring-CCD amplitudes, the particle-particle channel (A_pp, B_pp, C_pp) the
ladder-CCD ones (Scuseria, Henderson, Bulik, J. Chem. Phys. 139, 104113
(2013)). From T, the channel quantities used by Faddeev-ADC(3) follow as

    X^{-+} Omega X^{-1} = A + B T + T^+ B^+ + T^+ D T,
    (X X^+)^{-1}        = 1 - T^+ T.

solve_riccati is the dense Newton solve of any channel. The particle-hole
channel (A, B, A) has the closed form of GW.quasi_boson.RPA instead
(ring_channel). The closed-shell ladder channel at scale is
adc_r_faddeev.solve_ladder_amplitudes_restricted, DF-streamed.
"""
import numpy as np
import scipy.linalg as la

from src.Base.constants import RICCATI_CONV_TOL, RICCATI_MAX_ITER
from src.SingleReference.GW.quasi_boson import RPA


def riccati_residual(A, B, D, T):
    """B^+ + D T + T A + T B T for amplitudes T of shape (nD, nA)."""
    return B.conj().T + D @ T + T @ A + T @ (B @ T)


def solve_riccati(A, B, D, T0=None, conv_tol=RICCATI_CONV_TOL,
                  max_iter=RICCATI_MAX_ITER):
    """Newton solve of B^+ + D T + T A + T B T = 0.

    Each step solves the Sylvester equation of the linearized residual,
    (D + T B) dT + dT (A + B T) = -R, which needs the spectra of D + T B and
    -(A + B T) to be disjoint: a gap between the positive- and negative-norm
    solutions of the channel. From T0 = 0 the first step gives the
    first-order amplitudes, and the iteration follows the solution that
    connects to them.

    Parameters
    ----------
    A : ndarray, shape (nA, nA)
    B : ndarray, shape (nA, nD)
    D : ndarray, shape (nD, nD)
    T0 : ndarray, shape (nD, nA), optional
        Starting amplitudes; zero by default.
    conv_tol : float
        Converged when max|residual| < conv_tol.
    max_iter : int

    Returns
    -------
    T : ndarray, shape (nD, nA)
    info : dict
        'niter' and the final 'residual' (max abs).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    D = np.asarray(D)
    nA, nD = A.shape[0], D.shape[0]
    if B.shape != (nA, nD):
        raise ValueError(f"B has shape {B.shape}; expected ({nA}, {nD})")
    dtype = np.result_type(A, B, D)
    T = np.zeros((nD, nA), dtype=dtype) if T0 is None else np.array(T0, dtype=dtype)
    for it in range(max_iter + 1):
        R = riccati_residual(A, B, D, T)
        rmax = float(np.max(np.abs(R))) if R.size else 0.0
        if rmax < conv_tol:
            return T, {'niter': it, 'residual': rmax}
        if it == max_iter:
            break
        T = T + la.solve_sylvester(D + T @ B, A + B @ T, -R)
    raise RuntimeError(
        f"solve_riccati: not converged after {max_iter} Newton steps "
        f"(max|R| = {rmax:.3e} > {conv_tol:.1e}); the channel may have no gap "
        "between its positive- and negative-norm solutions")


def require_stable_channel(A, B, what):
    """Raise unless the particle-hole channel [[A, B], [B, A]] is positive
    definite, i.e. A + B and A - B are.

    Otherwise the reference is unstable in that channel (a triplet instability
    of RHF, or an RHF that is a saddle point): some phonon energy is imaginary
    or has the wrong norm, T = Y X^{-1} does not exist as a stabilizing
    solution, and a Casida solve that clips omega^2 would return a wrong
    channel without saying so."""
    for M, name in ((A + B, 'A+B'), (A - B, 'A-B')):
        try:
            la.cholesky(M, lower=True)
        except la.LinAlgError:
            w = la.eigvalsh(M, subset_by_index=[0, 0])[0]
            raise ValueError(
                f"{what}: the reference is unstable in this channel ({name} has "
                f"eigenvalue {w:+.4e} Ha); its RPA phonons are not real and the "
                "Faddeev-ADC(3) pair channel does not exist") from None


def amplitudes_from_phonons(X, Y):
    """T = Y X^{-1} from the positive-norm eigenvectors (X square)."""
    return la.solve(X.T, Y.T).T


def channel_corrections_from_phonons(A, omega, X, Y):
    """(T, dH, dN) of a channel from its positive-norm phonons.

    dH = X^{-+} Omega X^{-1} - A and dN = (X X^+)^{-1} - 1: what the RPA
    phonons add to the Tamm--Dancoff pair Hamiltonian A and to the unit
    metric."""
    Xinv = la.inv(X)
    W = (Xinv.conj().T * omega[None, :]) @ Xinv
    W = 0.5 * (W + W.conj().T)
    XXinv = Xinv.conj().T @ Xinv
    XXinv = 0.5 * (XXinv + XXinv.conj().T)
    dN = XXinv - np.eye(X.shape[0])
    return amplitudes_from_phonons(X, Y), W - A, dN


def ring_channel(A, B):
    """(T, dH, dN) of the particle-hole channel (A, B, A), real symmetric A
    and B, in closed form: GW.quasi_boson.RPA gives T = tanh(t) and
    cosh(t) from one eigendecomposition of t, so dN = -T^2 and
    dH = cosh(t)^{-1} Abar cosh(t)^{-1} - A. Raises for an unstable channel."""
    r = RPA(A, B)
    T = (r.Vt * np.tanh(r.wt)[None, :]) @ r.Vt.T
    ci = (r.Vt / np.cosh(r.wt)[None, :]) @ r.Vt.T
    dH = ci @ r.Abar @ ci - A
    return T, 0.5 * (dH + dH.T), -(T @ T)


def channel_corrections_from_amplitudes(B, D, T):
    """(dH, dN) of a channel from its Riccati amplitudes.

    dH = B T + T^+ B^+ + T^+ D T and dN = -T^+ T; identical to
    channel_corrections_from_phonons for T = Y X^{-1}."""
    BT = B @ T
    dH = BT + BT.conj().T + T.conj().T @ (D @ T)
    return 0.5 * (dH + dH.conj().T), -(T.conj().T @ T)
