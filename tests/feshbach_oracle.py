"""The dense Feshbach reference of the fragment partition, for tests.

`FeshbachOracle` diagonalizes the pencil of the Q space outright, so the
self-energy Sigma(Omega), the roots of A_eff(Omega) c = Omega c and their
weights Z are exact for a small operator: the reference the matrix-free
partition (`src.fragments.partition`) is checked against
(tests/test_fragment_diabatic.py, tests/test_fragment_adc2.py,
tests/test_fragment_triplet_transfer.py).
"""
import numpy as np
from scipy.linalg import eigh, null_space

from src.Base.numerics.constants import FESHBACH_ROOT_MAX_ITER, FESHBACH_ROOT_TOL
from src.fragments.partition import flag_numerics


class FeshbachOracle:
    """Dense reference: Sigma(Omega), roots of A_eff(Omega) c = Omega c, Z.
    The oracle the fragment-diabatic solves are checked against.

    K (n, n), a P basis (n, n_p) with P^T S P = 1, and the metric S as its
    diagonal (`metric`, default ones: the Tamm-Dancoff case) in the same basis;
    Q is the complement orthogonal in the metric. Small systems only -- it
    diagonalizes the pencil on Q: with V its eigenvectors normalized to
    V^T K_QQ V = 1 and V^T S_QQ V = diag(mu), mu = 1 / lambda,
    (K_QQ - Omega S_QQ)^-1 = V diag(1 / (1 - Omega mu)) V^T.
    """

    def __init__(self, k, p, metric=None):
        k = 0.5 * (np.asarray(k, float) + np.asarray(k, float).T)
        p = np.asarray(p, float)
        metric = np.ones(len(k)) if metric is None else np.asarray(metric, float)
        self.q = null_space((metric[:, None] * p).T)
        self.k, self.p = k, p
        self.a_pp = p.T @ k @ p
        kq = self.q.T @ k @ self.q
        mq = self.q.T @ (metric[:, None] * self.q)
        self.mu, v = eigh(0.5 * (mq + mq.T), 0.5 * (kq + kq.T))
        self.c = p.T @ k @ self.q @ v              # couplings to Q eigenstates

    def sigma(self, omega):
        return -(self.c / (1.0 - omega * self.mu)) @ self.c.T

    def dsigma(self, omega):
        return -(self.c * (self.mu / (1.0 - omega * self.mu) ** 2)) @ self.c.T

    def root(self, guess, tol=FESHBACH_ROOT_TOL,
             maxiter=FESHBACH_ROOT_MAX_ITER,
             flags=None):
        """(Omega, c, Z) of A_eff(Omega) c = Omega c by Newton, near `guess`;
        a Newton that has not stepped below `tol` after `maxiter` steps keeps
        its last iterate and is flagged 'feshbach_root_not_converged'
        (`flag_numerics`)."""
        om = float(guess)
        for _ in range(maxiter):
            w, v = np.linalg.eigh(self.a_pp + self.sigma(om))
            k = int(np.argmin(np.abs(w - om)))
            c = v[:, k]
            slope = float(c @ self.dsigma(om) @ c)
            step = (w[k] - om) / (1.0 - slope)
            om += step
            if abs(step) < tol:
                break
        else:
            flag_numerics(flags, 'feshbach_root_not_converged',
                          f'Feshbach root: Newton took {maxiter} steps without '
                          f'its step falling below {tol}; the last iterate '
                          f'{om:.10f} Ha is returned', omega=float(om))
        z = 1.0 / (1.0 - float(c @ self.dsigma(om) @ c))
        return om, c, z
