import numpy as np
import scipy.linalg as la
from src.Base.utils.linear_algebra.diagonalization import eigh_symmetric
from src.Base.numerics.constants import (BSE_DENSE_MAX_GB, BYTES_PER_GB,
                                         CASIDA_NUMERICAL_EPS)
from src.Base.step_failed import StepFailed


class CasidaInstability(StepFailed, ValueError):
    """The Casida problem has (A - B), (A + B) or omega^2 not positive
    definite: the reference is unstable there and the omega^2 reduction is
    invalid. A ValueError as the dense BSE's instability always was, and a
    `StepFailed` as the Davidson's `CasidaBreakdown` is, so a geometry walk
    answers it with a shorter step."""


def _refuse_nonpositive(omega2):
    """Raise `CasidaInstability` where an omega^2 is not positive.

    `eigh_symmetric` returns the eigenvalues whole on every rank of a
    distributed solve, after releasing the solver, so every rank raises
    together.
    """
    if omega2.min() > 0.0:
        return
    raise CasidaInstability(
        f'Casida problem has omega^2 = {omega2.min():.3e} <= 0 Ha^2: the '
        f'mean-field reference is unstable and the omega^2 reduction is '
        f'invalid there.')


def _combine_in_place(XpY, XmY):
    """Recover X, Y from X+Y and X-Y; Y overwrites XmY's buffer.

    Parameters
    ----------
    XpY : ndarray, shape (n_ov, n_state)
        X + Y.
    XmY : ndarray, shape (n_ov, n_state)
        X - Y. Overwritten in place with Y.

    Returns
    -------
    X : ndarray, shape (n_ov, n_state)
        A new array, (XpY + XmY) / 2.
    Y : ndarray, shape (n_ov, n_state)
        XmY's buffer, holding (XpY - XmY) / 2.
    """
    X = XpY + XmY
    X *= 0.5
    np.subtract(XpY, XmY, out=XmY)
    XmY *= 0.5
    return X, XmY


def _frequencies(omega2, eta):
    """omega from the omega^2 of the Hermitian form, refused where one is not
    positive (`_refuse_nonpositive`) and floored at eta."""
    _refuse_nonpositive(omega2)
    return np.sqrt(np.clip(omega2, eta**2, None))


def casida_eigensolve(halves, factor='amb', lowest=None, threshold=5000,
                      eta=CASIDA_NUMERICAL_EPS):
    """(omega, X, Y, Z, is_distributed): THE dense Casida eigensolve,
    [[A, B], [B, A]] [X, Y] = omega [X, -Y] with <X|X> - <Y|Y> = 1, from its
    Hermitian halves halves = [A + B, A - B], real or complex Hermitian. The
    call empties the list and overwrites the halves, so where the list holds
    the last reference to each the solve frees them as it goes (a dense
    Casida problem is memory-bound: one n x n array more is the limit). One half is factored, L L^H, the other turned into the
    Hermitian form H = L^H (other) L, H Z = Z omega^2 (`eigh_symmetric`,
    distributed above `threshold`):

        factor 'amb', A - B = L L^H:  X + Y = L Z omega^-1/2,
                                      X - Y = L^-H Z omega^1/2;
        factor 'apb', A + B = L L^H:  X - Y = L Z omega^-1/2,
                                      X + Y = L^-H Z omega^1/2.

    Under 'amb' `minus` may be the diagonal of a diagonal A - B, a 1-D array,
    and L its square root. A factored half that is not positive definite, or
    an omega^2 <= 0, raises `CasidaInstability`: the reference is unstable
    there and the omega^2 reduction invalid. omega is floored at eta.
    lowest: how many of the lowest roots are kept (None: all). Distributed, Z, X and Y are
    None on every rank but the one holding Z whole (`eigh_symmetric`).
    """
    if factor not in ('amb', 'apb'):
        raise ValueError(f"factor={factor!r}: 'amb' or 'apb'")
    plus, minus = halves
    halves.clear()
    keep = slice(None) if lowest is None else slice(0, int(lowest))
    if factor == 'amb' and np.ndim(minus) == 1:
        d = np.asarray(minus).real
        if d.min() <= 0.0:
            raise CasidaInstability(
                f'Casida problem: (A - B) is diagonal with min {d.min():.3e} '
                f'<= 0 Ha, so it is not positive definite; the mean-field '
                f'reference is unstable and the omega^2 reduction is invalid '
                f'there.')
        # a floor for a numerically zero transition, not a sign repair
        root = np.sqrt(np.clip(d, eta, None))
        inv_root = 1.0 / root
        del d, minus
        # H = (A-B)^{1/2} (A+B) (A-B)^{1/2}, formed in plus's buffer
        H = plus
        H *= root[:, None]
        H *= root[None, :]
        del plus
        omega2, Z, distributed = eigh_symmetric(H, threshold=threshold)
        del H
        omega = _frequencies(omega2, eta)[keep]
        if Z is None:
            return omega, None, None, None, True
        Z = Z[:, keep]
        sqrt_omega = np.sqrt(omega)
        XpY = Z * root[:, None]
        XpY /= sqrt_omega[None, :]
        XmY = Z * inv_root[:, None]
        XmY *= sqrt_omega[None, :]
    else:
        factored, other = (minus, plus) if factor == 'amb' else (plus, minus)
        del plus, minus
        try:
            L = la.cholesky(factored, lower=True)
        except la.LinAlgError as exc:
            raise CasidaInstability(
                f'Casida problem: (A {"-" if factor == "amb" else "+"} B) is '
                f'not positive definite; the mean-field reference is unstable '
                f'and the omega^2 reduction is invalid there.') from exc
        del factored
        H = L.conj().T @ (other @ L)
        del other
        omega2, Z, distributed = eigh_symmetric(H, threshold=threshold)
        del H
        omega = _frequencies(omega2, eta)[keep]
        if Z is None:
            return omega, None, None, None, True
        Z = Z[:, keep]
        sqrt_omega = np.sqrt(omega)
        far = la.solve_triangular(L, Z, lower=True, trans='C')
        far *= sqrt_omega[None, :]
        near = L @ Z
        near /= sqrt_omega[None, :]
        del L
        XpY, XmY = (near, far) if factor == 'amb' else (far, near)
    X, Y = _combine_in_place(XpY, XmY)
    return omega, X, Y, Z, distributed


class CasidaResult(tuple):
    """3-tuple (omega, X, Y) that also carries an is_distributed flag for MPI post-processing."""
    def __new__(cls, omega, X, Y, is_distributed=False):
        return super().__new__(cls, (omega, X, Y))
    def __init__(self, omega, X, Y, is_distributed=False):
        self.is_distributed = is_distributed

class CasidaSolver:
    """Solves [[A,B],[B,A]] [X,Y] = omega [X,-Y] via local scipy.linalg, or parallel ELPA/MPI for larger systems.

    Handles both the real-symmetric case (molecular RPA/TDDFT/BSE, q=0) and the
    complex-Hermitian case (Sander/Maggio/Kresse full BSE at finite momentum
    transfer q, PRB 92, 045209). The algebra is identical; only the transpose
    convention differs (Hermitian adjoint vs. transpose), and all conventions
    below are Hermitian, which reduces to the real case for real inputs.
    """
    def __init__(self, A, B, eta=CASIDA_NUMERICAL_EPS, keep_intermediates=False):
        """Build the solver from one pair of Casida matrices.

        Parameters
        ----------
        A : ndarray, shape (n, n)
            The (A) Casida block, Hermitian.
        B : ndarray, shape (n, n)
            The (B) Casida block.
        eta : float
            Numerical floor kept away from zero, e.g. when clipping
            eigenvalues before a square root.
        keep_intermediates : bool
            The instance is single-use by default: solve() releases A and B
            unless keep_intermediates=True, and a second solve() call on a
            released instance raises RuntimeError. Setting it True also
            keeps the eigenvectors Z of the transformed problem on the
            instance, and allows solve() to be called again. The cost is two
            more n x n arrays held through a non-TDA solve, one more for TDA.
        """
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.eta = eta
        self.ndim = self.A.shape[0]
        self.is_complex = np.iscomplexobj(self.A) or np.iscomplexobj(self.B)
        self.is_distributed = False
        # False: solve() releases A and B and leaves Z unset. True: solve()
        # keeps A, B and the eigenvectors Z of the transformed problem on the
        # instance; see this method's docstring for the cost.
        self.keep_intermediates = keep_intermediates
        self.Z = None

    def solve(self, threshold=5000, tda=False):
        """Cholesky factorization where possible; switches to parallel ELPA above `threshold` dim when MPI is available.

        tda=True ignores B (Tamm-Dancoff): plain Hermitian diagonalization of A,
        omega = eig(A), X = eigenvectors (X^T X = 1 normalization), Y = 0.
        """
        if self.A is None:
            raise RuntimeError(
                "solve() already released A and B on this instance; construct "
                "with keep_intermediates=True to solve() more than once.")
        if tda:
            A = self.A
            if not self.keep_intermediates:
                self.A = None
                self.B = None
            omega, Z_res, is_distributed = eigh_symmetric(A, threshold=threshold)
            del A
            if Z_res is None:
                self.is_distributed = True
                return CasidaResult(omega, None, None, True)
            X = Z_res
            # calloc'd zeros stay unresident until written; zeros_like writes them.
            Y = np.zeros(Z_res.shape, dtype=Z_res.dtype,
                         order='F' if Z_res.flags.f_contiguous else 'C')
            if self.keep_intermediates:
                self.Z = Z_res
            self.is_distributed = is_distributed
            return CasidaResult(omega, X, Y, is_distributed)

        A, B = self.A, self.B
        if not self.keep_intermediates:
            self.A = None
            self.B = None
        ApB = A + B
        AmB = A - B
        del A, B

        # Check if A-B is diagonal: only check off-diagonal norm.
        # Computed from the Frobenius norms rather than by forming
        # AmB - diag(diag(AmB)), which allocates a whole extra n_ov x n_ov array
        # purely to be normed and discarded, where the solve is already
        # memory-bound. Identical value to 1e-8 relative.
        diag_AmB_full = np.diag(AmB)
        offdiag_sq = (np.linalg.norm(AmB)**2 - np.linalg.norm(diag_AmB_full)**2)
        offdiag_norm = np.sqrt(max(offdiag_sq, 0.0))
        is_AmB_diag = offdiag_norm < self.eta * self.ndim
        # The view keeps AmB's buffer alive for as long as its name lives.
        del diag_AmB_full
        if is_AmB_diag:
            # A-B is Hermitian, so its diagonal is real (orbital energy
            # differences): its square root is the factor
            minus, factor = np.diag(AmB).real.copy(), 'amb'
            del AmB
        else:
            minus, factor = AmB, 'apb'
            del AmB
        halves = [ApB, minus]
        del ApB, minus
        omega, X, Y, Z_res, is_distributed = casida_eigensolve(
            halves, factor, threshold=threshold, eta=self.eta)
        if Z_res is None:
            # Z came back whole on rank 0; the other ranks are done.
            self.is_distributed = True
            return CasidaResult(omega, None, None, True)
        if self.keep_intermediates:
            self.Z = Z_res
        self.is_distributed = is_distributed
        return CasidaResult(omega, X, Y, is_distributed)


def solver_choice(n_ov, tda=False):
    """'dense' or 'davidson' for a particle-hole space of n_ov pairs.

    Dense while the Casida pair (A, B) fits in BSE_DENSE_MAX_GB -- the dense
    route's own storage, 2 * n_ov**2 * 8 bytes of float64 -- and matrix-free
    above it. That pair is what the rule protects: the block action holds a
    three-index factor and a handful of trial vectors instead, so nothing else
    on the dense route grows as n_ov**2.

    tda: not exempt, and it does not move the boundary. The Tamm-Dancoff
    problem diagonalizes A alone, but the dense route BUILDS both blocks either
    way (`build_casida_matrices` returns the pair; only `CasidaSolver.solve`
    drops B), so a TDA solve holds the same memory and flips to the matrix-free
    route at the same size. The argument is here to state that, not to change
    the answer: a rule that exempts TDA lets a dense solve ask for terabytes.
    """
    return 'dense' if 2 * int(n_ov)**2 * 8 / BYTES_PER_GB <= BSE_DENSE_MAX_GB else 'davidson'
