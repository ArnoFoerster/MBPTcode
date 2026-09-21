import numpy as np
import scipy.linalg as la
from src.Base.utils.linearAlgebra.diagonalization import diagonalize_matrix
from src.Base.constants import CASIDA_NUMERICAL_EPS

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

    @staticmethod
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
            omega, Z_res, is_distributed, solver, comm = diagonalize_matrix(
                A, threshold=threshold)
            del A
            if is_distributed:
                solver.destroy()
                if comm.Get_rank() != 0:
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
        # purely to be normed and discarded -- 50 GB at hexacene/cc-pVTZ, where
        # the solve is already memory-bound. Identical value to 1e-8 relative.
        diag_AmB_full = np.diag(AmB)
        offdiag_sq = (np.linalg.norm(AmB)**2 - np.linalg.norm(diag_AmB_full)**2)
        offdiag_norm = np.sqrt(max(offdiag_sq, 0.0))
        is_AmB_diag = offdiag_norm < self.eta * self.ndim
        # The view keeps AmB's buffer alive for as long as its name lives.
        del diag_AmB_full

        global_N = self.ndim

        if is_AmB_diag:
            # A-B is Hermitian, so its diagonal is real (orbital energy differences).
            diag_AmB = np.diag(AmB).real
            diag_AmB = np.clip(diag_AmB, self.eta, None)
            del AmB
            sqrt_AmB_diag = np.sqrt(diag_AmB)
            inv_sqrt_AmB_diag = 1.0 / sqrt_AmB_diag
            
            # Target matrix: (A-B)^{1/2} (A+B) (A-B)^{1/2}, formed IN PLACE in
            # ApB's buffer. ApB is dead after this line in this branch, and the
            # out-of-place form costs two further n_ov x n_ov temporaries (one
            # per multiply) on top of the result -- at hexacene/cc-pVTZ that is
            # the difference between ~220 GB and ~270 GB on a 252 GB node.
            M = ApB
            M *= sqrt_AmB_diag[:, None]
            M *= sqrt_AmB_diag[None, :]
            del ApB

            # Perform diagonalization via backend
            omega2, Z_res, is_distributed, solver, comm = diagonalize_matrix(M, threshold=threshold)
            del M

            omega2 = np.clip(omega2, self.eta**2, None)
            omega = np.sqrt(omega2)
            
            if is_distributed:
                # Z came back whole on rank 0; the other ranks are done.
                solver.destroy()
                if comm.Get_rank() != 0:
                    self.is_distributed = True
                    return CasidaResult(omega, None, None, True)
            sqrt_omega = np.sqrt(omega)
            X_plus_Y = Z_res * sqrt_AmB_diag[:, None]
            X_plus_Y /= sqrt_omega[None, :]
            X_minus_Y = Z_res * inv_sqrt_AmB_diag[:, None]
            X_minus_Y *= sqrt_omega[None, :]
            if self.keep_intermediates:
                self.Z = Z_res
            del Z_res
            X, Y = self._combine_in_place(X_plus_Y, X_minus_Y)
        else:
            try:
                L = la.cholesky(ApB, lower=True)
                del ApB
            except la.LinAlgError:
                eigvals = la.eigvalsh(ApB)
                shift = max(0, -eigvals[0] + self.eta)
                L = la.cholesky(ApB + shift * np.eye(self.ndim), lower=True)
                del ApB

            M = L.conj().T @ (AmB @ L)
            del AmB

            # Perform diagonalization via backend
            omega2, Z_res, is_distributed, solver, comm = diagonalize_matrix(M, threshold=threshold)
            del M

            omega2 = np.clip(omega2, self.eta**2, None)
            omega = np.sqrt(omega2)
            
            if is_distributed:
                # Z came back whole on rank 0; the other ranks are done.
                solver.destroy()
                if comm.Get_rank() != 0:
                    self.is_distributed = True
                    return CasidaResult(omega, None, None, True)
            sqrt_omega = np.sqrt(omega)
            X_plus_Y = la.solve_triangular(L, Z_res, lower=True, trans='C')
            X_plus_Y *= sqrt_omega[None, :]
            X_minus_Y = L @ Z_res
            X_minus_Y /= sqrt_omega[None, :]
            if self.keep_intermediates:
                self.Z = Z_res
            del L, Z_res
            X, Y = self._combine_in_place(X_plus_Y, X_minus_Y)

        self.is_distributed = is_distributed
        return CasidaResult(omega, X, Y, is_distributed)
