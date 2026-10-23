"""The quasi-boson RPA: the closed-form direct-RPA ground state on factor-2
singlet (A, B) and the bare boson couplings, which the dense quasi-boson GW
(`GW.quasi_boson`) and the Riccati solve (`riccati`) are built on."""
import numpy as np

from src.SingleReference.common.eri_blocks import as_blocks
from src.Base.utils.linear_algebra.matrix_functions import eigh_sym, sqrtm_sym


class RPA:
    """Closed-form direct-RPA ground state for factor-2 singlet (A, B).

    Solved in the Bogoliubov/drUCCD form -- ring-diagram coupled-cluster
    doubles, whose amplitudes ARE the RPA ones -- which is where t comes from.
    The energy is dRPA's, and `gradients.rpa_ground_state.RPAGroundStateChain`
    is the same quantity on the cubic space-time factorization rather than the
    dense one.

    Holds t and every derived object needed downstream, with one eigh of G
    shared by all matrix functions of t (t = log(G)/2 has G's eigenvectors).
    TDA screening (B = 0) gives t = 0 exactly and Abar = A.

    The reverse mode -- the boson density matrices and the Frechet/adjoint maps
    at this t -- is `gradients.quasi_boson_adjoint.RPAAdjoint`.
    """

    def __init__(self, A, B):
        self.A = A
        self.B = B
        n = A.shape[0]
        P = A + B
        Mm = A - B
        wP, VP = eigh_sym(P)
        wM = np.linalg.eigvalsh(0.5 * (Mm + Mm.T))
        if wP.min() <= 0 or wM.min() <= 0:
            raise ValueError(f"RPA instability: min eig(A+B)={wP.min():.3e}, "
                             f"min eig(A-B)={wM.min():.3e}")
        Ph = (VP * np.sqrt(wP)[None, :]) @ VP.T
        Pmh = (VP * (1.0 / np.sqrt(wP))[None, :]) @ VP.T
        K = Ph @ Mm @ Ph
        Kh = sqrtm_sym(K)
        G = Pmh @ Kh @ Pmh                     # e^{2t}
        self.eigG = eigh_sym(G)
        wG, VG = self.eigG
        if wG.min() <= 0:
            raise ValueError("e^{2t} not positive definite")
        self.t = (VG * (0.5 * np.log(wG))[None, :]) @ VG.T
        self.wt = 0.5 * np.log(wG)             # eigenvalues of t (vectors VG)
        self.Vt = VG
        self.exp_t = (VG * np.sqrt(wG)[None, :]) @ VG.T          # e^t
        self.exp_mt = (VG * (1.0 / np.sqrt(wG))[None, :]) @ VG.T  # e^-t
        self.cosh_t = 0.5 * (self.exp_t + self.exp_mt)
        self.sinh_t = 0.5 * (self.exp_t - self.exp_mt)
        self.Abar = 0.5 * (self.exp_t @ P @ self.exp_t
                           + self.exp_mt @ Mm @ self.exp_mt)
        self.eigAbar = eigh_sym(self.Abar)     # (Omega, U)

    # -- diagnostics ------------------------------------------------------
    def residual(self, t=None):
        """J(t) = [e^t(A+B)e^t - e^-t(A-B)e^-t]/2 (symmetric)."""
        if t is None:
            et, emt = self.exp_t, self.exp_mt
        else:
            w, V = eigh_sym(t)
            et = (V * np.exp(w)[None, :]) @ V.T
            emt = (V * np.exp(-w)[None, :]) @ V.T
        return 0.5 * (et @ (self.A + self.B) @ et
                      - emt @ (self.A - self.B) @ emt)

    @property
    def omega(self):
        return self.eigAbar[0]

    # -- energies ---------------------------------------------------------
    def e_corr(self):
        s2 = self.sinh_t @ self.sinh_t
        sc = self.sinh_t @ self.cosh_t
        return float(np.einsum('ij,ji->', self.A, s2)
                     + np.einsum('ij,ji->', self.B, sc))

    def e_corr_plasmon(self):
        """Furche's trace formula E_c = (sum_s Omega_s - Tr[A])/2.

        The same quantity `rpa_energy.rpa_correlation_energy_casida` returns
        from a Casida solve, but not the same arithmetic: Omega comes from one
        eigh of Abar rather than from the symplectic Casida solver, and the two
        differ by 1.8e-15 Ha on H4/sto-3g and 5.7e-14 Ha on water/cc-pVDZ.
        """
        return float(0.5 * (self.omega.sum() - np.trace(self.A)))


def build_rpa_AB(eps, eri_mo, nocc, tda_screening=False):
    """Factor-2 singlet RPA blocks over I=(ia), row-major i*nvirt+(a-nocc).

    tda_screening=True zeroes B (TDA screening variant: bosons from the
    Hermitian TDA problem, t = 0 downstream).
    """
    norb = len(eps)
    occ = np.arange(nocc)
    virt = np.arange(nocc, norb)
    d = (eps[virt][None, :] - eps[occ][:, None]).ravel()
    n_ov = d.size
    V_iajb = as_blocks(eri_mo, nocc).ovov.reshape(n_ov, n_ov)
    A = np.diag(d) + 2.0 * V_iajb
    B = np.zeros_like(A) if tda_screening else 2.0 * V_iajb
    return A, B, d


def couplings_V(eri_mo, nocc):
    """Bare boson couplings V^I_pq = (pq|ia), shape (norb, norb, n_ov).

    eri_mo: a full (pq|rs) or an `MOEriBlocks`; both give the same numbers.

    The singlet spin factor for the self-energy/supermatrix is applied at
    the point of use (sqrt(2) per coupling => factor 2 in |W|^2), pinned
    against `SelfEnergySolver` in the quasiparticle gate.
    """
    b = as_blocks(eri_mo, nocc)
    return b.pqov.reshape(b.norb, b.norb, b.n_ov)
