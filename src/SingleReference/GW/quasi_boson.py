"""The quasi-boson dRPA ground state and its EOM G0W0 quasiparticle energies.

The drUCCD/Bogoliubov layer of Toelle (arXiv:2412.17085) and Toelle, Kitsaras
and Loos (arXiv:2507.02160), in closed form:

  J(t)   = [e^t (A+B) e^t - e^-t (A-B) e^-t]/2 = 0     (amplitude equation)
  e^{2t} = P^{-1/2} [P^{1/2} (A-B) P^{1/2}]^{1/2} P^{-1/2},  P = A+B
  Abar   = e^t (A+B) e^t,  eig(Abar) = Omega_RPA
  E_c    = Tr[A sinh^2 t] + Tr[B sinh t cosh t] = (sum Omega - Tr A)/2

All matrices are real symmetric; every matrix function goes through one eigh
(Daleckii-Krein divided differences), so there is no truncated-series machinery
anywhere and no iteration to converge.

The quasiparticle energies sit on those bosons. On the usual Dyson scale (holes
negative), in the boson eigenmode basis,

  H = [[ diag(eps_p) ,  W^nu_pj            ,  W^nu_pb            ],
       [   .         ,  diag(eps_j - Om_nu),        0            ],
       [   .         ,        0            ,  diag(eps_b + Om_nu)]]

with dressed couplings W^nu_pq = sqrt(2) * sum_I (pq|I) [e^t U]_{I,nu} (singlet
factor sqrt(2); e^t = 1 and Om/U from A for TDA screening). Its diagonal mode
downfolds exactly to the eta = 0 quasiparticle equation

  w = eps_p + Sigma_pp(w),
  Sigma_pp(w) = sum_{j,nu} (W^nu_pj)^2/(w - eps_j + Om_nu)
              + sum_{b,nu} (W^nu_pb)^2/(w - eps_b - Om_nu),

with closed-form eigenvector and weight Z = 1/(1 - Sigma'_pp(w*)).

Spin convention: restricted/spatial, singlet channel, factor-2 RPA blocks
A = Delta + 2(ia|jb), B = 2(ia|jb) -- the same numbers as restricted
`LinearResponse.linear_response.build_casida_matrices(lBSE=False)`, which the
move gate measures bitwise on H4/sto-3g and water/cc-pVDZ. They stay separate
routines because this one also takes an `MOEriBlocks` (the Casida builder needs
the nao^4 array the blocks exist to avoid) and because `tda_screening=True`
zeroes B, which the Casida builder's `triplet=True` does not: that one zeroes
the exchange in A instead.

This route holds the full four-index interaction (or its three blocks) rather
than factorizing it, which bounds it at a few hundred basis functions and makes
it the reference the cubic space-time route is checked against.
"""
import numpy as np

from src.Base.eri_blocks import as_blocks


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


class QPqb:
    """Diagonal-mode quasi-boson EOM G0W0 quasiparticle energies.

    The eta = 0 quasiparticle equation exactly, by Newton on the closed-form
    Sigma_pp: no broadening, no analytic continuation and no residue set. Z is
    the exact slope of the equation that was solved.

    The reverse mode is `gradients.quasi_boson_adjoint.QPqbAdjoint`, which
    differs only in `boson`.
    """

    #: The quasi-boson solver the screening is built with. The reverse mode
    #: overrides it, so an object reaching a gradient carries the Frechet maps.
    boson = RPA

    def __init__(self, eps, eri_mo, nocc, screening='rpa', delta=None):
        """delta: <p|Sigma_x - v_xc|p>, the static shift a Kohn-Sham starting
        point needs.

        G0W0 replaces the reference's exchange-correlation potential by the GW
        self-energy, so whatever static part the mean field already counted has
        to come back out. On a gas-phase Hartree-Fock reference v_xc IS Sigma_x
        and the shift vanishes, which is why None (all zeros) reproduces the
        Hartree-Fock route bitwise. Build it with
        `SingleReference.GW.qp_solve.static_exchange_diagonal(mf, mol, states,
        exchange='mf')`, which is the same function the cubic chain
        differentiates, so the two routes mean the same thing by it.
        """
        if screening not in ('rpa', 'tda'):
            raise ValueError(screening)
        self.eps = np.asarray(eps, float)
        self.nocc = nocc
        self.norb = len(eps)
        self.screening = screening
        self.delta = (np.zeros(self.norb) if delta is None
                      else np.asarray(delta, float))
        if self.delta.shape != (self.norb,):
            raise ValueError(f'delta must be one value per orbital, got '
                             f'{self.delta.shape} for {self.norb} orbitals')
        A, B, _ = build_rpa_AB(eps, eri_mo, nocc, tda_screening=(screening == 'tda'))
        self.qb = self.boson(A, B)
        self.omega, self.U = self.qb.eigAbar
        self.Vbare = couplings_V(eri_mo, nocc)             # (norb, norb, n_ov)
        self.etU = self.qb.exp_t @ self.U                  # (I, nu)
        self.Wnu = np.sqrt(2.0) * np.einsum('pqI,In->pqn', self.Vbare, self.etU,
                                            optimize=True)

    # -- self-energy and diagonal QP solve --------------------------------
    def _pole_arrays(self, p):
        nocc, eps, om = self.nocc, self.eps, self.omega
        num_h = self.Wnu[p, :nocc, :] ** 2                 # (j, nu)
        pol_h = eps[:nocc, None] - om[None, :]
        num_p = self.Wnu[p, nocc:, :] ** 2                 # (b, nu)
        pol_p = eps[nocc:, None] + om[None, :]
        return num_h, pol_h, num_p, pol_p

    def sigma(self, p, w):
        nh, ph, npp, pp = self._pole_arrays(p)
        return float((nh / (w - ph)).sum() + (npp / (w - pp)).sum())

    def sigma_prime(self, p, w):
        nh, ph, npp, pp = self._pole_arrays(p)
        return float(-(nh / (w - ph) ** 2).sum() - (npp / (w - pp) ** 2).sum())

    def solve_diag(self, p, w0=None, tol=1e-12, max_iter=200):
        """Newton QP root; returns (w*, Z). Z <= QP_WINDOW_Z_MIN is a lost QP.

        AN UNGUARDED NEWTON, deliberately, where `Solvers.qp_equation`'s
        guarded one holds the iterate off the poles of Sigma and replaces a
        root below its pole-strength floor by a linearized one. That guarded
        solver returns this root bitwise wherever a quasiparticle exists
        (every water/cc-pVDZ orbital with Z > 0.23), and a DIFFERENT number
        for the low-weight satellites: 79 mHa on orbital 22. Those satellites
        are what the pole-strength filter reads to decide which states carry a
        quasiparticle at all, so they have to be the roots of this equation
        and not a fallback.
        """
        w = self.eps[p] + self.delta[p] if w0 is None else w0
        for _ in range(max_iter):
            f = w - self.eps[p] - self.delta[p] - self.sigma(p, w)
            fp = 1.0 - self.sigma_prime(p, w)
            dw = -f / fp
            w += dw
            if abs(dw) < tol:
                break
        else:
            raise RuntimeError(f"QP Newton p={p} not converged, |dw|={abs(dw):.2e}")
        Z = 1.0 / (1.0 - self.sigma_prime(p, w))
        return w, Z

    def eigvec_diag(self, p, w):
        """Closed-form normalized eigenvector (rp, R_h (j,nu), R_p (b,nu))."""
        nh, ph, npp, pp = self._pole_arrays(p)
        Rh = self.Wnu[p, :self.nocc, :] / (w - ph)
        Rp = self.Wnu[p, self.nocc:, :] / (w - pp)
        n2 = 1.0 + (Rh ** 2).sum() + (Rp ** 2).sum()
        rp = 1.0 / np.sqrt(n2)
        return rp, rp * Rh, rp * Rp

    # -- dense supermatrix (validation only) ------------------------------
    def supermatrix_diag(self, p):
        nocc, norb = self.nocc, self.norb
        nvirt = norb - nocc
        nb = self.omega.size
        dim = 1 + nocc * nb + nvirt * nb
        H = np.zeros((dim, dim))
        H[0, 0] = self.eps[p] + self.delta[p]
        Wh = self.Wnu[p, :nocc, :].ravel()                 # (j, nu) row-major
        Wp = self.Wnu[p, nocc:, :].ravel()
        H[0, 1:1 + nocc * nb] = Wh
        H[1:1 + nocc * nb, 0] = Wh
        H[0, 1 + nocc * nb:] = Wp
        H[1 + nocc * nb:, 0] = Wp
        dh = (self.eps[:nocc, None] - self.omega[None, :]).ravel()
        dp = (self.eps[nocc:, None] + self.omega[None, :]).ravel()
        idx = np.arange(nocc * nb)
        H[1 + idx, 1 + idx] = dh
        idx = np.arange(nvirt * nb)
        H[1 + nocc * nb + idx, 1 + nocc * nb + idx] = dp
        return H

    def solve_diag_dense(self, p):
        """Max-weight eigenvalue of the dense diagonal supermatrix."""
        H = self.supermatrix_diag(p)
        w, V = np.linalg.eigh(H)
        k = np.argmax(V[0, :] ** 2)
        return w[k], V[0, k] ** 2


# ---------------------------------------------------------------------------
# symmetric matrix-function calculus
# ---------------------------------------------------------------------------

def eigh_sym(M):
    return np.linalg.eigh(0.5 * (M + M.T))


def funm_sym(M, f, eig=None):
    """f(M) for symmetric M via eigh. `eig` reuses a precomputed (w, V)."""
    w, V = eigh_sym(M) if eig is None else eig
    return (V * f(w)[None, :]) @ V.T


def sqrtm_sym(M, eig=None):
    return funm_sym(M, np.sqrt, eig)


def invsqrtm_sym(M, eig=None):
    return funm_sym(M, lambda w: 1.0 / np.sqrt(w), eig)


# ---------------------------------------------------------------------------
# RPA blocks and couplings (restricted singlet, factor 2)
# ---------------------------------------------------------------------------

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


def qp_energy_general(epsF, eri, nocc, p, screening='rpa', w0=None):
    """Diagonal-mode QP energy for a general (non-diagonal) Fock matrix.

    Dense supermatrix in the boson original basis with full F blocks -- the
    exact functional the gradient Lagrangian differentiates; used by the finite
    difference gates. It carries no static <Sigma_x - v_xc> shift, so on a
    Kohn-Sham reference it is not the same quantity as `QPqb.solve_diag`.
    """
    norb = epsF.shape[0]
    nvirt = norb - nocc
    occ, virt = np.arange(nocc), np.arange(nocc, norb)
    n_ov = nocc * nvirt
    A = (np.einsum('ij,ab->iajb', np.eye(nocc), epsF[np.ix_(virt, virt)])
         - np.einsum('ab,ij->iajb', np.eye(nvirt), epsF[np.ix_(occ, occ)])).reshape(n_ov, n_ov)
    A += 2.0 * eri[np.ix_(occ, virt, occ, virt)].reshape(n_ov, n_ov)
    B = (np.zeros((n_ov, n_ov)) if screening == 'tda'
         else 2.0 * eri[np.ix_(occ, virt, occ, virt)].reshape(n_ov, n_ov))
    qb = RPA(A, B)
    Abar = qb.Abar
    V = eri[:, :, occ[:, None], virt[None, :]].reshape(norb, norb, n_ov)
    W = np.sqrt(2.0) * np.einsum('pqI,IJ->pqJ', V, qb.exp_t, optimize=True)
    dim = 1 + nocc * n_ov + nvirt * n_ov
    H = np.zeros((dim, dim))
    H[0, 0] = epsF[p, p]
    Wh = W[p, :nocc, :].ravel()
    Wp = W[p, nocc:, :].ravel()
    o0, o1 = 1, 1 + nocc * n_ov
    H[0, o0:o1] = Wh; H[o0:o1, 0] = Wh
    H[0, o1:] = Wp;   H[o1:, 0] = Wp
    H[o0:o1, o0:o1] = (np.einsum('jk,IJ->jIkJ', epsF[np.ix_(occ, occ)], np.eye(n_ov))
                       - np.einsum('jk,IJ->jIkJ', np.eye(nocc), Abar)).reshape(nocc * n_ov, nocc * n_ov)
    H[o1:, o1:] = (np.einsum('bc,IJ->bIcJ', epsF[np.ix_(virt, virt)], np.eye(n_ov))
                   + np.einsum('bc,IJ->bIcJ', np.eye(nvirt), Abar)).reshape(nvirt * n_ov, nvirt * n_ov)
    w, Vv = np.linalg.eigh(H)
    if w0 is None:
        k = np.argmax(Vv[0, :] ** 2)
    else:
        cand = np.where(Vv[0, :] ** 2 > 0.2)[0]
        k = cand[np.argmin(np.abs(w[cand] - w0))]
    return w[k]
