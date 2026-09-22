"""Reverse mode of the dense quasi-boson dRPA, G0W0 and BSE route.

The forward half is `SingleReference.GW.quasi_boson` and
`SingleReference.LinearResponse.quasi_boson_bse`; nothing here re-evaluates the
physics. Each adjoint class is the forward one plus its reverse half, so an
object built through this module IS the forward object as far as every energy
is concerned and carries the maps a gradient needs on top:

  RPAAdjoint     the boson density matrices dE_c/dA, dE_c/dB at fixed t, and
                 every Frechet/adjoint map of the amplitude equation J(t) and
                 of Abar(t). All of them go through ONE Daleckii-Krein divided
                 difference in the eigenbasis of t, which `RPA` already holds,
                 so no matrix function is re-diagonalized here
  QPqbAdjoint    the same quasiparticle solver, with `boson` pointing at the
                 class above so the chain rule finds those maps
  BSEqbAdjoint   the exact (F, ERI, t) partials of Omega_nu by
                 Hellmann-Feynman on (X, Y), with the per-quasiparticle-state
                 chain weight-folded -- everything downstream of the two
                 per-state carriers is LINEAR in them, so the weights fold
                 inside and the (n_ov)^3 maps run once instead of norb times

The composition seam is the two class attributes `QPqb.boson` and
`BSEqb.quasiparticle`: a forward-only object would carry no adjoint at all, and
a gradient caller has to reach the objects below rather than their bases.
"""
import numpy as np

from src.Base.constants import XC_SHIFT_GRADIENT_TOL
from src.SingleReference.GW.quasi_boson import QPqb, RPA, eigh_sym
from src.SingleReference.LinearResponse.quasi_boson_bse import BSEqb
from src.gradients.targets import chain_AB


class RPAAdjoint(RPA):
    """The closed-form dRPA ground state with its reverse half."""

    # -- boson density matrices (partials of E_c at fixed t) --------------
    def gamma_A(self):
        """dE_c/dA_IJ = [sinh^2 t]_IJ."""
        return self.sinh_t @ self.sinh_t

    def gamma_B(self):
        """dE_c/dB_IJ = [sinh t cosh t]_IJ."""
        return self.sinh_t @ self.cosh_t

    # -- Frechet helpers in the t eigenbasis ------------------------------
    def d_expt(self, D, sign=+1):
        """Frechet derivative of e^{sign*t} at t in direction D."""
        f = (lambda w: np.exp(sign * w))
        df = (lambda w: sign * np.exp(sign * w))
        return frechet_funm_sym(self.t, f, df, D, eig=(self.wt, self.Vt))

    def dJ_dt(self, D):
        """Directional derivative of the residual J w.r.t. t (A, B fixed)."""
        P = self.A + self.B
        Mm = self.A - self.B
        dep = self.d_expt(D, +1)
        dem = self.d_expt(D, -1)
        r = (dep @ P @ self.exp_t + self.exp_t @ P @ dep
             - dem @ Mm @ self.exp_mt - self.exp_mt @ Mm @ dem)
        return 0.5 * r

    def dJ_dAB(self, dA, dB):
        """Directional derivative of J w.r.t. (A, B) at fixed t."""
        return 0.5 * (self.exp_t @ (dA + dB) @ self.exp_t
                      - self.exp_mt @ (dA - dB) @ self.exp_mt)

    def dJ_dAB_adjoint(self, Z):
        """Adjoint: d(sum_IJ Z_IJ J_IJ)/dA and /dB at fixed t (Z symmetric)."""
        Zs = 0.5 * (Z + Z.T)
        ep = self.exp_t @ Zs @ self.exp_t
        em = self.exp_mt @ Zs @ self.exp_mt
        return 0.5 * (ep - em), 0.5 * (ep + em)

    def _L_exp(self, S, sign=+1):
        """Self-adjoint Frechet map L_{e^{sign t}}(S), symmetric S."""
        return frechet_funm_sym(self.t, lambda w: np.exp(sign * w),
                                lambda w: sign * np.exp(sign * w),
                                S, eig=(self.wt, self.Vt))

    def dJ_dt_adjoint(self, Z):
        """t-gradient of sum_IJ Z_IJ J_IJ (A, B fixed; symmetric output).

        Tr[Z dJ(D)] = Tr[D {L_e^t(sym(P e^t Z))} - D {L_e^-t(sym(M e^-t Z))}]
        with P = A+B, M = A-B.
        """
        Zs = 0.5 * (Z + Z.T)
        P = self.A + self.B
        Mm = self.A - self.B
        Q1 = P @ self.exp_t @ Zs
        Q2 = Mm @ self.exp_mt @ Zs
        return (self._L_exp(0.5 * (Q1 + Q1.T), +1)
                - self._L_exp(0.5 * (Q2 + Q2.T), -1))

    def abar_t_gradient(self, T):
        """t-gradient of Tr[T Abar(t)], Abar = [e^t P e^t + e^-t M e^-t]/2."""
        Ts = 0.5 * (T + T.T)
        P = self.A + self.B
        Mm = self.A - self.B
        Q1 = P @ self.exp_t @ Ts
        Q2 = Mm @ self.exp_mt @ Ts
        return (self._L_exp(0.5 * (Q1 + Q1.T), +1)
                + self._L_exp(0.5 * (Q2 + Q2.T), -1))

    def abar_AB_adjoint(self, T):
        """(dA, dB)-adjoints of Tr[T Abar(t)] at fixed t."""
        Ts = 0.5 * (T + T.T)
        ep = self.exp_t @ Ts @ self.exp_t
        em = self.exp_mt @ Ts @ self.exp_mt
        return 0.5 * (ep + em), 0.5 * (ep - em)

    def linear_expt_t_gradient(self, K):
        """t-gradient of Tr[e^t K^T] = sum_IJ [e^t]_IJ K_IJ."""
        return self._L_exp(0.5 * (K + K.T), +1)


class QPqbAdjoint(QPqb):
    """The diagonal quasi-boson EOM solve carrying differentiable bosons."""

    boson = RPAAdjoint


class BSEqbAdjoint(BSEqb):
    """BSE@G0W0 with the exact (F, ERI, t) partials of one excitation energy."""

    quasiparticle = QPqbAdjoint

    def partials(self, n):
        """(gammaF, Gamma4, t_grad) of Omega_n. t_grad None for TDA screening."""
        shift = float(np.abs(self.qp.delta).max())
        # By MAGNITUDE, not by nonzeroness: Sigma_x - v_xc is zero on a
        # Hartree-Fock reference analytically and round-off numerically, so
        # `delta.any()` refuses the one reference this route exists to
        # differentiate.
        if shift > XC_SHIFT_GRADIENT_TOL:
            raise NotImplementedError(
                f'the dense route differentiates a Hartree-Fock reference '
                f'only: the static <p|Sigma_x - v_xc|p> shift reaches '
                f'{shift:.3e} Ha here, and while it enters the energy its '
                f'nuclear derivative is not assembled on this route. '
                f'Energies on a Kohn-Sham starting point are fine; a '
                f'gradient there needs the differentiated xc correction of '
                f'the cubic chain.')
        nocc, nvirt, norb = self.nocc, self.nvirt, self.norb
        occ, virt = slice(0, nocc), slice(nocc, norb)
        X = self.X[:, n]
        Y = self.Y[:, n]
        CA = np.outer(X, X) + np.outer(Y, Y)                # (ia, jb)
        CB = np.outer(X, Y) + np.outer(Y, X)

        gammaF = np.zeros((norb, norb))
        Gamma4 = np.zeros((norb,) * 4)
        qb = self.qp.qb
        t_grad = None if self.screening == 'tda' else np.zeros_like(qb.t)

        # (i) QP-diagonal weights
        dCA = np.einsum('II->I', CA).reshape(nocc, nvirt)
        c = np.zeros(norb)
        c[nocc:] = dCA.sum(axis=0)
        c[:nocc] = -dCA.sum(axis=1)
        self._qp_chain(c, gammaF, Gamma4, t_grad)

        # (ii) bare exchange kappa (ia|jb) in A (CA) and B (CB); absent for a
        # triplet, where kappa = 0
        CAB4 = (CA + CB).reshape(nocc, nvirt, nocc, nvirt)
        Gamma4[occ, virt, occ, virt] += self.kappa * CAB4

        # (iii) -W_direct in A, coefficient -CA on W_direct[ia,jb] = Wd4[i,j,a,b]
        Cf4 = CA.reshape(nocc, nvirt, nocc, nvirt).transpose(0, 2, 1, 3)  # (i,j,a,b)
        Gamma4[occ, occ, virt, virt] -= Cf4
        # screening +4 sum V_oo Pinv V_vv
        T1 = 4.0 * np.einsum('ijab,abJ,JI->ijI', Cf4, self.Vvv, self.Pinv,
                             optimize=True)
        Gamma4[occ, occ, occ, virt] += T1.reshape(nocc, nocc, nocc, nvirt)
        T2 = 4.0 * np.einsum('ijab,ijI,IJ->abJ', Cf4, self.Voo, self.Pinv,
                             optimize=True)
        Gamma4[virt, virt, occ, virt] += T2.reshape(nvirt, nvirt, nocc, nvirt)
        SP = 4.0 * np.einsum('ijab,ijI,abJ->IJ', Cf4, self.Voo, self.Vvv,
                             optimize=True)
        XP = -self.Pinv @ (0.5 * (SP + SP.T)) @ self.Pinv
        self._chain_P(XP, gammaF, Gamma4)

        # (iv) -W_swap in B, coefficient -CB on W_swap[ia,jb] = Ws4[a,j,b,i]
        if not self.bse_tda:
            Cs4 = np.einsum('iajb->ajbi',
                            CB.reshape(nocc, nvirt, nocc, nvirt))  # (a,j,b,i)
            Gamma4[virt, occ, virt, occ] -= Cs4
            T3 = 8.0 * np.einsum('ajbi,biJ,JI->ajI', Cs4, self.Vvo, self.Pinv,
                                 optimize=True)
            Gamma4[virt, occ, occ, virt] += T3.reshape(nvirt, nocc, nocc, nvirt)
            SP2 = 4.0 * np.einsum('ajbi,ajI,biJ->IJ', Cs4, self.Vvo, self.Vvo,
                                  optimize=True)
            XP2 = -self.Pinv @ (0.5 * (SP2 + SP2.T)) @ self.Pinv
            self._chain_P(XP2, gammaF, Gamma4)

        return gammaF, Gamma4, t_grad

    def _qp_chain(self, c, gammaF, Gamma4, t_grad):
        """Accumulate sum_p c_p d eps^QP_p / d(F, ERI, t) in ONE pass.

        Everything downstream of the two per-state carriers is LINEAR in them:
        T_p (the Abar adjoint) enters through abar_AB_adjoint/abar_t_gradient
        and K_p (the coupling adjoint) through linear_expt_t_gradient. So the
        weights fold inside -- sum T and K over p first and run the (n_ov)^3
        maps once, instead of norb times as a loop over per-orbital partials
        does. Same numbers, one power of N cheaper.

        T is accumulated in the boson EIGENmode basis, where T_p = U Tnu_p U^T
        with U orthogonal, so only the accumulated matrix is rotated.
        """
        nocc, nvirt, norb = self.nocc, self.nvirt, self.norb
        qp = self.qp
        qb = qp.qb
        U = qp.U
        nb = U.shape[1]
        Tnu = np.zeros((nb, nb))
        Knu = np.zeros((nb, nb))
        UtE = U.T @ qb.exp_t                          # (nu, I), for the couplings
        Vb = qp.Vbare
        for p in range(norb):
            if abs(c[p]) < 1e-15:
                continue
            if p not in self.qp_roots:
                gammaF[p, p] += c[p]                  # unrelaxed HF eigenvalue
                continue
            rp, Rh, Rp = qp.eigvec_diag(p, self.qp_roots[p])   # (j,nu), (b,nu)
            gammaF[p, p] += c[p] * rp ** 2
            gammaF[:nocc, :nocc] += c[p] * (Rh @ Rh.T)
            gammaF[nocc:, nocc:] += c[p] * (Rp @ Rp.T)
            Tnu += c[p] * (Rp.T @ Rp - Rh.T @ Rh)
            # d eps/dW^J_pq = 2 rp R_qJ, and W = sqrt(2) V e^t
            cf = c[p] * 2.0 * rp * np.sqrt(2.0)
            Gamma4[p, :nocc, :nocc, nocc:] += (cf * (Rh @ UtE)).reshape(
                nocc, nocc, nvirt)
            Gamma4[p, nocc:, :nocc, nocc:] += (cf * (Rp @ UtE)).reshape(
                nvirt, nocc, nvirt)
            if t_grad is not None:
                Knu += cf * (Vb[p, :nocc, :].T @ Rh + Vb[p, nocc:, :].T @ Rp)

        T = U @ Tnu @ U.T
        XA, XB = qb.abar_AB_adjoint(T)
        chain_AB(XA, XB, nocc, norb, gammaF, Gamma4)
        if t_grad is not None:
            t_grad += qb.abar_t_gradient(T)
            t_grad += qb.linear_expt_t_gradient(Knu @ U.T)

    def _chain_P(self, XP, gammaF, Gamma4):
        """Chain an adjoint on P (= A+B or A at mean-field) into (F, ERI)."""
        if self.screening == 'rpa':
            chain_AB(XP, XP, self.nocc, self.norb, gammaF, Gamma4)
        else:
            chain_AB(XP, np.zeros_like(XP), self.nocc, self.norb, gammaF, Gamma4)


def frechet_funm_sym(M, f, df, D, eig=None):
    """Frechet derivative L_f(M; D) for symmetric M and direction D.

    Daleckii-Krein: L = V (Phi o (V^T D V)) V^T with
    Phi_ij = (f(w_i)-f(w_j))/(w_i-w_j), Phi_ii = df(w_i); near-degenerate
    pairs fall back to df at the midpoint (relative threshold).
    """
    w, V = eigh_sym(M) if eig is None else eig
    dw = w[:, None] - w[None, :]
    fw = f(w)
    num = fw[:, None] - fw[None, :]
    scale = np.maximum(np.abs(w[:, None]), np.abs(w[None, :])) + 1.0
    small = np.abs(dw) < 1e-12 * scale
    Phi = np.where(small, df(0.5 * (w[:, None] + w[None, :])),
                   num / np.where(small, 1.0, dw))
    Dm = V.T @ (0.5 * (D + D.T)) @ V
    return V @ (Phi * Dm) @ V.T
