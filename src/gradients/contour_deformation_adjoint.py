"""Reverse mode of the contour-deformation quasiparticle energy.

The forward half is `SingleReference.GW.contour_deformation`; this is its
adjoint, and nothing here is a second evaluation of the physics. Every term of

    Sigma^c_pp(omega) = Sigma^int + Sigma^res

is a rational function of (eps, Bp, C_ov) with no pole on the contour, so the
reverse pass is an ordinary one -- which is the whole point of deforming the
contour rather than continuing Sigma_c: the Pade map has a derivative of order
1e11. The quasiparticle energy itself is differentiated implicitly,
d eps^QP = Z [d eps_p + dSigma^c(w)|_w fixed], so the Newton is never unrolled.

Two things are chained separately. The imaginary-axis term's adjoint on
chi0(i.nu) is handed back per frequency (a streamed driver folds it into
projbar(tau) block by block and never holds the frequency axis), or, when the
screening was built here from C_ov, chained straight onto C_ov at O(N^4). Each
RESIDUE's adjoint on chi0 at a real frequency is rank one, c y y^T with
y = W Bp[:,q], and is PUSHED into the real-frequency backend, which owns the
chain from there: `ExplicitRealScreeningAdjoint` below lands it on (C_ov, d),
`LaplaceRealScreeningAdjoint` on proj(tau). Both subclass the forward-only
class of `SingleReference.GW.real_screening` and add nothing but the reverse
half, so a residue's value and its adjoint can never come from two spellings.
"""
import numpy as np

from src.Base.constants import LAPLACE_SCREENING_TOL
from src.SingleReference.GW.contour_deformation import _need
from src.SingleReference.GW.real_screening import (ExplicitRealScreening,
                                                   LaplaceRealScreening,
                                                   ov_energies, screening_aux)
from src.SingleReference.LinearResponse.imaginary_frequency import \
    frequency_factor
from src.SingleReference.base import get_occ_virt_indices


class ExplicitRealScreeningAdjoint(ExplicitRealScreening):
    """The explicit real-frequency backend with its reverse half.

    Adds push(freq, y, c), accumulating the rank-one adjoint c y y^T on
    chi0(freq), and adjoints(), reading back what every push since construction
    came to. The adjoint is chained onto (C_ov, d) at once (`screening_chain`)
    and adjoints() folds d back onto eps.

    It SUMS, so a reverse pass that pushed onto an instance already used by an
    earlier one gets both; build a fresh instance per pass.
    """

    def __init__(self, C_ov, eps, nocc, eta=0.0):
        super().__init__(C_ov, eps, nocc, eta)
        self.Cov_bar = np.zeros_like(C_ov)
        self.d_bar = np.zeros_like(self.d)

    def push(self, freq, y, c):
        cb, db, _ = screening_chain(y[:, None], np.array([c]), self.C_ov,
                                    self.d, freq, False, self.eta)
        self.Cov_bar += cb
        self.d_bar += db

    def adjoints(self):
        """(eps_bar, Cov_bar) of everything pushed since construction."""
        occ, virt = get_occ_virt_indices(self.eps, self.nocc)
        eps_bar = np.zeros_like(self.eps)
        d4 = self.d_bar.reshape(len(occ), len(virt))
        eps_bar[virt] += d4.sum(axis=0)
        eps_bar[occ] -= d4.sum(axis=1)
        return eps_bar, self.Cov_bar


class LaplaceRealScreeningAdjoint(LaplaceRealScreening):
    """The cubic real-frequency backend with its reverse half.

    Adds push(freq, y, c), accumulating the rank-one adjoint c y y^T on
    chi0(freq), and adjoints(), reading back what every push since construction
    came to. chi0(freq) = sum_k c_k(freq) proj(tau_k) with the cosh weights of
    `real_frequency_weights`, so the chain is projbar[k] += c_k(freq) c y y^T
    and the residues ride the same polarizability sweep as the integral term --
    the whole reason the route stays O(N^3).

    It SUMS, so a reverse pass that pushed onto an instance already used by an
    earlier one gets both; build a fresh instance per pass.
    """

    def __init__(self, proj_tau, grid, eps, nocc, tol=LAPLACE_SCREENING_TOL):
        super().__init__(proj_tau, grid, eps, nocc, tol)
        self.proj_bar = np.zeros_like(proj_tau)

    def push(self, freq, y, c):
        ck, _ = self._weights(freq)
        yy = c * np.outer(y, y)
        for k in range(len(ck)):
            self.proj_bar[k] += ck[k] * yy

    def adjoints(self):
        """projbar, (ntau, naux, naux): the adjoint on proj(tau) of everything
        pushed since construction -- to be added to the integral term's before
        the one polarizability-sweep backward pass."""
        return self.proj_bar


def _residue_backend(real_screening, C_ov, eps, nocc, eta):
    """The backend the residue term uses: the one given, else explicit from C_ov.

    The ADJOINT class by default, because everything below pushes onto it.
    """
    if real_screening is not None:
        return real_screening
    _need(C_ov, 'the residue term')
    return ExplicitRealScreeningAdjoint(C_ov, eps, nocc, eta)


def screening_chain(P, weights, C_ov, d, freq, imaginary, eta=0.0):
    """Push an adjoint U diag(weights) U^T on (W - I) back onto (C_ov, d, freq).

    P = W U, (naux, r). The adjoint on W is W U diag(w) U^T W = P diag(w) P^T
    and stays in that form: it is LOW RANK -- r = norb for the integral term,
    1 for a residue -- so two naux^2 r products replace two naux^3 ones, and
    there are hundreds of frequencies. Through chi0 = 2 (C_ov f) C_ov^T,
    Cov_bar = 4 f * (chi0_bar C_ov) and f_bar = 2 diag(C_ov^T chi0_bar C_ov),
    and f's slopes in d and in the frequency close it. Returns (Cov_bar, d_bar,
    dfreq); dfreq is None on the imaginary axis, where the frequency is a
    quadrature node and not a variable.
    """
    CB = (P * weights) @ (P.T @ C_ov)                # chi0_bar @ C_ov, low rank
    f, df_dd, df_dw = frequency_factor(d, freq, imaginary, eta, slopes=True)
    f_bar = 2.0 * np.einsum('PI,PI->I', C_ov, CB)
    return 4.0 * f * CB, f_bar * df_dd, \
        (None if df_dw is None else float(f_bar @ df_dw))


def integral_term_backward(Bp, WtB, weights, want_chi0=True):
    """Adjoint pieces of sum_q g_q Bp[:,q]^T (W - I) Bp[:,q] at ONE frequency.

    WtB = W Bp; weights = coeff * g, (norb,). Returns (Bp_bar, wc, chi0_bar):
    the direct adjoint on Bp; the contraction wc_q = Bp[:,q]^T (W - I) Bp[:,q]
    that the omega-dependence's adjoint needs; and the adjoint on chi0,
    WtB diag(weights) WtB^T -- rank norb, symmetric by construction -- or None.
    A streamed driver calls this per frequency and folds chi0_bar into
    projbar(tau) block by block, so the frequency axis is never held.
    """
    WB = WtB - Bp
    wc = np.einsum('Pq,Pq->q', Bp, WB)
    chi0_bar = (WtB * weights) @ WtB.T if want_chi0 else None
    return 2.0 * WB * weights, wc, chi0_bar


def residue_terms_backward(p, omega, Bp, eps, nocc, residues, sigma_bar,
                           real_screening):
    """(eps_bar, Bp_bar, omega_bar) of sigma_bar * Sigma^res, through a backend.

    Sigma^res = sum_(q,weight) weight Bp[:,q]^T [W(|eps_q - omega|) - I] Bp[:,q].
    What is returned is the part every backend shares: the direct adjoint on
    Bp, and the frequency slope routed to eps_q and omega. Each residue's
    adjoint on chi0(freq) is rank one, c y y^T with y = W Bp[:,q], and is
    PUSHED into the backend, which chains it onto its own variables -- (C_ov, d)
    for the explicit one, proj(tau) for the Laplace one; the caller reads
    `real_screening.adjoints()` afterwards.
    """
    eps_bar = np.zeros_like(eps)
    Bp_bar = np.zeros_like(Bp)
    omega_bar = 0.0
    for q, weight in residues:
        gap = eps[q] - omega
        sgn = np.sign(gap) if gap != 0.0 else 0.0
        freq = abs(gap)
        b = Bp[:, q]
        y = real_screening.apply(freq, b)
        c = sigma_bar * weight
        Bp_bar[:, q] += 2.0 * c * (y - b)
        dfreq = c * real_screening.slope(freq, y)
        real_screening.push(freq, y, c)
        eps_bar[q] += dfreq * sgn
        omega_bar -= dfreq * sgn
    return eps_bar, Bp_bar, omega_bar


def sigma_cd_backward(p, omega, Bp, eps, nocc, nu_points, nu_weights, residues,
                      sigma_bar=1.0, wbp=None, eta=0.0, want_chi0=True,
                      C_ov=None, real_screening=None):
    """(eps_bar, Bp_bar, Cov_bar, omega_bar, chi0_bar) of sigma_bar * Sigma^c_pp.

    Two ways to own the imaginary-axis screening:
      wbp given -- W(i.nu) Bp came from outside (a space-time route). The
                   adjoint on chi0(i.nu) comes back as chi0_bar, (nfreq, naux,
                   naux), or None with want_chi0=False, which also skips the
                   product that fills it; the caller owns the chain from there,
                   which is where the cubic scaling lives.
      wbp None  -- the screening is built here from C_ov and differentiated
                   through it: Cov_bar carries that, chi0_bar is None. The
                   O(N^4) reference the cubic route is checked against.
    Residues go through `real_screening`; with none given, through the explicit
    backend built from C_ov here, whose adjoints are folded into eps_bar and
    Cov_bar before returning. A backend passed in keeps what was pushed and the
    caller reads its adjoints() -- that is how the Laplace backend hands the
    residues' adjoint on proj(tau) to a streamed driver. Cov_bar is
    (naux, nocc*nvirt) when anything was chained through C_ov, else None.

    omega is held fixed here; the quasiparticle chain rule multiplies by Z
    outside. Every term is a rational function of (eps, Bp, C_ov) with no pole
    on the contour, so this is an ordinary reverse pass -- which is the whole
    point of contour deformation over a continuation.
    """
    de = omega - eps
    eps_bar = np.zeros_like(eps)
    Bp_bar = np.zeros_like(Bp)
    omega_bar = 0.0
    explicit = wbp is None
    chi0_bar_out = None
    Cov_bar = None
    if explicit:
        _need(C_ov, 'the imaginary-axis screening without wbp')
        d = ov_energies(eps, nocc)
        Cov_bar = np.zeros_like(C_ov)
        d_bar = np.zeros_like(d)
    elif want_chi0:
        chi0_bar_out = np.zeros((len(nu_points),) + (Bp.shape[0],) * 2)
    for k, (nu, wt) in enumerate(zip(nu_points, nu_weights)):
        WtB = screening_aux(C_ov, d, nu, True)[1] @ Bp if explicit else wbp[k]
        g = de / (de ** 2 + nu ** 2)
        coeff = -sigma_bar * wt / np.pi
        # Sigma^int = coeff * sum_q g_q Bp[:,q]^T (W - I) Bp[:,q]
        bb, wc, cb = integral_term_backward(Bp, WtB, coeff * g,
                                            want_chi0=chi0_bar_out is not None)
        Bp_bar += bb
        if explicit:
            cvb, db, _ = screening_chain(WtB, coeff * g, C_ov, d, nu, True)
            Cov_bar += cvb
            d_bar += db
        elif cb is not None:
            chi0_bar_out[k] = cb
        # d g_q / d(omega - eps_q)
        dg = (nu ** 2 - de ** 2) / (de ** 2 + nu ** 2) ** 2
        eps_bar -= coeff * wc * dg
        omega_bar += coeff * float(wc @ dg)
    if explicit:
        occ, virt = get_occ_virt_indices(eps, nocc)
        d4 = d_bar.reshape(len(occ), len(virt))
        eps_bar[virt] += d4.sum(axis=0)
        eps_bar[occ] -= d4.sum(axis=1)
    if residues:
        rs = _residue_backend(real_screening, C_ov, eps, nocc, eta)
        e_r, b_r, o_r = residue_terms_backward(p, omega, Bp, eps, nocc, residues,
                                               sigma_bar, rs)
        eps_bar += e_r
        Bp_bar += b_r
        omega_bar += o_r
        if real_screening is None:            # built here: fold it here
            e_c, c_r = rs.adjoints()
            eps_bar += e_c
            Cov_bar = c_r if Cov_bar is None else Cov_bar + c_r
    return eps_bar, Bp_bar, Cov_bar, omega_bar, chi0_bar_out


def qp_energy_cd_backward(p, w_star, z_factor, residues, Bp, eps, nocc,
                          nu_points, nu_weights, wbp=None, eta=0.0, C_ov=None,
                          want_chi0=True, real_screening=None):
    """(eps_bar, Bp_bar, Cov_bar, chi0_bar) of eps^QP_p, by implicit differentiation.

    d eps^QP = Z [ d eps_p + dSigma^c(w)|_w fixed ], the partial being the one
    `sigma_cd_backward` returns.
    """
    eps_bar, Bp_bar, Cov_bar, _, chi0_bar = sigma_cd_backward(
        p, w_star, Bp, eps, nocc, nu_points, nu_weights, residues,
        sigma_bar=z_factor, wbp=wbp, eta=eta, want_chi0=want_chi0, C_ov=C_ov,
        real_screening=real_screening)
    eps_bar[p] += z_factor
    return eps_bar, Bp_bar, Cov_bar, chi0_bar
