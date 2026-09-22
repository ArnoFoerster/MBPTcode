"""BSE@G0W0 on the quasi-boson route, in the four Toelle/Kitsaras/Loos variants.

Variants (arXiv:2507.02160 Table I): screening in {rpa, tda} (both the G0W0
stage and the BSE kernel), BSE eigenproblem in {full, tda}.

Static kernel in closed form,

    W = v - 4 V P^{-1} V^T,   P = A_mf + B_mf (RPA screening) or A_mf (TDA),

with no t dependence: the screening is the mean-field response, and the
amplitudes only enter through the quasiparticle energies on the BSE diagonal.
Those come from the diagonal quasi-boson EOM solve of
`GW.quasi_boson.QPqb`, filtered on pole strength (Hartree-Fock fallback for a
state whose root is a satellite), while the screening keeps mean-field
energies: the standard G0W0-BSE split.

kappa weights the bare exchange (ia|jb) in A and in B and nothing else -- 2 for
a singlet, 0 for a triplet. W, the quasiparticle energies and every quasi-boson
normalization are spin-independent.

The reverse mode -- the exact (F, ERI, t) partials of Omega_nu by
Hellmann-Feynman on (X, Y), the per-quasiparticle-state chain and the P^{-1}
screening response -- is `gradients.quasi_boson_adjoint.BSEqbAdjoint`.
"""
import numpy as np

from src.Base.constants import KAPPA, QP_WINDOW_Z_MIN
from src.Base.eri_blocks import as_blocks
from src.SingleReference.GW.quasi_boson import QPqb


class BSEqb:
    """BSE@G0W0 eigenpairs on the dense quasi-boson route."""

    #: The quasiparticle solver the BSE diagonal is dressed with. The reverse
    #: mode overrides it, so an object reaching a gradient carries the boson
    #: Frechet maps its chain rule needs.
    quasiparticle = QPqb

    def __init__(self, mf, eri_mo, nocc, screening='rpa', bse_tda=False,
                 qp_orbs='all', seeds=None, pinned=False, delta=None,
                 spin='singlet'):
        """delta: the static <p|Sigma_x - v_xc|p> shift; see `QPqb`. It reaches
        the BSE through `eps_qp` alone, which is where the quasiparticle
        energies enter the kernel.

        spin: 'singlet' (kappa = 2) or 'triplet' (kappa = 0). Kappa weights the
        bare exchange (ia|jb) in A and in B, and nothing else: the screened term
        W, the quasiparticle energies and every quasi-boson normalization are
        spin-independent. The gradient needs no separate treatment because the
        one place kappa appears there, on Gamma4[o,v,o,v], is the derivative of
        those two terms and carries the same factor.
        """
        if spin not in KAPPA:
            raise ValueError(f"spin {spin!r}: one of {', '.join(sorted(KAPPA))}")
        self.spin = spin
        self.kappa = KAPPA[spin]
        eps = np.asarray(mf.mo_energy, float)
        self.eps = eps
        self.nocc = nocc
        self.norb = len(eps)
        self.nvirt = self.norb - nocc
        self.screening = screening
        self.bse_tda = bse_tda
        self.eri = as_blocks(eri_mo, nocc)

        self.qp = self.quasiparticle(eps, eri_mo, nocc, screening=screening,
                                     delta=delta)
        if qp_orbs == 'all':
            qp_orbs = list(range(self.norb))
        self.qp_orbs = list(qp_orbs)
        # seeds: dict p -> w0 (root-following across geometries); pinned=True
        # accepts every seeded/listed state regardless of its weight, so the
        # QP set is DECIDED ONCE at the reference geometry -- the per-geometry
        # pole-strength filter makes the PES discontinuous at filter flips.
        self.eps_qp = eps.copy()
        self.qp_roots = {}
        for p in self.qp_orbs:
            w0 = seeds.get(p) if seeds else None
            w, Z = self.qp.solve_diag(p, w0=w0)
            if pinned or Z > QP_WINDOW_Z_MIN:
                self.eps_qp[p] = w
                self.qp_roots[p] = w

        # screening resolvent P^-1 at mean-field energies
        qb = self.qp.qb
        P = (qb.A + qb.B) if screening == 'rpa' else qb.A
        self.Pinv = np.linalg.inv(P)

        # kernel pieces
        nocc_, nvirt_, norb_ = nocc, self.nvirt, self.norb
        n_ov = nocc_ * nvirt_
        V = self.qp.Vbare                                   # (p,q,I) = (pq|ia)
        occ, virt = slice(0, nocc_), slice(nocc_, norb_)
        self.Voo = V[occ, occ, :]
        self.Vvv = V[virt, virt, :]
        self.Vvo = V[virt, occ, :]
        eri4 = as_blocks(eri_mo, nocc)
        v_iajb = eri4.ovov.reshape(n_ov, n_ov)

        S_d = 4.0 * np.einsum('ijI,IJ,abJ->ijab', self.Voo, self.Pinv, self.Vvv,
                              optimize=True)
        Wd4 = eri4.oovv - S_d                              # (i,j,a,b)
        W_direct = Wd4.transpose(0, 2, 1, 3).reshape(n_ov, n_ov)

        d_qp = (self.eps_qp[virt][None, :] - self.eps_qp[occ][:, None]).ravel()
        self.A_bse = np.diag(d_qp) + self.kappa * v_iajb - W_direct

        if bse_tda:
            self.B_bse = None
            self.Omega, X = np.linalg.eigh(self.A_bse)
            self.X = X
            self.Y = np.zeros_like(X)
        else:
            S_s = 4.0 * np.einsum('ajI,IJ,biJ->ajbi', self.Vvo, self.Pinv,
                                  self.Vvo, optimize=True)
            Ws4 = eri4.vovo - S_s                          # (a,j,b,i)
            # element (ia, jb) of the swap kernel is Ws4[a, j, b, i]
            W_swap = np.einsum('ajbi->iajb', Ws4).reshape(n_ov, n_ov)
            self.B_bse = self.kappa * v_iajb - W_swap
            AmB = self.A_bse - self.B_bse
            L = np.linalg.cholesky(0.5 * (AmB + AmB.T))
            Mh = L.T @ (self.A_bse + self.B_bse) @ L
            w2, Zv = np.linalg.eigh(0.5 * (Mh + Mh.T))
            if w2.min() <= 0:
                raise ValueError("BSE instability: Omega^2 <= 0")
            self.Omega = np.sqrt(w2)
            XpY = (L @ Zv) / np.sqrt(self.Omega)[None, :]
            XmY = np.linalg.solve(L.T, Zv) * np.sqrt(self.Omega)[None, :]
            self.X = 0.5 * (XpY + XmY)
            self.Y = 0.5 * (XpY - XmY)
