"""BSE@G0W0 on the quasi-boson route, in the four Toelle/Kitsaras/Loos variants.

Variants (arXiv:2507.02160 Table I): screening in {rpa, tda} (both the G0W0
stage and the BSE kernel), BSE eigenproblem in {full, tda}.

Static kernel in closed form,

    W = v - 4 V P^{-1} V^T,   P = A_mf + B_mf (RPA screening) or A_mf (TDA),

with no t dependence: the screening is the mean-field response, and the
amplitudes only enter through the quasiparticle energies on the BSE diagonal.
Those come from the diagonal quasi-boson EOM solve of
`GW.quasi_boson.QPqb` -- the root of largest pole strength, or, from a seed,
the root that follows it -- unless filtered on pole strength (the mean-field
energy kept, warned and flagged, for a state whose root is a satellite), while
the screening keeps mean-field energies: the standard G0W0-BSE split.

kappa weights the bare exchange (ia|jb) in A and in B and nothing else -- 2 for
a singlet, 0 for a triplet. W, the quasiparticle energies and every quasi-boson
normalization are spin-independent.

The reverse mode -- the exact (F, ERI, t) partials of Omega_nu by
Hellmann-Feynman on (X, Y), the per-quasiparticle-state chain and the P^{-1}
screening response -- is `gradients.gwbse.quasi_boson_adjoint.BSEqbAdjoint`.
"""
import numpy as np

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import KAPPA, QP_WINDOW_Z_MIN
from src.Base.orbital_spaces import ov_energies
from src.SingleReference.common.eri_blocks import as_blocks
from src.Base.utils.parallel.threads import openmp_threads, row_map
from src.SingleReference.GW.quasi_boson import QPqb
from src.SingleReference.LinearResponse.casida import casida_eigensolve

class QPWindowMeanField(RuntimeWarning):
    """An orbital of the quasiparticle set whose root has a pole strength at or
    below QP_WINDOW_Z_MIN keeps its mean-field energy on the BSE diagonal."""


class BSEqb:
    """BSE@G0W0 eigenpairs on the dense quasi-boson route."""

    #: The quasiparticle solver the BSE diagonal is dressed with. The reverse
    #: mode overrides it, so an object reaching a gradient carries the boson
    #: Frechet maps its chain rule needs.
    quasiparticle = QPqb

    def __init__(self, mf, eri_mo, nocc, screening='rpa', bse_tda=False,
                 qp_orbs='all', seeds=None, seed_labels=None, pinned=False,
                 delta=None, spin='singlet', blocks=()):
        """delta: the static <p|Sigma_x - v_xc|p> shift; see `QPqb`. It reaches
        the BSE through `eps_qp` alone, which is where the quasiparticle
        energies enter the kernel.

        spin: 'singlet' (kappa = 2) or 'triplet' (kappa = 0). Kappa weights the
        bare exchange (ia|jb) in A and in B, and nothing else: the screened term
        W, the quasiparticle energies and every quasi-boson normalization are
        spin-independent. The gradient needs no separate treatment because the
        one place kappa appears there, on Gamma4[o,v,o,v], is the derivative of
        those two terms and carries the same factor.

        seeds: {p: w0}, roots decided at another geometry. Each listed orbital's
        diagonal carries `QPqb.solve_diag`: without a seed the root of largest
        pole strength at this geometry, with one the root in the interval
        between the poles that bracket the seed, the root followed.
        seed_labels: {p: (lo, hi)}, the labels of the poles that bracketed each
        seed at its geometry (`qp_labels` there): the seeded root is followed
        between those same poles wherever they have moved, not into the
        interval that holds the seed's energy here.

        blocks: tuples of solved orbitals that are near-degenerate; each block
        carries ONE correction, the mean of its members' eps^QP_p - eps_p, on
        its members' own eps_p. Diagonal G0W0 gives each index its own root,
        which two orbitals rotating into each other across an avoided crossing
        trade within a fraction of the gap's geometric scale.
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
        self.qp_z = {}
        # the labels of the poles bracketing each kept root, which a walk
        # passes back as seed_labels
        self.qp_labels = {}
        seeds = seeds or {}
        seed_labels = seed_labels or {}
        # the unseeded orbitals' largest-weight roots over the threads, the
        # seeded ones followed
        solved = largest_weight_roots(
            self.qp, [p for p in self.qp_orbs if seeds.get(p) is None])
        solved.update({p: self.qp.solve_diag(p, w0=seeds[p],
                                             labels=seed_labels.get(p))
                       for p in self.qp_orbs if seeds.get(p) is not None})
        # the followed roots kept on a branch the largest weight has left
        # (`QPqb.solve_diag`), for the caller's record
        self.qp_branch_log = list(self.qp.branch_log)
        # the roots of largest weight kept with Z outside (0, 1]
        # (`QPqb.quasiparticle_root`), in the set's order whatever thread found them
        place = {p: k for k, p in enumerate(self.qp_orbs)}
        self.qp_z_outside = sorted(self.qp.z_outside, key=lambda r: place[r['orbital']])
        for p in self.qp_orbs:
            w, Z = solved[p]
            if pinned or Z > QP_WINDOW_Z_MIN:
                self.eps_qp[p] = w
                self.qp_roots[p] = w
                self.qp_z[p] = Z
                self.qp_labels[p] = self.qp.root_labels(p, w)
            else:
                flag('qp_window_mean_field',
                     f'orbital {p}: the quasiparticle root {w:.8f} Ha has pole '
                     f'strength {Z:.4f}, not above {QP_WINDOW_Z_MIN}; the BSE '
                     f'diagonal keeps its mean-field energy', QPWindowMeanField)
        self.qp_blocks = tuple(tuple(int(p) for p in b) for b in blocks)
        for b in self.qp_blocks:
            if any(p not in self.qp_roots for p in b):
                raise ValueError(f'block {b} shares one correction, so every '
                                 f'member needs a solved root')
            b = list(b)
            self.eps_qp[b] = eps[b] + np.mean(self.eps_qp[b] - eps[b])

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
        eri4 = self.eri
        v_iajb = eri4.ovov.reshape(n_ov, n_ov)

        S_d = 4.0 * np.einsum('ijI,IJ,abJ->ijab', self.Voo, self.Pinv, self.Vvv,
                              optimize=True)
        Wd4 = eri4.oovv - S_d                              # (i,j,a,b)
        W_direct = Wd4.transpose(0, 2, 1, 3).reshape(n_ov, n_ov)

        d_qp = ov_energies(self.eps_qp, nocc_)
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
            self.Omega, self.X, self.Y, _, _ = casida_eigensolve(
                [self.A_bse + self.B_bse, self.A_bse - self.B_bse], 'amb',
                threshold=np.inf)


def largest_weight_roots(qp, orbitals):
    """{p: (w*, Z)} of `QPqb.quasiparticle_root` over `orbitals`, on the
    process's threads.

    Each orbital's search is element-wise numpy over its own poles, so the
    orbitals are split over threads (`row_map`) and every root is the bits the
    serial search makes. The order is dealt round-robin, which spreads the
    continuum virtuals, the slowest searches, over the threads.
    """
    threads = max(1, min(openmp_threads(), len(orbitals)))
    dealt = [orbitals[i] for t in range(threads)
             for i in range(t, len(orbitals), threads)]
    parts = row_map(lambda r0, r1: [(p, qp.quasiparticle_root(p))
                                    for p in dealt[r0:r1]],
                    len(dealt), threads)
    return dict(pair for part in parts for pair in part)
