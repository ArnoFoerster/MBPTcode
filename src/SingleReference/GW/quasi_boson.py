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
import warnings

import numpy as np

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import QP_WINDOW_Z_MIN
from src.SingleReference.LinearResponse.quasi_boson_rpa import (RPA,
                                                                build_rpa_AB,
                                                                couplings_V)
from src.Solvers.qp_equation import (bracket_labels, bracketed_root,
                                     largest_weight_root)


class QPLowWeightRoot(RuntimeWarning):
    """A root followed from a seed whose pole strength is at or below QP_WINDOW_Z_MIN."""


class QPqb:
    """Diagonal-mode quasi-boson EOM G0W0 quasiparticle energies.

    The eta = 0 quasiparticle equation exactly, by the bracketed root search on the closed-form
    Sigma_pp: no broadening, no analytic continuation and no residue set. Z is
    the exact slope of the equation that was solved.

    The reverse mode is `gradients.gwbse.quasi_boson_adjoint.QPqbAdjoint`, which
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
        # each followed root the largest weight has left, kept on its branch
        # (`solve_diag`): {orbital, w, z, w0, largest, labels}
        self.branch_log = []
        # each root of largest weight kept with Z outside (0, 1]
        # (`quasiparticle_root`): {orbital, w, z}
        self.z_outside = []

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

    def solve_diag(self, p, w0=None, labels=None):
        """(w*, Z) of w = eps_p + delta_p + Sigma_pp(w), solved exactly over
        every root: the largest-weight one (`quasiparticle_root`), or with a
        seed w0 a reference root followed (`bracketed_root`): between the
        poles labelled `labels` (`root_labels` at the reference) wherever they
        have moved, else between the poles bracketing w0 here. The labels
        keep the branch where a pole drifts across w0, which the interval
        holding w0 would trade for a satellite. Z <= QP_WINDOW_Z_MIN is a lost QP. A
        followed root the largest weight has left is kept on its branch,
        warned about and logged in `branch_log` (refuse inputs, warn numerics)."""
        if w0 is None:
            return self.quasiparticle_root(p)
        num, pol, a = self._all_poles(p)
        seen = {}
        w, z = bracketed_root(num, pol, a, w0, labels=labels, out=seen)
        if seen.get('flipped'):
            largest = seen.get('largest')
            self.branch_log.append({
                'orbital': int(p), 'w': w, 'z': z, 'w0': float(w0),
                'largest': (None if largest is None
                            else [float(x) for x in largest]),
                'labels': seen.get('labels')})
            warnings.warn(
                f'quasi-boson orbital {p}: the root {w:.10f} Ha followed from '
                f'{float(w0):.10f} Ha has Z = {z:.4f}, the largest weight is '
                f'elsewhere ({largest}); the branch is kept and logged in '
                f'branch_log', RuntimeWarning, stacklevel=2)
        return w, z

    def root_labels(self, p, w):
        """(lo, hi): the labels of the poles bracketing the root w of orbital
        p, the branch `solve_diag` follows at another geometry."""
        return bracket_labels(*self._all_poles(p), w)

    def quasiparticle_root(self, p):
        """(w*, Z) of the root of w = eps_p + Sigma_pp(w) carrying the largest Z.

        Every root is an eigenvalue of the arrowhead `supermatrix_diag(p)` and
        Z its weight on the bare orbital, so the weights sum to one and the
        quasiparticle is decidable from the exact poles alone: no seed, no
        starting point, the same root at every geometry the weights single it
        out at. A local search from eps_p reaches whichever root is nearest, a
        satellite with Z ~ 1e-3 for formaldehyde/cc-pVDZ's virtuals above
        38 eV. Where two roots share the weight the choice changes branch;
        nothing in Sigma_pp distinguishes them there.
        """
        nh, ph, npp, pp = self._pole_arrays(p)
        num = np.concatenate([nh.ravel(), npp.ravel()])
        pol = np.concatenate([ph.ravel(), pp.ravel()])
        seen = {}
        w, z = largest_weight_root(num, pol, self.eps[p] + self.delta[p],
                                   label=f'orbital {p}', out=seen)
        if seen['z_outside']:
            self.z_outside.append({'orbital': int(p), 'w': w, 'z': z})
        return w, z

    def _all_poles(self, p):
        """(num, pol, a) of orbital p's secular equation, every pole kept."""
        nh, ph, npp, pp = self._pole_arrays(p)
        return (np.concatenate([nh.ravel(), npp.ravel()]),
                np.concatenate([ph.ravel(), pp.ravel()]),
                self.eps[p] + self.delta[p])

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


def qp_energy_general(epsF, eri, nocc, p, screening='rpa', w0=None):
    """Diagonal-mode QP energy for a general (non-diagonal) Fock matrix.

    Dense supermatrix in the boson original basis with full F blocks -- the
    exact functional the gradient Lagrangian differentiates; used by the finite
    difference gates. It carries no static <Sigma_x - v_xc> shift, so on a
    Kohn-Sham reference it is not the same quantity as `QPqb.solve_diag`.

    The satellite blocks are diagonalized for the poles of Sigma_pp and its
    couplings to them, and the root comes from the one finder of
    `Solvers.qp_equation`: the largest weight, or with w0 the root in the
    interval between the poles that bracket it (`bracketed_root`), kept with
    a warning at a weight no quasiparticle has.
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
    Wh = W[p, :nocc, :].ravel()
    Wp = W[p, nocc:, :].ravel()
    Hh = (np.einsum('jk,IJ->jIkJ', epsF[np.ix_(occ, occ)], np.eye(n_ov))
          - np.einsum('jk,IJ->jIkJ', np.eye(nocc), Abar)).reshape(nocc * n_ov, nocc * n_ov)
    Hp = (np.einsum('bc,IJ->bIcJ', epsF[np.ix_(virt, virt)], np.eye(n_ov))
          + np.einsum('bc,IJ->bIcJ', np.eye(nvirt), Abar)).reshape(nvirt * n_ov, nvirt * n_ov)
    num, pol = [], []
    for block, coupling in ((Hh, Wh), (Hp, Wp)):
        lam, U = np.linalg.eigh(block)
        num.append((U.T @ coupling) ** 2)
        pol.append(lam)
    num, pol = np.concatenate(num), np.concatenate(pol)
    if w0 is None:
        return largest_weight_root(num, pol, epsF[p, p],
                                   label=f'orbital {p}')[0]
    w, z = bracketed_root(num, pol, epsF[p, p], w0)
    if not z > QP_WINDOW_Z_MIN:
        flag('qp_followed_root_low_weight',
             f'the root of orbital {p} that follows {w0} has weight {z:.4f}, '
             f'not above {QP_WINDOW_Z_MIN}: kept on its branch', QPLowWeightRoot)
    return w
