"""The fragment-partitioned Tamm-Dancoff BSE: diabatic states, their effective
Hamiltonian, and the charge-transfer self-energy that dresses it.

In the fragment-localized orbitals of `src.Base.fragment_localization` every
electron-hole pair (i, a) is local to one fragment or moves charge from the
hole's fragment to the electron's. A DIABATIC STATE is the lowest eigenvector
(or several) of the BSE matrix restricted to one such block: a SITE state for
a local block (K, K), a CHARGE-TRANSFER diabat for a block (K, L). The chosen
diabats span P; Q is everything orthogonal to them -- the other charge-transfer
configurations and the higher local excitations alike -- and is eliminated
EXACTLY at a fixed energy Omega_0 (Feshbach / Loewdin partitioning):

    A_eff(Omega) = A_PP + A_PQ (Omega - A_QQ)^-1 A_QP ,
    Sigma(Omega) = A_PQ (Omega - A_QQ)^-1 A_QP .

A root of A_eff(Omega) c = Omega c with complete Q is an eigenvalue of the full
TDA-BSE, and its P weight is Z = [1 - c^T Sigma'(Omega) c]^-1; that is the
identity the tests check. The diabatic quantities a vibronic model needs are
A_eff(Omega_0) (site and charge-transfer energies on the diagonal, effective
couplings off it), its Coulomb-only part A_PP, and dA_eff/dOmega, which says
how far the energy-independent matrix is from exact.

THE RESOLVENT VECTORS. For each diabat p_a, y_a = (Omega_0 - A_QQ)^-1 Q A p_a.
Then Sigma_ab(Omega_0) = (Q A p_a)^T y_b, dSigma_ab/dOmega = -y_a^T y_b, and the
nuclear derivative of A_eff at fixed Omega_0 is u_a^T dA u_b with
u_a = p_a + y_a -- a derivative of the full matrix contracted with two fixed
vectors, which is what the excited-state reverse chain can take as a seed
(`src.gradients.fragment_diabatic`).

THE POLE GUARD. Omega_0 must lie below the lowest eigenvalue of A_QQ: then
A_QQ - Omega_0 is positive definite, the resolvent is a conjugate-gradient
solve, and Sigma has no pole between the diabats and Q. A charge-transfer
configuration that comes close to the site energies belongs in P as an
explicit diabat, never in Q; the partition refuses otherwise.

DENSE AND MATRIX-FREE ARE ONE CODE PATH. `BSEOperator` applies the canonical
TDA-BSE matrix A to a block of vectors, densely from `bse_blocks` for small
systems and through the ISDF block action of the Davidson solver otherwise;
everything here works through `apply`, rotating into and out of the local
basis on the way. Iteratively, block eigenpairs come from the shared symmetric
Davidson and the resolvent from conjugate gradients.

WHAT IS NOT HERE. No full-BSE (A, B) partition: the elimination is a Schur
complement only for a Hermitian problem. No nuclear derivatives: those are
`src.gradients.fragment_diabatic`, and their finite-difference reference is
`src.properties.diabatic`.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from src.Base.constants import (FRAGMENT_POLE_MARGIN, FRAGMENT_SOLVE_TOL,
                                FRAGMENT_DENSE_MAX)
from src.Base.sliced_factors import SlicedFactors
from src.gradients.bse_isdf import bse_blocks
from src.SingleReference.LinearResponse.davidson import isdf_block_action
from src.Solvers.davidson import solve_symmetric
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)


class BSEOperator:
    """y = A x for the canonical singlet TDA-BSE matrix, x of shape (n_ov, k).

    Built from a chain's `kernel_pieces`, so it is the same A that chain's
    roots and gradients use: the same quasiparticle energies, the same static
    screening, the same factors.
    """

    def __init__(self, nocc, eps_qp, apply, dense=None):
        self.nocc = int(nocc)
        self.eps_qp = np.asarray(eps_qp, float)
        self.nvir = len(self.eps_qp) - self.nocc
        self._apply = apply
        self.dense = dense

    @property
    def n_ov(self):
        return self.nocc * self.nvir

    def apply(self, x):
        x = np.asarray(x, float)
        vec = x.ndim == 1
        x = x.reshape(self.n_ov, -1)
        y = self._apply(x)
        return y[:, 0] if vec else y

    @classmethod
    def from_chain(cls, chain, mol=None, mf=None, route='auto'):
        """(operator, pieces) at one geometry; route 'auto' | 'dense' | 'isdf'."""
        mol, mf = chain.mean_field(mol, mf)
        pieces = chain.kernel_pieces(mol, mf)
        x_mo, d, eps_qp, w_aux = pieces[4], pieces[5], pieces[7], pieces[8]
        nocc = chain.nocc
        n_ov = nocc * (len(eps_qp) - nocc)
        if route == 'auto':
            route = 'dense' if n_ov <= FRAGMENT_DENSE_MAX else 'isdf'
        if route == 'dense':
            a = bse_blocks(x_mo, d, eps_qp, w_aux, nocc, spin=chain.spin,
                           bse_tda=True)[0]
            a = 0.5 * (a + a.T)
            return cls(nocc, eps_qp, lambda x: a @ x, dense=a), pieces
        if route != 'isdf':
            raise ValueError(f"route must be 'auto', 'dense' or 'isdf', got "
                             f"{route!r}")
        lr = LinearResponseSolver(np.asarray(eps_qp, float),
                                  spin_mode='restricted')
        factors = x_mo if isinstance(x_mo, SlicedFactors) else (x_mo, d)
        apply_ab = isdf_block_action(lr, nocc, True, w_aux, factors,
                                     spin=chain.spin)[0]
        nv = len(eps_qp) - nocc

        def apply(x):
            z = np.ascontiguousarray(x.T.reshape(-1, nocc, nv))
            return apply_ab(z)[0].reshape(-1, nocc * nv).T
        return cls(nocc, eps_qp, apply), pieces


def local_pair_diagonal(orbitals, eps_qp):
    """(n_ov,) the quasiparticle part of A_loc's diagonal, F_aa - F_ii in the
    local basis: the preconditioner and the Davidson seeds of every iterative
    solve on the partition."""
    eps_qp = np.asarray(eps_qp, float)
    n = orbitals.nocc
    f_oo = np.einsum('pi,p,pi->i', orbitals.u_occ, eps_qp[:n], orbitals.u_occ)
    f_vv = np.einsum('pa,p,pa->a', orbitals.u_vir, eps_qp[n:], orbitals.u_vir)
    return (f_vv[None, :] - f_oo[:, None]).ravel()


def _lowest(apply, diag, n, dim, tol):
    """(values, vectors) of the n lowest eigenpairs of a symmetric operator:
    dense below FRAGMENT_DENSE_MAX, the shared symmetric Davidson above."""
    if dim <= FRAGMENT_DENSE_MAX:
        m = apply(np.eye(dim))
        w, v = np.linalg.eigh(0.5 * (m + m.T))
        return w[:n], v[:, :n]
    e, x, conv = solve_symmetric(
        lambda v: apply(np.reshape(v, (dim, 1)))[:, 0], np.asarray(diag, float),
        nroots=n, tol_residual=tol, max_cycle=500,
        label='fragment diabats')
    if not np.all(conv):
        raise RuntimeError(f'block eigenpairs did not converge to {tol}')
    return e, x


@dataclass
class FragmentPartition:
    """Diabats, A_eff(Omega_0), and the vectors its derivative needs.

    sites: {fragment index: number of site states}; ct: {(hole fragment,
    electron fragment): number of charge-transfer diabats}. Diabats are
    ordered sites first (by fragment), then charge transfer, each block's
    states by energy; `labels` names them.
    """
    operator: BSEOperator
    orbitals: object
    omega0: float
    labels: list
    p_local: np.ndarray            # (n_ov, n_p) diabats, local basis
    y_local: np.ndarray            # (n_ov, n_p) resolvent vectors, local basis
    a_pp: np.ndarray               # (n_p, n_p) direct (Coulomb-only) part
    sigma: np.ndarray              # (n_p, n_p) Sigma(Omega_0)
    dsigma: np.ndarray             # (n_p, n_p) dSigma/dOmega at Omega_0
    q_lowest: float                # lowest eigenvalue of A_QQ
    block_energies: list = field(default_factory=list)
    gaps: np.ndarray = None        # each diabat's gap to its block neighbours

    @property
    def a_eff(self):
        return self.a_pp + self.sigma

    @property
    def u_local(self):
        """u_a = p_a + y_a: the nuclear derivative of A_eff_ab is u_a^T dA u_b."""
        return self.p_local + self.y_local

    def u_canonical(self):
        return self.orbitals.to_canonical(self.u_local)

    def p_canonical(self):
        return self.orbitals.to_canonical(self.p_local)

    @classmethod
    def build(cls, operator, orbitals, sites, ct=None, omega0=None,
              tol=FRAGMENT_SOLVE_TOL, margin=FRAGMENT_POLE_MARGIN):
        ct = {} if ct is None else dict(ct)
        if orbitals.nocc != operator.nocc:
            raise ValueError('orbitals and operator disagree on nocc')
        hole, elec = orbitals.pair_labels()
        n_ov = operator.n_ov

        def a_loc(x):
            return orbitals.to_local(operator.apply(orbitals.to_canonical(x)))

        diag = local_pair_diagonal(orbitals, operator.eps_qp)

        blocks = [((k, k), n, f'site {k}') for k, n in sorted(sites.items())]
        blocks += [((k, l), n, f'ct {k}->{l}') for (k, l), n in
                   sorted(ct.items())]
        p_cols, labels, block_energies, gaps = [], [], [], []
        for (k, l), n, name in blocks:
            idx = np.flatnonzero((hole == k) & (elec == l))
            if idx.size < n:
                raise ValueError(f'{name}: block has {idx.size} pairs, '
                                 f'{n} states asked for')

            def block_apply(x, idx=idx):
                full = np.zeros((n_ov, x.shape[1]))
                full[idx] = x
                return a_loc(full)[idx]
            n_get = min(n + 1, idx.size)
            w, v = _lowest(block_apply, diag[idx], n_get, idx.size, tol)
            for s in range(n):
                others = np.delete(w, s)
                gaps.append(float(np.abs(others - w[s]).min())
                            if others.size else np.inf)
                col = np.zeros(n_ov)
                col[idx] = v[:, s] * np.sign(v[np.abs(v[:, s]).argmax(), s])
                p_cols.append(col)
                labels.append(f'{name}.{s}')
            block_energies.append((name, w[:n]))
        p = np.array(p_cols).T
        ap = a_loc(p)
        a_pp = p.T @ ap
        a_pp = 0.5 * (a_pp + a_pp.T)
        if omega0 is None:
            omega0 = float(np.diag(a_pp)[:sum(sites.values())].mean())

        def project(x):
            return x - p @ (p.T @ x)

        def qaq(x):
            return project(a_loc(project(x)))
        # the lowest eigenvalue of A_QQ: P directions pushed far up
        big = float(np.abs(diag).max())

        def qaq_guarded(x):
            return qaq(x) + big * (p @ (p.T @ x))
        # a guard, not a result: 1e-7 Ha is ample against the margin
        q_low = float(_lowest(qaq_guarded,
                              diag + big * (p ** 2).sum(axis=1), 1, n_ov,
                              max(tol, 1e-7))[0][0])
        if not omega0 < q_low - margin:
            raise ValueError(
                f'Omega_0 = {omega0:.6f} Ha is not below the lowest eigenvalue '
                f'of A_QQ ({q_low:.6f} Ha) by the margin {margin}: Sigma has '
                f'a pole there. Make the configuration that comes close an '
                f'explicit diabat (sites / ct) or lower Omega_0.')

        rhs = -project(ap)                          # (A_QQ - Omega_0) y = -Q A p
        prec_d = np.maximum(diag - omega0, 1e-3)
        y = np.zeros_like(rhs)
        if n_ov <= FRAGMENT_DENSE_MAX:
            m = qaq(np.eye(n_ov)) - omega0 * project(np.eye(n_ov)) \
                + p @ p.T
            y = np.linalg.solve(0.5 * (m + m.T), rhs)
            y = project(y)
        else:
            op = LinearOperator(
                (n_ov, n_ov), dtype=float,
                matvec=lambda x: qaq(np.reshape(x, (n_ov, 1)))[:, 0]
                - omega0 * project(np.reshape(x, (n_ov, 1)))[:, 0])
            prec = LinearOperator((n_ov, n_ov), dtype=float,
                                  matvec=lambda x: project(
                                      (np.ravel(x) / prec_d)[:, None])[:, 0])
            for c in range(rhs.shape[1]):
                sol, info = cg(op, rhs[:, c], M=prec, rtol=tol, maxiter=2000)
                if info != 0:
                    raise RuntimeError(f'resolvent solve for diabat {c} did '
                                       f'not converge (info={info})')
                y[:, c] = project(sol[:, None])[:, 0]
        sigma = -rhs.T @ y                          # (Q A p_a)^T y_b
        sigma = 0.5 * (sigma + sigma.T)
        dsigma = -(y.T @ y)
        return cls(operator=operator, orbitals=orbitals, omega0=float(omega0),
                   labels=labels, p_local=p, y_local=y, a_pp=a_pp,
                   sigma=sigma, dsigma=dsigma, q_lowest=q_low,
                   block_energies=block_energies, gaps=np.array(gaps))


class FeshbachOracle:
    """Dense reference: Sigma(Omega), roots of A_eff(Omega) c = Omega c, Z.

    A (n, n) and an orthonormal P basis (n, n_p) in the same basis; Q is the
    orthogonal complement. Small systems only -- it diagonalizes A_QQ.
    """

    def __init__(self, a, p):
        a = 0.5 * (np.asarray(a, float) + np.asarray(a, float).T)
        p = np.asarray(p, float)
        q_full = np.linalg.svd(np.eye(len(a)) - p @ p.T)[0]
        self.q = q_full[:, :len(a) - p.shape[1]]
        self.a, self.p = a, p
        self.a_pp = p.T @ a @ p
        self.a_pq = p.T @ a @ self.q
        self.e_q, self.v_q = np.linalg.eigh(self.q.T @ a @ self.q)
        self.c = self.a_pq @ self.v_q               # couplings to Q eigenstates

    def sigma(self, omega):
        return (self.c / (omega - self.e_q)) @ self.c.T

    def dsigma(self, omega):
        return -(self.c / (omega - self.e_q) ** 2) @ self.c.T

    def root(self, guess, tol=1e-13, maxiter=100):
        """(Omega, c, Z) of A_eff(Omega) c = Omega c by Newton, near `guess`."""
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
        z = 1.0 / (1.0 - float(c @ self.dsigma(om) @ c))
        return om, c, z
