"""The fragment-partitioned BSE, Tamm-Dancoff or full: diabatic states, their
effective Hamiltonian, and the charge-transfer self-energy that dresses it.

In the fragment-localized orbitals of `src.Base.fragment_localization` every
electron-hole pair (i, a) is local to one fragment or moves charge from the
hole's fragment to the electron's. A DIABATIC STATE is the lowest eigenvector
(or several) of the BSE restricted to one such block: a SITE state for a local
block (K, K), a CHARGE-TRANSFER diabat for a block (K, L). The chosen diabats
span P; Q is everything else -- the other charge-transfer configurations and
the higher local excitations alike -- and is eliminated EXACTLY at a fixed
energy Omega_0 (Feshbach / Loewdin partitioning).

ONE FORM FOR BOTH KERNELS. The BSE is written as a symmetric pencil,

    K v = Omega S v ,

with K = A and S = 1 in the Tamm-Dancoff approximation, and for the full BSE
v = (X, Y), K = [[A, B], [B, A]] and the metric S = diag(1, -1). K is
positive definite for a stable reference, and the diabats are normalized in
the metric, v^T S v = 1 (X^T X - Y^T Y = 1). Q is the complement of P that
is ORTHOGONAL IN THE METRIC (P^T S Q = 0); then the metric has no P-Q block,
the elimination is a Schur complement of the symmetric matrix K - Omega S,
and

    A_eff(Omega) = K_PP + Sigma(Omega) ,
    Sigma(Omega) = -K_PQ (K_QQ - Omega S_QQ)^-1 K_QP .

A root of A_eff(Omega) c = Omega c with complete Q is an eigenvalue of the full
problem, and its P weight is Z = [1 - c^T Sigma'(Omega) c]^-1; that is the
identity the tests check. For the full BSE, Q contains the diabats'
de-excitation partners (Y, X) as well, so A_eff stays one (n_p, n_p) matrix of
excitations, with the partners folded into Sigma at Omega_0; in the
Tamm-Dancoff limit the partners decouple and A_eff is
A_PP + A_PQ (Omega - A_QQ)^-1 A_QP. The diabatic quantities a vibronic model
needs are A_eff(Omega_0) (site and charge-transfer energies on the diagonal,
effective couplings off it), its Coulomb-only part K_PP, and dA_eff/dOmega,
which says how far the energy-independent matrix is from exact.

THE RESOLVENT VECTORS. For each diabat p_a, y_a in Q solves
(K - Omega_0 S) y_a = -K p_a projected on Q. Then Sigma_ab(Omega_0) =
p_a^T K y_b, dSigma_ab/dOmega = -y_a^T S y_b, and the nuclear derivative of
A_eff at fixed Omega_0 is u_a^T dK u_b with u_a = p_a + y_a -- a derivative
of the full matrix contracted with two fixed vectors, which is what the
excited-state reverse chain can take as a seed
(`src.gradients.fragment_diabatic`).

THE POLE GUARD. Omega_0 must lie below the lowest positive eigenvalue of the
pencil on Q: then K_QQ - Omega_0 S_QQ is positive definite (it is at
Omega = 0, and turns singular first there), the resolvent is a
conjugate-gradient solve, and Sigma has no pole between the diabats and Q. A
charge-transfer configuration that comes close to the site energies belongs in
P as an explicit diabat, never in Q; the partition refuses otherwise.

DENSE AND MATRIX-FREE ARE ONE CODE PATH. `BSEOperator` applies K to a block of
vectors, densely from `bse_blocks` for small systems and through the ISDF
block action of the Davidson solver otherwise; everything here works through
`apply`, rotating into and out of the local basis on the way. Above
`dense_limit` rows, Tamm-Dancoff block eigenpairs come from the shared
symmetric Davidson, full-BSE ones from a Davidson on the (A + B, A - B) pair,
and the resolvent from conjugate gradients.

WHICH KERNEL is the chain's: `ExcitedStateChain(bse_tda=...)`.

WHAT IS NOT HERE. No nuclear derivatives: those are
`src.gradients.fragment_diabatic`, and their finite-difference reference is
`src.properties.diabatic`.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh, null_space

from src.Base.constants import (FRAGMENT_DAVIDSON_EXTRA_ROOTS,
                                FRAGMENT_DAVIDSON_MIN_SPACE,
                                FRAGMENT_GUARD_NEWTON_TOL,
                                FRAGMENT_POLE_MARGIN, FRAGMENT_SOLVE_TOL,
                                FRAGMENT_DENSE_MAX,
                                FRAGMENT_MATRIX_FREE_DENSE_MAX)
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.krylov import root_driven_solve
from src.Base.utils.mpi_grid import lockstep
from src.gradients.bse_isdf import bse_blocks, bse_solve
from src.SingleReference.LinearResponse.davidson import isdf_block_action
from src.Solvers.davidson import solve_symmetric
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)


class BSEOperator:
    """y = K x for the canonical singlet BSE, x of shape (dim, k).

    Tamm-Dancoff (`tda=True`): K = A and dim = n_ov. Full BSE: x stacks the
    (X, Y) halves, K = [[A, B], [B, A]] and dim = 2 n_ov. Built from a chain's
    `kernel_pieces`, so it is the same kernel that chain's roots and gradients
    use: the same quasiparticle energies, the same static screening, the same
    factors.
    """

    def __init__(self, nocc, eps_qp, apply, dense=None, tda=True):
        self.nocc = int(nocc)
        self.eps_qp = np.asarray(eps_qp, float)
        self.nvir = len(self.eps_qp) - self.nocc
        self._apply = apply
        self.dense = dense
        self.tda = bool(tda)

    @property
    def n_ov(self):
        return self.nocc * self.nvir

    @property
    def dim(self):
        return self.n_ov if self.tda else 2 * self.n_ov

    @property
    def metric(self):
        """(dim,) the diagonal of S: ones, and -1 on a full BSE's Y half."""
        one = np.ones(self.n_ov)
        return one if self.tda else np.concatenate([one, -one])

    def apply(self, x):
        x = np.asarray(x, float)
        vec = x.ndim == 1
        # the ISDF action is collective: every rank hands it rank 0's vectors
        x = lockstep(np.ascontiguousarray(x.reshape(self.dim, -1)), check=True)
        y = self._apply(x)
        return y[:, 0] if vec else y

    @classmethod
    def from_chain(cls, chain, mol=None, mf=None, route='auto'):
        """(operator, pieces) at one geometry; route 'auto' | 'dense' | 'isdf'.
        The kernel, Tamm-Dancoff or full, is the chain's (`bse_tda`)."""
        mol, mf = chain.mean_field(mol, mf)
        pieces = chain.kernel_pieces(mol, mf)
        x_mo, d, eps_qp, w_aux = pieces[4], pieces[5], pieces[7], pieces[8]
        nocc, tda = chain.nocc, bool(chain.bse_tda)
        nv = len(eps_qp) - nocc
        n_ov = nocc * nv
        if route == 'auto':
            route = ('dense' if (n_ov if tda else 2 * n_ov) <= FRAGMENT_DENSE_MAX
                     else 'isdf')
        if route == 'dense':
            a, b = bse_blocks(x_mo, d, eps_qp, w_aux, nocc, spin=chain.spin,
                              bse_tda=tda)[:2]
            k = 0.5 * (a + a.T)
            if not tda:
                b = 0.5 * (b + b.T)
                k = np.block([[k, b], [b, k]])
            return cls(nocc, eps_qp, lambda x: k @ x, dense=k, tda=tda), pieces
        if route != 'isdf':
            raise ValueError(f"route must be 'auto', 'dense' or 'isdf', got "
                             f"{route!r}")
        lr = LinearResponseSolver(np.asarray(eps_qp, float),
                                  spin_mode='restricted')
        factors = x_mo if isinstance(x_mo, SlicedFactors) else (x_mo, d)
        apply_ab = isdf_block_action(lr, nocc, True, w_aux, factors,
                                     spin=chain.spin)[0]

        def apply(x):
            k = x.shape[1]
            cols = x if tda else np.hstack([x[:n_ov], x[n_ov:]])
            z = np.ascontiguousarray(cols.T.reshape(-1, nocc, nv))
            az, bz = apply_ab(z)[:2]
            az = np.asarray(az).reshape(-1, n_ov).T
            if tda:
                return az
            bz = np.asarray(bz).reshape(-1, n_ov).T
            return np.vstack([az[:, :k] + bz[:, k:], bz[:, :k] + az[:, k:]])
        return cls(nocc, eps_qp, apply, tda=tda), pieces


def dense_limit(operator):
    """Rows up to which a block of `operator` is solved densely:
    FRAGMENT_DENSE_MAX with the matrix at hand, FRAGMENT_MATRIX_FREE_DENSE_MAX
    matrix-free, where forming a block costs one action per row."""
    return (FRAGMENT_DENSE_MAX if operator.dense is not None
            else FRAGMENT_MATRIX_FREE_DENSE_MAX)


def _halves(rotate, orbitals, v):
    v = np.asarray(v, float)
    n_ov = orbitals.nocc * orbitals.nvir
    if v.shape[0] == n_ov:
        return rotate(v)
    if v.shape[0] != 2 * n_ov:
        raise ValueError(f'{v.shape[0]} rows is neither n_ov = {n_ov} nor '
                         f'2 n_ov')
    return np.vstack([rotate(v[:n_ov]), rotate(v[n_ov:])])


def to_local(orbitals, v):
    """(dim, k) canonical -> local, a full-BSE vector's X and Y halves alike
    (both carry the pair index ia and rotate the same way)."""
    return _halves(orbitals.to_local, orbitals, v)


def to_canonical(orbitals, v):
    """(dim, k) local -> canonical; the inverse of `to_local`."""
    return _halves(orbitals.to_canonical, orbitals, v)


def split_xy(v, n_ov):
    """(X, Y) of a stacked vector or block; Y = 0 for a Tamm-Dancoff one."""
    v = np.asarray(v, float)
    if v.shape[0] == n_ov:
        return v, np.zeros_like(v)
    return v[:n_ov], v[n_ov:]


class FragmentCanonical:
    """The partition's working basis: each fragment's local occupied and
    virtual orbitals rotated among themselves to diagonalize the
    quasiparticle Fock block.

    The Fock matrix of Pipek-Mezey orbitals is far from diagonal inside a
    fragment, so F_aa - F_ii there is a poor Davidson seed and
    preconditioner. Rotations inside one fragment change no diabatic quantity
    and map every pair block onto itself, so the partition is solved here, on
    the exact quasiparticle diagonal e_a - e_i, and its vectors are returned
    in the local basis.
    """

    def __init__(self, orbitals, eps_qp):
        eps_qp = np.asarray(eps_qp, float)
        n = orbitals.nocc
        self.nocc, self.nvir = n, orbitals.nvir
        self.r_occ, self.e_occ = self._rotate(orbitals.u_occ, eps_qp[:n],
                                              orbitals.occ_labels)
        self.r_vir, self.e_vir = self._rotate(orbitals.u_vir, eps_qp[n:],
                                              orbitals.vir_labels)

    @staticmethod
    def _rotate(u, eps, labels):
        f = u.T @ (eps[:, None] * u)
        r = np.zeros_like(f)
        e = np.zeros(len(f))
        for k in np.unique(labels):
            idx = np.flatnonzero(labels == k)
            w, v = np.linalg.eigh(f[np.ix_(idx, idx)])
            r[np.ix_(idx, idx)] = v
            e[idx] = w
        return r, e

    def diagonal(self):
        """(n_ov,) e_a - e_i: the quasiparticle part of A, exactly."""
        return (self.e_vir[None, :] - self.e_occ[:, None]).ravel()

    def _pairs(self, x, a, b):
        n_ov = self.nocc * self.nvir
        out = []
        for h in range(x.shape[0] // n_ov):
            z = x[h * n_ov:(h + 1) * n_ov].reshape(self.nocc, self.nvir, -1)
            out.append(np.einsum('ij,jbk,ab->iak', a, z, b,
                                 optimize=True).reshape(n_ov, -1))
        return np.vstack(out)

    def to_local(self, x):
        """(dim, k) working -> local, a full-BSE vector's halves alike."""
        return self._pairs(np.asarray(x, float), self.r_occ, self.r_vir)

    def from_local(self, x):
        """(dim, k) local -> working; the transpose of `to_local`."""
        return self._pairs(np.asarray(x, float), self.r_occ.T, self.r_vir.T)


def _lowest(apply, diag, n, dim, tol, x0=None, dense_max=FRAGMENT_DENSE_MAX):
    """(values, vectors) of the n lowest eigenpairs of a symmetric operator:
    dense up to `dense_max` rows (`dense_limit`), above it the shared
    symmetric Davidson, started from `x0` when given and root-driven over
    ranks (`root_driven_solve`)."""
    if dim <= dense_max:
        m = apply(np.eye(dim))
        w, v = np.linalg.eigh(0.5 * (m + m.T))
        return w[:n], v[:, :n]

    def solve(act):
        e, x, conv = solve_symmetric(
            lambda v: act(np.reshape(v, (dim, 1)))[:, 0],
            np.asarray(diag, float), nroots=n, x0=x0, tol_residual=tol,
            max_cycle=500, label='fragment diabats')
        if not np.all(conv):
            raise RuntimeError(f'block eigenpairs did not converge to {tol}')
        return np.asarray(e, float), np.ascontiguousarray(x)
    return root_driven_solve(solve, apply, dim)


def _lowest_pencil(apply, diag, n, m, tol, dense_max=FRAGMENT_DENSE_MAX):
    """(values, vectors) of the n lowest positive roots of a full BSE on m
    pairs; `apply` acts on stacked (2m, k) blocks, `diag` (m,) estimates A's
    diagonal, and the vectors come stacked (X, Y) with X^T X - Y^T Y = 1.
    Dense up to `dense_max` rows (`dense_limit`), `_casida_davidson` above,
    root-driven over ranks (`root_driven_solve`)."""
    if 2 * m <= dense_max:
        k = apply(np.eye(2 * m))
        k = 0.5 * (k + k.T)
        w, x, y = bse_solve(k[:m, :m], k[:m, m:])
        return w[:n], np.vstack([x[:, :n], y[:, :n]])
    return root_driven_solve(
        lambda act: _casida_davidson(act, diag, n, m, tol), apply, 2 * m)


def _batched_cg(apply, rhs, prec, tol, maxiter=2000):
    """Conjugate gradients on every column of `rhs` at once, one block action
    per iteration. A column stops once its residual is below tol |rhs|; the
    stopping mask is rank 0's (`lockstep`), so every rank applies the
    operator to the same columns."""
    x = np.zeros_like(rhs)
    r = rhs.copy()
    z = prec(r)
    p = z.copy()
    rz = np.einsum('ij,ij->j', r, z)
    bnorm = np.linalg.norm(rhs, axis=0)
    bnorm[bnorm == 0.0] = 1.0
    active = lockstep(np.linalg.norm(r, axis=0) > tol * bnorm, check=True)
    for _ in range(maxiter):
        if not active.any():
            return x
        idx = np.flatnonzero(active)
        ap = apply(p[:, idx])
        alpha = rz[idx] / np.einsum('ij,ij->j', p[:, idx], ap)
        x[:, idx] += p[:, idx] * alpha
        r[:, idx] -= ap * alpha
        znew = prec(r[:, idx])
        rz_new = np.einsum('ij,ij->j', r[:, idx], znew)
        p[:, idx] = znew + p[:, idx] * (rz_new / rz[idx])
        rz[idx] = rz_new
        active = lockstep(np.linalg.norm(r, axis=0) > tol * bnorm, check=True)
    raise RuntimeError(f'resolvent solve did not converge in {maxiter} '
                       f'iterations ({int(active.sum())} columns left)')


def _casida_davidson(apply, diag, n, m, tol, max_cycle=500):
    """The n lowest roots of a full BSE on m pairs by a Davidson on ONE
    subspace V for both T = X + Y and S = X - Y (Stratmann, Scuseria and
    Frisch, J. Chem. Phys. 109, 8218 (1998)): the projected A + B and A - B
    are positive definite, so the small problem is (A - B)(A + B) T = Omega^2 T
    in its symmetric form. New directions are the Jacobi corrections of the X
    and Y residuals, -r_X / (d - Omega) and -r_Y / (d + Omega).

    A few roots beyond the n asked for are carried and kept through every
    restart: the diagonal is the quasiparticle one only (the kernel's diagonal
    is not at hand matrix-free), and a root just above the last one asked
    for otherwise leaves the subspace at each restart and the iteration
    stalls.
    """
    diag = np.asarray(diag, float)
    nwork = min(m, n + FRAGMENT_DAVIDSON_EXTRA_ROOTS)
    nseed = min(m, 2 * nwork)
    v = np.zeros((m, nseed))
    v[np.argsort(diag)[:nseed], np.arange(nseed)] = 1.0
    max_space = min(m, max(FRAGMENT_DAVIDSON_MIN_SPACE, 8 * nwork))

    def act(block):
        kp = apply(np.vstack([block, block]))[:m]          # (A + B) block
        km = apply(np.vstack([block, -block]))[:m]         # (A - B) block
        return kp, km
    apb, amb = act(v)
    for _ in range(max_cycle):
        ap = v.T @ apb
        am = v.T @ amb
        low = np.linalg.cholesky(0.5 * (am + am.T))
        w2, z = np.linalg.eigh(low.T @ (0.5 * (ap + ap.T)) @ low)
        k = min(nwork, len(w2))
        om = np.sqrt(w2[:k])
        t = (low @ z[:, :k]) / np.sqrt(om)
        s = np.linalg.solve(low.T, z[:, :k]) * np.sqrt(om)
        r1 = apb @ t - (v @ s) * om                       # (A+B)T - Omega S
        r2 = amb @ s - (v @ t) * om                       # (A-B)S - Omega T
        r_x, r_y = 0.5 * (r1 + r2), 0.5 * (r1 - r2)
        res = np.maximum(np.abs(r_x).max(axis=0), np.abs(r_y).max(axis=0))
        if res[:n].max() < tol:
            vt, vs = v @ t[:, :n], v @ s[:, :n]
            return om[:n], np.vstack([0.5 * (vt + vs), 0.5 * (vt - vs)])
        new = []
        for j in np.flatnonzero(res >= tol):
            for r, den in ((r_x[:, j], diag - om[j]), (r_y[:, j], diag + om[j])):
                den = np.where(np.abs(den) < 1e-3, np.copysign(1e-3, den), den)
                new.append(-r / den)
        new = np.array(new).T
        if v.shape[1] + new.shape[1] > max_space:
            q, rr = np.linalg.qr(np.hstack([t, s]))
            q = q[:, np.abs(np.diag(rr)) > 1e-12]
            v, apb, amb = v @ q, apb @ q, amb @ q
        for _ in range(2):
            new -= v @ (v.T @ new)
        q, rr = np.linalg.qr(new)
        q = q[:, np.abs(np.diag(rr)) > 1e-10 * max(1.0, np.abs(new).max())]
        if q.shape[1] == 0:
            break
        kp, km = act(q)
        v, apb, amb = np.hstack([v, q]), np.hstack([apb, kp]), np.hstack([amb, km])
    raise RuntimeError(f'full-BSE block eigenpairs did not converge to {tol} '
                       f'(residual {res[:n].max():.2e})')


@dataclass
class FragmentPartition:
    """Diabats, A_eff(Omega_0), and the vectors its derivative needs.

    sites: {fragment index: number of site states}; ct: {(hole fragment,
    electron fragment): number of charge-transfer diabats}. Diabats are
    ordered sites first (by fragment), then charge transfer, each block's
    states by energy; `labels` names them. Vectors are (dim, n_p): n_ov rows
    in the Tamm-Dancoff approximation, the stacked (X, Y) for the full BSE.
    """
    operator: BSEOperator
    orbitals: object
    omega0: float
    labels: list
    p_local: np.ndarray            # (dim, n_p) diabats, local basis
    y_local: np.ndarray            # (dim, n_p) resolvent vectors, local basis
    a_pp: np.ndarray               # (n_p, n_p) direct (Coulomb-only) part
    sigma: np.ndarray              # (n_p, n_p) Sigma(Omega_0)
    dsigma: np.ndarray             # (n_p, n_p) dSigma/dOmega at Omega_0
    q_lowest: float                # lowest positive eigenvalue on Q
    block_energies: list = field(default_factory=list)
    gaps: np.ndarray = None        # each diabat's gap to its block neighbours
    block_rows: list = field(default_factory=list)   # each diabat's block rows

    @property
    def tda(self):
        return self.operator.tda

    @property
    def metric(self):
        return self.operator.metric

    @property
    def a_eff(self):
        return self.a_pp + self.sigma

    @property
    def u_local(self):
        """u_a = p_a + y_a: the nuclear derivative of A_eff_ab is u_a^T dK u_b."""
        return self.p_local + self.y_local

    def u_canonical(self):
        return to_canonical(self.orbitals, self.u_local)

    def p_canonical(self):
        return to_canonical(self.orbitals, self.p_local)

    def xy(self, v):
        """(X, Y) halves of a stacked block; Y = 0 in the Tamm-Dancoff case."""
        return split_xy(v, self.operator.n_ov)

    @classmethod
    def build(cls, operator, orbitals, sites, ct=None, omega0=None,
              tol=FRAGMENT_SOLVE_TOL, margin=FRAGMENT_POLE_MARGIN,
              progress=None):
        """`progress`, a callable taking one string, hears each stage."""
        say = progress if progress is not None else (lambda msg: None)
        ct = {} if ct is None else dict(ct)
        if orbitals.nocc != operator.nocc:
            raise ValueError('orbitals and operator disagree on nocc')
        hole, elec = orbitals.pair_labels()
        tda, n_ov, dim = operator.tda, operator.n_ov, operator.dim
        sig = operator.metric

        # every solve below runs in the fragment-canonical working basis
        # (`FragmentCanonical`); p and y return to the local basis at the end
        work = FragmentCanonical(orbitals, operator.eps_qp)

        def k_loc(x):
            x = to_canonical(orbitals, work.to_local(x))
            return work.from_local(to_local(orbitals, operator.apply(x)))

        d_ov = work.diagonal()
        dense_max = dense_limit(operator)
        diag = d_ov if tda else np.concatenate([d_ov, d_ov])

        blocks = [((k, k), n, f'site {k}') for k, n in sorted(sites.items())]
        blocks += [((k, l), n, f'ct {k}->{l}') for (k, l), n in
                   sorted(ct.items())]
        p_cols, labels, block_energies, gaps, block_rows = [], [], [], [], []
        for (k, l), n, name in blocks:
            idx = np.flatnonzero((hole == k) & (elec == l))
            if idx.size < n:
                raise ValueError(f'{name}: block has {idx.size} pairs, '
                                 f'{n} states asked for')
            rows = idx if tda else np.concatenate([idx, idx + n_ov])

            def block_apply(x, rows=rows):
                full = np.zeros((dim, x.shape[1]))
                full[rows] = x
                return k_loc(full)[rows]
            n_get = min(n + 1, idx.size)
            if tda:
                w, v = _lowest(block_apply, d_ov[idx], n_get, idx.size, tol,
                               dense_max=dense_max)
            else:
                w, v = _lowest_pencil(block_apply, d_ov[idx], n_get, idx.size,
                                      tol, dense_max=dense_max)
            w, v = lockstep((np.asarray(w, float), np.ascontiguousarray(v)),
                            check=True)
            for s in range(n):
                others = np.delete(w, s)
                gaps.append(float(np.abs(others - w[s]).min())
                            if others.size else np.inf)
                col = np.zeros(dim)
                col[rows] = v[:, s]
                # the phase convention is the LOCAL basis's: largest local
                # X component positive, whatever basis the block was solved in
                loc = work.to_local(col[:, None])[:n_ov, 0]
                col *= np.sign(loc[np.abs(loc).argmax()])
                p_cols.append(col)
                labels.append(f'{name}.{s}')
                block_rows.append(rows)
            block_energies.append((name, w[:n]))
            say(f'{name}: {idx.size} pairs, lowest {np.round(w[:n], 6)} Ha')
        p = np.array(p_cols).T
        kp = k_loc(p)
        a_pp = p.T @ kp
        a_pp = 0.5 * (a_pp + a_pp.T)
        if omega0 is None:
            # the mean of each site's lowest state: with two states per site
            # the mean of all sits between the two bands, next to whichever
            # stayed in Q
            omega0 = float(np.mean([e[0] for name, e in block_energies
                                    if name.startswith('site')]))

        # Q is the complement orthogonal in the metric: Pi = 1 - p (S p)^T
        # projects on it along P, and P^T S p = 1 makes Pi idempotent
        sp = sig[:, None] * p

        def proj(x):
            return x - p @ (sp.T @ x)

        def proj_t(x):
            return x - sp @ (p.T @ x)

        def shifted(x, om):
            """Pi^T (K - om S) Pi x: the Q block of the pencil at om."""
            px = proj(x)
            return proj_t(k_loc(px) - om * sig[:, None] * px)

        say(f'pole guard on Q ({dim} rows), Omega_0 = {omega0:.6f} Ha')
        q_low = lockstep(float(cls._q_lowest(k_loc, shifted, proj, sig, sp,
                                             diag, dim, tda, omega0, margin,
                                             tol, dense_max)))
        say(f'lowest positive eigenvalue on Q {q_low:.6f} Ha')
        if not omega0 < q_low - margin:
            raise ValueError(
                f'Omega_0 = {omega0:.6f} Ha is not below the lowest positive '
                f'eigenvalue on Q ({q_low:.6f} Ha) by the margin {margin}: '
                f'Sigma has a pole there. Make the configuration that comes '
                f'close an explicit diabat (sites / ct) or lower Omega_0.')

        rhs = -proj_t(kp)                   # Pi^T (K - Omega_0 S) y = -Pi^T K p
        if dim <= dense_max:
            # + p p^T fills the null space Pi leaves, without touching the
            # solution: the right-hand side is orthogonal to p
            m = shifted(np.eye(dim), omega0) + p @ p.T
            y = proj(np.linalg.solve(0.5 * (m + m.T), rhs))
        else:
            prec_d = np.maximum(diag - omega0 * sig, 1e-3)
            say(f'resolvent: {rhs.shape[1]} columns, batched conjugate gradients')
            # Pi D^-1 Pi^T: symmetric, and positive on the range the
            # residuals live in
            y = proj(_batched_cg(lambda x: shifted(x, omega0), rhs,
                                 lambda x: proj(proj_t(x) / prec_d[:, None]),
                                 tol))
        y = lockstep(np.ascontiguousarray(y), check=True)
        sigma = -rhs.T @ y                          # p_a^T K y_b
        sigma = 0.5 * (sigma + sigma.T)
        dsigma = -(y.T @ (sig[:, None] * y))
        p, y = work.to_local(p), work.to_local(y)
        return cls(operator=operator, orbitals=orbitals, omega0=float(omega0),
                   labels=labels, p_local=p, y_local=y, a_pp=a_pp,
                   sigma=sigma, dsigma=dsigma, q_lowest=q_low,
                   block_energies=block_energies, gaps=np.array(gaps),
                   block_rows=block_rows)

    @staticmethod
    def _q_lowest(k_loc, shifted, proj, sig, sp, diag, dim, tda, omega0,
                  margin, tol, dense_max=FRAGMENT_DENSE_MAX):
        """The lowest positive eigenvalue of the pencil on Q.

        Through H(om) = Pi^T (K - om S) Pi + c (S p)(S p)^T, whose lowest
        eigenvalue is positive exactly when K_QQ - om S_QQ is positive definite
        (the second term lifts P and vanishes on Q). Tamm-Dancoff: H(0)'s
        lowest eigenvalue is the answer. Full BSE: dense, the pencil on an
        explicit Q basis; iteratively, H(Omega_0 + margin) is checked
        positive -- the guard itself -- and Newton on its lowest eigenvalue,
        d mu / d om = -(Pi x)^T S (Pi x), locates the eigenvalue.
        """
        # a guard, not a result: 1e-7 Ha is ample against the margin
        gtol = max(tol, 1e-7)
        if not tda and dim <= dense_max:
            q = null_space(sp.T)
            kq = q.T @ k_loc(q)
            mq = q.T @ (sig[:, None] * q)
            mu = eigh(0.5 * (mq + mq.T), 0.5 * (kq + kq.T), eigvals_only=True)
            return float(1.0 / mu.max())

        def lowest(om, x0=None):
            big = float(np.abs(diag).max()) + abs(om)

            def h(x):
                return shifted(x, om) + big * (sp @ (sp.T @ x))
            w, x = _lowest(h, diag - om * sig + big * (sp ** 2).sum(axis=1),
                           1, dim, gtol, x0=x0, dense_max=dense_max)
            return float(w[0]), x[:, 0]
        if tda:
            return lowest(0.0)[0]
        om = omega0 + margin
        mu, x = lowest(om)
        # The sign of mu at Omega_0 + margin decides the guard; Newton,
        # each Davidson warm-started from the last eigenvector, locates the
        # eigenvalue a refusal names. mu and x are rank 0's
        # (`root_driven_solve`), and so are the slope and the stop, so every
        # rank runs the same steps.
        for _ in range(30):
            px = proj(x[:, None])[:, 0]
            slope = lockstep(-float(px @ (sig * px)))
            if slope >= 0.0:
                break
            step = -mu / slope
            om += step
            mu, x = lowest(om, x)
            if lockstep(bool(abs(step) < FRAGMENT_GUARD_NEWTON_TOL)):
                break
        return om


class FeshbachOracle:
    """Dense reference: Sigma(Omega), roots of A_eff(Omega) c = Omega c, Z.

    K (n, n), a P basis (n, n_p) with P^T S P = 1, and the metric S as its
    diagonal (`metric`, default ones: the Tamm-Dancoff case) in the same basis;
    Q is the complement orthogonal in the metric. Small systems only -- it
    diagonalizes the pencil on Q: with V its eigenvectors normalized to
    V^T K_QQ V = 1 and V^T S_QQ V = diag(mu), mu = 1 / lambda,
    (K_QQ - Omega S_QQ)^-1 = V diag(1 / (1 - Omega mu)) V^T.
    """

    def __init__(self, k, p, metric=None):
        k = 0.5 * (np.asarray(k, float) + np.asarray(k, float).T)
        p = np.asarray(p, float)
        metric = np.ones(len(k)) if metric is None else np.asarray(metric, float)
        self.q = null_space((metric[:, None] * p).T)
        self.k, self.p = k, p
        self.a_pp = p.T @ k @ p
        kq = self.q.T @ k @ self.q
        mq = self.q.T @ (metric[:, None] * self.q)
        self.mu, v = eigh(0.5 * (mq + mq.T), 0.5 * (kq + kq.T))
        self.c = p.T @ k @ self.q @ v              # couplings to Q eigenstates

    def sigma(self, omega):
        return -(self.c / (1.0 - omega * self.mu)) @ self.c.T

    def dsigma(self, omega):
        return -(self.c * (self.mu / (1.0 - omega * self.mu) ** 2)) @ self.c.T

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
