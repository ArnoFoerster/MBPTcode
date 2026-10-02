"""Particle-particle RPA (ppRPA), next to the particle-hole CasidaSolver.

The ppRPA eigenproblem in restricted pair spaces (c<d virtual, k<l occupied)
of spin orbitals is

    [[A, B], [B^T, C]] (X; Y) = Omega diag(1, -1) (X; Y),

    A_{cd,ef} = (eps_c + eps_d) delta_{ce} delta_{df} + <cd||ef>,
    B_{cd,kl} = <cd||kl>,
    C_{kl,mn} = -(eps_k + eps_l) delta_{km} delta_{ln} + <kl||mn>.

Unlike the particle-hole problem its spectrum is not symmetric: the n_vv
positive-norm solutions (X^T X - Y^T Y = 1) are the N+2 (double attachment)
energies Omega^+, the n_oo negative-norm ones (Y^T Y - X^T X = 1) the N-2
(double removal) energies Omega^-, both on the scale of eps_c + eps_d and
eps_k + eps_l. The two sets share one amplitude matrix: T = Y^+ (X^+)^{-1}
equals (X^- (Y^-)^{-1})^T, the ladder-CCD amplitudes of the Riccati equation
B^T + C T + T A + T B T = 0, and the ppRPA correlation energy is

    E_c = sum Omega^+ - Tr A = -sum Omega^- - Tr C = Tr(B T)

(van Aggelen, Yang, Yang, Phys. Rev. A 88, 030501 (2013); Scuseria,
Henderson, Bulik, J. Chem. Phys. 139, 104113 (2013)).

Solved as a Hermitian pencil: for mu between the two sets, H - mu*eta is
positive definite, and eta z = lambda (H - mu*eta) z has lambda = 1/(Omega-mu),
positive exactly for the positive-norm solutions.

build_pprpa_matrices is spin-orbital (g the antisymmetrized <pq||rs> of the
ADC solvers); build_pprpa_matrices_restricted gives the closed-shell singlet
and triplet channels from the DF factor or the dense chemist tensor. The
amplitude route (riccati.solve_riccati, and the DF-streamed
adc_r_faddeev.solve_ladder_amplitudes_restricted) needs no eigenvectors.
"""
import numpy as np
import scipy.linalg as la

from src.Base.eri_blocks import g_slice


def pair_indices(n):
    """(p, q) with p < q over range(n), in np.triu_indices order."""
    return np.triu_indices(n, k=1)


def build_pprpa_matrices(eps, g, nocc):
    """(A, B, C) of the spin-orbital ppRPA.

    Parameters
    ----------
    eps : ndarray, shape (nso,)
        Spin-orbital energies, occupied first.
    g : ndarray, shape (nso, nso, nso, nso)
        Antisymmetrized <pq||rs>.
    nocc : int
        Occupied spin orbitals.

    Returns
    -------
    A : (n_vv, n_vv), B : (n_vv, n_oo), C : (n_oo, n_oo)
        Pairs c<d and k<l in pair_indices order (virtual indices counted
        from the first virtual).
    """
    eps = np.asarray(eps)
    nso = len(eps)
    o, v = slice(0, nocc), slice(nocc, nso)
    eo, ev = eps[o], eps[v]
    ku, lu = pair_indices(nocc)
    cu, du = pair_indices(nso - nocc)
    A = g[v, v, v, v][cu, du][:, cu, du]
    A[np.diag_indices_from(A)] += ev[cu] + ev[du]
    B = g[v, v, o, o][cu, du][:, ku, lu]
    C = g[o, o, o, o][ku, lu][:, ku, lu]
    C[np.diag_indices_from(C)] -= eo[ku] + eo[lu]
    return A, B, C


def singlet_pair_indices(n):
    """(p, q) with p <= q over range(n): the n diagonal pairs first, then
    p < q in pair_indices order -- the order of the restricted ADC classes
    (I', III') and (I, III), whose pair is spin-singlet coupled."""
    pu, qu = pair_indices(n)
    d = np.arange(n)
    return np.concatenate([d, pu]), np.concatenate([d, qu])


def _pair_block(B_aa, eri_chemist, x, y, spin, rows, cols, chunk=16):
    """Pair-space matrix n_r n_c [<x x'|y y'> +- <x x'|y' y>] over the pairs
    rows = (x, x') in index range x and cols = (y, y') in range y, with
    <pq|rs> = (pr|qs); + for the singlet, - for the triplet. Streamed over
    the first row orbital so the (x, y, x, y) block is never whole."""
    r1, r2 = rows
    c1, c2 = cols
    sgn = 1.0 if spin == 'singlet' else -1.0
    nr = 1.0 / np.sqrt(1.0 + (r1 == r2)) if spin == 'singlet' else np.ones(len(r1))
    nc = 1.0 / np.sqrt(1.0 + (c1 == c2)) if spin == 'singlet' else np.ones(len(c1))
    x0, xn = x.start, x.stop - x.start
    out = np.empty((len(r1), len(c1)))
    for lo in range(0, xn, chunk):
        hi = min(lo + chunk, xn)
        sel = np.where((r1 >= lo) & (r1 < hi))[0]
        if not len(sel):
            continue
        G = g_slice(B_aa, eri_chemist, slice(x0 + lo, x0 + hi), x, y, y)
        p, pp = r1[sel] - lo, r2[sel]
        # G[p, p', y, y'] = <p p'|y y'>
        out[sel] = (G[p[:, None], pp[:, None], c1[None, :], c2[None, :]]
                    + sgn * G[p[:, None], pp[:, None], c2[None, :], c1[None, :]])
    return out * nr[:, None] * nc[None, :]


def build_pprpa_matrices_restricted(eps, nocc, spin, B_aa=None,
                                    eri_chemist=None):
    """(A, B, C) of the closed-shell spin-adapted ppRPA.

    A pair of spatial orbitals couples to a singlet (p <= q, spatially
    symmetric, singlet_pair_indices order, normalized by 1/sqrt(1+delta_pq))
    or a triplet (p < q, antisymmetric, pair_indices order; the three M_S
    components are degenerate). With <pq|rs> = (pr|qs),

        A_{ab,cd} = (eps_a + eps_b) delta + n_ab n_cd [<ab|cd> +- <ab|dc>],
        B_{ab,kl} = n_ab n_kl [<ab|kl> +- <ab|lk>],
        C_{kl,mn} = -(eps_k + eps_l) delta + n_kl n_mn [<kl|mn> +- <kl|nm>],

    + and n = 1/sqrt(1+delta) for the singlet, - and n = 1 for the triplet.
    The triplet is the alpha-alpha block of build_pprpa_matrices; the
    spin-orbital spectrum is the singlet one plus three copies of the
    triplet one.

    Parameters
    ----------
    eps : ndarray, shape (norb,)
        Spatial orbital energies.
    nocc : int
        Doubly occupied orbitals.
    spin : {'singlet', 'triplet'}
    B_aa : ndarray, shape (naux, norb, norb), optional
        DF factor, (pq|rs) = sum_Q B_aa[Q,p,q] B_aa[Q,r,s].
    eri_chemist : ndarray, shape (norb,)*4, optional
        Dense (pq|rs), used when B_aa is None.
    """
    if spin not in ('singlet', 'triplet'):
        raise ValueError(f"spin={spin!r}; expected 'singlet' or 'triplet'")
    if B_aa is None and eri_chemist is None:
        raise ValueError("build_pprpa_matrices_restricted needs B_aa or "
                         "eri_chemist")
    eps = np.asarray(eps)
    norb = len(eps)
    nv = norb - nocc
    o, v = slice(0, nocc), slice(nocc, norb)
    pairs = singlet_pair_indices if spin == 'singlet' else pair_indices
    po, pv = pairs(nocc), pairs(nv)
    A = _pair_block(B_aa, eri_chemist, v, v, spin, pv, pv)
    A[np.diag_indices_from(A)] += eps[nocc + pv[0]] + eps[nocc + pv[1]]
    B = _pair_block(B_aa, eri_chemist, v, o, spin, pv, po)
    C = _pair_block(B_aa, eri_chemist, o, o, spin, po, po)
    C[np.diag_indices_from(C)] -= eps[po[0]] + eps[po[1]]
    return A, B, C


class PPRPAResult:
    """N+2 and N-2 solutions of the ppRPA, each ascending in energy.

    omega_add (n_vv,), X_add (n_vv, n_vv), Y_add (n_oo, n_vv): positive norm.
    omega_rem (n_oo,), X_rem (n_vv, n_oo), Y_rem (n_oo, n_oo): negative norm.
    """
    def __init__(self, omega_add, X_add, Y_add, omega_rem, X_rem, Y_rem, mu):
        self.omega_add, self.X_add, self.Y_add = omega_add, X_add, Y_add
        self.omega_rem, self.X_rem, self.Y_rem = omega_rem, X_rem, Y_rem
        self.mu = mu


class PPRPASolver:
    """Solves [[A,B],[B^T,C]] z = Omega diag(1,-1) z for the N+2 and N-2
    pair energies."""

    def __init__(self, A, B, C, mu=None):
        """
        Parameters
        ----------
        A, B, C : ndarray
            From build_pprpa_matrices.
        mu : float, optional
            A pair chemical potential between the N-2 and N+2 energies. The
            default is midway between max diag(-C) and min diag(A); when that
            is not inside the gap, solve() locates the gap from the
            non-Hermitian spectrum and retries.
        """
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C)
        self.n_vv, self.n_oo = self.B.shape
        if mu is None:
            lo = np.max(-np.diag(self.C)) if self.n_oo else -np.inf
            hi = np.min(np.diag(self.A)) if self.n_vv else np.inf
            if not np.isfinite(lo):
                lo = hi - 1.0
            if not np.isfinite(hi):
                hi = lo + 1.0
            mu = 0.5 * (lo + hi)
        self.mu = float(mu)

    def hamiltonian(self):
        """The full (n_vv + n_oo) square [[A, B], [B^T, C]]."""
        return np.block([[self.A, self.B], [self.B.T, self.C]])

    def metric(self):
        """diag(1, ..., 1, -1, ..., -1)."""
        return np.concatenate([np.ones(self.n_vv), -np.ones(self.n_oo)])

    def solve(self, tda=False):
        """Returns a PPRPAResult.

        tda=True drops B: the N+2 energies are eig(A) with Y = 0 and the N-2
        energies eig(-C) with X = 0.
        """
        nv, no = self.n_vv, self.n_oo
        if tda:
            wa, Xa = la.eigh(self.A)
            wr, Yr = la.eigh(-self.C)
            return PPRPAResult(wa, Xa, np.zeros((no, nv)), wr,
                               np.zeros((nv, no)), Yr, self.mu)
        H = self.hamiltonian()
        eta = self.metric()
        try:
            return self._solve_pencil(H, eta, self.mu)
        except la.LinAlgError:
            mu = self._gap_from_spectrum(H, eta)
            return self._solve_pencil(H, eta, mu)

    def _solve_pencil(self, H, eta, mu):
        nv = self.n_vv
        Hs = H - mu * np.diag(eta)
        lam, Z = la.eigh(np.diag(eta), Hs)       # raises unless Hs is PD
        # eta z = lam Hs z with z^T Hs z = 1, so z^T eta z = lam and
        # Omega - mu = 1/lam; rescale to z^T eta z = sign(lam).
        Z = Z / np.sqrt(np.abs(lam))[None, :]
        omega = mu + 1.0 / lam
        add = lam > 0
        if np.count_nonzero(add) != nv:
            raise la.LinAlgError("ppRPA inertia: expected n_vv positive-norm "
                                 f"solutions, found {np.count_nonzero(add)}")
        ia = np.where(add)[0][np.argsort(omega[add])]
        ir = np.where(~add)[0][np.argsort(omega[~add])]
        return PPRPAResult(omega[ia], Z[:nv, ia], Z[nv:, ia],
                           omega[ir], Z[:nv, ir], Z[nv:, ir], mu)

    def _gap_from_spectrum(self, H, eta):
        """A mu inside the gap between the N-2 and N+2 energies, from the
        non-Hermitian eta*H; raises when there is none (a ppRPA instability)."""
        w, V = la.eig(eta[:, None] * H)
        if np.max(np.abs(w.imag)) > 1e-8 * max(1.0, np.max(np.abs(w.real))):
            raise la.LinAlgError("ppRPA has complex eigenvalues: the reference "
                                 "is unstable in the particle-particle channel")
        w = w.real
        V = V.real
        norm = np.einsum('i,ij,ij->j', eta, V, V)
        add, rem = w[norm > 0], w[norm < 0]
        if len(add) != self.n_vv or (len(rem) and len(add)
                                     and rem.max() >= add.min()):
            raise la.LinAlgError("ppRPA: no gap between the N-2 and N+2 "
                                 "energies; H - mu*eta is positive definite for "
                                 "no mu")
        lo = rem.max() if len(rem) else add.min() - 1.0
        hi = add.min() if len(add) else lo + 1.0
        return 0.5 * (lo + hi)

    def correlation_energy(self, result=None):
        """ppRPA correlation energy sum Omega^+ - Tr A (solves if needed)."""
        if result is None:
            result = self.solve()
        return float(np.sum(result.omega_add) - np.trace(self.A))

