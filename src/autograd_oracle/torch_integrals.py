"""Gaussian-basis integrals in torch, differentiable in the nuclear coordinates.

McMurchie-Davidson (Helgaker, Jorgensen, Olsen, Molecular Electronic-Structure
Theory, ch. 9): every charge distribution -- a primitive pair, a single
primitive, a normalized Gaussian surface charge -- is expanded in Hermite
Gaussians, and every Coulomb integral is a Hermite Coulomb integral R_tuv of a
Boys function. Nothing here comes from libcint; the only pyscf input is the
shell table (centres, angular momenta, exponents, contraction coefficients) and
the Cartesian-to-spherical matrices, read once as constants.

Conventions match pyscf's spherical AOs: a shell is
    chi_lm = sum_p c_p [ sum_cart C2S[cart, m] x^i y^j z^k ] exp(-a_p r^2)
with c_p = `Mole._libcint_ctr_coeff` (radial normalization included) and C2S =
`gto.cart2sph(l)`, Cartesian components in pyscf's order; `gate_nuclear
.shell_table_from_mole` reads them. Pinned against pyscf to 1e-13 by
`pin_integrals.py`.

Batch shape: primitives are vectorized, shells are looped in Python -- an
oracle for tens of shells, not a route.
"""
import math

import numpy as np
import torch

from src.autograd_oracle.frozen import shell_table_from_npz

torch.set_default_dtype(torch.float64)

BOYS_SERIES_TERMS = 220
BOYS_SWITCH = 45.0


# ---------------------------------------------------------------------------
# shell tables
# ---------------------------------------------------------------------------

def cart_components(l):
    """pyscf's Cartesian ordering: lx descending, then ly descending."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


class ShellTable:
    """Constants of one basis: per shell the atom, l, exponents, contraction
    coefficients (nprim, nctr) and the Cartesian->spherical matrix."""

    def __init__(self, atom_of, ls, exps, coeffs, c2s):
        self.atom_of = [int(a) for a in atom_of]
        self.ls = [int(l) for l in ls]
        self.exps = [torch.as_tensor(np.asarray(e, float)) for e in exps]
        self.coeffs = [torch.as_tensor(np.asarray(c, float)) for c in coeffs]
        self.c2s = [torch.as_tensor(np.asarray(c, float)) for c in c2s]
        self.nsh = len(self.ls)
        # function offsets: nctr blocks of (2l+1)
        self.nfun = [c.shape[1] * (2 * l + 1) for c, l in zip(self.coeffs, self.ls)]
        self.offsets = np.concatenate([[0], np.cumsum(self.nfun)])
        self.nao = int(self.offsets[-1])

    @classmethod
    def from_npz(cls, z, prefix):
        return cls(*shell_table_from_npz(z, prefix))


# ---------------------------------------------------------------------------
# Boys function and Hermite Coulomb integrals
# ---------------------------------------------------------------------------

def boys(nmax, T):
    """F_n(T), n = 0..nmax, shape (nmax+1, *T.shape). Series below BOYS_SWITCH,
    asymptotic above, downward recursion from n = nmax; both branches are
    evaluated on clamped arguments so the unused one carries no inf into
    autograd."""
    Ts = torch.clamp(T, max=BOYS_SWITCH)
    Ta = torch.clamp(T, min=BOYS_SWITCH)
    # series: F_n(T) = e^-T sum_k (2T)^k / [(2n+1)(2n+3)...(2n+2k+1)]
    term = torch.ones_like(Ts) / (2 * nmax + 1)
    acc = term.clone()
    for k in range(1, BOYS_SERIES_TERMS):
        term = term * (2.0 * Ts) / (2 * nmax + 2 * k + 1)
        acc = acc + term
    f_series = torch.exp(-Ts) * acc
    # asymptotic: (2n-1)!! / 2^(n+1) sqrt(pi / T^(2n+1))
    dfact = 1.0
    for k in range(1, 2 * nmax, 2):
        dfact *= k
    f_asym = dfact / 2.0 ** (nmax + 1) * torch.sqrt(math.pi / Ta ** (2 * nmax + 1))
    f = torch.where(T < BOYS_SWITCH, f_series, f_asym)
    out = [None] * (nmax + 1)
    out[nmax] = f
    e = torch.exp(-T)
    for n in range(nmax, 0, -1):
        out[n - 1] = (2.0 * T * out[n] + e) / (2 * n - 1)
    return torch.stack(out)


def hermite_coulomb(L, alpha, X, Y, Z):
    """R_tuv(alpha, R) for t+u+v <= L as a tensor (*batch, L+1, L+1, L+1);
    entries beyond t+u+v = L are zero."""
    T = alpha * (X * X + Y * Y + Z * Z)
    F = boys(L, T)                                  # (L+1, *batch)
    m2a = -2.0 * alpha
    R = {}
    pw = torch.ones_like(alpha)
    for n in range(L + 1):
        R[(n, 0, 0, 0)] = pw * F[n]
        pw = pw * m2a
    for order in range(1, L + 1):
        for t in range(order + 1):
            for u in range(order - t + 1):
                v = order - t - u
                for n in range(L - order + 1):
                    if t > 0:
                        val = X * R[(n + 1, t - 1, u, v)]
                        if t > 1:
                            val = val + (t - 1) * R[(n + 1, t - 2, u, v)]
                    elif u > 0:
                        val = Y * R[(n + 1, t, u - 1, v)]
                        if u > 1:
                            val = val + (u - 1) * R[(n + 1, t, u - 2, v)]
                    else:
                        val = Z * R[(n + 1, t, u, v - 1)]
                        if v > 1:
                            val = val + (v - 1) * R[(n + 1, t, u, v - 2)]
                    R[(n, t, u, v)] = val
    zero = torch.zeros_like(alpha)
    rows = []
    for t in range(L + 1):
        cols = []
        for u in range(L + 1):
            cols.append(torch.stack([R.get((0, t, u, v), zero) for v in range(L + 1)], -1))
        rows.append(torch.stack(cols, -2))
    return torch.stack(rows, -3)


# ---------------------------------------------------------------------------
# Hermite expansion of a primitive pair
# ---------------------------------------------------------------------------

def hermite_E(la, lb, a, b, XA, XB):
    """E_t^{ij} for i <= la, j <= lb, t <= i+j; shape (*batch, la+1, lb+1, la+lb+1).
    a, b exponents; XA = P - A, XB = P - B (P the product centre)."""
    p = a + b
    mu = a * b / p
    XAB = XA - XB
    K = torch.exp(-mu * XAB * XAB)
    nt = la + lb + 1
    E = {}
    zero = torch.zeros_like(a)
    E[(0, 0, 0)] = K

    def get(i, j, t):
        return E.get((i, j, t), zero)

    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            for t in range(i + j + 1):
                if i > 0:
                    E[(i, j, t)] = (get(i - 1, j, t - 1) / (2.0 * p) + XA * get(i - 1, j, t)
                                    + (t + 1) * get(i - 1, j, t + 1))
                else:
                    E[(i, j, t)] = (get(i, j - 1, t - 1) / (2.0 * p) + XB * get(i, j - 1, t)
                                    + (t + 1) * get(i, j - 1, t + 1))
    out = torch.stack([torch.stack([torch.stack([get(i, j, t) for t in range(nt)], -1)
                                    for j in range(lb + 1)], -2)
                       for i in range(la + 1)], -3)
    return out


class Distribution:
    """A charge distribution as Hermite Gaussians: exponents p (n,), centres P
    (n, 3), Hermite coefficients Ecoef (n, ncart, T, U, V) for its Cartesian
    functions, and the contraction map `ctr` (n, ncart, nfun_cart) folding
    primitives and contractions into the functions the caller wants."""

    def __init__(self, p, P, E, coef, L):
        self.p, self.P, self.E, self.coef, self.L = p, P, E, coef, L


def pair_distribution(la, lb, ea, eb, ca, cb, A, B):
    """Primitive-pair distribution of shells (la, A, ea, ca) and (lb, B, eb, cb).

    coef (npp, nctr_a, nctr_b) carries both contractions; E (npp, ncart_a,
    ncart_b, T, U, V)."""
    a = ea[:, None].expand(-1, len(eb)).reshape(-1)
    b = eb[None, :].expand(len(ea), -1).reshape(-1)
    p = a + b
    P = (a[:, None] * A[None, :] + b[:, None] * B[None, :]) / p[:, None]
    XA, XB = P - A[None, :], P - B[None, :]
    Ex = hermite_E(la, lb, a, b, XA[:, 0], XB[:, 0])
    Ey = hermite_E(la, lb, a, b, XA[:, 1], XB[:, 1])
    Ez = hermite_E(la, lb, a, b, XA[:, 2], XB[:, 2])
    ca_c, cb_c = cart_components(la), cart_components(lb)
    E = torch.stack([torch.stack([
        Ex[:, ia, ib][:, :, None, None] * Ey[:, ja, jb][:, None, :, None]
        * Ez[:, ka, kb][:, None, None, :]
        for (ib, jb, kb) in cb_c], 1) for (ia, ja, ka) in ca_c], 1)
    coef = (ca[:, None, :, None] * cb[None, :, None, :]).reshape(-1, ca.shape[1], cb.shape[1])
    # one Cartesian index for the pair, (a, b) row-major, as `coulomb` expects
    E = E.reshape(E.shape[0], len(ca_c) * len(cb_c), *E.shape[3:])
    return Distribution(p, P, E, coef, la + lb)


def single_distribution(la, ea, ca, A):
    """One shell as a distribution: E_t^{i0} about its own centre."""
    zero_b = torch.zeros_like(ea)
    XA = torch.zeros_like(ea)
    Ex = hermite_E(la, 0, ea, zero_b, XA, XA)[:, :, 0, :]    # (np, la+1, la+1)
    comps = cart_components(la)
    E = torch.stack([Ex[:, i][:, :, None, None] * Ex[:, j][:, None, :, None]
                     * Ex[:, k][:, None, None, :] for (i, j, k) in comps], 1)
    P = A[None, :].expand(len(ea), -1)
    return Distribution(ea, P, E, ca[:, :, None], la)


def coulomb(d1, d2, point2=False, charge2=None):
    """(d1 | d2) over 1/r12, contracted and spherical: (nfun1, nfun2).

    point2: d2 is a set of point charges at d2.P (its E is 1, p irrelevant).
    charge2: optional prefactor per member of d2 (a normalized Gaussian charge).
    """
    n1, n2 = d1.p.shape[0], d2.P.shape[0]
    L = d1.L + (0 if point2 else d2.L)
    p = d1.p[:, None]
    if point2:
        alpha = p.expand(n1, n2)
        pref = 2.0 * math.pi / p
    else:
        q = d2.p[None, :]
        alpha = p * q / (p + q)
        pref = 2.0 * math.pi ** 2.5 / (p * q * torch.sqrt(p + q))
    if charge2 is not None:
        pref = pref * charge2[None, :]
    D = d1.P[:, None, :] - d2.P[None, :, :]
    R = hermite_coulomb(L, alpha, D[..., 0], D[..., 1], D[..., 2])   # (n1,n2,L+1,L+1,L+1)
    T1 = d1.L + 1
    if point2:
        G = R[:, :, None, :T1, :T1, :T1]                                 # (n1,n2,1,T,U,V)
    else:
        G = 0.0
        T2 = d2.L + 1
        for tau in range(T2):
            for nu in range(T2 - tau):
                for phi in range(T2 - tau - nu):
                    sgn = -1.0 if (tau + nu + phi) % 2 else 1.0
                    e2 = d2.E[:, :, tau, nu, phi]                          # (n2, ncart2)
                    G = G + sgn * e2[None, :, :, None, None, None] * R[
                        :, :, None, tau:tau + T1, nu:nu + T1, phi:phi + T1]
    # sum over Hermite indices of d1: (n1, n2, ncart1, ncart2)
    E1 = d1.E                                                             # (n1, c1, T,U,V)
    val = torch.einsum('actuv,abztuv->abcz', E1, G) * pref[..., None, None]
    # contractions: d1 coef (n1, k1a, k1b) or (n1, k1, 1); d2 coef (n2, k2, 1) or none
    return val


# ---------------------------------------------------------------------------
# the integral classes a DF-RHF + ISDF + PCM chain needs
# ---------------------------------------------------------------------------

class Basis:
    """A ShellTable placed at nuclear coordinates `coords` (natm, 3), torch."""

    def __init__(self, table, coords):
        self.t = table
        self.coords = coords
        self.centres = [coords[a] for a in table.atom_of]

    def shell(self, i):
        t = self.t
        return t.ls[i], t.exps[i], t.coeffs[i], self.centres[i], t.c2s[i]

    def single(self, i):
        l, e, c, A, _ = self.shell(i)
        return single_distribution(l, e, c, A)

    def pair(self, i, j):
        la, ea, ca, A, _ = self.shell(i)
        lb, eb, cb, B, _ = self.shell(j)
        return pair_distribution(la, lb, ea, eb, ca, cb, A, B)

    def place(self, out, i, j, block):
        o = self.t.offsets
        out[o[i]:o[i + 1], o[j]:o[j + 1]] = block

    def values(self, points):
        """chi_mu(r_k) for every function: (npts, nao)."""
        cols = []
        for i in range(self.t.nsh):
            l, e, c, A, c2s = self.shell(i)
            d = points - A[None, :]
            r2 = (d * d).sum(-1)
            rad = torch.exp(-r2[:, None] * e[None, :]) @ c                   # (npts, nctr)
            mon = torch.stack([d[:, 0] ** lx * d[:, 1] ** ly * d[:, 2] ** lz
                               for lx, ly, lz in cart_components(l)], -1)    # (npts, ncart)
            sph = mon @ c2s                                                   # (npts, 2l+1)
            cols.append((rad[:, :, None] * sph[:, None, :]).reshape(len(points), -1))
        return torch.cat(cols, 1)


def _fold_pair(val, d, c2s_a, c2s_b, nfa, nfb):
    """(npp, ncart_a*ncart_b) or (npp, ncart_a, ncart_b) primitive values ->
    spherical, contracted (nfa, nfb)."""
    npp = val.shape[0]
    na, nb = c2s_a.shape[0], c2s_b.shape[0]
    val = val.reshape(npp, na, nb)
    v = torch.einsum('pab,pxy->xayb', val, d.coef)
    v = torch.einsum('xayb,am,bn->xmyn', v, c2s_a, c2s_b)
    return v.reshape(nfa, nfb)


def overlap_kinetic(basis):
    """(S, T) over the spherical functions."""
    t = basis.t
    S = torch.zeros(t.nao, t.nao)
    K = torch.zeros(t.nao, t.nao)
    for i in range(t.nsh):
        la, ea, ca, A, c2s_a = basis.shell(i)
        for j in range(i, t.nsh):
            lb, eb, cb, B, c2s_b = basis.shell(j)
            a = ea[:, None].expand(-1, len(eb)).reshape(-1)
            b = eb[None, :].expand(len(ea), -1).reshape(-1)
            p = a + b
            P = (a[:, None] * A[None, :] + b[:, None] * B[None, :]) / p[:, None]
            XA, XB = P - A[None, :], P - B[None, :]
            # 1D overlaps for i <= la+2 (kinetic needs j+2 on the ket)
            Es = [hermite_E(la, lb + 2, a, b, XA[:, k], XB[:, k])[:, :, :, 0] for k in range(3)]
            pref = (math.pi / p) ** 1.5
            ca_c, cb_c = cart_components(la), cart_components(lb)
            sv = torch.zeros(len(p), len(ca_c), len(cb_c))
            kv = torch.zeros(len(p), len(ca_c), len(cb_c))
            for ia, (i1, j1, k1) in enumerate(ca_c):
                for ib, (i2, j2, k2) in enumerate(cb_c):
                    sx, sy, sz = Es[0][:, i1, i2], Es[1][:, j1, j2], Es[2][:, k1, k2]
                    sv[:, ia, ib] = sx * sy * sz

                    def tt(E1, i_, j_):
                        val = -2.0 * b * (2 * j_ + 1) * E1[:, i_, j_] + 4.0 * b * b * E1[:, i_, j_ + 2]
                        if j_ >= 2:
                            val = val + j_ * (j_ - 1) * E1[:, i_, j_ - 2]
                        return -0.5 * val
                    kv[:, ia, ib] = (tt(Es[0], i1, i2) * sy * sz + sx * tt(Es[1], j1, j2) * sz
                                     + sx * sy * tt(Es[2], k1, k2))
            d = Distribution(p, P, None, (ca[:, None, :, None] * cb[None, :, None, :]).reshape(
                -1, ca.shape[1], cb.shape[1]), la + lb)
            nfa, nfb = t.nfun[i], t.nfun[j]
            sb = _fold_pair(sv * pref[:, None, None], d, c2s_a, c2s_b, nfa, nfb)
            kb = _fold_pair(kv * pref[:, None, None], d, c2s_a, c2s_b, nfa, nfb)
            basis.place(S, i, j, sb)
            basis.place(K, i, j, kb)
            if j != i:
                basis.place(S, j, i, sb.T)
                basis.place(K, j, i, kb.T)
    return S, K


def _pair_vs_dist(basis, i, j, d2, point2=False, charge2=None):
    """(ab | d2) for the shell pair (i, j): (nfa, nfb, n2fun_or_points)."""
    t = basis.t
    la, ea, ca, A, c2s_a = basis.shell(i)
    lb, eb, cb, B, c2s_b = basis.shell(j)
    d1 = basis.pair(i, j)
    val = coulomb(d1, d2, point2=point2, charge2=charge2)   # (npp, n2, c1, c2)
    npp, n2 = val.shape[0], val.shape[1]
    na, nb = c2s_a.shape[0], c2s_b.shape[0]
    val = val.reshape(npp, n2, na, nb, -1)                                # (npp, n2, na, nb, ncart2)
    v = torch.einsum('pzabw,pxy->xaybzw', val, d1.coef)
    v = torch.einsum('xaybzw,am,bn->xmynzw', v, c2s_a, c2s_b)
    return v.reshape(t.nfun[i], t.nfun[j], n2, -1)


def nuclear_attraction(basis, charges, nuc_coords):
    """V_mu nu = -sum_A Z_A (mu | 1/|r - R_A| | nu)."""
    t = basis.t
    V = torch.zeros(t.nao, t.nao)
    d2 = Distribution(None, nuc_coords, None, None, 0)
    for i in range(t.nsh):
        for j in range(i, t.nsh):
            blk = _pair_vs_dist(basis, i, j, d2, point2=True)[..., 0]      # (nfa, nfb, natm)
            blk = -(blk * charges[None, None, :]).sum(-1)
            basis.place(V, i, j, blk)
            if j != i:
                basis.place(V, j, i, blk.T)
    return V


def gaussian_charge_distribution(points, xi):
    """Normalized Gaussian charges (xi^2/pi)^(3/2) exp(-xi^2 |r - r_k|^2) as an
    s distribution; returns (Distribution, charge prefactors)."""
    q = xi * xi
    E = torch.ones(len(xi), 1, 1, 1, 1)
    d = Distribution(q, points, E, torch.ones(len(xi), 1, 1), 0)
    return d, (q / math.pi) ** 1.5


def pair_potential_on_charges(basis, points, xi):
    """(mu nu | g_k): (nao, nao, npts) for Gaussian charges g_k."""
    t = basis.t
    d2, pref = gaussian_charge_distribution(points, xi)
    out = torch.zeros(t.nao, t.nao, len(points))
    for i in range(t.nsh):
        for j in range(i, t.nsh):
            blk = _pair_vs_dist(basis, i, j, d2, charge2=pref)[..., 0]
            o = t.offsets
            out[o[i]:o[i + 1], o[j]:o[j + 1]] = blk
            if j != i:
                out[o[j]:o[j + 1], o[i]:o[i + 1]] = blk.transpose(0, 1)
    return out


def _single_vs_dist(basis, i, d2, point2=False, charge2=None):
    la, ea, ca, A, c2s_a = basis.shell(i)
    d1 = basis.single(i)
    val = coulomb(d1, d2, point2=point2, charge2=charge2)   # (np, n2, ncart1, ncart2)
    v = torch.einsum('pzaw,px->xazw', val, ca)
    v = torch.einsum('xazw,am->xmzw', v, c2s_a)
    return v.reshape(basis.t.nfun[i], val.shape[1], -1)


def single_potential_on_charges(basis, points, xi):
    """(P | g_k): (naux, npts)."""
    d2, pref = gaussian_charge_distribution(points, xi)
    out = torch.zeros(basis.t.nao, len(points))
    for i in range(basis.t.nsh):
        o = basis.t.offsets
        out[o[i]:o[i + 1]] = _single_vs_dist(basis, i, d2, charge2=pref)[..., 0]
    return out


def two_centre(basis):
    """(P | Q) over the spherical functions."""
    t = basis.t
    V = torch.zeros(t.nao, t.nao)
    singles = [basis.single(i) for i in range(t.nsh)]
    for i in range(t.nsh):
        la, ea, ca, A, c2s_a = basis.shell(i)
        for j in range(i, t.nsh):
            lb, eb, cb, B, c2s_b = basis.shell(j)
            val = coulomb(singles[i], singles[j])             # (na, nb, ca, cb)
            v = torch.einsum('pqab,px,qy->xayb', val, ca, cb)
            v = torch.einsum('xayb,am,bn->xmyn', v, c2s_a, c2s_b)
            blk = v.reshape(t.nfun[i], t.nfun[j])
            basis.place(V, i, j, blk)
            if j != i:
                basis.place(V, j, i, blk.T)
    return V


def three_centre(basis, aux):
    """(mu nu | P): (nao, nao, naux)."""
    t, ta = basis.t, aux.t
    out = torch.zeros(t.nao, t.nao, ta.nao)
    singles = [aux.single(k) for k in range(ta.nsh)]
    for i in range(t.nsh):
        for j in range(i, t.nsh):
            d1 = basis.pair(i, j)
            la, ea, ca, A, c2s_a = basis.shell(i)
            lb, eb, cb, B, c2s_b = basis.shell(j)
            for k in range(ta.nsh):
                lc, ec, cc, C, c2s_c = aux.shell(k)
                val = coulomb(d1, singles[k])                  # (npp, nc, cab, cc)
                npp, nc = val.shape[0], val.shape[1]
                na, nb = c2s_a.shape[0], c2s_b.shape[0]
                val = val.reshape(npp, nc, na, nb, -1)
                v = torch.einsum('pzabw,pxy,zu->xaybuw', val, d1.coef, cc)
                v = torch.einsum('xaybuw,am,bn,wo->xmynuo', v, c2s_a, c2s_b, c2s_c)
                blk = v.reshape(t.nfun[i], t.nfun[j], ta.nfun[k])
                o, oa = t.offsets, ta.offsets
                out[o[i]:o[i + 1], o[j]:o[j + 1], oa[k]:oa[k + 1]] = blk
                if j != i:
                    out[o[j]:o[j + 1], o[i]:o[i + 1], oa[k]:oa[k + 1]] = blk.transpose(0, 1)
    return out


def nuclear_potential_on_charges(nuc_coords, charges, points, xi):
    """Potential of the nuclei at the Gaussian charges: sum_A Z_A erf(xi_k R)/R."""
    d = points[:, None, :] - nuc_coords[None, :, :]
    r = torch.sqrt((d * d).sum(-1))
    return (charges[None, :] * torch.erf(xi[:, None] * r) / r).sum(1)
