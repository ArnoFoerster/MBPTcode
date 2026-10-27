"""Determinant-space union-space oracle for the generated ADC(n) hierarchy.

What it computes. The one-particle propagator of a small molecule by Loewdin
inner projection in the union space H_{N+1} (+) H_{N-1} (Tarantelli and
Cederbaum, Phys. Rev. A 45, 2790 (1992)), order by order in the
Moller-Plesset partitioning H(lambda) = F + lambda (H - F):

  * the Rayleigh-Schroedinger ground state psi_n, E_n of the full
    configuration-interaction space (intermediate normalization);
  * the raw metric S^(k) and superoperator M^(k) series over a set of
    character-(+1) operators, each operator X mapped to the union-space vector
    (X psi0 ; X^dag psi0), so that

        (X|Y)    = <X psi0|Y psi0> + <Y^dag psi0|X^dag psi0>
        (X|H^|Y) = <X psi0|(H - E0)|Y psi0> - <Y^dag psi0|(H - E0)|X^dag psi0>;

  * sequential class Gram-Schmidt plus within-class Loewdin orthonormalization
    as power series (the intermediate-state construction executed exactly),
    and the formal-power-series fold Sigma_dyn = B (omega - D)^-1 B^T with the
    ADC(n) manifold rule (block (I, J) carried through order n - k_I - k_J);
  * the exact lambda-Taylor coefficients of Sigma_dyn from full
    diagonalization on a complex-lambda circle.

The operator set is {a+_p}, the 2p1h-type C operators a+ a+ a and the
D operators (2h1p)^dag = a+_i a+_j a_a, then the higher classes. The
two-sector construction of production ISR-ADC equals this supermatrix with
every D x C block deleted ('twosector'); 'union' keeps them.
decouple_cross_blocks removes the D x C blocks by a unitary series instead
(the diagrammatic ADC elements). SectorISR is the non-Dyson construction in
one particle-number sector (IP-ADC(n) in H_{N-1}, EA-ADC(n) in H_{N+1}).

What it is for. The reference the tests compare the generated union-space
cross blocks, the union and diagrammatic ADC(4)/(5) supermatrices and the
non-Dyson ADC(n) blocks against, on systems small enough for full
configuration interaction (sto-3g H2, H4, LiH, BeH2).
"""
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from pyscf import gto, scf

from src.CodeGeneration.diagrammatic import exact_oracle as EO
from src.SingleReference.ADC.isr_engine import lowdin_sqrt_inv_series
from src.SingleReference.CC.determinant_space import (
    SectorBasis, build_hamiltonian, apply_ops, occupied_orbitals)

# complex probe frequencies: the imaginary part keeps every lambda disc and
# every fold away from the poles
OMEGAS_C = [0.41 + 0.77j, -1.13 + 0.65j, 1.55 + 0.5j, -0.35 + 1.1j,
            2.05 + 0.45j, -2.15 + 0.85j, 0.9 + 1.3j, -0.5 + 0.5j]

SYSTEMS = {
    'H4lin_r1.0': ('h4_chain', 1.0),
    'H4rect': ('atoms', 'H 0 0 0; H 0 0 1.0; H 0 1.4 0; H 0 1.4 1.0'),
    'H2': ('h2', 0.74),
    'LiH': ('atoms', 'Li 0 0 0; H 0 0 1.6'),
    'BeH2': ('atoms', 'Be 0 0 0; H 0 0 1.3; H 0 0 -1.3'),
}

PRINCIPAL_STRENGTH = 0.3      # pole strength above which a pole is a principal (quasiparticle) state


def maxabs(x):
    x = np.asarray(x)
    return float(np.max(np.abs(x))) if x.size else 0.0


def build_system(name):
    """Spin-orbital system of SYSTEMS[name] in sto-3g, occupied-first layout
    [occ_a, occ_b, vir_a, vir_b]. The SCF is converged to conv_tol 1e-14 and
    gradient 1e-12: the block comparisons are canonical-HF identities whose
    floor is the residual f[o,v], so the SCF has to sit below the 1e-12 gates."""
    kind, arg = SYSTEMS[name]
    if kind == 'h4_chain':
        atom = '; '.join(f'H 0 0 {i * arg}' for i in range(4))
    elif kind == 'h2':
        atom = f'H 0 0 0; H 0 0 {arg}'
    else:
        atom = arg
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 300
    mf.run()
    eps, h1, g, nocc = EO.spinorb_system(mol, mf)
    occ = np.arange(nocc)
    fock = h1 + np.einsum('piqi->pq', g[:, occ][:, :, :, occ])
    return dict(name=name, eps=eps, h1=h1, g=g, nocc=nocc, norb=len(eps),
                f_ov=maxabs(fock[:nocc, nocc:]), f_diag=maxabs(np.diag(fock) - eps),
                mol=mol, mf=mf)


# ---------------------------------------------------------------------------
# union space: determinant sectors, Hamiltonians, operator set
# ---------------------------------------------------------------------------
def enumerate_ops(nocc, norb, kmax):
    """Character-(+1) operator set. Each op: dict(cre, ann, k, typ, lab);
    apply_ops(det, cre, ann) realizes a+_{cre[0]} a+_{cre[1]} ... a_{ann[0]} ...
    k = 1: C (2p1h) has hole j outer, particle pair a < b inner, string
    a+_a a+_b a_j; D = E^dag with E (2h1p) = a+_a a_i a_j, particle a outer,
    hole pair i < j inner."""
    occ = list(range(nocc))
    vir = list(range(nocc, norb))
    ops = [dict(cre=[p], ann=[], k=0, typ='o', lab=(p,)) for p in range(norb)]
    for k in range(1, kmax + 1):
        if k == 1:
            for j in occ:
                for a in vir:
                    for b in vir:
                        if b > a:
                            ops.append(dict(cre=[a, b], ann=[j], k=1, typ='C', lab=(j, a, b)))
        else:
            for holes in combinations(occ, k):
                for parts in combinations(vir, k + 1):
                    ops.append(dict(cre=list(parts), ann=list(holes), k=k,
                                    typ='C', lab=tuple(holes) + tuple(parts)))
        if k == 1:
            for a in vir:
                for i in occ:
                    for j in occ:
                        if j > i:
                            # E = a+_a a_i a_j ; D = E^dag = a+_j a+_i a_a
                            ops.append(dict(cre=[j, i], ann=[a], k=1, typ='D', lab=(a, i, j)))
        else:
            for parts in combinations(vir, k):
                for holes in combinations(occ, k + 1):
                    e_cre, e_ann = list(parts), list(holes)
                    ops.append(dict(cre=e_ann[::-1], ann=e_cre[::-1], k=k,
                                    typ='D', lab=tuple(parts) + tuple(holes)))
    return ops


def kmax_complete(nocc, nvir):
    """Largest k with a nonempty C- or D-type class."""
    k = 0
    while (k + 2 <= nvir and k + 1 <= nocc) or (k + 2 <= nocc and k + 1 <= nvir):
        k += 1
    return k


def op_matrix(bt, bf, cre, ann):
    """Sparse matrix of the operator string (cre, ann) from basis bf to basis bt."""
    rows, cols, vals = [], [], []
    for col, det in enumerate(bf.dets):
        r = apply_ops(det, list(cre), list(ann))
        if r is None:
            continue
        ph, nd = r
        rows.append(bt.index[nd])
        cols.append(col)
        vals.append(float(ph))
    return sp.csr_matrix((vals, (rows, cols)), shape=(bt.dim, bf.dim))


class UnionSpace:
    """Determinant sectors N, N-1, N+1 of one system and the operator set
    through class kmax (default: every nonempty class)."""

    def __init__(self, sysd, kmax=None):
        eps, h1, g, nocc, norb = (sysd['eps'], sysd['h1'], sysd['g'],
                                  sysd['nocc'], sysd['norb'])
        self.eps, self.h1, self.g, self.nocc, self.norb = eps, h1, g, nocc, norb
        self.kmax = kmax_complete(nocc, norb - nocc) if kmax is None else kmax
        self.bN = SectorBasis(norb, nocc, nocc)
        self.bM = SectorBasis(norb, nocc - 1, nocc)
        self.bP = SectorBasis(norb, nocc + 1, nocc)
        self.H, self.F = {}, {}
        for key, b in (('N', self.bN), ('M', self.bM), ('P', self.bP)):
            self.H[key] = build_hamiltonian(b, h1, g).toarray()
            self.F[key] = np.diag([sum(eps[p] for p in occupied_orbitals(d)) for d in b.dets])
        self.hf = self.bN.index[(1 << nocc) - 1]
        self.ops = enumerate_ops(nocc, norb, self.kmax)
        self.nops = len(self.ops)
        self.typ = np.array([o['typ'] for o in self.ops])
        self.kcls = np.array([o['k'] for o in self.ops])
        # X: N -> P  and  X^dag: N -> M
        self.XP = [op_matrix(self.bP, self.bN, o['cre'], o['ann']) for o in self.ops]
        self.XM = [op_matrix(self.bM, self.bN, o['ann'][::-1], o['cre'][::-1]) for o in self.ops]
        self.idx = {(k, t): np.where((self.kcls == k) & (self.typ == t))[0]
                    for k in range(0, self.kmax + 1) for t in ('o', 'C', 'D')}
        self.gamma_hf = np.diag((np.arange(norb) < nocc).astype(float))

    def vectors(self, Psi):
        """V_P[:, I] = X_I Psi and V_M[:, I] = X_I^dag Psi for a (dimN,) Psi."""
        VP = np.column_stack([X @ Psi for X in self.XP])
        VM = np.column_stack([X @ Psi for X in self.XM])
        return VP, VM

    def hamiltonians(self, lam):
        return {k: self.F[k] + lam * (self.H[k] - self.F[k]) for k in ('N', 'M', 'P')}

    def ground_state(self, lam):
        """Ground state of H(lam). Complex lam: H is complex symmetric; the
        HF-dominated root with BILINEAR normalization (psi^T psi = 1), the
        analytic continuation the contour extraction needs."""
        HN = self.hamiltonians(lam)['N']
        if np.iscomplexobj(HN):
            w, V = np.linalg.eig(HN)
            i0 = int(np.argmax(np.abs(V[self.hf, :])))
            v = V[:, i0]
            return w[i0], v / np.sqrt(v @ v)
        w, V = np.linalg.eigh(HN)
        i0 = int(np.argmax(np.abs(V[self.hf, :])))
        return w[i0], V[:, i0]

    # ----- exact lambda-Taylor coefficients by the complex-lambda contour --
    def sigma_dyn_exact(self, lam, omega):
        """Exact Sigma_dyn(omega, lam) for real or complex lam (bilinear products)."""
        Hl = self.hamiltonians(lam)
        E0, psi = self.ground_state(lam)
        o = self.idx[(0, 'o')]
        Y = np.column_stack([self.XP[i] @ psi for i in o])      # <P| a+_q |0>
        X = np.column_stack([self.XM[i] @ psi for i in o])      # <M| a_p  |0>
        AP = omega * np.eye(self.bP.dim) - (Hl['P'] - E0 * np.eye(self.bP.dim))
        AM = omega * np.eye(self.bM.dim) + (Hl['M'] - E0 * np.eye(self.bM.dim))
        G = Y.T @ np.linalg.solve(AP, Y) + X.T @ np.linalg.solve(AM, X)
        Sig = np.diag(omega - self.eps) - np.linalg.inv(G)
        dg = X.T @ X - self.gamma_hf
        return Sig - lam * np.einsum('prqs,rs->pq', self.g, dg)

    def contour_coeffs(self, omega, orders, radius=0.3, n_points=32):
        """c_n = (1/N r^n) sum_k Sigma_dyn(omega, r e^{2 pi i k/N}) e^{-2 pi i nk/N}
        (Lyness and Moler); exact up to aliasing c_{n+N} r^N."""
        ks = np.arange(n_points)
        lams = radius * np.exp(2j * np.pi * ks / n_points)
        vals = np.array([self.sigma_dyn_exact(l, omega) for l in lams])
        coefs = np.fft.fft(vals, axis=0) / n_points
        return {n: coefs[n] / radius ** n for n in orders}

    # ----- RSPT series ---------------------------------------------------
    def rspt_series(self, nord):
        """psi_0..psi_{nord-1}, E_0..E_nord and the norm series (intermediate normalization)."""
        hf = self.hf
        H0, W = self.F['N'], self.H['N'] - self.F['N']
        d = H0[hf, hf] - np.diag(H0)
        d[hf] = 1.0
        R0 = 1.0 / d
        R0[hf] = 0.0
        p0 = np.zeros(self.bN.dim)
        p0[hf] = 1.0
        psi = [p0]
        E = {0: H0[hf, hf], 1: W[hf, hf]}
        for n in range(1, nord):
            v = W @ psi[n - 1] - E[1] * psi[n - 1]
            for k in range(2, n):
                v = v - E[k] * psi[n - k]
            pn = R0 * v
            pn[hf] = 0.0
            psi.append(pn)
            E[n + 1] = W[hf, :] @ pn
        nser = [1.0] + [0.0] * (nord - 1)
        for k in range(2, nord):
            nser[k] = float(sum(psi[a] @ psi[k - a] for a in range(1, k)))
        return psi, E, nser

    def series_matrices(self, psi, E, nser, nord):
        """Normalized raw S^(k), M^(k), k = 0..nord-1, over the full operator set."""
        VP = [None] * nord
        VM = [None] * nord
        for n in range(nord):
            VP[n], VM[n] = self.vectors(psi[n])
        IP, IM = np.eye(self.bP.dim), np.eye(self.bM.dim)
        HcP = {0: self.F['P'] - E[0] * IP, 1: (self.H['P'] - self.F['P']) - E[1] * IP}
        HcM = {0: self.F['M'] - E[0] * IM, 1: (self.H['M'] - self.F['M']) - E[1] * IM}
        HcVP = {(b, c): HcP[c] @ VP[b] for b in range(nord) for c in (0, 1)}
        HcVM = {(b, c): HcM[c] @ VM[b] for b in range(nord) for c in (0, 1)}
        SS = [np.zeros((self.nops, self.nops)) for _ in range(nord)]
        SJ = [np.zeros((self.nops, self.nops)) for _ in range(nord)]   # P-part minus M-part
        MM = [np.zeros((self.nops, self.nops)) for _ in range(nord)]
        for a in range(nord):
            for b in range(nord - a):
                k = a + b
                PP, MMm = VP[a].T @ VP[b], VM[a].T @ VM[b]
                SS[k] += PP + MMm
                SJ[k] += PP - MMm
                for c in (0, 1):
                    if k + c >= nord:
                        continue
                    MM[k + c] += VP[a].T @ HcVP[(b, c)] - VM[a].T @ HcVM[(b, c)]
        # the E_c (c >= 2) pieces of H_comp:  -E_c * (P-part - M-part)
        for c in range(2, nord):
            for k in range(c, nord):
                MM[k] -= E[c] * SJ[k - c]
        inv_n = series_reciprocal(nser, nord)
        return ndiv(SS, inv_n, nord), ndiv(MM, inv_n, nord)


# ---------------------------------------------------------------------------
# series helpers (order-general)
# ---------------------------------------------------------------------------
def series_reciprocal(n, nord):
    assert abs(n[0] - 1.0) < 1e-12
    inv = [1.0] + [0.0] * (nord - 1)
    for k in range(1, nord):
        inv[k] = -sum(n[j] * inv[k - j] for j in range(1, k + 1))
    return inv


def ndiv(E, inv_n, nord):
    return [sum(inv_n[k - i] * E[i] for i in range(k + 1)) for k in range(nord)]


def smul(A, B, nord):
    return [sum(A[i] @ B[k - i] for i in range(k + 1)) for k in range(nord)]


def sT(A):
    return [a.T for a in A]


def isr_orthonormalize(Hsup, Smet, slices, nord):
    """Sequential class Gram-Schmidt (against the orbitals and all lower
    classes) plus within-class Loewdin, series-exact. slices: list of index
    arrays, class 0 = orbitals. Returns blocks[(ci, cj)][k] and the per-class
    post-GS overlap series."""
    nfull = Hsup[0].shape[0]
    X = []
    for sl in slices:
        Xc = [np.zeros((nfull, len(sl))) for _ in range(nord)]
        Xc[0][sl, np.arange(len(sl))] = 1.0
        X.append(Xc)
    eff, sperp = [], []
    for ci, sl in enumerate(slices):
        Xc = X[ci]
        for cj in range(ci):
            P = smul(smul(sT(eff[cj]), Smet, nord), Xc, nord)
            corr = smul(eff[cj], P, nord)
            Xc = [Xc[k] - corr[k] for k in range(nord)]
        Wm = smul(smul(sT(Xc), Smet, nord), Xc, nord)
        sperp.append(Wm)
        nc = Wm[0].shape[0]
        ds = {k: Wm[k] for k in range(1, nord) if maxabs(Wm[k]) > 1e-13}
        if ds:
            Y = lowdin_sqrt_inv_series(ds, nord - 1)
            Yser = [Y.get(k, np.zeros((nc, nc))) for k in range(nord)]
        else:
            Yser = [np.eye(nc)] + [np.zeros((nc, nc)) for _ in range(nord - 1)]
        Xc = smul(Xc, Yser, nord)
        eff.append(Xc)
    blocks = {}
    for ci in range(len(slices)):
        for cj in range(len(slices)):
            blocks[(ci, cj)] = smul(smul(sT(eff[ci]), Hsup, nord), eff[cj], nord)
    return blocks, sperp


class SeriesFold:
    """Effective (orthonormalized) supermatrix series B_k (orbital x classes)
    and D_k (class x class) in a common segment layout [k1C, k1D, k2C, k2D, ...]
    for the 'union' and the 'twosector' construction, plus the truncated
    formal-power-series fold."""

    def __init__(self, us, Ssr, Msr, nord, mode, kmax):
        self.us, self.nord, self.mode, self.kmax = us, nord, mode, kmax
        norb = us.norb
        segs = [(k, t) for k in range(1, kmax + 1) for t in ('C', 'D') if len(us.idx[(k, t)])]
        self.segs = segs
        nseg = {s: len(us.idx[s]) for s in segs}
        offs, off = {}, 0
        for s in segs:
            offs[s] = off
            off += nseg[s]
        self.ncls = off
        self.offs, self.nseg = offs, nseg
        B = [np.zeros((norb, off)) for _ in range(nord)]
        D = [np.zeros((off, off)) for _ in range(nord)]
        orb = us.idx[(0, 'o')]

        def run(type_set):
            # sequential classes: orb, k1(type_set), k2(type_set), ...
            slices = [orb]
            cls_of_slice = [None]
            for k in range(1, kmax + 1):
                parts = [us.idx[(k, t)] for t in type_set if len(us.idx[(k, t)])]
                if not parts:
                    continue
                slices.append(np.concatenate(parts))
                cls_of_slice.append(k)
            full = np.concatenate(slices)
            Ssub = [S[np.ix_(full, full)] for S in Ssr]
            Msub = [M[np.ix_(full, full)] for M in Msr]
            loc, o2 = [], 0
            for sl in slices:
                loc.append(np.arange(o2, o2 + len(sl)))
                o2 += len(sl)
            blocks, _ = isr_orthonormalize(Msub, Ssub, loc, nord)
            # scatter into the common layout
            seg_pos = {}       # (k, t) -> positions within its slice
            for ci in range(1, len(slices)):
                k = cls_of_slice[ci]
                p0 = 0
                for t in type_set:
                    n = len(us.idx[(k, t)])
                    if n:
                        seg_pos[(k, t)] = (ci, np.arange(p0, p0 + n))
                        p0 += n
            for s, (ci, pos) in seg_pos.items():
                for kk in range(nord):
                    B[kk][:, offs[s]:offs[s] + nseg[s]] += blocks[(0, ci)][kk][:, pos]
                for s2, (cj, pos2) in seg_pos.items():
                    for kk in range(nord):
                        D[kk][offs[s]:offs[s] + nseg[s], offs[s2]:offs[s2] + nseg[s2]] += \
                            blocks[(ci, cj)][kk][np.ix_(pos, pos2)]

        if mode == 'union':
            run(('C', 'D'))
        elif mode == 'twosector':
            for t in ('C', 'D'):
                if any(len(us.idx[(k, t)]) for k in range(1, kmax + 1)):
                    run((t,))
        else:
            raise ValueError(mode)
        self.B, self.D = B, D
        self.kseg = np.concatenate([[k] * nseg[(k, t)] for (k, t) in segs]).astype(int)
        self.tseg = np.concatenate([[t] * nseg[(k, t)] for (k, t) in segs])

    def truncated(self, rule, kcap=None):
        """B_k, D_k with block (I, J) kept through order rule(kI, kJ) (rule None
        keeps everything); kcap drops every class with k > kcap."""
        B = [b.copy() for b in self.B]
        D = [d.copy() for d in self.D]
        if kcap is not None:
            drop = self.kseg > kcap
            for kk in range(self.nord):
                B[kk][:, drop] = 0.0
                D[kk][drop, :] = 0.0
                D[kk][:, drop] = 0.0
        if rule is not None:
            # masks per segment pair, broadcast to the state matrices
            segs = self.segs
            kb_seg = np.array([rule(0, k) for (k, t) in segs])
            kd_seg = np.array([[rule(kI, kJ) for (kJ, tJ) in segs] for (kI, tI) in segs])
            seg_of_state = np.concatenate([[i] * self.nseg[sg]
                                           for i, sg in enumerate(segs)]).astype(int)
            kb = kb_seg[seg_of_state]
            kd = kd_seg[np.ix_(seg_of_state, seg_of_state)]
            for kk in range(self.nord):
                B[kk][:, kb < kk] = 0.0
                D[kk][kd < kk] = 0.0
        return B, D

    def fold_coeffs(self, omega, kmax_order, rule=None, kcap=None):
        """c_k(omega), k = 0..kmax_order, of Sigma_dyn = B (omega - D)^-1 B^T as
        a formal power series in lambda."""
        B, D = self.truncated(rule, kcap)
        n = self.ncls
        A0 = omega * np.eye(n) - D[0]
        R = [np.linalg.inv(A0)]
        for m in range(1, max(kmax_order - 1, 0)):        # B_0 = 0: c_k needs R_b with b <= k-2
            acc = np.zeros((n, n), dtype=complex)
            for j in range(1, m + 1):
                acc += D[j] @ R[m - j]
            R.append(np.linalg.solve(A0, acc))
        c = {}
        for k in range(kmax_order + 1):
            ck = np.zeros((self.us.norb, self.us.norb), dtype=complex)
            for a in range(1, k):
                for b in range(k - a):
                    cc = k - a - b
                    if cc >= 1:
                        ck += B[a] @ R[b] @ B[cc].T
            c[k] = ck
        return c


def adc_rule(n):
    """Manifold rule: block (I, J) carried through order n - k_I - k_J
    (orbital class k = 0)."""
    def rule(kI, kJ):
        return n - kI - kJ
    return rule


# ---------------------------------------------------------------------------
# decoupling of the D x C blocks (diagrammatic ADC)
# ---------------------------------------------------------------------------
def series_exp(S, nord):
    """exp(S) for a matrix series S with S^(0) = S^(1) = 0 (few powers survive)."""
    n = S[0].shape[0]
    W = [np.eye(n)] + [np.zeros((n, n)) for _ in range(nord - 1)]
    term = [np.eye(n)] + [np.zeros((n, n)) for _ in range(nord - 1)]
    j = 1
    while True:
        term = smul(term, S, nord)
        term = [t / j for t in term]
        if all(maxabs(t) == 0.0 for t in term):
            break
        W = [W[k] + term[k] for k in range(nord)]
        j += 1
    return W


def decouple_cross_blocks(sf, nord):
    """Tarantelli-Cederbaum decoupling: a unitary series W = exp(S), S
    antisymmetric with only D x C blocks, such that W^T D W has no D x C content
    through order nord-1. Returns the couplings B W, the class blocks W^T D W
    (the diagrammatic ADC elements), W and the residual cross content."""
    cross = (sf.tseg[:, None] != sf.tseg[None, :])
    K = np.diag(sf.D[0])
    denom = K[None, :] - K[:, None]                      # S_ab = R_ab / (K_b - K_a) on the cross mask
    n = sf.ncls
    S = [np.zeros((n, n)) for _ in range(nord)]
    for m in range(2, nord):
        W = series_exp(S, nord)
        Dp = smul(smul(sT(W), sf.D, nord), W, nord)
        R = np.where(cross, Dp[m], 0.0)
        S[m] = np.where(cross, R / np.where(cross, denom, 1.0), 0.0)
    W = series_exp(S, nord)
    Dp = smul(smul(sT(W), sf.D, nord), W, nord)
    Bp = smul(sf.B, W, nord)
    resid = max(maxabs(np.where(cross, Dp[m], 0.0)) for m in range(nord))
    return Bp, Dp, W, resid


# ---------------------------------------------------------------------------
# non-Dyson ISR in one particle-number sector
# ---------------------------------------------------------------------------
class SectorISR:
    """Precursors E_J psi (IP: E = D^dag, class 0 = a_i for occupied i) or
    C_J psi (EA: class 0 = a+_a for virtual a) with the ordinary overlap as
    metric and the sector Hamiltonian minus E_0 as operator, sequential class
    Gram-Schmidt plus within-class Loewdin as series. Its eigenvalues are IPs
    or EAs directly: there is no second sector and hence no cross block."""

    def __init__(self, us, psi, E, nser, nord, sector):
        self.us, self.nord, self.sector = us, nord, sector
        typ = 'D' if sector == 'IP' else 'C'
        Xops = us.XM if sector == 'IP' else us.XP
        orb0 = [i for i in us.idx[(0, 'o')] if (i < us.nocc) == (sector == 'IP')]
        classes = [np.array(orb0)] + [us.idx[(k, typ)] for k in range(1, us.kmax + 1)
                                      if len(us.idx[(k, typ)])]
        self.kcls = np.concatenate([[k] * len(c) for k, c in enumerate(classes)]).astype(int)
        ops = np.concatenate(classes)
        self.n0 = len(orb0)
        self.labels = [us.ops[i]['lab'] for i in ops]
        key = 'M' if sector == 'IP' else 'P'
        dim = us.bM.dim if sector == 'IP' else us.bP.dim
        Ieye = np.eye(dim)
        Hc = {0: us.F[key] - E[0] * Ieye, 1: (us.H[key] - us.F[key]) - E[1] * Ieye}
        V = [np.column_stack([Xops[i] @ psi[n] for i in ops]) for n in range(nord)]
        nops = len(ops)
        SS = [np.zeros((nops, nops)) for _ in range(nord)]
        MM = [np.zeros((nops, nops)) for _ in range(nord)]
        HcV = {(b, c): Hc[c] @ V[b] for b in range(nord) for c in (0, 1)}
        for a in range(nord):
            for b in range(nord - a):
                k = a + b
                VV = V[a].T @ V[b]
                SS[k] += VV
                for c in (0, 1):
                    if k + c < nord:
                        MM[k + c] += V[a].T @ HcV[(b, c)]
                for c in range(2, nord - k):
                    MM[k + c] -= E[c] * VV
        inv_n = series_reciprocal(nser, nord)
        Ssr, Msr = ndiv(SS, inv_n, nord), ndiv(MM, inv_n, nord)
        loc, o2 = [], 0
        for c in classes:
            loc.append(np.arange(o2, o2 + len(c)))
            o2 += len(c)
        blocks, _ = isr_orthonormalize(Msr, Ssr, loc, nord)
        self.M = [np.zeros((nops, nops)) for _ in range(nord)]
        for ci in range(len(classes)):
            for cj in range(len(classes)):
                for k in range(nord):
                    self.M[k][np.ix_(loc[ci], loc[cj])] = blocks[(ci, cj)][k]

    def truncated(self, n):
        """Effective sector matrix series with block (I, J) through order
        n - k_I - k_J and classes 2k <= n."""
        keep = self.kcls <= n // 2
        rule = n - self.kcls[:, None] - self.kcls[None, :]
        out = []
        for kk in range(self.nord):
            Mk = np.where(rule >= kk, self.M[kk], 0.0)
            out.append(Mk[np.ix_(keep, keep)])
        return out, keep


def eigenvalue_series(Mser, idx, nmax, degenerate_tol=1e-10):
    """Rayleigh-Schroedinger series of the eigenvalue of M(lambda) = sum_k
    lambda^k M_k that starts at the diagonal entry idx of M_0 (M_0 diagonal).
    Partners exactly degenerate with it must be symmetry-decoupled, otherwise
    the state is not perturbatively defined (ValueError)."""
    n = Mser[0].shape[0]
    d0 = np.diag(Mser[0])
    assert maxabs(Mser[0] - np.diag(d0)) < 1e-12, 'M_0 is not diagonal'
    e = np.zeros(n)
    e[idx] = 1.0
    w0 = d0[idx]
    gap = w0 - d0
    degen = (np.abs(gap) < degenerate_tol)
    degen[idx] = True
    R = np.where(degen, 0.0, 1.0 / np.where(degen, 1.0, gap))
    psi = [e]
    w = [w0]
    for k in range(1, nmax + 1):
        rhs = sum(Mser[j] @ psi[k - j] for j in range(1, k + 1) if j < len(Mser))
        w.append(float(e @ rhs))
        rhs = rhs - sum(w[j] * psi[k - j] for j in range(1, k + 1))
        leak = maxabs(rhs[degen & (np.arange(n) != idx)])
        if leak > 1e-9:
            raise ValueError(f'degenerate partner coupled at order {k}: {leak:.2e}')
        psi.append(R * rhs)
    return np.array(w)


def sector_state_series(us, sector, hole_or_particle, E_N, nord):
    """Exact RSPT series of E(N-+1, lambda) - E_0(N, lambda) for the state that
    starts at the Koopmans determinant a_i|HF> (IP) or a+_a|HF> (EA): the exact
    IP / EA Taylor coefficients."""
    key, basis = ('M', us.bM) if sector == 'IP' else ('P', us.bP)
    hf = us.bN.dets[us.hf]
    _, det = (apply_ops(hf, [], [hole_or_particle]) if sector == 'IP'
              else apply_ops(hf, [hole_or_particle], []))
    idx = basis.index[det]
    Hser = [us.F[key], us.H[key] - us.F[key]] + [np.zeros_like(us.F[key]) for _ in range(nord - 2)]
    Esec = eigenvalue_series(Hser, idx, nord - 1)
    EN = np.array([E_N[k] for k in range(nord)])
    return Esec - EN


def principal_poles(omegas, strengths, sign, nstates, tol=1e-8):
    """First nstates distinct principal poles: sign = -1 selects IPs (omega < 0,
    nearest zero first), sign = +1 EAs (omega > 0, ascending). Degenerate poles
    are merged; missing ones are nan."""
    sel = np.where((sign * omegas > 0) & (strengths > PRINCIPAL_STRENGTH))[0]
    sel = sel[np.argsort(sign * omegas[sel])]
    out_w, out_s = [], []
    for i in sel:
        if out_w and abs(omegas[i] - out_w[-1]) < tol:
            continue
        out_w.append(float(omegas[i]))
        out_s.append(float(strengths[i]))
        if len(out_w) == nstates:
            break
    while len(out_w) < nstates:
        out_w.append(float('nan'))
        out_s.append(float('nan'))
    return out_w, out_s
