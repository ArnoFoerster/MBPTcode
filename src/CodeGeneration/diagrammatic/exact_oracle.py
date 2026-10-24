"""Exact dynamic-self-energy oracle for the diagram-ADC gates.

Representation-independent ground truth: full determinant-space
diagonalization of H(lambda) = F + lambda*(H - F) in the N and N+-1 sectors,
exact Lehmann G(omega, lambda), Dyson inversion

    Sigma(omega, lambda) = G0(omega)^-1 - G(omega, lambda)^-1 ,
    Sigma_dyn = Sigma - Sigma_inf(lambda),
    Sigma_inf(lambda) = lambda * <pr||qs> dgamma_rs   (dgamma = gamma - gamma_HF),

then pointwise lambda-Taylor coefficients on a real lambda grid at COMPLEX
omega (Im omega keeps the whole lambda fit disc away from the poles of
Sigma).  The order-n coefficient is the exact diagrammatic M(omega; n) the
Goldstone engine must reproduce.

Self-contained on purpose (Slater-Condon rules included): this file is the
reference for the diagram-ADC generator, independent of the ISR
machinery.  Validation scale only
(dense sector diagonalization per lambda grid point).
"""
import itertools

import numpy as np


# --------------------------------------------------------------------------
# spin-orbital system setup (pyscf)
# --------------------------------------------------------------------------

def spinorb_system(mol, mf):
    """RHF -> spin-orbital (eps, h1, g_antisym, nocc), occupied-first layout
    [occ_a, occ_b, vir_a, vir_b]; g is <pq||rs> (physicist, antisymmetrized).
    """
    from pyscf import ao2mo
    nmo = mf.mo_coeff.shape[1]
    no_sp = mol.nelectron // 2
    h1_sp = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.full(mol, mf.mo_coeff), nmo)  # (pq|rs) chemist

    order = ([(i, 0) for i in range(no_sp)] + [(i, 1) for i in range(no_sp)] +
             [(i, 0) for i in range(no_sp, nmo)] + [(i, 1) for i in range(no_sp, nmo)])
    spat = np.array([s for s, _ in order])
    spin = np.array([s for _, s in order])

    same = spin[:, None] == spin[None, :]
    h1 = np.where(same, h1_sp[np.ix_(spat, spat)], 0.0)

    coul = eri[np.ix_(spat, spat, spat, spat)]        # (ab|cd) with spatial parts
    phys = coul.transpose(0, 2, 1, 3)                 # <pq|rs> = (pr|qs)
    g = phys * (same[:, None, :, None] * same[None, :, None, :])
    g = g - g.transpose(0, 1, 3, 2)                   # <pq||rs>

    eps = np.concatenate([mf.mo_energy[:no_sp], mf.mo_energy[:no_sp],
                          mf.mo_energy[no_sp:], mf.mo_energy[no_sp:]])
    return eps, h1, g, 2 * no_sp


# --------------------------------------------------------------------------
# determinant space with Slater-Condon rules
# --------------------------------------------------------------------------

def determinants(norb, nelec):
    return [frozenset(c) for c in itertools.combinations(range(norb), nelec)]


def _phase(det_sorted, p):
    """Sign of c_p acting on an ascending-ordered determinant:
    (-1)^(number of occupied orbitals below p)."""
    return -1.0 if sum(1 for q in det_sorted if q < p) % 2 else 1.0


def h_element(d1, d2, h1, g):
    """<d1|H|d2> by Slater-Condon (frozensets of spin orbitals)."""
    diff1 = d1 - d2
    if len(diff1) > 2:
        return 0.0
    o1, o2 = sorted(d1), sorted(d2)
    if not diff1:
        val = sum(h1[p, p] for p in o1)
        val += 0.5 * sum(g[p, q, p, q] for p in o1 for q in o1)
        return val
    if len(diff1) == 1:
        (p,), (q,) = sorted(diff1), sorted(d2 - d1)
        common = d1 & d2
        val = h1[p, q] + sum(g[p, r, q, r] for r in common)
        return _phase(o1, p) * _phase(o2, q) * val
    p, q = sorted(diff1)
    r, s = sorted(d2 - d1)
    sgn = (_phase(o1, p) * _phase(sorted(d1 - {p}), q) *
           _phase(o2, r) * _phase(sorted(d2 - {r}), s))
    return sgn * g[p, q, r, s]


def _sector_h(dets, h1, g):
    n = len(dets)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            H[i, j] = H[j, i] = h_element(dets[i], dets[j], h1, g)
    return H


def _annihilation_map(dets_n, dets_m, norb):
    """For each p: entries (row_m, col_n, sign) of c_p mapping
    span(dets_n) -> span(dets_m), dets_m one electron fewer."""
    index_m = {d: i for i, d in enumerate(dets_m)}
    out = {p: [] for p in range(norb)}
    for j, d in enumerate(dets_n):
        srt = sorted(d)
        for p in d:
            out[p].append((index_m[d - {p}], j, _phase(srt, p)))
    return out


class ExactSelfEnergy:
    """Exact Sigma(omega, lambda) machinery for one small system."""

    def __init__(self, eps, h1, g, nocc):
        self.eps, self.h1, self.g, self.nocc = eps, h1, g, nocc
        self.norb = len(eps)
        self.dets_n = determinants(self.norb, nocc)
        self.dets_m = determinants(self.norb, nocc - 1)
        self.dets_p = determinants(self.norb, nocc + 1)
        self.ann = _annihilation_map(self.dets_n, self.dets_m, self.norb)
        self.ann_p = _annihilation_map(self.dets_p, self.dets_n, self.norb)
        self.hf_index = self.dets_n.index(frozenset(range(nocc)))
        self._cache = {}

    def _sector_matrices(self, lam):
        """(H_N, H_{N-1}, H_{N+1}) for H(lam) = (1-lam) F + lam H, built at
        the sector-matrix level (F = sum_p eps_p n_p is diagonal there)."""
        if lam in self._cache:
            return self._cache[lam]
        out = []
        for dets in (self.dets_n, self.dets_m, self.dets_p):
            Hfull = _sector_h(dets, self.h1, self.g)
            F = np.diag([sum(self.eps[p] for p in d) for d in dets])
            out.append((1 - lam) * F + lam * Hfull)
        self._cache[lam] = tuple(out)
        return self._cache[lam]

    def _ground_state(self, lam):
        """Ground state of H(lam), continued analytically for complex lam.

        Real lam: eigh as before.  Complex lam: H(lam) is complex SYMMETRIC
        (not Hermitian); the analytic continuation of every downstream
        matrix element uses the BILINEAR form (transpose, no conjugation) --
        which the rest of this class already does (psi0 enters _b_matrix /
        green / gamma unconjugated) -- so the state must be normalized
        bilinearly, psi^T psi = 1.  The sqrt sign ambiguity cancels: all
        observables are quadratic in psi0.  Conjugated (eigh-style)
        normalization would break analyticity in lam and silently corrupt
        the contour Taylor extraction.
        """
        Hn, _, _ = self._sector_matrices(lam)
        if np.iscomplexobj(Hn):
            En, Cn = np.linalg.eig(Hn)
            i0 = int(np.argmax(np.abs(Cn[self.hf_index, :])))
            v = Cn[:, i0]
            nrm2 = v @ v
            if abs(nrm2) < 1e-8:
                raise ValueError(
                    f"quasi-null bilinear norm at lam={lam} (near an "
                    "exceptional point); shrink the contour radius")
            return En[i0], v / np.sqrt(nrm2)
        En, Cn = np.linalg.eigh(Hn)
        i0 = int(np.argmax(np.abs(Cn[self.hf_index, :])))
        return En[i0], Cn[:, i0]

    def _b_matrix(self, psi0):
        """B[m, p] = <m|c_p|0> over the N-1 determinant basis."""
        B = np.zeros((len(self.dets_m), self.norb), dtype=psi0.dtype)
        for p in range(self.norb):
            for (im, jn, s) in self.ann[p]:
                B[im, p] += s * psi0[jn]
        return B

    def green(self, omega, lam):
        """Exact G_pq(omega, lambda) = <0|c_p R+(omega) c_q^dag|0>
        + <0|c_q^dag R-(omega) c_p|0>; omega may be complex."""
        E0, psi0 = self._ground_state(lam)
        _, Hm, Hp = self._sector_matrices(lam)

        A = np.zeros((len(self.dets_p), self.norb),   # A[P, q] = <P|c_q^dag|0>
                     dtype=psi0.dtype)
        for q in range(self.norb):
            for (in_, jp, s) in self.ann_p[q]:
                A[jp, q] += s * psi0[in_]
        B = self._b_matrix(psi0)

        eyep = np.eye(len(self.dets_p))
        eyem = np.eye(len(self.dets_m))
        X = np.linalg.solve(omega * eyep - (Hp - E0 * eyep), A.astype(complex))
        Y = np.linalg.solve(omega * eyem + (Hm - E0 * eyem), B.astype(complex))
        g_plus = A.T @ X                # [p, q]
        g_minus = (B.T @ Y).T           # (B.T Y)[q, p] -> transpose to [p, q]
        return g_plus + g_minus

    def gamma(self, lam):
        """gamma[p, q] = <0|c_q^dag c_p|0>."""
        _, psi0 = self._ground_state(lam)
        B = self._b_matrix(psi0)
        return (B.T @ B).T

    def sigma_dyn(self, omega, lam):
        """Sigma_dyn(omega, lambda) = G0^-1 - G^-1 - Sigma_inf(lambda)."""
        G = self.green(omega, lam)
        G0inv = np.diag(omega - self.eps.astype(complex))
        sig = G0inv - np.linalg.inv(G)
        gam_hf = np.zeros((self.norb, self.norb))
        gam_hf[:self.nocc, :self.nocc] = np.eye(self.nocc)
        dg = self.gamma(lam) - gam_hf
        sig_inf = lam * np.einsum('prqs,rs->pq', self.g, dg)
        return sig - sig_inf

    def m_coefficients(self, omega, orders, lam_max=0.35, n_lam=17, deg=None):
        """lambda-Taylor coefficients of Sigma_dyn at one (complex) omega.

        Returns {n: (norb, norb) complex array} for n in orders.  Use complex
        omega (e.g. +0.4j offset) so the lambda disc stays pole-free.
        """
        if deg is None:
            deg = max(orders) + 4
        lams = np.linspace(-lam_max, lam_max, n_lam)
        vals = np.array([self.sigma_dyn(omega, l) for l in lams])
        V = np.vander(lams, deg + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(V, vals.reshape(n_lam, -1), rcond=None)
        return {n: coef[n].reshape(self.norb, self.norb) for n in orders}

    def m_coefficients_contour(self, omega, orders, radius=0.3, n_points=32,
                               check_radius=None):
        """lambda-Taylor coefficients via a complex-lambda circle + FFT.

        c_n = (1/N r^n) sum_k Sigma_dyn(omega, r e^{2pi i k/N}) e^{-2pi i nk/N}.
        Exact up to aliasing (c_{n+N} r^N, negligible for N >> max order)
        and machine noise amplified by 1/r^n -- no polynomial-fit
        conditioning, so this is the order >= 5 route (the real-lambda
        Vandermonde fit above floors near 1e-5..1e-8 already at order 4).
        Analyticity in lambda is supplied by _ground_state's
        complex-symmetric continuation (see its docstring).

        check_radius: if given, re-extract on that radius and return
        (coeffs, max abs deviation) -- radius-independence is the built-in
        self-gate that the contour lies inside the analyticity disc of
        THIS omega.  Returns just coeffs when check_radius is None.
        """
        ks = np.arange(n_points)
        lams = radius * np.exp(2j * np.pi * ks / n_points)
        vals = np.array([self.sigma_dyn(omega, l) for l in lams])
        # c_n needs the e^{-2pi i kn/N} kernel = numpy's FORWARD fft (its
        # ifft has the +i kernel and would return c_{N-n}: silently ~0 for
        # small n, and "radius-independent" because zero is).
        coefs = np.fft.fft(vals, axis=0) / n_points
        out = {n: coefs[n] / radius ** n for n in orders}
        if check_radius is None:
            return out
        ref = self.m_coefficients_contour(omega, orders, radius=check_radius,
                                          n_points=n_points)
        err = max(float(np.max(np.abs(out[n] - ref[n]))) for n in orders)
        return out, err


def _tight_rhf(mol):
    """RHF converged to machine-level Brillouin residual.  The Goldstone
    engine's self-loop cancellation assumes EXACT canonical HF (f_ov = 0);
    pyscf's default conv_tol leaves f_ov ~ 1e-6-ish on H4, which shows up
    as a spurious ~3e-8 diagram-vs-exact mismatch at orders >= 3 (H2 is
    immune: f_ov = 0 there by symmetry).  Tight SCF pushes it to ~1e-13."""
    from pyscf import scf
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.run()
    return mf


def h2_sto3g(r=0.74):
    from pyscf import gto
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {r}', basis='sto-3g', verbose=0)
    return mol, _tight_rhf(mol)


def h4_chain(r=1.0):
    from pyscf import gto
    atoms = '; '.join(f'H 0 0 {i * r}' for i in range(4))
    mol = gto.M(atom=atoms, basis='sto-3g', verbose=0)
    return mol, _tight_rhf(mol)
