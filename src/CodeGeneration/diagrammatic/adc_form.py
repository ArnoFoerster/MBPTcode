"""ADC-form objects and matching for the diagram-ADC generator (numerical layer).

Determinant-basis (string-phase-aligned) builders for the ADC configuration
spaces and the low-order objects, all validated against the Goldstone engine
and the exact oracle:

  - k1 / k2 configuration spaces (2p1h/3p2h for sector I, mirrors for II)
  - U1 (first-order couplings), C1_k1k1 and C1_k1k2 (first-order interaction
    blocks; Slater-Condon, E0^(1)-subtracted diagonal = linked part)
  - U2_k1: SCW83 (44b)/(45a) -- reproduces the engine's U-class orderings at
    machine precision (the engine re-derives the published formulas)
  - known_m4: every 4th-order ADC-form term built from the objects above
  - fit_u3_c2: omega-sampled linear extraction of U3/C2 from the order-4
    residual (exact ADC-form closure on H2, tests/test_diagram_adc_form.py)

Sector-II conventions (SCW83 Eq. 45): resolvents 1/(omega + K-bar), C blocks
enter as -C-bar, U2 as U1 - U2-bar (the minus signs are folded in here so
callers see uniform expressions).
"""
import numpy as np

from src.CodeGeneration.diagrammatic.exact_oracle import (
    h_element, determinants)


def apply_ops(occ_set, ops):
    """Apply (kind, orbital) ops RIGHT to LEFT to an ascending determinant;
    kind 'a' annihilates, 'c' creates.  Returns (frozenset, sign) or None."""
    occ = set(occ_set)
    sign = 1.0
    for kind, p in reversed(ops):
        below = sum(1 for q in occ if q < p)
        if kind == 'a':
            if p not in occ:
                return None
            sign *= (-1) ** below
            occ.remove(p)
        else:
            if p in occ:
                return None
            sign *= (-1) ** below
            occ.add(p)
    return frozenset(occ), sign


class SectorSpaces:
    """k1/k2 determinant spaces + validated low-order ADC objects for one
    sector of one (validation-scale) system."""

    def __init__(self, eps, h1, g, nocc, sector):
        self.eps, self.h1, self.g, self.nocc, self.sector = eps, h1, g, nocc, sector
        norb = len(eps)
        self.norb = norb
        hf = frozenset(range(nocc))
        self.hf = hf
        ehf = sum(eps[p] for p in hf)
        occ, vir = range(nocc), range(nocc, norb)

        self.k1 = []
        if sector == +1:
            for j in occ:
                for k in vir:
                    for l in vir:
                        if l <= k:
                            continue
                        det, sgn = apply_ops(hf, [('c', k), ('c', l), ('a', j)])
                        self.k1.append((det, sgn, (j, k, l)))
        else:
            for j in vir:
                for k in occ:
                    for l in occ:
                        if l <= k:
                            continue
                        det, sgn = apply_ops(hf, [('c', j), ('a', k), ('a', l)])
                        self.k1.append((det, sgn, (j, k, l)))

        self.k2 = []
        if sector == +1:
            for i in occ:
                for j in occ:
                    if j <= i:
                        continue
                    for k in vir:
                        for l in vir:
                            if l <= k:
                                continue
                            for m in vir:
                                if m <= l:
                                    continue
                                det, sgn = apply_ops(
                                    hf, [('c', k), ('c', l), ('c', m),
                                         ('a', i), ('a', j)])
                                self.k2.append((det, sgn))
        else:
            for i in vir:
                for j in vir:
                    if j <= i:
                        continue
                    for k in occ:
                        for l in occ:
                            if l <= k:
                                continue
                            for m in occ:
                                if m <= l:
                                    continue
                                det, sgn = apply_ops(
                                    hf, [('c', i), ('c', j), ('a', k),
                                         ('a', l), ('a', m)])
                                self.k2.append((det, sgn))

        e0_1 = h_element(hf, hf, h1, g) - ehf
        self.n1, self.n2 = len(self.k1), len(self.k2)

        self.U1 = np.array([sgn * g[:, j, k, l]
                            for (_, sgn, (j, k, l)) in self.k1])
        self.K1 = np.array([sum(eps[p] for p in d) - ehf
                            for (d, _, _) in self.k1])
        self.K2 = np.array([sum(eps[p] for p in d) - ehf
                            for (d, _) in self.k2])

        self.C11 = np.zeros((self.n1, self.n1))
        for a, (da, sa, _) in enumerate(self.k1):
            for b in range(a, self.n1):
                db, sb, _ = self.k1[b]
                if a == b:
                    v = h_element(da, da, h1, g) - sum(eps[p] for p in da) - e0_1
                else:
                    v = sa * sb * h_element(da, db, h1, g)
                self.C11[a, b] = self.C11[b, a] = v
        self.C12 = np.zeros((self.n1, self.n2))
        for a, (da, sa, _) in enumerate(self.k1):
            for b, (db, sb) in enumerate(self.k2):
                self.C12[a, b] = sa * sb * h_element(da, db, h1, g)

        self.doubles = []
        for D in determinants(norb, nocc):
            if len(D - hf) == 2:
                cD = h_element(D, hf, h1, g) / (-(sum(eps[p] for p in D) - ehf))
                self.doubles.append((D, cD))

    def u2_k1(self):
        """SCW83 (44b) / (45a) second-order k1 couplings, det phases folded
        in; equals the engine's U-class orderings at machine precision."""
        eps, g, nocc = self.eps, self.g, self.nocc
        norb = self.norb
        occ, vir = range(nocc), range(nocc, norb)
        U2 = np.zeros((self.n1, norb))
        osum, vsum = (occ, vir) if self.sector == +1 else (vir, occ)
        for ic, (det, sgn, (j, k, l)) in enumerate(self.k1):
            t = np.zeros(norb)
            for u in osum:
                for v in osum:
                    gam = g[k, l, u, v] / (eps[k] + eps[l] - eps[u] - eps[v])
                    t += -0.5 * gam * g[:, j, u, v]
            for r in vsum:
                for v in osum:
                    gam_k = g[k, r, v, j] / (eps[k] + eps[r] - eps[v] - eps[j])
                    gam_l = g[l, r, v, j] / (eps[l] + eps[r] - eps[v] - eps[j])
                    t += gam_k * g[:, r, v, l] - gam_l * g[:, r, v, k]
            if self.sector == -1:
                t = -t
            U2[ic] = sgn * t
        return U2

    def m3_parts(self, omega):
        """(C-part, U-part) of M(omega;3) for this sector -- the SCW83
        third-order decomposition (identity gates in the tests)."""
        s = 1.0 if self.sector == +1 else -1.0
        R1 = 1.0 / (omega - s * self.K1)
        C11 = self.C11 if self.sector == +1 else -self.C11
        U2 = self.u2_k1()
        c_part = np.einsum('cp,c,cd,d,dq->pq', self.U1, R1, C11, R1, self.U1)
        u_part = (np.einsum('cp,c,cq->pq', self.U1, R1, U2)
                  + np.einsum('cp,c,cq->pq', U2, R1, self.U1))
        return c_part, u_part

    def known_m4_k1(self, omega):
        """All PURE-k1 4th-order ADC-form terms from validated objects
        (U1/C11/U2 only; no k2 content)."""
        s = 1.0 if self.sector == +1 else -1.0
        R1 = 1.0 / (omega - s * self.K1)
        C11 = self.C11 if self.sector == +1 else -self.C11
        U1 = self.U1
        U2 = self.u2_k1()
        m = np.einsum('cp,c,cd,d,de,e,eq->pq', U1, R1, C11, R1, C11, R1, U1)
        m += np.einsum('cp,c,cd,d,dq->pq', U2, R1, C11, R1, U1)
        m += np.einsum('cp,c,cd,d,dq->pq', U1, R1, C11, R1, U2)
        m += np.einsum('cp,c,cq->pq', U2, R1, U2)
        return m


def fit_u3_c2(spaces, delta, omegas):
    """Joint (both sectors) omega-sampled linear lstsq for U3 and symmetric
    C2 from  Delta(w) == sum_sectors [U3'RU1 + U1'RU3 + U1'R C2 R U1].
    Returns (solution vector, in-sample max residual, design matrix)."""
    norb = spaces[+1].norb
    rows, rhs = [], []
    for iw, w in enumerate(omegas):
        Arow = []
        for s in (+1, -1):
            sp = spaces[s]
            sg = 1.0 if s == +1 else -1.0
            R1 = 1.0 / (w - sg * sp.K1)
            n1 = sp.n1
            nU = norb * n1
            pairs = [(a, b) for a in range(n1) for b in range(a, n1)]
            A = np.zeros((norb * norb, nU + len(pairs)), complex)
            for c in range(n1):
                for p in range(norb):
                    col = c * norb + p
                    A[p * norb:(p + 1) * norb, col] += R1[c] * sp.U1[c]
                    A[p::norb, col] += R1[c] * sp.U1[c]
            csign = 1.0 if s == +1 else -1.0
            for k, (a, b) in enumerate(pairs):
                blk = np.outer(sp.U1[a] * R1[a], sp.U1[b] * R1[b])
                if a != b:
                    blk = blk + blk.T
                A[:, nU + k] = csign * blk.ravel()
            Arow.append(A)
        rows.append(np.hstack(Arow))
        rhs.append(delta[iw].ravel())
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x, np.max(np.abs(A @ x - b)), A
