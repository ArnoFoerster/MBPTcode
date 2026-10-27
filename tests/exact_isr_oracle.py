"""Exact intermediate-state (ISR) blocks in the determinant space, order by order.

What it computes. For one sector (IP or EA) of a small system held by
tests.lambda_oracle.LambdaSelfEnergyOracle, in the Moller-Plesset
partitioning H(lambda) = F + lambda (H - F):

  * rspt_series: the Rayleigh-Schroedinger ground state psi_0..psi_{nord-1},
    E_0..E_nord in intermediate normalization, psi_n = R0 [(W - E1) psi_{n-1}
    - sum_k E_k psi_{n-k}], E_{n+1} = <0|W|psi_n>, and the norm series;
  * build_sector: the raw precursor supermatrix Hsup^(k) and metric Smet^(k)
    over the orbital class and the precursor classes k = 1..kmax (configurations
    in the order of precursor_classes.class_configs), with the anticommutator
    metric <Psi0|{C_I^dag, C_J}|Psi0> and the sector-signed four-term
    superoperator, each divided by the norm series;
  * isr_orthonormalize: sequential class Gram-Schmidt (against the orbitals
    and all lower classes) plus within-class Loewdin orthonormalization, as
    power series -- the intermediate-state construction (Schirmer, Phys. Rev.
    A 43, 4647 (1991)) executed exactly, with no dressing model and no
    generated code.

What it is for. The determinant-space reference for the generated ADC(4)/(5)
blocks, per (class pair, order): raw collector series, effective blocks,
cross-class overlaps, and the per-column orientation of the k = 2 classes
(column_sign_fix).
"""
import numpy as np

from src.SingleReference.ADC.isr_engine import lowdin_sqrt_inv_series
from src.SingleReference.ADC.precursor_classes import class_configs
from src.SingleReference.CC.determinant_space import apply_ops

NORD = 6          # default series orders 0 .. NORD-1


def maxabs(x):
    x = np.asarray(x)
    return float(np.max(np.abs(x))) if x.size else 0.0


def smul(A, B):
    nord = len(A)
    return [sum(A[i] @ B[k - i] for i in range(k + 1)) for k in range(nord)]


def sT(A):
    return [a.T for a in A]


def series_reciprocal(n):
    """Coefficients of 1/n(lambda) for a scalar series with n[0] = 1."""
    assert abs(n[0] - 1.0) < 1e-12
    nord = len(n)
    inv = [1.0] + [0.0] * (nord - 1)
    for k in range(1, nord):
        inv[k] = -sum(n[j] * inv[k - j] for j in range(1, k + 1))
    return inv


def ndiv_series(E, inv_n):
    """Series division by the norm: (E/n)[k] = sum_i inv_n[k-i] E[i]."""
    return [sum(inv_n[k - i] * E[i] for i in range(k + 1)) for k in range(len(inv_n))]


def rspt_series(orc, nord=NORD):
    """psi_0..psi_{nord-1}, E_0..E_nord, norm series n_0..n_{nord-1}, HF index."""
    hf = orc.bN.index[(1 << orc.nocc) - 1]
    H0, HN = orc.Hf['N'], orc.Hfull['N']
    W = HN - H0
    d = H0[hf, hf] - np.diag(H0)
    d[hf] = 1.0
    R0 = 1.0 / d
    R0[hf] = 0.0
    p0 = np.zeros(orc.bN.dim)
    p0[hf] = 1.0
    psi = [p0]
    E = {0: H0[hf, hf], 1: W[hf, hf]}
    for n in range(1, nord):
        v = W @ psi[n - 1] - E[1] * psi[n - 1]
        for k in range(2, n):                      # E_k psi_{n-k}, k = 2..n-1
            v = v - E[k] * psi[n - k]
        pn = R0 * v
        pn[hf] = 0.0
        psi.append(pn)
        E[n + 1] = W[hf, :] @ pn
    nser = [1.0] + [0.0] * (nord - 1)
    for k in range(2, nord):
        nser[k] = float(sum(psi[a] @ psi[k - a] for a in range(1, k)))
    return psi, E, nser, hf


def op_coo(bt, bf, cre, ann):
    rows, cols, vals = [], [], []
    for col, det in enumerate(bf.dets):
        r = apply_ops(det, list(cre), list(ann))
        if r is None:
            continue
        ph, nd = r
        i = bt.index.get(nd)
        if i is None:
            continue
        rows.append(i)
        cols.append(col)
        vals.append(float(ph))
    return np.array(rows, dtype=int), np.array(cols, dtype=int), np.array(vals)


def dense_coo(M):
    r, c = np.nonzero(M)
    return r, c, M[r, c]


def sector_ops_coo(orc, nocc, norb, sector, kmax):
    """xco[J]: COO of the x-operator (dimX x dimN); yco[J]: COO of the stored
    y-operator (dimN x dimY) as (rowsN, colsY, phase)."""
    bN, bM, bP = orc.bN, orc.bM, orc.bP
    bX, bY = (bM, bP) if sector == 'IP' else (bP, bM)
    A = [np.asarray(a.todense()) for a in orc.A]
    Cr = [np.asarray(c.todense()) for c in orc.Cr]
    xco, yco, slices = [], [], []
    for p in range(norb):
        if sector == 'IP':
            xco.append(dense_coo(A[p]))                    # a_p: N -> M
            r, c, v = dense_coo(Cr[p])                     # Cr: N -> P
            yco.append((c, r, v))                          # Cr.T (N x P)
        else:
            xco.append(dense_coo(Cr[p]))                   # a_p^dag: N -> P
            r, c, v = dense_coo(A[p])
            yco.append((c, r, v))                          # A.T (N x M)
    slices.append(slice(0, norb))
    off = norb
    for k in range(1, kmax + 1):
        cols = class_configs(nocc, norb, k, sector)
        nh = k + 1 if sector == 'IP' else k
        npt = 2 * k + 1 - nh
        opl = []
        for j in range(len(cols[0])):
            holes = [int(cols[t][j]) for t in range(nh)]
            parts = [int(cols[t][j]) for t in range(nh, nh + npt)]
            opl.append((parts, holes[::-1]))
        for cre, ann in opl:
            xco.append(op_coo(bX, bN, cre, ann))
            yco.append(op_coo(bN, bY, cre, ann))
        slices.append(slice(off, off + len(opl)))
        off += len(opl)
    return xco, yco, slices, off, bX.dim, bY.dim


def build_sector(orc, eps, nocc, norb, sector, kmax, psi, nser):
    """Raw precursor supermatrix and metric series of one sector, orders
    0..len(psi)-1: (Hsup, Smet, slices, nfull). Smet[k][slices[i], slices[j]]
    is the raw metric block <Psi0|{C_I^dag, C_J}|Psi0>/<Psi0|Psi0> at order k."""
    nord = len(psi)
    xco, yco, slices, nfull, dimX, dimY = sector_ops_coo(orc, nocc, norb, sector, kmax)
    H = {k: {0: orc.Hf[k], 1: orc.Hfull[k] - orc.Hf[k]} for k in ('N', 'M', 'P')}
    Hx = H['M'] if sector == 'IP' else H['P']
    Hy = H['P'] if sector == 'IP' else H['M']

    hN = [[H['N'][c] @ psi[b] for c in (0, 1)] for b in range(nord)]
    VN = np.column_stack(list(psi) + [hN[b][c] for b in range(nord) for c in (0, 1)])
    ncol = VN.shape[1]

    M = np.zeros((nord, nfull, dimX))
    M2 = np.zeros((nord, 2, nfull, dimX))
    Q = np.zeros((nord, nfull, dimY))
    Q2 = np.zeros((nord, 2, nfull, dimY))
    for J in range(nfull):
        rX, cN, vx = xco[J]
        OX = np.zeros((dimX, ncol))
        OX[rX] = vx[:, None] * VN[cN]            # x-operator applied to all columns
        rN, cY, vy = yco[J]
        OY = np.zeros((dimY, ncol))
        OY[cY] = vy[:, None] * VN[rN]            # y-operator transpose applied to all columns
        for b in range(nord):
            M[b, J] = OX[:, b]
            Q[b, J] = OY[:, b]
            for c in (0, 1):
                M2[b, c, J] = OX[:, nord + 2 * b + c]
                Q2[b, c, J] = OY[:, nord + 2 * b + c]

    HX = np.zeros((nord, 2, nfull, dimX))
    HY = np.zeros((nord, 2, nfull, dimY))
    for b in range(nord):
        for c in (0, 1):
            HX[b, c] = M[b] @ Hx[c].T
            HY[b, c] = Q[b] @ Hy[c].T

    Smet = [np.zeros((nfull, nfull)) for _ in range(nord)]
    Hsup = [np.zeros((nfull, nfull)) for _ in range(nord)]
    for a in range(nord):
        for b in range(nord):
            k = a + b
            if k < nord:
                Smet[k] += M[a] @ M[b].T + Q[b] @ Q[a].T
            for c in (0, 1):
                k = a + b + c
                if k >= nord:
                    continue
                if sector == 'IP':
                    Hsup[k] += M[a] @ M2[b, c].T
                    Hsup[k] -= M[a] @ HX[b, c].T
                    Hsup[k] += HY[b, c] @ Q[a].T
                    Hsup[k] -= Q[b] @ Q2[a, c].T
                else:
                    Hsup[k] += M[a] @ HX[b, c].T
                    Hsup[k] -= M[a] @ M2[b, c].T
                    Hsup[k] += Q[b] @ Q2[a, c].T
                    Hsup[k] -= HY[b, c] @ Q[a].T
    inv_n = series_reciprocal(nser)
    return ndiv_series(Hsup, inv_n), ndiv_series(Smet, inv_n), slices, nfull


def isr_orthonormalize(Hsup, Smet, slices):
    """Sequential class Gram-Schmidt plus within-class Loewdin, series-exact.
    Returns blocks[(ci, cj)][k] and the per-class post-GS overlap series."""
    nord = len(Hsup)
    nfull = Hsup[0].shape[0]
    X = [[np.zeros((nfull, sl.stop - sl.start)) for _ in range(nord)] for sl in slices]
    for ci, sl in enumerate(slices):
        X[ci][0][sl, :] = np.eye(sl.stop - sl.start)
    eff, sperp = [], []
    for ci, sl in enumerate(slices):
        Xc = X[ci]
        for cj in range(ci):
            P = smul(smul(sT(eff[cj]), Smet), Xc)
            corr = smul(eff[cj], P)
            Xc = [Xc[k] - corr[k] for k in range(nord)]
        Wm = smul(smul(sT(Xc), Smet), Xc)
        sperp.append(Wm)
        nc = Wm[0].shape[0]
        ds = {k: Wm[k] for k in range(1, nord) if np.max(np.abs(Wm[k])) > 1e-11}
        if ds:
            Y = lowdin_sqrt_inv_series(ds, nord - 1)
            Yser = [Y.get(k, np.zeros((nc, nc))) for k in range(nord)]
        else:
            Yser = [np.eye(nc)] + [np.zeros((nc, nc)) for _ in range(nord - 1)]
        Xc = smul(Xc, Yser)
        eff.append(Xc)
    blocks = {}
    for ci in range(len(slices)):
        for cj in range(len(slices)):
            blocks[(ci, cj)] = smul(smul(sT(eff[ci]), Hsup), eff[cj])
    return blocks, sperp


def column_sign_fix(Mex, Tgen, thr_rel=1e-6):
    """s_j in {+-1} with Mex * s[None, :] ~= Tgen (rows anchored); returns (s, residual)."""
    scale = max(maxabs(Mex), maxabs(Tgen))
    thr = thr_rel * scale
    s = np.ones(Mex.shape[1])
    for j in range(Mex.shape[1]):
        rows = np.where(np.abs(Mex[:, j]) > thr)[0]
        if len(rows) == 0:
            continue
        i = rows[int(np.argmax(np.abs(Mex[rows, j])))]
        if abs(Tgen[i, j]) > 0:
            s[j] = float(np.sign(Tgen[i, j] / Mex[i, j]))
    resid = maxabs(Mex * s[None, :] - Tgen)
    return s, resid
