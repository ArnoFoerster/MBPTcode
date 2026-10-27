"""Order-general blockwise ISR assembly: the effective (orthonormalized)
secular-matrix blocks of every precursor class from raw precursor-basis
series, classes orthogonalized in order against the earlier ones, then
Loewdin within the class (isr_engine.lowdin_sqrt_inv_series).
"""
import numpy as np
from src.SingleReference.ADC.isr_engine import lowdin_sqrt_inv_series


def sconv(a, b, nmax, ta=False):
    """Series product {m: sum_{i+j=m} A_i @ B_j} (A_i.T if ta), truncated at
    nmax. Missing orders are zero; result stores only nonempty orders."""
    out = {}
    for i, Ai in a.items():
        Ai_ = Ai.T if ta else Ai
        for j, Bj in b.items():
            m = i + j
            if m > nmax:
                continue
            t = Ai_ @ Bj
            out[m] = out[m] + t if m in out else t
    return out


def _acc(target, extra):
    for m, A in extra.items():
        target[m] = target[m] + A if m in target else A
    return target


def norm_divide(series, norm, nmax):
    """Series division by the ground-state norm <Psi|Psi> = 1 + sum_{m>=2}
    N^(m): out^(m) = A^(m) - sum_{j>=2} N^(j) out^(m-j) -- what the arbiter's
    ndiv_series does globally, applied per block."""
    out = {}
    for m in range(nmax + 1):
        v = series.get(m)
        for j in range(2, m + 1):
            nj = norm.get(j, 0.0)
            if nj and (m - j) in out:
                c = nj * out[m - j]
                v = -c if v is None else v - c
        if v is not None:
            out[m] = v
    return out


def isr_effective_blocks(H, S, dims, nmax, norm=None):
    """Effective ISR blocks through order nmax.

    H, S: {(ci, cj): {order: array}} raw precursor-basis Hamiltonian and
    overlap series for ci <= cj (the transposes are filled in); dims:
    {class: dimension}, in the order the classes are orthogonalized; norm:
    {order: N^(m)}, the ground-state norm series, divided out of every block
    when given.

    Returns (blocks, eff): blocks[(ci, cj)] = {order: array}, the
    orthonormalized secular-matrix block series; eff[ci] = {c1: {order:
    array}}, the effective state family of class ci in the raw basis."""
    classes = list(dims)

    def full(mp):
        F = {}
        for (i, j), ser in mp.items():
            F[(i, j)] = ser
            if i != j:
                F[(j, i)] = {m: A.T for m, A in ser.items()}
        return F

    Hf, Sf = full(H), full(S)
    if norm and any(m >= 2 and norm[m] for m in norm):
        Hf = {k: norm_divide(ser, norm, nmax) for k, ser in Hf.items()}
        Sf = {k: norm_divide(ser, norm, nmax) for k, ser in Sf.items()}

    def op_apply(op, fam):
        """{c1: (op (*) fam)[c1]} for a blockwise operator series op."""
        out = {}
        for c1 in classes:
            acc = {}
            for c2, ser in fam.items():
                if (c1, c2) in op:
                    _acc(acc, sconv(op[(c1, c2)], ser, nmax))
            if acc:
                out[c1] = acc
        return out

    def fam_T_dot(famA, famB_applied):
        """famA^T (*) (applied family) -- plain series."""
        out = {}
        for c1, ser in famA.items():
            if c1 in famB_applied:
                _acc(out, sconv(ser, famB_applied[c1], nmax, ta=True))
        return out

    eff = {}
    for ci in classes:
        d = dims[ci]
        X = {ci: {0: np.eye(d)}}
        for cj in classes:
            if cj == ci:
                break
            SX = op_apply(Sf, X)
            P = fam_T_dot(eff[cj], SX)
            for c1, ser in eff[cj].items():
                corr = sconv(ser, P, nmax)
                Xc = X.setdefault(c1, {})
                for m, A in corr.items():
                    Xc[m] = Xc[m] - A if m in Xc else -A
        W = fam_T_dot(X, op_apply(Sf, X))
        ds = {m: W[m] for m in W if m >= 1}
        Y = lowdin_sqrt_inv_series(ds, nmax) if ds else {0: np.eye(d)}
        eff[ci] = {c1: sconv(ser, Y, nmax) for c1, ser in X.items()}

    blocks = {}
    for ci in classes:
        for cj in classes:
            blocks[(ci, cj)] = fam_T_dot(eff[ci], op_apply(Hf, eff[cj]))
    return blocks, eff


def collect_spinorbital_raw(solver, sector):
    """RAW precursor-basis block series (H, S, dims, norm) for the engine,
    from the spin-orbital BuildSuperMatrixADCn's raw accessors. PRECURSOR
    orientation throughout (NO K2_CLASS_SIGN -- consumers comparing against
    accessors that carry the k2 class sign must apply it themselves), norm-UNDIVIDED
    (pass `norm` to the engine).

    Coverage = the ADC(5) raw content:
      H(0,0) {m: N^(m) diag eps}      S(0,0) {m: N^(m) I}
      H(0,1) {1,2,3}  (k1_raw_S)     S(0,1) {2,3}
      H(1,1) {0: K, 1: C1, 2, 3}      S(1,1) {0: I, 2, 3}   (class_raw_S)
      H(0,2) {1,2,3}  (k2_raw_S)     S(0,2) {1,2,3}
      H(1,2) {1, 2}                   S(1,2) {2, 3}  (cross_class_overlap)
      H(2,2) {0: f-block, 1: C1}      S(2,2) {0: I}
      norm   {0: 1, m: N^(m)} m=2..5  (n_norm_order; complete through 5
                                       since amp_at carries order 4)

    The (0,0) raws CARRY the norm series: S(0,0) = <Psi|{a_p+,a_q}|Psi> =
    delta_pq <Psi|Psi> is EXACT (the anticommutator is a c-number), and the
    F-part of H(0,0) is f_pq <Psi|Psi> = eps-diagonal times the norm the
    same way -- both divide back to {0: identity-like} under the engine's
    norm division, with no spurious -N^(m) tails. The V-part of H(0,0)
    (static-self-energy-like, order >= 2) is DROPPED -- the production
    convention at every level; it cannot touch any block gated at ADC(5)
    orders (it first reaches (0,1) at order 4 through an eff-component that
    the arbiter-validated dense oracle u_dot_orb_isr (tests/u_isr_oracle.py) also omits).

    NOT collected: the spin-orbital (0,1)@4 (that family is restricted-native)
    and the H(0,0) static content (production convention); the ADC(6) layers
    ((1,1)@4, (0,2)@4, (1,2)@3, (2,2)@2, the k3 class) are not in MBPTcode.
    Consumers must not read beyond the collected orders. A class entry is
    omitted entirely when the sector has no configs for it (e.g. H2 IP has no
    k2).
    """
    eps = solver.eps
    norb = solver.norb
    norm = {0: 1.0}
    for m in range(2, 6):
        norm[m] = solver.n_norm_order(m)

    Hm, Sm = {}, {}
    Hm[(0, 0)] = {m: v * np.diag(eps) for m, v in norm.items() if v}
    Sm[(0, 0)] = {m: v * np.eye(norb) for m, v in norm.items() if v}

    n1 = len(solver.class_idx(sector)[0])
    u = {m: solver.k1_raw_S(sector, m) for m in (1, 2, 3)}
    Hm[(0, 1)] = {m: u[m][0].T for m in u}
    Sm[(0, 1)] = {m: u[m][1].T for m in (2, 3)}

    K = solver.k_block(sector)          # gathered (n1, n1) f-part MATRIX
    Hm[(1, 1)] = {0: K, 1: solver.c1_block(sector)}
    Sm[(1, 1)] = {0: np.eye(n1)}
    for m in (2, 3):
        raw_m, S_m = solver.class_raw_S(sector, m)
        if raw_m is not None:
            Hm[(1, 1)][m] = raw_m
        if S_m is not None:
            Sm[(1, 1)][m] = S_m

    dims = {0: norb, 1: n1}
    n2 = len(solver.class_idx_k2(sector)[0])
    if n2:
        k2 = {m: solver.k2_raw_S(sector, m) for m in (1, 2, 3)}
        Hm[(0, 2)] = {m: k2[m][0].T for m in k2}
        Sm[(0, 2)] = {m: k2[m][1].T for m in k2}
        sign = solver.K2_CLASS_SIGN[sector]
        Hm[(1, 2)] = {1: sign * solver.coupling_k1_k2_generated(sector),
                      2: solver.c12_raw2(sector)}
        Sm[(1, 2)] = {2: solver.cross_class_overlap(sector, 2),
                      3: solver.cross_class_overlap(sector, 3)}
        Hm[(2, 2)] = {0: solver.k2_self_block_generated(sector, ham='f'),
                      1: solver.k2_self_block_generated(sector, ham='v')}
        Sm[(2, 2)] = {0: np.eye(n2)}
        dims[2] = n2

    return Hm, Sm, dims, norm
