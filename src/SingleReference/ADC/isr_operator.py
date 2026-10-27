"""Order-general matrix-free intermediate-state representation: the operator
form of isr_assembly.isr_effective_blocks.

Block (I, J) of the ISR matrix is eff_I^T H eff_J, eff_J the orthonormalized
states of class J in the precursor basis: Gram-Schmidt against the lower
classes, then Loewdin within the class, both as series in the perturbation
(Mertins and Schirmer, Phys. Rev. A 53, 2140 (1996)). Here each raw block is a
series of linear maps and the effective states are applied, never formed:

    X_c u      = u - sum_{c' < c} eff_c'( eff_c'^T S X u )   (sequential class GS)
    X_c^T G    = (product of (1 - S eff_c' eff_c'^T), last class first) G, class-c part
    Y_c        = (1 + Delta W_c)^(-1/2),  W_c = X_c^T S X_c   (binomial series)
    eff_c      = X_c Y_c,    eff_c^T = Y_c X_c^T              (Y_c symmetric)

with H and S divided by the ground-state norm series. A matvec costs a fixed
number of raw-block applications per order. Class c's series stop at order
n - k_c, the highest order any of its blocks is carried through (manifold rule),
which is exact for every carried block. Vector series are {order: array};
families are {class: vector series}; the same series arithmetic runs on matrices
(one column per vector).

dense_max (matrix elements) trades memory for the recursion: raw blocks within
it are held as arrays, and the lowest classes whose diagonal blocks fit get
their effective states eff_c precomputed as dense series, so the Gram-Schmidt
and Loewdin series of those classes are not re-run inside every
application of a higher class. dense_max = 0 keeps everything matrix-free.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator

from src.SingleReference.ADC.isr_assembly import norm_divide


def _as_map(A):
    return A if hasattr(A, 'rmatvec') else np.asarray(A, dtype=float)


# columns of the identity applied per matmat call when a map is densified
DENSE_COLUMN_BLOCK = 512


def _apply_map(A, x, transpose):
    """A x or A^T x for an array or a linear map, x one vector or a matrix of columns
    (a matrix goes through the map's matmat, batched when the map provides one)."""
    if isinstance(A, np.ndarray):
        return A.T @ x if transpose else A @ x
    if x.ndim == 2:
        return A.rmatmat(x) if transpose else A.matmat(x)
    return A.rmatvec(x) if transpose else A.matvec(x)


def dense_of(A, shape):
    """The array of a linear map, from its application on the smaller side, the
    identity's columns in blocks of DENSE_COLUMN_BLOCK."""
    n_out, n_in = shape
    d = min(shape)
    M = np.empty(shape)
    for j0 in range(0, d, DENSE_COLUMN_BLOCK):
        j1 = min(j0 + DENSE_COLUMN_BLOCK, d)
        E = np.zeros((d, j1 - j0))
        E[np.arange(j0, j1), np.arange(j1 - j0)] = 1.0
        if n_in <= n_out:
            M[:, j0:j1] = A.matmat(E)
        else:
            M[j0:j1, :] = A.rmatmat(E).T
    return M


def _add(series, m, v):
    series[m] = series[m] + v if m in series else v


def _sub_family(F, G):
    """F - G for two families (series per class)."""
    out = {c: dict(s) for c, s in F.items()}
    for c, s in G.items():
        tgt = out.setdefault(c, {})
        for m, v in s.items():
            _add(tgt, m, -v)
    return out


class OperatorISR:
    """The ISR matrix of raw block series H, S = {(ci, cj): {m: map}} (ci <= cj in
    the order of `dims`, each map an ndarray or anything with matvec/rmatvec,
    class ci rows x class cj columns; S carries the order-0 identity of its
    diagonal blocks), class dimensions `dims` in Gram-Schmidt order, norm
    = {m: N^(m)}, and top = {(ci, cj): order} through which each block is kept.
    matvec applies sum over carried orders of every block; the result equals
    the dense isr_effective_blocks matrix restricted the same way. dense_max:
    see the module docstring."""

    def __init__(self, H, S, dims, top, norm=None, dense_max=0):
        self.classes = list(dims)
        self.dims = dict(dims)

        def held(key, A):
            A = _as_map(A)
            if isinstance(A, np.ndarray) or self.dims[key[0]] * self.dims[key[1]] > dense_max:
                return A
            return dense_of(A, (self.dims[key[0]], self.dims[key[1]]))
        self.H = {k: {m: held(k, A) for m, A in ser.items()} for k, ser in H.items()}
        self.S = {k: {m: held(k, A) for m, A in ser.items()} for k, ser in S.items()}
        self.norm = {m: c for m, c in (norm or {}).items() if m >= 2 and c}
        self.top = {}
        for ci in self.classes:
            for cj in self.classes:
                self.top[(ci, cj)] = max(top.get((ci, cj), -1), top.get((cj, ci), -1))
        # a class's series are needed through the highest order of its blocks
        self.cap = {c: max(self.top[(c, d)] for d in self.classes) for c in self.classes}
        offs, off = {}, 0
        for c in self.classes:
            offs[c] = off
            off += self.dims[c]
        self.offs, self.size = offs, off
        # effective states of the lowest classes as dense series {class: {order: array}},
        # each from the recursion on the identity, the classes below it already stored
        self._eff = {}
        for c in self.classes:
            if self.dims[c] ** 2 > dense_max:
                break
            self._eff[c] = self.eff_fwd(c, {0: np.eye(self.dims[c])}, self.cap[c])

    # ----- raw blocks ------------------------------------------------------
    def _block_apply(self, kind, c1, c2, v, cap):
        """(raw block (c1, c2)) applied to the class-c2 series v, as a series."""
        store = self.H if kind == 'H' else self.S
        if (c1, c2) in store:
            ser, tr = store[(c1, c2)], False
        elif (c2, c1) in store:
            ser, tr = store[(c2, c1)], True
        else:
            return {}
        out = {}
        for a, A in ser.items():
            for b, x in v.items():
                if a + b <= cap:
                    _add(out, a + b, _apply_map(A, x, tr))
        return out

    def _apply(self, kind, F, cap, upto=None):
        """(H or S) applied to the family F, divided by the norm series, onto the
        classes up to `upto` (all by default)."""
        out = {}
        last = len(self.classes) if upto is None else self.classes.index(upto) + 1
        for c1 in self.classes[:last]:
            acc = {}
            for c2, v in F.items():
                for m, w in self._block_apply(kind, c1, c2, v, cap).items():
                    _add(acc, m, w)
            if acc:
                out[c1] = norm_divide(acc, self.norm, cap) if self.norm else acc
        return out

    def _lower(self, c):
        return self.classes[:self.classes.index(c)]

    # ----- effective states ------------------------------------------------
    # eff_d^T reads only the classes <= d, and X_c^T G only the classes <= c, so
    # each overlap application is projected onto those
    def _x_fwd(self, c, u, cap):
        F = {c: u}
        for d in self._lower(c):
            F = _sub_family(F, self.eff_fwd(d, self.eff_adj(d, self._apply('S', F, cap, d), cap),
                                            cap))
        return F

    def _x_adj(self, c, G, cap):
        for d in reversed(self._lower(c)):
            G = _sub_family(G, self._apply('S', self.eff_fwd(d, self.eff_adj(d, G, cap), cap),
                                           cap, c))
        return G.get(c, {})

    def _delta_w(self, c, v, cap):
        """(W_c - 1) v as a series."""
        w = self._x_adj(c, self._apply('S', self._x_fwd(c, v, cap), cap, c), cap)
        for m, x in v.items():
            _add(w, m, -x)
        return {m: x for m, x in w.items() if m <= cap and np.any(x)}

    def _lowdin(self, c, v, cap):
        """Y_c v = sum_k binom(-1/2, k) (Delta W_c)^k v."""
        out, term, coef = dict(v), dict(v), 1.0
        for k in range(1, cap + 1):
            coef *= -(2 * k - 1) / (2 * k)
            term = self._delta_w(c, term, cap)
            if not term:
                break
            for m, x in term.items():
                _add(out, m, coef * x)
        return out

    def eff_fwd(self, c, v, cap):
        """eff_c v: the class-c vector series v in the precursor basis."""
        cap = min(cap, self.cap[c])
        if c in self._eff:
            out = {}
            for d, E in self._eff[c].items():
                acc = {}
                for a, M in E.items():
                    for b, x in v.items():
                        if a + b <= cap:
                            _add(acc, a + b, M @ x)
                if acc:
                    out[d] = acc
            return out
        return self._x_fwd(c, self._lowdin(c, v, cap), cap)

    def eff_adj(self, c, G, cap):
        """eff_c^T G: a family projected on the orthonormalized class-c states."""
        cap = min(cap, self.cap[c])
        if c in self._eff:
            acc = {}
            for d, E in self._eff[c].items():
                for b, x in G.get(d, {}).items():
                    for a, M in E.items():
                        if a + b <= cap:
                            _add(acc, a + b, M.T @ x)
            return acc
        return self._lowdin(c, self._x_adj(c, G, cap), cap)

    # ----- the ISR matrix --------------------------------------------------
    def matvec(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.zeros(self.size)
        for cj in self.classes:
            xj = x[self.offs[cj]:self.offs[cj] + self.dims[cj]]
            if not np.any(xj):
                continue
            capj = max(self.top[(ci, cj)] for ci in self.classes)
            HF = self._apply('H', self.eff_fwd(cj, {0: xj}, capj), capj)
            for ci in self.classes:
                t = self.top[(ci, cj)]
                if t < 0:
                    continue
                for m, v in self.eff_adj(ci, HF, t).items():
                    if m <= t:
                        y[self.offs[ci]:self.offs[ci] + self.dims[ci]] += v
        return y

    def linear_operator(self):
        return LinearOperator((self.size, self.size), matvec=self.matvec, dtype=float)
