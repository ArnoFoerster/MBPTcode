"""Spin-blocked spatial tensors for the spin-free EE-ADC route.

A `SB` holds one SPATIAL array per spin case: SB({'abab': X, 'aaaa': Y, ...}),
where the key gives the spin (a/b) of each tensor index in order. Missing key
= zero block. `sb_einsum` then evaluates a spin-orbital einsum by enumerating
the spin assignments of every index and contracting the corresponding spatial
blocks:

    sb_einsum('ikcd,jkcd->ij', g_oovv, t1)

is the same subscript string the spin-orbital route uses, so the ADC working
equations are transcribed ONCE (ee_equations) and evaluated either
spin-orbital (numpy.einsum on full spin-orbital arrays) or spin-free
(sb_einsum on spatial blocks). No spin-summed equation is derived by hand --
the same reasoning as spin_adapt.py's numerical isometry, applied to the
contraction layer instead of the basis.

Blocks that vanish by spin never get written, so the enumeration cost is a
few spatial einsums per term, exactly what a hand spin-summation would
produce. Storage is spatial: a four-index quantity that costs (2n)^4 in spin
orbitals costs a handful of n^4 blocks here.

Antisymmetry convention: `anti4` builds every nonzero block of a
<pq||rs>-antisymmetric quantity from the plain spatial physicist integral
V[p,q,r,s], for a closed-shell reference:

    aaaa = bbbb = V - V(pqsr)     abab = baba = V
    abba = baab = -V(pqsr)
"""
import contextlib
import functools
import itertools
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.Base.utils.threads import (_BlasLimit, _pool_is_ours, blas_single_threaded,
                                    row_map)

_EW_MIN = 1 << 20       # elements: a smaller elementwise pass stays on one thread
_KSUM_MIN = 1 << 22     # elements of the long operand: a smaller product is not split
# GB a streamed kernel's slabs in flight may hold together; the job's environment
# sets it through MBPT_STREAM_GB (one slab needs n * live * per_index * 8 bytes).
_STREAM_GB = float(os.environ.get('MBPT_STREAM_GB', 16.0))
_STREAM_POOLS = {}
_STREAM_LOCK = threading.Lock()


def _threads():
    """The thread pin, OMP_NUM_THREADS, else the cores this process may run on."""
    pin = os.environ.get('OMP_NUM_THREADS', '')
    if pin.isdigit() and int(pin) > 0:
        return int(pin)
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') \
        else (os.cpu_count() or 1)


def _ew(op, a, b, out=None):
    """op(a, b) for a NumPy ufunc op, a an array and b a scalar or an array of a's
    shape, split over the leading axis across the thread pin. A ufunc releases
    the GIL, so the slabs run in parallel; each element sees the same operation,
    so the result equals op(a, b) bit for bit. out may be a, never a view of b
    under another axis order."""
    a = np.asarray(a)
    nt = min(_threads(), a.shape[0] if a.ndim else 1)
    if nt < 2 or a.size < _EW_MIN or (np.ndim(b) and np.shape(b) != a.shape):
        return op(a, b) if out is None else op(a, b, out=out)
    if out is None:
        out = np.empty(a.shape, np.result_type(a, b))

    def slab(r0, r1):
        s = slice(r0, r1)
        op(a[s], b[s] if np.ndim(b) else b, out=out[s])

    row_map(slab, a.shape[0], nt)
    return out


def _ksum(fn, n, size):
    """sum of fn(r0, r1) over ranges tiling [0, n), one per thread of the pin, with
    BLAS held at one thread meanwhile. For a product whose output is small and
    whose summed axis is long, which BLAS splits over the output and so runs on a
    few cores; fn returns the product over its slab of that axis, and runs on a
    pool thread, so it calls neither _ew nor _ksum. size, the long operand's
    element count, keeps a product below _KSUM_MIN whole, where the wrap and the
    dispatch would cost more than the split wins. The partials are added in range
    order, so the sum is reordered against fn(0, n), not random. Off the main
    thread, inside a _stream slab, the product stays whole: the slabs already
    share the cores, and the BLAS count is not this call's to hold at one."""
    nt = min(_threads(), n)
    if (nt < 2 or size < _KSUM_MIN
            or threading.current_thread() is not threading.main_thread()):
        return fn(0, n)
    with blas_single_threaded():
        parts = row_map(fn, n, nt)
    return functools.reduce(np.add, parts)


def _stream_workers():
    """Slabs in flight: one per eight threads of the pin, at least one."""
    return max(1, _threads() // 8)


def _slab_plan(n, per_index, live):
    """(width, workers) for streaming the indices [0, n) in slabs, where one slab
    index carries per_index elements of a slab block and a slab in flight holds
    `live` such blocks at once. The slabs in flight hold workers * width * live *
    per_index float64 together, at most _STREAM_GB, but never less than one index
    of one slab; workers is at most _stream_workers(). A budget that holds every
    index gives one slab on one worker."""
    fit = max(1, int(_STREAM_GB * 1e9 // (8 * per_index * live)))
    if fit >= n:
        return n, 1
    workers = min(_stream_workers(), fit)
    width = fit // workers
    return width, min(workers, -(-n // width))


def _add(a, b):
    """a + b for a tuple of (list of addable, float), elementwise."""
    return ([x + y for x, y in zip(a[0], b[0])], a[1] + b[1])


def _stream_pool(workers):
    """This process's executor of `workers` slab threads, made once. It is not one
    of row_map's: a slab's own _ew passes run on those, and a slab waiting on a
    pool it occupies would never return."""
    key = (os.getpid(), workers)
    with _STREAM_LOCK:
        if key not in _STREAM_POOLS:
            _STREAM_POOLS[key] = ThreadPoolExecutor(workers)
        return _STREAM_POOLS[key]


def _stream(fn, n, width, workers=None):
    """fn(j0, j1) summed over the slabs [j0, j1) of width `width` tiling [0, n), the
    last one shorter. `workers` slabs run at once (default _stream_workers()), each
    worker taking a contiguous group of slabs in turn, with BLAS held at
    _threads() // workers threads meanwhile, so every core is busy while only
    `workers` slabs are alive. fn returns (list of SB or ndarray, float); the
    partial sums are added in slab order within a group and the groups in order,
    so the sum is reordered against fn(0, n), not random. The BLAS count is
    process-global, so only the main thread with threadpoolctl loaded moves it."""
    slabs = [(j0, min(j0 + width, n)) for j0 in range(0, n, width)]
    workers = min(workers or _stream_workers(), len(slabs))

    def group(g0, g1):
        out = fn(*slabs[g0])
        for s in range(g0 + 1, g1):
            out = _add(out, fn(*slabs[s]))
        return out

    if workers < 2:
        return group(0, len(slabs))
    ns = len(slabs)
    groups = [(p * ns // workers, (p + 1) * ns // workers) for p in range(workers)]
    limit = (_BlasLimit(max(1, _threads() // workers)) if _pool_is_ours()
             else contextlib.nullcontext())
    with limit:
        parts = list(_stream_pool(workers).map(lambda g: group(*g), groups))
    return functools.reduce(_add, parts)


class SB:
    """Spin-blocked spatial tensor: {spin string -> ndarray}."""

    __slots__ = ('blocks', 'derived', 'parity')

    def __init__(self, blocks=None, derived=None, parity=None):
        self.blocks = {k: v for k, v in (blocks or {}).items() if v is not None}
        # Optional provenance: {key: (base_key, axis_permutation, sign)} when
        # a block is a signed transpose of another. A kernel that commutes
        # with such a transpose can then run once on the base and permute the
        # RESULT instead of running again -- see DFVvvvKernels.ladder, where
        # it halves the dominant cost. Every arithmetic operation below drops
        # it, so a stale map can never be used.
        self.derived = derived
        # Optional flip parity p = +1 or -1: X[flip(key)] == p * X[key] for
        # every key, flip being alpha <-> beta. An operation on such tensors
        # then forms the alpha-first blocks alone and takes the rest from
        # them (_with_flip). None promises nothing.
        self.parity = parity

    def get(self, key):
        return self.blocks.get(key)

    def keys(self):
        return self.blocks.keys()

    def items(self):
        return self.blocks.items()

    def __repr__(self):
        return f'SB({sorted(self.blocks)})'

    # ---- arithmetic (mirrors what the equations do to plain ndarrays) ----
    def _map(self, fn):
        """fn, a linear map of one block, on every block; on the alpha-first
        half alone when the parity gives the rest."""
        if self.parity is None:
            return SB({k: fn(v) for k, v in self.blocks.items()})
        return SB(_with_flip({k: fn(v) for k, v in self.blocks.items()
                              if k[0] == 'a'}, self.parity), parity=self.parity)

    def _combine(self, other, sign):
        if not isinstance(other, SB):
            return NotImplemented
        if self.parity is not None and self.parity == other.parity:
            half = {k: v for k, v in self.blocks.items() if k[0] == 'a'}
            for k, v in other.blocks.items():
                if k[0] != 'a':
                    continue
                if k not in half:
                    half[k] = _ew(np.multiply, v, sign)
                elif sign > 0:
                    half[k] = _ew(np.add, half[k], v)
                else:
                    half[k] = _ew(np.subtract, half[k], v)
            return SB(_with_flip(half, self.parity), parity=self.parity)
        out = dict(self.blocks)
        for k, v in other.blocks.items():
            # one pass over the block: sign * v would be a second full temporary
            if k not in out:
                out[k] = _ew(np.multiply, v, sign)
            elif sign > 0:
                out[k] = _ew(np.add, out[k], v)
            else:
                out[k] = _ew(np.subtract, out[k], v)
        return SB(out)

    def __add__(self, other):
        return self._combine(other, +1.0)

    def __sub__(self, other):
        return self._combine(other, -1.0)

    def __neg__(self):
        return self._map(lambda v: _ew(np.multiply, v, -1.0))

    def __mul__(self, c):
        return self._map(lambda v: _ew(np.multiply, v, c))

    __rmul__ = __mul__

    def transpose(self, *axes):
        """Permute tensor axes AND the spin string the same way."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return SB({''.join(k[a] for a in axes): v.transpose(axes)
                   for k, v in self.blocks.items()}, parity=self.parity)

    def scale(self, arr):
        """Multiply elementwise by a weight. A plain array is applied to every
        block (a spin-free weight such as an MP denominator); an SB is applied
        block by block (a spin-RESOLVED weight such as an Epstein-Nesbet
        denominator, whose value differs between same- and opposite-spin
        configurations)."""
        if isinstance(arr, SB):
            return SB({k: _ew(np.multiply, v, arr.get(k))
                       for k, v in self.blocks.items() if arr.get(k) is not None})
        return self._map(lambda v: _ew(np.multiply, v, arr))

    def divide(self, arr):
        """scale() by the reciprocal, without forming 1/arr for the whole SB."""
        if isinstance(arr, SB):
            return SB({k: _ew(np.true_divide, v, arr.get(k))
                       for k, v in self.blocks.items() if arr.get(k) is not None})
        return self._map(lambda v: _ew(np.true_divide, v, arr))

    def spin_flip(self):
        """The closed-shell involution alpha <-> beta."""
        tr = str.maketrans('ab', 'ba')
        return SB({k.translate(tr): v for k, v in self.blocks.items()},
                  parity=self.parity)


_FLIP = str.maketrans('ab', 'ba')


def _with_flip(half, parity):
    """The alpha-first blocks `half` completed by their alpha <-> beta
    partners, X[flip(key)] = parity * X[key]: the same array for +1."""
    out = dict(half)
    for k, v in half.items():
        out[k.translate(_FLIP)] = v if parity > 0 else -v
    return out


def sb_einsum(subs, *ops, optimize=True):
    """einsum over spin-blocked tensors -- see the module docstring."""
    lhs, rhs = subs.split('->')
    terms = lhs.split(',')
    if len(terms) != len(ops):
        raise ValueError(f'{subs!r} expects {len(terms)} operands, got {len(ops)}')
    idx = sorted(set(''.join(terms)))
    parities = [getattr(op, 'parity', None) for op in ops]
    # flip eigenvectors in, a flip eigenvector out: its alpha-first half only
    half = bool(rhs) and None not in parities
    out = {}
    for assign in itertools.product('ab', repeat=len(idx)):
        smap = dict(zip(idx, assign))
        if half and smap[rhs[0]] == 'b':
            continue
        arrays = []
        for t, op in zip(terms, ops):
            arr = op.get(''.join(smap[c] for c in t))
            if arr is None:
                break
            arrays.append(arr)
        else:
            key = ''.join(smap[c] for c in rhs)
            val = np.einsum(subs, *arrays, optimize=optimize)
            out[key] = out[key] + val if key in out else val
    if half:
        parity = float(np.prod(parities))
        return SB(_with_flip(out, parity), parity=parity)
    return SB(out)


# ----------------------------------------------------------------------
# constructors
# ----------------------------------------------------------------------

def anti4(V, Vx=None):
    """Every nonzero spin block of a <pq||rs>-antisymmetric quantity from the
    plain spatial physicist integrals, for a closed-shell reference.

    V  = V_spatial[p, q, r, s]
    Vx = V_spatial[p, q, s, r]  (the exchange partner, already transposed back
         into the pqrs axis order). Defaults to V.transpose(0,1,3,2), which is
         only valid when the r and s axes have the same length -- for a mixed
         block such as ooov the partner lives in a DIFFERENT spatial block
         (oovo) and must be passed explicitly."""
    if Vx is None:
        Vx = V.transpose(0, 1, 3, 2)
    same = V - Vx
    neg = -Vx                      # ONE array, referenced twice: writing
                                   # `-Vx` in both slots allocated a SECOND
                                   # copy, another nv^4 (458 MB at nv = 87)
    return SB({'aaaa': same, 'bbbb': same, 'abab': V, 'baba': V,
               'abba': neg, 'baab': neg})


_G_BLOCKS = {'oooo': 'oooo', 'ooov': 'oovo', 'oovv': 'oovv',
             'ovov': 'ovvo', 'ovvv': 'ovvv', 'vvvv': 'vvvv'}


def ovvv_ia(S, X):
    """sum_jbc <ja||bc> X_ijbc for a closed-shell reference, per output spin.

    S : ndarray, shape (nv, no, nv, nv), index order (a, j, b, c)
        The spatial V[j, a, b, c] = <ja|bc>, stored so (j, b, c) is contiguous.
    X : SB of the six doubles blocks, index order (i, j, b, c)

    Returns an SB with blocks 'aa' and 'bb', shape (no, nv), index order
    (i, a). anti4's blocks of <ja||bc> are V, V - Vx and -Vx with
    Vx[j,a,b,c] = V[j,a,c,b]; moving every Vx onto X as a (b, c) transpose
    leaves one V contraction per output spin,

        sigma^a_ia = sum_jbc V_jabc (X^aaaa_ijbc - X^aaaa_ijcb
                                     + X^abba_ijbc - X^abab_ijcb),

    and alpha <-> beta for 'bb': matmuls on reshape views of S, one per j, where
    the block-wise einsum reorders (copies) the o v^3 block once per spin block,
    summed over j-slabs by _ksum. An X with a flip parity needs the 'aa' matmuls
    alone."""
    nv = S.shape[0]
    out = {}
    spins = (('aa', ('aaaa', 'abba', 'abab')), ('bb', ('bbbb', 'baab', 'baba')))
    for s, (same, with_v, with_neg) in (spins if X.parity is None else spins[:1]):
        # Z = X^same - X^same_ijcb + X^v - X^neg_ijcb, left to right, in place
        Z = _ew(np.subtract, X.get(same), X.get(same).transpose(0, 1, 3, 2))
        _ew(np.add, Z, X.get(with_v), out=Z)
        _ew(np.subtract, Z, X.get(with_neg).transpose(0, 1, 3, 2), out=Z)

        def part(j0, j1):
            # sum_jbc Z_ijbc S_ajbc over the slab j0 <= j < j1, one j at a time:
            # S[:, j] is contiguous in (b, c), so no slice of S is copied
            acc = Z[:, j0].reshape(Z.shape[0], -1) @ S[:, j0].reshape(nv, -1).T
            for j in range(j0 + 1, j1):
                acc += Z[:, j].reshape(Z.shape[0], -1) @ S[:, j].reshape(nv, -1).T
            return acc

        out[s] = _ksum(part, Z.shape[1], Z.size)
        del Z                               # before the next spin's Z is formed
    if X.parity is not None:
        return SB(_with_flip(out, X.parity), parity=X.parity)
    return SB(out)


def ooov_ia(G, X):
    """sum_jkb <jk||ib> X_jkab for a closed-shell reference, per output spin.

    G : SB, anti4's six blocks of <jk||ib>, index order (j, k, i, b): 'abab' and
        'baba' hold V_jkib = <jk|ib>, 'baab' and 'abba' -Vx with Vx_jkib =
        <jk|bi>, 'aaaa' and 'bbbb' V - Vx
    X : SB of the six doubles blocks, index order (j, k, a, b)

    Returns an SB with blocks 'aa' and 'bb', shape (no, nv), index order
    (i, a). With aaaa = abab + baab the three blocks of an output spin take two
    contractions,

        sigma^a_ia = sum_jkb V_jkib (X^aaaa + X^abab)_jkab
                     - sum_jkb Vx_jkib (X^aaaa + X^baab)_jkab,

    and alpha <-> beta for 'bb'. Each is a batch of (i, b) x (b, a) products
    over (j, k), summed over j-slabs by _ksum, where the block-wise einsum
    copies every doubles block to bring (j, k, b) together. An X with a flip
    parity needs 'aa' alone."""

    def pair(g, z):
        # sum_kb g_jkib z_jkab over the slab's (j, k)
        no, nv = g.shape[2], g.shape[3]
        return np.matmul(g.reshape(-1, no, nv),
                         z.reshape(-1, nv, nv).transpose(0, 2, 1)).sum(axis=0)

    out = {}
    spins = (('aa', 'abab', 'baab', ('aaaa', 'abab', 'baab')),
             ('bb', 'baba', 'abba', ('bbbb', 'baba', 'abba')))
    for s, kv, kn, (same, with_v, with_n) in (spins if X.parity is None
                                              else spins[:1]):
        V, N = G.get(kv), G.get(kn)
        Xs, Xv, Xn = X.get(same), X.get(with_v), X.get(with_n)

        def part(j0, j1):
            j = slice(j0, j1)
            return pair(V[j], Xs[j] + Xv[j]) + pair(N[j], Xs[j] + Xn[j])

        out[s] = _ksum(part, V.shape[0], Xs.size)
    if X.parity is not None:
        return SB(_with_flip(out, X.parity), parity=X.parity)
    return SB(out)


def ovvv_ijab(S, x):
    """sum_c <ic||ab> x_jc for a closed-shell reference, per output spin block.

    S : ndarray, shape (nv, no, nv, nv), index order (a, j, b, c), as in ovvv_ia
    x : SB with blocks 'aa' and 'bb', shape (no, nv), index order (j, c)

    Returns an SB of the six doubles blocks, index order (i, j, a, b). With
    T^s_ijab = sum_c V_icab x^s_jc, one matmul per spin, every block follows
    by an (a, b) transpose:

        aaaa = T^a - T^a_ijba,   baba = T^a,   baab = -T^a_ijba,

    and alpha <-> beta for bbbb, abab, abba. An x with a flip parity p has
    T^b = p T^a, so one matmul gives every block."""
    nv, no = S.shape[0], S.shape[1]
    Sm = S.reshape(nv, -1)

    def t(xs):
        # [j, (i, a, b)], so every (i, j) slab stays contiguous in (a, b)
        return (xs @ Sm).reshape(no, no, nv, nv).transpose(1, 0, 2, 3)

    swap = (0, 1, 3, 2)

    def anti(T):
        return _ew(np.subtract, T, T.transpose(swap))       # T_ijab - T_ijba

    def neg_swap(T):
        return _ew(np.multiply, T.transpose(swap), -1.0)    # -T_ijba

    if x.parity is not None:
        Ta = t(x.get('aa'))
        Tb = Ta if x.parity > 0 else _ew(np.multiply, Ta, -1.0)
        return SB(_with_flip({'aaaa': anti(Ta), 'abab': Tb,
                              'abba': neg_swap(Tb)}, x.parity),
                  parity=x.parity)
    Ta, Tb = t(x.get('aa')), t(x.get('bb'))
    return SB({'aaaa': anti(Ta), 'bbbb': anti(Tb),
               'baba': Ta, 'abab': Tb,
               'baab': neg_swap(Ta), 'abba': neg_swap(Tb)})


def ovvv_ijab_pij_slab(S, x, j0, j1, out=None):
    """(1 - P_ij) sum_c <ic||ab> x_jc on the slab j0 <= j < j1 of the second index,
    added into `out`.

    S : ndarray, shape (nv, no, nv, nv), index order (c, i, a, b), <ic|ab>, as in
        ovvv_ia
    x : SB with blocks 'aa' and 'bb', shape (no, nv), index order (j, c), without a
        flip parity
    out : SB of the six doubles blocks, shape (no, j1 - j0, nv, nv), index order
        (i, j, a, b), arrays the caller owns, no two sharing memory; zeros if None

    Returns out with X_ijab - X_jiab added in place, X = ovvv_ijab(S, x). With
    T^s_ijab = sum_c V_icab x^s_jc, A^s = T^s on the slab is x^s[j0:j1] @ S
    reordered, both spins in one product, and B^s_ijab = T^s_jiab is x^s @ S[:, j]
    for each j of the slab, on views of S. X_jiab under a spin string is X's block
    of the string with its first two letters swapped, so

        aaaa += A^a - A^a_ijba - B^a + B^a_ijba,
        baba += A^a + B^b_ijba,        baab -= A^a_ijba + B^b,

    and alpha <-> beta for bbbb, abab, abba. Several slabs together form T twice,
    where ovvv_ijab forms it once; one slab over every j takes B^s as A^s with i
    and j swapped and forms it once."""
    if x.parity is not None:
        raise NotImplementedError('ovvv_ijab_pij_slab takes x without a flip parity')
    nv, no = S.shape[0], S.shape[1]
    nj = j1 - j0
    swap = (0, 1, 3, 2)
    if out is None:
        out = SB({k: np.zeros((no, nj, nv, nv))
                  for k in ('aaaa', 'bbbb', 'abab', 'baba', 'abba', 'baab')})
    x2 = np.stack([x.get('aa'), x.get('bb')])                 # (s, j, c)
    # A[s] at [i, j] for j in the slab: (s, i, j, a, b)
    A = ((x2[:, j0:j1].reshape(2 * nj, nv) @ S.reshape(nv, -1))
         .reshape(2, nj, no, nv, nv).transpose(0, 2, 1, 3, 4))
    if nj == no:
        B = A.transpose(0, 2, 1, 3, 4)
    else:
        # B[s] at [i, j], T^s_jiab = sum_c x^s_ic <jc|ab>, one j at a time
        B = np.empty((2, no, nj, nv, nv))
        for jj in range(nj):
            B[:, :, jj] = (x2.reshape(2 * no, nv) @ S[:, j0 + jj].reshape(nv, -1)
                           ).reshape(2, no, nv, nv)
    for s, t, same, mix, mixneg in ((0, 1, 'aaaa', 'baba', 'baab'),
                                    (1, 0, 'bbbb', 'abab', 'abba')):
        a, bs, bt = A[s], B[s], B[t]
        for key, terms in ((same, ((np.add, a), (np.subtract, a.transpose(swap)),
                                   (np.subtract, bs), (np.add, bs.transpose(swap)))),
                           (mix, ((np.add, a), (np.add, bt.transpose(swap)))),
                           (mixneg, ((np.subtract, a.transpose(swap)),
                                     (np.subtract, bt)))):
            o = out.get(key)
            for op, v in terms:
                _ew(op, o, v, out=o)
    return out


def ovvv_ia_slab(S, Y, j0, j1):
    """ovvv_ia for Y the slab j0 <= j < j1 of the doubles' second index.

    S : ndarray, shape (nv, no, nv, nv), index order (a, j, b, c), as in ovvv_ia
    Y : SB of the six doubles blocks, shape (no, j1 - j0, nv, nv), index order
        (i, j, b, c)

    Returns an SB with blocks 'aa' and 'bb', shape (no, nv), index order (i, a):
    the slab's share of ovvv_ia(S, Y_full), which is ovvv_ia on S[:, j0:j1]."""
    return ovvv_ia(S[:, j0:j1], Y)


def ooov_ia_slab(G, Y, j0, j1):
    """ooov_ia for Y the slab j0 <= k < j1 of the doubles' second index.

    G : SB, anti4's six blocks of <jk||ib>, shape (no, no, no, nv), index order
        (j, k, i, b), as in ooov_ia
    Y : SB of the six doubles blocks, shape (no, j1 - j0, nv, nv), index order
        (j, k, a, b)

    Returns an SB with blocks 'aa' and 'bb', shape (no, nv), index order (i, a):
    the slab's share of ooov_ia(G, Y_full), which is ooov_ia on G[:, j0:j1]."""
    return ooov_ia(SB({k: v[:, j0:j1] for k, v in G.items()}), Y)


def g_blocks_sb(V, nocc_spatial, norb_spatial):
    """The six <pq||rs> occupied/virtual blocks the ADC(3) equations touch,
    spin-blocked, from the full spatial physicist tensor V[p,q,r,s]."""
    o = slice(0, nocc_spatial)
    v = slice(nocc_spatial, norb_spatial)
    sl = {'o': o, 'v': v}
    out = {}
    for fam, xfam in _G_BLOCKS.items():
        direct = V[tuple(sl[c] for c in fam)]
        # the exchange partner has the last two families swapped; transpose it
        # back so both arrays carry the same pqrs axis order
        partner = V[tuple(sl[c] for c in xfam)].transpose(0, 1, 3, 2)
        out[fam] = anti4(direct, partner)
    return out


def diag2(X):
    """A spin-diagonal two-index quantity (identity, density block, ...)."""
    return SB({'aa': X, 'bb': X})


def eye2(n):
    return diag2(np.eye(n))


# ----------------------------------------------------------------------
# spin-orbital <-> spin-blocked conversion (validation only)
# ----------------------------------------------------------------------

def _axis_index(family, spin, nocc, norb):
    """Positions of one spin within an ALREADY-SLICED spin-orbital axis: the
    occupied and virtual segments are each interleaved alpha/beta, so spatial
    orbital p sits at 2p + spin within its own segment."""
    n = nocc if family == 'o' else norb - nocc
    return np.arange(n // 2) * 2 + spin


def from_spin_orbital(T, families, nocc, norb):
    """Slice a spin-orbital tensor block (interleaved spins) into spatial spin
    blocks. `families` is a string of 'o'/'v' per axis, matching the block T
    was sliced from (ee_utils.g_blocks' convention)."""
    out = {}
    for assign in itertools.product((0, 1), repeat=len(families)):
        sl = [_axis_index(f, s, nocc, norb) for f, s in zip(families, assign)]
        blk = T[np.ix_(*sl)]
        if np.abs(blk).max() > 0:
            out[''.join('ab'[s] for s in assign)] = blk
    return SB(out)


def to_spin_orbital(sb, families, nocc, norb):
    """Inverse of from_spin_orbital (zero-filled where blocks are absent)."""
    shape = tuple(nocc if f == 'o' else norb - nocc for f in families)
    T = np.zeros(shape)
    for key, blk in sb.items():
        sl = [_axis_index(f, 'ab'.index(s), nocc, norb)
              for f, s in zip(families, key)]
        T[np.ix_(*sl)] = blk
    return T


# ----------------------------------------------------------------------
# unrestricted constructors
# ----------------------------------------------------------------------

def anti4_uhf(D_aa, X_aa, D_bb, X_bb, D_ab, X_ab, D_ba, X_ba):
    """Every nonzero spin block of a <pq||rs>-antisymmetric quantity for an
    UNRESTRICTED reference, where the alpha and beta spatial orbitals differ.

    The eight arguments are plain spatial physicist integrals:

        D_st[p,q,r,s] = <p_s q_t | r_s s_t>        X_st = the same, r <-> s

    all already in the pqrs axis order. Two blocks that `anti4` gets for free
    from a closed shell are now independent: `aaaa != bbbb` and
    `abab != baba`.

    The `ba` pair is passed in rather than transposed out of the `ab` pair.
    The identity <p_b q_a||r_b s_a> = <q_a p_b||s_a r_b> is true, but it
    permutes the ORBITAL FAMILIES along with the indices: applied to an
    `ooov` block it lands in `oovo`, which is a different array of a
    different shape. It is only self-consistent for families symmetric under
    (1,0,3,2) -- oooo, oovv, vvvv -- so relying on it silently produced
    transposed garbage for ooov, ovov and ovvv."""
    return SB({'aaaa': D_aa - X_aa, 'bbbb': D_bb - X_bb,
               'abab': D_ab, 'baba': D_ba,
               'abba': -X_ab, 'baab': -X_ba})


def _axis_index_blockstacked(family, spin, no_a, no_b, nv_a, nv_b):
    """Positions of one spin within an already-sliced BLOCK-STACKED axis.

    ee_driver.spin_orbital_arrays lays a UHF reference out as
    [occ_a, occ_b, virt_a, virt_b], so within the occupied segment alpha runs
    first and beta follows -- contiguous ranges, not the interleaving an RHF
    reference gives."""
    n_a, n_b = (no_a, no_b) if family == 'o' else (nv_a, nv_b)
    return np.arange(n_a) if spin == 0 else n_a + np.arange(n_b)


def from_spin_orbital_uhf(T, families, no_a, no_b, nv_a, nv_b, tol=0.0):
    """Slice a BLOCK-STACKED spin-orbital tensor block into spatial spin
    blocks -- the unrestricted counterpart of from_spin_orbital, and the
    oracle the unrestricted DF integral blocks are checked against."""
    out = {}
    for assign in itertools.product((0, 1), repeat=len(families)):
        sl = [_axis_index_blockstacked(f, s, no_a, no_b, nv_a, nv_b)
              for f, s in zip(families, assign)]
        blk = T[np.ix_(*sl)]
        if blk.size and np.abs(blk).max() > tol:
            out[''.join('ab'[s] for s in assign)] = blk
    return SB(out)
