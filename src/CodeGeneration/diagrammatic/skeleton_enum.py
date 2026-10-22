"""Graph-native skeleton enumeration -- the route-B generator.

Replaces the pdaggerq fixed-end stream (whose internal Wick space cost
247 GB at order 5) by direct orderly enumeration of
the KEPT diagram classes: connected, self-loop-free, one-particle-
irreducible, time-orderable typed multigraphs with the two externals on
distinct marked vertices.  The whole downstream engine (orderings, cuts,
value rule, quartet grouping, atlas) consumes the emitted Skeleton objects
unchanged.

Structure rules (all measured on the orders 2-4 pdaggerq streams,
and gated in
tests/test_diagram_skeleton_enum.py):
  * bra external sits in a CREATED pair, ket external in an ANNIHILATED
    pair -- every block, every order;
  * vertices are 2-created/2-annihilated slots; internal edge =
    (src = creating vertex, dst = annihilating vertex, kind h|p);
    no self-edges (self-loop class), src != dst kept only if the h/p
    precedence DAG is acyclic (Wick terms always admit a time ordering);
  * coefficient law, ZERO free parameters, verified on all 736 exact-iso
    classes of orders 2-4 AND per-class against a brute-force connected
    Wick arbiter at order 5 (H2, 4.15M contractions == exact at 2.7e-14):

        sum_i coeff_i * parity_i = sigma_ket * (1/2)^n_ep * n_ext / |Aut|

    (parity = canonical-routing (-1)^(h+loops) of the written slot
    arrangement; n_ep = number of doubled (src,dst,kind) line groups;
    n_ext = linear extensions; sigma_ket = +1/-1 for vir/occ ket; |Aut| =
    vertex-automorphism order, see _n_automorphisms -- the two halves of
    the textbook symmetry factor).  The emitted representative carries
        coeff = sigma_ket * parity(arrangement) * (1/2)^n_ep * n_ext / |Aut|,
    which the engine's value rule (coeff/n_ext)*sigma_ket*prod(1/D)
    turns into the textbook parity*(1/2)^n_ep/|Aut| per time ordering.

pdaggerq's merged stream may split one diagram class across several
sign-coupled written arrangements (e.g. coefficient pairs {+2,-2},
{+1.5,-0.5}); the parity projection above is exactly what makes the class
content well defined, and the value-level gates confirm equality.
"""
import itertools
from collections import defaultdict
from functools import lru_cache

from src.Base.pdaggerq_interface import OCC_LETTERS, VIR_LETTERS
from src.SingleReference.ADC.adc_order_rule import refuse_above_max_order
from src.CodeGeneration.diagrammatic.goldstone import (
    _BLOCKS, Line, Skeleton, _enumerate_orderings)

# external letters (per _BLOCKS) are excluded from the dummy pools;
# uppercase members of the same index families extend the pools far enough
# for every order this package emits (up to 2n-1 lines of one kind)
_EXTERNALS = {'e', 'f', 'm', 'n'}
_OCC_POOL = sorted(set('ijklopqrst')) + sorted(set('IJKLMNOPQRST'))
_VIR_POOL = sorted(set('abcdghuvwxyz')) + sorted(set('ABCDEFGH'))


def _edge_multisets(n):
    """All degree-exact edge multisets: out(v) = 2 - [v==0],
    in(v) = 2 - [v==1]; edges (src, dst, kind), src != dst, multiplicity
    <= 2 per symbol.  Yields tuples of symbols (sorted by construction)."""
    out_cap = [2 - (v == 0) for v in range(n)]
    in_cap = [2 - (v == 1) for v in range(n)]
    symbols = [(s, d, k) for s in range(n) for d in range(n) if s != d
               for k in ('h', 'p')]

    # remaining out-capacity feasibility: symbols are src-major sorted, so
    # once the recursion passes the last symbol of a src, its out must be 0
    last_of_src = {}
    for i, (s, d, k) in enumerate(symbols):
        last_of_src[s] = i

    edges = []
    out = list(out_cap)
    inn = list(in_cap)

    def rec(i):
        if i == len(symbols):
            if not any(out) and not any(inn):
                yield tuple(edges)
            return
        s, d, k = symbols[i]
        maxm = min(out[s], inn[d], 2)
        for m in range(maxm, -1, -1):
            out[s] -= m
            inn[d] -= m
            edges.extend([symbols[i]] * m)
            if not (last_of_src.get(s) == i and out[s] > 0):
                yield from rec(i + 1)
            del edges[len(edges) - m:]
            out[s] += m
            inn[d] += m

    yield from rec(0)


def _closure(adj, start_mask, n):
    seen = start_mask
    frontier = start_mask
    while frontier:
        nxt = 0
        f = frontier
        while f:
            b = f & -f
            nxt |= adj[b.bit_length() - 1]
            f ^= b
        frontier = nxt & ~seen
        seen |= nxt
    return seen


def _keep(n, edges):
    """connected + time-orderable + 1p-irreducible, all on integer
    bitmasks (the networkx versions dominated the runtime)."""
    adj = [0] * n
    for s, d, k in edges:
        adj[s] |= 1 << d
        adj[d] |= 1 << s
    if _closure(adj, 1, n) != (1 << n) - 1:
        return False

    # h/p precedence DAG acyclic (Kahn over unique edges)
    succ = [0] * n
    indeg = [0] * n
    for s, d, k in edges:
        a, b = (s, d) if k == 'p' else (d, s)
        if not (succ[a] >> b) & 1:
            succ[a] |= 1 << b
            indeg[b] += 1
    stack = [v for v in range(n) if indeg[v] == 0]
    cnt = 0
    while stack:
        v = stack.pop()
        cnt += 1
        m = succ[v]
        while m:
            b = m & -m
            u = b.bit_length() - 1
            m ^= b
            indeg[u] -= 1
            if indeg[u] == 0:
                stack.append(u)
    if cnt != n:
        return False

    # irreducibility: only a multiplicity-1 vertex pair can be severed by
    # removing one line; 0 and 1 must stay connected without it
    pair_mult = defaultdict(int)
    for s, d, k in edges:
        pair_mult[(s, d) if s < d else (d, s)] += 1
    for (a, b), mult in pair_mult.items():
        if mult != 1:
            continue
        A = list(adj)
        A[a] &= ~(1 << b)
        A[b] &= ~(1 << a)
        if not (_closure(A, 1, n) >> 1) & 1:
            return False
    return True


def _canonical(n, edges):
    """Minimum edge tuple over permutations of the unmarked vertices."""
    if n <= 2:
        return tuple(sorted(edges))
    best = None
    for perm in itertools.permutations(range(2, n)):
        m = {0: 0, 1: 1}
        m.update({v: p for v, p in zip(range(2, n), perm)})
        cand = tuple(sorted((m[s], m[d], k) for s, d, k in edges))
        if best is None or cand < best:
            best = cand
    return best


def _n_automorphisms(n, edges):
    """Order of the vertex-automorphism group (marks 0 = bra, 1 = ket
    fixed; permutations of the unmarked vertices preserving the typed edge
    multiset).  The class weight carries 1/|Aut|: equivalent-line pairs
    give the (1/2)^n_ep factor, equivalent VERTICES give this one.  No
    class at orders 2-4 has |Aut| > 1 (why the pdaggerq calibration could
    not see it); the first |Aut| = 2 classes appear at order 5 and were
    caught by the brute-force Wick arbiter (H2, alpha = 2 exactly)."""
    if n <= 2:
        return 1
    es = sorted(edges)
    aut = 0
    for perm in itertools.permutations(range(2, n)):
        m = {0: 0, 1: 1}
        m.update({v: p for v, p in zip(range(2, n), perm)})
        if sorted((m[s], m[d], k) for s, d, k in edges) == es:
            aut += 1
    return aut


def arrangement_parity(factors, lines, bra_label, ket_label):
    """(-1)^(h + loops) of the written slot arrangement with the identity
    routing (created slot i pairs annihilated slot i at each vertex)."""
    ext = {bra_label, ket_label}
    created = [list(c) for c, a in factors]
    annihilated = [list(a) for c, a in factors]
    a_at = {}
    for v, as_ in enumerate(annihilated):
        for s, lbl in enumerate(as_):
            if lbl not in ext:
                a_at[lbl] = (v, s)

    def step(lbl):
        v, s = a_at[lbl]
        return created[v][s]

    h = sum(1 for ln in lines if ln.kind == 'h')
    visited = set()
    for v, as_ in enumerate(annihilated):
        for s, lbl in enumerate(as_):
            if lbl in ext:
                cur = created[v][s]
                while cur not in ext:
                    visited.add(cur)
                    cur = step(cur)
    loops = 0
    for ln in lines:
        if ln.idx in visited:
            continue
        loops += 1
        cur = ln.idx
        while True:
            visited.add(cur)
            cur = step(cur)
            if cur == ln.idx:
                break
    return (-1) ** ((h + loops) % 2)


def _emit(edges, block, n):
    """One canonical edge multiset -> a Skeleton the engine can evaluate."""
    spec = _BLOCKS[block]
    bra, ket = spec['bra'], spec['ket']
    occ_iter = iter(_OCC_POOL)
    vir_iter = iter(_VIR_POOL)
    labels = []
    for s, d, k in edges:
        labels.append(next(occ_iter) if k == 'h' else next(vir_iter))

    created = [[] for _ in range(n)]
    annihilated = [[] for _ in range(n)]
    created[0].append(bra)
    annihilated[1].append(ket)
    lines = []
    for lbl, (s, d, k) in zip(labels, edges):
        created[s].append(lbl)
        annihilated[d].append(lbl)
        lines.append(Line(idx=lbl, kind=k, src=s, dst=d))
    factors = tuple((tuple(c), tuple(a))
                    for c, a in zip(created, annihilated))

    skel = Skeleton(factors=factors, coeff=0.0,
                    bra_label=bra, ket_label=ket,
                    bra_vertex=0, ket_vertex=1,
                    lines=tuple(lines),
                    raw='enum ' + ' '.join(f'{s}{k}{d}' for s, d, k in edges))
    _enumerate_orderings(skel)

    groups = defaultdict(int)
    for s, d, k in edges:
        groups[(s, d, k)] += 1
    n_ep = sum(1 for m in groups.values() if m == 2)
    sigma = 1.0 if ket in VIR_LETTERS else -1.0
    par = arrangement_parity(factors, skel.lines, bra, ket)
    aut = _n_automorphisms(n, edges)
    skel.coeff = sigma * par * skel.n_ext * 0.5 ** n_ep / aut
    return skel


@lru_cache(maxsize=None)
def kept_topologies(order):
    """Canonical kept-class edge multisets at one PT order.

    Block-independent: the kept-class filters never reference the external
    letter families (externals are the fixed marked slots at vertices 0/1),
    and the per-block class counts are measured identical at orders 2-5."""
    refuse_above_max_order(order, 'kept_topologies')
    n = order
    seen = set()
    for edges in _edge_multisets(n):
        if not _keep(n, edges):
            continue
        canon = _canonical(n, edges)
        if canon not in seen:
            seen.add(canon)
    return tuple(sorted(seen))


def enumerate_skeletons(order, block):
    """All kept-class skeletons of one external block at PT `order`,
    one Skeleton per iso class, coefficients from the calibrated law."""
    return [_emit(canon, block, order) for canon in kept_topologies(order)]
