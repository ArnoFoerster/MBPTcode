"""Goldstone-diagram engine for the diagram-ADC(n) generator.

Generates the time-ordered (Goldstone) expansion of the dynamic self-energy
M_pq(omega) at order n from pdaggerq vertex streams, with the time-ordered
diagram (typed skeleton + vertex linear order) as the central data structure.

Pipeline per external (bra, ket) o/v block:

  1. pdaggerq fixed-end stream <ext| (0.25 g)^n |ext'> with the operator
     placement that is nonzero for that block (see BLOCKS; the forward
     product vanishes identically for occupied externals -- verified
     empirically, probe P2a/P2c).  Bare 'g' is used and every diagram
     containing a within-vertex loop (a repeated index inside one integral
     factor) is dropped: for a canonical-HF reference those self-loop
     diagrams are exactly the mean-field (W) insertions that cancel against
     the -W counter-vertex of the fluctuation potential, so the filtered
     bare-g stream equals the normal-ordered-vertex stream.  (pdaggerq's own
     'v' = bare g minus an occ_repulsion one-body operator mixes two symbolic
     forms whose cancellation simplify() does not complete at order >= 3.)
  2. Each surviving term is one TYPED SKELETON: vertices = integral factors,
     internal lines typed hole/particle by index family, with the merged
     pdaggerq coefficient as its numerator certificate (numerator, sign and
     weight of a Goldstone diagram are skeleton-level data -- they do not
     depend on the time ordering).
  3. Goldstone time orderings of a skeleton = linear extensions of its h/p
     precedence DAG (particle line: creating vertex before annihilating
     vertex; hole line: annihilating vertex before creating vertex).  The
     per-ordering weight is coeff / n_extensions: distinct fixed-end Wick
     contractions of one skeleton are exactly its linear extensions, and
     pdaggerq merges them into one coefficient (verified: the operator
     placements in BLOCKS impose no constraint beyond the internal DAG, so
     every linear extension corresponds to one nonzero contraction).
  4. Cut denominators by construction: the cut between consecutive vertices
     of an ordering crosses internal lines; D = dir*omega + sum eps(holes) -
     sum eps(particles), where dir = +1 if the bra-attachment vertex is on
     the later side of the cut and the ket-attachment on the earlier side,
     -1 for the reverse, 0 if both attachments are on one side.  A diagram's
     omega-cuts all share one dir; its sign classifies the sector
     (M^I 2p1h-type vs M^II 2h1p-type).
  5. Classification/filters, all graph rules: disconnected -> drop;
     self-loop -> drop (step 1); both externals on one vertex -> static
     Sigma(inf) class, excluded from M (kept in the census); a single
     internal line whose removal separates the two attachment vertices ->
     one-particle-reducible, belongs to the Dyson iteration of G, not to M.

Sign conventions (two global calibration bits, fixed ONCE by the order-2/3
gates and then frozen -- see tests/test_diagram_goldstone.py):
  sector_sign[dir]: overall sign multiplying diagrams of each sector.
  FREE_CUT_FLIP: whether omega-free cuts of dir=-1-sector diagrams enter as
  -(sum eps_h - sum eps_p) ("upside-down" reading) instead of as-is.

Everything here is validation-scale dense numerics (explicit index arrays);
production emission reuses the matched ADC objects, not this evaluator.
"""
import re
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from src.CodeGeneration.pdaggerq_interface import pq_helper, OCC_LETTERS, VIR_LETTERS
from src.SingleReference.ADC.adc_order_rule import refuse_above_max_order


# --------------------------------------------------------------------------
# pdaggerq streams
# --------------------------------------------------------------------------

# Operator placements per external (bra, ket) o/v block.  Labels are the
# external index names; 'left'/'right' are pdaggerq operator lists.  The
# placements are the unique (up to equivalence) ones with a nonzero stream:
# an annihilator acting directly on |HF> must hit an occupied level, a
# creator an unoccupied one (probes P2a-P2e).
BLOCKS = {
    ('v', 'v'): dict(bra='e', ket='f', left=['a(e)'], right=['a*(f)']),
    ('o', 'o'): dict(bra='m', ket='n', left=['a*(n)'], right=['a(m)']),
    ('v', 'o'): dict(bra='e', ket='n', left=['a(e)', 'a*(n)'], right=['1']),
    ('o', 'v'): dict(bra='m', ket='f', left=['1'], right=['a*(f)', 'a(m)']),
}


def raw_stream(order, block):
    """Raw pdaggerq terms for the 1-body fixed-end stream at PT `order`.

    block: ('v'|'o', 'v'|'o') for the (bra, ket) external index families.
    Returns (terms, bra_label, ket_label); each term is [coeff_str, fac, ...].
    """
    refuse_above_max_order(order, 'raw_stream')
    spec = BLOCKS[block]
    pq = pq_helper('fermi')
    pq.set_left_operators([spec['left']])
    pq.set_right_operators([spec['right']])
    pq.add_operator_product(0.25 ** order, ['g'] * order)
    pq.simplify()
    terms = pq.strings()
    pq.clear()
    return terms, spec['bra'], spec['ket']


# --------------------------------------------------------------------------
# term parsing -> typed skeleton
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    """One internal propagator line of a skeleton."""
    idx: str    # index label
    kind: str   # 'h' (occupied) or 'p' (virtual)
    src: int    # vertex creating the index
    dst: int    # vertex annihilating the index


@dataclass
class Skeleton:
    """A typed Goldstone skeleton: vertices + typed internal lines + the
    merged pdaggerq coefficient as numerator certificate."""
    factors: tuple            # per vertex: (created labels, annihilated labels)
    coeff: float              # merged pdaggerq coefficient
    bra_label: str
    ket_label: str
    bra_vertex: int           # vertex carrying the bra external index
    ket_vertex: int
    lines: tuple = ()         # Line instances
    orderings: list = field(default_factory=list)   # linear extensions
    raw: str = ''             # printable original term

    @property
    def order(self):
        return len(self.factors)

    @property
    def n_ext(self):
        return len(self.orderings)


class SkeletonError(ValueError):
    pass


def _parse_factor(fac):
    """'g(a,b,c,d)' or '<a,b||c,d>' -> (created, annihilated) label tuples.

    pdaggerq's two-body factor g(p,q,r,s) == <p,q||r,s>: p,q are created
    (bra), r,s annihilated (ket).
    """
    fac = fac.strip()
    m = re.match(r'^(?:g\((.*)\)|<(.*)>)$', fac)
    if not m:
        raise SkeletonError(f"unsupported factor {fac!r}")
    inner = (m.group(1) or m.group(2)).replace('||', ',')
    idxs = [i.strip() for i in inner.split(',')]
    if len(idxs) != 4:
        raise SkeletonError(f"non-two-body factor {fac!r}")
    return tuple(idxs[:2]), tuple(idxs[2:])


def parse_term(term, bra_label, ket_label):
    """One pdaggerq term -> Skeleton, or a str reason for exclusion.

    Exclusion reasons: 'delta' (d(..) factor: disconnected external
    identification), 'self-loop' (mean-field insertion, cancels for
    canonical HF), 'disconnected', 'static' (both externals on one vertex),
    'reducible' (one-particle-reducible between the attachment vertices).
    """
    coeff = float(term[0])
    facs = [f.strip() for f in term[1:]]
    if any(f.startswith('P(') for f in facs):
        raise SkeletonError(f"unexpected P(..) token in 1-body stream: {term}")
    if any(f.startswith('d(') for f in facs):
        return 'delta'

    factors = [_parse_factor(f) for f in facs]

    # self-loop filter (W insertions)
    for created, annihilated in factors:
        labels = list(created) + list(annihilated)
        if len(labels) != len(set(labels)):
            return 'self-loop'

    # locate external attachments; build internal line list
    created_at, annihilated_at = {}, {}
    for v, (created, annihilated) in enumerate(factors):
        for i in created:
            if i in created_at:
                raise SkeletonError(f"label {i} created twice: {term}")
            created_at[i] = v
        for i in annihilated:
            if i in annihilated_at:
                raise SkeletonError(f"label {i} annihilated twice: {term}")
            annihilated_at[i] = v

    externals = {bra_label, ket_label}
    lines = []
    # sorted, not set order: the label strings are hashed, so an unsorted
    # union reorders skel.lines per process and the emitted denominators come
    # out summed in a different (commutative, but not byte-equal) order.
    for idx in sorted(set(created_at) | set(annihilated_at)):
        if idx in externals:
            continue
        if idx not in created_at or idx not in annihilated_at:
            raise SkeletonError(f"unbalanced internal label {idx}: {term}")
        kind = 'h' if idx in OCC_LETTERS else 'p'
        lines.append(Line(idx=idx, kind=kind,
                          src=created_at[idx], dst=annihilated_at[idx]))

    # external attachment vertices: the bra label appears on exactly one
    # factor (created or annihilated depending on the block), same for ket
    def _attach(label):
        hits = [v for v, (c, a) in enumerate(factors) if label in c + a]
        if len(hits) != 1:
            raise SkeletonError(f"external {label} attaches {len(hits)} times: {term}")
        return hits[0]

    bra_vertex = _attach(bra_label)
    ket_vertex = _attach(ket_label)

    skel = Skeleton(factors=tuple(factors), coeff=coeff,
                    bra_label=bra_label, ket_label=ket_label,
                    bra_vertex=bra_vertex, ket_vertex=ket_vertex,
                    lines=tuple(lines), raw=' '.join([term[0]] + facs))

    # connectivity (over internal lines)
    g = nx.MultiGraph()
    g.add_nodes_from(range(len(factors)))
    for ln in lines:
        g.add_edge(ln.src, ln.dst, key=ln.idx)
    if len(factors) > 1 and not nx.is_connected(g):
        return 'disconnected'

    if bra_vertex == ket_vertex:
        return 'static'

    # one-particle-reducible: removing a single internal line separates the
    # attachment vertices
    for ln in lines:
        g.remove_edge(ln.src, ln.dst, key=ln.idx)
        separated = not nx.has_path(g, bra_vertex, ket_vertex)
        g.add_edge(ln.src, ln.dst, key=ln.idx)
        if separated:
            return 'reducible'

    enumerate_orderings(skel)
    return skel


def enumerate_orderings(skel):
    """All Goldstone time orderings = linear extensions of the h/p DAG.

    DAG edge u -> v means u is EARLIER.  Particle line: created before
    annihilated (src -> dst).  Hole line: annihilated before created
    (dst -> src).  Orderings are stored earliest-first.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(range(skel.order))
    for ln in skel.lines:
        if ln.kind == 'p':
            dag.add_edge(ln.src, ln.dst)
        else:
            dag.add_edge(ln.dst, ln.src)
    if not nx.is_directed_acyclic_graph(dag):
        raise SkeletonError(f"cyclic precedence graph (no valid time ordering): {skel.raw}")
    skel.orderings = [list(o) for o in nx.all_topological_sorts(dag)]
    if not skel.orderings:
        raise SkeletonError(f"no linear extension: {skel.raw}")


_SKELETON_CACHE = {}


def build_skeletons(order, block):
    """Full filtered skeleton set for one block at one PT order.

    Returns (skeletons, census) where census counts the exclusion classes.
    Memoized: the stream is system-independent and pdaggerq generation
    dominates the cost (~30 s per block at order 4).
    """
    refuse_above_max_order(order, 'build_skeletons')
    key = (order, block)
    if key in _SKELETON_CACHE:
        return _SKELETON_CACHE[key]
    terms, bra, ket = raw_stream(order, block)
    skels, census = [], {'delta': 0, 'self-loop': 0, 'disconnected': 0,
                         'static': 0, 'reducible': 0, 'kept': 0}
    for term in terms:
        out = parse_term(term, bra, ket)
        if isinstance(out, str):
            census[out] += 1
        else:
            census['kept'] += 1
            skels.append(out)
    _SKELETON_CACHE[key] = (skels, census)
    return skels, census


# --------------------------------------------------------------------------
# skeleton isomorphism grouping (census / atlas support)
# --------------------------------------------------------------------------

def skeleton_hash(skel):
    """WL hash of the typed multigraph, external attachments marked."""
    g = nx.DiGraph()
    for v in range(skel.order):
        role = ''
        if v == skel.bra_vertex:
            role += 'B'
        if v == skel.ket_vertex:
            role += 'K'
        g.add_node(f'v{v}', type=f'vertex{role}')
    for k, ln in enumerate(skel.lines):
        mid = f'l{k}'
        g.add_node(mid, type=ln.kind)
        g.add_edge(f'v{ln.src}', mid)
        g.add_edge(mid, f'v{ln.dst}')
    return nx.weisfeiler_lehman_graph_hash(g, node_attr='type')


# --------------------------------------------------------------------------
# numerical assembly of M(omega; n)
# --------------------------------------------------------------------------

# Frozen calibration bits (fixed by the n=2/3 gates; see module docstring).
#
# GLOBAL_SIGN: one overall sign relating the pdaggerq vacuum-expectation sign
# to the self-energy diagram value.  The omega-cut denominators are built in
# STANDARD form D = omega + dir*(sum eps_h - sum eps_p), which makes the
# sector-II (dir=-1) diagrams come out with the same global sign as sector I
# at every omega-cut count (a dir*omega form would instead need a
# (-1)^(number of omega-cuts) bookkeeping).  Calibrated at n=2: -1.
#
# Diagram value rule (the complete convention, no free parameters left):
#
#     value(ordering) = (coeff / n_ext) * sigma_ket * prod_cuts 1/D_cut,
#     D_cut = dir*omega + sum eps(holes crossed) - sum eps(particles crossed),
#     sigma_ket = +1 if the ket external index is virtual, -1 if occupied.
#
# Calibrated at n=2 elementwise on H2 AND H4 (H4 essential: H2's vv/sector-I
# and oo/sector-II entries vanish by parity -- a "two zeros compared" trap)
# and CONFIRMED at n=3 against the exact lambda^3 self-energy: this exact
# form also reproduces SCW83's published third-order objects including the
# Eq. (45b) sign C^II = -C^(1)-bar, which any per-sector constant-sign
# variant gets wrong once two omega-cuts occur.  sigma_ket is the hole-line
# parity of the amputated ket external line ((-1)^h counts it; the
# pdaggerq vacuum-expectation sign does not).  Free cuts (dir = 0) enter
# as-is -- no upside-down flip (empirically excluded at n=3).
def ordering_sign(skel, sector):
    """sigma_ket (see the value-rule comment; `sector` unused, kept for the
    call-site symmetry with ordering_sector)."""
    return 1.0 if skel.ket_label in VIR_LETTERS else -1.0


def ordering_cuts(skel, ordering):
    """Cut data for one ordering: list of (dir, holes, particles) with
    holes/particles = index labels of internal lines crossing the cut.
    Ordering is earliest-first; cut k separates ordering[:k+1] (earlier)
    from ordering[k+1:] (later)."""
    pos = {v: i for i, v in enumerate(ordering)}
    cuts = []
    for k in range(len(ordering) - 1):
        holes, particles = [], []
        for ln in skel.lines:
            a, b = pos[ln.src], pos[ln.dst]
            if (a <= k) != (b <= k):
                (holes if ln.kind == 'h' else particles).append(ln.idx)
        bra_later = pos[skel.bra_vertex] > k
        ket_later = pos[skel.ket_vertex] > k
        if bra_later and not ket_later:
            d = +1
        elif ket_later and not bra_later:
            d = -1
        else:
            d = 0
        cuts.append((d, tuple(holes), tuple(particles)))
    return cuts


def ordering_sector(skel, ordering):
    """+1 (bra-attachment later: 2p1h-type / M^I) or -1 (M^II).  Every
    omega-cut of one ordering shares the same dir."""
    pos = {v: i for i, v in enumerate(ordering)}
    return +1 if pos[skel.bra_vertex] > pos[skel.ket_vertex] else -1


class BlockEvaluator:
    """Dense numerical evaluator for M_block(omega; n) on a validation-scale
    system.

    eps: (norb,) HF spin-orbital energies; g: (norb,)*4 antisymmetrized
    <pq||rs>; nocc: number of occupied spin orbitals.  Index families are
    resolved to occ/vir slices; external axes come first in the result.
    """

    def __init__(self, eps, g, nocc):
        self.eps = np.asarray(eps, float)
        self.g = np.asarray(g, float)
        self.nocc = nocc
        self.o = slice(0, nocc)
        self.v = slice(nocc, len(eps))

    def _sl(self, label):
        return self.o if label in OCC_LETTERS else self.v

    def _numerator(self, skel):
        """Product of integral factors as one dense array over
        (bra, ket, *dummies); returns (array, dummy label list)."""
        dummies = sorted({ln.idx for ln in skel.lines})
        axis = {skel.bra_label: 0, skel.ket_label: 1}
        for i, d in enumerate(dummies):
            axis[d] = 2 + i
        subs = []
        ops = []
        for created, annihilated in skel.factors:
            labels = list(created) + list(annihilated)
            ops.append(self.g[tuple(self._sl(l) for l in labels)])
            subs.append(''.join(chr(97 + axis[l]) for l in labels))
        out = ''.join(chr(97 + i) for i in range(2 + len(dummies)))
        sig = ','.join(subs) + '->' + out
        return np.einsum(sig, *ops, optimize=True), dummies

    def _cut_denominator(self, cut, dummies, omega, sector):
        """Reciprocal denominator array broadcast over (bra, ket, *dummies)."""
        d, holes, particles = cut
        shape = [1, 1] + [1] * len(dummies)
        D = np.zeros(shape, dtype=complex)
        D += d * omega
        for lbl in holes + particles:
            ax = 2 + dummies.index(lbl)
            e = self.eps[self._sl(lbl)]
            sh = list(shape)
            sh[ax] = e.size
            sign = 1.0 if lbl in holes else -1.0
            D = D + sign * e.reshape(sh)
        return D

    def eval_skeleton(self, skel, omega, sectors=(+1, -1),
                      ordering_filter=None):
        """Sum over this skeleton's orderings at one omega value.
        Returns the (nbra, nket) block contribution.

        ordering_filter: optional predicate (skel, ordering) -> bool
        restricting the sum to a subclass of time orderings (e.g. the
        seven-line-cut quartets of the delta extraction)."""
        num, dummies = self._numerator(skel)
        weight = skel.coeff / skel.n_ext
        total = np.zeros(num.shape[:2], dtype=complex)
        for ordering in skel.orderings:
            sec = ordering_sector(skel, ordering)
            if sec not in sectors:
                continue
            if ordering_filter is not None and not ordering_filter(skel, ordering):
                continue
            val = num.astype(complex)
            for cut in ordering_cuts(skel, ordering):
                val = val / self._cut_denominator(cut, dummies, omega, sec)
            w = weight * ordering_sign(skel, sec)
            total += w * val.sum(axis=tuple(range(2, val.ndim)))
        return total

    def eval_block(self, skels, omega, sectors=(+1, -1), ordering_filter=None):
        out = None
        for skel in skels:
            contrib = self.eval_skeleton(skel, omega, sectors=sectors,
                                         ordering_filter=ordering_filter)
            out = contrib if out is None else out + contrib
        return out


def assemble_m(eps, g, nocc, order, omega_values, sectors=(+1, -1),
               ordering_filter=None):
    """Full M(omega; order) matrix over all four external blocks.

    Returns an (n_omega, norb, norb) array in the given spin-orbital basis.
    ordering_filter restricts the time-ordering sum (see eval_skeleton).
    """
    norb = len(eps)
    ev = BlockEvaluator(eps, g, nocc)
    out = np.zeros((len(omega_values), norb, norb), dtype=complex)
    ranges = {'o': np.arange(nocc), 'v': np.arange(nocc, norb)}
    for block in BLOCKS:
        skels, _ = build_skeletons(order, block)
        if not skels:
            continue
        rb, rk = ranges[block[0]], ranges[block[1]]
        for iw, w in enumerate(omega_values):
            blk = ev.eval_block(skels, w, sectors=sectors,
                                ordering_filter=ordering_filter)
            if blk is not None:
                out[np.ix_([iw], rb, rk)] += blk[None]
    return out


def sevenline_filter(skel, ordering):
    """True for time orderings with a seven-line (4p3h/4h3p) omega-cut --
    the SCW83 Appendix-A quartet class that carries the content missing
    from the ISR-ADC(4) representation (delta = diagram - ISR)."""
    return any(d != 0 and len(h) + len(p) >= 7
               for d, h, p in ordering_cuts(skel, ordering))
