"""Order-by-order ADC(n) assembly rule -- the piece that makes the generator
genuinely ADC(n) instead of needing a hand-written `_uN_split_defs` per order.

Everything else in the pipeline was already order-general: pdaggerq derives any
tag structure and PRUNES the vanishing ones automatically (measured: of the
order-2 candidates only l2v/l2ft2/l1f/s2_l1 survive; vt2, ft1/2/3, l2f, l3f,
l2t2, t*, l2, l3 all emit 0 blocks -- and of the order-3 candidates vt3, l3v,
l2ft3, l1ft2, l2t3, l1t2 likewise emit 0). Spin-blocking (spin_labels), DF
dressing, Laplace fusion (rank-general) and the matrix-free sigma backend are
all order-agnostic too.

The ONLY thing that was hand-written is which (tag structure x amplitude-order
assignment) pairs make up U^(n)/S^(n). That is what this module computes:

    U^(n) = sum over surviving structures (l_j, ham, t_k)
            sum over (a, b) with a + b + [ham == 'v'] == n,
                       a >= mp_min_order(j), b >= mp_min_order(k)
            block(structure) evaluated at (l_j^(a), t_k^(b))

`ham == 'v'` costs one order (the two-body perturbation); `f` costs none.
Amplitude of rank r first exists at MP order `mp_min_order(r)` -- rank 2 at
order 1, ranks 1/3/4 at order 2 (singles are Brillouin-suppressed at order 1),
ranks 5/6 at order 3, ...

VALIDATED: for n = 3 this reproduces the hand-derived U^(3) assembly's
raw/S^(3) term list EXACTLY -- all 9 raw terms and all 3 S^(3) terms, including
l2ft2 appearing TWICE at (l2^1, t2^2) and (l2^2, t2^1), with nothing missing and
nothing extra (tests/test_adc_order_rule.py).
"""
from itertools import product
from math import ceil

# Highest ADC order the generators (src/CodeGeneration) emit for. Order-general
# emitters take an order argument; above this they refuse rather than emit.
MAX_ADC_ORDER = 5


def refuse_above_max_order(n, what):
    """Raise ValueError if ADC(n) is above MAX_ADC_ORDER; `what` names the caller."""
    if n > MAX_ADC_ORDER:
        raise ValueError(f"{what}: ADC({n}) is above ADC({MAX_ADC_ORDER}), the highest "
                         "order the generators emit")


def mp_min_order(rank):
    """Lowest MP order at which an amplitude of this excitation rank exists.

    rank 0 (absent) -> 0; rank 1 -> 2 (no singles at order 1, Brillouin);
    rank 2 -> 1; rank r>=3 -> ceil(r/2)  (Psi^(n) carries ranks up to 2n)."""
    if rank == 0:
        return 0
    if rank == 1:
        return 2
    return max(1, ceil(rank / 2))


def structure_name(j, ham, k):
    """Tag-structure name as the generators/driver spell it, e.g.
    (2,'v',2) -> 'l2vt2', (0,'v',1) -> 'vt1', (1,'f',0) -> 'l1f'."""
    return (f'l{j}' if j else '') + ham + (f't{k}' if k else '')


def _assignments(j, k, budget):
    """(a, b) amplitude-order splits of `budget` respecting existence."""
    for a in range(budget + 1):
        b = budget - a
        if (j == 0) != (a == 0):
            continue
        if (k == 0) != (b == 0):
            continue
        if j and a < mp_min_order(j):
            continue
        if k and b < mp_min_order(k):
            continue
        yield a, b


def u_order_terms(n, max_rank=4, hams=('v', 'f')):
    """Every (name, j, ham, k, l_order, t_order) contributing to U^(n)/C^(n).

    Includes structures that pdaggerq may prune -- the caller simply skips the
    ones with no generated function (getattr(..., None)), exactly as the
    existing drivers already do. That is what keeps this order-general: the
    physics selection is pdaggerq's, the order bookkeeping is here."""
    terms = []
    for j, k in product(range(max_rank + 1), repeat=2):
        for ham in hams:
            budget = n - (1 if ham == 'v' else 0)
            if budget < 0:
                continue
            for a, b in _assignments(j, k, budget):
                terms.append((structure_name(j, ham, k), j, ham, k, a, b))
    return terms


def s_order_terms(n, max_rank=4):
    """Every (name, j, k, l_order, t_order) contributing to the within-class
    overlap S^(n) (no hamiltonian factor, so no order cost)."""
    terms = []
    for j, k in product(range(max_rank + 1), repeat=2):
        if j == 0 and k == 0:
            continue
        for a, b in _assignments(j, k, n):
            nm = (f'l{j}' if j else '') + (f't{k}' if k else '')
            terms.append((nm, j, k, a, b))
    return terms


def structures_up_to(n, max_rank=4):
    """Distinct tag structures reachable at or below order n -- what a
    generate_adc_u(n) should emit (pdaggerq prunes the impossible ones)."""
    seen = {}
    for order in range(1, n + 1):
        for nm, j, ham, k, _, _ in u_order_terms(order, max_rank):
            seen.setdefault(nm, (j, ham, k))
    return seen


def s_structures_up_to(n, max_rank=4):
    """Same for the within-class overlap S structures (no hamiltonian)."""
    seen = {}
    for order in range(1, n + 1):
        for nm, j, k, _, _ in s_order_terms(order, max_rank):
            seen.setdefault(nm, (j, k))
    return seen


def new_structures_at(n, max_rank=4, kind='u'):
    """Tag structures FIRST reachable at order n (what the order-n generator
    module should contain; lower-order ones live in the lower-order modules).

    NOTE the rule places a structure at its pure order-counting order, which
    for the two always-pruned f-family cases (ft2, l2f) is one order lower than
    the hand-written _u2_split_defs put them -- immaterial, since pdaggerq
    emits 0 blocks for both. Generation is a superset game: emitting a
    structure that prunes costs nothing, missing one that survives would be a
    bug, and the ASSEMBLY rule (u_order_terms) is what is exactness-critical
    (validated against u3_generated)."""
    up = structures_up_to if kind == 'u' else s_structures_up_to
    prev = set(up(n - 1, max_rank)) if n > 1 else set()
    return {nm: spec for nm, spec in up(n, max_rank).items() if nm not in prev}


def struct_home_order(name, kind, max_order=8):
    """The order at which tag structure `name` FIRST appears (new_structures_at),
    or None. kind 'u' (or 'c', the class-class K+C element, which shares the
    hamiltonian structures) or 's' (overlap)."""
    k = 's' if kind == 's' else 'u'
    for o in range(1, max_order + 1):
        if name in new_structures_at(o, kind=k):
            return o
    return None


def struct_ungenerated(name, kind, max_generated):
    """True when `name` is missing because its home order is above the highest
    GENERATED module order (a resolver returning None would silently
    UNDER-COMPUTE), False when a module that could hold it exists and pdaggerq
    pruned it there (legitimately null). The always-pruned f-family pair sits
    at order 1 by order counting; within range either way, so 'pruned'."""
    home = struct_home_order(name, kind)
    return home is not None and home > max_generated


# ----------------------------------------------------------------------
# MANIFOLD axis: which excitation classes exist at ADC(n), and to what order
# each supermatrix block is carried.
#
# The module above is order-general along the AMPLITUDE/dressing axis
# (l_j, ham, t_k) only. Which CLASSES appear -- 2h1p, 3h2p, 4h3p, ... -- and
# the order of each block used to be hand-wired per level in the runtime
# (`has_c2 = level in ('adc4_base','adc4')`, `has_k2 = ...`), i.e. hand-added per
# order. That is what these functions replace.
#
# RULE (Schirmer's ISR block-order rule): the supermatrix block between
# excitation classes of level mu_I and mu_J is required through order
#
#       order(I, J) = n - mu_I - mu_J
#
# with mu = 0 for the orbital (1h/1p) block, 1 for 2h1p/2p1h, 2 for 3h2p/3p2h,
# k for the (k+1)h-kp class. A block with order(I, J) < 0 is absent; a block
# with order(I, J) == 0 carries only its zeroth-order part (the K diagonal).
#
# VALIDATED against every hand-wired fact in this repo (test_adc_order_rule):
#   ADC(4): orb-2h1p -> 3 (U^(3)); 2h1p-2h1p -> 2 (C^(2)); orb-3h2p -> 2
#           (U2_3h2p); 2h1p-3h2p -> 1 (C135, first order); 3h2p-3h2p -> 0,
#           i.e. K ONLY -- which is precisely why the downfold's Schur
#           complement over the 3-particle manifold is EXACT at ADC(4).
#   ADC(3): kmax = 1, so no 3h2p at all -- matches level 'adc3'.
#   ADC(5): 3h2p-3h2p -> 1, the "3h2p x 3h2p self-coupling" that is the last block to make matrix-free.
#           Predicted, not assumed.
# ----------------------------------------------------------------------


def kmax(n):
    """Highest excitation class mu present in ADC(n).

    A class-mu manifold reaches the self-energy through U_mu G U_mu, i.e. at
    order 2*mu, so it contributes iff 2*mu <= n. Gives mu<=1 for ADC(2)/(3)
    (2h1p only), mu<=2 for ADC(4)/(5) (+3h2p)."""
    return n // 2


def block_order(n, mu_i, mu_j):
    """Order through which the (mu_i, mu_j) block is carried in ADC(n).
    Negative means the block is absent from ADC(n) entirely."""
    return n - mu_i - mu_j


def blocks_at(n):
    """Every supermatrix block of ADC(n) as (mu_i, mu_j, order), mu_i <= mu_j.

    This is the object a general `generate_adc(n)` should loop over: for each
    block it knows the class pair (-> precursor_classes builds the operator
    strings for any (k, sector)) and the order (-> u_order_terms / s_order_terms
    enumerate the amplitude assignments). Nothing per-order is hand-listed."""
    out = []
    K = kmax(n)
    for mu_i in range(K + 1):
        for mu_j in range(mu_i, K + 1):
            o = block_order(n, mu_i, mu_j)
            if o >= 0:
                out.append((mu_i, mu_j, o))
    return out


def cross_block_leading_order(k_i, k_j):
    """Lowest order at which the union-space Hamiltonian block between a
    (k_i+1)h k_i p-dagger class (D-type) and a (k_j+1)p k_j h class (C-type)
    is nonzero: the two-body perturbation must bridge the rank-(k_i+k_j+1)
    excitation the pair creates, i.e. ceil((k_i+k_j+1)/2). Orders 0 and 1
    vanish for every pair (Tarantelli and Cederbaum, PRA 45, 2790 (1992),
    Eq. 4.10); the k1-k1 block starts at order 2. The metric cross blocks
    vanish identically at every order."""
    return -(-(k_i + k_j + 1) // 2)


def cross_block_order(n, k_i, k_j):
    """Order through which the D_{k_i} x C_{k_j} cross block is carried in
    union-space ADC(n): the manifold rule n - k_i - k_j, or -1 when that is
    below the block's leading order (block absent). k1-k1 enters at n=4,
    k1-k2 at n=5, k2-k2 and k1-k3 at n=7, k2-k3 and k1-k4 at n=8, k3-k3 at
    n=10."""
    o = block_order(n, k_i, k_j)
    return o if o >= cross_block_leading_order(k_i, k_j) else -1


def cross_blocks_at(n):
    """Every D x C cross block of union-space ADC(n) as (k_i, k_j, order),
    k_i the D-type (IP) class, k_j the C-type (EA) class, both >= 1 (the
    orbital class has one character and no cross partner). Classes obey the
    same 2k <= n rule as the same-sector blocks."""
    out = []
    K = kmax(n)
    for k_i in range(1, K + 1):
        for k_j in range(1, K + 1):
            o = cross_block_order(n, k_i, k_j)
            if o >= 0:
                out.append((k_i, k_j, o))
    return out


def blocks_at_union(n):
    """blocks_at(n) with a sector axis: ('same', mu_i, mu_j, order) for the
    blocks both constructions share (each same-sector block appears once per
    sector at runtime) and ('cross', k_i, k_j, order) for the D x C blocks
    the two-sector convention deletes."""
    return ([('same',) + b for b in blocks_at(n)]
            + [('cross',) + b for b in cross_blocks_at(n)])


def amplitude_max_rank(n):
    """LOOSE order-counting bound on amplitude rank in ADC(n): r <= 2*(n-1).

    Use ONLY as the `max_rank` GENERATION cutoff -- generation must be a
    superset and pdaggerq prunes the rest. Do NOT quote it as the rank that
    actually appears; it is 2x too large (see amplitude_rank_surviving).
    NOTE the callers' historical default is max_rank=4, correct only through
    ADC(3)-ish; higher orders must pass this."""
    return max(2, 2 * (n - 1))


def amplitude_rank_surviving(n):
    """MEASURED highest amplitude rank that actually survives pdaggerq pruning
    in ADC(n) -- i.e. n-1, HALF the loose bound above.

    Every candidate structure at U orders 2/3/4 was derived and checked for
    nonzero terms (2026-07-19/20):

        U order  candidates  loose bound  SURVIVING max rank
              2           9            4                   2
              3          17            6                   3
              4          27            8                   4

    so surviving max rank == U order == n-1. The pruning is physical, not
    accidental: the amplitudes must bridge a 1h/1p orbital to a (mu+1)h mu-p
    class through a TWO-body v, so rank is limited by operator content rather
    than by order counting.

    CONSEQUENCE: ADC(9) needs t8, not t16. And since every U structure is
    (l_j, ham, t_k) -- at most ONE l and ONE t -- the Laplace FUSION DEPTH per
    term is <= 2, so cost ~ntau^2, NOT ntau^rank. Depth beyond 2 can only come
    from recursion into an unmaterializable amplitude's own numerator.

    CAVEAT: confirmed by derivation at U orders 2/3/4; order 5 timed out
    (>10 min, vs 33 s at order 4), so n-1 is EXTRAPOLATED above ADC(5)."""
    return max(1, n - 1)
