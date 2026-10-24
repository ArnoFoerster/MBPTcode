"""Codegen script (run once, output committed): derives the Delta_gamma^(n)
Moller-Plesset 1-particle-density-matrix correction, spin-orbital, for
arbitrary perturbation order n via a genuinely order-generic pdaggerq
recursion -- rather than the order-specific hand derivations in
src/SingleReference/DensityMatrix/density_matrix.py (MP2DensityMatrixSolver,
MP3DensityMatrixSolver).

Physics (see latex/ writeup, "Explicit expressions for static self-energies",
Eq. 2.2/2.15/2.19 for the n=2,3 special cases this generalizes):

Wavefunction amplitudes (Rayleigh-Schrodinger PT, intermediate normalization,
canonical HF orbitals so f_ov=0 and E^(1)=0):
    T_rank^(1)     = <bra_rank|v|Phi0> / D_rank              (nonzero only rank=2)
    T_rank^(order) = ( <bra_rank|v|Psi^(order-1)>
                        - sum_{k=2}^{order-1} E^(k) T_rank^(order-k) ) / D_rank
    E^(order)      = 1/4 sum_ijab <ij||ab> t2^(order-1)_ijab

<bra_rank|v|Psi^(order-1)> only needs the rank' in {rank-2..rank+2} components
of Psi^(order-1) (v is a 2-body operator, Slater-Condon rank-change <=2) --
see t_numerator_terms/ranks_present below.

Density (derived by expanding <Psi|a+_s a_r|Psi>/<Psi|Psi> order-by-order
under intermediate normalization -- this IS the general-n form of the doc's
Eq. 2.15 (n=2) and Eq. 2.19 (n=3), confirmed by hand to reduce to exactly
those two equations):
    M^(n) = sum_{m=0}^n <Psi^(m)| a+_s a_r |Psi^(n-m)>      (Psi^(0) = Phi0)
    N^(n) = sum_{m=1}^{n-1} <Psi^(m)|Psi^(n-m)>              (N^(1) = 0)
    c^(0) = 1,  c^(n) = -sum_{k=1}^{n} N^(k) c^(n-k)
    Delta_gamma^(n) = sum_{k=0}^{n} c^(k) M^(n-k)             (M^(0) = gamma_HF)

M^(n)'s raw pdaggerq pieces are NOT pre-filtered to "connected" diagrams --
e.g. the m=n-m=1 piece of M^(2) contains a disconnected
'0.25 * delta(m,n) * sum t2^2' term (a literal Kronecker-delta factor in the
raw pq.strings() output). This is expected and self-consistent: it exactly
cancels c^(2)*M^(0) = -N^(2)*gamma_HF once the full recursion above is
summed, reproducing the doc's Eq. 2.15 derivation (which shows this
cancellation by hand for n=2). No manual "drop the disconnected piece"
filtering is done anywhere in this module -- the recursion handles it.

Scope/limitation: pdaggerq operator strings only accept index letters a-f
(virtual) / i-n (occupied) safely (src/CodeGeneration/pdaggerq_interface.py's
check_safe_indices gotcha), capping any single bra/ket rank at 6.
ranks_present() is capped at MAX_RANK=4: MP4 (n=4) needs T4^(2) (quadruples)
for N^(4)'s <Psi2|Psi2> term and M^(4)'s <Psi2|op|Psi2> term, but does NOT
need any rank-5 amplitude (verified by the same adjacency argument: n=4's
M^(4)/N^(4) pieces only ever pair ranks differing by <=1, or exactly equal
for the bare overlap, and Psi^(2)'s own max rank is 4) -- MAX_RANK=4 is
sufficient for n=4 without paying for rank-5 derivations n=5 would need.

Demand-driven scheduling caveat: generate_amplitude_numerators's blanket
"for rank in ranks_present(order)" loop, called for order=4, would derive
T2^(4)/T3^(4)/T4^(4) in addition to the T1^(4) MP4's density actually needs
(ranks_present(order-1) is a cheap adjacency lookup, but ranks_present(order)
itself walks every rank up to MAX_RANK at that order, potentially deriving
large equations nothing downstream reads). MP4's own driver setup therefore
calls t_numerator_terms(1, 4) and t_numerator_terms(4, 2) directly rather
than routing order=4 through the generic per-order bulk generator -- true
"only derive what's demand-scheduled" would need this function to accept an
explicit rank whitelist per order; not implemented since a single explicit
rank suffices at n=4 (T1^(4) only).

Energy-correction term: T_rank^(order)'s "- sum_k E^(k) T_rank^(order-k)"
piece (see the recursion above) is bare amplitude reuse, no Wick contraction
needed. T_rank^(1)=0 for rank!=2, so for order<=3 the sum only multiplies
E^(2) by T_rank^(1); the term is nonzero in T2^(3) (needs -E^(2)*T2^(1),
since order=3's sum is E^(2)*T2^(order-2)=E^(2)*T2^(1)) and T1^(4) (needs
-E^(2)*T1^(2)). The n<=3
density never consumes T2^(3) itself, but at n=4 M^(4)'s <Psi1|op|Psi3> term
consumes T2^(3) as a bra. mpn_density_driver.py's compute_E/MP4 methods
apply the term.
"""
import os
import re

from src.CodeGeneration.pdaggerq_interface import raw_terms_to_einsum, raw_terms_to_einsum_freeing
from src.SingleReference.ADC.adc_order_rule import refuse_above_max_order
from src.CodeGeneration.codegen_common import (
    DM_OUT_DIR, OCC_LETTERS, VIR_LETTERS, density_insert, density_out_idx, density_shape, pq_terms,
    generated_header, write_module)

OUT_DIR = os.path.join(DM_OUT_DIR, 'generated_mpn')

MAX_RANK = 4  # see module docstring; needed for T4^(2) (MP4's N^(4)/M^(4))


def bra_string(rank):
    """DEPRECATED for rank > 4 -- use bra_ops(). e{rank}(occ ascending..., vir
    DESCENDING...) with pairing (occ[0],vir[0]),(occ[1],vir[1]),... -- same
    convention as T_BRA in generate_restricted_ccsdt.py, generalized to any rank.

    pdaggerq REJECTS e5 and above ("only e1, e2, e3, and e4 operators are
    supported"), and the error aborts the process at C++ level, so buffered
    stdout is lost -- run probes with `python -u`. Kept only so the e-form
    remains available as the reference bra_ops() is validated against.
    """
    occ_idx = list(OCC_LETTERS[:rank])
    vir_idx = list(reversed(VIR_LETTERS[:rank]))
    return f'e{rank}(' + ','.join(occ_idx + vir_idx) + ')'


def bra_ops(rank):
    """bra_string(rank) as an EXPLICIT a*/a operator product, which pdaggerq
    accepts at ANY rank -- this is what lifts the derivation past the e1-e4 cap
    and makes ADC(5)+ (t5..t8) reachable at all.

    Ordering: every a*(occ) in ascending order, THEN every a(vir) in the
    descending order bra_string uses. That specific ordering was determined
    EMPIRICALLY, not assumed: it reproduces the e-form's output EXACTLY -- same
    terms, same order, same signs -- at ranks 2, 3 and 4 (tests/
    test_bra_ops_equivalence.py). An interleaved a*(occ_k) a(vir_k) form happens
    to match at rank 4 by permutation parity but is WRONG at ranks 2 and 3, so
    the equivalence must be checked per rank rather than reasoned about.
    """
    occ_idx = list(OCC_LETTERS[:rank])
    vir_idx = list(reversed(VIR_LETTERS[:rank]))
    return [f'a*({o})' for o in occ_idx] + [f'a({v})' for v in vir_idx]


def amp_out_indices(rank):
    """Storage convention t{rank}[vir(rank)..., occ(rank)...], vir/occ each
    ascending -- matches density_matrix.py's t2[a,b,i,j] convention."""
    return list(VIR_LETTERS[:rank]) + list(OCC_LETTERS[:rank])


def _factor_indices(fac):
    fac = fac.strip()
    if fac.startswith('<'):
        inner = fac.split('>')[0][1:]
        return set(x for x in re.split(r'[,|]+', inner) if x)
    m = re.search(r'\((.*?)\)$', fac)
    return set(x.strip() for x in m.group(1).split(',')) if m else set()


def filter_connected(raw_terms, free_indices):
    """Drop raw pdaggerq terms that factorize into a vacuum-bubble sub-piece
    touching NONE of the target amplitude's own free (external) indices --
    e.g. t_numerator_terms(2, 2) raw output includes
    '-0.5 * <l,k||l,k> * t2(a,b,i,j)': <l,k||l,k> is a self-contracted
    bubble (both its own bra and ket use the SAME two dummy letters l,k,
    touching none of free_indices={a,b,i,j}) multiplying t2(a,b,i,j)
    UNCHANGED. This is the standard linked-cluster-theorem artifact of a raw
    (non-normal-ordered-relative-to-Psi) Wick expansion --
    density_matrix.py's compute_t2_second_order/compute_t1_third_order
    docstrings flag and manually drop the identical term ("exactly cancels
    against unlinked contributions elsewhere"). Confirmed here the same way:
    numerically, T2^(2) built WITHOUT this filter disagreed with
    density_matrix.py's independently-derived t2_2 by O(1) (not roundoff);
    filtering it out (this function) reproduces t2_2/t1_3 to 1e-12.

    Deliberately NOT "drop any term with >1 connected component" (an earlier,
    WRONG version of this function did exactly that): t_numerator_terms(4, 2)
    (T4^(2), first needed for MP4) has a raw term where pdaggerq's own V
    factor and the t2 factor share NO index at all, yet BOTH factors carry
    only free_indices letters (v's 4 legs -> 4 of T4's own 8 external
    indices, t2's 4 legs -> the other 4) -- this is the standard
    "disconnected quadruple ~ T2 x T2" CI/CC contribution, a real and
    necessary part of T4, not a vacuum artifact (a vacuum bubble is
    specifically a component with NO free indices at all -- a genuine
    scalar sub-diagram -- not merely "a component that happens not to touch
    some OTHER component"). Confirmed the old per-component version
    incorrectly zeroed T4^(2) entirely (0 raw terms survived, physically
    wrong -- T4^(2) is a large, well-known nonzero quantity); this
    free-index-aware version keeps it.

    Only applied to amplitude-numerator generation (t_numerator_terms) --
    NOT to overlap_terms/cross_density_terms, whose disconnected pieces are
    genuine, required parts of the M^(n)/N^(n) recursion (see this module's
    docstring): those cancel algebraically against c^(k)*M^(n-k) terms
    downstream, they don't just vanish on their own.
    """
    free = set(free_indices)
    kept = []
    for term in raw_terms:
        factors = [_factor_indices(fac) for fac in term[1:] if not fac.strip().startswith('P(')]
        if len(factors) <= 1:
            kept.append(term)
            continue
        parent = list(range(len(factors)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                if factors[i] & factors[j]:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        components = {}
        for i, f in enumerate(factors):
            components.setdefault(find(i), set()).update(f)
        if all(comp & free for comp in components.values()):
            kept.append(term)
    return kept


_numerator_cache = {}


def t_numerator_terms(rank, order, extra_connect=()):
    """Raw pdaggerq terms for the V-connected piece of T_rank^(order)'s
    numerator, <bra_rank|v|Psi^(order-1)>. The '- sum_k E^(k) T_rank^(order-k)'
    piece is bare amplitude reuse (no Wick contraction needed) and is added
    numerically by the driver, not derived here.

    extra_connect: ranks of Psi^(order-1) above MAX_RANK to connect as well,
    each an explicit, individually verified declaration (the discipline
    density_pieces' extra_ranks_by_order follows). ranks_present caps at
    MAX_RANK, so without this a quintuple-capable system loses the V.T5^(3)
    rung of T3^(4)/T4^(4) silently -- the truncation the driver warns about.
    """
    key = (rank, order, tuple(sorted(extra_connect)))
    if key in _numerator_cache:
        return _numerator_cache[key]
    if order < 1:
        raise ValueError("order must be >= 1")

    if order == 1:
        products = [(1.0, ['v'])] if rank == 2 else []
    else:
        products = [(1.0, ['v', f't{r}'])
                    for r in sorted(set(ranks_present(order - 1)) | set(extra_connect))
                    if abs(r - rank) <= 2]
    terms = filter_connected(pq_terms([bra_ops(rank)], products), amp_out_indices(rank)) if products else []
    _numerator_cache[key] = terms
    return terms


_ranks_present_cache = {}


def ranks_present(order):
    """Ranks with a nonzero T_rank^(order), capped at MAX_RANK (see module
    docstring). order==1 is a hard-coded special case (Brillouin theorem:
    only rank 2 is nonzero at first order) since t_numerator_terms's
    order==1 branch would otherwise need this function to bootstrap itself.
    """
    if order in _ranks_present_cache:
        return _ranks_present_cache[order]
    if order == 1:
        ranks = [2]
    else:
        ranks = [r for r in range(1, MAX_RANK + 1) if t_numerator_terms(r, order)]
    _ranks_present_cache[order] = ranks
    return ranks


_overlap_cache = {}


def overlap_terms(rank):
    """Raw pdaggerq terms for <l{rank}|t{rank}> (one N^(n) piece; order-agnostic
    -- which order's amplitude array is fed into l{rank}/t{rank} is a
    numeric-evaluation choice made by the driver, not a symbolic one here)."""
    if rank not in _overlap_cache:
        _overlap_cache[rank] = pq_terms([[f'l{rank}']], [(1.0, [f't{rank}'])])
    return _overlap_cache[rank]


_cross_cache = {}


def cross_density_terms(insert_op, bra_ranks, ket_ranks):
    """Raw pdaggerq terms for sum_{r in bra_ranks, r' in ket_ranks}
    <l{r}|insert_op|t{r'}>, one M^(n) piece for a fixed (bra order, ket order)
    split (bra_ranks/ket_ranks = the ranks present at those two orders).
    Empty bra_ranks means bra=Phi0 (pure <Phi0|insert_op|Psi^(k)>); empty
    ket_ranks means ket=Phi0. At least one of the two must be non-empty.
    """
    key = (insert_op, tuple(bra_ranks), tuple(ket_ranks))
    if key in _cross_cache:
        return _cross_cache[key]
    if not bra_ranks and not ket_ranks:
        raise ValueError("bra_ranks and ket_ranks cannot both be empty")

    products = [(1.0, [insert_op, f't{r}']) for r in ket_ranks] if ket_ranks else [(1.0, [insert_op])]
    terms = pq_terms([[f'l{r}'] for r in bra_ranks] if bra_ranks else [['1']], products)
    _cross_cache[key] = terms
    return terms


# --- code assembly -----------------------------------------------------

def assemble(func_name, params, aliases, out_var, out_shape_expr, raw_terms, out_indices,
             free_temps=False):
    """Like pdaggerq_interface.assemble_function, but nv/no are passed in
    explicitly (params must include 'nv, no') instead of inferred from a
    fixed-name (nv, no)-shaped arg -- this module's functions don't all share
    a single canonical 2D leaf to infer shape from. `aliases` maps a generic
    pdaggerq tag (e.g. 't2', 'l3') used inside raw_terms to the real
    parameter name holding that specific order's amplitude array.

    free_temps=True emits the low-memory (raw_terms_to_einsum_freeing) body --
    byte-identical einsum lines plus `del _tmpN` after each temp's last use, so
    large (T3) numerators don't hold every contraction temporary resident at
    once (default False keeps the exact original body)."""
    emit = raw_terms_to_einsum_freeing if free_temps else raw_terms_to_einsum
    src = [f"def {func_name}({', '.join(params)}):"]
    for generic, real in aliases.items():
        if generic != real:
            src.append(f"    {generic} = {real}")
    src.append(f"    {out_var} = np.zeros({out_shape_expr})")
    src.extend(emit(raw_terms, out_var, out_indices))
    src.append(f"    return {out_var}")
    src.append("")
    return "\n".join(src)


def _shape(rank):
    return '(' + ', '.join(['nv'] * rank + ['no'] * rank) + ')'


COMMON_ARGS = ['g', 'kd', 'o', 'v', 'nv', 'no']



def _amplitude_numerator_func(rank, order, extra_connect=()):
    """One (rank, order) amplitude numerator -- factored out of
    generate_amplitude_numerators so MP4's driver setup can request a single
    explicit (rank, order) pair (e.g. just T1^(4)) without also deriving
    every other rank ranks_present(order) would otherwise walk through (see
    this module's docstring, "Demand-driven scheduling caveat")."""
    terms = t_numerator_terms(rank, order, extra_connect=extra_connect)
    connect_ranks = ([r for r in sorted(set(ranks_present(order - 1)) | set(extra_connect))
                      if abs(r - rank) <= 2] if order > 1 else [])
    aliases = {f't{r}': f't{r}_{order - 1}' for r in connect_ranks}
    params = COMMON_ARGS + sorted(set(aliases.values()))
    name = f't{rank}_{order}_numerator'
    print(f"{name}: {len(terms)} terms")
    return assemble(name, params, aliases, 'result', _shape(rank), terms, amp_out_indices(rank))


def generate_amplitude_numerators(orders):
    """One function per (rank, order) with rank in ranks_present(order),
    computing T_rank^(order)'s V-connected numerator piece."""
    for o in orders:
        refuse_above_max_order(o + 1, 'generate_amplitude_numerators')
    funcs = []
    for order in orders:
        for rank in ranks_present(order):
            funcs.append(_amplitude_numerator_func(rank, order))
    return funcs


def generate_amplitude_numerator_module(orders, basename, extra_connect_by_order=None):
    """Emit ALL amplitude numerators for `orders` into their own incremental
    module generated_mpn/<basename>.py -- the generateADC(n) cache pattern:
    each new MP order gets its own module file, existing files are never
    rewritten.

    The module carries a RANKS_PRESENT manifest ({order: [ranks]}, the
    pdaggerq survival census at generation time) so the RUNTIME can
    distinguish 'pruned by pdaggerq' from 'never generated' without importing
    pdaggerq (a missing function is ambiguous by itself).
    Complete-per-order: every rank in ranks_present(order) is emitted even if
    another module already carries one of them (derivation is deterministic,
    so the duplicate is byte-identical content, and per-order completeness is
    what makes the manifest trustworthy)."""
    for o in orders:
        # MP order o amplitudes are first consumed by ADC(o + 1)
        refuse_above_max_order(o + 1, 'generate_amplitude_numerator_module')
    extra_by_order = extra_connect_by_order or {}
    funcs = []
    manifest = {}
    for order in orders:
        manifest[order] = ranks_present(order)
        extra = extra_by_order.get(order, ())
        for rank in manifest[order]:
            funcs.append(_amplitude_numerator_func(rank, order, extra_connect=extra))
    return write_module(os.path.join(OUT_DIR, f'{basename}.py'),
                        [f"RANKS_PRESENT = {manifest!r}"] + funcs, generated_header(
                            f'generate_mpn_density.py (generate_amplitude_numerator_module({orders!r}))',
                            'MPn amplitude V-connected numerators.'))


def generate_overlaps(ranks):
    funcs = []
    for rank in ranks:
        terms = overlap_terms(rank)
        aliases = {f'l{rank}': 'l_amp', f't{rank}': 't_amp'}
        params = ['l_amp', 't_amp']
        name = f'overlap{rank}'
        print(f"{name}: {len(terms)} terms")
        funcs.append(assemble(name, params, aliases, 'result', '()', terms, []))
    return funcs


def generate_density_pieces(pieces):
    """pieces: list of (name, insert_op_key, bra_ranks, bra_arg_order, ket_ranks, ket_arg_order).
    bra_arg_order/ket_arg_order map rank -> real amplitude-array param name.
    """
    funcs = []
    for name, block, bra_ranks, bra_args, ket_ranks, ket_args in pieces:
        terms = cross_density_terms(density_insert(block), bra_ranks, ket_ranks)
        aliases = {}
        params = list(COMMON_ARGS)
        for r in bra_ranks:
            aliases[f'l{r}'] = bra_args[r]
            params.append(bra_args[r])
        for r in ket_ranks:
            aliases[f't{r}'] = ket_args[r]
            params.append(ket_args[r])
        print(f"{name}: {len(terms)} terms")
        funcs.append(assemble(name, params, aliases, 'result', density_shape(block), terms,
                              density_out_idx(block)))
    return funcs


_ADJ_BOUND = 1  # e1(...) density insertion is 1-body: bridges ph-rank by <=1


def density_pieces_for_order(order, extra_ranks_by_order=None):
    """General M^(n) cross-density-term piece list for ANY order (no
    per-order hand-typed piece lists m2_*/m3_*/m4_*).

    Enumerates every bra/ket order split a+b=order (a,b>=0; a==b==0 skipped,
    that's gamma_HF, added by the driver, not a cross term), using
    ranks_present(a)/ranks_present(b) on each side, filtered to the pairs a
    1-body e1(...) insertion can actually connect: |bra_rank - ket_rank| <=
    _ADJ_BOUND. a==0 or b==0 means that side is literally Phi0 (rank 0), so
    the filter automatically reduces to "only the rank-1 component of the
    other side survives" (for m2_20/m4_40 this is bra_ranks=[1]), one rule
    instead of a per-order judgment call. It reproduces every order-2/3/4
    piece's raw pdaggerq terms exactly, with m3_12/m3_21 (drop the rank-4
    side) and m3_30/m3_03 (drop ranks 2,3) losing only parameters proven
    unused (see mpn_density_driver.py's compute_delta_gamma3 comments).

    extra_ranks_by_order: optional {order: [ranks]} of rank > MAX_RANK
    content independently known nonzero via its own pdaggerq census --
    merged into ranks_present(order). Does NOT widen ranks_present's own
    MAX_RANK search: that pushes unconnectable names into lower orders and
    poisons the pruned-vs-ungenerated manifest. Extra rank content must
    always be an explicit, individually pdaggerq-verified declaration.

    A split whose bra or ket side has no adjacency-connected rank left after
    filtering is skipped entirely (provably zero -- e.g. order=1 yields no
    pieces at all: M^(1) is identically 0, Psi^(1) is pure rank 2 and a
    1-body operator cannot bridge rank 2 <-> Phi0's rank 0)."""
    # the MP order-n density is the static self-energy input of ADC(n + 1)
    refuse_above_max_order(order + 1, 'density_pieces_for_order')
    extra = extra_ranks_by_order or {}

    def ranks_for(k):
        if k == 0:
            return []
        return sorted(set(ranks_present(k)) | set(extra.get(k, [])))

    pieces = []
    for block in ('oo', 'vv', 'ov'):
        for a in range(order + 1):
            b = order - a
            if a == 0 and b == 0:
                continue
            raw_bra = ranks_for(a) if a > 0 else [0]
            raw_ket = ranks_for(b) if b > 0 else [0]
            bra_ranks = [r for r in raw_bra if r > 0
                         and any(abs(r - r2) <= _ADJ_BOUND for r2 in raw_ket)]
            ket_ranks = [r for r in raw_ket if r > 0
                         and any(abs(r - r2) <= _ADJ_BOUND for r2 in raw_bra)]
            if a > 0 and not bra_ranks:
                continue
            if b > 0 and not ket_ranks:
                continue
            name = f'm{order}_{block}_{a}{b}'
            bra_args = {r: f'l{r}' for r in bra_ranks}
            ket_args = {r: f't{r}' for r in ket_ranks}
            pieces.append((name, block, bra_ranks, bra_args, ket_ranks, ket_args))
    return pieces


def generate():
    """Write OUT_DIR/mpn_density_pieces.py."""

    # --- amplitude numerators: T2^(1); T1^(2),T2^(2),T3^(2),T4^(2) (T4^(2)
    # for MP4's N^(4)/M^(4)); T1^(3),T2^(3),T3^(3) (T2^(3) gets its
    # -E^(2)*T2^(1) correction applied numerically in the driver, see module
    # docstring); T1^(4) (MP4's M^(4) boundary term).
    # Calling generate_amplitude_numerators([1,2,3]) under MAX_RANK=4 would
    # also derive ranks_present(3)'s rank-4 member (T4^(3)) as an unwanted
    # side effect of that blanket loop (see module docstring's scheduling
    # caveat) -- accepted here rather than hand-restricting ranks_present's
    # own walk, since it is merely wasted derivation time, not incorrect.
    amp_funcs = generate_amplitude_numerators([1, 2, 3])
    amp_funcs.append(_amplitude_numerator_func(1, 4))

    # --- overlap pieces: rank 2 (N^(2), and N^(3) reusing the same rank with
    # different order args); ranks 1,3,4 newly needed for N^(4)'s full
    # <Psi2|Psi2> self-overlap (ranks 1-4 of Psi^(2), rank-matched only).
    overlap_funcs = generate_overlaps([1, 2, 3, 4])

    # --- M^(2)/M^(3)/M^(4) cross-density pieces: every bra/ket order split
    # a+b=n via density_pieces_for_order (see that function's docstring for
    # the adjacency rule replacing the old per-order hand-typed piece lists).
    density_pieces = (density_pieces_for_order(2) + density_pieces_for_order(3)
                       + density_pieces_for_order(4))
    density_funcs = generate_density_pieces(density_pieces)

    write_module(os.path.join(OUT_DIR, 'mpn_density_pieces.py'), amp_funcs + overlap_funcs + density_funcs,
                 generated_header('generate_mpn_density.py',
                                  'MPn amplitude-numerator, overlap and density cross-term pieces',
                                  '(spin-orbital) of the order-generic pdaggerq recursion.'),
                 package_init=True)


if __name__ == '__main__':
    generate()
