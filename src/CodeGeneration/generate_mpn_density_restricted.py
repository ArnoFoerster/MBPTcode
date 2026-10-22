"""Codegen script (run once, output committed): restricted (spin-blocked,
closed-shell RHF) analogue of generate_mpn_density.py -- same order-generic
pdaggerq recursion (see that module's docstring for the physics: T_rank^(order)
amplitude numerators, N^(n)/M^(n)/c^(n) density recursion), but deriving every
equation with pdaggerq's spin-integration feature (pq.strings(spin_labels=...))
instead of plain spin-orbital indices, so the generated code operates on
spatial-MO spin-block arrays (t2_aaaa/t2_abab/..., matching
src/CodeGeneration/generate_restricted_ccsdt.py's storage convention) rather
than the full (2*nmo)-dimensional spin-orbital tensors.

Reuses this repo's existing restricted infrastructure rather than
reinventing it:
- Bra-projector strings and their spin-block dicts: T_BRA, DENSITY_BLOCKS
  from generate_restricted_ccsdt.py (verified byte-identical to this
  package's own bra_string(rank) for ranks 1-3 -- both build
  'e{rank}(occ ascending..., vir DESCENDING...)').
- Rank-presence (which T_rank^(order) is nonzero) and the disconnected-piece
  filter for amplitude numerators: ranks_present/_filter_connected from
  generate_mpn_density.py -- both are spin-orbital-derived facts about the
  underlying physics, unaffected by how the result is later spin-decomposed.
- Codegen backend (raw_terms_to_einsum via `assemble`): also imported
  directly from generate_mpn_density.py -- spin suffixes just become part of
  each factor's literal name (pdaggerq_interface.py's _parse_factor/_array_expr
  already handle e.g. 't2_abab(...)'/'d_aa(...)' generically), no separate
  codegen logic needed.
- Closed-shell spin-block storage convention (only '_aa'/'_aaaa'/'_abab'/
  '_aaaaaa'/'_aabaab'/'_abbabb' computed and stored; '_bb'-type leaves a raw
  equation may reference are numerically identical to their '_aa'-type twin
  and substituted at call time -- see restricted_solver.py and
  generate_restricted_ccsdt.py's module docstring for the same convention).
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Base.pdaggerq_interface import pq_helper, raw_terms_to_einsum
from src.SingleReference.ADC.adc_order_rule import refuse_above_max_order
from src.CodeGeneration.generate_restricted_ccsdt import T_BRA as _T_BRA_CCSDT, DENSITY_BLOCKS
from src.CodeGeneration.generate_mpn_density import (
    ranks_present, _filter_connected, amp_out_indices, assemble, _shape,
)
from src.CodeGeneration.df_codegen import dress_integral_factor

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'SingleReference', 'DensityMatrix', 'generated_mpn_restricted')


def _build_t_bra(rank):
    """Programmatic T_BRA[rank] entry: bra string + spin-block dict, for any
    rank -- generate_restricted_ccsdt.py's own T_BRA only covers ranks 1-3
    (CCSDT stops at triples), but MP4 needs rank 4. Confirmed
    this construction reproduces T_BRA[1], T_BRA[2], T_BRA[3] byte-for-byte,
    so it's trusted for rank 4 too (same convention, just extended):
    bra = 'e{rank}(occ ascending, vir DESCENDING)', pairing (occ[p],vir[p])
    each sharing one spin. Canonical (stored) blocks are exactly the `rank`
    ways to place k=0..rank-1 beta-spin pairs at the END of the pair
    ordering (k=rank, all-beta, is the redundant global-spin-flip twin of
    k=0, all-alpha -- same convention as ALL_BLOCKS/BB_TWIN below, where
    e.g. rank 2 stores {aaaa, abab} and treats bbbb as aaaa's twin, never
    storing/deriving a separate 'baba').
    """
    occ_letters = list('ijklmn'[:rank])
    vir_letters = list('abcdef'[:rank])
    bra = f'e{rank}(' + ','.join(occ_letters + list(reversed(vir_letters))) + ')'
    blocks = {}
    for k in range(rank):
        spins = ['a'] * (rank - k) + ['b'] * k
        block_name = ''.join(spins) * 2
        spin_labels = {}
        for p in range(rank):
            spin_labels[occ_letters[p]] = spins[p]
            spin_labels[vir_letters[p]] = spins[p]
        blocks[block_name] = spin_labels
    return bra, blocks


T_BRA = dict(_T_BRA_CCSDT)
T_BRA[4] = _build_t_bra(4)


def _all_blocks_and_twin_for_rank(rank):
    """Canonical (stored) block names for a rank, plus the one "all-beta"
    twin name (substituted at call time by its all-alpha canonical
    counterpart -- see _build_t_bra's docstring)."""
    canonical = [''.join(['a'] * (rank - k) + ['b'] * k) * 2 for k in range(rank)]
    all_beta = ('b' * rank) * 2
    return canonical + [all_beta], {all_beta: canonical[0]}


ALL_BLOCKS = {}
BB_TWIN = {}
for _rank in (1, 2, 3, 4):
    _blocks, _twin = _all_blocks_and_twin_for_rank(_rank)
    ALL_BLOCKS[_rank] = _blocks
    BB_TWIN.update(_twin)

COMMON_ARGS_R = ['g_aaaa', 'g_abab', 'g_bbbb', 'd_aa', 'o', 'v', 'nv', 'no']


def _connect_ranks(rank, order):
    return [r for r in ranks_present(order - 1) if abs(r - rank) <= 2] if order > 1 else []


def _t_aliases_and_params(connect_ranks, order):
    """Every possible spin-block leaf name pq_helper's generic t{r} tag can
    print (per ALL_BLOCKS), aliased to this module's own storage name
    t{r}_{order-1}_{block} -- with '_bb'-type blocks aliased to their '_aa'-
    type twin (never separately computed/stored, per this repo's closed-shell
    convention)."""
    aliases, params = {}, list(COMMON_ARGS_R)
    for r in connect_ranks:
        for sub in ALL_BLOCKS[r]:
            real_sub = BB_TWIN.get(sub, sub)
            generic_name = f't{r}_{sub}'
            real_name = f't{r}_{order - 1}_{real_sub}'
            aliases[generic_name] = real_name
            if real_name not in params:
                params.append(real_name)
    return aliases, params


_numerator_cache_r = {}


def t_numerator_terms_restricted(rank, order, block):
    """Restricted analogue of generate_mpn_density.t_numerator_terms, one
    spin block at a time (block is a key of T_BRA[rank][1], e.g. 'aaaa')."""
    key = (rank, order, block)
    if key in _numerator_cache_r:
        return _numerator_cache_r[key]
    bra, blocks = T_BRA[rank]
    spin_labels = blocks[block]

    if order == 1:
        terms = []
        if rank == 2:
            pq = pq_helper('fermi')
            pq.set_left_operators([[bra]])
            pq.set_right_operators([['1']])
            pq.add_operator_product(1.0, ['v'])
            pq.simplify()
            terms = _filter_connected(pq.strings(spin_labels=spin_labels), amp_out_indices(rank))
            pq.clear()
        _numerator_cache_r[key] = terms
        return terms

    connect_ranks = _connect_ranks(rank, order)
    terms = []
    if connect_ranks:
        pq = pq_helper('fermi')
        pq.set_left_operators([[bra]])
        pq.set_right_operators([['1']])
        for r in connect_ranks:
            pq.add_operator_product(1.0, ['v', f't{r}'])
        pq.simplify()
        terms = _filter_connected(pq.strings(spin_labels=spin_labels), amp_out_indices(rank))
        pq.clear()
    _numerator_cache_r[key] = terms
    return terms


def _amplitude_numerator_funcs_restricted(rank, order):
    """One (rank, order) amplitude numerator, every spin block -- factored
    out of generate_amplitude_numerators_restricted so MP4's setup can
    request a single explicit (rank, order) pair (e.g. just T1^(4)) without
    also deriving every other rank ranks_present(order) would walk through
    (mirrors generate_mpn_density.py's _amplitude_numerator_func)."""
    bra, blocks = T_BRA[rank]
    connect_ranks = _connect_ranks(rank, order)
    aliases, params = _t_aliases_and_params(connect_ranks, order)
    funcs = []
    for block in blocks:
        terms = t_numerator_terms_restricted(rank, order, block)
        name = f't{rank}_{order}_{block}_numerator'
        print(f"{name}: {len(terms)} terms")
        funcs.append(assemble(name, params, aliases, 'result', _shape(rank), terms, amp_out_indices(rank), free_temps=True))
    return funcs


def _amplitude_numerator_no_t3_funcs_restricted(rank, order, max_connect_rank):
    """Same physics/derivation as _amplitude_numerator_funcs_restricted, but
    excludes any connect rank above max_connect_rank from both the pdaggerq
    derivation and the emitted params -- produces a '..._no_t3'-suffixed
    numerator variant whose higher-rank (T3^(2)) contribution is supplied
    externally instead of materialized as a rank>4 array (see
    generate_mp3_t3_laplace_restricted.py, which fills that contribution in
    via a Laplace-fused replacement). Used for t1_3_aa_numerator's production
    (laplace-mode) variant: connect_ranks capped at 2 means this body never
    references t3_2_aaaaaa/aabaab/abbabb at all, not even as a dead param."""
    bra, blocks = T_BRA[rank]
    connect_ranks = [r for r in _connect_ranks(rank, order) if r <= max_connect_rank]
    aliases, params = _t_aliases_and_params(connect_ranks, order)
    funcs = []
    for block in blocks:
        spin_labels = blocks[block]
        terms = []
        if connect_ranks:
            pq = pq_helper('fermi')
            pq.set_left_operators([[bra]])
            pq.set_right_operators([['1']])
            for r in connect_ranks:
                pq.add_operator_product(1.0, ['v', f't{r}'])
            pq.simplify()
            terms = _filter_connected(pq.strings(spin_labels=spin_labels), amp_out_indices(rank))
            pq.clear()
        name = f't{rank}_{order}_{block}_numerator_no_t3'
        print(f"{name}: {len(terms)} terms")
        funcs.append(assemble(name, params, aliases, 'result', _shape(rank), terms, amp_out_indices(rank), free_temps=True))
    return funcs


COMMON_ARGS_R_DF = ['B_aa', 'B_bb', 'd_aa', 'o', 'v', 'nv', 'no']

# Staged three-operand contractions in the DF bodies. A DF term B x B x amplitude
# has three pairwise contractions; one can form an intermediate larger than every
# operand and the output (B_vv x B_ov -> (o, v, v, v) in t1_2). An intermediate is
# bounded when its indices map one-to-one onto those of some operand or of the
# output with occ <= vir <= aux, so it is no larger for any no <= nv <= naux.
# A term with an unbounded and a bounded pairing is emitted as two two-operand
# einsums through the bounded one (cheapest under the size model if several).
# Terms with no bounded pairing (vvvv-type 'Qac,Qbd,cdij->abij') are left alone.
_STAGE_CLASS = {'o': 0, 'v': 1, 'Q': 2}
_STAGE_MODEL_SIZE = {'o': 10, 'v': 100, 'Q': 400}
_EINSUM3_LINE = re.compile(r"^    (_tmp\d+) = einsum\('([^']*)', (.*), optimize=True\)$")
_ACCUM_LINE = re.compile(r"^    (\w+) ([+-])= (\S+) \* (_tmp\d+)$")


def _split_operands(text):
    ops, depth, cur = [], 0, ''
    for ch in text:
        if ch == ',' and depth == 0:
            ops.append(cur.strip())
            cur = ''
            continue
        depth += ch in '[('
        depth -= ch in '])'
        cur += ch
    ops.append(cur.strip())
    return ops


def _operand_classes(expr, sub):
    """Index classes ('o'/'v'/'Q') of one operand, read from its expression:
    B factors are (aux, sliced, sliced), amplitude leaves t{r}_/l{r}_ are
    (vir^r, occ^r). None for anything else (the term is then left alone)."""
    m = re.match(r'^B\w*\[:, ([ov]), ([ov])\]$', expr)
    if m:
        return ['Q', m.group(1), m.group(2)]
    m = re.match(r'^[tl](\d)_\w+$', expr)
    if m and len(sub) == 2 * int(m.group(1)):
        r = int(m.group(1))
        return ['v'] * r + ['o'] * r
    return None


def _bounded_by(x, y):
    xs = sorted((_STAGE_CLASS[c] for c in x), reverse=True)
    ys = sorted((_STAGE_CLASS[c] for c in y), reverse=True)
    return len(xs) <= len(ys) and all(a <= b for a, b in zip(xs, ys))


def _model_size(classes):
    n = 1
    for c in classes:
        n *= _STAGE_MODEL_SIZE[c]
    return n


def _staging_plan(sig, exprs):
    """(pair, third, intermediate subscript) for a three-operand einsum that
    the rule above stages, else None."""
    ins, out = sig.split('->')
    subs = ins.split(',')
    if len(subs) != 3 or len(exprs) != 3:
        return None
    cls = {}
    for sub, expr in zip(subs, exprs):
        oc = _operand_classes(expr, sub)
        if oc is None or len(oc) != len(sub):
            return None
        for c, k in zip(sub, oc):
            if cls.setdefault(c, k) != k:
                return None
    arrays = [[cls[c] for c in s] for s in subs] + [[cls[c] for c in out]]
    plans = []
    for pair in ((0, 1), (0, 2), (1, 2)):
        third = 3 - sum(pair)
        keep = set(subs[third]) | set(out)
        inter = ''
        for c in subs[pair[0]] + subs[pair[1]]:
            if c in keep and c not in inter:
                inter += c
        bounded = any(_bounded_by([cls[c] for c in inter], a) for a in arrays)
        cost = (_model_size(cls[c] for c in set(subs[pair[0]] + subs[pair[1]]))
                + _model_size(cls[c] for c in set(subs[third] + inter)))
        plans.append((bounded, cost, pair, third, inter))
    if all(p[0] for p in plans) or not any(p[0] for p in plans):
        return None
    _, _, pair, third, inter = min((p for p in plans if p[0]), key=lambda p: p[1])
    return pair, third, inter


def _stage_three_operand_einsums(lines):
    """Rewrite every three-operand `_tmpN = einsum(...)` line the rule above
    stages. When _tmpN is only read by an unbroken run of untransposed
    accumulations right after it, the second contraction goes straight into
    the output (one line, coefficients summed); otherwise _tmpN is rebound to
    the full contraction and the accumulation lines stay as they are."""
    out = []
    i = 0
    while i < len(lines):
        m = _EINSUM3_LINE.match(lines[i])
        plan = None
        if m:
            tmp, sig, ops = m.group(1), m.group(2), _split_operands(m.group(3))
            plan = _staging_plan(sig, ops)
        if plan is None:
            out.append(lines[i])
            i += 1
            continue
        (p0, p1), third, inter = plan
        subs, res = sig.split('->')
        subs = subs.split(',')
        out.append(f"    {tmp} = einsum('{subs[p0]},{subs[p1]}->{inter}', {ops[p0]}, {ops[p1]}, optimize=True)")
        second = f"einsum('{subs[third]},{inter}->{res}', {ops[third]}, {tmp}, optimize=True)"
        j = i + 1
        run = []
        while j < len(lines):
            a = _ACCUM_LINE.match(lines[j])
            if not (a and a.group(4) == tmp):
                break
            run.append(a)
            j += 1
        later = any(re.search(rf'\b{tmp}\b', ln) for ln in lines[j:])
        if run and not later and len({a.group(1) for a in run}) == 1:
            if len(run) == 1:
                sign, coef = run[0].group(2), run[0].group(3)
            else:
                total = sum((1 if a.group(2) == '+' else -1) * float(a.group(3)) for a in run)
                sign, coef = ('+' if total >= 0 else '-'), f"{abs(total):.15g}"
            out.append(f"    {run[0].group(1)} {sign}= {coef} * {second}")
            i = j
        else:
            out.append(f"    {tmp} = {second}")
            i += 1
    return out


def _free_temps(lines):
    """`del _tmpN` after each temporary's last use (same placement as
    pdaggerq_interface.raw_terms_to_einsum_freeing)."""
    last = {}
    for i, ln in enumerate(lines):
        for m in re.finditer(r'_tmp\d+', ln):
            last[m.group(0)] = i
    for i in sorted(set(last.values()), reverse=True):
        for t in sorted([t for t, li in last.items() if li == i], reverse=True):
            lines.insert(i + 1, f"    del {t}")
    return lines


def _assemble_df(func_name, params, aliases, out_shape_expr, raw_terms, out_indices):
    """generate_mpn_density.assemble(..., free_temps=True) for the DF bodies,
    with the three-operand staging pass applied before the temps are freed."""
    src = [f"def {func_name}({', '.join(params)}):"]
    for generic, real in aliases.items():
        if generic != real:
            src.append(f"    {generic} = {real}")
    src.append(f"    result = np.zeros({out_shape_expr})")
    src.extend(_free_temps(_stage_three_operand_einsums(raw_terms_to_einsum(raw_terms, 'result', out_indices))))
    src.append("    return result")
    src.append("")
    return "\n".join(src)



def _t_aliases_and_params_df(connect_ranks, order):
    """Like _t_aliases_and_params, but starting from COMMON_ARGS_R_DF
    (B_aa/B_bb) instead of COMMON_ARGS_R (g_aaaa/g_abab/g_bbbb) -- see
    _amplitude_numerator_funcs_restricted_df."""
    aliases, params = {}, list(COMMON_ARGS_R_DF)
    for r in connect_ranks:
        for sub in ALL_BLOCKS[r]:
            real_sub = BB_TWIN.get(sub, sub)
            generic_name = f't{r}_{sub}'
            real_name = f't{r}_{order - 1}_{real_sub}'
            aliases[generic_name] = real_name
            if real_name not in params:
                params.append(real_name)
    return aliases, params


def _amplitude_numerator_funcs_restricted_df(rank, order):
    """DF/RI variant of _amplitude_numerator_funcs_restricted (see
    the DF rewrite, Phase 1): same physics, but every raw term's
    bracket integral factor is expanded via df_codegen.dress_integral_factor
    before assembly, so the emitted functions take B_aa/B_bb (3-index RI
    factors) instead of g_aaaa/g_abab/g_bbbb and never form a norb^4-scale
    integral block.

    Only meant to be called for (rank, order) pairs that actually contain a
    bracket factor -- Phase 1's (rank=2, order=1)=t2_1, (rank=1, order=2)=
    t1_2, and Phase 3's (rank=2, order=2)=t2_2 (includes the memory-critical
    g_aaaa[v,v,v,v]/g_abab[v,v,v,v] terms), (rank=3, order=2)=t3_2 (non-
    Laplace oracle/MP4 path only), (rank=1, order=3)=t1_3's full (non-'_no_t3')
    variant -- every OTHER density piece (overlap2/3, m2_*/m3_* cross-terms)
    has no bracket factor at all, confirmed by inspecting the generated
    dense code, so those are reused UNCHANGED by the DF driver, not
    regenerated here.
    """
    bra, blocks = T_BRA[rank]
    connect_ranks = _connect_ranks(rank, order)
    aliases, params = _t_aliases_and_params_df(connect_ranks, order)
    funcs = []
    for block in blocks:
        terms = t_numerator_terms_restricted(rank, order, block)
        df_terms = dress_integral_factor(terms)
        name = f't{rank}_{order}_{block}_numerator_df'
        print(f"{name}: {len(df_terms)} DF terms (from {len(terms)} dense terms)")
        funcs.append(_assemble_df(name, params, aliases, _shape(rank), df_terms, amp_out_indices(rank)))
    return funcs


def _amplitude_numerator_no_t3_funcs_restricted_df(rank, order, max_connect_rank):
    """DF/RI variant of _amplitude_numerator_no_t3_funcs_restricted (Phase 3)
    -- production t1_3_aa_numerator_no_t3's DF twin: connect_ranks capped at
    max_connect_rank, so this body never references t3_2_aaaaaa/aabaab/abbabb
    (dense or DF) at all, matching the dense version's own "no T3^(2), not
    even as a dead param" property."""
    bra, blocks = T_BRA[rank]
    connect_ranks = [r for r in _connect_ranks(rank, order) if r <= max_connect_rank]
    aliases, params = _t_aliases_and_params_df(connect_ranks, order)
    funcs = []
    for block in blocks:
        spin_labels = blocks[block]
        terms = []
        if connect_ranks:
            pq = pq_helper('fermi')
            pq.set_left_operators([[bra]])
            pq.set_right_operators([['1']])
            for r in connect_ranks:
                pq.add_operator_product(1.0, ['v', f't{r}'])
            pq.simplify()
            terms = _filter_connected(pq.strings(spin_labels=spin_labels), amp_out_indices(rank))
            pq.clear()
        df_terms = dress_integral_factor(terms)
        name = f't{rank}_{order}_{block}_numerator_no_t3_df'
        print(f"{name}: {len(df_terms)} DF terms (from {len(terms)} dense terms)")
        funcs.append(_assemble_df(name, params, aliases, _shape(rank), df_terms, amp_out_indices(rank)))
    return funcs


def generate_amplitude_numerators_restricted(orders):
    for o in orders:
        refuse_above_max_order(o + 1, 'generate_amplitude_numerators_restricted')
    funcs = []
    for order in orders:
        for rank in ranks_present(order):
            if rank not in T_BRA:
                continue  # T_BRA only defines ranks 1-4 (this module's MAX_RANK budget)
            funcs += _amplitude_numerator_funcs_restricted(rank, order)
    return funcs


_overlap_cache_r = {}


def overlap_terms_restricted(rank):
    """Restricted analogue of generate_mpn_density.overlap_terms -- a
    scalar, spin-decomposes on its own (spin_labels={}) into aaaa/abab.../
    bbbb pieces even though there are no free indices to assign spins to."""
    if rank in _overlap_cache_r:
        return _overlap_cache_r[rank]
    pq = pq_helper('fermi')
    pq.set_left_operators([[f'l{rank}']])
    pq.set_right_operators([['1']])
    pq.add_operator_product(1.0, [f't{rank}'])
    pq.simplify()
    terms = pq.strings(spin_labels={})
    pq.clear()
    _overlap_cache_r[rank] = terms
    return terms


def generate_overlaps_restricted(ranks):
    """Params are generic (l_aaaa/t_aaaa/...), not order-tied -- the driver
    supplies whichever specific order's amplitude array plays the l/t role
    at each call site (mirrors generate_mpn_density.generate_overlaps)."""
    funcs = []
    for rank in ranks:
        terms = overlap_terms_restricted(rank)
        canonical_blocks = [s for s in ALL_BLOCKS[rank] if BB_TWIN.get(s, s) == s]
        aliases = {}
        for sub in ALL_BLOCKS[rank]:
            real_sub = BB_TWIN.get(sub, sub)
            aliases[f'l{rank}_{sub}'] = f'l_{real_sub}'
            aliases[f't{rank}_{sub}'] = f't_{real_sub}'
        params = [f'l_{s}' for s in canonical_blocks] + [f't_{s}' for s in canonical_blocks]
        name = f'overlap{rank}_restricted'
        print(f"{name}: {len(terms)} terms")
        funcs.append(assemble(name, params, aliases, 'result', '()', terms, []))
    return funcs


_cross_cache_r = {}


def cross_density_terms_restricted(block, bra_ranks, ket_ranks):
    """Restricted analogue of generate_mpn_density.cross_density_terms, for
    one DENSITY_BLOCKS entry (always the '_aa' spin block, per this repo's
    closed-shell convention -- off-diagonal spin blocks vanish exactly)."""
    key = (block, tuple(bra_ranks), tuple(ket_ranks))
    if key in _cross_cache_r:
        return _cross_cache_r[key]
    if not bra_ranks and not ket_ranks:
        raise ValueError("bra_ranks and ket_ranks cannot both be empty")
    insert_op, spin_labels = DENSITY_BLOCKS[block]

    pq = pq_helper('fermi')
    pq.set_left_operators([[f'l{r}'] for r in bra_ranks] if bra_ranks else [['1']])
    pq.set_right_operators([['1']])
    if ket_ranks:
        for r in ket_ranks:
            pq.add_operator_product(1.0, [insert_op, f't{r}'])
    else:
        pq.add_operator_product(1.0, [insert_op])
    pq.simplify()
    terms = pq.strings(spin_labels=spin_labels)
    pq.clear()
    _cross_cache_r[key] = terms
    return terms


DENSITY_SHAPE_R = {'oo': '(no, no)', 'vv': '(nv, nv)', 'ov': '(no, nv)', 'vo': '(nv, no)'}
DENSITY_OUT_IDX_R = {'oo': ['m', 'n'], 'vv': ['e', 'f'], 'ov': ['m', 'e'], 'vo': ['e', 'm']}


def _rank_aliases_and_params(ranks, side, order_of):
    """side: 'l' or 't'. order_of: dict rank -> order (the specific stored
    amplitude order feeding this bra/ket role for that rank)."""
    aliases, params = {}, []
    for r in ranks:
        order = order_of[r]
        for sub in ALL_BLOCKS[r]:
            real_sub = BB_TWIN.get(sub, sub)
            generic_name = f'{side}{r}_{sub}'
            real_name = f'{side}_{r}_{order}_{real_sub}'
            aliases[generic_name] = real_name
            if real_name not in params:
                params.append(real_name)
    return aliases, params


def generate_density_pieces_restricted(pieces):
    """pieces: list of (name, block, bra_ranks, bra_order_of, ket_ranks, ket_order_of)."""
    funcs = []
    for name, block, bra_ranks, bra_order_of, ket_ranks, ket_order_of in pieces:
        terms = cross_density_terms_restricted(block, bra_ranks, ket_ranks)
        aliases, params = dict(), list(COMMON_ARGS_R)
        bra_al, bra_params = _rank_aliases_and_params(bra_ranks, 'l', bra_order_of)
        ket_al, ket_params = _rank_aliases_and_params(ket_ranks, 't', ket_order_of)
        aliases.update(bra_al)
        aliases.update(ket_al)
        params += bra_params + ket_params
        print(f"{name}: {len(terms)} terms")
        funcs.append(assemble(name, params, aliases, 'result', DENSITY_SHAPE_R[block], terms, DENSITY_OUT_IDX_R[block]))
    return funcs


def _write(path, funcs, header):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(header)
        fh.write("import numpy as np\nfrom src.SingleReference.CC.cached_einsum import einsum\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"wrote {path}")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    open(os.path.join(OUT_DIR, '__init__.py'), 'a').close()

    # --- amplitude numerators: T2^(1); T1^(2),T2^(2),T3^(2),T4^(2) (T4^(2)
    # new for MP4); T1^(3),T2^(3),T3^(3) (T2^(3) needs its -E^(2)*T2^(1)
    # correction applied numerically in the driver, see
    # generate_mpn_density.py's docstring); T1^(4) (new, MP4's boundary
    # term). Mirrors generate_mpn_density.py's spin-orbital __main__ exactly.
    amp_funcs = generate_amplitude_numerators_restricted([1, 2, 3])
    amp_funcs += _amplitude_numerator_funcs_restricted(1, 4)
    # Production (laplace-mode) variant of t1_3_aa_numerator: excludes rank 3
    # (T3^(2)) from connect_ranks entirely, so this body never takes
    # t3_2_aaaaaa/aabaab/abbabb as a parameter at all -- their contribution
    # is instead added by generate_mp3_t3_laplace_restricted.py's Laplace-
    # fused pieces (mpn_density_driver_restricted.py's compute_t1_3 adds
    # both). See that module's docstring for why even a zero rank-6
    # placeholder isn't acceptable here (the ndim<=4 invariant is enforced
    # per-operand, not per-value).
    amp_funcs += _amplitude_numerator_no_t3_funcs_restricted(1, 3, max_connect_rank=2)

    # --- overlap pieces: rank 2 (N^(2), N^(3)); ranks 1,3,4 newly needed for
    # N^(4)'s full <Psi2|Psi2> self-overlap.
    overlap_funcs = generate_overlaps_restricted([1, 2, 3, 4])

    density_pieces = []
    for block in ('oo', 'vv', 'ov'):
        density_pieces += [
            (f'm2_{block}_11_restricted', block, [2], {2: 1}, [2], {2: 1}),
            (f'm2_{block}_20_restricted', block, [1], {1: 2}, [], {}),
            (f'm2_{block}_02_restricted', block, [], {}, [1], {1: 2}),
            # rank 4 of Psi^(2) is excluded from the m3 12/21 pieces: a
            # one-body operator cannot couple <D2| and |D4>, so pdaggerq
            # produces no surviving t4/l4 terms for them anyway (verified:
            # the previously-generated bodies contained zero t4/l4 einsums,
            # only dead parameter aliases) -- listing rank 4 here only forced
            # the driver to materialize the O(nv^4*no^4) T4^(2) tensors for a
            # plain MP3 density.
            (f'm3_{block}_12_restricted', block, [2], {2: 1},
             [r for r in ranks_present(2) if r <= 3], {r: 2 for r in ranks_present(2) if r <= 3}),
            (f'm3_{block}_21_restricted', block,
             [r for r in ranks_present(2) if r <= 3], {r: 2 for r in ranks_present(2) if r <= 3}, [2], {2: 1}),
            (f'm3_{block}_30_restricted', block, [1, 2, 3], {r: 3 for r in (1, 2, 3)}, [], {}),
            (f'm3_{block}_03_restricted', block, [], {}, [1, 2, 3], {r: 3 for r in (1, 2, 3)}),
        ]
        if block == 'ov':
            # Production (laplace-mode) variant of m3_ov_12_restricted:
            # ket_ranks capped at 2, so this body never takes
            # t_3_2_aaaaaa/aabaab/abbabb as a parameter at all -- see
            # t1_3_aa_numerator_no_t3's comment above for why (and
            # generate_mp3_t3_laplace_restricted.py for where the T3^(2)
            # contribution comes from instead). oo/vv's m3_12 bodies don't
            # need this: they never reference t_3_2_* in an einsum anyway
            # (dead parameter, verified by inspection), so a zero rank-6
            # placeholder there is merely unused, never passed to einsum.
            density_pieces.append(
                ('m3_ov_12_restricted_no_t3', block, [2], {2: 1}, [1, 2], {1: 2, 2: 2}))
    for block in ('oo', 'vv', 'ov'):
        density_pieces += [
            (f'm4_{block}_13_restricted', block, [2], {2: 1}, [1, 2, 3], {r: 3 for r in (1, 2, 3)}),
            (f'm4_{block}_31_restricted', block, [1, 2, 3], {r: 3 for r in (1, 2, 3)}, [2], {2: 1}),
            (f'm4_{block}_22_restricted', block, ranks_present(2), {r: 2 for r in ranks_present(2)},
             ranks_present(2), {r: 2 for r in ranks_present(2)}),
            (f'm4_{block}_40_restricted', block, [1], {1: 4}, [], {}),
            (f'm4_{block}_04_restricted', block, [], {}, [1], {1: 4}),
        ]
    density_funcs = generate_density_pieces_restricted(density_pieces)

    _write(os.path.join(OUT_DIR, 'mpn_density_pieces_restricted.py'), amp_funcs + overlap_funcs + density_funcs,
           "# GENERATED CODE -- restricted (spin-blocked) MPn amplitude-numerator,\n"
           "# overlap, and density cross-term pieces, order-generic pdaggerq recursion.\n"
           "# Produced by src/CodeGeneration/generate_mpn_density_restricted.py; see\n"
           "# that module's docstring and generate_mpn_density.py's docstring for the\n"
           "# recursion these pieces are combined through. Do not edit by hand.\n")

    # --- DF/RI variant (Phase 1 of the DF rewrite): only t2_1 and
    # t1_2 carry a bracket integral factor at n=2 -- see
    # _amplitude_numerator_funcs_restricted_df's docstring. Phase 3 adds the
    # n=3 pieces: t2_2 (production, includes the memory-critical vvvv term),
    # t3_2 (non-Laplace oracle/MP4 path only), t1_3's full and '_no_t3'
    # (production) variants. t1_3_aa_numerator_df additionally needs
    # t3_2_aaaaaa/aabaab/abbabb_df as params (full connect_ranks includes
    # rank 3) -- generated before it so those DF functions already exist.
    df_funcs = _amplitude_numerator_funcs_restricted_df(2, 1) + _amplitude_numerator_funcs_restricted_df(1, 2)
    df_funcs += _amplitude_numerator_funcs_restricted_df(2, 2)
    df_funcs += _amplitude_numerator_funcs_restricted_df(3, 2)
    df_funcs += _amplitude_numerator_funcs_restricted_df(1, 3)
    df_funcs += _amplitude_numerator_no_t3_funcs_restricted_df(1, 3, max_connect_rank=2)
    _write(os.path.join(OUT_DIR, 'mpn_density_pieces_restricted_df.py'), df_funcs,
           "# GENERATED CODE -- DF/RI variant of t2_1/t1_2/t2_2/t3_2/t1_3 numerators\n"
           "# (Phases 1+3 of the DF rewrite): takes B_aa/B_bb (3-index RI\n"
           "# factors) instead of g_aaaa/g_abab/g_bbbb, never forming a norb^4-scale\n"
           "# integral block. Every OTHER density piece (overlap2/3, m2_*/m3_* cross-\n"
           "# terms) has no bracket integral factor at all and is reused UNCHANGED from\n"
           "# mpn_density_pieces_restricted.py.\n"
           "# Produced by src/CodeGeneration/generate_mpn_density_restricted.py.\n"
           "# Do not edit by hand.\n")
