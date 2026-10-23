"""Codegen script (run once, output committed): the spin-blocked MPn density
pieces, restricted (closed-shell RHF) and unrestricted (UHF), from the
order-generic pdaggerq recursion of generate_mpn_density.py (see its
docstring for the physics: T_rank^(order) amplitude numerators, the
N^(n)/M^(n)/c^(n) density recursion). Every equation is derived with
pdaggerq's spin integration (pq.strings(spin_labels=...)), so the generated
code works on spatial-MO spin-block arrays (t2_aaaa/t2_abab/..., the
generate_restricted_ccsdt.py storage convention) instead of the
(2*nmo)-dimensional spin-orbital tensors.

One recursion, two spin models (SpinModel):
- RESTRICTED: only the alpha-majority blocks are stored ('_aa', '_aaaa',
  '_abab', '_aaaaaa', '_aabaab', '_abbabb', ...); a '_bb'-type leaf a raw
  equation references is numerically its '_aa'-type twin and is aliased at call
  time (restricted_solver.py's convention); one shared orbital window
  (o, v, nv, no); the density insertion is derived for the alpha block only
  (off-diagonal spin blocks vanish, the spatial density is twice it).
- UNRESTRICTED: every spin block is its own array (the all-beta block is not a
  twin), every occ/vir dimension is per spin (no_a/nv_a/no_b/nv_b), the
  density is derived per spin channel. Scope MP2 + MP3 (MAX_RANK_U = 3): rank
  4 of Psi^(2) cannot couple to <D2| through a one-body operator and ranks 2/3
  of Psi^(3) cannot couple to <Phi0|, so nothing above rank 3 reaches the n <= 3
  density. pdaggerq_interface.array_expr slices with spin-blind 'o'/'v'
  letters, so every emitted line passes spinify_slices: the factor's own spin
  suffix gives the spin of each axis positionally (g_abab[o, o, v, v] ->
  g_abab[o_a, o_b, v_a, v_b]).

The bra projector is the explicit a*/a product generate_mpn_density.bra_ops
(any rank; the e{rank} string pdaggerq caps at rank 4 reproduces it exactly
at ranks 1-4), with the stored blocks' spin labels on its pairs
(occ[p], vir[p]). Rank presence (which T_rank^(order) is nonzero) and the
disconnected-piece filter are the spin-orbital facts of generate_mpn_density.

Both models have a DF/RI twin of the numerators that carry a two-electron
bracket (t2_1, t1_2, t2_2, t3_2, t1_3): B_aa/B_bb in place of g_aaaa/g_abab/
g_bbbb, no norb^4 integral block; the overlap and density pieces have no
bracket and are reused unchanged. The restricted DF bodies stage their
three-operand contractions (_stage_three_operand_einsums).
"""
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.CodeGeneration.pdaggerq_interface import raw_terms_to_einsum, raw_terms_to_term_list
from src.SingleReference.ADC.adc_order_rule import refuse_above_max_order
from src.CodeGeneration import codegen_common
from src.CodeGeneration.codegen_common import (
    eri_leaves, R_DIMS, U_DIMS, density_insert, density_out_idx, density_shape,
    density_spin_labels, pq_terms, generated_header, write_module)
from src.CodeGeneration.generate_mpn_density import (
    ranks_present, filter_connected, amp_out_indices, assemble, bra_ops, OCC_LETTERS, VIR_LETTERS)
from src.CodeGeneration.df_codegen import dress_integral_factor
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import ORDER4_MODULE

MAX_RANK_U = 3  # MP2+MP3 density only -- see module docstring


@dataclass(frozen=True)
class SpinModel:
    """restricted: closed-shell twins and one orbital window; else UHF."""
    name: str
    restricted: bool
    max_rank: int

    def blocks(self, rank):
        """{stored block: spin labels of the bra letters}. Block k puts k beta
        pairs at the END of the pair ordering (pair p = (occ[p], vir[p]), one
        spin); restricted stores k = 0..rank-1 (k = rank, all-beta, is the
        global spin-flip twin of k = 0), unrestricted k = 0..rank."""
        occ, vir = OCC_LETTERS[:rank], VIR_LETTERS[:rank]
        out = {}
        for k in range(rank + (0 if self.restricted else 1)):
            spins = ['a'] * (rank - k) + ['b'] * k
            labels = {}
            for p in range(rank):
                labels[occ[p]] = spins[p]
                labels[vir[p]] = spins[p]
            out[''.join(spins) * 2] = labels
        return out

    def all_blocks(self, rank):
        """Every block name a generic t{rank} tag can print as (restricted: the
        stored ones plus the all-beta twin)."""
        names = list(self.blocks(rank))
        return names + [('b' * rank) * 2] if self.restricted else names

    def twin(self, block):
        """The stored block holding `block`'s values."""
        if self.restricted and len(set(block)) == 1 and block[0] == 'b':
            return 'a' * len(block)
        return block

    def common_args(self, df=False):
        if self.restricted:
            return eri_leaves(df) + ['d_aa'] + R_DIMS
        return eri_leaves(df) + ['d_aa', 'd_bb'] + U_DIMS

    def amp_shape(self, rank, block):
        if self.restricted:
            return '(' + ', '.join(['nv'] * rank + ['no'] * rank) + ')'
        spins = block[:rank]
        return '(' + ', '.join([f'nv_{s}' for s in spins] + [f'no_{s}' for s in spins]) + ')'

    def connect_ranks(self, rank, order):
        if order <= 1:
            return []
        return [r for r in ranks_present(order - 1) if abs(r - rank) <= 2 and r <= self.max_rank]

    def assemble(self, name, params, aliases, shape, terms, out_idx, free_temps=False):
        if self.restricted:
            return assemble(name, params, aliases, 'result', shape, terms, out_idx,
                            free_temps=free_temps)
        return assemble_spinified(name, params, aliases, 'result', shape, terms, out_idx)


RESTRICTED = SpinModel('restricted', True, 4)
UNRESTRICTED = SpinModel('unrestricted', False, MAX_RANK_U)

T_BRA = {r: (bra_ops(r), RESTRICTED.blocks(r)) for r in range(1, RESTRICTED.max_rank + 1)}
ALL_BLOCKS_U = {r: UNRESTRICTED.all_blocks(r) for r in range(1, MAX_RANK_U + 1)}


# ---------------------------------------------------------------------------
# UHF slicing
# ---------------------------------------------------------------------------

_SPIN_SLICE_RE = re.compile(r'\b([gdf])(_[ab]+)\[([ov](?:, [ov])*)\]')
# B (DF/RI factor)'s slice list carries a leading unsliced aux axis (':')
# ahead of the o/v tokens -- see pdaggerq_interface.array_expr's B branch --
# so it needs its own pattern. B_aa/B_bb's suffix is always two equal letters
# (both axes share the block's one spin, per df_codegen._b_name).
_SPIN_SLICE_RE_B = re.compile(r'\bB(_[ab]+)\[:, ([ov](?:, [ov])*)\]')


def spinify_slices(line):
    """Rewrite spin-blind slice lists emitted by raw_terms_to_einsum into
    per-spin ones using the factor's own spin suffix positionally:
    'g_abab[o, o, v, v]' -> 'g_abab[o_a, o_b, v_a, v_b]'. See module
    docstring."""
    def repl(m, lead):
        suffix, slices = m.group(lead), m.group(lead + 1)
        spins = suffix[1:]
        parts = [p.strip() for p in slices.split(',')]
        if len(parts) != len(spins):
            raise ValueError(f"slice/suffix arity mismatch in '{m.group(0)}'")
        return ', '.join(f'{p}_{s}' for p, s in zip(parts, spins))

    line = _SPIN_SLICE_RE.sub(lambda m: m.group(1) + m.group(2) + '[' + repl(m, 2) + ']', line)
    return _SPIN_SLICE_RE_B.sub(lambda m: 'B' + m.group(1) + '[:, ' + repl(m, 1) + ']', line)


def assemble_spinified(func_name, params, aliases, out_var, out_shape_expr, raw_terms, out_indices):
    """generate_mpn_density.assemble with spinify_slices applied to every
    emitted einsum line."""
    src = [f"def {func_name}({', '.join(params)}):"]
    for generic, real in aliases.items():
        if generic != real:
            src.append(f"    {generic} = {real}")
    src.append(f"    {out_var} = np.zeros({out_shape_expr})")
    src.extend(spinify_slices(line) for line in raw_terms_to_einsum(raw_terms, out_var, out_indices))
    src.append(f"    return {out_var}")
    src.append("")
    return "\n".join(src)


# ---------------------------------------------------------------------------
# DF bodies (restricted): staged three-operand contractions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Amplitude numerators
# ---------------------------------------------------------------------------

_numerator_cache = {}


def t_numerator_terms(model, rank, order, block, max_connect_rank=None):
    """Raw pdaggerq terms of T_rank^(order)'s V-connected numerator
    <bra_rank|v|Psi^(order-1)> for one stored spin block (the spin-blocked
    generate_mpn_density.t_numerator_terms). max_connect_rank drops the higher
    ranks of Psi^(order-1) from the derivation (the '_no_t3' variants)."""
    key = (model.name, rank, order, block, max_connect_rank)
    if key in _numerator_cache:
        return _numerator_cache[key]
    spin_labels = model.blocks(rank)[block]
    if order == 1:
        products = [(1.0, ['v'])] if rank == 2 else []
    else:
        products = [(1.0, ['v', f't{r}']) for r in model.connect_ranks(rank, order)
                    if max_connect_rank is None or r <= max_connect_rank]
    terms = (filter_connected(pq_terms([bra_ops(rank)], products, spin_labels), amp_out_indices(rank))
             if products else [])
    _numerator_cache[key] = terms
    return terms


def t_numerator_terms_restricted(rank, order, block):
    """t_numerator_terms of the restricted model."""
    return t_numerator_terms(RESTRICTED, rank, order, block)


def t_numerator_terms_u(rank, order, block):
    """t_numerator_terms of the unrestricted model."""
    return t_numerator_terms(UNRESTRICTED, rank, order, block)


def t_numerator_terms_stored(rank, order, block):
    """Restricted t_numerator_terms for any spin block of the rank: a block
    whose numerator is not stored (the all-beta bb..b) is its closed-shell
    a<->b twin, whose numerator arrays (g, t2) alias the stored ones, so the
    twin's terms reproduce it numerically."""
    if block not in RESTRICTED.blocks(rank):
        flipped = block.translate(str.maketrans('ab', 'ba'))
        if flipped not in RESTRICTED.blocks(rank):
            raise KeyError(f"no numerator key for rank-{rank} block {block!r}")
        block = flipped
    return t_numerator_terms_restricted(rank, order, block)


def _amp_aliases(model, ranks, side, order_of, params):
    """Every block name a generic {side}{r} tag prints as, aliased to the stored
    array {side}_{r}_{order}_{block} (restricted twins onto the stored block);
    new real names appended to params."""
    aliases = {}
    for r in ranks:
        for sub in model.all_blocks(r):
            generic = f'{side}{r}_{sub}'
            real = f'{side}_{r}_{order_of[r]}_{model.twin(sub)}'
            aliases[generic] = real
            if real not in params:
                params.append(real)
    return aliases


def amplitude_numerator_funcs(model, rank, order, df=False, max_connect_rank=None):
    """One (rank, order) amplitude numerator, every stored spin block:
    t{rank}_{order}_{block}_numerator[_no_t3][_df]. max_connect_rank caps the
    connected ranks of Psi^(order-1) in the derivation AND the parameters: the
    '_no_t3' production (Laplace-mode) variant of t1_3, which never takes
    t3_2_* at all, not even as a dead parameter (the T3^(2) contribution comes
    from the Laplace-fused pieces of generate_mpn_laplace, and a zero rank-6
    placeholder is not acceptable: the ndim <= 4 invariant is checked per
    operand). df=True: the DF/RI twin."""
    # ADC(n) reads amplitudes through order n (T1^(n) in its f-dressed structures)
    refuse_above_max_order(order, 'amplitude_numerator_funcs')
    connect = [r for r in model.connect_ranks(rank, order)
               if max_connect_rank is None or r <= max_connect_rank]
    params = model.common_args(df)
    aliases = {}
    for r in connect:
        for sub in model.all_blocks(r):
            real = f't{r}_{order - 1}_{model.twin(sub)}'
            aliases[f't{r}_{sub}'] = real
            if real not in params:
                params.append(real)
    tail = ('_no_t3' if max_connect_rank is not None else '') + ('_df' if df else '')
    funcs = []
    for block in model.blocks(rank):
        terms = t_numerator_terms(model, rank, order, block, max_connect_rank)
        name = f't{rank}_{order}_{block}_numerator{tail}'
        shape = model.amp_shape(rank, block)
        if df:
            df_terms = dress_integral_factor(terms)
            print(f"{name}: {len(df_terms)} DF terms (from {len(terms)} dense terms)")
            funcs.append(_assemble_df(name, params, aliases, shape, df_terms, amp_out_indices(rank))
                         if model.restricted else
                         model.assemble(name, params, aliases, shape, df_terms, amp_out_indices(rank)))
        else:
            print(f"{name}: {len(terms)} terms")
            funcs.append(model.assemble(name, params, aliases, shape, terms, amp_out_indices(rank),
                                        free_temps=True))
    return funcs


def generate_amplitude_numerators_restricted(orders):
    """Every restricted numerator of `orders` (the ranks ranks_present finds)."""
    funcs = []
    for order in orders:
        for rank in ranks_present(order):
            if rank in T_BRA:
                funcs += amplitude_numerator_funcs(RESTRICTED, rank, order)
    return funcs


def _product_aliases(factors):
    """(aliases, params) of a product rung: the generic t{r}_{block} tags of the
    factors ((rank, order), ...) onto the stored t{r}_{order}_{block} arrays."""
    aliases, params = {}, RESTRICTED.common_args()
    for r, k in factors:
        for sub in RESTRICTED.all_blocks(r):
            real = f't{r}_{k}_{RESTRICTED.twin(sub)}'
            aliases[f't{r}_{sub}'] = real
            if real not in params:
                params.append(real)
    return aliases, params


def product_rung_terms(rank, factors, coef, block, connected):
    """Raw restricted terms of <bra_rank|v X_1 ... X_n|0> for one stored spin
    block, X_t the connected amplitudes of `factors` ((rank, order), ...).
    connected=True drops the vacuum pieces (v closed on one factor), which are
    exactly the E^(k) T_rank^(order-k) products of RSPT's subtraction."""
    key = ('product', rank, tuple(factors), coef, block, connected)
    if key not in _numerator_cache:
        terms = pq_terms([bra_ops(rank)], [(coef, ['v'] + [f't{r}' for r, _ in factors])],
                         RESTRICTED.blocks(rank)[block])
        if connected:
            terms = filter_connected(terms, amp_out_indices(rank))
        _numerator_cache[key] = terms
    return _numerator_cache[key]


def product_rung_funcs(rank, order, factors, coef, connected=False):
    """The v . (X_1 ... X_n) rung of T_rank^(order)'s numerator for a rank of
    Psi^(order-1) above the restricted MAX rank whose CI coefficient is an exact
    product of connected amplitudes (c5^(3) = A[t3^(2) t2^(1)]): derived by
    pdaggerq on the product, so no rank-5 amplitude is spin-traced or formed.
    t{rank}_{order}_{block}_numerator_{t..t}; coef is the product's 1/m!
    symmetry factor. connected=False keeps the unlinked piece, which cancels
    against the E^(k) T_rank^(order-k) terms the driver subtracts."""
    aliases, params = _product_aliases(factors)
    tag = ''.join(f't{r}' for r, _ in factors)
    funcs = []
    for block in RESTRICTED.blocks(rank):
        terms = product_rung_terms(rank, factors, coef, block, connected)
        name = f't{rank}_{order}_{block}_numerator_{tag}'
        print(f"{name}: {len(terms)} terms")
        funcs.append(RESTRICTED.assemble(name, params, aliases, RESTRICTED.amp_shape(rank, block),
                                         terms, amp_out_indices(rank), free_temps=True))
    return funcs


def term_table_funcs(rank, order):
    """t{rank}_{order}_{block}_terms: the numerator <bra_rank|v|Psi^(order-1)> of
    each stored block as a term table [(coef, ((subscript, operand), ...))]
    (pdaggerq_interface.raw_terms_to_term_list), external letters
    amp_out_indices(rank): the input of an amplitude that is never formed
    (laplace_amplitudes.LaplaceAmplitude)."""
    params = RESTRICTED.common_args()
    aliases = {}
    for r in RESTRICTED.connect_ranks(rank, order):
        for sub in RESTRICTED.all_blocks(r):
            real = f't{r}_{order - 1}_{RESTRICTED.twin(sub)}'
            aliases[f't{r}_{sub}'] = real
            if real not in params:
                params.append(real)
    funcs = []
    for block in RESTRICTED.blocks(rank):
        terms = t_numerator_terms(RESTRICTED, rank, order, block)
        name = f't{rank}_{order}_{block}_terms'
        print(f"{name}: {len(terms)} raw terms")
        src = [f"def {name}({', '.join(params)}):"]
        src += [f"    {g} = {r}" for g, r in aliases.items() if g != r]
        src.append("    terms = []")
        src += raw_terms_to_term_list(terms, 'terms', amp_out_indices(rank))
        src += ["    return terms", ""]
        funcs.append("\n".join(src))
    return funcs


# the quintuples of Psi^(3) as the exact product c5^(3) = A[t3^(2) t2^(1)]
QUINTUPLE_RUNG = (((3, 2), (2, 1)), 1.0)


def generate_order4_ground_state_restricted():
    """generated_mpn_restricted/mpn_order4_ground_state_restricted.py (not
    committed; GenerateRuntime writes it with ADC(5)): the order-4 ground state
    non-Dyson ADC(5) reads, T2^(4), T3^(4) (its quintuples rung as the product
    t3^(2) t2^(1)) and T1^(5) as numerators, and T4^(3)'s numerator as term
    tables for the Laplace-fused, never-formed T4^(3) (its dense numerator is
    in mpn_density_pieces_restricted)."""
    R = RESTRICTED
    funcs = amplitude_numerator_funcs(R, 2, 4) + amplitude_numerator_funcs(R, 3, 4)
    funcs += product_rung_funcs(3, 4, *QUINTUPLE_RUNG)
    funcs += amplitude_numerator_funcs(R, 1, 5)
    funcs += term_table_funcs(4, 3)
    write_module(_out(R, f'{ORDER4_MODULE}.py'), funcs, generated_header(
        'generate_mpn_density_spin.py (generate_order4_ground_state_restricted)',
        'Restricted (spin-blocked) order-4 ground state: T2^(4), T3^(4) (quintuples rung as',
        'the product t3^(2) t2^(1)) and T1^(5) numerators, T4^(3) numerator term tables.'))


# ---------------------------------------------------------------------------
# Overlaps and density cross terms
# ---------------------------------------------------------------------------

_overlap_cache = {}


def overlap_terms_spin(rank):
    """<l{rank}|t{rank}> spin-decomposed (spin_labels={}: a scalar decomposes
    into aaaa/abab/.../bbbb pieces with no free index to assign)."""
    if rank not in _overlap_cache:
        _overlap_cache[rank] = pq_terms([[f'l{rank}']], [(1.0, [f't{rank}'])], {})
    return _overlap_cache[rank]


def generate_overlap(model, rank):
    """overlap{rank}_{model}: the l/t parameters are generic (l_aaaa, t_aaaa,
    ...), not order-tied -- the driver supplies whichever order's amplitudes
    play the l/t roles at each call site."""
    aliases = {}
    for sub in model.all_blocks(rank):
        aliases[f'l{rank}_{sub}'] = f'l_{model.twin(sub)}'
        aliases[f't{rank}_{sub}'] = f't_{model.twin(sub)}'
    stored = list(model.blocks(rank))
    params = [f'l_{s}' for s in stored] + [f't_{s}' for s in stored]
    name = f'overlap{rank}_{model.name}'
    terms = overlap_terms_spin(rank)
    print(f"{name}: {len(terms)} terms")
    return model.assemble(name, params, aliases, '()', terms, [])


_cross_cache = {}


def cross_density_terms(block, spin, bra_ranks, ket_ranks):
    """Raw terms of sum <l{r}| e1(block) |t{r'}> on spin channel `spin`, one
    M^(n) piece for a fixed (bra order, ket order) split; empty bra_ranks or
    ket_ranks means that side is Phi0 (not both)."""
    key = (block, spin, tuple(bra_ranks), tuple(ket_ranks))
    if key in _cross_cache:
        return _cross_cache[key]
    if not bra_ranks and not ket_ranks:
        raise ValueError("bra_ranks and ket_ranks cannot both be empty")
    op = density_insert(block)
    products = ([(1.0, [op, f't{r}']) for r in ket_ranks] if ket_ranks else [(1.0, [op])])
    terms = pq_terms([[f'l{r}'] for r in bra_ranks] if bra_ranks else [['1']], products,
                     density_spin_labels(block, spin))
    _cross_cache[key] = terms
    return terms


def cross_density_terms_restricted(block, bra_ranks, ket_ranks):
    """cross_density_terms on the alpha channel (the restricted density block)."""
    return cross_density_terms(block, 'a', bra_ranks, ket_ranks)


def cross_density_terms_u(block, spin, bra_ranks, ket_ranks):
    """cross_density_terms of the unrestricted model."""
    return cross_density_terms(block, spin, bra_ranks, ket_ranks)


def generate_density_pieces(model, pieces):
    """pieces: (name, block, spin, bra_ranks, bra_order_of, ket_ranks,
    ket_order_of); order_of maps rank -> the stored amplitude order feeding
    that bra/ket role."""
    funcs = []
    for name, block, spin, bra_ranks, bra_order_of, ket_ranks, ket_order_of in pieces:
        terms = cross_density_terms(block, spin, bra_ranks, ket_ranks)
        params = model.common_args()
        aliases = _amp_aliases(model, bra_ranks, 'l', bra_order_of, params)
        aliases.update(_amp_aliases(model, ket_ranks, 't', ket_order_of, params))
        print(f"{name}: {len(terms)} terms")
        funcs.append(model.assemble(name, params, aliases,
                                    density_shape(block, None if model.restricted else spin),
                                    terms, density_out_idx(block)))
    return funcs


def restricted_density_pieces():
    """M^(2)..M^(5) pieces of the restricted density (alpha block)."""
    p2 = ranks_present(2)
    low2 = [r for r in p2 if r <= 3]
    pieces = []
    for block in ('oo', 'vv', 'ov'):
        pieces += [
            (f'm2_{block}_11_restricted', block, 'a', [2], {2: 1}, [2], {2: 1}),
            (f'm2_{block}_20_restricted', block, 'a', [1], {1: 2}, [], {}),
            (f'm2_{block}_02_restricted', block, 'a', [], {}, [1], {1: 2}),
            # rank 4 of Psi^(2) is excluded from the m3 12/21 pieces: a
            # one-body operator cannot couple <D2| and |D4> (pdaggerq gives no
            # t4/l4 terms), and listing it only forced the driver to
            # materialize the O(nv^4*no^4) T4^(2) for a plain MP3 density
            (f'm3_{block}_12_restricted', block, 'a', [2], {2: 1}, low2, {r: 2 for r in low2}),
            (f'm3_{block}_21_restricted', block, 'a', low2, {r: 2 for r in low2}, [2], {2: 1}),
            (f'm3_{block}_30_restricted', block, 'a', [1, 2, 3], {r: 3 for r in (1, 2, 3)}, [], {}),
            (f'm3_{block}_03_restricted', block, 'a', [], {}, [1, 2, 3], {r: 3 for r in (1, 2, 3)}),
        ]
        if block == 'ov':
            # production (Laplace-mode) m3_ov_12: ket ranks capped at 2, so it
            # never takes t_3_2_*; the oo/vv m3_12 bodies never reference
            # t_3_2_* in an einsum (dead parameter), so they need no variant
            pieces.append(('m3_ov_12_restricted_no_t3', block, 'a', [2], {2: 1}, [1, 2],
                           {1: 2, 2: 2}))
    r123 = {r: 3 for r in (1, 2, 3)}
    for block in ('oo', 'vv', 'ov'):
        pieces += [
            (f'm4_{block}_13_restricted', block, 'a', [2], {2: 1}, [1, 2, 3], r123),
            (f'm4_{block}_31_restricted', block, 'a', [1, 2, 3], r123, [2], {2: 1}),
            (f'm4_{block}_22_restricted', block, 'a', p2, {r: 2 for r in p2}, p2,
             {r: 2 for r in p2}),
            (f'm4_{block}_40_restricted', block, 'a', [1], {1: 4}, [], {}),
            (f'm4_{block}_04_restricted', block, 'a', [], {}, [1], {1: 4}),
        ]
    r1234 = {r: 4 for r in (1, 2, 3, 4)}
    for block in ('oo', 'vv', 'ov'):
        # MP5 M^(5) (the ADC(6) static self-energy input). RANK-5 CAP: m5_23/m5_32
        # carry a rank-5 channel (Psi^(3)'s T5) that rank-capped T_BRA cannot
        # derive; the driver adds it. m5_14/41/50/05 are complete (rank <= 4).
        pieces += [
            (f'm5_{block}_14_restricted', block, 'a', [2], {2: 1}, [1, 2, 3, 4], r1234),
            (f'm5_{block}_41_restricted', block, 'a', [1, 2, 3, 4], r1234, [2], {2: 1}),
            (f'm5_{block}_23_restricted', block, 'a', [1, 2, 3, 4], {r: 2 for r in (1, 2, 3, 4)},
             [1, 2, 3, 4], {r: 3 for r in (1, 2, 3, 4)}),
            (f'm5_{block}_32_restricted', block, 'a', [1, 2, 3, 4], {r: 3 for r in (1, 2, 3, 4)},
             [1, 2, 3, 4], {r: 2 for r in (1, 2, 3, 4)}),
            (f'm5_{block}_50_restricted', block, 'a', [1], {1: 5}, [], {}),
            (f'm5_{block}_05_restricted', block, 'a', [], {}, [1], {1: 5}),
        ]
    return pieces


def unrestricted_density_pieces():
    """M^(2), M^(3) pieces of the unrestricted density, both spin channels.
    12/21 exclude rank 4 (no coupling to <D2| through a one-body operator),
    30/03 keep rank 1 only (<Phi0|e|D_r> = 0 for r >= 2)."""
    pieces = []
    for block in ('oo', 'vv', 'ov'):
        for s in ('a', 'b'):
            pieces += [
                (f'm2_{block}_{s}_11_unrestricted', block, s, [2], {2: 1}, [2], {2: 1}),
                (f'm2_{block}_{s}_20_unrestricted', block, s, [1], {1: 2}, [], {}),
                (f'm2_{block}_{s}_02_unrestricted', block, s, [], {}, [1], {1: 2}),
                (f'm3_{block}_{s}_12_unrestricted', block, s, [2], {2: 1},
                 [1, 2, 3], {r: 2 for r in (1, 2, 3)}),
                (f'm3_{block}_{s}_21_unrestricted', block, s,
                 [1, 2, 3], {r: 2 for r in (1, 2, 3)}, [2], {2: 1}),
                (f'm3_{block}_{s}_30_unrestricted', block, s, [1], {1: 3}, [], {}),
                (f'm3_{block}_{s}_03_unrestricted', block, s, [], {}, [1], {1: 3}),
            ]
            if block == 'ov':
                pieces.append((f'm3_ov_{s}_12_unrestricted_no_t3', block, s, [2], {2: 1},
                               [1, 2], {1: 2, 2: 2}))
    return pieces


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

def _out(model, fname):
    return os.path.join(codegen_common.DM_OUT_DIR, f'generated_mpn_{model.name}', fname)


def generate_restricted():
    """generated_mpn_restricted/mpn_density_pieces_restricted[_df].py."""
    R = RESTRICTED
    # amplitude numerators: T2^(1); T1^(2), T2^(2), T3^(2), T4^(2) (MP4); T1^(3),
    # T2^(3), T3^(3) (T2^(3)'s -E^(2) T2^(1) correction is applied numerically by
    # the driver); T1^(4) (MP4's boundary term); and t1_3's '_no_t3' variant
    funcs = generate_amplitude_numerators_restricted([1, 2, 3])
    funcs += amplitude_numerator_funcs(R, 1, 4)
    funcs += amplitude_numerator_funcs(R, 1, 3, max_connect_rank=2)
    # overlaps: rank 2 (N^(2), N^(3)); ranks 1, 3, 4 for N^(4)'s <Psi2|Psi2>
    funcs += [generate_overlap(R, r) for r in (1, 2, 3, 4)]
    funcs += generate_density_pieces(R, restricted_density_pieces())
    write_module(_out(R, 'mpn_density_pieces_restricted.py'), funcs, generated_header(
        'generate_mpn_density_spin.py (generate_restricted)',
        'Restricted (spin-blocked) MPn amplitude-numerator, overlap and density cross-term',
        'pieces of the order-generic pdaggerq recursion (generate_mpn_density.py).'),
        package_init=True)
    # DF/RI twins of the bracket-carrying numerators: t2_1, t1_2, t2_2 (with the
    # memory-critical vvvv term), t3_2 (non-Laplace oracle/MP4 path), t1_3 full
    # (needs t3_2_*_df, generated before it) and '_no_t3'
    df = []
    for rank, order in ((2, 1), (1, 2), (2, 2), (3, 2), (1, 3)):
        df += amplitude_numerator_funcs(R, rank, order, df=True)
    df += amplitude_numerator_funcs(R, 1, 3, df=True, max_connect_rank=2)
    write_module(_out(R, 'mpn_density_pieces_restricted_df.py'), df, generated_header(
        'generate_mpn_density_spin.py (generate_restricted)',
        'DF/RI twins of the t2_1/t1_2/t2_2/t3_2/t1_3 numerators: B_aa/B_bb in place of',
        'g_aaaa/g_abab/g_bbbb; every other piece is reused from mpn_density_pieces_restricted.'))


def generate_unrestricted():
    """generated_mpn_unrestricted/mpn_density_pieces_unrestricted[_df].py."""
    U = UNRESTRICTED
    # the numerators the MP2/MP3 density consumes: T2^(1); T1^(2), T2^(2),
    # T3^(2); T1^(3) and its '_no_t3' variant
    funcs = amplitude_numerator_funcs(U, 2, 1)
    for rank in (1, 2, 3):
        funcs += amplitude_numerator_funcs(U, rank, 2)
    funcs += amplitude_numerator_funcs(U, 1, 3)
    funcs += amplitude_numerator_funcs(U, 1, 3, max_connect_rank=2)
    funcs += [generate_overlap(U, 2)]    # N^(2) = <Psi1|Psi1>, N^(3) = 2<Psi1|Psi2>
    funcs += generate_density_pieces(U, unrestricted_density_pieces())
    write_module(_out(U, 'mpn_density_pieces_unrestricted.py'), funcs, generated_header(
        'generate_mpn_density_spin.py (generate_unrestricted)',
        'Unrestricted (separate alpha/beta) MPn amplitude-numerator, overlap and density',
        'cross-term pieces (MP2+MP3) of the order-generic pdaggerq recursion.'),
        package_init=True)
    df = amplitude_numerator_funcs(U, 2, 1, df=True) + amplitude_numerator_funcs(U, 1, 2, df=True)
    for rank in (2, 3):
        df += amplitude_numerator_funcs(U, rank, 2, df=True)
    df += amplitude_numerator_funcs(U, 1, 3, df=True)
    df += amplitude_numerator_funcs(U, 1, 3, df=True, max_connect_rank=2)
    write_module(_out(U, 'mpn_density_pieces_unrestricted_df.py'), df, generated_header(
        'generate_mpn_density_spin.py (generate_unrestricted)',
        'DF/RI twins of the t2_1/t1_2/t2_2/t3_2/t1_3 numerators (UHF).'))


if __name__ == '__main__':
    generate_restricted()
    generate_unrestricted()
