"""Single point of contact between this repo and pdaggerq.

pdaggerq is OPTIONAL: it is needed only to run the code generators in
src/CodeGeneration, never by a calculation. It is not on PyPI; install it from
source (https://github.com/edeprince3/pdaggerq, the generators were last run
against commit f3a1f29). This module imports without it; pq_helper() raises
ImportError when it is missing, and pdaggerq_available() says whether it works.
Setting MBPT_NO_PDAGGERQ to anything but empty or 0 makes both behave as if it
were missing.

Gotchas (each silently produced wrong numbers before being diagnosed):
1. Only index letters a-f (virtual) and i-n (occupied) are safe in operator
   strings like e3(d,e,f,k,n,m); anything else derives a different equation
   with no error.
2. P(x,y) in raw pq.strings() output means: evaluate the contraction as
   printed, then also add it to the output with axes x,y transposed and a
   sign flip. Multiple P(...) tokens compose independently.
3. Generated einsums with 3+ operands need optimize=True explicitly, or
   numpy's default contraction path can be catastrophically slow.
4. An operator product whose total creation-minus-annihilation count is
   unbalanced returns 0 terms with NO error or warning (particle-number
   superselection) -- a wrong bra/ket pairing looks identical to a
   genuinely vanishing element. Sanity-check a nonzero case first.
5. When a derivation needs more than 6 same-type dummy indices (deep
   Omega-dressed elements, e.g. the ADC(5)-family class-class l2vt2 with
   4 free occupied labels), pq.strings() OVERFLOWS into capital letters of
   the same family: occupied -> I, J, ...; virtual -> A, B, ... (verified
   empirically: <I,l||n,j> occ, t2(A,e,k,l) vir). Capitals are legal
   numpy-einsum subscripts, but any occ/vir classification that only
   knows the lowercase sets silently mis-slices them -- OCC_LETTERS/
   VIR_LETTERS below therefore include the capital counterparts.
"""
import os
import re
from fractions import Fraction
from functools import lru_cache
from itertools import product

from src.Base.constants import PDAGGERQ_COEFFICIENT_PRINT_TOLERANCE

# Denominators a Wick-contraction coefficient can have with amplitudes of rank
# <= 5: products of the rank factorials (2!, 3!, 4!, 5!)^2 and powers of 1/2
# from the antisymmetrized integrals, i.e. 2^a 3^b 5^c.
COEFFICIENT_DENOMINATORS = sorted({2 ** a * 3 ** b * 5 ** c
                                   for a in range(13) for b in range(5) for c in range(3)})


@lru_cache(maxsize=None)
def exact_coefficient(text):
    """The exact rational value of a pq.strings() term coefficient.

    pdaggerq prints the coefficient rounded to seven decimals, so 1/576 (a
    rank-4 amplitude on each side) arrives as 0.0017361, a relative error of
    6e-6 that the emitted code would inherit -- and terms that cancel exactly
    (the rank-4 dressings of a four-electron system) then survive at 1e-9.
    The true coefficient is p/q with q a product of the rank factorials and
    powers of two; the printed value is snapped to the unique such rational
    within the printing tolerance (smallest denominator first). A value that
    matches no candidate is returned as printed."""
    x = float(text)
    hits = []
    for q in COEFFICIENT_DENOMINATORS:
        p = round(x * q)
        if abs(x - p / q) <= PDAGGERQ_COEFFICIENT_PRINT_TOLERANCE:
            hits.append(Fraction(p, q))
    if not hits:
        return x
    values = {h for h in hits}
    if len(values) > 1:
        raise ValueError(f"coefficient {text!r} is within the printing tolerance of "
                         f"several rationals: {sorted(values)}")
    return float(hits[0])

class _IndexFamily(frozenset):
    """Index-family membership by LEADING letter.

    Replaces the plain single-character sets so that NUMBERED pdaggerq labels
    classify correctly: 'i1', 'i12' -> occupied; 'a7', 'a10' -> virtual. This
    is what lets the operator-string layer use an unbounded label pool instead
    of a 6-letter budget per family.

    Subclasses frozenset so iteration/len still see the underlying letters
    (and the capital-overflow counterparts of gotcha 5); only __contains__ is
    widened. Empty labels are not members.
    """
    def __contains__(self, idx):
        s = str(idx)
        return bool(s) and frozenset.__contains__(self, s[0])


OCC_LETTERS = _IndexFamily(set('ijklmnopqrst') | set('IJKLMNOPQRST'))
VIR_LETTERS = _IndexFamily(set('abcdefghuvwxyz') | set('ABCDEFGH'))

# Subsets actually safe in pdaggerq operator strings (gotcha 1 above).
PDAGGERQ_SAFE_OCC = set('ijklmn')
PDAGGERQ_SAFE_VIR = set('abcdef')


def pq_helper(vacuum='fermi'):
    """Return a pdaggerq pq_helper. Mind the reserved-index-letter gotcha (1) above."""
    if os.environ.get('MBPT_NO_PDAGGERQ', '0') not in ('', '0'):
        raise ImportError('pdaggerq disabled by MBPT_NO_PDAGGERQ')
    try:
        import pdaggerq
        return pdaggerq.pq_helper(vacuum)
    except (ImportError, AttributeError) as exc:
        # an empty namespace directory named pdaggerq imports without error
        # and has no pq_helper
        raise ImportError('the code generators need pdaggerq, an optional '
                          'dependency: install it from '
                          'https://github.com/edeprince3/pdaggerq') from exc


def pdaggerq_available():
    """True if pdaggerq can build a pq_helper here (the generator tests skip otherwise)."""
    try:
        pq_helper('fermi')
    except ImportError:
        return False
    return True


def check_safe_indices(op_str):
    """Raise ValueError if an operator string uses an index letter pdaggerq mis-handles (gotcha 1)."""
    m = re.search(r'\((.*?)\)', op_str)
    if not m:
        return
    for idx in m.group(1).split(','):
        idx = idx.strip()
        if idx and idx not in PDAGGERQ_SAFE_OCC | PDAGGERQ_SAFE_VIR:
            raise ValueError(
                f"index '{idx}' in '{op_str}' is outside pdaggerq's safe sets "
                f"(virtual {sorted(PDAGGERQ_SAFE_VIR)}, occupied "
                f"{sorted(PDAGGERQ_SAFE_OCC)}); pdaggerq would silently derive "
                f"a wrong equation")


def _slice_letter(idx):
    return 'o' if idx in OCC_LETTERS else 'v'


def _parse_factor(fac):
    """Return (array_name, index_list) for one tensor factor string.

    Restricted (spin-blocked) equations from pq.strings(spin_labels=...)
    suffix integral factors with a trailing spin label, e.g.
    '<a,b||i,j>_abab' -> name 'g_abab'; amplitude/Fock/identity leaves
    already carry the suffix in their own name (e.g. 't2_abab(a,b,i,j)',
    'd_aa(m,n)'), which the widened name regex below picks up directly with
    no special-casing -- they fall through _array_expr's generic (unsliced)
    branch, which is exactly right since a restricted array is already
    shaped to its own occ/vir dimensions.
    """
    fac = fac.strip()
    if fac.startswith('<'):
        m = re.match(r'^<(.*)>(_[ab]+)?$', fac)
        inner, spin_suffix = m.group(1), (m.group(2) or '')
        bra, ket = inner.split('||')
        idxs = bra.split(',') + ket.split(',')
        return 'g' + spin_suffix, idxs
    if fac.startswith('d('):
        idxs = fac[2:-1].split(',')
        return 'kd', idxs
    m = re.match(r'^([a-zA-Z0-9_]+)\((.*)\)$', fac)
    name, idxstr = m.group(1), m.group(2)
    idxs = idxstr.split(',')
    return name, idxs


def _array_expr(name, idxs):
    """g/kd/f (and their restricted spin-blocked counterparts g_aaaa, f_aa,
    d_aa, ...) span both occ and vir in a single full-sized array and need
    per-occurrence o/v slicing by index role -- unlike amplitude leaves
    (t1_aa, t2_abab, l2_aaaa, ...), which are already shaped to their own
    fixed vir/occ axes and must NOT be sliced. Strip a trailing spin suffix
    (_aa, _abab, _aaaa, ...) before the g/kd/f/d membership check so this
    dispatch is spin-suffix-agnostic; harmless no-op for the non-restricted
    (unsuffixed) pipeline.

    B (DF/RI 3-index factor, plain or spin-blocked B_aa/B_bb -- see
    src/CodeGeneration/SingleReference/df_codegen.py::dress_integral_factor)
    is norb/nv-scale on its LAST two axes only; its first axis is the
    auxiliary (naux) index and must never be o/v-sliced -- unlike g/kd/f/d,
    which are norb/nv-scale on every axis.
    """
    base = re.sub(r'_[ab]+$', '', name)
    if base == 'B':
        aux_idx, *rest = idxs
        sl_rest = ', '.join(_slice_letter(i) for i in rest)
        return f"{name}[:, {sl_rest}]"
    sl = ', '.join(_slice_letter(i) for i in idxs)
    if base in ('g', 'kd', 'f', 'd'):
        return f"{name}[{sl}]"
    return name


# ----------------------------------------------------------------------
# Index-label -> einsum-subscript remap (layer 3 of the general labelling).
#
# pdaggerq accepts UNBOUNDED numbered index labels (i1, i2, ..., a1, a2, ...;
# verified: 'i' and 'i1' are distinct, classification is by LEADING letter,
# and pq.strings(spin_labels=...) works with them). That removes the 6-letter
# budget ('ijklmn'/'abcdef').
#
# But numpy einsum subscripts are SINGLE CHARACTERS, and this emitter builds
# them by joining labels verbatim -- so 'i7' would contribute TWO subscript
# characters and silently produce a wrong contraction. Hence every label is
# mapped to one character here.
#
# The map PRESERVES the occ/vir family, because _slice_letter derives the o/v
# array slice FROM the label; remapping an occupied label onto a virtual
# character would mis-slice every array with no error.
#
# It also PREFERS the identity: a label that is already a legal single
# character of its own family maps to itself. The remap is unconditional, but
# it is therefore a no-op on everything generated to date -- which makes
# "regenerate and diff" a real regression check rather than a rewrite.
#
# CEILING: 24 occupied / 22 virtual distinct indices per einsum term. Ample for
# any realistic order, but it is a genuine bound -- the unbounded part is the
# pdaggerq layer, not this one.
# ----------------------------------------------------------------------

_OCC_SUBSCRIPTS = 'ijklmnopqrst' + 'IJKLMNOPQRST'
_VIR_SUBSCRIPTS = 'abcdefghuvwxyz' + 'ABCDEFGH'


def _subscript_map(labels):
    """Ordered labels -> {label: single subscript char}, family-preserving.

    `labels` must be ordered with the FREE (output) indices first so their
    assignment is stable across every term of a call -- the output subscript
    is shared, so an unstable free map would silently transpose results.
    """
    out, used = {}, set()
    for lbl in labels:
        if lbl in out:
            continue
        occ = lbl in OCC_LETTERS
        pool = _OCC_SUBSCRIPTS if occ else _VIR_SUBSCRIPTS
        if len(lbl) == 1 and lbl in pool and lbl not in used:
            out[lbl] = lbl                      # identity where already legal
        else:
            for c in pool:
                if c not in used:
                    out[lbl] = c
                    break
            else:
                raise RuntimeError(
                    f"einsum subscript pool exhausted for the "
                    f"{'occupied' if occ else 'virtual'} family "
                    f"({len(pool)} chars) on label {lbl!r}; term needs more "
                    "distinct indices of one family than einsum can express")
        used.add(out[lbl])
    return out


def _term_labels(factors, perms, out_indices):
    """All index labels of one term, FREE ONES FIRST (see _subscript_map)."""
    ordered = list(out_indices)
    for a, b in perms:
        ordered.extend((a, b))
    for _, idxs in factors:
        ordered.extend(idxs)
    return ordered


def _canonicalize_contraction(array_exprs, subs, out_indices):
    """CSE key for one term's contraction, invariant under (a) operand
    reordering (einsum's input list is commutative) and (b) which literal
    dummy-index letters pdaggerq happened to assign (contracted/summed
    indices are arbitrary bookkeeping -- alpha-renaming them changes
    nothing, same as renaming a summation variable in ordinary algebra).
    Free indices (those in out_indices) are NEVER renamed: their identity is
    load-bearing, tied to the shared output subscript every term in this
    call must match. Both properties mean two terms sharing a canonical key
    compute the byte-identical _tmp value -- exact CSE, not a heuristic.
    """
    free = set(out_indices)
    operands = sorted(zip(array_exprs, subs))
    rename = {}
    counter = [0]

    def canon_idx(c):
        if c in free:
            return c
        if c not in rename:
            rename[c] = f"_d{counter[0]}"
            counter[0] += 1
        return rename[c]

    return tuple((expr, ''.join(canon_idx(c) for c in sub)) for expr, sub in operands)


def raw_terms_to_einsum(raw_terms, out_var, out_indices, accumulator=None):
    """Convert raw pq.strings() output into einsum source lines, honoring literal P(x,y) semantics (gotcha 2).

    Includes exact (not heuristic) common-subexpression elimination via
    _canonicalize_contraction: reusing an earlier _tmp instead of
    recomputing an identical contraction is provably exact, not an
    approximation. Confirmed a real, non-trivial win by scanning the
    generated restricted-CCSDT files: ~11% of all contractions were
    duplicates (up to dummy-index relabeling) within their own function,
    concentrated in the most expensive (T3) equations.

    accumulator: how the P(x,y) permutation group of each term is added into
    out_var. None (default) emits the plain

        out_var += 1 * _tmp0
        out_var -= 1 * _tmp0.transpose(0, 2, 1)

    form; 'pacc' emits a single

        pacc(out_var, ((1, _tmp0), (-1, _tmp0.transpose(0, 2, 1)),))

    call instead (see src/SingleReference/CC/parallel_accum.py). The two are
    numerically equivalent up to summation order, but the plain form is
    single-threaded and materializes a full-size temporary per permutation --
    for the rank-5/6 EOM-CCSDT sigma blocks that dominated wall time and was
    completely insensitive to MKL_NUM_THREADS, since numpy elementwise ufuncs
    never touch BLAS. Opt in per generator (rather than switching the default)
    so every other generator's output stays byte-identical and keeps working
    without importing pacc.
    """
    if accumulator not in (None, 'pacc'):
        raise ValueError(f"unknown accumulator {accumulator!r}")
    lines = []
    tmp_counter = [0]
    seen_contractions = {}

    for term in raw_terms:
        factor = exact_coefficient(term[0])
        rest = term[1:]

        perms = []
        factors = []
        for fac in rest:
            fac = fac.strip()
            pm = re.match(r'^P\((.*)\)$', fac)
            if pm:
                a, b = pm.group(1).split(',')
                perms.append((a, b))
                continue
            factors.append(_parse_factor(fac))

        # Map every label of this term to ONE einsum subscript character.
        # Free indices lead, so their assignment is identical in every term
        # (the output subscript is shared across the whole call).
        smap = _subscript_map(_term_labels(factors, perms, out_indices))
        factors = [(name, [smap[i] for i in idxs]) for name, idxs in factors]
        perms = [(smap[a], smap[b]) for a, b in perms]
        term_out = [smap[i] for i in out_indices]

        subs = []
        array_exprs = []
        for name, idxs in factors:
            subs.append(''.join(idxs))
            array_exprs.append(_array_expr(name, idxs))

        base_out_sub = ''.join(term_out)
        einsum_sig = ','.join(subs) + '->' + base_out_sub

        cse_key = _canonicalize_contraction(array_exprs, subs, term_out)
        if cse_key in seen_contractions:
            tmp_name = seen_contractions[cse_key]
        else:
            tmp_name = f"_tmp{tmp_counter[0]}"
            tmp_counter[0] += 1
            opt = ", optimize=True" if len(array_exprs) > 1 else ""
            lines.append(f"    {tmp_name} = einsum('{einsum_sig}', {', '.join(array_exprs)}{opt})")
            seen_contractions[cse_key] = tmp_name

        k = len(perms)
        pacc_terms = []
        for toggles in product([False, True], repeat=k):
            sign = factor
            active_swaps = [perms[i] for i, t in enumerate(toggles) if t]
            for _ in active_swaps:
                sign = -sign

            axis_order = list(term_out)
            for a, b in active_swaps:
                ia, ib = axis_order.index(a), axis_order.index(b)
                axis_order[ia], axis_order[ib] = axis_order[ib], axis_order[ia]
            perm_positions = tuple(term_out.index(lbl) for lbl in axis_order)

            s = '+' if sign >= 0 else '-'
            if perm_positions == tuple(range(len(term_out))):
                operand = tmp_name
            else:
                operand = f"{tmp_name}.transpose{perm_positions}"

            if accumulator == 'pacc':
                pacc_terms.append(f"({'-' if s == '-' else ''}{abs(sign):.15g}, {operand})")
            else:
                lines.append(f"    {out_var} {s}= {abs(sign):.15g} * {operand}")

        if accumulator == 'pacc' and pacc_terms:
            lines.append(f"    pacc({out_var}, ({', '.join(pacc_terms)},))")
    return lines


def raw_terms_to_einsum_freeing(raw_terms, out_var, out_indices):
    """Append-only low-memory variant of raw_terms_to_einsum: byte-for-byte the
    same einsum/accumulate lines, PLUS a `del _tmpN` inserted right after each
    temporary's LAST reference, so the (often large -- e.g. nv^3*no^3 for a T3
    numerator) contraction temporaries are freed as soon as they are fully
    consumed instead of all remaining resident until the function returns
    (Python keeps every distinct `_tmpN` name alive otherwise). Pure memory
    optimization: the numerical result is bit-identical (`del` only drops a
    Python reference). CSE-safe -- because raw_terms_to_einsum reuses a temp for
    duplicate contractions across terms, the `del` is placed after the temp's
    last use across ALL emitted lines, never before a later reuse."""
    lines = raw_terms_to_einsum(raw_terms, out_var, out_indices)
    last = {}
    for i, ln in enumerate(lines):
        for m in re.finditer(r'_tmp\d+', ln):
            last[m.group(0)] = i
    for i in sorted(set(last.values()), reverse=True):
        for t in sorted([t for t, li in last.items() if li == i], reverse=True):
            lines.insert(i + 1, f"    del {t}")
    return lines


def raw_terms_to_sigma_einsum(raw_terms, out_var, bra_indices, ket_indices,
                              vec_var='v'):
    """Sigma-vector (matrix-free) twin of raw_terms_to_einsum: instead of
    building the dense (bra x ket) block, emit einsum lines that CONTRACT
    each term against a trial vector `vec_var` (carrying the ket-index
    subscripts), accumulating

        y[bra] = sum_ket  block[bra, ket] * v[ket]

    directly, with no (bra x ket) tensor ever allocated -- the largest
    object is the (bra,)-shaped output. This is the emission mode that lets
    ADC(5)'s 3h2p x 3h2p self-coupling be generated at all (its dense form
    is (n3h2p)^2, unmaterializable at production scale).

    The trial vector is appended as an extra einsum operand; einsum sums the
    ket indices automatically since they are absent from the (bra-only)
    output. pdaggerq P(x,y) permutation tokens (gotcha 2) are honored
    WITHOUT materializing any block:
      - a swap of two BRA letters -> swap those letters in the OUTPUT
        subscript (an axis relabel that IS the transpose of the result);
      - a swap of two KET letters -> swap those letters in the trial
        vector's subscript (contracting the transposed block against v is
        identical to contracting the block against the index-swapped v --
        provable by relabeling the summed ket dummies).
    A MIXED bra<->ket swap has no such factorization; it cannot occur for
    class-class blocks (the bra and ket antisymmetrizers act on disjoint
    letter sets -- see precursor_classes.class_letters' offsets) and raises
    if ever seen, rather than emitting a silently-wrong contraction.

    bra_indices/ket_indices are the ordered free-letter lists (the same
    split assemble_block passes as out_indices = bra + ket); vec_var's axes
    are ket_indices in that order.
    """
    bra_set, ket_set = set(bra_indices), set(ket_indices)
    bra_sub0 = list(bra_indices)
    ket_sub0 = list(ket_indices)
    lines = []

    for term in raw_terms:
        factor = exact_coefficient(term[0])
        rest = term[1:]

        perms = []
        factors = []
        for fac in rest:
            fac = fac.strip()
            pm = re.match(r'^P\((.*)\)$', fac)
            if pm:
                a, b = pm.group(1).split(',')
                perms.append((a.strip(), b.strip()))
                continue
            factors.append(_parse_factor(fac))

        subs = [''.join(idxs) for _, idxs in factors]
        array_exprs = [_array_expr(name, idxs) for name, idxs in factors]

        for toggles in product([False, True], repeat=len(perms)):
            sign = factor
            bra_sub = list(bra_sub0)
            ket_sub = list(ket_sub0)
            for i, t in enumerate(toggles):
                if not t:
                    continue
                sign = -sign
                a, b = perms[i]
                if a in bra_set and b in bra_set:
                    ia, ib = bra_sub.index(a), bra_sub.index(b)
                    bra_sub[ia], bra_sub[ib] = bra_sub[ib], bra_sub[ia]
                elif a in ket_set and b in ket_set:
                    ia, ib = ket_sub.index(a), ket_sub.index(b)
                    ket_sub[ia], ket_sub[ib] = ket_sub[ib], ket_sub[ia]
                else:
                    raise ValueError(
                        f"mixed bra<->ket permutation P({a},{b}) has no "
                        "sigma-vector factorization (should not occur for "
                        "class-class blocks -- bra/ket letters are disjoint)")
            einsum_sig = ','.join(subs + [''.join(ket_sub)]) + '->' + ''.join(bra_sub)
            operands = ', '.join(array_exprs + [vec_var])
            s = '+' if sign >= 0 else '-'
            lines.append(
                f"    {out_var} {s}= {abs(sign):.15g} * "
                f"einsum('{einsum_sig}', {operands}, optimize=True)")
    return lines


def infer_free_indices(raw_terms, type_order=('vir', 'occ')):
    """Infer an equation's free/output index letters and axis order from its raw pq.strings() output.

    An index appearing exactly once in a term is free; any term is
    representative since pdaggerq keeps free-index labels consistent across
    an equation -- but a term can fail to mention every free index in a
    parenthesized factor at all (e.g. a bare bra-ket integral term with no
    amplitude leaf, like a Lambda residual's '<m,n||e,f>' zeroth-order term:
    '<...>' factors don't end in ')' and are skipped, same as the
    already-tolerated bra-ket gap noted in _parse_factor's docstring), so
    scan forward until a term actually yields something instead of trusting
    raw_terms[0] blindly.
    """
    for term in raw_terms:
        counts = {}
        order = []
        for fac in term[1:]:
            fac = fac.strip()
            if fac.startswith('P('):
                continue
            m = re.search(r'\((.*?)\)$', fac)
            if not m:
                continue
            for idx in m.group(1).split(','):
                idx = idx.strip()
                if idx not in counts:
                    counts[idx] = 0
                    order.append(idx)
                counts[idx] += 1
        free = [idx for idx in order if counts[idx] == 1]
        if free:
            groups = {'vir': [], 'occ': []}
            for idx in free:
                groups['vir' if idx in VIR_LETTERS else 'occ'].append(idx)
            return [idx for t in type_order for idx in groups[t]]
    return []


def infer_output_order(raw_terms, self_leaf_name, type_order=('vir', 'occ')):
    """Like infer_free_indices, but anchors the axis order to how
    `self_leaf_name` (e.g. 'r2') is itself referenced as a plain leaf
    (no extra summed indices beyond the free ones) somewhere in these terms.

    This matters whenever an equation's output IS the same amplitude that
    appears as one of its own inputs on the next iteration (e.g. r2's sigma
    equation, fed back as r2 for the next Davidson step): the equation's own
    free-index labeling comes from the bra/ket PROJECTOR operator, which has
    no reason to align with the amplitude's own canonical slot order (which
    axis is "first" for an antisymmetric pair is a real, fixed convention,
    not an arbitrary relabeling, since terms elsewhere reference e.g. r2 with
    a specific literal axis order). Picking the naive first-occurrence order
    (plain infer_free_indices) silently produces a self-inconsistent sigma
    function -- confirmed the hard way: it gave a wrong, badly-off IP-CCSD
    eigenvalue (dense-diagonalized and compared against the determinant_space
    oracle to isolate it to exactly this mismatch) despite every individual
    equation looking like a perfectly ordinary generated contraction.

    Falls back to infer_free_indices if no such self-referencing leaf term
    is found (e.g. rank-1 outputs, where there is no axis-order ambiguity).
    """
    free = infer_free_indices(raw_terms, type_order=type_order)
    free_set = set(free)
    prefix = f'{self_leaf_name}('
    for term in raw_terms:
        for fac in term[1:]:
            fac = fac.strip()
            if not fac.startswith(prefix) or not fac.endswith(')'):
                continue
            idxs = [x.strip() for x in fac[len(prefix):-1].split(',')]
            if len(idxs) == len(free) and set(idxs) == free_set:
                groups = {'vir': [], 'occ': []}
                for idx in idxs:
                    groups['vir' if idx in VIR_LETTERS else 'occ'].append(idx)
                return [idx for t in type_order for idx in groups[t]]
    return free


def assemble_function(func_name, args, out_var, out_shape_expr, raw_terms, out_indices, shape_source='t1',
                      accumulator=None):
    """Wrap raw_terms_to_einsum output into a complete python function
    definition (string), in the exact shape convention the generated files in
    src/SingleReference/CC/ follow.

    shape_source: name of a (nv, no)-shaped arg to read nv/no from -- 't1' for
    the generalized spin-orbital pipeline's convention, 't1_aa' for restricted
    (spin-blocked) generated functions whose amplitude args are suffixed.
    accumulator: passed through to raw_terms_to_einsum (see there); 'pacc'
    requires the emitted file to import pacc from parallel_accum.
    """
    src = []
    src.append(f"def {func_name}({', '.join(args)}):")
    src.append(f"    nv, no = {shape_source}.shape")
    src.append(f"    {out_var} = np.zeros({out_shape_expr})")
    src.extend(raw_terms_to_einsum(raw_terms, out_var, out_indices, accumulator=accumulator))
    src.append(f"    return {out_var}")
    src.append("")
    return "\n".join(src)
