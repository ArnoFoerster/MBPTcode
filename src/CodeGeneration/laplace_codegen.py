"""Generic (rank-agnostic) Laplace-quadrature term-fusion code generator.

Generalizes the hand-derivation in mpn_density_driver_restricted.py's
_laplace_aaaaaa_contribution (itself ported from density_matrix.py's
_laplace_aaa_contribution) into an automatic transformation, so the same
trick applies to doubles now and to quadruples/quintuples later without
re-deriving anything by hand per rank/equation.

Physics: for a rank-R amplitude T_R with numerator N_R (i.e. T_R = N_R / D_R,
D_R = sum(eps_occ) - sum(eps_vir) over T_R's own R virtual + R occupied
external indices), any equation that consumes T_R as one factor can be
rewritten using
    1/D_R = -sum_tau sigma_tau * exp(D_R * tau)   (minimax quadrature)
    exp(D_R * tau) = prod_{occ i in T_R} exp(eps_i*tau) * prod_{vir a in T_R} exp(-eps_a*tau)
so that T_R's numerator terms can be dressed (per-orbital exponential
factors on T_R's own external legs) and fused DIRECTLY into the consumer's
contraction, one tau at a time -- T_R itself is never materialized.

Algorithm (fuse_amplitude_into_consumer): for every raw term of the consumer
equation that references the amplitude tag (e.g. 't3_aaaaaa') as one of its
factors, and every raw term of the amplitude's own numerator (pre-division),
substitute the amplitude factor's local index list into the numerator term
positionally (amp_out_indices(R)'s canonical order -> the consumer's local
letters), rename every OTHER (dummy) index in the numerator term to a fresh
letter (avoiding collision with anything already used in the consumer term),
and append one Oe(...)/Ve(...) dressing factor per amplitude external index.
The result is a new raw-term list in the exact string format
pdaggerq.pq_helper.strings() produces, so it feeds unmodified into
src/CodeGeneration/pdaggerq_interface.py's existing raw_terms_to_einsum -- no new codegen
logic needed there, just new *terms* to feed it, one set per tau iteration.

Validated when written (reproduces mpn_density_driver_restricted.py's
hand-derived _laplace_aaaaaa_contribution to machine precision when fusing
t3_2_aaaaaa's own numerator into t1_3_aa_numerator's and
m3_ov_12_restricted's raw terms) -- see tests/test_laplace_codegen_generic.py.
"""
import re
from itertools import product

from src.CodeGeneration.pdaggerq_interface import exact_coefficient
from src.CodeGeneration.generate_mpn_density import amp_out_indices
from src.CodeGeneration.codegen_common import (
    OCC_FAMILY as OCC_LETTERS, OCC_SUBSCRIPTS, VIR_SUBSCRIPTS, parse_factor, array_expr,
    term_has_factor)
from src.CodeGeneration.generate_mpn_density_spin import spinify_slices

# Fresh-letter pools for renaming an amplitude numerator's own dummy indices
# during fusion -- disjoint from PDAGGERQ_SAFE_OCC/VIR ('i'-'n'/'a'-'f') on
# purpose, since these letters never go back into pdaggerq (they're only
# einsum subscript characters in the generated Python), so there's no reason
# to stay inside pdaggerq's safe-letter budget here.
def _fresh_pool(preferred, full):
    """`preferred` first, then every remaining subscript of the same family.

    The historic pools were deliberately disjoint from PDAGGERQ_SAFE_OCC/VIR
    ('i'-'n'/'a'-'f') as belt-and-braces, capping fusion at 12 occ / 16 vir
    fresh letters and raising "ran out of fresh letters" beyond that -- a
    fourth index wall, alongside the three removed in pdaggerq_interface /
    precursor_classes. Disjointness was never the actual safety mechanism:
    `_fresh_letter` skips anything in `used`, and `used` is seeded with the
    outer term's letters, so a collision cannot occur regardless of pool.

    Appending (rather than reordering) keeps the preferred letters chosen
    first, so everything generated to date regenerates byte-identically while
    the pool widens to the full 24 occ / 22 vir einsum-subscript budget.
    """
    return list(preferred) + [c for c in full if c not in set(preferred)]


_FRESH_OCC_POOL = _fresh_pool('opqrstOPQRST', OCC_SUBSCRIPTS)
_FRESH_VIR_POOL = _fresh_pool('ghuvwxyzGHUVWXYZ', VIR_SUBSCRIPTS)


def lambda_canonical_and_sign(rank):
    """Occ-first external-index orders + signs for fusing a lambda (l{rank})
    amplitude's numerator, one entry per (same-spin, mixed-spin) case, matching
    mpn_density_driver_restricted._to_l_restricted's signed transpose
    l = sign * t.transpose(axes). RANK-GENERAL -- the same rule gives l2, l3,
    l4 (the O(N^8) MP4/ADC5 denominators), l5, ... so a higher-order U-coupling
    that consumes l4/l5 fuses through fuse_amplitude_into_consumer with exactly
    these (canonical, sign) pairs; nothing here is hard-coded to rank 3.

    Returns {'same': (canonical, sign), 'mixed': (canonical, sign)} where
    `canonical` is passed as fuse_amplitude_into_consumer(..., canonical=...)
    and `sign` multiplies the numerator term coefficients. Pick 'same' when the
    l-block's spin string is uniform (aa..a / bb..b), else 'mixed'."""
    ext = amp_out_indices(rank)                       # t{rank} legs: vir-first
    same_axes = list(range(rank, 2 * rank)) + list(range(rank - 1, -1, -1))
    mixed_axes = list(range(rank, 2 * rank)) + list(range(rank))
    return {
        'same': ([ext[a] for a in same_axes], (-1.0) ** (rank // 2)),
        'mixed': ([ext[a] for a in mixed_axes], 1.0),
    }


def _split_term(term):
    coef = exact_coefficient(term[0])
    perms = []
    factors = []
    for tok in term[1:]:
        tok = tok.strip()
        m = re.match(r'^P\((.*)\)$', tok)
        if m:
            x, y = m.group(1).split(',')
            perms.append((x.strip(), y.strip()))
        else:
            factors.append(tok)
    return coef, perms, factors


def _factor_name_and_idxs(fac):
    """Return (kind, idxs) where kind is enough to rebuild the exact same
    factor string later with renamed indices -- mirrors
    pdaggerq_interface.parse_factor's three cases (bracket integral,
    kronecker delta, generic paren leaf)."""
    fac = fac.strip()
    if fac.startswith('<'):
        m = re.match(r'^<(.*)>(_[ab]+)?$', fac)
        inner, suffix = m.group(1), (m.group(2) or '')
        bra, ket = inner.split('||')
        bra_idxs, ket_idxs = bra.split(','), ket.split(',')
        return ('bracket', suffix, len(bra_idxs)), bra_idxs + ket_idxs
    if fac.startswith('d('):
        idxs = fac[2:-1].split(',')
        return ('d',), idxs
    m = re.match(r'^([a-zA-Z0-9_]+)\((.*)\)$', fac)
    name, idxstr = m.group(1), m.group(2)
    return ('paren', name), idxstr.split(',')


def _rebuild_factor(kind, idxs):
    if kind[0] == 'bracket':
        _, suffix, bra_len = kind
        bra, ket = idxs[:bra_len], idxs[bra_len:]
        return f"<{','.join(bra)}||{','.join(ket)}>{suffix}"
    if kind[0] == 'd':
        return f"d({','.join(idxs)})"
    _, name = kind
    return f"{name}({','.join(idxs)})"


def _swap_letters(parsed_factors, x, y):
    def sw(idxs):
        return [y if c == x else (x if c == y else c) for c in idxs]
    return [(kind, sw(idxs)) for kind, idxs in parsed_factors]


def _expand_perms(coef, perms, parsed_factors):
    """Resolve P(x,y) tokens into their explicit sign-flipped, letter-swapped
    copies BEFORE any fusion/renaming happens -- necessary because after
    fusion some of a P(...) pair's two letters can end up playing different
    roles (one becomes a free/output index, the other a contracted dummy),
    which raw_terms_to_einsum's post-hoc P(...) handling (a pure output-axis
    transpose) cannot express. Swapping symbolically on the STILL-lettered
    term is well-defined regardless of what those letters later become.
    Multiple P(...) tokens compose independently (pdaggerq_interface.py's
    own documented convention), so each toggle in the product below applies
    (or doesn't) its own pair's swap independently.
    """
    results = []
    for toggles in product([False, True], repeat=len(perms)):
        cur = parsed_factors
        sign = coef
        for i, active in enumerate(toggles):
            if active:
                x, y = perms[i]
                cur = _swap_letters(cur, x, y)
                sign = -sign
        results.append((sign, cur))
    return results


def _fresh_letter(used, is_occ):
    pool = _FRESH_OCC_POOL if is_occ else _FRESH_VIR_POOL
    for c in pool:
        if c not in used:
            return c
    raise RuntimeError("ran out of fresh letters for Laplace-fusion renaming "
                       "(need a bigger _FRESH_OCC_POOL/_FRESH_VIR_POOL)")


def fuse_amplitude_into_consumer(inner_terms, inner_rank, outer_terms, amp_tag,
                                 canonical=None, dress_names=('Oe', 'Ve')):
    """See module docstring. `amp_tag` is the amplitude's exact factor name
    as it appears in `outer_terms` (e.g. 't3_aaaaaa' for a spin-labeled
    restricted equation, or plain 't3' for the spin-orbital pipeline).

    Returns a new raw-term list (pdaggerq string format) representing the
    fused, denominator-NOT-yet-divided contraction: each returned term
    additionally carries one Oe(<letter>)/Ve(<letter>) factor per one of the
    amplitude's own 2*inner_rank external indices, meant to be evaluated
    fresh each tau iteration by assemble_laplace_fused.

    `canonical` overrides the amplitude's external-index order used to align
    `inner_terms`' external legs with the outer factor's local indices
    (positional zip). Defaults to amp_out_indices(inner_rank) -- the
    vir-first storage order of a `t{rank}` factor. Supply an explicit order
    when the outer factor is a DIFFERENTLY-laid-out relabel of the same
    numerator, e.g. a lambda `l{rank}` factor (occ-first, per
    mpn_density_driver_restricted._to_l_restricted's transpose): then
    `inner_terms` must be the numerator whose external legs, listed in THIS
    order, correspond position-by-position to the outer l-factor's indices
    (apply the _to_l sign to the term coefficients yourself before calling).
    """
    if canonical is None:
        canonical = amp_out_indices(inner_rank)
    canonical_set = set(canonical)
    fused = []

    for outer_term in outer_terms:
        coef_o0, perms_o, factor_strs_o0 = _split_term(outer_term)
        parsed_o0 = [_factor_name_and_idxs(f) for f in factor_strs_o0]

        # P(...) resolved into explicit copies FIRST (see _expand_perms'
        # docstring for why this must happen before any renaming).
        for coef_o, parsed_o in _expand_perms(coef_o0, perms_o, parsed_o0):
            outer_letters = set()
            for _, idxs in parsed_o:
                outer_letters.update(idxs)

            for occ_i, (kind_o, idxs_o) in enumerate(parsed_o):
                if kind_o[0] != 'paren' or kind_o[1] != amp_tag:
                    continue
                local_idx = idxs_o
                rename_ext = dict(zip(canonical, local_idx))
                remaining_parsed = [parsed_o[j] for j in range(len(parsed_o)) if j != occ_i]

                for inner_term in inner_terms:
                    coef_i0, perms_i, factor_strs_i0 = _split_term(inner_term)
                    parsed_i0 = [_factor_name_and_idxs(f) for f in factor_strs_i0]

                    for coef_i, parsed_i in _expand_perms(coef_i0, perms_i, parsed_i0):
                        all_letters_i = set()
                        for _, idxs in parsed_i:
                            all_letters_i.update(idxs)
                        dummy_i = all_letters_i - canonical_set

                        used = set(outer_letters)
                        rename = dict(rename_ext)
                        # sorted(): `dummy_i` is a SET, and iterating a set of
                        # strings has process-dependent order (str hashing is
                        # randomized), which made the fresh-letter assignment --
                        # and hence the emitted einsum dummy indices --
                        # NON-REPRODUCIBLE between generation runs (semantically
                        # identical, but byte-different: 'g,o' vs 'o,p'). Sorting
                        # makes the codegen deterministic/auditable. Most visible
                        # on the DF path, which carries extra aux (Q) dummies.
                        for letter in sorted(dummy_i):
                            fresh = _fresh_letter(used, letter in OCC_LETTERS)
                            used.add(fresh)
                            rename[letter] = fresh

                        renamed_factors = [(kind, [rename[c] for c in idxs]) for kind, idxs in parsed_i]

                        dress_factors = [(('paren', dress_names[0]), [letter]) if letter in OCC_LETTERS
                                         else (('paren', dress_names[1]), [letter]) for letter in local_idx]

                        combined_coef = coef_o * coef_i
                        combined_parsed = remaining_parsed + renamed_factors + dress_factors
                        fused.append([f"{combined_coef:+.15g}"]
                                    + [_rebuild_factor(kind, idxs) for kind, idxs in combined_parsed])

    return fused


def _is_dressing_name(name):
    """Oe/Ve (restricted), their per-spin Oe_a/Oe_b/Ve_a/Ve_b variants
    (unrestricted codegen), or the per-tau-slot Oe2/Ve2/... variants that
    nested two-amplitude fusion uses (each independently-denominatored
    amplitude gets its own tau; see assemble_laplace_fused nslots>1)."""
    base = name[:2]
    return (name in ('Oe', 'Ve') or name.startswith('Oe_') or name.startswith('Ve_')
            or (base in ('Oe', 'Ve') and name[2:].isdigit()))


def _collect_dressed_term_specs(raw_terms):
    """Shared front end for the flatten-p emitters: parse each fused term into
    (coef, real-operand exprs/subs, E-factor (evar, subscript) list), register
    the CSE order of the bare (UNDRESSED) real-operand expressions, and register
    the hoisted combined-E products (one per axis dressed by >1 kind/slot).

    Returns (term_specs, real_expr_order, combo_order) where each term_spec is
    (coef, real_exprs, real_subs, e_ops) and combo_order is a list of
    (evar, sorted-dname-tuple) to emit once as `evar = d0 * d1 * ...`."""
    real_expr_order, real_seen = [], set()
    combo_keys, combo_order = {}, []
    term_specs = []
    for term in raw_terms:
        coef = exact_coefficient(term[0])
        factors = [parse_factor(f) for f in term[1:]]
        real, dressings = [], []
        for name, idxs in factors:
            if _is_dressing_name(name):
                dressings.append((name, idxs[0]))
            else:
                real.append([name, idxs])

        idx_owner = {}
        for pos, (name, idxs) in enumerate(real):
            for letter in idxs:
                idx_owner.setdefault(letter, pos)

        dress_map = {}
        for dname, letter in dressings:
            rpos = idx_owner[letter]
            axis_pos = real[rpos][1].index(letter)
            # a list, not a scalar: an index shared by two nested amplitudes
            # (different tau slots) carries BOTH Oe/Ve and Oe2/Ve2 -- a scalar
            # slot would silently overwrite one, dropping a denominator factor.
            dress_map.setdefault(rpos, {}).setdefault(axis_pos, []).append(dname)

        real_exprs, real_subs = [], []
        for pos, (name, idxs) in enumerate(real):
            expr = array_expr(name, idxs)          # UNDRESSED (dressing -> E ops)
            real_exprs.append(expr)
            real_subs.append(''.join(idxs))
            if expr not in real_seen:
                real_seen.add(expr)
                real_expr_order.append(expr)

        e_ops = []          # (evar, <axis letter>); the batch label is prefixed
        for rpos in sorted(dress_map):              # at emit time (see _batch_label)
            for axis_pos in sorted(dress_map[rpos]):
                dnames = dress_map[rpos][axis_pos]
                letter = real[rpos][1][axis_pos]
                key = tuple(sorted(dnames))
                if len(key) == 1:
                    evar = key[0]                   # reference the (P,orb) full directly
                else:
                    evar = combo_keys.get(key)
                    if evar is None:                # hoist the multi-kind product once
                        evar = f"_Ep{len(combo_keys)}"
                        combo_keys[key] = evar
                        combo_order.append((evar, key))
                e_ops.append((evar, letter))

        term_specs.append((coef, real_exprs, real_subs, e_ops))
    return term_specs, real_expr_order, combo_order


# Einsum label for the flattened tau-quadrature batch axis p. It must never be
# an orbital index: pdaggerq_interface.OCC_SUBSCRIPTS/VIR_SUBSCRIPTS (and the
# _FRESH_*_POOLs built from them) draw orbital labels from a-z + A-T, so the
# uppercase letters U-Z are the only einsum characters guaranteed never to be an
# occ/vir index. (An early version used lowercase 'p' and collided with the
# occupied index 'p' -> "Size of label 'p' ... does not match" at einsum time.)
_BATCH_LABEL_POOL = 'ZYXWVU'


def _batch_label(term_specs, *reserved):
    """A single einsum char for the batch axis p, disjoint from every index
    letter used by `term_specs`' real operands and from `reserved` (the output/
    ket/bra letters). Prefers U-Z (never orbital labels); the scan is a
    belt-and-braces guard in case the subscript pools are ever widened."""
    used = set(reserved)
    for _, _, real_subs, _ in term_specs:
        for sub in real_subs:
            used.update(sub)
    for c in _BATCH_LABEL_POOL:
        if c not in used:
            return c
    raise RuntimeError("no free einsum label for the flatten-p batch axis "
                       "(widen _BATCH_LABEL_POOL)")


def _emit_dressed_hoists(real_expr_order, combo_order, term_specs):
    """CSE lines (bare real-operand slices used >1x) + combined-E product hoists,
    returning (lines, cse_var map)."""
    counts = {}
    for _, real_exprs, _, _ in term_specs:
        for e in real_exprs:
            counts[e] = counts.get(e, 0) + 1
    cse_var, lines = {}, []
    for e in real_expr_order:
        if counts[e] > 1:
            var = f"_cse{len(cse_var)}"
            cse_var[e] = var
            lines.append(f"    {var} = {e}")
    for evar, key in combo_order:
        lines.append(f"    {evar} = {' * '.join(key)}")
    return lines, cse_var


def raw_terms_to_einsum_dressed(raw_terms, out_var, out_indices, weight_var=None):
    """Flatten-p Laplace emission: ONE einsum per fused term that sums the ENTIRE
    (multi-)tau quadrature as a single contracted batch axis `p`, replacing the
    old python tau-loop (ntau**(#kinds) tiny einsum dispatches per term, plus a
    per-tau rebuild of the Oe(x)/Ve(x) dressing vectors).

    Each independently-denominatored amplitude ("kind"/slot) contributes a tau
    sub-grid; assemble_laplace_fused[/_multikind] flattens their outer product
    into one axis of length P = ntau**(#kinds) and defines, per kind k, the
    dressing MATRICES Oe_<k>/Ve_<k> of shape (P, no)/(P, nv) (row p = exp(+/-
    eps*tau) at grid point p), plus the flattened quadrature weight `weight_var`
    of shape (P,) (product of the slots' -sigma). This function then, for every
    amplitude-external axis of every operand, appends ONE E-factor operand with
    subscript 'p'+<axis letter> -- the elementwise product of the Oe_<k>/Ve_<k>
    of the kinds dressing that axis (a single-kind axis references its matrix
    directly; a shared axis dressed by >1 kind gets a hoisted `_EpN = Oe_a*Oe_b`,
    computed once) -- and one final `weight_var` operand (subscript 'p'). einsum
    contracts p (absent from the output), so

        out[free] = sum_p weight[p] * prod_axes E_axis[p, orb] * base_contraction

    is EXACTLY the tau sum the old loop accumulated, up to floating-point
    summation order (machine precision; tests/test_laplace_nested_fusion.py).
    This removes the ntau**(#kinds) python einsum dispatches per term -- the
    dominant cost at active-space norb, where the tiny contractions are pure
    interpreter/dispatch overhead -- AND the per-tau dressing rebuilds, at the
    price of ONE extra contracted axis of length P in the intermediates (a single
    controllable batch axis: greedy-path-safe and memory-bounded for the small
    active spaces these blocks run on, NOT the rank+#kinds blow-up a per-operand
    pre-multiply of both tau axes would cause).

    Real operands are left UNDRESSED (bare array_expr) so base tensors stay
    rank<=4; only the E-factors carry p. No P(...) handling (_expand_perms
    already resolved them). `weight_var=None` (a group with no dressed amplitude,
    e.g. the empty-kindset multikind group) emits plain undressed einsums, no p.
    """
    term_specs, real_expr_order, combo_order = _collect_dressed_term_specs(raw_terms)
    lines, cse_var = _emit_dressed_hoists(real_expr_order, combo_order, term_specs)
    out_sub = ''.join(out_indices)
    pL = _batch_label(term_specs, *out_indices)
    for coef, real_exprs, real_subs, e_ops in term_specs:
        ops = [cse_var.get(e, e) for e in real_exprs]
        subs = list(real_subs)
        for evar, letter in e_ops:
            ops.append(evar)
            subs.append(pL + letter)
        if weight_var is not None and e_ops:
            ops.append(weight_var)
            subs.append(pL)
        opt = ", optimize=True" if len(ops) > 1 else ""
        lines.append(f"    _tmp = einsum('{','.join(subs)}->{out_sub}', {', '.join(ops)}{opt})")
        s = '+' if coef >= 0 else '-'
        lines.append(f"    {out_var} {s}= {abs(coef):.15g} * _tmp")
    return lines


def assemble_laplace_fused(func_name, params, aliases, out_var, out_shape_expr,
                           fused_terms, out_indices, eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]',
                           ntau_default=6, nslots=1, unrestricted=False):
    """Wrap fuse_amplitude_into_consumer's output into a complete function that
    sums `ntau` minimax quadrature points, dressing the exp factors fresh each
    point and accumulating -sigma_tau * (fused contraction) into `out_var` --
    the amplitude's own denominator division never happens as a separate step;
    it's folded into this sum (see module docstring). `eps_o_expr`/`eps_v_expr`
    are Python expressions (in terms of `params`) for the occupied/virtual
    orbital-energy 1-D arrays.

    `nslots` > 1 handles a nested structure like l4ft3 = l4^(2)*f*t3^(2), a
    PRODUCT of two amplitudes with SEPARATE denominators D_l4, D_t3, so
    (N_l4/D_l4)(N_t3/D_t3) = int int N_l4 N_t3 e^{-D_l4 tau1} e^{-D_t3 tau2} --
    a 2-D quadrature, NOT 1/(D_l4+D_t3) from a single shared tau. Flatten-p sums
    that whole ntau^nslots quadrature as ONE contracted batch axis p (length
    ntau^nslots): the nslots tau sub-grids are flattened into p (mixed-radix via
    np.indices), slot 0's dressing MATRICES named Oe/Ve, slot s>=1's Oe{s+1}/
    Ve{s+1} (shape (P,no)/(P,nv)), and the flattened weight _wP (P,) = product of
    each slot's -sigma. raw_terms_to_einsum_dressed then emits ONE einsum per term
    (p summed) -- no python tau loop. A one-time (stored-coupling build) cost;
    base tensors stay <=4-index (only the p-carrying E-factors/intermediates add
    the single batch axis).

    unrestricted=True (UHF, one slot): per-spin dressings Oe_a/Oe_b/Ve_a/Ve_b
    from eps_a[o_a]/eps_b[o_b]/..., one shared grid wide enough for either spin
    (gap over the union of alpha and beta orbital energies: the quadrature
    identity holds for any D inside the grid's range, mixed-spin D3 included),
    and every emitted line passed through generate_mpn_density_spin.spinify_slices
    (the bracket-integral slices are spin-blind as emitted)."""
    src = [f"def {func_name}({', '.join(params)}, ntau={ntau_default}):"]
    for generic, real in aliases.items():
        if generic != real:
            src.append(f"    {generic} = {real}")
    if unrestricted:
        src.append("    ei_a, ea_a = eps_a[o_a], eps_a[v_a]")
        src.append("    ei_b, ea_b = eps_b[o_b], eps_b[v_b]")
        src.append("    gap_min = max(min(ea_a.min(), ea_b.min()) - max(ei_a.max(), ei_b.max()), 1e-3)")
        src.append("    gap_max = max(ea_a.max(), ea_b.max()) - min(ei_a.min(), ei_b.min())")
    else:
        src.append(f"    ei = {eps_o_expr}")
        src.append(f"    ea = {eps_v_expr}")
        src.append("    gap_min = max(ea.min() - ei.max(), 1e-3)")
        src.append("    gap_max = ea.max() - ei.min()")
    src.append("    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)")
    src.append(f"    {out_var} = np.zeros({out_shape_expr})")
    # flatten the nslots tau sub-grids into one axis p of length ntau**nslots
    grid_shape = ", ".join(["ntau"] * nslots)
    src.append(f"    _tk = np.indices(({grid_shape},)).reshape({nslots}, -1)")
    if unrestricted:
        for e, sgn, sub in (('Oe', '', 'ei'), ('Ve', '-', 'ea')):
            for spin in 'ab':
                src.append(f"    {e}_{spin} = np.exp({sgn}{sub}_{spin}[None, :] * tau[_tk[0]][:, None])")
    else:
        for s in range(nslots):
            oe = 'Oe' if s == 0 else f'Oe{s + 1}'
            ve = 'Ve' if s == 0 else f'Ve{s + 1}'
            src.append(f"    {oe} = np.exp(ei[None, :] * tau[_tk[{s}]][:, None])")
            src.append(f"    {ve} = np.exp(-ea[None, :] * tau[_tk[{s}]][:, None])")
    src.append("    _wP = " + " * ".join(f"(-sigma[_tk[{s}]])" for s in range(nslots)))
    for line in raw_terms_to_einsum_dressed(fused_terms, out_var, out_indices,
                                            weight_var='_wP'):
        src.append(spinify_slices(line) if unrestricted else line)
    src.append(f"    return {out_var}")
    src.append("")
    return "\n".join(src)


# ---------------------------------------------------------------------------
# NESTED fusion: an OUTER amplitude whose OWN numerator carries rank>=3 leaves
# (e.g. l3^(3), whose order-3 numerator has t3^(2)/t4^(2) leaves). Fusing it
# without materializing anything rank>=3 needs a fusion NESTED inside a fusion,
# and two things the flat (uniform-order) fuser never had to track. Resolved
# 2026-07-23 to machine precision on l3ft2/l3t2 (3,1).
# ---------------------------------------------------------------------------

_HR_TAG_RE = re.compile(r'^([lt])([3-9])_([ab]+)\(')       # rank>=3 l/t leaf
_LOWRANK_LEAF_RE = re.compile(r'^(t[12])_([ab]+)\(')       # bare t1_/t2_ leaf


def highrank_tags(terms):
    """{tag: rank} for every rank>=3 l/t amplitude factor in `terms`
    (e.g. {'t3_aaaaaa': 3, 't4_aabbaabb': 4})."""
    tags = {}
    for term in terms:
        for tok in term[1:]:
            m = _HR_TAG_RE.match(tok.strip())
            if m:
                tags[f'{m.group(1)}{m.group(2)}_{m.group(3)}'] = int(m.group(2))
    return tags


def tag_lowrank_leaves_by_order(terms, order):
    """Rewrite each MATERIALIZED low-rank (t1/t2) leaf `t{r}_{block}(` ->
    `t{r}_{order}_{block}(`, so leaves emitted at DIFFERENT nesting depths --
    hence different perturbation orders -- carry DISTINCT tags and can be fed
    the right-order amplitude data.

    This is the root-cause fix for the l3ft2/l3t2 (3,1) nested-fusion defect:
    l3^(3)'s numerator has a DIRECT t2 at order 2, while its inner t3^(2)/t4^(2)
    numerators have t2 at order 1; both otherwise print as the generic `t2_aaaa`
    and a single alias would silently feed one of them the wrong order (a
    same-shape array for no=nv, so no error -- just wrong numbers). Leaves that
    already carry an order digit (t2_1_aaaa, from a deeper level) and the
    separately-renamed outer ket dressing (t2out_*) are left untouched."""
    out = []
    for term in terms:
        new = [term[0]]
        for tok in term[1:]:
            tok = tok.strip()
            m = _LOWRANK_LEAF_RE.match(tok)
            if m:
                tok = f"{m.group(1)}_{order}_{m.group(2)}(" + tok[m.end():]
            new.append(tok)
        out.append(new)
    return out


def fuse_nested_numerator(numerator_fn, rank, order, block, _depth=0):
    """Fully-fused (T-frame) numerator of amplitude (rank, order, block).

    `numerator_fn(rank, order, block)` returns the raw numerator terms (e.g.
    generate_mpn_density_spin.t_numerator_terms_restricted). Every rank>=3
    leaf is RECURSIVELY Laplace-fused -- each distinct leaf rank gets its own
    kind-named dressing (Oe_t{r}/Ve_t{r}) so assemble_laplace_fused_multikind
    integrates one independent tau per kind (1/(D_outer * D_inner ...)). Every
    rank<=2 leaf is order-tagged (tag_lowrank_leaves_by_order) at its order.

    Inner leaves are ALWAYS fused in the pure T-frame (amp_out_indices, sign
    +1): t_numerator_terms_restricted returns T-frame numerators, so the leaf
    amplitudes are genuine t-amplitudes. The _to_l transpose+sign of an OUTER
    lambda consumer is applied ONLY when that consumer fuses the l-tag (by the
    caller, via fuse_amplitude_into_consumer(..., canonical=lambda_canonical_
    and_sign(...))); it then transposes the leaf's SHARED (external) legs while
    its internal V-contracted legs stay put -- the correct per-leg behaviour,
    achieved automatically because fuse-then-transpose commutes with the leg
    relabel. (Propagating the L-role INTO the inner leaf is wrong -- gives a
    structural error.)

    DEPTH (the two-same-rank-at-different-depths case): inner kinds are named
    by leaf RANK, which collides when a term carries two same-rank t-amplitudes
    from DIFFERENT nesting depths -- e.g. t1_4 -> t3_3 (rank 3) -> t3_2 (rank 3):
    the outer t3_3 and the inner t3_2 would both be dressed 'Oe_t3', so
    assemble_laplace_fused_multikind collapses them to ONE shared tau (WRONG --
    they have independent denominators). Fixed by DEPTH-QUALIFYING the kind name
    below the root: at recursion `_depth` d>0 the kind is 't{rk}_d{d}', so each
    amplitude instance in a term gets its own tau. `_depth == 0` keeps the plain
    't{rk}' name, so every existing single-nest caller (l3ft2/l3t2, l4ft2, the
    U^(4) blocks -- inner leaves fused at depth 0) is BYTE-UNCHANGED. Validated:
    l3ft2-31 (tests/test_laplace_nested_fusion.py) and t1_4 (tests/
    test_mp4_t1_4_laplace.py) both machine/ntau-precise."""
    num = numerator_fn(rank, order, block)
    num = tag_lowrank_leaves_by_order(num, order - 1)
    while True:
        tags = highrank_tags(num)
        if not tags:
            break
        tag, rk = sorted(tags.items())[0]
        blk = tag.split('_', 1)[1]
        matching = [t for t in num if term_has_factor(t, tag)]
        rest = [t for t in num if not term_has_factor(t, tag)]
        inner = fuse_nested_numerator(numerator_fn, rk, order - 1, blk, _depth + 1)
        kind = f't{rk}' if _depth == 0 else f't{rk}_d{_depth}'
        num = rest + fuse_amplitude_into_consumer(
            inner, rk, matching, tag, canonical=amp_out_indices(rk),
            dress_names=(f'Oe_{kind}', f'Ve_{kind}'))
    return num


def fuse_amplitude_rank3_tags(numerator_fn, rank, order, block):
    """SPLIT-fusion for BUILDING a Laplace AMPLITUDE (not a consumer block):
    {tag: (rk, fused_terms)} for each rank>=3 leaf tag in amplitude
    (rank, order, block)'s numerator, each fused via fuse_amplitude_into_consumer
    (inner = the leaf's own numerator at order-1, T-frame). The rank<=2 DIRECT
    part is NOT returned: compute it from the COMPILED numerator with the rank>=3
    amplitudes zeroed -- it carries the numerator's MIXED-ORDER direct leaves at
    their correct per-term orders (which a single tag_lowrank_leaves_by_order
    cannot) and, being compiled, has no P() tokens.

    Why not fuse_nested_numerator here: that fuser is for a CONSUMER block whose
    OUTER terms all pass through fuse_amplitude_into_consumer (P() resolved). For
    an amplitude, the un-fused 'rest' (direct) terms keep raw P() tokens that
    raw_terms_to_einsum_dressed cannot fold, AND its uniform order-1 tag mis-tags
    the amplitude's own mixed-order direct leaves (e.g. t2_3's numerator has BOTH
    a direct t2_2 (order 2) and, from a V.V.t2_1 term, a direct t2_1 (order 1)).
    The split avoids both: fuse ONLY the pure integral.rank>=3 terms (whose sole
    amplitude is the rank>=3 leaf, inner leaves uniformly order-1), leave the
    mixed-order direct part to the compiled numerator.

    Validated reconstructing t2_3_aaaa/abab's numerator to the ntau=6 quadrature
    accuracy on NH3/sto-3g (tests/test_mp4_t2_3_laplace.py). SINGLE-LEVEL only:
    each rank>=3 leaf must itself have a rank<=2 numerator (true for the order-2
    t3_2/t4_2 that t2_3 consumes). A leaf that is ITSELF order>=3 (e.g. the t3_3
    that t1_4 consumes) needs its OWN direct part fused in too -- the nested
    frontier, not handled here (raises)."""
    outer = numerator_fn(rank, order, block)
    out = {}
    for tag, rk in highrank_tags(outer).items():
        blk = tag.split('_', 1)[1]
        inner = numerator_fn(rk, order - 1, blk)
        if highrank_tags(inner):
            raise NotImplementedError(
                f"fuse_amplitude_rank3_tags: leaf {tag} (order {order - 1}) is "
                f"ITSELF rank>=3-dependent -- the nested-amplitude frontier "
                f"(t1_4 via t3_3). Its own direct part must be fused into the "
                f"outer amplitude too; single-level split does not cover it.")
        matching = [t for t in outer if term_has_factor(t, tag)]
        out[tag] = (rk, fuse_amplitude_into_consumer(inner, rk, matching, tag))
    return out


def _term_kinds(term):
    """Frozenset of dressing KINDS a term carries (from Oe_<kind>/Ve_<kind>
    factors), e.g. frozenset({'l3', 't4'}). Each kind is one independent tau
    slot in assemble_laplace_fused_multikind."""
    ks = set()
    for f in term[1:]:
        m = re.match(r'^(?:Oe|Ve)_(\w+)\(', f.strip())
        if m:
            ks.add(m.group(1))
    return frozenset(ks)


def assemble_laplace_fused_multikind(func_name, params, aliases, out_var,
                                     out_shape_expr, fused_terms, out_indices,
                                     eps_o_expr, eps_v_expr, ntau_default=6):
    """assemble_laplace_fused's variant for terms with a NON-uniform number of
    independently-denominatored amplitudes. Terms are grouped by their KIND SET
    (_term_kinds); each group gets one tau loop PER kind (kind-named dressings
    Oe_<kind>/Ve_<kind>), and integrates ONLY over the tau slots it carries.

    A nested l3ft2(3,1) block mixes terms with kinds {l3} (t3/t4 leaf was a
    materialized t1/t2), {l3,t3} and {l3,t4} -- so a single fixed nslots (as in
    assemble_laplace_fused) would over-integrate the light terms. Grouping fixes
    that: {l3} gets 1/D_l3, {l3,t4} gets 1/(D_l3 D_t4), etc.

    Flatten-p: each group flattens ITS OWN kinds' tau sub-grids into one batch
    axis p of length ntau**(#kinds) (mixed-radix via np.indices), defines that
    group's Oe_<k>/Ve_<k> as (P,no)/(P,nv) matrices + the flattened weight _wP,
    and raw_terms_to_einsum_dressed emits ONE einsum per term (p summed). No
    python tau loops. The empty-kindset group (weight 1, no denominator) emits
    plain undressed einsums with no p axis. Groups are emitted sequentially and
    each group's Oe_<k>/Ve_<k>/_wP<gi> are (re)defined immediately before its own
    terms, so reused kind names across groups (different P) never collide."""
    from collections import defaultdict
    groups = defaultdict(list)
    for t in fused_terms:
        groups[_term_kinds(t)].append(t)
    src = [f"def {func_name}({', '.join(params)}, ntau={ntau_default}):"]
    for generic, real in aliases.items():
        if generic != real:
            src.append(f"    {generic} = {real}")
    src.append(f"    ei = {eps_o_expr}")
    src.append(f"    ea = {eps_v_expr}")
    src.append("    gap_min = max(ea.min() - ei.max(), 1e-3)")
    src.append("    gap_max = ea.max() - ei.min()")
    src.append("    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)")
    src.append(f"    {out_var} = np.zeros({out_shape_expr})")
    for gi, kindset in enumerate(sorted(groups, key=lambda s: sorted(s))):
        kinds = sorted(kindset)
        d = len(kinds)
        if d == 0:
            for line in raw_terms_to_einsum_dressed(groups[kindset], out_var,
                                                    out_indices, weight_var=None):
                src.append(line)
            continue
        grid_shape = ", ".join(["ntau"] * d)
        src.append(f"    _tk{gi} = np.indices(({grid_shape},)).reshape({d}, -1)")
        for j, k in enumerate(kinds):
            src.append(f"    Oe_{k} = np.exp(ei[None, :] * tau[_tk{gi}[{j}]][:, None])")
            src.append(f"    Ve_{k} = np.exp(-ea[None, :] * tau[_tk{gi}[{j}]][:, None])")
        wP = f"_wP{gi}"
        src.append(f"    {wP} = " + " * ".join(f"(-sigma[_tk{gi}[{j}]])" for j in range(d)))
        for line in raw_terms_to_einsum_dressed(groups[kindset], out_var,
                                                out_indices, weight_var=wP):
            src.append(line)
    src.append(f"    return {out_var}")
    src.append("")
    return "\n".join(src)
