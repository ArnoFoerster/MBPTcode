"""Sigma-vector (matrix-free) emission for the ADC(n) generator: a class-class
or orbital-class block emitted as an APPLY (y[bra] = sum_ket block[bra,ket] *
x[ket]) instead of a dense (bra x ket) tensor -- the representation in which a
block too large to hold densely can be generated at all.

The elements, layouts and dressing structures are generate_adcn's
(superop_element_terms, class_pair_layout, orbital_layout, block_structures);
this module only changes the emission: assemble_apply_block hands the same
raw terms to pdaggerq_interface.raw_terms_to_sigma_einsum, which folds the ket
letters against the trial vector. Spin-orbital and restricted (spin-blocked,
optionally DF/RI) twins come from one emitter (emit_apply_blocks), and a
block whose transpose the symmetric supermatrix also needs gets both
directions from ONE term list: raw_terms_to_sigma_einsum only cares which
letters are output (bra) and which are folded (ket), so swapping the two
lists over the same terms is the exact config transpose -- no second
derivation, no hermiticity assumption (verified == forward^T to machine zero).

Trial-vector operand is named 'x' (NOT 'v' -- 'v' is the virtual-slice
argument used inside the emitted g[v,v,o,o]-style operands).
"""
import re
from itertools import product

from src.CodeGeneration.pdaggerq_interface import raw_terms_to_sigma_einsum, exact_coefficient
from src.CodeGeneration.codegen_common import (
    OCC_FAMILY, ADC_DIMS, r_leaves, shape_expr, amp_leaves, generated_header, write_adc_module)
from src.CodeGeneration.df_codegen import dress_integral_factor
from src.CodeGeneration.generate_adcn import (
    SECTORS, CLASS_PAIRS, superop_element_terms, overlap_element_terms, union_element_terms,
    class_pair_layout, orbital_layout, block_structures, dressing, variant_suffix,
    cross_pair_layout, cross_structures, nondyson_pair_layout, nondyson_structures,
    nondyson_element_terms_ops, nondyson_overlap_terms_ops)
from src.SingleReference.ADC.adc_order_rule import new_structures_at, refuse_above_max_order
from src.SingleReference.ADC.precursor_classes import (
    CLASS_TAGS, cross_module_name, nondyson_apply_module_name)

_VEC = 'x'


def assemble_apply_block(func_name, bra_indices, ket_indices, raw_terms, doc,
                         extra_args=(), restricted=False, df=False, vec_var=_VEC):
    """One apply function

        def {func_name}(<leaves>, *extra_args, o, v, no, nv, {vec_var}):
            y = zeros(bra_shape); <sigma einsums>; return y

    computing y[bra] = sum_ket block[bra,ket] * {vec_var}[ket] with no
    (bra x ket) tensor ever allocated. raw_terms is the same pq.strings()
    output the dense generator receives for this block; bra_indices/
    ket_indices is its out_indices split. Leaves (f, g, kd), or the
    spin-blocked (f_aa, ..., g_aaaa, ..., d_bb) with restricted=True.

    df=True composes the DF/RI term rewrite (dress_integral_factor) with the
    sigma emission: every <pq||rs> becomes a pair of 3-index B factors before
    emission and the signature takes B (spin-orbital) or B_aa/B_bb in place of
    g -- pure composition, since dress_integral_factor returns pdaggerq-format
    terms and array_expr already emits B factors with the aux axis un-sliced."""
    if df:
        raw_terms = dress_integral_factor(raw_terms)
    leaves = r_leaves(df) if restricted else (['f', 'B', 'kd'] if df else ['f', 'g', 'kd'])
    args = leaves + list(extra_args) + ADC_DIMS + [vec_var]
    src = [f"def {func_name}({', '.join(args)}):"]
    src.append(f'    """{doc}"""')
    src.append(f"    y = np.zeros({shape_expr(bra_indices)})")
    src.extend(raw_terms_to_sigma_einsum(raw_terms, 'y', bra_indices, ket_indices,
                                         vec_var=vec_var))
    src.append("    return y")
    src.append("")
    return "\n".join(src)


def emit_apply_blocks(name, term_fn, bra_idx, ket_idx, doc, extra_args=(), restricted=False,
                      df=False, transpose=None, transpose_suffix='forward'):
    """The apply functions of one element: term_fn(spin_labels) -> raw terms.
    Spin-orbital: one function `name` (spin_labels None). Restricted: one
    function `name_<spins>` per Ms-conserving assignment of bra_idx + ket_idx,
    amplitude arguments detected from the terms (generate_adcn.emit_blocks).

    transpose=(name_t, doc_t) also emits the transpose (y[ket] = block^T x[bra])
    from the same term list. Precondition: bra_idx and ket_idx are disjoint
    letter sets (no mixed bra<->ket P() swap), true for every class pair by
    the letter offsets of class_pair_layout / orbital_layout.
    transpose_suffix -- the spin-index order of the transpose's name suffix,
    which must match the runtime lookup: 'forward' keeps the forward
    (bra_idx + ket_idx) order for both (u2_k2_dot_orb / orb_dot_u2_k2 always
    put the k2 pattern first); 'ket_bra' reorders it to the transpose's own
    (ket_idx + bra_idx) order (restricted_class_matvec derives the suffix from
    the call's (bra, ket), so it swaps between the two directions; getting
    this wrong leaves the transposes under un-findable names, only the
    spin-symmetric patterns match)."""
    def both(nm, suffix_t, terms, doc_s, doc_ts, amps, rflag):
        out = [assemble_apply_block(nm, bra_idx, ket_idx, terms, doc_s, extra_args=amps,
                                    restricted=rflag, df=df)]
        if transpose:
            out.append(assemble_apply_block(f'{transpose[0]}{suffix_t}', ket_idx, bra_idx, terms,
                                            doc_ts, extra_args=amps, restricted=rflag, df=df))
        return out

    if not restricted:
        terms = term_fn(None)
        print(f"{name}: {len(terms)} terms, bra={bra_idx}, ket={ket_idx}{' [DF]' if df else ''}",
              flush=True)
        if not terms:
            return []
        return both(name, '', terms, doc, transpose and transpose[1], list(extra_args), False)
    out_idx = list(bra_idx) + list(ket_idx)
    funcs = []
    for spins in product('ab', repeat=len(out_idx)):
        sl = dict(zip(out_idx, spins))
        terms = term_fn(sl)
        if not terms:
            continue
        suffix = ''.join(spins)
        suffix_t = (''.join(sl[l] for l in list(ket_idx) + list(bra_idx))
                    if transpose_suffix == 'ket_bra' else suffix)
        funcs += both(f'{name}_{suffix}', f'_{suffix_t}', terms, f"{doc} Spin block {suffix}.",
                      transpose and f"{transpose[1]} Spin block {suffix_t}.", amp_leaves(terms),
                      True)
    return funcs


def _write(fname, funcs, producer, what):
    write_adc_module(fname, funcs, generated_header(
        f'sigma_codegen.py ({producer})', what,
        "Matrix-free APPLY form: y[bra] = sum_ket block[bra,ket] * x[ket], no dense block."))


def _is_fov_zero(raw_terms):
    """True when every term carries an occupied-virtual Fock factor
    f(occ, vir) / f(vir, occ): the structure vanishes for a canonical
    Hartree-Fock reference (Brillouin) and is not emitted in the production
    (apply) form, whose amplitude pool would evaluate it at the wrong order
    anyway. The dense validation modules keep it and the gates show it zero.
    Labels are classified by family (pdaggerq_interface), so the overflow
    capitals pdaggerq prints (f(I,b)) count too."""
    def fov(tok):
        m = re.match(r'^f\((\w+),(\w+)\)$', tok.strip())
        return bool(m) and ((m.group(1) in OCC_FAMILY) != (m.group(2) in OCC_FAMILY))
    return bool(raw_terms) and all(any(fov(t) for t in term[1:]) for term in raw_terms)


def generate_c1_self_apply(k=1, restricted=False, df=False):
    """The first-order C^(1) class-k self-block (v part) in apply form.
    k=1 (2h1p/2p1h, adc4_c1_self_apply_blocks) is the other half, with the
    assembled C^(2), of the excitation-space class-class matvec of ADC(4);
    k=2 (3h2p/3p2h, adc5_k2k2_apply_blocks) is the ADC(5) (2,2) block
    (block_order(5,2,2) == 1; absent at ADC(4), where the order-0 K-only
    diagonal makes the closed-form downfold Schur exact at even orders) --
    the representation that matters for it, since the dense twin
    (generate_adcn.generate_adc5_k2k2) is a 10-index tensor. The block is its
    own transpose. restricted/df as in emit_apply_blocks (k=2 restricted
    enumerates 2^10 spin assignments per sector at generation time)."""
    funcs = []
    for sector in SECTORS:
        tag = CLASS_TAGS[k][sector]
        bra_dag, ket, bl, kl = class_pair_layout(k, k, sector)
        funcs += emit_apply_blocks(
            f'apply_c1_{tag}',
            lambda sl, bd=bra_dag, kt=ket, sec=sector: superop_element_terms(
                bd, kt, sec, ('v',), spin_labels=sl),
            bl, kl, f"First-order C^(1) {tag}-{tag} self-block APPLY (v part): "
            f"y[bra] = sum_ket C1 * x[ket].", restricted=restricted, df=df)
    stem = 'adc4_c1_self_apply' if k == 1 else f'adc5_k{k}k{k}_apply'
    _write(f'{stem}{variant_suffix(restricted, df)}_blocks.py', funcs,
           f'generate_c1_self_apply(k={k})', f'First-order class-{k} self-block C^(1).')


def generate_adc_class_pair_apply(order, pair='k1k1', restricted=False, df=False,
                                  drop_fov_zero=False, fname=None):
    """The dressed class-class element of generate_adcn.generate_adc_class_pair
    (kind 'u') in apply form: apply_c{order}_{tag}_{structure} for the self
    pairs (one direction: the block is symmetric), and both directions
    apply_c{order}_{tag1}_{tag2}_... / apply_c{order}_{tag2}_{tag1}_... for
    k1k2 ('ket_bra' suffixes, restricted_class_matvec's lookup). k1k1 order 2
    is ADC(4)'s C^(2) (adc4_c2_apply[...]_blocks), order 3 ADC(5)'s C^(3)
    (adc5_c3_apply_restricted[_df]_blocks); k1k2 order 2 is ADC(5)'s raw C12
    (adc5_c12_apply_restricted[_df]_blocks; the consumer applies the
    class-GS corrections, reusing the U2_k2 apply module's s2_l2 split and the
    S12 apply).

    drop_fov_zero skips every structure whose terms all carry f[o,v]
    (_is_fov_zero): zero on this code path, since the class blocks always
    see the canonical f (BuildSuperMatrixADCnRestricted.rargs sets
    f_aa = diag(eps)); at C^(3) these are l2ft3 and l1ft2, which leaves
    C^(3) with rank <= 2 amplitudes only, so it needs no Laplace fusion.
    No s3c apply exists: pdaggerq prunes every order-3 within-class overlap,
    so Delta S^(3) is the order-2 s2c structure at higher amplitude orders."""
    bra_k, ket_k = CLASS_PAIRS[pair]
    refuse_above_max_order(order + bra_k + ket_k, 'generate_adc_class_pair_apply')
    funcs = []
    for sector in SECTORS:
        tagb, tagk = CLASS_TAGS[bra_k][sector], CLASS_TAGS[ket_k][sector]
        bra_dag, ket, bl, kl = class_pair_layout(bra_k, ket_k, sector)
        for nm, spec in sorted(block_structures(order, 'u').items()):
            lo, kd, amps = dressing(spec)
            fn = (lambda sl, h=(spec[1],), lo=lo, kd=kd: superop_element_terms(
                bra_dag, ket, sector, h, left_ops=lo, ket_dressing=kd, spin_labels=sl))
            if drop_fov_zero and _is_fov_zero(fn(None)):
                print(f"apply_c{order}_{tagb}_{nm}: every term carries f[o,v] -- not emitted")
                continue
            doc = (f"Order-{order} {tagb}<->{tagk} structure {nm} APPLY: dressing left={lo}, "
                   f"ket={kd}; amplitude orders are the driver's choice.")
            if pair == 'k1k2':
                funcs += emit_apply_blocks(
                    f'apply_c{order}_{tagb}_{tagk}_{nm}', fn, bl, kl, f"{doc} Forward.",
                    extra_args=amps, restricted=restricted, df=df,
                    transpose=(f'apply_c{order}_{tagk}_{tagb}_{nm}', f"{doc} Transpose."),
                    transpose_suffix='ket_bra')
            else:
                funcs += emit_apply_blocks(f'apply_c{order}_{tagb}_{nm}', fn, bl, kl, doc,
                                           extra_args=amps, restricted=restricted, df=df)
    stem = f'adc{order + bra_k + ket_k}_c{order if pair == "k1k1" else f"{bra_k}{ket_k}"}_apply'
    variant = ('_restricted' if restricted else '') + ('_df' if df else '')
    _write(fname or f'{stem}{variant}_blocks.py', funcs,
           f'generate_adc_class_pair_apply({order}, {pair!r})',
           f'Order-{order} dressed {pair} class-class structures.')


def generate_adc4_base_s2c_apply(restricted=False, df=False):
    """The within-class S^(2) overlap structures in apply form
    (adc4_c2_s2c_apply[...]_blocks), for the assembled C^(2) matvec
    (sigma_driver.assembled_c2_matvec[_restricted]), which applies the ISR
    Loewdin correction -1/2 (K s2 + s2 K) with s2 = s2_raw - N^(2) I. Same
    structures and element as the dense generator's s2c leg
    (generate_adcn.generate_adc_class_pair(2))."""
    funcs = []
    for sector in SECTORS:
        tag = CLASS_TAGS[1][sector]
        bra_dag, ket, bl, kl = class_pair_layout(1, 1, sector)
        for nm, spec in sorted(block_structures(2, 's').items()):
            lo, kd, amps = dressing(spec)
            funcs += emit_apply_blocks(
                f'apply_s2c_{tag}_{nm}',
                lambda sl, lo=lo, kd=kd: overlap_element_terms(bra_dag, ket, left_ops=lo,
                                                               ket_dressing=kd, spin_labels=sl),
                bl, kl, f"S^(2) {tag}-{tag} within-class overlap structure {nm} APPLY: "
                f"y[bra] = sum_ket overlap * x[ket].", extra_args=amps,
                restricted=restricted, df=df)
    _write(f'adc4_c2_s2c_apply{variant_suffix(restricted, df)}_blocks.py', funcs,
           'generate_adc4_base_s2c_apply', 'Within-class S^(2) overlap structures.')


def generate_adc4_base_k2_apply(restricted=False, df=False):
    """The first-order k=1<->k=2 coupling (generate_adcn.generate_adc4_base_k2's
    c1_{tag1}_{tag2}) in apply form, both directions: apply_c1_{tag1}_{tag2}
    folds a 3h2p/3p2h trial into the k1 class, apply_c1_{tag2}_{tag1} (the
    transpose C1^T, 'ket_bra' suffixes) a k1 trial into the k2 class; the
    symmetric ADC(4) supermatrix (H[k1,k2] = C1, H[k2,k1] = C1^T) needs both,
    and the 3h2p/3p2h block is the ~16 GB Ne/aVDZ wall, so the transpose must be
    matrix-free too."""
    funcs = []
    for sector in SECTORS:
        tag1, tag2 = CLASS_TAGS[1][sector], CLASS_TAGS[2][sector]
        bra_dag, ket, bl, kl = class_pair_layout(1, 2, sector)
        funcs += emit_apply_blocks(
            f'apply_c1_{tag1}_{tag2}',
            lambda sl, bd=bra_dag, kt=ket, sec=sector: superop_element_terms(
                bd, kt, sec, ('f', 'v'), spin_labels=sl),
            bl, kl, f"First-order {tag1}<->{tag2} coupling APPLY (forward): "
            f"y[{tag1}] = sum_ket C1 * x[{tag2} ket].", restricted=restricted, df=df,
            transpose=(f'apply_c1_{tag2}_{tag1}',
                       f"First-order {tag1}<->{tag2} coupling APPLY (transpose C1^T): "
                       f"y[{tag2}] = sum_ket C1 * x[{tag1} ket]; same term list."),
            transpose_suffix='ket_bra')
    _write(f'adc4_k2_apply{variant_suffix(restricted, df)}_blocks.py', funcs,
           'generate_adc4_base_k2_apply', 'First-order 2h1p<->3h2p coupling, both directions.')


def generate_adc5_s12_apply_restricted(df=False):
    """The order-2 k1<->k2 cross-class overlap S_{k1,k2} (dense spin-orbital
    reference adc5_s_k1k2_blocks) in apply form, spin-blocked, both directions
    -- the -K_k1*S12 class-GS correction of the effective C12 matvec
    (c12_o2_generated). Structures: s_order_terms(2) at this pair
    (block_structures with leading order 2), of which pdaggerq keeps only l1
    (at l1^(2)); the order-3 survivors of the dense module (l2t1, l3t2) need
    budget >= 3 and are NOT in this module -- bundling them fed
    restricted_class_matvec's prefix sweep order-3 content (a KeyError)."""
    funcs = []
    for sector in SECTORS:
        tag1, tag2 = CLASS_TAGS[1][sector], CLASS_TAGS[2][sector]
        bra_dag, ket, bl, kl = class_pair_layout(1, 2, sector)
        for nm, spec in sorted(block_structures(2, 's', leading_order=2).items()):
            lo, kd, _ = dressing(spec)
            doc = f"Order-2 {tag1}<->{tag2} overlap structure {nm} APPLY"
            funcs += emit_apply_blocks(
                f'apply_s12_{tag1}_{tag2}_{nm}',
                lambda sl, lo=lo, kd=kd: overlap_element_terms(bra_dag, ket, left_ops=lo,
                                                               ket_dressing=kd, spin_labels=sl),
                bl, kl, f"{doc} (forward).", restricted=True, df=df,
                transpose=(f'apply_s12_{tag2}_{tag1}_{nm}', f"{doc} (transpose)."),
                transpose_suffix='ket_bra')
    _write(f'adc5_s12_apply{variant_suffix(True, df)}_blocks.py', funcs,
           'generate_adc5_s12_apply_restricted', 'Order-2 2h1p<->3h2p overlap, both directions.')


def _orbital_k2_apply(orders, structures_at, df, prefix_of):
    """Orbital<->k2 U/S couplings in orbital-folded apply form, both directions:
    forward apply_{u|s}{o}_{tag2}_{row}_{nm} folds p_orb over the orbital index
    -> y[k2], transpose apply_{u|s}{o}t_... folds a k2 trial -> y[orbital row];
    'forward' suffixes for both (the orbital output is a single-index scatter,
    not a class transpose). Neither ever forms the dense (n_k2, norb) coupling
    -- the ~16 GB / 14 min aug-cc-pVTZ 3p2h blocker, and the
    restricted_dyson_operator peak of forming U2k2 only to apply its adjoint."""
    funcs = []
    for sector in SECTORS:
        tag2 = CLASS_TAGS[2][sector]
        ket2, ket2_letters, rows, p_tok = orbital_layout(2, sector)
        for o in orders:
            for kind in ('u', 's'):
                for nm, spec in sorted(structures_at(o, kind).items()):
                    lo, kd, _ = dressing(spec)
                    for p, row in rows:
                        if kind == 'u':
                            fn = (lambda sl, p=p, h=(spec[1],), lo=lo, kd=kd: superop_element_terms(
                                p_tok(p), ket2, sector, h, left_ops=lo, ket_dressing=kd,
                                spin_labels=sl))
                        else:
                            fn = (lambda sl, p=p, lo=lo, kd=kd: overlap_element_terms(
                                p_tok(p), ket2, left_ops=lo, ket_dressing=kd, spin_labels=sl))
                        pre = prefix_of(kind, o)
                        doc = (f"{'U' if kind == 'u' else 'S'}^({o})-family {tag2}/orbital {nm} "
                               f"({row}) APPLY")
                        funcs += emit_apply_blocks(
                            f'apply_{pre}_{tag2}_{row}_{nm}', fn, ket2_letters, [p],
                            f"{doc}: fold p_orb over the orbital index -> y[{tag2}].",
                            restricted=True, df=df,
                            transpose=(f'apply_{pre}t_{tag2}_{row}_{nm}',
                                       f"{doc} TRANSPOSE: fold the {tag2} trial -> y[orbital {row}]."))
    return funcs


def generate_u2_k2_apply_restricted(df=False):
    """adc4_u2_k2_apply_restricted[_df]_blocks: the orbital<->k2 U^(2)/S^(2)
    structures of generate_adcn.generate_adc4_base_k2 (leading order 2) in
    orbital-folded apply form, both directions (_orbital_k2_apply)."""
    funcs = _orbital_k2_apply((2,), lambda o, kind: block_structures(o, kind, leading_order=2),
                              df, lambda kind, o: f'{kind}2')
    _write(f'adc4_u2_k2_apply{variant_suffix(True, df)}_blocks.py', funcs,
           'generate_u2_k2_apply_restricted', 'Orbital<->3h2p U^(2)/S^(2), both directions.')


def generate_adc_u_k2_apply_restricted(orders=(1, 2, 3), df=False, fname=None):
    """The apply twin of generate_adcn.generate_adc_u_k2: every orbital<->k2
    structure first reachable at each order in `orders`, named by its home
    order (apply_u{o}_ / apply_u{o}t_ / apply_s{o}_ ...), so the driver resolves
    a structure by its home-order prefix exactly as the dense k2_raw_S does.
    (1, 2, 3) is ADC(5)'s U^(3)_k2 apply; it is self-contained (orders 1-2
    re-derived rather than resolved across modules)."""
    for o in orders:
        refuse_above_max_order(o + 2, 'generate_adc_u_k2_apply_restricted')
    funcs = _orbital_k2_apply(orders, lambda o, kind: new_structures_at(o, kind=kind), df,
                              lambda kind, o: f'{kind}{o}')
    _write(fname or f'adc5_u3_k2_apply{variant_suffix(True, df)}_blocks.py', funcs,
           f'generate_adc_u_k2_apply_restricted({tuple(orders)})',
           'Orbital<->3h2p U/S structures by home order, both directions.')


_PERM_TOKEN = re.compile(r'^P\((\w+),(\w+)\)$')


def _swap_labels(token, a, b):
    """Exchange the index labels a and b inside one pdaggerq factor string
    ('<j,k||b,l>_abab', 'l2(i,l,c,a)', 'f(k,a)', 'd(i,k)', 'P(i,j)')."""
    def sw(m):
        lab = m.group(0)
        return b if lab == a else (a if lab == b else lab)
    head, sep, tail = token.partition('(') if '(' in token else token.partition('<')
    if not sep:
        return token
    body, close, suffix = tail.partition(')' if sep == '(' else '>')
    body = re.sub(r'[A-Za-z]\w*', sw, body)
    return head + sep + body + close + suffix


def expand_mixed_permutations(raw_terms, bra_letters, ket_letters):
    """Expand every P(x,y) token that exchanges a bra letter with a ket letter
    into its two explicit terms (the term as printed, and the term with x<->y
    relabelled in every factor and the sign flipped). Same-side permutations
    are left for raw_terms_to_sigma_einsum, which factorizes them. Needed for
    the D x C cross blocks: their bra and ket antisymmetrizers share hole and
    particle lines across the two classes, so pdaggerq emits mixed swaps,
    which have no sigma-vector factorization."""
    bra = set(bra_letters)
    out = list(raw_terms)
    while True:
        new, changed = [], False
        for term in out:
            coef, rest = term[0], [t.strip() for t in term[1:]]
            hit = None
            for idx, tok in enumerate(rest):
                m = _PERM_TOKEN.match(tok)
                if m and ((m.group(1) in bra) != (m.group(2) in bra)):
                    hit = (idx, m.group(1), m.group(2))
                    break
            if hit is None:
                new.append([coef] + rest)
                continue
            idx, a, b = hit
            base = rest[:idx] + rest[idx + 1:]
            new.append([coef] + base)
            new.append([f'{-exact_coefficient(coef):+.15g}'] + [_swap_labels(t, a, b) for t in base])
            changed = True
        out = new
        if not changed:
            return out


def generate_adc_cross_pair_apply_restricted(order=2, pair='k1k1', max_rank=2, df=False):
    """Spin-blocked sigma-vector form of the D x C cross block
    (generate_adcn.generate_adc_cross_pair), BOTH directions from one
    per-spin term list: apply_x{order}_{bra tag}_{ket tag}_{structure}_{s_bra s_ket}
    folds a ket-class (EA-type) trial into the bra class (IP-type),
    apply_x{order}_{ket tag}_{bra tag}_{structure}_{s_ket s_bra} folds a bra-class
    trial into the ket class (the transpose), suffixes in the
    restricted_class_matvec (bra-of-the-call, ket-of-the-call) order. max_rank=2:
    the production route materializes no rank>=3 amplitude, and the rank-3
    structures of this block contribute nothing for a canonical Fock operator
    (the dense gate carries them and shows them zero). df=True DF-factorizes
    every two-electron bracket."""
    k_i, k_j, tagb, tagk, conv = cross_pair_layout(pair)
    refuse_above_max_order(order + k_i + k_j, 'generate_adc_cross_pair_apply_restricted')
    bra_idx, ket_idx = conv['bra_letters'], conv['ket_letters']
    funcs = []
    for nm, spec in sorted(cross_structures(order, pair, max_rank).items()):
        lo, kd, _ = dressing(spec)
        spin_free = union_element_terms(k_i, 'IP', k_j, 'EA', hams=(spec[1],), left_ops=lo,
                                        ket_dressing=kd)
        if not spin_free:
            continue
        if _is_fov_zero(spin_free):
            print(f"apply_x{order}_{tagb}_{tagk}_{nm}: every term carries f[o,v] "
                  f"(zero for canonical HF) -- not emitted", flush=True)
            continue
        doc = f"Order-{order} {tagb} x {tagk} union-space cross block, structure {nm} APPLY"
        funcs += emit_apply_blocks(
            f'apply_x{order}_{tagb}_{tagk}_{nm}',
            lambda sl, h=(spec[1],), lo=lo, kd=kd: expand_mixed_permutations(
                union_element_terms(k_i, 'IP', k_j, 'EA', hams=h, left_ops=lo,
                                    ket_dressing=kd, spin_labels=sl),
                bra_idx, ket_idx),
            bra_idx, ket_idx, f"{doc} (forward): y[{tagb}] = sum_ket X * x[{tagk} ket].",
            restricted=True, df=df,
            transpose=(f'apply_x{order}_{tagk}_{tagb}_{nm}',
                       f"{doc} (transpose X^T): y[{tagk}] = sum_ket X * x[{tagb} ket]."),
            transpose_suffix='ket_bra')
    _write(cross_module_name(order, pair, df=df, apply=True), funcs,
           f'generate_adc_cross_pair_apply_restricted({order}, {pair!r})',
           f'Order-{order} union-space D x C cross block {tagb} x {tagk}, both directions.')


def generate_nondyson_pair_apply_restricted(order, k_i, k_j, df=False):
    """Spin-blocked sigma-vector form of the non-Dyson ISR block between classes
    k_i (bra) and k_j (ket) at `order` (generate_adcn.generate_nondyson_pair's
    structures, both sectors in one module): apply_nd{order}_{tag_i}_{tag_j}_{s}
    folds a class-k_j trial into class k_i through <E_I^dag H E_J>,
    apply_ndt{order}_{tag_i}_{tag_j}_{s} is its transpose (spin suffix in the
    restricted_class_matvec bra-of-the-call order); apply_nds / apply_ndst are
    the overlap <E_I^dag E_J> likewise. df=True DF-factorizes every bracket."""
    refuse_above_max_order(order + k_i + k_j, 'generate_nondyson_pair_apply_restricted')
    funcs = []
    for sector in SECTORS:
        tag_i, tag_j = CLASS_TAGS[k_i][sector], CLASS_TAGS[k_j][sector]
        bra_dag, ket, conv = nondyson_pair_layout(k_i, k_j, sector)
        bra_idx, ket_idx = conv['bra_letters'], conv['ket_letters']
        for kind, pre, what in (('u', 'nd', 'element <E_I^dag H E_J>'),
                                ('s', 'nds', 'overlap <E_I^dag E_J>')):
            for nm, spec in sorted(nondyson_structures(order, kind).items()):
                lo, kd, _ = dressing(spec)
                if kind == 'u':
                    def term_fn(sl, h=(spec[1],), lo=lo, kd=kd):
                        return nondyson_element_terms_ops(bra_dag, ket, hams=h, left_ops=lo,
                                                          ket_dressing=kd, spin_labels=sl)
                else:
                    def term_fn(sl, lo=lo, kd=kd):
                        return nondyson_overlap_terms_ops(bra_dag, ket, left_ops=lo,
                                                          ket_dressing=kd, spin_labels=sl)
                doc = f"Order-{order} non-Dyson {tag_i} x {tag_j} {what}, structure {nm} APPLY"
                funcs += emit_apply_blocks(
                    f'apply_{pre}{order}_{tag_i}_{tag_j}_{nm}', term_fn, bra_idx, ket_idx,
                    f"{doc}: y[{tag_i}] = sum_ket B * x[{tag_j} ket].", restricted=True, df=df,
                    transpose=(f'apply_{pre}t{order}_{tag_i}_{tag_j}_{nm}',
                               f"{doc} (transpose): y[{tag_j}] = sum_ket B^T * x[{tag_i} ket]."),
                    transpose_suffix='ket_bra')
    _write(nondyson_apply_module_name(order, k_i, k_j, df), funcs,
           f'generate_nondyson_pair_apply_restricted({order}, {k_i}, {k_j}, df={df})',
           f'Order-{order} non-Dyson ISR block, classes ({k_i}, {k_j}), both sectors.')


if __name__ == '__main__':
    for df in (False, True):
        generate_adc_class_pair_apply(2, df=df)
        generate_adc_class_pair_apply(2, restricted=True, df=df)
        generate_adc4_base_s2c_apply(restricted=True, df=df)
        generate_c1_self_apply(restricted=True, df=df)
        generate_adc4_base_k2_apply(restricted=True, df=df)
        generate_u2_k2_apply_restricted(df=df)
        generate_c1_self_apply(k=2, restricted=True, df=df)
        generate_adc_class_pair_apply(3, restricted=True, df=df, drop_fov_zero=True)
    generate_adc4_base_k2_apply()
    generate_adc4_base_s2c_apply()
    generate_c1_self_apply()
    generate_c1_self_apply(k=2)
