"""Automated ADC(n) block generator: superoperator elements derived with
pdaggerq and emitted as NumPy modules.

Every raw secular-matrix block is a SUPEROPERATOR (anticommutator-metric)
element -- the resolution of the Phase-2 open question, verified
numerically against the hand-written spin-orbital ADC's first-order blocks
on H2/sto-3g + H2O/sto-3g:

    M_IJ  = <Phi0| {C_I^dag, [H, C_J]} |Phi0>      (K + C blocks)
    U_pJ  = <Phi0| {a_p^dag or a_p, [H, C_J]} |Phi0>  (orbital-class coupling;
             a_p^dag for IP-sector classes, a_p for EA-sector ones)
    S_IJ  = <Phi0| {C_I^dag, C_J} |Phi0>            (overlap / Loewdin input)

expanded into 4 signed operator orderings per Hamiltonian piece
(+ C_I^dag H C_J, - C_I^dag C_J H, + H C_J C_I^dag, - C_J H C_I^dag) --
NOT the plain state sandwich (provably zero on virtual-p U rows: a leftmost
virtual creator cannot Wick-contract) and NOT the spectroscopic amplitude
<Phi0|C_J^dag a_p Omega|Phi0> (a different physical object -- the pole
residue -- numerically falsified as the secular U). The commutator [H, C_J]
handles the E0 subtraction automatically. Above first order the elements are
Omega-dressed (<l_j Phi0| ... t_k|Phi0>); the dressing STRUCTURES a block
carries at each order come from adc_order_rule (block_structures), and the
amplitude ORDERS fed into them are the runtime's choice.

Every block family has one emitter for both representations: spin-orbital
(dense tensors over the full spin-orbital space) and restricted (one function
per Ms-conserving spin assignment of the free indices, spin-blocked leaves
f_aa/g_aaaa/... or, with df=True, the DF/RI factors B_aa/B_bb). Output goes to
src/SingleReference/ADC/generatedADC/adc{N}/ (not version controlled;
generated_modules.RUNTIME_MODULES names the call that writes each module), in
the letter/axis conventions of precursor_classes.class_letters (holes
ascending, then particles ascending; bra letters at offset 0, ket letters
past them).

Sign convention: IP-sector elements use the REVERSED commutator [C_J, H]
(the Dyson hole-propagator convention -- see superop_element_terms'
docstring for the numerical diagnosis that pinned this); with that, every
generated block matches the hand-written spin-orbital ADC's convention at
+1 exactly (no residual per-class signs).
"""
import importlib
import os
import sys
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.CodeGeneration.pdaggerq_interface import raw_terms_to_einsum
from src.CodeGeneration.codegen_common import (
    OCC_LETTERS, VIR_LETTERS, ADC_DIMS, r_leaves, shape_expr, amp_leaves, pq_terms,
    generated_header, write_adc_module)
from src.CodeGeneration.df_codegen import dress_integral_factor
from src.SingleReference.ADC.adc_order_rule import (
    new_structures_at, u_order_terms, s_order_terms, s_structures_up_to,
    cross_block_leading_order, refuse_above_max_order)
from src.SingleReference.ADC.precursor_classes import (
    class_letters, precursor_ket_string, precursor_bra_string, class_index_convention,
    adjoint_string, union_operator_string, pair_index_convention,
    CROSS_PAIRS, CLASS_TAGS, cross_module_name, nondyson_module_name)

SECTORS = ('IP', 'EA')


# ---------------------------------------------------------------------------
# Elements (raw pdaggerq strings)
# ---------------------------------------------------------------------------

def _commutator_products(bra, ket, hams, tail, sign):
    """The four signed orderings of <bra| {., [h, ket]} tail> per ham h."""
    out = []
    for h in hams:
        out += [(+sign, bra + [h] + ket + tail), (-sign, bra + ket + [h] + tail),
                (+sign, [h] + ket + bra + tail), (-sign, ket + [h] + bra + tail)]
    return out


def superop_element_terms(bra_dag_tokens, ket_tokens, sector, hams=('f', 'v'),
                          left_ops=None, ket_dressing=None, spin_labels=None):
    """Raw pq.strings() for the anticommutator-metric secular element
    (4 signed orderings per ham):

        EA ket class:  <Phi0| {bra_dag, [ham, ket]} |Phi0>
        IP ket class:  <Phi0| {bra_dag, [ket, ham]} |Phi0>   (reversed
                       commutator = overall -1)

    The sector-dependent commutator direction is the standard Dyson
    hole/particle-propagator convention: (N-1)-sector poles enter the
    supermatrix as E0 - E(N-1) (below the chemical potential), so IP-class
    elements need [C_J, H]; (N+1)-sector poles as E(N+1) - E0, needing
    [H, C_J]. Diagnosed numerically (the generated-ADC(2) gate,
    first run): with [H, C_J] used for BOTH sectors, every IP-sector
    element (U rows, K diag, C^(1)) came out exactly -1x the validated
    hand-written spin-orbital convention while every EA-sector element matched
    at +1 -- one uniform sector sign, not per-block sign errors (e.g.
    K_2h1p came out eps_a - eps_i - eps_j).

    Omega-dressing on either side: `left_ops` dresses the bra (e.g. [['l2']]
    for <Omega^(1) Phi0| -- amplitude ORDER is a numeric choice made by the
    runtime driver, same design as generate_mpn_density.py's overlap/cross
    terms), `ket_dressing` is a list of amplitude tags appended rightmost so
    they act on |Phi0> first (e.g. ['t2']). hams=('f',) is the zeroth-order
    (K/diagonal) piece alone, ('v',) the first-order (C^(1)/U^(1)) piece."""
    sign = -1.0 if sector == 'IP' else +1.0
    tail = list(ket_dressing) if ket_dressing else []
    return pq_terms(left_ops or [['1']],
                    _commutator_products(list(bra_dag_tokens), list(ket_tokens), hams, tail, sign),
                    spin_labels)


def overlap_element_terms(bra_dag_tokens, ket_tokens, left_ops=None,
                          ket_dressing=None, spin_labels=None):
    """Raw pq.strings() for the anticommutator overlap
    <Phi0|{bra_dag, ket}|Phi0> ({A,B} = AB + BA, both orderings PLUS),
    optionally Omega-dressed on either side (as superop_element_terms)."""
    bra, ket = list(bra_dag_tokens), list(ket_tokens)
    tail = list(ket_dressing) if ket_dressing else []
    return pq_terms(left_ops or [['1']], [(+1.0, bra + ket + tail), (+1.0, ket + bra + tail)],
                    spin_labels)


def union_element_terms_ops(x_tokens, y_tokens, hams=('f', 'v'), left_ops=None,
                            ket_dressing=None, spin_labels=None):
    """Raw pq.strings() of the union-space superoperator element between two
    character-(+1) operators X (bra) and Y (ket), given as ket-form token
    strings,

        (X|H^|Y) = <Phi0| {X^dag, [H, Y]} |Phi0>
                 = <X Psi0|(H-E0)|Y Psi0> - <Y^dag Psi0|(H-E0)|X^dag Psi0>

    (Tarantelli and Cederbaum, PRA 45, 2790 (1992), Eq. 2.5a): the ordinary
    inner product of the union-space vectors (X Psi0; X^dag Psi0) with the
    composite Hamiltonian (H_P - E0) (+) -(H_M - E0). The normal commutator
    [H, Y] is used for every pair; no sector sign appears. For an EA-type
    pair this is superop_element_terms itself. For an IP-type pair, X = D_J =
    E_J^dag and Y = D_K, and {E_J, [H, E_K^dag]} = ({E_J^dag, [E_K, H]})^dag,
    so the element equals the two-sector one with the reversed commutator;
    under Omega-dressing the adjoint swaps the l- and t-dressed splits
    (l2v <-> vt2, l1f <-> ft1, (l2^(a), t2^(b)) <-> (l2^(b), t2^(a))), so the
    two conventions agree split-pair by split-pair and hence in every
    per-order sum. For a D x C pair (IP bra, EA ket) the element is the
    cross block the two-sector convention deletes.

    left_ops / ket_dressing as in superop_element_terms."""
    tail = list(ket_dressing) if ket_dressing else []
    return pq_terms(left_ops or [['1']],
                    _commutator_products(adjoint_string(list(x_tokens)), list(y_tokens), hams,
                                         tail, +1.0),
                    spin_labels)


def union_overlap_terms_ops(x_tokens, y_tokens, left_ops=None, ket_dressing=None,
                            spin_labels=None):
    """Raw pq.strings() of the union-space metric (X|Y) = <Phi0|{X^dag, Y}|Phi0>
    between two character-(+1) operators in ket-form tokens; identically zero
    for a D x C pair (odd operator strings on disjoint orbital sets
    anticommute), which the emission checks by getting no terms."""
    return overlap_element_terms(adjoint_string(list(x_tokens)), list(y_tokens),
                                 left_ops=left_ops, ket_dressing=ket_dressing,
                                 spin_labels=spin_labels)


def nondyson_element_terms_ops(bra_dag_tokens, ket_tokens, hams=('f', 'v'),
                               left_ops=None, ket_dressing=None, spin_labels=None):
    """Raw pq.strings() of the non-Dyson ISR element between two precursor
    states of ONE particle-number sector,

        <Psi0| E_I^dag (H - E_HF) E_J |Psi0>   (Omega-dressed on either side),

    the single ordering bra_dag . H . ket -- no commutator, no second
    ordering: the metric is the ordinary overlap of E_I Psi0 and E_J Psi0
    (Schirmer, Trofimov and Stelter, J. Chem. Phys. 109, 4734 (1998)).
    pdaggerq's f and v are the full operators (their vacuum parts sum_k f_kk
    and 1/2 sum_kl <kl||kl> appear as d(i,j) f(k,k)-type terms), so the raw
    element is <Psi0|E_I^dag H E_J|Psi0>; the runtime subtracts the whole
    E0 series (E^(0) + E^(1) + the MP correlation energies) times the
    lower-order overlaps and divides out the ground-state norm.
    left_ops / ket_dressing as in superop_element_terms."""
    tail = list(ket_dressing) if ket_dressing else []
    bra, ket = list(bra_dag_tokens), list(ket_tokens)
    return pq_terms(left_ops or [['1']], [(+1.0, bra + [h] + ket + tail) for h in hams],
                    spin_labels)


def nondyson_overlap_terms_ops(bra_dag_tokens, ket_tokens, left_ops=None,
                               ket_dressing=None, spin_labels=None):
    """Raw pq.strings() of the ordinary overlap <Psi0| E_I^dag E_J |Psi0>
    (Omega-dressed), the metric of the non-Dyson ISR."""
    tail = list(ket_dressing) if ket_dressing else []
    return pq_terms(left_ops or [['1']], [(+1.0, list(bra_dag_tokens) + list(ket_tokens) + tail)],
                    spin_labels)


def _union_pair_tokens(bra_k, bra_sector, ket_k, ket_sector):
    conv = pair_index_convention(bra_k, bra_sector, ket_k, ket_sector)
    return (union_operator_string(bra_k, bra_sector),
            union_operator_string(ket_k, ket_sector, occ_offset=conv['ket_occ_offset'],
                                  vir_offset=conv['ket_vir_offset']))


def union_element_terms(bra_k, bra_sector, ket_k, ket_sector, hams=('f', 'v'),
                        left_ops=None, ket_dressing=None, spin_labels=None):
    """union_element_terms_ops for a (bra class, ket class) pair of any
    character, with the pair_index_convention letter layout (bra letters
    first, ket letters past the bra's pool)."""
    x, y = _union_pair_tokens(bra_k, bra_sector, ket_k, ket_sector)
    return union_element_terms_ops(x, y, hams=hams, left_ops=left_ops,
                                   ket_dressing=ket_dressing, spin_labels=spin_labels)


def union_overlap_terms(bra_k, bra_sector, ket_k, ket_sector, left_ops=None,
                        ket_dressing=None, spin_labels=None):
    """union_overlap_terms_ops for a (bra class, ket class) pair."""
    x, y = _union_pair_tokens(bra_k, bra_sector, ket_k, ket_sector)
    return union_overlap_terms_ops(x, y, left_ops=left_ops, ket_dressing=ket_dressing,
                                   spin_labels=spin_labels)


# ---------------------------------------------------------------------------
# Block layouts and dressing structures (shared with sigma_codegen)
# ---------------------------------------------------------------------------

def class_pair_layout(bra_k, ket_k, sector):
    """(bra_dag tokens, ket tokens, bra letters, ket letters) of a class-class
    element of one sector: bra class at letter offset 0, ket class past the
    bra's letter pool."""
    conv = class_index_convention(bra_k, sector)
    bh, bp = class_letters(bra_k, sector)
    kh, kp = class_letters(ket_k, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
    ket = precursor_ket_string(ket_k, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
    return precursor_bra_string(bra_k, sector), ket, bh + bp, kh + kp


def orbital_layout(k, sector):
    """(ket tokens, ket letters, [(p letter, 'occ'), (p letter, 'vir')], p_tok)
    of the orbital<->class-k coupling U_pJ of one sector. The k=1 ket sits past
    one class-1 letter pool (as if a 2h1p bra were present) and p past both;
    the k=2 ket sits at offset 0 and p past it -- the layouts every generated
    k1 and k2 U module was written with. p_tok(p) is the orbital operator,
    a_p^dag for IP-sector classes, a_p for EA ones."""
    conv = class_index_convention(k, sector)
    if k == 1:
        oo, vo = conv['nh'], conv['np']
    elif k == 2:
        oo, vo = 0, 0
    else:
        raise ValueError(f'no orbital<->class-{k} layout')
    kh, kp = class_letters(k, sector, occ_offset=oo, vir_offset=vo)
    ket = precursor_ket_string(k, sector, occ_offset=oo, vir_offset=vo)
    rows = [(OCC_LETTERS[oo + conv['nh']], 'occ'), (VIR_LETTERS[vo + conv['np']], 'vir')]
    p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])
    return ket, kh + kp, rows, p_tok


def block_structures(order, kind='u', leading_order=1, max_rank=4):
    """{structure name: spec} a block module at `order` carries, spec
    (j, ham, k) for kind 'u' (dressed Hamiltonian element <l_j| ham |t_k>) and
    (j, k) for kind 's' (dressed overlap). At the block's leading order every
    structure the order rule assigns to that order (u_order_terms /
    s_order_terms: the module must be self-contained); above it only the
    structures first reachable at `order` (new_structures_at: lower ones live
    in the lower-order modules and are re-evaluated by the runtime at higher
    amplitude orders). Structures pdaggerq prunes for the block get no
    function."""
    if order <= leading_order:
        if kind == 'u':
            return {nm: (j, ham, k) for nm, j, ham, k, _, _ in u_order_terms(order, max_rank)}
        return {nm: (j, k) for nm, j, k, _, _ in s_order_terms(order, max_rank)}
    return dict(new_structures_at(order, max_rank=max_rank, kind=kind))


def dressing(spec):
    """(left_ops, ket_dressing, spin-orbital amplitude args) of a structure spec."""
    j, k = (spec[0], spec[2]) if len(spec) == 3 else spec
    return (([[f'l{j}']] if j else None), ([f't{k}'] if k else None),
            ([f'l{j}'] if j else []) + ([f't{k}'] if k else []))


def assemble_block(func_name, out_shape_expr, raw_terms, out_indices, doc,
                   extra_args=(), restricted=False, df=False):
    """One dense block function. Spin-orbital leaves (f, g, kd), or with
    restricted=True the spin-blocked leaves (f_aa, ..., g_aaaa, ..., d_aa, d_bb);
    raw_terms_to_einsum parses the '_aaaa'/'_abab' factor suffixes, dead leaf
    or amplitude args are harmless.

    df=True (restricted only) composes the DF/RI term rewrite before emission:
    each <pq||rs> becomes a pair of 3-index B factors and the signature takes
    B_aa/B_bb in place of g_aaaa/g_abab/g_bbbb (dress_integral_factor returns
    pdaggerq-format terms; array_expr leaves B's auxiliary axis un-sliced)."""
    if df:
        raw_terms = dress_integral_factor(raw_terms)
    leaves = r_leaves(df) if restricted else ['f', 'g', 'kd']
    args = leaves + list(extra_args) + ADC_DIMS
    src = [f"def {func_name}({', '.join(args)}):"]
    src.append(f'    """{doc}"""')
    src.append(f"    out = np.zeros({out_shape_expr})")
    src.extend(raw_terms_to_einsum(raw_terms, 'out', out_indices))
    src.append("    return out")
    src.append("")
    return "\n".join(src)


def emit_blocks(name, term_fn, out_idx, doc, extra_args=(), restricted=False, df=False,
                census=None, key=None):
    """The block functions of one element: term_fn(spin_labels) -> raw terms.
    Spin-orbital: one function `name` (spin_labels None) taking `extra_args`;
    its term count goes into census[key or name] when a census dict is given.
    Restricted: one function `name_<spins>` per Ms-conserving assignment of the
    free indices out_idx (the suffix is the a/b string of out_idx; pruned and
    Ms-violating assignments give no terms and no function), whose amplitude
    arguments are detected from the terms (a spin pattern's leaf set is
    pdaggerq's decision; the runtime introspects the signature). A pruned
    element (no terms) emits nothing."""
    if not restricted:
        terms = term_fn(None)
        print(f"{name}: {len(terms)} terms, out={out_idx}", flush=True)
        if census is not None:
            census[key or name] = len(terms)
        if not terms:
            return []
        return [assemble_block(name, shape_expr(out_idx), terms, out_idx, doc,
                               extra_args=extra_args)]
    funcs = []
    for spins in product('ab', repeat=len(out_idx)):
        terms = term_fn(dict(zip(out_idx, spins)))
        if not terms:
            continue
        suffix = ''.join(spins)
        amps = amp_leaves(terms)
        print(f"{name}_{suffix}: {len(terms)} terms, out={out_idx}, amps={amps}", flush=True)
        funcs.append(assemble_block(
            f'{name}_{suffix}', shape_expr(out_idx), terms, out_idx,
            f"{doc} Spin block {suffix}.", extra_args=amps, restricted=True, df=df))
    return funcs


def variant_suffix(restricted, df):
    """'_restricted', '_restricted_df' or '' -- the module-name suffix of a twin."""
    return ('_restricted' if restricted else '') + ('_df' if df else '')


def _write(fname, funcs, producer, what):
    write_adc_module(fname, funcs, generated_header(
        f'generate_adcn.py ({producer})', what,
        "See generate_adcn's docstring for the superoperator-element conventions."))


def check_df(restricted, df, who):
    """The DF/RI twins exist for the restricted (spin-blocked) blocks only."""
    if df and not restricted:
        raise ValueError(f'{who}: the DF/RI twin exists for the restricted blocks only')


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_adc2(restricted=False, df=False):
    """ADC(2) blocks of both sectors: U^(1) (orbital<->2h1p/2p1h, occ and vir
    rows), K (f part) and C^(1) (v part) of the class-class block, and the
    S^(0) overlap (gathers to the identity on ordered configs: a sanity
    block). restricted/df as in emit_blocks: the DF twin
    (adc2_blocks_restricted_df) is what the g-free ADC(3) coupling is built
    from (aug-cc-pVQZ g tensors would be 15-115 GB)."""
    check_df(restricted, df, 'generate_adc2')
    funcs = []
    for sector in SECTORS:
        tag = CLASS_TAGS[1][sector]
        ket, ket_letters, rows, p_tok = orbital_layout(1, sector)
        for p, row in rows:
            out_idx = ket_letters + [p]
            funcs += emit_blocks(
                f'u1_{tag}_{row}',
                lambda sl, p=p: superop_element_terms(p_tok(p), ket, sector, ('f', 'v'),
                                                      spin_labels=sl),
                out_idx, f"U^(1) {tag}/orbital coupling, {row} rows: "
                f"<Phi0|{{a_p{'^dag' if sector == 'IP' else ''}, [f+v, C_{tag}]}}|Phi0>, "
                f"axes ({', '.join(out_idx)}).", restricted=restricted, df=df)
        bra_dag, ket, bl, kl = class_pair_layout(1, 1, sector)
        for ham, name, what in (('f', f'k0_{tag}', 'K (zeroth-order, f part)'),
                                ('v', f'c1_{tag}', 'C^(1) (first-order, v part)')):
            funcs += emit_blocks(
                name, lambda sl, h=ham: superop_element_terms(bra_dag, ket, sector, (h,),
                                                              spin_labels=sl),
                bl + kl, f"{what} {tag}-{tag} block <Phi0|{{C_bra^dag, [{ham}, C_ket]}}|Phi0>, "
                f"axes bra ({', '.join(bl)}) x ket ({', '.join(kl)}).",
                restricted=restricted, df=df)
        funcs += emit_blocks(
            f's0_{tag}', lambda sl: overlap_element_terms(bra_dag, ket, spin_labels=sl),
            bl + kl, f"S^(0) {tag}-{tag} anticommutator overlap <Phi0|{{C_bra^dag, C_ket}}|Phi0> "
            "(gathers to the identity on ordered configs).", restricted=restricted, df=df)
    _write(f'adc2_blocks{variant_suffix(restricted, df)}.py', funcs, 'generate_adc2',
           'ADC(2) U^(1), K, C^(1) and S^(0) blocks.')


def _emit_orbital_family(prefix, k, sector, structures, kind, doc, restricted, df):
    """U (kind 'u') or S (kind 's') orbital<->class-k functions
    {prefix}_{tag}_{row}_{structure}, both rows."""
    tag = CLASS_TAGS[k][sector]
    ket, ket_letters, rows, p_tok = orbital_layout(k, sector)
    funcs = []
    for nm, spec in sorted(structures.items()):
        lo, kd, amps = dressing(spec)
        for p, row in rows:
            out_idx = ket_letters + [p]
            if kind == 'u':
                fn = (lambda sl, p=p, h=(spec[1],), lo=lo, kd=kd: superop_element_terms(
                    p_tok(p), ket, sector, h, left_ops=lo, ket_dressing=kd, spin_labels=sl))
            else:
                fn = (lambda sl, p=p, lo=lo, kd=kd: overlap_element_terms(
                    p_tok(p), ket, left_ops=lo, ket_dressing=kd, spin_labels=sl))
            funcs += emit_blocks(
                f'{prefix}_{tag}_{row}_{nm}', fn, out_idx,
                f"{doc} {tag}/orbital, structure {nm} ({row} rows): dressing left={lo}, "
                f"ket={kd}; amplitude orders are the driver's choice "
                f"(adc_order_rule.u_order_terms). Axes ({', '.join(out_idx)}).",
                extra_args=amps, restricted=restricted, df=df)
    return funcs


def generate_adc_u(u_order, restricted=False, df=False, max_rank=4, fname=None):
    """Orbital<->k=1 U^(u_order)/S^(u_order) couplings, the ADC(u_order+1)
    increment: every dressing structure first reachable at u_order
    (block_structures; lower-order ones live in their own modules and are
    reused by the runtime at higher amplitude orders). u_order 2 is ADC(3)'s
    U^(2) (adc3_u_blocks), 3 is ADC(4)'s U^(3) (adc4_u3_blocks), 4 is ADC(5)'s
    U^(4) (adc5_u4_blocks_restricted). Since U^(0)=0 (canonical) and
    S^(1)=0 identically, U_ortho^(2) is the plain sum of the order-2 splits;
    from order 3 the runtime adds the normalization and class-GS terms.
    restricted/df as in emit_blocks."""
    check_df(restricted, df, 'generate_adc_u')
    refuse_above_max_order(u_order + 1, 'generate_adc_u')
    funcs = []
    for sector in SECTORS:
        for kind in ('u', 's'):
            funcs += _emit_orbital_family(
                f'{kind}{u_order}', 1, sector, block_structures(u_order, kind, max_rank=max_rank),
                kind, f"{'U' if kind == 'u' else 'S'}^({u_order})", restricted, df)
    name = 'adc3_u_blocks' if u_order == 2 else f'adc{u_order + 1}_u{u_order}_blocks'
    _write(fname or f'{name}{variant_suffix(restricted, df)}.py', funcs,
           f'generate_adc_u({u_order})',
           f'Orbital<->2h1p/2p1h U^({u_order}) and S^({u_order}) dressing structures.')


CLASS_PAIRS = {'k1k1': (1, 1), 'k1k2': (1, 2), 'k2k2': (2, 2)}


def generate_adc_class_pair(order, pair='k1k1', restricted=False, df=False, kinds=('u', 's'),
                            max_rank_ext=None, fname=None):
    """Dressed class-class element at `order` (above the pair's first order)
    for the pair 'k1k1', 'k1k2' or 'k2k2': the block enters ADC(order + k_bra +
    k_ket) and carries the structures first reachable at `order`. Function
    names follow the runtime resolvers: c{order}_{tag}_{nm} / s{order}c_{tag}_{nm}
    for the self pairs (the k2 tags 3h2p/3p2h keep them apart from k1's),
    c{order}_{tag1}_{tag2}_{nm} / s_{tag1}_{tag2}_{nm} for k1k2. k1k1 at order 2
    is ADC(4)'s C^(2) + within-class S^(2) (adc4_c2_blocks; the runtime adds the
    N^(2) normalization and the Loewdin -1/2(S2 K + K S2)), at order 3 ADC(5)'s
    C^(3) (adc5_c3_blocks); k1k2 at order 2, kinds ('u',), is ADC(5)'s C12
    (adc5_c12_blocks; the runtime adds the class-GS corrections).

    max_rank_ext: also offer the same-order structures of amplitude rank up to
    it (u_order_terms(order, max_rank_ext) minus rank <= 4) and write an
    OFFERED/SURVIVORS census -- class-class elements have more operators to
    bridge rank than the (0,1) element, so each pair gets its own census.
    restricted/df as in emit_blocks."""
    check_df(restricted, df, 'generate_adc_class_pair')
    bra_k, ket_k = CLASS_PAIRS[pair]
    adc_order = order + bra_k + ket_k
    refuse_above_max_order(adc_order, 'generate_adc_class_pair')
    struct = {kind: block_structures(order, kind) for kind in kinds}
    ext = {kind: {} for kind in kinds}
    if max_rank_ext:
        for kind in kinds:
            full = block_structures(order, kind, leading_order=order, max_rank=max_rank_ext)
            base = block_structures(order, kind, leading_order=order)
            ext[kind] = {nm: s for nm, s in full.items() if nm not in base}
        print(f"--- class pair {pair} at order {order}: rank>4 offers "
              f"{ {k: sorted(v) for k, v in ext.items()} } ---", flush=True)
    funcs = []
    survivors = {kind: [] for kind in kinds}
    for sector in SECTORS:
        tagb, tagk = CLASS_TAGS[bra_k][sector], CLASS_TAGS[ket_k][sector]
        bra_dag, ket, bl, kl = class_pair_layout(bra_k, ket_k, sector)
        for kind in kinds:
            for nm, spec in sorted({**struct[kind], **ext[kind]}.items()):
                lo, kd, amps = dressing(spec)
                if kind == 'u':
                    name = f'c{order}_{tagb}_{nm}' if pair != 'k1k2' else f'c{order}_{tagb}_{tagk}_{nm}'
                    fn = (lambda sl, h=(spec[1],), lo=lo, kd=kd: superop_element_terms(
                        bra_dag, ket, sector, h, left_ops=lo, ket_dressing=kd, spin_labels=sl))
                else:
                    name = f's{order}c_{tagb}_{nm}' if pair != 'k1k2' else f's_{tagb}_{tagk}_{nm}'
                    fn = (lambda sl, lo=lo, kd=kd: overlap_element_terms(
                        bra_dag, ket, left_ops=lo, ket_dressing=kd, spin_labels=sl))
                new = emit_blocks(
                    name, fn, bl + kl,
                    f"Order-{order} {tagb}<->{tagk} {'class-class' if kind == 'u' else 'overlap'} "
                    f"structure {nm} (raw dressed element): dressing left={lo}, ket={kd}. "
                    f"Axes bra ({', '.join(bl)}) x ket ({', '.join(kl)}).",
                    extra_args=amps, restricted=restricted, df=df)
                if new and nm in ext[kind] and nm not in survivors[kind]:
                    survivors[kind].append(nm)
                funcs += new
    if max_rank_ext:
        funcs.insert(0, f"OFFERED_RANK_EXT = { {k: sorted(v) for k, v in ext.items()}!r}\n"
                        f"SURVIVORS_RANK_EXT = {survivors!r}\nORDER = {order}\nPAIR = {pair!r}")
    default = (f'adc{adc_order}_c{order}_blocks' if pair == 'k1k1'
               else f'adc{adc_order}_c{order}_{pair}_blocks')
    _write(fname or f'{default}{variant_suffix(restricted, df)}.py', funcs,
           f'generate_adc_class_pair({order}, {pair!r})',
           f'Order-{order} dressed {pair} class-class (and overlap) structures.')


def generate_adc4_base_k2(restricted=False):
    """The ADC(4)-base blocks involving the k=2 classes (3h2p/3p2h):

    1. the first-order k=1 <-> k=2 coupling c1_{tag1}_{tag2}
       <{C_k1^dag, [C_k2, H]}> (IP) / [H, C_k2] (EA);
    2. the orbital <-> k=2 U^(2)/S^(2) dressed structures at their leading
       order 2 (U^(1)_k2 vanishes as an effective coupling); the runtime builds
       U = U_raw^(2) - eps_p * S^(2)_pJ (the class-GS representation).

    The 10-index k=2 class-class blocks are not generated here: K_3h2p/K_3p2h
    are plain eps sums and the (2,2) coupling first enters at ADC(5)
    (generate_adc5_k2k2). restricted as in emit_blocks."""
    funcs = []
    for sector in SECTORS:
        tag1, tag2 = CLASS_TAGS[1][sector], CLASS_TAGS[2][sector]
        bra_dag, ket, bl, kl = class_pair_layout(1, 2, sector)
        funcs += emit_blocks(
            f'c1_{tag1}_{tag2}',
            lambda sl: superop_element_terms(bra_dag, ket, sector, ('f', 'v'), spin_labels=sl),
            bl + kl, f"First-order {tag1}<->{tag2} coupling "
            f"<Phi0|{{C_{tag1}^dag, [{'C,H' if sector == 'IP' else 'H,C'}]}}|Phi0>, axes bra "
            f"({', '.join(bl)}) x ket ({', '.join(kl)}).", restricted=restricted)
        for kind in ('u', 's'):
            funcs += _emit_orbital_family(
                f'{kind}2', 2, sector, block_structures(2, kind, leading_order=2), kind,
                f"{'U' if kind == 'u' else 'S'}^(2)", restricted, False)
    _write(f'adc4_k2_blocks{variant_suffix(restricted, False)}.py', funcs, 'generate_adc4_base_k2',
           'First-order 2h1p<->3h2p coupling and the orbital<->3h2p U^(2)/S^(2) structures.')


def generate_adc5_k2k2():
    """SPIN-ORBITAL k=2 class-class self-coupling -- the ADC(5) (2,2) block.

    `adc_order_rule.block_order(n, 2, 2)` is 0 at ADC(4) (K only, diagonal) and
    1 at ADC(5), so this first-order 3h2p x 3h2p / 3p2h x 3p2h coupling is
    the block that makes ADC(5) structurally different from ADC(4); the rule
    says the top manifold's self-block is order 0 for every even n and 1 for
    every odd n. Two blocks per sector, as generate_adc2 splits the k=1
    class-class element: k0_{tag} (f part; gathers to sigma_driver.diag_K(...,
    k=2), a free end-to-end check of the class enumeration, letter offsets and
    sector sign) and c1_{tag} (v part, the new ADC(5) coupling).

    SIZE: 10-index tensors (no^3 nv^2)^2 for IP -- validation scale only
    (H2O/sto-3g is already ~2 GB). Production uses the matrix-free form
    (sigma_codegen.generate_adc5_k2k2_apply)."""
    funcs = []
    for sector in SECTORS:
        tag = CLASS_TAGS[2][sector]
        bra_dag, ket, bl, kl = class_pair_layout(2, 2, sector)
        for ham, name, what in (('f', f'k0_{tag}', 'K (zeroth-order, f part)'),
                                ('v', f'c1_{tag}', 'C^(1) (first-order, v part)')):
            funcs += emit_blocks(
                name, lambda sl, h=ham: superop_element_terms(bra_dag, ket, sector, (h,),
                                                              spin_labels=sl),
                bl + kl, f"{what} {tag}-{tag} self-coupling <Phi0|{{C_bra^dag, [{ham}, C_ket]}}|Phi0>, "
                f"axes bra ({', '.join(bl)}) x ket ({', '.join(kl)}). ADC(5) block -- "
                f"block_order(5,2,2) == 1.")
    _write('adc5_k2k2_blocks.py', funcs, 'generate_adc5_k2k2',
           'ADC(5) 3h2p/3p2h self-coupling, K and C^(1) (dense, validation scale).')


def _generated_module(subpackage, name):
    """An already generated module of generatedADC/<subpackage>/, imported when a
    generator that reuses its function names runs (never at import of this
    module: generated output is not part of the repository)."""
    try:
        return importlib.import_module(
            f'src.SingleReference.ADC.generatedADC.{subpackage}.{name}')
    except ModuleNotFoundError as exc:
        raise RuntimeError(f'{name} must be generated first (generatedADC/'
                           f'{subpackage}/{name}.py is missing)') from exc


def generate_adc_u_k2(orders, fname=None):
    """SPIN-ORBITAL orbital<->k=2 U/S tag structures at the amplitude orders
    `orders`, every structure first reachable at each (new_structures_at),
    named by its home order (u{o}_/s{o}_). A structure an already generated
    lower k=2 module carries (adc4_k2_blocks' u2_/s2_; adc5_u3_k2_blocks'
    u1_..u3_ when orders go above 3) is skipped and reused by the runtime at
    the higher amplitude orders. orders (1, 2, 3) is ADC(5)'s U^(3)_k2
    (adc5_u3_k2_blocks: block_order(5, 0, 2) == 3). Consumed through
    Sz-block-diagonal gathering (adc_autogenerated._k2_raw_S).

    Census (both sectors, row-asymmetric as usual at k=2) of (1, 2, 3): kept
    u: l1v, l3v, l2vt2, l2ft1, l3ft2, l4ft2 (rank-4 lambda survives at k=2
    although it prunes at k=1), and the undressed u1_v: generate_adc4_base_k2's
    "U^(1)_k2 = 0" is true of the EFFECTIVE coupling, not of the raw element,
    which completes the order-1 gauge layer raw^(1) = eps_p * S^(1) and enters
    the order-3 N^(2) normalization; kept s: l4t2; reused: l2v, l1f, l2f, s2_l2."""
    top = max(orders)
    refuse_above_max_order(top + 2, 'generate_adc_u_k2')
    lower = [_generated_module('adc4', 'adc4_k2_blocks')]
    if top > 3:
        lower.append(_generated_module('adc5', 'adc5_u3_k2_blocks'))
    funcs = []
    for sector in SECTORS:
        tag2 = CLASS_TAGS[2][sector]
        ket2, ket2_letters, rows, p_tok = orbital_layout(2, sector)
        for o in orders:
            for kind in ('u', 's'):
                for nm, spec in sorted(new_structures_at(o, kind=kind).items()):
                    lo, kd, amps = dressing(spec)
                    for p, row in rows:
                        name = f'{kind}{o}_{tag2}_{row}_{nm}'
                        have = [f'{m.__name__.rsplit(".", 1)[-1]}.{kind}{q}_{tag2}_{row}_{nm}'
                                for m in lower for q in (1, 2, 3)
                                if hasattr(m, f'{kind}{q}_{tag2}_{row}_{nm}')]
                        if have:
                            print(f"{name}: reused ({have[0]})", flush=True)
                            continue
                        if kind == 'u':
                            fn = (lambda sl, p=p, h=(spec[1],), lo=lo, kd=kd: superop_element_terms(
                                p_tok(p), ket2, sector, h, left_ops=lo, ket_dressing=kd))
                        else:
                            fn = (lambda sl, p=p, lo=lo, kd=kd: overlap_element_terms(
                                p_tok(p), ket2, left_ops=lo, ket_dressing=kd))
                        out_idx = ket2_letters + [p]
                        funcs += emit_blocks(
                            name, fn, out_idx,
                            f"{'U' if kind == 'u' else 'S'}^({o})-family {tag2}/orbital structure "
                            f"{nm} ({row} rows): dressing left={lo}, ket={kd}; amplitude orders "
                            f"are the driver's choice. Axes ({', '.join(out_idx)}).",
                            extra_args=amps)
    _write(fname or f'adc{top + 2}_u{top}_k2_blocks.py', funcs,
           f'generate_adc_u_k2({tuple(orders)})', 'Orbital<->3h2p/3p2h U/S structures by home order.')


def generate_adc5_s_k1k2():
    """Cross-class overlap S_{k1,k2} blocks (2h1p<->3h2p / 2p1h<->3p2h) -- the
    object the per-class ISR assembly never forms (it builds within-class
    Delta S and the orbital<->class S_pJ only), nonzero from order 2
    (|S|~9.77e-04 on H4/sto-3g). Element <Phi0|{C_k1^dag, C_k2}|Phi0>
    Omega-dressed per s_structures_up_to(3).

    Census (identical for both sectors): kept l1 (serves l1^(2) at order 2 and
    l1^(3) at order 3), l2t1, l3t2; the rest prune. Function names are
    ORDER-AGNOSTIC (s_{tag1}_{tag2}_{structure}): amplitude order is the
    driver's choice (adc_autogenerated.cross_class_overlap). Raw precursor
    orientation on both classes, no _K2_CLASS_SIGN (the consumers mixing this
    with adc4_base-convention k=2 objects apply it)."""
    funcs = []
    for sector in SECTORS:
        tag1, tag2 = CLASS_TAGS[1][sector], CLASS_TAGS[2][sector]
        bra_dag, ket, bl, kl = class_pair_layout(1, 2, sector)
        for nm, spec in sorted(s_structures_up_to(3).items()):
            lo, kd, amps = dressing(spec)
            funcs += emit_blocks(
                f's_{tag1}_{tag2}_{nm}',
                lambda sl, lo=lo, kd=kd: overlap_element_terms(bra_dag, ket, left_ops=lo,
                                                               ket_dressing=kd),
                bl + kl, f"Cross-class overlap S {tag1}<->{tag2} structure {nm}: dressing "
                f"left={lo}, ket={kd}; order-agnostic, raw precursor orientation. Axes bra "
                f"({', '.join(bl)}) x ket ({', '.join(kl)}).", extra_args=amps)
    _write('adc5_s_k1k2_blocks.py', funcs, 'generate_adc5_s_k1k2',
           '2h1p<->3h2p cross-class overlap structures, orders 2 and 3.')


def nondyson_pair_layout(k_i, k_j, sector):
    """(bra_dag tokens, ket tokens, pair_index_convention) of the non-Dyson
    element between precursor classes k_i (bra) and k_j (ket) of `sector`:
    E_I^dag = precursor_bra_string, E_J = precursor_ket_string past the bra's
    letter pool."""
    conv = pair_index_convention(k_i, sector, k_j, sector)
    bra_dag = precursor_bra_string(k_i, sector)
    ket = precursor_ket_string(k_j, sector, occ_offset=conv['ket_occ_offset'],
                               vir_offset=conv['ket_vir_offset'])
    return bra_dag, ket, conv


def nondyson_structures(order, kind='u', max_rank=4):
    """{structure name: spec} first reachable at `order` for the non-Dyson
    block modules: the bare Fock element at order 0 (kind 'u'; no overlap
    structure exists at order 0, the zeroth-order metric is the identity),
    new_structures_at(order) above it. Lower-order structures live in the
    lower-order modules and are re-evaluated at higher amplitude orders."""
    if order == 0:
        return {'f': (0, 'f', 0)} if kind == 'u' else {}
    return new_structures_at(order, max_rank=max_rank, kind=kind)


def generate_nondyson_pair(order, k_i, k_j, fname=None):
    """Emit the non-Dyson ISR block between precursor classes k_i (bra) and
    k_j (ket) at `order`, both sectors in one module (functions
    nd{order}_{tag_i}_{tag_j}_{structure} for the Hamiltonian element,
    nds{order}_{tag_i}_{tag_j}_{structure} for the overlap), dense
    spin-orbital tensors with axes bra (holes, particles) x ket (holes,
    particles). Every structure first reachable at `order` is derived
    (pdaggerq prunes the ones that vanish; the census is recorded in TERMS)."""
    refuse_above_max_order(order + k_i + k_j, 'generate_nondyson_pair')
    funcs = []
    census = {'u': {}, 's': {}}
    for sector in SECTORS:
        tag_i, tag_j = CLASS_TAGS[k_i][sector], CLASS_TAGS[k_j][sector]
        bra_dag, ket, conv = nondyson_pair_layout(k_i, k_j, sector)
        axes = (f"Axes = bra ({', '.join(conv['bra_letters'])}) x ket "
                f"({', '.join(conv['ket_letters'])}).")
        for kind, pre in (('u', 'nd'), ('s', 'nds')):
            for nm, spec in sorted(nondyson_structures(order, kind).items()):
                lo, kd, amps = dressing(spec)
                if kind == 'u':
                    fn = (lambda sl, h=(spec[1],), lo=lo, kd=kd: nondyson_element_terms_ops(
                        bra_dag, ket, hams=h, left_ops=lo, ket_dressing=kd))
                    what = 'element <E_I^dag (H - E_HF) E_J>'
                else:
                    fn = (lambda sl, lo=lo, kd=kd: nondyson_overlap_terms_ops(
                        bra_dag, ket, left_ops=lo, ket_dressing=kd))
                    what = 'overlap <E_I^dag E_J>'
                funcs += emit_blocks(
                    f'{pre}{order}_{tag_i}_{tag_j}_{nm}', fn, conv['out_idx'],
                    f"Order-{order} non-Dyson {tag_i} x {tag_j} {what}, structure {nm} (raw, "
                    f"spin-orbital; amplitude order is the runtime's choice). {axes}",
                    extra_args=amps, census=census[kind])
    manifest = f"TERMS = {census!r}\nORDER = {order}\nCLASSES = ({k_i}, {k_j})"
    _write(fname or nondyson_module_name(order, k_i, k_j), [manifest] + funcs,
           f'generate_nondyson_pair({order}, {k_i}, {k_j})',
           'Non-Dyson ISR element and overlap structures, both sectors.')


def cross_pair_layout(pair):
    """(k_i, k_j, bra tag, ket tag, pair_index_convention) of the D x C cross
    pair `pair`: bra = the (k_i+1)h k_i p-dagger class (IP-type), ket = the
    (k_j+1)p k_j h class (EA-type)."""
    k_i, k_j = CROSS_PAIRS[pair]
    return (k_i, k_j, CLASS_TAGS[k_i]['IP'], CLASS_TAGS[k_j]['EA'],
            pair_index_convention(k_i, 'IP', k_j, 'EA'))


def cross_structures(order, pair, max_rank=4):
    """{structure name: (j, ham, k)} of the dressing structures the D x C
    block module at `order` carries: block_structures with the pair's leading
    order (cross_block_leading_order)."""
    k_i, k_j = CROSS_PAIRS[pair]
    return block_structures(order, 'u', leading_order=cross_block_leading_order(k_i, k_j),
                            max_rank=max_rank)


def generate_adc_cross_pair(order, pair='k1k1', restricted=False, df=False, max_rank=4,
                            fname=None):
    """D x C cross block at `order`: the union-space Hamiltonian block
    X = (D_kI|H^|C_kJ) = <Phi0|{E, [H, C]}|Phi0> between the (kI+1)h kI
    p-dagger class and the (kJ+1)p kJ h class, which the two-sector convention
    deletes (Tarantelli and Cederbaum, PRA 45, 2790 (1992), Eq. 4.5d). One
    function x{order}_{bra tag}_{ket tag}_{structure} per surviving dressing
    structure, axes bra (holes, particles) x ket (holes, particles); the
    amplitude order of each leaf is the runtime's choice
    (adc_order_rule.u_order_terms). Spin-orbital: the metric partner is
    derived too and must give no terms (it vanishes identically), and orders
    0 and 1 give no terms either -- both are recorded in the module's TERMS
    census, so 'pruned' and 'ungenerated' stay distinct. restricted/df as in
    emit_blocks. Validation scale: the (2h1p) x (2p1h) block is dense here."""
    check_df(restricted, df, 'generate_adc_cross_pair')
    k_i, k_j, tagb, tagk, conv = cross_pair_layout(pair)
    refuse_above_max_order(order + k_i + k_j, 'generate_adc_cross_pair')
    out_idx = conv['out_idx']
    funcs = []
    census = {'u': {}, 's': {}}
    print(f"--- cross pair {pair} ({tagb} x {tagk}) at order {order} ---", flush=True)
    for nm, spec in sorted(cross_structures(order, pair, max_rank).items()):
        lo, kd, amps = dressing(spec)
        funcs += emit_blocks(
            f'x{order}_{tagb}_{tagk}_{nm}',
            lambda sl, h=(spec[1],), lo=lo, kd=kd: union_element_terms(
                k_i, 'IP', k_j, 'EA', hams=h, left_ops=lo, ket_dressing=kd, spin_labels=sl),
            out_idx,
            f"Order-{order} {tagb}(dagger-type bra) x {tagk} union-space cross block, "
            f"structure {nm}: <Phi0| {lo or ''} {{E, [{spec[1]}, C]}} {kd or ''} |Phi0>, raw "
            f"(no normalization, no class-GS; both first act at order 3). Axes = bra "
            f"({', '.join(conv['bra_letters'])}) x ket ({', '.join(conv['ket_letters'])}).",
            extra_args=amps, restricted=restricted, df=df, census=census['u'], key=nm)
    if not restricted:
        for nm, j, k, a, b in s_order_terms(order, max_rank=max_rank):
            if nm in census['s']:
                continue
            lo, kd, _ = dressing((j, k))
            terms = union_overlap_terms(k_i, 'IP', k_j, 'EA', left_ops=lo, ket_dressing=kd)
            census['s'][nm] = len(terms)
            if terms:
                raise RuntimeError(
                    f"metric cross block {tagb} x {tagk} structure {nm} has {len(terms)} "
                    f"terms; the union-space metric between a D-type and a C-type "
                    f"operator vanishes identically, so this is an emitter defect")
        funcs.insert(0, f"ORDER = {order}\nPAIR = {pair!r}\nBRA = {tagb!r}\nKET = {tagk!r}\n"
                        f"TERMS = {census!r}")
    _write(fname or cross_module_name(order, pair, restricted=restricted, df=df), funcs,
           f'generate_adc_cross_pair({order}, {pair!r})',
           f'Order-{order} union-space D x C cross block {tagb} x {tagk}.')
    return census


if __name__ == '__main__':
    generate_adc2()
    generate_adc2(restricted=True)
    generate_adc_u(2)
    generate_adc_u(2, restricted=True)
    generate_adc_u(2, restricted=True, df=True)
    generate_adc_class_pair(2)
    generate_adc_class_pair(2, restricted=True)
    generate_adc4_base_k2()
    generate_adc4_base_k2(restricted=True)
    generate_adc_u(3)
    generate_adc_u(3, restricted=True)
