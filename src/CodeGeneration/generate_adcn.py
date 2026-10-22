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
handles the E0 subtraction automatically. At this (ADC(2)) order every
block is an UNDRESSED element (no Omega/t2 dressing anywhere: S^(1) = 0
identically by particle counting, so the Loewdin layer is exactly trivial
here); Omega-dressed elements + the Loewdin assembly enter at
order 2 blocks (ADC(3)-level), Phase 3.

Emits src/SingleReference/ADC/generatedADC/adc2_blocks.py: one function
per (block, row-type), plain dense-tensor outputs over sliced occ/vir axes
in the fixed letter/axis conventions of precursor_classes.class_letters
(holes ascending, then particles ascending; bra letters at offset 0, ket
letters offset past them). Runtime config-space gathering/assembly lives in
the generated-ADC runtime (not part of this repository), mirroring
mpn_density_driver.py's split between generated pieces and driver.

Conventions match the repo's other generators (assemble via
raw_terms_to_einsum, GENERATED-header output file, run once + commit).
Sign convention: IP-sector elements use the REVERSED commutator [C_J, H]
(the Dyson hole-propagator convention -- see superop_element_terms'
docstring for the numerical diagnosis that pinned this); with that, every
generated block matches the hand-written spin-orbital ADC's convention at
+1 exactly (no residual per-class signs).
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Base.pdaggerq_interface import pq_helper, raw_terms_to_einsum
from src.CodeGeneration.df_codegen import dress_integral_factor
from src.SingleReference.ADC.precursor_classes import (
    OCC_LETTERS, VIR_LETTERS, class_letters, precursor_ket_string,
    precursor_bra_string, class_index_convention, adjoint_string,
    union_operator_string, union_bra_string, pair_index_convention,
    CROSS_PAIRS, CLASS_TAGS, cross_module_name, nondyson_module_name)

OUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'SingleReference', 'ADC', 'generatedADC'))


def superop_element_terms(bra_dag_tokens, ket_tokens, sector, hams=('f', 'v'),
                          spin_labels=None):
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

    Pass hams=('f',) for the zeroth-order (K/diagonal) piece alone,
    ('v',) for the first-order (C^(1)/U^(1)) piece alone."""
    sign = -1.0 if sector == 'IP' else +1.0
    pq = pq_helper('fermi')
    pq.set_left_operators([['1']])
    pq.set_right_operators([['1']])
    for h in hams:
        pq.add_operator_product(+sign, bra_dag_tokens + [h] + ket_tokens)
        pq.add_operator_product(-sign, bra_dag_tokens + ket_tokens + [h])
        pq.add_operator_product(+sign, [h] + ket_tokens + bra_dag_tokens)
        pq.add_operator_product(-sign, ket_tokens + [h] + bra_dag_tokens)
    pq.simplify()
    terms = pq.strings() if spin_labels is None else pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


def overlap_element_terms(bra_dag_tokens, ket_tokens, left_ops=None,
                          ket_dressing=None, spin_labels=None):
    """Raw pq.strings() for the anticommutator overlap
    <Phi0|{bra_dag, ket}|Phi0> ({A,B} = AB + BA, both orderings PLUS),
    optionally Omega-dressed on either side (same conventions as
    superop_element_terms_dressed)."""
    tail = list(ket_dressing) if ket_dressing else []
    pq = pq_helper('fermi')
    pq.set_left_operators(left_ops if left_ops else [['1']])
    pq.set_right_operators([['1']])
    pq.add_operator_product(+1.0, bra_dag_tokens + ket_tokens + tail)
    pq.add_operator_product(+1.0, ket_tokens + bra_dag_tokens + tail)
    pq.simplify()
    terms = pq.strings() if spin_labels is None else pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


def superop_element_terms_dressed(bra_dag_tokens, ket_tokens, sector, hams,
                                   left_ops=None, ket_dressing=None,
                                   spin_labels=None):
    """superop_element_terms with Omega-dressing on either side: `left_ops`
    dresses the bra (e.g. [['l2']] for <Omega^(1) Phi0| -- amplitude ORDER
    is a numeric choice made by the runtime driver, same design as
    generate_mpn_density.py's overlap/cross terms), `ket_dressing` a list of
    amplitude tags appended rightmost so they act on |Phi0> first (e.g.
    ['t2'])."""
    sign = -1.0 if sector == 'IP' else +1.0
    tail = list(ket_dressing) if ket_dressing else []
    pq = pq_helper('fermi')
    pq.set_left_operators(left_ops if left_ops else [['1']])
    pq.set_right_operators([['1']])
    for h in hams:
        pq.add_operator_product(+sign, bra_dag_tokens + [h] + ket_tokens + tail)
        pq.add_operator_product(-sign, bra_dag_tokens + ket_tokens + [h] + tail)
        pq.add_operator_product(+sign, [h] + ket_tokens + bra_dag_tokens + tail)
        pq.add_operator_product(-sign, ket_tokens + [h] + bra_dag_tokens + tail)
    pq.simplify()
    terms = pq.strings() if spin_labels is None else pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


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

    left_ops / ket_dressing as in superop_element_terms_dressed."""
    bra_dag = adjoint_string(list(x_tokens))
    ket = list(y_tokens)
    tail = list(ket_dressing) if ket_dressing else []
    pq = pq_helper('fermi')
    pq.set_left_operators(left_ops if left_ops else [['1']])
    pq.set_right_operators([['1']])
    for h in hams:
        pq.add_operator_product(+1.0, bra_dag + [h] + ket + tail)
        pq.add_operator_product(-1.0, bra_dag + ket + [h] + tail)
        pq.add_operator_product(+1.0, [h] + ket + bra_dag + tail)
        pq.add_operator_product(-1.0, ket + [h] + bra_dag + tail)
    pq.simplify()
    terms = pq.strings() if spin_labels is None else pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


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
    left_ops / ket_dressing as in superop_element_terms_dressed."""
    bra_dag = list(bra_dag_tokens)
    ket = list(ket_tokens)
    tail = list(ket_dressing) if ket_dressing else []
    pq = pq_helper('fermi')
    pq.set_left_operators(left_ops if left_ops else [['1']])
    pq.set_right_operators([['1']])
    for h in hams:
        pq.add_operator_product(+1.0, bra_dag + [h] + ket + tail)
    pq.simplify()
    terms = pq.strings() if spin_labels is None else pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


def nondyson_overlap_terms_ops(bra_dag_tokens, ket_tokens, left_ops=None,
                               ket_dressing=None, spin_labels=None):
    """Raw pq.strings() of the ordinary overlap <Psi0| E_I^dag E_J |Psi0>
    (Omega-dressed), the metric of the non-Dyson ISR."""
    bra_dag = list(bra_dag_tokens)
    ket = list(ket_tokens)
    tail = list(ket_dressing) if ket_dressing else []
    pq = pq_helper('fermi')
    pq.set_left_operators(left_ops if left_ops else [['1']])
    pq.set_right_operators([['1']])
    pq.add_operator_product(+1.0, bra_dag + ket + tail)
    pq.simplify()
    terms = pq.strings() if spin_labels is None else pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


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


def nondyson_element_terms(k_i, k_j, sector, hams=('f', 'v'), left_ops=None,
                           ket_dressing=None, spin_labels=None):
    """nondyson_element_terms_ops for the class pair (k_i, k_j) of `sector`."""
    bra_dag, ket, _ = nondyson_pair_layout(k_i, k_j, sector)
    return nondyson_element_terms_ops(bra_dag, ket, hams=hams, left_ops=left_ops,
                                      ket_dressing=ket_dressing, spin_labels=spin_labels)


def nondyson_overlap_terms(k_i, k_j, sector, left_ops=None, ket_dressing=None,
                           spin_labels=None):
    """nondyson_overlap_terms_ops for the class pair (k_i, k_j) of `sector`."""
    bra_dag, ket, _ = nondyson_pair_layout(k_i, k_j, sector)
    return nondyson_overlap_terms_ops(bra_dag, ket, left_ops=left_ops,
                                      ket_dressing=ket_dressing, spin_labels=spin_labels)


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
    u_new = nondyson_structures(order, 'u')
    s_new = nondyson_structures(order, 's')
    funcs = []
    census = {'u': {}, 's': {}}
    for sector in ('IP', 'EA'):
        tag_i, tag_j = CLASS_TAGS[k_i][sector], CLASS_TAGS[k_j][sector]
        bra_dag, ket, conv = nondyson_pair_layout(k_i, k_j, sector)
        out_idx = conv['out_idx']
        for nm, (j, ham, k) in sorted(u_new.items()):
            lo = [[f'l{j}']] if j else None
            kd = [f't{k}'] if k else None
            extra_args = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
            name = f'nd{order}_{tag_i}_{tag_j}_{nm}'
            terms = nondyson_element_terms_ops(bra_dag, ket, hams=(ham,), left_ops=lo,
                                               ket_dressing=kd)
            census['u'][name] = len(terms)
            print(f"{name}: {len(terms)} terms", flush=True)
            if not terms:
                continue
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"Order-{order} non-Dyson {tag_i} x {tag_j} element <E_I^dag (H - E_HF) E_J>, "
                f"structure {nm} (raw, spin-orbital; amplitude order is the runtime's "
                f"choice). Axes = bra ({', '.join(conv['bra_letters'])}) x ket "
                f"({', '.join(conv['ket_letters'])}).",
                extra_args=extra_args))
        for nm, (j, k) in sorted(s_new.items()):
            lo = [[f'l{j}']] if j else None
            kd = [f't{k}'] if k else None
            extra_args = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
            name = f'nds{order}_{tag_i}_{tag_j}_{nm}'
            terms = nondyson_overlap_terms_ops(bra_dag, ket, left_ops=lo, ket_dressing=kd)
            census['s'][name] = len(terms)
            print(f"{name}: {len(terms)} terms", flush=True)
            if not terms:
                continue
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"Order-{order} non-Dyson {tag_i} x {tag_j} overlap <E_I^dag E_J>, "
                f"structure {nm} (raw, spin-orbital). Axes = bra "
                f"({', '.join(conv['bra_letters'])}) x ket ({', '.join(conv['ket_letters'])}).",
                extra_args=extra_args))
    manifest = f"TERMS = {census!r}\nORDER = {order}\nCLASSES = ({k_i}, {k_j})"
    _write_module(fname or nondyson_module_name(order, k_i, k_j), [manifest] + funcs)


def union_element_terms(bra_k, bra_sector, ket_k, ket_sector, hams=('f', 'v'),
                        left_ops=None, ket_dressing=None, spin_labels=None):
    """union_element_terms_ops for a (bra class, ket class) pair of any
    character, with the pair_index_convention letter layout (bra letters
    first, ket letters past the bra's pool)."""
    conv = pair_index_convention(bra_k, bra_sector, ket_k, ket_sector)
    x = union_operator_string(bra_k, bra_sector)
    y = union_operator_string(ket_k, ket_sector, occ_offset=conv['ket_occ_offset'],
                              vir_offset=conv['ket_vir_offset'])
    return union_element_terms_ops(x, y, hams=hams, left_ops=left_ops,
                                   ket_dressing=ket_dressing, spin_labels=spin_labels)


def union_overlap_terms(bra_k, bra_sector, ket_k, ket_sector, left_ops=None,
                        ket_dressing=None, spin_labels=None):
    """union_overlap_terms_ops for a (bra class, ket class) pair."""
    conv = pair_index_convention(bra_k, bra_sector, ket_k, ket_sector)
    x = union_operator_string(bra_k, bra_sector)
    y = union_operator_string(ket_k, ket_sector, occ_offset=conv['ket_occ_offset'],
                              vir_offset=conv['ket_vir_offset'])
    return union_overlap_terms_ops(x, y, left_ops=left_ops, ket_dressing=ket_dressing,
                                   spin_labels=spin_labels)


def assemble_block(func_name, out_shape_expr, raw_terms, out_indices, doc,
                   extra_args=()):
    args = ['f', 'g', 'kd'] + list(extra_args) + ['o', 'v', 'no', 'nv']
    src = [f"def {func_name}({', '.join(args)}):"]
    src.append(f'    """{doc}"""')
    src.append(f"    out = np.zeros({out_shape_expr})")
    src.extend(raw_terms_to_einsum(raw_terms, 'out', out_indices))
    src.append("    return out")
    src.append("")
    return "\n".join(src)


def _shape_expr(letters):
    return "(" + ", ".join(('no' if l in OCC_LETTERS else 'nv') for l in letters) + ",)"


def generate_adc2():
    funcs = []

    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)              # bra letters, offset 0
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])  # disjoint ket letters
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts        # class axis order: holes asc, parts asc
        ket_letters = ket_holes + ket_parts

        # --- U^(1): orbital-class coupling, one function per row type ---
        # IP-sector classes couple to the orbital block via a_p^dag; EA via a_p.
        p_occ = OCC_LETTERS[2 * conv['nh']]         # first occ letter past bra+ket
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])
        for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
            terms = superop_element_terms(p_tok(p_letter), ket, sector, hams=('f', 'v'))
            out_idx = ket_letters + [p_letter]
            print(f"u1_{tag}_{row}: {len(terms)} terms, out={out_idx}")
            funcs.append(assemble_block(
                f'u1_{tag}_{row}', _shape_expr(out_idx), terms, out_idx,
                f"U^(1) {tag}/orbital coupling, {row} orbital rows: "
                f"<Phi0|{{a_p{'^dag' if sector == 'IP' else ''}, [f+v, C_{tag}]}}|Phi0>, "
                f"axes ({', '.join(out_idx)}) = class ({', '.join(ket_letters)}) x orbital {p_letter}."))

        # --- K (f part) and C^(1) (v part) class-class blocks, separately ---
        for ham, name in (('f', f'k0_{tag}'), ('v', f'c1_{tag}')):
            terms = superop_element_terms(bra_dag, ket, sector, hams=(ham,))
            out_idx = bra_letters + ket_letters
            print(f"{name}: {len(terms)} terms, out={out_idx}")
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"{'K (zeroth-order, f part)' if ham == 'f' else 'C^(1) (first-order, v part)'} "
                f"{tag}-{tag} block: <Phi0|{{C_bra^dag, [{ham}, C_ket]}}|Phi0>, "
                f"axes = bra ({', '.join(bra_letters)}) x ket ({', '.join(ket_letters)})."))

        # --- S^(0) overlap (sanity: must gather to the identity) ---
        terms = overlap_element_terms(bra_dag, ket)
        out_idx = bra_letters + ket_letters
        print(f"s0_{tag}: {len(terms)} terms, out={out_idx}")
        funcs.append(assemble_block(
            f's0_{tag}', _shape_expr(out_idx), terms, out_idx,
            f"S^(0) {tag}-{tag} anticommutator overlap <Phi0|{{C_bra^dag, C_ket}}|Phi0> "
            "(must gather to the identity on ordered configs)."))

    _write_module('adc2_blocks.py', funcs)


# =====================================================================
# Restricted (spin-blocked) generation -- increment 1 of the spin-free
# ADC(2)-(5) plan.
#
# Every spin-orbital superoperator element above is re-derived once per
# Ms-conserving spin assignment of its FREE indices via pq.strings(
# spin_labels=...), exactly as generate_mpn_density_restricted.py does for
# the MPn amplitude/density pieces (the restricted MPn pipeline is the
# template; a probe confirmed spin_labels works on these elements). The
# result operates on
# spatial-MO spin-block tensors (g_aaaa/g_abab/g_bbbb antisymmetrized-
# within-spin / bare, f_aa/f_bb Fock, d_aa/d_bb the spin-block Kronecker
# delta -- see src.Base.pyscf_interface.get_antisymmetrized_spin_block_eri
# for g's convention) instead of the full (2*nmo)-dim spin-orbital tensors.
#
# Spin adaptation stays NUMERICAL (spin_adapt.py's isometry, increment 5):
# these blocks are ms-resolved spatial tensors, NOT doublet CSFs. A block
# function is emitted per NONZERO free-index spin pattern (pdaggerq prunes
# the Ms-violating ones to 0 terms, skipped with a printout); the runtime
# driver picks the pattern matching each determinant config. Closed-shell
# twins (g_bbbb/f_bb/d_bb == g_aaaa/f_aa/d_aa) are aliased at call time by
# the driver, not separately derived.
# =====================================================================

_R_LEAVES = ['f_aa', 'f_bb', 'g_aaaa', 'g_abab', 'g_bbbb', 'd_aa', 'd_bb']
# DF/RI twin: the 4-index g blocks are replaced by the 3-index B factors that
# df_codegen.dress_integral_factor emits (<pq||rs> -> sum_Q B[Q,p,r] B[Q,q,s]).
_R_LEAVES_DF = ['f_aa', 'f_bb', 'B_aa', 'B_bb', 'd_aa', 'd_bb']


def assemble_block_restricted(func_name, out_shape_expr, raw_terms, out_indices,
                              doc, extra_args=(), df=False):
    """assemble_block's restricted twin: fixed spin-blocked leaf args
    (_R_LEAVES) instead of ('f', 'g', 'kd'). raw_terms_to_einsum already
    parses the '_aaaa'/'_abab' factor suffixes (pdaggerq_interface gotcha
    handling); dead leaf/amplitude args are harmless.

    df=True composes the DF/RI term-rewrite before emission, exactly as
    sigma_codegen.assemble_apply_block(df=True) does: each <pq||rs> becomes a
    pair of 3-index B factors and the leaf signature takes B_aa/B_bb in place
    of g_aaaa/g_abab/g_bbbb. Pure composition -- dress_integral_factor returns
    terms in the same pdaggerq format, and _array_expr already knows to leave
    B's auxiliary axis un-sliced."""
    if df:
        raw_terms = dress_integral_factor(raw_terms)
    args = list(_R_LEAVES_DF if df else _R_LEAVES) + list(extra_args) + ['o', 'v', 'no', 'nv']
    src = [f"def {func_name}({', '.join(args)}):"]
    src.append(f'    """{doc}"""')
    src.append(f"    out = np.zeros({out_shape_expr})")
    src.extend(raw_terms_to_einsum(raw_terms, 'out', out_indices))
    src.append("    return out")
    src.append("")
    return "\n".join(src)


import importlib
import re as _re
from src.SingleReference.ADC.adc_order_rule import (
    new_structures_at, u_order_terms, s_order_terms, s_structures_up_to,
    cross_block_leading_order, refuse_above_max_order)


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

_AMP_LEAF = _re.compile(r'^([tl]\d+_[ab]+)\(')


def _amp_leaves(raw_terms):
    """Sorted spin-blocked amplitude leaf names (t2_abab, l1_aa, ...) a raw
    pq.strings() output references -- the restricted split function's extra
    arguments (a spin pattern's exact leaf set is pdaggerq's decision, so it
    is detected, not hard-coded; the runtime driver introspects the
    signature the same way the spin-orbital driver does)."""
    leaves = set()
    for term in raw_terms:
        for fac in term[1:]:
            m = _AMP_LEAF.match(fac.strip())
            if m:
                leaves.add(m.group(1))
    return sorted(leaves)


def _emit_spin_blocks(base_name, term_fn, out_idx, doc_fn, extra_args=None, df=False):
    """Emit one restricted block function per Ms-conserving spin assignment
    of the free indices out_idx: term_fn(spin_labels) -> raw pq.strings()
    terms (empty for a pruned/Ms-violating assignment), doc_fn(suffix,
    spin_labels) -> the function docstring. Suffix is the concatenated
    a/b spins of out_idx (a unique key per assignment). Amplitude leaf args
    are auto-detected per pattern (extra_args, if given, is a fixed prefix
    unioned in)."""
    from itertools import product as _iproduct
    fixed = list(extra_args) if extra_args else []
    funcs = []
    for spins in _iproduct('ab', repeat=len(out_idx)):
        sl = dict(zip(out_idx, spins))
        terms = term_fn(sl)
        if not terms:
            continue
        suffix = ''.join(spins)
        name = f'{base_name}_{suffix}'
        amp_args = [a for a in _amp_leaves(terms) if a not in fixed]
        print(f"{name}: {len(terms)} terms, out={out_idx}, amps={amp_args}")
        funcs.append(assemble_block_restricted(
            name, _shape_expr(out_idx), terms, out_idx, doc_fn(suffix, sl),
            extra_args=fixed + amp_args, df=df))
    return funcs


def generate_adc2_restricted(df=False):
    """Restricted (spin-blocked) ADC(2) blocks: U^(1), K, C^(1), S^(0).

    df=True emits the DF/RI twin (adc2_blocks_restricted_df.py), whose leaves
    are the 3-index B_aa/B_bb factors instead of the 4-index g_ssss -- the
    same rewrite generate_adc3_u_restricted(df=True) applies to U^(2)/S^(2).
    Both twins are needed to build the ADC(3) coupling without ever
    materializing a norb^4 integral tensor (aug-cc-pVQZ: 15-115 GB)."""
    funcs = []

    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts

        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        # --- U^(1) orbital-class coupling, spin-blocked ---
        for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
            out_idx = ket_letters + [p_letter]
            funcs += _emit_spin_blocks(
                f'u1_{tag}_{row}',
                lambda sl, pl=p_letter: superop_element_terms(
                    p_tok(pl), ket, sector, hams=('f', 'v'), spin_labels=sl),
                out_idx,
                lambda suffix, sl, row=row, oi=out_idx: (
                    f"U^(1) {tag}/orbital coupling, {row} rows, spin block "
                    f"{suffix} (axes {', '.join(oi)})."), df=df)

        # --- K (f) and C^(1) (v) class-class blocks, spin-blocked ---
        for ham, name in (('f', f'k0_{tag}'), ('v', f'c1_{tag}')):
            out_idx = bra_letters + ket_letters
            funcs += _emit_spin_blocks(
                name,
                lambda sl, h=ham: superop_element_terms(
                    bra_dag, ket, sector, hams=(h,), spin_labels=sl),
                out_idx,
                lambda suffix, sl, nm=name, oi=out_idx: (
                    f"{nm} class-class block, spin block {suffix} "
                    f"(axes {', '.join(oi)})."), df=df)

        # --- S^(0) overlap, spin-blocked (sanity) ---
        out_idx = bra_letters + ket_letters
        funcs += _emit_spin_blocks(
            f's0_{tag}',
            lambda sl: overlap_element_terms(bra_dag, ket, spin_labels=sl),
            out_idx,
            lambda suffix, sl, oi=out_idx: (
                f"S^(0) {tag} anticommutator overlap, spin block {suffix} "
                f"(axes {', '.join(oi)})."), df=df)

    _write_module('adc2_blocks_restricted%s.py' % ('_df' if df else ''),
                  funcs)


def generate_adc3_u():
    """Order-2 U blocks (the ADC(3)-level U^(2) coupling corrections) --
    Phase 3's falsification test. Every order-2 dressing split of the
    superoperator U element is emitted MECHANICALLY (pdaggerq prunes the
    zero ones -- splits returning 0 terms are skipped, with a printout, so
    the pruning is visible and auditable rather than pre-decided by hand):

      m1=0, m2=1, v:  <Phi0|{p, [v, C_J]} t2|Phi0>            (_vt2)
      m1=1, m2=0, v:  <l2 Phi0|{p, [v, C_J]}|Phi0>            (_l2v)
      m1=1, m2=1, f:  <l2 Phi0|{p, [f, C_J]} t2|Phi0>         (_l2ft2)
      m1=0, m2=2, f:  <Phi0|{p, [f, C_J]} t{r}|Phi0>, r=1,2,3 (_ft{r})
      m1=2, m2=0, f:  <l{r} Phi0|{p, [f, C_J]}|Phi0>          (_l{r}f)

    (IP sector: [C_J, .] commutator, see superop_element_terms.) Amplitude
    ORDER is the runtime driver's choice (generate_mpn_density.py's
    design): _vt2/_l2v/_l2ft2 take first-order t2/l2; _ft{r}/_l{r}f take
    SECOND-order t{r}^(2)/l{r}^(2) arrays (that's what makes them order-2
    splits). Since U^(0)=0 (canonical) and S^(1)=0 identically (particle
    counting), the Loewdin layer
    contributes nothing at this order and U_ortho^(2) = the sum of these
    raw splits: the comparison against the hand-written spin-orbital ADC's validated
    U^(2) (term0+corr extracted from build_supermatrix) is direct.
    """
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        ket_letters = ket_holes + ket_parts
        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        splits = [
            ('vt2', ('v',), None, ['t2'], ['t2']),
            ('l2v', ('v',), [['l2']], None, ['l2']),
            ('l2ft2', ('f',), [['l2']], ['t2'], ['l2', 't2']),
        ]
        for r in (1, 2, 3):
            splits.append((f'ft{r}', ('f',), None, [f't{r}'], [f't{r}']))
            splits.append((f'l{r}f', ('f',), [[f'l{r}']], None, [f'l{r}']))

        for split_name, hams, left_ops, ket_dressing, extra_args in splits:
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                terms = superop_element_terms_dressed(
                    p_tok(p_letter), ket, sector, hams,
                    left_ops=left_ops, ket_dressing=ket_dressing)
                name = f'u2_{tag}_{row}_{split_name}'
                if not terms:
                    print(f"{name}: 0 terms (pruned)")
                    continue
                out_idx = ket_letters + [p_letter]
                print(f"{name}: {len(terms)} terms, out={out_idx}")
                funcs.append(assemble_block(
                    name, _shape_expr(out_idx), terms, out_idx,
                    f"U^(2) {tag} split {split_name} ({row} rows): dressing "
                    f"left={left_ops}, ket={ket_dressing}, hams={hams}; amplitude "
                    f"ORDER is the driver's choice (first-order for vt2/l2v/l2ft2, "
                    f"second-order for ft*/l*f). Axes ({', '.join(out_idx)}).",
                    extra_args=extra_args))

        # --- S^(2) orbital-class overlap blocks (Loewdin input at order 2:
        #     the cross-block S^(2)_{pJ} that feeds U_ortho^(2) corrections;
        #     same mechanical order-2 dressing splits, no Hamiltonian) ---
        s2_splits = [('l2t2', [['l2']], ['t2'], ['l2', 't2'])]
        for r in (1, 2, 3):
            s2_splits.append((f't{r}', None, [f't{r}'], [f't{r}']))
            s2_splits.append((f'l{r}', [[f'l{r}']], None, [f'l{r}']))
        for split_name, left_ops, ket_dressing, extra_args in s2_splits:
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                terms = overlap_element_terms(
                    p_tok(p_letter), ket, left_ops=left_ops,
                    ket_dressing=ket_dressing)
                name = f's2_{tag}_{row}_{split_name}'
                if not terms:
                    print(f"{name}: 0 terms (pruned)")
                    continue
                out_idx = ket_letters + [p_letter]
                print(f"{name}: {len(terms)} terms, out={out_idx}")
                funcs.append(assemble_block(
                    name, _shape_expr(out_idx), terms, out_idx,
                    f"S^(2) {tag}/orbital overlap split {split_name} ({row} rows): "
                    f"<...|{{a_p, C_{tag}}}|...> dressed left={left_ops}, "
                    f"ket={ket_dressing}; amplitude order: first for l2t2, second "
                    f"for t*/l*. Axes ({', '.join(out_idx)}).",
                    extra_args=extra_args))

    _write_module('adc3_u_blocks.py', funcs)


def _u2_split_defs():
    """The order-2 dressed U/C^(2) splits (name, hams, left_ops,
    ket_dressing) -- shared by the spin-orbital and restricted generators
    so they stay in lockstep."""
    splits = [
        ('vt2', ('v',), None, ['t2']),
        ('l2v', ('v',), [['l2']], None),
        ('l2ft2', ('f',), [['l2']], ['t2']),
    ]
    for r in (1, 2, 3):
        splits.append((f'ft{r}', ('f',), None, [f't{r}']))
        splits.append((f'l{r}f', ('f',), [[f'l{r}']], None))
    return splits


def _s2_split_defs():
    s2 = [('l2t2', [['l2']], ['t2'])]
    for r in (1, 2, 3):
        s2.append((f't{r}', None, [f't{r}']))
        s2.append((f'l{r}', [[f'l{r}']], None))
    return s2


def generate_adc3_u_restricted(df=False):
    """Restricted (spin-blocked) order-2 U^(2)/S^(2) orbital<->k=1 dressing
    splits -- increment 3 of the spin-free plan, the spin-blocked twin of
    generate_adc3_u. Same mechanical splits (_u2_split_defs/_s2_split_defs),
    now re-derived per Ms-conserving free-index spin assignment via
    spin_labels; the dressed amplitude leaves become spin-blocked
    (t2_aaaa/t2_abab/l2_aaaa/...), auto-detected as each function's extra
    args (see _emit_spin_blocks). The runtime driver feeds them restricted
    MPn amplitude arrays at the split's amplitude order (same order
    bookkeeping as the spin-orbital driver)."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        ket_letters = ket_holes + ket_parts
        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        for split_name, hams, left_ops, ket_dressing in _u2_split_defs():
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                funcs += _emit_spin_blocks(
                    f'u2_{tag}_{row}_{split_name}',
                    lambda sl, pl=p_letter, h=hams, lo=left_ops, kd=ket_dressing:
                        superop_element_terms_dressed(
                            p_tok(pl), ket, sector, h, left_ops=lo,
                            ket_dressing=kd, spin_labels=sl),
                    out_idx,
                    lambda suffix, sl, sn=split_name, row=row, oi=out_idx: (
                        f"U^(2) {tag} split {sn} ({row} rows), spin block "
                        f"{suffix} (axes {', '.join(oi)})."), df=df)

        for split_name, left_ops, ket_dressing in _s2_split_defs():
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                funcs += _emit_spin_blocks(
                    f's2_{tag}_{row}_{split_name}',
                    lambda sl, pl=p_letter, lo=left_ops, kd=ket_dressing:
                        overlap_element_terms(
                            p_tok(pl), ket, left_ops=lo, ket_dressing=kd,
                            spin_labels=sl),
                    out_idx,
                    lambda suffix, sl, sn=split_name, row=row, oi=out_idx: (
                        f"S^(2) {tag}/orbital overlap split {sn} ({row} rows), "
                        f"spin block {suffix} (axes {', '.join(oi)})."), df=df)

    _write_module('adc3_u_blocks_restricted%s.py' % ('_df' if df else ''), funcs)


def generate_adc4_base_c2():
    """Order-2 class-class (k=1) blocks -- the generator's ARBITER role for
    the disputed ADC(4) C^(2)
    (the lambda-oracle REJECTED the printed SCW83 B7 equations; on H2/sto-3g
    the exact lambda^4 residue needs NO C^(2)-fold at all). Same mechanical
    order-2 dressing splits as generate_adc3_u, now with bra AND ket both
    k=1 classes, plus the within-class S^(2) overlap splits feeding the
    Loewdin correction (isr_engine.assemble_kplusc applies
    -1/2(S2 K + K S2) at runtime) and the N^(2) normalization subtraction
    (<Psi|X|Psi>/<Psi|Psi> at order 2 = raw - N2 * K; the disconnected
    <l2 t2><X> piece survives pdaggerq's simplify, same as in the MPn
    density recursion's c^(n) machinery)."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        out_idx = (bra_holes + bra_parts) + (ket_holes + ket_parts)

        splits = [
            ('vt2', ('v',), None, ['t2'], ['t2']),
            ('l2v', ('v',), [['l2']], None, ['l2']),
            ('l2ft2', ('f',), [['l2']], ['t2'], ['l2', 't2']),
        ]
        for r in (1, 2, 3):
            splits.append((f'ft{r}', ('f',), None, [f't{r}'], [f't{r}']))
            splits.append((f'l{r}f', ('f',), [[f'l{r}']], None, [f'l{r}']))

        for split_name, hams, left_ops, ket_dressing, extra_args in splits:
            terms = superop_element_terms_dressed(
                bra_dag, ket, sector, hams,
                left_ops=left_ops, ket_dressing=ket_dressing)
            name = f'c2_{tag}_{split_name}'
            if not terms:
                print(f"{name}: 0 terms (pruned)")
                continue
            print(f"{name}: {len(terms)} terms, out={out_idx}")
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"C^(2) {tag}-{tag} split {split_name}: dressing left={left_ops}, "
                f"ket={ket_dressing}, hams={hams}; amplitude order: first for "
                f"vt2/l2v/l2ft2, second for ft*/l*f. Axes ({', '.join(out_idx)}).",
                extra_args=extra_args))

        # within-class S^(2) overlap splits (Loewdin input)
        s2_splits = [('l2t2', [['l2']], ['t2'], ['l2', 't2'])]
        for r in (1, 2, 3):
            s2_splits.append((f't{r}', None, [f't{r}'], [f't{r}']))
            s2_splits.append((f'l{r}', [[f'l{r}']], None, [f'l{r}']))
        for split_name, left_ops, ket_dressing, extra_args in s2_splits:
            terms = overlap_element_terms(bra_dag, ket, left_ops=left_ops,
                                          ket_dressing=ket_dressing)
            name = f's2c_{tag}_{split_name}'
            if not terms:
                print(f"{name}: 0 terms (pruned)")
                continue
            print(f"{name}: {len(terms)} terms, out={out_idx}")
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"S^(2) {tag}-{tag} within-class overlap split {split_name}: "
                f"left={left_ops}, ket={ket_dressing}; amplitude order: first "
                f"for l2t2, second for t*/l*. Axes ({', '.join(out_idx)}).",
                extra_args=extra_args))

    _write_module('adc4_c2_blocks.py', funcs)


def generate_adc4_base_c2_restricted():
    """Restricted (spin-blocked) order-2 class-class C^(2) + within-class
    S^(2) blocks -- increment 4 of the spin-free plan, the spin-blocked twin
    of generate_adc4_base_c2 (bra AND ket k=1 classes). Same mechanical splits
    (_u2_split_defs/_s2_split_defs), re-derived per Ms-conserving 6-free-index
    spin assignment; amplitude leaves auto-detected. The runtime driver
    applies the SAME ISR assembly (N^(2) normalization, Loewdin
    assemble_kplusc) on the alpha-sector-gathered blocks."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        out_idx = (bra_holes + bra_parts) + (ket_holes + ket_parts)

        for split_name, hams, left_ops, ket_dressing in _u2_split_defs():
            funcs += _emit_spin_blocks(
                f'c2_{tag}_{split_name}',
                lambda sl, h=hams, lo=left_ops, kd=ket_dressing:
                    superop_element_terms_dressed(
                        bra_dag, ket, sector, h, left_ops=lo,
                        ket_dressing=kd, spin_labels=sl),
                out_idx,
                lambda suffix, sl, sn=split_name, oi=out_idx: (
                    f"C^(2) {tag}-{tag} split {sn}, spin block {suffix} "
                    f"(axes {', '.join(oi)})."))

        for split_name, left_ops, ket_dressing in _s2_split_defs():
            funcs += _emit_spin_blocks(
                f's2c_{tag}_{split_name}',
                lambda sl, lo=left_ops, kd=ket_dressing:
                    overlap_element_terms(
                        bra_dag, ket, left_ops=lo, ket_dressing=kd,
                        spin_labels=sl),
                out_idx,
                lambda suffix, sl, sn=split_name, oi=out_idx: (
                    f"S^(2) {tag}-{tag} within-class overlap split {sn}, spin "
                    f"block {suffix} (axes {', '.join(oi)})."))

    _write_module('adc4_c2_blocks_restricted.py', funcs)


def generate_adc4_base_k2():
    """Remaining ADC(4)-base blocks involving the k=2 classes (3h2p/3p2h):

    1. First-order k=1 <-> k=2 coupling <{C_k1^dag, [C_k2, H]}> (IP) /
       [H, C_k2] (EA) -- the extended-space C^(1) family (adc4_base's
       _coupling_2h1p_3h2p_block, validated, is the reference).
    2. Order-2 orbital <-> k=2 U couplings (U^(2)_3h2p/3p2h) -- same
       mechanical dressing splits as generate_adc3_u (U^(1)_k2 = 0 by
       operator counting: 4 uncontractable virtual creators), plus the
       orbital/k=2 S^(2) overlap splits feeding the class-GS correction
       U = U_raw^(2) - eps_p * S^(2)_pJ (the representation identified in
       Phase 3). ARBITER TARGET #2: adc4_base's printed U^(2)_3h2p shares its
       (revoked) validation history with the rejected C^(2); the
       lambda-oracle work derived its own 5-line replacement
       (verify_derived_u2_5line.derived_u2) to compare against.

    SIZE NOTE: k=2 class-class blocks (10-index dense) are deliberately
    NOT generated -- K_3h2p/K_3p2h diagonals are plain eps sums (runtime),
    and second-order k=2 class blocks are beyond ADC(4)-base.
    """
    funcs = []
    for sector, tag1, tag2 in (('IP', '2h1p', '3h2p'), ('EA', '2p1h', '3p2h')):
        conv1 = class_index_convention(1, sector)
        # bra = k=1 class at offset 0; ket = k=2 class at offsets past it
        bra_dag = precursor_bra_string(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(2, sector,
                                             occ_offset=conv1['nh'],
                                             vir_offset=conv1['np'])
        ket = precursor_ket_string(2, sector, occ_offset=conv1['nh'],
                                   vir_offset=conv1['np'])
        out_idx = (bra_holes + bra_parts) + (ket_holes + ket_parts)
        terms = superop_element_terms(bra_dag, ket, sector, hams=('f', 'v'))
        name = f'c1_{tag1}_{tag2}'
        print(f"{name}: {len(terms)} terms, out={out_idx}")
        funcs.append(assemble_block(
            name, _shape_expr(out_idx), terms, out_idx,
            f"First-order {tag1}<->{tag2} coupling "
            f"<Phi0|{{C_{tag1}^dag, [{'C,H' if sector == 'IP' else 'H,C'}]}}|Phi0>, "
            f"axes = bra ({', '.join(bra_holes + bra_parts)}) x ket "
            f"({', '.join(ket_holes + ket_parts)})."))

        # --- U^(2) orbital <-> k=2 + its S^(2) splits ---
        conv2 = class_index_convention(2, sector)
        k2_holes, k2_parts = class_letters(2, sector)  # k=2 at offset 0 now
        ket2 = precursor_ket_string(2, sector)
        ket2_letters = k2_holes + k2_parts
        p_occ = OCC_LETTERS[conv2['nh']]
        p_vir = VIR_LETTERS[conv2['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        u_splits = [
            ('vt2', ('v',), None, ['t2'], ['t2']),
            ('l2v', ('v',), [['l2']], None, ['l2']),
            ('l2ft2', ('f',), [['l2']], ['t2'], ['l2', 't2']),
        ]
        for r in (1, 2, 3):
            u_splits.append((f'ft{r}', ('f',), None, [f't{r}'], [f't{r}']))
            u_splits.append((f'l{r}f', ('f',), [[f'l{r}']], None, [f'l{r}']))
        for split_name, hams, left_ops, ket_dressing, extra_args in u_splits:
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                terms = superop_element_terms_dressed(
                    p_tok(p_letter), ket2, sector, hams,
                    left_ops=left_ops, ket_dressing=ket_dressing)
                name = f'u2_{tag2}_{row}_{split_name}'
                if not terms:
                    print(f"{name}: 0 terms (pruned)")
                    continue
                out_idx = ket2_letters + [p_letter]
                print(f"{name}: {len(terms)} terms, out={out_idx}")
                funcs.append(assemble_block(
                    name, _shape_expr(out_idx), terms, out_idx,
                    f"U^(2) {tag2}/orbital split {split_name} ({row} rows); "
                    f"amplitude order: first for vt2/l2v/l2ft2, second for "
                    f"ft*/l*f. Axes ({', '.join(out_idx)}).",
                    extra_args=extra_args))

        s2_splits = [('l2t2', [['l2']], ['t2'], ['l2', 't2'])]
        for r in (1, 2, 3):
            s2_splits.append((f't{r}', None, [f't{r}'], [f't{r}']))
            s2_splits.append((f'l{r}', [[f'l{r}']], None, [f'l{r}']))
        for split_name, left_ops, ket_dressing, extra_args in s2_splits:
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                terms = overlap_element_terms(
                    p_tok(p_letter), ket2, left_ops=left_ops,
                    ket_dressing=ket_dressing)
                name = f's2_{tag2}_{row}_{split_name}'
                if not terms:
                    print(f"{name}: 0 terms (pruned)")
                    continue
                out_idx = ket2_letters + [p_letter]
                print(f"{name}: {len(terms)} terms, out={out_idx}")
                funcs.append(assemble_block(
                    name, _shape_expr(out_idx), terms, out_idx,
                    f"S^(2) {tag2}/orbital overlap split {split_name} ({row} "
                    f"rows). Axes ({', '.join(out_idx)}).",
                    extra_args=extra_args))

    _write_module('adc4_k2_blocks.py', funcs)


def generate_adc4_base_k2_restricted():
    """Restricted (spin-blocked) k=2 (3h2p/3p2h) extended-space blocks --
    ADC(5) push increment A, the spin-blocked twin of generate_adc4_base_k2:

    1. First-order k=1<->k=2 coupling c1_{tag1}_{tag2} (bra k=1, ket k=2),
       re-derived per Ms-conserving spin assignment of its 8 free indices.
    2. Order-2 orbital<->k=2 U^(2) + S^(2) dressed splits
       (_u2_split_defs/_s2_split_defs), spin-blocked, amplitude leaves
       auto-detected.

    The runtime driver gathers these at the alpha-sector k=2 configs and
    applies the SAME _K2_CLASS_SIGN / class-GS assembly as the spin-orbital
    driver's coupling_k1_k2_generated/u2_k2_generated."""
    funcs = []
    for sector, tag1, tag2 in (('IP', '2h1p', '3h2p'), ('EA', '2p1h', '3p2h')):
        conv1 = class_index_convention(1, sector)
        bra_dag = precursor_bra_string(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(2, sector,
                                             occ_offset=conv1['nh'],
                                             vir_offset=conv1['np'])
        ket = precursor_ket_string(2, sector, occ_offset=conv1['nh'],
                                   vir_offset=conv1['np'])
        out_idx = (bra_holes + bra_parts) + (ket_holes + ket_parts)
        funcs += _emit_spin_blocks(
            f'c1_{tag1}_{tag2}',
            lambda sl: superop_element_terms(bra_dag, ket, sector,
                                             hams=('f', 'v'), spin_labels=sl),
            out_idx,
            lambda suffix, sl, t1=tag1, t2=tag2, oi=out_idx: (
                f"First-order {t1}<->{t2} coupling, spin block {suffix} "
                f"(axes {', '.join(oi)})."))

        # --- U^(2) orbital <-> k=2 + S^(2) splits, spin-blocked ---
        conv2 = class_index_convention(2, sector)
        k2_holes, k2_parts = class_letters(2, sector)
        ket2 = precursor_ket_string(2, sector)
        ket2_letters = k2_holes + k2_parts
        p_occ = OCC_LETTERS[conv2['nh']]
        p_vir = VIR_LETTERS[conv2['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        for split_name, hams, left_ops, ket_dressing in _u2_split_defs():
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                oi = ket2_letters + [p_letter]
                funcs += _emit_spin_blocks(
                    f'u2_{tag2}_{row}_{split_name}',
                    lambda sl, pl=p_letter, h=hams, lo=left_ops, kd=ket_dressing:
                        superop_element_terms_dressed(
                            p_tok(pl), ket2, sector, h, left_ops=lo,
                            ket_dressing=kd, spin_labels=sl),
                    oi,
                    lambda suffix, sl, sn=split_name, row=row, o=oi, t2=tag2: (
                        f"U^(2) {t2}/orbital split {sn} ({row} rows), spin "
                        f"block {suffix} (axes {', '.join(o)})."))

        for split_name, left_ops, ket_dressing in _s2_split_defs():
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                oi = ket2_letters + [p_letter]
                funcs += _emit_spin_blocks(
                    f's2_{tag2}_{row}_{split_name}',
                    lambda sl, pl=p_letter, lo=left_ops, kd=ket_dressing:
                        overlap_element_terms(
                            p_tok(pl), ket2, left_ops=lo, ket_dressing=kd,
                            spin_labels=sl),
                    oi,
                    lambda suffix, sl, sn=split_name, row=row, o=oi, t2=tag2: (
                        f"S^(2) {t2}/orbital overlap split {sn} ({row} rows), "
                        f"spin block {suffix} (axes {', '.join(o)})."))

    _write_module('adc4_k2_blocks_restricted.py', funcs)


def generate_adc4_u3():
    """Order-3 orbital<->k=1 U couplings (U^(3)_2p1h/2h1p) -- Phase 5,
    arbiter target #3 (the lambda-oracle REJECTED the printed SCW83 B3
    23-term U^(3)_2p1h; on H2/sto-3g the exact lambda^4 requires 0.0126
    where B3 gives 0.0671).

    Only NEW tag structures are emitted here -- pdaggerq is order-blind, so
    any tag structure already generated in adc3_u_blocks.py (l2v, l1f,
    l2ft2, ...) is REUSED by the runtime driver at higher amplitude orders
    rather than re-derived. New structures at order 3: the both-side v
    element l2vt2, the mixed-rank f elements l2ft1/l2ft3/l1ft2/l3ft2, and
    the corresponding overlap structures (sl2t1/sl2t3/sl1t2/sl3t2 etc.)
    feeding S^(3). Amplitude ORDERS are the driver's choice throughout
    (generate_mpn_density design)."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        ket_letters = ket_holes + ket_parts
        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        # (name, hams, left_ops, ket_dressing, extra_args) -- new structures only
        u_splits = [
            ('l2vt2', ('v',), [['l2']], ['t2'], ['l2', 't2']),
            ('vt1', ('v',), None, ['t1'], ['t1']),
            ('vt3', ('v',), None, ['t3'], ['t3']),
            ('l1v', ('v',), [['l1']], None, ['l1']),
            ('l3v', ('v',), [['l3']], None, ['l3']),
            ('l2ft1', ('f',), [['l2']], ['t1'], ['l2', 't1']),
            ('l2ft3', ('f',), [['l2']], ['t3'], ['l2', 't3']),
            ('l1ft2', ('f',), [['l1']], ['t2'], ['l1', 't2']),
            ('l3ft2', ('f',), [['l3']], ['t2'], ['l3', 't2']),
        ]
        for split_name, hams, left_ops, ket_dressing, extra_args in u_splits:
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                terms = superop_element_terms_dressed(
                    p_tok(p_letter), ket, sector, hams,
                    left_ops=left_ops, ket_dressing=ket_dressing)
                name = f'u3_{tag}_{row}_{split_name}'
                if not terms:
                    print(f"{name}: 0 terms (pruned)")
                    continue
                out_idx = ket_letters + [p_letter]
                print(f"{name}: {len(terms)} terms, out={out_idx}")
                funcs.append(assemble_block(
                    name, _shape_expr(out_idx), terms, out_idx,
                    f"U^(3)-family {tag} tag structure {split_name} ({row} rows); "
                    f"amplitude orders are the driver's choice. Axes ({', '.join(out_idx)}).",
                    extra_args=extra_args))

        s_splits = [
            ('l2t1', [['l2']], ['t1'], ['l2', 't1']),
            ('l2t3', [['l2']], ['t3'], ['l2', 't3']),
            ('l1t2', [['l1']], ['t2'], ['l1', 't2']),
            ('l3t2', [['l3']], ['t2'], ['l3', 't2']),
        ]
        for split_name, left_ops, ket_dressing, extra_args in s_splits:
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                terms = overlap_element_terms(
                    p_tok(p_letter), ket, left_ops=left_ops,
                    ket_dressing=ket_dressing)
                name = f's3_{tag}_{row}_{split_name}'
                if not terms:
                    print(f"{name}: 0 terms (pruned)")
                    continue
                out_idx = ket_letters + [p_letter]
                print(f"{name}: {len(terms)} terms, out={out_idx}")
                funcs.append(assemble_block(
                    name, _shape_expr(out_idx), terms, out_idx,
                    f"S^(3)-family {tag}/orbital overlap tag structure {split_name} "
                    f"({row} rows). Axes ({', '.join(out_idx)}).",
                    extra_args=extra_args))

    _write_module('adc4_u3_blocks.py', funcs)


def _u3_split_defs():
    """New order-3 U^(3) tag structures (name, hams, left_ops, ket_dressing)
    -- shared by the spin-orbital and restricted U^(3) generators."""
    return [
        ('l2vt2', ('v',), [['l2']], ['t2']),
        ('vt1', ('v',), None, ['t1']),
        ('vt3', ('v',), None, ['t3']),
        ('l1v', ('v',), [['l1']], None),
        ('l3v', ('v',), [['l3']], None),
        ('l2ft1', ('f',), [['l2']], ['t1']),
        ('l2ft3', ('f',), [['l2']], ['t3']),
        ('l1ft2', ('f',), [['l1']], ['t2']),
        ('l3ft2', ('f',), [['l3']], ['t2']),
    ]


def _s3_split_defs():
    return [
        ('l2t1', [['l2']], ['t1']),
        ('l2t3', [['l2']], ['t3']),
        ('l1t2', [['l1']], ['t2']),
        ('l3t2', [['l3']], ['t2']),
    ]


def generate_adc4_u3_restricted(df=False):
    """Restricted (spin-blocked) new-order-3 U^(3)/S^(3) tag structures --
    increment 4 of the spin-free plan, spin-blocked twin of generate_adc4_u3.
    Same NEW tag structures (_u3_split_defs/_s3_split_defs); the driver reuses
    the already-generated restricted u2/s2 splits (adc3_u_blocks_restricted)
    at higher amplitude orders, same as the spin-orbital U^(3) driver."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        ket_letters = ket_holes + ket_parts
        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        for split_name, hams, left_ops, ket_dressing in _u3_split_defs():
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                funcs += _emit_spin_blocks(
                    f'u3_{tag}_{row}_{split_name}',
                    lambda sl, pl=p_letter, h=hams, lo=left_ops, kd=ket_dressing:
                        superop_element_terms_dressed(
                            p_tok(pl), ket, sector, h, left_ops=lo,
                            ket_dressing=kd, spin_labels=sl),
                    out_idx,
                    lambda suffix, sl, sn=split_name, row=row, oi=out_idx: (
                        f"U^(3) {tag} tag structure {sn} ({row} rows), spin "
                        f"block {suffix} (axes {', '.join(oi)})."), df=df)

        for split_name, left_ops, ket_dressing in _s3_split_defs():
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                funcs += _emit_spin_blocks(
                    f's3_{tag}_{row}_{split_name}',
                    lambda sl, pl=p_letter, lo=left_ops, kd=ket_dressing:
                        overlap_element_terms(
                            p_tok(pl), ket, left_ops=lo, ket_dressing=kd,
                            spin_labels=sl),
                    out_idx,
                    lambda suffix, sl, sn=split_name, row=row, oi=out_idx: (
                        f"S^(3) {tag}/orbital overlap tag structure {sn} ({row} "
                        f"rows), spin block {suffix} (axes {', '.join(oi)})."), df=df)

    _write_module('adc4_u3_blocks_restricted%s.py' % ('_df' if df else ''), funcs)


def generate_adc_u_restricted(u_order, max_rank=4, fname=None):
    """ORDER-GENERAL restricted U^(u_order)/S^(u_order) block generator -- the
    ADC(u_order+1) increment, driven entirely by adc_order_rule instead of a
    hand-written _uN_split_defs. `generate_adc_u_restricted(3)` is the
    order-general replacement for generate_adc4_u3_restricted;
    `generate_adc_u_restricted(4)` is ADC(5)'s U^(4).

    Emits the tag structures FIRST reachable at this order (lower-order ones
    already live in their own modules and are reused by the driver at higher
    amplitude orders). pdaggerq prunes the impossible ones automatically -- a
    structure that emits 0 blocks simply never gets a function, and the
    driver's getattr(..., None) skips it. See adc_order_rule's docstring for
    why the ORDER bookkeeping (u_order_terms) is the exactness-critical part
    and generation is only required to be a superset."""
    refuse_above_max_order(u_order + 1, 'generate_adc_u_restricted')
    u_new = new_structures_at(u_order, max_rank, kind='u')
    s_new = new_structures_at(u_order, max_rank, kind='s')
    print(f"--- U^({u_order}): {len(u_new)} new tag structures, "
          f"S^({u_order}): {len(s_new)} ---")
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        kh, kp = class_letters(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        ket_letters = kh + kp
        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        for nm, (j, ham, k) in sorted(u_new.items()):
            left_ops = [[f'l{j}']] if j else None
            ket_dressing = [f't{k}'] if k else None
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                funcs += _emit_spin_blocks(
                    f'u{u_order}_{tag}_{row}_{nm}',
                    lambda sl, pl=p_letter, h=(ham,), lo=left_ops, kd=ket_dressing:
                        superop_element_terms_dressed(
                            p_tok(pl), ket, sector, h, left_ops=lo,
                            ket_dressing=kd, spin_labels=sl),
                    out_idx,
                    lambda suffix, sl, s=nm, r=row, oi=out_idx, o=u_order: (
                        f"U^({o}) {tag} tag structure {s} ({r} rows), spin block "
                        f"{suffix} (axes {', '.join(oi)}). Amplitude orders are the "
                        f"driver's choice (adc_order_rule.u_order_terms)."))

        for nm, (j, k) in sorted(s_new.items()):
            left_ops = [[f'l{j}']] if j else None
            ket_dressing = [f't{k}'] if k else None
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                funcs += _emit_spin_blocks(
                    f's{u_order}_{tag}_{row}_{nm}',
                    lambda sl, pl=p_letter, lo=left_ops, kd=ket_dressing:
                        overlap_element_terms(
                            p_tok(pl), ket, left_ops=lo, ket_dressing=kd, spin_labels=sl),
                    out_idx,
                    lambda suffix, sl, s=nm, r=row, oi=out_idx, o=u_order: (
                        f"S^({o}) {tag}/orbital overlap tag structure {s} ({r} rows), "
                        f"spin block {suffix} (axes {', '.join(oi)})."))

    _write_module(fname or f'adc{u_order + 1}_u{u_order}_blocks_restricted.py', funcs)


def generate_adc5_c3():
    """SPIN-ORBITAL order-3 class-class C^(3) + within-class S^(3) blocks --
    the oracle reference for generate_adc5_c3_restricted (block-level
    spin-blocking gate). Same k=1 bra+ket classes and the same order-3 tag
    splits (_u3_split_defs/_s3_split_defs) as the restricted twin, emitted
    without spin_labels (full spin-orbital tensors)."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        out_idx = (bra_holes + bra_parts) + (ket_holes + ket_parts)

        def _amps(left_ops, ket_dressing):
            # explicit amplitude leaves (spin-orbital bare names) from the split
            # -- _amp_leaves only detects spin-LABELED names, so build directly.
            return [lo[0] for lo in (left_ops or [])] + list(ket_dressing or [])

        for split_name, hams, left_ops, ket_dressing in _u3_split_defs():
            terms = superop_element_terms_dressed(
                bra_dag, ket, sector, hams, left_ops=left_ops, ket_dressing=ket_dressing)
            name = f'c3_{tag}_{split_name}'
            if not terms:
                print(f"{name}: 0 terms (pruned)"); continue
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"C^(3) {tag}-{tag} split {split_name} (spin-orbital). "
                f"Axes ({', '.join(out_idx)}).",
                extra_args=_amps(left_ops, ket_dressing)))

        for split_name, left_ops, ket_dressing in _s3_split_defs():
            terms = overlap_element_terms(bra_dag, ket, left_ops=left_ops, ket_dressing=ket_dressing)
            name = f's3c_{tag}_{split_name}'
            if not terms:
                print(f"{name}: 0 terms (pruned)"); continue
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"S^(3) {tag}-{tag} within-class overlap split {split_name} "
                f"(spin-orbital). Axes ({', '.join(out_idx)}).",
                extra_args=_amps(left_ops, ket_dressing)))

    _write_module('adc5_c3_blocks.py', funcs)


def generate_adc5_c3_restricted():
    """Restricted ORDER-3 class-class C^(3) + within-class S^(3) blocks -- the
    ADC(5) analog of generate_adc4_base_c2_restricted (bra AND ket k=1 classes),
    using the order-3 tag splits (_u3_split_defs/_s3_split_defs) in place of
    order-2. Order-3 amplitudes already exist, so this piece is UNBLOCKED
    (increment 2a of the ADC(5) operator). The
    runtime driver applies the same ISR/Loewdin assembly as C^(2), one order up.
    Block-level codegen needs no order-4 amps (order-3 amps feed these directly);
    validate each block against the spin-orbital oracle at sliced amps."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        out_idx = (bra_holes + bra_parts) + (ket_holes + ket_parts)

        for split_name, hams, left_ops, ket_dressing in _u3_split_defs():
            funcs += _emit_spin_blocks(
                f'c3_{tag}_{split_name}',
                lambda sl, h=hams, lo=left_ops, kd=ket_dressing:
                    superop_element_terms_dressed(
                        bra_dag, ket, sector, h, left_ops=lo,
                        ket_dressing=kd, spin_labels=sl),
                out_idx,
                lambda suffix, sl, sn=split_name, oi=out_idx: (
                    f"C^(3) {tag}-{tag} split {sn}, spin block {suffix} "
                    f"(axes {', '.join(oi)})."))

        for split_name, left_ops, ket_dressing in _s3_split_defs():
            funcs += _emit_spin_blocks(
                f's3c_{tag}_{split_name}',
                lambda sl, lo=left_ops, kd=ket_dressing:
                    overlap_element_terms(
                        bra_dag, ket, left_ops=lo, ket_dressing=kd,
                        spin_labels=sl),
                out_idx,
                lambda suffix, sl, sn=split_name, oi=out_idx: (
                    f"S^(3) {tag}-{tag} within-class overlap split {sn}, spin "
                    f"block {suffix} (axes {', '.join(oi)})."))

    _write_module('adc5_c3_blocks_restricted.py', funcs)


def generate_adc5_k2k2():
    """SPIN-ORBITAL k=2 class-class self-coupling -- the ADC(5) (2,2) block.

    `adc_order_rule.block_order(n, 2, 2)` is 0 at ADC(4) (K only, diagonal) and
    **1 at ADC(5)**, so this first-order 3h2p x 3h2p / 3p2h x 3p2h coupling is
    the block that makes ADC(5) structurally different from ADC(4). The rule
    PREDICTS it rather than it being hand-noticed (section 12), and the same
    rule says the top manifold's self-block is order 0 for every EVEN n and 1
    for every ODD n -- so this is the generic odd-order object, not an ADC(5)
    special case.

    Two blocks per sector, exactly as generate_adc2 splits the k=1 class-class
    element:
      k0_{tag}  (f part)  -- zeroth order; must gather to the SAME diagonal as
                             sigma_driver.diag_K(..., k=2). That agreement is a
                             free end-to-end check on the class enumeration,
                             the letter offsets and the sector sign, so it is
                             worth emitting even though the diagonal is already
                             known analytically.
      c1_{tag}  (v part)  -- the new ADC(5) first-order self-coupling.

    SIZE -- READ BEFORE USING THE DENSE FORM. These are 10-index tensors
    (no^3 nv^2)^2 for IP. That is fine at validation scale (H4/sto-3g: 1M
    entries) and hopeless in production (H2O/sto-3g already ~256M entries,
    2 GB) -- which is why generate_adc4_base_k2 deliberately skipped k=2
    class-class blocks. The production representation MUST be the matrix-free
    sigma/apply form (sigma_codegen), with this dense twin kept only as the
    small-system arbiter, exactly as elsewhere in the generator.

    Index budget: 6 free occ + 4 free vir (IP). The historic 6-letter pool made
    that unspellable, which is what section 6 recorded as the "letter-budget
    risk" blocking this derivation. precursor_classes._label_pool is unbounded
    now, so the risk is stale: both sectors derive in <0.1 s, 15 f-terms and
    26 v-terms each.
    """
    funcs = []
    for sector, tag in (('IP', '3h2p'), ('EA', '3p2h')):
        conv = class_index_convention(2, sector)
        bra_holes, bra_parts = class_letters(2, sector)
        ket_holes, ket_parts = class_letters(2, sector,
                                             occ_offset=conv['nh'],
                                             vir_offset=conv['np'])
        bra_dag = precursor_bra_string(2, sector)
        ket = precursor_ket_string(2, sector, occ_offset=conv['nh'],
                                   vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        out_idx = bra_letters + ket_letters
        for ham, name in (('f', f'k0_{tag}'), ('v', f'c1_{tag}')):
            terms = superop_element_terms(bra_dag, ket, sector, hams=(ham,))
            print(f"{name}: {len(terms)} terms, out={out_idx}")
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"{'K (zeroth-order, f part)' if ham == 'f' else 'C^(1) (first-order, v part)'} "
                f"{tag}-{tag} self-coupling: "
                f"<Phi0|{{C_bra^dag, [{ham}, C_ket]}}|Phi0>, axes = bra "
                f"({', '.join(bra_letters)}) x ket ({', '.join(ket_letters)}). "
                f"ADC(5) block -- order_rule.block_order(5,2,2) == 1."))
    _write_module('adc5_k2k2_blocks.py', funcs)


def generate_adc5_u3_k2():
    """SPIN-ORBITAL order-3 orbital<->k=2 U^(3)/S^(3) tag structures -- the
    ADC(5) coupling block (`adc_order_rule.block_order(5, 0, 2) == 3`; only the
    order-2 twin existed, from generate_adc4_base_k2).

    Emission is ORDER-RULE-DRIVEN, no hand list: every tag structure the rule
    first reaches at orders 1..3 (`new_structures_at(o)`) is probed at the k=2
    precursor ket; pdaggerq prunes the impossible ones (0 terms, printed), and
    a structure already generated in adc4_k2_blocks.py (l2v/l1f/l2f/s2_l2,
    checked by hasattr -- the same reuse rule generate_adc4_u3 documents) is
    skipped so the runtime driver reuses it at higher amplitude orders.

    Census (2026-07-21 probe, both sectors, row-asymmetric as usual at k=2):
      kept NEW u: l1v, l3v, l2vt2, l2ft1, l3ft2, **l4ft2** (rank-4 lambda
                  survives at k=2 although it prunes at k=1 -- the 3h2p
                  manifold bridges it), and the order-1 undressed element
                  u1_v (occ rows IP / vir rows EA).
      kept NEW s: l4t2.
      reused    : l2v, l1f, l2f, s2_l2 (adc4_k2_blocks).
      pruned    : ft1-4, vt1-4, l1ft2, l2ft2, l2ft3, l2ft4, l3f, l4f, l4v;
                  s: l1, l3, l4, t1-4, l1t2, l2t1..l2t4, l3t2.

    The u1_v element deserves a note: generate_adc4_base_k2's docstring claims
    "U^(1)_k2 = 0 by operator counting", which is true of the EFFECTIVE
    coupling but NOT of the raw element -- the undressed v element is nonzero
    (2 terms) and completes the order-1 gauge layer raw^(1) = eps_p * S^(1)
    (l2f(l2^1) alone is K_J * S^(1)), which the class-GS then annihilates.
    The order-3 assembly needs raw^(1) explicitly for its N^(2) normalization
    (ndiv subtracts N^(2)*raw^(1) at order 3), which is why u1_v is emitted.

    Function names carry the structure's HOME order (u1_/u3_/s3_ prefixes);
    amplitude ORDERS stay the driver's choice (adc_autogenerated._k2_raw_S)."""
    _genk2mod = _generated_module('adc4', 'adc4_k2_blocks')
    funcs = []
    for sector, tag2 in (('IP', '3h2p'), ('EA', '3p2h')):
        conv2 = class_index_convention(2, sector)
        k2_holes, k2_parts = class_letters(2, sector)
        ket2 = precursor_ket_string(2, sector)
        ket2_letters = k2_holes + k2_parts
        p_occ = OCC_LETTERS[conv2['nh']]
        p_vir = VIR_LETTERS[conv2['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        for o in (1, 2, 3):
            for nm, (j, ham, k) in sorted(new_structures_at(o, kind='u').items()):
                left_ops = [[f'l{j}']] if j else None
                ket_dressing = [f't{k}'] if k else None
                extra_args = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
                for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                    name = f'u{o}_{tag2}_{row}_{nm}'
                    if hasattr(_genk2mod, f'u2_{tag2}_{row}_{nm}'):
                        print(f"{name}: reused (adc4_k2_blocks.u2_{tag2}_{row}_{nm})")
                        continue
                    terms = superop_element_terms_dressed(
                        p_tok(p_letter), ket2, sector, (ham,),
                        left_ops=left_ops, ket_dressing=ket_dressing)
                    if not terms:
                        print(f"{name}: 0 terms (pruned)")
                        continue
                    out_idx = ket2_letters + [p_letter]
                    print(f"{name}: {len(terms)} terms, out={out_idx}")
                    funcs.append(assemble_block(
                        name, _shape_expr(out_idx), terms, out_idx,
                        f"U^({o})-family {tag2}/orbital tag structure {nm} "
                        f"({row} rows): dressing left={left_ops}, "
                        f"ket={ket_dressing}, hams=('{ham}',); amplitude orders "
                        f"are the driver's choice (adc_order_rule.u_order_terms). "
                        f"Axes ({', '.join(out_idx)}).",
                        extra_args=extra_args))

            for nm, (j, k) in sorted(new_structures_at(o, kind='s').items()):
                left_ops = [[f'l{j}']] if j else None
                ket_dressing = [f't{k}'] if k else None
                extra_args = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
                for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                    name = f's{o}_{tag2}_{row}_{nm}'
                    if hasattr(_genk2mod, f's2_{tag2}_{row}_{nm}'):
                        print(f"{name}: reused (adc4_k2_blocks.s2_{tag2}_{row}_{nm})")
                        continue
                    terms = overlap_element_terms(
                        p_tok(p_letter), ket2, left_ops=left_ops,
                        ket_dressing=ket_dressing)
                    if not terms:
                        print(f"{name}: 0 terms (pruned)")
                        continue
                    out_idx = ket2_letters + [p_letter]
                    print(f"{name}: {len(terms)} terms, out={out_idx}")
                    funcs.append(assemble_block(
                        name, _shape_expr(out_idx), terms, out_idx,
                        f"S^({o})-family {tag2}/orbital overlap tag structure "
                        f"{nm} ({row} rows): left={left_ops}, ket={ket_dressing}; "
                        f"amplitude orders are the driver's choice "
                        f"(adc_order_rule.s_order_terms). Axes ({', '.join(out_idx)}).",
                        extra_args=extra_args))

    _write_module('adc5_u3_k2_blocks.py', funcs)


def generate_adc_u_k2(u_order, fname=None):
    """ORDER-GENERAL orbital<->k=2 U/S tag-structure generator -- the
    u_order-parametric form of generate_adc5_u3_k2 (which is the u_order<=3
    instance). Emits the structures FIRST reachable at `u_order`
    (new_structures_at; lower orders live in adc4_k2_blocks /
    adc5_u3_k2_blocks and are reused by the driver at higher amplitude
    orders -- the incremental-cache rule). Spin-orbital dense tensors,
    consumed via Sz-block-diagonal gathering (_k2_raw_S), exactly like the
    order-3 module. ADC(N) = u_order + 2, refused above ADC(5)."""
    refuse_above_max_order(u_order + 2, 'generate_adc_u_k2')
    _m2 = _generated_module('adc4', 'adc4_k2_blocks')
    _m3 = _generated_module('adc5', 'adc5_u3_k2_blocks')
    funcs = []
    for sector, tag2 in (('IP', '3h2p'), ('EA', '3p2h')):
        conv2 = class_index_convention(2, sector)
        k2_holes, k2_parts = class_letters(2, sector)
        ket2 = precursor_ket_string(2, sector)
        ket2_letters = k2_holes + k2_parts
        p_occ = OCC_LETTERS[conv2['nh']]
        p_vir = VIR_LETTERS[conv2['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])

        def reused(nm, row, kind):
            for mod, pref in ((_m2, f'{kind}2_'), (_m3, f'{kind}1_'),
                              (_m3, f'{kind}2_'), (_m3, f'{kind}3_')):
                if hasattr(mod, f'{pref}{tag2}_{row}_{nm}'):
                    return f'{mod.__name__.rsplit(".", 1)[-1]}.{pref}{tag2}_{row}_{nm}'
            return None

        for nm, (j, ham, k) in sorted(new_structures_at(u_order, kind='u').items()):
            left_ops = [[f'l{j}']] if j else None
            ket_dressing = [f't{k}'] if k else None
            extra_args = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                name = f'u{u_order}_{tag2}_{row}_{nm}'
                r = reused(nm, row, 'u')
                if r:
                    print(f"{name}: reused ({r})", flush=True)
                    continue
                terms = superop_element_terms_dressed(
                    p_tok(p_letter), ket2, sector, (ham,),
                    left_ops=left_ops, ket_dressing=ket_dressing)
                if not terms:
                    print(f"{name}: 0 terms (pruned)", flush=True)
                    continue
                out_idx = ket2_letters + [p_letter]
                print(f"{name}: {len(terms)} terms, out={out_idx}", flush=True)
                funcs.append(assemble_block(
                    name, _shape_expr(out_idx), terms, out_idx,
                    f"U^({u_order})-family {tag2}/orbital tag structure {nm} "
                    f"({row} rows): dressing left={left_ops}, "
                    f"ket={ket_dressing}, hams=('{ham}',); amplitude orders "
                    f"are the driver's choice. Axes ({', '.join(out_idx)}).",
                    extra_args=extra_args))

        for nm, (j, k) in sorted(new_structures_at(u_order, kind='s').items()):
            left_ops = [[f'l{j}']] if j else None
            ket_dressing = [f't{k}'] if k else None
            extra_args = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                name = f's{u_order}_{tag2}_{row}_{nm}'
                r = reused(nm, row, 's')
                if r:
                    print(f"{name}: reused ({r})", flush=True)
                    continue
                terms = overlap_element_terms(
                    p_tok(p_letter), ket2, left_ops=left_ops,
                    ket_dressing=ket_dressing)
                if not terms:
                    print(f"{name}: 0 terms (pruned)", flush=True)
                    continue
                out_idx = ket2_letters + [p_letter]
                print(f"{name}: {len(terms)} terms, out={out_idx}", flush=True)
                funcs.append(assemble_block(
                    name, _shape_expr(out_idx), terms, out_idx,
                    f"S^({u_order})-family {tag2}/orbital overlap tag "
                    f"structure {nm} ({row} rows): left={left_ops}, "
                    f"ket={ket_dressing}. Axes ({', '.join(out_idx)}).",
                    extra_args=extra_args))

    _write_module(fname or f'adc{u_order + 2}_u{u_order}_k2_blocks.py', funcs)


def generate_adc5_s_k1k2():
    """Cross-class overlap S_{k1,k2} blocks (2h1p<->3h2p / 2p1h<->3p2h) --
    the object the per-class ISR assembly NEVER forms (it builds within-class
    Delta S and the orbital<->class S_pJ only),
    measured nonzero from order 2 (|S|~9.77e-04 on H4/sto-3g).

    Bra = k=1 precursor at letter offset 0, ket = k=2 precursor at offsets
    past it (the generate_adc4_base_k2 c1-coupling letter pattern), element =
    <Phi0|{C_k1^dag, C_k2}|Phi0> Omega-dressed per s_structures_up_to(3) --
    every overlap structure the order rule reaches by order 3, pdaggerq
    pruning the rest (printed).

    Census (2026-07-21 probe, identical for both sectors):
      kept  : l1 (3 terms -- serves BOTH l1^(2) at order 2 and l1^(3) at
              order 3, one order-agnostic function), l2t1 (12), l3t2 (13).
      pruned: l2, l3, l4, t1-4, l1t2, l2t2, l2t3, l2t4, l4t2.

    Function names are ORDER-AGNOSTIC (s_{tag1}_{tag2}_{structure}) because
    amplitude order is the driver's choice (adc_autogenerated.cross_class_overlap).
    Emitted in the RAW precursor-basis orientation on BOTH classes -- no
    _K2_CLASS_SIGN here; that phase belongs to consumers mixing this with
    adc4_base-convention k=2 objects (the driver documents it)."""
    funcs = []
    for sector, tag1, tag2 in (('IP', '2h1p', '3h2p'), ('EA', '2p1h', '3p2h')):
        conv1 = class_index_convention(1, sector)
        bra_dag = precursor_bra_string(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(2, sector,
                                             occ_offset=conv1['nh'],
                                             vir_offset=conv1['np'])
        ket = precursor_ket_string(2, sector, occ_offset=conv1['nh'],
                                   vir_offset=conv1['np'])
        out_idx = (bra_holes + bra_parts) + (ket_holes + ket_parts)

        for nm, (j, k) in sorted(s_structures_up_to(3).items()):
            left_ops = [[f'l{j}']] if j else None
            ket_dressing = [f't{k}'] if k else None
            extra_args = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
            terms = overlap_element_terms(bra_dag, ket, left_ops=left_ops,
                                          ket_dressing=ket_dressing)
            name = f's_{tag1}_{tag2}_{nm}'
            if not terms:
                print(f"{name}: 0 terms (pruned)")
                continue
            print(f"{name}: {len(terms)} terms, out={out_idx}")
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"Cross-class overlap S {tag1}<->{tag2} structure {nm}: "
                f"<Phi0|{{C_{tag1}^dag, C_{tag2}}}|Phi0> dressed left={left_ops}, "
                f"ket={ket_dressing}; amplitude order is the driver's choice "
                f"(order-agnostic function). RAW precursor orientation, no "
                f"_K2_CLASS_SIGN. Axes = bra ({', '.join(bra_holes + bra_parts)}) "
                f"x ket ({', '.join(ket_holes + ket_parts)}).",
                extra_args=extra_args))

    _write_module('adc5_s_k1k2_blocks.py', funcs)


def generate_adc_class_pair(order, pair, max_rank_ext=5, fname=None):
    """ORDER-PARAMETRIC class-class element generator (spin-orbital) -- the
    Phase-3 generalization of generate_adc5_c3 / generate_adc5_c12 /
    generate_adc5_k2k2's dressed layers, driven by new_structures_at instead
    of hand split lists. `pair` in {'k1k1', 'k1k2', 'k2k2'}; the block enters
    ADC(order + k_bra + k_ket), refused above ADC(5), and is written to
    adc{N}_c{order}_{pair}_blocks.py.

    Function names follow the runtime resolvers so consumption is uniform:
    c{o}_{tag}_{nm} / s{o}c_{tag}_{nm} for the self pairs (the k2 tags
    3h2p/3p2h keep them collision-free from k1's), c{o}_{tag1}_{tag2}_{nm} /
    s_{tag1}_{tag2}_{nm} for k1k2 (gens12's s-convention).

    Every emission run ALSO offers the same-order rank>4 candidates (the
    u_order_terms(order, max_rank_ext) minus max_rank=4 diff) and writes an
    OFFERED/SURVIVORS census -- class-class elements have MORE operators to
    bridge rank than the (0,1) element (l4ft2 survives at k2 though pruned
    at k1), so each pair gets its own honest census. Dense spin-orbital
    tensors: validation scale (see generate_adc5_k2k2's SIZE note for the
    10-index k2k2 case)."""
    adc_order = order + int(pair[1]) + int(pair[3])
    refuse_above_max_order(adc_order, 'generate_adc_class_pair')
    bra_k, ket_k = {'k1k1': (1, 1), 'k1k2': (1, 2), 'k2k2': (2, 2)}[pair]

    def _extras(kind):
        full = (u_order_terms(order, max_rank=max_rank_ext) if kind == 'u'
                else s_order_terms(order, max_rank=max_rank_ext))
        base = set(t[0] for t in (u_order_terms(order, max_rank=4)
                                  if kind == 'u'
                                  else s_order_terms(order, max_rank=4)))
        out = {}
        for t in full:
            if t[0] not in base:
                out[t[0]] = (t[1], t[2], t[3]) if kind == 'u' else (t[1], t[2])
        return out

    u_new = dict(new_structures_at(order, kind='u'))
    s_new = dict(new_structures_at(order, kind='s'))
    u_ext, s_ext = _extras('u'), _extras('s')
    print(f"--- class pair {pair} at order {order}: {len(u_new)} u / "
          f"{len(s_new)} s structures, rank>4 offers u:{sorted(u_ext)} "
          f"s:{sorted(s_ext)} ---", flush=True)

    funcs = []
    offered = {'u': sorted(u_ext), 's': sorted(s_ext)}
    survivors = {'u': [], 's': []}
    tags = {1: {'IP': '2h1p', 'EA': '2p1h'}, 2: {'IP': '3h2p', 'EA': '3p2h'}}
    for sector in ('IP', 'EA'):
        tagb, tagk = tags[bra_k][sector], tags[ket_k][sector]
        convb = class_index_convention(bra_k, sector)
        bh, bp = class_letters(bra_k, sector)
        kh, kp = class_letters(ket_k, sector, occ_offset=convb['nh'],
                               vir_offset=convb['np'])
        bra_dag = precursor_bra_string(bra_k, sector)
        ket = precursor_ket_string(ket_k, sector, occ_offset=convb['nh'],
                                   vir_offset=convb['np'])
        out_idx = (bh + bp) + (kh + kp)

        def cname(nm):
            return (f'c{order}_{tagb}_{nm}' if pair != 'k1k2'
                    else f'c{order}_{tagb}_{tagk}_{nm}')

        def sname(nm):
            return (f's{order}c_{tagb}_{nm}' if pair != 'k1k2'
                    else f's_{tagb}_{tagk}_{nm}')

        for nm, (j, ham, k) in sorted({**u_new, **u_ext}.items()):
            lo = [[f'l{j}']] if j else None
            kd = [f't{k}'] if k else None
            extra_args = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
            name = cname(nm)
            terms = superop_element_terms_dressed(
                bra_dag, ket, sector, (ham,), left_ops=lo, ket_dressing=kd)
            print(f"{name}: {len(terms)} terms", flush=True)
            if not terms:
                continue
            if nm in u_ext and nm not in survivors['u']:
                survivors['u'].append(nm)
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"Order-{order} {tagb}<->{tagk} class-class structure {nm} "
                f"(raw dressed superoperator element, spin-orbital). Axes = "
                f"bra ({', '.join(bh + bp)}) x ket ({', '.join(kh + kp)}).",
                extra_args=extra_args))

        for nm, spec in sorted({**s_new, **s_ext}.items()):
            j, k = spec
            lo = [[f'l{j}']] if j else None
            kd = [f't{k}'] if k else None
            extra_args = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
            name = sname(nm)
            terms = overlap_element_terms(bra_dag, ket, left_ops=lo,
                                          ket_dressing=kd)
            print(f"{name}: {len(terms)} terms", flush=True)
            if not terms:
                continue
            if nm in s_ext and nm not in survivors['s']:
                survivors['s'].append(nm)
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"Order-{order} {tagb}<->{tagk} overlap structure {nm} "
                f"(raw, spin-orbital). Axes = bra ({', '.join(bh + bp)}) x "
                f"ket ({', '.join(kh + kp)}).",
                extra_args=extra_args))

    manifest = (f"OFFERED_RANK_EXT = {offered!r}\n"
                f"SURVIVORS_RANK_EXT = {survivors!r}\n"
                f"ORDER = {order}\nPAIR = {pair!r}")
    _write_module(fname or f'adc{adc_order}_c{order}_{pair}_blocks.py',
                  [manifest] + funcs)


def cross_pair_layout(pair):
    """(k_i, k_j, bra tag, ket tag, pair_index_convention) of the D x C cross
    pair `pair`: bra = the (k_i+1)h k_i p-dagger class (IP-type), ket = the
    (k_j+1)p k_j h class (EA-type)."""
    k_i, k_j = CROSS_PAIRS[pair]
    return (k_i, k_j, CLASS_TAGS[k_i]['IP'], CLASS_TAGS[k_j]['EA'],
            pair_index_convention(k_i, 'IP', k_j, 'EA'))


def cross_structures(order, pair, max_rank=4):
    """{structure name: (j, ham, k)} of the dressing structures the D x C
    block module at `order` carries. At the pair's leading order every
    structure of u_order_terms(order) (lower orders carry nothing, so the
    module must be self-contained); above it only the structures first
    reachable at `order` (lower ones live in the lower-order modules and are
    re-evaluated at the higher amplitude assignments, the rule of every
    other family)."""
    k_i, k_j = CROSS_PAIRS[pair]
    if order <= cross_block_leading_order(k_i, k_j):
        out = {}
        for nm, j, ham, k, a, b in u_order_terms(order, max_rank=max_rank):
            out.setdefault(nm, (j, ham, k))
        return out
    return dict(new_structures_at(order, max_rank=max_rank, kind='u'))


def generate_adc_cross_pair(order, pair='k1k1', max_rank=4, fname=None):
    """Dense spin-orbital D x C cross block at `order`: the union-space
    Hamiltonian block X = (D_kI|H^|C_kJ) = <Phi0|{E, [H, C]}|Phi0> between the
    (kI+1)h kI p-dagger class and the (kJ+1)p kJ h class, which the two-sector
    convention deletes (Tarantelli and Cederbaum, PRA 45, 2790 (1992),
    Eq. 4.5d). One function x{order}_{bra tag}_{ket tag}_{structure} per
    surviving dressing structure, axes bra (holes, particles) x ket (holes,
    particles); the amplitude order of each leaf is the runtime's choice
    (adc_order_rule.u_order_terms), as for every class-class block. The
    metric partner is derived too and must give no terms (it vanishes
    identically); orders 0 and 1 give no terms either -- both are recorded in
    the module's TERMS census, so 'pruned' and 'ungenerated' stay distinct.
    Validation scale: the (2h1p) x (2p1h) block is dense here."""
    k_i, k_j, tagb, tagk, conv = cross_pair_layout(pair)
    refuse_above_max_order(order + k_i + k_j, 'generate_adc_cross_pair')
    out_idx = conv['out_idx']
    funcs = []
    census = {'u': {}, 's': {}}
    print(f"--- cross pair {pair} ({tagb} x {tagk}) at order {order} ---", flush=True)
    for nm, (j, ham, k) in sorted(cross_structures(order, pair, max_rank).items()):
        lo = [[f'l{j}']] if j else None
        kd = [f't{k}'] if k else None
        terms = union_element_terms(k_i, 'IP', k_j, 'EA', hams=(ham,), left_ops=lo,
                                    ket_dressing=kd)
        census['u'][nm] = len(terms)
        name = f'x{order}_{tagb}_{tagk}_{nm}'
        print(f"{name}: {len(terms)} terms", flush=True)
        if not terms:
            continue
        extra = ([f'l{j}'] if j else []) + ([f't{k}'] if k else [])
        funcs.append(assemble_block(
            name, _shape_expr(out_idx), terms, out_idx,
            f"Order-{order} {tagb}(dagger-type bra) x {tagk} union-space cross block, "
            f"structure {nm}: <Phi0| {lo or ''} {{E, [{ham}, C]}} {kd or ''} |Phi0>, raw "
            f"(no normalization, no class-GS; both first act at order 3). Axes = bra "
            f"({', '.join(conv['bra_letters'])}) x ket ({', '.join(conv['ket_letters'])}).",
            extra_args=extra))
    for nm, j, k, a, b in s_order_terms(order, max_rank=max_rank):
        if nm in census['s']:
            continue
        lo = [[f'l{j}']] if j else None
        kd = [f't{k}'] if k else None
        terms = union_overlap_terms(k_i, 'IP', k_j, 'EA', left_ops=lo, ket_dressing=kd)
        census['s'][nm] = len(terms)
        if terms:
            raise RuntimeError(
                f"metric cross block {tagb} x {tagk} structure {nm} has {len(terms)} "
                f"terms; the union-space metric between a D-type and a C-type "
                f"operator vanishes identically, so this is an emitter defect")
    manifest = (f"ORDER = {order}\nPAIR = {pair!r}\nBRA = {tagb!r}\nKET = {tagk!r}\n"
                f"TERMS = {census!r}")
    _write_module(fname or cross_module_name(order, pair), [manifest] + funcs)
    return census


def generate_adc_cross_pair_restricted(order, pair='k1k1', max_rank=4, df=False,
                                       fname=None):
    """Spin-blocked twin of generate_adc_cross_pair (one function per
    Ms-conserving spin assignment of the bra and ket letters, amplitude leaves
    auto-detected); df=True DF-factorizes every two-electron bracket."""
    k_i, k_j, tagb, tagk, conv = cross_pair_layout(pair)
    refuse_above_max_order(order + k_i + k_j, 'generate_adc_cross_pair_restricted')
    out_idx = conv['out_idx']
    funcs = []
    for nm, (j, ham, k) in sorted(cross_structures(order, pair, max_rank).items()):
        lo = [[f'l{j}']] if j else None
        kd = [f't{k}'] if k else None
        funcs += _emit_spin_blocks(
            f'x{order}_{tagb}_{tagk}_{nm}',
            lambda sl, h=ham, lo=lo, kd=kd: union_element_terms(
                k_i, 'IP', k_j, 'EA', hams=(h,), left_ops=lo, ket_dressing=kd,
                spin_labels=sl),
            out_idx,
            lambda suffix, sl, sn=nm, d=df: (
                f"Order-{order} {tagb} x {tagk} union-space cross block, structure "
                f"{sn}, spin block {suffix}{' (DF/RI)' if d else ''} (axes "
                f"{', '.join(out_idx)})."),
            df=df)
    _write_module(fname or cross_module_name(order, pair, restricted=True, df=df), funcs)


def generate_adc5_c12():
    """SPIN-ORBITAL k1<->k2 coupling at ORDER 2 -- the ADC(5) (1,2) increment
    (block_order(5,1,2) = 2; found missing by generate_adc.generateADC(5)).

    Order-rule assignments (u_order_terms(2) at this bra/ket pair; pdaggerq
    prunes the rest): l1f at l1^(2) (15 terms), l2v at l2^(1) (34 terms),
    l2ft2 at (l2^(1), t2^(1)) (52 terms -- f[o,v]-carrying, evaluates to 0 in
    canonical HF; emitted anyway: generation is a superset, basis-general).

    The RAW elements only. The EFFECTIVE coupling adds the sequential
    class-GS corrections at runtime (adc_autogenerated.c12_o2_generated):
        eff = _K2_CLASS_SIGN * [ raw^(2) - U^(1) @ S^(1)_orb,k2^T
                                          - K_k1 * S_{k1,k2}^(2) ]
    validated against the exact-ISR (k1,k2)^[2] block at 6e-10 (H4) /
    6e-9 (LiH), both sectors, with the k2 column orientation pinned at
    order 1 (tests/test_adcn_generated_c12.py)."""
    funcs = []
    for sector, tag1, tag2 in (('IP', '2h1p', '3h2p'), ('EA', '2p1h', '3p2h')):
        conv1 = class_index_convention(1, sector)
        bh, bp = class_letters(1, sector)
        kh, kp = class_letters(2, sector, occ_offset=conv1['nh'],
                               vir_offset=conv1['np'])
        bra = precursor_bra_string(1, sector)
        ket = precursor_ket_string(2, sector, occ_offset=conv1['nh'],
                                   vir_offset=conv1['np'])
        out_idx = (bh + bp) + (kh + kp)
        for split, hams, lo, kd, amps in (
                ('l1f', ('f',), [['l1']], None, ['l1']),
                ('l2v', ('v',), [['l2']], None, ['l2']),
                ('l2ft2', ('f',), [['l2']], ['t2'], ['l2', 't2'])):
            terms = superop_element_terms_dressed(
                bra, ket, sector, hams, left_ops=lo, ket_dressing=kd)
            name = f'c2_{tag1}_{tag2}_{split}'
            if not terms:
                print(f"{name}: 0 terms (pruned)")
                continue
            print(f"{name}: {len(terms)} terms")
            funcs.append(assemble_block(
                name, _shape_expr(out_idx), terms, out_idx,
                f"Order-2 {tag1}<->{tag2} coupling split {split} "
                f"(raw dressed superoperator element). Axes = bra "
                f"({', '.join(bh + bp)}) x ket ({', '.join(kh + kp)}).",
                extra_args=amps))
    _write_module('adc5_c12_blocks.py', funcs)


def _module_subdir(fname):
    """Resolve (and create) the generatedADC/adc{N}/ subpackage a module
    named adc{N}_....py belongs in -- the layout every consumer (the
    generated-ADC runtime, generate_adc.py's coverage()) actually imports from. Every
    module writer routes through here so OUT_DIR-vs-adc{N}/ never drifts
    again (it drifted across 6 call sites before this helper existed --
    coverage() reported every already-generated module as missing)."""
    m = re.match(r'adc(\d+)_', fname)
    if not m:
        raise ValueError(
            f"cannot derive an adc{{N}} subpackage from module name {fname!r} "
            "(expected an 'adc<N>_...' prefix)")
    subdir = os.path.join(OUT_DIR, f'adc{m.group(1)}')
    os.makedirs(subdir, exist_ok=True)
    init_path = os.path.join(subdir, '__init__.py')
    if not os.path.exists(init_path):
        open(init_path, 'w').close()
    return subdir


def _write_module(fname, funcs):
    os.makedirs(OUT_DIR, exist_ok=True)
    init_path = os.path.join(OUT_DIR, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as fh:
            fh.write("")
    path = os.path.join(_module_subdir(fname), fname)
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- do not edit by hand.\n")
        fh.write("# Produced by src/CodeGeneration/generate_adcn.py; see that\n")
        fh.write("# module's docstring for the superoperator-element derivation/conventions.\n")
        # cached_einsum, not numpy.einsum -- a strict drop-in that memoizes the
        # contraction path AND finds it with opt_einsum. Both matter for the DF
        # twins: their "triangle" contractions (two B factors sharing the
        # auxiliary axis) are the case numpy's own optimizer gets badly wrong,
        # measured 19 s vs 1 s for one restricted U^(2) build on H2O/aug-cc-pVDZ.
        # See src/SingleReference/CC/cached_einsum.py's docstring.
        fh.write("import numpy as np\n")
        fh.write("from src.SingleReference.CC.cached_einsum import einsum\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"wrote {path}")


if __name__ == '__main__':
    generate_adc2()
    generate_adc2_restricted()
    generate_adc3_u()
    generate_adc3_u_restricted()
    generate_adc3_u_restricted(df=True)
    generate_adc4_base_c2()
    generate_adc4_base_c2_restricted()
    generate_adc4_base_k2()
    generate_adc4_base_k2_restricted()
    generate_adc4_u3()
    generate_adc4_u3_restricted()
