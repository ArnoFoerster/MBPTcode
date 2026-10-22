"""Sigma-vector (matrix-free) emission entry points for the ADC(n) generator
-- the backend leg that lets a class-class block be emitted as a
matrix-vector APPLY (y[bra] = sum_ket block[bra,ket] * v[ket]) instead of a
dense (bra x ket) tensor. It is the representation in which a class-class
block too large to hold densely can be generated at all.

This module only ADDS the apply-form entry points to generate_adcn.py's
dense/restricted assemblers, importing their derivation helpers. Ported here:
the shared assemblers, the first-order C^(1) self block and the order-general
union cross block; the per-order ADC(4)/(5) apply emitters are not part of
this package.

Both assemble_apply_block (spin-orbital) and assemble_apply_block_restricted
(spin-blocked) delegate the actual einsum emission to
pdaggerq_interface.raw_terms_to_sigma_einsum, which reuses the SAME
_parse_factor/_array_expr helpers as the dense emitter -- so these entry
points already work for restricted (_aaaa/_abab suffixes) and DF (B factors)
terms without change, the moment the derivation feeds them.

Trial-vector operand is named 'x' (NOT 'v' -- 'v' is the virtual-slice
argument used inside the emitted g[v,v,o,o]-style operands).
"""
import os
import re

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Base.pdaggerq_interface import raw_terms_to_sigma_einsum, exact_coefficient
from src.SingleReference.ADC.precursor_classes import (
    class_index_convention, class_letters, precursor_bra_string, precursor_ket_string,
    precursor_ket_string as _pk, OCC_LETTERS, VIR_LETTERS)
from src.CodeGeneration.generate_adcn import (
    superop_element_terms, superop_element_terms_dressed, overlap_element_terms,
    union_element_terms, cross_pair_layout, cross_module_name, cross_structures,
    _shape_expr, _R_LEAVES, OUT_DIR, _module_subdir, _amp_leaves, _u2_split_defs,
    _s2_split_defs)
from src.CodeGeneration.df_codegen import dress_integral_factor
from src.SingleReference.ADC.adc_order_rule import new_structures_at, refuse_above_max_order
from src.CodeGeneration.generate_adcn import _u3_split_defs
from src.CodeGeneration.generate_adcn import _emit_spin_blocks

_VEC = 'x'


def assemble_apply_block(func_name, bra_indices, ket_indices, raw_terms, doc,
                         extra_args=(), vec_var=_VEC, df=False):
    """Sigma-vector twin of generate_adcn.assemble_block. Emits

        def {func_name}(f, g, kd, *extra_args, o, v, no, nv, {vec_var}):
            y = zeros(bra_shape); <sigma einsums>; return y

    computing y[bra] = sum_ket block[bra,ket] * {vec_var}[ket] with no
    (bra x ket) tensor ever allocated (largest object is the bra-shaped
    output). raw_terms is the SAME pq.strings() output assemble_block would
    receive for this block; bra_indices/ket_indices is its out_indices split
    (bra letters then ket letters, the class_letters convention).

    df=True composes the DF/RI term-rewrite (df_codegen.dress_integral_factor)
    with the sigma emission: every 2-electron bracket <pq||rs> is expanded
    into a pair of 3-index B factors BEFORE emission, and the leaf signature
    takes B (naux, nso, nso) in place of g. The two passes commute cleanly
    because dress_integral_factor returns terms in the same pdaggerq format
    raw_terms_to_sigma_einsum consumes, and _array_expr already emits B
    factors (aux axis un-sliced) -- so DF x sigma is pure composition, no new
    emitter logic. This is the ADC(5)-target form: matrix-free AND
    DF-factorized in one generated routine."""
    if df:
        raw_terms = dress_integral_factor(raw_terms)
    leaves = ['f', 'B', 'kd'] if df else ['f', 'g', 'kd']
    args = leaves + list(extra_args) + ['o', 'v', 'no', 'nv', vec_var]
    src = [f"def {func_name}({', '.join(args)}):"]
    src.append(f'    """{doc}"""')
    src.append(f"    y = np.zeros({_shape_expr(bra_indices)})")
    src.extend(raw_terms_to_sigma_einsum(raw_terms, 'y', bra_indices, ket_indices,
                                         vec_var=vec_var))
    src.append("    return y")
    src.append("")
    return "\n".join(src)


# spin-blocked leaves with the three g_ssss replaced by the DF factors
# B_aa/B_bb (dress_integral_factor emits <..>_ssss -> B_aa/B_bb via
# df_codegen._b_name); f_aa/f_bb/d_aa/d_bb are unchanged.
_R_LEAVES_DF = ['f_aa', 'f_bb', 'B_aa', 'B_bb', 'd_aa', 'd_bb']


def assemble_apply_block_restricted(func_name, bra_indices, ket_indices, raw_terms,
                                    doc, extra_args=(), vec_var=_VEC, df=False):
    """assemble_apply_block's restricted twin: spin-blocked leaf args
    (_R_LEAVES) instead of (f, g, kd). Same as generate_adcn.
    assemble_block_restricted differs from assemble_block -- the emitter
    (raw_terms_to_sigma_einsum) is identical, only the leaf signature
    changes.

    df=True composes the DF/RI term-rewrite (dress_integral_factor, which
    handles the _aaaa/_abab spin suffixes) with the restricted sigma
    emission: the spin-blocked g_ssss leaves become B_aa/B_bb (restricted x
    sigma x DF -- all three axes on one emit path, still zero new emitter
    logic since _array_expr already slices spin-suffixed B factors)."""
    if df:
        raw_terms = dress_integral_factor(raw_terms)
    leaves = _R_LEAVES_DF if df else list(_R_LEAVES)
    args = leaves + list(extra_args) + ['o', 'v', 'no', 'nv', vec_var]
    src = [f"def {func_name}({', '.join(args)}):"]
    src.append(f'    """{doc}"""')
    src.append(f"    y = np.zeros({_shape_expr(bra_indices)})")
    src.extend(raw_terms_to_sigma_einsum(raw_terms, 'y', bra_indices, ket_indices,
                                         vec_var=vec_var))
    src.append("    return y")
    src.append("")
    return "\n".join(src)


def _write_apply_module(fname, funcs):
    """Module writer with a provenance header naming THIS file (kept
    distinct from generate_adcn.py's _write_module so the apply modules are
    unambiguously the sigma-vector product)."""
    path = os.path.join(_module_subdir(fname), fname)
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- do not edit by hand.\n")
        fh.write("# Produced by src/CodeGeneration/sigma_codegen.py\n")
        fh.write("# (matrix-free sigma-vector APPLY form of the class-class blocks;\n")
        fh.write("# y[bra] = sum_ket block[bra,ket] * x[ket], no dense block allocated).\n")
        # cached_einsum, not numpy.einsum: these applies run once per Davidson
        # matvec, and the DF ones are exactly the "triangle" contractions
        # (two B factors sharing the auxiliary axis, each sharing another axis
        # with x) whose path numpy's own optimizer gets badly wrong -- measured
        # scaling 6 / 2.4 s vs opt_einsum's 5 ms on ONE term of the restricted
        # DF C^(1) 2p1h apply (H2O/aug-cc-pVDZ), i.e. a ~100x slower matvec.
        # See src/SingleReference/CC/cached_einsum.py's docstring.
        fh.write("import numpy as np\n")
        fh.write("from src.SingleReference.CC.cached_einsum import einsum\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"wrote {path}")


def _c2_splits():
    splits = [
        ('vt2', ('v',), None, ['t2'], ['t2']),
        ('l2v', ('v',), [['l2']], None, ['l2']),
        ('l2ft2', ('f',), [['l2']], ['t2'], ['l2', 't2']),
    ]
    for r in (1, 2, 3):
        splits.append((f'ft{r}', ('f',), None, [f't{r}'], [f't{r}']))
        splits.append((f'l{r}f', ('f',), [[f'l{r}']], None, [f'l{r}']))
    return splits


def generate_adc4_base_c2_apply(df=False):
    """Emit the C^(2) class-class splits in matrix-free apply form
    (spin-orbital). One function per surviving split per sector, byte-parallel
    to generate_adc4_base_c2's dense functions but contracting the ket axes
    against a trial vector x. df=True additionally DF-factorizes every
    2-electron bracket (leaf B in place of g) -> adc4_c2_apply_df_blocks.py,
    the matrix-free + density-fitted target form."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts

        for split_name, hams, left_ops, ket_dressing, extra_args in _c2_splits():
            terms = superop_element_terms_dressed(
                bra_dag, ket, sector, hams, left_ops=left_ops, ket_dressing=ket_dressing)
            name = f'apply_c2_{tag}_{split_name}'
            if not terms:
                print(f"{name}: 0 terms (pruned)")
                continue
            print(f"{name}: {len(terms)} terms, bra={bra_letters}, ket={ket_letters}"
                  f"{' [DF]' if df else ''}")
            funcs.append(assemble_apply_block(
                name, bra_letters, ket_letters, terms,
                f"C^(2) {tag}-{tag} split {split_name} APPLY{' (DF/RI)' if df else ''}: "
                f"y[{', '.join(bra_letters)}] = sum_ket block * x[{', '.join(ket_letters)}]; "
                f"dressing left={left_ops}, ket={ket_dressing}, hams={hams}.",
                extra_args=extra_args, df=df))
    _write_apply_module('adc4_c2_apply_df_blocks.py' if df else 'adc4_c2_apply_blocks.py',
                        funcs)


def generate_adc4_base_k2_apply():
    """Emit adc4_k2_apply_blocks.py: the first-order k=1<->k=2 coupling in
    matrix-free apply form, BOTH directions.

    Forward (y[k1] = sum_{k2} C1[k1,k2] * x[k2], trial folded over the
    3h2p/3p2h ket): apply_c1_{tag1}_{tag2}. Mirrors generate_adcn.
    generate_adc4_base_k2's c1_{tag1}_{tag2} derivation exactly (same
    superop_element_terms call), so it validates against that shipped dense
    block.

    Transpose (y[k2] = sum_{k1} C1[k1,k2] * x[k1] = C1^T @ x, trial folded
    over the 2h1p/2p1h ket): apply_c1_{tag2}_{tag1}. Emitted from the SAME
    term-list by swapping which letter-set is the (output) bra and which is
    the (folded) trial -- raw_terms_to_sigma_einsum's P()-factorization is
    symmetric in bra/ket (a bra swap relabels the output, a ket swap relabels
    the trial), so folding the SAME block over its bra indices instead of its
    ket indices yields exactly the config transpose with no new derivation and
    no hermiticity assumption (verified == forward-block^T to machine zero).
    Both directions are needed to apply the symmetric ADC(4)
    supermatrix (H[2h1p,3h2p]=C1, H[3h2p,2h1p]=C1^T) fully matrix-free -- the
    3h2p/3p2h block is the ~16 GB Ne/aVDZ wall, so its transpose must be
    matrix-free too."""
    funcs = []
    for sector, tag1, tag2 in (('IP', '2h1p', '3h2p'), ('EA', '2p1h', '3p2h')):
        conv1 = class_index_convention(1, sector)
        bra_dag = precursor_bra_string(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(2, sector,
                                             occ_offset=conv1['nh'], vir_offset=conv1['np'])
        ket = precursor_ket_string(2, sector, occ_offset=conv1['nh'], vir_offset=conv1['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        terms = superop_element_terms(bra_dag, ket, sector, hams=('f', 'v'))
        fwd = f'apply_c1_{tag1}_{tag2}'
        print(f"{fwd}: {len(terms)} terms, bra={bra_letters}, ket={ket_letters}")
        funcs.append(assemble_apply_block(
            fwd, bra_letters, ket_letters, terms,
            f"First-order {tag1}<->{tag2} coupling APPLY (forward): "
            f"y[{', '.join(bra_letters)}] = sum_ket C1 * x[{', '.join(ket_letters)}]; "
            f"trial vector folded over the {tag2} ket (3-hole + 2-particle "
            f"antisymmetric class)."))
        trn = f'apply_c1_{tag2}_{tag1}'
        print(f"{trn}: {len(terms)} terms, bra={ket_letters}, ket={bra_letters} (transpose)")
        funcs.append(assemble_apply_block(
            trn, ket_letters, bra_letters, terms,
            f"First-order {tag1}<->{tag2} coupling APPLY (transpose, C1^T): "
            f"y[{', '.join(ket_letters)}] = sum_ket C1 * x[{', '.join(bra_letters)}]; "
            f"SAME term-list as {fwd} with the {tag2} letters as output and the "
            f"{tag1} trial folded (== forward config block transposed)."))
    _write_apply_module('adc4_k2_apply_blocks.py', funcs)


# ---------------------------------------------------------------------
# Restricted (spin-blocked) apply emitters: one apply function per
# Ms-conserving spin assignment of the free indices, via
# assemble_apply_block_restricted, consuming the same term lists
# (superop_element_terms_dressed with spin_labels) as the dense generator.
# ---------------------------------------------------------------------

def _emit_spin_apply_blocks(base_name, term_fn, bra_idx, ket_idx, doc_fn,
                            fixed_args=(), df=False):
    """Restricted apply emitter: one apply function per Ms-conserving spin
    assignment of bra_idx+ket_idx (term_fn(spin_labels)->raw terms, empty for
    a pruned assignment), via assemble_apply_block_restricted. Mirrors
    generate_adcn._emit_spin_blocks; amplitude leaves auto-detected. df=True
    additionally DF-factorizes each block (B_aa/B_bb in place of g_ssss)."""
    from itertools import product as _iproduct
    fixed = list(fixed_args)
    out_idx = list(bra_idx) + list(ket_idx)
    funcs = []
    for spins in _iproduct('ab', repeat=len(out_idx)):
        sl = dict(zip(out_idx, spins))
        terms = term_fn(sl)
        if not terms:
            continue
        suffix = ''.join(spins)
        name = f'{base_name}_{suffix}'
        amp_args = [a for a in _amp_leaves(terms) if a not in fixed]
        funcs.append(assemble_apply_block_restricted(
            name, bra_idx, ket_idx, terms, doc_fn(suffix, sl),
            extra_args=fixed + amp_args, df=df))
    return funcs


def _emit_spin_apply_blocks_bidirectional(base_name, base_name_t, term_fn,
                                          bra_idx, ket_idx, doc_fn, doc_fn_t,
                                          fixed_args=(), df=False,
                                          transpose_suffix='forward'):
    """`_emit_spin_apply_blocks` plus its TRANSPOSE, from the SAME per-spin
    term-list (one term_fn(sl) call serves both) -- the identical
    bra/ket-role-swap trick `generate_adc4_base_k2_apply` uses for the k1<->k2
    forward/transpose pair (apply_c1_{tag1}_{tag2} / apply_c1_{tag2}_{tag1}):
    `raw_terms_to_sigma_einsum` only cares which letters are handed to it as
    bra_indices (free output axes) vs ket_indices (contracted against the
    trial) -- swapping the two argument lists against the SAME raw terms
    gives the exact config transpose, no new pdaggerq derivation, no
    hermiticity assumption.

    SAFETY PRECONDITION (per that emitter's own docstring): bra_idx and
    ket_idx must be DISJOINT letter sets (no P(x,y) mixed bra<->ket swap can
    occur in the raw terms) -- true here exactly as for k1<->k2: the folded
    orbital letter (`ket_idx`, from OCC_LETTERS[conv['nh']:]/
    VIR_LETTERS[conv['np']:]) sits strictly past the class's own letter pool
    (`bra_idx`, class_letters(2, sector)), by construction of
    class_index_convention's offsets.

    transpose_suffix -- which spin-index ORDER the transpose function's name
    suffix uses. This MUST match how the runtime consumer builds the lookup
    suffix, and the two consumer families differ:
      'forward'  -- suffix in the FORWARD (bra_idx + ket_idx) order for BOTH
                    functions. Use when the consumer hardcodes one class
                    first for both directions (u2_k2_dot_orb / orb_dot_u2_k2
                    always put the k2 pattern first, then the orbital 'a').
      'ket_bra'  -- transpose suffix REORDERED to the transpose's OWN
                    (ket_idx + bra_idx) order. Use when the consumer is
                    restricted_class_matvec, which derives the suffix from
                    (bra-of-the-call, ket-of-the-call) and so SWAPS order
                    between forward and transpose (the k1<->k2 class-class
                    couplings). Getting this wrong makes the transpose
                    functions exist under un-findable names -- only the
                    spin-symmetric patterns match (caught by
                    c12_matfree_validate.py: forward exact, transpose gives
                    zero for asymmetric spins)."""
    from itertools import product as _iproduct
    fixed = list(fixed_args)
    out_idx = list(bra_idx) + list(ket_idx)
    funcs = []
    for spins in _iproduct('ab', repeat=len(out_idx)):
        sl = dict(zip(out_idx, spins))
        terms = term_fn(sl)
        if not terms:
            continue
        suffix = ''.join(spins)
        if transpose_suffix == 'ket_bra':
            suffix_t = ''.join(sl[l] for l in list(ket_idx) + list(bra_idx))
        else:
            suffix_t = suffix
        amp_args = [a for a in _amp_leaves(terms) if a not in fixed]
        funcs.append(assemble_apply_block_restricted(
            f'{base_name}_{suffix}', bra_idx, ket_idx, terms,
            doc_fn(suffix, sl), extra_args=fixed + amp_args, df=df))
        funcs.append(assemble_apply_block_restricted(
            f'{base_name_t}_{suffix_t}', ket_idx, bra_idx, terms,
            doc_fn_t(suffix_t, sl), extra_args=fixed + amp_args, df=df))
    return funcs


def generate_adc4_base_c2_apply_restricted(df=False):
    """Emit the spin-blocked C^(2) splits in matrix-free apply form
    (restricted x sigma). Uses the SHARED _u2_split_defs so the dense and
    apply forms cannot drift. df=True DF-factorizes every block (leaf B_aa/B_bb
    in place of g_ssss) -> adc4_c2_apply_restricted_df_blocks.py, the
    restricted x sigma x DF target form."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        for split_name, hams, left_ops, ket_dressing in _u2_split_defs():
            funcs += _emit_spin_apply_blocks(
                f'apply_c2_{tag}_{split_name}',
                lambda sl, h=hams, lo=left_ops, kd=ket_dressing:
                    superop_element_terms_dressed(bra_dag, ket, sector, h,
                                                  left_ops=lo, ket_dressing=kd,
                                                  spin_labels=sl),
                bra_letters, ket_letters,
                lambda suffix, sl, sn=split_name, d=df: (
                    f"C^(2) {tag}-{tag} split {sn} APPLY (restricted"
                    f"{' DF/RI' if d else ''}), spin block {suffix}: "
                    f"y[bra] = sum_ket block * x[ket]."),
                df=df)
    _write_apply_module(
        'adc4_c2_apply_restricted_df_blocks.py' if df else
        'adc4_c2_apply_restricted_blocks.py', funcs)


def generate_adc4_base_s2c_apply():
    """Emit adc4_c2_s2c_apply_blocks.py: the within-class S^(2) overlap splits
    in matrix-free apply form (spin-orbital). Needed by the ASSEMBLED C^(2)
    matvec (sigma_driver.assembled_c2_matvec), which applies the ISR Loewdin
    correction -1/2 (K s2 + s2 K) with s2 = s2_raw - N^(2) I. Shares
    _s2_split_defs with the dense generator so they cannot drift."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        for split_name, left_ops, ket_dressing in _s2_split_defs():
            amps = (([op[0] for op in left_ops] if left_ops else [])
                    + (list(ket_dressing) if ket_dressing else []))
            terms = overlap_element_terms(bra_dag, ket, left_ops=left_ops,
                                          ket_dressing=ket_dressing)
            name = f'apply_s2c_{tag}_{split_name}'
            if not terms:
                print(f"{name}: 0 terms (pruned)")
                continue
            print(f"{name}: {len(terms)} terms, bra={bra_letters}, ket={ket_letters}")
            funcs.append(assemble_apply_block(
                name, bra_letters, ket_letters, terms,
                f"S^(2) {tag}-{tag} within-class overlap split {split_name} APPLY: "
                f"y[bra] = sum_ket overlap * x[ket].",
                extra_args=amps))
    _write_apply_module('adc4_c2_s2c_apply_blocks.py', funcs)


def generate_c1_self_apply():
    """Emit adc4_c1_self_apply_blocks.py: the first-order C^(1) k=1 self-block
    (2h1p/2p1h, v-part) in matrix-free apply form -- the other half of the
    excitation-space class-class matvec (K + C^(1) + C^(2)) that the full
    ADC(4)-base sector operator needs alongside the assembled C^(2)."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        terms = superop_element_terms(bra_dag, ket, sector, hams=('v',))
        name = f'apply_c1_{tag}'
        print(f"{name}: {len(terms)} terms, bra={bra_letters}, ket={ket_letters}")
        funcs.append(assemble_apply_block(
            name, bra_letters, ket_letters, terms,
            f"First-order C^(1) {tag}-{tag} self-block APPLY (v-part): "
            f"y[bra] = sum_ket C1 * x[ket]."))
    _write_apply_module('adc4_c1_self_apply_blocks.py', funcs)


def generate_adc4_base_s2c_apply_restricted(df=False):
    """Emit adc4_c2_s2c_apply_restricted_blocks.py: the within-class S^(2)
    overlap splits in matrix-free apply form, spin-blocked (restricted x
    sigma) -- the restricted twin of generate_adc4_base_s2c_apply. Needed by the
    RESTRICTED assembled C^(2) matvec (sigma_driver.assembled_c2_matvec_
    restricted), which applies the same ISR Loewdin correction
    -1/2 (K s2 + s2 K) with s2 = s2_raw - N^(2) I as the spin-orbital driver,
    but dispatching per Ms-conserving spin pattern. Shares _s2_split_defs +
    overlap_element_terms with the dense restricted generator
    (generate_adcn.generate_adc4_base_c2_restricted's s2c leg) so they can't
    drift; one apply function per surviving spin assignment.

    df=True DF-factorizes the g-dependent overlap terms (leaf B_aa/B_bb in
    place of g_ssss) -> adc4_c2_s2c_apply_restricted_df_blocks.py, so the
    restricted DF operator runs the Loewdin correction off B too."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        for split_name, left_ops, ket_dressing in _s2_split_defs():
            funcs += _emit_spin_apply_blocks(
                f'apply_s2c_{tag}_{split_name}',
                lambda sl, lo=left_ops, kd=ket_dressing:
                    overlap_element_terms(bra_dag, ket, left_ops=lo,
                                          ket_dressing=kd, spin_labels=sl),
                bra_letters, ket_letters,
                lambda suffix, sl, sn=split_name, d=df: (
                    f"S^(2) {tag}-{tag} within-class overlap split {sn} APPLY "
                    f"(restricted{' DF/RI' if d else ''}), spin block {suffix}: "
                    f"y[bra] = sum_ket overlap * x[ket]."),
                df=df)
    _write_apply_module(
        'adc4_c2_s2c_apply_restricted_df_blocks.py' if df else
        'adc4_c2_s2c_apply_restricted_blocks.py', funcs)


def generate_c1_self_apply_restricted(df=False):
    """Emit adc4_c1_self_apply_restricted_blocks.py: the first-order C^(1)
    k=1 self-block (2h1p/2p1h, v-part) in matrix-free apply form, spin-blocked
    -- the restricted twin of generate_c1_self_apply. Same superop element
    (bra_dag, ket, hams=('v',)) the dense restricted c1_{tag}_{spins} blocks
    (adc2_blocks_restricted.py, generate_adcn line ~325) use, per Ms-conserving
    spin pattern; carries no amplitude leaves. The other half (with the
    assembled C^(2)) of the restricted excitation-space class-class matvec the
    restricted sector operator needs.

    df=True DF-factorizes the two-electron (v) term (leaf B_aa/B_bb) ->
    adc4_c1_self_apply_restricted_df_blocks.py."""
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        funcs += _emit_spin_apply_blocks(
            f'apply_c1_{tag}',
            lambda sl: superop_element_terms(bra_dag, ket, sector,
                                             hams=('v',), spin_labels=sl),
            bra_letters, ket_letters,
            lambda suffix, sl, t=tag, d=df: (
                f"First-order C^(1) {t}-{t} self-block APPLY (v-part, "
                f"restricted{' DF/RI' if d else ''}), spin block {suffix}: "
                f"y[bra] = sum_ket C1 * x[ket]."),
            df=df)
    _write_apply_module(
        'adc4_c1_self_apply_restricted_df_blocks.py' if df else
        'adc4_c1_self_apply_restricted_blocks.py', funcs)


def generate_adc4_base_k2_apply_restricted(df=False):
    """Emit adc4_k2_apply_restricted_blocks.py: the first-order k=1<->k=2
    coupling (2h1p<->3h2p / 2p1h<->3p2h) in matrix-free apply form,
    spin-blocked and BOTH directions -- the restricted twin of
    generate_adc4_base_k2_apply. Mirrors generate_adcn.generate_adc4_base_k2_restricted's
    c1_{tag1}_{tag2} superop derivation (bra k=1, ket k=2, hams=('f','v')) per
    Ms-conserving spin assignment of the 8 free indices, so it validates against
    that shipped dense block (adc4_k2_blocks_restricted.py) and hence against
    BuildSuperMatrixADCnRestricted.coupling_k1_k2_generated.

    Forward  apply_c1_{tag1}_{tag2}_{s_k1 s_k2}: bra k1 (3), ket k2 (5), trial
        folded over the 3h2p/3p2h ket.
    Transpose apply_c1_{tag2}_{tag1}_{s_k2 s_k1}: SAME per-spin term-list with
        the k2 letters as output and the k1 trial folded (== forward config
        block transposed, per raw_terms_to_sigma_einsum's bra/ket-symmetric
        P()-factorization). Both are needed to apply the symmetric restricted
        supermatrix (H[k1,k2]=C, H[k2,k1]=C^T) fully matrix-free.
    """
    funcs = []
    for sector, tag1, tag2 in (('IP', '2h1p', '3h2p'), ('EA', '2p1h', '3p2h')):
        conv1 = class_index_convention(1, sector)
        bra_dag = precursor_bra_string(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(2, sector,
                                             occ_offset=conv1['nh'], vir_offset=conv1['np'])
        ket = precursor_ket_string(2, sector, occ_offset=conv1['nh'], vir_offset=conv1['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        term_fn = (lambda sl, bd=bra_dag, kt=ket, sec=sector:
                   superop_element_terms(bd, kt, sec, hams=('f', 'v'), spin_labels=sl))
        funcs += _emit_spin_apply_blocks(
            f'apply_c1_{tag1}_{tag2}', term_fn, bra_letters, ket_letters,
            lambda suffix, sl, t1=tag1, t2=tag2, d=df: (
                f"First-order {t1}<->{t2} coupling APPLY (forward, restricted"
                f"{' DF/RI' if d else ''}), "
                f"spin block {suffix}: y[{t1}] = sum_ket C1 * x[{t2} ket]."),
            df=df)
        funcs += _emit_spin_apply_blocks(
            f'apply_c1_{tag2}_{tag1}', term_fn, ket_letters, bra_letters,
            lambda suffix, sl, t1=tag1, t2=tag2, d=df: (
                f"First-order {t1}<->{t2} coupling APPLY (transpose C1^T, "
                f"restricted{' DF/RI' if d else ''}), spin block {suffix}: "
                f"y[{t2}] = sum_ket C1 * x[{t1} ket]; SAME term-list, {t2} "
                f"letters output, {t1} folded."),
            df=df)
    _write_apply_module(
        'adc4_k2_apply_restricted_df_blocks.py' if df else
        'adc4_k2_apply_restricted_blocks.py', funcs)


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


def _is_fov_zero(raw_terms):
    """True when every term carries an occupied-virtual Fock factor
    f(occ, vir) / f(vir, occ): the structure vanishes for a canonical
    Hartree-Fock reference (Brillouin) and is not emitted in the production
    (apply) form, whose amplitude pool would evaluate it at the wrong order
    anyway. The dense validation modules keep it and the gates show it zero."""
    def fov(tok):
        m = re.match(r'^f\((\w+),(\w+)\)$', tok.strip())
        return bool(m) and ((m.group(1) in OCC_LETTERS) != (m.group(2) in OCC_LETTERS))
    return bool(raw_terms) and all(any(fov(t) for t in term[1:]) for term in raw_terms)


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
    for nm, (j, ham, k) in sorted(cross_structures(order, pair, max_rank).items()):
        lo = [[f'l{j}']] if j else None
        kd = [f't{k}'] if k else None
        spin_free = union_element_terms(k_i, 'IP', k_j, 'EA', hams=(ham,), left_ops=lo,
                                        ket_dressing=kd)
        if not spin_free:
            continue
        if _is_fov_zero(spin_free):
            print(f"apply_x{order}_{tagb}_{tagk}_{nm}: every term carries f[o,v] "
                  f"(zero for canonical HF) -- not emitted", flush=True)
            continue
        funcs += _emit_spin_apply_blocks_bidirectional(
            f'apply_x{order}_{tagb}_{tagk}_{nm}', f'apply_x{order}_{tagk}_{tagb}_{nm}',
            lambda sl, h=ham, lo=lo, kd=kd: expand_mixed_permutations(
                union_element_terms(k_i, 'IP', k_j, 'EA', hams=(h,), left_ops=lo,
                                    ket_dressing=kd, spin_labels=sl),
                bra_idx, ket_idx),
            bra_idx, ket_idx,
            lambda suffix, sl, sn=nm, d=df: (
                f"Order-{order} {tagb} x {tagk} union-space cross block, structure {sn} "
                f"APPLY (forward, restricted{' DF/RI' if d else ''}), spin block {suffix}: "
                f"y[{tagb}] = sum_ket X * x[{tagk} ket]."),
            lambda suffix, sl, sn=nm, d=df: (
                f"Order-{order} {tagb} x {tagk} union-space cross block, structure {sn} "
                f"APPLY (transpose X^T, restricted{' DF/RI' if d else ''}), spin block "
                f"{suffix}: y[{tagk}] = sum_ket X * x[{tagb} ket]; same term list, "
                f"{tagk} letters output, {tagb} folded."),
            df=df, transpose_suffix='ket_bra')
    _write_apply_module(cross_module_name(order, pair, df=df, apply=True), funcs)


def generate_adc5_c12_apply_restricted(df=False):
    """Emit adc5_c12_apply_restricted_blocks.py: the ORDER-2 k1<->k2 coupling
    (C12, the ADC(5) (1,2) increment; generate_adc5_c12's dense spin-orbital
    reference) in matrix-free apply form, spin-blocked, BOTH directions --
    generate_adc4_base_k2_apply_restricted's bidirectional pattern one order up
    (l1f/l2v/l2ft2 splits instead of the bare hams=('f','v') order-1 element),
    using _emit_spin_apply_blocks_bidirectional (one derivation per spin
    assignment, not two).

    Forward  apply_c2_{tag1}_{tag2}_{split}_{suffix}: bra k1, ket k2.
    Transpose apply_c2_{tag2}_{tag1}_{split}_{suffix}: SAME per-spin term
        list, k2 letters output, k1 trial folded.

    RAW splits only (matches generate_adc5_c12's scope); the sequential
    class-GS corrections (adc_autogenerated.c12_o2_generated's -U^(1)@S^(1) and
    -K_k1*S12^(2) terms) are applied by the consumer (sigma_driver.c12_dot_k2/
    k1_dot_c12), reusing the U2_k2 apply module's s2_l2 split at a DIFFERENT
    amplitude order (apply functions are amplitude-order-agnostic -- the
    array is a runtime argument) plus the NEW S12 apply
    (generate_adc5_s12_apply_restricted, below)."""
    funcs = []
    for sector, tag1, tag2 in (('IP', '2h1p', '3h2p'), ('EA', '2p1h', '3p2h')):
        conv1 = class_index_convention(1, sector)
        bra_dag = precursor_bra_string(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(2, sector,
                                             occ_offset=conv1['nh'], vir_offset=conv1['np'])
        ket = precursor_ket_string(2, sector, occ_offset=conv1['nh'], vir_offset=conv1['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        for split, hams, lo, kd in (
                ('l1f', ('f',), [['l1']], None),
                ('l2v', ('v',), [['l2']], None),
                ('l2ft2', ('f',), [['l2']], ['t2'])):
            term_fn = (lambda sl, bd=bra_dag, kt=ket, sec=sector, h=hams,
                      lo=lo, kd=kd:
                      superop_element_terms_dressed(bd, kt, sec, h,
                                                    left_ops=lo, ket_dressing=kd,
                                                    spin_labels=sl))
            funcs += _emit_spin_apply_blocks_bidirectional(
                f'apply_c2_{tag1}_{tag2}_{split}',
                f'apply_c2_{tag2}_{tag1}_{split}',
                term_fn, bra_letters, ket_letters,
                lambda suffix, sl, sp=split, t1=tag1, t2=tag2, d=df: (
                    f"Order-2 {t1}<->{t2} coupling split {sp} APPLY (forward, "
                    f"restricted{' DF/RI' if d else ''}), spin {suffix}: "
                    f"y[{t1}] = sum_ket C2 * x[{t2} ket]."),
                lambda suffix, sl, sp=split, t1=tag1, t2=tag2, d=df: (
                    f"Order-2 {t1}<->{t2} coupling split {sp} APPLY TRANSPOSE "
                    f"(restricted{' DF/RI' if d else ''}), spin {suffix}: "
                    f"y[{t2}] = sum_ket C2 * x[{t1} ket]; SAME term list, no "
                    f"new derivation."),
                df=df, transpose_suffix='ket_bra')
    _write_apply_module(
        'adc5_c12_apply_restricted_df_blocks.py' if df else
        'adc5_c12_apply_restricted_blocks.py', funcs)


def generate_adc5_s12_apply_restricted(df=False):
    """Emit adc5_s12_apply_restricted_blocks.py: the ORDER-2 k1<->k2
    CROSS-CLASS OVERLAP S_{k1,k2} (adc_autogenerated.cross_class_overlap's dense
    spin-orbital reference, adc5_s_k1k2_blocks.py) in matrix-free apply form,
    spin-blocked, BOTH directions -- needed for the sequential class-GS
    correction -K_k1*S12 in the EFFECTIVE C12 matvec (c12_o2_generated).

    ORDER-2 has exactly ONE surviving split: s_order_terms(2) at this
    bra/ket pair yields only ('l1', 1, 0, 2, 0) -- l1 dressing, evaluated at
    l1^(2) (verified directly against adc_order_rule.s_order_terms(2), not
    guessed from the spin-orbital module's function names -- that module
    ALSO carries l2t1/l3t2, but those need budget>=3 (mp_min_order(2)=1 for
    l2 PLUS mp_min_order(1)=2 for t1 => a+b>=3), i.e. they are the next order's
    order-3 (1,2) coupling content, NOT this one; bundling them into THIS
    module fed restricted_class_matvec's blanket prefix-sweep the wrong
    (order-3) content when computing the order-2 S12 -- caught by the
    c12_matfree_validate.py KeyError this order-check fixes).
    Pure overlap -- overlap_element_terms takes no hams.
    Forward apply_s12_{tag1}_{tag2}_l1_{suffix}: bra k1, ket k2.
    Transpose apply_s12_{tag2}_{tag1}_l1_{suffix}: SAME term list."""
    funcs = []
    for sector, tag1, tag2 in (('IP', '2h1p', '3h2p'), ('EA', '2p1h', '3p2h')):
        conv1 = class_index_convention(1, sector)
        bra_dag = precursor_bra_string(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(2, sector,
                                             occ_offset=conv1['nh'], vir_offset=conv1['np'])
        ket = precursor_ket_string(2, sector, occ_offset=conv1['nh'], vir_offset=conv1['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        for split, lo, kd in (
                ('l1', [['l1']], None),):
            term_fn = (lambda sl, bd=bra_dag, kt=ket, lo=lo, kd=kd:
                      overlap_element_terms(bd, kt, left_ops=lo,
                                            ket_dressing=kd, spin_labels=sl))
            funcs += _emit_spin_apply_blocks_bidirectional(
                f'apply_s12_{tag1}_{tag2}_{split}',
                f'apply_s12_{tag2}_{tag1}_{split}',
                term_fn, bra_letters, ket_letters,
                lambda suffix, sl, sp=split, t1=tag1, t2=tag2, d=df: (
                    f"Order-2 {t1}<->{t2} overlap split {sp} APPLY (forward, "
                    f"restricted{' DF/RI' if d else ''}), spin {suffix}: "
                    f"y[{t1}] = sum_ket S12 * x[{t2} ket]."),
                lambda suffix, sl, sp=split, t1=tag1, t2=tag2, d=df: (
                    f"Order-2 {t1}<->{t2} overlap split {sp} APPLY TRANSPOSE "
                    f"(restricted{' DF/RI' if d else ''}), spin {suffix}: "
                    f"y[{t2}] = sum_ket S12 * x[{t1} ket]; SAME term list, no "
                    f"new derivation."),
                df=df, transpose_suffix='ket_bra')
    _write_apply_module(
        'adc5_s12_apply_restricted_df_blocks.py' if df else
        'adc5_s12_apply_restricted_blocks.py', funcs)


def generate_u2_k2_apply_restricted(df=False):
    """Emit adc4_u2_k2_apply_restricted_blocks.py: the ORBITAL<->k2 (3h2p/3p2h)
    U^(2)/S^(2) couplings in ORBITAL-FOLDED apply form, BOTH directions:

        forward   apply_u2_{tag2}_{row}_{split}:
                  y[k2] = sum_p U2_{k2,p} * p_orb[p]  (== u2_k2_generated @ p_orb)
        transpose apply_u2t_{tag2}_{row}_{split}:
                  y[p]  = sum_k2 U2_{k2,p} * x[k2]     (== u2_k2_generated.T @ x)

    Both are computed WITHOUT ever materializing the dense (n_k2, norb)
    coupling or its (k2_spatial, norb) build intermediate -- the ~16 GB /
    14 min blocker for the aug-cc-pVTZ 3p2h manifold (1.28M configs), and
    (for the transpose, newly) the memory peak `restricted_dyson_operator`
    hit forming U2k2 solely to apply its adjoint.
    The transpose costs NO extra pdaggerq derivation --
    `_emit_spin_apply_blocks_bidirectional` reuses the SAME per-spin term
    list for both, via the bra/ket-role-swap trick `generate_adc4_base_k2_apply`
    already validates for the k1<->k2 coupling (disjointness precondition
    verified in that emitter's docstring: the orbital letter is drawn from
    OCC_LETTERS[conv2['nh']:]/VIR_LETTERS[conv2['np']:], strictly past the
    k2 class's own letter pool). Mirrors generate_adcn.
    generate_adc4_base_k2_restricted's u2/s2 legs (same
    superop_element_terms_dressed/overlap_element_terms per split).
    df=True -> B_aa/B_bb leaves."""
    funcs = []
    for sector, tag2 in (('IP', '3h2p'), ('EA', '3p2h')):
        conv2 = class_index_convention(2, sector)
        k2_holes, k2_parts = class_letters(2, sector)
        ket2 = _pk(2, sector)
        ket2_letters = k2_holes + k2_parts
        p_occ = OCC_LETTERS[conv2['nh']]
        p_vir = VIR_LETTERS[conv2['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])
        for split_name, hams, left_ops, ket_dressing in _u2_split_defs():
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                funcs += _emit_spin_apply_blocks_bidirectional(
                    f'apply_u2_{tag2}_{row}_{split_name}',
                    f'apply_u2t_{tag2}_{row}_{split_name}',
                    lambda sl, pl=p_letter, h=hams, lo=left_ops, kd=ket_dressing:
                        superop_element_terms_dressed(p_tok(pl), ket2, sector, h,
                                                      left_ops=lo, ket_dressing=kd,
                                                      spin_labels=sl),
                    ket2_letters, [p_letter],
                    lambda suffix, sl, sn=split_name, row=row, t2=tag2, d=df: (
                        f"U^(2) {t2}/orbital split {sn} ({row}) APPLY (restricted"
                        f"{' DF/RI' if d else ''}), spin {suffix}: fold p_orb over "
                        f"the orbital index -> y[{t2}]."),
                    lambda suffix, sl, sn=split_name, row=row, t2=tag2, d=df: (
                        f"U^(2) {t2}/orbital split {sn} ({row}) APPLY TRANSPOSE "
                        f"(restricted{' DF/RI' if d else ''}), spin {suffix}: "
                        f"fold the {t2} trial over the k2 config -> y[orbital "
                        f"{row}] (== forward config block transposed; SAME term "
                        f"list, no new derivation)."),
                    df=df)
        for split_name, left_ops, ket_dressing in _s2_split_defs():
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                funcs += _emit_spin_apply_blocks_bidirectional(
                    f'apply_s2_{tag2}_{row}_{split_name}',
                    f'apply_s2t_{tag2}_{row}_{split_name}',
                    lambda sl, pl=p_letter, lo=left_ops, kd=ket_dressing:
                        overlap_element_terms(p_tok(pl), ket2, left_ops=lo,
                                              ket_dressing=kd, spin_labels=sl),
                    ket2_letters, [p_letter],
                    lambda suffix, sl, sn=split_name, row=row, t2=tag2, d=df: (
                        f"S^(2) {t2}/orbital split {sn} ({row}) APPLY (restricted"
                        f"{' DF/RI' if d else ''}), spin {suffix}: fold trial over "
                        f"the orbital index -> y[{t2}]."),
                    lambda suffix, sl, sn=split_name, row=row, t2=tag2, d=df: (
                        f"S^(2) {t2}/orbital split {sn} ({row}) APPLY TRANSPOSE "
                        f"(restricted{' DF/RI' if d else ''}), spin {suffix}: "
                        f"fold the {t2} trial over the k2 config -> y[orbital "
                        f"{row}] (== forward config block transposed)."),
                    df=df)
    _write_apply_module(
        'adc4_u2_k2_apply_restricted_df_blocks.py' if df else
        'adc4_u2_k2_apply_restricted_blocks.py', funcs)


def generate_adc_u_k2_apply_restricted(orders=(1, 2, 3), df=False,
                                       fname=None):
    """ORDER-GENERAL orbital<->k2 U/S coupling in ORBITAL-FOLDED apply form,
    BOTH directions -- the apply twin of generate_adcn.generate_adc_u_k2 (the
    dense adc5_u3_k2_blocks generator), covering every structure first
    reachable at each order in `orders` (adc_order_rule.new_structures_at).
    Emits apply_u{o}_{tag2}_{row}_{nm} / apply_u{o}t_... (+ s{o}/s{o}t) with
    the home-order o prefix mirroring adc5_u3_k2_blocks' naming, so the
    driver (sigma_driver.u3_k2_dot_orb / orb_dot_u3_k2, forthcoming) resolves
    a structure by its home-order prefix exactly as the dense _k2_raw_S does.

    generate_adc_u_k2_apply_restricted((1,2,3)) is ADC(5)'s U^(3)_k2 apply
    (orbital<->3h2p/3p2h, orders 1-3). SELF-CONTAINED: emits orders 1-2 too
    (re-derives the order-2 structures the adc4_base_u2_k2 module already has --
    correct byte-identical content, chosen for single-module resolution over
    cross-module prefix bookkeeping). transpose_suffix='forward' because both
    directions gather at the SAME (k2 spin pattern + 'a') suffix, exactly as
    generate_u2_k2_apply_restricted -- the orbital output is a single-index
    scatter, not a class transpose."""
    for o in orders:
        refuse_above_max_order(o + 2, 'generate_adc_u_k2_apply_restricted')
    funcs = []
    for sector, tag2 in (('IP', '3h2p'), ('EA', '3p2h')):
        conv2 = class_index_convention(2, sector)
        k2_holes, k2_parts = class_letters(2, sector)
        ket2 = _pk(2, sector)
        ket2_letters = k2_holes + k2_parts
        p_occ = OCC_LETTERS[conv2['nh']]
        p_vir = VIR_LETTERS[conv2['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])
        for o in orders:
            for nm, (j, ham, k) in sorted(new_structures_at(o, kind='u').items()):
                lo = [[f'l{j}']] if j else None
                kd = [f't{k}'] if k else None
                for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                    funcs += _emit_spin_apply_blocks_bidirectional(
                        f'apply_u{o}_{tag2}_{row}_{nm}',
                        f'apply_u{o}t_{tag2}_{row}_{nm}',
                        lambda sl, pl=p_letter, h=(ham,), lo=lo, kd=kd:
                            superop_element_terms_dressed(p_tok(pl), ket2, sector, h,
                                                          left_ops=lo, ket_dressing=kd,
                                                          spin_labels=sl),
                        ket2_letters, [p_letter],
                        lambda suffix, sl, nm=nm, row=row, o=o, t2=tag2, d=df: (
                            f"U^({o})-family {t2}/orbital {nm} ({row}) APPLY "
                            f"(restricted{' DF/RI' if d else ''}), spin {suffix}: "
                            f"fold p_orb over the orbital index -> y[{t2}]."),
                        lambda suffix, sl, nm=nm, row=row, o=o, t2=tag2, d=df: (
                            f"U^({o})-family {t2}/orbital {nm} ({row}) APPLY "
                            f"TRANSPOSE (restricted{' DF/RI' if d else ''}), spin "
                            f"{suffix}: fold the {t2} trial -> y[orbital {row}]."),
                        df=df)
            for nm, (j, k) in sorted(new_structures_at(o, kind='s').items()):
                lo = [[f'l{j}']] if j else None
                kd = [f't{k}'] if k else None
                for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                    funcs += _emit_spin_apply_blocks_bidirectional(
                        f'apply_s{o}_{tag2}_{row}_{nm}',
                        f'apply_s{o}t_{tag2}_{row}_{nm}',
                        lambda sl, pl=p_letter, lo=lo, kd=kd:
                            overlap_element_terms(p_tok(pl), ket2, left_ops=lo,
                                                  ket_dressing=kd, spin_labels=sl),
                        ket2_letters, [p_letter],
                        lambda suffix, sl, nm=nm, row=row, o=o, t2=tag2, d=df: (
                            f"S^({o})-family {t2}/orbital {nm} ({row}) APPLY "
                            f"(restricted{' DF/RI' if d else ''}), spin {suffix}: "
                            f"fold trial over the orbital index -> y[{t2}]."),
                        lambda suffix, sl, nm=nm, row=row, o=o, t2=tag2, d=df: (
                            f"S^({o})-family {t2}/orbital {nm} ({row}) APPLY "
                            f"TRANSPOSE (restricted{' DF/RI' if d else ''}), spin "
                            f"{suffix}: fold the {t2} trial -> y[orbital {row}]."),
                        df=df)
    _write_apply_module(
        fname or ('adc5_u3_k2_apply_restricted_df_blocks.py' if df else
                  'adc5_u3_k2_apply_restricted_blocks.py'), funcs)


def generate_adc5_k2k2_apply():
    """Emit adc5_k2k2_apply_blocks.py: the ADC(5) (2,2) k=2 class-class
    self-coupling C^(1)_k2 in matrix-free apply form (spin-orbital).

    THIS IS THE REPRESENTATION THAT MATTERS for this block. The dense twin
    (generate_adcn.generate_adc5_k2k2) is a 10-index (no^3 nv^2)^2 tensor --
    already ~2 GB at H2O/sto-3g and hopeless beyond -- so it exists only as the
    small-system arbiter this apply form is validated against. Everything at
    production scale must go through here.

    Same object as generate_c1_self_apply emits at k=1: the v-part superoperator
    element with bra and ket both in the class, folded as
    y[bra] = sum_ket C1 * x[ket]. Because `superop_element_terms` is
    class-general, going from k=1 to k=2 is purely the class argument.
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
        terms = superop_element_terms(bra_dag, ket, sector, hams=('v',))
        name = f'apply_c1_{tag}'
        print(f"{name}: {len(terms)} terms, bra={bra_letters}, ket={ket_letters}")
        funcs.append(assemble_apply_block(
            name, bra_letters, ket_letters, terms,
            f"ADC(5) first-order C^(1) {tag}-{tag} SELF-block APPLY (v-part): "
            f"y[bra] = sum_ket C1 * x[ket]. block_order(5,2,2) == 1; this block "
            f"is absent at ADC(4) (order 0 => K only, diagonal), which is what "
            f"makes the closed-form downfold Schur exact at EVEN orders."))
    _write_apply_module('adc5_k2k2_apply_blocks.py', funcs)


def generate_adc5_k2k2_apply_restricted(df=False):
    """Emit adc5_k2k2_apply_restricted[_df]_blocks.py: the restricted (spin-
    blocked) twin of generate_adc5_k2k2_apply -- the ADC(5) (2,2) block with all
    four required properties (spin-restricted + matrix-free, and DF/RI when df=True).

    The block is its own transpose (bra and ket are the SAME class), so unlike
    generate_adc4_base_k2_apply_restricted there is no separate transpose emission:
    the symmetric supermatrix needs only this one direction.

    COST NOTE: 10 free indices means 2^10 spin assignments enumerated per
    sector, most pruned by Ms conservation. That is generation-time only.
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
        funcs += _emit_spin_apply_blocks(
            f'apply_c1_{tag}',
            lambda sl, bd=bra_dag, kt=ket, sec=sector:
                superop_element_terms(bd, kt, sec, hams=('v',), spin_labels=sl),
            bra_holes + bra_parts, ket_holes + ket_parts,
            lambda suffix, sl, t=tag, d=df: (
                f"ADC(5) C^(1) {t}-{t} self-block APPLY (restricted"
                f"{' DF/RI' if d else ''}), spin block {suffix}: "
                f"y[{t}] = sum_ket C1 * x[{t} ket]."),
            df=df)
    _write_apply_module(
        'adc5_k2k2_apply_restricted_df_blocks.py' if df else
        'adc5_k2k2_apply_restricted_blocks.py', funcs)


def generate_adc5_c3_apply_restricted(df=False):
    """Emit adc5_c3_apply_restricted[_df]_blocks.py: the ADC(5) order-3
    class-class C^(3) splits in matrix-free apply form (restricted x sigma
    [x DF]) -- the order-3 twin of generate_adc4_base_c2_apply_restricted.

    Splits come from the SAME _u3_split_defs + superop_element_terms_dressed
    the dense generator (generate_adcn.generate_adc5_c3_restricted) consumes,
    so dense and apply forms cannot drift. Of the five splits that survive
    pdaggerq in the dense module (l2vt2, vt1, l1v, l2ft3, l1ft2), only the
    first THREE are emitted here:

      * l2ft3 / l1ft2 are IDENTICALLY ZERO on this code path: every term forces the Fock factor into the
        off-diagonal f[o,v] slice, which vanishes for canonical HF -- and the
        class blocks always see the canonical f (BuildSuperMatrixADCnRestricted.
        _rargs sets f_aa = diag(eps); the f_orb hook applies to the orbital
        block only). Emitting them would ship dead weight that "validates"
        against zero. l2ft3 is also C^(3)'s ONLY rank-3 consumer, which is why
        C^(3) needs NO Laplace fusion (section 19) -- the surviving splits are
        all rank <= 2.
      * vt3, l3v, l2ft1, l3ft2 are pruned by pdaggerq for the class-class
        element (no function in the dense module either).

    NO s3c apply blocks exist to emit: pdaggerq pruned every order-3
    within-class overlap split, so Delta S^(3) comes ENTIRELY from the order-2
    s2c structure (adc4_c2_s2c_apply_restricted_blocks) evaluated at higher
    amplitude orders (section 17) -- the driver's job, not this generator's.

    df=True DF-factorizes every block (leaf B_aa/B_bb in place of g_ssss) ->
    adc5_c3_apply_restricted_df_blocks.py: restricted + matrix-free + DF in
    one generated routine."""
    surviving = ('l2vt2', 'vt1', 'l1v')
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        bra_holes, bra_parts = class_letters(1, sector)
        ket_holes, ket_parts = class_letters(1, sector,
                                             occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_dag = precursor_bra_string(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        bra_letters = bra_holes + bra_parts
        ket_letters = ket_holes + ket_parts
        for split_name, hams, left_ops, ket_dressing in _u3_split_defs():
            if split_name not in surviving:
                continue
            funcs += _emit_spin_apply_blocks(
                f'apply_c3_{tag}_{split_name}',
                lambda sl, h=hams, lo=left_ops, kd=ket_dressing:
                    superop_element_terms_dressed(bra_dag, ket, sector, h,
                                                  left_ops=lo, ket_dressing=kd,
                                                  spin_labels=sl),
                bra_letters, ket_letters,
                lambda suffix, sl, sn=split_name, d=df: (
                    f"C^(3) {tag}-{tag} split {sn} APPLY (restricted"
                    f"{' DF/RI' if d else ''}), spin block {suffix}: "
                    f"y[bra] = sum_ket block * x[ket]. Amplitude orders are the "
                    f"driver's choice (adc_order_rule.u_order_terms at n=3)."),
                df=df)
    _write_apply_module(
        'adc5_c3_apply_restricted_df_blocks.py' if df else
        'adc5_c3_apply_restricted_blocks.py', funcs)


def generate_adc5_u4_restricted_df():
    """Emit adc5_u4_blocks_restricted_df.py: the DF/RI twin of the ADC(5)
    U^(4)/S^(4) orbital<->k1 block module (adc5_u4_blocks_restricted, produced
    by generate_adcn.generate_adc_u_restricted(4), which has no df flag) --
    closing U^(4)'s DF gap.

    Byte-parallel to how generate_adc3_u_restricted(df=True) produces
    adc3_u_blocks_restricted_df.py: the SAME structure enumeration
    (adc_order_rule.new_structures_at(4)) and the SAME per-spin emitter
    (generate_adcn._emit_spin_blocks), only with df=True so every 2-electron
    bracket is rewritten through df_codegen.dress_integral_factor (leaf
    B_aa/B_bb in place of g_ssss) before emission -- DF rewriting is purely
    compositional, no new emitter logic. Function names/signatures otherwise
    match the dense twin one-to-one, so sigma_driver can resolve a stitch from
    either module interchangeably."""
    u_order = 4
    u_new = new_structures_at(u_order, kind='u')
    s_new = new_structures_at(u_order, kind='s')
    print(f"--- U^({u_order}) DF: {len(u_new)} new tag structures, "
          f"S^({u_order}) DF: {len(s_new)} ---")
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
                        f"{suffix} (axes {', '.join(oi)}), DF/RI (B_aa/B_bb). "
                        f"Amplitude orders are the driver's choice "
                        f"(adc_order_rule.u_order_terms)."), df=True)

        for nm, (j, k) in sorted(s_new.items()):
            left_ops = [[f'l{j}']] if j else None
            ket_dressing = [f't{k}'] if k else None
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                funcs += _emit_spin_blocks(
                    f's{u_order}_{tag}_{row}_{nm}',
                    lambda sl, pl=p_letter, lo=left_ops, kd=ket_dressing:
                        overlap_element_terms(
                            p_tok(pl), ket, left_ops=lo, ket_dressing=kd,
                            spin_labels=sl),
                    out_idx,
                    lambda suffix, sl, s=nm, r=row, oi=out_idx, o=u_order: (
                        f"S^({o}) {tag}/orbital overlap tag structure {s} ({r} "
                        f"rows), spin block {suffix} (axes {', '.join(oi)}), "
                        f"DF/RI (B_aa/B_bb)."), df=True)

    # Accurate-provenance writer (generate_adcn._write_module would stamp the
    # wrong producer): dense-block module layout, header naming THIS file.
    fname = f'adc{u_order + 1}_u{u_order}_blocks_restricted_df.py'
    path = os.path.join(_module_subdir(fname), fname)
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- do not edit by hand.\n")
        fh.write("# Produced by src/CodeGeneration/sigma_codegen.py\n")
        fh.write("# (generate_adc5_u4_restricted_df: DF/RI twin of adc5_u4_blocks_restricted;\n")
        fh.write("# same derivation as generate_adcn.generate_adc_u_restricted(4), B_aa/B_bb\n")
        fh.write("# leaves in place of g_ssss via df_codegen.dress_integral_factor).\n")
        fh.write("import numpy as np\nfrom numpy import einsum\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"wrote {path}")


if __name__ == '__main__':
    generate_adc4_base_c2_apply()
    generate_adc4_base_c2_apply(df=True)
    generate_adc4_base_k2_apply()
    generate_adc4_base_c2_apply_restricted()
    generate_adc4_base_c2_apply_restricted(df=True)
    generate_adc4_base_s2c_apply()
    generate_c1_self_apply()
    generate_adc4_base_s2c_apply_restricted()
    generate_adc4_base_s2c_apply_restricted(df=True)
    generate_c1_self_apply_restricted()
    generate_c1_self_apply_restricted(df=True)
    generate_adc4_base_k2_apply_restricted()
    generate_adc4_base_k2_apply_restricted(df=True)
    generate_u2_k2_apply_restricted()
    generate_u2_k2_apply_restricted(df=True)
    generate_adc5_k2k2_apply()
    generate_adc5_k2k2_apply_restricted()
    generate_adc5_k2k2_apply_restricted(df=True)
    generate_adc5_c3_apply_restricted()
    generate_adc5_c3_apply_restricted(df=True)
    generate_adc5_u4_restricted_df()
