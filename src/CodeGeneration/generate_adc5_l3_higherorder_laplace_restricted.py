"""Laplace-fused l3-consuming U^(4)/S^(4) stitches at the ADC(5) order-4
amplitude combos of the l3ft2 / l3t2 tag structures -- the NESTED-fusion piece.

adc_order_rule puts two amplitude assignments of l3ft2 (and l3t2) at n=4:
    (l3^2, t2^2)  -- l3 at order 2, outer ket t2 at order 2
    (l3^3, t2^1)  -- l3 at order 3, outer ket t2 at order 1
The (2,2) combo is an ordinary single fusion (l3^(2)'s order-2 numerator has only
rank<=2 leaves). The (3,1) combo is the frontier NEST: l3^(3)'s order-3 numerator
carries t3^(2)/t4^(2) leaves that must themselves be Laplace-fused (a fusion
inside a fusion). Both are handled uniformly by laplace_codegen.fuse_nested_
numerator: rank>=3 leaves recursively fused (each its own tau slot, kind-named
by rank), rank<=2 leaves ORDER-TAGGED so l3^(3)'s DIRECT t2 (order 2) and the
inner t3/t4 numerators' t2 (order 1) get distinct, correctly-fed tags. The outer
l3 factor is fused with the _to_l (occ-first canonical + sign) as in
generate_adc4_u3_l3_laplace_restricted; the inner leaves stay pure T-frame
(laplace_codegen.fuse_nested_numerator).

Emitted names: {u4|s4}_{tag}_{row}_{fam}_{combo}_{suffix}, combo in {31,22}. The
driver supplies t2out_* = T2^(1) for the 31 combo and T2^(2) for the 22 combo
(the outer ket order); t2_1_*/t2_2_*/t1_2_* are l3's numerator leaves. l3/t3/t4
are NEVER materialized.

Writes generatedADC/adc5/adc5_l3_higherorder_laplace_restricted.py (generated on
the user's machine, never committed).
"""
import os
import re

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Base.pdaggerq_interface import exact_coefficient

from src.SingleReference.ADC.precursor_classes import (
    OCC_LETTERS, VIR_LETTERS, class_letters, precursor_ket_string,
    class_index_convention)
from src.CodeGeneration.generate_adcn import (
    superop_element_terms_dressed, overlap_element_terms, _shape_expr)
from src.CodeGeneration.generate_mpn_density_restricted import (
    t_numerator_terms_restricted, T_BRA)
from src.CodeGeneration.laplace_codegen import (
    fuse_amplitude_into_consumer, lambda_canonical_and_sign,
    fuse_nested_numerator, assemble_laplace_fused_multikind,
    highrank_tags, _term_has_factor)

OUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'SingleReference', 'ADC', 'generatedADC', 'adc5'))

RANK = 3   # the consumed lambda rank (l3). Rank-general via the fuser.

# (combo suffix, l3 order) -- outer ket t2 order is 4 - l3_order (fed by driver).
_COMBOS = [('31', 3), ('22', 2)]

# Uniform param superset -- unused leaves in a given combo are harmless. eps_a
# drives the quadrature grid; t2_1/t2_2/t1_2 are l3's numerator leaves at their
# orders; t2out_* is the OUTER ket t2 (order supplied by the driver per combo).
PARAMS = ['f_aa', 'f_bb', 'g_aaaa', 'g_abab', 'g_bbbb', 'd_aa', 'd_bb',
          'o', 'v', 'nv', 'no', 'eps_a',
          't2_1_aaaa', 't2_1_abab', 't2_2_aaaa', 't2_2_abab', 't1_2_aa',
          't2out_aaaa', 't2out_abab']
ALIASES = {'t2_1_bbbb': 't2_1_aaaa', 't2_2_bbbb': 't2_2_aaaa',
           't1_2_bb': 't1_2_aa', 't2out_bbbb': 't2out_aaaa'}


def _numerator_block(block, rank=RANK):
    keys = T_BRA[rank][1]
    if block in keys:
        return block
    flipped = block.translate(str.maketrans('ab', 'ba'))
    if flipped in keys:
        return flipped
    raise KeyError(f"no numerator key for l{rank} block {block!r}")


def _num_fn(rank, order, block):
    return t_numerator_terms_restricted(rank, order, _numerator_block(block, rank))


def _signed(terms, sign):
    return [[f"{exact_coefficient(t[0]) * sign:+.15g}"] + list(t[1:]) for t in terms]


def _rename_outer_t2(outer):
    """The OUTER ket t2 dressing -> t2out_* (a distinct order slot from l3's
    numerator t2). Its perturbation order is the combo's t2 order; the driver
    feeds t2out the right-order array."""
    return [[t[0]] + [re.sub(r'^t2_', 't2out_', f.strip()) for f in t[1:]] for t in outer]


def _fuse_l3(outer, l3_order):
    """Fuse every outer l3 tag at `l3_order`: its numerator via fuse_nested_
    numerator (nested when l3_order>=3), the l3 factor via the _to_l canonical."""
    terms = outer
    while True:
        tags = {t: rk for t, rk in highrank_tags(terms).items() if t.startswith(f'l{RANK}_')}
        if not tags:
            break
        tag = sorted(tags)[0]
        block = tag.split('_', 1)[1]
        same = len(set(block)) == 1
        lcs = lambda_canonical_and_sign(RANK)
        canon, sign = lcs['same'] if same else lcs['mixed']
        inner = _signed(fuse_nested_numerator(_num_fn, RANK, l3_order, block), sign)
        matching = [t for t in terms if _term_has_factor(t, tag)]
        rest = [t for t in terms if not _term_has_factor(t, tag)]
        terms = rest + fuse_amplitude_into_consumer(
            inner, RANK, matching, tag, canonical=canon,
            dress_names=(f'Oe_l{RANK}', f'Ve_l{RANK}'))
    return terms


def _build_functions(validator=None):
    """Emit every {u4|s4}_{tag}_{row}_{fam}_{combo}_{suffix} function. If
    `validator` is given, it is called (name, out_idx, fused, l3_order, spins,
    sector, row, fam) to numerically check each block as it is built."""
    from itertools import product as iprod
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        kh, kp = class_letters(1, sector, occ_offset=conv['nh'], vir_offset=conv['np'])
        ket_letters = kh + kp
        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' else (lambda l: [f'a({l})'])
        # (fam, out-kind, outer builder)
        specs = [
            ('l3ft2', 'u4', lambda pl, sl: superop_element_terms_dressed(
                p_tok(pl), ket, sector, ('f',), left_ops=[['l3']],
                ket_dressing=['t2'], spin_labels=sl)),
            ('l3t2', 's4', lambda pl, sl: overlap_element_terms(
                p_tok(pl), ket, left_ops=[['l3']], ket_dressing=['t2'], spin_labels=sl)),
        ]
        for fam, okind, build in specs:
            for combo, l3_order in _COMBOS:
                for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                    out_idx = ket_letters + [p_letter]
                    for spins in iprod('ab', repeat=len(out_idx)):
                        sl = dict(zip(out_idx, spins))
                        outer = build(p_letter, sl)
                        if not outer:
                            continue
                        fused = _fuse_l3(_rename_outer_t2(outer), l3_order)
                        if not fused:
                            continue
                        suffix = ''.join(spins)
                        name = f'{okind}_{tag}_{row}_{fam}_{combo}_{suffix}'
                        if validator is not None:
                            validator(name, out_idx, fused, l3_order, spins, sector, row, fam)
                        funcs.append(assemble_laplace_fused_multikind(
                            name, PARAMS, ALIASES, 'out', _shape_expr(out_idx),
                            fused, out_idx, eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]',
                            ntau_default=6))
    return funcs


def generate():
    funcs = _build_functions()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'adc5_l3_higherorder_laplace_restricted.py')
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- do not edit by hand.\n"
                 "# Laplace-fused l3-consuming U^(4)/S^(4) stitches (l3ft2/l3t2 at the\n"
                 "# ADC(5) (l3^2,t2^2) and (l3^3,t2^1) combos); produced by\n"
                 "# generate_adc5_l3_higherorder_laplace_restricted.py (see its docstring).\n"
                 "# l3/t3/t4 are never materialized.\n")
        fh.write("import numpy as np\n"
                 "from src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"wrote {path}  ({len(funcs)} functions)")


if __name__ == '__main__':
    generate()
