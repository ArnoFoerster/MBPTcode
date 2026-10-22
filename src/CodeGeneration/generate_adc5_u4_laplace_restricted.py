"""Laplace-fused rank>=3-consuming U^(4) (0,1) stitches -- eliminates the
materialized l3/l4/t3/t4 (O(no^3 nv^3)/O(no^4 nv^4)) tensors from the ADC(5)
orbital<->k1 coupling. The amplitude-side production floor for U^(4)
(nothing production-ready while a rank>=3
amplitude is materialized).

Scope: the 6 rank>=3 structures the dense adc5_u4_blocks_restricted carries --
  l2vt3 (t3^2), l3vt2 (l3^2), l4vt2 (l4^2)   [single rank>=3 amplitude]
  l3ft3 (l3^2,t3^2), l4ft3 (l4^2,t3^2), l4ft4 (l4^2,t4^2)   [NESTED, two]
ALL at order 2, so every rank>=3 amplitude's numerator is rank<=2 -- single-
level fusion per amplitude (the two-amplitude nesting for the ft3/ft4 trio is
the validated l4ft3-type case). The rank<=2 U^(4) structures (l1ft1, l1vt2,
l2vt1) are untouched -- the dense blocks handle them.

Mechanism: the generic fuser (laplace_codegen.fuse_amplitude_into_consumer),
rank-general via lambda_canonical_and_sign, substitutes each rank>=3
amplitude's own numerator (t{r}'s numerator, relabelled into l{r}'s occ-first
order for the bra-role l-tags) into the stitch contraction, one minimax tau
point at a time -- t3/t4/l3/l4 never materialized. ntau=6 is the production
grid (validate fused-vs-exact AT ntau=6, not just the high-ntau limit).
Writes generatedADC/adc5/adc5_u4_laplace_restricted[_df].py (generated on the
user's machine, never committed).
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
    superop_element_terms_dressed, _shape_expr, _module_subdir)
from src.CodeGeneration.generate_mpn_density_restricted import (
    t_numerator_terms_restricted, T_BRA)
from src.CodeGeneration.laplace_codegen import (
    fuse_amplitude_into_consumer, assemble_laplace_fused,
    lambda_canonical_and_sign, amp_out_indices)
from src.CodeGeneration.df_codegen import dress_integral_factor

OUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'SingleReference', 'ADC',
    'generatedADC'))

# (split, hams, left_ops, ket_dressing) -- the dense adc5_u4_blocks_restricted
# rank>=3 structures, all order 2.
_U4_SPECS = [
    ('l2vt3', ('v',), [['l2']], ['t3']),
    ('l3vt2', ('v',), [['l3']], ['t2']),
    ('l4vt2', ('v',), [['l4']], ['t2']),
    ('l3ft3', ('f',), [['l3']], ['t3']),
    ('l4ft3', ('f',), [['l4']], ['t3']),
    ('l4ft4', ('f',), [['l4']], ['t4']),
]
_ORDER = 2   # every rank>=3 amplitude here is consumed at order 2

# The rank<=2 amplitudes each block's UNFUSED outer factor consumes, at their
# n=4 (ADC(5)) orders: the l2vt3 outer l2 is l2^(1) (u_order_terms n=4:
# l2vt3 -> l_order 1), and the l3vt2/l4vt2 outer t2 is t2^(1). Both are rank-2
# (<=4-index), so materialized + passed in -- the same production floor as the
# fused rank>=3 numerators, which are all built from t2^(1). Same-spin bbbb
# blocks fold onto aaaa (closed-shell). l3ft3/l4ft3/l4ft4 have no outer
# amplitude (their only leaves come from the fused numerators -> t2^(1)).
PARAMS = ['f_aa', 'f_bb', 'g_aaaa', 'g_abab', 'g_bbbb', 'd_aa', 'd_bb',
          'o', 'v', 'nv', 'no', 'eps_a', 't2_1_aaaa', 't2_1_abab',
          'l2_1_aaaa', 'l2_1_abab']
PARAMS_DF = ['f_aa', 'f_bb', 'B_aa', 'B_bb', 'd_aa', 'd_bb',
             'o', 'v', 'nv', 'no', 'eps_a', 't2_1_aaaa', 't2_1_abab',
             'l2_1_aaaa', 'l2_1_abab']
ALIASES = {'t2_aaaa': 't2_1_aaaa', 't2_abab': 't2_1_abab',
           't2_bbbb': 't2_1_aaaa',
           'l2_aaaa': 'l2_1_aaaa', 'l2_abab': 'l2_1_abab',
           'l2_bbbb': 'l2_1_aaaa'}


def _numerator_block(block, rank):
    keys = T_BRA[rank][1]
    if block in keys:
        return block
    flipped = block.translate(str.maketrans('ab', 'ba'))
    if flipped in keys:
        return flipped
    raise KeyError(f"no numerator key for rank-{rank} block {block!r}")


def _signed(terms, sign):
    return [[f"{exact_coefficient(t[0]) * sign:+.15g}"] + list(t[1:]) for t in terms]


def _highrank_tags(outer):
    """{tag: rank} for every l{r}/t{r} factor with r>=3 in the outer terms."""
    tags = {}
    for term in outer:
        for tok in term[1:]:
            m = re.match(r'^([lt])([3-9])_([ab]+)\(', tok.strip())
            if m:
                tags[f'{m.group(1)}{m.group(2)}_{m.group(3)}'] = int(m.group(2))
    return tags


def _term_has_tag(term, tag):
    """True iff `tag` appears as a factor (e.g. 'l3_aaaaaa') in this raw term."""
    return any(f.strip().startswith(tag + '(') for f in term[1:])


def _fuse_highrank(outer, df=False):
    """Fuse EVERY rank>=3 l/t amplitude factor (nested) at order _ORDER, and
    return (fused_terms, nslots).  l-tags: occ-first canonical + _to_l sign
    (lambda_canonical_and_sign); t-tags: default vir-first canonical
    (amp_out_indices), sign +1.

    CRITICAL 1 (multi-tag drop): fuse_amplitude_into_consumer DROPS every outer
    term that does NOT carry the exact `tag` being fused. So each pass fuses
    ONLY the terms carrying this tag and PRESERVES the rest -- other spin
    blocks of the same amplitude (l3_aabaab vs l3_aaaaaa), other amplitudes,
    and nested partners (l4 AND t3 in one term) are handled in later passes.

    CRITICAL 2 (double denominator): each distinct rank>=3 amplitude KIND
    ((lead, rank), e.g. ('l',4) vs ('t',3)) is an INDEPENDENTLY-denominatored
    factor, so it gets its OWN tau slot -- its exp factors are tagged
    Oe/Ve (slot 0) or Oe{s+1}/Ve{s+1} (slot s>=1) via dress_names, and
    assemble_laplace_fused integrates one tau per slot (nslots-D quadrature).
    A nested l4ft3 (kinds {('l',4),('t',3)}) is therefore 1/(D_l4 * D_t3), NOT
    the wrong 1/(D_l4 + D_t3) a single shared tau would give."""
    if df:
        outer = dress_integral_factor(outer)
    kinds = sorted({(tg[0], rk) for tg, rk in _highrank_tags(outer).items()})
    slot_of = {k: i for i, k in enumerate(kinds)}
    nslots = max(len(kinds), 1)
    terms = outer
    # one tag per pass; matching terms fused, the rest carried forward. A pass
    # strictly reduces the count of terms carrying `tag` to zero, and fusion
    # never reintroduces a rank>=3 tag, so this terminates.
    while True:
        tags = _highrank_tags(terms)
        if not tags:
            break
        tag, rank = sorted(tags.items())[0]
        lead, block = tag[0], tag.split('_', 1)[1]
        slot = slot_of[(lead, rank)]
        dn = ('Oe', 'Ve') if slot == 0 else (f'Oe{slot + 1}', f'Ve{slot + 1}')
        matching = [t for t in terms if _term_has_tag(t, tag)]
        rest = [t for t in terms if not _term_has_tag(t, tag)]
        if lead == 'l':
            same = len(set(block)) == 1
            lcs = lambda_canonical_and_sign(rank)
            canon, sign = lcs['same'] if same else lcs['mixed']
            inner = _signed(t_numerator_terms_restricted(
                rank, _ORDER, _numerator_block(block, rank)), sign)
            if df:
                inner = dress_integral_factor(inner)
            fused = fuse_amplitude_into_consumer(inner, rank, matching, tag,
                                                 canonical=canon, dress_names=dn)
        else:   # t-tag: vir-first default canonical
            inner = t_numerator_terms_restricted(
                rank, _ORDER, _numerator_block(block, rank))
            if df:
                inner = dress_integral_factor(inner)
            fused = fuse_amplitude_into_consumer(inner, rank, matching, tag,
                                                 canonical=amp_out_indices(rank),
                                                 dress_names=dn)
        terms = rest + fused
    return terms, nslots


def generate(df=False):
    from itertools import product as _iprod
    funcs = []
    for sector, tag in (('IP', '2h1p'), ('EA', '2p1h')):
        conv = class_index_convention(1, sector)
        ket = precursor_ket_string(1, sector, occ_offset=conv['nh'],
                                   vir_offset=conv['np'])
        kh, kp = class_letters(1, sector, occ_offset=conv['nh'],
                               vir_offset=conv['np'])
        ket_letters = kh + kp
        p_occ = OCC_LETTERS[2 * conv['nh']]
        p_vir = VIR_LETTERS[2 * conv['np']]
        p_tok = (lambda l: [f'a*({l})']) if sector == 'IP' \
            else (lambda l: [f'a({l})'])
        for split, hams, lo, kd in _U4_SPECS:
            for p_letter, row in ((p_occ, 'occ'), (p_vir, 'vir')):
                out_idx = ket_letters + [p_letter]
                for spins in _iprod('ab', repeat=len(out_idx)):
                    sl = dict(zip(out_idx, spins))
                    outer = superop_element_terms_dressed(
                        p_tok(p_letter), ket, sector, hams, left_ops=lo,
                        ket_dressing=kd, spin_labels=sl)
                    if not outer:
                        continue
                    fused, nslots = _fuse_highrank(outer, df=df)
                    if not fused:
                        continue
                    suffix = ''.join(spins)
                    fname = f'u4_{tag}_{row}_{split}_{suffix}'
                    print(f"{fname}: {len(outer)} outer -> {len(fused)} fused "
                          f"(nslots={nslots}, {sorted(_highrank_tags(outer))})",
                          flush=True)
                    funcs.append(assemble_laplace_fused(
                        fname, PARAMS_DF if df else PARAMS, ALIASES, 'out',
                        _shape_expr(out_idx), fused, out_idx,
                        eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]',
                        ntau_default=6, nslots=nslots))
    fname = 'adc5_u4_laplace_restricted%s.py' % ('_df' if df else '')
    path = os.path.join(_module_subdir(fname), fname)
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- do not edit by hand.\n"
                 "# Laplace-fused rank>=3 U^(4) stitches; l3/l4/t3/t4 never\n"
                 "# materialized. Produced by generate_adc5_u4_laplace_restricted.py.\n")
        fh.write("import numpy as np\n"
                 "from src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"\nwrote {path}  ({len(funcs)} functions)", flush=True)


if __name__ == '__main__':
    generate()
