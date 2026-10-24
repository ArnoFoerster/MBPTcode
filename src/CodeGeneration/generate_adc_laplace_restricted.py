"""Laplace-fused orbital<->k1 U/S stitches of the restricted ADC(4)/(5)
coupling: every dressing structure that carries a rank>=3 amplitude (l3, l4,
t3, t4: O(no^3 nv^3) and up) is emitted with that amplitude never
materialized. One spec table (SPECS) names the modules; generate(module, df)
writes one.

Mechanism (numerical, convergeable, stays Dyson): a rank-R amplitude is
numerator / D_R, and 1/D_R = -sum_tau sigma_tau exp(D_R tau) on a minimax
grid; the generic fuser (laplace_codegen.fuse_amplitude_into_consumer)
substitutes the amplitude's own numerator into each stitch's contraction,
dressed by one exp factor per external leg, one tau point at a time. A lambda
l{r} is a signed axis-transpose of t{r} (common/mp_amplitudes.to_l), so its
numerator enters relabelled into the occ-first order with the to_l sign
(laplace_codegen.lambda_canonical_and_sign: same-spin i,j,k,c,b,a with sign
(-1)^(r//2), mixed-spin i,j,k,a,b,c with +1). Validated block by block at
ntau=6 (the production grid) and at large ntau against the dense
adc{N}_u{n}_blocks_restricted twins.

Which structures: those of generate_adcn.block_structures(order) with an
amplitude of rank >= 3; pdaggerq prunes the rest (no outer terms, no
function). The surviving ones are l3ft2/l3t2 at order 3 and l2vt3, l3vt2,
l4vt2, l3ft3, l4ft3, l4ft4 / l4t3 at order 4; the rank<=2 structures stay in
the dense modules.

Two fusions:
  'flat'   every rank>=3 amplitude at order 2 (its numerator has rank<=2
           leaves only). Each distinct amplitude KIND (l4 vs t3) is an
           independently denominatored factor and gets its own tau slot
           (Oe/Ve, Oe2/Ve2, ...), so a nested l4ft3 is 1/(D_l4 D_t3), not
           1/(D_l4 + D_t3); assemble_laplace_fused integrates nslots taus.
           fuse_amplitude_into_consumer drops every outer term without the
           fused tag, so each pass fuses the matching terms and carries the
           rest forward.
  'nested' the ADC(5) n=4 amplitude assignments of the order-3 l3ft2/l3t2
           structures, (l3^(2), t2^(2)) and (l3^(3), t2^(1)): l3^(3)'s order-3
           numerator carries t3^(2)/t4^(2) leaves, themselves fused
           (laplace_codegen.fuse_nested_numerator, one tau per kind, rank<=2
           leaves order-tagged so l3^(3)'s direct t2 (order 2) and the inner
           numerators' t2 (order 1) are fed separately); the outer ket t2 is
           renamed t2out_* and fed by the driver at the combo's order
           (4 - l3 order). Names {u4|s4}_{tag}_{row}_{structure}_{combo}_{spins}.

DF x Laplace ('flat'): the outer terms and every inner numerator are DF-dressed
before fusion; dress_integral_factor's B_xx(Q,p,r) output is a plain factor to
the fuser's parser and its auxiliary letters are renamed like any dummy.

Writes generatedADC/adc{N}/<module>[_df].py (generated on the user's machine,
never committed).
"""
import re
from itertools import product

from src.CodeGeneration.codegen_common import (
    R_LEAVES, R_LEAVES_DF, R_DIMS, LAPLACE_IMPORTS, shape_expr, term_has_factor, signed_terms,
    generated_header, write_adc_module)
from src.CodeGeneration.df_codegen import dress_integral_factor
from src.CodeGeneration.generate_adcn import (
    SECTORS, superop_element_terms, overlap_element_terms, orbital_layout, block_structures,
    dressing)
from src.CodeGeneration.generate_mpn_density_spin import t_numerator_terms_stored
from src.CodeGeneration.laplace_codegen import (
    fuse_amplitude_into_consumer, assemble_laplace_fused, assemble_laplace_fused_multikind,
    lambda_canonical_and_sign, fuse_nested_numerator, highrank_tags, amp_out_indices)
from src.SingleReference.ADC.adc_order_rule import u_order_terms, s_order_terms
from src.SingleReference.ADC.precursor_classes import CLASS_TAGS

_T2_1 = ['t2_1_aaaa', 't2_1_abab']
_ALIAS_T2_1 = {'t2_aaaa': 't2_1_aaaa', 't2_abab': 't2_1_abab', 't2_bbbb': 't2_1_aaaa'}

# module -> spec. order: the U/S order whose structures are fused; at_order:
# the ADC amplitude order the 'nested' fusion evaluates them at. amps: the
# rank<=2 amplitude parameters (eps_a drives the grid), aliases: pdaggerq's
# generic leaf names -> those parameters (bbbb onto aaaa, closed shell).
SPECS = {
    'adc4_u3_l3_laplace_restricted': dict(
        order=3, kinds=('u', 's'), fusion='flat', amps=_T2_1, aliases=_ALIAS_T2_1,
        what='Laplace-fused l3-consuming U^(3)/S^(3) stitches; l3/t3 are never materialized.'),
    'adc5_u4_laplace_restricted': dict(
        order=4, kinds=('u',), fusion='flat', amps=_T2_1 + ['l2_1_aaaa', 'l2_1_abab'],
        aliases={**_ALIAS_T2_1, 'l2_aaaa': 'l2_1_aaaa', 'l2_abab': 'l2_1_abab',
                 'l2_bbbb': 'l2_1_aaaa'},
        what='Laplace-fused rank>=3 U^(4) stitches; l3/l4/t3/t4 are never materialized.'),
    'adc5_s4_laplace_restricted': dict(
        order=4, kinds=('s',), fusion='flat', amps=_T2_1 + ['l2_1_aaaa', 'l2_1_abab'],
        aliases={**_ALIAS_T2_1, 'l2_aaaa': 'l2_1_aaaa', 'l2_abab': 'l2_1_abab',
                 'l2_bbbb': 'l2_1_aaaa'},
        what='Laplace-fused rank>=3 S^(4) overlap (l4t3); l4/t3 are never materialized.'),
    'adc5_l3_higherorder_laplace_restricted': dict(
        order=3, at_order=4, kinds=('u', 's'), fusion='nested',
        amps=_T2_1 + ['t2_2_aaaa', 't2_2_abab', 't1_2_aa', 't2out_aaaa', 't2out_abab'],
        aliases={'t2_1_bbbb': 't2_1_aaaa', 't2_2_bbbb': 't2_2_aaaa', 't1_2_bb': 't1_2_aa',
                 't2out_bbbb': 't2out_aaaa'},
        what='Laplace-fused l3ft2/l3t2 U^(4)/S^(4) stitches at the ADC(5) (l3^2,t2^2) and '
             '(l3^3,t2^1) combos; l3/t3/t4 are never materialized.'),
}

_FLAT_ORDER = 2   # every rank>=3 amplitude of a 'flat' module is consumed at order 2


def highrank_structures(order, kind):
    """The structures of block_structures(order, kind) with a rank>=3 amplitude."""
    out = {}
    for nm, spec in block_structures(order, kind).items():
        j, k = (spec[0], spec[2]) if kind == 'u' else spec
        if max(j, k) >= 3:
            out[nm] = spec
    return out


def _fuse_flat(outer, df=False):
    """Fuse every rank>=3 l/t factor of `outer` at order 2, one tau slot per
    amplitude kind; returns (fused terms, nslots). l-tags fuse through the
    occ-first canonical and the to_l sign, t-tags through the vir-first
    amp_out_indices order with sign +1."""
    if df:
        outer = dress_integral_factor(outer)
    kinds = sorted({(tg[0], rk) for tg, rk in highrank_tags(outer).items()})
    slot_of = {k: i for i, k in enumerate(kinds)}
    terms = outer
    # one tag per pass; a pass removes every term carrying `tag` and fusion
    # never reintroduces a rank>=3 tag, so this terminates
    while True:
        tags = highrank_tags(terms)
        if not tags:
            break
        tag, rank = sorted(tags.items())[0]
        lead, block = tag[0], tag.split('_', 1)[1]
        slot = slot_of[(lead, rank)]
        dn = ('Oe', 'Ve') if slot == 0 else (f'Oe{slot + 1}', f'Ve{slot + 1}')
        matching = [t for t in terms if term_has_factor(t, tag)]
        rest = [t for t in terms if not term_has_factor(t, tag)]
        inner = t_numerator_terms_stored(rank, _FLAT_ORDER, block)
        if lead == 'l':
            lcs = lambda_canonical_and_sign(rank)
            canon, sign = lcs['same'] if len(set(block)) == 1 else lcs['mixed']
            inner = signed_terms(inner, sign)
        else:
            canon = amp_out_indices(rank)
        if df:
            inner = dress_integral_factor(inner)
        terms = rest + fuse_amplitude_into_consumer(inner, rank, matching, tag,
                                                    canonical=canon, dress_names=dn)
    return terms, max(len(kinds), 1)


def _fuse_nested(outer, l_order, rank=3):
    """The 'nested' fusion: the outer ket t2 renamed t2out_*, every outer
    l{rank} tag fused with its fuse_nested_numerator numerator at `l_order`
    through the to_l canonical (dressing kind l{rank})."""
    terms = [[t[0]] + [re.sub(r'^t2_', 't2out_', f.strip()) for f in t[1:]] for t in outer]
    while True:
        tags = sorted(t for t in highrank_tags(terms) if t.startswith(f'l{rank}_'))
        if not tags:
            break
        tag = tags[0]
        block = tag.split('_', 1)[1]
        lcs = lambda_canonical_and_sign(rank)
        canon, sign = lcs['same'] if len(set(block)) == 1 else lcs['mixed']
        inner = signed_terms(fuse_nested_numerator(t_numerator_terms_stored, rank, l_order, block),
                             sign)
        matching = [t for t in terms if term_has_factor(t, tag)]
        rest = [t for t in terms if not term_has_factor(t, tag)]
        terms = rest + fuse_amplitude_into_consumer(
            inner, rank, matching, tag, canonical=canon,
            dress_names=(f'Oe_l{rank}', f'Ve_l{rank}'))
    return terms


def _combos(nm, kind, at_order):
    """(combo suffix, l order) of the at_order amplitude assignments of a
    structure (u_order_terms / s_order_terms), highest l order first."""
    rows = (u_order_terms(at_order) if kind == 'u' else s_order_terms(at_order))
    return sorted(((f'{r[-2]}{r[-1]}', r[-2]) for r in rows if r[0] == nm), key=lambda c: -c[1])


def build_functions(module, df=False, validator=None):
    """The function sources of one SPECS module. validator, if given, is
    called (name, out_idx, fused, l_order, spins, sector, row, structure) for
    every 'nested' block as it is built."""
    spec = SPECS[module]
    if df and spec['fusion'] != 'flat':
        raise ValueError(f'{module}: no DF/RI twin')
    params = (R_LEAVES_DF if df else R_LEAVES) + R_DIMS + ['eps_a'] + spec['amps']
    order = spec['order']
    funcs = []
    for sector in SECTORS:
        tag = CLASS_TAGS[1][sector]
        ket, ket_letters, rows, p_tok = orbital_layout(1, sector)
        for kind in spec['kinds']:
            fam = f"{kind}{spec.get('at_order', order)}"
            for nm, st in sorted(highrank_structures(order, kind).items()):
                lo, kd, _ = dressing(st)
                combos = (_combos(nm, kind, spec['at_order']) if spec['fusion'] == 'nested'
                          else [(None, None)])
                for combo, l_order in combos:
                    for p, row in rows:
                        out_idx = ket_letters + [p]
                        for spins in product('ab', repeat=len(out_idx)):
                            sl = dict(zip(out_idx, spins))
                            outer = (superop_element_terms(p_tok(p), ket, sector, (st[1],),
                                                           left_ops=lo, ket_dressing=kd, spin_labels=sl)
                                     if kind == 'u' else
                                     overlap_element_terms(p_tok(p), ket, left_ops=lo,
                                                           ket_dressing=kd, spin_labels=sl))
                            if not outer:
                                continue
                            suffix = ''.join(spins)
                            if spec['fusion'] == 'flat':
                                fused, nslots = _fuse_flat(outer, df=df)
                                if not fused:
                                    continue
                                name = f'{fam}_{tag}_{row}_{nm}_{suffix}'
                                print(f"{name}: {len(outer)} outer -> {len(fused)} fused "
                                      f"(nslots={nslots})", flush=True)
                                funcs.append(assemble_laplace_fused(
                                    name, params, spec['aliases'], 'out', shape_expr(out_idx),
                                    fused, out_idx, eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]',
                                    ntau_default=6, nslots=nslots))
                            else:
                                fused = _fuse_nested(outer, l_order)
                                if not fused:
                                    continue
                                name = f'{fam}_{tag}_{row}_{nm}_{combo}_{suffix}'
                                if validator is not None:
                                    validator(name, out_idx, fused, l_order, spins, sector, row, nm)
                                funcs.append(assemble_laplace_fused_multikind(
                                    name, params, spec['aliases'], 'out', shape_expr(out_idx),
                                    fused, out_idx, eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]',
                                    ntau_default=6))
    return funcs


def generate(module, df=False):
    """Write generatedADC/adc{N}/<module>[_df].py."""
    funcs = build_functions(module, df=df)
    fname = f"{module}{'_df' if df else ''}.py"
    write_adc_module(fname, funcs, generated_header(
        f'generate_adc_laplace_restricted.py ({module}{", df" if df else ""})',
        SPECS[module]['what'], "See that module's docstring for the fusion."),
        imports=LAPLACE_IMPORTS)
    print(f"{fname}: {len(funcs)} functions", flush=True)


if __name__ == '__main__':
    for m in SPECS:
        generate(m)
