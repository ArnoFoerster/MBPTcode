"""Codegen script (run once, output committed): the Laplace-fused pieces of
the spin-blocked MP3/MP4 density, so no rank>=3 amplitude (t3_2, t4_2, t3_3)
is materialized. Each piece substitutes the amplitude's own numerator into its
consumer through laplace_codegen (1/D = -sum_tau sigma_tau exp(D tau), one exp
factor per external leg); one table (MODULES) names every committed module.

Pieces:
- mp3_t3_laplace_restricted[_df]: the cross-spin T3^(2) blocks (aabaab,
  abbabb) fused into their only two live consumers, t1_3_aa_numerator and
  m3_ov_12_restricted (the oo/vv m3_12 and all m3_21 bodies declare t3/l3 as
  dead parameters); t3_2_aaaaaa is the driver's own
  _laplace_aaaaaa_contribution. Their numerators reference t2_aaaa AND t2_abab
  (_connect_ranks(3, 2) = rank 2), and abbabb's the all-beta t2_bbbb, aliased
  to its t2_aaaa twin. The m3_ov terms also carry a bare l2 bra factor.
- mp3_t3_laplace_unrestricted[_df]: all four T3^(2) blocks into
  t1_3_{aa,bb}_numerator and m3_ov_{a,b}_12_unrestricted. A UHF block's
  external legs carry mixed alpha/beta spin, so each dressing needs the array
  of ITS OWN spin: the fuser appends one Oe(x)/Ve(x) factor per external leg in
  amp_out_indices order as the last 2*rank factors of every term, and
  _relabel_dressings renames them positionally to Oe_a/Oe_b/Ve_a/Ve_b from the
  block's pair-spin pattern (block[:rank]); one shared grid covers both spins.
- mp4_laplace_restricted: T4^(2)_aaaaaaaa into its two single-role (ket-only)
  consumers, t2_3_aaaa (1 of 18 raw terms) and t3_3_aaaaaa (4 of 36). The
  m4_22/overlap4 pieces carry t4 in BOTH bra and ket roles (the l-tag frame and
  a 1/D4^2 double quadrature), not attempted, so t4_2_aaaaaaaa is still built
  once for those.
- mp4_t2_3_laplace_restricted: every rank>=3 leaf of t2_3's numerator (t3_2,
  t4_2, all spins), one function per leaf tag through the split builder
  fuse_amplitude_rank3_tags; the mixed-order rank<=2 direct part is the
  compiled t2_3 numerator with t3/t4 zeroed. Consumed by
  compute_t2_3_laplace: t2_3 = (direct + sum_tag fused - E2 t2_1) / D_2.
- mp4_t1_4_laplace_restricted: t1_4 nested two deep (t1_4 -> t3_3 ->
  {t3_2, t4_2}) through fuse_nested_numerator; depth-qualified kinds give each
  amplitude instance its own tau. Consumed by compute_t1_4_laplace:
  t1_4 = (fused - E2 t1_2) / D_1.

DF x Laplace: the outer terms and every inner numerator are DF-dressed before
fusion (dress_integral_factor's B_xx(Q,p,r) is a plain factor to the fuser's
parser; its auxiliary letters are renamed like any dummy), validated against
the dense twin to ~1e-17 (H2O/6-31g, naux = norb^2).
"""
import os
import sys

from src.CodeGeneration import codegen_common
from src.CodeGeneration.codegen_common import (
    eri_leaves, R_DIMS, U_DIMS, LAPLACE_IMPORTS, generated_header, write_module)
from src.CodeGeneration.df_codegen import dress_integral_factor
from src.CodeGeneration.generate_mpn_density import amp_out_indices
from src.CodeGeneration.generate_mpn_density_spin import (
    RESTRICTED, UNRESTRICTED, t_numerator_terms, t_numerator_terms_stored, cross_density_terms)
from src.CodeGeneration.laplace_codegen import (
    fuse_amplitude_into_consumer, assemble_laplace_fused, assemble_laplace_fused_multikind,
    fuse_nested_numerator, fuse_amplitude_rank3_tags)

_T2_1_R = {'t2_aaaa': 't2_1_aaaa', 't2_abab': 't2_1_abab', 't2_bbbb': 't2_1_aaaa'}
_L2_1_R = {'l2_aaaa': 'l_2_1_aaaa', 'l2_abab': 'l_2_1_abab', 'l2_bbbb': 'l_2_1_aaaa'}
_T2_1_U = {f't2_{b}': f't2_1_{b}' for b in UNRESTRICTED.blocks(2)}
_L2_1_U = {f'l2_{b}': f'l_2_1_{b}' for b in UNRESTRICTED.blocks(2)}


def _relabel_dressings(fused_terms, inner_rank, block):
    """Rename the trailing 2*inner_rank Oe(x)/Ve(x) dressings of each fused
    term to Oe_<s>/Ve_<s> by the block's pair-spin pattern, twice (vir pairs,
    then occ pairs: amp_out_indices order)."""
    spins = list(block[:inner_rank]) * 2
    n = 2 * inner_rank
    out = []
    for term in fused_terms:
        head, dress = term[1:-n], term[-n:]
        out.append([term[0]] + list(head) + [f"{f.split('(', 1)[0]}_{s}({f.split('(', 1)[1]}"
                                             for f, s in zip(dress, spins)])
    return out


def _fuse_blocks(model, outer, rank, blocks, df):
    """Every listed rank-`rank` order-2 amplitude block fused into `outer`
    (unrestricted: dressings relabelled per spin)."""
    if df:
        outer = dress_integral_factor(outer)
    fused = []
    for block in blocks:
        inner = t_numerator_terms(model, rank, 2, block)
        if df:
            inner = dress_integral_factor(inner)
        one = fuse_amplitude_into_consumer(inner, rank, outer, f't{rank}_{block}')
        fused += one if model.restricted else _relabel_dressings(one, rank, block)
    return fused


def _consumer_pieces(model, consumers, rank, blocks, df, name_tail):
    """One Laplace function per consumer (name, out var, shape, outer terms,
    out indices, extra params, aliases)."""
    params0 = eri_leaves(df) + (['d_aa'] + R_DIMS + ['eps_a'] if model.restricted else
                                ['d_aa', 'd_bb'] + U_DIMS + ['eps_a', 'eps_b'])
    funcs = []
    for name, out_var, shape, outer, out_idx, extra, aliases in consumers:
        fused = _fuse_blocks(model, outer, rank, blocks, df)
        name = name + name_tail(df)
        print(f"{name}: {len(fused)} fused terms", flush=True)
        funcs.append(assemble_laplace_fused(name, params0 + extra, aliases, out_var, shape, fused,
                                            out_idx, ntau_default=6,
                                            unrestricted=not model.restricted))
    return funcs


def mp3_t3_restricted(df=False):
    R = RESTRICTED
    t2 = ['t2_1_aaaa', 't2_1_abab']
    consumers = [
        ('t1_3_aa_t3crossspin_laplace', 't1_3_aa_t3cs', '(nv, no)',
         t_numerator_terms(R, 1, 3, 'aa'), ['a', 'i'], t2, dict(_T2_1_R)),
        # bra rank 2 order 1, ket ranks 1-3 order 2 (mpn_density_pieces' m3_ov_12)
        ('m3_ov_a_t3crossspin_laplace', 'm3_ov_t3cs', '(no, nv)',
         cross_density_terms('ov', 'a', [2], [1, 2, 3]), ['m', 'e'],
         t2 + ['l_2_1_aaaa', 'l_2_1_abab'], {**_T2_1_R, **_L2_1_R}),
    ]
    return _consumer_pieces(R, consumers, 3, ('aabaab', 'abbabb'), df,
                            lambda d: '_df' if d else '')


def mp3_t3_unrestricted(df=False):
    U = UNRESTRICTED
    t2 = [f't2_1_{b}' for b in U.blocks(2)]
    l2 = [f'l_2_1_{b}' for b in U.blocks(2)]
    consumers = [
        ('t1_3_aa_t3_laplace', 't1_3_aa_t3', '(nv_a, no_a)', t_numerator_terms(U, 1, 3, 'aa'),
         ['a', 'i'], t2, dict(_T2_1_U)),
        ('t1_3_bb_t3_laplace', 't1_3_bb_t3', '(nv_b, no_b)', t_numerator_terms(U, 1, 3, 'bb'),
         ['a', 'i'], t2, dict(_T2_1_U)),
        ('m3_ov_a_t3_laplace', 'm3_ov_a_t3', '(no_a, nv_a)',
         cross_density_terms('ov', 'a', [2], [1, 2, 3]), ['m', 'e'], t2 + l2, {**_T2_1_U, **_L2_1_U}),
        ('m3_ov_b_t3_laplace', 'm3_ov_b_t3', '(no_b, nv_b)',
         cross_density_terms('ov', 'b', [2], [1, 2, 3]), ['m', 'e'], t2 + l2, {**_T2_1_U, **_L2_1_U}),
    ]
    return _consumer_pieces(U, consumers, 3, list(U.blocks(3)), df, lambda d: '_df' if d else '')


def mp4_t4_restricted():
    R = RESTRICTED
    consumers = [
        ('t2_3_aaaa_t4_laplace', 't2_3_aaaa_t4', '(nv, nv, no, no)',
         t_numerator_terms(R, 2, 3, 'aaaa'), amp_out_indices(2), ['t2_1_aaaa'],
         {'t2_aaaa': 't2_1_aaaa'}),
        ('t3_3_aaaaaa_t4_laplace', 't3_3_aaaaaa_t4', '(nv, nv, nv, no, no, no)',
         t_numerator_terms(R, 3, 3, 'aaaaaa'), amp_out_indices(3), ['t2_1_aaaa'],
         {'t2_aaaa': 't2_1_aaaa'}),
    ]
    return _consumer_pieces(R, consumers, 4, ('aaaaaaaa',), False, lambda d: '')


def mp4_t2_3_restricted():
    # the fused contractions reference only the two-body integrals and the
    # order-1 T2^(1) leaves (no Fock or d factor survives)
    params = eri_leaves(False) + R_DIMS + ['eps_a', 't2_1_aaaa', 't2_1_abab']
    funcs = []
    for block in ('aaaa', 'abab'):
        for tag, (rk, fused) in sorted(fuse_amplitude_rank3_tags(
                t_numerator_terms_stored, 2, 3, block).items()):
            name = f't2_3_{block}_{tag}_laplace'
            print(f"{name}: {len(fused)} fused terms", flush=True)
            funcs.append(assemble_laplace_fused(name, params, dict(_T2_1_R), 'out',
                                                '(nv, nv, no, no)', fused, amp_out_indices(2),
                                                ntau_default=6))
    return funcs


def mp4_t1_4_restricted():
    params = (eri_leaves(False) + R_DIMS + ['eps_a', 't1_3_aa', 't2_3_aaaa', 't2_3_abab',
                                           't2_2_aaaa', 't2_2_abab', 't1_2_aa', 't2_1_aaaa',
                                           't2_1_abab'])
    aliases = {'t1_3_bb': 't1_3_aa', 't2_3_bbbb': 't2_3_aaaa', 't2_2_bbbb': 't2_2_aaaa',
               't1_2_bb': 't1_2_aa', 't2_1_bbbb': 't2_1_aaaa'}
    fused = fuse_nested_numerator(t_numerator_terms_stored, 1, 4, 'aa')
    print(f"t1_4_aa_laplace: {len(fused)} fused terms", flush=True)
    return [assemble_laplace_fused_multikind('t1_4_aa_laplace', params, aliases, 'out', '(nv, no)',
                                             fused, amp_out_indices(1), eps_o_expr='eps_a[o]',
                                             eps_v_expr='eps_a[v]', ntau_default=6)]


# module -> (output package, builder, what it is)
MODULES = {
    'mp3_t3_laplace_restricted': ('restricted', lambda: mp3_t3_restricted(),
                                  'Laplace-fused cross-spin T3^(2) (aabaab, abbabb) contribution to '
                                  't1_3_aa / m3_ov_12.'),
    'mp3_t3_laplace_restricted_df': ('restricted', lambda: mp3_t3_restricted(df=True),
                                     'DF/RI twin of mp3_t3_laplace_restricted (B_aa/B_bb).'),
    'mp3_t3_laplace_unrestricted': ('unrestricted', lambda: mp3_t3_unrestricted(),
                                    'Laplace-fused T3^(2) (all four spin blocks) contribution to '
                                    't1_3_{aa,bb} / m3_ov_{a,b}_12.'),
    'mp3_t3_laplace_unrestricted_df': ('unrestricted', lambda: mp3_t3_unrestricted(df=True),
                                       'DF/RI twin of mp3_t3_laplace_unrestricted (B_aa/B_bb).'),
    'mp4_laplace_restricted': ('restricted', mp4_t4_restricted,
                               'Laplace-fused T4^(2)_aaaaaaaa contribution to t2_3_aaaa / t3_3_aaaaaa.'),
    'mp4_t2_3_laplace_restricted': ('restricted', mp4_t2_3_restricted,
                                    'Fully Laplace-fused t2_3 numerator, one function per rank>=3 '
                                    'leaf tag (compute_t2_3_laplace).'),
    'mp4_t1_4_laplace_restricted': ('restricted', mp4_t1_4_restricted,
                                    'Fully Laplace-fused t1_4 numerator, nested t1_4 -> t3_3 -> '
                                    '{t3_2, t4_2} (compute_t1_4_laplace).'),
}


def generate(module):
    package, build, what = MODULES[module]
    write_module(os.path.join(codegen_common.DM_OUT_DIR, f'generated_mpn_{package}', f'{module}.py'), build(),
                 generated_header(f'generate_mpn_laplace.py ({module})', what,
                                  'No rank>=3 amplitude is materialized; see the generator docstring.'),
                 imports=LAPLACE_IMPORTS)


if __name__ == '__main__':
    for m in (sys.argv[1:] or MODULES):
        generate(m)
