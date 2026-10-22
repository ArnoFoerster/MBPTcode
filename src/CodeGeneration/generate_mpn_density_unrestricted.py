"""Codegen script (run once, output committed): unrestricted (spin-blocked,
genuinely separate alpha/beta channels, UHF) analogue of
generate_mpn_density_restricted.py -- same order-generic pdaggerq recursion
(see generate_mpn_density.py's docstring for the physics), derived with
pdaggerq's spin-integration feature exactly like the restricted module, but
with NO closed-shell identifications: every spin block is its own stored
array (t2_aaaa/t2_abab/t2_bbbb all distinct, the all-beta block is NOT the
all-alpha block's twin), every occ/vir dimension is per-spin (no_a/nv_a/
no_b/nv_b), and the one-body density insertion is derived separately for the
alpha and the beta channel (both are needed for UHF; the restricted module
only ever derives the alpha block).

Scope: MP2 + MP3 density only (Delta_gamma^(2), Delta_gamma^(3)) -- that is
all the UHF production branches (compute_mp2/mp3_density_matrix_ao,
build_mp2/mp3_static_correction) consume. MAX rank is therefore 3:
- rank 4 of Psi^(2) cannot couple to <D2| through a one-body operator, so
  T4^(2) contributes exactly nothing below MP4 (same argument, and same
  verified-zero pdaggerq output, as the restricted module's m3 pieces);
- ranks 2/3 of Psi^(3) cannot couple to <Phi0| through a one-body operator,
  so the m3 30/03 boundary pieces only ever need T1^(3) (the restricted
  generated m3_*_30/03 bodies have zero terms for rank >= 2).

Codegen backend: reuses generate_mpn_density.py's raw_terms_to_einsum via a
local assemble_u wrapper whose ONE genuinely new job is spin-aware slicing --
pdaggerq_interface._array_expr slices full-size tensors with spin-blind
'o'/'v' letters, which is correct for the restricted module (one shared
orbital window) but wrong for UHF, where e.g. <i,j||a,b>_abab must be sliced
g_abab[o_a, o_b, v_a, v_b]. The factor's own spin suffix determines the spin
of every axis positionally, so _spinify_slices rewrites each generated slice
list from that suffix (a pure source-to-source pass over the emitted einsum
lines; amplitude leaves carry no slice brackets and are untouched).
"""
import os
import re

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Base.pdaggerq_interface import pq_helper, raw_terms_to_einsum
from src.SingleReference.ADC.adc_order_rule import refuse_above_max_order
from src.CodeGeneration.generate_mpn_density import (
    ranks_present, _filter_connected, amp_out_indices,
)
from src.CodeGeneration.df_codegen import dress_integral_factor

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'SingleReference', 'DensityMatrix', 'generated_mpn_unrestricted')

MAX_RANK_U = 3  # MP2+MP3 density only -- see module docstring


def _build_t_bra_u(rank):
    """Bra string + spin-block dict for one rank -- same construction as
    generate_mpn_density_restricted._build_t_bra (bra = 'e{rank}(occ
    ascending, vir DESCENDING)', pair p = (occ[p], vir[p]) sharing one spin,
    canonical blocks put the beta pairs at the END of the pair ordering),
    except k runs 0..rank INCLUSIVE: the all-beta block is a genuinely
    distinct stored array here, not the all-alpha block's closed-shell twin.
    """
    occ_letters = list('ijklmn'[:rank])
    vir_letters = list('abcdef'[:rank])
    bra = f'e{rank}(' + ','.join(occ_letters + list(reversed(vir_letters))) + ')'
    blocks = {}
    for k in range(rank + 1):
        spins = ['a'] * (rank - k) + ['b'] * k
        block_name = ''.join(spins) * 2
        spin_labels = {}
        for p in range(rank):
            spin_labels[occ_letters[p]] = spins[p]
            spin_labels[vir_letters[p]] = spins[p]
        blocks[block_name] = spin_labels
    return bra, blocks


T_BRA_U = {r: _build_t_bra_u(r) for r in range(1, MAX_RANK_U + 1)}
ALL_BLOCKS_U = {r: list(T_BRA_U[r][1].keys()) for r in T_BRA_U}

COMMON_ARGS_U = ['g_aaaa', 'g_abab', 'g_bbbb', 'd_aa', 'd_bb',
                 'o_a', 'v_a', 'o_b', 'v_b', 'nv_a', 'no_a', 'nv_b', 'no_b']

_SPIN_SLICE_RE = re.compile(r'\b([gdf])(_[ab]+)\[([ov](?:, [ov])*)\]')
# B (DF/RI factor)'s slice list carries a leading unsliced aux axis (':')
# ahead of the o/v tokens -- see pdaggerq_interface._array_expr's B branch --
# so it needs its own pattern rather than reusing _SPIN_SLICE_RE's (which
# assumes every bracketed token is an o/v slice). B_aa/B_bb's suffix is
# always two equal letters (both axes share the block's one spin, per
# df_codegen._b_name), so both o/v tokens get the same '_a'/'_b' suffix.
_SPIN_SLICE_RE_B = re.compile(r'\bB(_[ab]+)\[:, ([ov](?:, [ov])*)\]')


def _spinify_slices(line):
    """Rewrite spin-blind slice lists emitted by raw_terms_to_einsum into
    per-spin ones using the factor's own spin suffix positionally:
    'g_abab[o, o, v, v]' -> 'g_abab[o_a, o_b, v_a, v_b]'. See module
    docstring."""
    def repl(m):
        name, suffix, slices = m.group(1), m.group(2), m.group(3)
        spins = suffix[1:]
        parts = [p.strip() for p in slices.split(',')]
        if len(parts) != len(spins):
            raise ValueError(f"slice/suffix arity mismatch in '{m.group(0)}'")
        return name + suffix + '[' + ', '.join(f'{p}_{s}' for p, s in zip(parts, spins)) + ']'

    def repl_b(m):
        suffix, slices = m.group(1), m.group(2)
        spins = suffix[1:]
        parts = [p.strip() for p in slices.split(',')]
        if len(parts) != len(spins):
            raise ValueError(f"slice/suffix arity mismatch in '{m.group(0)}'")
        return 'B' + suffix + '[:, ' + ', '.join(f'{p}_{s}' for p, s in zip(parts, spins)) + ']'

    line = _SPIN_SLICE_RE.sub(repl, line)
    return _SPIN_SLICE_RE_B.sub(repl_b, line)


def assemble_u(func_name, params, aliases, out_var, out_shape_expr, raw_terms, out_indices):
    """generate_mpn_density.assemble with the _spinify_slices pass applied to
    every emitted einsum line."""
    src = [f"def {func_name}({', '.join(params)}):"]
    for generic, real in aliases.items():
        if generic != real:
            src.append(f"    {generic} = {real}")
    src.append(f"    {out_var} = np.zeros({out_shape_expr})")
    src.extend(_spinify_slices(line) for line in raw_terms_to_einsum(raw_terms, out_var, out_indices))
    src.append(f"    return {out_var}")
    src.append("")
    return "\n".join(src)


def _amp_shape_u(block):
    """Storage t{rank}_{block}[vir(rank)..., occ(rank)...] with per-spin
    dimensions taken from the block's pair spin pattern (block[:rank], since
    occ pattern == vir pattern by construction)."""
    rank = len(block) // 2
    spins = block[:rank]
    return '(' + ', '.join([f'nv_{s}' for s in spins] + [f'no_{s}' for s in spins]) + ')'


def _connect_ranks_u(rank, order):
    return ([r for r in ranks_present(order - 1) if abs(r - rank) <= 2 and r <= MAX_RANK_U]
            if order > 1 else [])


_numerator_cache_u = {}


def t_numerator_terms_u(rank, order, block):
    key = (rank, order, block)
    if key in _numerator_cache_u:
        return _numerator_cache_u[key]
    bra, blocks = T_BRA_U[rank]
    spin_labels = blocks[block]

    if order == 1:
        terms = []
        if rank == 2:
            pq = pq_helper('fermi')
            pq.set_left_operators([[bra]])
            pq.set_right_operators([['1']])
            pq.add_operator_product(1.0, ['v'])
            pq.simplify()
            terms = _filter_connected(pq.strings(spin_labels=spin_labels), amp_out_indices(rank))
            pq.clear()
        _numerator_cache_u[key] = terms
        return terms

    connect_ranks = _connect_ranks_u(rank, order)
    terms = []
    if connect_ranks:
        pq = pq_helper('fermi')
        pq.set_left_operators([[bra]])
        pq.set_right_operators([['1']])
        for r in connect_ranks:
            pq.add_operator_product(1.0, ['v', f't{r}'])
        pq.simplify()
        terms = _filter_connected(pq.strings(spin_labels=spin_labels), amp_out_indices(rank))
        pq.clear()
    _numerator_cache_u[key] = terms
    return terms


def _t_aliases_and_params_u(connect_ranks, order):
    aliases, params = {}, list(COMMON_ARGS_U)
    for r in connect_ranks:
        for sub in ALL_BLOCKS_U[r]:
            real_name = f't{r}_{order - 1}_{sub}'
            aliases[f't{r}_{sub}'] = real_name
            if real_name not in params:
                params.append(real_name)
    return aliases, params


def amplitude_numerator_funcs_u(rank, order):
    refuse_above_max_order(order + 1, 'amplitude_numerator_funcs_u')
    aliases, params = _t_aliases_and_params_u(_connect_ranks_u(rank, order), order)
    funcs = []
    for block in T_BRA_U[rank][1]:
        terms = t_numerator_terms_u(rank, order, block)
        name = f't{rank}_{order}_{block}_numerator'
        print(f"{name}: {len(terms)} terms")
        funcs.append(assemble_u(name, params, aliases, 'result', _amp_shape_u(block),
                                terms, amp_out_indices(rank)))
    return funcs


COMMON_ARGS_U_DF = ['B_aa', 'B_bb', 'd_aa', 'd_bb',
                    'o_a', 'v_a', 'o_b', 'v_b', 'nv_a', 'no_a', 'nv_b', 'no_b']


def _t_aliases_and_params_u_df(connect_ranks, order):
    """Like _t_aliases_and_params_u, but starting from COMMON_ARGS_U_DF
    (B_aa/B_bb) instead of COMMON_ARGS_U (g_aaaa/g_abab/g_bbbb) -- see
    amplitude_numerator_funcs_u_df."""
    aliases, params = {}, list(COMMON_ARGS_U_DF)
    for r in connect_ranks:
        for sub in ALL_BLOCKS_U[r]:
            real_name = f't{r}_{order - 1}_{sub}'
            aliases[f't{r}_{sub}'] = real_name
            if real_name not in params:
                params.append(real_name)
    return aliases, params


def amplitude_numerator_funcs_u_df(rank, order):
    """DF/RI variant of amplitude_numerator_funcs_u (see
    the DF rewrite, Phase 1) -- restricted module's
    _amplitude_numerator_funcs_restricted_df's UHF analogue. Only meant to
    be called for (rank=2, order=1) = t2_1 and (rank=1, order=2) = t1_2, the
    only n=2 pieces carrying a bracket integral factor (same reasoning as
    the restricted module: overlap2/m2_*_11/20/02 have none, confirmed by
    inspection, and are reused unchanged from the dense generated file)."""
    refuse_above_max_order(order + 1, 'amplitude_numerator_funcs_u_df')
    aliases, params = _t_aliases_and_params_u_df(_connect_ranks_u(rank, order), order)
    funcs = []
    for block in T_BRA_U[rank][1]:
        terms = t_numerator_terms_u(rank, order, block)
        df_terms = dress_integral_factor(terms)
        name = f't{rank}_{order}_{block}_numerator_df'
        print(f"{name}: {len(df_terms)} DF terms (from {len(terms)} dense terms)")
        funcs.append(assemble_u(name, params, aliases, 'result', _amp_shape_u(block),
                                df_terms, amp_out_indices(rank)))
    return funcs


def amplitude_numerator_no_t3_funcs_u(rank, order, max_connect_rank):
    """Same physics/derivation as amplitude_numerator_funcs_u, but excludes
    any connect rank above max_connect_rank -- produces a '..._no_t3'-
    suffixed numerator variant whose higher-rank (T3^(2)) contribution is
    supplied externally instead of materialized as a rank>4 array (see
    generate_mp3_t3_laplace_unrestricted.py, restricted analogue's own
    _amplitude_numerator_no_t3_funcs_restricted). Used for
    t1_3_{aa,bb}_numerator's production (laplace-mode) variant: connect_ranks
    capped at 2 means these bodies never reference any t3_2_* block at all,
    not even as a dead param."""
    connect_ranks = [r for r in _connect_ranks_u(rank, order) if r <= max_connect_rank]
    aliases, params = _t_aliases_and_params_u(connect_ranks, order)
    funcs = []
    for block in T_BRA_U[rank][1]:
        bra, blocks = T_BRA_U[rank]
        spin_labels = blocks[block]
        terms = []
        if connect_ranks:
            pq = pq_helper('fermi')
            pq.set_left_operators([[bra]])
            pq.set_right_operators([['1']])
            for r in connect_ranks:
                pq.add_operator_product(1.0, ['v', f't{r}'])
            pq.simplify()
            terms = _filter_connected(pq.strings(spin_labels=spin_labels), amp_out_indices(rank))
            pq.clear()
        name = f't{rank}_{order}_{block}_numerator_no_t3'
        print(f"{name}: {len(terms)} terms")
        funcs.append(assemble_u(name, params, aliases, 'result', _amp_shape_u(block),
                                terms, amp_out_indices(rank)))
    return funcs


def amplitude_numerator_no_t3_funcs_u_df(rank, order, max_connect_rank):
    """DF/RI variant of amplitude_numerator_no_t3_funcs_u (Phase 3) --
    production t1_3_{aa,bb}_numerator_no_t3's DF twin."""
    connect_ranks = [r for r in _connect_ranks_u(rank, order) if r <= max_connect_rank]
    aliases, params = _t_aliases_and_params_u_df(connect_ranks, order)
    funcs = []
    for block in T_BRA_U[rank][1]:
        bra, blocks = T_BRA_U[rank]
        spin_labels = blocks[block]
        terms = []
        if connect_ranks:
            pq = pq_helper('fermi')
            pq.set_left_operators([[bra]])
            pq.set_right_operators([['1']])
            for r in connect_ranks:
                pq.add_operator_product(1.0, ['v', f't{r}'])
            pq.simplify()
            terms = _filter_connected(pq.strings(spin_labels=spin_labels), amp_out_indices(rank))
            pq.clear()
        df_terms = dress_integral_factor(terms)
        name = f't{rank}_{order}_{block}_numerator_no_t3_df'
        print(f"{name}: {len(df_terms)} DF terms (from {len(terms)} dense terms)")
        funcs.append(assemble_u(name, params, aliases, 'result', _amp_shape_u(block),
                                df_terms, amp_out_indices(rank)))
    return funcs


def generate_overlap_u(rank):
    """<l{rank}|t{rank}> scalar, all spin blocks distinct params (no twin
    aliasing); which order's amplitudes play the l/t roles is the driver's
    call-site choice, as in the restricted module."""
    pq = pq_helper('fermi')
    pq.set_left_operators([[f'l{rank}']])
    pq.set_right_operators([['1']])
    pq.add_operator_product(1.0, [f't{rank}'])
    pq.simplify()
    terms = pq.strings(spin_labels={})
    pq.clear()
    aliases = {}
    for sub in ALL_BLOCKS_U[rank]:
        aliases[f'l{rank}_{sub}'] = f'l_{sub}'
        aliases[f't{rank}_{sub}'] = f't_{sub}'
    params = [f'l_{s}' for s in ALL_BLOCKS_U[rank]] + [f't_{s}' for s in ALL_BLOCKS_U[rank]]
    name = f'overlap{rank}_unrestricted'
    print(f"{name}: {len(terms)} terms")
    return assemble_u(name, params, aliases, 'result', '()', terms, [])


DENSITY_BLOCKS_U = {}
for _s in ('a', 'b'):
    DENSITY_BLOCKS_U[('oo', _s)] = ('e1(m,n)', {'m': _s, 'n': _s})
    DENSITY_BLOCKS_U[('vv', _s)] = ('e1(e,f)', {'e': _s, 'f': _s})
    DENSITY_BLOCKS_U[('ov', _s)] = ('e1(m,e)', {'m': _s, 'e': _s})

DENSITY_SHAPE_U = {'oo': ('no_{s}', 'no_{s}'), 'vv': ('nv_{s}', 'nv_{s}'), 'ov': ('no_{s}', 'nv_{s}')}
DENSITY_OUT_IDX_U = {'oo': ['m', 'n'], 'vv': ['e', 'f'], 'ov': ['m', 'e']}


def cross_density_terms_u(block, spin, bra_ranks, ket_ranks):
    if not bra_ranks and not ket_ranks:
        raise ValueError("bra_ranks and ket_ranks cannot both be empty")
    insert_op, spin_labels = DENSITY_BLOCKS_U[(block, spin)]
    pq = pq_helper('fermi')
    pq.set_left_operators([[f'l{r}'] for r in bra_ranks] if bra_ranks else [['1']])
    pq.set_right_operators([['1']])
    if ket_ranks:
        for r in ket_ranks:
            pq.add_operator_product(1.0, [insert_op, f't{r}'])
    else:
        pq.add_operator_product(1.0, [insert_op])
    pq.simplify()
    terms = pq.strings(spin_labels=spin_labels)
    pq.clear()
    return terms


def _rank_aliases_and_params_u(ranks, side, order_of):
    aliases, params = {}, []
    for r in ranks:
        order = order_of[r]
        for sub in ALL_BLOCKS_U[r]:
            real_name = f'{side}_{r}_{order}_{sub}'
            aliases[f'{side}{r}_{sub}'] = real_name
            if real_name not in params:
                params.append(real_name)
    return aliases, params


def generate_density_pieces_u(pieces):
    """pieces: list of (name, block, spin, bra_ranks, bra_order_of, ket_ranks, ket_order_of)."""
    funcs = []
    for name, block, spin, bra_ranks, bra_order_of, ket_ranks, ket_order_of in pieces:
        terms = cross_density_terms_u(block, spin, bra_ranks, ket_ranks)
        aliases, params = dict(), list(COMMON_ARGS_U)
        bra_al, bra_params = _rank_aliases_and_params_u(bra_ranks, 'l', bra_order_of)
        ket_al, ket_params = _rank_aliases_and_params_u(ket_ranks, 't', ket_order_of)
        aliases.update(bra_al)
        aliases.update(ket_al)
        params += bra_params + ket_params
        shape = '(' + ', '.join(d.format(s=spin) for d in DENSITY_SHAPE_U[block]) + ')'
        print(f"{name}: {len(terms)} terms")
        funcs.append(assemble_u(name, params, aliases, 'result', shape, terms, DENSITY_OUT_IDX_U[block]))
    return funcs


def _write(path, funcs, header):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(header)
        fh.write("import numpy as np\nfrom src.SingleReference.CC.cached_einsum import einsum\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"wrote {path}")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    open(os.path.join(OUT_DIR, '__init__.py'), 'a').close()

    # --- amplitude numerators actually consumed by the MP2/MP3 density:
    # T2^(1); T1^(2), T2^(2), T3^(2); T1^(3) (rank >= 2 of Psi^(3) never
    # feeds the n<=3 density -- see module docstring).
    amp_funcs = []
    amp_funcs += amplitude_numerator_funcs_u(2, 1)
    for rank in (1, 2, 3):
        amp_funcs += amplitude_numerator_funcs_u(rank, 2)
    amp_funcs += amplitude_numerator_funcs_u(1, 3)
    # Production (laplace-mode) variants of t1_3_{aa,bb}_numerator: exclude
    # rank 3 (T3^(2)) entirely, so these bodies never reference any t3_2_*
    # block at all -- their contribution is instead added by
    # generate_mp3_t3_laplace_unrestricted.py's Laplace-fused pieces (see
    # that module's docstring, and mpn_density_driver_unrestricted.py's
    # compute_t1_3).
    amp_funcs += amplitude_numerator_no_t3_funcs_u(1, 3, max_connect_rank=2)

    # --- overlap: rank 2 only (N^(2) = <Psi1|Psi1>, N^(3) = 2<Psi1|Psi2>,
    # both rank-2-matched).
    overlap_funcs = [generate_overlap_u(2)]

    # --- density pieces, both spin channels. Rank lists: 12/21 exclude rank
    # 4 (cannot couple to <D2| through a one-body operator), 30/03 keep only
    # rank 1 (<Phi0|e|D_r> = 0 for r >= 2) -- see module docstring.
    density_pieces = []
    for block in ('oo', 'vv', 'ov'):
        for s in ('a', 'b'):
            density_pieces += [
                (f'm2_{block}_{s}_11_unrestricted', block, s, [2], {2: 1}, [2], {2: 1}),
                (f'm2_{block}_{s}_20_unrestricted', block, s, [1], {1: 2}, [], {}),
                (f'm2_{block}_{s}_02_unrestricted', block, s, [], {}, [1], {1: 2}),
                (f'm3_{block}_{s}_12_unrestricted', block, s, [2], {2: 1},
                 [1, 2, 3], {r: 2 for r in (1, 2, 3)}),
                (f'm3_{block}_{s}_21_unrestricted', block, s,
                 [1, 2, 3], {r: 2 for r in (1, 2, 3)}, [2], {2: 1}),
                (f'm3_{block}_{s}_30_unrestricted', block, s, [1], {1: 3}, [], {}),
                (f'm3_{block}_{s}_03_unrestricted', block, s, [], {}, [1], {1: 3}),
            ]
            if block == 'ov':
                # Production (laplace-mode) variant of
                # m3_ov_{a,b}_12_unrestricted: ket_ranks capped at 2, so this
                # body never references any t3_2_* block at all -- see
                # amplitude_numerator_no_t3_funcs_u's comment above (and
                # generate_mp3_t3_laplace_unrestricted.py for where the
                # T3^(2) contribution comes from instead). oo/vv's m3_12
                # bodies don't need this: they never reference t_3_2_* in an
                # einsum anyway (dead parameter, verified by inspection), so
                # a zero rank-6 placeholder there is merely unused, never
                # passed to einsum.
                density_pieces.append(
                    (f'm3_ov_{s}_12_unrestricted_no_t3', block, s, [2], {2: 1}, [1, 2], {1: 2, 2: 2}))
    density_funcs = generate_density_pieces_u(density_pieces)

    _write(os.path.join(OUT_DIR, 'mpn_density_pieces_unrestricted.py'),
           amp_funcs + overlap_funcs + density_funcs,
           "# GENERATED CODE -- unrestricted (spin-blocked, separate alpha/beta) MPn\n"
           "# amplitude-numerator, overlap, and density cross-term pieces (MP2+MP3),\n"
           "# order-generic pdaggerq recursion. Produced by\n"
           "# src/CodeGeneration/generate_mpn_density_unrestricted.py; see\n"
           "# that module's docstring and generate_mpn_density.py's for the recursion\n"
           "# these pieces are combined through. Do not edit by hand.\n")

    # --- DF/RI variant (Phase 1 of the DF rewrite): only t2_1 and
    # t1_2 carry a bracket integral factor at n=2. Phase 3 adds the n=3
    # pieces: t2_2 (production, includes the memory-critical vvvv term),
    # t3_2 (non-Laplace oracle path only), t1_3's full and '_no_t3'
    # (production) variants.
    df_funcs = amplitude_numerator_funcs_u_df(2, 1) + amplitude_numerator_funcs_u_df(1, 2)
    for rank in (2, 3):
        df_funcs += amplitude_numerator_funcs_u_df(rank, 2)
    df_funcs += amplitude_numerator_funcs_u_df(1, 3)
    df_funcs += amplitude_numerator_no_t3_funcs_u_df(1, 3, max_connect_rank=2)
    _write(os.path.join(OUT_DIR, 'mpn_density_pieces_unrestricted_df.py'), df_funcs,
           "# GENERATED CODE -- DF/RI variant of t2_1/t1_2/t2_2/t3_2/t1_3 numerators\n"
           "# (Phases 1+3 of the DF rewrite): takes B_aa/B_bb (3-index RI\n"
           "# factors) instead of g_aaaa/g_abab/g_bbbb, never forming a norb^4-scale\n"
           "# integral block. Every OTHER density piece (overlap2, m2_*/m3_* cross-\n"
           "# terms) has no bracket integral factor at all and is reused UNCHANGED from\n"
           "# mpn_density_pieces_unrestricted.py.\n"
           "# Produced by src/CodeGeneration/generate_mpn_density_unrestricted.py.\n"
           "# Do not edit by hand.\n")
