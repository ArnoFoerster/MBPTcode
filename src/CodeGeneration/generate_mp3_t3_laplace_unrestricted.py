# pdaggerq-GENERATED codegen script (run once, output committed): UHF analogue
# of generate_mp3_t3_laplace_restricted.py -- applies the generic Laplace-
# fusion mechanism (laplace_codegen.py) to ALL FOUR T3^(2) spin blocks
# (aaaaaa, aabaab, abbabb, bbbbbb), eliminating every O(nv^3*no^3) t3_2_*
# tensor from the two live consumers (t1_3_{aa,bb}_numerator,
# m3_ov_{a,b}_12_unrestricted) -- the UHF driver previously had NO Laplace
# acceleration at all (module docstring of mpn_density_driver_unrestricted.py),
# unlike the restricted driver which already had it for the aaaaaa sector.
#
# Genuinely new relative to the restricted module: a T3^(2) block's own
# external indices can carry MIXED alpha/beta spin (e.g. aabaab: 2 alpha
# pairs + 1 beta pair), so a single shared Oe/Ve dressing array (valid in the
# restricted case only because RHF's alpha and beta spatial orbitals share
# one eps array) no longer works -- each dressed index needs the array from
# ITS OWN spin. laplace_codegen.py's generic fuser has no spin concept (it
# only tags indices "occ" or "vir", via OCC_LETTERS/VIR_LETTERS), so this
# module post-processes its raw output: fuse_amplitude_into_consumer appends
# exactly one Oe(letter)/Ve(letter) dressing factor per amplitude external
# index, in canonical order (amp_out_indices(rank): vir letters then occ
# letters, one per pair), as the LAST 2*rank factors of every returned term
# (see its docstring) -- so _relabel_dressings renames each dressing
# positionally to Oe_a/Oe_b/Ve_a/Ve_b using the amplitude BLOCK's own
# per-pair spin pattern (block[:rank], the same pair-spin string
# T_BRA_U/ALL_BLOCKS_U already key every block name on).
#
# The two output equations' own params (eps_a/eps_b) are then also used to
# build ONE shared minimax quadrature grid wide enough for either spin's own
# occ/vir gap (gap_min/gap_max taken over the UNION of alpha+beta orbital
# energies) -- the quadrature identity 1/D = -sum_tau sigma_tau*exp(D*tau)
# holds for any real D as long as the grid's covered range contains D's
# actual value, so a conservative shared grid is safe for a mixed-spin D3
# too, just as it already is for a single-spin one.
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.CodeGeneration.generate_mpn_density_unrestricted import (
    t_numerator_terms_u, cross_density_terms_u, _spinify_slices, ALL_BLOCKS_U,
)
from src.CodeGeneration.laplace_codegen import fuse_amplitude_into_consumer, raw_terms_to_einsum_dressed
from src.CodeGeneration.df_codegen import dress_integral_factor

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'SingleReference', 'DensityMatrix', 'generated_mpn_unrestricted')

T2_BLOCKS_U = ALL_BLOCKS_U[2]   # ('aaaa', 'abab', 'bbbb')
T3_BLOCKS_U = ALL_BLOCKS_U[3]   # ('aaaaaa', 'aabaab', 'abbabb', 'bbbbbb')

COMMON_PARAMS = (['g_aaaa', 'g_abab', 'g_bbbb', 'd_aa', 'd_bb',
                  'o_a', 'v_a', 'o_b', 'v_b', 'nv_a', 'no_a', 'nv_b', 'no_b', 'eps_a', 'eps_b']
                 + [f't2_1_{b}' for b in T2_BLOCKS_U])
COMMON_ALIASES = {f't2_{b}': f't2_1_{b}' for b in T2_BLOCKS_U}

COMMON_PARAMS_DF = (['B_aa', 'B_bb', 'd_aa', 'd_bb',
                     'o_a', 'v_a', 'o_b', 'v_b', 'nv_a', 'no_a', 'nv_b', 'no_b', 'eps_a', 'eps_b']
                    + [f't2_1_{b}' for b in T2_BLOCKS_U])


def _block_spin_of_canonical(block, rank):
    """block[:rank] is the pair-spin pattern shared by both the vir and occ
    half of the block name (T_BRA_U's own construction: pair p's occ and vir
    letter always get the same spin) -- amp_out_indices(rank)'s canonical
    order is vir letters (pair 0..rank-1) then occ letters (pair 0..rank-1),
    so the per-canonical-position spin list is simply the pattern twice."""
    pattern = block[:rank]
    return list(pattern) + list(pattern)


def _relabel_dressings(fused_terms, inner_rank, block):
    """See module docstring: rename the trailing 2*inner_rank Oe(x)/Ve(x)
    dressing factors of each fused term to Oe_a/Oe_b/Ve_a/Ve_b positionally,
    using `block`'s own per-canonical-position spin."""
    spins = _block_spin_of_canonical(block, inner_rank)
    n_dress = 2 * inner_rank
    out = []
    for term in fused_terms:
        coef, factors = term[0], term[1:]
        head, dress = factors[:-n_dress], factors[-n_dress:]
        new_dress = []
        for factor, spin in zip(dress, spins):
            name, rest = factor.split('(', 1)
            new_dress.append(f"{name}_{spin}({rest}")
        out.append([coef] + list(head) + new_dress)
    return out


def _fuse_all_t3_blocks(outer_terms):
    """Fuse every T3^(2) spin block's own numerator into `outer_terms`;
    blocks that don't actually appear as a factor in any outer term
    contribute an empty list (fuse_amplitude_into_consumer's own no-op case),
    so it's safe/simplest to always try all four rather than pre-filtering
    which blocks are "live" for a given consumer."""
    fused = []
    for block in T3_BLOCKS_U:
        inner = t_numerator_terms_u(3, 2, block)
        one = fuse_amplitude_into_consumer(inner, 3, outer_terms, f't3_{block}')
        fused += _relabel_dressings(one, 3, block)
    return fused


def _fuse_all_t3_blocks_df(outer_terms):
    """DF/RI variant of _fuse_all_t3_blocks (Phase 3 of
    the DF rewrite): dress_integral_factor applied to both the
    outer terms and each inner T3^(2) block's own numerator BEFORE fusion --
    same composition order validated for the restricted module (see
    generate_mp3_t3_laplace_restricted.py::_build_df's docstring)."""
    outer_df = dress_integral_factor(outer_terms)
    fused = []
    for block in T3_BLOCKS_U:
        inner_df = dress_integral_factor(t_numerator_terms_u(3, 2, block))
        one = fuse_amplitude_into_consumer(inner_df, 3, outer_df, f't3_{block}')
        fused += _relabel_dressings(one, 3, block)
    return fused


def assemble_laplace_fused_u(func_name, params, aliases, out_var, out_shape_expr,
                             fused_terms, out_indices, ntau_default=6):
    """UHF analogue of laplace_codegen.assemble_laplace_fused -- per-spin
    Oe_a/Oe_b/Ve_a/Ve_b dressing arrays instead of one shared Oe/Ve, one
    shared tau/sigma quadrature grid wide enough for either spin (see module
    docstring), and _spinify_slices applied to every emitted line (the
    bracket-integral g_xxxx[o, o, v, v]-type factors raw_terms_to_einsum
    emits are spin-blind by construction, same reason generate_mpn_density_
    unrestricted.py's own assemble_u needs this pass)."""
    src = [f"def {func_name}({', '.join(params)}, ntau={ntau_default}):"]
    for generic, real in aliases.items():
        if generic != real:
            src.append(f"    {generic} = {real}")
    src.append("    ei_a, ea_a = eps_a[o_a], eps_a[v_a]")
    src.append("    ei_b, ea_b = eps_b[o_b], eps_b[v_b]")
    src.append("    gap_min = max(min(ea_a.min(), ea_b.min()) - max(ei_a.max(), ei_b.max()), 1e-3)")
    src.append("    gap_max = max(ea_a.max(), ea_b.max()) - min(ei_a.min(), ei_b.min())")
    src.append("    tau, sigma = minimax_time_grid(ntau, 3.0 * gap_min, 3.0 * gap_max)")
    src.append(f"    {out_var} = np.zeros({out_shape_expr})")
    # flatten-p: single amplitude -> one tau slot -> batch axis of length ntau;
    # per-spin Oe_a/Oe_b/Ve_a/Ve_b become (ntau, dim) matrices over that grid.
    src.append("    _tk = np.indices((ntau,)).reshape(1, -1)")
    src.append("    Oe_a = np.exp(ei_a[None, :] * tau[_tk[0]][:, None])")
    src.append("    Oe_b = np.exp(ei_b[None, :] * tau[_tk[0]][:, None])")
    src.append("    Ve_a = np.exp(-ea_a[None, :] * tau[_tk[0]][:, None])")
    src.append("    Ve_b = np.exp(-ea_b[None, :] * tau[_tk[0]][:, None])")
    src.append("    _wP = -sigma[_tk[0]]")
    for line in raw_terms_to_einsum_dressed(fused_terms, out_var, out_indices,
                                            weight_var='_wP'):
        src.append(_spinify_slices(line))
    src.append(f"    return {out_var}")
    src.append("")
    return "\n".join(src)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    funcs = []

    outer_t1_3_aa = t_numerator_terms_u(1, 3, 'aa')
    fused = _fuse_all_t3_blocks(outer_t1_3_aa)
    print(f"t1_3_aa_t3_laplace: {len(fused)} fused terms")
    funcs.append(assemble_laplace_fused_u(
        't1_3_aa_t3_laplace', COMMON_PARAMS, COMMON_ALIASES, 't1_3_aa_t3', '(nv_a, no_a)',
        fused, ['a', 'i']))

    outer_t1_3_bb = t_numerator_terms_u(1, 3, 'bb')
    fused = _fuse_all_t3_blocks(outer_t1_3_bb)
    print(f"t1_3_bb_t3_laplace: {len(fused)} fused terms")
    funcs.append(assemble_laplace_fused_u(
        't1_3_bb_t3_laplace', COMMON_PARAMS, COMMON_ALIASES, 't1_3_bb_t3', '(nv_b, no_b)',
        fused, ['a', 'i']))

    l2_params = [f'l_2_1_{b}' for b in T2_BLOCKS_U]
    l2_aliases = {f'l2_{b}': f'l_2_1_{b}' for b in T2_BLOCKS_U}

    outer_ov_a = cross_density_terms_u('ov', 'a', [2], [1, 2, 3])
    fused = _fuse_all_t3_blocks(outer_ov_a)
    print(f"m3_ov_a_t3_laplace: {len(fused)} fused terms")
    funcs.append(assemble_laplace_fused_u(
        'm3_ov_a_t3_laplace', COMMON_PARAMS + l2_params, {**COMMON_ALIASES, **l2_aliases},
        'm3_ov_a_t3', '(no_a, nv_a)', fused, ['m', 'e']))

    outer_ov_b = cross_density_terms_u('ov', 'b', [2], [1, 2, 3])
    fused = _fuse_all_t3_blocks(outer_ov_b)
    print(f"m3_ov_b_t3_laplace: {len(fused)} fused terms")
    funcs.append(assemble_laplace_fused_u(
        'm3_ov_b_t3_laplace', COMMON_PARAMS + l2_params, {**COMMON_ALIASES, **l2_aliases},
        'm3_ov_b_t3', '(no_b, nv_b)', fused, ['m', 'e']))

    path = os.path.join(OUT_DIR, 'mp3_t3_laplace_unrestricted.py')
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- Laplace-fused T3^(2) (all four spin blocks)\n"
                "# contribution to t1_3_{aa,bb}_numerator/m3_ov_{a,b}_12_unrestricted\n"
                "# (the only live consumers; see\n"
                "# generate_mp3_t3_laplace_unrestricted.py's module docstring for scope).\n"
                "# Do not edit by hand.\n")
        fh.write("import numpy as np\nfrom src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"wrote {path}")

    # --- DF/RI variant (Phase 3 of the DF rewrite): same four
    # equations, DF-dressed before fusion (see _fuse_all_t3_blocks_df's
    # docstring). assemble_laplace_fused_u itself needs no change -- its
    # _spinify_slices pass already handles B_xx[:, ...]'s aux-prefixed slice
    # form (added in Phase 1/2 for the plain generated-numerator pipeline).
    df_funcs = []

    fused = _fuse_all_t3_blocks_df(outer_t1_3_aa)
    print(f"t1_3_aa_t3_laplace_df: {len(fused)} fused terms")
    df_funcs.append(assemble_laplace_fused_u(
        't1_3_aa_t3_laplace_df', COMMON_PARAMS_DF, COMMON_ALIASES, 't1_3_aa_t3', '(nv_a, no_a)',
        fused, ['a', 'i']))

    fused = _fuse_all_t3_blocks_df(outer_t1_3_bb)
    print(f"t1_3_bb_t3_laplace_df: {len(fused)} fused terms")
    df_funcs.append(assemble_laplace_fused_u(
        't1_3_bb_t3_laplace_df', COMMON_PARAMS_DF, COMMON_ALIASES, 't1_3_bb_t3', '(nv_b, no_b)',
        fused, ['a', 'i']))

    fused = _fuse_all_t3_blocks_df(outer_ov_a)
    print(f"m3_ov_a_t3_laplace_df: {len(fused)} fused terms")
    df_funcs.append(assemble_laplace_fused_u(
        'm3_ov_a_t3_laplace_df', COMMON_PARAMS_DF + l2_params, {**COMMON_ALIASES, **l2_aliases},
        'm3_ov_a_t3', '(no_a, nv_a)', fused, ['m', 'e']))

    fused = _fuse_all_t3_blocks_df(outer_ov_b)
    print(f"m3_ov_b_t3_laplace_df: {len(fused)} fused terms")
    df_funcs.append(assemble_laplace_fused_u(
        'm3_ov_b_t3_laplace_df', COMMON_PARAMS_DF + l2_params, {**COMMON_ALIASES, **l2_aliases},
        'm3_ov_b_t3', '(no_b, nv_b)', fused, ['m', 'e']))

    path_df = os.path.join(OUT_DIR, 'mp3_t3_laplace_unrestricted_df.py')
    with open(path_df, 'w') as fh:
        fh.write("# GENERATED CODE -- DF/RI variant of mp3_t3_laplace_unrestricted.py (Phase 3\n"
                "# of the DF rewrite): takes B_aa/B_bb instead of\n"
                "# g_aaaa/g_abab/g_bbbb, never forming a norb^4-scale integral block.\n"
                "# Do not edit by hand.\n")
        fh.write("import numpy as np\nfrom src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join(df_funcs))
        fh.write("\n")
    print(f"wrote {path_df}")
