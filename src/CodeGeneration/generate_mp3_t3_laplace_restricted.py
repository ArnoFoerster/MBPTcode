# pdaggerq-GENERATED codegen script (run once, output committed): applies the
# generic Laplace-fusion mechanism (laplace_codegen.py) to the aabaab/abbabb
# cross-spin sector of T3^(2), eliminating the last two O(nv^3*no^3)
# materialized tensors (t3_2_aabaab, t3_2_abbabb) from the restricted MP3
# density path -- mirrors the already-established t3_2_aaaaaa precedent
# (mpn_density_driver_restricted.py's _laplace_aaaaaa_contribution,
# generate_mp4_laplace_restricted.py), just for the cross-spin blocks and via
# the generic fuser instead of a hand derivation.
#
# Scope: only two production sites reference t3_aabaab/t3_abbabb as a live
# (non-dead) factor -- t1_3_aa_numerator and m3_ov_12_restricted (confirmed by
# direct inspection of generated_mpn_restricted/mpn_density_pieces_restricted.py:
# m3_{oo,vv}_12_restricted declare t_3_2_aabaab/abbabb as parameters but never
# reference them in an einsum, and all of m3_{oo,vv,ov}_21_restricted declare
# l_3_2_aabaab/abbabb likewise dead -- same "declared but unused" pattern
# already documented for t3_2_aaaaaa's l3/m3_21 role).
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.CodeGeneration.generate_mpn_density_restricted import (
    t_numerator_terms_restricted, cross_density_terms_restricted,
)
from src.CodeGeneration.laplace_codegen import fuse_amplitude_into_consumer, assemble_laplace_fused
from src.CodeGeneration.df_codegen import dress_integral_factor

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'SingleReference', 'DensityMatrix', 'generated_mpn_restricted')

# t3_aabaab/t3_abbabb's own numerator terms reference t2_aaaa AND t2_abab
# (both are within _connect_ranks(3, 2) = rank 2 only), so both must be
# explicit params here -- matches compute_order2_amplitudes' own call
# (t3_2_aabaab_numerator(**args, t2_1_aaaa=..., t2_1_abab=...)). t3_abbabb's
# own numerator additionally references the all-beta t2_bbbb leaf, which this
# repo's closed-shell convention never stores separately -- aliased to its
# t2_aaaa twin (BB_TWIN in generate_mpn_density_restricted.py), same as every
# other piece in that module.
COMMON_PARAMS = ['g_aaaa', 'g_abab', 'g_bbbb', 'd_aa', 'o', 'v', 'nv', 'no', 'eps_a', 't2_1_aaaa', 't2_1_abab']
COMMON_ALIASES = {'t2_aaaa': 't2_1_aaaa', 't2_abab': 't2_1_abab', 't2_bbbb': 't2_1_aaaa'}


def _build(func_name, out_var, out_shape_expr, outer_terms, out_indices, extra_params, aliases):
    inner_aabaab = t_numerator_terms_restricted(3, 2, 'aabaab')
    inner_abbabb = t_numerator_terms_restricted(3, 2, 'abbabb')
    fused = (fuse_amplitude_into_consumer(inner_aabaab, 3, outer_terms, 't3_aabaab')
             + fuse_amplitude_into_consumer(inner_abbabb, 3, outer_terms, 't3_abbabb'))
    print(f"{func_name}: {len(fused)} fused terms")
    return assemble_laplace_fused(
        func_name, COMMON_PARAMS + extra_params, {**COMMON_ALIASES, **aliases}, out_var, out_shape_expr,
        fused, out_indices, eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]', ntau_default=6)


COMMON_PARAMS_DF = ['B_aa', 'B_bb', 'd_aa', 'o', 'v', 'nv', 'no', 'eps_a', 't2_1_aaaa', 't2_1_abab']


def _build_df(func_name, out_var, out_shape_expr, outer_terms, out_indices, extra_params, aliases):
    """DF/RI variant of _build (Phase 3 of the DF rewrite): applies
    dress_integral_factor to BOTH the outer (consumer) raw terms and the
    inner (T3^(2) numerator) raw terms BEFORE fusion, then runs the exact
    same fuse_amplitude_into_consumer + assemble_laplace_fused pipeline
    unmodified. This composes cleanly because dress_integral_factor's output
    (B_xx(Q,p,r)-style factors) is just another plain-paren factor as far as
    laplace_codegen.py's own factor parser is concerned (no bracket syntax
    survives dressing), and its fresh auxiliary letters get renamed like any
    other dummy index during fusion -- verified empirically (H2O/6-31g,
    naux=norb^2 exact decomposition) to reproduce the dense-generated
    function to ~1e-17 absolute before trusting this composition for real.
    """
    outer_df = dress_integral_factor(outer_terms)
    inner_aabaab = dress_integral_factor(t_numerator_terms_restricted(3, 2, 'aabaab'))
    inner_abbabb = dress_integral_factor(t_numerator_terms_restricted(3, 2, 'abbabb'))
    fused = (fuse_amplitude_into_consumer(inner_aabaab, 3, outer_df, 't3_aabaab')
             + fuse_amplitude_into_consumer(inner_abbabb, 3, outer_df, 't3_abbabb'))
    df_name = f"{func_name}_df"
    print(f"{df_name}: {len(fused)} fused terms")
    return assemble_laplace_fused(
        df_name, COMMON_PARAMS_DF + extra_params, {**COMMON_ALIASES, **aliases}, out_var, out_shape_expr,
        fused, out_indices, eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]', ntau_default=6)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    # t1_3_aa_numerator's own raw terms -- aabaab/abbabb-referencing ones get
    # fused; the rest (t1_2_aa/t2_2_*/t3_2_aaaaaa-referencing terms) are
    # simply dropped by fuse_amplitude_into_consumer (unaffected, still built
    # by the ordinary generated t1_3_aa_numerator call).
    outer_t1_3 = t_numerator_terms_restricted(1, 3, 'aa')
    t1_3_lap = _build('t1_3_aa_t3crossspin_laplace', 't1_3_aa_t3cs', '(nv, no)',
                      outer_t1_3, ['a', 'i'], [], {})

    # m3_ov_12_restricted's own raw terms (bra rank 2 order 1, ket ranks
    # 1,2,3 order 2 -- same wiring as generate_mpn_density_restricted.py's
    # __main__ for this piece). Surviving terms after fusion also carry a
    # bare l2_aaaa/l2_abab bra factor (the <D2|v T3> structure), so those
    # need their own param/alias pair.
    outer_ov = cross_density_terms_restricted('ov', [2], [1, 2, 3])
    ov_lap = _build('m3_ov_a_t3crossspin_laplace', 'm3_ov_t3cs', '(no, nv)',
                    outer_ov, ['m', 'e'],
                    ['l_2_1_aaaa', 'l_2_1_abab'],
                    {'l2_aaaa': 'l_2_1_aaaa', 'l2_abab': 'l_2_1_abab', 'l2_bbbb': 'l_2_1_aaaa'})

    path = os.path.join(OUT_DIR, 'mp3_t3_laplace_restricted.py')
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- Laplace-fused T3^(2)_aabaab/T3^(2)_abbabb\n"
                "# cross-spin contribution to t1_3_aa_numerator/m3_ov_12_restricted\n"
                "# (the only two live consumers; see\n"
                "# generate_mp3_t3_laplace_restricted.py's module docstring for scope).\n"
                "# Do not edit by hand.\n")
        fh.write("import numpy as np\nfrom src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join([t1_3_lap, ov_lap]))
        fh.write("\n")
    print(f"wrote {path}")

    # --- DF/RI variant (Phase 3 of the DF rewrite): same two
    # equations, DF-dressed before fusion (see _build_df's docstring).
    t1_3_lap_df = _build_df('t1_3_aa_t3crossspin_laplace', 't1_3_aa_t3cs', '(nv, no)',
                             outer_t1_3, ['a', 'i'], [], {})
    ov_lap_df = _build_df('m3_ov_a_t3crossspin_laplace', 'm3_ov_t3cs', '(no, nv)',
                          outer_ov, ['m', 'e'],
                          ['l_2_1_aaaa', 'l_2_1_abab'],
                          {'l2_aaaa': 'l_2_1_aaaa', 'l2_abab': 'l_2_1_abab', 'l2_bbbb': 'l_2_1_aaaa'})

    path_df = os.path.join(OUT_DIR, 'mp3_t3_laplace_restricted_df.py')
    with open(path_df, 'w') as fh:
        fh.write("# GENERATED CODE -- DF/RI variant of mp3_t3_laplace_restricted.py (Phase 3\n"
                "# of the DF rewrite): takes B_aa/B_bb instead of\n"
                "# g_aaaa/g_abab/g_bbbb, never forming a norb^4-scale integral block.\n"
                "# Do not edit by hand.\n")
        fh.write("import numpy as np\nfrom src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join([t1_3_lap_df, ov_lap_df]))
        fh.write("\n")
    print(f"wrote {path_df}")
