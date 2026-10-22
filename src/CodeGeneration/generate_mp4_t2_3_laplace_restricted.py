"""Emit the FULLY Laplace-fused t2_3 (order-3 doubles) numerator: one function
per rank>=3 leaf tag (t3_2 and t4_2, every spin block), so t2_3 can be built
with NONE of the O(no^3 nv^3) t3_2 / O(no^4 nv^4) t4_2 tensors materialized.

Generalizes generate_mp4_laplace_restricted.py, which fused ONLY the t4_aaaaaaaa
contribution (the piece compute_delta_gamma4 needed); here EVERY rank>=3 leaf of
t2_3's numerator is fused, via laplace_codegen.fuse_amplitude_rank3_tags (the
split builder: fuse each rank>=3-leaf contribution, leave the mixed-order
rank<=2 DIRECT part to the compiled t2_3_{block}_numerator with t3/t4 zeroed --
see that helper's docstring for why fuse_nested_numerator can't build an
amplitude). Consumed by MPnDensityDriverRestricted.compute_t2_3_laplace:
  t2_3_{block} = (direct_numerator|t3=t4=0  +  sum_tag this-module's fused tag
                  -  E2 * t2_1_{block}) / D_2

Each emitted function integrates ONE minimax tau (its leaf's own denominator);
different tags are summed by the driver (kept as separate functions rather than
one multikind block so each stays a plain single-tau loop). Names:
t2_3_{block}_{tag}_laplace, e.g. t2_3_aaaa_t3_aaaaaa_laplace.

Run once; output committed to
generated_mpn_restricted/mp4_t2_3_laplace_restricted.py.
"""
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.CodeGeneration.generate_mpn_density_restricted import (
    t_numerator_terms_restricted)
from src.CodeGeneration.generate_mpn_density import amp_out_indices
from src.CodeGeneration.laplace_codegen import (
    fuse_amplitude_rank3_tags, assemble_laplace_fused)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'SingleReference',
                       'DensityMatrix', 'generated_mpn_restricted')

# The fused contractions reference the two-body integrals + the order-1 T2^(1)
# leaves (as the pdaggerq-generic t2_{spin} tags, aliased to t2_1_{spin}) --
# NO Fock/d factor survives in any t2_3 tag contribution (verified: no d_aa/d_bb
# in the emitted bodies), unlike gen_lap4's t4-only build.
PARAMS = ['g_aaaa', 'g_abab', 'g_bbbb', 'o', 'v', 'nv', 'no',
          'eps_a', 't2_1_aaaa', 't2_1_abab']
ALIASES = {'t2_aaaa': 't2_1_aaaa', 't2_abab': 't2_1_abab',
           't2_bbbb': 't2_1_aaaa'}
SHAPE = '(nv, nv, no, no)'


def _build_functions():
    funcs = []
    for block in ('aaaa', 'abab'):
        out_idx = amp_out_indices(2)          # ['a', 'b', 'i', 'j']
        tag_terms = fuse_amplitude_rank3_tags(
            t_numerator_terms_restricted, 2, 3, block)
        for tag, (rk, fused) in sorted(tag_terms.items()):
            name = f't2_3_{block}_{tag}_laplace'
            print(f"  {name}: {len(fused)} fused terms")
            funcs.append(assemble_laplace_fused(
                name, PARAMS, ALIASES, 'out', SHAPE, fused, out_idx,
                eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]', ntau_default=6))
    return funcs


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)
    funcs = _build_functions()
    path = os.path.join(OUT_DIR, 'mp4_t2_3_laplace_restricted.py')
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- do not edit by hand.\n"
                 "# Fully Laplace-fused t2_3 (order-3 doubles) numerator: one\n"
                 "# function per rank>=3 leaf tag (t3_2/t4_2, all spins) so t2_3\n"
                 "# is built with no rank>=3 tensor materialized. Produced by\n"
                 "# generate_mp4_t2_3_laplace_restricted.py (see its docstring);\n"
                 "# consumed by MPnDensityDriverRestricted.compute_t2_3_laplace.\n")
        fh.write("import numpy as np\n"
                 "from src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join(funcs))
        fh.write("\n")
    print(f"wrote {path}  ({len(funcs)} functions)")


if __name__ == '__main__':
    generate()
