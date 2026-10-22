"""Emit the fully Laplace-fused t1_4 (order-4 singles) numerator -- the
NESTED-amplitude case: t1_4 consumes t3_3 (order 3), whose OWN numerator
consumes t3_2/t4_2 (order 2), so building t1_4 without materializing any rank>=3
tensor needs a fusion nested TWO deep (t1_4 -> t3_3 -> {t3_2, t4_2}).

Unlike t2_3 (whose numerator carries raw P() tokens, forcing the split builder
fuse_amplitude_rank3_tags), t1_4's numerator is P-free, so laplace_codegen.
fuse_nested_numerator builds the whole thing directly -- direct terms (V.t2_3 /
V.t1_3, kind ()) and the nested t3_3 contributions together. The DEPTH-QUALIFIED
kind names (t3 for the outer t3_3, t3_d1/t4_d1 for its inner t3_2/t4_2) give each
amplitude instance its own tau; without them the two rank-3 amplitudes would
share one tau (a ~7% constant error, see fuse_nested_numerator's docstring).

Leaves consumed (all rank<=2, all Laplace-available): t1_3/t2_3 (order 3, the
DIRECT t1_4 leaves -- fed the driver's own Laplace T1^(3)/T2^(3)), t2_2/t1_2
(order 2, from t3_3's direct part), t2_1 (order 1, from t3_2/t4_2). Consumed by
MPnDensityDriverRestricted.compute_t1_4_laplace:
    t1_4 = (this-module's fused numerator  -  E2 * t1_2) / D_1.

Run once; output committed to
generated_mpn_restricted/mp4_t1_4_laplace_restricted.py.
"""
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.CodeGeneration.generate_mpn_density_restricted import (
    t_numerator_terms_restricted, T_BRA)
from src.CodeGeneration.generate_mpn_density import amp_out_indices
from src.CodeGeneration.laplace_codegen import (
    fuse_nested_numerator, assemble_laplace_fused_multikind)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'SingleReference',
                       'DensityMatrix', 'generated_mpn_restricted')

PARAMS = ['g_aaaa', 'g_abab', 'g_bbbb', 'o', 'v', 'nv', 'no', 'eps_a',
          't1_3_aa', 't2_3_aaaa', 't2_3_abab', 't2_2_aaaa', 't2_2_abab',
          't1_2_aa', 't2_1_aaaa', 't2_1_abab']
ALIASES = {'t1_3_bb': 't1_3_aa', 't2_3_bbbb': 't2_3_aaaa',
           't2_2_bbbb': 't2_2_aaaa', 't1_2_bb': 't1_2_aa',
           't2_1_bbbb': 't2_1_aaaa'}


def _num_fn(rank, order, block):
    keys = T_BRA[rank][1]
    b = block if block in keys else block.translate(str.maketrans('ab', 'ba'))
    return t_numerator_terms_restricted(rank, order, b)


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)
    fused = fuse_nested_numerator(_num_fn, 1, 4, 'aa')
    print(f"  t1_4_aa_laplace: {len(fused)} fused terms")
    src = assemble_laplace_fused_multikind(
        't1_4_aa_laplace', PARAMS, ALIASES, 'out', '(nv, no)', fused,
        amp_out_indices(1), eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]',
        ntau_default=6)
    path = os.path.join(OUT_DIR, 'mp4_t1_4_laplace_restricted.py')
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- do not edit by hand.\n"
                 "# Fully Laplace-fused t1_4 (order-4 singles) numerator, nested\n"
                 "# t1_4 -> t3_3 -> {t3_2,t4_2} (depth-qualified taus). No rank>=3\n"
                 "# tensor materialized. Produced by generate_mp4_t1_4_laplace_\n"
                 "# restricted.py; consumed by compute_t1_4_laplace.\n")
        fh.write("import numpy as np\n"
                 "from src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write(src)
        fh.write("\n")
    print(f"wrote {path}")


if __name__ == '__main__':
    generate()
