"""Codegen script (run once, output committed): applies the generic
Laplace-fusion mechanism (src/CodeGeneration/laplace_codegen.py) to MP4's
new T4^(2)_aaaaaaaa-dependent equations, eliminating the O(nv^4*no^4)
t4_2_aaaaaaaa tensor from its two direct single-role (ket-only) consumers.

Scope: only t2_3_aaaa_numerator (1 of 18 raw terms references t4_aaaaaaaa)
and t3_3_aaaaaa_numerator (4 of 36) actually reference t4_aaaaaaaa --
confirmed by direct inspection (t2_3_abab/t3_3_aabaab/t3_3_abbabb never
reference it, same "aaaaaa sector only" scope as the existing T3 Laplace
port). m4_{oo,vv,ov}_22_restricted and overlap4_restricted also reference
t4_aaaaaaaa, but in BOTH bra (l4) and ket (t4) roles simultaneously for
their own rank-4-vs-rank-4 "diagonal" term (Psi^(2)'s own rank-4 component
overlapping/inserting against itself) -- fusing a bra role needs the
l-tag's own axis/sign convention (see mpn_density_driver_restricted.py's
_to_l_restricted) generalized into the fuser, and overlap4's self-contraction
is quadratic in the SAME amplitude (needs 1/D4^2, a genuinely different
double-quadrature substitution, not just the linear 1/D case this module's
fuser implements) -- NOT attempted here. t4_2_aaaaaaaa is therefore still
materialized once (for those two consumers only, down from six), which this
module's two fusions let the OTHER five single-role consumers skip.
"""
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.CodeGeneration.generate_mpn_density_restricted import t_numerator_terms_restricted
from src.CodeGeneration.laplace_codegen import fuse_amplitude_into_consumer, assemble_laplace_fused

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'SingleReference', 'DensityMatrix', 'generated_mpn_restricted')

# T4^(2)_aaaaaaaa's OWN build is <bra4|v|T2^(1)>, so its raw numerator terms
# reference 't2' (T2^(1)_aaaa) themselves -- the fused expression inherits
# that dependency (confirmed by direct inspection of the generated code: the
# fused terms reference a bare 't2_aaaa' leaf), so t2_1_aaaa must be an
# explicit parameter, aliased from pdaggerq's generic 't2_aaaa' tag name.
COMMON_PARAMS = ['g_aaaa', 'g_abab', 'g_bbbb', 'd_aa', 'o', 'v', 'nv', 'no', 'eps_a', 't2_1_aaaa']
COMMON_ALIASES = {'t2_aaaa': 't2_1_aaaa'}


def _build(func_name, out_var, out_shape_expr, outer_terms, out_indices, extra_params, aliases):
    inner = t_numerator_terms_restricted(4, 2, 'aaaaaaaa')
    fused = fuse_amplitude_into_consumer(inner, 4, outer_terms, 't4_aaaaaaaa')
    print(f"{func_name}: {len(fused)} fused terms")
    return assemble_laplace_fused(
        func_name, COMMON_PARAMS + extra_params, {**COMMON_ALIASES, **aliases}, out_var, out_shape_expr,
        fused, out_indices, eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]', ntau_default=6)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    outer_t2_3 = t_numerator_terms_restricted(2, 3, 'aaaa')
    t2_3_lap = _build('t2_3_aaaa_t4_laplace', 't2_3_aaaa_t4', '(nv, nv, no, no)',
                      outer_t2_3, ['a', 'b', 'i', 'j'], [], {})

    outer_t3_3 = t_numerator_terms_restricted(3, 3, 'aaaaaa')
    t3_3_lap = _build('t3_3_aaaaaa_t4_laplace', 't3_3_aaaaaa_t4', '(nv, nv, nv, no, no, no)',
                      outer_t3_3, ['a', 'b', 'c', 'i', 'j', 'k'], [], {})

    path = os.path.join(OUT_DIR, 'mp4_laplace_restricted.py')
    with open(path, 'w') as fh:
        fh.write("# GENERATED CODE -- Laplace-fused T4^(2)_aaaaaaaa contribution to\n"
                "# t2_3_aaaa_numerator/t3_3_aaaaaa_numerator (the only two of five\n"
                "# T2^(3)/T3^(3) equations that reference t4_aaaaaaaa; see\n"
                "# generate_mp4_laplace_restricted.py's module docstring for scope).\n"
                "# Do not edit by hand.\n")
        fh.write("import numpy as np\nfrom src.SingleReference.CC.cached_einsum import einsum\n"
                 "from src.Base.utils.grids import minimax_time_grid\n\n\n")
        fh.write("\n\n".join([t2_3_lap, t3_3_lap]))
        fh.write("\n")
    print(f"wrote {path}")
