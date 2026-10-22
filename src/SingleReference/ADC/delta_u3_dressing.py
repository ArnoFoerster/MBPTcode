"""Opt-in delta-U3 dressing: the closed-form order-4 defect rows.

SUPERSEDED as the production route by the union-space cross block. The rows
below are the order-3 image of the 2h1p x 2p1h block X^(2) = (D|H^|C) under
the Tarantelli-Cederbaum decoupling, delta u_C = sum_D U^(1)_D X^(2)_DC /
(K_C - K_D) (cross_block_decoupling.delta_u_rows_k1 applied to the generated
X^(2), gated equal to these rows to 1e-12 on linear H4 and LiH in
tests/test_adc4_union_poles.py). construction='union' carries the block
itself (exact through lambda^4, no 1/E_6f anywhere), construction=
'diagrammatic' adds its decoupled image; this module is kept as the
diagram-side validation identity of that image.

THE ORDER-4 DEFECT THEOREM (diagram-ADC track, machine-gated at two H4
geometries): at every k1 pole the ISR-ADC(4) defect
delta4 = diagram-M4 - ISR-M4 is the residue of the [3w|6f|4f] Goldstone
orderings -- an uncontracted 3p3h intermediate between the two external
vertices, content with no image in the {2h1p/2p1h, 3h2p/3p2h} ADC(4)
spaces.  Those orderings carry exactly one omega-cut, so the correction
is a pure STATIC dressing of the k1 coupling rows,

    U_k1 -> U^(1) + U^(2) + U^(3) + du3,

evaluated by the generated module delta_u3_blocks.py (emit.py's
emit_delta_u3_order4; gated in tests/test_diagram_delta_u3.py and
an external gate script (not in this repository): the per-pole residue
fold reproduces the measured defect at each geometry's extraction floor).

This module converts the emitted rows from their determinant basis
(diagrammatic SectorSpaces convention: det phases folded in) to the
generated solver's precursor index-tuple basis (class_configs order, no
det phases).  The bridge is exact: U1_tuple[c] = sgn_det(c) *
U1_det[c] on every alive row of both sectors (verified numerically on
H4; dead U1 rows carry identically zero du3, so no phase ambiguity
survives), plus the 2h1p nesting permutation -- SectorSpaces enumerates
(particle outer; hole pair inner), class_configs (hole pair outer;
particle inner).

Evaluator: the PRODUCTION path is delta_u3_laplace_blocks.py -- every
intermediate rank <= 4 (E_4f absorbed into a stored t2-like amplitude,
1/E_6f by self-gated minimax Laplace; n_tau * O(N^5) one-time static).
mode='dense' selects the rank-8-grid validation twin delta_u3_blocks.py
(small systems only; the two agree at 4.4e-13 on H4 at the default
ntau, the emission gate).  Remaining production
transformations (restricted spin-blocked emission, DF factors) follow
the standard generated-code routes.
"""
import numpy as np
from src.SingleReference.ADC.generated_modules import lazy

_du3_blocks = lazy('diagrammatic.delta_u3_blocks')
_du3_laplace_blocks = lazy('diagrammatic.delta_u3_laplace_blocks')


def delta_u3_rows(eps, g, nocc, mode='laplace', ntau=14):
    """(du_2h1p, du_2p1h) delta-U3 rows in the generated precursor basis.

    Shapes (n_2h1p, norb) / (n_2p1h, norb), class_configs enumeration
    (2h1p: hole pairs i<j outer, particle inner; 2p1h: hole outer,
    particle pairs a<b inner) -- add to the k1 coupling blocks alongside
    U^(1)/U^(2)/U^(3).  mode: 'laplace' (production, rank<=4) or
    'dense' (validation twin)."""
    norb = len(eps)
    nvir = norb - nocc
    if mode == 'laplace':
        du_p, du_m = _du3_laplace_blocks.assemble_du3_laplace(
            eps, np.asarray(g), nocc, ntau=ntau)
    elif mode == 'dense':
        du_p, du_m = _du3_blocks.assemble_du3(eps, np.asarray(g), nocc)
    else:
        raise ValueError(f"mode={mode!r}; expected 'laplace' or 'dense'")

    # 2p1h (EA): identical enumeration; tuple-basis row = det phase * row
    sg = np.repeat([(-1.0) ** j for j in range(nocc)],
                   nvir * (nvir - 1) // 2)
    du_2p1h = sg[:, None] * du_p

    # 2h1p (IP): re-nest (particle outer -> hole pair outer) + det phase
    npair = nocc * (nocc - 1) // 2
    du_2h1p = np.empty_like(du_m)
    pidx = 0
    for k in range(nocc):
        for l in range(k + 1, nocc):
            ph = (-1.0) ** (k + l + nocc)
            for a in range(nvir):
                du_2h1p[pidx * nvir + a] = ph * du_m[a * npair + pidx]
            pidx += 1
    return du_2h1p, du_2p1h
