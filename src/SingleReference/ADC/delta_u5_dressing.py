"""Opt-in order-5 diagrammatic defect dressing: loader + status.

SUPERSEDED as a production track. The order-5 content below is the
decoupled image of two union-space cross blocks, X^(3) of the 2h1p x 2p1h
pair and X^(2) of the 2h1p x 3p2h pair (and the transpose pair); the
production route at fifth order is construction='union' with those blocks
emitted (generate_adcn.generate_adc_cross_pair(3, 'k1k1') and
(2, 'k1k2')) or their decoupling (cross_block_decoupling.decouple_cross_
blocks), not a diagram-side evaluator. The loader and its gated row
artifacts are kept as validation identities for that image.

THE CLOSED FORM (order-5 structure theorem, two H4 geometries): the genuinely-order-5 ISR defect is pure static rows,

    k1 coupling += dU3 + dU4,   orbital<->k2 coupling += dU3k2,

with the rows given oracle-free by per-pole division of

    D(kappa) = [full-stream diagram Laurent residues]
             - [ISR resolvent-chain residues (ladder blocks)]
             - [cross(dU3)]

onto the on-pole coupling families (U1 at k1; V = U2k2 + C135' R1 U1
at k2).  Gated: the closed rows reproduce the exact lambda^5 defect
field out-of-sample at 6.4e-12 / 1.1e-11 (r = 1.0 / 1.6,
delta5_closed_rows.py) and the dressed ladder reproduces the stage-3
diagrammatic-ADC(5) IPs at [0.15, 1.72] meV (the dense prototype).

CONSUMER (production, wired): sigma_driver.restricted_dyson_operator
(delta_u5_rows=...) threads the row dict through the matrix-free
two-sector operator.  Strictly opt-in; every pinned route untouched.

BASIS BRIDGE (this module): adc5_ladder's class space is literally
BuildSuperMatrixADCnRestricted's own configs concatenated as
[k1_IP | k1_EA | k2_IP | k2_EA], so the ladder-basis row artifacts
(delta5_ladder_rows_*.npz, produced by delta5_closed_wirein.py) map to
the production hook by slicing -- load_rows() below.  End-to-end gate:
an external gate script (not in this repository) (the hook-dressed
restricted_dyson_operator('adc5') reproduces the dressed-ladder
eigenvalues).

PRODUCTION EVALUATOR STATUS (why delta_u5_rows() still raises): the delta5 row content splits into FIVE nonzero
position-pattern groups (X(0,3); X(01)+X(23); X(02)+X(13);
X(013)+X(023); the inner (1,2) singles -- X(012)+X(123)==0 is a proven
zero and needs nothing).  **X(0,3) is now CLOSED at rank<=4 production
scale, both H4 geometries, machine precision**
(`delta5_x03_evaluator.x03_rows_full`; worst gate against the dense
prototype 3.7e-17/1.9e-16) -- the mechanism: group configs by exact K1 value
once, then for each row loop over only its own (small,
system-size-independent) degenerate class as the energy-match
candidate set, feeding the existing rank<=4 D1/D2 Laplace machinery
per candidate.  The other four groups remain VALIDATION-SCALE ONLY (or
entirely un-derived) -- the full-stream Laurent aggregation for those
still materializes dense index grids above the rank<=4 floor, AND no
closed form is known yet (the mechanism that closed X(0,3) relies on a
symmetric two-independent-boundary topology the other patterns don't
have, which is why it doesn't transfer mechanically).

**IMPORTANT: X(0,3) is NOT a usable partial/approximate answer on its
own.** Measured both geometries: |X(0,3)| is 20-45x LARGER in norm
than the final physical delta5 field |d5n+cross| -- it is large,
indefinite-scale content that must cancel against the other four
position-pattern groups to reach the small physical answer (the same
"multi-omega vs fold, 600-2300x cancellation" phenomenon the row
ledger already documented at the whole-field level, now localized to
one position pattern).  Wiring X(0,3)'s rows in alone would inject a
wrongly-scaled correction, not a smaller-but-useful approximation.
`delta_u5_rows()` therefore correctly keeps raising until every
nonzero position-pattern group is closed -- there is no honest
partial-credit path here.
"""
import numpy as np


def load_rows(npz_path):
    """delta5_ladder_rows_*.npz -> the delta_u5_rows dict for
    sigma_driver.restricted_dyson_operator.

    The npz carries (du3, dU4, dU3k2) over the ladder class space
    [k1_IP | k1_EA | k2_IP | k2_EA] (== the restricted solver's own
    config orderings, adc5_ladder.assemble construction) plus the four
    segment sizes.  k1 entries get dU3 + dU4 (both dress the k1
    coupling); k2 entries get dU3k2."""
    z = np.load(npz_path)
    n1i, n1e, n2i, n2e = (int(x) for x in z['sizes'])
    du3, dU4, dU3k2 = z['du3'], z['dU4'], z['dU3k2']
    k1 = du3 + dU4
    o1, o2, o3 = n1i, n1i + n1e, n1i + n1e + n2i
    return {('IP', 1): k1[:o1],
            ('EA', 1): k1[o1:o2],
            ('IP', 2): dU3k2[o2:o3],
            ('EA', 2): dU3k2[o3:]}


def delta_u5_rows(*_args, **_kw):
    raise NotImplementedError(
        "order-5 defect rows: production (rank<=4) evaluator pending -- "
        "the multi-omega on-shell denominators are sign-indefinite until "
        "the per-X(P) closed-form reduction is derived (module docstring). "
        "Validation-scale gated rows: load_rows('.../delta5_ladder_rows_"
        "*.npz').")
