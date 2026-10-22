"""Density-fitting (DF/RI) term-rewrite pass: replaces every bare bracket
integral factor <p,q||r,s>(_ssss) in a pdaggerq raw-term list with its RI
expansion in terms of a 3-index auxiliary factor B_Q,pq, so the resulting
terms feed unmodified into src/Base/pdaggerq_interface.py::raw_terms_to_einsum
-- exactly the "no new codegen logic downstream" pattern
laplace_codegen.py::fuse_amplitude_into_consumer uses for the amplitude side
(Phase 0 of the DF rewrite).

Physics: <pq||rs> = <pq|rs> - <pq|sr>, and each Coulomb-type piece factors
over the RI auxiliary index Q as a simple product of two 2-index B slices:
    <pq|rs> = sum_Q B_(spin p),Q,p,r * B_(spin q),Q,q,s
so
    <pq||rs> = sum_Q B_p[Q,p,r]*B_q[Q,q,s] - sum_Q B_p[Q,p,s]*B_q[Q,q,r]
(same-spin block, both terms present) or, for the restricted spin-blocked
pipeline's cross-spin block (suffix '_abab', p/r alpha and q/s beta):
    <p_a q_b|r_a s_b> = sum_Q B_aa[Q,p,r]*B_bb[Q,q,s]      (exchange vanishes:
                                                              needs spin(p)==spin(s),
                                                              false here)
The general rule this module implements: the exchange term is spin-allowed
iff spin(p) == spin(s) (equivalently spin(q) == spin(r), which always holds
together with it once spin(p)==spin(r) and spin(q)==spin(s), the two
conditions pdaggerq's own nonzero spin blocks already guarantee). Unsuffixed
brackets (the plain spin-orbital pipeline) fold spin into a single generic
B tensor and always carry both terms.

Each output term is left in the same low-operand-count form the input had
(one bracket -> two 2-index B factors in place of one 4-index bracket) and
relies on cached_einsum.py's existing 'optimal' path search (<=4 operands)
to pick the low-storage contraction order automatically -- no separate
batching logic needed for terms with <=2 other (non-bracket) factors, which
covers MP2/MP3-density-scale equations. Terms needing an explicit vvvv-shaped
intermediate (batching) are a later-phase concern
and are NOT handled specially here.
"""
import re
from itertools import product

from src.Base.pdaggerq_interface import exact_coefficient

_AUX_POOL = 'QRSTUVWXYZ'


def _bracket_parts(fac):
    """Parse '<p,q||r,s>' or '<p,q||r,s>_ssss' -> (p, q, r, s, spins),
    spins = the 4-letter suffix string (e.g. 'abab') or None if unsuffixed."""
    fac = fac.strip()
    m = re.match(r'^<(.*)>(_[ab]+)?$', fac)
    if not m:
        return None
    inner, suffix = m.group(1), (m.group(2) or '')
    bra, ket = inner.split('||')
    p, q = [x.strip() for x in bra.split(',')]
    r, s = [x.strip() for x in ket.split(',')]
    spins = suffix[1:] if suffix else None
    return p, q, r, s, spins


def _b_name(spin_letter):
    return f"B_{spin_letter}{spin_letter}" if spin_letter else "B"


def dress_integral_factor(raw_terms):
    """Rewrite every bracket integral factor in raw_terms into its DF/RI
    expansion (see module docstring). Returns a new raw-term list in the
    exact pdaggerq string format raw_terms_to_einsum expects.

    Terms with no bracket factor pass through unchanged. A term with N
    bracket factors is dressed independently per bracket (Coulomb vs.
    exchange, when spin-allowed), yielding up to 2^N output terms whose
    coefficients carry the corresponding sign; each output term gets its own
    fresh auxiliary-index letter per bracket (drawn from an uppercase pool
    disjoint from pdaggerq's own lowercase occ/vir letters, the same
    freshness discipline laplace_codegen.py's _FRESH_OCC_POOL/_FRESH_VIR_POOL
    use), so distinct brackets in one term never share (and never
    accidentally contract against) each other's auxiliary sum.
    """
    out = []
    for term in raw_terms:
        coef0 = exact_coefficient(term[0])
        rest = term[1:]
        prefix_tokens = [t for t in rest if t.strip().startswith('P(')]
        other_tokens = [t for t in rest if not t.strip().startswith('P(')]

        bracket_idx = [i for i, tok in enumerate(other_tokens) if tok.strip().startswith('<')]
        if not bracket_idx:
            out.append(term)
            continue

        used_letters = set()
        for tok in rest:
            used_letters.update(re.findall(r'[a-zA-Z]', tok))
        aux_free = [c for c in _AUX_POOL if c not in used_letters]
        if len(aux_free) < len(bracket_idx):
            raise RuntimeError(
                "dress_integral_factor: ran out of fresh auxiliary letters "
                f"(need {len(bracket_idx)}, term already uses {sorted(used_letters)})")

        per_bracket_choices = []
        for k, i in enumerate(bracket_idx):
            p, q, r, s, spins = _bracket_parts(other_tokens[i])
            Q = aux_free[k]
            if spins is None:
                p_spin = q_spin = s_spin = None
            else:
                p_spin, q_spin, r_spin, s_spin = spins[0], spins[1], spins[2], spins[3]
            b_p, b_q = _b_name(p_spin), _b_name(q_spin)

            choices = [(1.0, [f"{b_p}({Q},{p},{r})", f"{b_q}({Q},{q},{s})"])]
            if spins is None or p_spin == s_spin:
                choices.append((-1.0, [f"{b_p}({Q},{p},{s})", f"{b_q}({Q},{q},{r})"]))
            per_bracket_choices.append(choices)

        for combo in product(*per_bracket_choices):
            sign = 1.0
            replacement = {}
            for i, (mult, facs) in zip(bracket_idx, combo):
                sign *= mult
                replacement[i] = facs
            new_coef = coef0 * sign
            new_body = []
            for i, tok in enumerate(other_tokens):
                new_body.extend(replacement[i]) if i in replacement else new_body.append(tok)
            out.append([f"{new_coef:+.15g}"] + prefix_tokens + new_body)

    return out
