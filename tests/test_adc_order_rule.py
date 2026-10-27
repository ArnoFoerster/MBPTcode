"""Gate for the order-by-order ADC(n) assembly rule (adc_order_rule.py): it must
reproduce the HAND-DERIVED, oracle-validated order-3 assembly in
the hand-derived U^(3) assembly exactly -- every (tag structure, l-order, t-order)
triple, nothing missing, nothing extra.

This is the regression that lets the generator be genuinely ADC(n): if the rule
reproduces the order-3 assembly that the lambda-oracle validated, then applying
it at order 4/5 needs no new hand derivation.

The 'surviving' sets below are the structures that pdaggerq actually emitted
(non-zero term count) -- measured from the generated modules; every other
candidate the rule proposes has no generated function and is skipped by the
driver's getattr(..., None), so it cannot contribute.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.SingleReference.ADC.adc_order_rule import (
    mp_min_order, u_order_terms, s_order_terms)

# Tag structures with a non-zero generated block (adc3_u_blocks_restricted /
# adc4_u3_blocks_restricted). Everything else pdaggerq pruned to 0 terms.
SURVIVING_U = {'l2v', 'l2ft2', 'l1f',            # order-2 survivors
               'l2vt2', 'vt1', 'l1v', 'l2ft1', 'l3ft2'}   # order-3 survivors
SURVIVING_S = {'l1', 'l2t1', 'l3t2'}

# the hand-derived U^(3) raw list: (structure, l_order, t_order)
U3_RAW = {('l2vt2', 1, 1), ('vt1', 0, 2), ('l1v', 2, 0), ('l2v', 2, 0),
          ('l2ft2', 1, 2), ('l2ft2', 2, 1), ('l2ft1', 1, 2),
          ('l3ft2', 2, 1), ('l1f', 3, 0)}
# its S^(3) list
U3_S = {('l2t1', 1, 2), ('l3t2', 2, 1), ('l1', 3, 0)}


def check(label, ok, extra=''):
    print(f"[{'OK' if ok else 'FAIL'}] {label}{extra}")
    return ok


def main():
    all_ok = True

    all_ok &= check('mp_min_order: rank2->1, rank1->2, rank3->2, rank4->2, rank5->3',
                    (mp_min_order(2), mp_min_order(1), mp_min_order(3),
                     mp_min_order(4), mp_min_order(5)) == (1, 2, 2, 2, 3))

    pred = {(nm, a, b) for nm, j, ham, k, a, b in u_order_terms(3) if nm in SURVIVING_U}
    all_ok &= check('U^(3) rule == u3_generated raw list', pred == U3_RAW,
                    f"  (missing {U3_RAW - pred or '-'}, extra {pred - U3_RAW or '-'})")

    preds = {(nm, a, b) for nm, j, k, a, b in s_order_terms(3) if nm in SURVIVING_S}
    all_ok &= check('S^(3) rule == u3_generated S^(3) list', preds == U3_S,
                    f"  (missing {U3_S - preds or '-'}, extra {preds - U3_S or '-'})")

    # order 2 sanity: the surviving order-2 structures are reachable at n=2
    p2 = {nm for nm, *_ in u_order_terms(2) if nm in SURVIVING_U}
    all_ok &= check('U^(2) rule reaches l2v/l2ft2/l1f', {'l2v', 'l2ft2', 'l1f'} <= p2)

    # GENERATION superset: the order rule's structures cover every split the
    # hand-written tables of the ADC(3) U^(2) and ADC(4) U^(3) generators
    # carried before the generators derived them from this rule (extras are
    # fine -- pdaggerq prunes them)
    from src.SingleReference.ADC.adc_order_rule import structures_up_to, new_structures_at
    hand2 = {'vt2', 'l2v', 'l2ft2', 'ft1', 'l1f', 'ft2', 'l2f', 'ft3', 'l3f'}
    rule2 = set(structures_up_to(2))
    all_ok &= check('generation: structures_up_to(2) covers the U^(2) splits',
                    hand2 <= rule2, f"  (missing {sorted(hand2 - rule2) or '-'})")
    hand3 = {'l2vt2', 'vt1', 'vt3', 'l1v', 'l3v', 'l2ft1', 'l2ft3', 'l1ft2', 'l3ft2'}
    rule3 = set(new_structures_at(3))
    all_ok &= check('generation: new_structures_at(3) covers the U^(3) splits',
                    hand3 <= rule3, f"  (missing {sorted(hand3 - rule3) or '-'})")

    print("\nALL OK" if all_ok else "\nSOME FAILED")
    return all_ok


def test_main():
    assert main()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
