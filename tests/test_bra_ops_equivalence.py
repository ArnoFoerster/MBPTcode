"""bra_ops() must be EXACTLY the e{rank} form, and must reach past rank 4.

Why this matters
----------------
`t_numerator_terms` derived MPn amplitude numerators via
`set_left_operators([[bra_string(rank)]])`, i.e. pdaggerq's `e{rank}` operator.
pdaggerq supports only e1-e4 ("only e1, e2, e3, and e4 operators are supported")
and the error ABORTS THE PROCESS at C++ level -- so rank>=5 amplitudes, which
ADC(5)+ needs (measured: surviving rank == n-1, so ADC(9) needs t8), could not
be derived at all.

`bra_ops(rank)` spells the same bra as an explicit a*/a product, which pdaggerq
accepts at any rank. The ordering (all a*(occ) ascending, THEN all a(vir)
descending) was found EMPIRICALLY -- an interleaved a*(occ_k) a(vir_k) form
matches at rank 4 by permutation parity but is WRONG at ranks 2 and 3. So this
gate checks exact term-by-term equality against the e-form wherever the e-form
is available, which is the only way to know the swap did not silently change
every MPn equation.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.CodeGeneration.pdaggerq_interface import pdaggerq_available

pytestmark = pytest.mark.skipif(not pdaggerq_available(),
                                reason='needs pdaggerq (optional)')

from src.CodeGeneration.pdaggerq_interface import pq_helper
from src.CodeGeneration.generate_mpn_density import (
    bra_string, bra_ops, amp_out_indices, filter_connected, t_numerator_terms)


def _terms(left, rank, dress):
    pq = pq_helper('fermi')
    pq.set_left_operators([left])
    pq.set_right_operators([['1']])
    for r in dress:
        pq.add_operator_product(1.0, ['v', f't{r}'])
    pq.simplify()
    out = [tuple(t) for t in filter_connected(pq.strings(), amp_out_indices(rank))]
    pq.clear()
    return out


def main():
    all_ok = True

    # ---- 1. EXACT equivalence wherever pdaggerq can express the e-form ----
    for rank in (2, 3, 4):
        ref = _terms([bra_string(rank)], rank, [rank])
        got = _terms(bra_ops(rank), rank, [rank])
        ok = got == ref and len(ref) > 0
        print(f"[{'OK' if ok else 'FAIL'}] rank {rank}: bra_ops == e{rank} exactly "
              f"({len(ref)} terms, order-sensitive compare)")
        all_ok &= ok

    # ---- 2. the interleaved ordering is WRONG -- guards against 'simplifying' ----
    occ = amp_out_indices(3)[3:]
    vir = list(reversed(amp_out_indices(3)[:3]))
    interleaved = [x for k in range(3) for x in (f'a*({occ[k]})', f'a({vir[k]})')]
    ok = _terms(interleaved, 3, [3]) != _terms(bra_ops(3), 3, [3])
    print(f"[{'OK' if ok else 'FAIL'}] rank 3: interleaved ordering differs "
          "(ordering is load-bearing, not cosmetic)")
    all_ok &= ok

    # ---- 3. the whole point: rank 5 now derives (e5 would abort the process) ----
    terms5 = t_numerator_terms(5, 3)
    ok = len(terms5) > 0
    print(f"[{'OK' if ok else 'FAIL'}] rank 5 numerator derives past the e1-e4 cap "
          f"({len(terms5)} terms at order 3)")
    all_ok &= ok

    # ---- 4. rank<=4 numerators unchanged by the swap (regression pins) ----
    for (r, o, n) in ((2, 1, 1), (2, 2, 3), (3, 2, 8), (4, 2, 9)):
        got = len(t_numerator_terms(r, o))
        ok = got == n
        print(f"[{'OK' if ok else 'FAIL'}] t{r}^({o}) still {n} terms (got {got})")
        all_ok &= ok

    print('ALL OK' if all_ok else 'FAILURES')
    return all_ok


def test_main():
    assert main()


if __name__ == '__main__':
    if not pdaggerq_available():
        print('SKIPPED: needs pdaggerq (optional)')
        sys.exit(0)
    sys.exit(0 if main() else 1)
