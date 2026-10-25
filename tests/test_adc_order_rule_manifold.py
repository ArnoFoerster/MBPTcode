"""The MANIFOLD axis of the ADC(n) order rule: which excitation classes exist
at order n, and to what order each supermatrix block is carried.

Why this test exists
--------------------
The order rule emits the class manifolds of each order instead of a per-level
hand-maintained list: an order rule that EMITS the manifolds generalizes from
the orders it was validated at (2..6) to 7..25, a hand-maintained list does
not.

The rule under test is Schirmer's: block(mu_I, mu_J) is carried through order
n - mu_I - mu_J, and class mu is present iff 2*mu <= n.

Every assertion below is a fact stated independently of the rule, so this is a
reproduction test, not a restatement:

  * ADC(3) == "K + first-order C^(1) + order-2 U, NO C^(2)/3h2p"
    (restricted_dyson_operator docstring, level 'adc3')
  * ADC(4) carries U^(3) (orb-2h1p), C^(2) (2h1p-2h1p), U^(2)_3h2p (orb-3h2p),
    C^(1) 2h1p-3h2p coupling, and K ONLY on 3h2p-3h2p
  * "U order = n-1, C order = n-2"
  * the 3h2p-3h2p block is order 0 at ADC(4) -- which is WHY
    solve_dyson_adc4_downfold's Schur complement over the 3-particle manifold
    is exact rather than approximate -- and order 1 at ADC(5), the
    3h2p x 3h2p self-coupling.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.SingleReference.ADC.adc_order_rule import (
    kmax, block_order, blocks_at, amplitude_max_rank)


def check(ok, msg):
    print(f"[{'OK' if ok else 'FAIL'}] {msg}")
    return ok


def main():
    all_ok = True

    # ---- ADC(3): the level-'adc3' docstring, four claims ----
    all_ok &= check(kmax(3) == 1, "ADC(3): kmax=1 -> no 3h2p manifold")
    all_ok &= check(block_order(3, 1, 1) == 1, "ADC(3): 2h1p-2h1p order 1 = C^(1)")
    all_ok &= check(block_order(3, 0, 1) == 2, "ADC(3): orb-2h1p order 2 = order-2 U")
    all_ok &= check(block_order(3, 1, 1) < 2, "ADC(3): NO C^(2)")

    # ---- ADC(4): every hand-wired block, exactly ----
    expect4 = [(0, 0, 4), (0, 1, 3), (0, 2, 2), (1, 1, 2), (1, 2, 1), (2, 2, 0)]
    all_ok &= check(blocks_at(4) == expect4, f"ADC(4) block set == {expect4}")
    all_ok &= check(block_order(4, 0, 1) == 3, "ADC(4): orb-2h1p = U^(3)")
    all_ok &= check(block_order(4, 1, 1) == 2, "ADC(4): 2h1p-2h1p = C^(2)")
    all_ok &= check(block_order(4, 0, 2) == 2, "ADC(4): orb-3h2p = U^(2)_3h2p")
    all_ok &= check(block_order(4, 1, 2) == 1, "ADC(4): 2h1p-3h2p = C^(1) coupling")

    # ---- the downfold-exactness fact ----
    all_ok &= check(block_order(4, 2, 2) == 0,
                    "ADC(4): 3h2p-3h2p order 0 (K only) -> Schur downfold is EXACT")
    all_ok &= check(block_order(5, 2, 2) == 1,
                    "ADC(5): 3h2p-3h2p order 1 -> the 'last matrix-free gate'")

    # ---- the general U/C statement ----
    ok = all(block_order(n, 0, 1) == n - 1 and block_order(n, 1, 1) == n - 2
             for n in range(2, 26))
    all_ok &= check(ok, "U order = n-1 and C order = n-2 for all n in 2..25")

    # ---- manifold growth: new class every SECOND order ----
    all_ok &= check([kmax(n) for n in range(2, 9)] == [1, 1, 2, 2, 3, 3, 4],
                    "kmax: new manifold every second order (2/3->1, 4/5->2, 6/7->3)")

    # ---- no block is ever emitted with negative order ----
    ok = all(o >= 0 for n in range(2, 26) for _, _, o in blocks_at(n))
    all_ok &= check(ok, "no negative-order blocks emitted, n in 2..25")

    # ---- the historical max_rank=4 default is NOT safe past ADC(3) ----
    all_ok &= check(amplitude_max_rank(3) == 4 and amplitude_max_rank(5) > 4,
                    "amplitude_max_rank: 4 at ADC(3), >4 beyond (default is unsafe there)")

    print('ALL OK' if all_ok else 'FAILURES')
    return all_ok


def test_main():
    assert main()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
