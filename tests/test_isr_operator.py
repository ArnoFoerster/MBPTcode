"""The matrix-free ISR engine (isr_operator.OperatorISR) equals the dense one
(isr_assembly.isr_effective_blocks), block for block and order for order.

1. Synthetic: random raw H/S series over three classes (diagonal overlaps
   symmetric with the order-0 identity, a norm series), every block carried
   through its manifold-rule order; the operator matvec against the dense
   matrix of the same truncation, with and without the norm series (every
   class is Loewdin-normalized, so the norm drops out exactly).
2. Non-Dyson IP/EA-ADC(n), n = 2..4, on LiH and H2O in sto-3g: the generated
   spin-orbital raw blocks of NonDysonISR fed to both engines, the operator
   matvec against NonDysonISR.matrix(n).
Both at several dense_max: every block matrix-free (0), the lowest classes'
effective states precomputed, and everything held dense.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.SingleReference.ADC.adc_order_rule import block_order
from src.SingleReference.ADC.isr_assembly import isr_effective_blocks
from src.SingleReference.ADC.isr_operator import OperatorISR
from src.SingleReference.ADC.nondyson_isr import MAX_NONDYSON_ORDER, NonDysonISR
from tests.generated_runtime import requires_mp_numerators, requires_runtime
from tests.test_nondyson_adc import _spinorb

TOL = 1e-12
NVEC = 3
# synthetic class dims 3, 4, 6: none, class 0, classes 0 and 1, all precomputed
SYNTHETIC_DENSE_MAX = [0, 9, 16, 10 ** 6]
NONDYSON_DENSE_MAX = [0, 10 ** 8]


def _dense(blocks, dims, top):
    cls = list(dims)
    offs = np.cumsum([0] + [dims[c] for c in cls])
    M = np.zeros((offs[-1], offs[-1]))
    for a, ci in enumerate(cls):
        for b, cj in enumerate(cls):
            for m, A in blocks[(ci, cj)].items():
                if m <= top[(ci, cj)]:
                    M[offs[a]:offs[a + 1], offs[b]:offs[b + 1]] += A
    return M


def _worst(op, M, rng):
    worst = 0.0
    for _ in range(NVEC):
        x = rng.standard_normal(M.shape[0])
        ref = M @ x
        worst = max(worst, float(np.max(np.abs(op.matvec(x) - ref))) / float(np.max(np.abs(ref))))
    return worst


@pytest.mark.parametrize('dense_max', SYNTHETIC_DENSE_MAX)
def test_synthetic_series(dense_max):
    rng = np.random.default_rng(11)
    n = 5
    dims = {0: 3, 1: 4, 2: 6}
    top = {(i, j): block_order(n, i, j) for i in dims for j in dims if i <= j}
    H, S = {}, {}
    for (i, j), t in top.items():
        lead = 0 if i == j else 1
        H[(i, j)], S[(i, j)] = {}, ({0: np.eye(dims[i])} if i == j else {})
        for m in range(lead, t + 1):
            A = 0.3 ** m * rng.standard_normal((dims[i], dims[j]))
            H[(i, j)][m] = 0.5 * (A + A.T) if i == j else A
            if m >= (2 if i == j else 1):
                B = 0.2 ** m * rng.standard_normal((dims[i], dims[j]))
                S[(i, j)][m] = 0.5 * (B + B.T) if i == j else B
    norm = {0: 1.0, 2: 0.07, 3: -0.02, 4: 0.01, 5: 0.004}
    blocks, _ = isr_effective_blocks(H, S, dims, n, norm=norm)
    full_top = {**top, **{(j, i): t for (i, j), t in top.items()}}
    M = _dense(blocks, dims, full_top)
    w = _worst(OperatorISR(H, S, dims, top, norm=norm, dense_max=dense_max), M, rng)
    # every class Loewdin-normalized: a scalar norm series rescales H and S alike
    # and drops out of the effective blocks
    w0 = _worst(OperatorISR(H, S, dims, top, dense_max=dense_max), M, rng)
    print(f"synthetic n={n} dense_max={dense_max}: operator vs dense {w:.2e}, "
          f"without the norm {w0:.2e}")
    assert w <= TOL, w
    assert w0 <= TOL, w0


@requires_runtime(MAX_NONDYSON_ORDER)
@requires_mp_numerators(MAX_NONDYSON_ORDER)
@pytest.mark.parametrize('atom', ['Li 0 0 0; H 0 0 1.6', 'O 0 0 0; H 0 0 0.96; H 0.96 0 0'])
@pytest.mark.parametrize('sector', ['IP', 'EA'])
@pytest.mark.parametrize('dense_max', NONDYSON_DENSE_MAX)
def test_nondyson_matches_dense(atom, sector, dense_max):
    _, eps, g, nocc = _spinorb(atom)
    nd = NonDysonISR(eps, g, nocc, sector)
    rng = np.random.default_rng(5)
    for n in range(2, MAX_NONDYSON_ORDER + 1):
        H, S, dims = nd.raw_series(n)
        top = {(i, j): nd.top_order(n, i, j, None) for (i, j) in H}
        op = OperatorISR(H, S, dims, top, norm=nd.norm(n), dense_max=dense_max)
        M, _, _ = nd.matrix(n)
        w = _worst(op, M, rng)
        print(f"{atom.split()[0]} {sector} ADC({n}) dense_max={dense_max}: dim {M.shape[0]}, "
              f"operator vs dense {w:.2e}")
        assert w <= TOL, (n, w)


def main():
    for dense_max in SYNTHETIC_DENSE_MAX:
        test_synthetic_series(dense_max)
    for atom in ['Li 0 0 0; H 0 0 1.6', 'O 0 0 0; H 0 0 0.96; H 0.96 0 0']:
        for sector in ('IP', 'EA'):
            for dense_max in NONDYSON_DENSE_MAX:
                test_nondyson_matches_dense(atom, sector, dense_max)
    print('\nALL OK')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
