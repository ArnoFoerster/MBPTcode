"""Production union-space and diagrammatic ADC(5)
(sigma_driver.restricted_dyson_operator, construction='union'/'diagrammatic',
level='adc5') against the lambda oracle's union and decoupled ADC(5).

Union ADC(5) is the two-sector ISR-ADC(5) supermatrix plus the D x C cross
blocks the two-sector convention deletes, each carried through its manifold-rule
order: the 2h1p-dagger x 2p1h block through order 3 (X^(2) and the effective
X^(3), raw block minus its order-3 class Gram-Schmidt terms) and the
2h1p-dagger x 3p2h and 3h2p-dagger x 2p1h blocks at order 2 (Tarantelli and
Cederbaum, PRA 45, 2790 (1992)). The oracle folds the same theory in the
determinant space (tests/union_space_oracle.py: SeriesFold, mode 'union', adc_rule(5)).
Diagrammatic ADC(5) is the two-sector supermatrix plus the coupling rows of the
cross blocks decoupled through fifth order (sigma_driver._diagrammatic_adc5_rows);
the oracle decouples its union series (decouple_cross_blocks) and truncates.

Gates on linear H4 (r = 1.0) and LiH in sto-3g, SCF to the f[o,v] floor, the
orbital block carrying Sigma(infinity) through order 5 (MP4 density):
  1. each operator is symmetric;
  2. its principal IP/EA poles equal the oracle's poles of the same construction;
  3. those differ from the two-sector poles by more than the gate (the cross
     blocks are not a no-op on these systems).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.SingleReference.ADC.sigma_driver import restricted_dyson_operator
from tests.generated_runtime import requires_runtime
from tests.test_adc5_poles import NORD, _production
from tests import union_space_oracle as CB

SYSTEMS = ['H4lin_r1.0', 'LiH']
NSTATES = 2
TOL = 1e-10
TOL_SYM = 1e-12

pytestmark = requires_runtime(5)


def _principal(M, norb):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    strength = np.sum(V[:norb] ** 2, axis=0)
    ip, _ = CB.principal_poles(w, strength, -1, NSTATES)
    ea, _ = CB.principal_poles(w, strength, +1, NSTATES)
    return np.asarray(ip), np.asarray(ea)


def _oracle_poles(us, mode):
    """Principal poles of the oracle's ADC(5) in construction `mode`; the
    diagrammatic one is the union series decoupled (decouple_cross_blocks)."""
    psi, E, nser = us.rspt_series(NORD)
    Ssr, Msr = us.series_matrices(psi, E, nser, NORD)
    orb = us.idx[(0, 'o')]
    Fs = sum(Msr[k][np.ix_(orb, orb)] for k in range(NORD))
    sf = CB.SeriesFold(us, Ssr, Msr, NORD, 'twosector' if mode == 'twosector' else 'union',
                       us.kmax)
    if mode == 'diagrammatic':
        sf.B, sf.D, _, _ = CB.decouple_cross_blocks(sf, NORD)
    B, D = sf.truncated(CB.adc_rule(5), kcap=2)
    keep = sf.kseg <= 2
    M = np.block([[Fs, sum(B)[:, keep]], [sum(B)[:, keep].T, sum(D)[np.ix_(keep, keep)]]])
    return _principal(M, us.norb)


def _dense(op):
    return np.column_stack([op.matvec(e) for e in np.eye(op.shape[0])])


def _diff(a, b):
    """Largest |difference| over paired IP and EA poles; a pole present on one
    side only (nan on the other) counts as infinitely far."""
    d = 0.0
    for xs, ys in zip(a, b):
        for x, y in zip(xs, ys):
            if np.isnan(x) and np.isnan(y):
                continue
            d = max(d, float('inf') if np.isnan(x) or np.isnan(y) else abs(x - y))
    return d


@pytest.mark.parametrize('name', SYSTEMS)
@pytest.mark.parametrize('construction', ['union', 'diagrammatic'])
def test_adc5_construction_matches_oracle(name, construction):
    sysd = CB.build_system(name)
    us = CB.UnionSpace(sysd)
    ref = _oracle_poles(us, construction)
    ref_two = _oracle_poles(us, 'twosector')
    r, f_orb = _production(sysd)
    M = _dense(restricted_dyson_operator(r, 'adc5', f_orb=f_orb, construction=construction)[0])
    asym = float(np.max(np.abs(M - M.T)))
    mine = _principal(M, len(f_orb))
    d = _diff(mine, ref)
    d_two = _diff(ref, ref_two)
    print(f"{name} {construction} ADC(5): IP {np.round(mine[0], 9)} EA {np.round(mine[1], 9)}; "
          f"oracle IP {np.round(ref[0], 9)} EA {np.round(ref[1], 9)}; "
          f"|production - oracle| {d:.2e}; |oracle {construction} - two-sector| {d_two:.2e}; "
          f"symmetry {asym:.1e}")
    assert asym <= TOL_SYM, asym
    assert d <= TOL, d
    assert d_two > 1e3 * TOL, d_two


def main():
    for name in SYSTEMS:
        for construction in ('union', 'diagrammatic'):
            test_adc5_construction_matches_oracle(name, construction)
    print('\nALL OK')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
