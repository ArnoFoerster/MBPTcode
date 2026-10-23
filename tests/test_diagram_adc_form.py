"""ADC-form gates for the diagram-ADC track.

1. n=3 identity: the engine's C-class orderings == U1 R C1 R U1 and its
   U-class == the SCW83 (44b)/(45) U^(2)-part, both at machine precision on
   H2 -- the engine re-derives the published third-order ADC objects.
2. ADC(4)-form closure on H2: after subtracting all pure-k1 known 4th-order
   terms, the residual is EXACTLY representable as U3'RU1 + U1'RU3 +
   U1'R C2 R U1 (overdetermined fit, machine precision).  H2 has no 3p2h
   space (2 virtuals), so the k1 closure is the complete statement there --
   and H2 is precisely where the PRINTED SCW83 B3/B7 objects fail the
   lambda-oracle, so these machine-extracted objects supersede them.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.CodeGeneration.pdaggerq_interface import pdaggerq_available

pytestmark = pytest.mark.skipif(not pdaggerq_available(),
                                reason='needs pdaggerq (optional)')

from src.CodeGeneration.diagrammatic import exact_oracle as EO
from src.CodeGeneration.diagrammatic import goldstone as GS
from src.CodeGeneration.diagrammatic.adc_form import (
    SectorSpaces, fit_u3_c2)


@pytest.fixture(scope='module')
def h2():
    mol, mf = EO.h2_sto3g()
    eps, h1, g, nocc = EO.spinorb_system(mol, mf)
    spaces = {s: SectorSpaces(eps, h1, g, nocc, s) for s in (+1, -1)}
    return eps, h1, g, nocc, spaces


def _engine_by_class(eps, g, nocc, order, omega, want):
    """Engine assembly restricted to C-class (all cuts omega) or U-class."""
    norb = len(eps)
    ev = GS.BlockEvaluator(eps, g, nocc)
    out = np.zeros((norb, norb), complex)
    ranges = {'o': np.arange(nocc), 'v': np.arange(nocc, norb)}
    for block in GS.BLOCKS:
        skels, _ = GS.build_skeletons(order, block)
        rb, rk = ranges[block[0]], ranges[block[1]]
        blk = np.zeros((len(rb), len(rk)), complex)
        for skel in skels:
            num, dummies = ev._numerator(skel)
            weight = skel.coeff / skel.n_ext
            for ordering in skel.orderings:
                sec = GS.ordering_sector(skel, ordering)
                cuts = GS.ordering_cuts(skel, ordering)
                cls = 'C' if all(c[0] != 0 for c in cuts) else 'U'
                if cls != want:
                    continue
                val = num.astype(complex)
                for cut in cuts:
                    val = val / ev._cut_denominator(cut, dummies, omega, sec)
                blk += (weight * GS.ordering_sign(skel, sec)
                        * val.sum(axis=tuple(range(2, val.ndim))))
        out[np.ix_(rb, rk)] += blk
    return out


def test_n3_engine_equals_scw83_objects(h2):
    eps, h1, g, nocc, spaces = h2
    omega = 0.3 + 0.4j
    ref_c = sum(spaces[s].m3_parts(omega)[0] for s in (+1, -1))
    ref_u = sum(spaces[s].m3_parts(omega)[1] for s in (+1, -1))
    eng_c = _engine_by_class(eps, g, nocc, 3, omega, 'C')
    eng_u = _engine_by_class(eps, g, nocc, 3, omega, 'U')
    assert np.max(np.abs(eng_c - ref_c)) < 1e-14
    assert np.max(np.abs(eng_u - ref_u)) < 1e-14


def test_adc4_form_closure_h2(h2):
    eps, h1, g, nocc, spaces = h2
    assert spaces[+1].n2 == 0 and spaces[-1].n2 == 0   # no 3p2h on H2
    rng = np.random.default_rng(3)
    omegas = [complex(re, im) for re, im in
              zip(rng.uniform(-1.5, 1.5, 12), rng.uniform(0.35, 1.2, 12))]
    delta = np.array([
        GS.assemble_m(eps, g, nocc, 4, [w])[0]
        - sum(spaces[s].known_m4_k1(w) for s in (+1, -1))
        for w in omegas])
    x, resid, A = fit_u3_c2(spaces, delta, omegas)
    assert A.shape[0] > A.shape[1]        # genuinely overdetermined
    assert resid < 1e-13, resid           # exact ADC-form closure


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
