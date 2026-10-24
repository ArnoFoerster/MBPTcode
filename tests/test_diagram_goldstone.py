"""Gates for the diagram-primary Goldstone engine (diagram-ADC generator).

Three layers, in the order they were built:
  1. n=2 vs the textbook closed-form 2nd-order self-energy, per sector, on
     H2 AND H4 (H4 is essential: H2's vv/sector-I and oo/sector-II entries
     vanish by parity, so H2 alone is blind to half the sign table).
  2. n=3 census vs SCW83 Fig. 3 (6 M^I + 6 M^II Goldstone orderings per
     block; static class 6; self-loop/W class 54).
  3. n=2 + n=3 pointwise vs the exact lambda-Taylor coefficients of the
     determinant-space dynamic self-energy (the committed arbiter in
     exact_oracle.py).  Tolerance is the oracle's polynomial-fit floor, not
     the engine's (engine == closed form at ~1e-17 where comparable).
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


@pytest.fixture(scope='module')
def h2():
    mol, mf = EO.h2_sto3g()
    return EO.spinorb_system(mol, mf)


@pytest.fixture(scope='module')
def h4():
    mol, mf = EO.h4_chain()
    return EO.spinorb_system(mol, mf)


def closed_form_m2(eps, g, nocc, omega, sector):
    norb = len(eps)
    o, v = slice(0, nocc), slice(nocc, norb)
    eo, ev = eps[:nocc], eps[nocc:]
    if sector == +1:
        num = np.einsum('piab,qiab->pqiab', g[:, o, v, v], g[:, o, v, v])
        den = omega + eo[:, None, None] - ev[None, :, None] - ev[None, None, :]
    else:
        num = np.einsum('paij,qaij->pqaij', g[:, v, o, o], g[:, v, o, o])
        den = omega + ev[:, None, None] - eo[None, :, None] - eo[None, None, :]
    return 0.5 * (num / den[None, None]).sum(axis=(2, 3, 4))


@pytest.mark.parametrize('system', ['h2', 'h4'])
@pytest.mark.parametrize('sector', [+1, -1])
def test_n2_closed_form(system, sector, request):
    eps, h1, g, nocc = request.getfixturevalue(system)
    omega = 0.3 + 0.4j
    eng = GS.assemble_m(eps, g, nocc, 2, [omega], sectors=(sector,))[0]
    cf = closed_form_m2(eps, g, nocc, omega, sector)
    assert np.max(np.abs(eng - cf)) < 1e-14


def test_n3_census():
    for block in [('v', 'v'), ('o', 'o'), ('v', 'o'), ('o', 'v')]:
        skels, census = GS.build_skeletons(3, block)
        n_ord = {+1: 0, -1: 0}
        for s in skels:
            for o in s.orderings:
                n_ord[GS.ordering_sector(s, o)] += 1
        assert census['kept'] == 12, (block, census)
        assert census['static'] == 6, (block, census)
        assert census['self-loop'] == 54, (block, census)
        assert census['reducible'] == 0, (block, census)
        assert n_ord == {+1: 6, -1: 6}, (block, n_ord)   # SCW83 Fig. 3


@pytest.mark.parametrize('system,tol', [('h2', 5e-9), ('h4', 5e-7)])
def test_n23_exact_lambda_taylor(system, tol, request):
    eps, h1, g, nocc = request.getfixturevalue(system)
    ora = EO.ExactSelfEnergy(eps, h1, g, nocc)
    omega = 0.3 + 0.4j
    cs = ora.m_coefficients(omega, orders=[2, 3], lam_max=0.3, n_lam=19, deg=9)
    for n in (2, 3):
        eng = GS.assemble_m(eps, g, nocc, n, [omega])[0]
        assert np.max(np.abs(eng - cs[n])) < tol, (system, n)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
