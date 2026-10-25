"""Gates for the delta = diagram - ISR extraction (diagram-ADC, order 4).

Fast gates (no pdaggerq streams: diagram side = the EMITTED m{2,3,4}_blocks,
ISR side = isr_reference.ISRBlocks from the generated supermatrix blocks):

  1. basis-mapping gate: ISR-M2/M3 == diagram-M2/M3 on H4, machine
     precision (pins the identity mapping before order 4 is trusted).
  2. the ISR-ADC(4) lambda^4 defect on linear H4 is real (~8e-4) and
     delta(omega) closes EXACTLY in the pure-2p1h ADC correction form
     dU3'RU1 + U1'RdU3 + U1'R dC2 R U1 (out-of-sample machine precision):
     no k2-side correction objects are needed.
  3. delta == 0 identically on H2 (the quartet-blind system).
  4. the emitted q4_blocks (resummed quartet class) reproduce quartet_pf's
     closed form, and vanish identically on H2.

Stream gates (order-4 pdaggerq streams, ~2 min, shared with the other
diagram tests in one session):

  5. quartet census: every seven-line-cut skeleton has EXACTLY the four
     App-A orderings ([3w 7w 3w], [3w 7w 4f], [4f 7w 3w], [4f 7w 4f]),
     one common middle bipartition (QuartetGroup.consistency).
  6. the closed form G = (u+Em)/((u+Ea)(u+Eb) E_f E_f') equals the
     engine's filtered ordering sum pointwise (proves the per-tuple
     4p3h-pole cancellation on the real system).
"""
import numpy as np
import pytest

from src.CodeGeneration.diagrammatic import exact_oracle as EO
from src.CodeGeneration.diagrammatic import goldstone as GS
from src.CodeGeneration.diagrammatic import quartet_pf as QP
from src.CodeGeneration.diagrammatic.adc_form import (
    SectorSpaces, fit_u3_c2)
from src.CodeGeneration.diagrammatic.isr_reference import ISRBlocks
from tests.generated_runtime import requires_runtime

pytestmark = requires_runtime(4)
from src.SingleReference.ADC.generated_modules import lazy
m2_blocks = lazy('diagrammatic.m2_blocks')
m3_blocks = lazy('diagrammatic.m3_blocks')
m4_blocks = lazy('diagrammatic.m4_blocks')
q4_blocks = lazy('diagrammatic.q4_blocks')

OMEGAS = [0.3 + 0.5j, -0.9 + 0.7j, 1.7 + 0.8j, -1.7 + 0.9j,
          0.1 + 0.9j, -0.5 + 0.5j, 0.9 + 1.3j, 2.1 + 0.6j,
          -2.3 + 0.4j, 1.1 + 0.4j, -1.3 + 1.1j, 0.5 + 0.7j]
TEST_OMEGAS = [0.62 + 0.83j, -1.45 + 0.55j, 1.9 + 1.2j, -0.15 + 0.45j]


@pytest.fixture(scope='module')
def h4():
    mol, mf = EO.h4_chain()
    eps, h1, g, nocc = EO.spinorb_system(mol, mf)
    return eps, h1, g, nocc, ISRBlocks(eps, g, nocc)


@pytest.fixture(scope='module')
def h2():
    mol, mf = EO.h2_sto3g()
    eps, h1, g, nocc = EO.spinorb_system(mol, mf)
    return eps, h1, g, nocc, ISRBlocks(eps, g, nocc)


def test_isr_mapping_n23(h4):
    eps, h1, g, nocc, isr = h4
    worst = 0.0
    for w in OMEGAS[:4]:
        m = isr.m_orders(w)
        worst = max(worst,
                    np.max(np.abs(m[2] - m2_blocks.assemble_m2(eps, g, nocc, w))),
                    np.max(np.abs(m[3] - m3_blocks.assemble_m3(eps, g, nocc, w))))
    assert worst < 1e-13


def _delta(eps, g, nocc, isr, omegas):
    return np.array([m4_blocks.assemble_m4(eps, g, nocc, w)
                     - isr.m_orders(w)[4] for w in omegas])


def test_delta_closes_as_du3_dc2_h4(h4):
    eps, h1, g, nocc, isr = h4
    norb = len(eps)
    spaces = {s: SectorSpaces(eps, h1, g, nocc, s) for s in (+1, -1)}
    delta = _delta(eps, g, nocc, isr, OMEGAS)
    assert np.max(np.abs(delta)) > 5e-4          # the defect is real
    x, rin, _ = fit_u3_c2(spaces, delta, OMEGAS)
    assert rin < 1e-12

    # out-of-sample verdict: rebuild the fold from the fitted objects
    delta_t = _delta(eps, g, nocc, isr, TEST_OMEGAS)
    off = 0
    pred = np.zeros((len(TEST_OMEGAS), norb, norb), complex)
    for s in (+1, -1):
        sp = spaces[s]
        sg = 1.0 if s == +1 else -1.0
        nU = norb * sp.n1
        U3 = x[off:off + nU].reshape(sp.n1, norb)
        off += nU
        pairs = [(a, b) for a in range(sp.n1) for b in range(a, sp.n1)]
        C2 = np.zeros((sp.n1, sp.n1), complex)
        for k, (a, b) in enumerate(pairs):
            C2[a, b] = C2[b, a] = x[off + k]
        off += len(pairs)
        for iw, w in enumerate(TEST_OMEGAS):
            R = 1.0 / (w - sg * sp.K1)
            X = (U3.T * R) @ sp.U1
            pred[iw] += X + X.T
            pred[iw] += sg * (sp.U1.T * R) @ C2 @ (R[:, None] * sp.U1)
    assert np.max(np.abs(pred - delta_t)) < 1e-12


def test_delta_zero_h2(h2):
    eps, h1, g, nocc, isr = h2
    delta = _delta(eps, g, nocc, isr, OMEGAS[:4])
    assert np.max(np.abs(delta)) < 1e-13


def test_emitted_q4_matches_closed_form(h4):
    eps, h1, g, nocc, _ = h4
    ev = QP.QuartetEvaluator(eps, g, nocc)
    norb = len(eps)
    ranges = {'o': np.arange(nocc), 'v': np.arange(nocc, norb)}
    for w in OMEGAS[:2]:
        ref = np.zeros((norb, norb), complex)
        for block in GS.BLOCKS:
            grps = QP.quartet_groups(block)
            rb, rk = ranges[block[0]], ranges[block[1]]
            ref[np.ix_(rb, rk)] += ev.eval_groups(grps, w)
        got = q4_blocks.assemble_q4(eps, g, nocc, w)
        assert np.max(np.abs(got - ref)) < 1e-16
        assert np.max(np.abs(ref)) > 5e-4        # the class is nonzero on H4


def test_emitted_q4_zero_h2(h2):
    eps, h1, g, nocc, _ = h2
    for w in OMEGAS[:2]:
        assert np.max(np.abs(q4_blocks.assemble_q4(eps, g, nocc, w))) < 1e-15


def test_quartet_census_and_group_structure(h4):
    for block in GS.BLOCKS:
        grps = QP.quartet_groups(block)
        assert len(grps) >= 16
        for grp in grps:
            assert set(grp.orderings) == {'O1', 'O2', 'O3', 'O4'}
            assert grp.consistency()


def test_closed_form_equals_ordering_sum_h4(h4):
    eps, h1, g, nocc, _ = h4
    ev = QP.QuartetEvaluator(eps, g, nocc)
    worst = 0.0
    for block in GS.BLOCKS:
        grps = QP.quartet_groups(block)
        for w in OMEGAS[:2]:
            a = ev.eval_groups(grps, w)
            b = ev.eval_block([grp.skel for grp in grps], w,
                              ordering_filter=GS.sevenline_filter)
            worst = max(worst, np.max(np.abs(a - b)))
    assert worst < 1e-15
