"""Gates for the emitted delta-U3 rows (order-4 defect theorem, 6f class).

The generated module delta_u3_blocks.py carries the closed-form rows of
the ISR-ADC(4) defect: delta4-residue == diag_6f (the [3w|6f|4f]
uncontracted-3p3h-ladder orderings; one omega-cut each, so the rows are
static).  Fast gates only (the emitted module is standalone numpy; the
diagram side of the closure gate is the emitted m4_blocks, the ISR side
isr_reference.ISRBlocks -- no pdaggerq streams):

  1. closure: on linear H4 the order-4 defect field
     delta4(omega) = diagram-M4 - ISR-M4 is reproduced out-of-sample by
     the pure single-pole fold sym(U1' R du3) of the emitted rows.
  2. t3 == 0 arbiters: on H2 AND rectangular H4 (1.0 x 1.4) the emitted
     rows vanish IDENTICALLY (the 6f numerators are zero there), while
     linear H4 keeps them alive (~1e-3).
"""
import numpy as np
import pytest

from src.CodeGeneration.diagrammatic import exact_oracle as EO
from src.CodeGeneration.diagrammatic.adc_form import SectorSpaces
from src.CodeGeneration.diagrammatic.isr_reference import ISRBlocks
from tests.generated_runtime import requires_runtime

pytestmark = requires_runtime(4)
from src.SingleReference.ADC.generated_modules import lazy
delta_u3_blocks = lazy('diagrammatic.delta_u3_blocks')
delta_u3_laplace_blocks = lazy('diagrammatic.delta_u3_laplace_blocks')
m4_blocks = lazy('diagrammatic.m4_blocks')

OMEGAS = [0.41 + 0.77j, -1.13 + 0.65j, 1.55 + 0.5j, -0.35 + 1.1j,
          2.05 + 0.45j, -2.15 + 0.85j]


@pytest.fixture(scope='module')
def h4():
    mol, mf = EO.h4_chain()
    eps, h1, g, nocc = EO.spinorb_system(mol, mf)
    return eps, h1, g, nocc


def test_du3_closes_delta4_h4(h4):
    eps, h1, g, nocc = h4
    norb = len(eps)
    spaces = {s: SectorSpaces(eps, h1, g, nocc, s) for s in (+1, -1)}
    isr = ISRBlocks(eps, g, nocc)
    du = dict(zip((+1, -1), delta_u3_blocks.assemble_du3(eps, g, nocc)))
    worst = scale = 0.0
    for w in OMEGAS:
        delta = m4_blocks.assemble_m4(eps, g, nocc, w) - isr.m_orders(w)[4]
        rec = np.zeros((norb, norb), complex)
        for s in (+1, -1):
            sp = spaces[s]
            R = 1.0 / (w - (1.0 if s > 0 else -1.0) * sp.K1)
            rec += (sp.U1.T * R) @ du[s] + (du[s].T * R) @ sp.U1
        scale = max(scale, np.max(np.abs(delta)))
        worst = max(worst, np.max(np.abs(rec - delta)))
    assert scale > 1e-4          # the defect is real on linear H4
    assert worst < 1e-9


def test_du3_laplace_matches_dense(h4):
    """The production (rank<=4 Laplace) evaluator == the dense validation
    twin at the default ntau (machine floor of the minimax grid)."""
    eps, h1, g, nocc = h4
    dp, dm = delta_u3_blocks.assemble_du3(eps, g, nocc)
    lp, lm = delta_u3_laplace_blocks.assemble_du3_laplace(eps, g, nocc)
    assert np.max(np.abs(lp - dp)) < 1e-11
    assert np.max(np.abs(lm - dm)) < 1e-11


@pytest.mark.parametrize('system', ['h2', 'h4_rect'])
def test_du3_vanishes_where_t3_vanishes(system):
    if system == 'h2':
        mol, mf = EO.h2_sto3g()
    else:
        from pyscf import gto
        mol = gto.M(atom='H 0 0 0; H 0 0 1.0; H 0 1.4 0; H 0 1.4 1.0',
                    basis='sto-3g', verbose=0)
        mf = EO._tight_rhf(mol)
    eps, h1, g, nocc = EO.spinorb_system(mol, mf)
    du_p, du_m = delta_u3_blocks.assemble_du3(eps, g, nocc)
    assert np.max(np.abs(du_p)) < 1e-14
    assert np.max(np.abs(du_m)) < 1e-14
