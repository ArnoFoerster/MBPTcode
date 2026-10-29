"""The block solver's start (ee_fold.solve_folded(..., start=)) and the ADC(2)
state model that hands it the last call's roots. Water 6-31G (bent,
non-symmetric), frozen 1, cc-pvdz-ri / cc-pvdz-jkfit, 3 singlet roots.

  1. no start: solve_folded(solver='block') gives the same roots and vectors,
     bit for bit, as the call without the keyword.
  2. a start that is the roots themselves: the same roots to 1e-10 Ha in fewer
     products than the cold solve.
  3. the state model (water cc-pVDZ, above the dense limit) in a moved field
     (the call before at another field, its roots carried as AO transition
     amplitudes): the cold model's roots to 1e-9 Ha (both to 1e-10), in fewer
     products.
  4. a start with another solver than 'block' is refused.

Run: python -m pytest -q tests/test_ee_adc_fold_start.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.gradients.adc.ee_state_model import ADC2State
from src.Base.numerics.numerics import Numerics
from src.gradients.state_models import HFState, MP2State
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
from tests.adc_common import ATOMS, AUX, NF, make_scf_loop, molecule

MOL = molecule(ATOMS)
TOL = 1e-10


def operator():
    hf = HFState(MOL, make_scf_loop)
    hf(np.zeros((MOL.nao, MOL.nao)), 0.0)
    f = ee_ri_fold.bare_factors(MOL, hf.mf, frozen=NF, auxbasis=AUX)
    return ee_ri_fold.RIFoldADC2(f, 'singlet')


def solve(op, **kw):
    n0 = op.matvecs + op.image_columns
    r = ee_fold.solve_folded(op.pieces(), 3, spin='singlet', tol_omega=TOL,
                             tol_residual=1e-6, solver='block', **kw)
    return r, op.matvecs + op.image_columns - n0


OP = operator()
COLD, N_COLD = solve(OP)


def test_no_start_is_the_default_bitwise():
    r, _ = solve(OP, start=None)
    assert np.array_equal(r.omega, COLD.omega) and np.array_equal(r.y, COLD.y)


def test_start_from_the_roots():
    r, n = solve(OP, start=np.asarray(COLD.y))
    assert np.max(np.abs(r.omega - COLD.omega)) < TOL
    assert n < N_COLD, (n, N_COLD)


# above EE_FOLD_DENSE_LIMIT singles, where the state model's solve is the block
MOL_DZ = molecule(ATOMS, basis='cc-pvdz')


def model():
    mp2 = MP2State(HFState(MOL_DZ, make_scf_loop), NF, AUX, force_route='dense',
                   numerics=Numerics(adc={'conv_tol': TOL}))
    return ADC2State(mp2, 'singlet', 0, 3, None, follow='index')


class Columns:
    def __enter__(self):
        self.n0 = ee_ri_fold.RIFoldADC2.__init__
        self.ops = []
        outer = self

        def init(op, *a, **k):
            outer.ops.append(op)
            return outer.n0(op, *a, **k)
        ee_ri_fold.RIFoldADC2.__init__ = init
        return self

    def __exit__(self, *exc):
        ee_ri_fold.RIFoldADC2.__init__ = self.n0

    @property
    def n(self):
        return sum(op.matvecs + op.image_columns for op in self.ops)


def test_state_model_carries_its_roots():
    h = 1e-3 * MOL_DZ.intor('int1e_r')[2]
    warm = model()
    warm(np.zeros((MOL_DZ.nao, MOL_DZ.nao)), 0.0)
    with Columns() as cw:
        rw = warm(h, 0.0)
    cold = model()
    with Columns() as cc:
        rc = cold(h, 0.0)
    assert np.max(np.abs(rw.info['omega_all'] - rc.info['omega_all'])) < 1e-9
    assert cw.n < cc.n, (cw.n, cc.n)


def test_start_needs_the_block_solver():
    with pytest.raises(ValueError, match='block'):
        ee_fold.solve_folded(OP.pieces(), 3, spin='singlet', solver='newton',
                             start=np.asarray(COLD.y))
