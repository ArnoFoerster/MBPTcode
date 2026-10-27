"""A failed analytic continuation is NaN with its exception named; a defect raises.

`ac_quasiparticle_shifts` feeds the adaptive selection and the outside curve,
which drop a state whose continuation failed ('continuation failed', never
scissored on its word). Only the continuation's own numerics may fail that way:
an ArithmeticError or LinAlgError becomes NaN, and the exception class reaches
the record (`ac_unjudgeable`, `fit_outside_curve`). Any other exception is a
defect and propagates; before this test a RuntimeError or ValueError raised
inside the continuation also came back as NaN with nothing said.

Water/cc-pVDZ DF-RHF, G1 ISDF factors, the two frontier orbitals, with the
failure planted in `qp_solve.solve_qp_from_imaginary_axis` for the LUMO.
"""
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

import src.SingleReference.GW.qp_solve as qp_solve
from src.Base.numerics.numerics import Numerics
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.SingleReference.GW.qp_selection import (ac_quasiparticle_shifts,
                                                 ac_unjudgeable)
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.GW.qp_states import fit_outside_curve
from src.SingleReference.GW.space_time import separable_factors

ATOM = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
SHIFT_MAX_EV = 10.0


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = 1e-10
    mf.kernel()
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', ['H', 'O'],
                                        auxbasis='cc-pvdz-ri')
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri', counts=counts,
                                n_start=n_start)
    nocc = mol.nelectron // 2
    states = np.array([nocc - 1, nocc])
    xc = static_exchange_diagonal(mf, mol, states)
    return mol, mf, nocc, states, tuple(factors[:2]), xc


def _plant(monkeypatch, exc, lumo):
    """The continuation of `lumo` raises `exc`; every other state is solved."""
    real = qp_solve.solve_qp_from_imaginary_axis

    def continuation(eps, p, *args, **kwargs):
        if p == lumo:
            raise exc
        return real(eps, p, *args, **kwargs)
    monkeypatch.setattr(qp_solve, 'solve_qp_from_imaginary_axis',
                        continuation)


def test_a_numerical_failure_is_nan_with_its_exception_recorded(water,
                                                                monkeypatch):
    mol, mf, nocc, states, factors, xc = water
    _plant(monkeypatch, np.linalg.LinAlgError('planted singular fit'), nocc)
    failures = {}
    a, z = ac_quasiparticle_shifts(mf, mol, nocc, states, factors=factors,
                                   xc_diagonal=xc, failures=failures,
                                   numerics=Numerics())
    assert np.isnan(a[nocc]) and np.isnan(z[nocc])
    assert np.isfinite(a[nocc - 1]) and 0 < z[nocc - 1] <= 1
    assert failures == {nocc: 'LinAlgError: planted singular fit'}
    bad = ac_unjudgeable(a, z, SHIFT_MAX_EV, failures)
    assert bad == {nocc: 'continuation failed '
                         '(LinAlgError: planted singular fit)'}
    eps = np.asarray(mf.mo_energy, float)
    curve = fit_outside_curve(eps, 'virtual', nocc - 1, a[nocc - 1],
                              [nocc], {nocc: a[nocc]}, {nocc: z[nocc]},
                              failures=failures)
    assert curve.probes[nocc]['dropped'].startswith('continuation failed '
                                                    '(LinAlgError')


@pytest.mark.parametrize('exc', [RuntimeError('planted defect'),
                                 ValueError('planted defect')])
def test_any_other_exception_propagates(water, monkeypatch, exc):
    mol, mf, nocc, states, factors, xc = water
    _plant(monkeypatch, exc, nocc)
    with pytest.raises(type(exc), match='planted defect'):
        ac_quasiparticle_shifts(mf, mol, nocc, states, factors=factors,
                                xc_diagonal=xc, numerics=Numerics())
