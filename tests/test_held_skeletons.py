"""One pass per linear skeleton channel of a force, whichever terms feed it.

The orbital response's skeleton (`fock_partial_skeleton` of gamma + Lambda)
and the static correction's (`qp_xc_correction_skeleton`) each carry a
four-centre long-range exchange channel and the exchange-correlation grid
skeleton. Both are linear in their density at the SCF density, so inside the
force's fit-adjoint window the two terms hold their densities there
(`PendingFitAdjoint.hold_skeleton`) and the assembly makes one pass per
channel on the sums (`mean_field_response.held_skeletons`).

Gated here on a density-fitted LRC-wPBEh water:
  * the held route is the two skeletons evaluated apart, to the sums'
    re-association;
  * it makes one four-centre pass per attenuation and one grid pass, where
    the separate route makes two of each;
  * a window left with held channels raises, and one whose force was
    discarded (a relaxed density) does not.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto

from tests.pins import assert_within_class
from src.gradients import mean_field_response
from src.gradients.gwbse.static_correction import qp_xc_correction_skeleton
from src.gradients.isdf_derivatives import one_fit_adjoint
from src.gradients.mean_field_response import (fock_partial_skeleton,
                                               held_skeletons)

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


@pytest.fixture(scope='module')
def mf():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    m = dft.RKS(mol, xc='lrc-wpbeh').density_fit()
    m.conv_tol = 1e-11
    m.kernel()
    return m


def partials(mf, seed=5):
    """(gamma, static weights): symmetric MO matrices with every block."""
    rng = np.random.default_rng(seed)
    n = mf.mo_coeff.shape[1]
    a, b = rng.standard_normal((2, n, n))
    return a + a.T, b + b.T


def counted(monkeypatch):
    """Count the four-centre and grid passes the module makes."""
    calls = {'exchange': 0, 'xc': 0}
    by_atoms = mean_field_response.exchange_skeleton_by_atoms
    grid = mean_field_response.xc_grid_skeleton

    def exchange(*args, **kwargs):
        calls['exchange'] += 1
        return by_atoms(*args, **kwargs)

    def xc(*args, **kwargs):
        calls['xc'] += 1
        return grid(*args, **kwargs)

    monkeypatch.setattr(mean_field_response, 'exchange_skeleton_by_atoms',
                        exchange)
    monkeypatch.setattr(mean_field_response, 'xc_grid_skeleton', xc)
    return calls


def test_held_route_is_the_separate_sum_in_one_pass_each(mf, monkeypatch):
    nocc = mf.mol.nelectron // 2
    gamma, static = partials(mf)
    calls = counted(monkeypatch)
    apart = (fock_partial_skeleton(mf, gamma, nocc)
             + qp_xc_correction_skeleton(mf, static, nocc))
    assert calls == {'exchange': 2, 'xc': 2}
    calls.update(exchange=0, xc=0)
    with one_fit_adjoint(mf) as pending:
        held = (qp_xc_correction_skeleton(mf, static, nocc, pending=pending)
                + fock_partial_skeleton(mf, gamma, nocc, pending=pending))
        assert calls == {'exchange': 0, 'xc': 0}
        held = held + held_skeletons(mf, pending)
    assert calls == {'exchange': 1, 'xc': 1}
    assert_within_class(held, apart, 'gradient', label='held skeletons')


def test_unevaluated_channels_raise_and_discard_drops_them(mf):
    nocc = mf.mol.nelectron // 2
    _, static = partials(mf)
    with pytest.raises(RuntimeError, match='no nuclear assembly evaluated'):
        with one_fit_adjoint(mf) as pending:
            qp_xc_correction_skeleton(mf, static, nocc, pending=pending)
    with one_fit_adjoint(mf) as pending:
        qp_xc_correction_skeleton(mf, static, nocc, pending=pending)
        pending.discard()
