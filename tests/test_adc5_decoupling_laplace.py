"""Matrix-free diagrammatic ADC(5) decoupling (sigma_driver._diagrammatic_adc5_rows
with decoupling_ntau): the decoupling amplitudes S = X / (K_C - K_D) applied by the
minimax Laplace quadrature of the denominator, so no D x C block is formed.

Gates on linear H4 (r = 1.0) and LiH in sto-3g (the systems of
test_adc5_union_poles):
  1. the adjoint orbital<->k2 folds are the transposes of the forward ones:
     orb_dot_s1_k2 (the first-order k2 overlap the effective k1-k2 blocks read)
     of s1_k2_dot_orb, orb_dot_u2_k2 of u2_k2_dot_orb;
  2. the coupling rows from the quadrature converge to the exact (dense) rows:
     at the widest tabulated grid they agree to TOL_ROWS relative, and the
     smallest grid is no closer;
  3. the operator with the quadrature rows has the diagrammatic ADC(5) poles of
     the union-space oracle (tests/union_space_oracle.py).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils.grids.grids import minimax_tau_supported_sizes
from src.SingleReference.ADC.generated_modules import lazy
from src.SingleReference.ADC.sigma_driver import (
    _diagrammatic_adc5_rows, orb_dot_s1_k2, orb_dot_u2_k2, restricted_dyson_operator,
    s1_k2_dot_orb, u2_k2_dot_orb)
from tests.generated_runtime import requires_runtime
from tests.test_adc5_archive_pins import H4, LIH, _solver
from tests.test_adc5_poles import _production
from tests import union_space_oracle as CB
from tests.test_adc5_union_poles import (SYSTEMS, TOL, TOL_SYM, _dense, _diff, _oracle_poles,
                                         _principal)

TOL_ADJ = 1e-13
TOL_ROWS = 1e-10
NTAU_MIN = min(minimax_tau_supported_sizes())
NTAU_MAX = max(minimax_tau_supported_sizes())

pytestmark = requires_runtime(5)

c1r_module = lazy('adc4_c1_self_apply_restricted_blocks')
k2r_module = lazy('adc4_k2_apply_restricted_blocks')
u2r_module = lazy('adc4_u2_k2_apply_restricted_blocks')


def _columns(apply, n_in, n_out):
    """(n_out, n_in) matrix of a linear map, one unit vector at a time."""
    return np.column_stack([apply(e) for e in np.eye(n_in)]).reshape(n_out, n_in)


def _system(name):
    sysd = CB.build_system(name)
    r, f_orb = _production(sysd)
    return sysd, r, f_orb


# the folds and the rows need only the solver; the poles test reads the oracle
SOLVERS = {'H4lin_r1.0': H4, 'LiH': LIH}


def _rows(r, ntau):
    return _diagrammatic_adc5_rows(r, c1r_module, k2r_module, u2r_module, decoupling_ntau=ntau)


FOLDS = {'S1_k2': (s1_k2_dot_orb, orb_dot_s1_k2), 'U2_k2': (u2_k2_dot_orb, orb_dot_u2_k2)}


@pytest.mark.parametrize('name', SYSTEMS)
@pytest.mark.parametrize('sector', ['IP', 'EA'])
@pytest.mark.parametrize('fold', sorted(FOLDS))
def test_k2_orbital_fold_adjoint(name, sector, fold):
    r = _solver(SOLVERS[name])
    forward, adjoint = FOLDS[fold]
    n2 = len(r.kconfigs(sector, 2)['K'])
    fwd = _columns(lambda p: forward(r, sector, p, u2r_module), r.norb, n2)
    adj = _columns(lambda v: adjoint(r, sector, v, u2r_module), n2, r.norb)
    scale = float(np.max(np.abs(fwd)))
    d = float(np.max(np.abs(fwd - adj.T)))
    print(f"{name} {sector} {fold}: |block| {scale:.3e}, |forward - adjoint^T| {d:.1e}")
    assert scale > 0.0
    assert d <= TOL_ADJ * scale, d


@pytest.mark.parametrize('name', SYSTEMS)
def test_rows_converge_to_exact(name):
    r = _solver(SOLVERS[name])
    exact = _rows(r, None)
    scale = max(float(np.max(np.abs(exact[k]))) for k in exact)
    err = {}
    for ntau in (NTAU_MIN, NTAU_MAX):
        lap = _rows(r, ntau)
        err[ntau] = max(float(np.max(np.abs(lap[k] - exact[k]))) for k in exact) / scale
    print(f"{name}: rows {scale:.3e}, relative error ntau={NTAU_MIN} {err[NTAU_MIN]:.1e}, "
          f"ntau={NTAU_MAX} {err[NTAU_MAX]:.1e}")
    assert scale > 0.0
    assert err[NTAU_MAX] <= TOL_ROWS, err
    assert err[NTAU_MIN] >= err[NTAU_MAX], err


@pytest.mark.parametrize('name', SYSTEMS)
def test_laplace_poles_match_oracle(name):
    sysd, r, f_orb = _system(name)
    us = CB.UnionSpace(sysd)
    ref = _oracle_poles(us, 'diagrammatic')
    M = _dense(restricted_dyson_operator(r, 'adc5', f_orb=f_orb, construction='diagrammatic',
                                         decoupling_ntau=NTAU_MAX)[0])
    asym = float(np.max(np.abs(M - M.T)))
    mine = _principal(M, len(f_orb))
    d = _diff(mine, ref)
    print(f"{name} diagrammatic ADC(5), Laplace decoupling ntau={NTAU_MAX}: "
          f"|production - oracle| {d:.2e}; symmetry {asym:.1e}")
    assert asym <= TOL_SYM, asym
    assert d <= TOL, d


def main():
    for name in SYSTEMS:
        for sector in ('IP', 'EA'):
            for fold in sorted(FOLDS):
                test_k2_orbital_fold_adjoint(name, sector, fold)
        test_rows_converge_to_exact(name)
        test_laplace_poles_match_oracle(name)
    print('\nALL OK')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
