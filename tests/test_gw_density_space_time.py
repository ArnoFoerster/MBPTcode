"""The GW density on the space-time route (mode='space-time', N^3).

- `correlation_self_energy_matrix_space_time`: the full MO Sigma_c,pq(mu + i w)
  of the space-time QP route on any query frequencies; its diagonal at the
  Pade nodes is the QP route's `_sigma_mo_diagonal` on the same factors.
- No ISDF grid (a bare Numerics()): the exception `separable_factors` raises.

Water/cc-pVDZ, HF, DF on cc-pvdz-ri (the factors' auxiliary basis too);
default quadratures (W grid minimax 20, outer minimax 34, the space-time
route's 'auto' time grid).

The density itself, space-time against casida per block of Delta gamma_MO
and on the natural occupations, per relaxation, is in test_gw_density_none,
test_gw_density_dyson and test_gw_density_zvector.

Recorded when that comparison lived here (space-time against imagfrequency,
water): Sigma_c,pq(mu + i w) on the outer nodes against the imagfrequency
route's: max |diff| 2-3e-6 (max |Sigma| 8e-2) on both G1 and GH, the finest
ISDF level for cc-pVDZ. GH is no closer than G1, so the floor is not the
ISDF grid's density; the time grid does not move it either (HF 'dyson' on GH unchanged to 3 digits from 16 to 26 tau points,
tau_target 1e-8 .. 1e-14).
"""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.factorization.sliced_factors import unpack_factors
from src.Base.numerics.numerics import Numerics
from src.Base.orbital_spaces import particle_hole_window
from src.Base.pyscf.pyscf_interface import get_orbital_energies
from src.Base.utils.grids.analytic_continuation import pade_nodes
from src.SingleReference.GW import space_time
from src.SingleReference.GW.density_matrix import compute_gw_density_matrix
from src.SingleReference.GW.space_time import (
    correlation_self_energy_matrix_space_time, separable_factors)
from src.SingleReference.LinearResponse.space_time import chi0_imaginary_frequency
from tests.isdf_test_grid import g1_numerics
from tests.gw_density_routes import AUXBASIS, fitted, water


@pytest.fixture(scope='module')
def hf():
    return fitted(scf.RHF(water()))


def test_matrix_diagonal_matches_qp(hf):
    """The diagonal of the full Sigma_c,pq at the Pade nodes is the QP
    route's `_sigma_mo_diagonal` for every state on the same factors."""
    mf = hf
    mol = mf.mol
    nocc = mol.nelectron // 2
    numerics = g1_numerics(mf)
    pade = pade_nodes(numerics.quadrature)
    sigma, mu = correlation_self_energy_matrix_space_time(mf, mol, nocc, pade, numerics)

    eps = get_orbital_energies(mf, representation='spatial')
    q = numerics.quadrature
    axes = space_time._space_time_grids(
        [eps], [nocc], [eps], dict(ntau=q.ntau_st, nfreq=q.nfreq_st,
                                   pade_freq=pade, tau_target=q.tau_target),
        q.w0_gl, None)
    X_mo, D, X_ao, coords = unpack_factors(separable_factors(mf, mol, numerics=numerics))
    W = chi0_imaginary_frequency(X_mo, D, eps, nocc, axes['grid'], mu=mu)
    space_time._dyson_in_place(W, range(W.shape[0]))
    states = np.arange(len(eps))
    qp = space_time._sigma_mo_diagonal(
        X_mo, D, W, mf, eps, nocc, axes['tau_points'], axes['freq_points'],
        pade, mu, states, X_ao=X_ao, coords=coords)
    diag = np.einsum('wpp->pw', sigma)
    err = np.abs(diag - qp).max()
    print(f'max |diag Sigma_c,pq - _sigma_mo_diagonal| {err:.2e}, '
          f'max |Sigma| {np.abs(qp).max():.2e}')
    assert sigma.shape == (len(pade), len(eps), len(eps))
    assert mu == particle_hole_window(eps, nocc)[2]
    assert err < 1e-10


def test_space_time_needs_an_isdf_grid(hf):
    """A Numerics() naming no ISDF grid: the exception separable_factors
    raises, with its message."""
    with pytest.raises(Exception) as expected:
        separable_factors(hf, hf.mol, numerics=Numerics())
    with pytest.raises(Exception) as got:
        compute_gw_density_matrix(hf, mode='space-time', relaxation='none',
                                  numerics=Numerics())
    print(f'{type(got.value).__name__}: {got.value}')
    assert type(got.value) is type(expected.value)
    assert str(got.value) == str(expected.value)


def test_open_shell_refused():
    """An ROHF reference (OH radical): mode='space-time' raises the named
    closed-shell error mode='imagfrequency' raises."""
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=0)
    rohf = scf.ROHF(mol).density_fit(auxbasis=AUXBASIS).run()
    with pytest.raises(NotImplementedError) as imag:
        compute_gw_density_matrix(rohf, mode='imagfrequency', relaxation='none',
                                  numerics=Numerics())
    with pytest.raises(Exception) as st:
        compute_gw_density_matrix(rohf, mode='space-time', relaxation='none',
                                  numerics=g1_numerics(rohf))
    print(f'{type(st.value).__name__}: {st.value}')
    assert type(st.value) is type(imag.value)
    assert 'closed-shell' in str(st.value)


def test_smeared_refused(hf):
    """The matrix Sigma itself refuses a thermally smeared RHF, before any
    ISDF work (a bare Numerics() has no grid, so reaching it would raise the
    other error)."""
    smeared = copy.copy(hf)
    smeared.sigma, smeared.smearing_method = 0.01, 'fermi'
    with pytest.raises(NotImplementedError, match='T = 0'):
        correlation_self_energy_matrix_space_time(
            smeared, smeared.mol, 5, np.array([0.5]), Numerics())


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-v']))
