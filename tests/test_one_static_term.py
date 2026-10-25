"""One Sigma_x - v_xc for every quasiparticle route.

The static term is `qp_solve.static_exchange_mean_field_matrix`: the mean
field's own J, K and v_xc. Building Sigma_Hx from a fresh non-fitted RHF
(exact J and K) against the fitted v_eff would leave, on a density-fitted or
ISDF-K reference, (J_exact - J_fit) - (K_exact - K_fit)/2 in every
quasiparticle energy and give `dm_correction` a Hartree term the
imaginary-axis routes do not have. Water/cc-pVDZ, density-fitted on
cc-pvdz-ri.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.SingleReference.GW.qp_energy import _static_correction, calc_qp_energy
from src.SingleReference.GW.qp_solve import (static_exchange_diagonal,
                                             static_exchange_mean_field_matrix)
from tests.pins import assert_same

WATER = 'O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692'
AUX = 'cc-pvdz-ri'


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom=WATER, basis='cc-pvdz', verbose=0)


def converged(mf):
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def hf(mol):
    return converged(scf.RHF(mol).density_fit(auxbasis=AUX))


@pytest.fixture(scope='module')
def pbe0(mol):
    return converged(dft.RKS(mol, xc='PBE0').density_fit(auxbasis=AUX))


def correlated_density(mf):
    """A density other than the mean field's: natural occupations moved off
    0/2 by a fixed rotation of one occupied-virtual pair, trace kept."""
    c = mf.mo_coeff
    nocc = mf.mol.nelectron // 2
    occ = mf.mo_occ.astype(float).copy()
    occ[nocc - 1] -= 0.05
    occ[nocc] += 0.05
    return (c * occ) @ c.T


def test_the_casida_route_reads_the_one_builder(mol, pbe0):
    """GATE. The Casida diagonal IS the shared builder's on a fitted hybrid
    reference -- the case where exact J and K against the fitted v_eff moved
    the HOMO by meV. Two calls build J, K and v_xc twice, so to a few ulp
    (`assert_same` of tests/pins.py), not bit for bit."""
    nmo = pbe0.mo_coeff.shape[1]
    casida = _static_correction(pbe0, mol, None, None, 'alpha', False)
    shared = static_exchange_diagonal(pbe0, mol, np.arange(nmo))
    assert_same(casida, shared, quantity='energy', label='diagonal')


def test_the_mean_field_density_is_no_correction(mol, hf):
    """dm_correction = the mean field's own density changes no QP energy on the
    Casida route: J[gamma'] - J[gamma] is then zero through one Coulomb. Exact
    J and K against the fitted v_eff left the fitting error behind instead."""
    homo = mol.nelectron // 2 - 1
    plain = calc_qp_energy(hf, state=homo)
    same = calc_qp_energy(hf, state=homo, dm_correction=hf.make_rdm1())
    assert abs(same - plain) < 1e-9, same - plain


def test_dm_correction_is_hartree_and_exchange(mol, pbe0):
    """One meaning on every route: Sigma_Hx at gamma' against v_Hxc at gamma,
    so the move is diag(J[gamma'] - J[gamma] - (K[gamma'] - K[gamma])/2)."""
    nmo = pbe0.mo_coeff.shape[1]
    states = np.arange(nmo)
    dm, dm_x = pbe0.make_rdm1(), correlated_density(pbe0)
    moved = (static_exchange_diagonal(pbe0, mol, states, dm_correction=dm_x)
             - static_exchange_diagonal(pbe0, mol, states))
    delta = (pbe0.get_j(mol, dm_x) - pbe0.get_j(mol, dm)
             - 0.5 * (pbe0.get_k(mol, dm_x) - pbe0.get_k(mol, dm)))
    expect = np.einsum('mp,mn,np->p', pbe0.mo_coeff, delta, pbe0.mo_coeff)
    assert np.abs(moved - expect).max() < 1e-12
    assert np.abs(expect).max() > 1e-3          # a correction that does something
    casida = (_static_correction(pbe0, mol, dm_x, None, 'alpha', False)
              - _static_correction(pbe0, mol, None, None, 'alpha', False))
    assert np.abs(casida - expect).max() < 1e-12


def test_unrestricted_is_the_restricted_term_per_spin(mol, pbe0):
    """A closed shell solved unrestricted carries the restricted static term in
    each spin, with and without a density correction: Sigma_x^s = -K[gamma^s],
    v_xc^s = v_eff^s - J[gamma^a + gamma^b]."""
    u = dft.UKS(mol, xc='PBE0').density_fit(auxbasis=AUX)
    converged(u)
    r = static_exchange_mean_field_matrix(pbe0, mol)
    m = static_exchange_mean_field_matrix(u, mol)
    assert m.shape == (2,) + r.shape
    # the two SCFs agree to their convergence; the static term follows them
    for s in (0, 1):
        sign = np.sign(np.einsum('mp,mn,np->p', u.mo_coeff[s],
                                 pbe0.get_ovlp(), pbe0.mo_coeff))
        assert np.abs(sign[:, None] * m[s] * sign[None, :] - r).max() < 1e-7
    dm_x = correlated_density(pbe0)
    r_x = static_exchange_mean_field_matrix(pbe0, mol, dm_correction=dm_x)
    m_x = static_exchange_mean_field_matrix(u, mol,
                                            dm_correction=np.stack([dm_x / 2] * 2))
    for s in (0, 1):
        assert np.abs(np.diag(m_x[s]) - np.diag(m[s])
                      - (np.diag(r_x) - np.diag(r))).max() < 1e-7
