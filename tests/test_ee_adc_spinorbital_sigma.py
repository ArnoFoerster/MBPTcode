"""The matrix-free EE-ADC operator must reproduce the dense supermatrix.

This is a genuine cross-check, not a tautology: ee_u_sigma_full transcribes
the paper's ph-ROW equations (A41/A42 for the doubles contribution to the
singles sigma) while ee_u_dense_full builds the same coupling from the
2p2h-ROW equations (A54/A55) and transposes. Agreement therefore also
verifies the doubles metric convention (ee_utils.PAPER_DOUBLES_SCALE) on
both sides independently.
"""
import numpy as np
import pytest

from pyscf import gto, scf

from src.Base.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri)
from src.SingleReference.ADC.eeADC import (ee_u_dense_full as ppd,
                                     ee_u_sigma_full as pps, ee_utils)

ATOMS = ['H 0 0 0; F 0 0 0.917', 'Li 0 0 0; H 0 0 1.6',
         'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587']


def _system(atom):
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    eps = np.repeat(get_orbital_energies(mf, representation='spatial'), 2)
    g = get_antisymmetrized_spin_eri(
        get_two_electron_integrals_chemist(mol, mf, representation='spatial'))
    return eps, g, mol.nelectron


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('level', ['adc1', 'adc2', 'adc2x', 'adc3'])
def test_sigma_equals_dense(atom, level):
    eps, g, nocc = _system(atom)
    H = ppd.build_supermatrix(eps, g, nocc, level=level)
    aop, diag, _ = pps.build_operator(eps, g, nocc, level=level)
    n = H.shape[0]
    S = np.column_stack([aop(np.eye(n)[:, k]) for k in range(n)])
    assert np.abs(S - H).max() < 1e-11


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('level', ['adc2', 'adc2x', 'adc3'])
def test_analytic_diagonal(atom, level):
    """The Davidson preconditioner diagonal must be the true diagonal."""
    eps, g, nocc = _system(atom)
    H = ppd.build_supermatrix(eps, g, nocc, level=level)
    _, diag, _ = pps.build_operator(eps, g, nocc, level=level)
    assert np.abs(diag - np.diag(H)).max() < 1e-11


@pytest.mark.parametrize('atom', ATOMS)
def test_doubles_fold_roundtrip(atom):
    eps, g, nocc = _system(atom)
    norb = len(eps)
    y = np.random.default_rng(0).normal(size=ee_utils.dimensions(nocc, norb)['n_d'])
    T = ee_utils.unfold_doubles(y, nocc, norb)
    assert np.abs(T + T.transpose(1, 0, 2, 3)).max() < 1e-14
    assert np.abs(T + T.transpose(0, 1, 3, 2)).max() < 1e-14
    assert np.abs(ee_utils.fold_doubles(T, nocc, norb) - y).max() < 1e-14
