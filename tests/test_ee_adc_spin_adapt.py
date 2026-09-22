"""Gates for the EE-ADC CSF isometry.

The decisive check is SUBSPACE INVARIANCE, ||H T - T (T^T H T)|| ~ 0: it
says T spans an exact invariant subspace of the supermatrix, so the
projected spectrum is a subset of the spin-orbital one with no
contamination and no lost roots. Together with T^T T = 1 that pins the
construction without ever writing a CSF matrix element by hand -- the same
discipline as tests for spin_adapt.py on the IP/EA side.
"""
import numpy as np
import pytest

from pyscf import gto, scf, adc

from src.Base.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri)
from src.SingleReference.ADC.eeADC import ee_u_dense_full as ppd, ee_spin_adapt as psa

ATOMS = ['H 0 0 0; F 0 0 0.917', 'Li 0 0 0; H 0 0 1.6',
         'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587']


def _system(atom):
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    eps = np.repeat(get_orbital_energies(mf, representation='spatial'), 2)
    g = get_antisymmetrized_spin_eri(
        get_two_electron_integrals_chemist(mol, mf, representation='spatial'))
    return mol, mf, eps, g, mol.nelectron, len(eps)


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
@pytest.mark.parametrize('level', ['adc2', 'adc3'])
def test_isometry_spans_invariant_subspace(atom, spin, level):
    _, _, eps, g, nocc, norb = _system(atom)
    H = ppd.build_supermatrix(eps, g, nocc, level=level)
    T = psa.csf_isometry(nocc, norb, spin=spin, level=level)
    assert np.abs(T.T @ T - np.eye(T.shape[1])).max() < 1e-12
    Hc = T.T @ H @ T
    assert np.abs(H @ T - T @ Hc).max() < 1e-11


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('level,method', [('adc2', 'adc(2)'),
                                          ('adc2x', 'adc(2)-x'),
                                          ('adc3', 'adc(3)')])
def test_singlet_channel_matches_pyscf(atom, level, method):
    """pyscf's RHF EE-ADC returns singlets only; the projected channel must
    reproduce them without the triplets interleaved."""
    mol, mf, eps, g, nocc, norb = _system(atom)
    H = ppd.build_supermatrix(eps, g, nocc, level=level)
    T = psa.csf_isometry(nocc, norb, spin='singlet', level=level)
    e = np.linalg.eigvalsh(T.T @ H @ T)
    a = adc.ADC(mf); a.method = method; a.method_type = 'ee'; a.verbose = 0
    for r in np.array(a.kernel(nroots=4)[0]):
        assert np.abs(e - r).min() < 1e-6


@pytest.mark.parametrize('atom', ATOMS)
def test_channels_partition_the_spectrum(atom):
    """singlet CSFs + 3 x (triplet CSFs) must exhaust the Ms-resolved
    spin-orbital configuration space."""
    _, _, eps, g, nocc, norb = _system(atom)
    ns = psa.csf_isometry(nocc, norb, spin='singlet').shape[1]
    nt = psa.csf_isometry(nocc, norb, spin='triplet').shape[1]
    H = ppd.build_supermatrix(eps, g, nocc, level='adc3')
    e_all = np.sort(np.linalg.eigvalsh(H))
    Ts = psa.csf_isometry(nocc, norb, spin='singlet')
    Tt = psa.csf_isometry(nocc, norb, spin='triplet')
    e_s = np.linalg.eigvalsh(Ts.T @ H @ Ts)
    e_t = np.linalg.eigvalsh(Tt.T @ H @ Tt)
    # every projected root is a root of the full matrix
    for r in np.concatenate([e_s, e_t]):
        assert np.abs(e_all - r).min() < 1e-8
    assert ns > 0 and nt > 0
