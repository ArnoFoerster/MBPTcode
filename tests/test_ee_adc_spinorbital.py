"""Gates for the spin-orbital polarization-propagator (EE) ADC route.

Four independent arbiters, in increasing strength:

  1. ADC(1) == CIS (singlets AND triplets) -- pins the ph/ph block at first
     order and the spin-orbital layout.
  2. The first-order ph/2p2h coupling == the exact Slater-Condon
     <Phi_ij^ab|H|Phi_k^c> -- pins the doubles metric factor
     (ee_utils.PAPER_DOUBLES_SCALE); an error there is silent, since the
     supermatrix stays symmetric and only the coupling strength moves.
  3. ADC(2)/ADC(2)-x/ADC(3) == pyscf's EE-ADC.
  4. ADC(3) excitation energies are exact through lambda^3 against a
     determinant-space expansion of H(lambda) = F + lambda(H - F). This is
     the gate that caught the Z^(A) sign (see z_intermediates): pyscf
     agreement alone would not have, because pyscf's precomputed M_ carries
     only part of the ph/ph block.
"""
import itertools

import numpy as np
import pytest

from pyscf import gto, scf, adc, tdscf

from src.Base.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri)
from src.SingleReference.ADC.eeADC import ee_u_dense_full as ppd, ee_utils


def _system(atom, basis='sto-3g'):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    eps = np.repeat(get_orbital_energies(mf, representation='spatial'), 2)
    g = get_antisymmetrized_spin_eri(
        get_two_electron_integrals_chemist(mol, mf, representation='spatial'))
    return mol, mf, eps, g, mol.nelectron, len(eps)


SMALL = ['H 0 0 0; F 0 0 0.917', 'Li 0 0 0; H 0 0 1.6', 'B 0 0 0; H 0 0 1.23']
MED = SMALL + ['O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
               'Be 0 0 0; H 0 0 1.34; H 0 0 -1.34']


# ---------------------------------------------------------------- 1. CIS
@pytest.mark.parametrize('atom', MED)
def test_adc1_is_cis(atom):
    mol, mf, eps, g, nocc, norb = _system(atom)
    e = np.linalg.eigvalsh(ppd.build_supermatrix(eps, g, nocc, level='adc1'))
    ref = []
    for singlet in (True, False):
        td = tdscf.TDA(mf); td.nstates = 4; td.singlet = singlet; td.kernel()
        ref.extend(td.e)
    for r in ref:
        assert np.abs(e - r).min() < 1e-9


@pytest.mark.parametrize('level', ['adc1', 'adc2', 'adc2x', 'adc3'])
def test_supermatrix_is_hermitian(level):
    _, _, eps, g, nocc, _ = _system(MED[3])
    H = ppd.build_supermatrix(eps, g, nocc, level=level)
    assert np.abs(H - H.T).max() < 1e-12


# ------------------------------------------- 2. exact first-order coupling
def _sc_hmat(d1, d2, h1, g):
    """<D1|H|D2> by Slater-Condon; d1/d2 are sorted occupation tuples."""
    s1, s2 = set(d1), set(d2)
    df1, df2 = sorted(s1 - s2), sorted(s2 - s1)
    if len(df1) > 2:
        return 0.0
    common = sorted(s1 & s2)
    ph = _parity(df1 + common) * _parity(df2 + common)
    if not df1:
        return (sum(h1[p, p] for p in d1)
                + 0.5 * sum(g[p, q, p, q] for p in d1 for q in d1))
    if len(df1) == 1:
        p, q = df1[0], df2[0]
        return ph * (h1[p, q] + sum(g[p, m, q, m] for m in common))
    return ph * g[df1[0], df1[1], df2[0], df2[1]]


def _parity(seq):
    seq, sign = list(seq), 1
    for i in range(len(seq)):
        for j in range(len(seq) - 1):
            if seq[j] > seq[j + 1]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
                sign = -sign
    return sign


def _config_det(nocc, holes, parts):
    """Phase of C_I|Phi0> for C_I = c+_a c+_b c_i c_j (paper Eq. 2): the
    annihilators act right-to-left, then each creator is pushed onto the
    front of the occupation list."""
    L, sign = list(range(nocc)), 1
    for h in reversed(holes):
        pos = L.index(h); sign *= (-1) ** pos; L.pop(pos)
    for p in reversed(parts):
        L = [p] + L
    return tuple(sorted(L)), sign * _parity(L)


@pytest.mark.parametrize('atom', SMALL)
def test_first_order_coupling_is_exact(atom):
    """The A54 block times PAPER_DOUBLES_SCALE is the bare Hamiltonian
    element between the doubly and singly excited determinants."""
    _, _, eps, g, nocc, norb = _system(atom)
    h1 = np.diag(eps) - np.einsum('pmqm->pq', g[:, :nocc, :, :nocc])
    I, J, A, B = ee_utils.configs_doubles(nocc, norb)
    K, C = ee_utils.configs_singles(nocc, norb)
    dS = [_config_det(nocc, (k,), (c,)) for k, c in zip(K, C)]
    exact = np.zeros((len(I), len(K)))
    for m, (i, j, a, b) in enumerate(zip(I, J, A, B)):
        d1, s1 = _config_det(nocc, (i, j), (a, b))
        for n, (d2, s2) in enumerate(dS):
            exact[m, n] = s1 * s2 * _sc_hmat(d1, d2, h1, g)
    assert np.abs(exact - ppd.m_ds(g, nocc, norb, 1)).max() < 1e-12


# --------------------------------------------------------- 3. pyscf EE-ADC
@pytest.mark.parametrize('atom', MED)
@pytest.mark.parametrize('level,method', [('adc2', 'adc(2)'),
                                          ('adc2x', 'adc(2)-x'),
                                          ('adc3', 'adc(3)')])
def test_matches_pyscf(atom, level, method):
    mol, mf, eps, g, nocc, _ = _system(atom)
    e = np.linalg.eigvalsh(ppd.build_supermatrix(eps, g, nocc, level=level))
    a = adc.ADC(mf); a.method = method; a.method_type = 'ee'; a.verbose = 0
    for r in np.array(a.kernel(nroots=4)[0]):
        assert np.abs(e - r).min() < 1e-6      # pyscf Davidson tolerance


# ------------------------------------------------ 4. exact lambda expansion
def _lambda_oracle(eps, g, nocc, norb, lam):
    """Lowest excitation energies of H(lam) = F + lam(H - F) in the full
    determinant space."""
    h1 = np.diag(eps) - np.einsum('pmqm->pq', g[:, :nocc, :, :nocc])
    dets = [tuple(sorted(c)) for c in itertools.combinations(range(norb), nocc)]
    H = np.array([[_sc_hmat(d1, d2, h1, g) for d2 in dets] for d1 in dets])
    F = np.diag([sum(eps[p] for p in d) for d in dets])
    e = np.sort(np.linalg.eigvalsh(F + lam * (H - F)))
    return e[1:] - e[0]


@pytest.mark.parametrize('atom', ['H 0 0 0; F 0 0 0.917', 'Li 0 0 0; H 0 0 1.6'])
def test_third_order_lambda_exact(atom):
    """ADC(3) excitation energies must be correct through lambda^3.

    This is what pins the Z^(A) sign: with the A19 form as printed the
    residual here is ~1e-2/lambda^3 instead of ~1e-5."""
    _, _, eps, g0, nocc, norb = _system(atom)
    lam, nst = 0.005, 6
    w_ex = _lambda_oracle(eps, g0, nocc, norb, lam)[:nst]
    H = ppd.build_supermatrix(eps, lam * g0, nocc, level='adc3')
    w = np.sort(np.linalg.eigvalsh(H))[:nst]
    assert np.abs(w - w_ex).max() / lam ** 3 < 1e-3


@pytest.mark.parametrize('atom', ['H 0 0 0; F 0 0 0.917'])
def test_mp_density_matches_exact_lambda2(atom):
    """rho^(2) (A21-A23), including the factor-2 singles denominator of A4,
    against the exact lambda^2 density coefficient."""
    _, _, eps, g0, nocc, norb = _system(atom)
    h1 = np.diag(eps) - np.einsum('pmqm->pq', g0[:, :nocc, :, :nocc])
    dets = [tuple(sorted(c)) for c in itertools.combinations(range(norb), nocc)]
    ref = dets.index(tuple(range(nocc)))
    H = np.array([[_sc_hmat(d1, d2, h1, g0) for d2 in dets] for d1 in dets])
    F = np.diag([sum(eps[p] for p in d) for d in dets])
    lams = np.linspace(-0.2, 0.2, 21)
    R = []
    for l in lams:
        w, v = np.linalg.eigh(F + l * (H - F))
        k = int(np.argmax(np.abs(v[ref]))); vec = v[:, k]
        vec = vec if vec[ref] > 0 else -vec
        rho = np.zeros((norb, norb))
        for n1, d1 in enumerate(dets):
            for n2, d2 in enumerate(dets):
                df1, df2 = sorted(set(d1) - set(d2)), sorted(set(d2) - set(d1))
                if len(df1) > 1:
                    continue
                c = vec[n1] * vec[n2]
                if not df1:
                    for p in d1:
                        rho[p, p] += c
                else:
                    com = sorted(set(d1) & set(d2))
                    rho[df1[0], df2[0]] += (_parity([df1[0]] + com)
                                            * _parity([df2[0]] + com) * c)
        R.append(rho)
    c2 = np.polyfit(lams, np.array(R).reshape(len(lams), -1), 6)[::-1][2]
    c2 = c2.reshape(norb, norb)
    amps = ee_utils.mp_amplitudes(eps, g0, nocc, norb, order=3)
    rho = ee_utils.mp_density2(amps)
    assert np.abs(c2[:nocc, :nocc] - rho['oo']).max() < 1e-7
    assert np.abs(c2[:nocc, nocc:] - rho['ov']).max() < 1e-7
    assert np.abs(c2[nocc:, nocc:] - rho['vv']).max() < 1e-7
