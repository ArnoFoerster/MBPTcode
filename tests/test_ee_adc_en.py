"""Gates for the Epstein-Nesbet amplitude dressing of EE-ADC.

What has to hold:
  * the dressing touches the DOUBLES amplitudes only -- t1^(2) is bit-for-bit
    unchanged, and so is the supermatrix's MP zeroth order;
  * the spin-orbital, spin-free and density-fitted routes agree to machine
    precision under dressing, for every channel combination. That is the real
    check on the spin-resolved denominators (EpsteinNesbet.dressed_doubles_block):
    <pq||pq> = J - delta K is the only rule they encode, and a slip in it would
    move the same-spin and
    opposite-spin configurations differently -- invisible in the spin-orbital
    route (where the spin bookkeeping sits inside g) but not in the spin-free
    one;
  * ADC(1) refuses the option, since it carries no amplitudes to dress;
  * en_dress=True means hole-hole + particle-particle in every ADC.
"""
import numpy as np
import pytest

from pyscf import gto, scf

from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri, get_coulomb_exchange_diagonals)
from src.SingleReference.ADC.eeADC import (ee_utils, ee_en, ee_spin_blocks as psb,
                                     ee_u_sigma_full as pus,
                                     ee_r_sigma as prs, ee_r_sigma_df as pdf)
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from src.SingleReference.ADC.base import (ADCSolverRestricted,
                                          ADCSolverUnrestricted)
from src.SingleReference.EpsteinNesbet import (epstein_nesbet_shift,
                                               validate_en_dress)

ATOMS = ['H 0 0 0; F 0 0 0.917', 'Li 0 0 0; H 0 0 1.6',
         'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587']
DRESS = [{'hh': True, 'pp': True}, {'hh': True}, {'pp': True},
         {'hh': True, 'pp': True, 'hp': True}]


def _system(atom):
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    eps_sp = get_orbital_energies(mf, representation='spatial')
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    return mol, mf, eps_sp, eri


def _exact_B(eri, norb):
    M = eri.reshape(norb * norb, norb * norb)
    w, U = np.linalg.eigh(0.5 * (M + M.T))
    k = np.abs(w) > 1e-12
    return (U[:, k] * np.sqrt(w[k])).T.reshape(-1, norb, norb)


def test_default_channels_are_hh_pp():
    """EE-ADC checks en_dress with the one EpsteinNesbet validator, on its
    own keys: the doubles channels. The IP/EA keys it does not implement are
    refused by name, as is a dict that switches nothing on."""
    keys = ee_en.DRESS_KEYS
    assert validate_en_dress(True, keys=keys) == {'hh': True, 'pp': True}
    assert validate_en_dress(None, keys=keys) is None
    assert validate_en_dress(False, keys=keys) is None
    with pytest.raises(ValueError, match='switches no channel on'):
        validate_en_dress({'hh': False, 'pp': False}, keys=keys)
    for key in ('nonsense', 'singles', 'shift', 'spin_adapted'):
        with pytest.raises(ValueError, match=key):
            validate_en_dress({'hh': True, key: True}, keys=keys)


@pytest.mark.parametrize('nocc', [2, 4])
def test_hhpp_layout_places_the_cross_pairs(nocc):
    """epstein_nesbet_shift in the [i,j,a,b] layout is the [a,b,i,j] one
    transposed, hole-particle cross pairs included (EE-ADC's spin-orbital
    route reads it in that layout)."""
    norb = 10
    rng = np.random.default_rng(3)
    g = rng.normal(size=(norb,) * 4)
    for hp in (False, True):
        pphh = epstein_nesbet_shift(g, nocc, layout='pphh', hh=True, pp=True, hp=hp)
        hhpp = epstein_nesbet_shift(g, nocc, layout='hhpp', hh=True, pp=True, hp=hp)
        assert np.array_equal(hhpp, pphh.transpose(2, 3, 0, 1)), hp


def test_true_means_hh_pp_in_every_adc():
    """en_dress=True dresses the hole-hole and particle-particle channels in
    the charged (IP/EA) solvers exactly as in EE-ADC."""
    hh_pp = {'hh': True, 'pp': True}
    assert validate_en_dress(True) == hh_pp
    assert validate_en_dress(True, keys=ee_en.DRESS_KEYS) == hh_pp
    mol = gto.M(atom=ATOMS[0], basis='sto-3g', verbose=0)
    assert ADCSolverRestricted(scf.RHF(mol).run(), level='adc3',
                               en_dress=True).en_dress == hh_pp
    assert ADCSolverUnrestricted(scf.UHF(mol).run(), level='adc3',
                                 matrix_free=True,
                                 en_dress=True).en_dress == hh_pp


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('dress', DRESS)
def test_only_the_doubles_denominator_is_dressed(atom, dress):
    """Doubles only: the singles DENOMINATOR keeps its MP form in both
    routes. (t1^(2) itself still moves -- A4 builds it out of t2^(1), so a
    dressed t2^(1) propagates into it by construction; what must not change
    is the denominator the dressing is applied through.)"""
    mol, _, eps_sp, eri = _system(atom)
    eps = np.repeat(eps_sp, 2)
    g = get_antisymmetrized_spin_eri(eri)
    nocc, norb = mol.nelectron, len(eps)
    no, nv = nocc // 2, (norb - nocc) // 2

    d_mp, d_ia_mp = ee_en.en_denominators_spin_orbital(eps, g, nocc, None)
    d_en, d_ia_en = ee_en.en_denominators_spin_orbital(eps, g, nocc, dress)
    assert np.abs(d_ia_mp - d_ia_en).max() < 1e-14      # singles untouched
    assert np.abs(d_mp - d_en).max() > 1e-8             # doubles dressed

    J, K = get_coulomb_exchange_diagonals(eri.transpose(0, 2, 1, 3))
    sb_mp = ee_en.en_denominators_spin_free(eps_sp, J, K, no, nv, None)[1]
    sb_en = ee_en.en_denominators_spin_free(eps_sp, J, K, no, nv, dress)[1]
    for key in ('aa', 'bb'):
        assert np.abs(sb_mp.get(key) - sb_en.get(key)).max() < 1e-14


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('dress', DRESS)
def test_spin_free_denominator_matches_spin_orbital(atom, dress):
    """The spin-resolved J - delta K rule against the spin-orbital
    denominator, block by block."""
    mol, _, eps_sp, eri = _system(atom)
    eps = np.repeat(eps_sp, 2)
    g = get_antisymmetrized_spin_eri(eri)
    nocc, norb = mol.nelectron, len(eps)
    no, nv = nocc // 2, (norb - nocc) // 2
    d_so = ee_en.en_denominators_spin_orbital(eps, g, nocc, dress)[0]
    J, K = get_coulomb_exchange_diagonals(eri.transpose(0, 2, 1, 3))
    d_sf = ee_en.en_denominators_spin_free(eps_sp, J, K, no, nv, dress)[0]
    ref = psb.from_spin_orbital(d_so, 'oovv', nocc, norb)
    # ee_en defines the six blocks an antisymmetric doubles tensor actually
    # occupies; the rest of the spin-orbital denominator is never contracted
    for key in ('aaaa', 'bbbb', 'abab', 'baba', 'abba', 'baab'):
        assert np.abs(ref.get(key) - d_sf.get(key)).max() < 1e-12, key


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('dress', DRESS)
@pytest.mark.parametrize('level', ['adc2', 'adc3'])
def test_routes_agree_under_dressing(atom, dress, level):
    mol, _, eps_sp, eri = _system(atom)
    eps = np.repeat(eps_sp, 2)
    g = get_antisymmetrized_spin_eri(eri)
    V = eri.transpose(0, 2, 1, 3)
    nocc, norb = mol.nelectron, len(eps)
    no, nv = nocc // 2, (norb - nocc) // 2
    B = _exact_B(eri, len(eps_sp))

    a_r, _, d = prs.build_operator(eps_sp, V, no, level=level, en_dress=dress)
    a_d, _, _ = pdf.build_operator(eps_sp, B, no, level=level, en_dress=dress)
    a_u, _, _ = pus.build_operator(eps, g, nocc, level=level, en_dress=dress)

    v = np.random.default_rng(11).normal(size=d['nH'])
    y1, Y = prs.to_blocks(v, no, nv, level)
    v_u = np.concatenate([
        psb.to_spin_orbital(y1, 'ov', nocc, norb).ravel(),
        prs.SCALE * ee_utils.fold_doubles(
            psb.to_spin_orbital(Y, 'oovv', nocc, norb), nocc, norb)])
    w_u = a_u(v_u)
    n_su = nocc * (norb - nocc)
    ref = prs.from_blocks(
        psb.from_spin_orbital(w_u[:n_su].reshape(nocc, norb - nocc), 'ov',
                              nocc, norb),
        psb.from_spin_orbital(
            ee_utils.unfold_doubles(w_u[n_su:], nocc, norb) / prs.SCALE,
            'oovv', nocc, norb),
        no, nv, level)
    assert np.abs(a_r(v) - ref).max() < 1e-10          # spin-free == spin-orbital
    assert np.abs(a_r(v) - a_d(v)).max() < 1e-10       # DF == dense integrals


@pytest.mark.parametrize('atom', ATOMS[:2])
def test_dressing_shifts_the_spectrum(atom):
    _, mf, _, _ = _system(atom)
    plain = solve_ee_adc(mf, level='adc3', nroots=2, spin='singlet')[0]
    en = solve_ee_adc(mf, level='adc3', nroots=2, spin='singlet',
                      en_dress=True)[0]
    assert np.abs(np.sort(plain) - np.sort(en)).max() > 1e-4


def test_adc1_refuses_dressing():
    _, mf, _, _ = _system(ATOMS[0])
    with pytest.raises(ValueError):
        solve_ee_adc(mf, level='adc1', nroots=1, en_dress=True)
