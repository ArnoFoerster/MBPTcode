"""The E_0 terms of the downfolded Hamiltonian and the `Energies` record.

With every occupied orbital inactive the environment HF energy is the whole
HF energy of water (measured 1.4e-14, tolerance 1e-9). The per-spin test gives
the two channels different orbitals and different active sets, with a
different number of active occupied orbitals, and compares every term with
the spin-orbital oracle (tests/downfolding_oracle.py); measured agreement at
most 1.4e-14 (tolerance 1e-10). The reference determinant of the assembled
H_eff has the energy E_HF + E_c[v] - E_c^A whatever vtilde, Sigma_bar and the
`dc` mode are (measured 7.1e-14, tolerance 1e-8).
"""
import numpy as np

from src.MultiReference.Downfolding.energy import (
    Energies, active_hf_energy, assemble_energies, environment_hf_energy,
    exchange_energy_term, sigma_trace)
from src.MultiReference.Downfolding.one_body import (
    core_fock, double_counting, effective_one_body)
from src.MultiReference.Downfolding.partition import partition
from src.MultiReference.Downfolding.terms import DownfoldingTerms
from src.SingleReference.common.active_space import ActiveSpace
from tests.downfolding_oracle import (
    mo_integrals, reference_energy, spin_orbital_energies,
    spin_orbital_integrals, water)

TOL = 1e-10


def _water_part(mf, occ_active, virt_active):
    nocc = mf.mol.nelectron // 2
    space = ActiveSpace.from_indices(len(mf.mo_energy), nocc, occ_active,
                                     virt_active)
    return partition(space, mf.mo_energy)


def _random_vtilde(rng, n):
    """(aa, ab, bb) with the symmetries of a real chemist interaction: aa and
    bb 8-fold, ab only within each pair (t<->u, v<->w), so the alpha-beta
    orientation is visible."""
    def pair_sym(x):
        x = x + x.transpose(1, 0, 2, 3)
        return x + x.transpose(0, 1, 3, 2)
    aa, ab, bb = (pair_sym(rng.standard_normal((n,) * 4)) for _ in range(3))
    aa = aa + aa.transpose(2, 3, 0, 1)
    bb = bb + bb.transpose(2, 3, 0, 1)
    return aa, ab, bb


def _orthogonal(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def _symmetric(rng, n):
    x = rng.standard_normal((n, n))
    return x + x.T


def test_environment_energy_all_environment_is_hf():
    mf = water()
    part = _water_part(mf, [], [])
    assert len(part.inactive) == mf.mol.nelectron // 2 and part.n_act == 0
    c = mf.mo_coeff
    e_core, e_ee = environment_hf_energy(mf.mol, mf.get_hcore(), (c, c),
                                         (part, part))
    assert abs(mf.energy_nuc() + e_core + e_ee - mf.e_tot) < 1e-9


def test_per_spin_energy_terms():
    """Alpha and beta with different orbitals (beta rotated within the
    occupied and within the virtual orbitals), different active sets and a
    different number of active occupied orbitals, against the spin-orbital
    oracle."""
    mf = water()
    rng = np.random.default_rng(1)
    nmo, nocc = mf.mo_coeff.shape[1], mf.mol.nelectron // 2
    rot = np.zeros((nmo, nmo))
    rot[:nocc, :nocc] = _orthogonal(rng, nocc)
    rot[nocc:, nocc:] = _orthogonal(rng, nmo - nocc)
    c_a = mf.mo_coeff
    c_b = c_a @ rot
    part_a = _water_part(mf, [3, 4], [5, 6])
    part_b = _water_part(mf, [4], [6, 7, 8])
    n = part_a.n_act
    n_occ = (part_a.n_act_occ, part_b.n_act_occ)
    aa, ab, bb = _random_vtilde(rng, n)
    sigma_bar = (_symmetric(rng, n), _symmetric(rng, n))

    e_core, e_ee = environment_hf_energy(mf.mol, mf.get_hcore(), (c_a, c_b),
                                         (part_a, part_b))
    hartree, exchange = active_hf_energy((aa, ab, bb), n_occ)

    h_so, eri_so = spin_orbital_integrals(mf.mol, c_a, c_b)
    occ_env = list(part_a.inactive) + list(nmo + part_b.inactive)
    occ_act = list(range(n_occ[0])) + list(n + np.arange(n_occ[1]))
    al, be = slice(0, n), slice(n, 2 * n)
    vt_so = np.zeros((2 * n,) * 4)
    vt_so[al, al, al, al] = aa
    vt_so[be, be, be, be] = bb
    vt_so[al, al, be, be] = ab                       # (t_a u_a | v_b w_b)
    vt_so[be, be, al, al] = ab.transpose(2, 3, 0, 1)
    sigma_so = np.zeros((2 * n, 2 * n))
    sigma_so[al, al], sigma_so[be, be] = sigma_bar
    ref = spin_orbital_energies(h_so, eri_so, occ_env, vt_so, occ_act,
                                sigma_so)

    assert abs(e_core - ref['e_core']) < TOL
    assert abs(e_ee - ref['e_ee']) < TOL
    assert abs(hartree - ref['hartree']) < TOL
    assert abs(exchange - ref['exchange']) < TOL
    assert abs(exchange_energy_term((aa, ab, bb), n_occ)
               - ref['exchange_term']) < TOL
    assert abs(sigma_trace(sigma_bar, n_occ) - ref['sigma_trace']) < TOL
    # the alpha-beta block counts: swapping its orientation changes the Hartree term
    swapped = active_hf_energy((aa, ab.transpose(2, 3, 0, 1), bb), n_occ)[0]
    assert abs(swapped - hartree) > 1e-2


def test_e0_assembly_and_dc_modes():
    rng = np.random.default_rng(2)
    n, n_occ = 4, (2, 2)
    vtilde = _random_vtilde(rng, n)
    sigma_bar = (_symmetric(rng, n), _symmetric(rng, n))
    e_nuc, e_core, e_ee = 9.1, -120.3, 37.7
    hartree, exchange = active_hf_energy(vtilde, n_occ)
    k_term = exchange_energy_term(vtilde, n_occ)
    s_term = sigma_trace(sigma_bar, n_occ)
    assert abs(k_term + 2 * exchange) < TOL
    assert abs(k_term) > 1e-2 and abs(s_term) > 1e-2

    energies = {}
    for dc in ('full', 'hartree'):
        terms = DownfoldingTerms(vtilde=vtilde, sigma_bar=sigma_bar,
                                 e_c_full=-0.31, e_c_active=-0.12, dc=dc,
                                 diagnostics={'probe': dc})
        en = assemble_energies(e_nuc, e_core, e_ee, terms, n_occ)
        assert isinstance(en, Energies)
        assert en.diagnostics == {'probe': dc}
        assert (en.e_nuc, en.e_hf_core, en.e_hf_ee) == (e_nuc, e_core, e_ee)
        assert abs(en.e_hf_ee_vtilde - (hartree + exchange)) < TOL
        assert (en.e_c_full, en.e_c_active) == (-0.31, -0.12)
        assert abs(en.e0 - (e_nuc + e_core + e_ee + hartree + exchange - 0.31
                            + 0.12 + en.e_exchange_term
                            - en.e_sigma_trace)) < TOL
        energies[dc] = en
    # dc='full': neither Sigma^E term, whatever vtilde and sigma_bar are
    assert energies['full'].e_exchange_term == 0.0
    assert energies['full'].e_sigma_trace == 0.0
    assert abs(energies['hartree'].e_exchange_term - k_term) < TOL
    assert abs(energies['hartree'].e_sigma_trace - s_term) < TOL
    assert abs(energies['hartree'].e0 - energies['full'].e0
               - (k_term - s_term)) < TOL
    # dc='hartree' without a Sigma_bar keeps the exchange term only
    bare = DownfoldingTerms(vtilde=vtilde, sigma_bar=None, e_c_full=0.0,
                            e_c_active=0.0, dc='hartree')
    en = assemble_energies(e_nuc, e_core, e_ee, bare, n_occ)
    assert en.e_sigma_trace == 0.0
    assert abs(en.e_exchange_term - k_term) < TOL


def test_reference_determinant_energy_is_hf_plus_correlation_pair():
    """The closed-shell reference determinant of H_eff = E_0 + (f - t^DC
    [+ Sigma_bar]) + (v + vtilde) on the active space has energy
    E_HF + E_c[v] - E_c^A for any vtilde: E_HF,ee[vtilde] in E_0 undoes what
    vtilde and its double counting add, and with dc='hartree' the exchange
    term and -Tr[Sigma_bar gamma] undo the missing exchange half and the
    added Sigma_bar."""
    mf = water()
    part = _water_part(mf, [3, 4], [5, 6])
    c = mf.mo_coeff
    n, k = part.n_act, part.n_act_occ
    rng = np.random.default_rng(3)
    vt = _random_vtilde(rng, n)[0]
    vtilde = (vt, vt, vt)
    sig = _symmetric(rng, n)
    _, eri = mo_integrals(mf)
    a = part.act
    v_act = eri[np.ix_(a, a, a, a)]

    e_core, e_ee = environment_hf_energy(mf.mol, mf.get_hcore(), (c, c),
                                         (part, part))
    f = core_fock(mf.mol, mf.get_hcore(), (c, c), (part, part))
    hartree, exchange = double_counting(vtilde, (k, k))
    expected = mf.e_tot - 0.31 + 0.12
    for dc, sigma_bar in (('full', None), ('hartree', None),
                          ('hartree', (sig, sig))):
        terms = DownfoldingTerms(vtilde=vtilde, sigma_bar=sigma_bar,
                                 e_c_full=-0.31, e_c_active=-0.12, dc=dc)
        en = assemble_energies(mf.energy_nuc(), e_core, e_ee, terms, (k, k))
        h_eff = effective_one_body(f, hartree, exchange, dc, sigma_bar)
        g_eff = {0: en.e0, 1: h_eff[0],
                 2: (v_act + vt).transpose(0, 2, 1, 3)}
        assert abs(reference_energy(g_eff, k) - expected) < 1e-8
