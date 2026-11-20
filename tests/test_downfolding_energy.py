"""The E_0 terms of the downfolded Hamiltonian and the `Energies` record.

With every occupied orbital inactive the environment HF energy is the whole
HF energy of water (measured 1.4e-14, tolerance 1e-9). The per-spin test gives
the two channels different orbitals and different active sets, with a
different number of active occupied orbitals, and compares every term with
the spin-orbital oracle (tests/downfolding_oracle.py); measured agreement at
most 1.4e-14 (tolerance 1e-10). The reference determinant of the assembled
H_eff has the energy E_HF + E_c[v] - E_c^A whatever vtilde, Sigma_bar and the
`dc` mode are (measured 7.1e-14, tolerance 1e-8).

The driver with method='hf' gives the CASCI Hamiltonian (tolerance 1e-8):
water/6-31G (4,4) on RHF, PBE and PBE0 orbitals (measured 0, 0, 1.4e-14),
non-contiguous (0), on the orbitals of a density-fitted RHF against an
exact-integral CASCI (0; a DF-CASCI differs by 4.1e-6), and packed in spin
orbitals (4.3e-14). Its reference determinant is the RHF energy (5.7e-14), and
with every H2/6-31G orbital active it is FCI (4.4e-16, tolerance 1e-9).

The driver with method='rpa' (the default) on casida, for bare, crpa and
crpa_ph: the reference determinant is E_HF + E_c[v] - E_c^A[v_eff]; with
every orbital active the E_c pair cancels and H_eff gives FCI, the same on
PBE and HF orbitals; with no orbital active E_0 = E_HF + E_c[v]; 'bare' is
CASCI shifted by the E_c pair; crpa_ph leaves the active HF energy alone;
the crpa static defect is zero (also on a density-fitted mean field). The
diagnostics keyword, the option refusals and one low-level result reused
for many active spaces and screenings are tested too. Measured values in
each test's docstring.

With sigma='srg' or 'mode_a' (Sigma^E, Hartree-only double counting, the
exchange energy term and -Tr[Sigma_bar gamma^A,0] in E_0): the reference
determinant keeps its energy with the real Sigma_bar and with Sigma_bar
forced to 0, for every screening; crpa_ph has no exchange term and h_eff =
f + Sigma_bar; every orbital active still gives FCI and no orbital active
E_HF + E_c[v]; at s = inf both forms give the same E_0; method='hf' refuses
sigma.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import dft, fci, gto, mcscf, scf

from src.Base.numerics.numerics import Numerics
from src.MultiReference.Downfolding import (build_effective_hamiltonian,
                                            effective_hamiltonian,
                                            run_low_level)
from src.MultiReference.Downfolding.energy import (
    Energies, active_hf_energy, assemble_energies, environment_hf_energy,
    exchange_energy_term, sigma_trace)
from src.MultiReference.Downfolding.methods import LOW_LEVEL, rpa
from src.MultiReference.Downfolding.one_body import (
    core_fock, double_counting, effective_one_body)
from src.MultiReference.Downfolding.partition import partition
from src.MultiReference.Downfolding.terms import DownfoldingTerms
from src.SingleReference.common.active_space import ActiveSpace
from src.Solvers.active_space import solve
from tests.downfolding_oracle import (
    h2, mo_integrals, reference_energy, spin_orbital_energies,
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
    e_core, e_h, e_x = environment_hf_energy(mf.mol, mf.get_hcore(), (c, c),
                                             (part, part))
    assert abs(mf.energy_nuc() + e_core + e_h + e_x - mf.e_tot) < 1e-9
    # the Hartree and exchange halves, against pyscf's own J and K
    dm = mf.make_rdm1()
    vj, vk = mf.get_jk(mf.mol, dm)
    assert abs(e_h - 0.5 * np.einsum('ij,ji->', vj, dm)) < 1e-9
    assert abs(e_x + 0.25 * np.einsum('ij,ji->', vk, dm)) < 1e-9
    assert e_h > 0 > e_x


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

    e_core, e_h, e_x = environment_hf_energy(
        mf.mol, mf.get_hcore(), (c_a, c_b), (part_a, part_b))
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
    assert abs(e_h - ref['e_hartree']) < TOL
    assert abs(e_x - ref['e_exchange']) < TOL
    assert abs(e_h + e_x - ref['e_ee']) < TOL
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
    e_nuc, e_core, e_h, e_x = 9.1, -120.3, 60.2, -22.5
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
        en = assemble_energies(e_nuc, e_core, e_h, e_x, terms, n_occ)
        assert isinstance(en, Energies)
        assert en.diagnostics == {'probe': dc}
        # every term on its own: nuclear, core, and the Hartree and exchange
        # energies of the environment and of the double counting
        assert (en.e_nuc, en.e_core, en.e_hartree, en.e_exchange) == (
            e_nuc, e_core, e_h, e_x)
        assert abs(en.e_hartree_dc - hartree) < TOL
        assert abs(en.e_exchange_dc - exchange) < TOL
        assert (en.e_c_full, en.e_c_active) == (-0.31, -0.12)
        # the sums: AMS's Eh, Ex, Ec columns
        assert abs(en.eh - (e_h + hartree)) < TOL
        assert abs(en.ex - (e_x + exchange + en.e_exchange_term)) < TOL
        assert abs(en.ec - (-0.31 + 0.12)) < TOL
        assert (en.e_core, en.e_hartree, en.e_exchange) == (e_core, e_h, e_x)
        # one name per number: no second names for the HF terms
        for second in ('e_hf_core', 'e_hf_ee', 'e_hf_ee_vtilde'):
            assert not hasattr(en, second)
        assert abs(en.e0 - (en.e_nuc + en.e_core + en.eh + en.ex + en.ec
                            - en.e_sigma_trace)) < TOL
        assert abs(en.e0 - (e_nuc + e_core + e_h + e_x + hartree + exchange
                            - 0.31 + 0.12 + en.e_exchange_term
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
    en = assemble_energies(e_nuc, e_core, e_h, e_x, bare, n_occ)
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

    env = environment_hf_energy(mf.mol, mf.get_hcore(), (c, c), (part, part))
    f = core_fock(mf.mol, mf.get_hcore(), (c, c), (part, part))
    hartree, exchange = double_counting(vtilde, (k, k))
    expected = mf.e_tot - 0.31 + 0.12
    for dc, sigma_bar in (('full', None), ('hartree', None),
                          ('hartree', (sig, sig))):
        terms = DownfoldingTerms(vtilde=vtilde, sigma_bar=sigma_bar,
                                 e_c_full=-0.31, e_c_active=-0.12, dc=dc)
        en = assemble_energies(mf.energy_nuc(), *env, terms, (k, k))
        h_eff = effective_one_body(f, hartree, exchange, dc, sigma_bar)
        g_eff = {0: en.e0, 1: h_eff[0],
                 2: (v_act + vt).transpose(0, 2, 1, 3)}
        assert abs(reference_energy(g_eff, k) - expected) < 1e-8


# --- the driver and the HF low-level method (HF = CASCI) --------------------

def _water_mean_field(ref):
    """Water/6-31G (spherical): RHF, or RKS with PBE or PBE0."""
    if ref == 'hf':
        return water()
    mf = dft.RKS(water().mol, xc=ref)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


def _casci(mf, mo=None):
    """The (4,4) CASCI energy of `mf`, on `mo` when given."""
    mc = mcscf.CASCI(mf, 4, 4)
    mc.verbose = 0
    return mc.kernel(mo)[0]


def _hf_heff(mf, aspace, **kwargs):
    return build_effective_hamiltonian(run_low_level(mf, method='hf'), aspace,
                                       **kwargs)


def _solve(g, space):
    return solve('pyscf_fci', g, space['n_orbs'], space['n_elec_active'],
                 spin_free=True)[0]


@pytest.mark.parametrize('ref', ['hf', 'pbe', 'pbe0'])
def test_hf_method_is_casci(ref):
    """Contiguous (4,4) on water: H_eff of method='hf' is the CASCI
    Hamiltonian, whose core potential is J - K/2 on the reference orbitals
    also for a KS reference (pyscf's CASCI does not use the KS potential)."""
    mf = _water_mean_field(ref)
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    g, space, en = _hf_heff(mf, aspace)
    assert isinstance(en, Energies) and g[0] == en.e0
    assert abs(_solve(g, space) - _casci(mf)) < 1e-8


def test_hf_method_non_contiguous():
    """Active orbitals 2, 4 (occupied) and 5, 7 (virtual); the core is 0, 1, 3."""
    mf = water()
    aspace = ActiveSpace.from_indices(len(mf.mo_energy), 5, [2, 4], [5, 7])
    g, space, _ = _hf_heff(mf, aspace)
    mc = mcscf.CASCI(mf, 4, 4)
    mc.verbose = 0
    e_ref = mc.kernel(mc.sort_mo([3, 5, 6, 8]))[0]
    assert abs(_solve(g, space) - e_ref) < 1e-8
    assert sorted(space) == ['act_idx', 'act_occ_idx', 'act_vir_idx',
                             'active_space', 'n_elec_active', 'n_orbs',
                             'spin_free']
    assert (space['n_orbs'], space['n_elec_active']) == (4, 4)
    assert space['spin_free'] is True and space['active_space'] is aspace
    assert space['act_idx'].tolist() == [2, 4, 5, 7]
    assert space['act_occ_idx'].tolist() == [2, 4]
    assert space['act_vir_idx'].tolist() == [5, 7]


def test_spin_orbital_g_eff_solves_the_same():
    mf = water()
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    g_sf, space_sf, _ = _hf_heff(mf, aspace)
    g_so, space_so, _ = _hf_heff(mf, aspace, spin_free=False)
    assert (space_so['n_orbs'], space_so['n_elec_active']) == (8, 4)
    assert space_so['spin_free'] is False
    e_so = solve('openfermion', g_so, space_so['n_orbs'],
                 space_so['n_elec_active'])[0]
    assert abs(e_so - _solve(g_sf, space_sf)) < 1e-8


def test_density_fitted_mean_field_still_exact():
    """A density-fitted mean field gives only the orbitals: every integral of
    H_eff is exact, so it is CASCI with exact integrals on the same orbitals,
    built on a plain RHF object (a CASCI of the DF object would be a DF-CASCI)."""
    mol = water().mol
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.verbose = 0
    mf.run()
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    g, space, _ = _hf_heff(mf, aspace)
    exact = scf.RHF(mol)
    exact.verbose = 0
    exact.mo_coeff, exact.mo_occ = mf.mo_coeff, mf.mo_occ
    exact.mo_energy = mf.mo_energy
    assert abs(_solve(g, space) - _casci(exact, mf.mo_coeff)) < 1e-8


def test_reference_determinant_is_the_scf_energy():
    mf = water()
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    g, _, en = _hf_heff(mf, aspace)
    assert abs(reference_energy(g, 2) - mf.e_tot) < 1e-8
    assert (en.e_c_full, en.e_c_active) == (0.0, 0.0)
    assert (en.e_hartree_dc, en.e_exchange_dc) == (0.0, 0.0)


def test_unsupported_mean_fields_are_refused():
    mol = water().mol
    cation = gto.M(atom=mol.atom, basis='6-31g', charge=1, spin=1,
                   cart=False, verbose=0)
    with pytest.raises(NotImplementedError, match='closed-shell'):
        run_low_level(scf.ROHF(cation), method='hf')
    with pytest.raises(NotImplementedError, match='gas phase'):
        run_low_level(scf.RHF(mol).PCM(), method='hf')
    screened = scf.RHF(mol)
    screened.with_screening = SimpleNamespace()   # any attached environment
    with pytest.raises(NotImplementedError, match='gas phase'):
        run_low_level(screened, method='hf')


def test_unknown_method_and_stray_options():
    mf = water()
    with pytest.raises(ValueError, match="'hf', 'rpa'"):
        run_low_level(mf, method='mp2')
    with pytest.raises(TypeError, match="method='hf'.*screening"):
        run_low_level(mf, method='hf', screening='crpa')
    low = run_low_level(mf, method='hf')
    assert low.method == 'hf' and low.mf is mf
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    with pytest.raises(TypeError, match="method='hf'.*screening"):
        build_effective_hamiltonian(low, aspace, screening='crpa')


def test_all_orbitals_active_is_fci():
    mf = h2()
    aspace = ActiveSpace.from_indices(len(mf.mo_energy), 1, [0], [1, 2, 3])
    g, space, en = _hf_heff(mf, aspace)
    assert (en.e_core, en.e_hartree, en.e_exchange) == (0.0, 0.0, 0.0)
    assert abs(_solve(g, space) - fci.FCI(mf).kernel()[0]) < 1e-9


def test_active_space_of_another_reference_is_refused():
    mf = water()
    aspace = ActiveSpace.from_indices(len(mf.mo_energy), 4, [3], [4])
    with pytest.raises(ValueError, match='occupied'):
        _hf_heff(mf, aspace)


def test_sigma_bar_with_full_double_counting_is_refused(monkeypatch):
    """A static self-energy comes with the Hartree-only double counting; a
    method that hands one with dc='full' is refused by name."""
    mf = water()
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)

    def terms(low, part):
        n = part.n_act
        z = np.zeros((n,) * 4)
        return DownfoldingTerms(vtilde=(z, z, z),
                                sigma_bar=(np.eye(n), np.eye(n)),
                                e_c_full=0.0, e_c_active=0.0, dc='full')

    probe = SimpleNamespace(
        run=lambda mf: SimpleNamespace(method='probe', mf=mf), terms=terms)
    monkeypatch.setitem(LOW_LEVEL, 'probe', probe)
    low = run_low_level(mf, method='probe')
    with pytest.raises(ValueError, match="probe"):
        build_effective_hamiltonian(low, aspace)


def test_mpi_rank_guard(monkeypatch):
    monkeypatch.setattr(effective_hamiltonian, 'grid_comm',
                        lambda comm=None: (object(), 0, 2))
    with pytest.raises(NotImplementedError, match='serial'):
        run_low_level(water(), method='hf')


# --- the RPA low-level method on casida: the E_0 limits ---------------------

SCREENINGS_1 = ('bare', 'crpa', 'crpa_ph')


@pytest.fixture(scope='module')
def wat_rpa():
    """(mf, low, aspace): water/6-31G RHF, its casida RPA (default df=True)
    and the contiguous (4,4) active space."""
    mf = water()
    low = run_low_level(mf)
    return mf, low, ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)


def test_default_method_is_rpa(wat_rpa):
    mf, low, aspace = wat_rpa
    assert low.method == 'rpa' and low.mode == 'casida' and low.df is True
    g, space, en = build_effective_hamiltonian(low, aspace)
    assert g[0] == en.e0 and en.e_c_full == low.e_c_full
    assert set(en.diagnostics) == {'static_defect'}


@pytest.mark.parametrize('s', SCREENINGS_1)
def test_reference_determinant_energy(wat_rpa, s):
    """The reference determinant of H_eff is E_HF + E_c[v] - E_c^A[v_eff]
    for every screening. Measured 5.7e-14 for each."""
    mf, low, aspace = wat_rpa
    g, space, en = build_effective_hamiltonian(low, aspace, screening=s)
    assert abs(reference_energy(g, 2)
               - (mf.e_tot + en.e_c_full - en.e_c_active)) < 1e-9
    assert en.e_c_active < 0.0 and en.e_c_full < en.e_c_active


@pytest.mark.parametrize('s', SCREENINGS_1)
def test_all_active_is_fci_and_ec_cancels(s):
    """H2/6-31G with every orbital active: vtilde = 0, the E_c pair cancels
    (both by the plasmon formula on the same v) and H_eff is the full
    Hamiltonian. Measured |E_c - E_c^A| = 0.0 (the same eigensolve of the
    same matrices); |E - E_FCI| = 4.4e-16."""
    mf = h2()
    low = run_low_level(mf)
    aspace = ActiveSpace.from_indices(len(mf.mo_energy), 1, [0], [1, 2, 3])
    g, space, en = build_effective_hamiltonian(low, aspace, screening=s)
    e = _solve(g, space)
    assert abs(en.e_c_full - en.e_c_active) < 1e-10
    assert abs(e - fci.FCI(mf).kernel()[0]) < 1e-9


def test_ph_screenings_leave_the_active_mean_field(wat_rpa):
    """crpa_ph changes only the particle-hole elements, which neither the
    Hartree nor the exchange energy of the active reference reads:
    E_HF,ee[vtilde] = 0, and the rows of h_eff of the occupied active
    orbitals are the CASCI ones. (The virtual-virtual block does change:
    the full double counting subtracts sum_k vt_ak,kb, a ph element.)"""
    mf, low, aspace = wat_rpa
    g, _, en = build_effective_hamiltonian(low, aspace, screening='crpa_ph')
    g_hf, _, _ = _hf_heff(mf, aspace)
    assert (en.e_hartree_dc, en.e_exchange_dc) == (0.0, 0.0)
    assert np.array_equal(g[1][:2], g_hf[1][:2])
    assert np.abs(g[1][2:, 2:] - g_hf[1][2:, 2:]).max() > 1e-4
    _, _, en_crpa = build_effective_hamiltonian(low, aspace, screening='crpa')
    assert abs(en_crpa.e_hartree_dc + en_crpa.e_exchange_dc) > 1e-3


def test_bare_is_casci_plus_the_rpa_pair(wat_rpa):
    """'bare' is the mechanical embedding: the CASCI Hamiltonian shifted by
    E_c[v] - E_c^A[v]. Measured 0.0."""
    mf, low, aspace = wat_rpa
    g, space, en = build_effective_hamiltonian(low, aspace, screening='bare')
    assert (en.e_hartree_dc, en.e_exchange_dc) == (0.0, 0.0)
    assert abs(_solve(g, space)
               - (_casci(mf) + en.e_c_full - en.e_c_active)) < 1e-8


@pytest.mark.parametrize('density_fit', [False, True], ids=['rhf', 'df_rhf'])
def test_static_defect_is_zero_for_crpa(density_fit):
    """W^A[v + vtilde](0) = W(0) for crpa when v_eff is built on the v of the
    full RPA: also with a density-fitted mean field, whose fitted
    v is not the exact one. 'bare' leaves the environment screening out.
    Measured crpa (rhf / df_rhf): 3.8e-15 / 4.4e-15; bare: 0.143 / 0.143."""
    mf = water()
    if density_fit:
        mf = scf.RHF(mf.mol).density_fit()
        mf.conv_tol = 1e-11
        mf.verbose = 0
        mf.run()
    low = run_low_level(mf)
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    _, _, en = build_effective_hamiltonian(low, aspace, screening='crpa')
    _, _, en_bare = build_effective_hamiltonian(low, aspace, screening='bare')
    assert en.diagnostics['static_defect'] < 1e-8
    assert en_bare.diagnostics['static_defect'] > 1e-4


def test_diagnostics_keyword(wat_rpa):
    mf, low, aspace = wat_rpa
    _, _, en = build_effective_hamiltonian(low, aspace, diagnostics=())
    assert en.diagnostics == {}
    # None is no diagnostics, as (), not a TypeError
    _, _, en = build_effective_hamiltonian(low, aspace, sigma='srg',
                                           diagnostics=None)
    assert en.diagnostics == {}
    with pytest.raises(ValueError, match="'static_defect', 'ec_active_casida'"):
        build_effective_hamiltonian(low, aspace, diagnostics=('nope',))
    # names that do not apply (casida, no sigma) are skipped, not zero
    _, _, en = build_effective_hamiltonian(
        low, aspace, diagnostics=('sigma_poles', 'ec_active_casida',
                                  'sigma_continuation'))
    assert en.diagnostics == {}
    _, _, en = build_effective_hamiltonian(
        low, aspace, diagnostics=['Static-Defect', 'sigma_poles'])
    assert list(en.diagnostics) == ['static_defect']
    _, _, en = build_effective_hamiltonian(low, aspace,
                                           diagnostics='static_defect')
    assert list(en.diagnostics) == ['static_defect']


def test_rpa_option_refusals(wat_rpa):
    mf, low, aspace = wat_rpa
    with pytest.raises(ValueError, match="'crpa'"):
        build_effective_hamiltonian(low, aspace, screening='rpa')
    for sigma in ('None', 'none', '', 'gw', 1):
        with pytest.raises(ValueError, match="None, 'srg', 'mode_a'"):
            build_effective_hamiltonian(low, aspace, sigma=sigma)
    # the string members in any case and either of '-', '_',
    # as screening and mode take them; None alone is no Sigma^E
    for canonical, spellings in (('srg', ('SRG', 'Srg')),
                                 ('mode_a', ('MODE_A', 'mode-a', 'Mode-A'))):
        want = build_effective_hamiltonian(low, aspace, sigma=canonical)
        for sigma in spellings:
            _same(build_effective_hamiltonian(low, aspace, sigma=sigma), want)
    with pytest.raises(TypeError, match="method='rpa'.*nope"):
        build_effective_hamiltonian(low, aspace, nope=1)


def _same(a, b):
    (g1, _, e1), (g2, _, e2) = a, b
    assert abs(g1[0] - g2[0]) < 1e-12
    assert np.abs(g1[1] - g2[1]).max() < 1e-12
    assert np.abs(g1[2] - g2[2]).max() < 1e-12
    for name in ('e_hartree_dc', 'e_exchange_dc', 'e_c_full', 'e_c_active',
                 'e0'):
        assert abs(getattr(e1, name) - getattr(e2, name)) < 1e-12
    assert e1.diagnostics.keys() == e2.diagnostics.keys()


def test_one_low_serves_many_calls(wat_rpa):
    """One low-level result, used for crpa, crpa_ph, bare and a second active
    space in turn, gives each the same H_eff as a fresh one: no call leaves
    anything behind that a later call reads."""
    mf, low, aspace = wat_rpa
    other = ActiveSpace.from_indices(len(mf.mo_energy), 5, [2, 4], [5, 7])
    calls = [(aspace, 'crpa'), (aspace, 'crpa_ph'), (aspace, 'bare'),
             (other, 'crpa'), (aspace, 'crpa')]
    shared = [build_effective_hamiltonian(low, a, screening=s)
              for a, s in calls]
    for (a, s), got in zip(calls, shared):
        fresh = build_effective_hamiltonian(run_low_level(mf), a, screening=s)
        _same(got, fresh)


@pytest.mark.parametrize('s', SCREENINGS_1)
def test_pbe_all_active_fci_equals_hf(s):
    """With every H2/6-31G orbital active H_eff is the full Hamiltonian on
    the orbitals it is given, so PBE and HF orbitals give the same FCI
    energy. Measured 0.0 for every screening."""
    mf_hf = h2()
    mf_pbe = dft.RKS(mf_hf.mol, xc='pbe')
    mf_pbe.conv_tol = 1e-11
    mf_pbe.verbose = 0
    mf_pbe.run()
    energies = []
    for mf in (mf_hf, mf_pbe):
        aspace = ActiveSpace.from_indices(len(mf.mo_energy), 1, [0],
                                          [1, 2, 3])
        g, space, _ = build_effective_hamiltonian(run_low_level(mf), aspace,
                                                  screening=s)
        energies.append(_solve(g, space))
    assert abs(energies[0] - energies[1]) < 1e-7


def test_empty_active_space_e0(wat_rpa):
    """With no active orbital E_0 = E_HF + E_c[v]; ActiveSpace
    takes the empty selection. No solver is called. On the fitted and on the
    exact integrals of the low-level run. Measured 1.4e-14."""
    mf, low_df, _ = wat_rpa
    empty = ActiveSpace.from_indices(len(mf.mo_energy), 5, [], [])
    for low in (low_df, run_low_level(mf, df=False)):
        for s in SCREENINGS_1:
            g, space, en = build_effective_hamiltonian(low, empty,
                                                       screening=s)
            assert space['n_orbs'] == 0 and en.e_c_active == 0.0
            assert abs(en.e0 - (mf.e_tot + low.e_c_full)) < 1e-9
            assert g[0] == en.e0 and g[2].shape == (0, 0, 0, 0)


# --- Sigma^E in H_eff and E_0 -----------------------------------------------

SIGMAS_1 = ('srg', 'mode_a')


@pytest.mark.parametrize('s', SCREENINGS_1)
@pytest.mark.parametrize('sigma', SIGMAS_1)
def test_reference_determinant_is_unchanged_by_sigma(wat_rpa, s, sigma):
    """Switching Sigma^E on changes h_eff by Sigma_bar minus the exchange
    double counting and E_0 by 2 K~ - 2 sum_k Sigma_bar_kk, so the reference
    determinant keeps E_HF + E_c[v] - E_c^A[v_eff] with the real Sigma_bar.
    The switch does change H_eff (Sigma_bar and h_eff differ by more than
    1e-3), not v_eff and not the E_c pair. Measured 0.0 in all six cases
    (tolerance 1e-9): Sigma_trace -0.0777 for every screening."""
    mf, low, aspace = wat_rpa
    g_plain, _, en_plain = build_effective_hamiltonian(low, aspace,
                                                       screening=s)
    g, _, en = build_effective_hamiltonian(low, aspace, screening=s,
                                           sigma=sigma)
    assert abs(reference_energy(g, 2) - reference_energy(g_plain, 2)) < 1e-9
    assert abs(en.e_sigma_trace) > 1e-3
    assert np.abs(g[1] - g_plain[1]).max() > 1e-3
    assert np.array_equal(g[2], g_plain[2])
    assert (en.e_c_full, en.e_c_active) == (en_plain.e_c_full,
                                            en_plain.e_c_active)


@pytest.mark.parametrize('s', SCREENINGS_1)
@pytest.mark.parametrize('sigma', SIGMAS_1)
def test_sigma_switch_with_zero_sigma_bar(wat_rpa, monkeypatch, s, sigma):
    """The sign test: with Sigma_bar forced to 0 (the residues of W
    zeroed) the switch changes only the double counting and E_0 (the exchange
    energy term), and the reference determinant keeps its energy. Measured
    0.0 in all six cases (tolerance 1e-9); for crpa the switch moves h_eff by the
    exchange double counting (more than 1e-3)."""
    mf, low, aspace = wat_rpa
    g_plain, _, _ = build_effective_hamiltonian(low, aspace, screening=s)
    monkeypatch.setattr(rpa, 'screened_residues',
                        lambda low, part: np.zeros((len(low.omega),
                                                    part.n_act,
                                                    len(part.env))))
    g, _, en = build_effective_hamiltonian(low, aspace, screening=s,
                                           sigma=sigma)
    assert en.e_sigma_trace == 0.0
    assert abs(reference_energy(g, 2) - reference_energy(g_plain, 2)) < 1e-9
    if s == 'crpa':
        assert abs(en.e_exchange_term) > 1e-3
        assert np.abs(g[1] - g_plain[1]).max() > 1e-3


@pytest.mark.parametrize('sigma', SIGMAS_1)
def test_exchange_term_vanishes_for_ph_screening(wat_rpa, sigma):
    """crpa_ph: vtilde has no (kl|lk) and no (tu|kk) element, so K~ = 0 and
    the Hartree double counting is zero: with Sigma^E the exchange energy
    term is exactly 0 and h_eff = f + Sigma_bar, f the core Fock operator of
    method='hf' (nothing double counted is left). Measured 0.0 (tolerance
    1e-12); crpa's exchange term is -0.585."""
    mf, low, aspace = wat_rpa
    g, _, en = build_effective_hamiltonian(low, aspace, screening='crpa_ph',
                                           sigma=sigma)
    assert en.e_exchange_term == 0.0
    g_hf, _, _ = _hf_heff(mf, aspace)
    part = partition(aspace, mf.mo_energy)
    s_bar = rpa.embedding_self_energy(low, part, sigma)
    assert np.abs(g[1] - (g_hf[1] + s_bar)).max() < 1e-12
    assert abs(en.e_sigma_trace - 2.0 * np.trace(s_bar[:2, :2])) < 1e-12
    _, _, en_crpa = build_effective_hamiltonian(low, aspace, screening='crpa',
                                                sigma=sigma)
    assert abs(en_crpa.e_exchange_term) > 1e-3


@pytest.mark.parametrize('s', SCREENINGS_1)
@pytest.mark.parametrize('sigma', SIGMAS_1)
def test_all_active_sigma_is_zero(s, sigma):
    """H2/6-31G with every orbital active: no environment, so Sigma_bar = 0,
    and vtilde = 0, so both Sigma^E terms of E_0 are exactly zero and H_eff
    is still the full Hamiltonian. Measured |E - E_FCI| at most 4.4e-16
    (tolerance 1e-9)."""
    mf = h2()
    low = run_low_level(mf)
    aspace = ActiveSpace.from_indices(len(mf.mo_energy), 1, [0], [1, 2, 3])
    g, space, en = build_effective_hamiltonian(low, aspace, screening=s,
                                               sigma=sigma)
    assert (en.e_exchange_term, en.e_sigma_trace) == (0.0, 0.0)
    assert abs(_solve(g, space) - fci.FCI(mf).kernel()[0]) < 1e-9


def test_empty_active_space_with_sigma(wat_rpa):
    """No active orbital: Sigma_bar is (0, 0) and E_0 = E_HF + E_c[v] with
    Sigma^E too. Measured 1.4e-14 (tolerance 1e-9)."""
    mf, low, _ = wat_rpa
    empty = ActiveSpace.from_indices(len(mf.mo_energy), 5, [], [])
    for sigma in SIGMAS_1:
        _, _, en = build_effective_hamiltonian(low, empty, sigma=sigma)
        assert (en.e_exchange_term, en.e_sigma_trace) == (0.0, 0.0)
        assert abs(en.e0 - (mf.e_tot + low.e_c_full)) < 1e-9


@pytest.mark.parametrize('s', SCREENINGS_1)
def test_e0_terms_do_not_depend_on_the_form_at_infinite_s(wat_rpa, s):
    """srg_s = None: the diagonal of 'srg' is that of 'mode_a' and
    gamma^A,0 is diagonal, so every E_0 term agrees between the two forms
    while the off-diagonal h_eff does not. Measured |Delta E_0| = 0.0 for every
    screening (tolerance 1e-12); the off-diagonal h_eff differs by up to
    1.9e-4."""
    mf, _, aspace = wat_rpa
    low = run_low_level(mf, numerics=Numerics(downfolding={'srg_s': None}))
    (g_srg, _, srg), (g_a, _, mode_a) = (
        build_effective_hamiltonian(low, aspace, screening=s, sigma=sigma)
        for sigma in SIGMAS_1)
    for name in ('e_hartree_dc', 'e_exchange_dc', 'e_exchange_term',
                 'e_sigma_trace', 'e0'):
        assert abs(getattr(srg, name) - getattr(mode_a, name)) < 1e-12
    assert np.abs(np.diag(g_srg[1]) - np.diag(g_a[1])).max() < 1e-12
    assert np.abs(g_srg[1] - g_a[1]).max() > 1e-6
    poles = srg.diagnostics['sigma_poles']
    assert poles['srg_s'] is None and poles['srg_damping'] == 0.0


def test_sigma_with_hf_is_refused():
    """method='hf' has no self-energy and takes no options."""
    mf = water()
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    for sigma in SIGMAS_1:
        with pytest.raises(TypeError, match="method='hf'.*sigma"):
            _hf_heff(mf, aspace, sigma=sigma)


# --- one restricted spin channel: nothing computed twice ---------------------

def test_one_restricted_channel_of_the_energy_terms():
    """The E_0 terms of ONE restricted channel are those of two identical
    channels (tolerance 1e-11): the environment HF energy, E_HF,ee[vtilde]
    and its two halves, the exchange energy term and the Sigma_bar trace."""
    mf = water()
    part = _water_part(mf, [2, 4], [5, 6])
    c, hcore = mf.mo_coeff, mf.get_hcore()
    n, k = part.n_act, part.n_act_occ
    rng = np.random.default_rng(6)
    vt = _random_vtilde(rng, n)[0]
    sig = _symmetric(rng, n)
    one = environment_hf_energy(mf.mol, hcore, (c,), (part,))
    two = environment_hf_energy(mf.mol, hcore, (c, c), (part, part))
    assert np.allclose(one, two, rtol=0, atol=1e-11)
    assert np.allclose(active_hf_energy((vt,), (k,)),
                       active_hf_energy((vt, vt, vt), (k, k)),
                       rtol=0, atol=1e-12)
    assert abs(exchange_energy_term((vt,), (k,))
               - exchange_energy_term((vt, vt, vt), (k, k))) < 1e-12
    assert abs(sigma_trace((sig,), (k,))
               - sigma_trace((sig, sig), (k, k))) < 1e-12
    for dc in ('full', 'hartree'):
        en = [assemble_energies(
            1.0, *env, DownfoldingTerms(vtilde=v, sigma_bar=s, e_c_full=-0.3,
                                        e_c_active=-0.1, dc=dc), occ).e0
              for env, v, s, occ in ((one, (vt,), (sig,), (k,)),
                                     (two, (vt, vt, vt), (sig, sig), (k, k)))]
        assert abs(en[0] - en[1]) < 1e-11


@pytest.mark.parametrize('method', ['hf', 'rpa'])
def test_the_driver_builds_each_integral_once(monkeypatch, method):
    """One restricted channel end to end: `build_effective_hamiltonian` does
    ONE exact J/K build (shared by the core Fock operator and the environment
    energy) and ONE integral transformation of the active block, and the
    method hands back one vtilde."""
    from src.MultiReference.Downfolding import one_body, two_body
    mf = water()
    low = run_low_level(mf, method=method)
    calls = {'jk': 0, 'ao2mo': 0}
    real_jk, real_general = one_body.get_jk, two_body.ao2mo.general

    def counted_jk(*args, **kwargs):
        calls['jk'] += 1
        return real_jk(*args, **kwargs)

    def counted_general(*args, **kwargs):
        calls['ao2mo'] += 1
        return real_general(*args, **kwargs)

    monkeypatch.setattr(one_body, 'get_jk', counted_jk)
    monkeypatch.setattr(two_body.ao2mo, 'general', counted_general)
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    part = partition(aspace, mf.mo_energy)
    assert len(LOW_LEVEL[method].terms(low, part).vtilde) == 1
    build_effective_hamiltonian(low, aspace)
    assert calls == {'jk': 1, 'ao2mo': 1}


def test_the_blocks_of_a_method_must_fit_the_reference(monkeypatch):
    """A restricted reference is one spin channel, vtilde = (v,) and
    sigma_bar = (s,); an unrestricted one is two, (aa, ab, bb) and (a, b),
    whose `terms` gets the pair of partitions. A method that hands back the
    blocks of the other kind is refused by name."""
    mf = water()
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    seen = []

    def terms(low, part, nblocks=1, nsigma=None):
        seen.append(part)
        z = np.zeros((aspace.n_act,) * 4)
        sigma = None if nsigma is None else (np.zeros((4, 4)),) * nsigma
        return DownfoldingTerms(
            vtilde=(z,) * nblocks, sigma_bar=sigma, e_c_full=0.0,
            e_c_active=0.0, dc='full' if nsigma is None else 'hartree')

    monkeypatch.setitem(LOW_LEVEL, 'probe', SimpleNamespace(terms=terms))
    low = SimpleNamespace(mf=mf, method='probe')
    build_effective_hamiltonian(low, aspace)
    assert seen[-1].n_act == 4                    # one Partition, not a tuple
    with pytest.raises(ValueError, match='3 vtilde blocks.*restricted'):
        build_effective_hamiltonian(low, aspace, nblocks=3)
    with pytest.raises(ValueError, match='2 Sigma_bar blocks.*restricted'):
        build_effective_hamiltonian(low, aspace, nsigma=2)

    low = SimpleNamespace(mf=mf.to_uhf(), method='probe')
    g_eff, _, _ = build_effective_hamiltonian(low, (aspace, aspace), nblocks=3,
                                              nsigma=2)
    assert isinstance(seen[-1], tuple) and len(seen[-1]) == 2
    assert len(g_eff[1]) == 2 and len(g_eff[2]) == 3
    with pytest.raises(ValueError, match='1 vtilde blocks.*unrestricted'):
        build_effective_hamiltonian(low, (aspace, aspace), nblocks=1)
    with pytest.raises(ValueError, match='1 Sigma_bar blocks.*unrestricted'):
        build_effective_hamiltonian(low, (aspace, aspace), nblocks=3, nsigma=1)


# --- named flags are collected into the results -------------------------------

def test_named_flags_reach_the_results(monkeypatch):
    """A named flag raised under `run_low_level` is in low.numerical_flags,
    and `energies.numerical_flags` holds those plus the ones raised under
    `build_effective_hamiltonian`, each distinct entry once, sorted, as
    {'flag': name, 'message': text} (src.Base.numerical_flags). A later
    build does not inherit an earlier build's flags."""
    from src.Base.numerical_flags import flag
    mf = water()
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    raise_in_terms = [True]

    def run(mf):
        flag('probe_run', 'raised by run', RuntimeWarning)
        return SimpleNamespace(mf=mf, method='probe')

    def terms(low, part):
        if raise_in_terms[0]:
            flag('probe_terms', 'raised by terms', RuntimeWarning)
            flag('probe_terms', 'raised by terms', RuntimeWarning)
        z = np.zeros((part.n_act,) * 4)
        return DownfoldingTerms(vtilde=(z,), sigma_bar=None, e_c_full=0.0,
                                e_c_active=0.0, dc='full')

    monkeypatch.setitem(LOW_LEVEL, 'probe', SimpleNamespace(run=run,
                                                            terms=terms))
    with pytest.warns(RuntimeWarning, match='raised by run'):
        low = run_low_level(mf, method='probe')
    ran = {'flag': 'probe_run', 'message': 'raised by run'}
    assert low.numerical_flags == [ran]
    with pytest.warns(RuntimeWarning, match='raised by terms'):
        _, _, en = build_effective_hamiltonian(low, aspace)
    assert en.numerical_flags == [
        ran, {'flag': 'probe_terms', 'message': 'raised by terms'}]
    raise_in_terms[0] = False
    _, _, en = build_effective_hamiltonian(low, aspace)
    assert en.numerical_flags == [ran]
    assert low.numerical_flags == [ran]


def test_a_clean_run_has_no_flags():
    mf = water()
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    for method in ('hf', 'rpa'):
        low = run_low_level(mf, method=method)
        assert low.numerical_flags == []
        assert build_effective_hamiltonian(low, aspace)[2].numerical_flags == []
