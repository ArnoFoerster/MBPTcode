"""Downfolding from an unrestricted (UHF/UKS) reference: two spin channels
with their own orbitals, orbital energies, occupied counts and active spaces,
and an effective Hamiltonian with different alpha and beta blocks.

The driver with method='hf' gives the CASCI Hamiltonian of the unrestricted
determinant in its own orbitals. It is checked against
- pyscf's UCASCI where that is right (equal core counts, contiguous spaces):
  the OH radical (N^A = (2, 1) in 3 orbitals) and triplet CH2 ((3, 1) in 4);
- pyscf's FCI of the molecule, in restricted orbitals, with every orbital
  active (broken-symmetry H2, OH/STO-3G);
- the unrestricted SCF energy with no orbital active, and the HF energy
  expression on the Kohn-Sham determinant for a UKS reference;
- the spin-orbital oracle (tests/downfolding_oracle.py) block by block, for
  different non-contiguous index sets per spin, unequal core counts and
  N^A_alpha != N^A_beta, where the reference determinant of H_eff also has
  the UHF energy;
- the restricted result on the same orbitals handed over as a UHF object
  (`to_uhf()`): h_alpha = h_beta, the three two-body blocks equal, every
  energy term the same;
- a second solver: fci.direct_uhf on the per-spin blocks against the
  determinant-basis diagonalization of the spin-orbital form (unequal cores,
  N^A = (2, 2)).

The forms of the active space that do not fit the reference, and the
references that are not supported, are refused with a message that names the
spin.

The RPA formulas for two spins are checked against the oracle's spin-orbital
functions, which use no spin adaptation; on a closed shell those reduce to
the oracle's restricted functions (the first test below the method='hf'
ones).
"""
import copy
import warnings
from types import SimpleNamespace
import numpy as np
import pytest
from pyscf import fci, gto, mcscf, scf
from pyscf.gw import urpa
from src.Base.numerics.constants import DOWNFOLDING_SRG_S
from src.Base.pyscf import pyscf_interface
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.methods import rpa
from src.EffectiveHamiltonian.Downfolding.masks import ph_element_mask
from src.EffectiveHamiltonian.active_space import (ActiveSpace,
                                                 unrestricted_from_counts)
from src.EffectiveHamiltonian.solvers.solve import (
    EmptyActiveSpaceWarning, solve)
from tests.downfolding_oracle import (CH2, NH, OH, active_pairs_so, drpa_so,
                                      env_residues_uhf,
                                      fitted_spin_orbital_integrals,
                                      h2_broken,
                                      mo_integrals, reference_energy_uhf,
                                      rpa_energy, sigma_mode_a,
                                      sigma_mode_a_uhf, sigma_srg,
                                      sigma_srg_uhf, spin_blocks,
                                      spin_orbital_energies,
                                      spin_orbital_integrals,
                                      spin_orbital_one_body, unrestricted,
                                      vtilde_crpa, vtilde_crpa_uhf,
                                      vtilde_mrpa, vtilde_mrpa_uhf,
                                      w0_minus_v, w0_minus_v_so, water)


# run_low_level warns of a spin-contaminated or internally unstable UHF/UKS
# determinant at the first run on each mean field, and two references of
# this file are such: the broken-symmetry H2 (<S^2> = 0.98 against 0) and
# the OH UKS/PBE0 determinant, which pyscf's stability analysis reports as
# unstable. Which test meets them first depends on the selection, so the two
# warnings are let through here;
# test_reference_diagnostics_are_flags_of_the_result checks them.
pytestmark = pytest.mark.filterwarnings(
    'ignore:Downfolding. the unrestricted reference is:RuntimeWarning')


@pytest.fixture(scope='module')
def oh():
    return unrestricted(OH, 1)


@pytest.fixture(scope='module')
def ch2():
    return unrestricted(CH2, 2)


@pytest.fixture(scope='module')
def h2_bs():
    return h2_broken()


def _nocc(mf):
    return tuple(int(np.count_nonzero(occ)) for occ in mf.mo_occ)


def _hf(mf, spaces, **kwargs):
    return build_effective_hamiltonian(run_low_level(mf, method='hf'), spaces,
                                       **kwargs)


def _solve(g_eff, space, **kwargs):
    return solve('pyscf_fci', g_eff, space['n_orbs'], space['n_elec_active'],
                 spin_free=space['spin_free'], **kwargs)


def _all_active(mf):
    nmo = mf.mo_coeff.shape[-1]
    return tuple(ActiveSpace.from_indices(nmo, n, range(n), range(n, nmo))
                 for n in _nocc(mf))


def _none_active(mf):
    nmo = mf.mo_coeff.shape[-1]
    return tuple(ActiveSpace.from_indices(nmo, n, [], []) for n in _nocc(mf))


def _asymmetric(mf):
    """OH/6-31G: different, non-contiguous orbitals per spin, cores (3, 3) of
    different orbitals, N^A = (2, 1) in 4 orbitals."""
    nmo = mf.mo_coeff.shape[-1]
    return (ActiveSpace.from_indices(nmo, 5, [2, 4], [5, 7]),
            ActiveSpace.from_indices(nmo, 4, [3], [4, 6, 7]))


def _unequal_cores(mf):
    """OH/6-31G: cores (3, 2), N^A = (2, 2) in 4 orbitals."""
    nmo = mf.mo_coeff.shape[-1]
    return (ActiveSpace.from_indices(nmo, 5, [3, 4], [5, 6]),
            ActiveSpace.from_indices(nmo, 4, [2, 3], [4, 5]))


def _no_beta_pair(mf):
    """OH/6-31G: N^A = (2, 0) in 3 orbitals. The beta channel has no active
    occupied orbital, so no active particle-hole pair."""
    nmo = mf.mo_coeff.shape[-1]
    return (ActiveSpace.from_indices(nmo, 5, [3, 4], [5]),
            ActiveSpace.from_indices(nmo, 4, [], [4, 5, 6]))


def _occupied_only(mf):
    """OH/6-31G: two occupied orbitals per spin and no virtual one, so no
    active particle-hole pair in either channel; cores (3, 2)."""
    nmo = mf.mo_coeff.shape[-1]
    return (ActiveSpace.from_indices(nmo, 5, [3, 4], []),
            ActiveSpace.from_indices(nmo, 4, [2, 3], []))


def _beta_lumo_outside(mf):
    """OH/6-31G: the beta LUMO, orbital 4, is left in the environment. It is
    empty in beta but lies below the alpha occupied count (5), so it tells
    which spin's occupied count sorts the beta environment into occupied
    and empty orbitals. N^A = (2, 2) in 4 orbitals. (Measured against the
    oracle on this space: vtilde 4.6e-15, Sigma_bar 4.5e-15 with 'srg' and
    3.5e-14 with 'mode_a'. With the alpha count used for beta, the Sigma^E
    tests and the H_eff test on this space fail; tried.)"""
    nmo = mf.mo_coeff.shape[-1]
    return (ActiveSpace.from_indices(nmo, 5, [3, 4], [6, 7]),
            ActiveSpace.from_indices(nmo, 4, [2, 3], [5, 6]))


SPACES = {'asymmetric': _asymmetric, 'unequal_cores': _unequal_cores,
          'no_beta_pair': _no_beta_pair, 'occupied_only': _occupied_only,
          'beta_lumo_outside': _beta_lumo_outside}


def _parts(mf, spaces):
    return tuple(space
                 for space, eps in zip(spaces, mf.mo_energy))


def _spin_orbital(g_eff):
    """The interleaved spin-orbital form (alpha at 2p, beta at 2p + 1) with
    the antisymmetrized physicist two-body term, written out."""
    (h_a, h_b), (aa, ab, bb) = g_eff[1], g_eff[2]
    n = h_a.shape[0]
    h = np.zeros((2 * n,) * 2)
    h[0::2, 0::2], h[1::2, 1::2] = h_a, h_b
    phys = np.zeros((2 * n,) * 4)
    phys[0::2, 0::2, 0::2, 0::2] = aa
    phys[1::2, 1::2, 1::2, 1::2] = bb
    phys[0::2, 1::2, 0::2, 1::2] = ab                       # <p_a q_b|r_a s_b>
    phys[1::2, 0::2, 1::2, 0::2] = ab.transpose(1, 0, 3, 2)
    return {0: g_eff[0], 1: h, 2: phys - phys.transpose(0, 1, 3, 2)}


# --- method='hf': the CASCI Hamiltonian of the unrestricted determinant -------

@pytest.mark.parametrize('name, n_act, nelec', [('oh', 3, (2, 1)),
                                                ('ch2', 4, (3, 1))])
def test_hf_matches_ucasci(request, name, n_act, nelec):
    """Equal core counts and contiguous spaces, where pyscf's UCASCI is the
    same Hamiltonian. Measured |dE|: 1.4e-14 (OH), 7.1e-15 (CH2), one unit in
    the last place of each energy."""
    mf = request.getfixturevalue(name)
    spaces = unrestricted_from_counts(mf.mo_energy, _nocc(mf), n_act, nelec)
    ham = _hf(mf, spaces)
    g_eff, space, energies = ham.g_eff, ham.space, ham.energies
    assert space['n_elec_active'] == nelec and space['n_orbs'] == n_act
    assert g_eff[0] == energies.e0
    ref = mcscf.UCASCI(mf, n_act, nelec)
    ref.verbose = 0
    assert abs(_solve(g_eff, space)[0] - ref.kernel()[0]) < 1e-10


@pytest.mark.parametrize('name', ['h2_bs', 'oh_sto3g'])
def test_hf_all_active_is_fci(request, name):
    """Every orbital of both spins active: the FCI energy of the molecule,
    here from pyscf's FCI in restricted (RHF, ROHF) orbitals. Measured |dE|:
    0 for both (16 and 90 determinants, diagonalized exactly on both sides)."""
    if name == 'oh_sto3g':
        mf = unrestricted(OH, 1, basis='sto-3g')
        restricted = scf.ROHF(mf.mol)
    else:
        mf = request.getfixturevalue(name)
        restricted = scf.RHF(mf.mol)
    restricted.conv_tol = 1e-11
    restricted.verbose = 0
    restricted.kernel()
    ham = _hf(mf, _all_active(mf))
    g_eff, space, energies = ham.g_eff, ham.space, ham.energies
    assert (energies.e_core, energies.e_hartree, energies.e_exchange) == (0, 0, 0)
    assert space['n_elec_active'] == mf.mol.nelec
    assert abs(_solve(g_eff, space)[0] - fci.FCI(restricted).kernel()[0]) < 1e-9


def test_hf_none_active_is_uhf(oh):
    """No orbital active: E_0 is the unrestricted SCF energy, and for a UKS
    reference the HF energy expression on the Kohn-Sham determinant.
    Measured: 1.4e-14 for both; the HF expression on the PBE0 determinant is
    0.265 Ha above the UKS energy."""
    ham = _hf(oh, _none_active(oh))
    g_eff, space, energies = ham.g_eff, ham.space, ham.energies
    assert abs(energies.e0 - oh.e_tot) < 1e-10
    assert space['n_elec_active'] == (0, 0) and space['n_orbs'] == 0
    assert [h.shape for h in g_eff[1]] == [(0, 0)] * 2
    assert [v.shape for v in g_eff[2]] == [(0, 0, 0, 0)] * 3
    with pytest.warns(EmptyActiveSpaceWarning):
        assert np.array_equal(_solve(g_eff, space), [energies.e0])

    uks = unrestricted(OH, 1, xc='pbe0')
    hf_expression = scf.UHF(uks.mol).energy_tot(dm=uks.make_rdm1())
    assert abs(hf_expression - uks.e_tot) > 1e-2
    ham = _hf(uks, _none_active(uks))
    energies = ham.energies
    assert abs(energies.e0 - hf_expression) < 1e-10


def test_hf_blocks_against_spin_orbital_oracle(oh):
    """Different non-contiguous index sets per spin, N^A = (2, 1): every
    block of H_eff and every E_0 term against the spin-orbital oracle, and
    the reference determinant of H_eff has the UHF energy. Measured: one-body
    blocks 2.9e-15, two-body blocks 6.2e-16, energy terms at most 2.8e-14,
    reference determinant 4.3e-14; h_alpha and h_beta differ by 0.45."""
    spaces = _asymmetric(oh)
    ham = _hf(oh, spaces)
    g_eff, space, energies = ham.g_eff, ham.space, ham.energies
    nmo = oh.mo_coeff.shape[-1]
    n = space['n_orbs']
    assert n == 4 and space['n_elec_active'] == (2, 1)
    h_so, eri_so = spin_orbital_integrals(oh.mol, oh.mo_coeff[0],
                                          oh.mo_coeff[1])
    act = list(spaces[0].act) + list(nmo + spaces[1].act)
    occ_env = list(spaces[0].env_occ) + list(nmo + spaces[1].env_occ)
    occ_act = [0, 1, n]
    zero = np.zeros((2 * n,) * 4)
    f, _ = spin_orbital_one_body(h_so, eri_so, occ_env, act, zero, occ_act,
                                 'full')
    al, be = slice(0, n), slice(n, 2 * n)
    assert np.max(np.abs(g_eff[1][0] - f[al, al])) < 1e-10
    assert np.max(np.abs(g_eff[1][1] - f[be, be])) < 1e-10
    assert np.max(np.abs(g_eff[1][0] - g_eff[1][1])) > 1e-2
    chem = eri_so[np.ix_(act, act, act, act)]
    for block, ref in zip(g_eff[2], (chem[al, al, al, al], chem[al, al, be, be],
                                     chem[be, be, be, be])):
        assert np.max(np.abs(block.transpose(0, 2, 1, 3) - ref)) < 1e-10
    ref = spin_orbital_energies(h_so, eri_so, occ_env, zero, occ_act)
    assert abs(energies.e_core - ref['e_core']) < 1e-10
    assert abs(energies.e_hartree - ref['e_hartree']) < 1e-10
    assert abs(energies.e_exchange - ref['e_exchange']) < 1e-10
    assert abs(energies.e0 - (oh.energy_nuc() + ref['e_core'] + ref['e_ee'])) < 1e-10
    assert abs(reference_energy_uhf(g_eff, (2, 1)) - oh.e_tot) < 1e-9


def test_closed_shell_limit_hf():
    """A closed-shell determinant handed over as a UHF object runs on the
    unrestricted path, never collapsed, and gives the restricted numbers in
    every block and every energy term. Measured: 0 in every comparison."""
    rhf = water()
    uhf = rhf.to_uhf()
    nmo = len(rhf.mo_energy)
    one = ActiveSpace.from_indices(nmo, 5, [2, 4], [5, 7])
    ham_r = _hf(rhf, one)
    g_r, space_r, en_r = ham_r.g_eff, ham_r.space, ham_r.energies
    ham_u = _hf(uhf, (one, one))
    g_u, space_u, en_u = ham_u.g_eff, ham_u.space, ham_u.energies
    assert isinstance(g_u[1], tuple) and len(g_u[1]) == 2 and len(g_u[2]) == 3
    assert all(np.max(np.abs(h - g_r[1])) < 1e-12 for h in g_u[1])
    assert all(np.max(np.abs(v - g_r[2])) < 1e-12 for v in g_u[2])
    for name in ('e_nuc', 'e_core', 'e_hartree', 'e_exchange', 'e_hartree_dc',
                 'e_exchange_dc', 'e_c_full', 'e_c_active', 'e_exchange_term',
                 'e_sigma_trace', 'e0'):
        assert abs(getattr(en_u, name) - getattr(en_r, name)) < 1e-12, name
    assert space_u['n_elec_active'] == (2, 2) and space_r['n_elec_active'] == 4
    assert space_u['n_orbs'] == space_r['n_orbs'] == 4
    e_r = solve('pyscf_fci', g_r, 4, 4, spin_free=True)[0]
    assert abs(_solve(g_u, space_u)[0] - e_r) < 1e-10


def test_spin_orbital_form_matches_blocks(oh):
    """spin_free=False is the per-spin blocks interleaved (alpha at 2p, beta
    at 2p + 1) and antisymmetrized, element by element, for N^A = (2, 1).
    Measured: 0 in both arrays."""
    spaces = _asymmetric(oh)
    ham_blocks = _hf(oh, spaces)
    g_blocks, space = ham_blocks.g_eff, ham_blocks.space
    ham_so = _hf(oh, spaces, spin_free=False)
    g_so, space_so = ham_so.g_eff, ham_so.space
    ref = _spin_orbital(g_blocks)
    assert g_so[0] == ref[0]
    assert g_so[1].shape == (8, 8) and g_so[2].shape == (8,) * 4
    assert np.max(np.abs(g_so[1] - ref[1])) < 1e-13
    assert np.max(np.abs(g_so[2] - ref[2])) < 1e-13
    assert space_so['n_orbs'] == 8 and space['n_orbs'] == 4
    assert space_so['n_elec_active'] == space['n_elec_active'] == (2, 1)
    assert space_so['spin_free'] is False and space['spin_free'] is True


def test_unequal_cores_against_second_solver(oh):
    """Cores (3, 2), N^A = (2, 2) in 4 orbitals: fci.direct_uhf on the
    per-spin blocks and the determinant-basis diagonalization of the
    spin-orbital form give the same lowest roots. Measured: 5.7e-14 over
    three roots; the reference determinant has the UHF energy to 4.3e-14."""
    spaces = _unequal_cores(oh)
    ham_blocks = _hf(oh, spaces)
    g_blocks, space = ham_blocks.g_eff, ham_blocks.space
    ham_so = _hf(oh, spaces, spin_free=False)
    g_so, space_so = ham_so.g_eff, ham_so.space
    assert space['n_elec_active'] == (2, 2)
    blocks = _solve(g_blocks, space, nroots=3)
    dets = solve('openfermion', g_so, space_so['n_orbs'],
                 space_so['n_elec_active'], spin_free=False, nroots=3)
    assert np.abs(blocks - dets).max() < 1e-9
    assert abs(reference_energy_uhf(g_blocks, (2, 2)) - oh.e_tot) < 1e-9


# --- the active space: a pair, one ActiveSpace per spin -----------------------

def test_active_space_forms_are_refused(oh):
    nmo = oh.mo_coeff.shape[-1]
    low = run_low_level(oh, method='hf')
    alpha = ActiveSpace.from_indices(nmo, 5, [4], [5, 6])
    beta = ActiveSpace.from_indices(nmo, 4, [3], [4, 5])
    build_effective_hamiltonian(low, (alpha, beta))
    build_effective_hamiltonian(low, [alpha, beta])
    with pytest.raises(ValueError, match='pair'):
        build_effective_hamiltonian(low, alpha)
    with pytest.raises(ValueError, match='pair'):
        build_effective_hamiltonian(low, (alpha, beta, beta))
    with pytest.raises(ValueError, match='3 and 2'):
        build_effective_hamiltonian(
            low, (alpha, ActiveSpace.from_indices(nmo, 4, [3], [4])))
    # built with the other spin's occupied count: the message names the spin
    # and both counts
    with pytest.raises(ValueError,
                       match='beta active space.*5 occupied.*4 beta occupied'):
        build_effective_hamiltonian(low, (alpha, alpha))
    with pytest.raises(ValueError,
                       match='alpha active space.*4 occupied.*5 alpha occupied'):
        build_effective_hamiltonian(low, (beta, beta))
    with pytest.raises(ValueError, match='beta active space.*of 9 orbitals'):
        build_effective_hamiltonian(
            low, (alpha, ActiveSpace.from_indices(9, 4, [3], [4, 5])))

    rhf = water()
    one = ActiveSpace.from_counts(rhf.mo_energy, 5, 2, 2)
    with pytest.raises(ValueError, match='one ActiveSpace'):
        build_effective_hamiltonian(run_low_level(rhf, method='hf'), (one, one))


def test_unsupported_unrestricted_references(oh):
    # an open shell run restricted: the message names the way in
    with pytest.raises(NotImplementedError, match='UHF'):
        run_low_level(scf.ROHF(oh.mol).run(), method='hf')

    # an occupation that is not aufbau in one spin
    hole = copy.copy(oh)
    hole.mo_occ = oh.mo_occ.copy()
    hole.mo_occ[1][[3, 4]] = hole.mo_occ[1][[4, 3]]
    with pytest.raises(ValueError, match='beta'):
        run_low_level(hole, method='hf')

    # fractional occupations are not one determinant: counting them as
    # occupied would describe a system with another number of electrons
    half = copy.copy(oh)
    half.mo_occ = oh.mo_occ.copy()
    half.mo_occ[1][3:5] = 0.5
    for method in ('hf', 'rpa'):
        with pytest.raises(ValueError, match='beta.*0 or 1'):
            run_low_level(half, method=method)
    with pytest.raises(ValueError, match='beta.*0 or 1'):
        build_effective_hamiltonian(SimpleNamespace(mf=half, method='hf'),
                                    _none_active(oh))
    smeared = scf.addons.frac_occ(scf.UHF(oh.mol))
    smeared.verbose = 0
    smeared.kernel()
    assert np.any((smeared.mo_occ > 1e-3) & (smeared.mo_occ < 1 - 1e-3))
    with pytest.raises(ValueError, match='0 or 1'):
        run_low_level(smeared, method='rpa')

    # triplet NH: the pi pair is degenerate in both spins, occupied in alpha
    # (3, 4) and virtual in beta (3, 4)
    nh = unrestricted(NH, 2, basis='sto-3g')
    eps_a, eps_b = nh.mo_energy
    assert abs(eps_a[3] - eps_a[4]) < 1e-6 and abs(eps_b[3] - eps_b[4]) < 1e-6
    low = run_low_level(nh, method='hf')
    whole_a = ActiveSpace.from_indices(6, 5, [3, 4], [])
    whole_b = ActiveSpace.from_indices(6, 3, [], [3, 4])
    build_effective_hamiltonian(low, (whole_a, whole_b))
    with pytest.raises(ValueError, match='alpha active space cuts a degenerate'):
        build_effective_hamiltonian(
            low, (ActiveSpace.from_indices(6, 5, [2, 4], []), whole_b))
    with pytest.raises(ValueError, match='beta active space cuts a degenerate'):
        build_effective_hamiltonian(
            low, (whole_a, ActiveSpace.from_indices(6, 3, [2], [3])))


def test_unrestricted_from_counts(oh):
    """Per spin the N^A highest occupied and the n_act - N^A lowest virtual
    orbitals: pyscf UCASCI's convention."""
    eps, nocc = oh.mo_energy, _nocc(oh)
    alpha, beta = unrestricted_from_counts(eps, nocc, 3, (2, 1))
    assert (alpha.norb, alpha.nocc, beta.norb, beta.nocc) == (11, 5, 11, 4)
    assert np.array_equal(alpha.occ_active, [3, 4])
    assert np.array_equal(alpha.virt_active, [5])
    assert np.array_equal(beta.occ_active, [3])
    assert np.array_equal(beta.virt_active, [4, 5])
    alpha, beta = unrestricted_from_counts(eps, nocc, 2, (2, 0))
    assert alpha.n_act == beta.n_act == 2 and beta.n_act_occ == 0
    with pytest.raises(ValueError, match='beta.*3 active electrons in 2'):
        unrestricted_from_counts(eps, nocc, 2, (1, 3))
    with pytest.raises(ValueError, match='alpha'):
        unrestricted_from_counts(eps, nocc, 7, (6, 1))
    with pytest.raises(ValueError, match='per spin'):
        unrestricted_from_counts(eps[0], 5, 3, (2, 1))


def test_spin_overlap_in_space(oh):
    """space['spin_overlap'] holds the blocks of O = C_alpha^T S C_beta that
    <S^2> of the whole molecule's state needs; with only occupied orbitals
    active the one state is the UHF determinant and <S^2> is the UHF value
    (0.753774; measured difference 2.2e-16, energy 2.8e-14)."""
    spaces = _asymmetric(oh)
    ham = _hf(oh, spaces)
    space = ham.space
    overlap = oh.mo_coeff[0].T @ oh.get_ovlp() @ oh.mo_coeff[1]
    rows = {'core': spaces[0].env_occ, 'active': spaces[0].act}
    cols = {'core': spaces[1].env_occ, 'active': spaces[1].act}
    assert sorted(space['spin_overlap']) == ['active_active', 'active_core',
                                             'core_active', 'core_core']
    for left, i in rows.items():
        for right, j in cols.items():
            block = space['spin_overlap'][f'{left}_{right}']
            assert block.shape == (len(i), len(j))
            assert np.max(np.abs(block - overlap[np.ix_(i, j)])) < 1e-13
    assert [list(x) for x in space['act_idx']] == [[2, 4, 5, 7], [3, 4, 6, 7]]
    assert [list(x) for x in space['act_occ_idx']] == [[2, 4], [3]]
    assert [list(x) for x in space['act_vir_idx']] == [[5, 7], [4, 6, 7]]
    assert space['active_space'] == tuple(spaces)

    nmo = oh.mo_coeff.shape[-1]
    occupied = (ActiveSpace.from_indices(nmo, 5, [3, 4], []),
                ActiveSpace.from_indices(nmo, 4, [2, 3], []))
    ham = _hf(oh, occupied)
    g_eff, space = ham.g_eff, ham.space
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        e = _solve(g_eff, space, out=out, spin_overlap=space['spin_overlap'])
    assert abs(e[0] - oh.e_tot) < 1e-9
    assert abs(out['spin_square'][0] - oh.spin_square()[0]) < 1e-10


# --- the two-spin oracle itself ------------------------------------------------

def _spin_orbitals(mf, parts):
    """(eri_so, eps_so, occ_so, act_so) for the oracle's two-spin functions:
    alpha spin orbitals first, the active ones in the order of each
    partition (occupied first)."""
    mo = mf.mo_coeff if np.ndim(mf.mo_coeff) == 3 else (mf.mo_coeff,) * 2
    eps = mf.mo_energy if np.ndim(mf.mo_energy) == 2 else (mf.mo_energy,) * 2
    nmo = mo[0].shape[1]
    _, eri_so = spin_orbital_integrals(mf.mol, mo[0], mo[1])
    occ_so = (list(range(parts[0].nocc))
              + [nmo + i for i in range(parts[1].nocc)])
    act_so = list(parts[0].act) + [nmo + p for p in parts[1].act]
    return eri_so, np.concatenate(eps), occ_so, act_so


def test_oracle_reduces_to_the_restricted_one():
    """On a closed shell, with the same orbitals for both spins, the
    spin-orbital functions give the restricted ones: the direct RPA energy
    (the triplet modes add 1/2 (Omega - delta) = 0 each), W(0) - v, the cRPA
    and mRPA corrections in all three blocks, and the static environment
    self-energy of both spins in both forms. Measured: E_c 1.4e-13, W - v
    1.0e-15, cRPA 1.6e-15, mRPA 3.7e-14, Sigma 5.7e-16 (srg) and 1.6e-15
    (mode_a). A wrong factor in the two-spin functions shows at 6e-3 or
    more (delta + 4K, a pole-sum factor -4, mRPA with 1/4, swapped pole
    signs, Tr A with 2K; each tried)."""
    mf = water()
    eps, nocc, nmo = mf.mo_energy, 5, len(mf.mo_energy)
    _, eri = mo_integrals(mf)
    part = ActiveSpace.from_indices(nmo, nocc, [2, 4], [5, 7])
    so = _spin_orbitals(mf, (part, part))
    n = part.n_act

    e_c, omega, xpy, pairs = drpa_so(*so[:3])
    assert len(pairs) == 2 * nocc * (nmo - nocc) == len(omega)
    assert pairs[0] == (0, nocc) and pairs[len(pairs) // 2] == (nmo, nmo + nocc)
    assert abs(e_c - rpa_energy(eri, eps, nocc)) < 1e-10

    full = w0_minus_v_so(*so[:3], so[3])
    ref = w0_minus_v(eri, eps, nocc, part.act)
    for block in (full[:n, :n, :n, :n], full[:n, :n, n:, n:],
                  full[n:, n:, n:, n:]):
        assert np.max(np.abs(block - ref)) < 1e-10
    assert not full[:n, n:].any() and not full[:, :, :n, n:].any()

    ref = vtilde_crpa(eri, eps, nocc, part)
    for block in vtilde_crpa_uhf(*so):
        assert np.max(np.abs(block - ref)) < 1e-10

    ref_aa, ref_ab = vtilde_mrpa(eri, eps, nocc, part)
    aa, ab, bb = vtilde_mrpa_uhf(*so)
    assert np.max(np.abs(aa - ref_aa)) < 1e-10
    assert np.max(np.abs(ab - ref_ab)) < 1e-10
    assert np.max(np.abs(bb - ref_aa)) < 1e-10
    assert np.max(np.abs(aa)) > 1e-3

    for s in (DOWNFOLDING_SRG_S, None):
        ref = sigma_srg(eri, eps, nocc, part, s)
        for block in sigma_srg_uhf(*so, s):
            assert np.max(np.abs(block - ref)) < 1e-10
    ref = sigma_mode_a(eri, eps, nocc, part)
    for block in sigma_mode_a_uhf(*so):
        assert np.max(np.abs(block - ref)) < 1e-10


# --- method='rpa' on casida: the full RPA of both spins ------------------------

def _molecule_so(mf):
    """(eri_so, eps_so, occ_so) of the whole molecule for the oracle."""
    nmo = mf.mo_coeff.shape[-1]
    _, eri_so = spin_orbital_integrals(mf.mol, *mf.mo_coeff)
    n_a, n_b = _nocc(mf)
    occ_so = list(range(n_a)) + [nmo + i for i in range(n_b)]
    return eri_so, np.concatenate(mf.mo_energy), occ_so


@pytest.mark.parametrize('name', ['oh', 'ch2'])
@pytest.mark.parametrize('df', [False, True])
def test_ec_full_against_oracle(request, name, df):
    """The full direct RPA of an unrestricted reference, solved once on the
    stacked pairs of both spins (the alpha pairs in C order, then the beta
    ones): E_c, the excitation energies and X+Y against the oracle's
    spin-orbital RPA on exact integrals. df=True on a mean field without a
    density fit takes the eigendecomposition of the exact integrals. With
    the exact blocks handed to the solver in another order (the
    opposite-spin block as the beta-beta one) E_c of OH is -0.1360 and not
    -0.1057. Measured: E_c at most 5.7e-14, the excitation energies 1.4e-13,
    (A+B)^-1 5.4e-13 (58 pairs for OH, 70 for CH2)."""
    mf = request.getfixturevalue(name)
    nmo = mf.mo_coeff.shape[-1]
    low = run_low_level(mf, method='rpa', df=df)
    e_c, omega, xpy, pairs = drpa_so(*_molecule_so(mf))
    assert low.unrestricted and low.mode == 'casida'
    assert low.nocc == _nocc(mf) and len(low.eps) == 2
    assert abs(low.e_c_full - e_c) < 1e-9
    assert low.omega.shape == (len(pairs),)
    assert low.xpy.shape == (len(pairs),) * 2
    assert np.abs(np.sort(low.omega) - omega).max() < 1e-9
    # rows in the stacked pair order, and normalized so that
    # xpy diag(1/omega) xpy^T = (A+B)^-1 = (delta + 2K)^-1
    ours = (low.xpy / low.omega) @ low.xpy.T
    assert np.abs(ours - (xpy / omega) @ xpy.T).max() < 1e-9
    if df:
        assert low.eri is None and len(low.factors) == 2
        assert all(b.shape[1:] == (nmo, nmo) for b in low.factors)
        assert low.factors[0].shape[0] == low.factors[1].shape[0]
    else:
        assert low.factors is None and len(low.eri) == 3
        assert all(v.shape == (nmo,) * 4 for v in low.eri)


def test_ec_full_against_pyscf_urpa():
    """On a density-fitted UHF the plasmon-formula E_c on the mean field's
    own fit against pyscf's URPA, an imaginary-frequency quadrature on the
    same with_df object (80 points). Measured: 1.5e-12 (7e-10 with pyscf's
    default 40 points)."""
    mf = unrestricted(OH, 1, density_fit=True)
    low = run_low_level(mf, method='rpa')
    ref = urpa.URPA(mf)
    ref.verbose = 0
    ref.kernel(nw=80)
    assert abs(low.e_c_full - ref.e_corr) < 1e-9


def test_a_restricted_run_is_one_channel():
    low = run_low_level(water(), method='rpa')
    assert not low.unrestricted
    assert isinstance(low.nocc, int) and low.eps.ndim == 1


# --- the screening correction of an active space ------------------------------

@pytest.fixture(scope='module')
def oh_low(oh):
    """The casida low-level results of OH on the exact four-index blocks
    (df=False) and on three-index factors (df=True). This UHF has no density
    fit, so those factors are the eigendecomposition of the exact integrals:
    the same v in another form, without a fitting error. A density-fitted
    UHF is `test_density_fitted_uhf_through_the_driver`."""
    return {df: run_low_level(oh, method='rpa', df=df) for df in (False, True)}


def _ph_mask(first, second):
    """True on (tu|vw) where t, u are one occupied and one virtual active
    orbital of the first partition and v, w of the second, written out."""
    n = first.n_act
    mask = np.zeros((n,) * 4, dtype=bool)
    for t in range(n):
        for u in range(n):
            for v in range(n):
                for w in range(n):
                    mask[t, u, v, w] = (
                        (t < first.n_act_occ) != (u < first.n_act_occ)
                        and (v < second.n_act_occ) != (w < second.n_act_occ))
    return mask


@pytest.mark.parametrize('space', sorted(SPACES))
@pytest.mark.parametrize('df', [False, True])
def test_vtilde_against_oracle(oh, oh_low, space, df):
    """The cRPA correction of an unrestricted reference, three blocks
    (aa, ab, bb): the active pairs of both spins deleted from the stacked
    delta + 2K and one Cholesky solve, against the oracle's pole sum of the
    deleted problem in spin orbitals. crpa_ph keeps the elements with a
    particle-hole pair of each side's own spin on both sides; 'bare' is
    zero. The spaces have different orbitals per spin, N^A_alpha !=
    N^A_beta, unequal cores, a channel without an active pair, and no
    active pair at all (nothing deleted: the fully screened W(0) - v).
    Measured: at most 5.3e-15 over the four spaces and both df settings,
    on blocks of size 0.09 to 0.16 whose aa and bb differ by 0.05 to 0.14."""
    low = oh_low[df]
    parts = _parts(oh, SPACES[space](oh))
    n = parts[0].n_act
    so = _spin_orbitals(oh, parts)
    ref = vtilde_crpa_uhf(*so)
    blocks = rpa.screening_correction(low, parts, 'crpa')
    assert isinstance(blocks, tuple) and len(blocks) == 3
    for ours, exact in zip(blocks, ref):
        assert ours.shape == (n,) * 4
        assert np.max(np.abs(ours - exact)) < 1e-9
        assert np.max(np.abs(ours - ours.transpose(1, 0, 2, 3))) < 1e-10
    for same in (blocks[0], blocks[2]):
        assert np.max(np.abs(same - same.transpose(2, 3, 0, 1))) < 1e-10
    assert np.max(np.abs(blocks[0] - blocks[2])) > 1e-3

    masks = (_ph_mask(parts[0], parts[0]), _ph_mask(parts[0], parts[1]),
             _ph_mask(parts[1], parts[1]))
    assert np.array_equal(ph_element_mask(parts[0], parts[1]), masks[1])
    ph = rpa.screening_correction(low, parts, 'crpa_ph')
    for ours, exact, mask in zip(ph, ref, masks):
        assert np.max(np.abs(ours - exact * mask)) < 1e-9
        assert not ours[~mask].any()
    bare = rpa.screening_correction(low, parts, 'bare')
    assert len(bare) == 3 and not any(b.any() for b in bare)

    if space == 'occupied_only':
        full = spin_blocks(w0_minus_v_so(*so[:3], so[3]), n)
        for ours, exact in zip(blocks, full):
            assert np.max(np.abs(ours - exact)) < 1e-9
        assert not any(b.any() for b in ph)
    if space == 'no_beta_pair':
        assert np.max(np.abs(ph[0])) > 1e-3
        assert not ph[1].any() and not ph[2].any()


def test_vtilde_triplet_ch2(ch2):
    """Triplet CH2 on its density-fit-free factors: N^A = (2, 1) in 3
    orbitals, cores (3, 2). Measured: 2.3e-15."""
    nmo = ch2.mo_coeff.shape[-1]
    spaces = (ActiveSpace.from_indices(nmo, 5, [3, 4], [5]),
              ActiveSpace.from_indices(nmo, 3, [2], [3, 4]))
    parts = _parts(ch2, spaces)
    low = run_low_level(ch2, method='rpa')
    ref = vtilde_crpa_uhf(*_spin_orbitals(ch2, parts))
    for ours, exact in zip(rpa.screening_correction(low, parts, 'crpa'), ref):
        assert np.max(np.abs(ours - exact)) < 1e-9


def test_all_active_vtilde_is_zero(h2_bs):
    """Every orbital of both spins active: no pair is left to screen."""
    parts = _parts(h2_bs, _all_active(h2_bs))
    low = run_low_level(h2_bs, method='rpa')
    for screening in ('crpa', 'crpa_ph', 'bare'):
        blocks = rpa.screening_correction(low, parts, screening)
        assert len(blocks) == 3 and not any(b.any() for b in blocks)


def test_pair_factor_functions_need_the_three_index_factors(oh, oh_low):
    """The exact four-index blocks (df=False) hold no three-index factors:
    active_pair_factors and active_polarizability_static refuse such a
    result by name, and take the one on factors."""
    parts = _parts(oh, _asymmetric(oh))
    for function in (rpa.active_pair_factors,
                     rpa.active_polarizability_static):
        with pytest.raises(NotImplementedError,
                           match='three-index factors.*df=True'):
            function(oh_low[False], parts)
    b_alpha, b_beta = rpa.active_pair_factors(oh_low[True], parts)
    assert b_alpha.shape[1:] == b_beta.shape[1:] == (4, 4)


def test_a_mean_field_changed_after_the_run_is_refused():
    """As for a restricted reference: a UHF object converged again at
    another bond length does not go with the low-level result of the first
    one, and the build is refused."""
    mf = unrestricted(OH, 1)
    spaces = _unequal_cores(mf)
    low = run_low_level(mf, method='rpa')
    build_effective_hamiltonian(low, spaces)
    mf.mol.set_geom_('O 0 0 0; H 0 0 1.1')
    mf.kernel()
    with pytest.raises(ValueError,
                       match='mean field has changed.*run_low_level'):
        build_effective_hamiltonian(low, spaces)


def test_partition_form_must_fit_the_low_level_result(oh, oh_low):
    """The functions of the RPA method take one ActiveSpace with a restricted
    result and the pair (alpha, beta) with an unrestricted one."""
    parts = _parts(oh, _asymmetric(oh))
    with pytest.raises(ValueError, match='pair'):
        rpa.screening_correction(oh_low[True], parts[0], 'crpa')
    rhf = water()
    one = ActiveSpace.from_counts(rhf.mo_energy, 5, 2, 2)
    low = run_low_level(rhf, method='rpa')
    with pytest.raises(ValueError, match='one ActiveSpace'):
        rpa.screening_correction(low, (one, one), 'crpa')
    assert isinstance(rpa.screening_correction(low, one, 'crpa'), np.ndarray)


# --- E_c^A[v_eff] and the static defect ---------------------------------------

def _local_so(blocks):
    """The chemist array over 2n local spin orbitals, alpha first, of the
    blocks (aa, ab, bb)."""
    aa, ab, bb = blocks
    n = aa.shape[0]
    out = np.zeros((2 * n,) * 4)
    out[:n, :n, :n, :n], out[n:, n:, n:, n:] = aa, bb
    out[:n, :n, n:, n:] = ab
    out[n:, n:, :n, :n] = ab.transpose(2, 3, 0, 1)
    return out


def _oracle_v_eff(so, parts, screening):
    """(aa, ab, bb) of v + vtilde from the oracle alone: the exact active
    integrals plus its cRPA correction (masked for crpa_ph, none for bare)."""
    eri_so, _, _, act_so = so
    n = parts[0].n_act
    v = spin_blocks(eri_so[np.ix_(act_so, act_so, act_so, act_so)], n)
    if screening == 'bare':
        return tuple(np.array(x) for x in v)
    vt = vtilde_crpa_uhf(*so)
    if screening == 'crpa_ph':
        masks = (_ph_mask(parts[0], parts[0]), _ph_mask(parts[0], parts[1]),
                 _ph_mask(parts[1], parts[1]))
        vt = tuple(x * mask for x, mask in zip(vt, masks))
    return tuple(x + y for x, y in zip(v, vt))


def _oracle_active_rpa(so, v_eff):
    """(E_c^A, defect) from the oracle: the direct RPA of the active spin
    orbitals alone with v_eff, and the largest |W(0) - W^A[v_eff](0)| over
    the three blocks."""
    eri_so, eps_so, occ_so, act_so = so
    m = len(act_so)
    v_so = _local_so(v_eff)
    eps_act = np.asarray(eps_so)[act_so]
    occ_act = [x for x, q in enumerate(act_so) if q in occ_so]
    e_c = drpa_so(v_so, eps_act, occ_act)[0]
    w_full = (eri_so[np.ix_(act_so, act_so, act_so, act_so)]
              + w0_minus_v_so(eri_so, eps_so, occ_so, act_so))
    w_act = v_so + w0_minus_v_so(v_so, eps_act, occ_act, list(range(m)))
    defect = max(np.max(np.abs(x - y)) for x, y in zip(
        spin_blocks(w_full, m // 2), spin_blocks(w_act, m // 2)))
    return e_c, defect


@pytest.mark.parametrize('space', sorted(SPACES))
@pytest.mark.parametrize('screening', ['bare', 'crpa', 'crpa_ph'])
@pytest.mark.parametrize('df', [False, True])
def test_ec_active_and_defect(oh, oh_low, space, screening, df):
    """E_c^A[v_eff], the direct RPA of the active pairs of both spins with
    K^A = the particle-hole blocks of v_eff, and the static defect, the
    largest |W(0) - W^A[v_eff](0)| over the three blocks, against the
    oracle's RPA of the active spin orbitals alone. The crpa defect is zero.
    With no active pair in a channel the stacked problem holds the other
    spin's pairs; with none in either, E_c^A = 0 and W^A = v_eff. Measured
    over the four spaces, three screenings and both df settings: E_c^A to
    8.9e-16 (values -9.2e-3 to -6.0e-5), the defect to 2.4e-14 (0.08 to
    0.16 for bare and crpa_ph, at most 2.6e-14 for crpa)."""
    low = oh_low[df]
    parts = _parts(oh, SPACES[space](oh))
    so = _spin_orbitals(oh, parts)
    v_eff = _oracle_v_eff(so, parts, screening)
    e_ref, defect_ref = _oracle_active_rpa(so, v_eff)
    e_c = rpa.active_correlation_energy(low, parts, v_eff)
    defect = rpa.static_defect(low, parts, v_eff)
    assert isinstance(e_c, float) and isinstance(defect, float)
    assert abs(e_c - e_ref) < 1e-9
    assert abs(defect - defect_ref) < 1e-9
    if screening == 'crpa':
        assert defect < 1e-10
    else:
        assert defect > 1e-3
    if space == 'occupied_only':
        assert e_c == 0.0
    else:
        assert e_c < -1e-5


@pytest.mark.parametrize('name', ['h2_bs', 'oh_sto3g'])
def test_all_active_ec_pair_cancels(request, name):
    """Every orbital of both spins active: E_c^A[v] on the v of the full RPA
    is E_c[v] itself, and the static defect is zero. Measured: the pair
    cancels to 4.3e-14, the defect is at most 8.3e-15 (E_c = -0.0080 for
    H2, -0.0307 for OH/STO-3G)."""
    mf = (unrestricted(OH, 1, basis='sto-3g') if name == 'oh_sto3g'
          else request.getfixturevalue(name))
    parts = _parts(mf, _all_active(mf))
    so = _spin_orbitals(mf, parts)
    v = _oracle_v_eff(so, parts, 'bare')
    for df in (False, True):
        low = run_low_level(mf, method='rpa', df=df)
        e_c = rpa.active_correlation_energy(low, parts, v)
        assert abs(e_c - low.e_c_full) < 1e-10 and e_c < -1e-3
        assert rpa.static_defect(low, parts, v) < 1e-10


def test_two_body_blocks_must_fit_the_reference(oh, oh_low):
    parts = _parts(oh, _asymmetric(oh))
    so = _spin_orbitals(oh, parts)
    v_eff = _oracle_v_eff(so, parts, 'crpa')
    for function in (rpa.active_correlation_energy, rpa.static_defect):
        with pytest.raises(ValueError, match='three blocks'):
            function(oh_low[True], parts, v_eff[0])
        with pytest.raises(ValueError, match='three blocks'):
            function(oh_low[True], parts, v_eff[:2])


# --- the environment self-energy Sigma_bar^E, per spin -------------------------

def _no_core(mf):
    """OH/6-31G: every occupied orbital of both spins active, so no inactive
    orbital; N^A = (5, 4) in 6 orbitals."""
    nmo = mf.mo_coeff.shape[-1]
    return (ActiveSpace.from_indices(nmo, 5, range(5), [5]),
            ActiveSpace.from_indices(nmo, 4, range(4), [4, 5]))


@pytest.mark.parametrize('space', ['asymmetric', 'unequal_cores',
                                   'no_beta_pair', 'beta_lumo_outside'])
@pytest.mark.parametrize('df', [False, True])
def test_residues_against_oracle(oh, oh_low, space, df):
    """The residues of W between the active and the environment orbitals of
    each spin, w^s[n, t, r] = sum over the pairs of BOTH spins of
    (t_s r_s|i a) (X+Y)_ia,n, with no spin factor (g = 1). The modes of two
    eigensolvers can differ by sign and by rotations among degenerate ones,
    so the comparison is on sum_n w[n,t,r] w[n,u,s] / Omega_n, between the
    same spin and between the two spins, against the oracle's. Measured:
    5.4e-15 over three spaces and both df settings."""
    low = oh_low[df]
    parts = _parts(oh, SPACES[space](oh))
    nmo = oh.mo_coeff.shape[-1]
    so = _spin_orbitals(oh, parts)
    omega, w_ref, _, env = env_residues_uhf(*so)
    ours = rpa.screened_residues(low, parts)
    assert isinstance(ours, tuple) and len(ours) == 2
    n = parts[0].n_act
    cols = [[x for x, r in enumerate(env) if (r >= nmo) == bool(spin)]
            for spin in (0, 1)]
    for spin, part in enumerate(parts):
        assert ours[spin].shape == (len(low.omega), n, len(part.env))
        assert [env[x] - spin * nmo for x in cols[spin]] == list(part.env)
    rows = [slice(0, n), slice(n, 2 * n)]
    for first in (0, 1):
        for second in (0, 1):
            got = np.einsum('ntr,nus->trus', ours[first] / low.omega[:, None, None],
                            ours[second])
            ref = np.einsum('ntr,nus->trus',
                            w_ref[:, rows[first]][:, :, cols[first]]
                            / omega[:, None, None],
                            w_ref[:, rows[second]][:, :, cols[second]])
            assert np.max(np.abs(got - ref)) < 1e-9
            assert np.max(np.abs(ref)) > 1e-3
    # any orbitals per spin, here every orbital
    every = rpa.screened_residues(low, parts,
                                  orbitals=(np.arange(nmo), np.arange(nmo)))
    for spin, part in enumerate(parts):
        assert every[spin].shape == (len(low.omega), n, nmo)
        assert np.max(np.abs(every[spin][:, :, part.env] - ours[spin])) < 1e-12


@pytest.mark.parametrize('space', ['asymmetric', 'unequal_cores',
                                   'no_beta_pair', 'beta_lumo_outside'])
@pytest.mark.parametrize('df', [False, True])
def test_sigma_against_oracle(oh, oh_low, space, df):
    """Sigma_bar^E of each spin, with that spin's orbital energies and the
    Omega_n both spins share: the SRG form (Marie and Loos Eq. 44) and mode
    A (Eq. 12 at eta = 0) against the oracle's pole sums in spin orbitals.
    Measured over three spaces and both df settings: 'srg' 4.9e-15, 'mode_a'
    1.9e-13, on elements up to 0.02 Ha whose alpha and beta blocks differ by
    0.01 to 0.03 Ha."""
    low = oh_low[df]
    parts = _parts(oh, SPACES[space](oh))
    so = _spin_orbitals(oh, parts)
    srg = rpa.embedding_self_energy(low, parts, 'srg')
    assert isinstance(srg, tuple) and len(srg) == 2
    for ours, ref in zip(srg, sigma_srg_uhf(*so, DOWNFOLDING_SRG_S)):
        assert ours.shape == (parts[0].n_act,) * 2
        assert np.max(np.abs(ours - ref)) < 1e-9
        assert np.max(np.abs(ours - ours.T)) < 1e-12
    if space != 'no_beta_pair':
        assert np.max(np.abs(srg[0] - srg[1])) > 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rpa.SigmaNearPoleWarning)
        mode_a = rpa.embedding_self_energy(low, parts, 'mode_a')
    for ours, ref in zip(mode_a, sigma_mode_a_uhf(*so)):
        assert np.max(np.abs(ours - ref)) < 1e-9


def test_pole_window_per_spin(oh, oh_low):
    """The pole-free window of each spin, (max eps_I - Omega_min,
    min eps_Ev + Omega_min) with that spin's energies and the one Omega_min
    of the two-spin problem; a side without orbitals is open."""
    low = oh_low[True]
    om = drpa_so(*_molecule_so(oh))[1].min()
    assert abs(rpa.lowest_excitation(low) - om) < 1e-9
    for make in (_asymmetric, _no_core):
        parts = _parts(oh, make(oh))
        windows = rpa.pole_window(low, parts)
        assert len(windows) == 2
        for (lo, hi), part, eps in zip(windows, parts, oh.mo_energy):
            if len(part.env_occ):
                assert abs(lo - (eps[part.env_occ].max() - om)) < 1e-9
            else:
                assert lo == -np.inf
            assert abs(hi - (eps[part.env_virt].min() + om)) < 1e-9
    assert windows[0][1] != windows[1][1]
    assert rpa.pole_window(low, _parts(oh, _all_active(oh))) == (
        (-np.inf, np.inf), (-np.inf, np.inf))


def test_near_pole_warning_names_the_spin(oh, oh_low):
    """'mode_a' warns of the active orbitals near a pole, naming their spin.
    In the asymmetric space the beta orbitals 6 and 7 lie above the beta
    window (distances -0.20 and -0.27 Ha) and every alpha orbital is at
    least 0.64 Ha inside the alpha one. 'srg' is finite everywhere and does
    not warn; with no orbital near a pole neither does 'mode_a'."""
    low = oh_low[True]
    parts = _parts(oh, _asymmetric(oh))
    with pytest.warns(rpa.SigmaNearPoleWarning,
                      match=r'beta orbital 6 \(.*beta orbital 7 \(') as rec:
        rpa.embedding_self_energy(low, parts, 'mode_a')
    assert rec[0].filename == __file__
    text = str(rec[0].message)
    assert 'alpha orbital' not in text
    assert 'windows alpha (' in text and ' and beta (' in text
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rpa.embedding_self_energy(low, parts, 'srg')
        rpa.embedding_self_energy(low, _parts(oh, _unequal_cores(oh)),
                                  'mode_a')


def test_a_sigma_that_is_not_finite_names_the_spin(oh, oh_low, monkeypatch):
    """A Sigma_bar with an element that is not finite is refused, naming
    the spin of the orbital pair."""
    parts = _parts(oh, _unequal_cores(oh))
    exact, calls = rpa.mode_a_self_energy, []

    def with_a_pole(*args):
        out = exact(*args)
        calls.append(1)
        if len(calls) == 2:                      # the beta channel
            out[-1, -1] = np.inf
        return out

    monkeypatch.setattr(rpa, 'mode_a_self_energy', with_a_pole)
    act = parts[1].act
    with pytest.raises(FloatingPointError,
                       match=rf"\({act[-1]}, {act[-1]}\) \(beta orbital "
                             rf"pairs\).*sigma='srg'"):
        rpa.embedding_self_energy(oh_low[True], parts, 'mode_a')


# --- the driver with method='rpa' (casida) on an unrestricted reference --------

SCREENINGS_RPA = ('bare', 'crpa', 'crpa_ph')
SIGMAS_RPA = (None, 'srg', 'mode_a')
ENERGY_TERMS = ('e_nuc', 'e_core', 'e_hartree', 'e_exchange', 'e_hartree_dc',
                'e_exchange_dc', 'e_c_full', 'e_c_active', 'e_exchange_term',
                'e_sigma_trace', 'e0')


def _quiet(function, *args, **kwargs):
    """`function` with the near-pole and damping warnings of Sigma^E off."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rpa.SigmaNearPoleWarning)
        warnings.simplefilter('ignore', rpa.SigmaDampingWarning)
        return function(*args, **kwargs)


@pytest.fixture(scope='module')
def closed_shell():
    """Water/6-31G as RHF and as the same determinant in a UHF object, with
    the casida low-level results of both on the exact four-index integrals
    (df=False) and on their exact three-index factors (df=True; the mean
    field has no density fit)."""
    rhf = water()
    uhf = rhf.to_uhf()
    lows = {df: (run_low_level(rhf, method='rpa', df=df),
                 run_low_level(uhf, method='rpa', df=df))
            for df in (False, True)}
    return rhf, lows


@pytest.mark.parametrize('df', [False, True])
@pytest.mark.parametrize('sigma', SIGMAS_RPA)
@pytest.mark.parametrize('screening', SCREENINGS_RPA)
def test_closed_shell_limit_rpa(closed_shell, screening, sigma, df):
    """A closed-shell determinant through the two-channel code gives the
    restricted H_eff: h_alpha = h_beta, the three two-body blocks equal,
    every energy term, the static defect and the FCI energy. Not compared:
    the pole-free window, which is narrower for two channels (its Omega_min
    includes the spin-density modes: 0.7050 against the singlet 0.7278 Ha).
    Measured over the 18 cases: h 5.4e-15, v 1.3e-15, the energy terms and
    the FCI energy 1.1e-13, the static defect 3.4e-15."""
    rhf, lows = closed_shell
    low_r, low_u = lows[df]
    one = ActiveSpace.from_indices(len(rhf.mo_energy), 5, [2, 4], [5, 7])
    ham_r = _quiet(build_effective_hamiltonian, low_r, one,
                   screening=screening, sigma=sigma)
    g_r, en_r = ham_r.g_eff, ham_r.energies
    ham_u = _quiet(build_effective_hamiltonian, low_u, (one, one),
                   screening=screening, sigma=sigma)
    g_u, space, en_u = ham_u.g_eff, ham_u.space, ham_u.energies
    assert abs(low_u.e_c_full - low_r.e_c_full) < 1e-10
    for h in g_u[1]:
        assert np.max(np.abs(h - g_r[1])) < 1e-10
    for v in g_u[2]:
        assert np.max(np.abs(v - g_r[2])) < 1e-10
    for name in ENERGY_TERMS:
        assert abs(getattr(en_u, name) - getattr(en_r, name)) < 1e-10, name
    assert abs(ham_u.diagnostics['static_defect']
               - ham_r.diagnostics['static_defect']) < 1e-10
    e_r = solve('pyscf_fci', g_r, 4, 4, spin_free=True)[0]
    assert abs(_solve(g_u, space)[0] - e_r) < 1e-9
    if sigma is not None:
        wide, narrow = (ham_r.diagnostics['sigma_poles']['window'],
                        ham_u.diagnostics['sigma_poles']['window'])
        assert narrow[0] == narrow[1]
        assert narrow[0][0] > wide[0] and narrow[0][1] < wide[1]


@pytest.mark.parametrize('df', [False, True])
@pytest.mark.parametrize('sigma', SIGMAS_RPA)
@pytest.mark.parametrize('screening', SCREENINGS_RPA)
@pytest.mark.parametrize('space', sorted(SPACES))
def test_reference_determinant_identity(oh, oh_low, space, screening, sigma,
                                        df):
    """The reference determinant of H_eff has the energy E_UHF + E_c[v] -
    E_c^A[v_eff], whatever the screening and with or without Sigma^E: the
    double counting of vtilde and of Sigma_bar in h_eff is undone in E_0,
    per spin. method='hf' has vtilde = 0 and never exercises it. Measured:
    7.1e-14 over the 72 builds, with Tr[Sigma_bar gamma] of 0.02 to 0.07 Ha
    where Sigma^E is on."""
    spaces = SPACES[space](oh)
    ham = _quiet(build_effective_hamiltonian, oh_low[df], spaces,
                 screening=screening, sigma=sigma)
    g_eff, sp, en = ham.g_eff, ham.space, ham.energies
    assert len(g_eff[1]) == 2 and len(g_eff[2]) == 3
    assert en.e_c_full == oh_low[df].e_c_full
    assert abs(reference_energy_uhf(g_eff, sp['n_elec_active'])
               - (oh.e_tot + en.e_c_full - en.e_c_active)) < 1e-9
    assert abs(g_eff[0] - en.e0) == 0.0
    if sigma is None:
        assert en.e_sigma_trace == 0.0 and en.e_exchange_term == 0.0
    else:
        assert abs(en.e_sigma_trace) > 1e-4


@pytest.mark.parametrize('space', ['asymmetric', 'beta_lumo_outside'])
@pytest.mark.parametrize('screening', ['crpa', 'crpa_ph'])
@pytest.mark.parametrize('sigma', [None, 'srg'])
def test_h_eff_against_oracle(oh, oh_low, screening, sigma, space):
    """Every block of H_eff against the oracle alone, for the asymmetric
    space and for the one with the beta LUMO in the environment:
    h^s = f^s - t^DC,s (+ Sigma_bar^s) from the spin-orbital one-body
    formulas with the oracle's vtilde, and v_eff = v + vtilde. Measured:
    one-body 1.4e-14, two-body 4.7e-15."""
    spaces = SPACES[space](oh)
    parts = _parts(oh, spaces)
    so = _spin_orbitals(oh, parts)
    eri_so, _, _, act_so = so
    nmo, n = oh.mo_coeff.shape[-1], parts[0].n_act
    ham = _quiet(build_effective_hamiltonian, oh_low[True], spaces,
                 screening=screening, sigma=sigma)
    g_eff, sp = ham.g_eff, ham.space
    vt = vtilde_crpa_uhf(*so)
    if screening == 'crpa_ph':
        vt = tuple(x * m for x, m in zip(vt, (
            _ph_mask(parts[0], parts[0]), _ph_mask(parts[0], parts[1]),
            _ph_mask(parts[1], parts[1]))))
    v = spin_blocks(eri_so[np.ix_(act_so, act_so, act_so, act_so)], n)
    for ours, bare, corr in zip(g_eff[2], v, vt):
        assert np.max(np.abs(ours.transpose(0, 2, 1, 3) - bare - corr)) < 1e-9
    h_so, _ = spin_orbital_integrals(oh.mol, *oh.mo_coeff)
    occ_env = list(spaces[0].env_occ) + list(nmo + spaces[1].env_occ)
    occ_act = list(range(sp['n_elec_active'][0])) + [
        n + k for k in range(sp['n_elec_active'][1])]
    f, t_dc = spin_orbital_one_body(
        h_so, eri_so, occ_env, act_so, _local_so(vt), occ_act,
        'full' if sigma is None else 'hartree')
    ref = f - t_dc
    sigma_bar = (np.zeros((n, n)),) * 2 if sigma is None else sigma_srg_uhf(
        *so, DOWNFOLDING_SRG_S)
    for spin, block in enumerate((slice(0, n), slice(n, 2 * n))):
        assert np.max(np.abs(g_eff[1][spin] - ref[block, block]
                             - sigma_bar[spin])) < 1e-9


@pytest.mark.parametrize('name', ['h2_bs', 'oh_sto3g'])
def test_all_active_is_fci_rpa(request, name):
    """Every orbital of both spins active: vtilde = 0, Sigma_bar = 0, the
    E_c pair cancels and H_eff gives the FCI energy of the molecule (from
    pyscf in restricted orbitals), for every screening and Sigma^E form.
    Measured: 0 for the pair (the active problem is the full one, matrix
    for matrix) and for the FCI energy; E_c = -0.0080 (H2), -0.0307 (OH)."""
    if name == 'oh_sto3g':
        mf = unrestricted(OH, 1, basis='sto-3g')
        restricted = scf.ROHF(mf.mol)
    else:
        mf = request.getfixturevalue(name)
        restricted = scf.RHF(mf.mol)
    restricted.conv_tol = 1e-11
    restricted.verbose = 0
    restricted.kernel()
    e_fci = fci.FCI(restricted).kernel()[0]
    low = run_low_level(mf, method='rpa')
    assert low.e_c_full < -1e-3
    ham_hf = _hf(mf, _all_active(mf))
    hf = ham_hf.g_eff
    for screening in SCREENINGS_RPA:
        for sigma in SIGMAS_RPA:
            ham = build_effective_hamiltonian(
                low, _all_active(mf), screening=screening, sigma=sigma)
            g_eff, space, en = ham.g_eff, ham.space, ham.energies
            assert abs(en.ec) < 1e-10 and abs(en.e0 - mf.energy_nuc()) < 1e-10
            for ours, bare in zip(g_eff[1] + g_eff[2], hf[1] + hf[2]):
                assert np.max(np.abs(ours - bare)) < 1e-10
            assert abs(_solve(g_eff, space)[0] - e_fci) < 1e-9


def test_none_active_is_uhf_plus_ec(oh, oh_low):
    """No orbital active: E_0 = E_UHF + E_c[v] of the two-spin RPA.
    Measured: 1.4e-14."""
    for df in (False, True):
        low = oh_low[df]
        for sigma in SIGMAS_RPA:
            ham = build_effective_hamiltonian(
                low, _none_active(oh), sigma=sigma)
            g_eff, space, en = ham.g_eff, ham.space, ham.energies
            assert abs(en.e0 - (oh.e_tot + low.e_c_full)) < 1e-10
            assert en.e_c_active == 0.0 and space['n_elec_active'] == (0, 0)
            assert [h.shape for h in g_eff[1]] == [(0, 0)] * 2


def test_reference_diagnostics_are_flags_of_the_result(h2_bs, monkeypatch):
    """A spin-contaminated or internally unstable UHF determinant is taken,
    with a warning from run_low_level and the named flags
    'uhf_spin_contaminated' / 'uhf_reference_unstable' in
    low.numerical_flags, for every method and for every later run on the
    same mean field, and in the energies of every Hamiltonian built on it.
    Broken-symmetry H2 has <S^2> = 0.98 against 0; the unstable case is a
    stood-in stability analysis on OH. A clean reference has no flag."""
    mf = h2_broken()
    with pytest.warns(RuntimeWarning, match='Downfolding: the unrestricted '
                      'reference is spin-contaminated.*its effective '
                      'Hamiltonians describe') as rec:
        low = run_low_level(mf, method='rpa')
    assert rec[0].filename == __file__
    lows = [low]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lows += [run_low_level(mf, method='hf'),
                 run_low_level(mf, method='rpa', df=False)]
        for low in lows:
            assert [entry['flag'] for entry in low.numerical_flags] == [
                'uhf_spin_contaminated']
            ham = build_effective_hamiltonian(low, _all_active(mf))
            en = ham.energies
            assert 'uhf_spin_contaminated' in [
                entry['flag'] for entry in ham.numerical_flags]

    oh = unrestricted(OH, 1)
    monkeypatch.setattr(pyscf_interface.scf_stability, 'uhf_internal',
                        lambda mf, return_status, verbose: (None, False))
    with pytest.warns(RuntimeWarning, match='internally UNSTABLE.*the '
                      'effective Hamiltonians start from a saddle point'):
        low = run_low_level(oh, method='rpa')
    assert [entry['flag'] for entry in low.numerical_flags] == [
        'uhf_reference_unstable']
    monkeypatch.undo()

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        clean = run_low_level(unrestricted(OH, 1), method='rpa')
    assert clean.numerical_flags == []
    assert clean.mf.reference_diagnostics['spin_contaminated'] is False


def test_uks_reference():
    """A UKS reference (OH, PBE0): the Kohn-Sham orbitals and orbital
    energies go into the RPA, and E_0 is built on the HF energy expression
    of the Kohn-Sham determinant, not on the KS energy. With no orbital
    active E_0 is that expression plus E_c; a crpa build has the oracle's
    v_eff on the KS orbitals and the reference-determinant identity.
    Measured: E_c = -0.1364 to 2.0e-13, E_0 to 1.8e-13 (the HF expression
    is 0.265 Ha above the KS energy), v_eff 4.8e-15, the identity 1.4e-14,
    the crpa defect 3.1e-14."""
    uks = unrestricted(OH, 1, xc='pbe0')
    e_hf = scf.UHF(uks.mol).energy_tot(dm=uks.make_rdm1())
    low = run_low_level(uks, method='rpa', df=False)
    e_c = drpa_so(*_molecule_so(uks))[0]
    assert abs(low.e_c_full - e_c) < 1e-9
    ham = build_effective_hamiltonian(low, _none_active(uks))
    en = ham.energies
    assert abs(en.e0 - (e_hf + e_c)) < 1e-9
    assert abs(e_hf - uks.e_tot) > 1e-2

    spaces = _unequal_cores(uks)
    parts = _parts(uks, spaces)
    so = _spin_orbitals(uks, parts)
    ham = build_effective_hamiltonian(low, spaces, screening='crpa')
    g_eff, sp, en = ham.g_eff, ham.space, ham.energies
    eri_so, _, _, act_so = so
    v = spin_blocks(eri_so[np.ix_(act_so, act_so, act_so, act_so)],
                    parts[0].n_act)
    for ours, bare, corr in zip(g_eff[2], v, vtilde_crpa_uhf(*so)):
        assert np.max(np.abs(ours.transpose(0, 2, 1, 3) - bare - corr)) < 1e-9
    assert abs(reference_energy_uhf(g_eff, sp['n_elec_active'])
               - (e_hf + en.e_c_full - en.e_c_active)) < 1e-9
    assert ham.diagnostics['static_defect'] < 1e-10


def test_diagnostics_and_flags_for_two_spins(oh, oh_low):
    """The records of an unrestricted build: one number for a summary of the
    whole (the static defect, the SRG damping, the largest over both spins),
    a pair (alpha, beta) for what belongs to one spin (the pole-free window,
    the pole distances, Z). The near-pole warning of 'mode_a' points at the
    caller, names the spin, and is in energies.numerical_flags; so does the
    damping warning of 'srg', whose number (2.5e-3 Ha here, on beta orbital
    6) equals the oracle's |Sigma(s) - Sigma(inf)| to 1.3e-14."""
    low = oh_low[True]
    spaces = _asymmetric(oh)
    parts = _parts(oh, spaces)
    so = _spin_orbitals(oh, parts)
    n = parts[0].n_act
    with pytest.warns(rpa.SigmaNearPoleWarning, match='beta orbital 6') as rec:
        ham = build_effective_hamiltonian(low, spaces, sigma='mode_a')
        en = ham.energies
    near = [w for w in rec if w.category is rpa.SigmaNearPoleWarning]
    assert len(near) == 1 and near[0].filename == __file__
    assert {'flag': 'sigma_near_pole',
            'message': str(near[0].message)} in ham.numerical_flags
    poles = ham.diagnostics['sigma_poles']
    assert isinstance(ham.diagnostics['static_defect'], float)
    assert poles['window'] == rpa.pole_window(low, parts)
    assert len(poles['pole_distance']) == len(poles['z']) == 2
    for dist, z in zip(poles['pole_distance'], poles['z']):
        assert dist.shape == z.shape == (n,)
        assert np.all((z > 0) & (z <= 1))
    assert poles['pole_distance'][1][2] < 0 < poles['pole_distance'][0].min()
    for dist, (lo, hi), part, eps in zip(poles['pole_distance'],
                                         poles['window'], parts,
                                         oh.mo_energy):
        own = np.minimum(eps[part.act] - lo, hi - eps[part.act])
        assert np.array_equal(dist, own)
    assert poles['srg_s'] is None and poles['srg_damping'] == 0.0

    with pytest.warns(rpa.SigmaDampingWarning,
                      match=r'beta orbital 6 \(') as rec:
        ham = build_effective_hamiltonian(low, spaces, sigma='srg')
        en = ham.energies
    damping = [w for w in rec if w.category is rpa.SigmaDampingWarning]
    assert len(damping) == 1 and damping[0].filename == __file__
    assert 'alpha orbital' not in str(damping[0].message)
    assert [f['flag'] for f in ham.numerical_flags] == ['sigma_srg_damping']
    poles = ham.diagnostics['sigma_poles']
    damped = sigma_srg_uhf(*so, DOWNFOLDING_SRG_S)
    undamped = sigma_srg_uhf(*so, None)
    moved = max(np.max(np.abs(x - y)) for x, y in zip(damped, undamped))
    assert poles['srg_s'] == DOWNFOLDING_SRG_S
    assert abs(poles['srg_damping'] - moved) < 1e-9
    ham = build_effective_hamiltonian(low, spaces, diagnostics=None)
    en = ham.energies
    assert ham.diagnostics == {}


# --- mRPA: the interaction that keeps the zeroth moment ------------------------

MRPA_SPACES = ('asymmetric', 'unequal_cores', 'no_beta_pair')


def _mrpa_v_eff(so, parts, vt):
    """(aa, ab, bb) of the exact active integrals plus the blocks vt."""
    eri_so, _, _, act_so = so
    v = spin_blocks(eri_so[np.ix_(act_so, act_so, act_so, act_so)],
                    parts[0].n_act)
    return tuple(x + y for x, y in zip(v, vt))


@pytest.mark.parametrize('space', MRPA_SPACES)
@pytest.mark.parametrize('df', [False, True])
def test_mrpa_against_oracle(oh, oh_low, space, df):
    """The mRPA correction of an unrestricted reference, on the active
    particle-hole pairs of both spins stacked,
    vt = 1/2 [eta0_A^-1 delta_A eta0_A^-1 - delta_A] - (ia|jb), against the
    oracle's own construction; it touches only the elements with a
    particle-hole pair of each side's spin on both sides. For a
    spin-polarized reference the same-spin and opposite-spin blocks differ.
    E_c^A and the static defect of v + vt against the oracle too. Measured
    over three spaces and both df settings: vt 1.6e-13 (on the alpha
    HOMO-LUMO pair, where eta0 of the two eigensolvers differs by 2.1e-13),
    E_c^A 1.8e-15, the defect 5.4e-15; the blocks are up to 0.047 and the
    aa and ab ones differ by 0.015 to 0.021 (9e-4 with two active pairs)."""
    low = oh_low[df]
    parts = _parts(oh, SPACES[space](oh))
    so = _spin_orbitals(oh, parts)
    ref = vtilde_mrpa_uhf(*so)
    blocks = rpa.screening_correction(low, parts, 'mrpa')
    assert isinstance(blocks, tuple) and len(blocks) == 3
    masks = (_ph_mask(parts[0], parts[0]), _ph_mask(parts[0], parts[1]),
             _ph_mask(parts[1], parts[1]))
    for ours, exact, mask in zip(blocks, ref, masks):
        assert np.max(np.abs(ours - exact)) < 1e-9
        assert not ours[~mask].any()
    assert np.max(np.abs(blocks[0])) > 1e-4
    if space == 'no_beta_pair':
        assert not blocks[1].any() and not blocks[2].any()
    else:
        assert np.max(np.abs(blocks[0] - blocks[1])) > 1e-3
    v_eff = _mrpa_v_eff(so, parts, ref)
    e_ref, defect_ref = _oracle_active_rpa(so, v_eff)
    assert abs(rpa.active_correlation_energy(low, parts, v_eff) - e_ref) < 1e-9
    assert abs(rpa.static_defect(low, parts, v_eff) - defect_ref) < 1e-9


@pytest.mark.parametrize('space', MRPA_SPACES)
def test_mrpa_conserves_the_zeroth_moment(oh, oh_low, space):
    """What defines mRPA: the direct RPA of the active pairs alone with
    v_eff = v + vt has the zeroth moment eta0 = (X+Y)(X+Y)^T of the full
    RPA on those pairs. Both RPA problems are solved by the oracle; only vt
    is the production code's. Measured: 2.0e-13 with mRPA (7, 8 and 2 active
    pairs) against 2.1e-2, 1.8e-2 and 1.1e-3 with the bare interaction."""
    parts = _parts(oh, SPACES[space](oh))
    so = _spin_orbitals(oh, parts)
    eri_so, eps_so, occ_so, act_so = so
    vt = rpa.screening_correction(oh_low[True], parts, 'mrpa')
    _, _, xpy, pairs = drpa_so(eri_so, eps_so, occ_so)
    keep = np.flatnonzero(active_pairs_so(pairs, act_so))
    eta0_full = (xpy @ xpy.T)[np.ix_(keep, keep)]
    occ_act = [x for x, q in enumerate(act_so) if q in occ_so]
    _, _, xpy_act, pairs_act = drpa_so(
        _local_so(_mrpa_v_eff(so, parts, vt)), np.asarray(eps_so)[act_so],
        occ_act)
    assert [(act_so[i], act_so[a]) for i, a in pairs_act] == [
        pairs[x] for x in keep]
    assert np.max(np.abs(xpy_act @ xpy_act.T - eta0_full)) < 1e-10
    # the bare interaction does not: the test can fail
    _, _, xpy_bare, _ = drpa_so(
        _local_so(_mrpa_v_eff(so, parts, [0.0, 0.0, 0.0])),
        np.asarray(eps_so)[act_so], occ_act)
    assert np.max(np.abs(xpy_bare @ xpy_bare.T - eta0_full)) > 1e-4


@pytest.mark.parametrize('name', ['h2_bs', 'oh_sto3g'])
def test_mrpa_all_active_is_zero(request, name):
    """Every orbital active: the moment-conserving interaction is the bare
    one. It is computed, not short-cut, so this is the identity
    eta0^-1 delta eta0^-1 = delta + 2K of the full problem. Measured:
    1.9e-15 (H2), 3.9e-14 (OH/STO-3G)."""
    mf = (unrestricted(OH, 1, basis='sto-3g') if name == 'oh_sto3g'
          else request.getfixturevalue(name))
    parts = _parts(mf, _all_active(mf))
    for df in (False, True):
        low = run_low_level(mf, method='rpa', df=df)
        blocks = rpa.screening_correction(low, parts, 'mrpa')
        assert len(blocks) == 3
        assert max(np.max(np.abs(b)) for b in blocks) < 1e-9


def test_mrpa_without_an_active_pair_is_zero(oh, oh_low):
    blocks = rpa.screening_correction(
        oh_low[True], _parts(oh, _occupied_only(oh)), 'mrpa')
    assert len(blocks) == 3 and not any(b.any() for b in blocks)


@pytest.mark.parametrize('df', [False, True])
def test_mrpa_closed_shell_limit(closed_shell, df):
    """On a closed shell the two-channel mRPA gives the restricted one in
    all three blocks: the triplet part of the direct RPA is non-interacting,
    so the same-spin and the opposite-spin corrections coincide. Through
    the driver, with and without Sigma^E, H_eff and every energy term are
    the restricted ones. Measured: vt 1.8e-13 (blocks up to 6.7e-3), h, the
    energy terms and the FCI energy 1.1e-13."""
    rhf, lows = closed_shell
    low_r, low_u = lows[df]
    one = ActiveSpace.from_indices(len(rhf.mo_energy), 5, [2, 4], [5, 7])
    part = one
    ref = rpa.screening_correction(low_r, part, 'mrpa')
    for block in rpa.screening_correction(low_u, (part, part), 'mrpa'):
        assert np.max(np.abs(block - ref)) < 1e-9
    assert np.max(np.abs(ref)) > 1e-3
    for sigma in SIGMAS_RPA:
        ham_r = _quiet(build_effective_hamiltonian, low_r, one,
                       screening='mrpa', sigma=sigma)
        g_r, en_r = ham_r.g_eff, ham_r.energies
        ham_u = _quiet(build_effective_hamiltonian, low_u,
                       (one, one), screening='mrpa', sigma=sigma)
        g_u, space, en_u = ham_u.g_eff, ham_u.space, ham_u.energies
        for h in g_u[1]:
            assert np.max(np.abs(h - g_r[1])) < 1e-9
        for v in g_u[2]:
            assert np.max(np.abs(v - g_r[2])) < 1e-9
        for name in ENERGY_TERMS:
            assert abs(getattr(en_u, name) - getattr(en_r, name)) < 1e-9, name
        e_r = solve('pyscf_fci', g_r, 4, 4, spin_free=True)[0]
        assert abs(_solve(g_u, space)[0] - e_r) < 1e-9


@pytest.mark.parametrize('sigma', SIGMAS_RPA)
@pytest.mark.parametrize('space', sorted(SPACES))
def test_mrpa_reference_determinant_identity(oh, oh_low, space, sigma):
    """mRPA through the driver: the reference determinant of H_eff keeps
    the energy E_UHF + E_c - E_c^A; a particle-hole-only vtilde leaves the
    mean-field energy of the active space alone (E_HF,ee[vtilde] = 0).
    Measured: 5.7e-14 over the 12 builds."""
    ham = _quiet(build_effective_hamiltonian, oh_low[True],
                 SPACES[space](oh), screening='mrpa', sigma=sigma)
    g_eff, sp, en = ham.g_eff, ham.space, ham.energies
    assert abs(reference_energy_uhf(g_eff, sp['n_elec_active'])
               - (oh.e_tot + en.e_c_full - en.e_c_active)) < 1e-9
    assert en.e_hartree_dc == 0.0 and en.e_exchange_dc == 0.0
    assert en.e_exchange_term == 0.0
    assert isinstance(ham.diagnostics['static_defect'], float)


def test_mrpa_is_casida_only_whatever_the_reference(oh):
    low = run_low_level(water(), method='rpa', mode='imagfrequency')
    part = ActiveSpace.from_counts(low.eps, 5, 2, 2)
    with pytest.raises(NotImplementedError, match="mode='casida'"):
        rpa.screening_correction(low, part, 'mrpa')


# --- a density-fitted mean field -----------------------------------------------

def test_density_fitted_uhf_through_the_driver():
    """OH on a density-fitted UHF: the RPA then runs on the mean field's own
    fit, a v that differs from the exact one (elsewhere in this file
    df=True is the exact v in three-index form). Every RPA quantity is
    compared with the oracle fed with that fitted v, built here by pyscf
    from the auxiliary basis: E_c, vtilde for crpa, crpa_ph and mrpa, E_c^A,
    the static defect (zero for crpa) and Sigma_bar^E in both forms. The
    bare v of H_eff and the core Fock operator stay exact: v_eff is the
    exact active integrals plus the fitted vtilde, h_eff has the exact f,
    and the reference determinant has the energy of the exact HF expression
    on these orbitals plus E_c - E_c^A, not the density-fitted SCF energy.

    Measured (93 auxiliary functions): the fit moves v by up to 1.2e-3, E_c
    by 3.1e-5, the SCF energy by 3.6e-7 and Sigma_bar by 1.1e-5. Against the
    oracle on the fitted v: E_c 5.7e-14, vtilde 1.2e-15 (crpa), 5.4e-16
    (crpa_ph), 8.8e-14 (mrpa), E_c^A 8.9e-16, the defect 3.2e-15 (9.4e-16
    for crpa; 6.5e-4 if v_eff is put on the exact v), Sigma_bar 3.7e-15
    ('srg') and 7.3e-14 ('mode_a'), the reference-determinant identity
    5.7e-14 (3.6e-7 off with the density-fitted SCF energy). Tried against
    three wrong codes, each of which fails it: the spin weight 2 on two
    channels, the active v scaled by 1 + 1e-6, the beta pairs left out of
    the stacked factors."""
    mf = unrestricted(OH, 1, density_fit=True)
    low = run_low_level(mf, method='rpa')
    assert low.factors is not None and low.eri is None
    spaces = _asymmetric(mf)
    parts = _parts(mf, spaces)
    nmo, n = mf.mo_coeff.shape[-1], parts[0].n_act
    eri_exact, eps_so, occ_so, act_so = _spin_orbitals(mf, parts)
    eri_fit = fitted_spin_orbital_integrals(mf)
    so = (eri_fit, eps_so, occ_so, act_so)
    active = np.ix_(act_so, act_so, act_so, act_so)
    e_hf = scf.UHF(mf.mol).energy_tot(dm=mf.make_rdm1())
    masks = (_ph_mask(parts[0], parts[0]), _ph_mask(parts[0], parts[1]),
             _ph_mask(parts[1], parts[1]))

    # the fit is a real one: v, E_c and the SCF energy carry its error
    assert 1e-6 < np.max(np.abs(eri_fit - eri_exact)) < 1e-2
    assert abs(low.e_c_full - drpa_so(eri_fit, eps_so, occ_so)[0]) < 1e-9
    assert abs(low.e_c_full - drpa_so(eri_exact, eps_so, occ_so)[0]) > 1e-7
    assert abs(e_hf - mf.e_tot) > 1e-7

    crpa = vtilde_crpa_uhf(*so)
    corrections = {'crpa': crpa,
                   'crpa_ph': tuple(x * m for x, m in zip(crpa, masks)),
                   'mrpa': vtilde_mrpa_uhf(*so)}
    v_fit = spin_blocks(eri_fit[active], n)
    v_exact = spin_blocks(eri_exact[active], n)
    for screening, vt in corrections.items():
        ours = rpa.screening_correction(low, parts, screening)
        for block, ref in zip(ours, vt):
            assert np.max(np.abs(block - ref)) < 1e-9
        # E_c^A and the defect are on the fitted v, the v of the full RPA
        e_ref, defect_ref = _oracle_active_rpa(
            so, tuple(x + y for x, y in zip(v_fit, vt)))
        ham = build_effective_hamiltonian(low, spaces,
                                          screening=screening)
        g_eff, sp, en = ham.g_eff, ham.space, ham.energies
        assert abs(en.e_c_active - e_ref) < 1e-9
        assert abs(ham.diagnostics['static_defect'] - defect_ref) < 1e-9
        # H_eff carries the exact bare v
        for block, bare, corr in zip(g_eff[2], v_exact, vt):
            assert np.max(np.abs(block.transpose(0, 2, 1, 3) - bare
                                 - corr)) < 1e-9
        assert abs(reference_energy_uhf(g_eff, sp['n_elec_active'])
                   - (e_hf + en.e_c_full - en.e_c_active)) < 1e-9
        if screening == 'crpa':
            assert ham.diagnostics['static_defect'] < 1e-10
            # on the exact v instead, the crpa defect is the fitting error
            mixed = tuple(x + y for x, y in zip(v_exact, vt))
            assert rpa.static_defect(low, parts, mixed) > 1e-7

    exact_so = (eri_exact, eps_so, occ_so, act_so)
    for ours, ref, other in zip(
            rpa.embedding_self_energy(low, parts, 'srg'),
            sigma_srg_uhf(*so, DOWNFOLDING_SRG_S),
            sigma_srg_uhf(*exact_so, DOWNFOLDING_SRG_S)):
        assert np.max(np.abs(ours - ref)) < 1e-9
        assert np.max(np.abs(ours - other)) > 1e-8
    mode_a = _quiet(rpa.embedding_self_energy, low, parts, 'mode_a')
    for ours, ref in zip(mode_a, sigma_mode_a_uhf(*so)):
        assert np.max(np.abs(ours - ref)) < 1e-9

    # h_eff with Sigma^E: the exact f, the Hartree double counting of the
    # fitted vtilde, and Sigma_bar on the fitted W
    ham2 = _quiet(build_effective_hamiltonian, low, spaces,
                  screening='crpa', sigma='srg')
    g_eff, sp, en = ham2.g_eff, ham2.space, ham2.energies
    h_so, _ = spin_orbital_integrals(mf.mol, *mf.mo_coeff)
    occ_env = list(spaces[0].env_occ) + list(nmo + spaces[1].env_occ)
    occ_act = list(range(sp['n_elec_active'][0])) + [
        n + k for k in range(sp['n_elec_active'][1])]
    f, t_dc = spin_orbital_one_body(h_so, eri_exact, occ_env, act_so,
                                    _local_so(crpa), occ_act, 'hartree')
    sigma_bar = sigma_srg_uhf(*so, DOWNFOLDING_SRG_S)
    for spin, block in enumerate((slice(0, n), slice(n, 2 * n))):
        assert np.max(np.abs(g_eff[1][spin] - (f - t_dc)[block, block]
                             - sigma_bar[spin])) < 1e-9
    assert abs(reference_energy_uhf(g_eff, sp['n_elec_active'])
               - (e_hf + en.e_c_full - en.e_c_active)) < 1e-9


# --- a reference without any particle-hole pair --------------------------------

@pytest.mark.parametrize('df', [False, True])
def test_a_reference_without_any_pair(df):
    """Triplet H2 in a minimal basis: both electrons in the two alpha
    orbitals, so alpha has no virtual and beta no occupied orbital, and
    there is no particle-hole pair at all. casida takes it: the RPA is
    empty, E_c = 0, nothing is screened and Sigma_bar is zero, so H_eff is
    the bare one and its one determinant has the UHF energy; the numerical
    routes refuse it. The restricted counterpart, He in a one-function
    basis, runs the same way on casida and is refused the same way on the
    numerical routes."""
    mf = unrestricted('H 0 0 0; H 0 0 0.74', 2, basis='sto-3g')
    assert _nocc(mf) == (2, 0) and mf.mo_coeff.shape[-1] == 2
    low = run_low_level(mf, method='rpa', df=df)
    assert low.e_c_full == 0.0
    assert low.omega.shape == (0,) and low.xpy.shape == (0, 0)
    assert rpa.lowest_excitation(low) is None
    spaces = (ActiveSpace.from_indices(2, 2, [0, 1], []),
              ActiveSpace.from_indices(2, 0, [], [0, 1]))
    ham_bare = _hf(mf, spaces)
    bare = ham_bare.g_eff
    for screening in SCREENINGS_RPA + ('mrpa',):
        for sigma in SIGMAS_RPA:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                ham = build_effective_hamiltonian(
                    low, spaces, screening=screening, sigma=sigma)
                g_eff, space, en = ham.g_eff, ham.space, ham.energies
            assert en.e_c_full == en.e_c_active == 0.0
            assert ham.diagnostics['static_defect'] == 0.0
            for ours, hf in zip(g_eff[1] + g_eff[2], bare[1] + bare[2]):
                assert np.max(np.abs(ours - hf)) < 1e-12
            assert space['n_elec_active'] == (2, 0)
            assert abs(_solve(g_eff, space)[0] - mf.e_tot) < 1e-10
            if sigma is not None:
                assert ham.diagnostics['sigma_poles']['window'] == (
                    (-np.inf, np.inf), (-np.inf, np.inf))
    ham = build_effective_hamiltonian(low, _none_active(mf))
    en = ham.energies
    assert abs(en.e0 - mf.e_tot) < 1e-10
    for mode in ('imagfrequency', 'space-time'):
        with pytest.raises(NotImplementedError,
                           match="alpha channel has 2 occupied of 2.*"
                                 "mode='casida'"):
            run_low_level(mf, method='rpa', mode=mode)

    he = scf.RHF(gto.M(atom='He', basis='sto-3g', verbose=0))
    he.verbose = 0
    he.kernel()
    low = run_low_level(he, method='rpa', df=df)
    assert low.e_c_full == 0.0 and low.omega.shape == (0,)
    ham = build_effective_hamiltonian(
        low, ActiveSpace.from_indices(1, 1, [0], []), sigma='srg')
    g_eff, space, en = ham.g_eff, ham.space, ham.energies
    assert abs(solve('pyscf_fci', g_eff, 1, 2, spin_free=True)[0]
               - he.e_tot) < 1e-10
    for mode in ('imagfrequency', 'space-time'):
        with pytest.raises(NotImplementedError,
                           match="the reference has 1 occupied of 1.*"
                                 "mode='casida'"):
            run_low_level(he, method='rpa', mode=mode)


# --- a spin without occupied orbitals, with exact integrals --------------------

@pytest.mark.parametrize('df', [False, True])
def test_a_spin_without_occupied_orbitals_runs_on_casida(df):
    """Triplet H2 in 6-31G: both electrons alpha, so the beta channel has no
    occupied orbital and no particle-hole pair, and the stacked problem
    holds the alpha pairs alone. Every screening builds, also from the
    exact four-index blocks, where the beta-beta and alpha-beta pair blocks
    have no row or no column: vtilde against the oracle, the
    reference-determinant identity, and the solve. Measured for both df
    settings: crpa 2.8e-16 (on blocks up to 0.045), mrpa 4.9e-16, the
    identity 3.3e-16."""
    mf = unrestricted('H 0 0 0; H 0 0 0.74', 2)
    assert _nocc(mf) == (2, 0)
    low = run_low_level(mf, method='rpa', df=df)
    spaces = (ActiveSpace.from_indices(4, 2, [1], [2]),
              ActiveSpace.from_indices(4, 0, [], [0, 1]))
    parts = _parts(mf, spaces)
    so = _spin_orbitals(mf, parts)
    ref = vtilde_crpa_uhf(*so)
    assert max(np.max(np.abs(block)) for block in ref) > 1e-3
    for ours, exact in zip(rpa.screening_correction(low, parts, 'crpa'), ref):
        assert np.max(np.abs(ours - exact)) < 1e-9
    for ours, exact in zip(rpa.screening_correction(low, parts, 'mrpa'),
                           vtilde_mrpa_uhf(*so)):
        assert np.max(np.abs(ours - exact)) < 1e-9
    for screening in SCREENINGS_RPA + ('mrpa',):
        for sigma in SIGMAS_RPA:
            ham = _quiet(build_effective_hamiltonian, low,
                         spaces, screening=screening,
                         sigma=sigma)
            g_eff, space, en = ham.g_eff, ham.space, ham.energies
            assert space['n_elec_active'] == (1, 0)
            assert abs(reference_energy_uhf(g_eff, (1, 0))
                       - (mf.e_tot + en.e_c_full - en.e_c_active)) < 1e-9
            assert np.isfinite(_solve(g_eff, space)[0])


def test_the_record_solves_itself_with_its_spin_overlap(oh):
    """For an unrestricted reference h.solve also passes the record's
    spin_overlap, so <S^2> is reported without the caller handing it in."""
    nmo = oh.mo_coeff.shape[-1]
    occupied = (ActiveSpace.from_indices(nmo, 5, [3, 4], []),
                ActiveSpace.from_indices(nmo, 4, [2, 3], []))
    h = build_effective_hamiltonian(run_low_level(oh, method='hf'), occupied)
    out, want = {}, {}
    e = h.solve('pyscf_fci', out=out)
    e_want = _solve(h.g_eff, h.space, out=want,
                    spin_overlap=h.space['spin_overlap'])
    assert np.array_equal(e, e_want)
    assert out['spin_square'] == want['spin_square']
    assert abs(out['spin_square'][0] - oh.spin_square()[0]) < 1e-10
