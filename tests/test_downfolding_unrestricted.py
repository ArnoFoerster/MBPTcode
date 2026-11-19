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

import numpy as np
import pytest
from pyscf import fci, mcscf, scf
from pyscf.gw import urpa

from src.Base.numerics.constants import DOWNFOLDING_SRG_S
from src.MultiReference.Downfolding import (build_effective_hamiltonian,
                                            run_low_level)
from src.MultiReference.Downfolding.methods import rpa
from src.MultiReference.Downfolding.partition import (partition,
                                                      ph_element_mask)
from src.SingleReference.common.active_space import (ActiveSpace,
                                                     unrestricted_from_counts)
from src.Solvers.active_space import EmptyActiveSpaceWarning, solve
from tests.downfolding_oracle import (CH2, NH, OH, drpa_so, h2_broken,
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


SPACES = {'asymmetric': _asymmetric, 'unequal_cores': _unequal_cores,
          'no_beta_pair': _no_beta_pair, 'occupied_only': _occupied_only}


def _parts(mf, spaces):
    return tuple(partition(space, eps)
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
    g_eff, space, energies = _hf(mf, spaces)
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
    g_eff, space, energies = _hf(mf, _all_active(mf))
    assert (energies.e_core, energies.e_hartree, energies.e_exchange) == (0, 0, 0)
    assert space['n_elec_active'] == mf.mol.nelec
    assert abs(_solve(g_eff, space)[0] - fci.FCI(restricted).kernel()[0]) < 1e-9


def test_hf_none_active_is_uhf(oh):
    """No orbital active: E_0 is the unrestricted SCF energy, and for a UKS
    reference the HF energy expression on the Kohn-Sham determinant.
    Measured: 1.4e-14 for both; the HF expression on the PBE0 determinant is
    0.265 Ha above the UKS energy."""
    g_eff, space, energies = _hf(oh, _none_active(oh))
    assert abs(energies.e0 - oh.e_tot) < 1e-10
    assert space['n_elec_active'] == (0, 0) and space['n_orbs'] == 0
    assert [h.shape for h in g_eff[1]] == [(0, 0)] * 2
    assert [v.shape for v in g_eff[2]] == [(0, 0, 0, 0)] * 3
    with pytest.warns(EmptyActiveSpaceWarning):
        assert np.array_equal(_solve(g_eff, space), [energies.e0])

    uks = unrestricted(OH, 1, xc='pbe0')
    hf_expression = scf.UHF(uks.mol).energy_tot(dm=uks.make_rdm1())
    assert abs(hf_expression - uks.e_tot) > 1e-2
    _, _, energies = _hf(uks, _none_active(uks))
    assert abs(energies.e0 - hf_expression) < 1e-10


def test_hf_blocks_against_spin_orbital_oracle(oh):
    """Different non-contiguous index sets per spin, N^A = (2, 1): every
    block of H_eff and every E_0 term against the spin-orbital oracle, and
    the reference determinant of H_eff has the UHF energy. Measured: one-body
    blocks 2.9e-15, two-body blocks 6.2e-16, energy terms at most 2.8e-14,
    reference determinant 4.3e-14; h_alpha and h_beta differ by 0.45."""
    spaces = _asymmetric(oh)
    g_eff, space, energies = _hf(oh, spaces)
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
    g_r, space_r, en_r = _hf(rhf, one)
    g_u, space_u, en_u = _hf(uhf, (one, one))
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
    g_blocks, space, _ = _hf(oh, spaces)
    g_so, space_so, _ = _hf(oh, spaces, spin_free=False)
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
    g_blocks, space, _ = _hf(oh, spaces)
    g_so, space_so, _ = _hf(oh, spaces, spin_free=False)
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
    _, space, _ = _hf(oh, spaces)
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
    g_eff, space, _ = _hf(oh, occupied)
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
    part = partition(ActiveSpace.from_indices(nmo, nocc, [2, 4], [5, 7]), eps)
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


def test_unrestricted_numerical_routes_point_to_casida(oh):
    """The imaginary-frequency and space-time routes do not take an
    unrestricted reference yet."""
    for mode in ('imagfrequency', 'space-time'):
        with pytest.raises(NotImplementedError, match="mode='casida'"):
            run_low_level(oh, method='rpa', mode=mode)


# --- the screening correction of an active space ------------------------------

@pytest.fixture(scope='module')
def oh_low(oh):
    """The casida low-level results of OH on exact (df=False) and on fitted
    (df=True, here the eigendecomposition of the exact) integrals."""
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


def test_partition_form_must_fit_the_low_level_result(oh, oh_low):
    """The functions of the RPA method take one Partition with a restricted
    result and the pair (alpha, beta) with an unrestricted one."""
    parts = _parts(oh, _asymmetric(oh))
    with pytest.raises(ValueError, match='pair'):
        rpa.screening_correction(oh_low[True], parts[0], 'crpa')
    rhf = water()
    one = partition(ActiveSpace.from_counts(rhf.mo_energy, 5, 2, 2),
                    rhf.mo_energy)
    low = run_low_level(rhf, method='rpa')
    with pytest.raises(ValueError, match='one Partition'):
        rpa.screening_correction(low, (one, one), 'crpa')
    assert isinstance(rpa.screening_correction(low, one, 'crpa'), np.ndarray)
    with pytest.raises(NotImplementedError, match='mrpa'):
        rpa.screening_correction(oh_low[True], parts, 'mrpa')
