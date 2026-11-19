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
"""
import copy
import warnings

import numpy as np
import pytest
from pyscf import fci, mcscf, scf

from src.MultiReference.Downfolding import (build_effective_hamiltonian,
                                            run_low_level)
from src.SingleReference.common.active_space import (ActiveSpace,
                                                     unrestricted_from_counts)
from src.Solvers.active_space import EmptyActiveSpaceWarning, solve
from tests.downfolding_oracle import (CH2, NH, OH, h2_broken,
                                      reference_energy_uhf,
                                      spin_orbital_energies,
                                      spin_orbital_integrals,
                                      spin_orbital_one_body, unrestricted,
                                      water)


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
