"""`solve` on a spin-resolved effective Hamiltonian: different alpha and beta
one-body blocks and the three two-body blocks (aa, ab, bb), as an
unrestricted (UHF/UKS) reference gives them.

The Hamiltonians are built here by hand, from pyscf alone, as the CASCI
Hamiltonian of a UHF determinant in its own orbitals: the core Fock operator
of each spin and the exact integrals between each spin's active orbitals.
'pyscf_fci' solves the per-spin blocks with fci.direct_uhf at the given
(N_alpha, N_beta); it is checked against pyscf's UCASCI (equal core counts,
where UCASCI is right) and against the determinant-basis diagonalization of
the spin-orbital form (N_alpha = N_beta). <S^2> is that of the whole
molecule's state, the active CI vector with the core determinant, and needs
the alpha-beta overlaps of the orbitals: pyscf's own value assumes alpha
orbital p and beta orbital p are one function.
"""
import warnings

import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, mcscf, scf

from src.Solvers.active_space import EmptyActiveSpaceWarning, solve


def _uhf(atom, spin, basis='6-31g'):
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def oh():
    return _uhf('O 0 0 0; H 0 0 0.97', 1)


@pytest.fixture(scope='module')
def h2_broken():
    """H2 at 2.5 A with one electron on each atom: N_alpha = N_beta, the
    alpha and beta orbitals different."""
    mol = gto.M(atom='H 0 0 0; H 0 0 2.5', basis='6-31g', verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    dm = mf.init_guess_by_atom()
    left = mol.aoslice_by_atom()[0]
    dm[0][left[2]:left[3]] = dm[0][:, left[2]:left[3]] = 0.0     # alpha on H2
    dm[1][left[3]:] = dm[1][:, left[3]:] = 0.0                    # beta on H1
    mf.kernel(dm0=dm)
    assert mf.converged and mf.spin_square()[0] > 0.5
    return mf


def _blocks(mf, cores, n_act):
    """(g_eff, spin_overlap, n_elec) of the UHF-orbital CASCI Hamiltonian:
    `cores[s]` inactive orbitals and the next n_act orbitals active, per
    spin. g_eff = {0: E_core, 1: (h_a, h_b), 2: (aa, ab, bb) physicist}."""
    mol = mf.mol
    c_core = [mf.mo_coeff[s][:, :cores[s]] for s in (0, 1)]
    c_act = [mf.mo_coeff[s][:, cores[s]:cores[s] + n_act] for s in (0, 1)]
    dm = np.array([c @ c.T for c in c_core])
    hcore = mf.get_hcore()
    vj, vk = mf.get_jk(mol, dm)
    fock = [hcore + vj[0] + vj[1] - vk[s] for s in (0, 1)]
    e_core = mf.energy_nuc() + sum(
        np.einsum('ij,ji->', dm[s], hcore + 0.5 * (vj[0] + vj[1] - vk[s]))
        for s in (0, 1))
    h = tuple(c_act[s].T @ fock[s] @ c_act[s] for s in (0, 1))

    def eri(c1, c2):
        return ao2mo.general(mol, (c1, c1, c2, c2), compact=False).reshape(
            (n_act,) * 4)

    chem = (eri(c_act[0], c_act[0]), eri(c_act[0], c_act[1]),
            eri(c_act[1], c_act[1]))
    g_eff = {0: e_core, 1: h, 2: tuple(x.transpose(0, 2, 1, 3) for x in chem)}
    s = mf.get_ovlp()
    overlap = {'core_core': c_core[0].T @ s @ c_core[1],
               'core_active': c_core[0].T @ s @ c_act[1],
               'active_core': c_act[0].T @ s @ c_core[1],
               'active_active': c_act[0].T @ s @ c_act[1]}
    n_elec = tuple(mol.nelec[s] - cores[s] for s in (0, 1))
    return g_eff, overlap, n_elec


def _spin_orbital(g_eff):
    """The interleaved spin-orbital form (alpha at 2p, beta at 2p + 1) with
    the antisymmetrized physicist two-body term, written out here."""
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


def test_direct_uhf_path_matches_ucasci(oh):
    """Per-spin blocks with (N_alpha, N_beta) = (2, 1) in 3 orbitals: the
    lowest root is pyscf's UCASCI energy (equal core counts)."""
    g_eff, _, n_elec = _blocks(oh, (3, 3), 3)
    assert n_elec == (2, 1)
    e = solve('pyscf_fci', g_eff, 3, n_elec, spin_free=True)
    ref = mcscf.UCASCI(oh, 3, (2, 1))
    ref.verbose = 0
    assert abs(e[0] - ref.kernel()[0]) < 1e-10


def test_two_solvers_agree_for_equal_spin_counts(oh):
    """Unequal core counts (3, 2), N = (2, 2) in 4 orbitals: fci.direct_uhf on
    the blocks and the determinant-basis diagonalization of the spin-orbital
    form give the same lowest roots."""
    g_eff, _, n_elec = _blocks(oh, (3, 2), 4)
    assert n_elec == (2, 2)
    blocks = solve('pyscf_fci', g_eff, 4, n_elec, spin_free=True, nroots=3)
    dets = solve('openfermion', _spin_orbital(g_eff), 8, n_elec,
                 spin_free=False, nroots=3)
    assert np.abs(blocks - dets).max() < 1e-9


def test_total_spin_square(oh, h2_broken):
    """out['spin_square'] is <S^2> of the whole molecule's state. With every
    orbital active the roots are spin-pure: broken-symmetry H2 gives 0 for
    the ground state and 2 for the triplet, where pyscf's identity-overlap
    value gives 0 for both. With a core, OH equals pyscf's
    UCASCI.spin_square. Without the overlaps nothing is reported."""
    nmo = h2_broken.mo_coeff.shape[-1]
    g_eff, overlap, n_elec = _blocks(h2_broken, (0, 0), nmo)
    out = {}
    e = solve('pyscf_fci', g_eff, nmo, n_elec, spin_free=True, nroots=3,
              out=out, spin_overlap=overlap)
    assert len(out['spin_square']) == len(e) == 3
    assert abs(out['spin_square'][0]) < 1e-8
    assert abs(out['spin_square'][1] - 2.0) < 1e-8
    wrong = fci.direct_uhf.FCI()
    wrong.verbose = 0
    h1 = g_eff[1]
    chem = tuple(x.transpose(0, 2, 1, 3) for x in g_eff[2])
    _, vecs = wrong.kernel(h1, chem, nmo, n_elec, nroots=3)
    assert abs(wrong.spin_square(vecs[1], nmo, n_elec)[0] - 2.0) > 0.5

    # an active space of occupied orbitals only: the one state is the UHF
    # determinant, whose <S^2> is exact and involves every overlap block
    # (unequal cores, so the core-active blocks are not square)
    g_eff, overlap, n_elec = _blocks(oh, (3, 2), 2)
    assert n_elec == (2, 2)
    out = {}
    solve('pyscf_fci', g_eff, 2, n_elec, spin_free=True, out=out,
          spin_overlap=overlap)
    assert abs(out['spin_square'][0] - oh.spin_square()[0]) < 1e-10

    # a correlated state on a core: close to pyscf's UCASCI value, which adds
    # <S^2> of the core determinant and of the active space and leaves out
    # their cross terms (3e-7 here)
    g_eff, overlap, n_elec = _blocks(oh, (3, 3), 3)
    out = {}
    solve('pyscf_fci', g_eff, 3, n_elec, spin_free=True, out=out,
          spin_overlap=overlap)
    ref = mcscf.UCASCI(oh, 3, (2, 1))
    ref.verbose = 0
    ref.kernel()
    assert abs(out['spin_square'][0] - ref.spin_square()[0]) < 1e-5
    assert abs(out['spin_square'][0] - 0.75) > 1e-3     # contaminated doublet

    out = {}
    solve('pyscf_fci', g_eff, 3, n_elec, spin_free=True, out=out)
    assert 'spin_square' not in out


def test_spin_resolved_corners(oh):
    """An empty space returns the constant with the empty-space warning;
    electrons in no orbitals are refused; nroots above the sector's size
    returns the sector's roots; symmetrize=True takes the Hermitian part of
    each block and symmetrize=False refuses a non-Hermitian one."""
    empty = np.zeros((0, 0))
    empty4 = np.zeros((0, 0, 0, 0))
    g0 = {0: -1.25, 1: (empty, empty), 2: (empty4, empty4, empty4)}
    with pytest.warns(EmptyActiveSpaceWarning):
        e = solve('pyscf_fci', g0, 0, (0, 0), spin_free=True)
    assert np.array_equal(e, [-1.25])
    with pytest.raises(ValueError, match='no active orbitals'):
        solve('pyscf_fci', g0, 0, (1, 0), spin_free=True)

    g_eff, _, n_elec = _blocks(oh, (3, 3), 3)
    every = solve('pyscf_fci', g_eff, 3, n_elec, spin_free=True, nroots=20)
    assert len(every) == 9                    # C(3,2) * C(3,1) determinants
    assert np.all(np.diff(every) > -1e-12)

    rng = np.random.default_rng(3)
    skew = rng.normal(size=(3, 3))
    skew = 1e-2 * (skew - skew.T)
    bent = {0: g_eff[0], 1: (g_eff[1][0] + skew, g_eff[1][1]), 2: g_eff[2]}
    with pytest.raises(ValueError, match='Hermitian'):
        solve('pyscf_fci', bent, 3, n_elec, spin_free=True, symmetrize=False)
    assert abs(solve('pyscf_fci', bent, 3, n_elec, spin_free=True)[0]
               - every[0]) < 1e-10
    with pytest.raises(ValueError, match='pair'):
        solve('pyscf_fci', g_eff, 3, 3, spin_free=True)


def test_spin_resolved_refusals(oh):
    """What cannot solve a spin-resolved Hamiltonian says what can."""
    g_eff, _, n_elec = _blocks(oh, (3, 3), 3)
    for name in ('dmrg', 'ccsdt', 'ccsdtq'):
        with pytest.raises(NotImplementedError, match='pyscf_fci'):
            solve(name, g_eff, 3, n_elec, spin_free=True)
    with pytest.raises(ValueError, match='spin_free=False'):
        solve('openfermion', g_eff, 3, n_elec, spin_free=True)
    g_so = _spin_orbital(g_eff)
    with pytest.raises(NotImplementedError, match='N_alpha = N_beta'):
        solve('openfermion', g_so, 6, n_elec, spin_free=False)
    with pytest.raises(ValueError, match='spin_free=True'):
        solve('pyscf_fci', g_so, 6, 3, spin_free=False)
