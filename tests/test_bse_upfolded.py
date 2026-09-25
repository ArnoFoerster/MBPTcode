"""Upfolded (frequency-free) dynamical BSE -- Bintrim & Berkelbach 2022.

The validation chain here is unusually strong because the paper hands us an
exact algebraic identity to test against. Downfolding the doubles out of the
upfolded matrix (Eq. 9) must return the frequency-dependent BSE matrix
(Eq. 1), and that matrix can be built a completely different way -- by
diagonalizing the RPA problem and summing poles (Eq. 6). Two independent
routes to the same object, exact at every frequency, is what pins the
construction before any solver is involved.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_two_electron_integrals_chemist,
                                      DFIntegrals)
from src.SingleReference.BSE import bse_upfolded as B
from src.Base.constants import HARTREE_TO_EV

SPINS = ('singlet', 'triplet')


def _case(basis='sto-3g', atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59'):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'))
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    no, norb = mol.nelectron // 2, mol.nao
    gb = B.eri_blocks(eri, no, norb)
    # quasiparticle energies deliberately DIFFERENT from eps: the two enter
    # different blocks (E in A and the outer doubles pair, eps in S), and a
    # mixup is invisible if they are equal
    rng = np.random.default_rng(0)
    e_qp = eps + 0.05 * rng.standard_normal(norb)
    return mol, mf, eps, e_qp, gb, no, norb - no


@pytest.mark.parametrize('spin', SPINS)
@pytest.mark.parametrize('omega', [0.3, 0.7, -0.2])
def test_downfolding_equals_sum_over_states(spin, omega):
    """Eq. (9) == Eq. (1) with the kernel (6). The decisive test."""
    _, _, eps, e_qp, gb, no, nv = _case()
    a = B.downfold(eps, e_qp, gb, no, nv, omega, spin)
    b = B.dynamical_bse_matrix(eps, e_qp, gb, no, nv, omega, spin)
    assert np.abs(a - b).max() < 1e-10


@pytest.mark.parametrize('spin', SPINS)
def test_eigenvalues_solve_the_frequency_dependent_problem(spin):
    """Every eigenvalue Omega of the upfolded H must be an eigenvalue of
    A(Omega) -- the self-consistency the upfolding is supposed to deliver."""
    _, _, eps, e_qp, gb, no, nv = _case()
    w, _ = B.solve_dense(eps, e_qp, gb, no, nv, spin, nroots=5)
    for om in w:
        Aw = B.downfold(eps, e_qp, gb, no, nv, om, spin)
        assert np.abs(np.linalg.eigvals(Aw) - om).min() < 1e-9


@pytest.mark.parametrize('spin', SPINS)
def test_sigma_matches_dense_hamiltonian(spin):
    _, _, eps, e_qp, gb, no, nv = _case()
    n = B.dimensions(no, nv)['nH']
    H = B.build_hamiltonian(eps, e_qp, gb, no, nv, spin)
    Hs = np.column_stack([B.sigma(np.eye(n)[:, k], eps, e_qp, gb, no, nv, spin)
                          for k in range(n)])
    assert np.abs(H - Hs).max() < 1e-10
    assert np.abs(np.diag(H)
                  - B.diagonal(eps, e_qp, gb, no, nv, spin)).max() < 1e-10


@pytest.mark.parametrize('spin', SPINS)
def test_df_sigma_matches_exact_integrals(spin):
    mol, mf, eps, e_qp, gb, no, nv = _case()
    Bf = DFIntegrals.from_scf(mol, mf, exact=True).B_aa
    n = B.dimensions(no, nv)['nH']
    for k in range(0, n, max(1, n // 12)):
        v = np.eye(n)[:, k]
        a = B.sigma(v, eps, e_qp, gb, no, nv, spin)
        b = B.sigma_df(v, eps, e_qp, gb, Bf, no, nv, spin)
        assert np.abs(a - b).max() < 1e-10


@pytest.mark.parametrize('spin', SPINS)
def test_davidson_matches_dense(spin):
    _, _, eps, e_qp, gb, no, nv = _case()
    w_ref, _ = B.solve_dense(eps, e_qp, gb, no, nv, spin, nroots=4)
    diag = B.diagonal(eps, e_qp, gb, no, nv, spin)
    w, _ = B.davidson_nonsymmetric(
        lambda v: B.sigma(v, eps, e_qp, gb, no, nv, spin), diag, nroots=4)
    assert np.abs(np.sort(w) - w_ref).max() * HARTREE_TO_EV < 1e-5


def test_hamiltonian_is_asymmetric():
    """Not a detail to be tidied away: the (1,2) blocks are -V^e/-V^h while
    the (2,1) blocks are +(V^h)^T/+(V^e)^T. This is the structural difference
    from ADC, whose ISR matrix is Hermitian by construction, and it is why
    the solver has to be a non-symmetric one."""
    _, _, eps, e_qp, gb, no, nv = _case()
    H = B.build_hamiltonian(eps, e_qp, gb, no, nv, 'singlet')
    assert np.abs(H - H.T).max() > 1e-3


def test_screening_matrix_is_the_repo_direct_rpa_casida_matrix():
    """S must be the direct (Hartree-only) RPA matrix in the TDA. The repo
    already builds that for the GW module, so compare rather than trust."""
    from src.SingleReference.LinearResponse.linear_response import (
        LinearResponseSolver)
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g',
                verbose=0)
    mf = scf.RHF(mol).run()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'))
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    no, norb = mol.nelectron // 2, mol.nao
    gb = B.eri_blocks(eri, no, norb)
    S = B.block_S(eps, gb, no, norb - no).reshape(no * (norb - no), -1)
    lr = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted')
    A_rpa, _ = lr.build_casida_matrices(no, lBSE=False)
    assert np.abs(S - np.asarray(A_rpa)).max() < 1e-12


def test_triplet_drops_the_coulomb_term_only():
    """kappa multiplies (ia|jb) and nothing else: the doubles blocks and the
    couplings are direct-only screening, identical in both spin channels."""
    _, _, eps, e_qp, gb, no, nv = _case()
    Hs = B.build_hamiltonian(eps, e_qp, gb, no, nv, 'singlet')
    Ht = B.build_hamiltonian(eps, e_qp, gb, no, nv, 'triplet')
    n_s = no * nv
    assert np.abs((Hs - Ht)[n_s:, :]).max() < 1e-12
    assert np.abs((Hs - Ht)[:, n_s:]).max() < 1e-12
    assert np.abs((Hs - Ht)[:n_s, :n_s]
                  - 2.0 * gb['ovov'].reshape(n_s, n_s)).max() < 1e-12


def test_familiar_form_is_symmetric_and_lies_below():
    """Eq. (12). The paper reports it comes out 2-3 eV below the exact
    dynamical BSE; reproducing the SIGN and rough size of that gap is a check
    on both constructions at once."""
    _, _, eps, e_qp, gb, no, nv = _case()
    Hf = B.build_hamiltonian_familiar(eps, gb, no, nv, 'singlet')
    assert np.abs(Hf - Hf.T).max() < 1e-12
    lowest_familiar = np.sort(np.linalg.eigvalsh(Hf))
    lowest_familiar = lowest_familiar[lowest_familiar > 0][0]
    w, _ = B.solve_dense(eps, eps, gb, no, nv, 'singlet', nroots=1)
    assert lowest_familiar < w[0]
    assert 0.5 < (w[0] - lowest_familiar) * HARTREE_TO_EV < 6.0


def test_doubles_weight_is_a_fraction():
    _, _, eps, e_qp, gb, no, nv = _case()
    _, X = B.solve_dense(eps, e_qp, gb, no, nv, 'singlet', nroots=3)
    for k in range(X.shape[1]):
        r2 = B.doubles_weight(X[:, k], no, nv)
        assert 0.0 <= r2 <= 100.0


def test_driver_end_to_end_with_gw_energies():
    """The full path: GW quasiparticle energies -> upfolded BSE."""
    from src.SingleReference.BSE.bse_upfolded import (solve_bse_upfolded,
                                                      qp_energies)
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g',
                verbose=0)
    mf = scf.RHF(mol).run()
    e_qp = qp_energies(mf)
    no = mol.nelectron // 2
    assert e_qp[no - 1] > mf.mo_energy[no - 1]          # GW raises the HOMO
    ws, _ = solve_bse_upfolded(mf, mol, spin='singlet', nroots=3, e_qp=e_qp)
    wt, _ = solve_bse_upfolded(mf, mol, spin='triplet', nroots=3, e_qp=e_qp)
    assert wt[0] < ws[0]                                # triplet below singlet
    assert 5.0 < ws[0] * HARTREE_TO_EV < 12.0
