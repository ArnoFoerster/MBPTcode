"""CoupledClusterSolver (src/Solvers/active_space_solver.py): CCSDT and CCSDTQ
on a Hartree-Fock reference in the active space, and the 'ccsdt' / 'ccsdtq'
backends of the downfolding dispatcher (src/Solvers/active_space.py).

Checks:
  1. H4/STO-3G (4 electrons in 4 orbitals): CCSDTQ is FCI to round-off;
     CCSDT is not (it misses the quadruples), and its error is small
  2. the lowest triplet of the same H4 by spin-unrestricted CCSDT equals the
     FCI one (3 alpha and 1 beta electron in 4 orbitals: doubles are exact)
  3. water/6-31G CASCI(4,6): CCSDTQ on pyscf's own active-space Hamiltonian
     gives the CASCI total energy
  4. a QDPT-downfolded H4 (one core, three active, four external orbitals),
     whose vertex lacks the p <-> q symmetry of real-orbital integrals:
     'ccsdtq' equals 'pyscf_fci' through the dispatcher
  5. a Hermitian vertex without that symmetry is solved as it is: CCSDTQ and
     the open-shell CCSDT equal FCI where they are exact, and CCSDT on six
     electrons in six orbitals equals the spin-orbital CCSDT of
     src/SingleReference/CC, which assumes no integral symmetry
  6. where a downfolded Hamiltonian's lowest state is a triplet, 'ccsdtq'
     on paired electrons gives the lowest singlet and CCSDT at spin=2 the
     triplet, each equal to FCI in that sector
  7. refusals: a three-body vertex, a non-Hermitian vertex, more than one
     root, density matrices, an order without an engine, open-shell CCSDTQ
  8. an amplitude iteration cut short warns and flags, and keeps its energy
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, gto, mcscf, scf

from src.MultiReference.QDPT.perturbative import build_effective_hamiltonian
from src.SingleReference.CC import amplitudes
from src.SingleReference.CC.integrals import energy_denominators
from src.Solvers import active_space_solver
from src.Solvers.active_space import _spatial_active_hamiltonian, solve
from src.Solvers.active_space_solver import (ActiveSpaceCCNotConverged,
                                             CoupledClusterSolver,
                                             PyscfFCISolver)

H4_CHAIN = 'H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0'
H4_BENT = 'H 0 0 0; H 1.8 0 0; H 0.54 2.34 0; H 2.52 1.62 0.9'
H2O = 'O 0 0 0.117; H 0 0.755 -0.471; H 0 -0.755 -0.471'


@pytest.fixture(scope='module')
def h4():
    """(h1, eri, norb, nelec) of H4/STO-3G in its RHF orbitals."""
    mol = gto.M(atom=H4_CHAIN, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
    return h1, eri, mol.nao, mol.nelectron


def test_ccsdtq_is_fci_and_ccsdt_is_not_on_four_electrons(h4):
    h1, eri, norb, nelec = h4
    e_fci = PyscfFCISolver().roots(h1, eri, norb, nelec, spin=0)[0][0]
    e_q = CoupledClusterSolver(order=4).roots(h1, eri, norb, nelec, spin=0)[0][0]
    solver = CoupledClusterSolver(order=3)
    e_t = solver.roots(h1, eri, norb, nelec, spin=0)[0][0]
    assert abs(e_q - e_fci) < 1e-9
    # CCSDT under FCI by 7e-5 Ha here: truncated, and not far off
    assert 1e-6 < abs(e_t - e_fci) < 1e-3
    record = solver.last_result
    assert record['method'] == 'RCCSDT' and record['converged']
    assert abs(record['e_reference'] + record['e_corr'] - e_t) < 1e-12
    assert record['numerical_flags'] == []


def test_the_triplet_sector_by_unrestricted_ccsdt(h4):
    h1, eri, norb, nelec = h4
    e_fci = PyscfFCISolver().roots(h1, eri, norb, nelec, spin=2)[0][0]
    solver = CoupledClusterSolver(order=3)
    e_cc = solver.roots(h1, eri, norb, nelec, spin=2)[0][0]
    assert solver.last_result['method'] == 'UCCSDT'
    assert abs(e_cc - e_fci) < 1e-9


def test_ccsdtq_reproduces_casci_of_a_real_active_space():
    mol = gto.M(atom=H2O, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASCI(mf, 6, 4)
    mc.verbose = 0
    mc.kernel()
    h1eff, ecore = mc.get_h1eff()
    eri_cas = ao2mo.restore(1, mc.get_h2eff(), 6)
    e = CoupledClusterSolver(order=4).roots(h1eff, eri_cas, 6, (2, 2))[0][0]
    assert abs(e + ecore - mc.e_tot) < 1e-9


def test_the_dispatcher_solves_a_downfolded_hamiltonian():
    mol = gto.M(atom=H4_BENT, basis='6-31g', unit='Bohr', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-9)
    g_eff, space = build_effective_hamiltonian(mol, mf, 1, 3, max_pt_order=2,
                                               max_vertex=2, spin_free=True)
    args = (g_eff, space['n_orbs'], space['n_elec_active'])
    eri = g_eff[2].transpose(0, 2, 1, 3)
    assert np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))) > 1e-3   # not eightfold
    e_fci = solve('pyscf_fci', *args, spin_free=True)[0]
    e_cc = solve('ccsdtq', *args, spin_free=True)[0]
    assert abs(g_eff[0]) > 1e-3                  # the core energy is carried
    assert abs(e_cc - e_fci) < 1e-9


def test_paired_electrons_give_the_singlet_where_a_triplet_lies_lower():
    """H4/6-31G, one core and three active orbitals, PT3 with a
    Brillouin-Wigner shift: the effective Hamiltonian's lowest state at Sz = 0
    is a triplet, which 'pyscf_fci' returns; coupled cluster on the RHF
    reference is the singlet."""
    mol = gto.M(atom=H4_BENT, basis='6-31g', unit='Bohr', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-9)
    g_eff, space = build_effective_hamiltonian(mol, mf, 1, 3, max_pt_order=3,
                                               max_vertex=2, spin_free=True,
                                               shift=-0.5)
    args = (g_eff, space['n_orbs'], space['n_elec_active'])
    h1, eri, norb, nelec = _spatial_active_hamiltonian(
        g_eff, space['n_orbs'], space['n_elec_active'], True, True, 'ccsdtq')
    fci_solver = PyscfFCISolver()
    singlet = fci_solver.roots(h1, eri, norb, nelec, spin=0)[0][0]
    triplet = fci_solver.roots(h1, eri, norb, nelec, spin=2)[0][0]
    assert triplet < singlet - 1e-4
    e_cc = solve('ccsdtq', *args, spin_free=True)[0] - g_eff.get(0, 0.0)
    assert abs(e_cc - singlet) < 1e-9
    assert abs(solve('pyscf_fci', *args, spin_free=True)[0]
               - g_eff.get(0, 0.0) - triplet) < 1e-9
    e_t = CoupledClusterSolver(order=3).roots(h1, eri, norb, nelec, spin=2)[0][0]
    assert abs(e_t - triplet) < 1e-9


def _hermitian_without_eightfold(norb, seed=7):
    """A real two-body tensor with (pq|rs) = (qp|sr) = (rs|pq) but (pq|rs) !=
    (qp|rs): Hermitian, and not the integral of real orbitals."""
    x = np.random.default_rng(seed).standard_normal((norb,) * 4)
    x = x + x.transpose(1, 0, 3, 2)
    return 0.05 * (x + x.transpose(2, 3, 0, 1))


def test_a_hermitian_vertex_without_eightfold_symmetry_is_solved_as_it_is(h4):
    h1, eri, norb, nelec = h4
    eri = eri + 0.04 * _hermitian_without_eightfold(norb)
    fci_solver = PyscfFCISolver()
    for order, spin in ((4, 0), (3, 2)):        # both exact on 4 electrons here
        e_fci = fci_solver.roots(h1, eri, norb, nelec, spin=spin)[0][0]
        e_cc = CoupledClusterSolver(order=order).roots(h1, eri, norb, nelec,
                                                       spin=spin)[0][0]
        assert abs(e_cc - e_fci) < 1e-9, (order, spin)


def test_ccsdt_matches_the_spin_orbital_ccsdt_on_a_hermitian_vertex():
    """Water/6-31G CASCI(6,6) plus a Hermitian perturbation without the
    eightfold symmetry: CCSDT is not exact here, so this compares two CCSDT
    codes term by term rather than either with FCI."""
    spinorb_from_spatial = pytest.importorskip(
        'openfermion.chem.molecular_data').spinorb_from_spatial
    mol = gto.M(atom=H2O, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASCI(mf, 6, 6)
    mc.verbose = 0
    mc.kernel()
    h1 = mc.get_h1eff()[0]
    eri = (ao2mo.restore(1, mc.get_h2eff(), 6)
           + 0.04 * _hermitian_without_eightfold(6, seed=3))
    reference = active_space_solver._active_space_reference(h1, eri, 6, 3, 3)
    solver = CoupledClusterSolver(order=3)
    e_cc = solver.roots(h1, eri, 6, 6)[0][0]

    c = reference.mo_coeff
    soei, stei = spinorb_from_spatial(
        c.T @ h1 @ c, np.einsum('pqrs,pi,qj,rk,sl->iklj', eri, c, c, c, c))
    astei = stei - stei.transpose(0, 1, 3, 2)
    g = astei.transpose(0, 1, 3, 2)                       # <pq||rs>
    o, v = slice(None, 6), slice(6, None)
    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    e_hf = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    t = [np.zeros((6,) * (2 * k)) for k in (1, 2, 3)]
    t1, t2, _ = amplitudes.kernel(*t, fock, g, o, v,
                                  *energy_denominators(fock, 6, 6), e_hf,
                                  max_iter=300, stopping_eps=1e-11)
    e_so = amplitudes.cc_energy(t1, t2, fock, g, o, v)
    assert abs(e_hf - solver.last_result['e_reference']) < 1e-10
    assert abs(e_cc - e_so) < 1e-9


def test_refusals(h4):
    h1, eri, norb, nelec = h4
    solver = CoupledClusterSolver(order=3)
    with pytest.raises(ValueError, match='real Hermitian'):
        solver.roots(h1, eri + 1e-3 * np.random.default_rng(1).standard_normal(eri.shape),
                     norb, nelec)
    with pytest.raises(ValueError, match='at most 1 root'):
        solver.roots(h1, eri, norb, nelec, nroots=2)
    with pytest.raises(NotImplementedError, match='energies only'):
        solver.rdms(None, norb, nelec)
    with pytest.raises(ValueError, match='orders'):
        CoupledClusterSolver(order=5)
    with pytest.raises(NotImplementedError, match='closed-shell only'):
        CoupledClusterSolver(order=4).roots(h1, eri, norb, nelec, spin=2)

    g_eff = {1: h1, 2: eri.transpose(0, 2, 1, 3)}
    args = (norb, nelec)
    three_body = {**g_eff, 3: np.full((norb,) * 6, 1e-3)}
    for name in ('ccsdt', 'ccsdtq'):
        with pytest.raises(ValueError, match='one- and two-body'):
            solve(name, three_body, *args, spin_free=True)
    x = 1e-3 * np.random.default_rng(2).standard_normal(eri.shape)
    non_hermitian = {1: h1, 2: (eri + x - x.transpose(1, 0, 3, 2)).transpose(0, 2, 1, 3)}
    with pytest.raises(ValueError, match='needs a Hermitian Hamiltonian'):
        solve('ccsdt', non_hermitian, *args, spin_free=True, symmetrize=False)
    # one state, refused rather than truncated or padded
    with pytest.raises(ValueError, match='at most 1 root'):
        solve('ccsdt', g_eff, *args, spin_free=True, nroots=2)


def test_an_unconverged_iteration_warns_flags_and_keeps_its_energy(h4, monkeypatch):
    h1, eri, norb, nelec = h4
    monkeypatch.setattr(active_space_solver, 'CC_ACTIVE_MAX_CYCLE', 2)
    solver = CoupledClusterSolver(order=3)
    with pytest.warns(ActiveSpaceCCNotConverged, match='did not converge'):
        energies, _ = solver.roots(h1, eri, norb, nelec)
    record = solver.last_result
    assert not record['converged'] and np.isfinite(energies[0])
    assert [e['flag'] for e in record['numerical_flags']] == ['active_space_cc_not_converged']
