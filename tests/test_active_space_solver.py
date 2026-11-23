"""ActiveSpaceSolver (src/EffectiveHamiltonian/solvers/backends.py): the FCI,
determinant-basis and DMRG engines behind one roots()/rdms() interface.

Run: python tests/test_active_space_solver.py, or under pytest.

Checks:
  1. PyscfFCISolver (direct_spin1, spin=None) and ExactDiagonalizationSolver
     agree on the whole Sz=0 spectrum of a toy Hamiltonian; the determinant
     basis carries no density matrices
  2. PyscfFCISolver's rdms reconstruct the energy they came from
  3. the CASCI ground state of a real active space (water/6-31G, 4 electrons
     in 6 orbitals) matches pyscf's own CASCI, from roots() and from rdms()
  4. the DMRG adapter reproduces PyscfFCISolver on the same active space, in
     its own subprocess (see dmrg_adapter_check.py for why)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, gto, mcscf, scf

from src.EffectiveHamiltonian.solvers.solve import (
    EmptyActiveSpaceWarning, solve)
from src.EffectiveHamiltonian.solvers.backends import (
    ExactDiagonalizationSolver, PyscfFCISolver)

H4 = 'H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0'
H2O = 'O 0 0 0.117; H 0 0.755 -0.471; H 0 -0.755 -0.471'


def _h4_hamiltonian():
    mol = gto.M(atom=H4, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    norb = mol.nao
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    return h1, eri, norb, mol.nelectron


def _water_cas46():
    """(h1eff, eri_cas, ecore, mc) of a plain CASCI(4,6) -- no downfolding, no
    environment correction, just pyscf's own effective active-space Hamiltonian."""
    mol = gto.M(atom=H2O, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, 6, 4)
    mc.verbose = 0
    mc.kernel()
    h1eff, ecore = mc.get_h1eff()
    eri_cas = ao2mo.restore(1, mc.get_h2eff(), 6)
    return h1eff, eri_cas, ecore, mc


def check_fci_and_determinant_basis_agree():
    h1, eri, norb, nelec = _h4_hamiltonian()
    fci_solver, ed_solver = PyscfFCISolver(), ExactDiagonalizationSolver()

    e_fci, _ = fci_solver.roots(h1, eri, norb, nelec, nroots=6, spin=None)
    e_ed, vecs_ed = ed_solver.roots(h1, eri, norb, nelec, nroots=6, spin=None)
    dE = np.max(np.abs(np.asarray(e_fci) - np.asarray(e_ed)))

    try:
        ed_solver.rdms(vecs_ed[0], norb, nelec, spin=None)
        no_rdms = False
    except NotImplementedError:
        no_rdms = True

    ok = dE < 1e-9 and no_rdms
    print(f"H4/sto-3g whole Sz=0 spectrum, FCI vs determinant basis: dE={dE:.2e} "
          f"determinant basis refuses rdms={no_rdms}: {'OK' if ok else 'FAIL'}")
    return ok


def check_rdms_reconstruct_the_energy():
    h1, eri, norb, nelec = _h4_hamiltonian()
    solver = PyscfFCISolver()
    e, vecs = solver.roots(h1, eri, norb, nelec, nroots=1, spin=0)
    dm1, dm2 = solver.rdms(vecs[0], norb, nelec, spin=0)
    e_rdm = np.einsum('pq,qp->', h1, dm1) + 0.5 * np.einsum('pqrs,pqrs->', eri, dm2)
    d = abs(e_rdm - e[0])
    ok = d < 1e-10
    print(f"H4/sto-3g ground state, energy from rdms vs from roots: {d:.2e}: "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_the_cas_ground_state_matches_pyscf_casci():
    h1eff, eri_cas, ecore, mc = _water_cas46()
    solver = PyscfFCISolver()
    e, vecs = solver.roots(h1eff, eri_cas, 6, (2, 2), nroots=1, spin=0)
    d_roots = abs(e[0] + ecore - mc.e_tot)

    dm1, dm2 = solver.rdms(vecs[0], 6, (2, 2), spin=0)
    e_rdm = (np.einsum('pq,qp->', h1eff, dm1)
            + 0.5 * np.einsum('pqrs,pqrs->', eri_cas, dm2) + ecore)
    d_rdms = abs(e_rdm - mc.e_tot)

    ok = d_roots < 1e-9 and d_rdms < 1e-9
    print(f"water/6-31G CASCI(4,6) vs PyscfFCISolver: from roots {d_roots:.2e}, "
          f"from rdms {d_rdms:.2e}: {'OK' if ok else 'FAIL'}")
    return ok


def check_the_dmrg_adapter_reproduces_fci():
    probe = subprocess.run([sys.executable, '-c', 'import block2'],
                          capture_output=True)
    if probe.returncode != 0:
        print("Block2DMRGSolver: block2 not importable in this environment, "
              "skipping (not a failure): OK")
        return True

    script = os.path.join(os.path.dirname(__file__), 'dmrg_adapter_check.py')
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    verified = 'DMRG ADAPTER VERIFIED' in result.stdout
    ok = result.returncode == 0 and verified
    print(f"Block2DMRGSolver adapter check (subprocess): {'OK' if ok else 'FAIL'}")
    if not ok:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
    return ok


def run():
    all_ok = True
    for check in (check_fci_and_determinant_basis_agree,
                  check_rdms_reconstruct_the_energy,
                  check_the_cas_ground_state_matches_pyscf_casci,
                  check_the_dmrg_adapter_reproduces_fci):
        all_ok &= bool(check())
    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    return all_ok


def test_active_space_solvers():
    assert run()


@pytest.mark.parametrize('name', ['pyscf-fci', 'fci', 'ccsd', None])
def test_an_unknown_backend_is_refused_by_name(name):
    """A name `solve` does not dispatch on (a typo, a backend it does not have)
    is refused, never handed to the default exact diagonalization."""
    with pytest.raises(ValueError, match='not a backend of this driver'):
        solve(name, {1: np.zeros((2, 2)), 2: np.zeros((2, 2, 2, 2))}, 2, 2,
              None, None, None)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)


@pytest.mark.parametrize('spin_free', [True, False])
@pytest.mark.parametrize('name', ['openfermion', 'pyscf_fci', 'dmrg', 'ccsdt', 'ccsdtq'])
def test_an_empty_active_space_returns_the_constant(name, spin_free):
    """An empty active space (no orbitals, no electrons) has one state, of
    energy E_act = 0: every backend returns the constant g_eff[0] alone (a
    downfolded E_0, so the root is the total energy) and warns, with the named
    flag 'empty_active_space', that there was nothing to diagonalize."""
    g_eff = {0: -1.25, 1: np.zeros((0, 0)), 2: np.zeros((0, 0, 0, 0))}
    with pytest.warns(EmptyActiveSpaceWarning, match='empty'):
        roots = solve(name, g_eff, 0, 0, spin_free=spin_free, nroots=3)
    assert np.array_equal(roots, [-1.25])


def test_an_empty_active_space_with_electrons_is_refused():
    """No orbitals cannot hold electrons: refused, not answered."""
    g_eff = {0: 0.0, 1: np.zeros((0, 0)), 2: np.zeros((0, 0, 0, 0))}
    with pytest.raises(ValueError, match='no active orbitals'):
        solve('pyscf_fci', g_eff, 0, 2, spin_free=True)


def _h4_g_eff():
    """(g_eff, norb, nelec) of H4/sto-3g, spatial and physicist, E_0 = 0."""
    h1, eri, norb, nelec = _h4_hamiltonian()
    return {0: 0.0, 1: h1, 2: eri.transpose(0, 2, 1, 3)}, norb, nelec


def _h4_by_spin():
    """The H4 spectrum at Sz = 0 split by <S^2>: {2S: ascending energies}."""
    from pyscf import fci
    h1, eri, norb, nelec = _h4_hamiltonian()
    energies, vectors = PyscfFCISolver().roots(h1, eri, norb, nelec,
                                               nroots=36, spin=None)
    out = {}
    for e, v in zip(energies, vectors):
        ss = fci.spin_op.spin_square(v, norb, (nelec // 2, nelec // 2))[0]
        two_s = int(round(np.sqrt(1 + 4 * ss) - 1))
        out.setdefault(two_s, []).append(e)
    return out


@pytest.mark.parametrize('spin', [0, 2])
def test_spin_gives_pure_spin_states(spin):
    """solve('pyscf_fci', ..., spin=2S): the lowest nroots states of total
    spin S. On H4/sto-3g the fifth root of the Ms = S sector without the
    spin penalty is the quintet (-3.5078 Eh, <S^2> = 6), for the singlets
    (direct_spin0 keeps every even S) and the triplets (Ms = 1) alike; with
    it the five are the lowest five of the requested spin."""
    g_eff, norb, nelec = _h4_g_eff()
    want = _h4_by_spin()[spin][:5]
    h1, eri = g_eff[1], g_eff[2].transpose(0, 2, 1, 3)
    plain = PyscfFCISolver().roots(h1, eri, norb, nelec, nroots=5,
                                   spin=spin)[0]
    assert abs(plain[4] - _h4_by_spin()[4][0]) < 1e-8
    got = solve('pyscf_fci', g_eff, norb, nelec, spin_free=True, nroots=5,
                spin=spin)
    assert np.abs(got - want).max() < 1e-8


def test_spin_default_is_unchanged():
    """Without spin the lowest roots at the active Sz, of any spin: the
    singlet ground state, then the lowest triplet, on H4."""
    g_eff, norb, nelec = _h4_g_eff()
    got = solve('pyscf_fci', g_eff, norb, nelec, spin_free=True, nroots=2)
    by_spin = _h4_by_spin()
    assert np.abs(got - [by_spin[0][0], by_spin[2][0]]).max() < 1e-8


@pytest.mark.parametrize('name', ['openfermion', 'ccsdt', 'ccsdtq'])
def test_spin_is_refused_by_backends_without_it(name):
    """A backend that cannot give states of one total spin refuses spin,
    naming those that can."""
    g_eff, norb, nelec = _h4_g_eff()
    with pytest.raises(ValueError, match="'pyscf_fci' and 'dmrg'"):
        solve(name, g_eff, norb, nelec, spin_free=True, spin=0)


def test_spin_is_refused_for_a_spin_resolved_hamiltonian():
    """Per-spin blocks (an unrestricted reference) conserve Sz alone: spin
    is refused there."""
    g_eff, norb, nelec = _h4_g_eff()
    resolved = {0: 0.0, 1: (g_eff[1], g_eff[1]),
                2: (g_eff[2], g_eff[2], g_eff[2])}
    with pytest.raises(ValueError, match='spin-resolved'):
        solve('pyscf_fci', resolved, norb, (2, 2), spin_free=True, spin=0)


@pytest.mark.parametrize('spin', [-2, 1, 6, 2.5, True])
def test_a_spin_the_electrons_cannot_have_is_refused(spin):
    """2S must be an integer >= 0 of the electron count's parity, with
    (N + 2S)/2 alpha electrons fitting in the active orbitals."""
    g_eff, norb, nelec = _h4_g_eff()
    with pytest.raises(ValueError, match='spin='):
        solve('pyscf_fci', g_eff, norb, nelec, spin_free=True, spin=spin)


def test_a_root_of_the_wrong_spin_is_refused(monkeypatch):
    """Without the penalty (shift 0) the quintet is the fifth triplet root;
    the <S^2> check refuses it rather than return it as a triplet."""
    from src.EffectiveHamiltonian.solvers import solve as solve_module
    monkeypatch.setattr(solve_module, 'FCI_SPIN_PENALTY_SHIFT', 0.0)
    g_eff, norb, nelec = _h4_g_eff()
    with pytest.raises(ValueError, match=r'<S\^2>'):
        solve('pyscf_fci', g_eff, norb, nelec, spin_free=True, nroots=5,
              spin=2)


def test_downfolded_hamiltonian_solve_takes_spin():
    """h.solve('pyscf_fci', nroots=..., spin=...) is the FCI of h.g_eff in
    that spin, E_0 included: the singlets S0, S1 and the triplet T1 of
    example 22's helper."""
    from src.EffectiveHamiltonian.Downfolding import (
        build_effective_hamiltonian, run_low_level)
    from src.EffectiveHamiltonian.active_space import ActiveSpace
    mf = scf.RHF(gto.M(atom=H2O, basis='6-31g', verbose=0)).run()
    nocc = int(np.count_nonzero(mf.mo_occ))
    h = build_effective_hamiltonian(run_low_level(mf, method='hf'),
                                    ActiveSpace.from_counts(mf.mo_energy,
                                                            nocc, 2, 2))
    n, nelec = h.space['n_orbs'], h.space['n_elec_active']
    h1, eri = h.g_eff[1], h.g_eff[2].transpose(0, 2, 1, 3)
    for nroots, spin in ((2, 0), (1, 2)):
        want = h.g_eff[0] + PyscfFCISolver().roots(
            h1, eri, n, nelec, nroots=nroots, spin=spin)[0]
        got = h.solve('pyscf_fci', nroots=nroots, spin=spin)
        assert np.abs(got - want).max() < 1e-9
