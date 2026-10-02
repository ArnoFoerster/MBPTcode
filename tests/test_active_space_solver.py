"""ActiveSpaceSolver (src/Solvers/active_space_solver.py): the FCI,
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
from pyscf import ao2mo, gto, mcscf, scf

from src.Solvers.active_space_solver import (ExactDiagonalizationSolver,
                                             PyscfFCISolver)

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


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
