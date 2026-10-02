"""Standalone check of Block2DMRGSolver, run in its own interpreter.

block2's wheel links its own OpenMP runtime; with an MKL build of NumPy,
importing block2 after NumPy aborts the process with `OMP: Error #15`, not a
catchable ImportError, so block2 must be imported before numpy/pyscf in this
process -- which is also why this
script cannot run inside the main test process via pytest.importorskip and is
instead invoked as a subprocess from test_active_space_solver.py.

Builds a plain CASCI(4,6) active space of water/6-31G (no downfolding, no
environment correction) and checks the DMRG adapter's singlet and triplet
energies against PyscfFCISolver, and that its rdms reconstruct its own
eigenvalue -- which is what proves the SU2 -> pyscf density-matrix convention
mapping in Block2DMRGSolver.rdms, not just convergence.
"""
import block2  # noqa: F401  -- MUST precede numpy

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf

from src.Solvers.active_space_solver import Block2DMRGSolver, PyscfFCISolver

H2O = 'O 0 0 0.117; H 0 0.755 -0.471; H 0 -0.755 -0.471'


def main():
    mol = gto.M(atom=H2O, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, 6, 4)
    mc.verbose = 0
    mc.kernel()
    h1eff, ecore = mc.get_h1eff()
    eri_cas = ao2mo.restore(1, mc.get_h2eff(), 6)

    fci = PyscfFCISolver()
    dmrg = Block2DMRGSolver(bond_dim=200, n_sweeps=20)

    ok = True
    for spin, nelec, label in ((0, (2, 2), 'singlet'), (2, (3, 1), 'triplet')):
        e_fci, _ = fci.roots(h1eff, eri_cas, 6, nelec, nroots=1, spin=spin)
        e_dmrg, vecs = dmrg.roots(h1eff, eri_cas, 6, nelec, nroots=1, spin=spin)
        d = abs(e_dmrg[0] - e_fci[0])
        this_ok = d < 1e-8
        ok &= this_ok
        print(f"  {label}: DMRG {e_dmrg[0] + ecore:.10f} Ha vs FCI "
              f"{e_fci[0] + ecore:.10f} Ha, |dE|={d:.2e}: {'OK' if this_ok else 'FAIL'}")

        if spin == 0:
            dm1, dm2 = dmrg.rdms(vecs[0], 6, nelec, spin=spin)
            e_rdm = (np.einsum('pq,qp->', h1eff, dm1)
                    + 0.5 * np.einsum('pqrs,pqrs->', eri_cas, dm2) + ecore)
            d_rdm = abs(e_rdm - mc.e_tot)
            rdm_ok = d_rdm < 1e-8
            ok &= rdm_ok
            print(f"  singlet: energy from DMRG rdms vs CASCI e_tot: "
                  f"|dE|={d_rdm:.2e}: {'OK' if rdm_ok else 'FAIL'}")

    if ok:
        print("DMRG ADAPTER VERIFIED")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
