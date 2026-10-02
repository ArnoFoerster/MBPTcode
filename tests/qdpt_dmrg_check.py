"""Standalone check of the QDPT driver's 'dmrg' backend, run in its own
interpreter for the same reason as dmrg_adapter_check.py: block2 must be
imported before numpy/pyscf in this process, or its OpenMP runtime clashes
with numpy's MKL one and aborts the process -- not a catchable ImportError,
so this can't run via pytest.importorskip in the main test process and is
instead invoked as a subprocess from test_qdpt_perturbative.py.

Downfolds H4/sto-3g (QDPT PT2, max_vertex=2) and checks that solver='dmrg'
agrees with solver='pyscf_fci' on the same g_eff.
"""
import block2  # noqa: F401  -- MUST precede numpy

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import gto, scf

from src.MultiReference.QDPT.perturbative import run_pipeline

H4 = 'H 0 0 0; H 1.8 0 0; H 0.54 2.34 0; H 2.52 1.62 0.9'


def main():
    mol = gto.M(atom=H4, basis='sto-3g', unit='Bohr', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-9)
    kw = dict(max_pt_order=2, max_vertex=2, mol=mol, mf=mf, n_occ_spatial=0,
              n_act_spatial=3)

    e_fci = run_pipeline(solver='pyscf_fci', **kw)
    e_dmrg = run_pipeline(solver='dmrg', **kw)
    d = abs(e_dmrg[0] - e_fci[0])
    ok = d < 1e-2  # downfolded g_eff is not exactly Hermitian
    print(f"dmrg {e_dmrg[0]:.10f} Ha vs pyscf_fci {e_fci[0]:.10f} Ha, "
          f"|dE|={d:.2e}: {'OK' if ok else 'FAIL'}")

    if ok:
        print("QDPT DMRG BACKEND VERIFIED")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
