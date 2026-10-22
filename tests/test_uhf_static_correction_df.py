"""Parity check for build_mp2_static_correction_uhf_df -- the committed test
its docstring's "Validated identical to the dense route at an exact DF factor
(see tests)" claim points to.

Two tiers, same split as tests/test_mp2_density_df.py:

1. Exact-plumbing: B_so from get_uhf_spin_orbital_df_factor_blockstacked(...,
   exact=True) (naux=norb^2 eigh fallback, not real DF), direct builder vs
   builder at ~1e-10. What this isolates is the DF builder's OWN plumbing --
   the blockstacked spin-orbital density build, the EN singles-denominator
   dressing replication, and the rank<=3 final G[dgamma] contraction -- not
   the shared Z-vector solve: both routes call the same
   solve_cphf_relaxation_uhf on the same mf (that solver has its own oracle,
   tests/test_uhf_mp2_relaxed_finite_field.py). Both solves are taken to
   CORRELATION_CPHF_TOL, and the bar is that same number: the test compares
   the two builders, not the solver's stop points. At that tolerance both
   routes' solves stop in the same refinement round of src.Base.utils.cphf
   (OH relaxed EN 1.0e-14 apart, workstation). A tighter tolerance measures
   the solver instead: OH's orbital Hessian is nearly singular (lowest
   eigenvalue about 2.5e-10, tests/test_cphf_residual.py), so the last bits
   of the two right-hand sides, amplified along that soft rotation, decide
   where each solve stops (OH relaxed EN: 6.3e-12 at 1e-12, 5.0e-10 at
   1e-13 on the workstation; 1.6e-9 at 1e-12 under MKL).
2. Realistic-RI, through the production dispatcher: build_static_correction(
   mf, mol, kind=..., df=True) vs df=False on a density-fitted mf. The dense
   branch always rebuilds exact ao2mo integrals regardless of mf.with_df and
   the CPHF Hessian comes from the same mf.gen_response on both sides, so the
   difference IS the RI error of the density + final contraction (~1e-4).

Covers kind='mp2_relaxed' and 'mp2_unrelaxed', bare and EN-dressed
(u2_denom_dress={'hh': True, 'pp': True}), on OH/6-31G; NH2/sto-3g repeats
the fullest combo (relaxed + dressed) as a second genuine open-shell geometry.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC.static_correction import (
    build_static_correction,
    build_mp2_static_correction, build_mp2_static_correction_uhf_df,
)
from src.Base.constants import CORRELATION_CPHF_TOL
from src.Base.pyscf_interface import get_uhf_spin_orbital_df_factor_blockstacked


if __name__ == '__main__':
    all_ok = True
    DRESS = {'hh': True, 'pp': True}

    CASES = [
        ('O 0 0 0; H 0 0 0.97', '6-31g', 1, 'OH radical',
         [(False, None), (False, DRESS), (True, None), (True, DRESS)]),
        ('N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6', 'sto-3g', 1, 'NH2 radical',
         [(True, DRESS)]),
    ]
    for atom, basis, spin, label, combos in CASES:
        mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
        mf = scf.UHF(mol).density_fit().run()
        B_exact = get_uhf_spin_orbital_df_factor_blockstacked(mol, mf, exact=True)

        for relax, dress in combos:
            kind = 'mp2_relaxed' if relax else 'mp2_unrelaxed'
            dname = 'hh+pp EN' if dress else 'bare'

            dense = build_mp2_static_correction(
                mf, mol, relax=relax, u2_denom_dress=dress,
                cphf_tol=CORRELATION_CPHF_TOL)
            df_exact = build_mp2_static_correction_uhf_df(
                mf, mol, B_exact, relax=relax, u2_denom_dress=dress,
                cphf_tol=CORRELATION_CPHF_TOL)
            diff = np.max(np.abs(dense - df_exact))
            ok = diff < CORRELATION_CPHF_TOL
            all_ok &= ok
            print(f"{label} {kind:14s} ({dname:8s}): DF (naux=norb^2 exact) "
                  f"matches dense (max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

            sc_dense = build_static_correction(mf, mol, kind=kind, en_dress=dress)
            sc_df = build_static_correction(mf, mol, kind=kind, en_dress=dress,
                                            df=True)
            diff_ri = np.max(np.abs(sc_dense - sc_df))
            ok_ri = diff_ri < 1e-4
            all_ok &= ok_ri
            print(f"{label} {kind:14s} ({dname:8s}): dispatcher df=True (real "
                  f"JK-fit RI) matches df=False (max diff={diff_ri:.2e}): "
                  f"{'OK' if ok_ri else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
