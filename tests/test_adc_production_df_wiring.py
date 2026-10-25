"""Regression test for the production DF/RI wiring: a genuine end-to-end DF
pipeline -- one auxiliary basis used for SCF, ADC, and density/static-
correction building -- runs through the STANDARD production entry points, not
just the validated-but-opt-in `_df`-suffixed methods.

Covers:
1. src.SingleReference.ADC.static_correction._static_correction_from_dgamma_restricted_df
   (the RI-J/RI-K form of the static correction).
2. src.SingleReference.DensityMatrix.mpn_density_driver_restricted
   .MPnDensityDriverRestricted.compute_delta_gamma23_df (mirrors the dense
   compute_delta_gamma23 but calls the _df methods).
3. build_mp2_static_correction_restricted/build_mp3_static_correction_restricted's
   B_aa= parameter (unrelaxed AND relaxed -- relax=True exercises
   solve_cphf_relaxation, an independent code path DF doesn't touch, so this
   is also a check that DF density plugs correctly into an otherwise-
   untouched pipeline).
4. ADCSolverRestricted's df= flag: df=True on a DF mean field sets .B_aa and
   skips the dense eri, the default leaves .B_aa unset, df=True on a non-DF
   mean field raises; end to end, the DF solve matches the dense reference at
   RI level. level='adc4' without a pdaggerq construction raises.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.SingleReference.ADC import ADCSolverRestricted
from src.SingleReference.ADC.static_correction import (
    build_static_correction,
    _static_correction_from_dgamma_restricted, _static_correction_from_dgamma_restricted_df,
    build_mp2_static_correction_restricted, build_mp3_static_correction_restricted,
)
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import MPnDensityDriverRestricted
from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies, get_antisymmetrized_spin_block_eri, get_two_electron_integrals_chemist, DFIntegrals,
)
from tests.generated_runtime import requires_runtime

pytestmark = requires_runtime(4)


def main():
    all_ok = True

    mol = gto.M(atom='H 0 0 0; Li 0 0 1.6', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()

    # --- 1. _static_correction_from_dgamma_restricted_df ---
    rng = np.random.default_rng(0)
    norb = mf.mo_coeff.shape[1]
    dgamma = rng.standard_normal((norb, norb))
    dgamma = 0.5 * (dgamma + dgamma.T)
    eri_chemist = get_two_electron_integrals_chemist(mol, mf)
    dense_gk = _static_correction_from_dgamma_restricted(eri_chemist, dgamma)
    dfi_exact = DFIntegrals.from_scf(mol, mf, exact=True)
    df_gk = _static_correction_from_dgamma_restricted_df(dfi_exact.B_aa, dgamma)
    diff = np.max(np.abs(dense_gk - df_gk))
    ok = diff < 1e-10
    all_ok &= ok
    print(f"_static_correction_from_dgamma_restricted_df matches dense J-0.5K "
          f"(max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    # --- 2. compute_delta_gamma23_df ---
    nocc = mol.nelectron // 2
    eps_a = get_orbital_energies(mf, representation='spatial')
    g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
    f_aa = np.diag(eps_a)
    dense_drv = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc)
    g2d, g3d = dense_drv.compute_delta_gamma23()
    df_drv = MPnDensityDriverRestricted(f_aa, None, None, None, nocc, B_aa=dfi_exact.B_aa, B_bb=dfi_exact.B_bb)
    g2f, g3f = df_drv.compute_delta_gamma23_df()
    md = lambda a, b: max(np.max(np.abs(x - y)) for x, y in zip(a, b))
    diff2, diff3 = md(g2d, g2f), md(g3d, g3f)
    ok = diff2 < 1e-10 and diff3 < 1e-10
    all_ok &= ok
    print(f"compute_delta_gamma23_df matches dense compute_delta_gamma23 "
          f"(gamma2 diff={diff2:.2e}, gamma3 diff={diff3:.2e}): {'OK' if ok else 'FAIL'}")

    # --- 3. build_mp2/mp3_static_correction_restricted(B_aa=...), unrelaxed + relaxed ---
    for relax in (False, True):
        dense2 = build_mp2_static_correction_restricted(mf, relax=relax)
        df2 = build_mp2_static_correction_restricted(mf, relax=relax, B_aa=dfi_exact.B_aa)
        diff = np.max(np.abs(dense2 - df2))
        ok = diff < 1e-9
        all_ok &= ok
        print(f"build_mp2_static_correction_restricted(relax={relax}) DF matches dense "
              f"(max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    dense3 = build_mp3_static_correction_restricted(mf, relax=False)
    df3 = build_mp3_static_correction_restricted(mf, relax=False, B_aa=dfi_exact.B_aa)
    diff = np.max(np.abs(dense3 - df3))
    ok = diff < 1e-9
    all_ok &= ok
    print(f"build_mp3_static_correction_restricted(relax=False) DF matches dense "
          f"(max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    # --- 4. the df= flag wiring on the unified solver ---
    mol2 = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', verbose=0)
    mf_dense = scf.RHF(mol2).run()
    homo_index = mol2.nelectron // 2 - 1
    mf_df = scf.RHF(mol2).density_fit(auxbasis='cc-pvdz-jkfit').run()

    # unit-level: the DF DECISION itself, directly on the solver
    s_dense = ADCSolverRestricted(mf_dense, level='adc3')
    s_true = ADCSolverRestricted(mf_df, level='adc3', df=True)
    s_false = ADCSolverRestricted(mf_df, level='adc3')
    ok = s_dense.B_aa is None and s_true.B_aa is not None and s_false.B_aa is None
    all_ok &= ok
    print(f"df= flag -> .B_aa: default=None, df=True=set, df=False on DF-mf=None: "
          f"{'OK' if ok else 'FAIL'}")
    # memory-lean invariant: the hand-written DF route never builds dense eri
    ok = s_true.eri is None and s_dense.eri is not None
    all_ok &= ok
    print(f"df=True skips the dense eri build (eri=None), dense keeps it: "
          f"{'OK' if ok else 'FAIL'}")

    # end-to-end: DF solve matches the dense reference at RI level, with the
    # dressed static built through the promoted dispatcher (B_aa threaded)
    sc_dense = build_static_correction(mf_dense, kind='mp3_unrelaxed')
    e_ref, _ = ADCSolverRestricted(mf_dense, level='adc3').solve(
        static_correction=sc_dense, homo_index=homo_index, nroots=1)
    sc_df = build_static_correction(mf_df, kind='mp3_unrelaxed', B_aa=s_true.B_aa)
    e_df, _ = s_true.solve(static_correction=sc_df, homo_index=homo_index, nroots=1)
    diff = abs(e_ref[0] - e_df[0])
    ok = diff < 1e-3
    all_ok &= ok
    print(f"df=True end-to-end matches dense-mf reference at RI level "
          f"(pole diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    try:
        ADCSolverRestricted(mf_dense, level='adc3', df=True)
        ok = False
    except ValueError:
        ok = True
    all_ok &= ok
    print(f"df=True on a non-DF mf raises ValueError: {'OK' if ok else 'FAIL'}")

    try:
        ADCSolverRestricted(mf_dense, level='adc4', construction='twosector')
        ok = False
    except ValueError:
        ok = True
    all_ok &= ok
    print(f"level='adc4' without pdaggerq=True raises ValueError: "
          f"{'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    return all_ok


def test_main():
    assert main()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
