"""The BSE front door and its guards, end to end.

Covers what landed with `solve_bse_isdf`:

  1. The driver itself on water / LRC-wPBEh: BSE@G0W0 and BSE@mean-field pins,
     factor reuse across calls, and the info payload.
  2. Oscillator strengths against pyscf's own TDHF on the IDENTICAL DF
     operator -- the sqrt(2) / (2/3) normalization for this module's
     <X|X>-<Y|Y>=1 spatial convention is a measured fact here, not a cited one.
  3. The instability machinery on 90-degree twisted ethene, the textbook
     singlet-unstable reference: the matrix-free min eig(A-B) probe against
     the dense value, the solver's RuntimeError diagnosis (never pyscf's bare
     LinAlgError), and the driver refusing BEFORE the solve.
     NOTE mf.stability() calls this system internally stable -- it is a
     different Hessian and no substitute for the probe.
  4. The DF block action is invariant to the trial-vector chunking that keeps
     its (nvec, naux, nvirt, nvirt) exchange intermediate bounded.

Run: python tests/test_bse_isdf_driver.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import dft, gto, scf

from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_density_fitting_coefficients)
from src.SingleReference.LinearResponse.davidson import (
    df_block_action, lowest_amb_eigenvalue, oscillator_strengths,
    solve_bse_isdf, solve_casida_davidson)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.Base.constants import HARTREE_TO_EV

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
TWISTED_ETHENE = """C 0 0 0.669; C 0 0 -0.669;
                    H 0  0.923 1.238; H 0 -0.923 1.238;
                    H  0.923 0 -1.238; H -0.923 0 -1.238"""


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def main():
    ok = True

    print('\n=== solve_bse_isdf on water / LRC-wPBEh / cc-pVDZ ===')
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = dft.RKS(mol, xc='LRC-WPBEh').density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    om, X, Y, info = solve_bse_isdf(mf, mol, nocc, nroots=3,
                                    auxbasis='cc-pvdz-ri')
    om0, _, _, info0 = solve_bse_isdf(mf, mol, nocc, nroots=3, qp=False,
                                      factors=info['factors'])
    # Regression pins: the ISDF grid is deterministic, so these hold to far
    # better than the tolerance.
    #
    # Up from 7.8613 and 5.0724, which were fitted on H and O radii from a
    # single ungated descent. The radii table holds one row per
    # (element, basis, auxbasis, counts) now, so one request can no longer be
    # served different grids, and the probe-gated multi-start row wins. Against
    # `solve_bse_df` on the same mean field the lowest root sits 18.2 meV out,
    # where the ungated radii sat 11.3 meV: the gated grid is the
    # better-evidenced one, not the closer one on this molecule.
    d_gw = abs(np.sort(om)[0] * HARTREE_TO_EV - 7.8682)
    d_ks = abs(np.sort(om0)[0] * HARTREE_TO_EV - 5.0778)
    ok &= check(d_gw < 5e-3, "qp='G0W0' lowest root pin 7.8682 eV", f'|d| {d_gw:.1e}')
    ok &= check(d_ks < 5e-3, 'qp=False lowest root pin 5.0778 eV', f'|d| {d_ks:.1e}')
    ok &= check((np.asarray(om) > 0).all() and info['min_eig_amb'] > 0
                and np.isfinite(info['oscillator_strength']).all()
                and 'qp' in info['timings'] and 'factors' not in info0['timings'],
                'info payload complete; factors reused, not rebuilt')
    gap_opens = (info['eps'][nocc] - info['eps'][nocc - 1]
                 > info['eps_mf'][nocc] - info['eps_mf'][nocc - 1])
    ok &= check(gap_opens, 'G0W0 opens the KS gap')

    print('\n=== oscillator strengths == pyscf TDHF, same DF operator ===')
    mf_hf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf_hf.kernel()
    td = mf_hf.TDHF()
    td.nstates = 5
    td.kernel()
    f_ref = np.sort(td.oscillator_strength(gauge='length'))
    eps = get_orbital_energies(mf_hf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=get_density_fitting_coefficients(
        mol, mf_hf, representation='spatial'), spin_mode='restricted')
    om_td, X_td, Y_td = solve_casida_davidson(lr, nocc, nroots=5,
                                              polarizability='TDHF')
    f_mine, _ = oscillator_strengths(mf_hf, mol, nocc, om_td, X_td, Y_td)
    d_f = np.abs(np.sort(f_mine) - f_ref).max()
    ok &= check(d_f < 1e-10, 'f matches pyscf to machine precision', f'max |df| {d_f:.1e}')

    print('\n=== instability machinery on twisted ethene ===')
    mol_e = gto.M(atom=TWISTED_ETHENE, basis='cc-pvdz', verbose=0)
    mf_e = scf.RHF(mol_e).density_fit(auxbasis='cc-pvdz-ri')
    mf_e.kernel()
    nocc_e = mol_e.nelectron // 2
    eps_e = get_orbital_energies(mf_e, representation='spatial')
    lr_e = LinearResponseSolver(eps_e, coeff_df=get_density_fitting_coefficients(
        mol_e, mf_e, representation='spatial'), spin_mode='restricted')
    A, B = lr_e.build_casida_matrices(nocc_e, lBSE=True, W_aux=None)
    amb_dense = np.linalg.eigvalsh(A - B).min()
    amb_probe = float(lowest_amb_eigenvalue(lr_e, nocc_e, polarizability='TDHF')[0])
    ok &= check(amb_dense < 0, 'reference is singlet-unstable (dense)',
                f'{amb_dense:+.4f} Ha')
    ok &= check(abs(amb_probe - amb_dense) < 1e-8,
                'matrix-free probe == dense min eig(A-B)',
                f'|d| {abs(amb_probe - amb_dense):.1e}')
    # The probe starts its Lanczos from a FIXED vector, 1/d, so it is a
    # function of the data alone: the same call again in the same process is
    # the same number bit for bit. ARPACK's own start comes from a seed it
    # keeps between calls, so without it the second call starts elsewhere and
    # lands elsewhere inside its tolerance.
    amb_again = float(lowest_amb_eigenvalue(lr_e, nocc_e, polarizability='TDHF')[0])
    ok &= check(amb_again == amb_probe,
                'the probe is reproducible bitwise within one process',
                f'|d| {abs(amb_again - amb_probe):.1e}')
    try:
        solve_casida_davidson(lr_e, nocc_e, nroots=3, polarizability='TDHF')
        ok &= check(False, 'solver raises on the unstable reference')
    except RuntimeError as exc:
        ok &= check('not positive definite' in str(exc),
                    'solver raises the DIAGNOSIS, not a bare LinAlgError')
    except np.linalg.LinAlgError:
        ok &= check(False, 'bare LinAlgError leaked through the guard')
    # On this reference (the LOWEST RHF solution of twisted ethene -- pyscf's
    # default guess lands on a second solution 31 mHa HIGHER for slightly
    # different coordinates, where both minima are far more negative), TDHF is
    # indefinite while BSE is positive: the screening moves min eig(A-B)
    # upward. The probe is a property of the reference AND the kernel, so the
    # driver must SOLVE this system; its refusal is exercised below with a
    # diagonal that is unstable by construction.
    W_e = np.asarray(lr_e.static_screening_aux(nocc_e))
    amb_bse = float(lowest_amb_eigenvalue(lr_e, nocc_e, polarizability='BSE',
                                          W_aux=W_e)[0])
    ok &= check(amb_bse > 0 > amb_dense,
                'TDHF and BSE differ in sign on the SAME reference -- probe the kernel in use',
                f'BSE {amb_bse:+.4f}, TDHF {amb_dense:+.4f} Ha')

    print('\n=== driver refuses an unstable diagonal before the solve ===')
    mol_w = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf_w = dft.RKS(mol_w, xc='LRC-WPBEh').density_fit(auxbasis='cc-pvdz-ri')
    mf_w.kernel()
    nocc_w = mol_w.nelectron // 2
    eps_inv = get_orbital_energies(mf_w, representation='spatial').copy()
    eps_inv[nocc_w - 1] += 1.0            # HOMO pushed far above LUMO
    try:
        solve_bse_isdf(mf_w, mol_w, nocc_w, nroots=3, qp=eps_inv,
                       auxbasis='cc-pvdz-ri')
        ok &= check(False, 'driver refuses before the solve')
    except RuntimeError as exc:
        ok &= check('refused before the solve' in str(exc),
                    'driver refuses before the solve, naming min eig(A-B)')

    print('\n=== DF block action invariant to trial-vector chunking ===')
    W_df = lr.static_screening_aux(nocc)
    act, _ = df_block_action(lr, nocc, True, W_df)
    act1, _ = df_block_action(lr, nocc, True, W_df, tile_memory_gb=1e-9)
    z = np.random.default_rng(0).standard_normal((5, nocc, len(eps) - nocc))
    a, b = act(z)
    a1, b1 = act1(z)
    d_c = max(np.abs(a - a1).max() / np.abs(a).max(),
              np.abs(b - b1).max() / np.abs(b).max())
    ok &= check(d_c < 1e-12, 'chunk-per-vector == one batch', f'rel {d_c:.1e}')

    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return 0 if ok else 1


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    sys.exit(main())
