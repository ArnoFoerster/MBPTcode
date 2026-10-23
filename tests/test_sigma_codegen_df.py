"""Regression test for DF x sigma composition: the matrix-free + density-
fitted C^(2) apply module (generatedADC/adc4_c2_apply_df_blocks.py, from
sigma_codegen.generate_adc_class_pair_apply(2, df=True)).

This is the ADC(5)-target emission form -- matrix-free AND DF-factorized in
one generated routine. It validates that composing the DF/RI term-rewrite
(df_codegen.dress_integral_factor: <pq||rs> -> B B) with the sigma-vector
emitter reproduces the dense block contracted with a trial vector:

    apply_c2_TAG_SPLIT(f, B, kd, ..., x)  ==  einsum('bra,ket->bra', dense, x)

with B the EXACT spin-orbital RI factor (naux = norb^2 eigh decomposition, no
fitting error), so the only thing under test is the DFxsigma composition, not
DF accuracy. A sanity check first confirms B reconstructs the antisymmetrized
g the dense path uses.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.SingleReference.DensityMatrix.mpn_density_driver import MPnDensityDriver
from src.SingleReference.common.mp_amplitudes import to_l
from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri, get_density_fitting_coefficients)

SPLITS = ['vt2', 'l2ft2', 'ft1']   # vt2 carries the 2-electron bracket (real DF work)


from src.CodeGeneration.pdaggerq_interface import pdaggerq_available


def _generate():
    """(dense, dfappl): adc4_c2_blocks and adc4_c2_apply_df_blocks, generated into a temporary directory
    (generated output is not part of the repository) and loaded from there."""
    import importlib.util
    import tempfile
    from src.CodeGeneration import codegen_common, generate_adcn, sigma_codegen
    out = tempfile.mkdtemp(prefix='sigma_codegen_')
    saved = codegen_common.ADC_OUT_DIR
    codegen_common.ADC_OUT_DIR = out
    try:
        generate_adcn.generate_adc_class_pair(2)
        sigma_codegen.generate_adc_class_pair_apply(2, df=True)
    finally:
        codegen_common.ADC_OUT_DIR = saved
    mods = []
    for name in ('adc4_c2_blocks', 'adc4_c2_apply_df_blocks'):
        spec = importlib.util.spec_from_file_location(name, os.path.join(out, 'adc4', name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods.append(mod)
    return tuple(mods)


def main():
    dense, dfappl = _generate()
    all_ok = True
    for atom, basis, label in [
        ('Li 0 0 0; H 0 0 1.6', 'sto-3g', 'LiH/sto-3g'),
        ('H 0 0 0; H 0 0 0.74; H 0 0 1.5; H 0 0 2.2', 'sto-3g', 'H4/sto-3g'),
    ]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc = 2 * (mol.nelectron // 2)
        eps = get_orbital_energies(mf, representation='spin')
        eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
        g = get_antisymmetrized_spin_eri(eri)
        norb, no = len(eps), nocc
        nv = norb - nocc
        B = get_density_fitting_coefficients(mol, mf, representation='spin')  # exact, (naux, nso, nso)

        # sanity: B reconstructs the antisymmetrized g the dense path uses
        g_rec = (np.einsum('Qpr,Qqs->pqrs', B, B, optimize=True)
                 - np.einsum('Qps,Qqr->pqrs', B, B, optimize=True))
        rec = np.abs(g_rec - g).max()
        ok = rec < 1e-10
        all_ok &= ok
        print(f"\n=== {label}  (no={no}, nv={nv}, naux={B.shape[0]}) ===")
        print(f"  [{'OK' if ok else 'FAIL'}] B reconstructs antisymmetrized g: {rec:.2e}")

        mpn = MPnDensityDriver(eps, g, nocc)
        t2_1 = mpn.compute_t2_1()
        t1_2, _, _, _ = mpn.compute_order2_amplitudes(t2_1)
        pool = {'f': np.diag(eps), 'g': g, 'B': B, 'kd': np.eye(norb),
                'o': slice(0, nocc), 'v': slice(nocc, norb), 'no': no, 'nv': nv,
                't2': t2_1, 'l2': to_l(t2_1, 2), 't1': t1_2, 'l1': to_l(t1_2, 1)}
        rng = np.random.default_rng(0)

        for tag in ('2h1p', '2p1h'):
            for split in SPLITS:
                dfn = getattr(dense, f'c2_{tag}_{split}')
                afn = getattr(dfappl, f'apply_c2_{tag}_{split}')
                T = dfn(**{p: pool[p] for p in inspect.signature(dfn).parameters})
                nbra = T.ndim // 2
                x = rng.standard_normal(T.shape[nbra:])
                ref = np.tensordot(T, x, axes=(list(range(nbra, T.ndim)),
                                               list(range(x.ndim))))
                akw = {p: pool[p] for p in inspect.signature(afn).parameters if p != 'x'}
                y = afn(x=x, **akw)
                diff = np.abs(y - ref).max()
                ok = diff < 1e-11
                all_ok &= ok
                print(f"  [{'OK' if ok else 'FAIL'}] DF apply_c2_{tag}_{split:6s}: "
                      f" max|DF-apply - dense@x| = {diff:.2e}")

    print("\nALL OK" if all_ok else "\nSOME FAILED")
    return all_ok


@pytest.mark.skipif(not pdaggerq_available(), reason='needs pdaggerq (optional)')
def test_main():
    assert main()


if __name__ == '__main__':
    if not pdaggerq_available():
        print('SKIPPED: needs pdaggerq (optional)')
        sys.exit(0)
    sys.exit(0 if main() else 1)
