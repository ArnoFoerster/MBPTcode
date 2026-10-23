"""Regression test for the ADC(5)-critical sigma-vector rung: the generated
matrix-free APPLY of the k=1<->k=2 coupling over a 3h2p/3p2h KET
(generatedADC/adc4_k2_apply_blocks.py, from sigma_codegen.py).

Validates each generated apply_c1_TAG1_TAG2 against the dense
generated coupling (adc4_k2_blocks.py c1_TAG1_TAG2) contracted with a
trial vector over the 3h2p/3p2h ket (5 free indices: 3 antisymmetric holes
+ 2 antisymmetric particles):

    apply_c1_2h1p_3h2p(..., x)  ==  einsum('ija klmbc, klmbc -> ija', C1, x)

to machine precision. This is the exact ket-antisymmetrizer machinery the
(never-materialized, 10-index) ADC(5) 3h2p x 3h2p self-coupling will need;
the coupling's own dense form (n2h1p x n3h2p) is the profiled ~16 GB Ne/aVDZ
memory blocker, so applying it matrix-free is directly production-relevant.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri)

COUPLINGS = [('IP', 'c1_2h1p_3h2p', 'apply_c1_2h1p_3h2p'),
             ('EA', 'c1_2p1h_3p2h', 'apply_c1_2p1h_3p2h')]


from src.CodeGeneration.pdaggerq_interface import pdaggerq_available


def _generate():
    """(dense, appl): adc4_k2_blocks and adc4_k2_apply_blocks, generated into a temporary directory
    (generated output is not part of the repository) and loaded from there."""
    import importlib.util
    import tempfile
    from src.CodeGeneration import codegen_common, generate_adcn, sigma_codegen
    out = tempfile.mkdtemp(prefix='sigma_codegen_')
    saved = codegen_common.ADC_OUT_DIR
    codegen_common.ADC_OUT_DIR = out
    try:
        generate_adcn.generate_adc4_base_k2()
        sigma_codegen.generate_adc4_base_k2_apply()
    finally:
        codegen_common.ADC_OUT_DIR = saved
    mods = []
    for name in ('adc4_k2_blocks', 'adc4_k2_apply_blocks'):
        spec = importlib.util.spec_from_file_location(name, os.path.join(out, 'adc4', name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods.append(mod)
    return tuple(mods)


def main():
    dense, appl = _generate()
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
        pool = {'f': np.diag(eps), 'g': g, 'kd': np.eye(norb),
                'o': slice(0, nocc), 'v': slice(nocc, norb), 'no': no, 'nv': nv}
        rng = np.random.default_rng(0)

        print(f"\n=== {label}  (no={no}, nv={nv}) ===")
        for sector, dname, aname in COUPLINGS:
            dfn = getattr(dense, dname)
            afn = getattr(appl, aname)
            C1 = dfn(**{p: pool[p] for p in inspect.signature(dfn).parameters})  # (bra3, ket5)
            nbra = 3
            x = rng.standard_normal(C1.shape[nbra:])                              # 3h2p ket
            ref = np.tensordot(C1, x, axes=(list(range(nbra, C1.ndim)),
                                            list(range(x.ndim))))
            akw = {p: pool[p] for p in inspect.signature(afn).parameters if p != 'x'}
            y = afn(x=x, **akw)
            diff = np.abs(y - ref).max()
            ok = diff < 1e-11
            all_ok &= ok
            print(f"  [{'OK' if ok else 'FAIL'}] {aname}  (ket=3h2p, {C1.ndim}-index dense): "
                  f" max|apply - dense@x| = {diff:.2e}")

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
