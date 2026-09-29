"""Closed-shell DF EE-ADC operator with <ja|bc> stored once as [a, j, b, c]
against the same operator on anti4's six spin blocks of it (the sb_einsum
route), at ADC(2), ADC(2)-x and ADC(3). Random orbital energies and DF factor,
so no SCF: the check is the algebra, to rounding."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.SingleReference.ADC.eeADC import ee_r_sigma_df as df  # noqa: E402
from src.SingleReference.ADC.eeADC.ee_spin_blocks import anti4  # noqa: E402


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def main():
    """Run every level; 0 when all pass, else 1."""
    rng = np.random.default_rng(7)
    no, nv, naux = 4, 9, 30
    n = no + nv
    eps = np.concatenate([np.sort(rng.uniform(-2.0, -0.5, no)),
                          np.sort(rng.uniform(0.5, 3.0, nv))])
    B = 0.1 * rng.standard_normal((naux, n, n))
    B = B + B.transpose(0, 2, 1)
    all_ok = True
    for level in ('adc2', 'adc2x', 'adc3'):
        cache = {}
        aop, diag, _ = df.build_operator(eps, B, no, level=level, cache=cache)
        gb_ref = df.g_blocks_df(B, no, n)
        gb_ref['ovvv'] = anti4(gb_ref.pop('ovvv_ajbc').transpose(1, 0, 2, 3))
        ref_cache = {'gb': gb_ref, 'vk': df.DFVvvvKernels(B, no, n)}
        aop_ref, diag_ref, _ = df.build_operator(eps, B, no, level=level,
                                                 cache=ref_cache)
        vecs = rng.standard_normal((3, len(diag)))
        got = np.array([aop(v) for v in vecs])
        ref = np.array([aop_ref(v) for v in vecs])
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        all_ok &= check(rel < 1e-12, f'{level}: sigma, [a,j,b,c] layout vs spin blocks',
                        f'rel max |diff| {rel:.1e}, tolerance 1e-12')
        rel_d = np.abs(diag - diag_ref).max() / np.abs(diag_ref).max()
        all_ok &= check(rel_d < 1e-12, f'{level}: diagonal',
                        f'rel max |diff| {rel_d:.1e}, tolerance 1e-12')
        built = 'ovvv' in cache['gb']
        all_ok &= check(built == (level == 'adc3'),
                        f'{level}: six ovvv spin blocks built only for adc3',
                        f'built = {built}')
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
