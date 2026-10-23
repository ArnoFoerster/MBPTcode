"""Gate for the Laplace-fused l-consuming U^(3)/S^(3) stitches
(adc4_u3_l3_laplace_restricted) that let the Dyson-ADC(4) downfold avoid ever
materializing the O(no^3 nv^3) t3/l3 amplitudes.

Two guarantees:

  1. RANK-GENERAL convention: laplace_codegen.lambda_canonical_and_sign(rank)
     reproduces mp_amplitudes.to_l's signed
     transpose EXACTLY for rank 2..5 -- so the same fusion machinery applies to
     l4 (the O(N^8) MP4/ADC(5) denominators) and l5, not just l3. This is the
     generalization the whole approach rests on.

  2. l3 CORRECTNESS: every generated rank-3 Laplace block reproduces its dense
     adc4_u3_blocks_restricted twin (built on a real, materialized l3) to the
     minimax-quadrature accuracy at ntau=16 -- with l3/t3 never formed.
"""
import inspect
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies, get_antisymmetrized_spin_block_eri)
from src.CodeGeneration.generate_mpn_density import amp_out_indices
from src.CodeGeneration.laplace_codegen import lambda_canonical_and_sign
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import MPnDensityDriverRestricted
from src.SingleReference.common.mp_amplitudes import to_l
from src.CodeGeneration.pdaggerq_interface import pdaggerq_available


def generate():
    """(dense, lap): adc4_u3_blocks_restricted and adc4_u3_l3_laplace_restricted,
    generated into a temporary directory (generated output is not part of the
    repository) and loaded from there."""
    import importlib.util
    import tempfile
    from src.CodeGeneration import codegen_common, generate_adcn
    from src.CodeGeneration import generate_adc_laplace_restricted
    out = tempfile.mkdtemp(prefix='l3_laplace_')
    saved = codegen_common.ADC_OUT_DIR
    codegen_common.ADC_OUT_DIR = out
    try:
        generate_adcn.generate_adc_u(3, restricted=True)
        generate_adc_laplace_restricted.generate('adc4_u3_l3_laplace_restricted')
    finally:
        codegen_common.ADC_OUT_DIR = saved
    mods = []
    for name in ('adc4_u3_blocks_restricted', 'adc4_u3_l3_laplace_restricted'):
        spec = importlib.util.spec_from_file_location(name, os.path.join(out, 'adc4', name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods.append(mod)
    return tuple(mods)


def check_rank_general_lambda_canonical():
    """lambda_canonical_and_sign(rank) == to_l for rank 2..5."""
    ok = True
    for rank in (2, 3, 4, 5):
        ext = amp_out_indices(rank)
        lcs = lambda_canonical_and_sign(rank)
        blocks = {'same': 'a' * (2 * rank),
                  'mixed': 'a' * rank + 'b' * rank}
        for kind, block in blocks.items():
            canon, sign = lcs[kind]
            axes = [ext.index(c) for c in canon]
            t = np.random.default_rng(0).standard_normal([4] * rank + [3] * rank)
            d = np.abs(to_l(t, rank, block) - sign * t.transpose(*axes)).max()
            ok &= d < 1e-14
            print(f"[{'OK' if d < 1e-14 else 'FAIL'}] rank={rank} {kind}: {d:.1e}")
    assert ok
    return ok


def check_l3_blocks_vs_dense(dense, lap, ntau=16):
    """Every *_lap block == its dense adc4_u3_blocks_restricted twin (real l3)."""
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    ns = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    norb = len(eps); nv = norb - ns; no = ns
    ga, gb, gab = get_antisymmetrized_spin_block_eri(mol, mf)
    o, v = slice(0, no), slice(no, norb)
    mpn = MPnDensityDriverRestricted(np.diag(eps), ga, gab, gb, ns)
    t2a, t2b = mpn.compute_t2_1()
    _, _, _, t3aaa, t3aab, t3abb, *_ = mpn.compute_order2_amplitudes(t2a, t2b, max_rank=3)
    rng = np.random.default_rng(1)
    f_aa = rng.standard_normal((norb, norb)); f_aa = 0.5 * (f_aa + f_aa.T)
    pool = dict(
        f_aa=f_aa, f_bb=f_aa, g_aaaa=ga, g_abab=gab, g_bbbb=gb,
        d_aa=np.eye(norb), d_bb=np.eye(norb),
        l3_aaaaaa=to_l(t3aaa, 3, 'aaaaaa'),
        l3_aabaab=to_l(t3aab, 3, 'aabaab'),
        l3_abbabb=to_l(t3abb, 3, 'abbabb'),
        l3_bbbbbb=to_l(t3aaa, 3, 'aaaaaa'),
        t2_aaaa=t2a, t2_abab=t2b, t2_bbbb=t2a, o=o, v=v, no=no, nv=nv,
        eps_a=eps, t2_1_aaaa=t2a, t2_1_abab=t2b)

    def call(fn, **extra):
        p = {**pool, **extra}
        return fn(*[p[a] for a in inspect.signature(fn).parameters if a in p])

    worst = 0.0; nfail = 0; ncheck = 0
    for name in sorted(n for n in dir(lap) if n[0] in 'us' and '_l3' in n):
        dfn = getattr(dense, name, None)
        if dfn is None:
            print(f"  MISSING dense twin for {name}"); nfail += 1; continue
        ex = call(dfn); fu = call(getattr(lap, name), ntau=ntau)
        scale = max(np.abs(ex).max(), 1e-30)
        rel = np.abs(fu - ex).max() / scale
        ncheck += 1; worst = max(worst, rel)
        if rel >= 1e-8:
            nfail += 1
            print(f"  FAIL {name}: rel={rel:.2e}")
    print(f"[{'OK' if nfail == 0 else 'FAIL'}] {ncheck} l3 Laplace blocks vs dense, worst rel={worst:.2e}")
    return nfail == 0 and ncheck >= 30


def run():
    a = check_rank_general_lambda_canonical()
    b = check_l3_blocks_vs_dense(*generate())
    print("\nALL OK" if (a and b) else "\nSOME FAILED")
    return a and b


def test_rank_general_lambda_canonical():
    assert check_rank_general_lambda_canonical()


@pytest.mark.skipif(not pdaggerq_available(), reason='needs pdaggerq (optional)')
def test_l3_blocks_vs_dense():
    assert check_l3_blocks_vs_dense(*generate())


if __name__ == '__main__':
    if not pdaggerq_available():
        print('SKIPPED: needs pdaggerq (optional)')
        sys.exit(0)
    sys.exit(0 if run() else 1)
