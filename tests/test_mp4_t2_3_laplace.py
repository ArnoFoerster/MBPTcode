"""GATE for laplace_codegen.fuse_amplitude_rank3_tags -- the SPLIT-fusion
builder for a Laplace AMPLITUDE (the order-4 amplitude-chain tool, and the MP5
static-self-energy enabler).

Target: t2_3 (order-3 doubles), whose numerator consumes the rank>=3 amplitudes
t3_2 (O(no^3 nv^3)) and t4_2 (O(no^4 nv^4)). The split fuses each of those
CONTRIBUTIONS (minimax quadrature, never materializing t3_2/t4_2) and takes the
rank<=2 DIRECT part from the compiled numerator with t3_2/t4_2 zeroed -- so the
reconstructed numerator equals the exact (fully materialized) one to the ntau
quadrature accuracy, WITHOUT ever forming a rank>=3 array.

WHY t2_3 IS THE RIGHT ARBITER (and NH3 the right system): t2_3's numerator has
MIXED-ORDER direct leaves (a direct t2_2 at order 2 AND, from a V.V.t2_1 term, a
direct t2_1 at order 1), which is exactly what breaks the simpler
fuse_nested_numerator (uniform order tag). NH3/sto-3g (5 occ / 3 vir) is the
smallest system where t3_2 AND t4_2 are genuinely nonzero, so the split really
replaces nonzero materialized tensors (LiH/H2O zero t3_aaaaaa -> vacuous).

Run:  ~/anaconda3/envs/py310/bin/python -u tests/test_mp4_t2_3_laplace.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.Base.pdaggerq_interface import pdaggerq_available

pytestmark = pytest.mark.skipif(not pdaggerq_available(),
                                reason='needs pdaggerq (optional)')

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_antisymmetrized_spin_block_eri)
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import (
    MPnDensityDriverRestricted)
from src.SingleReference.DensityMatrix.generated_mpn_restricted import (
    mpn_density_pieces_restricted as gen)
from src.CodeGeneration.generate_mpn_density_restricted import (
    t_numerator_terms_restricted)
from src.CodeGeneration.generate_mpn_density import amp_out_indices
from src.CodeGeneration.laplace_codegen import (
    fuse_amplitude_rank3_tags, assemble_laplace_fused)
from src.Base.utils.grids import minimax_time_grid
from src.SingleReference.CC.cached_einsum import einsum

NTAU = 6
TOL = 1e-7          # ntau=6 minimax accuracy on NH3 (converges with ntau)


def _shape_expr():
    return ('(t2_1_aaaa.shape[0], t2_1_aaaa.shape[0], '
            't2_1_aaaa.shape[2], t2_1_aaaa.shape[2])')


def main():
    mol = gto.M(atom='N 0 0 0; H 0 0 1.01; H 0.95 0 -0.3; H -0.48 0.82 -0.3',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.conv_tol = 1e-13; mf.run()
    eps_a = get_orbital_energies(mf, representation='spatial')
    ga, gb, gab = get_antisymmetrized_spin_block_eri(mol, mf)
    no = mol.nelectron // 2
    d = MPnDensityDriverRestricted(np.diag(eps_a), ga, gab, gb, no)
    t2_1a, t2_1b = d.compute_t2_1()
    o2 = d.compute_order2_amplitudes(t2_1a, t2_1b, max_rank=4)
    t1_2a, t2_2a, t2_2b = o2[0], o2[1], o2[2]
    t3_2 = o2[3:6]
    t4_2 = o2[6:10]
    args = d._args()
    print(f"NH3/sto-3g norb={d.no + d.nv} nocc={d.no} nvir={d.nv}")
    mx3 = max(float(np.abs(x).max()) for x in t3_2)
    mx4 = max(float(np.abs(x).max()) for x in t4_2)
    ok3 = mx3 > 1e-8 and mx4 > 1e-8
    print(f"[{'OK' if ok3 else 'FAIL'}] t3_2 (|max|={mx3:.2e}) AND t4_2 "
          f"(|max|={mx4:.2e}) genuinely nonzero -- the split replaces real tensors")

    pool = {'g_aaaa': ga, 'g_abab': gab, 'g_bbbb': gb, 'o': d.o, 'v': d.v,
            'eps_a': eps_a, 't2_1_aaaa': t2_1a, 't2_1_abab': t2_1b}
    all_ok = ok3

    for blk in ('aaaa', 'abab'):
        numfn = getattr(gen, f't2_3_{blk}_numerator')
        kw = dict(t1_2_aa=t1_2a, t2_2_aaaa=t2_2a, t2_2_abab=t2_2b)
        exact = numfn(**args, **kw,
                      t3_2_aaaaaa=t3_2[0], t3_2_aabaab=t3_2[1], t3_2_abbabb=t3_2[2],
                      t4_2_aaaaaaaa=t4_2[0], t4_2_aaabaaab=t4_2[1],
                      t4_2_aabbaabb=t4_2[2], t4_2_abbbabbb=t4_2[3])
        z = lambda t: [np.zeros_like(x) for x in t]
        z3, z4 = z(t3_2), z(t4_2)
        approx = numfn(**args, **kw,
                       t3_2_aaaaaa=z3[0], t3_2_aabaab=z3[1], t3_2_abbabb=z3[2],
                       t4_2_aaaaaaaa=z4[0], t4_2_aaabaaab=z4[1],
                       t4_2_aabbaabb=z4[2], t4_2_abbbabbb=z4[3]).copy()

        tag_terms = fuse_amplitude_rank3_tags(t_numerator_terms_restricted,
                                              2, 3, blk)
        for tag, (rk, fused) in sorted(tag_terms.items()):
            src = assemble_laplace_fused(
                'f', ['g_aaaa', 'g_abab', 'g_bbbb', 'o', 'v', 'eps_a',
                      't2_1_aaaa', 't2_1_abab'],
                {'t2_aaaa': 't2_1_aaaa', 't2_abab': 't2_1_abab',
                 't2_bbbb': 't2_1_aaaa'}, 'out', _shape_expr(), fused,
                amp_out_indices(2), eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]')
            ns = {'np': np, 'minimax_time_grid': minimax_time_grid, 'einsum': einsum}
            exec(compile(src, '<f>', 'exec'), ns)
            fn = ns['f']
            approx = approx + fn(
                **{p: pool[p] for p in inspect.signature(fn).parameters
                   if p != 'ntau'}, ntau=NTAU)

        err = float(np.abs(approx - exact).max())
        mag = float(np.abs(exact).max())
        ok = err < TOL
        all_ok &= ok
        print(f"[{'OK' if ok else 'FAIL'}] t2_3_{blk} numerator: split-fused == "
              f"exact ({err:.2e} < {TOL:.0e}, |exact|={mag:.2e})")

    print('\nALL OK' if all_ok else '\nFAILURES')
    return 0 if all_ok else 1


def test_main():
    assert not main(), 'main() returned a nonzero exit code'


if __name__ == '__main__':
    if not pdaggerq_available():
        print('SKIPPED: needs pdaggerq (optional)')
        sys.exit(0)
    sys.exit(main())
