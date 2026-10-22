"""Validates the generic Laplace-fusion code generator
(src/CodeGeneration/laplace_codegen.py) by using it to independently
re-derive the SAME two quantities mpn_density_driver_restricted.py's
hand-ported _laplace_aaaaaa_contribution computes (the t3_2_aaaaaa
contribution to t1_3_aa's numerator, and to the ov M3 block) -- if the
generic, rank-agnostic fuser reproduces that already-oracle-validated
hand derivation exactly, that's strong evidence it's correct and ready to
apply to doubles/quadruples/quintuples.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.Base.pdaggerq_interface import pdaggerq_available

pytestmark = pytest.mark.skipif(not pdaggerq_available(),
                                reason='needs pdaggerq (optional)')

import numpy as np
from pyscf import gto, scf

from src.CodeGeneration.generate_mpn_density import ranks_present
from src.CodeGeneration.generate_mpn_density_restricted import (
    t_numerator_terms_restricted, cross_density_terms_restricted, DENSITY_OUT_IDX_R,
)
from src.CodeGeneration.laplace_codegen import fuse_amplitude_into_consumer, assemble_laplace_fused
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import MPnDensityDriverRestricted, _laplace_aaaaaa_contribution
from src.Base.pyscf_interface import get_orbital_energies, get_antisymmetrized_spin_block_eri


def build_generic_t1_3_laplace():
    inner = t_numerator_terms_restricted(3, 2, 'aaaaaa')
    outer = t_numerator_terms_restricted(1, 3, 'aa')
    fused = fuse_amplitude_into_consumer(inner, 3, outer, 't3_aaaaaa')
    src = assemble_laplace_fused(
        'laplace_t1_3_aa_generic',
        ['g_aaaa', 'o', 'v', 'eps_a', 't2_1_aaaa'],
        {'t2_aaaa': 't2_1_aaaa'}, 't1_3_lap', '(nv, no)', fused, ['a', 'i'],
        eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]')
    from src.Base.utils.grids import minimax_time_grid
    from src.SingleReference.CC.cached_einsum import einsum
    ns = {'np': np, 'minimax_time_grid': minimax_time_grid, 'einsum': einsum}
    # nv/no aren't in params; patch shape expr to use local shapes instead
    src = src.replace('(nv, no)', '(t2_1_aaaa.shape[0], t2_1_aaaa.shape[2])')
    exec(compile(src, '<laplace_t1_3_generic>', 'exec'), ns)
    return ns['laplace_t1_3_aa_generic']


def build_generic_ov_laplace():
    inner = t_numerator_terms_restricted(3, 2, 'aaaaaa')
    outer = cross_density_terms_restricted('ov', [2], ranks_present(2))
    fused = fuse_amplitude_into_consumer(inner, 3, outer, 't3_aaaaaa')
    out_idx = DENSITY_OUT_IDX_R['ov']  # ['m', 'e']
    src = assemble_laplace_fused(
        'laplace_ov_generic',
        ['g_aaaa', 'o', 'v', 'eps_a', 't2_1_aaaa', 'l2_1_aaaa'],
        {'t2_aaaa': 't2_1_aaaa', 'l2_aaaa': 'l2_1_aaaa'}, 'ov_lap', '(no, nv)', fused, out_idx,
        eps_o_expr='eps_a[o]', eps_v_expr='eps_a[v]')
    src = src.replace('(no, nv)', '(t2_1_aaaa.shape[2], t2_1_aaaa.shape[0])')
    from src.Base.utils.grids import minimax_time_grid
    from src.SingleReference.CC.cached_einsum import einsum
    ns = {'np': np, 'minimax_time_grid': minimax_time_grid, 'einsum': einsum}
    exec(compile(src, '<laplace_ov_generic>', 'exec'), ns)
    return ns['laplace_ov_generic']


def main():
    all_ok = True
    laplace_t1_3_generic = build_generic_t1_3_laplace()
    laplace_ov_generic = build_generic_ov_laplace()

    for atom, basis in [('H 0 0 0; Li 0 0 1.6', 'sto-3g'),
                        ('O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', '6-31g')]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc = mol.nelectron // 2
        eps_a = get_orbital_energies(mf, representation='spatial')
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
        f_aa = np.diag(eps_a)

        d = MPnDensityDriverRestricted(f_aa, g_aaaa, g_abab, g_bbbb, nocc)
        t2_1_aaaa, t2_1_abab = d.compute_t2_1()
        from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import _to_l_restricted
        l2_1_aaaa = _to_l_restricted(t2_1_aaaa, 2, 'aaaa')

        ntau = 8
        hand_t1_3, hand_ov = _laplace_aaaaaa_contribution(eps_a[:nocc], eps_a[nocc:], g_aaaa, d.o, d.v, t2_1_aaaa, ntau)

        gen_t1_3 = laplace_t1_3_generic(g_aaaa=g_aaaa, o=d.o, v=d.v, eps_a=eps_a, t2_1_aaaa=t2_1_aaaa, ntau=ntau)
        gen_ov = laplace_ov_generic(g_aaaa=g_aaaa, o=d.o, v=d.v, eps_a=eps_a, t2_1_aaaa=t2_1_aaaa, l2_1_aaaa=l2_1_aaaa, ntau=ntau)

        # generic t1_3 output is (nv,no) (this module's own convention); hand
        # code's is (no,nv) -- transpose before comparing.
        diff_t1_3 = np.max(np.abs(gen_t1_3.T - hand_t1_3))
        diff_ov = np.max(np.abs(gen_ov - hand_ov))
        ok = diff_t1_3 < 1e-10 and diff_ov < 1e-10
        all_ok &= ok
        print(f"{basis:8s} {atom:35s}: generic fuser vs hand-port  "
              f"t1_3 diff={diff_t1_3:.2e}  ov diff={diff_ov:.2e}: {'OK' if ok else 'FAIL'}")

    print("ALL PASSED" if all_ok else "FAILURES DETECTED")
    return all_ok


def test_main():
    assert main()


if __name__ == '__main__':
    if not pdaggerq_available():
        print('SKIPPED: needs pdaggerq (optional)')
        sys.exit(0)
    sys.exit(0 if main() else 1)
