"""Validation of the config-space sigma driver (src/SingleReference/ADC/
sigma_driver.py): sigma_matvec(apply_fn, ..., v_cfg) must equal the dense
config-space block matvec block_cfg @ v_cfg, for the generated matrix-free
apply blocks.

Two blocks exercised:
  1. k=1<->k=2 coupling (2h1p<->3h2p / 2p1h<->3p2h) -- a 3h2p/3p2h KET
     (dense form is the ~16 GB Ne/aVDZ memory blocker), the ADC(5)-relevant
     case. bra_k=1, ket_k=2.
  2. C^(2) class-class split vt2 (2h1p/2p1h self-block) -- bra_k=ket_k=1.

The dense config block is the shipped generated tensor gathered at the
ordered configs on both sides (same class_configs machinery the driver
scatters/gathers with). A round-trip match to machine precision confirms the
canonical-position scatter/gather is sign-correct with no permutation
multiplicity (see sigma_driver's docstring).
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC.generated_modules import lazy
k2dense = lazy('adc4_k2_blocks')
k2appl = lazy('adc4_k2_apply_blocks')
c2dense = lazy('adc4_c2_blocks')
c2appl = lazy('adc4_c2_apply_blocks')
from src.SingleReference.ADC.sigma_driver import sigma_matvec, local_configs
from src.SingleReference.DensityMatrix.mpn_density_driver import MPnDensityDriver
from src.Base.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri)
from tests.generated_runtime import requires_runtime

pytestmark = requires_runtime(4)


def dense_config_block(tensor, nocc, norb, bra_k, ket_k, sector):
    """Gather a full class-class tensor at the ordered configs (bra rows x
    ket cols) -- the dense supermatrix block the driver must reproduce."""
    bra = local_configs(nocc, norb, bra_k, sector)
    ket = local_configs(nocc, norb, ket_k, sector)
    bra_i = tuple(x[:, None] for x in bra)
    ket_i = tuple(x[None, :] for x in ket)
    return tensor[bra_i + ket_i]


def main():
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
        f, kd = np.diag(eps), np.eye(norb)
        o, v = slice(0, nocc), slice(nocc, norb)
        t2 = MPnDensityDriver(eps, g, nocc).compute_t2_1()
        rng = np.random.default_rng(0)
        print(f"\n=== {label}  (no={no}, nv={nv}) ===")

        # 1. k=1<->k=2 coupling (3h2p/3p2h ket)
        for sector, dname, aname in [('IP', 'c1_2h1p_3h2p', 'apply_c1_2h1p_3h2p'),
                                     ('EA', 'c1_2p1h_3p2h', 'apply_c1_2p1h_3p2h')]:
            T = getattr(k2dense, dname)(f=f, g=g, kd=kd, o=o, v=v, no=no, nv=nv)
            block = dense_config_block(T, nocc, norb, 1, 2, sector)
            v_cfg = rng.standard_normal(block.shape[1])
            ref = block @ v_cfg
            y = sigma_matvec(getattr(k2appl, aname), (f, g, kd, o, v, no, nv),
                             v_cfg, nocc, norb, bra_k=1, ket_k=2, sector=sector)
            diff = np.abs(y - ref).max()
            ok = diff < 1e-11
            all_ok &= ok
            print(f"  [{'OK' if ok else 'FAIL'}] {aname} (bra=k1, ket=k2/3h2p, "
                  f"block {block.shape}):  max|sigma_matvec - block@v| = {diff:.2e}")

        # 2. C^(2) vt2 self-block (k1 x k1)
        for sector, dname, aname in [('IP', 'c2_2h1p_vt2', 'apply_c2_2h1p_vt2'),
                                     ('EA', 'c2_2p1h_vt2', 'apply_c2_2p1h_vt2')]:
            T = getattr(c2dense, dname)(f=f, g=g, kd=kd, t2=t2, o=o, v=v, no=no, nv=nv)
            block = dense_config_block(T, nocc, norb, 1, 1, sector)
            v_cfg = rng.standard_normal(block.shape[1])
            ref = block @ v_cfg
            y = sigma_matvec(getattr(c2appl, aname), (f, g, kd, t2, o, v, no, nv),
                             v_cfg, nocc, norb, bra_k=1, ket_k=1, sector=sector)
            diff = np.abs(y - ref).max()
            ok = diff < 1e-11
            all_ok &= ok
            print(f"  [{'OK' if ok else 'FAIL'}] {aname} (k1 x k1, "
                  f"block {block.shape}):  max|sigma_matvec - block@v| = {diff:.2e}")

    print("\nALL OK" if all_ok else "\nSOME FAILED")
    return all_ok


def test_main():
    assert main()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
