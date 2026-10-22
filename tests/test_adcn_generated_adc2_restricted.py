"""Increment 1 gate of the spin-free ADC(2)-(5) plan
every spin-blocked
ADC(2) block function (generate_adcn.generate_adc2_restricted ->
generatedADC/adc2_blocks_restricted.py) must reproduce its spin-orbital
counterpart (adc2_blocks.py) EXACTLY at the corresponding interleaved
spin-orbital indices.

For a block over free indices with a fixed spin pattern (the function's
`_aaaa`/`_abab`/... suffix), the spatial-tensor output element at spatial
(p, q, ...) equals the spin-orbital block element at (2p+s_p, 2q+s_q, ...)
where s = 0 for an alpha index, 1 for beta (alpha=even interleaving, the
same convention spin_adapt.py locks in). This is a direct, per-element
equivalence -- no config-space assembly involved -- so it isolates the
spin-blocked codegen itself before the restricted driver (increment 2) is
built on top.

H2O/sto-3g is the decisive system (H2's degeneracies mask real errors);
H2/sto-3g kept as a smoke
case. RHF -> alpha==beta spin blocks, so the spin-orbital block is the exact
oracle.
"""
import inspect
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri, get_antisymmetrized_spin_block_eri)
from tests.generated_runtime import requires_runtime

pytestmark = requires_runtime(4)
from src.SingleReference.ADC.generated_modules import lazy
gen = lazy('adc2_blocks')
genr = lazy('adc2_blocks_restricted')


def _params(fn):
    return list(inspect.signature(fn).parameters)


def _spin_offset(s):
    return 0 if s == 'a' else 1


def check(label, diff, tol=1e-11):
    ok = diff < tol
    print(f"[{'OK' if ok else 'FAIL'}] {label}: max diff = {diff:.3e}")
    return ok


def main():
    all_ok = True

    for atom, basis, label in [
        ('H 0 0 0; H 0 0 0.74', 'sto-3g', 'H2/sto-3g (smoke)'),
        ('O 0 0 0; H 0 0 0.96; H 0.96 0 0', 'sto-3g', 'H2O/sto-3g (decisive)'),
    ]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        nocc_sp = mol.nelectron // 2
        eps_sp = get_orbital_energies(mf, representation='spatial')
        norb_sp = len(eps_sp)
        nvir_sp = norb_sp - nocc_sp

        # spin-orbital oracle args (ADCSolver convention: antisymmetrized <pq||rs>)
        nocc = 2 * nocc_sp
        eps = get_orbital_energies(mf, representation='spin')
        norb = len(eps)
        eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
        g = get_antisymmetrized_spin_eri(eri)
        so_args = dict(f=np.diag(eps), g=g, kd=np.eye(norb),
                       o=slice(0, nocc), v=slice(nocc, norb), no=nocc, nv=norb - nocc)

        # restricted spin-block args (spatial-MO)
        g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
        r_args = dict(f_aa=np.diag(eps_sp), f_bb=np.diag(eps_sp),
                      g_aaaa=g_aaaa, g_bbbb=g_bbbb, g_abab=g_abab,
                      d_aa=np.eye(norb_sp), d_bb=np.eye(norb_sp),
                      o=slice(0, nocc_sp), v=slice(nocc_sp, norb_sp),
                      no=nocc_sp, nv=nvir_sp)

        print(f"\n=== {label} ===")
        n_checked = 0
        worst = 0.0
        for name in dir(genr):
            fn_r = getattr(genr, name)
            if name.startswith('_') or not isinstance(fn_r, types.FunctionType):
                continue
            if '_' not in name:
                continue
            base, suffix = name.rsplit('_', 1)
            fn_so = getattr(gen, base, None)
            if fn_so is None:   # not one of our blocks (e.g. an imported helper)
                continue

            T_r = fn_r(*[r_args[p] for p in _params(fn_r)])
            T_so = fn_so(*[so_args[p] for p in _params(fn_so)])

            # occ axes have spatial dim nocc_sp; vir axes have nvir_sp. Both the
            # occ and (block-local) vir spin-orbital axes are alpha=even
            # interleaved, so spatial index p at spin s -> 2p+s on either.
            idx = []
            for ax, s in enumerate(suffix):
                idx.append(2 * np.arange(T_r.shape[ax]) + _spin_offset(s))
            diff = np.abs(T_so[np.ix_(*idx)] - T_r).max()
            worst = max(worst, diff)
            n_checked += 1
            if diff >= 1e-11:
                all_ok = False
                print(f"[FAIL] {name}: {diff:.3e}")

        all_ok &= check(f"all {n_checked} spin-blocked ADC(2) blocks == spin-orbital", worst)

    print("\nALL OK" if all_ok else "\nSOME FAILED")
    return all_ok


def test_main():
    assert main()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
