"""The Brueckner reference (ADC/cc_amplitudes.brueckner_reference) and the
static self-energy from a correlated density
(ADC/static_correction.build_density_static_correction_restricted).

1. On canonical HF orbitals the density builder, fed the CCSD Lambda density,
   equals build_static_correction(kind='ccsd'); and the DF hook of kind='ccsd'
   equals the dense contraction with the integrals rebuilt from the same DF
   factor (so B_aa enters exactly, not as an approximation of something else).
2. Brueckner: CCSD on the returned orbitals has T1 = 0 (the Brueckner
   condition, checked by an independent CCSD), the orbitals are orthonormal,
   the occupied and virtual blocks of their Fock matrix are diagonal with
   mo_energy on the diagonal, and the input mean field is left untouched.
3. Brueckner static block: mo_energy + correction must be the one-body
   operator h + G[rho_BCCD] in the Brueckner orbitals, i.e. the reference's
   occupied-virtual Fock block is carried; dropping it is caught.

Run: python tests/test_brueckner_reference.py, or under pytest.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import cc, gto, scf

from src.Base.pyscf_interface import DFIntegrals
from src.SingleReference.ADC import (build_density_static_correction_restricted,
                                     build_static_correction)
from src.SingleReference.ADC.cc_amplitudes import brueckner_reference
from src.SingleReference.ADC.static_correction import (
    _dgamma_spatial_from_ao_density, _static_correction_from_dgamma_restricted)
from src.SingleReference.CC.pipeline import compute_ccsd_density_matrix


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f"   ({detail})" if detail else ''))
    return bool(ok)


def water():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf


def check_density_static(mol, mf):
    ok = True
    nocc = mol.nelectron // 2
    B = DFIntegrals.from_scf(mol, mf).B_aa
    ref = build_static_correction(mf, mol, kind='ccsd', B_aa=B, spin='restricted')
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    got = build_density_static_correction_restricted(mycc.make_rdm1(), nocc, B)
    d = np.abs(got - ref).max()
    ok &= check(d < 1e-6, "density builder == kind='ccsd' on HF orbitals", f'{d:.1e}')
    eri = np.einsum('Qpq,Qrs->pqrs', B, B)
    dgamma = _dgamma_spatial_from_ao_density(mf, compute_ccsd_density_matrix(mf))
    d = np.abs(_static_correction_from_dgamma_restricted(eri, dgamma) - ref).max()
    ok &= check(d < 1e-6, "kind='ccsd' DF hook == dense contraction of B^T B", f'{d:.1e}')
    return ok


def check_brueckner(mol, mf):
    ok = True
    nocc = mol.nelectron // 2
    C0 = mf.mo_coeff.copy()
    mfb, info = brueckner_reference(mf, conv_tol=1e-10, conv_tol_normt=1e-8)
    ok &= check(np.array_equal(mf.mo_coeff, C0), 'input mean field untouched')
    ok &= check(info['t1_norm_hf'] > 1e-3 and info['t1_norm'] < 1e-6,
                '|T1| vanishes on the Brueckner orbitals',
                f"{info['t1_norm_hf']:.1e} -> {info['t1_norm']:.1e}")
    mycc = cc.CCSD(mfb)
    mycc.conv_tol, mycc.conv_tol_normt = 1e-10, 1e-8
    mycc.kernel()
    t1 = np.linalg.norm(mycc.t1)
    ok &= check(t1 < 1e-5, 'independent CCSD on them: T1 = 0', f'{t1:.1e}')
    C, S = mfb.mo_coeff, mol.intor('int1e_ovlp')
    d = np.abs(C.T @ S @ C - np.eye(C.shape[1])).max()
    ok &= check(d < 1e-10, 'Brueckner orbitals orthonormal', f'{d:.1e}')
    F = info['fock']
    off = max(np.abs(F[:nocc, :nocc] - np.diag(np.diag(F[:nocc, :nocc]))).max(),
              np.abs(F[nocc:, nocc:] - np.diag(np.diag(F[nocc:, nocc:]))).max())
    ok &= check(off < 1e-10 and np.allclose(np.diag(F), mfb.mo_energy),
                'occupied and virtual Fock blocks diagonal, mo_energy on the diagonal',
                f'{off:.1e}')
    fov = np.abs(F[:nocc, nocc:]).max()
    ok &= check(fov > 1e-4, 'occupied-virtual Fock block of the Brueckner determinant is not zero',
                f'{fov:.1e}')
    B = DFIntegrals.from_scf(mol, mfb).B_aa
    one = C.T @ mf.get_fock(dm=C @ info['dm1'] @ C.T) @ C
    full = np.diag(mfb.mo_energy) + build_density_static_correction_restricted(
        info['dm1'], nocc, B, fock=F)
    d = np.abs(full - one).max()
    ok &= check(d < 1e-8, 'mo_energy + correction == h + G[rho_BCCD]', f'{d:.1e}')
    bare = np.diag(mfb.mo_energy) + build_density_static_correction_restricted(
        info['dm1'], nocc, B)
    d = np.abs(bare - one).max()
    ok &= check(d > 1e-4, 'without the Fock block the identity fails (the check can fail)',
                f'{d:.1e}')
    return ok


def run():
    mol, mf = water()
    print('-- static self-energy from a correlated density (H2O/cc-pVDZ)')
    all_ok = check_density_static(mol, mf)
    print('-- Brueckner reference (H2O/cc-pVDZ)')
    all_ok &= check_brueckner(mol, mf)
    print('\nALL PASSED' if all_ok else '\nFAILURES')
    return all_ok


def test_brueckner_reference():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
