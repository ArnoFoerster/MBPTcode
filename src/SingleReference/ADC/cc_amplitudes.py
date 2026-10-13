"""Coupled-cluster doubles amplitudes for the Faddeev-ADC(3) pair channels.

pair_route='ccd' of the Faddeev-ADC(3) replaces every channel amplitude
(TDHF ring, ppRPA ladder) by the one converged CCD T2, as CCD-ADC(3) replaces
ADC(3)'s first-order doubles. The bra side takes the adjoint of the same
amplitude, never the CC Lambda, so the secular matrix stays symmetric.

Convention (pyscf's): t2[i,j,a,b] = t_(i alpha, j beta)^(a alpha, b beta),
so that the first-order amplitude is (ia|jb) / (eps_i + eps_j - eps_a - eps_b).

brueckner_reference supplies the orbitals in which the singles vanish
(Brueckner CCD): the Faddeev-ADC(3) can then be built on them with the BCCD
amplitude and a static self-energy from the BCCD density
(static_correction.build_density_static_correction_restricted).
"""
import copy
import functools

import numpy as np
from pyscf import cc as _pyscf_cc
from pyscf import scf as _pyscf_scf
from pyscf.cc import bccd as _pyscf_bccd


@functools.lru_cache(maxsize=None)
def _ccd_class(base):
    """CCD on top of `base`, the CC class pyscf dispatches to for this mean
    field. pyscf.cc.ccd.CCD hard-codes the conventional update_amps and so
    drops the density fitting of a DF mean field; subclassing the dispatched
    class keeps CCD on the integrals CCSD gets. Zero T1 after every update,
    start from T1 = 0."""
    class _CCD(base):
        def update_amps(self, t1, t2, eris):
            t1, t2 = super().update_amps(t1, t2, eris)
            return np.zeros_like(t1), t2

        def kernel(self, t2=None, eris=None):
            t1 = np.zeros((self.nocc, self.nmo - self.nocc))
            self.ccsd(t1, t2, eris)
            return self.e_corr, self.t2

    _CCD.__name__ = f'CCD_{base.__name__}'
    return _CCD


def ccd_t2_restricted(mf, return_energy=False, **cc_opts):
    """t2[i,j,a,b], the alpha-beta amplitude of a converged all-electron CCD
    on an RHF mean field (DF-CCD for a density-fitted mf); cc_opts are set on
    the pyscf CC object."""
    if isinstance(mf, _pyscf_scf.uhf.UHF):
        raise NotImplementedError("ccd_t2_restricted is RHF-only")
    mycc = _ccd_class(type(_pyscf_cc.CCSD(mf)))(mf)
    for key, val in cc_opts.items():
        setattr(mycc, key, val)
    mycc.run()
    if not mycc.converged:
        raise RuntimeError("CCD did not converge")
    t2 = np.asarray(mycc.t2)
    return (t2, mycc.e_corr) if return_energy else t2


def brueckner_reference(mf, conv_tol_normu=1e-6, max_cycle=50, **cc_opts):
    """Brueckner CCD reference of an RHF mean field: the determinant e^{T1}|HF>
    (occupied orbitals phi_i + sum_a t_i^a phi_a), with the orbitals rotated by
    exp(T1 - T1^dagger) and CCSD re-solved until |T1| < conv_tol_normu
    (pyscf.cc.bccd), then semicanonicalized (occupied and virtual blocks of the
    Fock matrix of the Brueckner determinant diagonal; its occupied-virtual
    block stays, of second order).

    Returns a copy of mf carrying the Brueckner orbitals (mo_coeff) and the
    diagonal of their Fock matrix (mo_energy), and a dict with
    't2' (alpha-beta t2[i,j,a,b], the BCCD amplitude), 'dm1' (the CC Lambda
    one-particle density in that orbital basis, spin-summed), 'fock' (the
    Fock matrix of the Brueckner determinant in that basis), 'e_corr',
    't1_norm' and 't1_norm_hf' (|T1| of the CCSD on mf). cc_opts are set on the
    first CCSD only (pyscf.cc.bccd re-creates the CC object per macro
    iteration). mf itself is not modified."""
    if isinstance(mf, _pyscf_scf.uhf.UHF):
        raise NotImplementedError("brueckner_reference is RHF-only")
    mfb = copy.copy(mf)
    mfb.mo_coeff = np.array(mf.mo_coeff, copy=True)
    mycc = _pyscf_cc.CCSD(mfb)
    for key, val in cc_opts.items():
        setattr(mycc, key, val)
    mycc.kernel()
    t1_hf = float(np.linalg.norm(mycc.t1))
    mycc = _pyscf_bccd.bccd_kernel_(mycc, conv_tol_normu=conv_tol_normu,
                                    max_cycle=max_cycle, verbose=0)
    t1_norm = float(np.linalg.norm(mycc.t1))
    # the semicanonicalization re-creates the CC object (converged is reset),
    # so the Brueckner condition itself is the test
    if t1_norm >= conv_tol_normu:
        raise RuntimeError(f"BCCD did not converge (|T1| = {t1_norm:.2e})")
    C = mfb.mo_coeff
    dm_ao = mfb.make_rdm1(C, mfb.mo_occ)
    fock = C.T @ mfb.get_fock(dm=dm_ao) @ C
    mfb.mo_energy = np.diag(fock).copy()
    mfb.e_tot = mfb.energy_tot(dm=dm_ao)
    info = {'t2': np.asarray(mycc.t2), 'dm1': np.asarray(mycc.make_rdm1()),
            'fock': fock, 'e_corr': float(mycc.e_corr), 't1_norm': t1_norm,
            't1_norm_hf': t1_hf}
    return mfb, info
