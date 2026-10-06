"""Coupled-cluster doubles amplitudes for the Faddeev-ADC(3) pair channels.

pair_route='ccd' of the Faddeev-ADC(3) replaces every channel amplitude
(TDHF ring, ppRPA ladder) by the one converged CCD T2, as CCD-ADC(3) replaces
ADC(3)'s first-order doubles. The bra side takes the adjoint of the same
amplitude, never the CC Lambda, so the secular matrix stays symmetric.

Convention (pyscf's): t2[i,j,a,b] = t_(i alpha, j beta)^(a alpha, b beta),
so that the first-order amplitude is (ia|jb) / (eps_i + eps_j - eps_a - eps_b).
"""
import functools

import numpy as np
from pyscf import cc as _pyscf_cc
from pyscf import scf as _pyscf_scf


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
