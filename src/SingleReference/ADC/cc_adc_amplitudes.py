"""Coupled-cluster ground-state amplitudes in the ADC amplitude-pool
conventions: the CC-ADC(n) amplitude substitution.

ADC(3)'s order-2 coupling blocks U^(2)/S^(2) are built from exactly two
ground-state amplitudes: the first-order doubles T2^(1) and the second-order
singles T1^(2). EN-ADC(3) renormalizes those two by dressing their energy
denominators; this module is the coupled-cluster analogue, replacing them
outright with the converged CCSD (or CCD) T1/T2.

Hermiticity. The generated blocks take the ket amplitude (`t*` tags) and the
bra amplitude (`l*` tags) as separate arguments. Coupled-cluster theory would
put the Lambda amplitudes in the bra slots, which is what makes EOM-CC
non-Hermitian. ADC's secular matrix is Hermitian, so this module sets the bra
amplitude to the adjoint of the ket amplitude (`_to_l(t)`, the same map the MP
pools use), never to Lambda. The supermatrix stays symmetric, as in
ADC(3)/EN-ADC(3): CC amplitudes inside the ADC/ISR representation, not EOM-CC.

Conventions (pinned against pyscf in tests/test_cc_adc_amplitudes.py):
    spin-orbital   t2[a,b,i,j] = t_ij^ab   (antisymmetrized), t1[a,i] = t_i^a
    restricted     t2_abab[a,b,i,j] = t_(i alpha, j beta)^(a alpha, b beta)
                   t2_aaaa = t2_abab - t2_abab.transpose(1, 0, 2, 3)
i.e. mpn_density_driver{,_restricted}'s own [vir..., occ...] storage.

cc_amplitudes.py (the Faddeev-ADC(3) CCD channel amplitude and the Brueckner
reference) is the other CC module of this package; both run their CC
through its _run_cc.
"""
import numpy as np
from pyscf.cc import addons as _cc_addons

from src.Base.pyscf_interface import (get_antisymmetrized_spin_block_eri,
                                      get_orbital_energies)
from src.SingleReference.ADC.cc_amplitudes import _run_cc
from src.SingleReference.DensityMatrix.mpn_density_driver import _to_l
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import (
    MPnDensityDriverRestricted, _to_l_restricted)


def cc_amplitudes_spinorbital(mf, method='ccsd', frozen=None,
                              return_energy=False, **cc_opts):
    """{'t1', 't2', 'l1', 'l2'} for the spin-orbital amplitude pool, from a
    converged RHF-based CCSD/CCD. `l*` are the adjoints of `t*` (see the
    module docstring on Hermiticity), not the CC Lambda amplitudes.
    return_energy=True gives (pool, e_corr), the CC correlation energy."""
    mycc = _run_cc(mf, method, frozen, **cc_opts)
    t1 = _cc_addons.spatial2spin(mycc.t1).T                  # (nv, no)
    t2 = _cc_addons.spatial2spin(mycc.t2).transpose(2, 3, 0, 1)   # (nv,nv,no,no)
    pool = {'t1': t1, 'l1': _to_l(t1, 1),
            't2': t2, 'l2': _to_l(t2, 2)}
    return (pool, mycc.e_corr) if return_energy else pool


def _restricted_pool(t1_aa, t2_aaaa, t2_abab):
    """The restricted amplitude pool keyed by spin block (closed shell: the
    _bbbb/_bb blocks alias the _aaaa/_aa ones; bra = ket adjoint)."""
    return {'t1_aa': t1_aa, 't1_bb': t1_aa,
            'l1_aa': _to_l_restricted(t1_aa, 1, 'aa'),
            'l1_bb': _to_l_restricted(t1_aa, 1, 'aa'),
            't2_aaaa': t2_aaaa, 't2_bbbb': t2_aaaa, 't2_abab': t2_abab,
            'l2_aaaa': _to_l_restricted(t2_aaaa, 2, 'aaaa'),
            'l2_bbbb': _to_l_restricted(t2_aaaa, 2, 'aaaa'),
            'l2_abab': _to_l_restricted(t2_abab, 2, 'abab')}


def cc_amplitudes_restricted(mf, method='ccsd', frozen=None,
                             return_energy=False, **cc_opts):
    """Spin-block twin of cc_amplitudes_spinorbital, keyed for the restricted
    amplitude pool."""
    mycc = _run_cc(mf, method, frozen, **cc_opts)
    t1_aa = np.asarray(mycc.t1).T                            # (nv, no)
    t2_abab = np.asarray(mycc.t2).transpose(2, 3, 0, 1)      # (nv,nv,no,no)
    t2_aaaa = t2_abab - t2_abab.transpose(1, 0, 2, 3)
    pool = _restricted_pool(t1_aa, t2_aaaa, t2_abab)
    return (pool, mycc.e_corr) if return_energy else pool


def mbpt_amplitudes_restricted(mf, nocc=None):
    """The MBPT pool the substitution replaces (T2^(1)/T1^(2) and their
    adjoints), in the same keying as cc_amplitudes_restricted, so a caller
    can mix the two dicts key by key and attribute a CC-ADC(3) shift to the
    singles or the doubles channel separately. Dense spin-block integrals
    (norb^4); mbpt_amplitudes_restricted_df is the g-free twin."""
    nocc = mf.mol.nelectron // 2 if nocc is None else nocc
    eps = get_orbital_energies(mf, representation='spatial')
    g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mf.mol, mf)
    drv = MPnDensityDriverRestricted(np.diag(eps), g_aaaa, g_abab, g_bbbb, nocc)
    t2_aaaa, t2_abab = drv.compute_t2_1()
    t1_2_aa = drv.compute_order2_amplitudes(t2_aaaa, t2_abab, max_rank=1)[0]
    return _restricted_pool(t1_2_aa, t2_aaaa, t2_abab)


def mbpt_amplitudes_restricted_df(mf, B_aa, nocc=None):
    """g-free twin of mbpt_amplitudes_restricted: the MBPT pool (T2^(1),
    T1^(2) and their adjoints) built from the DF factor instead of the dense
    spin-block ERI, so a mixed pool (CC doubles + MP singles, the like-for-like
    partner of EN-ADC(3) doubles-only) is constructible at production basis
    (the dense twin's norb^4 array is 15 GB for CO and 115 GB for SiH4 at
    aug-cc-pVQZ). Same keys and the same bra convention as its dense twin."""
    nocc = mf.mol.nelectron // 2 if nocc is None else nocc
    eps = get_orbital_energies(mf, representation='spatial')
    drv = MPnDensityDriverRestricted(np.diag(eps), None, None, None, nocc,
                                     B_aa=B_aa, B_bb=B_aa)
    t2_aaaa, t2_abab = drv.compute_t2_1_df()
    # compute_t1_2_aa_df, not compute_order2_amplitudes_df(max_rank=1): that
    # function's max_rank gates the triples only, so it would also build
    # t2_2_aaaa and t2_2_abab, both discarded here
    t1_2_aa = drv.compute_t1_2_aa_df(t2_aaaa, t2_abab)
    return _restricted_pool(t1_2_aa, t2_aaaa, t2_abab)


class InjectedAmplitudePool(dict):
    """Amplitude pool with foreign (non-MBPT) amplitudes in it. Missing keys
    raise a named error instead of a KeyError deep inside a generated split:
    a tag that is absent here (t3/l3, the second-order triples ADC(4)+
    consumes) has no counterpart in a CCSD wavefunction, so the substitution
    is undefined at that level."""

    def __init__(self, mapping, source):
        super().__init__(mapping)
        self.source = source

    def __missing__(self, key):
        raise NotImplementedError(
            f"amplitude tag {key!r} is not defined for the {self.source} "
            f"amplitude substitution (available: {sorted(self)}). The "
            "substitution covers the order-2 U/S blocks of ADC(3) only; "
            "ADC(4)+ additionally consumes T3^(2), which a CCSD/CCD "
            "wavefunction has no counterpart for.")
