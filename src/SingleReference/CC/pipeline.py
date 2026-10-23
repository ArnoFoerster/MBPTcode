"""End-to-end coupled-cluster 1-RDMs.

CCSDT: integrals -> T-CCSDT -> Lambda-CCSDT -> 1-RDM (generated
equations). CCSD: thin wrapper around pyscf's CCSD/make_rdm1. Both density
functions return the full AO-basis 1-RDM, usable as
calc_qp_energy(..., dm_correction=...).
"""
import warnings

import numpy as np
from pyscf import cc, scf

from . import amplitudes
from . import solver
from .integrals import (build_spinorbital_integrals,
                        build_spinorbital_integrals_from_mf,
                        energy_denominators)
from .fixed_point import CCNotConverged


def solve_ccsdt_1rdm_from_ints(ints, t_stopping_eps=1e-9, l_stopping_eps=1e-8,
                               max_iter=300, verbose=True, out=None):
    """Core solve: given the integral dict from build_spinorbital_integrals*,
    converge T-CCSDT, Lambda-CCSDT, and assemble the spin-orbital 1-RDM.

    out: optional dict; receives 't_converged' and 'lambda_converged' (each
    solve warns when it stops unconverged and returns its last iterate)."""
    fock, g = ints['fock'], ints['g']
    nocc, nvir = ints['nocc'], ints['nvir']
    o, v = slice(None, nocc), slice(nocc, None)
    e_ai, e_abij, e_abcijk = energy_denominators(fock, nocc, nvir)

    t1 = np.zeros((nvir, nocc))
    t2 = np.zeros((nvir, nvir, nocc, nocc))
    t3 = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))
    t_out, l_out = {}, {}
    t1, t2, t3 = amplitudes.kernel(t1, t2, t3, fock, g, o, v, e_ai, e_abij, e_abcijk,
                                   ints['hf_energy'], max_iter=max_iter,
                                   stopping_eps=t_stopping_eps, out=t_out)

    l1, l2, l3 = solver.solve_lambda_ccsdt(t1, t2, t3, fock, g, o, v,
                                           e_ai, e_abij, e_abcijk,
                                           max_iter=max_iter,
                                           stopping_eps=l_stopping_eps,
                                           verbose=verbose, out=l_out)
    if out is not None:
        out['t_converged'] = t_out['converged']
        out['lambda_converged'] = l_out['converged']

    return solver.ccsdt_one_rdm(t1, t2, t3, l1, l2, l3, o, v)


def compute_ccsdt_1rdm(geometry, basis, charge=0, **kwargs):
    """Returns (opdm, ints) where opdm is the spin-orbital CCSDT 1-RDM
    (shape (nmo,nmo), nmo = nocc+nvir spin orbitals) and ints is the dict
    from build_spinorbital_integrals (mo_coeff/mo_energy included, for an
    AO backtransform). Original densityMatrix-repo entry point."""
    ints = build_spinorbital_integrals(geometry, basis, charge=charge)
    opdm = solve_ccsdt_1rdm_from_ints(ints, **kwargs)
    return opdm, ints


def spin_sum_opdm(opdm):
    """Spin-orbital 1-RDM (interleaved 2p=alpha, 2p+1=beta) -> spin-summed
    spatial 1-RDM, for a closed-shell (spin-restricted) reference."""
    return opdm[0::2, 0::2] + opdm[1::2, 1::2]


def compute_ccsdt_density_matrix(mf, symmetrize=True, **kwargs):
    """CCSDT AO-basis 1-RDM from an already-converged, closed-shell
    spin-restricted pyscf mean-field object.

    The CC response density is intrinsically slightly non-Hermitian
    (~1e-6-1e-7); symmetrize=True (default) returns (D + D.T)/2, matching
    the Hermitian densities everything downstream expects.

    Cost warning: full spin-orbital CCSDT with no frozen core -- fine for
    few-atom molecules in small bases, O(n^8)-ish beyond that (FH/cc-pVDZ
    already takes ~10+ minutes for the T equations alone).

    UHF/UKS references are refused: the spin-orbital integrals of a UHF
    reference are block-stacked by spin, while the spin sum below assumes the
    interleaved RHF order and a single mo_coeff.
    kwargs go to solve_ccsdt_1rdm_from_ints (including out= for the
    convergence flags).
    """
    if isinstance(mf, scf.uhf.UHF) or np.ndim(mf.mo_coeff) != 2:
        raise ValueError('compute_ccsdt_density_matrix needs a closed-shell '
                         'spin-restricted (RHF/RKS) mean field; this CCSDT density '
                         'has no UHF path')
    ints = build_spinorbital_integrals_from_mf(mf)
    opdm = solve_ccsdt_1rdm_from_ints(ints, **kwargs)
    dm_ao = ints['mo_coeff'] @ spin_sum_opdm(opdm) @ ints['mo_coeff'].T
    if symmetrize:
        dm_ao = 0.5 * (dm_ao + dm_ao.T)
    return dm_ao


def ccsdt_ao_density(geometry, basis, charge=0, **kwargs):
    """Same as compute_ccsdt_1rdm, but returns the spin-summed density
    matrix in the AO basis (mo_coeff @ D_spatial @ mo_coeff.T), ready to
    compare against a pyscf AO density matrix. Returns (dm_ao, ints)."""
    opdm, ints = compute_ccsdt_1rdm(geometry, basis, charge=charge, **kwargs)
    dm_spatial = spin_sum_opdm(opdm)
    mo_coeff = ints['mo_coeff']
    dm_ao = mo_coeff @ dm_spatial @ mo_coeff.T
    return dm_ao, ints


def _flag_ccsd_convergence(mycc, who, out):
    """pyscf CC that stopped unconverged: warn and flag, keep
    its last amplitudes."""
    if out is not None:
        out['converged'] = bool(mycc.converged)
    if not mycc.converged:
        warnings.warn(f'{who}: pyscf {type(mycc).__name__} did not converge; its '
                      f'last amplitudes are used (converged=False)',
                      CCNotConverged, stacklevel=3)


def compute_ccsd_density_matrix(mf, ncore=0, out=None):
    """CCSD AO-basis 1-RDM via pyscf (the same construction the test sweeps
    used inline: mo_coeff @ mycc.make_rdm1() @ mo_coeff.T). Needs an RHF
    reference -- pass a converged RHF mf, not a KS object.

    ncore: number of frozen spatial core orbitals (e.g. 2 for C+O 1s).
    Passed as frozen=ncore to pyscf's CCSD.
    out: optional dict; receives 'converged' (an unconverged CCSD warns)."""
    mycc = cc.CCSD(mf)
    if ncore > 0:
        mycc.frozen = ncore
    mycc.kernel()
    _flag_ccsd_convergence(mycc, 'compute_ccsd_density_matrix', out)
    return mf.mo_coeff @ mycc.make_rdm1() @ mf.mo_coeff.T


def compute_ccsd_density_matrix_uhf(mf, ncore=0, out=None):
    """UCCSD AO-basis 1-RDM per spin channel via pyscf, the UHF counterpart of
    compute_ccsd_density_matrix. Needs a converged UHF reference.

    ncore: number of frozen spatial core orbitals, passed as frozen=ncore to
    pyscf's UCCSD. Returns (dm_a_ao, dm_b_ao).
    out: optional dict; receives 'converged' (an unconverged UCCSD warns)."""
    mycc = cc.UCCSD(mf)
    if ncore > 0:
        mycc.frozen = ncore
    mycc.kernel()
    _flag_ccsd_convergence(mycc, 'compute_ccsd_density_matrix_uhf', out)
    dm1a, dm1b = mycc.make_rdm1()
    mo_a, mo_b = mf.mo_coeff
    return mo_a @ dm1a @ mo_a.T, mo_b @ dm1b @ mo_b.T
