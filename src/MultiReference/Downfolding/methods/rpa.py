"""The direct RPA as the low-level method of the downfolding.

`run` solves the RPA of the whole system once, before any active space is
known, and keeps what every later active space needs: on the casida route
the excitation energies Omega, the amplitudes X+Y, the full correlation
energy E_RPA,corr = 1/2 (sum Omega - Tr A), and the integrals the Casida
matrices were built from. The reference is restricted closed shell and the
response the singlet one (A = delta + 2K, B = 2K, K_ia,jb = (ia|jb)); triplet
modes contribute nothing to the direct RPA.

`screening_correction` gives the static (omega = 0) screening correction of an
active space on the casida route: the active pairs deleted from A+B and one
Cholesky solve. Its assembly into an active space's terms (`terms`) is not
there yet.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.SingleReference.common.active_space import closed_shell_nocc
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver

from ..partition import active_pair_mask, ph_element_mask
from ..terms import choice

METHOD = 'rpa'
MODES = ('casida', 'imagfrequency', 'space-time')
BUILT_MODES = ('casida',)
SCREENINGS = ('bare', 'crpa', 'crpa_ph', 'mrpa')
SIGMAS = (None, 'srg', 'mode_a')


@dataclass(eq=False)
class RPALowLevel:
    """The RPA low-level result: the mean field, the full RPA, and a cache.

    eps, nocc: the orbital energies and the occupied count of `mf`.
    e_c_full: E_RPA,corr of the whole system, 1/2 (sum omega - Tr A).
    omega: (n_pair,) the RPA excitation energies, ascending.
    xpy: (n_pair, n_pair) X+Y, one excitation per column; the rows are the
        n_pair = nocc*nvirt (i, a) pairs in C order, i over all occupied and
        a over all virtual orbitals, the order of `active_pair_mask`.
        Normalized as the Casida solver leaves it (X^T X - Y^T Y = 1), which
        makes, with factor 1,
            xpy diag(1/omega) xpy^T = (A+B)^-1 = (delta + 4K)^-1,
        so the static density response is chi(0)_ia,jb = -4 [(A+B)^-1]_ia,jb
        = -4 sum_n xpy_ia,n xpy_jb,n / omega_n (tests/test_downfolding_two_body.py
        pins the factor).
    factors: the three-index object B[Q, p, q] of the MOs, (naux, nmo, nmo),
        with (pq|rs) = sum_Q B_Q,pq B_Q,rs, when df; else None.
    eri: the exact chemist MO integrals (pq|rs) when not df; else None.
    cache: what a later `terms` call builds the first time it needs it.
    """
    mf: object
    mode: str
    df: bool
    numerics: object
    eps: np.ndarray
    nocc: int
    e_c_full: float
    omega: np.ndarray
    xpy: np.ndarray
    factors: Optional[np.ndarray]
    eri: Optional[np.ndarray]
    cache: dict = field(default_factory=dict)
    method: str = METHOD


def _refuse_options(opts, which):
    if opts:
        raise TypeError(f"method={METHOD!r} {which} does not take "
                        + ', '.join(sorted(opts)) + '.')


def run(mf, mode='casida', df=True, numerics=None, **opts):
    """The RPALowLevel of `mf`: the full direct RPA of `mode`, solved once.

    mode: 'casida' | 'imagfrequency' | 'space-time'; only 'casida' is built.
    df: casida only. True builds the Casida matrices from the mean field's
        density fit (for a mean field without one, the helper's exact
        eigendecomposition of the MO integrals, eigenvalues below 1e-12
        dropped); False from the exact 4-centre integrals.
    numerics: the run's Numerics (None: the defaults), kept on the result
        and read by the eigensolve.

    E_RPA,corr comes from the same solve as Omega and X+Y; A and B are freed
    by the solver, so Tr A is taken before it.
    """
    _refuse_options(opts, 'run')
    mode = choice(mode, MODES, 'mode')
    if mode not in BUILT_MODES:
        raise NotImplementedError(
            f"method={METHOD!r} mode={mode!r} is not implemented yet; built: "
            + ', '.join(repr(m) for m in BUILT_MODES) + '.')
    mol = mf.mol
    eps = np.asarray(get_orbital_energies(mf))
    nocc = closed_shell_nocc(mf.mo_occ)
    df = bool(df)
    if df:
        factors = get_density_fitting_coefficients(mol, mf, 'spatial')
        eri = None
    else:
        factors = None
        eri = get_two_electron_integrals_chemist(mol, mf, 'spatial')
    lr = LinearResponseSolver(eps, coeff_df=factors, eri_chemist=eri,
                              spin_mode='restricted')
    A, B = lr.build_casida_matrices(nocc, lBSE=False)
    trace_a = np.trace(A)
    solver = CasidaSolver(A, B)
    del A, B
    omega, xpy, y = solver.solve(numerics=numerics)
    xpy += y
    del y
    return RPALowLevel(
        mf=mf, mode=mode, df=df, numerics=numerics, eps=eps, nocc=nocc,
        e_c_full=float(0.5 * (np.sum(omega) - trace_a)), omega=omega,
        xpy=xpy, factors=factors, eri=eri)


def screening_correction(low, part, screening):
    """The static screening correction vt = W^r(0) - v on the active orbitals.

    Returns (n_act,)*4, chemist order (tu|vw), on part.act (occupied first).
    screening: 'bare' gives zeros; 'crpa' the restricted cRPA correction
        vt_tu,vw = -4 sum_{ia,jb in r} (tu|ia) [(delta + 4K)_rr^-1]_ia,jb (jb|vw),
    r the (i, a) pairs that are not both active (spec §3.3, §3.4): the
    response of the active pairs is deleted from A+B = delta + 4K, which is
    the paper's zeroing of A+B and A-B on the active block with its Omega = 0
    modes dropped, and the static response chi^r(0) = -4 [(A+B)_rr]^-1 of
    the spin-summed restricted singlet is one Cholesky solve on the deleted
    matrix. 'crpa_ph' keeps that correction on the particle-hole elements
    alone (`ph_element_mask`). 'mrpa' is not built yet.

    (A+B)_rr and (tu|ia) are built here from the same integrals as the full
    RPA, low.factors (K = B_ov^T B_ov) or low.eri. With every orbital active
    no pair is left to screen and the correction is zero, without a solve;
    with no active occupied or no active virtual orbital nothing is deleted
    and it is the fully screened W(0) - v on those orbitals.
    """
    screening = choice(screening, SCREENINGS, 'screening')
    if screening == 'mrpa':
        raise NotImplementedError(
            f"method={METHOD!r} screening='mrpa' is not implemented yet; "
            f"built: 'bare', 'crpa', 'crpa_ph'.")
    n = part.n_act
    vt = np.zeros((n,) * 4)
    res = ~active_pair_mask(part)
    if screening == 'bare' or n == 0 or not res.any():
        return vt
    nocc, act = low.nocc, part.act
    if low.factors is not None:
        b = low.factors
        b_res = b[:, :nocc, nocc:].reshape(b.shape[0], -1)[:, res]
        k_rr = b_res.T @ b_res
        v_rt = b_res.T @ b[:, act][:, :, act].reshape(b.shape[0], n * n)
    else:
        eri = low.eri
        ov = eri[:nocc, nocc:].reshape(res.size, *eri.shape[2:])[res]
        k_rr = ov[:, :nocc, nocc:].reshape(len(ov), -1)[:, res]
        v_rt = ov[:, act][:, :, act].reshape(len(ov), n * n)
    eps = low.eps
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()[res]
    apb = 4.0 * k_rr
    apb[np.diag_indices_from(apb)] += delta
    vt = (-4.0 * v_rt.T @ cho_solve(cho_factor(apb), v_rt)).reshape((n,) * 4)
    if screening == 'crpa_ph':
        vt *= ph_element_mask(part)
    return vt


def terms(low, part, **opts):
    """Not built yet: the RPA's DownfoldingTerms of an active space."""
    raise NotImplementedError(
        f"method={METHOD!r}: the screened interaction of an active space is "
        f"not implemented yet (mode={low.mode!r}); run_low_level gives the "
        f"full RPA only.")
