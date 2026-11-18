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
Cholesky solve. `terms` assembles an active space's DownfoldingTerms from it:
vtilde, the correlation-energy pair E_c[v] - E_c^A[v_eff] (both by the
plasmon formula, on the same v) and the static defect
|W(0) - W^A[v_eff](0)|, zero for crpa.

`embedding_self_energy` gives the static environment self-energy
Sigma_bar^E on the casida route (spec §3.6): the residues of the full W
between the active and the environment orbitals (`screened_residues`) and
the mean-field energies, reduced by Marie and Loos's SRG (Eq. 44, 'srg') or
by mode A at eta = 0 (Eq. 12, 'mode_a', refused near the poles,
`pole_window`). `terms(..., sigma=...)` hands it to the general layer in
both spin channels with the Hartree-only double counting (dc='hartree'), and
records the 'sigma_poles' diagnostic.
"""
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.Base.numerics.constants import (DOWNFOLDING_SRG_DAMPING_WARN,
                                         DOWNFOLDING_SRG_S)
from src.Base.numerics.numerics import numerics_or_default
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.SingleReference.common.active_space import closed_shell_nocc
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver

from ..partition import active_pair_mask, ph_element_mask
from ..terms import DownfoldingTerms, choice

METHOD = 'rpa'
MODES = ('casida', 'imagfrequency', 'space-time')
BUILT_MODES = ('casida',)
SCREENINGS = ('bare', 'crpa', 'crpa_ph', 'mrpa')
SIGMAS = (None, 'srg', 'mode_a')
DIAGNOSTICS = ('static_defect', 'ec_active_casida', 'sigma_poles',
               'sigma_continuation')


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

    E_RPA,corr comes from the same solve as Omega and X+Y.
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
    e_c, omega, xpy = _plasmon(eps, nocc, factors, eri, numerics)
    return RPALowLevel(
        mf=mf, mode=mode, df=df, numerics=numerics, eps=eps, nocc=nocc,
        e_c_full=e_c, omega=omega, xpy=xpy, factors=factors, eri=eri)


def _plasmon(eps, nocc, factors, eri, numerics):
    """(E_c, omega, X+Y) of the restricted singlet direct RPA on the
    integrals `factors` (three-index) or `eri` (chemist), by the plasmon
    formula E_c = 1/2 (sum omega - Tr A). A and B are freed by the solver,
    so Tr A is taken before it. The full RPA and the active one go through
    here, so the E_c pair is the same formula on the same solver."""
    lr = LinearResponseSolver(eps, coeff_df=factors, eri_chemist=eri,
                              spin_mode='restricted')
    A, B = lr.build_casida_matrices(nocc, lBSE=False)
    trace_a = np.trace(A)
    solver = CasidaSolver(A, B)
    del A, B
    omega, xpy, y = solver.solve(numerics=numerics)
    xpy += y
    del y
    return float(0.5 * (np.sum(omega) - trace_a)), omega, xpy


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


def _active_v(low, act):
    """The v of the full RPA (low.factors or low.eri) on the orbitals `act`:
    (v_act, v_ov), v_act the (n,)*4 chemist (pq|rs) and v_ov the
    (n*n, n_pair) (pq|ia) over every (i, a) pair in C order."""
    n, nocc = len(act), low.nocc
    if low.factors is not None:
        b = low.factors
        b_act = b[:, act][:, :, act].reshape(b.shape[0], n * n)
        b_ov = b[:, :nocc, nocc:].reshape(b.shape[0], -1)
        return (b_act.T @ b_act).reshape((n,) * 4), b_act.T @ b_ov
    eri = low.eri[np.ix_(act, act)]
    n_pair = nocc * (low.eri.shape[0] - nocc)  # spelt out: n may be 0
    return (eri[:, :, act][:, :, :, act],
            eri[:, :, :nocc, nocc:].reshape(n * n, n_pair))


def _active_ph(low, part, v_eff):
    """(delta, K, v_ph) of the active ph pairs (k active occupied times the
    active virtual, local indices, C order): delta the orbital energy
    differences, K_ia,jb = v_eff[i,a,j,b] and v_ph (n*n, n_ph) the columns
    v_eff[:, :, i, a]."""
    n, k = part.n_act, part.n_act_occ
    eps = low.eps[part.act]
    delta = (eps[k:][None, :] - eps[:k][:, None]).ravel()
    v_ph = v_eff[:, :, :k, k:].reshape(n * n, delta.size)
    return delta, v_eff[:k, k:, :k, k:].reshape(delta.size, delta.size), v_ph


def active_correlation_energy(low, part, v_eff):
    """E_c^A[v_eff], the direct RPA correlation energy of the active space
    alone: the restricted singlet dRPA on the n_A_o n_A_v active ph pairs
    with K_ia,jb = v_eff[i,a,j,b] (local indices, occupied first) and the
    orbital energies of the reference, by the plasmon formula of the full
    RPA. v_eff is (n_act,)*4 chemist; for the E_c pair to cancel when every
    orbital is active it has to be built on the same v as low.e_c_full
    (spec §3.2). Zero when there is no active ph pair."""
    k = part.n_act_occ
    if k == 0 or k == part.n_act:
        return 0.0
    return _plasmon(low.eps[part.act], k, None, v_eff, low.numerics)[0]


def static_defect(low, part, v_eff):
    """max |W(0) - W^A[v_eff](0)| over the active (pq|rs): how far the
    active-space RPA of v_eff falls short of the full static screening.

    W(0) - v = -4 V xpy diag(1/omega) xpy^T V^T, V = (pq|ia) on the v of the
    full RPA; W^A[v_eff](0) = v_eff - 4 v_eff[:, ph] (delta_A + 4 K^A)^-1
    v_eff[ph, :] on the active ph pairs, K^A the ph block of v_eff. v_eff has
    to be built on that same v (spec §3.2): for crpa, W^A[W^r](0) = W(0)
    exactly and the defect is zero to rounding. delta_A + 4 K^A is positive
    definite for every built screening (for crpa it is a Schur complement of
    delta + 4K), so one Cholesky solve gives the active response.
    Zero for an empty active space.
    """
    n = part.n_act
    if n == 0:
        return 0.0
    v_act, v_ov = _active_v(low, part.act)
    r = v_ov @ low.xpy
    w_full = v_act - 4.0 * ((r / low.omega) @ r.T).reshape((n,) * 4)
    w_act = v_eff.copy()
    delta, k_ph, v_ph = _active_ph(low, part, v_eff)
    if delta.size:
        apb = 4.0 * k_ph
        apb[np.diag_indices_from(apb)] += delta
        w_act -= (4.0 * v_ph @ cho_solve(cho_factor(apb), v_ph.T)).reshape(
            (n,) * 4)
    return float(np.abs(w_full - w_act).max())


def screened_residues(low, part, orbitals=None):
    """w (nexc, n_act, n_r): the residues of the full W between the active
    orbitals (part.act, occupied first) and `orbitals` (default part.env,
    the inactive then the environment virtual orbitals), one slice per RPA
    excitation:

        w[n, t, r] = sqrt(2) sum_ia (t r|i a) xpy_ia,n.

    The sqrt(2) makes w the spin-orbital residue for either spin of t and r
    (a restricted singlet mode puts xpy/sqrt(2) on both spins of each pair),
    so sum_r w_tr w_ur over spatial r is the spin-orbital sum over the r of
    t's spin and the self-energy formulas need no further spin factor; W - v
    = sum_n w w^T 2 Omega_n/(omega^2 - Omega_n^2), which at omega = 0 is the
    -4/Omega_n of `static_defect` on the residues without sqrt(2). With r
    over every orbital, `srg_self_energy` of these w is the active block of
    the qsGW SRG kernel (`SelfEnergySolver.static_self_energy_matrix`,
    2 sum chi chi K with chi = w/sqrt(2)); tests/test_downfolding_sigma.py
    pins the factor there. (tr|ia) is built from the integrals of the full
    RPA, low.factors or low.eri.
    """
    act = part.act
    orbs = part.env if orbitals is None else np.asarray(orbitals, dtype=int)
    n, m, nocc = len(act), len(orbs), low.nocc
    if low.factors is not None:
        b = low.factors
        rho = b[:, :nocc, nocc:].reshape(b.shape[0], -1) @ low.xpy
        w = rho.T @ b[:, act][:, :, orbs].reshape(b.shape[0], n * m)
    else:
        v = low.eri[np.ix_(act, orbs)][:, :, :nocc, nocc:]
        w = low.xpy.T @ v.reshape(n * m, low.xpy.shape[0]).T
    return np.sqrt(2.0) * w.reshape(len(low.omega), n, m)


def _denominators(eps_t, eps_r, occupied_r, omega):
    """D[n, t, r] = eps_t - eps_r - sgn(eps_r - eps_F) Omega_n: + Omega_n for
    an occupied r, - Omega_n for an empty one."""
    sign = np.where(np.asarray(occupied_r, dtype=bool), 1.0, -1.0)
    return (np.asarray(eps_t)[None, :, None] - np.asarray(eps_r)[None, None, :]
            + sign[None, None, :] * np.asarray(omega)[:, None, None])


def srg_self_energy(w, eps_t, eps_r, occupied_r, omega, s):
    """The SRG static self-energy (n_act, n_act), Marie and Loos Eq. 44:

        Sigma_tu = sum_rn (D_tr + D_ur)/(D_tr^2 + D_ur^2) w_tr w_ur
                   [1 - exp(-(D_tr^2 + D_ur^2) s)],

    D_tr = eps_t - eps_r - sgn(eps_r - eps_F) Omega_n (`_denominators`), w
    the residues of `screened_residues` (nexc, n_act, n_r), occupied_r a bool
    array over the internal orbitals r. s = None is s = inf, Eq. 42, the
    bracket 1. Every term is finite for finite s (the kernel is bounded by
    (1 + sqrt 2)/(2 max(|a|, |b|)) and vanishes with a^2 + b^2), so this
    needs no pole check; it is symmetric in t, u by construction. A term
    exactly at a pole (D_tr = D_ur = 0, 0/0 as written) is 0, its limit at
    finite s; Eq. 42 has no value there and gets the same 0.
    """
    d = _denominators(eps_t, eps_r, occupied_r, omega)
    a, b = d[:, :, None, :], d[:, None, :, :]
    lam = a * a + b * b
    kern = np.divide(a + b, lam, out=np.zeros_like(lam), where=lam > 0)
    if s is not None:
        kern *= -np.expm1(-s * lam)
    out = np.einsum('ntr,nur,ntur->tu', w, w, kern)
    return 0.5 * (out + out.T)


def mode_a_self_energy(w, eps_t, eps_r, occupied_r, omega):
    """Mode A (n_act, n_act), Marie and Loos Eq. 12 at eta = 0:

        Sigma_tu = 1/2 sum_rn [1/D_tr + 1/D_ur] w_tr w_ur
                 = 1/2 [S_tu(eps_t) + S_tu(eps_u)],

    S the pole sum of iG(W - v) over the internal orbitals r, arguments as
    in `srg_self_energy`. Undamped: a D_tr near zero makes it diverge, which
    `embedding_self_energy` refuses before calling it.
    """
    d = _denominators(eps_t, eps_r, occupied_r, omega)
    half = np.einsum('ntr,nur,ntr->tu', w, w, 1.0 / d)
    return 0.5 * (half + half.T)


def pole_window(low, part):
    """(lo, hi), the pole-free window of the environment self-energy:
    (max eps_I - Omega_min, min eps_Ev + Omega_min). Every pole of S^E lies
    at eps_i - Omega_n <= lo (inactive i) or eps_a + Omega_n >= hi
    (environment virtual a), so between them every D_tr is non-zero. A side
    without orbitals (or no excitation at all) is -inf or +inf."""
    if low.omega.size == 0:
        return -np.inf, np.inf
    om = low.omega.min()
    eps = low.eps
    lo = eps[part.inactive].max() - om if len(part.inactive) else -np.inf
    hi = eps[part.env_virt].min() + om if len(part.env_virt) else np.inf
    return float(lo), float(hi)


def _refuse_near_poles(low, part, settings):
    """Refuse (ValueError) an active orbital whose energy is within
    pole_margin/sqrt(2 s) of an edge of `pole_window`, or outside it (spec
    §3.6, sigma='mode_a'); s = DOWNFOLDING_SRG_S when srg_s is None."""
    s = DOWNFOLDING_SRG_S if settings.srg_s is None else settings.srg_s
    margin = settings.pole_margin / np.sqrt(2.0 * s)
    lo, hi = pole_window(low, part)
    eps = low.eps[part.act]
    dist = np.minimum(eps - lo, hi - eps)
    bad = dist <= margin
    if bad.any():
        named = ', '.join(f'orbital {p} (eps = {e:.6f}, distance {x:.6f})'
                          for p, e, x in zip(part.act[bad], eps[bad],
                                             dist[bad]))
        raise ValueError(
            f"sigma='mode_a' diverges at the poles of the environment "
            f"self-energy: the active {named} are within the margin "
            f"{margin:.6f} Ha (pole_margin {settings.pole_margin:g} / "
            f"sqrt(2 s), s = {s:g}) of the pole-free window "
            f"({lo:.6f}, {hi:.6f}) or outside it (a negative distance). "
            f"Use sigma='srg', which is finite everywhere, or move these "
            f"orbitals out of the active space.")


def embedding_self_energy(low, part, sigma):
    """Sigma_bar^E (n_act, n_act), the static environment self-energy on
    part.act (occupied first): iG^E(W - v) with r over part.env, the full
    RPA's W (`screened_residues`) and the mean-field energies eps and
    Omega (spec §3.6).

    sigma: 'srg', Eq. 44 at s = numerics.downfolding.srg_s (None: Eq. 42),
        finite everywhere; 'mode_a', Eq. 12 at eta = 0, refused (naming the
        orbitals) when an active energy is within pole_margin/sqrt(2 s) of
        the pole-free window's edge or outside it (`pole_window`).
    Zero when the environment is empty.
    """
    return _embedding(low, part, sigma)[0]


def _embedding(low, part, sigma):
    """(Sigma_bar, args, srg_s) of `embedding_self_energy`: args = (w, eps_t,
    eps_r, occupied_r, omega) of the environment, kept so that the
    'sigma_poles' diagnostic does not build the residues again, and srg_s the
    s of the run's numerics."""
    sigma = choice(sigma, SIGMAS[1:], 'sigma')
    settings = numerics_or_default(low.numerics).downfolding
    if sigma == 'mode_a':
        _refuse_near_poles(low, part, settings)
    env = part.env
    args = (screened_residues(low, part), low.eps[part.act], low.eps[env],
            env < low.nocc, low.omega)
    if sigma == 'srg':
        return srg_self_energy(*args, settings.srg_s), args, settings.srg_s
    return mode_a_self_energy(*args), args, settings.srg_s


class SigmaDampingWarning(RuntimeWarning):
    """The SRG damping of Sigma_bar^E is not negligible: an active orbital
    energy lies within a few 1/sqrt(2 s) of a pole of the environment
    self-energy, so 'srg' there depends on srg_s."""


def _pole_weights(w, eps_t, eps_r, occupied_r, omega):
    """Z_t = 1/(1 - S'_tt(eps_t)) of each active orbital t, S the pole sum
    of `mode_a_self_energy`, S_tt(x) = sum_rn w_tr^2 / (x - eps_r - sgn
    Omega_n), so S'_tt(eps_t) = -sum_rn w_tr^2 / D_tr^2 and Z_t = 1/(1 +
    sum_rn w_tr^2 / D_tr^2), in (0, 1]: 1 far from every pole, towards 0
    near one. A term exactly at a pole (D_tr = 0) makes Z_t 0."""
    d = _denominators(eps_t, eps_r, occupied_r, omega)
    w2, d2 = w * w, d * d
    ratio = np.divide(w2, d2, out=np.where(w2 > 0, np.inf, 0.0),
                      where=d2 > 0)
    return 1.0 / (1.0 + ratio.sum(axis=(0, 2)))


def _sigma_poles(low, part, sigma, sigma_bar, args, srg_s):
    """The 'sigma_poles' record (spec §4.6), and a SigmaDampingWarning naming
    the active orbitals whose row of Sigma_bar the SRG damping moves by more
    than DOWNFOLDING_SRG_DAMPING_WARN.

    window: `pole_window`. pole_distance: per active orbital, the distance
        of eps_t to the nearer edge of the window, negative outside it.
    z: per active orbital, Z_t = 1/(1 - S'_tt(eps_t)) (`_pole_weights`).
    srg_s: the s applied, None for s = inf and for 'mode_a' (undamped).
    srg_damping: max |Sigma_bar_srg(s) - Sigma_bar_srg(inf)|; 0.0 when no
        finite s is applied.
    """
    lo, hi = pole_window(low, part)
    eps = low.eps[part.act]
    dist = np.minimum(eps - lo, hi - eps)
    applied = srg_s if sigma == 'srg' else None
    damping = 0.0
    if applied is not None and sigma_bar.size:
        moved = np.abs(sigma_bar - srg_self_energy(*args, None))
        damping = float(moved.max())
        rows = moved.max(axis=1) > DOWNFOLDING_SRG_DAMPING_WARN
        if rows.any():
            named = ', '.join(f'orbital {p} (eps = {e:.6f}, distance to the '
                              f'window {x:.6f})' for p, e, x in
                              zip(part.act[rows], eps[rows], dist[rows]))
            warnings.warn(
                f"sigma='srg': the SRG damping (s = {applied:g}) changes "
                f"Sigma_bar^E by up to {damping:.2e} Ha on the active {named}"
                f", within a few 1/sqrt(2 s) = {1 / np.sqrt(2 * applied):.4f}"
                f" Ha of a pole of the environment self-energy (pole-free "
                f"window ({lo:.6f}, {hi:.6f})); Sigma_bar depends on srg_s "
                f"there.", SigmaDampingWarning, stacklevel=4)
    return {'window': (lo, hi), 'pole_distance': dist,
            'z': _pole_weights(*args), 'srg_s': applied,
            'srg_damping': damping}


def _diagnostic_names(diagnostics):
    """The DIAGNOSTICS entries `diagnostics` names (a name or an iterable of
    names), each refused unless it is one."""
    if isinstance(diagnostics, str):
        diagnostics = (diagnostics,)
    return {choice(d, DIAGNOSTICS, 'diagnostics') for d in diagnostics}


def terms(low, part, screening='crpa', sigma=None, diagnostics=DIAGNOSTICS,
          **opts):
    """The DownfoldingTerms of the RPA on the active space `part`.

    screening: 'bare' | 'crpa' | 'crpa_ph' ('mrpa' not built yet); vtilde
        is its static correction (`screening_correction`), the same in every
        spin block (restricted).
    sigma: None, the paper's scheme: full double counting, no Sigma_bar.
        'srg' or 'mode_a': the static environment self-energy
        (`embedding_self_energy`) in both spin channels with dc='hartree',
        the Hartree-only double counting that goes with it; E_0 then gains
        the exchange energy term and -Tr[Sigma_bar gamma^A,0] (spec §3.6,
        §3.7). Allowed with every screening; the third term of the
        embedding self-energy, iG^A(W - W^A[v_eff]), is dropped, which is
        exact only for 'crpa' in the static limit.
    diagnostics: the DIAGNOSTICS to record in energies.diagnostics; a name
        that does not apply here (on casida 'ec_active_casida' and
        'sigma_continuation', and 'sigma_poles' without sigma) is skipped and
        absent, an unknown name refused. 'sigma_poles' (`_sigma_poles`)
        warns (SigmaDampingWarning) when the SRG damping is not negligible.

    E_c[v] is the full RPA's; E_c^A[v_eff] is the dRPA of the active ph
    pairs with v_eff = v + vtilde, v the same integrals as the full RPA
    (low.factors or low.eri, spec §3.2), not the exact v of H_eff. 'bare'
    keeps E_c[v] - E_c^A[v], the mechanical embedding.
    """
    _refuse_options(opts, 'terms')
    names = _diagnostic_names(diagnostics)
    screening = choice(screening, SCREENINGS, 'screening')
    if sigma not in SIGMAS:
        raise ValueError(f"Unknown sigma={sigma!r}; choose one of "
                         + ', '.join(repr(s) for s in SIGMAS) + '.')
    vt = screening_correction(low, part, screening)
    v_eff = _active_v(low, part.act)[0] + vt
    found = {}
    if 'static_defect' in names:
        found['static_defect'] = static_defect(low, part, v_eff)
    sigma_bar, dc = None, 'full'
    if sigma is not None:
        s_bar, args, srg_s = _embedding(low, part, sigma)
        sigma_bar, dc = (s_bar, s_bar), 'hartree'
        if 'sigma_poles' in names:
            found['sigma_poles'] = _sigma_poles(low, part, sigma, s_bar,
                                                args, srg_s)
    return DownfoldingTerms(
        vtilde=(vt, vt, vt), sigma_bar=sigma_bar, e_c_full=low.e_c_full,
        e_c_active=active_correlation_energy(low, part, v_eff), dc=dc,
        diagnostics=found)
