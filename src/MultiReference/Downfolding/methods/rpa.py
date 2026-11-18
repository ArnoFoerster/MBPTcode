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

On the imagfrequency route (density fitted, N^4) `run` keeps the existing
imaginary-axis energy, its minimax frequency grid and the static
polarizability P(0) of the whole system in the normalized fitting basis.
The screening correction is P - P^A and one Dyson step, with P^A the active
ph pairs' part of P(0) (`active_polarizability_static`), projected on the
factors of every active pair (`active_pair_factors`): the casida route's
deletion written another way, equal to rounding on the same three-index
object. E_c^A[v_eff] is the same quadrature on the same grid with the bare
response of the active ph pairs, so the E_c pair is one formula on both
sides; the dense-Casida E_c^A is kept as the 'ec_active_casida' diagnostic.

The space-time route (separable ISDF factors (X, D), imaginary time, N^3,
serial) is the same on its own machinery: `run` keeps the factors, the static
tau grid with P(0) on it, and the energy of `rpa_correlation_energy_space_time`
with its time-frequency grid. P^A(0) is built as P(0) was, with the active
columns of X on the same grid, so with every orbital active P - P^A is zero
to the bit; E_c^A takes the active ph pairs' exp(-delta tau) through the
energy's own tau -> omega transform. Sigma^E is not built there yet.

`embedding_self_energy` gives the static environment self-energy
Sigma_bar^E on the casida route (spec §3.6): the residues of the full W
between the active and the environment orbitals (`screened_residues`) and
the mean-field energies, reduced by Marie and Loos's SRG (Eq. 44, 'srg') or
by mode A at eta = 0 (Eq. 12, 'mode_a', refused near the poles,
`pole_window`). `terms(..., sigma=...)` hands it to the general layer in
both spin channels with the Hartree-only double counting (dc='hartree'), and
records the 'sigma_poles' diagnostic.

On imagfrequency there are no residues: S^E(mu + i w) comes from the
Lorentzian fit of W on the minimax W grid with the internal orbital
restricted to the environment (`sigma_env_imag_axis`, W cached on the first
request), is continued by one Pade fit per element, and each element is
evaluated at its own point (`continued_self_energy`): Eq. 42, Re S^E_tu(z_tu),
for 'srg' (s = inf; srg_s enters the pole check alone), the real axis for
'mode_a'. Both refuse near the poles and point to casida; Omega_min of the
window is the lowest root of the same Casida problem (`lowest_excitation`).
The 'sigma_continuation' diagnostic is the spread of two Pade fits on the
even and the odd nodes.
"""
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.Base.factorization.sliced_factors import unpack_factors
from src.Base.numerics.constants import (DOWNFOLDING_SIGMA_CONTINUATION_WARN,
                                         DOWNFOLDING_SRG_DAMPING_WARN,
                                         DOWNFOLDING_SRG_S,
                                         GW_DENSITY_W_FIT_HIGH,
                                         GW_DENSITY_W_FIT_LOW,
                                         GW_DENSITY_W_FIT_TIKHONOV)
from src.Base.numerics.numerics import numerics_or_default, rpa_quadrature
from src.Base.orbital_spaces import particle_hole_window
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.Base.utils.grids.analytic_continuation import (pade_eval_elementwise,
                                                        pade_fit_elementwise,
                                                        pade_nodes)
from src.Base.utils.grids.grids import (gap_scaled_w0, gauss_legendre_grid,
                                        minimax_frequency_grid)
from src.Base.utils.grids.time_frequency import (TimeFrequencyGrid,
                                                 minimax_points_for_accuracy)
from src.SingleReference.common.active_space import closed_shell_nocc
from src.SingleReference.GW.imaginary_axis import \
    fitted_self_energy_matrix_imaginary_axis
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.casida import (CasidaSolver,
                                                       solver_choice)
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.LinearResponse.imaginary_frequency import \
    frequency_factor
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import (
    rpa_correlation_energy_imaginary_axis, solve_polarizability_imaginary_axis)
from src.SingleReference.LinearResponse.space_time import (
    chi0_aux, chi0_imaginary_frequency, dielectric_log_determinant,
    dyson_screening, rpa_correlation_energy_space_time, static_screening_grid)

from ..partition import active_pair_mask, ph_element_mask
from ..terms import DownfoldingTerms, choice

METHOD = 'rpa'
MODES = ('casida', 'imagfrequency', 'space-time')
SCREENINGS = ('bare', 'crpa', 'crpa_ph', 'mrpa')
SIGMAS = (None, 'srg', 'mode_a')
DIAGNOSTICS = ('static_defect', 'ec_active_casida', 'sigma_poles',
               'sigma_continuation')


@dataclass(eq=False)
class RPALowLevel:
    """The RPA low-level result: the mean field, the full RPA, and a cache.

    eps, nocc: the orbital energies and the occupied count of `mf`.
    e_c_full: E_RPA,corr of the whole system: on casida 1/2 (sum omega -
        Tr A), on imagfrequency the minimax quadrature of
        `rpa_correlation_energy_imaginary_axis`, on space-time
        `rpa_correlation_energy_space_time`.
    omega: (n_pair,) the RPA excitation energies, ascending; None on
        imagfrequency and space-time, which do not diagonalize.
    xpy: (n_pair, n_pair) X+Y, one excitation per column; the rows are the
        n_pair = nocc*nvirt (i, a) pairs in C order, i over all occupied and
        a over all virtual orbitals, the order of `active_pair_mask`.
        Normalized as the Casida solver leaves it (X^T X - Y^T Y = 1), which
        makes, with factor 1,
            xpy diag(1/omega) xpy^T = (A+B)^-1 = (delta + 4K)^-1,
        so the static density response is chi(0)_ia,jb = -4 [(A+B)^-1]_ia,jb
        = -4 sum_n xpy_ia,n xpy_jb,n / omega_n (tests/test_downfolding_two_body.py
        pins the factor). None on imagfrequency and space-time.
    factors: the three-index object B[Q, p, q] of the MOs, (naux, nmo, nmo),
        with (pq|rs) = sum_Q B_Q,pq B_Q,rs, when df; else None, and None on
        space-time, which never builds it whole (`active_pair_factors`).
    eri: the exact chemist MO integrals (pq|rs) when not df; else None.
    cache: what a later `terms` call builds the first time it needs it. On
        imagfrequency `run` puts there 'grid', the (points, weights) of the
        energy's minimax frequency quadrature, and 'p0', P(0) = 1 - W(0)^-1
        = chi0(0) = -4 B_ov diag(1/delta) B_ov^T (naux, naux) in the
        normalized fitting basis (the convention of
        `solve_polarizability_imaginary_axis`). The first Sigma^E request
        adds 'omega_min' (`lowest_excitation`) and 'w_grid', the (points, W)
        of `_screening_grid`. On space-time `run` puts there 'factors',
        'static_grid', 'p0' and 'energy_grid' (`_space_time`).
    """
    mf: object
    mode: str
    df: bool
    numerics: object
    eps: np.ndarray
    nocc: int
    e_c_full: float
    omega: Optional[np.ndarray]
    xpy: Optional[np.ndarray]
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

    mode: 'casida' | 'imagfrequency' | 'space-time'.
    df: True builds everything from the mean field's density fit (for a
        mean field without one, the helper's exact eigendecomposition of the
        MO integrals, eigenvalues below 1e-12 dropped); on space-time from
        the separable (ISDF) factors instead. False, casida only: the exact
        4-centre integrals; imagfrequency is density fitted and space-time
        ISDF fitted by construction, and both refuse it.
    numerics: the run's Numerics (None: the defaults), kept on the result
        and read by the eigensolve (casida), the frequency count of the
        energy (imagfrequency, `rpa_quadrature`), or on space-time the ISDF
        grid (numerics.isdf, which has no default: None is refused there)
        and the tau and frequency counts of `_space_time`.

    On casida E_RPA,corr comes from the same solve as Omega and X+Y; on
    imagfrequency from `rpa_correlation_energy_imaginary_axis` on the
    minimax grid kept in low.cache['grid'], with P(0) in low.cache['p0'];
    on space-time from `rpa_correlation_energy_space_time` (`_space_time`).
    """
    _refuse_options(opts, 'run')
    mode = choice(mode, MODES, 'mode')
    df = bool(df)
    if mode != 'casida' and not df:
        how = ('density fitted' if mode == 'imagfrequency'
               else 'ISDF fitted (separable factors)')
        raise ValueError(
            f"method={METHOD!r} mode={mode!r} is {how}; df=False (the exact "
            f"4-centre screening) is mode='casida' only.")
    eps = np.asarray(get_orbital_energies(mf))
    nocc = closed_shell_nocc(mf.mo_occ)
    if mode == 'space-time':
        e_c, cache = _space_time(mf, eps, nocc, numerics)
        return RPALowLevel(
            mf=mf, mode=mode, df=df, numerics=numerics, eps=eps, nocc=nocc,
            e_c_full=e_c, omega=None, xpy=None, factors=None, eri=None,
            cache=cache)
    mol = mf.mol
    if df:
        factors = get_density_fitting_coefficients(mol, mf, 'spatial')
        eri = None
    else:
        factors = None
        eri = get_two_electron_integrals_chemist(mol, mf, 'spatial')
    if mode == 'imagfrequency':
        e_c, cache = _imaginary_axis(eps, nocc, factors, numerics)
        return RPALowLevel(
            mf=mf, mode=mode, df=df, numerics=numerics, eps=eps, nocc=nocc,
            e_c_full=e_c, omega=None, xpy=None, factors=factors, eri=None,
            cache=cache)
    e_c, omega, xpy = _plasmon(eps, nocc, factors, eri, numerics)
    return RPALowLevel(
        mf=mf, mode=mode, df=df, numerics=numerics, eps=eps, nocc=nocc,
        e_c_full=e_c, omega=omega, xpy=xpy, factors=factors, eri=eri)


def _imaginary_axis(eps, nocc, factors, numerics):
    """(E_c, cache) of the restricted direct RPA on the imaginary axis:
    E_c by `rpa_correlation_energy_imaginary_axis` (minimax grid), and the
    cache with 'grid', the points and weights that energy integrates on
    (the same `rpa_quadrature` count and particle-hole window), and 'p0',
    P(0) = 1 - W(0)^-1 in the normalized fitting basis. The active E_c^A is
    integrated on this grid so that the E_c pair is one quadrature."""
    lr = LinearResponseSolver(eps, coeff_df=factors, spin_mode='restricted')
    e_c = rpa_correlation_energy_imaginary_axis(lr, nocc, numerics)
    nfreq = rpa_quadrature(numerics_or_default(numerics), 'minimax')[1]
    e_min, e_max, _ = particle_hole_window(eps, nocc)
    grid = minimax_frequency_grid(nfreq, e_min, e_max)
    p0 = solve_polarizability_imaginary_axis(lr, nocc, [0.0])[0]
    return float(e_c), {'grid': grid, 'p0': p0}


def _space_time(mf, eps, nocc, numerics):
    """(E_c, cache) of the restricted direct RPA in imaginary time, serial,
    on the separable factors (X, D) of `separable_factors` on
    numerics.isdf's grid (refused, ValueError, when it names none):

    'factors': (X, D), X (M, nmo) the MO collocation, D (M, naux), so the
        fitted v is (pq|rs) = sum_Q B_Q,pq B_Q,rs with B_Q,pq = sum_P D_PQ
        X_Pp X_Pq.
    'static_grid': the one-frequency (omega = 0) minimax grid of a static
        screening (`static_screening_grid` at quadrature.ntau_w, as the BSE's
        W), and 'p0' P(0) = chi0(0) on it (`chi0_imaginary_frequency`), the
        chi0 `static_screening_matrix` inverts: 1 - W(0)^-1, built directly.
    'energy_grid': the TimeFrequencyGrid of E_c, the grid of the RPA
        ground-state chain: minimax tau (quadrature.ntau_rpa, else ntau_w,
        else the minimax count of the window at tau_target) transformed onto
        a Gauss-Legendre frequency axis (quadrature.nfreq_rpa, else
        RPA_ENERGY_NFREQ; `rpa_quadrature`) scaled to the gap.

    E_c = `rpa_correlation_energy_space_time` on 'energy_grid'. P^A(0) and
    E_c^A are built on these same grids and transforms, so the active part
    cancels to rounding with every orbital active."""
    num = numerics_or_default(numerics)
    if not num.isdf.grid_named():
        raise ValueError(
            f"method={METHOD!r} mode='space-time' builds separable (ISDF) "
            f"factors on the grid numerics.isdf names, and this run's names "
            f"none: pass numerics with numerics.isdf.level (e.g. 'G1') or "
            f"counts.")
    x, d = unpack_factors(separable_factors(mf, mf.mol, numerics=numerics,
                                            comm=None))[:2]
    q = num.quadrature
    static_grid = static_screening_grid(eps, nocc, ntau=q.ntau_w,
                                        tau_target=q.tau_target)
    p0 = chi0_imaginary_frequency(x, d, eps, nocc, static_grid)[0]
    ntau, nfreq = rpa_quadrature(num)
    e_min, e_max, _ = particle_hole_window(eps, nocc)
    if ntau is None:
        ntau = minimax_points_for_accuracy(e_min, e_max, q.tau_target)[0]
    nu, wt = gauss_legendre_grid(nfreq, gap_scaled_w0(eps, nocc))
    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max, nu, wt,
                                           with_sine=False,
                                           with_inverse=False)
    e_c = rpa_correlation_energy_space_time(x, d, eps, nocc, grid, comm=None)
    return float(e_c), {'factors': (x, d), 'static_grid': static_grid,
                        'p0': p0, 'energy_grid': grid}


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

    On imagfrequency the same correction is P - P^A and one Dyson step
    (spec §3.4), vt_tu,vw = sum_PQ B_P,tu [W^r - 1]_PQ B_Q,vw with W^r =
    [1 - (P(0) - P^A(0))]^-1 in the normalized fitting basis, P(0) from
    low.cache['p0'], P^A(0) `active_polarizability_static` and B the
    factors of every active pair (`active_pair_factors`). By the
    push-through identity this is the casida formula on the same factors,
    and P - P^A, not W - W^A: screening is not additive. On space-time it is
    the same Dyson step on the ISDF factors, P(0) and P^A(0) both on the
    static tau grid (low.cache['static_grid']).
    """
    screening = choice(screening, SCREENINGS, 'screening')
    if screening == 'mrpa':
        if low.mode != 'casida':
            raise NotImplementedError(
                f"method={METHOD!r} screening='mrpa' is built on the "
                f"Casida eigenvectors only: use mode='casida' (this low-level"
                f" result is mode={low.mode!r}).")
        raise NotImplementedError(
            f"method={METHOD!r} screening='mrpa' is not implemented yet; "
            f"built: 'bare', 'crpa', 'crpa_ph'.")
    n = part.n_act
    vt = np.zeros((n,) * 4)
    res = ~active_pair_mask(part)
    if screening == 'bare' or n == 0 or not res.any():
        return vt
    if low.mode != 'casida':
        b_act = active_pair_factors(low, part).reshape(-1, n * n)
        w_r = dyson_screening(
            low.cache['p0'] - active_polarizability_static(low, part),
            correlation=True)
        vt = (b_act.T @ w_r @ b_act).reshape((n,) * 4)
        if screening == 'crpa_ph':
            vt *= ph_element_mask(part)
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


def active_pair_factors(low, part):
    """B[Q, t, u] (naux, n_act, n_act): the fitting factors of every active
    pair, occupied-occupied, occupied-virtual and virtual-virtual, on
    part.act (occupied first), so (tu|vw) = sum_Q B_Q,tu B_Q,vw is the
    full RPA's fitted v on the active orbitals. vtilde needs all of them,
    not only the ov block (spec §3.4). On space-time they are built from
    the active columns of the collocation X alone, B_Q,tu = sum_P D_PQ X_Pt
    X_Pu (low.cache['factors']), the factors chi0 is built from."""
    return _pair_factors(low, part.act)


def _pair_factors(low, act):
    """`active_pair_factors` on the orbitals `act`."""
    if low.mode == 'space-time':
        x, d = low.cache['factors']
        n, x_act = len(act), x[:, act]
        pairs = (x_act[:, :, None] * x_act[:, None, :]).reshape(len(x), n * n)
        return (d.T @ pairs).reshape(d.shape[1], n, n)
    return low.factors[:, act][:, :, act]


def active_polarizability_static(low, part):
    """P^A(0) (naux, naux): the static bare response of the active ph pairs
    alone (i active occupied, a active virtual) in the normalized fitting
    basis, chi0(0) of the full P(0) restricted to those columns of B_ov,
    2 sum_ia B_ia f_ia(0) B_ia^T = -4 sum_ia B_ia B_ia^T / delta_ia, with
    the same frequency factor and spin factor as the full one, so P - P^A
    keeps every other pair exactly. Zero without an active ph pair.

    On space-time it is the full P(0)'s own build on the stored static grid
    (`chi0_imaginary_frequency`: `polarizability_projected_tau` at every
    tau point, transformed by the grid's cosine weights) with X restricted
    to the active columns and the full system's mid-gap mu, i.e. the
    transform of -2 sum_ia B_ia B_ia^T exp(-delta_ia tau) over the active ph
    pairs: with every orbital active it is P(0) itself, to the bit."""
    if low.mode == 'space-time':
        x, d = low.cache['factors']
        k = part.n_act_occ
        if k == 0 or k == part.n_act:
            return np.zeros((d.shape[1],) * 2)
        act = part.act
        mu = particle_hole_window(low.eps, low.nocc)[2]
        return chi0_imaginary_frequency(x[:, act], d, low.eps[act], k,
                                        low.cache['static_grid'], mu=mu)[0]
    nocc = low.nocc
    b = low.factors
    mask = active_pair_mask(part)
    c_a = b[:, :nocc, nocc:].reshape(b.shape[0], -1)[:, mask]
    eps = low.eps
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()[mask]
    return chi0_aux([(c_a, frequency_factor(delta, 0.0, True))], 2.0)


def _active_v(low, act):
    """The v of the full RPA (low.factors or low.eri) on the orbitals `act`:
    (v_act, v_ov), v_act the (n,)*4 chemist (pq|rs) and v_ov the
    (n*n, n_pair) (pq|ia) over every (i, a) pair in C order. On space-time
    v_act is the ISDF v of `active_pair_factors` and v_ov None: only the
    casida static defect reads it."""
    n, nocc = len(act), low.nocc
    if low.mode == 'space-time':
        b_act = _pair_factors(low, act).reshape(-1, n * n)
        return (b_act.T @ b_act).reshape((n,) * 4), None
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
    (spec §3.2). Zero when there is no active ph pair.

    On imagfrequency it is the full energy's quadrature on the full
    energy's grid (low.cache['grid'], not one of the active window):

        E_c^A = 1/(2 pi) sum_k w_k Tr[ln(1 - Pi^A(i w_k) U) + Pi^A(i w_k) U],

    in the active ph space, U = K^A the ph block of v_eff and Pi^A the
    diagonal bare response 2 f(i w) = -4 delta/(delta^2 + w^2), the spin
    and frequency factor of the full chi0
    (`solve_polarizability_imaginary_axis`). With every orbital active and
    v_eff the fitted v, U = B^T B and
    Tr ln(1 - Pi B^T B) = ln det(1 - B Pi B^T): the full integrand at every
    point, so the pair cancels to rounding. The dense-Casida value on the
    same v_eff is the 'ec_active_casida' diagnostic.

    On space-time the same quadrature on the energy's grid
    (low.cache['energy_grid']), with Pi^A(i w_k) = -2 sum_tau
    cosft_wt[k, tau] exp(-delta tau): exp(-delta tau) pushed through the
    energy's own tau -> omega transform, as the full chi0 was
    (`rpa_correlation_energy_space_time`), not the exact Lorentzian, so the
    pair again cancels to rounding with every orbital active."""
    k = part.n_act_occ
    if k == 0 or k == part.n_act:
        return 0.0
    if low.mode != 'casida':
        return _active_ec_quadrature(low, part, v_eff)
    return _active_ec_casida(low, part, v_eff)


def _active_ec_casida(low, part, v_eff):
    """The dense dRPA E_c^A[v_eff] of the active ph pairs by the plasmon
    formula (there is at least one active ph pair)."""
    return _plasmon(low.eps[part.act], part.n_act_occ, None, v_eff,
                    low.numerics)[0]


def _active_ec_quadrature(low, part, v_eff):
    """E_c^A[v_eff] on the imaginary axis, the formula of
    `active_correlation_energy`, with `dielectric_log_determinant` (its
    warning and flag for a non-positive det(1 - Pi^A U))."""
    delta, k_ph, _ = _active_ph(low, part, v_eff)
    eye = np.eye(delta.size)
    if low.mode == 'space-time':
        grid = low.cache['energy_grid']
        weights = grid.omega_weights
        pi = -2.0 * grid.cosft_wt @ np.exp(-np.outer(grid.tau_points, delta))
    else:
        points, weights = low.cache['grid']
        pi = [2.0 * frequency_factor(delta, w, True) for w in points]
    e_c = 0.0
    for k, (pi_k, wt) in enumerate(zip(pi, weights)):
        pu = pi_k[:, None] * k_ph
        e_c += wt * (dielectric_log_determinant(eye - pu, k) + np.trace(pu))
    return float(e_c / (2.0 * np.pi))


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

    On imagfrequency W(0) is the cache's: B_act^T [1 - P(0)]^-1 B_act with
    B_act `active_pair_factors` (ruling R4), against the same W^A on v_eff
    of the fitted v + vtilde, so the crpa defect is again an identity. On
    space-time likewise, with the static grid's response of each active ph
    pair in W^A: -4/delta becomes pi(0) = -2 sum_tau cosft_wt[0, tau]
    exp(-delta tau), the one P^A(0) carries (`active_polarizability_static`),
    i.e. delta_A -> -4/pi(0) above, without which the crpa defect would be
    the transform's error and not zero.
    """
    n = part.n_act
    if n == 0:
        return 0.0
    if low.mode != 'casida':
        b_act = active_pair_factors(low, part).reshape(-1, n * n)
        w_full = (b_act.T @ dyson_screening(low.cache['p0'])
                  @ b_act).reshape((n,) * 4)
    else:
        v_act, v_ov = _active_v(low, part.act)
        r = v_ov @ low.xpy
        w_full = v_act - 4.0 * ((r / low.omega) @ r.T).reshape((n,) * 4)
    w_act = v_eff.copy()
    delta, k_ph, v_ph = _active_ph(low, part, v_eff)
    if delta.size:
        if low.mode == 'space-time':
            grid = low.cache['static_grid']
            delta = 2.0 / (np.exp(-np.outer(delta, grid.tau_points))
                           @ grid.cosft_wt[0])
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
    RPA, low.factors or low.eri. Casida only: the other routes keep no
    eigenvectors, and refuse (NotImplementedError).
    """
    if low.mode != 'casida':
        raise NotImplementedError(
            f"screened_residues needs the Casida eigenvectors X+Y, which "
            f"mode={low.mode!r} does not keep: use mode='casida'.")
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


def lowest_excitation(low, solver=None):
    """Omega_min, the lowest excitation energy of the full RPA (None without
    a particle-hole pair), the one number the pole-free window needs.

    casida: the lowest of low.omega. imagfrequency, which diagonalizes
    nothing: the lowest root of the same restricted singlet Casida problem
    on low.factors, by the dense solve while the pair (A, B) fits in memory
    and by the Davidson above that (`solver_choice`), at the run's numerics;
    cached in low.cache['omega_min'] on the first request. solver: 'dense'
    or 'davidson' forces that solver and bypasses the cache (for tests).
    """
    if low.mode == 'casida':
        return float(low.omega.min()) if low.omega.size else None
    if solver is None and 'omega_min' in low.cache:
        return low.cache['omega_min']
    nocc = low.nocc
    n_pair = nocc * (len(low.eps) - nocc)
    om = None
    if n_pair:
        how = choice(solver or solver_choice(n_pair), ('dense', 'davidson'),
                     'solver')
        lr = LinearResponseSolver(low.eps, coeff_df=low.factors,
                                  spin_mode='restricted')
        if how == 'dense':
            omega = CasidaSolver(*lr.build_casida_matrices(
                nocc, lBSE=False)).solve(numerics=low.numerics)[0]
        else:
            omega = solve_casida_davidson(lr, nocc, numerics=low.numerics)[0]
        om = float(np.min(omega))
    if solver is None:
        low.cache['omega_min'] = om
    return om


def pole_window(low, part):
    """(lo, hi), the pole-free window of the environment self-energy:
    (max eps_I - Omega_min, min eps_Ev + Omega_min), Omega_min from
    `lowest_excitation`. Every pole of S^E lies at eps_i - Omega_n <= lo
    (inactive i) or eps_a + Omega_n >= hi (environment virtual a), so between
    them every D_tr is non-zero. A side without orbitals (or no excitation at
    all) is -inf or +inf; with neither side Omega_min is not computed."""
    if not len(part.inactive) and not len(part.env_virt):
        return -np.inf, np.inf
    om = lowest_excitation(low)
    if om is None:
        return -np.inf, np.inf
    eps = low.eps
    lo = eps[part.inactive].max() - om if len(part.inactive) else -np.inf
    hi = eps[part.env_virt].min() + om if len(part.env_virt) else np.inf
    return float(lo), float(hi)


def _refuse_near_poles(low, part, settings, sigma):
    """Refuse (ValueError) an active orbital whose energy is within the
    margin of an edge of `pole_window`, or outside it (spec §3.6): on casida
    pole_margin/sqrt(2 s), s = DOWNFOLDING_SRG_S when srg_s is None,
    guarding sigma='mode_a' alone and pointing to 'srg'; on imagfrequency
    the larger of that and continuation_margin (ruling R22: the Pade
    continuation loses accuracy towards the poles), guarding both forms and
    pointing to casida."""
    s = DOWNFOLDING_SRG_S if settings.srg_s is None else settings.srg_s
    margin = settings.pole_margin / np.sqrt(2.0 * s)
    why = (f"pole_margin {settings.pole_margin:g} / sqrt(2 s), s = {s:g}")
    if low.mode != 'casida' and settings.continuation_margin > margin:
        margin = settings.continuation_margin
        why = (f"continuation_margin {margin:g} Ha, where the Pade "
               f"continuation from the imaginary axis is accurate")
    lo, hi = pole_window(low, part)
    eps = low.eps[part.act]
    dist = np.minimum(eps - lo, hi - eps)
    bad = dist <= margin
    if bad.any():
        named = ', '.join(f'orbital {p} (eps = {e:.6f}, distance {x:.6f})'
                          for p, e, x in zip(part.act[bad], eps[bad],
                                             dist[bad]))
        if low.mode == 'casida':
            advice = ("Use sigma='srg', which is finite everywhere, or move "
                      "these orbitals out of the active space.")
        else:
            advice = (f"On mode={low.mode!r} sigma={sigma!r} is continued "
                      f"from the imaginary axis without damping (s = inf); "
                      f"use mode='casida', where sigma='srg' is finite "
                      f"everywhere, or move these orbitals out of the active "
                      f"space.")
        raise ValueError(
            f"sigma={sigma!r} diverges at the poles of the environment "
            f"self-energy: the active {named} are within the margin "
            f"{margin:.6f} Ha ({why}) of the pole-free window "
            f"({lo:.6f}, {hi:.6f}) or outside it (a negative distance). "
            + advice)


def embedding_self_energy(low, part, sigma):
    """Sigma_bar^E (n_act, n_act), the static environment self-energy on
    part.act (occupied first): iG^E(W - v) with r over part.env, the full
    RPA's W and the mean-field energies (spec §3.6).

    casida: the residues of W (`screened_residues`) and the RPA's Omega.
    sigma 'srg' is Eq. 44 at s = numerics.downfolding.srg_s (None: Eq. 42),
        finite everywhere; 'mode_a', Eq. 12 at eta = 0, refused (naming the
        orbitals) when an active energy is within pole_margin/sqrt(2 s) of
        the pole-free window's edge or outside it (`pole_window`).
    imagfrequency: S^E(mu + i w) on the Pade nodes (`sigma_env_imag_axis`),
        continued once and evaluated per element (`continued_self_energy`):
        'srg' is Eq. 42, Re S^E_tu(z_tu), at s = inf whatever srg_s is
        (srg_s enters the pole check alone), 'mode_a' 1/2 Re[S^E_tu(eps_t)
        + S^E_tu(eps_u)]. Both are refused near the poles as 'mode_a' is
        on casida, within the larger of pole_margin/sqrt(2 s) and
        numerics.downfolding.continuation_margin, pointing to mode='casida'.
    space-time: not built yet (NotImplementedError, pointing to casida and
        imagfrequency).
    Zero when the environment is empty.
    """
    return _embedding(low, part, sigma)[0]


def _embedding(low, part, sigma):
    """(Sigma_bar, kept) of `embedding_self_energy`, kept what the
    diagnostics read without building it again: on casida 'args' = (w, eps_t,
    eps_r, occupied_r, omega) of the environment and 'srg_s', the s applied
    by 'srg'; on imagfrequency 'z' and 'samples', the Pade nodes and S^E
    there, and 'fit', the (Z, A) of `pade_fit_elementwise` (None without an
    environment). Not built on space-time (NotImplementedError)."""
    sigma = choice(sigma, SIGMAS[1:], 'sigma')
    if low.mode == 'space-time':
        raise NotImplementedError(
            f"method={METHOD!r} sigma={sigma!r} (the environment self-energy)"
            f" is not built on mode='space-time' yet: use mode='casida' or "
            f"mode='imagfrequency'.")
    settings = numerics_or_default(low.numerics).downfolding
    if sigma == 'mode_a' or low.mode != 'casida':
        _refuse_near_poles(low, part, settings, sigma)
    if low.mode != 'casida':
        return _embedding_imag_axis(low, part, sigma)
    env = part.env
    args = (screened_residues(low, part), low.eps[part.act], low.eps[env],
            env < low.nocc, low.omega)
    kept = {'args': args, 'srg_s': settings.srg_s}
    if sigma == 'srg':
        return srg_self_energy(*args, settings.srg_s), kept
    return mode_a_self_energy(*args), kept


def _screening_grid(low):
    """(points, W): the full RPA's W(i w) = [1 - chi0(i w)]^-1 in the
    normalized fitting basis on the minimax grid of numerics.imagfreq.nfreq
    points over the particle-hole window, the W grid of the GW density's
    `self_energy_matrix_imaginary_axis`; built on the first request and kept
    in low.cache['w_grid'] (spec §4.5)."""
    if 'w_grid' not in low.cache:
        nfreq = numerics_or_default(low.numerics).imagfreq.nfreq
        e_min, e_max, _ = particle_hole_window(low.eps, low.nocc)
        points = minimax_frequency_grid(nfreq, e_min, e_max)[0]
        lr = LinearResponseSolver(low.eps, coeff_df=low.factors,
                                  spin_mode='restricted')
        low.cache['w_grid'] = (points, lr.solve_rpa_screening(
            points, low.nocc, is_imaginary=True))
    return low.cache['w_grid']


def sigma_env_imag_axis(low, part, query_freqs):
    """S^E_tu(mu + i w) (nquery, n_act, n_act) complex, at w = query_freqs:
    the correlation self-energy iG(W - v) of the full RPA's W on part.act
    (occupied first) with the internal orbital restricted to part.env, mu
    the whole system's mid-gap. imagfrequency only (NotImplementedError
    otherwise).

    This is the GW density's `fitted_self_energy_matrix_imaginary_axis` on
    the three-index factors sliced to B[Q, t, r] (t active, r environment)
    and the environment's eps_r - mu: the Lorentzian fit of W - 1 on
    `_screening_grid` with its poles and regularization
    (GW_DENSITY_W_FIT_*), whose one-pole convolution with G_r is exact; the
    full system's mu and fit range, so only the internal sum changes.
    """
    if low.mode != 'imagfrequency':
        raise NotImplementedError(
            f"sigma_env_imag_axis samples the density-fitted W of "
            f"mode='imagfrequency'; this low-level result is mode="
            f"{low.mode!r} (its self-energy is `embedding_self_energy`): "
            f"use mode='imagfrequency'.")
    n, env = part.n_act, part.env
    query_freqs = np.atleast_1d(query_freqs)
    if n == 0 or len(env) == 0:
        return np.zeros((len(query_freqs), n, n), dtype=complex)
    points, w_grid = _screening_grid(low)
    e_min, e_max, mu = particle_hole_window(low.eps, low.nocc)
    poles = np.geomspace(GW_DENSITY_W_FIT_LOW * e_min,
                         GW_DENSITY_W_FIT_HIGH * e_max, len(points))
    b = low.factors[:, part.act][:, :, env]
    return fitted_self_energy_matrix_imaginary_axis(
        b, low.eps[env] - mu, points, w_grid, query_freqs, poles,
        GW_DENSITY_W_FIT_TIKHONOV)


def _evaluate_continued(fit, eps_act, sigma):
    """The static reduction of a `pade_fit_elementwise` fit (Z, A) of S^E:
    'srg', Re S_tu(z_tu) with z_tu = (eps_t + eps_u)/2 + i |eps_u - eps_t|/2
    (Eq. 42; S(z*) = S(z)*, so the half plane of the fit's nodes serves both
    t, u and u, t); 'mode_a', 1/2 Re[S_tu(eps_t) + S_tu(eps_u)]."""
    e = np.asarray(eps_act, dtype=float)
    n = len(e)
    if sigma == 'srg':
        z = 0.5 * (e[:, None] + e[None, :]) + 0.5j * np.abs(e[None, :]
                                                            - e[:, None])
        at = pade_eval_elementwise(z.ravel(), *fit).reshape((n,) * 4)
        return np.einsum('tutu->tu', at).real
    half = np.einsum('ttu->tu', pade_eval_elementwise(e, *fit)).real
    return 0.5 * (half + half.T)


def continued_self_energy(z, samples, eps_act, sigma):
    """Sigma_bar (n_act, n_act) from S^E sampled at the complex frequencies z
    (samples (nz, n_act, n_act), symmetric): one Thiele-Pade fit per element
    (`pade_fit_elementwise`, greedy order, the upper triangle), then each
    element at its own point, Re S_tu(z_tu) for 'srg' (Eq. 42, s = inf) or
    1/2 Re[S_tu(eps_t) + S_tu(eps_u)] for 'mode_a', eps_act the active
    orbitals' energies."""
    if len(eps_act) == 0:
        return np.zeros((0, 0))
    fit = pade_fit_elementwise(z, samples, symmetric=True)
    return _evaluate_continued(fit, eps_act, sigma)


def _embedding_imag_axis(low, part, sigma):
    """`_embedding` on imagfrequency: S^E on the Pade nodes of the run's
    numerics (`pade_nodes`, on the line mu + i w), fitted once."""
    nodes = pade_nodes(numerics_or_default(low.numerics).quadrature)
    z = particle_hole_window(low.eps, low.nocc)[2] + 1j * nodes
    samples = sigma_env_imag_axis(low, part, nodes)
    kept = {'z': z, 'samples': samples, 'fit': None}
    n = part.n_act
    if n == 0 or len(part.env) == 0:
        return np.zeros((n, n)), kept
    kept['fit'] = pade_fit_elementwise(z, samples, symmetric=True)
    return _evaluate_continued(kept['fit'], low.eps[part.act], sigma), kept


class SigmaDampingWarning(RuntimeWarning):
    """The SRG damping of Sigma_bar^E is not negligible: an active orbital
    energy lies within a few 1/sqrt(2 s) of a pole of the environment
    self-energy, so 'srg' there depends on srg_s."""


class SigmaContinuationWarning(RuntimeWarning):
    """Two Pade fits of S^E on disjoint halves of the nodes disagree by more
    than DOWNFOLDING_SIGMA_CONTINUATION_WARN at an element's evaluation
    point: the continuation of Sigma_bar^E is not to be trusted there."""


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


# The step of the central difference that gives S'_tt(eps_t) from the
# continued fraction, Ha: the fraction is smooth on the real axis inside the
# pole-free window (the poles are at least the pole margin away), so the
# O(h^2) error is far below the continuation's own.
_SLOPE_STEP = 1e-5


def _continued_pole_weights(fit, eps_act):
    """Z_t = 1/(1 - S'_tt(eps_t)) from a Pade fit of S^E, the slope by a
    central difference of the continued fraction at eps_t. 1 for every t
    without a fit (no environment)."""
    e = np.asarray(eps_act, dtype=float)
    if fit is None:
        return np.ones(len(e))
    h = _SLOPE_STEP
    step = (pade_eval_elementwise(e + h, *fit)
            - pade_eval_elementwise(e - h, *fit)).real / (2.0 * h)
    return 1.0 / (1.0 - np.einsum('ttt->t', step))


def _sigma_poles(low, part, sigma, sigma_bar, kept):
    """The 'sigma_poles' record (spec §4.6), and a SigmaDampingWarning naming
    the active orbitals whose row of Sigma_bar the SRG damping moves by more
    than DOWNFOLDING_SRG_DAMPING_WARN.

    window: `pole_window`. pole_distance: per active orbital, the distance
        of eps_t to the nearer edge of the window, negative outside it.
    z: per active orbital, Z_t = 1/(1 - S'_tt(eps_t)): of the pole sum on
        casida (`_pole_weights`), of the continued fraction on imagfrequency.
    srg_s: the s applied, None for s = inf (always on imagfrequency, which
        applies no damping) and for 'mode_a' (undamped).
    srg_damping: max |Sigma_bar_srg(s) - Sigma_bar_srg(inf)|; 0.0 when no
        finite s is applied.
    """
    lo, hi = pole_window(low, part)
    eps = low.eps[part.act]
    dist = np.minimum(eps - lo, hi - eps)
    if low.mode != 'casida':
        return {'window': (lo, hi), 'pole_distance': dist,
                'z': _continued_pole_weights(kept['fit'], eps),
                'srg_s': None, 'srg_damping': 0.0}
    args, srg_s = kept['args'], kept['srg_s']
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


def continuation_spread(z, samples, eps_act, sigma):
    """(n_act, n_act): per element, |Sigma_bar from the even-indexed z -
    Sigma_bar from the odd-indexed z|, each by `continued_self_energy` at
    the element's own evaluation point. Two fits on disjoint halves of the
    nodes: an estimate of the continuation's error that, with half the
    nodes each, overstates that of the full fit."""
    z, samples = np.asarray(z), np.asarray(samples)
    even, odd = (continued_self_energy(z[k::2], samples[k::2], eps_act,
                                       sigma) for k in (0, 1))
    return np.abs(even - odd)


def _sigma_continuation(low, part, sigma, kept):
    """The 'sigma_continuation' record (spec §4.6): `continuation_spread` on
    the nodes Sigma_bar was continued from, and a SigmaContinuationWarning
    naming the elements (as orbital pairs) above
    DOWNFOLDING_SIGMA_CONTINUATION_WARN. Zero without an environment."""
    n = part.n_act
    if kept['fit'] is None:
        return np.zeros((n, n))
    spread = continuation_spread(kept['z'], kept['samples'],
                                 low.eps[part.act], sigma)
    bad = np.argwhere(np.triu(spread > DOWNFOLDING_SIGMA_CONTINUATION_WARN))
    if len(bad):
        act = part.act
        named = ', '.join(f'({act[t]}, {act[u]}) {spread[t, u]:.2e}'
                          for t, u in bad)
        warnings.warn(
            f"sigma={sigma!r} on mode={low.mode!r}: two Pade fits of the "
            f"environment self-energy on the even and the odd nodes differ "
            f"by more than {DOWNFOLDING_SIGMA_CONTINUATION_WARN:g} Ha at the "
            f"elements {named} (orbital pairs, Ha); the continued Sigma_bar^E "
            f"there is uncertain. More nodes (numerics.quadrature.npade) or "
            f"mode='casida' avoid the continuation.",
            SigmaContinuationWarning, stacklevel=4)
    return spread


def _diagnostic_names(diagnostics):
    """The DIAGNOSTICS entries `diagnostics` names (a name or an iterable of
    names), each refused unless it is one."""
    if isinstance(diagnostics, str):
        diagnostics = (diagnostics,)
    return {choice(d, DIAGNOSTICS, 'diagnostics') for d in diagnostics}


def terms(low, part, screening='crpa', sigma=None, diagnostics=DIAGNOSTICS,
          **opts):
    """The DownfoldingTerms of the RPA on the active space `part`.

    screening: 'bare' | 'crpa' | 'crpa_ph' ('mrpa' not built yet, and
        casida only); vtilde is its static correction
        (`screening_correction`), the same in every spin block (restricted).
    sigma: None, the paper's scheme: full double counting, no Sigma_bar.
        'srg' or 'mode_a': the static environment self-energy
        (`embedding_self_energy`) in both spin channels with dc='hartree',
        the Hartree-only double counting that goes with it; E_0 then gains
        the exchange energy term and -Tr[Sigma_bar gamma^A,0] (spec §3.6,
        §3.7). Allowed with every screening; the third term of the
        embedding self-energy, iG^A(W - W^A[v_eff]), is dropped, which is
        exact only for 'crpa' in the static limit. On imagfrequency 'srg'
        is s = inf (Eq. 42) and both forms are refused near the poles,
        pointing to casida (`embedding_self_energy`); not built on
        space-time (NotImplementedError).
    diagnostics: the DIAGNOSTICS to record in energies.diagnostics; a name
        that does not apply here (on casida 'ec_active_casida' and
        'sigma_continuation', and 'sigma_poles' and 'sigma_continuation'
        without sigma) is skipped and absent, an unknown name refused.
        'sigma_poles' (`_sigma_poles`) warns (SigmaDampingWarning) when the
        SRG damping is not negligible (casida). 'sigma_continuation'
        (imagfrequency, `_sigma_continuation`): per element the spread of
        two Pade fits on the even and the odd nodes, warning
        (SigmaContinuationWarning) above DOWNFOLDING_SIGMA_CONTINUATION_WARN.
        'ec_active_casida' (imagfrequency, space-time): {'casida': the
        dense-Casida E_c^A on the same v_eff, 'difference': the route's
        quadrature value minus it}.

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
    e_c_active = active_correlation_energy(low, part, v_eff)
    if 'ec_active_casida' in names and low.mode != 'casida':
        k = part.n_act_occ
        dense = (0.0 if k in (0, part.n_act)
                 else _active_ec_casida(low, part, v_eff))
        found['ec_active_casida'] = {'casida': dense,
                                     'difference': e_c_active - dense}
    sigma_bar, dc = None, 'full'
    if sigma is not None:
        s_bar, kept = _embedding(low, part, sigma)
        sigma_bar, dc = (s_bar, s_bar), 'hartree'
        if 'sigma_poles' in names:
            found['sigma_poles'] = _sigma_poles(low, part, sigma, s_bar, kept)
        if 'sigma_continuation' in names and low.mode != 'casida':
            found['sigma_continuation'] = _sigma_continuation(low, part,
                                                              sigma, kept)
    return DownfoldingTerms(
        vtilde=(vt, vt, vt), sigma_bar=sigma_bar, e_c_full=low.e_c_full,
        e_c_active=e_c_active, dc=dc, diagnostics=found)
