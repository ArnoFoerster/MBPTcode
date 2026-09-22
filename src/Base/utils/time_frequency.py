"""One imaginary-axis grid object carrying BOTH axes and the transforms between
them, so a space-time RPA/GW driver can be written once and run on either the
T = 0 minimax grids or the finite-temperature IR basis.

    grid = TimeFrequencyGrid.minimax(nfreq, e_min, e_max)
    grid = TimeFrequencyGrid.ir(beta, omega_max, eps=1e-10, statistics='boson')

    Pi_w = grid.to_omega(Pi_tau, parity='even')      # cosine transform
    Sig_w = grid.to_omega(Sig_tau, parity='odd')     # sine transform

Both carry `tau_points`, `tau_weights`, `omega_points`, `omega_weights` and the
four transform matrices, so the consumer never branches on `method`.

WHY FOUR TRANSFORM MATRICES AND NOT ONE
---------------------------------------
The natural request is a single `transform_weights` whose inverse runs the other
way. That does not exist for the minimax grids, and the reason is structural
rather than a tolerance that can be tightened.

GreenX builds each direction by its OWN regularized least-squares fit
(minimax_grids.F90::get_transformation_weights): the tau -> omega cosine weights
solve `sum_j w_j cos(w_k tau_j) exp(-x tau_j) = 2x/(x^2 + w_k^2)` over a
logarithmic node set x in [e_min, e_max], and the omega -> tau weights solve the
mirrored problem. Nothing couples the two fits, so their product is not the
identity -- GreenX itself returns `cosft_duality_error = max|A B - I|` as an
output precisely because callers are expected to look at it.

Measured here (e_min = 0.4, e_max = 40, tests/test_time_frequency_grid.py).
"model" is the worst relative error of the forward transform on the pair it is
fitted for, e^{-x tau} <-> 2x/(x^2 + w^2); "roundtrip" is to_tau(to_omega(f))
against f on that same family:

    n     model      roundtrip   max|A B - I|   cond(A)
    6     1.1e-01    2.9e-05     1.1e-03        1.4e+02
    10    2.6e-03    3.9e-05     1.3e-01        3.7e+02
    14    1.4e-04    6.2e-08     1.5e-02        6.1e+02
    18    1.0e-05    8.7e-09     5.0e-01        1.1e+03
    20    2.4e-06    8.9e-08     9.6e+00        1.2e+03
    24    2.1e-06    6.4e-09     1.5e+00        5.7e+11
    30    1.2e-06    5.1e-09     8.9e-01        1.2e+63
    34    1.6e-06    2.1e-08     1.2e+00        5.8e+223

Read the last two columns together. `max|A B - I|` is LARGE and does not shrink
with n -- A and B are mutual inverses only on the model subspace, not on all of
R^n -- and the transform is not orthogonal either (`max|A^T A - I| = 1.2e4`,
singular values 0.023 .. 14). But every physical Pi(i.tau) IS in that subspace,
being a sum of exponentials with x in [e_min, e_max], so the round trip on real
data is fine: 6e-8 at n = 14. Use `roundtrip_error()` to compare methods and
treat `duality_error()` as the raw matrix diagnostic it is HERE; judging
minimax by `|A B - I|` alone rejects a transform that works. On an IR grid
`duality_error` is not that product at all but the round trip over the
represented subspace -- see its own docstring for why the raw product is
unusable there.

The real constraint on minimax is CONDITIONING, not duality. A sum of
exponentials is an exponentially ill-conditioned basis, so cond(A) climbs
without bound in n -- 6e11 at 24 points and 6e223 at 34 -- and the fit is only
obtainable at all because a Tikhonov term is applied automatically above
`_REGULARIZATION_ABOVE`. With it the forward error does not turn over: it falls
to ~2e-06 and then sits on the tabulated coefficients' own precision. What does
bite is too FEW points for a wide range, where the fit is genuinely
under-resolved and the warning fires; the answer there is always more points.
Use method='IR' for a finite-temperature grid, not to escape a point count.

Both directions are still carried as separate matrices, since neither is the
other's inverse as a matrix and inverting one would throw away the fit that was
done for the other.

This costs nothing for the algorithms that matter, because space-time RPA/GW
never round-trips: Pi(i.tau) is built directly from the THC factors, transformed
ONCE tau -> omega, and the Dyson equation is solved on the frequency axis; the
self-energy makes one sine transform back. Each is a single forward use on the
model space the fit targets. It does mean `to_tau(to_omega(f))` is not f, and
any algorithm needing that must use `method='IR'`, whose transform goes through
basis coefficients and is a genuine two-sided representation.

References
----------
GreenX GX-TimeFrequency/src/minimax_grids.F90, whose
`get_transformation_weights` / `calculate_psi_and_mat_A` this module ports,
and minimax_utils.F90, whose transformation-type codes it keeps. GreenX
(https://github.com/nomad-coe/greenX) is licensed under the Apache License,
Version 2.0; this port and the modifications made to it are recorded in the
NOTICE file at the repository root, which must be retained in redistributions.
The finite-temperature (IR) backend reached through `TimeFrequencyGrid.ir` is
not GreenX-derived -- see `src/Base/utils/matsubara.py` for its own references.
Kaltak, Klimes and Kresse, J. Chem. Theory Comput. 10, 2498 (2014) for the
minimax quadratures the tables encode.
Liu et al., Phys. Rev. B 94, 165109 (2016), Eq. 71 for the sine transform.
"""
import warnings

import numpy as np

from src.Base.utils.grids import (minimax_frequency_grid, minimax_time_grid,
                                  gauss_legendre_grid, _load_minimax_tau_data)
from src.Base.utils.matsubara import IRBasis, matsubara_frequencies

# GreenX minimax_utils.F90 transformation-type codes, kept so the port stays
# line-comparable with the Fortran.
COSINE_TW, COSINE_WT, SINE_TW, SINE_WT = 1, 2, 3, 4

#: Above this many points the unregularized SVD solve loses conditioning (see
#: the module docstring's table); a Tikhonov term is applied automatically.
_REGULARIZATION_ABOVE = 20
_DEFAULT_REGULARIZATION = 1e-8

#: A transform fitting worse than this is reported: the accuracy is set by the
#: INTERPLAY of n and the energy range, not by either alone, so the measured fit
#: is the only honest guard -- see `minimax_convergence_floor`.
_FIT_ERROR_WARN = 1e-3


def minimax_convergence_floor(npoints):
    """Narrowest e_max/e_min GreenX tabulated a Remez column for at this size.

    Approximating 1/x by a sum of exponentials over a NARROW range is
    exponentially ill-conditioned in the number of terms, so the generator was
    only run once the window was wide enough; the floor climbs with n while the
    number of tabulated windows thins out --

        n      6     8    14    16    20     24     28     30     32     34
        floor  1.6   10    10   100   200    700   1545   2906   4862   9649
        rows    13   13    21    36    39     40     28     28     18      7

    This is NOT an accuracy limit. Below the floor the lookup takes the widest
    column, which is a valid fit on any sub-range of its own, and above 20
    points GreenX slides that column onto the requested range exactly
    (`grids._table_row`). Measured tau -> omega cosine fit error, which falls
    with n at every range (tests/test_time_frequency_grid.py):

        range     n=14     n=20     n=24     n=30     n=34
        1e2       1.2e-5   1.0e-7   7.7e-7   5.0e-7   6.1e-7
        1e3       6.2e-5   2.4e-6   1.3e-7   5.2e-8   6.4e-8
        1e4       8.0e-5   1.7e-5   2.1e-6   1.1e-7   1.9e-8
        1e5       9.0e-5   3.8e-5   9.3e-6   2.5e-7   5.3e-8

    What the floor is good for is naming the direction of a failed fit: a
    transform that misses is under-resolved for its range, and the cure is more
    points, never fewer.
    """
    data = _load_minimax_tau_data()
    row = data.get(str(npoints))
    return float(row['energy_range'][0]) if row else None


def laplace_test_error(npoints, R, nsample=4001, relative=False):
    """
    The Kaltak-Klimes-Kresse test integral (JCTC 2014, 10, 2498, eqs 4-6): how
    well does an `npoints` minimax grid represent 1/x on I = [1, R]?

        eta(x) = 1/x - sum_i beta_i exp(-alpha_i x),     x in I = [1, R]
        ||eta||_inf = max { |eta(x)| : x in I }

    with R = e_max / e_min their section 2.5. This is not a diagnostic bolted on
    after the fact -- it is the very problem the minimax grid solves, so the
    returned number is the Remez residual of the grid actually in use. Their
    eq 6 is Chebyshev alternation, eta(x_j) = (-1)^j eps over 2N+1 extrema, and
    the tabulated GreenX coefficients satisfy it in exactly this norm: at
    R = 100 and N = 12 the interior extrema of eta are equal to five digits
    (max/min = 1.00), whereas the x-weighted relative error spreads by a factor
    80. So ABSOLUTE is the norm the grids were built in, and the norm to judge
    them by; `relative=True` returns |1 - x sum_i ...| instead, which is the
    more intuitive "how many digits" measure but does not equioscillate.

    R grows as the gap closes -- e_min is the smallest transition energy -- so a
    fixed npoints is wrong for a size series: the acenes need more time points
    at 12 rings than at 2 for the same accuracy.

    The error equioscillates, so sampling logarithmically on [1, R] finds the
    extrema without needing the Remez nodes themselves.
    """
    tau, sigma = minimax_time_grid(npoints, 1.0, float(R))
    x = np.exp(np.linspace(0.0, np.log(float(R)), nsample))
    approx = np.exp(-np.outer(x, tau)) @ sigma
    if relative:
        return float(np.abs(1.0 - x * approx).max())
    return float(np.abs(1.0 / x - approx).max())


def minimax_points_for_accuracy(e_min, e_max, target=1e-6, npoints_max=34,
                                relative=False):
    """
    Smallest minimax grid whose Laplace test error meets `target`.

    Returns (npoints, achieved_error). If nothing in range reaches the target
    the best available is returned, and the caller should treat the second value
    as the accuracy it is actually getting rather than assume `target` was met.
    """
    R = float(e_max) / float(e_min)
    best = (None, np.inf)
    for n in range(6, npoints_max + 1, 2):      # GreenX tabulates even sizes >= 6
        try:
            err = laplace_test_error(n, R, relative=relative)
        except Exception:              # this size is not tabulated; try the next
            continue
        if err < best[1]:
            best = (n, err)
        if err <= target:
            return n, err
    return best


#: Kaltak-Klimes-Kresse residual that buys ~0.1 meV on the quasiparticle
#: energy. Calibrated on naphthalene/cc-pVDZ at R_Sigma = 486, against the
#: ntau -> infinity limit of the space-time HOMO:
#:     eta 1.2e-08 -> 1.96 meV     eta 6.0e-10 -> 0.32 meV
#:     eta 4.8e-11 -> 0.08 meV     eta 3.7e-12 -> 0.03 meV
#: The mapping is empirical -- eta bounds the quadrature, not the self-energy
#: it is used to integrate -- so this is a calibrated default, not a bound.
#:
#: It lives HERE, with the grid machinery, and `GW.imaginary_time` re-exports
#: it: every space-time route needs it, and a constant defined twice is a
#: constant that eventually differs.
DEFAULT_TAU_TARGET = 1e-10

#: Padding on the two self-energy fit ranges, as (below, above).
#:
#: The 0.3 is not a fudge factor. The screened interaction has weight BELOW the
#: smallest transition -- collective excitations sit under the gap -- so a range
#: starting at the gap misfits where it is largest. Measured on the round trip
#: at ntau = 18: 1.3e-3 unpadded, 5.0e-8 widened. The 3.0 mirrors it for the
#: tail.
#:
#: DO NOT MERGE THIS WITH `matsubara.self_energy_range`. That one is unpadded on
#: purpose: it sizes an analytic continuation, which wants the honest extent,
#: where this sizes a least-squares fit, which needs room at both ends. They
#: share a core expression, which is what makes them look mergeable.
SELF_ENERGY_PAD = (0.3, 3.0)


def _psi_and_matrix(kind, tau, omega, i, x):
    """GreenX calculate_psi_and_mat_A: the fit target psi(x) and design matrix.

    kind selects which of the four transforms is being fitted; `i` indexes the
    point of the OUTPUT axis whose row of weights is being solved for.
    """
    if kind == COSINE_TW:                       # cos: tau -> omega, fit at omega[i]
        w = omega[i]
        return (2 * x / (x**2 + w**2),
                np.cos(w * tau)[None, :] * np.exp(-np.outer(x, tau)))
    if kind == COSINE_WT:                       # cos: omega -> tau, fit at tau[i]
        t = tau[i]
        return (np.exp(-x * t),
                np.cos(t * omega)[None, :] * (2 * x[:, None] / (x[:, None]**2 + omega[None, :]**2)))
    if kind == SINE_TW:                         # sin: tau -> omega, fit at omega[i]
        w = omega[i]
        return (2 * w / (x**2 + w**2),
                np.sin(w * tau)[None, :] * np.exp(-np.outer(x, tau)))
    if kind == SINE_WT:                         # sin: omega -> tau, fit at tau[i]
        t = tau[i]
        return (np.exp(-x * t),
                np.sin(t * omega)[None, :] * (2 * omega[None, :] / (x[:, None]**2 + omega[None, :]**2)))
    raise ValueError(f"unknown transformation type {kind}")


def minimax_transform_weights(kind, tau, omega, e_min, e_max,
                              regularization=None, nodes_factor=200,
                              warn=True):
    """One transform matrix by GreenX's per-point regularized least squares.

    Returns (W, max_fit_error). W already carries the cos/sin factor absorbed,
    matching GreenX's default (`bare_cos_sin_weights = .false.`), so it is used
    as a plain matrix-vector product.

    regularization=None picks 0 below _REGULARIZATION_ABOVE points and
    _DEFAULT_REGULARIZATION at or above it -- see the module docstring for the
    conditioning measurement that sets that threshold.

    warn: flag a short fit as a RuntimeWarning. The residual is a signal about
    the GRID only when both axes come from ONE energy range, which is how
    `TimeFrequencyGrid.minimax` builds them -- there it discriminates a grid too
    dense for a narrow range from one too sparse for a wide one, and the hint
    below points the right way. The self-energy transforms pass warn=False
    because their axes are DECOUPLED by construction: tau carries the
    self-energy's range, omega the polarizability's, and the fit range is a
    third. A short residual there says the ranges differ, which is the method,
    not a grid to fix -- and the range is set by core states and high-lying
    virtuals, so it does not bound the frontier energies anyone reads. The
    number is returned either way.
    """
    # Each OUTPUT point is fitted independently, and the design matrix has one
    # column per INPUT point -- so the conditioning is governed by n_in alone
    # and the two axes need not have the same length. Keeping n_tau inside the
    # Remez window while transforming onto an arbitrarily dense frequency grid
    # is what lets the space-time route converge its self-energy quadrature.
    n_out = len(omega) if kind in (COSINE_TW, SINE_TW) else len(tau)
    n_in = len(tau) if kind in (COSINE_TW, SINE_TW) else len(omega)
    n = n_in
    if regularization is None:
        regularization = 0.0 if n < 20 else _DEFAULT_REGULARIZATION

    nx = max((int(np.log10(e_max / e_min)) + 1) * nodes_factor, n)
    x = e_min * (e_max / e_min) ** (np.arange(nx) / (nx - 1.0))

    W = np.zeros((n_out, n_in))
    max_error = 0.0
    for i in range(n_out):
        psi, A = _psi_and_matrix(kind, tau, omega, i, x)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        # S/(reg^2 + S^2) is 0/0 at an exactly singular value when reg is 0,
        # which is its default below _REGULARIZATION_ABOVE points. That NaNs
        # the weights and the residual, and a NaN residual passes the threshold
        # test below silently -- the diagnostic disables itself exactly where
        # the fit is most degenerate. Drop the null space instead, as the
        # pseudo-inverse does; every non-zero value is untouched.
        filt = np.zeros_like(S)
        nz = S > 0.0
        filt[nz] = S[nz] / (regularization**2 + S[nz]**2)
        W[i] = Vt.T @ (filt * (U.T @ psi))
        max_error = max(max_error, float(np.abs(A @ W[i] - psi).max()))

    if warn and not max_error <= _FIT_ERROR_WARN:   # NaN must not pass
        floor = minimax_convergence_floor(n)
        hint = ''
        if floor is not None:
            hint = (f" The usable n RISES with the energy range and this fit "
                    f"is under-resolved at e_max/e_min = {e_max/e_min:.3g}, so "
                    f"use MORE points; n={n} is tabulated down to "
                    f"e_max/e_min = {floor:.3g} and stretched below that.")
        warnings.warn(
            f"minimax transform fit reached only {max_error:.2e} at {n} points "
            f"over e_max/e_min = {e_max/e_min:.3g}.{hint}",
            RuntimeWarning, stacklevel=3)

    trig = np.cos if kind in (COSINE_TW, COSINE_WT) else np.sin
    phase = trig(np.outer(omega, tau))
    # *_TW produces the omega axis (rows = omega); *_WT produces the tau axis.
    return (W * phase if kind in (COSINE_TW, SINE_TW) else W * phase.T), max_error


class TimeFrequencyGrid:
    """Imaginary-time and imaginary-frequency axes plus the maps between them.

    Attributes
    ----------
    tau_points, tau_weights     : (ntau,)   imaginary-time quadrature
    omega_points, omega_weights : (nfreq,)  imaginary-frequency quadrature.
        `omega_weights` absorbs the difference between the T = 0 integral
        `(1/2pi) int dw` and the Matsubara sum `(1/beta) sum_n`, so a consumer
        contracts against them identically in both methods.
    cosft_wt : (nfreq, ntau)  even-in-tau  -> omega
    cosft_tw : (ntau, nfreq)  even-in-omega -> tau
    sinft_wt : (nfreq, ntau)  odd-in-tau   -> omega
    sinft_tw : (ntau, nfreq)  odd-in-omega -> tau
    method   : 'minimax' | 'IR' | 'gauss_legendre'
    fit_errors : dict of the per-transform construction error, or None for IR
    """

    def __init__(self, tau_points, tau_weights, omega_points, omega_weights,
                 cosft_wt=None, cosft_tw=None, sinft_wt=None, sinft_tw=None,
                 method='minimax', fit_errors=None, meta=None):
        self.tau_points = np.asarray(tau_points)
        self.tau_weights = np.asarray(tau_weights)
        self.omega_points = np.asarray(omega_points)
        self.omega_weights = np.asarray(omega_weights)
        self.cosft_wt, self.cosft_tw = cosft_wt, cosft_tw
        self.sinft_wt, self.sinft_tw = sinft_wt, sinft_tw
        self.method = method
        self.fit_errors = fit_errors or {}
        self.meta = meta or {}

    # ---- constructors -----------------------------------------------------

    @classmethod
    def minimax(cls, npoints, e_min, e_max, regularization=None, with_sine=True):
        """T = 0 minimax grids (GreenX tables) + the ported transform fits.

        e_min / e_max are the smallest and largest transition energies; for a
        particle-hole bubble those are the HOMO-LUMO gap and the full
        eps_max - eps_min spread. A metal has no e_min -- use `ir` there, or
        `matsubara.thermal_e_min` to supply one.
        """
        omega_points, omega_weights = minimax_frequency_grid(npoints, e_min, e_max)
        tau_points, tau_weights = minimax_time_grid(npoints, e_min, e_max)

        # The SAME tabulated tau coefficients serve two different purposes with
        # two different scalings, and picking the wrong one silently costs two
        # to three orders of magnitude. `minimax_time_grid` scales by 1/e_min,
        # which is right for its own job -- the Laplace representation of 1/x
        # that laplace_codegen.py consumes. GreenX's time-FREQUENCY pair
        # (minimax_grids.F90:134, `scaling = 2.0_dp * e_min`) scales by
        # 1/(2 e_min) instead, and that is what the cosine/sine fits below want.
        # Measured on the model pair e^{-x tau} <-> 2x/(x^2+w^2), worst relative
        # error over x in [e_min, e_max] at e_max/e_min = 100:
        #     n=10   1/e_min 4.7e-01   vs   1/(2 e_min) 1.4e-02
        #     n=14   1/e_min 4.5e-02   vs   1/(2 e_min) 1.1e-04
        #     n=18   1/e_min 1.5e-03   vs   1/(2 e_min) 5.2e-06
        # so use GreenX's. The narrow-range e_ratio stretch is orthogonal to
        # this factor and cancels against the frequency axis, so it does not
        # enter here; any residual loss shows up in `fit_errors`.
        tau_points = 0.5 * tau_points
        tau_weights = 0.5 * tau_weights

        kinds = [('cosft_wt', COSINE_TW), ('cosft_tw', COSINE_WT)]
        if with_sine:
            kinds += [('sinft_wt', SINE_TW), ('sinft_tw', SINE_WT)]

        mats, errs = {}, {}
        for name, kind in kinds:
            mats[name], errs[name] = minimax_transform_weights(
                kind, tau_points, omega_points, e_min, e_max,
                regularization=regularization)

        return cls(tau_points, tau_weights, omega_points, omega_weights,
                   method='minimax', fit_errors=errs,
                   meta={'e_min': e_min, 'e_max': e_max, 'npoints': npoints},
                   **mats)

    @classmethod
    def minimax_split(cls, ntau, e_min, e_max, omega_points, omega_weights,
                      regularization=None, with_sine=True, with_inverse=True):
        """Minimax tau axis of size `ntau`, transformed onto an INDEPENDENT
        frequency grid supplied by the caller.

        `TimeFrequencyGrid.minimax` ties n_omega to n_tau, which conflates two
        unrelated jobs: the tau->omega transform is a fit, and n_tau is set by
        how many exponentials the energy range needs, while the self-energy
        convolution is a quadrature that wants many frequency points to
        converge. Tying them caps the space-time GW routes at ~2 meV, because
        buying frequency points means paying for time points that the fit does
        not need and whose design matrix is exponentially ill-conditioned.

        Splitting them lets the frequency quadrature converge on its own --
        pass e.g. a 26-point minimax or 40-point Gauss-Legendre frequency grid
        here.

        with_inverse=False skips the omega -> tau matrices. A caller that wants
        only chi0 on the frequency axis -- a STATIC screening, say, which is one
        point at omega = 0 -- never uses them, and building them from a single
        input point is not merely wasted: the fit is unconstrained, so it
        reports an error of order 1 and warns. That warning is then informative
        about nothing and lands in the log next to real ones.
        """
        tau_points, tau_weights = minimax_time_grid(ntau, e_min, e_max)
        tau_points, tau_weights = 0.5 * tau_points, 0.5 * tau_weights
        omega_points = np.asarray(omega_points)

        kinds = [('cosft_wt', COSINE_TW)]
        if with_inverse:
            kinds += [('cosft_tw', COSINE_WT)]
        if with_sine:
            kinds += [('sinft_wt', SINE_TW)]
            if with_inverse:
                kinds += [('sinft_tw', SINE_WT)]
        mats, errs = {}, {}
        for nm, kd in kinds:
            mats[nm], errs[nm] = minimax_transform_weights(
                kd, tau_points, omega_points, e_min, e_max,
                regularization=regularization)
        return cls(tau_points, tau_weights, omega_points, np.asarray(omega_weights),
                   method='minimax', fit_errors=errs,
                   meta={'e_min': e_min, 'e_max': e_max, 'ntau': ntau,
                         'npoints': ntau, 'nfreq': len(omega_points)},
                   **mats)

    @classmethod
    def gauss_legendre(cls, nfreq, w0=0.5):
        """Frequency axis only -- no imaginary-time partner, so no transforms.

        Kept in the same container so a frequency-only driver (the existing
        `solve_rpa_screening_df` route) can take a TimeFrequencyGrid too.
        `to_omega`/`to_tau` raise on it rather than silently returning nonsense.
        """
        omega_points, omega_weights = gauss_legendre_grid(nfreq, w0=w0)
        empty = np.empty(0)
        return cls(empty, empty, omega_points, omega_weights,
                   method='gauss_legendre', meta={'w0': w0})

    @classmethod
    def ir(cls, beta, omega_max, eps=1e-10, statistics='boson',
           n_matsubara_factor=4):
        """Finite-temperature IR basis: transforms go through basis coefficients.

        Unlike the minimax route this IS a two-sided representation -- fit at the
        sampling points, evaluate anywhere -- so `to_tau(to_omega(f))` returns f
        to the basis tolerance. The price is a temperature: Lambda = beta*omega_max.
        For a molecule at T = 0, pass a large beta; the IR size grows only like
        log(Lambda).

        statistics='boson' is the right choice for a polarizability; the
        self-energy and Green's function are 'fermion'.
        """

        basis = IRBasis(beta * omega_max, eps=eps, statistics=statistics)
        x_tau = basis.default_tau_sampling()
        n_mats = basis.default_matsubara_sampling(n_max_factor=n_matsubara_factor)

        tau_points = 0.5 * beta * (x_tau + 1.0)
        omega_points = np.abs(matsubara_frequencies(n_mats, beta, statistics))

        U_tau = basis.u_at(x_tau)               # (L, ntau) real
        U_mat = basis.uhat(n_mats)              # (L, nfreq) complex

        # uhat_l is purely real for half the l and purely imaginary for the
        # other half (Li et al. 2020 Sec. II C; boson: even l real, fermion:
        # odd l real). Split on the measured magnitudes rather than the parity
        # rule so this stays correct if the sign convention ever moves. The
        # real half carries the even-in-tau (cosine-like) sector -- the one a
        # polarizability lives in -- and the imaginary half the odd sector.
        #
        # Each function against its OWN two parts, not against an absolute cut:
        # the vanishing part is that u_l's truncation floor, which grows with l
        # and with eps (5e-12 at eps=1e-6 up to 7e-6 at 1e-12), so a cut read
        # off the largest |u| misassigns the high-l functions. One misassigned
        # function gives its sector more unknowns than there are sampling
        # points -- at beta = 100, omega_max = 2, eps = 1e-10 that was 20 on
        # 19, a rank-deficient omega -> tau fit and `duality_error` 5.8e-01.
        # The ratio is also scale free, so no threshold is left to go stale.
        real_l = np.abs(U_mat.real).max(axis=1) > np.abs(U_mat.imag).max(axis=1)
        imag_l = ~real_l

        # Each sector's omega -> tau pinv fits L_s coefficients from nfreq
        # points, so it needs nfreq >= max(L_even, L_odd), where
        # `default_matsubara_sampling` sizes for the stacked real fit's
        # 2 nfreq >= L. The two coincide while the parity split is even to
        # within one function, which the ratio test above buys, so this does
        # not fire; it is here because the failure is silent -- a rank-deficient
        # sector fit is wrong by 0.2 to 0.5 with every other diagnostic clean.
        biggest = max(int(real_l.sum()), int(imag_l.sum()))
        if biggest > len(n_mats):
            raise ValueError(
                f"TimeFrequencyGrid.ir: the {statistics} basis at "
                f"Lambda = {beta * omega_max:.4g}, eps = {eps:.0e} splits "
                f"{int(real_l.sum())}/{int(imag_l.sum())} over the even/odd "
                f"sectors, and the larger needs more than the {len(n_mats)} "
                f"Matsubara sampling points available, so its omega -> tau fit "
                f"would be underdetermined. Raise n_matsubara_factor, or loosen "
                f"eps so the basis stops before the functions whose parity the "
                f"quadrature no longer resolves.")

        def _pair(mask, part):
            """(tau->omega, omega->tau) through the coefficients of one sector."""
            if not mask.any():
                return None, None
            Ut, Um = U_tau[mask], part(U_mat[mask])
            #  f_tau = c @ Ut  =>  c = f_tau @ pinv(Ut);  f_omega = c @ Um
            to_w = (np.linalg.pinv(Ut) @ Um).T          # (nfreq, ntau)
            to_t = (np.linalg.pinv(Um) @ Ut).T          # (ntau, nfreq)
            return to_w, to_t

        cosft_wt, cosft_tw = _pair(real_l, lambda m: m.real)
        sinft_wt, sinft_tw = _pair(imag_l, lambda m: m.imag)

        # The tau-axis basis of each sector, kept because it is what makes the
        # round trip checkable: `duality_error` has to know which subspace the
        # transforms are a two-sided representation OF, and recomputing the
        # split there would put the parity test in two places.
        sector_u_tau = {'even': U_tau[real_l], 'odd': U_tau[imag_l]}

        # dtau/dx Jacobian. uhat_l(n) integrates over the DIMENSIONLESS
        # x in [-1, 1], not over tau in [0, beta], so the coefficient map above
        # is short of beta/2 relative to the int dtau e^{i omega tau} that the
        # minimax transforms implement (and that `to_omega` consumers assume).
        # Without it the IR route silently returns chi0 too small by ~beta/2 --
        # measured relative deviations 0.95 / 0.98 / 0.995 at beta = 20 / 50 /
        # 200, i.e. exactly 1 - 2/beta, and dropping to 3e-11 once applied.
        jac = 0.5 * beta
        cosft_wt = None if cosft_wt is None else jac * cosft_wt
        sinft_wt = None if sinft_wt is None else jac * sinft_wt
        cosft_tw = None if cosft_tw is None else cosft_tw / jac
        sinft_tw = None if sinft_tw is None else sinft_tw / jac

        # Matsubara sum (1/beta) sum_n in place of the T = 0 (1/2pi) int dw, so
        # a consumer contracts against omega_weights identically either way.
        omega_weights = np.full(len(omega_points), 1.0 / beta)
        tau_weights = (np.gradient(tau_points) if len(tau_points) > 1
                       else np.ones(1))

        return cls(tau_points, tau_weights, omega_points, omega_weights,
                   cosft_wt=cosft_wt, cosft_tw=cosft_tw,
                   sinft_wt=sinft_wt, sinft_tw=sinft_tw,
                   method='IR',
                   meta={'beta': beta, 'omega_max': omega_max, 'eps': eps,
                         'statistics': statistics, 'basis': basis,
                         'lambda': beta * omega_max, 'size': basis.size,
                         'n_matsubara': n_mats,
                         'n_even_sector': int(real_l.sum()),
                         'sector_u_tau': sector_u_tau})

    # ---- use --------------------------------------------------------------

    def _require(self, mat, what):
        if mat is None:
            raise ValueError(
                f"this {self.method!r} grid carries no {what} transform "
                f"(gauss_legendre is frequency-only; pass with_sine=True for "
                f"the minimax sine transforms)")
        return mat

    def to_omega(self, f_tau, parity='even'):
        """Transform along the LAST axis of f_tau, tau -> omega."""
        mat = self._require(self.cosft_wt if parity == 'even' else self.sinft_wt,
                            f'{parity}-parity tau->omega')
        return np.asarray(f_tau) @ mat.T

    def to_tau(self, f_omega, parity='even'):
        """Transform along the LAST axis of f_omega, omega -> tau."""
        mat = self._require(self.cosft_tw if parity == 'even' else self.sinft_tw,
                            f'{parity}-parity omega->tau')
        return np.asarray(f_omega) @ mat.T

    def duality_error(self, parity='even'):
        """How far this grid's forward/backward pair is from a two-sided map.

        For 'minimax' this is max|A B - I|, GreenX's own `cosft_duality_error`,
        and it is O(1) or worse -- see the module docstring.

        For 'IR' it is the round-trip error of the basis fit, measured ON THE
        SUBSPACE THE BASIS REPRESENTS: worst |u_l - u_l A^T B^T| over the
        sector's tau-axis basis, relative to max|u_l|. Check it before writing
        any algorithm that transforms both ways.

        Why IR cannot use max|A B - I|
        ------------------------------
        Because A B is not a candidate for the identity there. The IR
        transforms factor through the sector's L_s coefficients, so A B is
        (pinv(Um) Um)^T -- an orthogonal projector of rank min(L_s, nfreq).
        Whenever the sampling set is larger than the sector, which is the
        normal case, max|A B - I| >= 1 - L_s/nfreq no matter how good the grid
        is, and the raw product is ANTI-CORRELATED with the grid's health: the
        one way to make it the identity is to starve the sampling set until
        nfreq <= L_s, which is exactly where the omega -> tau fit becomes
        underdetermined. Measured over beta = 100 .. 800, omega_max = 2 .. 4,
        eps = 1e-8 .. 1e-10, the raw product reads 1.9e-14 on a grid whose
        round trip is 4.8e-01, and 9.1e-01 on one whose round trip is 8.6e-10.
        The restricted quantity returns 1.03e+00 and 2.35e-09 on that pair.

        For a boson omega_0 = 0 and every odd u_l integrates to zero there, so
        the n = 0 column is structurally zero, Um is rank deficient, and
        (pinv(Um) Um)[0, 0] = 0 puts the raw product at exactly 1.0. The point
        carries nothing for the odd sector and `pinv` is right to discard it,
        so the restricted quantity is unaffected.
        """
        A = self._require(self.cosft_wt if parity == 'even' else self.sinft_wt, parity)
        B = self._require(self.cosft_tw if parity == 'even' else self.sinft_tw, parity)
        if self.method == 'IR':
            u = self.meta['sector_u_tau'][parity]
            # f_tau = c u  ->  to_omega  ->  to_tau  =  c u (B A)^T
            return float(np.abs(u @ (B @ A).T - u).max() / np.abs(u).max())
        n = A.shape[0]
        return float(np.abs(A @ B - np.eye(n)).max())

    def roundtrip_error(self, x_probe=None):
        """Worst relative error of `to_tau(to_omega(f))` vs f, same probe for
        both backends so the two are directly comparable.

        The probe is the family the imaginary-time route actually carries: a
        particle-hole bubble is a sum of exponentials e^{-x tau} with x the
        transition energies (mirrored to tau -> beta - tau for the periodic IR
        case). Prefer this over `duality_error` when comparing METHODS, since
        it is the same probe on both and minimax has no basis to restrict to.

        Within IR, `duality_error` is the stricter of the two and the one to
        trust: it sweeps the whole represented subspace, where this sweeps a
        handful of exponentials that may sit in the part of it that survives.
        Measured at beta = 100, omega_max = 2, eps = 1e-10, where the sampling
        set was one point short of the even sector (nfreq = 19, L_even = 20):
        this returns 4.1e-08 and `duality_error` 5.8e-01, and the sector fit
        really is rank deficient. A four-exponential probe is not a basis.
        """
        if self.ntau == 0:
            raise ValueError(f"{self.method!r} grid has no imaginary-time axis")
        if x_probe is None:
            if self.method == 'IR':
                x_probe = np.array([0.1, 0.5, 1.0, 2.0]) * self.meta['omega_max']
            else:
                lo, hi = self.meta['e_min'], self.meta['e_max']
                x_probe = np.geomspace(lo, hi, 6)
        worst = 0.0
        for x in np.atleast_1d(x_probe):
            f = np.exp(-x * self.tau_points)
            if self.method == 'IR':                    # bosonic periodicity
                f = f + np.exp(-x * (self.meta['beta'] - self.tau_points))
            back = self.to_tau(self.to_omega(f, 'even'), 'even')
            worst = max(worst, float(np.abs(back - f).max() / np.abs(f).max()))
        return worst

    @property
    def ntau(self):
        return len(self.tau_points)

    @property
    def nfreq(self):
        return len(self.omega_points)

    def __repr__(self):
        bits = [f"method={self.method!r}", f"ntau={self.ntau}", f"nfreq={self.nfreq}"]
        if self.fit_errors:
            bits.append(f"max_fit_err={max(self.fit_errors.values()):.2e}")
        return f"TimeFrequencyGrid({', '.join(bits)})"
