"""Finite-temperature (Matsubara) imaginary-axis grids: the intermediate
representation, plus the thermal regularization the T=0 grids need.

Why this is needed
------------------
Every imaginary-axis grid in `grids.py` is a T = 0 construction keyed on the
smallest transition energy e_min -- the HOMO-LUMO gap. `minimax_frequency_grid`
rescales a tabulated dimensionless grid by e_min and looks the table up by the
range ratio e_max/e_min; `gap_scaled_w0` sets the Gauss-Legendre scale to half
the gap. For a metal the gap is zero and all of it degenerates, silently:

    gap > 0 but tiny  ->  the grid collapses toward omega = 0
    gap == 0          ->  ZeroDivisionError
    gap < 0 (a numerically inverted, degenerate Fermi level)
                      ->  NEGATIVE frequency points AND negative weights,
                          returned without complaint

That is the whole of the metallic frequency-grid problem: not that the grids
are inaccurate for a metal, but that they are undefined for one and do not say
so.

The finite-temperature fix
--------------------------
At temperature T the low-energy cutoff is no longer the gap, it is the
temperature. Two things follow, and this module provides both.

1. `thermal_e_min` -- the cheap fix. Replace e_min by the first fermionic
   Matsubara frequency pi/beta when that exceeds the gap. The range ratio
   e_max*beta/pi is finite for a metal, so the EXISTING minimax and
   Gauss-Legendre quadratures become well defined again with no change to the
   self-energy assembly that consumes them. This is the temperature doing what
   the gap did, and it is the same physical smearing a Fermi occupation
   introduces.

2. `IRBasis` -- the principled one. The intermediate representation
   (Shinaoka, Otsuki, Ohzeki, Yoshimi, PRB 96, 035147 (2017); sparse sampling
   from Li, Wallerberger, Chikano, Yeh, Gull, Shinaoka, PRB 101, 035144
   (2020)), the finite-T basis used for Matsubara-axis GW by Zgid and
   co-workers. It is built from the SVD of the analytic-continuation kernel

       K^F(tau, omega) = e^{-tau omega} / (1 + e^{-beta omega})

   and depends on temperature and bandwidth ONLY through the dimensionless
   Lambda = beta * omega_max. No gap appears anywhere, so nothing degenerates
   as the gap closes. The singular values fall exponentially, so a metal at
   Lambda = 10^4 still needs only a few tens of coefficients, and the sparse
   sampling points are where those coefficients must be measured.

   This also settles the order of the analytic continuation. The basis size at a given
   (Lambda, eps) is the number of independent structures the screening can
   carry at that temperature and bandwidth -- so it is the natural, non-arbitrary
   order for the greedy-Pade continuation in GW/imaginary_axis, instead of a
   hand-set node count that has no reason to be right for metallic low-energy
   structure.

Nothing here is tabulated: the basis is constructed on demand from a composite
Gauss-Legendre discretization of the kernel and a dense SVD. GreenX, which
supplies the T = 0 minimax tables this repo already uses, has no
finite-temperature grids to borrow.
"""
import warnings

import numpy as np
from scipy.special import spherical_jn

#: Matsubara statistics -> the zeta in omega_n = (2n + zeta) pi / beta.
_ZETA = {'fermion': 1, 'boson': 0}

#: Largest design-matrix condition number a Matsubara sampling set may carry
#: before `default_matsubara_sampling` adds the next basis function's peaks,
#: and the value past which adding more of them has stopped helping.
_COND_MAX = 1e6
_COND_WARN = 1e8


def thermal_e_min(beta, gap=0.0, statistics='fermion'):
    """The e_min a T = 0 quadrature should be given at temperature 1/beta.

    Returns max(gap, |first Matsubara frequency|) = max(gap, zeta' pi / beta),
    so a gapped system at low temperature keeps its own gap and a metal falls
    back on the temperature. beta is in inverse Hartree (beta = 1/kT).

    This is the minimal change that makes `minimax_frequency_grid` and
    `gauss_legendre_grid` defined for a metal: the range ratio e_max/e_min
    stays finite and the grid keeps a finite lowest frequency.
    """
    if beta <= 0:
        raise ValueError(f"beta = {beta} must be positive")
    zeta = _ZETA[statistics]
    first = (2 * 0 + zeta) * np.pi / beta if zeta else 2 * np.pi / beta
    return max(float(gap), float(first))


def beta_from_mf(mf, default=None):
    """1/sigma of a smeared mean field, or `default` when it is not smeared.

    pyscf's `scf.addons.smearing_` / `pbc.scf.addons.smearing_` leave the
    smearing width on the object as `mf.sigma` (with `mf.smearing_method`
    naming the distribution). For Fermi-Dirac that width IS kT, so beta = 1/kT
    is read straight off it -- which is what lets a caller thread the
    temperature through to the grids without the user restating it.

    Returns `default` when there is no smearing, so a gapped calculation keeps
    its T = 0 grid untouched.
    """
    sigma = getattr(mf, 'sigma', None)
    if sigma is None or not sigma > 0:
        return default
    method = str(getattr(mf, 'smearing_method', 'fermi')).lower()
    if 'fermi' not in method:
        raise ValueError(
            f"beta_from_mf: smearing_method is {method!r}, and only a "
            f"Fermi-Dirac width is a temperature. A Gaussian or "
            f"Methfessel-Paxton width is a broadening parameter with no "
            f"1/kT reading, so pass beta explicitly if you want one.")
    return 1.0 / float(sigma)


def _composite_gauss_legendre(edges, order):
    """Nodes, weights, panel centres and half-widths of a composite GL rule.

    The panel structure is kept, not just the flattened nodes: the Matsubara
    transform needs to integrate u_l against a rapidly oscillating phase, and
    that is done panel by panel in closed form (see `IRBasis.uhat`).
    """
    x, w = np.polynomial.legendre.leggauss(order)
    nodes, weights, centres, halves = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        nodes.append(mid + half * x)
        weights.append(half * w)
        centres.append(mid)
        halves.append(half)
    return (np.concatenate(nodes), np.concatenate(weights),
            np.array(centres), np.array(halves))


def _segment_scale(lambda_, n_extra):
    return max(2, int(np.ceil(np.log2(max(lambda_, 4.0)))) + n_extra)


def _omega_edges(lambda_, n_extra=0):
    """Segment edges on [-1, 1] refined toward y = 0, on the scale 1/Lambda.

    The kernel's real-frequency structure is the 1/(2 cosh(Lambda y / 2))
    factor, which turns over at |y| ~ 1/Lambda.
    """
    n_seg = _segment_scale(lambda_, n_extra)
    pos = np.concatenate([[0.0],
                          np.logspace(np.log10(1.0 / max(lambda_, 4.0)), 0.0, n_seg)])
    return np.concatenate([-pos[::-1][:-1], pos])


def _tau_edges(lambda_, n_extra=0):
    """Segment edges on [-1, 1] refined toward x = +-1, NOT toward 0.

    In imaginary time the structure sits at the ENDS: for y > 0 the kernel is
    e^{-Lambda y (x+1)/2}, a boundary layer of width ~2/(Lambda y) at x = -1,
    and its mirror at x = +1 for y < 0. Refining toward the middle instead --
    the obvious thing to do, and wrong -- leaves the basis size still climbing
    with the quadrature at Lambda >~ 10^3 rather than converging.
    """
    n_seg = _segment_scale(lambda_, n_extra)
    d = np.logspace(np.log10(1.0 / max(lambda_, 4.0)), 0.0, n_seg)
    return np.unique(np.concatenate([[-1.0], -1.0 + d, 1.0 - d[::-1], [1.0]]))


def _log_kernel(x, y, lambda_):
    """log of the symmetric fermionic kernel, evaluated stably.

    K^F(x, y) = e^{-Lambda y x / 2} / (2 cosh(Lambda y / 2)), the standard
    dimensionless form of e^{-tau omega}/(1 + e^{-beta omega}) under
    tau = beta (x+1)/2 and omega = omega_max y. cosh overflows for
    Lambda|y|/2 beyond ~700, so its log is taken as |z| + log1p(e^{-2|z|}).
    """
    z = 0.5 * lambda_ * y
    return -0.5 * lambda_ * np.outer(x, y) - (np.abs(z) + np.log1p(np.exp(-2 * np.abs(z))))


class IRBasis:
    """Intermediate representation of the finite-temperature kernel.

    Depends on temperature and bandwidth only through Lambda = beta * omega_max,
    so it is defined for a metal exactly as it is for an insulator -- there is
    no gap in the construction.

    Attributes
    ----------
    s : (L,) singular values, normalized to s[0] = 1 and truncated at `eps`.
    x, wx : the imaginary-time quadrature (x in [-1, 1], tau = beta (x+1)/2).
    y, wy : the real-frequency quadrature (y in [-1, 1], omega = omega_max y).
    u : (L, nx) basis functions on the tau axis, orthonormal in x.
    v : (L, ny) basis functions on the omega axis, orthonormal in y.
    """

    def __init__(self, lambda_, eps=1e-8, statistics='fermion', order=16,
                 n_extra=2):
        if lambda_ <= 0:
            raise ValueError(f"lambda_ = {lambda_} must be positive")
        if statistics not in _ZETA:
            raise ValueError(f"statistics={statistics!r}; expected one of {list(_ZETA)}")
        self.lambda_ = float(lambda_)
        self.eps = float(eps)
        self.statistics = statistics

        self._order = order
        self.x, self.wx, self._xc, self._xh = _composite_gauss_legendre(
            _tau_edges(lambda_, n_extra=n_extra), order)
        self.y, self.wy, _, _ = _composite_gauss_legendre(
            _omega_edges(lambda_, n_extra=n_extra), order)

        kernel = np.exp(_log_kernel(self.x, self.y, self.lambda_))
        scaled = (np.sqrt(self.wx)[:, None] * kernel * np.sqrt(self.wy)[None, :])
        U, S, Vt = np.linalg.svd(scaled, full_matrices=False)

        keep = S / S[0] > self.eps
        self.s = S[keep] / S[0]
        self._s_abs = S[keep]
        self.u = (U[:, keep] / np.sqrt(self.wx)[:, None]).T
        self.v = (Vt[keep] / np.sqrt(self.wy)[None, :])
        # Fix the sign so u_l is positive at tau -> 0, making the basis
        # reproducible run to run (SVD sign is otherwise arbitrary).
        signs = np.sign(self.u[:, 0])
        signs[signs == 0] = 1.0
        self.u *= signs[:, None]
        self.v *= signs[:, None]

    @property
    def size(self):
        return len(self.s)

    def _legendre_coefficients(self):
        """u_l expanded in Legendre polynomials on each quadrature panel."""
        if getattr(self, '_leg', None) is None:
            order = self._order
            t, w = np.polynomial.legendre.leggauss(order)
            P = np.array([np.polynomial.legendre.Legendre.basis(k)(t)
                          for k in range(order)])          # (order, order)
            u_panels = self.u.reshape(self.size, len(self._xc), order)
            # a[l, p, k] = (2k+1)/2 * sum_j w_j P_k(t_j) u_l(x_pj)
            # optimize=True: without it numpy runs its naive nested-loop kernel
            # on this four-operand contraction instead of pairing it into BLAS
            # calls. Measured 3.2x at Lambda = 1e3 and 6.7x at 1e4 -- small in
            # absolute terms (milliseconds against a 0.13 s basis build), but
            # free.
            self._leg = np.einsum('k,kj,j,lpj->lpk',
                                  (2 * np.arange(order) + 1) / 2.0, P, w,
                                  u_panels, optimize=True)
        return self._leg

    def u_at(self, x):
        """u_l at arbitrary x in [-1, 1]: (L, len(x)).

        `self.u` only holds u_l on the composite quadrature nodes, but the
        sparse tau sampling points (`default_tau_sampling`) are panel midpoints
        that are NOT quadrature nodes, so any tau-axis fit needs u off-grid.
        Evaluated from the same per-panel Legendre expansion `uhat` uses, so the
        two axes stay consistent to machine precision.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        leg = self._legendre_coefficients()               # (L, npanel, order)
        # locate each x in its panel; panels are contiguous and cover [-1, 1]
        edges = np.concatenate([self._xc - self._xh, [self._xc[-1] + self._xh[-1]]])
        panel = np.clip(np.searchsorted(edges, x, side='right') - 1,
                        0, len(self._xc) - 1)
        t = (x - self._xc[panel]) / self._xh[panel]       # (nx,) local coord
        P = np.array([np.polynomial.legendre.Legendre.basis(k)(t)
                      for k in range(self._order)])       # (order, nx)
        return np.einsum('lnk,kn->ln', leg[:, panel, :], P)

    def uhat(self, n):
        """u_l on the Matsubara axis: (L, len(n)) complex.

        uhat_l(n) = int_{-1}^{1} dx u_l(x) exp(i pi (2n + zeta) (x + 1) / 2),
        the dimensionless form of int_0^beta dtau u_l(tau) e^{i omega_n tau}.

        Evaluated panel by panel in CLOSED FORM, via
        int_{-1}^{1} P_k(t) e^{izt} dt = 2 i^k j_k(z). Quadrature on the
        composite grid is not an option here: the phase has period 2/(2n+zeta)
        in x, so by |n| ~ 50 it oscillates many times inside the wide central
        panels and a Gauss-Legendre sum over them returns the wrong sign.
        """
        n = np.atleast_1d(np.asarray(n, dtype=float))
        zeta = _ZETA[self.statistics]
        a = np.pi * (2 * n + zeta) / 2.0                    # (nn,)
        leg = self._legendre_coefficients()                 # (L, npanel, order)
        order = self._order

        z = np.multiply.outer(a, self._xh)                  # (nn, npanel)
        jl = np.array([spherical_jn(k, z) for k in range(order)])   # (order, nn, npanel)
        ik = 1j ** np.arange(order)
        # panel integral = h_p e^{i a c_p} sum_k a_lpk 2 i^k j_k(a h_p)
        weight = 2.0 * ik[:, None, None] * jl * self._xh[None, None, :]
        phase = np.exp(1j * np.multiply.outer(a, self._xc))  # (nn, npanel)
        # optimize=True is worth 3.3x at Lambda = 1e4; at 1e3 the path search
        # costs slightly more than it saves, and the sizes that matter here grow
        # with Lambda.
        return np.exp(1j * a)[None, :] * np.einsum(
            'lpk,knp,np->ln', leg, weight, phase, optimize=True)

    def default_tau_sampling(self):
        """Sparse sampling points in x, where tau = beta (x + 1) / 2.

        Li et al. (2020) Sec. II B: the midpoints of the grid formed by the
        L-1 roots of the highest retained u_{L-1} together with the two
        boundary points. Their roots, not the roots of the next basis function
        (the Chebyshev recipe), because the IR basis is numerical and u_L is
        not available.
        """
        top = self.u[-1]
        cross = np.where(np.sign(top[:-1]) != np.sign(top[1:]))[0]
        # linear interpolation of each root between the bracketing nodes
        x0, x1 = self.x[cross], self.x[cross + 1]
        y0, y1 = top[cross], top[cross + 1]
        roots = x0 - y0 * (x1 - x0) / (y1 - y0)
        edges = np.concatenate([[-1.0], roots, [1.0]])
        return 0.5 * (edges[:-1] + edges[1:])

    @staticmethod
    def _sampling_part(row):
        """The non-vanishing part of one uhat_l row, chosen by MEASURED size.

        uhat_l is purely real for even l and purely imaginary for odd l (Li et
        al. 2020 Sec. II C), so which part carries the function ALTERNATES with
        l. It is not fixed by the statistics, and reading the other part reads
        the truncation floor: measured |Re| = 5e-12 .. 7e-6 against
        |Im| = 0.23 .. 0.47 (and the reverse), so the "sign changes" are then
        round-off. That is not a small effect -- it puts the sampling points in
        the far tail where uhat carries nothing, and it took the design matrix
        to cond 7e7 .. 2e10 in 12 of 24 bases over lambda = 100 .. 1e4,
        eps = 1e-6 .. 1e-12, for round-trip errors of 1e-8 .. 4e-5 on O(1)
        coefficients.

        Choosing by magnitude needs no parity rule and so cannot fall out of
        step with one.
        """
        return (row.real if np.abs(row.real).max() > np.abs(row.imag).max()
                else row.imag)

    def _peaks_per_group(self, n, row):
        """The n of largest |uhat_l| in each interval between its sign changes.

        Li et al. (2020) Sec. II C: the sign changes of uhat_l partition the
        Matsubara axis into groups, and the sampling point of each group is the
        n that MAXIMIZES |uhat_l| there. One point per group, at the peak --
        not at the zeros (which give near-null rows, cond ~1e13) and not every
        local maximum on the axis (which sweeps up thousands of far-tail peaks
        of vanishing magnitude).
        """
        part = self._sampling_part(row)
        groups = np.concatenate([[0], np.cumsum(np.sign(part[:-1])
                                                != np.sign(part[1:]))])
        return [int(n[groups == g][np.argmax(np.abs(part[groups == g]))])
                for g in range(groups[-1] + 1)]

    def sampling_is_determined(self, n, real=True, cond_max=_COND_MAX):
        """Whether a fit at `n` pins all `size` coefficients: (ok, nrows, cond).

        `real` selects the convention `fit_matsubara` will use, and it sets the
        ROW COUNT rather than merely the arithmetic. real=True stacks the real
        and imaginary parts of each equation, so len(n) points give 2 len(n)
        rows; a complex fit gives len(n).

        Check the row count BEFORE the conditioning, which is the trap here: on
        a WIDE matrix (fewer rows than `size`) numpy's `cond` is the ratio over
        only the min(nrows, size) singular values, all of which are healthy, so
        an underdetermined fit reports cond ~1e1 while returning coefficients
        wrong by O(1). Measured cond 1.3e1 at lambda = 1600, eps = 1e-12, on a
        fit whose worst coefficient was out by 0.89.
        """
        n = np.atleast_1d(np.asarray(n))
        nrows = (2 if real else 1) * len(n)
        if nrows < self.size:
            return False, nrows, np.inf
        cond = self.condition_number(n, real=real)
        return bool(cond < cond_max), nrows, float(cond)

    def default_matsubara_sampling(self, n_max_factor=4, positive_only=True,
                                   real=None, cond_max=_COND_MAX):
        """Sparse sampling indices n on the Matsubara axis.

        One point per sign change of the highest retained uhat_{L-1} (Li,
        Wallerberger, Chikano, Yeh, Gull, Shinaoka, PRB 101, 035144 (2020)),
        AUGMENTED with the sign-change groups of uhat_{L-2}, uhat_{L-3}, ...
        until the set determines all `size` coefficients and is conditioned
        (`sampling_is_determined`). Both halves of that matter:

        * The paper's own count is MARGINAL BY CONSTRUCTION for a one-sided
          set. The number of groups of uhat_{L-1} is ~L, `positive_only` keeps
          half, and a real-coefficient fit needs 2 n_pos >= L -- so the
          criterion sits exactly on 2 (L/2) = L and tips either way depending
          on where the crossings land. Measured over lambda = 100 .. 1600 and
          eps = 1e-6 .. 1e-12, the unaugmented one-sided set left 10 of 20
          bases underdetermined, returning coefficients wrong by 0.05 .. 0.89
          with no warning.
        * The augmentation is not the same thing as more points from the same
          function. The negative half is genuinely redundant -- uhat_l(-n-1) =
          conj(uhat_l(n)) makes those rows duplicates up to a sign -- so
          positive_only=False does NOT repair a real fit: measured rank 54 for
          both halves where `size` is 57, and an IDENTICAL error of 0.46. Only
          new n carry new rows, and the next basis function's crossings
          interleave with this one's, which is why one extra round suffices in
          almost every case (three at lambda = 1e4, eps = 1e-12).

        positive_only keeps the n >= 0 half, which for the real-coefficient fit
        `fit_matsubara` performs by default halves the number of self-energy
        evaluations. Pass positive_only=False for the paper's symmetric set,
        which a complex-valued fit needs.

        `real` defaults to `positive_only`, pairing each set with the fit it is
        meant for -- one-sided with the real stacking, symmetric with the
        complex solve, which is the pairing the two docstrings describe. Pass
        it explicitly to size a set for the other convention. The symmetric set
        is marginal for the COMPLEX fit in exactly the same way (its ~L points
        against L unknowns), so this augmentation is what makes the
        `positive_only=False` path sound too: measured 8.6e-01 -> 9.6e-13.
        """
        real = positive_only if real is None else real
        n_max = int(n_max_factor * max(self.size, 4) + self.lambda_ / (2 * np.pi))
        n = np.arange(-n_max, n_max + 1)
        rows = self.uhat(n)

        picked = []
        for l in range(self.size - 1, -1, -1):
            picked = sorted(set(picked) | set(self._peaks_per_group(n, rows[l])))
            out = np.array([p for p in picked if p >= 0]) if positive_only \
                else np.array(picked)
            ok, nrows, cond = self.sampling_is_determined(out, real=real,
                                                          cond_max=cond_max)
            if ok:
                return out
            if cond > _COND_WARN and nrows >= self.size:
                break
        warnings.warn(
            f'Matsubara sampling for lambda = {self.lambda_:.4g}, '
            f'eps = {self.eps:.0e} did not become determined: {len(out)} '
            f'points give {nrows} rows for {self.size} coefficients at '
            f'condition {cond:.2e}. Raise n_max_factor, or loosen eps so the '
            f'basis stops before the functions the quadrature cannot resolve.',
            RuntimeWarning, stacklevel=2)
        return out

    def fit_matsubara(self, n, values, rcond=1e-12, real=True):
        """Least-squares IR coefficients from values sampled at Matsubara n.

        real=True (default) solves for REAL coefficients by stacking the real
        and imaginary parts of the system. That is not a convenience: uhat_l is
        purely imaginary for even l and purely real for odd l, so the columns of
        the complex fitting matrix are alternately real and imaginary. A
        complex least squares over a one-sided (n >= 0) sampling set is then
        rank deficient -- measured cond 1.3e15, against 6.6e2 for the same
        points stacked. Any G(tau) that is real, which is every Green's
        function and self-energy here, has real IR coefficients.

        Pass real=False only for genuinely complex-in-tau data, and then give a
        sampling set symmetric under n -> -n-1. Periodic data is the case that
        needs it: Wt^q is complex Hermitian at every q != 0, so its
        tau-dependence is genuinely complex and the real path would discard the
        imaginary part rather than fail. Measured there: condition 13 at
        beta = 200, residual 5e-15 -- the rank deficiency above is specific to
        ONE-SIDED sampling, not to complex fits.

        The coefficients are DIMENSIONLESS, and mixing axes needs the Jacobian
        ----------------------------------------------------------------------
        `uhat` integrates over x in [-1, 1], not over tau in [0, beta], so the
        coefficients this returns are short of the physical ones by
        dtau/dx = beta/2. Two consequences, and only the second bites:

          * fit_matsubara -> evaluate_matsubara is SELF-CONSISTENT. Both sides
            use uhat, the factor cancels, and no correction is needed. This is
            what the round-trip test exercises.
          * fit_matsubara -> `u_at` is NOT. Transporting a Matsubara fit onto
            the tau axis overshoots by exactly beta/2 unless you apply (2/beta)
            yourself, as TimeFrequencyGrid.ir does
            internally (cosft_tw / jac).

        Measured beta/2 to 1.6e-15 at four (beta, wmax) and pinned by
        test_matsubara_ir.test_tau_transport_jacobian. Worth stating rather
        than leaving to be rediscovered: a clean constant reads as a convention
        rather than a bug, and no convergence study can see it -- the error is
        beta-dependent but not beta-CONVERGENT, so scanning beta shows a stable
        wrong number. It cost a self-energy calculation a factor of 200 at
        beta = 400, caught only by an exact reduction against a T = 0 route.
        """
        A = self.uhat(n).T                                # (npoints, L)
        values = np.asarray(values)
        if not real:
            coeffs, *_ = np.linalg.lstsq(A, values, rcond=rcond)
            return coeffs
        stacked = np.vstack([A.real, A.imag])
        rhs = np.concatenate([values.real, values.imag])
        coeffs, *_ = np.linalg.lstsq(stacked, rhs, rcond=rcond)
        return coeffs

    def evaluate_matsubara(self, coeffs, n):
        return np.asarray(coeffs) @ self.uhat(n)

    def condition_number(self, n, real=True):
        """Condition number of the Matsubara fitting matrix at sampling set n,
        for the same real/complex convention `fit_matsubara` uses."""
        A = self.uhat(n).T
        return np.linalg.cond(np.vstack([A.real, A.imag]) if real else A)

    def __repr__(self):
        return (f"IRBasis(lambda_={self.lambda_:.4g}, eps={self.eps:.1e}, "
                f"statistics={self.statistics!r}, size={self.size})")


def matsubara_frequencies(n, beta, statistics='fermion'):
    """omega_n = (2n + zeta) pi / beta for Matsubara indices n."""
    zeta = _ZETA[statistics]
    return (2 * np.asarray(n) + zeta) * np.pi / beta


def ir_continuation_order(beta, wmax, eps=1e-8, statistics='fermion'):
    """A non-arbitrary Pade/multipole order for a system at (beta, wmax).

    The IR size at Lambda = beta*wmax is the number of independent structures
    the imaginary-axis data can resolve at that temperature and bandwidth, so
    asking a continuation for more nodes than that is asking it to fit noise --
    which is exactly how metallic low-energy structure is usually mishandled.
    """
    return IRBasis(beta * wmax, eps=eps, statistics=statistics).size


def self_energy_range(mo_energy, mu, screening_max):
    """Spectral range of Sigma_c, for sizing a continuation at (beta, wmax).

    Use this, NOT the screening range, whenever the quantity being continued
    or discretised is the self-energy. Sigma = -G Wt is a PRODUCT, so its
    poles sit at eps_m +/- omega_s and its range is the SUM of the orbital
    range measured from mu and the screening range. Passing the screening
    range alone is the natural mistake -- it is the range of the object the
    frequency grid was built for -- and it silently under-resolves Sigma.

    Two independent measurements of the same 1.6x ratio: on diamond gth-szv
    2x2x1 an imaginary-TIME route found Sigma spanning
    0.1275-9.1685 Ha against 0.0850-5.6976 for the screening, where a molecular
    ntau that looked converged was still 3 meV out; on this frequency route the
    same ratio costs 4 Pade nodes of 40 at beta = 100. The node count grows
    only logarithmically in Lambda, which is why the error is small -- and also
    why it never announces itself.
    """
    return float(np.abs(np.asarray(mo_energy) - mu).max() + screening_max)
