"""Quasiparticle peaks of a spectral function A(omega) on a real frequency axis.

`analyze_peaks` finds the peaks (and dips) of one A(omega) and integrates
their weights; `analyze_orbital_peaks` does this per (degenerate group of)
orbital, from the diagonal terms A_i. Only numpy/scipy on the arrays: how A
was built (in scGW, from a Pade fit of Sigma) does not enter.
"""
import numpy as np
import scipy.integrate
import scipy.signal


def analyze_peaks(w_real, A, eta, w_step, min_qp_weight=0.7, verbose=True):
    """Quasiparticle peaks of the spectral function A(omega).

    Peaks are searched for in A and in -A: a quasiparticle pole can come out
    of the Pade continuation as a dip (e.g. the Ne HOMO at eta = 1e-3), which
    a search for maxima alone misses. Between neighbouring peaks the boundary
    is the point where |A| is smallest (the zero crossing between a peak and
    a dip), and the weight of a peak is the integral of A between its two
    boundaries. Peaks with |weight| < min_qp_weight (satellites, or spurious
    peaks from a wrong eta) are dropped.

    A is a trace over orbitals, so a degenerate level has weight ~ Z times
    its degeneracy.

    Returns (qpes, weights, negative) for the kept peaks: the positions,
    refined with a parabola through the extremum and its two neighbours,
    their weights, and whether each is a dip (negative peak) in A.
    verbose=False skips the printout.
    """
    # A pole of weight Z, broadened by eta, peaks at Z / (pi * eta), so the
    # threshold is set as a minimum weight rather than a fixed number. It is
    # applied to the height and to the prominence: a ripple on the flank of
    # a large peak clears the height on that peak's tail alone, and would
    # otherwise get a boundary of its own and cut the large peak's weight
    # (possibly below min_qp_weight, dropping a real quasiparticle).
    min_peak_weight = 0.05
    threshold = min_peak_weight / (np.pi * eta)
    peaks_pos, _ = scipy.signal.find_peaks(A, height=threshold, prominence=threshold)
    peaks_neg, _ = scipy.signal.find_peaks(-A, height=threshold, prominence=threshold)
    peaks = np.sort(np.concatenate([peaks_pos, peaks_neg]))
    negative = np.isin(peaks, peaks_neg)

    if len(peaks) == 0:
        if verbose:
            print(f'\nNo peaks of A found above {min_peak_weight} weight (eta = {eta}).')
        return np.empty(0), np.empty(0), np.empty(0, dtype=bool)

    # Boundaries at the smallest |A| between neighbouring peaks, and the data
    # edges outside the first and last peak; the weights then add up to the
    # integral of A over the whole axis.
    bounds = ([0] + [p0 + np.argmin(np.abs(A[p0:p1])) for p0, p1 in zip(peaks[:-1], peaks[1:])]
              + [len(A) - 1])
    weights = np.array([scipy.integrate.trapezoid(A[l:r + 1], w_real[l:r + 1])
                        for l, r in zip(bounds[:-1], bounds[1:])])

    # Refine each peak position with a parabola through the extremum and its
    # two neighbours, instead of taking the grid point itself (the same
    # formula holds for a minimum). A flat top (zero curvature, which
    # find_peaks can report) keeps the grid point, and the shift is clipped
    # to half a step: the extremum lies between the two midpoints.
    qpes = np.empty(len(peaks))
    for i, peak in enumerate(peaks):
        a_left, a_mid, a_right = A[peak - 1], A[peak], A[peak + 1]
        curvature = a_left - 2 * a_mid + a_right
        shift = 0.5 * (a_left - a_right) / curvature if curvature != 0 else 0.0
        qpes[i] = w_real[peak] + np.clip(shift, -0.5, 0.5) * w_step

    keep = np.abs(weights) >= min_qp_weight

    if verbose:
        print(f'\nPeaks of A (|weight| >= {min_qp_weight} kept):')
        for qpe, weight, neg, kept in zip(qpes, weights, negative, keep):
            print(f'  w = {qpe:.6f} Ha  weight = {weight:7.3f}'
                  f'{"  (negative peak)" if neg else ""}{"" if kept else "  dropped"}')

    return qpes[keep], weights[keep], negative[keep]


def analyze_orbital_peaks(w_real, A_diag, eps_F, mu, eta, w_step, min_qp_weight=0.7,
                          degen_tol=1e-4):
    """Quasiparticle peak per orbital, from the diagonal terms A_i of A.

    A_diag is (nfreq, nmo), A_i = -Im G_ii / pi in the eigenbasis of F, with
    eps_F its (ascending) eigenvalues. Orbitals whose eps_F lie within
    degen_tol of each other (chained) form one group, and the peaks of the
    group's summed A_i are searched for with `analyze_peaks`: within a
    degenerate subspace the eigenvectors are arbitrary, and Sigma can split
    one pole over its A_i, but their sum does not depend on that choice.

    The quasiparticle peak of a group is its peak of largest |weight|, kept
    if |weight| / degeneracy >= min_qp_weight; the other peaks (weight >=
    0.05, see `analyze_peaks`) are its satellites. Weights are integrals over
    the window, so for an orbital near or past its edge they are cut.

    Returns (qpes, info): qpes (nmo,) the quasiparticle energy of every
    orbital (that of its group; nan if none is kept), and info a dict of
    (nmo,) arrays 'weights' (the group's quasiparticle weight, as with
    analysis='sum'), 'Z' (weights / degeneracy), 'negative' (a dip),
    'degeneracy', 'group' (group index), plus 'satellites', one list of
    (w, weight) per group.
    """
    nmo = len(eps_F)
    # Groups: consecutive eps_F (sorted) closer than degen_tol.
    group = np.concatenate([[0], np.cumsum(np.diff(eps_F) > degen_tol)])
    n_groups = group[-1] + 1

    qpes = np.full(nmo, np.nan)
    weights = np.full(nmo, np.nan)
    w_best = np.full(nmo, np.nan)       # also for dropped groups, for the printout
    negative = np.zeros(nmo, dtype=bool)
    degeneracy = np.bincount(group)[group]
    satellites = []

    for g in range(n_groups):
        members = np.flatnonzero(group == g)
        q, wt, neg = analyze_peaks(w_real, A_diag[:, members].sum(axis=1), eta, w_step,
                                   min_qp_weight=0.0, verbose=False)
        if len(q) == 0:
            satellites.append([])
            continue
        best = np.argmax(np.abs(wt))
        satellites.append([(q[k], wt[k]) for k in range(len(q)) if k != best])
        w_best[members] = q[best]
        weights[members] = wt[best]
        negative[members] = neg[best]
        if abs(wt[best]) / len(members) >= min_qp_weight:
            qpes[members] = q[best]

    Z = weights / degeneracy

    print(f'\nQuasiparticle peak per orbital group (eigenbasis of F, |Z| >= {min_qp_weight} '
          f'kept; Z = weight / degeneracy):')
    print(f'  {"orbitals":>9} {"eps_F (Ha)":>11} {"w_qp (Ha)":>11} {"Z":>7}  satellites')
    for g in range(n_groups):
        members = np.flatnonzero(group == g)
        i = members[0]
        if np.isnan(weights[i]):
            continue    # no peak in the window
        orbs = f'{i}' if len(members) == 1 else f'{i}-{members[-1]}'
        status = '' if not np.isnan(qpes[i]) else '  dropped'
        sats = ', '.join(f'{w:.4f} [{wt:.4f}]' for w, wt in satellites[g])
        print(f'  {orbs:>9} {eps_F[i]:11.5f} {w_best[i]:11.6f} {Z[i]:7.3f}'
              f'{"  (negative peak)" if negative[i] else ""}{status}  {sats}')

    # The orbital order of F (occupied below mu) should survive Sigma: a
    # quasiparticle on the other side of mu than its orbital means levels
    # have crossed, or a wrong peak was taken.
    crossed = np.flatnonzero(~np.isnan(qpes) & ((eps_F < mu) != (qpes < mu)))
    if len(crossed):
        print(f'WARNING: orbitals {crossed.tolist()} have their quasiparticle peak on the '
              f'other side of mu than eps_F.')

    return qpes, {'weights': weights, 'Z': Z, 'negative': negative,
                  'degeneracy': degeneracy, 'group': group, 'satellites': satellites}
