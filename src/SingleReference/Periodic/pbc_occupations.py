"""Occupations and the occupation-weighted transition space: (f_m - f_n) and
smearing.

An INTEGER occupied count -- `PBCDFIntegrals.nocc` is
`(mo_occ > 1e-8).sum(axis=1)` -- gives the rectangle nocc[ki] x nvirt[ki] of
transitions. For a metal that is
the wrong object, and not by a little: on a Li monolayer with 0.5 eV Fermi
smearing the collapsed count comes out as 8, 9, 10 or 12 depending on the
k-point, so the rectangle is not even the same shape at every k, and every
transition INSIDE the partially filled band -- which is where a metal's
low-energy response lives -- is either counted with weight one or dropped
entirely.

The decision this module rests on, settled by measurement
--------------------------------------------------------
A metal forces a choice between a Hermitian formulation with an explicit
regularization and a non-Hermitian Casida: with fractional occupations the
sqrt(f_i - f_a) factor approaches zero and can change sign within the
partially filled band.

The sign change is real, and it is a property of keeping a FIXED occ/virt
partition rather than of fractional occupations as such. Take each transition
instead as an ORDERED PAIR (m, n) with f_m > f_n and the weight is positive by
construction. What then decides the formulation is whether the ENERGY
difference is positive on that same set, because the Casida squaring trick
needs A - B = diag(e_n - e_m) positive definite. It is, provided

    f is a monotone non-increasing function of e ,

which Fermi-Dirac occupations at a single chemical potential satisfy
identically. Measured on a Li monolayer (gth-dzvp, 28 orbitals, dimension=2)
at 3x3 and 4x4 with sigma = 0.02 and 0.005 Ha: **zero** monotonicity
violations out of 447 steps, and **zero** of 67028 ordered pairs across all
k-pairs with e_n <= e_m. So the Hermitian route survives, the squaring trick
survives with it, and the regularization needed is a threshold on the
OCCUPATION difference rather than on the energy.

That threshold is close to free. Discarding pairs with f_m - f_n below 1e-4
removes about 45% of the pairs and 0.0008% of the total weight, so it is a
sparsity mechanism as much as a safeguard: the occupation-weighted transition
space, which is nmo^2 rather than nocc*nvirt before pruning, comes out no
larger than the integer one after it.

`check_occupation_monotonic` verifies the assumption rather than trusting it,
because everything else here depends on it and a violation would silently
produce a negative A - B, which the solver would then clip into nonsense.

Conventions
-----------
pyscf's `mo_occ` runs 0..2 for a restricted calculation, i.e. it already
carries the spin factor, so no separate factor of two appears below. With
occupations of 0 or 2 every pair is an occupied-virtual one with weight 2, so a
gapped system needs no separate integer route.

This module holds the occupations and the transition space; the response built
on it is in pbc_casida (Pi0, the Casida blocks, W) and the self-energy in
pbc_self_energy.
"""
import numpy as np

from src.Base.numerics.constants import (INTEGER_OCCUPATION_TOL,
                                         PBC_OCCUPATION_THRESHOLD)



def fermi_level(mo_energy, sigma, nelectron, nkpts=None, method='fermi'):
    """The chemical potential that puts `nelectron` in the cell, by root-find.

        (1/N_k) sum_k sum_m f((eps_m^k - mu)/sigma) = nelectron

    THE ONLY DEFENSIBLE mu FOR A METAL, and worth having because the two
    obvious shortcuts are both wrong, measured on a Li monolayer against this:

      * `mf.get_fermi()` is off by **0.1 to 1.5 eV** at every point tested. It
        is not the smearing chemical potential and must not be used as one.
      * `chemical_potential`'s heuristic -- the mean of the partially
        occupied eigenvalues, falling back to midgap -- is excellent while
        NOTHING is partially occupied (<= 7 meV, since a Fermi-Dirac mu does
        sit in the gap there) and degrades precisely in the regime it exists
        for: **+97 meV** on a 4x4 mesh with 8 partial states, **+1360 meV** at
        sigma = 0.05.

    So the heuristic is not a cheap version of this; it is a different quantity
    that happens to agree when the system is not really metallic. Verified the
    other way round too: Fermi-Dirac at the mu returned here reproduces pyscf's
    own `mo_occ` to 8.7e-13.
    """
    from scipy.optimize import brentq

    e = np.asarray(mo_energy, dtype=float)
    nk = e.shape[0] if (nkpts is None and e.ndim > 1) else (nkpts or 1)
    if method != 'fermi':
        raise NotImplementedError(
            f"smearing method {method!r}: only Fermi-Dirac is inverted here. A "
            f"Gaussian or Methfessel-Paxton occupation is not a temperature "
            f"and its 'mu' is a different object -- see matsubara.beta_from_mf, "
            f"which refuses the same case.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    def excess(mu):
        return (2.0 / (1.0 + np.exp((e - mu) / sigma))).sum() / nk - nelectron

    lo, hi = float(e.min()) - 50.0 * sigma, float(e.max()) + 50.0 * sigma
    if excess(lo) > 0 or excess(hi) < 0:
        raise ValueError(
            f"no chemical potential in [{lo:.4f}, {hi:.4f}] gives "
            f"{nelectron} electrons (range spans {excess(lo) + nelectron:.3f} "
            f"to {excess(hi) + nelectron:.3f}) -- the orbital set cannot hold "
            f"the requested count.")
    return float(brentq(excess, lo, hi, xtol=1e-14, rtol=1e-15))


def fermi_level_from_mf(mf):
    """`fermi_level` for a Fermi-smeared mean field, or None if not smeared.

    Refuses a Gaussian width for the same reason `matsubara.beta_from_mf`
    does: a broadening parameter is not a temperature, so inverting it as one
    would return a number with no thermodynamic meaning.
    """
    sigma = getattr(mf, 'sigma', None)
    if not sigma:
        return None
    method = getattr(mf, 'smearing_method', 'fermi')
    if method != 'fermi':
        raise NotImplementedError(
            f"mf carries smearing_method={method!r}; only Fermi-Dirac defines "
            f"the chemical potential this inverts.")
    e = np.asarray(mf.mo_energy)
    return fermi_level(e, sigma, mf.cell.nelectron, nkpts=e.shape[0])


def is_integer_occupation(mo_occ, tol=INTEGER_OCCUPATION_TOL):
    """True when every occupation is (numerically) 0 or 2."""
    f = np.asarray(mo_occ, dtype=float)
    return bool(np.all((np.abs(f) < tol) | (np.abs(f - 2.0) < tol)))


def check_occupation_monotonic(mo_energy, mo_occ, tol=1e-9, raise_on_fail=True):
    """Verify f is a non-increasing function of e over the whole k-set.

    This is the assumption that lets the Hermitian formulation work: ordering a
    transition so its occupation difference is positive must also make its
    energy difference positive, and that is exactly monotonicity. Fermi-Dirac
    occupations at one chemical potential satisfy it by construction, so a
    violation means the occupations did not come from one -- a hand-set
    configuration, a constrained or excited-state occupation, or two k-points
    converged to different chemical potentials.

    Returns the list of violating (energy, energy, occ, occ) tuples; raises by
    default, because the consequence downstream is a negative entry in
    A - B that the Casida solver would clip rather than reject.
    """
    e = np.asarray(mo_energy, dtype=float).ravel()
    f = np.asarray(mo_occ, dtype=float).ravel()
    order = np.argsort(e, kind='stable')
    es, fs = e[order], f[order]
    rises = np.where(np.diff(fs) > tol)[0]
    bad = [(float(es[i]), float(es[i + 1]), float(fs[i]), float(fs[i + 1]))
           for i in rises]
    if bad and raise_on_fail:
        e0, e1, f0, f1 = bad[int(np.argmax([b[3] - b[2] for b in bad]))]
        raise ValueError(
            f"occupations are not a monotone function of orbital energy: "
            f"{len(bad)} rising step(s), the worst at e = {e0:.6f} -> {e1:.6f} "
            f"with f = {f0:.6f} -> {f1:.6f}. The occupation-weighted Casida "
            f"construction here orders each transition so that f_m > f_n and "
            f"relies on that implying e_n > e_m, which is what keeps A - B "
            f"positive definite and the squaring trick valid. Occupations from "
            f"Fermi-Dirac smearing at a single chemical potential always "
            f"satisfy it; these do not, so they need the non-Hermitian "
            f"formulation instead.")
    return bad


def transition_pairs(mo_energy, mo_occ, ki, ka,
                     threshold=PBC_OCCUPATION_THRESHOLD):
    """Ordered transitions (m @ ki -> n @ ka) with f_m > f_n, above threshold.

    Returns (m_idx, n_idx, weight, gap) with weight = f_m - f_n > 0 and
    gap = e_n - e_m. Every pair of orbitals appears at most once: the ordering
    is what makes the weight positive, and the reversed pair carries the same
    physics with both signs flipped.
    """
    e = np.asarray(mo_energy, dtype=float)
    f = np.asarray(mo_occ, dtype=float)
    w = f[ki][:, None] - f[ka][None, :]
    gap = e[ka][None, :] - e[ki][:, None]
    keep = w > threshold
    m_idx, n_idx = np.nonzero(keep)
    return m_idx, n_idx, w[keep], gap[keep]


def transition_layout(dfints, kconserv, mo_energy=None, mo_occ=None,
                      threshold=PBC_OCCUPATION_THRESHOLD):
    """Per-k transition lists and flat offsets for one momentum transfer.

    The space is ragged -- a different number of transitions survives at each k --
    which is the point: with a metal the integer nocc is not even the same at
    every k-point.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    if mo_occ is None:
        mo_occ = dfints.mo_occ
    mo_energy = np.asarray(mo_energy, dtype=float)
    mo_occ = np.asarray(mo_occ, dtype=float)

    pairs, sizes = [], []
    for ki in range(dfints.nkpts):
        p = transition_pairs(mo_energy, mo_occ, ki, kconserv[ki],
                             threshold=threshold)
        pairs.append(p)
        sizes.append(len(p[0]))
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    return pairs, offsets, int(offsets[-1])


def stacked_Lw(dfints, kconserv, mo_energy=None, mo_occ=None,
               threshold=PBC_OCCUPATION_THRESHOLD, sqrt_weight=True):
    """Mtilde[L, t] = sqrt(f_m - f_n) L^{ki,ka}[L, m, n], stacked over k.

    Folding sqrt(w) into the three-center factor is what makes the whole
    occupation weighting invisible downstream: Mtilde^H Mtilde is then the
    weight-scaled coupling that the Casida blocks want, and Mtilde with the
    plain weight is what the polarizability wants. `sqrt_weight=False` folds in
    w rather than sqrt(w), for the latter.

    Returns (Mtilde, pairs, offsets, N, weight, gap) with `weight` and `gap`
    flattened over the same layout.
    """
    pairs, offsets, N = transition_layout(dfints, kconserv, mo_energy, mo_occ,
                                          threshold)
    naux = dfints.Lblock(0, kconserv[0]).shape[0]
    M = np.zeros((naux, N), dtype=np.complex128)
    weight = np.zeros(N)
    gap = np.zeros(N)
    for ki in range(dfints.nkpts):
        m_idx, n_idx, w, g = pairs[ki]
        if not len(m_idx):
            continue
        L = dfints.Lblock(ki, kconserv[ki])           # (naux, nmo, nmo)
        scale = np.sqrt(w) if sqrt_weight else w
        M[:, offsets[ki]:offsets[ki + 1]] = L[:, m_idx, n_idx] * scale[None, :]
        weight[offsets[ki]:offsets[ki + 1]] = w
        gap[offsets[ki]:offsets[ki + 1]] = g
    return M, pairs, offsets, N, weight, gap


def split_transition_window(mo_energy, occupied):
    """(e_min, e_max) of the transitions across a FIXED occupied/virtual split,
    over all k-points: smallest virtual minus largest occupied (the
    fundamental, possibly indirect, gap) and largest virtual minus smallest
    occupied. `occupied` is a boolean mask shaped like mo_energy, or an
    integer count applied at every k-point. Defined for a gap; for a metal
    use `transition_window`.
    """
    e = np.asarray(mo_energy)
    if np.ndim(occupied) == 0:
        occupied = np.arange(e.shape[-1]) < occupied
    occ = np.broadcast_to(occupied, e.shape)
    e_occ, e_vir = e[occ], e[~occ]
    return e_vir.min() - e_occ.max(), e_vir.max() - e_occ.min()


def transition_window(mo_energy, mo_occ, threshold=PBC_OCCUPATION_THRESHOLD):
    """(e_min, e_max) over the OCCUPATION-WEIGHTED transition set, all k-pairs.

    The window an imaginary-frequency grid should be scaled by. Every grid in
    `Base/utils/grids/grids.py` is keyed on e_min, and without `mo_occ` that comes
    from `split_transition_window`, which applies ONE integer occupied count
    at every k-point. For a metal that boundary is an artifact, and it fails in
    two different ways depending on how ragged the collapse happens to be.
    Measured on a Li monolayer at 3x3 with sigma = 0.02 Ha:

      gth-dzvp: per-k nocc is [8, 9, 11], so one global count gives
        e_min = -0.0418 Ha -- NEGATIVE, and grids.py raises. This is the loud
        case: the run stops until a beta is supplied.
      gth-szv:  per-k nocc is [3, 4], and the same construction gives
        e_min = +0.0888 Ha -- positive, plausible, and fictitious, against an
        honest 0.0673. Nothing raises. The grid is simply mis-scaled, silently.

    So the guard is necessary and not sufficient, and which failure a given
    system shows is an accident of its basis. This function returns the honest
    window instead, over the occupation-weighted transition set. At finite
    temperature `matsubara.thermal_e_min` then floors it at pi/beta, which for
    the dzvp case is what actually sets the scale: the honest 0.0019 Ha lies
    well below pi/beta = 0.0628 Ha.
    """
    e = np.asarray(mo_energy, dtype=float)
    f = np.asarray(mo_occ, dtype=float)
    nk = e.shape[0]
    lo, hi = np.inf, 0.0
    for k1 in range(nk):
        for k2 in range(nk):
            _, _, _, gap = transition_pairs(e, f, k1, k2, threshold=threshold)
            if gap.size:
                lo = min(lo, float(gap.min()))
                hi = max(hi, float(gap.max()))
    if not np.isfinite(lo):
        raise ValueError(
            "no transition carries occupation weight above the threshold, so "
            "there is no window to scale a frequency grid by. Either the "
            "system has no partially occupied states and every band is full, "
            "or the threshold is too large.")
    return lo, hi
