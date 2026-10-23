"""Occupation-weighted response for a metal: (f_m - f_n) and smearing.

Everything in Periodic/ builds its transition space from an INTEGER occupied
count -- `PBCDFIntegrals.nocc` is `(mo_occ > 1e-8).sum(axis=1)`, and
pbc_casida's layout is the rectangle nocc[ki] x nvirt[ki]. For a metal that is
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
carries the spin factor, so no separate factor of two appears below. Both
builders reduce EXACTLY to their integer-occupation counterparts in
pbc_casida when the occupations are 0 or 2 -- that equality is the main test.
"""
import warnings

import numpy as np

from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.Periodic.pbc_amplitudes import get_chi_a
from src.Base.numerics.constants import PBC_QP_Z_MIN
from src.Solvers.qp_equation import SampledEquation, find_qp_root
from src.SingleReference.Periodic.pbc_self_energy import (qp_energy_linearized,
                                                          spectral_sigma)

#: Pairs whose occupation difference falls below this carry no useful weight
#: and would contribute a zero row to A - B (0/0 for a degenerate pair), so
#: they are dropped. Measured cost on a Li monolayer: 0.0008% of the weight.
DEFAULT_OCCUPATION_THRESHOLD = 1e-4

#: Occupations are treated as integer, and the cheaper pbc_casida path is
#: exactly equivalent, when every mo_occ is within this of 0 or 2.
INTEGER_OCCUPATION_TOL = 1e-8


def fermi_level(mo_energy, sigma, nelectron, nkpts=None, method='fermi'):
    """The chemical potential that puts `nelectron` in the cell, by root-find.

        (1/N_k) sum_k sum_m f((eps_m^k - mu)/sigma) = nelectron

    THE ONLY DEFENSIBLE mu FOR A METAL, and worth having because the two
    obvious shortcuts are both wrong, measured on a Li monolayer against this:

      * `mf.get_fermi()` is off by **0.1 to 1.5 eV** at every point tested. It
        is not the smearing chemical potential and must not be used as one.
      * `chemical_potential_occ`'s heuristic -- the mean of the partially
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
                     threshold=DEFAULT_OCCUPATION_THRESHOLD):
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
                      threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """Per-k transition lists and flat offsets for one momentum transfer.

    The occupation-weighted analogue of pbc_casida.transition_layout. The
    space is ragged -- a different number of transitions survives at each k --
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
               threshold=DEFAULT_OCCUPATION_THRESHOLD, sqrt_weight=True):
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


def build_chi0_aux_occ(dfints, kconserv, omega, mo_energy=None, mo_occ=None,
                       imaginary=True, eta=0.0,
                       threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """Occupation-weighted Pi0^q(omega) in the auxiliary basis.

        Pi0 = (2/nk) sum_t (f_m - f_n) * g(omega, -gap_t) * L_t L_t^*

    summed over ORDERED pairs with f_m > f_n, which is half the pairs; the
    reversed pair contributes identically, hence the 2. With integer
    occupations f_m - f_n = 2 and this is exactly pbc_casida.build_chi0_aux's
    (4/nk) sum over occ-virt, which the tests pin to machine precision.

    g is d/(omega^2 + d^2) on the imaginary axis with d = e_m - e_n = -gap
    (negative, as there), or the real-axis form otherwise.
    """
    M, pairs, offsets, N, weight, gap = stacked_Lw(
        dfints, kconserv, mo_energy, mo_occ, threshold, sqrt_weight=True)
    if N == 0:
        naux = dfints.Lblock(0, kconserv[0]).shape[0]
        return np.zeros((naux, naux), dtype=np.complex128)
    d = -gap                                     # e_m - e_n < 0
    if imaginary:
        g = d / (omega ** 2 + d * d)
    else:
        g = 1.0 / (omega - d + 1j * eta) - 1.0 / (omega + d + 1j * eta)
    # sqrt(w) sits in M on both sides, so M diag(g) M^H already carries w.
    return (2.0 / dfints.nkpts) * (M * g[None]) @ M.conj().T


def build_rpa_matrices_occ(dfints, kconserv, mo_energy=None, mo_occ=None,
                           threshold=DEFAULT_OCCUPATION_THRESHOLD,
                           check_monotonic=True):
    """Occupation-weighted Kresse RPA Casida blocks A(q), B(q).

        A = diag(gap_t) + (1/nk) Mtilde^H Mtilde ,   B = (1/nk) Mtilde^H Mtilde

    with Mtilde carrying sqrt(f_m - f_n). A - B = diag(e_n - e_m) stays
    diagonal and, by the monotonicity checked above, positive -- so the
    CasidaSolver squaring trick applies unchanged, which is the whole reason
    the ordered-pair formulation was chosen over a non-Hermitian one.

    Integer occupations give f_m - f_n = 2, hence Mtilde^H Mtilde = 2 V_dir and
    A = diag + (2/nk) V_dir: pbc_casida.build_rpa_matrices exactly.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    if mo_occ is None:
        mo_occ = dfints.mo_occ
    if check_monotonic:
        check_occupation_monotonic(mo_energy, mo_occ)
    M, pairs, offsets, N, weight, gap = stacked_Lw(
        dfints, kconserv, mo_energy, mo_occ, threshold, sqrt_weight=True)
    V = M.conj().T @ M
    A = np.diag(gap).astype(np.complex128) + V / dfints.nkpts
    B = V / dfints.nkpts
    return A, B


def solve_rpa_spectral_occ(dfints, kconserv, mo_energy=None, mo_occ=None,
                           threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """(Omega, rho) at one transfer, occupation-weighted.

    rho[P, S] = sum_t Mtilde[P, t] (X + Y)_S[t]. Because Mtilde carries
    sqrt(w), the spectral W built from these amplitudes takes the constant
    2/nk rather than build_W_aux_spectral's 4/nk -- with integer occupations
    Mtilde = sqrt(2) M, so rho is larger by sqrt(2) and rho rho^H by 2. Use
    `build_W_aux_spectral_occ`, which carries that constant.
    """
    M, pairs, offsets, N, weight, gap = stacked_Lw(
        dfints, kconserv, mo_energy, mo_occ, threshold, sqrt_weight=True)
    A, B = build_rpa_matrices_occ(dfints, kconserv, mo_energy, mo_occ,
                                  threshold=threshold)
    Omega, X, Y = CasidaSolver(A, B).solve()
    return Omega, M @ (X + Y)


def build_W_aux_spectral_occ(dfints, Omega, rho, omega=0.0, imaginary=True,
                             eta=0.0):
    """Screened interaction from occupation-weighted eigenpairs.

    Identical to pbc_casida.build_W_aux_spectral except for the constant, which
    is 2/nk here because sqrt(f_m - f_n) is folded into rho -- see
    `solve_rpa_spectral_occ`. Validated the same way, against the direct
    (1 - Pi0)^{-1} inversion.
    """
    naux = rho.shape[0]
    c = 2.0 / dfints.nkpts
    if imaginary:
        denom = Omega / (omega ** 2 + Omega ** 2)
    else:
        denom = 0.5 * (1.0 / (Omega - omega - 1j * eta)
                       + 1.0 / (Omega + omega + 1j * eta))
    return np.eye(naux, dtype=np.complex128) - c * (rho * denom[None]) @ rho.conj().T


def solve_rpa_all_q_occ(dfints, mo_energy=None, mo_occ=None,
                        threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """(Omega, X, Y) at every momentum transfer, occupation-weighted.

    The counterpart of pbc_self_energy.solve_rpa_all_q. The eigenvectors live
    on the ragged occupation-weighted transition space, so they belong with
    `stacked_Lw`'s Mtilde and with `sigma_c_diag_occ`, not with the integer-
    occupation amplitudes.
    """
    out = []
    for q in range(dfints.nkpts):
        A, B = build_rpa_matrices_occ(dfints, dfints.kconserv_pair[q],
                                      mo_energy=mo_energy, mo_occ=mo_occ,
                                      threshold=threshold)
        out.append(CasidaSolver(A, B).solve())
    return out


def sigma_c_diag_occ(dfints, eig_all_q, kn, n, freq, mo_energy=None,
                     mo_occ=None, eta=1e-3,
                     threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """Occupation-weighted diagonal correlation self-energy Sigma_c^{n,kn}(freq).

    pbc_self_energy.sigma_c_diag splits the internal orbital m by a BINARY
    label, s_m = +1 if m is occupied at km and -1 otherwise, which is the same
    integer-occupation assumption the response carried. With smearing the
    spectral weight splits continuously instead: an orbital of occupation f_m
    contributes to the hole branch with weight f_m/2 and to the particle branch
    with 1 - f_m/2,

        Sigma_c = sum_q w_q/nk sum_S sum_m |chi_a|^2
                  [ (f_m/2) D(w - e_m + Omega_S)
                    + (1 - f_m/2) D(w - e_m - Omega_S) ] ,

    with D(x) = x/(x^2 + eta^2). At f_m = 2 the first term survives alone and
    at f_m = 0 the second does, reproducing s_m = +1 and -1 exactly.

    The prefactor is 1/nk rather than sigma_c_diag's 2/nk because the
    amplitudes here are built from Mtilde, which carries sqrt(f_m - f_n): with
    integer occupations that makes |chi_a|^2 larger by exactly two, and the two
    factors cancel. Reduction to sigma_c_diag is pinned in the tests.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    if mo_occ is None:
        mo_occ = dfints.mo_occ
    mo_energy = np.asarray(mo_energy)
    mo_occ = np.asarray(mo_occ, dtype=float)

    def amp2(q, X, Y):
        M = stacked_Lw(dfints, dfints.kconserv_pair[q], mo_energy, mo_occ,
                       threshold, sqrt_weight=True)[0]
        chi_a = get_chi_a(dfints, q, M @ (X + Y), kn)  # (nstates, nmo, nmo)
        return np.abs(chi_a[:, n, :]) ** 2             # (nstates, nmo)

    # weight f_m/2 on the +Omega (hole) branch
    return spectral_sigma(dfints, eig_all_q, kn, freq, amp2,
                          lambda km: mo_occ[km] / 2.0, scale=1.0,
                          mo_energy=mo_energy, eta=eta)


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


def transition_window(mo_energy, mo_occ, threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """(e_min, e_max) over the OCCUPATION-WEIGHTED transition set, all k-pairs.

    The window an imaginary-frequency grid should be scaled by. Every grid in
    `Base/utils/grids.py` is keyed on e_min, and without `mo_occ` that comes
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


def qp_energy_g0w0_occ(dfints, eig_all_q, kn, n, mo_energy=None, mo_occ=None,
                       eta=1e-3, de=1e-3, exchange_minus_vxc=0.0,
                       threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """Linearized G0W0 QP energy on the occupation-weighted self-energy.

        QP = eps + Z (Sigma_c(eps) + (Sigma_x - v_xc)_nn),
        Z  = 1 / (1 - dSigma_c/domega|eps),

    `pbc_self_energy.qp_energy_linearized`, the linearization
    `qp_energy_g0w0` also uses, on `sigma_c_diag_occ`. It is a separate entry
    point rather than a flag because the integer one reaches `sigma_c_diag`,
    which splits the internal orbital by a BINARY occupied/virtual label, so
    pointing a metal at it yields a number rather than an error.

    `exchange_minus_vxc` must be supplied for a DFT starting point exactly as
    in the integer routine. The derivative is a forward difference at `de`,
    which is what makes Z sensitive to `eta`: a broadening comparable to `de`
    smooths the very slope being measured, so both are the caller's to
    converge together, not independently.

    Returns (QP, Z), a PAIR -- unlike the integer routine, which returns the
    energy alone. Z is not a diagnostic afterthought here: see below. Z
    outside (0, 1] is returned with a RuntimeWarning.

    For a metal, note what the linearization assumes rather than what it
    computes: a single well-separated pole near eps. That is a statement about
    the spectral function, not about occupations, and it degrades where the
    quasiparticle is short-lived -- which is more common at a metallic Fermi
    level than in the gapped case the integer routine was written for. A root
    solve on the full Sigma_c(omega) is the check when Z drifts far from 1.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    kw = dict(mo_energy=mo_energy, mo_occ=mo_occ, eta=eta, threshold=threshold)
    return qp_energy_linearized(
        lambda w: sigma_c_diag_occ(dfints, eig_all_q, kn, n, w, **kw),
        mo_energy[kn][n].real, de, exchange_minus_vxc)


def qp_energy_root_occ(dfints, eig_all_q, kn, n, mo_energy=None, mo_occ=None,
                       eta=1e-3, exchange_minus_vxc=0.0, half_width=0.5,
                       npts=81, z_min=PBC_QP_Z_MIN,
                       threshold=DEFAULT_OCCUPATION_THRESHOLD):
    """Solve the QP equation w = eps + Sigma_c(w) + dVxc by the one root finder
    (`Solvers.qp_equation.find_qp_root`): a scan of eps +/- half_width on
    `npts` points, every sign change bisected and weighed.

    `qp_energy_g0w0_occ` LINEARIZES about eps, which presumes one
    well-separated pole there. On a metal that presumption fails visibly --
    measured Z = -0.18 for a virtual band of bulk Al at 2x2x2, meaning
    dSigma_c/domega > 1 and the linearization has inverted. No amount of
    k-refinement repairs that, because it is a property of the spectral
    function; the fix is to stop linearizing.

    Returns a dict rather than a scalar, because on a metal the useful output
    is not only where the root is but HOW MANY there are. The keys are the
    same on every path:

        qp            `find_qp_root`'s root: the largest weight Z in (0, 1]
                      (the rule of every quasiparticle equation in the code
                      base), the root nearest eps if no Z is in (0, 1], NaN if
                      the window holds no root
        Z             1/(1 - dSigma_c/domega) evaluated AT that root, not at eps
        quasiparticle False when Z <= z_min (or Z outside (0, 1]): the root is
                      kept and flagged, with a RuntimeWarning
        bracketed     False if the window caught no sign change at all
        nroots        roots with |Z| >= z_min
        n_small_z     roots with |Z| < z_min (pole artifacts, Z ~ 0)
        roots         the roots with |Z| >= z_min, ascending
        all_roots     every (root, Z) found, ascending

    `nroots > 1` means the spectral weight is split and there is no single
    quasiparticle to converge -- report it rather than silently taking one
    root and calling it converged. `bracketed=False` means the root is
    outside eps +/- half_width, which is itself information: widen the window
    deliberately rather than trusting an extrapolation.

    A root sitting ON a pole of Sigma_c has dSigma/domega -> -inf and so
    Z -> 0: it solves the equation and carries no spectral weight -- measured
    Z = -3.2e-06 for bulk Al at 6x6x6, on a case that reported a single
    bracketed root and looked clean in every other column. PBC_QP_Z_MIN is
    this scan's own artifact floor, not the molecular QP_WINDOW_Z_MIN, and it
    is what `quasiparticle` is judged by.

    The scan costs `npts` self-energy evaluations, but `sigma_c_diag_occ`
    accepts a frequency ARRAY, so it is one call, not npts of them.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    ep = mo_energy[kn][n].real
    kw = dict(mo_energy=mo_energy, mo_occ=mo_occ, eta=eta, threshold=threshold)

    def residual(w):
        sig = np.real(np.asarray(sigma_c_diag_occ(dfints, eig_all_q, kn, n, w,
                                                  **kw)))
        return w - ep - sig - exchange_minus_vxc

    # the one root finder on this window: every sign change bisected, a pole
    # of Sigma_c told from a root, every root weighed, the largest weight kept
    # and judged against z_min
    source = SampledEquation(residual, vectorized=True, n_omega=npts,
                             tol=1e-10,
                             windows=[(ep - half_width, ep + half_width)])
    got = find_qp_root(source, ep, z_min=z_min,
                       label=f'the QP equation of state {n} at k-point {kn}')
    found = sorted((float(r), float(z)) for r, z in got.found)
    keep = [r for r, z in found if abs(z) >= z_min]
    if found and not got.quasiparticle:
        warnings.warn(
            f"state {n} at k-point {kn}: the QP root kept, {got.w:.6f} Ha, has "
            f"Z = {got.z:.4g}, not in ({z_min}, 1]: no quasiparticle in the "
            f"window ({len(found)} root(s)). The root is returned and flagged "
            f"(quasiparticle=False); widen eta or refine the grid.",
            RuntimeWarning, stacklevel=2)
    return dict(qp=float(got.w), Z=float(got.z),
                quasiparticle=bool(got.quasiparticle), bracketed=bool(found),
                nroots=len(keep), n_small_z=len(found) - len(keep),
                roots=keep, all_roots=found)
