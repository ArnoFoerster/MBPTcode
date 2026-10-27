"""Periodic GW self-energy assembly + QP solve -- the k-point generalization of
GW.self_energy.SelfEnergySolver's diagonal correlation self-energy.

Plain-GW correlation self-energy (spectral form), for target orbital n @ kn on
a gap:

    Sigma_c^{n,kn}(omega) = 2 sum_q w_q (1/nkpts) sum_S sum_{m}
        |chi_a^{S,q}[n @ kn, m @ km]|^2 * (omega - eps_{m,km} + s_m Omega_S^q)
                                        / ((omega - eps_{m,km} + s_m Omega_S^q)^2 + eta^2)

with km = kn - q, s_m = +1 for m occupied at km else -1, and the sum over S
running over the transfer-q RPA excitons (Omega_S^q, X, Y). This is the
periodic form of SelfEnergySolver.calculate_self_energy(vertex_mode='GW'):
the molecular prefactor 2 (restricted) times the BZ q-average and the response
normalization. Reduces to the molecular Sigma_c at nkpts=1. `sigma_c_diag`
builds it on the occupation-weighted transition space, where s_m becomes the
weights f_m/2 and 1 - f_m/2 and the 2 moves into the amplitudes.

THERE ARE TWO DISTINCT 1/nkpts FACTORS and they are easy to conflate:
  - w_q  is the BZ average over momentum transfers, 1/nkpts (via `q_grid`);
  - the explicit 1/nkpts is the normalization of the response/W that chi_a is
    built from. build_W_aux_spectral carries c = 2/nkpts for exactly this reason;
    since Sigma_c is assembled from |chi_a|^2 rather than from W directly, that
    factor has to be reinstated here.
With only the first, every finite-k Sigma is too large by exactly nkpts. That
is INVISIBLE to an nk=1 oracle (the factor is 1), which is what all the
molecular-reduction tests are -- tests/test_pbc_sigma_folding.py is the
supercell-folding check that catches it (ratio exactly 2.000000 at nk=2,
3.000000 at nk=3), and the missing factor is also the difference between
36.7 meV and 0.65 meV in the krgw_ac finite-q comparison.

The exciton set feeding chi_a is RPA (plain GW) or BSE-screened (vertex); this
module builds Sigma_c generically from whatever (Omega, X, Y) per q are passed.
"""
import warnings

import numpy as np
from src.Base.numerics.constants import (DEFAULT_BROADENING_ETA,
                                         PBC_OCCUPATION_THRESHOLD,
                                         PBC_QP_ROOT_TOL, PBC_QP_Z_MIN)
from src.Solvers.qp_equation import SampledEquation, find_qp_root
from src.SingleReference.Periodic.pbc_occupations import stacked_Lw
from src.SingleReference.Periodic.pbc_casida import build_rpa_matrices, build_bse_matrices
from src.SingleReference.Periodic.pbc_amplitudes import (
    project_XpY, get_chi_a, get_chi_b_vertex)
from src.SingleReference.LinearResponse.casida import CasidaSolver


def q_grid(dfints):
    """(nq, q_weights) for the momentum-transfer sum: the k-points, uniformly.

    The transfers ARE the k-points and each carries the BZ weight 1/nkpts, so
    the weighted sum reduces exactly to `1/nk sum_q`. Kept as a helper rather
    than inlined because every consumer of the q-sum should get its weights
    from one place.
    """
    return dfints.nkpts, np.full(dfints.nkpts, 1.0 / dfints.nkpts)


def solve_rpa_all_q(dfints, mo_energy=None, mo_occ=None,
                    threshold=PBC_OCCUPATION_THRESHOLD):
    """RPA eigenpairs (Omega, X, Y) for every momentum transfer q (plain GW).

    The eigenvectors live on the occupation-weighted transition space of
    `pbc_occupations.stacked_Lw`, so they belong with `sigma_c_diag`.
    """
    out = []
    for q in range(q_grid(dfints)[0]):
        A, B = build_rpa_matrices(dfints, dfints.kconserv_pair[q],
                                  mo_energy=mo_energy, mo_occ=mo_occ,
                                  threshold=threshold)
        out.append(CasidaSolver(A, B).solve())
    return out


def solve_bse_all_q(dfints, W_all, mo_energy=None):
    """BSE-screened Casida eigenpairs (Omega, X, Y) for every q (vertex path).
    These are the excitons feeding both chi_a and chi_b for GWGammaInf/PSD1.

    build_bse_matrices' B block pulls its antiresonant partner from the
    CONJUGATE k-points (Kresse Eq. 58, via kpts_helper.conj_mapping), which
    assumes the transfers are k-mesh points -- as they are.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    out = []
    for q in range(dfints.nkpts):
        A, B = build_bse_matrices(dfints, q, W_all, mo_energy=mo_energy)
        Omega, X, Y = CasidaSolver(A, B).solve()
        out.append((Omega, X, Y))
    return out


def spectral_sigma(dfints, eig_all_q, kn, freq, numerator, hole_weight,
                   scale=2.0, mo_energy=None, eta=DEFAULT_BROADENING_ETA):
    """The spectral sum every diagonal Sigma_c here is, for a target at kn:

        Sigma(w) = sum_q scale w_q/nk sum_{S,m} num_q[S, m]
                   [ h_m D(w - e_m + Omega_S) + (1 - h_m) D(w - e_m - Omega_S) ]

    with D(x) = x/(x^2 + eta^2), m running over the internal orbitals at
    km = kn - q, num_q = numerator(q, X, Y) of shape (nstates, nmo) and
    h = hole_weight(km) the weight of the hole (+Omega) branch: 1 or 0 by the
    integer occupation, f_m/2 with smearing. freq may be a scalar or 1D
    array; returns matching shape (real part).
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    nq, q_weights = q_grid(dfints)
    freq_grid = np.atleast_1d(freq).astype(float)
    sigma = np.zeros(len(freq_grid))

    for q in range(nq):
        Omega, X, Y = eig_all_q[q]
        num = numerator(q, X, Y)                       # (nstates, nmo), m@km
        km = dfints.kpoint_minus_q(q, kn)             # km = kn - q
        eps_m = mo_energy[km].real
        hole = hole_weight(km)
        part = 1.0 - hole
        for iw, w in enumerate(freq_grid):
            e_h = w - eps_m[None, :] + Omega[:, None].real
            e_p = w - eps_m[None, :] - Omega[:, None].real
            dh = e_h / (e_h ** 2 + eta ** 2)
            dp = e_p / (e_p ** 2 + eta ** 2)
            term = num * (hole[None, :] * dh + part[None, :] * dp)
            sigma[iw] += scale * q_weights[q] / dfints.nkpts * np.sum(term)

    return sigma if not np.isscalar(freq) else sigma[0]


def integer_hole_weight(dfints):
    """h_m = 1 for m occupied at km, else 0: `spectral_sigma`'s binary split."""
    return lambda km: (np.arange(dfints.nmo) < dfints.nocc[km]).astype(float)


def sigma_c_diag(dfints, eig_all_q, kn, n, freq, mo_energy=None, mo_occ=None,
                 eta=DEFAULT_BROADENING_ETA, threshold=PBC_OCCUPATION_THRESHOLD):
    """Diagonal correlation self-energy Sigma_c^{n,kn}(freq), spectral form.

    eig_all_q[q] = (Omega, X, Y) at transfer q from `solve_rpa_all_q`. An
    orbital of occupation f_m contributes to the hole branch with weight f_m/2
    and to the particle branch with 1 - f_m/2,

        Sigma_c = sum_q w_q/nk sum_S sum_m |chi_a|^2
                  [ (f_m/2) D(w - e_m + Omega_S)
                    + (1 - f_m/2) D(w - e_m - Omega_S) ] ,

    with D(x) = x/(x^2 + eta^2); at f_m = 2 the first term survives alone and
    at f_m = 0 the second does. The prefactor is 1/nk where `sigma_vertex_diag`
    has 2/nk because the amplitudes here are built from Mtilde, which carries
    sqrt(f_m - f_n) = sqrt(2) on a gap. freq may be a scalar or 1D array;
    returns matching shape (real part).
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


def sigma_vertex_diag(dfints, eig_all_q, W_all, kn, n, freq, vertex_mode='GWGammaInf',
                      mo_energy=None, eta=DEFAULT_BROADENING_ETA):
    """Diagonal vertex-corrected self-energy (GWGammaInf or PSD1) for (n, kn).
    Integer occupations only; no driver calls it yet, its tests do.

    Periodic generalization of SelfEnergySolver.calculate_self_energy's vertex
    branches, spectral form. The excitons eig_all_q must be the BSE-screened
    Casida states (solve_bse_all_q); chi_b is W-dressed with the same static
    W_all. Per exciton S and internal m @ km = kn - q:
        GWGammaInf:  (2/nk) Re[ conj(chi_a) * (chi_a - 0.5 chi_b) ] * pole
        PSD1:        (2/nk) * 0.25 |2 chi_a - chi_b|^2 * pole
    chi_a is indexed [target@kn, internal@km] (target first), chi_b is indexed
    [internal@km, target@kn] (target last) -- extracted consistently below.
    Reduces to the molecular vertex self-energy at nkpts=1.
    """
    if vertex_mode not in ('GWGammaInf', 'PSD1'):
        raise ValueError(f"unknown vertex_mode '{vertex_mode}'")

    def vertex_numerator(q, X, Y):
        rho = project_XpY(dfints, dfints.kconserv_pair[q], X, Y)
        chi_a = get_chi_a(dfints, q, rho, kn)                 # [S, target@kn, internal@km]
        chi_b = get_chi_b_vertex(dfints, q, W_all, X, Y, kn)  # [S, internal@km, target@kn]
        a = chi_a[:, n, :]                                    # (S, m) target n, internal m
        b = chi_b[:, :, n]                                    # (S, m) internal m, target n
        if vertex_mode == 'GWGammaInf':
            return np.real(np.conj(a) * (a - 0.5 * b))
        return 0.25 * np.abs(2.0 * a - b) ** 2

    return spectral_sigma(dfints, eig_all_q, kn, freq, vertex_numerator,
                          integer_hole_weight(dfints), mo_energy=mo_energy,
                          eta=eta)


def get_exchange_minus_vxc(mf, exxdiv=None):
    """Diagonal <p| Sigma_x - v_xc |p> per k-point, computed from PySCF.

    Sigma_x is the exact (Fock) exchange -- taken from PySCF's own get_veff/get_k
    (do NOT hand-roll it), and v_xc is the mean-field xc potential of `mf`
    (get_veff - get_j, which is the HF exchange for KRHF or the DFT xc potential
    for KRKS). This is the general G0W0 starting-point correction, the periodic
    krgw_ac `vk - v_mf` term: it is ~0 (up to the exxdiv convention) for a HF
    start but genuinely nonzero for a DFT (e.g. PBE) start, where Sigma_x differs
    from v_xc^PBE. Returns an (nkpts, nmo) real array.

    exxdiv controls the exact-exchange divergence treatment of Sigma_x (krgw_ac
    uses None); the mean-field v_xc keeps whatever convention `mf` was run with.
    """
    from pyscf.pbc import scf as pbcscf
    cell = mf.cell
    kpts = mf.kpts
    mo = np.asarray(mf.mo_coeff)
    dm = mf.make_rdm1()
    v_mf = np.asarray(mf.get_veff()) - np.asarray(mf.get_j(dm_kpts=dm))
    rhf = pbcscf.KRHF(cell, kpts, exxdiv=exxdiv)
    vk = np.asarray(rhf.get_veff(cell, dm_kpts=dm)) - np.asarray(rhf.get_j(cell, dm_kpts=dm))
    nk, nmo = mo.shape[0], mo.shape[2]
    out = np.zeros((nk, nmo))
    for k in range(nk):
        sx = mo[k].conj().T @ vk[k] @ mo[k]
        vxc = mo[k].conj().T @ v_mf[k] @ mo[k]
        out[k] = np.diag(sx - vxc).real
    return out


def qp_energy_g0w0(dfints, eig_all_q, kn, n, mo_energy=None, mo_occ=None,
                   eta=DEFAULT_BROADENING_ETA, exchange_minus_vxc=0.0, half_width=0.5,
                   npts=81, z_min=PBC_QP_Z_MIN,
                   threshold=PBC_OCCUPATION_THRESHOLD):
    """G0W0 quasiparticle energy of orbital (n, kn): the QP equation
    w = eps + Sigma_c(w) + (Sigma_x - v_xc)_nn solved by the one root finder
    (`Solvers.qp_equation.find_qp_root`) on `sigma_c_diag`: a scan of
    eps +/- half_width on `npts` points, every sign change bisected and weighed.

    `exchange_minus_vxc` is the scalar <n,kn| Sigma_x - v_xc |n,kn> (from
    get_exchange_minus_vxc[kn, n]); pass 0.0 only for a HF start in a convention
    where it vanishes. For a DFT start it MUST be supplied (Sigma_x != v_xc).

    The equation is not linearized about eps: that presumes one well-separated
    pole there, and on a metal it fails visibly -- measured Z = -0.18 for a
    virtual band of bulk Al at 2x2x2, meaning dSigma_c/domega > 1. No amount of
    k-refinement repairs that, because it is a property of the spectral
    function.

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

    The scan costs `npts` self-energy evaluations, but `sigma_c_diag`
    accepts a frequency ARRAY, so it is one call, not npts of them.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    ep = mo_energy[kn][n].real
    kw = dict(mo_energy=mo_energy, mo_occ=mo_occ, eta=eta, threshold=threshold)

    def residual(w):
        sig = np.real(np.asarray(sigma_c_diag(dfints, eig_all_q, kn, n, w,
                                                  **kw)))
        return w - ep - sig - exchange_minus_vxc

    # the one root finder on this window: every sign change bisected, a pole
    # of Sigma_c told from a root, every root weighed, the largest weight kept
    # and judged against z_min
    source = SampledEquation(residual, vectorized=True, n_omega=npts,
                             tol=PBC_QP_ROOT_TOL,
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
