"""Periodic GW self-energy assembly + QP solve -- the k-point generalization of
GW.self_energy.SelfEnergySolver's diagonal correlation self-energy.

Plain-GW correlation self-energy (spectral form), for target orbital n @ kn:

    Sigma_c^{n,kn}(omega) = 2 sum_q w_q (1/nkpts) sum_S sum_{m}
        |chi_a^{S,q}[n @ kn, m @ km]|^2 * (omega - eps_{m,km} + s_m Omega_S^q)
                                        / ((omega - eps_{m,km} + s_m Omega_S^q)^2 + eta^2)

with km = kn - q, s_m = +1 for m occupied at km else -1, and the sum over S
running over the transfer-q RPA excitons (Omega_S^q, X, Y). This is the
periodic form of SelfEnergySolver.calculate_self_energy(vertex_mode='GW'):
the molecular prefactor 2 (restricted) times the BZ q-average and the response
normalization. Reduces to the molecular Sigma_c at nkpts=1.

THERE ARE TWO DISTINCT 1/nkpts FACTORS and they are easy to conflate:
  - w_q  is the BZ average over momentum transfers, 1/nkpts (via `q_grid`);
  - the explicit 1/nkpts is the normalization of the response/W that chi_a is
    built from. build_W_aux_spectral carries c = 4/nkpts for exactly this reason;
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
import numpy as np
from src.SingleReference.Periodic.pbc_casida import build_rpa_matrices, build_bse_matrices
from src.SingleReference.Periodic.pbc_amplitudes import (
    project_XpY, get_chi_a, get_chi_b_vertex, kpoint_minus_q)
from src.SingleReference.LinearResponse.casida import CasidaSolver


def q_grid(dfints):
    """(nq, q_weights) for the momentum-transfer sum: the k-points, uniformly.

    The transfers ARE the k-points and each carries the BZ weight 1/nkpts, so
    the weighted sum reduces exactly to `1/nk sum_q`. Kept as a helper rather
    than inlined because every consumer of the q-sum should get its weights
    from one place.
    """
    return dfints.nkpts, np.full(dfints.nkpts, 1.0 / dfints.nkpts)


def solve_rpa_all_q(dfints, mo_energy=None):
    """RPA eigenpairs (Omega, X, Y) for every momentum transfer q (plain GW)."""
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    nq, _ = q_grid(dfints)
    out = []
    for q in range(nq):
        A, B = build_rpa_matrices(dfints, dfints.kconserv_pair[q], mo_energy=mo_energy)
        Omega, X, Y = CasidaSolver(A, B).solve()
        out.append((Omega, X, Y))
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


def sigma_c_diag(dfints, eig_all_q, kn, n, freq, mo_energy=None, eta=1e-3):
    """Diagonal correlation self-energy Sigma_c^{n,kn}(freq), spectral form.

    eig_all_q[q] = (Omega, X, Y) at transfer q (from solve_rpa_all_q for plain
    GW, or the BSE eigenpairs for the vertex path). freq may be a scalar or 1D
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
        rho = project_XpY(dfints, dfints.kconserv_pair[q], X, Y)
        chi_a = get_chi_a(dfints, q, rho, kn)          # (nstates, nmo, nmo) [p@kn, m@km]
        km = kpoint_minus_q(dfints, q, kn)             # km = kn - q
        amp2 = np.abs(chi_a[:, n, :]) ** 2             # (nstates, nmo), internal m@km
        nocc_km = dfints.nocc[km]
        s_m = np.where(np.arange(dfints.nmo) < nocc_km, 1.0, -1.0)
        eps_m = mo_energy[km].real
        for iw, w in enumerate(freq_grid):
            energy = w - eps_m[None, :] + s_m[None, :] * Omega[:, None].real
            denom = energy / (energy**2 + eta**2)
            sigma[iw] += 2.0 * q_weights[q] / dfints.nkpts * np.sum(amp2 * denom)

    return sigma if not np.isscalar(freq) else sigma[0]


def sigma_vertex_diag(dfints, eig_all_q, W_all, kn, n, freq, vertex_mode='GWGammaInf',
                      mo_energy=None, eta=1e-3):
    """Diagonal vertex-corrected self-energy (GWGammaInf or PSD1) for (n, kn).

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
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    nq, q_weights = q_grid(dfints)
    freq_grid = np.atleast_1d(freq).astype(float)
    sigma = np.zeros(len(freq_grid))

    for q in range(nq):
        Omega, X, Y = eig_all_q[q]
        rho = project_XpY(dfints, dfints.kconserv_pair[q], X, Y)
        chi_a = get_chi_a(dfints, q, rho, kn)                 # [S, target@kn, internal@km]
        chi_b = get_chi_b_vertex(dfints, q, W_all, X, Y, kn)  # [S, internal@km, target@kn]
        km = kpoint_minus_q(dfints, q, kn)
        a = chi_a[:, n, :]                                    # (S, m) target n, internal m
        b = chi_b[:, :, n]                                    # (S, m) internal m, target n

        if vertex_mode == 'GWGammaInf':
            num = np.real(np.conj(a) * (a - 0.5 * b))
        elif vertex_mode == 'PSD1':
            num = 0.25 * np.abs(2.0 * a - b) ** 2
        else:
            raise ValueError(f"unknown vertex_mode '{vertex_mode}'")

        nocc_km = dfints.nocc[km]
        s_m = np.where(np.arange(dfints.nmo) < nocc_km, 1.0, -1.0)
        eps_m = mo_energy[km].real
        for iw, w in enumerate(freq_grid):
            energy = w - eps_m[None, :] + s_m[None, :] * Omega[:, None].real
            denom = energy / (energy**2 + eta**2)
            sigma[iw] += 2.0 * q_weights[q] / dfints.nkpts * np.sum(num * denom)

    return sigma if not np.isscalar(freq) else sigma[0]


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


def qp_energy_g0w0(dfints, eig_all_q, kn, n, mo_energy=None, eta=1e-3, de=1e-3,
                   exchange_minus_vxc=0.0):
    """Linearized G0W0 QP energy for orbital (n, kn):
        QP = eps + Z * (Sigma_c(eps) + (Sigma_x - v_xc)_nn),
        Z = 1/(1 - dSigma_c/domega|eps).

    `exchange_minus_vxc` is the scalar <n,kn| Sigma_x - v_xc |n,kn> (from
    get_exchange_minus_vxc[kn, n]); pass 0.0 only for a HF start in a convention
    where it vanishes. For a DFT start it MUST be supplied (Sigma_x != v_xc).
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    ep = mo_energy[kn][n].real
    s0 = sigma_c_diag(dfints, eig_all_q, kn, n, ep, mo_energy=mo_energy, eta=eta)
    s1 = sigma_c_diag(dfints, eig_all_q, kn, n, ep + de, mo_energy=mo_energy, eta=eta)
    dsigma = (s1 - s0) / de
    Z = 1.0 / (1.0 - dsigma)
    return ep + Z * (s0 + exchange_minus_vxc)
