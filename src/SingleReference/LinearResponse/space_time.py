"""Space-time RPA screening: the polarizability in imaginary time, then one
cosine transform to the imaginary-frequency axis.

With a separable (ISDF) ERI,

    (ia|jb) = sum_PQ X_o[P,i] X_v[P,a] Z[P,Q] X_o[Q,j] X_v[Q,b]

the particle-hole bubble separates into an occupied and a virtual half that each
carry only ONE orbital index,

    Pi_PQ(i.tau) = G^o_PQ(tau) * G^v_PQ(tau)          (elementwise)
    G^o_PQ(tau)  = sum_i X_o[P,i] X_o[Q,i] e^{+eps_i tau}
    G^v_PQ(tau)  = sum_a X_v[P,a] X_v[Q,a] e^{-eps_a tau}

so a tau point costs O(M^2 (n_occ + n_vir)) -- two GEMMs and a Hadamard product
-- against O(naux^2 n_occ n_vir) per frequency for the direct summation in
`imaginary_frequency.py`. That is the N^3-against-N^4 step, crossing over near
120 basis functions.

The transform is used one way only, on the model space it is fitted for
(Pi(i.tau) is a sum of e^{-Delta_ia tau} with Delta_ia in [e_min, e_max]), so
the minimax transform's lack of matrix duality costs nothing.

Sign convention follows `imaginary_frequency._f_rpa`: f(i.w) = -2 d/(d^2 + w^2)
and chi0 = 2 (C_ov f) C_ov^T; the cosine transform maps e^{-d tau} to
2d/(d^2 + w^2), hence the leading -2 below.

References
----------
Rojas, Godby and Needs, Phys. Rev. Lett. 74, 1827 (1995) -- building the
response in imaginary time, where the occupied and virtual sums decouple.
Kaltak, Klimes and Kresse, J. Chem. Theory Comput. 10, 2498 (2014) -- the
minimax imaginary-time/frequency quadrature the transform below runs on.
Duchemin and Blase, J. Chem. Phys. 150, 174120 (2019) -- the separable RI
supplying X_o and X_v.
"""
import numpy as np

from src.Base.constants import ISDF_TILE_GB
from src.SingleReference.base import get_occ_virt_indices


def polarizability_imaginary_time(X_o, X_v, eps_o, eps_v, tau_points,
                                  out=None, beta=None):
    """Pi_PQ(i.tau) on the interpolation grid, shape (ntau, M, M).

    X_o, X_v :     (M, n_occ) and (M, n_vir) collocation, occupied and virtual.
    eps_o, eps_v : orbital energies SHIFTED so every eps_v - eps_o > 0; any
                   chemical potential inside the gap does this.
    beta :         inverse temperature, giving the bosonic periodic object
                   Pi(tau) + Pi(beta - tau) that a Matsubara/IR grid needs.
                   Omit for the T = 0 half-line function of the minimax grids.

    The mirror term is not a small correction: e^{i nu_n beta} = 1 for bosonic
    frequencies, so tau -> beta - tau maps the integral onto itself and the
    mirror contributes exactly as much as the direct term. Dropping it is a
    factor of two at every beta.
    """
    M = X_o.shape[0]
    ntau = len(tau_points)
    if out is None:
        out = np.empty((ntau, M, M))
    for k, tau in enumerate(tau_points):
        Go = (X_o * np.exp(eps_o * tau)) @ X_o.T
        Gv = (X_v * np.exp(-eps_v * tau)) @ X_v.T
        np.multiply(Go, Gv, out=out[k])
        if beta is not None:
            tb = beta - tau
            out[k] += (((X_o * np.exp(eps_o * tb)) @ X_o.T)
                       * ((X_v * np.exp(-eps_v * tb)) @ X_v.T))
    return out


def polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau,
                                 tile_memory_gb=ISDF_TILE_GB):
    """chi0 at ONE imaginary time, already projected to the auxiliary basis:

        proj_ab(tau) = -2 sum_PQ D[P,a] (Go_PQ Gv_PQ) D[Q,b]

    Tiled over grid rows, which is exact because the expression is a sum over
    them, so the M x M object never exists. Contract the Pi block with D first:
    the other order builds an (naux, M) intermediate instead.
    """
    M = X_o.shape[0]
    naux = D.shape[1]
    rows = max(1, min(M, int(tile_memory_gb * 1e9 / max(3 * M * 8, 1))))
    eo_t, ev_t = np.exp(e_o * tau), np.exp(-e_v * tau)
    proj = np.zeros((naux, naux))
    for p0 in range(0, M, rows):
        p1 = min(p0 + rows, M)
        Go = (X_o[p0:p1] * eo_t) @ X_o.T          # (b, M)
        Gv = (X_v[p0:p1] * ev_t) @ X_v.T          # (b, M)
        Go *= Gv                                   # Pi block, in Go's buffer
        proj += D[p0:p1].T @ (Go @ D)              # (naux, naux)
        del Go, Gv
    proj *= -2.0
    return proj


def chi0_imaginary_frequency(X, D, eps, nocc, grid, mu=None, stream=True,
                             tau_indices=None, tile_memory_gb=ISDF_TILE_GB):
    """chi0(i.omega) in the DF auxiliary basis, shape (nfreq, naux, naux).

    X :    (M, norb) collocation in the MO basis.
    D :    (M, naux) Coulomb factor, Z = D D^T; carries the result back to the
           auxiliary basis every DF consumer speaks.
    grid : TimeFrequencyGrid with an imaginary-time axis.

    Same chi0 as `solve_rpa_screening_df`, so W follows as [I - chi0]^-1.

    stream=True projects and accumulates each Pi(i.tau) immediately. Identical
    algebraically, since

        chi0(i.w) = -2 sum_tau cosft_wt[w,tau] (D^T Pi(tau) D),

    but the peak drops from (ntau, M, M) to one (M, M).
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    eps_o, eps_v = eps[occ], eps[virt]
    if mu is None:
        mu = 0.5 * (eps_o.max() + eps_v.min())
    X_o = np.ascontiguousarray(X[:, occ])
    X_v = np.ascontiguousarray(X[:, virt])
    e_o, e_v = eps_o - mu, eps_v - mu

    if not stream:
        Pi_tau = polarizability_imaginary_time(X_o, X_v, e_o, e_v,
                                               grid.tau_points)
        Pi_w = np.tensordot(grid.cosft_wt, Pi_tau, axes=(1, 0))
        return -2.0 * np.einsum('Pa,wPQ,Qb->wab', D, Pi_w, D, optimize=True)

    naux = D.shape[1]
    chi0 = np.zeros((grid.nfreq, naux, naux))
    which = range(grid.ntau) if tau_indices is None else np.atleast_1d(tau_indices)
    for k in which:
        proj = polarizability_projected_tau(X_o, X_v, e_o, e_v, D,
                                            grid.tau_points[k],
                                            tile_memory_gb=tile_memory_gb)
        chi0 += grid.cosft_wt[:, k, None, None] * proj
    return chi0


